"""Run the frozen early-recipient reverse transfer at native layer zero."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path

import torch
import transformers
from transformers import DynamicCache
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.recurrence_dynamics import recurrence_native_trajectory as trajectory
from src.artifacts.utf8_json import literal_binding
from probes.recurrence_dynamics.recurrence_donor_tracking import require, write
from probes.recurrence_dynamics.recurrence_first_layer_readout import cpu_patch_selfcheck, install_o_proj_patch_observer
from probes.model_profiles.mature_tied_untied import load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-22-recurrence-first-layer-reverse-transfer"
OUT = ROOT / "attempt-001"
UNIT = Path("research/experiments/2026-09-22-recurrence-first-layer-reverse-transfer/unit.md")
SELECTION = ROOT / "selection.json"
ORACLE = trajectory.ORACLE
GROUPS_ROOT = BASE / "2026-09-22-recurrence-first-layer-groups"
GROUPS_ACCEPTANCE = GROUPS_ROOT / "lead-acceptance.json"
GROUPS_READBACK = GROUPS_ROOT / "attempt-001/readback.json"
NATIVE_CAPTURE = GROUPS_ROOT / "attempt-001/native.pt"
READOUT_ACCEPTANCE = BASE / "2026-09-22-recurrence-first-layer-readout/lead-acceptance.json"

TARGET, FULL_WIDTH = 2, 2127
RAW_RECIPIENT, RAW_DONOR = 384, 807
RECIPIENT_POSITION, DONOR_POSITION = 1703, 2126
VOCAB = 152670
Q_HEADS, HEAD_DIM, HIDDEN, LAYERS = 16, 128, 2048, 28
P38, P999 = 151708, 152669
CELL_ORDER = ("native", "old_sham", "late")
MAX_MODEL_FORWARDS = 3
MAX_VISION_FORWARDS = 3
MAX_SECONDS = 10 * 60
MAX_TENSOR_BYTES = 32 << 20
ATOL = 2e-4


def binding(path: Path) -> dict:
    return literal_binding(path)


def assert_binding(path: Path, expected: dict) -> None:
    actual = binding(path)
    require(all(actual[key] == expected[key] for key in ("path", "sha256", "size_bytes")),
            f"source binding drift: {path}")


def load_sources() -> tuple[dict, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    selection = json.loads(SELECTION.read_text())
    require(selection["status"] == "root-frozen" and selection["target"] == TARGET and
            selection["full_width"] == FULL_WIDTH and selection["query_position"] == RECIPIENT_POSITION and
            selection["raw_action_offset"] == RAW_RECIPIENT and
            selection["cell_order"] == list(CELL_ORDER), "reverse-transfer selection changed")
    require(selection["source_query"]["physical_query_index"] == RECIPIENT_POSITION and
            selection["donor_query"]["physical_query_index"] == DONOR_POSITION,
            "recipient/donor physical source binding changed")
    groups_acceptance = json.loads(GROUPS_ACCEPTANCE.read_text())
    readout_acceptance = json.loads(READOUT_ACCEPTANCE.read_text())
    require(groups_acceptance["status"] == "lead-accepted" and
            readout_acceptance["status"] == "lead-accepted", "predecessor acceptance changed")
    for key, expected in selection["sources"].items():
        assert_binding(Path(expected["path"]), expected)
    native_capture = torch.load(NATIVE_CAPTURE, map_location="cpu", weights_only=True)
    require(native_capture["full_logits"].shape == (2, VOCAB) and
            native_capture["head_output"].shape == (2, Q_HEADS, HEAD_DIM) and
            native_capture["query_positions"].tolist() == [RECIPIENT_POSITION, DONOR_POSITION],
            "predecessor native capture shape or query binding changed")
    early = native_capture["head_output"][0].float().reshape(-1).contiguous()
    late = native_capture["head_output"][1].float().reshape(-1).contiguous()
    require(tensor_hash(early) == selection["native_head_flat_hash"] and
            tensor_hash(early) == selection["replacement_fp32_flat_hash"]["old_sham"] and
            tensor_hash(late) == selection["replacement_fp32_flat_hash"]["late"],
            "predecessor replacement hashes changed")
    refs = {key: expected for key, expected in selection["sources"].items()}
    refs["selection"] = binding(SELECTION)
    refs["groups_readback"] = binding(GROUPS_READBACK)
    refs["producer"] = binding(Path(__file__))
    return selection, refs, native_capture["full_logits"][0].float().clone(), early, late


def source_crosswalk(selection: dict) -> dict:
    """Qualify both raw offsets against the original trace and physical inputs."""
    oracle = json.loads(ORACLE.read_text())
    require(oracle["status"] == "root-independent-source-oracle", "source oracle status changed")
    raw_path = Path(oracle["cases"]["val"]["source_bindings"]["raw"]["path"])
    trace_path = Path(oracle["cases"]["val"]["source_bindings"]["trace"]["path"])
    raw = json.loads(raw_path.read_text())["rows"]
    trace = json.loads(trace_path.read_text())
    tokens = raw[TARGET]["token_ids"]
    base = FULL_WIDTH - RAW_DONOR
    fake_ids = torch.zeros((4, FULL_WIDTH), dtype=torch.long)
    fake_ids[TARGET, base:base + RAW_DONOR] = torch.tensor(tokens[:RAW_DONOR])
    queries = trajectory.source_queries(
        "val", {"raw_action_offset": RAW_DONOR}, oracle, raw, trace, {"input_ids": fake_ids})
    by_offset = {item["raw_action_offset"]: item for item in queries}
    require(set(by_offset) >= {RAW_RECIPIENT, RAW_DONOR}, "recipient/donor trace entries missing")
    recipient, donor = by_offset[RAW_RECIPIENT], by_offset[RAW_DONOR]
    expected_recipient = selection["source_query"]
    expected_donor = selection["donor_query"]
    require(recipient["physical_query_index"] == RECIPIENT_POSITION and
            recipient["next_token"] == P38 and
            recipient["query_input_token"] == 152241 and
            recipient["minus_one_action_token"] == 152241 and
            donor["physical_query_index"] == DONOR_POSITION and
            donor["next_token"] == P999 and
            donor["query_input_token"] == 152241 and
            donor["minus_one_action_token"] == 152241,
            "raw action to physical query crosswalk changed")
    require(recipient["physical_query_index"] == expected_recipient["physical_query_index"] and
            recipient["next_token"] == expected_recipient["chosen_token"] and
            donor["physical_query_index"] == expected_donor["physical_query_index"] and
            donor["next_token"] == expected_donor["chosen_token"],
            "selection query records disagree with original trace")
    return {"recipient": recipient, "donor": donor, "base": base,
            "source_bindings": {"raw": binding(raw_path), "trace": binding(trace_path)}}


def cpu_selfcheck() -> None:
    selection, _refs, accepted_logits, early, late = load_sources()
    crosswalk = source_crosswalk(selection)
    require(accepted_logits.shape == (VOCAB,) and early.shape == (HIDDEN,) and late.shape == (HIDDEN,),
            "accepted reverse-transfer source vectors changed")
    cpu_patch_selfcheck()
    print(json.dumps({
        "status": "selfcheck_ok",
        "recipient": {"raw": RAW_RECIPIENT, "physical": RECIPIENT_POSITION,
                       "next": crosswalk["recipient"]["next_token"]},
        "donor": {"raw": RAW_DONOR, "physical": DONOR_POSITION,
                   "next": crosswalk["donor"]["next_token"]},
        "source_head_hashes": {"early": tensor_hash(early), "late": tensor_hash(late)},
        "transformers": transformers.__version__,
    }))


def summary(logits: torch.Tensor, label: str, native: torch.Tensor | None) -> dict:
    target = logits[TARGET]
    top = torch.topk(target, 5)
    probs = torch.softmax(target.double(), dim=0)
    out = {
        "cell": label,
        "target_top5": [{"token": int(token), "logit": float(value)}
                         for token, value in zip(top.indices, top.values, strict=True)],
        "target_winner": int(top.indices[0]),
        "target_top1_top2_gap": float(top.values[0] - top.values[1]),
        "target_p38": float(probs[P38]),
        "target_p999": float(probs[P999]),
        "target_d_z38_minus_z999": float(target[P38] - target[P999]),
    }
    if native is not None:
        out.update({
            "max_abs_from_native_all_batch": float((logits - native).abs().max()),
            "max_abs_from_native_target": float((target - native[TARGET]).abs().max()),
            "companion_max_abs_from_native": float((logits[[0, 1, 3]] - native[[0, 1, 3]]).abs().max()),
        })
    return out


def attention_identity(current: list[dict], reference: list[dict]) -> tuple[bool, list[dict]]:
    require(len(current) == len(reference) == LAYERS, "attention layer count changed")
    mismatches = []
    for i, (cur, ref) in enumerate(zip(current, reference, strict=True)):
        keys = ("mask_shape", "mask_hash", "cache_slots_hash", "phase_shapes")
        if any(cur.get(key) != ref.get(key) for key in keys):
            mismatches.append({"layer": i, "current": {key: cur.get(key) for key in keys},
                               "reference": {key: ref.get(key) for key in keys}})
    return not mismatches, mismatches


def run(device: str) -> None:
    cpu_selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA device required")
    require(not OUT.exists(), "attempt path already exists; preserve failed attempts")
    selection, refs, accepted_logits, early_head, late_head = load_sources()
    crosswalk = source_crosswalk(selection)
    OUT.mkdir(parents=True)
    state = {
        "schema": "recurrence_first_layer_reverse_transfer.receipt.v1",
        "status": "preparing", "pid": os.getpid(), "device": device,
        "model_forwards": 0, "vision_forwards": 0, "cell_order": list(CELL_ORDER),
        "completed_cells": [], "calls": [], "started_unix": time.time(),
        "started": time.monotonic(), "source": refs,
    }
    write(OUT / "receipt.json", state)
    cell_readbacks: dict[str, dict] = {}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        oracle = json.loads(ORACLE.read_text())
        case, _meta, queries, _raw, _trace = trajectory.source_inputs("val", q, identity, oracle, device)
        native = case["native"]
        require(case["target"] == TARGET and native["input_ids"].shape == (4, FULL_WIDTH),
                "native val input shape changed")
        selected = [item for item in queries if item["physical_query_index"] == RECIPIENT_POSITION]
        require(len(selected) == 1 and selected[0]["raw_action_offset"] == RAW_RECIPIENT,
                "native recipient query selection changed")
        native_hashes = {key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")}
        require(native_hashes == selection["native_input_hashes"], "native input hashes changed")
        previous_attention = json.loads(GROUPS_READBACK.read_text())["attention"]

        producer_capture = preserve_source(Path(__file__), run_root=OUT,
                                            relative_name="recurrence_first_layer_reverse_transfer.py")
        unit_capture = preserve_source(UNIT, run_root=OUT, relative_name="experiment/unit.md")
        dependency_paths = [
            Path('probes/recurrence_dynamics/recurrence_native_trajectory.py'),
            Path('probes/recurrence_dynamics/recurrence_first_layer_readout.py'),
            Path('probes/recurrence_dynamics/recurrence_first_layer_groups.py'),
            Path('probes/recurrence_dynamics/recurrence_attention_mass.py'),
            Path('probes/recurrence_dynamics/recurrence_fixed_template.py'),
            Path('probes/recurrence_dynamics/recurrence_position_history.py'),
            Path("probes/model_profiles/mature_tied_untied.py"),
            Path('probes/recurrence_dynamics/numerical_feedback/runtime.py'),
            Path("src/qwen/native.py"), Path("src/qwen/input_identity.py"),
            Path("src/inference/bound_requests.py"), Path("src/qwen/untied_embeddings.py"),
            Path(inspect.getfile(modeling_qwen3_vl)),
        ]
        dependency_captures = [
            preserve_source(path, run_root=OUT,
                            relative_name=str(path) if not path.is_absolute() else f"transformers/{path.name}")
            for path in dependency_paths
        ]
        source_manifest = {
            "schema": "recurrence_first_layer_reverse_transfer.v1",
            "status": "frozen_before_forward",
            "source": refs,
            "producer": binding(Path(__file__)),
            "producer_capture": binding(producer_capture),
            "unit_capture": binding(unit_capture),
            "dependency_captures": [binding(path) for path in dependency_captures],
            "transformers_version": transformers.__version__,
            "model_identity": identity,
            "case": {
                "name": "val", "target_batch": TARGET, "full_width": FULL_WIDTH,
                "recipient": {"row": 42, "raw_action_offset": RAW_RECIPIENT,
                              "physical_query_index": RECIPIENT_POSITION,
                              "query_metadata": selected[0]},
                "donor": {"row": 89, "raw_action_offset": RAW_DONOR,
                          "physical_query_index": DONOR_POSITION,
                          "query_metadata": crosswalk["donor"]},
                "native_input_hashes": native_hashes,
                "native_input_shapes": {key: list(native[key].shape)
                                         for key in ("input_ids", "attention_mask", "position_ids")},
                "source_bindings": case["source_bindings"],
                "accepted_paths": case["accepted_paths"],
                "fixed_artifacts": case["fixed_bindings"],
            },
            "donor_head_output": {
                "source": refs["native_capture"], "shape": [Q_HEADS, HEAD_DIM],
                "recipient_index": 0, "donor_index": 1,
                "old_sham_fp32_flat_hash": tensor_hash(early_head),
                "late_fp32_flat_hash": tensor_hash(late_head),
            },
            "collection": {
                "cell_order": list(CELL_ORDER), "fresh_empty_cache_per_cell": True,
                "one_full_native_forward_per_cell": True, "no_generation": True,
                "patch": "layer0 self_attn.o_proj input [target2,physical1703,:] only",
                "consumer_observer": "second o_proj pre-hook; exact replacement and off-target equality",
                "causal_visibility": "target2 query1703 sees arange(2127)<=1703; donor2126 is not visible",
                "model_forward_cap": MAX_MODEL_FORWARDS, "vision_forward_cap": MAX_VISION_FORWARDS,
                "wall_seconds_cap": MAX_SECONDS, "tensor_cap_bytes": MAX_TENSOR_BYTES,
                "parity_atol": ATOL,
            },
        }
        write(OUT / "source-to-cell.json", source_manifest)
        state.update(status="executing", manifest=binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)

        text = q.model.model.language_model
        query_indices = torch.tensor([RECIPIENT_POSITION], device=device, dtype=torch.long)
        native_checks = None
        native_logits = None
        for label in CELL_ORDER:
            state["current_cell"] = label
            write(OUT / "receipt.json", state)
            cache = DynamicCache()
            inputs = dict(native)
            inputs.update(logits_to_keep=query_indices, past_key_values=cache,
                          cache_position=torch.arange(FULL_WIDTH, device=device), use_cache=True)
            observer_state: dict = {}
            replacement = None if label == "native" else (early_head.reshape(Q_HEADS, HEAD_DIM)
                                                            if label == "old_sham" else
                                                            late_head.reshape(Q_HEADS, HEAD_DIM))
            expected_flat = early_head if label != "late" else late_head
            patch_handles = install_o_proj_patch_observer(
                text.layers[0].self_attn.o_proj, replacement, expected_flat=expected_flat,
                patch_batch=TARGET, patch_position=RECIPIENT_POSITION,
                observe_batch=TARGET, observe_position=RECIPIENT_POSITION,
                heads=Q_HEADS, head_dim=HEAD_DIM, state=observer_state)
            observers, observer_handles = trajectory.attention_attestors(
                text, q.model.lm_head, inputs, cache, TARGET, query_indices, FULL_WIDTH)
            handles = patch_handles + observer_handles

            def count_model(_module, _args, _kwargs):
                state["model_forwards"] += 1
                require(state["model_forwards"] <= MAX_MODEL_FORWARDS and
                        time.monotonic() - state["started"] <= MAX_SECONDS,
                        "model forward/time cap exceeded")

            def count_vision(*_args):
                state["vision_forwards"] += 1
                require(state["vision_forwards"] <= MAX_VISION_FORWARDS,
                        "vision forward cap exceeded")

            count_handles = [q.model.register_forward_pre_hook(count_model, with_kwargs=True),
                             q.model.model.visual.register_forward_pre_hook(count_vision)]
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    output = q.model(**inputs)
                elapsed = time.monotonic() - started
                logits = output.logits[:, 0].detach().float().cpu().clone()
                before_target = observer_state["before_target"]
                consumed_target = observer_state["consumed_target"]
            finally:
                for handle in handles + count_handles:
                    handle.remove()
            state["calls"].append({"cell": label, "seconds": elapsed})

            # Persist the expensive raw vectors before any numerical parity gate.
            require(logits.ndim == 2 and before_target.ndim == 1 and consumed_target.ndim == 1,
                    f"{label} raw capture dimensions unavailable")
            raw_payload = {
                "logits": logits, "before_o_proj_target": before_target,
                "consumed_o_proj_target": consumed_target,
                "replacement_o_proj_target": expected_flat.float().cpu().clone(),
                "query_position": torch.tensor([RECIPIENT_POSITION]), "target_batch": torch.tensor([TARGET]),
            }
            tensor_path = OUT / f"{label}.pt"
            torch.save(raw_payload, tensor_path)
            require(tensor_path.stat().st_size <= MAX_TENSOR_BYTES, f"{label} tensor payload cap exceeded")

            current_hashes = {key: tensor_hash(inputs[key]) for key in ("input_ids", "attention_mask", "position_ids")}
            attention_checks = observers["attention"]
            attention_match = False
            attention_mismatches = []
            if all(item is not None for item in attention_checks):
                reference_attention = previous_attention if label == "native" else native_checks["attention"]
                attention_match, attention_mismatches = attention_identity(attention_checks, reference_attention)
            before_error = float((before_target - early_head).abs().max())
            accepted_error = float((logits[TARGET] - accepted_logits).abs().max())
            record = {
                "cell": label, "tensor_artifact": binding(tensor_path),
                "logits_shape": list(logits.shape), "before_target_shape": list(before_target.shape),
                "consumed_target_shape": list(consumed_target.shape),
                "before_target_hash": tensor_hash(before_target),
                "consumed_target_hash": tensor_hash(consumed_target),
                "replacement_target_hash": tensor_hash(expected_flat.float().cpu()),
                "before_target_max_abs_vs_native_capture": before_error,
                "accepted_predecessor_target_max_abs": accepted_error,
                "consumer": {
                    "second_hook_seen": bool(observer_state.get("second_hook_seen", False)),
                    "tensor_identity_exact": observer_state.get("consumer_tensor_id") ==
                    observer_state.get("patch_tensor_id", observer_state.get("before_tensor_id")),
                    "off_target_exact": bool(observer_state.get("off_target_exact", False)),
                    "off_target_max_abs": observer_state.get("off_target_max_abs"),
                },
                "query_consumer": observers["lm_head"], "attention": attention_checks,
                "attention_identity_exact": attention_match, "attention_mismatches": attention_mismatches,
                "embedding_calls": observers["embedding"], "rotary_calls": observers["rotary"],
                "cache_length": cache.get_seq_length(),
                "summary": summary(logits, label, native_logits),
            }
            cell_readbacks[label] = record
            write(OUT / "readback.json", {
                "schema": "recurrence_first_layer_reverse_transfer.readback.v1",
                "status": "running", "source_manifest": binding(OUT / "source-to-cell.json"),
                "cells": cell_readbacks, "native_checks": native_checks,
            })

            # Mechanical shape/consumer checks now that the evidence is durable.
            require(output.past_key_values is cache and cache.get_seq_length() == FULL_WIDTH and
                    logits.shape == (4, VOCAB) and before_target.shape == (HIDDEN,) and
                    consumed_target.shape == (HIDDEN,) and observer_state.get("second_hook_seen") and
                    observer_state.get("off_target_exact"), f"{label} native consumer capture incomplete")
            require(observers["embedding"] == observers["rotary"] == 1 and
                    observers["lm_head"] == {"shape": [4, 1, HIDDEN],
                                              "physical_indices": [RECIPIENT_POSITION], "exact": True} and
                    all(item is not None for item in attention_checks),
                    f"{label} native helper attestations incomplete")
            require(current_hashes == native_hashes and attention_match,
                    f"{label} native input/mask/slot identity changed")
            require(before_error <= ATOL, f"{label} pre-patch o_proj input differs from native capture")
            if native_checks is None:
                native_checks = {"input_hashes": current_hashes, "attention": attention_checks,
                                 "query_consumer": observers["lm_head"]}
                native_logits = logits.clone()
                record["accepted_predecessor_winner"] = int(accepted_logits.argmax())
                record["native_winner"] = int(logits[TARGET].argmax())
                require(accepted_error <= ATOL and record["accepted_predecessor_winner"] == P38 and
                        record["native_winner"] == P38, "native qualification failed")
            else:
                require(observers["lm_head"] == native_checks["query_consumer"],
                        f"{label} LM head query identity changed")
                companion_error = float((logits[[0, 1, 3]] - native_logits[[0, 1, 3]]).abs().max())
                record["companion_max_abs_vs_native"] = companion_error
                require(companion_error <= ATOL, f"{label} companion batch parity failed")
                if label == "old_sham":
                    sham_error = float((logits - native_logits).abs().max())
                    record["sham_max_abs_vs_native"] = sham_error
                    require(sham_error <= ATOL and int(logits[TARGET].argmax()) == P38,
                            "old_sham qualification failed")
            state["completed_cells"].append(label)
            write(OUT / "receipt.json", state)
            del observer_state, output, cache, inputs

        require(state["model_forwards"] == MAX_MODEL_FORWARDS and
                state["vision_forwards"] == MAX_VISION_FORWARDS,
                "exactly three model and vision forwards required")
        final_readback = {
            "schema": "recurrence_first_layer_reverse_transfer.readback.v1",
            "status": "candidate", "source_manifest": binding(OUT / "source-to-cell.json"),
            "recipient": {"raw_action_offset": RAW_RECIPIENT, "physical_query_index": RECIPIENT_POSITION,
                          "expected_native_token": P38},
            "donor": {"raw_action_offset": RAW_DONOR, "physical_query_index": DONOR_POSITION,
                      "expected_next_token": P999},
            "cells": cell_readbacks, "native_checks": native_checks,
            "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
        }
        write(OUT / "readback.json", final_readback)
        result = {
            "status": "candidate", "cell_order": list(CELL_ORDER),
            "recipient": {"raw_action_offset": RAW_RECIPIENT, "physical_query_index": RECIPIENT_POSITION},
            "donor": {"raw_action_offset": RAW_DONOR, "physical_query_index": DONOR_POSITION},
            "target_winners": {label: cell_readbacks[label]["summary"]["target_winner"] for label in CELL_ORDER},
            "readback": binding(OUT / "readback.json"),
            "source_manifest": binding(OUT / "source-to-cell.json"),
            "artifacts": {label: cell_readbacks[label]["tensor_artifact"] for label in CELL_ORDER},
            "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
        }
        write(OUT / "result.json", result)
        state.update(status="candidate_complete", result=binding(OUT / "result.json"),
                     elapsed_seconds=time.monotonic() - state["started"],
                     peak_reserved_bytes=int(torch.cuda.max_memory_reserved(torch.device(device))))
        write(OUT / "receipt.json", state)
        print(json.dumps({"status": state["status"], "model_forwards": state["model_forwards"],
                          "vision_forwards": state["vision_forwards"], "result": str(OUT / "result.json")}))
    except BaseException as error:
        state.update(status="technical_invalid", error=repr(error),
                     elapsed_seconds=time.monotonic() - state["started"],
                     peak_reserved_bytes=int(torch.cuda.max_memory_reserved(torch.device(device)))
                     if torch.cuda.is_available() else None)
        if cell_readbacks:
            write(OUT / "readback.json", {
                "schema": "recurrence_first_layer_reverse_transfer.readback.v1",
                "status": "technical_invalid", "source_manifest": binding(OUT / "source-to-cell.json")
                if (OUT / "source-to-cell.json").exists() else None,
                "cells": cell_readbacks, "error": repr(error),
            })
        write(OUT / "receipt.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        cpu_selfcheck()
    else:
        run(args.device)
