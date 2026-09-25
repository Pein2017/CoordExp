"""Run the frozen downstream readout cells at the native exit query."""
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
from probes.model_profiles.mature_tied_untied import load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-22-recurrence-first-layer-readout"
OUT = ROOT / "attempt-001"
UNIT = Path("research/experiments/2026-09-22-recurrence-first-layer-readout/unit.md")
SELECTION = ROOT / "selection.json"
ORACLE = trajectory.ORACLE
PREDECESSOR_ROOT = BASE / "2026-09-22-recurrence-first-layer-groups"
PREDECESSOR_ACCEPTANCE = PREDECESSOR_ROOT / "lead-acceptance.json"
FACTORIAL = PREDECESSOR_ROOT / "lead-checks/factorial.pt"
NATIVE_CAPTURE = PREDECESSOR_ROOT / "attempt-001/native.pt"
TARGET, FULL_WIDTH, QUERY_POSITION = 2, 2127, 2126
RAW_ACTION_OFFSET = 807
VOCAB = 152670
Q_HEADS, HEAD_DIM, HIDDEN, LAYERS = 16, 128, 2048, 28
CELL_ORDER = ("native", "11", "01", "10", "00")
MAX_MODEL_FORWARDS = 5
MAX_VISION_FORWARDS = 5
MAX_SECONDS = 10 * 60
MAX_TENSOR_BYTES = 64 << 20
ATOL = 2e-4
P38, P999 = 151708, 152669


def binding(path: Path) -> dict:
    return literal_binding(path)


def assert_binding(path: Path, expected: dict) -> None:
    actual = binding(path)
    require(all(actual[key] == expected[key] for key in ("path", "sha256", "size_bytes")),
            f"source binding drift: {path}")


def load_sources() -> tuple[dict, dict, dict, torch.Tensor, torch.Tensor]:
    selection = json.loads(SELECTION.read_text())
    require(selection["status"] == "root-frozen" and selection["target"] == TARGET and
            selection["full_width"] == FULL_WIDTH and selection["query_position"] == QUERY_POSITION and
            selection["raw_action_offset"] == RAW_ACTION_OFFSET and
            selection["cell_order"] == list(CELL_ORDER), "readout selection changed")
    acceptance = json.loads(PREDECESSOR_ACCEPTANCE.read_text())
    require(acceptance["status"] == "lead-accepted", "predecessor first-layer acceptance changed")
    factorial_ref = selection["sources"]["factorial"]
    native_ref = selection["sources"]["native_capture"]
    assert_binding(FACTORIAL, factorial_ref)
    assert_binding(NATIVE_CAPTURE, native_ref)
    factorial = torch.load(FACTORIAL, map_location="cpu", weights_only=True)
    native_capture = torch.load(NATIVE_CAPTURE, map_location="cpu", weights_only=True)
    require(set(factorial["cells"]) == {"00", "01", "10", "11"} and
            all(factorial["cells"][name].shape == (Q_HEADS, HEAD_DIM) and
                factorial["cells"][name].dtype == torch.float64 for name in factorial["cells"]),
            "factorial cell shapes/dtypes changed")
    require(native_capture["full_logits"].shape == (2, VOCAB) and
            native_capture["head_output"].shape == (2, Q_HEADS, HEAD_DIM),
            "predecessor native capture shape changed")
    for name in ("00", "01", "10", "11"):
        flat = factorial["cells"][name].float().reshape(-1)
        require(tensor_hash(flat) == selection["replacement_fp32_flat_hash"][name],
                f"{name} independent replacement flatten hash changed")
    refs = {
        "unit": binding(UNIT),
        "selection": binding(SELECTION),
        "source_oracle": binding(ORACLE),
        "predecessor_acceptance": binding(PREDECESSOR_ACCEPTANCE),
        "factorial": factorial_ref,
        "native_capture": native_ref,
    }
    expected_native_logits = native_capture["full_logits"][1].float().clone()
    expected_native_head = native_capture["head_output"][1].float().reshape(-1).clone()
    return selection, refs, factorial, expected_native_logits, expected_native_head


def install_o_proj_patch_observer(
    module: torch.nn.Module,
    replacement_cell: torch.Tensor | None,
    *,
    expected_flat: torch.Tensor,
    patch_batch: int,
    patch_position: int,
    observe_batch: int,
    observe_position: int,
    heads: int,
    head_dim: int,
    state: dict,
) -> list:
    """Install the production patch and the immediately following consumer observer."""
    require(expected_flat.ndim == 1 and expected_flat.numel() == heads * head_dim,
            "expected o_proj vector shape changed")
    if replacement_cell is not None:
        require(replacement_cell.shape == (heads, head_dim), "replacement head/feature shape changed")
    handles = []

    def capture_before(_module, args):
        x = args[0]
        require(x.ndim == 3 and x.shape[-1] == heads * head_dim and
                0 <= patch_batch < x.shape[0] and 0 <= patch_position < x.shape[1],
                "o_proj input batch/sequence/hidden shape changed")
        state["before_full"] = x.detach().clone()
        state["before_target"] = x[observe_batch, observe_position].detach().float().cpu().clone()
        state["before_tensor_id"] = id(x)

    def patch(_module, args):
        x = args[0]
        if replacement_cell is None:
            return None
        replacement = replacement_cell.to(device=x.device, dtype=x.dtype).reshape(-1)
        patched = x.clone()
        patched[patch_batch, patch_position, :].copy_(replacement)
        state["replacement_target"] = replacement.detach().float().cpu().clone()
        state["patch_tensor_id"] = id(patched)
        return (patched, *args[1:])

    def consumer(_module, args):
        consumed = args[0]
        before = state.get("before_full")
        require(before is not None and consumed.shape == before.shape,
                "o_proj consumer did not receive the captured input shape")
        expected = expected_flat.to(device=consumed.device, dtype=consumed.dtype)
        if replacement_cell is None:
            require(torch.equal(consumed[observe_batch, observe_position], expected),
                    "native o_proj target drifted before consumption")
        else:
            require(torch.equal(consumed[observe_batch, observe_position], expected),
                    "o_proj consumer did not receive the selected replacement")
            require(state.get("patch_tensor_id") == id(consumed),
                    "second o_proj observer did not see the patch tensor")
        restored = consumed.detach().clone()
        restored[observe_batch, observe_position, :] = before[observe_batch, observe_position, :]
        require(torch.equal(restored, before), "off-target o_proj input changed")
        state.update(
            second_hook_seen=True,
            consumer_tensor_id=id(consumed),
            consumed_target=consumed[observe_batch, observe_position].detach().float().cpu().clone(),
            off_target_exact=True,
            off_target_max_abs=float((restored - before).abs().max()),
        )

    handles.append(module.register_forward_pre_hook(capture_before))
    if replacement_cell is not None:
        handles.append(module.register_forward_pre_hook(patch))
    handles.append(module.register_forward_pre_hook(consumer))
    return handles


def cpu_patch_selfcheck() -> None:
    batch, seq, heads, features, out_dim = 3, 5, 3, 5, 7
    hidden = heads * features
    x = torch.arange(batch * seq * hidden, dtype=torch.float32).reshape(batch, seq, hidden) + 0.25
    cell = torch.arange(heads * features, dtype=torch.float64).reshape(heads, features) + 100.5
    expected = cell.float().reshape(-1)
    linear = torch.nn.Linear(hidden, out_dim, bias=False)
    state = {}
    handles = install_o_proj_patch_observer(
        linear, cell, expected_flat=expected, patch_batch=2, patch_position=4,
        observe_batch=2, observe_position=4, heads=heads, head_dim=features, state=state)
    try:
        observed = linear(x)
    finally:
        for handle in handles:
            handle.remove()
    expected_x = x.clone()
    expected_x[2, 4] = expected
    require(torch.equal(observed, linear(expected_x)) and state["second_hook_seen"] and
            state["off_target_exact"], "real nn.Linear patch consumer self-check failed")

    wrong_query = {}
    handles = install_o_proj_patch_observer(
        linear, cell, expected_flat=expected, patch_batch=2, patch_position=3,
        observe_batch=2, observe_position=4, heads=heads, head_dim=features, state=wrong_query)
    try:
        try:
            linear(x)
        except ValueError:
            pass
        else:
            raise AssertionError("one-position query shift escaped the source-aligned observer")
    finally:
        for handle in handles:
            handle.remove()

    wrong_flat_cell = cell.transpose(0, 1).reshape(heads, features)
    wrong_flat = {}
    handles = install_o_proj_patch_observer(
        linear, wrong_flat_cell, expected_flat=expected, patch_batch=2, patch_position=4,
        observe_batch=2, observe_position=4, heads=heads, head_dim=features, state=wrong_flat)
    try:
        try:
            linear(x)
        except ValueError:
            pass
        else:
            raise AssertionError("head/feature flatten swap escaped the source-aligned observer")
    finally:
        for handle in handles:
            handle.remove()


def cpu_selfcheck() -> None:
    selection, refs, factorial, expected_native_logits, expected_native_head = load_sources()
    oracle = json.loads(ORACLE.read_text())
    require(oracle["status"] == "root-independent-source-oracle", "source oracle status changed")
    meta = json.loads(trajectory.SELECTION.read_text())["cases"]["val"]
    raw_path = Path(oracle["cases"]["val"]["source_bindings"]["raw"]["path"])
    trace_path = Path(oracle["cases"]["val"]["source_bindings"]["trace"]["path"])
    raw = json.loads(raw_path.read_text())["rows"]
    trace = json.loads(trace_path.read_text())
    tokens = raw[TARGET]["token_ids"]
    base = FULL_WIDTH - RAW_ACTION_OFFSET
    fake_ids = torch.zeros((4, FULL_WIDTH), dtype=torch.long)
    fake_ids[TARGET, base:base + RAW_ACTION_OFFSET] = torch.tensor(tokens[:RAW_ACTION_OFFSET])
    queries = trajectory.source_queries(
        "val", {"raw_action_offset": RAW_ACTION_OFFSET}, oracle, raw, trace, {"input_ids": fake_ids})
    selected = next(item for item in queries if item["raw_action_offset"] == RAW_ACTION_OFFSET)
    source_query = selection["source_query"]
    require(selected["physical_query_index"] == QUERY_POSITION and selected["next_token"] == source_query["chosen_token"] == 152669 and
            selected["minus_one_action_token"] == source_query["minus_one_action_token"] == 152241 and
            source_query["expected_query_input_token"] == 152241,
            "raw807 to physical2126 source binding changed")
    require(expected_native_logits.shape == (VOCAB,) and expected_native_head.shape == (HIDDEN,),
            "accepted native source vectors changed")
    cpu_patch_selfcheck()
    print(json.dumps({"status": "selfcheck_ok", "physical_query": QUERY_POSITION,
                      "raw_action_offset": RAW_ACTION_OFFSET, "cells": list(factorial["cells"]),
                      "transformers": transformers.__version__}))


def cell_summary(logits: torch.Tensor, label: str, native_logits: torch.Tensor | None) -> dict:
    target = logits[TARGET]
    top = torch.topk(target, 5)
    probs = torch.softmax(target.double(), dim=0)
    summary = {
        "cell": label,
        "target_top5": [{"token": int(token), "logit": float(value)}
                         for token, value in zip(top.indices, top.values, strict=True)],
        "target_winner": int(top.indices[0]),
        "target_top1_top2_gap": float(top.values[0] - top.values[1]),
        "target_p38": float(probs[P38]),
        "target_p999": float(probs[P999]),
        "target_d_z38_minus_z999": float(target[P38] - target[P999]),
    }
    if native_logits is not None:
        summary["max_abs_from_native_all_batch"] = float((logits - native_logits).abs().max())
        summary["max_abs_from_native_target"] = float((target - native_logits[TARGET]).abs().max())
        summary["companion_max_abs_from_native"] = float(
            (logits[[0, 1, 3]] - native_logits[[0, 1, 3]]).abs().max())
    return summary


def run(device: str) -> None:
    cpu_selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA device required")
    require(not OUT.exists(), "attempt path already exists; preserve failed attempts")
    selection, refs, factorial, accepted_native_logits, accepted_native_head = load_sources()
    OUT.mkdir(parents=True)
    state = {
        "status": "preparing", "pid": os.getpid(), "device": device,
        "model_forwards": 0, "vision_forwards": 0, "cell_order": list(CELL_ORDER),
        "completed_cells": [], "calls": [], "started_unix": time.time(), "started": time.monotonic(),
        "source": refs,
    }
    write(OUT / "receipt.json", state)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        oracle = json.loads(ORACLE.read_text())
        case, meta, all_queries, raw, trace = trajectory.source_inputs("val", q, identity, oracle, device)
        require(case["target"] == TARGET and case["native"]["input_ids"].shape == (4, FULL_WIDTH),
                "native val input shape changed")
        queries = [item for item in all_queries if item["physical_query_index"] == QUERY_POSITION]
        require(len(queries) == 1 and queries[0]["raw_action_offset"] == RAW_ACTION_OFFSET,
                "native query selection changed")
        native = case["native"]
        producer_capture = preserve_source(Path(__file__), run_root=OUT,
                                           relative_name="recurrence_first_layer_readout.py")
        unit_capture = preserve_source(UNIT, run_root=OUT, relative_name="experiment/unit.md")
        dependency_paths = [
            Path('probes/recurrence_dynamics/recurrence_native_trajectory.py'),
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
            "schema": "recurrence_first_layer_readout.v1",
            "status": "frozen_before_forward",
            "source": refs,
            "producer": binding(Path(__file__)),
            "producer_capture": binding(producer_capture),
            "unit_capture": binding(unit_capture),
            "dependency_captures": [binding(path) for path in dependency_captures],
            "transformers_version": transformers.__version__,
            "model_identity": identity,
            "case": {
                "target_batch": TARGET, "full_width": FULL_WIDTH, "query_position": QUERY_POSITION,
                "raw_action_offset": RAW_ACTION_OFFSET, "query_metadata": queries,
                "native_input_hashes": {key: tensor_hash(native[key])
                                         for key in ("input_ids", "attention_mask", "position_ids")},
                "native_input_shapes": {key: list(native[key].shape)
                                         for key in ("input_ids", "attention_mask", "position_ids")},
                "source_bindings": case["source_bindings"], "accepted_paths": case["accepted_paths"],
                "fixed_artifacts": case["fixed_bindings"],
            },
            "factorial": {"binding": refs["factorial"], "cell_shapes": {name: list(factorial["cells"][name].shape)
                                                                            for name in ("00", "01", "10", "11")},
                          "replacement_fp32_flat_hash": selection["replacement_fp32_flat_hash"]},
            "collection": {"cell_order": list(CELL_ORDER), "fresh_empty_cache_per_cell": True,
                           "one_full_native_forward_per_cell": True, "no_generation": True,
                           "patch": "layer0 self_attn.o_proj input [target2,physical2126,:] only",
                           "off_target_exact": True, "model_forward_cap": MAX_MODEL_FORWARDS,
                           "vision_forward_cap": MAX_VISION_FORWARDS, "wall_seconds_cap": MAX_SECONDS,
                           "tensor_cap_bytes": MAX_TENSOR_BYTES, "parity_atol": ATOL},
        }
        write(OUT / "source-to-cell.json", source_manifest)
        state.update(status="executing", manifest=binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)

        text = q.model.model.language_model
        query_indices = torch.tensor([QUERY_POSITION], device=device, dtype=torch.long)
        native_logits = None
        native_checks = None
        cell_readbacks = {}
        for label in CELL_ORDER:
            state["current_cell"] = label
            write(OUT / "receipt.json", state)
            cache = DynamicCache()
            inputs = dict(native)
            inputs.update(logits_to_keep=query_indices, past_key_values=cache,
                          cache_position=torch.arange(FULL_WIDTH, device=device), use_cache=True)
            observer_state = {}
            if label == "native":
                replacement = None
                expected_flat = accepted_native_head
            else:
                replacement = factorial["cells"][label]
                expected_flat = replacement.float().reshape(-1)
            patch_handles = install_o_proj_patch_observer(
                text.layers[0].self_attn.o_proj, replacement, expected_flat=expected_flat,
                patch_batch=TARGET, patch_position=QUERY_POSITION,
                observe_batch=TARGET, observe_position=QUERY_POSITION,
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
            require(output.past_key_values is cache and cache.get_seq_length() == FULL_WIDTH and
                    logits.shape == (4, VOCAB) and before_target.shape == (HIDDEN,) and
                    consumed_target.shape == (HIDDEN,) and observer_state["second_hook_seen"] and
                    observer_state["off_target_exact"], f"{label} native consumer capture incomplete")
            require(observers["embedding"] == observers["rotary"] == 1 and
                    observers["lm_head"] is not None and all(item is not None for item in observers["attention"]),
                    f"{label} native helper attestations incomplete")
            input_hashes = {key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")}
            attention_checks = observers["attention"]
            if native_checks is None:
                native_checks = {"input_hashes": input_hashes, "attention": attention_checks,
                                 "query_consumer": observers["lm_head"]}
                native_logits = logits.clone()
            else:
                require(input_hashes == native_checks["input_hashes"] and
                        attention_checks == native_checks["attention"] and
                        observers["lm_head"] == native_checks["query_consumer"],
                        f"{label} native input/mask/slot consumer identity changed")
            before_error = float((before_target - accepted_native_head).abs().max())
            require(before_error <= ATOL, f"{label} pre-patch o_proj input differs from native capture")
            payload = {
                "logits": logits,
                "before_o_proj_target": before_target,
                "consumed_o_proj_target": consumed_target,
                "replacement_o_proj_target": expected_flat.float().cpu().clone(),
                "query_position": torch.tensor([QUERY_POSITION]),
                "target_batch": torch.tensor([TARGET]),
            }
            tensor_path = OUT / f"{label}.pt"
            torch.save(payload, tensor_path)
            require(tensor_path.stat().st_size <= MAX_TENSOR_BYTES, f"{label} tensor payload cap exceeded")
            record = {
                "cell": label, "tensor_artifact": binding(tensor_path),
                "logits_shape": list(logits.shape), "before_target_shape": list(before_target.shape),
                "consumed_target_shape": list(consumed_target.shape),
                "before_target_hash": tensor_hash(before_target),
                "consumed_target_hash": tensor_hash(consumed_target),
                "replacement_target_hash": tensor_hash(expected_flat.float().cpu()),
                "before_target_max_abs_vs_native_capture": before_error,
                "consumer": {"second_hook_seen": observer_state["second_hook_seen"],
                              "tensor_identity_exact": observer_state["consumer_tensor_id"] ==
                              (observer_state.get("patch_tensor_id", observer_state["before_tensor_id"])),
                              "off_target_exact": observer_state["off_target_exact"],
                              "off_target_max_abs": observer_state["off_target_max_abs"]},
                "query_consumer": observers["lm_head"], "attention": attention_checks,
                "embedding_calls": observers["embedding"], "rotary_calls": observers["rotary"],
                "cache_length": cache.get_seq_length(),
                "summary": cell_summary(logits, label, native_logits if label != "native" else None),
            }
            if label == "native":
                record["accepted_predecessor_target_max_abs"] = float((logits[TARGET] - accepted_native_logits).abs().max())
                record["accepted_predecessor_winner"] = int(accepted_native_logits.argmax())
                record["native_winner"] = int(logits[TARGET].argmax())
                require(record["accepted_predecessor_target_max_abs"] <= ATOL and
                        record["native_winner"] == 152669 and record["accepted_predecessor_winner"] == 152669,
                        "native qualification failed")
            else:
                record["companion_max_abs_vs_native"] = float((logits[[0, 1, 3]] - native_logits[[0, 1, 3]]).abs().max())
                require(record["companion_max_abs_vs_native"] <= ATOL,
                        f"{label} companion batch parity failed")
                if label == "11":
                    record["sham_max_abs_vs_native"] = float((logits - native_logits).abs().max())
                    require(record["sham_max_abs_vs_native"] <= ATOL and int(logits[TARGET].argmax()) == 152669,
                            "sham11 qualification failed")
            cell_readbacks[label] = record
            write(OUT / "readback.json", {
                "schema": "recurrence_first_layer_readout.readback.v1",
                "status": "running", "source_manifest": binding(OUT / "source-to-cell.json"),
                "cells": cell_readbacks, "native_checks": native_checks,
            })
            state["completed_cells"].append(label)
            write(OUT / "receipt.json", state)
            del observer_state, output, cache, inputs
        require(state["model_forwards"] == MAX_MODEL_FORWARDS and state["vision_forwards"] == MAX_VISION_FORWARDS,
                "exactly five model and vision forwards required")
        d = {label: cell_readbacks[label]["summary"]["target_d_z38_minus_z999"] for label in ("00", "01", "10", "11")}
        interaction = d["11"] - d["10"] - d["01"] + d["00"]
        final_readback = {
            "schema": "recurrence_first_layer_readout.readback.v1", "status": "candidate",
            "source_manifest": binding(OUT / "source-to-cell.json"), "cells": cell_readbacks,
            "native_checks": native_checks, "d_interaction_11_minus_10_minus_01_plus_00": interaction,
            "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
        }
        write(OUT / "readback.json", final_readback)
        result = {"status": "candidate", "cell_order": list(CELL_ORDER),
                  "readback": binding(OUT / "readback.json"),
                  "source_manifest": binding(OUT / "source-to-cell.json"),
                  "artifacts": {label: cell_readbacks[label]["tensor_artifact"] for label in CELL_ORDER},
                  "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                  "target_winners": {label: cell_readbacks[label]["summary"]["target_winner"] for label in CELL_ORDER},
                  "d_interaction_11_minus_10_minus_01_plus_00": interaction}
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
