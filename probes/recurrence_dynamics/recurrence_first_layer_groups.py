"""Capture the native layer-0 score-group inputs for the frozen val contrast."""
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
ROOT = BASE / "2026-09-22-recurrence-first-layer-groups"
OUT = ROOT / "attempt-001"
UNIT = Path("research/experiments/2026-09-22-recurrence-first-layer-groups/unit.md")
SELECTION = trajectory.SELECTION
ORACLE = trajectory.ORACLE
FIXED_ACCEPTANCE = trajectory.FIXED_ACCEPTANCE
TRAJECTORY_READBACK = trajectory.ROOT / "lead-checks/trajectory-readback.json"
TRAJECTORY_ACCEPTANCE = trajectory.ROOT / "lead-acceptance.json"
TRAJECTORY_RECEIPT = trajectory.ROOT / "attempt-002/receipt.json"
TRAJECTORY_VAL_READBACK = trajectory.ROOT / "attempt-002/val-readback.json"
TRAJECTORY_TENSOR = trajectory.ROOT / "attempt-002/val-trajectory.pt"
FIXED_ROOT = trajectory.FIXED_ROOT
TARGET, FULL_WIDTH = 2, 2127
QUERY_INDICES = (15, 62)
PHYSICAL_QUERIES = (1703, 2126)
RAW_OFFSETS = (384, 807)
NEXT_TOKENS = (151708, 152669)
Q_HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 16, 8, 128, 2048
LAYERS = 28
ATOL = 2e-4
MAX_MODEL_FORWARDS = 1
MAX_VISION_FORWARDS = 1
MAX_SECONDS = 10 * 60
MAX_TENSOR_BYTES = 64 << 20


def binding(path: Path) -> dict:
    return literal_binding(path)


def assert_binding(path: Path, expected: dict) -> None:
    actual = binding(path)
    require(all(actual[key] == expected[key] for key in ("path", "sha256", "size_bytes")),
            f"binding drift: {path}")


def trajectory_sources() -> tuple[dict, torch.Tensor]:
    lead = json.loads(TRAJECTORY_READBACK.read_text())
    acceptance = json.loads(TRAJECTORY_ACCEPTANCE.read_text())
    val_readback = json.loads(TRAJECTORY_VAL_READBACK.read_text())
    receipt = json.loads(TRAJECTORY_RECEIPT.read_text())
    require(lead["status"] == "independently-verified" and acceptance["status"] == "lead-accepted" and
            "val" in lead["cases"],
            "independent native trajectory receipt is not accepted")
    tensor_binding = val_readback["tensor_artifact"]
    require(tensor_binding["path"] == str(TRAJECTORY_TENSOR), "trajectory tensor path changed")
    assert_binding(TRAJECTORY_TENSOR, tensor_binding)
    prior = torch.load(TRAJECTORY_TENSOR, map_location="cpu", weights_only=True)
    require(prior["full_logits"].shape == (63, 152670), "accepted val trajectory logits shape changed")
    refs = {
        "lead_readback": binding(TRAJECTORY_READBACK),
        "trajectory_acceptance": binding(TRAJECTORY_ACCEPTANCE),
        "trajectory_receipt": binding(TRAJECTORY_RECEIPT),
        "trajectory_val_readback": binding(TRAJECTORY_VAL_READBACK),
        "trajectory_tensor": tensor_binding,
        "trajectory_receipt_status": receipt["status"],
    }
    return refs, prior["full_logits"].index_select(0, torch.tensor(QUERY_INDICES))


def cpu_selfcheck() -> None:
    """Reuse the frozen oracle/index checks without constructing a new test framework."""
    selection = json.loads(SELECTION.read_text())
    oracle = json.loads(ORACLE.read_text())
    require(selection["status"] == "root-verified-frozen" and
            oracle["status"] == "root-independent-source-oracle", "frozen source status changed")
    meta = selection["cases"]["val"]
    raw_path = Path(oracle["cases"]["val"]["source_bindings"]["raw"]["path"])
    trace_path = Path(oracle["cases"]["val"]["source_bindings"]["trace"]["path"])
    raw = json.loads(raw_path.read_text())["rows"]
    trace = json.loads(trace_path.read_text())
    tokens = raw[TARGET]["token_ids"]
    base = int(meta["full_width"]) - int(meta["raw_action_offset"])
    fake_ids = torch.zeros((4, int(meta["full_width"])), dtype=torch.long)
    fake_ids[TARGET, base:base + int(meta["raw_action_offset"])] = torch.tensor(tokens[:int(meta["raw_action_offset"])])
    queries = trajectory.source_queries("val", {"raw_action_offset": meta["raw_action_offset"]},
                                       oracle, raw, trace, {"input_ids": fake_ids})
    selected = [queries[i] for i in QUERY_INDICES]
    require([x["physical_query_index"] for x in selected] == list(PHYSICAL_QUERIES),
            "selected physical query indices changed")
    require([x["raw_action_offset"] for x in selected] == list(RAW_OFFSETS) and
            [x["next_token"] for x in selected] == list(NEXT_TOKENS),
            "selected raw next-token binding changed")
    require(all(x["minus_one_action_token"] != x["next_token"] for x in selected),
            "one-token source shift did not fail")
    trajectory.fixed_artifact_bindings("val")
    refs, prior = trajectory_sources()
    require(prior.shape == (2, 152670) and refs["trajectory_receipt_status"] == "technical_invalid",
            "immutable accepted trajectory reference changed")
    print(json.dumps({"status": "selfcheck_ok", "queries": list(PHYSICAL_QUERIES),
                      "next_tokens": list(NEXT_TOKENS), "transformers": transformers.__version__}))


def run(device: str) -> None:
    cpu_selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA device required")
    require(not OUT.exists(), "attempt path already exists; preserve failed attempts")
    oracle = json.loads(ORACLE.read_text())
    refs, accepted_logits = trajectory_sources()
    OUT.mkdir(parents=True)
    state = {
        "status": "preparing",
        "pid": os.getpid(),
        "device": device,
        "model_forwards": 0,
        "vision_forwards": 0,
        "calls": [],
        "started_unix": time.time(),
        "started": time.monotonic(),
        "source": {
            "unit": binding(UNIT),
            "trajectory_acceptance": refs["trajectory_acceptance"],
            "selection": binding(SELECTION),
            "oracle": binding(ORACLE),
            "fixed_acceptance": binding(FIXED_ACCEPTANCE),
            **refs,
        },
    }
    write(OUT / "receipt.json", state)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        case, meta, all_queries, raw, trace = trajectory.source_inputs("val", q, identity, oracle, device)
        queries = [all_queries[i] for i in QUERY_INDICES]
        require([x["physical_query_index"] for x in queries] == list(PHYSICAL_QUERIES),
                "native selected query positions changed")
        require([x["raw_action_offset"] for x in queries] == list(RAW_OFFSETS) and
                [x["next_token"] for x in queries] == list(NEXT_TOKENS),
                "native selected query crosswalk changed")
        native = case["native"]
        require(native["input_ids"].shape == (4, FULL_WIDTH) and case["target"] == TARGET,
                "val native batch or width changed")
        producer_capture = preserve_source(
            Path(__file__), run_root=OUT, relative_name="recurrence_first_layer_groups.py")
        dependency_paths = [
            Path('probes/recurrence_dynamics/recurrence_native_trajectory.py'),
            Path('probes/recurrence_dynamics/recurrence_attention_mass.py'),
            Path('probes/recurrence_dynamics/recurrence_fixed_template.py'),
            Path('probes/recurrence_dynamics/recurrence_position_history.py'),
            Path("probes/model_profiles/mature_tied_untied.py"),
            Path('probes/recurrence_dynamics/numerical_feedback/runtime.py'),
            Path("src/qwen/native.py"),
            Path("src/qwen/input_identity.py"),
            Path("src/inference/bound_requests.py"),
            Path("src/qwen/untied_embeddings.py"),
            Path(inspect.getfile(modeling_qwen3_vl)),
        ]
        dependency_captures = [
            preserve_source(path, run_root=OUT,
                            relative_name=str(path) if not path.is_absolute() else f"transformers/{path.name}")
            for path in dependency_paths
        ]
        source_manifest = {
            "schema": "recurrence_first_layer_groups.v1",
            "status": "frozen_before_forward",
            "source": state["source"],
            "producer": binding(Path(__file__)),
            "producer_capture": binding(producer_capture),
            "dependency_captures": [binding(path) for path in dependency_captures],
            "transformers_version": transformers.__version__,
            "model_identity": identity,
            "case": {
                "name": "val",
                "target_batch": TARGET,
                "full_width": FULL_WIDTH,
                "native_input_hashes": {
                    key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")
                },
                "native_input_shapes": {
                    key: list(native[key].shape) for key in ("input_ids", "attention_mask", "position_ids")
                },
                "source_bindings": case["source_bindings"],
                "accepted_paths": case["accepted_paths"],
                "fixed_artifacts": case["fixed_bindings"],
                "query_indices": list(PHYSICAL_QUERIES),
                "query_metadata": queries,
            },
            "previous_native_trajectory": refs,
            "collection": {
                "one_full_native_forward": True,
                "no_generation": True,
                "no_intervention": True,
                "selected_logits": "physical query indices via tensor logits_to_keep",
                "model_forward_cap": MAX_MODEL_FORWARDS,
                "vision_forward_cap": MAX_VISION_FORWARDS,
                "wall_seconds_cap": MAX_SECONDS,
                "tensor_cap_bytes": MAX_TENSOR_BYTES,
                "logit_parity_atol": ATOL,
                "captured_layer": 0,
                "captured_cache": "layer0 post-K and V only",
            },
        }
        write(OUT / "source-to-cell.json", source_manifest)
        state.update(status="executing", manifest=binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        query_indices = torch.tensor(PHYSICAL_QUERIES, device=device, dtype=torch.long)
        cache = DynamicCache()
        inputs = dict(native)
        inputs.update(logits_to_keep=query_indices, past_key_values=cache,
                      cache_position=torch.arange(FULL_WIDTH, device=device), use_cache=True)
        text = q.model.model.language_model
        pre_q = query_cos = query_sin = head_output = None
        handles = []

        def rotary_hook(_module, _args, output):
            nonlocal query_cos, query_sin
            cos, sin = output
            require(cos.shape == sin.shape == (4, FULL_WIDTH, HEAD_DIM), "native phase shape changed")
            query_cos = cos[TARGET].index_select(0, query_indices).detach().float().cpu().clone()
            query_sin = sin[TARGET].index_select(0, query_indices).detach().float().cpu().clone()

        def q_norm_hook(_module, _args, output):
            nonlocal pre_q
            require(output.shape == (4, FULL_WIDTH, Q_HEADS, HEAD_DIM), "layer0 pre-Q shape changed")
            pre_q = output[TARGET].index_select(0, query_indices).detach().float().cpu().clone()

        def o_proj_input(_module, args):
            nonlocal head_output
            selected = args[0][TARGET].index_select(0, query_indices)
            require(selected.shape == (2, HIDDEN), "layer0 o_proj input shape changed")
            head_output = selected.reshape(2, Q_HEADS, HEAD_DIM).detach().float().cpu().clone()

        handles.extend((text.rotary_emb.register_forward_hook(rotary_hook),
                        text.layers[0].self_attn.q_norm.register_forward_hook(q_norm_hook),
                        text.layers[0].self_attn.o_proj.register_forward_pre_hook(o_proj_input)))
        observers, observer_handles = trajectory.attention_attestors(
            text, q.model.lm_head, inputs, cache, TARGET, query_indices, FULL_WIDTH)
        handles.extend(observer_handles)

        def count_model(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_MODEL_FORWARDS and
                    time.monotonic() - state["started"] <= MAX_SECONDS, "model forward/time cap exceeded")

        def count_vision(*_args):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= MAX_VISION_FORWARDS, "vision forward cap exceeded")

        count_handles = [q.model.register_forward_pre_hook(count_model, with_kwargs=True),
                         q.model.model.visual.register_forward_pre_hook(count_vision)]
        started = time.monotonic()
        try:
            with torch.inference_mode():
                output = q.model(**inputs)
            state["calls"].append({"case": "val", "seconds": time.monotonic() - started})
            logits = output.logits[TARGET].detach().float().cpu().clone()
            post_k = cache.layers[0].keys[TARGET].detach().float().cpu().clone()
            values = cache.layers[0].values[TARGET].detach().float().cpu().clone()
        finally:
            for handle in handles + count_handles:
                handle.remove()
        state.update(status="captured", model_forwards=state["model_forwards"],
                     vision_forwards=state["vision_forwards"])
        write(OUT / "receipt.json", state)
        require(output.past_key_values is cache and cache.get_seq_length() == FULL_WIDTH and
                logits.shape == (2, 152670) and post_k.shape == (KV_HEADS, FULL_WIDTH, HEAD_DIM) and
                values.shape == post_k.shape and pre_q is not None and pre_q.shape == (2, Q_HEADS, HEAD_DIM) and
                query_cos is not None and query_sin is not None and head_output is not None and
                head_output.shape == (2, Q_HEADS, HEAD_DIM), "native layer0 capture incomplete")
        payload = {
            "post_K": post_k,
            "V": values,
            "pre_Q": pre_q,
            "query_cos": query_cos,
            "query_sin": query_sin,
            "head_output": head_output,
            "full_logits": logits,
            "query_positions": query_indices.cpu(),
        }
        tensor_path = OUT / "native.pt"
        torch.save(payload, tensor_path)
        require(tensor_path.stat().st_size <= MAX_TENSOR_BYTES, "native tensor payload cap exceeded")

        fixed = torch.load(FIXED_ROOT / "val/first-template-and-phases.pt", map_location="cpu", weights_only=True)
        repeated_v = values[:, 1563:2121].reshape(KV_HEADS, 62, 9, HEAD_DIM)
        repeated_error = float((repeated_v - fixed["first_V"][0].unsqueeze(1)).abs().max())
        current_error = float((values[:, 2121:2127] - fixed["first_V"][0][:, :6]).abs().max())
        selected_error = (logits - accepted_logits).abs().amax(dim=1)
        actual_winners = logits.argmax(dim=1)
        accepted_winners = accepted_logits.argmax(dim=1)
        parity = {
            "max_abs_vs_accepted_trajectory": float(selected_error.max()),
            "per_query_max_abs": [float(x) for x in selected_error],
            "actual_winners": [int(x) for x in actual_winners],
            "accepted_winners": [int(x) for x in accepted_winners],
            "winner_exact": bool(torch.equal(actual_winners, accepted_winners)),
        }
        readback = {
            "case": "val",
            "target_batch": TARGET,
            "query_rows": [int(all_queries[i]["row"]) for i in QUERY_INDICES],
            "query_metadata": queries,
            "native_input_hashes": {key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")},
            "native_input_shapes": {key: list(native[key].shape) for key in ("input_ids", "attention_mask", "position_ids")},
            "native_input_dtypes": {key: str(native[key].dtype) for key in ("input_ids", "attention_mask", "position_ids")},
            "query_consumer": observers["lm_head"],
            "attention": observers["attention"],
            "embedding_calls": observers["embedding"],
            "rotary_calls": observers["rotary"],
            "cache_length": cache.get_seq_length(),
            "stationary_template": {
                "repeated_row_V_max_abs": repeated_error,
                "current_prefix_V_max_abs": current_error,
                "atol": ATOL,
            },
            "parity": parity,
            "tensor_artifact": binding(tensor_path),
            "tensor_bytes": tensor_path.stat().st_size,
            "source": state["source"],
            "source_manifest": binding(OUT / "source-to-cell.json"),
            "model_identity": identity,
        }
        write(OUT / "readback.json", readback)
        require(state["model_forwards"] == 1 and state["vision_forwards"] == 1,
                "exactly one model and vision forward required")
        require(observers["embedding"] == observers["rotary"] == 1 and observers["lm_head"] is not None and
                all(x is not None for x in observers["attention"]), "native helper attestations incomplete")
        require(parity["max_abs_vs_accepted_trajectory"] <= ATOL and parity["winner_exact"],
                "selected trajectory logit parity failed")
        require(repeated_error <= ATOL and current_error <= ATOL,
                "layer0 stationary V/template parity failed")
        result = {
            "status": "candidate",
            "case": "val",
            "queries": list(PHYSICAL_QUERIES),
            "model_forwards": state["model_forwards"],
            "vision_forwards": state["vision_forwards"],
            "parity": parity,
            "readback": binding(OUT / "readback.json"),
            "tensor_artifact": binding(tensor_path),
            "source_manifest": binding(OUT / "source-to-cell.json"),
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
