"""Capture one exact native trajectory for the frozen recurrence cases."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch
import transformers
from transformers import DynamicCache, Qwen3VLConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_attention_mass import (
    HEAD_DIM,
    KV_HEADS,
    LAYERS,
    prepare_case,
)
from probes.training_set_completion.recurrence_donor_tracking import require, write
from probes.training_set_completion.recurrence_key_phase import rotate
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-22-recurrence-native-trajectory"
OUT = ROOT / "attempt-003"
UNIT = Path("research/experiments/2026-09-22-recurrence-native-trajectory/unit.md")
EXECUTION_RULING = Path("research/experiments/2026-09-22-recurrence-native-trajectory/execution-ruling.md")
SELECTION = BASE / "2026-09-22-recurrence-fixed-template/selection.json"
ORACLE = ROOT / "lead-checks/selection-oracle.json"
FIXED_ACCEPTANCE = BASE / "2026-09-22-recurrence-fixed-template/lead-acceptance.json"
FIXED_ROOT = BASE / "2026-09-22-recurrence-fixed-template/attempt-001"
ATOL = 2e-4
MAX_MODEL_FORWARDS = 4
MAX_VISION_FORWARDS = 2
MAX_SECONDS = 15 * 60
MAX_CASE_BYTES = 256 << 20
Q_HEADS, HIDDEN = 16, 2048

CASE_NAMES = ("train",)


def binding(path: Path) -> dict:
    return literal_binding(path)


def assert_binding(path: Path, expected: dict) -> None:
    actual = binding(path)
    require(
        actual["path"] == expected["path"]
        and actual["sha256"] == expected["sha256"]
        and actual["size_bytes"] == expected["size_bytes"],
        f"source binding drift: {path}",
    )


def fixed_artifact_bindings(name: str) -> dict:
    """Bind the exact fixed-template and accepted NN artifacts before collection."""
    root = FIXED_ROOT / name
    prefill = json.loads((root / "prefill-readback.json").read_text())
    result = json.loads((root / "result.json").read_text())
    template = prefill["template_and_phases"]
    nn = result["cells"]["NN"]["full_logits"]
    require(template["path"] == str(root / "first-template-and-phases.pt"),
            f"{name} fixed-template receipt points to a different artifact")
    require(nn["path"] == str(root / "NN.pt"), f"{name} fixed NN receipt points to a different artifact")
    assert_binding(Path(template["path"]), template)
    assert_binding(Path(nn["path"]), nn)
    return {"first_template": template, "NN": nn}


def source_queries(name: str, case: dict, oracle: dict, raw: list[dict], trace: dict, native: dict) -> list[dict]:
    selected = oracle["cases"][name]
    target = int(selected["target"])
    queries = selected["queries"]
    require(len(queries) == (63 if name == "val" else 5), f"{name} query denominator changed")
    full_width = int(native["input_ids"].shape[1])
    base = full_width - int(case["raw_action_offset"])
    trace_by_offset = {int(step["offset"]): step for step in trace["steps"]}
    tokens = raw[target]["token_ids"]
    out = []
    for item in queries:
        row = int(item["row"])
        offset = int(item["raw_action_offset"])
        physical = int(item["physical_query_index"])
        require(physical == base + offset - 1, f"{name} raw/physical query map changed at row {row}")
        require(0 <= physical < full_width and offset > 0 and offset < len(tokens), f"{name} query bounds changed")
        require(tokens[offset - 1] == item["expected_query_input_token"], f"{name} source query input token drift")
        require(tokens[offset] == item["chosen_token"], f"{name} source next-token drift")
        require(int(native["input_ids"][target, physical]) == item["expected_query_input_token"],
                f"{name} native input/query binding drift at row {row}")
        saved = trace_by_offset.get(offset)
        require(saved is not None and int(saved["offset"]) == offset, f"{name} trace offset {offset} missing")
        saved_winner = int(saved["raw_winners"][target])
        saved_chosen = int(saved["chosen"][target])
        require(saved_winner == saved_chosen == int(item["chosen_token"]), f"{name} trace chosen/winner mismatch at {offset}")
        require(tuple(saved["raw_top2"][target]) == tuple(item["top2_logits"]), f"{name} oracle top2 drift at {offset}")
        out.append({
            "row": row,
            "raw_action_offset": offset,
            "physical_query_index": physical,
            "query_input_token": int(item["expected_query_input_token"]),
            "next_token": int(item["chosen_token"]),
            "trace_top2_tokens": [int(x) for x in saved["raw_winners"][target:target + 1] + saved["raw_runnerups"][target:target + 1]],
            "trace_top2_logits": [float(x) for x in saved["raw_top2"][target]],
            "minus_one_action_token": int(item["minus_one_action_token"]),
        })
    require(len({x["physical_query_index"] for x in out}) == len(out), f"{name} query positions duplicate")
    return out


def row_geometry(name: str, selection: dict, native: dict) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    meta = selection["cases"][name]
    source = tuple(meta["first_source"])
    all_dest = tuple(meta["all_destination"])
    width = int(meta["historical_width"])
    rows = [source] + [(p, p + 9) for p in range(all_dest[0], all_dest[1], 9)]
    require(len(rows) == int(meta["row_count"]) and rows[-1][1] == width, f"{name} repeated-row geometry changed")
    target = int(meta["target_batch"])
    tokens = meta["template_token_ids"]
    require(all(native["input_ids"][target, a:b].tolist() == tokens for a, b in rows), f"{name} repeated tokens changed")
    return rows, source


def source_inputs(name: str, q, identity: dict, oracle: dict, device: str) -> tuple[dict, dict, list[dict], dict, dict]:
    case = prepare_case(name, q, identity, device)
    selection = json.loads(SELECTION.read_text())
    meta = selection["cases"][name]
    require(int(meta["target_batch"]) == case["target"], f"{name} target batch changed")
    require(int(meta["historical_width"]) == case["width"] and int(meta["full_width"]) == case["full"],
            f"{name} native width changed")
    raw_path = Path(case["source_bindings"]["raw"]["path"])
    trace_path = Path(case["source_bindings"]["trace"]["path"])
    for kind in ("raw", "trace"):
        expected = oracle["cases"][name]["source_bindings"][kind]
        observed = case["source_bindings"][kind]
        require(all(observed[key] == expected[key] for key in ("path", "sha256", "size_bytes")),
                f"{name} {kind} source binding differs from frozen oracle")
        assert_binding(Path(expected["path"]), expected)
    raw = json.loads(raw_path.read_text())["rows"]
    trace = json.loads(trace_path.read_text())
    queries = source_queries(name, {"raw_action_offset": meta["raw_action_offset"]}, oracle, raw, trace, case["native"])
    rows, source = row_geometry(name, selection, case["native"])
    require(tuple(case["source"]) == rows[-2] and tuple(case["dest"]) == rows[-1],
            f"{name} accepted intervention boundary differs from selection")
    case["fixed_bindings"] = fixed_artifact_bindings(name)
    return case, meta, queries, raw, trace


def capture_phase(output: torch.Tensor, target: int, indices: torch.Tensor) -> torch.Tensor:
    return output[target].index_select(0, indices).detach().float().cpu().clone()


def capture_rows(output: torch.Tensor, target: int, rows: list[tuple[int, int]], kv_heads: int, head_dim: int) -> torch.Tensor:
    start, end = rows[0][0], rows[-1][1]
    require(output.shape[0] > target and output.shape[1] >= end, "activation batch/length changed")
    block = output[target, start:end]
    require(block.shape[0] == sum(b - a for a, b in rows), "row activation span changed")
    if block.ndim == 2:
        block = block.reshape(len(rows), 9, kv_heads, head_dim)
    else:
        require(block.ndim == 3 and block.shape[-2:] == (kv_heads, head_dim), "activation head shape changed")
        block = block.reshape(len(rows), 9, kv_heads, head_dim)
    return block.permute(0, 2, 1, 3).detach().float().cpu().clone()


def trajectory_metrics(tensor: torch.Tensor, *, name: str) -> dict:
    # tensor is [rows, heads, 9, dim]; retain every transition and the frozen val gate.
    delta = torch.linalg.vector_norm((tensor[1:] - tensor[:-1]).double().flatten(1), dim=1)
    total = float(delta.sum())
    row_norm = torch.linalg.vector_norm(tensor.double().flatten(1), dim=1)
    threshold = float(64 * torch.finfo(torch.float32).eps * row_norm.max())
    active = total > threshold
    if name == "val":
        late = float(delta[15:].sum() / total) if active else None
        burn = 15
    else:
        late = None
        burn = None
    return {
        "rows": int(tensor.shape[0]),
        "transitions": int(delta.numel()),
        "transition_frobenius": [float(x) for x in delta],
        "total_movement": total,
        "active_threshold": threshold,
        "active": bool(active),
        "burn_in_transitions": burn,
        "late_fraction": late,
    }


def phase_metrics(pre_k: list[torch.Tensor], row_cos: torch.Tensor, row_sin: torch.Tensor, packet: dict, first_template: dict) -> tuple[list[dict], dict]:
    require(len(pre_k) == len(packet["layers"]) == LAYERS and row_cos.shape == row_sin.shape and row_cos.ndim == 3,
            "phase/layer axes changed")
    metrics = []
    for i, (key, saved) in enumerate(zip(pre_k, packet["layers"], strict=True)):
        post = rotate(key[-1], row_cos[-1], row_sin[-1])
        penultimate_post = rotate(key[-2], row_cos[-2], row_sin[-2])
        layer = {
            "layer": i,
            "first_preK_max_abs_vs_fixed_template": float((key[0] - first_template["first_preK"][i]).abs().max()),
            "penultimate_preK_max_abs_vs_phase_packet_pre_old": float((key[-2] - saved["pre_old"]).abs().max()),
            "penultimate_postK_replay_max_abs_vs_phase_packet_post_old": float((penultimate_post - saved["post_old"]).abs().max()),
            "first_V_max_abs_vs_fixed_template": None,
            "last_preK_max_abs_vs_phase_packet": float((key[-1] - saved["pre_new"]).abs().max()),
            "last_postK_replay_max_abs_vs_phase_packet": float((post - saved["post_new"]).abs().max()),
            "last_V_max_abs_vs_phase_packet": None,
        }
        metrics.append(layer)
    phase = {
        "first_cos_hash": tensor_hash(row_cos[0]),
        "first_sin_hash": tensor_hash(row_sin[0]),
        "last_cos_hash": tensor_hash(row_cos[-1]),
        "last_sin_hash": tensor_hash(row_sin[-1]),
    }
    return metrics, phase


def cache_attestations(pre_k: list[torch.Tensor], values: list[torch.Tensor], row_cos: torch.Tensor,
                       row_sin: torch.Tensor, row_indices: torch.Tensor, cache, target: int,
                       full_width: int) -> list[dict]:
    """Compare the live cache's selected rows with this forward's captured K/V."""
    require(cache.get_seq_length() == full_width, "native cache full length changed")
    records = []
    for i, (key, value) in enumerate(zip(pre_k, values, strict=True)):
        require(key.shape == value.shape and key.ndim == 4, "captured K/V row shape changed")
        key_replay = rotate(key.permute(1, 0, 2, 3).reshape(key.shape[1], -1, key.shape[3]),
                            row_cos.reshape(-1, row_cos.shape[-1]),
                            row_sin.reshape(-1, row_sin.shape[-1]))
        value_replay = value.permute(1, 0, 2, 3).reshape(value.shape[1], -1, value.shape[3])
        layer = cache.layers[i]
        require(layer.keys.shape[0] > target and layer.keys.shape[1:] ==
                (key.shape[1], full_width, key.shape[3]) and cache.get_seq_length(i) == full_width,
                "native cached K shape changed")
        require(layer.values.shape == layer.keys.shape, "native cached V shape changed")
        actual_key = layer.keys[target].index_select(1, row_indices).detach().float().cpu()
        actual_value = layer.values[target].index_select(1, row_indices).detach().float().cpu()
        records.append({
            "layer": i,
            "stored_postK_max_abs_vs_preK_phase": float((actual_key - key_replay.cpu()).abs().max()),
            "stored_V_max_abs_vs_v_proj": float((actual_value - value_replay.cpu()).abs().max()),
            "stored_postK_hash": tensor_hash(actual_key),
            "replayed_postK_hash": tensor_hash(key_replay),
            "stored_V_hash": tensor_hash(actual_value),
            "captured_V_hash": tensor_hash(value_replay),
        })
    return records


def query_summary(logits: torch.Tensor, queries: list[dict], trace: dict, target: int, name: str, accepted: torch.Tensor) -> tuple[dict, dict]:
    top = torch.topk(logits, 5, dim=-1)
    trace_by_offset = {int(x["offset"]): x for x in trace["steps"]}
    rows = []
    max_top2_error = 0.0
    identity_failures = []
    for i, q in enumerate(queries):
        saved = trace_by_offset[q["raw_action_offset"]]
        ids = [int(x) for x in top.indices[i, :2]]
        values = [float(x) for x in top.values[i, :2]]
        error = max(abs(values[j] - float(saved["raw_top2"][target][j])) for j in range(2))
        max_top2_error = max(max_top2_error, error)
        if ids[0] != q["next_token"] or ids[0] != int(saved["raw_winners"][target]):
            identity_failures.append(q["raw_action_offset"])
        rows.append({
            **q,
            "argmax_token": ids[0],
            "top1_top2_gap": float(top.values[i, 0] - top.values[i, 1]),
            "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices[i], top.values[i], strict=True)],
            "trace_top2_max_abs_error": error,
            "z38": float(logits[i, 151708]) if name == "val" else None,
            "z999": float(logits[i, 152669]) if name == "val" else None,
            "z38_minus_z999": float(logits[i, 151708] - logits[i, 152669]) if name == "val" else None,
            "z350": float(logits[i, 152020]) if name == "train" else None,
            "z348": float(logits[i, 152018]) if name == "train" else None,
            "z591": float(logits[i, 152261]) if name == "train" else None,
            "z350_minus_z348": float(logits[i, 152020] - logits[i, 152018]) if name == "train" else None,
            "z591_minus_z348": float(logits[i, 152261] - logits[i, 152018]) if name == "train" else None,
        })
    delta = logits.double()[1:] - logits.double()[:-1]
    contextual = {
        "consecutive_l2": [float(torch.linalg.vector_norm(x)) for x in delta],
        "consecutive_max_abs": [float(x.abs().max()) for x in delta],
        "from_first_l2": [float(torch.linalg.vector_norm(x - logits.double()[0])) for x in logits.double()],
    }
    final_error = float((logits[-1] - accepted.float()).abs().max())
    result = {
        "queries": rows,
        "max_trace_top2_abs_error": max_top2_error,
        "trace_identity_failures": identity_failures,
        "final_full_vector_max_abs_vs_accepted_NN": final_error,
        "final_argmax": int(top.indices[-1, 0]),
        "final_top1_top2_gap": float(top.values[-1, 0] - top.values[-1, 1]),
        "contextual_change": contextual,
    }
    bins = (151708, 152669) if name == "val" else (152020, 152018, 152261)
    probabilities = torch.softmax(logits.double(), dim=-1)
    result["fp64_probabilities"] = [[float(probabilities[i, token]) for token in bins] for i in range(len(queries))]
    return result, {"max_abs": final_error, "max_top2": max_top2_error, "argmax": int(top.indices[-1, 0])}


def attention_attestors(text, lm_head, inputs: dict, cache, target: int,
                        query_positions: torch.Tensor, row_end: int):
    expected_ids = inputs["input_ids"]
    expected_positions = inputs["position_ids"]
    query_cpu = query_positions.cpu()
    state = {"embedding": 0, "rotary": 0, "attention": [None] * LAYERS, "lm_head": None, "backbone": None}
    handles = []

    def embedding(_module, args):
        require(torch.equal(args[0], expected_ids), "embedding consumed changed native input ids")
        state["embedding"] += 1

    def rotary(_module, args):
        require(torch.equal(args[1], expected_positions), "rotary consumed changed native positions")
        state["rotary"] += 1

    def backbone(_module, _args, output):
        state["backbone"] = output.last_hidden_state

    def lm_head_hook(_module, args):
        selected = args[0]
        full = state["backbone"]
        require(full is not None and selected.shape[1] == len(query_positions), "LM head selection shape changed")
        expected = full.index_select(1, query_positions)
        require(torch.equal(selected, expected), "LM head did not consume declared physical query rows")
        state["lm_head"] = {"shape": list(selected.shape), "physical_indices": query_cpu.tolist(), "exact": True}

    handles.extend((text.embed_tokens.register_forward_pre_hook(embedding),
                    text.rotary_emb.register_forward_pre_hook(rotary),
                    text.register_forward_hook(backbone),
                    lm_head.register_forward_pre_hook(lm_head_hook)))

    def attention(layer_index):
        def hook(_module, _args, kwargs):
            consumed_cache = kwargs.get("past_key_values")
            require(consumed_cache is cache and cache.get_seq_length(layer_index) == 0,
                    "attention did not consume the empty native cache at layer entry")
            mask = kwargs.get("attention_mask")
            slots = kwargs.get("cache_position")
            phase = kwargs.get("position_embeddings")
            require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.ndim == 4,
                    "attention did not consume an explicit boolean causal mask")
            require(mask.shape[0] == expected_ids.shape[0] and mask.shape[1] == 1 and
                    mask.shape[-2:] == (row_end, row_end),
                    "attention mask batch/length changed")
            require(isinstance(slots, torch.Tensor) and torch.equal(slots, torch.arange(row_end, device=slots.device)),
                    "attention cache slots changed")
            require(isinstance(phase, tuple) and len(phase) == 2, "attention did not consume rotary phases")
            target_mask = mask[target, 0].index_select(0, query_positions)
            expected_visible = torch.arange(row_end, device=mask.device)[None, :] <= query_positions[:, None]
            require(torch.equal(target_mask, expected_visible), "query causal/left-padding visibility changed")
            state["attention"][layer_index] = {
                "mask_shape": list(mask.shape),
                "mask_hash": tensor_hash(mask),
                "cache_slots_hash": tensor_hash(slots),
                "query_mask_hash": tensor_hash(target_mask),
                "query_visibility_exact": True,
                "cache_is_native_empty_at_entry": True,
                "phase_shapes": [list(x.shape) for x in phase],
            }
        return hook

    for i, layer in enumerate(text.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(attention(i), with_kwargs=True))
    return state, handles


def collect_case(name: str, case: dict, queries: list[dict], q, device: str, state: dict) -> tuple[dict, dict]:
    target = int(case["target"])
    native = case["native"]
    rows, _ = row_geometry(name, json.loads(SELECTION.read_text()), native)
    row_indices = torch.tensor([p for a, b in rows for p in range(a, b)], device=device, dtype=torch.long)
    require(torch.equal(row_indices, torch.arange(rows[0][0], rows[-1][1], device=device)),
            f"{name} repeated-row capture span is not contiguous")
    query_indices = torch.tensor([x["physical_query_index"] for x in queries], device=device, dtype=torch.long)
    inputs = dict(native)
    cache = DynamicCache()
    inputs.update(logits_to_keep=query_indices, past_key_values=cache,
                  cache_position=torch.arange(native["input_ids"].shape[1], device=device), use_cache=True)
    require(torch.equal(native["input_ids"], inputs["input_ids"]) and torch.equal(native["attention_mask"], inputs["attention_mask"]),
            f"{name} native input changed while selecting logits")
    text = q.model.model.language_model
    n_rows, n_queries = len(rows), len(queries)
    pre_k, values, pre_q = [None] * LAYERS, [None] * LAYERS, [None] * LAYERS
    residual_in, residual_out = [None] * LAYERS, [None] * LAYERS
    row_cos, row_sin, query_cos, query_sin = None, None, None, None

    def rotary(_module, _args, output):
        nonlocal row_cos, row_sin, query_cos, query_sin
        cos, sin = output
        require(cos.shape == sin.shape == (native["input_ids"].shape[0], native["input_ids"].shape[1], HEAD_DIM),
                f"{name} rotary shape changed")
        row_cos = cos[target].index_select(0, row_indices).reshape(n_rows, 9, HEAD_DIM).detach().float().cpu().clone()
        row_sin = sin[target].index_select(0, row_indices).reshape(n_rows, 9, HEAD_DIM).detach().float().cpu().clone()
        query_cos = cos[target].index_select(0, query_indices).detach().float().cpu().clone()
        query_sin = sin[target].index_select(0, query_indices).detach().float().cpu().clone()

    handles = [text.rotary_emb.register_forward_hook(rotary)]
    for i, layer in enumerate(text.layers):
        attn = layer.self_attn

        def k_norm(_module, _args, output, *, index=i):
            require(output.shape[-2:] == (KV_HEADS, HEAD_DIM), f"{name} normalized K shape changed")
            pre_k[index] = capture_rows(output, target, rows, KV_HEADS, HEAD_DIM)

        def v_proj(_module, _args, output, *, index=i):
            require(output.shape[-1] == KV_HEADS * HEAD_DIM, f"{name} V projection width changed")
            values[index] = capture_rows(output, target, rows, KV_HEADS, HEAD_DIM)

        def q_norm(_module, _args, output, *, index=i):
            require(output.shape[-2:] == (Q_HEADS, HEAD_DIM), f"{name} normalized Q shape changed")
            pre_q[index] = capture_phase(output, target, query_indices)

        def layer_in(_module, args, *, index=i):
            residual_in[index] = capture_phase(args[0], target, query_indices)

        def layer_out(_module, _args, output, *, index=i):
            residual_out[index] = capture_phase(output, target, query_indices)

        handles.extend((attn.k_norm.register_forward_hook(k_norm), attn.v_proj.register_forward_hook(v_proj),
                        attn.q_norm.register_forward_hook(q_norm), layer.register_forward_pre_hook(layer_in),
                        layer.register_forward_hook(layer_out)))
    observers, observer_handles = attention_attestors(text, q.model.lm_head, inputs, cache, target,
                                                      query_indices, int(native["input_ids"].shape[1]))
    handles.extend(observer_handles)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            output = q.model(**inputs)
        state["calls"].append({"case": name, "seconds": time.monotonic() - started})
        logits = output.logits[target].detach().float().cpu().clone()
    finally:
        for handle in handles:
            handle.remove()
    require(logits.shape[0] == n_queries and logits.shape[1] == q.model.config.text_config.vocab_size,
            f"{name} selected full-vocabulary logits shape changed")
    require(all(x is not None for x in (*pre_k, *values, *pre_q, *residual_in, *residual_out)),
            f"{name} layer capture incomplete")
    require(row_cos is not None and row_sin is not None and query_cos is not None and query_sin is not None,
            f"{name} phase capture incomplete")
    require(output.past_key_values is cache and cache.get_seq_length() == int(native["input_ids"].shape[1]) and
            observers["embedding"] == observers["rotary"] == 1 and observers["lm_head"] is not None,
            f"{name} native consumer observation incomplete")
    require(all(x is not None for x in observers["attention"]), f"{name} layer attention observation incomplete")
    require(state["model_forwards"] <= MAX_MODEL_FORWARDS and time.monotonic() - state["started"] <= MAX_SECONDS,
            "forward/time cap exceeded")

    # Persist the expensive forward captures before any packet, cache, or logit postprocessing.
    tensors = {
        "row_ranges": rows,
        "query_positions": query_indices.cpu(),
        "query_metadata": queries,
        "pre_rope_K": pre_k,
        "V": values,
        "pre_rope_Q": pre_q,
        "residual_input": residual_in,
        "residual_output": residual_out,
        "row_cos": row_cos,
        "row_sin": row_sin,
        "query_cos": query_cos,
        "query_sin": query_sin,
        "full_logits": logits,
    }
    tensor_path = OUT / f"{name}-trajectory.pt"
    torch.save(tensors, tensor_path)

    packet = case["packet"]
    fixed_path = Path(case["fixed_bindings"]["first_template"]["path"])
    fixed = torch.load(fixed_path, map_location="cpu", weights_only=True)
    phase_layers, phase_hashes = phase_metrics(pre_k, row_cos, row_sin, packet, fixed)
    for i, layer in enumerate(phase_layers):
        layer["first_V_max_abs_vs_fixed_template"] = float((values[i][0] - fixed["first_V"][i]).abs().max())
        layer["last_V_max_abs_vs_phase_packet"] = float((values[i][-1] - packet["layers"][i]["native_V_dest"]).abs().max())
    cache_layers = cache_attestations(pre_k, values, row_cos, row_sin, row_indices, cache, target,
                                      int(native["input_ids"].shape[1]))
    phase_gate = max(max(float(x[key]) for x in phase_layers)
                     for key in ("first_preK_max_abs_vs_fixed_template", "first_V_max_abs_vs_fixed_template",
                                 "penultimate_preK_max_abs_vs_phase_packet_pre_old",
                                 "penultimate_postK_replay_max_abs_vs_phase_packet_post_old",
                                 "last_preK_max_abs_vs_phase_packet", "last_postK_replay_max_abs_vs_phase_packet",
                                 "last_V_max_abs_vs_phase_packet"))
    cache_gate = max(max(float(x[key]) for x in cache_layers)
                     for key in ("stored_postK_max_abs_vs_preK_phase", "stored_V_max_abs_vs_v_proj"))
    accepted = torch.load(Path(case["fixed_bindings"]["NN"]["path"]), map_location="cpu", weights_only=True)
    summary, parity = query_summary(logits, queries, case["trace"], target, name, accepted)
    readback = {
        "schema": "recurrence_native_trajectory.readback.v1",
        "case": name,
        "target_batch": target,
        "native_input_hashes": {key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")},
        "fixed_artifacts": case["fixed_bindings"],
        "rows": rows,
        "query_consumer": observers["lm_head"],
        "attention": observers["attention"],
        "cache": cache_layers,
        "phase": phase_hashes,
        "layers": phase_layers,
        "phase_gate_max_abs": phase_gate,
        "cache_gate_max_abs": cache_gate,
        "gate_atol": ATOL,
        "parity": parity,
        "tensor_artifact": binding(tensor_path),
        "tensor_bytes": tensor_path.stat().st_size,
    }
    write(OUT / f"{name}-readback.json", readback)
    require(tensor_path.stat().st_size <= MAX_CASE_BYTES, f"{name} tensor payload cap exceeded")
    require(phase_gate <= ATOL, f"{name} fixed-template/phase packet parity failed")
    require(cache_gate <= ATOL, f"{name} native cache K/V parity failed")
    require(summary["trace_identity_failures"] == [] and summary["max_trace_top2_abs_error"] <= ATOL,
            f"{name} original trace top-two parity failed")
    require(summary["final_full_vector_max_abs_vs_accepted_NN"] <= ATOL,
            f"{name} final accepted native vector parity failed")
    result = {
        "status": "candidate",
        "case": name,
        "query_count": n_queries,
        "row_count": n_rows,
        "raw_action_offsets": [x["raw_action_offset"] for x in queries],
        "physical_query_indices": [x["physical_query_index"] for x in queries],
        "trajectory": {
            "pre_rope_K": trajectory_metrics(pre_k[0], name=name),
            "V": trajectory_metrics(values[0], name=name),
            "all_layers": [{"layer": i, "pre_rope_K": trajectory_metrics(pre_k[i], name=name),
                            "V": trajectory_metrics(values[i], name=name)} for i in range(LAYERS)],
        },
        "readout": summary,
        "readback": binding(OUT / f"{name}-readback.json"),
        "tensor_artifact": binding(tensor_path),
    }
    write(OUT / f"{name}-result.json", result)
    return result, readback


def synthetic_cache_selfcheck() -> None:
    """Exercise full-length cache indexing, row flattening, and corruption detection."""
    class Layer:
        def __init__(self, keys: torch.Tensor, values: torch.Tensor):
            self.keys, self.values = keys, values

    class Cache:
        def __init__(self, layers: list[Layer], full_width: int):
            self.layers, self.full_width = layers, full_width

        def get_seq_length(self, _layer_index=None):
            return self.full_width

    for row_count, full_width, target, start in ((62, 2127, 2, 1563), (4, 1546, 1, 1506)):
        row_indices = torch.arange(start, start + row_count * 9, dtype=torch.long)
        pre_k = [torch.arange(row_count * KV_HEADS * 9 * HEAD_DIM, dtype=torch.float32)
                 .reshape(row_count, KV_HEADS, 9, HEAD_DIM) + 1.0 for _ in range(LAYERS)]
        values = [x + 100000.0 for x in pre_k]
        row_cos = torch.full((row_count, 9, HEAD_DIM), 0.8)
        row_sin = torch.full((row_count, 9, HEAD_DIM), 0.6)
        key_replay = rotate(pre_k[0].permute(1, 0, 2, 3).reshape(KV_HEADS, -1, HEAD_DIM),
                            row_cos.reshape(-1, HEAD_DIM), row_sin.reshape(-1, HEAD_DIM))
        value_replay = values[0].permute(1, 0, 2, 3).reshape(KV_HEADS, -1, HEAD_DIM)
        keys = torch.zeros((4, KV_HEADS, full_width, HEAD_DIM))
        cached_values = torch.zeros_like(keys)
        keys[target].index_copy_(1, row_indices, key_replay)
        cached_values[target].index_copy_(1, row_indices, value_replay)
        cache = Cache([Layer(keys, cached_values) for _ in range(LAYERS)], full_width)
        good = cache_attestations(pre_k, values, row_cos, row_sin, row_indices, cache, target, full_width)
        require(max(x["stored_postK_max_abs_vs_preK_phase"] for x in good) == 0.0 and
                max(x["stored_V_max_abs_vs_v_proj"] for x in good) == 0.0,
                f"synthetic full cache parity failed for {row_count} rows")
        cache.layers[0].values[target, 0, row_indices[0], 0] += 1.0
        corrupted = cache_attestations(pre_k, values, row_cos, row_sin, row_indices, cache, target, full_width)
        require(corrupted[0]["stored_V_max_abs_vs_v_proj"] > ATOL,
                f"synthetic cache corruption was not detected for {row_count} rows")


def cpu_selfcheck() -> None:
    selection = json.loads(SELECTION.read_text())
    oracle = json.loads(ORACLE.read_text())
    require(json.loads(FIXED_ACCEPTANCE.read_text())["status"] == "lead-accepted",
            "fixed-template lead acceptance receipt missing")
    for name in CASE_NAMES:
        fixed_artifact_bindings(name)
        meta = selection["cases"][name]
        raw = json.loads(Path(oracle["cases"][name]["source_bindings"]["raw"]["path"]).read_text())["rows"]
        trace = json.loads(Path(oracle["cases"][name]["source_bindings"]["trace"]["path"]).read_text())
        tokens = raw[int(oracle["cases"][name]["target"])] ["token_ids"]
        full_width = int(meta["full_width"])
        base = full_width - int(meta["raw_action_offset"])
        target = int(oracle["cases"][name]["target"])
        fake_ids = torch.zeros((4, full_width), dtype=torch.long)
        fake_ids[target, base:base + int(meta["raw_action_offset"])] = torch.tensor(tokens[: int(meta["raw_action_offset"])])
        fake = {"input_ids": fake_ids}
        queries = source_queries(name, {"raw_action_offset": meta["raw_action_offset"]}, oracle, raw, trace, fake)
        require(all(q["physical_query_index"] == fake["input_ids"].shape[1] - meta["raw_action_offset"] + q["raw_action_offset"] - 1 for q in queries),
                f"{name} CPU query index relation failed")
        require(sum(q["minus_one_action_token"] != q["next_token"] for q in queries) == len(queries),
                f"{name} one-token source shift did not fail")

    synthetic_cache_selfcheck()
    for row_count in (62, 4):
        fake_pre = [torch.arange(row_count * KV_HEADS * 9 * HEAD_DIM, dtype=torch.float32)
                         .reshape(row_count, KV_HEADS, 9, HEAD_DIM) + float(layer)
                    for layer in range(LAYERS)]
        fake_packet = {"layers": [{"pre_old": x[-2].clone(), "post_old": x[-2].clone(),
                                    "pre_new": x[-1].clone(), "post_new": x[-1].clone()} for x in fake_pre]}
        fake_template = {"first_preK": [x[0].clone() for x in fake_pre]}
        metrics, _ = phase_metrics(fake_pre, torch.ones(row_count, 9, HEAD_DIM),
                                   torch.zeros(row_count, 9, HEAD_DIM), fake_packet, fake_template)
        require(len(metrics) == LAYERS and metrics[0]["layer"] == 0,
                f"phase postprocessing changed layer/row axes for {row_count} rows")
        require(not torch.equal(fake_pre[0][0], fake_pre[0][-2]) and
                metrics[0]["first_preK_max_abs_vs_fixed_template"] == 0.0 and
                metrics[0]["penultimate_preK_max_abs_vs_phase_packet_pre_old"] == 0.0 and
                metrics[0]["penultimate_postK_replay_max_abs_vs_phase_packet_post_old"] == 0.0 and
                float((fake_pre[0][0] - fake_packet["layers"][0]["pre_old"]).abs().max()) > 1.0,
                f"phase synthetic fixture did not distinguish first and penultimate rows for {row_count}")
        require(trajectory_metrics(fake_pre[0], name="val")["rows"] == row_count,
                f"trajectory postprocessing changed row axis for {row_count} rows")

    text = Qwen3VLTextConfig(vocab_size=17, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=4,
                             max_position_embeddings=32,
                             rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    vision = Qwen3VLVisionConfig(depth=1, hidden_size=4, intermediate_size=8, num_heads=1,
                                 out_hidden_size=8, num_position_embeddings=64)
    model = modeling_qwen3_vl.Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    keep = torch.tensor([1, 4])
    captured = {}
    handles = []
    text_model = model.model.language_model
    def capture_phase(_module, _args, output):
        captured["phase"] = tuple(x.detach().clone() for x in output)
    def capture_k(_module, _args, output):
        captured["k"] = output.detach().clone()
    def capture_v(_module, _args, output):
        captured["v"] = output.detach().clone()
    handles.extend((text_model.rotary_emb.register_forward_hook(capture_phase),
                    text_model.layers[0].self_attn.k_norm.register_forward_hook(capture_k),
                    text_model.layers[0].self_attn.v_proj.register_forward_hook(capture_v)))
    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions, logits_to_keep=0).logits
        selected = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions, logits_to_keep=keep).logits
        shifted = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions, logits_to_keep=keep - 1).logits
        cache = DynamicCache()
        cached = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=positions,
                       cache_position=torch.arange(ids.shape[1]), past_key_values=cache,
                       use_cache=True, logits_to_keep=keep)
    for handle in handles:
        handle.remove()
    require(torch.equal(selected, full[:, keep]) and not torch.equal(shifted, selected),
            "installed logits_to_keep tensor indexing or one-token shift failed")
    replay = rotate(captured["k"][0].permute(1, 0, 2), captured["phase"][0][0], captured["phase"][1][0])
    actual_v = captured["v"][0].reshape(ids.shape[1], 2, 4).permute(1, 0, 2)
    require(cached.past_key_values is cache and cache.get_seq_length() == ids.shape[1] and
            float((cache.layers[0].keys[0].float() - replay).abs().max()) <= 1e-6 and
            torch.equal(cache.layers[0].values[0].float(), actual_v.float()),
            "installed DynamicCache did not retain the native post-K/V consumer values")
    print(json.dumps({"status": "selfcheck_ok", "transformers": transformers.__version__, "cases": {name: len(json.loads(ORACLE.read_text())["cases"][name]["queries"]) for name in CASE_NAMES}}))


def run(device: str) -> None:
    cpu_selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA device required")
    require(not OUT.exists(), "attempt path already exists; preserve previous attempt")
    oracle = json.loads(ORACLE.read_text())
    selection = json.loads(SELECTION.read_text())
    require(selection["status"] == "root-verified-frozen" and oracle["status"] == "root-independent-source-oracle",
            "frozen selection/oracle status changed")
    require(json.loads(FIXED_ACCEPTANCE.read_text())["status"] == "lead-accepted", "fixed-template acceptance missing")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0,
             "vision_forwards": 0, "calls": [], "started_unix": time.time(), "started": time.monotonic(),
             "source": {"unit": binding(UNIT), "selection": binding(SELECTION), "oracle": binding(ORACLE),
                        "fixed_acceptance": binding(FIXED_ACCEPTANCE),
                        "execution_ruling": binding(EXECUTION_RULING)}}
    write(OUT / "receipt.json", state)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        cases = {}
        for name in CASE_NAMES:
            case, meta, queries, raw, trace = source_inputs(name, q, identity, oracle, device)
            case["meta"], case["queries"], case["raw"], case["trace"] = meta, queries, raw, trace
            cases[name] = case
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_native_trajectory-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ("probes/training_set_completion/recurrence_attention_mass.py",
                        "probes/training_set_completion/recurrence_fixed_template.py",
                        "probes/training_set_completion/recurrence_position_history.py",
                        "probes/training_set_completion/untied_shared.py",
                        "probes/training_set_completion/numerical_feedback/runtime.py",
                        "src/qwen/native.py", "src/qwen/input_identity.py", "src/inference/bound_requests.py",
                        "src/qwen/untied_embeddings.py", inspect.getfile(modeling_qwen3_vl))]
        captures = [preserve_source(path, run_root=OUT, relative_name=str(path) if not path.is_absolute() else f"transformers/{path.name}") for path in dependencies]
        manifest = {"schema": "recurrence_native_trajectory.v1", "status": "frozen_before_forward",
                    "producer": binding(Path(__file__)), "producer_capture": binding(producer),
                    "dependency_captures": [binding(path) for path in captures], "transformers_version": transformers.__version__,
                    "model_identity": identity, "selection": binding(SELECTION), "oracle": binding(ORACLE),
                    "execution_ruling": binding(EXECUTION_RULING),
                    "cases": {name: {"target_batch": case["target"], "raw_action_offset": case["meta"]["raw_action_offset"],
                                     "query_count": len(case["queries"]), "rows": [list(x) for x in row_geometry(name, selection, case["native"])[0]],
                                     "query_metadata": case["queries"],
                                     "native_input_hashes": {key: tensor_hash(case["native"][key]) for key in ("input_ids", "attention_mask", "position_ids")},
                                     "prepare_case_sources": case["source_bindings"], "accepted_paths": case["accepted_paths"],
                                     "fixed_artifacts": case["fixed_bindings"]} for name, case in cases.items()},
                    "collection": {"one_full_forward_per_case": True, "no_generation": True, "no_intervention": True,
                                   "selected_logits": "declared physical query indices via tensor logits_to_keep", "model_forward_cap": MAX_MODEL_FORWARDS,
                                   "vision_forward_cap": MAX_VISION_FORWARDS, "parity_atol": ATOL, "case_payload_cap_bytes": MAX_CASE_BYTES}}
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        def count_model(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_MODEL_FORWARDS and time.monotonic() - state["started"] <= MAX_SECONDS,
                    "model forward/time cap exceeded")
        def count_vision(*_):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= MAX_VISION_FORWARDS, "vision forward cap exceeded")
        counter_handles = [q.model.register_forward_pre_hook(count_model, with_kwargs=True), q.model.model.visual.register_forward_pre_hook(count_vision)]
        try:
            results = {}
            for name in CASE_NAMES:
                results[name], _ = collect_case(name, cases[name], cases[name]["queries"], q, device, state)
                state["last_case"] = name
                state["last_result"] = binding(OUT / f"{name}-result.json")
                write(OUT / "receipt.json", state)
            require(state["model_forwards"] == len(CASE_NAMES) and state["vision_forwards"] == len(CASE_NAMES),
                    "one native forward/one vision pass per selected case required")
        finally:
            for handle in counter_handles:
                handle.remove()
        result = {"schema": "recurrence_native_trajectory.result.v1", "status": "candidate",
                  "cases": {name: binding(OUT / f"{name}-result.json") for name in CASE_NAMES},
                  "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                  "source_manifest": binding(OUT / "source-to-cell.json"), "elapsed_seconds": time.monotonic() - state["started"]}
        write(OUT / "result.json", result)
        state.update(status="candidate_complete", result=binding(OUT / "result.json"), elapsed_seconds=result["elapsed_seconds"],
                     peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        write(OUT / "receipt.json", state)
        print(json.dumps({"status": state["status"], "model_forwards": state["model_forwards"],
                          "vision_forwards": state["vision_forwards"], "result": str(OUT / "result.json")}))
    except BaseException as error:
        state.update(status="technical_invalid", error=repr(error), elapsed_seconds=time.monotonic() - state["started"])
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
