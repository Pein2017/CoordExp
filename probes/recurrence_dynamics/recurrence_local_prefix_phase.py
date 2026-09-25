"""Target-query-only phase coherence at the val row42 recipient."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.recurrence_dynamics import recurrence_downstream_query_phase as phase
from probes.recurrence_dynamics import recurrence_native_trajectory as trajectory
from src.artifacts.utf8_json import literal_binding
from probes.recurrence_dynamics.recurrence_donor_tracking import require, write
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-22-recurrence-local-prefix-phase"
OUT = ROOT / "attempt-002"
UNIT = Path("research/experiments/2026-09-22-recurrence-local-prefix-phase/unit.md")
SELECTION = ROOT / "selection.json"
PREDECESSOR_LOGITS = BASE / "2026-09-22-recurrence-downstream-query-phase/attempt-002/held_late_phase.pt"
PREDECESSOR_SELECTION = BASE / "2026-09-22-recurrence-downstream-query-phase/selection.json"
PREDECESSOR_ACCEPTANCE = BASE / "2026-09-22-recurrence-downstream-query-phase/lead-acceptance.json"

TARGET = 2
FULL_WIDTH = 2127
RECIPIENT_POSITION = 1703
DONOR_POSITION = 2126
PREFIX_START, PREFIX_END = 1698, 1703
DONOR_PREFIX_START, DONOR_PREFIX_END = 2121, 2126
VOCAB = 152670
Q_HEADS, KV_HEADS, HEAD_DIM, HIDDEN, LAYERS = 16, 8, 128, 2048, 28
Q_TO_KV = Q_HEADS // KV_HEADS
P38, P999, P579 = 151708, 152669, 152249
CELL_ORDER = ("query_only", "identity_recompute", "prefix_coherent")
PHASE_LAYERS = tuple(range(1, LAYERS))
MAX_MODEL_FORWARDS = 3
MAX_VISION_FORWARDS = 3
MAX_SECONDS = 10 * 60
MAX_TENSOR_BYTES = 48 << 20
ATOL = 2e-4


def binding(path: Path) -> dict:
    return literal_binding(path)


def assert_binding(path: Path, expected: dict) -> None:
    actual = binding(path)
    require(all(actual[key] == expected[key] for key in ("path", "sha256", "size_bytes")),
            f"source binding drift: {path}")


def rotate_rows_fp32(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    require(value.ndim == 3 and cos.shape == sin.shape == value.shape[1:],
            "prefix rotation shape changed")
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cos.unsqueeze(0) + rotated * sin.unsqueeze(0)


def rotate_rows_independent(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    require(value.ndim == 3 and cos.shape == sin.shape == value.shape[1:],
            "independent prefix rotation shape changed")
    half = value.shape[-1] // 2
    a, b = value[..., :half].double(), value[..., half:].double()
    c, s = cos[..., :half].double(), sin[..., :half].double()
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1)


def _hash(tensor: torch.Tensor) -> str:
    return tensor_hash(tensor)


def selected_sdpa_replacement(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    native_sdpa,
    scale: float,
    mode: str,
    capture: dict | None,
    target_batch: int = TARGET,
    prefix_start: int = PREFIX_START,
    prefix_end: int = PREFIX_END,
    output_position: int = RECIPIENT_POSITION,
    inject_global_key_mutation: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Run the full SDPA and replace one target output from a one-row SDPA."""
    batch, q_heads, full_width, head_dim = query.shape
    require(query.ndim == 4 and key.shape == value.shape == (batch, q_heads, full_width, head_dim),
            "post-GQA full SDPA shape changed")
    require(0 <= target_batch < batch and 0 <= prefix_start < prefix_end <= full_width and
            0 <= output_position < full_width,
            "selected SDPA indices changed")
    require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and
            mask.shape == (batch, 1, full_width, full_width),
            "full SDPA mask shape changed")
    require(float(scale) == head_dim ** -0.5, "SDPA scale changed")
    before_q = _hash(query)
    before_k = _hash(key)
    before_v = _hash(value)
    before_mask = _hash(mask)

    full_key = key
    if inject_global_key_mutation:
        full_key = key.clone()
        full_key[target_batch, :, prefix_start, 0] += 0.75
    full_key_hash = _hash(full_key)
    full_output = native_sdpa(query, full_key, value, attn_mask=mask,
                              dropout_p=0.0, is_causal=False, scale=scale)
    require(full_key_hash == before_k, "global full-call K mutation reached SDPA")
    require(_hash(query) == before_q and _hash(key) == before_k and
            _hash(value) == before_v and _hash(mask) == before_mask,
            "full SDPA input mutated across execution")

    record = {
        "full_input_q_hash_before": before_q,
        "full_input_k_hash_before": before_k,
        "full_input_v_hash_before": before_v,
        "full_input_mask_hash": before_mask,
        "full_call_k_hash": full_key_hash,
        "full_input_q_hash_after": _hash(query),
        "full_input_k_hash_after": _hash(key),
        "full_input_v_hash_after": _hash(value),
        "full_input_unchanged": True,
        "full_output_target": full_output[target_batch, :, output_position].detach().float().cpu().clone(),
        "selected_output_target": None,
        "selected_q_target": None,
        "selected_k_prefix_before": None,
        "selected_k_prefix_used": None,
        "selected_v_prefix_hash": None,
        "selected_mask": None,
        "selected_mask_hash": None,
        "selected_call": False,
        "off_target_output_exact": True,
        "mode": mode,
    }
    if mode == "none":
        return full_output, record

    require(mode in ("identity", "prefix_coherent") and capture is not None,
            "selected SDPA mode/capture changed")
    q_selected = query[target_batch:target_batch + 1, :, output_position:output_position + 1, :]
    k_selected = full_key[target_batch:target_batch + 1].clone()
    v_selected = value[target_batch:target_batch + 1]
    mask_selected = mask[target_batch:target_batch + 1, :, output_position:output_position + 1, :]
    record["selected_q_target"] = q_selected[0, :, 0].detach().float().cpu().clone()
    record["selected_k_prefix_before"] = k_selected[0, :, prefix_start:prefix_end].detach().float().cpu().clone()
    record["selected_v_prefix_hash"] = _hash(v_selected[:, :, prefix_start:prefix_end, :])
    record["selected_mask"] = mask_selected.detach().cpu().clone()
    record["selected_mask_hash"] = _hash(mask_selected)

    if mode == "prefix_coherent":
        pre_k = capture["pre_k_device"].to(device=key.device, dtype=key.dtype)
        late_cos = capture["late_cos_device"].to(device=key.device, dtype=key.dtype)
        late_sin = capture["late_sin_device"].to(device=key.device, dtype=key.dtype)
        require(pre_k.shape == (key.shape[1] // Q_TO_KV, prefix_end - prefix_start, head_dim) and
                late_cos.shape == late_sin.shape == (prefix_end - prefix_start, head_dim),
                "live prefix K/phase shape changed")
        used_pre_k = rotate_rows_fp32(pre_k, late_cos, late_sin)
        used_post_k = used_pre_k.repeat_interleave(Q_TO_KV, dim=0)
        k_selected[0, :, prefix_start:prefix_end] = used_post_k
    else:
        used_post_k = k_selected[0, :, prefix_start:prefix_end].detach().clone()
    record["selected_k_prefix_used"] = used_post_k.detach().float().cpu().clone()

    selected_output = native_sdpa(q_selected, k_selected, v_selected,
                                   attn_mask=mask_selected, dropout_p=0.0,
                                   is_causal=False, scale=scale)
    restored_k = k_selected.clone()
    restored_k[:, :, prefix_start:prefix_end] = key[target_batch:target_batch + 1, :, prefix_start:prefix_end]
    require(torch.equal(restored_k, key[target_batch:target_batch + 1]) and
            torch.equal(v_selected, value[target_batch:target_batch + 1]) and
            torch.equal(mask_selected, mask[target_batch:target_batch + 1, :, output_position:output_position + 1]),
            "selected call changed non-prefix K, V or mask")
    require(_hash(query) == before_q and _hash(key) == before_k and
            _hash(value) == before_v and _hash(mask) == before_mask,
            "full SDPA input mutated across selected execution")
    require(selected_output.shape == (1, q_heads, 1, head_dim),
            "selected SDPA output shape changed")
    replaced = full_output.clone()
    replaced[target_batch, :, output_position, :] = selected_output[0, :, 0, :]
    require(torch.equal(replaced[:target_batch], full_output[:target_batch]) and
            torch.equal(replaced[target_batch, :, :output_position], full_output[target_batch, :, :output_position]) and
            torch.equal(replaced[target_batch, :, output_position + 1:], full_output[target_batch, :, output_position + 1:]) and
            torch.equal(replaced[target_batch + 1:], full_output[target_batch + 1:]),
            "selected replacement changed an off-target full output")
    record.update(
        selected_output_target=selected_output[0, :, 0].detach().float().cpu().clone(),
        selected_call=True,
        off_target_output_exact=True,
        selected_nonprefix_k_exact=True, selected_v_exact=True,
        selected_mask_exact=True, full_inputs_unchanged_after_selected=True,
    )
    return replaced, record


def _manual_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      mask: torch.Tensor, scale: float) -> torch.Tensor:
    scores = torch.matmul(query.double(), key.double().transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1).matmul(value.double()).float()


def cpu_local_selfcheck() -> None:
    """Exercise the exact post-GQA replacement plus an actual Linear consumer."""
    batch, q_heads, kv_heads, seq, dim = 3, 4, 2, 7, 6
    target, position = 2, 5
    scale = dim ** -0.5
    q_pre = torch.arange(batch * q_heads * seq * dim, dtype=torch.float32).reshape(
        batch, q_heads, seq, dim) / 31.0
    k_pre = torch.arange(batch * kv_heads * seq * dim, dtype=torch.float32).reshape(
        batch, kv_heads, seq, dim) / 23.0
    v_pre = (torch.arange(batch * kv_heads * seq * dim, dtype=torch.float32).reshape(
        batch, kv_heads, seq, dim) + 0.37) / 19.0
    mask = torch.tril(torch.ones(batch, 1, seq, seq, dtype=torch.bool))
    pre_prefix = torch.arange(kv_heads * 2 * dim, dtype=torch.float32).reshape(kv_heads, 2, dim) / 17.0 + 0.19
    angles = torch.tensor([[0.13, -0.27, 0.41], [0.23, -0.37, 0.51]])
    late_cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    late_sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    independent = rotate_rows_independent(pre_prefix, late_cos, late_sin).float()
    expected_used = independent.repeat_interleave(q_heads // kv_heads, dim=0)
    native_f = F.scaled_dot_product_attention
    module = torch.nn.Module()
    module.num_key_value_groups = q_heads // kv_heads
    module.config = type("Config", (), {"_attn_implementation": "sdpa"})()

    def run_case(mode: str, *, wrong_query: bool = False, global_bad: bool = False):
        records = []
        output_position = position - 1 if wrong_query else position
        capture = {"pre_k_device": pre_prefix.clone(), "late_cos_device": late_cos.clone(),
                   "late_sin_device": late_sin.clone()}

        def outer(query, key, value, *, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
            require(not enable_gqa and is_causal is False, "CPU selected route changed SDPA flags")
            observed, record = selected_sdpa_replacement(
                query, key, value, attn_mask, native_sdpa=native_f,
                scale=float(scale), mode=mode, capture=capture,
                target_batch=target, prefix_start=3, prefix_end=5,
                output_position=output_position,
                inject_global_key_mutation=global_bad)
            records.append(record)
            return observed

        try:
            F.scaled_dot_product_attention = outer
            observed, _ = phase.sdpa_attention_forward(
                module, q_pre, k_pre, v_pre, mask, scaling=scale)
        finally:
            F.scaled_dot_product_attention = native_f
        require(len(records) == 1, "CPU actual F.sdpa observer call count changed")
        consumed = observed.reshape(batch, seq, q_heads * dim)
        expected_native = native_f(
            q_pre, k_pre.repeat_interleave(2, dim=1), v_pre.repeat_interleave(2, dim=1),
            attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale)
        expected_target = expected_native[target, :, position].detach().clone()
        if mode == "prefix_coherent" and not wrong_query and not global_bad:
            selected_q = q_pre[target:target + 1, :, position:position + 1, :]
            selected_mask = mask[target:target + 1, :, position:position + 1, :]
            selected_k = k_pre[target:target + 1].repeat_interleave(2, dim=1).clone()
            selected_k[:, :, 3:5] = expected_used
            selected_v = v_pre[target:target + 1].repeat_interleave(2, dim=1)
            expected_target = _manual_attention(selected_q, selected_k, selected_v,
                                                selected_mask, scale)[0, :, 0]
        if mode == "identity" and not global_bad and not wrong_query:
            require(float((consumed[target, position] - expected_target.reshape(-1)).abs().max()) <= ATOL and
                    torch.equal(consumed[target, :position], expected_native.transpose(1, 2).reshape(batch, seq, -1)[target, :position]),
                    "identity selected replacement changed native consumer")
        return consumed, expected_target, records[0]

    # Identity recomputation must reach the real Linear consumer exactly.
    consumed, expected_target, _ = run_case("identity")
    linear = torch.nn.Linear(q_heads * dim, 5, bias=False)
    expected_input = consumed.clone()
    expected_input[target, position] = expected_target.reshape(-1)
    seen = {}

    def consumer(_module, args):
        value = args[0]
        require(float((value[target, position] - expected_input[target, position]).abs().max()) <= ATOL and
                torch.equal(value[:target], expected_input[:target]) and
                torch.equal(value[target, :position], expected_input[target, :position]) and
                torch.equal(value[target, position + 1:], expected_input[target, position + 1:]) and
                torch.equal(value[target + 1:], expected_input[target + 1:]),
                "selected output did not reach the actual consumer")
        seen["ok"] = True

    handle = linear.register_forward_pre_hook(consumer)
    try:
        linear(consumed)
    finally:
        handle.remove()
    require(seen.get("ok", False), "CPU consumer observer did not run")

    # Prefix treatment must match an independent FP64 attention calculation.
    prefix_consumed, prefix_target, prefix_record = run_case("prefix_coherent")
    require(float((prefix_record["selected_k_prefix_used"] - expected_used).abs().max()) <= ATOL,
            "production prefix rotation differs from independent rotation")
    require(float((prefix_consumed[target, position] - prefix_target.reshape(-1)).abs().max()) <= ATOL,
            "production selected attention differs from independent attention")

    # A wrong output query must fail at the same consumer boundary.
    wrong_consumed, _, _ = run_case("prefix_coherent", wrong_query=True)
    wrong_expected = prefix_consumed.clone()
    wrong_expected[target, position] = prefix_target.reshape(-1)
    wrong_handle = linear.register_forward_pre_hook(
        lambda _module, args: require(torch.equal(args[0], wrong_expected),
                                      "wrong query replacement escaped the consumer"))
    try:
        try:
            linear(wrong_consumed)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong query replacement escaped CPU consumer")
    finally:
        wrong_handle.remove()

    # A deliberately global K mutation must also fail the actual consumer.
    try:
        run_case("identity", global_bad=True)
    except ValueError as error:
        require(str(error) == "global full-call K mutation reached SDPA",
                "global mutation failed before the shared actual-consumer check")
    else:
        raise AssertionError("global key mutation escaped shared CPU/production consumer")


def cpu_selfcheck() -> dict:
    sources = phase.cpu_selfcheck()
    cpu_local_selfcheck()
    print(json.dumps({"status": "selfcheck_ok", "cell_order": list(CELL_ORDER),
                      "prefix": [PREFIX_START, PREFIX_END],
                      "donor_prefix": [DONOR_PREFIX_START, DONOR_PREFIX_END]}))
    return sources


def load_local_sources() -> tuple[dict, torch.Tensor, dict]:
    selection = json.loads(SELECTION.read_text())
    require(selection["status"] == "root-frozen" and selection["target"] == TARGET and
            selection["recipient_query"] == RECIPIENT_POSITION and
            selection["donor_query"] == DONOR_POSITION and
            selection["recipient_prefix"] == [PREFIX_START, PREFIX_END] and
            selection["donor_prefix"] == [DONOR_PREFIX_START, DONOR_PREFIX_END] and
            selection["cell_order"] == list(CELL_ORDER), "local-prefix selection changed")
    for key in ("predecessor_selection", "predecessor_acceptance", "predecessor_logits", "protocol"):
        expected = selection[key]
        assert_binding(Path(expected["path"]), expected)
    predecessor_acceptance = json.loads(PREDECESSOR_ACCEPTANCE.read_text())
    require(predecessor_acceptance["status"] == "lead-accepted", "predecessor acceptance changed")
    predecessor_logits = torch.load(PREDECESSOR_LOGITS, map_location="cpu", weights_only=True)
    require(predecessor_logits["full_logits"].shape == (4, 2, VOCAB),
            "predecessor selected logits shape changed")
    return selection, predecessor_logits["full_logits"].float().clone(), {
        "local_unit": binding(UNIT), "local_selection": binding(SELECTION),
        "predecessor_selection": binding(PREDECESSOR_SELECTION),
        "predecessor_acceptance": binding(PREDECESSOR_ACCEPTANCE),
        "predecessor_logits": binding(PREDECESSOR_LOGITS),
    }


def capture_prefix_hooks(text: torch.nn.Module, capture: dict) -> list:
    handles = []
    for index in PHASE_LAYERS:
        layer = text.layers[index]
        capture[index] = {
            "pre_k_device": None, "pre_k": None,
            "early_cos_device": None, "early_sin_device": None,
            "late_cos_device": None, "late_sin_device": None,
            "early_cos": None, "early_sin": None,
            "late_cos": None, "late_sin": None,
        }

        def phase_hook(_module, _args, kwargs, *, index=index):
            rotary = kwargs.get("position_embeddings")
            require(isinstance(rotary, tuple) and len(rotary) == 2 and
                    rotary[0].shape == rotary[1].shape == (4, FULL_WIDTH, HEAD_DIM),
                    "live full-position phase shape changed")
            item = capture[index]
            item["early_cos_device"] = rotary[0][TARGET, PREFIX_START:PREFIX_END].detach().clone()
            item["early_sin_device"] = rotary[1][TARGET, PREFIX_START:PREFIX_END].detach().clone()
            item["late_cos_device"] = rotary[0][TARGET, DONOR_PREFIX_START:DONOR_PREFIX_END].detach().clone()
            item["late_sin_device"] = rotary[1][TARGET, DONOR_PREFIX_START:DONOR_PREFIX_END].detach().clone()
            item["early_cos"] = item["early_cos_device"].float().cpu().clone()
            item["early_sin"] = item["early_sin_device"].float().cpu().clone()
            item["late_cos"] = item["late_cos_device"].float().cpu().clone()
            item["late_sin"] = item["late_sin_device"].float().cpu().clone()

        def k_hook(_module, _args, output, *, index=index):
            require(output.shape == (4, FULL_WIDTH, KV_HEADS, HEAD_DIM),
                    "live prefix pre-K shape changed")
            item = capture[index]
            item["pre_k_device"] = output[TARGET, PREFIX_START:PREFIX_END].permute(1, 0, 2).detach().clone()
            item["pre_k"] = item["pre_k_device"].float().cpu().clone()

        handles.append(layer.self_attn.register_forward_pre_hook(phase_hook, with_kwargs=True))
        handles.append(layer.self_attn.k_norm.register_forward_hook(k_hook))
    return handles


def raw_payload(capture: dict, outer_records: dict, label: str) -> dict:
    layer_payload = {}
    for index in PHASE_LAYERS:
        item = capture.get(index, {})
        record = outer_records.get(index, {})
        layer_payload[str(index)] = {
            key: value for key, value in {
                "pre_K": item.get("pre_k"),
                "early_cos": item.get("early_cos"), "early_sin": item.get("early_sin"),
                "late_cos": item.get("late_cos"), "late_sin": item.get("late_sin"),
                "full_output_target": record.get("full_output_target"),
                "selected_output_target": record.get("selected_output_target"),
                "selected_q_target": record.get("selected_q_target"),
                "selected_k_prefix_before": record.get("selected_k_prefix_before"),
                "selected_k_prefix_used": record.get("selected_k_prefix_used"),
                "selected_mask": record.get("selected_mask"),
            }.items() if value is not None
        }
    return {"schema": "recurrence_local_prefix_phase.raw.v1", "cell": label,
            "layers": layer_payload}


def install_outer(mode: str, capture: dict) -> tuple[object, dict]:
    original_sdpa = F.scaled_dot_product_attention
    records: dict[int, dict] = {}
    full_index = 0

    def outer(query, key, value, *, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
        nonlocal full_index
        text_call = (query.ndim == 4 and query.shape == (4, Q_HEADS, FULL_WIDTH, HEAD_DIM) and
                     key.shape == value.shape == (4, Q_HEADS, FULL_WIDTH, HEAD_DIM))
        if not text_call:
            return original_sdpa(query, key, value, attn_mask=attn_mask,
                                 dropout_p=dropout_p, is_causal=is_causal,
                                 scale=scale, enable_gqa=enable_gqa)
        index = full_index
        full_index += 1
        require(index < LAYERS and is_causal is False and enable_gqa is False,
                "text SDPA call ordering/flags changed")
        effective_mode = mode if index in PHASE_LAYERS else "none"
        output, record = selected_sdpa_replacement(
            query, key, value, attn_mask, native_sdpa=original_sdpa,
            scale=float(scale), mode=effective_mode,
            capture=capture.get(index), output_position=RECIPIENT_POSITION)
        require(record["full_call_k_hash"] == record["full_input_k_hash_before"],
                "full attention K changed before selected-query replacement")
        require(record["full_input_unchanged"], "full attention input identity changed")
        records[index] = record
        return output

    F.scaled_dot_product_attention = outer
    return original_sdpa, records


def _cpu_tensor_or_none(value):
    return value.detach().float().cpu().clone() if isinstance(value, torch.Tensor) else value


def enrich_tensor(tensor_path: Path, capture: dict, outer_records: dict, label: str) -> None:
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    local = raw_payload(capture, outer_records, label)
    for key in ("pre_K", "early_cos", "early_sin", "late_cos", "late_sin"):
        values = [local["layers"][str(i)].get(key) for i in PHASE_LAYERS]
        if all(value is not None for value in values):
            local[key] = torch.stack(values)
    for key in ("full_output_target", "selected_output_target", "selected_q_target",
                "selected_k_prefix_before", "selected_k_prefix_used", "selected_mask"):
        values = [local["layers"][str(i)].get(key) for i in PHASE_LAYERS]
        if all(value is not None for value in values):
            local[key] = torch.stack(values)
    payload.update({"local_prefix": local})
    torch.save(payload, tensor_path)


def local_record(label: str, result: dict, outer_records: dict, capture: dict) -> dict:
    record = dict(result["record"])
    record["tensor_artifact"] = binding(result["tensor_path"])
    record["cell"] = label
    record["base_cell"] = "held_late_phase"
    record["text_full_sdpa_calls"] = len(outer_records)
    record["selected_sdpa_calls"] = sum(bool(item["selected_call"]) for item in outer_records.values())
    record["local_prefix"] = {
        "mode": "none" if label == "query_only" else
                ("identity" if label == "identity_recompute" else "prefix_coherent"),
        "layers": {
            str(index): {
                "full_call_k_hash": outer_records[index]["full_call_k_hash"],
                "full_input_k_hash_before": outer_records[index]["full_input_k_hash_before"],
                "full_input_k_hash_after": outer_records[index]["full_input_k_hash_after"],
                "full_input_unchanged": outer_records[index]["full_input_unchanged"],
                "selected_call": outer_records[index]["selected_call"],
                "selected_mask_hash": outer_records[index]["selected_mask_hash"],
                "off_target_output_exact": outer_records[index]["off_target_output_exact"],
                "selected_v_prefix_hash": outer_records[index]["selected_v_prefix_hash"],
                **{key: outer_records[index].get(key) for key in
                   ("selected_nonprefix_k_exact", "selected_v_exact", "selected_mask_exact",
                    "full_inputs_unchanged_after_selected")},
                "prefix_phase_shapes": [list(capture[index]["early_cos"].shape),
                                         list(capture[index]["late_cos"].shape)],
            } for index in PHASE_LAYERS
        },
    }
    return record


def run_cell(label: str, sources: dict, q, identity: dict, device: str,
             state: dict, native_logits: torch.Tensor | None) -> tuple[dict, torch.Tensor]:
    cell_dir = OUT / label
    cell_dir.mkdir(parents=True)
    previous_out = phase.OUT
    phase.OUT = cell_dir
    text = q.model.model.language_model
    capture: dict = {}
    handles = capture_prefix_hooks(text, capture)
    mode = "none" if label == "query_only" else ("identity" if label == "identity_recompute" else "prefix_coherent")
    original_sdpa, outer_records = install_outer(mode, capture)
    original_torch_save = torch.save
    raw_saved = False
    tensor_path = cell_dir / "held_late_phase.pt"

    def save_proxy(obj, path, *args, **kwargs):
        nonlocal raw_saved
        original_torch_save(obj, path, *args, **kwargs)
        if not raw_saved and Path(path) == tensor_path:
            original_torch_save(raw_payload(capture, outer_records, label), cell_dir / "raw-prefix.pt")
            raw_saved = True

    started = time.monotonic()
    try:
        torch.save = save_proxy
        result, logits = phase.collect_cell("held_late_phase", sources, q, identity,
                                           device, state, native_logits)
    finally:
        torch.save = original_torch_save
        F.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
        phase.OUT = previous_out
    if not raw_saved:
        original_torch_save(raw_payload(capture, outer_records, label), cell_dir / "raw-prefix-fallback.pt")
    require(len(outer_records) == LAYERS, "text full SDPA layer count changed")
    require(all(capture[index]["pre_k"] is not None and capture[index]["late_cos"] is not None
                for index in PHASE_LAYERS), "live prefix captures incomplete")
    enrich_tensor(tensor_path, capture, outer_records, label)
    require(sum(path.stat().st_size for path in OUT.rglob("*.pt")) <= MAX_TENSOR_BYTES,
            "local tensor payload cap exceeded")
    record = local_record(label, result, outer_records, capture)
    record["elapsed_seconds"] = time.monotonic() - started
    return {"record": record, "result": result, "logits": logits,
            "tensor_path": tensor_path, "capture": capture,
            "outer_records": outer_records}, logits


def run(device: str) -> None:
    sources = cpu_selfcheck()
    selection, predecessor_logits, local_refs = load_local_sources()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA device required")
    require(not OUT.exists(), "attempt path already exists; preserve previous attempts")
    OUT.mkdir(parents=True)
    state = {"schema": "recurrence_local_prefix_phase.receipt.v1", "status": "preparing",
             "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "cell_order": list(CELL_ORDER), "completed_cells": [], "calls": [],
             "started_unix": time.time(), "started": time.monotonic()}
    write(OUT / "receipt.json", state)
    cell_readbacks = {}
    native_logits = None
    original_phase_out = phase.OUT
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = phase.load_model("untied", torch.device(device))
        oracle = json.loads(trajectory.ORACLE.read_text())
        case, _meta, _queries, _raw, _trace = trajectory.source_inputs("val", q, identity, oracle, device)
        native = case["native"]
        require(native["input_ids"].shape == (4, FULL_WIDTH), "native input shape changed")
        require(native["input_ids"][TARGET, PREFIX_START:PREFIX_END].tolist() == selection["prefix_token_ids"] and
                native["input_ids"][TARGET, DONOR_PREFIX_START:DONOR_PREFIX_END].tolist() == selection["prefix_token_ids"],
                "recipient/donor prefix token ids changed")
        native_hashes = {key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")}
        require(native_hashes == sources["selection"]["native_input_hashes"], "native input hashes changed")

        producer_capture = preserve_source(Path(__file__), run_root=OUT,
                                            relative_name="recurrence_local_prefix_phase.py")
        unit_capture = preserve_source(UNIT, run_root=OUT, relative_name="experiment/unit.md")
        dependency_paths = [
            Path('probes/recurrence_dynamics/recurrence_downstream_query_phase.py'),
            Path('probes/recurrence_dynamics/recurrence_native_trajectory.py'),
            Path('probes/recurrence_dynamics/recurrence_first_layer_readout.py'),
            Path("probes/model_profiles/mature_tied_untied.py"),
            Path("src/qwen/input_identity.py"), Path("src/inference/bound_requests.py"),
            Path(inspect.getfile(modeling_qwen3_vl)),
            Path(inspect.getfile(phase.sdpa_attention_forward)),
        ]
        dependency_captures = [preserve_source(path, run_root=OUT,
                               relative_name=str(path) if not path.is_absolute() else f"transformers/{path.name}")
                               for path in dependency_paths]
        manifest = {
            "schema": "recurrence_local_prefix_phase.v1", "status": "frozen_before_forward",
            "source": {**local_refs, "phase_sources": sources["refs"]},
            "producer": binding(Path(__file__)), "producer_capture": binding(producer_capture),
            "repair_ruling": binding(UNIT.parent / "repair-ruling-01.md"),
            "unit_capture": binding(unit_capture),
            "dependency_captures": [binding(path) for path in dependency_captures],
            "transformers_version": transformers.__version__, "model_identity": identity,
            "native_input_hashes": native_hashes,
            "native_prefix": {"recipient": [PREFIX_START, PREFIX_END],
                              "donor": [DONOR_PREFIX_START, DONOR_PREFIX_END],
                              "token_ids": selection["prefix_token_ids"]},
            "collection": {
                "cell_order": list(CELL_ORDER), "base_cell": "held_late_phase",
                "target": TARGET, "recipient_query": RECIPIENT_POSITION,
                "donor_query": DONOR_POSITION, "phase_layers": list(PHASE_LAYERS),
                "full_sdpa": "[4,16,2127,128] bool mask [4,1,2127,2127] scale 1/sqrt(128)",
                "selected_sdpa": "[1,16,1,128] with mask [1,1,1,2127], is_causal=False",
                "prefix": "live pre-K positions 1698:1703 with phases 2121:2126 only in selected call",
                "model_forward_cap": MAX_MODEL_FORWARDS,
                "vision_forward_cap": MAX_VISION_FORWARDS,
                "wall_seconds_cap": MAX_SECONDS, "tensor_cap_bytes": MAX_TENSOR_BYTES,
                "parity_atol": ATOL,
            },
        }
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)

        def count_model(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_MODEL_FORWARDS and
                    time.monotonic() - state["started"] <= MAX_SECONDS,
                    "model forward/time cap exceeded")

        def count_vision(*_args):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= MAX_VISION_FORWARDS,
                    "vision forward cap exceeded")

        counter_handles = [q.model.register_forward_pre_hook(count_model, with_kwargs=True),
                           q.model.model.visual.register_forward_pre_hook(count_vision)]
        try:
            for label in CELL_ORDER:
                state["current_cell"] = label
                write(OUT / "receipt.json", state)
                result, logits = run_cell(label, sources, q, identity, device, state, native_logits)
                if native_logits is None:
                    native_logits = logits.clone()
                record = result["record"]
                cell_readbacks[label] = record
                write(OUT / "readback.json", {
                    "schema": "recurrence_local_prefix_phase.readback.v1", "status": "running",
                    "source_manifest": binding(OUT / "source-to-cell.json"), "cells": cell_readbacks,
                    "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                })
                # Preserve evidence before gates, then apply only mechanical identity checks.
                require(record["query_consumer"] == {"shape": [4, 2, HIDDEN],
                                                     "physical_indices": [RECIPIENT_POSITION, DONOR_POSITION],
                                                     "exact": True}, "selected LM consumer changed")
                payload = torch.load(result["tensor_path"], map_location="cpu", weights_only=True)
                require(float((payload["layer1_input"][0] - sources["late_input"]).abs().max()) <= ATOL,
                        "late incoming layer1 state mismatch")
                require(record["native_input_hashes"] == sources["selection"]["native_input_hashes"],
                        "native input identity mismatch")
                for index in PHASE_LAYERS:
                    metric = record["layer_metrics"][str(index)]
                    reference = cell_readbacks["query_only"]["layer_metrics"][str(index)]
                    require(all(metric[key] <= ATOL for key in metric if
                                key.startswith("fp64_") or key.endswith("max_abs") and
                                key != "phase_delta_max_abs"), "base rotation/norm/phase gate failed")
                    require(metric["q_k_off_target_exact"] and metric["gqa_expansion_exact"] and
                            metric["o_proj_consumer_exact"] and metric["sdpa_calls"] == 1 and
                            all(metric[key] == reference[key] for key in
                                ("historical_K_hash", "historical_V_hash", "mask_hash", "cache_slots_hash")),
                            "base history/consumer identity gate failed")
                    if label != "query_only":
                        raw = payload["local_prefix"]["layers"][str(index)]
                        expected_k = rotate_rows_independent(raw["pre_K"], raw["late_cos"] if label == "prefix_coherent" else raw["early_cos"],
                                                            raw["late_sin"] if label == "prefix_coherent" else raw["early_sin"]).repeat_interleave(2, dim=0)
                        require(float((raw["selected_k_prefix_used"].double() - expected_k).abs().max()) <= ATOL,
                                "local prefix rotation gate failed")
                        require(torch.equal(raw["selected_q_target"], payload["consumed_Q_target"][index - 1]),
                                "selected Q differs from base consumed Q")
                        if label == "prefix_coherent":
                            early_q = phase.rotate_fp64(payload["pre_Q"][index - 1], sources["early_cos"], sources["early_sin"])
                            early_k = rotate_rows_independent(raw["pre_K"], raw["early_cos"], raw["early_sin"]).repeat_interleave(2, dim=0)
                            old_scores = torch.einsum("hd,hpd->hp", early_q, early_k) * HEAD_DIM ** -0.5
                            new_scores = torch.einsum("hd,hpd->hp", raw["selected_q_target"].double(), raw["selected_k_prefix_used"].double()) * HEAD_DIM ** -0.5
                            require(float((new_scores - old_scores).abs().max()) <= ATOL,
                                    "local relative-phase score preservation failed")
                require(float((logits[[0, 1, 3]] - native_logits[[0, 1, 3]]).abs().max()) <= ATOL,
                        "companion logits changed")
                require(all(item["full_input_unchanged"] and
                            item["full_call_k_hash"] == item["full_input_k_hash_before"] and
                            item["full_input_k_hash_after"] == item["full_input_k_hash_before"] and
                            item["off_target_output_exact"]
                            for item in record["local_prefix"]["layers"].values()),
                        f"{label} full-call identity gate failed")
                if label == "query_only":
                    require(float((logits - predecessor_logits).abs().max()) <= ATOL and
                            logits[TARGET, 0].argmax().item() == P579,
                            "fresh query-only baseline diverged from accepted held-late result")
                elif label == "identity_recompute":
                    require(float((logits - native_logits).abs().max()) <= ATOL and
                            logits[TARGET, 0].argmax().item() == P579,
                            "identity selected recomputation changed baseline")
                    require(all(item["selected_call"] and item["selected_output_target"] is not None
                                for item in result["outer_records"].values() if item["mode"] == "identity"),
                            "identity selected consumer evidence incomplete")
                else:
                    require(all(item["selected_call"] and item["selected_output_target"] is not None
                                for item in result["outer_records"].values() if item["mode"] == "prefix_coherent"),
                            "prefix selected consumer evidence incomplete")
                state["completed_cells"].append(label)
                state["calls"].append({"cell": label, "seconds": record["elapsed_seconds"]})
                write(OUT / "receipt.json", state)
                del result, payload
        finally:
            for handle in counter_handles:
                handle.remove()
        require(state["model_forwards"] == MAX_MODEL_FORWARDS and
                state["vision_forwards"] == MAX_VISION_FORWARDS,
                "exactly three model and vision forwards required")
        final_readback = {"schema": "recurrence_local_prefix_phase.readback.v1", "status": "candidate",
                          "source_manifest": binding(OUT / "source-to-cell.json"), "cells": cell_readbacks,
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"]}
        write(OUT / "readback.json", final_readback)
        result = {"status": "candidate", "cell_order": list(CELL_ORDER),
                  "target_winners": {label: [cell_readbacks[label]["summary"]["queries"][i]["winner"]
                                               for i in range(2)] for label in CELL_ORDER},
                  "target_probabilities": {
                      label: [{"p38": row["p38"], "p999": row["p999"], "d_z38_minus_z999": row["d_z38_minus_z999"]}
                              for row in cell_readbacks[label]["summary"]["queries"]]
                      for label in CELL_ORDER},
                  "readback": binding(OUT / "readback.json"),
                  "source_manifest": binding(OUT / "source-to-cell.json"),
                  "artifacts": {label: cell_readbacks[label]["tensor_artifact"] for label in CELL_ORDER},
                  "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"]}
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
        if cell_readbacks:
            write(OUT / "readback.json", {"schema": "recurrence_local_prefix_phase.readback.v1",
                                           "status": "technical_invalid",
                                           "source_manifest": binding(OUT / "source-to-cell.json")
                                           if (OUT / "source-to-cell.json").exists() else None,
                                           "cells": cell_readbacks, "error": repr(error)})
        raise
    finally:
        phase.OUT = original_phase_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        cpu_selfcheck()
    else:
        run(args.device)
