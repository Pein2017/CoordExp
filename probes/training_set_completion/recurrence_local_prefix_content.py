"""Transfer a repeated row's contextual prefix K/V at native destination phases."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
import src.adapters.dora as dora_runtime
import src.qwen.runtime_loading as qwen_runtime_loading
import src.qwen.untied_embeddings as untied_embeddings
import src.artifacts.source_provenance as source_provenance
from transformers import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion import artifacts as probe_artifacts
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion import untied_shared
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


REPO = Path(__file__).resolve().parents[2]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-local-prefix-content-transfer/unit.md"
SELECTION = UNIT.with_name("selection.json")
CELL_ORDER = ("native", "identity", "donor_content")
LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM = 28, 16, 8, 128
ATOL = 2e-4
MAX_SECONDS = 900
MAX_TENSOR_BYTES = 96 << 20


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def binding(path: Path) -> dict:
    return literal_binding(path)


def verify_binding(expected: dict) -> dict:
    path = Path(expected["path"])
    actual = binding(path)
    require(all(actual[key] == expected[key] for key in ("path", "sha256", "size_bytes")),
            f"source binding changed: {path}")
    return actual


def json_sha256(value) -> str:
    # Match the frozen source-selection identity hash: default JSON separators.
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def rotate_prefix_fp32(pre_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate [KV,prefix,dim] pre-K with the destination's native phases."""
    require(pre_k.ndim == 3 and cos.shape == sin.shape == (pre_k.shape[1], pre_k.shape[2]),
            "prefix rotation dimensions changed")
    half = pre_k.shape[-1] // 2
    rotated = torch.cat((-pre_k[..., half:], pre_k[..., :half]), dim=-1)
    return pre_k * cos.unsqueeze(0) + rotated * sin.unsqueeze(0)


def rotate_prefix_independent(pre_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Independent FP64 real/imaginary-half rotation used by the CPU gate."""
    half = pre_k.shape[-1] // 2
    a, b = pre_k[..., :half].double(), pre_k[..., half:].double()
    c, s = cos[..., :half].double(), sin[..., :half].double()
    return torch.cat((a * c.unsqueeze(0) - b * s.unsqueeze(0),
                      b * c.unsqueeze(0) + a * s.unsqueeze(0)), dim=-1)


def selected_sdpa_replacement(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    native_sdpa,
    scale: float,
    mode: str,
    target_batch: int,
    query_position: int,
    prefix_start: int,
    prefix_end: int,
    donor_pre_k: torch.Tensor | None = None,
    donor_v: torch.Tensor | None = None,
    destination_cos: torch.Tensor | None = None,
    destination_sin: torch.Tensor | None = None,
    wrapper_entry_hashes: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run native full attention, then optionally recompute only one query."""
    batch, q_heads, width, dim = query.shape
    require(mode in CELL_ORDER and key.shape == value.shape == query.shape,
            "post-GQA SDPA layout changed")
    require(mask.dtype == torch.bool and mask.shape == (batch, 1, width, width),
            "native SDPA mask layout changed")
    require(0 <= target_batch < batch and 0 <= prefix_start < prefix_end <= query_position < width,
            "selected query/prefix indices changed")
    require(float(scale) == dim ** -0.5, "native SDPA scale changed")
    before = {name: tensor_hash(item) for name, item in
              (("q", query), ("k", key), ("v", value), ("mask", mask))}
    if wrapper_entry_hashes is not None:
        require(before == wrapper_entry_hashes,
                "full SDPA inputs differ from wrapper-entry tensors")
    full_output = native_sdpa(query, key, value, attn_mask=mask, dropout_p=0.0,
                              is_causal=False, scale=scale)
    selected_q = query[target_batch:target_batch + 1, :, query_position:query_position + 1, :]
    k_selected = key[target_batch:target_batch + 1].clone()
    v_selected = value[target_batch:target_batch + 1].clone()
    selected_mask = mask[target_batch:target_batch + 1, :, query_position:query_position + 1, :]
    selected_k_before = k_selected[0, :, prefix_start:prefix_end].detach().float().cpu().clone()
    selected_v_before = v_selected[0, :, prefix_start:prefix_end].detach().float().cpu().clone()
    selected_k_used = selected_k_before.clone()
    selected_v_used = selected_v_before.clone()
    selected_output = None
    if mode != "native":
        if mode == "donor_content":
            require(all(isinstance(item, torch.Tensor) for item in
                        (donor_pre_k, donor_v, destination_cos, destination_sin)),
                    "donor K/V or destination phases are missing")
            kv_heads = donor_pre_k.shape[1]
            require(donor_pre_k.shape == donor_v.shape ==
                    (prefix_end - prefix_start, kv_heads, dim) and
                    q_heads % kv_heads == 0,
                    "live donor K/V dimensions changed")
            require(destination_cos.shape == destination_sin.shape ==
                    (prefix_end - prefix_start, dim), "destination phase dimensions changed")
            k_rows = rotate_prefix_fp32(donor_pre_k.permute(1, 0, 2),
                                        destination_cos, destination_sin)
            k_rows = k_rows.repeat_interleave(q_heads // kv_heads, dim=0)
            v_rows = donor_v.permute(1, 0, 2).repeat_interleave(
                q_heads // kv_heads, dim=0)
            k_selected[0, :, prefix_start:prefix_end, :] = k_rows
            v_selected[0, :, prefix_start:prefix_end, :] = v_rows
        selected_k_used = k_selected[0, :, prefix_start:prefix_end].detach().float().cpu().clone()
        selected_v_used = v_selected[0, :, prefix_start:prefix_end].detach().float().cpu().clone()
        selected_output = native_sdpa(selected_q, k_selected, v_selected,
                                      attn_mask=selected_mask, dropout_p=0.0,
                                      is_causal=False, scale=scale)
        replaced = full_output.clone()
        replaced[target_batch, :, query_position, :] = selected_output[0, :, 0, :]
    else:
        replaced = full_output
    after = {name: tensor_hash(item) for name, item in
             (("q", query), ("k", key), ("v", value), ("mask", mask))}
    outside_prefix = torch.ones(width, dtype=torch.bool, device=key.device)
    outside_prefix[prefix_start:prefix_end] = False
    nonprefix_kv_exact = (
        torch.equal(k_selected[:, :, outside_prefix], key[target_batch:target_batch + 1, :, outside_prefix]) and
        torch.equal(v_selected[:, :, outside_prefix],
                    value[target_batch:target_batch + 1, :, outside_prefix]))
    selected_q_exact = torch.equal(selected_q, query[target_batch:target_batch + 1, :, query_position:query_position + 1])
    selected_mask_exact = torch.equal(
        selected_mask, mask[target_batch:target_batch + 1, :, query_position:query_position + 1, :])
    off_target_exact = (
        torch.equal(replaced[:target_batch], full_output[:target_batch]) and
        torch.equal(replaced[target_batch, :, :query_position],
                    full_output[target_batch, :, :query_position]) and
        torch.equal(replaced[target_batch, :, query_position + 1:],
                    full_output[target_batch, :, query_position + 1:]) and
        torch.equal(replaced[target_batch + 1:], full_output[target_batch + 1:]))
    return replaced, {
        "full_input_hashes_before": before,
        "full_input_hashes_after": after,
        "full_inputs_unchanged": before == after,
        "query": selected_q[0, :, 0].detach().float().cpu().clone(),
        "mask_row": selected_mask[0, 0, 0:1].detach().cpu().clone(),
        "full_output_target": full_output[target_batch, :, query_position].detach().float().cpu().clone(),
        "selected_output_target": (None if selected_output is None else
                                   selected_output[0, :, 0].detach().float().cpu().clone()),
        "selected_k_before": selected_k_before,
        "selected_v_before": selected_v_before,
        "selected_k_used": selected_k_used,
        "selected_v_used": selected_v_used,
        "selected_mask_hash": tensor_hash(selected_mask),
        "selected_k_before_hash": tensor_hash(selected_k_before),
        "selected_v_before_hash": tensor_hash(selected_v_before),
        "selected_k_used_hash": (None if selected_k_used is None else tensor_hash(selected_k_used)),
        "selected_v_used_hash": (None if selected_v_used is None else tensor_hash(selected_v_used)),
        "selected_call": mode != "native",
        "selected_shape": ([1, q_heads, 1, dim] if mode != "native" else None),
        "off_target_output_exact": off_target_exact,
        "selected_q_exact": selected_q_exact,
        "selected_mask_exact": selected_mask_exact,
        "selected_nonprefix_kv_exact": nonprefix_kv_exact,
    }


def _manual_attention(query, key, value, mask, scale):
    scores = torch.matmul(query.double(), key.double().transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1).matmul(value.double()).float()


def _cpu_consumer(value: torch.Tensor, expected: torch.Tensor, target: int, position: int) -> None:
    """Exercise the same flattened attention output consumed by Qwen's o_proj."""
    projection = torch.nn.Linear(value.shape[-1], 5, bias=False)
    seen = {}

    def check(_module, args):
        actual = args[0]
        require(torch.equal(actual, value), "o_proj input differed from returned attention output")
        require(torch.allclose(actual, expected, atol=ATOL, rtol=0.0),
                "o_proj consumed a different selected-query output")
        seen["target"] = actual[target, position].detach().clone()

    hook = projection.register_forward_pre_hook(check)
    try:
        projection(value)
    finally:
        hook.remove()
    require("target" in seen, "o_proj consumer hook did not run")


def _cpu_call(q, k, v, mask, *, target, position, prefix, mode,
              donor_k, donor_v, cos, sin, original_sdpa,
              wrong_position=False, global_history=False):
    module = torch.nn.Module()
    module.num_key_value_groups = q.shape[1] // k.shape[1]
    module.config = type("Config", (), {"_attn_implementation": "sdpa"})()
    capture = {}

    def outer(query, key, value, *, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
        require(not enable_gqa and is_causal is False and dropout_p == 0.0,
                "installed SDPA caller flags changed")
        entry_hashes = {name: tensor_hash(item) for name, item in
                        (("q", query), ("k", key), ("v", value), ("mask", attn_mask))}
        if global_history:
            key = key.clone()
            key[target, :, prefix[0], 0] += 0.75
        pos = position + 1 if wrong_position else position
        actual, record = selected_sdpa_replacement(
            query, key, value, attn_mask, native_sdpa=original_sdpa,
            scale=float(scale), mode=mode, target_batch=target,
            query_position=pos, prefix_start=prefix[0], prefix_end=prefix[1],
            donor_pre_k=donor_k, donor_v=donor_v,
            destination_cos=cos, destination_sin=sin,
            wrapper_entry_hashes=entry_hashes)
        capture.update(record)
        return actual

    before = F.scaled_dot_product_attention
    try:
        F.scaled_dot_product_attention = outer
        output, _weights = sdpa_attention_forward(module, q, k, v, mask,
                                                  scaling=q.shape[-1] ** -0.5)
    finally:
        F.scaled_dot_product_attention = before
    return output.reshape(output.shape[0], output.shape[1], -1), capture


def cpu_selfcheck() -> dict:
    """Qualify installed SDPA, GQA, left padding, variable prefixes, and mutations."""
    torch.manual_seed(8147)
    batch, q_heads, kv_heads, width, dim = 3, 4, 2, 9, 8
    target, position = 2, 7
    q = torch.randn(batch, q_heads, width, dim)
    k = torch.randn(batch, kv_heads, width, dim)
    v = torch.randn(batch, kv_heads, width, dim)
    valid = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    indices = torch.arange(width)
    mask = (indices[None, None, None, :] <= indices[None, None, :, None]) & valid[:, None, None, :]
    scale = dim ** -0.5
    original_sdpa = F.scaled_dot_product_attention
    reports = []
    for prefix_count, prefix_start in ((2, 4), (4, 3)):
        selected_target = target if prefix_count == 2 else 1
        selected_position = position if prefix_count == 2 else 7
        prefix = (prefix_start, prefix_start + prefix_count)
        angles = torch.arange(prefix_count * (dim // 2), dtype=torch.float32).reshape(
            prefix_count, dim // 2) / 19.0 + 0.11
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
        donor_k = torch.arange(prefix_count * kv_heads * dim, dtype=torch.float32).reshape(
            prefix_count, kv_heads, dim) / 29.0 + 0.17
        donor_v = torch.flip(donor_k, dims=(0,)).clone() + 0.31
        rotated = rotate_prefix_independent(donor_k.permute(1, 0, 2), cos, sin)
        used_k = rotated.repeat_interleave(q_heads // kv_heads, dim=0)
        used_v = donor_v.permute(1, 0, 2).repeat_interleave(q_heads // kv_heads, dim=0)
        repeated_k = k.repeat_interleave(q_heads // kv_heads, dim=1)
        repeated_v = v.repeat_interleave(q_heads // kv_heads, dim=1)
        native_full = original_sdpa(q, repeated_k, repeated_v, attn_mask=mask,
                                    dropout_p=0.0, is_causal=False, scale=scale)
        expected_native = native_full.transpose(1, 2).contiguous().reshape(batch, width, q_heads * dim)
        native, native_record = _cpu_call(
            q, k, v, mask, target=selected_target, position=selected_position,
            prefix=prefix, mode="native", donor_k=donor_k, donor_v=donor_v,
            cos=cos, sin=sin, original_sdpa=original_sdpa)
        require(torch.equal(native, expected_native), "native installed SDPA caller changed output")
        identity, identity_record = _cpu_call(
            q, k, v, mask, target=selected_target, position=selected_position,
            prefix=prefix, mode="identity", donor_k=donor_k, donor_v=donor_v,
            cos=cos, sin=sin, original_sdpa=original_sdpa)
        require(torch.allclose(identity, native, atol=ATOL, rtol=0.0),
                "identity recomputation changed installed caller output")
        _cpu_consumer(identity, expected_native, selected_target, selected_position)
        donor, donor_record = _cpu_call(
            q, k, v, mask, target=selected_target, position=selected_position,
            prefix=prefix, mode="donor_content", donor_k=donor_k, donor_v=donor_v,
            cos=cos, sin=sin, original_sdpa=original_sdpa)
        expected_k = repeated_k[selected_target:selected_target + 1].clone()
        expected_v = repeated_v[selected_target:selected_target + 1].clone()
        expected_k[:, :, prefix[0]:prefix[1], :] = used_k.unsqueeze(0)
        expected_v[:, :, prefix[0]:prefix[1], :] = used_v.unsqueeze(0)
        selected_q = q[selected_target:selected_target + 1, :, selected_position:selected_position + 1]
        selected_mask = mask[selected_target:selected_target + 1, :, selected_position:selected_position + 1]
        expected_selected = _manual_attention(selected_q, expected_k, expected_v,
                                              selected_mask, scale)[0, :, 0]
        expected_donor = expected_native.clone()
        expected_donor[selected_target, selected_position] = expected_selected.reshape(-1)
        require(float((donor_record["selected_k_used"] - used_k).abs().max()) <= ATOL and
                float((donor_record["selected_v_used"] - used_v).abs().max()) == 0.0,
                "donor K phase or V content differs from independent expectation")
        require(donor_record["full_inputs_unchanged"] and donor_record["off_target_output_exact"] and
                donor_record["selected_nonprefix_kv_exact"] and donor_record["selected_q_exact"] and
                donor_record["selected_mask_exact"],
                "donor CPU call failed production identity/consumer invariants")
        require(torch.allclose(donor, expected_donor, atol=ATOL, rtol=0.0),
                "donor SDPA output differs from independent FP64 attention")
        _cpu_consumer(donor, expected_donor, selected_target, selected_position)

        for label, mutation in (("wrong_phase", "phase"), ("wrong_v", "v"),
                                ("wrong_query", "query"), ("global_history", "global")):
            bad_cos, bad_sin, bad_v = cos, sin, donor_v
            wrong_query = global_history = False
            if mutation == "phase":
                bad_sin = -sin
            elif mutation == "v":
                bad_v = donor_v + 0.75
            elif mutation == "query":
                wrong_query = True
            else:
                global_history = True
            bad_mode = "identity" if mutation == "global" else "donor_content"
            try:
                bad, _record = _cpu_call(
                    q, k, v, mask, target=selected_target, position=selected_position,
                    prefix=prefix, mode=bad_mode, donor_k=donor_k, donor_v=bad_v,
                    cos=bad_cos, sin=bad_sin, original_sdpa=original_sdpa,
                    wrong_position=wrong_query, global_history=global_history)
                _cpu_consumer(bad, expected_donor if mutation != "global" else expected_native,
                              selected_target, selected_position)
            except ValueError as error:
                expected_error = ("full SDPA inputs differ from wrapper-entry tensors"
                                  if mutation == "global" else
                                  "o_proj consumed a different selected-query output")
                require(expected_error in str(error),
                        f"{label} failed outside the production identity/consumer invariant: {error}")
            else:
                raise AssertionError(f"{label} mutation escaped the production identity/consumer check")
            if mutation == "query":
                require(selected_position + 1 < width and prefix[1] <= selected_position + 1,
                        "wrong-query mutation no longer uses a valid adjacent query")
        reports.append({"prefix_count": prefix_count,
                        "native": native_record["full_inputs_unchanged"],
                        "identity": identity_record["full_inputs_unchanged"],
                        "donor": donor_record["full_inputs_unchanged"],
                        "mutations_rejected_at_o_proj":
                            ["wrong_phase", "wrong_v", "wrong_query"],
                        "global_history_rejected_at": "wrapper-entry full-input identity"})
    result = {"status": "cpu-qualified", "caller": "transformers.integrations.sdpa_attention.sdpa_attention_forward",
              "consumer": "Linear(o_proj input)", "gqa": [q_heads, kv_heads],
              "left_padding": True, "prefix_counts": [item["prefix_count"] for item in reports],
              "cases": reports}
    return result


def _case_sources(case: dict) -> tuple[list[dict], list[dict], dict]:
    raw_binding = verify_binding(case["raw"])
    trace_binding = verify_binding(case["trace"])
    receipt_binding = verify_binding(case["runtime_receipt"])
    raw_data = json.loads(Path(raw_binding["path"]).read_text())
    rows = raw_data["rows"] if isinstance(raw_data, dict) else raw_data
    trace_data = json.loads(Path(trace_binding["path"]).read_text())
    steps = trace_data["steps"] if isinstance(trace_data, dict) else trace_data
    receipt = json.loads(Path(receipt_binding["path"]).read_text())
    require(receipt.get("condition") == "untied-original" and
            receipt.get("group") == case["group"] and receipt.get("status") == "candidate_complete",
            "source runtime receipt identity/status changed")
    require(receipt.get("raw") == raw_binding and receipt.get("trace") == trace_binding,
            "source runtime receipt no longer binds selected raw/trace")
    return rows, steps, receipt


def _check_trace(case: dict, rows: list[dict], steps: list[dict]) -> dict:
    batch = int(case["batch_index"])
    action = int(case["raw_action"])
    row = rows[batch]
    require(int(row["image_id"]) == int(case["image_id"]) and row["row_id"] == case["row_id"],
            "selected raw row identity changed")
    tokens = row["token_ids"]
    require(0 < action < len(tokens), "selected raw action is outside source tokens")
    source_step = next((item for item in steps if int(item["offset"]) == action), None)
    require(source_step is not None, "selected raw/trace action offset missing")
    saved = case["saved_trace"]
    observed = {"winner": int(source_step["raw_winners"][batch]),
                "runnerup": int(source_step["raw_runnerups"][batch]),
                "top2": [float(x) for x in source_step["raw_top2"][batch]],
                "logsumexp": float(source_step["logsumexp"][batch])}
    require(observed["winner"] == int(saved["winner"]) == int(case["native_token"]) and
            observed["runnerup"] == int(saved["runnerup"]) and
            max(abs(a - float(b)) for a, b in zip(observed["top2"], saved["top2"], strict=True)) <= ATOL and
            abs(observed["logsumexp"] - float(saved["logsumexp"])) <= ATOL,
            "selected natural trace no longer matches root selection")
    return {"trace_step": observed, "query_input_token": int(tokens[action - 1]),
            "next_source_token": int(tokens[action])}


def _prepared_case(case: dict, panel: dict, q, device: str) -> dict:
    rows, steps, receipt = _case_sources(case)
    trace_crosswalk = _check_trace(case, rows, steps)
    group = next((item for item in panel["groups"] if item["key"] == case["group"]), None)
    require(group is not None and len(group["cases"]) == 4, "source native group changed")
    native_case = group["cases"][int(case["batch_index"])]
    require(native_case["row_id"] == case["row_id"] and
            str(Path(native_case["image_path"]).resolve()) == str(Path(case["image"]["path"]).resolve()),
            "selected image/panel row identity changed")
    image_binding = verify_binding(case["image"])
    config = dict(panel["configs"]["untied"])
    config["data"] = {"input_jsonl": group["input_jsonl"]}
    requests, _metadata = build_bound_native_requests(q, config, group["cases"])
    batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
    require(input_identity(batch) == receipt["input_identity"],
            "reconstructed native prompt/media identity differs from source receipt")
    prompts = [list(row) for row in batch.prompt_token_ids]
    prompt_lengths = [len(row) for row in prompts]
    width = int(batch.inputs["input_ids"].shape[1])
    require(prompt_lengths == [int(x) for x in case["prompt_lengths"]] and
            width == max(prompt_lengths) == int(case["prompt_width"]),
            "actual padded prompt width differs from selection")
    action = int(case["raw_action"])
    tails = _prefix_tokens(rows, action, int(q.tokenizer.pad_token_id))
    histories = [prompt + tail for prompt, tail in zip(prompts, tails, strict=True)]
    inputs = exact_history_inputs(q.model, batch.inputs, histories,
                                  pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
    batch_index, query = int(case["batch_index"]), int(case["query"])
    require(int(inputs["input_ids"].shape[1]) == width + action and
            query == width + action - 1 == int(inputs["input_ids"].shape[1]) - 1 and
            bool(inputs["attention_mask"][batch_index, query]),
            "physical query/padded-prefix relation changed")
    require(int(inputs["input_ids"][batch_index, query]) == trace_crosswalk["query_input_token"],
            "native query input token differs from saved raw/trace")
    require(int(inputs["input_ids"][batch_index, query]) == int(rows[batch_index]["token_ids"][action - 1]),
            "selected query input token differs from source raw history")
    count = int(case["prefix_count"])
    donor_raw = [int(x) for x in case["donor_prefix_raw"]]
    dest_raw = [int(x) for x in case["destination_prefix_raw"]]
    donor_pos = [int(x) for x in case["donor_prefix_physical"]]
    dest_pos = [int(x) for x in case["destination_prefix_physical"]]
    require(len(donor_raw) == len(dest_raw) == len(donor_pos) == len(dest_pos) == 2 and
            donor_raw[1] - donor_raw[0] == dest_raw[1] - dest_raw[0] == count and
            donor_pos == [width + donor_raw[0], width + donor_raw[1]] and
            dest_pos == [width + dest_raw[0], width + dest_raw[1]] and
            dest_pos[1] == query and dest_raw == [action - int(case["row_difference_offset"]), action - 1] and
            count == int(case["row_difference_offset"]) - 1,
            "literal prefix mapping includes a self-key or has changed")
    tokens = rows[batch_index]["token_ids"]
    prefix_tokens = [int(x) for x in case["prefix_tokens"]]
    require(tokens[donor_raw[0]:donor_raw[1]] == prefix_tokens and
            tokens[dest_raw[0]:dest_raw[1]] == prefix_tokens and
            inputs["input_ids"][batch_index, donor_pos[0]:donor_pos[1]].tolist() == prefix_tokens and
            inputs["input_ids"][batch_index, dest_pos[0]:dest_pos[1]].tolist() == prefix_tokens,
            "donor/destination tokens differ in raw source or actual model input")
    full_width = int(inputs["input_ids"].shape[1])
    media_hashes = {key: tensor_hash(value) for key, value in inputs.items()
                    if key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
                    and isinstance(value, torch.Tensor)}
    input_hashes = {key: tensor_hash(inputs[key]) for key in
                    ("input_ids", "attention_mask", "position_ids")}
    expected_attention_row = (inputs["attention_mask"][batch_index].bool() &
                              (torch.arange(full_width, device=inputs["input_ids"].device) <= query))
    return {"selection": case, "raw_rows": rows, "trace_steps": steps,
            "trace_crosswalk": trace_crosswalk, "receipt": receipt,
            "image_binding": image_binding, "group": group, "batch": batch,
            "batch_identity": input_identity(batch), "inputs": inputs,
            "prompt_width": width, "input_width": full_width,
            "batch_index": batch_index, "query": query,
            "donor_raw": donor_raw, "dest_raw": dest_raw,
            "donor_pos": donor_pos, "dest_pos": dest_pos,
            "prefix_count": count, "input_hashes": input_hashes,
            "media_hashes": media_hashes,
            "expected_attention_row": expected_attention_row,
            "position_ids_target": inputs["position_ids"][:, batch_index, query].detach().cpu().tolist()}


def _model_contract(q, identity: dict, selection: dict, receipts: list[dict]) -> dict:
    expected = dict(receipts[0]["identity"])
    source_identity = dict(expected)
    current = dict(identity)
    loader, saved_loader = current.pop("loader_source"), expected.pop("loader_source")
    require(current == expected and (loader["sha256"], loader["size_bytes"]) ==
            (saved_loader["sha256"], saved_loader["size_bytes"]),
            "loaded model/effective-row identity differs from natural source")
    identity_hash = json_sha256(source_identity)
    for receipt in receipts:
        receipt_identity = dict(receipt["identity"])
        receipt_loader = receipt_identity.pop("loader_source")
        require(receipt_identity == expected and
                (receipt_loader["sha256"], receipt_loader["size_bytes"]) ==
                (saved_loader["sha256"], saved_loader["size_bytes"]) and
                json_sha256(receipt["identity"]) == identity_hash,
                "selected natural receipts do not share one exact model identity")
    require(all(identity_hash == case["model_identity_sha256"] for case in selection["cases"]),
            "loaded model identity hash differs from source selection")
    text_config = q.model.config.text_config
    output_head = q.model.get_output_embeddings()
    vocab_size = int(text_config.vocab_size)
    require(vocab_size == 152670 and int(output_head.base.weight.shape[0]) == vocab_size,
            "loaded native vocabulary differs from frozen source contract")
    contract = {"layers": int(text_config.num_hidden_layers),
                "query_heads": int(text_config.num_attention_heads),
                "kv_heads": int(text_config.num_key_value_heads),
                "head_dim": int(text_config.head_dim),
                "hidden_size": int(text_config.hidden_size),
                "vocab_size": vocab_size}
    require(contract == {"layers": LAYERS, "query_heads": Q_HEADS, "kv_heads": KV_HEADS,
                         "head_dim": HEAD_DIM, "hidden_size": 2048, "vocab_size": 152670},
            "loaded native architecture differs from frozen GQA contract")
    return {"loaded_identity": identity, "source_receipt_identity": source_identity,
            "identity_sha256": identity_hash,
            "hash_format": "sha256(json.dumps(receipt.identity, sort_keys=True).encode())",
            "checkpoint": {"base_model_path": str(q.base_model_path),
                           "untied_checkpoint": str(untied_shared.UNTIED.resolve()),
                           "effective_adapter": identity["adapter"],
                           "effective_embedding": identity["embedding"]},
            "architecture": contract,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__}


def _full_input_hashes(inputs: dict) -> dict:
    return {key: tensor_hash(value) for key, value in inputs.items()
            if isinstance(value, torch.Tensor)}


def _capture_sources(output: Path, source_paths: list[tuple[Path, str]]) -> list[dict]:
    result = []
    for path, relative_name in source_paths:
        captured = preserve_source(path, run_root=output, relative_name=relative_name)
        result.append({"source": binding(path), "capture": binding(captured),
                       "archive_name": relative_name})
    return result


def _layer_payload(capture: dict, cell: str) -> dict:
    record = capture["sdpa"]
    item = {
        "query": record["query"], "mask_row": record["mask_row"],
        "full_output_target": record["full_output_target"],
        "o_proj_input": capture["o_proj_input"],
        "o_proj_output": capture["o_proj_output"],
        "donor_pre_k": capture["donor_pre_k"], "donor_v": capture["donor_v"],
        "destination_pre_k": capture["destination_pre_k"], "destination_v": capture["destination_v"],
        "donor_cos": capture["donor_cos"], "donor_sin": capture["donor_sin"],
        "destination_cos": capture["destination_cos"], "destination_sin": capture["destination_sin"],
        "position_ids_donor": capture["position_ids_donor"],
        "position_ids_destination": capture["position_ids_destination"],
        "position_ids_query": capture["position_ids_query"],
        "selected_k_before": record["selected_k_before"],
        "selected_v_before": record["selected_v_before"],
        "selected_k_used": record["selected_k_used"],
        "selected_v_used": record["selected_v_used"],
    }
    if cell != "native":
        item["selected_output_target"] = record["selected_output_target"]
    if capture.get("incoming_layer1_target") is not None:
        item["incoming_layer1_target"] = capture["incoming_layer1_target"]
    return {key: value for key, value in item.items() if value is not None}


def _run_cell(prepared: dict, q, cell: str, output: Path, state: dict,
              expected_native: dict | None = None) -> tuple[dict, dict]:
    case = prepared["selection"]
    b, query = prepared["batch_index"], prepared["query"]
    donor_start, donor_end = prepared["donor_pos"]
    dest_start, dest_end = prepared["dest_pos"]
    device = prepared["inputs"]["input_ids"].device
    model, text = q.model, q.model.model.language_model
    call_inputs = dict(prepared["inputs"])
    cache = DynamicCache()
    call_inputs.update(past_key_values=cache, use_cache=True,
                       cache_position=torch.arange(prepared["input_width"], device=device))
    captures = [{"o_proj_input": None, "o_proj_output": None,
                 "donor_pre_k": None, "donor_v": None,
                 "destination_pre_k": None, "destination_v": None,
                 "donor_cos": None, "donor_sin": None,
                 "destination_cos": None, "destination_sin": None,
                 "position_ids_donor": None, "position_ids_destination": None,
                 "position_ids_query": None, "incoming_layer1_target": None,
                 "sdpa": None} for _ in range(LAYERS)]
    checks = [{"layer": i, "sdpa_calls": 0, "layer_attention_consumed": False,
               "o_proj_consumed": False} for i in range(LAYERS)]
    active_layer = [None]
    full_calls = [0]
    vision_calls = [0]
    model_calls = [0]
    lm_head_output = [None]
    rotary_positions = [None]
    returned_attention_inputs = [None] * LAYERS
    expected_row = prepared["expected_attention_row"]
    original_sdpa = F.scaled_dot_product_attention
    handles = []

    def model_before(_module, _args, kwargs):
        model_calls[0] += 1
        state["model_forwards"] += 1
        require(all(torch.equal(kwargs[key], prepared["inputs"][key]) for key in
                    ("input_ids", "attention_mask", "position_ids", "pixel_values", "image_grid_thw")
                    if key in prepared["inputs"]),
                "real model call did not receive frozen exact-history tensors")
        actual_slots = kwargs.get("cache_position")
        require(kwargs.get("past_key_values") is cache and kwargs.get("use_cache") is True and
                isinstance(actual_slots, torch.Tensor) and
                torch.equal(actual_slots, call_inputs["cache_position"]),
                "full replay did not use its fresh native cache and slots")

    def visual_before(_module, args, kwargs):
        vision_calls[0] += 1
        state["vision_forwards"] += 1
        values = dict(kwargs)
        if args:
            values.setdefault("pixel_values", args[0])
        if len(args) > 1:
            values.setdefault("grid_thw", args[1])
        require(torch.equal(values.get("pixel_values"), prepared["inputs"]["pixel_values"]) and
                torch.equal(values.get("grid_thw"), prepared["inputs"]["image_grid_thw"]),
                "visual consumer image or grid input changed")

    def rotary_before(_module, args):
        require(len(args) >= 2 and isinstance(args[1], torch.Tensor) and
                torch.equal(args[1], prepared["inputs"]["position_ids"]),
                "rotary consumer did not receive exact three-axis source positions")
        rotary_positions[0] = args[1].detach().clone()

    def lm_after(_module, _args, result):
        lm_head_output[0] = result.detach().float().cpu().clone()

    handles.extend((model.register_forward_pre_hook(model_before, with_kwargs=True),
                    model.model.visual.register_forward_pre_hook(visual_before, with_kwargs=True),
                    text.rotary_emb.register_forward_pre_hook(rotary_before),
                    model.get_output_embeddings().register_forward_hook(lm_after)))

    def make_attention_pre(index):
        def before(_module, args, kwargs):
            active_layer[0] = index
            mask = kwargs.get("attention_mask")
            phase = kwargs.get("position_embeddings")
            require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and
                    mask.shape == (prepared["inputs"]["input_ids"].shape[0], 1,
                                   prepared["input_width"], prepared["input_width"]),
                    "native full-call attention mask changed")
            require(isinstance(phase, tuple) and len(phase) == 2 and
                    phase[0].shape == phase[1].shape ==
                    (prepared["inputs"]["input_ids"].shape[0], prepared["input_width"], HEAD_DIM),
                    "native destination rotary phases changed")
            text_positions = kwargs.get("position_ids")
            require(isinstance(text_positions, torch.Tensor) and
                    torch.equal(text_positions, prepared["inputs"]["position_ids"][0]) and
                    torch.equal(rotary_positions[0], prepared["inputs"]["position_ids"]),
                    "text attention or rotary position consumer changed")
            require(torch.equal(mask[b, 0, query], expected_row),
                    "actual text consumer mask lost native left padding or causal slots")
            item = captures[index]
            item["donor_cos"] = phase[0][b, donor_start:donor_end].detach().float().cpu().clone()
            item["donor_sin"] = phase[1][b, donor_start:donor_end].detach().float().cpu().clone()
            item["destination_cos"] = phase[0][b, dest_start:dest_end].detach().float().cpu().clone()
            item["destination_sin"] = phase[1][b, dest_start:dest_end].detach().float().cpu().clone()
            item["position_ids_donor"] = rotary_positions[0][:, b, donor_start:donor_end].detach().cpu().clone()
            item["position_ids_destination"] = rotary_positions[0][:, b, dest_start:dest_end].detach().cpu().clone()
            item["position_ids_query"] = rotary_positions[0][:, b, query].detach().cpu().clone()
            checks[index]["layer_attention_consumed"] = True
            checks[index]["mask_hash"] = tensor_hash(mask[b:b + 1, :, query:query + 1])
            checks[index]["mask_shape"] = list(mask.shape)
            checks[index]["mask_dtype"] = str(mask.dtype)
            checks[index]["phase_hashes"] = [tensor_hash(x) for x in phase]
            cache_position = kwargs.get("cache_position")
            require(isinstance(cache_position, torch.Tensor) and
                    torch.equal(cache_position, call_inputs["cache_position"]) and
                    kwargs.get("past_key_values") is cache and cache.get_seq_length(index) == 0,
                    "attention did not consume the fresh empty cache at native slots")
            checks[index]["cache_position"] = cache_position.detach().cpu().tolist()
            checks[index]["cache_empty_at_entry"] = True
        return before

    def make_attention_after(index):
        def after(_module, _args, _kwargs, _output):
            active_layer[0] = None
        return after

    for i, layer in enumerate(text.layers):
        attn = layer.self_attn
        handles.append(attn.register_forward_pre_hook(make_attention_pre(i), with_kwargs=True))
        handles.append(attn.register_forward_hook(make_attention_after(i), with_kwargs=True))

        if i == 1:
            def layer1_before(_module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                require(isinstance(hidden, torch.Tensor), "layer1 incoming residual was not passed")
                captures[1]["incoming_layer1_target"] = hidden[b, query].detach().float().cpu().clone()
            handles.append(layer.register_forward_pre_hook(layer1_before, with_kwargs=True))

        def k_norm(_module, _args, result, *, index=i):
            require(result.ndim == 4 and result.shape[-2:] == (KV_HEADS, HEAD_DIM),
                    "normalized pre-RoPE K layout changed")
            item = captures[index]
            item["donor_pre_k"] = result[b, donor_start:donor_end].detach().float().cpu().clone()
            item["destination_pre_k"] = result[b, dest_start:dest_end].detach().float().cpu().clone()

        def v_proj(_module, _args, result, *, index=i):
            require(result.shape[-1] == KV_HEADS * HEAD_DIM, "native V projection width changed")
            values = result.reshape(result.shape[0], result.shape[1], KV_HEADS, HEAD_DIM)
            item = captures[index]
            item["donor_v"] = values[b, donor_start:donor_end].detach().float().cpu().clone()
            item["destination_v"] = values[b, dest_start:dest_end].detach().float().cpu().clone()

        def o_proj_before(_module, args, *, index=i):
            require(len(args) == 1 and isinstance(args[0], torch.Tensor), "o_proj input changed")
            expected = returned_attention_inputs[index]
            require(expected is not None and torch.equal(args[0][b, query].detach().float().cpu(), expected),
                    "actual o_proj did not consume the selected attention output")
            captures[index]["o_proj_input"] = args[0][b, query].detach().float().cpu().clone()
            checks[index]["o_proj_consumed"] = True
            checks[index]["o_proj_input_hash"] = tensor_hash(args[0][b, query])

        def o_proj_after(_module, _args, result, *, index=i):
            captures[index]["o_proj_output"] = result[b, query].detach().float().cpu().clone()

        handles.extend((attn.k_norm.register_forward_hook(k_norm),
                        attn.v_proj.register_forward_hook(v_proj),
                        attn.o_proj.register_forward_pre_hook(o_proj_before),
                        attn.o_proj.register_forward_hook(o_proj_after)))

    def outer(query_tensor, key, value, *, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
        index = active_layer[0]
        if index is None:
            return original_sdpa(query_tensor, key, value, attn_mask=attn_mask,
                                 dropout_p=dropout_p, is_causal=is_causal,
                                 scale=scale, enable_gqa=enable_gqa)
        require(not enable_gqa and is_causal is False and dropout_p == 0.0,
                "native text SDPA flags changed")
        require(query_tensor.shape == key.shape == value.shape ==
                (4, Q_HEADS, prepared["input_width"], HEAD_DIM) and
                isinstance(attn_mask, torch.Tensor) and attn_mask.dtype == torch.bool,
                "native text SDPA tensors changed")
        full_calls[0] += 1
        state["full_sdpa_calls"] += 1
        if cell != "native":
            state["selected_sdpa_calls"] += 1
        checks[index]["sdpa_calls"] += 1
        capture = captures[index]
        require(all(capture[key_name] is not None for key_name in
                    ("donor_pre_k", "donor_v", "destination_pre_k", "destination_v",
                     "donor_cos", "donor_sin", "destination_cos", "destination_sin")),
                "live source/destination K/V or phases missing at SDPA consumer")
        entry_hashes = {name: tensor_hash(item) for name, item in
                        (("q", query_tensor), ("k", key), ("v", value), ("mask", attn_mask))}
        result, record = selected_sdpa_replacement(
            query_tensor, key, value, attn_mask, native_sdpa=original_sdpa,
            scale=float(scale), mode=cell, target_batch=b, query_position=query,
            prefix_start=dest_start, prefix_end=dest_end,
            donor_pre_k=capture["donor_pre_k"].to(device),
            donor_v=capture["donor_v"].to(device),
            destination_cos=capture["destination_cos"].to(device),
            destination_sin=capture["destination_sin"].to(device),
            wrapper_entry_hashes=entry_hashes)
        capture["sdpa"] = record
        flat = result.transpose(1, 2).reshape(result.shape[0], result.shape[2], -1)
        returned_attention_inputs[index] = flat[b, query].detach().float().cpu().clone()
        checks[index].update({key_name: record[key_name] for key_name in
                              ("full_input_hashes_before", "full_input_hashes_after",
                               "full_inputs_unchanged", "off_target_output_exact",
                               "selected_nonprefix_kv_exact", "selected_q_exact",
                               "selected_mask_exact", "selected_call",
                               "selected_shape", "selected_mask_hash",
                               "selected_k_before_hash", "selected_v_before_hash",
                               "selected_k_used_hash", "selected_v_used_hash")})
        return result

    input_hashes_before = _full_input_hashes(call_inputs)
    try:
        F.scaled_dot_product_attention = outer
        with torch.inference_mode():
            result = model(**call_inputs)
    finally:
        F.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
    input_hashes_after = _full_input_hashes(call_inputs)
    logits = result.logits[:, -1].detach().float().cpu().clone()
    cache_ok = cache.get_seq_length() == prepared["input_width"] and result.past_key_values is cache
    lm_head_exact = (lm_head_output[0] is not None and
                     torch.equal(lm_head_output[0], result.logits.detach().float().cpu()))
    payload_layers = {str(i): _layer_payload(captures[i], cell) for i in range(LAYERS)}
    payload = {"schema": "recurrence_local_prefix_content.cell.v1",
               "image_id": int(case["image_id"]), "cell": cell,
               "full_logits": logits, "layers": payload_layers}
    case_dir = output / "cases" / str(case["image_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = case_dir / f"{cell}.pt"
    torch.save(payload, tensor_path)
    tensor_ref = binding(tensor_path)
    tensor_total = sum(path.stat().st_size for path in output.rglob("*.pt"))
    write(case_dir / f"{cell}-checks.json", {
        "schema": "recurrence_local_prefix_content.checks.v1",
        "image_id": int(case["image_id"]), "cell": cell,
        "model_call_count": model_calls[0], "vision_call_count": vision_calls[0],
        "text_sdpa_call_count": full_calls[0], "layer_count": LAYERS,
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "input_unchanged": input_hashes_before == input_hashes_after,
        "cache_sequence_length": cache.get_seq_length(),
        "fresh_cache_exact": cache_ok,
        "lm_head_output_shape": (None if lm_head_output[0] is None else list(lm_head_output[0].shape)),
        "lm_head_output_hash": (None if lm_head_output[0] is None else tensor_hash(lm_head_output[0])),
        "lm_head_output_equals_returned_logits": lm_head_exact,
        "layers": checks, "tensor_payload": tensor_ref,
        "tensor_payload_bytes_total": tensor_total,
        "elapsed_seconds": time.monotonic() - state["setup_started_monotonic"],
    })
    require(tensor_total <= MAX_TENSOR_BYTES, "tensor payload budget exceeded")
    require(cache_ok and lm_head_exact,
            "fresh full-sequence cache or actual language-model head output check failed")
    require(model_calls[0] == vision_calls[0] == 1 and full_calls[0] == LAYERS and
            all(item["sdpa_calls"] == item["o_proj_consumed"] == 1 for item in checks),
            "production call/consumer counts changed")
    require(input_hashes_before == input_hashes_after and all(
        item["full_inputs_unchanged"] and item["off_target_output_exact"] and
        item["selected_nonprefix_kv_exact"] and item["selected_q_exact"] and
        item["selected_mask_exact"] and
        item["layer_attention_consumed"] and item["o_proj_consumed"]
        for item in checks), "native input/output/consumer identity gate failed")
    require(time.monotonic() - state["setup_started_monotonic"] <= MAX_SECONDS,
            "15-minute model setup/run budget exceeded")
    state["completed_cells"].append(f"{case['image_id']}:{cell}")
    state["tensor_bytes"] = tensor_total
    state["last_cell"] = f"{case['image_id']}:{cell}"
    return {"payload": payload, "tensor_binding": tensor_ref,
            "layers_capture": captures, "checks": checks,
            "logits": logits, "lm_head_output": lm_head_output[0],
            "input_hashes_before": input_hashes_before,
            "input_hashes_after": input_hashes_after}, payload


def _top_readback(logits: torch.Tensor, case: dict) -> dict:
    scores = logits.float()
    top = torch.topk(scores, 10)
    probabilities = torch.softmax(scores.double(), dim=-1)

    def token(item: int) -> dict:
        score = scores[item]
        return {"token": int(item), "logit": float(score),
                "probability": float(probabilities[item]),
                "rank": int((scores > score).sum()) + 1}

    winner = int(top.indices[0])
    return {"winner": winner, "runnerup": int(top.indices[1]),
            "top1_top2_gap": float(top.values[0] - top.values[1]),
            "top10": [{"token": int(t), "logit": float(v)}
                      for t, v in zip(top.indices, top.values, strict=True)],
            "repeat_coordinate": token(int(case["repeat_token"])),
            "native_coordinate": token(int(case["native_token"])),
            "global_winner": token(winner),
            "logsumexp": float(torch.logsumexp(scores.double(), dim=-1))}


def _source_manifest(output: Path, selection: dict, prepared: list[dict],
                     panel_binding: dict, unit_binding: dict, model_contract: dict,
                     source_captures: list[dict]) -> dict:
    cases = []
    for item in prepared:
        case = item["selection"]
        cases.append({"image_id": int(case["image_id"]), "group": case["group"],
                      "batch_index": item["batch_index"], "query": item["query"],
                      "prompt_width": item["prompt_width"], "input_width": item["input_width"],
                      "prefix_count": item["prefix_count"],
                      "donor_raw": item["donor_raw"], "destination_raw": item["dest_raw"],
                      "donor_physical": item["donor_pos"], "destination_physical": item["dest_pos"],
                      "prefix_tokens": case["prefix_tokens"],
                      "trace_crosswalk": item["trace_crosswalk"],
                      "sources": {key: case[key] for key in
                                  ("raw", "trace", "runtime_receipt", "image")},
                      "source_batch_identity": item["batch_identity"],
                      "model_input_hashes": item["input_hashes"],
                      "media_hashes": item["media_hashes"],
                      "position_ids_target": item["position_ids_target"],
                      "cells": [f"cases/{case['image_id']}/{name}.pt" for name in CELL_ORDER]})
    return {"schema": "recurrence_local_prefix_content.source_to_cell.v1",
            "status": "frozen-before-forward", "selection": binding(SELECTION),
            "unit": unit_binding, "panel": panel_binding,
            "event_selection_order": selection["event_selection_order"],
            "condition": selection["condition"], "model": model_contract,
            "producer": binding(Path(__file__)),
            "producer_capture": next(x["capture"] for x in source_captures
                                      if x["archive_name"] == "recurrence_local_prefix_content.py"),
            "source_captures": source_captures,
            "cell_order": list(CELL_ORDER), "cases": cases,
            "budget": {"model_forwards": 9, "vision_forwards": 9,
                       "tensor_bytes": MAX_TENSOR_BYTES, "wall_seconds": MAX_SECONDS,
                       "expected_full_sdpa_calls": 252,
                       "expected_selected_sdpa_calls": 168}}


def _native_trace_gate(logits: torch.Tensor, case: dict) -> dict:
    target_logits = logits[int(case["batch_index"])]
    top = torch.topk(target_logits, 2)
    saved = case["saved_trace"]
    lse = float(torch.logsumexp(target_logits.double(), dim=-1))
    record = {"winner": int(top.indices[0]), "runnerup": int(top.indices[1]),
              "top2": [float(x) for x in top.values], "logsumexp": lse,
              "winner_matches": int(top.indices[0]) == int(saved["winner"]),
              "runnerup_matches": int(top.indices[1]) == int(saved["runnerup"]),
              "top2_max_abs": max(abs(float(a) - float(b))
                                  for a, b in zip(top.values, saved["top2"], strict=True)),
              "logsumexp_abs": abs(lse - float(saved["logsumexp"]))}
    return record


def _case_run(prepared: dict, q, output: Path, state: dict) -> dict:
    case = prepared["selection"]
    native_result, native_payload = _run_cell(prepared, q, "native", output, state)
    native_gate = _native_trace_gate(native_result["logits"], case)
    write(output / "cases" / str(case["image_id"]) / "native-gate.json",
          {"schema": "recurrence_local_prefix_content.native_gate.v1",
           "trace_gate": native_gate,
           "tensor_payload": native_result["tensor_binding"],
           "status": ("native-trace-qualified" if native_gate["winner_matches"] and
                      native_gate["runnerup_matches"] and native_gate["top2_max_abs"] <= ATOL and
                      native_gate["logsumexp_abs"] <= ATOL else "native-trace-failed")})
    require(native_gate["winner_matches"] and native_gate["runnerup_matches"] and
            native_gate["top2_max_abs"] <= ATOL and native_gate["logsumexp_abs"] <= ATOL,
            "fresh native logits do not reproduce saved global trace")
    write(output / "receipt.json", state)

    identity_result, identity_payload = _run_cell(prepared, q, "identity", output, state)
    identity_error = float((identity_result["logits"] - native_result["logits"]).abs().max())
    companion_indices = [i for i in range(native_result["logits"].shape[0])
                         if i != prepared["batch_index"]]
    require(torch.equal(identity_result["logits"][companion_indices],
                        native_result["logits"][companion_indices]),
            "identity recomputation changed a companion batch row")
    identity_winner = int(identity_result["logits"][prepared["batch_index"]].argmax())
    identity_gate = {"full_batch_max_abs": identity_error,
                     "target_winner": identity_winner,
                     "native_target_winner": int(native_result["logits"][prepared["batch_index"]].argmax()),
                     "target_winner_matches": identity_winner == int(native_result["logits"][prepared["batch_index"]].argmax())}
    write(output / "cases" / str(case["image_id"]) / "identity-gate.json",
          {"status": ("identity-qualified" if identity_error <= ATOL and
                      identity_gate["target_winner_matches"] else "identity-failed"),
           "gate": identity_gate, "tensor_payload": identity_result["tensor_binding"]})
    require(identity_error <= ATOL and identity_gate["target_winner_matches"],
            "identity recomputation failed fresh-native full-logit parity")
    ref_layers = native_result["layers_capture"]
    for index, (native_layer, identity_layer) in enumerate(zip(ref_layers, identity_result["layers_capture"], strict=True)):
        require(all(torch.equal(native_layer[key], identity_layer[key]) for key in
                    ("donor_pre_k", "donor_v", "destination_pre_k", "destination_v",
                     "donor_cos", "donor_sin", "destination_cos", "destination_sin")),
                f"identity cell changed causal source/destination content or phases at layer {index}")
    _check_layer0_and_layer1(native_result, identity_result, case)
    write(output / "receipt.json", state)

    donor_result, donor_payload = _run_cell(prepared, q, "donor_content", output, state)
    require(torch.equal(donor_result["logits"][companion_indices],
                        native_result["logits"][companion_indices]),
            "donor treatment changed a companion batch row")
    donor_logits = donor_result["logits"][prepared["batch_index"]]
    donor_summary = _top_readback(donor_logits, case)
    donor_summary["restores_repeated_coordinate"] = (
        donor_summary["winner"] == int(case["repeat_token"]) and
        donor_summary["top1_top2_gap"] > 0.001)
    donor_summary["inconclusive_near_tie"] = donor_summary["top1_top2_gap"] <= 0.001
    for index, (native_layer, donor_layer) in enumerate(zip(ref_layers, donor_result["layers_capture"], strict=True)):
        require(all(torch.equal(native_layer[key], donor_layer[key]) for key in
                    ("donor_pre_k", "donor_v", "destination_pre_k", "destination_v",
                     "donor_cos", "donor_sin", "destination_cos", "destination_sin")),
                f"donor treatment changed pre-query source/destination content or phases at layer {index}")
    _check_layer0_and_layer1(native_result, donor_result, case)
    case_result = {"schema": "recurrence_local_prefix_content.case_result.v1",
                   "image_id": int(case["image_id"]), "role": case["role"],
                   "native_trace_gate": native_gate, "identity_gate": identity_gate,
                   "native": _top_readback(native_result["logits"][prepared["batch_index"]], case),
                   "donor_content": donor_summary,
                   "restores_repeated_coordinate": donor_summary["restores_repeated_coordinate"],
                   "tensor_payloads": {"native": native_result["tensor_binding"],
                                       "identity": identity_result["tensor_binding"],
                                       "donor_content": donor_result["tensor_binding"]},
                   "status": "candidate"}
    write(output / "cases" / str(case["image_id"]) / "result.json", case_result)
    state["case_results"].append({"image_id": int(case["image_id"]),
                                  "winner": donor_summary["winner"],
                                  "repeat_token": int(case["repeat_token"]),
                                  "gap": donor_summary["top1_top2_gap"],
                                  "restores_repeated_coordinate": donor_summary["restores_repeated_coordinate"]})
    if len(state["case_results"]) == 1:
        state["status"] = "first-case-qualified"
        write(output / "receipt.json", state)
        print(json.dumps({"milestone": "first-case-qualified", **state["case_results"][0]}), flush=True)
    return case_result


def _check_layer0_and_layer1(native_result: dict, treatment_result: dict, case: dict) -> None:
    native_layers = native_result["layers_capture"]
    treatment_layers = treatment_result["layers_capture"]
    require(torch.equal(native_layers[0]["donor_pre_k"], native_layers[0]["destination_pre_k"]) and
            torch.equal(native_layers[0]["donor_v"], native_layers[0]["destination_v"]),
            f"{case['image_id']} layer0 same-token donor/destination K/V differ")
    require(torch.equal(treatment_layers[0]["donor_pre_k"], treatment_layers[0]["destination_pre_k"]) and
            torch.equal(treatment_layers[0]["donor_v"], treatment_layers[0]["destination_v"]),
            f"{case['image_id']} treatment layer0 same-token donor/destination K/V differ")
    native_layer1 = native_layers[1].get("incoming_layer1_target")
    treatment_layer1 = treatment_layers[1].get("incoming_layer1_target")
    require(isinstance(native_layer1, torch.Tensor) and isinstance(treatment_layer1, torch.Tensor) and
            float((native_layer1 - treatment_layer1).abs().max()) <= ATOL,
            f"{case['image_id']} incoming layer1 target differs from native")


def run(device: str = "cuda:4") -> None:
    require(device.startswith("cuda") and torch.cuda.is_available(), "authorized CUDA device unavailable")
    require(SELECTION.exists(), "root-frozen selection.json is missing")
    selection = json.loads(SELECTION.read_text())
    require(selection.get("schema_version") == 1 and
            selection.get("unit_id") == "2026-09-23-recurrence-local-prefix-content-transfer" and
            selection.get("condition") == "untied-original" and len(selection.get("cases", [])) == 3,
            "frozen source selection contract changed")
    cpu_report = cpu_selfcheck()
    require(cpu_report["status"] == "cpu-qualified", "CPU caller qualification failed")
    output_root = Path(selection["output_root"])
    output = output_root / "attempt-001"
    require(not output.exists(), "attempt-001 already exists; preserve failed evidence")
    output.mkdir(parents=True)
    state = {"schema": "recurrence_local_prefix_content.receipt.v1", "status": "preparing",
             "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "full_sdpa_calls": 0, "selected_sdpa_calls": 0,
             "tensor_bytes": 0, "cell_order": list(CELL_ORDER), "completed_cells": [],
             "case_results": [], "started_unix": time.time(),
             "cpu_selfcheck": cpu_report,
             "setup_started_monotonic": time.monotonic()}
    write(output / "receipt.json", state)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        model_contract = _model_contract(q, identity, selection,
                                         [json.loads(Path(c["runtime_receipt"]["path"]).read_text())
                                          for c in selection["cases"]])
        panel_binding = verify_binding(selection["panel"])
        event_binding = verify_binding(selection["event_selection_order"])
        panel = json.loads(Path(panel_binding["path"]).read_text())
        require(binding(untied_shared.ROOT / "panel.json")["sha256"] == panel_binding["sha256"],
                "shared untied loader panel differs from root selection")
        prepared = [_prepared_case(case, panel, q, device) for case in selection["cases"]]
        source_paths = [
            (Path(__file__), "recurrence_local_prefix_content.py"),
            (UNIT, "experiment/unit.md"), (SELECTION, "experiment/selection.json"),
            (Path(panel_binding["path"]), "source/panel.json"),
            (Path(inspect.getfile(build_bound_native_requests)), "src/inference/bound_requests.py"),
            (Path(inspect.getfile(exact_history_inputs)), "src/qwen/native.py"),
            (Path(inspect.getfile(_prefix_tokens)),
             "probes/training_set_completion/numerical_feedback/runtime.py"),
            (Path(untied_shared.__file__), "probes/training_set_completion/untied_shared.py"),
            (Path(inspect.getfile(input_identity)), "src/qwen/input_identity.py"),
            (Path(probe_artifacts.__file__), "probes/training_set_completion/artifacts.py"),
            (Path(source_provenance.__file__), "src/artifacts/source_provenance.py"),
            (Path(dora_runtime.__file__), "src/adapters/dora.py"),
            (Path(qwen_runtime_loading.__file__), "src/qwen/runtime_loading.py"),
            (Path(untied_embeddings.__file__), "src/qwen/untied_embeddings.py"),
            (Path(inspect.getfile(modeling_qwen3_vl)), "transformers/modeling_qwen3_vl.py"),
            (Path(inspect.getfile(sdpa_attention_forward)), "transformers/sdpa_attention.py"),
        ]
        captures = _capture_sources(output, source_paths)
        source_to_cell = _source_manifest(output, selection, prepared, panel_binding,
                                          binding(UNIT), model_contract, captures)
        source_to_cell["event_selection_order_verified"] = event_binding
        write(output / "source-to-cell.json", source_to_cell)
        state["status"] = "running"
        state["source_to_cell"] = binding(output / "source-to-cell.json")
        write(output / "receipt.json", state)
        results = []
        for item in prepared:
            result = _case_run(item, q, output, state)
            results.append(result)
            write(output / "receipt.json", state)
        restoration_count = sum(item["restores_repeated_coordinate"] for item in results)
        require(state["model_forwards"] == state["vision_forwards"] == 9 and
                state["full_sdpa_calls"] == 252 and state["selected_sdpa_calls"] == 168,
                "experiment call budget differs from the frozen nine-cell plan")
        state["status"] = "candidate-complete"
        state["restoration_count"] = restoration_count
        state["case_count"] = len(results)
        state["prediction_passed"] = restoration_count >= 2
        state["finished_unix"] = time.time()
        state["elapsed_seconds"] = time.monotonic() - state["setup_started_monotonic"]
        state.pop("setup_started_monotonic", None)
        write(output / "result.json", {"schema": "recurrence_local_prefix_content.result.v1",
                                        "status": "candidate", "restoration_count": restoration_count,
                                        "case_count": len(results),
                                        "prediction_passed": restoration_count >= 2,
                                        "cases": results,
                                        "source_to_cell": state["source_to_cell"]})
        write(output / "receipt.json", state)
    except Exception as exc:
        state["status"] = "technical-failure"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        state["finished_unix"] = time.time()
        state["elapsed_seconds"] = time.monotonic() - state["setup_started_monotonic"]
        state.pop("setup_started_monotonic", None)
        write(output / "receipt.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-selfcheck", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda:4")
    args = parser.parse_args()
    result = cpu_selfcheck() if args.cpu_selfcheck else run(args.device)
    if result is not None:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
