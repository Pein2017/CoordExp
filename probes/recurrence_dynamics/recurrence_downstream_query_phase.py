"""Matched incoming-state and downstream query-phase transfer at native SDPA."""
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
from transformers import DynamicCache
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from probes.recurrence_dynamics import recurrence_native_trajectory as trajectory
from src.artifacts.utf8_json import literal_binding
from probes.recurrence_dynamics.recurrence_donor_tracking import require, write
from probes.recurrence_dynamics.recurrence_first_layer_readout import install_o_proj_patch_observer
from probes.model_profiles.mature_tied_untied import load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-22-recurrence-downstream-query-phase"
OUT = ROOT / "attempt-002"
RULING = Path("research/experiments/2026-09-22-recurrence-downstream-query-phase/repair-ruling-01.md")
UNIT = Path("research/experiments/2026-09-22-recurrence-downstream-query-phase/unit.md")
SELECTION = ROOT / "selection.json"
ORACLE = trajectory.ORACLE
GROUPS_ROOT = BASE / "2026-09-22-recurrence-first-layer-groups"
GROUPS_ACCEPTANCE = GROUPS_ROOT / "lead-acceptance.json"
GROUPS_CAPTURE = GROUPS_ROOT / "attempt-001/native.pt"
REVERSE_ROOT = BASE / "2026-09-22-recurrence-first-layer-reverse-transfer"
REVERSE_ACCEPTANCE = REVERSE_ROOT / "lead-acceptance.json"
REVERSE_LATE = REVERSE_ROOT / "attempt-001/late.pt"
TRAJECTORY_ROOT = BASE / "2026-09-22-recurrence-native-trajectory"
TRAJECTORY_ACCEPTANCE = TRAJECTORY_ROOT / "lead-acceptance.json"
TRAJECTORY_TENSOR = TRAJECTORY_ROOT / "attempt-002/val-trajectory.pt"
TRAJECTORY_READBACK = TRAJECTORY_ROOT / "attempt-002/val-readback.json"

TARGET, FULL_WIDTH = 2, 2127
RECIPIENT_POSITION, DONOR_POSITION = 1703, 2126
RECIPIENT_QUERY_INDEX, DONOR_QUERY_INDEX = 15, 62
VOCAB = 152670
Q_HEADS, KV_HEADS, HEAD_DIM, HIDDEN, LAYERS = 16, 8, 128, 2048, 28
Q_TO_KV = Q_HEADS // KV_HEADS
P38, P999 = 151708, 152669
CELL_ORDER = ("native", "held_native_phase", "held_late_phase")
PHASE_LAYERS = tuple(range(1, LAYERS))
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


def rotate_fp32(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    require(value.ndim == 2 and value.shape[-1] == cos.shape[-1] == sin.shape[-1],
            "phase rotation vector shape changed")
    half = value.shape[-1] // 2
    rotated_half = torch.cat((-value[:, half:], value[:, :half]), dim=-1)
    return value * cos.unsqueeze(0) + rotated_half * sin.unsqueeze(0)


def rotate_fp64(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Independent real/imaginary-half RoPE check, separate from producer rotate."""
    require(value.ndim == 2 and value.shape[-1] == cos.shape[-1] == sin.shape[-1],
            "independent phase vector shape changed")
    half = value.shape[-1] // 2
    a, b = value[:, :half].double(), value[:, half:].double()
    c, s = cos[:half].double(), sin[:half].double()
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1)


def source_crosswalk(selection: dict) -> dict:
    oracle = json.loads(ORACLE.read_text())
    require(oracle["status"] == "root-independent-source-oracle", "source oracle status changed")
    raw_path = Path(oracle["cases"]["val"]["source_bindings"]["raw"]["path"])
    trace_path = Path(oracle["cases"]["val"]["source_bindings"]["trace"]["path"])
    raw = json.loads(raw_path.read_text())["rows"]
    trace = json.loads(trace_path.read_text())
    tokens = raw[TARGET]["token_ids"]
    base = FULL_WIDTH - 807
    fake_ids = torch.zeros((4, FULL_WIDTH), dtype=torch.long)
    fake_ids[TARGET, base:base + 807] = torch.tensor(tokens[:807])
    queries = trajectory.source_queries(
        "val", {"raw_action_offset": 807}, oracle, raw, trace, {"input_ids": fake_ids})
    by_offset = {item["raw_action_offset"]: item for item in queries}
    recipient, donor = by_offset[384], by_offset[807]
    require(recipient["physical_query_index"] == RECIPIENT_POSITION and
            recipient["next_token"] == P38 and recipient["query_input_token"] == 152241 and
            donor["physical_query_index"] == DONOR_POSITION and donor["next_token"] == P999 and
            donor["query_input_token"] == 152241 and
            recipient["minus_one_action_token"] == donor["minus_one_action_token"] == 152241,
            "raw action crosswalk changed")
    require(recipient["physical_query_index"] == selection["recipient"]["physical_query_index"] and
            donor["physical_query_index"] == selection["donor"]["physical_query_index"],
            "selection/oracle query crosswalk changed")
    return {"recipient": recipient, "donor": donor,
            "source_bindings": {"raw": binding(raw_path), "trace": binding(trace_path)}}


def load_sources() -> dict:
    selection = json.loads(SELECTION.read_text())
    require(selection["status"] == "root-frozen" and selection["target"] == TARGET and
            selection["full_width"] == FULL_WIDTH and
            selection["query_positions"] == [RECIPIENT_POSITION, DONOR_POSITION] and
            selection["cell_order"] == list(CELL_ORDER), "downstream phase selection changed")
    require(selection["phase_layers"] == list(PHASE_LAYERS), "phase layer scope changed")
    for expected in selection["sources"].values():
        assert_binding(Path(expected["path"]), expected)
    for path, expected_status in ((GROUPS_ACCEPTANCE, "lead-accepted"),
                                   (REVERSE_ACCEPTANCE, "lead-accepted"),
                                   (TRAJECTORY_ACCEPTANCE, "lead-accepted")):
        require(json.loads(path.read_text())["status"] == expected_status,
                f"accepted source status changed: {path}")

    native_capture = torch.load(GROUPS_CAPTURE, map_location="cpu", weights_only=True)
    require(native_capture["full_logits"].shape == (2, VOCAB) and
            native_capture["head_output"].shape == (2, Q_HEADS, HEAD_DIM) and
            native_capture["query_cos"].shape == native_capture["query_sin"].shape == (2, HEAD_DIM) and
            native_capture["query_positions"].tolist() == [RECIPIENT_POSITION, DONOR_POSITION],
            "groups native capture shape changed")
    early_head = native_capture["head_output"][0].float().reshape(-1).contiguous()
    late_head = native_capture["head_output"][1].float().reshape(-1).contiguous()
    require(tensor_hash(early_head) == selection["head_hashes"]["old_sham"] and
            tensor_hash(late_head) == selection["head_hashes"]["late"] and
            tensor_hash(native_capture["query_cos"][0]) == selection["phase_hashes"]["early"]["query_cos"] and
            tensor_hash(native_capture["query_sin"][0]) == selection["phase_hashes"]["early"]["query_sin"] and
            tensor_hash(native_capture["query_cos"][1]) == selection["phase_hashes"]["late"]["query_cos"] and
            tensor_hash(native_capture["query_sin"][1]) == selection["phase_hashes"]["late"]["query_sin"],
            "accepted head or phase hash changed")

    reverse_late = torch.load(REVERSE_LATE, map_location="cpu", weights_only=True)
    require(reverse_late["logits"].shape == (4, VOCAB), "reverse late logits shape changed")
    trajectory_capture = torch.load(TRAJECTORY_TENSOR, map_location="cpu", weights_only=True)
    require(trajectory_capture["query_positions"].shape == (63,) and
            int(trajectory_capture["query_positions"][RECIPIENT_QUERY_INDEX]) == RECIPIENT_POSITION and
            int(trajectory_capture["query_positions"][DONOR_QUERY_INDEX]) == DONOR_POSITION and
            len(trajectory_capture["residual_input"]) == LAYERS and
            trajectory_capture["residual_input"][1].shape == (63, HIDDEN),
            "native trajectory incoming-state source changed")
    early_input = trajectory_capture["residual_input"][1][RECIPIENT_QUERY_INDEX].float().contiguous()
    late_input = trajectory_capture["residual_input"][1][DONOR_QUERY_INDEX].float().contiguous()
    require(tensor_hash(early_input) == selection["layer1_input_hashes"]["early"] and
            tensor_hash(late_input) == selection["layer1_input_hashes"]["late"],
            "layer1 incoming-state hashes changed")
    refs = {key: expected for key, expected in selection["sources"].items()}
    refs.update({"selection": binding(SELECTION), "groups_readback": binding(GROUPS_ROOT / "attempt-001/readback.json"),
                 "trajectory_readback": binding(TRAJECTORY_READBACK)})
    return {
        "selection": selection, "refs": refs, "native_capture": native_capture,
        "accepted_logits": native_capture["full_logits"].float().clone(),
        "early_head": early_head, "late_head": late_head,
        "early_cos": native_capture["query_cos"][0].float().clone(),
        "early_sin": native_capture["query_sin"][0].float().clone(),
        "late_cos": native_capture["query_cos"][1].float().clone(),
        "late_sin": native_capture["query_sin"][1].float().clone(),
        "reverse_late_logits": reverse_late["logits"].float().clone(),
        "early_input": early_input, "late_input": late_input,
        "trajectory_capture": trajectory_capture,
        "crosswalk": source_crosswalk(selection),
    }


def patch_phase_inputs(query: torch.Tensor, key: torch.Tensor, pre_q: torch.Tensor,
                      pre_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      *, target: int, patch_position: int, expected_position: int,
                      q_heads: int = Q_HEADS, kv_heads: int = KV_HEADS,
                      seq_len: int = FULL_WIDTH, head_dim: int = HEAD_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    require(patch_position == expected_position, "query/key phase patch position changed")
    require(query.ndim == key.ndim == 4 and query.shape[0] == key.shape[0] and
            query.shape[1:] == (q_heads, seq_len, head_dim) and
            key.shape[1:] == (kv_heads, seq_len, head_dim) and
            pre_q.shape == (q_heads, head_dim) and pre_k.shape == (kv_heads, head_dim),
            "live SDPA Q/K shape changed")
    q_replacement = rotate_fp32(pre_q, cos.to(device=pre_q.device, dtype=pre_q.dtype),
                                sin.to(device=pre_q.device, dtype=pre_q.dtype))
    k_replacement = rotate_fp32(pre_k, cos.to(device=pre_k.device, dtype=pre_k.dtype),
                                sin.to(device=pre_k.device, dtype=pre_k.dtype))
    patched_q, patched_k = query.clone(), key.clone()
    patched_q[target, :, patch_position, :] = q_replacement
    patched_k[target, :, patch_position, :] = k_replacement
    q_restored, k_restored = patched_q.clone(), patched_k.clone()
    q_restored[target, :, patch_position, :] = query[target, :, patch_position, :]
    k_restored[target, :, patch_position, :] = key[target, :, patch_position, :]
    require(torch.equal(q_restored, query) and torch.equal(k_restored, key),
            "phase patch changed an off-target Q/K entry")
    return patched_q, patched_k


def check_sdpa_arguments(query, key, value, mask, *, expected_query, expected_key,
                         expected_value, expected_mask):
    """Check actual kernel arguments against the independently supplied call contract."""
    groups = expected_query.shape[1] // expected_key.shape[1]
    require(expected_query.shape[1] == groups * expected_key.shape[1], "GQA head ratio changed")
    require(torch.equal(query, expected_query), "actual SDPA query differs from declared query")
    require(torch.equal(key, expected_key.repeat_interleave(groups, dim=1)) and
            torch.equal(value, expected_value.repeat_interleave(groups, dim=1)),
            "actual SDPA K/V differ from declared expanded K/V")
    require(torch.equal(mask, expected_mask), "actual SDPA mask changed")


def cpu_sdpa_selfcheck() -> None:
    """Real integration/F.sdpa path, with an independent patch oracle and mutations."""
    batch, seq, q_heads, kv_heads, dim = 3, 5, 4, 2, 6
    target, position = 2, 3
    query = torch.arange(batch*q_heads*seq*dim, dtype=torch.float32).reshape(batch,q_heads,seq,dim)/300
    key = torch.arange(batch*kv_heads*seq*dim, dtype=torch.float32).reshape(batch,kv_heads,seq,dim)/200
    value = key + 0.3
    pre_q, pre_k = query[target,:,position].clone(), key[target,:,position].clone()
    angle = torch.tensor([0.2,-0.4,0.7])
    cos, sin = angle.cos().repeat(2), angle.sin().repeat(2)
    mask = torch.tril(torch.ones(batch,1,seq,seq,dtype=torch.bool))
    expected_q, expected_k = query.clone(), key.clone()
    # Explicit paired coordinates, independent of patch_phase_inputs/rotate_fp32.
    for original, expected in ((pre_q,expected_q),(pre_k,expected_k)):
        for pair in range(dim//2):
            a,b = original[:,pair],original[:,pair+dim//2]
            expected[target,:,position,pair] = a*cos[pair]-b*sin[pair]
            expected[target,:,position,pair+dim//2] = b*cos[pair]+a*sin[pair]
    original_sdpa = F.scaled_dot_product_attention
    module = torch.nn.Module()
    module.num_key_value_groups = q_heads//kv_heads
    seen = []

    def spy(q,k,v,*,attn_mask,dropout_p,is_causal,scale,enable_gqa=False):
        seen.append(True)
        require(not enable_gqa, "CPU masked route must expand GQA")
        check_sdpa_arguments(q,k,v,attn_mask,expected_query=expected_q,expected_key=expected_k,
                             expected_value=value,expected_mask=mask)
        return original_sdpa(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,
                             is_causal=is_causal,scale=scale)

    try:
        F.scaled_dot_product_attention = spy
        for mutation in ('correct','wrong_query','wrong_sine'):
            patch_at = position-1 if mutation=='wrong_query' else position
            use_sin = -sin if mutation=='wrong_sine' else sin
            pq,pk = patch_phase_inputs(query,key,pre_q,pre_k,cos,use_sin,target=target,
                                      patch_position=patch_at,expected_position=patch_at,
                                      q_heads=q_heads,kv_heads=kv_heads,seq_len=seq,head_dim=dim)
            before = len(seen)
            try:
                observed,_ = sdpa_attention_forward(module,pq,pk,value,mask,scaling=dim**-0.5)
            except ValueError:
                require(mutation!='correct' and len(seen)==before+1,
                        "mutation failed before reaching actual SDPA observer")
            else:
                require(mutation=='correct', f"{mutation} escaped actual consumer")
                expected = original_sdpa(expected_q,expected_k.repeat_interleave(2,dim=1),
                                         value.repeat_interleave(2,dim=1),attn_mask=mask,
                                         dropout_p=0.0,is_causal=False,scale=dim**-0.5)
                require(torch.equal(observed,expected.transpose(1,2).contiguous()),
                        "independent SDPA output mismatch")
        require(len(seen)==3, "actual consumer mutation call count changed")
    finally:
        F.scaled_dot_product_attention = original_sdpa


def cpu_selfcheck() -> dict:
    sources = load_sources()
    cpu_sdpa_selfcheck()
    print(json.dumps({
        "status": "selfcheck_ok", "transformers": transformers.__version__,
        "recipient": sources["crosswalk"]["recipient"], "donor": sources["crosswalk"]["donor"],
        "phase_layers": list(PHASE_LAYERS),
    }))
    return sources


def target_summary(logits: torch.Tensor, label: str, native: torch.Tensor | None) -> dict:
    target_logits = logits[TARGET]
    out = {"cell": label, "queries": []}
    for query_index, position in enumerate((RECIPIENT_POSITION, DONOR_POSITION)):
        values, indices = torch.topk(target_logits[query_index], 5)
        probs = torch.softmax(target_logits[query_index].double(), dim=0)
        row = {
            "physical_query_index": position,
            "top5": [{"token": int(token), "logit": float(value)}
                     for token, value in zip(indices, values, strict=True)],
            "winner": int(indices[0]), "top1_top2_gap": float(values[0] - values[1]),
            "p38": float(probs[P38]), "p999": float(probs[P999]),
            "d_z38_minus_z999": float(target_logits[query_index, P38] - target_logits[query_index, P999]),
        }
        if native is not None:
            row["max_abs_from_native_target"] = float((target_logits[query_index] - native[TARGET, query_index]).abs().max())
        out["queries"].append(row)
    if native is not None:
        out["companion_max_abs_from_native"] = float((logits[[0, 1, 3]] - native[[0, 1, 3]]).abs().max())
    return out


def collect_cell(label: str, sources: dict, q, identity: dict, device: str,
                 state: dict, native_logits: torch.Tensor | None) -> tuple[dict, torch.Tensor]:
    selection = sources["selection"]
    case, _meta, queries, _raw, _trace = trajectory.source_inputs("val", q, identity,
                                                                   json.loads(ORACLE.read_text()), device)
    native = case["native"]
    require(case["target"] == TARGET and native["input_ids"].shape == (4, FULL_WIDTH),
            "native input shape changed")
    query_indices = torch.tensor([RECIPIENT_POSITION, DONOR_POSITION], device=device, dtype=torch.long)
    inputs = dict(native)
    cache = DynamicCache()
    inputs.update(logits_to_keep=query_indices, past_key_values=cache,
                  cache_position=torch.arange(FULL_WIDTH, device=device), use_cache=True)
    text = q.model.model.language_model
    ctx = {
        "cell": label, "phase_mode": None if label == "native" else
        ("early" if label == "held_native_phase" else "late"),
        "pre_q_device": {}, "pre_k_device": {}, "pre_q": {}, "pre_k": {},
        "phases": {}, "f_records": {}, "layer1_input": None,
        "attention": None, "active": None, "layer0_o_proj": {},
    }
    phase_source = {
        "early": (sources["early_cos"], sources["early_sin"]),
        "late": (sources["late_cos"], sources["late_sin"]),
    }
    all_handles = []
    expected_layer0 = sources["early_head"] if label == "native" else sources["late_head"]
    replacement_layer0 = None if label == "native" else sources["late_head"].reshape(Q_HEADS, HEAD_DIM)
    all_handles.extend(install_o_proj_patch_observer(
        text.layers[0].self_attn.o_proj, replacement_layer0, expected_flat=expected_layer0,
        patch_batch=TARGET, patch_position=RECIPIENT_POSITION,
        observe_batch=TARGET, observe_position=RECIPIENT_POSITION,
        heads=Q_HEADS, head_dim=HEAD_DIM, state=ctx["layer0_o_proj"]))
    observers, observer_handles = trajectory.attention_attestors(
        text, q.model.lm_head, inputs, cache, TARGET, query_indices, FULL_WIDTH)
    ctx["attention"] = observers
    all_handles.extend(observer_handles)

    def phase_observer(_module, _args, kwargs, *, index):
        phase = kwargs.get("position_embeddings")
        require(isinstance(phase, tuple) and len(phase) == 2 and
                phase[0].shape == phase[1].shape == (4, FULL_WIDTH, HEAD_DIM),
                "actual attention phase shape changed")
        ctx["phases"][index] = {
            "recipient_cos": phase[0][TARGET, RECIPIENT_POSITION].detach().float().cpu().clone(),
            "recipient_sin": phase[1][TARGET, RECIPIENT_POSITION].detach().float().cpu().clone(),
            "donor_cos": phase[0][TARGET, DONOR_POSITION].detach().float().cpu().clone(),
            "donor_sin": phase[1][TARGET, DONOR_POSITION].detach().float().cpu().clone(),
        }

    for index, layer in enumerate(text.layers):
        all_handles.append(layer.self_attn.register_forward_pre_hook(
            lambda module, args, kwargs, index=index: phase_observer(module, args, kwargs, index=index),
            with_kwargs=True))

    for index in PHASE_LAYERS:
        layer = text.layers[index]

        def q_norm(_module, _args, output, *, index=index):
            require(output.shape == (4, FULL_WIDTH, Q_HEADS, HEAD_DIM), "live pre-Q shape changed")
            ctx["pre_q_device"][index] = output[TARGET, RECIPIENT_POSITION].detach().clone()
            ctx["pre_q"][index] = output[TARGET, RECIPIENT_POSITION].detach().float().cpu().clone()

        def k_norm(_module, _args, output, *, index=index):
            require(output.shape == (4, FULL_WIDTH, KV_HEADS, HEAD_DIM), "live pre-K shape changed")
            ctx["pre_k_device"][index] = output[TARGET, RECIPIENT_POSITION].detach().clone()
            ctx["pre_k"][index] = output[TARGET, RECIPIENT_POSITION].detach().float().cpu().clone()

        all_handles.extend((layer.self_attn.q_norm.register_forward_hook(q_norm),
                            layer.self_attn.k_norm.register_forward_hook(k_norm)))

    def layer1_input(_module, args):
        x = args[0]
        require(x.shape == (4, FULL_WIDTH, HIDDEN), "layer1 incoming residual shape changed")
        ctx["layer1_input"] = x[TARGET].index_select(0, query_indices).detach().float().cpu().clone()

    all_handles.append(text.layers[1].register_forward_pre_hook(layer1_input))

    def output_consumer(_module, args, *, index):
        value = args[0]
        require(value.shape == (4, FULL_WIDTH, HIDDEN), "actual o_proj input shape changed")
        record = ctx["f_records"].get(index)
        require(record is not None and record.get("f_output") is not None,
                "SDPA observer output did not reach o_proj")
        expected = record.pop("f_output").transpose(1, 2).contiguous().reshape(4, FULL_WIDTH, HIDDEN)
        if index == 0 and replacement_layer0 is not None:
            expected = expected.clone()
            expected[TARGET, RECIPIENT_POSITION] = expected_layer0.to(expected)
        require(torch.equal(value, expected), "actual SDPA output changed before o_proj consumer")
        record["o_proj_consumer_exact"] = True

    for index, layer in enumerate(text.layers):
        all_handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, args, index=index: output_consumer(module, args, index=index)))

    original_registry = modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS["sdpa"]
    original_sdpa = F.scaled_dot_product_attention

    def observed_sdpa(query, key, value, *, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None, enable_gqa=False):
        active = ctx.get("active")
        if active is None:
            return original_sdpa(query, key, value, attn_mask=attn_mask,
                                 dropout_p=dropout_p, is_causal=is_causal,
                                 scale=scale, enable_gqa=enable_gqa)
        index = active["layer"]
        require(active["module"] is text.layers[index].self_attn and
                query.shape == (4, Q_HEADS, FULL_WIDTH, HEAD_DIM) and
                key.shape == value.shape == (4, Q_HEADS, FULL_WIDTH, HEAD_DIM) and
                enable_gqa is False, "actual post-GQA SDPA consumer shape changed")
        require(isinstance(attn_mask, torch.Tensor) and attn_mask.dtype == torch.bool and
                attn_mask.shape == (4, 1, FULL_WIDTH, FULL_WIDTH),
                "actual SDPA mask changed before consumption")
        record = ctx["f_records"][index]
        check_sdpa_arguments(query, key, value, attn_mask, expected_query=active["query"],
                             expected_key=active["key"], expected_value=active["value"],
                             expected_mask=active["mask"])
        attention_record = observers["attention"][index]
        require(tensor_hash(attn_mask) == attention_record["mask_hash"],
                "actual SDPA consumer mask differs from native attention mask")
        record.update(
            f_calls=int(record.get("f_calls", 0)) + 1,
            consumed_q_target=query[TARGET, :, RECIPIENT_POSITION].detach().float().cpu().clone(),
            consumed_k_target=key[TARGET, :, RECIPIENT_POSITION].detach().float().cpu().clone(),
            consumed_v_target=value[TARGET, :, RECIPIENT_POSITION].detach().float().cpu().clone(),
            historical_k_hash=tensor_hash(key[TARGET, :, :RECIPIENT_POSITION, :]),
            historical_v_hash=tensor_hash(value[TARGET, :, :RECIPIENT_POSITION, :]),
            mask_hash=tensor_hash(attn_mask),
            cache_gqa_heads=int(key.shape[1]), gqa_repetitions=Q_TO_KV,
            gqa_expansion_exact=True,
        )
        raw_output = original_sdpa(query, key, value, attn_mask=attn_mask,
                                   dropout_p=dropout_p, is_causal=is_causal,
                                   scale=scale, enable_gqa=enable_gqa)
        record["f_output"] = raw_output
        return raw_output

    text_indices = {id(layer.self_attn): i for i, layer in enumerate(text.layers)}

    def registry_wrapper(module, query, key, value, attention_mask, dropout=0.0,
                         scaling=None, is_causal=None, **kwargs):
        index = text_indices.get(id(module))
        if index is None:
            return original_registry(module, query, key, value, attention_mask,
                                     dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        require(module is text.layers[index].self_attn, "non-text attention reached scoped registry wrapper")
        record = ctx["f_records"].setdefault(index, {})
        record["native_q_target"] = query[TARGET, :, RECIPIENT_POSITION].detach().float().cpu().clone()
        record["native_k_target_pre_gqa"] = key[TARGET, :, RECIPIENT_POSITION].detach().float().cpu().clone()
        record["native_v_target_pre_gqa"] = value[TARGET, :, RECIPIENT_POSITION].detach().float().cpu().clone()
        record["q_off_target_native"] = True
        patched_query, patched_key = query, key
        if index in PHASE_LAYERS and ctx["phase_mode"] is not None:
            cos, sin = phase_source[ctx["phase_mode"]]
            patched_query, patched_key = patch_phase_inputs(
                query, key, ctx["pre_q_device"][index], ctx["pre_k_device"][index],
                cos.to(device=query.device, dtype=query.dtype), sin.to(device=query.device, dtype=query.dtype),
                target=TARGET, patch_position=RECIPIENT_POSITION, expected_position=RECIPIENT_POSITION)
            record["phase_mode"] = ctx["phase_mode"]
        else:
            record["phase_mode"] = "native"
        record["q_off_target_exact"] = torch.equal(
            patched_query[:, :, :, :].index_select(0, torch.tensor([0, 1, 3], device=query.device)),
            query[:, :, :, :].index_select(0, torch.tensor([0, 1, 3], device=query.device)))
        q_restored, k_restored = patched_query.clone(), patched_key.clone()
        q_restored[TARGET, :, RECIPIENT_POSITION, :] = query[TARGET, :, RECIPIENT_POSITION, :]
        k_restored[TARGET, :, RECIPIENT_POSITION, :] = key[TARGET, :, RECIPIENT_POSITION, :]
        record["q_k_off_target_exact"] = torch.equal(q_restored, query) and torch.equal(k_restored, key)
        previous = ctx.get("active")
        ctx["active"] = {"layer": index, "module": module,
                          "query": patched_query, "key": patched_key, "value": value,
                          "mask": attention_mask}
        try:
            return original_registry(module, patched_query, patched_key, value, attention_mask,
                                     dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        finally:
            ctx["active"] = previous

    query_indices = torch.tensor([RECIPIENT_POSITION, DONOR_POSITION], device=device, dtype=torch.long)
    model_output = None
    started = time.monotonic()
    try:
        modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS["sdpa"] = registry_wrapper
        F.scaled_dot_product_attention = observed_sdpa
        with torch.inference_mode():
            model_output = q.model(**inputs)
        elapsed = time.monotonic() - started
        logits = model_output.logits.detach().float().cpu().clone()
    finally:
        F.scaled_dot_product_attention = original_sdpa
        modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS["sdpa"] = original_registry
        for handle in all_handles:
            handle.remove()

    # Preserve logits immediately; detailed postprocessing and numerical gates follow.
    tensor_path = OUT / f"{label}.pt"
    torch.save({"full_logits": logits}, tensor_path)
    raw_records = []
    for index in PHASE_LAYERS:
        record = ctx["f_records"].get(index, {})
        raw_records.append(record)
    require(logits.shape == (4, 2, VOCAB), "selected full-vocabulary logits shape changed")
    require(model_output.past_key_values is cache and cache.get_seq_length() == FULL_WIDTH,
            "native cache length changed")
    require(observers["embedding"] == observers["rotary"] == 1 and observers["lm_head"] ==
            {"shape": [4, 2, HIDDEN], "physical_indices": [RECIPIENT_POSITION, DONOR_POSITION], "exact": True} and
            all(item is not None for item in observers["attention"]),
            "native attention attestations incomplete")
    require(len(ctx["f_records"]) == LAYERS and
            all(ctx["f_records"][i].get("f_calls") == 1 and
                ctx["f_records"][i].get("o_proj_consumer_exact") for i in range(LAYERS)),
            "actual SDPA observer count/consumer proof incomplete")
    require(ctx["layer1_input"] is not None and ctx["layer1_input"].shape == (2, HIDDEN),
            "layer1 incoming residual capture incomplete")

    layer_metrics = {}
    payload = {
        "full_logits": logits,
        "query_positions": query_indices.cpu(),
        "layer1_input": ctx["layer1_input"],
        "layer0_before": ctx["layer0_o_proj"]["before_target"],
        "layer0_consumed": ctx["layer0_o_proj"]["consumed_target"],
        "layer0_expected": expected_layer0.cpu(),
    }
    for index in PHASE_LAYERS:
        record = ctx["f_records"][index]
        phase = ctx["phases"][index]
        used_cos, used_sin = phase_source[ctx["phase_mode"]] if ctx["phase_mode"] else (
            phase["recipient_cos"], phase["recipient_sin"])
        pre_q, pre_k = ctx["pre_q"][index], ctx["pre_k"][index]
        q_native, q_consumed = record["native_q_target"], record["consumed_q_target"]
        k_native = record["native_k_target_pre_gqa"].repeat_interleave(Q_TO_KV, dim=0)
        k_consumed = record["consumed_k_target"]
        q_native_expected = rotate_fp64(pre_q, phase["recipient_cos"], phase["recipient_sin"])
        k_native_expected = rotate_fp64(pre_k, phase["recipient_cos"], phase["recipient_sin"]).repeat_interleave(Q_TO_KV, dim=0)
        q_used_expected = rotate_fp64(pre_q, used_cos, used_sin)
        k_used_expected = rotate_fp64(pre_k, used_cos, used_sin).repeat_interleave(Q_TO_KV, dim=0)
        native_self = (q_native.double() * k_native.double()).sum(dim=-1) * (HEAD_DIM ** -0.5)
        consumed_self = (q_consumed.double() * k_consumed.double()).sum(dim=-1) * (HEAD_DIM ** -0.5)
        layer_metrics[index] = {
            "layer": index, "phase_mode": record["phase_mode"],
            "pre_Q": pre_q, "pre_K": pre_k,
            "native_Q_target": q_native, "consumed_Q_target": q_consumed,
            "native_K_target_after_gqa": k_native, "consumed_K_target_after_gqa": k_consumed,
            "consumed_V_target_after_gqa": record["consumed_v_target"],
            "recipient_phase_cos": phase["recipient_cos"], "recipient_phase_sin": phase["recipient_sin"],
            "donor_phase_cos": phase["donor_cos"], "donor_phase_sin": phase["donor_sin"],
            "used_phase_cos": used_cos, "used_phase_sin": used_sin,
            "fp64_native_Q_rotation_max_abs": float((q_native.double() - q_native_expected).abs().max()),
            "fp64_native_K_rotation_max_abs": float((k_native.double() - k_native_expected).abs().max()),
            "fp64_used_Q_rotation_max_abs": float((q_consumed.double() - q_used_expected).abs().max()),
            "fp64_used_K_rotation_max_abs": float((k_consumed.double() - k_used_expected).abs().max()),
            "Q_norm_before_after_max_abs": float((q_native.double().norm(dim=-1) - q_consumed.double().norm(dim=-1)).abs().max()),
            "K_norm_before_after_max_abs": float((k_native.double().norm(dim=-1) - k_consumed.double().norm(dim=-1)).abs().max()),
            "self_score_before_after_max_abs": float((native_self - consumed_self).abs().max()),
            "phase_delta_max_abs": max(float((used_cos - phase["recipient_cos"]).abs().max()),
                                        float((used_sin - phase["recipient_sin"]).abs().max())),
            "actual_recipient_phase_vs_source_early_max_abs": max(
                float((phase["recipient_cos"] - sources["early_cos"]).abs().max()),
                float((phase["recipient_sin"] - sources["early_sin"]).abs().max())),
            "actual_donor_phase_vs_source_late_max_abs": max(
                float((phase["donor_cos"] - sources["late_cos"]).abs().max()),
                float((phase["donor_sin"] - sources["late_sin"]).abs().max())),
            "q_k_off_target_exact": bool(record["q_k_off_target_exact"]),
            "gqa_expansion_exact": bool(record["gqa_expansion_exact"]),
            "historical_K_hash": record["historical_k_hash"],
            "historical_V_hash": record["historical_v_hash"],
            "mask_hash": record["mask_hash"], "cache_gqa_heads": record["cache_gqa_heads"],
            "cache_slots_hash": ctx["attention"]["attention"][index]["cache_slots_hash"],
            "gqa_repetitions": record["gqa_repetitions"], "sdpa_calls": record["f_calls"],
            "o_proj_consumer_exact": bool(record["o_proj_consumer_exact"]),
        }
        raw_records[index - 1] = {key: value for key, value in layer_metrics[index].items()
                                  if isinstance(value, torch.Tensor)}
    payload.update({
        "pre_Q": torch.stack([layer_metrics[i]["pre_Q"] for i in PHASE_LAYERS]),
        "pre_K": torch.stack([layer_metrics[i]["pre_K"] for i in PHASE_LAYERS]),
        "native_Q_target": torch.stack([layer_metrics[i]["native_Q_target"] for i in PHASE_LAYERS]),
        "consumed_Q_target": torch.stack([layer_metrics[i]["consumed_Q_target"] for i in PHASE_LAYERS]),
        "native_K_target_after_gqa": torch.stack([layer_metrics[i]["native_K_target_after_gqa"] for i in PHASE_LAYERS]),
        "consumed_K_target_after_gqa": torch.stack([layer_metrics[i]["consumed_K_target_after_gqa"] for i in PHASE_LAYERS]),
        "consumed_V_target_after_gqa": torch.stack([layer_metrics[i]["consumed_V_target_after_gqa"] for i in PHASE_LAYERS]),
        "used_phase_cos": torch.stack([layer_metrics[i]["used_phase_cos"] for i in PHASE_LAYERS]),
        "used_phase_sin": torch.stack([layer_metrics[i]["used_phase_sin"] for i in PHASE_LAYERS]),
    })
    torch.save(payload, tensor_path)
    require(tensor_path.stat().st_size <= MAX_TENSOR_BYTES, f"{label} tensor payload cap exceeded")
    state["tensor_bytes"] = sum(path.stat().st_size for path in OUT.glob("*.pt"))
    require(state["tensor_bytes"] <= MAX_TENSOR_BYTES, "total tensor payload cap exceeded")
    summary = target_summary(logits, label, native_logits)
    metrics_json = {str(index): {key: value for key, value in metric.items()
                                 if not isinstance(value, torch.Tensor)}
                    for index, metric in layer_metrics.items()}
    record = {
        "cell": label, "tensor_artifact": binding(tensor_path), "elapsed_seconds": elapsed,
        "logits_shape": list(logits.shape), "layer1_input_shape": list(ctx["layer1_input"].shape),
        "native_input_hashes": {key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")},
        "layer1_input_hash": tensor_hash(ctx["layer1_input"]), "layer1_input": ctx["layer1_input"].tolist(),
        "layer_metrics": metrics_json, "summary": summary,
        "query_consumer": observers["lm_head"], "attention": observers["attention"],
        "embedding_calls": observers["embedding"], "rotary_calls": observers["rotary"],
        "cache_length": cache.get_seq_length(),
        "layer0_consumer": {key: ctx["layer0_o_proj"][key]
                            for key in ("second_hook_seen", "off_target_exact", "off_target_max_abs")},
        "phase_treatment_nonzero_max_abs": max(metrics_json[str(i)]["phase_delta_max_abs"] for i in PHASE_LAYERS),
    }
    return {"record": record, "logits": logits, "native": native, "case": case,
            "queries": queries, "cache": cache, "context": ctx, "tensor_path": tensor_path}, logits


def run(device: str) -> None:
    sources = cpu_selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA device required")
    require(not OUT.exists(), "attempt path already exists; preserve previous attempts")
    selection, refs = sources["selection"], sources["refs"]
    OUT.mkdir(parents=True)
    state = {"schema": "recurrence_downstream_query_phase.receipt.v1", "status": "preparing",
             "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "cell_order": list(CELL_ORDER), "completed_cells": [], "calls": [],
             "started_unix": time.time(), "started": time.monotonic(), "source": refs}
    write(OUT / "receipt.json", state)
    cell_readbacks = {}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        oracle = json.loads(ORACLE.read_text())
        case, _meta, _queries, _raw, _trace = trajectory.source_inputs("val", q, identity, oracle, device)
        native = case["native"]
        native_hashes = {key: tensor_hash(native[key]) for key in ("input_ids", "attention_mask", "position_ids")}
        require(native_hashes == selection["native_input_hashes"], "native input hashes changed")
        producer_capture = preserve_source(Path(__file__), run_root=OUT,
                                            relative_name="recurrence_downstream_query_phase.py")
        unit_capture = preserve_source(UNIT, run_root=OUT, relative_name="experiment/unit.md")
        dependency_paths = [
            Path('probes/recurrence_dynamics/recurrence_native_trajectory.py'),
            Path('probes/recurrence_dynamics/recurrence_first_layer_readout.py'),
            Path('probes/recurrence_dynamics/recurrence_first_layer_groups.py'),
            Path('probes/recurrence_dynamics/recurrence_first_layer_reverse_transfer.py'),
            Path('probes/recurrence_dynamics/recurrence_attention_mass.py'),
            Path('probes/recurrence_dynamics/recurrence_fixed_template.py'),
            Path("probes/model_profiles/mature_tied_untied.py"),
            Path('probes/recurrence_dynamics/numerical_feedback/runtime.py'),
            Path("src/qwen/input_identity.py"), Path("src/inference/bound_requests.py"),
            Path("src/qwen/untied_embeddings.py"), Path(inspect.getfile(modeling_qwen3_vl)),
            Path(inspect.getfile(sdpa_attention_forward)),
        ]
        dependency_captures = [
            preserve_source(path, run_root=OUT,
                            relative_name=str(path) if not path.is_absolute() else f"transformers/{path.name}")
            for path in dependency_paths
        ]
        manifest = {
            "schema": "recurrence_downstream_query_phase.v1", "status": "frozen_before_forward",
            "source": refs, "repair_ruling": binding(RULING), "producer": binding(Path(__file__)),
            "producer_capture": binding(producer_capture), "unit_capture": binding(unit_capture),
            "dependency_captures": [binding(path) for path in dependency_captures],
            "transformers_version": transformers.__version__, "model_identity": identity,
            "case": {"target_batch": TARGET, "full_width": FULL_WIDTH,
                     "query_positions": [RECIPIENT_POSITION, DONOR_POSITION],
                     "native_input_hashes": native_hashes,
                     "source_bindings": case["source_bindings"], "accepted_paths": case["accepted_paths"],
                     "fixed_artifacts": case["fixed_bindings"]},
            "source_vectors": {
                "head_output_shape": [Q_HEADS, HEAD_DIM], "early_head_hash": tensor_hash(sources["early_head"]),
                "late_head_hash": tensor_hash(sources["late_head"]),
                "early_phase_hashes": {"cos": tensor_hash(sources["early_cos"]), "sin": tensor_hash(sources["early_sin"])},
                "late_phase_hashes": {"cos": tensor_hash(sources["late_cos"]), "sin": tensor_hash(sources["late_sin"])},
                "layer1_early_input_hash": tensor_hash(sources["early_input"]),
                "layer1_late_input_hash": tensor_hash(sources["late_input"]),
            },
            "collection": {
                "cell_order": list(CELL_ORDER), "fresh_empty_cache_per_cell": True,
                "selected_logits": "physical indices [1703,2126] via tensor logits_to_keep",
                "layer0": "actual late head_output[1] replacement at [target2,1703,:] for both held cells",
                "phase_layers": list(PHASE_LAYERS), "phase": "co-rotate current pre-Q and self-key with selected saved phase",
                "actual_consumer": "scoped text SDPA registry plus F.scaled_dot_product_attention observer after GQA",
                "historical_hash_slice": "target2, all expanded heads, positions 0:1703",
                "model_forward_cap": MAX_MODEL_FORWARDS, "vision_forward_cap": MAX_VISION_FORWARDS,
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
            native_logits = None
            for label in CELL_ORDER:
                state["current_cell"] = label
                write(OUT / "receipt.json", state)
                result, logits = collect_cell(label, sources, q, identity, device, state, native_logits)
                if native_logits is None:
                    native_logits = logits.clone()
                cell_readbacks[label] = result["record"]
                state["completed_cells"].append(label)
                state["calls"].append({"cell": label, "seconds": result["record"]["elapsed_seconds"]})
                # Persist this cell before its numerical gates.
                write(OUT / "readback.json", {"schema": "recurrence_downstream_query_phase.readback.v1",
                                               "status": "running", "source_manifest": binding(OUT / "source-to-cell.json"),
                                               "cells": cell_readbacks, "model_forwards": state["model_forwards"],
                                               "vision_forwards": state["vision_forwards"]})
                record = result["record"]
                summary = record["summary"]
                metrics = record["layer_metrics"]
                require(all(float(metrics[str(i)]["fp64_native_Q_rotation_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["fp64_native_K_rotation_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["fp64_used_Q_rotation_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["fp64_used_K_rotation_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["Q_norm_before_after_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["K_norm_before_after_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["self_score_before_after_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["actual_recipient_phase_vs_source_early_max_abs"]) <= ATOL and
                            float(metrics[str(i)]["actual_donor_phase_vs_source_late_max_abs"]) <= ATOL and
                            metrics[str(i)]["q_k_off_target_exact"] and metrics[str(i)]["gqa_expansion_exact"] and
                            metrics[str(i)]["o_proj_consumer_exact"] and metrics[str(i)]["sdpa_calls"] == 1
                            for i in PHASE_LAYERS), f"{label} per-layer consumer/rotation gate failed")
                require(record["query_consumer"] == {"shape": [4, 2, HIDDEN],
                                                     "physical_indices": [RECIPIENT_POSITION, DONOR_POSITION], "exact": True},
                        f"{label} selected LM consumer changed")
                require(all(item["mask_hash"] == cell_readbacks["native"]["layer_metrics"][str(1)]["mask_hash"]
                            for item in metrics.values()), f"{label} mask identity changed")
                require(all(item["cache_slots_hash"] == cell_readbacks["native"]["layer_metrics"][str(1)]["cache_slots_hash"]
                            for item in metrics.values()) and
                        record["native_input_hashes"] == cell_readbacks["native"]["native_input_hashes"],
                        f"{label} cache-slot or input identity changed")
                if label == "native":
                    require(float((logits[TARGET] - sources["accepted_logits"]).abs().max()) <= ATOL and
                            logits[TARGET, 0].argmax().item() == P38 and logits[TARGET, 1].argmax().item() == P999,
                            "native selected logits qualification failed")
                    require(float((result["context"]["layer1_input"][0] - sources["early_input"]).abs().max()) <= ATOL,
                            "native early layer1 incoming state mismatch")
                    require(all(metrics[str(i)]["historical_K_hash"] and metrics[str(i)]["historical_V_hash"]
                                for i in PHASE_LAYERS), "native historical K/V hashes were not captured")
                else:
                    require(float((logits[[0, 1, 3]] - native_logits[[0, 1, 3]]).abs().max()) <= ATOL,
                            f"{label} companion parity failed")
                    require(all(metrics[str(i)]["historical_K_hash"] == cell_readbacks["native"]["layer_metrics"][str(i)]["historical_K_hash"] and
                                metrics[str(i)]["historical_V_hash"] == cell_readbacks["native"]["layer_metrics"][str(i)]["historical_V_hash"]
                                for i in PHASE_LAYERS), f"{label} historical K/V identity changed")
                    require(float((result["context"]["layer1_input"][0] - sources["late_input"]).abs().max()) <= ATOL,
                            f"{label} matched late layer1 incoming state failed")
                    if label == "held_native_phase":
                        require(float((logits[TARGET, 0] - sources["reverse_late_logits"][TARGET]).abs().max()) <= ATOL and
                                logits[TARGET, 0].argmax().item() == P38,
                                "held native-phase reverse-transfer parity failed")
                del result
            require(state["model_forwards"] == MAX_MODEL_FORWARDS and
                    state["vision_forwards"] == MAX_VISION_FORWARDS,
                    "exactly three model and vision forwards required")
        finally:
            for handle in counter_handles:
                handle.remove()
        phase_nonzero = max(cell_readbacks["held_late_phase"]["phase_treatment_nonzero_max_abs"], 0.0)
        require(phase_nonzero > 1e-6, "phase treatment was numerically zero")
        final_readback = {"schema": "recurrence_downstream_query_phase.readback.v1", "status": "candidate",
                          "source_manifest": binding(OUT / "source-to-cell.json"), "cells": cell_readbacks,
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "phase_treatment_nonzero_max_abs": phase_nonzero}
        write(OUT / "readback.json", final_readback)
        result = {"status": "candidate", "cell_order": list(CELL_ORDER),
                  "target_winners": {label: [cell_readbacks[label]["summary"]["queries"][i]["winner"]
                                               for i in range(2)] for label in CELL_ORDER},
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
        if cell_readbacks:
            write(OUT / "readback.json", {"schema": "recurrence_downstream_query_phase.readback.v1",
                                           "status": "technical_invalid",
                                           "source_manifest": binding(OUT / "source-to-cell.json")
                                           if (OUT / "source-to-cell.json").exists() else None,
                                           "cells": cell_readbacks, "error": repr(error)})
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
