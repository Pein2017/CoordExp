"""Cross old/new normalized keys with old/new observed MRoPE phase at native exit."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_donor_tracking import ROW, probability_readback, require, write
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest, check_cache
from probes.training_set_completion.recurrence_native_row_mass import block_hashes, blocks
from probes.training_set_completion.recurrence_position_history import PANEL, TARGET, source_and_rows
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase/attempt-003")
FAILED_PREVIOUS = (OUT.parent / "attempt-001", OUT.parent / "attempt-002")
NUMERIC_DIAGNOSIS = OUT.parent / "lead-checks/failed-rotation-diagnosis.json"
UNIT = Path("research/experiments/2026-09-22-recurrence-key-phase/unit.md")
PREVIOUS = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/native-row-mass-002")
PREVIOUS_ACCEPTANCE = PREVIOUS.parent / "lead-checks/native-bridge-readback.json"
WIDTH, FULL_WIDTH, SUFFIX, TARGET_OFFSET = 2121, 2127, 6, 807
SOURCE, DEST = (2103, 2112), (2112, 2121)
LAYERS, KV_HEADS, Q_HEADS, HEAD_DIM = 28, 8, 16, 128
MAX_FORWARDS, MAX_SECONDS, ATOL = 8, 900, 2e-4


def rotate_half(value):
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def rotate(value, cos, sin):
    require(value.shape[-1] == cos.shape[-1] == sin.shape[-1] and value.shape[-2] == cos.shape[-2] == sin.shape[-2],
            "key/phase dimension mismatch")
    return value * cos.unsqueeze(0) + rotate_half(value) * sin.unsqueeze(0)


def unrotate(value, cos, sin):
    denom = cos.square() + sin.square()
    half = HEAD_DIM // 2
    require(bool(torch.all(denom > 0)) and torch.equal(cos[..., :half], cos[..., half:]) and
            torch.equal(sin[..., :half], sin[..., half:]), "RoPE half-frequency coefficients changed")
    return (value * cos.unsqueeze(0) - rotate_half(value) * sin.unsqueeze(0)) / denom.unsqueeze(0)


def make_keys(pre_old, pre_new, old_cos, old_sin, new_cos, new_sin, old_post, new_post):
    require(pre_old.shape == pre_new.shape == old_post.shape == new_post.shape == (KV_HEADS, 9, HEAD_DIM),
            "source/destination pre/post-K shape changed")
    require(old_cos.shape == old_sin.shape == new_cos.shape == new_sin.shape == (9, HEAD_DIM), "observed phase shape changed")
    old_replayed, new_replayed = rotate(pre_old, old_cos, old_sin), rotate(pre_new, new_cos, new_sin)
    replay_error = max(float((old_replayed - old_post).abs().max()), float((new_replayed - new_post).abs().max()))
    recovered_old, recovered_new = unrotate(old_post, old_cos, old_sin), unrotate(new_post, new_cos, new_sin)
    inverse_error = max(float((recovered_old - pre_old).abs().max()), float((recovered_new - pre_new).abs().max()))
    phase = {"NN": new_post.clone(), "OO": old_post.clone(),
             "ON": rotate(recovered_old, new_cos, new_sin), "NO": rotate(recovered_new, old_cos, old_sin)}
    roundtrip_error = max(float((rotate(unrotate(phase["ON"], new_cos, new_sin), old_cos, old_sin) - old_post).abs().max()),
                          float((rotate(unrotate(phase["NO"], old_cos, old_sin), new_cos, new_sin) - new_post).abs().max()))
    norm_error = max(float((phase["ON"].norm(dim=-1) - old_post.norm(dim=-1)).abs().max()),
                     float((phase["NO"].norm(dim=-1) - new_post.norm(dim=-1)).abs().max()))
    return phase, {"native_replay_max_abs": replay_error, "inverse_preK_max_abs": inverse_error,
                   "phase_roundtrip_max_abs": roundtrip_error, "phase_norm_max_abs": norm_error}


def phase_qualification(pre_old, pre_new, old_post, new_post, new_cos, new_sin, old_cos, old_sin, keys, errors):
    """Scale FP32 roundoff checks; use separate FP64 complex-pair crossed oracles."""
    eps = torch.finfo(torch.float32).eps
    key_scale = max(1.0, *(float(x.abs().max()) for x in (pre_old, pre_new, old_post, new_post)))
    post_scale = max(1.0, float(old_post.abs().max()), float(new_post.abs().max()))
    norm_scale = max(1.0, float(old_post.norm(dim=-1).max()), float(new_post.norm(dim=-1).max()))
    bounds = {"native_replay_max_abs": 2e-5, "inverse_preK_max_abs": 8 * eps * key_scale,
              "phase_roundtrip_max_abs": 8 * eps * key_scale, "phase_norm_max_abs": 8 * eps * norm_scale,
              "crossed_complex_max_abs": 8 * eps * post_scale}

    def complex_rephase(post, from_cos, from_sin, to_cos, to_sin):
        post = post.double()
        a, b = post[..., :HEAD_DIM // 2], post[..., HEAD_DIM // 2:]
        c, s = from_cos.double()[..., :HEAD_DIM // 2], from_sin.double()[..., :HEAD_DIM // 2]
        denominator = c.square() + s.square()
        real = (a * c.unsqueeze(0) + b * s.unsqueeze(0)) / denominator.unsqueeze(0)
        imag = (b * c.unsqueeze(0) - a * s.unsqueeze(0)) / denominator.unsqueeze(0)
        c, s = to_cos.double()[..., :HEAD_DIM // 2], to_sin.double()[..., :HEAD_DIM // 2]
        return torch.cat((real * c.unsqueeze(0) - imag * s.unsqueeze(0),
                          imag * c.unsqueeze(0) + real * s.unsqueeze(0)), dim=-1)

    crossed = {"ON": float((keys["ON"].double() - complex_rephase(old_post, old_cos, old_sin, new_cos, new_sin)).abs().max()),
               "NO": float((keys["NO"].double() - complex_rephase(new_post, new_cos, new_sin, old_cos, old_sin)).abs().max())}
    measured = {**errors, "crossed_complex_max_abs": max(crossed.values())}
    return {"qualified": all(measured[k] <= bounds[k] for k in bounds), "scales":
            {"key_max_abs": key_scale, "post_max_abs": post_scale, "vector_norm_max": norm_scale},
            "bounds": bounds, "measured": measured, "crossed_complex": crossed,
            "ratios": {k: measured[k] / bounds[k] for k in bounds}}


@contextmanager
def patch_key(cache, candidate, original_digest, *, width=WIDTH, dest=DEST):
    require(dest == (width - 9, width) and len(candidate) == len(cache.layers), "destination or layer count changed")
    saved = []
    try:
        for layer, selected in zip(cache.layers, candidate, strict=True):
            require(selected.shape == (KV_HEADS, 9, HEAD_DIM), "candidate K block shape changed")
            original = layer.keys[TARGET, :, dest[0]:dest[1], :].clone()
            saved.append((layer, original))
            layer.keys[TARGET, :, dest[0]:dest[1], :].copy_(selected)
        yield
    finally:
        cache.crop(width)
        for layer, original in saved:
            layer.keys[TARGET, :, dest[0]:dest[1], :].copy_(original)
        require(cache_digest(cache) == original_digest, "native cache did not restore exactly")


def observe_attention(q_post, keys, values, mask, actual_headout, scaling):
    require(q_post.shape == (Q_HEADS, SUFFIX, HEAD_DIM) and keys.shape == values.shape == (KV_HEADS, FULL_WIDTH, HEAD_DIM),
            "target Q/K/V observation shape changed")
    require(mask.dtype == torch.bool and mask.shape == (SUFFIX, FULL_WIDTH), "actual SDPA mask type/shape changed")
    repeated_keys = keys.repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
    repeated_values = values.repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
    scores = torch.matmul(q_post, repeated_keys.transpose(-1, -2)) * scaling
    masked_scores = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)
    probs = torch.softmax(masked_scores, dim=-1)
    reconstructed = torch.matmul(probs, repeated_values)
    require(bool(torch.isfinite(probs).all()) and bool(torch.isfinite(reconstructed).all()), "attention reconstruction nonfinite")
    error = float((reconstructed - actual_headout).abs().max())
    return scores, probs, reconstructed, error


def fixed_native_headout(q_nn, p_nn, o_nn, v_native, k_native, k_candidate, scaling):
    """Exact nine-key softmax reweighting at fixed native Q and all other K/V."""
    require(q_nn.shape == o_nn.shape == (Q_HEADS, SUFFIX, HEAD_DIM) and p_nn.shape == (Q_HEADS, SUFFIX, 9),
            "fixed-native target shape changed")
    kv_groups = Q_HEADS // KV_HEADS
    delta = torch.matmul(q_nn.double(), (k_candidate - k_native).double().repeat_interleave(kv_groups, dim=0).transpose(-1, -2)) * scaling
    weighted = p_nn.double() * torch.expm1(delta)
    denominator = 1 + weighted.sum(dim=-1, keepdim=True)
    require(bool(torch.all(denominator > 0)), "fixed-native softmax denominator invalid")
    value = v_native.double().repeat_interleave(kv_groups, dim=0)
    return (o_nn.double() + torch.matmul(weighted, value)) / denominator, denominator


def suffix_observers(model, cache, expected_blocks, candidates=None):
    """Observe the real SDPA path; reconstruct weights without changing backend."""
    records = [dict() for _ in range(LAYERS)]
    rope, handles = [], []
    text = model.model.language_model

    def rotary(_module, _args, output):
        cos, sin = output
        require(cos.shape == sin.shape == (4, SUFFIX, HEAD_DIM), "actual current-S phase shape changed")
        rope.append((cos[TARGET].detach(), sin[TARGET].detach()))
    handles.append(text.rotary_emb.register_forward_hook(rotary))

    for i, layer in enumerate(text.layers):
        def before_layer(_module, args, _kwargs, index=i):
            require(args[0].shape[0:2] == (4, SUFFIX), "decoder S input shape changed")
            records[index]["residual_input"] = args[0][TARGET].detach().float().cpu().clone()

        def before_attention(_module, _args, kwargs, index=i):
            require(kwargs.get("past_key_values") is cache and cache.get_seq_length(index) == WIDTH,
                    "attention consumed wrong cache/prefix length")
            mask = kwargs.get("attention_mask")
            require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.shape == (4, 1, SUFFIX, FULL_WIDTH),
                    "actual SDPA mask type/shape changed")
            phase = kwargs.get("position_embeddings")
            require(len(rope) == 1 and isinstance(phase, tuple) and len(phase) == 2 and
                    torch.equal(phase[0][TARGET], rope[0][0]) and torch.equal(phase[1][TARGET], rope[0][1]),
                    "attention did not consume observed current-S phase")
            position = kwargs.get("cache_position")
            require(isinstance(position, torch.Tensor) and torch.equal(position, torch.arange(WIDTH, FULL_WIDTH, device=position.device)),
                    "physical S cache slots changed")
            layer_cache = cache.layers[index]
            actual = {axis: tensor_hash(getattr(layer_cache, axis)[TARGET, :, DEST[0]:DEST[1], :]) for axis in ("keys", "values")}
            require(actual == expected_blocks[index]["dest"], "attention consumed wrong candidate K/native V block")
            records[index]["dest_hashes"] = actual
            records[index]["source_hashes"] = {axis: tensor_hash(getattr(layer_cache, axis)[TARGET, :, SOURCE[0]:SOURCE[1], :])
                                                for axis in ("keys", "values")}
            require(records[index]["source_hashes"] == expected_blocks[index]["source"], "source block changed")
            records[index]["mask_hash"] = tensor_hash(mask)
            records[index]["cache_length_before"] = cache.get_seq_length(index)
            records[index]["_mask"] = mask[TARGET, 0].detach()

        def query_normalized(_module, _args, output, index=i):
            require(output.shape == (4, SUFFIX, Q_HEADS, HEAD_DIM), "normalized query shape changed")
            records[index]["_pre_q"] = output[TARGET].transpose(0, 1).detach()
            records[index]["q_norm_pre_rope"] = records[index]["_pre_q"].float().cpu().clone()

        def key_normalized(_module, _args, output, index=i):
            require(output.shape == (4, SUFFIX, KV_HEADS, HEAD_DIM), "normalized current-S K shape changed")
            records[index]["_pre_k"] = output[TARGET].transpose(0, 1).detach()
            records[index]["current_S_k_norm_pre_rope"] = records[index]["_pre_k"].float().cpu().clone()

        def value_projected(_module, _args, output, index=i):
            require(output.shape == (4, SUFFIX, KV_HEADS * HEAD_DIM), "current-S V projection shape changed")
            records[index]["_value"] = output[TARGET].reshape(SUFFIX, KV_HEADS, HEAD_DIM).transpose(0, 1).detach()
            records[index]["current_S_value"] = records[index]["_value"].float().cpu().clone()

        def before_output_projection(_module, args, index=i):
            record = records[index]
            require(len(rope) == 1 and "_pre_q" in record and "_mask" in record and cache.get_seq_length(index) == FULL_WIDTH,
                    "actual attention observation ordering changed")
            actual = args[0][TARGET].reshape(SUFFIX, Q_HEADS, HEAD_DIM).transpose(0, 1).detach()
            post_q = rotate(record["_pre_q"], *rope[0])
            layer_cache = cache.layers[index]
            keys, values = layer_cache.keys[TARGET], layer_cache.values[TARGET]
            if candidates is not None:
                current_k_error = float((rotate(record["_pre_k"], *rope[0]) - keys[:, WIDTH:FULL_WIDTH]).abs().max())
                current_v_error = float((record["_value"] - values[:, WIDTH:FULL_WIDTH]).abs().max())
                require(current_k_error <= 2e-5 and current_v_error <= 2e-5, "observed current-S K/V did not reach cache")
                record["current_S_K_cache_max_abs"] = current_k_error
                record["current_S_V_cache_max_abs"] = current_v_error
            scale = layer.self_attn.scaling
            scores, probs, reconstructed, error = observe_attention(post_q, keys, values, record["_mask"], actual, scale)
            record.update(q_post_reconstructed=post_q.float().cpu(), actual_headout=actual.float().cpu(),
                          reconstructed_headout=reconstructed.float().cpu(),
                          attention_probabilities_reconstructed=probs.float().cpu(),
                          qk_scores_source_dest=scores[:, :, SOURCE[0]:DEST[1]].float().cpu(),
                          attention_reconstruction_max_abs=error)
            if candidates is not None:
                repeated_values = values.repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
                baseline_dest = keys[:, DEST[0]:DEST[1]]
                fixed_brute = {}
                for name in ("OO", "ON", "NO"):
                    changed = candidates[name][index]
                    delta = torch.matmul(post_q, (changed - baseline_dest).repeat_interleave(Q_HEADS // KV_HEADS, dim=0).transpose(-1, -2)) * scale
                    altered = scores.clone()
                    altered[:, :, DEST[0]:DEST[1]] += delta
                    altered = altered.masked_fill(~record["_mask"].unsqueeze(0), -torch.inf)
                    weights = torch.softmax(altered, dim=-1)
                    fixed_brute[name] = torch.matmul(weights, repeated_values).float().cpu()
                record["fixed_native_brute_headout"] = fixed_brute
                record["native_V_destination"] = values[:, DEST[0]:DEST[1]].float().cpu()
            del record["_pre_q"], record["_mask"]
            if candidates is not None:
                del record["_pre_k"], record["_value"]

        def after_attention(_module, _args, output, index=i):
            records[index]["attention_projected_output"] = output[0][TARGET].detach().float().cpu().clone()

        def after_mlp(_module, _args, output, index=i):
            records[index]["mlp_output"] = output[TARGET].detach().float().cpu().clone()

        def after_layer(_module, _args, output, index=i):
            records[index]["residual_output"] = output[TARGET].detach().float().cpu().clone()
            records[index]["post_attention_residual"] = records[index]["residual_input"] + records[index]["attention_projected_output"]

        handles.extend((layer.register_forward_pre_hook(before_layer, with_kwargs=True),
                        layer.self_attn.register_forward_pre_hook(before_attention, with_kwargs=True),
                        layer.self_attn.q_norm.register_forward_hook(query_normalized),
                        *([layer.self_attn.k_norm.register_forward_hook(key_normalized),
                           layer.self_attn.v_proj.register_forward_hook(value_projected)] if candidates is not None else []),
                        layer.self_attn.o_proj.register_forward_pre_hook(before_output_projection),
                        layer.self_attn.register_forward_hook(after_attention),
                        layer.mlp.register_forward_hook(after_mlp),
                        layer.register_forward_hook(after_layer)))
    return records, rope, handles


def selfcheck():
    blocks((0, 9), (9, 18), width=18)
    try:
        blocks((1, 10), (9, 18), width=18)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong source block admitted")
    theta = torch.linspace(0.1, 1.0, HEAD_DIM // 2)
    old_angle, new_angle = torch.cat((theta, theta)), torch.cat((theta + 0.2, theta + 0.2))
    old_cos, old_sin = old_angle.cos().repeat(9, 1), old_angle.sin().repeat(9, 1)
    new_cos, new_sin = new_angle.cos().repeat(9, 1), new_angle.sin().repeat(9, 1)
    old, new = torch.randn(KV_HEADS, 9, HEAD_DIM), torch.randn(KV_HEADS, 9, HEAD_DIM)
    keys, errors = make_keys(old, new, old_cos, old_sin, new_cos, new_sin,
                             rotate(old, old_cos, old_sin), rotate(new, new_cos, new_sin))
    require(max(errors.values()) < 1e-4 and torch.equal(keys["OO"], rotate(old, old_cos, old_sin)), "RoPE CPU roundtrip")
    qualification = phase_qualification(old, new, keys["OO"], keys["NN"], new_cos, new_sin,
                                        old_cos, old_sin, keys, errors)
    require(qualification["qualified"], "valid crossed phase failed FP32-scaled/complex oracle")
    wrong = dict(keys); wrong["ON"] = keys["OO"]
    require(not phase_qualification(old, new, keys["OO"], keys["NN"], new_cos, new_sin,
                                    old_cos, old_sin, wrong, errors)["qualified"], "wrong phase escaped complex oracle")
    try:
        rotate(old, old_cos[:, :-1], old_sin)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong RoPE axis dimension admitted")
    # The observer must recover the actual SDPA output on the boolean-mask route.
    q, k, v = torch.randn(Q_HEADS, SUFFIX, HEAD_DIM), torch.randn(KV_HEADS, FULL_WIDTH, HEAD_DIM), torch.randn(KV_HEADS, FULL_WIDTH, HEAD_DIM)
    mask = torch.ones(SUFFIX, FULL_WIDTH, dtype=torch.bool)
    mask[:, DEST[0]:DEST[1]] = False
    kv_k, kv_v = k.repeat_interleave(2, dim=0), v.repeat_interleave(2, dim=0)
    actual = F.scaled_dot_product_attention(q, kv_k, kv_v, attn_mask=mask.unsqueeze(0), scale=HEAD_DIM**-0.5)
    scores, probs, output, error = observe_attention(q, k, v, mask, actual, HEAD_DIM**-0.5)
    require(bool(torch.all(probs[:, :, DEST[0]:DEST[1]] == 0)) and error < 1e-5, "boolean SDPA reconstruction parity")
    try:
        observe_attention(q, k, v, mask.float(), output, HEAD_DIM**-0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("float mask admitted on SDPA bool route")
    visible = torch.ones_like(mask)
    _, p0, o0, _ = observe_attention(q, k, v, visible, torch.zeros_like(q), HEAD_DIM**-0.5)
    shifted = k[:, DEST[0]:DEST[1]].clone() + 0.3
    fixed, _ = fixed_native_headout(q, p0[:, :, DEST[0]:DEST[1]], o0, v[:, DEST[0]:DEST[1]],
                                    k[:, DEST[0]:DEST[1]], shifted, HEAD_DIM**-0.5)
    changed_k = k.clone(); changed_k[:, DEST[0]:DEST[1]] = shifted
    _, _, brute, _ = observe_attention(q, changed_k, v, visible, torch.zeros_like(q), HEAD_DIM**-0.5)
    require(float((fixed - brute.double()).abs().max()) < 1e-5, "fixed-native softmax identity failed")
    key_cache = DynamicCache(((torch.zeros(4, KV_HEADS, 18, HEAD_DIM), torch.ones(4, KV_HEADS, 18, HEAD_DIM)),))
    baseline = cache_digest(key_cache)
    patch = [torch.ones(KV_HEADS, 9, HEAD_DIM)]
    with torch.inference_mode():
        with patch_key(key_cache, patch, baseline, width=18, dest=(9, 18)):
            require(bool(torch.all(key_cache.layers[0].keys[TARGET, :, 9:18] == 1)) and
                    bool(torch.all(key_cache.layers[0].values == 1)) and
                    bool(torch.all(key_cache.layers[0].keys[0] == 0)), "K-only/companion mutation failed")
            key_cache.update(torch.zeros(4, KV_HEADS, 6, HEAD_DIM), torch.ones(4, KV_HEADS, 6, HEAD_DIM), 0)
        require(cache_digest(key_cache) == baseline, "post-append key restoration failed")
        for bad_patch, bad_dest in ((patch, (8, 17)), ([torch.ones(KV_HEADS, 8, HEAD_DIM)], (9, 18))):
            try:
                with patch_key(key_cache, bad_patch, baseline, width=18, dest=bad_dest):
                    pass
            except ValueError:
                pass
            else:
                raise AssertionError("wrong destination or K shape admitted")
        contaminated = DynamicCache(((torch.zeros(4, KV_HEADS, 18, HEAD_DIM), torch.ones(4, KV_HEADS, 18, HEAD_DIM)),))
        try:
            with patch_key(contaminated, patch, cache_digest(contaminated), width=18, dest=(9, 18)):
                contaminated.layers[0].keys[0, 0, 0, 0] = 1
        except ValueError:
            pass
        else:
            raise AssertionError("companion contamination escaped cache-restoration check")


def vector_metrics(direct, feedback, actual):
    def component(value, *, final):
        source = value[:, -1, :] if final else value
        return torch.linalg.vector_norm(source.double(), dim=-1 if final else (-2, -1))
    def cosine(a, b, *, final):
        aa, bb = (a[:, -1, :], b[:, -1, :]) if final else (a.reshape(Q_HEADS, -1), b.reshape(Q_HEADS, -1))
        aa, bb = aa.double(), bb.double()
        denom = torch.linalg.vector_norm(aa, dim=-1) * torch.linalg.vector_norm(bb, dim=-1)
        return [None if float(d) <= 1e-12 else float(x / d) for x, d in zip((aa * bb).sum(dim=-1), denom, strict=True)]
    return {scope: {"direct_head_norms": component(direct, final=final).tolist(),
                    "feedback_head_norms": component(feedback, final=final).tolist(),
                    "actual_delta_head_norms": component(actual, final=final).tolist(),
                    "direct_actual_alignment": cosine(direct, actual, final=final),
                    "feedback_actual_alignment": cosine(feedback, actual, final=final)}
            for scope, final in (("all_S", False), ("final_y1_query", True))}


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA required")
    require(not OUT.exists(), "attempt path already exists")
    failed_receipts = [json.loads((path / "receipt.json").read_text()) for path in FAILED_PREVIOUS]
    for path, receipt_before in zip(FAILED_PREVIOUS, failed_receipts, strict=True):
        require(receipt_before["status"] == "technical_invalid" and receipt_before["model_forwards"] == 1 and
                receipt_before["vision_forwards"] == 1 and "layer 0 observed pre/post RoPE qualification failed" in
                receipt_before["error"] and receipt_before["manifest"] == literal_binding(path / "source-to-cell.json"),
                "prior attempt identity/cost changed")
    require(json.loads(NUMERIC_DIAGNOSIS.read_text())["status"] == "independent_complex_oracle",
            "independent CPU rotation diagnosis changed")
    prior_forwards = sum(value["model_forwards"] for value in failed_receipts)
    prior_seconds = sum(value["elapsed_seconds"] for value in failed_receipts)
    acceptance = json.loads(PREVIOUS_ACCEPTANCE.read_text())
    require(acceptance["status"] == "lead-accepted-native-bridge", "native bridge not accepted")
    previous_result_path, previous_manifest_path = PREVIOUS / "result.json", PREVIOUS / "source-to-cell.json"
    previous_result, previous_manifest = json.loads(previous_result_path.read_text()), json.loads(previous_manifest_path.read_text())
    require(previous_result["status"] == "candidate", "native bridge result missing")
    references = {"NN": PREVIOUS / "N.pt", "OO": PREVIOUS / "K.pt"}
    require(all(literal_binding(path) == previous_result["cells"][{"NN": "N", "OO": "K"}[name]]["full_logits"]
                for name, path in references.items()), "accepted N/K full-vector binding changed")
    chosen, raw, trace, receipt, panel, group, S, offsets, _ = source_and_rows()
    tokens = raw[TARGET]["token_ids"]
    require(offsets[1] == TARGET_OFFSET and tokens[783:792] == tokens[792:801] == ROW and tokens[801:807] == S,
            "source/destination native row or current S changed")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "started_unix": time.time(), "source": {"frozen_unit": literal_binding(UNIT),
             "failed_previous_attempts": [{"receipt": literal_binding(path / "receipt.json"),
                                           "manifest": literal_binding(path / "source-to-cell.json")}
                                          for path in FAILED_PREVIOUS],
             "numeric_diagnosis": literal_binding(NUMERIC_DIAGNOSIS),
             "previous_acceptance": literal_binding(PREVIOUS_ACCEPTANCE),
             "previous_result": literal_binding(previous_result_path), "previous_manifest": literal_binding(previous_manifest_path),
             "reference_vectors": {name: literal_binding(path) for name, path in references.items()},
             "selection": previous_manifest["source"]["selection"], **{key: chosen[key] for key in ("raw", "trace", "runtime_receipt", "image")}}}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        current, saved = dict(identity), dict(receipt["identity"])
        current_loader, saved_loader = current.pop("loader_source"), saved.pop("loader_source")
        require(current == saved and (current_loader["sha256"], current_loader["size_bytes"]) == (saved_loader["sha256"], saved_loader["size_bytes"]),
                "source/effective model identity changed")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"], "source batch/image identity changed")
        tails = _prefix_tokens(raw, TARGET_OFFSET, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        require(native["input_ids"].shape == (4, FULL_WIDTH) and native["position_ids"].shape == (3, 4, FULL_WIDTH) and
                native["input_ids"][TARGET, SOURCE[0]:SOURCE[1]].tolist() == ROW and
                native["input_ids"][TARGET, DEST[0]:DEST[1]].tolist() == ROW and
                native["input_ids"][TARGET, WIDTH:FULL_WIDTH].tolist() == S, "physical row/S placement changed")
        require(all(tensor_hash(native[k]) == previous_manifest["native_input_hashes"][k] for k in
                    ("input_ids", "attention_mask", "position_ids")), "accepted native replay inputs changed")
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer_capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_key_phase-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ("probes/training_set_completion/recurrence_native_row_mass.py",
                        "probes/training_set_completion/recurrence_history_cache_partition.py",
                        "probes/training_set_completion/recurrence_donor_tracking.py",
                        "probes/training_set_completion/recurrence_position_history.py",
                        "probes/training_set_completion/recurrence_written_content.py",
                        "probes/training_set_completion/untied_shared.py",
                        "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py",
                        "src/inference/bound_requests.py", "src/qwen/untied_embeddings.py",
                        inspect.getfile(cache_utils), inspect.getfile(modeling_qwen3_vl), inspect.getfile(sdpa_attention))]
        captures = [preserve_source(p, run_root=OUT, relative_name=str(p) if not p.is_absolute() else f"transformers/{p.name}") for p in dependencies]
        require(literal_binding(captures[9])["sha256"] == current_loader["sha256"], "loader source capture changed")
        manifest = {"schema": "recurrence_key_phase.cells.v1", "status": "frozen_before_forward", "source": state["source"],
                    "producer": literal_binding(Path(__file__)), "producer_capture": literal_binding(producer_capture),
                    "dependency_captures": [literal_binding(p) for p in captures], "transformers_version": transformers.__version__,
                    "model_identity": identity, "native_batch_identity": input_identity(batch), "source_panel": literal_binding(PANEL),
                    "target_batch": TARGET, "target_offset": TARGET_OFFSET, "current_S": S, "prefill_width": WIDTH,
                    "source_block": list(SOURCE), "destination_block": list(DEST),
                    "native_input_hashes": {k: tensor_hash(native[k]) for k in ("input_ids", "attention_mask", "position_ids")},
                    "suffix_hashes": {k: tensor_hash(v) for k, v in
                    {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH], "attention_mask": native["attention_mask"],
                     "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH],
                     "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device)}.items()},
                    "conditions": {"NN": ["new", "new"], "OO": ["old", "old"],
                                   "ON": ["old", "new"], "NO": ["new", "old"]},
                    "model_forward_cap_cumulative": MAX_FORWARDS, "prior_model_forwards": prior_forwards,
                    "prior_elapsed_seconds": prior_seconds,
                    "model_seconds_cap": MAX_SECONDS, "parity_atol": ATOL,
                    "intermediate_headout_atol": ATOL, "target_payload_cap_bytes": 256 << 20}
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=literal_binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        clock = time.monotonic()
        def count(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(prior_forwards + state["model_forwards"] <= MAX_FORWARDS and
                    prior_seconds + time.monotonic() - clock <= MAX_SECONDS, "cumulative forward/time cap exceeded")
        def vision(*_):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= 1, "unexpected S image pass")
        counters = [q.model.register_forward_pre_hook(count, with_kwargs=True), q.model.model.visual.register_forward_pre_hook(vision)]
        cells, traces = {}, {}
        try:
            with torch.inference_mode():
                cache = DynamicCache()
                prefill = dict(native)
                prefill.update(input_ids=native["input_ids"][:, :WIDTH], attention_mask=native["attention_mask"][:, :WIDTH],
                               position_ids=native["position_ids"][:, :, :WIDTH], cache_position=torch.arange(WIDTH, device=device),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                phase_seen, pre_seen, pre_handles = [], {}, []
                pre_norm = [dict() for _ in range(LAYERS)]
                def pre_rotary(_module, _args, output):
                    cos, sin = output
                    require(cos.shape == sin.shape == (4, WIDTH, HEAD_DIM), "historical rotary output shape changed")
                    phase_seen.append({"old_cos": cos[TARGET, SOURCE[0]:SOURCE[1]].detach().clone(),
                                       "old_sin": sin[TARGET, SOURCE[0]:SOURCE[1]].detach().clone(),
                                       "new_cos": cos[TARGET, DEST[0]:DEST[1]].detach().clone(),
                                       "new_sin": sin[TARGET, DEST[0]:DEST[1]].detach().clone()})
                pre_handles.append(q.model.model.language_model.rotary_emb.register_forward_hook(pre_rotary))
                for i, layer in enumerate(q.model.model.language_model.layers):
                    def normalized_key(_module, _args, output, index=i):
                        require(output.shape == (4, WIDTH, KV_HEADS, HEAD_DIM), "historical normalized K shape changed")
                        pre_norm[index] = {"old": output[TARGET, SOURCE[0]:SOURCE[1]].transpose(0, 1).detach().clone(),
                                           "new": output[TARGET, DEST[0]:DEST[1]].transpose(0, 1).detach().clone()}
                    pre_handles.append(layer.self_attn.k_norm.register_forward_hook(normalized_key))
                pre_seen, standard_handles = hooks(q.model, prefill["input_ids"], prefill["position_ids"])
                pre_handles.extend(standard_handles)
                try:
                    output = q.model(**prefill)
                finally:
                    for handle in pre_handles:
                        handle.remove()
                require(output.past_key_values is cache and state["vision_forwards"] == 1 and len(phase_seen) == 1 and
                        all(set(record) == {"old", "new"} for record in pre_norm), "one-pass pre-K/phase capture invalid")
                check_cache(cache, width=WIDTH)
                require(len(pre_seen["embedding_inputs"]) == len(pre_seen["rotary_positions"]) == 1 and
                        len(pre_seen["masks"]) == len(pre_seen["cache_slots"]) == 2, "prefill token/position/mask attestation invalid")
                native_digest = cache_digest(cache)
                native_blocks = block_hashes(cache)
                phase = phase_seen[0]
                candidate = {name: [] for name in ("NN", "OO", "ON", "NO")}
                phase_errors = []
                phase_artifact = {"actual_phase": {k: v.float().cpu() for k, v in phase.items()}, "layers": []}
                for i, layer in enumerate(cache.layers):
                    old_post = layer.keys[TARGET, :, SOURCE[0]:SOURCE[1], :]
                    new_post = layer.keys[TARGET, :, DEST[0]:DEST[1], :]
                    keys, errors = make_keys(pre_norm[i]["old"], pre_norm[i]["new"], phase["old_cos"], phase["old_sin"],
                                             phase["new_cos"], phase["new_sin"], old_post, new_post)
                    qualification = phase_qualification(pre_norm[i]["old"], pre_norm[i]["new"], old_post, new_post,
                                                        phase["new_cos"], phase["new_sin"], phase["old_cos"],
                                                        phase["old_sin"], keys, errors)
                    for name in candidate:
                        candidate[name].append(keys[name])
                    phase_errors.append(qualification)
                    phase_artifact["layers"].append({"pre_old": pre_norm[i]["old"].float().cpu(), "pre_new": pre_norm[i]["new"].float().cpu(),
                                                      "post_old": old_post.float().cpu(), "post_new": new_post.float().cpu(),
                                                      "native_V_dest": layer.values[TARGET, :, DEST[0]:DEST[1], :].float().cpu(),
                                                      "candidate_keys": {name: keys[name].float().cpu() for name in candidate},
                                                      "phase_qualification": qualification})
                require(not torch.equal(phase["old_cos"], phase["new_cos"]) or not torch.equal(phase["old_sin"], phase["new_sin"]),
                        "old and new observed phases unexpectedly identical")
                torch.save(phase_artifact, OUT / "phase-and-prekeys.pt")
                write(OUT / "prefill-readback.json", {"consumed": pre_seen, "cache_digest": native_digest,
                                                       "block_hashes": native_blocks, "phase_errors": phase_errors,
                                                       "phase_and_prekeys": literal_binding(OUT / "phase-and-prekeys.pt"),
                                                       "cache_length": cache.get_seq_length()})
                state["prefill_readback"] = literal_binding(OUT / "prefill-readback.json")
                write(OUT / "receipt.json", state)
                for i, qualification in enumerate(phase_errors):
                    require(qualification["qualified"], f"layer {i} observed pre/post RoPE qualification failed: {qualification}")
                cache_position = torch.arange(WIDTH, FULL_WIDTH, device=device)
                for name in ("NN", "OO", "ON", "NO"):
                    with patch_key(cache, candidate[name], native_digest):
                        expected_blocks = block_hashes(cache)
                        for i in range(LAYERS):
                            require(expected_blocks[i]["source"] == native_blocks[i]["source"] and
                                    expected_blocks[i]["dest"]["values"] == native_blocks[i]["dest"]["values"] and
                                    expected_blocks[i]["dest"]["keys"] == tensor_hash(candidate[name][i]),
                                    "source/companion/native-V/K patch invariant changed")
                        suffix = {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH], "attention_mask": native["attention_mask"],
                                  "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH], "cache_position": cache_position,
                                  "past_key_values": cache, "use_cache": True, "return_dict": True, "logits_to_keep": SUFFIX}
                        consumed, standard = hooks(q.model, suffix["input_ids"], suffix["position_ids"])
                        record, rope, observers = suffix_observers(q.model, cache, expected_blocks, candidate if name == "NN" else None)
                        try:
                            output = q.model(**suffix)
                            logits = output.logits[TARGET, -1].detach().float().cpu()
                        finally:
                            for handle in standard + observers:
                                handle.remove()
                        require(torch.isfinite(logits).all() and cache.get_seq_length() == FULL_WIDTH and state["vision_forwards"] == 1,
                                "current S output/cache/vision invalid")
                        require(len(rope) == 1 and len(consumed["embedding_inputs"]) == len(consumed["rotary_positions"]) == 1 and
                                len(consumed["masks"]) == len(consumed["cache_slots"]) == 2 and len(record) == LAYERS,
                                "current S observation incomplete")
                        for i, layer_record in enumerate(record):
                            require(layer_record["cache_length_before"] == WIDTH and
                                    layer_record["dest_hashes"] == expected_blocks[i]["dest"] and
                                    layer_record["source_hashes"] == expected_blocks[i]["source"],
                                    "actual consumed layer cache changed")
                        if name != "NN":
                            for field in ("embedding_inputs", "rotary_positions", "masks", "cache_slots"):
                                require(consumed[field] == cells["NN"]["consumed"][field], f"{name} consumed S/position/mask changed")
                            require(all(record[i]["mask_hash"] == traces["NN"][i]["mask_hash"] for i in range(LAYERS)),
                                    f"{name} changed an intermediate decoder mask")
                        trace_path = OUT / f"{name}-intermediates.pt"
                        torch.save({"layers": record, "actual_current_phase": {"cos": rope[0][0].float().cpu(),
                                                                                "sin": rope[0][1].float().cpu()}}, trace_path)
                        traces[name] = record
                        top = torch.topk(logits, 5)
                        vector_path = OUT / f"{name}.pt"
                        torch.save(logits, vector_path)
                        cells[name] = {"full_logits": literal_binding(vector_path), "intermediates": literal_binding(trace_path),
                                       "argmax_token": int(top.indices[0]), "top1_top2_gap": float(top.values[0] - top.values[1]),
                                       "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices, top.values, strict=True)],
                                       "z38_minus_z999": float(logits[151708] - logits[152669]),
                                       "absolute_coordinates": probability_readback(logits, (0, 38, 640, 999)),
                                       "consumed": consumed, "layer_headout_reconstruction_max_abs":
                                       [layer["attention_reconstruction_max_abs"] for layer in record],
                                       "cache_length_after": cache.get_seq_length()}
                        state["last_cell"] = name
                        write(OUT / "partial-results.json", {"status": "running", "cells": cells})
                        write(OUT / "receipt.json", state)
                    cells[name]["native_cache_restored_sha256"] = hashlib.sha256(json.dumps(native_digest, sort_keys=True).encode()).hexdigest()
                    if name in references:
                        accepted = torch.load(references[name], map_location="cpu", weights_only=True)
                        cells[name]["max_abs_vs_accepted_full_vocab"] = float((logits - accepted).abs().max())
                        expected_argmax = previous_result["cells"][{"NN": "N", "OO": "K"}[name]]["argmax_token"]
                        require(cells[name]["max_abs_vs_accepted_full_vocab"] <= ATOL and cells[name]["argmax_token"] == expected_argmax,
                                f"{name} full-vector anchor qualification failed")
                require(state["model_forwards"] == 5 and state["vision_forwards"] == 1 and time.monotonic() - clock <= MAX_SECONDS,
                        "frozen forward/time budget changed")
                d = {name: cell["z38_minus_z999"] for name, cell in cells.items()}
                contrast = {"phase_new_minus_old_old_prekey": d["ON"] - d["OO"],
                            "phase_new_minus_old_new_prekey": d["NN"] - d["NO"],
                            "old_minus_new_prekey_old_phase": d["OO"] - d["NO"],
                            "old_minus_new_prekey_new_phase": d["ON"] - d["NN"],
                            "factorial_interaction": d["ON"] - d["OO"] - d["NN"] + d["NO"]}
                max_attention_error = max(max(cell["layer_headout_reconstruction_max_abs"]) for cell in cells.values())
                intermediate = {"status": "candidate" if max_attention_error <= ATOL else "HOLD",
                                "attention_output_max_abs": max_attention_error, "attention_output_atol": ATOL}
                decomposition = {name: [] for name in ("NN", "OO", "ON", "NO")}
                decomposition_summary = {name: [] for name in decomposition}
                brute_max = 0.0
                for i in range(LAYERS):
                    nn = traces["NN"][i]
                    q_nn = nn["q_post_reconstructed"]
                    p_nn = nn["attention_probabilities_reconstructed"][:, :, DEST[0]:DEST[1]]
                    v_nn = nn["native_V_destination"]
                    k_nn = phase_artifact["layers"][i]["candidate_keys"]["NN"]
                    o_nn_actual, o_nn_reconstructed = nn["actual_headout"], nn["reconstructed_headout"]
                    for name in decomposition:
                        k_changed = phase_artifact["layers"][i]["candidate_keys"][name]
                        fixed_actual, denominator = fixed_native_headout(q_nn, p_nn, o_nn_actual, v_nn, k_nn, k_changed,
                                                                           HEAD_DIM**-0.5)
                        fixed_reconstructed, _ = fixed_native_headout(q_nn, p_nn, o_nn_reconstructed, v_nn, k_nn, k_changed,
                                                                       HEAD_DIM**-0.5)
                        brute = o_nn_reconstructed if name == "NN" else nn["fixed_native_brute_headout"][name]
                        brute_error = float((fixed_reconstructed - brute.double()).abs().max())
                        brute_max = max(brute_max, brute_error)
                        actual = traces[name][i]["actual_headout"].double()
                        direct = fixed_actual - o_nn_actual.double()
                        feedback = actual - fixed_actual
                        delta = actual - o_nn_actual.double()
                        decomposition[name].append({"fixed_native_headout": fixed_actual.float(), "direct": direct.float(),
                                                    "feedback_remainder": feedback.float(), "actual_delta": delta.float(),
                                                    "denominator": denominator.float()})
                        decomposition_summary[name].append({"layer": i, "fixed_brute_max_abs": brute_error,
                                                            "identity_residual_max_abs": float((direct + feedback - delta).abs().max()),
                                                            "metrics": vector_metrics(direct, feedback, delta)})
                intermediate["fixed_state_brute_max_abs"] = brute_max
                if brute_max > ATOL:
                    intermediate["status"] = "HOLD"
                torch.save(decomposition, OUT / "fixed-native-decomposition.pt")
                write(OUT / "intermediate-summary.json", {"status": intermediate["status"], "qualification": intermediate,
                                                           "per_layer": decomposition_summary})
                artifact_bytes = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
                require(artifact_bytes <= 256 << 20, "target-only intermediate payload cap exceeded")
                result = {"schema": "recurrence_key_phase.result.v1", "status": "candidate", "cells": cells,
                          "contrast": contrast, "intermediate": intermediate,
                          "intermediate_summary": literal_binding(OUT / "intermediate-summary.json"),
                          "decomposition": literal_binding(OUT / "fixed-native-decomposition.pt"),
                          "phase_and_prekeys": literal_binding(OUT / "phase-and-prekeys.pt"),
                          "prefill_readback": literal_binding(OUT / "prefill-readback.json"),
                          "source_manifest": literal_binding(OUT / "source-to-cell.json"),
                          "artifact_bytes_before_result": artifact_bytes,
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "model_seconds": time.monotonic() - clock}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"),
                             model_seconds=result["model_seconds"], elapsed_seconds=time.monotonic() - started,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), artifact_bytes=sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "result": str(OUT / "result.json"), "forwards": state["model_forwards"]}))
        finally:
            for handle in counters:
                handle.remove()
    except BaseException as error:
        state.update(status="technical_invalid", error=repr(error), elapsed_seconds=time.monotonic() - started)
        write(OUT / "receipt.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); print("selfcheck ok")
    else:
        run(args.device)
