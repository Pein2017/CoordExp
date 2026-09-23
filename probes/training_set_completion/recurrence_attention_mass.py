"""Cross live-query row log-mass with historical-key profile at two native exits."""
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
import transformers
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_donor_tracking import probability_readback, require, write
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest, check_cache
from probes.training_set_completion.recurrence_key_phase import rotate
from probes.training_set_completion.recurrence_position_history import PANEL as VAL_PANEL, source_and_rows
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.config.inference import InferConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.bound_requests import build_bound_native_requests
from src.inference.inputs import plan_examples
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
OUT = BASE / "2026-09-22-recurrence-attention-mass/attempt-001"
UNIT = Path("research/experiments/2026-09-22-recurrence-attention-mass/unit.md")
VAL_ROOT = BASE / "2026-09-22-recurrence-key-phase/attempt-003"
VAL_ACCEPTANCE = BASE / "2026-09-22-recurrence-key-phase/lead-checks/phase-readback.json"
TRAIN_ROOT = BASE / "2026-09-22-recurrence-phase-transfer/attempt-002"
TRAIN_ACCEPTANCE = BASE / "2026-09-22-recurrence-phase-transfer/lead-checks/transfer-readback.json"
TRAIN_SELECTION = BASE / "2026-09-22-recurrence-phase-transfer/selection/selected-case-metadata-v1.json"
MAX_FORWARDS, MAX_SECONDS, ATOL = 12, 1200, 2e-4
LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM = 28, 16, 8, 128
RULES = {"NN": ("new", "new"), "OO": ("old", "old"),
         "ON": ("old", "new"), "NO": ("new", "old")}
CASES = {
    "val": {"target": 2, "width": 2121, "full": 2127, "suffix": 6,
            "source": (2103, 2112), "dest": (2112, 2121), "root": VAL_ROOT,
            "acceptance": VAL_ACCEPTANCE, "new_token": 152669, "old_token": 151708,
            "diagnostic_token": None},
    "train": {"target": 1, "width": 1542, "full": 1546, "suffix": 4,
              "source": (1524, 1533), "dest": (1533, 1542), "root": TRAIN_ROOT,
              "acceptance": TRAIN_ACCEPTANCE, "new_token": 152018, "old_token": 152261,
              "diagnostic_token": 152020},
}


def same_binding(path, expected):
    actual = literal_binding(Path(path))
    return all(actual[k] == expected[k] for k in ("path", "sha256", "size_bytes"))


def row_scores(post_q, native_k, old_k, scale):
    require(post_q.ndim == native_k.ndim == old_k.ndim == 3 and
            native_k.shape == old_k.shape and native_k.shape[1] == 9 and
            post_q.shape[-1] == native_k.shape[-1] and post_q.shape[0] % native_k.shape[0] == 0,
            "live Q or historical key block shape changed")
    group = post_q.shape[0] // native_k.shape[0]
    new_scores = torch.matmul(post_q, native_k.repeat_interleave(group, dim=0).transpose(-1, -2)) * scale
    old_scores = torch.matmul(post_q, old_k.repeat_interleave(group, dim=0).transpose(-1, -2)) * scale
    return {"new": new_scores, "old": old_scores}


def row_rule(scores, *, mass, profile):
    require(mass in ("new", "old") and profile in ("new", "old") and
            scores["new"].shape == scores["old"].shape, "invalid mass/profile assignment")
    lse = {kind: torch.logsumexp(value, dim=-1) for kind, value in scores.items()}
    bias = lse[mass] - lse[profile]
    after = scores[profile] + bias.unsqueeze(-1)
    return {"bias": bias, "native_lse": lse, "post_lse_error": float((torch.logsumexp(after, dim=-1) - lse[mass]).abs().max()),
            "centered_profile_error": float(((after - torch.logsumexp(after, dim=-1).unsqueeze(-1)) -
                                               (scores[profile] - lse[profile].unsqueeze(-1))).abs().max()),
            "after_scores": after}


def float_mask(bool_mask, q_heads):
    require(bool_mask.dtype == torch.bool and bool_mask.ndim == 4 and bool_mask.shape[1] == 1,
            "actual causal mask is not the expected bool route")
    base = torch.zeros(bool_mask.shape, device=bool_mask.device, dtype=torch.float32)
    base.masked_fill_(~bool_mask, -torch.inf)
    return base.expand(-1, q_heads, -1, -1).clone()


def apply_bias(mask, base_bool, bias, *, target, dest):
    require(mask.ndim == 4 and mask.dtype == torch.float32 and base_bool.dtype == torch.bool and
            mask.shape[0] == base_bool.shape[0] and mask.shape[2:] == base_bool.shape[2:] and
            0 <= target < mask.shape[0] and dest[1] - dest[0] == 9 and dest[1] <= mask.shape[-1] - mask.shape[-2] and
            bias.shape == (mask.shape[1], mask.shape[2]) and
            bool(torch.all(base_bool[target, 0, :, dest[0]:dest[1]])),
            "wrong destination/head/current-S bias shape or visibility")
    mask[target, :, :, dest[0]:dest[1]].add_(bias.unsqueeze(-1))
    base = float_mask(base_bool, mask.shape[1])
    outside = (torch.equal(mask[..., :dest[0]], base[..., :dest[0]]) and
               torch.equal(mask[..., dest[1]:], base[..., dest[1]:]))
    companions = [i for i in range(mask.shape[0]) if i != target]
    unchanged_companions = torch.equal(mask[companions], base[companions])
    visibility = torch.equal(torch.isfinite(mask), base_bool.expand_as(mask))
    return {"outside_row_exact": outside, "companions_exact": unchanged_companions,
            "visibility_exact": visibility}


def check_rule(scores, bias, mass, profile):
    desired = torch.logsumexp(scores[mass], dim=-1)
    actual = torch.logsumexp(scores[profile] + bias.unsqueeze(-1), dim=-1)
    centered = (scores[profile] + bias.unsqueeze(-1)) - actual.unsqueeze(-1)
    expected_centered = scores[profile] - torch.logsumexp(scores[profile], dim=-1).unsqueeze(-1)
    return float((actual - desired).abs().max()), float((centered - expected_centered).abs().max())


def fixed_history_snapshot(cache, target, source_start):
    others = [i for i in range(4) if i != target]
    return [{axis: {"others": getattr(layer, axis)[others].clone(),
                    "target_before_source": getattr(layer, axis)[target, :, :source_start, :].clone()}
             for axis in ("keys", "values")} for layer in cache.layers]


def fixed_history_equal(layer, saved, target, source_start):
    others = [i for i in range(4) if i != target]
    return all(torch.equal(getattr(layer, axis)[others], saved[axis]["others"]) and
               torch.equal(getattr(layer, axis)[target, :, :source_start, :],
                           saved[axis]["target_before_source"]) for axis in ("keys", "values"))


@contextmanager
def patch_profile(cache, blocks, digest, case):
    target, width, dest = case["target"], case["width"], case["dest"]
    require(dest == (width - 9, width) and len(blocks) == len(cache.layers), "profile patch boundary changed")
    saved = []
    try:
        for layer, block in zip(cache.layers, blocks, strict=True):
            require(block.shape == (KV_HEADS, 9, HEAD_DIM), "profile K shape changed")
            old = layer.keys[target, :, dest[0]:dest[1], :].clone()
            saved.append((layer, old))
            layer.keys[target, :, dest[0]:dest[1], :].copy_(block)
        yield
    finally:
        cache.crop(width)
        for layer, old in saved:
            layer.keys[target, :, dest[0]:dest[1], :].copy_(old)
        require(cache_digest(cache) == digest, "native historical cache not restored")


def observed_attention(attn, cache, case, cell, native_k, old_k, native_blocks, fixed_history):
    """Actuate the installed Qwen attention using its live q_norm output and mask."""
    target, width, full, suffix, dest, source = (case[k] for k in
                                                  ("target", "width", "full", "suffix", "dest", "source"))
    mass, profile = RULES[cell]
    q_heads, kv_heads, dim = attn.config.num_attention_heads, attn.config.num_key_value_heads, attn.head_dim
    require(q_heads == Q_HEADS and kv_heads == KV_HEADS and dim == HEAD_DIM,
            "installed Qwen attention head dimensions changed")
    seen, handles = {}, []
    def before(_module, args, kwargs):
        require(kwargs.get("past_key_values") is cache and cache.get_seq_length(attn.layer_idx) == width,
                "attention consumed wrong historical cache")
        original = kwargs.get("attention_mask")
        slots = kwargs.get("cache_position")
        phase = kwargs.get("position_embeddings")
        require(isinstance(original, torch.Tensor) and original.shape == (4, 1, suffix, full) and
                isinstance(slots, torch.Tensor) and torch.equal(slots, torch.arange(width, full, device=slots.device)) and
                isinstance(phase, tuple) and len(phase) == 2 and
                phase[0].shape == phase[1].shape == (4, suffix, dim),
                "actual mask/current positions/rotary phase changed")
        layer = cache.layers[attn.layer_idx]
        actual_profile = layer.keys[target, :, dest[0]:dest[1], :]
        expected_profile = native_k if profile == "new" else old_k
        require(torch.equal(actual_profile, expected_profile) and
                torch.equal(layer.values[target, :, dest[0]:dest[1], :], native_blocks["dest_V"]) and
                torch.equal(layer.keys[target, :, source[0]:source[1], :], native_blocks["source_K"]) and
                torch.equal(layer.values[target, :, source[0]:source[1], :], native_blocks["source_V"]) and
                fixed_history_equal(layer, fixed_history, target, source[0]),
                "actual historical K/V profile, source or companions changed")
        changed = float_mask(original, q_heads)
        kwargs["attention_mask"] = changed
        seen.update(mask=changed, original_bool=original, phase=phase,
                    original_mask_hash=tensor_hash(original), slots_hash=tensor_hash(slots),
                    native_destination_V_hash=tensor_hash(native_blocks["dest_V"]),
                    source_K_hash=tensor_hash(native_blocks["source_K"]),
                    historical_and_companions_exact=True, q_norm_calls=0)
        return args, kwargs
    def q_normalized(_module, _args, output):
        require("mask" in seen and output.shape == (4, suffix, q_heads, dim) and seen["q_norm_calls"] == 0,
                "actual Q hook ordering/count changed")
        seen["q_norm_calls"] += 1
        cos, sin = seen["phase"]
        post_q = rotate(output[target].transpose(0, 1), cos[target], sin[target])
        scores = row_scores(post_q, native_k, old_k, attn.scaling)
        rule = row_rule(scores, mass=mass, profile=profile)
        flags = apply_bias(seen["mask"], seen["original_bool"], rule["bias"], target=target, dest=dest)
        seen.update(post_q=post_q.detach(), scores=scores, rule=rule, mask_flags=flags,
                    float_mask_hash=tensor_hash(seen["mask"]),
                    original_phase_hashes=[tensor_hash(x) for x in seen["phase"]])
    def output_projection(_module, args):
        require("post_q" in seen and cache.get_seq_length(attn.layer_idx) == full,
                "attention output observed before Q/bias/cache append")
        layer = cache.layers[attn.layer_idx]
        keys = layer.keys[target].repeat_interleave(q_heads // kv_heads, dim=0)
        values = layer.values[target].repeat_interleave(q_heads // kv_heads, dim=0)
        scores = torch.matmul(seen["post_q"], keys.transpose(-1, -2)) * attn.scaling
        probs = torch.softmax(scores + seen["mask"][target], dim=-1)
        rebuilt = torch.matmul(probs, values)
        actual = args[0][target].reshape(suffix, q_heads, dim).transpose(0, 1)
        seen["headout_max_abs"] = float((rebuilt - actual).abs().max())
        seen["headout_nonfinite"] = not bool(torch.isfinite(rebuilt).all())
        seen["cache_length_after_append"] = cache.get_seq_length(attn.layer_idx)
    handles.extend((attn.register_forward_pre_hook(before, with_kwargs=True),
                    attn.q_norm.register_forward_hook(q_normalized),
                    attn.o_proj.register_forward_pre_hook(output_projection)))
    return seen, handles


def selfcheck():
    """Exercise the installed Qwen attention caller, not only the row formula."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
    torch.manual_seed(260922)
    config = Qwen3VLTextConfig(hidden_size=2048, num_attention_heads=Q_HEADS,
                              num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    attention = Qwen3VLTextAttention(config, 0).eval()
    case = {"target": 2, "width": 18, "full": 20, "suffix": 2,
            "source": (0, 9), "dest": (9, 18)}
    base_keys, base_values = torch.randn(4, KV_HEADS, 18, HEAD_DIM), torch.randn(4, KV_HEADS, 18, HEAD_DIM)
    native_k = base_keys[2, :, 9:18].clone()
    old_k = native_k + torch.randn_like(native_k) * .3
    blocks = {"source_K": base_keys[2, :, 0:9].clone(), "source_V": base_values[2, :, 0:9].clone(),
              "dest_V": base_values[2, :, 9:18].clone()}
    def new_cache():
        return DynamicCache(((base_keys.clone(), base_values.clone()),))
    hidden = torch.randn(4, 2, 2048)
    cos, sin = torch.ones(4, 2, HEAD_DIM), torch.zeros(4, 2, HEAD_DIM)
    allowed = torch.ones(4, 1, 2, 20, dtype=torch.bool)
    allowed[:, :, 0, 19] = False
    slots = torch.arange(18, 20)
    def call(cache, mask):
        return attention(hidden, position_embeddings=(cos, sin), attention_mask=mask,
                         past_key_values=cache, cache_position=slots)[0]
    with torch.inference_mode():
        boolean_output = call(new_cache(), allowed)
        zero_float_output = call(new_cache(), float_mask(allowed, Q_HEADS))
        require(float((boolean_output - zero_float_output).abs().max()) <= 2e-5,
                "installed Qwen bool/float-zero caller differs")
        cache = new_cache()
        fixed = fixed_history_snapshot(cache, 2, 0)[0]
        seen, handles = observed_attention(attention, cache, case, "ON", native_k, old_k, blocks, fixed)
        q_proj_calls = []
        handles.append(attention.q_proj.register_forward_hook(lambda *_: q_proj_calls.append(1)))
        try:
            hooked = call(cache, allowed)
        finally:
            for handle in handles:
                handle.remove()
        require(len(q_proj_calls) == seen["q_norm_calls"] == 1 and seen["headout_max_abs"] <= 2e-5 and
                seen["rule"]["post_lse_error"] <= 2e-5 and seen["rule"]["centered_profile_error"] <= 2e-5 and
                all(seen["mask_flags"].values()), "installed Qwen live-query hook failed")
        direct_mask = float_mask(allowed, Q_HEADS)
        apply_bias(direct_mask, allowed, seen["rule"]["bias"], target=2, dest=(9, 18))
        direct = call(new_cache(), direct_mask)
        require(float((hooked - direct).abs().max()) <= 2e-5,
                "Q-hook bias differs from direct float-mask caller")
        for wrong_dest, wrong_bias in (((9, 17), seen["rule"]["bias"]),
                                       ((9, 18), seen["rule"]["bias"][:-1])):
            try:
                apply_bias(float_mask(allowed, Q_HEADS), allowed, wrong_bias, target=2, dest=wrong_dest)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong destination/head shape admitted")
        scores = seen["scores"]
        require(check_rule(scores, torch.zeros_like(seen["rule"]["bias"]), "old", "new")[0] > 1e-4 and
                check_rule(scores, seen["rule"]["bias"].roll(1, 0), "old", "new")[0] > 1e-4,
                "wrong mass assignment or head permutation escaped row-LSE check")
        wrong_cache = new_cache()
        wrong_seen, wrong_handles = observed_attention(attention, wrong_cache, case, "OO",
                                                        native_k, old_k, blocks,
                                                        fixed_history_snapshot(wrong_cache, 2, 0)[0])
        try:
            try:
                call(wrong_cache, allowed)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong cached profile admitted")
        finally:
            for handle in wrong_handles:
                handle.remove()


def prepare_case(name, q, identity, device):
    case = dict(CASES[name])
    root = case["root"]
    accepted = json.loads(case["acceptance"].read_text())
    expected_status = ("lead-accepted-crossing-and-intermediates" if name == "val"
                       else "lead-accepted-directional-transfer-counterexample")
    require(accepted["status"] == expected_status, f"{name} source contrast not lead-accepted")
    result_path, manifest_path = root / "result.json", root / "source-to-cell.json"
    result, manifest = json.loads(result_path.read_text()), json.loads(manifest_path.read_text())
    require(result["status"] == "candidate", f"{name} source candidate changed")
    native_packet_path = root / "phase-and-prekeys.pt"
    require(same_binding(native_packet_path, result["phase_and_prekeys"]),
            f"{name} accepted historical key packet changed")
    for cell, predecessor in (("NN", "NN"), ("OO", "NO")):
        require(same_binding(root / f"{predecessor}.pt", result["cells"][predecessor]["full_logits"]),
                f"{name} diagonal reference {predecessor} changed")
    if name == "val":
        selected, raw, trace, receipt, panel, group, current_S, offsets, _ = source_and_rows()
        require(offsets[1] == 807 and len(current_S) == case["suffix"], "val current S offset changed")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        source = {"selection": manifest["source"]["selection"], "panel": literal_binding(VAL_PANEL),
                  **{k: selected[k] for k in ("image", "raw", "trace", "runtime_receipt")}}
    else:
        selection = json.loads(TRAIN_SELECTION.read_text())
        require(selection["status"] == "candidate", "train selection changed")
        selected = selection["root_selected"]
        raw = json.loads(Path(selected["raw"]["path"]).read_text())["rows"]
        receipt = json.loads(Path(selected["runtime_receipt"]["path"]).read_text())
        panel = json.loads(Path(selected["source_panel"]["path"]).read_text())
        group = next(x for x in panel["groups"] if x["key"] == "new-14")
        current_S = selected["common_current_row_prefix_tokens"]
        require(current_S == [151646, 8987, 151647, 151648], "train current S changed")
        config_model = InferConfig.model_validate(panel["configs"]["untied"])
        examples = [raw_example_from_jsonl_row(item["input_record"],
                    jsonl_path=Path(group["input_jsonl"]), row_number=int(item["row_index"]) + 1,
                    raw_line=json.dumps(item["input_record"])) for item in group["cases"]]
        planned = plan_examples(examples, config=config_model, components=q,
                                row_indices=[int(item["row_index"]) for item in group["cases"]])
        requests = [item.request for item in planned]
        source = {"selection": literal_binding(TRAIN_SELECTION), "panel": selected["source_panel"],
                  **{k: selected[k] for k in ("image", "raw", "trace", "runtime_receipt")}}
    current, saved = dict(identity), dict(receipt["identity"])
    loader, saved_loader = current.pop("loader_source"), saved.pop("loader_source")
    require(current == saved and (loader["sha256"], loader["size_bytes"]) ==
            (saved_loader["sha256"], saved_loader["size_bytes"]), f"{name} model/effective-row identity changed")
    batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
    batch_identity = input_identity(batch)
    require(batch_identity == receipt["input_identity"], f"{name} native image/batch identity changed")
    action_offset = 807 if name == "val" else 184
    tails = _prefix_tokens(raw, action_offset, int(q.tokenizer.pad_token_id))
    histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
    native = exact_history_inputs(q.model, batch.inputs, histories,
                                  pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
    require(native["input_ids"].shape == (4, case["full"]) and
            native["position_ids"].shape == (3, 4, case["full"]) and
            native["input_ids"][case["target"], case["width"]:case["full"]].tolist() == current_S and
            all(tensor_hash(native[key]) == manifest["native_input_hashes"][key]
                for key in ("input_ids", "attention_mask", "position_ids")),
            f"{name} native history/position identity changed")
    packet = torch.load(native_packet_path, map_location="cpu", weights_only=True)
    require(len(packet["layers"]) == LAYERS, f"{name} phase packet incomplete")
    if name == "val":
        mask_path = root / "NN-intermediates.pt"
        require(same_binding(mask_path, result["cells"]["NN"]["intermediates"]),
                "accepted val NN mask readback changed")
        mask_readback = torch.load(mask_path, map_location="cpu", weights_only=True)
        prior_mask_hashes = [layer["mask_hash"] for layer in mask_readback["layers"]]
        del mask_readback
    else:
        mask_path = root / "NN-readback.json"
        require(same_binding(mask_path, result["cells"]["NN"]["readback_binding"]),
                "accepted train NN mask readback changed")
        mask_readback = json.loads(mask_path.read_text())
        prior_mask_hashes = [mask_readback["layers"][str(i)]["mask_hash"] for i in range(LAYERS)]
    require(len(prior_mask_hashes) == LAYERS and all(len(x) == 64 for x in prior_mask_hashes),
            f"{name} accepted NN mask hash list incomplete")
    case.update(native=native, batch=batch, packet=packet, current_S=current_S, source_bindings=source,
                accepted_result=result, accepted_manifest=manifest,
                prior_mask_hashes=prior_mask_hashes,
                accepted_paths={"result": literal_binding(result_path), "manifest": literal_binding(manifest_path),
                                "phase_packet": literal_binding(native_packet_path),
                                "prior_NN_mask_readback": literal_binding(mask_path),
                                "NN_vector": literal_binding(root / "NN.pt"),
                                "OO_vector": literal_binding(root / "NO.pt"),
                                "acceptance": literal_binding(case["acceptance"])},
                original_batch_identity=batch_identity, original_receipt=receipt)
    return case


def run_case(name, case, q, forward, state, device):
    target, width, full, suffix, source, dest = (case[k] for k in
                                                ("target", "width", "full", "suffix", "source", "dest"))
    case_out = OUT / name
    case_out.mkdir()
    native = case["native"]
    cache = DynamicCache()
    prefill = dict(native)
    prefill.update(input_ids=native["input_ids"][:, :width], attention_mask=native["attention_mask"][:, :width],
                   position_ids=native["position_ids"][:, :, :width], cache_position=torch.arange(width, device=device),
                   past_key_values=cache, use_cache=True, logits_to_keep=1)
    phase_seen, pre_norm = [], [None] * LAYERS
    handles = []
    def rotary(_module, _args, output):
        cos, sin = output
        require(cos.shape == sin.shape == (4, width, HEAD_DIM), "historical phase shape changed")
        phase_seen.append({"old_cos": cos[target, source[0]:source[1]].detach().clone(),
                           "old_sin": sin[target, source[0]:source[1]].detach().clone(),
                           "new_cos": cos[target, dest[0]:dest[1]].detach().clone(),
                           "new_sin": sin[target, dest[0]:dest[1]].detach().clone()})
    handles.append(q.model.model.language_model.rotary_emb.register_forward_hook(rotary))
    for i, layer in enumerate(q.model.model.language_model.layers):
        def normalized(_module, _args, output, *, index=i):
            require(output.shape == (4, width, KV_HEADS, HEAD_DIM), "historical normalized K shape changed")
            pre_norm[index] = output[target, dest[0]:dest[1]].transpose(0, 1).detach().clone()
        handles.append(layer.self_attn.k_norm.register_forward_hook(normalized))
    consumed, standard = hooks(q.model, prefill["input_ids"], prefill["position_ids"])
    handles.extend(standard)
    try:
        output = forward(f"{name}:historical_prefill", prefill)
    finally:
        for h in handles:
            h.remove()
    require(output.past_key_values is cache and len(phase_seen) == 1 and all(x is not None for x in pre_norm) and
            len(consumed["embedding_inputs"]) == len(consumed["rotary_positions"]) == 1 and
            len(consumed["masks"]) == len(consumed["cache_slots"]) == 2,
            "native historical prefill/phase observation invalid")
    del output
    check_cache(cache, width=width)
    phase = phase_seen[0]
    accepted = case["packet"]
    live_phase_errors = {k: float((phase[k].cpu() - accepted["actual_phase"][k]).abs().max()) for k in phase}
    native_keys, old_keys, blocks = [], [], []
    prekey_errors = []
    for i, layer in enumerate(cache.layers):
        packet = accepted["layers"][i]
        native_key = layer.keys[target, :, dest[0]:dest[1], :]
        old_key = packet["candidate_keys"]["NO"].to(device)
        native_k_error = float((native_key.cpu() - packet["post_new"]).abs().max())
        prekey_error = float((pre_norm[i].cpu() - packet["pre_new"]).abs().max())
        V = layer.values[target, :, dest[0]:dest[1], :]
        V_error = float((V.cpu() - packet["native_V_dest"]).abs().max())
        prekey_errors.append({"layer": i, "native_K_max_abs": native_k_error,
                              "native_preK_max_abs": prekey_error, "native_V_max_abs": V_error})
        native_keys.append(native_key.detach().clone())
        old_keys.append(old_key)
        blocks.append({"source_K": layer.keys[target, :, source[0]:source[1], :].detach().clone(),
                       "source_V": layer.values[target, :, source[0]:source[1], :].detach().clone(),
                       "dest_V": V.detach().clone()})
    digest = cache_digest(cache)
    fixed = fixed_history_snapshot(cache, target, source[0])
    write(case_out / "prefill-readback.json", {"consumed": consumed,
                                                "accepted_phase_packet": case["accepted_paths"]["phase_packet"],
                                                "live_phase_errors": live_phase_errors,
                                                "per_layer_native_errors": prekey_errors,
                                                "cache_digest": digest, "cache_length": cache.get_seq_length()})
    state["last_prefill"] = name
    write(OUT / "receipt.json", state)
    require(max(live_phase_errors.values()) == 0 and
            all(max(x["native_K_max_abs"], x["native_preK_max_abs"], x["native_V_max_abs"]) == 0
                for x in prekey_errors), f"{name} native phase/K/V differs from accepted source")
    cells = {}
    for cell in ("NN", "OO", "ON", "NO"):
        mass, profile = RULES[cell]
        profile_keys = native_keys if profile == "new" else old_keys
        with patch_profile(cache, profile_keys, digest, case):
            suffix_inputs = {"input_ids": native["input_ids"][:, width:full],
                             "attention_mask": native["attention_mask"],
                             "position_ids": native["position_ids"][:, :, width:full],
                             "cache_position": torch.arange(width, full, device=device),
                             "past_key_values": cache, "use_cache": True,
                             "return_dict": True, "logits_to_keep": suffix}
            standard_seen, standard = hooks(q.model, suffix_inputs["input_ids"], suffix_inputs["position_ids"])
            layer_seen, observers = [], []
            for i, layer in enumerate(q.model.model.language_model.layers):
                current, handles = observed_attention(layer.self_attn, cache, case, cell,
                                                       native_keys[i], old_keys[i], blocks[i], fixed[i])
                layer_seen.append(current)
                observers.extend(handles)
            try:
                output = forward(f"{name}:{cell}", suffix_inputs)
                logits = output.logits[target, -1].detach().float().cpu()
                companion_hash = tensor_hash(output.logits[[i for i in range(4) if i != target], -1].detach())
            finally:
                for h in standard + observers:
                    h.remove()
            require(torch.isfinite(logits).all() and cache.get_seq_length() == full and
                    len(standard_seen["embedding_inputs"]) == len(standard_seen["rotary_positions"]) == 1 and
                    len(standard_seen["masks"]) == len(standard_seen["cache_slots"]) == 2 and
                    all("headout_max_abs" in x and x["q_norm_calls"] == 1 for x in layer_seen),
                    f"{name}:{cell} current-S consumer observation incomplete")
            trace_layers, layer_metrics = [], []
            for i, observed in enumerate(layer_seen):
                rule = observed["rule"]
                errors = {"post_row_lse_max_abs": rule["post_lse_error"],
                          "centered_profile_max_abs": rule["centered_profile_error"],
                          "headout_max_abs": observed["headout_max_abs"]}
                layer_metrics.append({"layer": i, **errors, "headout_nonfinite": observed["headout_nonfinite"],
                                      "mask_flags": observed["mask_flags"],
                                      "original_bool_mask_hash": observed["original_mask_hash"],
                                      "consumed_float_mask_hash": observed["float_mask_hash"],
                                      "current_phase_hashes": observed["original_phase_hashes"],
                                      "cache_length_after_append": observed["cache_length_after_append"],
                                      "historical_and_companions_exact": observed["historical_and_companions_exact"]})
                trace_layers.append({"layer": i, "post_Q": observed["post_q"].float().cpu(),
                                     "new_phase_K": native_keys[i].float().cpu(),
                                     "old_phase_K": old_keys[i].float().cpu(),
                                     "native_destination_V": blocks[i]["dest_V"].float().cpu(),
                                     "new_row_scores": observed["scores"]["new"].float().cpu(),
                                     "old_row_scores": observed["scores"]["old"].float().cpu(),
                                     "post_bias_profile_scores": rule["after_scores"].float().cpu(),
                                     "row_logsumexp": {k: v.float().cpu() for k, v in rule["native_lse"].items()},
                                     "uniform_bias": rule["bias"].float().cpu(), "errors": errors})
            trace_path = case_out / f"{cell}-layers.pt"
            torch.save({"layers": trace_layers, "cell_mass_profile": [mass, profile]}, trace_path)
            vector_path = case_out / f"{cell}.pt"
            torch.save(logits, vector_path)
            readback = {"cell_mass_profile": [mass, profile], "consumed": standard_seen,
                        "per_layer": layer_metrics, "intermediates": literal_binding(trace_path),
                        "companion_final_logits_hash": companion_hash,
                        "cache_length_after": cache.get_seq_length()}
            readback_path = case_out / f"{cell}-readback.json"
            write(readback_path, readback)
            top = torch.topk(logits, 5)
            requested_bins = (38, 999) if name == "val" else (348, 350, 591)
            cells[cell] = {"full_logits": literal_binding(vector_path),
                           "readback": literal_binding(readback_path),
                           "intermediates": literal_binding(trace_path),
                           "argmax_token": int(top.indices[0]),
                           "top1_top2_gap": float(top.values[0] - top.values[1]),
                           "top5": [{"token": int(t), "logit": float(v)} for t, v in
                                    zip(top.indices, top.values, strict=True)],
                           "old_minus_new_margin": float(logits[case["old_token"]] - logits[case["new_token"]]),
                           "coordinate_probs": probability_readback(logits, requested_bins),
                           "diagnostic_350_logit": float(logits[case["diagnostic_token"]])
                           if case["diagnostic_token"] is not None else None,
                           "max_post_row_lse_abs": max(x["post_row_lse_max_abs"] for x in layer_metrics),
                           "max_centered_profile_abs": max(x["centered_profile_max_abs"] for x in layer_metrics),
                           "max_headout_abs": max(x["headout_max_abs"] for x in layer_metrics)}
            state["last_cell"] = f"{name}:{cell}"
            write(OUT / "partial-results.json", {"status": "running", "last_case": name, "cells": cells})
            write(OUT / "receipt.json", state)
            require(all(max(x["post_row_lse_max_abs"], x["centered_profile_max_abs"],
                            x["headout_max_abs"]) <= ATOL and not x["headout_nonfinite"] and
                        all(x["mask_flags"].values()) and x["historical_and_companions_exact"] and
                        x["cache_length_after_append"] == full and
                        x["original_bool_mask_hash"] == case["prior_mask_hashes"][i]
                        for i, x in enumerate(layer_metrics)),
                    f"{name}:{cell} actual mass/profile/attention/mask qualification failed")
            if cell != "NN":
                prior = json.loads((case_out / "NN-readback.json").read_text())
                require(standard_seen == prior["consumed"] and
                        companion_hash == prior["companion_final_logits_hash"] and
                        all(layer_metrics[i]["original_bool_mask_hash"] ==
                            prior["per_layer"][i]["original_bool_mask_hash"] and
                            layer_metrics[i]["current_phase_hashes"] ==
                            prior["per_layer"][i]["current_phase_hashes"] for i in range(LAYERS)),
                        f"{name}:{cell} changed S/mask/positions or companions")
        if cell in ("NN", "OO"):
            predecessor = "NN" if cell == "NN" else "NO"
            accepted_logits = torch.load(case["root"] / f"{predecessor}.pt", map_location="cpu", weights_only=True)
            cells[cell]["fullvector_max_abs_vs_accepted"] = float((logits - accepted_logits).abs().max())
            require(cells[cell]["fullvector_max_abs_vs_accepted"] <= ATOL and
                    cells[cell]["argmax_token"] == case["accepted_result"]["cells"][predecessor]["argmax_token"],
                    f"{name}:{cell} float-mask diagonal fullvector parity failed")
    d = {cell: data["old_minus_new_margin"] for cell, data in cells.items()}
    contrast = {"old_minus_new_mass_new_profile": d["ON"] - d["NN"],
                "old_minus_new_mass_old_profile": d["OO"] - d["NO"],
                "old_minus_new_profile_new_mass": d["NO"] - d["NN"],
                "old_minus_new_profile_old_mass": d["OO"] - d["ON"],
                "factorial_interaction": d["OO"] - d["ON"] - d["NO"] + d["NN"]}
    predicted = {"NN": case["new_token"], "OO": case["old_token"],
                 "ON": case["old_token"], "NO": case["new_token"]}
    mass_winner_prediction = all(cells[cell]["argmax_token"] == token and
                                 cells[cell]["top1_top2_gap"] > .001 for cell, token in predicted.items())
    payload_bytes = sum(path.stat().st_size for path in case_out.rglob("*") if path.is_file())
    require(payload_bytes <= 256 << 20, f"{name} intermediate payload cap exceeded")
    summary = {"status": "candidate", "case": name, "cells": cells, "contrasts": contrast,
               "mass_winner_prediction": mass_winner_prediction,
               "prefill_readback": literal_binding(case_out / "prefill-readback.json"),
               "case_payload_bytes_before_summary": payload_bytes}
    write(case_out / "result.json", summary)
    return {"result": literal_binding(case_out / "result.json"), "mass_winner_prediction": mass_winner_prediction,
            "cells": {cell: {"argmax_token": data["argmax_token"], "top1_top2_gap": data["top1_top2_gap"],
                             "old_minus_new_margin": data["old_minus_new_margin"]} for cell, data in cells.items()},
            "contrasts": contrast}


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda") and not OUT.exists(),
            "CUDA and unused output path required")
    require(json.loads(VAL_ACCEPTANCE.read_text())["status"] == "lead-accepted-crossing-and-intermediates" and
            json.loads(TRAIN_ACCEPTANCE.read_text())["status"] == "lead-accepted-directional-transfer-counterexample",
            "source contrasts lack lead acceptance")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device,
             "model_forwards": 0, "vision_forwards": 0, "started_unix": time.time(), "calls": [],
             "source": {"unit": literal_binding(UNIT), "val_acceptance": literal_binding(VAL_ACCEPTANCE),
                        "train_acceptance": literal_binding(TRAIN_ACCEPTANCE)}}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        cases = {name: prepare_case(name, q, identity, device) for name in ("val", "train")}
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer_capture = preserve_source(Path(__file__), run_root=OUT,
                                           relative_name=f"recurrence_attention_mass-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ("probes/training_set_completion/recurrence_key_phase.py",
                        "probes/training_set_completion/recurrence_phase_transfer.py",
                        "probes/training_set_completion/recurrence_position_history.py",
                        "probes/training_set_completion/recurrence_census/natural.py",
                        "probes/training_set_completion/recurrence_written_content.py",
                        "probes/training_set_completion/recurrence_history_cache_partition.py",
                        "probes/training_set_completion/recurrence_donor_tracking.py",
                        "probes/training_set_completion/untied_shared.py",
                        "probes/training_set_completion/numerical_feedback/runtime.py",
                        "src/qwen/native.py", "src/qwen/input_identity.py", "src/qwen/untied_embeddings.py",
                        "src/data/examples.py", "src/inference/bound_requests.py", "src/inference/inputs.py",
                        "src/config/inference.py", inspect.getfile(cache_utils), inspect.getfile(modeling_qwen3_vl),
                        inspect.getfile(sdpa_attention))]
        captures = [preserve_source(path, run_root=OUT, relative_name=str(path) if not path.is_absolute()
                                    else f"transformers/{path.name}") for path in dependencies]
        require(literal_binding(captures[11])["sha256"] == identity["loader_source"]["sha256"],
                "current untied loader source capture changed")
        manifest = {"schema": "recurrence_attention_mass.cells.v1", "status": "frozen_before_forward",
                    "source": state["source"], "producer": literal_binding(Path(__file__)),
                    "producer_capture": literal_binding(producer_capture),
                    "dependency_captures": [literal_binding(path) for path in captures],
                    "transformers_version": transformers.__version__, "model_identity": identity,
                    "cells_mass_profile": {cell: list(value) for cell, value in RULES.items()},
                    "predecessor_cell_mapping": {"NN": "NN", "OO": "NO"},
                    "cases": {name: {"target_batch": case["target"], "width": case["width"],
                                     "full_width": case["full"], "suffix": case["suffix"],
                                     "source": list(case["source"]), "destination": list(case["dest"]),
                                     "native_input_hashes": {key: tensor_hash(case["native"][key]) for key in
                                                             ("input_ids", "attention_mask", "position_ids")},
                                     "batch_identity": case["original_batch_identity"],
                                     "source_bindings": case["source_bindings"],
                                     "accepted_paths": case["accepted_paths"]} for name, case in cases.items()},
                    "model_forward_cap": MAX_FORWARDS, "wall_cap_seconds": MAX_SECONDS,
                    "diagonal_and_consumer_atol": ATOL, "case_payload_cap_bytes": 256 << 20}
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=literal_binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        clock = time.monotonic()
        def count(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_FORWARDS and time.monotonic() - clock <= MAX_SECONDS,
                    "model forward/time cap exceeded")
        def vision(*_):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= 2, "unexpected vision pass")
        handles = [q.model.register_forward_pre_hook(count, with_kwargs=True),
                   q.model.model.visual.register_forward_pre_hook(vision)]
        def forward(tag, inputs):
            state["current_forward"] = tag
            begin = time.monotonic()
            try:
                output = q.model(**inputs)
            except BaseException:
                state["last_forward_seconds"] = time.monotonic() - begin
                write(OUT / "receipt.json", state)
                raise
            state["calls"].append({"tag": tag, "model_forward": state["model_forwards"],
                                   "vision_forwards_so_far": state["vision_forwards"],
                                   "seconds": time.monotonic() - begin})
            state.pop("current_forward", None)
            write(OUT / "receipt.json", state)
            return output
        try:
            with torch.inference_mode():
                results = {name: run_case(name, case, q, forward, state, device) for name, case in cases.items()}
                require(state["model_forwards"] == 10 and state["vision_forwards"] == 2 and
                        time.monotonic() - clock <= MAX_SECONDS,
                        "frozen ten-call package incomplete")
                result = {"schema": "recurrence_attention_mass.result.v1", "status": "candidate",
                          "cases": results,
                          "both_cases_mass_winner_prediction": all(x["mass_winner_prediction"] for x in results.values()),
                          "source_manifest": literal_binding(OUT / "source-to-cell.json"),
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "model_seconds": time.monotonic() - clock}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"),
                             model_seconds=result["model_seconds"], elapsed_seconds=time.monotonic() - started,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "both_cases_mass_winner_prediction":
                                  result["both_cases_mass_winner_prediction"], "forwards": state["model_forwards"]}))
        finally:
            for h in handles:
                h.remove()
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
