"""Clamp current-row Q/K/V to native values in the accepted key-phase contrast."""
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
from transformers import DynamicCache, cache_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_donor_tracking import ROW, probability_readback, require, write
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest, check_cache
from probes.training_set_completion.recurrence_key_phase import (
    DEST, FULL_WIDTH, HEAD_DIM, KV_HEADS, LAYERS, Q_HEADS, SOURCE, SUFFIX, TARGET_OFFSET, WIDTH,
    patch_key, rotate, suffix_observers,
)
from probes.training_set_completion.recurrence_native_row_mass import block_hashes
from probes.training_set_completion.recurrence_position_history import PANEL, TARGET, source_and_rows
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-key-phase")
OUT = ROOT / "feedback-clamp-001"
PARENT = ROOT / "attempt-003"
ACCEPTANCE = ROOT / "lead-checks/phase-readback.json"
PROTOCOL = Path("research/experiments/2026-09-22-recurrence-key-phase/feedback-clamp.md")
MAX_FORWARDS, MAX_SECONDS, ATOL = 3, 600, 2e-4
UNCLAMPED_CONTRAST = 0.7541065216064453


def clamp_output(output, expected, *, target=TARGET):
    require(target == TARGET and output.shape[0] == 4 and output.shape[1:] == expected.shape,
            "Q/K/V clamp target or shape changed")
    changed = output.clone()
    changed[target].copy_(expected)
    require(torch.equal(changed[target], expected) and
            torch.equal(changed[[0, 1, 3]], output[[0, 1, 3]]), "Q/K/V clamp touched wrong batch")
    return changed


def check_appended(cache, index, expected_k, expected_v, cos, sin, *, width=WIDTH, full_width=FULL_WIDTH):
    require(cache.get_seq_length(index) == full_width and expected_k.shape == expected_v.shape ==
            (KV_HEADS, SUFFIX, HEAD_DIM) and cos.shape == sin.shape == (SUFFIX, HEAD_DIM),
            "actual appended S shape or length changed")
    actual_k = cache.layers[index].keys[TARGET, :, width:full_width, :]
    actual_v = cache.layers[index].values[TARGET, :, width:full_width, :]
    k_error = float((actual_k - rotate(expected_k, cos, sin)).abs().max())
    v_error = float((actual_v - expected_v).abs().max())
    require(k_error <= 2e-5 and v_error == 0, "clamped current-S K/V did not reach attention cache")
    companions = [0, 1, 3]
    return {"postK_max_abs": k_error, "V_max_abs": v_error,
            "actual_postK_hash": tensor_hash(actual_k), "actual_V_hash": tensor_hash(actual_v),
            "companion_postK_hash": tensor_hash(cache.layers[index].keys[companions, :, width:full_width, :]),
            "companion_V_hash": tensor_hash(cache.layers[index].values[companions, :, width:full_width, :])}


def clamp_hooks(model, reference, records):
    handles = []
    for index, layer in enumerate(model.model.language_model.layers):
        captured = reference[index]
        expected = {"Q": captured["q_norm_pre_rope"].transpose(0, 1),
                    "K": captured["current_S_k_norm_pre_rope"].transpose(0, 1),
                    "V": captured["current_S_value"].transpose(0, 1).reshape(SUFFIX, KV_HEADS * HEAD_DIM)}
        require(expected["Q"].shape == (SUFFIX, Q_HEADS, HEAD_DIM) and
                expected["K"].shape == (SUFFIX, KV_HEADS, HEAD_DIM) and
                expected["V"].shape == (SUFFIX, KV_HEADS * HEAD_DIM), "accepted NN current-S reference shape changed")
        records[index] = {"expected_hashes": {kind: tensor_hash(value) for kind, value in expected.items()}}
        for kind, module in (("Q", layer.self_attn.q_norm), ("K", layer.self_attn.k_norm),
                             ("V", layer.self_attn.v_proj)):
            def actuate(_module, _args, output, *, i=index, axis=kind, target_value=expected[kind]):
                require(axis not in records[i] and output.dtype == target_value.dtype and
                        output.device == target_value.device, "Q/K/V clamp repeated or dtype/device changed")
                records[i][axis] = {"before_companions": tensor_hash(output[[0, 1, 3]]),
                                    "before_target": tensor_hash(output[TARGET])}
                changed = clamp_output(output, target_value)
                records[i][axis]["after_companions"] = tensor_hash(changed[[0, 1, 3]])
                return changed
            def observe(_module, _args, output, *, i=index, axis=kind, target_value=expected[kind]):
                require(torch.equal(output[TARGET], target_value) and
                        tensor_hash(output[[0, 1, 3]]) == records[i][axis]["before_companions"],
                        "actual normalized Q/K or projected V did not consume clamp")
                records[i][axis]["consumed_target"] = tensor_hash(output[TARGET])
                records[i][axis]["consumed_companions"] = tensor_hash(output[[0, 1, 3]])
            handles.append(module.register_forward_hook(actuate))
            handles.append(module.register_forward_hook(observe))
    return handles


def selfcheck():
    q = torch.arange(4 * SUFFIX * Q_HEADS * HEAD_DIM, dtype=torch.float32).reshape(4, SUFFIX, Q_HEADS, HEAD_DIM)
    expected = torch.full_like(q[TARGET], -5)
    module = torch.nn.Identity()
    seen = []
    def act(_module, _args, output):
        return clamp_output(output, expected)
    def observe(_module, _args, output):
        seen.append(output.clone())
    handles = (module.register_forward_hook(act), module.register_forward_hook(observe))
    try:
        actual = module(q)
    finally:
        for handle in handles:
            handle.remove()
    require(len(seen) == 1 and torch.equal(actual[TARGET], expected) and
            torch.equal(seen[0], actual) and torch.equal(actual[[0, 1, 3]], q[[0, 1, 3]]),
            "CPU actual consumer did not see target-only clamp")
    for bad, target in ((expected[:, :-1], TARGET), (expected, 1)):
        try:
            clamp_output(q, bad, target=target)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong S length or target batch admitted")
    cache = DynamicCache(((torch.zeros(4, KV_HEADS, 18, HEAD_DIM), torch.zeros(4, KV_HEADS, 18, HEAD_DIM)),))
    k, v = torch.ones(KV_HEADS, SUFFIX, HEAD_DIM), torch.full((KV_HEADS, SUFFIX, HEAD_DIM), 2.0)
    cos, sin = torch.ones(SUFFIX, HEAD_DIM), torch.zeros(SUFFIX, HEAD_DIM)
    cache.layers[0].keys[TARGET, :, 12:18] = k
    cache.layers[0].values[TARGET, :, 12:18] = v
    check_appended(cache, 0, k, v, cos, sin, width=12, full_width=18)
    for axis in ("keys", "values"):
        value = getattr(cache.layers[0], axis)
        value[TARGET, 0, 12, 0] += 1
        try:
            check_appended(cache, 0, k, v, cos, sin, width=12, full_width=18)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong appended S K/V escaped readback")
        value[TARGET, 0, 12, 0] -= 1


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda") and not OUT.exists(), "CUDA/unused output required")
    require(json.loads(ACCEPTANCE.read_text())["status"] == "lead-accepted-crossing-and-intermediates",
            "crossing/intermediate acceptance missing")
    parent_result_path, parent_manifest_path = PARENT / "result.json", PARENT / "source-to-cell.json"
    parent_result = json.loads(parent_result_path.read_text())
    parent_manifest = json.loads(parent_manifest_path.read_text())
    require(parent_result["status"] == "candidate", "accepted parent candidate changed")
    ref_trace_path, ref_logits_path = PARENT / "NN-intermediates.pt", PARENT / "NN.pt"
    native_packet_path = PARENT / "phase-and-prekeys.pt"
    require(literal_binding(ref_trace_path) == parent_result["cells"]["NN"]["intermediates"] and
            literal_binding(ref_logits_path) == parent_result["cells"]["NN"]["full_logits"] and
            literal_binding(native_packet_path) == parent_result["phase_and_prekeys"],
            "accepted NN reference changed")
    reference = torch.load(ref_trace_path, map_location="cpu", weights_only=True)
    require(len(reference["layers"]) == LAYERS and reference["actual_current_phase"]["cos"].shape ==
            reference["actual_current_phase"]["sin"].shape == (SUFFIX, HEAD_DIM), "accepted NN trace incomplete")
    chosen, raw, trace, receipt, panel, group, S, offsets, _ = source_and_rows()
    tokens = raw[TARGET]["token_ids"]
    require(offsets[1] == TARGET_OFFSET and tokens[783:792] == tokens[792:801] == ROW and tokens[801:807] == S,
            "native source/destination/current S changed")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "started_unix": time.time(), "source": {"protocol": literal_binding(PROTOCOL),
             "parent_acceptance": literal_binding(ACCEPTANCE), "parent_result": literal_binding(parent_result_path),
             "parent_manifest": literal_binding(parent_manifest_path), "NN_trace": literal_binding(ref_trace_path),
             "NN_vector": literal_binding(ref_logits_path), "native_key_packet": literal_binding(native_packet_path),
             "selection": parent_manifest["source"]["selection"],
             **{key: chosen[key] for key in ("raw", "trace", "runtime_receipt", "image")}}}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        current, saved = dict(identity), dict(receipt["identity"])
        current_loader, saved_loader = current.pop("loader_source"), saved.pop("loader_source")
        require(current == saved and (current_loader["sha256"], current_loader["size_bytes"]) ==
                (saved_loader["sha256"], saved_loader["size_bytes"]), "source/effective model identity changed")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"], "native batch/image identity changed")
        tails = _prefix_tokens(raw, TARGET_OFFSET, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        require(native["input_ids"].shape == (4, FULL_WIDTH) and native["position_ids"].shape == (3, 4, FULL_WIDTH) and
                native["input_ids"][TARGET, SOURCE[0]:SOURCE[1]].tolist() == ROW and
                native["input_ids"][TARGET, DEST[0]:DEST[1]].tolist() == ROW and
                native["input_ids"][TARGET, WIDTH:FULL_WIDTH].tolist() == S and
                all(tensor_hash(native[k]) == parent_manifest["native_input_hashes"][k] for k in
                    ("input_ids", "attention_mask", "position_ids")), "accepted native replay inputs changed")
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_key_phase_feedback-{producer_hash[:12]}.py")
        dependency_paths = [Path(p) for p in ("probes/training_set_completion/recurrence_key_phase.py",
                            "probes/training_set_completion/recurrence_native_row_mass.py",
                            "probes/training_set_completion/recurrence_history_cache_partition.py",
                            "probes/training_set_completion/recurrence_donor_tracking.py",
                            "probes/training_set_completion/recurrence_position_history.py",
                            "probes/training_set_completion/recurrence_written_content.py",
                            "probes/training_set_completion/untied_shared.py",
                            "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py",
                            "src/inference/bound_requests.py", "src/qwen/untied_embeddings.py",
                            inspect.getfile(cache_utils), inspect.getfile(modeling_qwen3_vl), inspect.getfile(sdpa_attention))]
        captures = [preserve_source(p, run_root=OUT, relative_name=str(p) if not p.is_absolute() else
                                    f"transformers/{p.name}") for p in dependency_paths]
        require(literal_binding(captures[10])["sha256"] == current_loader["sha256"], "loader source capture changed")
        manifest = {"schema": "recurrence_key_phase_feedback.cells.v1", "status": "frozen_before_forward",
                    "source": state["source"], "producer": literal_binding(Path(__file__)),
                    "producer_capture": literal_binding(capture),
                    "dependency_captures": [literal_binding(p) for p in captures],
                    "transformers_version": transformers.__version__, "model_identity": identity,
                    "native_batch_identity": input_identity(batch), "source_panel": literal_binding(PANEL),
                    "target_batch": TARGET, "target_offset": TARGET_OFFSET, "current_S": S,
                    "prefill_width": WIDTH, "source_block": list(SOURCE), "destination_block": list(DEST),
                    "native_input_hashes": {k: tensor_hash(native[k]) for k in ("input_ids", "attention_mask", "position_ids")},
                    "suffix_hashes": {k: tensor_hash(v) for k, v in
                                      {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH],
                                       "attention_mask": native["attention_mask"],
                                       "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH],
                                       "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device)}.items()},
                    "conditions": {"NN": "native historical K plus accepted NN current-S Q/K/V",
                                   "OO": "old-row post-RoPE K at destination plus accepted NN current-S Q/K/V"},
                    "max_forwards": MAX_FORWARDS, "max_seconds": MAX_SECONDS,
                    "NN_fullvector_atol": ATOL, "current_postK_atol": 2e-5,
                    "unclamped_contrast": UNCLAMPED_CONTRAST}
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
            require(state["vision_forwards"] <= 1, "unexpected current-S vision pass")
        counters = [q.model.register_forward_pre_hook(count, with_kwargs=True),
                    q.model.model.visual.register_forward_pre_hook(vision)]
        try:
            with torch.inference_mode():
                cache = DynamicCache()
                prefill = dict(native)
                prefill.update(input_ids=native["input_ids"][:, :WIDTH], attention_mask=native["attention_mask"][:, :WIDTH],
                               position_ids=native["position_ids"][:, :, :WIDTH],
                               cache_position=torch.arange(WIDTH, device=device), past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
                consumed, standard = hooks(q.model, prefill["input_ids"], prefill["position_ids"])
                try:
                    output = q.model(**prefill)
                finally:
                    for handle in standard:
                        handle.remove()
                require(output.past_key_values is cache and state["vision_forwards"] == 1 and
                        len(consumed["embedding_inputs"]) == len(consumed["rotary_positions"]) == 1 and
                        len(consumed["masks"]) == len(consumed["cache_slots"]) == 2,
                        "native prefill consumption invalid")
                check_cache(cache, width=WIDTH)
                original_digest = cache_digest(cache)
                original_blocks = block_hashes(cache)
                phase_packet = torch.load(native_packet_path, map_location="cpu", weights_only=True)
                require(len(phase_packet["layers"]) == LAYERS, "accepted native key packet changed")
                old_keys, new_keys = [], []
                for i, layer in enumerate(cache.layers):
                    packet = phase_packet["layers"][i]
                    require(torch.equal(layer.keys[TARGET, :, SOURCE[0]:SOURCE[1], :].cpu(), packet["post_old"]) and
                            torch.equal(layer.keys[TARGET, :, DEST[0]:DEST[1], :].cpu(), packet["post_new"]) and
                            torch.equal(layer.values[TARGET, :, DEST[0]:DEST[1], :].cpu(), packet["native_V_dest"]),
                            f"native historical K/V layer {i} differs from accepted packet")
                    old_keys.append(layer.keys[TARGET, :, SOURCE[0]:SOURCE[1], :].clone())
                    new_keys.append(layer.keys[TARGET, :, DEST[0]:DEST[1], :].clone())
                write(OUT / "prefill-readback.json", {"consumed": consumed, "cache_digest": original_digest,
                                                       "block_hashes": original_blocks,
                                                       "accepted_native_packet": literal_binding(native_packet_path)})
                state["prefill_readback"] = literal_binding(OUT / "prefill-readback.json")
                write(OUT / "receipt.json", state)
                reference_layers = [{k: v.to(device) for k, v in record.items() if k in
                                    ("q_norm_pre_rope", "current_S_k_norm_pre_rope", "current_S_value")}
                                    for record in reference["layers"]]
                expected_phase = {k: value.to(device) for k, value in reference["actual_current_phase"].items()}
                cells = {}
                for name, candidate in (("NN", new_keys), ("OO", old_keys)):
                    with patch_key(cache, candidate, original_digest):
                        expected_blocks = block_hashes(cache)
                        for i in range(LAYERS):
                            require(expected_blocks[i]["source"] == original_blocks[i]["source"] and
                                    expected_blocks[i]["dest"]["values"] == original_blocks[i]["dest"]["values"] and
                                    expected_blocks[i]["dest"]["keys"] == tensor_hash(candidate[i]),
                                    "historical source/native-V/declared-K changed")
                        suffix = {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH],
                                  "attention_mask": native["attention_mask"],
                                  "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH],
                                  "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device),
                                  "past_key_values": cache, "use_cache": True, "return_dict": True,
                                  "logits_to_keep": SUFFIX}
                        actuation = [dict() for _ in range(LAYERS)]
                        actuators = clamp_hooks(q.model, reference_layers, actuation)
                        seen, standard = hooks(q.model, suffix["input_ids"], suffix["position_ids"])
                        trace, rope, observers = suffix_observers(q.model, cache, expected_blocks)
                        appended, after = {}, []
                        for i, layer in enumerate(q.model.model.language_model.layers):
                            def check(_module, _args, *, index=i):
                                require(len(rope) == 1 and torch.equal(rope[0][0], expected_phase["cos"]) and
                                        torch.equal(rope[0][1], expected_phase["sin"]),
                                        "actual current-S rotary phase changed")
                                ref = reference_layers[index]
                                appended[index] = check_appended(cache, index, ref["current_S_k_norm_pre_rope"],
                                                                 ref["current_S_value"], *rope[0])
                            after.append(layer.self_attn.o_proj.register_forward_pre_hook(check))
                        try:
                            output = q.model(**suffix)
                            logits = output.logits[TARGET, -1].detach().float().cpu()
                            companion_logits = tensor_hash(output.logits[[0, 1, 3], -1].detach())
                        finally:
                            for handle in actuators + standard + observers + after:
                                handle.remove()
                        require(torch.isfinite(logits).all() and cache.get_seq_length() == FULL_WIDTH and
                                state["vision_forwards"] == 1 and len(rope) == 1 and
                                len(trace) == len(actuation) == len(appended) == LAYERS,
                                "clamped suffix observation incomplete")
                        require(len(seen["embedding_inputs"]) == len(seen["rotary_positions"]) == 1 and
                                len(seen["masks"]) == len(seen["cache_slots"]) == 2,
                                "current-S token/position/mask attestation invalid")
                        require(all(trace[i]["cache_length_before"] == WIDTH and
                                    trace[i]["dest_hashes"] == expected_blocks[i]["dest"] and
                                    trace[i]["source_hashes"] == expected_blocks[i]["source"] and
                                    torch.equal(trace[i]["q_norm_pre_rope"], reference["layers"][i]["q_norm_pre_rope"])
                                    for i in range(LAYERS)), "actual Q/history/cache consumption changed")
                        require(all(max(trace[i]["attention_reconstruction_max_abs"],
                                        appended[i]["postK_max_abs"]) <= ATOL for i in range(LAYERS)),
                                "Q/K/V attention readback failed")
                        current = {"consumed": seen, "actuation": actuation, "appended": appended,
                                   "mask_hashes": [trace[i]["mask_hash"] for i in range(LAYERS)],
                                   "attention_reconstruction_max_abs":
                                   [trace[i]["attention_reconstruction_max_abs"] for i in range(LAYERS)],
                                   "companion_final_logits_hash": companion_logits,
                                   "cache_length_after": cache.get_seq_length()}
                        if name == "OO":
                            baseline = cells["NN"]["readback"]
                            require(seen == baseline["consumed"] and
                                    current["mask_hashes"] == baseline["mask_hashes"] and
                                    companion_logits == baseline["companion_final_logits_hash"] and
                                    all(actuation[i][kind]["consumed_companions"] ==
                                        baseline["actuation"][i][kind]["consumed_companions"] for i in range(LAYERS)
                                        for kind in ("Q", "K", "V")) and
                                    all(appended[i][kind] == baseline["appended"][i][kind]
                                        for i in range(LAYERS) for kind in ("companion_postK_hash", "companion_V_hash")),
                                    "OO changed current-S positions/mask/companions")
                        readback_path = OUT / f"{name}-readback.json"
                        write(readback_path, current)
                        vector_path = OUT / f"{name}.pt"
                        torch.save(logits, vector_path)
                        top = torch.topk(logits, 5)
                        cells[name] = {"full_logits": literal_binding(vector_path),
                                       "readback_binding": literal_binding(readback_path), "readback": current,
                                       "argmax_token": int(top.indices[0]),
                                       "top1_top2_gap": float(top.values[0] - top.values[1]),
                                       "top5": [{"token": int(t), "logit": float(v)} for t, v in
                                                zip(top.indices, top.values, strict=True)],
                                       "z38_minus_z999": float(logits[151708] - logits[152669]),
                                       "absolute_coordinates": probability_readback(logits, (38, 999))}
                        state["last_cell"] = name
                        write(OUT / "partial-results.json", {"status": "running", "cells": cells})
                        write(OUT / "receipt.json", state)
                    if name == "NN":
                        accepted = torch.load(ref_logits_path, map_location="cpu", weights_only=True)
                        cells[name]["fullvector_max_abs_vs_accepted_NN"] = float((logits - accepted).abs().max())
                        require(cells[name]["fullvector_max_abs_vs_accepted_NN"] <= ATOL and
                                cells[name]["argmax_token"] == 152669,
                                "clamped NN baseline full-vector qualification failed")
                require(state["model_forwards"] == MAX_FORWARDS and state["vision_forwards"] == 1 and
                        time.monotonic() - clock <= MAX_SECONDS, "frozen forward/time budget changed")
                d = {name: value["z38_minus_z999"] for name, value in cells.items()}
                ratio = (d["OO"] - d["NN"]) / UNCLAMPED_CONTRAST
                band = ("substantial retention" if 0.8 <= ratio <= 1.2 else
                        "substantial collapse" if abs(ratio) <= 0.2 else
                        "reversal" if ratio < -0.2 else "partial retention or enhancement")
                result = {"schema": "recurrence_key_phase_feedback.result.v1", "status": "candidate",
                          "cells": cells, "clamped_contrast": d["OO"] - d["NN"],
                          "unclamped_contrast": UNCLAMPED_CONTRAST, "ratio": ratio, "descriptive_band": band,
                          "source_manifest": literal_binding(OUT / "source-to-cell.json"),
                          "prefill_readback": literal_binding(OUT / "prefill-readback.json"),
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "model_seconds": time.monotonic() - clock}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"),
                             model_seconds=result["model_seconds"], elapsed_seconds=time.monotonic() - started,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "ratio": ratio, "forwards": state["model_forwards"]}))
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
