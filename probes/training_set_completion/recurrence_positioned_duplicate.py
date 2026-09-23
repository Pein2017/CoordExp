"""Test preceding-row contextual K/V at the new row's native cache position."""
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
from transformers.models.qwen3_vl import modeling_qwen3_vl

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.recurrence_attention_mass import (
    CASES, LAYERS, KV_HEADS, HEAD_DIM, prepare_case, same_binding,
    fixed_history_snapshot, fixed_history_equal,
)
from probes.training_set_completion.recurrence_donor_tracking import probability_readback, require, write
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest, check_cache
from probes.training_set_completion.recurrence_key_phase import rotate
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
OUT = BASE / "2026-09-22-recurrence-positioned-duplicate/attempt-001"
UNIT = Path("research/experiments/2026-09-22-recurrence-positioned-duplicate/unit.md")
MASS_ACCEPTANCE = BASE / "2026-09-22-recurrence-attention-mass/lead-acceptance.json"
MASS_RESULT = BASE / "2026-09-22-recurrence-attention-mass/attempt-001/result.json"
VAL_CONTROLS = BASE / "2026-09-22-recurrence-history-location/native-row-mass-002/result.json"
ATOL, MAX_FORWARDS, MAX_SECONDS = 2e-4, 8, 900


def prior_nn_observer(case, name):
    """Recover source hashes and S phase from an accepted native NN observer."""
    bound = case["accepted_paths"]["prior_NN_mask_readback"]
    path = Path(bound["path"])
    require(same_binding(path, bound), "accepted native observer changed")
    if name == "val":
        trace = torch.load(path, map_location="cpu", weights_only=True)
        rows = [{"K": x["source_hashes"]["keys"], "V": x["source_hashes"]["values"]}
                for x in trace["layers"]]
        current_phase = {k: v.clone() for k, v in trace["actual_current_phase"].items()}
    else:
        readback = json.loads(path.read_text())
        rows = [{"K": readback["layers"][str(i)]["blocks"]["source"]["keys"],
                 "V": readback["layers"][str(i)]["blocks"]["source"]["values"]}
                for i in range(LAYERS)]
        current_phase = list(readback["actual_S_phase_hashes"])
    require(len(rows) == LAYERS, "accepted source K/V observer incomplete")
    return rows, current_phase


def qualify_blocks(key, value, expected_key, expected_value, *, dest, width):
    require(dest == (width - 9, width) and
            key.shape == value.shape == expected_key.shape == expected_value.shape ==
            (KV_HEADS, 9, HEAD_DIM), "wrong destination or K/V block shape")
    require(torch.equal(key, expected_key) and torch.equal(value, expected_value),
            "wrong final phase or preceding-row V at destination")


@contextmanager
def patch_pair(cache, candidate_keys, source_values, case, native_digest):
    target, dest, width = (case[k] for k in ("target", "dest", "width"))
    require(dest == (width - 9, width) and len(cache.layers) == len(candidate_keys) == len(source_values) == LAYERS,
            "wrong destination or layer count")
    saved = []
    try:
        for layer, key, value in zip(cache.layers, candidate_keys, source_values, strict=True):
            old_keys, old_values = layer.keys, layer.values
            saved.append((layer, old_keys, old_values,
                          old_keys[target, :, dest[0]:dest[1], :].clone(),
                          old_values[target, :, dest[0]:dest[1], :].clone()))
            require(key.shape == value.shape == (KV_HEADS, 9, HEAD_DIM), "wrong destination K/V shape")
            layer.keys[target, :, dest[0]:dest[1], :].copy_(key)
            layer.values[target, :, dest[0]:dest[1], :].copy_(value)
        yield
    finally:
        for layer, old_keys, old_values, old_key_block, old_value_block in saved:
            layer.keys = old_keys
            layer.values = old_values
            layer.keys[target, :, dest[0]:dest[1], :].copy_(old_key_block)
            layer.values[target, :, dest[0]:dest[1], :].copy_(old_value_block)
        require(cache_digest(cache) == native_digest and cache.get_seq_length() == width,
                "native cache reference/block restoration failed")


def selfcheck():
    """Falsify a wrong phase, wrong V slot, companion mutation and S leakage."""
    case = {"target": 2, "source": (0, 9), "dest": (9, 18), "width": 18, "full": 20, "suffix": 2}
    native_k = torch.zeros(4, KV_HEADS, 18, HEAD_DIM)
    native_v = torch.zeros_like(native_k)
    native_v[2, :, 0:9] = 2
    native_v[2, :, 9:18] = 5
    cache = DynamicCache(tuple((native_k.clone(), native_v.clone()) for _ in range(LAYERS)))
    digest = cache_digest(cache)
    old_pre = torch.ones(KV_HEADS, 9, HEAD_DIM)
    cos, sin = torch.ones(9, HEAD_DIM), torch.zeros(9, HEAD_DIM)
    candidate = rotate(old_pre, cos, sin)
    value = cache.layers[0].values[2, :, 0:9].clone()
    qualify_blocks(candidate, value, candidate, value, dest=(9, 18), width=18)
    for wrong_key, wrong_value, wrong_dest in ((candidate + 1, value, (9, 18)),
                                                (candidate, cache.layers[0].values[2, :, 9:18], (9, 18)),
                                                (candidate, value, (8, 17))):
        try:
            qualify_blocks(wrong_key, wrong_value, candidate, value, dest=wrong_dest, width=18)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong phase, V or destination admitted")
    with torch.inference_mode():
        with patch_pair(cache, [candidate] * LAYERS, [value] * LAYERS,
                        case, digest):
            require(torch.equal(cache.layers[0].keys[2, :, 9:18], candidate) and
                    torch.equal(cache.layers[0].values[2, :, 9:18], value) and
                    torch.equal(cache.layers[0].keys[[0, 1, 3]], native_k[[0, 1, 3]]),
                    "target-only K/V patch failed")
            cache.update(torch.zeros(4, KV_HEADS, 2, HEAD_DIM),
                         torch.zeros(4, KV_HEADS, 2, HEAD_DIM), 0)
        require(torch.equal(cache.layers[0].keys, native_k) and
                torch.equal(cache.layers[0].values, native_v), "S append leaked through restoration")
        try:
            with patch_pair(cache, [candidate] * LAYERS, [value] * LAYERS,
                            case, digest):
                raise RuntimeError("injected consumer failure")
        except RuntimeError:
            pass
        require(cache_digest(cache) == digest, "exception leaked K/V mutation")


def run_case(name, case, q, forward, state, device):
    target, source, dest, width, full, suffix = (case[k] for k in
                                                ("target", "source", "dest", "width", "full", "suffix"))
    case_out = OUT / name
    case_out.mkdir()
    native = case["native"]
    cache = DynamicCache()
    prefill = dict(native)
    prefill.update(input_ids=native["input_ids"][:, :width],
                   attention_mask=native["attention_mask"][:, :width],
                   position_ids=native["position_ids"][:, :, :width],
                   cache_position=torch.arange(width, device=device),
                   past_key_values=cache, use_cache=True, logits_to_keep=1)
    observed_phase, pre_norm, prefill_handles = [], [None] * LAYERS, []
    def rotary(_module, _args, output):
        cos, sin = output
        require(cos.shape == sin.shape == (4, width, HEAD_DIM), "historical phase shape changed")
        observed_phase.append({"old_cos": cos[target, source[0]:source[1]].detach().clone(),
                               "old_sin": sin[target, source[0]:source[1]].detach().clone(),
                               "new_cos": cos[target, dest[0]:dest[1]].detach().clone(),
                               "new_sin": sin[target, dest[0]:dest[1]].detach().clone()})
    prefill_handles.append(q.model.model.language_model.rotary_emb.register_forward_hook(rotary))
    for i, layer in enumerate(q.model.model.language_model.layers):
        def normalized(_module, _args, output, *, index=i):
            require(output.shape == (4, width, KV_HEADS, HEAD_DIM), "historical pre-K shape changed")
            pre_norm[index] = {"old": output[target, source[0]:source[1]].transpose(0, 1).detach().clone(),
                               "new": output[target, dest[0]:dest[1]].transpose(0, 1).detach().clone()}
        prefill_handles.append(layer.self_attn.k_norm.register_forward_hook(normalized))
    prefill_seen, standard = hooks(q.model, prefill["input_ids"], prefill["position_ids"])
    prefill_handles.extend(standard)
    try:
        output = forward(f"{name}:historical_prefill", prefill)
    finally:
        for handle in prefill_handles:
            handle.remove()
    require(output.past_key_values is cache and len(observed_phase) == 1 and
            all(x is not None for x in pre_norm), "historical cache/phase observation incomplete")
    del output
    check_cache(cache, width=width)
    packet, phase = case["packet"], observed_phase[0]
    phase_errors = {k: float((phase[k].cpu() - packet["actual_phase"][k]).abs().max()) for k in phase}
    accepted_source_hashes, accepted_current_phase = prior_nn_observer(case, name)
    blocks, prefill_metrics = [], []
    for i, layer in enumerate(cache.layers):
        old_pre, new_pre = pre_norm[i]["old"], pre_norm[i]["new"]
        candidate = packet["layers"][i]["candidate_keys"]["ON"].to(device)
        source_key = layer.keys[target, :, source[0]:source[1], :]
        source_v = layer.values[target, :, source[0]:source[1], :]
        native_key = layer.keys[target, :, dest[0]:dest[1], :]
        native_v = layer.values[target, :, dest[0]:dest[1], :]
        key_expected = rotate(old_pre, phase["new_cos"], phase["new_sin"])
        errors = {"old_preK": float((old_pre.cpu() - packet["layers"][i]["pre_old"]).abs().max()),
                  "new_preK": float((new_pre.cpu() - packet["layers"][i]["pre_new"]).abs().max()),
                  "native_postK": float((native_key.cpu() - packet["layers"][i]["post_new"]).abs().max()),
                  "native_V": float((native_v.cpu() - packet["layers"][i]["native_V_dest"]).abs().max()),
                  "ON_key_vs_live_rotation": float((candidate - key_expected).abs().max())}
        source_hashes = {"K": tensor_hash(source_key), "V": tensor_hash(source_v)}
        prefill_metrics.append({"layer": i, "errors": errors, "source_hashes": source_hashes,
                                "source_matches_accepted": source_hashes == accepted_source_hashes[i],
                                "ON_key_hash": tensor_hash(candidate),
                                "native_destination_K_hash": tensor_hash(native_key),
                                "native_destination_V_hash": tensor_hash(native_v)})
        blocks.append({"source_K": source_key.detach().clone(), "source_V": source_v.detach().clone(),
                       "native_K": native_key.detach().clone(), "native_V": native_v.detach().clone(),
                       "D_K": candidate.detach().clone()})
    digest = cache_digest(cache)
    fixed = fixed_history_snapshot(cache, target, source[0])
    torch.save({"phase": {k: v.cpu() for k, v in phase.items()},
                "layers": [{k: v.cpu() for k, v in x.items()} for x in blocks]},
               case_out / "source-and-destination.pt")
    write(case_out / "prefill-readback.json", {"consumed": prefill_seen,
         "accepted_phase_packet": case["accepted_paths"]["phase_packet"],
         "accepted_source_observer": case["accepted_paths"]["prior_NN_mask_readback"],
         "live_phase_errors": phase_errors, "per_layer": prefill_metrics,
         "source_and_destination": literal_binding(case_out / "source-and-destination.pt"),
         "cache_digest": digest, "cache_length": cache.get_seq_length()})
    state["last_prefill"] = name
    write(OUT / "receipt.json", state)
    require(max(phase_errors.values()) == 0 and
            all(x["source_matches_accepted"] and all(value <= ATOL for value in x["errors"].values())
                for x in prefill_metrics), f"{name} live old/new pre-K, phase or source V differs from accepted")

    cells = {}
    for cell in ("NN", "D"):
        expected = [{"K": x["native_K"], "V": x["native_V"]} if cell == "NN" else
                    {"K": x["D_K"], "V": x["source_V"]} for x in blocks]
        with patch_pair(cache, [x["K"] for x in expected], [x["V"] for x in expected], case, digest):
            suffix_inputs = {"input_ids": native["input_ids"][:, width:full],
                             "attention_mask": native["attention_mask"],
                             "position_ids": native["position_ids"][:, :, width:full],
                             "cache_position": torch.arange(width, full, device=device),
                             "past_key_values": cache, "use_cache": True,
                             "return_dict": True, "logits_to_keep": suffix}
            suffix_seen, suffix_handles = hooks(q.model, suffix_inputs["input_ids"], suffix_inputs["position_ids"])
            layer_seen, observers = [None] * LAYERS, []
            for i, layer in enumerate(q.model.model.language_model.layers):
                def before(_module, _args, kwargs, *, index=i):
                    mask, slots, phase_now = (kwargs.get(k) for k in
                                              ("attention_mask", "cache_position", "position_embeddings"))
                    require(kwargs.get("past_key_values") is cache and
                            cache.get_seq_length(index) == width and
                            isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and
                            mask.shape == (4, 1, suffix, full) and
                            isinstance(slots, torch.Tensor) and
                            torch.equal(slots, torch.arange(width, full, device=slots.device)) and
                            isinstance(phase_now, tuple) and len(phase_now) == 2 and
                            phase_now[0].shape == phase_now[1].shape == (4, suffix, HEAD_DIM),
                            "actual S mask, slots, phase or cache changed")
                    phase_matches_prior = (all(torch.equal(phase_now[j][target].cpu(),
                                              accepted_current_phase[k]) for j, k in enumerate(("cos", "sin")))
                                           if name == "val" else
                                           [tensor_hash(x) for x in phase_now] == accepted_current_phase)
                    require(phase_matches_prior, "actual S phase differs from accepted native NN")
                    current = cache.layers[index]
                    key = current.keys[target, :, dest[0]:dest[1], :]
                    value = current.values[target, :, dest[0]:dest[1], :]
                    qualify_blocks(key, value, expected[index]["K"], expected[index]["V"],
                                   dest=dest, width=width)
                    actual_source = {"K": tensor_hash(current.keys[target, :, source[0]:source[1], :]),
                                     "V": tensor_hash(current.values[target, :, source[0]:source[1], :])}
                    require(torch.equal(current.keys[target, :, source[0]:source[1], :], blocks[index]["source_K"]) and
                            torch.equal(current.values[target, :, source[0]:source[1], :], blocks[index]["source_V"]) and
                            fixed_history_equal(current, fixed[index], target, source[0]) and
                            actual_source == accepted_source_hashes[index],
                            "actual source row, prior history or companions changed")
                    layer_seen[index] = {"layer": index, "prefix_length": cache.get_seq_length(index),
                                         "mask_hash": tensor_hash(mask), "slots_hash": tensor_hash(slots),
                                         "phase_hashes": [tensor_hash(x) for x in phase_now],
                                         "phase_matches_accepted_NN": phase_matches_prior,
                                         "source_hashes": actual_source,
                                         "destination_hashes": {"K": tensor_hash(key), "V": tensor_hash(value)},
                                         "fixed_history_and_companions_exact": True}
                observers.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
            try:
                output = forward(f"{name}:{cell}", suffix_inputs)
                logits = output.logits[target, -1].detach().float().cpu()
                companion_hash = tensor_hash(output.logits[[i for i in range(4) if i != target], -1].detach())
            finally:
                for handle in suffix_handles + observers:
                    handle.remove()
            require(torch.isfinite(logits).all() and cache.get_seq_length() == full and
                    all(x is not None for x in layer_seen), f"{name}:{cell} S observation incomplete")
            vector_path = case_out / f"{cell}.pt"
            torch.save(logits, vector_path)
            readback = {"consumed": suffix_seen, "layers": layer_seen,
                        "companion_final_logits_hash": companion_hash,
                        "cache_length_after": cache.get_seq_length()}
            readback_path = case_out / f"{cell}-readback.json"
            write(readback_path, readback)
            top = torch.topk(logits, 5)
            requested = (38, 999) if name == "val" else (348, 350, 591)
            cells[cell] = {"full_logits": literal_binding(vector_path),
                           "readback": literal_binding(readback_path),
                           "argmax_token": int(top.indices[0]),
                           "top1_top2_gap": float(top.values[0] - top.values[1]),
                           "top5": [{"token": int(t), "logit": float(v)} for t, v in
                                    zip(top.indices, top.values, strict=True)],
                           "absolute_coordinates": probability_readback(logits, requested),
                           "margins": ({"z38_minus_z999": float(logits[151708] - logits[152669])}
                                       if name == "val" else
                                       {"z350_minus_z348": float(logits[152020] - logits[152018]),
                                        "z591_minus_z348": float(logits[152261] - logits[152018])})}
            state["last_cell"] = f"{name}:{cell}"
            write(OUT / "receipt.json", state)
            require(all(x["mask_hash"] == case["prior_mask_hashes"][i] and
                        x["prefix_length"] == width and x["fixed_history_and_companions_exact"]
                        for i, x in enumerate(layer_seen)) and
                    len(suffix_seen["embedding_inputs"]) == len(suffix_seen["rotary_positions"]) == 1 and
                    len(suffix_seen["masks"]) == len(suffix_seen["cache_slots"]) == 2,
                    f"{name}:{cell} actual native S/mask/cache consumer qualification failed")
            if cell == "D":
                nn = json.loads((case_out / "NN-readback.json").read_text())
                require(suffix_seen == nn["consumed"] and companion_hash == nn["companion_final_logits_hash"] and
                        all(x["mask_hash"] == y["mask_hash"] and x["slots_hash"] == y["slots_hash"] and
                            x["phase_hashes"] == y["phase_hashes"]
                            for x, y in zip(layer_seen, nn["layers"], strict=True)),
                        f"{name}:D changed current S, positions, mask or companions")
        prior = case["accepted_result"]
        ref_cell = "NN" if cell == "NN" else "ON"
        ref_path = case["root"] / f"{ref_cell}.pt"
        require(same_binding(ref_path, prior["cells"][ref_cell]["full_logits"]),
                f"{name} predecessor reference changed")
        reference = torch.load(ref_path, map_location="cpu", weights_only=True)
        cells[cell]["vs_predecessor_" + ref_cell] = {
            "reference": literal_binding(ref_path),
            "fullvector_max_abs": float((logits - reference).abs().max()),
            "fullvector_l2": float(torch.linalg.vector_norm(logits.double() - reference.double()))}
        if cell == "D":
            native_reference = torch.load(case["root"] / "NN.pt", map_location="cpu", weights_only=True)
            cells[cell]["vs_predecessor_NN"] = {
                "reference": case["accepted_paths"]["NN_vector"],
                "fullvector_max_abs": float((logits - native_reference).abs().max()),
                "fullvector_l2": float(torch.linalg.vector_norm(logits.double() - native_reference.double()))}
        if cell == "NN":
            require(cells[cell]["vs_predecessor_NN"]["fullvector_max_abs"] <= ATOL and
                    cells[cell]["argmax_token"] == prior["cells"]["NN"]["argmax_token"],
                    f"{name} fresh NN fullvector parity failed")
    payload = sum(x.stat().st_size for x in case_out.rglob("*") if x.is_file())
    require(payload <= 128 << 20, f"{name} intermediate payload cap exceeded")
    success = cells["D"]["argmax_token"] == case["new_token"] and cells["D"]["top1_top2_gap"] > .001
    result = {"status": "candidate", "case": name, "cells": cells,
              "native_winner_retained": success,
              "prefill_readback": literal_binding(case_out / "prefill-readback.json"),
              "payload_bytes_before_summary": payload}
    write(case_out / "result.json", result)
    return {"result": literal_binding(case_out / "result.json"),
            "native_winner_retained": success,
            "NN_winner": cells["NN"]["argmax_token"], "D_winner": cells["D"]["argmax_token"],
            "D_gap": cells["D"]["top1_top2_gap"],
            "D_margins": cells["D"]["margins"]}


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda") and not OUT.exists(),
            "CUDA and unused attempt path required")
    closure = json.loads(MASS_ACCEPTANCE.read_text())
    require(closure["status"] == "lead-accepted" and
            same_binding(MASS_RESULT, next(x for x in closure["artifacts"] if x["path"] == str(MASS_RESULT))),
            "preceding mass/profile unit lacks accepted closure")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device,
             "model_forwards": 0, "vision_forwards": 0, "calls": [], "started_unix": time.time(),
             "source": {"unit": literal_binding(UNIT), "mass_acceptance": literal_binding(MASS_ACCEPTANCE),
                        "mass_result": literal_binding(MASS_RESULT),
                        "val_controls": literal_binding(VAL_CONTROLS)}}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        cases = {name: prepare_case(name, q, identity, device) for name in ("val", "train")}
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer_capture = preserve_source(Path(__file__), run_root=OUT,
                                           relative_name=f"recurrence_positioned_duplicate-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in (
            "probes/training_set_completion/recurrence_attention_mass.py",
            "probes/training_set_completion/recurrence_key_phase.py",
            "probes/training_set_completion/recurrence_phase_transfer.py",
            "probes/training_set_completion/recurrence_history_cache_partition.py",
            "probes/training_set_completion/recurrence_written_content.py",
            "probes/training_set_completion/recurrence_donor_tracking.py",
            "probes/training_set_completion/untied_shared.py",
            "src/qwen/native.py", "src/qwen/input_identity.py", "src/qwen/untied_embeddings.py",
            inspect.getfile(cache_utils), inspect.getfile(modeling_qwen3_vl))]
        captures = [preserve_source(p, run_root=OUT,
                                    relative_name=str(p) if not p.is_absolute() else f"transformers/{p.name}")
                    for p in dependencies]
        require(literal_binding(captures[9])["sha256"] == identity["loader_source"]["sha256"],
                "current untied loader source capture changed")
        manifest = {"schema": "recurrence_positioned_duplicate.cells.v1", "status": "frozen_before_forward",
                    "source": state["source"], "producer": literal_binding(Path(__file__)),
                    "producer_capture": literal_binding(producer_capture),
                    "dependency_captures": [literal_binding(p) for p in captures],
                    "transformers_version": transformers.__version__, "model_identity": identity,
                    "predecessor_ON_semantics": "old pre-K rotated to native new-row phase; not mass/profile ON",
                    "conditions": {"NN": "native destination K/V", "D": "predecessor ON K plus native source-row V"},
                    "cases": {name: {"target_batch": case["target"], "historical_width": case["width"],
                                     "full_width": case["full"], "suffix": case["suffix"],
                                     "source": list(case["source"]), "destination": list(case["dest"]),
                                     "native_input_hashes": {key: tensor_hash(case["native"][key]) for key in
                                                             ("input_ids", "attention_mask", "position_ids")},
                                     "batch_identity": case["original_batch_identity"],
                                     "source_bindings": case["source_bindings"],
                                     "accepted_paths": {**case["accepted_paths"],
                                                        "ON_vector": literal_binding(case["root"] / "ON.pt")}}
                              for name, case in cases.items()},
                    "model_forward_cap": MAX_FORWARDS, "wall_cap_seconds": MAX_SECONDS,
                    "native_NN_parity_atol": ATOL, "winner_gap_guard": .001,
                    "case_payload_cap_bytes": 128 << 20}
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
                results = {name: run_case(name, case, q, forward, state, device)
                           for name, case in cases.items()}
                require(state["model_forwards"] == 6 and state["vision_forwards"] == 2 and
                        time.monotonic() - clock <= MAX_SECONDS,
                        "frozen six-call package incomplete")
                result = {"schema": "recurrence_positioned_duplicate.result.v1", "status": "candidate",
                          "cases": results,
                          "both_cases_native_winner_retained": all(x["native_winner_retained"]
                                                                   for x in results.values()),
                          "source_manifest": literal_binding(OUT / "source-to-cell.json"),
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "model_seconds": time.monotonic() - clock}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"),
                             model_seconds=result["model_seconds"], elapsed_seconds=time.monotonic() - started,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "both_cases_native_winner_retained":
                                  result["both_cases_native_winner_retained"],
                                  "forwards": state["model_forwards"]}))
        finally:
            for handle in handles:
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
