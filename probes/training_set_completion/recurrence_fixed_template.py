"""Replace later repeated-history rows with one position-correct first-row template."""
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
    HEAD_DIM, KV_HEADS, LAYERS, prepare_case, same_binding, fixed_history_snapshot,
    fixed_history_equal,
)
from probes.training_set_completion.recurrence_donor_tracking import probability_readback, logprob, require, write
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest, check_cache
from probes.training_set_completion.recurrence_key_phase import rotate
from probes.training_set_completion.recurrence_positioned_duplicate import prior_nn_observer
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import tensor_hash


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-22-recurrence-fixed-template"
OUT = ROOT / "attempt-001"
UNIT = Path("research/experiments/2026-09-22-recurrence-fixed-template/unit.md")
SELECTION = ROOT / "selection.json"
PREVIOUS = BASE / "2026-09-22-recurrence-positioned-duplicate"
PREVIOUS_ACCEPTANCE = PREVIOUS / "lead-acceptance.json"
PREVIOUS_RESULT = PREVIOUS / "attempt-001/result.json"
ATOL, MAX_FORWARDS, MAX_SECONDS = 2e-4, 10, 900


def geometry(meta, native, target):
    source = tuple(meta["first_source"])
    all_dest = tuple(meta["all_destination"])
    last_dest = tuple(meta["last_destination"])
    width, full, tokens = meta["historical_width"], meta["full_width"], meta["template_token_ids"]
    require(meta["target_batch"] == target and meta["all_rows_exact_token_match"] and
            meta["intervening_nonrepeat_positions"] == 0 and len(tokens) == 9 and
            source[1] == all_dest[0] and all_dest[1] == last_dest[1] == width and
            last_dest[0] == width - 9 and full == native["input_ids"].shape[1] and
            (all_dest[1] - source[0]) // 9 == meta["row_count"] and
            (all_dest[1] - source[0]) % 9 == 0,
            "frozen first/later repeated row geometry changed")
    rows = [(p, p + 9) for p in range(all_dest[0], all_dest[1], 9)]
    require(len(rows) == meta["row_count"] - 1 and rows[-1] == last_dest and
            all(native["input_ids"][target, a:b].tolist() == tokens
                for a, b in [source, *rows]) and
            native["input_ids"].shape[0] == 4,
            "actual native historical rows differ from frozen template")
    return source, rows


def layout_rows(cell, rows):
    require(cell in ("NN", "LAST", "ALL") and rows and
            all(b - a == 9 for a, b in rows), "invalid fixed-template arm or rows")
    return [] if cell == "NN" else rows[-1:] if cell == "LAST" else rows


def rotated_template(pre_key, cos, sin, row):
    a, b = row
    require(a >= 0 and a % 9 == 0 and b == a + 9 and
            pre_key.shape == (KV_HEADS, 9, HEAD_DIM) and
            cos[a:b].shape == sin[a:b].shape == (9, HEAD_DIM),
            "template/phase destination mismatch")
    return rotate(pre_key, cos[a:b], sin[a:b])


@contextmanager
def patch_rows(cache, assignments, *, cell, target, source, rows, width, digest):
    chosen = layout_rows(cell, rows)
    require(source[1] == rows[0][0] and rows[-1][1] == width and
            len(cache.layers) == len(assignments) == LAYERS,
            "source/row boundary or layer count changed")
    saved = []
    try:
        for layer, records in zip(cache.layers, assignments, strict=True):
            require([tuple(record["row"]) for record in records] == chosen,
                    "wrong destination rows for arm")
            old_k, old_v = layer.keys, layer.values
            saved.append((layer, old_k, old_v,
                          old_k[target, :, rows[0][0]:width, :].clone(),
                          old_v[target, :, rows[0][0]:width, :].clone()))
            for record in records:
                a, b = record["row"]
                key, value = record["K"], record["V"]
                require(key.shape == value.shape == (KV_HEADS, 9, HEAD_DIM),
                        "wrong key/value row shape")
                layer.keys[target, :, a:b, :].copy_(key)
                layer.values[target, :, a:b, :].copy_(value)
        yield
    finally:
        for layer, old_k, old_v, block_k, block_v in saved:
            layer.keys, layer.values = old_k, old_v
            layer.keys[target, :, rows[0][0]:width, :].copy_(block_k)
            layer.values[target, :, rows[0][0]:width, :].copy_(block_v)
        require(cache_digest(cache) == digest and cache.get_seq_length() == width,
                "native historical cache reference/block restoration failed")


def selfcheck():
    """Reject row/phase/V mistakes and S-append or exceptional restoration leakage."""
    rows = [(9, 18), (18, 27), (27, 36)]
    source = (0, 9)
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9] * 4).expand(4, 36).clone()
    native = {"input_ids": ids}
    meta = {"first_source": [0, 9], "all_destination": [9, 36], "last_destination": [27, 36],
            "historical_width": 36, "full_width": 36, "template_token_ids": ids[0, :9].tolist(),
            "target_batch": 2, "all_rows_exact_token_match": True, "intervening_nonrepeat_positions": 0,
            "row_count": 4}
    assert geometry(meta, native, 2) == (source, rows)
    k = torch.zeros(4, KV_HEADS, 36, HEAD_DIM)
    v = torch.zeros_like(k)
    v[2, :, :9] = 3
    cache = DynamicCache(tuple((k.clone(), v.clone()) for _ in range(LAYERS)))
    digest = cache_digest(cache)
    pre = torch.ones(KV_HEADS, 9, HEAD_DIM)
    cos, sin = torch.ones(36, HEAD_DIM), torch.zeros(36, HEAD_DIM)
    sin[18:27] = 1; cos[18:27] = 0
    candidate = rotated_template(pre, cos, sin, rows[1])
    require(not torch.equal(candidate, rotated_template(pre, cos, sin, rows[0])),
            "phase mismatch sensitivity absent")
    for bad in ((9, 17), (17, 26)):
        try:
            rotated_template(pre, cos, sin, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong row phase slice admitted")
    def records(which):
        return [[{"row": row, "K": rotated_template(pre, cos, sin, row),
                  "V": cache.layers[i].values[2, :, :9].clone()}
                 for row in layout_rows(which, rows)] for i in range(LAYERS)]
    wrong = records("ALL")
    wrong[0][1]["K"] = wrong[0][0]["K"]
    require(not torch.equal(wrong[0][1]["K"], candidate), "wrong row-phase alignment escaped")
    wrong[0][1]["V"] = torch.zeros_like(wrong[0][1]["V"])
    require(not torch.equal(wrong[0][1]["V"], v[2, :, :9]), "wrong V assignment escaped")
    with torch.inference_mode():
        for cell in ("LAST", "ALL"):
            with patch_rows(cache, records(cell), cell=cell, target=2, source=source,
                            rows=rows, width=36, digest=digest):
                require(torch.equal(cache.layers[0].keys[2, :, 18:27],
                                    candidate if cell == "ALL" else k[2, :, 18:27]) and
                        torch.equal(cache.layers[0].values[2, :, 27:36], v[2, :, :9]) and
                        torch.equal(cache.layers[0].keys[[0, 1, 3]], k[[0, 1, 3]]),
                        "actual-cache LAST/ALL assignment wrong")
                cache.update(torch.zeros(4, KV_HEADS, 2, HEAD_DIM),
                             torch.zeros(4, KV_HEADS, 2, HEAD_DIM), 0)
            require(cache_digest(cache) == digest, "append leaked through restoration")
        try:
            with patch_rows(cache, records("ALL"), cell="ALL", target=2, source=source,
                            rows=rows, width=36, digest=digest):
                raise RuntimeError("injected S failure")
        except RuntimeError:
            pass
        require(cache_digest(cache) == digest, "exception restoration leaked")
        try:
            with patch_rows(cache, records("LAST"), cell="ALL", target=2, source=source,
                            rows=rows, width=36, digest=digest):
                pass
        except ValueError:
            pass
        else:
            raise AssertionError("LAST layout admitted as ALL")


def run_case(name, case, chosen, q, forward, state, device):
    target, width, full, suffix = (case[k] for k in ("target", "width", "full", "suffix"))
    native = case["native"]
    source, rows = geometry(chosen, native, target)
    require(tuple(chosen["raw_binding"][k] for k in ("sha256", "size_bytes")) ==
            tuple(case["source_bindings"]["raw"][k] for k in ("sha256", "size_bytes")) and
            chosen["raw_action_offset"] == (807 if name == "val" else 184),
            "frozen selection/raw binding changed")
    case_out = OUT / name
    case_out.mkdir()
    cache = DynamicCache()
    prefill = dict(native)
    prefill.update(input_ids=native["input_ids"][:, :width],
                   attention_mask=native["attention_mask"][:, :width],
                   position_ids=native["position_ids"][:, :, :width],
                   cache_position=torch.arange(width, device=device),
                   past_key_values=cache, use_cache=True, logits_to_keep=1)
    prekeys, phase_seen, prefill_handles = [None] * LAYERS, [], []
    def rotary(_module, _args, output):
        cos, sin = output
        require(cos.shape == sin.shape == (4, width, HEAD_DIM), "historical phase shape changed")
        phase_seen.append((cos[target, source[0]:width].detach().clone(),
                           sin[target, source[0]:width].detach().clone()))
    prefill_handles.append(q.model.model.language_model.rotary_emb.register_forward_hook(rotary))
    for i, layer in enumerate(q.model.model.language_model.layers):
        def normalized(_module, _args, output, *, index=i):
            require(output.shape == (4, width, KV_HEADS, HEAD_DIM), "historical normalized pre-K changed")
            prekeys[index] = output[target, source[0]:source[1]].transpose(0, 1).detach().clone()
        prefill_handles.append(layer.self_attn.k_norm.register_forward_hook(normalized))
    prefill_seen, standard = hooks(q.model, prefill["input_ids"], prefill["position_ids"])
    prefill_handles.extend(standard)
    try:
        output = forward(f"{name}:historical_prefill", prefill)
    finally:
        for h in prefill_handles:
            h.remove()
    require(output.past_key_values is cache and len(phase_seen) == 1 and
            all(x is not None for x in prekeys), "historical first-template observation incomplete")
    del output
    check_cache(cache, width=width)
    cos, sin = phase_seen[0]
    cos = cos.contiguous(); sin = sin.contiguous()
    first_v, first_k, metrics = [], [], []
    for i, layer in enumerate(cache.layers):
        K = layer.keys[target, :, source[0]:source[1], :]
        V = layer.values[target, :, source[0]:source[1], :]
        replay = rotate(prekeys[i], cos[:9], sin[:9])
        error = float((replay - K).abs().max())
        first_k.append(K.detach().clone())
        first_v.append(V.detach().clone())
        metrics.append({"layer": i, "first_preK_hash": tensor_hash(prekeys[i]),
                        "first_native_postK_hash": tensor_hash(K), "first_V_hash": tensor_hash(V),
                        "first_preK_native_phase_replay_max_abs": error})
    digest = cache_digest(cache)
    fixed = fixed_history_snapshot(cache, target, source[0])
    native_other_hashes = [{axis: tensor_hash(getattr(layer, axis)[target, :, rows[0][0]:rows[-1][0], :])
                            for axis in ("keys", "values")} for layer in cache.layers]
    template_path = case_out / "first-template-and-phases.pt"
    torch.save({"first_source": source, "later_rows": rows,
                "first_preK": [x.cpu() for x in prekeys], "first_V": [x.cpu() for x in first_v],
                "native_phase_cos": cos.cpu(), "native_phase_sin": sin.cpu()}, template_path)
    write(case_out / "prefill-readback.json", {"consumed": prefill_seen,
         "first_source": source, "later_rows": rows,
         "template_and_phases": literal_binding(template_path), "per_layer": metrics,
         "cache_digest": digest, "cache_length": cache.get_seq_length()})
    state["last_prefill"] = name
    write(OUT / "receipt.json", state)
    require(all(x["first_preK_native_phase_replay_max_abs"] <= ATOL for x in metrics),
            f"{name} first-row native pre-K/post-K replay failed")
    _, accepted_current_phase = prior_nn_observer(case, name)
    cells = {}
    for cell in ("NN", "LAST", "ALL"):
        chosen_rows = layout_rows(cell, rows)
        assignments = []
        for i in range(LAYERS):
            records = []
            for a, b in chosen_rows:
                key = rotated_template(prekeys[i], cos, sin, (a - source[0], b - source[0]))
                records.append({"row": (a, b), "K": key, "V": first_v[i]})
            assignments.append(records)
        with patch_rows(cache, assignments, cell=cell, target=target, source=source,
                        rows=rows, width=width, digest=digest):
            suffix_inputs = {"input_ids": native["input_ids"][:, width:full],
                             "attention_mask": native["attention_mask"],
                             "position_ids": native["position_ids"][:, :, width:full],
                             "cache_position": torch.arange(width, full, device=device),
                             "past_key_values": cache, "use_cache": True,
                             "return_dict": True, "logits_to_keep": suffix}
            suffix_seen, suffix_hooks = hooks(q.model, suffix_inputs["input_ids"], suffix_inputs["position_ids"])
            layer_seen, consumers = [None] * LAYERS, []
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
                            "actual current S mask/slots/phase/cache changed")
                    phase_ok = (all(torch.equal(phase_now[j][target].cpu(), accepted_current_phase[k])
                                    for j, k in enumerate(("cos", "sin"))) if name == "val" else
                                [tensor_hash(x) for x in phase_now] == accepted_current_phase)
                    current = cache.layers[index]
                    require(phase_ok and fixed_history_equal(current, fixed[index], target, source[0]) and
                            torch.equal(current.keys[target, :, source[0]:source[1], :], first_k[index]) and
                            torch.equal(current.values[target, :, source[0]:source[1], :], first_v[index]),
                            "first row, outside history, companions or native S phase changed")
                    records = []
                    for j, (a, b) in enumerate(rows):
                        key = current.keys[target, :, a:b, :]
                        value = current.values[target, :, a:b, :]
                        native_row = cell == "NN" or (cell == "LAST" and j < len(rows) - 1)
                        if not native_row:
                            expected = assignments[index][j if cell == "ALL" else 0]
                            require(torch.equal(key, expected["K"]) and torch.equal(value, expected["V"]),
                                    "wrong destination phase/template K/V consumed")
                        records.append({"row": [a, b], "native_unchanged": native_row,
                                        "K": tensor_hash(key), "V": tensor_hash(value)})
                    other_exact = (cell == "ALL" or
                                   all(tensor_hash(getattr(current, axis)[target, :, rows[0][0]:rows[-1][0], :]) ==
                                       native_other_hashes[index][axis] for axis in ("keys", "values")))
                    require(other_exact, "LAST changed earlier repeated rows")
                    layer_seen[index] = {"layer": index, "prefix_length": cache.get_seq_length(index),
                                         "mask_hash": tensor_hash(mask), "slots_hash": tensor_hash(slots),
                                         "phase_hashes": [tensor_hash(x) for x in phase_now],
                                         "phase_matches_accepted_NN": phase_ok,
                                         "first_K_hash": tensor_hash(current.keys[target, :, source[0]:source[1], :]),
                                         "first_V_hash": tensor_hash(current.values[target, :, source[0]:source[1], :]),
                                         "outside_and_companions_exact": True,
                                         "other_repeated_rows_exact": other_exact,
                                         "destination_rows": records}
                consumers.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
            try:
                output = forward(f"{name}:{cell}", suffix_inputs)
                logits = output.logits[target, -1].detach().float().cpu()
                companion_hash = tensor_hash(output.logits[[i for i in range(4) if i != target], -1].detach())
            finally:
                for h in suffix_hooks + consumers:
                    h.remove()
            require(torch.isfinite(logits).all() and cache.get_seq_length() == full and
                    all(x is not None for x in layer_seen), f"{name}:{cell} consumer observation incomplete")
            vector_path = case_out / f"{cell}.pt"
            torch.save(logits, vector_path)
            readback_path = case_out / f"{cell}-readback.json"
            write(readback_path, {"consumed": suffix_seen, "layers": layer_seen,
                                 "companion_final_logits_hash": companion_hash,
                                 "cache_length_after": cache.get_seq_length()})
            top = torch.topk(logits, 5)
            declared = (38, 999) if name == "val" else (348, 350, 591)
            lp = logprob(logits)
            cells[cell] = {"full_logits": literal_binding(vector_path),
                           "readback": literal_binding(readback_path),
                           "argmax_token": int(top.indices[0]),
                           "top1_top2_gap": float(top.values[0] - top.values[1]),
                           "top5": [{"token": int(t), "logit": float(v),
                                     "logprob": float(lp[int(t)]), "probability": float(lp[int(t)].exp())}
                                    for t, v in zip(top.indices, top.values, strict=True)],
                           "declared_coordinates": probability_readback(logits, declared),
                           "margins": ({"z38_minus_z999": float(logits[151708] - logits[152669])}
                                       if name == "val" else
                                       {"z350_minus_z348": float(logits[152020] - logits[152018]),
                                        "z591_minus_z348": float(logits[152261] - logits[152018])})}
            state["last_cell"] = f"{name}:{cell}"
            write(OUT / "receipt.json", state)
            require(all(x["mask_hash"] == case["prior_mask_hashes"][i] and
                        x["outside_and_companions_exact"] and x["other_repeated_rows_exact"] and
                        x["prefix_length"] == width for i, x in enumerate(layer_seen)) and
                    len(suffix_seen["embedding_inputs"]) == len(suffix_seen["rotary_positions"]) == 1 and
                    len(suffix_seen["masks"]) == len(suffix_seen["cache_slots"]) == 2,
                    f"{name}:{cell} original S/mask/phase/cache consumption failed")
            if cell != "NN":
                nn = json.loads((case_out / "NN-readback.json").read_text())
                require(suffix_seen == nn["consumed"] and companion_hash == nn["companion_final_logits_hash"] and
                        all(x["mask_hash"] == y["mask_hash"] and x["slots_hash"] == y["slots_hash"] and
                            x["phase_hashes"] == y["phase_hashes"] for x, y in
                            zip(layer_seen, nn["layers"], strict=True)),
                        f"{name}:{cell} changed S/positions/mask or companions")
        if cell == "NN":
            ref_path = case["root"] / "NN.pt"
            require(same_binding(ref_path, case["accepted_result"]["cells"]["NN"]["full_logits"]),
                    "accepted native NN reference changed")
            ref = torch.load(ref_path, map_location="cpu", weights_only=True)
            cells[cell]["fullvector_max_abs_vs_accepted_NN"] = float((logits - ref).abs().max())
            require(cells[cell]["fullvector_max_abs_vs_accepted_NN"] <= ATOL and
                    cells[cell]["argmax_token"] == case["accepted_result"]["cells"]["NN"]["argmax_token"],
                    f"{name} native NN fullvector parity failed")
            nn_logits = logits
        else:
            cells[cell]["vs_fresh_NN"] = {"fullvector_max_abs": float((logits - nn_logits).abs().max()),
                                          "fullvector_l2": float(torch.linalg.vector_norm(logits.double() - nn_logits.double()))}
    payload = sum(p.stat().st_size for p in case_out.rglob("*") if p.is_file())
    require(payload <= 128 << 20, f"{name} intermediate payload cap exceeded")
    outcomes = {cell: {"native_winner_retained": cells[cell]["argmax_token"] == case["new_token"] and
                                          cells[cell]["top1_top2_gap"] > .001,
                       "near_tie": cells[cell]["top1_top2_gap"] <= .001}
                for cell in ("LAST", "ALL")}
    write(case_out / "result.json", {"status": "candidate", "case": name,
                                     "first_donor": source, "repeated_rows": len(rows) + 1,
                                     "cells": cells, "outcomes": outcomes,
                                     "prefill_readback": literal_binding(case_out / "prefill-readback.json"),
                                     "payload_bytes_before_summary": payload})
    return {"result": literal_binding(case_out / "result.json"), "outcomes": outcomes,
            "winners": {cell: cells[cell]["argmax_token"] for cell in cells},
            "gaps": {cell: cells[cell]["top1_top2_gap"] for cell in cells}}


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda") and not OUT.exists(),
            "CUDA and unused attempt path required")
    selected = json.loads(SELECTION.read_text())
    require(selected["status"] == "root-verified-frozen" and set(selected["cases"]) == {"val", "train"},
            "root's fixed-template selection changed")
    closure = json.loads(PREVIOUS_ACCEPTANCE.read_text())
    require(closure["status"] == "lead-accepted" and
            same_binding(PREVIOUS_RESULT, next(x for x in closure["artifacts"]
                                               if x["path"] == str(PREVIOUS_RESULT))),
            "preceding positioned-duplicate unit lacks accepted closure")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device,
             "model_forwards": 0, "vision_forwards": 0, "calls": [], "started_unix": time.time(),
             "source": {"unit": literal_binding(UNIT), "selection": literal_binding(SELECTION),
                        "previous_acceptance": literal_binding(PREVIOUS_ACCEPTANCE),
                        "previous_result": literal_binding(PREVIOUS_RESULT)}}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        cases = {name: prepare_case(name, q, identity, device) for name in ("val", "train")}
        for name, case in cases.items():
            geometry(selected["cases"][name], case["native"], case["target"])
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer_capture = preserve_source(Path(__file__), run_root=OUT,
                                           relative_name=f"recurrence_fixed_template-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in (
            "probes/training_set_completion/recurrence_positioned_duplicate.py",
            "probes/training_set_completion/recurrence_attention_mass.py",
            "probes/training_set_completion/recurrence_key_phase.py",
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
        manifest = {"schema": "recurrence_fixed_template.cells.v1", "status": "frozen_before_forward",
                    "source": state["source"], "producer": literal_binding(Path(__file__)),
                    "producer_capture": literal_binding(producer_capture),
                    "dependency_captures": [literal_binding(p) for p in captures],
                    "transformers_version": transformers.__version__, "model_identity": identity,
                    "conditions": {"NN": "native", "LAST": "first template in last row only",
                                   "ALL": "first template in every later repeated row"},
                    "cases": {name: {"target_batch": case["target"],
                                     "first_source": selected["cases"][name]["first_source"],
                                     "all_destination": selected["cases"][name]["all_destination"],
                                     "last_destination": selected["cases"][name]["last_destination"],
                                     "native_input_hashes": {key: tensor_hash(case["native"][key]) for key in
                                                             ("input_ids", "attention_mask", "position_ids")},
                                     "batch_identity": case["original_batch_identity"],
                                     "source_bindings": case["source_bindings"],
                                     "accepted_paths": case["accepted_paths"]}
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
                results = {name: run_case(name, case, selected["cases"][name], q, forward, state, device)
                           for name, case in cases.items()}
                require(state["model_forwards"] == 8 and state["vision_forwards"] == 2 and
                        time.monotonic() - clock <= MAX_SECONDS,
                        "frozen eight-call package incomplete")
                result = {"schema": "recurrence_fixed_template.result.v1", "status": "candidate",
                          "cases": results,
                          "both_ALL_native_winner_retained": all(x["outcomes"]["ALL"]["native_winner_retained"]
                                                                 for x in results.values()),
                          "source_manifest": literal_binding(OUT / "source-to-cell.json"),
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "model_seconds": time.monotonic() - clock}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"),
                             model_seconds=result["model_seconds"], elapsed_seconds=time.monotonic() - started,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "both_ALL_native_winner_retained":
                                  result["both_ALL_native_winner_retained"],
                                  "forwards": state["model_forwards"]}))
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
