"""Partition one historical coordinate's cached state from its downstream history."""
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
from transformers import DynamicCache
from transformers import cache_utils
from transformers.models.qwen3_vl import modeling_qwen3_vl

from src.artifacts.utf8_json import literal_binding
from src.qwen.saved_prefix import prefix_tokens as _prefix_tokens
from probes.recurrence_dynamics.recurrence_census.prepare import _rows_from_tokens
from probes.recurrence_dynamics.recurrence_donor_tracking import ROW, probability_readback, require, write
from probes.recurrence_dynamics.recurrence_history_location import edit_location
from probes.recurrence_dynamics.recurrence_position_history import PANEL, TARGET, source_and_rows
from probes.recurrence_dynamics.recurrence_written_content import hooks
from probes.model_profiles.mature_tied_untied import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/cache-partition-001")
STAGE_A = OUT.parent / "attempt-001"
STAGE_A_ACCEPTANCE = OUT.parent / "lead-checks/stage-A-readback.json"
WIDTH, SUFFIX, FULL_WIDTH, TARGET_OFFSET = 2112, 6, 2118, 798
SOURCES = {"87": (783, 2109), "86": (774, 2100)}
LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128
MAX_FORWARDS, MAX_SECONDS, ATOL = 14, 900, 2e-4


def partition(native, edited, source_start, p, *, width=WIDTH, target_end=TARGET_OFFSET):
    expected = width + SUFFIX - (target_end - (source_start + 6))
    require(p == expected and 0 <= p < width - 1, "source/target cache boundary changed or overlaps S")
    require(native["input_ids"].shape[1] == width + SUFFIX and edited["input_ids"].shape == native["input_ids"].shape,
            "target replay length changed")
    require(int(native["input_ids"][TARGET, p]) == ROW[6] and int(edited["input_ids"][TARGET, p]) == 152310,
            "wrong source x2 slot")
    require(torch.nonzero(native["input_ids"] != edited["input_ids"], as_tuple=False).tolist() == [[TARGET, p]],
            "edit changed source, companion, or current S")
    require(torch.equal(native["position_ids"], edited["position_ids"]) and torch.equal(native["attention_mask"], edited["attention_mask"]),
            "edit changed positions or mask")
    return {"P": [0, p], "D": [p, p + 1], "H": [p + 1, width], "S": [width, width + SUFFIX]}


def check_cache(cache, *, width=WIDTH, layers=LAYERS, heads=KV_HEADS, dim=HEAD_DIM):
    require(len(cache.layers) == layers, "decoder layer count changed")
    for i, layer in enumerate(cache.layers):
        for name in ("keys", "values"):
            value = getattr(layer, name)
            require(isinstance(value, torch.Tensor) and value.shape == (4, heads, width, dim) and value.dtype == torch.float32,
                    f"layer {i} {name} cache shape/dtype changed")


def check_pair(native, edited, p, *, width=WIDTH, layers=LAYERS):
    check_cache(native, width=width, layers=layers, heads=native.layers[0].keys.shape[1], dim=native.layers[0].keys.shape[-1])
    check_cache(edited, width=width, layers=layers, heads=edited.layers[0].keys.shape[1], dim=edited.layers[0].keys.shape[-1])
    require(0 <= p < width - 1, "source/H overlap or empty H")
    for i, (a, b) in enumerate(zip(native.layers, edited.layers, strict=True)):
        for name in ("keys", "values"):
            x, y = getattr(a, name), getattr(b, name)
            require(torch.equal(x[:TARGET], y[:TARGET]) and torch.equal(x[TARGET + 1:], y[TARGET + 1:]),
                    f"layer {i} {name} companion cache changed")
            require(torch.equal(x[TARGET, :, :p], y[TARGET, :, :p]), f"layer {i} {name} pre-source cache changed")


def cache_digest(cache):
    return [{name: tensor_hash(getattr(layer, name)) for name in ("keys", "values")} for layer in cache.layers]


def segment_hashes(cache, p, *, width=WIDTH):
    return {i: {segment: {name: tensor_hash(getattr(layer, name)[TARGET, :, start:end, :])
                          for name in ("keys", "values")}
                for segment, (start, end) in {"source": (p, p + 1), "downstream": (p + 1, width)}.items()}
            for i, layer in enumerate(cache.layers)}


@contextmanager
def graft(native, edited, p, route, *, width=WIDTH):
    require(route in ("D", "H") and 0 <= p < width - 1, "invalid source/H patch boundary")
    start, end = (p, p + 1) if route == "D" else (p + 1, width)
    require(start < end and end <= width, "source/H overlap")
    saved = []
    try:
        for a, b in zip(native.layers, edited.layers, strict=True):
            for name in ("keys", "values"):
                target, donor = getattr(a, name), getattr(b, name)
                old = target[TARGET, :, start:end, :].clone()
                saved.append((a, name, old))
                target[TARGET, :, start:end, :].copy_(donor[TARGET, :, start:end, :])
        yield
    finally:
        native.crop(width)
        for layer, name, old in saved:
            getattr(layer, name)[TARGET, :, start:end, :].copy_(old)


def layer_hooks(model, cache, cache_position, p, *, width=WIDTH):
    seen = {}
    handles = []
    for i, layer in enumerate(model.model.language_model.layers):
        def observe(_module, _args, kwargs, index=i):
            require(kwargs.get("past_key_values") is cache, "attention consumed another cache")
            require(torch.equal(kwargs.get("cache_position"), cache_position), "attention cache_position changed")
            require(cache.get_seq_length(index) == width, "attention consumed cache with wrong past length")
            mask = kwargs.get("attention_mask")
            require(isinstance(mask, torch.Tensor) and mask.shape[-2:] == (SUFFIX, FULL_WIDTH), "actual causal mask shape changed")
            layer_cache = cache.layers[index]
            seen[index] = {"length_before_update": cache.get_seq_length(index), "mask": tensor_hash(mask),
                           "source": {name: tensor_hash(getattr(layer_cache, name)[TARGET, :, p:p + 1, :]) for name in ("keys", "values")},
                           "downstream": {name: tensor_hash(getattr(layer_cache, name)[TARGET, :, p + 1:width, :]) for name in ("keys", "values")}}
        handles.append(layer.self_attn.register_forward_pre_hook(observe, with_kwargs=True))
    return seen, handles


def selfcheck():
    rows = ROW + ROW + [1, 2, 3, 4, 5, 6]
    base = {"input_ids": torch.tensor([[0] * 24, [0] * 24, rows, [0] * 24]),
            "attention_mask": torch.ones(4, 24, dtype=torch.long), "position_ids": torch.arange(24).expand(3, 4, 24).clone()}
    edit, _ = edit_location(base, rows[-6:], rows, 0, 24)
    require(partition(base, edit, 0, 6, width=18, target_end=24) == {"P": [0, 6], "D": [6, 7], "H": [7, 18], "S": [18, 24]}, "partition selfcheck")
    for bad_start, bad_p in ((0, 7), (12, 17)):
        try:
            partition(base, edit, bad_start, bad_p, width=18, target_end=24)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong source/target/overlap admitted")
    def make_cache():
        return DynamicCache(((torch.zeros(4, 2, 18, 3), torch.zeros(4, 2, 18, 3)),))
    native, edited = make_cache(), make_cache()
    edited.layers[0].keys[TARGET, :, 6:] = 1
    edited.layers[0].values[TARGET, :, 6:] = 2
    check_pair(native, edited, 6, width=18, layers=1)
    baseline = cache_digest(native)
    with torch.inference_mode():
        with graft(native, edited, 6, "D", width=18):
            require(torch.equal(native.layers[0].keys[TARGET, :, 6:7], edited.layers[0].keys[TARGET, :, 6:7]), "source graft failed")
            require(torch.count_nonzero(native.layers[0].keys[TARGET, :, 7:]) == 0, "source/H overlap")
        require(cache_digest(native) == baseline, "no-op restore failed")
        with graft(native, edited, 6, "H", width=18):
            require(torch.count_nonzero(native.layers[0].keys[TARGET, :, 6:7]) == 0, "H/source overlap")
        require(cache_digest(native) == baseline, "H restore failed")
        with graft(native, edited, 6, "D", width=18):
            native.update(torch.zeros(4, 2, 6, 3), torch.zeros(4, 2, 6, 3), 0)
        require(cache_digest(native) == baseline, "S-append restoration failed")
    for kind, row, span in (("companion", 0, slice(None)), ("pre-source", TARGET, slice(0, 6))):
        wrong = make_cache()
        wrong.layers[0].keys[row, :, span, :] = 1
        try:
            check_pair(native, wrong, 6, width=18, layers=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"changed {kind} admitted")


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA required")
    require(not OUT.exists(), "attempt path already exists")
    acceptance = json.loads(STAGE_A_ACCEPTANCE.read_text())
    require(acceptance["status"] == "lead-accepted-stage-A", "Stage A not independently accepted")
    stage_result_path, stage_manifest_path = STAGE_A / "result.json", STAGE_A / "source-to-cell.json"
    stage_result, stage_manifest = json.loads(stage_result_path.read_text()), json.loads(stage_manifest_path.read_text())
    require(stage_result["status"] == "candidate", "Stage A candidate missing")
    reference = {name: STAGE_A / f"{name}.pt" for name in ("R", "L87", "L86")}
    require(all(literal_binding(path) == stage_result["cells"][name]["full_logits"] for name, path in reference.items()), "Stage A vector changed")
    chosen, raw, trace, receipt, panel, group, S, offsets, _ = source_and_rows()
    tokens = raw[TARGET]["token_ids"]
    parsed = _rows_from_tokens(tokens)
    require(offsets[0] == TARGET_OFFSET and S == tokens[792:798] and len(S) == SUFFIX, "target S/offset changed")
    require(all(parsed[int(row)]["start"] == start and tokens[start:start + 9] == ROW for row, (start, _) in SOURCES.items()), "frozen source rows changed")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "started_unix": time.time(), "source": {"stage_A_acceptance": literal_binding(STAGE_A_ACCEPTANCE),
             "stage_A_result": literal_binding(stage_result_path), "stage_A_manifest": literal_binding(stage_manifest_path),
             "stage_A_vectors": {name: literal_binding(path) for name, path in reference.items()},
             "selection": stage_manifest["source"]["selection"], **{key: chosen[key] for key in ("raw", "trace", "runtime_receipt", "image")}}}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        current, saved = dict(identity), dict(receipt["identity"])
        current_loader, saved_loader = current.pop("loader_source"), saved.pop("loader_source")
        require(current == saved and (current_loader["sha256"], current_loader["size_bytes"]) == (saved_loader["sha256"], saved_loader["size_bytes"]), "source/effective model identity changed")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"], "native batch identity changed")
        suffixes = _prefix_tokens(raw, TARGET_OFFSET, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, suffixes, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        require(native["input_ids"].shape == (4, FULL_WIDTH) and all(tensor_hash(native[k]) == stage_manifest["R_tensor_hashes"][k]
                for k in ("input_ids", "attention_mask", "position_ids")), "native Stage A input changed")
        edits = {}
        partitions = {}
        for row, (start, p) in SOURCES.items():
            edited, info = edit_location(native, S, tokens, start)
            partitions[row] = partition(native, edited, start, p)
            require(info["physical_index"] == [TARGET, p], "source physical slot changed")
            require(tensor_hash(edited["input_ids"]) == stage_manifest["cells"][f"L{row}"]["tensor_hashes"]["input_ids"], "Stage A edited input changed")
            edits[row] = edited
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer_capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_history_cache_partition-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ('probes/recurrence_dynamics/recurrence_history_location.py', 'probes/recurrence_dynamics/recurrence_donor_tracking.py',
                        'probes/recurrence_dynamics/recurrence_position_history.py', 'probes/recurrence_dynamics/recurrence_written_content.py',
                        "probes/model_profiles/mature_tied_untied.py", 'probes/recurrence_dynamics/numerical_feedback/runtime.py',
                        "src/qwen/native.py", "src/inference/bound_requests.py", "src/qwen/untied_embeddings.py",
                        inspect.getfile(cache_utils), inspect.getfile(modeling_qwen3_vl))]
        captures = [preserve_source(p, run_root=OUT, relative_name=str(p) if not p.is_absolute() else f"transformers/{p.name}") for p in dependencies]
        require(literal_binding(captures[8])["sha256"] == current_loader["sha256"], "loader source capture changed")
        manifest = {"schema": "recurrence_history_cache_partition.cells.v1", "status": "frozen_before_forward", "source": state["source"],
                    "producer": literal_binding(Path(__file__)), "producer_capture": literal_binding(producer_capture),
                    "dependency_captures": [literal_binding(p) for p in captures], "transformers_version": transformers.__version__,
                    "model_identity": identity, "native_batch_identity": input_identity(batch), "source_panel": literal_binding(PANEL),
                    "target_batch": TARGET, "target_offset": TARGET_OFFSET, "current_S": S, "prefill_width": WIDTH,
                    "partitions": partitions, "full_input_hashes": {name: {k: tensor_hash(inputs[k]) for k in ("input_ids", "attention_mask", "position_ids")}
                     for name, inputs in {"R": native, "L87": edits["87"], "L86": edits["86"]}.items()},
                    "suffix_hashes": {k: tensor_hash(v) for k, v in {"input_ids": native["input_ids"][:, -SUFFIX:],
                     "attention_mask": native["attention_mask"], "position_ids": native["position_ids"][:, :, -SUFFIX:],
                     "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device)}.items()},
                    "model_forward_cap": MAX_FORWARDS, "model_seconds_cap": MAX_SECONDS, "parity_atol": ATOL}
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=literal_binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        clock = time.monotonic()
        def count(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_FORWARDS and time.monotonic() - clock <= MAX_SECONDS, "forward/time cap exceeded")
        def vision(*_):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= 3, "unexpected vision forward during S")
        handles = [q.model.register_forward_pre_hook(count, with_kwargs=True), q.model.model.visual.register_forward_pre_hook(vision)]
        cells, prefills = {}, {}
        common_suffix = {"input_ids": native["input_ids"][:, -SUFFIX:], "attention_mask": native["attention_mask"],
                         "position_ids": native["position_ids"][:, :, -SUFFIX:],
                         "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device), "use_cache": True,
                         "return_dict": True, "logits_to_keep": SUFFIX}
        def observed_forward(name, inputs, *, cache=None, p=None, prefill=False):
            prior_vision = state["vision_forwards"]
            seen, observers = hooks(q.model, inputs["input_ids"], inputs["position_ids"])
            layer_seen, cache_observers = ({}, []) if prefill else layer_hooks(q.model, cache, common_suffix["cache_position"], p)
            try:
                with torch.inference_mode():
                    output = q.model(**inputs)
                    logits = output.logits[TARGET, -1].detach().float().cpu()
            finally:
                for observer in observers + cache_observers:
                    observer.remove()
            require(torch.isfinite(logits).all() and len(seen["embedding_inputs"]) == len(seen["rotary_positions"]) == 1
                    and len(seen["masks"]) == len(seen["cache_slots"]) == 2, "actual token/position/mask consumption invalid")
            require(seen["masks"][0] == seen["masks"][1] and seen["cache_slots"][0] == seen["cache_slots"][1], "decoder layer mask/cache mismatch")
            if prefill:
                require(output.past_key_values is inputs["past_key_values"], "prefill returned another cache")
                check_cache(output.past_key_values)
                require(state["vision_forwards"] == prior_vision + 1, "prefill did not consume one image pass")
                prefills[name] = {"consumed": seen, "cache_length": output.past_key_values.get_seq_length(),
                                  "cache_digest": cache_digest(output.past_key_values)}
                state["last_cell"] = name
                write(OUT / "partial-results.json", {"status": "running", "prefills": prefills, "cells": cells})
                write(OUT / "receipt.json", state)
                return output.past_key_values, logits
            else:
                require(set(layer_seen) == set(range(LAYERS)), "not all cache layers consumed")
                require(cache.get_seq_length() == FULL_WIDTH, "S cache append length changed")
                require(state["vision_forwards"] == prior_vision, "vision ran in S continuation")
            top = torch.topk(logits, 5)
            path = OUT / f"{name}.pt"
            torch.save(logits, path)
            cells[name] = {"full_logits": literal_binding(path), "argmax_token": int(top.indices[0]),
                           "top1_top2_gap": float(top.values[0] - top.values[1]),
                           "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices, top.values, strict=True)],
                           "z38_minus_z999": float(logits[151708] - logits[152669]),
                           "absolute_coordinates": probability_readback(logits, (0, 38, 640, 999)),
                           "consumed": seen, "consumed_cache_layers": layer_seen,
                           "cache_length_after": output.past_key_values.get_seq_length()}
            state["last_cell"] = name
            write(OUT / "partial-results.json", {"status": "running", "prefills": prefills, "cells": cells})
            write(OUT / "receipt.json", state)
            return output.past_key_values, logits
        try:
            with torch.inference_mode():
                def prefill_history(label, full):
                    cache = DynamicCache()
                    prefill = dict(full)
                    prefill.update(input_ids=full["input_ids"][:, :WIDTH], attention_mask=full["attention_mask"][:, :WIDTH],
                                   position_ids=full["position_ids"][:, :, :WIDTH], cache_position=torch.arange(WIDTH, device=device),
                                   past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache, _ = observed_forward(f"prefill_{label}", prefill, prefill=True)
                    return cache

                def score_suffix(label, cache, p):
                    suffix = dict(common_suffix, past_key_values=cache)
                    cache, logits = observed_forward(label, suffix, cache=cache, p=p)
                    cache.crop(WIDTH)
                    return logits

                def qualify(label, cache, p):
                    logits = score_suffix(label, cache, p)
                    require(cache_digest(cache) == prefills[f"prefill_{label}"]["cache_digest"], "baseline cache changed after crop")
                    accepted = torch.load(reference[label], map_location="cpu", weights_only=True)
                    cells[label]["max_abs_vs_stage_A_full_vocab"] = float((logits - accepted).abs().max())
                    require(cells[label]["max_abs_vs_stage_A_full_vocab"] <= ATOL and cells[label]["argmax_token"] == stage_result["cells"][label]["argmax_token"],
                            f"cached {label} full-vector parity failed")
                    if label != "R":
                        for field in ("embedding_inputs", "rotary_positions", "masks", "cache_slots"):
                            require(cells[label]["consumed"][field] == cells["R"]["consumed"][field], f"{label} S consumption changed {field}")
                    return logits

                native_cache = prefill_history("R", native)
                native_digest = prefills["prefill_R"]["cache_digest"]
                qualify("R", native_cache, 2109)
                for row in ("87", "86"):
                    label = f"L{row}"
                    p = SOURCES[row][1]
                    edited_cache = prefill_history(label, edits[row])
                    check_pair(native_cache, edited_cache, p)
                    for field in ("rotary_positions", "masks", "cache_slots"):
                        require(prefills[f"prefill_{label}"]["consumed"][field] == prefills["prefill_R"]["consumed"][field],
                                f"{label} prefill changed consumed {field}")
                    edited_digest = prefills[f"prefill_{label}"]["cache_digest"]
                    native_parts, edited_parts = segment_hashes(native_cache, p), segment_hashes(edited_cache, p)
                    qualify(label, edited_cache, p)
                    require(cache_digest(edited_cache) == edited_digest, "edited baseline cache changed")
                    for route in ("D", "H"):
                        hybrid = f"{route}{row}"
                        with graft(native_cache, edited_cache, p, route):
                            check_cache(native_cache)
                            seen_parts = segment_hashes(native_cache, p)
                            for i in range(LAYERS):
                                require(seen_parts[i]["source"] == (edited_parts if route == "D" else native_parts)[i]["source"]
                                        and seen_parts[i]["downstream"] == (native_parts if route == "D" else edited_parts)[i]["downstream"],
                                        "hybrid source/H membership changed")
                            score_suffix(hybrid, native_cache, p)
                            consumed = cells[hybrid]["consumed_cache_layers"]
                            for i in range(LAYERS):
                                require(consumed[i]["source"] == seen_parts[i]["source"] and consumed[i]["downstream"] == seen_parts[i]["downstream"],
                                        "attention did not consume hybrid cache slices")
                            for field in ("embedding_inputs", "rotary_positions", "masks", "cache_slots"):
                                require(cells[hybrid]["consumed"][field] == cells["R"]["consumed"][field],
                                        f"{hybrid} S consumption changed {field}")
                        require(cache_digest(native_cache) == native_digest, "native cache did not restore exactly")
                        cells[hybrid]["native_cache_restored_sha256"] = hashlib.sha256(json.dumps(native_digest, sort_keys=True).encode()).hexdigest()
                    del edited_cache
                require(state["model_forwards"] == 10 and state["vision_forwards"] == 3, "frozen forward count changed")
                require(time.monotonic() - clock <= MAX_SECONDS, "model execution time cap exceeded")
                d0 = cells["R"]["z38_minus_z999"]
                effects = {}
                for row in ("87", "86"):
                    f, d, h = (cells[f"{prefix}{row}"]["z38_minus_z999"] for prefix in ("L", "D", "H"))
                    effects[row] = {"E_F": f - d0, "E_D": d - d0, "E_H": h - d0, "interaction": f - d - h + d0}
                delta = {key: effects["87"][key] - effects["86"][key] for key in ("E_F", "E_D", "E_H")}
                delta["interaction_difference"] = effects["87"]["interaction"] - effects["86"]["interaction"]
                delta["identity_residual"] = delta["E_F"] - delta["E_D"] - delta["E_H"] - delta["interaction_difference"]
                tol = max(0.01, 0.1 * abs(delta["E_F"]))
                delta["interpretation_tolerance"] = tol
                delta["D_preserves"] = delta["E_D"] * delta["E_F"] > 0 and abs(delta["E_D"] - delta["E_F"]) <= tol
                delta["H_preserves"] = delta["E_H"] * delta["E_F"] > 0 and abs(delta["E_H"] - delta["E_F"]) <= tol
                delta["D_negligible"] = abs(delta["E_D"]) <= tol
                delta["H_negligible"] = abs(delta["E_H"]) <= tol
                result = {"schema": "recurrence_history_cache_partition.result.v1", "status": "candidate", "cells": cells,
                          "prefills": prefills, "effects": effects, "near_minus_older": delta, "numerical_guard": 0.001,
                          "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"],
                          "model_seconds": time.monotonic() - clock, "source_manifest": literal_binding(OUT / "source-to-cell.json")}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"),
                             model_seconds=result["model_seconds"], elapsed_seconds=time.monotonic() - started,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "result": str(OUT / "result.json"), "forwards": state["model_forwards"]}))
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
