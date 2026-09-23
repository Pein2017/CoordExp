"""Test native repeated-row cached attention mass at the row89 x2 exit."""
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
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_donor_tracking import ROW, probability_readback, require, write
from probes.training_set_completion.recurrence_history_cache_partition import cache_digest, check_cache
from probes.training_set_completion.recurrence_position_history import PANEL, TARGET, source_and_rows
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/native-row-mass-002")
ATTEMPT1 = OUT.parent / "native-row-mass-001"
PRIMARY = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/attempt-002")
STAGE_B_ACCEPTANCE = OUT.parent / "lead-checks/stage-B-readback.json"
WIDTH, FULL_WIDTH, SUFFIX, TARGET_OFFSET = 2121, 2127, 6, 807
SOURCE, DEST = (2103, 2112), (2112, 2121)
LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128
MAX_FORWARDS, MAX_SECONDS, ATOL = 9, 900, 2e-4


def blocks(source=SOURCE, dest=DEST, *, width=WIDTH):
    require(source == (width - 18, width - 9) and dest == (width - 9, width),
            "source/destination block boundary changed or overlaps S")
    return source, dest


def mask_added_row(mask, *, dest=DEST):
    blocks(dest=dest, width=mask.shape[1] - SUFFIX)
    require(mask.shape == (4, FULL_WIDTH) and bool(torch.all(mask[TARGET, dest[0]:dest[1]] == 1)),
            "late full attention mask changed")
    changed = mask.clone()
    changed[TARGET, dest[0]:dest[1]] = 0
    require(torch.nonzero(changed != mask, as_tuple=False).tolist() == [[TARGET, i] for i in range(*dest)],
            "mask changed outside added target row")
    return changed


def block_hashes(cache, *, source=SOURCE, dest=DEST):
    return {i: {part: {name: tensor_hash(getattr(layer, name)[TARGET, :, start:end, :])
                      for name in ("keys", "values")}
                for part, (start, end) in {"source": source, "dest": dest}.items()}
            for i, layer in enumerate(cache.layers)}


def check_actual_mask(mask, native_mask, mode, *, dest=DEST):
    require(mask.dtype == native_mask.dtype == torch.bool and mode in ("native", "M"), "actual SDPA mask type/mode changed")
    changed = torch.nonzero(mask != native_mask, as_tuple=False).tolist()
    expected = [] if mode == "native" else [[TARGET, 0, q, k] for q in range(SUFFIX) for k in range(*dest)]
    require(changed == expected, "actual mask changed outside declared added-row visibility")
    if mode == "M":
        require(bool(torch.all(~mask[TARGET, 0, :, dest[0]:dest[1]])) and
                bool(torch.all(native_mask[TARGET, 0, :, dest[0]:dest[1]])), "M did not exclude added row")
    return len(changed)


@contextmanager
def graft(cache, axes, *, source=SOURCE, dest=DEST, width=WIDTH):
    blocks(source, dest, width=width)
    require(axes in (("keys", "values"), ("keys",), ("values",)), "wrong K/V selection")
    saved = []
    try:
        for layer in cache.layers:
            for axis in axes:
                value = getattr(layer, axis)
                original = value[TARGET, :, dest[0]:dest[1], :].clone()
                saved.append((layer, axis, original))
                value[TARGET, :, dest[0]:dest[1], :].copy_(value[TARGET, :, source[0]:source[1], :])
        yield
    finally:
        cache.crop(width)
        for layer, axis, original in saved:
            getattr(layer, axis)[TARGET, :, dest[0]:dest[1], :].copy_(original)


@contextmanager
def verify_restored(cache, digest, *, width=WIDTH):
    try:
        yield
    finally:
        cache.crop(width)
        require(cache_digest(cache) == digest, "native cache not restored exactly after S")


def layer_hooks(model, cache, cache_position, native_mask, mask_mode, *, source=SOURCE, dest=DEST):
    seen, handles, actual_mask = {}, [], []
    for i, layer in enumerate(model.model.language_model.layers):
        def observe(_module, _args, kwargs, index=i):
            require(kwargs.get("past_key_values") is cache, "attention consumed another cache")
            position = kwargs.get("cache_position")
            require(isinstance(position, torch.Tensor) and torch.equal(position, cache_position), "attention cache slots changed")
            require(cache.get_seq_length(index) == WIDTH, "attention past length changed")
            mask = kwargs.get("attention_mask")
            require(isinstance(mask, torch.Tensor) and mask.shape == (4, 1, SUFFIX, FULL_WIDTH), "actual causal mask shape changed")
            if native_mask is None:
                if not actual_mask:
                    actual_mask.append(mask.detach().clone())
                else:
                    require(torch.equal(mask, actual_mask[0]), "native mask differs across layers")
                change_count = 0
            else:
                change_count = check_actual_mask(mask, native_mask, mask_mode, dest=dest)
            layer_cache = cache.layers[index]
            seen[index] = {"length_before_update": cache.get_seq_length(index), "mask_hash": tensor_hash(mask),
                           "mask_change_count_vs_N": change_count,
                           "source": {axis: tensor_hash(getattr(layer_cache, axis)[TARGET, :, source[0]:source[1], :]) for axis in ("keys", "values")},
                           "dest": {axis: tensor_hash(getattr(layer_cache, axis)[TARGET, :, dest[0]:dest[1], :]) for axis in ("keys", "values")}}
        handles.append(layer.self_attn.register_forward_pre_hook(observe, with_kwargs=True))
    return seen, handles, actual_mask


def selfcheck():
    blocks((0, 9), (9, 18), width=18)
    for source, dest in (((1, 10), (9, 18)), ((0, 9), (8, 17)), ((0, 9), (18, 27))):
        try:
            blocks(source, dest, width=18)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong or overlapping block admitted")
    def make_cache():
        k, v = torch.zeros(4, 2, 18, 3), torch.zeros(4, 2, 18, 3)
        k[TARGET, :, :9] = 1; k[TARGET, :, 9:] = 3
        v[TARGET, :, :9] = 2; v[TARGET, :, 9:] = 4
        return DynamicCache(((k, v),))
    cache = make_cache()
    baseline = cache_digest(cache)
    with torch.inference_mode():
        for axes in (("keys", "values"), ("keys",), ("values",)):
            with verify_restored(cache, baseline, width=18):
                with graft(cache, axes, source=(0, 9), dest=(9, 18), width=18):
                    for axis in ("keys", "values"):
                        expected = 1 if axis == "keys" else 2
                        untouched = 3 if axis == "keys" else 4
                        require(bool(torch.all(getattr(cache.layers[0], axis)[TARGET, :, 9:] == (expected if axis in axes else untouched))),
                                "wrong K/V selection or source block")
                    require(bool(torch.all(cache.layers[0].keys[0] == 0)), "companion changed")
                    cache.update(torch.zeros(4, 2, 6, 3), torch.zeros(4, 2, 6, 3), 0)
        try:
            with graft(cache, ("queries",), source=(0, 9), dest=(9, 18), width=18):
                pass
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported K/V axis admitted")
        cache.layers[0].keys[0, 0, 0, 0] += 1
        try:
            with verify_restored(cache, baseline, width=18):
                pass
        except ValueError:
            pass
        else:
            raise AssertionError("companion contamination escaped restoration check")
    native_mask = torch.ones(4, FULL_WIDTH, dtype=torch.long)
    changed = mask_added_row(native_mask)
    require(torch.nonzero(changed != native_mask, as_tuple=False).tolist() == [[TARGET, i] for i in range(*DEST)], "M wrong target")
    native_actual = torch.ones(4, 1, SUFFIX, FULL_WIDTH, dtype=torch.bool)
    masked_actual = native_actual.clone()
    masked_actual[TARGET, 0, :, DEST[0]:DEST[1]] = False
    old_float_predicate = bool(torch.all(masked_actual[TARGET, 0, :, DEST[0]:DEST[1]] < -1e20)) and bool(torch.all(native_actual[TARGET, 0, :, DEST[0]:DEST[1]] == 0))
    require(not old_float_predicate and check_actual_mask(masked_actual, native_actual, "M") == SUFFIX * 9,
            "boolean SDPA mask repair has no sensitivity")
    wrong_index = masked_actual.clone()
    wrong_index[TARGET, 0, 0, DEST[0]] = True
    wrong_index[TARGET, 0, 0, DEST[0] - 1] = False
    wrong_companion = masked_actual.clone()
    wrong_companion[0, 0, 0, DEST[0]] = False
    for wrong in (native_actual, wrong_index, wrong_companion):
        try:
            check_actual_mask(wrong, native_actual, "M")
        except ValueError:
            pass
        else:
            raise AssertionError("unmasked or companion-contaminated M mask admitted")


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA required")
    require(not OUT.exists(), "attempt path already exists")
    failed = json.loads((ATTEMPT1 / "receipt.json").read_text())
    failed_manifest_path = ATTEMPT1 / "source-to-cell.json"
    failed_manifest = json.loads(failed_manifest_path.read_text())
    require(failed["status"] == "technical_invalid" and failed["model_forwards"] == 3 and failed["vision_forwards"] == 1,
            "previous technical attempt/cumulative budget changed")
    require(json.loads(STAGE_B_ACCEPTANCE.read_text())["status"] == "lead-accepted-stage-B", "Stage B not accepted")
    primary_result_path, primary_manifest_path = PRIMARY / "result.json", PRIMARY / "source-to-cell.json"
    primary_result, primary_manifest = json.loads(primary_result_path.read_text()), json.loads(primary_manifest_path.read_text())
    references = {"N": PRIMARY / "L_L.pt", "M": PRIMARY / "E_L.pt"}
    require(all(literal_binding(path) == primary_result["cells"][{"N": "L_L", "M": "E_L"}[name]]["logits_binding"]
                for name, path in references.items()), "accepted reference vector changed")
    chosen, raw, trace, receipt, panel, group, S, offsets, _ = source_and_rows()
    tokens = raw[TARGET]["token_ids"]
    require(offsets[1] == TARGET_OFFSET and tokens[783:792] == tokens[792:801] == ROW and tokens[801:807] == S,
            "native repeated row or S changed")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "started_unix": time.time(), "source": {"attempt1_failure": literal_binding(ATTEMPT1 / "receipt.json"),
             "attempt1_manifest": literal_binding(failed_manifest_path), "attempt1_producer_capture": failed_manifest["producer_capture"],
             "stage_B_acceptance": literal_binding(STAGE_B_ACCEPTANCE),
             "primary_result": literal_binding(primary_result_path), "primary_manifest": literal_binding(primary_manifest_path),
             "reference_vectors": {name: literal_binding(path) for name, path in references.items()},
             "selection": primary_manifest["source"]["selection"], **{key: chosen[key] for key in ("raw", "trace", "runtime_receipt", "image")}}}
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
        require(input_identity(batch) == receipt["input_identity"], "native batch identity changed")
        tails = _prefix_tokens(raw, TARGET_OFFSET, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, tails, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        blocks()
        require(native["input_ids"].shape == (4, FULL_WIDTH) and native["position_ids"].shape == (3, 4, FULL_WIDTH), "late replay shape changed")
        require(native["input_ids"][TARGET, SOURCE[0]:SOURCE[1]].tolist() == ROW and
                native["input_ids"][TARGET, DEST[0]:DEST[1]].tolist() == ROW and
                native["input_ids"][TARGET, WIDTH:FULL_WIDTH].tolist() == S, "physical row/S placement changed")
        require(all(tensor_hash(native[k]) == primary_manifest["cells"]["L_L"]["tensor_hashes"][k]
                    for k in ("input_ids", "attention_mask", "position_ids")), "late native replay inputs changed")
        masked = mask_added_row(native["attention_mask"])
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        producer_capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_native_row_mass-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ("probes/training_set_completion/recurrence_history_cache_partition.py",
                        "probes/training_set_completion/recurrence_donor_tracking.py", "probes/training_set_completion/recurrence_position_history.py",
                        "probes/training_set_completion/recurrence_written_content.py", "probes/training_set_completion/untied_shared.py",
                        "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py", "src/inference/bound_requests.py",
                        "src/qwen/untied_embeddings.py", inspect.getfile(cache_utils), inspect.getfile(modeling_qwen3_vl))]
        captures = [preserve_source(p, run_root=OUT, relative_name=str(p) if not p.is_absolute() else f"transformers/{p.name}") for p in dependencies]
        require(literal_binding(captures[8])["sha256"] == current_loader["sha256"], "loader source capture changed")
        manifest = {"schema": "recurrence_native_row_mass.cells.v1", "status": "frozen_before_forward", "source": state["source"],
                    "producer": literal_binding(Path(__file__)), "producer_capture": literal_binding(producer_capture),
                    "dependency_captures": [literal_binding(p) for p in captures], "transformers_version": transformers.__version__,
                    "model_identity": identity, "native_batch_identity": input_identity(batch), "source_panel": literal_binding(PANEL),
                    "target_batch": TARGET, "target_offset": TARGET_OFFSET, "current_S": S, "prefill_width": WIDTH,
                    "source_block": list(SOURCE), "destination_block": list(DEST),
                    "native_input_hashes": {k: tensor_hash(native[k]) for k in ("input_ids", "attention_mask", "position_ids")},
                    "M_mask_hash": tensor_hash(masked), "suffix_hashes": {k: tensor_hash(v) for k, v in
                    {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH], "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH],
                     "cache_position": torch.arange(WIDTH, FULL_WIDTH, device=device)}.items()},
                    "model_forward_cap": MAX_FORWARDS, "model_seconds_cap": MAX_SECONDS, "parity_atol": ATOL}
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=literal_binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        clock = time.monotonic()
        def count(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(failed["model_forwards"] + state["model_forwards"] <= MAX_FORWARDS and
                    failed["elapsed_seconds"] + time.monotonic() - clock <= MAX_SECONDS,
                    "cumulative forward/time cap exceeded")
        def vision(*_):
            state["vision_forwards"] += 1
            require(state["vision_forwards"] <= 1, "unexpected image pass during S")
        counter_handles = [q.model.register_forward_pre_hook(count, with_kwargs=True), q.model.model.visual.register_forward_pre_hook(vision)]
        cells = {}
        try:
            with torch.inference_mode():
                cache = DynamicCache()
                prefill = dict(native)
                prefill.update(input_ids=native["input_ids"][:, :WIDTH], attention_mask=native["attention_mask"][:, :WIDTH],
                               position_ids=native["position_ids"][:, :, :WIDTH], cache_position=torch.arange(WIDTH, device=device),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                prefill_seen, prefill_hooks = hooks(q.model, prefill["input_ids"], prefill["position_ids"])
                try:
                    output = q.model(**prefill)
                finally:
                    for handle in prefill_hooks:
                        handle.remove()
                require(output.past_key_values is cache and state["vision_forwards"] == 1, "native prefill/vision identity changed")
                check_cache(cache, width=WIDTH)
                require(len(prefill_seen["embedding_inputs"]) == len(prefill_seen["rotary_positions"]) == 1 and
                        len(prefill_seen["masks"]) == len(prefill_seen["cache_slots"]) == 2, "prefill consumption invalid")
                baseline_digest = cache_digest(cache)
                native_blocks = block_hashes(cache)
                write(OUT / "prefill-readback.json", {"consumed": prefill_seen, "cache_digest": baseline_digest,
                                                       "block_hashes": native_blocks, "cache_length": cache.get_seq_length()})
                state["prefill_readback"] = literal_binding(OUT / "prefill-readback.json")
                write(OUT / "receipt.json", state)
                cache_position = torch.arange(WIDTH, FULL_WIDTH, device=device)
                native_actual_mask = None
                for name, axes in (("N", ()), ("M", ()), ("B", ("keys", "values")), ("K", ("keys",)), ("V", ("values",))):
                    actual_2d_mask = masked if name == "M" else native["attention_mask"]
                    suffix = {"input_ids": native["input_ids"][:, WIDTH:FULL_WIDTH], "attention_mask": actual_2d_mask,
                              "position_ids": native["position_ids"][:, :, WIDTH:FULL_WIDTH], "cache_position": cache_position,
                              "past_key_values": cache, "use_cache": True, "return_dict": True, "logits_to_keep": SUFFIX}
                    with verify_restored(cache, baseline_digest):
                        with graft(cache, axes) if axes else _no_graft():
                            expected_blocks = block_hashes(cache)
                            expected_source = native_blocks
                            for i in range(LAYERS):
                                require(expected_blocks[i]["source"] == expected_source[i]["source"], "source block changed")
                                for axis in ("keys", "values"):
                                    expected = native_blocks[i]["source"][axis] if axis in axes else native_blocks[i]["dest"][axis]
                                    require(expected_blocks[i]["dest"][axis] == expected, "wrong K/V block selected")
                            consumed, observers = hooks(q.model, suffix["input_ids"], suffix["position_ids"])
                            layer_seen, layer_observers, actual_mask = layer_hooks(q.model, cache, cache_position, native_actual_mask,
                                                                                    "M" if name == "M" else "native")
                            try:
                                output = q.model(**suffix)
                                logits = output.logits[TARGET, -1].detach().float().cpu()
                            finally:
                                for handle in observers + layer_observers:
                                    handle.remove()
                            if name == "N":
                                require(len(actual_mask) == 1, "native actual mask was not captured")
                                native_actual_mask = actual_mask[0]
                            require(torch.isfinite(logits).all() and cache.get_seq_length() == FULL_WIDTH and state["vision_forwards"] == 1,
                                    "S output/cache/vision invalid")
                            require(set(layer_seen) == set(range(LAYERS)), "not all decoder caches consumed")
                            require(len(consumed["embedding_inputs"]) == len(consumed["rotary_positions"]) == 1 and
                                    len(consumed["masks"]) == len(consumed["cache_slots"]) == 2, "S token/position/mask consumption invalid")
                            for i in range(LAYERS):
                                require(layer_seen[i]["source"] == expected_blocks[i]["source"] and layer_seen[i]["dest"] == expected_blocks[i]["dest"],
                                        "attention did not consume declared K/V blocks")
                            if name != "N":
                                for field in ("embedding_inputs", "rotary_positions", "cache_slots"):
                                    require(consumed[field] == cells["N"]["consumed"][field], f"{name} changed consumed {field}")
                                if name != "M":
                                    require(consumed["masks"] == cells["N"]["consumed"]["masks"], "graft changed causal mask")
                            top = torch.topk(logits, 5)
                            path = OUT / f"{name}.pt"
                            torch.save(logits, path)
                            cells[name] = {"full_logits": literal_binding(path), "argmax_token": int(top.indices[0]),
                                           "top1_top2_gap": float(top.values[0] - top.values[1]),
                                           "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices, top.values, strict=True)],
                                           "z38_minus_z999": float(logits[151708] - logits[152669]),
                                           "absolute_coordinates": probability_readback(logits, (0, 38, 640, 999)),
                                           "consumed": consumed, "consumed_cache_layers": layer_seen,
                                           "cache_length_after": cache.get_seq_length(), "actuation_axes": list(axes)}
                            state["last_cell"] = name
                            write(OUT / "partial-results.json", {"status": "running", "cells": cells})
                            write(OUT / "receipt.json", state)
                    cells[name]["native_cache_restored_sha256"] = hashlib.sha256(json.dumps(baseline_digest, sort_keys=True).encode()).hexdigest()
                    if name in references:
                        accepted = torch.load(references[name], map_location="cpu", weights_only=True)
                        cells[name]["max_abs_vs_accepted_full_vocab"] = float((logits - accepted).abs().max())
                        expected_argmax = primary_result["cells"][{"N": "L_L", "M": "E_L"}[name]]["argmax_token"]
                        require(cells[name]["max_abs_vs_accepted_full_vocab"] <= ATOL and cells[name]["argmax_token"] == expected_argmax,
                                f"{name} full-vector qualification failed")
                require(state["model_forwards"] == 6 and state["vision_forwards"] == 1 and
                        failed["elapsed_seconds"] + time.monotonic() - clock <= MAX_SECONDS,
                        "frozen forward count/time changed")
                d = {name: cell["z38_minus_z999"] for name, cell in cells.items()}
                contrasts = {"B_minus_N": d["B"] - d["N"], "M_to_B": d["B"] - d["M"], "M_to_N": d["N"] - d["M"],
                             "K_effect_native_V": d["K"] - d["N"], "K_effect_old_V": d["B"] - d["V"],
                             "V_effect_native_K": d["V"] - d["N"], "V_effect_old_K": d["B"] - d["K"],
                             "KV_interaction": d["B"] - d["K"] - d["V"] + d["N"],
                             "B_reproduces_999_winner": cells["B"]["argmax_token"] == 152669 and cells["B"]["top1_top2_gap"] > 0.001}
                result = {"schema": "recurrence_native_row_mass.result.v1", "status": "candidate", "cells": cells,
                          "contrasts": contrasts, "numerical_guard": 0.001, "prefill_readback": literal_binding(OUT / "prefill-readback.json"),
                          "source_manifest": literal_binding(OUT / "source-to-cell.json"), "model_forwards": state["model_forwards"],
                          "vision_forwards": state["vision_forwards"], "model_seconds": time.monotonic() - clock}
                write(OUT / "result.json", result)
                state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"), model_seconds=result["model_seconds"],
                             elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                write(OUT / "receipt.json", state)
                print(json.dumps({"status": state["status"], "result": str(OUT / "result.json"), "forwards": state["model_forwards"]}))
        finally:
            for handle in counter_handles:
                handle.remove()
    except BaseException as error:
        state.update(status="technical_invalid", error=repr(error), elapsed_seconds=time.monotonic() - started)
        write(OUT / "receipt.json", state)
        raise


@contextmanager
def _no_graft():
    yield


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); print("selfcheck ok")
    else:
        run(args.device)
