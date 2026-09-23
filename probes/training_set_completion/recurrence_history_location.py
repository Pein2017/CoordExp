"""Move one written x2 donor across four identical native history rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.numerical_feedback.runtime import _prefix_tokens
from probes.training_set_completion.recurrence_census.prepare import _rows_from_tokens
from probes.training_set_completion.recurrence_donor_tracking import ROW, donor_metrics, probability_readback, require, response, write
from probes.training_set_completion.recurrence_position_history import PANEL, TARGET, source_and_rows
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/attempt-001")
PRIMARY = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/attempt-002")
DONOR = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-donor-tracking")
ROWS = (87, 86, 72, 27)
TARGET_OFFSET = 798
DONOR_VALUE = 640
MAX_FORWARDS = 8
MAX_SECONDS = 900
ATOL = 2e-4


def edit_location(native, S, raw_tokens, source_start, target_end=TARGET_OFFSET):
    ids = native["input_ids"]
    require(ids.ndim == 2 and ids.shape[0] == 4 and len(S) == 6, "native batch/current S length changed")
    require(ids[TARGET, -len(S):].tolist() == S and raw_tokens[target_end - len(S):target_end] == S, "current S changed")
    require(raw_tokens[source_start:source_start + 9] == ROW and source_start + 9 <= target_end - len(S), "source is not an earlier exact row")
    raw_offset = source_start + 6
    physical = ids.shape[1] - (target_end - raw_offset)
    require(int(ids[TARGET, physical]) == ROW[6], "source x2 physical slot changed")
    edited = ids.clone()
    edited[TARGET, physical] = 151670 + DONOR_VALUE
    require(torch.nonzero(edited != ids, as_tuple=False).tolist() == [[TARGET, physical]], "edit changed more than one token")
    require(edited[TARGET, -len(S):].tolist() == S and edited.shape == ids.shape, "edit changed S/length")
    result = dict(native)
    result["input_ids"] = edited
    require(result["attention_mask"] is native["attention_mask"] and result["position_ids"] is native["position_ids"], "edit changed mask/position")
    return result, {"source_row_start": source_start, "raw_offset": raw_offset, "physical_index": [TARGET, physical],
                    "old_token": ROW[6], "new_token": 151670 + DONOR_VALUE}


def check_variant_multisets(native, variants):
    expected = native["input_ids"][TARGET].tolist()
    expected.remove(ROW[6])
    expected.append(151670 + DONOR_VALUE)
    for inputs in variants:
        require(sorted(inputs["input_ids"][TARGET].tolist()) == sorted(expected), "mutated token multiset mismatch")


def selfcheck():
    S = [1, 2, 3, 4, 5, 6]
    raw = ROW + S
    base = {"input_ids": torch.tensor([[0] * 15, [0] * 15, raw, [0] * 15]),
            "attention_mask": torch.ones(4, 15, dtype=torch.long), "position_ids": torch.arange(15).expand(3, 4, 15).clone()}
    edited, info = edit_location(base, S, raw, 0, 15)
    require(info["raw_offset"] == 6 and info["physical_index"] == [2, 6] and edited["input_ids"][2, 6] == 152310, "location selfcheck")
    for bad_start, bad_S in ((1, S), (0, S[:-1]), (0, S[:-1] + [7])):
        try:
            edit_location(base, bad_S, raw, bad_start, 15)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong source offset or S admitted")
    require(sorted(edited["input_ids"][2].tolist()) == sorted(ROW[:6] + [152310] + ROW[7:] + S), "singleton multiset selfcheck")
    check_variant_multisets(base, (edited,))
    corrupted = dict(edited)
    corrupted["input_ids"] = edited["input_ids"].clone()
    corrupted["input_ids"][TARGET, 0] += 1
    try:
        check_variant_multisets(base, (edited, corrupted))
    except ValueError:
        pass
    else:
        raise AssertionError("unequal edited-arm multiset admitted")


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA required")
    require(not OUT.exists(), "attempt path already exists")
    chosen, raw, trace, receipt, panel, group, S, offsets, _ = source_and_rows()
    tokens = raw[TARGET]["token_ids"]
    parsed = _rows_from_tokens(tokens)
    require(offsets[0] == TARGET_OFFSET and all(tokens[parsed[j]["start"]:parsed[j]["end"]] == ROW for j in ROWS), "four source rows changed")
    require([parsed[j]["start"] for j in ROWS] == [783, 774, 648, 243], "frozen source locations changed")
    require(151670 + DONOR_VALUE not in tokens[:792], "donor already in preceding native output")
    primary_result_path, primary_manifest_path, reference_path = PRIMARY / "result.json", PRIMARY / "source-to-cell.json", PRIMARY / "E_E.pt"
    primary_result = json.loads(primary_result_path.read_text())
    primary_manifest = json.loads(primary_manifest_path.read_text())
    require(literal_binding(reference_path) == primary_result["cells"]["E_E"]["logits_binding"], "primary native vector drifted")
    donor_result_path, donor_vector_path, donor_acceptance_path = DONOR / "attempt-001/result.json", DONOR / "attempt-001/X640.pt", DONOR / "lead-acceptance.json"
    donor_result = json.loads(donor_result_path.read_text())
    require(json.loads(donor_acceptance_path.read_text())["status"] == "lead-accepted" and literal_binding(donor_vector_path) == donor_result["cells"]["X640"]["full_logits"], "accepted row87 donor vector drifted")
    OUT.mkdir(parents=True)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "source": {"primary_result": literal_binding(primary_result_path), "primary_manifest": literal_binding(primary_manifest_path), "primary_vector": literal_binding(reference_path),
                        "donor_result": literal_binding(donor_result_path), "donor_vector": literal_binding(donor_vector_path), "donor_acceptance": literal_binding(donor_acceptance_path),
                        "selection": primary_manifest["source"]["selection"], **{key: chosen[key] for key in ("raw", "trace", "runtime_receipt", "image")}}, "started_unix": time.time()}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        current, saved = dict(identity), dict(receipt["identity"])
        current_loader, saved_loader = current.pop("loader_source"), saved.pop("loader_source")
        require(current == saved and (current_loader["sha256"], current_loader["size_bytes"]) == (saved_loader["sha256"], saved_loader["size_bytes"]), "source model/effective-row identity changed")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"], "native batch identity changed")
        suffixes = _prefix_tokens(raw, TARGET_OFFSET, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, suffixes, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        require(all(tensor_hash(native[k]) == primary_manifest["cells"]["E_E"]["tensor_hashes"][k] for k in ("input_ids", "attention_mask", "position_ids")), "native early replay inputs changed")
        variants = {f"L{j}": edit_location(native, S, tokens, parsed[j]["start"]) for j in ROWS}
        check_variant_multisets(native, (pair[0] for pair in variants.values()))
        for inputs, _ in variants.values():
            require(torch.equal(inputs["position_ids"], native["position_ids"]) and torch.equal(inputs["attention_mask"], native["attention_mask"]), "position/mask changed")
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        source_capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_history_location-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ("probes/training_set_completion/recurrence_donor_tracking.py", "probes/training_set_completion/recurrence_position_history.py",
                        "probes/training_set_completion/recurrence_written_content.py", "probes/training_set_completion/untied_shared.py",
                        "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py", "src/inference/bound_requests.py", "src/qwen/untied_embeddings.py")]
        captures = [str(preserve_source(p, run_root=OUT, relative_name=str(p))) for p in dependencies]
        require(literal_binding(Path(captures[-1]))["sha256"] == current_loader["sha256"], "loader source capture changed")
        manifest = {"schema": "recurrence_history_location.cells.v1", "status": "frozen_before_forward", "source": state["source"],
                    "producer": literal_binding(Path(__file__)), "producer_capture": literal_binding(source_capture), "dependency_captures": [literal_binding(Path(p)) for p in captures],
                    "model_identity": identity, "native_batch_identity": input_identity(batch), "source_panel": literal_binding(PANEL),
                    "target_batch": TARGET, "target_row": 88, "target_offset": TARGET_OFFSET, "current_S": S, "donor": DONOR_VALUE,
                    "R_tensor_hashes": {k: tensor_hash(native[k]) for k in ("input_ids", "attention_mask", "position_ids")},
                    "cells": {name: {"change": info, "tensor_hashes": {k: tensor_hash(inputs[k]) for k in ("input_ids", "attention_mask", "position_ids")}}
                              for name, (inputs, info) in variants.items()},
                    "max_forwards": MAX_FORWARDS, "max_model_seconds": MAX_SECONDS, "parity_atol": ATOL}
        write(OUT / "source-to-cell.json", manifest)
        state.update(status="executing", manifest=literal_binding(OUT / "source-to-cell.json"))
        write(OUT / "receipt.json", state)
        clock = time.monotonic()
        def count(_module, _args, _kwargs):
            state["model_forwards"] += 1
            require(state["model_forwards"] <= MAX_FORWARDS and time.monotonic() - clock <= MAX_SECONDS, "forward/time cap exceeded")
        def vision(*_):
            state["vision_forwards"] += 1
        handles = [q.model.register_forward_pre_hook(count, with_kwargs=True), q.model.model.visual.register_forward_pre_hook(vision)]
        cells = {}
        vectors = {}
        try:
            for name, inputs in (("R", native), *((name, pair[0]) for name, pair in variants.items())):
                seen, observers = hooks(q.model, inputs["input_ids"], inputs["position_ids"])
                try:
                    with torch.inference_mode():
                        logits = q.model(**inputs).logits[TARGET, -1].detach().float().cpu()
                finally:
                    for observer in observers:
                        observer.remove()
                require(torch.isfinite(logits).all() and len(seen["embedding_inputs"]) == len(seen["rotary_positions"]) == 1 and len(seen["masks"]) == 2, "actual consumption invalid")
                require(seen["masks"][0] == seen["masks"][1] and seen["cache_slots"][0] == seen["cache_slots"][1], "decoder mask/cache changed")
                top = torch.topk(logits, 5)
                path = OUT / f"{name}.pt"
                torch.save(logits, path)
                vectors[name] = logits
                cells[name] = {"full_logits": literal_binding(path), "argmax_token": int(top.indices[0]), "top1_top2_gap": float(top.values[0] - top.values[1]),
                               "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices, top.values, strict=True)], "consumed": seen}
                if name == "R":
                    accepted = torch.load(reference_path, map_location="cpu", weights_only=True)
                    cells[name]["max_abs_vs_primary_full_vocab"] = float((logits - accepted).abs().max())
                    saved_trace = trace["steps"][TARGET_OFFSET]
                    topids = (saved_trace["raw_winners"][TARGET], saved_trace["raw_runnerups"][TARGET])
                    topvals = saved_trace["raw_top2"][TARGET]
                    cells[name]["saved_top2_max_abs_error"] = max(abs(float(logits[t]) - float(v)) for t, v in zip(topids, topvals, strict=True))
                    require(cells[name]["max_abs_vs_primary_full_vocab"] <= ATOL and cells[name]["saved_top2_max_abs_error"] <= ATOL and cells[name]["argmax_token"] == 151708, "native full-vector qualification failed")
                else:
                    for field in ("rotary_positions", "masks", "cache_slots"):
                        require(seen[field] == cells["R"]["consumed"][field], f"{name} changed consumed {field}")
                    if name == "L87":
                        accepted = torch.load(donor_vector_path, map_location="cpu", weights_only=True)
                        cells[name]["max_abs_vs_accepted_X640"] = float((logits - accepted).abs().max())
                        require(cells[name]["max_abs_vs_accepted_X640"] <= ATOL and cells[name]["argmax_token"] == donor_result["cells"]["X640"]["argmax_token"], "accepted source-row87 parity failed")
                state["last_cell"] = name
                write(OUT / "partial-results.json", {"status": "running", "cells": cells})
                write(OUT / "receipt.json", state)
                require(time.monotonic() - clock <= MAX_SECONDS, "model execution time cap exceeded")
        finally:
            for handle in handles:
                handle.remove()
        bins = {0, 38, 999, DONOR_VALUE, *range(DONOR_VALUE - 4, DONOR_VALUE + 5)}
        responses = {}
        for name, logits in vectors.items():
            cells[name]["absolute_coordinates"] = probability_readback(logits, bins)
            cells[name]["z38_minus_z999"] = float(logits[151708] - logits[152669])
            if name == "R":
                continue
            F, centered, peaks = response(vectors["R"], logits)
            cells[name]["donor_metrics"] = donor_metrics(F, centered, DONOR_VALUE)
            cells[name]["centered_delta_peaks"] = peaks
            responses[name] = [float(x) for x in F]
        native_d = cells["R"]["z38_minus_z999"]
        near_d = cells["L87"]["z38_minus_z999"]
        contrasts = {name: {"delta_d_vs_R": cell["z38_minus_z999"] - native_d,
                            "delta_d_vs_L87": cell["z38_minus_z999"] - near_d,
                            "different_from_L87_above_guard": abs(cell["z38_minus_z999"] - near_d) > 0.001}
                     for name, cell in cells.items() if name != "R"}
        write(OUT / "coordinate-responses.json", {"schema": "recurrence_history_location.responses.v1", "F_logprob_delta": responses})
        result = {"schema": "recurrence_history_location.result.v1", "status": "candidate", "cells": cells,
                  "contrasts": contrasts, "numerical_guard": 0.001,
                  "responses": literal_binding(OUT / "coordinate-responses.json"), "model_forwards": state["model_forwards"],
                  "vision_forwards": state["vision_forwards"], "model_seconds": time.monotonic() - clock}
        write(OUT / "result.json", result)
        state.update(status="candidate_complete", result=literal_binding(OUT / "result.json"), model_seconds=result["model_seconds"],
                     elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        write(OUT / "receipt.json", state)
        print(json.dumps({"status": state["status"], "result": str(OUT / "result.json"), "forwards": state["model_forwards"]}))
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
