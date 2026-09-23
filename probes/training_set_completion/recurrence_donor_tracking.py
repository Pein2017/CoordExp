"""Bounded donor-coordinate response readback at native row88 x2."""
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
from probes.training_set_completion.recurrence_position_history import PANEL, TARGET, source_and_rows
from probes.training_set_completion.recurrence_written_content import hooks
from probes.training_set_completion.untied_shared import load_model
from src.artifacts.source_provenance import preserve_source
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs, prepare_native_inputs


OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-donor-tracking/attempt-001")
PRIMARY = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/attempt-002")
PREVIOUS = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-written-content")
ROW = [151646, 8987, 151647, 151648, 151670, 152241, 151708, 152245, 151649]
DONORS = (640, 832)
MAX_FORWARDS = 8
MAX_SECONDS = 900
ATOL = 2e-4
GUARD = 1e-3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def edit_one(native, S, axis, donor):
    require(axis in ("x2", "y2") and donor in DONORS, "unsupported singleton donor")
    ids = native["input_ids"]
    n = len(S)
    require(ids.ndim == 2 and ids.shape[0] == 4 and n == 6, "native batch/S length changed")
    require(ids[TARGET, -n:].tolist() == S, "current S differs")
    require(ids[TARGET, -(n + 9):-n].tolist() == ROW, "preceding row not exact native template")
    row_slot = 6 if axis == "x2" else 7
    raw_offset = 783 + row_slot
    physical = ids.shape[1] - n - 9 + row_slot
    replacement = 151670 + donor
    require(int(ids[TARGET, physical]) == ROW[row_slot], "singleton source slot mismatch")
    changed = ids.clone()
    changed[TARGET, physical] = replacement
    indices = torch.nonzero(changed != ids, as_tuple=False).tolist()
    require(indices == [[TARGET, physical]], "mutation changed more than one token")
    require(changed[TARGET, -n:].tolist() == S and changed.shape == ids.shape, "mutation changed S/length")
    result = dict(native)
    result["input_ids"] = changed
    require(result["position_ids"] is native["position_ids"] and result["attention_mask"] is native["attention_mask"], "mutation changed positions/mask")
    return result, {"axis": axis, "donor": donor, "raw_offset": raw_offset, "physical_index": [TARGET, physical],
                    "old_token": ROW[row_slot], "new_token": replacement}


def logprob(logits):
    value = logits.double()
    return value - torch.logsumexp(value, dim=0)


def response(base, variant):
    baseline, edited = logprob(base), logprob(variant)
    F = (edited - baseline)[151670:152670]
    dz = (variant.double() - base.double())[151670:152670]
    centered = dz - dz.mean()
    top = torch.topk(centered, 12)
    return F, centered, [{"bin": int(i), "centered_logit_delta": float(v)} for i, v in zip(top.indices, top.values, strict=True)]


def donor_metrics(F, centered, donor):
    v = donor
    score = float(F[v])
    return {"donor": v, "F_v": score, "abs_F_v": abs(score),
            "curvature": float(F[v] - (F[v - 1] + F[v + 1]) / 2),
            "strict_neighbor_margin": float(torch.minimum(F[v] - F[v - 1], F[v] - F[v + 1])),
            "F_rank": int((F > F[v]).sum()) + 1,
            "centered_logit_delta": float(centered[v]), "centered_logit_delta_rank": int((centered > centered[v]).sum()) + 1,
            "localized_absolute_amplification": score > GUARD and float(torch.minimum(F[v] - F[v - 1], F[v] - F[v + 1])) > GUARD}


def probability_readback(logits, bins):
    lp = logprob(logits)
    return {str(v): {"token": 151670 + v, "logprob": float(lp[151670 + v]), "probability": float(lp[151670 + v].exp())} for v in sorted(bins)}


def prior_readback():
    pair, split = PREVIOUS / "attempt-001", PREVIOUS / "coordinate-split-001"
    acceptance = PREVIOUS / "lead-acceptance.json"
    require(json.loads(acceptance.read_text())["status"] == "lead-accepted", "prior response acceptance missing")
    pair_result = json.loads((pair / "result.json").read_text())
    split_result = json.loads((split / "result.json").read_text())
    paths = {"R": split / "R.pt", "X": split / "X.pt", "Y": split / "Y.pt", "B": pair / "B.pt"}
    data = {}
    for name, path in paths.items():
        expected = pair_result if name == "B" else split_result
        require(literal_binding(path) == expected["cells"][name]["full_logits"], f"accepted prior {name} vector drifted")
        logits = torch.load(path, map_location="cpu", weights_only=True)
        read = probability_readback(logits, (38, 999))
        data[name] = {"binding": literal_binding(path), "p38": read["38"]["probability"], "p999": read["999"]["probability"],
                      "log_odds_999_vs_38": read["999"]["logprob"] - read["38"]["logprob"]}
    paradox = data["Y"]["log_odds_999_vs_38"] > data["R"]["log_odds_999_vs_38"] and data["Y"]["p999"] < data["R"]["p999"]
    require(paradox, "accepted prior odds/probability readback changed")
    return {"acceptance": literal_binding(acceptance), "pair_result": literal_binding(pair / "result.json"),
            "split_result": literal_binding(split / "result.json"), "cells": data,
            "Y_improves_999_vs_38_odds_while_p999_falls": paradox}


def selfcheck():
    sample = {"input_ids": torch.tensor([[0] * 15, [0] * 15, ROW + [1, 2, 3, 4, 5, 6], [0] * 15]),
              "attention_mask": torch.ones(4, 15, dtype=torch.long), "position_ids": torch.arange(15).expand(3, 4, 15).clone()}
    S = [1, 2, 3, 4, 5, 6]
    for axis, expected in (("x2", 6), ("y2", 7)):
        edited, change = edit_one(sample, S, axis, 640)
        require(change["physical_index"] == [2, expected] and edited["input_ids"][2, expected] == 152310, "singleton offset selfcheck")
    for wrong in (S[:-1], S[:-1] + [7]):
        try:
            edit_one(sample, wrong, "x2", 640)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong S/length admitted")
    a = torch.zeros(152670, dtype=torch.float64)
    uniform = a + 1.0
    Fu, _, _ = response(a, uniform)
    require(float(Fu.abs().max()) < 1e-9, "uniform shift made a donor peak")
    linear = a.clone(); linear[151670:152670] = torch.arange(1000, dtype=torch.float64) * 1e-3
    Fl, _, _ = response(a, linear)
    require(abs(donor_metrics(Fl, Fl, 640)["curvature"]) < 1e-9 and donor_metrics(Fl, Fl, 640)["strict_neighbor_margin"] < 0, "linear shift made a donor spike")
    spike = a.clone(); spike[151670 + 640] = 2.0
    Fs, cs, _ = response(a, spike)
    require(donor_metrics(Fs, cs, 640)["curvature"] > 1.0 and donor_metrics(Fs, cs, 640)["strict_neighbor_margin"] > 1.0, "exact donor spike missed")
    b = a.clone(); b[151670 + 999] = 0.1; b[151670 + 38] = -1.0; b[151670] = 15.0
    pa, pb = probability_readback(a, (38, 999)), probability_readback(b, (38, 999))
    require(pb["999"]["probability"] < pa["999"]["probability"] and pb["999"]["logprob"] - pb["38"]["logprob"] > pa["999"]["logprob"] - pa["38"]["logprob"], "odds/probability distinction lost")


def run(device):
    selfcheck()
    require(torch.cuda.is_available() and device.startswith("cuda"), "CUDA required")
    require(not OUT.exists(), "attempt path already exists")
    chosen, raw, trace, receipt, panel, group, S, offsets, _ = source_and_rows()
    token_stream = raw[TARGET]["token_ids"]
    require(offsets[0] == 798 and token_stream[783:792] == ROW and token_stream[792:798] == S, "row88 source alignment changed")
    require(all(151670 + donor not in token_stream[:792] for donor in DONORS), "donor occurred before target row")
    reference_path = PRIMARY / "E_E.pt"
    primary_manifest_path = PRIMARY / "source-to-cell.json"
    primary_result_path = PRIMARY / "result.json"
    primary = json.loads(primary_result_path.read_text())
    require(literal_binding(reference_path) == primary["cells"]["E_E"]["logits_binding"], "accepted early vector drifted")
    primary_manifest = json.loads(primary_manifest_path.read_text())
    previous = prior_readback()
    OUT.mkdir(parents=True)
    write(OUT / "prior-response.json", previous)
    state = {"status": "preparing", "pid": os.getpid(), "device": device, "model_forwards": 0, "vision_forwards": 0,
             "source": {"primary_manifest": literal_binding(primary_manifest_path), "primary_vector": literal_binding(reference_path),
                        "primary_result": literal_binding(primary_result_path), "prior_response": literal_binding(OUT / "prior-response.json"),
                        "selection": primary_manifest["source"]["selection"], **{key: chosen[key] for key in ("raw", "trace", "runtime_receipt", "image")}},
             "started_unix": time.time()}
    write(OUT / "receipt.json", state)
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", torch.device(device))
        current, saved = dict(identity), dict(receipt["identity"])
        current_loader, saved_loader = current.pop("loader_source"), saved.pop("loader_source")
        require(current == saved and (current_loader["sha256"], current_loader["size_bytes"]) == (saved_loader["sha256"], saved_loader["size_bytes"]), "model/effective-row identity differs")
        config = dict(panel["configs"]["untied"])
        config["data"] = {"input_jsonl": group["input_jsonl"]}
        requests, _ = build_bound_native_requests(q, config, group["cases"])
        batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
        require(input_identity(batch) == receipt["input_identity"], "native batch identity differs")
        end = offsets[0]
        suffixes = _prefix_tokens(raw, end, int(q.tokenizer.pad_token_id))
        histories = [list(prompt) + tail for prompt, tail in zip(batch.prompt_token_ids, suffixes, strict=True)]
        native = exact_history_inputs(q.model, batch.inputs, histories, pad_token_id=int(q.tokenizer.pad_token_id), logits_to_keep=1)
        require(all(tensor_hash(native[k]) == primary_manifest["cells"]["E_E"]["tensor_hashes"][k] for k in ("input_ids", "attention_mask", "position_ids")), "native early replay inputs differ")
        variants = {f"{label}{donor}": edit_one(native, S, axis, donor) for label, axis in (("X", "x2"), ("Y", "y2")) for donor in DONORS}
        producer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        source_capture = preserve_source(Path(__file__), run_root=OUT, relative_name=f"recurrence_donor_tracking-{producer_hash[:12]}.py")
        dependencies = [Path(p) for p in ("probes/training_set_completion/recurrence_position_history.py", "probes/training_set_completion/recurrence_written_content.py",
                        "probes/training_set_completion/untied_shared.py", "probes/training_set_completion/numerical_feedback/runtime.py", "src/qwen/native.py",
                        "src/inference/bound_requests.py", "src/qwen/untied_embeddings.py")]
        captures = [str(preserve_source(p, run_root=OUT, relative_name=str(p))) for p in dependencies]
        require(literal_binding(Path(captures[-1]))["sha256"] == current_loader["sha256"], "loader source capture differs")
        manifest = {"schema": "recurrence_donor_tracking.cells.v1", "status": "frozen_before_forward", "source": state["source"],
                    "producer": literal_binding(Path(__file__)), "producer_capture": literal_binding(source_capture), "dependency_captures": [literal_binding(Path(p)) for p in captures],
                    "model_identity": identity, "native_batch_identity": input_identity(batch), "source_panel": literal_binding(PANEL),
                    "target_batch": TARGET, "source_row": 87, "target_row": 88, "target_offset": end, "current_S": S, "donors": DONORS,
                    "cells": {name: {"change": info, "tensor_hashes": {k: tensor_hash(inputs[k]) for k in ("input_ids", "attention_mask", "position_ids")}}
                              for name, (inputs, info) in variants.items()},
                    "R_tensor_hashes": {k: tensor_hash(native[k]) for k in ("input_ids", "attention_mask", "position_ids")},
                    "max_forwards": MAX_FORWARDS, "max_model_seconds": MAX_SECONDS, "parity_atol": ATOL, "numerical_guard": GUARD}
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
        logits_by_cell = {}
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
                require(seen["masks"][0] == seen["masks"][1] and seen["cache_slots"][0] == seen["cache_slots"][1], "decoder causal consumption differs")
                top = torch.topk(logits, 5)
                path = OUT / f"{name}.pt"
                torch.save(logits, path)
                logits_by_cell[name] = logits
                cells[name] = {"full_logits": literal_binding(path), "argmax_token": int(top.indices[0]), "top1_top2_gap": float(top.values[0] - top.values[1]),
                               "top5": [{"token": int(t), "logit": float(v)} for t, v in zip(top.indices, top.values, strict=True)], "consumed": seen}
                if name == "R":
                    accepted = torch.load(reference_path, map_location="cpu", weights_only=True)
                    cells[name]["max_abs_vs_primary_full_vocab"] = float((logits - accepted).abs().max())
                    saved = trace["steps"][end]
                    topids = (saved["raw_winners"][TARGET], saved["raw_runnerups"][TARGET])
                    topvals = saved["raw_top2"][TARGET]
                    cells[name]["saved_top2_max_abs_error"] = max(abs(float(logits[t]) - float(v)) for t, v in zip(topids, topvals, strict=True))
                    require(cells[name]["max_abs_vs_primary_full_vocab"] <= ATOL and cells[name]["saved_top2_max_abs_error"] <= ATOL and cells[name]["argmax_token"] == 151708, "native early full-vector parity failed")
                else:
                    for field in ("rotary_positions", "masks", "cache_slots"):
                        require(seen[field] == cells["R"]["consumed"][field], f"{name} changed consumed {field}")
                state["last_cell"] = name
                write(OUT / "partial-results.json", {"status": "running", "cells": cells})
                write(OUT / "receipt.json", state)
                require(time.monotonic() - clock <= MAX_SECONDS, "model execution time cap exceeded")
        finally:
            for handle in handles:
                handle.remove()
        # Readout is CPU-only and cannot change any model state.
        bins = {0, 38, 999, 640, 832, *range(636, 645), *range(828, 837)}
        coordinate_responses = {}
        for name, logits in logits_by_cell.items():
            cells[name]["absolute_coordinates"] = probability_readback(logits, bins)
            if name == "R":
                continue
            F, centered, peaks = response(logits_by_cell["R"], logits)
            donor = int(name[1:])
            cells[name]["donor_metrics"] = donor_metrics(F, centered, donor)
            cells[name]["centered_delta_peaks"] = peaks
            topF = torch.topk(F, 12)
            cells[name]["response_peaks"] = [{"bin": int(i), "F": float(v)} for i, v in zip(topF.indices, topF.values, strict=True)]
            coordinate_responses[name] = [float(x) for x in F]
        write(OUT / "coordinate-responses.json", {"schema": "recurrence_donor_tracking.responses.v1", "F_logprob_delta": coordinate_responses})
        transfer = {}
        for axis in ("X", "Y"):
            f640, f832 = coordinate_responses[f"{axis}640"], coordinate_responses[f"{axis}832"]
            d640, d832 = f640[640] - f832[640], f832[832] - f640[832]
            transfer[axis] = {"D640": d640, "D832": d832, "T": d640 + d832,
                              "two_donor_transfer": all((cells[f"{axis}{v}"]["donor_metrics"]["localized_absolute_amplification"] for v in DONORS)) and d640 > GUARD and d832 > GUARD}
        result = {"schema": "recurrence_donor_tracking.result.v1", "status": "candidate", "cells": cells,
                  "responses": literal_binding(OUT / "coordinate-responses.json"), "prior_response": literal_binding(OUT / "prior-response.json"),
                  "transfer": transfer, "cross_axis_localized_transfer": transfer["X"]["two_donor_transfer"] and transfer["Y"]["two_donor_transfer"],
                  "model_forwards": state["model_forwards"], "vision_forwards": state["vision_forwards"], "model_seconds": time.monotonic() - clock}
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
