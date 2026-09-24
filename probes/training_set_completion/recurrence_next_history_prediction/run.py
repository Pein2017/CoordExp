"""Five qualified readouts for the frozen next-history forecast."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path

import torch

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.native_row_choice.runtime import _trace_compare
from probes.training_set_completion.numerical_feedback.select import rows, token_hash
from probes.training_set_completion.recurrence_chair_history_position.run import contract as prior_contract
from probes.training_set_completion.recurrence_first_arrivals.prepare import _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import _write_new
from probes.training_set_completion.recurrence_position_history import observed_hooks
from probes.training_set_completion.untied_shared import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-next-history-prediction"
PROTOCOL = UNIT / "unit.md"
MANIFEST = UNIT / "prediction-manifest.json"
PROTOCOL_SHA = "7ed5b1a9791c2144d1d79f910e903443f9ece5f623d5a15ccf2b7fd559aab851"
MANIFEST_SHA = "b4cfac3c89a7170f9ce6c3728c4bf2c88de9aa5f9ad2e1a2ad59db5454adc8ec"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001")
CAP_SECONDS = 900
TOL = 2e-4
HEADER = [151646, 34196, 151647, 151648]
IMPORTS = [
    "probes/training_set_completion/recurrence_next_history_prediction/run.py",
    "probes/training_set_completion/recurrence_chair_history_position/run.py",
    "probes/training_set_completion/artifacts.py",
    "probes/training_set_completion/coordinate_continuity/runtime.py",
    "probes/training_set_completion/native_row_choice/runtime.py",
    "probes/training_set_completion/numerical_feedback/select.py",
    "probes/training_set_completion/recurrence_first_arrivals/prepare.py",
    "probes/training_set_completion/recurrence_first_arrivals/stage1_case.py",
    "probes/training_set_completion/recurrence_position_history.py",
    "probes/training_set_completion/untied_shared.py",
    "src/artifacts/source_provenance.py",
    "src/qwen/input_identity.py", "src/qwen/native.py", "src/qwen/runtime_loading.py",
    "src/inference/bound_requests.py",
]


def bound(item: dict) -> Path:
    p = Path(item["path"])
    if not p.is_absolute():
        p = REPO / p
    actual = literal_binding(p)
    _require(actual["sha256"] == item["sha256"] and actual["size_bytes"] == item["size_bytes"],
             f"bound bytes changed: {p}")
    return p


def contract():
    _require(literal_binding(PROTOCOL)["sha256"] == PROTOCOL_SHA and
             literal_binding(MANIFEST)["sha256"] == MANIFEST_SHA, "lead protocol/manifest changed")
    m = json.loads(MANIFEST.read_text())
    _require(m["status"] == "lead-frozen-before-target-model-evaluation" and
             m["budget"]["new_model_forwards"] == m["budget"]["new_vision_forwards"] == 5 and
             m["budget"]["free_generation_tokens"] == 0 and
             m["budget"]["incremental_allocated_gpu_hours_ceiling"] == .25 and
             len(m["cells"]) == 5 and len(m["probes"]) == 2 and
             m["source_group"] == "refined-04" and m["batch_index"] == 0 and
             m["request_id"] == "coco2017_train_000000477415", "finite contract changed")
    for key in ("predecessor_acceptance", "predecessor_manifest", "predecessor_reduction"):
        bound(m[key])
    for key in ("raw", "trace", "runtime_receipt", "image"):
        bound(m["source_bindings"][key])
    pm, receipt, native, panel = prior_contract()
    _require(m["source_bindings"] == pm["source_bindings"] and
             m["repeated_A_tokens"] == native[36:45] == native[45:54] and
             native[:54] == native[:36] + m["repeated_A_tokens"] * 2 and
             rows(native)[6]["start"] == 54 and native[54:58] == m["shared_header_tokens"] == HEADER and
             native[58] == 151670 + m["target_native_x1"] == 151768 and
             native[59] == 151670 + m["target_native_y1"] == 152249 and
             m["target_first_free_y1_offset"] == 59,
             "U chronology or native y1/x1 slot changed")
    _require([(p["candidate_owner"], p["supplied_x1_bin"], p["supplied_x1_token"])
              for p in m["probes"]] == [(1589003, 510, 152180), (1586761, 418, 152088)],
             "fixed probes changed")
    expected = [("unforced_native_U_source_control", None, 6, 98),
                ("fresh_L_reference_full_vector", 1589003, 5, 510),
                ("held_out_U_y1", 1589003, 6, 510),
                ("fresh_L_reference_full_vector", 1586761, 5, 418),
                ("held_out_U_y1", 1586761, 6, 418)]
    _require([(c["mode"], c.get("candidate_owner"), c["row"], c["x1_bin"])
              for c in m["cells"]] == expected, "five cells changed")
    return m, receipt, native, panel


def boundary(m, native):
    s = m["source_bindings"]
    return {"group": "refined-04", "batch_index": 0, "image_id": 477415,
            "raw_path": s["raw"]["path"], "trace_path": s["trace"]["path"],
            "receipt_path": s["runtime_receipt"]["path"],
            "native_tokens": native, "native_token_hash": token_hash(native)}


def suffix(native, row, x1):
    _require(row in (5, 6) and x1 in (151768, 152180, 152088), "wrong row or x1")
    start = 45 if row == 5 else 54
    _require(native[start:start + 4] == HEADER, "wrong current header")
    return native[:start] + HEADER + [x1]


def forecasts(m, out):
    records = {}
    for p in m["probes"]:
        owner = p["candidate_owner"]
        z = {k: torch.load(bound(v), map_location="cpu", weights_only=True).to(torch.float64)
             for k, v in p["source_vectors"].items()}
        _require(all(v.ndim == 1 and v.shape == z["L"].shape and torch.isfinite(v).all()
                     for v in z.values()), "predecessor vocabulary incompatible")
        logits = {"persistence": z["L"], "affine_native_step": 2*z["L"]-z["E"],
                  "local_position_step": 2*z["L"]-z["LE"]}
        records[str(owner)] = {}
        for mode, values in logits.items():
            top = torch.topk(values, 2)
            frozen = p["frozen_forecasts"][mode]
            _require(top.indices.tolist() == frozen["top2_token_ids"] and
                     abs(float(top.values[0]-top.values[1])-frozen["top2_gap"]) < 1e-5,
                     f"CPU frozen forecast differs: {owner}/{mode}")
            logp = torch.log_softmax(values, -1)
            _require(torch.isfinite(logp).all() and abs(float(torch.logsumexp(logp, -1))) < 1e-10,
                     "invalid full-vocabulary log distribution")
            path = out / "forecasts" / f"{owner}-{mode}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            _require(not path.exists(), "forecast already exists")
            torch.save(logp, path)
            records[str(owner)][mode] = {"log_probability": literal_binding(path),
                                         "top2_token_ids": top.indices.tolist(),
                                         "top2_gap": float(top.values[0]-top.values[1]),
                                         "vocabulary_size": int(values.numel())}
    return records


def preflight(out):
    m, receipt, native, panel = contract()
    _require(not out.exists(), "attempt output already exists")
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    _require(q.model is None, "CPU preflight loaded model")
    batch, raw, trace, group, planning = _source(boundary(m, native), "untied", panel, q, torch.device("cpu"))
    _require(len(raw) == len(group["cases"]) == 1 and
             input_identity(batch) == receipt["input_identity"], "single-request identity changed")
    prompt = list(batch.prompt_token_ids[0])
    _require(len(prompt) == 1362 and len(prompt) == batch.inputs["input_ids"].shape[1],
             "source prompt width changed")
    checks = []
    for p in m["probes"]:
        owner, x1 = p["candidate_owner"], p["supplied_x1_token"]
        saved = json.loads((Path(p["source_vectors"]["L"]["path"]).parent / "consumer-raw.json").read_text())
        expected = prompt + suffix(native, 5, x1)
        _require(saved["input_ids"] == [expected] and saved["attention_mask"] == [[1]*len(expected)],
                 "accepted L actual consumer input differs")
        lpos = [axis[0][-5:] for axis in saved["position_ids"]]
        _require(lpos == [[432+i for i in range(5)]]*3, "accepted L position crosswalk differs")
        checks.append({"owner": owner, "L_consumer": literal_binding(Path(p["source_vectors"]["L"]["path"]).parent / "consumer-raw.json"),
                       "L_position_suffix": lpos, "U_expected_position_suffix": [[441+i for i in range(5)]]*3,
                       "L_physical_current": [1407, 1412], "U_physical_current": [1416, 1421]})
    native_input = prompt + suffix(native, 6, 151768)
    for p in m["probes"]:
        proposed = prompt + suffix(native, 6, p["supplied_x1_token"])
        _require(len(proposed) == 1421 and proposed[:-1] == native_input[:-1] and
                 proposed[-1] != native_input[-1] and proposed[1420] == p["supplied_x1_token"],
                 "U supplied input changes more than current x1")
    for wrong in ((4, 152180), (6, 152181)):
        try:
            suffix(native, *wrong)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("wrong row or content escaped CPU check")
    out.mkdir(parents=True)
    prediction = forecasts(m, out)
    captures = []
    for name in IMPORTS:
        source = REPO / name
        capture = preserve_source(source, run_root=out, relative_name=Path(name))
        captures.append({"maintained": literal_binding(source), "capture": literal_binding(capture)})
    old = json.loads(bound(m["predecessor_acceptance"]).read_text())
    prior = json.loads(bound(old["receipt"]).read_text())
    old_seconds = prior["cost"]["allocated_gpu_seconds"]
    shape_factor = max(1., 1421/1412)
    estimate = 2*old_seconds*5/18*shape_factor
    _require(estimate < CAP_SECONDS and
             m["budget"]["sequence_cumulative_prior_gpu_hours"] + estimate/3600 <
             m["budget"]["sequence_ceiling_gpu_hours"], "shape-aware cost forecast exceeds cap")
    packet = {"schema": "recurrence_next_history_prediction.preflight.v1", "status": "cpu_qualified_before_gpu",
              "protocol": literal_binding(PROTOCOL), "manifest": literal_binding(MANIFEST),
              "input_identity": input_identity(batch), "source_planning": planning,
              "shape": {"batch_size": 1, "prompt_width": len(prompt), "L_width": 1412, "U_width": 1421,
                        "image_grids": [list(x) for x in batch.image_grids],
                        "pixel_elements": int(batch.inputs["pixel_values"].numel())},
              "crosswalk": checks, "forecast_distributions": prediction,
              "cpu_checks": ["native_U_is_E_plus_two_exact_A", "row6_header_and_source_xy",
                             "accepted_L_consumer_and_positions", "U_single_x1_change",
                             "wrong_row_and_content_rejected"],
              "direct_source_captures": captures,
              "cost_forecast": {"prior_18_forward_seconds": old_seconds, "new_forwards": 5,
                                "shape_factor": shape_factor, "planning_margin": 2,
                                "estimated_seconds": estimate, "incremental_cap_seconds": CAP_SECONDS,
                                "sequence_prior_hours": m["budget"]["sequence_cumulative_prior_gpu_hours"]},
              "commands": {"gpu": ["python", "-B", "-m", "probes.training_set_completion.recurrence_next_history_prediction.run", "run", "--output", str(out), "--device", "cuda:0"],
                           "readback": ["python", "-B", "-m", "probes.training_set_completion.recurrence_next_history_prediction.run", "readback", "--output", str(out)]}}
    _write_new(out / "preflight.json", packet)
    print(json.dumps({"status": packet["status"], "forecast_seconds": estimate,
                      "six_forecasts": sum(map(len, prediction.values())), "shape": packet["shape"]}))


def make_input(model, batch, native, row, x1, pad):
    request = list(batch.prompt_token_ids[0]) + suffix(native, row, x1)
    inputs = exact_history_inputs(model, batch.inputs, [request], pad_token_id=pad, logits_to_keep=1)
    _require(inputs["input_ids"][0].tolist() == request and
             inputs["attention_mask"][0].tolist() == [1]*len(request), "model input changed")
    return inputs


def run(out, device_name):
    start = time.monotonic()
    m, source_receipt, native, panel = contract()
    pre_path = out / "preflight.json"
    pre = json.loads(pre_path.read_text())
    _require(pre["status"] == "cpu_qualified_before_gpu" and
             pre["protocol"] == literal_binding(PROTOCOL) and pre["manifest"] == literal_binding(MANIFEST) and
             device_name == "cuda:0" and not (out / "receipt.json").exists(), "GPU authority changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"]); bound(item["capture"])
    for modes in pre["forecast_distributions"].values():
        for v in modes.values():
            bound(v["log_probability"])
    launch = _write_new(out / "launch.json", {"status": "frozen_before_model_load", "pid": os.getpid(),
                        "started_unix": time.time(), "preflight": literal_binding(pre_path),
                        "producer": literal_binding(Path(__file__)), "hard_cap_seconds": CAP_SECONDS})
    device = torch.device(device_name)
    counts = {"model_forwards": 0, "vision_forwards": 0}
    completed = []
    hooks = []
    active = None
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", device)
        saved = source_receipt["identity"]
        _require({k:v for k,v in identity.items() if k != "loader_source"} ==
                 {k:v for k,v in saved.items() if k != "loader_source"} and
                 all(identity["loader_source"][k] == saved["loader_source"][k]
                     for k in ("sha256", "size_bytes")), "effective model identity changed")
        model = q.model.eval()
        batch, raw, trace, group, planning = _source(boundary(m, native), "untied", panel, q, device)
        _require(len(raw) == len(group["cases"]) == 1 and
                 input_identity(batch) == source_receipt["input_identity"] == pre["input_identity"],
                 "GPU source input identity changed")
        pad = int(q.tokenizer.pad_token_id)
        plan = [("unforced_native_U_source_control", None, 6, 151768),
                ("fresh_L_reference_full_vector", 1589003, 5, 152180),
                ("fresh_L_reference_full_vector", 1586761, 5, 152088),
                ("held_out_U_y1", 1589003, 6, 152180),
                ("held_out_U_y1", 1586761, 6, 152088)]
        inputs_by_cell = {}
        for mode, owner, row, x1 in plan:
            inp = make_input(model, batch, native, row, x1, pad)
            target = [[432+i for i in range(5)]]*3 if row == 5 else [[441+i for i in range(5)]]*3
            _require(inp["position_ids"][:,0,-5:].tolist() == target and
                     inp["input_ids"].shape[1] == (1412 if row == 5 else 1421),
                     "model position/physical input crosswalk changed")
            inputs_by_cell[mode, owner] = inp
        unforced = inputs_by_cell["unforced_native_U_source_control", None]
        for owner in (1589003, 1586761):
            supplied = inputs_by_cell["held_out_U_y1", owner]
            _require(torch.equal(supplied["input_ids"][:,:-1], unforced["input_ids"][:,:-1]) and
                     torch.equal(supplied["position_ids"], unforced["position_ids"]) and
                     torch.equal(supplied["attention_mask"], unforced["attention_mask"]),
                     "U history, masks or positions changed with supplied x1")

        def before_model(_module, _args, kwargs):
            counts["model_forwards"] += 1
            _require(active is not None and counts["model_forwards"] <= 5 and
                     time.monotonic()-start < CAP_SECONDS, "unexpected or over-cap model forward")
            for key in ("input_ids", "attention_mask", "position_ids"):
                actual = kwargs.get(key)
                _require(isinstance(actual, torch.Tensor) and torch.equal(actual, active["input"][key]),
                         f"actual consumer {key} changed")
                active["observed"][key] = actual.detach().cpu().tolist()
            for key in ("pixel_values", "image_grid_thw"):
                actual = kwargs.get(key)
                _require(isinstance(actual, torch.Tensor) and
                         tensor_hash(actual) == tensor_hash(active["input"][key]),
                         f"actual consumer {key} changed")
                active["observed"][key+"_sha256"] = tensor_hash(actual)
        def before_vision(_module, _args):
            counts["vision_forwards"] += 1
            _require(counts["vision_forwards"] <= 5, "vision forward ceiling reached")
        hooks.extend((model.register_forward_pre_hook(before_model, with_kwargs=True),
                      model.model.visual.register_forward_pre_hook(before_vision)))
        refs = {}
        for index, (mode, owner, row, x1) in enumerate(plan, 1):
            if mode == "held_out_U_y1":
                _require(len(refs) == 2 and len(completed) >= 3 and
                         all(z["passed"] for z in refs.values()), "controls incomplete before target")
            inp = inputs_by_cell[mode, owner]
            active = {"input": inp, "observed": {}}
            seen, scoped = observed_hooks(model, inp["position_ids"], 5, 0)
            rotary = []
            extra = model.model.language_model.rotary_emb.register_forward_pre_hook(
                lambda _module, args: rotary.append(args[1].detach().cpu().tolist()))
            try:
                with torch.inference_mode():
                    vec = model(**inp).logits[0,-1].detach().float().cpu()
            finally:
                extra.remove()
                for hook in scoped:
                    hook.remove()
            torch.cuda.synchronize(device)
            _require(len(rotary) == seen["rotary"] == 1 and
                     active["observed"]["position_ids"] == rotary[0] and
                     len(seen["masks"]) == len(seen["caches"]) == seen["embedding"] == 2,
                     "actual rotary/attention evidence incomplete")
            observed = {**active["observed"], "rotary_position_ids": rotary[0],
                        "attention_mask_hashes": seen["masks"],
                        "cache_position_hashes": seen["caches"],
                        "position_embedding_hooks": seen["embedding"]}
            active = None
            cell_dir = out / "cells" / f"{index:02d}-{mode}-{owner or 'native'}"
            cell_dir.mkdir(parents=True, exist_ok=False)
            torch.save(vec, cell_dir / "vocabulary.pt")
            vector = literal_binding(cell_dir / "vocabulary.pt")
            consumer = _write_new(cell_dir / "consumer-raw.json", observed)
            _require(vec.ndim == 1 and torch.isfinite(vec).all(), "invalid full vocabulary vector")
            top = torch.topk(vec, 2)
            top_ids, top_logits = top.indices.tolist(), top.values.tolist()
            record = {"mode": mode, "owner": owner, "row": row, "x1_token": x1,
                      "vector": vector, "consumer": consumer, "top2_ids": top_ids,
                      "top2_logits": top_logits, "top2_gap": top_logits[0]-top_logits[1],
                      "logsumexp": float(torch.logsumexp(vec, -1)),
                      "vocabulary_size": int(vec.numel()),
                      "counters_after": dict(counts), "allocated_gpu_seconds_after": time.monotonic()-start}
            if mode == "unforced_native_U_source_control":
                check = _trace_compare(logits=vec, trace=trace, batch_index=0,
                                       absolute_offset=59, token_id=152249, role="native_y1")
                record["source_parity"] = check
                _require(check["passed"], "native U source parity failed")
            elif mode == "fresh_L_reference_full_vector":
                source = next(p for p in m["probes"] if p["candidate_owner"] == owner)
                old = torch.load(bound(source["source_vectors"]["L"]), map_location="cpu", weights_only=True)
                error = float((vec-old).abs().max())
                record["reference_max_abs_error"] = error
                refs[owner] = {"passed": error <= TOL, "max_abs_error": error}
                _require(error <= TOL, "fresh L reference full vector diverged")
            _write_new(cell_dir / "cell.json", record)
            completed.append({"record": literal_binding(cell_dir / "cell.json"),
                              "mode": mode, "owner": owner, "row": row, "x1_token": x1})
            _write_new(out / f"checkpoint-{index:02d}.json",
                       {"completed": completed, "counts": dict(counts), "allocated_gpu_seconds": time.monotonic()-start})
            _require(time.monotonic()-start < CAP_SECONDS, "incremental GPU cap reached")
        _require(counts == {"model_forwards": 5, "vision_forwards": 5}, "five calls incomplete")
        cost = {"allocated_gpu_seconds": time.monotonic()-start,
                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), **counts}
        _require(cost["allocated_gpu_seconds"] < CAP_SECONDS and
                 m["budget"]["sequence_cumulative_prior_gpu_hours"] + cost["allocated_gpu_seconds"]/3600 < 8,
                 "GPU cap exceeded")
        pilot = _write_new(out / "pilot.json", {"schema": "recurrence_next_history_prediction.pilot.v1",
                  "status": "candidate_complete", "preflight": literal_binding(pre_path), "launch": launch,
                  "protocol": literal_binding(PROTOCOL), "manifest": literal_binding(MANIFEST),
                  "effective_identity": identity, "input_identity": input_identity(batch),
                  "source_planning": planning, "completed": completed, "controls": refs, "cost": cost})
        terminal = {"status": "candidate_complete", "terminal": True, "pilot": pilot,
                    "completed_cells": 5, "cost": cost}
    except BaseException as exc:
        terminal = {"status": "technical_invalid", "terminal": True, "error": repr(exc),
                    "traceback": traceback.format_exc(), "completed": completed,
                    "cost": {"allocated_gpu_seconds": time.monotonic()-start,
                             "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                             "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), **counts}}
    finally:
        for hook in hooks:
            hook.remove()
    terminal["artifact_bytes_before_receipt"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    _write_new(out / "receipt.json", terminal)
    if terminal["status"] != "candidate_complete":
        raise RuntimeError(terminal["error"])
    print(json.dumps({"status": terminal["status"], "cost": terminal["cost"]}))


def readback(out):
    m, _, native, _ = contract()
    pre = json.loads((out / "preflight.json").read_text())
    _require(pre["status"] == "cpu_qualified_before_gpu", "cold preflight changed")
    for item in pre["direct_source_captures"]:
        bound(item["capture"])
    forecast = {}
    for owner, modes in pre["forecast_distributions"].items():
        forecast[owner] = {}
        for mode, record in modes.items():
            vec = torch.load(bound(record["log_probability"]), map_location="cpu", weights_only=True)
            _require(vec.dtype == torch.float64 and vec.ndim == 1 and
                     torch.isfinite(vec).all() and abs(float(torch.logsumexp(vec,-1))) < 1e-10 and
                     int(vec.argmax()) == record["top2_token_ids"][0], "cold forecast invalid")
            forecast[owner][mode] = vec
    terminal = json.loads((out / "receipt.json").read_text())
    _require(terminal["terminal"] and terminal["status"] == "candidate_complete" and
             terminal["completed_cells"] == 5, "cold terminal incomplete")
    pilot = json.loads(bound(terminal["pilot"]).read_text())
    _require(pilot["cost"]["model_forwards"] == pilot["cost"]["vision_forwards"] == 5 and
             len(pilot["completed"]) == 5, "cold cell/forward count changed")
    cells = []
    for item in pilot["completed"]:
        cell = json.loads(bound(item["record"]).read_text())
        vec = torch.load(bound(cell["vector"]), map_location="cpu", weights_only=True)
        consumer = json.loads(bound(cell["consumer"]).read_text())
        _require(vec.ndim == 1 and torch.isfinite(vec).all() and
                 vec.numel() == cell["vocabulary_size"] and int(vec.argmax()) == cell["top2_ids"][0] and
                 math.isclose(float(torch.logsumexp(vec,-1)), cell["logsumexp"], abs_tol=1e-6) and
                 consumer["position_ids"] == consumer["rotary_position_ids"] and
                 len(consumer["attention_mask_hashes"]) == 2,
                 "cold vector/consumer invalid")
        prompt_len = pre["shape"]["prompt_width"]
        _require(len(consumer["input_ids"]) == 1 and
                 consumer["input_ids"][0][prompt_len:] == suffix(native, cell["row"], cell["x1_token"]) and
                 consumer["position_ids"][0][0][-5:] ==
                 ([432+i for i in range(5)] if cell["row"] == 5 else [441+i for i in range(5)]),
                 "cold supplied history/position invalid")
        cells.append({"record": item["record"], "cell": cell})
    expected = [("unforced_native_U_source_control", None, 6, 151768),
                ("fresh_L_reference_full_vector", 1589003, 5, 152180),
                ("fresh_L_reference_full_vector", 1586761, 5, 152088),
                ("held_out_U_y1", 1589003, 6, 152180),
                ("held_out_U_y1", 1586761, 6, 152088)]
    _require([(x["cell"]["mode"],x["cell"]["owner"],x["cell"]["row"],x["cell"]["x1_token"])
              for x in cells] == expected, "cold five cells changed")
    return {"status": "passed", "pilot": literal_binding(out/"pilot.json"),
            "terminal": literal_binding(out/"receipt.json"), "cells": 5,
            "forecasts": 6, "allocated_gpu_seconds": pilot["cost"]["allocated_gpu_seconds"]}


def reduce(out):
    m, _, _, _ = contract()
    cold = readback(out)
    pre = json.loads((out/"preflight.json").read_text())
    pilot = json.loads((out/"pilot.json").read_text())
    results = {}
    for item in pilot["completed"]:
        cell = json.loads(Path(item["record"]["path"]).read_text())
        if cell["mode"] != "held_out_U_y1":
            continue
        owner = str(cell["owner"])
        actual = torch.load(Path(cell["vector"]["path"]), map_location="cpu", weights_only=True).double()
        logp = torch.log_softmax(actual, -1)
        p = logp.exp()
        forecasts = {}
        for mode, v in pre["forecast_distributions"][owner].items():
            q = torch.load(Path(v["log_probability"]["path"]), map_location="cpu", weights_only=True)
            forecasts[mode] = {"forecast": v["log_probability"],
                               "winner": v["top2_token_ids"][0],
                               "winner_match": v["top2_token_ids"][0] == cell["top2_ids"][0],
                               "kl_p_actual_q": float(torch.sum(p*(logp-q))),
                               "probability_of_actual_winner": float(q[cell["top2_ids"][0]].exp())}
        improve = forecasts["persistence"]["kl_p_actual_q"] - forecasts["affine_native_step"]["kl_p_actual_q"]
        results[owner] = {"actual": {"winner": cell["top2_ids"][0],
                                     "runner": cell["top2_ids"][1], "gap": cell["top2_gap"],
                                     "vector": cell["vector"]},
                          "forecasts": forecasts, "affine_kl_improvement_over_persistence": improve,
                          "affine_strict_probe_pass": forecasts["affine_native_step"]["winner_match"] and improve > .001,
                          "kl_numerical_guard": abs(improve) <= .001}
    _require(len(results) == 2, "two targets missing")
    verdict = all(v["affine_strict_probe_pass"] for v in results.values())
    report = {"schema": "recurrence_next_history_prediction.reduction.v1", "status": "candidate_complete",
              "protocol": literal_binding(PROTOCOL), "manifest": literal_binding(MANIFEST),
              "preflight": literal_binding(out/"preflight.json"), "pilot": literal_binding(out/"pilot.json"),
              "terminal": literal_binding(out/"receipt.json"), "cold_readback": cold,
              "strict_shared_affine_prediction_pass": verdict, "probes": results,
              "five_cells": pilot["completed"], "cost": pilot["cost"],
              "sequence_cumulative_gpu_hours": m["budget"]["sequence_cumulative_prior_gpu_hours"] + pilot["cost"]["allocated_gpu_seconds"]/3600,
              "artifact_bytes_before_reduction": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
    _write_new(out/"reduction.json", report)
    print(json.dumps({"status": report["status"], "strict_shared_affine_prediction_pass": verdict,
                      "probes": results, "cost": pilot["cost"]}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("preflight", "run", "readback", "reduce"))
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.mode == "preflight": preflight(a.output)
    elif a.mode == "run": run(a.output, a.device)
    elif a.mode == "readback":
        result = readback(a.output)
        _write_new(a.output/"cold-readback.json", result)
        print(json.dumps(result))
    else: reduce(a.output)


if __name__ == "__main__":
    main()
