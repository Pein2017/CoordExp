"""Execute the frozen 18-forward chair y1 history/position grid."""

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
from probes.training_set_completion.recurrence_first_arrivals.prepare import MATURE, _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import _write_new
from probes.training_set_completion.recurrence_position_history import observed_hooks, swap_prefix_positions
from probes.training_set_completion.untied_shared import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.qwen.input_identity import input_identity, tensor_hash
from src.qwen.native import exact_history_inputs
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-chair-history-position"
PROTOCOL = UNIT / "unit.md"
MANIFEST = UNIT / "manifest.json"
PROTOCOL_SHA = "acd71e7484f18600000dd66343bd4abebba3f7c74ed16f5c08293d2087960efe"
MANIFEST_SHA = "3410f250e0113f907a8fbd07d8e402a4aca6f41fd987569d48a884c3c65587d8"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-chair-history-position/attempt-001")
CAP_SECONDS = 3600
TOL = 2e-4
PROBES = (1589003, 1586761)
PREFIX = 5


def bound(item: dict) -> Path:
    path = Path(item["path"])
    if not path.is_absolute():
        path = REPO / path
    actual = literal_binding(path)
    _require(actual["sha256"] == item["sha256"] and actual["size_bytes"] == item["size_bytes"],
             f"source binding changed: {path}")
    return path


def contract() -> tuple[dict, dict, list[int], dict]:
    _require(literal_binding(PROTOCOL)["sha256"] == PROTOCOL_SHA, "corrected protocol changed")
    _require(literal_binding(MANIFEST)["sha256"] == MANIFEST_SHA, "lead manifest changed")
    m = json.loads(MANIFEST.read_text())
    _require(m["status"] == "lead-frozen-before-successor-model-calls" and
             m["family"] == "mature:477415:4" and m["source_group"] == "refined-04" and
             m["batch_index"] == 0 and m["early_row_start"] == 36 and
             m["late_row_start"] == 45 and m["current_prefix_length_through_supplied_x1"] == PREFIX,
             "frozen source/slot contract changed")
    _require(len(m["cells"]) == 18 and sum(c["mode"] == "y1_grid" for c in m["cells"]) == 12 and
             sum(c["mode"] == "explicit_identity_positions" for c in m["cells"]) == 4 and
             sum(c["mode"] == "unforced_native_y1_source_control" for c in m["cells"]) == 2 and
             m["budget"]["model_forward_ceiling_without_repair"] == 18 and
             m["budget"]["vision_forward_ceiling_without_repair"] == 18 and
             m["budget"]["incremental_allocated_gpu_hour_ceiling"] == 1,
             "finite cell/budget contract changed")
    _require([p["candidate_owner"] for p in m["x1_probes"]] == list(PROBES), "fixed probes changed")
    expected = [(probe, history, position) for probe in PROBES
                for history in ("E", "L_A", "L_B") for position in ("E", "L")]
    actual = [(c["candidate_owner"], c["history"], c["positions"])
              for c in m["cells"] if c["mode"] == "y1_grid"]
    _require(actual == expected, "12-grid-cell order changed")
    bound(m["predecessor_acceptance"])
    registry = json.loads(bound(m["frozen_source_registry"]).read_text())
    for key in ("raw", "trace", "runtime_receipt", "image"):
        bound(m["source_bindings"][key])
    panel = MATURE / "panel.json"
    panel_match = [x for x in registry["source_files"] if Path(x["path"]) == panel]
    _require(len(panel_match) == 1, "frozen source panel absent")
    bound(panel_match[0])
    for x in m["prior_branch_bindings"]:
        for key in ("scores_raw", "branch_raw", "overlay"):
            bound(x[key])
    source = m["source_bindings"]
    raw = json.loads(Path(source["raw"]["path"]).read_text())["rows"]
    receipt = json.loads(Path(source["runtime_receipt"]["path"]).read_text())
    _require(len(raw) == 1 and receipt["input_identity"]["request_ids"] ==
             ["coco2017_train_000000477415"] and
             source["input_identity_summary"]["request_ids"] == receipt["input_identity"]["request_ids"],
             "original single-request source changed")
    native = [int(t) for t in raw[0]["token_ids"]]
    parsed = rows(native)
    _require(parsed[1]["start"] == 9 and parsed[1]["end"] == 18 and
             parsed[4]["start"] == 36 and parsed[4]["end"] == 45 and
             parsed[5]["start"] == 45 and
             native[36:45] == native[45:54] == m["A_row_tokens"] and
             native[9:18] == m["B_row_tokens"], "native A/B row boundaries changed")
    a, b = m["A_row_tokens"], m["B_row_tokens"]
    _require(len(a) == len(b) == 9 and [i for i in range(9) if a[i] != b[i]] == [5, 6] and
             a[0:5] == b[0:5] and a[7:] == b[7:], "B changes more than y1/x2")
    for p in m["x1_probes"]:
        _require(p["supplied_x1_token"] == 151670 + p["supplied_x1_bin"] and
                 p["reference_E_y1_bin"] != p["reference_L_y1_bin"], "fixed x1/y1 changed")
    return m, receipt, native, json.loads(panel.read_text())


def boundary(m: dict, native: list[int]) -> dict:
    source = m["source_bindings"]
    return {"group": m["source_group"], "batch_index": 0, "image_id": 477415,
            "raw_path": source["raw"]["path"], "trace_path": source["trace"]["path"],
            "receipt_path": source["runtime_receipt"]["path"],
            "native_tokens": native, "native_token_hash": token_hash(native)}


def tokens(m: dict, native: list[int], history: str, x1: int) -> list[int]:
    start = m["early_row_start"] if history == "E" else m["late_row_start"]
    prior = native[:start]
    if history == "L_B":
        prior = native[:m["early_row_start"]] + list(m["B_row_tokens"])
    current = list(m["current_shared_prefix_tokens"]) + [int(x1)]
    _require(len(current) == PREFIX and current[:4] == native[start:start + 4], "current prefix changed")
    return prior + current


def cpu_preflight(out: Path) -> dict:
    m, receipt, native, panel = contract()
    _require(not out.exists(), "attempt output already exists")
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    _require(q.model is None, "CPU preflight unexpectedly loaded model")
    batch, raw, trace, group, planning = _source(boundary(m, native), "untied", panel, q, torch.device("cpu"))
    _require(len(raw) == len(group["cases"]) == 1 and input_identity(batch) == receipt["input_identity"],
             "full original CPU source identity changed")
    prompt = list(batch.prompt_token_ids[0])
    _require(len(prompt) == int(batch.inputs["input_ids"].shape[1]), "single-request prompt width changed")
    refs = {}
    for probe in m["x1_probes"]:
        x1 = probe["supplied_x1_token"]
        for history, position, row in (("E", "E", 4), ("L_A", "L", 5)):
            expected = prompt + tokens(m, native, history, x1)
            source = next(z for z in m["prior_branch_bindings"] if
                          z["candidate"] == probe["candidate_owner"] and z["arrival_row"] == row)
            saved = json.loads(Path(source["branch_raw"]["path"]).read_text())
            _require(saved["consumer_input_ids"] == [expected] and
                     saved["consumer_attention_mask"] == [[1] * len(expected)] and
                     saved["prefix_end"] == m["early_row_start" if row == 4 else "late_row_start"] + PREFIX,
                     "retained Stage2 actual consumer reference differs from CPU planned input")
            score = json.loads(Path(source["scores_raw"]["path"]).read_text())
            s_positions = score["A_and_N"]["A"]["position_ids"][1:6]
            _require(len(s_positions) == PREFIX and
                     all(z == [s_positions[0][0] + i] * 3 for i, z in enumerate(s_positions)) and
                     saved["first_decisions"][0]["top2_ids"][0] == 151670 +
                     probe["reference_E_y1_bin" if row == 4 else "reference_L_y1_bin"],
                     "reference S positions or y1 winner changed")
            refs[f"{probe['candidate_owner']}:{history}"] = {
                "physical_S_range": [len(prompt) + (36 if row == 4 else 45),
                                     len(prompt) + (36 if row == 4 else 45) + PREFIX],
                "raw_S_range": [36 if row == 4 else 45, (36 if row == 4 else 45) + PREFIX],
                "raw_y1_offset": (36 if row == 4 else 45) + PREFIX,
                "rotary_S_positions": s_positions,
                "actual_stage2_consumer": source["branch_raw"]}
    for probe in PROBES:
        e, l = refs[f"{probe}:E"], refs[f"{probe}:L_A"]
        _require(all(l["rotary_S_positions"][i][axis] - e["rotary_S_positions"][i][axis] == 9
                     for i in range(PREFIX) for axis in range(3)), "E/L positional shift changed")
    # Falsify the nearest wrong implementation: wrong slot, extra B edit, or historical position edit.
    sample = {"input_ids": torch.tensor([[11, 12, 13, 14, 15, 16, 17]]),
              "attention_mask": torch.ones(1, 7, dtype=torch.long),
              "position_ids": torch.arange(7).expand(3, 1, 7).clone()}
    donor = torch.arange(8, 13).expand(3, 5).clone()
    crossed = swap_prefix_positions(sample, donor, 0, 5, (13, 14, 15, 16, 17))
    _require(torch.equal(crossed["position_ids"][:, :, :2], sample["position_ids"][:, :, :2]),
             "history position mutation escaped")
    for wrong in ((14, 15, 16, 17, 18), (13, 14, 15, 16)):
        try:
            swap_prefix_positions(sample, donor, 0, 5, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong S slot escaped CPU test")
    altered = list(m["B_row_tokens"])
    altered[4] += 1
    _require([i for i, (a, b) in enumerate(zip(m["A_row_tokens"], altered, strict=True)) if a != b]
             != [5, 6], "wrong B content mutation escaped")
    imports = [
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
        "src/qwen/input_identity.py",
        "src/qwen/native.py",
        "src/qwen/runtime_loading.py",
        "src/inference/bound_requests.py",
    ]
    out.mkdir(parents=True)
    captures = []
    for name in imports:
        source = REPO / name
        captured = preserve_source(source, run_root=out, relative_name=Path(name))
        captures.append({"maintained": literal_binding(source), "capture": literal_binding(captured)})
    old = json.loads(Path(m["predecessor_acceptance"]["path"]).read_text())
    prior = json.loads(Path(old["receipt"]["path"]).read_text())
    old_seconds = prior["cost"]["allocated_gpu_seconds"]
    shape = {"batch_size": 1, "prompt_width": len(prompt), "image_grids": [list(x) for x in batch.image_grids],
             "max_raw_prefix": 50, "full_input_width": len(prompt) + 50,
             "pixel_elements": int(batch.inputs["pixel_values"].numel())}
    old_chair_width = 1362 + 45
    shape_factor = max(1., shape["full_input_width"] / old_chair_width)
    estimate = 2 * old_seconds * (18 / prior["cost"]["vision_forwards"]) * shape_factor
    _require(estimate < CAP_SECONDS and
             m["budget"]["sequence_cumulative_prior_gpu_hours"] + estimate / 3600 <
             m["budget"]["sequence_ceiling_gpu_hours"], "shape-aware cost forecast exceeds cap")
    packet = {"schema": "recurrence_chair_history_position.cpu_preflight.v1",
              "status": "cpu_qualified_before_gpu", "protocol": literal_binding(PROTOCOL),
              "manifest": literal_binding(MANIFEST), "source": m["source_bindings"],
              "input_identity": input_identity(batch), "source_planning": planning,
              "shape": shape, "reference_crosswalks": refs,
              "cpu_mutation_checks": ["wrong_S_slot_rejected", "extra_B_content_edit_rejected",
                                      "historical_position_edit_rejected"],
              "direct_source_captures": captures,
              "forecast": {"old_stage2_seconds": old_seconds,
                           "old_stage2_vision_calls": prior["cost"]["vision_forwards"],
                           "new_vision_calls": 18, "shape_factor": shape_factor,
                           "planning_multiplier": 2, "estimated_incremental_gpu_seconds": estimate,
                           "incremental_hard_cap_seconds": CAP_SECONDS,
                           "sequence_prior_gpu_hours": m["budget"]["sequence_cumulative_prior_gpu_hours"],
                           "sequence_hard_cap_gpu_hours": m["budget"]["sequence_ceiling_gpu_hours"]},
              "commands": {"gpu": ["python", "-B", "-m",
                                   "probes.training_set_completion.recurrence_chair_history_position.run",
                                   "run", "--output", str(out), "--device", "cuda:0"],
                           "readback": ["python", "-B", "-m",
                                        "probes.training_set_completion.recurrence_chair_history_position.run",
                                        "readback", "--output", str(out)]}}
    _write_new(out / "preflight.json", packet)
    print(json.dumps({"status": packet["status"], "forecast_gpu_hours": estimate / 3600,
                      "source_shape": shape, "references": len(refs)}))
    return packet


def make_input(model, batch, m: dict, native: list[int], history: str,
               x1: int, position: str, pad: int) -> dict:
    history_tokens = tokens(m, native, history, x1)
    request = list(batch.prompt_token_ids[0]) + history_tokens
    result = exact_history_inputs(model, batch.inputs, [request], pad_token_id=pad, logits_to_keep=1)
    _require(result["input_ids"].shape[0] == 1 and result["input_ids"][0].tolist() == request and
             result["attention_mask"][0].tolist() == [1] * len(request),
             "full original request/physical history changed")
    return result


def run(out: Path, device_name: str) -> dict:
    started = time.monotonic()
    m, source_receipt, native, panel = contract()
    preflight_path = out / "preflight.json"
    pre = json.loads(preflight_path.read_text())
    _require(pre["status"] == "cpu_qualified_before_gpu" and
             pre["protocol"] == literal_binding(PROTOCOL) and pre["manifest"] == literal_binding(MANIFEST) and
             device_name == "cuda:0" and not (out / "receipt.json").exists(),
             "preflight/run authority changed")
    for item in pre["direct_source_captures"]:
        bound(item["maintained"])
        bound(item["capture"])
    launch = _write_new(out / "launch.json", {"schema": "recurrence_chair_history_position.launch.v1",
                        "status": "frozen_before_model_load", "pid": os.getpid(), "started_unix": time.time(),
                        "device": device_name, "preflight": literal_binding(preflight_path),
                        "producer": literal_binding(Path(__file__)), "hard_cap_seconds": CAP_SECONDS})
    device = torch.device(device_name)
    counts = {"model_forwards": 0, "vision_forwards": 0}
    handles = []
    completed = []
    active = None
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q, identity = load_model("untied", device)
        saved = source_receipt["identity"]
        _require({k: v for k, v in identity.items() if k != "loader_source"} ==
                 {k: v for k, v in saved.items() if k != "loader_source"} and
                 all(identity["loader_source"][k] == saved["loader_source"][k]
                     for k in ("sha256", "size_bytes")), "effective model identity changed")
        model = q.model.eval()
        batch, raw, trace, group, planning = _source(boundary(m, native), "untied", panel, q, device)
        _require(len(raw) == len(group["cases"]) == 1 and input_identity(batch) ==
                 source_receipt["input_identity"] == pre["input_identity"],
                 "fresh single-request input identity changed")
        pad = int(q.tokenizer.pad_token_id)
        base = {}
        for probe in m["x1_probes"]:
            owner, x1 = probe["candidate_owner"], probe["supplied_x1_token"]
            for history in ("E", "L_A", "L_B"):
                base[owner, history] = make_input(model, batch, m, native, history, x1, "E", pad)
            _require(torch.equal(base[owner, "L_A"]["position_ids"],
                                 base[owner, "L_B"]["position_ids"]),
                     "B geometry changed position layout")
        base_native = {h: make_input(model, batch, m, native, h, 151670, "E", pad)
                       for h in ("E", "L_A")}
        for owner in PROBES:
            for h in ("E", "L_A"):
                check = pre["reference_crosswalks"][f"{owner}:{h}"]
                pos = base[owner, h]["position_ids"][:, 0, -PREFIX:].T.tolist()
                _require(pos == check["rotary_S_positions"], "actual native S MRoPE crosswalk changed")
        donor = {owner: {"E": base[owner, "E"]["position_ids"][:, 0, -PREFIX:].clone(),
                         "L": base[owner, "L_A"]["position_ids"][:, 0, -PREFIX:].clone()}
                 for owner in PROBES}
        _require(all(int(x) == 9 for owner in PROBES for x in
                     (donor[owner]["L"] - donor[owner]["E"]).flatten().tolist()),
                 "early/late MRoPE delta changed")
        for owner in PROBES:
            _require(torch.equal(donor[PROBES[0]]["E"], donor[owner]["E"]) and
                     torch.equal(donor[PROBES[0]]["L"], donor[owner]["L"]),
                     "probes have different S position donors")

        def before_model(_module, _args, kwargs):
            nonlocal active
            counts["model_forwards"] += 1
            _require(counts["model_forwards"] <= 18 and time.monotonic() - started < CAP_SECONDS,
                     "model-forward or GPU-hour ceiling reached")
            _require(active is not None, "unexpected model forward")
            for key in ("input_ids", "attention_mask", "position_ids"):
                observed = kwargs.get(key)
                _require(isinstance(observed, torch.Tensor) and
                         torch.equal(observed, active["expected"][key]),
                         f"actual model consumer {key} changed")
                active["observed"][key] = observed.detach().cpu().tolist()
            for key in ("pixel_values", "image_grid_thw"):
                observed = kwargs.get(key)
                _require(isinstance(observed, torch.Tensor) and
                         tensor_hash(observed) == tensor_hash(active["expected"][key]),
                         f"actual model consumer {key} changed")
                active["observed"][key + "_sha256"] = tensor_hash(observed)
        def before_vision(_module, _args):
            counts["vision_forwards"] += 1
            _require(counts["vision_forwards"] <= 18, "vision-forward ceiling reached")
        handles.extend((model.register_forward_pre_hook(before_model, with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision)))

        planned = []
        for h in ("E", "L_A"):
            planned.append(("unforced_native_y1_source_control", None, h, "E" if h == "E" else "L"))
        for owner in PROBES:
            planned.extend((("y1_grid", owner, "E", "E"), ("y1_grid", owner, "L_A", "L")))
        for owner in PROBES:
            planned.extend((("explicit_identity_positions", owner, "E", "E"),
                            ("explicit_identity_positions", owner, "L_A", "L")))
        for owner in PROBES:
            planned.extend((("y1_grid", owner, "E", "L"), ("y1_grid", owner, "L_A", "E"),
                            ("y1_grid", owner, "L_B", "E"), ("y1_grid", owner, "L_B", "L")))
        _require(len(planned) == 18 and sorted((x[0], x[1], x[2], x[3]) for x in planned if x[1] is not None)
                 == sorted((c["mode"], c["candidate_owner"], c["history"], c["positions"])
                           for c in m["cells"] if c["mode"] != "unforced_native_y1_source_control"),
                 "ordered execution differs from frozen cells")
        refs = {}
        for index, (mode, owner, history, position) in enumerate(planned, 1):
            if index > 10:
                _require(len(refs) == 4, "references incomplete before crossed/content cells")
            cell = f"{index:02d}-{mode}-{owner if owner is not None else 'native'}-{history}-{position}"
            if owner is None:
                inputs = base_native[history]
            else:
                original = base[owner, history]
                expected_tokens = tuple(original["input_ids"][0, -PREFIX:].tolist())
                inputs = swap_prefix_positions(original, donor[owner][position], 0, PREFIX,
                                               expected_tokens)
                _require(torch.equal(inputs["input_ids"], original["input_ids"]) and
                         torch.equal(inputs["attention_mask"], original["attention_mask"]) and
                         torch.equal(inputs["position_ids"][:, 0, :-PREFIX],
                                     original["position_ids"][:, 0, :-PREFIX]),
                         "history/mask/physical order changed by S crossing")
            active = {"expected": inputs, "observed": {}}
            seen, hookset = observed_hooks(model, inputs["position_ids"], PREFIX, 0)
            rotary_values = []
            def rotary_capture(_module, args):
                rotary_values.append(args[1].detach().cpu().tolist())
            extra = model.model.language_model.rotary_emb.register_forward_pre_hook(rotary_capture)
            try:
                with torch.inference_mode():
                    logits = model(**inputs).logits[0, -1].detach().float().cpu()
            finally:
                extra.remove()
                for handle in hookset:
                    handle.remove()
            torch.cuda.synchronize(device)
            _require(len(rotary_values) == seen["rotary"] == 1 and
                     active["observed"]["position_ids"] == rotary_values[0] and
                     len(seen["masks"]) == len(seen["caches"]) == seen["embedding"] == 2,
                     "actual rotary/attention consumer evidence incomplete")
            observed = {**active["observed"], "rotary_position_ids": rotary_values[0],
                        "attention_mask_hashes": seen["masks"], "cache_position_hashes": seen["caches"],
                        "position_embedding_hooks": seen["embedding"]}
            active = None
            cell_dir = out / "cells" / cell
            cell_dir.mkdir(parents=True, exist_ok=False)
            torch.save(logits, cell_dir / "vocabulary.pt")
            vector = literal_binding(cell_dir / "vocabulary.pt")
            consumer = _write_new(cell_dir / "consumer-raw.json", observed)
            _require(torch.isfinite(logits).all() and logits.ndim == 1, "invalid full-vocabulary result")
            top = torch.topk(logits, 2)
            top_ids = [int(x) for x in top.indices.tolist()]
            top_logits = [float(x) for x in top.values.tolist()]
            lse = float(torch.logsumexp(logits, dim=-1))
            record = {"mode": mode, "candidate_owner": owner, "history": history, "positions": position,
                      "vector": vector, "consumer": consumer, "top2_ids": top_ids,
                      "top2_logits": top_logits, "top2_gap": top_logits[0] - top_logits[1],
                      "logsumexp": lse, "vocabulary_size": int(logits.numel()),
                      "elapsed_allocated_gpu_seconds": time.monotonic() - started,
                      "counters_after": dict(counts)}
            if mode == "unforced_native_y1_source_control":
                offset = (36 if history == "E" else 45) + PREFIX
                check = _trace_compare(logits=logits, trace=trace, batch_index=0,
                                       absolute_offset=offset, token_id=152362, role="native_y1")
                record["source_parity"] = check
                _require(check["passed"], "unforced source y1 parity failed")
            elif mode == "y1_grid" and (history, position) in (("E", "E"), ("L_A", "L")):
                prior = next(z for z in m["prior_branch_bindings"] if
                             z["candidate"] == owner and z["arrival_row"] == (4 if history == "E" else 5))
                prior_raw = json.loads(Path(prior["branch_raw"]["path"]).read_text())
                decision = prior_raw["first_decisions"][0]
                errors = {"top2": max(abs(a - b) for a, b in zip(top_logits, decision["top2_logits"], strict=True)),
                          "logsumexp": abs(lse - decision["logsumexp"])}
                record["stage2_reference_errors"] = errors
                _require(top_ids == decision["top2_ids"] and max(errors.values()) <= TOL,
                         "supplied-prefix Stage2 reference changed")
                refs[owner, history] = logits
            elif mode == "explicit_identity_positions":
                prior_logits = refs[owner, history]
                record["identity_max_abs_error"] = float((logits - prior_logits).abs().max())
                _require(record["identity_max_abs_error"] <= TOL, "explicit identity S positions changed vector")
            _write_new(cell_dir / "cell.json", record)
            completed.append({"cell": cell, "mode": mode, "candidate_owner": owner,
                              "history": history, "positions": position,
                              "record": literal_binding(cell_dir / "cell.json")})
            _write_new(out / f"checkpoint-{index:02d}.json",
                       {"completed": completed, "counters": counts,
                        "allocated_gpu_seconds": time.monotonic() - started})
            _require(time.monotonic() - started < CAP_SECONDS, "allocated GPU-hour ceiling reached")
        _require(counts == {"model_forwards": 18, "vision_forwards": 18} and len(completed) == 18,
                 "finite 18-forward execution incomplete")
        cost = {"allocated_gpu_seconds": time.monotonic() - started,
                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), **counts}
        _require(cost["allocated_gpu_seconds"] < CAP_SECONDS and
                 m["budget"]["sequence_cumulative_prior_gpu_hours"] +
                 cost["allocated_gpu_seconds"] / 3600 < m["budget"]["sequence_ceiling_gpu_hours"],
                 "cumulative GPU cap exceeded")
        result = {"schema": "recurrence_chair_history_position.pilot.v1", "status": "candidate_complete",
                  "protocol": literal_binding(PROTOCOL), "manifest": literal_binding(MANIFEST),
                  "preflight": literal_binding(preflight_path), "launch": launch,
                  "effective_identity": identity, "input_identity": input_identity(batch),
                  "source_planning": planning, "completed": completed, "cost": cost}
        pilot = _write_new(out / "pilot.json", result)
        terminal = {"schema": "recurrence_chair_history_position.receipt.v1",
                    "status": "candidate_complete", "terminal": True, "pilot": pilot,
                    "completed_cells": 18, "cost": cost,
                    "artifact_bytes_before_receipt": sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    except BaseException as error:
        terminal = {"schema": "recurrence_chair_history_position.receipt.v1",
                    "status": "technical_invalid", "terminal": True, "error": repr(error),
                    "traceback": traceback.format_exc(), "completed": completed,
                    "cost": {"allocated_gpu_seconds": time.monotonic() - started,
                             "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                             "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), **counts},
                    "artifact_bytes_before_receipt": sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    finally:
        for handle in handles:
            handle.remove()
    _write_new(out / "receipt.json", terminal)
    if terminal["status"] != "candidate_complete":
        raise RuntimeError(terminal["error"])
    return terminal


def readback(out: Path) -> dict:
    m, _receipt, _native, _panel = contract()
    pre = json.loads((out / "preflight.json").read_text())
    _require(pre["status"] == "cpu_qualified_before_gpu", "cold preflight status changed")
    for item in pre["direct_source_captures"]:
        bound(item["capture"])
    terminal = json.loads((out / "receipt.json").read_text())
    _require(terminal["status"] == "candidate_complete" and terminal["terminal"] and
             terminal["completed_cells"] == 18, "cold terminal receipt incomplete")
    bound(terminal["pilot"])
    pilot = json.loads((out / "pilot.json").read_text())
    _require(len(pilot["completed"]) == 18 and pilot["cost"]["model_forwards"] ==
             pilot["cost"]["vision_forwards"] == 18, "cold forward/cell count changed")
    found = []
    for entry in pilot["completed"]:
        bound(entry["record"])
        cell = json.loads(Path(entry["record"]["path"]).read_text())
        bound(cell["vector"])
        bound(cell["consumer"])
        values = torch.load(cell["vector"]["path"], map_location="cpu", weights_only=True)
        _require(values.ndim == 1 and values.numel() == cell["vocabulary_size"] and
                 torch.isfinite(values).all() and int(values.argmax()) == cell["top2_ids"][0] and
                 math.isclose(float(torch.logsumexp(values, dim=-1)), cell["logsumexp"], abs_tol=1e-6),
                 "cold full-vector/top2/normalization changed")
        observed = json.loads(Path(cell["consumer"]["path"]).read_text())
        _require(observed["position_ids"] == observed["rotary_position_ids"] and
                 len(observed["input_ids"]) == 1 and len(observed["attention_mask_hashes"]) == 2,
                 "cold actual input/rotary evidence changed")
        found.append((cell["mode"], cell["candidate_owner"], cell["history"], cell["positions"]))
    planned = [(c["mode"], c.get("candidate_owner"), c["history"], c["positions"]) for c in m["cells"]]
    _require(sorted(str(x) for x in found) == sorted(str(x) for x in planned), "cold 18-cell manifest differs")
    return {"schema": "recurrence_chair_history_position.readback.v1", "status": "passed",
            "pilot": literal_binding(out / "pilot.json"), "terminal_receipt": literal_binding(out / "receipt.json"),
            "cells": 18, "allocated_gpu_seconds": pilot["cost"]["allocated_gpu_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "readback"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "preflight":
        cpu_preflight(args.output)
    elif args.mode == "run":
        result = run(args.output, args.device)
        print(json.dumps({"status": result["status"], "cost": result["cost"]}))
    else:
        result = readback(args.output)
        _write_new(args.output / "cold-readback.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
