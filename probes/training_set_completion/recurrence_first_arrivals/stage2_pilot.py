"""Run exactly the lead-admitted finite first-arrivals pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList

from probes.training_set_completion.artifacts import literal_binding
from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.native_row_choice.runtime import _score_candidate, _trace_compare
from probes.training_set_completion.numerical_feedback.runtime import full_prefix
from probes.training_set_completion.numerical_feedback.select import rows, token_hash
from probes.training_set_completion.recurrence_first_arrivals.prepare import CENSUS, MATURE, _require
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import (
    EOS, OPEN, _Scores, _ThreeRows, _bound, _complete_row, _consumer_readback,
    _owner, _write_new,
)
from probes.training_set_completion.untied_shared import BASE, load_model
from src.artifacts.source_provenance import preserve_source
from src.data.geometry import iou_xyxy, parse_source_bbox_tokens
from src.qwen.input_identity import input_identity
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


REPO = Path(__file__).resolve().parents[3]
UNIT = REPO / "research/experiments/2026-09-23-recurrence-first-arrivals"
ADMISSION = UNIT / "lead-stage2-admission-v1.json"
ACCEPTANCE = UNIT / "lead-stage1-acceptance-v1.json"
SNAPSHOT = UNIT / "supporting/selection-stage2-frozen-v1.json"
CPU_RECEIPT = UNIT / "supporting/cpu-qualification-attempt-008.json"
SHA_ADMISSION = "bfa62ff34e865772892717b60b183847f6a60725bf94dc59304dc56d75eaab95"
SHA_ACCEPTANCE = "4243d352a4c4511c380dc92337163f33d8af62e9da8d7af42cc998b5f5f0e7fc"
SHA_SNAPSHOT = "435c333520f5f31078a44d2afb49d07a31c986d07198f9c8140cd8f3ddf47395"
OUTPUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage2-pilot")
CAP_SECONDS = 7200


def contract() -> tuple[dict, dict, dict, dict, list[tuple[dict, int, list[dict]]]]:
    for path, sha in ((ADMISSION, SHA_ADMISSION), (ACCEPTANCE, SHA_ACCEPTANCE), (SNAPSHOT, SHA_SNAPSHOT)):
        _require(literal_binding(path)["sha256"] == sha, f"frozen contract changed: {path}")
    admission = json.loads(ADMISSION.read_text())
    accepted = json.loads(ACCEPTANCE.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    for item in (admission["predecessor_acceptance"], admission["snapshot"],
                 admission["cpu_qualification"], accepted["candidate_report"],
                 *snapshot["source_files"]):
        path = Path(item["path"])
        if not path.is_absolute():
            path = REPO / path
        actual = literal_binding(path)
        _require(actual["sha256"] == item["sha256"] and actual["size_bytes"] == item["size_bytes"],
                 f"frozen source changed: {path}")
    for key in ("case", "terminal_receipt", "cold_readback", "cost_ledger"):
        _bound(Path(accepted[key]["path"]), accepted[key])
    _require(admission["status"] == "lead-admitted-finite-pilot" and
             admission["budget"]["incremental_pilot_ceiling_gpu_hours"] == 2 and
             admission["budget"]["package_ceiling_gpu_hours"] == 8, "pilot admission changed")
    _require(len(admission["cells"]) == 27 and len(admission["reused_cells"]) == 3 and
             len(admission["new_cells"]) == 24 and
             all(cell in snapshot["cells"] for cell in admission["cells"]), "frozen cell set changed")
    _require(all(cell in admission["cells"] for cell in admission["new_cells"] + admission["reused_cells"]),
             "admitted partition changed")
    grouped = defaultdict(list)
    for cell in admission["new_cells"]:
        grouped[cell["family"], int(cell["arrival_row"])].append(cell)
    families = {f["id"]: f for f in snapshot["families"] if f["id"] in admission["families"]}
    _require(set(families) == set(admission["families"]) and len(grouped) == 6, "family/landmark support changed")
    landmarks = []
    for family_id, row_index in sorted(grouped, key=lambda x: (admission["families"].index(x[0]), x[1])):
        family, cells = families[family_id], grouped[family_id, row_index]
        n_ids = [int(c["owner_id"]) for c in cells if c["mode"] == "complete_row_score"]
        _require(len(cells) == 1 + 2 * len(n_ids) and 1 <= len(n_ids) <= 2 and
                 sum(c["mode"] == "native_replay_and_sham" for c in cells) == 1, "landmark cell shape changed")
        _require(row_index in family["arrival_row_indices"] and len(set(n_ids)) == len(n_ids), "arrival or N changed")
        for owner_id in n_ids:
            c = next(z for z in cells if z["mode"] == "one_coordinate_supplied_branch" and z["owner_id"] == owner_id)
            candidate = next(z for z in family["candidates"] if z["owner_id"] == owner_id)
            row = family["row_evidence"][row_index]
            _require(c["coordinate_slot"] == 0 and c["native_token_id"] == 151670 + row["bbox"][0] and
                     c["supplied_token_id"] == 151670 + candidate["bbox"][0] and
                     c["native_token_id"] != c["supplied_token_id"] and
                     candidate["row_token_ids"][len(candidate["row_token_ids"]) - 5] == c["supplied_token_id"],
                     "frozen x1 slot/source token changed")
        for key in ("raw", "trace", "runtime_receipt", "image"):
            source = family["source_bindings"][key]
            _bound(Path(source["path"]), source)
        landmarks.append((family, row_index, cells))
    _require(sum(len(c) for _, _, c in landmarks) == 24, "new cell denominator changed")
    return admission, accepted, snapshot, families, landmarks


def source_boundary(family: dict) -> dict:
    source = family["source_bindings"]
    target = int(family["batch_index"])
    native = [int(x) for x in json.loads(Path(source["raw"]["path"]).read_text())["rows"][target]["token_ids"]]
    return {"group": family["group"], "batch_index": target, "image_id": family["image_id"],
            "raw_path": source["raw"]["path"], "trace_path": source["trace"]["path"],
            "receipt_path": source["runtime_receipt"]["path"], "native_tokens": native,
            "native_token_hash": token_hash(native)}


def source_panel(family: dict) -> dict:
    root = MATURE if family["source"] == "mature" else CENSUS
    return json.loads((root / "panel.json").read_text())


def _source_shapes(landmarks: list) -> list[dict]:
    q = load_qwen_components_from_options(QwenLoadOptions(
        base_model=str(BASE), dtype="fp32", attn_implementation="sdpa",
        patch_embed_linearization="enabled", load_model=False))
    _require(q.model is None, "CPU preflight loaded a model")
    out = []
    seen = set()
    for family, row_index, _cells in landmarks:
        key = (family["source"], family["group"])
        if key in seen:
            continue
        seen.add(key)
        batch, raw, trace, group, planning = _source(source_boundary(family), "untied", source_panel(family),
                                                     q, torch.device("cpu"))
        receipt = json.loads(Path(family["source_bindings"]["runtime_receipt"]["path"]).read_text())
        _require(input_identity(batch) == receipt["input_identity"], "CPU full source input changed")
        prefix = max(row["start"] for fam, index, _ in landmarks if
                     (fam["source"], fam["group"]) == key
                     for row in [fam["row_evidence"][index]])
        grid = [list(x) for x in batch.image_grids]
        out.append({"source": key[0], "group": key[1], "batch_size": len(raw),
                    "prompt_width": int(batch.inputs["input_ids"].shape[1]),
                    "image_grid_sum": sum(int(x[1]) * int(x[2]) for x in grid),
                    "image_grids": grid, "max_boundary_prefix": prefix,
                    "pixel_elements": int(batch.inputs["pixel_values"].numel()),
                    "replanned_image_plan": planning["replanned_image_plan"],
                    "input_identity_sha256": family["source_bindings"]["input_identity_sha256"]})
    _require(len(out) == 5, "five admitted source groups changed")
    return out


def preflight(output: Path) -> dict:
    admission, accepted, snapshot, _families, landmarks = contract()
    _require(not output.exists(), "Stage2 output attempt already exists")
    output.mkdir(parents=True)
    shapes = _source_shapes(landmarks)
    for family, row_index, cells in landmarks:
        boundary = source_boundary(family)
        parsed = rows(boundary["native_tokens"])
        row = parsed[row_index]
        _require(row["start"] == family["row_evidence"][row_index]["start"] and
                 row["end"] == family["row_evidence"][row_index]["end"], "CPU row offset changed")
        a_tokens = boundary["native_tokens"][row["start"]:row["end"]]
        _complete_row(a_tokens)
        for cell in cells:
            if cell["mode"] != "one_coordinate_supplied_branch":
                continue
            candidate = next(c for c in family["candidates"] if c["owner_id"] == cell["owner_id"])
            n_tokens = [int(x) for x in candidate["row_token_ids"]]
            _complete_row(n_tokens)
            j = int(cell["coordinate_slot"])
            _require(j == 0 and a_tokens[:row["coordinate_offsets"][j] - row["start"]] ==
                     n_tokens[:row["coordinate_offsets"][j] - row["start"]],
                     "A/N literal common prefix changed")
            _require(boundary["native_tokens"][row["coordinate_offsets"][j]] == cell["native_token_id"] and
                     n_tokens[row["coordinate_offsets"][j] - row["start"]] == cell["supplied_token_id"],
                     "CPU supplied slot changed")
    case = accepted["case"]
    old = json.loads(Path(case["path"]).read_text())
    old_shapes = {"batch_size": len(old["input_identity"]["request_ids"]),
                  "prompt_width": old["runs"][0]["input_width"] - 10,
                  "image_grid_sum": sum(int(x[1]) * int(x[2]) for x in old["input_identity"]["image_grids"]),
                  "max_boundary_prefix": 20}
    factor = max(1.0, *(max(shape["batch_size"] / old_shapes["batch_size"],
                          (shape["prompt_width"] + shape["max_boundary_prefix"])
                          / (old_shapes["prompt_width"] + old_shapes["max_boundary_prefix"]),
                          shape["image_grid_sum"] / old_shapes["image_grid_sum"])
                        for shape in shapes))
    conservative_forwards = admission["budget"]["forecast_without_native_sham_sharing"]["total_model_forwards"]
    estimated_seconds = conservative_forwards * old["cost"]["allocated_gpu_seconds"] / old["counters"]["model_forwards"] * 2 * factor
    _require(estimated_seconds < CAP_SECONDS and
             (admission["budget"]["spent_before_this_pilot_gpu_hours"] + estimated_seconds / 3600) < 8,
             "shape-aware cost forecast exceeds admitted caps")
    imports = [Path(__file__), REPO / "probes/training_set_completion/recurrence_first_arrivals/stage1_case.py",
               REPO / "probes/training_set_completion/coordinate_continuity/runtime.py",
               REPO / "probes/training_set_completion/native_row_choice/runtime.py",
               REPO / "probes/training_set_completion/numerical_feedback/runtime.py",
               REPO / "probes/training_set_completion/numerical_feedback/select.py",
               REPO / "probes/training_set_completion/untied_shared.py",
               REPO / "src/qwen/native.py", REPO / "src/qwen/generation.py",
               REPO / "src/qwen/untied_embeddings.py", REPO / "src/data/geometry.py",
               REPO / "src/artifacts/source_provenance.py"]
    captures = []
    for path in imports:
        name = path.relative_to(REPO)
        capture = preserve_source(path, run_root=output, relative_name=name)
        captures.append({"maintained": literal_binding(path), "capture": literal_binding(capture)})
    packet = {"schema": "recurrence_first_arrivals.stage2_preflight.v1", "status": "cpu_passed_frozen_before_gpu",
              "admission": literal_binding(ADMISSION), "acceptance": literal_binding(ACCEPTANCE),
              "selection": literal_binding(SNAPSHOT), "cpu_receipt": literal_binding(CPU_RECEIPT),
              "reused_stage1_case": accepted["case"], "reused_stage1_readback": accepted["cold_readback"],
              "new_cells": admission["new_cells"], "reused_cells": admission["reused_cells"],
              "source_shapes": shapes, "source_bindings": {f["id"]: f["source_bindings"] for f, _, _ in landmarks},
              "producer_imports": captures,
              "forecast": {"unshared_model_forwards": conservative_forwards,
                           "measured_case_seconds_per_forward_including_setup": old["cost"]["allocated_gpu_seconds"] / old["counters"]["model_forwards"],
                           "shape_factor": factor, "planning_multiplier": 2,
                           "estimated_incremental_allocated_gpu_seconds": estimated_seconds,
                           "estimated_incremental_allocated_gpu_hours": estimated_seconds / 3600,
                           "incremental_hard_cap_seconds": CAP_SECONDS,
                           "package_spent_before_hours": admission["budget"]["spent_before_this_pilot_gpu_hours"],
                           "package_hard_cap_hours": 8,
                           "interpretation": "planning estimate; runtime interval and hard caps govern"},
              "commands": {"gpu": ["python", "-B", "-m",
                                    "probes.training_set_completion.recurrence_first_arrivals.stage2_pilot",
                                    "run", "--output", str(output), "--device", "cuda:0"],
                           "readback": ["python", "-B", "-m",
                                        "probes.training_set_completion.recurrence_first_arrivals.stage2_pilot",
                                        "readback", "--output", str(output)]}}
    binding = _write_new(output / "preflight.json", packet)
    print(json.dumps({"status": packet["status"], "preflight": binding,
                      "forecast_hours": estimated_seconds / 3600, "shapes": shapes}))
    return packet


def _landmark_name(family: dict, row_index: int) -> str:
    return f"{family['source']}-{family['image_id']}-row{row_index}"


def _annotation(family: dict, group: dict) -> list[dict]:
    record = next(c["input_record"] for c in group["cases"] if c["input_record"]["image_id"] == family["image_id"])
    return [{"owner_id": int(obj["coco_ann_id"]), "description": obj["desc"],
             "bbox": list(parse_source_bbox_tokens(obj["bbox_2d"], field="bbox_2d"))}
            for obj in record["objects"]]


def _overlay(path: Path, image_path: Path, first: dict | None, annotation: list[dict], description: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    def box(values: list[int], color: str, label: str, width: int) -> None:
        xy = [round(v * (image.width if i % 2 == 0 else image.height) / 1000) for i, v in enumerate(values)]
        draw.rectangle(xy, outline=color, width=width)
        draw.text((xy[0], max(0, xy[1] - 14)), label, fill=color)
    for obj in annotation:
        if obj["description"] == description:
            box(obj["bbox"], "#00e070", str(obj["owner_id"]), 2)
    if first is not None and first["valid"]:
        box(first["values"], "#ff3030", "branch first row", 5)
    _require(not path.exists(), "branch overlay exists")
    image.save(path)
    return literal_binding(path)


def _summary(tokens: list[int], parsed: list[dict], reason: str) -> dict:
    return {"stop_reason": reason, "complete_rows": len(parsed),
            "malformed_openers": max(0, tokens.count(OPEN) - len(parsed)),
            "unparsed_token_count": len(tokens) - sum(r["end"] - r["start"] for r in parsed) - int(EOS in tokens),
            "eos": EOS in tokens, "cap": reason == "free_token_cap",
            "invalid_geometry_rows": sum(not r["valid"] for r in parsed)}


def _run(output: Path, device_name: str) -> dict:
    started = time.monotonic()
    admission, accepted, _snapshot, _families, landmarks = contract()
    preflight_path = output / "preflight.json"
    _require(preflight_path.is_file(), "Stage2 CPU preflight is absent")
    packet = json.loads(preflight_path.read_text())
    _require(packet["status"] == "cpu_passed_frozen_before_gpu" and
             packet["admission"] == literal_binding(ADMISSION) and
             packet["selection"] == literal_binding(SNAPSHOT) and
             packet["new_cells"] == admission["new_cells"], "preflight/admission changed")
    for item in packet["producer_imports"]:
        _bound(Path(item["maintained"]["path"]), item["maintained"])
        _bound(Path(item["capture"]["path"]), item["capture"])
    _require(device_name == "cuda:0" and not (output / "receipt.json").exists(), "pilot device/run changed")
    launch = _write_new(output / "launch.json", {"schema": "recurrence_first_arrivals.stage2_launch.v1",
                   "status": "frozen_before_model_load", "pid": os.getpid(), "device": device_name,
                   "started_unix": time.time(), "preflight": literal_binding(preflight_path),
                   "admission": literal_binding(ADMISSION), "producer": literal_binding(Path(__file__)),
                   "incremental_cap_seconds": CAP_SECONDS})
    device = torch.device(device_name)
    counters = {"model_forwards": 0, "vision_forwards": 0, "generation_paths": 0,
                "complete_row_score_forwards": 0}
    handles = []
    model = None
    completed = []
    active_capture = None
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        q, identity = load_model("untied", device)
        model = q.model.eval()
        reference = json.loads(Path(accepted["case"]["path"]).read_text())["effective_identity"]
        _require({k: v for k, v in identity.items() if k != "loader_source"} ==
                 {k: v for k, v in reference.items() if k != "loader_source"} and
                 identity["loader_source"]["sha256"] == reference["loader_source"]["sha256"],
                 "fresh Stage2 effective model identity differs from accepted case")

        def before_forward(_module, _args, kwargs):
            nonlocal active_capture
            counters["model_forwards"] += 1
            _require(time.monotonic() - started < CAP_SECONDS, "incremental GPU-hour cap reached")
            if active_capture is not None and active_capture.get("input_ids") is None:
                ids = kwargs.get("input_ids")
                _require(isinstance(ids, torch.Tensor), "model consumer lacks actual input_ids")
                active_capture["input_ids"] = ids.detach().cpu().clone()
                mask = kwargs.get("attention_mask")
                active_capture["attention_mask"] = mask.detach().cpu().clone() if isinstance(mask, torch.Tensor) else None

        def before_vision(_module, _args):
            counters["vision_forwards"] += 1

        handles.extend((model.register_forward_pre_hook(before_forward, with_kwargs=True),
                        model.model.visual.register_forward_pre_hook(before_vision)))
        source_cache = {}
        pad = int(q.tokenizer.pad_token_id)

        def generate(label: str, batch, raw, trace, native: list[int], row: dict,
                     target: int, end: int, supplied: int | None) -> dict:
            nonlocal active_capture
            mutation = None if supplied is None else (target, end - 1, native[end - 1], supplied)
            inputs = full_prefix(batch, raw, end, pad, device, mutation)
            expected = inputs["input_ids"].detach().cpu()
            prompt_width = int(batch.inputs["input_ids"].shape[1])
            _require(torch.equal(inputs["input_ids"][:, :prompt_width], batch.inputs["input_ids"]),
                     "original prompt/companion prefix changed")
            width = int(expected.shape[1])
            active_capture = {"input_ids": None, "attention_mask": None}
            stopper = _ThreeRows(target, prompt_width + row["start"], width)
            score_capture = _Scores(target=target, width=width,
                                    start=end if label == "native" else None,
                                    trace=trace if label == "native" else None)
            config = GenerationConfig(max_new_tokens=256, do_sample=False,
                                      repetition_penalty=1.0, eos_token_id=EOS, pad_token_id=pad)
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=config, use_model_defaults=False,
                                           logits_processor=LogitsProcessorList([score_capture]),
                                           stopping_criteria=StoppingCriteriaList([stopper]))
            torch.cuda.synchronize(device)
            observed = active_capture["input_ids"]
            _require(isinstance(observed, torch.Tensor), "actual model input capture absent")
            _consumer_readback(observed, expected)
            _require(active_capture["attention_mask"] is not None and
                     torch.equal(active_capture["attention_mask"], inputs["attention_mask"].cpu()),
                     "actual consumer attention changed")
            falsification = {}
            if supplied is not None:
                for name, altered in (("dropped", observed[:, :-1].clone()),
                                      ("replaced", observed.clone())):
                    if name == "replaced":
                        altered[target, -1] = native[end - 1]
                    try:
                        _consumer_readback(altered, expected)
                    except ValueError:
                        falsification[name] = "rejected"
                    else:
                        raise ValueError("consumer input verifier accepted changed supplied token")
                _require(int(observed[target, -1]) == supplied, "actual consumer did not receive supplied N")
            free = [int(x) for x in generated[target, width:].cpu().tolist()]
            common = native[row["start"]:end]
            if supplied is not None:
                common = [*common[:-1], supplied]
            combined = common + free
            if EOS in combined:
                combined = combined[:combined.index(EOS) + 1]
            parsed = rows(combined)
            counters["generation_paths"] += 1
            active_capture = None
            return {"mode": label, "consumer_input_ids": observed.tolist(),
                    "consumer_attention_mask": inputs["attention_mask"].cpu().tolist(),
                    "consumer_falsification": falsification,
                    "prefix_end": end, "input_width": width,
                    "free_token_ids": free, "combined_row_start_token_ids": combined,
                    "parsed_rows": parsed, "summary": _summary(combined, parsed, stopper.reason or "generation_complete"),
                    "first_decisions": score_capture.first_decisions,
                    "source_trace_parity": score_capture.parity,
                    "companion_free_token_ids": [[int(x) for x in generated[i, width:].cpu().tolist()]
                                                 for i in range(len(raw)) if i != target]}

        for family, row_index, cells in landmarks:
            _require(time.monotonic() - started < CAP_SECONDS, "incremental GPU-hour cap reached before landmark")
            name = _landmark_name(family, row_index)
            path = output / "landmarks" / name
            path.mkdir(parents=True, exist_ok=False)
            key = (family["source"], family["group"])
            if key not in source_cache:
                batch, raw, trace, group, planning = _source(source_boundary(family), "untied",
                                                             source_panel(family), q, device)
                receipt = json.loads(Path(family["source_bindings"]["runtime_receipt"]["path"]).read_text())
                _require(input_identity(batch) == receipt["input_identity"], "original full-batch source changed")
                source_cache[key] = batch, raw, trace, group, planning
            batch, raw, trace, group, planning = source_cache[key]
            target = int(family["batch_index"])
            native = [int(x) for x in raw[target]["token_ids"]]
            row = rows(native)[row_index]
            a_tokens = native[row["start"]:row["end"]]
            _complete_row(a_tokens)
            candidates = [next(c for c in family["candidates"] if c["owner_id"] == z["owner_id"])
                          for z in cells if z["mode"] == "complete_row_score"]
            score_rows = {}
            parity = []
            a_logits = None
            common_logit_count = row["coordinate_offsets"][0] - row["start"] + 1
            for label, tokens in [("A", a_tokens), *[(str(c["owner_id"]), list(c["row_token_ids"])) for c in candidates]]:
                scored = _score_candidate(model=model, batch=batch, raw=raw, target=target,
                                          prefix=native[:row["start"]], tokens=tokens, pad=pad, device=device)
                counters["complete_row_score_forwards"] += 1
                logits = scored["action_logits"].double()
                selected = logits[torch.arange(len(tokens)), torch.tensor(tokens)]
                logp = selected - torch.logsumexp(logits, dim=-1)
                score_rows[label] = {"token_ids": tokens, "token_logprobs_fp64": logp.tolist(),
                                     "row_logprob_fp64": math.fsum(logp.tolist()),
                                     "position_ids": scored["positions"],
                                     "prefix_sha256": scored["prefix_sha256"],
                                     "input_ids_sha256": scored["input_ids_sha256"]}
                if label == "A":
                    a_logits = scored["action_logits"][:common_logit_count].clone()
                    parity = [_trace_compare(logits=scored["action_logits"][i], trace=trace,
                                             batch_index=target, absolute_offset=row["start"] + i,
                                             token_id=token, role="native_score")
                              for i, token in enumerate(tokens)]
                else:
                    _require(torch.allclose(scored["action_logits"][:common_logit_count], a_logits, atol=2e-4, rtol=0),
                             "A/N common-prefix logits changed")
            _require(len(parity) == len(a_tokens) and all(z["passed"] for z in parity),
                     "actual A full-row score/source parity failed")
            dropped = False
            try:
                _complete_row(a_tokens[:-1])
            except ValueError:
                dropped = True
            shifted = not _trace_compare(logits=a_logits[0], trace=trace, batch_index=target,
                                         absolute_offset=row["start"] + 1, token_id=a_tokens[0],
                                         role="shifted_slot")["passed"]
            _require(dropped and shifted, "score-path mutation falsification failed")
            score_raw = _write_new(path / "scores-raw.json", {"A_and_N": score_rows,
                        "A_trace_parity": parity, "dropped_terminator_rejected": dropped,
                        "shifted_slot_rejected": shifted})
            native_run = generate("native", batch, raw, trace, native, row, target, row["start"], None)
            native_raw = _write_new(path / "native-raw.json", native_run)
            _require(native_run["combined_row_start_token_ids"] ==
                     native[row["start"]:row["start"] + len(native_run["combined_row_start_token_ids"])],
                     "native IDs differ from saved source")
            _require(all(z["passed"] for z in native_run["source_trace_parity"]),
                     "native logits differ from saved source")
            end = row["coordinate_offsets"][0] + 1
            sham_run = generate("A_sham", batch, raw, trace, native, row, target, end, None)
            sham_raw = _write_new(path / "sham-raw.json", sham_run)
            _require(sham_run["combined_row_start_token_ids"] == native_run["combined_row_start_token_ids"],
                     "separate supplied-A sham differs from native")
            annotations = _annotation(family, group)
            branches = []
            for candidate in candidates:
                owner_id = int(candidate["owner_id"])
                c = next(z for z in cells if z["mode"] == "one_coordinate_supplied_branch" and
                         z["owner_id"] == owner_id)
                branch = generate(f"N_{owner_id}", batch, raw, trace, native, row, target, end,
                                  int(c["supplied_token_id"]))
                raw_binding = _write_new(path / f"N-{owner_id}-raw.json", branch)
                first = branch["parsed_rows"][0] if branch["parsed_rows"] else None
                owner = _owner(first, annotations, q.tokenizer)
                overlay = _overlay(path / f"N-{owner_id}-first-row.png",
                                   Path(family["source_bindings"]["image"]["path"]),
                                   first, annotations, family["description"])
                free_coordinates = sum(offset >= end - row["start"] for offset in first["coordinate_offsets"]) if first else 0
                branches.append({"candidate_owner_id": owner_id, "cell": c, "raw": raw_binding,
                                 "overlay": overlay, "owner_proxy": owner,
                                 "D_N_minus_A_fp64": score_rows[str(owner_id)]["row_logprob_fp64"]
                                                      - score_rows["A"]["row_logprob_fp64"],
                                 "deadband_nats": max(.01, .0004 * (len(a_tokens) + len(candidate["row_token_ids"]))),
                                 "coordinate_credit": {"supplied": 1, "free_first_row": free_coordinates},
                                 "summary": branch["summary"]})
            outcome = {"schema": "recurrence_first_arrivals.stage2_landmark.v1", "status": "candidate_complete",
                       "family": family["id"], "arrival_row": row_index, "cells": cells,
                       "source": family["source_bindings"], "group": family["group"], "batch_index": target,
                       "input_identity": input_identity(batch), "planning": planning,
                       "score_raw": score_raw, "native_raw": native_raw, "sham_raw": sham_raw,
                       "branches": branches, "native_summary": native_run["summary"],
                       "sham_summary": sham_run["summary"],
                       "score_A_trace_max_error": max(max(z["chosen_logit_abs_error"], z["logsumexp_abs_error"],
                                                           z["logprob_abs_error"], z["top2_max_abs_error"])
                                                        for z in parity),
                       "native_trace_max_error": max(max(z["chosen_logit_abs_error"], z["logsumexp_abs_error"],
                                                          z["logprob_abs_error"], z["top2_max_abs_error"])
                                                       for z in native_run["source_trace_parity"]),
                       "counters_after": dict(counters), "elapsed_gpu_seconds_after": time.monotonic() - started}
            result_binding = _write_new(path / "landmark.json", outcome)
            completed.append({"family": family["id"], "arrival_row": row_index, "cells": cells,
                              "landmark": result_binding})
            completed_pairs = sum((len(item["cells"]) - 1) // 2 for item in completed)
            remaining_pairs = 9 - completed_pairs
            elapsed = time.monotonic() - started
            static_remaining = packet["forecast"]["estimated_incremental_allocated_gpu_seconds"] * remaining_pairs / 9
            observed_remaining = elapsed * remaining_pairs / completed_pairs
            remaining_forecast = max(static_remaining, observed_remaining)
            _write_new(path / "receipt.json", {"status": "candidate_complete", "terminal": True,
                       "landmark": result_binding, "cells": cells,
                       "allocated_gpu_seconds_after": elapsed})
            _write_new(output / f"checkpoint-{len(completed):02d}.json",
                       {"completed": completed, "counters": counters,
                        "allocated_gpu_seconds": elapsed, "remaining_pairs": remaining_pairs,
                        "remaining_forecast_seconds": remaining_forecast,
                        "remaining_cell_count": 24 - sum(len(item["cells"]) for item in completed)})
            _require(elapsed + remaining_forecast < CAP_SECONDS,
                     "remaining work no longer fits incremental GPU-hour cap")
        _require(sum(len(x["cells"]) for x in completed) == 24, "Stage2 new cell count incomplete")
        cost = {"allocated_device": device_name, "allocated_gpu_seconds": time.monotonic() - started,
                "wall_seconds": time.monotonic() - started,
                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device), **counters}
        _require(cost["allocated_gpu_seconds"] < CAP_SECONDS, "incremental GPU-hour cap exceeded")
        result = {"schema": "recurrence_first_arrivals.stage2_pilot.v1", "status": "candidate_complete",
                  "admission": literal_binding(ADMISSION), "snapshot": literal_binding(SNAPSHOT),
                  "preflight": literal_binding(preflight_path), "effective_identity": identity,
                  "reused_cells": admission["reused_cells"], "reused_case": accepted["case"],
                  "new_cells": admission["new_cells"], "completed": completed,
                  "denominators": admission["denominators"], "cost": cost}
        result_binding = _write_new(output / "pilot.json", result)
        terminal = {"schema": "recurrence_first_arrivals.stage2_receipt.v1", "status": "candidate_complete",
                    "terminal": True, "launch": launch, "pilot": result_binding,
                    "completed_cells": 24, "reused_cells": 3, "cost": cost,
                    "artifact_bytes_before_receipt": sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
    except BaseException as error:
        terminal = {"schema": "recurrence_first_arrivals.stage2_receipt.v1", "status": "technical_invalid",
                    "terminal": True, "launch": launch, "error": repr(error), "traceback": traceback.format_exc(),
                    "completed": completed, "completed_cells": sum(len(x["cells"]) for x in completed),
                    "cost": {"allocated_device": device_name, "allocated_gpu_seconds": time.monotonic() - started,
                             "wall_seconds": time.monotonic() - started,
                             "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                             "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                             **counters},
                    "artifact_bytes_before_receipt": sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
    finally:
        for handle in handles:
            handle.remove()
    _write_new(output / "receipt.json", terminal)
    if terminal["status"] != "candidate_complete":
        raise RuntimeError(terminal["error"])
    return terminal


def readback(output: Path) -> dict:
    admission, accepted, _snapshot, _families, landmarks = contract()
    preflight = json.loads((output / "preflight.json").read_text())
    _require(preflight["status"] == "cpu_passed_frozen_before_gpu", "preflight changed")
    for item in preflight["producer_imports"]:
        _bound(Path(item["capture"]["path"]), item["capture"])
    receipt = json.loads((output / "receipt.json").read_text())
    _require(receipt["status"] == "candidate_complete" and receipt["terminal"] and
             receipt["completed_cells"] == 24 and receipt["reused_cells"] == 3,
             "pilot has no terminal complete receipt")
    _bound(Path(receipt["pilot"]["path"]), receipt["pilot"])
    pilot = json.loads((output / "pilot.json").read_text())
    _require(pilot["new_cells"] == admission["new_cells"] and
             pilot["reused_cells"] == admission["reused_cells"] and len(pilot["completed"]) == 6,
             "cold cell partition changed")
    _bound(Path(accepted["case"]["path"]), accepted["case"])
    _bound(Path(accepted["cold_readback"]["path"]), accepted["cold_readback"])
    seen = []
    for family, row_index, cells in landmarks:
        entry = next(x for x in pilot["completed"] if x["family"] == family["id"] and
                     x["arrival_row"] == row_index)
        _require(entry["cells"] == cells, "cold landmark cells changed")
        _bound(Path(entry["landmark"]["path"]), entry["landmark"])
        item = json.loads(Path(entry["landmark"]["path"]).read_text())
        _require(item["source"] == family["source_bindings"], "cold landmark source changed")
        for key in ("score_raw", "native_raw", "sham_raw"):
            _bound(Path(item[key]["path"]), item[key])
        score = json.loads(Path(item["score_raw"]["path"]).read_text())
        native = json.loads(Path(item["native_raw"]["path"]).read_text())
        sham = json.loads(Path(item["sham_raw"]["path"]).read_text())
        _require(all(x["passed"] for x in score["A_trace_parity"]) and
                 score["dropped_terminator_rejected"] and score["shifted_slot_rejected"],
                 "cold A score/source or mutation gate failed")
        _require(native["combined_row_start_token_ids"] == sham["combined_row_start_token_ids"] and
                 all(x["passed"] for x in native["source_trace_parity"]), "cold native/sham gate failed")
        _require(len(item["branches"]) == (len(cells) - 1) // 2, "cold branch count changed")
        for branch in item["branches"]:
            for key in ("raw", "overlay"):
                _bound(Path(branch[key]["path"]), branch[key])
            raw = json.loads(Path(branch["raw"]["path"]).read_text())
            _require(raw["consumer_falsification"] == {"dropped": "rejected", "replaced": "rejected"} and
                     raw["consumer_input_ids"][int(family["batch_index"])][-1] ==
                     branch["cell"]["supplied_token_id"], "cold actual supplied input gate failed")
            label = str(branch["candidate_owner_id"])
            _require(math.isclose(branch["D_N_minus_A_fp64"],
                                    score["A_and_N"][label]["row_logprob_fp64"]
                                    - score["A_and_N"]["A"]["row_logprob_fp64"], abs_tol=1e-12),
                     "cold finite score changed")
        seen.extend(cells)
    _require(seen == admission["new_cells"], "cold new cell ordering/denominator changed")
    _require(receipt["cost"]["allocated_gpu_seconds"] < CAP_SECONDS, "cold pilot cap failed")
    return {"schema": "recurrence_first_arrivals.stage2_readback.v1", "status": "passed",
            "pilot": literal_binding(output / "pilot.json"), "receipt": literal_binding(output / "receipt.json"),
            "new_cells": len(seen), "reused_cells": 3,
            "allocated_gpu_seconds": receipt["cost"]["allocated_gpu_seconds"],
            "artifact_bytes_before_readback": sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "readback"))
    parser.add_argument("--output", type=Path, default=OUTPUT / "attempt-001")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.output)
    elif args.mode == "run":
        result = _run(args.output, args.device)
        print(json.dumps({"status": result["status"], "cost": result["cost"]}))
    else:
        result = readback(args.output)
        _write_new(args.output / "cold-readback.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
