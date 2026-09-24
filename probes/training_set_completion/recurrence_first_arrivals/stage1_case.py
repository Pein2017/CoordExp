"""One admitted bowl qualification through the original full-batch model caller."""

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
from PIL import Image, ImageDraw
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from probes.training_set_completion.artifacts import literal_binding, write_pretty_json
from probes.training_set_completion.coordinate_continuity.runtime import _source
from probes.training_set_completion.native_row_choice.runtime import _score_candidate, _trace_compare
from probes.training_set_completion.numerical_feedback.runtime import full_prefix
from probes.training_set_completion.numerical_feedback.select import rows, token_hash
from probes.training_set_completion.recurrence_first_arrivals.prepare import MATURE, _require
from probes.training_set_completion.untied_shared import load_model
from src.data.geometry import iou_xyxy, parse_source_bbox_tokens
from src.qwen.input_identity import input_identity


UNIT = Path(__file__).resolve().parents[3] / "research/experiments/2026-09-23-recurrence-first-arrivals"
ADMISSION = UNIT / "lead-stage0-admission-v1.json"
SNAPSHOT = UNIT / "supporting/selection-attempt-004.json"
ADMISSION_SHA = "146605205625078300ef00016c8e1f715e9f78b4afa7edf4282d61a32a44b555"
SNAPSHOT_SHA = "325dfcd67c3ef21c717e80686da92e9020a4fc151b98d23ef1217a15363e6eeb"
EOS, OPEN, END = 151645, 151646, 151649
CASE_ID = "mature:313465:0"
ROW = 1
N_OWNER = 715278


def _bound(path: Path, expected: dict) -> None:
    actual = literal_binding(path)
    _require(all(actual.get(key) == value for key, value in expected.items()), f"binding changed: {path}")


def _write_new(path: Path, value: dict) -> dict:
    _require(not path.exists(), f"artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return literal_binding(path)


def _contract() -> tuple[dict, dict, dict, dict, list[dict]]:
    _require(literal_binding(ADMISSION)["sha256"] == ADMISSION_SHA, "lead admission changed")
    _require(literal_binding(SNAPSHOT)["sha256"] == SNAPSHOT_SHA, "admitted snapshot changed")
    admission = json.loads(ADMISSION.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    gate = admission["stage1_qualification"]
    _require(gate["status"] == "lead-admitted-single-case-only" and
             gate["family"] == CASE_ID and gate["arrival_row"] == ROW and
             gate["candidate_owner_id"] == N_OWNER and gate["target_owner_id"] == 716308 and
             gate["source_group"] == "fresh-18" and gate["batch_index"] == 3 and
             gate["allocated_gpu_hour_ceiling"] == 1, "case admission changed")
    _bound(SNAPSHOT, gate["snapshot"])
    expected = [cell for cell in snapshot["cells"] if cell["family"] == CASE_ID and cell["arrival_row"] == ROW]
    _require(expected == gate["cells"] and len(expected) == 3, "three admitted cells differ from snapshot")
    family = next(f for f in snapshot["families"] if f["id"] == CASE_ID)
    candidate = next(c for c in family["candidates"] if c["owner_id"] == N_OWNER)
    _require(family["group"] == "fresh-18" and family["batch_index"] == 3 and
             family["owner_id"] == 716308 and family["arrival_row_indices"] == [0, 1],
             "frozen bowl family changed")
    branch = expected[2]
    _require(branch["mode"] == "one_coordinate_supplied_branch" and branch["coordinate_slot"] == 0 and
             branch["native_token_id"] == 151675 and branch["supplied_token_id"] == 151831 and
             candidate["row_token_ids"][5] == 151831, "admitted x1 branch changed")
    for key in ("raw", "trace", "runtime_receipt", "image"):
        _bound(Path(family["source_bindings"][key]["path"]), family["source_bindings"][key])
    return admission, snapshot, family, candidate, expected


def _complete_row(tokens: list[int]) -> dict:
    parsed = rows(tokens)
    _require(len(parsed) == 1 and parsed[0]["start"] == 0 and parsed[0]["end"] == len(tokens)
             and tokens[0] == OPEN and tokens[-1] == END, "complete scored row grammar changed")
    return parsed[0]


def _consumer_readback(observed: torch.Tensor, expected: torch.Tensor) -> None:
    _require(observed.shape == expected.shape and torch.equal(observed, expected),
             "actual model consumer input differs from bound full prefix")


class _ThreeRows(StoppingCriteria):
    def __init__(self, target: int, row_start: int, width: int) -> None:
        self.target, self.row_start, self.width = target, row_start, width
        self.reason: str | None = None

    def __call__(self, input_ids: torch.Tensor, _scores, **_kwargs) -> bool:
        suffix = [int(x) for x in input_ids[self.target, self.row_start:].tolist()]
        if EOS in suffix:
            self.reason = "eos"
        elif len(rows(suffix)) >= 3:
            self.reason = "current_plus_two_complete_rows"
        elif input_ids.shape[1] - self.width >= 256:
            self.reason = "free_token_cap"
        return self.reason is not None


class _Scores(LogitsProcessor):
    def __init__(self, *, target: int, width: int, start: int | None, trace: dict | None) -> None:
        self.target, self.width, self.start, self.trace = target, width, start, trace
        self.parity: list[dict] = []
        self.first_decisions: list[dict] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = int(input_ids.shape[1] - self.width)
        logit = scores[self.target].detach().float()
        if self.trace is not None and self.start is not None:
            offset = self.start + step
            token = int(self.trace["steps"][offset]["chosen"][self.target])
            check = _trace_compare(logits=logit, trace=self.trace, batch_index=self.target,
                                   absolute_offset=offset, token_id=token, role="native_free")
            self.parity.append(check)
        if step < 12:
            values, ids = torch.topk(logit, 2)
            self.first_decisions.append({"free_step": step, "top2_ids": ids.cpu().tolist(),
                                         "top2_logits": values.cpu().tolist(),
                                         "logsumexp": float(torch.logsumexp(logit, dim=-1).item())})
        return scores


def _owner(row: dict | None, annotation: list[dict], tokenizer) -> dict:
    if row is None:
        return {"verdict": "no_complete_row", "same_class_matches": []}
    description = tokenizer.decode(row["description_tokens"], clean_up_tokenization_spaces=False)
    matching = sorted(({"owner_id": obj["owner_id"], "bbox": obj["bbox"],
                        "iou": iou_xyxy(row["values"], obj["bbox"])}
                       for obj in annotation if obj["description"] == description),
                      key=lambda x: (-x["iou"], x["owner_id"]))
    hits = [x for x in matching if x["iou"] >= .5]
    verdict = ("unique_iou50_proxy" if len(hits) == 1 else
               "ambiguous_iou50_proxy" if len(hits) > 1 else "no_iou50_proxy")
    return {"verdict": verdict, "description": description,
            "owner_id": hits[0]["owner_id"] if len(hits) == 1 else None,
            "iou80": bool(len(hits) == 1 and hits[0]["iou"] >= .8),
            "same_class_matches": matching, "row": row}


def _overlay(path: Path, image_path: Path, row: dict | None, annotation: list[dict]) -> dict:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for obj in annotation:
        if obj["description"] != "bowl":
            continue
        xy = [round(value * (image.width if i % 2 == 0 else image.height) / 1000)
              for i, value in enumerate(obj["bbox"])]
        draw.rectangle(xy, outline="#00e070", width=3)
        draw.text((xy[0], max(0, xy[1] - 14)), str(obj["owner_id"]), fill="#00e070")
    if row is not None and row["valid"]:
        xy = [round(value * (image.width if i % 2 == 0 else image.height) / 1000)
              for i, value in enumerate(row["values"])]
        draw.rectangle(xy, outline="#ff3030", width=5)
        draw.text((xy[0], max(0, xy[1] - 25)), "first free branch row", fill="#ff3030")
    _require(not path.exists(), "overlay already exists")
    image.save(path)
    return literal_binding(path)


def run(output: Path, device_name: str) -> dict:
    started = time.monotonic()
    started_utc = time.time()
    admission, _snapshot, family, candidate, cells = _contract()
    _require(device_name.startswith("cuda:"), "Stage1 requires a single CUDA device")
    _require(not output.exists(), "attempt output already exists")
    output.mkdir(parents=True)
    launch = {"schema": "recurrence_first_arrivals.stage1_launch.v1", "status": "frozen_before_model_load",
              "pid": os.getpid(), "started_unix": started_utc, "device": device_name,
              "admission": literal_binding(ADMISSION), "snapshot": literal_binding(SNAPSHOT),
              "producer": literal_binding(Path(__file__)), "source_bindings": family["source_bindings"],
              "cells": cells, "budget_gpu_seconds": 3600}
    launch_binding = _write_new(output / "launch.json", launch)
    counters = {"model_forwards": 0, "vision_forwards": 0, "generation_paths": 0}
    handles = []
    model = None
    raw_binding = None
    try:
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.empty(1, device=device)  # initialize the CUDA allocator before resetting peak counters
        torch.cuda.reset_peak_memory_stats(device)
        q, identity = load_model("untied", device)
        model = q.model.eval()
        source = family["source_bindings"]
        receipt = json.loads(Path(source["runtime_receipt"]["path"]).read_text())
        saved_identity = receipt["identity"]
        saved_loader = saved_identity["loader_source"]
        fresh_loader = identity["loader_source"]
        _require({k: v for k, v in identity.items() if k != "loader_source"} ==
                 {k: v for k, v in saved_identity.items() if k != "loader_source"} and
                 all(saved_loader[k] == fresh_loader[k] for k in ("sha256", "size_bytes")),
                 "fresh adapter/independent embedding or loader bytes differ from native source")
        identity_crosswalk = {"saved_loader_source": saved_loader, "fresh_maintained_loader_source": fresh_loader,
                              "only_loader_path_differs": saved_loader["path"] != fresh_loader["path"]}
        panel = json.loads((MATURE / "panel.json").read_text())
        raw_source = json.loads(Path(source["raw"]["path"]).read_text())["rows"][3]
        native = [int(x) for x in raw_source["token_ids"]]
        boundary = {"group": "fresh-18", "batch_index": 3, "image_id": 313465,
                    "raw_path": source["raw"]["path"], "trace_path": source["trace"]["path"],
                    "receipt_path": source["runtime_receipt"]["path"],
                    "native_tokens": native, "native_token_hash": token_hash(native)}
        batch, raw, trace, group, planning = _source(boundary, "untied", panel, q, device)
        _require(input_identity(batch) == receipt["input_identity"], "full native prompt/media identity changed")
        _require(len(raw) == len(group["cases"]) and raw[3]["token_ids"] == native,
                 "full source batch/target changed")
        target = 3
        row = rows(native)[ROW]
        _require(row["start"] == 10 and row["end"] == 20 and row["values"][0] == 5 and
                 native[row["coordinate_offsets"][0]] == 151675, "native row/slot changed")
        a_tokens = native[row["start"]:row["end"]]
        n_tokens = [int(x) for x in candidate["row_token_ids"]]
        _complete_row(a_tokens)
        _complete_row(n_tokens)
        _require(len(a_tokens) == len(n_tokens) == 10 and a_tokens[:5] == n_tokens[:5]
                 and a_tokens[5] == 151675 and n_tokens[5] == 151831,
                 "admitted A/N literal prefix changed")

        capture: dict | None = None

        def before_forward(_module, _args, kwargs):
            nonlocal capture
            counters["model_forwards"] += 1
            _require(time.monotonic() - started < 3600, "one-case allocated GPU-hour ceiling reached")
            if capture is not None and capture.get("input_ids") is None:
                ids = kwargs.get("input_ids")
                _require(isinstance(ids, torch.Tensor), "real generation consumer lacks input_ids")
                capture["input_ids"] = ids.detach().cpu().clone()
                mask = kwargs.get("attention_mask")
                capture["attention_mask"] = mask.detach().cpu().clone() if isinstance(mask, torch.Tensor) else None

        def before_vision(_module, _args):
            counters["vision_forwards"] += 1

        handles.append(model.register_forward_pre_hook(before_forward, with_kwargs=True))
        handles.append(model.model.visual.register_forward_pre_hook(before_vision))
        pad = int(q.tokenizer.pad_token_id)
        score_rows = {}
        score_parity = []
        for name, tokens in (("A", a_tokens), ("N", n_tokens)):
            scored = _score_candidate(model=model, batch=batch, raw=raw, target=target,
                                      prefix=native[:row["start"]], tokens=tokens,
                                      pad=pad, device=device)
            logits = scored["action_logits"].double()
            selected = logits[torch.arange(len(tokens)), torch.tensor(tokens)]
            logp = selected - torch.logsumexp(logits, dim=-1)
            score_rows[name] = {"token_ids": tokens, "token_logprobs_fp64": logp.tolist(),
                                "row_logprob_fp64": math.fsum(logp.tolist()),
                                "position_ids": scored["positions"],
                                "prefix_sha256": scored["prefix_sha256"],
                                "input_ids_sha256": scored["input_ids_sha256"]}
            if name == "A":
                score_parity = [_trace_compare(logits=scored["action_logits"][i], trace=trace,
                                               batch_index=target, absolute_offset=row["start"] + i,
                                               token_id=token, role="native_score")
                                for i, token in enumerate(tokens)]
            if name == "N":
                _require(torch.allclose(scored["action_logits"][:6], a_logits[:6], atol=2e-4, rtol=0),
                         "A/N common-prefix logits differ")
            else:
                a_logits = scored["action_logits"][:6].clone()
        _require(len(score_parity) == 10 and all(x["passed"] for x in score_parity),
                 "actual A complete-row score differs from source including entry/terminator")
        dropped = False
        shifted = False
        try:
            _complete_row(a_tokens[:-1])
        except ValueError:
            dropped = True
        shifted = not _trace_compare(logits=a_logits[0], trace=trace, batch_index=target,
                                     absolute_offset=row["start"] + 1,
                                     token_id=a_tokens[0], role="shifted_slot")["passed"]
        _require(dropped and shifted, "actual score path did not reject dropped terminator/shifted slot")

        prompt_width = int(batch.inputs["input_ids"].shape[1])

        def generate(label: str, end: int, supplied: bool) -> dict:
            nonlocal capture
            mutation = (target, end - 1, 151675, 151831) if supplied else None
            inputs = full_prefix(batch, raw, end, pad, device, mutation)
            expected = inputs["input_ids"].detach().cpu()
            _require(torch.equal(inputs["input_ids"][:, :prompt_width], batch.inputs["input_ids"]),
                     "consumer prompt differs from original full batch")
            capture = {"input_ids": None, "attention_mask": None}
            width = int(expected.shape[1])
            stopper = _ThreeRows(target, prompt_width + row["start"], width)
            scores = _Scores(target=target, width=width,
                             start=end if label == "native" else None,
                             trace=trace if label == "native" else None)
            config = GenerationConfig(max_new_tokens=256, do_sample=False,
                                      repetition_penalty=1.0, eos_token_id=EOS, pad_token_id=pad)
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=config,
                                           use_model_defaults=False,
                                           logits_processor=LogitsProcessorList([scores]),
                                           stopping_criteria=StoppingCriteriaList([stopper]))
            torch.cuda.synchronize(device)
            observed = capture["input_ids"]
            _require(isinstance(observed, torch.Tensor), "actual generation input was not captured")
            _consumer_readback(observed, expected)
            _require(capture["attention_mask"] is not None and
                     torch.equal(capture["attention_mask"], inputs["attention_mask"].cpu()),
                     "real consumer attention mask changed")
            falsification = {}
            if supplied:
                for test, altered in (("dropped", observed[:, :-1].clone()),
                                      ("replaced", observed.clone())):
                    if test == "replaced":
                        altered[target, -1] = 151675
                    try:
                        _consumer_readback(altered, expected)
                    except ValueError:
                        falsification[test] = "rejected"
                    else:
                        raise ValueError(f"actual consumer {test} supplied-token mutation passed")
                _require(int(observed[target, -1]) == 151831,
                         "actual model consumer did not receive supplied N x1")
            target_free = [int(x) for x in generated[target, width:].cpu().tolist()]
            combined = (native[row["start"]:end] if not supplied else
                        [*native[row["start"]:end - 1], 151831]) + target_free
            if EOS in combined:
                combined = combined[:combined.index(EOS) + 1]
            parsed = rows(combined)
            counters["generation_paths"] += 1
            return {"mode": label, "consumer_input_ids": observed.tolist(),
                    "consumer_attention_mask": capture["attention_mask"].tolist(),
                    "consumer_mutation_falsification": falsification,
                    "input_width": width, "prefix_end": end,
                    "free_token_ids": target_free, "combined_row_start_token_ids": combined,
                    "parsed_rows": parsed, "stop_reason": stopper.reason or "generation_complete",
                    "free_token_count": len(target_free), "first_decisions": scores.first_decisions,
                    "source_trace_parity": scores.parity,
                    "companion_free_token_ids": [[int(x) for x in generated[i, width:].cpu().tolist()]
                                                 for i in range(len(raw)) if i != target]}

        native_run = generate("native", row["start"], False)
        _require(native_run["combined_row_start_token_ids"] ==
                 native[row["start"]:row["start"] + len(native_run["combined_row_start_token_ids"])],
                 "native generated IDs differ from saved source")
        _require(all(x["passed"] for x in native_run["source_trace_parity"]),
                 "native free logits differ from saved source")
        end = row["coordinate_offsets"][0] + 1
        sham_run = generate("A_sham", end, False)
        _require(sham_run["combined_row_start_token_ids"] == native_run["combined_row_start_token_ids"],
                 "separate supplied-A sham differs from native continuation")
        branch_run = generate("N_x1", end, True)
        raw_binding = _write_new(output / "raw-evidence.json", {
            "schema": "recurrence_first_arrivals.stage1_raw_evidence.v1",
            "status": "generated_before_owner_reduction",
            "admission": literal_binding(ADMISSION), "snapshot": literal_binding(SNAPSHOT),
            "source": source, "score_rows": score_rows, "score_A_trace_parity": score_parity,
            "score_falsification": {"dropped_terminator_rejected": dropped,
                                    "shifted_slot_rejected": shifted},
            "runs": [native_run, sham_run, branch_run], "counters": counters})
        first = branch_run["parsed_rows"][0] if branch_run["parsed_rows"] else None
        record = next(c["input_record"] for c in group["cases"] if c["input_record"]["image_id"] == 313465)
        annotation = [{"owner_id": int(obj["coco_ann_id"]), "description": obj["desc"],
                       "bbox": list(parse_source_bbox_tokens(obj["bbox_2d"], field="bbox_2d"))}
                      for obj in record["objects"]]
        owner = _owner(first, annotation, q.tokenizer)
        free_first_coordinates = (sum(end - row["start"] <= offset <
                                      end - row["start"] + len(branch_run["free_token_ids"])
                                      for offset in first["coordinate_offsets"])
                                  if first is not None else 0)
        overlay = _overlay(output / "branch-first-row.png", Path(source["image"]["path"]), first, annotation)
        cost = {"allocated_device": device_name, "allocated_gpu_seconds": time.monotonic() - started,
                "model_forwards": counters["model_forwards"], "vision_forwards": counters["vision_forwards"],
                "wall_seconds": time.monotonic() - started,
                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        _require(cost["allocated_gpu_seconds"] < 3600, "one-case GPU-hour ceiling reached")
        result = {"schema": "recurrence_first_arrivals.stage1_case.v1", "status": "candidate_complete",
                  "admission": literal_binding(ADMISSION), "snapshot": literal_binding(SNAPSHOT),
                  "source": family["source_bindings"], "source_group": "fresh-18", "batch_index": target,
                  "input_identity": input_identity(batch), "source_planning": planning,
                  "effective_identity": identity, "identity_crosswalk": identity_crosswalk,
                  "score_rows": score_rows,
                  "score_A_trace_parity": score_parity,
                  "score_falsification": {"dropped_terminator_rejected": dropped,
                                          "shifted_slot_rejected": shifted},
                  "D_N_minus_A_fp64": score_rows["N"]["row_logprob_fp64"] - score_rows["A"]["row_logprob_fp64"],
                  "deadband_nats": max(.01, .0004 * (len(a_tokens) + len(n_tokens))),
                  "runs": [native_run, sham_run, branch_run],
                  "branch_first_row_owner_proxy": owner,
                  "branch_coordinate_credit": {"supplied_x1": 1, "free_first_row_coordinates": free_first_coordinates},
                  "annotation": annotation, "overlay": overlay, "raw_evidence": raw_binding,
                  "counters": counters, "cost": cost}
        case_binding = _write_new(output / "case.json", result)
        terminal = {"schema": "recurrence_first_arrivals.stage1_receipt.v1", "status": "candidate_complete",
                    "launch": launch_binding, "case": case_binding,
                    "overlay": overlay, "raw_evidence": raw_binding, "cost": cost,
                    "artifact_bytes_before_receipt": sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
                    "terminal": True}
    except BaseException as error:
        terminal = {"schema": "recurrence_first_arrivals.stage1_receipt.v1", "status": "technical_invalid",
                    "launch": launch_binding, "error": repr(error),
                    "traceback": traceback.format_exc(), "terminal": True,
                    "raw_evidence": raw_binding,
                    "cost": {"allocated_device": device_name,
                             "allocated_gpu_seconds": time.monotonic() - started,
                             "model_forwards": counters["model_forwards"],
                             "vision_forwards": counters["vision_forwards"],
                             "wall_seconds": time.monotonic() - started,
                             "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device_name)
                             if model is not None else None},
                    "artifact_bytes_before_receipt": sum(p.stat().st_size for p in output.iterdir() if p.is_file())}
    finally:
        for handle in handles:
            handle.remove()
    _write_new(output / "receipt.json", terminal)
    if terminal["status"] != "candidate_complete":
        raise RuntimeError(terminal["error"])
    return terminal


def readback(output: Path) -> dict:
    receipt = json.loads((output / "receipt.json").read_text())
    _require(receipt["status"] == "candidate_complete" and receipt["terminal"], "case has no terminal success")
    for key in ("launch", "case", "overlay", "raw_evidence"):
        _bound(Path(receipt[key]["path"]), receipt[key])
    result = json.loads((output / "case.json").read_text())
    admission, _snapshot, family, candidate, _cells = _contract()
    _require(result["admission"] == literal_binding(ADMISSION) and result["snapshot"] == literal_binding(SNAPSHOT),
             "cold case admission/snapshot changed")
    _require(result["source"] == family["source_bindings"], "cold source changed")
    _require(len(result["score_A_trace_parity"]) == 10 and all(x["passed"] for x in result["score_A_trace_parity"]),
             "cold score parity failed")
    _require(all(result["score_falsification"].values()), "cold score falsification missing")
    native, sham, branch = result["runs"]
    _require(native["combined_row_start_token_ids"] == sham["combined_row_start_token_ids"],
             "cold native/A sham mismatch")
    _require(all(x["passed"] for x in native["source_trace_parity"]), "cold native parity failed")
    _require(branch["consumer_mutation_falsification"] == {"dropped": "rejected", "replaced": "rejected"},
             "cold actual-consumer falsification missing")
    _require(branch["consumer_input_ids"][3][-1] == 151831 and branch["prefix_end"] == 16,
             "cold actual model input lacks supplied N x1")
    _require(result["score_rows"]["N"]["token_ids"] == candidate["row_token_ids"],
             "cold N row changed")
    _require(math.isclose(result["D_N_minus_A_fp64"], result["score_rows"]["N"]["row_logprob_fp64"]
                          - result["score_rows"]["A"]["row_logprob_fp64"], abs_tol=1e-12),
             "cold finite D changed")
    _require(receipt["cost"]["allocated_gpu_seconds"] < 3600, "cold case exceeded one-hour cap")
    return {"schema": "recurrence_first_arrivals.stage1_readback.v1", "status": "passed",
            "receipt": literal_binding(output / "receipt.json"),
            "case": literal_binding(output / "case.json"),
            "checked_family": CASE_ID, "arrival_row": ROW, "cell_count": 3,
            "artifact_bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
            "gpu_seconds": receipt["cost"]["allocated_gpu_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--readback", action="store_true")
    args = parser.parse_args()
    if args.readback:
        out = readback(args.output)
        _write_new(args.output / "cold-readback.json", out)
        print(json.dumps(out))
    else:
        result = run(args.output, args.device)
        print(json.dumps({"status": result["status"], "cost": result["cost"]}))


if __name__ == "__main__":
    main()
