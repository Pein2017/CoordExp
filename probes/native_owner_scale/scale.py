"""Frozen raw128 selection, force-c acquisition, and fixedP scale-bank preparation.

CPU preparation is outcome-blind. GPU acquisition always uses the literal
Stable50 native history plus one pre-nominated complete row, and gives the
model the entire remaining assistant-output allowance from the fixed 3084 cap.
The shared trainer remains ``probes.parallel_owner_research.training``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from probes.dora_owner_learning.candidate_opportunity import digest, file_hash, require, score
from probes.parallel_owner_research.history import complete_rows, continuation_ledger
from probes.source_rweak_row_cross.run import native_record
from src.data.geometry import iou_xyxy

BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
ROOT = BASE / "2026-09-12-native-owner-scale-and-state" / "scale"
SOURCE = BASE / "2026-09-11-positive-progress-matched-control" / "endpoint-preparation" / "packet.json"
REFERENCE = BASE / "2026-09-12-parallel-owner-research" / "history" / "preparation" / "training-inputs.json"
RAW_INPUTS = BASE / "2026-09-05-sft256-dev128-baseline" / "inputs-v3"
MATERIALIZATION_RAW_SOURCES = (RAW_INPUTS / "train.jsonl", RAW_INPUTS / "dev.jsonl")
SELECTION_V1 = ROOT / "preparation" / "selection.json"
ACQUISITION_V1 = ROOT / "preparation" / "acquisition.json"
V1_PRESERVATION = ROOT / "preparation" / "snapshots" / "v1-preservation.json"
EXECUTED_PRODUCER_SNAPSHOT = (
    ROOT / "preparation" / "snapshots" / "scale-v2-remainder-executed.py"
)
DONUT_SIDECAR = BASE / "2026-09-12-parallel-owner-research" / "data-flywheel" / "candidate-sidecar.json"
SELECTION = ROOT / "preparation" / "selection-v2.json"
ACQUISITION = ROOT / "preparation" / "acquisition-v2.json"
CAP = 3084
EOS = 151645
ROW_START = 151646
ROW_END = 151649
GPUS = (0, 1, 2, 3)
STRATA = ("repeat_or_drift", "missing_or_early_stop", "normal_control")
QUOTAS = {"repeat_or_drift": 25, "missing_or_early_stop": 71, "normal_control": 32}
SELECTION_SALT = "native-owner-scale-raw128-v2"
ABSENCE_MAX_IOU = 0.30
ARM = "native_owner_scale_fixedP"
ADAPTER_IDENTITY_KEYS = ("kind", "root", "file_count", "files", "semantic_identity", "tensor_manifest")
ADAPTER_VERSION_RENAME = ("coordexp-swift-dora-adapter-v1", "coordexp-infras-dora-adapter-v1")


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def binding(path: str | Path) -> dict[str, str]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def publish(path: str | Path, value: Any) -> None:
    from src.artifacts import publish_json_exclusive

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(path, value)


def valid_selection_source(name: str, source: Mapping[str, Any]) -> bool:
    """Keep immutable inputs exact, with one byte-exact executed-code snapshot bridge."""
    if file_hash(source["path"]) == source["sha256"]:
        return True
    return (name == "producer"
            and Path(source["path"]).resolve() == Path(__file__).resolve()
            and binding(EXECUTED_PRODUCER_SNAPSHOT)["sha256"] == source["sha256"])


def same_adapter_payload(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """Accept only the known metadata rename when all artifact-bearing identity is exact."""
    if any(expected.get(key) != observed.get(key) for key in ADAPTER_IDENTITY_KEYS):
        return False
    if expected.get("version") == observed.get("version"):
        return expected.get("fingerprint") == observed.get("fingerprint")
    return (expected.get("version"), observed.get("version")) == ADAPTER_VERSION_RENAME


def stable_stratum(row: Mapping[str, Any]) -> str:
    """Mutually exclusive baseline-only strata; fresh force-c outcomes are absent."""
    stable, overlaps = row["stable_score"], row["stable_overlap_counts"]
    fn = stable["50"]["fn"]
    if overlaps["80"] > 0:
        return "repeat_or_drift"
    if (stable["cap"] or stable["parser_drops"] or fn >= 2
            or (row["golden"]["gt"] and stable["50"]["recall"] <= 0.5)):
        return "missing_or_early_stop"
    if fn == 0 and not stable["cap"] and not stable["parser_drops"]:
        return "normal_control"
    return "normal_single_miss"


def _rank(example_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}\0{example_id}".encode()).hexdigest()


def selected_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """All available repeat/drift plus salted fixed samples of the other strata."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stable_stratum(row)].append(row)
    require(len(grouped["repeat_or_drift"]) == QUOTAS["repeat_or_drift"],
            "repeat/drift availability changed; do not silently revise raw128")
    selected = sorted(grouped["repeat_or_drift"], key=lambda r: r["example_id"])
    for name in STRATA[1:]:
        require(len(grouped[name]) >= QUOTAS[name], f"insufficient {name} source rows")
        selected.extend(sorted(grouped[name], key=lambda r: _rank(r["example_id"]))[:QUOTAS[name]])
    require(len(selected) == 128 and len({r["example_id"] for r in selected}) == 128,
            "raw128 identity count")
    return selected


def _maximal_complete_prefix(ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Keep the literal leading complete rows; never repair an incomplete tail."""
    position = 0
    while position < len(ids):
        if ids[position] != ROW_START:
            break
        try:
            end = list(ids).index(ROW_END, position + 1)
        except ValueError:
            break
        if ROW_START in ids[position + 1:end] or end + 1 - position < 8:
            break
        position = end + 1
    require(position > 0, "Stable50 lacks a complete native history row")
    prefix, tail = list(ids[:position]), list(ids[position:])
    complete_rows(prefix)
    return prefix, tail


def _history(row: Mapping[str, Any], tokenizer: Any) -> tuple[list[int], str, list[int], str]:
    ids = list(row["stable_ids"])
    full_text = tokenizer.decode(ids, skip_special_tokens=False)
    raw_text = row["stable_parsed"]["raw_decode_text"]
    require(full_text == raw_text, "Stable50 stored output text/IDs")
    if row["stable_score"]["stop_reason"] == "im_end":
        require(ids[-1] == EOS and EOS not in ids[:-1], "Stable50 EOS identity")
        ids = ids[:-1]
    else:
        require(row["stable_score"]["stop_reason"] == "length" and len(ids) == CAP
                and EOS not in ids, "Stable50 cap identity")
    complete, omitted = _maximal_complete_prefix(ids)
    rows = complete_rows(complete)
    chosen: list[list[int]] = []
    seen_boxes: list[list[int]] = []
    boundary = "intact_to_eos_or_first_incomplete"
    if row["stable_score"]["strict_repeats"] > 0:
        boundary = "strict_repeat_not_located"
        for literal_row in rows:
            row_text = tokenizer.decode(literal_row, skip_special_tokens=False)
            parsed = native_record(row_text, row["case"], row["golden"], "supplied_prefix")
            if len(parsed["pred"]) != 1 or parsed["dropped_predictions"]:
                boundary = "before_earlier_malformed_break"
                break
            box = parsed["pred"][0]["bbox"]
            if any(iou_xyxy(box, old) > 0.95 for old in seen_boxes):
                boundary = "before_first_later_strict_repeat"
                break
            chosen.append(literal_row)
            seen_boxes.append(box)
        require(boundary in ("before_earlier_malformed_break", "before_first_later_strict_repeat"),
                "strict-repeat boundary not reproduced")
        require(chosen, "strict-repeat boundary leaves no trusted native row")
    else:
        chosen = rows
    ids = [token for literal_row in chosen for token in literal_row]
    withheld = [token for literal_row in rows[len(chosen):] for token in literal_row] + omitted
    text = tokenizer.decode(ids, skip_special_tokens=False)
    require(raw_text.startswith(text) and full_text.startswith(text), "Stable50 literal h text/IDs")
    return ids, text, withheld, boundary


def _candidate_text(gt: Mapping[str, Any]) -> str:
    source = gt["metadata"]["source"]
    require(str(source["coco_ann_id"]) == str(gt["object_id"]), "GT owner provenance")
    return ("<|object_ref_start|>" + source["category_name"] + "<|object_ref_end|>"
            + "<|box_start|>" + "".join(source["bbox_2d"]) + "<|box_end|>")


def _image_record(row: Mapping[str, Any]) -> dict[str, Any]:
    plan, case = row["case"]["image_plan"], row["case"]
    return {
        "backend_prompt_token_count": plan["backend_prompt_token_count"],
        "executed_media_sha256": plan["executed_media_sha256"],
        "image_height": case["image_height"], "image_id": row["image_id"],
        "image_path": case["image_path"], "image_sha256": file_hash(case["image_path"]),
        "image_width": case["image_width"], "logical_transform_id": plan["logical_transform_id"],
        "merged_visual_tokens": plan["merged_visual_tokens"],
        "observed_image_grid_thw": plan["observed_image_grid_thw"],
        "row_id": row["example_id"], "row_index": case["row_index"],
    }


def candidate_rows(row: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Nominate up to two pre-outcome reviewed/GT owners conservatively absent from h.

    The 0.30 any-class screen is only a conservative absence gate. Rejected or
    ambiguous annotations remain neutral; it is not a negative-owner matcher.
    """
    h_ids, h_text, omitted_h_tail, history_boundary = _history(row, tokenizer)
    h_parsed = native_record(h_text, row["case"], row["golden"], "supplied_prefix")
    h_score = score(h_parsed, seed=None, length=len(h_ids), stop="supplied_prefix")
    history_boxes = [item["bbox"] for item in h_parsed["pred"]]
    covered = set(row["stable_score"]["50"]["owners"])
    eligible_gt, held = [], []
    for gt in row["golden"]["gt"]:
        owner = str(gt["object_id"])
        if owner in covered:
            continue
        text = _candidate_text(gt)
        ids = tokenizer.encode(text, add_special_tokens=False)
        require(tokenizer.decode(ids, skip_special_tokens=False) == text, "literal GT row roundtrip")
        require(len(complete_rows(ids)) == 1, "candidate c must be one complete non-EOS row")
        parsed = native_record(text, row["case"], row["golden"], "supplied_prefix")
        c_score = score(parsed, seed=None, length=len(ids), stop="supplied_prefix")
        require(len(parsed["pred"]) == 1 and c_score["50"]["owners"] == [owner],
                "literal GT c must recover its source owner alone")
        box = parsed["pred"][0]["bbox"]
        max_iou = max((iou_xyxy(box, old) for old in history_boxes), default=0.0)
        reason = None
        if max_iou > ABSENCE_MAX_IOU:
            reason = "neutral_history_physical_absence_not_clear"
        elif len(h_ids) + len(ids) >= CAP:
            reason = "neutral_no_remaining_fixed3084_successor_budget"
        record = {
            "owner_id": owner, "category": gt["description"], "source": "coco_gt_annotation",
            "source_annotation": gt["metadata"]["source"], "c_text": text, "c_token_ids": ids,
            "c_token_ids_sha256": digest(ids), "c_bbox_xyxy_pixels": box,
            "c_area_pixels": (box[2] - box[0]) * (box[3] - box[1]),
            "max_any_class_history_iou": max_iou, "absence_threshold_inclusive": ABSENCE_MAX_IOU,
            "status": "eligible" if reason is None else reason,
        }
        (eligible_gt if reason is None else held).append(record)
    eligible_gt.sort(key=lambda x: (-x["c_area_pixels"], x["owner_id"]))
    eligible_unlabeled = []
    if row["example_id"] == "coco2017_train_000000417044":
        sidecar = read(DONUT_SIDECAR)
        for proposal in sorted(sidecar["rows"], key=lambda item: item["generated_order"]):
            if (proposal.get("candidate_label") != "trusted_unlabeled_single_owner"
                    or proposal.get("proposed_action") != "keep"):
                continue
            text, ids = proposal["row_text"], proposal["row_token_ids"]
            require(ids == tokenizer.encode(text, add_special_tokens=False),
                    "reviewed unlabeled row token identity")
            parsed = native_record(text, row["case"], row["golden"], "supplied_prefix")
            require(len(parsed["pred"]) == 1, "reviewed unlabeled proposal one literal row")
            box = parsed["pred"][0]["bbox"]
            max_iou = max((iou_xyxy(box, old) for old in history_boxes), default=0.0)
            if max_iou <= ABSENCE_MAX_IOU and len(h_ids) + len(ids) < CAP:
                eligible_unlabeled.append({
                    "owner_id": proposal["candidate_owner_id"], "category": proposal["description"],
                    "source": "root_reviewed_unlabeled_model_proposal",
                    "source_proposal_row_id": proposal["row_id"],
                    "source_sidecar": binding(DONUT_SIDECAR),
                    "existing_visual_evidence": proposal["review_card"], "c_text": text,
                    "c_token_ids": ids, "c_token_ids_sha256": digest(ids),
                    "c_bbox_xyxy_pixels": box,
                    "c_area_pixels": (box[2] - box[0]) * (box[3] - box[1]),
                    "max_any_class_history_iou": max_iou,
                    "absence_threshold_inclusive": ABSENCE_MAX_IOU, "status": "eligible",
                })
                break
    eligible = [*eligible_unlabeled, *eligible_gt]
    for extra in eligible[2:]:
        extra["status"] = "neutral_beyond_frozen_two_per_image"
        held.append(extra)
    chosen = eligible[:2]
    for index, item in enumerate(chosen):
        item.update(candidate_index=index, h_token_ids=h_ids, h_token_ids_sha256=digest(h_ids),
                    h_text=h_text, h_row_count=len(complete_rows(h_ids)),
                    withheld_native_suffix_token_count=len(omitted_h_tail),
                    history_boundary=history_boundary,
                    h_owner_ids50=h_score["50"]["owners"],
                    remaining_budget=CAP - len(h_ids) - len(item["c_token_ids"]))
    return chosen, held


def _priority(nominations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frozen round-robin stratum order, all first c/image before any second c."""
    ordered = []
    for candidate_index in (0, 1):
        groups = {name: sorted((n for n in nominations
                                if n["stratum"] == name and n["candidate_index"] == candidate_index),
                               key=lambda n: _rank(n["example_id"])) for name in STRATA}
        for local_index in range(max((len(v) for v in groups.values()), default=0)):
            for name in STRATA:
                if local_index < len(groups[name]):
                    ordered.append(groups[name][local_index])
    require({n["job_id"] for n in ordered} == {n["job_id"] for n in nominations},
            "priority lost nomination")
    for index, item in enumerate(ordered):
        item["admission_priority"] = index
    return ordered


def prepare_selection(output: Path = SELECTION) -> dict[str, Any]:
    """Freeze source universe, raw128 identities, c nominations, and slice jobs."""
    from transformers import AutoTokenizer

    source = read(SOURCE)
    rows = source["eval_records"]
    require(source["policy"]["total_action_cap"] == CAP and source["policy"]["temperature"] == 0.0
            and source["policy"]["top_p"] == 1.0 and source["policy"]["repetition_penalty"] == 1.0,
            "Stable50 source decode contract")
    require(len(rows) == len({r["example_id"] for r in rows}) == 384, "Stable50 universe384")
    tokenizer = AutoTokenizer.from_pretrained(source["model"]["base_model_path"], local_files_only=True)
    selected = selected_rows(rows)
    nominations, held = [], []
    selected_records = []
    for row in selected:
        stratum = stable_stratum(row)
        candidates, rejected = candidate_rows(row, tokenizer)
        image = _image_record(row)
        selected_records.append({
            "example_id": row["example_id"], "image_id": str(row["image_id"]), "split": row["split"],
            "stratum": stratum, "selection_rank": _rank(row["example_id"]),
            "image": image, "baseline": {"score": row["stable_score"],
                "overlap_counts": row["stable_overlap_counts"],
                "gt_count": len(row["golden"]["gt"]), "valid_row_count": len(row["stable_parsed"]["pred"])},
            "nominated_count": len(candidates), "neutral_candidate_count": len(rejected),
        })
        for candidate in candidates:
            candidate.update(example_id=row["example_id"], image_id=str(row["image_id"]),
                             split=row["split"], stratum=stratum, image=image,
                             prompt_token_ids_sha256=digest(row["prompt_token_ids"]))
            candidate["job_id"] = f"{row['example_id']}:{candidate['owner_id']}"
            nominations.append(candidate)
        held.extend({"example_id": row["example_id"], "stratum": stratum, **item}
                    for item in rejected)
    nominations = _priority(nominations)
    strict_repeat_ids = {row["example_id"] for row in selected
                         if row["stable_score"]["strict_repeats"] > 0}
    first_missing = next(n["job_id"] for n in nominations if n["stratum"] == "missing_or_early_stop")
    second_missing = next(n["job_id"] for n in nominations
                          if n["stratum"] == "missing_or_early_stop" and n["job_id"] != first_missing)
    slice_jobs = [
        next(n["job_id"] for n in nominations
             if n["stratum"] == "repeat_or_drift" and n["example_id"] in strict_repeat_ids),
        next(n["job_id"] for n in nominations
             if n["stratum"] == "repeat_or_drift" and n["example_id"] not in strict_repeat_ids),
        first_missing, second_missing,
    ]
    counts = {
        "source_universe": len(rows), "raw_images": len(selected), "eligible_images": len({n["example_id"] for n in nominations}),
        "nominated_attempts": len(nominations), "held_candidate_rows": len(held),
        "by_stratum": {name: {"images": sum(r["stratum"] == name for r in selected_records),
            "eligible_images": len({n["example_id"] for n in nominations if n["stratum"] == name}),
            "nominations": sum(n["stratum"] == name for n in nominations)} for name in STRATA},
    }
    require(counts["raw_images"] == 128 and counts["nominated_attempts"] <= 256
            and all(sum(n["example_id"] == eid for n in nominations) <= 2
                    for eid in {n["example_id"] for n in nominations}), "fixed acquisition bounds")
    packet = {
        "schema": "native_owner_scale.selection.v2", "status": "frozen_cpu_no_model_calls",
        "question": "From Stable50, does fixedP learning on trusted native h plus complete c plus a verified local successor w scale beyond the two selected-image successes while retaining natural owners?",
        "sources": {"stable50_universe": binding(SOURCE),
                    "reviewed_unlabeled_sidecar": binding(DONUT_SIDECAR),
                    "superseded_v1": binding(SELECTION_V1), "v1_preservation": binding(V1_PRESERVATION),
                    "producer": binding(Path(__file__).resolve())},
        "source_universe_example_ids": [r["example_id"] for r in rows],
        "selection_rule": {"salt": SELECTION_SALT, "strata": list(STRATA), "quotas": QUOTAS,
            "repeat_rule": "stable any-class later-row overlap IoU>0.80; all 25 available",
            "missing_rule": "nonrepeat with cap/parser-drop, FN50>=2, or recall50<=0.5",
            "normal_rule": "nonrepeat, nonburden no-obvious-failure output with FN50=0",
            "within_stratum": "ascending sha256(salt NUL example_id)"},
        "nomination_rule": {"maximum_per_image": 2,
            "source": "GT-backed literal rows plus first eligible already root-reviewed donut unlabeled proposal",
            "history": "strict-repeat rows stop immediately before first later IoU>0.95 repeat or earlier malformed break; otherwise intact native prefix to EOS/first incomplete boundary",
            "absence": f"GT owner absent at IoU50 and candidate box max any-class IoU with h <= {ABSENCE_MAX_IOU}",
            "rank": "descending parsed pixel area then owner ID", "unknown_policy": "neutral",
            "no_alternative_after_outcome": True},
        "admission_rule": {"maximum_packages": 32, "minimum_to_train": 16,
            "priority": "round-robin strata within candidate index; every first c/image precedes every second c/image",
            "requires": ["trusted physical single-owner c absent from h", "complete literal c",
                "immediate complete nonduplicate local w", "root visual acceptance"],
            "does_not_require": "globally clean released suffix; full suffix burden and owner losses are retained separately"},
        "counts": counts, "selected": selected_records, "nominations": nominations,
        "neutral_candidates": held, "slice_job_ids": slice_jobs,
        "coordination": {"evaluation_exclude_entire_source_universe": True,
            "known_state_overlap_example_ids": [eid for eid in ("coco2017_train_000000009813",
                "coco2017_train_000000158044", "coco2017_train_000000417044",
                "coco2017_train_000000477415") if eid in {r["example_id"] for r in selected}]},
        "cost_stop": {"max_attempts": 256, "attempts_frozen": len(nominations),
            "total_output_cap_including_h_c": CAP, "train_only_if_admitted_at_least": 16,
            "single_terminal_fit": True},
        "preoutcome_delta_from_v1": {"model_calls_before_correction": 0,
            "normal_stratum": "replaced ordinary single-miss omissions with true FN50=0 controls",
            "history": "strict-repeat histories moved from near-terminal/cap to visited state before first strict repeat",
            "unlabeled": "exercised one existing root-reviewed proposal when absent from selected h",
            "conservative_iou_hold": "nomination screen only; not a physical-identity theorem"},
    }
    publish(output, packet)
    return validate_selection(output)


def validate_selection(path: str | Path = SELECTION) -> dict[str, Any]:
    packet = read(path)
    require(packet["schema"] == "native_owner_scale.selection.v2"
            and packet["status"] == "frozen_cpu_no_model_calls", "selection schema/status")
    for name, source in packet["sources"].items():
        # The executed acquisition producer was preserved byte-exactly before
        # the authorized, post-acquisition distinct-first consumer correction.
        # This narrow bridge does not relax any data/provenance source binding.
        require(valid_selection_source(name, source), "selection source changed")
    require(len(packet["source_universe_example_ids"]) == len(set(packet["source_universe_example_ids"])) == 384,
            "selection source universe")
    require(len(packet["selected"]) == 128 and Counter(r["stratum"] for r in packet["selected"]) == Counter(QUOTAS),
            "raw128 strata")
    jobs = packet["nominations"]
    require(len(jobs) <= 256 and [j["admission_priority"] for j in jobs] == list(range(len(jobs))),
            "frozen job priority")
    require(all(j["remaining_budget"] == CAP - len(j["h_token_ids"]) - len(j["c_token_ids"])
                and j["remaining_budget"] > 0 for j in jobs), "full remaining budget")
    require(all(digest(j["h_token_ids"]) == j["h_token_ids_sha256"]
                and digest(j["c_token_ids"]) == j["c_token_ids_sha256"] for j in jobs), "literal ID hashes")
    require(len(packet["slice_job_ids"]) == len(GPUS)
            and set(packet["slice_job_ids"]) <= {j["job_id"] for j in jobs}, "slice IDs")
    return packet


def prepare_acquisition(selection_path: Path = SELECTION, output: Path = ACQUISITION) -> dict[str, Any]:
    packet = validate_selection(selection_path)
    source = read(packet["sources"]["stable50_universe"]["path"])
    acquisition = {
        "schema": "native_owner_scale.acquisition.v2", "status": "sealed_requires_root_gpu_grant",
        "sources": {"selection": binding(selection_path), "stable50_universe": packet["sources"]["stable50_universe"],
                    "producer": binding(Path(__file__).resolve())},
        "adapter": source["model"]["current_adapter"], "policy": {"temperature": 0.0, "top_p": 1.0,
            "top_k": 0, "repetition_penalty": 1.0, "total_output_cap": CAP, "trace": "none"},
        "modes": {"slice": packet["slice_job_ids"],
            "remainder": [j["job_id"] for j in packet["nominations"]
                          if j["job_id"] not in packet["slice_job_ids"]],
            "full": [j["job_id"] for j in packet["nominations"]]},
        "physical_gpus": list(GPUS), "bounds": {"max_rank_seconds": 18000,
            "max_cuda_allocated_bytes": 32 * 1024**3, "max_cuda_reserved_bytes": 32 * 1024**3,
            "max_rss_bytes": 40 * 1024**3, "max_model_forwards_per_rank": 80 * CAP,
            "max_image_forwards_per_rank": 80},
    }
    publish(output, acquisition)
    return validate_acquisition(output)


def validate_acquisition(path: str | Path = ACQUISITION) -> dict[str, Any]:
    packet = read(path)
    require(packet["schema"] == "native_owner_scale.acquisition.v2"
            and packet["status"] == "sealed_requires_root_gpu_grant", "acquisition schema/status")
    for source in packet["sources"].values():
        require(file_hash(source["path"]) == source["sha256"], "acquisition source changed")
    selection = validate_selection(packet["sources"]["selection"]["path"])
    ids = {j["job_id"] for j in selection["nominations"]}
    require(set(packet["modes"]) == {"slice", "remainder", "full"}
            and set(packet["modes"]["full"]) == ids
            and set(packet["modes"]["slice"]).isdisjoint(packet["modes"]["remainder"])
            and set(packet["modes"]["slice"]) | set(packet["modes"]["remainder"]) == ids,
            "acquisition jobs changed")
    require(packet["physical_gpus"] == list(GPUS) and packet["policy"]["total_output_cap"] == CAP,
            "acquisition topology/cap")
    return packet


def acquisition_jobs(packet: Mapping[str, Any], mode: str, shard: int) -> list[dict[str, Any]]:
    require(mode in ("slice", "remainder", "full") and shard in range(len(GPUS)),
            "acquisition mode/shard")
    selection = validate_selection(packet["sources"]["selection"]["path"])
    by_id = {j["job_id"]: j for j in selection["nominations"]}
    return [by_id[job_id] for job_id in packet["modes"][mode]][shard::len(GPUS)]


def acquisition_rank(packet_path: Path, output: Path, mode: str, shard: int, physical_gpu: int) -> None:
    """One Stable50 model load and one immutable JSONL row per frozen job."""
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.source_rweak_row_cross.run import build_requests
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations
    from src.qwen.native import prepare_native_inputs

    packet = validate_acquisition(packet_path)
    require(GPUS[shard] == physical_gpu and os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu)
            and torch.cuda.device_count() == 1, "exact assigned one-GPU acquisition rank")
    jobs = acquisition_jobs(packet, mode, shard)
    selection = validate_selection(packet["sources"]["selection"]["path"])
    source = read(selection["sources"]["stable50_universe"]["path"])
    by_id = {r["example_id"]: r for r in source["eval_records"]}
    run = output / f"shard-{shard}"
    require(not run.exists(), "acquisition shard output collision")
    run.mkdir(parents=True)
    started = time.monotonic()
    terminal = {"schema": "native_owner_scale.acquisition_terminal.v1", "status": "running",
        "mode": mode, "shard": shard, "physical_gpu": physical_gpu, "pid": os.getpid(),
        "model_loads": 0, "model_forwards": 0, "image_forwards": 0,
        "continuations": 0, "new_tokens": 0, "packet": binding(packet_path)}
    publish(run / "launch.json", terminal)
    handles = []
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("acquisition wall bound")))
        signal.alarm(packet["bounds"]["max_rank_seconds"])
        observed_adapter = inspect_dora_adapter_payload(
            packet["adapter"]["root"], source["model"]["base_model_path"])
        publish(run / "adapter-inspection.json", {"expected": packet["adapter"],
            "observed": observed_adapter, "artifact_identity_keys": list(ADAPTER_IDENTITY_KEYS),
            "known_version_rename": list(ADAPTER_VERSION_RENAME)})
        require(same_adapter_payload(packet["adapter"], observed_adapter),
                "Stable50 adapter artifact identity changed")
        config = checkpoint_config(InferConfig.model_validate(source["config"]), packet["adapter"]["root"])
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        terminal["model_loads"] = 1
        live, effective = identity["model_identity"]["adapter"], identity["effective_settings"]
        require(live["adapter_path"] == packet["adapter"]["root"] and live["merged_adapters"] == [],
                "loaded Stable50 adapter identity")
        require(effective["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"]
                and effective["observed_attn_implementation"] == "sdpa", "FP32 SDPA acquisition")
        publish(run / "model.json", identity)
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)

        def model_hook(*_: Any) -> None:
            require(terminal["model_forwards"] < packet["bounds"]["max_model_forwards_per_rank"],
                    "model forward cap")
            terminal["model_forwards"] += 1

        def image_hook(*_: Any) -> None:
            require(terminal["image_forwards"] < packet["bounds"]["max_image_forwards_per_rank"],
                    "image forward cap")
            terminal["image_forwards"] += 1

        handles.append(qwen.model.register_forward_pre_hook(model_hook))
        visuals = [m for name, m in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "single visual module")
        handles.append(visuals[0].register_forward_pre_hook(image_hook))
        policy = NativeGenerationPolicy(temperature=0.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, use_model_defaults=False)
        with (run / "rows.jsonl").open("x") as stream:
            for job in jobs:
                frozen = by_id[job["example_id"]]
                requests, _ = build_requests(qwen, source["config"], [frozen["case"]])
                batch = prepare_native_inputs(qwen.processor, requests, device=torch.device("cuda:0"),
                                              record_media_identity=True)
                plan = frozen["case"]["image_plan"]
                require(list(batch.prompt_token_ids[0]) == frozen["prompt_token_ids"]
                        and batch.media_sha256[0] == plan["executed_media_sha256"]
                        and list(batch.image_grids[0]) == plan["observed_image_grid_thw"],
                        "native prompt/media/grid identity")
                prefix = [*job["h_token_ids"], *job["c_token_ids"]]
                require(qwen.tokenizer.decode(job["h_token_ids"], skip_special_tokens=False) == job["h_text"]
                        and qwen.tokenizer.decode(job["c_token_ids"], skip_special_tokens=False) == job["c_text"],
                        "literal h/c text identity")
                budget = CAP - len(prefix)
                require(budget == job["remaining_budget"] > 0, "fixed remaining3084 budget")
                with torch.inference_mode():
                    generated = generate_continuations(qwen.model, batch, extensions=[prefix], budgets=[budget],
                        eos_token_id=EOS, pad_token_id=qwen.tokenizer.pad_token_id, policy=policy, trace="none")[0]
                ids = list(generated.token_ids)
                require(generated.request_id == job["example_id"] and 0 < len(ids) <= budget,
                        "generated request/token count")
                require((generated.stop_reason == "im_end" and ids[-1] == EOS and EOS not in ids[:-1])
                        or (generated.stop_reason == "length" and len(ids) == budget and EOS not in ids),
                        "generated terminal identity")
                text = qwen.tokenizer.decode(ids, skip_special_tokens=False)
                row = {"schema": "native_owner_scale.acquisition_row.v1", "job_id": job["job_id"],
                    "example_id": job["example_id"], "owner_id": job["owner_id"], "stratum": job["stratum"],
                    "candidate_index": job["candidate_index"], "admission_priority": job["admission_priority"],
                    "shard": shard, "mode": mode, "packet_sha256": file_hash(packet_path),
                    "prefix_token_count": len(prefix), "remaining_budget": budget,
                    "free_token_ids": ids, "free_token_ids_sha256": digest(ids), "free_text": text,
                    "stop_reason": generated.stop_reason, "prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
                    "executed_media_sha256": batch.media_sha256[0],
                    **continuation_ledger(job["h_text"] + job["c_text"], text, frozen,
                                          len(prefix), len(ids), generated.stop_reason)}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                terminal["continuations"] += 1
                terminal["new_tokens"] += len(ids)
        require(terminal["continuations"] == len(jobs), "complete acquisition denominator")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        for handle in handles:
            handle.remove()
        terminal.update(elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if terminal["status"] == "completed" and any(
                terminal[k] > packet["bounds"]["max_" + k.removeprefix("peak_")]
                for k in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_rss_bytes")):
            terminal.update(status="failed", exit_code=1, error="acquisition memory bound")
        publish(run / "terminal.json", terminal)
    require(terminal["status"] == "completed", "acquisition terminal completion")


def launch_acquisition(packet_path: Path, output: Path, mode: str) -> None:
    packet = validate_acquisition(packet_path)
    require(not output.exists(), "acquisition output collision")
    output.mkdir(parents=True)
    running = []
    launches = []
    for shard, gpu in enumerate(GPUS):
        command = [sys.executable, "-m", "probes.native_owner_scale.scale", "acquire-rank",
            "--acquisition", str(packet_path.resolve()), "--output", str(output.resolve()),
            "--mode", mode, "--shard", str(shard), "--physical-gpu", str(gpu)]
        log = (output / f"shard-{shard}.log").open("x")
        process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                   stdout=log, stderr=subprocess.STDOUT)
        launches.append({"shard": shard, "gpu": gpu, "pid": process.pid, "command": command,
                         "jobs": len(acquisition_jobs(packet, mode, shard))})
        running.append((process, log))
    publish(output / "process-launch.json", {"packet": binding(packet_path), "mode": mode, "launches": launches})
    exits = []
    for launch, (process, log) in zip(launches, running, strict=True):
        exits.append({**launch, "exit_code": process.wait()})
        log.close()
    publish(output / "process-exits.json", {"results": exits})
    require(all(row["exit_code"] == 0 for row in exits), "acquisition subprocess failure; logs preserved")


def _literal_first_w(free_ids: Sequence[int], free_text: str, parsed: Mapping[str, Any],
                     seen_boxes: Sequence[Sequence[int]], tokenizer: Any) -> dict[str, Any]:
    """Propose only the immediate complete first free row; later burden is separate."""
    if not parsed["pred"]:
        return {"status": "no_valid_free_row"}
    first = parsed["pred"][0]
    text = first["raw_span_text"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    if first["generated_order"] != 0 or not free_text.startswith(text) or list(free_ids[:len(ids)]) != ids:
        return {"status": "first_free_content_not_one_literal_complete_row"}
    require(len(complete_rows(ids)) == 1, "w literal complete row")
    max_iou = max((iou_xyxy(first["bbox"], box) for box in seen_boxes), default=0.0)
    return {"status": "candidate_local_w" if max_iou <= 0.95 else "strict_duplicate_local_w",
        "w_text": text, "w_token_ids": ids, "w_token_ids_sha256": digest(ids),
        "w_bbox_xyxy_pixels": first["bbox"], "w_description": first["description"],
        "max_any_class_prefix_iou": max_iou, "strict_duplicate_threshold_exclusive": 0.95}


def reduce_acquisition(packet_path: Path, output: Path, mode: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    packet = validate_acquisition(packet_path)
    selection = validate_selection(packet["sources"]["selection"]["path"])
    source = read(selection["sources"]["stable50_universe"]["path"])
    by_example = {r["example_id"]: r for r in source["eval_records"]}
    by_job = {r["job_id"]: r for r in selection["nominations"]}
    tokenizer = AutoTokenizer.from_pretrained(source["model"]["base_model_path"], local_files_only=True)
    expected = packet["modes"][mode]
    rows, terminals = [], []
    exits = read(output / "process-exits.json")["results"]
    require(len(exits) == len(GPUS) and all(r["exit_code"] == 0 for r in exits), "outer acquisition exits")
    for shard in range(len(GPUS)):
        terminal = read(output / f"shard-{shard}" / "terminal.json")
        require(terminal["status"] == "completed" and terminal["exit_code"] == 0
                and terminal["packet"] == binding(packet_path), "acquisition terminal")
        actual = [json.loads(line) for line in (output / f"shard-{shard}" / "rows.jsonl").read_text().splitlines()
                  if line.strip()]
        jobs = acquisition_jobs(packet, mode, shard)
        require(len(actual) == len(jobs) == terminal["continuations"], "acquisition shard denominator")
        for observed, job in zip(actual, jobs, strict=True):
            require(observed["job_id"] == job["job_id"] and observed["shard"] == shard
                    and observed["packet_sha256"] == file_hash(packet_path), "acquisition row identity")
            require(digest(observed["free_token_ids"]) == observed["free_token_ids_sha256"]
                    and tokenizer.decode(observed["free_token_ids"], skip_special_tokens=False) == observed["free_text"],
                    "acquisition raw token/text")
            frozen = by_example[job["example_id"]]
            fresh = continuation_ledger(job["h_text"] + job["c_text"], observed["free_text"], frozen,
                len(job["h_token_ids"]) + len(job["c_token_ids"]), len(observed["free_token_ids"]),
                observed["stop_reason"])
            require(all(observed[k] == v for k, v in fresh.items()), "fresh acquisition consumer replay")
            prefix = native_record(job["h_text"] + job["c_text"], frozen["case"], frozen["golden"], "supplied_prefix")
            seen = [item["bbox"] for item in prefix["pred"]]
            observed["local_w"] = _literal_first_w(observed["free_token_ids"], observed["free_text"],
                observed["free_parsed"], seen, tokenizer)
        rows.extend(actual)
        terminals.append(terminal)
    require({r["job_id"] for r in rows} == set(expected) and len(rows) == len(expected), "acquisition total denominator")
    rows.sort(key=lambda r: by_job[r["job_id"]]["admission_priority"])
    result = {"schema": "native_owner_scale.acquisition_result.v1",
        "status": "candidate_cpu_verified_visual_review_pending", "mode": mode,
        "packet": binding(packet_path), "selection": packet["sources"]["selection"],
        "denominators": {"source_universe": 384, "raw_images": 128,
            "frozen_nominations": len(selection["nominations"]), "executed": len(rows),
            "terminal_failures": 0, "candidate_local_w": sum(r["local_w"]["status"] == "candidate_local_w" for r in rows)},
        "rows": rows, "cost": {"model_forwards": sum(t["model_forwards"] for t in terminals),
            "image_forwards": sum(t["image_forwards"] for t in terminals),
            "allocated_gpu_hours": sum(t["elapsed_seconds"] for t in terminals) / 3600},
        "raw_row_sources": [binding(output / f"shard-{s}" / "rows.jsonl") for s in range(len(GPUS))],
        "claim_boundary": "machine local-w candidacy is not visual owner acceptance and forced rows receive no natural credit",
    }
    publish(output / "result.json", result)
    return result


def merge_acquisition(packet_path: Path, slice_result_path: Path,
                      remainder_result_path: Path, output: Path) -> dict[str, Any]:
    """Bind the disjoint accepted slice and remainder into one 156-row denominator."""
    packet = validate_acquisition(packet_path)
    slice_result, remainder_result = read(slice_result_path), read(remainder_result_path)
    require(slice_result["schema"] == remainder_result["schema"]
            == "native_owner_scale.acquisition_result.v1", "merge result schemas")
    require(slice_result["mode"] == "slice" and remainder_result["mode"] == "remainder",
            "merge result partitions")
    selection = validate_selection(packet["sources"]["selection"]["path"])
    by_job = {row["job_id"]: row for row in selection["nominations"]}
    slice_ids = [row["job_id"] for row in slice_result["rows"]]
    remainder_ids = [row["job_id"] for row in remainder_result["rows"]]
    require(slice_ids == packet["modes"]["slice"] and set(remainder_ids) == set(packet["modes"]["remainder"]),
            "merge observed/frozen partitions")
    require(set(slice_ids).isdisjoint(remainder_ids)
            and set(slice_ids) | set(remainder_ids) == set(packet["modes"]["full"])
            and len(slice_ids) + len(remainder_ids) == len(set(packet["modes"]["full"])) == 156,
            "merge must contain 156 unique frozen IDs exactly once")
    rows = [*slice_result["rows"], *remainder_result["rows"]]
    for row in rows:
        frozen = by_job[row["job_id"]]
        require(all(row[key] == frozen[key] for key in
                    ("example_id", "owner_id", "stratum", "candidate_index", "admission_priority")),
                "merged row differs from frozen nomination")
    rows.sort(key=lambda row: by_job[row["job_id"]]["admission_priority"])
    require([row["admission_priority"] for row in rows] == list(range(156)),
            "merged frozen admission priority")
    result = {"schema": "native_owner_scale.acquisition_result.v1",
        "status": "candidate_cpu_verified_visual_review_pending", "mode": "full",
        "packet": binding(packet_path), "selection": packet["sources"]["selection"],
        "partitions": {"slice": binding(slice_result_path), "remainder": binding(remainder_result_path)},
        "denominators": {"source_universe": 384, "raw_images": 128,
            "frozen_nominations": 156, "executed": 156, "terminal_failures": 0,
            "candidate_local_w": sum(row["local_w"]["status"] == "candidate_local_w" for row in rows)},
        "rows": rows, "cost": {key: slice_result["cost"][key] + remainder_result["cost"][key]
                                for key in ("model_forwards", "image_forwards", "allocated_gpu_hours")},
        "raw_row_sources": [*slice_result["raw_row_sources"], *remainder_result["raw_row_sources"]],
        "claim_boundary": "machine local-w candidacy is not visual owner acceptance and forced rows receive no natural credit",
    }
    publish(output, result)
    return result


def exposure_steps(record_ids: Sequence[str]) -> list[list[dict[str, Any]]]:
    """Batch2 schedule with exactly 32 exposures per package, including odd N."""
    require(16 <= len(record_ids) <= 32 and len(set(record_ids)) == len(record_ids), "admitted N16..32")
    steps = []
    for step in range(16 * len(record_ids)):
        steps.append([{"record_id": record_ids[(2 * step) % len(record_ids)], "weight": 1.0},
                      {"record_id": record_ids[(2 * step + 1) % len(record_ids)], "weight": 1.0}])
    counts = Counter(item["record_id"] for step in steps for item in step)
    require(set(counts.values()) == {32}, "equal 32 exposures/package")
    return steps


def distinct_first_admission(
    rows: Sequence[Mapping[str, Any]], maximum: int = 32,
) -> list[Mapping[str, Any]]:
    """Prefer one accepted package per image, then fill by frozen priority."""
    require(maximum > 0, "positive admission maximum")
    ordered = sorted(rows, key=lambda row: row["admission_priority"])
    require(len({row["job_id"] for row in ordered}) == len(ordered), "unique admitted jobs")
    require(len({row["admission_priority"] for row in ordered}) == len(ordered),
            "unique admission priorities")
    seen_images = set()
    first_per_image, repeated_images = [], []
    for row in ordered:
        if row["example_id"] in seen_images:
            repeated_images.append(row)
        else:
            seen_images.add(row["example_id"])
            first_per_image.append(row)
    return [*first_per_image, *repeated_images][:maximum]


def validate_final_reviews(
    reviews: Mapping[str, Any], acquisition_result: Path,
) -> Mapping[str, Any]:
    """Admit only the sealed lead-accepted review artifact for this result."""
    require(reviews.get("schema") == "native_owner_scale.visual_reviews.final.v1",
            "final visual review schema")
    require(reviews.get("status") == "lead_accepted_physical_bank_admission",
            "physical bank is not lead-accepted")
    require(reviews.get("acquisition_result") == binding(acquisition_result),
            "visual review/acquisition binding")
    require(reviews.get("counts") == {"accept": 16, "neutral": 140}
            and reviews.get("candidate_decisions") == {"accept": 16, "neutral": 34},
            "sealed visual decision counts")
    return reviews


def prepare_training_bank(acquisition_result: Path, reviews_path: Path, output: Path) -> dict[str, Any]:
    """Consume root visual decisions and seal exactly one fixedP N16..32 fit."""
    from probes.parallel_owner_research.training import prepare_packet, work_counts

    result = read(acquisition_result)
    reviews = validate_final_reviews(read(reviews_path), acquisition_result)
    require(result["schema"] == "native_owner_scale.acquisition_result.v1" and result["mode"] == "full",
            "training bank requires complete frozen acquisition")
    selection = validate_selection(result["selection"]["path"])
    source = read(selection["sources"]["stable50_universe"]["path"])
    by_example = {r["example_id"]: r for r in source["eval_records"]}
    by_job = {r["job_id"]: r for r in selection["nominations"]}
    decisions = reviews.get("decisions", {})
    require(set(decisions) == {r["job_id"] for r in result["rows"]}, "visual decision denominator")
    eligible = []
    for row in result["rows"]:
        decision = decisions[row["job_id"]]
        require(decision.get("status") in ("accept", "neutral") and decision.get("evidence_paths"),
                "explicit visual decision/evidence")
        if decision["status"] == "accept":
            require(row["local_w"]["status"] == "candidate_local_w"
                    and decision.get("c_single_owner_absent_from_h") is True
                    and decision.get("w_single_owner_nonduplicate") is True, "visual c+w acceptance")
            eligible.append(row)
    admitted = distinct_first_admission(eligible, maximum=32)
    if len(admitted) < 16:
        shortfall = {"schema": "native_owner_scale.training_stop.v1", "status": "supply_shortfall_no_training",
            "acquisition_result": binding(acquisition_result), "visual_reviews": binding(reviews_path),
            "admitted": len(admitted), "minimum": 16, "frozen_nominations": len(result["rows"])}
        publish(output, shortfall)
        return shortfall
    reference = read(REFERENCE)
    active = {r["example_id"] for r in admitted}
    normal_keys = [key for key in reference["normal_keys"] if key not in active]
    require(normal_keys and not active.intersection(normal_keys), "actual retained normal denominator")
    positives, conditionals = [], []
    for row in admitted:
        job = by_job[row["job_id"]]
        frozen = by_example[job["example_id"]]
        common = {"example_id": job["example_id"], "image": job["image"],
            "prompt_token_ids": frozen["prompt_token_ids"],
            "prompt_token_ids_sha256": digest(frozen["prompt_token_ids"]),
            "case_id": row["job_id"], "candidate_owner_provenance": {"kind": job["source"],
                "owner_id": job["owner_id"], "acquisition_result": binding(acquisition_result),
                "visual_review": binding(reviews_path)}}
        positives.append({**common, "record_id": row["job_id"] + ":c",
            "prefix_token_ids": job["h_token_ids"], "target_token_ids": job["c_token_ids"]})
        conditionals.append({**common, "record_id": row["job_id"] + ":w_kl",
            "prefix_token_ids": [*job["h_token_ids"], *job["c_token_ids"]],
            "target_token_ids": row["local_w"]["w_token_ids"],
            "kl_positions": list(range(len(row["local_w"]["w_token_ids"]))),
            "unknown_mask_policy": "literal_positions_only"})
    positive_ids = [r["record_id"] for r in positives]
    steps = exposure_steps(positive_ids)
    runtime = {"world_sizes": [8], "max_rank_seconds": 36000,
        "max_cuda_allocated_bytes": 32 * 1024**3, "max_cuda_reserved_bytes": 32 * 1024**3,
        "max_rss_bytes": 40 * 1024**3, "max_model_forwards_per_rank": 7000,
        "max_image_forwards_per_rank": 7000}
    train = prepare_packet(output, lane="native-owner-scale-fixedP",
        anchor_input_path=reference["anchor_input"]["path"], margin_input_path=reference["margin_input"]["path"],
        normal_keys=normal_keys, positive_records=positives, conditional_records=conditionals,
        arms={ARM: {"steps": steps}}, weights={"positive": 1.0, "conditional_kl": 10.0,
            "normal_kl": 100.0, "margin": 10.0}, denominators={"positive": 2.0,
            "conditional_kl": float(len(conditionals)), "normal_kl": float(len(normal_keys)),
            "margin": float(len(normal_keys))}, optimizer=reference["optimizer"], clip_gradient_norm=1.0,
        runtime=runtime, materialization_raw_source_paths=MATERIALIZATION_RAW_SOURCES)
    counts = [work_counts(train, ARM, 8, rank) for rank in range(8)]
    require(all(c["optimizer_steps"] == 16 * len(admitted) for c in counts), "16*N optimizer steps")
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-selection", "verify-selection", "prepare-acquisition",
        "verify-acquisition", "acquire-rank", "acquire", "reduce", "merge", "prepare-training"])
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--acquisition", type=Path, default=ACQUISITION)
    parser.add_argument("--output", type=Path, default=ROOT / "acquisition-slice-v1")
    parser.add_argument("--mode", choices=["slice", "remainder", "full"], default="slice")
    parser.add_argument("--shard", type=int, choices=list(range(len(GPUS))))
    parser.add_argument("--physical-gpu", type=int, choices=list(GPUS))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--slice-result", type=Path)
    parser.add_argument("--remainder-result", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-selection":
        value = prepare_selection(args.selection)
    elif args.command == "verify-selection":
        value = validate_selection(args.selection)
    elif args.command == "prepare-acquisition":
        value = prepare_acquisition(args.selection, args.acquisition)
    elif args.command == "verify-acquisition":
        value = validate_acquisition(args.acquisition)
    elif args.command == "acquire-rank":
        require(args.shard is not None and args.physical_gpu is not None, "rank requires shard/GPU")
        acquisition_rank(args.acquisition, args.output, args.mode, args.shard, args.physical_gpu)
        return
    elif args.command == "acquire":
        launch_acquisition(args.acquisition, args.output, args.mode)
        return
    elif args.command == "reduce":
        value = reduce_acquisition(args.acquisition, args.output, args.mode)
    elif args.command == "merge":
        require(args.slice_result is not None and args.remainder_result is not None,
                "merge requires slice/remainder results")
        value = merge_acquisition(args.acquisition, args.slice_result, args.remainder_result, args.output)
    else:
        require(args.result is not None and args.reviews is not None, "training requires result/reviews")
        value = prepare_training_bank(args.result, args.reviews, args.output)
    print(json.dumps({"schema": value["schema"], "status": value.get("status", "validated"),
                      "counts": value.get("counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
