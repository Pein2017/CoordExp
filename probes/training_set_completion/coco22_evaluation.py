"""COCO22 saved-readback scoring with frozen coverage and complete-output review.

Match all frozen targets of an image once, then project old227/new cohort and
old218/prior-new9 ownership from the same assignment.  The current versioned
annotation ledger may acquire visually admitted extras without changing this
run's frozen teacher or annotation-relative target denominator.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion import coco22_annotations, coco227_evaluation as predecessor, paired_evaluation as prior, readback_selectors
from src.data.geometry import parse_source_bbox_tokens
from src.eval.detection_categories import COCO_80_CLASS_NAMES


ROOT = coco22_annotations.ROOT
OLD_BANK = predecessor.DATA
OLD_PREPARATION = predecessor.OUTPUT / "preparation.json"
INITIAL_ANNOTATIONS = ROOT / "annotations-v1/annotations.jsonl"
GT_ADMISSION = ROOT / "gt-review-admission-v1/lead-admission.json"
BANK = ROOT / "data-v1/bank.json"
OUTPUT = ROOT / "evaluation-preparation-v1"
SCHEMA = "training_set_completion.coco22_evaluation.v1"
PREPARATION_SCHEMA = SCHEMA + ".preparation"
IOU_PRIMARY, IOU_DIAGNOSTIC, CAP = prior.IOU_PRIMARY, prior.IOU_DIAGNOSTIC, prior.CAP
IMAGE_COUNT, OLD_COUNT, OLD218, PRIOR_NEW9 = 22, 227, 218, 9
SAVED_STEPS = (0, 8, 16, 32, 64, 128, 256)
READBACK_ADMISSION_SCHEMA = "training_set_completion.coco22_readback.v1.admission"
require, binding, read, digest, publish = prior.require, prior.binding, prior.read, prior.digest, prior.publish


def _keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[int, str]]:
    keys = [(int(row["image_id"]), str(row["owner_id"])) for row in rows]
    require(len(set(keys)) == len(keys), "duplicate evaluation owner identity")
    return set(keys)


def _target_from_card(image_id: int, card: Mapping[str, Any]) -> dict[str, Any]:
    fields = card.get("edited_fields", {})
    require(isinstance(fields, Mapping), "teacher trace fields")
    description = fields.get("selected_description")
    require(description in COCO_80_CLASS_NAMES, "frozen COCO80 teacher description")
    bins = coco22_annotations.validate_bbox(fields.get("catalog_reference_coord_bins_1000"),
                                               owner_id=str(card.get("owner_id", "")))
    return prior._target(image_id=image_id, owner_id=str(card.get("owner_id", "")),
                         bins=bins,
                         description=description, class_status="verified_coco80")


def _teacher_ledgers(bank: Mapping[str, Any], old_bank: Mapping[str, Any], old_prep: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    routes = bank.get("routes")
    require(isinstance(routes, list) and len(routes) == IMAGE_COUNT and bank.get("image_count") == IMAGE_COUNT, "22 frozen bank routes")
    old_routes = old_bank["routes"]
    by_image = {route.get("image_id"): route for route in routes}
    require(len(by_image) == IMAGE_COUNT and all(type(image) is int for image in by_image), "distinct frozen image routes")
    require(all(by_image.get(route["image_id"]) == route for route in old_routes), "old11 teacher routes changed")
    old_ids = {(int(row["image_id"]), str(row["owner_id"])) for row in old_prep["ledgers"]["scoped227"]}
    require(len(old_ids) == OLD_COUNT, "inherited old227 population")
    frozen = [_target_from_card(route["image_id"], card) for route in routes for card in route["provenance"]["trace"]]
    keys = _keys(frozen)
    require(old_ids <= keys, "old227 owner lost from 22 teacher")
    require(bank.get("fixed_owner_count") == len(frozen), "frozen owner count")
    new_keys = keys - old_ids
    declared_old = bank.get("old227_owner_ids")
    declared_new = bank.get("new_cohort_owner_ids")
    require(isinstance(declared_old, list) and len(declared_old) == OLD_COUNT and
            isinstance(declared_new, list) and len(declared_new) == len(new_keys), "declared frozen partition IDs")
    require(set(declared_old) == {owner for _, owner in old_ids} and
            set(declared_new) == {owner for _, owner in new_keys}, "declared old/new owner partition")
    require(all(image not in {r["image_id"] for r in old_routes} for image, _ in new_keys), "new cohort owner is on old image")
    old = copy.deepcopy(old_prep["ledgers"]["scoped227"])
    require([row for row in frozen if (row["image_id"], row["owner_id"]) in old_ids] == old,
            "inherited old227 target geometry/description changed")
    old218 = copy.deepcopy(old_prep["ledgers"]["old218"])
    prior9 = copy.deepcopy(old_prep["ledgers"]["new9"])
    require(len(old218) == OLD218 and len(prior9) == PRIOR_NEW9 and
            _keys(old218) | _keys(prior9) == old_ids and not _keys(old218) & _keys(prior9),
            "old218/prior-new9 inheritance")
    return {"frozen": frozen, "old227": old, "new_cohort": [row for row in frozen if
            (row["image_id"], row["owner_id"]) in new_keys], "old218": old218, "prior_new9": prior9}


def _gt_review_map(sidecar: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                   *, new_image_ids: set[int]) -> dict[tuple[int, str], dict[str, Any]]:
    """Require one accepted review decision per raw GT on the new images."""
    require(sidecar.get("status") == "lead-accepted", "new GT review sidecar must be lead-accepted")
    reviews = sidecar.get("gt_owners")
    require(isinstance(reviews, list), "new GT owner review records")
    raw = {(row["image_id"], str(item["coco_ann_id"])): item for row in rows
           if row["image_id"] in new_image_ids for item in row["objects"]}
    require(sum(len(row["objects"]) for row in rows if row["image_id"] in new_image_ids) == len(raw),
            "duplicate new raw GT annotation owner ID")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    allowed = {"verified_real_gt", "reject_false_gt", "reject_duplicate_gt", "hold"}
    for item in reviews:
        require(isinstance(item, Mapping) and type(item.get("image_id")) is int and
                item.get("coco_ann_id") is not None, "new GT review owner identity")
        key = (item["image_id"], str(item["coco_ann_id"]))
        require(key in raw and key not in result and item.get("decision") in allowed,
                "new GT owner reviewed exactly once with allowed decision")
        require(item.get("bbox_2d") == raw[key].get("bbox_2d"), "new GT review/raw bbox identity")
        if item["decision"] == "verified_real_gt":
            require(item.get("category") == raw[key].get("desc") and
                    item["category"] in COCO_80_CLASS_NAMES, "verified new GT category/raw identity")
        evidence = item.get("evidence")
        require(isinstance(evidence, Mapping), "individual GT review original/overlay/crop evidence")
        for name in ("original", "overlay", "crop"):
            source = evidence.get(name)
            require(isinstance(source, (str, Mapping)), f"GT {key}: {name} evidence")
            if isinstance(source, Mapping):
                prior._checked_binding(source, name=f"GT {key} {name}")
            else:
                binding(Path(source))
        result[key] = dict(item)
    require(set(result) == set(raw), "every new raw GT needs an individual lead-admitted review decision")
    return result


def _annotation_ledger(rows: Sequence[Mapping[str, Any]], *, new_image_ids: set[int],
                       gt_reviews: Mapping[tuple[int, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        image_id = row.get("image_id")
        require(type(image_id) is int and isinstance(row.get("objects"), list) and
                isinstance(row.get("unlabeled"), list), "current annotation image objects")
        for index, item in enumerate(row["objects"]):
            if image_id in new_image_ids:
                decision = gt_reviews.get((image_id, str(item.get("coco_ann_id"))))
                require(decision is not None, "new raw GT missing reviewed identity")
                if decision["decision"] != "verified_real_gt":
                    continue  # Raw source JSONL stays complete; this physical ledger does not.
            require(item.get("coco_ann_id") is not None and item.get("desc") in COCO_80_CLASS_NAMES,
                    "GT COCO80 identity/description")
            result.append(prior._target(image_id=image_id, owner_id=str(item["coco_ann_id"]),
                 bins=parse_source_bbox_tokens(item.get("bbox_2d"), field=f"image.{image_id}.objects[{index}].bbox_2d"),
                 description=item["desc"], class_status="verified_coco80"))
        for item in row["unlabeled"]:
            require(item.get("physical_status") == "valid_unlabeled" and
                    item.get("geometry_status") == "reasonable", "accepted non-GT physical/geometry status")
            status = item.get("class_status")
            require(status in {"verified", "unknown"}, "reviewed non-GT category status")
            description = item.get("desc") if status == "verified" else None
            require(status != "unknown" or (item.get("category_name") is None and item.get("desc") is None), "unknown class leaked")
            require(status != "verified" or (isinstance(description, str) and description == item.get("category_name")),
                    "verified owner description")
            class_status = ("verified_coco80" if description in COCO_80_CLASS_NAMES else
                            "verified_non_coco") if status == "verified" else "unknown"
            result.append(prior._target(image_id=image_id, owner_id=str(item.get("stable_owner_id", "")),
                bins=item.get("bbox_2d_bins_1000", []), description=description, class_status=class_status))
    _keys(result)
    return result


def build_preparation(*, teacher_bank_path: Path = BANK, old_bank_path: Path = OLD_BANK,
                      old_preparation_path: Path = OLD_PREPARATION,
                      annotations_path: Path = INITIAL_ANNOTATIONS,
                      gt_admission_path: Path = GT_ADMISSION,
                      output: Path = OUTPUT, artifact_name: str = "preparation.json") -> dict[str, Any]:
    """Freeze ownership and initial physical evidence before training/readback."""
    require(Path(artifact_name).name == artifact_name and not (output / artifact_name).exists(), "preparation artifact collision/name")
    sources = {name: binding(path) for name, path in (("teacher_bank", teacher_bank_path),
        ("old_bank", old_bank_path), ("old_evaluation_preparation", old_preparation_path),
        ("initial_annotations", annotations_path), ("new_gt_review_admission", gt_admission_path))}
    bank = read(Path(sources["teacher_bank"]["path"]))
    old_bank = read(Path(sources["old_bank"]["path"]))
    old_prep = read(Path(sources["old_evaluation_preparation"]["path"]))
    require(bank.get("sources", {}).get("gt_review_admission") == sources["new_gt_review_admission"],
            "frozen teacher / new GT admission source identity")
    require(old_prep.get("schema") == predecessor.PREPARATION_SCHEMA, "old227 preparation schema")
    predecessor.validate_preparation(old_prep)
    ledgers = _teacher_ledgers(bank, old_bank, old_prep)
    initial = coco22_annotations._rows(Path(sources["initial_annotations"]["path"]))
    require({row["image_id"] for row in initial} == {route["image_id"] for route in bank["routes"]}, "frozen image/initial annotations identity")
    old_images = {route["image_id"] for route in old_bank["routes"]}
    new_images = {row["image_id"] for row in initial} - old_images
    require(len(old_images) == 11 and len(new_images) == 11, "11 old / 11 new image partition")
    reviews = _gt_review_map(read(Path(sources["new_gt_review_admission"]["path"])), initial,
                              new_image_ids=new_images)
    initial_ledger = _annotation_ledger(initial, new_image_ids=new_images, gt_reviews=reviews)
    by_current = {(row["image_id"], row["owner_id"]): row for row in initial_ledger}
    for target in ledgers["frozen"]:
        physical = by_current.get((target["image_id"], target["owner_id"]))
        require(physical is not None and physical["class_status"] == "verified_coco80" and
                physical["description"] == target["description"] and
                physical["reference_coord_bins_1000"] == target["reference_coord_bins_1000"],
                f"frozen teacher owner absent or different annotation: {target['image_id']}:{target['owner_id']}")
    image_ids = [route["image_id"] for route in bank["routes"]]
    value = {"schema": PREPARATION_SCHEMA, "status": "candidate_ready", "image_ids": image_ids,
             "sources": {**sources, "producer": binding(Path(__file__))},
             "ledgers": {**ledgers, "initial_current_known": initial_ledger},
             "matching_contract": {"primary": "entire image one cardinality-first class-agnostic IoU>=0.5 assignment; old227/new cohort/old218/prior-new9 project this assignment",
                                   "diagnostic": "same frozen targets independent IoU>=0.8 diagnostic",
                                   "implementation": "readback_selectors.one_to_one_matches -> src.eval.assignment.global_matches"},
             "metric_contract": {"frozen_task": "all frozen owners matched with correct descriptions; immutable denominator",
                                 "full_output_review": "all valid generated rows resolved, verified extras permitted, no false/repeat/invalid/wrong class/cap/EOS",
                                 "annotation_relative": "TP from frozen joint assignment; denominator all valid parsed predictions, including verified extras"}}
    value["content_sha256"] = digest(value)
    publish(output / artifact_name, value)
    return value


def validate_preparation(value: Mapping[str, Any]) -> None:
    require(value.get("schema") == PREPARATION_SCHEMA and value.get("status") == "candidate_ready", "COCO22 preparation schema/status")
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    require(value.get("content_sha256") == digest(content), "COCO22 frozen preparation identity")
    require(len(value["image_ids"]) == IMAGE_COUNT and len(set(value["image_ids"])) == IMAGE_COUNT, "COCO22 cohort count")
    for name in ("teacher_bank", "old_bank", "old_evaluation_preparation", "initial_annotations",
                 "new_gt_review_admission", "producer"):
        prior._checked_binding(value["sources"][name], name=name)
    teacher = _teacher_ledgers(read(Path(value["sources"]["teacher_bank"]["path"])),
                  read(Path(value["sources"]["old_bank"]["path"])),
                  read(Path(value["sources"]["old_evaluation_preparation"]["path"])))
    require(read(Path(value["sources"]["teacher_bank"]["path"]))["sources"].get("gt_review_admission") ==
            value["sources"]["new_gt_review_admission"], "validated teacher / GT review identity")
    require(all(value["ledgers"][name] == teacher[name] for name in teacher), "COCO22 frozen teacher drift")
    initial_rows = coco22_annotations._rows(Path(value["sources"]["initial_annotations"]["path"]))
    old_images = {row["image_id"] for row in read(Path(value["sources"]["old_bank"]["path"]))["routes"]}
    new_images = {row["image_id"] for row in initial_rows} - old_images
    gt_reviews = _gt_review_map(read(Path(value["sources"]["new_gt_review_admission"]["path"])),
                                 initial_rows, new_image_ids=new_images)
    require(value["ledgers"]["initial_current_known"] ==
            _annotation_ledger(initial_rows, new_image_ids=new_images, gt_reviews=gt_reviews),
            "initial physical annotation drift")


def _projection(joint: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
                predictions: Sequence[Mapping[str, Any]], owner_ids: set[str]) -> dict[str, Any]:
    return predecessor._joint_partition(joint=joint, targets=targets, predictions=predictions,
                                         owner_ids=owner_ids, name="COCO22 frozen projection")


def score_parsed_image(image_id: int, ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
                       predictions: Sequence[Mapping[str, Any]], dropped: Sequence[Mapping[str, Any]],
                       stop: str, generated_token_ids: Sequence[int],
                       review_decisions: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Score caller-verified native parsed rows. No visual judgment is fabricated."""
    require(type(image_id) is int and isinstance(predictions, Sequence), "parsed image")
    valid = list(predictions)
    require(all(isinstance(row.get("coord_bins_1000"), list) and
                len(row["coord_bins_1000"]) == 4 and
                all(type(x) is int for x in row["coord_bins_1000"]) and
                0 <= row["coord_bins_1000"][0] < row["coord_bins_1000"][2] <= 999 and
                0 <= row["coord_bins_1000"][1] < row["coord_bins_1000"][3] <= 999
                for row in valid), "parsed image valid prediction geometry")
    ids = [str(row["prediction_id"]) for row in valid]
    require(len(ids) == len(set(ids)), "duplicate prediction IDs")
    all_targets = list(ledgers["frozen"])
    for name in ("old227", "new_cohort", "old218", "prior_new9"):
        require(prior._by_image(ledgers[name]).get(image_id, []) == list(ledgers[name]), "per-image partition input")
    keys = {row["owner_id"] for row in all_targets}
    for group in ("old227", "new_cohort"):
        require({row["owner_id"] for row in ledgers[group]} <= keys, "frozen subset target")
    require({row["owner_id"] for row in ledgers["old227"]} |
            {row["owner_id"] for row in ledgers["new_cohort"]} == keys, "old/new joint union")
    require({row["owner_id"] for row in ledgers["old218"]} |
            {row["owner_id"] for row in ledgers["prior_new9"]} ==
            {row["owner_id"] for row in ledgers["old227"]}, "old218/new9 joint union")
    joint = prior._ledger_image(all_targets, valid, threshold=IOU_PRIMARY)
    primary = {"frozen": joint, "current_known": prior._ledger_image(ledgers["current_known"], valid, threshold=IOU_PRIMARY)}
    for group in ("old227", "new_cohort", "old218", "prior_new9"):
        owner_ids = {str(row["owner_id"]) for row in ledgers[group]}
        primary[group] = _projection(joint, all_targets, valid, owner_ids)
    require(primary["old227"]["matched_count"] + primary["new_cohort"]["matched_count"] == joint["matched_count"], "joint old/new assignment sum")
    diagnostic = prior._ledger_image(all_targets, valid, threshold=IOU_DIAGNOSTIC)
    raw = prior._raw_debt(dropped)
    duplicates = readback_selectors.pairwise_iou95(valid)
    outside = prior._outside_coco80(valid)
    termination = readback_selectors.termination_metrics(len(generated_token_ids), list(generated_token_ids), stop, cap=CAP)
    physical = prior._physical_statuses(image_id=image_id, predictions=valid,
        current_matches=primary["current_known"]["matches"], current_targets=ledgers["current_known"], exact_reviews={})
    physical_by_id = {row["prediction_id"]: row for row in physical["rows"]}
    reviewed = review_decisions or {}
    for span, decision in reviewed.items():
        require(decision.get("status") == "lead-accepted", "readback review decisions must be lead-admitted")
        require(decision.get("decision") in {"confirmed_false_positive", "invalid_output", "unresolved", "out_of_scope",
                     "verified_extra_real"}, "accepted readback decision semantics")
        if decision["decision"] == "out_of_scope":
            require(isinstance(decision.get("review_category"), str) and decision["review_category"],
                    "out-of-scope owner requires individual review category")
        if decision["decision"] == "verified_extra_real":
            require(any(row.get("raw_span_sha256") == span and
                    row["physical_status"] == "matched_current_known_owner" and
                    row["current_owner_match"]["class_status"] == "verified_coco80" and
                    row["physical_owner_id"] not in {target["owner_id"] for target in all_targets}
                    for row in physical["rows"]),
                    "verified extra must be written in current annotated physical ledger")
    for prediction in valid:
        span = prediction.get("raw_span_sha256")
        if span not in reviewed:
            continue
        status = reviewed[span]["decision"]
        row = physical_by_id[prediction["prediction_id"]]
        if status == "confirmed_false_positive":
            require(row["physical_status"] == "unknown_no_accepted_owner_judgment", "review false contradicts accepted owner")
            row["physical_status"] = "confirmed_false_positive_lead_review"
        elif status == "invalid_output":
            require(row["physical_status"] == "unknown_no_accepted_owner_judgment", "review invalid contradicts accepted owner")
            row["physical_status"] = "invalid_output_lead_review"
        elif status == "out_of_scope":
            require(row["physical_status"] == "unknown_no_accepted_owner_judgment", "out-of-scope review contradicts admitted original-scene owner")
            row["physical_status"] = "out_of_scope_lead_review"
            row["review_category"] = reviewed[span]["review_category"]
    statuses = Counter(row["physical_status"] for row in physical["rows"])
    known_by_prediction = {row["prediction_id"]: row for row in physical["rows"] if row["current_owner_match"]}
    current_wrong = sum(known_by_prediction[row["prediction_id"]]["current_owner_match"]["class_status"].startswith("verified_") and
                        not known_by_prediction[row["prediction_id"]]["current_owner_match"]["class_correct"]
                        for row in valid if row["prediction_id"] in known_by_prediction)
    current_unknown = sum(row["current_owner_match"]["class_status"] == "unknown"
                          for row in physical["rows"] if row["current_owner_match"] and
                          row["physical_status"] != "recomputed_repeat_of_accepted_owner")
    task_failures = {"frozen_owner_fn": joint["fn_count"], "frozen_wrong_or_unresolved_class":
                     joint["class_wrong_count"] + joint["class_unknown_count"]}
    failures = {"confirmed_false_positive": statuses["confirmed_false_positive_lead_review"],
                "out_of_scope": statuses["out_of_scope_lead_review"],
                "unresolved_owner_or_category": statuses["unknown_no_accepted_owner_judgment"] + current_unknown,
                "duplicate_candidate_pairs": len(duplicates),
                "repeated_accepted_owner": statuses["recomputed_repeat_of_accepted_owner"],
                "malformed_or_invalid_geometry": raw["parser_dropped_total"] + statuses["invalid_output_lead_review"],
                "wrong_verified_class": max(current_wrong, joint["class_wrong_count"]),
                "outside_coco80": len(outside), "eos_or_cap": termination["eos_debt"] + termination["cap_debt"]}
    annotation = {"tp": joint["matched_count"], "target_denominator_frozen": len(all_targets),
                  "prediction_denominator_all_valid_parsed_rows": len(valid),
                  "annotation_unmatched_prediction_count": len(valid) - joint["matched_count"],
                  "precision": joint["matched_count"] / len(valid) if valid else 0.,
                  "f1": 2 * joint["matched_count"] / (len(all_targets) + len(valid)) if all_targets or valid else 0.,
                  "meaning": "annotation-relative only; unmatched rows are not automatically physical false positives"}
    return {"image_id": image_id, "ledgers_iou_0_5": primary,
            "frozen_iou_0_8_diagnostic": diagnostic,
            "raw": {"valid_prediction_count": len(valid), **raw, **termination},
            "annotation_relative_micro": annotation,
            "physical": {"rows": physical["rows"], "status_counts": dict(statuses)},
            "duplicate_candidates_iou_gt_0_95": duplicates,
            "outside_literal_coco80": outside,
            "frozen_task_completion": {"complete": not any(task_failures.values()),
                "failure_counts": {k: v for k, v in task_failures.items() if v}},
            "complete_output_review": {"closed": not any(failures.values()),
                "failure_counts": {k: v for k, v in failures.items() if v}}}


def _per_image(ledgers: Mapping[str, Sequence[Mapping[str, Any]]], image_id: int) -> dict[str, list[dict[str, Any]]]:
    return {name: [dict(row) for row in rows if row["image_id"] == image_id] for name, rows in ledgers.items()}


def score_admitted_rows(*, preparation_path: Path, label: str, rows: Sequence[Mapping[str, Any]],
                        readback_admission: Mapping[str, Any],
                        current_annotations_path: Path | None = None,
                        review_decisions_path: Path | None = None) -> dict[str, Any]:
    """Recheck mechanical admission and saved natural rows before scientific scoring."""
    prep_path = preparation_path.resolve(strict=True)
    prep = read(prep_path)
    validate_preparation(prep)
    require(isinstance(label, str) and label and len(rows) == IMAGE_COUNT, "COCO22 label/readback row count")
    arm, step = readback_admission.get("arm"), readback_admission.get("step")
    require(readback_admission.get("schema") == READBACK_ADMISSION_SCHEMA and
            readback_admission.get("status") == "admitted_natural_readback", "COCO22 readback admission schema/status")
    require(arm in {"S", "Source"} and type(step) is int and step in SAVED_STEPS, "COCO22 arm/step")
    require(readback_admission.get("source_kind") == ("cold_source_step0" if step == 0 else "scientific_checkpoint_readback"),
            "COCO22 readback source kind")
    require(readback_admission.get("teacher_bank") == prep["sources"]["teacher_bank"],
            "COCO22 admitted frozen bank identity")
    endpoint = read(prior._checked_binding(readback_admission.get("rows"), name="admitted endpoint rows"))
    require(isinstance(endpoint, Mapping) and endpoint.get("rows") == [dict(row) for row in rows],
            "COCO22 scorer rows differ from admitted bound endpoint")
    for name in ("training_manifest", "trial", "qualification_result", "producer"):
        if readback_admission.get(name) is not None:
            prior._checked_binding(readback_admission[name], name=f"admitted {name}")
    readback_selectors.validate_unique_image_rows(rows)
    saved = {row["image_id"]: row for row in rows}
    require(set(saved) == set(prep["image_ids"]), "COCO22 saved readback image cohort")
    bank = read(Path(prep["sources"]["teacher_bank"]["path"]))
    routes = {route["image_id"]: route for route in bank["routes"]}
    tokenizer = readback_selectors._load_tokenizer(
        read(Path(prep["sources"]["old_evaluation_preparation"]["path"]))["tokenizer_root"])
    current_binding = binding(current_annotations_path) if current_annotations_path else prep["sources"]["initial_annotations"]
    current_rows = coco22_annotations._rows(Path(current_binding["path"]))
    require({row["image_id"] for row in current_rows} == set(prep["image_ids"]), "current annotation image cohort")
    old_images = {route["image_id"] for route in read(Path(prep["sources"]["old_bank"]["path"]))["routes"]}
    new_images = {row["image_id"] for row in current_rows} - old_images
    gt_reviews = _gt_review_map(read(Path(prep["sources"]["new_gt_review_admission"]["path"])),
                                 current_rows, new_image_ids=new_images)
    current_ledger = _annotation_ledger(current_rows, new_image_ids=new_images, gt_reviews=gt_reviews)
    require(_keys(prep["ledgers"]["frozen"]) <= _keys(current_ledger), "frozen owner absent current annotation")
    by_current = {(target["image_id"], target["owner_id"]): target for target in current_ledger}
    for target in prep["ledgers"]["frozen"]:
        require(by_current[(target["image_id"], target["owner_id"])] == target,
                "current annotations changed frozen owner class/geometry")
    initial_keys = _keys(prep["ledgers"]["initial_current_known"])
    for annotation in current_rows:
        for item in annotation["unlabeled"]:
            if (annotation["image_id"], item["stable_owner_id"]) not in initial_keys:
                require(item.get("scene_scope") == "real_original_scene" and
                        isinstance(item.get("review_category"), str) and
                        item.get("provenance", {}).get("admission"),
                        "new annotation owner lacks original-scene review provenance")
    decisions = {}
    review_binding = None
    if review_decisions_path is not None:
        review_binding = binding(review_decisions_path)
        review = read(Path(review_binding["path"]))
        require(isinstance(review, Mapping) and review.get("status") == "lead-accepted" and
                isinstance(review.get("rows"), list), "lead-accepted readback review receipt")
        for entry in review["rows"]:
            image, span = entry.get("image_id"), entry.get("raw_span_sha256")
            require(type(image) is int and isinstance(span, str) and span and
                    (image, span) not in decisions, "unique reviewed readback span")
            visual = entry.get("visual")
            require(isinstance(visual, Mapping) and all(binding(Path(str(visual[k]))) for k in ("original", "overlay", "crop")),
                    "reviewed readback visual evidence")
            decisions[(image, span)] = {**dict(entry), "status": "lead-accepted"}
    per_image = []
    for image_id in prep["image_ids"]:
        row, route = saved[image_id], routes[image_id]
        require(row.get("arm") == arm and row.get("step") == step and
                row.get("checkpoint_step") == step and row.get("route_id") == route["route_id"], "saved arm/step/route identity")
        require(row.get("empty_assistant_prefix") is True and row.get("temperature") == 0. and
                row.get("top_p") == 1. and row.get("top_k") == 0 and
                row.get("repetition_penalty") == 1. and row.get("max_new_tokens") == CAP,
                "saved natural decode policy")
        ids = row.get("generated_token_ids")
        require(isinstance(ids, list) and ids and len(ids) <= CAP and all(type(x) is int and x >= 0 for x in ids) and
                row.get("generated_token_ids_sha256") == digest(ids), "saved generated tokens/hash/cap")
        require(row.get("prompt_token_ids") == route["prompt_token_ids"] and
                row.get("executed_media_sha256") == route["image_identity"]["executed_media_sha256"] and
                row.get("observed_image_grid_thw") == route["image_identity"]["observed_image_grid_thw"],
                "saved prompt/media/grid identity")
        stop = row.get("decode_stop_reason")
        require((stop == "im_end" and ids[-1] == 151645 and 151645 not in ids[:-1]) or
                (stop == "length" and len(ids) == CAP and 151645 not in ids), "saved decode stop")
        raw_text = row.get("raw_decode_text")
        require(isinstance(raw_text, str) and tokenizer.decode(ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False) == raw_text, "saved raw text/token decode identity")
        from src.eval.native_rows import native_detection_record as native_record

        case = route["case"]
        golden = {"example_id": route["example_id"], "gt": [], "image_height": case["image_height"],
                  "image_path": case["image_path"], "image_width": case["image_width"],
                  "row_id": case["row_id"], "row_index": case["row_index"]}
        parsed = native_record(raw_text, case, golden, stop)
        valid, dropped = prior._matchable_rows_with_geometry_debt(parsed)
        applicable = {span: decision for (image, span), decision in decisions.items() if image == image_id}
        valid_spans = {row.get("raw_span_sha256") for row in valid}
        require(set(applicable) <= valid_spans, "review receipt references absent generated span")
        ledger = _per_image(prep["ledgers"], image_id)
        ledger["current_known"] = _per_image({"current_known": current_ledger}, image_id)["current_known"]
        per_image.append(score_parsed_image(image_id, ledger, valid, dropped, stop, ids,
                                          review_decisions=applicable))
    target_count = len(prep["ledgers"]["frozen"])
    tp = sum(row["annotation_relative_micro"]["tp"] for row in per_image)
    pred_count = sum(row["annotation_relative_micro"]["prediction_denominator_all_valid_parsed_rows"] for row in per_image)
    frozen_fail = Counter(); output_fail = Counter()
    for row in per_image:
        frozen_fail.update(row["frozen_task_completion"]["failure_counts"])
        output_fail.update(row["complete_output_review"]["failure_counts"])
    aggregate = {"annotation_relative_micro": {"tp": tp, "target_denominator_frozen": target_count,
            "prediction_denominator_all_valid_parsed_rows": pred_count,
            "annotation_unmatched_prediction_count": pred_count - tp,
            "precision": tp / pred_count if pred_count else 0.,
            "f1": 2 * tp / (target_count + pred_count) if target_count or pred_count else 0.},
        "ledgers_iou_0_5": {name: {"target_count": len(prep["ledgers"][name]),
            "matched_count": sum(row["ledgers_iou_0_5"][name]["matched_count"] for row in per_image),
            "fn_count": sum(row["ledgers_iou_0_5"][name]["fn_count"] for row in per_image)}
            for name in ("frozen", "old227", "new_cohort", "old218", "prior_new9")}}
    require(aggregate["ledgers_iou_0_5"]["frozen"]["matched_count"] == tp and
            aggregate["ledgers_iou_0_5"]["old227"]["matched_count"] +
            aggregate["ledgers_iou_0_5"]["new_cohort"]["matched_count"] == tp, "COCO22 aggregate joint projection")
    return {"schema": SCHEMA, "status": "saved_readback_scored_pending_lead_acceptance", "label": label,
            "arm": arm, "step": step,
            "sources": {"preparation": binding(prep_path), "current_annotations": current_binding,
                "review_decisions": review_binding, "readback_admission": copy.deepcopy(dict(readback_admission)),
                "producer": binding(Path(__file__))}, "matching": prep["matching_contract"],
            "per_image": per_image, "aggregate": aggregate,
            "frozen_task_completion": {"complete": not frozen_fail,
                "failure_counts": dict(frozen_fail)},
            "complete_output_review": {"closed": not output_fail,
                "failure_counts": dict(output_fail)},
            "disposition": "Two independent decisions; pending review is not a learning failure or a control trigger."}


def sustained_completion_milestone(scores: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    require(arm in {"S", "Source"} and len(scores) == len(SAVED_STEPS) and
            {score.get("step") for score in scores} == set(SAVED_STEPS), "COCO22 saved checkpoint trajectory")
    require(all(score.get("schema") == SCHEMA and score.get("arm") == arm for score in scores), "COCO22 trajectory arm/schema")
    by_step = {score["step"]: score for score in scores}
    for step in SAVED_STEPS:
        require(by_step[step]["sources"]["preparation"] == by_step[0]["sources"]["preparation"],
                "COCO22 frozen trajectory preparation identity")
    complete = {step: bool(by_step[step]["frozen_task_completion"]["complete"] and
                           by_step[step]["complete_output_review"]["closed"]) for step in SAVED_STEPS}
    first = next((step for step in SAVED_STEPS if all(complete[later] for later in SAVED_STEPS if later >= step)), None)
    return {"schema": SCHEMA + ".sustained_completion", "arm": arm,
            "both_decisions_by_saved_step": complete, "earliest_sustained_saved_step": first,
            "disposition": "not_attained_by256" if first is None else "saved_checkpoint_milestone_no_unsaved_interval_claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--teacher-bank", type=Path, default=BANK)
    prepare.add_argument("--old-bank", type=Path, default=OLD_BANK)
    prepare.add_argument("--old-preparation", type=Path, default=OLD_PREPARATION)
    prepare.add_argument("--annotations", type=Path, default=INITIAL_ANNOTATIONS)
    prepare.add_argument("--gt-admission", type=Path, default=GT_ADMISSION)
    prepare.add_argument("--output", type=Path, default=OUTPUT)
    score = actions.add_parser("score")
    score.add_argument("--preparation", type=Path, default=OUTPUT / "preparation.json")
    score.add_argument("--admission", type=Path, required=True)
    score.add_argument("--rows", type=Path, required=True)
    score.add_argument("--annotations", type=Path)
    score.add_argument("--review-decisions", type=Path)
    score.add_argument("--label", required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = build_preparation(teacher_bank_path=args.teacher_bank, old_bank_path=args.old_bank,
            old_preparation_path=args.old_preparation, annotations_path=args.annotations,
            gt_admission_path=args.gt_admission, output=args.output)
        path = args.output / "preparation.json"
    else:
        admission = read(args.admission)
        value = score_admitted_rows(preparation_path=args.preparation, label=args.label,
            rows=read(args.rows)["rows"], readback_admission=admission,
            current_annotations_path=args.annotations, review_decisions_path=args.review_decisions)
        publish(args.output, value)
        path = args.output
    print(json.dumps({"path": str(path.resolve()), "status": value["status"]}))


if __name__ == "__main__":
    main()
