"""Publish a successor COCO22 image JSONL after individual lead-admitted owner review.

The old accepted rows and all original image fields survive unchanged.  A new
owner is written only after visual evidence and a lead decision are persisted.
This ledger may grow between readbacks; it never changes a frozen teacher bank.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion import paired_evaluation as prior, readback_selectors
from probes.training_set_completion.unlabeled_annotations import validate_bbox
from src.data.geometry import parse_source_bbox_tokens
from src.eval.detection_categories import COCO_80_CLASS_NAMES


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion")
PREDECESSOR = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/annotations-with-unlabeled-v4/annotations.jsonl")
SCHEMA = "training_set_completion.coco22_annotations.v1"
require, binding, canonical = prior.require, prior.binding, prior.canonical


def _rows(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve(strict=True)
    rows = prior.read_jsonl(resolved)
    require(rows and all(isinstance(row, dict) and type(row.get("image_id")) is int for row in rows), "annotation image records")
    ids = [row["image_id"] for row in rows]
    require(len(ids) == len(set(ids)), "duplicate annotation image ID")
    return rows


def _visual(value: Any, *, image_id: int, owner_id: str) -> dict[str, dict[str, Any]]:
    require(isinstance(value, Mapping), f"{image_id}:{owner_id}: visual evidence")
    return {name: binding(Path(str(value[name]))) for name in ("original", "overlay", "crop")}


def _existing_bbox(item: Mapping[str, Any], *, image_id: int, index: int, gt: bool) -> list[int]:
    if gt:
        return parse_source_bbox_tokens(item.get("bbox_2d"), field=f"image.{image_id}.objects[{index}].bbox_2d")
    return validate_bbox(item.get("bbox_2d_bins_1000"), owner_id=str(item.get("stable_owner_id")))


def _owner_record(entry: Mapping[str, Any], *, source: Mapping[str, Any], visual: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    image_id, owner_id = entry["image_id"], entry["stable_owner_id"]
    bins = validate_bbox(entry.get("bbox_2d_bins_1000"), owner_id=owner_id)
    status = entry.get("class_status")
    description = entry.get("category_name")
    require(status in {"verified", "unknown"}, f"{image_id}:{owner_id}: reviewed class status")
    require((status == "unknown" and description is None) or
            (status == "verified" and description in COCO_80_CLASS_NAMES),
            f"{image_id}:{owner_id}: category review")
    require(entry.get("scene_scope") == "real_original_scene", f"{image_id}:{owner_id}: only real original scene objects may be admitted")
    review_category = entry.get("review_category")
    require(isinstance(review_category, str) and review_category and
            review_category not in {"toy_or_figurine", "depicted_in_wall_photo", "image_within_image"},
            f"{image_id}:{owner_id}: individual scene review category")
    require(entry.get("decision") in {"distinct_real_owner", "resolve_prior_unknown_category"},
            f"{image_id}:{owner_id}: physical identity/category decision")
    # Retain the original export's seven provenance buckets, adding the
    # current lead receipt and per-owner visual chain rather than guessing facts.
    return {
        "stable_owner_id": owner_id,
        "bbox_2d": [f"<|coord_{value}|>" for value in bins],
        "bbox_2d_bins_1000": bins,
        "coordinate_convention": "normalized discrete xyxy bins 0..999; native <|coord_N|> spelling",
        "category_id": None,
        "category_name": description,
        "desc": description,
        "class_status": status,
        "physical_status": "valid_unlabeled",
        "geometry_status": "reasonable",
        "scene_scope": "real_original_scene",
        "review_category": review_category,
        "reference_status": entry.get("reference_status"),
        "reference_proposal_id": entry.get("reference_proposal_id"),
        "provenance": {
            "target_catalog": [], "class_mask_ruling": [],
            "review_source_index": [dict(source)],
            "admission": [dict(source)],
            "geometry": [dict(visual["overlay"])],
            "crop": [dict(visual["crop"])],
            "recorded_visual_or_review": [dict(visual[name]) for name in ("original", "overlay", "crop")] + [dict(source)],
        },
    }


def _checked_admissions(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return [], None
    source = binding(path)
    value = prior.read(Path(source["path"]))
    require(isinstance(value, Mapping) and value.get("status") == "lead-accepted", "owner admissions must be lead-accepted")
    entries = value.get("owners")
    require(isinstance(entries, list) and entries and all(isinstance(entry, Mapping) for entry in entries), "lead-admitted owner records")
    return [dict(entry) for entry in entries], source


def build_version(
    *,
    predecessor_annotations_path: Path = PREDECESSOR,
    new_originals_path: Path | None = None,
    lead_admission_path: Path | None = None,
    output: Path,
) -> dict[str, Any]:
    """Make a new complete JSONL version from an accepted predecessor.

    ``new_originals_path`` supplies previously absent images only at first
    11→22 publication. Later versions pass the 22-image predecessor alone.
    An empty first admission creates the initial 22-image ledger without
    inventing additional owners.
    """

    predecessor = binding(predecessor_annotations_path)
    source_rows = _rows(Path(predecessor["path"]))
    require(len(source_rows) in {11, 22}, "predecessor must contain accepted 11 or 22 images")
    fresh = _rows(new_originals_path) if new_originals_path is not None else []
    require((len(source_rows), len(fresh)) in {(11, 11), (22, 0)}, "11→22 first publication or 22-image successor")
    original_binding = binding(new_originals_path) if new_originals_path is not None else None
    old_ids = {row["image_id"] for row in source_rows}
    require(not any(row["image_id"] in old_ids for row in fresh), "duplicate predecessor/new image")
    for row in source_rows:
        require(isinstance(row.get("objects"), list) and isinstance(row.get("unlabeled"), list), "accepted predecessor object lists")
    for row in fresh:
        require(isinstance(row.get("objects"), list) and "unlabeled" not in row, "new originals must be raw, unadmitted image rows")
    rows = copy.deepcopy(source_rows) + [{**copy.deepcopy(row), "unlabeled": []} for row in fresh]
    by_image = {row["image_id"]: row for row in rows}
    require(len(by_image) == 22, "22-image annotation version")
    admissions, admission_binding = _checked_admissions(lead_admission_path)
    known_ids: set[str] = set()
    for row in rows:
        for item in row["objects"]:
            known_ids.add(str(item.get("coco_ann_id")))
        for item in row["unlabeled"]:
            stable = item.get("stable_owner_id")
            require(isinstance(stable, str) and stable not in known_ids, "duplicate existing accepted owner ID")
            known_ids.add(stable)
    new_owner_count, category_resolution_count = 0, 0
    for entry in admissions:
        image_id, owner_id = entry.get("image_id"), entry.get("stable_owner_id")
        require(type(image_id) is int and image_id in by_image and isinstance(owner_id, str) and owner_id, "lead-admitted owner/image identity")
        row = by_image[image_id]
        visual = _visual(entry.get("visual"), image_id=image_id, owner_id=owner_id)
        candidate = _owner_record(entry, source=admission_binding, visual=visual)
        if entry["decision"] == "resolve_prior_unknown_category":
            previous = [item for item in row["unlabeled"] if item.get("stable_owner_id") == owner_id]
            require(len(previous) == 1 and previous[0].get("class_status") == "unknown" and
                    candidate["class_status"] == "verified" and
                    previous[0].get("bbox_2d_bins_1000") == candidate["bbox_2d_bins_1000"],
                    f"{image_id}:{owner_id}: category resolution requires same prior unknown physical owner/geometry")
            previous[0].update({key: candidate[key] for key in
                                ("category_name", "desc", "class_status", "scene_scope", "review_category")})
            previous[0]["provenance"] = copy.deepcopy(previous[0]["provenance"])
            previous[0]["provenance"]["category_resolution_admission"] = (
                [dict(admission_binding)] + [dict(visual[key]) for key in ("original", "overlay", "crop")]
            )
            category_resolution_count += 1
            continue
        require(owner_id not in known_ids, f"duplicate lead-admitted owner: {owner_id}")
        overlaps = []
        for gt, kind in ((True, "objects"), (False, "unlabeled")):
            for index, item in enumerate(row[kind]):
                previous = _existing_bbox(item, image_id=image_id, index=index, gt=gt)
                if readback_selectors.iou_xyxy(candidate["bbox_2d_bins_1000"], previous) >= 0.5:
                    overlaps.append(str(item["coco_ann_id"] if gt else item["stable_owner_id"]))
        reviewed_distinct = entry.get("distinct_from_owner_ids", [])
        require(isinstance(reviewed_distinct, list) and set(overlaps) <= set(reviewed_distinct),
                f"{image_id}:{owner_id}: overlapping physical owner requires distinctness ruling")
        row["unlabeled"].append(candidate)
        known_ids.add(owner_id)
        new_owner_count += 1
    rows.sort(key=lambda row: row["image_id"])
    path = output / "annotations.jsonl"
    manifest_path = output / "manifest.json"
    require(not output.exists() and not output.is_symlink(), f"annotation version collision: {output}")
    output.mkdir(parents=True)
    content = b"".join(canonical(row) for row in rows)
    with path.open("xb") as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())
    require(path.read_bytes() == content, "annotation version readback")
    result = {
        "schema": SCHEMA, "status": "candidate_ready_for_lead_annotation_admission",
        "annotations": binding(path),
        "sources": {"predecessor_annotations": predecessor, "new_originals": original_binding,
                    "owner_root_admission": admission_binding, "producer": binding(Path(__file__))},
        "counts": {"images": 22, "gt_objects": sum(len(row["objects"]) for row in rows),
                   "valid_unlabeled": sum(len(row["unlabeled"]) for row in rows),
                   "newly_admitted": new_owner_count, "resolved_prior_unknown": category_resolution_count},
        "frozen_bank_policy": "Versioned annotations do not update this run's frozen teacher or target denominator.",
    }
    prior.publish(manifest_path, result)
    require(prior.read(manifest_path) == result, "annotation version manifest readback")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, default=PREDECESSOR)
    parser.add_argument("--new-originals", type=Path)
    parser.add_argument("--lead-admission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_version(predecessor_annotations_path=args.predecessor, new_originals_path=args.new_originals,
                          lead_admission_path=args.lead_admission, output=args.output)
    print(json.dumps({"annotations": value["annotations"], "counts": value["counts"]}))


if __name__ == "__main__":
    main()
