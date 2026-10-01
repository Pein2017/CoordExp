#!/usr/bin/env python3
"""Refine the existing COCO/LVIS coord proxy and add two vehicle subclasses.

The input is the materialized no-max60 proxy. Its image paths remain valid because
the output is a sibling of the input directory under public_data/coco.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import ijson


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000"
OUTPUT = ROOT / "public_data/coco/rescale_32_1024_bbox_lvis_proxy_v2"
HARD_OUTPUT = ROOT / "public_data/coco/rescale_32_1024_bbox_lvis_proxy_v2_hard"
LVIS = ROOT / "public_data/lvis/raw/annotations"
SUBCLASSES = {
    800: ("pickup_truck", 8, "truck"),
    207: ("car_(automobile)", 3, "car"),
}
DEPICTION_CONTAINERS = {"person", "tv", "laptop", "book", "cell phone"}
VEHICLES = {"car", "truck", "bus"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(obj: dict) -> tuple[int, int, int, int]:
    values = obj["bbox_2d"]
    return tuple(int(value[8:-2]) for value in values)  # type: ignore[return-value]


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[float, float]:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    if not intersection:
        return 0.0, 0.0
    area_a, area_b = _area(a), _area(b)
    return intersection / (area_a + area_b - intersection), intersection / area_a


def _tokens(bbox: list[float], width: int, height: int) -> list[str] | None:
    x, y, w, h = bbox
    if width <= 1 or height <= 1 or not all(math.isfinite(v) for v in bbox) or w <= 0 or h <= 0:
        return None
    x1 = max(0, min(999, math.floor(x / (width - 1) * 999)))
    y1 = max(0, min(999, math.floor(y / (height - 1) * 999)))
    x2 = max(0, min(999, math.ceil((x + w) / (width - 1) * 999)))
    y2 = max(0, min(999, math.ceil((y + h) / (height - 1) * 999)))
    if x2 <= x1:
        x1, x2 = (998, 999) if x1 == 999 else (x1, x1 + 1)
    if y2 <= y1:
        y1, y2 = (998, 999) if y1 == 999 else (y1, y1 + 1)
    return [f"<|coord_{value}|>" for value in (x1, y1, x2, y2)]


def _lvis_candidates() -> tuple[dict[int, list[dict]], dict[int, tuple[int, int]]]:
    selected: dict[int, list[dict]] = collections.defaultdict(list)
    for split in ("train", "val"):
        path = LVIS / f"lvis_v1_{split}.json"
        with path.open("rb") as stream:
            for ann in ijson.items(stream, "annotations.item"):
                if ann["category_id"] in SUBCLASSES:
                    selected[ann["image_id"]].append(
                        {"id": ann["id"], "category_id": ann["category_id"], "bbox": ann["bbox"]}
                    )
    dimensions: dict[int, tuple[int, int]] = {}
    for split in ("train", "val"):
        path = LVIS / f"lvis_v1_{split}.json"
        with path.open("rb") as stream:
            for image in ijson.items(stream, "images.item"):
                if image["id"] in selected:
                    dimensions[image["id"]] = (image["width"], image["height"])
    if set(selected) != set(dimensions):
        raise ValueError("LVIS selected annotations lack image dimensions")
    return selected, dimensions


def _old_proxy_rejection(obj: dict, coco_objects: list[dict]) -> str | None:
    source, target = obj.get("lvis_category_name"), obj.get("category_name")
    box = _box(obj)
    if (source, target) == ("soap", "bottle"):
        return "soap_is_not_bottle"
    if (source, target) == ("computer_keyboard", "keyboard"):
        if any(o["category_name"] == "laptop" and _overlap(box, _box(o))[1] >= 0.8 for o in coco_objects):
            return "integrated_laptop_keyboard"
    if (source, target) == ("person", "person"):
        people = [o for o in coco_objects if o["category_name"] == "person"]
        if not people:
            return "person_no_coco_person_context"
        if _area(box) < 100 or min(box[2] - box[0], box[3] - box[1]) < 8:
            return "person_tiny_or_fragment"
        if any(_overlap(box, _box(o))[0] >= 0.5 or _overlap(box, _box(o))[1] >= 0.7 for o in people):
            return "person_duplicate_or_body_part"
        if any(
            o["category_name"] in DEPICTION_CONTAINERS - {"person"}
            and _overlap(box, _box(o))[1] >= 0.8
            for o in coco_objects
        ):
            return "person_in_depiction"
    return None


def _new_proxy_rejection(obj: dict, existing: list[dict]) -> str | None:
    box = _box(obj)
    if _area(box) < 50 or min(box[2] - box[0], box[3] - box[1]) < 4:
        return "source_too_small_for_stable_box"
    target = obj["category_name"]
    competing = VEHICLES
    for old in existing:
        other = old["category_name"]
        iou, coverage = _overlap(box, _box(old))
        if other in competing and (iou >= 0.35 or coverage >= 0.7):
            return "existing_target_or_competing_class"
        if other in DEPICTION_CONTAINERS and coverage >= 0.8:
            return "source_inside_person_or_depiction"
    return None


def _refine(row: dict, annotations: list[dict], raw_size: tuple[int, int] | None, counts: collections.Counter) -> dict:
    objects = row["objects"]
    namespace = row["metadata"]["coordexp_proxy_supervision"]
    supervision = namespace["object_supervision"]
    if len(objects) != len(supervision):
        raise ValueError(f"image {row['image_id']}: object/supervision count mismatch")
    coco_objects = [obj for obj in objects if "coco_ann_id" in obj]
    pairs: list[tuple[dict, dict]] = []
    for obj, entry in zip(objects, supervision, strict=True):
        reason = _old_proxy_rejection(obj, coco_objects) if "lvis_ann_id" in obj else None
        if reason:
            counts[f"removed/{reason}"] += 1
        else:
            if obj.get("lvis_category_name") == "person":
                # Geometry alone cannot certify a distinct whole person here.
                entry = {
                    **entry,
                    "proxy_tier": "plausible",
                    "mapping_class": "person_presence_candidate",
                    "desc_ce_weight": 0.25,
                    "coord_weight": 0.0,
                    "why_recovered": "LVIS person candidate; no hard instance claim after fragment audit",
                }
                counts["person_downgraded_to_weak"] += 1
            pairs.append((obj, entry))
    if annotations and raw_size is None:
        raise ValueError(f"image {row['image_id']}: missing LVIS raw dimensions")
    existing_ids = {obj["lvis_ann_id"] for obj, _ in pairs if "lvis_ann_id" in obj}
    for ann in annotations:
        if ann["id"] in existing_ids:
            counts["skipped/already_present"] += 1
            continue
        source, target_id, target = SUBCLASSES[ann["category_id"]]
        bbox = _tokens(ann["bbox"], *raw_size)  # type: ignore[arg-type]
        if bbox is None:
            counts["skipped/invalid_bbox"] += 1
            continue
        obj = {
            "bbox_2d": bbox,
            "desc": target,
            "category_id": target_id,
            "category_name": target,
            "lvis_ann_id": ann["id"],
            "lvis_category_id": ann["category_id"],
            "lvis_category_name": source,
            "proxy_source": "lvis",
        }
        reason = _new_proxy_rejection(obj, [old for old, _ in pairs])
        if reason:
            counts[f"skipped/{source}/{reason}"] += 1
            continue
        entry = {
            "source": "lvis",
            "proxy_tier": "plausible",
            "mapping_class": "same_object_subcategory",
            "desc_ce_weight": 0.5,
            "coord_weight": 0.5,
            "mapping_kind": "semantic_subcategory_v2",
            "mapped_coco_category_id": target_id,
            "mapped_coco_category_name": target,
            "lvis_ann_id": ann["id"],
            "lvis_category_id": ann["category_id"],
            "lvis_category_name": source,
            "why_recovered": "LVIS same-object subtype; geometry-filtered against existing COCO/LVIS objects",
        }
        pairs.append((obj, entry))
        existing_ids.add(ann["id"])
        counts[f"added/{source}->{target}"] += 1
    pairs.sort(key=lambda pair: (_box(pair[0])[1], _box(pair[0])[0]))
    row["objects"] = [obj for obj, _ in pairs]
    namespace["object_supervision"] = [entry for _, entry in pairs]
    namespace["summary"] = {
        "real_count": sum(entry["proxy_tier"] == "real" for _, entry in pairs),
        "strict_count": sum(entry["proxy_tier"] == "strict" for _, entry in pairs),
        "plausible_count": sum(entry["proxy_tier"] == "plausible" for _, entry in pairs),
        "include_plausible": True,
    }
    namespace["policy_version"] = "coco-lvis-proxy-v2"
    counts["objects"] += len(pairs)
    counts["records"] += 1
    counts["records_gt_60"] += len(pairs) > 60
    counts["max_objects"] = max(counts["max_objects"], len(pairs))
    return row


def _hard_view(row: dict, counts: collections.Counter) -> dict | None:
    """Use only full-label proxies in the active coordexp-infras JSONL contract."""
    old_entries = row["metadata"]["coordexp_proxy_supervision"]["object_supervision"]
    objects = []
    for obj, entry in zip(row["objects"], old_entries, strict=True):
        if "coco_ann_id" in obj:
            objects.append(obj)
            continue
        if entry["proxy_tier"] != "strict" and entry.get("mapping_kind") != "semantic_subcategory_v2":
            counts["weak_proxy_excluded"] += 1
            continue
        objects.append({
            "bbox_2d": obj["bbox_2d"],
            "desc": obj["desc"],
            "category_id": obj["category_id"],
            "category_name": obj["category_name"],
            "coco_ann_id": -int(obj["lvis_ann_id"]),
            "metadata": {
                "proxy_source": "lvis",
                "lvis_ann_id": obj["lvis_ann_id"],
                "lvis_category_id": obj["lvis_category_id"],
                "lvis_category_name": obj["lvis_category_name"],
                "mapping_class": entry["mapping_class"],
                "weighted_view_desc_ce_weight": entry["desc_ce_weight"],
                "weighted_view_coord_weight": entry["coord_weight"],
            },
        })
        counts["hard_proxy_objects"] += 1
    if not objects:
        counts["empty_rows_excluded"] += 1
        return None
    row["objects"] = objects
    row["metadata"] = {
        "source": row["metadata"]["source"],
        "split": row["metadata"]["split"],
        "hard_proxy_policy": "strict_or_same_object_subcategory_v2; all emitted objects have full training weight",
    }
    counts["records"] += 1
    counts["objects"] += len(objects)
    counts["records_gt_60"] += len(objects) > 60
    counts["max_objects"] = max(counts["max_objects"], len(objects))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--hard-output", type=Path, default=HARD_OUTPUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    hard_output = args.hard_output.resolve()
    if not (source.parent == output.parent == hard_output.parent):
        raise ValueError("source and outputs must be siblings so image paths remain valid")
    if len({source, output, hard_output}) != 3:
        raise ValueError("source and output directories must be distinct")
    output.mkdir(parents=True, exist_ok=True)
    hard_output.mkdir(parents=True, exist_ok=True)
    selected, dimensions = _lvis_candidates()
    stats: dict[str, dict] = {}
    hard_stats: dict[str, dict] = {}
    for split in ("train", "val"):
        src = source / f"{split}.coord.jsonl"
        dst = output / f"{split}.coord.jsonl"
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        counts: collections.Counter = collections.Counter()
        with src.open(encoding="utf-8") as input_stream, tmp.open("w", encoding="utf-8") as output_stream:
            for line in input_stream:
                row = json.loads(line)
                image_id = row["image_id"]
                row = _refine(row, selected.get(image_id, []), dimensions.get(image_id), counts)
                output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(dst)
        stats[split] = {"counts": dict(sorted(counts.items())), "source_sha256": _sha256(src), "output_sha256": _sha256(dst)}
        hard_dst = hard_output / f"{split}.coord.jsonl"
        hard_tmp = hard_dst.with_suffix(hard_dst.suffix + ".tmp")
        hard_counts: collections.Counter = collections.Counter()
        with dst.open(encoding="utf-8") as input_stream, hard_tmp.open("w", encoding="utf-8") as output_stream:
            for line in input_stream:
                row = _hard_view(json.loads(line), hard_counts)
                if row is not None:
                    output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        hard_tmp.replace(hard_dst)
        hard_stats[split] = {
            "counts": dict(sorted(hard_counts.items())),
            "source_sha256": _sha256(dst),
            "output_sha256": _sha256(hard_dst),
        }
    manifest = {
        "artifact_id": "coco-lvis-proxy-v2",
        "format": "train/val.coord.jsonl with coord tokens and aligned coordexp_proxy_supervision",
        "source_root": str(source),
        "output_root": str(output),
        "image_root": str(source.parent / "rescale_32_1024_bbox/images"),
        "object_cap": None,
        "source_token_budget": 12000,
        "output_token_budget_filter": None,
        "new_mapping_weights": {"desc_ce_weight": 0.5, "coord_weight": 0.5},
        "tablecloth_policy": "retain legacy plausible presence supervision at 0.25/0.0; no table-box claim",
        "withheld_mapping": "armchair->chair: rejected after source-box visual audit of unmatched candidates",
        "script_sha256": _sha256(Path(__file__)),
        "lvis_annotations_sha256": {
            split: _sha256(LVIS / f"lvis_v1_{split}.json") for split in ("train", "val")
        },
        "splits": stats,
    }
    (output / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hard_manifest = {
        "artifact_id": "coco-lvis-proxy-v2-hard",
        "format": "current coordexp-infras train/val.coord.jsonl",
        "source_root": str(output),
        "output_root": str(hard_output),
        "object_cap": None,
        "training_weights": "all objects are fully supervised; weak presence-only proxies omitted",
        "synthetic_coco_ann_id": "negative LVIS annotation id, with original provenance in object.metadata",
        "script_sha256": _sha256(Path(__file__)),
        "splits": hard_stats,
    }
    (hard_output / "pipeline_manifest.json").write_text(
        json.dumps(hard_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"weighted": {split: data["counts"] for split, data in stats.items()}, "hard": {split: data["counts"] for split, data in hard_stats.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
