"""Fixed COCO box-record recovery, not a general data factory.

The grid-size calculation is the original recovery algorithm. Raw annotations
are ordered by image/annotation ID; resized pixels use round-and-clamp; normalized
boxes use floor/ceil on (width-1,height-1). Pixel and normalized rows are sorted
independently. No model, tokenizer, proxy labels or on-the-fly resizing is used.
"""
from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any

IMAGE_FACTOR = 32
MIN_PIXELS = 4096
MAX_PIXELS = 1048576
MAX_RATIO = 200

def _ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest integer >= ``number`` divisible by ``factor``."""
    return math.ceil(number / factor) * factor

def _floor_by_factor(number: int, factor: int) -> int:
    """Return the largest integer <= ``number`` divisible by ``factor``."""
    return math.floor(number / factor) * factor

def _aligned_candidates(value: float, factor: int) -> set[int]:
    """Return a small neighborhood of factor-aligned candidates around ``value``."""
    value_int = max(1, int(round(value)))
    floor_value = max(factor, _floor_by_factor(value_int, factor))
    ceil_value = max(factor, _ceil_by_factor(value_int, factor))
    return {
        candidate
        for candidate in {
            floor_value - factor,
            floor_value,
            ceil_value,
            ceil_value + factor,
        }
        if candidate >= factor
    }

def _maximize_aligned_size(
    *,
    height: int,
    width: int,
    factor: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Return the closest aspect-preserving aligned size under ``max_pixels``.

    The search stays local around the continuous optimum, then prefers the candidate
    with the smallest aspect-ratio error and, among ties, the largest area.
    """
    area = height * width
    scale = math.sqrt(max_pixels / area)
    ideal_h = height * scale
    ideal_w = width * scale
    aspect = float(width) / float(height)

    candidates: set[tuple[int, int]] = set()

    def _add_candidate(candidate_h: int, candidate_w: int) -> None:
        if candidate_h < factor or candidate_w < factor:
            return
        if candidate_h % factor != 0 or candidate_w % factor != 0:
            return
        if candidate_h * candidate_w > max_pixels:
            return
        candidates.add((candidate_h, candidate_w))

    for candidate_h in _aligned_candidates(ideal_h, factor):
        max_w_budget = max(factor, _floor_by_factor(max_pixels // candidate_h, factor))
        for candidate_w in _aligned_candidates(candidate_h * aspect, factor) | {
            max_w_budget,
            max_w_budget - factor,
        }:
            bounded_w = min(max_w_budget, candidate_w)
            if bounded_w >= factor:
                _add_candidate(candidate_h, bounded_w)

    for candidate_w in _aligned_candidates(ideal_w, factor):
        max_h_budget = max(factor, _floor_by_factor(max_pixels // candidate_w, factor))
        for candidate_h in _aligned_candidates(candidate_w / aspect, factor) | {
            max_h_budget,
            max_h_budget - factor,
        }:
            bounded_h = min(max_h_budget, candidate_h)
            if bounded_h >= factor:
                _add_candidate(bounded_h, candidate_w)

    floor_h = max(factor, _floor_by_factor(int(ideal_h), factor))
    floor_w = max(factor, _floor_by_factor(int(ideal_w), factor))
    _add_candidate(floor_h, floor_w)

    if not candidates:
        return factor, factor

    def _score(size: tuple[int, int]) -> tuple[float, int, float]:
        candidate_h, candidate_w = size
        ratio_error = abs(math.log((candidate_w / candidate_h) / aspect))
        ideal_error = abs(candidate_h - ideal_h) + abs(candidate_w - ideal_w)
        return (ratio_error, -(candidate_h * candidate_w), ideal_error)

    return min(candidates, key=_score)

def smart_resize(
    *,
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    max_ratio: int = MAX_RATIO,
) -> tuple[int, int]:
    """Compute resized dimensions under the detection pixel budget.

    The resized dimensions:
    - Preserve aspect ratio as closely as grid alignment allows
    - Snap to multiples of :data:`factor`
    - Maximize resolution under :data:`max_pixels`
    - Reject extreme aspect ratios that would break patch grids

    ``min_pixels`` is retained for API compatibility with existing callers. Under
    the current maximize-under-cap policy, practical configs are expected to keep
    ``max_pixels >= min_pixels``.
    """
    height = int(height)
    width = int(width)
    factor = int(factor)
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)

    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be positive, got {(height, width)}")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if max_pixels <= 0:
        raise ValueError(f"max_pixels must be positive, got {max_pixels}")
    if min_pixels <= 0:
        raise ValueError(f"min_pixels must be positive, got {min_pixels}")
    if max_pixels < min_pixels:
        raise ValueError(
            f"max_pixels must be >= min_pixels, got max_pixels={max_pixels}, min_pixels={min_pixels}"
        )

    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, "
            f"got {max(height, width) / min(height, width)}"
        )

    return _maximize_aligned_size(
        height=height,
        width=width,
        factor=factor,
        max_pixels=max_pixels,
    )

def raw_records(document: Mapping[str, Any], split: str) -> Iterator[dict[str, Any]]:
    """Reconstruct the accepted noncrowd, nonempty COCO bbox records."""
    if split not in {"train", "val"}:
        raise ValueError("only COCO2017 train and val are supported")
    categories = {int(c["id"]): str(c["name"]) for c in document["categories"]}
    images = {int(im["id"]): im for im in document["images"]}
    if len(images) != len(document["images"]):
        raise ValueError("duplicate raw image identity")
    annotations = defaultdict(list)
    for ann in document["annotations"]:
        annotations[int(ann["image_id"])].append(ann)
    for image_id, image in sorted(images.items()):
        width, height = int(image["width"]), int(image["height"])
        name = image["file_name"]
        if width <= 0 or height <= 0 or not isinstance(name, str) or "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise ValueError("invalid raw image path/dimensions")
        objects = []
        for ann in sorted(annotations[image_id], key=lambda a: (int(a["id"]), int(a["category_id"]))):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            x, y, w, h = map(float, ann["bbox"])
            box = [x, y, x+w, y+h]
            if not all(math.isfinite(v) for v in box):
                raise ValueError("nonfinite raw bbox")
            if w <= 1e-6 or h <= 1e-6 or x < -1e-6 or y < -1e-6 or x+w > width+1e-6 or y+h > height+1e-6:
                continue
            category = int(ann["category_id"])
            if category not in categories:
                raise ValueError("raw annotation category is unknown")
            objects.append({"bbox_2d": box, "desc": categories[category],
                            "category_id": category, "category_name": categories[category],
                            "coco_ann_id": int(ann["id"])})
        if objects:
            path = f"images/{split}2017/{name}"
            yield {"images": [path], "objects": objects, "width": width,
                   "height": height, "image_id": image_id, "file_name": path,
                   "metadata": {"source": "coco2017", "split": split}}


def pixel_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the original aligned resize and exact integer pixel boxes."""
    row = copy.deepcopy(dict(raw))
    old_w, old_h = int(row["width"]), int(row["height"])
    height, width = smart_resize(height=old_h, width=old_w)
    row["width"], row["height"] = width, height
    if (width, height) == (old_w, old_h):
        return row
    for obj in row["objects"]:
        scaled = [max(0, min((width if i % 2 == 0 else height)-1,
                            int(round(float(v)*(width/old_w if i % 2 == 0 else height/old_h)))))
                  for i, v in enumerate(obj["bbox_2d"])]
        for start, end, bound in [(0, 2, width), (1, 3, height)]:
            if scaled[end] < scaled[start]:
                scaled[start], scaled[end] = scaled[end], scaled[start]
            if scaled[end] <= scaled[start]:
                if scaled[end] < bound-1:
                    scaled[end] = min(bound-1, scaled[start]+1)
                elif scaled[start] > 0:
                    scaled[start] = max(0, scaled[end]-1)
        obj["bbox_2d"] = scaled
    return row


def coordinate_triplet(pixel: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Reconstruct this frozen all-record view, not a generic token-budget filter.

    The accepted corpus dropped zero records for its 12k bound. Its exact
    archive and final output hashes are mandatory at the recovery entry point.
    Different corpora require a new independently qualified recipe.
    """
    p = copy.deepcopy(dict(pixel))
    n = copy.deepcopy(dict(pixel))
    def order(obj):
        return obj["bbox_2d"][1], obj["bbox_2d"][0]
    p["objects"] = sorted(p["objects"], key=order)
    for obj in n["objects"]:
        coords = []
        for i, value in enumerate(obj["bbox_2d"]):
            size = n["width"] if i % 2 == 0 else n["height"]
            value = float(value) / max(1.0, float(size)-1.0) * 999.0
            coords.append(max(0, min(999, math.floor(value) if i < 2 else math.ceil(value))))
        for a, b in [(0, 2), (1, 3)]:
            if coords[b] <= coords[a]:
                if coords[a] < 999:
                    coords[b] = coords[a]+1
                else:
                    coords[a], coords[b] = 998, 999
        obj["bbox_2d"] = coords
    n["objects"] = sorted(n["objects"], key=order)
    c = copy.deepcopy(n)
    for obj in c["objects"]:
        obj["bbox_2d"] = [f"<|coord_{v}|>" for v in obj["bbox_2d"]]
    for row in (p, n, c):
        row["images"] = ["../rescale_32_1024_bbox/" + name for name in row["images"]]
    return p, n, c


def jsonl_bytes(row: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def apply_annotation_edit(row: dict[str, Any], edit: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply a content-bound observed annotation delta, without new label admission."""
    if edit is None:
        return row
    import hashlib
    if hashlib.sha256(jsonl_bytes(row)).hexdigest() != edit["before_sha256"]:
        raise ValueError("annotation delta input identity differs")
    objects = {o["coco_ann_id"]: o for o in row["objects"]}
    if len(objects) != len(row["objects"]):
        raise ValueError("annotation IDs are not unique")
    for ann_id in edit["remove_annotation_ids"]:
        if ann_id not in objects:
            raise ValueError("removed annotation ID is absent")
        del objects[ann_id]
    for obj in edit["upsert_objects"]:
        objects[obj["coco_ann_id"]] = copy.deepcopy(obj)
    order = edit["object_order"]
    if len(set(order)) != len(order) or set(order) != set(objects):
        raise ValueError("annotation delta order does not cover exact objects")
    out = copy.deepcopy(row)
    out["objects"] = [objects[i] for i in order]
    if hashlib.sha256(jsonl_bytes(out)).hexdigest() != edit["after_sha256"]:
        raise ValueError("annotation delta result identity differs")
    return out


def curated_triplet(pixel: Mapping[str, Any], edits: Mapping[tuple, Mapping]) -> tuple[dict[str, Any], ...]:
    split, image_id = pixel["metadata"]["split"], pixel["image_id"]
    p, _, _ = coordinate_triplet(pixel)
    p = apply_annotation_edit(p, edits.get((split, image_id, "pixel")))
    original_paths = copy.deepcopy(p)
    original_paths["images"] = list(pixel["images"])
    _, n, _ = coordinate_triplet(original_paths)
    n = apply_annotation_edit(n, edits.get((split, image_id, "norm1000")))
    c = copy.deepcopy(n)
    for obj in c["objects"]:
        obj["bbox_2d"] = [f"<|coord_{v}|>" for v in obj["bbox_2d"]]
    return p, n, c


def compact_jsonl_bytes(row: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(row), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
