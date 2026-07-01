from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ID = "post_x1_instance_basin_tomography"
PHASE_ID = "phase_a3_3"
SCHEMA_VERSION = "a3.3.v1"
BBOX_NORMALIZATION_POLICY_ID = "a3_3_coord_token_xyxy_v1"
PRIMARY_BASIN_LABEL_SOURCE = "same_desc_gt_instances"

_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d{1,3})\|>$")
_WS_RE = re.compile(r"\s+")


def canonical_desc(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def normalize_sample_objects(
    sample: Mapping[str, Any],
    *,
    split: str,
    source_line_id: int,
    source_jsonl_path: str | Path | None = None,
    source_jsonl_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize one JSONL sample's objects without losing source bbox surface.

    The len12000 training/val surface is ``bbox_2d`` coord-token strings. Numeric
    boxes are retained for unit fixtures and legacy dry-runs only, and are labeled
    as fixture surfaces instead of being passed off as primary pixel boxes.
    """

    objects: list[dict[str, Any]] = []
    raw_objects = sample.get("objects") or []
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes)):
        return objects

    source_path = None if source_jsonl_path is None else Path(source_jsonl_path)
    source_sha = source_jsonl_sha256
    if source_sha is None and source_path is not None:
        source_sha = sha256_path(source_path)

    for gt_idx, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            continue
        desc = canonical_desc(str(raw.get("desc") or raw.get("label") or ""))
        bbox_info = _bbox_coord_token_xyxy_from_raw(raw)
        if not desc or bbox_info is None:
            continue
        bbox, surface, source_field = bbox_info
        if not _valid_xyxy(bbox):
            continue
        obj = {
            "gt_idx": gt_idx,
            "original_index": gt_idx,
            "desc": desc,
            "bbox_coord_token_xyxy": list(bbox),
            "bbox_xyxy": list(bbox),
            "bbox_surface": surface,
            "bbox_source_field": source_field,
            "source_bbox_surface": surface,
            "source_bbox_field": source_field,
            "bbox_normalization_policy_id": BBOX_NORMALIZATION_POLICY_ID,
            "split": split,
            "source_line_id": int(source_line_id),
            "source_jsonl_path": None if source_path is None else str(source_path),
            "source_jsonl_sha256": source_sha,
        }
        objects.append(obj)
    return objects


def build_case_universe_rows(
    samples: Sequence[Mapping[str, Any]],
    *,
    split: str,
    max_images: int | None = None,
    max_target_instances: int | None = None,
    min_same_desc_count: int = 2,
    source_jsonl_path: str | Path | None = None,
    source_jsonl_sha256: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_budget = len(samples) if max_images is None else max(0, int(max_images))
    target_budget = (
        math.inf
        if max_target_instances is None
        else max(0, int(max_target_instances))
    )

    for source_line_id, sample in enumerate(samples[:image_budget]):
        objects = normalize_sample_objects(
            sample,
            split=split,
            source_line_id=source_line_id,
            source_jsonl_path=source_jsonl_path,
            source_jsonl_sha256=source_jsonl_sha256,
        )
        if not objects:
            continue
        by_desc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for obj in objects:
            by_desc[str(obj["desc"])].append(obj)

        for desc in sorted(by_desc):
            members = sorted(by_desc[desc], key=lambda obj: int(obj["gt_idx"]))
            if len(members) < min_same_desc_count:
                continue
            same_desc_gt_indices = [int(obj["gt_idx"]) for obj in members]
            for target in members:
                if len(rows) >= target_budget:
                    return rows
                rows.append(
                    _case_row(
                        sample=sample,
                        split=split,
                        source_line_id=source_line_id,
                        source_jsonl_path=source_jsonl_path,
                        source_jsonl_sha256=target.get("source_jsonl_sha256"),
                        objects=objects,
                        target=target,
                        members=members,
                        same_desc_gt_indices=same_desc_gt_indices,
                    )
                )
    return rows


def strict_r95_radius(axis_len: int, *, fraction: float = 0.04, cap: int = 8) -> int:
    return math.floor(min(int(cap), float(fraction) * max(0, int(axis_len))))


def compute_anchor_r95(
    *,
    axis_len: int | float,
    axis_fraction: float = 0.04,
    cap_bins: int = 8,
) -> int:
    return strict_r95_radius(int(axis_len), fraction=axis_fraction, cap=cap_bins)


def parse_coord_bbox(value: Sequence[Any]) -> tuple[int, int, int, int]:
    bbox = _coord_token_bbox(value)
    if bbox is None:
        bbox = _numeric_bbox(value)
    if bbox is None:
        raise ValueError("bbox must be four coord tokens or numeric coordinates")
    return tuple(bbox)  # type: ignore[return-value]


def build_case_universe(
    jsonl_path: str | Path,
    *,
    split: str,
    max_cases: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(jsonl_path)
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                import json

                samples.append(json.loads(line))
    rows = build_case_universe_rows(
        samples,
        split=split,
        max_images=len(samples),
        max_target_instances=max_cases,
        min_same_desc_count=1,
        source_jsonl_path=path,
    )
    for row in rows:
        if row.get("anchor_ambiguity_bucket") in {"exact_x1_collision", "near_collision"}:
            row["anchor_ambiguity_bucket"] = "ambiguous_same_desc_x1"
        row["source_jsonl"] = row.get("source_jsonl_path")
        row["object_index"] = row.get("target_gt_idx")
        row["image_objects"] = row.get("objects", [])
    return rows, {
        "split": split,
        "row_count": len(rows),
        "source_jsonl": str(path),
        "source_jsonl_sha256": sha256_path(path),
    }


def sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_row(
    *,
    sample: Mapping[str, Any],
    split: str,
    source_line_id: int,
    source_jsonl_path: str | Path | None,
    source_jsonl_sha256: str | None,
    objects: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    same_desc_gt_indices: Sequence[int],
) -> dict[str, Any]:
    target_gt_idx = int(target["gt_idx"])
    competitors = [obj for obj in members if int(obj["gt_idx"]) != target_gt_idx]
    competitor_gt_indices = [int(obj["gt_idx"]) for obj in competitors]
    anchor = _x1_anchor_bucket(target=target, competitors=competitors)
    image_id = _image_id(sample=sample, source_line_id=source_line_id)
    image_path = _image_path(sample)
    source_path = None if source_jsonl_path is None else str(Path(source_jsonl_path))
    target_bbox = list(target["bbox_coord_token_xyxy"])
    row = {
        "project_id": PROJECT_ID,
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "case_id": _case_id(split, image_id, str(target["desc"]), target_gt_idx),
        "split": split,
        "image_id": image_id,
        "image_path": image_path,
        "width": sample.get("width"),
        "height": sample.get("height"),
        "desc": target["desc"],
        "target_gt_idx": target_gt_idx,
        "bbox_coord_token_xyxy": target_bbox,
        "bbox_surface": target["bbox_surface"],
        "bbox_source_field": target["bbox_source_field"],
        "source_bbox_surface": target["bbox_surface"],
        "source_bbox_field": target["bbox_source_field"],
        "bbox_normalization_policy_id": target["bbox_normalization_policy_id"],
        "target_bbox_coord_token_xyxy": target_bbox,
        "target_bbox_surface": target["bbox_surface"],
        "target_bbox_source_field": target["bbox_source_field"],
        "same_desc_gt_indices": list(same_desc_gt_indices),
        "competitor_gt_indices": competitor_gt_indices,
        "same_desc_competitor_bboxes": [
            list(obj["bbox_coord_token_xyxy"]) for obj in competitors
        ],
        "same_desc_count": len(same_desc_gt_indices),
        "object_count": len(objects),
        "primary_basin_label_source": PRIMARY_BASIN_LABEL_SOURCE,
        "x1_anchor_unique_under_r95": anchor["x1_anchor_unique_under_r95"],
        "anchor_ambiguity_bucket": anchor["anchor_ambiguity_bucket"],
        "primary_denominator_eligible": anchor["primary_denominator_eligible"],
        "source_jsonl_path": source_path,
        "source_jsonl_sha256": source_jsonl_sha256,
        "source_line_id": int(source_line_id),
        "objects": [_case_object(obj) for obj in objects],
    }
    return row


def _bbox_coord_token_xyxy_from_raw(
    raw: Mapping[str, Any],
) -> tuple[list[int], str, str] | None:
    bbox_2d = raw.get("bbox_2d")
    if bbox_2d is not None:
        bbox = _coord_token_bbox(bbox_2d)
        if bbox is not None:
            return bbox, "bbox_2d_coord_token_xyxy", "bbox_2d"
        return None

    for field in ("bbox", "bbox_xyxy"):
        if field in raw:
            bbox = _numeric_bbox(raw.get(field))
            if bbox is not None:
                return bbox, "numeric_xyxy_fixture", field
            return None

    points = raw.get("points")
    if points is not None:
        bbox = _points_bbox(points)
        if bbox is not None:
            return bbox, "points_pixel_corner_fixture", "points"
    return None


def _coord_token_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4:
        return None
    coords: list[int] = []
    for token in value:
        match = _COORD_TOKEN_RE.fullmatch(str(token))
        if match is None:
            return None
        coord = int(match.group(1))
        if coord < 0 or coord > 999:
            return None
        coords.append(coord)
    return coords


def _numeric_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4:
        return None
    try:
        return [int(round(float(coord))) for coord in value]
    except (TypeError, ValueError):
        return None


def _points_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 2:
        return None
    first, second = value
    if (
        not isinstance(first, Sequence)
        or isinstance(first, (str, bytes))
        or not isinstance(second, Sequence)
        or isinstance(second, (str, bytes))
        or len(first) != 2
        or len(second) != 2
    ):
        return None
    try:
        return [
            int(round(float(first[0]))),
            int(round(float(first[1]))),
            int(round(float(second[0]))),
            int(round(float(second[1]))),
        ]
    except (TypeError, ValueError):
        return None


def _valid_xyxy(bbox: Sequence[int]) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return 0 <= x1 <= 999 and 0 <= y1 <= 999 and x1 < x2 <= 999 and y1 < y2 <= 999


def _x1_anchor_bucket(
    *,
    target: Mapping[str, Any],
    competitors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_bbox = [int(value) for value in target["bbox_coord_token_xyxy"]]
    target_x1 = target_bbox[0]
    target_width = target_bbox[2] - target_bbox[0]
    target_radius = strict_r95_radius(target_width)
    has_exact = False
    has_near = False
    for competitor in competitors:
        competitor_bbox = [int(value) for value in competitor["bbox_coord_token_xyxy"]]
        diff = abs(competitor_bbox[0] - target_x1)
        competitor_width = competitor_bbox[2] - competitor_bbox[0]
        radius = max(target_radius, strict_r95_radius(competitor_width))
        if diff == 0:
            has_exact = True
        elif diff <= radius:
            has_near = True

    if has_exact:
        bucket = "exact_x1_collision"
    elif has_near:
        bucket = "near_collision"
    else:
        bucket = "unique"
    unique = bucket == "unique"
    return {
        "anchor_ambiguity_bucket": bucket,
        "x1_anchor_unique_under_r95": unique,
        "primary_denominator_eligible": unique,
    }


def _case_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gt_idx": int(obj["gt_idx"]),
        "desc": obj["desc"],
        "bbox_coord_token_xyxy": list(obj["bbox_coord_token_xyxy"]),
        "bbox_xyxy": list(obj["bbox_coord_token_xyxy"]),
        "bbox_surface": obj["bbox_surface"],
        "bbox_source_field": obj["bbox_source_field"],
        "source_bbox_surface": obj["bbox_surface"],
        "source_bbox_field": obj["bbox_source_field"],
        "bbox_normalization_policy_id": obj["bbox_normalization_policy_id"],
    }


def _image_id(*, sample: Mapping[str, Any], source_line_id: int) -> Any:
    return sample.get("image_id") or sample.get("id") or f"line-{source_line_id}"


def _image_path(sample: Mapping[str, Any]) -> str | None:
    image = sample.get("image_path") or sample.get("file_name") or sample.get("image")
    if image is not None:
        return str(image)
    images = sample.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
        return str(images[0])
    return None


def _case_id(split: str, image_id: Any, desc: str, target_gt_idx: int) -> str:
    safe_desc = re.sub(r"[^a-z0-9]+", "-", canonical_desc(desc)).strip("-")
    return f"a33-{split}-{image_id}-{safe_desc}-{target_gt_idx}"


__all__ = [
    "BBOX_NORMALIZATION_POLICY_ID",
    "PRIMARY_BASIN_LABEL_SOURCE",
    "build_case_universe",
    "build_case_universe_rows",
    "canonical_desc",
    "compute_anchor_r95",
    "normalize_sample_objects",
    "parse_coord_bbox",
    "sha256_path",
    "strict_r95_radius",
]
