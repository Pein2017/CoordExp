"""Shared bbox-format helpers.

Canonical raw/eval artifacts remain ``xyxy``. This module provides explicit,
validated conversions for model-facing alternate parameterizations.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, cast

from .coord_utils import coerce_point_list

BBoxFormat = Literal["xyxy", "cxcywh"]
ALLOWED_BBOX_FORMATS: tuple[BBoxFormat, BBoxFormat] = ("xyxy", "cxcywh")


def normalize_bbox_format(
    value: Any,
    *,
    path: str = "bbox_format",
) -> BBoxFormat:
    normalized = str(value).strip().lower()
    if normalized not in ALLOWED_BBOX_FORMATS:
        raise ValueError(f"{path} must be one of {{'xyxy', 'cxcywh'}}; got {value!r}")
    return cast(BBoxFormat, normalized)


def convert_bbox_2d_points(
    points: Sequence[Any],
    *,
    src_format: BBoxFormat,
    dst_format: BBoxFormat,
    path: str = "bbox_2d",
) -> list[float]:
    src = normalize_bbox_format(src_format, path=f"{path}.src_format")
    dst = normalize_bbox_format(dst_format, path=f"{path}.dst_format")

    numeric_points, _had_tokens = coerce_point_list(points)
    if numeric_points is None or len(numeric_points) != 4:
        raise ValueError(f"{path} must contain exactly 4 numeric bbox values")

    if src == dst:
        return [float(v) for v in numeric_points]

    if src == "xyxy":
        x1_raw, y1_raw, x2_raw, y2_raw = (float(v) for v in numeric_points)
        x1 = min(x1_raw, x2_raw)
        y1 = min(y1_raw, y2_raw)
        x2 = max(x1_raw, x2_raw)
        y2 = max(y1_raw, y2_raw)
        return [
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            x2 - x1,
            y2 - y1,
        ]

    cx, cy, w, h = (float(v) for v in numeric_points)
    if w < 0 or h < 0:
        raise ValueError(f"{path} in cxcywh format must have non-negative w/h")
    half_w = w * 0.5
    half_h = h * 0.5
    return [
        cx - half_w,
        cy - half_h,
        cx + half_w,
        cy + half_h,
    ]

