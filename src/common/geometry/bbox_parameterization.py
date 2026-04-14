from __future__ import annotations

import math
from typing import Literal, Sequence, cast

AllowedBBoxFormat = Literal["xyxy", "cxcy_logw_logh"]

DEFAULT_BBOX_FORMAT: AllowedBBoxFormat = "xyxy"
CXCY_LOGW_LOGH_SLOT_ORDER = "cxcy_logw_logh"
CXCY_LOGW_LOGH_CONVERSION_VERSION = 1
BBOX_SIZE_FLOOR: float = 1.0 / 1024.0
MAX_BIN: int = 999


def normalize_bbox_format(
    value: object | None,
    *,
    path: str = "bbox_format",
) -> AllowedBBoxFormat:
    if value is None:
        return DEFAULT_BBOX_FORMAT
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string when provided")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"xyxy", "cxcy_logw_logh"}:
        raise ValueError(
            f"{path} must be one of ['cxcy_logw_logh', 'xyxy'], got {value!r}"
        )
    return cast(AllowedBBoxFormat, normalized)


def quantize_norm1000_slot(value: float) -> int:
    z = min(max(float(value), 0.0), 1.0)
    return int(min(max(math.floor((MAX_BIN * z) + 0.5), 0), MAX_BIN))


def decode_norm1000_slot(value: float | int) -> float:
    return min(max(float(value) / float(MAX_BIN), 0.0), 1.0)


def _canonical_xyxy_norm(points: Sequence[float]) -> tuple[float, float, float, float]:
    if len(points) != 4:
        raise ValueError(f"bbox_2d must contain 4 values, got {len(points)}")
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in points)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def log_size_encode(size: float, *, size_floor: float = BBOX_SIZE_FLOOR) -> float:
    clamped = min(max(float(size), float(size_floor)), 1.0)
    return (math.log(clamped) - math.log(float(size_floor))) / (
        -math.log(float(size_floor))
    )


def log_size_decode(encoded: float, *, size_floor: float = BBOX_SIZE_FLOOR) -> float:
    u = min(max(float(encoded), 0.0), 1.0)
    return float(size_floor) * ((1.0 / float(size_floor)) ** u)


def xyxy_norm_to_cxcy_logw_logh_norm(points: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = _canonical_xyxy_norm(points)
    width = max(x2 - x1, BBOX_SIZE_FLOOR)
    height = max(y2 - y1, BBOX_SIZE_FLOOR)
    return [
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        log_size_encode(width),
        log_size_encode(height),
    ]


def cxcy_logw_logh_norm_to_xyxy_norm(points: Sequence[float]) -> list[float]:
    if len(points) != 4:
        raise ValueError(f"bbox_2d must contain 4 values, got {len(points)}")
    cx = min(max(float(points[0]), 0.0), 1.0)
    cy = min(max(float(points[1]), 0.0), 1.0)
    width = log_size_decode(points[2])
    height = log_size_decode(points[3])
    return [
        cx - (width / 2.0),
        cy - (height / 2.0),
        cx + (width / 2.0),
        cy + (height / 2.0),
    ]


def xyxy_norm1000_to_cxcy_logw_logh_bins(points: Sequence[float | int]) -> list[int]:
    norm = [decode_norm1000_slot(v) for v in points]
    return [quantize_norm1000_slot(v) for v in xyxy_norm_to_cxcy_logw_logh_norm(norm)]


def cxcy_logw_logh_norm1000_to_xyxy_norm1000(
    points: Sequence[float | int],
) -> list[float]:
    norm = [decode_norm1000_slot(v) for v in points]
    return [float(MAX_BIN) * v for v in cxcy_logw_logh_norm_to_xyxy_norm(norm)]


__all__ = [
    "AllowedBBoxFormat",
    "BBOX_SIZE_FLOOR",
    "CXCY_LOGW_LOGH_CONVERSION_VERSION",
    "CXCY_LOGW_LOGH_SLOT_ORDER",
    "DEFAULT_BBOX_FORMAT",
    "MAX_BIN",
    "cxcy_logw_logh_norm1000_to_xyxy_norm1000",
    "cxcy_logw_logh_norm_to_xyxy_norm",
    "decode_norm1000_slot",
    "log_size_decode",
    "log_size_encode",
    "normalize_bbox_format",
    "quantize_norm1000_slot",
    "xyxy_norm1000_to_cxcy_logw_logh_bins",
    "xyxy_norm_to_cxcy_logw_logh_norm",
]
