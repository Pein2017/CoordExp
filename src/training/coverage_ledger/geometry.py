"""Coverage-ledger geometry helpers aligned with CoordExp coord-token bins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.tokens.coord.codec import value_in_coord_range

MAX_BIN = 999


def validate_norm1000_bbox_xyxy(values: Sequence[Any]) -> tuple[int, int, int, int]:
    """Return a strict CoordExp coord-token bbox in the inclusive 0..999 range."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("bbox_norm1000_xyxy must be a sequence")
    bbox = tuple(values)
    if len(bbox) != 4:
        raise ValueError("bbox_norm1000_xyxy must contain exactly four values")
    frozen = tuple(
        _require_plain_int(value, field_name=f"bbox_norm1000_xyxy[{index}]")
        for index, value in enumerate(bbox)
    )
    x1, y1, x2, y2 = frozen
    if not (
        x1 < x2
        and y1 < y2
        and all(value_in_coord_range(value) for value in frozen)
    ):
        raise ValueError(
            "bbox_norm1000_xyxy must satisfy "
            f"0 <= x1 < x2 <= {MAX_BIN} and 0 <= y1 < y2 <= {MAX_BIN}"
        )
    return frozen


def norm1000_bbox_to_pixel_bbox(
    values: Sequence[Any],
    *,
    width: int,
    height: int,
) -> list[int]:
    """Convert a strict coord-token bbox to clamped pixel xyxy coordinates."""

    bbox = validate_norm1000_bbox_xyxy(values)
    from src.common.geometry.coord_utils import denorm_and_clamp

    return denorm_and_clamp(bbox, width, height, coord_mode="norm1000")


def _require_plain_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


__all__ = [
    "MAX_BIN",
    "norm1000_bbox_to_pixel_bbox",
    "validate_norm1000_bbox_xyxy",
]
