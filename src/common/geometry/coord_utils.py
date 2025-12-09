"""Shared coordinate token + scaling utilities for CoordExp.

These helpers are the single source of truth for:
- coord token encode/decode (0-999 bin grid)
- flattening point containers
- norm1000 <-> pixel conversion with clamping/rounding
- basic geometry helpers (bbox from points, degenerate checks, line->bbox, bbox->segm)

They are re-exported via ``src.common.geometry`` so inference, visualization, and
evaluation can share the same logic and avoid double-scaling bugs.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Tuple

from src.coord_tokens.codec import (
    int_to_token,
    sequence_has_coord_tokens,
    token_to_int,
    value_in_coord_range,
)
from src.datasets.geometry import clamp_points as _clamp_points
from src.datasets.geometry import points_to_xyxy

COORD_TOKEN_RE = re.compile(r"<\|coord_(\d{1,4})\|>")
MAX_BIN = 999  # coord tokens are 0..999 inclusive


# ---------------- Token helpers ---------------- #


def is_coord_token(value: Any) -> bool:
    return isinstance(value, str) and COORD_TOKEN_RE.fullmatch(value) is not None


def encode_coord(value: int) -> str:
    if not value_in_coord_range(value):
        raise ValueError(f"coord value out of range 0-999: {value}")
    return int_to_token(value)


def decode_coord(value: Any) -> int | None:
    """Decode a coord token or numeric value into an int in [0, 999]."""

    if is_coord_token(value):
        v = token_to_int(value)
        return v if value_in_coord_range(v) else None
    try:
        v = int(round(float(value)))
    except Exception:
        return None
    return v if 0 <= v <= MAX_BIN else None


# ---------------- Point helpers ---------------- #


def flatten_points(value: Any) -> List[Any] | None:
    """Flatten nested [ [x,y], ... ] or already-flat lists.

    Returns None on shape/type mismatch.
    """

    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        flat: List[Any] = []
        for pair in value:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                return None
            flat.extend(pair)
        return flat
    return list(value)


def has_coord_tokens(seq: Sequence[Any]) -> bool:
    return sequence_has_coord_tokens(seq)


def ints_to_pixels_norm1000(ints: Sequence[int], width: float, height: float) -> List[float]:
    """Convert normalized 0..999 coords into pixel floats."""

    out: List[float] = []
    denom_x = max(1.0, float(width) - 1.0)
    denom_y = max(1.0, float(height) - 1.0)
    for i, v in enumerate(ints):
        frac = float(v) / float(MAX_BIN)
        if i % 2 == 0:
            out.append(frac * denom_x)
        else:
            out.append(frac * denom_y)
    return out


def clamp_and_round(points: Sequence[float], width: float, height: float) -> List[int]:
    """Clamp to [0, W-1]/[0, H-1] and round to nearest int."""

    return _clamp_points(points, width, height)


def denorm_and_clamp(
    points: Sequence[float] | Sequence[int],
    width: float,
    height: float,
    *,
    coord_mode: str = "norm1000",
) -> List[int]:
    """Convert coords to pixel ints according to coord_mode and clamp.

    coord_mode: "norm1000" (default) or "pixel".
    """

    if coord_mode == "pixel":
        pts_px = [float(v) for v in points]
    else:
        ints = [int(round(float(p))) for p in points]
        pts_px = ints_to_pixels_norm1000(ints, width, height)
    return clamp_and_round(pts_px, width, height)


def bbox_from_points(points: Sequence[float]) -> Tuple[float, float, float, float]:
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def is_degenerate_bbox(x1: float, y1: float, x2: float, y2: float) -> bool:
    return (x2 - x1) <= 0 or (y2 - y1) <= 0


def line_to_bbox(points: Sequence[float]) -> List[float]:
    return list(points_to_xyxy(points))


def bbox_to_quadrilateral(bbox: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def pair_points(points: Sequence[float]) -> List[Tuple[float, float]]:
    assert len(points) % 2 == 0
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]


def coerce_point_list(pts_raw: Iterable[Any]) -> Tuple[List[float] | None, bool]:
    """Coerce raw points into float list; returns (points, had_tokens)."""

    pts = list(pts_raw)
    if len(pts) % 2 != 0:
        return None, False
    had_tokens = has_coord_tokens(pts)
    numeric: List[float] = []
    for p in pts:
        coord = decode_coord(p)
        if coord is not None:
            numeric.append(float(coord))
            continue
        try:
            numeric.append(float(p))
        except Exception:
            return None, had_tokens
    return numeric, had_tokens


__all__ = [
    "COORD_TOKEN_RE",
    "MAX_BIN",
    "is_coord_token",
    "encode_coord",
    "decode_coord",
    "flatten_points",
    "has_coord_tokens",
    "ints_to_pixels_norm1000",
    "clamp_and_round",
    "denorm_and_clamp",
    "bbox_from_points",
    "is_degenerate_bbox",
    "line_to_bbox",
    "bbox_to_quadrilateral",
    "pair_points",
    "coerce_point_list",
]
