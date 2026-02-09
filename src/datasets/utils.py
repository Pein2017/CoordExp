"""Shared utilities for dataset operations"""

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from src.common.io import load_jsonl  # re-export for backward compatibility


def extract_object_points(obj: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """Extract exactly one valid geometry field from an object.

    Accepts numeric points or coord tokens; for numeric inputs it normalizes values
    to float while preserving coord-token strings as-is.
    """

    def _coerce(value: Any, key: str) -> List[Any]:
        if value is None:
            raise ValueError(f"object is missing required geometry value for {key}")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{key} must be a sequence, got {type(value)!r}")
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return list(value)

    if "line" in obj or "line_points" in obj:
        raise ValueError("line geometry is not supported")

    has_bbox = obj.get("bbox_2d") is not None
    has_poly = obj.get("poly") is not None

    if has_bbox and has_poly:
        raise ValueError(
            "object must contain exactly one geometry field (bbox_2d xor poly), got both"
        )
    if not has_bbox and not has_poly:
        raise ValueError(
            "object must contain exactly one geometry field (bbox_2d xor poly), got none"
        )

    if has_bbox:
        points = _coerce(obj.get("bbox_2d"), "bbox_2d")
        if len(points) != 4:
            raise ValueError(
                f"bbox_2d must contain exactly 4 values; got len={len(points)}"
            )
        return "bbox_2d", points

    points = _coerce(obj.get("poly"), "poly")
    if any(isinstance(v, Sequence) and not isinstance(v, (str, bytes)) for v in points):
        raise ValueError("poly must be a flat coordinate sequence")
    if len(points) < 6 or (len(points) % 2 != 0):
        raise ValueError(
            "poly must contain an even number of values and at least 6 coordinates"
        )
    return "poly", points


_TOKEN_RE = re.compile(r"<\|coord_(\d+)\|>")


def _coord_value_to_float(value: Any) -> float | None:
    """Best-effort conversion of coord token or numeric to float."""
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        m = _TOKEN_RE.fullmatch(value.strip())
        if m:
            try:
                return float(int(m.group(1)))
            except Exception:
                return None
        try:
            return float(value)
        except Exception:
            return None
    return None


def sort_objects_by_topleft(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return objects sorted top-to-bottom then left-to-right using geometry."""

    def anchor(obj: Dict[str, Any]) -> Tuple[float, float]:
        geom_type, points = extract_object_points(obj)
        if not points or not geom_type:
            return (float("inf"), float("inf"))
        xs: List[float] = []
        ys: List[float] = []
        pts_iter = list(points)
        if geom_type == "bbox_2d" and len(pts_iter) >= 4:
            # bbox ordering assumed [x1, y1, x2, y2]
            x1 = _coord_value_to_float(pts_iter[0])
            y1 = _coord_value_to_float(pts_iter[1])
            if x1 is not None:
                xs.append(x1)
            if y1 is not None:
                ys.append(y1)
        else:
            for i, v in enumerate(pts_iter):
                val = _coord_value_to_float(v)
                if val is None:
                    continue
                if i % 2 == 0:
                    xs.append(val)
                else:
                    ys.append(val)
        if not xs or not ys:
            return (float("inf"), float("inf"))
        return (min(ys), min(xs))

    return sorted(list(objects), key=anchor)


def extract_geometry(obj: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Extract a validated single-geometry mapping from an object.

    Useful for augmentation and processing pipelines.

    Args:
        obj: Object dictionary

    Returns:
        Mapping with exactly one geometry key/value pair.
    """
    geom_type, points = extract_object_points(obj)
    return {str(geom_type): list(points)}


def is_same_record(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> bool:
    """Check if two records are the same (identity check).

    Args:
        record_a: First record
        record_b: Second record

    Returns:
        True if records are the same object
    """
    return record_a is record_b


__all__ = [
    "load_jsonl",
    "extract_object_points",
    "extract_geometry",
    "is_same_record",
    "sort_objects_by_topleft",
]
