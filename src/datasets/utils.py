"""Shared utilities for dataset operations"""

from typing import Any, Dict, List, Sequence, Tuple

from src.common.geometry.coord_utils import coerce_point_list
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

    from src.common.geometry.object_geometry import extract_single_geometry

    geom_type, raw_points = extract_single_geometry(
        obj,
        allow_type_and_points=False,
        allow_nested_points=False,
        path="object",
    )
    points = _coerce(raw_points, geom_type)
    return str(geom_type), points



def _points_to_floats(points: Sequence[Any]) -> List[float] | None:
    numeric, _had_tokens = coerce_point_list(points)
    return numeric


def sort_objects_by_topleft(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return objects sorted top-to-bottom then left-to-right using geometry."""

    def anchor(obj: Dict[str, Any]) -> Tuple[float, float]:
        geom_type, points = extract_object_points(obj)
        if not points or not geom_type:
            return (float("inf"), float("inf"))

        numeric = _points_to_floats(points)
        if not numeric:
            return (float("inf"), float("inf"))

        if geom_type == "bbox_2d":
            # bbox ordering assumed [x1, y1, x2, y2]
            if len(numeric) < 2:
                return (float("inf"), float("inf"))
            return (numeric[1], numeric[0])

        xs = numeric[0::2]
        ys = numeric[1::2]
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
