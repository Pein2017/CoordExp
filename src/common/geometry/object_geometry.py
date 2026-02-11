"""Shared object-geometry extraction helpers.

This module provides a single, order-preserving implementation for extracting
exactly one geometry field from an object and validating basic shape invariants.

It is used across datasets, coord-token annotation, and inference/eval
standardization to avoid parallel implementations drifting.

Invariants:
- Exactly one of `bbox_2d` xor `poly` is present (unless type+points mode is used).
- Points are non-destructively validated (no reorder/drop of valid points).
- bbox_2d arity is exactly 4.
- poly arity is even and >= 6.
"""

from __future__ import annotations

from typing import Any, List, Literal, Mapping, Sequence, Tuple

from .coord_utils import flatten_points

GeomType = Literal["bbox_2d", "poly"]
GEOMETRY_KEYS: Tuple[GeomType, ...] = ("bbox_2d", "poly")


def extract_single_geometry(
    obj: Mapping[str, Any],
    *,
    allow_type_and_points: bool = False,
    allow_nested_points: bool = False,
    path: str = "object",
) -> Tuple[GeomType, List[Any]]:
    """Extract (geom_type, flat_points) from an object.

    Args:
        obj: object mapping.
        allow_type_and_points: if True, allow the prediction-style schema
            `{type: bbox_2d|poly, points: [...]}` when bbox_2d/poly keys are absent.
        allow_nested_points: if True, allow a list-of-pairs representation and
            flatten it into a flat sequence.
        path: string prefix for error messages.

    Returns:
        (geom_type, points_raw_flat)

    Raises:
        ValueError on missing/ambiguous/invalid geometry fields.
    """

    keys = [k for k in GEOMETRY_KEYS if k in obj and obj[k] is not None]
    if len(keys) > 1:
        raise ValueError(
            f"{path} must contain exactly one geometry field (bbox_2d xor poly), got both"
        )

    geom_type: GeomType | None = None
    raw: Any = None

    if len(keys) == 1:
        geom_type = keys[0]
        raw = obj.get(geom_type)
    elif allow_type_and_points:
        t = obj.get("type")
        if t is not None and t not in GEOMETRY_KEYS:
            raise ValueError(f"{path} type must be bbox_2d|poly, got {t!r}")
        if t in GEOMETRY_KEYS and obj.get("points") is not None:
            geom_type = t  # type: ignore[assignment]
            raw = obj.get("points")

    if geom_type is None:
        raise ValueError(
            f"{path} must contain exactly one geometry field (bbox_2d xor poly), got none"
        )

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{path}.{geom_type} must be a sequence")

    if not allow_nested_points:
        for v in raw:
            if isinstance(v, (list, tuple)):
                raise ValueError(f"{path}.{geom_type} must be a flat coordinate sequence")

    pts = flatten_points(raw)
    if pts is None:
        raise ValueError(f"{path}.{geom_type} has invalid point container shape")
    if len(pts) % 2 != 0:
        raise ValueError(f"{path}.{geom_type} must contain an even number of values")

    if geom_type == "bbox_2d" and len(pts) != 4:
        raise ValueError(
            f"{path}.bbox_2d must contain exactly 4 values; got len={len(pts)}"
        )

    if geom_type == "poly":
        if len(pts) < 6 or (len(pts) % 2 != 0):
            raise ValueError(
                f"{path}.poly must contain an even number of values and at least 6 coordinates; got len={len(pts)}"
            )

    return str(geom_type), list(pts)


__all__ = [
    "GeomType",
    "GEOMETRY_KEYS",
    "extract_single_geometry",
]
