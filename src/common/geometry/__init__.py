"""Centralized geometry type re-exports.

This keeps a single import path for geometry value objects and transforms
while preserving the existing implementations in ``src.datasets.geometry``.
"""

from __future__ import annotations

from src.datasets.geometry import (  # noqa: F401
    BBox,
    Polygon,
    Polyline,
    apply_affine,
    clamp_points,
    points_to_xyxy,
    transform_geometry,
    geometry_from_dict,
)
from .coord_utils import (  # noqa: F401
    COORD_TOKEN_RE,
    MAX_BIN,
    bbox_from_points,
    bbox_to_quadrilateral,
    clamp_and_round,
    coerce_point_list,
    decode_coord,
    denorm_and_clamp,
    encode_coord,
    flatten_points,
    has_coord_tokens,
    ints_to_pixels_norm1000,
    is_coord_token,
    is_degenerate_bbox,
    line_to_bbox,
    pair_points,
)

__all__ = [
    "BBox",
    "Polygon",
    "Polyline",
    "apply_affine",
    "clamp_points",
    "points_to_xyxy",
    "transform_geometry",
    "geometry_from_dict",
    # coord utils
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
