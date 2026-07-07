"""Matrix-based coordinate-bin geometry transforms."""

from __future__ import annotations

from typing import Literal

from src.common.errors import DataContractError
from src.data.geometry import validate_bbox_bins

GeometryTransformId = Literal["identity", "hflip", "vflip", "hvflip"]

GEOMETRY_FLIP_POLICY_VERSION = "coordexp-swift-geometry-flips-v1"

COORD_AFFINE_MATRICES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "identity": (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ),
    "hflip": (
        (-1, 0, 999),
        (0, 1, 0),
        (0, 0, 1),
    ),
    "vflip": (
        (1, 0, 0),
        (0, -1, 999),
        (0, 0, 1),
    ),
    "hvflip": (
        (-1, 0, 999),
        (0, -1, 999),
        (0, 0, 1),
    ),
}


def transform_bbox(
    bbox: tuple[int, int, int, int],
    transform_id: str,
) -> tuple[int, int, int, int]:
    matrix = _matrix_for_transform(transform_id)
    x1, y1, x2, y2 = validate_bbox_bins(bbox, field="bbox")
    corners = (
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    )
    transformed = tuple(_apply_matrix(matrix, x, y) for x, y in corners)
    xs = [x for x, _ in transformed]
    ys = [y for _, y in transformed]
    return validate_bbox_bins(
        (min(xs), min(ys), max(xs), max(ys)),
        field=f"augmentation.{transform_id}.bbox",
    )


def _matrix_for_transform(transform_id: str) -> tuple[tuple[int, int, int], ...]:
    try:
        return COORD_AFFINE_MATRICES[transform_id]
    except KeyError as exc:
        raise DataContractError(
            "unsupported geometry transform id",
            code="augmentation.transform_id",
            context={"transform_id": transform_id},
            cause=exc,
        ) from exc


def _apply_matrix(
    matrix: tuple[tuple[int, int, int], ...],
    x: int,
    y: int,
) -> tuple[int, int]:
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
    )


__all__ = [
    "COORD_AFFINE_MATRICES",
    "GEOMETRY_FLIP_POLICY_VERSION",
    "GeometryTransformId",
    "transform_bbox",
]
