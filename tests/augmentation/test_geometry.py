from __future__ import annotations

import pytest

from src.common.errors import DataContractError
from src.augmentation.geometry import (
    COORD_AFFINE_MATRICES,
    GEOMETRY_FLIP_POLICY_VERSION,
    transform_bbox,
)


def test_affine_matrices_are_explicit_over_coord_bin_plane() -> None:
    assert GEOMETRY_FLIP_POLICY_VERSION == "coordexp-swift-geometry-flips-v1"
    assert COORD_AFFINE_MATRICES["identity"] == (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )
    assert COORD_AFFINE_MATRICES["hflip"] == (
        (-1, 0, 999),
        (0, 1, 0),
        (0, 0, 1),
    )
    assert COORD_AFFINE_MATRICES["vflip"] == (
        (1, 0, 0),
        (0, -1, 999),
        (0, 0, 1),
    )
    assert COORD_AFFINE_MATRICES["hvflip"] == (
        (-1, 0, 999),
        (0, -1, 999),
        (0, 0, 1),
    )


@pytest.mark.parametrize(
    ("transform_id", "expected_bbox"),
    [
        ("identity", (100, 200, 300, 400)),
        ("hflip", (699, 200, 899, 400)),
        ("vflip", (100, 599, 300, 799)),
        ("hvflip", (699, 599, 899, 799)),
    ],
)
def test_transform_bbox_uses_corner_transform_and_canonicalizes_xyxy(
    transform_id: str,
    expected_bbox: tuple[int, int, int, int],
) -> None:
    assert transform_bbox((100, 200, 300, 400), transform_id) == expected_bbox


@pytest.mark.parametrize(
    ("transform_id", "expected_bbox"),
    [
        ("hflip", (0, 0, 999, 999)),
        ("vflip", (0, 0, 999, 999)),
        ("hvflip", (0, 0, 999, 999)),
    ],
)
def test_transform_bbox_preserves_valid_coord_boundaries(
    transform_id: str,
    expected_bbox: tuple[int, int, int, int],
) -> None:
    assert transform_bbox((0, 0, 999, 999), transform_id) == expected_bbox


@pytest.mark.parametrize(
    "bbox",
    [
        (1, 1, 1, 4),
        (-1, 1, 3, 4),
        (1, 1, 1000, 4),
    ],
)
def test_transform_bbox_rejects_invalid_source_or_transformed_boxes(
    bbox: tuple[int, int, int, int],
) -> None:
    with pytest.raises(DataContractError):
        transform_bbox(bbox, "hflip")


def test_transform_bbox_rejects_unknown_transform_id() -> None:
    with pytest.raises(DataContractError) as exc_info:
        transform_bbox((100, 200, 300, 400), "diagonal")

    assert exc_info.value.code == "augmentation.transform_id"
