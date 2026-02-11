from __future__ import annotations

import pytest

from src.common.geometry.object_geometry import extract_single_geometry


def test_extract_single_geometry_bbox_2d_flat() -> None:
    gtype, pts = extract_single_geometry({"bbox_2d": [1, 2, 3, 4]})
    assert gtype == "bbox_2d"
    assert pts == [1, 2, 3, 4]


def test_extract_single_geometry_poly_nested_requires_flag() -> None:
    obj = {"poly": [[1, 2], [3, 4], [5, 6]]}

    with pytest.raises(ValueError, match="flat coordinate sequence"):
        extract_single_geometry(obj, allow_nested_points=False)

    gtype, pts = extract_single_geometry(obj, allow_nested_points=True)
    assert gtype == "poly"
    assert pts == [1, 2, 3, 4, 5, 6]


def test_extract_single_geometry_rejects_both_keys() -> None:
    obj = {"bbox_2d": [1, 2, 3, 4], "poly": [1, 2, 3, 4, 5, 6]}
    with pytest.raises(ValueError, match="got both"):
        extract_single_geometry(obj)


def test_extract_single_geometry_rejects_missing_geometry() -> None:
    with pytest.raises(ValueError, match="got none"):
        extract_single_geometry({"desc": "x"})


def test_extract_single_geometry_rejects_odd_arity() -> None:
    with pytest.raises(ValueError, match="even number"):
        extract_single_geometry({"poly": [1, 2, 3, 4, 5]})


def test_extract_single_geometry_rejects_bbox_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        extract_single_geometry({"bbox_2d": [1, 2, 3, 4, 5, 6]})


def test_extract_single_geometry_allows_type_and_points_schema() -> None:
    gtype, pts = extract_single_geometry(
        {"type": "bbox_2d", "points": [1, 2, 3, 4]},
        allow_type_and_points=True,
    )
    assert gtype == "bbox_2d"
    assert pts == [1, 2, 3, 4]


def test_extract_single_geometry_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match=r"bbox_2d\|poly"):
        extract_single_geometry(
            {"type": "circle", "points": [1, 2]},
            allow_type_and_points=True,
        )
