import pytest

from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.utils import (
    extract_geometry,
    extract_object_points,
    find_first_unsorted_object_pair_by_topleft,
    sort_objects_by_topleft,
)


def _builder() -> JSONLinesBuilder:
    return JSONLinesBuilder(
        user_prompt="Locate objects",
        emit_norm="none",
        mode="dense",
        coord_tokens_enabled=False,
    )


def _record_with_objects(objects):
    return {
        "images": ["dummy.jpg"],
        "objects": list(objects),
        "width": 100,
        "height": 100,
    }


def test_extract_object_points_rejects_missing_geometry():
    with pytest.raises(ValueError, match="exactly one geometry field"):
        extract_object_points({"desc": "cat"})


def test_extract_object_points_rejects_multi_geometry():
    with pytest.raises(ValueError, match="exactly one geometry field"):
        extract_object_points(
            {
                "desc": "cat",
                "bbox_2d": [1, 2, 3, 4],
                "poly": [1, 2, 3, 4, 5, 6],
            }
        )


def test_extract_object_points_rejects_invalid_bbox_arity():
    with pytest.raises(ValueError, match="exactly 4"):
        extract_object_points({"desc": "cat", "bbox_2d": [1, 2, 3]})


def test_extract_object_points_rejects_invalid_poly_arity():
    with pytest.raises(ValueError, match="even number"):
        extract_object_points({"desc": "cat", "poly": [1, 2, 3, 4, 5]})


def test_extract_geometry_rejects_missing_geometry():
    with pytest.raises(ValueError, match="exactly one geometry field"):
        extract_geometry({"desc": "cat"})


def test_jsonlines_builder_rejects_geometry_less_object():
    builder = _builder()
    record = _record_with_objects([{"desc": "cat"}])
    with pytest.raises(ValueError, match="exactly one geometry field"):
        builder.build_many([record])


def test_jsonlines_builder_rejects_multi_geometry_object():
    builder = _builder()
    record = _record_with_objects(
        [
            {
                "desc": "cat",
                "bbox_2d": [1, 2, 3, 4],
                "poly": [1, 2, 3, 4, 5, 6],
            }
        ]
    )
    with pytest.raises(ValueError, match="exactly one geometry field"):
        builder.build_many([record])


def test_jsonlines_builder_rejects_invalid_bbox_arity():
    builder = _builder()
    record = _record_with_objects([{"desc": "cat", "bbox_2d": [1, 2, 3]}])
    with pytest.raises(ValueError, match="exactly 4"):
        builder.build_many([record])


def test_jsonlines_builder_rejects_empty_desc():
    builder = _builder()
    record = _record_with_objects([{"desc": "   ", "bbox_2d": [1, 2, 3, 4]}])
    with pytest.raises(ValueError, match="empty 'desc'"):
        builder.build_many([record])


def test_sort_objects_by_topleft_supports_coord_tokens() -> None:
    objects = [
        {
            "desc": "b",
            "bbox_2d": [
                "<|coord_10|>",
                "<|coord_20|>",
                "<|coord_30|>",
                "<|coord_40|>",
            ],
        },
        {
            "desc": "a",
            "bbox_2d": [
                "<|coord_0|>",
                "<|coord_0|>",
                "<|coord_1|>",
                "<|coord_1|>",
            ],
        },
    ]

    sorted_objects = sort_objects_by_topleft(objects)
    assert [o["desc"] for o in sorted_objects] == ["a", "b"]


def test_find_first_unsorted_object_pair_by_topleft_detects_unsorted() -> None:
    objects = [
        {
            "desc": "b",
            "bbox_2d": [
                "<|coord_10|>",
                "<|coord_20|>",
                "<|coord_30|>",
                "<|coord_40|>",
            ],
        },
        {
            "desc": "a",
            "bbox_2d": [
                "<|coord_0|>",
                "<|coord_0|>",
                "<|coord_1|>",
                "<|coord_1|>",
            ],
        },
    ]

    pair = find_first_unsorted_object_pair_by_topleft(objects)
    assert pair == (0, 1, (20.0, 10.0), (0.0, 0.0))


def test_find_first_unsorted_object_pair_by_topleft_accepts_sorted() -> None:
    objects = [
        {"desc": "a", "bbox_2d": [0, 0, 1, 1]},
        {"desc": "b", "bbox_2d": [10, 20, 30, 40]},
    ]

    assert find_first_unsorted_object_pair_by_topleft(objects) is None
