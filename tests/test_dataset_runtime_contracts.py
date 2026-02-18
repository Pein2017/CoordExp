import pytest

from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.utils import extract_geometry, extract_object_points, sort_objects_by_topleft


def _builder(object_field_order: str = "desc_first") -> JSONLinesBuilder:
    return JSONLinesBuilder(
        user_prompt="Locate objects",
        emit_norm="none",
        mode="dense",
        coord_tokens_enabled=False,
        object_field_order=object_field_order,
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


def test_jsonlines_builder_object_field_order_switches_bbox_per_object_key_order():
    record = _record_with_objects([{"desc": "cat", "bbox_2d": [1, 2, 3, 4]}])

    desc_first = _builder("desc_first").build_many([record])
    geometry_first = _builder("geometry_first").build_many([record])

    desc_first_obj = desc_first["assistant_payload"]["object_1"]
    geometry_first_obj = geometry_first["assistant_payload"]["object_1"]

    assert list(desc_first_obj.keys()) == ["desc", "bbox_2d"]
    assert list(geometry_first_obj.keys()) == ["bbox_2d", "desc"]

    desc_first_text = desc_first["messages"][1]["content"][0]["text"]
    geometry_first_text = geometry_first["messages"][1]["content"][0]["text"]
    assert desc_first_text.index('"desc"') < desc_first_text.index('"bbox_2d"')
    assert geometry_first_text.index('"bbox_2d"') < geometry_first_text.index('"desc"')


def test_jsonlines_builder_poly_output_omits_poly_points_metadata():
    record = _record_with_objects(
        [
            {
                "desc": "region",
                "poly": [10, 20, 30, 20, 30, 40, 10, 40],
                "poly_points": [[10, 20], [30, 20], [30, 40], [10, 40]],
            }
        ]
    )
    built = _builder("geometry_first").build_many([record])
    obj = built["assistant_payload"]["object_1"]
    assistant_text = built["messages"][1]["content"][0]["text"]

    assert list(obj.keys()) == ["poly", "desc"]
    assert "poly_points" not in obj
    assert "poly_points" not in assistant_text
