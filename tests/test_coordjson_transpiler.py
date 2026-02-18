import json

import pytest

from src.utils.assistant_json import dumps_canonical_json, dumps_coordjson
from src.utils.coordjson_transpiler import (
    CoordJSONValidationError,
    coordjson_to_strict_json,
    coordjson_to_strict_json_with_meta,
)


def test_coordjson_strict_bbox_desc_first_to_strict_json() -> None:
    text = (
        '{"objects": [{"desc": "cat", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )
    strict = coordjson_to_strict_json(
        text,
        mode="strict",
        object_field_order="desc_first",
    )
    assert strict == '{"objects": [{"desc": "cat", "bbox_2d": [1, 2, 3, 4]}]}'


def test_coordjson_strict_rejects_order_violation() -> None:
    text = (
        '{"objects": [{"desc": "cat", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )
    with pytest.raises(CoordJSONValidationError, match="order_violation"):
        coordjson_to_strict_json(
            text,
            mode="strict",
            object_field_order="geometry_first",
        )


def test_coordjson_strict_rejects_extra_record_keys() -> None:
    text = (
        '{"objects": [{"desc": "cat", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>], '
        '"score": 0.9}]}'
    )
    with pytest.raises(CoordJSONValidationError, match="unexpected_keys"):
        coordjson_to_strict_json(
            text,
            mode="strict",
            object_field_order="desc_first",
        )


def test_coordjson_salvage_drops_invalid_and_incomplete_tail() -> None:
    text = (
        '{"objects": ['
        '{"desc": "ok", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
        '{"desc": "bad", "bbox_2d": [<|coord_5|>, <|coord_6|>, <|coord_7|>]}, '
        '{"desc": "tail", "bbox_2d": [<|coord_8|>'
    )
    strict, meta = coordjson_to_strict_json_with_meta(
        text,
        mode="salvage",
        object_field_order="desc_first",
    )
    assert strict == '{"objects": [{"desc": "ok", "bbox_2d": [1, 2, 3, 4]}]}'
    assert meta.parse_failed is False
    assert meta.truncated is True
    assert meta.dropped_invalid_records == 1
    assert meta.dropped_incomplete_tail == 1
    assert meta.dropped_invalid_by_reason.get("wrong_arity", 0) == 1


def test_coordjson_salvage_first_container_only_with_junk() -> None:
    text = (
        'Answer: {"objects": [{"desc": "first", "bbox_2d": '
        '[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]} '
        '{"objects": [{"desc": "second", "bbox_2d": '
        '[<|coord_5|>, <|coord_6|>, <|coord_7|>, <|coord_8|>]}]}'
    )
    strict = coordjson_to_strict_json(
        text,
        mode="salvage",
        object_field_order="desc_first",
    )
    parsed = json.loads(strict)
    assert parsed["objects"][0]["desc"] == "first"
    assert len(parsed["objects"]) == 1


def test_coordjson_salvage_string_aware_container_scan() -> None:
    text = (
        '{"objects": [{"desc": "cat with ]} in text", '
        '"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )
    strict = coordjson_to_strict_json(
        text,
        mode="salvage",
        object_field_order="desc_first",
    )
    assert strict == (
        '{"objects": [{"desc": "cat with ]} in text", "bbox_2d": [1, 2, 3, 4]}]}'
    )


def test_coordjson_salvage_parse_fail_returns_empty_objects() -> None:
    strict, meta = coordjson_to_strict_json_with_meta(
        "<|im_end|> no json container",
        mode="salvage",
        object_field_order="desc_first",
    )
    assert strict == '{"objects": []}'
    assert meta.parse_failed is True


def test_dumps_coordjson_golden_strings_for_orders_and_poly() -> None:
    desc_first_payload = {
        "objects": [
            {
                "desc": "cat",
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ],
            }
        ]
    }
    geometry_first_payload = {
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ],
                "desc": "cat",
            }
        ]
    }
    poly_geometry_first_payload = {
        "objects": [
            {
                "poly": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                    "<|coord_5|>",
                    "<|coord_6|>",
                ],
                "desc": "poly-cat",
            }
        ]
    }

    assert dumps_coordjson(desc_first_payload) == (
        '{"objects": [{"desc": "cat", "bbox_2d": '
        '[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )
    assert dumps_coordjson(geometry_first_payload) == (
        '{"objects": [{"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>], '
        '"desc": "cat"}]}'
    )
    assert dumps_coordjson(poly_geometry_first_payload) == (
        '{"objects": [{"poly": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>, '
        '<|coord_5|>, <|coord_6|>], "desc": "poly-cat"}]}'
    )


def test_dumps_canonical_json_uses_single_line_spacing() -> None:
    strict = dumps_canonical_json({"objects": [{"desc": "cat", "bbox_2d": [1, 2, 3, 4]}]})
    assert strict == '{"objects": [{"desc": "cat", "bbox_2d": [1, 2, 3, 4]}]}'
    assert "\n" not in strict
