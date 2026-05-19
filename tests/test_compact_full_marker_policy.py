from __future__ import annotations

import pytest

from src.detection.teacher_forcing.compact_full_policy import (
    BOX_START_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_START_TOKEN,
    parse_compact_full,
    render_compact_full,
)


sample_with_two_objects = {
    "objects": [
        {
            "desc": "person",
            "bbox_2d": [
                "<|coord_10|>",
                "<|coord_20|>",
                "<|coord_30|>",
                "<|coord_40|>",
            ],
        },
        {
            "desc": "car",
            "bbox_2d": [
                "<|coord_100|>",
                "<|coord_200|>",
                "<|coord_300|>",
                "<|coord_400|>",
            ],
        },
    ]
}

text_with_newline_rows = (
    f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>\n"
    f"{OBJECT_REF_START_TOKEN}car{BOX_START_TOKEN}"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
)

two_marker_objects_then_im_end = (
    f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    f"{OBJECT_REF_START_TOKEN}car{BOX_START_TOKEN}"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
    f"{IM_END_TOKEN}"
)


def test_marker_delimited_render_has_no_newline_between_objects() -> None:
    rendered = render_compact_full(
        sample_with_two_objects,
        serialization_policy="marker_delimited",
    )
    assert "\n" not in rendered
    assert rendered.count("<|object_ref_start|>") == 2


def test_legacy_newline_render_policy_is_explicit() -> None:
    rendered = render_compact_full(
        sample_with_two_objects,
        serialization_policy="legacy_newline_delimited",
    )
    assert "\n" in rendered
    assert rendered.count("<|object_ref_start|>") == 2


def test_strict_marker_parser_rejects_legacy_newline() -> None:
    result = parse_compact_full(text_with_newline_rows, mode="marker_delimited_strict")
    assert result.error_code == "legacy_separator_in_new_format"


def test_legacy_compatible_parser_accepts_newline_for_historical_outputs() -> None:
    result = parse_compact_full(text_with_newline_rows, mode="legacy_compatible")
    assert result.objects
    assert result.mode == "legacy_compatible"


def test_marker_parser_allows_next_object_or_im_end_after_four_coords() -> None:
    result = parse_compact_full(
        two_marker_objects_then_im_end,
        mode="marker_delimited_strict",
    )
    assert result.objects[0].description == "person"
    assert result.objects[1].description == "car"
    assert result.terminal_token == "<|im_end|>"


@pytest.mark.parametrize(
    ("text", "error_code"),
    [
        ("", "empty_output"),
        (
            "person<|box_start|><|coord_10|><|coord_20|><|coord_30|><|coord_40|>",
            "missing_object_ref_start",
        ),
        (
            f"{OBJECT_REF_START_TOKEN}person"
            "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>",
            "missing_box_start",
        ),
        (
            f"{OBJECT_REF_START_TOKEN}{BOX_START_TOKEN}"
            "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>",
            "empty_description",
        ),
        (
            f"{OBJECT_REF_START_TOKEN}bad<|im_start|>{BOX_START_TOKEN}"
            "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>",
            "forbidden_description_token",
        ),
        (
            f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
            "<|coord_10|><|coord_20|><|coord_30|>",
            "wrong_coord_arity",
        ),
        (
            f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
            "<|coord_10|><|coord_20|><|coord_30|><|coord_1000|>",
            "invalid_coord_token",
        ),
        (
            f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
            "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>junk",
            "trailing_garbage",
        ),
        (
            f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
            "<|coord_30|><|coord_20|><|coord_10|><|coord_40|>",
            "invalid_geometry",
        ),
    ],
)
def test_strict_marker_parser_returns_stable_error_codes(
    text: str,
    error_code: str,
) -> None:
    result = parse_compact_full(text, mode="marker_delimited_strict")
    assert result.error_code == error_code
