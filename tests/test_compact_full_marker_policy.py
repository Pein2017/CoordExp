from __future__ import annotations

import pytest

from src.detection.teacher_forcing.compact_full_policy import (
    BOX_START_TOKEN,
    END_OF_TEXT_TOKEN,
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


def test_axis_sort_repair_parser_canonicalizes_inverted_xyxy() -> None:
    result = parse_compact_full(
        f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
        "<|coord_30|><|coord_40|><|coord_10|><|coord_20|>",
        mode="marker_delimited_axis_sort_repair",
    )

    assert result.ok
    assert result.objects[0].bbox_2d == [
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    ]
    assert result.mode == "marker_delimited_axis_sort_repair"


def test_axis_sort_repair_parser_still_rejects_degenerate_bbox() -> None:
    result = parse_compact_full(
        f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_10|><|coord_40|>",
        mode="marker_delimited_axis_sort_repair",
    )

    assert result.error_code == "invalid_geometry"


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


@pytest.mark.parametrize(
    ("bad_tail", "error_code"),
    [
        ("<|coord_1000|>", "invalid_coord_token"),
        ("", "wrong_coord_arity"),
        (
            "<|coord_30|><|coord_20|><|coord_10|><|coord_40|>",
            "invalid_geometry",
        ),
    ],
)
def test_legacy_compatible_parser_reports_second_row_global_error_offsets(
    bad_tail: str,
    error_code: str,
) -> None:
    first_row = (
        f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )
    second_row_prefix = f"{OBJECT_REF_START_TOKEN}car{BOX_START_TOKEN}"
    second_row = second_row_prefix + bad_tail
    result = parse_compact_full(
        f"{first_row}\n{second_row}",
        mode="legacy_compatible",
    )

    assert result.error_code == error_code
    assert result.error_offset == len(first_row) + 1 + len(second_row_prefix)


def test_render_rejects_endoftext_in_description() -> None:
    payload = {
        "objects": [
            {
                "desc": f"bad {END_OF_TEXT_TOKEN} desc",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="forbidden compact_full marker"):
        render_compact_full(payload)


def test_parser_rejects_endoftext_in_description() -> None:
    result = parse_compact_full(
        f"{OBJECT_REF_START_TOKEN}bad {END_OF_TEXT_TOKEN} desc{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>",
        mode="marker_delimited_strict",
    )

    assert result.error_code == "forbidden_description_token"


@pytest.mark.parametrize(
    "bbox_tokens",
    [
        ["<|coord_30|>", "<|coord_20|>", "<|coord_10|>", "<|coord_40|>"],
        ["<|coord_10|>", "<|coord_20|>", "<|coord_10|>", "<|coord_40|>"],
        ["<|coord_10|>", "<|coord_40|>", "<|coord_30|>", "<|coord_40|>"],
    ],
)
def test_render_rejects_invalid_xyxy_geometry(bbox_tokens: list[str]) -> None:
    payload = {
        "objects": [
            {
                "desc": "person",
                "bbox_2d": bbox_tokens,
            }
        ]
    }

    with pytest.raises(ValueError, match="valid xyxy positive-area box"):
        render_compact_full(payload)
