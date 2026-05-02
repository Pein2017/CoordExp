from __future__ import annotations

import pytest

from src.common.detection_sequence import (
    BOX_START_TOKEN,
    OBJECT_REF_START_TOKEN,
    parse_compact_detection_sequence,
    render_compact_detection_sequence,
    required_special_tokens_for_detection_sequence_format,
)


def _payload() -> dict:
    return {
        "objects": [
            {
                "desc": "traffic light",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
            {
                "desc": "person",
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ],
            },
        ]
    }


def test_compact_full_renderer_uses_qwen_native_grounding_tokens() -> None:
    text = render_compact_detection_sequence(
        _payload(), detection_sequence_format="compact_full"
    )

    assert text == (
        f"{OBJECT_REF_START_TOKEN}traffic light{BOX_START_TOKEN}<|coord_10|><|coord_20|><|coord_30|><|coord_40|>\n"
        f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
    )


def test_compact_min_renderer_omits_structural_tokens() -> None:
    text = render_compact_detection_sequence(
        _payload(), detection_sequence_format="compact_min"
    )

    assert OBJECT_REF_START_TOKEN not in text
    assert BOX_START_TOKEN not in text
    assert text.splitlines()[0] == (
        "traffic light<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    )


@pytest.mark.parametrize(
    "fmt",
    ["compact_full", "compact_no_desc", "compact_no_bbox", "compact_min"],
)
def test_compact_parser_round_trips_all_variants(fmt: str) -> None:
    rendered = render_compact_detection_sequence(_payload(), detection_sequence_format=fmt)
    parsed = parse_compact_detection_sequence(rendered, detection_sequence_format=fmt)

    assert parsed == {
        "objects": [
            {
                "desc": "traffic light",
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ],
            },
            {
                "desc": "person",
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ],
            },
        ]
    }


def test_compact_parser_auto_detects_full_format_and_strips_im_end() -> None:
    raw = (
        f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        "<|im_end|>"
    )

    parsed = parse_compact_detection_sequence(raw)

    assert parsed == {
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


def test_compact_parser_rejects_wrong_coord_arity() -> None:
    raw = f"{OBJECT_REF_START_TOKEN}cat{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_3|>"

    assert parse_compact_detection_sequence(raw, detection_sequence_format="compact_full") is None


def test_required_special_tokens_by_variant() -> None:
    assert required_special_tokens_for_detection_sequence_format("coordjson") == ()
    assert required_special_tokens_for_detection_sequence_format("compact_full") == (
        OBJECT_REF_START_TOKEN,
        BOX_START_TOKEN,
    )
    assert required_special_tokens_for_detection_sequence_format("compact_no_desc") == (
        BOX_START_TOKEN,
    )
    assert required_special_tokens_for_detection_sequence_format("compact_no_bbox") == (
        OBJECT_REF_START_TOKEN,
    )
    assert required_special_tokens_for_detection_sequence_format("compact_min") == ()
