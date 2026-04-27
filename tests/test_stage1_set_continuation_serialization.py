from __future__ import annotations

from src.common.object_field_order import build_object_payload
from src.trainers.stage1_set_continuation.serialization import (
    build_candidate_entry_text,
    build_close_prefix_text,
    build_global_close_text,
    build_prefix_text,
    render_indexed_object_list,
)
from src.utils.assistant_json import dumps_coordjson


def _object_entry(
    *,
    desc: str,
    bbox_2d: list[str],
    object_field_order: str = "desc_first",
) -> str:
    return dumps_coordjson(
        build_object_payload(
            desc=desc,
            geometry_key="bbox_2d",
            geometry_value=bbox_2d,
            object_field_order=object_field_order,
        )
    )


def test_render_indexed_object_list_tracks_duplicate_entries_by_index() -> None:
    objects = [
        {
            "desc": "cat",
            "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
        },
        {
            "desc": "cat",
            "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
        },
        {
            "desc": "cat",
            "bbox_2d": ["<|coord_11|>", "<|coord_21|>", "<|coord_31|>", "<|coord_41|>"],
        },
    ]

    rendered = render_indexed_object_list(objects, object_field_order="desc_first")

    entry0 = _object_entry(
        desc="cat",
        bbox_2d=["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
    )
    entry2 = _object_entry(
        desc="cat",
        bbox_2d=["<|coord_11|>", "<|coord_21|>", "<|coord_31|>", "<|coord_41|>"],
    )
    assert rendered.text == (
        '{"objects": [' + entry0 + ", " + entry0 + ", " + entry2 + "]}"
    )
    assert '"<|coord_10|>"' not in rendered.text

    span0 = rendered.object_spans_by_index[0]
    span1 = rendered.object_spans_by_index[1]
    span2 = rendered.object_spans_by_index[2]

    assert rendered.text[span0.start : span0.end] == entry0
    assert rendered.text[span1.start : span1.end] == entry0
    assert rendered.text[span2.start : span2.end] == entry2
    assert span0.start != span1.start
    assert span0.object_index == 0
    assert span1.object_index == 1
    assert span2.object_index == 2

    close_start = rendered.global_close_start_span
    full_close = rendered.global_full_close_span
    assert rendered.text[close_start.start : close_start.end] == "]"
    assert rendered.text[full_close.start : full_close.end] == "]}"
    assert close_start.start == span2.end


def test_build_prefix_text_supports_empty_non_empty_and_full_prefixes() -> None:
    objects = [
        {
            "desc": "left",
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
        },
        {
            "desc": "right",
            "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"],
        },
    ]
    rendered = render_indexed_object_list(objects, object_field_order="geometry_first")

    empty_prefix = build_prefix_text(rendered, [])
    non_empty_prefix = build_prefix_text(rendered, [1])
    full_prefix = build_prefix_text(rendered, [0, 1])

    entry0 = _object_entry(
        desc="left",
        bbox_2d=["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
        object_field_order="geometry_first",
    )
    entry1 = _object_entry(
        desc="right",
        bbox_2d=["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"],
        object_field_order="geometry_first",
    )

    assert empty_prefix.text == '{"objects": ['
    assert non_empty_prefix.text == '{"objects": [' + entry1 + ", "
    assert full_prefix.text == '{"objects": [' + entry0 + ", " + entry1

    assert [span.object_index for span in empty_prefix.object_spans] == []
    assert [span.object_index for span in non_empty_prefix.object_spans] == [1]
    assert [span.object_index for span in full_prefix.object_spans] == [0, 1]
    assert (
        non_empty_prefix.text[
            non_empty_prefix.object_spans[0].start : non_empty_prefix.object_spans[
                0
            ].end
        ]
        == entry1
    )


def test_build_close_prefix_text_closes_partial_prefix_without_dangling_comma() -> None:
    objects = [
        {
            "desc": "left",
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
        },
        {
            "desc": "right",
            "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"],
        },
    ]
    rendered = render_indexed_object_list(objects, object_field_order="desc_first")
    close_prefix = build_close_prefix_text(rendered, [0])
    close_text = build_global_close_text()

    entry0 = _object_entry(
        desc="left",
        bbox_2d=["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
    )

    assert close_prefix.text == '{"objects": [' + entry0
    assert close_prefix.text + close_text.text == '{"objects": [' + entry0 + "]}"
    assert ", ]}" not in close_prefix.text + close_text.text
    assert [span.object_index for span in close_prefix.object_spans] == [0]


def test_build_candidate_entry_text_keeps_nonterminal_candidate_append_ready() -> None:
    objects = [
        {
            "desc": "first",
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
        },
        {
            "desc": "second",
            "bbox_2d": ["<|coord_11|>", "<|coord_12|>", "<|coord_13|>", "<|coord_14|>"],
        },
        {
            "desc": "third",
            "bbox_2d": ["<|coord_21|>", "<|coord_22|>", "<|coord_23|>", "<|coord_24|>"],
        },
    ]
    rendered = render_indexed_object_list(objects, object_field_order="desc_first")

    prefix = build_prefix_text(rendered, [0])
    continuation = build_candidate_entry_text(
        rendered,
        prefix_indices=[0],
        candidate_index=1,
    )
    candidate_entry = _object_entry(
        desc="second",
        bbox_2d=["<|coord_11|>", "<|coord_12|>", "<|coord_13|>", "<|coord_14|>"],
    )

    assert continuation.text == candidate_entry + ", "
    full_text = prefix.text + continuation.text
    offset = len(prefix.text)

    candidate_span = continuation.candidate_span

    assert candidate_span.object_index == 1
    assert (
        full_text[offset + candidate_span.start : offset + candidate_span.end]
        == candidate_entry
    )
    assert continuation.text[candidate_span.end :] == ", "
    assert not continuation.text.endswith("]}")


def test_build_candidate_entry_text_closes_terminal_candidate() -> None:
    objects = [
        {
            "desc": "first",
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
        },
        {
            "desc": "second",
            "bbox_2d": ["<|coord_11|>", "<|coord_12|>", "<|coord_13|>", "<|coord_14|>"],
        },
    ]
    rendered = render_indexed_object_list(objects, object_field_order="desc_first")

    continuation = build_candidate_entry_text(
        rendered,
        prefix_indices=[0],
        candidate_index=1,
    )
    candidate_entry = _object_entry(
        desc="second",
        bbox_2d=["<|coord_11|>", "<|coord_12|>", "<|coord_13|>", "<|coord_14|>"],
    )

    assert continuation.text == candidate_entry + "]}"
    close_start = continuation.global_close_start_span
    full_close = continuation.global_full_close_span
    assert continuation.text[close_start.start : close_start.end] == "]"
    assert continuation.text[full_close.start : full_close.end] == "]}"
    assert continuation.candidate_span.end == close_start.start


def test_build_global_close_text_excludes_object_and_chat_end_tokens() -> None:
    close = build_global_close_text()

    assert close.text == "]}"
    assert (
        close.text[
            close.global_close_start_span.start : close.global_close_start_span.end
        ]
        == "]"
    )
    assert (
        close.text[
            close.global_full_close_span.start : close.global_full_close_span.end
        ]
        == "]}"
    )
    assert (
        "}"
        not in close.text[
            close.global_close_start_span.start : close.global_close_start_span.end
        ]
    )
    assert "<|im_end|>" not in close.text
    assert "<|end_of_text|>" not in close.text
