from __future__ import annotations

import re

import pytest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.template import get_detection_template
from src.detection.tokenization import tokenize_rendered_detection_conversation
from src.training.encoding.view import EncodedDetectionView
from src.training.span_adapters.residual_boundary import ResidualBoundaryAdapter


class SnapshotTokenizer:
    """Deterministic tokenizer stub that preserves compact-full token offsets."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        token_pattern = re.compile(
            r"<\|object_ref_start\|>|<\|box_start\|>|<\|coord_\d+\|>|"
            r"<\|im_end\|>|"
            r"[A-Za-z0-9_]+|"
            r"\s|"
            r".",
            re.DOTALL,
        )
        matches = list(token_pattern.finditer(text))
        token_texts = [match.group(0) for match in matches]
        vocab = {token: index + 1 for index, token in enumerate(dict.fromkeys(token_texts))}
        return {
            "input_ids": [vocab[token] for token in token_texts],
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


def _objects() -> list[dict[str, object]]:
    return [
        {
            "desc": "traffic light",
            "points_norm1000": [0, 1, 20, 30],
            "object_id": "ann-100",
            "source_object_index": 2,
        },
        {
            "desc": "red car",
            "points_norm1000": [22, 31, 80, 90],
            "object_id": "ann-101",
            "source_object_index": 4,
        },
        {
            "desc": "person",
            "points_norm1000": [900, 910, 998, 999],
            "object_id": "ann-102",
            "source_object_index": 7,
        },
    ]


def _span_text(tokenized, token_span) -> str:
    start = tokenized.offset_mapping[token_span.start][0]
    end = tokenized.offset_mapping[token_span.end - 1][1]
    return tokenized.chat_text[start:end]


def test_residual_boundary_adapter_slices_suffix_from_object_boundary() -> None:
    adapter = ResidualBoundaryAdapter(tokenizer=SnapshotTokenizer())
    rendered = adapter.render_objects(_objects())

    sliced = adapter.slice_from_boundary(rendered, boundary="object", object_index=1)
    tokenized = sliced.tokenized
    assistant_start = tokenized.assistant_token_span.start
    assistant_ids = tuple(
        tokenized.input_ids[
            tokenized.assistant_token_span.start : tokenized.assistant_token_span.end
        ]
    )

    expected_separator = get_detection_template("compact").render_separator(0, 1)
    if expected_separator:
        assert rendered.separator_spans[0].text(rendered.text) == expected_separator
        assert rendered.text[
            rendered.object_entries[0].entry_span.end : rendered.object_entries[1].entry_span.start
        ] == expected_separator
    else:
        assert rendered.object_entries[0].entry_span.end == rendered.object_entries[1].entry_span.start
    assert sliced.suffix_start == tokenized.object_entries[1].entry_span.start - assistant_start
    assert sliced.retained_prefix_input_ids == assistant_ids[: sliced.suffix_start]
    assert sliced.suffix_input_ids == assistant_ids[sliced.suffix_start :]
    assert sliced.retained_prefix_input_ids + sliced.suffix_input_ids == assistant_ids
    assert _span_text(tokenized, tokenized.object_entries[1].object_ref_start_span).startswith(
        OBJECT_REF_START_TOKEN
    )


def test_residual_boundary_adapter_drops_trailing_incomplete_object_span() -> None:
    adapter = ResidualBoundaryAdapter(tokenizer=SnapshotTokenizer())
    rendered = adapter.render_objects(_objects())

    sliced = adapter.slice_from_boundary(rendered, boundary="object", object_index=1)
    tokenized = sliced.tokenized
    assistant_start = tokenized.assistant_token_span.start
    second = tokenized.object_entries[1]
    naive_inside_object_prefix_end = second.desc_span.start - assistant_start

    assert sliced.suffix_start < naive_inside_object_prefix_end
    assert sliced.suffix_start == second.entry_span.start - assistant_start
    expected_prefix_end = (
        tokenized.object_entries[0].separator_span.end
        if tokenized.object_entries[0].separator_span is not None
        else tokenized.object_entries[0].entry_span.end
    )
    assert len(sliced.retained_prefix_input_ids) == expected_prefix_end - assistant_start
    assert sliced.suffix_input_ids[0] == tokenized.input_ids[second.object_ref_start_span.start]
    assert sliced.retained_prefix_input_ids[-1] == tokenized.input_ids[expected_prefix_end - 1]


def test_residual_boundary_adapter_matches_tokenized_detection_spans() -> None:
    tokenizer = SnapshotTokenizer()
    adapter = ResidualBoundaryAdapter(tokenizer=tokenizer)
    rendered = adapter.render_objects(_objects())

    sliced = adapter.slice_from_boundary(rendered, boundary="object", object_index=2)
    expected = tokenize_rendered_detection_conversation(rendered, tokenizer=tokenizer)
    expected_view = EncodedDetectionView.from_tokenized(expected)

    assert sliced.tokenized.object_entries == expected.object_entries
    assert sliced.tokenized.separator_spans == expected.separator_spans
    assert sliced.tokenized.terminal_span == expected.terminal_span
    assert sliced.tokenized.assistant_token_span == expected.assistant_token_span
    assert sliced.encoded_view == expected_view
    assert sliced.object_spans == adapter.object_spans(expected)
    assert [
        (span.object_start, span.desc_start, span.desc_end, span.box_start, span.coord_positions)
        for span in sliced.object_spans
    ] == [
        (
            entry.object_ref_start_span.start - expected.assistant_token_span.start,
            entry.desc_span.start - expected.assistant_token_span.start,
            entry.desc_span.end - expected.assistant_token_span.start,
            entry.bbox_start_span.start - expected.assistant_token_span.start,
            tuple(
                coord_span.start - expected.assistant_token_span.start
                for coord_span in entry.coord_spans
            ),
        )
        for entry in expected.object_entries
    ]


def test_residual_boundary_adapter_validates_no_adjacent_duplicate_boundary_token() -> None:
    adapter = ResidualBoundaryAdapter(tokenizer=SnapshotTokenizer())
    rendered = adapter.render_objects(_objects())
    sliced = adapter.slice_from_boundary(rendered, boundary="object", object_index=1)

    adapter.validate_no_adjacent_duplicate_boundary_token(
        retained_prefix_input_ids=sliced.retained_prefix_input_ids,
        suffix_input_ids=sliced.suffix_input_ids,
    )

    duplicated_boundary_prefix = (
        sliced.retained_prefix_input_ids + sliced.suffix_input_ids[:1]
    )
    with pytest.raises(ValueError, match="token duplicated"):
        adapter.validate_no_adjacent_duplicate_boundary_token(
            retained_prefix_input_ids=duplicated_boundary_prefix,
            suffix_input_ids=sliced.suffix_input_ids,
        )

    assert (
        rendered.text.count(OBJECT_REF_START_TOKEN)
        == (sliced.retained_prefix_input_ids + sliced.suffix_input_ids).count(
            sliced.suffix_input_ids[0]
        )
    )
    assert rendered.text.count(BOX_START_TOKEN) == len(rendered.object_entries)
