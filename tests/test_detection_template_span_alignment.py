from dataclasses import replace
import re

import pytest

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import CharSpan, CompactFullTemplate, RenderedAssistantSequence
from src.detection.tokenization import (
    TokenRole,
    align_char_span_to_token_span,
    tokenize_rendered_detection_conversation,
)


class CharOffsetTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        self.chat_template_calls += 1
        rendered = []
        for message in messages:
            rendered.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        return "".join(rendered)

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        offsets = [(index, index + 1) for index in range(len(text))]
        return {"input_ids": [ord(char) for char in text], "offset_mapping": offsets}


class NoStopMarkerTokenizer(CharOffsetTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        self.chat_template_calls += 1
        rendered = []
        for message in messages:
            rendered.append(f"{message['role']}\n{message['content']}\n")
        return "".join(rendered)


def _sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-9:ann-501:src-7",
                desc="traffic light",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=10,
                category_name="traffic light",
                coco_ann_id=501,
            ),
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7,)),
    )


def test_align_char_span_to_token_span_requires_exact_token_boundaries() -> None:
    offsets = [(0, 1), (1, 3), (3, 6), (6, 7)]

    token_span = align_char_span_to_token_span(
        offsets,
        CharSpan(1, 3, "desc"),
    )

    assert token_span.start == 1
    assert token_span.end == 2
    assert token_span.label == "desc"

    with pytest.raises(ValueError, match="zero tokens"):
        align_char_span_to_token_span(offsets, CharSpan(2, 2, "empty"))

    with pytest.raises(ValueError, match="token boundary"):
        align_char_span_to_token_span(offsets, CharSpan(1, 2, "partial"))


def test_tokenization_applies_chat_template_once_and_aligns_assistant_span() -> None:
    tokenizer = CharOffsetTokenizer()
    rendered = CompactFullTemplate().render_assistant(_sample())

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=tokenizer,
        system_prompt="detect objects",
        user_content="<image>\nList objects.",
    )

    assert tokenizer.chat_template_calls == 1
    assert tokenized.chat_text.count(rendered.text) == 1
    assert tokenized.assistant_char_span.text(tokenized.chat_text) == rendered.text
    assert tokenized.assistant_token_span.char_span == tokenized.assistant_char_span
    assert tokenized.assistant_token_span.end - tokenized.assistant_token_span.start == len(rendered.text)
    assert set(tokenized.labels[: tokenized.assistant_token_span.start]) == {-100}
    assert tokenized.assistant_stop_token_span is not None
    assert tokenized.assistant_stop_token_span.char_span is not None
    assert tokenized.assistant_stop_token_span.char_span.text(tokenized.chat_text) == "<|im_end|>"
    assert tokenized.assistant_stop_token_span.start == tokenized.assistant_token_span.end
    assert tokenized.token_roles[tokenized.assistant_stop_token_span.start] is TokenRole.TERMINAL
    assert all(
        tokenized.terminal_mask[index] is True
        for index in tokenized.assistant_stop_token_span.token_indices()
    )
    assert all(
        tokenized.assistant_mask[index] is False
        for index in tokenized.assistant_stop_token_span.token_indices()
    )
    assert set(tokenized.labels[tokenized.assistant_stop_token_span.end :]) == {-100}
    assert tokenized.labels[tokenized.assistant_token_span.start : tokenized.assistant_token_span.end] == tokenized.input_ids[
        tokenized.assistant_token_span.start : tokenized.assistant_token_span.end
    ]
    assert tokenized.labels[
        tokenized.assistant_stop_token_span.start : tokenized.assistant_stop_token_span.end
    ] == tokenized.input_ids[
        tokenized.assistant_stop_token_span.start : tokenized.assistant_stop_token_span.end
    ]
    assert all(
        role is TokenRole.ASSISTANT or role is not TokenRole.IGNORE
        for role in tokenized.token_roles[tokenized.assistant_token_span.start : tokenized.assistant_token_span.end]
    )


def test_tokenization_does_not_guess_stop_span_without_explicit_marker() -> None:
    tokenizer = NoStopMarkerTokenizer()
    rendered = CompactFullTemplate().render_assistant(_sample())

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=tokenizer,
    )

    assert tokenizer.chat_template_calls == 1
    assert tokenized.assistant_stop_token_span is None
    assert set(tokenized.labels[: tokenized.assistant_token_span.start]) == {-100}
    assert set(tokenized.labels[tokenized.assistant_token_span.end :]) == {-100}


def test_tokenization_rejects_renderer_span_that_crosses_token_boundary() -> None:
    class ChunkedTokenizer(CharOffsetTokenizer):
        def __call__(
            self,
            text: str,
            *,
            return_offsets_mapping: bool,
            add_special_tokens: bool,
        ) -> dict[str, list[int] | list[tuple[int, int]]]:
            assert return_offsets_mapping is True
            assert add_special_tokens is False
            offsets = [(match.start(), match.end()) for match in re.finditer(r".{1,5}", text, re.DOTALL)]
            return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    rendered = CompactFullTemplate().render_assistant(_sample())
    shifted_desc = replace(
        rendered.object_entries[0],
        desc_span=CharSpan(
            rendered.object_entries[0].desc_span.start,
            rendered.object_entries[0].desc_span.start + 3,
            "desc",
        ),
    )
    rendered = replace(rendered, object_entries=(shifted_desc,))

    with pytest.raises(ValueError, match="token boundary"):
        tokenize_rendered_detection_conversation(rendered, tokenizer=ChunkedTokenizer())


def test_tokenization_rejects_missing_offset_mapping() -> None:
    class NoOffsetTokenizer(CharOffsetTokenizer):
        def __call__(
            self,
            text: str,
            *,
            return_offsets_mapping: bool,
            add_special_tokens: bool,
        ) -> dict[str, list[int]]:
            return {"input_ids": [1] * len(text)}

    rendered: RenderedAssistantSequence = CompactFullTemplate().render_assistant(_sample())

    with pytest.raises(ValueError, match="offset_mapping"):
        tokenize_rendered_detection_conversation(rendered, tokenizer=NoOffsetTokenizer())


def test_tokenization_accepts_nested_list_and_tuple_offset_mappings() -> None:
    class NestedOffsetTokenizer(CharOffsetTokenizer):
        def __init__(self, *, as_tuple: bool) -> None:
            super().__init__()
            self.as_tuple = as_tuple

        def __call__(
            self,
            text: str,
            *,
            return_offsets_mapping: bool,
            add_special_tokens: bool,
        ) -> dict[str, object]:
            assert return_offsets_mapping is True
            assert add_special_tokens is False
            offsets = tuple((index, index + 1) for index in range(len(text)))
            offset_mapping = (offsets,) if self.as_tuple else [list(offsets)]
            return {"input_ids": [ord(char) for char in text], "offset_mapping": offset_mapping}

    rendered = CompactFullTemplate().render_assistant(_sample())

    for tokenizer in (NestedOffsetTokenizer(as_tuple=False), NestedOffsetTokenizer(as_tuple=True)):
        tokenized = tokenize_rendered_detection_conversation(rendered, tokenizer=tokenizer)

        assert tokenized.assistant_char_span.text(tokenized.chat_text) == rendered.text
        assert tokenized.offset_mapping[0] == (0, 1)


def test_tokenization_rejects_malformed_offset_mapping_with_value_error() -> None:
    class MalformedOffsetTokenizer(CharOffsetTokenizer):
        def __init__(self, offset_mapping: object) -> None:
            super().__init__()
            self.offset_mapping = offset_mapping

        def __call__(
            self,
            text: str,
            *,
            return_offsets_mapping: bool,
            add_special_tokens: bool,
        ) -> dict[str, object]:
            assert return_offsets_mapping is True
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text], "offset_mapping": self.offset_mapping}

    rendered = CompactFullTemplate().render_assistant(_sample())

    for offset_mapping, message in (
        ([[(0, 1)], [(1, 2)]], "single sequence"),
        ([(0, 1, 2)], "pairs of integer offsets"),
        ([(0.9, 1.9)], "pairs of integer offsets"),
        ([("0", "1")], "pairs of integer offsets"),
        ([(False, True)], "pairs of integer offsets"),
    ):
        with pytest.raises(ValueError, match=message):
            tokenize_rendered_detection_conversation(
                rendered,
                tokenizer=MalformedOffsetTokenizer(offset_mapping),
            )
