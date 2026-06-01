import re

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import CompactFullTemplate, Stage1JsonPrettyTemplate
from src.detection.tokenization import TokenRole, tokenize_rendered_detection_conversation


class SpecialAwareTokenizer:
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
            r'"bbox_2d": \[|'
            r"\],|"
            r"[A-Za-z0-9_]+|"
            r"\s+|"
            r".",
            re.DOTALL,
        )
        token_texts = [match.group(0) for match in token_pattern.finditer(text)]
        vocab = {token: index + 1 for index, token in enumerate(dict.fromkeys(token_texts))}
        return {
            "input_ids": [vocab[token] for token in token_texts],
            "offset_mapping": [
                (match.start(), match.end()) for match in token_pattern.finditer(text)
            ],
        }


class MultimodalAwareTokenizer(SpecialAwareTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<|im_start|>{message['role']}\n"
            f"{self._content_to_text(message['content'])}<|im_end|>\n"
            for message in messages
        )

    def _content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                assert isinstance(item, dict)
                if item.get("type") == "image":
                    parts.append("<image>")
                elif item.get("type") == "text":
                    parts.append(str(item.get("text")))
            return "".join(parts)
        raise TypeError(type(content).__name__)


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
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=3,
                object_instance_id="img-9:ann-502:src-3",
                desc="person",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_300|>",
                    "<|coord_400|>",
                ),
                category_id=1,
                category_name="person",
                coco_ann_id=502,
            ),
        ),
        width=640,
        height=480,
        image_id=9,
        file_name="image.jpg",
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7, 3)),
    )


def _token_texts(tokenized) -> list[str]:
    return [
        tokenized.chat_text[start:end]
        for start, end in tokenized.offset_mapping
    ]


def test_compact_full_token_roles_cover_markers_and_coordinate_special_tokens() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())
    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SpecialAwareTokenizer(),
    )
    token_texts = _token_texts(tokenized)

    object_ref_index = token_texts.index(OBJECT_REF_START_TOKEN)
    box_start_index = token_texts.index(BOX_START_TOKEN)
    coord_indices = [index for index, text in enumerate(token_texts) if text.startswith("<|coord_")]

    assert tokenized.token_roles[object_ref_index] is TokenRole.CONTROL
    assert tokenized.control_mask[object_ref_index] is True
    assert tokenized.token_roles[box_start_index] is TokenRole.BBOX_START
    assert tokenized.bbox_start_mask[box_start_index] is True
    assert coord_indices
    assert all(tokenized.token_roles[index] is TokenRole.COORD for index in coord_indices)
    assert all(tokenized.coord_mask[index] is True for index in coord_indices)
    assert tokenized.separator_spans == ()
    assert not any(tokenized.separator_mask)
    assert tokenized.assistant_stop_token_span is not None
    assert token_texts[tokenized.assistant_stop_token_span.start] == "<|im_end|>"
    assert tokenized.token_roles[tokenized.assistant_stop_token_span.start] is TokenRole.TERMINAL
    assert tokenized.terminal_mask[tokenized.assistant_stop_token_span.start] is True
    assert tokenized.assistant_mask[tokenized.assistant_stop_token_span.start] is False
    assert tokenized.labels[tokenized.assistant_stop_token_span.start] == tokenized.input_ids[
        tokenized.assistant_stop_token_span.start
    ]

    first_entry = tokenized.object_entries[0]
    assert first_entry.object_ref_start_span is not None
    assert token_texts[first_entry.object_ref_start_span.start] == OBJECT_REF_START_TOKEN
    assert token_texts[first_entry.bbox_start_span.start] == BOX_START_TOKEN
    assert [token_texts[span.start] for span in first_entry.coord_spans] == [
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
    ]


def test_multimodal_chat_template_content_lists_preserve_assistant_alignment() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())
    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=MultimodalAwareTokenizer(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image.jpg"},
                    {"type": "text", "text": "Detect every object."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rendered.text}],
            },
        ],
    )

    assistant_text = tokenized.chat_text[
        tokenized.assistant_char_span.start : tokenized.assistant_char_span.end
    ]
    assert assistant_text == rendered.text
    assert tokenized.assistant_mask[tokenized.assistant_token_span.start] is True
    assert tokenized.labels[tokenized.assistant_token_span.start] == tokenized.input_ids[
        tokenized.assistant_token_span.start
    ]


def test_stage1_json_pretty_token_roles_cover_object_desc_bbox_separator_and_closure() -> None:
    rendered = Stage1JsonPrettyTemplate().render_assistant(_sample())
    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SpecialAwareTokenizer(),
    )
    token_texts = _token_texts(tokenized)
    first_entry = tokenized.object_entries[0]

    assert any(tokenized.object_entry_mask[index] for index in range(first_entry.entry_span.start, first_entry.entry_span.end))
    assert any(
        tokenized.token_roles[index] is TokenRole.DESC
        for index in range(first_entry.desc_span.start, first_entry.desc_span.end)
    )
    assert tokenized.token_roles[first_entry.bbox_start_span.start] is TokenRole.BBOX_START
    assert any(tokenized.bbox_mask[index] for index in range(first_entry.bbox_span.start, first_entry.bbox_span.end))

    separator = tokenized.separator_spans[0]
    assert tokenized.token_roles[separator.start] is TokenRole.SEPARATOR
    assert tokenized.separator_mask[separator.start] is True
    assert tokenized.chat_text[
        tokenized.offset_mapping[separator.start][0] : tokenized.offset_mapping[separator.end - 1][1]
    ] == ", "

    assert tokenized.terminal_span is not None
    assert tokenized.token_roles[tokenized.terminal_span.start] is TokenRole.TERMINAL
    assert tokenized.terminal_mask[tokenized.terminal_span.start] is True
    assert tokenized.chat_text[
        tokenized.offset_mapping[tokenized.terminal_span.start][0] : tokenized.offset_mapping[tokenized.terminal_span.end - 1][1]
    ] == "]}"
    assert token_texts[first_entry.coord_spans[0].start] == "<|coord_10|>"
