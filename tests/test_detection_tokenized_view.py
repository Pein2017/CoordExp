import re

import pytest

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.template import (
    CharSpan,
    CompactFullTemplate,
    RenderSpanEvent,
    RenderedAssistantSequence,
)
from src.detection.tokenization import TokenRole, tokenize_rendered_detection_conversation


# Deterministic offset tokenizer for role/mask snapshotting. This intentionally
# models only the token boundaries needed by these tests, not real Qwen behavior.
class SnapshotTokenizer:
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


def _sample() -> NormalizedDetectionSample:
    return NormalizedDetectionSample(
        images=("image.jpg",),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=7,
                object_instance_id="img-9:ann-501:src-7",
                desc="cat",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ),
                category_id=17,
                category_name="cat",
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


def _two_object_sample() -> NormalizedDetectionSample:
    sample = _sample()
    return NormalizedDetectionSample(
        images=sample.images,
        objects=(
            sample.objects[0],
            NormalizedDetectionObject(
                normalized_object_index=1,
                source_object_index=8,
                object_instance_id="img-9:ann-502:src-8",
                desc="dog",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_30|>",
                    "<|coord_40|>",
                ),
                category_id=18,
                category_name="dog",
                coco_ann_id=502,
            ),
        ),
        width=sample.width,
        height=sample.height,
        image_id=sample.image_id,
        file_name=sample.file_name,
        metadata=sample.metadata,
        object_ordering=ObjectOrderingPlan.sorted().with_realized((7, 8)),
    )


def _token_texts(tokenized) -> tuple[str, ...]:
    return tuple(
        tokenized.chat_text[start:end]
        for start, end in tokenized.offset_mapping
    )


def _true_indices(mask: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(mask) if value)


def _manual_rendered(
    text: str,
    *,
    render_span_events: tuple[RenderSpanEvent, ...],
) -> RenderedAssistantSequence:
    return RenderedAssistantSequence(
        template_id="unit-test",
        template_version=1,
        text=text,
        object_entries=(),
        separator_spans=(),
        terminal_close_span=CharSpan(len(text), len(text), "terminal_close"),
        stop_marker_spans=(),
        structural_token_spans=(),
        trie_eligible_spans=(),
        render_span_events=render_span_events,
    )


def _render_event(
    start: int,
    end: int,
    label: str,
    *,
    primary_role: str | None,
    mask_groups: tuple[str, ...],
    classifying: bool,
    priority: int,
) -> RenderSpanEvent:
    return RenderSpanEvent(
        char_span=CharSpan(start, end, label),
        span_kind=label,
        primary_role=primary_role,
        mask_groups=frozenset(mask_groups),
        classifying=classifying,
        priority=priority,
    )


def test_current_token_roles_are_snapshot_before_refactor() -> None:
    rendered = CompactFullTemplate().render_assistant(_sample())
    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
    )
    token_texts = _token_texts(tokenized)

    assert token_texts == (
        "<",
        "|",
        "im_start",
        "|",
        ">",
        "user",
        "\n",
        "<",
        "image",
        ">",
        "<|im_end|>",
        "\n",
        "<",
        "|",
        "im_start",
        "|",
        ">",
        "assistant",
        "\n",
        OBJECT_REF_START_TOKEN,
        "cat",
        BOX_START_TOKEN,
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
        "<|im_end|>",
        "\n",
    )
    assert tuple(role.value for role in tokenized.token_roles) == (
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "control",
        "desc",
        "bbox_start",
        "coord",
        "coord",
        "coord",
        "coord",
        "terminal",
        "ignore",
    )
    assert tokenized.token_roles[19] is TokenRole.CONTROL
    assert tokenized.token_roles[20] is TokenRole.DESC
    assert tokenized.token_roles[21] is TokenRole.BBOX_START
    assert all(tokenized.token_roles[index] is TokenRole.COORD for index in (22, 23, 24, 25))
    assert tokenized.token_roles[26] is TokenRole.TERMINAL

    assert _true_indices(tokenized.assistant_mask) == (19, 20, 21, 22, 23, 24, 25)
    assert _true_indices(tokenized.object_entry_mask) == (19, 20, 21, 22, 23, 24, 25)
    assert _true_indices(tokenized.control_mask) == (19, 21)
    assert _true_indices(tokenized.desc_mask) == (20,)
    assert _true_indices(tokenized.bbox_start_mask) == (21,)
    assert _true_indices(tokenized.coord_mask) == (22, 23, 24, 25)
    assert _true_indices(tokenized.terminal_mask) == (26,)
    assert _true_indices(tokenized.separator_mask) == ()

    assert len(tokenized.object_entries) == 1
    assert tokenized.object_entries[0].object_ref_start_span.start == 19
    assert tokenized.object_entries[0].desc_span.start == 20
    assert tokenized.object_entries[0].bbox_start_span.start == 21
    assert tuple(span.start for span in tokenized.object_entries[0].coord_spans) == (22, 23, 24, 25)
    assert tokenized.assistant_stop_token_span is not None
    assert tokenized.assistant_stop_token_span.start == 26


def test_next_token_prediction_positions_are_explicit() -> None:
    view = tokenize_rendered_detection_conversation(
        CompactFullTemplate().render_assistant(_sample()),
        tokenizer=SnapshotTokenizer(),
    )

    assert view.token_position_origin == "TokenizedDetectionExample.tokenized"
    assert view.supervised_label_positions
    assert 0 not in view.supervised_label_positions
    assert view.next_token_prediction_positions == tuple(
        label_pos - 1 for label_pos in view.supervised_label_positions
    )
    for label_pos in view.supervised_label_positions:
        assert view.labels[label_pos] == view.input_ids[label_pos]
        assert view.next_token_prediction_position_for(label_pos) == label_pos - 1

    with pytest.raises(ValueError, match="position 0"):
        view.next_token_prediction_position_for(0)


def test_compact_two_object_tokenization_snapshots_separator_and_entry_boundaries() -> None:
    rendered = CompactFullTemplate().render_assistant(_two_object_sample())
    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
    )
    token_texts = _token_texts(tokenized)

    assert token_texts == (
        "<",
        "|",
        "im_start",
        "|",
        ">",
        "user",
        "\n",
        "<",
        "image",
        ">",
        "<|im_end|>",
        "\n",
        "<",
        "|",
        "im_start",
        "|",
        ">",
        "assistant",
        "\n",
        OBJECT_REF_START_TOKEN,
        "cat",
        BOX_START_TOKEN,
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
        "\n",
        OBJECT_REF_START_TOKEN,
        "dog",
        BOX_START_TOKEN,
        "<|coord_10|>",
        "<|coord_20|>",
        "<|coord_30|>",
        "<|coord_40|>",
        "<|im_end|>",
        "\n",
    )
    assert tuple(role.value for role in tokenized.token_roles) == (
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "ignore",
        "control",
        "desc",
        "bbox_start",
        "coord",
        "coord",
        "coord",
        "coord",
        "separator",
        "control",
        "desc",
        "bbox_start",
        "coord",
        "coord",
        "coord",
        "coord",
        "terminal",
        "ignore",
    )
    assert _true_indices(tokenized.assistant_mask) == tuple(range(19, 34))
    assert _true_indices(tokenized.object_entry_mask) == (
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    )
    assert _true_indices(tokenized.control_mask) == (19, 21, 27, 29)
    assert _true_indices(tokenized.desc_mask) == (20, 28)
    assert _true_indices(tokenized.bbox_start_mask) == (21, 29)
    assert _true_indices(tokenized.coord_mask) == (22, 23, 24, 25, 30, 31, 32, 33)
    assert _true_indices(tokenized.separator_mask) == (26,)
    assert _true_indices(tokenized.terminal_mask) == (34,)
    assert tokenized.token_roles[26] is TokenRole.SEPARATOR
    assert tokenized.separator_spans[0].start == 26
    assert tokenized.separator_spans[0].end == 27
    assert token_texts[tokenized.separator_spans[0].start] == "\n"

    first, second = tokenized.object_entries
    assert len(tokenized.object_entries) == 2
    assert first.entry_span.start == 19
    assert first.entry_span.end == 26
    assert first.separator_span is not None
    assert first.separator_span.start == 26
    assert first.separator_span.end == 27
    assert first.desc_span.start == 20
    assert tuple(span.start for span in first.coord_spans) == (22, 23, 24, 25)
    assert second.entry_span.start == 27
    assert second.entry_span.end == 34
    assert second.separator_span is None
    assert second.desc_span.start == 28
    assert tuple(span.start for span in second.coord_spans) == (30, 31, 32, 33)
    assert tokenized.assistant_stop_token_span is not None
    assert tokenized.assistant_stop_token_span.start == 34


def test_token_roles_use_priority_not_call_order() -> None:
    text = "cat<|coord_1|>"
    rendered = _manual_rendered(
        text,
        render_span_events=(
            _render_event(
                0,
                3,
                "desc",
                primary_role="DESC",
                mask_groups=("desc",),
                classifying=True,
                priority=10,
            ),
            _render_event(
                3,
                len(text),
                "coord",
                primary_role="COORD",
                mask_groups=("coord",),
                classifying=True,
                priority=10,
            ),
            _render_event(
                0,
                len(text),
                "broad_control",
                primary_role="CONTROL",
                mask_groups=("assistant", "control"),
                classifying=True,
                priority=1,
            ),
        ),
    )

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
    )
    token_texts = _token_texts(tokenized)
    desc_token_pos = token_texts.index("cat")
    coord_token_pos = token_texts.index("<|coord_1|>")

    assert tokenized.token_roles[desc_token_pos] is TokenRole.DESC
    assert tokenized.token_roles[coord_token_pos] is TokenRole.COORD
    assert tokenized.control_mask[desc_token_pos]
    assert tokenized.control_mask[coord_token_pos]


def test_control_mask_can_include_bbox_start_without_collapsing_role() -> None:
    rendered = _manual_rendered(
        BOX_START_TOKEN,
        render_span_events=(
            _render_event(
                0,
                len(BOX_START_TOKEN),
                "assistant",
                primary_role=None,
                mask_groups=("assistant",),
                classifying=False,
                priority=0,
            ),
            _render_event(
                0,
                len(BOX_START_TOKEN),
                "bbox_start",
                primary_role="BBOX_START",
                mask_groups=("bbox_start", "control"),
                classifying=True,
                priority=10,
            ),
        ),
    )

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
    )
    bbox_start_pos = _token_texts(tokenized).index(BOX_START_TOKEN)

    assert tokenized.token_roles[bbox_start_pos] is TokenRole.BBOX_START
    assert tokenized.bbox_start_mask[bbox_start_pos]
    assert tokenized.control_mask[bbox_start_pos]


def test_unknown_assistant_text_uses_assistant_fallback_not_control() -> None:
    rendered = _manual_rendered(
        "mystery",
        render_span_events=(
            _render_event(
                0,
                len("mystery"),
                "assistant",
                primary_role=None,
                mask_groups=("assistant",),
                classifying=False,
                priority=0,
            ),
        ),
    )

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
    )
    unknown_pos = _token_texts(tokenized).index("mystery")

    assert tokenized.token_roles[unknown_pos] is TokenRole.ASSISTANT
    assert tokenized.assistant_mask[unknown_pos]
    assert not tokenized.control_mask[unknown_pos]


def test_separator_control_group_preserves_legacy_non_control_mask() -> None:
    rendered = _manual_rendered(
        "\n",
        render_span_events=(
            _render_event(
                0,
                1,
                "assistant",
                primary_role=None,
                mask_groups=("assistant",),
                classifying=False,
                priority=0,
            ),
            _render_event(
                0,
                1,
                "separator",
                primary_role="SEPARATOR",
                mask_groups=("separator", "control"),
                classifying=True,
                priority=10,
            ),
        ),
    )

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
    )
    separator_pos = tokenized.assistant_token_span.start

    assert tokenized.assistant_char_span.text(tokenized.chat_text) == "\n"
    assert tokenized.token_roles[separator_pos] is TokenRole.SEPARATOR
    assert tokenized.separator_mask[separator_pos]
    assert not tokenized.control_mask[separator_pos]


def test_duplicate_payload_selects_final_assistant_stop_bounded_occurrence() -> None:
    rendered = _manual_rendered(
        "\n",
        render_span_events=(
            _render_event(
                0,
                1,
                "assistant",
                primary_role=None,
                mask_groups=("assistant",),
                classifying=False,
                priority=0,
            ),
        ),
    )

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
        user_content="\n",
    )
    assistant_header_start = tokenized.chat_text.rfind("<|im_start|>assistant\n")

    assert assistant_header_start >= 0
    assert tokenized.assistant_char_span.start > assistant_header_start
    assert tokenized.assistant_char_span.text(tokenized.chat_text) == "\n"


def test_messages_with_non_final_assistant_response_are_rejected() -> None:
    rendered = _manual_rendered(
        "cat",
        render_span_events=(
            _render_event(
                0,
                3,
                "assistant",
                primary_role=None,
                mask_groups=("assistant",),
                classifying=False,
                priority=0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="assistant response must be the final"):
        tokenize_rendered_detection_conversation(
            rendered,
            tokenizer=SnapshotTokenizer(),
            messages=(
                {"role": "assistant", "content": "cat"},
                {"role": "user", "content": "cat"},
            ),
        )


def test_equal_priority_overlap_is_tracked_below_winning_priority() -> None:
    rendered = _manual_rendered(
        "cat",
        render_span_events=(
            _render_event(
                0,
                3,
                "assistant",
                primary_role=None,
                mask_groups=("assistant",),
                classifying=False,
                priority=0,
            ),
            _render_event(
                0,
                3,
                "high_priority_desc",
                primary_role="DESC",
                mask_groups=("desc",),
                classifying=True,
                priority=20,
            ),
            _render_event(
                0,
                3,
                "lower_priority_control",
                primary_role="CONTROL",
                mask_groups=("control",),
                classifying=True,
                priority=5,
            ),
            _render_event(
                0,
                3,
                "lower_priority_coord",
                primary_role="COORD",
                mask_groups=("coord",),
                classifying=True,
                priority=5,
            ),
        ),
    )

    with pytest.raises(ValueError, match="equal-priority classifying token overlap"):
        tokenize_rendered_detection_conversation(
            rendered,
            tokenizer=SnapshotTokenizer(),
        )


def test_ignore_mask_group_is_explicit_noop() -> None:
    rendered = _manual_rendered(
        "mystery",
        render_span_events=(
            _render_event(
                0,
                len("mystery"),
                "assistant",
                primary_role=None,
                mask_groups=("assistant",),
                classifying=False,
                priority=0,
            ),
            _render_event(
                0,
                len("mystery"),
                "ignore",
                primary_role=None,
                mask_groups=("ignore",),
                classifying=False,
                priority=0,
            ),
        ),
    )

    tokenized = tokenize_rendered_detection_conversation(
        rendered,
        tokenizer=SnapshotTokenizer(),
    )
    ignored_pos = _token_texts(tokenized).index("mystery")

    assert tokenized.token_roles[ignored_pos] is TokenRole.ASSISTANT
    assert tokenized.assistant_mask[ignored_pos]
    assert not tokenized.control_mask[ignored_pos]
