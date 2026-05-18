"""Tokenization and span alignment for rendered detection templates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from src.detection.template import (
    CharSpan,
    RenderSpanEvent,
    RenderedAssistantSequence,
    RenderedConversation,
    RenderedObjectEntry,
)

_QWEN_CHAT_TEMPLATE_STOP_MARKER = "<|im_end|>"


class TokenizerWithOffsets(Protocol):
    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> Mapping[str, Any]:
        ...


class TokenRole(str, Enum):
    IGNORE = "ignore"
    ASSISTANT = "assistant"
    OBJECT_ENTRY = "object_entry"
    DESC = "desc"
    BBOX_START = "bbox_start"
    COORD = "coord"
    SEPARATOR = "separator"
    TERMINAL = "terminal"
    CONTROL = "control"


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    label: str
    char_span: CharSpan | None = None

    def token_indices(self) -> range:
        return range(self.start, self.end)


@dataclass(frozen=True)
class TokenizedObjectEntry:
    object_instance_id: str
    object_index: int
    source_object_index: int
    entry_span: TokenSpan
    desc_span: TokenSpan
    object_ref_start_span: TokenSpan | None
    bbox_start_span: TokenSpan
    bbox_span: TokenSpan
    coord_spans: tuple[TokenSpan, ...]
    separator_span: TokenSpan | None
    control_spans: tuple[TokenSpan, ...]
    trie_eligible_span: TokenSpan
    object_id: str | None = None
    source_role: str | None = None
    coordinate_weight: float | None = None
    regression_weight: float | None = None
    hard_bbox_supervision: bool | None = None


@dataclass(frozen=True)
class TokenizedDetectionExample:
    rendered_assistant: RenderedAssistantSequence
    chat_text: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    assistant_char_span: CharSpan
    assistant_token_span: TokenSpan
    assistant_stop_token_span: TokenSpan | None
    object_entries: tuple[TokenizedObjectEntry, ...]
    structural_spans: tuple[TokenSpan, ...]
    separator_spans: tuple[TokenSpan, ...]
    terminal_span: TokenSpan | None
    stop_marker_spans: tuple[TokenSpan, ...]
    token_roles: tuple[TokenRole, ...]
    assistant_mask: tuple[bool, ...]
    object_entry_mask: tuple[bool, ...]
    desc_mask: tuple[bool, ...]
    bbox_start_mask: tuple[bool, ...]
    bbox_mask: tuple[bool, ...]
    coord_mask: tuple[bool, ...]
    separator_mask: tuple[bool, ...]
    terminal_mask: tuple[bool, ...]
    control_mask: tuple[bool, ...]
    token_position_origin: str = "TokenizedDetectionExample.tokenized"

    @property
    def supervised_label_positions(self) -> tuple[int, ...]:
        """Label indices whose teacher tokens have valid next-token predictors.

        The label position stores the teacher token itself: ``labels[position]``.
        Under causal next-token training, ``logits[position - 1]`` predicts that
        teacher token, so position 0 is intentionally excluded from this view.
        """
        return tuple(
            position
            for position, label in enumerate(self.labels)
            if position > 0 and int(label) != -100
        )

    @property
    def next_token_prediction_positions(self) -> tuple[int, ...]:
        """Logit indices that predict ``supervised_label_positions``."""
        return tuple(
            self.next_token_prediction_position_for(position)
            for position in self.supervised_label_positions
        )

    def next_token_prediction_position_for(self, label_position: int) -> int:
        """Return the logit index that predicts ``labels[label_position]``."""
        label_position = int(label_position)
        if label_position <= 0:
            raise ValueError("label position 0 has no next-token prediction position")
        if label_position >= len(self.labels):
            raise IndexError("label position is outside labels")
        if int(self.labels[label_position]) == -100:
            raise ValueError("label position is not supervised")
        return label_position - 1


def align_char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    span: CharSpan,
    *,
    base_offset: int = 0,
) -> TokenSpan:
    target_start = base_offset + span.start
    target_end = base_offset + span.end
    if target_start >= target_end:
        raise ValueError(f"Span {span.label!r} maps to zero tokens")

    overlapping: list[tuple[int, int, int]] = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if token_end <= target_start:
            continue
        if token_start >= target_end:
            break
        if token_start < target_end and token_end > target_start:
            overlapping.append((token_index, token_start, token_end))

    if not overlapping:
        raise ValueError(f"Span {span.label!r} maps to zero tokens")

    first_index, first_start, _ = overlapping[0]
    last_index, _, last_end = overlapping[-1]
    if first_start != target_start or last_end != target_end:
        raise ValueError(
            f"Span {span.label!r} crosses an unexpected tokenizer token boundary"
        )

    previous_end = first_start
    for _, token_start, token_end in overlapping:
        if token_start != previous_end:
            raise ValueError(
                f"Span {span.label!r} crosses an unexpected tokenizer token boundary"
            )
        previous_end = token_end

    return TokenSpan(
        start=first_index,
        end=last_index + 1,
        label=span.label,
        char_span=span,
    )


def tokenize_rendered_detection_conversation(
    rendered: RenderedAssistantSequence | RenderedConversation,
    *,
    tokenizer: TokenizerWithOffsets,
    system_prompt: str | None = None,
    user_content: str = "<image>",
    messages: Sequence[Mapping[str, Any]] | None = None,
    assistant_stop_markers: Sequence[str] | None = None,
) -> TokenizedDetectionExample:
    stop_markers = _assistant_stop_marker_candidates(
        tokenizer,
        assistant_stop_markers=assistant_stop_markers,
    )
    rendered_assistant = _assistant_from_rendered(rendered)
    chat_messages = _build_messages(
        rendered_assistant,
        rendered=rendered,
        system_prompt=system_prompt,
        user_content=user_content,
        messages=messages,
    )
    chat_text = _apply_chat_template(tokenizer, chat_messages)
    assistant_char_span = _find_assistant_char_span(
        chat_text,
        rendered_assistant.text,
        assistant_stop_markers=stop_markers,
    )
    assistant_stop_char_span = _find_assistant_stop_char_span(
        chat_text,
        assistant_char_span,
        assistant_stop_markers=stop_markers,
    )

    encoded = tokenizer(
        chat_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = _as_flat_int_tuple(encoded.get("input_ids"), field_name="input_ids")
    offsets = _as_offset_tuple(encoded.get("offset_mapping"))
    if len(input_ids) != len(offsets):
        raise ValueError("input_ids and offset_mapping must have the same length")

    assistant_stop_token_span = _align_optional_chat_span(
        assistant_stop_char_span,
        offsets=offsets,
    )
    if assistant_char_span.start == assistant_char_span.end:
        if assistant_stop_token_span is None:
            raise ValueError("empty assistant payload requires a chat-template stop marker")
        assistant_token_span = TokenSpan(
            start=assistant_stop_token_span.start,
            end=assistant_stop_token_span.start,
            label=assistant_char_span.label,
            char_span=assistant_char_span,
        )
    else:
        assistant_token_span = align_char_span_to_token_span(
            offsets,
            assistant_char_span,
        )
    object_entries = tuple(
        _align_object_entry(
            entry,
            offsets=offsets,
            assistant_base_offset=assistant_char_span.start,
        )
        for entry in rendered_assistant.object_entries
    )
    structural_spans = _align_non_empty_spans(
        rendered_assistant.structural_token_spans,
        offsets=offsets,
        assistant_base_offset=assistant_char_span.start,
    )
    separator_spans = _align_non_empty_spans(
        rendered_assistant.separator_spans,
        offsets=offsets,
        assistant_base_offset=assistant_char_span.start,
    )
    terminal_span = _align_optional_non_empty_span(
        rendered_assistant.terminal_close_span,
        offsets=offsets,
        assistant_base_offset=assistant_char_span.start,
    )
    stop_marker_spans = _align_non_empty_spans(
        rendered_assistant.stop_marker_spans,
        offsets=offsets,
        assistant_base_offset=assistant_char_span.start,
    )
    if assistant_stop_token_span is not None:
        stop_marker_spans = (*stop_marker_spans, assistant_stop_token_span)

    masks = _build_masks_and_roles(
        input_ids=input_ids,
        assistant_token_span=assistant_token_span,
        object_entries=object_entries,
        structural_spans=structural_spans,
        separator_spans=separator_spans,
        terminal_span=terminal_span,
        stop_marker_spans=stop_marker_spans,
        render_span_events=rendered_assistant.render_span_events,
        offsets=offsets,
        assistant_base_offset=assistant_char_span.start,
    )
    labels = tuple(
        token_id
        if masks.assistant_mask[token_index] or masks.terminal_mask[token_index]
        else -100
        for token_index, token_id in enumerate(input_ids)
    )

    return TokenizedDetectionExample(
        rendered_assistant=rendered_assistant,
        chat_text=chat_text,
        input_ids=input_ids,
        labels=labels,
        offset_mapping=offsets,
        assistant_char_span=assistant_char_span,
        assistant_token_span=assistant_token_span,
        assistant_stop_token_span=assistant_stop_token_span,
        object_entries=object_entries,
        structural_spans=structural_spans,
        separator_spans=separator_spans,
        terminal_span=terminal_span,
        stop_marker_spans=stop_marker_spans,
        token_roles=masks.token_roles,
        assistant_mask=masks.assistant_mask,
        object_entry_mask=masks.object_entry_mask,
        desc_mask=masks.desc_mask,
        bbox_start_mask=masks.bbox_start_mask,
        bbox_mask=masks.bbox_mask,
        coord_mask=masks.coord_mask,
        separator_mask=masks.separator_mask,
        terminal_mask=masks.terminal_mask,
        control_mask=masks.control_mask,
    )


@dataclass(frozen=True)
class _MasksAndRoles:
    token_roles: tuple[TokenRole, ...]
    assistant_mask: tuple[bool, ...]
    object_entry_mask: tuple[bool, ...]
    desc_mask: tuple[bool, ...]
    bbox_start_mask: tuple[bool, ...]
    bbox_mask: tuple[bool, ...]
    coord_mask: tuple[bool, ...]
    separator_mask: tuple[bool, ...]
    terminal_mask: tuple[bool, ...]
    control_mask: tuple[bool, ...]


@dataclass(frozen=True)
class _AlignedRenderSpanEvent:
    event: RenderSpanEvent
    token_span: TokenSpan
    primary_role: TokenRole | None


def _assistant_from_rendered(
    rendered: RenderedAssistantSequence | RenderedConversation,
) -> RenderedAssistantSequence:
    if isinstance(rendered, RenderedAssistantSequence):
        return rendered
    return rendered.assistant


def _build_messages(
    rendered_assistant: RenderedAssistantSequence,
    *,
    rendered: RenderedAssistantSequence | RenderedConversation,
    system_prompt: str | None,
    user_content: str,
    messages: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    if messages is not None:
        chat_messages = [dict(message) for message in messages]
    elif isinstance(rendered, RenderedConversation):
        chat_messages = [dict(message) for message in rendered.messages]
    else:
        chat_messages = []
        if system_prompt is not None:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.append({"role": "user", "content": user_content})

    assistant_message_indices = [
        index
        for index, message in enumerate(chat_messages)
        if message.get("role") == "assistant"
    ]
    if not assistant_message_indices:
        chat_messages.append({"role": "assistant", "content": rendered_assistant.text})
    elif len(assistant_message_indices) == 1:
        assistant_message_index = assistant_message_indices[0]
        if assistant_message_index != len(chat_messages) - 1:
            raise ValueError("assistant response must be the final chat message")
        assistant_message = chat_messages[assistant_message_index]
        assistant_text = _extract_text_content(
            assistant_message.get("content"),
            require_text_only=True,
            context="assistant message content",
        )
        if assistant_text != rendered_assistant.text:
            raise ValueError("assistant message content must match rendered text")
    else:
        raise ValueError("messages must contain at most one assistant response")

    return tuple(chat_messages)


def _apply_chat_template(
    tokenizer: TokenizerWithOffsets,
    messages: Sequence[Mapping[str, Any]],
) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return "".join(
            f"<|im_start|>{message['role']}\n"
            f"{_chat_content_to_fallback_text(message.get('content'))}<|im_end|>\n"
            for message in messages
        )

    chat_text = apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(chat_text, str):
        raise ValueError("tokenizer.apply_chat_template(..., tokenize=False) must return str")
    return chat_text


def _extract_text_content(
    content: Any,
    *,
    require_text_only: bool,
    context: str,
) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        raise ValueError(f"{context} must be a string or chat-template content list")

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise ValueError(f"{context} list entries must be mappings")
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{context} text entries must contain string text")
            text_parts.append(text)
        elif require_text_only:
            raise ValueError(f"{context} may only contain text entries")

    return "".join(text_parts)


def _chat_content_to_fallback_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        raise ValueError("message content must be a string or chat-template content list")

    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise ValueError("chat-template content list entries must be mappings")
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("chat-template text entries must contain string text")
            parts.append(text)
        elif item_type == "image":
            parts.append("<image>")
        else:
            raise ValueError(f"unsupported chat-template content type: {item_type!r}")
    return "".join(parts)


def _find_assistant_char_span(
    chat_text: str,
    assistant_text: str,
    *,
    assistant_stop_markers: Sequence[str],
) -> CharSpan:
    if not assistant_text:
        return _find_empty_assistant_char_span(
            chat_text,
            assistant_stop_markers=assistant_stop_markers,
        )

    assistant_start = chat_text.find(assistant_text)
    if assistant_start < 0:
        raise ValueError("rendered assistant text was not found after chat-template rendering")
    duplicate_start = chat_text.find(assistant_text, assistant_start + 1)
    if duplicate_start >= 0:
        stop_bounded_starts = _assistant_text_starts_bounded_by_stop_marker(
            chat_text,
            assistant_text,
            assistant_stop_markers=assistant_stop_markers,
        )
        if stop_bounded_starts:
            assistant_start = stop_bounded_starts[-1]
        else:
            raise ValueError("rendered assistant text appears more than once in chat text")

    return CharSpan(
        assistant_start,
        assistant_start + len(assistant_text),
        "assistant",
    )


def _find_empty_assistant_char_span(
    chat_text: str,
    *,
    assistant_stop_markers: Sequence[str],
) -> CharSpan:
    header = "<|im_start|>assistant\n"
    starts: list[int] = []
    cursor = 0
    while True:
        header_start = chat_text.find(header, cursor)
        if header_start < 0:
            break
        content_start = header_start + len(header)
        if any(chat_text.startswith(marker, content_start) for marker in assistant_stop_markers):
            starts.append(content_start)
        cursor = content_start + 1
    if not starts:
        raise ValueError(
            "empty rendered assistant text was not found before an assistant stop marker"
        )
    content_start = starts[-1]
    return CharSpan(content_start, content_start, "assistant")


def _assistant_text_starts_bounded_by_stop_marker(
    chat_text: str,
    assistant_text: str,
    *,
    assistant_stop_markers: Sequence[str],
) -> tuple[int, ...]:
    starts: list[int] = []
    search_start = 0
    while True:
        candidate_start = chat_text.find(assistant_text, search_start)
        if candidate_start < 0:
            break
        candidate_end = candidate_start + len(assistant_text)
        if any(
            chat_text.startswith(marker, candidate_end)
            for marker in assistant_stop_markers
        ):
            starts.append(candidate_start)
        search_start = candidate_start + 1

    if starts:
        return tuple(starts)
    if assistant_stop_markers:
        raise ValueError("rendered assistant text appears more than once in chat text")
    return ()


def _find_assistant_stop_char_span(
    chat_text: str,
    assistant_char_span: CharSpan,
    *,
    assistant_stop_markers: Sequence[str],
) -> CharSpan | None:
    stop_start = assistant_char_span.end
    for marker in assistant_stop_markers:
        if chat_text.startswith(marker, stop_start):
            return CharSpan(stop_start, stop_start + len(marker), "assistant_stop")
    return None


def _assistant_stop_marker_candidates(
    tokenizer: TokenizerWithOffsets,
    *,
    assistant_stop_markers: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if assistant_stop_markers is not None:
        markers = list(assistant_stop_markers)
        if not markers or any(not isinstance(marker, str) or not marker for marker in markers):
            raise ValueError("assistant_stop_markers must contain non-empty strings")
        return tuple(dict.fromkeys(markers))

    markers = [_QWEN_CHAT_TEMPLATE_STOP_MARKER]
    eos_token = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_token, str) and eos_token and eos_token not in markers:
        markers.append(eos_token)
    return tuple(markers)


def _align_object_entry(
    entry: RenderedObjectEntry,
    *,
    offsets: Sequence[tuple[int, int]],
    assistant_base_offset: int,
) -> TokenizedObjectEntry:
    return TokenizedObjectEntry(
        object_instance_id=entry.object_instance_id,
        object_index=entry.object_index,
        source_object_index=entry.source_object_index,
        entry_span=align_char_span_to_token_span(
            offsets,
            entry.entry_span,
            base_offset=assistant_base_offset,
        ),
        desc_span=align_char_span_to_token_span(
            offsets,
            entry.desc_span,
            base_offset=assistant_base_offset,
        ),
        object_ref_start_span=_align_optional_non_empty_span(
            entry.object_ref_start_span,
            offsets=offsets,
            assistant_base_offset=assistant_base_offset,
        ),
        bbox_start_span=align_char_span_to_token_span(
            offsets,
            entry.bbox_start_span,
            base_offset=assistant_base_offset,
        ),
        bbox_span=align_char_span_to_token_span(
            offsets,
            entry.bbox_span,
            base_offset=assistant_base_offset,
        ),
        coord_spans=_align_non_empty_spans(
            entry.coord_spans,
            offsets=offsets,
            assistant_base_offset=assistant_base_offset,
        ),
        separator_span=_align_optional_non_empty_span(
            entry.separator_span,
            offsets=offsets,
            assistant_base_offset=assistant_base_offset,
        ),
        control_spans=_align_non_empty_spans(
            entry.control_spans,
            offsets=offsets,
            assistant_base_offset=assistant_base_offset,
        ),
        trie_eligible_span=align_char_span_to_token_span(
            offsets,
            entry.trie_eligible_span,
            base_offset=assistant_base_offset,
        ),
        object_id=entry.object_id,
        source_role=entry.source_role,
        coordinate_weight=entry.coordinate_weight,
        regression_weight=entry.regression_weight,
        hard_bbox_supervision=entry.hard_bbox_supervision,
    )


def _align_non_empty_spans(
    spans: Sequence[CharSpan],
    *,
    offsets: Sequence[tuple[int, int]],
    assistant_base_offset: int,
) -> tuple[TokenSpan, ...]:
    return tuple(
        token_span
        for span in spans
        if (token_span := _align_optional_non_empty_span(
            span,
            offsets=offsets,
            assistant_base_offset=assistant_base_offset,
        ))
        is not None
    )


def _align_optional_non_empty_span(
    span: CharSpan | None,
    *,
    offsets: Sequence[tuple[int, int]],
    assistant_base_offset: int,
) -> TokenSpan | None:
    if span is None or span.start == span.end:
        return None
    return align_char_span_to_token_span(
        offsets,
        span,
        base_offset=assistant_base_offset,
    )


def _align_optional_chat_span(
    span: CharSpan | None,
    *,
    offsets: Sequence[tuple[int, int]],
) -> TokenSpan | None:
    if span is None:
        return None
    return align_char_span_to_token_span(offsets, span)


def _build_masks_and_roles(
    *,
    input_ids: Sequence[int],
    assistant_token_span: TokenSpan,
    object_entries: Sequence[TokenizedObjectEntry],
    structural_spans: Sequence[TokenSpan],
    separator_spans: Sequence[TokenSpan],
    terminal_span: TokenSpan | None,
    stop_marker_spans: Sequence[TokenSpan],
    render_span_events: Sequence[RenderSpanEvent],
    offsets: Sequence[tuple[int, int]],
    assistant_base_offset: int,
) -> _MasksAndRoles:
    seq_len = len(input_ids)
    token_roles = [TokenRole.IGNORE for _ in range(seq_len)]
    assistant_mask = [False for _ in range(seq_len)]
    object_entry_mask = [False for _ in range(seq_len)]
    desc_mask = [False for _ in range(seq_len)]
    bbox_start_mask = [False for _ in range(seq_len)]
    bbox_mask = [False for _ in range(seq_len)]
    coord_mask = [False for _ in range(seq_len)]
    separator_mask = [False for _ in range(seq_len)]
    terminal_mask = [False for _ in range(seq_len)]
    control_mask = [False for _ in range(seq_len)]

    if render_span_events:
        aligned_events = _align_render_span_events(
            render_span_events,
            offsets=offsets,
            assistant_base_offset=assistant_base_offset,
        )
        _project_render_span_events(
            aligned_events,
            token_roles=token_roles,
            assistant_mask=assistant_mask,
            object_entry_mask=object_entry_mask,
            desc_mask=desc_mask,
            bbox_start_mask=bbox_start_mask,
            bbox_mask=bbox_mask,
            coord_mask=coord_mask,
            separator_mask=separator_mask,
            terminal_mask=terminal_mask,
            control_mask=control_mask,
        )
        _mark_container_projection_masks(
            assistant_token_span,
            object_entries,
            assistant_mask=assistant_mask,
            object_entry_mask=object_entry_mask,
        )
        _mark_object_projection_masks(
            object_entries,
            bbox_mask=bbox_mask,
            bbox_start_mask=bbox_start_mask,
        )
        _apply_container_role_fallbacks(
            token_roles,
            assistant_mask=assistant_mask,
            object_entry_mask=object_entry_mask,
        )
    else:
        _mark_legacy_span_roles_and_masks(
            assistant_token_span=assistant_token_span,
            object_entries=object_entries,
            structural_spans=structural_spans,
            separator_spans=separator_spans,
            token_roles=token_roles,
            assistant_mask=assistant_mask,
            object_entry_mask=object_entry_mask,
            desc_mask=desc_mask,
            bbox_start_mask=bbox_start_mask,
            bbox_mask=bbox_mask,
            coord_mask=coord_mask,
            separator_mask=separator_mask,
            control_mask=control_mask,
        )

    if terminal_span is not None:
        _mark_span(
            terminal_span,
            mask=terminal_mask,
            token_roles=token_roles,
            role=TokenRole.TERMINAL,
        )
    for stop_marker_span in stop_marker_spans:
        _mark_span(
            stop_marker_span,
            mask=terminal_mask,
            token_roles=token_roles,
            role=TokenRole.TERMINAL,
        )

    return _MasksAndRoles(
        token_roles=tuple(token_roles),
        assistant_mask=tuple(assistant_mask),
        object_entry_mask=tuple(object_entry_mask),
        desc_mask=tuple(desc_mask),
        bbox_start_mask=tuple(bbox_start_mask),
        bbox_mask=tuple(bbox_mask),
        coord_mask=tuple(coord_mask),
        separator_mask=tuple(separator_mask),
        terminal_mask=tuple(terminal_mask),
        control_mask=tuple(control_mask),
    )


def _mark_legacy_span_roles_and_masks(
    *,
    assistant_token_span: TokenSpan,
    object_entries: Sequence[TokenizedObjectEntry],
    structural_spans: Sequence[TokenSpan],
    separator_spans: Sequence[TokenSpan],
    token_roles: list[TokenRole],
    assistant_mask: list[bool],
    object_entry_mask: list[bool],
    desc_mask: list[bool],
    bbox_start_mask: list[bool],
    bbox_mask: list[bool],
    coord_mask: list[bool],
    separator_mask: list[bool],
    control_mask: list[bool],
) -> None:
    _mark_span(
        assistant_token_span,
        mask=assistant_mask,
        token_roles=token_roles,
        role=TokenRole.ASSISTANT,
    )
    for entry in object_entries:
        _mark_span(
            entry.entry_span,
            mask=object_entry_mask,
            token_roles=token_roles,
            role=TokenRole.OBJECT_ENTRY,
        )
        _mark_span(
            entry.bbox_span,
            mask=bbox_mask,
            token_roles=token_roles,
            role=None,
        )
        _mark_span(
            entry.desc_span,
            mask=desc_mask,
            token_roles=token_roles,
            role=TokenRole.DESC,
        )
        _mark_span(
            entry.bbox_start_span,
            mask=bbox_start_mask,
            token_roles=token_roles,
            role=TokenRole.BBOX_START,
        )
        for coord_span in entry.coord_spans:
            _mark_span(
                coord_span,
                mask=coord_mask,
                token_roles=token_roles,
                role=TokenRole.COORD,
            )
        for control_span in entry.control_spans:
            _mark_span(
                control_span,
                mask=control_mask,
                token_roles=token_roles,
                role=(
                    TokenRole.CONTROL
                    if control_span.label == "object_ref_start"
                    else None
                ),
            )

    for structural_span in structural_spans:
        _mark_span(
            structural_span,
            mask=control_mask,
            token_roles=token_roles,
            role=None,
        )
    for separator_span in separator_spans:
        _mark_span(
            separator_span,
            mask=separator_mask,
            token_roles=token_roles,
            role=TokenRole.SEPARATOR,
        )


def _align_render_span_events(
    render_span_events: Sequence[RenderSpanEvent],
    *,
    offsets: Sequence[tuple[int, int]],
    assistant_base_offset: int,
) -> tuple[_AlignedRenderSpanEvent, ...]:
    aligned_events: list[_AlignedRenderSpanEvent] = []
    for event in render_span_events:
        primary_role = _token_role_from_render_role(event.primary_role)
        if event.char_span.start == event.char_span.end:
            continue
        aligned_events.append(
            _AlignedRenderSpanEvent(
                event=event,
                token_span=align_char_span_to_token_span(
                    offsets,
                    event.char_span,
                    base_offset=assistant_base_offset,
                ),
                primary_role=primary_role,
            )
        )
    return tuple(aligned_events)


def _token_role_from_render_role(role_name: str | None) -> TokenRole | None:
    if role_name is None:
        return None
    try:
        return TokenRole[role_name]
    except KeyError as exc:
        raise ValueError(f"unknown render span primary role {role_name!r}") from exc


def _project_render_span_events(
    aligned_events: Sequence[_AlignedRenderSpanEvent],
    *,
    token_roles: list[TokenRole],
    assistant_mask: list[bool],
    object_entry_mask: list[bool],
    desc_mask: list[bool],
    bbox_start_mask: list[bool],
    bbox_mask: list[bool],
    coord_mask: list[bool],
    separator_mask: list[bool],
    terminal_mask: list[bool],
    control_mask: list[bool],
) -> None:
    role_priorities: list[int | None] = [None for _ in token_roles]
    classifying_priorities: list[set[int]] = [set() for _ in token_roles]
    for aligned_event in aligned_events:
        _mark_render_event_mask_groups(
            aligned_event,
            assistant_mask=assistant_mask,
            object_entry_mask=object_entry_mask,
            desc_mask=desc_mask,
            bbox_start_mask=bbox_start_mask,
            bbox_mask=bbox_mask,
            coord_mask=coord_mask,
            separator_mask=separator_mask,
            terminal_mask=terminal_mask,
            control_mask=control_mask,
        )
        if aligned_event.event.classifying:
            _project_classifying_render_event(
                aligned_event,
                token_roles=token_roles,
                role_priorities=role_priorities,
                classifying_priorities=classifying_priorities,
            )


def _mark_render_event_mask_groups(
    aligned_event: _AlignedRenderSpanEvent,
    *,
    assistant_mask: list[bool],
    object_entry_mask: list[bool],
    desc_mask: list[bool],
    bbox_start_mask: list[bool],
    bbox_mask: list[bool],
    coord_mask: list[bool],
    separator_mask: list[bool],
    terminal_mask: list[bool],
    control_mask: list[bool],
) -> None:
    for mask_group in aligned_event.event.mask_groups:
        if mask_group == "assistant":
            _mark_mask_only(aligned_event.token_span, assistant_mask)
        elif mask_group == "object_entry":
            _mark_mask_only(aligned_event.token_span, object_entry_mask)
        elif mask_group in {"desc", "description"}:
            _mark_mask_only(aligned_event.token_span, desc_mask)
        elif mask_group == "bbox_start":
            _mark_mask_only(aligned_event.token_span, bbox_start_mask)
        elif mask_group in {"coord", "coordinate"}:
            _mark_mask_only(aligned_event.token_span, coord_mask)
        elif mask_group == "bbox":
            _mark_mask_only(aligned_event.token_span, bbox_mask)
        elif mask_group == "separator":
            _mark_mask_only(aligned_event.token_span, separator_mask)
        elif mask_group == "terminal":
            _mark_mask_only(aligned_event.token_span, terminal_mask)
        elif mask_group in {"control", "schema"}:
            # Separator render events carry broad control provenance, but the
            # tokenized view preserves the historical mask contract: separators
            # belong to separator_mask and stay out of control_mask.
            if aligned_event.primary_role is not TokenRole.SEPARATOR:
                _mark_mask_only(aligned_event.token_span, control_mask)
        elif mask_group == "ignore":
            continue
        else:
            raise ValueError(
                f"unknown render span mask group {mask_group!r} for "
                f"{aligned_event.event.span_kind!r}"
            )


def _project_classifying_render_event(
    aligned_event: _AlignedRenderSpanEvent,
    *,
    token_roles: list[TokenRole],
    role_priorities: list[int | None],
    classifying_priorities: list[set[int]],
) -> None:
    if aligned_event.primary_role is None:
        raise ValueError(
            f"classifying render span {aligned_event.event.span_kind!r} "
            "is missing primary_role"
        )

    for token_index in aligned_event.token_span.token_indices():
        seen_priorities = classifying_priorities[token_index]
        if aligned_event.event.priority in seen_priorities:
            raise ValueError(
                "render span events have equal-priority classifying token overlap: "
                f"{aligned_event.event.span_kind}/{aligned_event.event.primary_role}@"
                f"{aligned_event.token_span.start}:{aligned_event.token_span.end}"
            )
        seen_priorities.add(aligned_event.event.priority)

        existing_priority = role_priorities[token_index]
        if existing_priority is None or aligned_event.event.priority > existing_priority:
            token_roles[token_index] = aligned_event.primary_role
            role_priorities[token_index] = aligned_event.event.priority


def _mark_object_projection_masks(
    object_entries: Sequence[TokenizedObjectEntry],
    *,
    bbox_mask: list[bool],
    bbox_start_mask: list[bool],
) -> None:
    # Compatibility overlay during the render-event migration: event mode owns
    # primary role and mask projection, while public object spans still reinforce
    # bbox/bbox-start masks so legacy downstream tokenized-view contracts remain
    # stable if an event omits those migration-era mask groups.
    for entry in object_entries:
        _mark_mask_only(entry.bbox_span, bbox_mask)
        _mark_mask_only(entry.bbox_start_span, bbox_start_mask)


def _mark_container_projection_masks(
    assistant_token_span: TokenSpan,
    object_entries: Sequence[TokenizedObjectEntry],
    *,
    assistant_mask: list[bool],
    object_entry_mask: list[bool],
) -> None:
    # Compatibility overlay during the render-event migration: public assistant
    # and object-entry spans continue to reinforce container masks so legacy
    # downstream tokenized-view contracts stay stable while events become the
    # source of truth for primary roles and fine-grained masks.
    _mark_mask_only(assistant_token_span, assistant_mask)
    for entry in object_entries:
        _mark_mask_only(entry.entry_span, object_entry_mask)


def _apply_container_role_fallbacks(
    token_roles: list[TokenRole],
    *,
    assistant_mask: Sequence[bool],
    object_entry_mask: Sequence[bool],
) -> None:
    for token_index, is_object_entry_token in enumerate(object_entry_mask):
        if is_object_entry_token and token_roles[token_index] is TokenRole.IGNORE:
            token_roles[token_index] = TokenRole.OBJECT_ENTRY
    for token_index, is_assistant_token in enumerate(assistant_mask):
        if is_assistant_token and token_roles[token_index] is TokenRole.IGNORE:
            token_roles[token_index] = TokenRole.ASSISTANT


def _mark_mask_only(span: TokenSpan, mask: list[bool]) -> None:
    for token_index in span.token_indices():
        mask[token_index] = True


def _mark_span(
    span: TokenSpan,
    *,
    mask: list[bool],
    token_roles: list[TokenRole],
    role: TokenRole | None,
) -> None:
    for token_index in span.token_indices():
        mask[token_index] = True
        if role is not None:
            token_roles[token_index] = role


def _as_flat_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"tokenizer output is missing {field_name}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"{field_name} must be a single sequence")
        value = value[0]
    return tuple(int(item) for item in value)


def _as_offset_tuple(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None:
        raise ValueError("tokenizer output is missing offset_mapping")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("offset_mapping must be a sequence of offset pairs")
    if not value:
        return ()

    if _is_offset_pair(value[0]):
        offset_sequence = value
    elif isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("offset_mapping must be a single sequence")
        offset_sequence = value[0]
        if not isinstance(offset_sequence, (list, tuple)):
            raise ValueError("offset_mapping must be a sequence of offset pairs")
    else:
        raise ValueError("offset_mapping entries must be pairs of integer offsets")

    return tuple(_as_single_offset_pair(offset) for offset in offset_sequence)


def _is_offset_pair(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    start, end = value
    return _is_plain_int(start) and _is_plain_int(end)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_single_offset_pair(value: object) -> tuple[int, int]:
    if not _is_offset_pair(value):
        raise ValueError("offset_mapping entries must be pairs of integer offsets")
    start, end = value
    return start, end


__all__ = [
    "TokenRole",
    "TokenSpan",
    "TokenizedDetectionExample",
    "TokenizedObjectEntry",
    "align_char_span_to_token_span",
    "tokenize_rendered_detection_conversation",
]
