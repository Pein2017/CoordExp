"""Detection assistant-sequence templates and assistant-local spans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from src.common.detection_compact_rows import (
    BOX_START_TOKEN,
    COMPACT_ROW_COORD_TOKEN_RE,
    OBJECT_REF_START_TOKEN,
    parse_compact_row,
    render_compact_row,
)
from src.detection.data import (
    CoordinateTokenBox,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
)
from src.utils.assistant_json import dumps_coordjson


TemplateId = Literal["stage1_json_pretty", "compact_full"]
TokenRoleName = Literal[
    "IGNORE",
    "ASSISTANT",
    "OBJECT_ENTRY",
    "DESC",
    "BBOX_START",
    "COORD",
    "SEPARATOR",
    "TERMINAL",
    "CONTROL",
]
# Terminal taxonomy contract for Task 4:
# - `terminal_close` is the assistant-rendered terminal close owned by templates.
# - `compact_full` has no rendered terminal bytes, so it emits a zero-length
#   `terminal_close` provenance event at the assistant text boundary.
# - `chat_stop_marker` is reserved for chat-template/tokenization-level terminal
#   projection and is not emitted by template-only render paths in this slice.
# - Current tokenization intentionally collapses Qwen chat stop markers and
#   tokenizer EOS candidates into the existing terminal projection. Richer
#   encoded-view terminal events are deferred to Task 6 or later.
SpanKind = Literal[
    "description_text",
    "coordinate_slot",
    "object_ref_marker",
    "bbox_start_marker",
    "bbox_field_binding",
    "json_key",
    "json_punctuation",
    "object_separator",
    "coordinate_separator",
    "terminal_close",
    "chat_stop_marker",
    "assistant_container",
    "object_entry_container",
]
MaskGroup = Literal[
    "assistant",
    "object_entry",
    "desc",
    "bbox",
    "coord",
    "schema",
    "control",
    "separator",
    "terminal",
    "ignore",
]
SpanProvenance = Literal[
    "assistant_projection",
    "object_entry_projection",
    "rendered_leaf",
    "rendered_control",
    "terminal_projection",
]

_COORD_TOKEN_RE = COMPACT_ROW_COORD_TOKEN_RE
_COMPACT_FORBIDDEN_DESC_SUBSTRINGS = (
    "\n",
    "\r",
    "\t",
    OBJECT_REF_START_TOKEN,
    BOX_START_TOKEN,
    "<|coord_",
    "<|im_start|>",
    "<|im_end|>",
)
_COORD_SLOT_NAMES = ("x1", "y1", "x2", "y2")
_ROLE_PRIORITIES: Mapping[TokenRoleName, int] = {
    "COORD": 100,
    "DESC": 90,
    "TERMINAL": 80,
    "BBOX_START": 70,
    "SEPARATOR": 60,
    "CONTROL": 50,
    "OBJECT_ENTRY": 20,
    "ASSISTANT": 10,
    "IGNORE": 0,
}


@dataclass(frozen=True)
class TemplateCapabilities:
    template_id: TemplateId
    version: int
    coordinate_surface: Literal["coord_token"]
    bbox_format: Literal["xyxy"]
    object_field_order: Literal["desc_first", "compact_full_row"]
    object_separator: str
    terminal_close: str
    supports_sft: bool = True
    supports_random_permutation_et_rmp_ce: bool = True
    supports_recursive_detection_ce: bool = True
    supports_et_rmp_ce: bool = True
    supports_static_packing: bool = True
    geometry_kinds: tuple[str, ...] = ("bbox_2d",)


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    label: str

    def text(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True)
class RenderSpanEvent:
    """Template-local semantic event before token-role projection.

    `primary_role` intentionally uses uppercase role-name strings instead of
    importing `TokenRole` from `src.detection.tokenization`; tokenization already
    imports this module, so a runtime enum import here would create a cycle. The
    values match `TokenRole.name` and are kept projection-ready for Task 6.
    """

    char_span: CharSpan
    span_kind: SpanKind
    primary_role: TokenRoleName | None
    mask_groups: frozenset[MaskGroup]
    classifying: bool
    priority: int
    object_instance_id: str | None = None
    object_index: int | None = None
    source_object_index: int | None = None
    object_id: str | None = None
    supervision_key: str | None = None
    span_family: str | None = None
    field_name: str | None = None
    source_role: str | None = None
    relation_snapshot: Mapping[str, Any] | None = None
    coordinate_weight: float | None = None
    regression_weight: float | None = None
    hard_bbox_supervision: bool | None = None
    geometry_kind: str | None = None
    slot_name: str | None = None
    provenance: SpanProvenance | None = None


@dataclass(frozen=True)
class RenderedObjectEntry:
    object_instance_id: str
    object_index: int
    source_object_index: int
    entry_span: CharSpan
    desc_span: CharSpan
    object_ref_start_span: CharSpan | None
    bbox_start_span: CharSpan
    bbox_span: CharSpan
    coord_spans: tuple[CharSpan, ...]
    separator_span: CharSpan | None
    control_spans: tuple[CharSpan, ...]
    trie_eligible_span: CharSpan
    object_id: str | None = None
    source_role: str | None = None
    relation_snapshot: Mapping[str, Any] | None = None
    coordinate_weight: float | None = None
    regression_weight: float | None = None
    hard_bbox_supervision: bool | None = None

    @property
    def bbox_opener_span(self) -> CharSpan:
        return self.bbox_start_span

    @property
    def coordinate_spans(self) -> tuple[CharSpan, ...]:
        return self.coord_spans

    @property
    def structural_token_spans(self) -> tuple[CharSpan, ...]:
        return self.control_spans


@dataclass(frozen=True)
class RenderedAssistantSequence:
    template_id: TemplateId
    template_version: int
    text: str
    object_entries: tuple[RenderedObjectEntry, ...]
    separator_spans: tuple[CharSpan, ...]
    terminal_close_span: CharSpan
    stop_marker_spans: tuple[CharSpan, ...]
    structural_token_spans: tuple[CharSpan, ...]
    trie_eligible_spans: tuple[CharSpan, ...]
    render_span_events: tuple[RenderSpanEvent, ...] = ()


@dataclass(frozen=True)
class RenderedConversation:
    assistant: RenderedAssistantSequence
    messages: tuple[Mapping[str, str], ...]


@runtime_checkable
class DetectionSequenceTemplate(Protocol):
    template_id: TemplateId
    capabilities: TemplateCapabilities

    def validate_sample(
        self,
        sample: NormalizedDetectionSample,
        *,
        coordinate_surface: str = "coord_token",
        bbox_format: str = "xyxy",
        prompt_template_id: str | None = None,
    ) -> None: ...

    def render_assistant(
        self,
        sample: NormalizedDetectionSample,
        *,
        coordinate_surface: str = "coord_token",
        bbox_format: str = "xyxy",
        prompt_template_id: str | None = None,
    ) -> RenderedAssistantSequence: ...

    def parse_assistant(self, text: str) -> dict[str, Any]: ...

    def render_entry(self, obj: NormalizedDetectionObject) -> str: ...

    def render_separator(self, before_index: int, after_index: int) -> str: ...

    def render_terminal_close(self) -> str: ...


class Stage1JsonPrettyTemplate:
    template_id: TemplateId = "stage1_json_pretty"
    capabilities = TemplateCapabilities(
        template_id="stage1_json_pretty",
        version=1,
        coordinate_surface="coord_token",
        bbox_format="xyxy",
        object_field_order="desc_first",
        object_separator=", ",
        terminal_close="]}",
    )

    def validate_sample(
        self,
        sample: NormalizedDetectionSample,
        *,
        coordinate_surface: str = "coord_token",
        bbox_format: str = "xyxy",
        prompt_template_id: str | None = None,
    ) -> None:
        _validate_common_surface(
            self,
            sample,
            coordinate_surface=coordinate_surface,
            bbox_format=bbox_format,
            prompt_template_id=prompt_template_id,
        )
        for obj in sample.objects:
            _validate_stage1_json_desc(obj.desc)

    def render_assistant(
        self,
        sample: NormalizedDetectionSample,
        *,
        coordinate_surface: str = "coord_token",
        bbox_format: str = "xyxy",
        prompt_template_id: str | None = None,
    ) -> RenderedAssistantSequence:
        self.validate_sample(
            sample,
            coordinate_surface=coordinate_surface,
            bbox_format=bbox_format,
            prompt_template_id=prompt_template_id,
        )

        builder = _SpanTextBuilder()
        event_builder = _RenderEventBuilder()
        structural_spans: list[CharSpan] = []
        separator_spans: list[CharSpan] = []
        entries: list[RenderedObjectEntry] = []

        json_root_open = builder.append("{", "json_root_open")
        structural_spans.append(json_root_open)
        event_builder.append(_json_punctuation_event(json_root_open))
        objects_key_span = builder.append('"objects": ', "objects_key")
        structural_spans.append(objects_key_span)
        event_builder.append(_json_key_event(objects_key_span))
        json_array_open = builder.append("[", "json_array_open")
        structural_spans.append(json_array_open)
        event_builder.append(_json_punctuation_event(json_array_open))
        for object_index, obj in enumerate(sample.objects):
            if object_index:
                separator_span = builder.append(
                    self.render_separator(object_index - 1, object_index),
                    "object_separator",
                )
                separator_spans.append(separator_span)
                structural_spans.append(separator_span)
                event_builder.append_object_separator(
                    separator_span,
                    previous_entry=entries[-1],
                )
                entries[-1] = replace(entries[-1], separator_span=separator_span)
            entry = _append_stage1_json_entry(builder, obj, object_index)
            entries.append(entry)
            structural_spans.extend(entry.control_spans)
            event_builder.extend(_stage1_json_entry_events(entry))

        terminal_start = len(builder)
        json_array_close = builder.append("]", "json_array_close")
        structural_spans.append(json_array_close)
        event_builder.append(_json_punctuation_event(json_array_close))
        json_root_close = builder.append("}", "json_root_close")
        structural_spans.append(json_root_close)
        event_builder.append(_json_punctuation_event(json_root_close))
        terminal_close_span = CharSpan(
            terminal_start,
            len(builder),
            "terminal_close",
        )
        event_builder.append_terminal_close(terminal_close_span)

        text = builder.text
        _assert_stage1_json_fixture_parity(text, sample)
        return _project_rendered_assistant_sequence(
            template_id=self.template_id,
            template_version=self.capabilities.version,
            text=text,
            object_entries=entries,
            separator_spans=separator_spans,
            terminal_close_span=terminal_close_span,
            structural_token_spans=structural_spans,
            event_builder=event_builder,
        )

    def parse_assistant(self, text: str) -> dict[str, Any]:
        if not text.endswith(self.render_terminal_close()):
            raise ValueError("stage1_json_pretty text must preserve strict terminal closure")

        payload = _loads_coordjson_with_bare_coord_tokens(text)
        _validate_stage1_json_payload(payload)
        if dumps_coordjson(payload) != text:
            raise ValueError("text is not canonical stage1_json_pretty")
        return payload

    def render_entry(self, obj: NormalizedDetectionObject) -> str:
        return dumps_coordjson({"objects": [_object_payload(obj)]})[
            len('{"objects": [') : -len("]}")
        ]

    def render_separator(self, before_index: int, after_index: int) -> str:
        del before_index, after_index
        return self.capabilities.object_separator

    def render_terminal_close(self) -> str:
        return self.capabilities.terminal_close


class CompactFullTemplate:
    """First-class strict compact training/eval template.

    The common compact facade keeps compatibility behavior for generated text
    repair, suffix stripping, and ``None`` diagnostics.  This template owns the
    strict ``compact_full`` training target surface and intentionally does not
    delegate parsing to the common compatibility parser.
    """

    template_id: TemplateId = "compact_full"
    capabilities = TemplateCapabilities(
        template_id="compact_full",
        version=1,
        coordinate_surface="coord_token",
        bbox_format="xyxy",
        object_field_order="compact_full_row",
        object_separator="\n",
        terminal_close="",
    )

    def validate_sample(
        self,
        sample: NormalizedDetectionSample,
        *,
        coordinate_surface: str = "coord_token",
        bbox_format: str = "xyxy",
        prompt_template_id: str | None = None,
    ) -> None:
        _validate_common_surface(
            self,
            sample,
            coordinate_surface=coordinate_surface,
            bbox_format=bbox_format,
            prompt_template_id=prompt_template_id,
        )
        for obj in sample.objects:
            _validate_compact_desc(obj.desc)

    def render_assistant(
        self,
        sample: NormalizedDetectionSample,
        *,
        coordinate_surface: str = "coord_token",
        bbox_format: str = "xyxy",
        prompt_template_id: str | None = None,
    ) -> RenderedAssistantSequence:
        self.validate_sample(
            sample,
            coordinate_surface=coordinate_surface,
            bbox_format=bbox_format,
            prompt_template_id=prompt_template_id,
        )

        builder = _SpanTextBuilder()
        event_builder = _RenderEventBuilder()
        separator_spans: list[CharSpan] = []
        entries: list[RenderedObjectEntry] = []
        structural_spans: list[CharSpan] = []

        for object_index, obj in enumerate(sample.objects):
            if object_index:
                separator_span = builder.append(
                    self.render_separator(object_index - 1, object_index),
                    "object_separator",
                )
                separator_spans.append(separator_span)
                structural_spans.append(separator_span)
                event_builder.append_object_separator(
                    separator_span,
                    previous_entry=entries[-1],
                )
                entries[-1] = replace(entries[-1], separator_span=separator_span)
            entry = _append_compact_full_entry(builder, obj, object_index)
            entries.append(entry)
            structural_spans.extend(entry.control_spans)
            event_builder.extend(_compact_full_entry_events(entry))

        terminal_close_span = CharSpan(len(builder), len(builder), "terminal_close")
        event_builder.append_terminal_close(terminal_close_span)
        text = builder.text
        return _project_rendered_assistant_sequence(
            template_id=self.template_id,
            template_version=self.capabilities.version,
            text=text,
            object_entries=entries,
            separator_spans=separator_spans,
            terminal_close_span=terminal_close_span,
            structural_token_spans=structural_spans,
            event_builder=event_builder,
        )

    def parse_assistant(self, text: str) -> dict[str, Any]:
        if not text:
            return {"objects": []}
        if text.endswith("\n") or "\n\n" in text:
            raise ValueError("text is not strict compact_full")

        objects: list[dict[str, Any]] = []
        for row in text.split("\n"):
            objects.append(_parse_compact_full_row(row))
        return {"objects": objects}

    def render_entry(self, obj: NormalizedDetectionObject) -> str:
        _validate_compact_desc(obj.desc)
        return render_compact_row(
            obj.desc,
            _render_bbox_coord_tokens(obj.bbox_2d),
            include_object_ref_marker=True,
            include_bbox_start_marker=True,
        )

    def render_separator(self, before_index: int, after_index: int) -> str:
        del before_index, after_index
        return self.capabilities.object_separator

    def render_terminal_close(self) -> str:
        return self.capabilities.terminal_close


class _SpanTextBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0

    def __len__(self) -> int:
        return self._length

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def append(self, value: str, label: str) -> CharSpan:
        span = CharSpan(self._length, self._length + len(value), label)
        self._parts.append(value)
        self._length = span.end
        return span


class _RenderEventBuilder:
    """Collect render events before projecting compatibility fields."""

    def __init__(self) -> None:
        self._events: list[RenderSpanEvent] = []

    def append(self, event: RenderSpanEvent) -> None:
        self._events.append(event)

    def extend(self, events: tuple[RenderSpanEvent, ...]) -> None:
        self._events.extend(events)

    def append_event(
        self,
        char_span: CharSpan,
        *,
        span_kind: SpanKind,
        primary_role: TokenRoleName | None,
        mask_groups: tuple[MaskGroup, ...],
        classifying: bool,
        object_entry: RenderedObjectEntry | None = None,
        span_family: str | None = None,
        field_name: str | None = None,
        geometry_kind: str | None = None,
        slot_name: str | None = None,
        provenance: SpanProvenance | None = None,
        priority: int | None = None,
    ) -> None:
        self.append(
            _render_event(
                char_span,
                span_kind=span_kind,
                primary_role=primary_role,
                mask_groups=mask_groups,
                classifying=classifying,
                object_entry=object_entry,
                span_family=span_family,
                field_name=field_name,
                geometry_kind=geometry_kind,
                slot_name=slot_name,
                provenance=provenance,
                priority=priority,
            )
        )

    def append_object_separator(
        self,
        separator_span: CharSpan,
        *,
        previous_entry: RenderedObjectEntry,
    ) -> None:
        # The separator span is sequence-level/interstitial, but its metadata
        # follows the legacy compatibility projection that attaches it to the
        # previous rendered object entry.
        self.append_event(
            separator_span,
            span_kind="object_separator",
            primary_role="SEPARATOR",
            mask_groups=("separator", "control"),
            classifying=True,
            object_entry=previous_entry,
            provenance="rendered_control",
        )

    def append_terminal_close(self, terminal_close_span: CharSpan) -> None:
        self.append(_terminal_close_event(terminal_close_span))

    def finalize(self, text: str) -> tuple[RenderSpanEvent, ...]:
        return _finalize_render_span_events(
            (_assistant_container_event(text), *self._events)
        )


def get_detection_template(template_id: TemplateId | str) -> DetectionSequenceTemplate:
    if template_id == "stage1_json_pretty":
        return Stage1JsonPrettyTemplate()
    if template_id == "compact_full":
        return CompactFullTemplate()
    raise ValueError(f"Unsupported detection template: {template_id!r}")


def _append_stage1_json_entry(
    builder: _SpanTextBuilder, obj: NormalizedDetectionObject, object_index: int
) -> RenderedObjectEntry:
    entry_start = len(builder)
    structural_spans: list[CharSpan] = []

    structural_spans.append(builder.append("{", "object_open"))
    structural_spans.append(builder.append('"desc": ', "desc_key"))
    desc_json = json.dumps(obj.desc, ensure_ascii=False)
    desc_json_span = builder.append(desc_json, "desc_json")
    desc_span = CharSpan(
        desc_json_span.start + 1,
        desc_json_span.end - 1,
        "desc",
    )
    structural_spans.append(builder.append(", ", "field_separator"))
    bbox_start_span = builder.append('"bbox_2d": [', "bbox_start")
    structural_spans.append(bbox_start_span)

    coordinate_spans: list[CharSpan] = []
    for coord_index, token in enumerate(_render_bbox_coord_tokens(obj.bbox_2d)):
        if coord_index:
            structural_spans.append(builder.append(", ", "coordinate_separator"))
        coordinate_spans.append(builder.append(token, f"coord_{coord_index}"))
    structural_spans.append(builder.append("]", "bbox_close"))
    structural_spans.append(builder.append("}", "object_close"))

    entry_span = CharSpan(entry_start, len(builder), "object_entry")
    bbox_span = CharSpan(
        bbox_start_span.end,
        structural_spans[-2].start,
        "bbox",
    )
    return RenderedObjectEntry(
        object_instance_id=obj.object_instance_id,
        object_index=object_index,
        source_object_index=obj.source_object_index,
        entry_span=entry_span,
        desc_span=desc_span,
        object_ref_start_span=None,
        bbox_start_span=bbox_start_span,
        bbox_span=bbox_span,
        coord_spans=tuple(coordinate_spans),
        separator_span=None,
        control_spans=tuple(structural_spans),
        trie_eligible_span=entry_span,
        object_id=obj.object_id,
        source_role=obj.source_role,
        relation_snapshot=obj.relation_snapshot,
        coordinate_weight=_metadata_weight(obj.relation_snapshot, "coordinate_weight"),
        regression_weight=_metadata_weight(obj.relation_snapshot, "regression_weight"),
        hard_bbox_supervision=_hard_bbox_supervision(
            source_role=obj.source_role,
            relation_snapshot=obj.relation_snapshot,
        ),
    )


def _append_compact_full_entry(
    builder: _SpanTextBuilder, obj: NormalizedDetectionObject, object_index: int
) -> RenderedObjectEntry:
    entry_start = len(builder)
    object_ref_span = builder.append(OBJECT_REF_START_TOKEN, "object_ref_start")
    desc_span = builder.append(obj.desc, "desc")
    bbox_start_span = builder.append(BOX_START_TOKEN, "bbox_start")

    coordinate_spans = tuple(
        builder.append(token, f"coord_{coord_index}")
        for coord_index, token in enumerate(_render_bbox_coord_tokens(obj.bbox_2d))
    )
    entry_span = CharSpan(entry_start, len(builder), "object_entry")
    bbox_span = CharSpan(bbox_start_span.end, len(builder), "bbox")
    return RenderedObjectEntry(
        object_instance_id=obj.object_instance_id,
        object_index=object_index,
        source_object_index=obj.source_object_index,
        entry_span=entry_span,
        desc_span=desc_span,
        object_ref_start_span=object_ref_span,
        bbox_start_span=bbox_start_span,
        bbox_span=bbox_span,
        coord_spans=coordinate_spans,
        separator_span=None,
        control_spans=(object_ref_span, bbox_start_span),
        trie_eligible_span=entry_span,
        object_id=obj.object_id,
        source_role=obj.source_role,
        relation_snapshot=obj.relation_snapshot,
        coordinate_weight=_metadata_weight(obj.relation_snapshot, "coordinate_weight"),
        regression_weight=_metadata_weight(obj.relation_snapshot, "regression_weight"),
        hard_bbox_supervision=_hard_bbox_supervision(
            source_role=obj.source_role,
            relation_snapshot=obj.relation_snapshot,
        ),
    )


def _metadata_weight(
    relation_snapshot: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    if relation_snapshot is None or key not in relation_snapshot:
        return None
    value = relation_snapshot[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"relation snapshot {key} must be numeric when present")
    return float(value)


def _hard_bbox_supervision(
    *,
    source_role: str | None,
    relation_snapshot: Mapping[str, Any] | None,
) -> bool | None:
    coordinate_weight = _metadata_weight(relation_snapshot, "coordinate_weight")
    regression_weight = _metadata_weight(relation_snapshot, "regression_weight")
    if source_role == "proxy_candidate":
        return False
    if coordinate_weight == 0.0 or regression_weight == 0.0:
        return False
    if coordinate_weight is None and regression_weight is None:
        return None
    return True


def _render_event(
    char_span: CharSpan,
    *,
    span_kind: SpanKind,
    primary_role: TokenRoleName | None,
    mask_groups: tuple[MaskGroup, ...],
    classifying: bool,
    object_entry: RenderedObjectEntry | None = None,
    span_family: str | None = None,
    field_name: str | None = None,
    geometry_kind: str | None = None,
    slot_name: str | None = None,
    provenance: SpanProvenance | None = None,
    priority: int | None = None,
) -> RenderSpanEvent:
    resolved_priority = (
        priority
        if priority is not None
        else (_ROLE_PRIORITIES[primary_role] if primary_role is not None else 0)
    )
    return RenderSpanEvent(
        char_span=char_span,
        span_kind=span_kind,
        primary_role=primary_role,
        mask_groups=frozenset(mask_groups),
        classifying=classifying,
        priority=resolved_priority,
        object_instance_id=(
            None if object_entry is None else object_entry.object_instance_id
        ),
        object_index=None if object_entry is None else object_entry.object_index,
        source_object_index=(
            None if object_entry is None else object_entry.source_object_index
        ),
        object_id=None if object_entry is None else object_entry.object_id,
        supervision_key=None if object_entry is None else object_entry.object_id,
        span_family=span_family,
        field_name=field_name,
        source_role=None if object_entry is None else object_entry.source_role,
        relation_snapshot=(
            None if object_entry is None else object_entry.relation_snapshot
        ),
        coordinate_weight=(
            None if object_entry is None else object_entry.coordinate_weight
        ),
        regression_weight=(
            None if object_entry is None else object_entry.regression_weight
        ),
        hard_bbox_supervision=(
            None if object_entry is None else object_entry.hard_bbox_supervision
        ),
        geometry_kind=geometry_kind,
        slot_name=slot_name,
        provenance=provenance,
    )


def _assistant_container_event(text: str) -> RenderSpanEvent:
    return _render_event(
        CharSpan(0, len(text), "assistant"),
        span_kind="assistant_container",
        primary_role=None,
        mask_groups=("assistant",),
        classifying=False,
        provenance="assistant_projection",
        priority=_ROLE_PRIORITIES["ASSISTANT"],
    )


def _object_entry_container_event(entry: RenderedObjectEntry) -> RenderSpanEvent:
    return _render_event(
        entry.entry_span,
        span_kind="object_entry_container",
        primary_role=None,
        mask_groups=("object_entry",),
        classifying=False,
        object_entry=entry,
        provenance="object_entry_projection",
        priority=_ROLE_PRIORITIES["OBJECT_ENTRY"],
    )


def _terminal_close_event(span: CharSpan) -> RenderSpanEvent:
    return _render_event(
        span,
        span_kind="terminal_close",
        primary_role="TERMINAL",
        mask_groups=("terminal",),
        classifying=True,
        provenance="terminal_projection",
    )


def _json_key_event(span: CharSpan, *, entry: RenderedObjectEntry | None = None) -> RenderSpanEvent:
    return _render_event(
        span,
        span_kind="json_key",
        primary_role="CONTROL",
        mask_groups=("schema", "control"),
        classifying=True,
        object_entry=entry,
        provenance="rendered_control",
    )


def _json_punctuation_event(
    span: CharSpan, *, entry: RenderedObjectEntry | None = None
) -> RenderSpanEvent:
    return _render_event(
        span,
        span_kind="json_punctuation",
        primary_role="CONTROL",
        mask_groups=("schema", "control"),
        classifying=True,
        object_entry=entry,
        provenance="rendered_control",
    )


def _stage1_json_entry_events(entry: RenderedObjectEntry) -> tuple[RenderSpanEvent, ...]:
    events: list[RenderSpanEvent] = [_object_entry_container_event(entry)]

    for control_span in entry.control_spans:
        if control_span.label == "desc_key":
            events.append(_json_key_event(control_span, entry=entry))
        elif control_span.label == "bbox_start":
            events.append(
                _render_event(
                    control_span,
                    span_kind="bbox_field_binding",
                    primary_role="BBOX_START",
                    mask_groups=("schema", "control"),
                    classifying=True,
                    object_entry=entry,
                    span_family="geometry",
                    field_name="bbox_2d",
                    geometry_kind="bbox_2d",
                    provenance="rendered_control",
                )
            )
            events.append(_json_key_event(control_span, entry=entry))
        elif control_span.label == "coordinate_separator":
            events.append(
                _render_event(
                    control_span,
                    span_kind="coordinate_separator",
                    primary_role="SEPARATOR",
                    mask_groups=("separator", "control"),
                    classifying=True,
                    object_entry=entry,
                    geometry_kind="bbox_2d",
                    provenance="rendered_control",
                )
            )
        else:
            events.append(_json_punctuation_event(control_span, entry=entry))

    events.append(
        _render_event(
            entry.desc_span,
            span_kind="description_text",
            primary_role="DESC",
            mask_groups=("desc",),
            classifying=True,
            object_entry=entry,
            span_family="description",
            field_name="desc",
            provenance="rendered_leaf",
        )
    )
    events.append(
        _json_punctuation_event(
            CharSpan(entry.desc_span.start - 1, entry.desc_span.start, "desc_quote_open"),
            entry=entry,
        )
    )
    events.append(
        _json_punctuation_event(
            CharSpan(entry.desc_span.end, entry.desc_span.end + 1, "desc_quote_close"),
            entry=entry,
        )
    )
    events.extend(_coordinate_slot_events(entry))
    return tuple(events)


def _compact_full_entry_events(entry: RenderedObjectEntry) -> tuple[RenderSpanEvent, ...]:
    events: list[RenderSpanEvent] = [_object_entry_container_event(entry)]
    if entry.object_ref_start_span is not None:
        events.append(
            _render_event(
                entry.object_ref_start_span,
                span_kind="object_ref_marker",
                primary_role="CONTROL",
                mask_groups=("schema", "control"),
                classifying=True,
                object_entry=entry,
                provenance="rendered_control",
            )
        )
    events.append(
        _render_event(
            entry.desc_span,
            span_kind="description_text",
            primary_role="DESC",
            mask_groups=("desc",),
            classifying=True,
            object_entry=entry,
            span_family="description",
            field_name="desc",
            provenance="rendered_leaf",
        )
    )
    events.append(
        _render_event(
            entry.bbox_start_span,
            span_kind="bbox_start_marker",
            primary_role="BBOX_START",
            mask_groups=("schema", "control"),
            classifying=True,
            object_entry=entry,
            span_family="geometry",
            field_name="bbox_2d",
            geometry_kind="bbox_2d",
            provenance="rendered_control",
        )
    )
    events.extend(_coordinate_slot_events(entry))
    return tuple(events)


def _coordinate_slot_events(entry: RenderedObjectEntry) -> tuple[RenderSpanEvent, ...]:
    return tuple(
        _render_event(
            coord_span,
            span_kind="coordinate_slot",
            primary_role="COORD",
            mask_groups=("coord",),
            classifying=True,
            object_entry=entry,
            span_family="geometry",
            field_name="bbox_2d",
            geometry_kind="bbox_2d",
            slot_name=_COORD_SLOT_NAMES[coord_index],
            provenance="rendered_leaf",
        )
        for coord_index, coord_span in enumerate(entry.coord_spans)
    )




def _project_rendered_assistant_sequence(
    *,
    template_id: TemplateId,
    template_version: int,
    text: str,
    object_entries: tuple[RenderedObjectEntry, ...] | list[RenderedObjectEntry],
    separator_spans: tuple[CharSpan, ...] | list[CharSpan],
    terminal_close_span: CharSpan,
    structural_token_spans: tuple[CharSpan, ...] | list[CharSpan],
    event_builder: _RenderEventBuilder,
) -> RenderedAssistantSequence:
    entries = tuple(object_entries)
    return RenderedAssistantSequence(
        template_id=template_id,
        template_version=template_version,
        text=text,
        object_entries=entries,
        separator_spans=tuple(separator_spans),
        terminal_close_span=terminal_close_span,
        stop_marker_spans=(),
        structural_token_spans=tuple(structural_token_spans),
        trie_eligible_spans=tuple(entry.trie_eligible_span for entry in entries),
        render_span_events=event_builder.finalize(text),
    )

def _finalize_render_span_events(
    events: tuple[RenderSpanEvent, ...] | list[RenderSpanEvent],
) -> tuple[RenderSpanEvent, ...]:
    event_tuple = tuple(events)
    _assert_no_equal_priority_classifying_overlaps(event_tuple)
    return event_tuple


def _assert_no_equal_priority_classifying_overlaps(
    events: tuple[RenderSpanEvent, ...],
) -> None:
    for left_index, left in enumerate(events):
        if not _participates_in_overlap_invariant(left):
            continue
        for right in events[left_index + 1 :]:
            if not _participates_in_overlap_invariant(right):
                continue
            if left.priority != right.priority:
                continue
            if _char_spans_overlap(left.char_span, right.char_span):
                raise ValueError(
                    "render span events have equal-priority classifying overlap: "
                    f"{left.span_kind}/{left.primary_role}@"
                    f"{left.char_span.start}:{left.char_span.end} overlaps "
                    f"{right.span_kind}/{right.primary_role}@"
                    f"{right.char_span.start}:{right.char_span.end}"
                )


def _participates_in_overlap_invariant(event: RenderSpanEvent) -> bool:
    return event.classifying and event.char_span.start < event.char_span.end


def _char_spans_overlap(left: CharSpan, right: CharSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _object_payload(obj: NormalizedDetectionObject) -> dict[str, Any]:
    return {
        "desc": obj.desc,
        "bbox_2d": list(_render_bbox_coord_tokens(obj.bbox_2d)),
    }


def _payload(sample: NormalizedDetectionSample) -> dict[str, Any]:
    return {"objects": [_object_payload(obj) for obj in sample.objects]}


def _assert_stage1_json_fixture_parity(
    rendered_text: str, sample: NormalizedDetectionSample
) -> None:
    expected = dumps_coordjson(_payload(sample))
    if rendered_text != expected:
        raise AssertionError("stage1_json_pretty renderer diverged from dumps_coordjson")


def _validate_common_surface(
    template: DetectionSequenceTemplate,
    sample: NormalizedDetectionSample,
    *,
    coordinate_surface: str,
    bbox_format: str,
    prompt_template_id: str | None,
) -> None:
    if coordinate_surface != template.capabilities.coordinate_surface:
        raise ValueError(
            f"{template.template_id} requires coordinate_surface="
            f"{template.capabilities.coordinate_surface}"
        )
    if bbox_format != template.capabilities.bbox_format:
        raise ValueError(
            f"{template.template_id} requires bbox_format={template.capabilities.bbox_format}"
        )
    if prompt_template_id is not None and prompt_template_id != template.template_id:
        raise ValueError(
            f"prompt template mismatch: {prompt_template_id!r} != {template.template_id!r}"
        )
    for obj in sample.objects:
        _validate_bbox_tokens(obj)


def _validate_bbox_tokens(obj: NormalizedDetectionObject) -> None:
    for token in _render_bbox_coord_tokens(obj.bbox_2d):
        if not _is_strict_coord_token(token):
            raise ValueError(
                f"object {obj.object_instance_id} must use coord-token bbox_2d values"
            )


def _render_bbox_coord_tokens(bbox_2d: CoordinateTokenBox) -> tuple[str, str, str, str]:
    """Return the coord-token render surface for legacy and norm1000 boxes."""

    return bbox_2d.tokens


def _is_strict_coord_token(token: object) -> bool:
    if not isinstance(token, str):
        return False
    match = _COORD_TOKEN_RE.fullmatch(token)
    if match is None:
        return False
    value_text = token.removeprefix("<|coord_").removesuffix("|>")
    value = int(value_text)
    return str(value) == value_text and 0 <= value <= 999


def _validate_strict_coord_tokens(tokens: list[Any], *, context: str) -> None:
    if not all(_is_strict_coord_token(token) for token in tokens):
        raise ValueError(f"{context} must use strict compact-v1 coord tokens")


def _validate_stage1_json_desc(desc: str) -> None:
    if json.dumps(desc, ensure_ascii=False)[1:-1] != desc:
        raise ValueError(
            "stage1_json_pretty desc must not contain JSON-escaped characters"
        )


def _validate_compact_desc(desc: str) -> None:
    if not desc.strip():
        raise ValueError("compact_full desc must be non-empty")
    for forbidden in _COMPACT_FORBIDDEN_DESC_SUBSTRINGS:
        if forbidden in desc:
            raise ValueError(f"compact_full desc contains forbidden marker {forbidden!r}")


def _parse_compact_full_row(row: str) -> dict[str, Any]:
    parts = parse_compact_row(
        row,
        require_object_ref_marker=True,
        require_bbox_start_marker=True,
        bbox_marker_split="first",
    )
    if parts is None:
        raise ValueError("text is not strict compact_full")
    _validate_compact_desc(parts.desc)
    coords = list(parts.bbox_tokens)
    _validate_strict_coord_tokens(coords, context="strict compact_full bbox_2d")
    return {"desc": parts.desc, "bbox_2d": coords}


def _loads_coordjson_with_bare_coord_tokens(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(_quote_bare_coord_tokens(text))
    except json.JSONDecodeError as exc:
        if not _has_balanced_json_delimiters(text):
            raise ValueError(
                "stage1_json_pretty text must preserve strict terminal closure"
            ) from exc
        raise ValueError("text is not canonical stage1_json_pretty") from exc
    if not isinstance(payload, dict):
        raise ValueError("stage1_json_pretty payload must be a JSON object")
    return payload


def _quote_bare_coord_tokens(text: str) -> str:
    parts: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            parts.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            parts.append(char)
            index += 1
            continue

        match = _COORD_TOKEN_RE.match(text, index)
        if match is not None:
            parts.append(json.dumps(match.group(0), ensure_ascii=False))
            index = match.end()
            continue

        parts.append(char)
        index += 1
    return "".join(parts)


def _has_balanced_json_delimiters(text: str) -> bool:
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"]": "[", "}": "{"}
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack and not in_string


def _validate_stage1_json_payload(payload: dict[str, Any]) -> None:
    if list(payload.keys()) != ["objects"]:
        raise ValueError("stage1_json_pretty payload must contain only objects")
    objects = payload["objects"]
    if not isinstance(objects, list):
        raise ValueError("stage1_json_pretty objects must be a list")
    for entry in objects:
        if not isinstance(entry, dict) or list(entry.keys()) != ["desc", "bbox_2d"]:
            raise ValueError("text is not canonical stage1_json_pretty")
        desc = entry["desc"]
        bbox = entry["bbox_2d"]
        if not isinstance(desc, str):
            raise ValueError("stage1_json_pretty desc must be a string")
        _validate_stage1_json_desc(desc)
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("stage1_json_pretty bbox_2d must contain four coordinates")
        _validate_strict_coord_tokens(
            bbox,
            context="stage1_json_pretty bbox_2d",
        )


__all__ = [
    "CharSpan",
    "CompactFullTemplate",
    "DetectionSequenceTemplate",
    "MaskGroup",
    "RenderedAssistantSequence",
    "RenderedConversation",
    "RenderedObjectEntry",
    "RenderSpanEvent",
    "SpanKind",
    "SpanProvenance",
    "Stage1JsonPrettyTemplate",
    "TemplateCapabilities",
    "TemplateId",
    "TokenRoleName",
    "get_detection_template",
]
