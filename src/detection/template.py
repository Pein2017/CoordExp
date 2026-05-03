"""Detection assistant-sequence templates and assistant-local spans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.data import NormalizedDetectionObject, NormalizedDetectionSample
from src.utils.assistant_json import dumps_coordjson


TemplateId = Literal["stage1_json_pretty", "compact_full"]

_COORD_TOKEN_RE = re.compile(r"<\|coord_\d+\|>")
_COMPACT_FORBIDDEN_DESC_SUBSTRINGS = (
    "\n",
    "\t",
    OBJECT_REF_START_TOKEN,
    BOX_START_TOKEN,
    "<|coord_",
    "<|im_start|>",
    "<|im_end|>",
)


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
        structural_spans: list[CharSpan] = []
        separator_spans: list[CharSpan] = []
        entries: list[RenderedObjectEntry] = []

        structural_spans.append(builder.append("{", "json_root_open"))
        structural_spans.append(builder.append('"objects": ', "objects_key"))
        structural_spans.append(builder.append("[", "json_array_open"))
        for object_index, obj in enumerate(sample.objects):
            if object_index:
                separator_span = builder.append(
                    self.render_separator(object_index - 1, object_index),
                    "object_separator",
                )
                separator_spans.append(separator_span)
                structural_spans.append(separator_span)
                entries[-1] = replace(entries[-1], separator_span=separator_span)
            entry = _append_stage1_json_entry(builder, obj, object_index)
            entries.append(entry)
            structural_spans.extend(entry.control_spans)

        terminal_start = len(builder)
        structural_spans.append(builder.append("]", "json_array_close"))
        structural_spans.append(builder.append("}", "json_root_close"))
        terminal_close_span = CharSpan(
            terminal_start,
            len(builder),
            "terminal_close",
        )

        text = builder.text
        _assert_stage1_json_fixture_parity(text, sample)
        return RenderedAssistantSequence(
            template_id=self.template_id,
            template_version=self.capabilities.version,
            text=text,
            object_entries=tuple(entries),
            separator_spans=tuple(separator_spans),
            terminal_close_span=terminal_close_span,
            stop_marker_spans=(),
            structural_token_spans=tuple(structural_spans),
            trie_eligible_spans=tuple(entry.trie_eligible_span for entry in entries),
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
                entries[-1] = replace(entries[-1], separator_span=separator_span)
            entry = _append_compact_full_entry(builder, obj, object_index)
            entries.append(entry)
            structural_spans.extend(entry.control_spans)

        terminal_close_span = CharSpan(len(builder), len(builder), "terminal_close")
        return RenderedAssistantSequence(
            template_id=self.template_id,
            template_version=self.capabilities.version,
            text=builder.text,
            object_entries=tuple(entries),
            separator_spans=tuple(separator_spans),
            terminal_close_span=terminal_close_span,
            stop_marker_spans=(),
            structural_token_spans=tuple(structural_spans),
            trie_eligible_spans=tuple(entry.trie_eligible_span for entry in entries),
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
        return (
            f"{OBJECT_REF_START_TOKEN}{obj.desc}{BOX_START_TOKEN}"
            f"{''.join(obj.bbox_2d.tokens)}"
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
    for coord_index, token in enumerate(obj.bbox_2d.tokens):
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
        for coord_index, token in enumerate(obj.bbox_2d.tokens)
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
    )


def _object_payload(obj: NormalizedDetectionObject) -> dict[str, Any]:
    return {
        "desc": obj.desc,
        "bbox_2d": list(obj.bbox_2d.tokens),
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
    for token in obj.bbox_2d.tokens:
        if not _is_strict_coord_token(token):
            raise ValueError(
                f"object {obj.object_instance_id} must use coord-token bbox_2d values"
            )


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
    if not row.startswith(OBJECT_REF_START_TOKEN):
        raise ValueError("text is not strict compact_full")
    body = row[len(OBJECT_REF_START_TOKEN) :]
    if BOX_START_TOKEN not in body:
        raise ValueError("text is not strict compact_full")

    desc, coord_tail = body.split(BOX_START_TOKEN, maxsplit=1)
    _validate_compact_desc(desc)
    coords = _COORD_TOKEN_RE.findall(coord_tail)
    if len(coords) != 4 or "".join(coords) != coord_tail:
        raise ValueError("text is not strict compact_full")
    _validate_strict_coord_tokens(coords, context="strict compact_full bbox_2d")
    return {"desc": desc, "bbox_2d": coords}


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
    "RenderedAssistantSequence",
    "RenderedConversation",
    "RenderedObjectEntry",
    "Stage1JsonPrettyTemplate",
    "TemplateCapabilities",
    "TemplateId",
    "get_detection_template",
]
