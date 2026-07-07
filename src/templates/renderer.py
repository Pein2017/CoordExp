"""English V1 object/box template rendering."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.common.errors import TemplateContractError
from src.common.qwen_aliases import find_invalid_qwen_wrapper_alias
from src.config.models import TemplateConfig
from src.coordinate_targets import (
    CoordinateLossTarget,
    coordinate_target_to_artifact,
)
from src.data import RawExample, RawObject
from src.templates.spans import RenderedSpan, validate_rendered_spans


OBJECT_REF_START_TOKEN = "<|object_ref_start|>"
OBJECT_REF_END_TOKEN = "<|object_ref_end|>"
BOX_START_TOKEN = "<|box_start|>"
BOX_END_TOKEN = "<|box_end|>"
IM_END_TOKEN = "<|im_end|>"
IM_END_SUFFIX = "<|im_end|>\n"
IMAGE_PLACEHOLDER = {"type": "image"}

UNSAFE_DESCRIPTION_SUBSTRINGS = (
    OBJECT_REF_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
    "<|coord_",
    "<|im_start|>",
    IM_END_TOKEN,
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
)

QWEN_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^>\s]+\|>")


@dataclass(frozen=True)
class RenderedObjectOrder:
    object_id: str
    source_index: int
    rendered_index: int

    def to_artifact_dict(self) -> dict[str, int | str]:
        return {
            "object_id": self.object_id,
            "source_index": self.source_index,
            "rendered_index": self.rendered_index,
        }


@dataclass(frozen=True)
class RenderedExample:
    example_id: str
    messages: tuple[dict[str, Any], ...]
    prompt_text: str
    assistant_content_text: str
    supervised_response_text: str
    spans: tuple[RenderedSpan, ...]
    realized_object_order: tuple[RenderedObjectOrder, ...]
    template_fingerprint: str
    object_ordering: str
    object_order_seed: int | None = None
    object_order_seed_source: str | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "messages": self.messages,
            "prompt_text": self.prompt_text,
            "assistant_content_text": self.assistant_content_text,
            "supervised_response_text": self.supervised_response_text,
            "spans": [_rendered_span_artifact(span) for span in self.spans],
            "realized_object_order": [
                item.to_artifact_dict() for item in self.realized_object_order
            ],
            "template_fingerprint": self.template_fingerprint,
            "object_ordering": self.object_ordering,
            "object_order_seed": self.object_order_seed,
            "object_order_seed_source": self.object_order_seed_source,
        }


def _rendered_span_artifact(span: RenderedSpan) -> dict[str, Any]:
    payload = {
        "kind": span.kind,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "text": span.text,
        "object_id": span.object_id,
        "field": span.field,
        "source": span.source,
    }
    coordinate_target = coordinate_target_to_artifact(span.coordinate_target)
    if coordinate_target is not None:
        payload["coordinate_target"] = coordinate_target
    return payload


def render_example(
    raw_example: RawExample,
    template_config: TemplateConfig,
    *,
    object_order_seed: int | None = None,
) -> RenderedExample:
    if not isinstance(raw_example, RawExample):
        raise TemplateContractError(
            "render_example requires a validated RawExample",
            code="template.raw_example_type",
            context={"value_type": type(raw_example).__name__},
        )
    if template_config.assistant_format != "object_box_closed":
        raise TemplateContractError(
            "unsupported assistant format",
            code="template.assistant_format",
            context={"assistant_format": template_config.assistant_format},
        )

    ordered_objects, order_seed_source = _ordered_objects(
        raw_example,
        template_config.object_ordering,
        object_order_seed=object_order_seed,
    )
    assistant_content, spans, object_order = _render_objects(
        ordered_objects,
        object_field_order=template_config.object_field_order,
    )
    supervised_response_text = assistant_content + IM_END_SUFFIX
    _validate_assistant_suffix(assistant_content, supervised_response_text)
    _append_suffix_spans(
        supervised_response_text,
        spans,
        assistant_content_length=len(assistant_content),
    )
    validate_rendered_spans(supervised_response_text, tuple(spans))

    prompt_text = _prompt_text(template_config)
    messages = _messages(raw_example, template_config, prompt_text, assistant_content)
    return RenderedExample(
        example_id=raw_example.example_id,
        messages=messages,
        prompt_text=prompt_text,
        assistant_content_text=assistant_content,
        supervised_response_text=supervised_response_text,
        spans=tuple(spans),
        realized_object_order=tuple(object_order),
        template_fingerprint=_template_fingerprint(template_config, prompt_text),
        object_ordering=template_config.object_ordering,
        object_order_seed=object_order_seed,
        object_order_seed_source=order_seed_source,
    )


def _ordered_objects(
    raw_example: RawExample,
    object_ordering: str,
    *,
    object_order_seed: int | None,
) -> tuple[tuple[int, RawObject], ...]:
    indexed = tuple(enumerate(raw_example.objects))
    if object_ordering == "source_order":
        return indexed, None
    if object_ordering == "geo_sorted":
        unsorted = _first_unsorted_top_left_pair(indexed)
        if unsorted is not None:
            prev_index, curr_index, prev_anchor, curr_anchor = unsorted
            raise TemplateContractError(
                "geo_sorted object ordering requires top-to-bottom then left-to-right rows",
                code="template.geo_sorted_order",
                context={
                    "example_id": raw_example.example_id,
                    "previous_index": prev_index,
                    "current_index": curr_index,
                    "previous_anchor": list(prev_anchor),
                    "current_anchor": list(curr_anchor),
                },
            )
        return indexed, None
    if object_ordering == "random":
        if object_order_seed is None:
            raise TemplateContractError(
                "random object ordering requires an explicit run-controlled seed",
                code="template.random_seed_required",
                context={"example_id": raw_example.example_id},
            )
        seed_source = f"{object_order_seed}:{raw_example.example_id}"
        seed_bytes = hashlib.sha256(seed_source.encode("utf-8")).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))
        shuffled = list(indexed)
        rng.shuffle(shuffled)
        return tuple(shuffled), seed_source
    raise TemplateContractError(
        "unsupported object ordering",
        code="template.object_ordering",
        context={"object_ordering": object_ordering},
    )


def _first_unsorted_top_left_pair(
    indexed_objects: Sequence[tuple[int, RawObject]],
) -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
    if len(indexed_objects) < 2:
        return None
    previous_index, previous_object = indexed_objects[0]
    previous_anchor = _top_left_anchor(previous_object)
    for current_index, current_object in indexed_objects[1:]:
        current_anchor = _top_left_anchor(current_object)
        if current_anchor < previous_anchor:
            return previous_index, current_index, previous_anchor, current_anchor
        previous_index = current_index
        previous_anchor = current_anchor
    return None


def _top_left_anchor(obj: RawObject) -> tuple[int, int]:
    x1, y1, _x2, _y2 = obj.bbox
    return y1, x1


def _render_objects(
    ordered_objects: Sequence[tuple[int, RawObject]],
    *,
    object_field_order: Literal["desc_first", "geometry_first"],
) -> tuple[str, list[RenderedSpan], list[RenderedObjectOrder]]:
    parts: list[str] = []
    spans: list[RenderedSpan] = []
    object_order: list[RenderedObjectOrder] = []
    cursor = 0
    for rendered_index, (source_index, obj) in enumerate(ordered_objects):
        object_start = cursor
        object_order.append(
            RenderedObjectOrder(
                object_id=obj.object_id,
                source_index=source_index,
                rendered_index=rendered_index,
            )
        )
        if object_field_order == "desc_first":
            segment, segment_spans = _desc_segment(obj, cursor)
            parts.append(segment)
            spans.extend(segment_spans)
            cursor += len(segment)
            segment, segment_spans = _box_segment(obj, cursor)
            parts.append(segment)
            spans.extend(segment_spans)
            cursor += len(segment)
        elif object_field_order == "geometry_first":
            segment, segment_spans = _box_segment(obj, cursor)
            parts.append(segment)
            spans.extend(segment_spans)
            cursor += len(segment)
            segment, segment_spans = _desc_segment(obj, cursor)
            parts.append(segment)
            spans.extend(segment_spans)
            cursor += len(segment)
        else:
            raise TemplateContractError(
                "unsupported object field order",
                code="template.object_field_order",
                context={"object_field_order": object_field_order},
            )
        object_text = "".join(parts)[object_start:cursor]
        spans.append(
            RenderedSpan(
                kind="object",
                char_start=object_start,
                char_end=cursor,
                text=object_text,
                object_id=obj.object_id,
                source=f"objects[{source_index}]",
            )
        )
    assistant_content = "".join(parts)
    if assistant_content:
        spans.append(
            RenderedSpan(
                kind="assistant_content",
                char_start=0,
                char_end=len(assistant_content),
                text=assistant_content,
            )
        )
    return assistant_content, spans, object_order


def _desc_segment(obj: RawObject, cursor: int) -> tuple[str, list[RenderedSpan]]:
    description = _safe_description(obj)
    segment = f"{OBJECT_REF_START_TOKEN}{description}{OBJECT_REF_END_TOKEN}"
    spans = [
        _span("schema_token", cursor, OBJECT_REF_START_TOKEN, obj, "object_ref_start"),
        _span(
            "description",
            cursor + len(OBJECT_REF_START_TOKEN),
            description,
            obj,
            "description",
        ),
        _span(
            "schema_token",
            cursor + len(OBJECT_REF_START_TOKEN) + len(description),
            OBJECT_REF_END_TOKEN,
            obj,
            "object_ref_end",
        ),
    ]
    return segment, spans


def _box_segment(obj: RawObject, cursor: int) -> tuple[str, list[RenderedSpan]]:
    coord_tokens = tuple(f"<|coord_{value}|>" for value in obj.bbox)
    coord_text = "".join(coord_tokens)
    segment = f"{BOX_START_TOKEN}{coord_text}{BOX_END_TOKEN}"
    spans = [_span("schema_token", cursor, BOX_START_TOKEN, obj, "box_start")]
    coord_cursor = cursor + len(BOX_START_TOKEN)
    for index, token in enumerate(coord_tokens):
        spans.append(
            _span(
                "coordinate_token",
                coord_cursor,
                token,
                obj,
                f"bbox[{index}]",
                coordinate_target=CoordinateLossTarget(
                    bbox=obj.bbox,
                    slot_index=index,
                ),
            )
        )
        coord_cursor += len(token)
    spans.append(_span("schema_token", coord_cursor, BOX_END_TOKEN, obj, "box_end"))
    return segment, spans


def _span(
    kind: RenderedSpanKind,
    char_start: int,
    text: str,
    obj: RawObject,
    field: str,
    *,
    coordinate_target: CoordinateLossTarget | None = None,
) -> RenderedSpan:
    return RenderedSpan(
        kind=kind,
        char_start=char_start,
        char_end=char_start + len(text),
        text=text,
        object_id=obj.object_id,
        field=field,
        source=f"object_id:{obj.object_id}",
        coordinate_target=coordinate_target,
    )


def _append_suffix_spans(
    supervised_response_text: str,
    spans: list[RenderedSpan],
    *,
    assistant_content_length: int,
) -> None:
    if not supervised_response_text.endswith(IM_END_SUFFIX):
        raise TemplateContractError(
            "supervised response must end with the Qwen assistant suffix",
            code="template.im_end_suffix",
        )
    spans.append(
        RenderedSpan(
            kind="eos_transition",
            char_start=assistant_content_length,
            char_end=assistant_content_length + len(IM_END_TOKEN),
            text=IM_END_TOKEN,
            field="assistant_terminal",
        )
    )
    spans.append(
        RenderedSpan(
            kind="ignored_text",
            char_start=assistant_content_length + len(IM_END_TOKEN),
            char_end=assistant_content_length + len(IM_END_SUFFIX),
            text="\n",
            field="assistant_terminal_newline",
        )
    )


def _validate_assistant_suffix(assistant_content: str, supervised_response_text: str) -> None:
    if IM_END_TOKEN in assistant_content:
        raise TemplateContractError(
            "assistant content must not contain an explicit im_end token before suffix insertion",
            code="template.duplicate_im_end",
        )
    if not supervised_response_text.endswith(IM_END_SUFFIX):
        raise TemplateContractError(
            "supervised response must end exactly with <|im_end|> followed by newline",
            code="template.im_end_suffix",
        )
    if supervised_response_text.count(IM_END_TOKEN) != 1:
        raise TemplateContractError(
            "supervised response must contain exactly one im_end transition token",
            code="template.duplicate_im_end",
            context={"count": supervised_response_text.count(IM_END_TOKEN)},
        )


def _safe_description(obj: RawObject) -> str:
    description = re.sub(r"[\n\r\t]+", " ", obj.description).strip()
    if not description:
        raise TemplateContractError(
            "object description must not be empty after normalization",
            code="template.description_empty",
            context={"object_id": obj.object_id},
        )
    for token in UNSAFE_DESCRIPTION_SUBSTRINGS:
        if token in description:
            raise TemplateContractError(
                "object description contains unsafe special-token text",
                code="template.description_unsafe_token",
                context={"object_id": obj.object_id, "token": token},
            )
    match = QWEN_CONTROL_TOKEN_PATTERN.search(description)
    if match is not None:
        raise TemplateContractError(
            "object description contains unsafe Qwen control-token text",
            code="template.description_unsafe_token",
            context={"object_id": obj.object_id, "token": match.group(0)},
        )
    return description


def _prompt_text(template_config: TemplateConfig) -> str:
    prompt_text = template_config.prompt.user.strip()
    _validate_prompt_text(prompt_text, "template.prompt.user")
    return prompt_text


def _messages(
    raw_example: RawExample,
    template_config: TemplateConfig,
    prompt_text: str,
    assistant_content: str,
) -> tuple[dict[str, Any], ...]:
    messages: list[dict[str, Any]] = []
    if template_config.prompt.system is not None:
        system_text = template_config.prompt.system.strip()
        if system_text:
            _validate_prompt_text(system_text, "template.prompt.system")
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(raw_example.image.path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    )
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]}
    )
    return tuple(messages)


def _validate_prompt_text(text: str, field: str) -> None:
    match = find_invalid_qwen_wrapper_alias(text)
    if match is None:
        return
    alias, canonical = match
    raise TemplateContractError(
        "prompt text contains a non-canonical Qwen wrapper alias",
        code="template.prompt_unsafe_alias",
        context={"field": field, "token": alias, "canonical_token": canonical},
    )


def _template_fingerprint(template_config: TemplateConfig, prompt_text: str) -> str:
    payload = {
        "assistant_format": template_config.assistant_format,
        "object_field_order": template_config.object_field_order,
        "object_ordering": template_config.object_ordering,
        "prompt_text": prompt_text,
        "renderer": "coordexp-swift-template-v1",
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BOX_END_TOKEN",
    "BOX_START_TOKEN",
    "IMAGE_PLACEHOLDER",
    "IM_END_SUFFIX",
    "IM_END_TOKEN",
    "OBJECT_REF_END_TOKEN",
    "OBJECT_REF_START_TOKEN",
    "RenderedExample",
    "RenderedObjectOrder",
    "render_example",
]
