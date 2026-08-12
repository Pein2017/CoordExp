"""Rendered-example tokenization and span alignment for Qwen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from src.common.errors import EncodingContractError
from src.config.models import ProcessorConfig
from src.coordinate_targets import (
    CoordinateLossTarget,
    coordinate_target_to_artifact,
)
from src.data import RawExample
from src.qwen.images import QwenImageEncoding, encode_qwen_image, plan_qwen_image
from src.qwen.tokens import IM_END_SUFFIX
from src.templates import RenderedExample, RenderedSpan, validate_rendered_spans


ASSISTANT_HEADER = "<|im_start|>assistant\n"
IMAGE_PAD_TOKEN = "<|image_pad|>"

EncodedTokenType = Literal["desc_text", "schema", "coordinate", "eos", "ignored"]


@dataclass(frozen=True)
class EncodedTokenSpan:
    token_type: EncodedTokenType
    text: str
    char_start: int
    char_end: int
    chat_char_start: int
    chat_char_end: int
    base_token_start: int
    base_token_end: int
    physical_token_start: int
    physical_token_end: int
    token_ids: tuple[int, ...]
    object_id: str | None
    field: str | None
    source: str | None
    coordinate_target: CoordinateLossTarget | None = None

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = {
            "token_type": self.token_type,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chat_char_start": self.chat_char_start,
            "chat_char_end": self.chat_char_end,
            "base_token_start": self.base_token_start,
            "base_token_end": self.base_token_end,
            "physical_token_start": self.physical_token_start,
            "physical_token_end": self.physical_token_end,
            "token_ids": list(self.token_ids),
            "object_id": self.object_id,
            "field": self.field,
            "source": self.source,
        }
        coordinate_target = coordinate_target_to_artifact(self.coordinate_target)
        if coordinate_target is not None:
            payload["coordinate_target"] = coordinate_target
        return payload


@dataclass(frozen=True)
class EncodedExample:
    example_id: str
    chat_text: str
    base_input_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    base_offset_mapping: tuple[tuple[int, int], ...]
    base_to_physical_start: tuple[int, ...]
    image_token_count: int
    image_encoding: QwenImageEncoding
    assistant_content_start_char: int
    image_pad_base_index: int
    image_pad_physical_start: int
    image_pad_physical_end: int
    supervised_token_spans: tuple[EncodedTokenSpan, ...]
    ignored_token_spans: tuple[EncodedTokenSpan, ...]
    global_max_length: int

    @property
    def base_token_count(self) -> int:
        return len(self.base_input_ids)

    @property
    def input_length(self) -> int:
        return len(self.input_ids)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "base_token_count": self.base_token_count,
            "input_length": self.input_length,
            "global_max_length": self.global_max_length,
            "image_token_count": self.image_token_count,
            "image_pad_base_index": self.image_pad_base_index,
            "image_pad_physical_start": self.image_pad_physical_start,
            "image_pad_physical_end": self.image_pad_physical_end,
            "assistant_content_start_char": self.assistant_content_start_char,
            "base_input_ids_sha256": _sha256_int_sequence(self.base_input_ids),
            "input_ids_sha256": _sha256_int_sequence(self.input_ids),
            "base_offset_mapping_sha256": _sha256_int_pairs(self.base_offset_mapping),
            "base_to_physical_start": list(self.base_to_physical_start),
            "supervised_span_count": len(self.supervised_token_spans),
            "ignored_span_count": len(self.ignored_token_spans),
            "supervised_token_count": sum(
                span.token_count for span in self.supervised_token_spans
            ),
            "ignored_token_count": sum(
                span.token_count for span in self.ignored_token_spans
            ),
            "image_encoding": self.image_encoding.to_artifact_dict(),
            "supervised_token_spans": [
                span.to_artifact_dict() for span in self.supervised_token_spans
            ],
            "ignored_token_spans": [
                span.to_artifact_dict() for span in self.ignored_token_spans
            ],
        }


def encode_rendered_example(
    raw_example: RawExample,
    rendered: RenderedExample,
    *,
    components: Any,
    processor_config: ProcessorConfig,
    global_max_length: int,
    materialize_image_pixels: bool = True,
) -> EncodedExample:
    if raw_example.example_id != rendered.example_id:
        raise EncodingContractError(
            "RawExample and RenderedExample example ids must match",
            code="qwen.encoded_example_id_mismatch",
            context={
                "raw_example_id": raw_example.example_id,
                "rendered_example_id": rendered.example_id,
            },
        )
    if not rendered.supervised_response_text.endswith(IM_END_SUFFIX):
        raise EncodingContractError(
            "rendered supervised response must include the assistant terminal suffix",
            code="qwen.assistant_suffix_alignment",
            context={"example_id": rendered.example_id},
        )
    validate_rendered_spans(rendered.supervised_response_text, rendered.spans)

    image_encoding = (
        encode_qwen_image(
            raw_example,
            components=components,
            processor_config=processor_config,
        )
        if materialize_image_pixels
        else plan_qwen_image(
            raw_example,
            components=components,
            processor_config=processor_config,
        )
    )
    chat_text = _apply_chat_template(components.processor, rendered)
    assistant_start = _assistant_content_start(chat_text, rendered)
    tokenized = components.tokenizer(
        chat_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    base_input_ids = tuple(int(token_id) for token_id in tokenized["input_ids"])
    base_offset_mapping = tuple(
        (int(start), int(end)) for start, end in tokenized["offset_mapping"]
    )
    expansion = _expand_image_pad_tokens(
        tokenizer=components.tokenizer,
        base_input_ids=base_input_ids,
        image_token_count=image_encoding.merged_visual_tokens,
        example_id=rendered.example_id,
    )
    input_ids = expansion.input_ids
    base_to_physical_start = expansion.base_to_physical_start

    if len(input_ids) > global_max_length:
        raise EncodingContractError(
            "encoded example exceeds packing.global_max_length",
            code="qwen.encoded_example_too_long",
            context={
                "example_id": rendered.example_id,
                "input_length": len(input_ids),
                "global_max_length": global_max_length,
            },
        )

    supervised_spans, ignored_spans = _align_rendered_spans(
        rendered,
        chat_text=chat_text,
        assistant_start=assistant_start,
        base_input_ids=base_input_ids,
        base_offset_mapping=base_offset_mapping,
        base_to_physical_start=base_to_physical_start,
    )
    return EncodedExample(
        example_id=rendered.example_id,
        chat_text=chat_text,
        base_input_ids=base_input_ids,
        input_ids=input_ids,
        base_offset_mapping=base_offset_mapping,
        base_to_physical_start=base_to_physical_start,
        image_token_count=image_encoding.merged_visual_tokens,
        image_encoding=image_encoding,
        assistant_content_start_char=assistant_start,
        image_pad_base_index=expansion.image_pad_base_index,
        image_pad_physical_start=expansion.image_pad_physical_start,
        image_pad_physical_end=expansion.image_pad_physical_end,
        supervised_token_spans=supervised_spans,
        ignored_token_spans=ignored_spans,
        global_max_length=global_max_length,
    )


def _apply_chat_template(processor: Any, rendered: RenderedExample) -> str:
    chat_text = processor.apply_chat_template(
        list(rendered.messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(chat_text, str):
        raise EncodingContractError(
            "Qwen processor chat template must return text",
            code="qwen.chat_template_text",
            context={
                "example_id": rendered.example_id,
                "value_type": type(chat_text).__name__,
            },
        )
    return chat_text


def _assistant_content_start(chat_text: str, rendered: RenderedExample) -> int:
    header_start = chat_text.rfind(ASSISTANT_HEADER)
    if header_start < 0:
        raise EncodingContractError(
            "Qwen chat template did not contain the assistant header",
            code="qwen.assistant_header_missing",
            context={"example_id": rendered.example_id},
        )
    assistant_start = header_start + len(ASSISTANT_HEADER)
    expected_end = assistant_start + len(rendered.supervised_response_text)
    if chat_text[assistant_start:expected_end] != rendered.supervised_response_text:
        raise EncodingContractError(
            "rendered assistant target does not align with Qwen chat template",
            code="qwen.assistant_suffix_alignment",
            context={
                "example_id": rendered.example_id,
                "assistant_start": assistant_start,
            },
        )
    return assistant_start


@dataclass(frozen=True)
class _ImagePadExpansion:
    input_ids: tuple[int, ...]
    base_to_physical_start: tuple[int, ...]
    image_pad_base_index: int
    image_pad_physical_start: int
    image_pad_physical_end: int


def _expand_image_pad_tokens(
    *,
    tokenizer: Any,
    base_input_ids: tuple[int, ...],
    image_token_count: int,
    example_id: str,
) -> _ImagePadExpansion:
    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if image_pad_id is None:
        raise EncodingContractError(
            "Qwen tokenizer is missing <|image_pad|>",
            code="qwen.image_pad_missing",
            context={"example_id": example_id},
        )
    image_pad_id = int(image_pad_id)
    image_pad_indices = [
        index
        for index, token_id in enumerate(base_input_ids)
        if token_id == image_pad_id
    ]
    if len(image_pad_indices) != 1:
        raise EncodingContractError(
            "V1 encoded examples must contain exactly one image placeholder",
            code="qwen.image_pad_count",
            context={
                "example_id": example_id,
                "image_pad_count": len(image_pad_indices),
            },
        )
    image_pad_base_index = image_pad_indices[0]

    expanded: list[int] = []
    base_to_physical_start: list[int] = []
    for token_id in base_input_ids:
        base_to_physical_start.append(len(expanded))
        if token_id == image_pad_id:
            expanded.extend([image_pad_id] * image_token_count)
        else:
            expanded.append(token_id)
    image_pad_physical_start = base_to_physical_start[image_pad_base_index]
    return _ImagePadExpansion(
        input_ids=tuple(expanded),
        base_to_physical_start=tuple(base_to_physical_start),
        image_pad_base_index=image_pad_base_index,
        image_pad_physical_start=image_pad_physical_start,
        image_pad_physical_end=image_pad_physical_start + image_token_count,
    )


def _align_rendered_spans(
    rendered: RenderedExample,
    *,
    chat_text: str,
    assistant_start: int,
    base_input_ids: tuple[int, ...],
    base_offset_mapping: tuple[tuple[int, int], ...],
    base_to_physical_start: tuple[int, ...],
) -> tuple[tuple[EncodedTokenSpan, ...], tuple[EncodedTokenSpan, ...]]:
    supervised: list[EncodedTokenSpan] = []
    ignored: list[EncodedTokenSpan] = []
    for span in rendered.spans:
        token_type = _span_token_type(span)
        if token_type is None:
            continue
        encoded_span = _align_one_span(
            span,
            token_type=token_type,
            assistant_start=assistant_start,
            chat_text=chat_text,
            base_input_ids=base_input_ids,
            base_offset_mapping=base_offset_mapping,
            base_to_physical_start=base_to_physical_start,
            example_id=rendered.example_id,
        )
        if token_type == "ignored":
            ignored.append(encoded_span)
        else:
            supervised.append(encoded_span)
    if not supervised:
        raise EncodingContractError(
            "encoded example must contain at least one supervised token span",
            code="qwen.supervision_empty",
            context={"example_id": rendered.example_id},
        )
    return tuple(supervised), tuple(ignored)


def _span_token_type(span: RenderedSpan) -> EncodedTokenType | None:
    if span.kind == "description":
        return "desc_text"
    if span.kind == "schema_token":
        return "schema"
    if span.kind == "coordinate_token":
        return "coordinate"
    if span.kind == "eos_transition":
        return "eos"
    if span.kind == "ignored_text":
        return "ignored"
    return None


def _align_one_span(
    span: RenderedSpan,
    *,
    token_type: EncodedTokenType,
    assistant_start: int,
    chat_text: str,
    base_input_ids: tuple[int, ...],
    base_offset_mapping: tuple[tuple[int, int], ...],
    base_to_physical_start: tuple[int, ...],
    example_id: str,
) -> EncodedTokenSpan:
    chat_char_start = assistant_start + span.char_start
    chat_char_end = assistant_start + span.char_end
    if chat_text[chat_char_start:chat_char_end] != span.text:
        raise EncodingContractError(
            "rendered span text does not match Qwen chat-template slice",
            code="qwen.span_chat_text_mismatch",
            context={
                "example_id": example_id,
                "span_kind": span.kind,
                "text": span.text,
                "actual_text": chat_text[chat_char_start:chat_char_end],
            },
        )
    token_indices = _token_indices_for_char_range(
        base_offset_mapping,
        char_start=chat_char_start,
        char_end=chat_char_end,
        example_id=example_id,
        span_text=span.text,
    )
    base_token_start = token_indices[0]
    base_token_end = token_indices[-1] + 1
    physical_token_start = base_to_physical_start[base_token_start]
    last_base_token = token_indices[-1]
    physical_token_end = base_to_physical_start[last_base_token] + 1
    return EncodedTokenSpan(
        token_type=token_type,
        text=span.text,
        char_start=span.char_start,
        char_end=span.char_end,
        chat_char_start=chat_char_start,
        chat_char_end=chat_char_end,
        base_token_start=base_token_start,
        base_token_end=base_token_end,
        physical_token_start=physical_token_start,
        physical_token_end=physical_token_end,
        token_ids=tuple(base_input_ids[index] for index in token_indices),
        object_id=span.object_id,
        field=span.field,
        source=span.source,
        coordinate_target=getattr(span, "coordinate_target", None),
    )


def _token_indices_for_char_range(
    offset_mapping: Sequence[tuple[int, int]],
    *,
    char_start: int,
    char_end: int,
    example_id: str,
    span_text: str,
) -> tuple[int, ...]:
    # Must stay a full linear scan, not a bisect/binary search: zero-width
    # special-token offsets (e.g. (0, 0)) can appear non-monotonically inside
    # offset_mapping, which would silently break a sorted-search shortcut.
    # See openspec/changes/archive/2026-08-06-streamline-coordexp-swift-base-infrastructure/
    # implementation-notes.md "M5a" and
    # test_token_span_lookup_handles_non_monotonic_zero_width_offset_without_bisect
    # for the concrete counterexample.
    indices: list[int] = []
    for index, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= char_start or token_start >= char_end:
            continue
        if token_start < char_start or token_end > char_end:
            raise EncodingContractError(
                "tokenizer offset crosses a rendered span boundary",
                code="qwen.span_token_boundary",
                context={
                    "example_id": example_id,
                    "span_text": span_text,
                    "span_range": [char_start, char_end],
                    "token_index": index,
                    "token_range": [token_start, token_end],
                },
            )
        indices.append(index)
    if not indices:
        raise EncodingContractError(
            "rendered span did not align to any tokenizer ids",
            code="qwen.span_token_missing",
            context={
                "example_id": example_id,
                "span_text": span_text,
                "span_range": [char_start, char_end],
            },
        )
    first_start = offset_mapping[indices[0]][0]
    last_end = offset_mapping[indices[-1]][1]
    if first_start != char_start or last_end != char_end:
        raise EncodingContractError(
            "tokenizer offsets do not exactly cover rendered span",
            code="qwen.span_token_coverage",
            context={
                "example_id": example_id,
                "span_text": span_text,
                "span_range": [char_start, char_end],
                "covered_range": [first_start, last_end],
            },
        )
    return tuple(indices)


def _sha256_int_sequence(values: Sequence[int]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_int_pairs(values: Sequence[tuple[int, int]]) -> str:
    encoded = json.dumps(
        [[left, right] for left, right in values],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ASSISTANT_HEADER",
    "IMAGE_PAD_TOKEN",
    "EncodedExample",
    "EncodedTokenSpan",
    "EncodedTokenType",
    "encode_rendered_example",
]
