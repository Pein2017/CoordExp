"""Training-aligned prompt construction for inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from PIL import Image

from src.common.errors import EncodingContractError
from src.config.models import TemplateConfig
from src.data import RawExample
from src.templates import render_example


TEMPLATE_ID = "coordexp-swift-template-v1"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
CHAT_TURN_START = "<|im_start|>"
ASSISTANT_TERMINATOR = "<|im_end|>"
IMAGE_PLACEHOLDER = "<|image_pad|>"
KNOWN_END_OF_SEQUENCE_TOKENS = ("<|endoftext|>", "<|end_of_text|>")


@dataclass(frozen=True)
class AssistantContinuation:
    """Canonical text that continues the currently open assistant response."""

    text: str


@dataclass(frozen=True)
class PromptRecord:
    row_id: str
    row_index: int
    example_id: str
    messages: tuple[dict[str, Any], ...]
    prompt_text: str
    chat_text: str
    prompt_token_ids: list[int]
    template_id: str
    template_fingerprint: str
    object_ordering: str
    object_field_order: str
    assistant_format: str
    realized_object_order: list[dict[str, int | str]]
    full_prompt_fingerprint: str
    continuation_text_sha256: str | None = None
    open_assistant_content_start_character: int | None = None
    open_assistant_content_start_byte: int | None = None
    continuation_character_span: tuple[int, int] | None = None
    continuation_byte_span: tuple[int, int] | None = None
    continuation_token_impact_span: tuple[int, int] | None = None
    image_placeholder_count: int | None = None
    open_assistant_interval_verified: bool | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = {
            "row_id": self.row_id,
            "row_index": self.row_index,
            "example_id": self.example_id,
            "prompt_text": self.prompt_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_token_count": len(self.prompt_token_ids),
            "template_id": self.template_id,
            "template_fingerprint": self.template_fingerprint,
            "object_ordering": self.object_ordering,
            "object_field_order": self.object_field_order,
            "assistant_format": self.assistant_format,
            "realized_object_order": list(self.realized_object_order),
            "full_prompt_fingerprint": self.full_prompt_fingerprint,
        }
        if self.continuation_text_sha256 is not None:
            payload.update(
                {
                    "full_chat_text": self.chat_text,
                    "continuation_text_sha256": self.continuation_text_sha256,
                    "open_assistant_content_start_character": (
                        self.open_assistant_content_start_character
                    ),
                    "open_assistant_content_start_byte": (
                        self.open_assistant_content_start_byte
                    ),
                    "continuation_character_span": list(
                        _required_span(
                            self.continuation_character_span,
                            field="continuation_character_span",
                        )
                    ),
                    "continuation_byte_span": list(
                        _required_span(
                            self.continuation_byte_span,
                            field="continuation_byte_span",
                        )
                    ),
                    "continuation_token_impact_span": list(
                        _required_span(
                            self.continuation_token_impact_span,
                            field="continuation_token_impact_span",
                        )
                    ),
                    "image_placeholder_count": self.image_placeholder_count,
                    "open_assistant_interval_verified": (
                        self.open_assistant_interval_verified
                    ),
                }
            )
        return payload


def build_prompt_record(
    raw_example: RawExample,
    template_config: TemplateConfig,
    *,
    processor: Any,
    row_index: int,
    object_order_seed: int | None = None,
    assistant_continuation: AssistantContinuation | None = None,
    max_prompt_tokens: int | None = None,
    visual_input_image: Image.Image | None = None,
) -> PromptRecord:
    rendered = render_example(
        raw_example,
        template_config,
        object_order_seed=object_order_seed,
    )
    messages = tuple(
        message for message in rendered.messages if message.get("role") != "assistant"
    )
    processing_messages = messages
    if visual_input_image is not None:
        processing_messages = _messages_with_visual_input_image(
            messages,
            visual_input_image=visual_input_image,
        )
    ordinary_chat_text = _apply_generation_chat_template(
        processor,
        messages=processing_messages,
        example_id=raw_example.example_id,
    )
    ordinary_prompt_token_ids = _tokenize_prompt(
        processor,
        chat_text=ordinary_chat_text,
        messages=processing_messages,
        example_id=raw_example.example_id,
    )
    chat_text = ordinary_chat_text
    prompt_token_ids = ordinary_prompt_token_ids
    continuation_fields: dict[str, Any] = {}
    if assistant_continuation is not None:
        chat_text, prompt_token_ids, continuation_fields = _continue_open_assistant(
            processor,
            ordinary_chat_text=ordinary_chat_text,
            ordinary_prompt_token_ids=ordinary_prompt_token_ids,
            messages=processing_messages,
            continuation=assistant_continuation,
            example_id=raw_example.example_id,
            max_prompt_tokens=max_prompt_tokens,
            expected_chat_turn_count=len(messages) + 1,
        )
    return PromptRecord(
        row_id=raw_example.example_id,
        row_index=row_index,
        example_id=raw_example.example_id,
        messages=messages,
        prompt_text=rendered.prompt_text,
        chat_text=chat_text,
        prompt_token_ids=prompt_token_ids,
        template_id=TEMPLATE_ID,
        template_fingerprint=rendered.template_fingerprint,
        object_ordering=rendered.object_ordering,
        object_field_order=template_config.object_field_order,
        assistant_format=template_config.assistant_format,
        realized_object_order=[
            item.to_artifact_dict() for item in rendered.realized_object_order
        ],
        full_prompt_fingerprint=_full_prompt_fingerprint(chat_text, prompt_token_ids),
        **continuation_fields,
    )


def _messages_with_visual_input_image(
    messages: tuple[dict[str, Any], ...],
    *,
    visual_input_image: Image.Image,
) -> tuple[dict[str, Any], ...]:
    """Replace the rendered source path with the exact executed RGB image."""

    return tuple(
        {
            **message,
            "content": [
                (
                    {**item, "image": visual_input_image}
                    if item.get("type") == "image"
                    else item
                )
                for item in message["content"]
            ],
        }
        for message in messages
    )


def verify_prompt_token_parity(
    prompt_record: PromptRecord,
    *,
    backend_prompt_token_ids: list[int],
) -> dict[str, Any]:
    local_ids = list(prompt_record.prompt_token_ids)
    backend_ids = list(backend_prompt_token_ids)
    if local_ids != backend_ids:
        raise EncodingContractError(
            "inference prompt token ids do not match backend prompt token ids",
            code="inference.prompt_token_parity",
            context={
                "row_id": prompt_record.row_id,
                "local_count": len(local_ids),
                "backend_count": len(backend_ids),
            },
        )
    return {
        "row_id": prompt_record.row_id,
        "prompt_token_parity": "verified",
        "prompt_token_count": len(local_ids),
    }


def _apply_generation_chat_template(
    processor: Any,
    *,
    messages: tuple[dict[str, Any], ...],
    example_id: str,
) -> str:
    chat_text = processor.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(chat_text, str):
        raise EncodingContractError(
            "Qwen processor chat template must return prompt text",
            code="inference.prompt_chat_template_text",
            context={"example_id": example_id, "value_type": type(chat_text).__name__},
        )
    return chat_text


def _tokenize_prompt(
    processor: Any,
    *,
    chat_text: str,
    messages: tuple[dict[str, Any], ...],
    example_id: str,
) -> list[int]:
    tokenized = processor.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(tokenized, list):
        return _int_ids(tokenized, example_id=example_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise EncodingContractError(
            "Qwen processor does not expose a tokenizer for prompt ids",
            code="inference.prompt_tokenizer_missing",
            context={"example_id": example_id},
        )
    encoded = tokenizer(chat_text, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise EncodingContractError(
            "Qwen tokenizer did not return input_ids",
            code="inference.prompt_tokenizer_output",
            context={"example_id": example_id},
        )
    return _int_ids(encoded["input_ids"], example_id=example_id)


def _int_ids(value: Any, *, example_id: str) -> list[int]:
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    try:
        ids = [int(item) for item in value]
    except TypeError as exc:
        raise EncodingContractError(
            "prompt token ids must be an integer sequence",
            code="inference.prompt_token_ids",
            context={"example_id": example_id, "value_type": type(value).__name__},
            cause=exc,
        ) from exc
    return ids


def _continue_open_assistant(
    processor: Any,
    *,
    ordinary_chat_text: str,
    ordinary_prompt_token_ids: list[int],
    messages: tuple[dict[str, Any], ...],
    continuation: AssistantContinuation,
    example_id: str,
    max_prompt_tokens: int | None,
    expected_chat_turn_count: int,
) -> tuple[str, list[int], dict[str, Any]]:
    text = continuation.text
    if not isinstance(text, str) or not text.strip():
        raise EncodingContractError(
            "assistant continuation must be nonempty canonical text",
            code="inference.assistant_continuation_empty",
            context={"example_id": example_id},
        )
    _validate_continuation_controls(processor, text=text, example_id=example_id)

    header_start = ordinary_chat_text.rfind(ASSISTANT_HEADER)
    if header_start < 0:
        raise EncodingContractError(
            "generation prompt does not contain a final open assistant turn",
            code="inference.open_assistant_header_missing",
            context={"example_id": example_id},
        )
    observed_chat_turn_count = ordinary_chat_text.count(CHAT_TURN_START)
    if (
        observed_chat_turn_count != expected_chat_turn_count
        or header_start != ordinary_chat_text.rfind(CHAT_TURN_START)
    ):
        raise EncodingContractError(
            "generation prompt contains an unexpected extra chat turn",
            code="inference.open_assistant_extra_turn",
            context={
                "example_id": example_id,
                "expected_chat_turn_count": expected_chat_turn_count,
                "observed_chat_turn_count": observed_chat_turn_count,
            },
        )
    content_start = header_start + len(ASSISTANT_HEADER)
    open_interval = ordinary_chat_text[content_start:]
    _validate_open_assistant_interval(
        processor,
        interval=open_interval,
        example_id=example_id,
        content_start=content_start,
    )
    if open_interval:
        raise EncodingContractError(
            "generation prompt contains pre-existing assistant content",
            code="inference.open_assistant_preexisting_content",
            context={
                "example_id": example_id,
                "open_assistant_content_start_character": content_start,
                "preexisting_character_count": len(open_interval),
            },
        )

    expected_full_chat_text = ordinary_chat_text + text
    full_chat_text, prompt_token_ids = _apply_continued_chat_template(
        processor,
        messages=messages,
        continuation_text=text,
        example_id=example_id,
    )
    if full_chat_text != expected_full_chat_text:
        raise EncodingContractError(
            "native continued chat template differs from the intended open assistant continuation",
            code="inference.continuation_native_chat_text",
            context={"example_id": example_id},
        )
    image_placeholder_count = full_chat_text.count(IMAGE_PLACEHOLDER)
    if image_placeholder_count != 1:
        raise EncodingContractError(
            "continued prompt must contain exactly one image placeholder",
            code="inference.continuation_image_placeholder_count",
            context={
                "example_id": example_id,
                "image_placeholder_count": image_placeholder_count,
            },
        )
    prompt_token_limit = _prompt_token_limit(
        processor,
        explicit_limit=max_prompt_tokens,
        example_id=example_id,
    )
    if prompt_token_limit is not None and len(prompt_token_ids) > prompt_token_limit:
        raise EncodingContractError(
            "continued prompt exceeds the declared prompt-token context limit",
            code="inference.continuation_context_limit",
            context={
                "example_id": example_id,
                "prompt_token_count": len(prompt_token_ids),
                "max_prompt_tokens": prompt_token_limit,
            },
        )

    character_start = len(ordinary_chat_text)
    character_end = len(full_chat_text)
    byte_start = len(ordinary_chat_text.encode("utf-8"))
    byte_end = len(full_chat_text.encode("utf-8"))
    token_impact_start = _longest_common_prefix_length(
        ordinary_prompt_token_ids,
        prompt_token_ids,
    )
    return full_chat_text, prompt_token_ids, {
        "continuation_text_sha256": _sha256_text(text),
        "open_assistant_content_start_character": content_start,
        "open_assistant_content_start_byte": len(
            ordinary_chat_text[:content_start].encode("utf-8")
        ),
        "continuation_character_span": (character_start, character_end),
        "continuation_byte_span": (byte_start, byte_end),
        "continuation_token_impact_span": (
            token_impact_start,
            len(prompt_token_ids),
        ),
        "image_placeholder_count": image_placeholder_count,
        "open_assistant_interval_verified": True,
    }


def _apply_continued_chat_template(
    processor: Any,
    *,
    messages: tuple[dict[str, Any], ...],
    continuation_text: str,
    example_id: str,
) -> tuple[str, list[int]]:
    continued_messages = [
        *messages,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": continuation_text}],
        },
    ]
    chat_text = processor.apply_chat_template(
        continued_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if not isinstance(chat_text, str):
        raise EncodingContractError(
            "Qwen processor continued chat template must return prompt text",
            code="inference.continuation_chat_template_text",
            context={"example_id": example_id, "value_type": type(chat_text).__name__},
        )
    tokenized = processor.apply_chat_template(
        continued_messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    return chat_text, _int_ids(tokenized, example_id=example_id)


def _validate_continuation_controls(
    processor: Any,
    *,
    text: str,
    example_id: str,
) -> None:
    forbidden = {
        "image_placeholder": (IMAGE_PLACEHOLDER,),
        "assistant_terminator": (ASSISTANT_TERMINATOR,),
        "end_of_sequence": _end_of_sequence_tokens(processor),
        "chat_turn_opener": (CHAT_TURN_START,),
    }
    for boundary_class, tokens in forbidden.items():
        matched = next((token for token in tokens if token and token in text), None)
        if matched is not None:
            raise EncodingContractError(
                "assistant continuation contains a forbidden control boundary",
                code="inference.assistant_continuation_forbidden_control",
                context={
                    "example_id": example_id,
                    "boundary_class": boundary_class,
                    "token": matched,
                },
            )


def _validate_open_assistant_interval(
    processor: Any,
    *,
    interval: str,
    example_id: str,
    content_start: int,
) -> None:
    forbidden = {
        "assistant_terminator": (ASSISTANT_TERMINATOR,),
        "end_of_sequence": _end_of_sequence_tokens(processor),
        "chat_turn_opener": (CHAT_TURN_START,),
    }
    for boundary_class, tokens in forbidden.items():
        matched = next((token for token in tokens if token and token in interval), None)
        if matched is not None:
            raise EncodingContractError(
                "final open-assistant interval contains a forbidden boundary",
                code="inference.open_assistant_forbidden_boundary",
                context={
                    "example_id": example_id,
                    "boundary_class": boundary_class,
                    "token": matched,
                    "open_assistant_content_start_character": content_start,
                },
            )


def _end_of_sequence_tokens(processor: Any) -> tuple[str, ...]:
    tokenizer = getattr(processor, "tokenizer", None)
    authored = getattr(tokenizer, "eos_token", None)
    values = [*KNOWN_END_OF_SEQUENCE_TOKENS]
    if isinstance(authored, str) and authored:
        values.append(authored)
    return tuple(dict.fromkeys(values))


def _prompt_token_limit(
    processor: Any,
    *,
    explicit_limit: int | None,
    example_id: str,
) -> int | None:
    if explicit_limit is not None:
        if (
            not isinstance(explicit_limit, int)
            or isinstance(explicit_limit, bool)
            or explicit_limit <= 0
        ):
            raise EncodingContractError(
                "max_prompt_tokens must be a positive integer",
                code="inference.continuation_context_limit_invalid",
                context={
                    "example_id": example_id,
                    "max_prompt_tokens": explicit_limit,
                },
            )
        return int(explicit_limit)
    tokenizer = getattr(processor, "tokenizer", None)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(model_max_length, int)
        and not isinstance(model_max_length, bool)
        and 0 < model_max_length < 1_000_000_000
    ):
        return model_max_length
    if isinstance(model_max_length, float) and math.isfinite(model_max_length):
        if 0 < model_max_length < 1_000_000_000:
            return int(model_max_length)
    return None


def _longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    index = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        index += 1
    return index


def _full_prompt_fingerprint(chat_text: str, prompt_token_ids: list[int]) -> str:
    payload = {
        "full_chat_text": chat_text,
        "prompt_token_ids": prompt_token_ids,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_span(
    value: tuple[int, int] | None,
    *,
    field: str,
) -> tuple[int, int]:
    if value is None:
        raise EncodingContractError(
            "continued prompt record is missing required span evidence",
            code="inference.continuation_evidence_missing",
            context={"field": field},
        )
    return value
