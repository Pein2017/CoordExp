"""Training-aligned prompt construction for inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.common.errors import EncodingContractError
from src.config.models import TemplateConfig
from src.data import RawExample
from src.templates import render_example


TEMPLATE_ID = "coordexp-swift-template-v1"
IMAGE_PAD_TOKEN = "<|image_pad|>"


@dataclass(frozen=True)
class PromptRecord:
    row_id: str
    row_index: int
    example_id: str
    messages: tuple[dict[str, Any], ...]
    prompt_text: str
    chat_text: str
    input_prompt_token_ids: list[int]
    expected_executed_prompt_token_ids: list[int]
    template_id: str
    template_fingerprint: str
    object_ordering: str
    object_field_order: str
    assistant_format: str
    realized_object_order: list[dict[str, int | str]]

    @property
    def prompt_token_ids(self) -> list[int]:
        """Backward-compatible artifact prompt ids: the executed prompt form."""

        return self.expected_executed_prompt_token_ids

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "row_index": self.row_index,
            "example_id": self.example_id,
            "prompt_text": self.prompt_text,
            "chat_text": self.chat_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_token_count": len(self.prompt_token_ids),
            "input_prompt_token_ids": list(self.input_prompt_token_ids),
            "input_prompt_token_count": len(self.input_prompt_token_ids),
            "expected_executed_prompt_token_ids": list(
                self.expected_executed_prompt_token_ids
            ),
            "expected_executed_prompt_token_count": len(
                self.expected_executed_prompt_token_ids
            ),
            "template_id": self.template_id,
            "template_fingerprint": self.template_fingerprint,
            "object_ordering": self.object_ordering,
            "object_field_order": self.object_field_order,
            "assistant_format": self.assistant_format,
            "realized_object_order": list(self.realized_object_order),
        }


def build_prompt_record(
    raw_example: RawExample,
    template_config: TemplateConfig,
    *,
    processor: Any,
    row_index: int,
    merged_visual_tokens: int,
    object_order_seed: int | None = None,
) -> PromptRecord:
    rendered = render_example(
        raw_example,
        template_config,
        object_order_seed=object_order_seed,
    )
    messages = tuple(message for message in rendered.messages if message.get("role") != "assistant")
    chat_text = _apply_generation_chat_template(
        processor,
        messages=messages,
        example_id=raw_example.example_id,
    )
    input_prompt_token_ids = _tokenize_prompt(
        processor,
        chat_text=chat_text,
        example_id=raw_example.example_id,
    )
    expected_executed_prompt_token_ids = _expand_image_placeholder(
        tokenizer=processor.tokenizer,
        input_prompt_token_ids=input_prompt_token_ids,
        merged_visual_tokens=merged_visual_tokens,
        example_id=raw_example.example_id,
    )
    return PromptRecord(
        row_id=raw_example.example_id,
        row_index=row_index,
        example_id=raw_example.example_id,
        messages=messages,
        prompt_text=rendered.prompt_text,
        chat_text=chat_text,
        input_prompt_token_ids=input_prompt_token_ids,
        expected_executed_prompt_token_ids=expected_executed_prompt_token_ids,
        template_id=TEMPLATE_ID,
        template_fingerprint=rendered.template_fingerprint,
        object_ordering=rendered.object_ordering,
        object_field_order=template_config.object_field_order,
        assistant_format=template_config.assistant_format,
        realized_object_order=[
            item.to_artifact_dict() for item in rendered.realized_object_order
        ],
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
    example_id: str,
) -> list[int]:
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


def _expand_image_placeholder(
    *,
    tokenizer: Any,
    input_prompt_token_ids: list[int],
    merged_visual_tokens: int,
    example_id: str,
) -> list[int]:
    if isinstance(merged_visual_tokens, bool) or not isinstance(
        merged_visual_tokens, int
    ) or merged_visual_tokens <= 0:
        raise EncodingContractError(
            "merged visual token count must be a positive integer",
            code="inference.prompt_visual_token_count",
            context={
                "example_id": example_id,
                "merged_visual_tokens": merged_visual_tokens,
            },
        )
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert_tokens_to_ids):
        raise EncodingContractError(
            "Qwen tokenizer does not expose image placeholder token lookup",
            code="inference.prompt_image_token_lookup",
            context={"example_id": example_id},
        )
    image_token_id = convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if image_token_id is None:
        raise EncodingContractError(
            "Qwen tokenizer is missing <|image_pad|>",
            code="inference.prompt_image_token_missing",
            context={"example_id": example_id},
        )
    image_token_id = int(image_token_id)
    image_indices = [
        index
        for index, token_id in enumerate(input_prompt_token_ids)
        if token_id == image_token_id
    ]
    if len(image_indices) != 1:
        raise EncodingContractError(
            "inference input prompt must contain exactly one image placeholder",
            code="inference.prompt_image_token_count",
            context={
                "example_id": example_id,
                "image_placeholder_count": len(image_indices),
            },
        )
    image_index = image_indices[0]
    return [
        *input_prompt_token_ids[:image_index],
        *([image_token_id] * merged_visual_tokens),
        *input_prompt_token_ids[image_index + 1 :],
    ]
