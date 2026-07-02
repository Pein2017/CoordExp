"""Training-aligned prompt construction for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.errors import EncodingContractError
from src.config.models import TemplateConfig
from src.data import RawExample
from src.templates import render_example


TEMPLATE_ID = "coordexp-swift-template-v1"


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

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
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
        }


def build_prompt_record(
    raw_example: RawExample,
    template_config: TemplateConfig,
    *,
    processor: Any,
    row_index: int,
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
    prompt_token_ids = _tokenize_prompt(
        processor,
        chat_text=chat_text,
        messages=messages,
        example_id=raw_example.example_id,
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
    if not isinstance(encoded, dict) or "input_ids" not in encoded:
        raise EncodingContractError(
            "Qwen tokenizer did not return input_ids",
            code="inference.prompt_tokenizer_output",
            context={"example_id": example_id},
        )
    return _int_ids(encoded["input_ids"], example_id=example_id)


def _int_ids(value: Any, *, example_id: str) -> list[int]:
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
