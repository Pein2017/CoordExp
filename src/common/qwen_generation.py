"""Qwen chat-template generation token contract helpers."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
import inspect
from typing import Any

from src.common.detection_sequence import END_OF_TEXT_TOKEN, IM_END_TOKEN


@dataclass(frozen=True)
class QwenChatGenerationTokenIds:
    eos_token_id: int
    pad_token_id: int


def resolve_qwen_chat_generation_token_ids(
    tokenizer: object,
) -> QwenChatGenerationTokenIds:
    """Resolve the global HF generate EOS/PAD ids for Qwen chat detection."""

    im_end_id = _resolve_single_token_id(tokenizer, IM_END_TOKEN)
    endoftext_id = _resolve_single_token_id(tokenizer, END_OF_TEXT_TOKEN)
    return QwenChatGenerationTokenIds(
        eos_token_id=im_end_id,
        pad_token_id=endoftext_id,
    )


def apply_qwen_chat_generation_token_ids(
    generation_kwargs: MutableMapping[str, Any],
    *,
    tokenizer: object,
) -> None:
    """Mutate HF generation kwargs to use Qwen chat EOS and text PAD ids."""

    token_ids = resolve_qwen_chat_generation_token_ids(tokenizer)
    generation_kwargs["eos_token_id"] = token_ids.eos_token_id
    generation_kwargs["pad_token_id"] = token_ids.pad_token_id


def qwen_processor_call_kwargs() -> dict[str, bool]:
    """Processor kwargs required to preserve dataset-time image geometry."""

    return {"do_resize": False}


def call_processor_with_qwen_geometry(processor: object, **kwargs: Any) -> Any:
    """Call an HF processor with `do_resize=False` when the callable accepts it."""

    call = getattr(processor, "__call__", None)
    if callable(call) and _callable_accepts_keyword(call, "do_resize"):
        kwargs = dict(kwargs)
        kwargs.update(qwen_processor_call_kwargs())
    return processor(**kwargs)  # type: ignore[misc]


def _callable_accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.kind in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        } and parameter.name == keyword:
            return True
    return False


def _resolve_single_token_id(tokenizer: object, token_text: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError(f"tokenizer must support convert_tokens_to_ids for {token_text}")
    token_id = convert(token_text)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError(f"tokenizer could not resolve required token {token_text}")
    if (
        isinstance(unk_token_id, int)
        and not isinstance(unk_token_id, bool)
        and int(token_id) == int(unk_token_id)
    ):
        raise ValueError(f"tokenizer resolved required token {token_text} to unk_token_id")

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        encoded = encode(token_text, add_special_tokens=False)
        if list(encoded) != [int(token_id)]:
            raise ValueError(
                f"required token {token_text} must encode as one tokenizer token"
            )
    return int(token_id)


__all__ = [
    "QwenChatGenerationTokenIds",
    "apply_qwen_chat_generation_token_ids",
    "call_processor_with_qwen_geometry",
    "qwen_processor_call_kwargs",
    "resolve_qwen_chat_generation_token_ids",
]
