"""Qwen tokenizer identity checks for CoordExp-swift."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.common.errors import EncodingContractError
from src.common.qwen_aliases import (
    INVALID_QWEN_WRAPPER_ALIASES,
    find_invalid_qwen_wrapper_alias,
)


OBJECT_REF_START_TOKEN = "<|object_ref_start|>"
OBJECT_REF_END_TOKEN = "<|object_ref_end|>"
BOX_START_TOKEN = "<|box_start|>"
BOX_END_TOKEN = "<|box_end|>"
IM_END_TOKEN = "<|im_end|>"
IM_END_SUFFIX = f"{IM_END_TOKEN}\n"

DEFAULT_WRAPPER_TOKENS = (
    OBJECT_REF_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
)
DEFAULT_COORDINATE_TOKENS = tuple(f"<|coord_{index}|>" for index in range(1000))
DEFAULT_REQUIRED_TOKENS = (*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS)
INVALID_WRAPPER_ALIASES = INVALID_QWEN_WRAPPER_ALIASES


@dataclass(frozen=True)
class QwenTokenIdentity:
    """Validated identity facts for the CoordExp Qwen tokenizer."""

    required_tokens: tuple[str, ...]
    wrapper_token_ids: dict[str, int]
    coordinate_token_ids: tuple[int, ...]
    im_end_newline_text: str
    im_end_token_ids: tuple[int, ...]
    newline_token_ids: tuple[int, ...]
    im_end_newline_token_ids: tuple[int, ...]
    tokenizer_vocab_size: int

    def to_artifact_dict(self) -> dict[str, Any]:
        coord_min = min(self.coordinate_token_ids)
        coord_max = max(self.coordinate_token_ids)
        return {
            "required_token_count": len(self.required_tokens),
            "wrapper_token_ids": dict(self.wrapper_token_ids),
            "coord_token_count": len(self.coordinate_token_ids),
            "coord_token_id_min": coord_min,
            "coord_token_id_max": coord_max,
            "coord_token_ids_contiguous": self.coordinate_token_ids
            == tuple(range(coord_min, coord_max + 1)),
            "im_end_newline_text": self.im_end_newline_text,
            "im_end_token_ids": list(self.im_end_token_ids),
            "newline_token_ids": list(self.newline_token_ids),
            "im_end_newline_token_ids": list(self.im_end_newline_token_ids),
            "im_end_newline_split_verified": True,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
        }


def validate_qwen_token_identity(tokenizer: Any) -> QwenTokenIdentity:
    """Validate the loaded tokenizer before Qwen forward or optimizer setup."""

    reject_invalid_qwen_aliases(DEFAULT_REQUIRED_TOKENS, context_label="qwen.required_tokens")

    wrapper_token_ids = {
        token: _single_token_id(tokenizer, token)
        for token in DEFAULT_WRAPPER_TOKENS
    }
    coordinate_token_ids = tuple(
        _single_token_id(tokenizer, token)
        for token in DEFAULT_COORDINATE_TOKENS
    )
    _ensure_no_duplicate_ids(
        {
            **wrapper_token_ids,
            **{
                token: token_id
                for token, token_id in zip(
                    DEFAULT_COORDINATE_TOKENS,
                    coordinate_token_ids,
                    strict=True,
                )
            },
        }
    )

    im_end_token_ids = tuple(_encode_without_specials(tokenizer, IM_END_TOKEN))
    if len(im_end_token_ids) != 1:
        raise EncodingContractError(
            "<|im_end|> must tokenize as exactly one token",
            code="qwen.im_end_token_identity",
            context={"token": IM_END_TOKEN, "token_ids": list(im_end_token_ids)},
        )

    newline_token_ids = tuple(_encode_without_specials(tokenizer, "\n"))
    if len(newline_token_ids) != 1:
        raise EncodingContractError(
            "newline must tokenize as exactly one token",
            code="qwen.newline_token_identity",
            context={"text": "\\n", "token_ids": list(newline_token_ids)},
        )

    im_end_newline_token_ids = tuple(_encode_without_specials(tokenizer, IM_END_SUFFIX))
    expected = im_end_token_ids + newline_token_ids
    if im_end_newline_token_ids != expected:
        raise EncodingContractError(
            "<|im_end|>\\n must tokenize as <|im_end|> followed by a separate newline",
            code="qwen.im_end_newline_split",
            context={
                "im_end_token_ids": list(im_end_token_ids),
                "newline_token_ids": list(newline_token_ids),
                "im_end_newline_token_ids": list(im_end_newline_token_ids),
                "expected_token_ids": list(expected),
            },
        )

    return QwenTokenIdentity(
        required_tokens=DEFAULT_REQUIRED_TOKENS,
        wrapper_token_ids=wrapper_token_ids,
        coordinate_token_ids=coordinate_token_ids,
        im_end_newline_text=IM_END_SUFFIX,
        im_end_token_ids=im_end_token_ids,
        newline_token_ids=newline_token_ids,
        im_end_newline_token_ids=im_end_newline_token_ids,
        tokenizer_vocab_size=_tokenizer_vocab_size(tokenizer),
    )


def reject_invalid_qwen_aliases(
    tokens: Iterable[str] | str,
    *,
    context_label: str,
) -> None:
    """Reject known non-canonical wrapper aliases if config/template references them."""

    token_iterable = (tokens,) if isinstance(tokens, str) else tuple(tokens)
    for text in token_iterable:
        match = find_invalid_qwen_wrapper_alias(text)
        if match is not None:
            alias, canonical = match
            raise EncodingContractError(
                "invalid Qwen wrapper alias referenced",
                code="qwen.invalid_wrapper_alias",
                context={
                    "context_label": context_label,
                    "token": alias,
                    "canonical_token": canonical,
                },
            )


def _single_token_id(tokenizer: Any, token: str) -> int:
    reject_invalid_qwen_aliases(token, context_label="qwen.single_token")
    converted = tokenizer.convert_tokens_to_ids(token)
    encoded = _encode_without_specials(tokenizer, token)
    if converted is None or isinstance(converted, list):
        raise EncodingContractError(
            "required Qwen token is missing from tokenizer vocabulary",
            code="qwen.required_token_missing",
            context={"token": token, "converted_id": converted, "encoded_ids": encoded},
        )
    token_id = int(converted)
    if len(encoded) != 1 or int(encoded[0]) != token_id:
        raise EncodingContractError(
            "required Qwen token must be a single tokenizer atom",
            code="qwen.required_token_not_atomic",
            context={"token": token, "converted_id": token_id, "encoded_ids": encoded},
        )
    return token_id


def _encode_without_specials(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError as exc:
        raise EncodingContractError(
            "tokenizer.encode must support add_special_tokens=False",
            code="qwen.tokenizer_encode_contract",
            context={"text": text},
            cause=exc,
        ) from exc
    try:
        return [int(token_id) for token_id in encoded]
    except TypeError as exc:
        raise EncodingContractError(
            "tokenizer.encode returned a non-iterable token id payload",
            code="qwen.tokenizer_encode_shape",
            context={"text": text, "encoded": encoded},
            cause=exc,
        ) from exc


def _ensure_no_duplicate_ids(token_ids: dict[str, int]) -> None:
    seen: dict[int, str] = {}
    for token, token_id in token_ids.items():
        previous = seen.get(token_id)
        if previous is not None:
            raise EncodingContractError(
                "required Qwen tokens must not share token ids",
                code="qwen.required_token_duplicate_id",
                context={
                    "token": token,
                    "previous_token": previous,
                    "token_id": token_id,
                },
            )
        seen[token_id] = token


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    try:
        return int(len(tokenizer))
    except TypeError:
        value = getattr(tokenizer, "vocab_size", None)
        if value is None:
            raise EncodingContractError(
                "tokenizer vocabulary size is unavailable",
                code="qwen.tokenizer_vocab_size",
                context={"tokenizer_class": type(tokenizer).__name__},
            )
        return int(value)
