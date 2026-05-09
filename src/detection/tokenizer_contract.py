from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


_IM_END_TOKEN = "<|im_end|>"


@dataclass(frozen=True)
class CompactTrainingStopContract:
    im_end_token_id: int
    training_eos_token_text: Literal["<|im_end|>"]
    training_eos_token_ids: frozenset[int]
    tokenizer_eos_token_text: str | None = None
    tokenizer_eos_token_id: int | None = None


def resolve_compact_training_stop_contract(
    tokenizer: object,
) -> CompactTrainingStopContract:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError("compact_full requires tokenizer.convert_tokens_to_ids")

    im_end_id = convert(_IM_END_TOKEN)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if (
        not isinstance(im_end_id, int)
        or isinstance(im_end_id, bool)
        or im_end_id < 0
    ):
        raise ValueError("compact_full requires im_end to resolve to one token id")
    if isinstance(unk_token_id, int) and int(im_end_id) == int(unk_token_id):
        raise ValueError("compact_full requires im_end to not resolve to unk")

    if not _tokenizer_declares_im_end(tokenizer):
        raise ValueError(
            "compact_full requires <|im_end|> in tokenizer vocab or special tokens"
        )

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        encoded = encode(_IM_END_TOKEN, add_special_tokens=False)
        if list(encoded) != [int(im_end_id)]:
            raise ValueError(
                "compact_full requires <|im_end|> to encode as a single token"
            )

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_text = eos_token if isinstance(eos_token, str) else None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_text:
        resolved_eos_id = convert(eos_token_text)
        if isinstance(resolved_eos_id, int) and not isinstance(resolved_eos_id, bool):
            eos_token_id = int(resolved_eos_id)

    _validate_closed_assistant_chat_template_uses_im_end(
        tokenizer,
        im_end_token_id=int(im_end_id),
    )

    return CompactTrainingStopContract(
        im_end_token_id=int(im_end_id),
        training_eos_token_text=_IM_END_TOKEN,
        training_eos_token_ids=frozenset({int(im_end_id)}),
        tokenizer_eos_token_text=eos_token_text,
        tokenizer_eos_token_id=eos_token_id if isinstance(eos_token_id, int) else None,
    )


def _tokenizer_declares_im_end(tokenizer: object) -> bool:
    vocab = _call_mapping(tokenizer, "get_vocab")
    added_vocab = _call_mapping(tokenizer, "get_added_vocab")
    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    if _IM_END_TOKEN in vocab or _IM_END_TOKEN in added_vocab:
        return True
    return _mapping_contains_value(special_tokens_map, _IM_END_TOKEN)


def _call_mapping(tokenizer: object, method_name: str) -> Mapping[str, Any]:
    method = getattr(tokenizer, method_name, None)
    if not callable(method):
        return {}
    value = method()
    return value if isinstance(value, Mapping) else {}


def _mapping_contains_value(value: object, expected: str) -> bool:
    if isinstance(value, Mapping):
        return any(_mapping_contains_value(item, expected) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_mapping_contains_value(item, expected) for item in value)
    return value == expected


def _validate_closed_assistant_chat_template_uses_im_end(
    tokenizer: object,
    *,
    im_end_token_id: int,
) -> None:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return

    messages = [{"role": "assistant", "content": "payload"}]
    text = apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(text, str):
        raise ValueError("tokenizer.apply_chat_template(..., tokenize=False) must return str")
    if "payload" not in text:
        raise ValueError("compact_full chat template must include assistant payload")
    payload_end = _find_payload_end_bounded_by_im_end(text, payload="payload")
    if payload_end is None:
        raise ValueError("compact_full chat template must place <|im_end|> after payload")
    _validate_im_end_token_delta(
        tokenizer,
        text=text,
        payload_end=payload_end,
        im_end_token_id=im_end_token_id,
    )

    encoded = apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise ValueError("tokenizer.apply_chat_template(..., tokenize=True) must return ids")
    encoded_ids = [int(token_id) for token_id in encoded]
    if int(im_end_token_id) not in encoded_ids:
        raise ValueError(
            "compact_full chat template must emit an <|im_end|> token id"
        )


def _find_payload_end_bounded_by_im_end(text: str, *, payload: str) -> int | None:
    search_start = 0
    payload_ends: list[int] = []
    while True:
        payload_start = text.find(payload, search_start)
        if payload_start < 0:
            break
        payload_end = payload_start + len(payload)
        if text.startswith(_IM_END_TOKEN, payload_end):
            payload_ends.append(payload_end)
        search_start = payload_start + 1
    return payload_ends[-1] if payload_ends else None


def _validate_im_end_token_delta(
    tokenizer: object,
    *,
    text: str,
    payload_end: int,
    im_end_token_id: int,
) -> None:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return

    prefix_ids = list(encode(text[:payload_end], add_special_tokens=False))
    with_stop_ids = list(
        encode(
            text[: payload_end + len(_IM_END_TOKEN)],
            add_special_tokens=False,
        )
    )
    if with_stop_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("compact_full chat template must preserve prefix tokenization")
    delta = with_stop_ids[len(prefix_ids) :]
    if delta != [int(im_end_token_id)]:
        raise ValueError(
            "compact_full chat template must encode assistant <|im_end|> as one token"
        )
