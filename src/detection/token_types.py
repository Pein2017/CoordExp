"""Compact detection token-type groups for schema-aware loss terms."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.common.detection_sequence import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)

_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d{1,3})\|>$")
_CONTROL_SPECIAL_RE = re.compile(r"^<\|[^|]+\|>$")
_TEXT_TERMINATORS = ("<|endoftext|>", "<|end_of_text|>")
_CONTROL_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    *_TEXT_TERMINATORS,
)
_COMPACT_STRUCT_TOKENS = (
    OBJECT_REF_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
)
_COMPACT_SEPARATOR_TOKENS = ("\n", "\r\n")


@dataclass(frozen=True)
class CompactTokenTypeGroups:
    struct: frozenset[int]
    coord: frozenset[int]
    eos: frozenset[int]
    excluded_control: frozenset[int]
    desc: frozenset[int]


def build_compact_token_type_groups(tokenizer: object) -> CompactTokenTypeGroups:
    """Build mutually exclusive compact token groups from tokenizer vocabulary."""

    vocab = _tokenizer_vocab(tokenizer)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    struct = set(_resolve_known_tokens(tokenizer, _COMPACT_STRUCT_TOKENS, unk_token_id))
    struct.update(_resolve_known_tokens(tokenizer, _COMPACT_SEPARATOR_TOKENS, unk_token_id))
    coord = {
        token_id
        for token, token_id in vocab.items()
        if _is_coord_token(token) and _is_valid_token_id(token_id, unk_token_id)
    }
    # Some lightweight test tokenizers do not materialize all coord tokens in vocab.
    for coord_index in range(1000):
        token_id = _convert_token(tokenizer, f"<|coord_{coord_index}|>")
        if _is_valid_token_id(token_id, unk_token_id):
            coord.add(int(token_id))

    im_end_id = _convert_token(tokenizer, "<|im_end|>")
    if not _is_valid_token_id(im_end_id, unk_token_id):
        raise ValueError("compact token type groups require <|im_end|>")
    eos = {int(im_end_id)}

    excluded_control = set(_resolve_known_tokens(tokenizer, _CONTROL_SPECIAL_TOKENS, unk_token_id))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int) and not isinstance(pad_token_id, bool):
        excluded_control.add(int(pad_token_id))
    if isinstance(unk_token_id, int) and not isinstance(unk_token_id, bool):
        excluded_control.add(int(unk_token_id))

    for token, token_id in vocab.items():
        if not _is_valid_token_id(token_id, unk_token_id):
            continue
        token_id = int(token_id)
        if token_id in struct or token_id in coord or token_id in eos:
            continue
        if _CONTROL_SPECIAL_RE.match(str(token)) is not None:
            excluded_control.add(token_id)

    reserved = struct | coord | eos | excluded_control
    desc = {
        int(token_id)
        for token_id in vocab.values()
        if _is_valid_token_id(token_id, unk_token_id) and int(token_id) not in reserved
    }
    return CompactTokenTypeGroups(
        struct=frozenset(struct),
        coord=frozenset(coord),
        eos=frozenset(eos),
        excluded_control=frozenset(excluded_control),
        desc=frozenset(desc),
    )


def allowed_type_token_ids_for_target(
    target: object,
    groups: CompactTokenTypeGroups,
) -> frozenset[int]:
    """Expand a target's positive token ids to the union of their compact types."""

    raw_positive_ids = getattr(target, "positive_token_ids", None)
    if raw_positive_ids is None:
        raw_positive_ids = getattr(target, "valid_token_ids", None)
    if raw_positive_ids is None:
        raw_positive_ids = (getattr(target, "teacher_token_id", None),)
    positive_token_ids = tuple(
        int(token_id) for token_id in raw_positive_ids if token_id is not None
    )
    if not positive_token_ids:
        raise ValueError("target must expose at least one positive token id")

    allowed: set[int] = set()
    for token_id in positive_token_ids:
        group = _group_for_token_id(token_id, groups)
        if not group:
            raise ValueError(
                f"positive token id {token_id} does not belong to any compact token type"
            )
        allowed.update(group)
    if not allowed:
        raise ValueError("positive token ids do not belong to any compact token type")
    return frozenset(allowed)


def combine_main_and_type_losses(
    *,
    main_loss: Any,
    type_loss: Any,
    type_weight: float,
):
    return main_loss + float(type_weight) * type_loss


def _group_for_token_id(
    token_id: int,
    groups: CompactTokenTypeGroups,
) -> frozenset[int]:
    if token_id in groups.struct:
        return groups.struct
    if token_id in groups.coord:
        return groups.coord
    if token_id in groups.eos:
        return groups.eos
    if token_id in groups.desc:
        return groups.desc
    return frozenset()


def _tokenizer_vocab(tokenizer: object) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("compact token type groups require tokenizer.get_vocab")
    raw_vocab = get_vocab()
    if not isinstance(raw_vocab, dict):
        raise ValueError("tokenizer.get_vocab must return a dict")
    return {str(token): int(token_id) for token, token_id in raw_vocab.items()}


def _resolve_known_tokens(
    tokenizer: object,
    tokens: tuple[str, ...],
    unk_token_id: object,
) -> tuple[int, ...]:
    token_ids: list[int] = []
    for token in tokens:
        token_id = _convert_token(tokenizer, token)
        if _is_valid_token_id(token_id, unk_token_id):
            token_ids.append(int(token_id))
    return tuple(token_ids)


def _convert_token(tokenizer: object, token: str) -> object:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    return convert(token)


def _is_valid_token_id(token_id: object, unk_token_id: object) -> bool:
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        return False
    if isinstance(unk_token_id, int) and not isinstance(unk_token_id, bool):
        return int(token_id) != int(unk_token_id)
    return True


def _is_coord_token(token: str) -> bool:
    match = _COORD_TOKEN_RE.match(str(token))
    return match is not None and 0 <= int(match.group(1)) <= 999


__all__ = [
    "CompactTokenTypeGroups",
    "allowed_type_token_ids_for_target",
    "build_compact_token_type_groups",
    "combine_main_and_type_losses",
]
