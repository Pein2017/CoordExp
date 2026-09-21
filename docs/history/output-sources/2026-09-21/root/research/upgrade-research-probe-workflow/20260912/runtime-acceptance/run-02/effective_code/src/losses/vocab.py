"""Closed V1 token-type vocabulary groups."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from src.common.errors import LossContractError
from src.qwen.tokens import QwenTokenIdentity
from src.supervision import TokenAtom


TokenType = Literal["desc_text", "schema", "coordinate", "eos"]
V1_TOKEN_TYPES: tuple[str, ...] = ("desc_text", "schema", "coordinate", "eos")

KNOWN_CONTROL_TOKENS = (
    "<|im_start|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|coord_*|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
)


@dataclass(frozen=True)
class TokenVocabularyGroups:
    vocab_size: int
    desc_text: tuple[int, ...]
    schema: tuple[int, ...]
    coordinate: tuple[int, ...]
    eos: tuple[int, ...]
    blocked: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "desc_text", _normalize_ids(self.desc_text))
        object.__setattr__(self, "schema", _normalize_ids(self.schema))
        object.__setattr__(self, "coordinate", _normalize_ids(self.coordinate))
        object.__setattr__(self, "eos", _normalize_ids(self.eos))
        object.__setattr__(self, "blocked", _normalize_ids(self.blocked))
        if self.vocab_size <= 0:
            raise LossContractError(
                "token vocabulary size must be positive",
                code="loss.vocab_size",
                context={"vocab_size": self.vocab_size},
            )
        for group_name in V1_TOKEN_TYPES:
            token_ids = self.allowed_ids(group_name)
            if not token_ids:
                raise LossContractError(
                    "V1 token vocabulary groups must be non-empty",
                    code="loss.vocab_group_empty",
                    context={"token_type": group_name},
                )
            _validate_ids_in_vocab(token_ids, vocab_size=self.vocab_size, group_name=group_name)
        _validate_ids_in_vocab(self.blocked, vocab_size=self.vocab_size, group_name="blocked")
        _validate_no_target_overlap(self)
        object.__setattr__(
            self,
            "_membership",
            {
                "desc_text": frozenset(self.desc_text),
                "schema": frozenset(self.schema),
                "coordinate": frozenset(self.coordinate),
                "eos": frozenset(self.eos),
                "blocked": frozenset(self.blocked),
            },
        )

    def allowed_ids(self, token_type: str) -> tuple[int, ...]:
        if token_type == "desc_text":
            return self.desc_text
        if token_type == "schema":
            return self.schema
        if token_type == "coordinate":
            return self.coordinate
        if token_type == "eos":
            return self.eos
        raise LossContractError(
            "unknown V1 token type",
            code="loss.token_type_unknown",
            context={"token_type": token_type, "known_token_types": list(V1_TOKEN_TYPES)},
        )

    def validate_atom(self, atom: TokenAtom) -> None:
        try:
            allowed = self.allowed_ids(atom.token_type)
        except LossContractError as exc:
            raise LossContractError(
                "TokenAtom token type must be a closed V1 token type",
                code=exc.code,
                context={
                    "token_type": atom.token_type,
                    "known_token_types": list(V1_TOKEN_TYPES),
                    **_atom_context(atom),
                },
                cause=exc,
            ) from exc
        if not self.contains(atom.token_type, atom.token_id):
            raise LossContractError(
                "TokenAtom target id must belong to its declared token-type group",
                code="loss.token_id_outside_group",
                context={
                    "example_id": atom.example_id,
                    "target_position": atom.target_position,
                    "token_type": atom.token_type,
                    "token_id": atom.token_id,
                    "allowed_count": len(allowed),
                    "source": atom.source,
                    "field": atom.field,
                    "object_id": atom.object_id,
                },
            )

    def contains(self, token_type: str, token_id: int) -> bool:
        try:
            membership = self._membership[token_type]
        except KeyError:
            allowed = self.allowed_ids(token_type)
            return _contains_sorted(allowed, token_id)
        return int(token_id) in membership

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "desc_text_count": len(self.desc_text),
            "schema": list(self.schema),
            "coordinate_count": len(self.coordinate),
            "coordinate_min": min(self.coordinate),
            "coordinate_max": max(self.coordinate),
            "eos": list(self.eos),
            "blocked_count": len(self.blocked),
        }


def build_token_vocabulary_groups(
    token_identity: QwenTokenIdentity,
    *,
    tokenizer: Any,
    extra_blocked_token_ids: Iterable[int] = (),
) -> TokenVocabularyGroups:
    if tokenizer is None:
        raise LossContractError(
            "token vocabulary group resolution requires a tokenizer",
            code="loss.vocab_tokenizer_required",
            context={},
        )
    vocab_size = int(token_identity.tokenizer_vocab_size)
    schema = tuple(token_identity.wrapper_token_ids.values())
    coordinate = tuple(token_identity.coordinate_token_ids)
    eos = tuple(token_identity.im_end_token_ids)
    blocked = set(int(token_id) for token_id in extra_blocked_token_ids)
    blocked.update(int(token_id) for token_id in token_identity.newline_token_ids)
    blocked.update(_tokenizer_special_ids(tokenizer))
    blocked.update(_known_control_token_ids(tokenizer))
    blocked.update(schema)
    blocked.update(coordinate)
    blocked.update(eos)
    target_or_blocked = set(blocked)
    desc_text = tuple(
        token_id
        for token_id in range(vocab_size)
        if token_id not in target_or_blocked
    )
    return TokenVocabularyGroups(
        vocab_size=vocab_size,
        desc_text=desc_text,
        schema=schema,
        coordinate=coordinate,
        eos=eos,
        blocked=tuple(blocked - set(schema) - set(coordinate) - set(eos)),
    )


def _normalize_ids(token_ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(token_id) for token_id in token_ids}))


def _validate_ids_in_vocab(
    token_ids: Iterable[int],
    *,
    vocab_size: int,
    group_name: str,
) -> None:
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise LossContractError(
                "token vocabulary group id is outside vocab range",
                code="loss.vocab_group_id",
                context={
                    "group_name": group_name,
                    "token_id": token_id,
                    "vocab_size": vocab_size,
                },
            )


def _validate_no_target_overlap(groups: TokenVocabularyGroups) -> None:
    owners: dict[int, str] = {}
    for group_name in V1_TOKEN_TYPES:
        for token_id in groups.allowed_ids(group_name):
            previous = owners.get(token_id)
            if previous is not None:
                raise LossContractError(
                    "V1 token vocabulary target groups must be disjoint",
                    code="loss.vocab_group_overlap",
                    context={
                        "token_id": token_id,
                        "first_group": previous,
                        "second_group": group_name,
                    },
                )
            owners[token_id] = group_name
    for token_id in groups.blocked:
        previous = owners.get(token_id)
        if previous is not None:
            raise LossContractError(
                "blocked token ids must not overlap V1 target vocabulary groups",
                code="loss.vocab_group_overlap",
                context={
                    "token_id": token_id,
                    "first_group": previous,
                    "second_group": "blocked",
                },
            )


def _contains_sorted(token_ids: tuple[int, ...], token_id: int) -> bool:
    index = bisect_left(token_ids, int(token_id))
    return index < len(token_ids) and token_ids[index] == int(token_id)


def _atom_context(atom: TokenAtom) -> dict[str, int | str | None]:
    return {
        "example_id": atom.example_id,
        "pack_index": atom.pack_index,
        "segment_index": atom.segment_index,
        "target_position": atom.target_position,
        "logical_target_position": atom.logical_target_position,
        "logical_base_token_start": atom.logical_base_token_start,
        "logical_base_token_end": atom.logical_base_token_end,
        "object_id": atom.object_id,
        "field": atom.field,
        "source": atom.source,
    }


def _tokenizer_special_ids(tokenizer: Any | None) -> tuple[int, ...]:
    if tokenizer is None:
        return ()
    value = getattr(tokenizer, "all_special_ids", ())
    try:
        return tuple(int(token_id) for token_id in value)
    except TypeError:
        return ()


def _known_control_token_ids(tokenizer: Any | None) -> tuple[int, ...]:
    if tokenizer is None:
        return ()
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(converter):
        return ()
    token_ids: list[int] = []
    for token in KNOWN_CONTROL_TOKENS:
        value = converter(token)
        if value is None:
            continue
        if isinstance(value, list):
            token_ids.extend(int(token_id) for token_id in value)
        else:
            token_ids.append(int(value))
    return tuple(token_ids)


__all__ = [
    "KNOWN_CONTROL_TOKENS",
    "TokenType",
    "TokenVocabularyGroups",
    "V1_TOKEN_TYPES",
    "build_token_vocabulary_groups",
]
