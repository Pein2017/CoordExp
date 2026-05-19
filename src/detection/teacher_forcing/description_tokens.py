"""Context-aware description token paths for compact_full teacher forcing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.detection_compact_rows import (
    BOX_START_TOKEN,
    COMPACT_DESC_FORBIDDEN_SUBSTRINGS,
    OBJECT_REF_START_TOKEN,
    STRICT_COMPACT_ROW_COORD_TOKEN_RE,
)


class DescriptionTokenizationError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DescriptionTokenPath:
    object_ref_start_token_id: int
    description_token_ids: tuple[int, ...]
    bbox_start_token_id: int
    context_token_ids: tuple[int, ...]


def tokenize_description_context(desc: str, tokenizer: Any) -> DescriptionTokenPath:
    """Tokenize ``<object_ref>{desc}<box_start>`` and validate marker boundaries."""

    desc_text = validate_description_text(desc)
    context = f"{OBJECT_REF_START_TOKEN}{desc_text}{BOX_START_TOKEN}"
    encoded = _tokenize_with_offsets(tokenizer, context)
    input_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
    offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])

    object_ref_span = (0, len(OBJECT_REF_START_TOKEN))
    box_start_span = (
        len(OBJECT_REF_START_TOKEN) + len(desc_text),
        len(context),
    )
    object_ref_indices = _token_indices_for_exact_span(offsets, object_ref_span)
    box_start_indices = _token_indices_for_exact_span(offsets, box_start_span)
    if len(object_ref_indices) != 1 or len(box_start_indices) != 1:
        raise DescriptionTokenizationError(
            "non_unique_marker_boundary",
            "compact_full marker must map to exactly one context token",
        )

    object_ref_index = object_ref_indices[0]
    box_start_index = box_start_indices[0]
    if object_ref_index >= box_start_index:
        raise DescriptionTokenizationError(
            "non_unique_marker_boundary",
            "compact_full marker token order is invalid",
        )

    desc_start = len(OBJECT_REF_START_TOKEN)
    desc_end = desc_start + len(desc_text)
    desc_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if start >= desc_start and end <= desc_end and start < end
    ]
    if any(
        start < desc_start or end > desc_end
        for index, (start, end) in enumerate(offsets)
        if object_ref_index < index < box_start_index
    ):
        raise DescriptionTokenizationError(
            "non_unique_marker_boundary",
            "description token crosses compact_full marker boundary",
        )
    if not desc_indices:
        raise DescriptionTokenizationError(
            "empty_description",
            "object desc must produce at least one text token",
        )

    return DescriptionTokenPath(
        object_ref_start_token_id=input_ids[object_ref_index],
        description_token_ids=tuple(input_ids[index] for index in desc_indices),
        bbox_start_token_id=input_ids[box_start_index],
        context_token_ids=input_ids,
    )


def validate_description_text(desc: str) -> str:
    if not isinstance(desc, str):
        raise DescriptionTokenizationError("invalid_description", "object desc must be a string")
    if not desc.strip():
        raise DescriptionTokenizationError("empty_description", "object desc must be non-empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in desc):
        raise DescriptionTokenizationError(
            "invalid_description",
            "object desc must not contain newline, tab, or control characters",
        )
    for forbidden in COMPACT_DESC_FORBIDDEN_SUBSTRINGS:
        if forbidden in desc:
            raise DescriptionTokenizationError(
                "invalid_description",
                "object desc contains reserved compact_full marker text",
            )
    if "<|image" in desc or "<|vision" in desc:
        raise DescriptionTokenizationError(
            "invalid_description",
            "object desc contains reserved image marker text",
        )
    if STRICT_COMPACT_ROW_COORD_TOKEN_RE.search(desc):
        raise DescriptionTokenizationError(
            "invalid_description",
            "object desc contains coordinate-token text",
        )
    return desc


def _tokenize_with_offsets(tokenizer: Any, text: str) -> dict[str, Any]:
    if callable(tokenizer):
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        if "input_ids" in encoded and "offset_mapping" in encoded:
            return encoded
    raise DescriptionTokenizationError(
        "missing_offsets",
        "tokenizer must expose context-aware offset mappings",
    )


def _token_indices_for_exact_span(
    offsets: tuple[tuple[int, int], ...],
    span: tuple[int, int],
) -> list[int]:
    return [index for index, offset in enumerate(offsets) if offset == span]


__all__ = [
    "DescriptionTokenPath",
    "DescriptionTokenizationError",
    "tokenize_description_context",
    "validate_description_text",
]
