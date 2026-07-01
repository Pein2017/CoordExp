"""Coord token utilities (codec, validation, template adapters, losses)."""

from .codec import (
    COORD_TOKEN_PATTERN,
    build_coord_token_id_mask,
    get_coord_token_ids,
    ints_to_tokens,
    is_coord_token,
    normalized_from_ints,
    sequence_has_coord_tokens,
    token_to_int,
    tokens_to_ints,
    value_in_coord_range,
)

__all__ = [
    "COORD_TOKEN_PATTERN",
    "build_coord_token_id_mask",
    "get_coord_token_ids",
    "ints_to_tokens",
    "is_coord_token",
    "normalized_from_ints",
    "sequence_has_coord_tokens",
    "token_to_int",
    "tokens_to_ints",
    "value_in_coord_range",
]
