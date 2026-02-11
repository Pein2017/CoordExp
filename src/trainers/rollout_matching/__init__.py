"""Public rollout-matching contracts.

This package provides stable, import-light helpers shared by rollout-matching SFT
and Stage-2 AB, without relying on trainer implementation modules.
"""

from .contracts import GTObject, GeomType, MatchResult, ParsedPredObject, RolloutParseResult
from .matching import hungarian_match_maskiou
from .packing import (
    AccumulationWindowLookahead,
    DropRemainderAccumulationWindow,
    RolloutMatchingPackWindow,
    WindowedMicroBatch,
)
from .parsing import (
    coerce_int,
    decode_pieces,
    find_desc_value_char_spans,
    find_desc_value_token_positions,
    parse_rollout_for_matching,
    points_from_coord_tokens,
    serialize_append_fragment,
)
from .telemetry import PendingTrainRolloutLog, slim_rollout_meta_for_logging

__all__ = [
    "AccumulationWindowLookahead",
    "DropRemainderAccumulationWindow",
    "RolloutMatchingPackWindow",
    "WindowedMicroBatch",
    "GeomType",
    "ParsedPredObject",
    "RolloutParseResult",
    "GTObject",
    "MatchResult",
    "decode_pieces",
    "parse_rollout_for_matching",
    "points_from_coord_tokens",
    "serialize_append_fragment",
    "find_desc_value_char_spans",
    "find_desc_value_token_positions",
    "coerce_int",
    "hungarian_match_maskiou",
    "slim_rollout_meta_for_logging",
    "PendingTrainRolloutLog",
]
