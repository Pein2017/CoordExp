"""Deterministic bbox confidence helpers for offline confidence post-op.

The v1 contract computes one confidence score per bbox from 4 coord-token
log-probabilities:

    mean_logprob = mean(lp1, lp2, lp3, lp4)
    score = exp(mean_logprob)

Span matching is deterministic and auditable:
- exact subsequence matching on `generated_token_text`,
- assignment in object order,
- earliest unused match wins under ambiguity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Set, Tuple

from src.common.geometry import encode_coord

EXPECTED_BBOX_COORD_TOKEN_COUNT = 4


@dataclass(frozen=True)
class SpanMatch:
    """Token-index match metadata for one bbox."""

    matched_token_indices: Tuple[int, ...]
    ambiguous_matches: int


def coord_bins_to_tokens(coord_bins: Sequence[int]) -> Tuple[str, ...]:
    """Convert 4 coord-bin ints into canonical coord tokens."""

    if len(coord_bins) != EXPECTED_BBOX_COORD_TOKEN_COUNT:
        raise ValueError(
            "coord_bins must contain exactly 4 values (x1,y1,x2,y2), "
            f"got {len(coord_bins)}"
        )
    return tuple(encode_coord(int(value)) for value in coord_bins)


def find_exact_subsequence_candidates(
    tokens: Sequence[str],
    expected_subsequence: Sequence[str],
    *,
    min_start: int = 0,
    used_indices: Set[int] | None = None,
) -> list[Tuple[int, ...]]:
    """Return all unused exact-match candidate index tuples (stable order)."""

    if not expected_subsequence:
        return []

    n = len(expected_subsequence)
    if n > len(tokens):
        return []

    used = used_indices or set()
    max_start = len(tokens) - n
    candidates: list[Tuple[int, ...]] = []

    for start in range(max(0, int(min_start)), max_start + 1):
        end = start + n
        if list(tokens[start:end]) != list(expected_subsequence):
            continue
        indices = tuple(range(start, end))
        if any(index in used for index in indices):
            continue
        candidates.append(indices)

    return candidates


def assign_spans_left_to_right(
    *,
    generated_token_text: Sequence[str],
    expected_sequences: Sequence[Sequence[str]],
) -> list[SpanMatch | None]:
    """Assign subsequence spans deterministically in object order."""

    used: set[int] = set()
    cursor = 0
    out: list[SpanMatch | None] = []

    for expected in expected_sequences:
        candidates = find_exact_subsequence_candidates(
            generated_token_text,
            expected,
            min_start=cursor,
            used_indices=used,
        )
        if not candidates:
            out.append(None)
            continue

        chosen = candidates[0]
        used.update(chosen)
        cursor = max(cursor, chosen[-1] + 1)
        out.append(
            SpanMatch(
                matched_token_indices=chosen,
                ambiguous_matches=max(0, len(candidates) - 1),
            )
        )

    return out


def reduce_mean_logprob(logprobs: Sequence[float]) -> float:
    """Mean reducer for natural-log probabilities."""

    if not logprobs:
        raise ValueError("logprobs must be non-empty")

    acc = 0.0
    for value in logprobs:
        lp = float(value)
        if not math.isfinite(lp):
            raise ValueError(f"logprob must be finite, got {value!r}")
        acc += lp
    return acc / float(len(logprobs))


def map_exp(mean_logprob: float) -> float:
    """Map mean log-probability to bounded probability score."""

    mean_lp = float(mean_logprob)
    if not math.isfinite(mean_lp):
        raise ValueError(f"mean_logprob must be finite, got {mean_logprob!r}")
    return float(math.exp(mean_lp))


def compute_bbox_confidence_from_logprobs(logprobs: Sequence[float]) -> float:
    """Compute `exp(mean_logprob)` for a bbox coord-token logprob sequence."""

    return map_exp(reduce_mean_logprob(logprobs))


def is_valid_confidence_score(score: float) -> bool:
    """Return True iff score satisfies the v1 kept-object range contract."""

    value = float(score)
    return math.isfinite(value) and 0.0 < value <= 1.0


__all__ = [
    "EXPECTED_BBOX_COORD_TOKEN_COUNT",
    "SpanMatch",
    "assign_spans_left_to_right",
    "compute_bbox_confidence_from_logprobs",
    "coord_bins_to_tokens",
    "find_exact_subsequence_candidates",
    "is_valid_confidence_score",
    "map_exp",
    "reduce_mean_logprob",
]
