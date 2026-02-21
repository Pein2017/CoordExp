from __future__ import annotations

import math

import pytest

from src.eval.bbox_confidence import (
    EXPECTED_BBOX_COORD_TOKEN_COUNT,
    assign_spans_left_to_right,
    compute_bbox_confidence_from_logprobs,
    coord_bins_to_tokens,
    is_valid_confidence_score,
    map_exp,
    reduce_mean_logprob,
)


def test_coord_bins_to_tokens_requires_exactly_four_values() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        coord_bins_to_tokens([1, 2, 3])

    assert coord_bins_to_tokens([1, 2, 3, 4]) == (
        "<|coord_1|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    )


def test_mean_logprob_and_exp_mapping() -> None:
    logprobs = [-0.1, -0.2, -0.3, -0.4]
    mean_logprob = reduce_mean_logprob(logprobs)
    expected_mean = sum(logprobs) / float(len(logprobs))
    assert mean_logprob == pytest.approx(expected_mean)

    score = map_exp(mean_logprob)
    assert score == pytest.approx(math.exp(expected_mean))
    assert compute_bbox_confidence_from_logprobs(logprobs) == pytest.approx(score)


def test_confidence_score_validation_contract() -> None:
    assert is_valid_confidence_score(1.0)
    assert is_valid_confidence_score(0.25)
    assert not is_valid_confidence_score(0.0)
    assert not is_valid_confidence_score(-0.1)
    assert not is_valid_confidence_score(1.1)
    assert not is_valid_confidence_score(float("nan"))
    assert not is_valid_confidence_score(float("inf"))


def test_repeated_sequences_choose_earliest_unused_match() -> None:
    seq = coord_bins_to_tokens([10, 20, 30, 40])
    generated_token_text = list(seq) + ["<|im_end|>"] + list(seq)

    assignments = assign_spans_left_to_right(
        generated_token_text=generated_token_text,
        expected_sequences=[seq, seq],
    )

    assert len(assignments) == 2
    assert assignments[0] is not None
    assert assignments[1] is not None
    assert assignments[0].matched_token_indices == (0, 1, 2, 3)
    assert assignments[0].ambiguous_matches == 1
    assert assignments[1].matched_token_indices == (5, 6, 7, 8)
    assert assignments[1].ambiguous_matches == 0


def test_missing_span_returns_none_assignment() -> None:
    seq = coord_bins_to_tokens([100, 200, 300, 400])
    generated_token_text = list(seq)

    assignments = assign_spans_left_to_right(
        generated_token_text=generated_token_text,
        expected_sequences=[seq, seq],
    )

    assert len(assignments) == 2
    assert assignments[0] is not None
    assert assignments[1] is None
    assert EXPECTED_BBOX_COORD_TOKEN_COUNT == 4


def test_subsequence_matching_allows_separator_tokens_between_coords() -> None:
    seq = coord_bins_to_tokens([704, 284, 724, 338])
    generated_token_text = [
        '{"',
        "objects",
        '":',
        "[",
        seq[0],
        ",",
        " ",
        seq[1],
        ",",
        " ",
        seq[2],
        ",",
        " ",
        seq[3],
        "]",
    ]

    assignments = assign_spans_left_to_right(
        generated_token_text=generated_token_text,
        expected_sequences=[seq],
    )

    assert len(assignments) == 1
    assert assignments[0] is not None
    assert assignments[0].matched_token_indices == (4, 7, 10, 13)
