from __future__ import annotations

from scripts.research.build_earliest_shared_prefix_branch_cases import (
    _staircase_for_row,
    longest_common_prefix_length,
    row_spans,
)


def test_longest_common_prefix_is_exact_and_does_not_decode_text() -> None:
    assert longest_common_prefix_length([1, 2, 3, 4], [1, 2, 9]) == 2
    assert longest_common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert longest_common_prefix_length([], [1]) == 0


def test_row_spans_refuse_incomplete_tail() -> None:
    assert row_spans([151646, 8987, 151647, 151648, 151700, 151701, 151702, 151703, 151649]) == [(0, 9)]


def test_two_level_staircase_starts_at_first_divergence_and_target_row() -> None:
    first = [151646, 8987, 151647, 151648, 151729, 151785, 151887, 151923, 151649]
    target = first + [151646, 8987, 151647, 151648, 151729, 151785, 151887, 151923, 151649]
    rungs = [
        *_staircase_for_row(first, 0, len(first), 4, include_level_a=True, include_level_b=False),
        *_staircase_for_row(target, 9, 18, 4, include_level_a=False, include_level_b=True),
    ]
    assert rungs[0]["rung_name"] == "level_a_first_divergence"
    assert rungs[0]["endpoint_role"] == "x1"
    assert rungs[0]["forced_row_prefix_token_ids"] == first[:5]
    assert rungs[-1]["rung_name"] == "level_b_box_end"
    assert all(rung["level"] in {"A", "B"} for rung in rungs)
