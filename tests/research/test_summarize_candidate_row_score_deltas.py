from __future__ import annotations

import copy

from scripts.research.summarize_candidate_row_score_deltas import _alternate_favor, _rank_candidates


def _candidate(candidate_id: str, score: float) -> dict:
    values = {"sum": score, "mean": score}
    return {
        "candidate_id": candidate_id,
        "full_row": values,
        "description": copy.deepcopy(values),
        "x1": copy.deepcopy(values),
        "y1": copy.deepcopy(values),
        "x2": copy.deepcopy(values),
        "y2": copy.deepcopy(values),
        "geometry": copy.deepcopy(values),
        "closure": copy.deepcopy(values),
    }


def test_alternate_owner_check_reports_relative_change_and_rank() -> None:
    left = {"left": _candidate("left", -4.0), "right": _candidate("right", -6.0)}
    right = {"left": _candidate("left", -5.0), "right": _candidate("right", -4.0)}
    result = _alternate_favor(left, right, _rank_candidates(left), _rank_candidates(right), "left", "right")
    assert result["status"] == "available"
    assert result["favors_observed_alternate_owner"] is True
    assert result["observed_owner_ranks"]["left_prefix"] == {"left_owner": 1, "right_owner": 2}
    assert result["observed_owner_ranks"]["right_prefix"] == {"left_owner": 2, "right_owner": 1}


def test_alternate_owner_check_leaves_fallback_owner_ambiguous() -> None:
    candidates = {"left": _candidate("left", -4.0), "fallback": _candidate("fallback", -6.0)}
    result = _alternate_favor(candidates, candidates, _rank_candidates(candidates), _rank_candidates(candidates), "left", None)
    assert result == {"status": "ambiguous", "favors_observed_alternate_owner": None}
