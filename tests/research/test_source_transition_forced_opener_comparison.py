from __future__ import annotations

from scripts.research.summarize_source_transition_forced_opener_comparison import (
    _aggregate,
    _exact_two_sided_sign_test,
)


def _release(outcome: str, owners: list[str]) -> dict[str, object]:
    return {
        "outcome": outcome,
        "verified_uncovered_owner_ids": owners,
    }


def _case(source: list[str], transition: list[str]) -> dict[str, object]:
    gained = sorted(set(transition) - set(source))
    lost = sorted(set(source) - set(transition))
    return {
        "source_native": _release("immediate_terminal", []),
        "transition_native": _release("immediate_terminal", []),
        "source_forced": _release(
            "verified_uncovered_owner" if source else "valid_unmatched_or_ambiguous",
            source,
        ),
        "transition_forced": _release(
            "verified_uncovered_owner" if transition else "valid_unmatched_or_ambiguous",
            transition,
        ),
        "checkpoint_gained_owner_ids": gained,
        "checkpoint_lost_owner_ids": lost,
        "checkpoint_retained_owner_ids": sorted(set(source) & set(transition)),
        "checkpoint_owner_net": len(gained) - len(lost),
        "forced_raw_row_token_ids_equal": source == transition,
    }


def test_exact_sign_test_is_two_sided() -> None:
    result = _exact_two_sided_sign_test(3, 0)
    assert result["discordant"] == 3
    assert result["p_value"] == 0.25


def test_aggregate_preserves_owner_exchange() -> None:
    result = _aggregate(
        [
            _case([], ["a"]),
            _case(["b"], []),
            _case(["c"], ["d"]),
            _case(["e"], ["e"]),
        ]
    )
    assert result["checkpoint_gained_owner_count"] == 2
    assert result["checkpoint_lost_owner_count"] == 2
    assert result["checkpoint_owner_net"] == 0
    assert result["owner_exchange_boundary_count"] == 1
    assert result["positive_net_boundary_count"] == 1
    assert result["negative_net_boundary_count"] == 1


def test_aggregate_uses_paired_any_owner_transitions() -> None:
    result = _aggregate([_case([], ["a"]), _case(["b"], ["b"]), _case([], [])])
    assert result["source_forced_any_uncovered_owner"]["success_count"] == 1
    assert result["transition_forced_any_uncovered_owner"]["success_count"] == 2
    assert result["forced_any_uncovered_owner_transitions"] == {
        "source_False_transition_False": 1,
        "source_False_transition_True": 1,
        "source_True_transition_True": 1,
    }
