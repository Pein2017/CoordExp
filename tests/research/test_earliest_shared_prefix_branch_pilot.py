from __future__ import annotations

from scripts.research.run_earliest_shared_prefix_branch_pilot import (
    _can_attach_reviewed_owner,
    _decoder_acquisition_eligible,
    _harm_receipt,
    _owner_set,
)


def _row(*owners: str, status: str = "success", stop: str = "complete_row") -> dict:
    return {
        "status": status,
        "row_stop": {"stop_reason": stop},
        "strict_matched_owner_ids": [owner for owner in owners if ":" not in owner],
        "reviewed_owner_ids_qualified": [owner for owner in owners if ":" in owner],
    }


def test_forced_context_and_released_suffix_are_disjoint_in_the_partition() -> None:
    context = [_row("left")]
    current = [_row("target")]
    suffix = [_row("right")]
    assert _owner_set(context, 1) == {"1:left"}
    assert _owner_set(current, 1) == {"1:target"}
    assert _owner_set(suffix, 1) == {"1:right"}
    assert not (_owner_set(context, 1) & _owner_set(suffix, 1))


def test_harm_receipt_reports_repeats_and_does_not_call_unknown_hallucinations() -> None:
    rows = [
        _row("same"),
        _row("same"),
        _row(status="success", stop="terminal"),
        _row(status="failed", stop="malformed"),
    ]
    receipt = _harm_receipt(rows, 7)
    assert receipt["duplicate_owner_ids"] == ["7:same"]
    assert receipt["duplicate_owner_count"] == 1
    assert receipt["unresolved_row_count"] == 1
    assert receipt["malformed_row_count"] == 1


def test_reviewed_owner_cannot_leak_onto_a_complete_non_target_row() -> None:
    sampled_row = [151646, 8987, 151647, 151648, 152100, 152200, 152300, 152400, 151649]
    assert not _can_attach_reviewed_owner(
        row_index=0,
        target_row_index=4,
        current_row_is_complete=True,
        current_row_token_ids=sampled_row,
        sampled_row_token_ids=sampled_row,
    )
    assert _can_attach_reviewed_owner(
        row_index=4,
        target_row_index=4,
        current_row_is_complete=True,
        current_row_token_ids=sampled_row,
        sampled_row_token_ids=sampled_row,
    )


def test_reviewed_owner_is_allowed_when_a_partial_rung_reconstructs_target_row() -> None:
    sampled_row = [151646, 8987, 151647, 151648, 152100, 152200, 152300, 152400, 151649]
    assert _can_attach_reviewed_owner(
        row_index=4,
        target_row_index=4,
        current_row_is_complete=True,
        current_row_token_ids=sampled_row,
        sampled_row_token_ids=sampled_row,
    )
    assert not _can_attach_reviewed_owner(
        row_index=4,
        target_row_index=4,
        current_row_is_complete=False,
        current_row_token_ids=sampled_row[:-1],
        sampled_row_token_ids=sampled_row,
    )


def test_fully_injected_target_is_not_decoder_acquisition_evidence() -> None:
    assert not _decoder_acquisition_eligible(
        row_index=4,
        target_row_index=4,
        current_row_is_complete=True,
        released_tail_token_count=0,
        target_owner_in_intervened_row=True,
    )
    assert _decoder_acquisition_eligible(
        row_index=4,
        target_row_index=4,
        current_row_is_complete=True,
        released_tail_token_count=1,
        target_owner_in_intervened_row=True,
    )
