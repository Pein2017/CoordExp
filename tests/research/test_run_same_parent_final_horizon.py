from pathlib import Path

from scripts.research import run_same_parent_final_horizon as stage4


ADMISSION = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-19-sampled-history-target-reachability-and-complete-row-value/"
    "stage4-final-horizon-admission.json"
)


def _row(token_id: int) -> dict:
    return {
        "raw_generated_token_ids": [token_id],
        "status": "success",
        "row_stop": {"stop_reason": "complete_row"},
        "parse_evidence": {"parse_status": "accepted"},
    }


def _summary(*, terminal: bool = True, exhausted: bool = False, duplicates=()) -> dict:
    return {
        "natural_terminal_row_index": 10 if terminal else None,
        "budget_exhausted": exhausted,
        "unresolved_row_indices": [],
        "malformed_row_indices": [],
        "crop_review_warnings": [],
        "duplicate_owner_ids": list(duplicates),
    }


def test_frozen_admission_counts_branch_inside_total_budget() -> None:
    frozen = stage4.load_stage_four_source(ADMISSION)
    contract = frozen["admission"]["execution_contract"]
    assert contract["branch_row_token_count"] == 9
    assert contract["post_branch_generated_token_budget"] == 503
    assert contract["total_trajectory_generated_token_budget"] == 512


def test_initial_and_full_row_parity_are_distinct() -> None:
    frozen = [_row(1), _row(2)]
    replay = [_row(1), _row(2), _row(3)]
    assert stage4.compare_initial_rows(frozen, replay)["passed"] is True
    assert stage4.compare_all_rows(frozen, replay)["passed"] is False
    replay[1]["raw_generated_token_ids"] = [9]
    assert stage4.compare_initial_rows(frozen, replay)["passed"] is False


def test_final_gain_requires_target_superset_and_no_sampled_duplicate() -> None:
    result = stage4.classify_final_comparison(
        target_owner_id="target",
        native_summary=_summary(duplicates=("old",)),
        sampled_summary=_summary(),
        native_owner_ids=["old"],
        sampled_owner_ids=["old", "target"],
        all_identity_and_parity_gates_passed=True,
    )
    assert result["classification"] == "sampled_final_gain"
    assert result["claim_allowed"] is True

    duplicate = stage4.classify_final_comparison(
        target_owner_id="target",
        native_summary=_summary(),
        sampled_summary=_summary(duplicates=("target",)),
        native_owner_ids=["old"],
        sampled_owner_ids=["old", "target"],
        all_identity_and_parity_gates_passed=True,
    )
    assert duplicate["classification"] == "unresolved"


def test_final_exchange_and_same_set_are_separate() -> None:
    exchange = stage4.classify_final_comparison(
        target_owner_id="target",
        native_summary=_summary(),
        sampled_summary=_summary(),
        native_owner_ids=["old"],
        sampled_owner_ids=["target"],
        all_identity_and_parity_gates_passed=True,
    )
    assert exchange["classification"] == "sampled_final_exchange"
    assert exchange["lost_owner_ids"] == ["old"]

    same = stage4.classify_final_comparison(
        target_owner_id="target",
        native_summary=_summary(),
        sampled_summary=_summary(),
        native_owner_ids=["old"],
        sampled_owner_ids=["old"],
        all_identity_and_parity_gates_passed=True,
    )
    assert same["classification"] == "same_final_set_or_timing"


def test_censoring_and_failed_gate_refuse_final_value() -> None:
    censored = stage4.classify_final_comparison(
        target_owner_id="target",
        native_summary=_summary(terminal=False, exhausted=True),
        sampled_summary=_summary(),
        native_owner_ids=["old"],
        sampled_owner_ids=["old", "target"],
        all_identity_and_parity_gates_passed=True,
    )
    assert censored["classification"] == "right_censored"
    assert censored["claim_allowed"] is False

    terminal_on_last_token = stage4.classify_final_comparison(
        target_owner_id="target",
        native_summary=_summary(exhausted=True),
        sampled_summary=_summary(exhausted=True),
        native_owner_ids=["old"],
        sampled_owner_ids=["old", "target"],
        all_identity_and_parity_gates_passed=True,
    )
    assert terminal_on_last_token["classification"] == "sampled_final_gain"

    failed_gate = stage4.classify_final_comparison(
        target_owner_id="target",
        native_summary=_summary(),
        sampled_summary=_summary(),
        native_owner_ids=["old"],
        sampled_owner_ids=["old", "target"],
        all_identity_and_parity_gates_passed=False,
    )
    assert failed_gate["classification"] == "unresolved"
    assert failed_gate["claim_allowed"] is False
