from __future__ import annotations

import pytest

from scripts.research.run_local_branch_causal_value import (
    INTERVENTION_ARMS,
    INTERVENTION_MODE,
    build_intervention_arms,
    append_row_if_complete,
    build_positive_entity_ledger,
    classify_greedy_state,
    deduplicate_prefix_records,
    downstream_outcome_bookkeeping,
    hash_prefix_token_ids,
    natural_action_record,
    select_verified_uncovered_owners,
    validate_intervention_arm_symmetry,
)


def _ledger(*statuses: str) -> list[dict[str, object]]:
    return [{"entity_id": f"o{index}", "verification": status} for index, status in enumerate(statuses)]


def test_exact_prefix_hash_and_dedup_preserve_provenance() -> None:
    records = deduplicate_prefix_records(
        [
            {"prefix_token_ids": [1, 2], "covered_entity_ids": ["o0"], "trajectory_provenance": {"trajectory_index": 0, "row_index": 1}, "natural_action": {"seed": 11, "raw_generated_token_ids": [3]}},
            {"prefix_token_ids": [1, 2], "covered_entity_ids": ["o0"], "trajectory_provenance": {"trajectory_index": 1, "row_index": 0}, "natural_action": {"seed": 12, "raw_generated_token_ids": [4]}},
            {"prefix_token_ids": [2, 1], "trajectory_provenance": {"trajectory_index": 2, "row_index": 0}},
        ]
    )
    assert len(records) == 2
    first = next(item for item in records if item["prefix_token_ids"] == [1, 2])
    assert len(first["trajectory_provenance"]) == 2
    assert [action["seed"] for action in first["natural_actions"]] == [11, 12]
    assert first["prefix_token_ids_sha256"] == hash_prefix_token_ids([1, 2])


def test_intervention_arms_have_noop_and_exposure_parity_contracts() -> None:
    arms = build_intervention_arms(
        parent_prefix_token_ids=[9, 8],
        shared_row_prefix_token_ids=[77],
        native_token_ids=[101],
        rescue_token_ids=[202],
        native_row_token_ids=[77, 101, 301, 302],
        rescue_row_token_ids=[77, 202, 401, 402],
    )
    assert tuple(arms) == INTERVENTION_ARMS
    assert arms["native"]["effective_prefix_token_ids"] == [9, 8]
    assert arms["force_native_token"]["effective_prefix_token_ids"] == [9, 8, 77, 101]
    assert arms["force_rescue_row"]["effective_prefix_token_ids"] == [9, 8, 77, 202, 401, 402]
    receipt = validate_intervention_arm_symmetry(arms)
    assert receipt["native_token_noop_expected"] is True
    assert receipt["native_row_noop_expected"] is True
    assert all(arms[name]["intervention_mode"] == INTERVENTION_MODE for name in INTERVENTION_ARMS)


def test_intervention_arm_rejects_multiple_token_forcing() -> None:
    with pytest.raises(ValueError, match="exactly one token"):
        build_intervention_arms(
            parent_prefix_token_ids=[1],
            shared_row_prefix_token_ids=[],
            native_token_ids=[2, 3],
            rescue_token_ids=[4],
            native_row_token_ids=[2, 3],
            rescue_row_token_ids=[4, 5],
        )


def test_dedup_refuses_exact_prefix_coverage_disagreement() -> None:
    records = deduplicate_prefix_records([
        {"prefix_token_ids": [1], "covered_entity_ids": ["o0"], "coverage_valid": True},
        {"prefix_token_ids": [1], "covered_entity_ids": ["o1"], "coverage_valid": True},
    ])
    assert records[0]["coverage_valid"] is False
    assert records[0]["coverage_refusal_reason"] == "exact_prefix_coverage_state_disagreement"


def test_natural_action_record_keeps_exact_emitted_row() -> None:
    action = natural_action_record(
        {
            "row_index": 2,
            "status": "success",
            "accepted_complete_row": True,
            "raw_generated_token_ids": [10, 11],
            "strict_matched_owner_ids": ["o1"],
        },
        trajectory_index=3,
        mode="sample",
        seed=14,
    )
    assert action["raw_generated_token_ids"] == [10, 11]
    assert action["strict_matched_owner_ids"] == ["o1"]
    assert action["seed"] == 14


def test_classification_refuses_unverified_or_unmatched_ownership() -> None:
    row = {"row_stop": {"stop_reason": "terminal"}, "strict_matched_owner_ids": []}
    result = classify_greedy_state(row, covered_entity_ids=["o0"], entity_ledger=_ledger("uncertain", "uncertain"))
    assert result["eligible_for_same_prefix_sampling"] is False
    assert result["refusal_reason"] == "no_verified_remaining_owner"


def test_terminal_with_verified_remaining_owner_is_eligible_but_not_unsupported() -> None:
    row = {"row_stop": {"stop_reason": "terminal"}, "strict_matched_owner_ids": []}
    result = classify_greedy_state(row, covered_entity_ids=["o0"], entity_ledger=_ledger("verified", "verified"))
    assert result["classification"] == "terminal_with_verified_remaining_owner"
    assert result["eligible_for_same_prefix_sampling"] is True
    assert result["verified_remaining_owner_ids"] == ["o1"]


def test_verified_uncovered_owner_selection_ignores_unmatched_predictions() -> None:
    row = {"strict_matched_owner_ids": ["o1", "unmatched-box"]}
    assert select_verified_uncovered_owners(row, covered_entity_ids=["o0"], entity_ledger=_ledger("verified", "approved", "uncertain")) == ["o1"]


def test_downstream_bookkeeping_separates_branch_and_future_rows() -> None:
    rows = [
        {"strict_matched_owner_ids": ["o1"]},
        {"strict_matched_owner_ids": ["o0", "o1"]},
        {"strict_matched_owner_ids": ["o2"]},
    ]
    result = downstream_outcome_bookkeeping(rows, covered_entity_ids=["o0"], branch_owner_ids=["o1"])
    assert result["branch_row_owner_ids"] == ["o1"]
    assert result["future_row_owner_ids"] == ["o0", "o1", "o2"]
    assert result["future_new_owner_ids"] == ["o2"]
    assert result["future_duplicate_owner_ids"] == ["o0"]
    assert result["future_branch_revisit_owner_ids"] == ["o1"]


def test_terminal_and_malformed_rows_are_not_appended() -> None:
    prefix = [1, 2]
    for reason in ("terminal", "malformed_limit", "contaminated_complete_row"):
        updated, receipt = append_row_if_complete(prefix, {"status": "success", "row_stop": {"stop_reason": reason}, "raw_generated_token_ids": [3]})
        assert updated == prefix
        assert receipt["appended"] is False


def test_positive_ledger_is_explicitly_positive_only() -> None:
    class Object:
        object_id = "person-1"
        description = "person"
        bbox = (10, 20, 30, 40)

    class Example:
        objects = (Object(),)

    ledger = build_positive_entity_ledger(Example())
    assert ledger == [{
        "entity_id": "person-1",
        "description": "person",
        "bbox_norm1000": [10.0, 20.0, 30.0, 40.0],
        "verification": "verified",
        "positive_only": True,
        "source": "selected_raw_example",
    }]
