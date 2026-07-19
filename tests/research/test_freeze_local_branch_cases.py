from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.freeze_local_branch_cases import (
    FreezeValidationError,
    clean_single_owner,
    derive_prefix_candidates,
    longest_common_prefix_and_difference,
    select_one_case_per_image,
    token_ids_hash,
)


def _ledger() -> list[dict[str, object]]:
    return [
        {"entity_id": "covered", "description": "person", "verification": "verified"},
        {"entity_id": "uncovered", "description": "person", "verification": "approved"},
        {"entity_id": "other", "description": "chair", "verification": "verified"},
    ]


def _row(ids: list[int], owner: str, *, unmatched: bool = False) -> dict[str, object]:
    matches: list[dict[str, object]] = [{"status": "matched", "matched_entity_id": owner}]
    if unmatched:
        matches.append({"status": "unmatched", "matched_entity_id": None})
    return {
        "status": "success",
        "accepted_complete_row": True,
        "row_stop": {"stop_reason": "complete_row"},
        "raw_generated_token_ids": ids,
        "entity_matches": matches,
        "strict_matched_owner_ids": [owner],
    }


def _prefix_evaluation(*, native: dict[str, object], sampled: dict[str, object], seed: int = 11) -> dict[str, object]:
    prefix_ids = [9, 8]
    prefix = {
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": token_ids_hash(prefix_ids),
        "covered_entity_ids": ["covered"],
        "coverage_valid": True,
        "trajectory_provenance": [
            {"trajectory_index": 0, "row_index": 1, "mode": "greedy", "seed": None},
            {"trajectory_index": 1, "row_index": 1, "mode": "sample", "seed": seed},
        ],
        "natural_actions": [{
            "trajectory_index": 1,
            "row_index": 1,
            "mode": "sample",
            "seed": seed,
            **sampled,
        }],
    }
    return {
        "image_id": "2299",
        "prefix": prefix,
        "native_greedy_row": native,
        "classification": {"classification": "verified_duplicate"},
    }


def test_longest_common_prefix_and_first_difference_uses_raw_ids() -> None:
    result = longest_common_prefix_and_difference([1, 2, 3, 4], [1, 2, 9, 4])
    assert result["shared_row_prefix_token_ids"] == [1, 2]
    assert result["first_difference_index"] == 2
    assert result["native_branch_token_id"] == 3
    assert result["sampled_branch_token_id"] == 9


def test_equal_rows_and_length_mismatch_are_refused() -> None:
    with pytest.raises(FreezeValidationError, match="identical_rows"):
        longest_common_prefix_and_difference([1], [1])
    with pytest.raises(FreezeValidationError, match="row_length_mismatch"):
        longest_common_prefix_and_difference([1, 2], [1])


def test_clean_single_owner_rejects_unmatched_companion_prediction() -> None:
    accepted = clean_single_owner(_row([1, 2], "covered"), covered_entity_ids=["covered"], entity_ledger=_ledger())
    assert accepted["accepted"] is True
    rejected = clean_single_owner(_row([1, 2], "covered", unmatched=True), covered_entity_ids=["covered"], entity_ledger=_ledger())
    assert rejected["refusal_code"] == "row_has_unmatched_or_ambiguous"


def test_native_replay_without_sample_acceptance_bit_remains_admissible() -> None:
    native = _row([1, 2], "covered")
    native.pop("accepted_complete_row")
    accepted = clean_single_owner(
        native,
        covered_entity_ids=["covered"],
        entity_ledger=_ledger(),
    )
    assert accepted["accepted"] is True


def test_explicit_false_acceptance_bit_is_still_rejected() -> None:
    incomplete = _row([1, 2], "covered")
    incomplete["accepted_complete_row"] = False
    rejected = clean_single_owner(
        incomplete,
        covered_entity_ids=["covered"],
        entity_ledger=_ledger(),
    )
    assert rejected["refusal_code"] == "row_not_complete"


def test_clean_single_owner_rejects_two_owner_row() -> None:
    row = _row([1, 2], "covered")
    row["entity_matches"] = [
        {"status": "matched", "matched_entity_id": "covered"},
        {"status": "matched", "matched_entity_id": "uncovered"},
    ]
    row["strict_matched_owner_ids"] = ["covered", "uncovered"]
    assert clean_single_owner(row, covered_entity_ids=["covered"], entity_ledger=_ledger())["refusal_code"] == "owner_count_not_one"


def test_primary_pair_requires_same_category_and_equal_row_length() -> None:
    native = _row([1, 2, 3, 4], "covered")
    sampled = _row([1, 2, 9, 4], "uncovered")
    candidates, refusals = derive_prefix_candidates(_prefix_evaluation(native=native, sampled=sampled), entity_ledger=_ledger())
    assert len(candidates) == 1
    assert not refusals
    assert candidates[0]["shared_row_prefix_token_ids"] == [1, 2]
    assert candidates[0]["native_root_greedy_reachable"] is True

    chair_sample = _row([1, 2, 9, 4], "other")
    candidates, refusals = derive_prefix_candidates(_prefix_evaluation(native=native, sampled=chair_sample), entity_ledger=_ledger())
    assert not candidates
    assert any(item["refusal_code"] == "category_mismatch" for item in refusals)

    short_sample = _row([1, 2, 9], "uncovered")
    candidates, refusals = derive_prefix_candidates(_prefix_evaluation(native=native, sampled=short_sample), entity_ledger=_ledger())
    assert not candidates
    assert any(item["refusal_code"] == "row_length_mismatch" for item in refusals)


def test_sampled_owner_already_covered_is_not_a_primary_case() -> None:
    native = _row([1, 2, 3], "covered")
    sampled = _row([1, 2, 9], "covered")
    candidates, refusals = derive_prefix_candidates(_prefix_evaluation(native=native, sampled=sampled), entity_ledger=_ledger())
    assert not candidates
    assert any(item["refusal_code"] == "sampled_owner_already_covered" for item in refusals)


def test_sampled_only_prefix_is_not_root_reachable() -> None:
    native = _row([1, 2, 3], "covered")
    sampled = _row([1, 2, 9], "uncovered")
    evaluation = _prefix_evaluation(native=native, sampled=sampled)
    evaluation["prefix"]["trajectory_provenance"] = [{"trajectory_index": 2, "row_index": 1, "mode": "sample", "seed": 12}]
    evaluation["prefix"]["natural_actions"][0]["trajectory_index"] = 2
    evaluation["prefix"]["natural_actions"][0]["seed"] = 12
    candidates, _ = derive_prefix_candidates(evaluation, entity_ledger=_ledger())
    assert len(candidates) == 1
    assert candidates[0]["native_root_greedy_reachable"] is False


def test_selection_is_deterministic_and_one_case_per_image() -> None:
    base = {
        "image_id": "2299",
        "natural_row_index": 3,
        "natural_sample_seed": 12,
        "parent_prefix_token_ids_sha256": "b",
    }
    earlier = {**base, "natural_row_index": 1, "parent_prefix_token_ids_sha256": "z"}
    tie_seed = {**base, "natural_sample_seed": 11, "parent_prefix_token_ids_sha256": "z"}
    assert select_one_case_per_image([base, earlier, tie_seed]) == earlier


def test_unmatched_evidence_is_not_negative_or_unsupported() -> None:
    native = _row([1, 2, 3], "covered")
    sampled = _row([1, 2, 9], "uncovered", unmatched=True)
    candidates, refusals = derive_prefix_candidates(_prefix_evaluation(native=native, sampled=sampled), entity_ledger=_ledger())
    assert not candidates
    assert any(item["outcome"] == "unresolved_refusal" for item in refusals)
    assert not any(item["outcome"] == "verified_unsupported" for item in refusals)


def test_terminal_and_malformed_remain_diagnostics() -> None:
    native = {
        "status": "success",
        "accepted_complete_row": False,
        "row_stop": {"stop_reason": "terminal"},
        "raw_generated_token_ids": [],
        "entity_matches": [],
        "strict_matched_owner_ids": [],
    }
    sampled = _row([1, 2, 9], "uncovered")
    evaluation = _prefix_evaluation(native=native, sampled=sampled)
    evaluation["classification"] = {"classification": "terminal_with_verified_remaining_owner"}
    candidates, refusals = derive_prefix_candidates(evaluation, entity_ledger=_ledger())
    assert not candidates
    assert any(item["outcome"] == "terminal_diagnostic" for item in refusals)


def test_prefix_hash_mismatch_is_refused() -> None:
    native = _row([1, 2, 3], "covered")
    sampled = _row([1, 2, 9], "uncovered")
    evaluation = _prefix_evaluation(native=native, sampled=sampled)
    evaluation["prefix"]["prefix_token_ids_sha256"] = "wrong"
    candidates, refusals = derive_prefix_candidates(evaluation, entity_ledger=_ledger())
    assert not candidates
    assert refusals[0]["refusal_code"] == "prefix_hash_mismatch"
