from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.run_fixed_encoding_earlier_query_only_spatial_key_eligibility_factorial import (
    QUERY_RECEIPT_SHA256,
    assess_cross_receipt_baseline,
    build_earlier_query_only_key_eligibility_mask,
    compute_factorial,
    earlier_query_range,
    inspect_factorial_mask_structure,
    validate_query_receipt,
)


def test_earlier_range_ends_before_first_row_scoring_query() -> None:
    assert earlier_query_range(prefix_length=5) == (0, 3)
    with pytest.raises(ValueError):
        earlier_query_range(prefix_length=1)


def test_earlier_mask_changes_only_prefix_queries() -> None:
    mask = build_earlier_query_only_key_eligibility_mask(
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
    )[0, 0]
    for query_index in range(12):
        for key_index in (1, 2, 3):
            expected = key_index <= query_index and not (
                query_index <= 3 and key_index in {1, 3}
            )
            assert bool(mask[query_index, key_index]) is expected
    assert bool(mask[4, 1])


def test_structural_receipt_proves_disjoint_exact_union() -> None:
    mask = build_earlier_query_only_key_eligibility_mask(
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
    )
    receipt = inspect_factorial_mask_structure(
        mask,
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        row_length=4,
    )
    assert receipt["passed"]
    assert receipt["earlier_and_row_changed_sets_disjoint"]
    assert receipt["earlier_row_union_equals_all_query_on_scored_slice"]
    assert receipt["changed_row_scoring_cell_count"] == 0
    assert receipt["excluded_unscored_final_query"] == 8


def test_structural_receipt_rejects_row_query_corruption() -> None:
    mask = build_earlier_query_only_key_eligibility_mask(
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
    )
    mask[0, 0, 4, 1] = False
    receipt = inspect_factorial_mask_structure(
        mask,
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        row_length=4,
    )
    assert not receipt["passed"]
    assert receipt["changed_row_scoring_cell_count"] > 0


def test_structural_receipt_rejects_missing_earlier_cell() -> None:
    mask = build_earlier_query_only_key_eligibility_mask(
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
    )
    mask[0, 0, 3, 1] = True
    receipt = inspect_factorial_mask_structure(
        mask,
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        row_length=4,
    )
    assert not receipt["passed"]
    assert not receipt["exact_expected_earlier_mask"]


def _owner(mean: float) -> dict[str, dict[str, float]]:
    return {"full_row": {"mean": mean, "sum": mean, "count": 1}}


def _arm(target: float, competitor: float) -> dict:
    return {"target": _owner(target), "competitor": _owner(competitor)}


def test_factorial_arithmetic_decomposes_gamma_interaction() -> None:
    result = compute_factorial(
        baseline=_arm(2.0, 1.0),
        row_only={"target": _arm(4.0, 1.0), "competitor": _arm(1.0, 3.0)},
        earlier_only={"target": _arm(3.0, 1.0), "competitor": _arm(1.0, 2.0)},
        all_query={"target": _arm(8.0, 1.0), "competitor": _arm(1.0, 6.0)},
    )
    target = result["target"]["full_row"]
    assert target["gammas"] == {
        "unrestricted": 1.0,
        "row_only": 3.0,
        "earlier_only": 2.0,
        "all_query": 7.0,
    }
    assert target["gamma_interaction"] == pytest.approx(3.0)
    assert target["interaction_identity_error"] == pytest.approx(0.0)


def test_cross_receipt_baseline_checks_logprobs_and_ranks() -> None:
    frozen = {
        "target": {"token_log_probabilities": [1.0, 2.0], "selected_token_ranks": [1, 2]},
        "competitor": {"token_log_probabilities": [3.0], "selected_token_ranks": [4]},
    }
    live = {
        "target": {"token_log_probabilities": [1.0, 2.00001], "selected_token_ranks": [1, 2]},
        "competitor": {"token_log_probabilities": [3.0], "selected_token_ranks": [4]},
    }
    assert assess_cross_receipt_baseline(live=live, frozen=frozen)["passed"]
    live["target"]["selected_token_ranks"] = [1, 3]
    assert not assess_cross_receipt_baseline(live=live, frozen=frozen)["passed"]


def test_query_receipt_validator_accepts_frozen_artifact() -> None:
    path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/"
        "cohort-four-float32-20260715a/receipt.json"
    )
    payload, digest = validate_query_receipt(path)
    assert payload["unit_id"].endswith("eligibility-crossover")
    assert digest == QUERY_RECEIPT_SHA256


def test_query_receipt_digest_is_checked_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_query_receipt(path)


def test_earlier_mask_rejects_non_image_eligible_key() -> None:
    with pytest.raises(ValueError, match="subset"):
        build_earlier_query_only_key_eligibility_mask(
            sequence_length=8,
            image_key_positions=[1, 2],
            eligible_image_positions=[3],
            prefix_length=4,
        )
