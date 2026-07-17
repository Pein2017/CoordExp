from __future__ import annotations

import math

import pytest

from scripts.research import run_donor_state_late_coordinate_transport_image7818 as probe


def _score(values: list[float]) -> dict:
    return {"token_log_probabilities": [-9.0, -9.0, -9.0, *values, -0.1]}


def test_candidate_histories_are_the_two_frozen_native_x1_y1_paths() -> None:
    histories = probe.candidate_common_histories()
    assert histories == {
        "target_x1_y1": probe.parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS[5:7],
        "paired_x1_y1": probe.parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS[5:7],
    }


def test_make_row_keeps_wrapper_and_coordinate_order() -> None:
    row = probe.make_row([probe.parent.COORDINATE_TOKEN_START + 1, probe.parent.COORDINATE_TOKEN_START + 2], [probe.parent.COORDINATE_TOKEN_START + 3, probe.parent.COORDINATE_TOKEN_START + 4])
    assert row[:5] == probe.parent.shared_row_prefix(
        probe.parent.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS,
        probe.parent.PAIRED_REALIZED_DONOR_ROW_TOKEN_IDS,
    )
    assert row[-5:-1] == [probe.parent.COORDINATE_TOKEN_START + 1, probe.parent.COORDINATE_TOKEN_START + 2, probe.parent.COORDINATE_TOKEN_START + 3, probe.parent.COORDINATE_TOKEN_START + 4]
    assert row[-1] == probe.parent.BOX_END


def test_native_support_requires_both_donor_branches_within_log_ten() -> None:
    result = probe.native_support_admission(
        candidate_scores={
            "target_x1_y1": {"target": -1.0, "paired": -1.5},
            "paired_x1_y1": {"target": -3.0, "paired": -1.2},
        },
        native_scores={"target": -1.0, "paired": -1.2},
    )
    assert result["admitted_histories"] == ["paired_x1_y1", "target_x1_y1"]
    assert result["histories"]["target_x1_y1"]["passed"]
    # A gap of log(10) is accepted; a larger gap is not.
    result = probe.native_support_admission(
        candidate_scores={"bad": {"target": -1.0 - math.log(10.0) - 1e-6, "paired": -1.0}},
        native_scores={"target": -1.0, "paired": -1.0},
    )
    assert result["admitted_histories"] == []
    with pytest.raises(ValueError, match="exactly target and paired"):
        probe.native_support_admission(
            candidate_scores={"incomplete": {"target": -1.0}},
            native_scores={"target": -1.0, "paired": -1.0},
        )


def test_late_score_and_transport_contrast_use_only_x2_y2() -> None:
    row = probe.make_row([1, 2], [3, 4])
    score = {"token_log_probabilities": [-7.0] * len(row)}
    # parent._coordinate_log_probs uses the four slots immediately before BOX_END.
    start = len(row) - 5
    score["token_log_probabilities"][start : start + 4] = [-1.0, -2.0, -3.0, -4.0]
    assert probe.early_coordinate_score(score, row) == -3.0
    assert probe.late_coordinate_score(score, row) == -7.0
    assert probe.donor_transport_contrast(target_margin_under_target=0.7, target_margin_under_paired=0.1) == 0.6


def test_noop_epsilon_uses_raw_coordinate_slots_not_margin_cancellation() -> None:
    epsilon = probe.compute_noop_epsilon(
        unrestricted_coordinate_log_probabilities={"target": [-1.0, -2.0, -3.0, -4.0], "paired": [-2.0, -3.0, -4.0, -5.0]},
        noop_coordinate_log_probabilities={"target": [-1.0, -2.0, -2.8, -4.0], "paired": [-2.0, -3.0, -4.2, -5.0]},
    )
    assert epsilon == pytest.approx(0.2)
    assert max(0.05, 5.0 * epsilon) == pytest.approx(1.0)


def test_recaptured_state_identity_requires_both_donors_and_exact_boundary() -> None:
    expected = {
        "states": {
            "23:target": {"state_sha256": "t", "token_history_sha256": "h", "position_ids_sha256": "p", "boundary_pos": 7},
            "23:paired": {"state_sha256": "q", "token_history_sha256": "h", "position_ids_sha256": "p", "boundary_pos": 7},
        }
    }
    receipts = {
        "target": {"state_sha256": "t"},
        "paired": {"state_sha256": "q"},
    }
    passed = probe.validate_recaptured_state_identity(
        layer_idx=23,
        donor_receipts=receipts,
        expected_receipt=expected,
        boundary_pos=7,
        position_ids_sha256="p",
        token_history_sha256="h",
    )
    assert passed["passed"]
    failed = probe.validate_recaptured_state_identity(
        layer_idx=23,
        donor_receipts=receipts,
        expected_receipt=expected,
        boundary_pos=8,
        position_ids_sha256="p",
        token_history_sha256="h",
    )
    assert not failed["passed"]


def test_completed_portability_receipt_is_frozen_and_contains_all_four_states() -> None:
    receipt = probe.validate_parent_portability_state_receipt()
    assert receipt["sha256"] == probe.PARENT_PORTABILITY_RECEIPT_SHA256
    assert set(receipt["states"]) == {f"{layer}:{owner}" for layer in ("23", "13") for owner in ("target", "paired")}


def test_crossed_owner_gate_rejects_invalid_and_ambiguous_boxes() -> None:
    accepted = [
        {"annotation_id": "A", "normalized_box": [0.1, 0.1, 0.4, 0.4]},
        {"annotation_id": "B", "normalized_box": [0.6, 0.1, 0.9, 0.4]},
    ]
    parsed = {
        "target_x1_y1": {
            "target": {"valid": True, "normalized_box": [0.1, 0.1, 0.4, 0.4]},
            "paired": {"valid": True, "normalized_box": [0.6, 0.1, 0.9, 0.4]},
        },
        "paired_x1_y1": {
            "target": {"valid": False, "normalized_box": [0.8, 0.8, 0.2, 0.2]},
            "paired": {"valid": True, "normalized_box": [0.1, 0.1, 0.4, 0.4]},
        },
    }
    result = probe.classify_crossed_history_owners(parsed_by_history=parsed, accepted_objects=accepted)
    assert result["admissible_histories"] == ["target_x1_y1"]
    assert result["physical_owner_claim_permitted"]
    assert result["per_history"]["paired_x1_y1"]["reason"] == "invalid_or_ambiguous_crossed_box"

    conjoined = probe.conjoin_owner_gate_with_branch_support(
        owner_gate=result,
        admitted_histories=["paired_x1_y1"],
    )
    assert conjoined["crossed_box_only_admissible_histories"] == ["target_x1_y1"]
    assert conjoined["admissible_histories"] == []
    assert not conjoined["physical_owner_claim_permitted"]


def test_smoke_classification_is_scope_limited_and_layer_veto_is_visible() -> None:
    owner_gate = {"physical_owner_claim_permitted": False, "reason": "invalid_or_ambiguous_crossed_box"}
    result = probe.classify_smoke(
        execution_trust_passed=True,
        admitted_histories=["target_x1_y1"],
        contrasts={"target_x1_y1": {"block23_transport": 0.4, "block13_transport": 0.01}},
        owner_gate=owner_gate,
    )
    assert result["classification"] == "donor_state_late_coordinate_transport_supported"
    assert result["conclusion_scope"] == "DonorStateLateCoordinateTransport"
    assert result["physical_owner_claim_prohibited"]

    result = probe.classify_smoke(
        execution_trust_passed=True,
        admitted_histories=["target_x1_y1", "paired_x1_y1"],
        contrasts={
            "target_x1_y1": {"block23_transport": 0.4, "block13_transport": 0.0},
            "paired_x1_y1": {"block23_transport": -0.4, "block13_transport": 0.0},
        },
        owner_gate=owner_gate,
    )
    assert result["classification"] == "reversed_donor_direction"
    assert result["physical_owner_claim_prohibited"]

    result = probe.classify_smoke(
        execution_trust_passed=True,
        admitted_histories=["target_x1_y1"],
        contrasts={"target_x1_y1": {"block23_transport": 0.4, "block13_transport": 0.2}},
        owner_gate={"physical_owner_claim_permitted": True, "reason": "admissible"},
    )
    assert result["classification"] == "negative_control_veto"
    assert not result["physical_owner_claim_permitted"]

    result = probe.classify_smoke(
        execution_trust_passed=False,
        admitted_histories=["target_x1_y1"],
        contrasts={"target_x1_y1": {"block23_transport": 0.4, "block13_transport": 0.0}},
        owner_gate=owner_gate,
    )
    assert result["classification"] == "invalid_execution_trust_gate"
    assert not result["interpreted"]
