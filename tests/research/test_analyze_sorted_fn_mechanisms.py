from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.research.attest_sorted_fn_successor_score_run import (
    ATTESTATION_NAME,
    attest,
)
from scripts.research.analyze_sorted_fn_mechanisms import (
    ANALYSIS_SCHEMA_VERSION,
    BEHAVIOR_LANDSCAPE_ADMISSION_SCHEMA_VERSION,
    BEHAVIOR_SCHEMA_VERSION,
    DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION,
    FIXED_BUDGET_SCHEMA_VERSION,
    L2_ADMISSION_SCHEMA_VERSION,
    MERGE_RECEIPT_SCHEMA_VERSION,
    PLANNER_RECEIPT_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    RUN_ATTESTATION_SCHEMA_VERSION,
    SUCCESSOR_DECISION_CHANNEL,
    SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
    SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
    UNIT_ID,
    MechanismAnalysisError,
    _apply_control_calibration,
    _autonomous_owner_accounting,
    _build_owner_mechanisms,
    _build_behavior_landscape_admission,
    _build_scalar_l1_required_admissions,
    _build_sampling_admission,
    _load_and_bind_inputs,
    _no_behavior_contrary_support,
    _role_results_by_id,
    analyze_sorted_fn_mechanisms,
    canonical_json_bytes,
    compute_collision_envelope,
    compute_landscape_statistics,
    evaluate_collision_candidate,
    quantile_linear,
    sha256_file,
    sha256_json,
    validate_common_candidate_neighborhood_ids,
    validate_common_exact_gt_singleton_ids,
)
from scripts.research.merge_sorted_fn_successor_score_shards import (
    MERGE_SCHEMA_VERSION,
    PREDECESSOR_PRIMITIVES_FILE,
)
from scripts.research.run_sorted_fn_successor_behavior import (
    validate_landscape_admission,
)


def _candidate(
    candidate_id: str,
    *,
    population: str,
    family: str,
    iou: float,
    region: str | None = None,
    other_owner: str | None = None,
    proposal_measure: str | None = None,
) -> dict[str, Any]:
    row = {
        "candidate_id": candidate_id,
        "rung": "L0",
        "population": population,
        "family_id": family,
        "iou_to_target": iou,
        "region": region or ("target_strict" if iou >= 0.5 else "background"),
        "box": [10, 10, 20, 20],
        "candidate_neighborhood_member": False,
        "candidate_neighborhood_id": None,
        "exact_gt_singleton_member": False,
        "exact_gt_singleton_id": None,
    }
    if other_owner is not None:
        row["other_owner_gt_owner_id"] = other_owner
    if proposal_measure is not None:
        row["proposal_measure_id"] = proposal_measure
    return row


@pytest.mark.parametrize(
    ("rung", "target_raw_count", "reference_count"),
    [("L0", 6, 7), ("L1", 64, 65)],
)
def test_exact_landscape_math_and_registered_other_owner_margin(
    rung: str, target_raw_count: int, reference_count: int
) -> None:
    candidates = []
    scores = {}
    for index in range(target_raw_count):
        candidates.extend(
            [
                {
                    **_candidate(
                        f"t{index}",
                        population="target",
                        family="near_gt_micro",
                        iou=1.0,
                        proposal_measure="paired-measure",
                    ),
                    "rung": rung,
                    "candidate_neighborhood_member": True,
                    "candidate_neighborhood_id": (
                        f"nbh:gt:1:1:{rung}:near_gt_micro:{index}"
                    ),
                    "exact_gt_singleton_member": index == 0,
                    "exact_gt_singleton_id": (
                        f"exact-gt:gt:1:1:{rung}" if index == 0 else None
                    ),
                },
                {
                    **_candidate(
                        f"d{index}",
                        population="decoy",
                        family="near_gt_micro",
                        iou=0.0,
                        proposal_measure="paired-measure",
                    ),
                    "rung": rung,
                },
            ]
        )
        scores[f"t{index}"] = -1.0 - index * 0.1
        scores[f"d{index}"] = -3.0 - index * 0.1
    for index in range(reference_count):
        candidates.append(
            {
                **_candidate(
                    f"r{index}",
                    population="reference",
                    family="near_other_micro",
                    iou=0.0,
                    region="other_owner",
                    other_owner="gt:1:9",
                    proposal_measure="must-not-enter-target-decoy-functional",
                ),
                "rung": rung,
                "proposal_weight": 999.0,
            }
        )
        scores[f"r{index}"] = -2.0 - index * 0.1

    stats = compute_landscape_statistics(
        candidates,
        scores,
        other_owner_reference_bound=True,
        target_gt_box=[10, 10, 20, 20],
    )

    assert stats["target_peak"] == -1.0
    assert stats["background_prominence"] == 2.0
    assert stats["other_owner_margin"] == 1.0
    assert stats["localized_rank"] == 0.0
    assert stats["target_peak_candidate_id"] == "t0"
    assert stats["target_peak_box"] == [10, 10, 20, 20]
    assert stats["collision_variant_peaks"] == {
        "F1_full_target_strict_max": -1.0,
        "F2_near_gt_micro_strict_max": -1.0,
        "F3_exact_gt_singleton": -1.0,
    }
    assert stats["reference_population_count"] == reference_count
    assert stats["other_owner_margin_population_accounting"] == {
        "target_near_gt_micro_raw_row_count": target_raw_count,
        "target_exact_gt_singleton_repeat_count": 1,
        "target_logical_multiset_count": reference_count,
        "target_unique_geometry_count": 1,
        "reference_near_other_micro_raw_row_count": reference_count,
        "reference_unique_geometry_count": 1,
    }
    assert stats["target_control_proposal_comparable"] is True


def test_reference_population_is_non_vacuous_and_structural_absence_is_explicit() -> None:
    candidates = [
        _candidate("t0", population="target", family="a", iou=1.0),
        _candidate("d0", population="decoy", family="a", iou=0.0),
    ]
    stats = compute_landscape_statistics(
        candidates,
        {"t0": -1.0, "d0": -3.0},
        other_owner_reference_bound=False,
    )
    assert stats["other_owner_margin"] is None
    assert stats["other_owner_margin_status"] == (
        "structurally_inapplicable_no_bound_same_description_neighbor"
    )

    with pytest.raises(MechanismAnalysisError, match="ledger binds"):
        compute_landscape_statistics(
            candidates,
            {"t0": -1.0, "d0": -3.0},
            other_owner_reference_bound=True,
        )


def test_f3_requires_one_explicit_exact_geometry_marker() -> None:
    target = {
        **_candidate("t0", population="target", family="near_gt_micro", iou=1.0),
        "schema_version": FIXED_BUDGET_SCHEMA_VERSION,
        "candidate_neighborhood_member": True,
        "candidate_neighborhood_id": "nbh:gt:1:0:L0:near_gt_micro:0",
        "exact_gt_singleton_member": True,
        "exact_gt_singleton_id": "exact-gt:gt:1:0:L0",
    }
    decoy = {
        **_candidate("d0", population="decoy", family="near_gt_micro", iou=0.0),
        "schema_version": FIXED_BUDGET_SCHEMA_VERSION,
        "candidate_neighborhood_member": False,
        "candidate_neighborhood_id": None,
    }
    stats = compute_landscape_statistics(
        [target, decoy],
        {"t0": -1.0, "d0": -3.0},
        target_gt_box=[10, 10, 20, 20],
        target_gt_owner_id="gt:1:0",
    )
    assert stats["collision_variant_peaks"]["F2_near_gt_micro_strict_max"] == -1.0
    assert stats["collision_variant_peaks"]["F3_exact_gt_singleton"] == -1.0

    missing = deepcopy(target)
    missing["exact_gt_singleton_member"] = False
    missing["exact_gt_singleton_id"] = None
    with pytest.raises(MechanismAnalysisError, match="exactly one explicitly marked"):
        compute_landscape_statistics(
            [missing, decoy],
            {"t0": -1.0, "d0": -3.0},
            target_gt_box=[10, 10, 20, 20],
            target_gt_owner_id="gt:1:0",
        )

    duplicate = {
        **target,
        "candidate_id": "t1",
        "candidate_neighborhood_id": "nbh:gt:1:0:L0:near_gt_micro:1",
    }
    with pytest.raises(MechanismAnalysisError, match="exactly one explicitly marked"):
        compute_landscape_statistics(
            [target, duplicate, decoy],
            {"t0": -1.0, "t1": -1.1, "d0": -3.0},
            target_gt_box=[10, 10, 20, 20],
            target_gt_owner_id="gt:1:0",
        )

    non_exact = {**target, "box": [11, 10, 20, 20]}
    with pytest.raises(MechanismAnalysisError, match="not the exact GT box"):
        compute_landscape_statistics(
            [non_exact, decoy],
            {"t0": -1.0, "d0": -3.0},
            target_gt_box=[10, 10, 20, 20],
            target_gt_owner_id="gt:1:0",
        )

    wrong_id = {**target, "exact_gt_singleton_id": "exact-gt:gt:9:9:L0"}
    with pytest.raises(MechanismAnalysisError, match="owner/rung identity"):
        compute_landscape_statistics(
            [wrong_id, decoy],
            {"t0": -1.0, "d0": -3.0},
            target_gt_box=[10, 10, 20, 20],
            target_gt_owner_id="gt:1:0",
        )


def test_quantile_is_linear_and_missing_matched_controls_never_pool() -> None:
    assert quantile_linear([0.0, 10.0], 0.10) == pytest.approx(1.0)
    assert quantile_linear([0.0, 10.0], 0.90) == pytest.approx(9.0)
    entry = {
        "description_size_crowding_stratum": "target-stratum",
        "context_family": "root",
        "rung": "L1",
        "is_control": False,
        "control_kind": None,
        "per_iou_threshold": {
            "0.5": {
                "background_prominence": 100.0,
                "localized_rank": 0.0,
                "equal_count_guard": True,
                "other_owner_margin": None,
            }
        },
        "iou_band_stability": {
            "peak_identity_stable": True,
            "positive_prominence_side_stable": True,
        },
    }
    pooled_control = deepcopy(entry)
    pooled_control.update(
        {
            "description_size_crowding_stratum": "different-stratum",
            "is_control": True,
            "control_kind": "strict_positive",
            "context_id": "control",
        }
    )
    entry["context_id"] = "target"

    table = _apply_control_calibration([entry, pooled_control], 1e-5)

    target_calibration = next(
        row for row in table if row["description_size_crowding_stratum"] == "target-stratum"
    )
    assert target_calibration["status"] == "unresolved_missing_matched_controls"
    assert entry["usable_target_support"] is False


def test_scalar_smoke_is_positive_sign_only_and_marginal_requires_l1() -> None:
    def scalar_entry(context_id: str, prominence: float) -> dict[str, Any]:
        return {
            "context_id": context_id,
            "description_size_crowding_stratum": "strict-rescue",
            "context_family": "root",
            "rung": "scalar_smoke",
            "is_control": True,
            "control_kind": "strict_rescue",
            "per_iou_threshold": {
                "0.5": {
                    "background_prominence": prominence,
                    "localized_rank": 0.0,
                    "equal_count_guard": True,
                    "other_owner_margin": None,
                }
            },
            "iou_band_stability": {
                "peak_identity_stable": True,
                "positive_prominence_side_stable": prominence > 0,
            },
        }

    positive = scalar_entry("positive", 1.0)
    marginal = scalar_entry("marginal", 0.0)
    table = _apply_control_calibration([positive, marginal], 1e-5)

    assert table[0]["status"] == "positive_sign_only_scalar_smoke"
    assert positive["usable_target_support"] is False
    assert positive["scalar_smoke_interpretation"]["l1_admission"]["status"] == (
        "positive_sign_observed"
    )
    assert marginal["scalar_smoke_interpretation"]["l1_admission"]["status"] == (
        "L1_required"
    )
    assert marginal["scalar_smoke_interpretation"]["negative_mechanism_eligible"] is False
    assert _build_scalar_l1_required_admissions([positive, marginal]) == [
        {
            "status": "L1_required",
            "context_id": "marginal",
            "role_id": None,
            "gt_owner_id": None,
            "reason": "scalar_margin_is_marginal_or_negative_and_cannot_decide_stop_rule_4",
            "stop_rule_4_eligible": False,
            "claim_scope": "scalar_smoke_marginal_or_negative_requires_frozen_L1",
        }
    ]
    assert (
        _build_behavior_landscape_admission(
            loaded={"registry_digest": "a" * 64, "merge_path": Path(__file__)},
            landscape_entries=[positive, marginal],
        )
        is None
    )


def test_equal_count_and_family_guards_block_primary_statistics() -> None:
    unequal = [
        _candidate("t0", population="target", family="a", iou=1.0),
        _candidate("t1", population="target", family="b", iou=1.0),
        _candidate("d0", population="decoy", family="a", iou=0.0),
    ]
    stats = compute_landscape_statistics(unequal, {"t0": -1.0, "t1": -2.0, "d0": -3.0})
    assert stats["equal_count_guard"] is False
    assert stats["background_prominence"] is None
    assert stats["localized_rank"] is None
    assert stats["target_bank_score"] is None

    nonbackground_decoy = [
        _candidate("t0", population="target", family="a", iou=1.0),
        _candidate("d0", population="decoy", family="a", iou=0.0, region="other_owner"),
    ]
    stats = compute_landscape_statistics(
        nonbackground_decoy, {"t0": -1.0, "d0": -3.0}
    )
    assert stats["equal_count_guard"] is True
    assert stats["equal_count_background_guard"] is False
    assert stats["background_prominence"] is None

    family_mismatch = [
        _candidate("t0", population="target", family="a", iou=1.0),
        _candidate("d0", population="decoy", family="b", iou=0.0),
    ]
    stats = compute_landscape_statistics(family_mismatch, {"t0": -1.0, "d0": -3.0})
    assert stats["equal_count_guard"] is True
    assert stats["family_multiset_guard"] is False
    assert stats["target_bank_score"] is None


def test_proposal_measure_mismatch_blocks_bank_score_only() -> None:
    candidates = [
        _candidate("t0", population="target", family="a", iou=1.0, proposal_measure="target-measure"),
        _candidate("d0", population="decoy", family="a", iou=0.0, proposal_measure="decoy-measure"),
    ]
    stats = compute_landscape_statistics(candidates, {"t0": -1.0, "d0": -3.0})
    assert stats["background_prominence"] == 2.0
    assert stats["proposal_measure_guard"] is False
    assert stats["target_bank_score"] is None


def test_proposal_weighted_target_bank_score_uses_full_measure_mass() -> None:
    candidates = [
        {**_candidate("t0", population="target", family="a", iou=1.0), "proposal_weight": 0.25},
        {**_candidate("t1", population="target", family="b", iou=1.0), "proposal_weight": 0.75},
        {**_candidate("d0", population="decoy", family="a", iou=0.0), "proposal_weight": 0.25},
        {**_candidate("d1", population="decoy", family="b", iou=0.0), "proposal_weight": 0.75},
    ]
    stats = compute_landscape_statistics(
        candidates, {"t0": -1.0, "t1": -2.0, "d0": -3.0, "d1": -4.0}
    )
    expected = math.log(0.25 * math.exp(-1.0) + 0.75 * math.exp(-2.0))
    assert stats["target_control_proposal_comparable"] is True
    assert stats["target_bank_score"] == pytest.approx(expected)


def test_collision_envelope_includes_mechanical_null_and_reports_leave_one_out() -> None:
    envelope = compute_collision_envelope(
        {"mechanical": -0.5, "null-a": -0.2, "null-b": -0.1}, 0.05
    )
    assert envelope["threshold"] == pytest.approx(-0.55)
    assert envelope["mechanical_null_included"] is True
    assert envelope["leave_one_out"]["mechanical"]["threshold"] == pytest.approx(-0.25)


def _envelopes(threshold: float = -0.5) -> dict[str, dict[str, Any]]:
    return {
        key: {"status": "admitted", "threshold": threshold}
        for key in ("0.4", "0.5", "0.6")
    }


def _variant_declines(
    declines: dict[str, float],
) -> dict[str, dict[str, float]]:
    return {
        "F1_full_target_strict_max": dict(declines),
        "F2_near_gt_micro_strict_max": dict(declines),
        "F3_exact_gt_singleton": dict(declines),
    }


def _variant_envelopes(
    envelopes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "F1_full_target_strict_max": deepcopy(envelopes),
        "F2_near_gt_micro_strict_max": deepcopy(envelopes),
        "F3_exact_gt_singleton": deepcopy(envelopes),
    }


def test_16228_and_2685_collision_asymmetry_and_behavior_requirement() -> None:
    declines = {"0.4": -1.0, "0.5": -1.0, "0.6": -1.0}
    without_behavior = evaluate_collision_candidate(
        target_gt_owner_id="gt:16228:30",
        image_id="16228",
        selective_declines_by_variant_and_iou=_variant_declines(declines),
        envelopes_by_variant_and_iou=_variant_envelopes(_envelopes()),
        behavior_co_movement=None,
        token_identical_matched_null=True,
        candidate_neighborhood_sign_stable=True,
    )
    assert without_behavior["supported"] is False
    assert without_behavior["status"] == "likelihood_only_collision_consistent_unresolved"

    supported = evaluate_collision_candidate(
        target_gt_owner_id="gt:16228:30",
        image_id="16228",
        selective_declines_by_variant_and_iou=_variant_declines(declines),
        envelopes_by_variant_and_iou=_variant_envelopes(_envelopes()),
        behavior_co_movement=True,
        token_identical_matched_null=True,
        candidate_neighborhood_sign_stable=True,
    )
    assert supported["supported"] is True

    missing_perturbation_freeze = evaluate_collision_candidate(
        target_gt_owner_id="gt:16228:30",
        image_id="16228",
        selective_declines_by_variant_and_iou=_variant_declines(declines),
        envelopes_by_variant_and_iou=_variant_envelopes(_envelopes()),
        behavior_co_movement=True,
        token_identical_matched_null=True,
        candidate_neighborhood_sign_stable=None,
    )
    assert missing_perturbation_freeze["supported"] is False
    assert "frozen_candidate_neighborhood_perturbation_band_missing" in missing_perturbation_freeze["reasons"]

    bottle = evaluate_collision_candidate(
        target_gt_owner_id="gt:2685:15",
        image_id="2685",
        selective_declines_by_variant_and_iou=_variant_declines(declines),
        envelopes_by_variant_and_iou=_variant_envelopes(_envelopes()),
        behavior_co_movement=True,
        token_identical_matched_null=True,
        candidate_neighborhood_sign_stable=True,
    )
    assert bottle["supported"] is False
    assert bottle["structural_status"] == "missing_bottle_stratum_null"


def test_collision_requires_f1_f2_f3_iou_band_sign_stability() -> None:
    declines = _variant_declines({"0.4": -1.0, "0.5": -1.0, "0.6": -1.0})
    declines["F2_near_gt_micro_strict_max"]["0.6"] = 0.1
    result = evaluate_collision_candidate(
        target_gt_owner_id="gt:16228:30",
        image_id="16228",
        selective_declines_by_variant_and_iou=declines,
        envelopes_by_variant_and_iou=_variant_envelopes(_envelopes()),
        behavior_co_movement=True,
        token_identical_matched_null=True,
        candidate_neighborhood_sign_stable=True,
    )
    assert result["likelihood_exceeds_envelope"] is True
    assert result["sign_stability_by_variant"] == {
        "F1_full_target_strict_max": True,
        "F2_near_gt_micro_strict_max": False,
        "F3_exact_gt_singleton": True,
    }
    assert result["supported"] is False


def test_candidate_neighborhood_ids_must_be_common_across_p_g_f() -> None:
    common = validate_common_candidate_neighborhood_ids(
        {
            "P": {"n0": -1.0, "n1": -2.0},
            "G": {"n0": -1.5, "n1": -2.5},
            "F": {"n0": -0.5, "n1": -1.5},
        }
    )
    assert common == ["n0", "n1"]
    with pytest.raises(MechanismAnalysisError, match="common non-empty"):
        validate_common_candidate_neighborhood_ids(
            {
                "P": {"n0": -1.0},
                "G": {"n0": -1.5, "foreign": -9.0},
                "F": {"n0": -0.5},
            }
        )
    assert (
        validate_common_exact_gt_singleton_ids(
            {"P": ["exact-0"], "G": ["exact-0"], "F": ["exact-0"]}
        )
        == "exact-0"
    )
    with pytest.raises(MechanismAnalysisError, match="not identical"):
        validate_common_exact_gt_singleton_ids(
            {"P": ["exact-0"], "G": ["exact-1"], "F": ["exact-0"]}
        )
    with pytest.raises(MechanismAnalysisError, match="exactly one"):
        validate_common_exact_gt_singleton_ids(
            {"P": ["exact-0", "exact-1"], "G": ["exact-0"], "F": ["exact-0"]}
        )


def test_legacy_forced_arm_aggregate_cannot_pass_no_support_gate() -> None:
    legacy_role = {
        "_validated_landscape_admission": {"status": "passed"},
        "_validated_prefix_owner_ids": [],
        "gt_owner_id": "gt:1:0",
        "arms": {
            "free_next_row": {"target_recovered_strict": False, "target_recovered_loose": False},
            "forced_description_greedy": {"target_recovered_strict": False, "target_recovered_loose": False},
            "forced_description_low_temperature_samples": [],
        },
    }
    assert _no_behavior_contrary_support([legacy_role]) is False

    partitioned = deepcopy(legacy_role)
    partitioned["arms"]["free_next_row"] = _sealed_behavior_arm(False)
    partitioned["arms"]["forced_description_greedy"] = _sealed_behavior_arm(True)
    assert _no_behavior_contrary_support([partitioned]) is True


def _sealed_behavior_arm(forced: bool, recovered_owner: str | None = None) -> dict[str, Any]:
    suffix_rows = []
    if recovered_owner is not None:
        suffix_rows = [
            {
                "pred_row_id": "autonomous:0",
                "strict_matched_owner_ids": [recovered_owner],
                "loose_matched_owner_ids": [recovered_owner],
                "semantic_drift_evidence": [],
            }
        ]
    intervention_row = {
        "pred_row_id": "intervention:0",
        "strict_matched_owner_ids": [],
        "loose_matched_owner_ids": [],
        "semantic_drift_evidence": [],
    }
    rows = ([intervention_row] if forced else []) + suffix_rows
    return {
        "arm_kind": "forced_description" if forced else "free_next_row",
        "rows": rows,
        "intervention": (
            {
                "row": intervention_row,
                "accounting_status": "excluded_from_autonomous_suffix_but_included_in_final_conditioned_set",
            }
            if forced
            else None
        ),
        "released_suffix_rows": suffix_rows,
        "autonomous_evidence_row_ids": [row["pred_row_id"] for row in suffix_rows],
        "target_recovered_strict": recovered_owner is not None,
        "target_recovered_loose": recovered_owner is not None,
        "autonomous_suffix_target_recovered_strict": recovered_owner is not None,
        "autonomous_suffix_target_recovered_loose": recovered_owner is not None,
        "final_unique_strict_owner_ids": ([] if recovered_owner is None else [recovered_owner]),
        "final_unique_loose_owner_ids": ([] if recovered_owner is None else [recovered_owner]),
        "semantic_drift_evidence": [],
    }


def _intervention_only_recovery_role(
    *,
    target: str = "gt:1:0",
    intervention_row_strict_owner_ids: list[str] | None = None,
    forced_final_unique_strict_owner_ids: list[str] | None = None,
) -> dict[str, Any]:
    """A minimal role where the target is recovered only by the forced
    intervention row (row0), never by the free arm's own or forced arm's
    autonomous suffix. This is the exact shape of the corrected
    due_turn:gt:7511:17 artifact: forced's
    autonomous_suffix_target_recovered_strict is False, yet the target is
    still a member of forced's declared final_unique_strict_owner_ids via
    the row0-inclusive accounting layer.
    """
    if intervention_row_strict_owner_ids is None:
        intervention_row_strict_owner_ids = [target]
    if forced_final_unique_strict_owner_ids is None:
        forced_final_unique_strict_owner_ids = [target]
    free_suffix_row = {
        "pred_row_id": "free:autonomous:0",
        "strict_matched_owner_ids": [target],
        "loose_matched_owner_ids": [target],
        "semantic_drift_evidence": [],
    }
    free_arm = {
        "arm_kind": "free_next_row",
        "rows": [free_suffix_row],
        "intervention": None,
        "released_suffix_rows": [free_suffix_row],
        "autonomous_evidence_row_ids": ["free:autonomous:0"],
        "target_gt_owner_id": target,
        "autonomous_suffix_target_recovered_strict": True,
        "autonomous_suffix_target_recovered_loose": True,
        "semantic_drift_evidence": [],
        "final_unique_strict_owner_ids": [target],
    }
    intervention_row = {
        "pred_row_id": "forced:intervention:0",
        "strict_matched_owner_ids": intervention_row_strict_owner_ids,
        "loose_matched_owner_ids": intervention_row_strict_owner_ids,
        "semantic_drift_evidence": [],
    }
    forced_arm = {
        "arm_kind": "forced_description",
        "rows": [intervention_row],
        "intervention": {
            "row": intervention_row,
            "accounting_status": "excluded_from_autonomous_suffix_but_included_in_final_conditioned_set",
            "owner_ids_strict": intervention_row_strict_owner_ids,
            "owner_ids_loose": intervention_row_strict_owner_ids,
        },
        "released_suffix_rows": [],
        "autonomous_evidence_row_ids": [],
        "target_gt_owner_id": target,
        "autonomous_suffix_target_recovered_strict": False,
        "autonomous_suffix_target_recovered_loose": False,
        "semantic_drift_evidence": [],
        "final_unique_strict_owner_ids": forced_final_unique_strict_owner_ids,
    }
    return {
        "_validated_landscape_admission": {"status": "passed"},
        "_validated_prefix_owner_ids": [],
        "gt_owner_id": target,
        "arms": {
            "free_next_row": free_arm,
            "forced_description_greedy": forced_arm,
        },
        "comparison": {
            "downstream_owner_accounting": {
                "gained_owner_ids": [],
                "retained_owner_ids": [target],
                "lost_owner_ids": [],
                "target_recovery_exchange": False,
                "exchange_lost_owner_ids": [],
            }
        },
    }


def test_autonomous_owner_accounting_admits_row0_intervention_owned_target() -> None:
    """Regression: forced_set must include the validated forced-intervention
    row's owners (row0), not just prefix + autonomous-suffix owners, because
    the arm's own declared final_unique_strict_owner_ids is intentionally
    row0-inclusive (accounting_layer=
    final_intervention_conditioned_row0_plus_autonomous_suffix). Before the
    fix, this role's declared final set could never be reproduced from
    prefix+suffix alone, so the gate closed and the analyzer fell back to
    likelihood-only, even though free and forced agree exactly and there is
    no gain or loss to report.
    """
    role = _intervention_only_recovery_role()
    accounting = _autonomous_owner_accounting(role)
    assert accounting is not None
    assert accounting["gained_owner_ids"] == []
    assert accounting["lost_owner_ids"] == []
    assert accounting["retained_owner_ids"] == ["gt:1:0"]
    assert accounting["target_recovery_exchange"] is False


@pytest.mark.parametrize(
    ("intervention_row_strict_owner_ids", "forced_final_unique_strict_owner_ids"),
    [
        # Row evidence recovers the target, but the arm's declared final set
        # omits it: under-declaring the row0-inclusive final set.
        (["gt:1:0"], []),
        # The declared final set claims the target, but the validated
        # intervention row itself carries no matching owner: the declared
        # final set is not reproducible from validated evidence.
        ([], ["gt:1:0"]),
    ],
)
def test_autonomous_owner_accounting_rejects_mismatched_intervention_declaration(
    intervention_row_strict_owner_ids: list[str],
    forced_final_unique_strict_owner_ids: list[str],
) -> None:
    role = _intervention_only_recovery_role(
        intervention_row_strict_owner_ids=intervention_row_strict_owner_ids,
        forced_final_unique_strict_owner_ids=forced_final_unique_strict_owner_ids,
    )
    assert _autonomous_owner_accounting(role) is None


def test_no_support_requires_all_contexts_controls_and_partitioned_behavior() -> None:
    role_id = "root:gt:1:0"
    registry = {
        "mechanism_cohort": {
            "targets": [
                {
                    "gt_owner_id": "gt:1:0",
                    "image_id": "1",
                    "stratum": "matched-person-small-crowded",
                    "declared_role": "no_free_target",
                    "expected_cohort": "no_free_spatial_support",
                }
            ]
        },
        "smoke": {
            "roles": [
                {
                    "role_id": role_id,
                    "role_kind": "root_context",
                    "gt_owner_id": "gt:1:0",
                }
            ],
            "null_pair_envelope": {"pairs": []},
        },
    }
    landscape = [
        {
            "gt_owner_id": "gt:1:0",
            "context_id": f"ctx:fn:{role_id}",
            "role_kind": "root_context",
            "rung": "L1",
            "usable_target_support": False,
            "matched_control_calibration": {
                "status": "calibrated",
                "positive_controls_pass": True,
            },
            "per_iou_threshold": {"0.5": {"target_peak": None}},
        }
    ]
    behavior_role = {
        "role_id": role_id,
        "gt_owner_id": "gt:1:0",
        "_validated_landscape_admission": {"status": "passed"},
        "_validated_prefix_owner_ids": [],
        "arms": {
            "free_next_row": _sealed_behavior_arm(False),
            "forced_description_greedy": _sealed_behavior_arm(True),
            "forced_description_low_temperature_samples": [],
        },
        "comparison": {
            "downstream_owner_accounting": {
                "gained_owner_ids": [],
                "retained_owner_ids": [],
                "lost_owner_ids": [],
                "target_recovery_exchange": False,
                "exchange_lost_owner_ids": [],
            }
        },
    }
    loaded = {"registry": registry, "behavior": {"sealed": True}, "tolerance": {"effective_tolerance": 1e-5}}

    rows = _build_owner_mechanisms(
        loaded,
        landscape,
        collision_by_owner={},
        role_results={role_id: behavior_role},
    )
    flag = rows[0]["mechanism_evidence_flags"][
        "no_usable_localization_support_under_tested_interface"
    ]
    assert flag["supported"] is True

    landscape[0]["matched_control_calibration"] = {
        "status": "unresolved_missing_matched_controls"
    }
    rows = _build_owner_mechanisms(
        loaded,
        landscape,
        collision_by_owner={},
        role_results={role_id: behavior_role},
    )
    flag = rows[0]["mechanism_evidence_flags"][
        "no_usable_localization_support_under_tested_interface"
    ]
    assert flag["supported"] is False


def test_sampling_admission_requires_landscape_gate_failed_partitioned_greedy_and_frozen_tuple(
    tmp_path: Path,
) -> None:
    role_id = "root:gt:1:0"
    mechanism_path = tmp_path / "mechanism-rules.json"
    mechanism_content = {
        "schema_version": "sorted-fn-mechanism-decision-rules.v1",
        "unit_id": UNIT_ID,
        "conditional_sampling": {
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "null_semantics": "finite_sampling_null_is_absence_neutral",
        },
    }
    _write_json(
        mechanism_path,
        {**mechanism_content, "self_digest": sha256_json(mechanism_content)},
    )
    loaded = {
        "registry_digest": "a" * 64,
        "merge_path": Path(__file__),
        "mechanism_rules": json.loads(mechanism_path.read_text()),
        "mechanism_rules_path": mechanism_path,
        "mechanism_rules_digest": json.loads(mechanism_path.read_text())[
            "self_digest"
        ],
    }
    landscape = [
        {
            "role_id": role_id,
            "context_id": f"ctx:fn:{role_id}",
            "gt_owner_id": "gt:1:0",
            "rung": "L1",
            "usable_target_support": True,
            "per_iou_threshold": {"0.5": {"raw_candidates": []}},
        }
    ]
    legacy = {
        role_id: {
            "role_id": role_id,
            "_validated_landscape_admission": {"status": "passed"},
            "_validated_prefix_owner_ids": [],
            "gt_owner_id": "gt:1:0",
            "arms": {"forced_description_greedy": {"target_recovered_strict": False}},
        }
    }
    parameters = {
        "temperature": 0.4,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "k": 2,
        "seeds": [11, 12],
        "horizon_rows": 4,
    }
    assert _build_sampling_admission(
        loaded=loaded,
        landscape_entries=landscape,
        role_results=legacy,
        parameters=parameters,
    ) is None

    partitioned = deepcopy(legacy)
    partitioned[role_id]["arms"]["forced_description_greedy"] = _sealed_behavior_arm(
        True
    )
    raw_behavior_role = {
        key: value
        for key, value in partitioned[role_id].items()
        if not key.startswith("_")
    }
    partitioned[role_id]["_behavior_role_content_sha256"] = sha256_json(
        raw_behavior_role
    )
    behavior_content = {
        "schema_version": BEHAVIOR_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "contract": {},
        "policy_views": [
            {
                "repetition_penalty": 1.0,
                "roles": [raw_behavior_role],
            }
        ],
    }
    behavior_document = {
        **behavior_content,
        "output_content_sha256": sha256_json(behavior_content),
    }
    behavior_path = tmp_path / "behavior.json"
    _write_json(behavior_path, behavior_document)
    loaded["behavior"] = behavior_document
    loaded["behavior_output_path"] = behavior_path
    admission = _build_sampling_admission(
        loaded=loaded,
        landscape_entries=landscape,
        role_results=partitioned,
        parameters=parameters,
    )
    assert admission is not None
    assert admission["decode_parameters"] == parameters
    assert admission["behavior_output"] == {
        "path": str(behavior_path),
        "sha256": sha256_file(behavior_path),
        "output_content_sha256": behavior_document["output_content_sha256"],
    }
    assert admission["landscape_condition"]["condition_by_role"][role_id][
        "greedy_canonical_description_recovery"
    ]["status"] == "failed"

    missing_declared_recovery = deepcopy(partitioned)
    del missing_declared_recovery[role_id]["arms"]["forced_description_greedy"][
        "autonomous_suffix_target_recovered_strict"
    ]
    assert _build_sampling_admission(
        loaded=loaded,
        landscape_entries=landscape,
        role_results=missing_declared_recovery,
        parameters=parameters,
    ) is None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _sealed_fixture(tmp_path: Path) -> dict[str, Path]:
    registry_path = tmp_path / "registry.json"
    roles = [
        {
            "role_id": "root:gt:1:0",
            "role_kind": "root_context",
            "gt_owner_id": "gt:1:0",
            "trajectory": {"image_id": "1", "decode_mode": "greedy", "seed": 0},
            "prefix": {"token_ids": [1], "token_ids_sha256": sha256_json([1])},
        },
        {
            "role_id": "root:gt:1:1",
            "role_kind": "root_context",
            "gt_owner_id": "gt:1:1",
            "trajectory": {"image_id": "1", "decode_mode": "greedy", "seed": 0},
            "prefix": {"token_ids": [1], "token_ids_sha256": sha256_json([1])},
        },
    ]
    registry_content = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "mechanism_cohort": {
            "targets": [
                {
                    "gt_owner_id": "gt:1:0",
                    "image_id": "1",
                    "stratum": "matched-person-small-crowded",
                    "declared_role": "no_free_target",
                    "expected_cohort": "no_free_spatial_support",
                },
                {
                    "gt_owner_id": "gt:1:1",
                    "image_id": "1",
                    "stratum": "matched-person-small-crowded",
                    "declared_role": "strict_positive_control",
                    "expected_cohort": "greedy_strict_present",
                },
            ]
        },
        "smoke": {
            "roles": roles,
            "null_pair_envelope": {"status": "insufficient", "pairs": []},
        },
    }
    _write_json(registry_path, {**registry_content, "registry_digest": sha256_json(registry_content)})

    rules_path = tmp_path / "rules.json"
    token_ids = list(range(1000, 2000))
    rules = {
        "numeric_tolerance": 1e-5,
        "token_registry": {
            "coordinate_bin_to_token_id": {"coordinate_bin_token_ids": token_ids}
        },
    }
    _write_json(rules_path, rules)
    rules_template_path = tmp_path / "rules-template.json"
    _write_json(rules_template_path, {"fixture": "execution-rules-parent"})
    mechanism_rules_path = tmp_path / "mechanism-decision-rules.json"
    mechanism_content = {
        "schema_version": "sorted-fn-mechanism-decision-rules.v1",
        "unit_id": UNIT_ID,
        "geometry": {"iou_thresholds": [0.4, 0.5, 0.6]},
        "calibration": {
            "quantile_algorithm": "type7",
            "lower_quantile": 0.10,
            "upper_quantile": 0.90,
        },
        "populations": {"target": "fixture", "decoy": "fixture", "reference": "fixture"},
        "rung_quotas": {
            "scalar_smoke": {
                "claim_direction": "positive_only",
                "target_count": 30,
                "decoy_count": "equal_to_target",
                "reference_count": 7,
            },
            "L1": {
                "target_count": 256,
                "decoy_count": "equal_to_target",
                "reference_count": 65,
            },
        },
        "collision": {"statistics": {"F1": "fixture", "F2": "fixture", "F3": "fixture"}},
        "neighborhood": {
            "eligible_family": "near_gt_micro",
            "eligible_population": "target",
            "member_field": "candidate_neighborhood_member",
            "id_field": "candidate_neighborhood_id",
        },
        "exact_gt_singleton": {
            "member_field": "exact_gt_singleton_member",
            "id_field": "exact_gt_singleton_id",
            "id_rule": "exact-gt:<gt_owner_id>:<rung>",
            "eligible_family": "near_gt_micro",
            "eligible_population": "target",
        },
        "conditional_sampling": {
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "null_semantics": "finite_sampling_null_is_absence_neutral",
        },
        "upstream_digests": {
            "execution_landscape_decision_rules_sha256": sha256_file(rules_path),
            "fn_mechanism_registry_sha256": json.loads(
                registry_path.read_text(encoding="utf-8")
            )["registry_digest"],
            "rules_template_sha256": sha256_file(rules_template_path),
        },
    }
    _write_json(
        mechanism_rules_path,
        {**mechanism_content, "self_digest": sha256_json(mechanism_content)},
    )
    mechanism_rules_sha256 = sha256_file(mechanism_rules_path)

    fixed_path = tmp_path / "fixed-budget-candidates.jsonl"
    fixed_rows = []
    for owner_index, (owner, is_control, kind) in enumerate(
        (("gt:1:0", False, None), ("gt:1:1", True, "strict_positive"))
    ):
        context_id = f"ctx:fn:root:{owner}"
        for index, (population, family, box, iou) in enumerate(
            (
                ("target", "near_gt_micro", [10, 10, 20, 20], 1.0),
                ("target", "b", [11, 11, 19, 19], 0.64),
                ("decoy", "near_gt_micro", [500, 500, 510, 510], 0.0),
                ("decoy", "b", [520, 520, 528, 528], 0.0),
            )
        ):
            candidate_id = f"cand:{owner_index}:{index}"
            fixed_rows.append(
                {
                    "schema_version": FIXED_BUDGET_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "coord_token_ids": [token_ids[value] for value in box],
                    "source_digest": sha256_json([context_id, candidate_id]),
                    "owner_context_id": context_id,
                    "rung": "L1",
                    "region": "target_strict" if population == "target" else "background",
                    "population": population,
                    "is_control": is_control,
                    "control_kind": kind,
                    "matched_control_group": "matched-person-small-crowded",
                    "description_size_crowding_stratum": "matched-person-small-crowded",
                    "family_id": family,
                    "iou_to_target": iou,
                    "candidate_neighborhood_member": (
                        population == "target" and family == "near_gt_micro"
                    ),
                    "candidate_neighborhood_id": (
                        f"nbh:{owner}:L1:near_gt_micro:0"
                        if population == "target" and family == "near_gt_micro"
                        else None
                    ),
                    "exact_gt_singleton_member": (
                        population == "target" and family == "near_gt_micro"
                    ),
                    "exact_gt_singleton_id": (
                        f"exact-gt:{owner}:L1"
                        if population == "target" and family == "near_gt_micro"
                        else None
                    ),
                    "mechanism_decision_rules_sha256": mechanism_rules_sha256,
                }
            )
    _write_jsonl(fixed_path, fixed_rows)

    ledger_path = tmp_path / "owner-context-ledger.jsonl"
    _write_jsonl(
        ledger_path,
        [
            {
                "context_id": f"ctx:fn:root:{owner}",
                "ground_truth": {"box": [10, 10, 20, 20]},
                "other_owner_reference_status": (
                    "no_same_description_non_overlapping_owner"
                ),
            }
            for owner in ("gt:1:0", "gt:1:1")
        ],
    )

    planner_path = tmp_path / "planner-receipt.json"
    planner_content = {
        "schema_version": PLANNER_RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "sources": {
            "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
            "rules_template": {
                "path": str(rules_template_path),
                "sha256": sha256_file(rules_template_path),
            },
        },
        "outputs": {
            "owner_context_ledger": {
                "path": str(ledger_path),
                "sha256": sha256_file(ledger_path),
                "row_count": 2,
            },
            "landscape_decision_rules": {"path": str(rules_path), "sha256": sha256_file(rules_path)},
            "mechanism_decision_rules": {
                "path": str(mechanism_rules_path),
                "sha256": mechanism_rules_sha256,
            },
            "fixed_budget_candidates": {"path": str(fixed_path), "sha256": sha256_file(fixed_path), "row_count": len(fixed_rows)},
        },
        "neighborhood_consistency": {
            "multi_context_groups_checked": 0,
            "status": "consistent",
        },
    }
    _write_json(planner_path, {**planner_content, "receipt_digest": sha256_json(planner_content)})

    scores_path = tmp_path / "scores.jsonl"
    scores = []
    for row in fixed_rows:
        is_control = row["is_control"]
        score = (
            -1.0
            if row["population"] == "target" and row["family_id"] == "a"
            else -1.2
            if row["population"] == "target"
            else -3.0
            if row["family_id"] == "a"
            else -4.0
        )
        if not is_control:
            score -= 4.0
        scores.append(
            {
                "schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
                "unit_id": UNIT_ID,
                "candidate_id": row["candidate_id"],
                "context_id": row["owner_context_id"],
                "rung": row["rung"],
                "region": row["region"],
                "population": row["population"],
                "candidate_neighborhood_member": row[
                    "candidate_neighborhood_member"
                ],
                "candidate_neighborhood_id": row["candidate_neighborhood_id"],
                "exact_gt_singleton_member": row[
                    "exact_gt_singleton_member"
                ],
                "exact_gt_singleton_id": row["exact_gt_singleton_id"],
                "mechanism_decision_rules_sha256": mechanism_rules_sha256,
                "native_repetition_penalty_stratum": 1.0,
                "raw_model_logprob": {"complete_box_logprob_sum": score},
                "auxiliary_policy_scores": {"rp_1_10": 99999.0},
            }
        )
    _write_jsonl(scores_path, scores)

    merge_path = tmp_path / "merge-receipt.json"
    shard_receipt_path = tmp_path / "successor-score-shard-receipt.json"
    _write_json(
        shard_receipt_path,
        {
            "likelihood_channels": {
                "raw": "unmodified fp32 lm-head log-softmax under use_cache=False"
            },
            "scoring_backend_admission": {
                "selected_backend": "full_reforward_fp32",
                "cache_enabled": False,
                "use_cache": False,
                "atol": 1e-5,
                "rtol": 1e-5,
                "batched_reforward_admission": {
                    "status": "passed",
                    "requested_batch_size": 1,
                    "effective_batch_size": 1,
                },
            },
        },
    )
    merge = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "generic_arbitrary_role_merger": True,
        "decision_channel": {
            "name": SUCCESSOR_DECISION_CHANNEL,
            "primary_repetition_penalty_stratum": 1.0,
            "auxiliary_policy_is_not_a_model_likelihood": True,
        },
        "successor_scorer_provenance": {
            "row_schema_version": SUCCESSOR_SCORE_ROW_SCHEMA_VERSION,
            "receipt_schema_version": SUCCESSOR_SCORE_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
        },
        "imported_predecessor_primitive_provenance": {
            "predecessor_primitives_file": PREDECESSOR_PRIMITIVES_FILE,
            "predecessor_primitives_file_sha256": "a" * 64,
        },
        "source_digests": {
            "fn_mechanism_registry": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
                "registry_digest": json.loads(
                    registry_path.read_text(encoding="utf-8")
                )["registry_digest"],
            },
            "fixed_budget_candidates": {"path": str(fixed_path), "sha256": sha256_file(fixed_path)},
            "decision_rules": {"path": str(rules_path), "sha256": sha256_file(rules_path)},
            "mechanism_decision_rules": {
                "path": str(mechanism_rules_path),
                "sha256": mechanism_rules_sha256,
                "self_digest": json.loads(
                    mechanism_rules_path.read_text(encoding="utf-8")
                )["self_digest"],
                "parent_execution_rules_sha256": sha256_file(rules_path),
            },
        },
        "planner_receipt": {
            "path": str(planner_path),
            "sha256": sha256_file(planner_path),
            "receipt_digest": json.loads(planner_path.read_text())["receipt_digest"],
        },
        "context_selection": {
            "selected_context_ids": sorted({row["owner_context_id"] for row in fixed_rows})
        },
        "selected_rungs": ["L1"],
        "identity_projection_sha256": "b" * 64,
        "shards": [
            {
                "receipt": {
                    "path": str(shard_receipt_path),
                    "sha256": sha256_file(shard_receipt_path),
                }
            }
        ],
        "output_artifacts": {
            "merged_scores": {"path": str(scores_path), "sha256": sha256_file(scores_path), "row_count": len(scores)}
        },
    }
    _write_json(merge_path, merge)

    attestation_dir = tmp_path / "attestation"
    attestation = attest(
        merge_receipt_path=merge_path,
        decision_rules_path=rules_path,
        mechanism_decision_rules_path=mechanism_rules_path,
        run_mode="scale",
        output_dir=attestation_dir,
    )
    assert attestation["schema_version"] == RUN_ATTESTATION_SCHEMA_VERSION
    assert MERGE_SCHEMA_VERSION == MERGE_RECEIPT_SCHEMA_VERSION
    attestation_path = attestation_dir / ATTESTATION_NAME
    return {
        "registry": registry_path,
        "planner": planner_path,
        "fixed": fixed_path,
        "scores": scores_path,
        "attestation": attestation_path,
    }


def _analyze(paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    return analyze_sorted_fn_mechanisms(
        fn_mechanism_registry=paths["registry"],
        planner_receipt=paths["planner"],
        fixed_budget_candidates=paths["fixed"],
        merged_scores=paths["scores"],
        run_attestation=paths["attestation"],
        output_dir=output_dir,
    )


def _behavior_contract_fixture(
    tmp_path: Path,
    paths: dict[str, Path],
    *,
    description_equivalence: Any,
    description_equivalence_receipt: Mapping[str, Any],
    tag: str,
) -> Path:
    """Build a minimal, exactly-bound behavior output naming zero roles.

    Zero selected roles keeps every other contract binding (registry,
    landscape, landscape admission) trivially satisfied so the fixture
    isolates the description-equivalence receipt binding under test.
    """
    loaded = _load_and_bind_inputs(
        registry_path=paths["registry"],
        planner_receipt_path=paths["planner"],
        fixed_budget_path=paths["fixed"],
        merged_scores_path=paths["scores"],
        run_attestation_path=paths["attestation"],
        behavior_output_path=None,
    )
    mechanism_decision_rules_binding = {
        "sha256": sha256_file(loaded["mechanism_rules_path"]),
        "self_digest": loaded["mechanism_rules_digest"],
        "parent_execution_rules_sha256": sha256_file(loaded["rules_path"]),
        "planner_receipt_digest": loaded["planner_digest"],
        "registry_digest": loaded["registry_digest"],
        "fixed_budget_candidates_sha256": sha256_file(loaded["fixed_budget_path"]),
    }
    admission_content = {
        "schema_version": BEHAVIOR_LANDSCAPE_ADMISSION_SCHEMA_VERSION,
        "status": "passed",
        "registry_digest": loaded["registry_digest"],
        "landscape_receipt_sha256": sha256_file(loaded["merge_path"]),
        "mechanism_decision_rules": mechanism_decision_rules_binding,
        "admitted_roles": [],
    }
    admission_path = tmp_path / f"admission-{tag}.json"
    _write_json(admission_path, admission_content)
    contract_content = {
        "registry": {
            "file_sha256": sha256_file(paths["registry"]),
            "registry_digest": loaded["registry_digest"],
        },
        "landscape": {"sha256": sha256_file(loaded["merge_path"])},
        "landscape_admission": {
            "path": str(admission_path),
            "sha256": sha256_file(admission_path),
            "schema_version": BEHAVIOR_LANDSCAPE_ADMISSION_SCHEMA_VERSION,
            "admitted_role_ids": [],
        },
        "selected_role_ids": [],
        "description_equivalence": description_equivalence,
        "description_equivalence_receipt": description_equivalence_receipt,
    }
    behavior_content = {
        "schema_version": BEHAVIOR_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "contract": contract_content,
        "selected_role_prefixes": [],
    }
    behavior_document = {
        **behavior_content,
        "output_content_sha256": sha256_json(behavior_content),
    }
    behavior_path = tmp_path / f"behavior-{tag}.json"
    _write_json(behavior_path, behavior_document)
    return behavior_path


def test_raw_channel_is_used_and_rp_policy_view_is_ignored(tmp_path: Path) -> None:
    paths = _sealed_fixture(tmp_path)
    document = _analyze(paths, tmp_path / "out")
    target_entry = next(
        row for row in document["owner_dispositions"] if row["gt_owner_id"] == "gt:1:0"
    )
    control_entry = next(
        row for row in document["owner_dispositions"] if row["gt_owner_id"] == "gt:1:1"
    )
    assert document["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert document["quantitative_functional"]["rp_policy_views_used_for_primary_decisions"] is False
    assert target_entry["final_set_gain"] is None
    assert target_entry["source_cohort"] == "no_free_spatial_support"
    assert control_entry["source_cohort"] == "greedy_strict_present"
    admission = document["behavior_landscape_admission"]
    assert admission["status"] == "passed"
    assert {row["role_id"] for row in admission["admitted_roles"]} == {
        "root:gt:1:0",
        "root:gt:1:1",
    }


def test_v2_attestor_analyzer_behavior_landscape_contract_gate_cpu_only(
    tmp_path: Path,
) -> None:
    paths = _sealed_fixture(tmp_path)
    output_dir = tmp_path / "out"
    analysis = _analyze(paths, output_dir)
    admission_path = output_dir / "behavior-landscape-admission.json"
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    merge_path = Path(
        json.loads(paths["attestation"].read_text(encoding="utf-8"))[
            "merge_receipt"
        ]["path"]
    )

    assert analysis["behavior_landscape_admission"]["status"] == "passed"
    _document, receipt = validate_landscape_admission(
        admission_path,
        registry_digest=registry["registry_digest"],
        landscape_receipt_sha256=sha256_file(merge_path),
        mechanism_decision_rules_binding=analysis[
            "behavior_landscape_admission"
        ]["mechanism_decision_rules"],
        selected_roles=registry["smoke"]["roles"],
        selected_rungs=["L1"],
    )
    assert receipt["admitted_role_ids"] == ["root:gt:1:0", "root:gt:1:1"]


def test_create_identical_and_digest_mismatch(tmp_path: Path) -> None:
    paths = _sealed_fixture(tmp_path)
    first = _analyze(paths, tmp_path / "out")
    second = _analyze(paths, tmp_path / "out")
    assert first == second

    rows = [json.loads(line) for line in paths["fixed"].read_text().splitlines()]
    rows[0]["family_id"] = "tampered"
    _write_jsonl(paths["fixed"], rows)
    with pytest.raises(MechanismAnalysisError, match="digest mismatch"):
        _analyze(paths, tmp_path / "other-out")


def test_mechanism_rules_and_plan_receipt_digests_are_required(tmp_path: Path) -> None:
    paths = _sealed_fixture(tmp_path)
    planner = json.loads(paths["planner"].read_text(encoding="utf-8"))
    mechanism_path = Path(planner["outputs"]["mechanism_decision_rules"]["path"])
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    mechanism["calibration"]["lower_quantile"] = 0.2
    _write_json(mechanism_path, mechanism)
    with pytest.raises(MechanismAnalysisError, match="digest mismatch"):
        _analyze(paths, tmp_path / "rules-drift")

    paths = _sealed_fixture(tmp_path / "plan")
    planner = json.loads(paths["planner"].read_text(encoding="utf-8"))
    planner["neighborhood_consistency"]["status"] = "failed"
    _write_json(paths["planner"], planner)
    with pytest.raises(MechanismAnalysisError, match="digest mismatch"):
        _analyze(paths, tmp_path / "plan-drift")


@pytest.mark.parametrize(
    ("artifact", "stale_schema", "message"),
    [
        ("merge", "sorted_fn_successor_score_shard_merge.v1", "merge receipt schema"),
        (
            "attestation",
            "sorted_fn_successor_score_run_attestation.v1",
            "run attestation schema",
        ),
    ],
)
def test_stale_v1_producer_artifacts_are_rejected(
    tmp_path: Path, artifact: str, stale_schema: str, message: str
) -> None:
    paths = _sealed_fixture(tmp_path)
    if artifact == "attestation":
        path = paths["attestation"]
    else:
        attestation = json.loads(paths["attestation"].read_text(encoding="utf-8"))
        path = Path(attestation["merge_receipt"]["path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = stale_schema
    _write_json(path, document)
    if artifact == "merge":
        attestation = json.loads(paths["attestation"].read_text(encoding="utf-8"))
        attestation["merge_receipt"]["sha256"] = sha256_file(path)
        _write_json(paths["attestation"], attestation)
    with pytest.raises(MechanismAnalysisError, match=message):
        _analyze(paths, tmp_path / "out")


def test_l2_requires_prospective_admission_schema_constant() -> None:
    # The constant itself is part of the public analyzer/manifest contract.
    assert L2_ADMISSION_SCHEMA_VERSION == "sorted-fn-l2-prospective-admission.v1"


def _not_supplied_receipt() -> dict[str, Any]:
    return {
        "semantic_drift_decision_eligible": False,
        "status": "not_supplied_semantic_relation_neutral",
    }


def test_load_and_bind_inputs_binds_receipt_not_raw_equivalence_document(
    tmp_path: Path,
) -> None:
    """Regression: description_equivalence carries the raw (possibly-null)
    document; description_equivalence_receipt carries the sealed binding.
    Reading the raw-document field previously raised on the common,
    semantically-neutral case where no equivalence artifact was supplied.
    """
    paths = _sealed_fixture(tmp_path)
    behavior_path = _behavior_contract_fixture(
        tmp_path,
        paths,
        description_equivalence=None,
        description_equivalence_receipt=_not_supplied_receipt(),
        tag="not-supplied",
    )
    loaded = _load_and_bind_inputs(
        registry_path=paths["registry"],
        planner_receipt_path=paths["planner"],
        fixed_budget_path=paths["fixed"],
        merged_scores_path=paths["scores"],
        run_attestation_path=paths["attestation"],
        behavior_output_path=behavior_path,
    )
    assert loaded["behavior"]["contract"]["description_equivalence"] is None
    assert (
        loaded["behavior"]["contract"]["description_equivalence_receipt"]
        == _not_supplied_receipt()
    )


def _frozen_equivalence_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """A frozen equivalence document, its binding receipt, and the raw
    contract projection the runner actually uses (document +
    alias_to_category; see validate_description_equivalence in
    run_sorted_fn_successor_behavior.py).
    """
    equivalence_content = {
        "schema_version": DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION,
        "status": "frozen",
        "categories": {"person": ["pedestrian"]},
    }
    equivalence_document = {
        **equivalence_content,
        "equivalence_digest": sha256_json(equivalence_content),
    }
    equivalence_path = tmp_path / "equivalence.json"
    _write_json(equivalence_path, equivalence_document)
    receipt = {
        "status": "frozen",
        "path": str(equivalence_path),
        "sha256": sha256_file(equivalence_path),
        "equivalence_digest": equivalence_document["equivalence_digest"],
        "semantic_drift_decision_eligible": True,
    }
    raw_projection = {
        **equivalence_document,
        "alias_to_category": {"pedestrian": "person", "person": "person"},
    }
    return equivalence_document, receipt, raw_projection


def test_load_and_bind_inputs_accepts_frozen_equivalence_receipt(
    tmp_path: Path,
) -> None:
    paths = _sealed_fixture(tmp_path)
    _equivalence_document, receipt, raw_projection = _frozen_equivalence_fixture(tmp_path)
    behavior_path = _behavior_contract_fixture(
        tmp_path,
        paths,
        description_equivalence=raw_projection,
        description_equivalence_receipt=receipt,
        tag="frozen",
    )
    loaded = _load_and_bind_inputs(
        registry_path=paths["registry"],
        planner_receipt_path=paths["planner"],
        fixed_budget_path=paths["fixed"],
        merged_scores_path=paths["scores"],
        run_attestation_path=paths["attestation"],
        behavior_output_path=behavior_path,
    )
    assert (
        loaded["behavior"]["contract"]["description_equivalence_receipt"]["status"]
        == "frozen"
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        # P1: receipt.equivalence_digest must be cross-checked against the
        # frozen document's own digest, not left unread once path/sha256 bind.
        (
            lambda receipt, raw_projection: (
                {**receipt, "equivalence_digest": "0" * 64},
                raw_projection,
            ),
            "digest does not match",
        ),
        # The raw projection actually used by the runner must be bound too:
        # a bare on-disk document (no alias_to_category) is not what
        # generation used.
        (
            lambda receipt, raw_projection: (
                receipt,
                {k: v for k, v in raw_projection.items() if k != "alias_to_category"},
            ),
            "alias_to_category",
        ),
        (
            lambda receipt, raw_projection: (
                receipt,
                {**raw_projection, "categories": {"person": ["pedestrian", "cyclist"]}},
            ),
            "diverges from the frozen document",
        ),
        # Adversarial: alias target tamper. The frozen categories are
        # untouched, but the contract retargets "pedestrian" to the wrong
        # category; the deterministic reconstruction from categories must
        # still win over whatever the contract claims.
        (
            lambda receipt, raw_projection: (
                receipt,
                {
                    **raw_projection,
                    "alias_to_category": {
                        **raw_projection["alias_to_category"],
                        "pedestrian": "vehicle",
                    },
                },
            ),
            "diverges from the deterministic reconstruction",
        ),
        # Adversarial: omitted alias. The contract's alias_to_category drops
        # an alias the frozen categories declare.
        (
            lambda receipt, raw_projection: (
                receipt,
                {
                    **raw_projection,
                    "alias_to_category": {
                        key: value
                        for key, value in raw_projection["alias_to_category"].items()
                        if key != "pedestrian"
                    },
                },
            ),
            "diverges from the deterministic reconstruction",
        ),
    ],
)
def test_load_and_bind_inputs_rejects_mismatched_frozen_equivalence_binding(
    tmp_path: Path, mutate: Any, match: str
) -> None:
    paths = _sealed_fixture(tmp_path)
    _equivalence_document, receipt, raw_projection = _frozen_equivalence_fixture(tmp_path)
    receipt, raw_projection = mutate(receipt, raw_projection)
    behavior_path = _behavior_contract_fixture(
        tmp_path,
        paths,
        description_equivalence=raw_projection,
        description_equivalence_receipt=receipt,
        tag="mismatch",
    )
    with pytest.raises(MechanismAnalysisError, match=match):
        _load_and_bind_inputs(
            registry_path=paths["registry"],
            planner_receipt_path=paths["planner"],
            fixed_budget_path=paths["fixed"],
            merged_scores_path=paths["scores"],
            run_attestation_path=paths["attestation"],
            behavior_output_path=behavior_path,
        )


def test_load_and_bind_inputs_rejects_uncrossable_category_collision_in_frozen_document(
    tmp_path: Path,
) -> None:
    """Adversarial: the frozen document itself declares an alias
    ("walker") under two categories. The producer would have refused to
    freeze this; the analyzer must independently refuse to admit it too,
    rather than reconstructing a silently-arbitrary winner.
    """
    paths = _sealed_fixture(tmp_path)
    equivalence_content = {
        "schema_version": DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION,
        "status": "frozen",
        "categories": {"person": ["walker"], "pedestrian_type": ["walker"]},
    }
    equivalence_document = {
        **equivalence_content,
        "equivalence_digest": sha256_json(equivalence_content),
    }
    equivalence_path = tmp_path / "equivalence.json"
    _write_json(equivalence_path, equivalence_document)
    receipt = {
        "status": "frozen",
        "path": str(equivalence_path),
        "sha256": sha256_file(equivalence_path),
        "equivalence_digest": equivalence_document["equivalence_digest"],
        "semantic_drift_decision_eligible": True,
    }
    raw_projection = {
        **equivalence_document,
        "alias_to_category": {"walker": "person", "person": "person", "pedestrian_type": "pedestrian_type"},
    }
    behavior_path = _behavior_contract_fixture(
        tmp_path,
        paths,
        description_equivalence=raw_projection,
        description_equivalence_receipt=receipt,
        tag="collision",
    )
    with pytest.raises(MechanismAnalysisError, match="cannot be deterministically reconstructed"):
        _load_and_bind_inputs(
            registry_path=paths["registry"],
            planner_receipt_path=paths["planner"],
            fixed_budget_path=paths["fixed"],
            merged_scores_path=paths["scores"],
            run_attestation_path=paths["attestation"],
            behavior_output_path=behavior_path,
        )


def test_load_and_bind_inputs_accepts_multi_category_reconstructed_alias_map(
    tmp_path: Path,
) -> None:
    """Valid derived map: a correct, multi-category alias_to_category must
    still be admitted after wiring in exact reconstruction equality.
    """
    paths = _sealed_fixture(tmp_path)
    equivalence_content = {
        "schema_version": DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION,
        "status": "frozen",
        "categories": {
            "person": ["pedestrian", "walker"],
            "vehicle": ["bike", "cyclist"],
        },
    }
    equivalence_document = {
        **equivalence_content,
        "equivalence_digest": sha256_json(equivalence_content),
    }
    equivalence_path = tmp_path / "equivalence.json"
    _write_json(equivalence_path, equivalence_document)
    receipt = {
        "status": "frozen",
        "path": str(equivalence_path),
        "sha256": sha256_file(equivalence_path),
        "equivalence_digest": equivalence_document["equivalence_digest"],
        "semantic_drift_decision_eligible": True,
    }
    raw_projection = {
        **equivalence_document,
        "alias_to_category": {
            "person": "person",
            "pedestrian": "person",
            "walker": "person",
            "vehicle": "vehicle",
            "bike": "vehicle",
            "cyclist": "vehicle",
        },
    }
    behavior_path = _behavior_contract_fixture(
        tmp_path,
        paths,
        description_equivalence=raw_projection,
        description_equivalence_receipt=receipt,
        tag="multi-category",
    )
    loaded = _load_and_bind_inputs(
        registry_path=paths["registry"],
        planner_receipt_path=paths["planner"],
        fixed_budget_path=paths["fixed"],
        merged_scores_path=paths["scores"],
        run_attestation_path=paths["attestation"],
        behavior_output_path=behavior_path,
    )
    assert (
        loaded["behavior"]["contract"]["description_equivalence"]["alias_to_category"]
        == raw_projection["alias_to_category"]
    )


@pytest.mark.parametrize(
    ("receipt", "match"),
    [
        ("not-a-mapping", "must be an object"),
        (None, "must be an object"),
        (
            {"status": "not_supplied_semantic_relation_neutral"},
            "eligibility must be a boolean",
        ),
        (
            {"semantic_drift_decision_eligible": "yes", "status": "frozen"},
            "eligibility must be a boolean",
        ),
        (
            {
                "semantic_drift_decision_eligible": True,
                "status": "not_supplied_semantic_relation_neutral",
            },
            "status is not frozen",
        ),
        (
            {"semantic_drift_decision_eligible": False, "status": "frozen"},
            "status/eligibility mismatch",
        ),
    ],
)
def test_load_and_bind_inputs_rejects_malformed_equivalence_receipt(
    tmp_path: Path, receipt: Any, match: str
) -> None:
    paths = _sealed_fixture(tmp_path)
    behavior_path = _behavior_contract_fixture(
        tmp_path,
        paths,
        description_equivalence=None,
        description_equivalence_receipt=receipt,
        tag="malformed",
    )
    if receipt is None:
        # Simulate a contract that omits the key entirely rather than
        # writing an explicit null, since a JSON `null` and a missing key
        # both decode to `.get(...)` returning `None`. The self-digest must
        # be recomputed over the mutated document to isolate this from the
        # unrelated digest-mismatch failure mode.
        behavior_document = json.loads(behavior_path.read_text(encoding="utf-8"))
        del behavior_document["contract"]["description_equivalence_receipt"]
        del behavior_document["output_content_sha256"]
        behavior_document["output_content_sha256"] = sha256_json(behavior_document)
        _write_json(behavior_path, behavior_document)
    with pytest.raises(MechanismAnalysisError, match=match):
        _load_and_bind_inputs(
            registry_path=paths["registry"],
            planner_receipt_path=paths["planner"],
            fixed_budget_path=paths["fixed"],
            merged_scores_path=paths["scores"],
            run_attestation_path=paths["attestation"],
            behavior_output_path=behavior_path,
        )


def test_role_results_by_id_reads_equivalence_receipt_not_raw_document(
    tmp_path: Path,
) -> None:
    """Regression for the owner-mechanism-construction equivalence read:
    eligibility must come from description_equivalence_receipt, never from
    the raw (possibly-null) description_equivalence document.
    """
    prediction_path = tmp_path / "predictions.jsonl"
    _write_jsonl(prediction_path, [{"pred_row_id": "p1"}])
    registry = {
        "smoke": {
            "roles": [
                {"role_id": "root:gt:1:0", "prefix": {"prefix_pred_row_ids": []}},
            ]
        },
        "sources": {
            "prediction_row_ledger": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            }
        },
    }

    def role_results(contract: dict[str, Any]) -> dict[str, Mapping[str, Any]]:
        behavior = {
            "contract": contract,
            "policy_views": [
                {
                    "repetition_penalty": 1.0,
                    "roles": [{"role_id": "root:gt:1:0"}],
                }
            ],
        }
        return _role_results_by_id(behavior, None, registry)

    not_supplied = role_results(
        {
            "description_equivalence": None,
            "description_equivalence_receipt": _not_supplied_receipt(),
        }
    )
    assert (
        not_supplied["root:gt:1:0"]["_validated_semantic_drift_eligible"] is False
    )

    frozen = role_results(
        {
            "description_equivalence": None,
            "description_equivalence_receipt": {
                "semantic_drift_decision_eligible": True,
                "status": "frozen",
            },
        }
    )
    assert frozen["root:gt:1:0"]["_validated_semantic_drift_eligible"] is True

    # A raw document under the wrong (description_equivalence) key that
    # looks eligible must never leak eligibility when the receipt says
    # otherwise; this guards against reverting to the pre-fix field name.
    wrong_field_eligible = role_results(
        {
            "description_equivalence": {
                "semantic_drift_decision_eligible": True,
                "status": "frozen",
            },
            "description_equivalence_receipt": _not_supplied_receipt(),
        }
    )
    assert (
        wrong_field_eligible["root:gt:1:0"]["_validated_semantic_drift_eligible"]
        is False
    )


def test_role_results_by_id_rejects_non_mapping_equivalence_receipt(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    _write_jsonl(prediction_path, [{"pred_row_id": "p1"}])
    registry = {
        "smoke": {
            "roles": [
                {"role_id": "root:gt:1:0", "prefix": {"prefix_pred_row_ids": []}},
            ]
        },
        "sources": {
            "prediction_row_ledger": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            }
        },
    }
    behavior = {
        "contract": {"description_equivalence_receipt": "not-a-mapping"},
        "policy_views": [
            {
                "repetition_penalty": 1.0,
                "roles": [{"role_id": "root:gt:1:0"}],
            }
        ],
    }
    with pytest.raises(MechanismAnalysisError, match="must be an object"):
        _role_results_by_id(behavior, None, registry)
