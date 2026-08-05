"""Focused tests for the full-canvas visual-token-budget intervention analyzer.

Compact synthetic fixtures only: the point is to pin the *decision surface*
(threshold derivation, support counts, retention gates, the minimum-decisive
block and every fail-closed condition), not to re-run a census.  Support
semantics themselves are the predecessor's code, so they are exercised through
``merge._summarize_bound`` rather than reimplemented here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import merge_sorted_owner_accessibility_census_shards as merge  # noqa: E402
from scripts.research import (  # noqa: E402
    prepare_sorted_full_canvas_token_budget_intervention as prepare,
)
from scripts.research import (  # noqa: E402
    analyze_sorted_full_canvas_token_budget_intervention as analyzer,
)
from scripts.research import (  # noqa: E402
    score_sorted_full_canvas_token_budget_intervention_shard as successor_module,
)

RESTRICTED = prepare.RESTRICTED_IMAGE_ID


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def owner_context_row(
    *,
    owner_id: str,
    image_id: str,
    context_id: str,
    boundary_index: int,
    description: str = "person",
    peak_lift: float = 5.0,
    local_concentration: float = 5.0,
    loop_tail: bool = False,
    value: float = -10.0,
) -> dict[str, Any]:
    """One ``(owner, context)`` feature row in the predecessor's shape."""

    bound_block = {
        "value": value,
        "candidate_id": "cand:x",
        "rank": 1,
        "count": 4,
        "log_posterior": -1.0,
        "unique_population_size": 4,
        "peak_lift": peak_lift,
        "local_concentration": local_concentration,
        "bank_median": value - local_concentration,
        "bank_size": 4,
        "geometry": None,
    }
    return {
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "split": "discovery",
        "context_id": context_id,
        "boundary_index": boundary_index,
        "normalized_description": description,
        "loop_marking": {"loop_tail": loop_tail},
        "frontier_features": {
            "frontier_present": boundary_index > 0,
            "signed_frontier_ordinal_distance": boundary_index or None,
            "abs_signed_frontier_ordinal_distance": boundary_index or None,
            "signed_frontier_sort_axis_pixel_distance": None,
            "frontier_overlap": {"intersection_over_union": None},
            "same_description_owners_ahead_of_frontier": 1,
            "passed_state": "root_no_frontier" if boundary_index == 0 else "ahead_of_frontier",
        },
        "localization": {
            "generator_local_max_excluding_other_owner_strict": {
                "ambiguity_excluded_l": dict(bound_block),
                "ambiguity_included_u": dict(bound_block),
            },
            "exact_anchor_score": {"value": value},
            "strict_assigned_max": {"value": value},
        },
        "owner_competition_l": {"rank": 1, "population_size": 2, "margin_to_best_owner": 0.0},
        "owner_competition_u": {"rank": 1, "population_size": 2, "margin_to_best_owner": 0.0},
        "category_proposal_channel": {"boundary_gate": {}, "category_routing_event": {}},
    }


def stub_plan(*, owners: dict[str, Any], contexts: dict[str, Any], sidecars: list[Any]):
    capture_rules = planner.build_capture_rules()
    return merge.PlanBundle(
        plan_dir=Path("/nowhere"),
        receipt={"receipt_content_sha256": "base", "capture_rules_sha256": "rules"},
        capture_rules=capture_rules,
        images={},
        owners=owners,
        categories={},
        contexts=contexts,
        candidates={},
        query_groups={},
        native_sidecars=sidecars,
        shards={},
    )


def calibration_plan(observations: list[tuple[str, float, float]]):
    """A plan whose discovery TPs map to due contexts carrying given features."""

    owners: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    sidecars: list[Any] = []
    rows: list[dict[str, Any]] = []
    for index, (owner_id, lift, concentration) in enumerate(observations):
        context_id = f"7511:boundary-{index:03d}"
        owners[owner_id] = {
            "gt_owner_id": owner_id,
            "image_id": "7511",
            "split": "discovery",
            "normalized_description": "person",
            "native_true_positive": True,
            "native_strict_match_pred_row_ids": [f"pred:{index}"],
        }
        contexts[context_id] = {"context_id": context_id, "image_id": "7511"}
        sidecars.append({"pred_row_id": f"pred:{index}", "row_index": index})
        rows.append(
            owner_context_row(
                owner_id=owner_id,
                image_id="7511",
                context_id=context_id,
                boundary_index=index,
                peak_lift=lift,
                local_concentration=concentration,
            )
        )
    return stub_plan(owners=owners, contexts=contexts, sidecars=sidecars), rows


OVERLAY = {"overlay_content_sha256": "overlay-digest"}

#: The unit freezes exactly this many discovery true-positive due-context
#: observations, so the happy-path fixtures below carry exactly that many.
N_CALIBRATION = analyzer.EXPECTED_DISCOVERY_TP_CALIBRATION_OBSERVATIONS


def calibration_overlay(owner_ids: list[str]) -> dict[str, Any]:
    """An overlay that declares its frozen discovery-TP calibration cohort."""

    return {
        "overlay_content_sha256": "overlay-digest",
        "query_group_selection": {"cohort_owner_ids": {"discovery_tp_calibration": owner_ids}},
    }


def linear_observations(count: int = N_CALIBRATION) -> list[tuple[str, float, float]]:
    return [(f"gt:7511:{i}", float(i + 1), float(i + 1) + 10.0) for i in range(count)]


# ---------------------------------------------------------------------------
# Threshold derivation
# ---------------------------------------------------------------------------


def test_thresholds_are_the_sealed_q10_of_the_treatment_arms_own_controls():
    observations = linear_observations()
    plan, rows = calibration_plan(observations)
    calibration = analyzer.calibrate_treatment_support(plan, rows, overlay=OVERLAY)

    contract = merge.load_support_contract(plan)
    assert calibration.quantile == contract.primary_quantile == 0.1
    assert calibration.epsilon == contract.support_epsilon == 0.002
    assert calibration.observation_count == N_CALIBRATION == 70
    lifts = [row[1] for row in observations]
    concentrations = [row[2] for row in observations]
    assert calibration.theta_peak_lift == pytest.approx(merge._quantile(lifts, 0.1))
    assert calibration.theta_local_concentration == pytest.approx(
        merge._quantile(concentrations, 0.1)
    )
    described = calibration.describe()
    assert described["rank_is_not_a_support_input"] is True
    assert described["rule"].startswith("peak_lift >= theta_peak_lift + epsilon")


def test_calibration_accepts_the_overlays_declared_frozen_cohort():
    observations = linear_observations()
    plan, rows = calibration_plan(observations)
    overlay = calibration_overlay([row[0] for row in observations])
    calibration = analyzer.calibrate_treatment_support(plan, rows, overlay=overlay)
    assert calibration.observation_count == N_CALIBRATION


def test_support_is_a_conjunction_with_the_sealed_epsilon():
    plan, rows = calibration_plan([(f"gt:7511:{i}", 2.0, 3.0) for i in range(N_CALIBRATION)])
    calibration = analyzer.calibrate_treatment_support(plan, rows, overlay=OVERLAY)
    theta_lift = calibration.theta_peak_lift
    theta_conc = calibration.theta_local_concentration
    epsilon = calibration.epsilon

    assert calibration.clears(
        {"peak_lift": theta_lift + epsilon, "local_concentration": theta_conc + epsilon}
    )
    # One statistic just under the band is enough to fail the conjunction.
    assert not calibration.clears(
        {
            "peak_lift": theta_lift + epsilon,
            "local_concentration": theta_conc + epsilon - 1e-6,
        }
    )
    assert not calibration.clears(
        {
            "peak_lift": theta_lift + epsilon - 1e-6,
            "local_concentration": theta_conc + epsilon,
        }
    )
    assert calibration.clears({"peak_lift": None, "local_concentration": 9.0}) is None


def test_calibration_excludes_loop_tails_and_untested_due_contexts():
    """Exclusions are recorded, and the frozen count is met by the survivors."""

    plan, rows = calibration_plan(linear_observations(N_CALIBRATION + 2))
    rows[0]["loop_marking"]["loop_tail"] = True
    rows.pop(1)
    calibration = analyzer.calibrate_treatment_support(plan, rows, overlay=OVERLAY)
    assert calibration.observation_count == N_CALIBRATION
    reasons = {row.get("reason") for row in calibration.exclusions}
    assert "due_context_is_a_loop_tail" in reasons
    assert "owner_not_tested_at_its_due_context" in reasons


def test_missing_calibration_controls_fail_closed():
    plan, rows = calibration_plan(linear_observations())
    with pytest.raises(analyzer.AnalysisContractError, match="cannot be calibrated"):
        analyzer.calibrate_treatment_support(plan, [], overlay=OVERLAY)

    rows[0]["localization"]["generator_local_max_excluding_other_owner_strict"][
        "ambiguity_included_u"
    ]["peak_lift"] = math.inf
    with pytest.raises(analyzer.AnalysisContractError, match="not finite"):
        analyzer.calibrate_treatment_support(plan, rows, overlay=OVERLAY)


def test_one_missing_calibration_observation_cannot_move_the_thresholds():
    """A short calibration set is refused, not silently re-quantiled.

    Dropping the weakest control would raise both thresholds and make the
    treatment arm look harder to pass than the predecessor's, so the frozen
    population size is a gate.
    """

    observations = linear_observations()
    plan, rows = calibration_plan(observations)
    short = [row for row in rows if row["gt_owner_id"] != observations[0][0]]
    assert len(short) == N_CALIBRATION - 1
    with pytest.raises(analyzer.AnalysisContractError, match="this unit freezes 70"):
        analyzer.calibrate_treatment_support(plan, short, overlay=OVERLAY)

    # And with one extra observation it is equally refused.
    plan_long, rows_long = calibration_plan(linear_observations(N_CALIBRATION + 1))
    with pytest.raises(analyzer.AnalysisContractError, match="this unit freezes 70"):
        analyzer.calibrate_treatment_support(plan_long, rows_long, overlay=OVERLAY)


def test_calibration_population_must_be_the_overlays_frozen_cohort():
    observations = linear_observations()
    plan, rows = calibration_plan(observations)
    declared = [row[0] for row in observations]

    swapped = [*declared[:-1], "gt:7511:not-in-this-arm"]
    with pytest.raises(analyzer.AnalysisContractError, match="frozen discovery"):
        analyzer.calibrate_treatment_support(
            plan, rows, overlay=calibration_overlay(swapped)
        )

    with pytest.raises(analyzer.AnalysisContractError, match="this unit freezes 70"):
        analyzer.calibrate_treatment_support(
            plan, rows, overlay=calibration_overlay(declared[:-1])
        )


# ---------------------------------------------------------------------------
# Per-owner support
# ---------------------------------------------------------------------------


def _calibration(theta_lift: float, theta_concentration: float) -> merge.SupportCalibration:
    return merge.SupportCalibration(
        theta_peak_lift=theta_lift,
        theta_local_concentration=theta_concentration,
        epsilon=0.002,
        quantile=0.1,
        observation_count=10,
        per_category_counts={},
        sensitivity={},
        exclusions=(),
        consumed_shard_digests=(),
        capture_manifest_sha256="overlay-digest",
        category_contribution_min=20,
        underrepresented_flag="pooled_underrepresented",
        cross_context_delta_epsilon=0.004,
        statistics=("peak_lift", "local_concentration"),
    )


def test_owner_support_uses_non_loop_contexts_only_and_names_them():
    calibration = _calibration(2.0, 2.0)
    rows = [
        owner_context_row(
            owner_id="gt:1584:0",
            image_id="1584",
            context_id="1584:boundary-000",
            boundary_index=0,
            peak_lift=1.0,
            local_concentration=1.0,
        ),
        owner_context_row(
            owner_id="gt:1584:0",
            image_id="1584",
            context_id="1584:boundary-001",
            boundary_index=1,
            peak_lift=9.0,
            local_concentration=9.0,
        ),
        owner_context_row(
            owner_id="gt:1584:0",
            image_id="1584",
            context_id="1584:boundary-002",
            boundary_index=2,
            peak_lift=9.0,
            local_concentration=9.0,
            loop_tail=True,
        ),
    ]
    support = analyzer.summarize_owner_support(rows, calibration=calibration)["gt:1584:0"]
    assert support["upper_bound_u"]["usable_support"] is True
    assert support["upper_bound_u"]["usable_support_context_ids"] == ["1584:boundary-001"]
    assert support["lower_bound_l"]["usable_support"] is True
    assert len(support["tested_context_ids"]) == 3

    # With every non-loop context under the band, the loop tail alone never
    # manufactures support.
    rows[1]["localization"]["generator_local_max_excluding_other_owner_strict"][
        "ambiguity_included_u"
    ].update({"peak_lift": 0.5, "local_concentration": 0.5})
    rows[1]["localization"]["generator_local_max_excluding_other_owner_strict"][
        "ambiguity_excluded_l"
    ].update({"peak_lift": 0.5, "local_concentration": 0.5})
    support = analyzer.summarize_owner_support(rows, calibration=calibration)["gt:1584:0"]
    assert support["upper_bound_u"]["usable_support"] is False
    assert support["upper_bound_u"]["loop_tail_only_support"] is True


def _role_scope_fixture() -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """A compact union capture in which same-category contexts leak across owners."""

    contexts: dict[str, Any] = {}
    query_groups: dict[str, Any] = {}
    selected_by_image: dict[str, list[str]] = {}

    def add_context(image_id: str, index: int, *, loop_tail: bool = False) -> str:
        context_id = f"{image_id}:boundary-{index:03d}"
        contexts[context_id] = {
            "context_id": context_id,
            "image_id": image_id,
            "boundary_index": index,
            "loop_marking": {"loop_tail": loop_tail},
        }
        group_id = f"{context_id}|person"
        query_groups[group_id] = {
            "query_group_id": group_id,
            "context_id": context_id,
            "image_id": image_id,
            "normalized_description": "person",
            "status": "admitted",
        }
        selected_by_image.setdefault(image_id, []).append(group_id)
        return context_id

    primary_contexts = [
        add_context("1584", 0),
        add_context("1584", 1),
        add_context("1584", 2, loop_tail=True),
    ]
    confirmation_contexts = [add_context("7511", i) for i in range(3)]
    resolved_contexts = [add_context("16228", i) for i in range(4)]
    restricted_contexts = [add_context(RESTRICTED, i) for i in range(4)]
    calibration_contexts = [add_context("10707", i) for i in range(N_CALIBRATION)]

    summaries: list[dict[str, Any]] = [
        {
            "gt_owner_id": "gt:primary",
            "image_id": "1584",
            "normalized_description": "person",
            "upper_bound_u": {"usable_support_context_ids": []},
        },
        {
            "gt_owner_id": "gt:confirmation",
            "image_id": "7511",
            "normalized_description": "person",
            "upper_bound_u": {"usable_support_context_ids": [confirmation_contexts[1]]},
        },
        {
            "gt_owner_id": "gt:resolved",
            "image_id": "16228",
            "normalized_description": "person",
            "upper_bound_u": {
                "usable_support_context_ids": [resolved_contexts[1]],
                "primary_first_non_loop_minimal_abs_frontier": {
                    "context_id": resolved_contexts[2]
                },
            },
        },
        {
            "gt_owner_id": "gt:restricted",
            "image_id": RESTRICTED,
            "normalized_description": "person",
            "upper_bound_u": {
                "usable_support_context_ids": [],
                "primary_first_non_loop_minimal_abs_frontier": {
                    "context_id": restricted_contexts[1]
                },
                "diagnostic_best_all": {"context_id": restricted_contexts[2]},
            },
        },
    ]
    calibration_owner_ids: list[str] = []
    due: dict[str, str] = {"gt:confirmation": confirmation_contexts[0]}
    for index, context_id in enumerate(calibration_contexts):
        owner_id = f"gt:cal:{index}"
        calibration_owner_ids.append(owner_id)
        due[owner_id] = context_id
        summaries.append(
            {
                "gt_owner_id": owner_id,
                "image_id": "10707",
                "normalized_description": "person",
                "upper_bound_u": {"usable_support_context_ids": []},
            }
        )

    owner_contexts: list[dict[str, Any]] = []
    context_sets = {
        "gt:primary": primary_contexts,
        "gt:confirmation": confirmation_contexts,
        "gt:resolved": resolved_contexts,
        "gt:restricted": restricted_contexts,
        **{owner_id: calibration_contexts for owner_id in calibration_owner_ids},
    }
    summary_by_owner = {row["gt_owner_id"]: row for row in summaries}
    for owner_id, context_ids in context_sets.items():
        summary = summary_by_owner[owner_id]
        allowed_high_extra = {
            "gt:confirmation": confirmation_contexts[2],
            "gt:resolved": resolved_contexts[3],
            "gt:restricted": restricted_contexts[3],
        }.get(owner_id)
        for context_id in context_ids:
            index = int(context_id.rsplit("-", 1)[1])
            owner_contexts.append(
                owner_context_row(
                    owner_id=owner_id,
                    image_id=str(summary["image_id"]),
                    context_id=context_id,
                    boundary_index=index,
                    peak_lift=9.0 if context_id == allowed_high_extra else 0.5,
                    local_concentration=9.0 if context_id == allowed_high_extra else 0.5,
                    loop_tail=bool(contexts[context_id]["loop_marking"]["loop_tail"]),
                )
            )

    plan = stub_plan(owners={}, contexts=contexts, sidecars=[])
    plan = merge.PlanBundle(**{**plan.__dict__, "query_groups": query_groups})
    overlay = {
        "query_group_selection": {
            "selection_policy": "frozen_before_any_treatment_score_never_score_selected",
            "cohort_owner_ids": {
                "persistent_outside_restricted": ["gt:primary"],
                "persistent_restricted": ["gt:restricted"],
                "discovery_tp_calibration": calibration_owner_ids,
                "confirmation_tp_retention": ["gt:confirmation"],
                "resolved_retention": ["gt:resolved"],
            },
            "due_context_by_owner": due,
            "query_group_ids_by_image": selected_by_image,
        }
    }
    return plan, summaries, owner_contexts, overlay


def test_frozen_owner_roles_exclude_high_scoring_superset_contexts():
    plan, summaries, rows, overlay = _role_scope_fixture()
    filtered, audit = analyzer.filter_owner_contexts_to_frozen_roles(
        plan, summaries, rows, overlay=overlay
    )
    support = analyzer.summarize_owner_support(filtered, calibration=_calibration(2.0, 2.0))

    # These controls would pass only because of another owner's selected context.
    assert support["gt:confirmation"]["upper_bound_u"]["usable_support"] is False
    assert support["gt:resolved"]["upper_bound_u"]["usable_support"] is False
    assert audit["owners"]["gt:confirmation"]["excluded_superset_context_ids"] == [
        "7511:boundary-002"
    ]
    assert audit["owners"]["gt:resolved"]["excluded_superset_context_ids"] == [
        "16228:boundary-000",
        "16228:boundary-003",
    ]


def test_frozen_role_mapping_is_exact_and_auditable():
    plan, summaries, rows, overlay = _role_scope_fixture()
    filtered, audit = analyzer.filter_owner_contexts_to_frozen_roles(
        plan, summaries, rows, overlay=overlay
    )
    assert audit["role_owner_counts"]["discovery_tp_calibration"] == 70
    calibration_rows = [row for row in filtered if row["gt_owner_id"].startswith("gt:cal:")]
    assert len(calibration_rows) == 70
    assert all(
        [row["context_id"]]
        == audit["owners"][row["gt_owner_id"]]["allowed_context_ids"]
        for row in calibration_rows
    )
    assert audit["owners"]["gt:restricted"]["allowed_context_ids"] == [
        f"{RESTRICTED}:boundary-000",
        f"{RESTRICTED}:boundary-001",
        f"{RESTRICTED}:boundary-002",
    ]
    assert audit["owners"]["gt:primary"]["allowed_context_ids"] == [
        "1584:boundary-000",
        "1584:boundary-001",
    ]
    assert audit["treatment_scores_used_for_context_selection"] is False


def test_missing_intended_owner_context_fails_closed():
    plan, summaries, rows, overlay = _role_scope_fixture()
    rows = [
        row
        for row in rows
        if not (
            row["gt_owner_id"] == "gt:confirmation"
            and row["context_id"] == "7511:boundary-001"
        )
    ]
    with pytest.raises(analyzer.AnalysisContractError, match="missing 1 intended"):
        analyzer.filter_owner_contexts_to_frozen_roles(
            plan, summaries, rows, overlay=overlay
        )


# ---------------------------------------------------------------------------
# Wilson intervals
# ---------------------------------------------------------------------------


def test_wilson_interval_is_bounded_and_centred():
    empty = analyzer.wilson_interval(0, 0)
    assert empty["point"] is None
    half = analyzer.wilson_interval(5, 10)
    assert half["point"] == 0.5
    assert 0.0 < half["low"] < 0.5 < half["high"] < 1.0
    edge = analyzer.wilson_interval(0, 20)
    assert edge["low"] == 0.0 and 0.0 < edge["high"] < 0.3
    full = analyzer.wilson_interval(20, 20)
    assert full["high"] == 1.0 and 0.7 < full["low"] < 1.0


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

PERSISTENT_IMAGES = ("1584", "7511", "16228", "13348", "5001")


def build_summaries() -> list[dict[str, Any]]:
    """The frozen cohorts: 63 persistent outside 4134 (51 person), 71 TPs, 114 resolved."""

    summaries: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        summaries.append(
            {
                "upper_bound_u": {"usable_support": kwargs.pop("baseline_support", False)},
                **kwargs,
            }
        )

    for index in range(analyzer.EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED):
        description = (
            "person"
            if index < analyzer.EXPECTED_PERSISTENT_OUTSIDE_RESTRICTED_PERSON
            else "book"
        )
        add(
            gt_owner_id=f"gt:persistent:{index}",
            image_id=PERSISTENT_IMAGES[index % len(PERSISTENT_IMAGES)],
            normalized_description=description,
            split="confirmation",
            disposition=prepare.PERSISTENT_DISPOSITION,
            native_true_positive=False,
        )
    for index in range(9):
        add(
            gt_owner_id=f"gt:{RESTRICTED}:p{index}",
            image_id=RESTRICTED,
            normalized_description="person",
            split="confirmation",
            disposition=prepare.PERSISTENT_DISPOSITION,
            native_true_positive=False,
        )
    for index in range(analyzer.EXPECTED_CONFIRMATION_TP_CONTROLS):
        add(
            gt_owner_id=f"gt:tp:{index}",
            image_id="1584",
            normalized_description="person",
            split="confirmation",
            disposition="native_true_positive_calibration_control",
            native_true_positive=True,
            baseline_support=True,
        )
    for index in range(analyzer.EXPECTED_RESOLVED_SUPPORT_OWNERS):
        add(
            gt_owner_id=f"gt:resolved:{index}",
            image_id="7511",
            normalized_description="person",
            split="confirmation",
            disposition=prepare.RESOLVED_DISPOSITION,
            native_true_positive=False,
            baseline_support=True,
        )
    return summaries


def build_support(
    summaries: list[dict[str, Any]],
    *,
    recovered_persistent: int,
    recovered_person: int,
    tp_retained: int,
    resolved_retained: int,
    restricted_recovered: int = 0,
    drop_owner_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Treatment support keyed by owner, in the analyzer's projection shape."""

    def entry(supported: bool, context_id: str = "c0") -> dict[str, Any]:
        block = {
            "usable_support": supported,
            "usable_support_context_ids": [context_id] if supported else [],
            "loop_tail_only_support": False,
        }
        return {
            "lower_bound_l": dict(block),
            "upper_bound_u": dict(block),
            "tested_context_ids": [context_id],
        }

    persistent = [
        row for row in summaries if row["disposition"] == prepare.PERSISTENT_DISPOSITION
    ]
    outside = [row for row in persistent if row["image_id"] != RESTRICTED]
    person = [row for row in outside if row["normalized_description"] == "person"]
    other = [row for row in outside if row["normalized_description"] != "person"]
    restricted = [row for row in persistent if row["image_id"] == RESTRICTED]
    tps = [row for row in summaries if row["native_true_positive"]]
    resolved = [
        row for row in summaries if row["disposition"] == prepare.RESOLVED_DISPOSITION
    ]

    support: dict[str, Any] = {}
    for index, row in enumerate(person):
        support[row["gt_owner_id"]] = entry(index < recovered_person)
    remaining = max(0, recovered_persistent - recovered_person)
    for index, row in enumerate(other):
        support[row["gt_owner_id"]] = entry(index < remaining)
    for index, row in enumerate(restricted):
        support[row["gt_owner_id"]] = entry(index < restricted_recovered)
    for index, row in enumerate(tps):
        support[row["gt_owner_id"]] = entry(index < tp_retained)
    for index, row in enumerate(resolved):
        support[row["gt_owner_id"]] = entry(index < resolved_retained)
    for owner_id in drop_owner_ids:
        support.pop(owner_id, None)
    return support


def call_build_report(support: dict[str, Any], summaries: list[dict[str, Any]]):
    overlay = {
        "arm_id": prepare.ARM_ID,
        "baseline_arm_id": prepare.BASELINE_ARM_ID,
        "overlay_content_sha256": "overlay-digest",
        "claim_scope": "denser_patch_token_sampling_over_the_same_raw_optical_information",
        "base": {
            "plan_receipt_content_sha256": "base-digest",
            "predecessor_run_root": "/nowhere",
        },
        "images": {
            image_id: {
                "current": {"merged_visual_tokens": 1000},
                "treatment": {
                    "merged_visual_tokens": 1980,
                    "merged_visual_token_ratio_vs_current": 1.98,
                },
            }
            for image_id in (*PERSISTENT_IMAGES, RESTRICTED)
        },
    }
    shard_ids = [*PERSISTENT_IMAGES, RESTRICTED]
    capture_artifact_seals = {
        "schema_version": analyzer.CAPTURE_ARTIFACT_SEALS_SCHEMA_VERSION,
        "role": "exact_receipt_and_score_bytes_consumed_by_this_analyzer",
        "image_count": len(shard_ids),
        "images": {image_id: {"image_id": image_id} for image_id in shard_ids},
    }
    capture_artifact_seals["capture_artifact_seals_sha256"] = planner.sha256_json(
        capture_artifact_seals
    )
    return analyzer.build_report(
        overlay=overlay,
        summaries=summaries,
        treatment_support=support,
        calibration=_calibration(2.0, 2.0),
        shards={image_id: None for image_id in shard_ids},
        capture_identity={"status": "complete_uniform_capture_identity"},
        capture_artifact_seals=capture_artifact_seals,
    )


def test_report_counts_recovery_spread_and_the_minimum_decisive_outcome():
    summaries = build_summaries()
    support = build_support(
        summaries,
        recovered_persistent=15,
        recovered_person=12,
        tp_retained=analyzer.EXPECTED_CONFIRMATION_TP_CONTROLS,
        resolved_retained=analyzer.EXPECTED_RESOLVED_SUPPORT_OWNERS,
        restricted_recovered=2,
    )
    report = call_build_report(support, summaries)

    primary = report["primary"]
    assert primary["cohort_size"] == 63
    assert primary["person_only_cohort_size"] == 51
    assert primary["recovered"] == 15
    assert primary["recovered_person_only"] == 12
    assert primary["recovered_rate"]["successes"] == 15
    assert 0.0 < primary["recovered_rate"]["low"] < primary["recovered_rate"]["point"]
    assert primary["image_spread"] == len(primary["images_with_recovery"])
    assert primary["image_spread"] >= 3
    assert sum(primary["recovered_by_image"].values()) == 15
    assert len(primary["leave_one_image_out"]) == len(PERSISTENT_IMAGES)
    for row in primary["leave_one_image_out"]:
        assert row["cohort_size"] < 63
        assert row["recovered"] <= 15

    retention = report["retention"]
    assert retention["confirmation_true_positive"]["retention_rate"] == 1.0
    assert retention["resolved_support"]["retention_rate"] == 1.0
    assert retention["gates_intact"] is True

    restricted = report["restricted_stratum"]
    assert restricted["image_id"] == RESTRICTED
    assert restricted["cohort_size"] == 9
    assert restricted["recovered"] == 2
    assert restricted["role"].startswith("descriptive_only")

    decision = report["minimum_decisive_outcome"]
    assert decision["minimum_decisive_met"] is True
    assert decision["next_step_is_a_lead_decision"] is True
    assert report["comparison_semantics"]["raw_logprob_compared_across_arms"] is False
    assert report["comparison_semantics"]["primary_bound"] == "u"
    assert report["comparison_semantics"]["baseline_dispositions_source"].startswith(
        "predecessor_presentation"
    )


#: 90% of 71 is 63.9 and of 114 is 102.6, so the gates turn at 64 and 103.
TP_ALL = analyzer.EXPECTED_CONFIRMATION_TP_CONTROLS
RESOLVED_ALL = analyzer.EXPECTED_RESOLVED_SUPPORT_OWNERS


@pytest.mark.parametrize(
    ("recovered", "person", "tp", "resolved", "expected"),
    [
        (15, 12, TP_ALL, RESOLVED_ALL, True),
        (13, 10, TP_ALL, RESOLVED_ALL, True),
        (12, 10, TP_ALL, RESOLVED_ALL, False),  # overall below the floor
        (15, 9, TP_ALL, RESOLVED_ALL, False),  # person-only below the floor
        (15, 12, 64, RESOLVED_ALL, True),  # exactly on the confirmation TP gate
        (15, 12, 63, RESOLVED_ALL, False),  # one below it
        (15, 12, TP_ALL, 103, True),  # exactly on the resolved gate
        (15, 12, TP_ALL, 102, False),  # one below it
    ],
)
def test_minimum_decisive_gate_and_retention_gates(
    recovered: int, person: int, tp: int, resolved: int, expected: bool
):
    summaries = build_summaries()
    report = call_build_report(
        build_support(
            summaries,
            recovered_persistent=recovered,
            recovered_person=person,
            tp_retained=tp,
            resolved_retained=resolved,
        ),
        summaries,
    )
    assert report["minimum_decisive_outcome"]["minimum_decisive_met"] is expected
    confirmation = report["retention"]["confirmation_true_positive"]
    resolved_block = report["retention"]["resolved_support"]
    # The denominator is the whole frozen cohort, never the tested subset.
    assert confirmation["cohort_size"] == TP_ALL
    assert resolved_block["cohort_size"] == RESOLVED_ALL
    assert confirmation["retention_rate"] == pytest.approx(tp / TP_ALL)
    assert resolved_block["retention_rate"] == pytest.approx(resolved / RESOLVED_ALL)
    assert confirmation["gate_met"] is (tp / TP_ALL >= 0.90)
    assert resolved_block["gate_met"] is (resolved / RESOLVED_ALL >= 0.90)
    assert confirmation["denominator"].startswith("complete_frozen_cohort")


def test_cohort_drift_fails_closed():
    summaries = build_summaries()
    summaries = [row for row in summaries if row["gt_owner_id"] != "gt:persistent:0"]
    with pytest.raises(analyzer.AnalysisContractError, match="cohort drift"):
        call_build_report(
            build_support(
                summaries,
                recovered_persistent=0,
                recovered_person=0,
                tp_retained=analyzer.EXPECTED_CONFIRMATION_TP_CONTROLS,
                resolved_retained=analyzer.EXPECTED_RESOLVED_SUPPORT_OWNERS,
            ),
            summaries,
        )


def test_an_untested_primary_cohort_owner_fails_closed():
    summaries = build_summaries()
    support = build_support(
        summaries,
        recovered_persistent=15,
        recovered_person=12,
        tp_retained=analyzer.EXPECTED_CONFIRMATION_TP_CONTROLS,
        resolved_retained=analyzer.EXPECTED_RESOLVED_SUPPORT_OWNERS,
        drop_owner_ids=("gt:persistent:5",),
    )
    with pytest.raises(analyzer.AnalysisContractError, match="never tested under the treatment"):
        call_build_report(support, summaries)


@pytest.mark.parametrize(
    ("dropped", "label"),
    [
        ("gt:tp:0", "confirmation_true_positive_control"),
        ("gt:resolved:0", "resolved_tested_localization_support"),
    ],
)
def test_one_missing_retention_control_cannot_pass_the_gate(dropped: str, label: str):
    """A single untested control is a contract error, never a smaller denominator.

    Without this, dropping the one control that would have failed raises the
    rate: 70/70 tested reads as 100% retention while the frozen cohort is 71.
    """

    summaries = build_summaries()
    # Everything that *is* tested retains support, so a tested-subset
    # denominator would score a perfect 1.0 and sail through the gate.
    support = build_support(
        summaries,
        recovered_persistent=15,
        recovered_person=12,
        tp_retained=TP_ALL,
        resolved_retained=RESOLVED_ALL,
        drop_owner_ids=(dropped,),
    )
    assert dropped not in support
    with pytest.raises(
        analyzer.AnalysisContractError, match="complete frozen cohort"
    ) as excinfo:
        call_build_report(support, summaries)
    assert label in str(excinfo.value)
    assert dropped in str(excinfo.value)


@pytest.mark.parametrize(
    ("removed_prefix", "expected_size"),
    [
        ("gt:tp:", analyzer.EXPECTED_CONFIRMATION_TP_CONTROLS),
        ("gt:resolved:", analyzer.EXPECTED_RESOLVED_SUPPORT_OWNERS),
    ],
)
def test_retention_cohort_drift_fails_closed(removed_prefix: str, expected_size: int):
    summaries = build_summaries()
    support = build_support(
        summaries,
        recovered_persistent=15,
        recovered_person=12,
        tp_retained=TP_ALL,
        resolved_retained=RESOLVED_ALL,
    )
    shrunk = [
        row for row in summaries if not row["gt_owner_id"].startswith(f"{removed_prefix}0")
    ]
    assert len(shrunk) == len(summaries) - 1
    with pytest.raises(analyzer.AnalysisContractError, match="cohort drift") as excinfo:
        call_build_report(support, shrunk)
    assert str(expected_size) in str(excinfo.value)


def test_markdown_report_states_the_decision_and_the_cross_arm_rule():
    summaries = build_summaries()
    report = call_build_report(
        build_support(
            summaries,
            recovered_persistent=15,
            recovered_person=12,
            tp_retained=TP_ALL,
            resolved_retained=RESOLVED_ALL,
        ),
        summaries,
    )
    text = analyzer.render_markdown(report)
    assert prepare.ARM_ID in text
    assert "never compared across arms" in text
    assert "not a scale-up trigger" in text
    assert "15/63" in text and "12/51" in text
    assert "Retention gates" in text


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------


def _write_shard(
    root: Path,
    image_id: str,
    *,
    stamp: dict[str, Any],
    group_ids: list[str],
    scored: list[str] | None = None,
    logprob: float = -1.0,
    completeness: Any = "auto",
) -> None:
    """Write one treatment shard.

    ``completeness="auto"`` derives an honest intervention-relative block from
    the frozen ``group_ids`` versus what was actually ``scored``.  ``None``
    omits the block entirely (a capture predating the contract), and a dict
    installs an arbitrary -- possibly dishonest -- block.
    """

    directory = root / image_id
    directory.mkdir(parents=True, exist_ok=True)
    executed = list(group_ids if scored is None else scored)
    if completeness == "auto":
        missing = sorted(set(group_ids) - set(executed))
        extra = sorted(set(executed) - set(group_ids))
        completeness = {
            "relative_to": "overlay_frozen_selection_for_this_image",
            "status": (
                analyzer.COMPLETENESS_COMPLETE
                if not missing
                else successor_module.COMPLETENESS_SUBSET
            ),
            "is_complete_frozen_overlay_selection": not missing,
            "frozen_expected_query_group_count": len(set(group_ids)),
            "executed_query_group_count": len(set(executed)),
            "missing_query_group_ids": missing,
            "extra_query_group_ids": extra,
            "derived_from": "query_group_ids_carrying_score_rows",
        }
    receipt = {
        "status": "captured",
        "intervention_capture_mode": {"score_only": True},
        analyzer.ROW_STAMP_KEY: stamp,
    }
    if completeness is not None:
        receipt[analyzer.COMPLETENESS_KEY] = completeness
    (directory / "shard-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    rows = [
        {
            "query_group_id": group_id,
            "complete_box_logprob_sum": logprob,
            analyzer.ROW_STAMP_KEY: stamp,
        }
        for group_id in executed
    ]
    (directory / "census-scores.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / "proposal-surface.jsonl").write_text("", encoding="utf-8")
    (directory / "x1-distributions.jsonl").write_text("", encoding="utf-8")


def test_capture_artifact_seals_bind_exact_receipt_and_score_bytes(tmp_path: Path):
    directory = tmp_path / "7511"
    directory.mkdir()
    receipt_path = directory / "shard-receipt.json"
    score_path = directory / "census-scores.jsonl"
    receipt_path.write_text('{"intervention":{"arm_id":"t"}}', encoding="utf-8")
    score_path.write_text('{"score":1}\n', encoding="utf-8")
    shard = analyzer.TreatmentShard(
        image_id="7511",
        directory=directory,
        receipt={"intervention": {"arm_id": "t"}},
        scores=[{"score": 1}],
        proposals=[],
        x1=[],
    )
    seals = analyzer.build_capture_artifact_seals({"7511": shard})
    block = seals["images"]["7511"]
    assert block["receipt_sha256"] == planner.sha256_file(receipt_path)
    assert block["census_scores_sha256"] == planner.sha256_file(score_path)
    assert seals["capture_artifact_seals_sha256"] == planner.sha256_json(
        {key: value for key, value in seals.items() if key != "capture_artifact_seals_sha256"}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("request", "request_id"),
        ("description", "normalized_description"),
        ("rank", "rank_key"),
    ],
)
def test_analyzer_rejects_redundant_treatment_score_identity_drift(
    tmp_path: Path, mutation: str, message: str
):
    group_id = "7511:boundary-000|person"
    candidate_id = "cand:1"
    plan = stub_plan(owners={}, contexts={}, sidecars=[])
    plan.query_groups[group_id] = {
        "context_id": "7511:boundary-000",
        "normalized_description": "person",
    }
    row = {
        "request_id": f"{group_id}|{candidate_id}",
        "query_group_id": group_id,
        "candidate_id": candidate_id,
        "normalized_description": "person",
        "rank_key": {
            "image_id": "7511",
            "context_id": "7511:boundary-000",
            "normalized_description": "person",
        },
    }
    if mutation == "request":
        row["request_id"] = "wrong"
    elif mutation == "description":
        row["normalized_description"] = "car"
    else:
        row["rank_key"] = {**row["rank_key"], "normalized_description": "car"}
    shard = analyzer.TreatmentShard(
        image_id="7511",
        directory=tmp_path,
        receipt={},
        scores=[row],
        proposals=[],
        x1=[],
    )
    with pytest.raises(analyzer.AnalysisContractError, match=message):
        analyzer.validate_treatment_score_row_joins(plan, shard)


def _overlay_for_shards(group_ids: list[str]) -> dict[str, Any]:
    return {
        "arm_id": prepare.ARM_ID,
        "baseline_arm_id": prepare.BASELINE_ARM_ID,
        "max_pixels": prepare.TREATMENT_MAX_PIXELS,
        "overlay_content_sha256": "overlay-digest",
        "base": {"plan_receipt_content_sha256": "base-digest"},
        "images": {
            "7511": {
                "current": {
                    "merged_visual_tokens": 1000,
                    "prompt_token_ids_sha256": "baseline-prompt",
                },
                "treatment": {
                    "image_grid_thw": [1, 40, 60],
                    "merged_visual_tokens": 1200,
                    "executed_media_sha256": "media-7511",
                    "width": 960,
                    "height": 640,
                    "prompt_token_ids_sha256": "prompt-7511",
                    "prompt_token_count": 1400,
                },
            }
        },
        "query_group_selection": {"query_group_ids_by_image": {"7511": group_ids}},
    }


def _stamp(**changes: Any) -> dict[str, Any]:
    stamp = {
        "intervention_unit_id": prepare.INTERVENTION_UNIT_ID,
        "arm_id": prepare.ARM_ID,
        "baseline_arm_id": prepare.BASELINE_ARM_ID,
        "overlay_content_sha256": "overlay-digest",
        "base_plan_receipt_content_sha256": "base-digest",
        "max_pixels": prepare.TREATMENT_MAX_PIXELS,
        "image_grid_thw": [1, 40, 60],
        "merged_visual_tokens": 1200,
        "baseline_merged_visual_tokens": 1000,
        "executed_media_sha256": "media-7511",
        "media_width": 960,
        "media_height": 640,
        "prompt_token_ids_sha256": "prompt-7511",
        "prompt_token_count": 1400,
        "baseline_prompt_token_ids_sha256": "baseline-prompt",
        "cross_arm_raw_logprob_comparison": "forbidden",
    }
    stamp.update(changes)
    return stamp


def test_complete_matching_shards_load(tmp_path: Path):
    groups = ["7511:boundary-000|person", "7511:boundary-001|person"]
    _write_shard(tmp_path, "7511", stamp=_stamp(), group_ids=groups)
    shards = analyzer.load_treatment_shards(
        tmp_path,
        _overlay_for_shards(groups),
        validate_capture_identity=False,
    )
    assert set(shards) == {"7511"}
    assert len(shards["7511"].scores) == 2
    completeness = shards["7511"].receipt[analyzer.COMPLETENESS_KEY]
    assert completeness["status"] == analyzer.COMPLETENESS_COMPLETE


def test_a_shard_predating_the_completeness_contract_fails_closed(tmp_path: Path):
    """An old capture is not assumed complete just because its rows line up.

    A shard written before ``intervention_completeness`` existed carries no
    coverage claim at all, so it cannot be admitted into a conclusion-bearing
    analysis even when its executed set happens to match the overlay.
    """

    groups = ["7511:boundary-000|person", "7511:boundary-001|person"]
    _write_shard(tmp_path, "7511", stamp=_stamp(), group_ids=groups, completeness=None)
    with pytest.raises(
        analyzer.AnalysisContractError, match="carries no 'intervention_completeness'"
    ):
        analyzer.load_treatment_shards(tmp_path, _overlay_for_shards(groups))


def test_a_subset_smoke_shard_is_refused_for_analysis(tmp_path: Path):
    groups = ["7511:boundary-000|person", "7511:boundary-001|person"]
    _write_shard(
        tmp_path, "7511", stamp=_stamp(), group_ids=groups, scored=groups[:1]
    )
    with pytest.raises(analyzer.AnalysisContractError, match="requires 'complete_frozen"):
        analyzer.load_treatment_shards(tmp_path, _overlay_for_shards(groups))


def test_the_base_plan_capture_completeness_field_is_never_the_authority(tmp_path: Path):
    """A base-relative ``subset_smoke`` must not block a complete capture.

    This is the seam that motivated the block: ``census.run_shard`` stamps
    ``capture_completeness = "subset_smoke"`` for every intervention shard,
    because the overlay selection is a subset of the base plan.  The analyzer
    must read the intervention-relative field and ignore the base one.
    """

    groups = ["7511:boundary-000|person", "7511:boundary-001|person"]
    _write_shard(tmp_path, "7511", stamp=_stamp(), group_ids=groups)
    receipt_path = tmp_path / "7511" / "shard-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["capture_completeness"] = "subset_smoke"
    receipt["subset_capture"] = {"is_subset": True, "usable_as_complete_shard_evidence": False}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    shards = analyzer.load_treatment_shards(
        tmp_path,
        _overlay_for_shards(groups),
        validate_capture_identity=False,
    )
    assert set(shards) == {"7511"}
    assert shards["7511"].receipt["capture_completeness"] == "subset_smoke"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"status": "something_else"}, "requires 'complete_frozen"),
        ({"is_complete_frozen_overlay_selection": False}, "internally inconsistent"),
        ({"missing_query_group_ids": ["7511:boundary-001|person"]}, "lists 1 missing"),
        ({"extra_query_group_ids": ["7511:boundary-009|person"]}, "lists 1 extra"),
        ({"missing_query_group_ids": "not-a-list"}, "is not a list"),
        ({"frozen_expected_query_group_count": 9}, "but this overlay freezes 2"),
        ({"executed_query_group_count": 9}, "but its score rows cover 2"),
    ],
)
def test_a_false_or_inconsistent_completeness_block_fails_closed(
    tmp_path: Path, mutation: dict[str, Any], message: str
):
    groups = ["7511:boundary-000|person", "7511:boundary-001|person"]
    honest = {
        "relative_to": "overlay_frozen_selection_for_this_image",
        "status": analyzer.COMPLETENESS_COMPLETE,
        "is_complete_frozen_overlay_selection": True,
        "frozen_expected_query_group_count": 2,
        "executed_query_group_count": 2,
        "missing_query_group_ids": [],
        "extra_query_group_ids": [],
        "derived_from": "query_group_ids_carrying_score_rows",
    }
    _write_shard(
        tmp_path,
        "7511",
        stamp=_stamp(),
        group_ids=groups,
        completeness={**honest, **mutation},
    )
    with pytest.raises(analyzer.AnalysisContractError, match=message):
        analyzer.load_treatment_shards(tmp_path, _overlay_for_shards(groups))


@pytest.mark.parametrize(
    ("stamp_changes", "message"),
    [
        ({"arm_id": "other-arm"}, "arms are never pooled"),
        ({"overlay_content_sha256": "other"}, "different sealed overlay"),
        ({"base_plan_receipt_content_sha256": "other"}, "different predecessor plan"),
        ({"intervention_unit_id": "other-unit"}, "another intervention unit"),
    ],
)
def test_foreign_identity_shards_fail_closed(
    tmp_path: Path, stamp_changes: dict[str, Any], message: str
):
    groups = ["7511:boundary-000|person"]
    _write_shard(tmp_path, "7511", stamp=_stamp(**stamp_changes), group_ids=groups)
    with pytest.raises(analyzer.AnalysisContractError, match=message):
        analyzer.load_treatment_shards(tmp_path, _overlay_for_shards(groups))


@pytest.mark.parametrize(
    "stamp_changes",
    [
        {"executed_media_sha256": "other-media"},
        {"prompt_token_ids_sha256": "other-prompt"},
        {"prompt_token_count": 999},
    ],
)
def test_per_image_media_or_prompt_stamp_drift_fails_closed(
    tmp_path: Path, stamp_changes: dict[str, Any]
):
    groups = ["7511:boundary-000|person"]
    _write_shard(tmp_path, "7511", stamp=_stamp(**stamp_changes), group_ids=groups)
    _update_receipt(
        tmp_path / "7511" / "shard-receipt.json", **_full_capture_identity("7511")
    )
    with pytest.raises(analyzer.AnalysisContractError, match="sealed overlay"):
        analyzer.load_treatment_shards(tmp_path, _overlay_for_shards(groups))


def _two_image_capture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    groups = ["7511:boundary-000|person"]
    overlay = _overlay_for_shards(groups)
    overlay["base"]["infer_config"] = "/frozen/infer.yaml"
    second = json.loads(json.dumps(overlay["images"]["7511"]))
    second["treatment"].update(
        {
            "executed_media_sha256": "media-7512",
            "prompt_token_ids_sha256": "prompt-7512",
        }
    )
    overlay["images"]["7512"] = second
    overlay["query_group_selection"]["query_group_ids_by_image"]["7512"] = [
        "7512:boundary-000|person"
    ]
    _write_shard(tmp_path, "7511", stamp=_stamp(), group_ids=groups)
    _write_shard(
        tmp_path,
        "7512",
        stamp=_stamp(
            executed_media_sha256="media-7512",
            prompt_token_ids_sha256="prompt-7512",
        ),
        group_ids=["7512:boundary-000|person"],
    )
    return overlay, {
        image_id: tmp_path / image_id / "shard-receipt.json"
        for image_id in ("7511", "7512")
    }


def _update_receipt(path: Path, **updates: Any) -> None:
    receipt = json.loads(path.read_text())
    receipt.update(updates)
    path.write_text(json.dumps(receipt), encoding="utf-8")


def _backend_identity(image_id: str, *, adapter_path: str = "adapter-a") -> dict[str, Any]:
    suffix = "7511" if image_id == "7511" else "7512"
    return {
        "backend": "hf",
        "infer_config": "/frozen/infer.yaml",
        "model_identity": {
            "family": "base-plus-adapter-plus-delta",
            "adapter": {"adapter_path": adapter_path},
            "embedding_delta": {"identity": {"delta_path": "embedding-a"}},
        },
        "tokenizer_identity": {"tokenizer_vocab_size": 152670},
        "adapter_identity": None,
        "repetition_penalty_stratum": 1.0,
        "session_scope": "image_shard",
        "is_real_model": True,
        "usable_as_evidence": True,
        "uses_model_generate": False,
        "executed_media_sha256": f"media-{suffix}",
        "executed_prompt_token_count": 1400,
        "image_grid_thw": [1, 40, 60],
    }


def _full_capture_identity(image_id: str) -> dict[str, Any]:
    return {
        "intervention_schema_version": successor_module.RECEIPT_SCHEMA_VERSION,
        "intervention_source_sha256": successor_module.EXECUTED_SOURCE_SHA256,
        "code": {
            "executed_source_sha256": analyzer.census.EXECUTED_SOURCE_SHA256,
            "planner_source_sha256": planner.sha256_file(Path(planner.__file__).resolve()),
            "runtime_seam_source_sha256": planner.sha256_file(
                REPO_ROOT / "scripts/research/score_sorted_owner_basin_landscape.py"
            ),
        },
        "plan": {
            "plan_schema_version": "synthetic-plan.v1",
            "receipt_content_sha256": "base-digest",
            "capture_rules_sha256": "rules-digest",
        },
        "backend_identity": _backend_identity(image_id),
        "granularity": {
            "candidate_batch_size": 16,
            "bulk_scoring_path": "admitted_kv_cache_batched_candidate_lanes",
            "candidate_batch_scope": "one_exact_query_group_and_admitted_prefix_only",
        },
        "numerics": {
            "matmul_precision": {"float32_matmul_precision": "highest"},
            "batched_path_max_abs_diff_bound": 0.0001,
            "batched_path_parity": {
                "admitted": True,
                "all_argmax_parity": True,
                "max_abs_diff_observed": 0.00001,
            },
        },
        "runtime_invariants": {"uses_model_generate": False},
        "checks": {"all_finite": True},
        "admission": {"all_admitted": True},
        "counts": {
            "query_group_count": 1,
            "localization_score_rows": 1,
            "proposal_surface_rows": 0,
            "x1_diagnostic_rows": 0,
            "free_decode_sidecar_rows": 0,
        },
        "phase_order": {
            "behavior_sidecars_captured": False,
            "behavior_sidecars_intentionally_disabled": True,
        },
        "intervention_selection": {
            "selection_is_frozen_in_the_overlay": True,
            "selection_uses_treatment_scores": False,
        },
    }


def test_mixed_wrapper_source_hash_is_rejected(tmp_path: Path):
    overlay, receipts = _two_image_capture(tmp_path)
    _update_receipt(receipts["7511"], **_full_capture_identity("7511"))
    second = _full_capture_identity("7512")
    second["intervention_source_sha256"] = "wrapper-b"
    _update_receipt(receipts["7512"], **second)
    with pytest.raises(analyzer.AnalysisContractError, match="intervention_source_sha256"):
        analyzer.load_treatment_shards(tmp_path, overlay)


def test_complete_real_hf_shaped_identity_loads_and_is_frozen(tmp_path: Path):
    overlay, receipts = _two_image_capture(tmp_path)
    for image_id, path in receipts.items():
        _update_receipt(path, **_full_capture_identity(image_id))
    shards = analyzer.load_treatment_shards(tmp_path, overlay)
    identity = analyzer.capture_identity_summary(shards, overlay)
    assert identity["status"] == "complete_uniform_capture_identity"
    assert identity["image_count"] == 2
    assert identity["common"]["intervention_source_sha256"] == (
        successor_module.EXECUTED_SOURCE_SHA256
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("backend", "fake"),
        ("repetition_penalty_stratum", 0.9),
        ("repetition_penalty_stratum", True),
        ("is_real_model", "true"),
        ("usable_as_evidence", 1),
        ("uses_model_generate", 0),
    ],
)
def test_wrong_production_backend_identity_values_fail_closed(
    tmp_path: Path, field: str, wrong_value: Any
):
    overlay, receipts = _two_image_capture(tmp_path)
    first = _full_capture_identity("7511")
    first["backend_identity"][field] = wrong_value
    _update_receipt(receipts["7511"], **first)
    _update_receipt(receipts["7512"], **_full_capture_identity("7512"))
    with pytest.raises(analyzer.AnalysisContractError, match=field):
        analyzer.load_treatment_shards(tmp_path, overlay)


@pytest.mark.parametrize("foreign_adapter", ["adapter-b", "adapter-c"])
def test_mixed_model_adapter_embedding_identity_is_rejected(
    tmp_path: Path, foreign_adapter: str
):
    overlay, receipts = _two_image_capture(tmp_path)
    _update_receipt(receipts["7511"], **_full_capture_identity("7511"))
    second = _full_capture_identity("7512")
    second["backend_identity"] = _backend_identity(
        "7512", adapter_path=foreign_adapter
    )
    _update_receipt(receipts["7512"], **second)
    with pytest.raises(analyzer.AnalysisContractError, match="model_identity"):
        analyzer.load_treatment_shards(tmp_path, overlay)


@pytest.mark.parametrize("missing_field", ["intervention_source_sha256", "backend_identity"])
def test_all_shards_missing_required_identity_fails_default_validation(
    tmp_path: Path, missing_field: str
):
    overlay, receipts = _two_image_capture(tmp_path)
    for image_id, path in receipts.items():
        identity = _full_capture_identity(image_id)
        identity.pop(missing_field)
        _update_receipt(path, **identity)
    with pytest.raises(analyzer.AnalysisContractError, match=missing_field):
        analyzer.load_treatment_shards(tmp_path, overlay)


def test_one_shard_missing_required_identity_fails_default_validation(tmp_path: Path):
    overlay, receipts = _two_image_capture(tmp_path)
    _update_receipt(receipts["7511"], **_full_capture_identity("7511"))
    second = _full_capture_identity("7512")
    second.pop("intervention_source_sha256")
    _update_receipt(receipts["7512"], **second)
    with pytest.raises(analyzer.AnalysisContractError, match="intervention_source_sha256"):
        analyzer.load_treatment_shards(tmp_path, overlay)


def test_backend_infer_config_must_match_sealed_overlay(tmp_path: Path):
    overlay, receipts = _two_image_capture(tmp_path)
    overlay["base"]["infer_config"] = "/frozen/infer.yaml"
    _update_receipt(receipts["7511"], **_full_capture_identity("7511"))
    second = _full_capture_identity("7512")
    second["backend_identity"]["infer_config"] = "/different/infer.yaml"
    _update_receipt(receipts["7512"], **second)
    with pytest.raises(analyzer.AnalysisContractError, match="sealed overlay base infer"):
        analyzer.load_treatment_shards(tmp_path, overlay)


def test_incomplete_or_over_broad_shards_fail_closed(tmp_path: Path):
    """The independent exact-set checks are unchanged and still decisive.

    Each shard here *claims* a consistent complete block, so the declared
    coverage passes; only the direct comparison of executed versus frozen IDs
    can catch it.  Adding the declared-completeness gate must never let these
    weaken into a self-report.
    """

    groups = ["7511:boundary-000|person", "7511:boundary-001|person"]
    lying_complete = {
        "relative_to": "overlay_frozen_selection_for_this_image",
        "status": analyzer.COMPLETENESS_COMPLETE,
        "is_complete_frozen_overlay_selection": True,
        "frozen_expected_query_group_count": 2,
        "executed_query_group_count": 1,
        "missing_query_group_ids": [],
        "extra_query_group_ids": [],
        "derived_from": "query_group_ids_carrying_score_rows",
    }
    _write_shard(
        tmp_path,
        "7511",
        stamp=_stamp(),
        group_ids=groups,
        scored=groups[:1],
        completeness=lying_complete,
    )
    with pytest.raises(analyzer.AnalysisContractError, match="incomplete"):
        analyzer.load_treatment_shards(tmp_path, _overlay_for_shards(groups))

    root = tmp_path / "wide"
    _write_shard(
        root,
        "7511",
        stamp=_stamp(),
        group_ids=groups,
        scored=[*groups, "7511:boundary-002|person"],
        completeness={**lying_complete, "executed_query_group_count": 3},
    )
    with pytest.raises(analyzer.AnalysisContractError, match="outside the sealed selection"):
        analyzer.load_treatment_shards(root, _overlay_for_shards(groups))


def test_missing_shard_and_non_finite_scores_fail_closed(tmp_path: Path):
    groups = ["7511:boundary-000|person"]
    with pytest.raises(analyzer.AnalysisContractError, match="is missing under"):
        analyzer.load_treatment_shards(tmp_path, _overlay_for_shards(groups))

    root = tmp_path / "nan"
    _write_shard(root, "7511", stamp=_stamp(), group_ids=groups, logprob=float("-inf"))
    with pytest.raises(analyzer.AnalysisContractError, match="non-finite"):
        analyzer.load_treatment_shards(root, _overlay_for_shards(groups))


# ---------------------------------------------------------------------------
# Integration: a real treatment capture must survive the predecessor merge's
# admission path against the overlaid plan
# ---------------------------------------------------------------------------


def test_treatment_shard_is_admitted_against_the_overlaid_plan(tmp_path: Path):
    """The overlay reseal must be exactly what the merge re-derives.

    This is the seam most likely to rot: the scorer writes prefix digests and
    admission IDs derived from the treatment prompt, and the merge independently
    re-derives them from the overlaid plan.  If those two ever disagree, every
    treatment row becomes unjoinable -- so it is proved on a real capture rather
    than asserted.
    """

    from scripts.research import score_sorted_owner_accessibility_census_shard as census
    from scripts.research import (
        score_sorted_full_canvas_token_budget_intervention_shard as successor,
    )
    from test_score_sorted_full_canvas_token_budget_intervention_shard import (
        make_overlay,
        write_overlay,
    )
    from test_score_sorted_owner_accessibility_census_shard import (
        IMAGE_ID,
        build_plan_rows,
        seal_plan,
    )

    plan_dir = seal_plan(tmp_path / "plan", build_plan_rows())
    overlay = make_overlay(plan_dir)
    bundle = successor.load_intervention_plan(
        plan_dir, write_overlay(tmp_path / "overlay", overlay)
    )
    shard_root = tmp_path / "shards"
    successor.run_intervention_shard(
        bundle,
        image_id=IMAGE_ID,
        backend=census.FakeCensusBackend(),
        output_dir=shard_root / IMAGE_ID,
    )

    # This integration fixture intentionally uses the fake backend to exercise
    # only the overlaid-prefix admission seam, not conclusion-bearing identity.
    shards = analyzer.load_treatment_shards(
        shard_root, bundle.overlay, validate_capture_identity=False
    )
    assert set(shards) == {IMAGE_ID}

    merged_plan = merge.PlanBundle(
        plan_dir=plan_dir,
        receipt=dict(bundle.plan.receipt),
        capture_rules=dict(bundle.plan.capture_rules),
        images=dict(bundle.plan.images),
        owners=dict(bundle.plan.owners),
        categories=dict(bundle.plan.categories),
        contexts=dict(bundle.plan.contexts),
        candidates=dict(bundle.plan.candidates),
        query_groups=dict(bundle.plan.query_groups),
        native_sidecars=list(bundle.plan.native_sidecars),
        shards=dict(bundle.plan.shards),
    )
    shard = shards[IMAGE_ID]
    artifacts = merge.ShardArtifacts(
        image_id=IMAGE_ID,
        split="discovery",
        status="captured",
        directory=shard.directory,
        receipt=shard.receipt,
        scores=shard.scores,
        proposals=shard.proposals,
        free_decodes=[],
        file_digests={},
    )
    admissions = merge.build_admission_index(merged_plan, artifacts)
    events = merge.admit_score_rows(merged_plan, artifacts, admissions)
    assert events
    surfaces = merge.admit_proposal_rows(merged_plan, artifacts, admissions)
    assert set(surfaces) == {row["context_id"] for row in bundle.plan.contexts.values()}

    table = merge.build_event_table(merged_plan, events, sidecars={})
    assert len(table) == len(events)
    for row in table:
        assert math.isfinite(float(row["complete_box_logprob_sum"]))
        # Every admitted row carries the *treatment* prefix identity.
        assert row["identity"]["query_prefix_sha256"] == (
            merged_plan.query_groups[row["query_group_id"]]["query_prefix_sha256"]
        )

    # The same rows are refused against the un-overlaid plan: an arm swap can
    # never be laundered through the merge.
    baseline = census.load_plan(plan_dir)
    stale = merge.PlanBundle(
        plan_dir=plan_dir,
        receipt=dict(baseline.receipt),
        capture_rules=dict(baseline.capture_rules),
        images=dict(baseline.images),
        owners=dict(baseline.owners),
        categories=dict(baseline.categories),
        contexts=dict(baseline.contexts),
        candidates=dict(baseline.candidates),
        query_groups=dict(baseline.query_groups),
        native_sidecars=list(baseline.native_sidecars),
        shards=dict(baseline.shards),
    )
    with pytest.raises(merge.MergeContractError):
        merge.admit_score_rows(
            stale, artifacts, merge.build_admission_index(stale, artifacts)
        )
