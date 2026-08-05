"""Focused tests for the primary branch analysis of the sorted crossing-boundary
owner release/realization unit.

The synthetic twelve-image capture family, its sealed plan and its smoke
admission are built by the merge module's test fixtures rather than duplicated
here: the analysis contract is defined over *merged* evidence, so the tests read
exactly what the merge publishes.  Every ladder in that fixture is serialized by
the scorer's own ``_ladder_payload``, so a capture-schema drift fails these
tests instead of silently passing.

The frozen denominators (26 U-bound primary owners, 12 matched-E / 14
unmatched-E, 24 exact same-context U&L, 14 timing controls, 12 TP replay
controls) are the contract under test and are never shrunk or monkeypatched.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.research import analyze_sorted_crossing_boundary_owner_release as analyze
from scripts.research import merge_sorted_crossing_boundary_owner_release as merge
from scripts.research import score_sorted_crossing_boundary_owner_release as scorer


def _load_merge_fixtures():
    """Import the merge test module by path so its builders are reused, not copied.

    ``tests/research`` is not a package, so a plain ``from tests.research...``
    import is not available; the module is registered in ``sys.modules`` before
    execution because its dataclasses need to resolve their own module.
    """

    name = "crossing_boundary_merge_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("test_merge_sorted_crossing_boundary_owner_release.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fx = _load_merge_fixtures()
OwnerSpec = fx.OwnerSpec
OTHER_OWNER = fx.OTHER_OWNER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def analyze_specs(tmp_path: Path, specs, *, name: str = "run") -> dict[str, Any]:
    """Build a full twelve-shard capture from 26 specs, merge it, and analyze it."""

    root = tmp_path / name
    capture = fx.build_capture(root, primary_specs=specs)
    capture.run_merge(root / "merged")
    return analyze.run_analysis(root / "merged")


def rows_by_owner(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["gt_owner_id"]: row for row in result["owner_rows"]}


def primary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in result["owner_rows"]
        if row["row_kind"] == "crossing_boundary_primary_owner_row"
    ]


def specs_with(intents: list[str]) -> list[OwnerSpec]:
    """26 specs from a list of intents, split 12 matched-E / 14 unmatched-E."""

    assert len(intents) == scorer.PRIMARY_OWNER_COUNT_U
    return [
        OwnerSpec(
            intent=intent,
            stratum=(
                merge.MATCHED_E_STRATUM
                if index < scorer.MATCHED_E_OWNER_COUNT
                else merge.UNMATCHED_E_STRATUM
            ),
        )
        for index, intent in enumerate(intents)
    ]


@pytest.fixture()
def routable(tmp_path: Path) -> dict[str, Any]:
    return analyze_specs(tmp_path, fx.routable_specs(), name="routable")


# ---------------------------------------------------------------------------
# Denominators, gate, routing
# ---------------------------------------------------------------------------


def test_frozen_denominators_are_reported_and_re_proven(routable: dict[str, Any]) -> None:
    report = routable["report"]
    primary = report["primary_cohort"]
    assert primary["denominator"] == 26
    assert primary["matched_e_count"] == 12
    assert primary["unmatched_e_count"] == 14
    assert primary["exact_same_context_u_and_l_count"] == 24
    assert primary["l_bound_crossing_count"] == 25
    assert primary["same_description_count"] + primary["different_description_count"] == 26
    assert len(primary_rows(routable)) == 26


def test_routing_denominator_is_interpretable_owners_not_all_twenty_six(
    routable: dict[str, Any],
) -> None:
    report = routable["report"]
    routing = report["routing"]
    evaluated = routing["evaluated"]

    assert routing["primary_cohort_denominator"] == 26
    assert routing["routing_denominator"] == "deterministic_interpretable_owners"
    assert routing["routing_denominator_count"] == 24
    assert evaluated["interpretable_count"] == 24
    assert evaluated["leading_branch_count"] == 17

    # 17/24 clears two thirds; 17/26 would not.  Routing therefore proves the
    # denominator is the interpretable cohort, not the full primary cohort.
    assert evaluated["leading_branch_count"] * 3 >= evaluated["interpretable_count"] * 2
    assert evaluated["leading_branch_count"] * 3 < 26 * 2
    assert evaluated["status"] == "routed"
    assert report["decision"] == analyze.DECISION_ROUTE


def test_interpretability_gate_is_reported_with_all_three_minimums(
    routable: dict[str, Any],
) -> None:
    gate = routable["report"]["interpretability_gate"]
    assert gate["minimum_interpretable"] == 20
    assert gate["minimum_matched_e"] == 6
    assert gate["minimum_unmatched_e"] == 7
    assert gate["interpretable_count"] == 24
    assert gate["matched_e_interpretable_count"] >= 6
    assert gate["unmatched_e_interpretable_count"] >= 7
    assert gate["passed"] is True


def test_gate_failure_blocks_routing_entirely(tmp_path: Path) -> None:
    intents = (
        ["displaced_likelihood"] * 10
        + ["release_lost"] * 6
        + ["realization_fail"] * 3
        + ["ambiguous_tie"] * 7
    )
    result = analyze_specs(tmp_path, specs_with(intents), name="gate-fail")
    report = result["report"]
    gate = report["interpretability_gate"]

    assert gate["interpretable_count"] == 19
    assert gate["passed"] is False
    assert report["routing"]["evaluated"] is None
    assert report["routing"]["routing_denominator_count"] == 19
    assert report["decision"] == analyze.DECISION_GATE_FAILED
    # The primary denominator is untouched by the gate outcome.
    assert report["primary_cohort"]["denominator"] == 26


def test_a_split_branch_share_closes_the_route(tmp_path: Path) -> None:
    intents = (
        ["displaced_likelihood"] * 12 + ["release_lost"] * 12 + ["ambiguous_tie"] * 2
    )
    report = analyze_specs(tmp_path, specs_with(intents), name="split")["report"]
    evaluated = report["routing"]["evaluated"]

    assert evaluated["interpretable_count"] == 24
    assert evaluated["leading_branch_count"] == 12
    assert evaluated["status"] == "split"
    assert evaluated["routed_branch"] is None
    assert report["decision"] == analyze.DECISION_CLOSE


def test_decoding_contradicted_exclusion_flips_a_route_to_split(tmp_path: Path) -> None:
    intents = (
        ["displaced_likelihood"] * 13
        + ["displaced_decoding_contradicted"] * 3
        + ["release_lost"] * 8
        + ["ambiguous_tie"] * 2
    )
    report = analyze_specs(tmp_path, specs_with(intents), name="contradicted")["report"]
    evaluated = report["routing"]["evaluated"]
    sensitivity = evaluated["decoding_contradicted_sensitivity"]

    assert report["decoding_contradicted_sensitivity"]["count"] == 3
    # 16/24 reaches exactly two thirds ...
    assert evaluated["leading_branch_count"] == 16
    assert evaluated["interpretable_count"] == 24
    assert evaluated["leading_branch_count"] * 3 >= evaluated["interpretable_count"] * 2
    # ... but 13/21 does not, so the frozen split rule fires.
    assert sensitivity["retained_interpretable_count"] == 21
    assert sensitivity["retained_branch_counts"]["displaced"] == 13
    assert sensitivity["retained_reaches_two_thirds"] is False
    assert evaluated["status"] == "split"
    assert report["decision"] == analyze.DECISION_CLOSE


def test_decoding_contradicted_cells_are_reported_separately(routable: dict[str, Any]) -> None:
    report = routable["report"]
    contradicted = report["decoding_contradicted_sensitivity"]
    assert contradicted["count"] == 1
    owner_id = contradicted["owner_ids"][0]

    row = rows_by_owner(routable)[owner_id]
    assert row["displacement"]["likelihood_displaced"] is True
    assert row["displacement"]["decoding_contradicted"] is True
    assert row["at_p_plus_e"]["greedy_status"] == scorer.GREEDY_TARGET_MATCH
    assert row["primary_branch"] == "displaced"
    # It is still a routed-cohort member; the sensitivity is an exclusion test.
    assert row["interpretable"] is True


# ---------------------------------------------------------------------------
# Branch semantics: precedence, reachability, ambiguity
# ---------------------------------------------------------------------------


def test_every_frozen_branch_is_reachable(routable: dict[str, Any]) -> None:
    counts = routable["report"]["branches"]["branch_counts_all_primary_owners"]
    assert set(counts) == {"displaced", "release_lost", "realization_fail", "ambiguous"}
    assert counts == {"displaced": 17, "release_lost": 6, "realization_fail": 1, "ambiguous": 2}


def test_displacement_is_evaluated_before_release_loss(routable: dict[str, Any]) -> None:
    row = next(
        row
        for row in primary_rows(routable)
        if row["displacement"] and row["displacement"]["likelihood_displaced"]
        and not row["displacement"]["decoding_contradicted"]
    )
    # The same owner satisfies every release_lost predicate as well ...
    assert row["at_p_plus_e"]["release_observable"] is True
    assert row["at_p_plus_e"]["release_target_minus_native_margin"] < 0
    assert row["at_p_plus_e"]["support_disposition_u"] == scorer.SUPPORT_SUPPORTED
    # ... and branch 1 still wins, exactly as unit.md orders it.
    assert row["primary_branch"] == "displaced"


def test_greedy_displacement_alone_reaches_the_displaced_branch(tmp_path: Path) -> None:
    intents = (
        ["displaced_greedy"] * 17 + ["release_lost"] * 6 + ["realization_fail"] * 3
    )
    result = analyze_specs(tmp_path, specs_with(intents), name="greedy-displaced")
    row = next(
        row for row in primary_rows(result) if row["primary_branch"] == "displaced"
    )
    assert row["displacement"]["likelihood_displaced"] is False
    assert row["displacement"]["greedy_displaced"] is True
    assert row["displacement"]["greedy_displaced_owner_id"] == OTHER_OWNER


def test_realization_fail_is_reachable_without_displacement(routable: dict[str, Any]) -> None:
    row = next(
        row for row in primary_rows(routable) if row["primary_branch"] == "realization_fail"
    )
    assert row["displacement"]["likelihood_displaced"] is False
    assert row["displacement"]["greedy_displaced"] is False
    assert row["at_p_plus_e"]["support_disposition_u"] == scorer.SUPPORT_UNSUPPORTED
    assert row["at_p_plus_e"]["greedy_status"] in scorer.GREEDY_TARGET_MISSED_STATUSES
    assert row["interpretable"] is True


def test_release_lost_requires_retained_calibrated_support(routable: dict[str, Any]) -> None:
    row = next(
        row for row in primary_rows(routable) if row["primary_branch"] == "release_lost"
    )
    assert row["at_p_plus_e"]["release_target_minus_native_margin"] < 0
    assert row["at_p_plus_e"]["support_disposition_u"] == scorer.SUPPORT_SUPPORTED
    assert row["interpretable"] is True


@pytest.mark.parametrize(
    ("intent", "reason"),
    [
        ("ambiguous_tie", "exact_tie_or_nonunique_owner_match"),
        ("ambiguous_nonunique", "exact_tie_or_nonunique_owner_match"),
        ("ambiguous_calibration", "branch_ambiguous"),
        ("ambiguous_missing", "missing_required_primary_score_field"),
    ],
)
def test_unknown_tie_nonunique_and_missing_route_to_ambiguous(
    tmp_path: Path, intent: str, reason: str
) -> None:
    intents = [intent] * 4 + ["displaced_likelihood"] * 22
    result = analyze_specs(tmp_path, specs_with(intents), name=f"amb-{intent}")
    ambiguous = [row for row in primary_rows(result) if row["primary_branch"] == "ambiguous"]
    assert len(ambiguous) == 4
    for row in ambiguous:
        assert row["interpretable"] is False
        assert reason in row["not_interpretable_reasons"]
    # An ambiguous owner never enters the routing denominator ...
    assert result["report"]["routing"]["routing_denominator_count"] == 22
    # ... and never leaves the 26-owner primary denominator either.
    assert result["report"]["primary_cohort"]["denominator"] == 26


def test_calibration_unavailable_never_claims_support(tmp_path: Path) -> None:
    intents = ["ambiguous_calibration"] * 4 + ["displaced_likelihood"] * 22
    result = analyze_specs(tmp_path, specs_with(intents), name="calib")
    row = next(
        row
        for row in primary_rows(result)
        if row["at_p_plus_e"]["support_disposition_u"]
        == scorer.SUPPORT_CALIBRATION_UNAVAILABLE
    )
    assert row["primary_branch"] == "ambiguous"
    assert row["paired_access"]["coordinate"]["access_at_p_plus_e"] is None
    assert row["paired_access"]["coordinate"]["tag"] == analyze.ACCESS_NOT_OBSERVABLE


def test_quarantined_owner_is_never_branched_but_stays_in_the_denominator(
    tmp_path: Path,
) -> None:
    specs = fx.routable_specs()
    specs[0] = replace(specs[0], quarantined=True)
    result = analyze_specs(tmp_path, specs, name="quarantined")
    report = result["report"]

    quarantined = [row for row in primary_rows(result) if row["quarantined"]]
    assert len(quarantined) == 1
    row = quarantined[0]
    assert row["primary_branch"] is None
    assert row["sensitivity_branch_l"] is None
    assert row["interpretable"] is False
    assert "replay_not_admitted" in row["not_interpretable_reasons"]

    assert report["primary_cohort"]["denominator"] == 26
    assert report["primary_cohort"]["quarantined_owner_ids"] == [row["gt_owner_id"]]
    assert report["branches"]["branch_counts_all_primary_owners"]["null"] == 1
    assert report["routing"]["routing_denominator_count"] == 23


# ---------------------------------------------------------------------------
# U primary vs L sensitivity
# ---------------------------------------------------------------------------


def test_l_bound_is_a_sensitivity_that_never_changes_the_u_branch(tmp_path: Path) -> None:
    specs = fx.routable_specs()
    # A release_lost owner whose L-bound support is absent: U keeps release_lost,
    # L falls through to ambiguous.
    release_lost_index = next(
        index for index, spec in enumerate(specs) if spec.intent == "release_lost"
    )
    specs[release_lost_index] = replace(specs[release_lost_index], intent="release_lost_l_absent")
    fx._INTENTS["release_lost_l_absent"] = {  # noqa: SLF001 - fixture-local intent
        **fx._INTENTS["release_lost"],
        "support_l": scorer.SUPPORT_UNSUPPORTED,
    }
    try:
        result = analyze_specs(tmp_path, specs, name="l-sensitivity")
    finally:
        fx._INTENTS.pop("release_lost_l_absent")  # noqa: SLF001

    row = next(
        row
        for row in primary_rows(result)
        if row["at_p_plus_e"]["support_disposition_l"] == scorer.SUPPORT_UNSUPPORTED
        and row["at_p_plus_e"]["support_disposition_u"] == scorer.SUPPORT_SUPPORTED
    )
    assert row["primary_branch"] == "release_lost"
    assert row["sensitivity_branch_l"] == "ambiguous"
    assert row["l_sensitivity_agrees_with_u"] is False

    sensitivity = result["report"]["l_bound_sensitivity"]
    assert sensitivity["bound"] == scorer.SUPPORT_BOUND_L
    assert row["gt_owner_id"] in sensitivity["disagreeing_owner_ids"]
    # The U-side routing is untouched by the L disagreement.
    assert result["report"]["routing"]["evaluated"]["routed_branch"] == "displaced"


def test_u_is_declared_the_primary_support_bound(routable: dict[str, Any]) -> None:
    for row in primary_rows(routable):
        assert row["at_p_plus_e"]["support_disposition_u"] in scorer.SUPPORT_DISPOSITIONS
        assert row["at_p_plus_e"]["support_disposition_l"] in scorer.SUPPORT_DISPOSITIONS
    definition = routable["report"]["operational_definitions"]["support_bound"]
    assert definition.startswith("U owns the primary support disposition")


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_controls_are_reported_separately_and_never_in_a_denominator(
    routable: dict[str, Any],
) -> None:
    report = routable["report"]
    controls = report["controls"]
    assert controls["in_primary_denominator"] is False
    assert controls["cohorts"][merge.TIMING_CONTROL_COHORT]["owner_count"] == 14
    assert controls["cohorts"][merge.TP_REPLAY_CONTROL_COHORT]["owner_count"] == 12

    control_rows = [
        row
        for row in routable["owner_rows"]
        if row["row_kind"] == "crossing_boundary_control_owner_row"
    ]
    assert len(control_rows) == 26
    assert all(row["primary_branch"] is None for row in control_rows)
    assert all(row["in_primary_denominator"] is False for row in control_rows)

    # 26 primary + 26 control owners exist, and only the primary 26 are counted.
    assert len(routable["owner_rows"]) == 52
    assert report["primary_cohort"]["denominator"] == 26
    assert report["routing"]["routing_denominator_count"] == 24
    control_ids = {row["gt_owner_id"] for row in control_rows}
    primary_ids = {row["gt_owner_id"] for row in primary_rows(routable)}
    assert control_ids.isdisjoint(primary_ids)


# ---------------------------------------------------------------------------
# Paired P -> P+E evidence
# ---------------------------------------------------------------------------


def test_paired_access_tags_cover_opened_retained_and_suppressed(tmp_path: Path) -> None:
    specs = fx.routable_specs()
    # coordinate opened: unsupported at P, supported at P+E.
    specs[0] = replace(specs[0], p_support_u=scorer.SUPPORT_UNSUPPORTED)
    # release opened: negative at P, positive at P+E (realization_fail's P+E margin).
    specs[24] = replace(specs[24], intent="realization_fail", p_release_margin=-1.0)
    # release retained: positive at both.
    specs[25] = replace(specs[25], intent="realization_fail", p_release_margin=1.0)
    result = analyze_specs(tmp_path, specs, name="access-tags")
    rows = rows_by_owner(result)

    opened_coordinate = rows["gt:10707:p0"]
    assert opened_coordinate["paired_access"]["coordinate"]["tag"] == analyze.ACCESS_OPENED
    assert opened_coordinate["paired_access"]["coordinate"]["access_at_p"] is False
    assert opened_coordinate["paired_access"]["coordinate"]["access_at_p_plus_e"] is True

    tags = result["report"]["paired_access_tags"]
    assert tags["natural_release"][analyze.ACCESS_SUPPRESSED] >= 1
    assert tags["natural_release"][analyze.ACCESS_OPENED] >= 1
    assert tags["natural_release"][analyze.ACCESS_RETAINED] >= 1
    assert tags["coordinate"][analyze.ACCESS_OPENED] >= 1
    assert tags["coordinate"][analyze.ACCESS_SUPPRESSED] >= 1
    assert tags["coordinate"][analyze.ACCESS_RETAINED] >= 1

    # ``retained`` never hides which side of the predicate it retained.
    for row in primary_rows(result):
        for ladder in ("natural_release", "coordinate"):
            paired = row["paired_access"][ladder]
            if paired["tag"] == analyze.ACCESS_RETAINED:
                assert paired["access_at_p"] == paired["access_at_p_plus_e"]
                assert paired["access_at_p"] is not None


def test_same_description_coordinate_readout_is_construction_determined(
    tmp_path: Path,
) -> None:
    specs = fx.routable_specs()
    specs[0] = replace(specs[0], same_description=True)
    result = analyze_specs(tmp_path, specs, name="same-desc")
    row = rows_by_owner(result)["gt:10707:p0"]

    assert row["same_description_as_e"] is True
    assert row["at_p"]["coordinate_readout_is_construction_determined"] is True
    assert (
        row["paired_access"]["coordinate"]["tag"]
        == analyze.ACCESS_CONSTRUCTION_DETERMINED_AT_P
    )
    assert row["paired_access"]["coordinate"]["access_at_p"] is None
    # Release is unobservable for a same-description owner, so its paired tag is
    # never invented either.
    assert row["paired_access"]["natural_release"]["tag"] == analyze.ACCESS_NOT_OBSERVABLE
    assert (
        row["paired_transitions"]["coordinate_transitions_are_construction_determined_at_p"]
        is True
    )
    # The raw P readout is still preserved for replay, tag or no tag.
    assert row["at_p"]["support_disposition_u"] == scorer.SUPPORT_SUPPORTED
    assert row["at_p"]["target_rank"] == 1
    assert row["paired_transitions"]["support_disposition_u"]["at_p"] == (
        scorer.SUPPORT_SUPPORTED
    )


def test_owner_rows_preserve_every_raw_paired_p_field(routable: dict[str, Any]) -> None:
    required = (
        "target_rank",
        "best_competitor_owner_id",
        "target_minus_competitor_margin",
        "family_rank_disposition",
        "greedy_status",
        "greedy_owner_match",
        "greedy_nonunique_match",
        "release_target_minus_native_margin",
        "release_argmax_follows_target",
        "release_observable",
        "support_disposition_u",
        "support_disposition_l",
    )
    for row in primary_rows(routable):
        for field_name in required:
            assert field_name in row["at_p"], field_name
            assert field_name in row["at_p_plus_e"], field_name

    row = rows_by_owner(routable)["gt:10707:p0"]
    assert row["at_p"]["target_rank"] == 1
    assert row["at_p"]["best_competitor_owner_id"] == OTHER_OWNER
    assert row["at_p"]["target_minus_competitor_margin"] == pytest.approx(1.5)
    assert row["at_p"]["greedy_status"] == scorer.GREEDY_TARGET_MATCH
    assert row["at_p"]["greedy_owner_match"] == "gt:10707:p0"
    assert row["at_p"]["release_target_minus_native_margin"] == pytest.approx(1.0)
    assert row["at_p"]["release_argmax_follows_target"] is True


def test_component_transitions_are_transparent_and_numeric(routable: dict[str, Any]) -> None:
    row = rows_by_owner(routable)["gt:10707:p0"]
    transitions = row["paired_transitions"]

    assert transitions["target_rank"]["at_p"] == 1
    assert transitions["target_rank"]["at_p_plus_e"] == 3
    assert transitions["target_rank"]["change"] == analyze.CHANGE_WORSENED
    assert transitions["target_rank"]["delta"] == pytest.approx(2.0)

    margin = transitions["target_minus_competitor_margin"]
    assert margin["sign_transition"] == "positive->negative"
    assert margin["delta"] == pytest.approx(-3.5)

    release = transitions["release_target_minus_native_margin"]
    assert release["sign_transition"] == "positive->negative"
    assert release["delta"] == pytest.approx(-2.0)
    assert release["argmax_follows_target_transition"] == "True->False"

    assert transitions["support_disposition_u"]["transition"] == "supported->supported"
    assert transitions["greedy_status"]["transition"] == "target_match->unmatched"
    assert transitions["best_competitor_owner_id"]["changed"] is False

    counts = routable["report"]["paired_component_transitions"]
    assert set(counts["all_primary_owners"]) == set(analyze.TRANSITION_COMPONENTS)
    assert sum(counts["all_primary_owners"]["target_rank"].values()) == 26
    assert sum(counts["interpretable_owners"]["target_rank"].values()) == 24
    assert "displaced" in counts["by_branch"]


def test_component_transitions_survive_an_undeterminable_boundary(tmp_path: Path) -> None:
    intents = ["ambiguous_missing"] * 4 + ["displaced_likelihood"] * 22
    result = analyze_specs(tmp_path, specs_with(intents), name="missing-transitions")
    row = next(row for row in primary_rows(result) if row["missing_required_fields"])
    transitions = row["paired_transitions"]

    assert row["at_p_plus_e"]["target_rank"] is None
    assert transitions["target_rank"]["change"] == analyze.CHANGE_NOT_DETERMINABLE
    assert transitions["target_rank"]["delta"] is None
    assert transitions["target_minus_competitor_margin"]["sign_at_p_plus_e"] == "null"
    assert transitions["target_minus_competitor_margin"]["delta"] is None


def test_pre_post_coherence_contradiction_closes_an_otherwise_routed_branch(
    tmp_path: Path,
) -> None:
    specs = fx.routable_specs()
    # One displaced owner whose coordinate ladder *opened* at the crossing: the
    # crossing row improved the very ladder the leading branch blames.
    specs[0] = replace(specs[0], p_support_u=scorer.SUPPORT_UNSUPPORTED)
    result = analyze_specs(tmp_path, specs, name="incoherent")
    report = result["report"]
    coherence = report["pre_post_coherence"]

    assert coherence["leading_branch"] == "displaced"
    assert coherence["mechanism_ladder"] == "coordinate"
    assert coherence["tag_counts"][analyze.ACCESS_OPENED] == 1
    assert coherence["status"] == analyze.COHERENCE_CONTRADICTED
    assert coherence["closes_route"] is True
    # Routing still says routed; the coherence clause is what closes it.
    assert report["routing"]["evaluated"]["status"] == "routed"
    assert report["decision"] == analyze.DECISION_CLOSE


def test_pre_post_coherence_reports_an_unchanged_cohort_without_deciding(
    routable: dict[str, Any],
) -> None:
    coherence = routable["report"]["pre_post_coherence"]
    assert coherence["status"] == analyze.COHERENCE_NO_PAIRED_CHANGE
    assert coherence["closes_route"] is False
    assert coherence["requires_adjudication"] is True
    assert routable["report"]["decision"] == analyze.DECISION_ROUTE
    # The coarse tag never stands alone: the raw component counts travel with it.
    assert coherence["component_transition_counts"]["target_rank"]
    assert coherence["definition"].startswith("an explicit operational definition")


def test_pre_post_coherence_records_a_real_suppression(tmp_path: Path) -> None:
    intents = (
        ["realization_fail"] * 17 + ["release_lost"] * 7 + ["ambiguous_tie"] * 2
    )
    specs = [
        replace(spec, p_support_u=scorer.SUPPORT_SUPPORTED)
        for spec in specs_with(intents)
    ]
    report = analyze_specs(tmp_path, specs, name="coherent")["report"]
    coherence = report["pre_post_coherence"]

    assert coherence["leading_branch"] == "realization_fail"
    assert coherence["mechanism_ladder"] == "coordinate"
    assert coherence["tag_counts"][analyze.ACCESS_SUPPRESSED] == 17
    assert coherence["status"] == analyze.COHERENCE_COHERENT_CHANGE
    assert coherence["closes_route"] is False
    assert report["decision"] == analyze.DECISION_ROUTE


# ---------------------------------------------------------------------------
# Contract guards, determinism, output shape
# ---------------------------------------------------------------------------


def test_secondary_compatibility_evidence_is_refused(routable: dict[str, Any]) -> None:
    record = {
        "gt_owner_id": "gt:10707:p0",
        "cohort": merge.PRIMARY_COHORT,
        "branch_assignment": merge.BRANCH_ASSIGNMENT_SENTINEL,
        "secondary_compatibility": {"p_plus_c_to_e": {"description_delta": -0.5}},
    }
    with pytest.raises(analyze.AnalysisContractError, match="secondary compatibility payload"):
        analyze.classify_owner(record)


def test_an_inherited_branch_assignment_is_refused() -> None:
    record = {
        "gt_owner_id": "gt:10707:p0",
        "cohort": merge.PRIMARY_COHORT,
        "branch_assignment": "displaced",
        "secondary_compatibility": merge.SECONDARY_COMPATIBILITY_SENTINEL,
    }
    with pytest.raises(analyze.AnalysisContractError, match="already carries a branch"):
        analyze.classify_owner(record)


def test_a_control_record_is_never_branched() -> None:
    record = {
        "gt_owner_id": "gt:10707:c0",
        "cohort": merge.TP_REPLAY_CONTROL_COHORT,
        "branch_assignment": merge.BRANCH_ASSIGNMENT_SENTINEL,
        "secondary_compatibility": merge.SECONDARY_COMPATIBILITY_SENTINEL,
    }
    with pytest.raises(analyze.AnalysisContractError, match="never classified"):
        analyze.classify_owner(record)


def test_tampered_merged_evidence_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    target = root / "merged" / merge.MERGED_OWNER_RECORDS_NAME
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(analyze.AnalysisContractError, match="does not match the digest"):
        analyze.run_analysis(root / "merged")


def test_an_edited_merge_receipt_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "edited"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    path = root / "merged" / merge.MERGE_RECEIPT_NAME
    receipt = json.loads(path.read_text())
    receipt["observed_cohorts"]["primary_owner_count"] = 24
    path.write_bytes(scorer.canonical_json_bytes(receipt) + b"\n")
    # The self-seal check is the merge module's own helper, reused so both
    # modules canonicalize identically; its contract error is what surfaces.
    with pytest.raises(merge.MergeContractError, match="does not reconstruct its own"):
        analyze.run_analysis(root / "merged")


def test_cohort_membership_is_not_score_dependent(tmp_path: Path) -> None:
    """Rescaling every score keeps the cohort, strata and denominators fixed."""

    baseline = analyze_specs(tmp_path, fx.routable_specs(), name="baseline")
    scaled_specs = [
        replace(spec, p_margin=(spec.p_margin or 0.0) * 1000.0, p_release_margin=(spec.p_release_margin or 0.0) * 1000.0)
        for spec in fx.routable_specs()
    ]
    scaled = analyze_specs(tmp_path, scaled_specs, name="scaled")

    for report in (baseline["report"], scaled["report"]):
        assert report["primary_cohort"]["denominator"] == 26
        assert report["primary_cohort"]["matched_e_count"] == 12
        assert report["primary_cohort"]["unmatched_e_count"] == 14
    assert (
        baseline["report"]["branches"]["branch_counts_all_primary_owners"]
        == scaled["report"]["branches"]["branch_counts_all_primary_owners"]
    )
    # Only the P-side raw evidence moved, so the paired transitions differ while
    # the cohort composition does not.
    baseline_row = rows_by_owner(baseline)["gt:10707:p0"]
    scaled_row = rows_by_owner(scaled)["gt:10707:p0"]
    assert (
        baseline_row["paired_transitions"]["target_minus_competitor_margin"]["delta"]
        != scaled_row["paired_transitions"]["target_minus_competitor_margin"]["delta"]
    )


def test_analysis_outputs_are_deterministic_and_self_sealed(tmp_path: Path) -> None:
    root = tmp_path / "deterministic"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")

    first = analyze.build_output_files(analyze.run_analysis(root / "merged"))
    second = analyze.build_output_files(analyze.run_analysis(root / "merged"))
    assert first == second
    assert sorted(first) == [
        analyze.OWNER_ROWS_NAME,
        analyze.RECEIPT_NAME,
        analyze.REPORT_JSON_NAME,
        analyze.REPORT_MD_NAME,
    ]

    receipt = json.loads(first[analyze.RECEIPT_NAME])
    merge.assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="analysis receipt"
    )
    assert receipt["unit_id"] == merge.UNIT_ID
    assert receipt["scorer_source_sha256"] == merge.FROZEN_SCORER_SOURCE_SHA256
    assert receipt["policy"]["secondary_compatibility_read"] is False
    assert receipt["policy"]["threshold_fitting"] is False
    assert receipt["primary_denominator"] == 26
    assert receipt["routing_denominator_count"] == 24
    for name, entry in receipt["output_file_digests"].items():
        assert scorer.sha256_bytes(first[name]) == entry["sha256"]

    published = merge.publish_merge(root / "analysis", first)
    assert published["published"] is True
    rerun = merge.publish_merge(root / "analysis", second)
    assert rerun["publish_mode"] == "no_op_identical_rerun"


def test_markdown_reports_both_denominators_and_the_decision(routable: dict[str, Any]) -> None:
    rendered = analyze.render_markdown(routable["report"])
    assert "primary cohort denominator: 26" in rendered
    assert "routing denominator (deterministic_interpretable_owners): 24" in rendered
    assert "2/3 of the deterministic interpretable owners" in rendered
    assert f"**Decision: `{analyze.DECISION_ROUTE}`**" in rendered
    assert "Raw P -> P+E component transitions" in rendered
    assert "Controls (never in the primary denominator)" in rendered


def test_cli_writes_the_analysis_family(tmp_path: Path) -> None:
    root = tmp_path / "cli"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    exit_code = analyze.main(
        [
            "--merged-dir",
            str(root / "merged"),
            "--output-dir",
            str(root / "analysis"),
        ]
    )
    assert exit_code == 0
    for name in (
        analyze.OWNER_ROWS_NAME,
        analyze.REPORT_JSON_NAME,
        analyze.REPORT_MD_NAME,
        analyze.RECEIPT_NAME,
    ):
        assert (root / "analysis" / name).is_file()
    report = json.loads((root / "analysis" / analyze.REPORT_JSON_NAME).read_text())
    assert report["decision"] == analyze.DECISION_ROUTE


# ---------------------------------------------------------------------------
# Conclusion fragility
#
# The pure slice helpers are exercised against owner rows shaped like the frozen
# run's own audited cells, so the expected counts and owner IDs in these tests
# are the audit's, not the fixture's convenience.  The end-to-end tests then
# prove the same slices are wired into report.json, report.md and the receipt.
# ---------------------------------------------------------------------------

#: The audited displaced cells of the frozen run: 10 carry both displacement
#: sub-tags, 5 carry likelihood only, 2 carry the coordinate-only greedy mirror
#: only, and none carries neither.
AUDIT_BOTH_SUB_TAGS = (
    "gt:13923:11",
    "gt:13923:14",
    "gt:14038:23",
    "gt:14439:3",
    "gt:1584:8",
    "gt:16228:11",
    "gt:16228:3",
    "gt:16228:44",
    "gt:5001:10",
    "gt:6040:13",
)
AUDIT_LIKELIHOOD_ONLY = (
    "gt:13348:7",
    "gt:14038:19",
    "gt:16228:38",
    "gt:4134:22",
    "gt:4134:32",
)
AUDIT_GREEDY_ONLY_MIRROR = ("gt:10707:16", "gt:16228:47")

#: The audited greedy displacers: (crossing owner, displacing owner, crossing E
#: owner or ``None`` for an unmatched E row).  No displacer equals E's owner.
AUDIT_GREEDY_DISPLACERS = (
    ("gt:10707:16", "gt:10707:17", None),
    ("gt:13923:11", "gt:13923:15", "gt:13923:12"),
    ("gt:13923:14", "gt:13923:16", None),
    ("gt:14038:23", "gt:14038:26", "gt:14038:25"),
    ("gt:14439:3", "gt:14439:5", "gt:14439:6"),
    ("gt:1584:8", "gt:1584:14", None),
    ("gt:16228:11", "gt:16228:18", None),
    ("gt:16228:3", "gt:16228:5", "gt:16228:2"),
    ("gt:16228:44", "gt:16228:48", None),
    ("gt:16228:47", "gt:16228:44", "gt:16228:48"),
    ("gt:5001:10", "gt:5001:12", None),
    ("gt:6040:13", "gt:6040:9", "gt:6040:14"),
)


def boundary_fields(**overrides: Any) -> dict[str, Any]:
    """One boundary's raw readout, in the exact shape ``classify_owner`` emits."""

    fields: dict[str, Any] = {
        "context_id": "10707:boundary-000",
        "boundary_label": "P",
        "native_action_kind": "native_row",
        "native_argmax_replay_admitted": True,
        "release_observable": True,
        "release_target_minus_native_margin": -1.0,
        "release_argmax_follows_target": False,
        "release_first_divergence_index": 1,
        "release_gate_margin_is_versus_stop": False,
        "target_rank": 1,
        "best_competitor_owner_id": OTHER_OWNER,
        "target_minus_competitor_margin": 1.0,
        "family_rank_disposition": scorer.RANK_TARGET_FIRST,
        "support_disposition_u": scorer.SUPPORT_SUPPORTED,
        "support_disposition_l": scorer.SUPPORT_SUPPORTED,
        "greedy_status": scorer.GREEDY_TARGET_MATCH,
        "greedy_owner_match": None,
        "greedy_nonunique_match": False,
        "coordinate_readout_is_construction_determined": False,
    }
    fields.update(overrides)
    return fields


def fragility_row(
    owner_id: str,
    *,
    branch: str = "displaced",
    likelihood_owner: str | None = None,
    greedy_owner: str | None = None,
    interpretable: bool = True,
    stratum: str = merge.UNMATCHED_E_STRATUM,
    description: str = "person",
    at_p: dict[str, Any] | None = None,
    at_p_plus_e: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One primary owner row carrying only what the fragility slices read."""

    p_fields = at_p or boundary_fields()
    ppe_fields = at_p_plus_e or boundary_fields(boundary_label="P_plus_E")
    return {
        "gt_owner_id": owner_id,
        "stratum": stratum,
        "normalized_description": description,
        "same_description_as_e": False,
        "primary_branch": branch,
        "interpretable": interpretable,
        "l_sensitivity_agrees_with_u": True,
        "at_p": p_fields,
        "at_p_plus_e": ppe_fields,
        "paired_transitions": analyze.build_paired_transitions(
            p_fields, ppe_fields, coordinate_is_construction_determined=False
        ),
        "displacement": {
            "likelihood_displaced": likelihood_owner is not None,
            "likelihood_displaced_owner_id": likelihood_owner,
            "greedy_displaced": greedy_owner is not None,
            "greedy_displaced_owner_id": greedy_owner,
            "decoding_contradicted": False,
        },
    }


def audit_shaped_rows() -> list[dict[str, Any]]:
    """The frozen run's 24 deterministic interpretable owners, cell for cell."""

    displacer_by_owner = {
        owner_id: displacer for owner_id, displacer, _ in AUDIT_GREEDY_DISPLACERS
    }
    rows = [
        fragility_row(
            owner_id,
            likelihood_owner=f"{owner_id}:likelihood",
            greedy_owner=displacer_by_owner[owner_id],
        )
        for owner_id in AUDIT_BOTH_SUB_TAGS
    ]
    rows += [
        fragility_row(
            owner_id,
            likelihood_owner=f"{owner_id}:likelihood",
            at_p_plus_e=boundary_fields(
                boundary_label="P_plus_E", greedy_status=scorer.GREEDY_UNMATCHED
            ),
        )
        for owner_id in AUDIT_LIKELIHOOD_ONLY
    ]
    rows += [
        fragility_row(owner_id, greedy_owner=displacer_by_owner[owner_id])
        for owner_id in AUDIT_GREEDY_ONLY_MIRROR
    ]
    # The seven non-displaced deterministic interpretable owners, so the routing
    # denominator is the audited 24 rather than the displaced cohort alone.
    rows += [
        fragility_row(f"gt:other:{index}", branch="release_lost")
        for index in range(6)
    ]
    rows.append(fragility_row("gt:other:6", branch="realization_fail"))
    return rows


def audit_plan_facts() -> dict[str, Any]:
    """A resolved plan-fact payload naming each audited crossing E owner."""

    return {
        "source": analyze.CROSSING_E_SOURCE_SEALED_PLAN,
        "reason": None,
        "plan_dir": "/frozen/plan",
        "cohort_registry_sha256": "0" * 64,
        "control_registry_sha256": "1" * 64,
        "by_owner": {
            owner_id: {
                "crossing_e_owner_id": crossing_e_owner,
                "crossing_e_strict_match_status": (
                    analyze.E_STRICT_MATCH_UNMATCHED
                    if crossing_e_owner is None
                    else "matched"
                ),
                "stratum": (
                    merge.UNMATCHED_E_STRATUM
                    if crossing_e_owner is None
                    else merge.MATCHED_E_STRATUM
                ),
                "p_boundary_index": 18,
            }
            for owner_id, _, crossing_e_owner in AUDIT_GREEDY_DISPLACERS
        },
        "control_by_owner": {},
    }


def test_two_thirds_threshold_is_the_scorers_own_integer_rule() -> None:
    for total in range(0, 41):
        threshold = analyze.two_thirds_threshold(total)
        for count in range(0, total + 1):
            expected = count * scorer.ROUTING_MAJORITY_DENOMINATOR >= (
                total * scorer.ROUTING_MAJORITY_NUMERATOR
            )
            assert (count >= threshold) is expected
    assert analyze.two_thirds_threshold(24) == 16
    assert analyze.two_thirds_threshold(22) == 15


def test_displaced_sub_tag_contingency_names_every_cell() -> None:
    contingency = analyze.displaced_sub_tag_contingency(audit_shaped_rows())
    cells = contingency["cells"]
    assert contingency["denominator"] == 17
    assert cells["likelihood_and_greedy_displaced"]["count"] == 10
    assert cells["likelihood_and_greedy_displaced"]["owner_ids"] == sorted(
        AUDIT_BOTH_SUB_TAGS
    )
    assert cells["likelihood_displaced_only"]["count"] == 5
    assert cells["likelihood_displaced_only"]["owner_ids"] == sorted(
        AUDIT_LIKELIHOOD_ONLY
    )
    assert cells["greedy_displaced_only"]["count"] == 2
    assert cells["greedy_displaced_only"]["owner_ids"] == sorted(
        AUDIT_GREEDY_ONLY_MIRROR
    )
    assert cells["neither_sub_tag"] == {"count": 0, "owner_ids": []}
    # The four cells are disjoint and exhaust the displaced cohort.
    assert sum(cell["count"] for cell in cells.values()) == contingency["denominator"]


def test_greedy_only_mirror_exclusion_keeps_the_route_directionally_stable() -> None:
    mirror = analyze.greedy_only_mirror_exclusion(
        audit_shaped_rows(), leading_branch="displaced"
    )
    assert mirror["excluded_owner_ids"] == sorted(AUDIT_GREEDY_ONLY_MIRROR)
    assert mirror["excluded_count"] == 2
    assert mirror["retained_interpretable_count"] == 22
    assert mirror["retained_leading_branch_count"] == 15
    assert mirror["retained_two_thirds_threshold"] == 15
    assert mirror["retained_reaches_two_thirds"] is True


def test_likelihood_only_cells_carry_no_realized_target_support() -> None:
    block = analyze.likelihood_only_realized_target_support(audit_shaped_rows())
    assert block["denominator"] == 5
    assert block["owner_ids"] == sorted(AUDIT_LIKELIHOOD_ONLY)
    assert block["greedy_status_counts_at_p_plus_e"] == {scorer.GREEDY_UNMATCHED: 5}
    assert block["unmatched_at_p_plus_e"]["count"] == 5
    assert block["unmatched_at_p_plus_e"]["owner_ids"] == sorted(AUDIT_LIKELIHOOD_ONLY)
    assert "no realized-target support" in block["note"]


def test_greedy_displacer_identity_pairs_every_displacer_against_crossing_e() -> None:
    identity = analyze.greedy_displacer_identity(
        audit_shaped_rows(), audit_plan_facts()
    )
    assert identity["denominator"] == 12
    assert identity["displacer_equals_crossing_e_count"] == 0
    assert identity["displacer_equals_crossing_e_owner_ids"] == []
    assert identity["not_determinable_count"] == 0
    assert identity["disposition_counts"] == {analyze.DISPLACER_NOT_E: 12}
    observed = {
        (pair["gt_owner_id"], pair["displacing_owner_id"], pair["crossing_e_owner_id"])
        for pair in identity["pairs"]
    }
    assert observed == set(AUDIT_GREEDY_DISPLACERS)
    # No causal reading is offered for the identity comparison.
    assert "never evidence that E caused" in identity["definition"]


def test_a_displacer_that_is_crossing_e_is_reported_as_such() -> None:
    """The comparison is a real test, not a constant: an equal cell is detected."""

    rows = [fragility_row("gt:10707:16", greedy_owner="gt:10707:17")]
    plan_facts = {
        **audit_plan_facts(),
        "by_owner": {
            "gt:10707:16": {
                "crossing_e_owner_id": "gt:10707:17",
                "crossing_e_strict_match_status": "matched",
                "stratum": merge.MATCHED_E_STRATUM,
                "p_boundary_index": 18,
            }
        },
    }
    identity = analyze.greedy_displacer_identity(rows, plan_facts)
    assert identity["displacer_equals_crossing_e_count"] == 1
    assert identity["displacer_equals_crossing_e_owner_ids"] == ["gt:10707:16"]


def test_an_unresolvable_crossing_e_owner_is_never_guessed() -> None:
    unresolved = {
        "source": analyze.CROSSING_E_SOURCE_NOT_RESOLVABLE,
        "reason": "no plan revision",
        "by_owner": {},
        "control_by_owner": {},
    }
    identity = analyze.greedy_displacer_identity(audit_shaped_rows(), unresolved)
    assert identity["disposition_counts"] == {analyze.DISPLACER_E_NOT_DETERMINABLE: 12}
    assert identity["not_determinable_count"] == 12
    assert identity["displacer_equals_crossing_e_count"] == 0
    assert all(pair["crossing_e_owner_id"] is None for pair in identity["pairs"])


def test_support_bound_disagreement_cells_are_reported_at_both_boundaries() -> None:
    rows = [
        fragility_row(
            "gt:4134:22",
            at_p=boundary_fields(support_disposition_l=scorer.SUPPORT_UNSUPPORTED),
        ),
        fragility_row(
            "gt:4134:27",
            branch="realization_fail",
            at_p=boundary_fields(support_disposition_l=scorer.SUPPORT_UNSUPPORTED),
        ),
        fragility_row(
            "gt:13923:1",
            branch="release_lost",
            at_p_plus_e=boundary_fields(
                boundary_label="P_plus_E",
                support_disposition_l=scorer.SUPPORT_UNSUPPORTED,
            ),
        ),
        fragility_row("gt:13923:2", branch="release_lost"),
    ]
    rows[2]["l_sensitivity_agrees_with_u"] = False
    block = analyze.support_bound_disagreement_cells(rows)
    assert block["total_count"] == 3
    assert block["count_by_context_role"] == {"at_p": 2, "at_p_plus_e": 1}
    assert block["owner_ids_by_context_role"] == {
        "at_p": ["gt:4134:22", "gt:4134:27"],
        "at_p_plus_e": ["gt:13923:1"],
    }
    assert [cell["boundary_label"] for cell in block["cells"]] == [
        "P_plus_E",
        "P",
        "P",
    ]
    assert all(
        cell["support_disposition_u"] == scorer.SUPPORT_SUPPORTED
        and cell["support_disposition_l"] == scorer.SUPPORT_UNSUPPORTED
        for cell in block["cells"]
    )
    # The support-level cells are strictly more than the branch disagreement.
    assert block["branch_disagreement_owner_ids"] == ["gt:13923:1"]
    assert "never collapsed into" in block["note"]


def test_forced_dc_greedy_target_realization_at_p_is_reported_over_all_owners() -> None:
    matched = ("gt:13923:11", "gt:14038:10", "gt:2685:20", "gt:5001:17", "gt:7511:1")
    rows = [fragility_row(owner_id) for owner_id in matched]
    rows += [
        fragility_row(
            f"gt:unmatched:{index}",
            at_p=boundary_fields(greedy_status=scorer.GREEDY_UNMATCHED),
        )
        for index in range(21)
    ]
    block = analyze.forced_dc_greedy_target_realization_at_p(rows)
    assert block["denominator"] == 26
    assert block["boundary_label"] == "P"
    assert block["target_match"]["count"] == 5
    assert block["target_match"]["owner_ids"] == sorted(matched)
    assert block["greedy_status_counts"] == {
        scorer.GREEDY_TARGET_MATCH: 5,
        scorer.GREEDY_UNMATCHED: 21,
    }
    assert block["target_match_construction_determined_at_p"]["count"] == 0


# ---------------------------------------------------------------------------
# Displacer characterization and the descriptive timing-control exhibits
# ---------------------------------------------------------------------------

#: The audited displacer facts: (displacing owner, native strict-match sorted row
#: indices, native true positive).  Every one of them is first covered *after*
#: the crossing owner's own P boundary index of 18.
AUDIT_DISPLACER_FACTS = {
    "gt:10707:17": ([], False),
    "gt:13923:15": ([], False),
    "gt:13923:16": ([20], True),
    "gt:14038:26": ([21], True),
    "gt:14439:5": ([19], True),
    "gt:1584:14": ([22], True),
    "gt:16228:18": ([23], True),
    "gt:16228:5": ([24], True),
    "gt:16228:48": ([25], True),
    "gt:16228:44": ([], False),
    "gt:5001:12": ([26], True),
    "gt:6040:9": ([27], True),
}


def audit_census_facts(**extra: Any) -> dict[str, Any]:
    by_owner = {
        owner_id: {
            "normalized_description": "person",
            "native_strict_match_pred_row_ids": [
                f"pred:sorted:greedy:0:10707:{index}" for index in indices
            ],
            "native_strict_match_row_indices": list(indices),
            "native_true_positive": true_positive,
            "native_false_negative": not true_positive,
        }
        for owner_id, (indices, true_positive) in AUDIT_DISPLACER_FACTS.items()
    }
    by_owner.update(extra)
    return {
        "source": analyze.CENSUS_OWNER_SOURCE_SEALED_LINEAGE,
        "reason": None,
        "census_run_root": "/frozen/census",
        "owner_registry_path": "/frozen/census/plan/owner-registry.jsonl",
        "owner_registry_sha256": "2" * 64,
        "by_owner": by_owner,
    }


def test_displacer_characterization_separates_construction_from_coverage() -> None:
    block = analyze.displacer_characterization(
        audit_shaped_rows(),
        plan_facts=audit_plan_facts(),
        census_facts=audit_census_facts(),
    )
    greedy = block["by_sub_tag"]["greedy_displaced"]
    assert greedy["denominator"] == 12
    # Same normalized description is construction-determined and is guarded.
    assert greedy["same_normalized_description"]["count"] == 12
    assert (
        "construction"
        in greedy["same_normalized_description"]["claim_guard"]
    )
    # The nontrivial readouts: nobody was covered before P, and the displacers
    # split 9 later-native-TP against 3 remaining native FN.
    assert greedy["covered_before_p"]["count"] == 0
    assert greedy["uncovered_before_p"]["count"] == 12
    assert greedy["uncovered_before_p"]["displacing_owner_ids"] == [
        displacer for _, displacer, _ in AUDIT_GREEDY_DISPLACERS
    ]
    assert greedy["coverage_not_determinable"]["count"] == 0
    assert greedy["native_disposition_counts"] == {
        analyze.NATIVE_DISPOSITION_TRUE_POSITIVE: 9,
        analyze.NATIVE_DISPOSITION_FALSE_NEGATIVE: 3,
    }
    assert greedy["displacer_native_true_positive"]["count"] == 9
    assert greedy["displacer_native_false_negative"]["count"] == 3
    assert "never evidence that it caused" in block["note"]
    assert (
        block["census_owner_registry_source"]["source"]
        == analyze.CENSUS_OWNER_SOURCE_SEALED_LINEAGE
    )


def test_covered_before_p_uses_the_sealed_boundary_convention() -> None:
    """Boundary ``b`` contains native rows ``< b``, so row ``b`` is not covered."""

    rows = [
        fragility_row("gt:10707:16", greedy_owner="gt:before"),
        fragility_row("gt:10707:17", greedy_owner="gt:at"),
        fragility_row("gt:10707:18", greedy_owner="gt:after"),
    ]
    plan_facts = {
        **audit_plan_facts(),
        "by_owner": {
            owner_id: {
                "crossing_e_owner_id": None,
                "crossing_e_strict_match_status": analyze.E_STRICT_MATCH_UNMATCHED,
                "stratum": merge.UNMATCHED_E_STRATUM,
                "p_boundary_index": 18,
            }
            for owner_id in ("gt:10707:16", "gt:10707:17", "gt:10707:18")
        },
    }
    census_facts = audit_census_facts(
        **{
            "gt:before": {
                "normalized_description": "person",
                "native_strict_match_pred_row_ids": ["pred:sorted:greedy:0:10707:17"],
                "native_strict_match_row_indices": [17],
                "native_true_positive": True,
                "native_false_negative": False,
            },
            "gt:at": {
                "normalized_description": "person",
                "native_strict_match_pred_row_ids": ["pred:sorted:greedy:0:10707:18"],
                "native_strict_match_row_indices": [18],
                "native_true_positive": True,
                "native_false_negative": False,
            },
            "gt:after": {
                "normalized_description": "bottle",
                "native_strict_match_pred_row_ids": ["pred:sorted:greedy:0:10707:19"],
                "native_strict_match_row_indices": [19],
                "native_true_positive": True,
                "native_false_negative": False,
            },
        }
    )
    greedy = analyze.displacer_characterization(
        rows, plan_facts=plan_facts, census_facts=census_facts
    )["by_sub_tag"]["greedy_displaced"]
    assert greedy["covered_before_p"]["gt_owner_ids"] == ["gt:10707:16"]
    assert greedy["uncovered_before_p"]["gt_owner_ids"] == [
        "gt:10707:17",
        "gt:10707:18",
    ]
    by_owner = {pair["gt_owner_id"]: pair for pair in greedy["pairs"]}
    assert by_owner["gt:10707:16"]["native_strict_match_row_indices_before_p"] == [17]
    assert by_owner["gt:10707:17"]["native_strict_match_row_indices_before_p"] == []
    # A displacer of a different description is still reported, not dropped.
    assert by_owner["gt:10707:18"]["same_normalized_description"] is False


def test_displacer_characterization_degrades_without_the_census_lineage() -> None:
    unresolved = {
        "source": analyze.CENSUS_OWNER_SOURCE_NOT_RESOLVABLE,
        "reason": "no lineage contract",
        "by_owner": {},
    }
    greedy = analyze.displacer_characterization(
        audit_shaped_rows(), plan_facts=audit_plan_facts(), census_facts=unresolved
    )["by_sub_tag"]["greedy_displaced"]
    assert greedy["denominator"] == 12
    assert greedy["coverage_not_determinable"]["count"] == 12
    assert greedy["covered_before_p"]["count"] == 0
    assert greedy["uncovered_before_p"]["count"] == 0
    assert greedy["same_normalized_description"]["count"] == 0
    assert greedy["native_disposition_counts"] == {
        analyze.NATIVE_DISPOSITION_NOT_RESOLVABLE: 12
    }


def control_row(
    owner_id: str,
    *,
    at_p: dict[str, Any] | None = None,
    at_p_plus_e: dict[str, Any] | None = None,
    cohort: str = merge.TIMING_CONTROL_COHORT,
) -> dict[str, Any]:
    p_fields = at_p or boundary_fields()
    ppe_fields = at_p_plus_e or boundary_fields(boundary_label="P_plus_E")
    return {
        "gt_owner_id": owner_id,
        "cohort": cohort,
        "quarantined": False,
        "paired_transitions": analyze.build_paired_transitions(
            p_fields, ppe_fields, coordinate_is_construction_determined=False
        ),
    }


def suppressed_control_row(owner_id: str, *, competitor: str) -> dict[str, Any]:
    """A control whose rank, support, margin sign and greedy match all fall."""

    return control_row(
        owner_id,
        at_p=boundary_fields(target_rank=1, target_minus_competitor_margin=1.0),
        at_p_plus_e=boundary_fields(
            boundary_label="P_plus_E",
            target_rank=3,
            target_minus_competitor_margin=-1.0,
            best_competitor_owner_id=competitor,
            support_disposition_u=scorer.SUPPORT_UNSUPPORTED,
            greedy_status=scorer.GREEDY_UNMATCHED,
        ),
    )


def audit_shaped_control_rows() -> list[dict[str, Any]]:
    """14 timing controls with the audited 5 / 6 / 5 / 1 suppression counts."""

    competitors = {
        "gt:10707:17": "gt:10707:18",
        "gt:10707:18": "gt:10707:16",
        "gt:14038:39": "gt:14038:42",
        "gt:2685:15": "gt:2685:17",
        "gt:4134:23": "gt:4134:22",
    }
    rows = [
        suppressed_control_row(owner_id, competitor=competitor)
        for owner_id, competitor in competitors.items()
    ]
    # Only one of the five also loses its greedy target match.
    for row in rows[1:]:
        row["paired_transitions"]["greedy_status"]["at_p"] = scorer.GREEDY_OTHER_OWNER_MATCH
    # One of the five keeps its U support, so the four readouts genuinely differ
    # from one another rather than moving as a single block.
    rows[3]["paired_transitions"]["support_disposition_u"]["at_p_plus_e"] = (
        scorer.SUPPORT_SUPPORTED
    )
    # Two further controls lose U support without any rank or margin move, so
    # U-support loss is 6 while rank and margin suppression stay at 5.
    rows += [
        control_row(
            owner_id,
            at_p_plus_e=boundary_fields(
                boundary_label="P_plus_E",
                support_disposition_u=scorer.SUPPORT_UNSUPPORTED,
            ),
        )
        for owner_id in ("gt:10707:4", "gt:14038:17")
    ]
    rows += [control_row(f"gt:quiet:{index}") for index in range(7)]
    return rows


def test_timing_control_suppression_is_descriptive_and_never_counted() -> None:
    block = analyze.timing_control_suppression(audit_shaped_control_rows())
    assert block["cohort"] == merge.TIMING_CONTROL_COHORT
    assert block["in_primary_denominator"] is False
    assert block["in_routing_denominator"] is False
    assert block["denominator"] == 14
    counts = {name: readout["count"] for name, readout in block["readouts"].items()}
    assert counts == {
        analyze.CONTROL_SUPPRESSION_RANK_WORSENED: 5,
        analyze.CONTROL_SUPPRESSION_U_SUPPORT_LOST: 6,
        analyze.CONTROL_SUPPRESSION_MARGIN_POSITIVE_TO_NEGATIVE: 5,
        analyze.CONTROL_SUPPRESSION_GREEDY_TARGET_LOST: 1,
    }
    assert block["readouts"][analyze.CONTROL_SUPPRESSION_RANK_WORSENED][
        "owner_ids"
    ] == [
        "gt:10707:17",
        "gt:10707:18",
        "gt:14038:39",
        "gt:2685:15",
        "gt:4134:23",
    ]
    assert block["readouts"][analyze.CONTROL_SUPPRESSION_GREEDY_TARGET_LOST][
        "owner_ids"
    ] == ["gt:10707:17"]
    assert block["readouts"][analyze.CONTROL_SUPPRESSION_U_SUPPORT_LOST]["count"] == 6
    assert "suppression is not unique to crossing owners" in block["note"]


def test_a_tp_replay_control_is_never_in_the_timing_control_readout() -> None:
    rows = audit_shaped_control_rows()
    rows.append(
        suppressed_control_row("gt:10707:0", competitor="gt:10707:1")
        | {"cohort": merge.TP_REPLAY_CONTROL_COHORT}
    )
    block = analyze.timing_control_suppression(rows)
    assert block["denominator"] == 14
    assert "gt:10707:0" not in block["owner_ids"]


def test_timing_control_competitor_coverage_is_confounded_and_labelled() -> None:
    plan_facts = {
        **audit_plan_facts(),
        "control_by_owner": {
            "gt:10707:17": {"control_boundary_index": 14},
            "gt:10707:18": {"control_boundary_index": 16},
            "gt:14038:39": {"control_boundary_index": 34},
            "gt:2685:15": {"control_boundary_index": 9},
            "gt:4134:23": {"control_boundary_index": 7},
        },
    }
    census_facts = audit_census_facts(
        **{
            owner_id: {
                "normalized_description": "person",
                "native_strict_match_pred_row_ids": (
                    [] if row_index is None else [f"pred:sorted:greedy:0:x:{row_index}"]
                ),
                "native_strict_match_row_indices": (
                    [] if row_index is None else [row_index]
                ),
                "native_true_positive": row_index is not None,
                "native_false_negative": row_index is None,
            }
            for owner_id, row_index in (
                ("gt:10707:18", None),
                ("gt:10707:16", None),
                ("gt:14038:42", None),
                ("gt:2685:17", 16),
                ("gt:4134:22", None),
            )
        }
    )
    block = analyze.timing_control_competitor_coverage(
        audit_shaped_control_rows(), plan_facts=plan_facts, census_facts=census_facts
    )
    assert block["in_primary_denominator"] is False
    assert block["in_routing_denominator"] is False
    assert block["denominator"] == 5
    assert block["uncovered_before_control_boundary"]["count"] == 5
    assert block["covered_before_control_boundary"]["count"] == 0
    assert block["uncovered_before_control_boundary"]["best_competitor_owner_ids"] == [
        "gt:10707:18",
        "gt:10707:16",
        "gt:14038:42",
        "gt:2685:17",
        "gt:4134:22",
    ]
    assert block["competitor_native_disposition_counts"] == {
        analyze.NATIVE_DISPOSITION_FALSE_NEGATIVE: 4,
        analyze.NATIVE_DISPOSITION_TRUE_POSITIVE: 1,
    }
    assert "within-category scheduling instability" in block["note"]
    assert "no causal contrast" in block["note"]


# ---------------------------------------------------------------------------
# Conclusion fragility, end to end
# ---------------------------------------------------------------------------

#: Every field the science audit requires, as a path into ``conclusion_fragility``.
REQUIRED_FRAGILITY_PATHS: tuple[tuple[str, ...], ...] = (
    ("route_margin", "leading_branch_count"),
    ("route_margin", "routing_denominator_count"),
    ("route_margin", "two_thirds_threshold"),
    ("route_margin", "owners_of_margin"),
    ("displaced_sub_tag_contingency", "cells", "likelihood_and_greedy_displaced"),
    ("displaced_sub_tag_contingency", "cells", "likelihood_displaced_only"),
    ("displaced_sub_tag_contingency", "cells", "greedy_displaced_only"),
    ("displaced_sub_tag_contingency", "cells", "neither_sub_tag"),
    ("greedy_only_mirror_exclusion", "excluded_owner_ids"),
    ("greedy_only_mirror_exclusion", "retained_leading_branch_count"),
    ("greedy_only_mirror_exclusion", "retained_two_thirds_threshold"),
    ("greedy_only_mirror_exclusion", "retained_reaches_two_thirds"),
    ("likelihood_only_realized_target_support", "unmatched_at_p_plus_e"),
    ("likelihood_only_realized_target_support", "greedy_status_counts_at_p_plus_e"),
    ("forced_dc_greedy_target_realization_at_p", "target_match"),
    ("forced_dc_greedy_target_realization_at_p", "greedy_status_counts"),
    ("greedy_displacer_identity", "pairs"),
    ("greedy_displacer_identity", "disposition_counts"),
    ("greedy_displacer_identity", "displacer_equals_crossing_e_count"),
    ("greedy_displacer_identity", "crossing_e_owner_identity_source", "source"),
    ("displacer_characterization", "census_owner_registry_source", "source"),
    ("displacer_characterization", "covered_before_p_definition"),
    ("displacer_characterization", "by_sub_tag", "greedy_displaced"),
    ("displacer_characterization", "by_sub_tag", "likelihood_displaced"),
    ("support_bound_disagreement_cells", "cells"),
    ("support_bound_disagreement_cells", "count_by_context_role"),
    ("support_bound_disagreement_cells", "branch_disagreement_owner_ids"),
    ("timing_control_suppression", "readouts"),
    ("timing_control_suppression", "in_primary_denominator"),
    ("timing_control_competitor_coverage", "cells"),
    ("timing_control_competitor_coverage", "uncovered_before_control_boundary"),
    ("timing_control_competitor_coverage", "competitor_native_disposition_counts"),
)


def mirror_specs() -> list[OwnerSpec]:
    """The routable cohort with two displaced cells carried by the greedy mirror."""

    specs = fx.routable_specs()
    replaced = 0
    for index, spec in enumerate(specs):
        if spec.intent == "displaced_likelihood" and replaced < 2:
            specs[index] = replace(spec, intent="displaced_greedy")
            replaced += 1
    assert replaced == 2
    return specs


def test_conclusion_fragility_is_reported_without_changing_the_decision(
    routable: dict[str, Any],
) -> None:
    report = routable["report"]
    fragility = report["conclusion_fragility"]
    assert report["schema_version"] == analyze.REPORT_SCHEMA_VERSION
    assert report["schema_version"].endswith(".v2")
    assert fragility["changes_branch_assignment_or_decision"] is False
    # The frozen v1 surface is untouched beside it.
    assert report["decision"] == analyze.DECISION_ROUTE
    assert report["routing"]["routing_denominator_count"] == 24
    assert report["routing"]["evaluated"]["routed_branch"] == "displaced"
    assert fragility["route_margin"] == {
        "definition": analyze.OPERATIONAL_DEFINITIONS["two_thirds_threshold"],
        "leading_branch": "displaced",
        "leading_branch_count": 17,
        "routing_denominator_count": 24,
        "two_thirds_threshold": 16,
        "owners_of_margin": 1,
    }


def test_every_required_fragility_field_is_present(routable: dict[str, Any]) -> None:
    fragility = routable["report"]["conclusion_fragility"]
    for path in REQUIRED_FRAGILITY_PATHS:
        node: Any = fragility
        for key in path:
            assert isinstance(node, dict), f"{'.'.join(path)} is not reachable"
            assert key in node, f"conclusion_fragility.{'.'.join(path)} is missing"
            node = node[key]
    # Every fragility slice states its own operational definition.
    for name, block in fragility.items():
        if isinstance(block, dict) and name != "route_margin":
            assert "definition" in block, f"{name} states no operational definition"


def test_the_mirror_exclusion_sensitivity_is_wired_end_to_end(tmp_path: Path) -> None:
    result = analyze_specs(tmp_path, mirror_specs(), name="mirror")
    report = result["report"]
    fragility = report["conclusion_fragility"]
    cells = fragility["displaced_sub_tag_contingency"]["cells"]
    assert cells["greedy_displaced_only"]["count"] == 2
    assert cells["likelihood_displaced_only"]["count"] == 15
    assert cells["neither_sub_tag"]["count"] == 0

    mirror = fragility["greedy_only_mirror_exclusion"]
    assert mirror["excluded_count"] == 2
    assert mirror["excluded_owner_ids"] == cells["greedy_displaced_only"]["owner_ids"]
    assert mirror["retained_interpretable_count"] == 22
    assert mirror["retained_leading_branch_count"] == 15
    assert mirror["retained_two_thirds_threshold"] == 15
    assert mirror["retained_reaches_two_thirds"] is True
    # The sensitivity is reported; the routed decision is not recomputed from it.
    assert report["decision"] == analyze.DECISION_ROUTE
    assert report["routing"]["evaluated"]["routed_branch"] == "displaced"


def test_control_rows_carry_the_paired_transitions_the_readout_needs(
    routable: dict[str, Any],
) -> None:
    control_rows = [
        row
        for row in routable["owner_rows"]
        if row["row_kind"] == "crossing_boundary_control_owner_row"
    ]
    assert all(
        row["schema_version"] == analyze.OWNER_ROW_SCHEMA_VERSION for row in control_rows
    )
    assert analyze.OWNER_ROW_SCHEMA_VERSION.endswith(".v2")
    timing = [row for row in control_rows if row["cohort"] == merge.TIMING_CONTROL_COHORT]
    replay = [
        row for row in control_rows if row["cohort"] == merge.TP_REPLAY_CONTROL_COHORT
    ]
    assert len(timing) == 14
    assert all(row["paired_transitions"] is not None for row in timing)
    # The TP replay controls have a single due boundary and therefore no pair.
    assert len(replay) == 12
    assert all(row["paired_transitions"] is None for row in replay)
    suppression = routable["report"]["conclusion_fragility"]["timing_control_suppression"]
    assert suppression["denominator"] == 14
    assert suppression["owners_without_a_paired_readout"] == []


# ---------------------------------------------------------------------------
# The two lineage-bound reference reads
# ---------------------------------------------------------------------------


def reseal_merge_receipt(merged_dir: Path, patch: dict[str, Any]) -> None:
    """Rewrite the merge receipt's plan lineage and re-seal it honestly."""

    path = merged_dir / merge.MERGE_RECEIPT_NAME
    receipt = json.loads(path.read_text())
    receipt["plan"]["lineage"] = patch
    receipt.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = scorer.sha256_json(receipt)
    path.write_bytes(scorer.canonical_json_bytes(receipt) + b"\n")


def test_a_sealed_plan_that_is_not_the_cohort_registry_is_never_guessed(
    routable: dict[str, Any],
) -> None:
    """The fixture's plan files are sealed stand-ins, so identity degrades."""

    identity = routable["report"]["conclusion_fragility"]["greedy_displacer_identity"]
    source = identity["crossing_e_owner_identity_source"]
    assert source["source"] == analyze.CROSSING_E_SOURCE_NOT_RESOLVABLE
    assert "e_row" in str(source["reason"])
    # The digest of the file that *was* read is still sealed into the report.
    assert len(str(source["cohort_registry_sha256"])) == 64


def test_a_missing_sealed_plan_file_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "no-plan-file"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    (capture.plan_dir / analyze.PLAN_COHORT_REGISTRY_NAME).unlink()
    with pytest.raises(analyze.AnalysisContractError, match="is missing at"):
        analyze.run_analysis(root / "merged")


def test_a_modified_sealed_plan_file_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "edited-plan-file"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    target = capture.plan_dir / analyze.PLAN_COHORT_REGISTRY_NAME
    target.write_bytes(target.read_bytes() + b'{"gt_owner_id": "gt:0:0"}\n')
    with pytest.raises(analyze.AnalysisContractError, match="does not match the digest"):
        analyze.run_analysis(root / "merged")


def test_the_census_owner_registry_degrades_without_a_lineage_contract(
    routable: dict[str, Any],
) -> None:
    source = routable["report"]["conclusion_fragility"]["displacer_characterization"][
        "census_owner_registry_source"
    ]
    assert source["source"] == analyze.CENSUS_OWNER_SOURCE_NOT_RESOLVABLE
    assert analyze.CENSUS_OWNER_REGISTRY_NAME in str(source["reason"])
    assert source["owner_registry_sha256"] is None


def test_a_declared_but_missing_census_owner_registry_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-census"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    reseal_merge_receipt(
        root / "merged",
        {
            "census_run_root": str(tmp_path / "absent-census"),
            "census_input_files": {
                analyze.CENSUS_OWNER_REGISTRY_NAME: {
                    "path": analyze.CENSUS_OWNER_REGISTRY_NAME,
                    "byte_size": 10,
                    "sha256": "3" * 64,
                }
            },
        },
    )
    with pytest.raises(analyze.AnalysisContractError, match="is missing at"):
        analyze.run_analysis(root / "merged")


def test_a_modified_census_owner_registry_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "edited-census"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    census_root = tmp_path / "census"
    registry = census_root / analyze.CENSUS_OWNER_REGISTRY_NAME
    registry.parent.mkdir(parents=True)
    payload = json.dumps({"gt_owner_id": "gt:10707:17"}).encode("utf-8") + b"\n"
    registry.write_bytes(payload)
    reseal_merge_receipt(
        root / "merged",
        {
            "census_run_root": str(census_root),
            "census_input_files": {
                analyze.CENSUS_OWNER_REGISTRY_NAME: {
                    "path": analyze.CENSUS_OWNER_REGISTRY_NAME,
                    "byte_size": len(payload),
                    "sha256": "4" * 64,
                }
            },
        },
    )
    with pytest.raises(analyze.AnalysisContractError, match="does not match the digest"):
        analyze.run_analysis(root / "merged")


def test_a_sealed_census_registry_without_the_fields_is_never_inferred(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shallow-census"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    census_root = tmp_path / "shallow"
    registry = census_root / analyze.CENSUS_OWNER_REGISTRY_NAME
    registry.parent.mkdir(parents=True)
    payload = json.dumps({"gt_owner_id": "gt:10707:17"}).encode("utf-8") + b"\n"
    registry.write_bytes(payload)
    reseal_merge_receipt(
        root / "merged",
        {
            "census_run_root": str(census_root),
            "census_input_files": {
                analyze.CENSUS_OWNER_REGISTRY_NAME: {
                    "path": analyze.CENSUS_OWNER_REGISTRY_NAME,
                    "byte_size": len(payload),
                    "sha256": scorer.sha256_bytes(payload),
                }
            },
        },
    )
    report = analyze.run_analysis(root / "merged")["report"]
    source = report["conclusion_fragility"]["displacer_characterization"][
        "census_owner_registry_source"
    ]
    assert source["source"] == analyze.CENSUS_OWNER_SOURCE_NOT_RESOLVABLE
    assert "native_strict_match_pred_row_ids" in str(source["reason"])


def test_the_receipt_seals_both_reference_read_sources(tmp_path: Path) -> None:
    root = tmp_path / "reference-receipt"
    capture = fx.build_capture(root)
    capture.run_merge(root / "merged")
    files = analyze.build_output_files(analyze.run_analysis(root / "merged"))
    receipt = json.loads(files[analyze.RECEIPT_NAME])
    policy = receipt["policy"]
    assert policy["conclusion_fragility_changed_the_decision"] is False
    assert policy["crossing_e_owner_identity_source"] == (
        analyze.CROSSING_E_SOURCE_NOT_RESOLVABLE
    )
    assert policy["census_owner_registry_source"] == (
        analyze.CENSUS_OWNER_SOURCE_NOT_RESOLVABLE
    )
    assert policy["census_owner_registry_sha256"] is None
    assert policy["secondary_compatibility_read"] is False
    merge.assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="analysis receipt"
    )


def test_markdown_carries_the_fragility_reading(tmp_path: Path) -> None:
    result = analyze_specs(tmp_path, mirror_specs(), name="fragility-md")
    rendered = analyze.render_markdown(result["report"])
    assert "## Conclusion fragility" in rendered
    assert "### Displaced sub-tag contingency (n=17)" in rendered
    assert "| greedy_displaced_only | 2 |" in rendered
    assert "### Greedy displacer identity" in rendered
    assert "### Displacer characterization" in rendered
    assert "### U/L support-disposition disagreement cells" in rendered
    assert "### Timing-control suppression" in rendered
    assert "### Reading" in rendered
    assert (
        "the route is threshold-crossing: `displaced` reaches 17/24 against a "
        "threshold of 16, so 1 owner(s) separate routing from split" in rendered
    )
    assert (
        "it is directionally stable to excluding the 2 greedy-only mirror cells: "
        "15/22 still reaches two thirds (True)" in rendered
    )
    assert "the displaced evidence is heterogeneous, not one mechanism" in rendered
    assert (
        "there is no evidence that crossing E itself is the displacer" in rendered
    )
    assert "never evidence of a same-description mechanism" in rendered
    assert "no causal contrast is claimed" in rendered
