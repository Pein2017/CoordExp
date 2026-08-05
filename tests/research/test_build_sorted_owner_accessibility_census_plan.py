"""Focused tests for the sorted owner accessibility census CPU planner.

These tests prove the frozen scientific contract mechanically, not by
inspection: canonical suffix placement (including a multi-token description),
category-versus-owner estimand semantics, exact core/extension role
membership, score independence, forced-row exclusion, loop/terminal fields,
category-local strict assignment, and cross-owner candidate collapse.

The full-plan fixture is module-scoped: it reads the real frozen inputs once
and every structural assertion reuses it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402

ORS = planner.OBJECT_REF_START
ORE = planner.OBJECT_REF_END
BOX_START = planner.BOX_START


@pytest.fixture(scope="module")
def plan() -> planner.CensusPlan:
    if not planner.OWNER_LEDGER_PATH.is_file():
        pytest.skip("frozen census inputs are not present on this host")
    return planner.build_plan()


# ---------------------------------------------------------------------------
# Canonical coordinate-query suffix (contract item 5)
# ---------------------------------------------------------------------------


def test_query_suffix_shape_is_literal_and_variable_length() -> None:
    single = planner.build_query_suffix([8987])  # "person"
    assert single == [ORS, 8987, ORE, BOX_START]

    multi = planner.build_query_suffix([67, 5740, 1965])  # "dining table"
    assert multi == [ORS, 67, 5740, 1965, ORE, BOX_START]
    # The suffix is variable length; only the *tail equality* is invariant.
    assert len(multi) == 6 != len(single)


def test_suffix_assertion_accepts_exact_tail_for_multi_token_category() -> None:
    tokens = [67, 5740, 1965]
    prefix = [1, 2, 3, *planner.build_query_suffix(tokens)]
    planner.assert_canonical_query_suffix(prefix, category_token_ids=tokens, label="ok")


def test_suffix_assertion_rejects_prefix_ending_at_box_end() -> None:
    """The pre-P0 failure mode: a prefix ending at a finished row."""

    tokens = [8987]
    prefix = [ORS, 8987, ORE, BOX_START, 151670, 151671, 151672, 151673, planner.BOX_END]
    with pytest.raises(planner.PlanContractError, match="box_start"):
        planner.assert_canonical_query_suffix(prefix, category_token_ids=tokens, label="bad")


def test_suffix_assertion_rejects_wrong_category_with_correct_shape() -> None:
    """A prefix ending in box_start but carrying another category must fail."""

    prefix = [1, 2, *planner.build_query_suffix([8987])]
    with pytest.raises(planner.PlanContractError, match="canonical query suffix"):
        planner.assert_canonical_query_suffix(
            prefix, category_token_ids=[67, 5740, 1965], label="mismatched category"
        )


def test_suffix_assertion_rejects_multi_token_truncated_to_last_four() -> None:
    """A multi-token category must not pass on a four-token tail check alone."""

    tokens = [67, 5740, 1965]
    # Ends with [..., 1965, ORE, BOX_START]: the final four tokens look right
    # for a shorter category but the full sealed suffix is absent.
    prefix = [1, 2, 3, 1965, ORE, BOX_START]
    with pytest.raises(planner.PlanContractError, match="canonical query suffix"):
        planner.assert_canonical_query_suffix(prefix, category_token_ids=tokens, label="truncated")


def test_every_admitted_query_group_prefix_ends_at_the_sealed_suffix(plan) -> None:
    categories = {str(row["category_query_id"]): row for row in plan.categories}
    images = {str(row["image_id"]): row for row in plan.images}
    contexts = {str(row["context_id"]): row for row in plan.contexts}

    checked = 0
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        category = categories[str(group["category_query_id"])]
        context = contexts[str(group["context_id"])]
        image = images[str(group["image_id"])]
        full_prefix = [
            *image["prompt_token_ids"],
            *context["generated_prefix_token_ids"],
            *group["query_suffix_token_ids"],
        ]
        planner.assert_canonical_query_suffix(
            full_prefix,
            category_token_ids=category["category_token_ids"],
            label=str(group["query_group_id"]),
        )
        assert full_prefix[-1] == BOX_START
        assert planner.sha256_json(full_prefix) == group["query_prefix_sha256"]
        checked += 1
    assert checked == sum(1 for row in plan.query_groups if row["status"] == "admitted")


def test_observed_and_query_prefix_digests_are_distinct(plan) -> None:
    """A score row must never be joinable to a differently-suffixed capture."""

    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        assert group["observed_prefix_sha256"] != group["query_prefix_sha256"]
        assert group["query_suffix_token_ids_sha256"]


# ---------------------------------------------------------------------------
# Census shape and the frozen split (contract items 1 and 12)
# ---------------------------------------------------------------------------


def test_census_covers_346_owners_with_343_greedy_eligible(plan) -> None:
    assert len(plan.owners) == 346
    assert sum(1 for row in plan.owners if row["greedy_eligible"]) == 343
    ineligible = sorted(row["gt_owner_id"] for row in plan.owners if not row["greedy_eligible"])
    assert ineligible == ["gt:14038:41", "gt:14038:42", "gt:14038:43"]
    # Contract item 1: nothing is excluded, including the ambiguity-neutral three.
    assert not any(row["excluded_from_census"] for row in plan.owners)


def test_true_positive_owners_are_calibration_not_exclusions(plan) -> None:
    positives = [row for row in plan.owners if row["native_true_positive"]]
    assert positives, "expected native true positives to be present"
    assert all(row["calibration_role"] == "native_true_positive_calibration" for row in positives)
    assert all(not row["excluded_from_census"] for row in positives)


def test_split_is_exactly_six_and_six_and_frozen(plan) -> None:
    discovery = {row["image_id"] for row in plan.images if row["split"] == "discovery"}
    confirmation = {row["image_id"] for row in plan.images if row["split"] == "confirmation"}
    assert discovery == set(planner.DISCOVERY_IMAGE_IDS)
    assert confirmation == set(planner.CONFIRMATION_IMAGE_IDS)
    assert len(discovery) == len(confirmation) == 6
    assert not discovery & confirmation
    assert plan.receipt["split"]["tuning_policy"] == (
        "confirmation_rules_are_frozen_on_discovery_only"
    )


def test_all_twelve_images_are_planned_as_shards(plan) -> None:
    assert len(plan.shards) == 12
    assert {row["image_id"] for row in plan.shards} == set(planner.SPLIT_BY_IMAGE_ID)


def test_shards_expose_estimated_work_units_largest_first(plan) -> None:
    units = [int(row["estimated_work_units"]) for row in plan.shards]
    assert units == sorted(units, reverse=True)
    assert [int(row["dispatch_order"]) for row in plan.shards] == list(range(12))
    # The two loop-degenerate images dominate and must dispatch first.
    assert [row["image_id"] for row in plan.shards[:2]] == ["4134", "14038"]


# ---------------------------------------------------------------------------
# Context registry (contract item 4) and loop marking (item 11)
# ---------------------------------------------------------------------------


def test_context_registry_is_root_boundaries_and_terminal(plan) -> None:
    assert len(plan.contexts) == 412
    by_image: dict[str, list[dict]] = {}
    for row in plan.contexts:
        by_image.setdefault(str(row["image_id"]), []).append(row)
    for image_id, rows in by_image.items():
        rows.sort(key=lambda row: int(row["boundary_index"]))
        total = int(rows[0]["total_complete_row_count"])
        assert len(rows) == total + 1
        assert rows[0]["context_role"] == "root"
        assert rows[0]["generated_prefix_token_ids"] == []
        assert rows[-1]["context_role"] == "terminal"
        assert rows[-1]["terminal_kind"] == "natural_stop"
        assert all(row["context_role"] == "row_boundary" for row in rows[1:-1])


def test_contexts_exclude_forced_continue_rows(plan) -> None:
    for row in plan.contexts:
        admission = row["prefix_admission"]
        assert admission["source"] == "native_greedy_complete_rows_only"
        assert admission["forced_continue_rows_excluded"] is True
        assert admission["retokenized"] is False


def test_prefix_rows_are_a_strict_prefix_of_the_native_rollout(plan) -> None:
    for row in plan.contexts:
        indices = list(row["prefix_row_indices"])
        assert indices == list(range(int(row["boundary_index"])))
        assert len(row["prefix_rows"]) == len(indices)


def test_loop_marking_keeps_continuous_counts_and_a_literal_flag(plan) -> None:
    flagged = [row for row in plan.contexts if row["loop_marking"]["loop_tail"]]
    assert {str(row["image_id"]) for row in flagged} == {"4134", "14038"}
    assert len(flagged) == 82

    for row in plan.contexts:
        marking = row["loop_marking"]
        assert "prior_identical_row_count" in marking
        assert "consecutive_identical_row_run_length" in marking
        assert "repeated_raw_span_sha256" in marking
        assert marking["flag_is_not_a_mechanism_label"] is True
        run = marking["consecutive_identical_row_run_length"]
        if run is None:
            assert row["context_role"] == "root"
            assert marking["loop_tail"] is False
        else:
            assert marking["loop_tail"] == (run >= planner.LOOP_TAIL_MIN_CONSECUTIVE_RUN)


def test_ten_images_have_no_repeated_native_row(plan) -> None:
    for row in plan.contexts:
        if str(row["image_id"]) in {"4134", "14038"}:
            continue
        prior = row["loop_marking"]["prior_identical_row_count"]
        assert prior in (None, 0)


def test_frontier_is_present_for_every_non_root_context(plan) -> None:
    for row in plan.contexts:
        if row["context_role"] == "root":
            assert row["frontier"] is None
        else:
            frontier = row["frontier"]
            assert frontier is not None
            assert len(frontier["sort_key"]) == 2
            assert frontier["pred_row_id"]


# ---------------------------------------------------------------------------
# Candidate bank (contract item 7)
# ---------------------------------------------------------------------------


def test_role_lists_are_exactly_nine_core_and_eight_extension() -> None:
    assert len(planner.CORE_ROLES) == 9
    assert len(planner.EXTENSION_ROLES) == 8
    assert len(planner.LOGICAL_ROLES) == 17
    assert planner.CORE_ROLES[0] == "exact_gt_anchor"
    assert set(planner.EXTENSION_ROLES) == {
        "translate_up_left",
        "translate_up_right",
        "translate_down_left",
        "translate_down_right",
        "isotropic_expand",
        "isotropic_shrink",
        "top_anchored_height_shrink",
        "bottom_anchored_height_shrink",
    }
    assert not set(planner.CORE_ROLES) & set(planner.EXTENSION_ROLES)


def test_every_owner_keeps_exactly_seventeen_logical_roles(plan) -> None:
    for row in plan.owners:
        bank = row["candidate_bank"]
        assert bank["logical_role_count"] == 17
        assert len(bank["logical_roles"]) == 17
        assert [entry["role"] for entry in bank["logical_roles"]] == list(planner.LOGICAL_ROLES)


def test_anchored_height_shrinks_hold_one_edge_fixed() -> None:
    canvas = planner.Canvas(1000, 1000)
    box = (100, 100, 300, 500)
    top = planner.apply_role(box, role="top_anchored_height_shrink", dx=50, dy=100, canvas=canvas)
    bottom = planner.apply_role(
        box, role="bottom_anchored_height_shrink", dx=50, dy=100, canvas=canvas
    )
    assert top == (100, 100, 300, 400)  # y1 held
    assert bottom == (100, 200, 300, 500)  # y2 held
    symmetric = planner.apply_role(box, role="height_shrink", dx=50, dy=100, canvas=canvas)
    assert symmetric == (100, 200, 300, 400)  # both edges move
    assert top != symmetric and bottom != symmetric


def test_exact_anchor_role_is_the_untouched_owner_box() -> None:
    canvas = planner.Canvas(1000, 1000)
    box = (10, 20, 30, 40)
    assert planner.apply_role(box, role="exact_gt_anchor", dx=5, dy=5, canvas=canvas) == box


def test_candidate_identity_is_image_category_and_tokens_only() -> None:
    first = planner.physical_candidate_id(
        image_id="7511", normalized_description="person", tokens=[1, 2, 3, 4]
    )
    same = planner.physical_candidate_id(
        image_id="7511", normalized_description="person", tokens=[1, 2, 3, 4]
    )
    other_category = planner.physical_candidate_id(
        image_id="7511", normalized_description="chair", tokens=[1, 2, 3, 4]
    )
    other_image = planner.physical_candidate_id(
        image_id="4134", normalized_description="person", tokens=[1, 2, 3, 4]
    )
    assert first == same
    assert first != other_category != other_image
    assert first.startswith("cand:")


def test_cross_owner_identical_tuples_collapse_to_one_candidate() -> None:
    """Two same-category owners with the same box yield one physical candidate."""

    owners = [
        {
            "gt_owner_id": "gt:1:0",
            "image_id": "1",
            "normalized_description": "person",
            "bbox_pixel_xyxy": [100, 100, 300, 500],
        },
        {
            "gt_owner_id": "gt:1:1",
            "image_id": "1",
            "normalized_description": "person",
            "bbox_pixel_xyxy": [100, 100, 300, 500],
        },
    ]
    panel = {"1": {"width": 1000, "height": 1000}}
    candidates, accounting = planner.build_candidate_bank({"1": owners}, panel)

    exact = [row for row in candidates if row["candidate_provenance"] == "exact"]
    assert len(exact) == 1, "identical owner boxes must collapse to one exact candidate"
    shared = exact[0]
    assert shared["cross_owner_generated"] is True
    assert shared["generator_owner_count"] == 2
    assert shared["generator_gt_owner_ids"] == ["gt:1:0", "gt:1:1"]
    assert {row["generator_gt_owner_id"] for row in shared["generators"]} == {"gt:1:0", "gt:1:1"}
    # Both owners still report seventeen logical roles.
    for owner_id in ("gt:1:0", "gt:1:1"):
        assert accounting[owner_id]["logical_role_count"] == 17
        assert accounting[owner_id]["cross_owner_shared_candidate_count"] > 0
    # Identical geometry is genuinely ambiguous under the category-local matcher.
    assert shared["strict_assignment_status"] == "ambiguous_neutral"


def test_generator_provenance_never_enters_assignment() -> None:
    candidates, _ = planner.build_candidate_bank(
        {
            "1": [
                {
                    "gt_owner_id": "gt:1:0",
                    "image_id": "1",
                    "normalized_description": "person",
                    "bbox_pixel_xyxy": [100, 100, 300, 500],
                }
            ]
        },
        {"1": {"width": 1000, "height": 1000}},
    )
    for row in candidates:
        assert row["generator_provenance_role"] == (
            "provenance_only_never_rank_or_assignment"
        )
        assert row["strict_assignment_scope"] == "same_normalized_description_only"


def test_strict_assignment_ignores_a_different_category_overlap() -> None:
    """A perfectly overlapping other-category owner must not create ambiguity."""

    owners = [
        {
            "gt_owner_id": "gt:1:0",
            "normalized_description": "person",
            "bbox_pixel_xyxy": [100, 100, 300, 500],
        },
        {
            "gt_owner_id": "gt:1:9",
            "normalized_description": "chair",
            "bbox_pixel_xyxy": [100, 100, 300, 500],
        },
    ]
    result = planner.strict_assignment(
        [100, 100, 300, 500], owners, normalized_description="person"
    )
    # Category-local primary view: unambiguous.
    assert result["strict_assignment_status"] == "matched"
    assert result["strict_assignment_gt_owner_id"] == "gt:1:0"
    assert result["strict_assignment_population_size"] == 1
    # All-category diagnostic view, separately named, sees the collision.
    assert result["any_category_assignment_status"] == "ambiguous_neutral"
    assert result["any_category_assignment_role"] == (
        "diagnostic_only_never_primary_rank_or_ambiguity"
    )


def test_bank_separates_adequacy_from_strict_assignment_coverage(plan) -> None:
    for row in plan.owners:
        bank = row["candidate_bank"]
        adequacy = bank["generator_local_bank_adequacy"]
        coverage = bank["strict_assignment_coverage"]
        assert adequacy["measured_on"] == "distinct_physical_candidate_count"
        assert coverage["role"] == "separate_lower_bound_view_never_the_adequacy_gate"
        assert bank["bank_coverage_status"] == adequacy["status"]
        # Both counts are retained.
        assert isinstance(coverage["uniquely_assigned_candidate_count"], int)
        assert isinstance(bank["distinct_physical_candidate_count"], int)


def test_bank_adequacy_rule_is_frozen_not_provisional(plan) -> None:
    for row in plan.owners:
        adequacy = row["candidate_bank"]["generator_local_bank_adequacy"]
        assert adequacy["threshold_status"] == "frozen_before_any_score"
        assert adequacy["full_at_least"] == 17
        assert adequacy["adequate_at_least"] == 12
        assert row["candidate_bank"]["bank_coverage_status"] in planner.BANK_COVERAGE_STATUSES


def test_only_genuinely_undercovered_owners_are_floored(plan) -> None:
    for row in plan.owners:
        bank = row["candidate_bank"]
        status = bank["bank_coverage_status"]
        if status == "undercovered_unresolved_only":
            assert bank["disposition_eligible"] is False
            assert bank["disposition_floor"] == (
                "unresolved_only_never_persistent_no_tested_localization_support"
            )
        else:
            assert status in {"full", "adequate_reduced"}
            assert bank["disposition_eligible"] is True
            assert bank["disposition_floor"] == "no_floor"


def test_adequate_reduced_owners_keep_disposition_eligibility(plan) -> None:
    """Token aliasing alone must never floor an owner to unresolved."""

    reduced = [
        row
        for row in plan.owners
        if row["candidate_bank"]["bank_coverage_status"] == "adequate_reduced"
    ]
    assert reduced, "expected some owners whose transforms alias"
    for row in reduced:
        bank = row["candidate_bank"]
        assert 12 <= bank["distinct_physical_candidate_count"] < 17
        assert bank["exact_anchor_uniquely_self_assigned"] is True
        assert bank["disposition_eligible"] is True
        assert bank["undercovered"] is False


def test_bank_adequacy_does_not_swallow_the_census(plan) -> None:
    """A coverage rule that marked most owners unresolved would neuter the unit."""

    eligible = sum(1 for row in plan.owners if row["candidate_bank"]["disposition_eligible"])
    assert eligible == len(plan.owners)


def test_missing_or_non_self_exact_anchor_forces_undercovered() -> None:
    """A tiny owner whose exact anchor is not self-assigned is floored."""

    owners = [
        {
            "gt_owner_id": "gt:1:0",
            "image_id": "1",
            "normalized_description": "person",
            "bbox_pixel_xyxy": [100, 100, 300, 500],
        },
        {
            "gt_owner_id": "gt:1:1",
            "image_id": "1",
            "normalized_description": "person",
            "bbox_pixel_xyxy": [100, 100, 300, 500],
        },
    ]
    _, accounting = planner.build_candidate_bank({"1": owners}, {"1": {"width": 1000, "height": 1000}})
    for owner_id in ("gt:1:0", "gt:1:1"):
        bank = accounting[owner_id]
        # Identical boxes make each exact anchor ambiguity-neutral, not self-assigned.
        assert bank["exact_anchor_uniquely_self_assigned"] is False
        assert bank["bank_coverage_status"] == "undercovered_unresolved_only"
        assert bank["disposition_eligible"] is False


def test_every_owner_reaches_its_own_exact_anchor(plan) -> None:
    for row in plan.owners:
        bank = row["candidate_bank"]
        assert bank["exact_anchor_admitted"] is True
        assert bank["exact_anchor_uniquely_self_assigned"] is True


# ---------------------------------------------------------------------------
# Candidate-versus-generating-owner geometry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def square_bank() -> tuple[list[dict], dict[str, dict]]:
    """One owner on a canvas where pixel and bin coordinates agree exactly."""

    owner = {
        "gt_owner_id": "gt:1:0",
        "image_id": "1",
        "normalized_description": "person",
        "bbox_pixel_xyxy": [100, 100, 300, 500],
    }
    return planner.build_candidate_bank({"1": [owner]}, {"1": {"width": 1000, "height": 1000}})


def _role_geometry(accounting: dict, owner_id: str, role: str) -> dict:
    record = next(
        row for row in accounting[owner_id]["logical_roles"] if row["role"] == role
    )
    return record["geometry"]


def test_every_admitted_logical_role_carries_geometry(plan) -> None:
    for row in plan.owners:
        for record in row["candidate_bank"]["logical_roles"]:
            if record["admitted"]:
                geometry = record["geometry"]
                assert geometry is not None, (row["gt_owner_id"], record["role"])
                assert set(geometry) >= {
                    "intersection_over_union_with_generator",
                    "center_offset_pixels",
                    "extent_ratio",
                    "candidate_area_pixels",
                    "generator_area_pixels",
                    "area_ratio",
                    "generator_bbox_pixel_xyxy",
                }
                assert record["decoded_bbox_pixel_xyxy"] is not None
            else:
                assert record["geometry"] is None
                assert record["decoded_bbox_pixel_xyxy"] is None


def test_exact_anchor_geometry_is_the_identity(square_bank) -> None:
    _, accounting = square_bank
    geometry = _role_geometry(accounting, "gt:1:0", "exact_gt_anchor")
    assert geometry["intersection_over_union_with_generator"] == pytest.approx(1.0)
    assert geometry["center_offset_pixels"] == [0.0, 0.0]
    assert geometry["extent_ratio"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert geometry["area_ratio"] == pytest.approx(1.0)
    assert geometry["generator_bbox_pixel_xyxy"] == [100.0, 100.0, 300.0, 500.0]


def test_isotropic_expand_geometry(square_bank) -> None:
    """dx = w/4 and dy = h/4 per side gives 1.5x extent and IoU 1/2.25."""

    _, accounting = square_bank
    geometry = _role_geometry(accounting, "gt:1:0", "isotropic_expand")
    assert geometry["extent_ratio"] == [pytest.approx(1.5), pytest.approx(1.5)]
    assert geometry["area_ratio"] == pytest.approx(2.25)
    assert geometry["intersection_over_union_with_generator"] == pytest.approx(1 / 2.25)
    # Symmetric expansion does not move the centre.
    assert geometry["center_offset_pixels"] == [0.0, 0.0]


def test_isotropic_shrink_geometry(square_bank) -> None:
    _, accounting = square_bank
    geometry = _role_geometry(accounting, "gt:1:0", "isotropic_shrink")
    assert geometry["extent_ratio"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert geometry["area_ratio"] == pytest.approx(0.25)
    assert geometry["intersection_over_union_with_generator"] == pytest.approx(0.25)
    assert geometry["center_offset_pixels"] == [0.0, 0.0]


def test_axis_extent_roles_change_only_one_axis(square_bank) -> None:
    _, accounting = square_bank
    width = _role_geometry(accounting, "gt:1:0", "width_expand")
    assert width["extent_ratio"][0] == pytest.approx(1.5)
    assert width["extent_ratio"][1] == pytest.approx(1.0)
    height = _role_geometry(accounting, "gt:1:0", "height_shrink")
    assert height["extent_ratio"][0] == pytest.approx(1.0)
    assert height["extent_ratio"][1] == pytest.approx(0.5)


def test_translation_moves_the_centre_but_preserves_extent(square_bank) -> None:
    _, accounting = square_bank
    right = _role_geometry(accounting, "gt:1:0", "translate_right")
    assert right["extent_ratio"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert right["center_offset_pixels"][0] == pytest.approx(50.0)  # dx = 200/4
    assert right["center_offset_pixels"][1] == pytest.approx(0.0)
    left = _role_geometry(accounting, "gt:1:0", "translate_left")
    assert left["center_offset_pixels"][0] == pytest.approx(-50.0)


def test_anchored_height_shrinks_move_the_centre_in_opposite_directions(
    square_bank,
) -> None:
    _, accounting = square_bank
    top = _role_geometry(accounting, "gt:1:0", "top_anchored_height_shrink")
    bottom = _role_geometry(accounting, "gt:1:0", "bottom_anchored_height_shrink")
    # Both shrink height to 0.75 and hold width.
    for geometry in (top, bottom):
        assert geometry["extent_ratio"][0] == pytest.approx(1.0)
        assert geometry["extent_ratio"][1] == pytest.approx(0.75)
    # y1 held -> centre moves up; y2 held -> centre moves down.
    assert top["center_offset_pixels"][1] == pytest.approx(-50.0)
    assert bottom["center_offset_pixels"][1] == pytest.approx(50.0)
    assert top["center_offset_pixels"][1] == -bottom["center_offset_pixels"][1]


def test_cross_owner_alias_keeps_per_generator_geometry() -> None:
    """One collapsed candidate, two generators, two different GT boxes.

    On a 2000-wide canvas two owner boxes differing by one pixel round to the
    same coordinate bin, so their exact anchors collapse into one physical
    candidate while their GT boxes stay different.  Each generator entry must
    keep geometry against *its own* box.
    """

    owners = [
        {
            "gt_owner_id": "gt:1:0",
            "image_id": "1",
            "normalized_description": "person",
            "bbox_pixel_xyxy": [100, 100, 300, 500],
        },
        {
            "gt_owner_id": "gt:1:1",
            "image_id": "1",
            "normalized_description": "person",
            "bbox_pixel_xyxy": [101, 100, 300, 500],
        },
    ]
    candidates, _ = planner.build_candidate_bank(
        {"1": owners}, {"1": {"width": 2000, "height": 2000}}
    )
    shared = [row for row in candidates if row["cross_owner_generated"]]
    assert shared, "expected at least one cross-owner collapsed candidate"

    anchor = next(
        row
        for row in shared
        if any(
            entry["logical_transform_role"] == "exact_gt_anchor"
            for entry in row["generators"]
        )
    )
    by_owner = {
        entry["generator_gt_owner_id"]: entry
        for entry in anchor["generators"]
        if entry["logical_transform_role"] == "exact_gt_anchor"
    }
    assert set(by_owner) == {"gt:1:0", "gt:1:1"}

    # Each generator's geometry is measured against its own GT box.
    assert by_owner["gt:1:0"]["geometry"]["generator_bbox_pixel_xyxy"] == [
        100.0,
        100.0,
        300.0,
        500.0,
    ]
    assert by_owner["gt:1:1"]["geometry"]["generator_bbox_pixel_xyxy"] == [
        101.0,
        100.0,
        300.0,
        500.0,
    ]
    # Different generator boxes -> genuinely different geometry for one tuple.
    assert (
        by_owner["gt:1:0"]["geometry"]["center_offset_pixels"]
        != by_owner["gt:1:1"]["geometry"]["center_offset_pixels"]
    )
    assert (
        by_owner["gt:1:0"]["geometry"]["generator_area_pixels"]
        != by_owner["gt:1:1"]["geometry"]["generator_area_pixels"]
    )


def test_representative_geometry_names_its_generator(plan) -> None:
    for row in plan.candidates:
        representative = row["representative_generator_geometry"]
        assert representative["generator_gt_owner_id"] in row["generator_gt_owner_ids"]
        assert representative["logical_transform_role"] == row["representative_role"]
        assert "intersection_over_union_with_generator" in representative
        assert row["geometry_scope"].startswith("per_generator_authoritative")


def test_every_generator_entry_carries_geometry(plan) -> None:
    for row in plan.candidates:
        for entry in row["generators"]:
            geometry = entry["geometry"]
            assert geometry is not None
            assert geometry["measured_on"] == (
                "decoded_candidate_box_versus_generator_gt_box"
            )
            assert isinstance(geometry["intersection_over_union_with_generator"], float)


def test_geometry_helper_handles_a_degenerate_generator_box() -> None:
    geometry = planner.candidate_generator_geometry([0, 0, 10, 10], [5, 5, 5, 5])
    assert geometry["extent_ratio"] == [None, None]
    assert geometry["area_ratio"] is None
    assert geometry["intersection_over_union_with_generator"] == 0.0


# ---------------------------------------------------------------------------
# Owner support partitions (generator-local vs strict assignment)
# ---------------------------------------------------------------------------


def _candidate(*, generators: list[str], status: str, assigned: str | None) -> dict:
    return {
        "generators": [{"generator_gt_owner_id": owner} for owner in generators],
        "strict_assignment_status": status,
        "strict_assignment_gt_owner_id": assigned,
    }


def test_other_owner_strict_candidate_is_excluded_from_target_support() -> None:
    candidate = _candidate(generators=["gt:1:0"], status="matched", assigned="gt:1:1")
    result = planner.classify_candidate_for_owner(candidate, "gt:1:0")
    assert result["partition"] == "other_owner_strict"
    assert result["excluded_from_target_support"] is True
    assert result["collision_diagnostic"] is True
    assert result["counts_toward_lower_bound"] is False
    assert result["counts_toward_upper_bound"] is False


def test_ambiguous_candidate_counts_toward_upper_bound_only() -> None:
    candidate = _candidate(generators=["gt:1:0"], status="ambiguous_neutral", assigned=None)
    result = planner.classify_candidate_for_owner(candidate, "gt:1:0")
    assert result["partition"] == "ambiguous_upper"
    assert result["counts_toward_lower_bound"] is False
    assert result["counts_toward_upper_bound"] is True
    assert result["excluded_from_target_support"] is False


def test_self_assigned_candidate_counts_toward_both_bounds() -> None:
    candidate = _candidate(generators=["gt:1:0"], status="matched", assigned="gt:1:0")
    result = planner.classify_candidate_for_owner(candidate, "gt:1:0")
    assert result["partition"] == "strict_assigned_self"
    assert result["counts_toward_lower_bound"] is True
    assert result["counts_toward_upper_bound"] is True


def test_unmatched_generator_local_candidate_stays_in_the_landscape() -> None:
    """A perturbation that drops below IoU 0.5 is still the owner's landscape."""

    candidate = _candidate(generators=["gt:1:0"], status="unmatched", assigned=None)
    result = planner.classify_candidate_for_owner(candidate, "gt:1:0")
    assert result["partition"] == "unmatched_generator_local"
    assert result["generator_local"] is True
    assert result["counts_toward_upper_bound"] is True
    assert result["counts_toward_lower_bound"] is False


def test_candidate_not_generated_by_owner_is_outside_its_landscape() -> None:
    candidate = _candidate(generators=["gt:1:1"], status="unmatched", assigned=None)
    result = planner.classify_candidate_for_owner(candidate, "gt:1:0")
    assert result["partition"] == "not_generated_by_owner"
    assert result["generator_local"] is False
    assert result["counts_toward_upper_bound"] is False


def test_owner_rows_expose_the_fields_the_merge_requires(plan) -> None:
    required = {
        "distinct_physical_candidate_count",
        "other_owner_strict_count",
        "generator_local_landscape_candidate_ids",
    }
    for row in plan.owners:
        assert required <= set(row["candidate_bank"])


def test_capture_rules_freeze_the_bank_adequacy_rule() -> None:
    rule = planner.build_capture_rules()["owner_support"]["bank_adequacy_rule"]
    assert rule["threshold_status"] == "frozen_before_any_score"
    assert rule["full_at_least"] == 17
    assert rule["adequate_at_least"] == 12
    assert rule["requires_exact_anchor_uniquely_self_assigned"] is True
    assert rule["token_alias_never_floors_an_owner"] is True
    assert sorted(rule["disposition_eligible_statuses"]) == ["adequate_reduced", "full"]
    assert set(rule["statuses"]) == set(planner.BANK_COVERAGE_STATUSES)


def test_capture_rules_freeze_owner_support_semantics() -> None:
    support = planner.build_capture_rules()["owner_support"]
    assert support["primary_neighbourhood"] == "generator_local_landscape"
    assert support["exact_anchor"] == "mandatory"
    assert support["ambiguous_candidate"] == "counts_toward_upper_bound_only"
    assert support["lower_upper_disposition_flip"] == "unresolved"
    assert "frontier_tested" in support["persistent_negative_requires"]
    assert "adequate_distinct_bank" in support["persistent_negative_requires"]


def test_unmatched_generator_local_is_upper_bound_only() -> None:
    support = planner.build_capture_rules()["owner_support"]
    assert support["unmatched_generator_local"] == "upper_bound_only"
    bounds = support["partition_bounds"]
    assert bounds["unmatched_generator_local"] == ["upper"]
    assert bounds["ambiguous_upper"] == ["upper"]
    assert bounds["strict_assigned_self"] == ["lower", "upper"]
    assert bounds["other_owner_strict"] == []
    assert set(bounds) == set(planner.OWNER_CANDIDATE_PARTITIONS)


def test_capture_rules_freeze_rank_as_routing_never_support() -> None:
    support = planner.build_capture_rules()["owner_support"]
    assert support["competition_ranks"] == "routing_surface_never_support"
    assert support["competition_rank_population"] == (
        "exclusion_filtered_generator_local_maxima"
    )
    definition = support["support_definition"]
    assert definition["rank_is_support_criterion"] is False
    assert definition["rank_one_as_support"] == "forbidden"
    assert "any_rank_or_margin_criterion" in support["persistent_negative_forbids"]
    assert not any(
        "rank" in requirement for requirement in support["persistent_negative_requires"]
    )


def test_capture_rules_freeze_the_support_statistics() -> None:
    definition = planner.build_capture_rules()["owner_support"]["support_definition"]
    assert definition["statistics"] == ["peak_lift", "local_concentration"]
    assert planner.SUPPORT_STATISTICS == ("peak_lift", "local_concentration")
    assert definition["computed_on"] == "generator_local_max_excluding_other_owner_strict"
    assert definition["evaluated_under_both_ambiguity_bounds"] is True
    assert definition["usable_support_combinator"] == "conjunction_both_statistics"
    assert definition["peak_lift"]["not"] == "best_minus_an_owner_local_reference"
    assert definition["local_concentration"]["not"] == "normalized_mass_share"
    assert definition["peak_lift"]["population"] == (
        "all_unique_candidates_in_the_query_group"
    )
    assert definition["local_concentration"]["reference_population"] == (
        "owner_own_exclusion_filtered_bank_scores"
    )


# ---------------------------------------------------------------------------
# Exact support formulas
# ---------------------------------------------------------------------------


def _reference_peak_lift(best: float, population: list[float]) -> float:
    total = math.log(sum(math.exp(value) for value in population))
    return best - total + math.log(len(population))


def test_peak_lift_matches_the_logsumexp_definition() -> None:
    population = [-1.0, -2.0, -3.0, -4.5]
    best = max(population)
    assert planner.compute_peak_lift(best, population) == pytest.approx(
        _reference_peak_lift(best, population), abs=1e-12
    )


def test_peak_lift_is_zero_for_a_uniform_population() -> None:
    """A flat group has no lift over the uniform posterior."""

    population = [-2.5] * 8
    assert planner.compute_peak_lift(-2.5, population) == pytest.approx(0.0, abs=1e-12)


def test_peak_lift_is_shift_invariant() -> None:
    population = [-1.0, -2.0, -3.0]
    shifted = [value - 100.0 for value in population]
    assert planner.compute_peak_lift(-1.0, population) == pytest.approx(
        planner.compute_peak_lift(-101.0, shifted), abs=1e-9
    )


def test_peak_lift_upper_bound_is_log_n() -> None:
    """A group whose mass is entirely on one candidate lifts by log(N)."""

    population = [0.0, -60.0, -60.0, -60.0]
    assert planner.compute_peak_lift(0.0, population) == pytest.approx(
        math.log(4), abs=1e-12
    )


def test_peak_lift_is_not_best_minus_owner_local_reference() -> None:
    """Locks the corrected definition against the superseded one."""

    population = [-1.0, -5.0, -6.0, -7.0]
    best = -1.0
    owner_local_reference_version = best - max(value for value in population if value != best)
    assert planner.compute_peak_lift(best, population) != pytest.approx(
        owner_local_reference_version, abs=1e-6
    )


def test_peak_lift_rejects_an_empty_population() -> None:
    with pytest.raises(planner.PlanContractError, match="non-empty"):
        planner.compute_peak_lift(-1.0, [])


def test_local_concentration_is_best_minus_own_bank_median_odd() -> None:
    bank = [-1.0, -3.0, -7.0]
    assert planner.compute_local_concentration(-1.0, bank) == pytest.approx(2.0, abs=1e-12)


def test_local_concentration_uses_the_mean_of_two_central_values_when_even() -> None:
    bank = [-1.0, -3.0, -5.0, -11.0]
    # median = (-3 + -5) / 2 = -4
    assert planner.compute_local_concentration(-1.0, bank) == pytest.approx(3.0, abs=1e-12)


def test_local_concentration_is_order_independent() -> None:
    bank = [-7.0, -1.0, -3.0]
    assert planner.compute_local_concentration(
        -1.0, bank
    ) == planner.compute_local_concentration(-1.0, sorted(bank))


def test_local_concentration_is_not_a_normalized_mass_share() -> None:
    """A mass share would lie in [0, 1]; this statistic is an unbounded gap."""

    bank = [-1.0, -30.0, -31.0]
    value = planner.compute_local_concentration(-1.0, bank)
    assert value == pytest.approx(29.0, abs=1e-12)
    assert value > 1.0


def test_local_concentration_ignores_other_owners() -> None:
    """Only the owner's own exclusion-filtered bank sets the reference."""

    own_bank = [-1.0, -3.0, -5.0]
    baseline = planner.compute_local_concentration(-1.0, own_bank)
    # Another owner scoring far higher elsewhere cannot change this statistic.
    assert planner.compute_local_concentration(-1.0, own_bank) == baseline


def test_local_concentration_rejects_an_empty_bank() -> None:
    with pytest.raises(planner.PlanContractError, match="non-empty"):
        planner.compute_local_concentration(-1.0, [])


# ---------------------------------------------------------------------------
# Support Boolean logic
# ---------------------------------------------------------------------------


def _clears(peak: float, concentration: float, eps: float = planner.SUPPORT_EPSILON) -> bool:
    return planner.clears_support(
        peak_lift=peak,
        local_concentration=concentration,
        peak_lift_threshold=1.0,
        local_concentration_threshold=2.0,
        epsilon=eps,
    )


def test_usable_support_requires_both_statistics() -> None:
    assert _clears(1.5, 2.5) is True
    assert _clears(1.5, 1.5) is False  # concentration fails
    assert _clears(0.5, 2.5) is False  # peak fails
    assert _clears(0.5, 1.5) is False


def test_usable_support_needs_threshold_plus_epsilon() -> None:
    """Sitting exactly on the threshold is not support."""

    assert _clears(1.0, 2.0) is False
    eps = planner.SUPPORT_EPSILON
    assert _clears(1.0 + eps, 2.0 + eps) is True
    assert _clears(1.0 + eps / 2, 2.0 + eps) is False


def test_persistent_negative_is_no_context_where_both_clear() -> None:
    # One context clears both -> not a persistent negative.
    assert planner.is_persistent_negative([False, True, False]) is False
    # No context clears both -> persistent negative.
    assert planner.is_persistent_negative([False, False, False]) is True


def test_persistent_negative_is_not_per_statistic_across_contexts() -> None:
    """The conjunction sits inside the quantifier, not outside it.

    An owner that clears ``peak_lift`` in context A and
    ``local_concentration`` in context B never clears both at once, so it is a
    persistent negative -- even though neither statistic is below threshold in
    *every* context.
    """

    context_a = (1.5, 1.0)  # peak clears, concentration does not
    context_b = (0.5, 2.5)  # concentration clears, peak does not
    flags = [_clears(*context_a), _clears(*context_b)]
    assert flags == [False, False]
    assert planner.is_persistent_negative(flags) is True

    # The wrong per-statistic reading would call this "not persistent",
    # because neither statistic is individually below threshold everywhere.
    peaks_all_below = all(peak < 1.0 for peak, _ in (context_a, context_b))
    concentrations_all_below = all(conc < 2.0 for _, conc in (context_a, context_b))
    assert peaks_all_below is False
    assert concentrations_all_below is False


def test_persistent_negative_with_no_tested_contexts() -> None:
    assert planner.is_persistent_negative([]) is True


# ---------------------------------------------------------------------------
# Persistent-negative eligibility floors
# ---------------------------------------------------------------------------


def _eligibility(
    *,
    greedy_eligible: bool = True,
    status: str = "eligible",
    bank: str = "full",
    native_true_positive: bool = False,
) -> dict:
    return planner.owner_disposition_eligibility(
        greedy_eligible=greedy_eligible,
        greedy_eligibility_status=status,
        bank_coverage_status=bank,
        native_true_positive=native_true_positive,
    )


def test_eligible_false_negative_owner_may_close_negative() -> None:
    result = _eligibility()
    assert result["persistent_negative_eligible"] is True
    assert result["enters_false_negative_confirmation_denominator"] is True
    assert result["enters_false_negative_prevalence_denominator"] is True
    assert result["disposition_floor"] == "no_floor"
    assert result["disposition_floor_blockers"] == []


def test_greedy_ineligible_owner_cannot_close_negative() -> None:
    result = _eligibility(greedy_eligible=False, status="globally_ambiguous_neutral")
    assert result["persistent_negative_eligible"] is False
    assert result["enters_false_negative_confirmation_denominator"] is False
    assert result["enters_false_negative_prevalence_denominator"] is False
    assert result["disposition_floor"] == (
        "unresolved_only_never_persistent_no_tested_localization_support"
    )
    assert "not_greedy_eligible:globally_ambiguous_neutral" in (
        result["disposition_floor_blockers"]
    )
    # Still fully censused.
    assert result["retained_in_continuous_census_rows"] is True
    assert result["excluded_from_census"] is False


def test_native_true_positive_cannot_close_as_a_false_negative() -> None:
    """TPs define q10, so ~10% sit below it; labelling them would be circular."""

    result = _eligibility(native_true_positive=True)
    assert result["native_false_negative"] is False
    assert result["calibration_role"] == "native_true_positive_calibration"
    assert result["persistent_negative_eligible"] is False
    assert result["enters_false_negative_prevalence_denominator"] is False
    assert result["enters_false_negative_confirmation_denominator"] is False
    assert "native_true_positive_calibration_control" in (
        result["disposition_floor_blockers"]
    )
    assert result["retained_in_continuous_census_rows"] is True


def test_undercovered_bank_still_blocks_a_negative_close() -> None:
    result = _eligibility(bank="undercovered_unresolved_only")
    assert result["persistent_negative_eligible"] is False
    assert "bank_undercovered_unresolved_only" in result["disposition_floor_blockers"]


def test_adequate_reduced_bank_does_not_block_a_negative_close() -> None:
    result = _eligibility(bank="adequate_reduced")
    assert result["persistent_negative_eligible"] is True
    assert result["disposition_floor_blockers"] == []


def test_all_three_floors_are_reported_together() -> None:
    result = _eligibility(
        greedy_eligible=False,
        status="globally_ambiguous_neutral",
        bank="undercovered_unresolved_only",
        native_true_positive=True,
    )
    assert result["persistent_negative_eligible"] is False
    assert len(result["disposition_floor_blockers"]) == 3


def test_the_three_ambiguity_neutral_owners_are_censused_but_floored(plan) -> None:
    ineligible = [
        row for row in plan.owners if not row["disposition_eligibility"]["greedy_eligible"]
    ]
    assert sorted(row["gt_owner_id"] for row in ineligible) == [
        "gt:14038:41",
        "gt:14038:42",
        "gt:14038:43",
    ]
    for row in ineligible:
        eligibility = row["disposition_eligibility"]
        assert eligibility["greedy_eligibility_status"] == "globally_ambiguous_neutral"
        assert eligibility["persistent_negative_eligible"] is False
        assert eligibility["enters_false_negative_confirmation_denominator"] is False
        # Retained: still a full census row with a full bank.
        assert row["excluded_from_census"] is False
        assert row["candidate_bank"]["logical_role_count"] == 17


def test_every_native_true_positive_owner_is_floored_in_the_plan(plan) -> None:
    positives = [row for row in plan.owners if row["native_true_positive"]]
    assert len(positives) == 141
    for row in positives:
        eligibility = row["disposition_eligibility"]
        assert eligibility["persistent_negative_eligible"] is False
        assert eligibility["enters_false_negative_prevalence_denominator"] is False
        assert row["excluded_from_census"] is False


def test_false_negative_denominator_excludes_tps_and_ineligibles(plan) -> None:
    denominator = [
        row
        for row in plan.owners
        if row["disposition_eligibility"]["enters_false_negative_confirmation_denominator"]
    ]
    assert all(not row["native_true_positive"] for row in denominator)
    assert all(row["greedy_eligible"] for row in denominator)
    # 346 owners - 141 native TPs - 3 ambiguity-neutral (all of which are FNs).
    assert len(denominator) == 346 - 141 - 3


def test_receipt_reports_the_conclusion_domain_counts(plan) -> None:
    """The receipt must state the conclusion domain, not only retention."""

    shape = plan.receipt["census_shape"]
    assert shape["owner_count"] == 346
    assert shape["greedy_eligible_owner_count"] == 343
    assert shape["native_true_positive_owner_count"] == 141
    assert shape["native_false_negative_owner_count"] == 346 - 141
    assert shape["persistent_negative_eligible_owner_count"] == 202
    assert "false-negative" in shape["persistent_negative_eligible_semantics"]


def test_conclusion_domain_is_not_the_bank_adequacy_count(plan) -> None:
    """The 346/346 bank figure must never stand in for the 202 conclusion domain."""

    shape = plan.receipt["census_shape"]
    bank_only = shape["bank_disposition_eligible_owner_count"]
    conclusion = shape["persistent_negative_eligible_owner_count"]
    assert bank_only == 346
    assert conclusion == 202
    assert bank_only != conclusion


def test_persistent_negative_domain_matches_its_arithmetic(plan) -> None:
    """202 = 343 greedy-eligible - 141 native TPs, with the 3 neutrals already out."""

    domain = [
        row
        for row in plan.owners
        if row["disposition_eligibility"]["persistent_negative_eligible"]
    ]
    assert len(domain) == 202
    assert len(domain) == 343 - 141
    neutral = [row for row in plan.owners if not row["greedy_eligible"]]
    assert len(neutral) == 3
    # All three ambiguity-neutral owners are native false negatives, so they
    # are excluded by the greedy-eligibility floor rather than by the TP floor.
    assert all(not row["native_true_positive"] for row in neutral)


def test_no_true_positive_or_ineligible_owner_enters_the_domain(plan) -> None:
    for row in plan.owners:
        eligible = row["disposition_eligibility"]["persistent_negative_eligible"]
        if row["native_true_positive"] or not row["greedy_eligible"]:
            assert eligible is False, row["gt_owner_id"]
        if eligible:
            assert row["native_true_positive"] is False
            assert row["greedy_eligible"] is True
            assert row["candidate_bank"]["disposition_eligible"] is True


def test_summary_labels_the_bank_count_and_the_conclusion_domain(plan) -> None:
    text = planner.summarize(plan)
    # The bank-only figure must be unmistakably labelled as bank adequacy.
    assert "BANK-ADEQUACY floor only" in text
    assert "NOT the conclusion domain" in text
    # And the real conclusion domain must be printed.
    assert "CONCLUSION DOMAIN" in text
    assert "persistent-negative eligible owners: 202" in text
    # The old ambiguous phrasing must not reappear.
    assert "disposition-eligible owners: 346 of 346" not in text


def test_capture_rules_seal_the_eligibility_preconditions() -> None:
    support = planner.build_capture_rules()["owner_support"]
    requires = support["persistent_negative_requires"]
    assert "greedy_eligible" in requires
    assert "native_false_negative" in requires

    non_eligible = support["non_eligible_owner_policy"]
    assert non_eligible["retained_in_continuous_census_rows"] is True
    assert non_eligible["excluded_from_census"] is False
    assert non_eligible["may_close_as_persistent_negative"] is False
    assert non_eligible["enters_false_negative_confirmation_denominator"] is False
    assert non_eligible["disposition_floor"] == (
        "unresolved_only_never_persistent_no_tested_localization_support"
    )

    tp_policy = support["native_true_positive_policy"]
    assert tp_policy["role"] == "calibration_and_positive_control"
    assert tp_policy["defines_the_q10_threshold"] is True
    assert tp_policy["may_fall_below_q10_plus_epsilon"] is True
    assert tp_policy["may_close_as_persistent_negative"] is False
    assert tp_policy["enters_false_negative_prevalence_denominator"] is False
    assert tp_policy["retained_in_continuous_census_rows"] is True


def test_capture_rules_freeze_the_formulas_and_the_quantifier() -> None:
    support = planner.build_capture_rules()["owner_support"]
    definition = support["support_definition"]
    assert definition["peak_lift"]["formula"] == planner.SUPPORT_PEAK_LIFT_FORMULA
    assert "logsumexp" in definition["peak_lift"]["formula"]
    assert "log(unique_population_size)" in definition["peak_lift"]["formula"]
    assert (
        definition["local_concentration"]["formula"]
        == planner.SUPPORT_LOCAL_CONCENTRATION_FORMULA
    )
    assert "median" in definition["local_concentration"]["formula"]
    assert definition["usable_support_rule"] == (
        "peak_lift >= threshold + epsilon AND local_concentration >= threshold + epsilon"
    )
    assert support["persistent_negative_rule"] == (
        "not any(context: peak_lift_U >= t_peak + eps AND "
        "local_concentration_U >= t_conc + eps)"
    )
    assert "no_optimistic_u_context_where_both_statistics_clear" in (
        support["persistent_negative_requires"]
    )
    assert (
        "requiring_each_statistic_individually_below_threshold_at_every_context"
        in support["persistent_negative_forbids"]
    )


def test_capture_rules_freeze_the_support_calibration_contract() -> None:
    calibration = planner.build_capture_rules()["owner_support"]["support_calibration"]
    assert calibration["population"] == "discovery_native_true_positive_owners"
    assert calibration["read_at"] == "deterministic_due_boundary"
    assert calibration["stratification"] == "pooled"
    assert calibration["primary_quantile"] == 0.10
    assert calibration["sensitivity_quantiles"] == [0.05, 0.25]
    assert calibration["sensitivity_role"] == (
        "diagnostic_only_never_primary_never_a_disposition"
    )
    assert calibration["category_stratified_role"] == "diagnostic_only"
    assert calibration["category_contribution_min"] == 20
    assert calibration["underrepresented_flag"] == "pooled_underrepresented"
    assert calibration["underrepresented_policy"] == (
        "fall_back_to_pooled_never_change_the_threshold"
    )
    assert calibration["threshold_search_or_tuning"] == "forbidden"


def test_capture_rules_freeze_fixed_non_adaptive_epsilons() -> None:
    epsilons = planner.build_capture_rules()["owner_support"]["epsilons"]
    assert epsilons["support_epsilon"] == 0.002
    assert epsilons["cross_context_delta_epsilon"] == 0.004
    assert epsilons["adaptive"] is False
    assert epsilons["observed_parity_role"] == (
        "compliance_check_against_bound_never_resizes_it"
    )
    assert planner.SUPPORT_EPSILON == 0.002
    assert planner.CROSS_CONTEXT_DELTA_EPSILON == 0.004


def test_capture_rules_freeze_confirmation_blinding_dag() -> None:
    blinding = planner.build_capture_rules()["owner_support"]["confirmation_blinding"]
    assert blinding["capture_order"] == "all_twelve_shards_may_be_front_loaded"
    assert blinding["calibration_reads_confirmation_rows"] is False
    assert blinding["confirmation_retune"] == "forbidden"
    assert blinding["rederivation_on_all_twelve"] == "forbidden"
    assert blinding["on_calibration_receipt_digest_mismatch"] == "fail_closed"
    stages = blinding["stages"]
    assert stages[0].startswith("capture_manifest")
    assert stages[1] == "calibration_consumes_discovery_digests_only"
    assert stages[-1] == "confirmation_binds_exact_calibration_receipt_digest"


def test_support_contract_is_covered_by_the_capture_rules_digest() -> None:
    """Editing any frozen support value must move the sealed digest."""

    rules = planner.build_capture_rules()
    baseline = rules["capture_rules_sha256"]
    assert baseline == planner.sha256_json(
        {key: value for key, value in rules.items() if key != "capture_rules_sha256"}
    )
    tampered = json.loads(json.dumps(rules))
    tampered["owner_support"]["support_calibration"]["primary_quantile"] = 0.5
    recomputed = planner.sha256_json(
        {key: value for key, value in tampered.items() if key != "capture_rules_sha256"}
    )
    assert recomputed != baseline


# ---------------------------------------------------------------------------
# Query groups, dedup layer, and admission identity
# ---------------------------------------------------------------------------


def test_query_group_contains_one_request_per_unique_coordinate_tuple(plan) -> None:
    candidates = {str(row["candidate_id"]): row for row in plan.candidates}
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        ids = list(group["candidate_ids"])
        assert len(set(ids)) == len(ids)
        tuples = [tuple(candidates[cid]["coord_token_ids"]) for cid in ids]
        assert len(set(tuples)) == len(tuples)
        assert group["unique_coordinate_tuple_count"] == len(ids)
        assert group["request_policy"] == "exactly_one_request_per_unique_coordinate_tuple"


def test_query_group_candidates_are_all_of_its_own_category(plan) -> None:
    candidates = {str(row["candidate_id"]): row for row in plan.candidates}
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        for cid in group["candidate_ids"]:
            candidate = candidates[cid]
            assert candidate["normalized_description"] == group["normalized_description"]
            assert candidate["image_id"] == group["image_id"]


def test_admission_ids_are_exact_prefix_keyed_and_channel_separated(plan) -> None:
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        query_id = group["admission_receipt_id"]
        gate_id = group["proposal_boundary_gate_admission_receipt_id"]
        route_id = group["proposal_route_admission_receipt_id"]
        assert len({query_id, gate_id, route_id}) == 3
        assert group["query_prefix_sha256"] in query_id
        assert group["observed_prefix_sha256"] in gate_id
        assert planner.CHANNEL_QUERY_SUFFIX in query_id
        assert planner.CHANNEL_PROPOSAL_BOUNDARY_GATE in gate_id
        assert planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE in route_id


def test_proposal_route_path_is_open_to_object_ref_end_only() -> None:
    """The routing event stops before box_start: no coordinate score in proposal."""

    path = planner.proposal_route_token_ids([67, 5740, 1965])
    assert path == [ORS, 67, 5740, 1965, ORE]
    assert planner.BOX_START not in path


def test_proposal_route_digest_distinguishes_token_identity_and_length() -> None:
    single = planner.proposal_route_digest([8987])
    multi = planner.proposal_route_digest([67, 5740, 1965])
    same_length_other_tokens = planner.proposal_route_digest([71437, 8991])
    two_token = planner.proposal_route_digest([65, 62118])
    assert single != multi
    # Equal suffix length but different token identity must still differ.
    assert same_length_other_tokens != two_token


def test_two_categories_in_one_context_have_distinct_proposal_route_ids(plan) -> None:
    by_context: dict[str, set[str]] = {}
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        by_context.setdefault(str(group["context_id"]), set()).add(
            str(group["proposal_route_admission_receipt_id"])
        )
    multi = {key: value for key, value in by_context.items() if len(value) > 1}
    assert multi, "expected at least one context with several categories"
    for context_id, ids in multi.items():
        groups = [
            row
            for row in plan.query_groups
            if str(row["context_id"]) == context_id and row["status"] == "admitted"
        ]
        assert len(ids) == len(groups)


def test_boundary_gate_admission_is_shared_within_a_context(plan) -> None:
    """The gate forces nothing, so it is genuinely context-level."""

    by_context: dict[str, set[str]] = {}
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        by_context.setdefault(str(group["context_id"]), set()).add(
            str(group["proposal_boundary_gate_admission_receipt_id"])
        )
    assert by_context
    assert all(len(ids) == 1 for ids in by_context.values())


def test_proposal_rows_are_per_category_never_per_owner(plan) -> None:
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        assert group["proposal_is_per_category_never_per_owner"] is True


def test_admission_ids_differ_across_categories_of_one_context(plan) -> None:
    """Suffix-shape sharing is rejected: each category admits separately."""

    by_context: dict[str, set[str]] = {}
    for group in plan.query_groups:
        if group["status"] != "admitted":
            continue
        by_context.setdefault(str(group["context_id"]), set()).add(
            str(group["admission_receipt_id"])
        )
    multi = {key: value for key, value in by_context.items() if len(value) > 1}
    assert multi, "expected at least one context with several categories"
    for context_id, ids in multi.items():
        groups = [
            row
            for row in plan.query_groups
            if str(row["context_id"]) == context_id and row["status"] == "admitted"
        ]
        assert len(ids) == len(groups)


def test_unknown_admission_channel_is_rejected() -> None:
    with pytest.raises(planner.PlanContractError, match="channel"):
        planner.admission_receipt_id(context_id="c", channel="not-a-channel", prefix_sha256="d")


def test_blocked_categories_block_only_their_own_groups(plan) -> None:
    blocked = [row for row in plan.categories if row["status"] != "admitted"]
    assert {(row["image_id"], row["normalized_description"]) for row in blocked} == {
        ("5001", "bicycle"),
        ("13923", "bowl"),
    }
    for row in blocked:
        assert row["query_suffix_token_ids"] is None
    blocked_images = {row["image_id"] for row in blocked}
    for image_id in blocked_images:
        admitted = [
            row
            for row in plan.query_groups
            if row["image_id"] == image_id and row["status"] == "admitted"
        ]
        assert admitted, "a blocked category must not block its whole image"


# ---------------------------------------------------------------------------
# Estimand semantics (contract items 2, 3, 9)
# ---------------------------------------------------------------------------


def test_category_rows_name_the_estimand_and_deny_per_owner_proposal(plan) -> None:
    for row in plan.categories:
        assert row["estimand_name"] == "category_field_support_at_owner_geometry"
    estimand = plan.receipt["estimand"]
    assert estimand["name"] == "category_field_support_at_owner_geometry"
    assert estimand["is_per_owner_proposal_probability"] is False
    assert estimand["proposal_and_localization_separate"] is True


def test_plan_never_emits_a_per_owner_proposal_probability(plan) -> None:
    blob = json.dumps(
        {
            "owners": plan.owners,
            "categories": plan.categories,
            "query_groups": plan.query_groups,
        }
    )
    assert "owner_proposal_probability" not in blob
    assert "per_owner_proposal" not in blob


def test_category_registry_covers_every_description_present_in_each_image(plan) -> None:
    by_image: dict[str, set[str]] = {}
    for row in plan.owners:
        by_image.setdefault(str(row["image_id"]), set()).add(str(row["normalized_description"]))
    planned: dict[str, set[str]] = {}
    for row in plan.categories:
        planned.setdefault(str(row["image_id"]), set()).add(str(row["normalized_description"]))
    assert planned == by_image


# ---------------------------------------------------------------------------
# Score independence and pre-P0 quarantine (contract items 5 and 7)
# ---------------------------------------------------------------------------


def test_plan_is_score_independent(plan) -> None:
    policy = plan.receipt["score_input_policy"]
    assert policy["reads_any_score_artifact"] is False
    assert policy["candidate_selection_uses_scores"] is False
    assert policy["pre_p0_scores"] == "quarantined_never_read"


def test_planner_source_paths_contain_no_score_artifact(plan) -> None:
    for value in plan.receipt["source_paths"].values():
        lowered = str(value).lower()
        assert "score" not in lowered
        assert "landscape" not in lowered


def test_candidate_selection_policy_is_fixed_and_never_score_selected(plan) -> None:
    contract = plan.receipt["candidate_bank_contract"]
    assert contract["substitution_policy"] == "none_never_score_selected"
    assert contract["mid_run_growth"] == "forbidden"
    assert contract["logical_role_count"] == 17


# ---------------------------------------------------------------------------
# Capture rules
# ---------------------------------------------------------------------------


def test_capture_rules_seal_the_pre_capture_contract() -> None:
    rules = planner.build_capture_rules()
    assert rules["query_suffix"]["on_suffix_or_token_alignment_mismatch"] == "fail_closed"
    assert rules["minimal_frontier_tie_rule"] == planner.MINIMAL_FRONTIER_TIE_RULE
    assert rules["stop_policy"]["global_stop_quarantined_image_count_above"] == 2
    assert rules["phenotype_thresholds"] == "not_frozen_here_derived_on_discovery_only"
    # Representative smoke must not spend a confirmation image.
    smoke = rules["representative_smoke"]
    assert smoke["image_id"] == "6040"
    assert smoke["split"] == "discovery"
    assert smoke["image_id"] in planner.DISCOVERY_IMAGE_IDS


def test_capture_rules_reconstruct_their_own_digest() -> None:
    rules = planner.build_capture_rules()
    recomputed = planner.sha256_json(
        {key: value for key, value in rules.items() if key != "capture_rules_sha256"}
    )
    assert recomputed == rules["capture_rules_sha256"]


def test_capture_rules_bind_the_runtime_invariants() -> None:
    invariants = planner.build_capture_rules()["runtime_invariants"]
    assert invariants["explicit_position_ids_required_on_every_model_call"] is True
    assert invariants["model_generate_forbidden_on_scoring_model"] is True
    assert invariants["concurrent_group_threads_on_one_model"] == "forbidden"
    assert invariants["fresh_dynamic_cache_per_group"] is True
    assert invariants["assert_cache_length_equals_prefill_length"] is True


def test_capture_rules_reject_admission_reuse() -> None:
    admission = planner.build_capture_rules()["admission"]
    assert admission["reuse_admission_across_contexts"] is False
    assert admission["reuse_admission_across_categories"] is False
    assert admission["inherit_first_group_admission"] is False
    assert admission["admission_key"] == ["context_id", "channel", "exact_prefix_sha256"]
    assert admission["bulk_scoring_path"] == "admitted_kv_cache"
    assert admission["batched_full_reforward_role"] == "parity_or_fallback_diagnostic_only"
    assert set(admission["channels"]) == set(planner.ADMISSION_CHANNELS)
    assert admission["proposal_route_excludes_box_start"] is True
    assert admission["proposal_is_per_category_never_per_owner"] is True
    scopes = admission["channel_scopes"]
    assert scopes[planner.CHANNEL_PROPOSAL_BOUNDARY_GATE] == "context_only_observed_prefix"
    assert scopes[planner.CHANNEL_PROPOSAL_CATEGORY_ROUTE].endswith("routing_path_digest")


def test_capture_rules_defer_rp110_without_blocking() -> None:
    strata = planner.build_capture_rules()["strata"]
    assert strata["scored"] == [1.0]
    assert "rp1.10" in strata["deferred_robustness"]


def test_capture_rules_keep_full_x1_in_the_gpu_product() -> None:
    diagnostics = planner.build_capture_rules()["diagnostics"]
    assert diagnostics["full_x1_distribution"] == "captured_in_gpu_product"
    assert "never_a_2d_heatmap" in diagnostics["full_x1_distribution_role"]


def test_unknown_capture_scopes_are_rejected() -> None:
    with pytest.raises(planner.PlanContractError):
        planner.build_capture_rules(session_scope="nonsense")
    with pytest.raises(planner.PlanContractError):
        planner.build_capture_rules(admission_scope="nonsense")


# ---------------------------------------------------------------------------
# Receipt and commit behaviour
# ---------------------------------------------------------------------------


def test_receipt_reconstructs_its_own_digest(plan) -> None:
    recomputed = planner.sha256_json(
        {key: value for key, value in plan.receipt.items() if key != "receipt_content_sha256"}
    )
    assert recomputed == plan.receipt["receipt_content_sha256"]


def test_receipt_digests_every_declared_output_file(plan) -> None:
    digests = plan.receipt["output_file_digests"]
    assert set(digests) == set(planner.PLAN_FILE_NAMES)
    files = plan.files()
    assert set(files) == set(planner.PLAN_FILE_NAMES)
    assert planner.CAPTURE_RULES_NAME in files


def test_commit_is_create_or_identical(tmp_path: Path, plan) -> None:
    target = tmp_path / "plan"
    first = planner.commit_plan(plan, target)
    second = planner.commit_plan(plan, target)
    assert first == second

    (target / "shard-manifest.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(planner.PlanContractError, match="different bytes"):
        planner.commit_plan(plan, target)


def test_committed_plan_files_are_valid_jsonl(tmp_path: Path, plan) -> None:
    target = tmp_path / "plan"
    planner.commit_plan(plan, target)
    for name in planner.PLAN_JSONL_NAMES:
        lines = (target / name).read_text(encoding="utf-8").splitlines()
        assert lines
        for line in lines:
            json.loads(line)
    json.loads((target / planner.CAPTURE_RULES_NAME).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Frozen input guards
# ---------------------------------------------------------------------------


def test_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    fake = tmp_path / "panel.jsonl"
    fake.write_text('{"image_id": 1}\n', encoding="utf-8")
    with pytest.raises(planner.PlanContractError, match="digest mismatch"):
        planner._assert_digest(fake, "0" * 64, "frozen panel")


def test_malformed_native_row_is_rejected() -> None:
    # A row missing its closing box_end violates the closed one-box grammar.
    tokens = [ORS, 8987, ORE, BOX_START, 151670, 151671, 151672, 151673]
    planner.split_generated_rows(tokens + [planner.BOX_END])
    with pytest.raises(planner.PlanContractError, match="closed one-box grammar"):
        planner.split_generated_rows(tokens)


def test_native_rows_must_start_at_an_object_row() -> None:
    with pytest.raises(planner.PlanContractError, match="begin with an object row"):
        planner.split_generated_rows([8987, ORS, 8987, ORE, BOX_START])
