"""Focused tests for the sorted supported false-negative native-prefix
reachability prevalence analyzer.

A small synthetic predecessor run root is written under ``tmp_path`` so these
tests are fast, CPU-only, and independent of the frozen twelve-image census.
Production denominators (``114``/``141``) are never hardcoded into a generic
helper here: the fixture below uses four false-negative owners and two native
true-positive owners, and tests that exercise the real production
denominators monkeypatch ``EXPECTED_FN_DENOMINATOR`` / ``EXPECTED_TP_DENOMINATOR``
explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import merge_sorted_owner_accessibility_census_shards as merge  # noqa: E402
from scripts.research import (  # noqa: E402
    analyze_sorted_supported_fn_native_prefix_reachability_prevalence as az,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(az.canonical_json_bytes(row) + b"\n" for row in rows))


def _context(*, context_id, image_id, boundary_index, context_role, prefix_rows=None) -> dict:
    return {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "context_id": context_id,
        "image_id": image_id,
        "boundary_index": boundary_index,
        "context_role": context_role,
        "prefix_admission": {
            "forced_continue_rows_excluded": True,
            "source": "native_greedy_complete_rows_only",
        },
        "prefix_rows": prefix_rows or [],
    }


def _owner_context_row(
    *,
    gt_owner_id,
    context_id,
    image_id,
    boundary_index,
    context_role,
    normalized_description,
    gate_margin,
    category_rank,
    owner_rank,
    best_gt_owner_id,
    passed_state,
    loop_tail=False,
    owner_rank_l=None,
    best_gt_owner_id_l=None,
    margin_l=0.0,
) -> dict:
    """Build one owner-context row with independent U- and L-bound competition.

    ``owner_rank_l``/``best_gt_owner_id_l`` default to mirror the U-bound
    values, so existing fixtures that do not care about bound divergence need
    not specify them; tests exercising divergence pass them explicitly.
    """

    return {
        "schema_version": merge.OWNER_CONTEXT_SCHEMA_VERSION,
        "owner_context_id": f"{gt_owner_id}@{context_id}",
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "image_id": image_id,
        "boundary_index": boundary_index,
        "context_role": context_role,
        "normalized_description": normalized_description,
        "loop_marking": {"loop_tail": loop_tail},
        "frontier_features": {"passed_state": passed_state},
        "category_proposal_channel": {
            "boundary_gate": {"continue_vs_stop_logprob_margin": gate_margin},
            "category_routing_event": (
                None if category_rank is None else {"within_context_rank": category_rank}
            ),
        },
        "owner_competition_u": {
            "rank": owner_rank,
            "best_gt_owner_id": best_gt_owner_id,
            "margin_to_best_owner": 0.0,
            "population_size": 2,
        },
        "owner_competition_l": {
            "rank": owner_rank if owner_rank_l is None else owner_rank_l,
            "best_gt_owner_id": best_gt_owner_id if best_gt_owner_id_l is None else best_gt_owner_id_l,
            "margin_to_best_owner": margin_l,
            "population_size": 2,
        },
    }


def _owner_summary(
    *,
    gt_owner_id,
    image_id,
    normalized_description,
    disposition,
    native_true_positive,
    upper_context_ids,
    lower_context_ids,
    non_loop_context_support=None,
) -> dict:
    return {
        "schema_version": merge.OWNER_SUMMARY_SCHEMA_VERSION,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "normalized_description": normalized_description,
        "disposition": disposition,
        "native_true_positive": native_true_positive,
        "ambiguity_bound_disposition_flip": False,
        "upper_bound_u": {
            "usable_support_context_ids": upper_context_ids,
            "non_loop_context_support": non_loop_context_support or {},
        },
        "lower_bound_l": {"usable_support_context_ids": lower_context_ids},
    }


def _native_sidecar(*, image_id, row_index, pred_row_id, gt_owner_id, status="matched") -> dict:
    return {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "image_id": image_id,
        "row_index": row_index,
        "pred_row_id": pred_row_id,
        "strict_match_status": status,
        "strict_match_gt_owner_id": gt_owner_id,
    }


def build_run_root(tmp_path: Path) -> Path:
    """Four supported false-negative owners and two native true-positive owners.

    Covers: an owner that attains rank one (target) with a mixed before/after
    frontier and a lower bound strictly narrower than the upper bound; an
    owner beaten only by a covered competitor; one beaten only by an
    uncovered competitor; one beaten by both (mixed); a native true-positive
    whose due boundary is supported and fully favorable; and one whose due
    boundary is unsupported despite an otherwise-favorable channel ladder.
    """

    root = tmp_path / "predecessor"
    plan_dir = root / "plan"
    presentation_dir = root / "phases" / "presentation"
    plan_dir.mkdir(parents=True)
    presentation_dir.mkdir(parents=True)

    contexts = [
        _context(context_id="9001:boundary-000", image_id="9001", boundary_index=0, context_role="root"),
        _context(
            context_id="9001:boundary-001",
            image_id="9001",
            boundary_index=1,
            context_role="row_boundary",
            prefix_rows=[{"strict_match_status": "matched", "strict_match_gt_owner_id": "gt:9001:9"}],
        ),
        _context(context_id="9003:boundary-000", image_id="9003", boundary_index=0, context_role="root"),
        _context(
            context_id="9003:boundary-001",
            image_id="9003",
            boundary_index=1,
            context_role="row_boundary",
            prefix_rows=[{"strict_match_status": "matched", "strict_match_gt_owner_id": "gt:9003:8"}],
        ),
        _context(context_id="9002:boundary-000", image_id="9002", boundary_index=0, context_role="root"),
        _context(context_id="9004:boundary-000", image_id="9004", boundary_index=0, context_role="root"),
    ]

    owner_context_rows = [
        _owner_context_row(
            gt_owner_id="gt:9001:0", context_id="9001:boundary-000", image_id="9001",
            boundary_index=0, context_role="root", normalized_description="cat_a",
            gate_margin=2.0, category_rank=1, owner_rank=1, best_gt_owner_id="gt:9001:0",
            passed_state="root_no_frontier",
            # U bound: this owner is the group best (target, rank one). L
            # bound ranks the same context differently (rank two, a
            # different -- uncovered -- best owner), so the sensitivity view
            # must disagree with the primary U view rather than reuse it.
            owner_rank_l=2, best_gt_owner_id_l="gt:9001:2",
        ),
        _owner_context_row(
            gt_owner_id="gt:9001:0", context_id="9001:boundary-001", image_id="9001",
            boundary_index=1, context_role="row_boundary", normalized_description="cat_a",
            gate_margin=-1.0, category_rank=5, owner_rank=2, best_gt_owner_id="gt:9001:1",
            passed_state="passed_by_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9001:1", context_id="9001:boundary-001", image_id="9001",
            boundary_index=1, context_role="row_boundary", normalized_description="cat_b",
            gate_margin=0.5, category_rank=2, owner_rank=2, best_gt_owner_id="gt:9001:9",
            passed_state="ahead_of_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9003:0", context_id="9003:boundary-000", image_id="9003",
            boundary_index=0, context_role="root", normalized_description="cat_a",
            gate_margin=1.0, category_rank=4, owner_rank=3, best_gt_owner_id="gt:9003:5",
            passed_state="root_no_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9003:1", context_id="9003:boundary-000", image_id="9003",
            boundary_index=0, context_role="root", normalized_description="cat_b",
            gate_margin=1.0, category_rank=2, owner_rank=2, best_gt_owner_id="gt:9003:7",
            passed_state="root_no_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9003:1", context_id="9003:boundary-001", image_id="9003",
            boundary_index=1, context_role="row_boundary", normalized_description="cat_b",
            gate_margin=1.0, category_rank=2, owner_rank=2, best_gt_owner_id="gt:9003:8",
            passed_state="ahead_of_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9002:0", context_id="9002:boundary-000", image_id="9002",
            boundary_index=0, context_role="root", normalized_description="cat_a",
            gate_margin=3.0, category_rank=1, owner_rank=1, best_gt_owner_id="gt:9002:0",
            passed_state="root_no_frontier",
        ),
        _owner_context_row(
            gt_owner_id="gt:9004:0", context_id="9004:boundary-000", image_id="9004",
            boundary_index=0, context_role="root", normalized_description="cat_b",
            gate_margin=2.0, category_rank=1, owner_rank=1, best_gt_owner_id="gt:9004:0",
            passed_state="root_no_frontier",
        ),
    ]

    owner_summaries = [
        _owner_summary(
            gt_owner_id="gt:9001:0", image_id="9001", normalized_description="cat_a",
            disposition=merge.DISPOSITION_RESOLVED, native_true_positive=False,
            upper_context_ids=["9001:boundary-000", "9001:boundary-001"],
            lower_context_ids=["9001:boundary-000"],
        ),
        _owner_summary(
            gt_owner_id="gt:9001:1", image_id="9001", normalized_description="cat_b",
            disposition=merge.DISPOSITION_RESOLVED, native_true_positive=False,
            upper_context_ids=["9001:boundary-001"], lower_context_ids=["9001:boundary-001"],
        ),
        _owner_summary(
            gt_owner_id="gt:9003:0", image_id="9003", normalized_description="cat_a",
            disposition=merge.DISPOSITION_RESOLVED, native_true_positive=False,
            upper_context_ids=["9003:boundary-000"], lower_context_ids=["9003:boundary-000"],
        ),
        _owner_summary(
            gt_owner_id="gt:9003:1", image_id="9003", normalized_description="cat_b",
            disposition=merge.DISPOSITION_RESOLVED, native_true_positive=False,
            upper_context_ids=["9003:boundary-000", "9003:boundary-001"],
            lower_context_ids=["9003:boundary-000", "9003:boundary-001"],
        ),
        _owner_summary(
            gt_owner_id="gt:9002:0", image_id="9002", normalized_description="cat_a",
            disposition=merge.DISPOSITION_TP_CONTROL, native_true_positive=True,
            upper_context_ids=[], lower_context_ids=[],
            non_loop_context_support={"9002:boundary-000": True},
        ),
        _owner_summary(
            gt_owner_id="gt:9004:0", image_id="9004", normalized_description="cat_b",
            disposition=merge.DISPOSITION_TP_CONTROL, native_true_positive=True,
            upper_context_ids=[], lower_context_ids=[],
            non_loop_context_support={"9004:boundary-000": False},
        ),
    ]

    native_sidecars = [
        _native_sidecar(image_id="9002", row_index=0, pred_row_id="pred:sorted:greedy:0:9002:0", gt_owner_id="gt:9002:0"),
        _native_sidecar(image_id="9004", row_index=0, pred_row_id="pred:sorted:greedy:0:9004:0", gt_owner_id="gt:9004:0"),
    ]

    _write_jsonl(plan_dir / "context-registry.jsonl", contexts)
    _write_jsonl(plan_dir / "native-sidecar-registry.jsonl", native_sidecars)
    _write_jsonl(presentation_dir / "owner-context-features.jsonl", owner_context_rows)
    _write_jsonl(presentation_dir / "owner-summaries.jsonl", owner_summaries)

    plan_receipt = {
        "schema_version": planner.PLAN_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "output_file_digests": {
            "context-registry.jsonl": az.sha256_bytes((plan_dir / "context-registry.jsonl").read_bytes()),
            "native-sidecar-registry.jsonl": az.sha256_bytes(
                (plan_dir / "native-sidecar-registry.jsonl").read_bytes()
            ),
        },
    }
    plan_receipt["receipt_content_sha256"] = az.sha256_json(plan_receipt)
    (plan_dir / "receipt.json").write_text(json.dumps(plan_receipt), encoding="utf-8")

    merge_receipt = {
        "schema_version": merge.MERGE_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "phase": merge.PHASE_PRESENTATION,
        "output_file_digests": {
            "owner-summaries.jsonl": az.sha256_bytes((presentation_dir / "owner-summaries.jsonl").read_bytes()),
            "owner-context-features.jsonl": az.sha256_bytes(
                (presentation_dir / "owner-context-features.jsonl").read_bytes()
            ),
        },
        "plan": {"receipt_content_sha256": plan_receipt["receipt_content_sha256"]},
        "usable_as_census_conclusion": True,
        "counts": {
            "owner_context_row_count": len(owner_context_rows),
            "owner_summary_row_count": len(owner_summaries),
        },
    }
    merge_receipt["receipt_content_sha256"] = az.sha256_json(merge_receipt)
    (presentation_dir / "merge-receipt.json").write_text(json.dumps(merge_receipt), encoding="utf-8")

    return root


def _reseal_merge_receipt_plan_binding(root: Path, plan_receipt: dict) -> None:
    """Re-bind merge-receipt.json.plan to a mutated plan receipt's new digest.

    Isolates a plan-file mutation test from the (already-covered) plan/merge
    binding check, so it exercises only the failure under test.
    """

    merge_receipt_path = root / az.MERGE_RECEIPT_REL
    merge_receipt = json.loads(merge_receipt_path.read_text())
    merge_receipt["plan"]["receipt_content_sha256"] = plan_receipt["receipt_content_sha256"]
    merge_receipt["receipt_content_sha256"] = az.sha256_json(
        {k: v for k, v in merge_receipt.items() if k != "receipt_content_sha256"}
    )
    merge_receipt_path.write_text(json.dumps(merge_receipt), encoding="utf-8")


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    return build_run_root(tmp_path)


@pytest.fixture(autouse=True)
def _small_denominators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(az, "EXPECTED_FN_DENOMINATOR", 4)
    monkeypatch.setattr(az, "EXPECTED_TP_DENOMINATOR", 2)


def _records_by_owner(records: list[dict]) -> dict[str, dict]:
    return {row["gt_owner_id"]: row for row in records}


def test_full_analysis_reproduces_expected_channel_ladder(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    report = result["report"]
    owner_records = result["owner_records"]

    assert report["false_negative_cohort"]["denominator"] == 4
    assert report["native_true_positive_reference"]["denominator"] == 2

    fn = _records_by_owner([r for r in owner_records if r["cohort"] == az.FN_COHORT])
    tp = _records_by_owner([r for r in owner_records if r["cohort"] == az.TP_COHORT])

    o1 = fn["gt:9001:0"]
    assert o1["upper_bound_u"]["any_owner_rank_one"] is True
    assert o1["upper_bound_u"]["any_gate_open"] is True
    assert o1["upper_bound_u"]["any_before_or_at_frontier"] is True
    assert o1["upper_bound_u"]["any_after_frontier"] is True
    assert o1["upper_bound_u"]["any_favorable_rank1_before_or_at_frontier"] is True
    assert o1["never_owner_rank_one"] is False
    # The lower bound keeps only the root context, so it never sees the
    # after-frontier row and must disagree with the upper bound here.
    assert o1["lower_bound_l"]["any_after_frontier"] is False
    assert o1["lower_bound_l"]["support_context_count"] == 1
    # At that same root context, owner_competition_l ranks this owner second
    # behind a different (uncovered) best owner: the L view must read its own
    # competition field rather than silently reusing the U bound's rank one
    # and target status.
    assert o1["lower_bound_l"]["any_owner_rank_one"] is False
    assert o1["lower_bound_l"]["any_uncovered_other"] is True
    root_channel_l = o1["lower_bound_l"]["contexts"][0]
    assert root_channel_l["owner_rank_within_group"] == 2
    assert root_channel_l["best_gt_owner_id"] == "gt:9001:2"
    assert root_channel_l["competitor_status"] == az.COMPETITOR_UNCOVERED_OTHER
    root_channel_u = o1["upper_bound_u"]["contexts"][0]
    assert root_channel_u["owner_rank_within_group"] == 1
    assert root_channel_u["best_gt_owner_id"] == "gt:9001:0"
    assert root_channel_u["competitor_status"] == az.COMPETITOR_TARGET

    o2 = fn["gt:9001:1"]
    assert o2["never_owner_rank_one"] is True
    assert o2["only_covered_other_among_never_rank_one"] is True
    assert o2["only_uncovered_other_among_never_rank_one"] is False

    o3 = fn["gt:9003:0"]
    assert o3["never_owner_rank_one"] is True
    assert o3["only_uncovered_other_among_never_rank_one"] is True
    assert o3["only_covered_other_among_never_rank_one"] is False

    o4 = fn["gt:9003:1"]
    assert o4["never_owner_rank_one"] is True
    assert o4["upper_bound_u"]["any_covered_other"] is True
    assert o4["upper_bound_u"]["any_uncovered_other"] is True
    assert o4["only_covered_other_among_never_rank_one"] is False
    assert o4["only_uncovered_other_among_never_rank_one"] is False

    t1 = tp["gt:9002:0"]
    assert t1["due_context_support"] is True
    assert t1["channel"]["gate_open"] is True
    assert t1["channel"]["owner_rank_one"] is True
    assert t1["favorable_rank1_before_or_at_frontier"] is True

    t2 = tp["gt:9004:0"]
    # T2's channel is otherwise fully favorable, but its due boundary is
    # explicitly unsupported, so the joint favorable flag must be False
    # purely from the support gate.
    assert t2["channel"]["gate_open"] is True
    assert t2["channel"]["category_rank_one"] is True
    assert t2["channel"]["owner_rank_one"] is True
    assert t2["due_context_support"] is False
    assert t2["favorable_rank1_before_or_at_frontier"] is False


def test_native_tp_headline_reports_full_ladder_and_strata(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    tp_headline = result["report"]["native_true_positive_reference"]["headline"]
    assert tp_headline["denominator"] == 2

    metrics = tp_headline["metrics"]
    assert set(az.TP_LADDER_KEYS) <= set(metrics)
    # Both due boundaries have category rank one (hence also top three).
    assert metrics["category_rank_top3"]["successes"] == 2
    assert metrics["category_rank_top3"]["total"] == 2
    assert metrics["category_rank_one"]["successes"] == 2
    # Only T1's due boundary is itself calibrated-supported.
    assert metrics["due_context_support"]["successes"] == 1
    assert metrics["due_context_support"]["total"] == 2

    for stratified in (tp_headline["per_image"], tp_headline["per_category"]):
        for stratum in stratified.values():
            assert set(az.TP_LADDER_KEYS) <= set(stratum)
            assert "owner_count" in stratum

    assert tp_headline["per_image"]["9002"]["owner_count"] == 1
    assert tp_headline["per_image"]["9002"]["due_context_support"] == 1
    assert tp_headline["per_image"]["9004"]["owner_count"] == 1
    assert tp_headline["per_image"]["9004"]["due_context_support"] == 0

    assert tp_headline["per_category"]["cat_a"]["due_context_support"] == 1
    assert tp_headline["per_category"]["cat_b"]["due_context_support"] == 0

    # Both due boundaries in this fixture are root contexts.
    assert metrics["due_context_role_root"]["successes"] == 2
    assert metrics["due_context_role_row_boundary"]["successes"] == 0
    assert metrics["due_context_role_terminal"]["successes"] == 0
    assert tp_headline["per_image"]["9002"]["due_context_role_root"] == 1

    category_hist = tp_headline["category_routing_rank_histogram"]
    owner_hist = tp_headline["owner_rank_histogram"]
    for histogram in (category_hist, owner_hist):
        assert histogram["unit"] == az.TP_HISTOGRAM_UNIT
        assert histogram["weighting"] == az.TP_HISTOGRAM_WEIGHTING
        assert histogram["total"] == sum(histogram["counts"].values())
    # Both due boundaries have category rank one and owner rank one.
    assert category_hist["counts"] == {1: 2}
    assert category_hist["total"] == 2
    assert owner_hist["counts"] == {1: 2}
    assert owner_hist["total"] == 2


def test_headline_wilson_intervals_match_manual_counts(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    headline = result["report"]["false_negative_cohort"]["headline"]["upper_bound_u"]["metrics"]
    assert headline["any_owner_rank_one"]["successes"] == 1
    assert headline["any_owner_rank_one"]["total"] == 4
    assert headline["any_owner_rank_one"]["proportion"] == pytest.approx(0.25)
    assert 0.0 <= headline["any_owner_rank_one"]["lower"] < headline["any_owner_rank_one"]["proportion"]
    assert headline["any_owner_rank_one"]["proportion"] < headline["any_owner_rank_one"]["upper"] <= 1.0


def test_headline_per_image_and_per_category_cover_the_full_ladder(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    upper = result["report"]["false_negative_cohort"]["headline"]["upper_bound_u"]
    lower = result["report"]["false_negative_cohort"]["headline"]["lower_bound_l"]

    for stratified in (upper["per_image"], upper["per_category"], lower["per_image"], lower["per_category"]):
        for stratum in stratified.values():
            assert set(az.LADDER_KEYS) <= set(stratum)
            assert "owner_count" in stratum

    # image 9001 holds O1 (attains owner rank one) and O2 (never does); image
    # 9003 holds O3 and O4 (neither ever does).
    assert upper["per_image"]["9001"]["owner_count"] == 2
    assert upper["per_image"]["9001"]["any_owner_rank_one"] == 1
    assert upper["per_image"]["9003"]["owner_count"] == 2
    assert upper["per_image"]["9003"]["any_owner_rank_one"] == 0

    # cat_a holds O1 and O3; cat_b holds O2 and O4.
    assert upper["per_category"]["cat_a"]["owner_count"] == 2
    assert upper["per_category"]["cat_a"]["any_owner_rank_one"] == 1
    assert upper["per_category"]["cat_b"]["owner_count"] == 2
    assert upper["per_category"]["cat_b"]["any_owner_rank_one"] == 0

    # The lower bound narrows O1 down to its root-only context, where
    # owner_competition_l ranks it second, so the L-stratified count must
    # differ from the U-stratified one for the same image.
    assert lower["per_image"]["9001"]["any_owner_rank_one"] == 0


def test_headline_reports_owner_level_wilson_intervals_for_competitor_status(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    upper = result["report"]["false_negative_cohort"]["headline"]["upper_bound_u"]
    metrics = upper["metrics"]

    # any_target/any_covered_other/any_uncovered_other are now part of the
    # published owner-level ladder (Wilson interval + strata), not only raw
    # counts on the never-owner-rank-one subset.
    assert set(az.LADDER_KEYS) >= {"any_target", "any_covered_other", "any_uncovered_other"}
    for key in ("any_target", "any_covered_other", "any_uncovered_other"):
        assert metrics[key]["total"] == 4
    # O1 attains rank one (target) but also loses to an uncovered competitor
    # at its second context; O2/O4 have a covered competitor at some
    # context; O1/O3/O4 have an uncovered competitor at some context.
    assert metrics["any_target"]["successes"] == 1
    assert metrics["any_covered_other"]["successes"] == 2
    assert metrics["any_uncovered_other"]["successes"] == 3

    for stratified in (upper["per_image"], upper["per_category"]):
        for stratum in stratified.values():
            assert {"any_target", "any_covered_other", "any_uncovered_other"} <= set(stratum)


def test_competitor_summary_all_supported_fn_owners_block(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    competitor = result["report"]["false_negative_cohort"]["headline"][
        "competitor_status_among_never_owner_rank_one"
    ]
    all_owners = competitor["all_supported_fn_owners"]
    assert all_owners["owner_count"] == 4
    assert all_owners["metrics"]["any_target"]["successes"] == 1
    assert all_owners["metrics"]["any_covered_other"]["successes"] == 2
    assert all_owners["metrics"]["any_uncovered_other"]["successes"] == 3
    for stratified in (all_owners["per_image"], all_owners["per_category"]):
        for stratum in stratified.values():
            assert {"owner_count", "any_target", "any_covered_other", "any_uncovered_other"} <= set(
                stratum
            )
    assert all_owners["per_image"]["9001"]["owner_count"] == 2
    assert all_owners["per_image"]["9001"]["any_target"] == 1
    # The original never-owner-rank-one block must be preserved unchanged
    # alongside the new all-owners block.
    assert competitor["never_owner_rank_one_count"] == 3
    assert set(competitor["metrics"]) == {
        "any_covered_other",
        "any_uncovered_other",
        "only_covered_other",
        "only_uncovered_other",
    }


def test_competitor_summary_has_wilson_metrics_and_strata_over_never_rank_one(
    run_root: Path,
) -> None:
    result = az.run_analysis(run_root)
    competitor = result["report"]["false_negative_cohort"]["headline"][
        "competitor_status_among_never_owner_rank_one"
    ]
    assert competitor["never_owner_rank_one_count"] == 3

    # Plain counts are preserved for continuity...
    assert competitor["any_covered_other"] == 2
    assert competitor["any_uncovered_other"] == 2
    assert competitor["only_covered_other"] == 1
    assert competitor["only_uncovered_other"] == 1
    # ...alongside owner-level Wilson intervals over the same subset.
    metrics = competitor["metrics"]
    for key in ("any_covered_other", "any_uncovered_other", "only_covered_other", "only_uncovered_other"):
        assert metrics[key]["total"] == 3
    assert metrics["any_covered_other"]["successes"] == 2
    assert metrics["only_covered_other"]["successes"] == 1

    for stratified in (competitor["per_image"], competitor["per_category"]):
        for stratum in stratified.values():
            assert "owner_count" in stratum
            assert {"any_covered_other", "any_uncovered_other", "only_covered_other", "only_uncovered_other"} <= set(
                stratum
            )
    assert competitor["per_image"]["9003"]["owner_count"] == 2


def test_fn_rank_histograms_are_labelled_context_row_weighted(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    upper = result["report"]["false_negative_cohort"]["headline"]["upper_bound_u"]
    for histogram in (upper["category_routing_rank_histogram"], upper["owner_rank_histogram"]):
        assert histogram["unit"] == az.FN_HISTOGRAM_UNIT
        assert histogram["weighting"] == az.FN_HISTOGRAM_WEIGHTING
        assert histogram["total"] == sum(histogram["counts"].values())
    # O1 alone contributes two usable-support contexts under U, so the
    # owner-rank histogram total must exceed the four-owner denominator: it
    # is context-row-weighted, never owner-deduplicated.
    total_support_contexts = sum(
        r["upper_bound_u"]["support_context_count"]
        for r in result["owner_records"]
        if r["cohort"] == az.FN_COHORT
    )
    assert upper["owner_rank_histogram"]["total"] == total_support_contexts
    assert total_support_contexts > 4


def test_obstruction_summary_reports_owner_ids_per_missing_channel(run_root: Path) -> None:
    result = az.run_analysis(run_root)
    obstruction = result["report"]["false_negative_cohort"]["headline"]["obstruction_summary"]
    assert set(az.OBSTRUCTION_FLAG_NAMES) <= set(obstruction)
    # O1 attains owner rank one at a fully favorable context, so it must not
    # be tagged by any never-* obstruction flag.
    assert "gt:9001:0" not in obstruction["never_owner_rank_one"]["owner_ids"]
    # O2 and O3 never attain owner rank one and never reach category rank
    # one, so both must appear in both buckets.
    assert obstruction["never_owner_rank_one"]["owner_ids"] == sorted(
        ["gt:9001:1", "gt:9003:0", "gt:9003:1"]
    )
    assert set(obstruction["never_category_rank_one"]["owner_ids"]) >= {"gt:9001:1", "gt:9003:0", "gt:9003:1"}
    assert obstruction["never_owner_rank_one"]["count"] == len(obstruction["never_owner_rank_one"]["owner_ids"])
    assert obstruction["role"].startswith("nonexclusive_descriptive")


def test_obstruction_flags_are_nonexclusive_and_can_coexist() -> None:
    """Flags overlap freely and are never forced into one exclusive label."""

    fully_obstructed = {
        "any_gate_open": False,
        "any_category_rank_top3": False,
        "any_category_rank_one": False,
        "any_owner_rank_one": False,
        "any_before_or_at_frontier": True,
        "any_after_frontier": False,
        "any_favorable_top3_any_frontier": False,
        "any_favorable_top3_before_or_at_frontier": False,
        "any_favorable_rank1_any_frontier": False,
        "any_favorable_rank1_before_or_at_frontier": False,
    }
    flags = az._obstruction_flags(fully_obstructed)
    # Every never-* flag fires at once: overlap, not exclusivity.
    assert flags["never_gate_open"] is True
    assert flags["never_category_rank_top3"] is True
    assert flags["never_category_rank_one"] is True
    assert flags["never_owner_rank_one"] is True

    # top3 and owner-rank-one both occur (somewhere), but never at the same
    # context, so the joint favorable surface never fires -- an asynchrony,
    # not a "never" obstruction.
    asynchronous = {
        "any_gate_open": True,
        "any_category_rank_top3": True,
        "any_category_rank_one": False,
        "any_owner_rank_one": True,
        "any_before_or_at_frontier": True,
        "any_after_frontier": False,
        "any_favorable_top3_any_frontier": False,
        "any_favorable_top3_before_or_at_frontier": False,
        "any_favorable_rank1_any_frontier": False,
        "any_favorable_rank1_before_or_at_frontier": False,
    }
    flags = az._obstruction_flags(asynchronous)
    assert flags["never_category_rank_top3"] is False
    assert flags["never_owner_rank_one"] is False
    assert flags["top3_and_owner_rank1_occur_but_never_at_the_same_context"] is True
    assert flags["rank1_and_owner_rank1_occur_but_never_at_the_same_context"] is False

    # A favorable surface that only ever occurs after the frontier has
    # passed is tagged distinctly from one that never occurs at all.
    late_only_favorable = {
        "any_gate_open": True,
        "any_category_rank_top3": True,
        "any_category_rank_one": True,
        "any_owner_rank_one": True,
        "any_before_or_at_frontier": False,
        "any_after_frontier": True,
        "any_favorable_top3_any_frontier": True,
        "any_favorable_top3_before_or_at_frontier": False,
        "any_favorable_rank1_any_frontier": True,
        "any_favorable_rank1_before_or_at_frontier": False,
    }
    flags = az._obstruction_flags(late_only_favorable)
    assert flags["support_only_after_frontier"] is True
    assert flags["favorable_top3_only_after_frontier"] is True
    assert flags["favorable_rank1_only_after_frontier"] is True
    assert flags["never_owner_rank_one"] is False


def test_context_count_imbalance_cannot_change_owner_counts() -> None:
    """An owner with many support contexts must weigh no more than one row."""

    def _channel(gate_open: bool) -> dict:
        return {
            "context_id": "9010:boundary-000",
            "boundary_index": 0,
            "context_role": "row_boundary",
            "loop_tail": False,
            "gate_continue_vs_stop_logprob_margin": 1.0 if gate_open else -1.0,
            "before_or_at_frontier": True,
            "passed_state": "ahead_of_frontier",
            "gate_open": gate_open,
            "category_routing_within_context_rank": None,
            "category_rank_one": False,
            "category_rank_top3": False,
            "owner_rank_within_group": None,
            "owner_rank_one": False,
            "owner_competition_margin_to_best_owner": None,
            "owner_competition_population_size": None,
            "best_gt_owner_id": None,
            "favorable_top3_any_frontier": False,
            "favorable_top3_before_or_at_frontier": False,
            "favorable_rank1_any_frontier": False,
            "favorable_rank1_before_or_at_frontier": False,
            "competitor_status": az.COMPETITOR_UNCOVERED_OTHER,
        }

    heavy_owner_channels = [_channel(gate_open=(index == 0)) for index in range(55)]
    light_owner_channels = [_channel(gate_open=True)]

    heavy = az._bound_rollup(heavy_owner_channels)
    light = az._bound_rollup(light_owner_channels)
    assert heavy["any_gate_open"] is True
    assert light["any_gate_open"] is True
    fn_records = [
        {"gt_owner_id": "gt:9010:0", "image_id": "9010", "normalized_description": "cat_x", "upper_bound_u": heavy},
        {"gt_owner_id": "gt:9011:0", "image_id": "9011", "normalized_description": "cat_x", "upper_bound_u": light},
    ]
    for record in fn_records:
        record["lower_bound_l"] = record["upper_bound_u"]
        record["never_owner_rank_one"] = True
        record["only_covered_other_among_never_rank_one"] = False
        record["only_uncovered_other_among_never_rank_one"] = True
        record["obstruction_flags"] = az._obstruction_flags(record["upper_bound_u"])
    headline = az.aggregate_fn_headline(fn_records)
    assert headline["upper_bound_u"]["metrics"]["any_gate_open"]["successes"] == 2
    assert headline["upper_bound_u"]["metrics"]["any_gate_open"]["total"] == 2


def test_conjunction_requires_the_same_context() -> None:
    """A favorable surface may not be assembled from two different contexts."""

    gate_only = {
        "context_role": "root", "before_or_at_frontier": True, "gate_open": True,
        "category_rank_one": False, "category_rank_top3": False, "owner_rank_one": False,
        "favorable_top3_any_frontier": False, "favorable_top3_before_or_at_frontier": False,
        "favorable_rank1_any_frontier": False, "favorable_rank1_before_or_at_frontier": False,
        "competitor_status": az.COMPETITOR_UNCOVERED_OTHER,
    }
    rank_only = {
        "context_role": "row_boundary", "before_or_at_frontier": True, "gate_open": False,
        "category_rank_one": True, "category_rank_top3": True, "owner_rank_one": True,
        "favorable_top3_any_frontier": False, "favorable_top3_before_or_at_frontier": False,
        "favorable_rank1_any_frontier": False, "favorable_rank1_before_or_at_frontier": False,
        "competitor_status": az.COMPETITOR_TARGET,
    }
    rollup = az._bound_rollup([gate_only, rank_only])
    assert rollup["any_gate_open"] is True
    assert rollup["any_category_rank_one"] is True
    assert rollup["any_owner_rank_one"] is True
    # None of the individual contexts satisfied every channel at once.
    assert rollup["any_favorable_rank1_any_frontier"] is False
    assert rollup["any_favorable_top3_any_frontier"] is False


@pytest.mark.parametrize(
    "gate_margin,category_rank,owner_rank,best_gt_owner_id,prefix_owner,expected_competitor,expected_gate_open",
    [
        (0.001, 1, 1, "gt:x:0", None, az.COMPETITOR_TARGET, True),
        (0.0, 1, 1, "gt:x:0", None, az.COMPETITOR_TARGET, False),
        (1.0, 2, 2, "gt:x:9", "gt:x:9", az.COMPETITOR_COVERED_OTHER, True),
        (1.0, 2, 2, "gt:x:9", None, az.COMPETITOR_UNCOVERED_OTHER, True),
    ],
)
def test_extract_context_channel_distinguishes_ranks_and_competitor_coverage(
    gate_margin, category_rank, owner_rank, best_gt_owner_id, prefix_owner, expected_competitor, expected_gate_open
) -> None:
    owner_context_row = _owner_context_row(
        gt_owner_id="gt:x:0", context_id="x:boundary-000", image_id="x", boundary_index=0,
        context_role="root", normalized_description="cat", gate_margin=gate_margin,
        category_rank=category_rank, owner_rank=owner_rank, best_gt_owner_id=best_gt_owner_id,
        passed_state="root_no_frontier",
    )
    prefix_rows = (
        [{"strict_match_status": "matched", "strict_match_gt_owner_id": prefix_owner}]
        if prefix_owner
        else []
    )
    context = _context(context_id="x:boundary-000", image_id="x", boundary_index=0, context_role="root", prefix_rows=prefix_rows)
    channel = az.extract_context_channel(owner_context_row, context, bound="u")
    assert channel["gate_open"] is expected_gate_open
    assert channel["competitor_status"] == expected_competitor
    # Category rank and owner rank must never be conflated with each other.
    assert channel["category_routing_within_context_rank"] == category_rank
    assert channel["owner_rank_within_group"] == owner_rank


def test_due_boundary_join_uses_only_native_sidecar_registry(run_root: Path) -> None:
    inputs = az.load_inputs(run_root)
    due_map = az.build_due_context_map(inputs)
    assert due_map["gt:9002:0"]["due_context_id"] == "9002:boundary-000"
    assert due_map["gt:9004:0"]["due_context_id"] == "9004:boundary-000"


def test_due_boundary_join_fails_closed_on_duplicate_strict_match(tmp_path: Path) -> None:
    root = build_run_root(tmp_path)
    sidecars = az._read_jsonl(root / az.NATIVE_SIDECAR_REGISTRY_REL, "native-sidecar-registry.jsonl")
    sidecars.append(
        _native_sidecar(image_id="9002", row_index=1, pred_row_id="pred:sorted:greedy:0:9002:1", gt_owner_id="gt:9002:0")
    )
    _write_jsonl(root / az.NATIVE_SIDECAR_REGISTRY_REL, sidecars)
    # Re-seal the plan receipt digest for the mutated sidecar registry so this
    # test isolates the duplicate-match failure, not a digest mismatch.
    plan_receipt = json.loads((root / az.PLAN_RECEIPT_REL).read_text())
    plan_receipt["output_file_digests"]["native-sidecar-registry.jsonl"] = az.sha256_bytes(
        (root / az.NATIVE_SIDECAR_REGISTRY_REL).read_bytes()
    )
    plan_receipt["receipt_content_sha256"] = az.sha256_json(
        {k: v for k, v in plan_receipt.items() if k != "receipt_content_sha256"}
    )
    (root / az.PLAN_RECEIPT_REL).write_text(json.dumps(plan_receipt), encoding="utf-8")

    inputs = az.load_inputs(root)
    with pytest.raises(az.AnalysisContractError, match="strict-matched native sidecar rows"):
        az.build_due_context_map(inputs)


def test_wrong_denominator_fails_closed_without_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Override the autouse small-denominator patch back to the frozen
    # production value: the four-owner fixture must not satisfy it.
    monkeypatch.setattr(az, "EXPECTED_FN_DENOMINATOR", 114)
    root = build_run_root(tmp_path)
    with pytest.raises(az.AnalysisContractError, match="114"):
        az.run_analysis(root)


def test_context_role_outside_native_universe_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(az, "EXPECTED_FN_DENOMINATOR", 4)
    monkeypatch.setattr(az, "EXPECTED_TP_DENOMINATOR", 2)
    root = build_run_root(tmp_path)
    contexts = az._read_jsonl(root / az.CONTEXT_REGISTRY_REL, "context-registry.jsonl")
    contexts[0]["context_role"] = "free_decode"
    _write_jsonl(root / az.CONTEXT_REGISTRY_REL, contexts)
    plan_receipt = json.loads((root / az.PLAN_RECEIPT_REL).read_text())
    plan_receipt["output_file_digests"]["context-registry.jsonl"] = az.sha256_bytes(
        (root / az.CONTEXT_REGISTRY_REL).read_bytes()
    )
    plan_receipt["receipt_content_sha256"] = az.sha256_json(
        {k: v for k, v in plan_receipt.items() if k != "receipt_content_sha256"}
    )
    (root / az.PLAN_RECEIPT_REL).write_text(json.dumps(plan_receipt), encoding="utf-8")
    _reseal_merge_receipt_plan_binding(root, plan_receipt)
    with pytest.raises(az.AnalysisContractError, match="outside the native"):
        az.run_analysis(root)


def test_forced_continue_rows_not_excluded_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(az, "EXPECTED_FN_DENOMINATOR", 4)
    monkeypatch.setattr(az, "EXPECTED_TP_DENOMINATOR", 2)
    root = build_run_root(tmp_path)
    contexts = az._read_jsonl(root / az.CONTEXT_REGISTRY_REL, "context-registry.jsonl")
    contexts[0]["prefix_admission"]["forced_continue_rows_excluded"] = False
    _write_jsonl(root / az.CONTEXT_REGISTRY_REL, contexts)
    plan_receipt = json.loads((root / az.PLAN_RECEIPT_REL).read_text())
    plan_receipt["output_file_digests"]["context-registry.jsonl"] = az.sha256_bytes(
        (root / az.CONTEXT_REGISTRY_REL).read_bytes()
    )
    plan_receipt["receipt_content_sha256"] = az.sha256_json(
        {k: v for k, v in plan_receipt.items() if k != "receipt_content_sha256"}
    )
    (root / az.PLAN_RECEIPT_REL).write_text(json.dumps(plan_receipt), encoding="utf-8")
    _reseal_merge_receipt_plan_binding(root, plan_receipt)
    with pytest.raises(az.AnalysisContractError, match="forced-continue"):
        az.run_analysis(root)


def test_input_digest_tamper_fails_closed(run_root: Path) -> None:
    context_path = run_root / az.CONTEXT_REGISTRY_REL
    context_path.write_bytes(context_path.read_bytes() + b'{"tampered": true}\n')
    with pytest.raises(az.AnalysisContractError, match="digest"):
        az.run_analysis(run_root)


def test_root_context_has_null_frontier_before_or_at_state(run_root: Path) -> None:
    inputs = az.load_inputs(run_root)
    root_row = inputs.owner_context_by_id["gt:9001:0@9001:boundary-000"]
    context = inputs.contexts_by_id["9001:boundary-000"]
    channel = az.extract_context_channel(root_row, context, bound="u")
    assert channel["passed_state"] == "root_no_frontier"
    assert channel["before_or_at_frontier"] is True


def test_extract_context_channel_reads_its_own_bound_never_the_other(run_root: Path) -> None:
    """L must read owner_competition_l, never silently reuse U's rank/best owner."""

    inputs = az.load_inputs(run_root)
    root_row = inputs.owner_context_by_id["gt:9001:0@9001:boundary-000"]
    context = inputs.contexts_by_id["9001:boundary-000"]

    channel_u = az.extract_context_channel(root_row, context, bound="u")
    channel_l = az.extract_context_channel(root_row, context, bound="l")

    assert channel_u["bound"] == "u"
    assert channel_u["owner_rank_within_group"] == 1
    assert channel_u["best_gt_owner_id"] == "gt:9001:0"
    assert channel_u["competitor_status"] == az.COMPETITOR_TARGET

    assert channel_l["bound"] == "l"
    assert channel_l["owner_rank_within_group"] == 2
    assert channel_l["best_gt_owner_id"] == "gt:9001:2"
    assert channel_l["competitor_status"] == az.COMPETITOR_UNCOVERED_OTHER

    with pytest.raises(az.AnalysisContractError, match="unknown bound"):
        az.extract_context_channel(root_row, context, bound="x")


def test_loop_tail_usable_support_context_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(az, "EXPECTED_FN_DENOMINATOR", 4)
    monkeypatch.setattr(az, "EXPECTED_TP_DENOMINATOR", 2)
    root = build_run_root(tmp_path)
    owner_context_rows = az._read_jsonl(
        root / az.OWNER_CONTEXT_FEATURES_REL, "owner-context-features.jsonl"
    )
    for row in owner_context_rows:
        if row["owner_context_id"] == "gt:9001:0@9001:boundary-000":
            row["loop_marking"]["loop_tail"] = True
    _write_jsonl(root / az.OWNER_CONTEXT_FEATURES_REL, owner_context_rows)
    merge_receipt = json.loads((root / az.MERGE_RECEIPT_REL).read_text())
    merge_receipt["output_file_digests"]["owner-context-features.jsonl"] = az.sha256_bytes(
        (root / az.OWNER_CONTEXT_FEATURES_REL).read_bytes()
    )
    merge_receipt["receipt_content_sha256"] = az.sha256_json(
        {k: v for k, v in merge_receipt.items() if k != "receipt_content_sha256"}
    )
    (root / az.MERGE_RECEIPT_REL).write_text(json.dumps(merge_receipt), encoding="utf-8")
    with pytest.raises(az.AnalysisContractError, match="loop_tail"):
        az.run_analysis(root)


def test_wilson_interval_edge_cases() -> None:
    empty = az.wilson_interval(0, 0)
    assert empty["proportion"] is None and empty["lower"] is None
    full = az.wilson_interval(5, 5)
    assert full["proportion"] == 1.0
    assert full["upper"] == pytest.approx(1.0)
    assert 0.0 < full["lower"] < 1.0
    none = az.wilson_interval(0, 5)
    assert none["proportion"] == 0.0
    assert none["lower"] == 0.0
    assert none["upper"] > 0.0


def test_main_writes_expected_output_files(run_root: Path, tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    exit_code = az.main(["--run-root", str(run_root), "--output-root", str(output_root)])
    assert exit_code == 0
    analysis_dir = output_root / "analysis"
    for name in (az.REPORT_JSON_NAME, az.REPORT_MD_NAME, az.OWNER_RECORDS_NAME, az.RECEIPT_NAME):
        assert (analysis_dir / name).is_file()
    receipt = json.loads((analysis_dir / az.RECEIPT_NAME).read_text())
    reconstructed = az.sha256_json(
        {k: v for k, v in receipt.items() if k != "receipt_content_sha256"}
    )
    assert reconstructed == receipt["receipt_content_sha256"]
    report_bytes = (analysis_dir / az.REPORT_JSON_NAME).read_bytes()
    assert az.sha256_bytes(report_bytes) == receipt["report_json_sha256"]
    report_md_bytes = (analysis_dir / az.REPORT_MD_NAME).read_bytes()
    assert az.sha256_bytes(report_md_bytes) == receipt["report_md_sha256"]
