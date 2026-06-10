from __future__ import annotations

from src.analysis.prefix_state_transition_tomography.merge_report import (
    assign_quadrant,
    build_quadrant_rows,
    summarize_quadrants,
)


def test_assign_quadrant_crosses_boundary_and_x1_goodness() -> None:
    assert assign_quadrant(boundary_good=True, x1_good=True) == "boundary_good_x1_good"
    assert assign_quadrant(boundary_good=True, x1_good=False) == "boundary_good_x1_bad"
    assert assign_quadrant(boundary_good=False, x1_good=True) == "boundary_bad_x1_good"
    assert assign_quadrant(boundary_good=False, x1_good=False) == "boundary_bad_x1_bad"


def test_build_quadrant_rows_requires_paired_checkpoint_roles() -> None:
    boundary = [
        _boundary("et_rmp_ce", "residual_favored"),
        _boundary("pure_ce", "eos_favored"),
        {**_boundary("et_rmp_ce", "residual_favored"), "prefix_state_id": "unpaired"},
    ]
    forced = [
        _forced("et_rmp_ce", 1.0),
        _forced("pure_ce", 0.0),
        {**_forced("et_rmp_ce", 1.0), "prefix_state_id": "unpaired"},
    ]

    rows = build_quadrant_rows(boundary, forced)

    assert len(rows) == 2
    by_role = {row["checkpoint_role"]: row for row in rows}
    assert by_role["et_rmp_ce"]["quadrant"] == "boundary_good_x1_good"
    assert by_role["pure_ce"]["quadrant"] == "boundary_bad_x1_bad"


def test_summarize_quadrants_reports_splits_and_paired_delta() -> None:
    rows = [
        {
            "paired_key": "a",
            "checkpoint_role": "et_rmp_ce",
            "split": "train",
            "transition_type": "same_desc_transition",
            "quadrant": "boundary_good_x1_good",
            "forced_x1_residual_coverage": 0.25,
        },
        {
            "paired_key": "a",
            "checkpoint_role": "pure_ce",
            "split": "train",
            "transition_type": "same_desc_transition",
            "quadrant": "boundary_good_x1_good",
            "forced_x1_residual_coverage": 0.75,
        },
    ]

    summary = summarize_quadrants(rows)

    assert summary["quadrant_counts"]["boundary_good_x1_good"] == 2
    assert summary["by_split"]["train|boundary_good_x1_good"] == 2
    assert summary["by_transition_type"]["same_desc_transition|boundary_good_x1_good"] == 2
    assert summary["paired_delta_metrics"]["paired_state_count"] == 1
    assert summary["paired_delta_metrics"]["mean_pure_minus_et_forced_x1_residual_coverage"] == 0.5


def _boundary(role: str, alignment: str) -> dict[str, object]:
    return {
        "checkpoint_role": role,
        "image_id": 1,
        "source_line_idx": 2,
        "prefix_state_id": "pst-1",
        "prefix_condition": "same_desc_prefix_k",
        "prefix_depth": "shallow_1",
        "prefix_order_policy_id": "same_desc_x1_order",
        "probe_desc": "person",
        "split": "train",
        "transition_type": "same_desc_transition",
        "boundary_alignment": alignment,
    }


def _forced(role: str, coverage: float) -> dict[str, object]:
    return {
        "checkpoint_role": role,
        "image_id": 1,
        "source_line_idx": 2,
        "prefix_state_id": "pst-1",
        "prefix_condition": "same_desc_prefix_k",
        "prefix_depth": "shallow_1",
        "prefix_order_policy_id": "same_desc_x1_order",
        "probe_desc": "person",
        "forced_x1_residual_coverage": coverage,
        "emitted_attraction_rate": 0.0,
    }

