from __future__ import annotations

from src.analysis.post_x1_instance_basin_tomography.trajectory import (
    build_attraction_matrix_rows,
    classify_trajectory,
)


def test_classify_trajectory_stays_target_all_slots() -> None:
    slots = [
        {"slot": "y1", "winner_bucket": "target", "winner_instance_id": "target"},
        {"slot": "x2", "winner_bucket": "target", "winner_instance_id": "target"},
        {"slot": "y2", "winner_bucket": "target", "winner_instance_id": "target"},
    ]

    row = classify_trajectory(slots)

    assert row["trajectory_taxonomy"] == "target"
    assert row["trajectory_bucket"] == "stay_target_all_slots"
    assert row["basin_stay_rate"] == 1.0
    assert row["boundary_extreme_slots"] == []


def test_classify_trajectory_detects_early_same_desc_competitor_switch() -> None:
    slots = [
        {"slot": "y1", "winner_bucket": "competitor_same_desc", "winner_instance_id": 2},
        {"slot": "x2", "winner_bucket": "competitor_same_desc", "winner_instance_id": 2},
        {"slot": "y2", "winner_bucket": "competitor_same_desc", "winner_instance_id": 2},
    ]

    row = classify_trajectory(slots)

    assert row["trajectory_taxonomy"] == "competitor_same_desc"
    assert row["trajectory_bucket"] == "early_switch"
    assert row["switch_slot"] == "y1"
    assert row["switched_to_instance_id"] == 2


def test_classify_trajectory_precedence_marks_invalid_low_coord_mass_first() -> None:
    slots = [
        {
            "slot": "y1",
            "winner_bucket": "competitor_same_desc",
            "winner_instance_id": 2,
            "coord_mass_low_flag": True,
        },
        {"slot": "x2", "winner_bucket": "target", "winner_instance_id": "target", "coord_mass_low_flag": False},
        {"slot": "y2", "winner_bucket": "target", "winner_instance_id": "target", "coord_mass_low_flag": False},
    ]

    row = classify_trajectory(slots)

    assert row["trajectory_taxonomy"] == "invalid_low_coord_mass"
    assert row["trajectory_bucket"] == "invalid_low_coord_mass"
    assert row["switch_slot"] is None


def test_classify_trajectory_preserves_boundary_extremes_as_orthogonal_flags() -> None:
    slots = [
        {
            "slot": "y1",
            "winner_bucket": "target",
            "winner_instance_id": "target",
            "boundary_extreme_flag": True,
        },
        {"slot": "x2", "winner_bucket": "target", "winner_instance_id": "target"},
        {"slot": "y2", "winner_bucket": "target", "winner_instance_id": "target"},
    ]

    row = classify_trajectory(slots)

    assert row["trajectory_taxonomy"] == "target"
    assert row["trajectory_bucket"] == "stay_target_all_slots"
    assert row["boundary_extreme_slots"] == ["y1"]


def test_classify_trajectory_labels_mixed_when_slots_disagree_across_taxa() -> None:
    slots = [
        {"slot": "y1", "winner_bucket": "target", "winner_instance_id": "target"},
        {"slot": "x2", "winner_bucket": "other_desc_object", "winner_instance_id": 9},
        {"slot": "y2", "winner_bucket": "background", "winner_instance_id": None},
    ]

    row = classify_trajectory(slots)

    assert row["trajectory_taxonomy"] == "mixed"
    assert row["trajectory_bucket"] == "mixed_drift"
    assert row["switch_slot"] == "x2"


def test_classify_trajectory_labels_tied_and_ambiguous() -> None:
    tied = classify_trajectory(
        [
            {"slot": "y1", "winner_bucket": "target", "winner_instance_id": "target"},
            {"slot": "x2", "winner_bucket": "tied", "winner_instance_id": None},
            {"slot": "y2", "winner_bucket": "target", "winner_instance_id": "target"},
        ]
    )
    ambiguous = classify_trajectory(
        [
            {"slot": "y1", "winner_bucket": "target", "winner_instance_id": "target"},
            {"slot": "x2", "winner_bucket": "ambiguous", "winner_instance_id": None},
            {"slot": "y2", "winner_bucket": "target", "winner_instance_id": "target"},
        ]
    )

    assert tied["trajectory_taxonomy"] == "tied"
    assert tied["trajectory_bucket"] == "tied"
    assert ambiguous["trajectory_taxonomy"] == "ambiguous"
    assert ambiguous["trajectory_bucket"] == "ambiguous"


def test_attraction_matrix_records_forced_anchor_and_slot_winners() -> None:
    trajectory_rows = [
        {
            "case_id": "c1",
            "checkpoint_role": "fullobj_sorted_pure_ce_ckpt3668",
            "desc": "person",
            "forced_anchor_gt_idx": 1,
            "slot_winners": {"y1": 1, "x2": 1, "y2": 2},
            "trajectory_bucket": "partial_target_then_switch",
        }
    ]

    rows = build_attraction_matrix_rows(trajectory_rows)

    assert rows[0]["case_id"] == "c1"
    assert rows[0]["forced_anchor_gt_idx"] == 1
    assert rows[0]["winner_y1_gt_idx"] == 1
    assert rows[0]["winner_y2_gt_idx"] == 2
    assert rows[0]["diagonal_y1"] is True
    assert rows[0]["diagonal_y2"] is False
