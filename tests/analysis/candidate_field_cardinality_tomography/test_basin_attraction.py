from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.basin_attraction import (
    classify_basin_attraction,
)


def test_target_tail_wins_but_greedy_binds_competitor_is_a3a() -> None:
    bucket = classify_basin_attraction(
        {
            "forced_x1_gt_idx": 2,
            "greedy_target_iou": 0.32,
            "greedy_target_iou_rank": 2,
            "best_same_desc_competitor_iou": 0.71,
            "teacher_forced_tail_margin_target_vs_best_competitor": 0.40,
            "target_tail_beats_competitor_tail": True,
        }
    )

    assert bucket == "A3a_decode_basin_failure"


def test_teacher_forced_target_tail_loses_is_a3b() -> None:
    bucket = classify_basin_attraction(
        {
            "forced_x1_gt_idx": 2,
            "greedy_target_iou": 0.80,
            "greedy_target_iou_rank": 1,
            "best_same_desc_competitor_iou": 0.12,
            "teacher_forced_tail_margin_target_vs_best_competitor": -0.15,
            "target_tail_beats_competitor_tail": False,
        }
    )

    assert bucket == "A3b_tail_representation_failure"


def test_missing_tail_score_is_unresolved() -> None:
    bucket = classify_basin_attraction(
        {
            "forced_x1_gt_idx": 2,
            "greedy_target_iou": 0.32,
            "greedy_target_iou_rank": 2,
            "best_same_desc_competitor_iou": 0.71,
        }
    )

    assert bucket == "A3_unresolved"
