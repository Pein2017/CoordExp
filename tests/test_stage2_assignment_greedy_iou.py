from __future__ import annotations

import pytest

from src.trainers.rollout_matching.contracts import GTObject
from src.trainers.rollout_matching.matching import greedy_match_iou
from src.training.stage2.assignment import (
    AssignmentObject,
    GreedyIoUAssignment,
)


def test_greedy_iou_assignment_prefers_iou_then_prediction_then_gt_index() -> None:
    strategy = GreedyIoUAssignment(iou_threshold=0.5)
    predictions = (
        AssignmentObject(object_id="pred-a", bbox=(0.0, 0.0, 10.0, 10.0)),
        AssignmentObject(object_id="pred-b", bbox=(0.0, 0.0, 10.0, 10.0)),
        AssignmentObject(object_id="pred-c", bbox=(100.0, 100.0, 110.0, 110.0)),
    )
    ground_truth = (
        AssignmentObject(object_id="gt-a", bbox=(0.0, 0.0, 10.0, 10.0)),
        AssignmentObject(object_id="gt-b", bbox=(20.0, 20.0, 30.0, 30.0)),
    )

    result = strategy.assign(predictions=predictions, ground_truth=ground_truth)

    assert [(pair.prediction_index, pair.ground_truth_index) for pair in result.pairs] == [
        (0, 0)
    ]
    assert result.pairs[0].prediction_id == "pred-a"
    assert result.pairs[0].ground_truth_id == "gt-a"
    assert result.pairs[0].reason == "matched"
    assert result.pairs[0].iou == pytest.approx(1.0)

    assert [(item.index, item.object_id, item.reason) for item in result.unmatched_predictions] == [
        (1, "pred-b", "assignment_conflict"),
        (2, "pred-c", "below_threshold"),
    ]
    assert result.unmatched_predictions[0].best_iou == pytest.approx(1.0)
    assert result.unmatched_predictions[1].best_iou == pytest.approx(0.0)

    assert [(item.index, item.object_id, item.reason) for item in result.unmatched_ground_truth] == [
        (1, "gt-b", "below_threshold")
    ]
    assert result.unmatched_ground_truth[0].best_iou == pytest.approx(0.0)

    assert result.metadata["assignment_strategy"] == "greedy_iou"
    assert result.metadata["iou_threshold"] == pytest.approx(0.5)


def test_greedy_iou_assignment_reports_empty_side_reasons() -> None:
    strategy = GreedyIoUAssignment(iou_threshold=0.5)

    no_gt = strategy.assign(
        predictions=(AssignmentObject(object_id="pred-only", bbox=(0.0, 0.0, 1.0, 1.0)),),
        ground_truth=(),
    )
    no_predictions = strategy.assign(
        predictions=(),
        ground_truth=(AssignmentObject(object_id="gt-only", bbox=(0.0, 0.0, 1.0, 1.0)),),
    )

    assert [(item.object_id, item.reason) for item in no_gt.unmatched_predictions] == [
        ("pred-only", "no_ground_truth")
    ]
    assert no_gt.unmatched_ground_truth == ()

    assert no_predictions.unmatched_predictions == ()
    assert [(item.object_id, item.reason) for item in no_predictions.unmatched_ground_truth] == [
        ("gt-only", "no_prediction")
    ]


def test_greedy_iou_assignment_rejects_invalid_boxes() -> None:
    strategy = GreedyIoUAssignment(iou_threshold=0.5)

    with pytest.raises(ValueError, match="non-degenerate"):
        strategy.assign(
            predictions=(AssignmentObject(object_id="bad", bbox=(0.0, 0.0, 0.0, 1.0)),),
            ground_truth=(),
        )


def test_rollout_matching_owner_uses_greedy_iou_assignment() -> None:
    match = greedy_match_iou(
        preds=(
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="matched",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[800, 800, 900, 900],
                desc="false positive",
            ),
        ),
        gts=(
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="matched",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[300, 300, 400, 400],
                desc="false negative",
            ),
        ),
        gate_threshold=0.5,
    )

    assert match.matched_pairs == [(0, 0)]
    assert match.fp_pred_indices == [1]
    assert match.fn_gt_indices == [1]
    assert match.gating_rejections == 0
    assert match.matched_maskiou_sum == pytest.approx(1.0)
    assert match.matched_maskiou_count == 1
