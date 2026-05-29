from __future__ import annotations

import inspect

from src.trainers.rollout_correction.target_builder import (
    RolloutCorrectionTargetContext,
    RolloutCorrectionTargetContextInput,
    construct_rollout_correction_target_context,
)
from src.trainers.rollout_matching.contracts import GTObject


def _bbox_object(index: int, desc: str, box: list[int]) -> GTObject:
    return GTObject(
        index=int(index),
        geom_type="bbox_2d",
        points_norm1000=list(box),
        desc=str(desc),
    )


def test_rollout_correction_target_context_uses_parsed_rollout_facts_only() -> None:
    matched_gt = _bbox_object(0, "matched", [100, 100, 200, 200])
    unmatched_anchor = _bbox_object(1, "unlabeled", [500, 500, 600, 600])
    explorer_support = _bbox_object(0, "unlabeled", [502, 502, 602, 602])

    context = construct_rollout_correction_target_context(
        RolloutCorrectionTargetContextInput(
            sample_id="sample-target-boundary",
            gt_objects=[matched_gt],
            accepted_objects_clean=[matched_gt, unmatched_anchor],
            suppressed_duplicate_objects_by_boundary={},
            explorer_objects_raw_by_view=[[explorer_support]],
            anchor_match_by_pred={0: 0},
            explorer_match_by_pred_by_view=[{}],
            anchor_policy_statuses=[],
            unlabeled_consistent_iou_threshold=0.5,
            duplicate_iou_threshold=0.9,
            pseudo_positive_enabled=True,
            expected_peer_count=1,
        )
    )

    assert isinstance(context, RolloutCorrectionTargetContext)
    assert context.sample_id == "sample-target-boundary"
    assert context.metrics["gt_objects"] == 1.0
    assert context.metrics["accepted_objects"] == 2.0
    assert context.metrics["anchor_gt_backed"] == 1.0
    assert context.metrics["valid_explorer_count"] == 1.0
    assert context.triage.anchor_gt_backed_indices == [0]
    assert context.triage.anchor_support_counts[1] == 1
    assert context.triage.association_pairs_by_view == [[(1, 0)]]


def test_rollout_correction_target_context_boundary_has_no_lifecycle_inputs() -> None:
    signature = inspect.signature(construct_rollout_correction_target_context)

    assert list(signature.parameters) == ["request"]
    request_fields = set(RolloutCorrectionTargetContextInput.__dataclass_fields__)
    forbidden_fields = {
        "owner",
        "trainer",
        "model",
        "vllm",
        "ddp",
        "barrier",
        "compute_loss",
        "rollout_many",
        "training_step",
    }
    assert request_fields.isdisjoint(forbidden_fields)

    source = inspect.getsource(construct_rollout_correction_target_context)
    for forbidden in (
        "_rollout_many",
        "vllm",
        "barrier",
        "compute_loss",
        "training_step",
    ):
        assert forbidden not in source
