from __future__ import annotations

import pytest

from src.training.ordering import LegacyTailAppendOrdering, TopLeftSpatialOrdering
from src.training.stage2.assignment import AssignmentObject, GreedyIoUAssignment
from src.training.stage2.duplicate_filter import DuplicateCandidate, DuplicateFilter
from src.training.stage2.planners import (
    LegacyStage2ChannelBTargetBuilderAdapter,
    Stage2ChannelAPlanner,
    Stage2ChannelBPlanner,
    Stage2GreedyIoUShadowPlanner,
    Stage2PlanningObject,
)
from src.training.supervision.plans import SupervisionPlan
from src.trainers.rollout_matching.contracts import GTObject
from src.trainers.stage2_rollout_aligned import _serialize_append_fragment
from src.trainers.stage2_two_channel import (
    _bbox_groups_from_token_ids,
    _build_channel_b_supervision_targets,
    _build_channel_b_triage,
    _matched_prefix_structure_positions,
)


class _CoordLiteralTokenizer:
    """Tiny tokenizer that preserves compact coordinate literals."""

    def __init__(self) -> None:
        self._next_id = 1000
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def _id_for(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = self._next_id
            self._next_id += 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        value = str(text)
        output: list[int] = []
        index = 0
        while index < len(value):
            if value.startswith("<|coord_", index):
                end = value.find("|>", index)
                if end >= 0:
                    literal = value[index + len("<|coord_") : end]
                    output.append(int(literal))
                    index = end + 2
                    continue
            output.append(self._id_for(value[index]))
            index += 1
        return output

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            value = int(token_id)
            if 0 <= value <= 999:
                pieces.append(f"<|coord_{value}|>")
            else:
                pieces.append(self._id_to_token.get(value, "?"))
        return "".join(pieces)


def test_stage2_channel_b_planner_inserts_missing_gt_and_records_tail_append_plan() -> None:
    planner = Stage2ChannelBPlanner(
        duplicate_filter=DuplicateFilter(iou_threshold=0.5),
        ordering=LegacyTailAppendOrdering(),
    )
    accepted_rollout = (
        Stage2PlanningObject(
            object_id="pred-dog",
            description="dog",
            bbox=(50.0, 50.0, 70.0, 70.0),
            provenance="rollout_accepted",
            confidence=0.82,
        ),
    )
    missing_gt = (
        Stage2PlanningObject(
            object_id="gt-cat",
            description="cat",
            bbox=(0.0, 0.0, 10.0, 10.0),
            provenance="gt_false_negative",
        ),
    )

    plan = planner.plan(
        sample_id="sample-stage2",
        template_id="compact_full",
        accepted_rollout_objects=accepted_rollout,
        missing_ground_truth_objects=missing_gt,
        context_id="ctx-stage2",
    )

    assert isinstance(plan, SupervisionPlan)
    assert plan.stage == "stage2"
    assert plan.channel == "channel_b"
    assert plan.template_id == "compact_full"
    assert plan.context_id == "ctx-stage2"
    assert [item.object_id for item in plan.objects] == ["pred-dog", "gt-cat"]

    assert plan.provenance == "stage2_channel_b_shadow_planner"
    assert plan.metadata["assignment_strategy"] == "external_missing_gt"
    assert plan.metadata["duplicate_filter"] == "deterministic_duplicate_filter"
    assert plan.metadata["object_ordering"] == "legacy_tail_append"

    assert plan.objects[0].provenance == "rollout_accepted"
    assert plan.objects[0].metadata["source_role"] == "accepted_rollout"
    assert plan.objects[0].metadata["duplicate_action"] == "kept"
    assert plan.objects[1].provenance == "gt_false_negative"
    assert plan.objects[1].metadata["source_role"] == "false_negative"
    assert plan.objects[1].metadata["source_index"] == 0
    assert plan.objects[1].metadata["plan_index"] == 1


def test_stage2_channel_b_planner_can_emit_sorted_top_left_snapshot() -> None:
    planner = Stage2ChannelBPlanner(
        duplicate_filter=DuplicateFilter(iou_threshold=0.5),
        ordering=TopLeftSpatialOrdering(),
    )
    plan = planner.plan(
        sample_id="sample-stage2",
        template_id="compact_full",
        accepted_rollout_objects=(
            Stage2PlanningObject(
                object_id="pred-bottom",
                description="bottom object",
                bbox=(50.0, 50.0, 70.0, 70.0),
                provenance="rollout_accepted",
            ),
        ),
        missing_ground_truth_objects=(
            Stage2PlanningObject(
                object_id="gt-top",
                description="top object",
                bbox=(0.0, 0.0, 10.0, 10.0),
                provenance="gt_false_negative",
            ),
        ),
    )

    assert [item.object_id for item in plan.objects] == ["gt-top", "pred-bottom"]
    assert plan.metadata["object_ordering"] == "top_left_spatial"
    assert [item.metadata["source_index"] for item in plan.objects] == [0, 0]
    assert [item.metadata["plan_index"] for item in plan.objects] == [0, 1]


def test_stage2_channel_a_planner_emits_ground_truth_plan() -> None:
    planner = Stage2ChannelAPlanner(ordering=TopLeftSpatialOrdering())
    plan = planner.plan(
        sample_id="sample-stage2-a",
        template_id="compact_full",
        ground_truth_objects=(
            AssignmentObject(
                object_id="gt-bottom",
                description="bottom",
                bbox=(20.0, 20.0, 30.0, 30.0),
            ),
            AssignmentObject(
                object_id="gt-top",
                description="top",
                bbox=(0.0, 0.0, 10.0, 10.0),
            ),
        ),
        context_id="ctx-stage2-a",
    )

    assert plan.stage == "stage2"
    assert plan.channel == "channel_a"
    assert plan.template_id == "compact_full"
    assert [item.object_id for item in plan.objects] == ["gt-top", "gt-bottom"]
    assert plan.provenance == "stage2_channel_a_shadow_planner"
    assert plan.metadata["object_ordering"] == "top_left_spatial"
    assert all(item.provenance == "ground_truth" for item in plan.objects)
    assert [item.metadata["source_index"] for item in plan.objects] == [1, 0]
    assert [item.metadata["plan_index"] for item in plan.objects] == [0, 1]


def test_stage2_channel_b_planner_deduplicates_accepted_rollout_before_inserting_fn() -> None:
    planner = Stage2ChannelBPlanner(
        duplicate_filter=DuplicateFilter(iou_threshold=0.5),
        ordering=LegacyTailAppendOrdering(),
    )

    plan = planner.plan(
        sample_id="sample-stage2-dedup",
        template_id="compact_full",
        accepted_rollout_objects=(
            Stage2PlanningObject(
                object_id="pred-supported",
                description="supported",
                bbox=(0.0, 0.0, 10.0, 10.0),
                provenance="rollout_accepted",
                evidence_count=1,
            ),
            Stage2PlanningObject(
                object_id="pred-duplicate",
                description="duplicate",
                bbox=(1.0, 1.0, 11.0, 11.0),
                provenance="rollout_accepted",
                confidence=0.99,
            ),
        ),
        missing_ground_truth_objects=(),
    )

    assert [item.object_id for item in plan.objects] == ["pred-supported"]
    assert plan.metadata["duplicate_suppressed_count"] == 1

    duplicate_candidate = DuplicateCandidate(
        object_id="pred-supported",
        bbox=(0.0, 0.0, 10.0, 10.0),
    )
    assert duplicate_candidate.policy_metadata["policy_id"] == "deterministic_duplicate_filter"


def test_legacy_target_builder_adapter_matches_tail_and_sorted_object_ordering() -> None:
    tokenizer = _CoordLiteralTokenizer()
    accepted_objects_clean = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="anchor",
        ),
        GTObject(
            index=2,
            geom_type="bbox_2d",
            points_norm1000=[600, 700, 650, 760],
            desc="unmatched-clean",
        ),
    ]
    gts = [
        GTObject(
            index=0,
            geom_type="bbox_2d",
            points_norm1000=[400, 500, 450, 560],
            desc="matched",
        ),
        GTObject(
            index=1,
            geom_type="bbox_2d",
            points_norm1000=[10, 20, 30, 40],
            desc="fn",
        ),
    ]
    triage = _build_channel_b_triage(
        accepted_objects_clean=accepted_objects_clean,
        suppressed_duplicate_objects_by_boundary={},
        explorer_objects_raw_by_view=[[accepted_objects_clean[1]]],
        anchor_match_by_pred={0: 0},
        explorer_match_by_pred_by_view=[{}],
        unlabeled_consistent_iou_threshold=0.9,
        duplicate_iou_threshold=0.9,
        pseudo_positive_enabled=True,
    )
    match = type("Match", (), {"matched_pairs": [(0, 0)]})()

    tail_targets = _build_channel_b_supervision_targets(
        tokenizer=tokenizer,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=gts,
        match=match,
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
    )
    sorted_targets = _build_channel_b_supervision_targets(
        tokenizer=tokenizer,
        prompt_ids=[],
        coord_id_set=set(range(1000)),
        gts=gts,
        match=match,
        triage=triage,
        recovered_ground_truth_weight_multiplier=2.0,
        pseudo_positive_enabled=True,
        pseudo_positive_coord_weight=0.4,
        object_field_order="desc_first",
        bbox_groups_from_token_ids_fn=_bbox_groups_from_token_ids,
        matched_prefix_structure_positions_fn=_matched_prefix_structure_positions,
        serialize_append_fragment_fn=_serialize_append_fragment,
        insertion_order="sorted",
    )
    adapter = LegacyStage2ChannelBTargetBuilderAdapter()

    tail_plan = adapter.plan_from_legacy(
        sample_id="legacy-tail",
        template_id="compact_full",
        accepted_objects_clean=accepted_objects_clean,
        gts=gts,
        match=match,
        insertion_order="tail_append",
    )
    sorted_plan = adapter.plan_from_legacy(
        sample_id="legacy-sorted",
        template_id="compact_full",
        accepted_objects_clean=accepted_objects_clean,
        gts=gts,
        match=match,
        insertion_order="sorted",
    )

    assert tail_targets.clean_target_text.find(
        '"desc": "anchor"'
    ) < tail_targets.clean_target_text.find('"desc": "unmatched-clean"')
    assert tail_targets.clean_target_text.find(
        '"desc": "unmatched-clean"'
    ) < tail_targets.clean_target_text.find('"desc": "fn"')
    assert [item.description for item in tail_plan.objects] == [
        "anchor",
        "unmatched-clean",
        "fn",
    ]
    assert tail_plan.objects[1].metadata["source_role"] == "accepted_rollout"
    assert tail_plan.objects[1].metadata["legacy_gt_index"] == 2
    assert tail_plan.objects[2].metadata["legacy_gt_index"] == 1
    assert tail_plan.objects[2].metadata["source_role"] == "false_negative"
    assert tail_plan.metadata["target_builder_adapter"] == (
        "legacy_stage2_channel_b_target_builder_adapter"
    )

    assert sorted_targets.clean_target_text.find(
        '"desc": "fn"'
    ) < sorted_targets.clean_target_text.find('"desc": "anchor"')
    assert sorted_targets.clean_target_text.find(
        '"desc": "anchor"'
    ) < sorted_targets.clean_target_text.find('"desc": "unmatched-clean"')
    assert [item.description for item in sorted_plan.objects] == [
        "fn",
        "anchor",
        "unmatched-clean",
    ]
    assert sorted_plan.objects[0].metadata["legacy_gt_index"] == 1
    assert sorted_plan.metadata["assignment_strategy"] == "legacy_match_result"


def test_stage2_channel_b_planner_reconstructs_duplicate_survivors_by_input_index() -> None:
    planner = Stage2ChannelBPlanner(
        duplicate_filter=DuplicateFilter(iou_threshold=0.5),
        ordering=LegacyTailAppendOrdering(),
    )

    plan = planner.plan(
        sample_id="sample-stage2-duplicate-id",
        template_id="compact_full",
        accepted_rollout_objects=(
            Stage2PlanningObject(
                object_id="same-id",
                description="evidence winner",
                bbox=(0.0, 0.0, 10.0, 10.0),
                provenance="rollout_accepted",
                evidence_count=1,
            ),
            Stage2PlanningObject(
                object_id="same-id",
                description="confidence loser",
                bbox=(1.0, 1.0, 11.0, 11.0),
                provenance="rollout_accepted",
                confidence=0.99,
            ),
        ),
        missing_ground_truth_objects=(),
    )

    assert len(plan.objects) == 1
    assert plan.objects[0].object_id == "same-id"
    assert plan.objects[0].description == "evidence winner"
    assert plan.objects[0].metadata["duplicate_action"] == "kept"
    assert plan.objects[0].metadata["source_index"] == 0
    assert plan.objects[0].metadata["plan_index"] == 0


def test_stage2_greedy_iou_shadow_planner_delegates_to_channel_b_diagnostics() -> None:
    planner = Stage2GreedyIoUShadowPlanner(
        channel_b_planner=Stage2ChannelBPlanner(
            duplicate_filter=DuplicateFilter(iou_threshold=0.5),
            ordering=LegacyTailAppendOrdering(),
        ),
    )

    plan = planner.plan(
        sample_id="sample-stage2-shadow",
        template_id="compact_full",
        predicted_objects=(
            Stage2PlanningObject(
                object_id="pred-dog",
                description="dog",
                bbox=(0.0, 0.0, 10.0, 10.0),
                provenance="rollout_accepted",
                evidence_count=1,
            ),
            Stage2PlanningObject(
                object_id="pred-extra",
                description="extra",
                bbox=(50.0, 50.0, 60.0, 60.0),
                provenance="rollout_accepted",
            ),
        ),
        ground_truth_objects=(
            Stage2PlanningObject(
                object_id="gt-dog",
                description="dog",
                bbox=(0.0, 0.0, 10.0, 10.0),
                provenance="ground_truth",
            ),
            Stage2PlanningObject(
                object_id="gt-cat",
                description="cat",
                bbox=(80.0, 80.0, 90.0, 90.0),
                provenance="ground_truth",
            ),
        ),
        context_id="ctx-shadow",
    )

    assert plan.stage == "stage2"
    assert plan.channel == "channel_b"
    assert plan.context_id == "ctx-shadow"
    assert plan.provenance == "stage2_greedy_iou_shadow_planner"
    assert [item.object_id for item in plan.objects] == [
        "pred-dog",
        "pred-extra",
        "gt-cat",
    ]

    assert plan.metadata["assignment_strategy"] == "greedy_iou"
    assert plan.metadata["matched_prediction_count"] == 1
    assert plan.metadata["matched_ground_truth_count"] == 1
    assert plan.metadata["unmatched_prediction_count"] == 1
    assert plan.metadata["unmatched_ground_truth_count"] == 1

    assert plan.metadata["duplicate_filter"] == "deterministic_duplicate_filter"
    assert plan.metadata["object_ordering"] == "legacy_tail_append"
    assert plan.metadata["accepted_rollout_count"] == 2
    assert plan.metadata["false_negative_count"] == 1
    assert plan.metadata["duplicate_suppressed_count"] == 0
    assert plan.metadata["post_duplicate_prediction_count"] == 2
    assert plan.objects[2].metadata["source_role"] == "false_negative"
    assert plan.objects[2].metadata["source_index"] == 1
    assert plan.objects[2].metadata["plan_index"] == 2


def test_stage2_greedy_iou_shadow_planner_assigns_after_duplicate_filtering() -> None:
    planner = Stage2GreedyIoUShadowPlanner(
        assignment_strategy=GreedyIoUAssignment(iou_threshold=0.5),
        channel_b_planner=Stage2ChannelBPlanner(
            duplicate_filter=DuplicateFilter(iou_threshold=0.1),
            ordering=LegacyTailAppendOrdering(),
        ),
    )

    plan = planner.plan(
        sample_id="sample-stage2-shadow-dedup-before-assignment",
        template_id="compact_full",
        predicted_objects=(
            Stage2PlanningObject(
                object_id="pred-match-suppressed",
                description="dog",
                bbox=(0.0, 0.0, 10.0, 10.0),
                provenance="rollout_accepted",
            ),
            Stage2PlanningObject(
                object_id="pred-duplicate-survivor",
                description="nearby dog",
                bbox=(7.0, 0.0, 17.0, 10.0),
                provenance="rollout_accepted",
                confidence=0.99,
            ),
        ),
        ground_truth_objects=(
            Stage2PlanningObject(
                object_id="gt-dog",
                description="dog",
                bbox=(0.0, 0.0, 10.0, 10.0),
                provenance="ground_truth",
            ),
        ),
    )

    assert [item.object_id for item in plan.objects] == [
        "pred-duplicate-survivor",
        "gt-dog",
    ]
    assert plan.metadata["matched_prediction_count"] == 0
    assert plan.metadata["unmatched_ground_truth_count"] == 1
    assert plan.metadata["duplicate_suppressed_count"] == 1
    assert plan.metadata["accepted_rollout_count"] == 2
    assert plan.metadata["post_duplicate_prediction_count"] == 1
    assert plan.objects[0].metadata["source_index"] == 1
    assert plan.objects[0].metadata["duplicate_action"] == "kept"
    assert plan.objects[1].metadata["source_role"] == "false_negative"
    assert plan.objects[1].metadata["source_index"] == 0


def test_stage2_planning_object_rejects_boolean_bbox_values() -> None:
    with pytest.raises(TypeError, match="planning bbox values"):
        Stage2PlanningObject(
            object_id="bool-box",
            description="bool box",
            bbox=(False, False, True, True),
            provenance="rollout_accepted",
        )


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    (
        ("evidence_count", 1.5, TypeError),
        ("evidence_count", True, TypeError),
        ("evidence_count", -1, ValueError),
        ("explorer_support", 1.5, TypeError),
        ("explorer_support", True, TypeError),
        ("explorer_support", -1, ValueError),
    ),
)
def test_stage2_planning_object_rejects_non_integral_support_counts(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    kwargs = {
        "object_id": "candidate",
        "description": "candidate",
        "bbox": (0.0, 0.0, 10.0, 10.0),
        "provenance": "rollout_accepted",
        field_name: value,
    }

    with pytest.raises(error_type, match=field_name):
        Stage2PlanningObject(**kwargs)


@pytest.mark.parametrize("value", ("false", "0", 1, None))
def test_stage2_planning_object_rejects_non_boolean_crowd_exempt(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="crowd_exempt"):
        Stage2PlanningObject(
            object_id="candidate",
            description="candidate",
            bbox=(0.0, 0.0, 10.0, 10.0),
            provenance="rollout_accepted",
            crowd_exempt=value,
        )
