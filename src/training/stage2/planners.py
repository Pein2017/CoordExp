from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias, cast

from src.training.ordering import (
    LegacyTailAppendOrdering,
    ObjectBBox,
    ObjectOrderingStrategy,
    TopLeftSpatialOrdering,
    resolve_object_ordering_strategy,
)
from src.training.stage2.assignment import (
    AssignmentObject,
    AssignmentResult,
    AssignmentStrategy,
    GreedyIoUAssignment,
)
from src.training.stage2.duplicate_filter import (
    DuplicateCandidate,
    DuplicateFilter,
    DuplicateFilterDecision,
    DuplicateFilterStrategy,
    PassthroughDuplicateFilter,
)
from src.training.supervision.plans import SupervisionObject, SupervisionPlan


PlanningScalar: TypeAlias = str | int | float | bool | None


def _freeze_metadata(
    metadata: Mapping[str, PlanningScalar] | None,
) -> Mapping[str, PlanningScalar]:
    """Return immutable scalar Stage-2 planning metadata."""

    if metadata is None:
        return cast(Mapping[str, PlanningScalar], MappingProxyType({}))

    frozen: dict[str, PlanningScalar] = {}
    for key, value in metadata.items():
        if type(key) is not str:
            raise TypeError("planning metadata keys must be strings")
        if (
            value is not None
            and type(value) is not str
            and type(value) is not int
            and type(value) is not float
            and type(value) is not bool
        ):
            raise TypeError("planning metadata values must be scalar")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("planning metadata float values must be finite")
        frozen[key] = value

    return cast(Mapping[str, PlanningScalar], MappingProxyType(frozen))


def _normalize_bbox(bbox: Sequence[int | float]) -> ObjectBBox:
    """Return a finite non-degenerate ``xyxy`` bounding box."""

    if isinstance(bbox, (str, bytes, Mapping)):
        raise TypeError("planning bbox must be a four-value numeric sequence")
    if len(bbox) != 4:
        raise ValueError("planning bbox must contain exactly four values")

    values: list[float] = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("planning bbox values must be finite numeric scalars")
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError("planning bbox values must be finite")
        values.append(coordinate)

    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("planning bbox must be non-degenerate xyxy coordinates")

    return (values[0], values[1], values[2], values[3])


def _validate_support_count(value: object, *, field_name: str) -> int:
    """Return a non-negative support count without lossy coercion."""

    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")

    return int(value)


def _validate_bool(value: object, *, field_name: str) -> bool:
    """Return a boolean without lossy truthiness coercion."""

    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")

    return value


@dataclass(frozen=True, slots=True)
class Stage2PlanningObject:
    """Semantic Stage-2 planning object before template rendering.

    :param object_id: Stable object identity.
    :param description: Semantic object description.
    :param bbox: Finite non-degenerate ``xyxy`` bounding box.
    :param provenance: Semantic provenance label.
    :param confidence: Optional rollout confidence.
    :param evidence_count: Optional deterministic evidence support count.
    :param explorer_support: Optional deterministic explorer support count.
    :param crowd_exempt: Whether duplicate filtering should preserve this object
        under crowd-scene exemption semantics.
    :param metadata: Optional scalar semantic metadata.
    """

    object_id: str
    description: str
    bbox: Sequence[int | float]
    provenance: str
    confidence: float | None = None
    evidence_count: int = 0
    explorer_support: int = 0
    crowd_exempt: bool = False
    metadata: Mapping[str, PlanningScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate semantic fields and freeze scalar metadata."""

        if type(self.object_id) is not str:
            raise TypeError("object_id must be a string")
        if self.object_id == "":
            raise ValueError("object_id must not be empty")
        if type(self.description) is not str:
            raise TypeError("description must be a string")
        if self.description == "":
            raise ValueError("description must not be empty")
        if type(self.provenance) is not str:
            raise TypeError("provenance must be a string")
        if self.provenance == "":
            raise ValueError("provenance must not be empty")

        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence):
                raise ValueError("confidence must be finite")
            object.__setattr__(self, "confidence", confidence)

        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        object.__setattr__(
            self,
            "evidence_count",
            _validate_support_count(
                self.evidence_count,
                field_name="evidence_count",
            ),
        )
        object.__setattr__(
            self,
            "explorer_support",
            _validate_support_count(
                self.explorer_support,
                field_name="explorer_support",
            ),
        )
        object.__setattr__(
            self,
            "crowd_exempt",
            _validate_bool(self.crowd_exempt, field_name="crowd_exempt"),
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class _IndexedStage2PlanningObject:
    """Planning object paired with its source-side provenance index."""

    planning_object: Stage2PlanningObject
    source_index: int
    source_role: str

    @property
    def object_id(self) -> str:
        """Stable object identifier used by ordering strategies."""

        return self.planning_object.object_id

    @property
    def bbox(self) -> ObjectBBox:
        """Bounding box used by ordering strategies."""

        return cast(ObjectBBox, self.planning_object.bbox)


@dataclass(frozen=True, slots=True)
class _Stage2DuplicatePlanningResult:
    """Duplicate-filter result aligned to original planning source indices."""

    accepted_survivors: tuple[_IndexedStage2PlanningObject, ...]
    decisions: tuple[DuplicateFilterDecision, ...]
    decisions_by_source_index: Mapping[int, DuplicateFilterDecision]
    metrics: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze duplicate-planning maps."""

        object.__setattr__(
            self,
            "accepted_survivors",
            tuple(self.accepted_survivors),
        )
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(
            self,
            "decisions_by_source_index",
            cast(
                Mapping[int, DuplicateFilterDecision],
                MappingProxyType(dict(self.decisions_by_source_index)),
            ),
        )
        object.__setattr__(
            self,
            "metrics",
            cast(Mapping[str, int], MappingProxyType(dict(self.metrics))),
        )


def _coerce_stage2_object(
    source: Stage2PlanningObject | AssignmentObject,
    *,
    default_provenance: str,
) -> Stage2PlanningObject:
    """Return a Stage-2 planning object from a supported source object."""

    if isinstance(source, Stage2PlanningObject):
        return source

    if isinstance(source, AssignmentObject):
        return Stage2PlanningObject(
            object_id=source.object_id,
            description=source.description or source.object_id,
            bbox=source.bbox,
            provenance=default_provenance,
            metadata=source.metadata,
        )

    raise TypeError("unsupported Stage-2 planning object")


class Stage2ChannelAPlanner:
    """Shadow planner for Stage-2 Channel-A ground-truth supervision."""

    def __init__(self, *, ordering: ObjectOrderingStrategy | None = None) -> None:
        """Initialize the Channel-A planner."""

        self.ordering = ordering if ordering is not None else TopLeftSpatialOrdering()

    def plan(
        self,
        *,
        sample_id: str,
        template_id: str,
        ground_truth_objects: Sequence[Stage2PlanningObject | AssignmentObject],
        context_id: str | None = None,
    ) -> SupervisionPlan:
        """Return a semantic Channel-A supervision plan for one example."""

        gt_objects = tuple(
            _IndexedStage2PlanningObject(
                planning_object=_coerce_stage2_object(
                    source,
                    default_provenance="ground_truth",
                ),
                source_index=index,
                source_role="ground_truth",
            )
            for index, source in enumerate(ground_truth_objects)
        )
        ordered_objects = self.ordering.order(
            accepted_objects=gt_objects,
            false_negative_objects=(),
        )

        supervision_objects = tuple(
            self._to_supervision_object(
                indexed_object=indexed_object,
                plan_index=plan_index,
            )
            for plan_index, indexed_object in enumerate(ordered_objects)
        )

        return SupervisionPlan(
            sample_id=sample_id,
            stage="stage2",
            template_id=template_id,
            objects=supervision_objects,
            channel="channel_a",
            provenance="stage2_channel_a_shadow_planner",
            context_id=context_id,
            metadata={
                "object_ordering": self.ordering.strategy_id,
                "object_count": len(supervision_objects),
            },
        )

    def _to_supervision_object(
        self,
        *,
        indexed_object: _IndexedStage2PlanningObject,
        plan_index: int,
    ) -> SupervisionObject:
        """Return a semantic supervision object with planner provenance."""

        planning_object = indexed_object.planning_object
        metadata: dict[str, PlanningScalar] = {
            "source_role": indexed_object.source_role,
            "object_ordering": self.ordering.strategy_id,
            "source_index": indexed_object.source_index,
            "plan_index": plan_index,
            "original_index": indexed_object.source_index,
        }
        for key, value in planning_object.metadata.items():
            metadata.setdefault(key, value)

        return SupervisionObject(
            object_id=planning_object.object_id,
            description=planning_object.description,
            bbox=planning_object.bbox,
            provenance=planning_object.provenance,
            metadata=metadata,
        )


class Stage2ChannelBPlanner:
    """Shadow planner for Stage-2 Channel-B rollout-plus-FN supervision."""

    def __init__(
        self,
        *,
        duplicate_filter: DuplicateFilterStrategy | None = None,
        ordering: ObjectOrderingStrategy | None = None,
    ) -> None:
        """Initialize the Channel-B planner."""

        self.duplicate_filter = (
            duplicate_filter
            if duplicate_filter is not None
            else DuplicateFilter(iou_threshold=0.5)
        )
        self.ordering = ordering if ordering is not None else LegacyTailAppendOrdering()

    def plan(
        self,
        *,
        sample_id: str,
        template_id: str,
        accepted_rollout_objects: Sequence[Stage2PlanningObject | AssignmentObject],
        missing_ground_truth_objects: Sequence[Stage2PlanningObject | AssignmentObject],
        context_id: str | None = None,
    ) -> SupervisionPlan:
        """Return a semantic Channel-B supervision plan for one example."""

        accepted_objects = self._index_planning_objects(
            sources=accepted_rollout_objects,
            default_provenance="rollout_accepted",
            source_role="accepted_rollout",
        )
        false_negative_objects = self._index_planning_objects(
            sources=missing_ground_truth_objects,
            default_provenance="gt_false_negative",
            source_role="false_negative",
        )

        return self._plan_indexed(
            sample_id=sample_id,
            template_id=template_id,
            accepted_objects=accepted_objects,
            false_negative_objects=false_negative_objects,
            context_id=context_id,
        )

    def _index_planning_objects(
        self,
        *,
        sources: Sequence[Stage2PlanningObject | AssignmentObject],
        default_provenance: str,
        source_role: str,
    ) -> tuple[_IndexedStage2PlanningObject, ...]:
        """Return indexed planning objects with stable source-side indices."""

        return tuple(
            _IndexedStage2PlanningObject(
                planning_object=_coerce_stage2_object(
                    source,
                    default_provenance=default_provenance,
                ),
                source_index=index,
                source_role=source_role,
            )
            for index, source in enumerate(sources)
        )

    def _filter_accepted_objects(
        self,
        accepted_objects: tuple[_IndexedStage2PlanningObject, ...],
    ) -> _Stage2DuplicatePlanningResult:
        """Return duplicate survivors and decisions keyed by source index."""

        duplicate_result = self.duplicate_filter.filter(
            tuple(
                self._to_duplicate_candidate(indexed_object.planning_object)
                for indexed_object in accepted_objects
            )
        )
        duplicate_decisions = {
            int(accepted_objects[int(decision.input_index)].source_index): decision
            for decision in duplicate_result.decisions
        }
        accepted_survivors = tuple(
            accepted_objects[int(decision.input_index)]
            for decision in duplicate_result.decisions
            if decision.action == "kept"
        )

        return _Stage2DuplicatePlanningResult(
            accepted_survivors=accepted_survivors,
            decisions=tuple(duplicate_result.decisions),
            decisions_by_source_index=duplicate_decisions,
            metrics=duplicate_result.metrics,
        )

    def _plan_indexed(
        self,
        *,
        sample_id: str,
        template_id: str,
        accepted_objects: tuple[_IndexedStage2PlanningObject, ...],
        false_negative_objects: tuple[_IndexedStage2PlanningObject, ...],
        context_id: str | None = None,
        duplicate_plan: _Stage2DuplicatePlanningResult | None = None,
        accepted_rollout_count: int | None = None,
    ) -> SupervisionPlan:
        """Return a Channel-B plan from already indexed semantic objects."""

        resolved_duplicate_plan = (
            duplicate_plan
            if duplicate_plan is not None
            else self._filter_accepted_objects(accepted_objects)
        )

        ordered_objects = self.ordering.order(
            accepted_objects=resolved_duplicate_plan.accepted_survivors,
            false_negative_objects=false_negative_objects,
        )
        supervision_objects = tuple(
            self._to_supervision_object(
                indexed_object=indexed_object,
                plan_index=plan_index,
                duplicate_decision=(
                    resolved_duplicate_plan.decisions_by_source_index.get(
                        int(indexed_object.source_index)
                    )
                    if indexed_object.source_role == "accepted_rollout"
                    else None
                ),
            )
            for plan_index, indexed_object in enumerate(ordered_objects)
        )

        return SupervisionPlan(
            sample_id=sample_id,
            stage="stage2",
            template_id=template_id,
            objects=supervision_objects,
            channel="channel_b",
            provenance="stage2_channel_b_shadow_planner",
            context_id=context_id,
            metadata={
                "assignment_strategy": "external_missing_gt",
                "duplicate_filter": self.duplicate_filter.policy_id,
                "duplicate_suppressed_count": resolved_duplicate_plan.metrics[
                    "suppressed_count"
                ],
                "object_ordering": self.ordering.strategy_id,
                "accepted_rollout_count": (
                    len(accepted_objects)
                    if accepted_rollout_count is None
                    else int(accepted_rollout_count)
                ),
                "false_negative_count": len(false_negative_objects),
                "object_count": len(supervision_objects),
            },
        )

    def _to_duplicate_candidate(
        self,
        planning_object: Stage2PlanningObject,
    ) -> DuplicateCandidate:
        """Return a duplicate-filter candidate for an accepted rollout object."""

        return DuplicateCandidate(
            object_id=planning_object.object_id,
            bbox=planning_object.bbox,
            confidence=planning_object.confidence,
            evidence_count=planning_object.evidence_count,
            explorer_support=planning_object.explorer_support,
            crowd_exempt=planning_object.crowd_exempt,
            metadata={
                "source_role": "accepted_rollout",
            },
        )

    def _to_supervision_object(
        self,
        *,
        indexed_object: _IndexedStage2PlanningObject,
        plan_index: int,
        duplicate_decision: DuplicateFilterDecision | None,
    ) -> SupervisionObject:
        """Return a semantic supervision object with planner provenance."""

        planning_object = indexed_object.planning_object
        metadata: dict[str, PlanningScalar] = {
            "source_role": indexed_object.source_role,
            "object_ordering": self.ordering.strategy_id,
            "source_index": indexed_object.source_index,
            "plan_index": plan_index,
            "original_index": indexed_object.source_index,
        }
        for key, value in planning_object.metadata.items():
            metadata.setdefault(key, value)
        if planning_object.confidence is not None:
            metadata["confidence"] = planning_object.confidence
        if planning_object.evidence_count:
            metadata["evidence_count"] = planning_object.evidence_count
        if planning_object.explorer_support:
            metadata["explorer_support"] = planning_object.explorer_support

        if duplicate_decision is not None:
            metadata["duplicate_action"] = duplicate_decision.action
            metadata["duplicate_reason"] = duplicate_decision.reason
            metadata["duplicate_cluster_id"] = duplicate_decision.cluster_id
            metadata["duplicate_survivor_id"] = duplicate_decision.survivor_id

        return SupervisionObject(
            object_id=planning_object.object_id,
            description=planning_object.description,
            bbox=planning_object.bbox,
            provenance=planning_object.provenance,
            metadata=metadata,
        )


class Stage2GreedyIoUShadowPlanner:
    """Diagnostic Stage-2 planner that shadows legacy Channel-B assignment.

    The planner performs greedy IoU assignment only to derive semantic
    diagnostics and false-negative GT insertion candidates. It delegates final
    Channel-B object planning to ``Stage2ChannelBPlanner`` so the live trainer
    defaults remain untouched.
    """

    def __init__(
        self,
        *,
        assignment_strategy: AssignmentStrategy | None = None,
        channel_b_planner: Stage2ChannelBPlanner | None = None,
    ) -> None:
        """Initialize the shadow planner.

        :param assignment_strategy: Optional assignment strategy. Defaults to
            greedy IoU with the legacy diagnostic threshold.
        :param channel_b_planner: Optional Channel-B semantic planner.
        """

        self.assignment_strategy = (
            assignment_strategy
            if assignment_strategy is not None
            else GreedyIoUAssignment(iou_threshold=0.5)
        )
        self.channel_b_planner = (
            channel_b_planner
            if channel_b_planner is not None
            else Stage2ChannelBPlanner()
        )

    def plan(
        self,
        *,
        sample_id: str,
        template_id: str,
        predicted_objects: Sequence[Stage2PlanningObject | AssignmentObject],
        ground_truth_objects: Sequence[Stage2PlanningObject | AssignmentObject],
        context_id: str | None = None,
    ) -> SupervisionPlan:
        """Return a Channel-B shadow plan with greedy-IoU diagnostics."""

        return self.plan_with_diagnostics(
            sample_id=sample_id,
            template_id=template_id,
            predicted_objects=predicted_objects,
            ground_truth_objects=ground_truth_objects,
            context_id=context_id,
        ).plan

    def plan_with_diagnostics(
        self,
        *,
        sample_id: str,
        template_id: str,
        predicted_objects: Sequence[Stage2PlanningObject | AssignmentObject],
        ground_truth_objects: Sequence[Stage2PlanningObject | AssignmentObject],
        context_id: str | None = None,
    ) -> "Stage2GreedyIoUShadowPlanningResult":
        """Return the Channel-B shadow plan and planner-owned diagnostics."""

        predictions = tuple(predicted_objects)
        ground_truth = tuple(ground_truth_objects)
        indexed_predictions = self.channel_b_planner._index_planning_objects(
            sources=predictions,
            default_provenance="rollout_accepted",
            source_role="accepted_rollout",
        )
        indexed_ground_truth = self.channel_b_planner._index_planning_objects(
            sources=ground_truth,
            default_provenance="gt_false_negative",
            source_role="false_negative",
        )
        duplicate_plan = self.channel_b_planner._filter_accepted_objects(
            indexed_predictions,
        )

        assignment_result = self.assignment_strategy.assign(
            predictions=tuple(
                self._to_assignment_object(
                    source=indexed_object.planning_object,
                    source_index=indexed_object.source_index,
                    default_provenance="rollout_accepted",
                )
                for indexed_object in duplicate_plan.accepted_survivors
            ),
            ground_truth=tuple(
                self._to_assignment_object(
                    source=indexed_object.planning_object,
                    source_index=indexed_object.source_index,
                    default_provenance="gt_false_negative",
                )
                for indexed_object in indexed_ground_truth
            ),
        )
        false_negative_objects = tuple(
            indexed_ground_truth[unmatched.index]
            for unmatched in assignment_result.unmatched_ground_truth
        )

        channel_b_plan = self.channel_b_planner._plan_indexed(
            sample_id=sample_id,
            template_id=template_id,
            accepted_objects=duplicate_plan.accepted_survivors,
            false_negative_objects=false_negative_objects,
            context_id=context_id,
            duplicate_plan=duplicate_plan,
            accepted_rollout_count=len(indexed_predictions),
        )

        plan = SupervisionPlan(
            sample_id=channel_b_plan.sample_id,
            stage=channel_b_plan.stage,
            template_id=channel_b_plan.template_id,
            objects=channel_b_plan.objects,
            channel=channel_b_plan.channel,
            provenance="stage2_greedy_iou_shadow_planner",
            context_id=channel_b_plan.context_id,
            metadata={
                **dict(channel_b_plan.metadata),
                "assignment_strategy": self.assignment_strategy.strategy_id,
                "matched_prediction_count": len(assignment_result.pairs),
                "matched_ground_truth_count": len(assignment_result.pairs),
                "unmatched_prediction_count": len(
                    assignment_result.unmatched_predictions
                ),
                "unmatched_ground_truth_count": len(
                    assignment_result.unmatched_ground_truth
                ),
                "post_duplicate_prediction_count": len(
                    duplicate_plan.accepted_survivors
                ),
            },
        )

        return Stage2GreedyIoUShadowPlanningResult(
            plan=plan,
            duplicate_decisions=duplicate_plan.decisions,
            assignment_result=assignment_result,
            post_duplicate_prediction_count=len(duplicate_plan.accepted_survivors),
        )

    def _to_assignment_object(
        self,
        *,
        source: Stage2PlanningObject | AssignmentObject,
        source_index: int,
        default_provenance: str,
    ) -> AssignmentObject:
        """Return an assignment object with stable shadow provenance."""

        if isinstance(source, AssignmentObject):
            return source

        if isinstance(source, Stage2PlanningObject):
            return AssignmentObject(
                object_id=source.object_id,
                description=source.description,
                bbox=source.bbox,
                metadata={
                    **dict(source.metadata),
                    "source_index": source_index,
                    "source_provenance": source.provenance or default_provenance,
                },
            )

        raise TypeError("unsupported Stage-2 shadow planning object")


@dataclass(frozen=True, slots=True)
class Stage2GreedyIoUShadowPlanningResult:
    """Planner-owned Stage-2 shadow plan with reusable diagnostics.

    :param plan: Final semantic Channel-B supervision plan.
    :param duplicate_decisions: Duplicate-filter decisions from the same pass
        that produced ``plan``.
    :param assignment_result: Greedy assignment result from the same
        post-duplicate survivor set used for false-negative insertion.
    :param post_duplicate_prediction_count: Survivor count used during
        assignment.
    """

    plan: SupervisionPlan
    duplicate_decisions: tuple[DuplicateFilterDecision, ...]
    assignment_result: AssignmentResult
    post_duplicate_prediction_count: int


class LegacyStage2ChannelBTargetBuilderAdapter:
    """Compatibility seam from live target-builder contracts to semantic plans."""

    strategy_id = "legacy_stage2_channel_b_target_builder_adapter"

    def __init__(
        self,
        *,
        duplicate_filter: DuplicateFilterStrategy | None = None,
    ) -> None:
        """Initialize the legacy target-builder adapter."""

        self.duplicate_filter = (
            duplicate_filter
            if duplicate_filter is not None
            else PassthroughDuplicateFilter()
        )

    def plan_from_legacy(
        self,
        *,
        sample_id: str,
        template_id: str,
        accepted_objects_clean: Sequence[object],
        gts: Sequence[object],
        match: object,
        insertion_order: str = "tail_append",
        context_id: str | None = None,
    ) -> SupervisionPlan:
        """Return a semantic plan from live Stage-2 target-builder inputs."""

        matched_gt_indices = {
            int(gt_index)
            for _prediction_index, gt_index in getattr(match, "matched_pairs", ())
        }
        missing_gt_objects = tuple(
            self._from_legacy_gt_object(
                gt_object=gt_object,
                default_provenance="legacy_target_builder_false_negative",
            )
            for index, gt_object in enumerate(gts)
            if int(index) not in matched_gt_indices
        )
        accepted_objects = tuple(
            self._from_legacy_gt_object(
                gt_object=gt_object,
                default_provenance="legacy_target_builder_accepted_clean",
            )
            for gt_object in accepted_objects_clean
        )
        planner = Stage2ChannelBPlanner(
            duplicate_filter=self.duplicate_filter,
            ordering=resolve_object_ordering_strategy(insertion_order),
        )
        plan = planner.plan(
            sample_id=sample_id,
            template_id=template_id,
            accepted_rollout_objects=accepted_objects,
            missing_ground_truth_objects=missing_gt_objects,
            context_id=context_id,
        )

        return SupervisionPlan(
            sample_id=plan.sample_id,
            stage=plan.stage,
            template_id=plan.template_id,
            objects=plan.objects,
            channel=plan.channel,
            provenance=self.strategy_id,
            context_id=plan.context_id,
            metadata={
                **dict(plan.metadata),
                "target_builder_adapter": self.strategy_id,
                "assignment_strategy": "legacy_match_result",
            },
        )

    def _from_legacy_gt_object(
        self,
        *,
        gt_object: object,
        default_provenance: str,
    ) -> Stage2PlanningObject:
        """Return a Stage-2 planning object from the live GTObject contract."""

        legacy_index = int(getattr(gt_object, "index"))
        return Stage2PlanningObject(
            object_id=f"legacy-{legacy_index}",
            description=str(getattr(gt_object, "desc", f"object-{legacy_index}")),
            bbox=tuple(float(value) for value in getattr(gt_object, "points_norm1000")),
            provenance=default_provenance,
            metadata={
                "legacy_gt_index": legacy_index,
            },
        )


__all__ = [
    "LegacyStage2ChannelBTargetBuilderAdapter",
    "Stage2ChannelAPlanner",
    "Stage2ChannelBPlanner",
    "Stage2GreedyIoUShadowPlanningResult",
    "Stage2GreedyIoUShadowPlanner",
    "Stage2PlanningObject",
]
