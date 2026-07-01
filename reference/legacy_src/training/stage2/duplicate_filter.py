from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast

from src.training.ordering import ObjectBBox


DuplicateScalar: TypeAlias = str | int | float | bool | None


def _freeze_metadata(
    metadata: Mapping[str, DuplicateScalar] | None,
) -> Mapping[str, DuplicateScalar]:
    """Return immutable scalar duplicate-filter metadata."""

    if metadata is None:
        return cast(Mapping[str, DuplicateScalar], MappingProxyType({}))

    frozen: dict[str, DuplicateScalar] = {}
    for key, value in metadata.items():
        if type(key) is not str:
            raise TypeError("duplicate metadata keys must be strings")
        if (
            value is not None
            and type(value) is not str
            and type(value) is not int
            and type(value) is not float
            and type(value) is not bool
        ):
            raise TypeError("duplicate metadata values must be scalar")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("duplicate metadata float values must be finite")
        frozen[key] = value

    return cast(Mapping[str, DuplicateScalar], MappingProxyType(frozen))


def _normalize_bbox(bbox: Sequence[int | float]) -> ObjectBBox:
    """Return a finite non-degenerate ``xyxy`` bounding box."""

    if isinstance(bbox, (str, bytes, Mapping)):
        raise TypeError("duplicate bbox must be a four-value numeric sequence")
    if len(bbox) != 4:
        raise ValueError("duplicate bbox must contain exactly four values")

    values: list[float] = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("duplicate bbox values must be finite numeric scalars")
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError("duplicate bbox values must be finite")
        values.append(coordinate)

    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("duplicate bbox must be non-degenerate xyxy coordinates")

    return (values[0], values[1], values[2], values[3])


def _bbox_iou(a: ObjectBBox, b: ObjectBBox) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0.0 else 0.0


def _validate_iou_threshold(value: float) -> float:
    """Return a valid closed-interval IoU threshold."""

    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("iou_threshold must be finite")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    return threshold


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
class DuplicateCandidate:
    """Candidate accepted rollout object for deterministic duplicate filtering.

    :param object_id: Stable object identity.
    :param bbox: Finite non-degenerate ``xyxy`` bounding box.
    :param description: Optional semantic description used by legacy adapters.
    :param confidence: Optional confidence score. Higher scores win after
        evidence and explorer support.
    :param evidence_count: Deterministic evidence count supporting the object.
    :param explorer_support: Deterministic explorer support count.
    :param crowd_exempt: Whether duplicate suppression should preserve this
        candidate under crowd-scene exemption semantics.
    :param metadata: Optional scalar provenance metadata.
    """

    object_id: str
    bbox: Sequence[int | float]
    description: str = ""
    confidence: float | None = None
    evidence_count: int = 0
    explorer_support: int = 0
    crowd_exempt: bool = False
    metadata: Mapping[str, DuplicateScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity, geometry, support signals, and metadata."""

        if type(self.object_id) is not str:
            raise TypeError("object_id must be a string")
        if self.object_id == "":
            raise ValueError("object_id must not be empty")
        if type(self.description) is not str:
            raise TypeError("description must be a string")

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

    @property
    def policy_metadata(self) -> Mapping[str, DuplicateScalar]:
        """Return stable duplicate-filter policy metadata for provenance."""

        return cast(
            Mapping[str, DuplicateScalar],
            MappingProxyType({"policy_id": DuplicateFilter.policy_id}),
        )


@dataclass(frozen=True, slots=True)
class DuplicateFilterDecision:
    """Per-candidate duplicate-filter decision."""

    object_id: str
    input_index: int
    cluster_id: int
    action: str
    reason: str
    survivor_id: str | None
    policy_id: str


@dataclass(frozen=True, slots=True)
class DuplicateFilterResult:
    """Deterministic duplicate-filter output with provenance."""

    survivors: tuple[DuplicateCandidate, ...]
    suppressed: tuple[DuplicateCandidate, ...]
    decisions: tuple[DuplicateFilterDecision, ...]
    metrics: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze duplicate-filter metrics."""

        object.__setattr__(
            self,
            "metrics",
            cast(Mapping[str, int], MappingProxyType(dict(self.metrics))),
        )


class DuplicateFilterStrategy(Protocol):
    """Protocol for Stage-2 duplicate filtering implementations."""

    policy_id: str

    def filter(
        self,
        candidates: Sequence[DuplicateCandidate],
    ) -> DuplicateFilterResult:
        """Return deterministic duplicate-filter output."""


class DuplicateFilter:
    """Deterministic duplicate filter for Stage-2 shadow planning."""

    policy_id = "deterministic_duplicate_filter"

    def __init__(self, *, iou_threshold: float) -> None:
        """Initialize the duplicate filter.

        :param iou_threshold: Minimum IoU for duplicate clustering.
        """

        self.iou_threshold = _validate_iou_threshold(iou_threshold)

    def filter(
        self,
        candidates: Sequence[DuplicateCandidate],
    ) -> DuplicateFilterResult:
        """Return deterministic survivors, suppressions, and decisions."""

        candidates_tuple = tuple(candidates)
        clusters = self._cluster_candidates(candidates_tuple)

        decisions_by_index: dict[int, DuplicateFilterDecision] = {}
        survivor_indices: set[int] = set()
        suppressed_indices: set[int] = set()
        for cluster_id, cluster_indices in enumerate(clusters):
            if any(candidates_tuple[index].crowd_exempt for index in cluster_indices):
                for index in cluster_indices:
                    survivor_indices.add(index)
                    decisions_by_index[index] = self._make_decision(
                        candidate=candidates_tuple[index],
                        input_index=index,
                        cluster_id=cluster_id,
                        action="kept",
                        reason="crowd_exempt",
                        survivor_id=candidates_tuple[index].object_id,
                    )
                continue

            survivor_index = self._select_survivor(
                candidates=candidates_tuple,
                cluster_indices=cluster_indices,
            )
            survivor_indices.add(survivor_index)
            for index in cluster_indices:
                candidate = candidates_tuple[index]
                if index == survivor_index:
                    decisions_by_index[index] = self._make_decision(
                        candidate=candidate,
                        input_index=index,
                        cluster_id=cluster_id,
                        action="kept",
                        reason="survivor",
                        survivor_id=candidate.object_id,
                    )
                else:
                    suppressed_indices.add(index)
                    decisions_by_index[index] = self._make_decision(
                        candidate=candidate,
                        input_index=index,
                        cluster_id=cluster_id,
                        action="suppressed",
                        reason="lower_duplicate_priority",
                        survivor_id=candidates_tuple[survivor_index].object_id,
                    )

        survivors = tuple(
            candidate
            for index, candidate in enumerate(candidates_tuple)
            if index in survivor_indices
        )
        suppressed = tuple(
            candidate
            for index, candidate in enumerate(candidates_tuple)
            if index in suppressed_indices
        )
        decisions = tuple(decisions_by_index[index] for index in range(len(candidates_tuple)))

        return DuplicateFilterResult(
            survivors=survivors,
            suppressed=suppressed,
            decisions=decisions,
            metrics={
                "input_count": len(candidates_tuple),
                "survivor_count": len(survivors),
                "suppressed_count": len(suppressed),
                "cluster_count": len(clusters),
            },
        )

    def _cluster_candidates(
        self,
        candidates: tuple[DuplicateCandidate, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Return transitive duplicate clusters in stable input order."""

        parent = list(range(len(candidates)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        for left_index, left_candidate in enumerate(candidates):
            for right_index in range(left_index + 1, len(candidates)):
                right_candidate = candidates[right_index]
                if (
                    _bbox_iou(
                        cast(ObjectBBox, left_candidate.bbox),
                        cast(ObjectBBox, right_candidate.bbox),
                    )
                    >= self.iou_threshold
                ):
                    union(left_index, right_index)

        clusters_by_root: dict[int, list[int]] = {}
        for index in range(len(candidates)):
            clusters_by_root.setdefault(find(index), []).append(index)

        return tuple(tuple(indices) for _, indices in sorted(clusters_by_root.items()))

    def _select_survivor(
        self,
        *,
        candidates: tuple[DuplicateCandidate, ...],
        cluster_indices: tuple[int, ...],
    ) -> int:
        """Return the deterministic survivor index for one cluster."""

        return min(
            cluster_indices,
            key=lambda index: (
                -int(candidates[index].evidence_count),
                -int(candidates[index].explorer_support),
                -(
                    float(candidates[index].confidence)
                    if candidates[index].confidence is not None
                    else float("-inf")
                ),
                int(index),
            ),
        )

    def _make_decision(
        self,
        *,
        candidate: DuplicateCandidate,
        input_index: int,
        cluster_id: int,
        action: str,
        reason: str,
        survivor_id: str | None,
    ) -> DuplicateFilterDecision:
        """Return a provenance decision for one candidate."""

        return DuplicateFilterDecision(
            object_id=candidate.object_id,
            input_index=input_index,
            cluster_id=cluster_id,
            action=action,
            reason=reason,
            survivor_id=survivor_id,
            policy_id=self.policy_id,
        )


class PassthroughDuplicateFilter:
    """Duplicate-filter strategy for already-filtered legacy owner outputs."""

    policy_id = "passthrough_duplicate_filter"

    def filter(
        self,
        candidates: Sequence[DuplicateCandidate],
    ) -> DuplicateFilterResult:
        """Keep every candidate while recording explicit pass-through decisions."""

        candidates_tuple = tuple(candidates)

        return DuplicateFilterResult(
            survivors=candidates_tuple,
            suppressed=(),
            decisions=tuple(
                DuplicateFilterDecision(
                    object_id=candidate.object_id,
                    input_index=index,
                    cluster_id=index,
                    action="kept",
                    reason="passthrough_already_filtered",
                    survivor_id=candidate.object_id,
                    policy_id=self.policy_id,
                )
                for index, candidate in enumerate(candidates_tuple)
            ),
            metrics={
                "input_count": len(candidates_tuple),
                "survivor_count": len(candidates_tuple),
                "suppressed_count": 0,
                "cluster_count": len(candidates_tuple),
            },
        )


class LegacyChannelBDuplicateControlAdapter:
    """Compatibility adapter around the live rollout-correction duplicate owner."""

    policy_id = "rollout_correction_duplicate_control"

    def __init__(
        self,
        *,
        iou_threshold: float,
        center_radius_scale: float = 0.0,
        unlabeled_consistent_iou_threshold: float = 0.0,
        explorer_candidates_by_view: Sequence[Sequence[DuplicateCandidate]] = (),
    ) -> None:
        """Initialize the live duplicate-control adapter."""

        self.iou_threshold = _validate_iou_threshold(iou_threshold)
        self.center_radius_scale = self._validate_non_negative(
            center_radius_scale,
            field_name="center_radius_scale",
        )
        self.unlabeled_consistent_iou_threshold = _validate_iou_threshold(
            unlabeled_consistent_iou_threshold,
        )
        self.explorer_candidates_by_view = tuple(
            tuple(view) for view in explorer_candidates_by_view
        )

    def filter(
        self,
        candidates: Sequence[DuplicateCandidate],
    ) -> DuplicateFilterResult:
        """Return duplicate-control output by delegating to the live owner."""

        from src.trainers.rollout_correction.target_builder import (
            _apply_rollout_correction_duplicate_control,
        )

        candidates_tuple = tuple(candidates)
        legacy_result = _apply_rollout_correction_duplicate_control(
            anchor_objects_raw=[
                self._to_legacy_gt_object(candidate=candidate, index=index)
                for index, candidate in enumerate(candidates_tuple)
            ],
            explorer_objects_raw_by_view=[
                [
                    self._to_legacy_gt_object(
                        candidate=explorer_candidate,
                        index=index,
                    )
                    for index, explorer_candidate in enumerate(view)
                ]
                for view in self.explorer_candidates_by_view
            ],
            duplicate_iou_threshold=self.iou_threshold,
            center_radius_scale=self.center_radius_scale,
            unlabeled_consistent_iou_threshold=self.unlabeled_consistent_iou_threshold,
        )

        decisions_by_index = {
            int(decision.object_index): decision for decision in legacy_result.decisions
        }
        survivor_indices = [int(obj.index) for obj in legacy_result.kept_anchor_objects]
        suppressed_indices = [int(index) for index in legacy_result.suppressed_anchor_indices]

        decisions = tuple(
            self._to_duplicate_decision(
                candidate=candidate,
                input_index=index,
                legacy_decision=decisions_by_index.get(index),
                candidates=candidates_tuple,
            )
            for index, candidate in enumerate(candidates_tuple)
        )

        return DuplicateFilterResult(
            survivors=tuple(candidates_tuple[index] for index in survivor_indices),
            suppressed=tuple(candidates_tuple[index] for index in suppressed_indices),
            decisions=decisions,
            metrics={
                "input_count": len(candidates_tuple),
                "survivor_count": len(survivor_indices),
                "suppressed_count": len(suppressed_indices),
                "cluster_count": int(
                    legacy_result.counter_metrics.get(
                        "stage2_rollout_correction/correction/dup/N_clusters_total",
                        0,
                    )
                ),
            },
        )

    def _to_legacy_gt_object(
        self,
        *,
        candidate: DuplicateCandidate,
        index: int,
    ):
        """Return the live rollout-matching object contract."""

        from src.trainers.rollout_matching.contracts import GTObject

        return GTObject(
            index=int(index),
            geom_type="bbox_2d",
            points_norm1000=[
                int(round(float(value))) for value in cast(ObjectBBox, candidate.bbox)
            ],
            desc=str(candidate.description or candidate.object_id),
        )

    def _to_duplicate_decision(
        self,
        *,
        candidate: DuplicateCandidate,
        input_index: int,
        legacy_decision: object | None,
        candidates: tuple[DuplicateCandidate, ...],
    ) -> DuplicateFilterDecision:
        """Return a canonical duplicate-filter decision from a legacy decision."""

        if legacy_decision is None:
            return DuplicateFilterDecision(
                object_id=candidate.object_id,
                input_index=int(input_index),
                cluster_id=int(input_index),
                action="kept",
                reason="legacy_unclustered",
                survivor_id=candidate.object_id,
                policy_id=self.policy_id,
            )

        legacy_action = str(getattr(legacy_decision, "action", "keep"))
        action = "suppressed" if legacy_action == "suppress" else "kept"
        survivor_index = int(getattr(legacy_decision, "survivor_index", input_index))
        survivor_id = (
            candidates[survivor_index].object_id
            if 0 <= survivor_index < len(candidates)
            else candidate.object_id
        )
        is_exempt = bool(getattr(legacy_decision, "is_exempt", False))
        reason = (
            "legacy_crowd_exempt"
            if is_exempt
            else "legacy_lower_duplicate_priority"
            if action == "suppressed"
            else "legacy_survivor"
        )

        return DuplicateFilterDecision(
            object_id=candidate.object_id,
            input_index=int(input_index),
            cluster_id=int(getattr(legacy_decision, "cluster_id", input_index) or 0),
            action=action,
            reason=reason,
            survivor_id=survivor_id,
            policy_id=self.policy_id,
        )

    def _validate_non_negative(self, value: float, *, field_name: str) -> float:
        """Return a finite non-negative float."""

        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{field_name} must be finite")
        if parsed < 0.0:
            raise ValueError(f"{field_name} must be >= 0")

        return float(parsed)


__all__ = [
    "DuplicateCandidate",
    "DuplicateFilter",
    "DuplicateFilterDecision",
    "DuplicateFilterResult",
    "DuplicateFilterStrategy",
    "LegacyChannelBDuplicateControlAdapter",
    "PassthroughDuplicateFilter",
]
