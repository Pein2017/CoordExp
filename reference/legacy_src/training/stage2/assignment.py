from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias, cast

from src.training.ordering import ObjectBBox


AssignmentScalar: TypeAlias = str | int | float | bool | None


def _freeze_metadata(
    metadata: Mapping[str, AssignmentScalar] | None,
) -> Mapping[str, AssignmentScalar]:
    """Return immutable scalar assignment metadata."""

    if metadata is None:
        return cast(Mapping[str, AssignmentScalar], MappingProxyType({}))

    frozen: dict[str, AssignmentScalar] = {}
    for key, value in metadata.items():
        if type(key) is not str:
            raise TypeError("assignment metadata keys must be strings")
        if (
            value is not None
            and type(value) is not str
            and type(value) is not int
            and type(value) is not float
            and type(value) is not bool
        ):
            raise TypeError("assignment metadata values must be scalar")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("assignment metadata float values must be finite")
        frozen[key] = value

    return cast(Mapping[str, AssignmentScalar], MappingProxyType(frozen))


def _normalize_bbox(bbox: Sequence[int | float]) -> ObjectBBox:
    """Return a finite non-degenerate ``xyxy`` bounding box."""

    if isinstance(bbox, (str, bytes, Mapping)):
        raise TypeError("assignment bbox must be a four-value numeric sequence")
    if len(bbox) != 4:
        raise ValueError("assignment bbox must contain exactly four values")

    values: list[float] = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("assignment bbox values must be finite numeric scalars")
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError("assignment bbox values must be finite")
        values.append(coordinate)

    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("assignment bbox must be non-degenerate xyxy coordinates")

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


@dataclass(frozen=True, slots=True)
class AssignmentObject:
    """Object participating in Stage-2 prediction-to-GT assignment.

    :param object_id: Stable per-example object identity.
    :param bbox: Finite non-degenerate ``xyxy`` bounding box.
    :param description: Optional semantic description.
    :param metadata: Optional scalar provenance metadata.
    """

    object_id: str
    bbox: Sequence[int | float]
    description: str = ""
    metadata: Mapping[str, AssignmentScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate semantic identity, geometry, and metadata."""

        if type(self.object_id) is not str:
            raise TypeError("object_id must be a string")
        if self.object_id == "":
            raise ValueError("object_id must not be empty")
        if type(self.description) is not str:
            raise TypeError("description must be a string")

        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class AssignmentPair:
    """Matched prediction and ground-truth pair.

    :param prediction_index: Input prediction index.
    :param ground_truth_index: Input ground-truth index.
    :param prediction_id: Stable prediction object identity.
    :param ground_truth_id: Stable ground-truth object identity.
    :param iou: Pair IoU score.
    :param reason: Match reason code.
    """

    prediction_index: int
    ground_truth_index: int
    prediction_id: str
    ground_truth_id: str
    iou: float
    reason: str = "matched"


@dataclass(frozen=True, slots=True)
class UnmatchedAssignmentObject:
    """Unmatched assignment-side object with reason and best score."""

    index: int
    object_id: str
    reason: str
    best_iou: float


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """Complete per-example assignment result."""

    pairs: tuple[AssignmentPair, ...]
    unmatched_predictions: tuple[UnmatchedAssignmentObject, ...]
    unmatched_ground_truth: tuple[UnmatchedAssignmentObject, ...]
    metadata: Mapping[str, AssignmentScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze assignment result metadata."""

        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class AssignmentStrategy(ABC):
    """Strategy abstraction for Stage-2 prediction-to-GT assignment."""

    strategy_id: str

    @abstractmethod
    def assign(
        self,
        *,
        predictions: Sequence[AssignmentObject],
        ground_truth: Sequence[AssignmentObject],
    ) -> AssignmentResult:
        """Return one-to-one assignment for a single example."""


class GreedyIoUAssignment(AssignmentStrategy):
    """Greedy one-to-one assignment sorted by IoU and stable indices."""

    strategy_id = "greedy_iou"

    def __init__(self, *, iou_threshold: float) -> None:
        """Initialize the greedy IoU strategy.

        :param iou_threshold: Minimum IoU required for a matched pair.
        """

        self.iou_threshold = _validate_iou_threshold(iou_threshold)

    def assign(
        self,
        *,
        predictions: Sequence[AssignmentObject],
        ground_truth: Sequence[AssignmentObject],
    ) -> AssignmentResult:
        """Return greedy one-to-one matches for a single example."""

        predictions_tuple = tuple(predictions)
        ground_truth_tuple = tuple(ground_truth)

        pairs = self._select_pairs(
            predictions=predictions_tuple,
            ground_truth=ground_truth_tuple,
        )
        matched_predictions = {pair.prediction_index for pair in pairs}
        matched_ground_truth = {pair.ground_truth_index for pair in pairs}

        unmatched_predictions = self._build_unmatched_predictions(
            predictions=predictions_tuple,
            ground_truth=ground_truth_tuple,
            matched_predictions=matched_predictions,
        )
        unmatched_ground_truth = self._build_unmatched_ground_truth(
            predictions=predictions_tuple,
            ground_truth=ground_truth_tuple,
            matched_ground_truth=matched_ground_truth,
        )

        return AssignmentResult(
            pairs=pairs,
            unmatched_predictions=unmatched_predictions,
            unmatched_ground_truth=unmatched_ground_truth,
            metadata={
                "assignment_strategy": self.strategy_id,
                "iou_threshold": self.iou_threshold,
                "matched_iou_sum": float(sum(pair.iou for pair in pairs)),
                "matched_iou_count": int(len(pairs)),
                "gating_rejections": int(0),
            },
        )

    def _select_pairs(
        self,
        *,
        predictions: tuple[AssignmentObject, ...],
        ground_truth: tuple[AssignmentObject, ...],
    ) -> tuple[AssignmentPair, ...]:
        """Return greedy matched pairs above threshold."""

        candidates: list[tuple[float, int, int]] = []
        for prediction_index, prediction in enumerate(predictions):
            for ground_truth_index, gt_object in enumerate(ground_truth):
                iou = _bbox_iou(
                    cast(ObjectBBox, prediction.bbox),
                    cast(ObjectBBox, gt_object.bbox),
                )
                if iou >= self.iou_threshold:
                    candidates.append((float(iou), prediction_index, ground_truth_index))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        matched_predictions: set[int] = set()
        matched_ground_truth: set[int] = set()
        pairs: list[AssignmentPair] = []
        for iou, prediction_index, ground_truth_index in candidates:
            if prediction_index in matched_predictions:
                continue
            if ground_truth_index in matched_ground_truth:
                continue

            prediction = predictions[prediction_index]
            gt_object = ground_truth[ground_truth_index]
            pairs.append(
                AssignmentPair(
                    prediction_index=prediction_index,
                    ground_truth_index=ground_truth_index,
                    prediction_id=prediction.object_id,
                    ground_truth_id=gt_object.object_id,
                    iou=iou,
                )
            )
            matched_predictions.add(prediction_index)
            matched_ground_truth.add(ground_truth_index)

        return tuple(pairs)

    def _build_unmatched_predictions(
        self,
        *,
        predictions: tuple[AssignmentObject, ...],
        ground_truth: tuple[AssignmentObject, ...],
        matched_predictions: set[int],
    ) -> tuple[UnmatchedAssignmentObject, ...]:
        """Return unmatched prediction records with deterministic reasons."""

        unmatched: list[UnmatchedAssignmentObject] = []
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched_predictions:
                continue

            if not ground_truth:
                reason = "no_ground_truth"
                best_iou = 0.0
            else:
                best_iou = max(
                    _bbox_iou(
                        cast(ObjectBBox, prediction.bbox),
                        cast(ObjectBBox, gt_object.bbox),
                    )
                    for gt_object in ground_truth
                )
                reason = (
                    "assignment_conflict"
                    if best_iou >= self.iou_threshold
                    else "below_threshold"
                )

            unmatched.append(
                UnmatchedAssignmentObject(
                    index=prediction_index,
                    object_id=prediction.object_id,
                    reason=reason,
                    best_iou=float(best_iou),
                )
            )

        return tuple(unmatched)

    def _build_unmatched_ground_truth(
        self,
        *,
        predictions: tuple[AssignmentObject, ...],
        ground_truth: tuple[AssignmentObject, ...],
        matched_ground_truth: set[int],
    ) -> tuple[UnmatchedAssignmentObject, ...]:
        """Return unmatched ground-truth records with deterministic reasons."""

        unmatched: list[UnmatchedAssignmentObject] = []
        for ground_truth_index, gt_object in enumerate(ground_truth):
            if ground_truth_index in matched_ground_truth:
                continue

            if not predictions:
                reason = "no_prediction"
                best_iou = 0.0
            else:
                best_iou = max(
                    _bbox_iou(
                        cast(ObjectBBox, prediction.bbox),
                        cast(ObjectBBox, gt_object.bbox),
                    )
                    for prediction in predictions
                )
                reason = (
                    "assignment_conflict"
                    if best_iou >= self.iou_threshold
                    else "below_threshold"
                )

            unmatched.append(
                UnmatchedAssignmentObject(
                    index=ground_truth_index,
                    object_id=gt_object.object_id,
                    reason=reason,
                    best_iou=float(best_iou),
                )
            )

        return tuple(unmatched)


__all__ = [
    "AssignmentObject",
    "AssignmentPair",
    "AssignmentResult",
    "AssignmentStrategy",
    "GreedyIoUAssignment",
    "UnmatchedAssignmentObject",
]
