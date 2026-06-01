"""DetectionScene-anchored Stage-2 rollout-correction projections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.common.duplicate_control import (
    DuplicateControlDecision,
    DuplicateControlResult,
    apply_duplicate_policy,
    duplicate_control_object_from_bbox,
    validate_duplicate_control_config,
)
from src.detection.scene import DetectionScene
from src.infer.backend import DetectionDecodeResult
from src.training.stage2.rollout_codec import (
    Stage2RolloutObject,
    Stage2RolloutParseResult,
)

from ..rollout_matching.contracts import GTObject
from ..rollout_matching.matching import associate_one_to_one_greedy_iou
from .residual_set import CorrectionEvent

PREDICTION_SOURCE_SHARED_DECODE = "shared_inference_runtime_decode"
_METRIC_BEARING_DECODE_PROVENANCE_KEYS = (
    "model_identity",
    "model_identity_fingerprint",
    "checkpoint_identity",
    "prompt_policy",
    "prompt_policy_fingerprint",
    "decode_policy",
    "decode_policy_fingerprint",
    "metric_eligibility",
)
_METRIC_BEARING_PARSER_METADATA_KEYS = ("rollout_parser_id", "parser_policy")


@dataclass(frozen=True)
class RolloutPrediction:
    """Stage-2 rollout-context view over shared runtime decode plus strict parse."""

    decoded_result: DetectionDecodeResult
    parse_result: Stage2RolloutParseResult
    valid_objects: tuple[GTObject, ...]
    source_label: str
    metric_bearing: bool
    invalid_drop_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    prediction_source: str = PREDICTION_SOURCE_SHARED_DECODE

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_objects", tuple(self.valid_objects))
        object.__setattr__(
            self,
            "invalid_drop_metadata",
            MappingProxyType(dict(self.invalid_drop_metadata)),
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class DetectionAssignment:
    """GT/pred matching decision derived from a DetectionScene and rollout view."""

    scene_gt_objects: tuple[GTObject, ...]
    prediction_objects: tuple[GTObject, ...]
    matched_pairs: tuple[tuple[int, int], ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_gt_indices: tuple[int, ...]
    min_iou: float
    prediction_source: str = PREDICTION_SOURCE_SHARED_DECODE

    @property
    def anchor_match_by_pred(self) -> dict[int, int]:
        return {int(pred_i): int(gt_i) for pred_i, gt_i in self.matched_pairs}


@dataclass(frozen=True)
class DuplicateFilteredRolloutPrediction:
    """RolloutPrediction plus Stage-2-owned duplicate-filter decisions."""

    prediction: RolloutPrediction
    suppressed_duplicate_objects_by_boundary: Mapping[int, tuple[GTObject, ...]]
    decisions: tuple[DuplicateControlDecision, ...]
    support_counts: tuple[int, ...]
    support_rates: tuple[float, ...]
    counter_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "suppressed_duplicate_objects_by_boundary",
            MappingProxyType(
                {
                    int(boundary): tuple(objects)
                    for boundary, objects in dict(
                        self.suppressed_duplicate_objects_by_boundary
                    ).items()
                }
            ),
        )
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(
            self, "support_counts", tuple(int(value) for value in self.support_counts)
        )
        object.__setattr__(
            self,
            "support_rates",
            tuple(float(value) for value in self.support_rates),
        )
        object.__setattr__(
            self,
            "counter_metrics",
            MappingProxyType(
                {str(key): float(value) for key, value in self.counter_metrics.items()}
            ),
        )


def detection_decode_result_from_stage2_rollout(
    *,
    response_text: str,
    response_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    decode_mode: str,
    source_label: str,
    backend: str = "stage2_rollout_runtime",
    backend_metadata: Mapping[str, Any] | None = None,
) -> DetectionDecodeResult:
    """Build the shared decode primitive for a Stage-2 rollout attempt."""

    metadata = {
        "response_family": "stage2_rollout_runtime",
        "decode_mode": str(decode_mode),
        "source_label": str(source_label),
    }
    if backend_metadata is not None:
        metadata.update(dict(backend_metadata))
    return DetectionDecodeResult(
        text=str(response_text),
        generated_token_ids=[int(token_id) for token_id in response_token_ids],
        generated_tokens=None,
        generated_logprobs=None,
        stop_reason=None,
        backend=str(backend),
        backend_metadata=metadata,
        prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
    )


def rollout_prediction_from_shared_decode(
    *,
    decoded_result: DetectionDecodeResult,
    parse_result: Stage2RolloutParseResult,
    metric_bearing: bool,
    source_label: str,
) -> RolloutPrediction:
    """Derive `RolloutPrediction` from shared decode output plus strict parse."""

    if str(parse_result.response_text) != str(decoded_result.text):
        raise ValueError(
            "RolloutPrediction requires parse_result.response_text to match "
            "DetectionDecodeResult.text"
        )
    if bool(metric_bearing):
        _validate_metric_bearing_rollout_prediction_provenance(
            decoded_result=decoded_result,
            parse_result=parse_result,
        )
    valid_objects = tuple(
        _gt_object_from_stage2_rollout_object(obj)
        for obj in tuple(parse_result.valid_objects)
    )
    invalid_drop_metadata = {
        "invalid_rollout": bool(parse_result.invalid_rollout),
        "empty_valid_object_set": bool(parse_result.empty_valid_object_set),
        "truncated": bool(parse_result.truncated),
        "fallback_reason": parse_result.fallback_reason,
        "dropped_invalid": int(parse_result.dropped_invalid),
        "dropped_ambiguous": int(parse_result.dropped_ambiguous),
        "dropped_invalid_by_reason": dict(parse_result.dropped_invalid_by_reason),
    }
    provenance = {
        "prediction_source": PREDICTION_SOURCE_SHARED_DECODE,
        "decode_result_type": type(decoded_result).__name__,
        "backend": str(decoded_result.backend),
        "backend_metadata": dict(decoded_result.backend_metadata),
        "stop_reason": decoded_result.stop_reason,
        "has_prompt_token_ids": decoded_result.prompt_token_ids is not None,
        "has_generated_token_ids": decoded_result.generated_token_ids is not None,
        "parser_id": str(parse_result.parser_id),
        "template_family": str(parse_result.template_family),
        "parser_metadata": dict(parse_result.metadata),
        "metric_bearing": bool(metric_bearing),
        "source_label": str(source_label),
    }
    return RolloutPrediction(
        decoded_result=decoded_result,
        parse_result=parse_result,
        valid_objects=valid_objects,
        source_label=str(source_label),
        metric_bearing=bool(metric_bearing),
        invalid_drop_metadata=invalid_drop_metadata,
        provenance=provenance,
    )


def rollout_prediction_from_legacy_stage2_parse(
    *,
    decoded_result: DetectionDecodeResult,
    parse_result: Any,
    valid_objects: Sequence[GTObject],
    metric_bearing: bool,
    source_label: str,
    template_family: str = "coordjson",
    parser_id: str = "coordjson_legacy",
) -> RolloutPrediction:
    """Bridge migration-only legacy rollout parse output into RolloutPrediction."""

    _ = metric_bearing
    stage2_parse = Stage2RolloutParseResult(
        template_family=str(template_family),  # type: ignore[arg-type]
        parser_id=str(parser_id),
        response_text=str(getattr(parse_result, "response_text", decoded_result.text)),
        valid_objects=tuple(
            Stage2RolloutObject(
                object_id=f"{source_label}-{index}",
                index=int(obj.index),
                desc=str(obj.desc),
                bbox_norm1000=tuple(int(value) for value in obj.points_norm1000),
                provenance="legacy_rollout_matching_migration",
                metadata={"migration_source": "rollout_matching.parse_rollout_for_matching"},
            )
            for index, obj in enumerate(valid_objects)
        ),
        invalid_rollout=bool(getattr(parse_result, "invalid_rollout", False)),
        empty_valid_object_set=not bool(valid_objects)
        and not bool(getattr(parse_result, "invalid_rollout", False)),
        truncated=bool(getattr(parse_result, "truncated", False)),
        fallback_reason=None,
        metadata={
            "parser_policy": "legacy_coordjson_migration",
            "migration_only": True,
            "diagnostic_private_parser": True,
        },
        response_token_ids=tuple(
            int(token_id)
            for token_id in list(getattr(parse_result, "response_token_ids", []) or [])
        ),
        prefix_token_ids=tuple(
            int(token_id)
            for token_id in list(getattr(parse_result, "prefix_token_ids", []) or [])
        ),
        prefix_text=str(getattr(parse_result, "prefix_text", "") or ""),
        dropped_invalid=int(getattr(parse_result, "dropped_invalid", 0) or 0),
        dropped_ambiguous=int(getattr(parse_result, "dropped_ambiguous", 0) or 0),
        dropped_invalid_by_reason=dict(
            getattr(parse_result, "dropped_invalid_by_reason", {}) or {}
        ),
    )
    return rollout_prediction_from_shared_decode(
        decoded_result=decoded_result,
        parse_result=stage2_parse,
        metric_bearing=False,
        source_label=source_label,
    )


def detection_scene_gt_objects(scene: DetectionScene) -> tuple[GTObject, ...]:
    """Project DetectionScene GT objects into the retained Stage-2 matcher shape."""

    projected: list[GTObject] = []
    for scene_object in scene.objects:
        bbox = scene_object.geometry.require_bbox_2d()
        projected.append(
            GTObject(
                index=int(scene_object.scene_object_index),
                geom_type="bbox_2d",
                points_norm1000=[int(value) for value in bbox],
                desc=str(scene_object.desc),
            )
        )
    return tuple(projected)


def assign_detection_scene_rollout_prediction(
    *,
    scene: DetectionScene,
    prediction: RolloutPrediction,
    min_iou: float,
) -> DetectionAssignment:
    """Match a scene's GT objects to parsed rollout predictions."""

    gt_objects = detection_scene_gt_objects(scene)
    prediction_objects = tuple(prediction.valid_objects)
    matched_pairs = tuple(
        (int(pred_i), int(gt_i))
        for pred_i, gt_i in associate_one_to_one_greedy_iou(
            anchors=prediction_objects,
            explorers=gt_objects,
            min_iou=float(min_iou),
        )
    )
    matched_pred_indices = {int(pred_i) for pred_i, _ in matched_pairs}
    matched_gt_indices = {int(gt_i) for _, gt_i in matched_pairs}
    return DetectionAssignment(
        scene_gt_objects=gt_objects,
        prediction_objects=prediction_objects,
        matched_pairs=matched_pairs,
        unmatched_prediction_indices=tuple(
            index
            for index in range(len(prediction_objects))
            if index not in matched_pred_indices
        ),
        unmatched_gt_indices=tuple(
            index for index in range(len(gt_objects)) if index not in matched_gt_indices
        ),
        min_iou=float(min_iou),
    )


def filter_rollout_prediction_duplicates(
    *,
    prediction: RolloutPrediction,
    explorer_predictions: Sequence[RolloutPrediction],
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    unlabeled_consistent_iou_threshold: float,
) -> DuplicateFilteredRolloutPrediction:
    """Apply Stage-2 duplicate filtering to a RolloutPrediction."""

    config = validate_duplicate_control_config(
        iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=float(center_radius_scale),
    )
    anchor_objects = tuple(prediction.valid_objects)
    result: DuplicateControlResult = apply_duplicate_policy(
        anchor_objects=[
            duplicate_control_object_from_bbox(
                index=int(index),
                desc=str(obj.desc),
                bbox_norm1000=obj.points_norm1000,
                source="anchor",
            )
            for index, obj in enumerate(anchor_objects)
        ],
        explorer_objects_by_view=[
            [
                duplicate_control_object_from_bbox(
                    index=int(index),
                    desc=str(obj.desc),
                    bbox_norm1000=obj.points_norm1000,
                    source="explorer",
                )
                for index, obj in enumerate(explorer.valid_objects)
            ]
            for explorer in explorer_predictions
        ],
        config=config,
        support_iou_threshold=float(unlabeled_consistent_iou_threshold),
    )
    kept_indices_sorted = [int(index) for index in result.kept_indices]
    kept_objects = tuple(anchor_objects[index] for index in kept_indices_sorted)
    suppressed_by_boundary: dict[int, list[GTObject]] = {}
    for suppressed_index in sorted(int(index) for index in result.suppressed_indices):
        boundary = sum(
            1 for kept_index in kept_indices_sorted if int(kept_index) < suppressed_index
        )
        suppressed_by_boundary.setdefault(int(boundary), []).append(
            anchor_objects[suppressed_index]
        )
    filtered_prediction = replace(prediction, valid_objects=kept_objects)
    return DuplicateFilteredRolloutPrediction(
        prediction=filtered_prediction,
        suppressed_duplicate_objects_by_boundary={
            int(boundary): tuple(objects)
            for boundary, objects in suppressed_by_boundary.items()
        },
        decisions=tuple(result.decisions),
        support_counts=tuple(int(value) for value in result.support_counts),
        support_rates=tuple(float(value) for value in result.support_rates),
        counter_metrics=dict(result.counter_metrics),
    )


def annotate_correction_events_with_projection_provenance(
    events: Sequence[CorrectionEvent],
    *,
    scene: DetectionScene,
    rollout_prediction: RolloutPrediction,
    assignment: DetectionAssignment,
) -> tuple[CorrectionEvent, ...]:
    """Attach scene/prediction/assignment provenance to retained events."""

    rollout_provenance = dict(rollout_prediction.provenance)
    backend_metadata = dict(rollout_provenance.get("backend_metadata", {}))
    parser_metadata = dict(rollout_provenance.get("parser_metadata", {}))
    prediction_id = (
        f"{rollout_prediction.source_label}:"
        f"{rollout_prediction.prediction_source}:"
        f"{rollout_prediction.parse_result.parser_id}"
    )
    assignment_id = (
        f"scene={scene.image_id}:prediction={prediction_id}:"
        f"matches={len(assignment.matched_pairs)}"
    )
    projection_metadata = {
        "detection_scene": {
            "image_id": int(scene.image_id),
            "image_reference": str(scene.image_reference),
            "object_count": int(len(scene.objects)),
            "coordinate_frame": str(scene.coordinate_frame),
            "coordinate_space": str(scene.coordinate_space),
            "bbox_chart": str(scene.bbox_chart),
        },
        "rollout_prediction": {
            "prediction_id": prediction_id,
            "prediction_source": str(rollout_prediction.prediction_source),
            "parser_id": str(rollout_prediction.parse_result.parser_id),
            "template_family": str(rollout_prediction.parse_result.template_family),
            "metric_bearing": bool(rollout_prediction.metric_bearing),
            "invalid_drop_metadata": dict(rollout_prediction.invalid_drop_metadata),
            "provenance": rollout_provenance,
            "backend_metadata": backend_metadata,
            "parser_metadata": parser_metadata,
            "model_identity": backend_metadata.get("model_identity"),
            "model_identity_fingerprint": backend_metadata.get(
                "model_identity_fingerprint"
            ),
            "checkpoint_identity": backend_metadata.get("checkpoint_identity"),
            "prompt_policy": backend_metadata.get("prompt_policy"),
            "prompt_policy_fingerprint": backend_metadata.get(
                "prompt_policy_fingerprint"
            ),
            "decode_policy": backend_metadata.get("decode_policy"),
            "decode_policy_fingerprint": backend_metadata.get(
                "decode_policy_fingerprint"
            ),
            "parser_policy": parser_metadata.get("parser_policy"),
            "metric_eligibility": backend_metadata.get("metric_eligibility"),
        },
        "detection_assignment": {
            "assignment_id": assignment_id,
            "matched_pairs": [tuple(pair) for pair in assignment.matched_pairs],
            "unmatched_prediction_indices": list(
                assignment.unmatched_prediction_indices
            ),
            "unmatched_gt_indices": list(assignment.unmatched_gt_indices),
            "min_iou": float(assignment.min_iou),
        },
    }
    annotated: list[CorrectionEvent] = []
    for event in events:
        annotated.append(
            replace(
                event,
                metadata={
                    **dict(event.metadata),
                    **projection_metadata,
                },
            )
        )
    return tuple(annotated)


def _gt_object_from_stage2_rollout_object(obj: Stage2RolloutObject) -> GTObject:
    if obj.geom_type != "bbox_2d" or obj.bbox_norm1000 is None:
        raise ValueError("RolloutPrediction only accepts strict bbox_2d rollout objects")
    coords = [int(value) for value in obj.bbox_norm1000]
    if len(coords) != 4:
        raise ValueError("RolloutPrediction bbox_2d requires four coordinates")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError("RolloutPrediction bbox_2d must have positive area")
    return GTObject(
        index=int(obj.index),
        geom_type="bbox_2d",
        points_norm1000=coords,
        desc=str(obj.desc),
    )


def _validate_metric_bearing_rollout_prediction_provenance(
    *,
    decoded_result: DetectionDecodeResult,
    parse_result: Stage2RolloutParseResult,
) -> None:
    backend_metadata = dict(decoded_result.backend_metadata)
    missing_decode = [
        key
        for key in _METRIC_BEARING_DECODE_PROVENANCE_KEYS
        if backend_metadata.get(key) in (None, "")
    ]
    metric_eligibility = backend_metadata.get("metric_eligibility")
    if metric_eligibility is not True:
        missing_decode.append("metric_eligibility=True")
    if missing_decode:
        raise ValueError(
            "metric-bearing RolloutPrediction requires shared decode provenance: "
            + ", ".join(str(key) for key in missing_decode)
        )
    blocked_backend_markers = [
        key
        for key in (
            "diagnostic_private_parser",
            "migration_source",
            "migration_only",
            "private_parser",
        )
        if backend_metadata.get(key) not in (None, "", False)
    ]
    if blocked_backend_markers:
        raise ValueError(
            "diagnostic/private parser decode provenance cannot create "
            "metric-bearing RolloutPrediction: "
            + ", ".join(str(key) for key in blocked_backend_markers)
        )
    parser_metadata = dict(parse_result.metadata)
    if str(parse_result.parser_id or "").strip() == "":
        raise ValueError(
            "metric-bearing RolloutPrediction requires parser provenance: parser_id"
        )
    missing_parser = [
        key
        for key in _METRIC_BEARING_PARSER_METADATA_KEYS
        if parser_metadata.get(key) in (None, "")
    ]
    if missing_parser:
        raise ValueError(
            "metric-bearing RolloutPrediction requires parser provenance: "
            + ", ".join(str(key) for key in missing_parser)
        )
    if bool(parser_metadata.get("diagnostic_private_parser", False)):
        raise ValueError(
            "diagnostic/private parser output cannot create metric-bearing "
            "RolloutPrediction"
        )


__all__ = [
    "CorrectionEvent",
    "DetectionAssignment",
    "DuplicateFilteredRolloutPrediction",
    "PREDICTION_SOURCE_SHARED_DECODE",
    "RolloutPrediction",
    "annotate_correction_events_with_projection_provenance",
    "assign_detection_scene_rollout_prediction",
    "detection_decode_result_from_stage2_rollout",
    "detection_scene_gt_objects",
    "filter_rollout_prediction_duplicates",
    "rollout_prediction_from_legacy_stage2_parse",
    "rollout_prediction_from_shared_decode",
]
