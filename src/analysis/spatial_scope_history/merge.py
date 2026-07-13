"""Pure object normalization and merge semantics for spatial-scope research."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal, cast

from src.analysis.spatial_scope_history.execution_evidence import (
    ExecutionEvidenceEnvelope,
)
from src.analysis.spatial_scope_history.parse_score_evidence import (
    CanonicalParseScoreReceipt,
    validate_parse_score_receipt_association,
)
from src.analysis.spatial_scope_history.schedule import (
    ResearchArmDefinition,
    primary_arm_definition,
)
from src.analysis.spatial_scope_history.spatial import (
    SpatialCoordinateReceipt,
    SpatialGrid,
    SpatialGridSpec,
    SpatialVariantMode,
    spatial_variant_mode_for_arm,
)
from src.common.errors import DataContractError
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
)
from src.inference.scoring import SCORE_POLICY
from src.vis.matching import iou_xyxy


COMPACT_OBJECT_SELECTED_TOKEN_SCORE_VERSION_1 = str(SCORE_POLICY["id"])
NON_MAXIMUM_SUPPRESSION_IOU_THRESHOLD = 0.70
STRICT_DUPLICATE_IOU_THRESHOLD = 0.85

OwnershipStatus = Literal["owned", "non_owning", "not_applicable"]
PredictionViewName = Literal[
    "raw_any_call",
    "raw_owning_call",
    "non_owning_diagnostic",
    "pre_merge",
    "post_merge",
]
PixelBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ExpectedImageFrame:
    """One sealed image identifier and source-canvas frame expected in evidence."""

    image_id: str
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        _require_nonempty_string(self.image_id, field="expected_image.image_id")
        _require_positive_integer(
            self.source_width, field="expected_image.source_width"
        )
        _require_positive_integer(
            self.source_height, field="expected_image.source_height"
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


@dataclass(frozen=True)
class NormalizedCallPrediction:
    """One valid parsed row normalized into the source-canvas frame."""

    image_id: str
    arm: ResearchArmDefinition
    canonical_call_id: str
    canonical_cell_index: int | None
    generated_row_index: int
    prediction_id: str
    normalized_category_name: str
    evaluator_category_id: int
    official_coco_category_id: int
    category_registry_sha256: str
    global_bbox_xyxy: PixelBox
    local_bbox_xyxy: PixelBox | None
    score: float
    score_source: str
    ownership_status: OwnershipStatus
    owner_cell_index: int | None
    spatial_coordinate_receipt: SpatialCoordinateReceipt | None
    source_width: int
    source_height: int
    execution_evidence: ExecutionEvidenceEnvelope
    parse_score_receipt: CanonicalParseScoreReceipt

    def __post_init__(self) -> None:
        expected_evaluator_category_id = (
            COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME.get(
                self.normalized_category_name
            )
        )
        expected_official_category_id = COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME.get(
            self.normalized_category_name
        )
        if (
            expected_evaluator_category_id != self.evaluator_category_id
            or expected_official_category_id != self.official_coco_category_id
            or self.category_registry_sha256 != COCO_80_CATEGORY_NAMESPACE_SHA256
        ):
            raise _contract_error(
                "prediction category namespaces differ from the canonical registry",
                "analysis.spatial_merge_prediction_category_namespace",
                prediction_id=self.prediction_id,
            )
        evidence = self.execution_evidence
        validate_parse_score_receipt_association(
            self.parse_score_receipt,
            execution_evidence=evidence,
        )
        receipt = self.parse_score_receipt
        if (
            self.generated_row_index != receipt.generated_row_index
            or self.prediction_id != receipt.object_span_id
            or self.normalized_category_name != receipt.normalized_category_name
            or self.score != receipt.score
            or self.score_source != receipt.score_policy_id
        ):
            raise _contract_error(
                "prediction semantics differ from canonical parse-and-score evidence",
                "analysis.spatial_merge_prediction_parse_score",
                prediction_id=self.prediction_id,
            )
        if receipt.coordinate_source == "source_canvas":
            geometry_matches = (
                self.global_bbox_xyxy == receipt.parsed_bbox_xyxy
                and self.local_bbox_xyxy is None
                and self.spatial_coordinate_receipt is None
                and self.ownership_status == "not_applicable"
                and self.owner_cell_index is None
            )
        else:
            spatial_receipt = self.spatial_coordinate_receipt
            geometry_matches = spatial_receipt is not None and (
                self.local_bbox_xyxy == receipt.parsed_bbox_xyxy
                and self.global_bbox_xyxy
                == tuple(
                    float(value) for value in spatial_receipt.clipped_global_integer_box
                )
                and spatial_receipt.original_coordinate_bins == receipt.coordinate_bins
                and spatial_receipt.local_integer_box == receipt.parsed_bbox_xyxy
                and spatial_receipt.source_width == self.source_width
                and spatial_receipt.source_height == self.source_height
            )
        if not geometry_matches:
            raise _contract_error(
                "prediction geometry differs from canonical parse evidence",
                "analysis.spatial_merge_prediction_parse_geometry",
                prediction_id=self.prediction_id,
            )
        if self.image_id != str(evidence.image_id):
            raise _contract_error(
                "prediction image differs from its execution evidence",
                "analysis.spatial_merge_prediction_execution_image",
                prediction_id=self.prediction_id,
            )
        if self.arm != evidence.arm:
            raise _contract_error(
                "prediction arm differs from its execution evidence",
                "analysis.spatial_merge_prediction_execution_arm",
                prediction_id=self.prediction_id,
            )
        if self.canonical_call_id != evidence.request_id:
            raise _contract_error(
                "prediction call differs from its execution evidence",
                "analysis.spatial_merge_prediction_execution_call",
                prediction_id=self.prediction_id,
            )
        if self.canonical_cell_index != evidence.canonical_cell_index:
            raise _contract_error(
                "prediction cell differs from its execution evidence",
                "analysis.spatial_merge_prediction_execution_cell",
                prediction_id=self.prediction_id,
            )
        if (self.source_width, self.source_height) != (
            evidence.source_width,
            evidence.source_height,
        ):
            raise _contract_error(
                "prediction frame differs from its execution evidence",
                "analysis.spatial_merge_prediction_execution_frame",
                prediction_id=self.prediction_id,
            )

    @property
    def deterministic_rank_key(self) -> tuple[Any, ...]:
        """Frozen descending-score rank represented as an ascending sort key."""

        return (
            -self.score,
            self.image_id,
            self.canonical_call_id,
            self.generated_row_index,
            self.prediction_id,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "arm": self.arm.to_artifact_dict(),
            "canonical_call_id": self.canonical_call_id,
            "canonical_cell_index": self.canonical_cell_index,
            "generated_row_index": self.generated_row_index,
            "prediction_id": self.prediction_id,
            "normalized_category_name": self.normalized_category_name,
            "evaluator_category_id": self.evaluator_category_id,
            "official_coco_category_id": self.official_coco_category_id,
            "category_registry_sha256": self.category_registry_sha256,
            "global_bbox_xyxy": list(self.global_bbox_xyxy),
            "local_bbox_xyxy": (
                list(self.local_bbox_xyxy) if self.local_bbox_xyxy is not None else None
            ),
            "score": self.score,
            "score_source": self.score_source,
            "ownership_status": self.ownership_status,
            "owner_cell_index": self.owner_cell_index,
            "spatial_coordinate_receipt": _receipt_to_json(
                self.spatial_coordinate_receipt
            ),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "execution_evidence": self.execution_evidence.to_artifact_dict(),
            "parse_score_receipt": self.parse_score_receipt.to_artifact_dict(),
        }


@dataclass(frozen=True)
class PredictionSet:
    """One named immutable view over predictions from one image and arm."""

    image_id: str
    arm: ResearchArmDefinition
    view_name: PredictionViewName
    predictions: tuple[NormalizedCallPrediction, ...]

    def __post_init__(self) -> None:
        identifiers: set[str] = set()
        source_frame: tuple[int, int] | None = None
        for prediction in self.predictions:
            if prediction.image_id != self.image_id:
                raise _contract_error(
                    "prediction image does not match prediction-set image",
                    "analysis.spatial_merge_prediction_set_image",
                    prediction_id=prediction.prediction_id,
                )
            if prediction.arm != self.arm:
                raise _contract_error(
                    "prediction arm does not match prediction-set arm",
                    "analysis.spatial_merge_prediction_set_arm",
                    prediction_id=prediction.prediction_id,
                )
            if prediction.prediction_id in identifiers:
                raise _contract_error(
                    "prediction identifiers must be unique within an image arm",
                    "analysis.spatial_merge_duplicate_prediction_id",
                    prediction_id=prediction.prediction_id,
                )
            identifiers.add(prediction.prediction_id)
            prediction_frame = (prediction.source_width, prediction.source_height)
            if source_frame is None:
                source_frame = prediction_frame
            elif prediction_frame != source_frame:
                raise _contract_error(
                    "predictions in one image arm must share a source-canvas frame",
                    "analysis.spatial_merge_prediction_set_frame",
                    prediction_id=prediction.prediction_id,
                    expected_source_frame=source_frame,
                    actual_source_frame=prediction_frame,
                )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "arm": self.arm.to_artifact_dict(),
            "view_name": self.view_name,
            "predictions": [item.to_json_dict() for item in self.predictions],
        }


@dataclass(frozen=True)
class SuppressionEdge:
    """One direct kept-box to suppressed-box Non-Maximum Suppression edge."""

    suppressor_prediction_id: str
    suppressed_prediction_id: str
    normalized_category_name: str
    intersection_over_union: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "suppressor_prediction_id": self.suppressor_prediction_id,
            "suppressed_prediction_id": self.suppressed_prediction_id,
            "normalized_category_name": self.normalized_category_name,
            "intersection_over_union": self.intersection_over_union,
        }


@dataclass(frozen=True)
class StrictDuplicateComponent:
    """One multi-member same-class IoU-graph component at threshold 0.85."""

    representative_prediction_id: str
    member_prediction_ids: tuple[str, ...]
    normalized_category_name: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "representative_prediction_id": self.representative_prediction_id,
            "member_prediction_ids": list(self.member_prediction_ids),
            "normalized_category_name": self.normalized_category_name,
        }


@dataclass(frozen=True)
class ImageMergeResult:
    """All mandatory raw, ownership-filtered, and merged views for one image."""

    image_id: str
    arm: ResearchArmDefinition
    source_width: int
    source_height: int
    raw_any_call: PredictionSet
    raw_owning_call: PredictionSet
    non_owning_diagnostic: PredictionSet
    pre_merge: PredictionSet
    post_merge: PredictionSet
    suppression_edges: tuple[SuppressionEdge, ...]
    pre_merge_strict_duplicate_components: tuple[StrictDuplicateComponent, ...]
    post_merge_strict_duplicate_components: tuple[StrictDuplicateComponent, ...]

    @property
    def pre_merge_strict_duplicate_rate(self) -> float | None:
        return _strict_duplicate_rate(
            self.pre_merge, self.pre_merge_strict_duplicate_components
        )

    @property
    def post_merge_strict_duplicate_rate(self) -> float | None:
        return _strict_duplicate_rate(
            self.post_merge, self.post_merge_strict_duplicate_components
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "arm": self.arm.to_artifact_dict(),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "raw_any_call": self.raw_any_call.to_json_dict(),
            "raw_owning_call": self.raw_owning_call.to_json_dict(),
            "non_owning_diagnostic": self.non_owning_diagnostic.to_json_dict(),
            "pre_merge": self.pre_merge.to_json_dict(),
            "post_merge": self.post_merge.to_json_dict(),
            "suppression_edges": [
                edge.to_json_dict() for edge in self.suppression_edges
            ],
            "pre_merge_strict_duplicate_components": [
                component.to_json_dict()
                for component in self.pre_merge_strict_duplicate_components
            ],
            "post_merge_strict_duplicate_components": [
                component.to_json_dict()
                for component in self.post_merge_strict_duplicate_components
            ],
            "pre_merge_strict_duplicate_rate": self.pre_merge_strict_duplicate_rate,
            "post_merge_strict_duplicate_rate": self.post_merge_strict_duplicate_rate,
        }


@dataclass(frozen=True)
class ArmMergeResult:
    """Deterministically ordered image-level merge results for one arm."""

    arm: ResearchArmDefinition
    expected_images: tuple[ExpectedImageFrame, ...]
    allowed_arms: tuple[ResearchArmDefinition, ...]
    image_results: tuple[ImageMergeResult, ...]

    def __post_init__(self) -> None:
        canonical_universe = _validate_expected_images(self.expected_images)
        if self.expected_images != canonical_universe:
            raise _contract_error(
                "sealed image universe must use canonical image-identifier order",
                "analysis.spatial_merge_arm_result_universe_order",
            )
        canonical_allowed = _resolve_allowed_arms(
            tuple(item.arm_code for item in self.allowed_arms)
        )
        if self.allowed_arms != canonical_allowed or self.arm not in canonical_allowed:
            raise _contract_error(
                "arm result is not bound to its canonical allowed-arm schedule",
                "analysis.spatial_merge_arm_result_allowed_arms",
                arm_identifier=self.arm.arm_code,
            )
        identifiers: set[str] = set()
        expected_by_id = {item.image_id: item for item in self.expected_images}
        for result in self.image_results:
            if result.arm != self.arm:
                raise _contract_error(
                    "image merge result belongs to a different arm",
                    "analysis.spatial_merge_arm_result_arm",
                    image_id=result.image_id,
                )
            if result.image_id in identifiers:
                raise _contract_error(
                    "image identifiers must be unique within an arm result",
                    "analysis.spatial_merge_duplicate_image_id",
                    image_id=result.image_id,
                )
            identifiers.add(result.image_id)
            expected = expected_by_id.get(result.image_id)
            if expected is None or (
                result.source_width != expected.source_width
                or result.source_height != expected.source_height
            ):
                raise _contract_error(
                    "image merge result differs from its sealed source frame",
                    "analysis.spatial_merge_arm_result_frame",
                    image_id=result.image_id,
                )
        result_identifiers = tuple(result.image_id for result in self.image_results)
        expected_identifiers = tuple(item.image_id for item in self.expected_images)
        if result_identifiers != expected_identifiers:
            raise _contract_error(
                "arm image results must exactly cover the sealed image universe",
                "analysis.spatial_merge_arm_result_universe",
                expected_image_ids=expected_identifiers,
                actual_image_ids=result_identifiers,
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.to_artifact_dict(),
            "expected_images": [item.to_json_dict() for item in self.expected_images],
            "allowed_arms": [item.to_artifact_dict() for item in self.allowed_arms],
            "image_results": [result.to_json_dict() for result in self.image_results],
        }


def normalize_call_prediction(
    *,
    execution_evidence: ExecutionEvidenceEnvelope,
    parse_score_receipt: CanonicalParseScoreReceipt,
    spatial_coordinate_receipt: SpatialCoordinateReceipt | None = None,
) -> NormalizedCallPrediction:
    """Normalize one canonical parsed/scored row into the source-canvas frame.

    Category, row, box, and score are intentionally not accepted as free
    arguments. Malformed attempts remain in parser diagnostics and cannot enter
    metric-bearing prediction views through this interface.
    """

    validate_parse_score_receipt_association(
        parse_score_receipt,
        execution_evidence=execution_evidence,
    )
    image_id = str(execution_evidence.image_id)
    canonical_call_id = execution_evidence.request_id
    canonical_cell_index = execution_evidence.canonical_cell_index
    source_width = execution_evidence.source_width
    source_height = execution_evidence.source_height
    generated_row_index = parse_score_receipt.generated_row_index
    prediction_id = parse_score_receipt.object_span_id
    normalized_category = parse_score_receipt.normalized_category_name
    finite_score = parse_score_receipt.score
    score_source = parse_score_receipt.score_policy_id
    _require_nonempty_string(prediction_id, field="prediction_id")
    _require_positive_integer(source_width, field="source_width")
    _require_positive_integer(source_height, field="source_height")
    resolved_arm = execution_evidence.arm
    evaluator_category_id = COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME.get(
        normalized_category
    )
    official_coco_category_id = COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME.get(
        normalized_category
    )
    if evaluator_category_id is None or official_coco_category_id is None:
        raise _contract_error(
            "prediction category is outside the frozen COCO-80 ontology",
            "analysis.spatial_merge_category_unknown",
            category_name=normalized_category,
        )
    if score_source != COMPACT_OBJECT_SELECTED_TOKEN_SCORE_VERSION_1:
        raise _contract_error(
            "prediction score source must be the frozen row-local score policy",
            "analysis.spatial_merge_score_source",
            prediction_id=prediction_id,
            score_source=score_source,
        )

    spatial_variant_mode = _spatial_variant_mode(resolved_arm)
    if spatial_variant_mode is not None:
        if canonical_cell_index is None:
            raise _contract_error(
                "spatial prediction requires a canonical cell index",
                "analysis.spatial_merge_cell_missing",
                prediction_id=prediction_id,
            )
        if spatial_coordinate_receipt is None:
            raise _contract_error(
                "spatial prediction requires a coordinate receipt",
                "analysis.spatial_merge_receipt_missing",
                prediction_id=prediction_id,
            )
        if parse_score_receipt.coordinate_source != "spatial_local_canvas":
            raise _contract_error(
                "spatial prediction receipt must use the local spatial frame",
                "analysis.spatial_merge_spatial_coordinate_source",
                prediction_id=prediction_id,
            )
        if (
            execution_evidence.grid_provenance.canonical_spatial_spec_sha256
            != SpatialGridSpec().fingerprint
        ):
            raise _contract_error(
                "spatial execution does not use the frozen merge grid",
                "analysis.spatial_merge_grid_contract",
                prediction_id=prediction_id,
            )
        spatial_grid = SpatialGrid.build(
            source_width=source_width,
            source_height=source_height,
        )
        _validate_grid_frame(spatial_grid, source_width, source_height)
        expected_receipt = spatial_grid.plan(
            cell_index=canonical_cell_index,
            variant_mode=spatial_variant_mode,
        ).coordinate_receipt(spatial_coordinate_receipt.original_coordinate_bins)
        if spatial_coordinate_receipt != expected_receipt:
            raise _contract_error(
                "spatial coordinate receipt does not reproduce from the frozen grid",
                "analysis.spatial_merge_receipt_mismatch",
                prediction_id=prediction_id,
            )
        if (
            spatial_coordinate_receipt.original_coordinate_bins
            != parse_score_receipt.coordinate_bins
            or tuple(
                float(value) for value in spatial_coordinate_receipt.local_integer_box
            )
            != parse_score_receipt.parsed_bbox_xyxy
            or spatial_coordinate_receipt.local_extent_width
            != parse_score_receipt.coordinate_extent_width
            or spatial_coordinate_receipt.local_extent_height
            != parse_score_receipt.coordinate_extent_height
        ):
            raise _contract_error(
                "spatial coordinate receipt differs from canonical parsed coordinates",
                "analysis.spatial_merge_receipt_parse_score",
                prediction_id=prediction_id,
            )
        global_box = tuple(
            float(value)
            for value in spatial_coordinate_receipt.clipped_global_integer_box
        )
        local_box: PixelBox | None = tuple(
            float(value) for value in spatial_coordinate_receipt.local_integer_box
        )
        _validate_global_box(global_box, source_width, source_height)
        ownership = spatial_grid.ownership(global_box)
        if not ownership.is_valid or ownership.owner_cell_index is None:
            raise _contract_error(
                "spatial prediction has no valid source-canvas core owner",
                "analysis.spatial_merge_prediction_unowned",
                prediction_id=prediction_id,
                ownership_reason=ownership.reason,
            )
        owner_cell_index = ownership.owner_cell_index
        ownership_status: OwnershipStatus = (
            "owned" if owner_cell_index == canonical_cell_index else "non_owning"
        )
        receipt = spatial_coordinate_receipt
    else:
        if spatial_coordinate_receipt is not None:
            raise _contract_error(
                "full-image prediction cannot carry a spatial coordinate receipt",
                "analysis.spatial_merge_full_ownership",
                prediction_id=prediction_id,
            )
        if parse_score_receipt.coordinate_source != "source_canvas":
            raise _contract_error(
                "full-image prediction receipt must use the source canvas",
                "analysis.spatial_merge_full_coordinate_source",
                prediction_id=prediction_id,
            )
        if (
            parse_score_receipt.coordinate_extent_width != source_width
            or parse_score_receipt.coordinate_extent_height != source_height
        ):
            raise _contract_error(
                "full-image parsed coordinate frame differs from source canvas",
                "analysis.spatial_merge_full_coordinate_frame",
                prediction_id=prediction_id,
            )
        global_box = _coerce_box(parse_score_receipt.parsed_bbox_xyxy)
        _validate_global_box(global_box, source_width, source_height)
        local_box = None
        owner_cell_index = None
        ownership_status = "not_applicable"
        receipt = None

    return NormalizedCallPrediction(
        image_id=image_id,
        arm=resolved_arm,
        canonical_call_id=canonical_call_id,
        canonical_cell_index=canonical_cell_index,
        generated_row_index=generated_row_index,
        prediction_id=prediction_id,
        normalized_category_name=normalized_category,
        evaluator_category_id=evaluator_category_id,
        official_coco_category_id=official_coco_category_id,
        category_registry_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        global_bbox_xyxy=global_box,
        local_bbox_xyxy=local_box,
        score=finite_score,
        score_source=score_source,
        ownership_status=ownership_status,
        owner_cell_index=owner_cell_index,
        spatial_coordinate_receipt=receipt,
        source_width=source_width,
        source_height=source_height,
        execution_evidence=execution_evidence,
        parse_score_receipt=parse_score_receipt,
    )


def merge_image_predictions(
    *,
    image_id: str,
    arm_identifier: str,
    predictions: Sequence[NormalizedCallPrediction],
    source_width: int | None = None,
    source_height: int | None = None,
) -> ImageMergeResult:
    """Apply ownership filtering and frozen kept-only class-wise greedy NMS."""

    resolved_arm = primary_arm_definition(arm_identifier)
    raw = tuple(predictions)
    resolved_width, resolved_height = _resolve_image_frame(
        predictions=raw,
        source_width=source_width,
        source_height=source_height,
    )
    raw_any = PredictionSet(image_id, resolved_arm, "raw_any_call", raw)
    owning = tuple(item for item in raw if item.ownership_status != "non_owning")
    non_owning = tuple(item for item in raw if item.ownership_status == "non_owning")
    raw_owning = PredictionSet(image_id, resolved_arm, "raw_owning_call", owning)
    non_owning_view = PredictionSet(
        image_id, resolved_arm, "non_owning_diagnostic", non_owning
    )
    pre_merge = PredictionSet(image_id, resolved_arm, "pre_merge", owning)
    ordered = tuple(sorted(owning, key=lambda item: item.deterministic_rank_key))
    suppressed: set[str] = set()
    kept: list[NormalizedCallPrediction] = []
    edges: list[SuppressionEdge] = []
    for index, candidate in enumerate(ordered):
        if candidate.prediction_id in suppressed:
            continue
        kept.append(candidate)
        for later in ordered[index + 1 :]:
            if later.prediction_id in suppressed:
                continue
            if later.normalized_category_name != candidate.normalized_category_name:
                continue
            overlap = iou_xyxy(candidate.global_bbox_xyxy, later.global_bbox_xyxy)
            if overlap < NON_MAXIMUM_SUPPRESSION_IOU_THRESHOLD:
                continue
            suppressed.add(later.prediction_id)
            edges.append(
                SuppressionEdge(
                    suppressor_prediction_id=candidate.prediction_id,
                    suppressed_prediction_id=later.prediction_id,
                    normalized_category_name=candidate.normalized_category_name,
                    intersection_over_union=overlap,
                )
            )
    post_merge = PredictionSet(image_id, resolved_arm, "post_merge", tuple(kept))
    return ImageMergeResult(
        image_id=image_id,
        arm=resolved_arm,
        source_width=resolved_width,
        source_height=resolved_height,
        raw_any_call=raw_any,
        raw_owning_call=raw_owning,
        non_owning_diagnostic=non_owning_view,
        pre_merge=pre_merge,
        post_merge=post_merge,
        suppression_edges=tuple(edges),
        pre_merge_strict_duplicate_components=_strict_duplicate_components(pre_merge),
        post_merge_strict_duplicate_components=_strict_duplicate_components(post_merge),
    )


def merge_arm_predictions(
    *,
    arm_identifier: str,
    predictions: Sequence[NormalizedCallPrediction],
    expected_images: Sequence[ExpectedImageFrame],
    allowed_arm_identifiers: Sequence[str],
) -> ArmMergeResult:
    """Merge one explicitly allowed arm over an exact sealed image universe."""

    allowed_arms = _resolve_allowed_arms(allowed_arm_identifiers)
    resolved_arm = primary_arm_definition(arm_identifier)
    if resolved_arm not in allowed_arms:
        raise _contract_error(
            "arm is not authorized by the sealed evidence-assembly schedule",
            "analysis.spatial_merge_arm_disallowed",
            arm_identifier=arm_identifier,
            allowed_arm_identifiers=tuple(arm.arm_code for arm in allowed_arms),
        )
    universe = _validate_expected_images(expected_images)
    expected_by_id = {item.image_id: item for item in universe}
    grouped: dict[str, list[NormalizedCallPrediction]] = defaultdict(list)
    prediction_identifiers: set[str] = set()
    for prediction in predictions:
        if prediction.arm != resolved_arm:
            raise _contract_error(
                "prediction belongs to a different arm",
                "analysis.spatial_merge_input_arm",
                prediction_id=prediction.prediction_id,
            )
        if prediction.prediction_id in prediction_identifiers:
            raise _contract_error(
                "prediction identifiers must be unique across an arm result",
                "analysis.spatial_merge_duplicate_prediction_id",
                prediction_id=prediction.prediction_id,
            )
        prediction_identifiers.add(prediction.prediction_id)
        expected = expected_by_id.get(prediction.image_id)
        if expected is None:
            raise _contract_error(
                "prediction image is outside the sealed image universe",
                "analysis.spatial_merge_unknown_prediction_image",
                prediction_id=prediction.prediction_id,
                image_id=prediction.image_id,
            )
        if (
            prediction.source_width != expected.source_width
            or prediction.source_height != expected.source_height
        ):
            raise _contract_error(
                "prediction source frame differs from its sealed image frame",
                "analysis.spatial_merge_prediction_universe_frame",
                prediction_id=prediction.prediction_id,
                image_id=prediction.image_id,
                expected_source_frame=(
                    expected.source_width,
                    expected.source_height,
                ),
                actual_source_frame=(
                    prediction.source_width,
                    prediction.source_height,
                ),
            )
        grouped[prediction.image_id].append(prediction)
    return ArmMergeResult(
        arm=resolved_arm,
        expected_images=universe,
        allowed_arms=allowed_arms,
        image_results=tuple(
            merge_image_predictions(
                image_id=expected.image_id,
                arm_identifier=arm_identifier,
                predictions=grouped[expected.image_id],
                source_width=expected.source_width,
                source_height=expected.source_height,
            )
            for expected in universe
        ),
    )


def _strict_duplicate_components(
    prediction_set: PredictionSet,
) -> tuple[StrictDuplicateComponent, ...]:
    ordered = tuple(
        sorted(prediction_set.predictions, key=lambda item: item.deterministic_rank_key)
    )
    by_id = {item.prediction_id: item for item in ordered}
    adjacency: dict[str, set[str]] = {item.prediction_id: set() for item in ordered}
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if left.normalized_category_name != right.normalized_category_name:
                continue
            if (
                iou_xyxy(left.global_bbox_xyxy, right.global_bbox_xyxy)
                < STRICT_DUPLICATE_IOU_THRESHOLD
            ):
                continue
            adjacency[left.prediction_id].add(right.prediction_id)
            adjacency[right.prediction_id].add(left.prediction_id)
    rank = {item.prediction_id: index for index, item in enumerate(ordered)}
    visited: set[str] = set()
    components: list[StrictDuplicateComponent] = []
    for prediction in ordered:
        if prediction.prediction_id in visited:
            continue
        frontier = [prediction.prediction_id]
        members: list[str] = []
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            members.append(current)
            frontier.extend(adjacency[current] - visited)
        if len(members) < 2:
            continue
        ordered_members = tuple(sorted(members, key=rank.__getitem__))
        components.append(
            StrictDuplicateComponent(
                representative_prediction_id=ordered_members[0],
                member_prediction_ids=ordered_members,
                normalized_category_name=by_id[
                    ordered_members[0]
                ].normalized_category_name,
            )
        )
    return tuple(components)


def _strict_duplicate_rate(
    prediction_set: PredictionSet,
    components: Sequence[StrictDuplicateComponent],
) -> float | None:
    denominator = len(prediction_set.predictions)
    if denominator == 0:
        return None
    numerator = sum(
        len(component.member_prediction_ids) - 1 for component in components
    )
    return numerator / denominator


def _spatial_variant_mode(
    arm: ResearchArmDefinition,
) -> SpatialVariantMode | None:
    """Compatibility wrapper over the single spatial-arm interpretation."""

    return spatial_variant_mode_for_arm(arm)


def _resolve_allowed_arms(
    identifiers: Sequence[str],
) -> tuple[ResearchArmDefinition, ...]:
    if isinstance(identifiers, (str, bytes)) or not identifiers:
        raise _contract_error(
            "evidence assembly requires a nonempty allowed-arm schedule",
            "analysis.spatial_merge_allowed_arms_missing",
        )
    seen: set[str] = set()
    resolved: list[ResearchArmDefinition] = []
    for identifier in identifiers:
        definition = primary_arm_definition(identifier)
        if definition.arm_code in seen:
            raise _contract_error(
                "allowed-arm schedule contains a duplicate arm",
                "analysis.spatial_merge_allowed_arm_duplicate",
                arm_identifier=definition.arm_code,
            )
        seen.add(definition.arm_code)
        resolved.append(definition)
    return tuple(sorted(resolved, key=lambda item: item.arm_code))


def _validate_expected_images(
    expected_images: Sequence[ExpectedImageFrame],
) -> tuple[ExpectedImageFrame, ...]:
    if isinstance(expected_images, (str, bytes)) or not expected_images:
        raise _contract_error(
            "evidence assembly requires a nonempty sealed image universe",
            "analysis.spatial_merge_expected_universe_missing",
        )
    seen: set[str] = set()
    validated: list[ExpectedImageFrame] = []
    for item in expected_images:
        if not isinstance(item, ExpectedImageFrame):
            raise _contract_error(
                "sealed image universe entries must be ExpectedImageFrame records",
                "analysis.spatial_merge_expected_universe_entry",
                entry_type=type(item).__name__,
            )
        if item.image_id in seen:
            raise _contract_error(
                "sealed image universe contains a duplicate image identifier",
                "analysis.spatial_merge_expected_universe_duplicate",
                image_id=item.image_id,
            )
        seen.add(item.image_id)
        validated.append(item)
    return tuple(sorted(validated, key=lambda item: item.image_id))


def _resolve_image_frame(
    *,
    predictions: Sequence[NormalizedCallPrediction],
    source_width: int | None,
    source_height: int | None,
) -> tuple[int, int]:
    if (source_width is None) != (source_height is None):
        raise _contract_error(
            "source-canvas width and height must be supplied together",
            "analysis.spatial_merge_image_frame_partial",
        )
    if source_width is None or source_height is None:
        if not predictions:
            raise _contract_error(
                "an empty image merge requires its sealed source-canvas frame",
                "analysis.spatial_merge_empty_image_frame_missing",
            )
        source_width = predictions[0].source_width
        source_height = predictions[0].source_height
    _require_positive_integer(source_width, field="source_width")
    _require_positive_integer(source_height, field="source_height")
    for prediction in predictions:
        if (
            prediction.source_width != source_width
            or prediction.source_height != source_height
        ):
            raise _contract_error(
                "prediction source frame differs from the image merge frame",
                "analysis.spatial_merge_image_frame_mismatch",
                prediction_id=prediction.prediction_id,
                expected_source_frame=(source_width, source_height),
                actual_source_frame=(
                    prediction.source_width,
                    prediction.source_height,
                ),
            )
    return cast(int, source_width), cast(int, source_height)


def _receipt_to_json(receipt: SpatialCoordinateReceipt | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    return {
        "variant_mode": receipt.variant_mode,
        "cell_index": receipt.cell_index,
        "original_coordinate_bins": list(receipt.original_coordinate_bins),
        "local_extent_width": receipt.local_extent_width,
        "local_extent_height": receipt.local_extent_height,
        "local_integer_box": list(receipt.local_integer_box),
        "tile_origin_xy": list(receipt.tile_origin_xy),
        "unclipped_global_integer_box": list(receipt.unclipped_global_integer_box),
        "clipped_global_integer_box": list(receipt.clipped_global_integer_box),
        "source_width": receipt.source_width,
        "source_height": receipt.source_height,
    }


def _validate_grid_frame(grid: SpatialGrid, width: int, height: int) -> None:
    if grid.source_width != width or grid.source_height != height:
        raise _contract_error(
            "spatial grid and prediction source-canvas frames differ",
            "analysis.spatial_merge_grid_frame",
            source_width=width,
            source_height=height,
            grid_width=grid.source_width,
            grid_height=grid.source_height,
        )


def _coerce_box(values: Sequence[float]) -> PixelBox:
    if isinstance(values, (str, bytes)) or len(values) != 4:
        raise _contract_error(
            "bounding box must contain exactly four coordinates",
            "analysis.spatial_merge_bbox_shape",
        )
    parsed: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _contract_error(
                "bounding-box coordinates must be numeric",
                "analysis.spatial_merge_bbox_type",
            )
        number = float(value)
        if not math.isfinite(number):
            raise _contract_error(
                "bounding-box coordinates must be finite",
                "analysis.spatial_merge_bbox_nonfinite",
            )
        parsed.append(number)
    return (parsed[0], parsed[1], parsed[2], parsed[3])


def _validate_global_box(box: PixelBox, width: int, height: int) -> None:
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise _contract_error(
            "prediction bounding box must have positive area",
            "analysis.spatial_merge_bbox_empty",
            bbox_xyxy=box,
        )
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise _contract_error(
            "prediction bounding box must lie inside the source canvas",
            "analysis.spatial_merge_bbox_outside",
            bbox_xyxy=box,
            source_width=width,
            source_height=height,
        )


def _require_nonempty_string(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise _contract_error(
            f"{field} must be a nonempty string",
            "analysis.spatial_merge_identifier",
            field=field,
        )


def _require_nonnegative_integer(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _contract_error(
            f"{field} must be a nonnegative integer",
            "analysis.spatial_merge_nonnegative_integer",
            field=field,
            value=value,
        )


def _require_positive_integer(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _contract_error(
            f"{field} must be a positive integer",
            "analysis.spatial_merge_positive_integer",
            field=field,
            value=value,
        )


def _contract_error(message: str, code: str, **context: Any) -> DataContractError:
    return DataContractError(message, code=code, context=context)
