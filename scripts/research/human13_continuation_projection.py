"""Pure forced-continuation projection for the Human-13 successor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

from scripts.research.build_human13_k_union_manifest import ImageRecord
from scripts.research.build_human13_on_policy_frontier import FrontierImage
from scripts.research.compare_clean_rollout_owner_coverage import (
    _global_matches,
    iou_xyxy,
)
from scripts.research.human13_forced_continuation import ForcedContinuationResult
from scripts.research.human13_frontier_selection import (
    CandidateScore,
    ContinuationOutcome,
    protected_owner_coverable,
    select_continuation,
)
from src.eval.detection_categories import normalize_coco_category_name


_DUPLICATE_IOU = 0.95
_OWNER_IOU = 0.5
_PARSE_STATUSES = {"accepted", "accepted_with_drops", "all_spans_dropped", "empty"}
_TERMINATION_STATUSES = {"natural_im_end", "cap_hit", "nonterminal_return"}


@dataclass(frozen=True)
class BranchPrediction:
    chronological_index: int
    origin: Literal["current", "forced", "released"]
    category: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ContinuationProjection:
    score: CandidateScore
    result: ForcedContinuationResult
    outcome: ContinuationOutcome
    matched_owner_ids: tuple[str, ...]
    full_branch_predictions: tuple[BranchPrediction, ...]
    retained_prediction_indices: tuple[int, ...]
    duplicate_prediction_indices: tuple[int, ...]
    duplicate_count: int
    duplicate_increase: int
    malformed_count: int
    row_count: int
    generated_tokens: int


def project_continuation(
    image: ImageRecord,
    frontier: FrontierImage,
    score: CandidateScore,
    result: ForcedContinuationResult,
) -> ContinuationProjection:
    """Compose and audit one forced row plus its released natural continuation."""

    _validate_binding(image, frontier, score, result)
    forced, forced_dropped = _parse_artifact(
        result.forced_row_parse_evidence,
        label="forced row",
    )
    if len(forced) != 1 or forced_dropped:
        raise ValueError("forced row must parse as exactly one clean prediction")
    released, released_dropped = _parse_artifact(
        result.parse_evidence,
        label="released continuation",
    )

    current = tuple(
        (row.category, _valid_bbox(row.bbox, label="current prediction"))
        for row in sorted(frontier.rows, key=lambda item: item.generated_order)
    )
    raw_predictions = (
        tuple((category, bbox, "current") for category, bbox in current)
        + ((forced[0][0], forced[0][1], "forced"),)
        + tuple((category, bbox, "released") for category, bbox in released)
    )
    branch = tuple(
        BranchPrediction(index, origin, category, bbox)  # type: ignore[arg-type]
        for index, (category, bbox, origin) in enumerate(raw_predictions)
    )
    retained, duplicates = _chronological_dedup(branch)
    current_retained, current_duplicates = _chronological_dedup(
        branch[: len(current)]
    )
    if len(current_duplicates) != len(frontier.duplicate_events):
        raise ValueError("current duplicate projection differs from frontier")

    owners = tuple(
        sorted(image.owners, key=lambda item: (item.source_object_index, item.owner_id))
    )
    owner_rows = tuple(
        (
            owner.owner_id,
            normalize_coco_category_name(owner.category),
            _valid_bbox(owner.bbox, label=f"owner {owner.owner_id}"),
        )
        for owner in owners
    )
    if len({owner_id for owner_id, _, _ in owner_rows}) != len(owner_rows):
        raise ValueError("owner IDs must be unique")
    current_matched = _matched_owner_ids(owner_rows, current_retained)
    if current_matched != tuple(sorted(frontier.canonical_owner_ids)):
        raise ValueError("current canonical owner set differs from frontier")
    matched_owner_ids = _matched_owner_ids(owner_rows, retained)

    forced_row = forced[0]
    selected_owner = next(
        (owner for owner in owner_rows if owner[0] == score.path.owner_id), None
    )
    if selected_owner is None:
        raise ValueError("selected owner is absent from image")
    if (
        normalize_coco_category_name(forced_row[0]) != selected_owner[1]
        or iou_xyxy(forced_row[1], selected_owner[2]) < _OWNER_IOU
    ):
        raise ValueError("forced prediction does not match selected owner")

    retained_rows = tuple((row.category, row.bbox) for row in retained)
    protected = protected_owner_coverable(
        owner_rows,
        retained_rows,
        protected_owner_ids=frontier.constrained_protected_owner_ids,
        threshold=_OWNER_IOU,
    )
    duplicate_count = len(duplicates)
    duplicate_increase = duplicate_count - len(current_duplicates)
    if duplicate_increase < 0:
        raise ValueError("continuation duplicate burden cannot shrink its fixed prefix")
    malformed_count = len(released_dropped)
    row_count = len(branch)
    generated_tokens = len(result.forced_row_token_ids) + len(result.released_token_ids)
    unique_owner_delta = len(matched_owner_ids) - len(current_matched)
    outcome = ContinuationOutcome(
        owner_id=score.path.owner_id,
        hf_barrier=float(score.hf_barrier),
        protected_coverable=protected,
        unique_owner_delta=unique_owner_delta,
        termination_status=result.termination_status,
        cap_hit=result.cap_hit,
        duplicate_increase=duplicate_increase,
        malformed_increase=malformed_count,
        row_count=row_count,
        generated_tokens=generated_tokens,
    )
    return ContinuationProjection(
        score=score,
        result=result,
        outcome=outcome,
        matched_owner_ids=matched_owner_ids,
        full_branch_predictions=branch,
        retained_prediction_indices=tuple(row.chronological_index for row in retained),
        duplicate_prediction_indices=tuple(
            row.chronological_index for row in duplicates
        ),
        duplicate_count=duplicate_count,
        duplicate_increase=duplicate_increase,
        malformed_count=malformed_count,
        row_count=row_count,
        generated_tokens=generated_tokens,
    )


def select_projected_continuation(
    projections: Sequence[ContinuationProjection],
) -> ContinuationProjection:
    """Apply the canonical selector while retaining the bound score/result pair."""

    if not projections:
        raise ValueError("no continuation projections")
    selected = select_continuation(tuple(item.outcome for item in projections))
    return next(item for item in projections if item.outcome is selected)


def _validate_binding(
    image: ImageRecord,
    frontier: FrontierImage,
    score: CandidateScore,
    result: ForcedContinuationResult,
) -> None:
    if image.image_id != frontier.image_id or score.path.image_id != image.image_id:
        raise ValueError("image, frontier, and candidate identities differ")
    if not math.isfinite(float(score.hf_barrier)) or score.hf_barrier < 0:
        raise ValueError("candidate HF barrier must be finite and nonnegative")
    alias = next(
        (
            item
            for item in frontier.candidate_aliases
            if item.owner_id == score.path.owner_id
            and item.row_id == score.path.alias_id
        ),
        None,
    )
    if alias is None or tuple(alias.token_ids) != tuple(score.path.token_ids):
        raise ValueError("candidate is absent from current frontier aliases")
    manifest_alias = next(
        (
            item
            for item in image.selected_rows
            if item.owner_id == score.path.owner_id
            and item.row_id == score.path.alias_id
        ),
        None,
    )
    if (
        manifest_alias is None
        or manifest_alias.trajectory_id != alias.trajectory_id
        or manifest_alias.seed != alias.seed
        or tuple(manifest_alias.token_ids) != tuple(alias.token_ids)
    ):
        raise ValueError("frontier candidate differs from its manifest alias")
    if score.path.owner_id not in frontier.uncovered_h_owner_ids:
        raise ValueError("candidate is outside the current uncovered H frontier")
    if tuple(result.forced_row_token_ids) != tuple(score.path.token_ids):
        raise ValueError("forced row differs from selected candidate")
    _valid_tokens(result.forced_row_token_ids, label="forced row")
    _valid_tokens(result.released_token_ids, label="released continuation")
    if result.termination_status not in _TERMINATION_STATUSES:
        raise ValueError("forced continuation termination status is invalid")
    if result.cap_hit != (result.termination_status == "cap_hit"):
        raise ValueError("forced continuation cap status is inconsistent")


def _valid_tokens(token_ids: Sequence[int], *, label: str) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in token_ids):
        raise ValueError(f"{label} token IDs are invalid")


def _parse_artifact(
    artifact: Mapping[str, Any],
    *,
    label: str,
) -> tuple[
    tuple[tuple[str, tuple[float, float, float, float]], ...],
    tuple[Mapping[str, Any], ...],
]:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{label} parse evidence is invalid")
    predictions_raw = artifact.get("predictions")
    dropped_raw = artifact.get("dropped_predictions")
    status = artifact.get("parse_status")
    if (
        status not in _PARSE_STATUSES
        or not isinstance(predictions_raw, list)
        or not isinstance(dropped_raw, list)
        or any(not isinstance(item, Mapping) for item in dropped_raw)
    ):
        raise ValueError(f"{label} parse evidence is invalid")
    expected_status = (
        "accepted_with_drops"
        if predictions_raw and dropped_raw
        else "accepted"
        if predictions_raw
        else "all_spans_dropped"
        if dropped_raw
        else "empty"
    )
    if status != expected_status:
        raise ValueError(f"{label} parse status is inconsistent")
    predictions: list[tuple[str, tuple[float, float, float, float]]] = []
    prior_generated_order = -1
    for index, raw in enumerate(predictions_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} prediction is invalid")
        generated_order = raw.get("generated_order")
        if (
            isinstance(generated_order, bool)
            or not isinstance(generated_order, int)
            or generated_order <= prior_generated_order
        ):
            raise ValueError(f"{label} prediction generated order is invalid")
        prior_generated_order = generated_order
        category = raw.get("description")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{label} prediction category is invalid")
        predictions.append(
            (
                category,
                _valid_bbox(raw.get("bbox"), label=f"{label} prediction {index}"),
            )
        )
    return tuple(predictions), tuple(dropped_raw)


def _valid_bbox(
    raw: Any,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"{label} bbox is invalid")
    bbox = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in bbox):
        raise ValueError(f"{label} bbox must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{label} bbox must be a valid rectangle")
    return bbox  # type: ignore[return-value]


def _chronological_dedup(
    predictions: Sequence[BranchPrediction],
) -> tuple[tuple[BranchPrediction, ...], tuple[BranchPrediction, ...]]:
    retained: list[BranchPrediction] = []
    duplicates: list[BranchPrediction] = []
    for prediction in predictions:
        if any(
            iou_xyxy(prior.bbox, prediction.bbox) > _DUPLICATE_IOU
            for prior in retained
        ):
            duplicates.append(prediction)
        else:
            retained.append(prediction)
    return tuple(retained), tuple(duplicates)


def _matched_owner_ids(
    owners: Sequence[tuple[str, str, tuple[float, float, float, float]]],
    predictions: Sequence[BranchPrediction],
) -> tuple[str, ...]:
    gt = [(category, bbox) for _, category, bbox in owners]
    pred = [
        (normalize_coco_category_name(row.category), row.bbox) for row in predictions
    ]
    matches = _global_matches(gt, pred, _OWNER_IOU)
    return tuple(sorted(owners[owner_index][0] for owner_index, _, _ in matches))


__all__ = [
    "BranchPrediction",
    "ContinuationProjection",
    "project_continuation",
    "select_projected_continuation",
]
