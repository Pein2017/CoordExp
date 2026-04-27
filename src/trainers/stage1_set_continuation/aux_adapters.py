from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch


CoordAuxHelper = Callable[..., torch.Tensor | Mapping[str, Any]]
BBoxAuxHelper = Callable[..., torch.Tensor | Mapping[str, Any]]


@dataclass(frozen=True)
class BBoxGeoDecodedState:
    pred_boxes_xyxy: torch.Tensor
    target_boxes_xyxy: torch.Tensor | None = None
    box_weights: torch.Tensor | None = None


@dataclass(frozen=True)
class BranchCandidateAuxState:
    candidate_name: str
    scored_valid: bool
    coord_logits: torch.Tensor | None = None
    coord_logits_full: torch.Tensor | None = None
    coord_target_bins: torch.Tensor | None = None
    coord_slot_mask: torch.Tensor | None = None
    bbox_geo_decoded_state: BBoxGeoDecodedState | None = None


@dataclass(frozen=True)
class CandidateAuxResult:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    state_update: dict[str, Any] = field(default_factory=dict)
    is_skipped: bool = False
    skip_reason: str | None = None

    @classmethod
    def skipped(
        cls,
        loss: torch.Tensor,
        *,
        reason: str,
        metrics: Mapping[str, float] | None = None,
    ) -> "CandidateAuxResult":
        return cls(
            loss=loss,
            metrics=dict(metrics or {}),
            state_update={},
            is_skipped=True,
            skip_reason=str(reason),
        )


@dataclass(frozen=True)
class AggregatedAuxResult:
    loss: torch.Tensor
    metrics: dict[str, float]
    scored_valid_candidates: int
    contributing_candidates: int
    skipped_candidates: int


def _normalize_weighting_mode(weighting_mode: str) -> str:
    mode = str(weighting_mode or "uniform").strip().lower()
    if mode in {"uniform", "mean"}:
        return "uniform"
    if mode in {"responsibility", "responsibility_weighted", "mp"}:
        raise ValueError(
            "responsibility-weighted auxiliary losses are unavailable in v1"
        )
    raise ValueError(f"unknown auxiliary weighting mode: {mode}")


def _coerce_loss_result(
    raw: torch.Tensor | Mapping[str, Any],
    *,
    default_state: Mapping[str, Any] | None = None,
) -> CandidateAuxResult:
    if isinstance(raw, torch.Tensor):
        return CandidateAuxResult(
            loss=raw,
            state_update=dict(default_state or {}),
        )
    if isinstance(raw, Mapping):
        loss = raw.get("loss")
        if not isinstance(loss, torch.Tensor):
            raise TypeError("aux helper mapping outputs must include a tensor 'loss'")
        metrics_raw = raw.get("metrics", {})
        metrics = (
            {str(key): float(value) for key, value in dict(metrics_raw).items()}
            if isinstance(metrics_raw, Mapping)
            else {}
        )
        state_update = dict(default_state or {})
        extra_state = raw.get("state_update", {})
        if isinstance(extra_state, Mapping):
            state_update.update(extra_state)
        return CandidateAuxResult(loss=loss, metrics=metrics, state_update=state_update)
    raise TypeError("aux helper must return a tensor loss or a mapping with 'loss'")


class CoordAuxAdapter:
    def __init__(
        self,
        *,
        enabled: bool,
        helper: CoordAuxHelper | None,
        weighting_mode: str = "uniform",
    ) -> None:
        self.enabled = bool(enabled)
        self.helper = helper
        self.weighting_mode = _normalize_weighting_mode(weighting_mode)

    def compute_candidate(
        self, candidate: BranchCandidateAuxState
    ) -> CandidateAuxResult:
        zero = _candidate_zero(candidate)
        if not self.enabled:
            return CandidateAuxResult.skipped(zero, reason="disabled")
        if self.helper is None:
            raise ValueError("coord aux adapter helper is required when enabled")
        if not isinstance(candidate.coord_logits, torch.Tensor):
            raise ValueError("coord aux adapter requires branch-local coord_logits")
        if not isinstance(candidate.coord_logits_full, torch.Tensor):
            raise ValueError(
                "coord aux adapter requires branch-local coord_logits_full"
            )
        if not isinstance(candidate.coord_target_bins, torch.Tensor):
            raise ValueError(
                "coord aux adapter requires branch-local coord_target_bins"
            )
        if not isinstance(candidate.coord_slot_mask, torch.Tensor):
            raise ValueError("coord aux adapter requires branch-local coord_slot_mask")
        raw = self.helper(
            coord_logits=candidate.coord_logits,
            coord_logits_full=candidate.coord_logits_full,
            coord_target_bins=candidate.coord_target_bins,
            coord_slot_mask=candidate.coord_slot_mask,
        )
        return _coerce_loss_result(raw)


class BBoxGeoAuxAdapter:
    def __init__(
        self,
        *,
        enabled: bool,
        helper: BBoxAuxHelper | None,
        weighting_mode: str = "uniform",
    ) -> None:
        self.enabled = bool(enabled)
        self.helper = helper
        self.weighting_mode = _normalize_weighting_mode(weighting_mode)

    def compute_candidate(
        self, candidate: BranchCandidateAuxState
    ) -> CandidateAuxResult:
        zero = _candidate_zero(candidate)
        if not self.enabled:
            return CandidateAuxResult.skipped(zero, reason="disabled")
        if self.helper is None:
            raise ValueError("bbox geo adapter helper is required when enabled")
        decoded = candidate.bbox_geo_decoded_state
        if decoded is None:
            raise ValueError("bbox geo adapter requires decoded bbox state")
        raw = self.helper(
            pred_boxes_xyxy=decoded.pred_boxes_xyxy,
            target_boxes_xyxy=decoded.target_boxes_xyxy,
            box_weights=decoded.box_weights,
        )
        return _coerce_loss_result(
            raw, default_state={"bbox_geo_decoded_state": decoded}
        )


class BBoxSizeAuxAdapter:
    def __init__(
        self,
        *,
        enabled: bool,
        helper: BBoxAuxHelper | None,
        weighting_mode: str = "uniform",
    ) -> None:
        self.enabled = bool(enabled)
        self.helper = helper
        self.weighting_mode = _normalize_weighting_mode(weighting_mode)

    def compute_candidate(
        self, candidate: BranchCandidateAuxState
    ) -> CandidateAuxResult:
        zero = _candidate_zero(candidate)
        if not self.enabled:
            return CandidateAuxResult.skipped(zero, reason="disabled")
        if self.helper is None:
            raise ValueError("bbox size aux adapter helper is required when enabled")
        decoded = candidate.bbox_geo_decoded_state
        if decoded is None:
            raise ValueError("bbox size aux requires bbox_geo decoded state")
        raw = self.helper(
            pred_boxes_xyxy=decoded.pred_boxes_xyxy,
            target_boxes_xyxy=decoded.target_boxes_xyxy,
            box_weights=decoded.box_weights,
        )
        return _coerce_loss_result(raw)


def aggregate_candidate_aux_losses(
    *,
    candidates: list[BranchCandidateAuxState],
    compute_candidate: Callable[[BranchCandidateAuxState], CandidateAuxResult],
    metric_prefix: str,
) -> AggregatedAuxResult:
    scored_valid = 0
    contributing = 0
    skipped = 0
    total_loss: torch.Tensor | None = None

    for candidate in candidates:
        if not bool(candidate.scored_valid):
            continue
        scored_valid += 1
        result = compute_candidate(candidate)
        if result.is_skipped:
            skipped += 1
            if total_loss is None:
                total_loss = result.loss
            continue
        contributing += 1
        total_loss = result.loss if total_loss is None else total_loss + result.loss

    if total_loss is None:
        total_loss = torch.tensor(0.0, dtype=torch.float32)

    if contributing > 0:
        loss = total_loss / float(contributing)
    else:
        loss = total_loss * 0.0

    prefix = str(metric_prefix).strip().rstrip("/")
    metrics = {
        f"{prefix}/scored_valid_candidates": float(scored_valid),
        f"{prefix}/contributing_candidates": float(contributing),
        f"{prefix}/skipped_candidates": float(skipped),
    }
    return AggregatedAuxResult(
        loss=loss,
        metrics=metrics,
        scored_valid_candidates=scored_valid,
        contributing_candidates=contributing,
        skipped_candidates=skipped,
    )


def _candidate_zero(candidate: BranchCandidateAuxState) -> torch.Tensor:
    for value in (
        candidate.coord_logits,
        candidate.coord_logits_full,
        candidate.coord_target_bins,
    ):
        if isinstance(value, torch.Tensor):
            return value.new_tensor(0.0, dtype=torch.float32)
    decoded = candidate.bbox_geo_decoded_state
    if decoded is not None and isinstance(decoded.pred_boxes_xyxy, torch.Tensor):
        return decoded.pred_boxes_xyxy.new_tensor(0.0, dtype=torch.float32)
    return torch.tensor(0.0, dtype=torch.float32)


__all__ = [
    "AggregatedAuxResult",
    "BBoxGeoAuxAdapter",
    "BBoxGeoDecodedState",
    "BBoxSizeAuxAdapter",
    "BranchCandidateAuxState",
    "CandidateAuxResult",
    "CoordAuxAdapter",
    "aggregate_candidate_aux_losses",
]
