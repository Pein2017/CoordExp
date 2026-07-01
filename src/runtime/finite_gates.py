"""Finite scalar and gradient gate decisions for training runtime."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import RuntimeContractError
from src.losses import LossBundle


@dataclass(frozen=True)
class RankScalarFiniteReport:
    planned_step_id: int
    rank: int
    world_size: int
    total_loss_finite: bool
    term_finite: dict[str, bool]
    term_weighted_losses: dict[str, float | None]
    term_raw_losses: dict[str, float | None]
    term_selected_counts: dict[str, int]
    term_eligible_segment_counts: dict[str, int]
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def from_loss_bundle(
        cls,
        bundle: LossBundle,
        *,
        planned_step_id: int,
        rank: int,
        world_size: int,
    ) -> "RankScalarFiniteReport":
        term_finite: dict[str, bool] = {}
        term_weighted: dict[str, float | None] = {}
        term_raw: dict[str, float | None] = {}
        term_selected: dict[str, int] = {}
        term_segments: dict[str, int] = {}
        for term in bundle.terms:
            term_finite[term.name] = _tensor_is_finite(term.weighted_loss)
            term_weighted[term.name] = _optional_float(term.weighted_loss)
            term_raw[term.name] = _optional_float(term.raw_loss)
            term_selected[term.name] = int(term.selected_count)
            term_segments[term.name] = int(term.denominator.eligible_segment_count)
        return cls(
            planned_step_id=planned_step_id,
            rank=rank,
            world_size=world_size,
            total_loss_finite=_tensor_is_finite(bundle.total_loss),
            term_finite=term_finite,
            term_weighted_losses=term_weighted,
            term_raw_losses=term_raw,
            term_selected_counts=term_selected,
            term_eligible_segment_counts=term_segments,
        )

    @classmethod
    def from_error(
        cls,
        *,
        planned_step_id: int,
        rank: int,
        world_size: int,
        error_code: str,
        error_message: str,
        term_eligible_segment_counts: Mapping[str, int],
    ) -> "RankScalarFiniteReport":
        return cls(
            planned_step_id=planned_step_id,
            rank=rank,
            world_size=world_size,
            total_loss_finite=False,
            term_finite={
                term: False for term in sorted(term_eligible_segment_counts)
            },
            term_weighted_losses={
                term: None for term in sorted(term_eligible_segment_counts)
            },
            term_raw_losses={
                term: None for term in sorted(term_eligible_segment_counts)
            },
            term_selected_counts={
                term: 0 for term in sorted(term_eligible_segment_counts)
            },
            term_eligible_segment_counts={
                str(term): int(count)
                for term, count in term_eligible_segment_counts.items()
            },
            error_code=error_code,
            error_message=error_message,
        )

    def is_safe(self) -> bool:
        return (
            self.error_code is None
            and self.total_loss_finite
            and all(self.term_finite.values())
        )

    def to_diagnostic_dict(self) -> dict[str, Any]:
        terms: dict[str, dict[str, Any]] = {}
        term_names = sorted(
            set(self.term_finite)
            | set(self.term_weighted_losses)
            | set(self.term_raw_losses)
            | set(self.term_selected_counts)
            | set(self.term_eligible_segment_counts)
        )
        for term in term_names:
            terms[term] = {
                "finite": self.term_finite.get(term, False),
                "weighted_loss": self.term_weighted_losses.get(term),
                "raw_loss": self.term_raw_losses.get(term),
                "selected_count": self.term_selected_counts.get(term, 0),
                "eligible_segments": self.term_eligible_segment_counts.get(term, 0),
            }
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "planned_step_id": self.planned_step_id,
            "total_loss_finite": self.total_loss_finite,
            "terms": terms,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class RankGradientFiniteReport:
    planned_step_id: int
    rank: int
    world_size: int
    gradients_finite: bool
    backend_overflow: bool
    grad_norm: float | None

    def is_safe(self) -> bool:
        return (
            self.gradients_finite
            and not self.backend_overflow
            and self.grad_norm is not None
            and math.isfinite(float(self.grad_norm))
        )

    def to_diagnostic_dict(self) -> dict[str, Any]:
        grad_norm_finite = (
            self.grad_norm is not None and math.isfinite(float(self.grad_norm))
        )
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "planned_step_id": self.planned_step_id,
            "gradients_finite": self.gradients_finite,
            "backend_overflow": self.backend_overflow,
            "grad_norm": float(self.grad_norm) if grad_norm_finite else None,
            "grad_norm_finite": grad_norm_finite,
        }


@dataclass(frozen=True)
class GateDecision:
    stage: str
    planned_step_id: int
    world_size: int
    ranks: tuple[int, ...]
    all_ranks_safe: bool
    should_call_backward: bool
    should_call_optimizer_step: bool
    should_clear_gradients: bool
    optimizer_update_status: str
    finite_status: str
    reason_codes: tuple[str, ...]
    rank_diagnostics: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "planned_step_id": self.planned_step_id,
            "world_size": self.world_size,
            "ranks": list(self.ranks),
            "all_ranks_safe": self.all_ranks_safe,
            "should_call_backward": self.should_call_backward,
            "should_call_optimizer_step": self.should_call_optimizer_step,
            "should_clear_gradients": self.should_clear_gradients,
            "optimizer_update_status": self.optimizer_update_status,
            "finite_status": self.finite_status,
            "reason_codes": list(self.reason_codes),
            "rank_diagnostics": list(self.rank_diagnostics),
            "diagnostics": self.diagnostics,
        }


def reduce_scalar_finite_reports(
    reports: Sequence[RankScalarFiniteReport],
) -> GateDecision:
    checked = _checked_reports(reports, stage="pre_backward_scalar")
    unsafe_reasons: list[str] = []
    for report in checked:
        if report.error_code is not None:
            unsafe_reasons.append(f"rank{report.rank}:{report.error_code}")
        elif not report.total_loss_finite or not all(report.term_finite.values()):
            unsafe_reasons.append(f"rank{report.rank}:non_finite_scalar")
    all_safe = not unsafe_reasons
    return GateDecision(
        stage="pre_backward_scalar",
        planned_step_id=checked[0].planned_step_id,
        world_size=checked[0].world_size,
        ranks=tuple(report.rank for report in checked),
        all_ranks_safe=all_safe,
        should_call_backward=all_safe,
        should_call_optimizer_step=False,
        should_clear_gradients=not all_safe,
        optimizer_update_status=(
            "pending_backward" if all_safe else "skipped_non_finite_scalar"
        ),
        finite_status="finite" if all_safe else "non_finite",
        reason_codes=tuple(unsafe_reasons),
        rank_diagnostics=tuple(report.to_diagnostic_dict() for report in checked),
        diagnostics={
            "unsafe_rank_count": len(unsafe_reasons),
            "policy": "all_rank_scalar_consensus",
        },
    )


def reduce_gradient_overflow_reports(
    reports: Sequence[RankGradientFiniteReport],
) -> GateDecision:
    checked = _checked_reports(reports, stage="post_backward_gradient")
    unsafe_reasons: list[str] = []
    unsafe_ranks: set[int] = set()
    grad_norms: list[float] = []
    for report in checked:
        if report.grad_norm is not None:
            grad_norm = float(report.grad_norm)
            if math.isfinite(grad_norm):
                grad_norms.append(grad_norm)
            elif report.gradients_finite:
                unsafe_reasons.append(f"rank{report.rank}:non_finite_grad_norm")
                unsafe_ranks.add(report.rank)
        elif report.gradients_finite:
            unsafe_reasons.append(f"rank{report.rank}:missing_grad_norm")
            unsafe_ranks.add(report.rank)
        if not report.gradients_finite:
            unsafe_reasons.append(f"rank{report.rank}:non_finite_gradient")
            unsafe_ranks.add(report.rank)
        if report.backend_overflow:
            unsafe_reasons.append(f"rank{report.rank}:backend_overflow")
            unsafe_ranks.add(report.rank)
    all_safe = not unsafe_reasons
    return GateDecision(
        stage="post_backward_gradient",
        planned_step_id=checked[0].planned_step_id,
        world_size=checked[0].world_size,
        ranks=tuple(report.rank for report in checked),
        all_ranks_safe=all_safe,
        should_call_backward=False,
        should_call_optimizer_step=all_safe,
        should_clear_gradients=not all_safe,
        optimizer_update_status=(
            "ready_to_step" if all_safe else "skipped_gradient_or_overflow"
        ),
        finite_status="finite" if all_safe else "non_finite",
        reason_codes=tuple(unsafe_reasons),
        rank_diagnostics=tuple(report.to_diagnostic_dict() for report in checked),
        diagnostics={
            "unsafe_rank_count": len(unsafe_ranks),
            "unsafe_reason_count": len(unsafe_reasons),
            "policy": "all_rank_gradient_consensus",
            "max_grad_norm": max(grad_norms) if grad_norms else None,
        },
    )


def build_gradient_finite_report(
    parameters: Iterable[torch.nn.Parameter],
    *,
    planned_step_id: int,
    rank: int,
    world_size: int,
    backend_overflow: bool,
) -> RankGradientFiniteReport:
    squared_norm = 0.0
    saw_grad = False
    gradients_finite = True
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        saw_grad = True
        detached = grad.detach()
        if detached.layout != torch.strided:
            raise RuntimeContractError(
                "gradient finite report does not support sparse gradients in V1",
                code="runtime.sparse_gradient_unsupported",
                context={
                    "planned_step_id": planned_step_id,
                    "rank": rank,
                    "world_size": world_size,
                    "shape": [int(item) for item in detached.shape],
                    "layout": str(detached.layout),
                },
            )
        if not bool(torch.isfinite(detached).all().item()):
            gradients_finite = False
        norm = torch.linalg.vector_norm(detached.float())
        if math.isfinite(float(norm.detach().cpu())) and math.isfinite(squared_norm):
            squared_norm += float(norm.detach().cpu()) ** 2
        else:
            squared_norm = float("inf")
    if not saw_grad:
        grad_norm = None
    elif math.isfinite(squared_norm):
        grad_norm = math.sqrt(squared_norm)
    else:
        grad_norm = float("inf")
    return RankGradientFiniteReport(
        planned_step_id=planned_step_id,
        rank=rank,
        world_size=world_size,
        gradients_finite=gradients_finite,
        backend_overflow=bool(backend_overflow),
        grad_norm=grad_norm,
    )


def _checked_reports(
    reports: Sequence[RankScalarFiniteReport] | Sequence[RankGradientFiniteReport],
    *,
    stage: str,
) -> tuple[RankScalarFiniteReport, ...] | tuple[RankGradientFiniteReport, ...]:
    checked = tuple(reports)
    if not checked:
        raise RuntimeContractError(
            "finite gate requires at least one rank report",
            code="runtime.gate_empty_reports",
            context={"stage": stage},
        )
    planned_step_ids = {report.planned_step_id for report in checked}
    if len(planned_step_ids) != 1:
        raise RuntimeContractError(
            "finite gate reports must share planned_step_id",
            code="runtime.gate_planned_step",
            context={"stage": stage, "planned_step_ids": sorted(planned_step_ids)},
        )
    world_sizes = {report.world_size for report in checked}
    if len(world_sizes) != 1:
        raise RuntimeContractError(
            "finite gate reports must share world_size",
            code="runtime.gate_world_size",
            context={"stage": stage, "world_sizes": sorted(world_sizes)},
        )
    world_size = checked[0].world_size
    if world_size <= 0:
        raise RuntimeContractError(
            "finite gate world_size must be positive",
            code="runtime.gate_world_size",
            context={"stage": stage, "world_size": world_size},
        )
    ranks = tuple(sorted(report.rank for report in checked))
    expected = tuple(range(world_size))
    if ranks != expected:
        raise RuntimeContractError(
            "finite gate reports must include exactly one report per rank",
            code="runtime.gate_rank_coverage",
            context={
                "stage": stage,
                "expected_ranks": list(expected),
                "observed_ranks": list(ranks),
            },
        )
    return tuple(sorted(checked, key=lambda report: report.rank))


def _tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _optional_float(value: torch.Tensor) -> float | None:
    scalar = float(value.detach().cpu())
    return scalar if math.isfinite(scalar) else None


__all__ = [
    "GateDecision",
    "RankGradientFiniteReport",
    "RankScalarFiniteReport",
    "build_gradient_finite_report",
    "reduce_gradient_overflow_reports",
    "reduce_scalar_finite_reports",
]
