"""Finite scalar and gradient gate decisions for training runtime."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import RuntimeContractError
from src.losses import LossBundle
from src.runtime.optimizer_boundary import (
    TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING,
    TERMINAL_PRE_WRAPPER_MIXED_SCALER_OVERFLOW,
    TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT,
    TERMINAL_PRE_WRAPPER_UNRELATED_UNSAFE,
)


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
            # Keyed on the RAW semantic value, not the configured-weight
            # product. For objective terms the two are non-finite together;
            # for a zero-weight protected gate ablation the weighted value is
            # a literal zero, so weighted-keying would hide a non-finite
            # protected diagnostic from this all-rank pre-backward decision
            # (entry-audit F-2).
            term_finite[term.name] = _tensor_is_finite(term.raw_loss)
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
    # fp16 candidacy inputs (tasks 3.1/3.2). They travel on the SAME all-rank
    # report the gradient gate already gathers, so the closed boundary action
    # converges without a second gradient-scan collective.
    scaler_active: bool = False
    unscale_completed: bool = False
    scaler_found_inf: bool = False
    report_error_code: str | None = None
    # The rank's resolved precision DECLARATION, carried so the all-rank
    # consensus can distinguish a genuine bf16/fp32 boundary from a
    # declared-fp16 boundary whose scaler is unreachable. It is never a
    # rank-local raise condition: the refusal converges after the gather.
    declared_fp16: bool = False

    def is_safe(self) -> bool:
        return (
            self.report_error_code is None
            and self.gradients_finite
            and not self.backend_overflow
            and not self.scaler_found_inf
            and self.grad_norm is not None
            and math.isfinite(float(self.grad_norm))
        )

    def is_scaler_overflow_candidate(self) -> bool:
        """The current-unscaled-gradient overflow predicate.

        Deliberately reads only THIS step's evidence: unscaled gradient
        finiteness and the `found_inf` record populated by this step's
        exactly-once unscale. A previous wrapper call's skip flag is never an
        input here.
        """

        return bool(self.scaler_active) and (
            not self.gradients_finite
            or self.scaler_found_inf
            or self.backend_overflow
            or (self.grad_norm is not None and not math.isfinite(float(self.grad_norm)))
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
            "scaler_active": self.scaler_active,
            "unscale_completed": self.unscale_completed,
            "scaler_found_inf": self.scaler_found_inf,
            "declared_fp16": self.declared_fp16,
            "report_error_code": self.report_error_code,
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
    # The closed all-rank boundary action. Exactly one of `apply`,
    # `scaler_skip`, `not_attempted` -- or `None` together with a
    # `terminal_reason`, which is a terminal distributed decision rather than a
    # fourth normal action. `None` with no terminal reason means this decision
    # does not own a boundary action (a safe pre-backward gate).
    optimizer_boundary_action: str | None = None
    terminal_reason: str | None = None
    scaler_active: bool = False
    unscale_completed: bool = False
    pre_clip_grad_norm_rank_max: float | None = None

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
            "optimizer_boundary_action": self.optimizer_boundary_action,
            "terminal_reason": self.terminal_reason,
            "pre_clip_grad_norm_rank_max": self.pre_clip_grad_norm_rank_max,
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
        # A rejected scalar gate is a SUPPORTED `not_attempted` boundary: no
        # backward, no wrapper, nothing unscaled.
        optimizer_boundary_action=None if all_safe else "not_attempted",
    )


def reduce_gradient_overflow_reports(
    reports: Sequence[RankGradientFiniteReport],
) -> GateDecision:
    checked = _checked_reports(reports, stage="post_backward_gradient")
    unsafe_reasons: list[str] = []
    unsafe_ranks: set[int] = set()
    grad_norms: list[float] = []
    for report in checked:
        if report.report_error_code is not None:
            unsafe_reasons.append(f"rank{report.rank}:{report.report_error_code}")
            unsafe_ranks.add(report.rank)
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
        if report.scaler_found_inf:
            unsafe_reasons.append(f"rank{report.rank}:scaler_found_inf")
            unsafe_ranks.add(report.rank)
    all_safe = not unsafe_reasons
    max_grad_norm = max(grad_norms) if grad_norms else None
    action, terminal_reason = _reduce_boundary_action(checked, all_safe=all_safe)
    scaler_active = all(report.scaler_active for report in checked)
    unscale_completed = any(report.unscale_completed for report in checked)
    if terminal_reason is not None:
        unsafe_reasons = [*unsafe_reasons, f"terminal:{terminal_reason}"]
    calls_wrapper = action in ("apply", "scaler_skip")
    return GateDecision(
        stage="post_backward_gradient",
        planned_step_id=checked[0].planned_step_id,
        world_size=checked[0].world_size,
        ranks=tuple(report.rank for report in checked),
        all_ranks_safe=all_safe,
        should_call_backward=False,
        should_call_optimizer_step=calls_wrapper,
        should_clear_gradients=not all_safe,
        optimizer_update_status=_post_backward_status(
            action=action,
            terminal_reason=terminal_reason,
        ),
        finite_status="finite" if all_safe else "non_finite",
        reason_codes=tuple(unsafe_reasons),
        rank_diagnostics=tuple(report.to_diagnostic_dict() for report in checked),
        diagnostics={
            "unsafe_rank_count": len(unsafe_ranks),
            "unsafe_reason_count": len(unsafe_reasons),
            "policy": "all_rank_gradient_consensus",
            "max_grad_norm": max_grad_norm,
        },
        optimizer_boundary_action=action,
        terminal_reason=terminal_reason,
        scaler_active=scaler_active,
        unscale_completed=unscale_completed,
        pre_clip_grad_norm_rank_max=max_grad_norm,
    )


def _post_backward_status(*, action: str | None, terminal_reason: str | None) -> str:
    if terminal_reason is not None:
        return f"terminal_{terminal_reason}"
    if action == "apply":
        # Preserved verbatim: downstream compatibility surfaces read this
        # exact string for the ready-to-step gate.
        return "ready_to_step"
    if action == "scaler_skip":
        return "ready_to_scaler_skip"
    return "skipped_gradient_or_overflow"


def _reduce_boundary_action(
    checked: Sequence[RankGradientFiniteReport],
    *,
    all_safe: bool,
) -> tuple[str | None, str | None]:
    """Converge ONE closed boundary action, or ONE terminal unsafe decision.

    This is a pure function of the already-gathered all-rank reports, so every
    rank computes the identical result from the identical inputs before any
    rank enters the optimizer wrapper. No rank-local branch precedes it.
    """

    scaler_ranks = {report.rank for report in checked if report.scaler_active}
    if not scaler_ranks:
        if all(report.declared_fp16 for report in checked):
            # Every rank declares fp16 and NO rank can resolve a scaler. The
            # launch gate refuses this state, so observing it here is
            # post-launch scaler drift. It is terminal on every rank: this
            # boundary is not the retained bf16/non-scaler path, so it may
            # neither `apply` (unprotected, possibly still-scaled gradients)
            # nor be reclassified as a SUPPORTED `not_attempted` completed
            # boundary -- the regime itself is broken, not just this step's
            # finiteness.
            return (None, TERMINAL_PRE_WRAPPER_FP16_SCALER_MISSING)
        # Retained bf16/non-scaler path, unchanged: safe applies, unsafe is a
        # SUPPORTED `not_attempted` completed boundary.
        return ("apply" if all_safe else "not_attempted", None)
    if len(scaler_ranks) != len(checked):
        return (None, TERMINAL_PRE_WRAPPER_SCALER_CANDIDACY_DIVERGENT)
    if any(report.report_error_code is not None for report in checked):
        return (None, TERMINAL_PRE_WRAPPER_UNRELATED_UNSAFE)
    if any(not report.unscale_completed for report in checked):
        return (None, TERMINAL_PRE_WRAPPER_UNRELATED_UNSAFE)
    candidates = {
        report.rank for report in checked if report.is_scaler_overflow_candidate()
    }
    if len(candidates) == len(checked):
        return ("scaler_skip", None)
    if candidates:
        return (None, TERMINAL_PRE_WRAPPER_MIXED_SCALER_OVERFLOW)
    if all(report.is_safe() for report in checked):
        return ("apply", None)
    # No rank is an overflow candidate, yet some rank is unsafe for an
    # unrelated reason (a missing or non-finite norm without non-finite
    # gradients). Under fp16 that is terminal, not a silent skip.
    return (None, TERMINAL_PRE_WRAPPER_UNRELATED_UNSAFE)


def build_gradient_finite_report(
    parameters: Iterable[torch.nn.Parameter],
    *,
    planned_step_id: int,
    rank: int,
    world_size: int,
    backend_overflow: bool,
    scaler_active: bool = False,
    unscale_completed: bool = False,
    scaler_found_inf: bool = False,
    report_error_code: str | None = None,
    declared_fp16: bool = False,
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
        scaler_active=bool(scaler_active),
        unscale_completed=bool(unscale_completed),
        scaler_found_inf=bool(scaler_found_inf),
        report_error_code=report_error_code,
        declared_fp16=bool(declared_fp16),
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
