"""Aggregate protected V1 losses into a backward-ready bundle."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import LossContractError
from src.config.models import LossesConfig
from src.losses.base_ce import BaseTokenCE
from src.losses.context import LossContext
from src.losses.normalizers import (
    PlannedStepLossSlice,
    SegmentBalancedDenominator,
    reduce_segment_balanced_planned_step,
)
from src.losses.token_type_gate import TokenTypeGateLoss
from src.losses.vocab import V1_TOKEN_TYPES
from src.supervision import TokenSequence


@dataclass(frozen=True)
class LossTermResult:
    name: str
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    weight: float
    segment_mean_numerator: torch.Tensor
    denominator: SegmentBalancedDenominator
    reducer_name: str
    selected_count: int
    skipped_count: int
    math_dtype: str
    token_weighted_diagnostic: torch.Tensor
    diagnostics: dict[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw_loss": _float_value(self.raw_loss),
            "weighted_loss": _float_value(self.weighted_loss),
            "weight": self.weight,
            "segment_mean_numerator": _float_value(self.segment_mean_numerator),
            "denominator": self.denominator.to_artifact_dict(),
            "reducer_name": self.reducer_name,
            "selected_count": self.selected_count,
            "skipped_count": self.skipped_count,
            "math_dtype": self.math_dtype,
            "token_weighted_diagnostic": _float_value(self.token_weighted_diagnostic),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class LossBundle:
    total_loss: torch.Tensor
    terms: tuple[LossTermResult, ...]
    metrics: dict[str, float]
    counts: dict[str, int]
    diagnostics: dict[str, Any]
    finite_status: dict[str, Any]

    def term_by_name(self, name: str) -> LossTermResult:
        for term in self.terms:
            if term.name == name:
                return term
        raise LossContractError(
            "unknown loss term requested from LossBundle",
            code="loss.bundle_unknown_term",
            context={"term": name, "known_terms": [term.name for term in self.terms]},
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "total_loss": _float_value(self.total_loss),
            "terms": [term.to_artifact_dict() for term in self.terms],
            "metrics": dict(self.metrics),
            "counts": dict(self.counts),
            "diagnostics": self.diagnostics,
            "finite_status": self.finite_status,
        }


@dataclass(frozen=True)
class LossRunner:
    base_ce_weight: float
    token_type_gate_weight: float
    token_type_gate_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_weight("base_ce", self.base_ce_weight)
        _validate_weight("token_type_gate", self.token_type_gate_weight)
        _validate_token_type_groups(self.token_type_gate_groups)

    @classmethod
    def from_config(cls, config: LossesConfig) -> "LossRunner":
        if config.normalizer != "segment_balanced":
            raise LossContractError(
                "LossRunner V1 only supports segment_balanced normalization",
                code="loss.normalizer_unsupported",
                context={"normalizer": config.normalizer},
            )
        return cls(
            base_ce_weight=config.protected.base_ce.weight,
            token_type_gate_weight=config.protected.token_type_gate.weight,
            token_type_gate_groups=tuple(config.protected.token_type_gate.groups),
        )

    def compute(self, contexts: Sequence[LossContext]) -> LossBundle:
        checked_contexts = _checked_contexts(contexts)
        base_result = _compute_token_term(
            name="base_ce",
            contexts=checked_contexts,
            weight=self.base_ce_weight,
            term=BaseTokenCE(),
            token_types=None,
        )
        gate_result = _compute_token_term(
            name="token_type_gate",
            contexts=checked_contexts,
            weight=self.token_type_gate_weight,
            term=TokenTypeGateLoss(),
            token_types=self.token_type_gate_groups,
        )
        terms = (base_result, gate_result)
        total_loss = sum(
            (term.weighted_loss for term in terms),
            terms[0].weighted_loss.new_zeros(()),
        )
        counts = _build_counts(checked_contexts, base_result.denominator)
        metrics = _build_metrics(
            total_loss=total_loss,
            terms=terms,
            contexts=checked_contexts,
            counts=counts,
        )
        finite_status = _build_finite_status(total_loss=total_loss, terms=terms)
        metrics["finite/total_loss"] = _finite_metric(total_loss)
        for term in terms:
            metrics[f"finite/{term.name}"] = _finite_metric(term.weighted_loss)
        diagnostics = {
            "normalizer": "segment_balanced",
            "term_order": [term.name for term in terms],
            "term_denominators": {
                term.name: term.denominator.to_artifact_dict() for term in terms
            },
        }
        return LossBundle(
            total_loss=total_loss,
            terms=terms,
            metrics=metrics,
            counts=counts,
            diagnostics=diagnostics,
            finite_status=finite_status,
        )


def _compute_token_term(
    *,
    name: str,
    contexts: tuple[LossContext, ...],
    weight: float,
    term: BaseTokenCE | TokenTypeGateLoss,
    token_types: tuple[str, ...] | None,
) -> LossTermResult:
    slices: list[PlannedStepLossSlice] = []
    for local_index, context in enumerate(contexts):
        term_context = (
            context
            if token_types is None
            else _filter_context_by_token_types(context, token_types=token_types)
        )
        if term_context.atoms:
            per_atom_losses = term.per_atom_loss(term_context)
        else:
            per_atom_losses = term_context.logits.new_empty((0,), dtype=torch.float32)
        slices.append(
            PlannedStepLossSlice(
                term_name=name,
                context=term_context,
                per_atom_losses=per_atom_losses,
                local_micro_step_index=local_index,
            )
        )
    reduced = reduce_segment_balanced_planned_step(tuple(slices))
    weighted = reduced.loss * float(weight)
    segment_mean_numerator = (
        reduced.loss.detach() * float(reduced.denominator.eligible_segment_count)
    )
    diagnostics = {
        "denominator_scope": reduced.denominator_scope,
        "context_count": reduced.context_count,
    }
    if token_types is not None:
        diagnostics["configured_token_types"] = list(token_types)
        diagnostics["selected_count_by_token_type"] = _selected_count_by_token_type(
            tuple(slices),
            token_types=token_types,
        )
    return LossTermResult(
        name=name,
        raw_loss=reduced.loss,
        weighted_loss=weighted,
        weight=float(weight),
        segment_mean_numerator=segment_mean_numerator,
        denominator=reduced.denominator,
        reducer_name="segment_balanced",
        selected_count=reduced.selected_atom_count,
        skipped_count=reduced.skipped_segment_count,
        math_dtype="float32",
        token_weighted_diagnostic=reduced.token_balanced_diagnostic,
        diagnostics=diagnostics,
    )


def _selected_count_by_token_type(
    slices: tuple[PlannedStepLossSlice, ...],
    *,
    token_types: tuple[str, ...],
) -> dict[str, int]:
    counts = {token_type: 0 for token_type in token_types}
    for item in slices:
        for atom in item.context.atoms:
            counts[atom.token_type] = counts.get(atom.token_type, 0) + 1
    return counts


def _filter_context_by_token_types(
    context: LossContext,
    *,
    token_types: tuple[str, ...],
) -> LossContext:
    allowed = frozenset(token_types)
    atoms = tuple(atom for atom in context.atoms if atom.token_type in allowed)
    atom_positions = frozenset(atom.target_position for atom in atoms)
    spans = tuple(
        span
        for span in context.token_sequence.spans
        if all(atom.target_position in atom_positions for atom in span.atoms)
    )
    return LossContext(
        logits=context.logits,
        token_sequence=TokenSequence(
            pack_index=context.token_sequence.pack_index,
            input_ids=context.token_sequence.input_ids,
            segments=context.token_sequence.segments,
            atoms=atoms,
            spans=spans,
        ),
        vocab_groups=context.vocab_groups,
        logits_position_ids=context.logits_position_ids,
    )


def _checked_contexts(contexts: Sequence[LossContext]) -> tuple[LossContext, ...]:
    checked = tuple(contexts)
    if not checked:
        raise LossContractError(
            "LossRunner requires at least one planned-step context",
            code="loss.runner_empty_contexts",
            context={},
        )
    for index, context in enumerate(checked):
        if not isinstance(context, LossContext):
            raise LossContractError(
                "LossRunner contexts must be LossContext records",
                code="loss.runner_context_type",
                context={"index": index, "value_type": type(context).__name__},
            )
    return checked


def _build_counts(
    contexts: tuple[LossContext, ...],
    base_denominator: SegmentBalancedDenominator,
) -> dict[str, int]:
    example_ids = {
        segment.example_id
        for context in contexts
        for segment in context.token_sequence.segments
    }
    return {
        "count/supervised_atoms": base_denominator.selected_atom_count,
        "count/eligible_segments": base_denominator.eligible_segment_count,
        "count/skipped_segments": base_denominator.skipped_segment_count,
        "count/packs": len(contexts),
        "count/examples": len(example_ids),
    }


def _build_metrics(
    *,
    total_loss: torch.Tensor,
    terms: tuple[LossTermResult, ...],
    contexts: tuple[LossContext, ...],
    counts: dict[str, int],
) -> dict[str, float]:
    metrics = {
        "loss/total": _float_value(total_loss),
        "acc_top1": _accuracy(contexts, k=1),
        "acc_top5": _accuracy(contexts, k=5),
    }
    for term in terms:
        metrics[f"loss/{term.name}"] = _float_value(term.weighted_loss)
        metrics[f"loss/{term.name}/token_weighted_diag"] = _float_value(
            term.token_weighted_diagnostic
        )
        metrics[f"loss/{term.name}/segment_count"] = float(
            term.denominator.eligible_segment_count
        )
    for name, value in counts.items():
        metrics[name] = float(value)
    return metrics


def _accuracy(contexts: tuple[LossContext, ...], *, k: int) -> float:
    with torch.no_grad():
        correct = 0
        total = 0
        for context in contexts:
            if not context.atoms:
                continue
            selected = context.logits[0].index_select(0, context.logits_positions)
            targets = context.target_ids
            if k == 1:
                predictions = torch.argmax(selected, dim=1)
                correct += int(predictions.eq(targets).sum().item())
            else:
                topk = torch.topk(
                    selected,
                    k=min(k, int(selected.shape[1])),
                    dim=1,
                ).indices
                correct += int(topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item())
            total += int(targets.numel())
    if total == 0:
        raise LossContractError(
            "accuracy metrics require at least one supervised atom",
            code="loss.accuracy_zero_atoms",
            context={},
        )
    return float(correct) / float(total)


def _build_finite_status(
    *,
    total_loss: torch.Tensor,
    terms: tuple[LossTermResult, ...],
) -> dict[str, Any]:
    return {
        "total_loss": _finite_label(total_loss),
        "terms": {
            term.name: _finite_label(term.weighted_loss)
            for term in terms
        },
    }


def _finite_label(value: torch.Tensor) -> str:
    return "finite" if bool(torch.isfinite(value.detach()).all().item()) else "non_finite"


def _finite_metric(value: torch.Tensor) -> float:
    return 1.0 if _finite_label(value) == "finite" else 0.0


def _float_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _validate_weight(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LossContractError(
            "loss term weight must be numeric",
            code="loss.weight_type",
            context={"term": name, "value_type": type(value).__name__},
        )
    if float(value) < 0.0 or not math.isfinite(float(value)):
        raise LossContractError(
            "loss term weight must be finite and non-negative",
            code="loss.weight_value",
            context={"term": name, "weight": value},
        )


def _validate_token_type_groups(token_types: tuple[str, ...]) -> None:
    if not token_types:
        raise LossContractError(
            "token_type_gate groups must not be empty",
            code="loss.token_type_gate_groups_empty",
            context={},
        )
    if len(set(token_types)) != len(token_types):
        raise LossContractError(
            "token_type_gate groups must not contain duplicates",
            code="loss.token_type_gate_groups_duplicate",
            context={"groups": list(token_types)},
        )
    unknown = sorted(set(token_types) - set(V1_TOKEN_TYPES))
    if unknown:
        raise LossContractError(
            "token_type_gate groups must be closed V1 token types",
            code="loss.token_type_gate_groups_unknown",
            context={"unknown": unknown, "known": list(V1_TOKEN_TYPES)},
        )


__all__ = [
    "LossBundle",
    "LossRunner",
    "LossTermResult",
]
