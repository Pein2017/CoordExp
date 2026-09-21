"""Aggregate protected V1 losses into a backward-ready bundle."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.common.errors import LossContractError
from src.config.models import LossesConfig
from src.losses.base_ce import BaseTokenCE
from src.losses.context import LossContext
from src.losses.coord_gaussian_rps import CoordGaussianRPSLoss
from src.losses.normalizers import (
    PlannedStepLossSlice,
    SegmentBalancedDenominator,
    reduce_segment_balanced_planned_step,
    segment_balanced_contribution,
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
class PlannedStepLossPlan:
    denominators: dict[str, SegmentBalancedDenominator]
    counts: dict[str, int]
    token_type_gate_groups: tuple[str, ...]
    denominator_scope: str = "planned_step"
    world_size: int = 1
    rank: int = 0
    backend_gradient_scale: float = 1.0


@dataclass(frozen=True)
class LossRunner:
    base_ce_weight: float
    token_type_gate_weight: float
    token_type_gate_groups: tuple[str, ...]
    coord_gaussian_rps_weight: float = 0.0
    coord_gaussian_rps: CoordGaussianRPSLoss | None = None

    def __post_init__(self) -> None:
        _validate_weight("base_ce", self.base_ce_weight)
        _validate_weight("token_type_gate", self.token_type_gate_weight)
        _validate_weight("coord_gaussian_rps", self.coord_gaussian_rps_weight)
        _validate_token_type_groups(self.token_type_gate_groups)
        if self.coord_gaussian_rps_weight > 0.0 and self.coord_gaussian_rps is None:
            raise LossContractError(
                "coord_gaussian_rps weight requires a configured loss term",
                code="loss.coord_gaussian_rps_missing_term",
                context={"weight": self.coord_gaussian_rps_weight},
            )

    @classmethod
    def from_config(cls, config: LossesConfig) -> "LossRunner":
        if config.normalizer != "segment_balanced":
            raise LossContractError(
                "LossRunner V1 only supports segment_balanced normalization",
                code="loss.normalizer_unsupported",
                context={"normalizer": config.normalizer},
            )
        coord_cfg = config.protected.coord_gaussian_rps
        coord_term = (
            CoordGaussianRPSLoss(
                gaussian_weight=coord_cfg.gaussian_weight,
                rps_weight=coord_cfg.rps_weight,
                temperature=coord_cfg.temperature,
                gaussian_r95_axis_fraction=coord_cfg.gaussian_r95_axis_fraction,
                gaussian_r95_cap_bins=coord_cfg.gaussian_r95_cap_bins,
                gaussian_r95_min_bins=coord_cfg.gaussian_r95_min_bins,
                gaussian_r95_fallback_bins=coord_cfg.gaussian_r95_fallback_bins,
            )
            if coord_cfg.weight > 0.0
            else None
        )
        return cls(
            base_ce_weight=config.protected.base_ce.weight,
            token_type_gate_weight=config.protected.token_type_gate.weight,
            token_type_gate_groups=tuple(config.protected.token_type_gate.groups),
            coord_gaussian_rps_weight=coord_cfg.weight,
            coord_gaussian_rps=coord_term,
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
        terms_list = [base_result, gate_result]
        if self.coord_gaussian_rps_weight > 0.0 and self.coord_gaussian_rps is not None:
            terms_list.append(
                _compute_token_term(
                    name="coord_gaussian_rps",
                    contexts=checked_contexts,
                    weight=self.coord_gaussian_rps_weight,
                    term=self.coord_gaussian_rps,
                    token_types=("coordinate",),
                )
            )
        terms = tuple(terms_list)
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

    def prepare_planned_step(
        self,
        micro_steps_or_sequences: Sequence[Any],
        *,
        denominator_gatherer: Callable[
            [Mapping[str, Mapping[str, Any]]],
            Sequence[Mapping[str, Mapping[str, Any]]],
        ]
        | None = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> PlannedStepLossPlan:
        token_sequences = tuple(
            _token_sequence_from_micro_step_or_sequence(item)
            for item in micro_steps_or_sequences
        )
        if not token_sequences:
            raise LossContractError(
                "LossRunner streaming mode requires at least one token sequence",
                code="loss.streaming_empty_window",
                context={},
            )
        base_denominator = _build_denominator_from_token_sequences(
            "base_ce",
            token_sequences,
            token_types=None,
        )
        gate_denominator = _build_denominator_from_token_sequences(
            "token_type_gate",
            token_sequences,
            token_types=self.token_type_gate_groups,
        )
        local_denominators = {
            "base_ce": base_denominator,
            "token_type_gate": gate_denominator,
        }
        if self.coord_gaussian_rps_weight > 0.0:
            local_denominators["coord_gaussian_rps"] = _build_denominator_from_token_sequences(
                "coord_gaussian_rps",
                token_sequences,
                token_types=("coordinate",),
            )
        denominators, denominator_scope, backend_gradient_scale = (
            _resolve_streaming_denominators(
                local_denominators,
                denominator_gatherer=denominator_gatherer,
                world_size=world_size,
                rank=rank,
            )
        )
        return PlannedStepLossPlan(
            denominators=denominators,
            counts=_build_counts_from_token_sequences(
                token_sequences,
                denominators["base_ce"],
            ),
            token_type_gate_groups=self.token_type_gate_groups,
            denominator_scope=denominator_scope,
            world_size=int(world_size),
            rank=int(rank),
            backend_gradient_scale=backend_gradient_scale,
        )

    def compute_micro_step(
        self,
        context: LossContext,
        plan: PlannedStepLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossBundle:
        if not isinstance(context, LossContext):
            raise LossContractError(
                "LossRunner streaming context must be a LossContext record",
                code="loss.streaming_context_type",
                context={"value_type": type(context).__name__},
            )
        base_result = _compute_token_term_contribution(
            name="base_ce",
            context=context,
            weight=self.base_ce_weight,
            term=BaseTokenCE(),
            token_types=None,
            denominator=plan.denominators["base_ce"],
            local_micro_step_index=local_micro_step_index,
            backend_gradient_scale=plan.backend_gradient_scale,
        )
        gate_result = _compute_token_term_contribution(
            name="token_type_gate",
            context=context,
            weight=self.token_type_gate_weight,
            term=TokenTypeGateLoss(),
            token_types=plan.token_type_gate_groups,
            denominator=plan.denominators["token_type_gate"],
            local_micro_step_index=local_micro_step_index,
            backend_gradient_scale=plan.backend_gradient_scale,
        )
        terms_list = [base_result, gate_result]
        if (
            self.coord_gaussian_rps_weight > 0.0
            and self.coord_gaussian_rps is not None
            and "coord_gaussian_rps" in plan.denominators
        ):
            terms_list.append(
                _compute_token_term_contribution(
                    name="coord_gaussian_rps",
                    context=context,
                    weight=self.coord_gaussian_rps_weight,
                    term=self.coord_gaussian_rps,
                    token_types=("coordinate",),
                    denominator=plan.denominators["coord_gaussian_rps"],
                    local_micro_step_index=local_micro_step_index,
                    backend_gradient_scale=plan.backend_gradient_scale,
                )
            )
        terms = tuple(terms_list)
        total_loss = sum(
            (term.weighted_loss for term in terms),
            terms[0].weighted_loss.new_zeros(()),
        )
        counts = _build_micro_counts(context, base_result)
        metrics = _build_metrics(
            total_loss=total_loss,
            terms=terms,
            contexts=(context,),
            counts=counts,
        )
        finite_status = _build_finite_status(total_loss=total_loss, terms=terms)
        metrics["finite/total_loss"] = _finite_metric(total_loss)
        for term_result in terms:
            metrics[f"finite/{term_result.name}"] = _finite_metric(
                term_result.weighted_loss
            )
        diagnostics = {
            "normalizer": "segment_balanced",
            "normalizer_scope": "planned_step_streaming",
            "denominator_scope": plan.denominator_scope,
            "backend_gradient_scale": plan.backend_gradient_scale,
            "world_size": plan.world_size,
            "rank": plan.rank,
            "local_micro_step_index": int(local_micro_step_index),
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

    def finalize_planned_step(
        self,
        micro_loss_artifacts: Sequence[dict[str, Any]],
        plan: PlannedStepLossPlan,
    ) -> dict[str, Any]:
        artifacts = tuple(dict(item) for item in micro_loss_artifacts)
        if not artifacts:
            raise LossContractError(
                "LossRunner streaming finalization requires at least one micro artifact",
                code="loss.streaming_empty_artifacts",
                context={},
            )
        terms = _merge_term_artifacts(artifacts, plan)
        total_loss = sum(float(term["weighted_loss"]) for term in terms)
        metrics = {
            "loss/total": total_loss,
            "acc_top1": _merge_accuracy_metric(artifacts, "acc_top1"),
            "acc_top5": _merge_accuracy_metric(artifacts, "acc_top5"),
        }
        for term in terms:
            name = str(term["name"])
            metrics[f"loss/{name}"] = float(term["weighted_loss"])
            metrics[f"loss/{name}/token_weighted_diag"] = float(
                term["token_weighted_diagnostic"]
            )
            denominator = term.get("denominator", {})
            metrics[f"loss/{name}/segment_count"] = float(
                denominator.get("eligible_segment_count", 0)
            )
        for name, value in plan.counts.items():
            metrics[name] = float(value)
        finite_status = {
            "total_loss": _finite_label_from_float(total_loss),
            "terms": {
                str(term["name"]): _finite_label_from_float(float(term["weighted_loss"]))
                for term in terms
            },
        }
        metrics["finite/total_loss"] = (
            1.0 if finite_status["total_loss"] == "finite" else 0.0
        )
        for name, label in finite_status["terms"].items():
            metrics[f"finite/{name}"] = 1.0 if label == "finite" else 0.0
        return {
            "total_loss": total_loss,
            "terms": terms,
            "metrics": metrics,
            "counts": dict(plan.counts),
            "diagnostics": {
                "normalizer": "segment_balanced",
                "normalizer_scope": "planned_step_streaming",
                "denominator_scope": plan.denominator_scope,
                "backend_gradient_scale": plan.backend_gradient_scale,
                "world_size": plan.world_size,
                "rank": plan.rank,
                "term_order": [str(term["name"]) for term in terms],
                "term_denominators": {
                    str(term["name"]): term["denominator"] for term in terms
                },
            },
            "finite_status": finite_status,
        }


def _compute_token_term(
    *,
    name: str,
    contexts: tuple[LossContext, ...],
    weight: float,
    term: BaseTokenCE | TokenTypeGateLoss | CoordGaussianRPSLoss,
    token_types: tuple[str, ...] | None,
) -> LossTermResult:
    slices: list[PlannedStepLossSlice] = []
    term_diagnostics: list[dict[str, Any]] = []
    for local_index, context in enumerate(contexts):
        term_context = (
            context
            if token_types is None
            else _filter_context_by_token_types(context, token_types=token_types)
        )
        if term_context.atoms:
            per_atom_losses = term.per_atom_loss(term_context)
            diagnostics_payload = getattr(term, "last_diagnostics", None)
            if diagnostics_payload is not None:
                term_diagnostics.append(
                    {
                        "local_micro_step_index": local_index,
                        **dict(diagnostics_payload),
                    }
                )
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
    if term_diagnostics:
        diagnostics["term_diagnostics"] = term_diagnostics
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


def _compute_token_term_contribution(
    *,
    name: str,
    context: LossContext,
    weight: float,
    term: BaseTokenCE | TokenTypeGateLoss | CoordGaussianRPSLoss,
    token_types: tuple[str, ...] | None,
    denominator: SegmentBalancedDenominator,
    local_micro_step_index: int,
    backend_gradient_scale: float = 1.0,
) -> LossTermResult:
    term_context = (
        context
        if token_types is None
        else _filter_context_by_token_types(context, token_types=token_types)
    )
    if term_context.atoms:
        per_atom_losses = term.per_atom_loss(term_context)
    else:
        per_atom_losses = term_context.logits.new_empty((0,), dtype=torch.float32)
    loss_slice = PlannedStepLossSlice(
        term_name=name,
        context=term_context,
        per_atom_losses=per_atom_losses,
        local_micro_step_index=local_micro_step_index,
    )
    raw = segment_balanced_contribution(loss_slice, denominator=denominator)
    raw = raw * float(backend_gradient_scale)
    weighted = raw * float(weight)
    if len(term_context.atoms) > 0:
        token_weighted = per_atom_losses.detach().mean()
        segment_mean_numerator = raw.detach() * float(
            denominator.eligible_segment_count
        )
    else:
        token_weighted = per_atom_losses.new_zeros(())
        segment_mean_numerator = raw.detach() * float(
            denominator.eligible_segment_count
        )
    diagnostics = {
        "denominator_scope": denominator.denominator_scope,
        "context_count": denominator.context_count,
        "local_micro_step_index": int(local_micro_step_index),
        "backend_gradient_scale": float(backend_gradient_scale),
    }
    if token_types is not None:
        diagnostics["configured_token_types"] = list(token_types)
        diagnostics["selected_count_by_token_type"] = {
            token_type: sum(
                1 for atom in term_context.atoms if atom.token_type == token_type
            )
            for token_type in token_types
        }
    diagnostics_payload = getattr(term, "last_diagnostics", None)
    if diagnostics_payload is not None:
        diagnostics["term_diagnostics"] = dict(diagnostics_payload)
    return LossTermResult(
        name=name,
        raw_loss=raw,
        weighted_loss=weighted,
        weight=float(weight),
        segment_mean_numerator=segment_mean_numerator,
        denominator=denominator,
        reducer_name="segment_balanced",
        selected_count=len(term_context.atoms),
        skipped_count=_skipped_segment_count(term_context),
        math_dtype="float32",
        token_weighted_diagnostic=token_weighted,
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


def _token_sequence_from_micro_step_or_sequence(value: Any) -> TokenSequence:
    if isinstance(value, TokenSequence):
        return value
    token_sequence = getattr(value, "token_sequence", None)
    if isinstance(token_sequence, TokenSequence):
        return token_sequence
    raise LossContractError(
        "LossRunner streaming plan requires TokenSequence inputs",
        code="loss.streaming_token_sequence_type",
        context={"value_type": type(value).__name__},
    )


def _build_denominator_from_token_sequences(
    term_name: str,
    token_sequences: tuple[TokenSequence, ...],
    *,
    token_types: tuple[str, ...] | None,
) -> SegmentBalancedDenominator:
    selected_atom_count = 0
    eligible_segment_count = 0
    skipped_segment_count = 0
    allowed = None if token_types is None else frozenset(token_types)
    for token_sequence in token_sequences:
        atom_counts_by_segment: dict[int, int] = {}
        for atom in token_sequence.atoms:
            if allowed is not None and atom.token_type not in allowed:
                continue
            selected_atom_count += 1
            atom_counts_by_segment[atom.segment_index] = (
                atom_counts_by_segment.get(atom.segment_index, 0) + 1
            )
        for segment in token_sequence.segments:
            if atom_counts_by_segment.get(segment.segment_index, 0) > 0:
                eligible_segment_count += 1
            else:
                skipped_segment_count += 1
    if eligible_segment_count == 0:
        raise LossContractError(
            "segment_balanced reducer requires at least one eligible segment",
            code="loss.segment_balanced_zero_eligible",
            context={
                "term": term_name,
                "context_count": len(token_sequences),
                "selected_atom_count": selected_atom_count,
                "skipped_segment_count": skipped_segment_count,
            },
        )
    return SegmentBalancedDenominator(
        term_name=term_name,
        denominator_scope="planned_step",
        eligible_segment_count=eligible_segment_count,
        selected_atom_count=selected_atom_count,
        skipped_segment_count=skipped_segment_count,
        context_count=len(token_sequences),
    )


def _resolve_streaming_denominators(
    local_denominators: Mapping[str, SegmentBalancedDenominator],
    *,
    denominator_gatherer: Callable[
        [Mapping[str, Mapping[str, Any]]],
        Sequence[Mapping[str, Mapping[str, Any]]],
    ]
    | None,
    world_size: int,
    rank: int,
) -> tuple[dict[str, SegmentBalancedDenominator], str, float]:
    checked_world_size = _checked_world_size(world_size)
    checked_rank = _checked_rank(rank, world_size=checked_world_size)
    local = {str(name): denominator for name, denominator in local_denominators.items()}
    if checked_world_size == 1:
        return local, "planned_step", 1.0
    if denominator_gatherer is None:
        raise LossContractError(
            "multi-rank streaming segment-balanced losses require denominator gathering",
            code="loss.global_denominator_gather_unavailable",
            context={"world_size": checked_world_size, "rank": checked_rank},
        )
    local_payload = {
        name: denominator.to_artifact_dict() for name, denominator in local.items()
    }
    gathered_payloads = tuple(denominator_gatherer(local_payload))
    if len(gathered_payloads) != checked_world_size:
        raise LossContractError(
            "global denominator gather returned the wrong number of rank payloads",
            code="loss.global_denominator_gather_count",
            context={
                "world_size": checked_world_size,
                "rank": checked_rank,
                "payload_count": len(gathered_payloads),
            },
        )
    return (
        _merge_global_denominators(local, gathered_payloads),
        "planned_step_global",
        float(checked_world_size),
    )


def _merge_global_denominators(
    local_denominators: Mapping[str, SegmentBalancedDenominator],
    gathered_payloads: tuple[Mapping[str, Mapping[str, Any]], ...],
) -> dict[str, SegmentBalancedDenominator]:
    merged: dict[str, SegmentBalancedDenominator] = {}
    for term_name, local_denominator in local_denominators.items():
        eligible_segment_count = 0
        selected_atom_count = 0
        skipped_segment_count = 0
        context_count = 0
        for rank_index, rank_payload in enumerate(gathered_payloads):
            if not isinstance(rank_payload, Mapping):
                raise LossContractError(
                    "global denominator gather payload must be a mapping",
                    code="loss.global_denominator_payload",
                    context={
                        "term": term_name,
                        "rank_index": rank_index,
                        "value_type": type(rank_payload).__name__,
                    },
                )
            if term_name not in rank_payload:
                raise LossContractError(
                    "global denominator gather payload is missing a term",
                    code="loss.global_denominator_payload",
                    context={
                        "term": term_name,
                        "rank_index": rank_index,
                        "available_terms": sorted(str(name) for name in rank_payload),
                    },
                )
            term_payload = rank_payload[term_name]
            if not isinstance(term_payload, Mapping):
                raise LossContractError(
                    "global denominator term payload must be a mapping",
                    code="loss.global_denominator_payload",
                    context={
                        "term": term_name,
                        "rank_index": rank_index,
                        "value_type": type(term_payload).__name__,
                    },
                )
            payload_term_name = str(term_payload.get("term_name", term_name))
            if payload_term_name != term_name:
                raise LossContractError(
                    "global denominator term payload name must match the gather key",
                    code="loss.global_denominator_payload",
                    context={
                        "expected_term": term_name,
                        "observed_term": payload_term_name,
                        "rank_index": rank_index,
                    },
                )
            eligible_segment_count += _int_payload_field(
                term_payload,
                "eligible_segment_count",
                term=term_name,
                rank_index=rank_index,
            )
            selected_atom_count += _int_payload_field(
                term_payload,
                "selected_atom_count",
                term=term_name,
                rank_index=rank_index,
            )
            skipped_segment_count += _int_payload_field(
                term_payload,
                "skipped_segment_count",
                term=term_name,
                rank_index=rank_index,
            )
            context_count += _int_payload_field(
                term_payload,
                "context_count",
                term=term_name,
                rank_index=rank_index,
            )
        if eligible_segment_count <= 0:
            raise LossContractError(
                "segment_balanced reducer requires at least one globally eligible segment",
                code="loss.segment_balanced_zero_eligible",
                context={
                    "term": term_name,
                    "selected_atom_count": selected_atom_count,
                    "skipped_segment_count": skipped_segment_count,
                    "context_count": context_count,
                },
            )
        merged[term_name] = SegmentBalancedDenominator(
            term_name=term_name,
            denominator_scope="planned_step_global",
            eligible_segment_count=eligible_segment_count,
            selected_atom_count=selected_atom_count,
            skipped_segment_count=skipped_segment_count,
            context_count=context_count,
        )
        if local_denominator.term_name != term_name:
            raise LossContractError(
                "local denominator term name must match its key",
                code="loss.global_denominator_payload",
                context={
                    "expected_term": term_name,
                    "observed_term": local_denominator.term_name,
                },
            )
    return merged


def _checked_world_size(world_size: int) -> int:
    if isinstance(world_size, bool) or int(world_size) <= 0:
        raise LossContractError(
            "global denominator world_size must be a positive integer",
            code="loss.global_denominator_world_size",
            context={"world_size": world_size},
        )
    return int(world_size)


def _checked_rank(rank: int, *, world_size: int) -> int:
    checked_rank = int(rank)
    if isinstance(rank, bool) or checked_rank < 0 or checked_rank >= world_size:
        raise LossContractError(
            "global denominator rank must be within world_size",
            code="loss.global_denominator_rank",
            context={"rank": rank, "world_size": world_size},
        )
    return checked_rank


def _int_payload_field(
    payload: Mapping[str, Any],
    field: str,
    *,
    term: str,
    rank_index: int,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        raise LossContractError(
            "global denominator payload field must be an integer",
            code="loss.global_denominator_payload",
            context={
                "term": term,
                "rank_index": rank_index,
                "field": field,
                "value_type": type(value).__name__,
            },
        )
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise LossContractError(
            "global denominator payload field must be an integer",
            code="loss.global_denominator_payload",
            context={
                "term": term,
                "rank_index": rank_index,
                "field": field,
                "value_type": type(value).__name__,
            },
            cause=exc,
        ) from exc
    if coerced < 0:
        raise LossContractError(
            "global denominator payload field must be non-negative",
            code="loss.global_denominator_payload",
            context={
                "term": term,
                "rank_index": rank_index,
                "field": field,
                "value": coerced,
            },
        )
    return coerced


def _build_counts_from_token_sequences(
    token_sequences: tuple[TokenSequence, ...],
    base_denominator: SegmentBalancedDenominator,
) -> dict[str, int]:
    example_ids = {
        segment.example_id
        for token_sequence in token_sequences
        for segment in token_sequence.segments
    }
    return {
        "count/supervised_atoms": base_denominator.selected_atom_count,
        "count/eligible_segments": base_denominator.eligible_segment_count,
        "count/skipped_segments": base_denominator.skipped_segment_count,
        "count/packs": len(token_sequences),
        "count/examples": len(example_ids),
    }


def _build_micro_counts(
    context: LossContext,
    base_result: LossTermResult,
) -> dict[str, int]:
    return {
        "count/supervised_atoms": base_result.selected_count,
        "count/eligible_segments": _eligible_segment_count(context),
        "count/skipped_segments": _skipped_segment_count(context),
        "count/packs": 1,
        "count/examples": len({segment.example_id for segment in context.token_sequence.segments}),
    }


def _eligible_segment_count(context: LossContext) -> int:
    selected = {atom.segment_index for atom in context.atoms}
    return sum(
        1
        for segment in context.token_sequence.segments
        if segment.segment_index in selected
    )


def _skipped_segment_count(context: LossContext) -> int:
    selected = {atom.segment_index for atom in context.atoms}
    return sum(
        1
        for segment in context.token_sequence.segments
        if segment.segment_index not in selected
    )


def _merge_term_artifacts(
    artifacts: tuple[dict[str, Any], ...],
    plan: PlannedStepLossPlan,
) -> list[dict[str, Any]]:
    terms_by_name: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        for term in artifact.get("terms", ()):
            term_dict = dict(term)
            terms_by_name.setdefault(str(term_dict["name"]), []).append(term_dict)

    merged: list[dict[str, Any]] = []
    for name in plan.denominators:
        term_items = terms_by_name.get(name, [])
        if not term_items:
            continue
        selected_count = sum(int(item.get("selected_count", 0)) for item in term_items)
        weighted_loss = sum(float(item["weighted_loss"]) for item in term_items)
        raw_loss = sum(float(item["raw_loss"]) for item in term_items)
        token_weighted = _weighted_average(
            (
                (
                    float(item.get("token_weighted_diagnostic", 0.0)),
                    int(item.get("selected_count", 0)),
                )
                for item in term_items
            )
        )
        denominator = plan.denominators[name].to_artifact_dict()
        diagnostics = _merge_term_diagnostics(
            name=name,
            term_items=tuple(term_items),
            denominator=denominator,
        )
        merged.append(
            {
                "name": name,
                "raw_loss": raw_loss,
                "weighted_loss": weighted_loss,
                "weight": float(term_items[0]["weight"]),
                "segment_mean_numerator": raw_loss
                * float(denominator["eligible_segment_count"]),
                "denominator": denominator,
                "reducer_name": "segment_balanced",
                "selected_count": selected_count,
                "skipped_count": int(denominator["skipped_segment_count"]),
                "math_dtype": "float32",
                "token_weighted_diagnostic": token_weighted,
                "diagnostics": diagnostics,
            }
        )
    return merged


def _merge_term_diagnostics(
    *,
    name: str,
    term_items: tuple[dict[str, Any], ...],
    denominator: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "denominator_scope": denominator["denominator_scope"],
        "context_count": denominator["context_count"],
    }
    source_diagnostics = [
        dict(item.get("diagnostics", {}))
        for item in term_items
        if isinstance(item.get("diagnostics", {}), Mapping)
    ]
    configured = next(
        (
            item.get("configured_token_types")
            for item in source_diagnostics
            if "configured_token_types" in item
        ),
        None,
    )
    if configured is not None:
        diagnostics["configured_token_types"] = list(configured)
    selected_counts = _merge_selected_count_by_token_type(source_diagnostics)
    if selected_counts:
        diagnostics["selected_count_by_token_type"] = selected_counts

    term_diagnostics: list[dict[str, Any]] = []
    for term_item in term_items:
        source = term_item.get("diagnostics", {})
        if not isinstance(source, Mapping):
            continue
        payload = source.get("term_diagnostics")
        if isinstance(payload, Mapping):
            diagnostic = dict(payload)
        else:
            continue
        diagnostic.setdefault("selected_count", int(term_item.get("selected_count", 0)))
        term_diagnostics.append(diagnostic)
    if term_diagnostics:
        diagnostics["term_diagnostics"] = term_diagnostics
        diagnostics.update(_aggregate_numeric_term_diagnostics(name, term_diagnostics))
    return diagnostics


def _merge_selected_count_by_token_type(
    source_diagnostics: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in source_diagnostics:
        payload = source.get("selected_count_by_token_type")
        if not isinstance(payload, Mapping):
            continue
        for token_type, count in payload.items():
            counts[str(token_type)] = counts.get(str(token_type), 0) + int(count)
    return counts


def _aggregate_numeric_term_diagnostics(
    name: str,
    term_diagnostics: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    diagnostics = tuple(term_diagnostics)
    weighted_keys = (
        "target_entropy_mean",
        "target_peak_prob_mean",
        "target_r95_radius_mean",
        "gaussian_ce_mean",
        "rps_mean",
    )
    aggregated: dict[str, float] = {}
    for key in weighted_keys:
        values_and_weights: list[tuple[float, int]] = []
        for diagnostic in diagnostics:
            if key not in diagnostic:
                continue
            value = _finite_diagnostic_value(name, key, diagnostic[key])
            weight = int(diagnostic.get("selected_count", 0))
            values_and_weights.append((value, weight))
        if values_and_weights:
            aggregated[key] = _weighted_average(values_and_weights)
    max_values = [
        _finite_diagnostic_value(
            name,
            "target_r95_radius_max",
            diagnostic["target_r95_radius_max"],
        )
        for diagnostic in diagnostics
        if "target_r95_radius_max" in diagnostic
    ]
    if max_values:
        aggregated["target_r95_radius_max"] = max(max_values)
    return aggregated


def _finite_diagnostic_value(name: str, key: str, value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise LossContractError(
            "loss term diagnostic must be finite",
            code="loss.term_diagnostic_non_finite",
            context={"term": name, "key": key, "value": parsed},
        )
    return parsed


def _merge_accuracy_metric(
    artifacts: tuple[dict[str, Any], ...],
    name: str,
) -> float:
    weighted_values = []
    for artifact in artifacts:
        metrics = artifact.get("metrics", {})
        counts = artifact.get("counts", {})
        weighted_values.append(
            (
                float(metrics.get(name, 0.0)),
                int(counts.get("count/supervised_atoms", 0)),
            )
        )
    return _weighted_average(weighted_values)


def _weighted_average(values_and_weights: Iterable[tuple[float, int]]) -> float:
    numerator = 0.0
    denominator = 0
    for value, weight in values_and_weights:
        numerator += float(value) * int(weight)
        denominator += int(weight)
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


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


def _finite_label_from_float(value: float) -> str:
    return "finite" if math.isfinite(float(value)) else "non_finite"


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
