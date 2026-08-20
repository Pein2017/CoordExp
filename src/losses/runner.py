"""Aggregate protected V1 losses into a backward-ready bundle."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

from src.common.errors import LossContractError
from src.config.models import LossesConfig
from src.losses.base_ce import BaseTokenCE
from src.losses.bindings import (
    BASE_CE_BINDING,
    COORD_GAUSSIAN_RPS_BINDING,
    COORDINATE_TOKEN_TYPES,
    PROTECTED_BASE_CE_WEIGHT,
    TOKEN_LOSS_BINDINGS,
    TOKEN_TYPE_GATE_BINDING,
    TokenLossBinding,
)
from src.losses.context import LossContext
from src.losses.coord_gaussian_rps import CoordGaussianRPSLoss
from src.losses.normalizers import (
    PlannedStepLossSlice,
    SegmentBalancedDenominator,
    segment_balanced_contribution,
)
from src.losses.token_type_gate import TokenTypeGateLoss
from src.losses.vocab import V1_TOKEN_TYPES
from src.supervision import TokenSequence


@dataclass(frozen=True)
class LossTermResult:
    """One term's three separated planned-step concepts.

    - `raw_loss` (1): the **semantic** globally normalized value before
      configured weighting -- rank-local numerator over the complete
      planned-step (global, when distributed) `segment_balanced` denominator.
      It carries NO backend compensation.
    - `weighted_loss` (2): `raw_loss x weight`, still uncompensated.
    - `backward_contribution` (3): the only differentiable value the runner
      offers to backward -- `weighted_loss` multiplied **exactly once** by
      `backend_gradient_scale` (the world size) to compensate the
      Accelerate/DDP mean-gradient reduction.

    At world size one `backend_gradient_scale` is `1.0` and (3) IS (2): the
    default `None` means exactly that identity, so a bundle constructed
    without an explicit contribution is a world-size-one bundle.
    """

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
    backend_gradient_scale: float = 1.0
    backward_contribution: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.backward_contribution is None:
            object.__setattr__(self, "backward_contribution", self.weighted_loss)

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
            "backend_gradient_scale": float(self.backend_gradient_scale),
            "backward_contribution": _float_value(self.backward_contribution),
        }


@dataclass(frozen=True)
class LossBundle:
    """`total_loss` is semantic telemetry; `backward_loss` is the objective.

    `total_loss` is the sum of the weighted **semantic** objective terms and
    is what every metric/artifact surface reports. `backward_loss` is the sum
    of the differentiable local **backward contributions** and is the only
    tensor a runtime may hand to `backward()`. They are equal at world size
    one, which is what the `None` default encodes.
    """

    total_loss: torch.Tensor
    terms: tuple[LossTermResult, ...]
    metrics: dict[str, float]
    counts: dict[str, int]
    diagnostics: dict[str, Any]
    finite_status: dict[str, Any]
    accuracy_stats: dict[str, int] = field(default_factory=dict)
    backward_loss: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.backward_loss is None:
            object.__setattr__(self, "backward_loss", self.total_loss)

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
            "backward_loss": _float_value(self.backward_loss),
            "terms": [term.to_artifact_dict() for term in self.terms],
            "metrics": dict(self.metrics),
            "counts": dict(self.counts),
            "diagnostics": self.diagnostics,
            "finite_status": self.finite_status,
            "accuracy_stats": dict(self.accuracy_stats),
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
class _ActiveTokenLoss:
    """One composed term: closed binding metadata + its resolved runtime inputs."""

    binding: TokenLossBinding
    weight: float
    term: BaseTokenCE | TokenTypeGateLoss | CoordGaussianRPSLoss
    token_types: tuple[str, ...] | None

    @property
    def is_detached_diagnostic(self) -> bool:
        """True only for a zero-weight `detached_diagnostic` protected term.

        Strict config makes gate weight `0.0` and the named
        `zero_weight_ablation` mode equivalent, so the resolved weight is the
        runner-side identity of that ablation.
        """

        return self.binding.zero_policy == "detached_diagnostic" and self.weight == 0.0


@dataclass(frozen=True)
class LossRunner:
    base_ce_weight: float
    token_type_gate_weight: float
    token_type_gate_groups: tuple[str, ...]
    coord_gaussian_rps_weight: float = 0.0
    coord_gaussian_rps: CoordGaussianRPSLoss | None = None

    def __post_init__(self) -> None:
        _validate_weight(BASE_CE_BINDING.name, self.base_ce_weight)
        _validate_weight(TOKEN_TYPE_GATE_BINDING.name, self.token_type_gate_weight)
        _validate_weight(
            COORD_GAUSSIAN_RPS_BINDING.name, self.coord_gaussian_rps_weight
        )
        _validate_token_type_groups(self.token_type_gate_groups)
        # Fail closed at construction: the composition path is the only place
        # zero policies are decided, so run it once here.
        self._active_token_losses()

    def _active_token_losses(self) -> tuple[_ActiveTokenLoss, ...]:
        """THE single closed composition path.

        Walks the compiled `TOKEN_LOSS_BINDINGS` inventory in canonical order
        and resolves each binding's zero policy. There is no registry lookup,
        import-by-name, callable configuration, or pass-through factory: a
        term that is not in this function is not composable at all.
        """

        active: list[_ActiveTokenLoss] = []
        for binding in TOKEN_LOSS_BINDINGS:
            if binding is BASE_CE_BINDING:
                # `forbid`: strict config already rejects omission/reweighting;
                # this is the runtime backstop at composition.
                if float(self.base_ce_weight) != PROTECTED_BASE_CE_WEIGHT:
                    raise LossContractError(
                        "base_ce is a protected loss with zero policy 'forbid': "
                        "its weight must be exactly 1.0 and it can never be "
                        "omitted, zeroed, or reweighted",
                        code="loss.base_ce_weight_forbidden",
                        context={
                            "term": binding.name,
                            "zero_policy": binding.zero_policy,
                            "weight": float(self.base_ce_weight),
                            "required_weight": PROTECTED_BASE_CE_WEIGHT,
                        },
                    )
                active.append(
                    _ActiveTokenLoss(
                        binding=binding,
                        weight=float(self.base_ce_weight),
                        term=BaseTokenCE(),
                        token_types=None,
                    )
                )
            elif binding is TOKEN_TYPE_GATE_BINDING:
                # `detached_diagnostic`: always composed. At weight zero the
                # term still computes its denominator and fp32 per-atom math
                # inside a no-grad boundary (see `compute_micro_step`).
                active.append(
                    _ActiveTokenLoss(
                        binding=binding,
                        weight=float(self.token_type_gate_weight),
                        term=TokenTypeGateLoss(),
                        token_types=tuple(self.token_type_gate_groups),
                    )
                )
            elif binding is COORD_GAUSSIAN_RPS_BINDING:
                weight = float(self.coord_gaussian_rps_weight)
                term = self.coord_gaussian_rps
                if weight == 0.0:
                    # `omit`: nothing is instantiated, so nothing may exist.
                    if term is not None:
                        raise LossContractError(
                            "an omitted auxiliary loss must not be instantiated: "
                            "zero policy 'omit' builds no term, denominator, "
                            "math, bundle entry, finite check, or metric field",
                            code="loss.auxiliary_omitted_term_instantiated",
                            context={
                                "term": binding.name,
                                "zero_policy": binding.zero_policy,
                                "weight": weight,
                            },
                        )
                    continue
                if term is None:
                    raise LossContractError(
                        "coord_gaussian_rps weight requires a configured loss term",
                        code="loss.coord_gaussian_rps_missing_term",
                        context={"weight": weight},
                    )
                active.append(
                    _ActiveTokenLoss(
                        binding=binding,
                        weight=weight,
                        term=term,
                        token_types=COORDINATE_TOKEN_TYPES,
                    )
                )
            else:  # pragma: no cover - defended closed inventory
                raise LossContractError(
                    "token loss binding has no composition branch",
                    code="loss.binding_unhandled",
                    context={"term": binding.name},
                )
        return tuple(active)

    def active_bindings(self) -> tuple[TokenLossBinding, ...]:
        """The closed bindings this runner composes, in canonical order."""

        return tuple(active.binding for active in self._active_token_losses())

    @classmethod
    def from_config(cls, config: LossesConfig) -> "LossRunner":
        if config.normalizer != "segment_balanced":
            raise LossContractError(
                "LossRunner V1 only supports segment_balanced normalization",
                code="loss.normalizer_unsupported",
                context={"normalizer": config.normalizer},
            )
        auxiliary = config.auxiliary
        coord_cfg = auxiliary.coord_gaussian_rps if auxiliary is not None else None
        coord_weight = 0.0 if coord_cfg is None else float(coord_cfg.weight)
        # `omit`: an absent or zero-weight auxiliary is never instantiated.
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
            if coord_cfg is not None and coord_weight > 0.0
            else None
        )
        return cls(
            base_ce_weight=config.protected.base_ce.weight,
            token_type_gate_weight=config.protected.token_type_gate.weight,
            token_type_gate_groups=tuple(config.protected.token_type_gate.groups),
            coord_gaussian_rps_weight=coord_weight,
            coord_gaussian_rps=coord_term,
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
        # One denominator per composed binding, in canonical order. An omitted
        # auxiliary is absent from the composition, so no local or global
        # denominator is built or gathered for it and it can never raise a
        # zero-eligible failure for the planned step.
        local_denominators = {
            active.binding.name: _build_denominator_from_token_sequences(
                active.binding.name,
                token_sequences,
                token_types=active.token_types,
            )
            for active in self._active_token_losses()
        }
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
                denominators[BASE_CE_BINDING.name],
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
        active_losses = self._active_token_losses()
        if plan.token_type_gate_groups != tuple(self.token_type_gate_groups):
            raise LossContractError(
                "planned-step plan was built by a differently configured runner",
                code="loss.streaming_plan_runner_mismatch",
                context={
                    "plan_groups": list(plan.token_type_gate_groups),
                    "runner_groups": list(self.token_type_gate_groups),
                },
            )
        results: list[LossTermResult] = []
        for active in active_losses:
            # `detached_diagnostic` at weight zero: denominator and fp32
            # per-atom math still run, but entirely inside a no-grad boundary.
            grad_context = (
                torch.no_grad() if active.is_detached_diagnostic else nullcontext()
            )
            with grad_context:
                results.append(
                    _compute_token_term_contribution(
                        name=active.binding.name,
                        context=context,
                        weight=active.weight,
                        term=active.term,
                        token_types=active.token_types,
                        denominator=plan.denominators[active.binding.name],
                        local_micro_step_index=local_micro_step_index,
                        backend_gradient_scale=plan.backend_gradient_scale,
                        detached_diagnostic=active.is_detached_diagnostic,
                    )
                )
        terms = tuple(results)
        # Protected base CE is always composed (`forbid` zero policy), so this
        # lookup cannot miss; counts are defined against its denominator.
        base_result = next(
            term for term in terms if term.name == BASE_CE_BINDING.name
        )
        objective_terms = tuple(
            term_result
            for active, term_result in zip(active_losses, terms, strict=True)
            if not active.is_detached_diagnostic
        )
        # Only objective terms enter either total; a detached diagnostic never
        # contributes a tensor to the optimized objective or its graph.
        # `total_loss` is the SEMANTIC sum (no backend compensation) and is
        # what every metric/artifact surface reports; `backward_loss` is the
        # compensated sum and is the only tensor offered to `backward()`.
        total_loss = sum(
            (term_result.weighted_loss for term_result in objective_terms),
            terms[0].weighted_loss.new_zeros(()),
        )
        if all(
            term_result.backward_contribution is term_result.weighted_loss
            for term_result in objective_terms
        ):
            # Nothing to compensate (world size one): the two totals are the
            # same node, so the autograd graph is identical to the
            # pre-separation implementation.
            backward_loss = total_loss
        else:
            backward_loss = sum(
                (term_result.backward_contribution for term_result in objective_terms),
                terms[0].weighted_loss.new_zeros(()),
            )
        counts = _build_micro_counts(context, base_result)
        metrics, accuracy_stats = _build_metrics(
            total_loss=total_loss,
            terms=terms,
            contexts=(context,),
            counts=counts,
        )
        finite_status = _build_finite_status(total_loss=total_loss, terms=terms)
        metrics["finite/total_loss"] = _finite_metric(total_loss)
        for term_result in terms:
            # Keyed on the RAW semantic value, never the configured-weight
            # product: a detached diagnostic's weighted value is a literal
            # zero and would mask a non-finite protected diagnostic.
            metrics[f"finite/{term_result.name}"] = _finite_metric(
                term_result.raw_loss
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
            accuracy_stats=accuracy_stats,
            backward_loss=backward_loss,
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
        # Semantic total: the sum of weighted objective terms, with no backend
        # compensation anywhere in it (design decision 4).
        total_loss = sum(float(term["weighted_loss"]) for term in terms)
        backward_loss = sum(float(term["backward_contribution"]) for term in terms)
        accuracy_stats = _merge_accuracy_stats(artifacts)
        metrics = {
            "loss/total": total_loss,
            "acc_top1": _ratio_or_raise(
                accuracy_stats["top1_correct"], accuracy_stats["atom_count"]
            ),
            "acc_top5": _ratio_or_raise(
                accuracy_stats["top5_correct"], accuracy_stats["atom_count"]
            ),
        }
        for term in terms:
            name = str(term["name"])
            # Explicit raw/weighted families (design decision 5). The
            # ambiguous bare `loss/<name>` alias is NOT dual-written: it named
            # the weighted value while reading like the term's own value.
            metrics[f"loss/{name}/raw"] = float(term["raw_loss"])
            metrics[f"loss/{name}/weighted"] = float(term["weighted_loss"])
            metrics[f"loss/{name}/selected_count"] = float(
                term.get("selected_count", 0)
            )
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
            # Raw-keyed for the same reason as the micro-step labels: a
            # detached diagnostic's weighted value is a literal zero.
            "terms": {
                str(term["name"]): _finite_label_from_float(float(term["raw_loss"]))
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
            "backward_loss": backward_loss,
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
            "accuracy_stats": accuracy_stats,
        }


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
    detached_diagnostic: bool = False,
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
    # (1) raw SEMANTIC value: rank-local numerator over the complete
    # planned-step denominator (global when distributed). The backend
    # mean-gradient compensation deliberately does NOT appear here -- it used
    # to (`raw = raw * backend_gradient_scale`), which made every raw/weighted
    # telemetry value world-size dependent even though the global objective is
    # not (design decision 4).
    raw = segment_balanced_contribution(loss_slice, denominator=denominator)
    scale = float(backend_gradient_scale)
    if detached_diagnostic:
        # Exactly 0.0, never `raw * 0.0`: a non-finite raw diagnostic must not
        # leak into the weighted value or the objective. Its finite status is
        # carried by `raw_loss` (see `_build_finite_status`). A detached
        # diagnostic has no backward contribution at all, so it is never
        # compensated either.
        weighted = raw.detach().new_zeros(())
        backward_contribution = weighted
    else:
        # (2) weighted SEMANTIC value.
        weighted = raw * float(weight)
        # (3) differentiable local contribution: compensated exactly once.
        # `scale == 1.0` reuses the identical tensor so the world-size-one
        # autograd graph is unchanged by the separation.
        backward_contribution = weighted if scale == 1.0 else weighted * scale
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
        backend_gradient_scale=scale,
        backward_contribution=backward_contribution,
    )


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
        "count/examples": len(
            {segment.example_id for segment in context.token_sequence.segments}
        ),
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
        # Semantic (uncompensated) sufficient statistics: summing the
        # rank-local micro-step contributions over one planned step. The
        # cross-rank sum happens once, in the runtime metric reducer.
        weighted_loss = sum(float(item["weighted_loss"]) for item in term_items)
        raw_loss = sum(float(item["raw_loss"]) for item in term_items)
        # FAIL-CLOSED (pre-DDP audit I-1/I-6, Wave-5 remainder): every real
        # term artifact carries `backward_contribution`
        # (`LossTermResult.to_artifact_dict`). Falling back to
        # `weighted_loss` would silently substitute the UNcompensated
        # semantic value, dividing the planned step's backward objective by
        # the world size with no exception, no telemetry difference, and no
        # finite-gate signal. There is no narrow exemption: a term item
        # without the field is a broken or foreign artifact stream, not a
        # world-size-one one.
        backward_contribution = sum(
            _checked_micro_backward_contribution(item, term_name=name)
            for item in term_items
        )
        backend_gradient_scale = float(
            term_items[0].get("backend_gradient_scale", 1.0)
        )
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
                "backend_gradient_scale": backend_gradient_scale,
                "backward_contribution": backward_contribution,
            }
        )
    return merged


def _checked_micro_backward_contribution(
    item: Mapping[str, Any], *, term_name: str
) -> float:
    if "backward_contribution" not in item:
        raise LossContractError(
            "streaming micro-step term artifact is missing "
            "backward_contribution; falling back to the uncompensated "
            "weighted_loss would scale the planned step's backward objective "
            "by 1/world_size",
            code="loss.micro_artifact_backward_contribution_missing",
            context={
                "term": term_name,
                "observed_fields": sorted(str(field) for field in item),
            },
        )
    return float(item["backward_contribution"])


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


def _merge_accuracy_stats(artifacts: tuple[dict[str, Any], ...]) -> dict[str, int]:
    """Sum exact rank-local-step integer sufficient statistics across micro-steps.

    Never reconstructs counts from rounded float ratios; summing the exact
    integers here (rather than weight-averaging the per-micro-step ratios)
    keeps the planned-step total exact by construction. Every micro-step
    artifact MUST carry valid `accuracy_stats`; a missing or malformed entry
    fails closed rather than silently contributing zero (which would
    undercount the planned-step total).
    """

    top1_correct = 0
    top5_correct = 0
    atom_count = 0
    for micro_step_index, artifact in enumerate(artifacts):
        stats = artifact.get("accuracy_stats")
        if not isinstance(stats, Mapping):
            raise LossContractError(
                "streaming micro-step artifact is missing accuracy_stats",
                code="loss.accuracy_stats_missing",
                context={"micro_step_index": micro_step_index},
            )
        expected_fields = {"top1_correct", "top5_correct", "atom_count"}
        if set(stats) != expected_fields:
            raise LossContractError(
                "streaming micro-step accuracy_stats must contain exactly the "
                "declared sufficient-statistic fields",
                code="loss.accuracy_stats_fields",
                context={
                    "micro_step_index": micro_step_index,
                    "expected_fields": sorted(expected_fields),
                    "observed_fields": sorted(str(field) for field in stats),
                },
            )
        micro_top1 = _checked_micro_accuracy_stat(
            stats, "top1_correct", micro_step_index=micro_step_index
        )
        micro_top5 = _checked_micro_accuracy_stat(
            stats, "top5_correct", micro_step_index=micro_step_index
        )
        micro_atoms = _checked_micro_accuracy_stat(
            stats, "atom_count", micro_step_index=micro_step_index
        )
        if micro_top1 > micro_atoms or micro_top5 > micro_atoms:
            raise LossContractError(
                "streaming micro-step accuracy_stats correct count cannot "
                "exceed the atom count",
                code="loss.accuracy_stats_correct_exceeds_atoms",
                context={
                    "micro_step_index": micro_step_index,
                    "top1_correct": micro_top1,
                    "top5_correct": micro_top5,
                    "atom_count": micro_atoms,
                },
            )
        top1_correct += micro_top1
        top5_correct += micro_top5
        atom_count += micro_atoms
    return {
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "atom_count": atom_count,
    }


def _checked_micro_accuracy_stat(
    stats: Mapping[str, Any], field: str, *, micro_step_index: int
) -> int:
    if field not in stats:
        raise LossContractError(
            "streaming micro-step accuracy_stats is missing a required field",
            code="loss.accuracy_stats_field_missing",
            context={"micro_step_index": micro_step_index, "field": field},
        )
    value = stats[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LossContractError(
            "streaming micro-step accuracy_stats field must be a non-negative integer",
            code="loss.accuracy_stats_field_type",
            context={
                "micro_step_index": micro_step_index,
                "field": field,
                "value": value,
            },
        )
    return value


def _weighted_average(values_and_weights: Iterable[tuple[float, int]]) -> float:
    numerator = 0.0
    denominator = 0
    for value, weight in values_and_weights:
        numerator += float(value) * int(weight)
        denominator += int(weight)
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


def _build_metrics(
    *,
    total_loss: torch.Tensor,
    terms: tuple[LossTermResult, ...],
    contexts: tuple[LossContext, ...],
    counts: dict[str, int],
) -> tuple[dict[str, float], dict[str, int]]:
    accuracy_stats = _accuracy_stats(contexts)
    metrics = {
        "loss/total": _float_value(total_loss),
        "acc_top1": _ratio_or_raise(
            accuracy_stats["top1_correct"], accuracy_stats["atom_count"]
        ),
        "acc_top5": _ratio_or_raise(
            accuracy_stats["top5_correct"], accuracy_stats["atom_count"]
        ),
    }
    for term in terms:
        # Same explicit family as `finalize_planned_step` (design decision 5):
        # one projection for micro-step bundles, planned-step finalization,
        # train rows, and forward-eval rows.
        metrics[f"loss/{term.name}/raw"] = _float_value(term.raw_loss)
        metrics[f"loss/{term.name}/weighted"] = _float_value(term.weighted_loss)
        metrics[f"loss/{term.name}/selected_count"] = float(term.selected_count)
        metrics[f"loss/{term.name}/token_weighted_diag"] = _float_value(
            term.token_weighted_diagnostic
        )
        metrics[f"loss/{term.name}/segment_count"] = float(
            term.denominator.eligible_segment_count
        )
    for name, value in counts.items():
        metrics[name] = float(value)
    return metrics, accuracy_stats


def _accuracy_stats(contexts: tuple[LossContext, ...]) -> dict[str, int]:
    """Exact integer top-1/top-5 correct counts and the shared atom count.

    These are the rank-local sufficient statistics: cross-rank accuracy
    reduction sums them before forming the ratio (never weights per-rank
    ratios by an already-global count, never reconstructs counts from
    rounded ratios).
    """

    with torch.no_grad():
        top1_correct = 0
        top5_correct = 0
        total = 0
        for context in contexts:
            if not context.atoms:
                continue
            selected = context.logits[0].index_select(0, context.logits_positions)
            targets = context.target_ids
            predictions = torch.argmax(selected, dim=1)
            top1_correct += int(predictions.eq(targets).sum().item())
            topk = torch.topk(
                selected,
                k=min(5, int(selected.shape[1])),
                dim=1,
            ).indices
            top5_correct += int(topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item())
            total += int(targets.numel())
    return {
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "atom_count": total,
    }


def _ratio_or_raise(correct: int, total: int) -> float:
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
    """Per-term finite labels derived from the RAW semantic value.

    For every objective term `weighted = raw * finite_positive_weight`, so raw
    and weighted are non-finite together and this is behaviour-preserving.
    It differs exactly for the zero-weight protected gate ablation, whose
    weighted value is a literal zero: keying that term on `weighted` would
    report a non-finite protected diagnostic as finite and let an unsafe
    planned step proceed to backward (entry-audit F-2).
    """

    return {
        "total_loss": _finite_label(total_loss),
        "terms": {term.name: _finite_label(term.raw_loss) for term in terms},
    }


def _finite_label(value: torch.Tensor) -> str:
    return (
        "finite" if bool(torch.isfinite(value.detach()).all().item()) else "non_finite"
    )


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
