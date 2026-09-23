"""CPU qualification checks for the maintained three-loss training route.

The production loss implementation remains in :mod:`src.losses`.  This module
only wraps the existing runner at the trainer boundary and calculates a small,
independent reference for the three configured terms.  It is deliberately
usable with either the streaming (planned-step) or ordinary ``compute`` path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from src.losses.context import LossContext
from src.losses.runner import LossBundle
from src.supervision import TokenSequence


THREE_LOSS_TERM_ORDER = (
    "base_ce",
    "token_type_gate",
    "raw_axis_validity_hinge",
)
DEFAULT_THREE_LOSS_WEIGHTS = {
    "base_ce": 1.0,
    "token_type_gate": 0.2,
    "raw_axis_validity_hinge": 0.01,
}
DEFAULT_TOKEN_TYPE_GATE_GROUPS = ("desc_text", "schema", "coordinate", "eos")
DEFAULT_RAW_AXIS_MARGIN = 1.0 / 999.0


class ThreeLossQualificationError(AssertionError):
    """A maintained three-loss contract did not match its independent check."""


@dataclass(frozen=True)
class _ReferenceTerm:
    name: str
    raw_sum: torch.Tensor
    denominator: int
    backend_gradient_scale: float
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    selected_atom_count: int
    segment_counts: tuple[int, ...]


@dataclass
class ThreeLossQualificationProbe:
    """Check one or more maintained loss calls and retain compact receipts.

    ``observe_micro_step`` is the primary production hook.  It receives the
    actual ``LossContext`` and ``PlannedStepLossPlan`` immediately after the
    maintained runner computes a micro-step contribution, before backward.
    ``observe_batch`` supports the ordinary non-streaming runner for CPU-only
    qualification.
    """

    expected_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THREE_LOSS_WEIGHTS)
    )
    token_type_gate_groups: tuple[str, ...] = DEFAULT_TOKEN_TYPE_GATE_GROUPS
    raw_axis_margin: float = DEFAULT_RAW_AXIS_MARGIN
    loss_tolerance: float = 2e-4
    gradient_tolerance: float = 2e-4
    require_unequal_segment_counts: bool = True
    records: list[dict[str, Any]] = field(default_factory=list, init=False)
    plan_records: list[dict[str, Any]] = field(default_factory=list, init=False)
    update_records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _validated_plan_artifact: dict[str, Any] | None = field(default=None, init=False)
    _segment_counts: list[int] = field(default_factory=list, init=False)

    @property
    def term_order(self) -> tuple[str, str, str]:
        third = (
            "conditional_order_gate"
            if "conditional_order_gate" in self.expected_weights
            else "raw_axis_validity_hinge"
        )
        return ("base_ce", "token_type_gate", third)

    def __post_init__(self) -> None:
        observed = {str(name): float(value) for name, value in self.expected_weights.items()}
        expected = set(self.term_order)
        if set(observed) != expected:
            raise ValueError(
                "three-loss expected_weights must contain exactly "
                f"{sorted(expected)}; got {sorted(observed)}"
            )
        for name, value in observed.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"three-loss weight must be positive for {name}: {value}")
        object.__setattr__(self, "expected_weights", observed)
        if tuple(self.token_type_gate_groups) != DEFAULT_TOKEN_TYPE_GATE_GROUPS:
            # The current V1 contract is closed; accepting a different group
            # list here would make the independent reference a new objective.
            raise ValueError(
                "three-loss check requires the maintained V1 token-type groups"
            )
        if not math.isfinite(float(self.raw_axis_margin)) or self.raw_axis_margin < 0.0:
            raise ValueError("raw_axis_margin must be finite and non-negative")
        if self.loss_tolerance < 0.0 or self.gradient_tolerance < 0.0:
            raise ValueError("three-loss tolerances must be non-negative")

    def observe_plan(
        self,
        micro_steps_or_sequences: Sequence[Any],
        plan: Any,
        *,
        gathered_payloads: Sequence[Mapping[str, Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Validate planned-step denominator construction independently.

        For one rank, expected counts come directly from the token sequences.
        For multiple ranks, the hook validates that the maintained plan equals
        the exact rank payloads returned by the caller's existing gatherer and
        checks the global scope/scale contract.  It does not perform a second
        collective or implement a trainer.
        """

        sequences = tuple(_token_sequence(item) for item in micro_steps_or_sequences)
        world_size = int(getattr(plan, "world_size", 1))
        rank = int(getattr(plan, "rank", 0))
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ThreeLossQualificationError(
                f"invalid planned-step rank metadata: rank={rank}, world_size={world_size}"
            )
        observed_groups = tuple(getattr(plan, "token_type_gate_groups", ()))
        if observed_groups != self.token_type_gate_groups:
            raise ThreeLossQualificationError(
                f"token-type gate groups mismatch: {observed_groups!r} != "
                f"{self.token_type_gate_groups!r}"
            )
        expected = _expected_denominator_artifacts(
            sequences,
            scope="planned_step" if world_size == 1 else "planned_step_global",
            gathered_payloads=gathered_payloads,
            term_order=self.term_order,
        )
        observed_denominators = getattr(plan, "denominators", None)
        if not isinstance(observed_denominators, Mapping):
            raise ThreeLossQualificationError("planned loss has no denominator mapping")
        observed: dict[str, dict[str, Any]] = {}
        for name in self.term_order:
            denominator = observed_denominators.get(name)
            if denominator is None:
                raise ThreeLossQualificationError(
                    f"three-loss term is disabled or missing from plan: {name}"
                )
            actual = _denominator_artifact(denominator)
            observed[name] = actual
            _assert_denominator(name, actual, expected[name])
        expected_scope = "planned_step" if world_size == 1 else "planned_step_global"
        observed_scope = str(getattr(plan, "denominator_scope", ""))
        if observed_scope != expected_scope:
            raise ThreeLossQualificationError(
                f"planned denominator scope mismatch: {observed_scope!r} != {expected_scope!r}"
            )
        backend_scale = float(getattr(plan, "backend_gradient_scale", math.nan))
        expected_scale = 1.0 if world_size == 1 else float(world_size)
        if backend_scale != expected_scale:
            raise ThreeLossQualificationError(
                f"planned backend gradient scale mismatch: {backend_scale} != {expected_scale}"
            )
        artifact = {
            "world_size": world_size,
            "rank": rank,
            "denominator_scope": observed_scope,
            "backend_gradient_scale": backend_scale,
            "denominators": observed,
        }
        self.plan_records.append(artifact)
        self._validated_plan_artifact = artifact
        return artifact

    def assert_plan_unchanged(self, plan: Any) -> None:
        """Reject a caller that swaps a validated plan before computing loss."""

        if self._validated_plan_artifact is None:
            return
        observed = {
            "world_size": int(getattr(plan, "world_size", -1)),
            "rank": int(getattr(plan, "rank", -1)),
            "denominator_scope": str(getattr(plan, "denominator_scope", "")),
            "backend_gradient_scale": float(
                getattr(plan, "backend_gradient_scale", math.nan)
            ),
            "denominators": {
                name: _denominator_artifact(getattr(plan, "denominators", {})[name])
                for name in self.term_order
                if name in getattr(plan, "denominators", {})
            },
        }
        if observed != self._validated_plan_artifact:
            raise ThreeLossQualificationError(
                "compute_micro_step received a plan different from the validated plan"
            )

    def observe_micro_step(
        self,
        context: LossContext,
        bundle: LossBundle,
        plan: Any,
        *,
        local_micro_step_index: int,
    ) -> dict[str, Any]:
        """Compare a maintained micro-step bundle with the independent reference."""

        self.assert_plan_unchanged(plan)
        if not isinstance(context, LossContext):
            raise ThreeLossQualificationError(
                f"three-loss hook received {type(context).__name__}, not LossContext"
            )
        backend_scale = float(getattr(plan, "backend_gradient_scale", 1.0))
        denominators = getattr(plan, "denominators", None)
        if not isinstance(denominators, Mapping):
            raise ThreeLossQualificationError("micro-step plan has no denominator mapping")
        references = tuple(
            _reference_term(
                (context,),
                name=name,
                denominator=denominators[name],
                weight=float(self.expected_weights[name]),
                backend_gradient_scale=backend_scale,
                token_type_gate_groups=self.token_type_gate_groups,
                raw_axis_margin=float(self.raw_axis_margin),
            )
            for name in self.term_order
            if name in denominators
        )
        record = self._compare_bundle(
            (context,),
            bundle,
            references,
            local_micro_step_index=local_micro_step_index,
            denominator_source="planned_step",
        )
        self.records.append(record)
        self._segment_counts.extend(record["segment_target_counts"])
        return record

    def observe_batch(
        self,
        contexts: Sequence[LossContext],
        bundle: LossBundle,
    ) -> dict[str, Any]:
        """Compare the ordinary ``LossRunner.compute`` path on CPU."""

        checked_contexts = tuple(contexts)
        if not checked_contexts:
            raise ThreeLossQualificationError("three-loss batch must contain a context")
        references = tuple(
            _reference_term(
                checked_contexts,
                name=name,
                denominator=bundle.term_by_name(name).denominator,
                weight=float(self.expected_weights[name]),
                backend_gradient_scale=1.0,
                token_type_gate_groups=self.token_type_gate_groups,
                raw_axis_margin=float(self.raw_axis_margin),
            )
            for name in self.term_order
        )
        for reference in references:
            _assert_denominator(
                reference.name,
                _denominator_artifact(bundle.term_by_name(reference.name).denominator),
                _expected_denominator_artifacts(
                    tuple(context.token_sequence for context in checked_contexts),
                    scope="planned_step",
                    term_order=self.term_order,
                )[reference.name],
            )
        record = self._compare_bundle(
            checked_contexts,
            bundle,
            references,
            local_micro_step_index=None,
            denominator_source="planned_step",
        )
        self.records.append(record)
        self._segment_counts.extend(record["segment_target_counts"])
        return record

    def note_update(
        self,
        *,
        planned_step_id: int,
        learning_rates: Sequence[float],
        gradient_nonzero: bool | None = None,
        parameter_delta_max: float | None = None,
    ) -> dict[str, Any]:
        """Record update metadata without requiring a nonzero update.

        LR0 warmup, an initially zero-B branch, and a valid zero hinge are
        ordinary qualification states.  The parent trainer can call this at
        its existing before/after optimizer hook; this method intentionally
        does not turn any of those states into a failure.
        """

        record = {
            "planned_step_id": int(planned_step_id),
            "learning_rates": [float(value) for value in learning_rates],
            "zero_learning_rate": any(float(value) == 0.0 for value in learning_rates),
            "gradient_nonzero": None if gradient_nonzero is None else bool(gradient_nonzero),
            "parameter_delta_max": (
                None if parameter_delta_max is None else float(parameter_delta_max)
            ),
        }
        self.update_records.append(record)
        return record

    def validate(self) -> dict[str, Any]:
        """Return a compact receipt or raise on a missing qualification condition."""

        if not self.records:
            raise ThreeLossQualificationError("three-loss qualification observed no loss bundle")
        if self.require_unequal_segment_counts and len(set(self._segment_counts)) <= 1:
            raise ThreeLossQualificationError(
                "three-loss qualification did not exercise unequal segment target counts"
            )
        return self.artifact(status="candidate")

    def artifact(self, *, status: str = "candidate") -> dict[str, Any]:
        return {
            "schema": "coordinate_codebook_alignment.three_loss_qualification.v1",
            "status": str(status),
            "expected_weights": dict(self.expected_weights),
            "token_type_gate_groups": list(self.token_type_gate_groups),
            "raw_axis_margin": float(self.raw_axis_margin),
            "loss_tolerance": float(self.loss_tolerance),
            "gradient_tolerance": float(self.gradient_tolerance),
            "plans": list(self.plan_records),
            "loss_checks": list(self.records),
            "updates": list(self.update_records),
        }

    def _compare_bundle(
        self,
        contexts: tuple[LossContext, ...],
        bundle: LossBundle,
        references: tuple[_ReferenceTerm, ...],
        *,
        local_micro_step_index: int | None,
        denominator_source: str,
    ) -> dict[str, Any]:
        if not isinstance(bundle, LossBundle):
            raise ThreeLossQualificationError(
                f"three-loss hook received {type(bundle).__name__}, not LossBundle"
            )
        observed_names = tuple(term.name for term in bundle.terms)
        if observed_names != self.term_order:
            raise ThreeLossQualificationError(
                f"maintained term order/activation mismatch: {observed_names!r}"
            )
        term_records: dict[str, dict[str, Any]] = {}
        expected_total = contexts[0].logits.new_zeros(())
        for reference in references:
            actual = bundle.term_by_name(reference.name)
            expected_total = expected_total + reference.weighted_loss
            if float(actual.weight) != float(self.expected_weights[reference.name]):
                raise ThreeLossQualificationError(
                    f"{reference.name} weight mismatch: {actual.weight} != "
                    f"{self.expected_weights[reference.name]}"
                )
            _assert_tensor_close(
                f"{reference.name} raw loss",
                actual.raw_loss,
                reference.raw_loss,
                self.loss_tolerance,
            )
            _assert_tensor_close(
                f"{reference.name} weighted loss",
                actual.weighted_loss,
                reference.weighted_loss,
                self.loss_tolerance,
            )
            actual_gradient = _gradient(actual.weighted_loss, contexts[0].logits)
            expected_gradient = _gradient(reference.weighted_loss, contexts[0].logits)
            gradient_delta = _gradient_delta(actual_gradient, expected_gradient)
            if gradient_delta is not None and gradient_delta > self.gradient_tolerance:
                raise ThreeLossQualificationError(
                    f"{reference.name} gradient mismatch: {gradient_delta} > "
                    f"{self.gradient_tolerance}"
                )
            denominator = _denominator_artifact(actual.denominator)
            term_records[reference.name] = {
                "raw_sum": float(reference.raw_sum.detach()),
                "denominator": int(reference.denominator),
                "backend_gradient_scale": float(reference.backend_gradient_scale),
                "raw_loss": float(reference.raw_loss.detach()),
                "weighted_loss": float(reference.weighted_loss.detach()),
                "actual_raw_loss": float(actual.raw_loss.detach()),
                "actual_weighted_loss": float(actual.weighted_loss.detach()),
                "gradient_max_delta": gradient_delta,
                "selected_atom_count": reference.selected_atom_count,
                "segment_counts": list(reference.segment_counts),
                "actual_denominator": denominator,
            }
        _assert_tensor_close(
            "total loss",
            bundle.total_loss,
            expected_total,
            self.loss_tolerance,
        )
        total_gradient_delta = _gradient_delta(
            _gradient(bundle.total_loss, contexts[0].logits),
            _gradient(expected_total, contexts[0].logits),
        )
        if total_gradient_delta is not None and total_gradient_delta > self.gradient_tolerance:
            raise ThreeLossQualificationError(
                f"total loss gradient mismatch: {total_gradient_delta} > "
                f"{self.gradient_tolerance}"
            )
        base_counts = _segment_counts(contexts, token_types=None)
        role_counts = {
            token_type: sum(
                _selected_atom_count(context, token_types=(token_type,))
                for context in contexts
            )
            for token_type in self.token_type_gate_groups
        }
        return {
            "local_micro_step_index": local_micro_step_index,
            "denominator_source": denominator_source,
            "term_order": list(observed_names),
            "segment_target_counts": list(base_counts),
            "role_target_counts": role_counts,
            "total_gradient_max_delta": total_gradient_delta,
            "terms": term_records,
        }


class ThreeLossRunnerHook:
    """Transparent runner decorator consumed by the maintained trainer wrapper."""

    def __init__(self, inner: Any, probe: ThreeLossQualificationProbe) -> None:
        self._inner = inner
        self._probe = probe
        self._gathered_payloads: list[Mapping[str, Mapping[str, Any]]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def prepare_planned_step(self, micro_steps_or_sequences: Sequence[Any], **kwargs: Any) -> Any:
        hinge = getattr(self._inner, "raw_axis_validity_hinge", None)
        if "conditional_order_gate" in self._probe.expected_weights:
            if float(getattr(self._inner, "raw_axis_validity_hinge_weight", math.nan)) != 0.0:
                raise ThreeLossQualificationError("corrected three-loss route must disable the old hinge")
            if float(getattr(self._inner, "conditional_order_gate_weight", math.nan)) != 0.2:
                raise ThreeLossQualificationError("corrected conditional gate weight mismatch")
        if hinge is not None:
            margin = float(getattr(hinge, "margin", math.nan))
            if margin != float(self._probe.raw_axis_margin):
                raise ThreeLossQualificationError(
                    f"raw-axis margin mismatch: {margin} != {self._probe.raw_axis_margin}"
                )
        original_gatherer = kwargs.get("denominator_gatherer")
        gathered: list[Mapping[str, Mapping[str, Any]]] | None = None
        if callable(original_gatherer):
            def gather(payload: Mapping[str, Mapping[str, Any]]) -> Sequence[Mapping[str, Mapping[str, Any]]]:
                nonlocal gathered
                result = tuple(original_gatherer(payload))
                gathered = list(result)
                return result

            kwargs["denominator_gatherer"] = gather
        plan = self._inner.prepare_planned_step(micro_steps_or_sequences, **kwargs)
        self._probe.observe_plan(
            micro_steps_or_sequences,
            plan,
            gathered_payloads=gathered,
        )
        self._gathered_payloads = gathered
        return plan

    def compute_micro_step(
        self,
        context: LossContext,
        plan: Any,
        *,
        local_micro_step_index: int,
    ) -> Any:
        bundle = self._inner.compute_micro_step(
            context,
            plan,
            local_micro_step_index=local_micro_step_index,
        )
        self._probe.observe_micro_step(
            context,
            bundle,
            plan,
            local_micro_step_index=local_micro_step_index,
        )
        return bundle

    def compute(self, contexts: Sequence[LossContext]) -> Any:
        bundle = self._inner.compute(contexts)
        self._probe.observe_batch(contexts, bundle)
        return bundle


def _reference_term(
    contexts: tuple[LossContext, ...],
    *,
    name: str,
    denominator: Any,
    weight: float,
    backend_gradient_scale: float,
    token_type_gate_groups: tuple[str, ...],
    raw_axis_margin: float,
) -> _ReferenceTerm:
    denominator_count = int(getattr(denominator, "eligible_segment_count", 0))
    if denominator_count <= 0:
        raise ThreeLossQualificationError(
            f"{name} reference has no eligible denominator segments"
        )
    raw_sum = contexts[0].logits.new_zeros((), dtype=torch.float32)
    selected_atom_count = 0
    segment_counts: list[int] = []
    for context in contexts:
        if name == "base_ce":
            values, _targets, atoms = context.select_logits_fp32()
            per_atom = -F.log_softmax(values, dim=-1).gather(
                1, context.target_ids.unsqueeze(1)
            ).squeeze(1)
            counts = _add_segment_means(atoms, per_atom)
        elif name == "token_type_gate":
            values, _targets, atoms = context.select_logits_fp32(
                token_types=token_type_gate_groups
            )
            per_atom = _independent_type_gate(values, atoms, context)
            counts = _add_segment_means(atoms, per_atom)
        elif name == "raw_axis_validity_hinge":
            segment_values, selected = _independent_axis_segment_losses(
                context,
                margin=raw_axis_margin,
            )
            raw_sum = raw_sum + sum(segment_values, raw_sum.new_zeros(()))
            selected_atom_count += selected
            counts = _segment_counts((context,), token_types=None)
        elif name == "conditional_order_gate":
            segment_values, selected = _independent_order_segment_losses(context)
            raw_sum = raw_sum + sum(segment_values, raw_sum.new_zeros(()))
            selected_atom_count += selected
            counts = _segment_counts((context,), token_types=None)
        else:  # pragma: no cover - term order is validated at construction.
            raise AssertionError(name)
        if name not in {"raw_axis_validity_hinge", "conditional_order_gate"}:
            raw_sum = raw_sum + sum(counts[0], raw_sum.new_zeros(()))
            selected_atom_count += counts[1]
            segment_counts.extend(counts[2])
        else:
            segment_counts.extend(counts)
    raw_loss = raw_sum / float(denominator_count) * float(backend_gradient_scale)
    weighted = raw_loss * float(weight)
    return _ReferenceTerm(
        name=name,
        raw_sum=raw_sum,
        denominator=denominator_count,
        backend_gradient_scale=float(backend_gradient_scale),
        raw_loss=raw_loss,
        weighted_loss=weighted,
        selected_atom_count=selected_atom_count,
        segment_counts=tuple(segment_counts),
    )


def _add_segment_means(
    atoms: tuple[Any, ...],
    per_atom: torch.Tensor,
) -> tuple[list[torch.Tensor], int, list[int]]:
    values_by_segment: dict[int, list[torch.Tensor]] = {}
    counts_by_segment: dict[int, int] = {}
    for value, atom in zip(per_atom, atoms, strict=True):
        values_by_segment.setdefault(int(atom.segment_index), []).append(value)
        counts_by_segment[int(atom.segment_index)] = counts_by_segment.get(
            int(atom.segment_index), 0
        ) + 1
    segment_means = [torch.stack(values).mean() for values in values_by_segment.values()]
    ordered_counts = [counts_by_segment[index] for index in sorted(counts_by_segment)]
    return segment_means, len(atoms), ordered_counts


def _independent_type_gate(
    logits: torch.Tensor,
    atoms: tuple[Any, ...],
    context: LossContext,
) -> torch.Tensor:
    all_logsumexp = torch.logsumexp(logits, dim=1)
    values: list[torch.Tensor] = []
    for row, atom, total in zip(logits, atoms, all_logsumexp, strict=True):
        allowed = torch.tensor(
            context.vocab_groups.allowed_ids(atom.token_type),
            dtype=torch.long,
            device=logits.device,
        )
        values.append(total - torch.logsumexp(row.index_select(0, allowed), dim=0))
    if values:
        return torch.stack(values)
    return logits.sum(dim=1)[:0]


def _independent_axis_segment_losses(
    context: LossContext,
    *,
    margin: float,
) -> tuple[list[torch.Tensor], int]:
    coordinate_logits, _targets, atoms = context.select_logits_fp32(
        token_types=("coordinate",)
    )
    coordinate_ids = tuple(int(value) for value in context.vocab_groups.coordinate)
    if len(coordinate_ids) != 1000:
        raise ThreeLossQualificationError(
            f"axis reference requires 1000 coordinate bins, got {len(coordinate_ids)}"
        )
    zero = (
        coordinate_logits.sum() * 0.0
        if coordinate_logits.numel()
        else context.logits.sum() * 0.0
    )
    expectations = (
        coordinate_logits[:, torch.tensor(coordinate_ids, device=coordinate_logits.device)]
        .softmax(dim=-1)
        @ (torch.arange(1000, device=coordinate_logits.device, dtype=torch.float32) / 999.0)
        if coordinate_logits.numel()
        else coordinate_logits.new_empty((0,))
    )
    groups: dict[tuple[int, int, int, str, str], dict[str, Any]] = {}
    for index, atom in enumerate(atoms):
        target = atom.coordinate_target
        if target is None or not isinstance(atom.object_id, str) or not atom.object_id:
            raise ThreeLossQualificationError(
                "coordinate role lacks object/target metadata for axis reference"
            )
        key = (
            int(atom.pack_index),
            int(atom.segment_index),
            int(atom.example_index),
            str(atom.example_id),
            str(atom.object_id),
        )
        group = groups.setdefault(key, {"bbox": tuple(target.bbox), "slots": {}})
        if tuple(group["bbox"]) != tuple(target.bbox):
            raise ThreeLossQualificationError("one object has inconsistent coordinate bboxes")
        slot = int(target.slot_index)
        if slot in group["slots"]:
            raise ThreeLossQualificationError("one object has duplicate coordinate slots")
        group["slots"][slot] = index

    losses_by_segment: dict[int, list[torch.Tensor]] = {
        int(segment.segment_index): []
        for segment in context.token_sequence.segments
        if any(atom.segment_index == segment.segment_index for atom in context.atoms)
    }
    for (pack_index, segment_index, _example_index, _example_id, _object_id), group in groups.items():
        del pack_index
        if set(group["slots"]) != {0, 1, 2, 3}:
            continue
        positions = [group["slots"][slot] for slot in range(4)]
        x1, y1, x2, y2 = (expectations[position] for position in positions)
        box_loss = (
            F.relu(float(margin) - (x2 - x1))
            + F.relu(float(margin) - (y2 - y1))
        ) / 2.0
        losses_by_segment.setdefault(int(segment_index), []).append(box_loss)
    segment_values = [
        torch.stack(values).mean() if values else zero
        for values in losses_by_segment.values()
    ]
    return segment_values, len(atoms)


def _independent_order_segment_losses(
    context: LossContext,
) -> tuple[list[torch.Tensor], int]:
    """Reference from actual prefix tokens, independent of the maintained term."""
    values, _targets, atoms = context.select_logits_fp32(token_types=("coordinate",))
    ids = tuple(int(value) for value in context.vocab_groups.coordinate)
    if len(ids) != 1000:
        raise ThreeLossQualificationError("order reference requires 1000 coordinate bins")
    rows = values.index_select(1, torch.tensor(ids, device=values.device))
    lookup = {token_id: index for index, token_id in enumerate(ids)}
    zero = values.sum() * 0.0 if values.numel() else context.logits.sum() * 0.0
    by_object: dict[tuple[int, int, int, str, str], dict[int, tuple[int, Any]]] = {}
    for row_index, atom in enumerate(atoms):
        target = atom.coordinate_target
        if target is None or not atom.object_id:
            raise ThreeLossQualificationError("order reference missing target/object")
        key = (atom.pack_index, atom.segment_index, atom.example_index, atom.example_id, atom.object_id)
        by_object.setdefault(key, {})[target.slot_index] = (row_index, atom)
    by_segment: dict[int, list[torch.Tensor]] = {
        segment.segment_index: [] for segment in context.token_sequence.segments
        if any(atom.segment_index == segment.segment_index for atom in context.atoms)
    }
    for key, slots in by_object.items():
        if set(slots) != {0, 1, 2, 3}:
            continue
        penalties = []
        for preceding, constrained in ((0, 2), (1, 3)):
            query_row = rows[slots[constrained][0]]
            prefix_atom = slots[preceding][1]
            threshold = lookup[context.token_sequence.input_ids[prefix_atom.target_position]]
            penalties.append(torch.logsumexp(query_row, 0) - torch.logsumexp(query_row[threshold + 1:], 0))
        by_segment[key[1]].append(torch.stack(penalties).mean())
    return [torch.stack(boxes).mean() if boxes else zero for boxes in by_segment.values()], len(atoms)


def _expected_denominator_artifacts(
    sequences: tuple[TokenSequence, ...],
    *,
    scope: str,
    gathered_payloads: Sequence[Mapping[str, Mapping[str, Any]]] | None = None,
    term_order: tuple[str, ...] = THREE_LOSS_TERM_ORDER,
) -> dict[str, dict[str, Any]]:
    local = {
        name: _sequence_denominator(name, sequences)
        for name in term_order
    }
    if gathered_payloads is None:
        return {
            name: {"term_name": name, "denominator_scope": scope, **values}
            for name, values in local.items()
        }
    merged: dict[str, dict[str, Any]] = {}
    for name in term_order:
        fields = (
            "eligible_segment_count",
            "selected_atom_count",
            "skipped_segment_count",
            "context_count",
        )
        values = {field_name: 0 for field_name in fields}
        if name in {"raw_axis_validity_hinge", "conditional_order_gate"}:
            values.update(
                complete_box_count=0,
                incomplete_box_count=0,
                zero_box_segment_count=0,
            )
        for payload in gathered_payloads:
            if name not in payload or not isinstance(payload[name], Mapping):
                raise ThreeLossQualificationError(
                    f"denominator gather payload missing {name}"
                )
            item = payload[name]
            for field_name in values:
                values[field_name] += int(item.get(field_name, 0))
        merged[name] = {
            "term_name": name,
            "denominator_scope": scope,
            **values,
        }
    return merged


def _sequence_denominator(name: str, sequences: tuple[TokenSequence, ...]) -> dict[str, int]:
    selected_atom_count = 0
    eligible_segment_count = 0
    skipped_segment_count = 0
    complete_box_count = 0
    incomplete_box_count = 0
    zero_box_segment_count = 0
    for sequence in sequences:
        if name == "base_ce":
            selected = tuple(sequence.atoms)
        elif name == "token_type_gate":
            selected = tuple(
                atom
                for atom in sequence.atoms
                if atom.token_type in DEFAULT_TOKEN_TYPE_GATE_GROUPS
            )
        else:
            selected = tuple(atom for atom in sequence.atoms if atom.token_type == "coordinate")
        selected_atom_count += len(selected)
        selected_by_segment: dict[int, int] = {}
        for atom in selected:
            selected_by_segment[int(atom.segment_index)] = (
                selected_by_segment.get(int(atom.segment_index), 0) + 1
            )
        if name in {"raw_axis_validity_hinge", "conditional_order_gate"}:
            eligible_segments = {
                int(atom.segment_index) for atom in sequence.atoms
            }
            groups: dict[tuple[int, int, int, str, str], set[int]] = {}
            bboxes: dict[tuple[int, int, int, str, str], tuple[int, ...]] = {}
            for atom in selected:
                if atom.coordinate_target is None or not atom.object_id:
                    raise ThreeLossQualificationError(
                        "coordinate denominator lacks object/target metadata"
                    )
                key = (
                    int(atom.pack_index),
                    int(atom.segment_index),
                    int(atom.example_index),
                    str(atom.example_id),
                    str(atom.object_id),
                )
                target = atom.coordinate_target
                bboxes.setdefault(key, tuple(target.bbox))
                if bboxes[key] != tuple(target.bbox):
                    raise ThreeLossQualificationError("coordinate denominator bbox mismatch")
                groups.setdefault(key, set()).add(int(target.slot_index))
            complete_by_segment = {segment: 0 for segment in eligible_segments}
            for key, slots in groups.items():
                if slots == {0, 1, 2, 3}:
                    complete_box_count += 1
                    complete_by_segment[int(key[1])] += 1
                else:
                    incomplete_box_count += 1
            zero_box_segment_count += sum(
                count == 0 for count in complete_by_segment.values()
            )
        for segment in sequence.segments:
            if selected_by_segment.get(int(segment.segment_index), 0) > 0:
                eligible_segment_count += 1
            else:
                skipped_segment_count += 1
    result = {
        "eligible_segment_count": eligible_segment_count,
        "selected_atom_count": selected_atom_count,
        "skipped_segment_count": skipped_segment_count,
        "context_count": len(sequences),
    }
    if name in {"raw_axis_validity_hinge", "conditional_order_gate"}:
        # The maintained hinge denominator includes every supervised segment,
        # including a segment with only non-coordinate roles.
        eligible_segment_count = sum(
            sum(
                1
                for segment in sequence.segments
                if any(atom.segment_index == segment.segment_index for atom in sequence.atoms)
            )
            for sequence in sequences
        )
        result.update(
            eligible_segment_count=eligible_segment_count,
            skipped_segment_count=sum(
                len(sequence.segments)
                - sum(
                    1
                    for segment in sequence.segments
                    if any(atom.segment_index == segment.segment_index for atom in sequence.atoms)
                )
                for sequence in sequences
            ),
            complete_box_count=complete_box_count,
            incomplete_box_count=incomplete_box_count,
            zero_box_segment_count=zero_box_segment_count,
        )
    return result


def _denominator_artifact(denominator: Any) -> dict[str, Any]:
    method = getattr(denominator, "to_artifact_dict", None)
    if not callable(method):
        raise ThreeLossQualificationError(
            f"loss denominator {type(denominator).__name__} has no artifact method"
        )
    return dict(method())


def _assert_denominator(
    name: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    fields = set(expected)
    observed = {field: actual.get(field) for field in fields}
    expected_values = {field: expected.get(field) for field in fields}
    if observed != expected_values:
        raise ThreeLossQualificationError(
            f"{name} denominator mismatch: observed={observed}, expected={expected_values}"
        )


def _token_sequence(value: Any) -> TokenSequence:
    if isinstance(value, TokenSequence):
        return value
    sequence = getattr(value, "token_sequence", None)
    if isinstance(sequence, TokenSequence):
        return sequence
    raise ThreeLossQualificationError(
        f"planned loss input has no TokenSequence: {type(value).__name__}"
    )


def _selected_atom_count(context: LossContext, *, token_types: tuple[str, ...]) -> int:
    allowed = frozenset(token_types)
    return sum(atom.token_type in allowed for atom in context.atoms)


def _segment_counts(
    contexts: Sequence[LossContext],
    *,
    token_types: tuple[str, ...] | None,
) -> list[int]:
    allowed = None if token_types is None else frozenset(token_types)
    counts: list[int] = []
    for context in contexts:
        for segment in context.token_sequence.segments:
            count = sum(
                atom.segment_index == segment.segment_index
                and (allowed is None or atom.token_type in allowed)
                for atom in context.atoms
            )
            if count:
                counts.append(count)
    return counts


def _assert_tensor_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    tolerance: float,
) -> None:
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise ThreeLossQualificationError(f"{name} is not tensor-valued")
    delta = float((actual.detach().float() - expected.detach().float()).abs().max())
    if not math.isfinite(delta) or delta > tolerance:
        raise ThreeLossQualificationError(f"{name} mismatch: {delta} > {tolerance}")


def _gradient(value: torch.Tensor, logits: torch.Tensor) -> torch.Tensor | None:
    if not logits.requires_grad:
        return None
    return torch.autograd.grad(
        value,
        logits,
        retain_graph=True,
        allow_unused=True,
    )[0]


def _gradient_delta(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
) -> float | None:
    if actual is None and expected is None:
        return None
    if actual is None:
        actual = torch.zeros_like(expected)
    if expected is None:
        expected = torch.zeros_like(actual)
    return float((actual.detach().float() - expected.detach().float()).abs().max())


__all__ = [
    "DEFAULT_RAW_AXIS_MARGIN",
    "DEFAULT_THREE_LOSS_WEIGHTS",
    "DEFAULT_TOKEN_TYPE_GATE_GROUPS",
    "THREE_LOSS_TERM_ORDER",
    "ThreeLossQualificationError",
    "ThreeLossQualificationProbe",
    "ThreeLossRunnerHook",
]
