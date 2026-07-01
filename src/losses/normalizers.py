"""Planned-step loss normalizers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch

from src.common.errors import LossContractError
from src.config.models import RuntimeBatchResolution
from src.losses.context import LossContext


@dataclass(frozen=True)
class PlannedStepLossSlice:
    term_name: str
    context: LossContext
    per_atom_losses: torch.Tensor
    rank: int = 0
    local_micro_step_index: int = 0

    def __post_init__(self) -> None:
        if not self.term_name:
            raise LossContractError(
                "planned-step loss slice requires a non-empty term name",
                code="loss.slice_term_name",
                context={"term_name": self.term_name},
            )
        if not isinstance(self.context, LossContext):
            raise LossContractError(
                "planned-step loss slice requires a LossContext",
                code="loss.slice_context_type",
                context={"value_type": type(self.context).__name__},
            )
        if not isinstance(self.per_atom_losses, torch.Tensor):
            raise LossContractError(
                "planned-step per-atom losses must be a tensor",
                code="loss.per_atom_loss_type",
                context={"value_type": type(self.per_atom_losses).__name__},
            )
        if self.per_atom_losses.ndim != 1:
            raise LossContractError(
                "planned-step per-atom losses must be one-dimensional",
                code="loss.per_atom_loss_shape",
                context={
                    "term": self.term_name,
                    "shape": [int(item) for item in self.per_atom_losses.shape],
                },
            )
        if int(self.per_atom_losses.shape[0]) != len(self.context.atoms):
            raise LossContractError(
                "planned-step per-atom losses must align with LossContext atoms",
                code="loss.per_atom_loss_count",
                context={
                    "term": self.term_name,
                    "loss_count": int(self.per_atom_losses.shape[0]),
                    "atom_count": len(self.context.atoms),
                    "pack_index": self.context.token_sequence.pack_index,
                },
            )
        if not torch.is_floating_point(self.per_atom_losses):
            raise LossContractError(
                "planned-step per-atom losses must be floating point",
                code="loss.per_atom_loss_dtype",
                context={"term": self.term_name, "dtype": str(self.per_atom_losses.dtype)},
            )
        if self.per_atom_losses.device != self.context.logits.device:
            raise LossContractError(
                "planned-step per-atom losses must share the LossContext device",
                code="loss.per_atom_loss_device",
                context={
                    "term": self.term_name,
                    "loss_device": str(self.per_atom_losses.device),
                    "context_device": str(self.context.logits.device),
                },
            )
        if self.rank < 0 or self.local_micro_step_index < 0:
            raise LossContractError(
                "planned-step slice rank and micro-step index must be non-negative",
                code="loss.slice_index",
                context={
                    "term": self.term_name,
                    "rank": self.rank,
                    "local_micro_step_index": self.local_micro_step_index,
                },
            )


@dataclass(frozen=True)
class SegmentBalancedDenominator:
    term_name: str
    denominator_scope: str
    eligible_segment_count: int
    selected_atom_count: int
    skipped_segment_count: int
    context_count: int

    def to_artifact_dict(self) -> dict[str, int | str]:
        return {
            "term_name": self.term_name,
            "denominator_scope": self.denominator_scope,
            "eligible_segment_count": self.eligible_segment_count,
            "selected_atom_count": self.selected_atom_count,
            "skipped_segment_count": self.skipped_segment_count,
            "context_count": self.context_count,
        }


@dataclass(frozen=True)
class SegmentBalancedLossResult:
    term_name: str
    loss: torch.Tensor
    token_balanced_diagnostic: torch.Tensor
    denominator: SegmentBalancedDenominator

    @property
    def denominator_scope(self) -> str:
        return self.denominator.denominator_scope

    @property
    def eligible_segment_count(self) -> int:
        return self.denominator.eligible_segment_count

    @property
    def selected_atom_count(self) -> int:
        return self.denominator.selected_atom_count

    @property
    def skipped_segment_count(self) -> int:
        return self.denominator.skipped_segment_count

    @property
    def context_count(self) -> int:
        return self.denominator.context_count


@dataclass(frozen=True)
class BackendScalingReceipt:
    normalizer_scope: str
    world_size: int
    effective_batch_size: int
    resolved_grad_accum_steps: int
    runtime_loss_divisor: float
    backend_loss_divisor: float

    def to_artifact_dict(self) -> dict[str, float | int | str]:
        return {
            "normalizer_scope": self.normalizer_scope,
            "world_size": self.world_size,
            "effective_batch_size": self.effective_batch_size,
            "resolved_grad_accum_steps": self.resolved_grad_accum_steps,
            "runtime_loss_divisor": self.runtime_loss_divisor,
            "backend_loss_divisor": self.backend_loss_divisor,
        }


def build_segment_balanced_denominator(
    slices: Sequence[PlannedStepLossSlice],
) -> SegmentBalancedDenominator:
    checked = _checked_slices(slices)
    term_name = checked[0].term_name
    selected_atom_count = 0
    eligible_segment_count = 0
    skipped_segment_count = 0
    for item in checked:
        selected_atom_count += len(item.context.atoms)
        atom_counts_by_segment = _atom_counts_by_segment(item)
        for segment in item.context.token_sequence.segments:
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
                "context_count": len(checked),
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
        context_count=len(checked),
    )


def reduce_segment_balanced_planned_step(
    slices: Sequence[PlannedStepLossSlice],
) -> SegmentBalancedLossResult:
    checked = _checked_slices(slices)
    denominator = build_segment_balanced_denominator(checked)
    loss = _zero_scalar_like(checked)
    token_numerator = _zero_scalar_like(checked)
    for item in checked:
        loss = loss + segment_balanced_contribution(item, denominator=denominator)
        if len(item.context.atoms) > 0:
            token_numerator = token_numerator + item.per_atom_losses.detach().sum()
    token_balanced = token_numerator / float(denominator.selected_atom_count)
    return SegmentBalancedLossResult(
        term_name=denominator.term_name,
        loss=loss,
        token_balanced_diagnostic=token_balanced,
        denominator=denominator,
    )


def segment_balanced_contribution(
    loss_slice: PlannedStepLossSlice,
    *,
    denominator: SegmentBalancedDenominator,
) -> torch.Tensor:
    if loss_slice.term_name != denominator.term_name:
        raise LossContractError(
            "segment-balanced contribution term must match the denominator",
            code="loss.segment_balanced_term_mismatch",
            context={
                "slice_term": loss_slice.term_name,
                "denominator_term": denominator.term_name,
            },
        )
    if denominator.denominator_scope != "planned_step":
        raise LossContractError(
            "segment-balanced contribution requires a planned-step denominator",
            code="loss.segment_balanced_scope",
            context={
                "term": loss_slice.term_name,
                "denominator_scope": denominator.denominator_scope,
            },
        )
    if denominator.eligible_segment_count <= 0:
        raise LossContractError(
            "segment-balanced denominator must include eligible segments",
            code="loss.segment_balanced_zero_eligible",
            context={"term": loss_slice.term_name},
        )
    segment_mean_sum = _zero_scalar(loss_slice.per_atom_losses)
    for indices in _segment_atom_indices(loss_slice).values():
        index_tensor = torch.tensor(
            indices,
            dtype=torch.long,
            device=loss_slice.per_atom_losses.device,
        )
        segment_losses = loss_slice.per_atom_losses.index_select(0, index_tensor)
        segment_mean_sum = segment_mean_sum + segment_losses.mean()
    return segment_mean_sum / float(denominator.eligible_segment_count)


def validate_planned_step_backend_scaling(
    runtime_batch: RuntimeBatchResolution,
    *,
    runtime_loss_divisor: int | float = 1,
    backend_loss_divisor: int | float = 1,
) -> BackendScalingReceipt:
    runtime_divisor = _positive_float_divisor(
        runtime_loss_divisor,
        name="runtime_loss_divisor",
    )
    backend_divisor = _positive_float_divisor(
        backend_loss_divisor,
        name="backend_loss_divisor",
    )
    if runtime_divisor != 1.0 or backend_divisor != 1.0:
        raise LossContractError(
            "planned-step normalized losses must not receive extra accumulation divisors",
            code="loss.backend_double_scaling",
            context={
                "normalizer_scope": "planned_step",
                "world_size": runtime_batch.world_size,
                "effective_batch_size": runtime_batch.effective_batch_size,
                "resolved_grad_accum_steps": runtime_batch.resolved_grad_accum_steps,
                "runtime_loss_divisor": runtime_divisor,
                "backend_loss_divisor": backend_divisor,
            },
        )
    return BackendScalingReceipt(
        normalizer_scope="planned_step",
        world_size=runtime_batch.world_size,
        effective_batch_size=runtime_batch.effective_batch_size,
        resolved_grad_accum_steps=runtime_batch.resolved_grad_accum_steps,
        runtime_loss_divisor=runtime_divisor,
        backend_loss_divisor=backend_divisor,
    )


def _checked_slices(
    slices: Sequence[PlannedStepLossSlice],
) -> tuple[PlannedStepLossSlice, ...]:
    checked = tuple(slices)
    if not checked:
        raise LossContractError(
            "segment_balanced reducer requires a non-empty planned-step window",
            code="loss.segment_balanced_empty_window",
            context={},
        )
    term_name = checked[0].term_name
    devices = {item.per_atom_losses.device for item in checked}
    for item in checked:
        if item.term_name != term_name:
            raise LossContractError(
                "segment_balanced reducer requires one term per planned-step call",
                code="loss.segment_balanced_term_mismatch",
                context={"expected": term_name, "observed": item.term_name},
            )
    if len(devices) != 1:
        raise LossContractError(
            "segment_balanced reducer requires all slices on the same device",
            code="loss.segment_balanced_device",
            context={"devices": sorted(str(device) for device in devices)},
        )
    return checked


def _atom_counts_by_segment(loss_slice: PlannedStepLossSlice) -> dict[int, int]:
    counts: dict[int, int] = {}
    for atom in loss_slice.context.atoms:
        counts[atom.segment_index] = counts.get(atom.segment_index, 0) + 1
    return counts


def _segment_atom_indices(
    loss_slice: PlannedStepLossSlice,
) -> dict[int, list[int]]:
    indices_by_segment: dict[int, list[int]] = {}
    known_segments = {
        segment.segment_index for segment in loss_slice.context.token_sequence.segments
    }
    for index, atom in enumerate(loss_slice.context.atoms):
        if atom.segment_index not in known_segments:
            raise LossContractError(
                "segment-balanced atom references an unknown segment",
                code="loss.segment_balanced_unknown_segment",
                context={
                    "term": loss_slice.term_name,
                    "pack_index": atom.pack_index,
                    "segment_index": atom.segment_index,
                    "target_position": atom.target_position,
                },
            )
        indices_by_segment.setdefault(atom.segment_index, []).append(index)
    return indices_by_segment


def _zero_scalar_like(slices: Sequence[PlannedStepLossSlice]) -> torch.Tensor:
    return _zero_scalar(slices[0].per_atom_losses)


def _zero_scalar(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.new_zeros(())


def _positive_float_divisor(value: int | float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LossContractError(
            "backend scaling divisor must be numeric",
            code="loss.backend_scaling_divisor",
            context={"field": name, "value_type": type(value).__name__},
        )
    divisor = float(value)
    if divisor <= 0.0:
        raise LossContractError(
            "backend scaling divisor must be positive",
            code="loss.backend_scaling_divisor",
            context={"field": name, "value": value},
        )
    return divisor


__all__ = [
    "BackendScalingReceipt",
    "PlannedStepLossSlice",
    "SegmentBalancedDenominator",
    "SegmentBalancedLossResult",
    "build_segment_balanced_denominator",
    "reduce_segment_balanced_planned_step",
    "segment_balanced_contribution",
    "validate_planned_step_backend_scaling",
]
