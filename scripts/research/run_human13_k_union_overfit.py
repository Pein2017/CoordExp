#!/usr/bin/env python3
"""Experiment-local packed panel-step adapter for the Human-13 probe.

The module plans CPU-only no-padding packs and exposes one planned-step training
entry.  It does not add Human-13 admission to the ordinary training config or
implement a second trainer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias

import torch

from scripts.research.build_human13_k_union_manifest import load_manifest
from src.config.models import RuntimeBatchResolution
from src.losses.human13_k_union import (
    coherent_full_chain_bottleneck_hinge,
    image_balanced_duplicate_token_unlikelihood,
    owner_mean_masked_row_cross_entropy,
    prefix_free_union_negative_log_mass,
)
from src.losses.normalizers import SegmentBalancedDenominator
from src.losses.runner import LossBundle, LossTermResult
from src.packing.planner import PackedSequence, plan_packed_sequences
from src.qwen.fa2 import Fa2VarlenPlan, build_fa2_varlen_plan
from src.qwen.positions import QwenPositionInputs, build_qwen_position_inputs
from src.training.schedule import ResolvedStepSchedule, StepScheduleEvent
from src.training.supervised_trainer import (
    SupervisedMicroStep,
    SupervisedTrainer,
    SupervisedTrainingResult,
)


GLOBAL_MAX_LENGTH = 12_000

LogicalRole: TypeAlias = Literal[
    "a1_full_h",
    "a8_full_h",
    "full_gt",
    "a4_union",
    "source_replay",
    "duplicate_event",
]

_COHERENT_ROLES = frozenset({"a1_full_h", "a8_full_h", "full_gt"})
_KNOWN_ROLES = _COHERENT_ROLES | {
    "a4_union",
    "source_replay",
    "duplicate_event",
}

LossFamily: TypeAlias = Literal["h", "replay", "duplicate", "full_gt"]
LossObjective: TypeAlias = Literal[
    "owner_ce", "union_mass", "bottleneck", "duplicate_unlikelihood"
]
_LOSS_FAMILIES = frozenset({"h", "replay", "duplicate", "full_gt"})
_LOSS_OBJECTIVES = frozenset(
    {"owner_ce", "union_mass", "bottleneck", "duplicate_unlikelihood"}
)


@dataclass(frozen=True)
class LogicalPanelSegment:
    """One indivisible, independently encoded causal segment."""

    segment_id: str
    image_id: int
    role: LogicalRole
    encoded_example: Any

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("logical segment_id must be nonempty")
        if isinstance(self.image_id, bool) or self.image_id <= 0:
            raise ValueError("logical image_id must be a positive integer")
        if self.role not in _KNOWN_ROLES:
            raise ValueError(f"unknown Human-13 logical role: {self.role!r}")
        if getattr(self.encoded_example, "example_id", None) != self.segment_id:
            raise ValueError("encoded example_id must equal its logical segment_id")
        input_ids = getattr(self.encoded_example, "input_ids", None)
        if not isinstance(input_ids, tuple) or not input_ids:
            raise ValueError("logical segment requires nonempty tuple input_ids")

    @property
    def encoded_length(self) -> int:
        return len(self.encoded_example.input_ids)


@dataclass(frozen=True)
class PackedPanelMicroStep:
    pack: PackedSequence
    logical_segments: tuple[LogicalPanelSegment, ...]
    encoded_examples: tuple[Any, ...]
    position_inputs: QwenPositionInputs
    fa2_varlen_plan: Fa2VarlenPlan


@dataclass(frozen=True)
class Human13PanelDenominators:
    """Complete-panel counts repeated unchanged on every physical pack."""

    family_counts: tuple[tuple[LossFamily, int], ...]
    duplicate_event_counts: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.family_counts)
        if len(set(names)) != len(names) or any(
            name not in _LOSS_FAMILIES for name in names
        ):
            raise ValueError("Human-13 family denominators must be unique known names")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for _, count in self.family_counts
        ):
            raise ValueError("Human-13 family denominators must be positive integers")
        image_ids = tuple(image_id for image_id, _ in self.duplicate_event_counts)
        if len(set(image_ids)) != len(image_ids) or any(
            isinstance(image_id, bool)
            or image_id <= 0
            or isinstance(count, bool)
            or count <= 0
            for image_id, count in self.duplicate_event_counts
        ):
            raise ValueError("duplicate event counts must be positive and image-unique")

    def family_count(self, name: LossFamily) -> int:
        try:
            return dict(self.family_counts)[name]
        except KeyError as exc:
            raise ValueError(f"missing complete panel denominator for {name}") from exc

    def duplicate_count(self, image_id: int) -> int:
        try:
            return dict(self.duplicate_event_counts)[image_id]
        except KeyError as exc:
            raise ValueError(
                f"missing duplicate-event count for image {image_id}"
            ) from exc


@dataclass(frozen=True)
class Human13LossSite:
    """Already selected causal-logit sites; no research labels are rederived."""

    family: LossFamily
    objective: LossObjective
    unit_id: str
    image_id: int
    logits_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    required_margin: float | None = None

    def __post_init__(self) -> None:
        if self.family not in _LOSS_FAMILIES or self.objective not in _LOSS_OBJECTIVES:
            raise ValueError("Human-13 loss site has an unknown family or objective")
        if not self.unit_id or self.image_id <= 0:
            raise ValueError("Human-13 loss sites require unit and image identities")
        if not self.logits_positions or len(self.logits_positions) != len(
            self.target_token_ids
        ):
            raise ValueError("loss positions and targets must be nonempty and aligned")
        if any(position < 0 for position in self.logits_positions):
            raise ValueError("loss logits positions must be non-negative")
        if self.objective == "bottleneck" and (
            self.required_margin is None
            or not math.isfinite(self.required_margin)
            or self.required_margin <= 0
        ):
            raise ValueError("bottleneck sites require one positive finite margin")


@dataclass(frozen=True)
class Human13PackLossContext:
    logits: torch.Tensor
    logits_position_ids: tuple[int, ...] | None
    sites: tuple[Human13LossSite, ...]

    def __post_init__(self) -> None:
        if self.logits.ndim != 3 or self.logits.shape[0] != 1:
            raise ValueError("Human-13 loss context logits must have shape [1,N,V]")
        if self.logits_position_ids is not None and len(
            self.logits_position_ids
        ) != int(self.logits.shape[1]):
            raise ValueError("compact logits positions must match the logits row count")


@dataclass(frozen=True)
class Human13CompactLogitsMetadata:
    selected_causal_logits_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        positions = self.selected_causal_logits_positions
        if not positions or tuple(sorted(set(positions))) != positions:
            raise ValueError(
                "Human-13 compact logits positions must be sorted and unique"
            )


@dataclass(frozen=True)
class Human13PlannedLossPlan:
    denominators: Human13PanelDenominators
    coefficients: tuple[tuple[LossFamily, float], ...]
    micro_step_count: int


@dataclass(frozen=True)
class PanelPerformanceCounters:
    pack_count: int
    logical_tokens: int
    packed_tokens: int
    padding_tokens: int
    utilization: float
    gpu_seconds: float | None
    wall_time_seconds: float | None
    peak_memory_bytes: int | None

    def to_artifact_dict(self) -> dict[str, int | float | None]:
        return {
            "pack_count": self.pack_count,
            "logical_tokens": self.logical_tokens,
            "packed_tokens": self.packed_tokens,
            "padding_tokens": self.padding_tokens,
            "utilization": self.utilization,
            "gpu_seconds": self.gpu_seconds,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass(frozen=True)
class PackedPanelPlan:
    logical_segments: tuple[LogicalPanelSegment, ...]
    packs: tuple[PackedPanelMicroStep, ...]
    global_max_length: int
    packing_claims: tuple[str, ...] = (
        "zero_padding",
        "physical_launch_reduction",
    )

    def performance_counters(
        self,
        *,
        gpu_seconds: float | None = None,
        wall_time_seconds: float | None = None,
        peak_memory_bytes: int | None = None,
    ) -> PanelPerformanceCounters:
        logical_tokens = sum(item.encoded_length for item in self.logical_segments)
        packed_tokens = sum(item.pack.length for item in self.packs)
        capacity = len(self.packs) * self.global_max_length
        return PanelPerformanceCounters(
            pack_count=len(self.packs),
            logical_tokens=logical_tokens,
            packed_tokens=packed_tokens,
            padding_tokens=0,
            utilization=packed_tokens / capacity,
            gpu_seconds=_optional_nonnegative_float(gpu_seconds, "gpu_seconds"),
            wall_time_seconds=_optional_nonnegative_float(
                wall_time_seconds, "wall_time_seconds"
            ),
            peak_memory_bytes=_optional_nonnegative_int(
                peak_memory_bytes, "peak_memory_bytes"
            ),
        )


def build_logical_segments(
    segments: Sequence[LogicalPanelSegment],
) -> tuple[LogicalPanelSegment, ...]:
    """Validate the experiment's coherent and atomic segment boundaries."""

    checked = tuple(segments)
    if not checked:
        raise ValueError("Human-13 panel step requires at least one logical segment")
    ids = [item.segment_id for item in checked]
    if len(set(ids)) != len(ids):
        raise ValueError("logical segment IDs must be unique")

    seen_coherent: set[tuple[LogicalRole, int]] = set()
    seen_a4: set[int] = set()
    for item in checked:
        if item.role in _COHERENT_ROLES:
            key = (item.role, item.image_id)
            if key in seen_coherent:
                raise ValueError(
                    f"{item.role} must keep one coherent segment per image"
                )
            seen_coherent.add(key)
        if item.role == "a4_union":
            if item.image_id in seen_a4:
                raise ValueError("A4 must keep one atomic candidate group per image")
            seen_a4.add(item.image_id)
    return checked


def plan_panel_packs(
    segments: Sequence[LogicalPanelSegment],
    *,
    global_max_length: int = GLOBAL_MAX_LENGTH,
) -> PackedPanelPlan:
    """Stable descending-length first-fit over indivisible logical segments."""

    checked = build_logical_segments(segments)
    limit = _validate_pack_limit(global_max_length)
    for item in checked:
        if item.encoded_length > GLOBAL_MAX_LENGTH:
            raise ValueError(
                f"logical segment {item.segment_id!r} exceeds the 12,000-token "
                "hard preflight"
            )
        if item.encoded_length > limit:
            raise ValueError(
                f"logical segment {item.segment_id!r} exceeds the configured pack limit"
            )

    ordered = tuple(
        sorted(checked, key=lambda item: (-item.encoded_length, item.segment_id))
    )
    bins: list[list[LogicalPanelSegment]] = []
    bin_lengths: list[int] = []
    for item in ordered:
        destination = next(
            (
                index
                for index, length in enumerate(bin_lengths)
                if length + item.encoded_length <= limit
            ),
            None,
        )
        if destination is None:
            bins.append([item])
            bin_lengths.append(item.encoded_length)
        else:
            bins[destination].append(item)
            bin_lengths[destination] += item.encoded_length

    packed_steps = tuple(
        _materialize_bin(pack_index, tuple(items), global_max_length=limit)
        for pack_index, items in enumerate(bins)
    )
    return PackedPanelPlan(
        logical_segments=ordered,
        packs=packed_steps,
        global_max_length=limit,
    )


def dry_run_receipt(plan: PackedPanelPlan) -> dict[str, Any]:
    """Return a plan receipt without importing or invoking any runtime action."""

    if not isinstance(plan, PackedPanelPlan):
        raise TypeError("dry_run_receipt requires a PackedPanelPlan")
    return {
        "mode": "dry_run",
        "actions": {
            "model_loads": 0,
            "forwards": 0,
            "backwards": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "gpu_allocations": 0,
        },
        "performance": plan.performance_counters().to_artifact_dict(),
        "packing_claims": list(plan.packing_claims),
    }


def build_supervised_micro_steps(
    plan: PackedPanelPlan,
    *,
    denominators: Human13PanelDenominators,
    token_sequences: Mapping[int, Any],
    vocab_groups: Any,
    sites_by_pack: Mapping[int, Sequence[Human13LossSite]],
    expected_vocab_size: int,
) -> tuple[SupervisedMicroStep, ...]:
    """Bridge a CPU pack plan onto the existing Qwen/trainer micro-step seam."""

    micro_steps: list[SupervisedMicroStep] = []
    for packed in plan.packs:
        pack_index = packed.pack.pack_index
        if pack_index not in token_sequences:
            raise ValueError(f"missing token sequence for Human-13 pack {pack_index}")
        sites = tuple(sites_by_pack.get(pack_index, ()))
        if not sites:
            raise ValueError(f"Human-13 pack {pack_index} has no selected loss sites")
        positions = tuple(
            sorted({position for site in sites for position in site.logits_positions})
        )
        micro_steps.append(
            SupervisedMicroStep(
                pack=packed.pack,
                encoded_examples=packed.encoded_examples,
                position_inputs=packed.position_inputs,
                token_sequence=token_sequences[pack_index],
                vocab_groups=vocab_groups,
                metadata={
                    "human13_panel_denominators": denominators,
                    "human13_loss_sites": sites,
                    "packing_claims": plan.packing_claims,
                },
                expected_vocab_size=expected_vocab_size,
                calibration_metadata=Human13CompactLogitsMetadata(positions),
            )
        )
    return tuple(micro_steps)


def human13_loss_context_factory(
    micro_step: SupervisedMicroStep,
    forward_result: Any,
) -> Human13PackLossContext:
    metadata = micro_step.metadata
    sites = (
        metadata.get("human13_loss_sites") if isinstance(metadata, Mapping) else None
    )
    if (
        not isinstance(sites, tuple)
        or not sites
        or not all(isinstance(site, Human13LossSite) for site in sites)
    ):
        raise ValueError("Human-13 micro-step has no valid frozen loss sites")
    return Human13PackLossContext(
        logits=forward_result.logits,
        logits_position_ids=forward_result.logits_position_ids,
        sites=sites,
    )


@dataclass(frozen=True)
class Human13PanelLossRunner:
    """Experiment-local streaming dispatch over the pure Human-13 objectives."""

    denominators: Human13PanelDenominators
    coefficients: tuple[tuple[LossFamily, float], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.coefficients)
        if len(set(names)) != len(names) or any(
            name not in _LOSS_FAMILIES for name in names
        ):
            raise ValueError("Human-13 coefficients must use unique known families")
        if any(
            not math.isfinite(weight) or weight < 0 for _, weight in self.coefficients
        ):
            raise ValueError("Human-13 coefficients must be finite and non-negative")
        for name, weight in self.coefficients:
            if weight > 0:
                self.denominators.family_count(name)

    def prepare_planned_step(
        self,
        micro_steps: Sequence[Any],
        *,
        denominator_gatherer: Any | None = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> Human13PlannedLossPlan:
        del denominator_gatherer, rank
        if world_size != 1:
            raise ValueError("each Human-13 arm must remain world-size one")
        checked = tuple(micro_steps)
        if not checked:
            raise ValueError("Human-13 planned step requires at least one pack")
        for micro_step in checked:
            metadata = getattr(micro_step, "metadata", None)
            observed = (
                metadata.get("human13_panel_denominators")
                if isinstance(metadata, Mapping)
                else None
            )
            if observed != self.denominators:
                raise ValueError("every pack must carry the complete panel denominator")
        return Human13PlannedLossPlan(
            denominators=self.denominators,
            coefficients=self.coefficients,
            micro_step_count=len(checked),
        )

    def compute_micro_step(
        self,
        context: Human13PackLossContext,
        plan: Human13PlannedLossPlan,
        *,
        local_micro_step_index: int,
    ) -> LossBundle:
        if not isinstance(context, Human13PackLossContext):
            raise TypeError("Human-13 loss runner requires Human13PackLossContext")
        if plan.denominators != self.denominators:
            raise ValueError("planned loss denominators changed after preflight")
        terms: list[LossTermResult] = []
        for family, weight in plan.coefficients:
            sites = tuple(site for site in context.sites if site.family == family)
            numerator, selected_count, diagnostics = _dispatch_family(
                context,
                sites,
                denominators=plan.denominators,
            )
            denominator_count = plan.denominators.family_count(family)
            raw_loss = numerator / denominator_count
            weighted_loss = raw_loss * weight
            denominator = SegmentBalancedDenominator(
                term_name=family,
                denominator_scope="planned_step",
                eligible_segment_count=denominator_count,
                selected_atom_count=selected_count,
                skipped_segment_count=0,
                context_count=plan.micro_step_count,
            )
            terms.append(
                LossTermResult(
                    name=family,
                    raw_loss=raw_loss,
                    weighted_loss=weighted_loss,
                    weight=weight,
                    segment_mean_numerator=numerator,
                    denominator=denominator,
                    reducer_name="human13_complete_panel",
                    selected_count=selected_count,
                    skipped_count=0,
                    math_dtype="float32",
                    token_weighted_diagnostic=raw_loss.detach(),
                    diagnostics={
                        "local_micro_step_index": local_micro_step_index,
                        "denominator_scope": "complete_panel",
                        **diagnostics,
                    },
                )
            )
        total = sum((term.weighted_loss for term in terms), context.logits.sum() * 0.0)
        finite = bool(torch.isfinite(total).item()) and all(
            bool(torch.isfinite(term.weighted_loss).item()) for term in terms
        )
        return LossBundle(
            total_loss=total,
            terms=tuple(terms),
            metrics={"loss/total": float(total.detach())},
            counts={
                "count/packs": 1,
                "count/sites": sum(len(s.logits_positions) for s in context.sites),
            },
            diagnostics={"denominator_scope": "complete_panel"},
            finite_status={
                "total_loss": "finite" if finite else "nonfinite",
                "terms": {
                    term.name: (
                        "finite"
                        if bool(torch.isfinite(term.weighted_loss).item())
                        else "nonfinite"
                    )
                    for term in terms
                },
            },
        )

    def finalize_planned_step(
        self,
        micro_loss_artifacts: Sequence[Mapping[str, Any]],
        plan: Human13PlannedLossPlan,
    ) -> dict[str, Any]:
        artifacts = tuple(micro_loss_artifacts)
        return {
            "total_loss": sum(float(item["total_loss"]) for item in artifacts),
            "micro_step_count": len(artifacts),
            "denominators": dict(plan.denominators.family_counts),
            "denominator_scope": "complete_panel",
        }


def load_sealed_training_manifest(path: str | Path) -> Any:
    """Load the canonical digest-bound manifest and require the full panel."""

    manifest = load_manifest(path, require_full_panel=True)
    return _require_full_panel_manifest(manifest)


def _require_full_panel_manifest(manifest: Any) -> Any:
    if getattr(manifest, "full_panel", False) is not True:
        raise ValueError("training requires a sealed full-panel Human-13 manifest")
    return manifest


def run_panel_exposure(
    *,
    manifest_path: str | Path,
    model: Any,
    micro_steps: Sequence[SupervisedMicroStep],
    loss_runner: Any,
    runtime: Any,
    run_writer: Any,
    checkpoint_writer: Any,
    checkpoint_kwargs: Mapping[str, Any],
    updated_at: str,
    gpu_seconds: float | None = None,
    wall_time_seconds: float | None = None,
    peak_memory_bytes: int | None = None,
    qwen_forward: Callable[[Any, SupervisedMicroStep], Any] | None = None,
    loss_context_factory: Callable[
        [SupervisedMicroStep, Any], Any
    ] = human13_loss_context_factory,
) -> SupervisedTrainingResult:
    """Execute exactly one existing planned trainer step over all panel packs."""

    manifest = load_sealed_training_manifest(manifest_path)
    _require_full_panel_manifest(manifest)
    packs = tuple(micro_steps)
    if not packs:
        raise ValueError("one panel exposure requires at least one physical pack")
    initial_optimizer_steps = int(getattr(runtime, "optimizer_step_count", 0))
    schedule = _one_exposure_schedule(pack_count=len(packs))
    checkpoint_count = 0

    def write_final(_event: StepScheduleEvent, _observation: Any) -> None:
        nonlocal checkpoint_count
        if (
            _observation.optimizer_update_status != "applied"
            or _observation.finite_status != "finite"
        ):
            raise RuntimeError(
                "Human-13 checkpoint requires one finite applied panel update"
            )
        payload = dict(checkpoint_kwargs)
        payload.update(
            {
                "step": 1,
                "model": model,
                "run_writer": run_writer,
                "is_final": True,
            }
        )
        if "accelerator" not in payload and hasattr(runtime, "accelerator"):
            payload["accelerator"] = runtime.accelerator
        checkpoint_writer.write_checkpoint(**payload)
        checkpoint_count += 1

    trainer = SupervisedTrainer(
        model=model,
        schedule=schedule,
        pack_stream=packs,
        qwen_forward=qwen_forward,
        loss_context_factory=loss_context_factory,
        loss_runner=loss_runner,
        runtime=runtime,
        on_final=write_final,
    )
    result = trainer.run()
    applied_steps = (
        int(getattr(runtime, "optimizer_step_count", 0)) - initial_optimizer_steps
    )
    if applied_steps != 1:
        raise RuntimeError(
            f"Human-13 exposure must apply exactly one optimizer step, got {applied_steps}"
        )
    latest = result.latest_observation
    packed_tokens = sum(
        int(getattr(micro_step.pack, "length", 1)) for micro_step in packs
    )
    pack_capacity = sum(
        int(
            getattr(
                micro_step.pack,
                "global_max_length",
                getattr(micro_step.pack, "length", 1),
            )
        )
        for micro_step in packs
    )
    performance = PanelPerformanceCounters(
        pack_count=len(packs),
        logical_tokens=packed_tokens,
        packed_tokens=packed_tokens,
        padding_tokens=0,
        utilization=packed_tokens / pack_capacity,
        gpu_seconds=_optional_nonnegative_float(gpu_seconds, "gpu_seconds"),
        wall_time_seconds=_optional_nonnegative_float(
            wall_time_seconds, "wall_time_seconds"
        ),
        peak_memory_bytes=_optional_nonnegative_int(
            peak_memory_bytes, "peak_memory_bytes"
        ),
    )
    run_writer.append_logging_row(
        {
            "split": "train",
            "step": 1,
            "non_finite_fields": [],
            "pack_count": len(packs),
            "performance": performance.to_artifact_dict(),
            "optimizer_update_status": (
                None if latest is None else latest.optimizer_update_status
            ),
            "finite_status": None if latest is None else latest.finite_status,
        }
    )
    run_writer.finalize(
        status="completed",
        updated_at=updated_at,
        completed_steps=result.completed_steps,
        consumed_packs=result.consumed_micro_steps,
        checkpoint_event_count=checkpoint_count,
        optimizer_update_status=(
            None if latest is None else latest.optimizer_update_status
        ),
        finite_status=None if latest is None else latest.finite_status,
    )
    return result


def _dispatch_family(
    context: Human13PackLossContext,
    sites: tuple[Human13LossSite, ...],
    *,
    denominators: Human13PanelDenominators,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    if not sites:
        return context.logits.sum() * 0.0, 0, {"objective_kinds": []}
    objectives = {site.objective for site in sites}
    if len(objectives) != 1:
        raise ValueError("one Human-13 family cannot mix objective geometries")
    objective = next(iter(objectives))
    selected_count = sum(len(site.logits_positions) for site in sites)

    if objective == "owner_ce":
        logits, targets, mask = _padded_site_tensors(context, sites)
        result = owner_mean_masked_row_cross_entropy(logits, targets, mask)
        return result.numerator, selected_count, {"objective_kinds": [objective]}
    if objective == "bottleneck":
        margins = {site.required_margin for site in sites}
        if len(margins) != 1:
            raise ValueError(
                "one coherent bottleneck family requires one frozen margin"
            )
        logits, targets, mask = _padded_site_tensors(context, sites)
        result = coherent_full_chain_bottleneck_hinge(
            logits,
            targets,
            mask,
            required_margin=float(next(iter(margins))),
        )
        return result.numerator, selected_count, {"objective_kinds": [objective]}
    if objective == "union_mass":
        numerator = context.logits.sum() * 0.0
        groups = _sites_by_image(sites)
        for group in groups.values():
            logits, targets, mask = _padded_site_tensors(context, group)
            numerator = (
                numerator
                + prefix_free_union_negative_log_mass(logits, targets, mask).numerator
            )
        return (
            numerator,
            selected_count,
            {
                "objective_kinds": [objective],
                "atomic_image_count": len(groups),
            },
        )
    if objective == "duplicate_unlikelihood":
        numerator = context.logits.sum() * 0.0
        for image_id, group in _sites_by_image(sites).items():
            if any(len(site.logits_positions) != 1 for site in group):
                raise ValueError("each duplicate event must select exactly one token")
            selected = torch.cat(
                tuple(_selected_site_logits(context, site) for site in group), dim=0
            )
            targets = torch.tensor(
                [site.target_token_ids[0] for site in group],
                dtype=torch.long,
                device=context.logits.device,
            )
            images = torch.full_like(targets, image_id)
            local = image_balanced_duplicate_token_unlikelihood(
                selected, targets, images
            )
            numerator = numerator + (
                local.numerator * len(group) / denominators.duplicate_count(image_id)
            )
        return numerator, selected_count, {"objective_kinds": [objective]}
    raise AssertionError(f"unhandled Human-13 objective: {objective}")


def _padded_site_tensors(
    context: Human13PackLossContext,
    sites: Sequence[Human13LossSite],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(site.logits_positions) for site in sites)
    vocab_size = int(context.logits.shape[-1])
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for site in sites:
        selected = _selected_site_logits(context, site)
        padding = max_length - len(site.logits_positions)
        if padding:
            selected = torch.cat(
                (selected, selected.new_zeros((padding, vocab_size))), dim=0
            )
        rows.append(selected)
        targets.append(
            torch.tensor(
                site.target_token_ids + (0,) * padding,
                dtype=torch.long,
                device=context.logits.device,
            )
        )
        masks.append(
            torch.tensor(
                (True,) * len(site.logits_positions) + (False,) * padding,
                dtype=torch.bool,
                device=context.logits.device,
            )
        )
    return torch.stack(rows), torch.stack(targets), torch.stack(masks)


def _selected_site_logits(
    context: Human13PackLossContext, site: Human13LossSite
) -> torch.Tensor:
    if context.logits_position_ids is None:
        rows = site.logits_positions
    else:
        row_by_position = {
            position: index
            for index, position in enumerate(context.logits_position_ids)
        }
        missing = [
            position
            for position in site.logits_positions
            if position not in row_by_position
        ]
        if missing:
            raise ValueError(f"compact logits omit selected Human-13 sites: {missing}")
        rows = tuple(row_by_position[position] for position in site.logits_positions)
    indices = torch.tensor(rows, dtype=torch.long, device=context.logits.device)
    return context.logits[0].index_select(0, indices).float()


def _sites_by_image(
    sites: Sequence[Human13LossSite],
) -> dict[int, tuple[Human13LossSite, ...]]:
    grouped: dict[int, list[Human13LossSite]] = {}
    for site in sites:
        grouped.setdefault(site.image_id, []).append(site)
    return {image_id: tuple(items) for image_id, items in grouped.items()}


def _one_exposure_schedule(*, pack_count: int) -> ResolvedStepSchedule:
    event = StepScheduleEvent(
        planned_step_id=1,
        event="final",
        trigger_reasons=("final",),
        source_config_path=None,
        deduped_from=(),
        required=True,
    )
    return ResolvedStepSchedule(
        resolved_max_steps=1,
        packs_per_epoch=pack_count,
        requested_pack_presentations=pack_count,
        actual_pack_presentations=pack_count,
        tail_fill_pack_count=0,
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=pack_count,
            resolved_grad_accum_steps=pack_count,
        ),
        events={"checkpoint": (), "eval.forward": (), "final": (event,)},
    )


def _materialize_bin(
    pack_index: int,
    items: tuple[LogicalPanelSegment, ...],
    *,
    global_max_length: int,
) -> PackedPanelMicroStep:
    examples = tuple(item.encoded_example for item in items)
    local_pack = plan_packed_sequences(
        examples,
        global_max_length=global_max_length,
    )
    if len(local_pack) != 1:
        raise AssertionError("first-fit bin must materialize as one no-padding pack")
    pack = _reindex_pack(local_pack[0], pack_index=pack_index)
    fa2_plan = build_fa2_varlen_plan(pack)
    positions = build_qwen_position_inputs(
        pack,
        examples,
        expected_cu_seq_lens=fa2_plan.segment_boundaries,
    )
    return PackedPanelMicroStep(
        pack=pack,
        logical_segments=items,
        encoded_examples=examples,
        position_inputs=positions,
        fa2_varlen_plan=fa2_plan,
    )


def _reindex_pack(pack: PackedSequence, *, pack_index: int) -> PackedSequence:
    segments = tuple(
        replace(segment, pack_index=pack_index) for segment in pack.segments
    )
    return replace(pack, pack_index=pack_index, segments=segments)


def _validate_pack_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("global_max_length must be a positive integer")
    if value > GLOBAL_MAX_LENGTH:
        raise ValueError("Human-13 global_max_length cannot exceed 12,000")
    return value


def _optional_nonnegative_float(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    checked = float(value)
    if checked < 0:
        raise ValueError(f"{field} must be non-negative")
    return checked


def _optional_nonnegative_int(value: int | None, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


__all__ = [
    "GLOBAL_MAX_LENGTH",
    "Human13CompactLogitsMetadata",
    "Human13LossSite",
    "Human13PackLossContext",
    "Human13PanelDenominators",
    "Human13PanelLossRunner",
    "LogicalPanelSegment",
    "LogicalRole",
    "PackedPanelMicroStep",
    "PackedPanelPlan",
    "PanelPerformanceCounters",
    "build_supervised_micro_steps",
    "build_logical_segments",
    "dry_run_receipt",
    "human13_loss_context_factory",
    "load_sealed_training_manifest",
    "plan_panel_packs",
    "run_panel_exposure",
]
