#!/usr/bin/env python3
"""Experiment-local packed panel-step adapter for the Human-13 probe.

The module plans CPU-only no-padding packs and exposes one planned-step training
entry.  It does not add Human-13 admission to the ordinary training config or
implement a second trainer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
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
    segment_id: str = ""
    manifest_row_ids: tuple[str, ...] = ()

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
class Human13ManifestIdentity:
    schema_version: str
    unit_id: str
    panel_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class SealedHuman13Manifest:
    manifest: Any
    manifest_sha256: str


@dataclass(frozen=True)
class Human13SegmentBinding:
    segment_id: str
    image_id: int
    role: LogicalRole
    start: int
    end: int


@dataclass(frozen=True)
class Human13ExecutionPlan:
    """Manifest-bound payload admitted to the one-exposure runner."""

    manifest_identity: Human13ManifestIdentity
    arm_id: str
    denominators: Human13PanelDenominators
    coefficients: tuple[tuple[LossFamily, float], ...]
    pack_segments: tuple[tuple[int, tuple[Human13SegmentBinding, ...]], ...]
    sites_by_pack: tuple[tuple[int, tuple[Human13LossSite, ...]], ...]


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


def build_execution_plan(
    sealed_manifest: SealedHuman13Manifest | Any,
    *,
    arm_id: str,
    packed_plan: PackedPanelPlan,
    sites_by_pack: Mapping[int, Sequence[Human13LossSite]],
) -> Human13ExecutionPlan:
    """Bind one immutable arm payload to its sealed manifest and physical packs."""

    sealed = _coerce_sealed_manifest(sealed_manifest)
    manifest = sealed.manifest
    _require_full_panel_manifest(manifest)
    declared_arms = {item.arm_id for item in manifest.arms}
    if arm_id not in declared_arms:
        raise ValueError(f"arm {arm_id!r} is absent from the sealed manifest")
    coefficients, expected = _arm_contract(arm_id)
    frozen_sites = tuple(
        (pack.pack.pack_index, tuple(sites_by_pack.get(pack.pack.pack_index, ())))
        for pack in packed_plan.packs
    )
    if any(not sites for _, sites in frozen_sites):
        raise ValueError("every Human-13 pack must contain a manifest-bound loss site")
    denominators = _manifest_denominators(manifest, arm_id, coefficients)
    execution = Human13ExecutionPlan(
        manifest_identity=_manifest_identity(sealed),
        arm_id=arm_id,
        denominators=denominators,
        coefficients=coefficients,
        pack_segments=tuple(
            (pack.pack.pack_index, _segment_bindings(pack))
            for pack in packed_plan.packs
        ),
        sites_by_pack=frozen_sites,
    )
    _validate_sites_against_manifest(
        manifest,
        execution,
        packed_plan=packed_plan,
        expected_h=expected,
    )
    return execution


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
        terms: list[dict[str, Any]] = []
        for family, _weight in plan.coefficients:
            family_terms = [
                term
                for artifact in artifacts
                for term in artifact["terms"]
                if term["name"] == family
            ]
            if not family_terms:
                continue
            representative = dict(family_terms[0])
            representative["raw_loss"] = sum(
                float(term["raw_loss"]) for term in family_terms
            )
            representative["weighted_loss"] = sum(
                float(term["weighted_loss"]) for term in family_terms
            )
            representative["segment_mean_numerator"] = sum(
                float(term["segment_mean_numerator"]) for term in family_terms
            )
            representative["selected_count"] = sum(
                int(term["selected_count"]) for term in family_terms
            )
            denominator = dict(representative["denominator"])
            denominator["selected_atom_count"] = representative["selected_count"]
            representative["denominator"] = denominator
            representative["diagnostics"] = _merge_family_diagnostics(family_terms)
            terms.append(representative)
        return {
            "total_loss": sum(float(item["total_loss"]) for item in artifacts),
            "micro_step_count": len(artifacts),
            "denominators": dict(plan.denominators.family_counts),
            "denominator_scope": "complete_panel",
            "terms": terms,
            "metrics": {
                "loss/total": sum(float(item["total_loss"]) for item in artifacts),
                **{f"loss/{term['name']}": float(term["raw_loss"]) for term in terms},
            },
        }


def load_sealed_training_manifest(path: str | Path) -> SealedHuman13Manifest:
    """Load the canonical digest-bound manifest and require the full panel."""

    manifest_path = Path(path)
    manifest = load_manifest(manifest_path, require_full_panel=True)
    _require_full_panel_manifest(manifest)
    return SealedHuman13Manifest(
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


def _require_full_panel_manifest(manifest: Any) -> Any:
    if getattr(manifest, "full_panel", False) is not True:
        raise ValueError("training requires a sealed full-panel Human-13 manifest")
    return manifest


def run_panel_exposure(
    *,
    manifest_path: str | Path,
    execution_plan: Human13ExecutionPlan,
    model: Any,
    micro_steps: Sequence[SupervisedMicroStep],
    runtime: Any,
    run_writer: Any,
    checkpoint_writer: Any,
    checkpoint_kwargs: Mapping[str, Any],
    updated_at: str,
    gpu_seconds: float | None = None,
    wall_time_seconds: float | None = None,
    peak_memory_bytes: int | None = None,
    qwen_forward: Callable[[Any, SupervisedMicroStep], Any] | None = None,
) -> SupervisedTrainingResult:
    """Execute exactly one existing planned trainer step over all panel packs."""

    packs = tuple(micro_steps)
    initial_optimizer_steps = int(getattr(runtime, "optimizer_step_count", 0))
    checkpoint_count = 0
    completed_steps = 0
    consumed_packs = 0
    terminal_optimizer_status: str | None = None
    terminal_finite_status: str | None = None

    def finalize_failed(error: BaseException) -> None:
        applied = (
            int(getattr(runtime, "optimizer_step_count", 0)) - initial_optimizer_steps
        )
        run_writer.finalize(
            status="failed",
            updated_at=updated_at,
            completed_steps=completed_steps,
            consumed_packs=consumed_packs,
            checkpoint_event_count=checkpoint_count,
            optimizer_update_status=(
                terminal_optimizer_status or ("applied" if applied == 1 else None)
            ),
            finite_status=terminal_finite_status,
            terminal_error=f"{type(error).__name__}: {error}",
        )

    def write_final(_event: StepScheduleEvent, _observation: Any) -> None:
        nonlocal checkpoint_count, completed_steps, consumed_packs
        nonlocal terminal_optimizer_status, terminal_finite_status
        consumed_packs = int(_observation.micro_step_count)
        terminal_optimizer_status = _observation.optimizer_update_status
        terminal_finite_status = _observation.finite_status
        applied = (
            int(getattr(runtime, "optimizer_step_count", 0)) - initial_optimizer_steps
        )
        if (
            _observation.optimizer_update_status != "applied"
            or _observation.finite_status != "finite"
            or applied != 1
        ):
            raise RuntimeError(
                "Human-13 checkpoint requires one finite applied panel update"
            )
        completed_steps = 1
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

    try:
        sealed = load_sealed_training_manifest(manifest_path)
        _validate_execution_payload(sealed, execution_plan, packs)
        if not packs:
            raise ValueError("one panel exposure requires at least one physical pack")
        trainer = SupervisedTrainer(
            model=model,
            schedule=_one_exposure_schedule(pack_count=len(packs)),
            pack_stream=packs,
            qwen_forward=qwen_forward,
            loss_context_factory=human13_loss_context_factory,
            loss_runner=Human13PanelLossRunner(
                denominators=execution_plan.denominators,
                coefficients=execution_plan.coefficients,
            ),
            runtime=runtime,
            on_final=write_final,
        )
        result = trainer.run()
        completed_steps = result.completed_steps
        consumed_packs = result.consumed_micro_steps
        applied_steps = (
            int(getattr(runtime, "optimizer_step_count", 0)) - initial_optimizer_steps
        )
        if applied_steps != 1:
            raise RuntimeError(
                "Human-13 exposure must apply exactly one optimizer step, "
                f"got {applied_steps}"
            )
    except BaseException as error:
        finalize_failed(error)
        raise
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
    try:
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
                "loss_bundle": (
                    None if latest is None else dict(latest.loss_bundle_artifact)
                ),
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
    except BaseException as error:
        finalize_failed(error)
        raise
    return result


def _coerce_sealed_manifest(
    value: SealedHuman13Manifest | Any,
) -> SealedHuman13Manifest:
    if isinstance(value, SealedHuman13Manifest):
        return value
    digest = getattr(value, "canonical_sha256", None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Human-13 manifest is not canonically digest bound")
    return SealedHuman13Manifest(manifest=value, manifest_sha256=digest)


def _manifest_identity(sealed: SealedHuman13Manifest) -> Human13ManifestIdentity:
    manifest = sealed.manifest
    return Human13ManifestIdentity(
        schema_version=str(manifest.schema_version),
        unit_id=str(manifest.binding.unit_id),
        panel_sha256=str(manifest.binding.panel.panel_sha256),
        manifest_sha256=sealed.manifest_sha256,
    )


def _arm_contract(
    arm_id: str,
) -> tuple[tuple[tuple[LossFamily, float], ...], tuple[LossObjective, LogicalRole]]:
    if arm_id == "A1":
        return (("h", 1.0), ("replay", 1.0), ("duplicate", 1.0)), (
            "owner_ce",
            "a1_full_h",
        )
    if arm_id in {"A8", "A8-prime", "A8_prime"}:
        return (("h", 1.0), ("replay", 1.0), ("duplicate", 1.0)), (
            "bottleneck",
            "a8_full_h",
        )
    if arm_id == "A4":
        return (("h", 1.0), ("replay", 1.0), ("duplicate", 1.0)), (
            "union_mass",
            "a4_union",
        )
    if arm_id in {"full_gt", "full-GT", "full_gt_capacity"}:
        return (("full_gt", 1.0),), ("owner_ce", "full_gt")
    raise ValueError(f"arm {arm_id!r} is not admitted by the packed panel runner")


def _manifest_denominators(
    manifest: Any,
    arm_id: str,
    coefficients: tuple[tuple[LossFamily, float], ...],
) -> Human13PanelDenominators:
    source = manifest.denominators
    counts = {
        "h": int(
            source.target_image_count if arm_id == "A4" else source.target_owner_count
        ),
        "replay": int(source.replay_owner_count),
        "duplicate": int(source.duplicate_image_count),
        "full_gt": int(manifest.binding.panel.owner_count),
    }
    active = tuple(
        (family, counts[family]) for family, weight in coefficients if weight
    )
    duplicate_counts = tuple(
        (image.image_id, len(image.duplicate_events))
        for image in manifest.images
        if image.duplicate_events
    )
    return Human13PanelDenominators(
        family_counts=active,
        duplicate_event_counts=duplicate_counts,
    )


def _segment_bindings(pack: PackedPanelMicroStep) -> tuple[Human13SegmentBinding, ...]:
    packed_by_id = {item.example_id: item for item in pack.pack.segments}
    return tuple(
        Human13SegmentBinding(
            segment_id=logical.segment_id,
            image_id=logical.image_id,
            role=logical.role,
            start=packed_by_id[logical.segment_id].start,
            end=packed_by_id[logical.segment_id].end,
        )
        for logical in pack.logical_segments
    )


def _validate_sites_against_manifest(
    manifest: Any,
    execution: Human13ExecutionPlan,
    *,
    packed_plan: PackedPanelPlan,
    expected_h: tuple[LossObjective, LogicalRole],
) -> None:
    del packed_plan
    segments = {
        binding.segment_id: (pack_index, binding)
        for pack_index, bindings in execution.pack_segments
        for binding in bindings
    }
    images = {image.image_id: image for image in manifest.images}
    selected_by_row = {
        row.row_id: row for image in manifest.images for row in image.selected_rows
    }
    seen_rows: dict[str, list[tuple[int, str]]] = {}
    seen_segment_ids: set[str] = set()
    for pack_index, sites in execution.sites_by_pack:
        for site in sites:
            location = segments.get(site.segment_id)
            if location is None or location[0] != pack_index:
                raise ValueError(
                    "loss site is not bound to its logical segment and pack"
                )
            binding = location[1]
            seen_segment_ids.add(site.segment_id)
            if binding.image_id != site.image_id or any(
                position < binding.start or position >= binding.end
                for position in site.logits_positions
            ):
                raise ValueError("loss site escapes its manifest-bound logical segment")
            if not site.manifest_row_ids or len(set(site.manifest_row_ids)) != len(
                site.manifest_row_ids
            ):
                raise ValueError("loss site requires unique canonical manifest rows")
            expected_objective, expected_role = expected_h
            if site.family == "h" and (
                site.objective != expected_objective or binding.role != expected_role
            ):
                if expected_role == "a4_union":
                    raise ValueError(
                        "A4 candidate group must be complete in one segment and one pack"
                    )
                raise ValueError(
                    "arm loss site has an unrelated objective or logical role"
                )
            if site.family == "h" and expected_role in {
                "a1_full_h",
                "a8_full_h",
            }:
                if len(site.manifest_row_ids) != 1:
                    raise ValueError("H loss site must bind exactly one selected row")
                selected = selected_by_row.get(site.manifest_row_ids[0])
                expected_targets = (
                    ()
                    if selected is None
                    else tuple(
                        token
                        for token, included in zip(
                            selected.token_ids,
                            selected.target_token_mask,
                            strict=True,
                        )
                        if included
                    )
                )
                if expected_targets != site.target_token_ids:
                    raise ValueError("H loss targets mismatch the frozen selected row")
            if site.family == "replay" and (
                site.objective != "owner_ce" or binding.role != "source_replay"
            ):
                raise ValueError(
                    "replay site has an unrelated objective or logical role"
                )
            if site.family == "duplicate" and (
                site.objective != "duplicate_unlikelihood"
                or binding.role != "duplicate_event"
            ):
                raise ValueError(
                    "duplicate site has an unrelated objective or logical role"
                )
            if site.family == "full_gt" and binding.role != "full_gt":
                raise ValueError("full-GT site has an unrelated logical role")
            for row_id in site.manifest_row_ids:
                seen_rows.setdefault(row_id, []).append((pack_index, site.segment_id))

    if seen_segment_ids != set(segments):
        raise ValueError("logical segment membership is partial or unrelated")
    allowed_roles = {expected_h[1], "source_replay", "duplicate_event"}
    if any(
        binding.image_id not in images or binding.role not in allowed_roles
        for _pack, bindings in execution.pack_segments
        for binding in bindings
    ):
        raise ValueError("logical segments mismatch the sealed arm and panel images")
    observed_role_images = {
        role: {
            binding.image_id
            for _pack, bindings in execution.pack_segments
            for binding in bindings
            if binding.role == role
        }
        for role in allowed_roles
    }
    expected_role_images = {
        expected_h[1]: {
            image.image_id
            for image in manifest.images
            if (
                image.candidate_row_ids
                if expected_h[1] == "a4_union"
                else image.owners
                if expected_h[1] == "full_gt"
                else image.h_owner_ids
            )
        },
        "source_replay": {
            image.image_id for image in manifest.images if image.replay_row_ids
        },
        "duplicate_event": {
            image.image_id for image in manifest.images if image.duplicate_events
        },
    }
    if any(
        observed_role_images[role] != expected_images
        for role, expected_images in expected_role_images.items()
    ):
        raise ValueError(
            "logical segment image coverage mismatches the sealed manifest"
        )

    if expected_h == ("union_mass", "a4_union"):
        for image_id, image in images.items():
            expected_rows = set(image.candidate_row_ids)
            if not expected_rows:
                continue
            observed = {
                row_id: locations
                for row_id, locations in seen_rows.items()
                if row_id in expected_rows
            }
            locations = {
                location for values in observed.values() for location in values
            }
            if (
                set(observed) != expected_rows
                or any(len(values) != 1 for values in observed.values())
                or len(locations) != 1
            ):
                raise ValueError(
                    "A4 candidate group must be complete in one segment and one pack"
                )
            only = next(iter(locations))
            binding = segments[only[1]][1]
            if binding.image_id != image_id or binding.role != "a4_union":
                raise ValueError(
                    "A4 candidate group must be complete in one segment and one pack"
                )
    else:
        expected_rows = (
            {owner.owner_id for image in manifest.images for owner in image.owners}
            if expected_h[1] == "full_gt"
            else set(selected_by_row)
        )
        observed_rows = {
            row_id
            for row_id, locations in seen_rows.items()
            if any(
                site.family in {"h", "full_gt"} and row_id in site.manifest_row_ids
                for _pack, sites in execution.sites_by_pack
                for site in sites
            )
            for _location in locations
        }
        if expected_rows and observed_rows != expected_rows:
            raise ValueError(
                "arm payload is partial or mismatched to selected manifest rows"
            )

    family_rows = {
        family: [
            row_id
            for _pack, sites in execution.sites_by_pack
            for site in sites
            if site.family == family
            for row_id in site.manifest_row_ids
        ]
        for family in ("replay", "duplicate")
    }
    expected_replay = {
        row_id for image in manifest.images for row_id in image.replay_row_ids
    }
    expected_duplicate = {
        event.event_id for image in manifest.images for event in image.duplicate_events
    }
    for family, expected_rows in (
        ("replay", expected_replay),
        ("duplicate", expected_duplicate),
    ):
        observed = family_rows[family]
        if set(observed) != expected_rows or len(observed) != len(expected_rows):
            raise ValueError(
                f"{family} payload is partial, duplicated, or mismatched to manifest"
            )


def _validate_execution_payload(
    sealed_value: SealedHuman13Manifest | Any,
    execution: Human13ExecutionPlan,
    packs: tuple[SupervisedMicroStep, ...],
) -> None:
    sealed = _coerce_sealed_manifest(sealed_value)
    if execution.manifest_identity != _manifest_identity(sealed):
        raise ValueError("execution plan does not match the sealed manifest identity")
    manifest = sealed.manifest
    coefficients, expected_h = _arm_contract(execution.arm_id)
    if execution.arm_id not in {item.arm_id for item in manifest.arms}:
        raise ValueError("execution arm is absent from the sealed manifest")
    if (
        coefficients != execution.coefficients
        or execution.denominators
        != _manifest_denominators(manifest, execution.arm_id, coefficients)
    ):
        raise ValueError(
            "execution arm or global denominators mismatch sealed manifest"
        )
    expected_packs = dict(execution.pack_segments)
    expected_sites = dict(execution.sites_by_pack)
    if tuple(sorted(expected_packs)) != tuple(range(len(packs))):
        raise ValueError("execution pack membership is partial or noncanonical")
    for micro_step in packs:
        index = micro_step.pack.pack_index
        actual_segments = tuple(item.example_id for item in micro_step.pack.segments)
        if actual_segments != tuple(item.segment_id for item in expected_packs[index]):
            raise ValueError("micro-step pack membership mismatches execution plan")
        metadata = micro_step.metadata
        if (
            metadata.get("human13_panel_denominators") != execution.denominators
            or metadata.get("human13_loss_sites") != expected_sites[index]
        ):
            raise ValueError("micro-step loss payload mismatches execution plan")
    synthetic_plan = PackedPanelPlan((), (), GLOBAL_MAX_LENGTH)
    _validate_sites_against_manifest(
        manifest, execution, packed_plan=synthetic_plan, expected_h=expected_h
    )


def _merge_family_diagnostics(terms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics = [term["diagnostics"] for term in terms]
    merged: dict[str, Any] = {}
    for name in ("raw_event_count", "consumed_event_count", "capped_event_count"):
        if any(name in item for item in diagnostics):
            merged[name] = sum(int(item.get(name, 0)) for item in diagnostics)
    weights = [
        float(weight)
        for item in diagnostics
        for weight in item.get("candidate_weights", ())
    ]
    if weights:
        merged["candidate_weights"] = weights
        merged["effective_owner_count"] = sum(
            float(item.get("effective_owner_count", 0.0)) for item in diagnostics
        )
        merged["candidate_groups"] = [
            dict(group)
            for item in diagnostics
            for group in item.get("candidate_groups", ())
        ]
    return merged


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
        candidate_weights: list[float] = []
        effective_owner_count = 0.0
        candidate_groups: list[dict[str, Any]] = []
        for image_id, group in groups.items():
            logits, targets, mask = _padded_site_tensors(context, group)
            local = prefix_free_union_negative_log_mass(logits, targets, mask)
            numerator = numerator + local.numerator
            candidate_weights.extend(local.candidate_weights)
            effective_owner_count += local.effective_owner_count
            candidate_groups.append(
                {
                    "image_id": image_id,
                    "candidate_weights": list(local.candidate_weights),
                    "effective_owner_count": local.effective_owner_count,
                    "candidate_count": local.candidate_count,
                }
            )
        return (
            numerator,
            selected_count,
            {
                "objective_kinds": [objective],
                "atomic_image_count": len(groups),
                "candidate_weights": candidate_weights,
                "effective_owner_count": effective_owner_count,
                "candidate_groups": candidate_groups,
            },
        )
    if objective == "duplicate_unlikelihood":
        numerator = context.logits.sum() * 0.0
        raw_event_count = 0
        consumed_event_count = 0
        capped_event_count = 0
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
            raw_event_count += local.raw_event_count
            consumed_event_count += local.consumed_event_count
            capped_event_count += local.capped_event_count
            numerator = numerator + (
                local.numerator * len(group) / denominators.duplicate_count(image_id)
            )
        return (
            numerator,
            selected_count,
            {
                "objective_kinds": [objective],
                "raw_event_count": raw_event_count,
                "consumed_event_count": consumed_event_count,
                "capped_event_count": capped_event_count,
            },
        )
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
    "Human13ExecutionPlan",
    "Human13LossSite",
    "Human13ManifestIdentity",
    "Human13PackLossContext",
    "Human13PanelDenominators",
    "Human13PanelLossRunner",
    "Human13SegmentBinding",
    "LogicalPanelSegment",
    "LogicalRole",
    "PackedPanelMicroStep",
    "PackedPanelPlan",
    "PanelPerformanceCounters",
    "SealedHuman13Manifest",
    "build_execution_plan",
    "build_supervised_micro_steps",
    "build_logical_segments",
    "dry_run_receipt",
    "human13_loss_context_factory",
    "load_sealed_training_manifest",
    "plan_panel_packs",
    "run_panel_exposure",
]
