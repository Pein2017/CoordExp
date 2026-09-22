"""Planned-step schedule resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.common.errors import ConfigContractError
from src.config.models import CadenceConfig, RuntimeBatchResolution, TrainConfig
from src.config.resolve import resolve_effective_batch_runtime


@dataclass(frozen=True)
class StepScheduleEvent:
    planned_step_id: int
    event: str
    trigger_reasons: tuple[str, ...]
    source_config_path: str | None
    deduped_from: tuple[str, ...]
    required: bool

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "planned_step_id": self.planned_step_id,
            "event": self.event,
            "trigger_reasons": list(self.trigger_reasons),
            "source_config_path": self.source_config_path,
            "deduped_from": list(self.deduped_from),
            "required": self.required,
        }


@dataclass(frozen=True)
class ResolvedStepSchedule:
    resolved_max_steps: int
    packs_per_epoch: int
    requested_pack_presentations: int
    actual_pack_presentations: int
    tail_fill_pack_count: int
    runtime_batch: RuntimeBatchResolution
    events: dict[str, tuple[StepScheduleEvent, ...]]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "resolved_max_steps": self.resolved_max_steps,
            "packs_per_epoch": self.packs_per_epoch,
            "requested_pack_presentations": self.requested_pack_presentations,
            "actual_pack_presentations": self.actual_pack_presentations,
            "tail_fill_pack_count": self.tail_fill_pack_count,
            "runtime_batch": self.runtime_batch.to_artifact_dict(),
            "events": {
                name: [event.to_artifact_dict() for event in event_list]
                for name, event_list in sorted(self.events.items())
            },
        }


def resolve_planned_step_schedule(
    config: TrainConfig,
    *,
    packs_per_epoch: int,
    world_size: int,
    source_config_path: str | None = None,
) -> ResolvedStepSchedule:
    if packs_per_epoch <= 0:
        raise ConfigContractError(
            "packs_per_epoch must be positive before schedule resolution",
            code="schedule.packs_per_epoch",
            context={"packs_per_epoch": packs_per_epoch},
        )
    runtime_batch = resolve_effective_batch_runtime(config, world_size=world_size)
    effective_batch_size = runtime_batch.effective_batch_size

    if config.training.max_steps is not None:
        resolved_max_steps = config.training.max_steps
        requested_pack_presentations = resolved_max_steps * effective_batch_size
        actual_pack_presentations = requested_pack_presentations
        tail_fill_pack_count = 0
    else:
        requested_pack_presentations = packs_per_epoch * config.training.epochs
        resolved_max_steps = math.ceil(requested_pack_presentations / effective_batch_size)
        actual_pack_presentations = resolved_max_steps * effective_batch_size
        tail_fill_pack_count = actual_pack_presentations - requested_pack_presentations

    eval_forward_events = _resolve_cadence_events(
        config.eval.forward,
        event_name="eval.forward",
        resolved_max_steps=resolved_max_steps,
        source_config_path=source_config_path,
        required=False,
        include_final=False,
    )
    if eval_forward_events and config.data.eval is None:
        raise ConfigContractError(
            "scheduled eval.forward requires an explicit eval data source",
            code="schedule.eval_forward_source_required",
            context={
                "event_steps": [event.planned_step_id for event in eval_forward_events],
                "source_config_path": source_config_path,
            },
        )

    events = {
        "checkpoint": _resolve_cadence_events(
            config.checkpoint,
            event_name="checkpoint",
            resolved_max_steps=resolved_max_steps,
            source_config_path=source_config_path,
            required=False,
            include_final=config.checkpoint.save_final,
        ),
        "eval.forward": eval_forward_events,
        "final": (
            StepScheduleEvent(
                planned_step_id=resolved_max_steps,
                event="final",
                trigger_reasons=("final",),
                source_config_path=source_config_path,
                deduped_from=(),
                required=True,
            ),
        ),
    }
    return ResolvedStepSchedule(
        resolved_max_steps=resolved_max_steps,
        packs_per_epoch=packs_per_epoch,
        requested_pack_presentations=requested_pack_presentations,
        actual_pack_presentations=actual_pack_presentations,
        tail_fill_pack_count=tail_fill_pack_count,
        runtime_batch=runtime_batch,
        events=events,
    )


def _resolve_cadence_events(
    cadence: CadenceConfig,
    *,
    event_name: str,
    resolved_max_steps: int,
    source_config_path: str | None,
    required: bool,
    include_final: bool,
) -> tuple[StepScheduleEvent, ...]:
    reasons_by_step: dict[int, list[str]] = {}
    for step in cadence.steps:
        _validate_step_in_range(
            step,
            resolved_max_steps=resolved_max_steps,
            event_name=event_name,
        )
        reasons_by_step.setdefault(step, []).append("explicit_step")

    if cadence.every_fraction is not None:
        interval = max(1, math.ceil(cadence.every_fraction * resolved_max_steps))
        current = interval
        while current < resolved_max_steps:
            reasons_by_step.setdefault(current, []).append(
                f"every_fraction:{cadence.every_fraction:g}"
            )
            current += interval
        reasons_by_step.setdefault(resolved_max_steps, []).append(
            f"every_fraction:{cadence.every_fraction:g}:clamped_final"
        )

    if include_final:
        reasons_by_step.setdefault(resolved_max_steps, []).append("save_final")

    events: list[StepScheduleEvent] = []
    for step in sorted(reasons_by_step):
        reasons = tuple(reasons_by_step[step])
        events.append(
            StepScheduleEvent(
                planned_step_id=step,
                event=event_name,
                trigger_reasons=reasons,
                source_config_path=source_config_path,
                deduped_from=reasons[1:],
                required=required or "save_final" in reasons,
            )
        )
    return tuple(events)


def _validate_step_in_range(
    step: int,
    *,
    resolved_max_steps: int,
    event_name: str,
) -> None:
    if step < 1 or step > resolved_max_steps:
        raise ConfigContractError(
            "cadence step is outside resolved planned-step range",
            code="schedule.step_out_of_range",
            context={
                "event": event_name,
                "step": step,
                "resolved_max_steps": resolved_max_steps,
            },
        )
