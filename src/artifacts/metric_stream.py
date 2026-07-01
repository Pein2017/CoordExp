"""Metric stream records for run artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.common.errors import ArtifactContractError


@dataclass(frozen=True)
class MetricStreamEvent:
    event_type: str
    planned_step_id: int
    split: str
    name: str
    value: float | None
    trigger_reasons: Sequence[str]
    optimizer_update_status: str
    finite_status: str
    warning_status: str
    reduction: str = "single_rank"
    rank: int | None = None
    world_size: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def selector_eligible(self) -> bool:
        return (
            self.value is not None
            and self.reduction != "rank_local"
            and self.optimizer_update_status == "applied"
            and self.finite_status == "finite"
        )

    def to_record(self) -> dict[str, Any]:
        _validate_metric_stream_event(self)
        return {
            "event_type": self.event_type,
            "planned_step_id": self.planned_step_id,
            "split": self.split,
            "name": self.name,
            "value": self.value,
            "trigger_reasons": [str(reason) for reason in self.trigger_reasons],
            "optimizer_update_status": self.optimizer_update_status,
            "finite_status": self.finite_status,
            "warning_status": self.warning_status,
            "reduction": self.reduction,
            "rank": self.rank,
            "world_size": self.world_size,
            "selector_eligible": self.selector_eligible,
            "metadata": dict(self.metadata),
        }


def metric_stream_events_from_runtime_payload(
    payload: Mapping[str, Any],
    *,
    event_type: str,
    trigger_reasons: Sequence[str],
    optimizer_update_status: str,
    finite_status: str,
    warning_status: str,
) -> tuple[MetricStreamEvent, ...]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ArtifactContractError(
            "runtime metric payload must include a metrics mapping",
            code="metric.runtime_payload_shape",
            context={"keys": sorted(str(key) for key in payload)},
        )
    return tuple(
        MetricStreamEvent(
            event_type=event_type,
            planned_step_id=int(payload["planned_step_id"]),
            split=str(payload["split"]),
            name=str(name),
            value=_optional_float(metrics[name]),
            trigger_reasons=trigger_reasons,
            optimizer_update_status=optimizer_update_status,
            finite_status=finite_status,
            warning_status=warning_status,
            reduction=str(payload.get("reduction", "single_rank")),
            rank=_optional_int(payload.get("rank")),
            world_size=_optional_int(payload.get("world_size")),
        )
        for name in sorted(metrics)
    )


def _validate_metric_stream_event(event: MetricStreamEvent) -> None:
    if event.planned_step_id <= 0:
        raise ArtifactContractError(
            "metric event planned_step_id must be positive",
            code="metric.planned_step_id",
            context={"planned_step_id": event.planned_step_id},
        )
    if not event.split:
        raise ArtifactContractError("metric event split is required", code="metric.split")
    if not event.name:
        raise ArtifactContractError("metric event name is required", code="metric.name")
    if event.value is not None and not math.isfinite(float(event.value)):
        raise ArtifactContractError(
            "metric event value must be finite or explicit null",
            code="metric.value_non_finite",
            context={
                "split": event.split,
                "name": event.name,
                "planned_step_id": event.planned_step_id,
            },
        )
    if event.world_size is not None and event.world_size <= 0:
        raise ArtifactContractError(
            "metric event world_size must be positive when present",
            code="metric.world_size",
            context={"world_size": event.world_size},
        )
    if event.rank is not None and event.world_size is not None:
        if event.rank < 0 or event.rank >= event.world_size:
            raise ArtifactContractError(
                "metric event rank must be inside world size",
                code="metric.rank",
                context={"rank": event.rank, "world_size": event.world_size},
            )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = ["MetricStreamEvent", "metric_stream_events_from_runtime_payload"]
