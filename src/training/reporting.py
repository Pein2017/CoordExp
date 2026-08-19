"""Existing per-completed-step logging callback, moved verbatim to its own owner.

``CompletedStepReporter`` replaces ``src.training.pipeline._train_logging_handler``
(design decision 9): the same lifecycle-counter mutation, loss/scheduler/timing/
resource scalar extraction, runtime gather, accuracy-stat validation, rank-resource
projection, current row keys, rank-zero append and outcome broadcast,
warmup-qualified measurement accumulation, and first-step phase completion. This
owner exposes no configurable sinks, cadence, ETA, TensorBoard integration, or
metric registry -- ``add-coordexp-swift-training-observability`` may extend this
seam later, but this change does not pre-build that feature or change an artifact
byte.

``_append_logging_row_shared``, ``_resource_scalar_metrics``, and
``_per_rank_measurement`` are also called directly by
``src.training.pipeline._eval_forward_handler`` (still pipeline-owned until wave
5), which imports this module rather than pipeline.py forwarding them under their
historical names.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from src.artifacts.resources import (
    collect_resource_snapshot,
    merge_rank_cpu_resource_receipts,
    rank_cpu_resources_from_metric_rows,
)
from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.training import cache_workflow
from src.training.supervised_trainer import CompletedStepObservation

try:
    from accelerate.utils import broadcast_object_list
except ImportError:  # pragma: no cover - exercised only in stripped environments.
    broadcast_object_list = None  # type: ignore[assignment]


def _scheduler_lr_metrics(scheduler_artifact: Any) -> dict[str, float]:
    if not isinstance(scheduler_artifact, Mapping):
        return {}
    learning_rates = scheduler_artifact.get("learning_rates")
    if not isinstance(learning_rates, Sequence):
        return {}
    metrics: dict[str, float] = {}
    for item in learning_rates:
        if not isinstance(item, Mapping):
            continue
        group_index = item.get("group_index")
        lr = item.get("lr")
        if group_index is None or lr is None:
            continue
        metrics[f"lr/group_{int(group_index)}"] = float(lr)
    return metrics


def _resource_scalar_metrics(snapshot: Mapping[str, Any]) -> dict[str, float]:
    cpu = snapshot.get("cpu")
    gpu = snapshot.get("gpu")
    metrics: dict[str, float] = {}
    if isinstance(cpu, Mapping):
        for field in ("max_rss_bytes", "io_read_bytes", "io_write_bytes"):
            value = cpu.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                metrics[f"resource/cpu_{field}"] = float(value)
    if isinstance(gpu, Mapping) and gpu.get("initialized") is True:
        for field in (
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
        ):
            value = gpu.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                metrics[f"resource/gpu_{field}"] = float(value)
    return metrics


def _per_rank_measurement(
    gathered: Mapping[str, Any],
) -> dict[str, dict[str, float]] | None:
    per_rank = gathered.get("per_rank_metrics")
    if not isinstance(per_rank, Mapping):
        return None
    prefixes = (
        "eval_duration_seconds",
        "input_build_seconds",
        "input_wait_seconds",
        "resource/",
        "step_duration_seconds",
    )
    receipt: dict[str, dict[str, float]] = {}
    for rank, metrics in per_rank.items():
        if not isinstance(metrics, Mapping):
            continue
        selected = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and any(
                str(key) == prefix or str(key).startswith(prefix) for prefix in prefixes
            )
        }
        if selected:
            receipt[str(rank)] = {key: selected[key] for key in sorted(selected)}
    return receipt or None


def _append_logging_row_shared(
    *, writer: RunWriter | None, row: Mapping[str, Any], runtime: Any
) -> None:
    """Append on rank zero and make its bounded outcome common to every rank."""
    accelerator = getattr(runtime, "accelerator", runtime)
    is_main = bool(
        getattr(
            runtime, "is_main_process", getattr(accelerator, "is_main_process", True)
        )
    )
    status: dict[str, Any] = {"ok": True}
    if is_main:
        try:
            if writer is None:
                raise RuntimeError("rank zero has no run writer")
            writer.append_logging_row(row)
        except BaseException as exc:
            status = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:1024],
            }
    values: list[Any] = [status]
    if (
        int(getattr(runtime, "world_size", getattr(accelerator, "num_processes", 1)))
        > 1
    ):
        broadcast = getattr(accelerator, "broadcast_object_list", None)
        if callable(broadcast):
            result = broadcast(values, from_process=0)
            if result is not None:
                values = result
        elif broadcast_object_list is not None:
            broadcast_object_list(values, from_process=0)
        else:
            raise RuntimeContractError(
                "logging outcome broadcast requires accelerate",
                code="runtime.logging_broadcast_unavailable",
            )
    shared = values[0]
    if not isinstance(shared, Mapping) or not bool(shared.get("ok")):
        error = (
            shared.get("error", "invalid status")
            if isinstance(shared, Mapping)
            else shared
        )
        raise RuntimeContractError(
            f"rank zero logging append failed: {error}",
            code="runtime.logging_append_failed",
        )


def _finish_first_optimizer_step_phase(
    writer: RunWriter | None,
    lifecycle: MutableMapping[str, Any],
    *,
    status: str = "completed",
    rank_resources: Mapping[str, Any] | None,
) -> None:
    """Duplicate of ``pipeline._finish_run_phase`` bound to one phase name.

    ``_finish_run_phase`` is a general phase-lifecycle utility pipeline.py uses
    across many unrelated phases and keeps for the eventual session owner; this
    reporter cannot import it back (``src.training.pipeline`` is a forbidden
    reverse edge from a leaf/domain owner). The "first_optimizer_step" call
    shape is reproduced verbatim rather than generalized, so the observable
    behavior -- including error codes -- stays exactly the historical one.
    """

    phase = "first_optimizer_step"
    if lifecycle.get("active_phase") != phase:
        raise RuntimeContractError(
            "training phase lifecycle can finish only its active phase",
            code="runtime.phase_not_active",
            context={
                "active_phase": lifecycle.get("active_phase"),
                "requested_phase": phase,
            },
        )
    started = lifecycle.get("phase_started_monotonic")
    if not isinstance(started, (int, float)):
        raise RuntimeContractError(
            "training phase lifecycle has no monotonic start",
            code="runtime.phase_start_missing",
            context={"phase": phase},
        )
    duration = max(0.0, float(time.monotonic()) - float(started))
    captured = lifecycle.get("phase_rank_receipts", {}).get(phase)
    rank_details: Mapping[str, Any] | None = None
    if isinstance(captured, Mapping):
        if rank_resources is None and isinstance(captured.get("rank_resources"), Mapping):
            rank_resources = captured["rank_resources"]
        if isinstance(captured.get("rank_details"), Mapping):
            rank_details = captured["rank_details"]
    if writer is not None:
        writer.finish_phase(
            phase,
            status=status,
            completed_at=cache_workflow._utc_now(),
            duration_seconds=duration,
            resources=collect_resource_snapshot(),
            accepted_measured_steps=None,
            expected_measured_steps=None,
            rank_resources=rank_resources,
            rank_details=rank_details,
        )
    lifecycle["active_phase"] = None
    lifecycle["phase_started_monotonic"] = None


class CompletedStepReporter:
    """Existing completed-step logging callback, owned here per design decision 9."""

    def __init__(
        self,
        *,
        writer: RunWriter | None,
        lifecycle: MutableMapping[str, Any],
        runtime: Any,
        resource_collector: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._writer = writer
        self._lifecycle = lifecycle
        self._runtime = runtime
        self._resource_collector = resource_collector

    def __call__(self, observation: CompletedStepObservation) -> None:
        writer = self._writer
        lifecycle = self._lifecycle
        runtime = self._runtime
        resource_collector = self._resource_collector

        lifecycle.update(
            completed_steps=observation.planned_step_id,
            consumed_packs=int(lifecycle.get("consumed_packs", 0))
            + observation.micro_step_count,
            optimizer_update_status=observation.optimizer_update_status,
            finite_status=observation.finite_status,
        )
        loss_bundle = dict(observation.loss_bundle_artifact)
        metrics = loss_bundle.get("metrics", {})
        accuracy_stats = loss_bundle.get("accuracy_stats")
        scalar_metrics = _scheduler_lr_metrics(observation.scheduler_artifact)
        if isinstance(metrics, Mapping):
            scalar_metrics.update({str(name): value for name, value in metrics.items()})
        timing_fields = {
            "step_duration_seconds": observation.step_duration_seconds,
            "input_build_seconds": observation.input_build_seconds,
            "input_wait_seconds": observation.input_wait_seconds,
        }
        scalar_metrics.update(
            {
                name: float(value)
                for name, value in timing_fields.items()
                if value is not None
            }
        )
        resource_snapshot = (
            None if resource_collector is None else dict(resource_collector())
        )
        if resource_snapshot is not None:
            scalar_metrics.update(_resource_scalar_metrics(resource_snapshot))
        gathered = runtime.gather_metrics(
            scalar_metrics,
            planned_step_id=observation.planned_step_id,
            split=cache_workflow.TRAIN_SPLIT,
            accuracy_stats=accuracy_stats
            if isinstance(accuracy_stats, Mapping)
            else None,
        )
        reduced = gathered.get("metrics") if isinstance(gathered, Mapping) else None
        if not isinstance(reduced, Mapping):
            raise RuntimeContractError(
                "train metric reduction returned no scalar mapping",
                code="runtime.train_metric_reduction_failed",
            )
        reduced_accuracy_stats = (
            gathered.get("accuracy_stats") if isinstance(gathered, Mapping) else None
        )
        if any(key in reduced for key in ("acc_top1", "acc_top5")) and not isinstance(
            reduced_accuracy_stats, Mapping
        ):
            raise RuntimeContractError(
                "train metric reduction returned no global integer accuracy_stats",
                code="runtime.train_accuracy_stats_reduction_failed",
            )
        world_size = int(
            getattr(
                runtime,
                "world_size",
                getattr(getattr(runtime, "accelerator", runtime), "num_processes", 1),
            )
        )
        rank_resources = rank_cpu_resources_from_metric_rows(
            gathered.get("per_rank_metrics") if isinstance(gathered, Mapping) else None,
            world_size=world_size,
        )
        row: dict[str, Any] = {
            "step": observation.planned_step_id,
            "split": cache_workflow.TRAIN_SPLIT,
            "micro_step_count": observation.micro_step_count,
            "optimizer_update_status": observation.optimizer_update_status,
            "finite_status": observation.finite_status,
            **dict(reduced),
        }
        if isinstance(reduced_accuracy_stats, Mapping):
            row["accuracy_stats"] = dict(reduced_accuracy_stats)
        per_rank_measurement = _per_rank_measurement(gathered)
        if per_rank_measurement is not None:
            row["per_rank_measurement"] = per_rank_measurement
        _append_logging_row_shared(writer=writer, row=row, runtime=runtime)
        warmup_steps = lifecycle.get("measurement_warmup_steps")
        reduced_step_duration = reduced.get("step_duration_seconds")
        if (
            isinstance(warmup_steps, int)
            and observation.planned_step_id > warmup_steps
            and observation.optimizer_update_status == "applied"
            and observation.finite_status == "finite"
            and isinstance(reduced_step_duration, (int, float))
            and not isinstance(reduced_step_duration, bool)
            and math.isfinite(float(reduced_step_duration))
            and float(reduced_step_duration) >= 0.0
        ):
            lifecycle["accepted_measured_steps"] = (
                int(lifecycle.get("accepted_measured_steps", 0)) + 1
            )
            lifecycle["steady_state_duration_seconds"] = float(
                lifecycle.get("steady_state_duration_seconds", 0.0)
            ) + float(reduced_step_duration)
            lifecycle["steady_state_rank_resources"] = merge_rank_cpu_resource_receipts(
                lifecycle.get("steady_state_rank_resources")
                if isinstance(lifecycle.get("steady_state_rank_resources"), Mapping)
                else None,
                rank_resources,
            )
        active_phase = lifecycle.get("active_phase")
        if (
            active_phase == "first_optimizer_step"
            and observation.optimizer_update_status == "applied"
        ):
            _finish_first_optimizer_step_phase(
                writer,
                lifecycle,
                rank_resources=rank_resources,
            )

        resolved_max_steps = lifecycle.get("resolved_max_steps")
        is_final_step = (
            isinstance(resolved_max_steps, int)
            and observation.planned_step_id == resolved_max_steps
        )
        if is_final_step and lifecycle.get("active_phase") == "first_optimizer_step":
            _finish_first_optimizer_step_phase(
                writer,
                lifecycle,
                status="failed",
                rank_resources=rank_resources,
            )
