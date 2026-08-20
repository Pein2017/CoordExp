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

``_resource_scalar_metrics`` is also called directly by
``src.training.session._eval_forward_handler`` (moved to the session owner in
wave 5), which imports this module rather than the facade forwarding it under
its historical name.

Wave 4 of ``add-coordexp-swift-training-observability`` moved the rank-zero
append plus its all-rank status handshake OUT of this module:
``_append_logging_row_shared`` now lives in
``src/artifacts/observation_publisher.py``, which owns JSONL-first publication
and the derived console/TensorBoard lifecycle. No forwarding alias is kept
here - this owner builds canonical rows and hands them to the injected
publisher (or, for the frozen single-owner fixtures, to the publication-only
handshake directly).

Wave 2 of ``add-coordexp-swift-training-observability`` made this owner declare
each observation scalar's reducer through ``src/runtime/metrics.py`` and stopped
serializing a per-rank measurement trace into the canonical row; per-rank
scalars remain available inside the reduction result for bounded lifecycle
resource accounting only.

Wave 3 (tasks 3.3-3.7) made it the canonical TRAIN ROW BUILDER. It now also:

* consumes the finalized loss artifact's configured weight and merged global
  denominator inputs for every actually computed term (never recomputing an
  objective from ``segment_mean_numerator`` or a backend-scaled tensor);
* publishes the runtime-owned ``AppliedUpdateReceipt``'s boundary truth and
  its pre-call applied learning rates, preserving JSON ``null`` where no
  update was applied;
* derives global work rates from SUMMED work over the ALL-RANK MAXIMUM step
  duration;
* carries honest input-timing and CUDA allocator observations, naming what a
  backend could not measure in ``unavailable_fields`` instead of publishing a
  fabricated zero;
* builds and publishes the ONE bounded terminal row for a failed optimizer
  boundary (:func:`publish_terminal_boundary_row`), which is the seam the
  Wave-4 JSONL-first publisher wraps.

Every new field is ADDITIVE and conditional on an input its producer actually
observed, so an observation that reports none of them yields exactly the
pre-Wave-3 row.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from dataclasses import dataclass

from src.artifacts import observation_publisher
from src.artifacts.observation_publisher import ObservationPublisher
from src.artifacts.resources import (
    CUDA_ALLOCATOR_BYTE_FIELDS,
    collect_resource_snapshot,
    cuda_allocator_counter_deltas,
    merge_rank_cpu_resource_receipts,
    rank_cpu_resources_from_metric_rows,
)
from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.runtime.metrics import (
    REDUCER_BOOL_ALL,
    REDUCER_IDENTICAL,
    REDUCER_MAX,
    REDUCER_SUM,
    ScalarSample,
    loss_telemetry_batch,
)
from src.runtime.optimizer_boundary import AppliedUpdateReceipt
from src.training import cache_workflow
from src.training.supervised_trainer import CompletedStepObservation

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


#: Canonical row names for the CUDA allocator observation. The process-lifetime
#: PEAK bytes keep their existing `resource/gpu_max_memory_*` names, which come
#: from the resource snapshot; these are the CURRENT occupancy and the per-step
#: counter deltas.
_CUDA_ALLOCATOR_ROW_FIELDS: dict[str, str] = {
    "current_allocated_bytes": "resource/gpu_current_memory_allocated_bytes",
    "current_reserved_bytes": "resource/gpu_current_memory_reserved_bytes",
    "num_alloc_retries": "resource/gpu_alloc_retries_delta",
    "num_ooms": "resource/gpu_ooms_delta",
}

#: Canonical names of the derived global work-rate fields.
THROUGHPUT_FIELDS: dict[str, str] = {
    "throughput/physical_tokens_per_second": "count/physical_tokens",
    "throughput/supervised_atoms_per_second": "count/supervised_atoms",
    "throughput/packs_per_second": "count/packs",
}


@dataclass(frozen=True)
class _GatedObservation:
    """A group of scalars whose availability is a RANK-LOCAL measurement fact.

    Cross-rank sample schemas must agree, so a metric that one rank can read
    and another cannot would otherwise fail the whole planned step closed.
    These groups instead carry an explicit ``BOOL_ALL`` availability gate: the
    paired scalars ride the collective with a placeholder that is discarded
    unless EVERY rank measured the group, and the row then either publishes
    real reduced values or names the fields in ``unavailable_fields``. The
    placeholder never reaches a durable row, so no field is ever published as
    a fabricated zero.
    """

    gate_name: str
    available: bool
    #: ``(field name, declared reducer, rank-local value, integral)``
    fields: tuple[tuple[str, str, float | None, bool], ...]

    def samples(self) -> list[ScalarSample]:
        samples: list[ScalarSample] = [
            ScalarSample(
                name=self.gate_name,
                reducer=REDUCER_BOOL_ALL,
                value=1.0 if self.available else 0.0,
            )
        ]
        for name, reducer, value, integral in self.fields:
            usable = self.available and value is not None
            samples.append(
                ScalarSample(
                    name=name,
                    reducer=reducer,
                    value=float(value) if usable else 0.0,
                    integral=integral,
                )
            )
        return samples

    def resolve(
        self, reduced_row: dict[str, Any], unavailable: set[str]
    ) -> None:
        gate = reduced_row.pop(self.gate_name, None)
        available = isinstance(gate, (int, float)) and float(gate) == 1.0
        for name, _reducer, _value, _integral in self.fields:
            if available:
                continue
            reduced_row.pop(name, None)
            unavailable.add(name)


def _loss_term_observation_samples(
    terms: Any,
) -> tuple[list[ScalarSample], dict[str, str]]:
    """Configured weight and denominator inputs for actually computed terms.

    These are CONSUMED from the finalized loss artifact, never recomputed: the
    supervised-loss contract owns raw/weighted values and the merged global
    denominator. A term the loss owner omitted (a zero-weight optional
    auxiliary) contributes no field here, and a computed zero-weight protected
    gate keeps its full family.
    """

    samples: list[ScalarSample] = []
    scopes: dict[str, str] = {}
    if not isinstance(terms, Sequence):
        return samples, scopes
    for term in terms:
        if not isinstance(term, Mapping):
            continue
        name = str(term.get("name"))
        weight = term.get("weight")
        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            samples.append(
                ScalarSample(
                    name=f"loss/{name}/weight",
                    reducer=REDUCER_IDENTICAL,
                    value=float(weight),
                )
            )
        denominator = term.get("denominator")
        if not isinstance(denominator, Mapping):
            continue
        scope = denominator.get("denominator_scope")
        if isinstance(scope, str):
            scopes[f"loss/{name}/denominator_scope"] = scope
        for field, published in (
            ("selected_atom_count", f"loss/{name}/selected_atom_count"),
            ("skipped_segment_count", f"loss/{name}/skipped_segment_count"),
        ):
            value = denominator.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                samples.append(
                    ScalarSample(
                        # The denominator is resolved once across ranks in
                        # `prepare_planned_step`, so it is already global.
                        name=published,
                        reducer=REDUCER_IDENTICAL,
                        value=float(value),
                        integral=True,
                    )
                )
    return samples, scopes


def _boundary_truth_fields(receipt: AppliedUpdateReceipt) -> dict[str, Any]:
    """Project the runtime-owned receipt; never synthesize a boundary boolean."""

    return {
        "optimizer_boundary_action": receipt.action,
        "optimizer_update_attempted": receipt.attempted,
        "optimizer_update_applied": receipt.applied,
        "optimizer_step_was_skipped": receipt.step_was_skipped,
        "optimizer_mutation_state": receipt.mutation_state,
    }


def _counter_fields(runtime: Any) -> dict[str, int]:
    fields: dict[str, int] = {}
    for name in ("optimizer_step_count", "scheduler_step_count"):
        value = getattr(runtime, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            fields[name] = int(value)
    return fields


def _throughput_fields(
    reduced_row: Mapping[str, Any], unavailable: set[str]
) -> dict[str, float]:
    """Global work rates: summed work over the ALL-RANK MAXIMUM step duration.

    Never a mean duration (which would flatter the slowest rank that actually
    owns the distributed critical path) and never rank-local work.
    """

    duration = reduced_row.get("step_duration_seconds")
    usable_duration = (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
        and float(duration) > 0.0
    )
    fields: dict[str, float] = {}
    for name, work_key in THROUGHPUT_FIELDS.items():
        work = reduced_row.get(work_key)
        usable_work = (
            isinstance(work, (int, float))
            and not isinstance(work, bool)
            and math.isfinite(float(work))
        )
        if not usable_duration or not usable_work:
            unavailable.add(name)
            continue
        fields[name] = float(work) / float(duration)
    return fields


def _publish_row(
    *,
    publisher: ObservationPublisher | None,
    writer: RunWriter | None,
    row: Mapping[str, Any],
    runtime: Any,
    terminal: bool = False,
) -> None:
    """Publish one canonical row through the JSONL-first publication owner.

    ``publisher`` is the composed rank-zero publisher
    ``src/training/session.py`` builds for a production run; it appends the row
    and only then presents it. ``None`` selects publication WITHOUT any derived
    sink, which is the shape the frozen characterization fixtures and the
    single-owner unit tests construct directly; it is the same bounded
    append/status handshake, so the published bytes are identical either way.
    """

    if publisher is None:
        observation_publisher._append_logging_row_shared(
            writer=writer, row=row, runtime=runtime
        )
        return
    publisher.publish(row, terminal=terminal)


def _finish_first_optimizer_step_phase(
    writer: RunWriter | None,
    lifecycle: MutableMapping[str, Any],
    *,
    status: str = "completed",
    rank_resources: Mapping[str, Any] | None,
) -> None:
    """Duplicate of ``session._finish_run_phase`` bound to one phase name.

    ``_finish_run_phase`` is a general phase-lifecycle utility session.py uses
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
        cuda_allocator_sampler: Callable[[], Mapping[str, Any]] | None = None,
        publisher: ObservationPublisher | None = None,
    ) -> None:
        self._writer = writer
        self._lifecycle = lifecycle
        self._runtime = runtime
        self._resource_collector = resource_collector
        self._cuda_allocator_sampler = cuda_allocator_sampler
        # The JSONL-first publication owner. It is injected, never built here:
        # composition belongs to `src/training/session.py`.
        self._publisher = publisher
        # The allocator retry/OOM counters are PROCESS-LIFETIME counters, so
        # the row publishes deltas against the previous completed observation.
        # There is no prior snapshot for the first observed step, and that is
        # reported as unavailable rather than as a zero delta.
        self._previous_allocator_sample: Mapping[str, Any] | None = None

    def _cuda_allocator_observations(self) -> list[_GatedObservation]:
        """Current allocator bytes plus per-step retry/OOM counter deltas.

        The process-lifetime PEAK bytes are already owned by the resource
        snapshot's ``max_memory_*`` fields; this adds the CURRENT occupancy
        and the deltas. Bytes reduce as the all-rank maximum (a high-water
        observation), counter deltas as the all-rank sum (disjoint per-rank
        events). Nothing here resets a peak statistic.
        """

        sampler = self._cuda_allocator_sampler
        if sampler is None:
            return []
        sample = dict(sampler())
        available = sample.get("available") is True
        byte_fields: list[tuple[str, str, float | None, bool]] = []
        for field in CUDA_ALLOCATOR_BYTE_FIELDS:
            value = sample.get(field)
            byte_fields.append(
                (
                    _CUDA_ALLOCATOR_ROW_FIELDS[field],
                    REDUCER_MAX,
                    float(value)
                    if isinstance(value, int) and not isinstance(value, bool)
                    else None,
                    False,
                )
            )
        deltas = cuda_allocator_counter_deltas(self._previous_allocator_sample, sample)
        delta_fields: list[tuple[str, str, float | None, bool]] = [
            (
                _CUDA_ALLOCATOR_ROW_FIELDS[field],
                REDUCER_SUM,
                None if deltas[field] is None else float(deltas[field]),
                True,
            )
            for field in ("num_alloc_retries", "num_ooms")
        ]
        self._previous_allocator_sample = sample if available else None
        return [
            _GatedObservation(
                gate_name="resource/gpu_allocator_bytes_measured",
                available=available and all(item[2] is not None for item in byte_fields),
                fields=tuple(byte_fields),
            ),
            _GatedObservation(
                gate_name="resource/gpu_allocator_deltas_measured",
                available=available and all(item[2] is not None for item in delta_fields),
                fields=tuple(delta_fields),
            ),
        ]

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
        terms = loss_bundle.get("terms", ())
        observation_samples: list[ScalarSample] = []
        unavailable: set[str] = set()
        # Fields whose JSON `null` is itself the truthful observation (an LR
        # that was not applied), as opposed to a scalar that is simply absent.
        null_preserving: set[str] = set()
        gated: list[_GatedObservation] = []
        receipt = observation.update_receipt
        # Applied learning rates and shared schedule values are one global
        # number every rank must already agree on (task 2.4): a rank-local
        # divergence is a fault, never something to average away.
        if receipt is None:
            for name, value in _scheduler_lr_metrics(
                observation.scheduler_artifact
            ).items():
                observation_samples.append(
                    ScalarSample(name=name, reducer=REDUCER_IDENTICAL, value=value)
                )
        else:
            # The runtime sampled these immediately BEFORE the wrapper call and
            # publishes them only where every rank is known to have applied
            # them; a scheduled or merely attempted value is JSON `null` here.
            for index, applied in enumerate(receipt.group_learning_rates):
                name = f"lr/group_{index}"
                observation_samples.append(
                    ScalarSample(
                        name=name,
                        reducer=REDUCER_IDENTICAL,
                        value=None if applied is None else float(applied),
                        required=applied is not None,
                    )
                )
                if applied is None:
                    unavailable.add(name)
                    null_preserving.add(name)
        # The slowest rank owns the distributed critical path, so every timing
        # and high-water resource scalar reduces as the all-rank maximum.
        timing_fields = {
            "step_duration_seconds": observation.step_duration_seconds,
            "input_build_seconds": observation.input_build_seconds,
            "input_wait_seconds": observation.input_wait_seconds,
        }
        for name, value in timing_fields.items():
            if value is None:
                continue
            observation_samples.append(
                ScalarSample(name=name, reducer=REDUCER_MAX, value=float(value))
            )
        # Host-to-device completion is a rank-local measurement fact, so it
        # rides an availability gate rather than failing the step closed on a
        # rank whose transfer events were not complete.
        if (
            observation.input_h2d_seconds is not None
            or observation.input_h2d_unavailable_reason is not None
        ):
            gated.append(
                _GatedObservation(
                    gate_name="input_h2d_measured",
                    available=observation.input_h2d_seconds is not None,
                    fields=(
                        (
                            "input_h2d_seconds",
                            REDUCER_MAX,
                            observation.input_h2d_seconds,
                            False,
                        ),
                    ),
                )
            )
        # Exact per-step packed work, captured before the step's tensors were
        # released, summed across the ranks that share the planned step.
        if observation.physical_token_count is not None or terms:
            gated.append(
                _GatedObservation(
                    gate_name="count/physical_tokens_measured",
                    available=observation.physical_token_count is not None,
                    fields=(
                        (
                            "count/physical_tokens",
                            REDUCER_SUM,
                            None
                            if observation.physical_token_count is None
                            else float(observation.physical_token_count),
                            True,
                        ),
                    ),
                )
            )
        # Already the all-rank maximum of rank-local finite pre-clip norms when
        # the gradient gate produced one: reducing it again as IDENTICAL is
        # value-preserving and fails closed on a divergence instead of hiding
        # it under a second maximum.
        if observation.pre_clip_grad_norm_rank_max is not None:
            observation_samples.append(
                ScalarSample(
                    name="grad_norm/pre_clip_rank_max",
                    reducer=REDUCER_IDENTICAL,
                    value=float(observation.pre_clip_grad_norm_rank_max),
                )
            )
        elif receipt is not None:
            unavailable.add("grad_norm/pre_clip_rank_max")
        term_samples, term_scope_fields = _loss_term_observation_samples(terms)
        observation_samples.extend(term_samples)
        resource_snapshot = (
            None if resource_collector is None else dict(resource_collector())
        )
        if resource_snapshot is not None:
            for name, value in _resource_scalar_metrics(resource_snapshot).items():
                observation_samples.append(
                    ScalarSample(name=name, reducer=REDUCER_MAX, value=float(value))
                )
        gated.extend(self._cuda_allocator_observations())
        for group in gated:
            observation_samples.extend(group.samples())
        gathered = runtime.gather_metrics(
            loss_telemetry_batch(
                planned_step_id=observation.planned_step_id,
                split=cache_workflow.TRAIN_SPLIT,
                loss_metrics=metrics if isinstance(metrics, Mapping) else {},
                loss_artifact=loss_bundle,
                # Every rank owns a disjoint slice of the planned step, so its
                # objective telemetry and pack/example counts are partial
                # contributions to one global value.
                partial_rank_contributions=True,
                accuracy_stats=accuracy_stats
                if isinstance(accuracy_stats, Mapping)
                else None,
                extra_samples=tuple(observation_samples),
            )
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
        reduced_row = dict(reduced)
        for group in gated:
            group.resolve(reduced_row, unavailable)
        throughput = (
            _throughput_fields(reduced_row, unavailable) if terms else {}
        )
        for name, value in list(reduced_row.items()):
            if value is not None:
                continue
            # A reduced `None` is an all-rank agreement that the scalar is not
            # available. It is named here rather than emitted as a zero; only
            # the null-preserving families keep an explicit JSON `null`.
            unavailable.add(name)
            if name not in null_preserving:
                reduced_row.pop(name)
        row: dict[str, Any] = {
            "step": observation.planned_step_id,
            "split": cache_workflow.TRAIN_SPLIT,
            "micro_step_count": observation.micro_step_count,
            "optimizer_update_status": observation.optimizer_update_status,
            "finite_status": observation.finite_status,
            **reduced_row,
            **term_scope_fields,
            **throughput,
        }
        if receipt is not None:
            row.update(_boundary_truth_fields(receipt))
        row.update(_counter_fields(runtime))
        if isinstance(reduced_accuracy_stats, Mapping):
            row["accuracy_stats"] = dict(reduced_accuracy_stats)
        if unavailable:
            row["unavailable_fields"] = sorted(unavailable)
        _publish_row(
            publisher=self._publisher, writer=writer, row=row, runtime=runtime
        )
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


def build_terminal_boundary_row(
    receipt: AppliedUpdateReceipt,
    *,
    runtime: Any,
) -> dict[str, Any]:
    """Build the ONE bounded terminal row for a failed optimizer boundary.

    It carries the runtime-owned receipt's truthful (and deliberately nullable)
    fields, its bounded terminal reason, the composite mutation state, and the
    CURRENT optimizer/scheduler counters -- which a terminal boundary does not
    advance. No boolean here is inferred from an exception, and no metric
    collective runs: a divergent boundary must not enter another gather.
    """

    if not isinstance(receipt, AppliedUpdateReceipt) or not receipt.terminal:
        raise RuntimeContractError(
            "a terminal boundary row requires the runtime-owned terminal receipt",
            code="runtime.terminal_row_receipt_invalid",
        )
    unavailable: set[str] = set()
    row: dict[str, Any] = {
        "step": receipt.planned_step_id,
        "split": cache_workflow.TRAIN_SPLIT,
        "optimizer_update_status": receipt.optimizer_update_status,
        # The terminal receipt carries no finite observation of its own and
        # the gate decision is gone by this point, so the only truthful value
        # is an explicit unavailability.
        "finite_status": "unavailable",
        "optimizer_boundary_terminal": True,
        "optimizer_terminal_reason": receipt.terminal_reason,
        **_boundary_truth_fields(receipt),
    }
    for index, applied in enumerate(receipt.group_learning_rates):
        name = f"lr/group_{index}"
        row[name] = None if applied is None else float(applied)
        if applied is None:
            unavailable.add(name)
    row.update(_counter_fields(runtime))
    if unavailable:
        row["unavailable_fields"] = sorted(unavailable)
    return row


def publish_terminal_boundary_row(
    *,
    writer: RunWriter | None,
    runtime: Any,
    lifecycle: Mapping[str, Any],
    terminal: Any,
    publisher: ObservationPublisher | None = None,
) -> None:
    """Publish/converge exactly one terminal row before failed finalization.

    This is the reporting-owned seam the JSONL-first publisher wraps later; it
    performs the existing rank-zero append plus its all-rank outcome
    broadcast, and nothing else. It deliberately does NOT touch the lifecycle
    counters: a terminal boundary is not a completed planned step and must not
    advance completed-step, scheduler, eval, checkpoint, exact-resume,
    selector, or final-success work. It also does not READ a lifecycle status:
    `lifecycle["finite_status"]` belongs to the last COMPLETED step, and
    republishing it under this planned step id would label a stale
    observation as this boundary's own truth.

    Publication is best effort BY CONTRACT: every rank observes the same
    broadcast outcome, so a failed append leaves every rank converging the
    same failed finalization while the caller keeps raising the primary
    optimizer-boundary failure ahead of the publication failure.
    """

    del lifecycle  # read for nothing on purpose; see the docstring above.
    receipt = getattr(terminal, "receipt", None)
    if not isinstance(receipt, AppliedUpdateReceipt) or not receipt.terminal:
        return
    row = build_terminal_boundary_row(receipt, runtime=runtime)
    try:
        _publish_row(
            publisher=publisher,
            writer=writer,
            row=row,
            runtime=runtime,
            terminal=True,
        )
    except BaseException:
        return
