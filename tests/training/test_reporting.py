"""Wave-4 exact-compare tests for ``src.training.reporting``.

``CompletedStepReporter`` is the sole owner of the historical
``pipeline._train_logging_handler`` behavior (design decision 9): lifecycle
mutation, loss/LR/timing/resource extraction, reduction requests, accuracy
validation, rank-resource accounting, every-completed-step row bytes, append
outcome broadcast, warmup accounting, and first-step phase behavior. These
tests exact-compare that behavior against the same baseline shapes the
pre-move pipeline handler used (see ``tests/training/test_pipeline_assembly.py``
and the frozen ``tests/fixtures/training_orchestration/completed_step_rows.json``
fixture, both of which now construct ``reporting.CompletedStepReporter``
directly rather than the deleted pipeline factory function).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.artifacts import observation_publisher
from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.runtime.metrics import reduce_rank_payloads
from src.training import reporting
from src.training.supervised_trainer import CompletedStepObservation


class _Accelerator:
    is_main_process = True
    num_processes = 1


class _Runtime:
    """DECLARED FLIP (add-coordexp-swift-training-observability, Wave 2, task
    2.3): the reporter hands the runtime a typed `MetricBatch`, so this double
    runs the real world-size-one reduction instead of echoing a mapping."""

    is_main_process = True
    world_size = 1
    accelerator = _Accelerator()

    def gather_metrics(self, batch: object) -> object:
        reduced = reduce_rank_payloads(
            [batch.to_rank_payload(rank=0, world_size=1)],  # type: ignore[attr-defined]
            world_size=1,
        )
        result: dict[str, object] = {"metrics": dict(reduced.metrics)}
        if reduced.accuracy_stats is not None:
            result["accuracy_stats"] = dict(reduced.accuracy_stats)
        return result


def _writer(tmp_path: Path) -> RunWriter:
    return RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
        resolved_max_steps=5,
    )


def _observation(step: int, **overrides: object) -> CompletedStepObservation:
    defaults: dict[str, object] = dict(
        planned_step_id=step,
        micro_step_count=2,
        loss_bundle_artifact={
            "metrics": {"loss/total": 1.0, "acc_top1": 0.5, "acc_top5": 1.0},
            "accuracy_stats": {
                "top1_correct": 1,
                "top5_correct": 2,
                "atom_count": 2,
            },
        },
        optimizer_update_status="applied",
        finite_status="finite",
        scheduler_artifact={"learning_rates": [{"group_index": 0, "lr": 1e-5}]},
    )
    defaults.update(overrides)
    return CompletedStepObservation(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shape: no cadence/sinks/fields/ETA/TensorBoard/metric-registry surface.
# ---------------------------------------------------------------------------


def test_reporter_is_keyword_only_constructed_and_callable(tmp_path: Path) -> None:
    """DECLARED FLIP (add-coordexp-swift-training-observability, Wave 3, task
    3.6).

    Old assertion: the constructor took exactly
    ``writer/lifecycle/runtime/resource_collector``.

    New assertion: it also takes the optional ``cuda_allocator_sampler``.
    The per-step allocator observation needs a retained prior snapshot to
    derive honest retry/OOM deltas, so its reader is injected here exactly
    like ``resource_collector`` rather than being reached for implicitly. The
    surface stays keyword-only, default-``None``, and adds no sink, cadence,
    ETA, TensorBoard, or metric-registry parameter.

    DECLARED FLIP (add-coordexp-swift-training-observability, Wave 4, task
    4.2): it also takes the optional ``publisher``. That is the JSONL-first
    publication OWNER (``src/artifacts/observation_publisher.py``), injected
    by ``src/training/session.py``, which is where cadence, console, and
    TensorBoard live. This owner still exposes no sink list, no cadence
    parameter, no ETA, and no metric registry: it builds the canonical row and
    hands it over. ``None`` keeps the identical publication-only handshake the
    frozen characterization fixtures depend on.
    """

    import inspect

    signature = inspect.signature(reporting.CompletedStepReporter.__init__)
    parameters = signature.parameters
    assert set(parameters) == {
        "self",
        "writer",
        "lifecycle",
        "runtime",
        "resource_collector",
        "cuda_allocator_sampler",
        "publisher",
    }
    for name in (
        "writer",
        "lifecycle",
        "runtime",
        "resource_collector",
        "cuda_allocator_sampler",
        "publisher",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["resource_collector"].default is None
    assert parameters["cuda_allocator_sampler"].default is None
    assert parameters["publisher"].default is None

    call_signature = inspect.signature(reporting.CompletedStepReporter.__call__)
    assert list(call_signature.parameters) == ["self", "observation"]


def test_reporter_exposes_no_configurable_sinks_or_metric_registry() -> None:
    public_class_attributes = {
        name for name in dir(reporting.CompletedStepReporter) if not name.startswith("_")
    }
    assert public_class_attributes == set()
    forbidden_surface = {
        "sink",
        "sinks",
        "cadence",
        "eta",
        "tensorboard",
        "metric_registry",
        "registry",
    }
    module_names = {name.lower() for name in dir(reporting)}
    assert forbidden_surface.isdisjoint(module_names)


# ---------------------------------------------------------------------------
# Lifecycle mutation
# ---------------------------------------------------------------------------


def test_reporter_mutates_lifecycle_counters_exactly(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {"consumed_packs": 3}
    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle=lifecycle, runtime=_Runtime()
    )

    reporter(_observation(1))

    assert lifecycle["completed_steps"] == 1
    assert lifecycle["consumed_packs"] == 5
    assert lifecycle["optimizer_update_status"] == "applied"
    assert lifecycle["finite_status"] == "finite"


# ---------------------------------------------------------------------------
# Loss / LR / timing / resource scalar extraction (pure helpers)
# ---------------------------------------------------------------------------


def test_scheduler_lr_metrics_extracts_named_group_rates() -> None:
    metrics = reporting._scheduler_lr_metrics(
        {"learning_rates": [{"group_index": 0, "lr": 1e-4}, {"group_index": 1, "lr": 2e-4}]}
    )
    assert metrics == {"lr/group_0": 1e-4, "lr/group_1": 2e-4}


def test_scheduler_lr_metrics_tolerates_missing_or_malformed_artifact() -> None:
    assert reporting._scheduler_lr_metrics(None) == {}
    assert reporting._scheduler_lr_metrics({"learning_rates": "not-a-sequence"}) == {}
    assert reporting._scheduler_lr_metrics(
        {"learning_rates": [{"group_index": None, "lr": 1.0}]}
    ) == {}


def test_resource_scalar_metrics_selects_bounded_cpu_and_gpu_fields() -> None:
    metrics = reporting._resource_scalar_metrics(
        {
            "cpu": {"max_rss_bytes": 100, "io_read_bytes": 10, "io_write_bytes": 5},
            "gpu": {
                "initialized": True,
                "max_memory_allocated_bytes": 200,
                "max_memory_reserved_bytes": 300,
            },
        }
    )
    assert metrics == {
        "resource/cpu_max_rss_bytes": 100.0,
        "resource/cpu_io_read_bytes": 10.0,
        "resource/cpu_io_write_bytes": 5.0,
        "resource/gpu_max_memory_allocated_bytes": 200.0,
        "resource/gpu_max_memory_reserved_bytes": 300.0,
    }


def test_resource_scalar_metrics_omits_uninitialized_gpu() -> None:
    metrics = reporting._resource_scalar_metrics(
        {"cpu": {"max_rss_bytes": 1}, "gpu": {"initialized": False}}
    )
    assert metrics == {"resource/cpu_max_rss_bytes": 1.0}


def test_reporter_train_row_carries_resource_scalars_when_collector_present(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporter = reporting.CompletedStepReporter(
        writer=writer,
        lifecycle={},
        runtime=_Runtime(),
        resource_collector=lambda: {
            "cpu": {"max_rss_bytes": 1024},
            "gpu": {"initialized": False},
        },
    )

    reporter(_observation(1))

    row = json.loads(writer.logging_path.read_text())
    assert row["resource/cpu_max_rss_bytes"] == 1024.0


def test_reporter_tolerates_absent_resource_collector(tmp_path: Path) -> None:
    """The historical factory default (``resource_collector=None``) must survive
    the move verbatim -- this is what every characterization call site relies on.
    """

    writer = _writer(tmp_path)
    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )

    reporter(_observation(1))

    row = json.loads(writer.logging_path.read_text())
    assert not any(key.startswith("resource/") for key in row)


# ---------------------------------------------------------------------------
# Reduction requests / accuracy validation
# ---------------------------------------------------------------------------


def test_reporter_requests_split_and_accuracy_stats_from_runtime_gather(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    calls: list[dict[str, object]] = []

    class Runtime(_Runtime):
        def gather_metrics(self, batch: object) -> object:
            calls.append(batch)
            return super().gather_metrics(batch)

    reporting.CompletedStepReporter(writer=writer, lifecycle={}, runtime=Runtime())(
        _observation(1)
    )

    batch = calls[0]
    assert batch.split == "train"
    assert batch.planned_step_id == 1
    assert batch.accuracy.top1_correct == 1
    assert batch.accuracy.top5_correct == 2
    assert batch.accuracy.atom_count == 2
    # The producer declares one exact reducer per sample; the reduction
    # boundary never infers one from a key name.
    declared = {sample.name: getattr(sample, "reducer", "RATIO") for sample in batch.samples}
    assert declared == {
        "loss/total": "SUM",
        "lr/group_0": "IDENTICAL",
    }


def test_reporter_rejects_reduction_without_scalar_mapping(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    class Runtime(_Runtime):
        def gather_metrics(self, batch: object) -> object:
            return {"metrics": None}

    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=Runtime()
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        reporter(_observation(1))
    assert exc_info.value.code == "runtime.train_metric_reduction_failed"


def test_reporter_rejects_accuracy_keys_without_global_accuracy_stats(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    class Runtime(_Runtime):
        def gather_metrics(self, batch: object) -> object:
            reduced = super().gather_metrics(batch)
            return {"metrics": reduced["metrics"], "accuracy_stats": None}  # type: ignore[index]

    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=Runtime()
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        reporter(_observation(1))
    assert exc_info.value.code == "runtime.train_accuracy_stats_reduction_failed"


# ---------------------------------------------------------------------------
# Rank-resource accounting
# ---------------------------------------------------------------------------


def test_normal_train_row_carries_no_per_rank_measurement_trace(
    tmp_path: Path,
) -> None:
    """DECLARED FLIP (add-coordexp-swift-training-observability, Wave 2, tasks
    2.3/2.6).

    Old assertions: `reporting._per_rank_measurement(...)` projected selected
    per-rank timing/resource scalars and the reporter serialized them into the
    canonical row under `per_rank_measurement`.

    New assertion: normal production rows carry aggregated values only. The
    per-rank scalars stay ephemeral inside the reduction result, where bounded
    lifecycle resource accounting still reads them (see
    `rank_cpu_resources_from_metric_rows`).
    """

    assert not hasattr(reporting, "_per_rank_measurement")

    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer,
        lifecycle={},
        runtime=_Runtime(),
        resource_collector=lambda: {
            "cpu": {"max_rss_bytes": 1024},
            "gpu": {"initialized": False},
        },
    )(_observation(1))

    row = json.loads(writer.logging_path.read_text())
    assert "per_rank_measurement" not in row
    assert row["resource/cpu_max_rss_bytes"] == 1024.0


# ---------------------------------------------------------------------------
# Every-completed-step row bytes (frozen fixture)
# ---------------------------------------------------------------------------


def test_reporter_train_row_key_set_matches_the_frozen_fixture(tmp_path: Path) -> None:
    frozen = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "training_orchestration"
            / "completed_step_rows.json"
        ).read_text(encoding="utf-8")
    )
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {"consumed_packs": 0}
    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle=lifecycle, runtime=_Runtime()
    )

    reporter(_observation(1))

    row = json.loads(writer.logging_path.read_text())
    assert set(row).issubset(set(frozen["rows"][0]))
    assert row["split"] == "train"
    assert row["non_finite_fields"] == []


# ---------------------------------------------------------------------------
# Append outcome broadcast
# ---------------------------------------------------------------------------


def test_append_logging_row_shared_broadcasts_rank_zero_failure() -> None:
    """DECLARED FLIP (add-coordexp-swift-training-observability, Wave 4).

    Old assertion: ``reporting`` owned the rank-zero append and its all-rank
    status handshake.

    New assertion: ``src/artifacts/observation_publisher.py`` owns JSONL-first
    publication, so this node calls the moved owner under its historical name.
    ``reporting`` keeps no forwarding alias; its behavior, including both
    bounded error codes, is unchanged.
    """

    shared: dict[str, object] = {}

    class Collective:
        is_main_process = True
        num_processes = 2

        def broadcast_object_list(
            self, values: list[object], from_process: int = 0
        ) -> None:
            if self.is_main_process:
                shared["status"] = values[0]
            else:
                values[0] = shared["status"]

    accelerator = Collective()
    main_runtime = SimpleNamespace(
        accelerator=accelerator, is_main_process=True, world_size=2
    )
    peer_accelerator = Collective()
    peer_accelerator.is_main_process = False
    peer_runtime = SimpleNamespace(
        accelerator=peer_accelerator, is_main_process=False, world_size=2
    )
    failing_writer = SimpleNamespace(
        append_logging_row=lambda row: (_ for _ in ()).throw(OSError("disk full"))
    )

    for runtime, writer in ((main_runtime, failing_writer), (peer_runtime, None)):
        with pytest.raises(RuntimeContractError) as exc_info:
            observation_publisher._append_logging_row_shared(
                writer=writer, row={"step": 1, "split": "train"}, runtime=runtime
            )
        assert exc_info.value.code == "runtime.logging_append_failed"
        assert "OSError: disk full" in str(exc_info.value)


def test_append_logging_row_shared_appends_successfully_on_rank_zero(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    observation_publisher._append_logging_row_shared(
        writer=writer, row={"step": 1, "split": "train"}, runtime=_Runtime()
    )
    row = json.loads(writer.logging_path.read_text())
    assert row["step"] == 1
    assert row["split"] == "train"


# ---------------------------------------------------------------------------
# Warmup-qualified measurement accumulation
# ---------------------------------------------------------------------------


def test_reporter_accumulates_steady_state_only_past_warmup_when_applied_and_finite(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {
        "measurement_warmup_steps": 1,
        "accepted_measured_steps": 0,
        "steady_state_duration_seconds": 0.0,
    }
    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle=lifecycle, runtime=_Runtime()
    )

    reporter(_observation(1, step_duration_seconds=0.3))  # at warmup boundary: excluded
    assert lifecycle["accepted_measured_steps"] == 0

    reporter(_observation(2, step_duration_seconds=0.3))  # past warmup: included
    assert lifecycle["accepted_measured_steps"] == 1
    assert lifecycle["steady_state_duration_seconds"] == pytest.approx(0.3)

    reporter(
        _observation(3, step_duration_seconds=0.3, optimizer_update_status="skipped")
    )  # not applied: excluded
    assert lifecycle["accepted_measured_steps"] == 1


# ---------------------------------------------------------------------------
# First-step phase behavior
# ---------------------------------------------------------------------------


def test_reporter_finishes_first_optimizer_step_phase_on_applied_update(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.begin_phase(
        "first_optimizer_step",
        started_at="now",
        resources=reporting.collect_resource_snapshot(),
    )
    lifecycle: dict[str, object] = {
        "active_phase": "first_optimizer_step",
        "phase_started_monotonic": 0.0,
    }
    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle=lifecycle, runtime=_Runtime()
    )

    reporter(_observation(1))

    assert lifecycle["active_phase"] is None
    assert lifecycle["phase_started_monotonic"] is None
    state = writer.read_run()
    phase = state["measurement"]["phases"]["first_optimizer_step"]
    assert phase["status"] == "completed"


def test_reporter_fails_first_optimizer_step_phase_on_final_step_without_update(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.begin_phase(
        "first_optimizer_step",
        started_at="now",
        resources=reporting.collect_resource_snapshot(),
    )
    lifecycle: dict[str, object] = {
        "active_phase": "first_optimizer_step",
        "phase_started_monotonic": 0.0,
        "resolved_max_steps": 1,
    }
    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle=lifecycle, runtime=_Runtime()
    )

    reporter(_observation(1, optimizer_update_status="skipped"))

    assert lifecycle["active_phase"] is None
    state = writer.read_run()
    phase = state["measurement"]["phases"]["first_optimizer_step"]
    assert phase["status"] == "failed"


def test_reporter_does_not_touch_phase_lifecycle_when_no_phase_is_active(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {}
    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle=lifecycle, runtime=_Runtime()
    )

    reporter(_observation(1))

    assert lifecycle.get("active_phase") is None


# ===========================================================================
# add-coordexp-swift-training-observability Wave 3 (tasks 3.3-3.7)
#
# The reporter CONSUMES completed loss telemetry, the runtime-owned update
# receipt, exact work counts, honest timing scopes, and allocator samples. It
# never recomputes an objective, never invents boundary booleans, and never
# publishes a fabricated zero for something it could not measure.
# ===========================================================================


import copy

from src.runtime.optimizer_boundary import (
    AppliedUpdateReceipt,
    OptimizerBoundaryTerminal,
)


def _denominator(name: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "term_name": name,
        "denominator_scope": "planned_step",
        "eligible_segment_count": 2,
        "selected_atom_count": 4,
        "skipped_segment_count": 1,
        "context_count": 2,
    }
    payload.update(overrides)
    return payload


def _loss_artifact(*, include_gate: bool = True) -> dict[str, object]:
    """A finalized planned-step loss artifact shaped like the real owner's.

    `segment_mean_numerator` and `backend_gradient_scale` are deliberately
    inconsistent with the published raw/weighted values: any reporter that
    reconstructed the objective from sufficient statistics or backend-scaled
    tensors would produce a visibly different row.
    """

    metrics: dict[str, float] = {
        "loss/total": 1.0,
        "loss/base_ce/raw": 1.0,
        "loss/base_ce/weighted": 1.0,
        "loss/base_ce/selected_count": 4.0,
        "loss/base_ce/segment_count": 2.0,
        "loss/base_ce/token_weighted_diag": 0.75,
        "count/supervised_atoms": 4.0,
        "count/eligible_segments": 2.0,
        "count/skipped_segments": 1.0,
        "count/packs": 2.0,
        "count/examples": 2.0,
        "finite/total_loss": 1.0,
        "finite/base_ce": 1.0,
        "acc_top1": 0.5,
        "acc_top5": 1.0,
    }
    terms: list[dict[str, object]] = [
        {
            "name": "base_ce",
            "weight": 1.0,
            "raw_loss": 1.0,
            "weighted_loss": 1.0,
            "selected_count": 4,
            "skipped_count": 1,
            "segment_mean_numerator": 400.0,
            "backend_gradient_scale": 2.0,
            "backward_contribution": 2.0,
            "denominator": _denominator("base_ce"),
        }
    ]
    if include_gate:
        metrics.update(
            {
                "loss/token_gate/raw": 0.5,
                "loss/token_gate/weighted": 0.0,
                "loss/token_gate/selected_count": 2.0,
                "loss/token_gate/segment_count": 2.0,
                "loss/token_gate/token_weighted_diag": 0.25,
                "finite/token_gate": 1.0,
            }
        )
        terms.append(
            {
                "name": "token_gate",
                "weight": 0.0,
                "raw_loss": 0.5,
                "weighted_loss": 0.0,
                "selected_count": 2,
                "skipped_count": 0,
                "segment_mean_numerator": 100.0,
                "backend_gradient_scale": 2.0,
                "backward_contribution": 0.0,
                "denominator": _denominator(
                    "token_gate", selected_atom_count=2, skipped_segment_count=0
                ),
            }
        )
    return {
        "metrics": metrics,
        "terms": terms,
        "counts": {
            "count/supervised_atoms": 4,
            "count/eligible_segments": 2,
            "count/skipped_segments": 1,
            "count/packs": 2,
            "count/examples": 2,
        },
        "accuracy_stats": {"top1_correct": 2, "top5_correct": 4, "atom_count": 4},
    }


def _loss_observation(step: int = 1, **overrides: object) -> CompletedStepObservation:
    defaults: dict[str, object] = dict(
        planned_step_id=step,
        micro_step_count=2,
        loss_bundle_artifact=_loss_artifact(),
        optimizer_update_status="applied",
        finite_status="finite",
        scheduler_artifact={"learning_rates": [{"group_index": 0, "lr": 2e-5}]},
        step_duration_seconds=0.5,
        input_build_seconds=0.05,
        input_wait_seconds=0.01,
        physical_token_count=16,
    )
    defaults.update(overrides)
    return CompletedStepObservation(**defaults)  # type: ignore[arg-type]


class _TwoRankRuntime:
    """Reduce the reporter's own batch against a deliberately asymmetric peer.

    The peer payload is derived from the local one, so cross-rank schema
    agreement is preserved while rank-local VALUES differ - which is exactly
    what makes SUM/MAX/ratio reducer mistakes visible in the row.
    """

    is_main_process = True
    world_size = 2

    class _Accelerator:
        is_main_process = True
        num_processes = 2

    accelerator = _Accelerator()
    optimizer_step_count = 7
    scheduler_step_count = 5

    def __init__(self, peer_values: dict[str, float] | None = None) -> None:
        self.peer_values = dict(peer_values or {})
        self.batches: list[object] = []

    def gather_metrics(self, batch: object) -> object:
        self.batches.append(batch)
        local = batch.to_rank_payload(rank=0, world_size=2)  # type: ignore[attr-defined]
        peer = copy.deepcopy(local)
        peer["rank"] = 1
        for sample in peer["samples"]:
            name = sample["name"]
            if name not in self.peer_values:
                continue
            if sample["form"] == "ratio":
                sample["numerator"] = self.peer_values[name] * sample["denominator"]
                continue
            sample["value"] = self.peer_values[name]
        reduced = reduce_rank_payloads([local, peer], world_size=2)
        result: dict[str, object] = {
            "metrics": dict(reduced.metrics),
            "per_rank_metrics": dict(reduced.per_rank_metrics),
        }
        if reduced.accuracy_stats is not None:
            result["accuracy_stats"] = dict(reduced.accuracy_stats)
        return result


def _row(writer: RunWriter) -> dict[str, object]:
    lines = writer.logging_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, lines
    return json.loads(lines[0])


# ---------------------------------------------------------------------------
# 3.3 - consumed, never reconstructed, loss telemetry
# ---------------------------------------------------------------------------


def test_train_row_exposes_configured_weight_and_denominator_inputs(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation())

    row = _row(writer)

    assert row["loss/base_ce/raw"] == 1.0
    assert row["loss/base_ce/weighted"] == 1.0
    assert row["loss/base_ce/weight"] == 1.0
    assert row["loss/base_ce/selected_atom_count"] == 4.0
    assert row["loss/base_ce/skipped_segment_count"] == 1.0
    # `segment_count` IS the eligible-segment count published by the loss owner.
    assert row["loss/base_ce/segment_count"] == 2.0
    assert row["loss/base_ce/denominator_scope"] == "planned_step"


def test_train_row_retains_a_computed_zero_weight_gate_diagnostic(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation())

    row = _row(writer)

    assert row["loss/token_gate/raw"] == 0.5
    assert row["loss/token_gate/weighted"] == 0.0
    assert row["loss/token_gate/weight"] == 0.0
    assert row["loss/token_gate/selected_count"] == 2.0
    assert row["finite/token_gate"] == 1.0


def test_train_row_has_no_field_family_for_an_omitted_optional_term(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation(loss_bundle_artifact=_loss_artifact(include_gate=False)))

    row = _row(writer)

    assert not any("token_gate" in key for key in row)
    assert not any(key.startswith("loss/coordinate_gaussian") for key in row)


def test_train_row_does_not_reconstruct_loss_from_backend_scaled_statistics(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation())

    row = _row(writer)

    # The artifact's numerator/backend scale would give 100.0 / 2.0-scaled
    # values; the row must be the loss owner's published telemetry.
    assert row["loss/total"] == 1.0
    assert row["loss/base_ce/raw"] == 1.0
    assert "segment_mean_numerator" not in str(sorted(row))
    assert not any("backend_gradient_scale" in key for key in row)


def test_asymmetric_two_rank_row_uses_declared_sum_max_and_ratio_reducers(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    runtime = _TwoRankRuntime(
        peer_values={
            "loss/total": 3.0,
            "loss/base_ce/raw": 3.0,
            "step_duration_seconds": 0.9,
            "input_build_seconds": 0.2,
            "count/physical_tokens": 32.0,
            "count/packs": 4.0,
        }
    )
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=runtime
    )(_loss_observation())

    row = _row(writer)

    assert row["loss/total"] == 4.0  # partial rank contributions SUM
    assert row["loss/base_ce/raw"] == 4.0
    assert row["step_duration_seconds"] == 0.9  # slowest rank owns the path
    assert row["input_build_seconds"] == 0.2
    assert row["count/packs"] == 6.0
    assert row["count/supervised_atoms"] == 4.0  # merged global denominator
    assert row["loss/base_ce/weight"] == 1.0  # configured, identical everywhere


# ---------------------------------------------------------------------------
# 3.4 - global throughput from summed work over the rank-max duration
# ---------------------------------------------------------------------------


def test_train_row_derives_throughput_from_summed_work_over_rank_max_duration(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    runtime = _TwoRankRuntime(
        peer_values={
            "step_duration_seconds": 1.0,
            "count/physical_tokens": 32.0,
            "count/packs": 4.0,
        }
    )
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=runtime
    )(_loss_observation())

    row = _row(writer)

    # 16 + 32 physical tokens over the 1.0s rank maximum (never the mean, and
    # never one rank's local work).
    assert row["count/physical_tokens"] == 48.0
    assert row["throughput/physical_tokens_per_second"] == pytest.approx(48.0)
    assert row["throughput/packs_per_second"] == pytest.approx(6.0)
    assert row["throughput/supervised_atoms_per_second"] == pytest.approx(4.0)


def test_throughput_is_unavailable_without_a_work_count(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation(physical_token_count=None))

    row = _row(writer)

    assert "throughput/physical_tokens_per_second" not in row
    assert "count/physical_tokens" not in row
    assert "throughput/physical_tokens_per_second" in row["unavailable_fields"]
    assert row["throughput/packs_per_second"] == pytest.approx(4.0)


def test_throughput_is_unavailable_without_a_positive_step_duration(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation(step_duration_seconds=0.0))

    row = _row(writer)

    for field in (
        "throughput/physical_tokens_per_second",
        "throughput/packs_per_second",
        "throughput/supervised_atoms_per_second",
    ):
        assert field not in row
        assert field in row["unavailable_fields"]


# ---------------------------------------------------------------------------
# 3.5 - honest input timing in the row
# ---------------------------------------------------------------------------


def test_train_row_publishes_a_resolved_h2d_measurement(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation(input_h2d_seconds=0.02))

    row = _row(writer)

    assert row["input_h2d_seconds"] == pytest.approx(0.02)
    assert "input_h2d_seconds" not in row.get("unavailable_fields", [])


def test_train_row_marks_an_unmeasurable_h2d_unavailable_without_a_zero(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation(input_h2d_unavailable_reason="h2d_target_not_cuda"))

    row = _row(writer)

    assert "input_h2d_seconds" not in row
    assert "input_h2d_seconds" in row["unavailable_fields"]


def test_train_row_publishes_the_all_rank_pre_clip_gradient_norm(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_loss_observation(pre_clip_grad_norm_rank_max=2.25))

    row = _row(writer)

    assert row["grad_norm/pre_clip_rank_max"] == pytest.approx(2.25)


def test_an_unsafe_branch_marks_the_pre_clip_norm_unavailable(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(
        _loss_observation(
            pre_clip_grad_norm_rank_max=None,
            update_receipt=AppliedUpdateReceipt.not_attempted(
                1, 1, "skipped_non_finite_scalar"
            ),
        )
    )

    row = _row(writer)

    assert "grad_norm/pre_clip_rank_max" not in row
    assert "grad_norm/pre_clip_rank_max" in row["unavailable_fields"]


# ---------------------------------------------------------------------------
# 3.6 - CUDA allocator observation
# ---------------------------------------------------------------------------


def _allocator_sample(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "scope": "current_process_current_device",
        "available": True,
        "unavailable_reason": None,
        "device_index": 0,
        "current_allocated_bytes": 1024,
        "current_reserved_bytes": 2048,
        "num_alloc_retries": 2,
        "num_ooms": 0,
    }
    payload.update(overrides)
    return payload


def test_train_row_reduces_memory_bytes_by_rank_max_and_deltas_by_rank_sum(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    runtime = _TwoRankRuntime(
        peer_values={
            "resource/gpu_current_memory_allocated_bytes": 4096.0,
            "resource/gpu_alloc_retries_delta": 3.0,
        }
    )
    samples = iter(
        (
            _allocator_sample(num_alloc_retries=2),
            _allocator_sample(num_alloc_retries=5, current_allocated_bytes=1024),
        )
    )
    reporter = reporting.CompletedStepReporter(
        writer=writer,
        lifecycle={},
        runtime=runtime,
        cuda_allocator_sampler=lambda: next(samples),
    )

    reporter(_loss_observation())
    reporter(_loss_observation(2))

    rows = [
        json.loads(line)
        for line in writer.logging_path.read_text(encoding="utf-8").splitlines()
    ]
    second = rows[1]
    assert second["resource/gpu_current_memory_allocated_bytes"] == 4096.0
    assert second["resource/gpu_current_memory_reserved_bytes"] == 2048.0
    # rank-local delta 3 plus the injected peer delta 3.
    assert second["resource/gpu_alloc_retries_delta"] == 6.0
    assert second["resource/gpu_ooms_delta"] == 0.0


def test_first_observed_step_has_no_allocator_delta_and_fabricates_no_zero(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer,
        lifecycle={},
        runtime=_Runtime(),
        cuda_allocator_sampler=_allocator_sample,
    )(_loss_observation())

    row = _row(writer)

    assert row["resource/gpu_current_memory_allocated_bytes"] == 1024.0
    assert "resource/gpu_alloc_retries_delta" not in row
    assert "resource/gpu_alloc_retries_delta" in row["unavailable_fields"]
    assert "resource/gpu_ooms_delta" in row["unavailable_fields"]


def test_train_row_omits_cuda_memory_fields_when_the_backend_cannot_measure(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    unavailable = {
        "schema_version": 1,
        "scope": "current_process_current_device",
        "available": False,
        "unavailable_reason": "cuda_not_initialized",
        "device_index": None,
        "current_allocated_bytes": {"status": "unavailable", "reason": "x"},
        "current_reserved_bytes": {"status": "unavailable", "reason": "x"},
        "num_alloc_retries": {"status": "unavailable", "reason": "x"},
        "num_ooms": {"status": "unavailable", "reason": "x"},
    }
    reporting.CompletedStepReporter(
        writer=writer,
        lifecycle={},
        runtime=_Runtime(),
        cuda_allocator_sampler=lambda: unavailable,
    )(_loss_observation())

    row = _row(writer)

    for field in (
        "resource/gpu_current_memory_allocated_bytes",
        "resource/gpu_current_memory_reserved_bytes",
        "resource/gpu_alloc_retries_delta",
        "resource/gpu_ooms_delta",
    ):
        assert field not in row
        assert field in row["unavailable_fields"]


# ---------------------------------------------------------------------------
# 3.7 - boundary truth, applied LR, counters, and the terminal row
# ---------------------------------------------------------------------------


def test_train_row_publishes_the_receipt_applied_learning_rates(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(
        _loss_observation(
            update_receipt=AppliedUpdateReceipt.applied_update(
                1, group_learning_rates=(1e-4,), post_wrapper_outcome="none_skipped"
            ),
            # The scheduler's post-advance value must NOT win.
            scheduler_artifact={"learning_rates": [{"group_index": 0, "lr": 9e-9}]},
        )
    )

    row = _row(writer)

    assert row["lr/group_0"] == pytest.approx(1e-4)
    assert row["optimizer_boundary_action"] == "apply"
    assert row["optimizer_update_attempted"] is True
    assert row["optimizer_update_applied"] is True
    assert row["optimizer_step_was_skipped"] is False
    assert row["optimizer_mutation_state"] == "applied"


def test_a_scaler_skip_nulls_group_lrs_and_names_them_unavailable(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    receipt = AppliedUpdateReceipt.scaler_skipped(1, 2)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(
        _loss_observation(
            update_receipt=receipt,
            optimizer_update_status=receipt.optimizer_update_status,
            scheduler_artifact={
                "learning_rates": [
                    {"group_index": 0, "lr": 1e-4},
                    {"group_index": 1, "lr": 2e-4},
                ]
            },
        )
    )

    row = _row(writer)

    assert row["lr/group_0"] is None
    assert row["lr/group_1"] is None
    assert "lr/group_0" in row["unavailable_fields"]
    assert "lr/group_1" in row["unavailable_fields"]
    assert row["optimizer_update_status"] == "skipped_scaler_overflow"
    assert row["optimizer_update_applied"] is False
    assert row["optimizer_step_was_skipped"] is True


def test_train_row_carries_the_current_optimizer_and_scheduler_counters(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_TwoRankRuntime()
    )(_loss_observation())

    row = _row(writer)

    assert row["optimizer_step_count"] == 7
    assert row["scheduler_step_count"] == 5


def test_terminal_boundary_row_publishes_once_without_advancing_counters(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {"completed_steps": 41, "consumed_packs": 82}
    runtime = _TwoRankRuntime()
    receipt = AppliedUpdateReceipt.terminal_post_wrapper(
        42, 2, action="apply", outcome="mixed", pre_call_learning_rates=(1e-4, 2e-4)
    )

    reporting.publish_terminal_boundary_row(
        writer=writer,
        runtime=runtime,
        lifecycle=lifecycle,
        terminal=OptimizerBoundaryTerminal(receipt),
    )

    row = _row(writer)
    assert row["step"] == 42
    assert row["split"] == "train"
    assert row["optimizer_boundary_terminal"] is True
    # Never the previous COMPLETED step's finite status relabeled as this
    # planned step's own truth.
    assert row["finite_status"] == "unavailable"
    assert row["optimizer_terminal_reason"] == "post_wrapper_mixed"
    assert row["optimizer_update_attempted"] is True
    assert row["optimizer_update_applied"] is None
    assert row["optimizer_step_was_skipped"] is None
    assert row["optimizer_mutation_state"] == "divergent_or_unknown"
    assert row["lr/group_0"] is None and row["lr/group_1"] is None
    assert row["optimizer_step_count"] == 7
    assert row["scheduler_step_count"] == 5
    # The boundary is NOT a completed planned step.
    assert lifecycle == {"completed_steps": 41, "consumed_packs": 82}
    # No metric collective may run on a divergent boundary.
    assert runtime.batches == []


def test_terminal_row_publication_failure_preserves_the_primary_boundary_code(
    tmp_path: Path,
) -> None:
    receipt = AppliedUpdateReceipt.terminal_not_attempted(
        42, 1, "pre_wrapper_mixed_scaler_overflow", unscale_completed=True
    )
    failing_writer = SimpleNamespace(
        append_logging_row=lambda row: (_ for _ in ()).throw(OSError("disk full"))
    )

    # Best effort: the caller keeps raising the optimizer-boundary failure.
    reporting.publish_terminal_boundary_row(
        writer=failing_writer,
        runtime=_Runtime(),
        lifecycle={},
        terminal=OptimizerBoundaryTerminal(receipt),
    )


def test_session_publishes_the_terminal_row_before_failed_finalization() -> None:
    source = (
        Path(reporting.__file__).resolve().parent / "session.py"
    ).read_text(encoding="utf-8")

    assert "except OptimizerBoundaryTerminal as" in source
    assert "publish_terminal_boundary_row(" in source
    # Session may HOLD boundary truth but must never build it: only the
    # runtime owner may produce an `AppliedUpdateReceipt`.
    assert "AppliedUpdateReceipt" not in source


# ---------------------------------------------------------------------------
# Additive-only envelope: the frozen characterization row is unchanged
# ---------------------------------------------------------------------------


def test_an_observation_without_wave3_inputs_gains_no_row_key(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=_Runtime()
    )(_observation(1, step_duration_seconds=0.25))

    row = _row(writer)

    assert set(row) == {
        "acc_top1",
        "acc_top5",
        "accuracy_stats",
        "finite_status",
        "loss/total",
        "lr/group_0",
        "micro_step_count",
        "non_finite_fields",
        "optimizer_update_status",
        "split",
        "step",
        "step_duration_seconds",
    }
