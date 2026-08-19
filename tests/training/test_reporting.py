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

from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.training import reporting
from src.training.supervised_trainer import CompletedStepObservation


class _Accelerator:
    is_main_process = True
    num_processes = 1


class _Runtime:
    is_main_process = True
    world_size = 1
    accelerator = _Accelerator()

    def gather_metrics(self, metrics: object, **kwargs: object) -> object:
        result: dict[str, object] = {"metrics": dict(metrics)}  # type: ignore[arg-type]
        accuracy_stats = kwargs.get("accuracy_stats")
        if isinstance(accuracy_stats, dict):
            result["accuracy_stats"] = dict(accuracy_stats)
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
    import inspect

    signature = inspect.signature(reporting.CompletedStepReporter.__init__)
    parameters = signature.parameters
    assert set(parameters) == {
        "self",
        "writer",
        "lifecycle",
        "runtime",
        "resource_collector",
    }
    for name in ("writer", "lifecycle", "runtime", "resource_collector"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["resource_collector"].default is None

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
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            calls.append({"metrics": dict(metrics), **kwargs})  # type: ignore[arg-type]
            return super().gather_metrics(metrics, **kwargs)

    reporting.CompletedStepReporter(writer=writer, lifecycle={}, runtime=Runtime())(
        _observation(1)
    )

    assert calls[0]["split"] == "train"
    assert calls[0]["planned_step_id"] == 1
    assert calls[0]["accuracy_stats"] == {
        "top1_correct": 1,
        "top5_correct": 2,
        "atom_count": 2,
    }


def test_reporter_rejects_reduction_without_scalar_mapping(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    class Runtime(_Runtime):
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
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
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            return {"metrics": dict(metrics), "accuracy_stats": None}  # type: ignore[arg-type]

    reporter = reporting.CompletedStepReporter(
        writer=writer, lifecycle={}, runtime=Runtime()
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        reporter(_observation(1))
    assert exc_info.value.code == "runtime.train_accuracy_stats_reduction_failed"


# ---------------------------------------------------------------------------
# Rank-resource accounting
# ---------------------------------------------------------------------------


def test_per_rank_measurement_selects_bounded_prefixes_and_sorts_keys() -> None:
    receipt = reporting._per_rank_measurement(
        {
            "per_rank_metrics": {
                "1": {
                    "step_duration_seconds": 0.2,
                    "resource/cpu_max_rss_bytes": 10,
                    "unrelated_field": 5,
                },
                "0": {"input_wait_seconds": 0.1},
            }
        }
    )
    assert receipt == {
        "1": {"resource/cpu_max_rss_bytes": 10.0, "step_duration_seconds": 0.2},
        "0": {"input_wait_seconds": 0.1},
    }


def test_per_rank_measurement_returns_none_when_nothing_selected() -> None:
    assert reporting._per_rank_measurement({"per_rank_metrics": {}}) is None
    assert reporting._per_rank_measurement({}) is None


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
            reporting._append_logging_row_shared(
                writer=writer, row={"step": 1, "split": "train"}, runtime=runtime
            )
        assert exc_info.value.code == "runtime.logging_append_failed"
        assert "OSError: disk full" in str(exc_info.value)


def test_append_logging_row_shared_appends_successfully_on_rank_zero(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    reporting._append_logging_row_shared(
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
