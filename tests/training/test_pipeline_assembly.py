from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.training.cache_workflow as cache_workflow
import src.training.control_plane as control_plane
import src.training.execution_plan as execution_plan
import src.training.pipeline as pipeline
import src.training.reporting as reporting
from src.artifacts.run_writer import RunWriter
from src.common.errors import RuntimeContractError
from src.config.models import RunDirectory
from src.losses import LossContext, LossRunner, TokenVocabularyGroups
from src.packing.planner import PackedSegment, plan_packed_sequences
from src.qwen.forward import build_qwen_forward_inputs
from src.qwen.positions import build_qwen_position_inputs
from src.supervision import TokenAtom, TokenSequence
from src.training.supervised_trainer import (
    CompletedStepObservation,
    SupervisedMicroStep,
)
from src.training.forward_input_provider import build_forward_input_provider
from src.runtime.seeding import seed_training_runtime

def _patch_shared_cache_import(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object
) -> None:
    """Replace one shared import on every module that now reads it.

    Wave 3 of ``decompose-coordexp-swift-training-orchestration`` moved the
    cache preparation/admission/hydration orchestration into
    ``src/training/cache_workflow.py``.  Names both owners import must be
    replaced on both, or a seam that used to be a single patch point would
    silently reach production through the other owner.
    """

    for module in (pipeline, cache_workflow):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)


class _Accelerator:
    is_main_process = True
    num_processes = 1
    process_index = 0


class _Runtime:
    is_main_process = True
    world_size = 1
    accelerator = _Accelerator()

    def gather_metrics(self, metrics: object, **kwargs: object) -> object:
        result: dict[str, object] = {"metrics": dict(metrics)}
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


def _observation(step: int) -> CompletedStepObservation:
    return CompletedStepObservation(
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


def test_runtime_determinism_policy_identity_is_a_supported_run_binding(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    receipt = seed_training_runtime(
        17,
        determinism_mode="legacy",
        phase="pipeline_entry",
    )

    writer.bind_policy_identity(
        "runtime_determinism",
        receipt.to_policy_identity_dict(),
    )

    assert writer.read_run()["policy_identities"]["runtime_determinism"] == (
        receipt.to_policy_identity_dict()
    )


def test_runtime_determinism_policy_must_converge_across_model_free_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    def gather(report: dict[str, object]) -> tuple[dict[str, object], ...]:
        rank_zero = json.loads(json.dumps(report))
        rank_one = json.loads(json.dumps(report))
        rank_one["rank"] = 1
        rank_one["rank_details"]["runtime_determinism"]["seed"] = 18
        return rank_zero, rank_one

    with pytest.raises(RuntimeContractError) as exc_info:
        cache_workflow._establish_converged_runtime_determinism(
            SimpleNamespace(
                seed=17,
                determinism=SimpleNamespace(mode="legacy"),
            ),
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
            phase="pipeline_entry",
        )

    assert exc_info.value.code == "runtime.determinism_rank_mismatch"


def test_runtime_determinism_consensus_binds_launcher_mapping_and_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    converged = cache_workflow._establish_converged_runtime_determinism(
        SimpleNamespace(
            seed=17,
            determinism=SimpleNamespace(mode="legacy"),
        ),
        rank=0,
        world_size=1,
        rank_report_gatherer=None,
        phase="pipeline_entry",
    )
    policy = cache_workflow._runtime_determinism_run_policy(
        converged,
        pinned_runtime_baseline={"baseline_sha256": "a" * 64},
    )

    assert converged["pre_apply_cuda_initialized"] is False
    assert converged["pre_apply_cuda_initialized_by_rank"] == {"0": False}
    assert converged["launcher_attestations"] == [
        {
            "cuda_visible_devices": ["GPU-a", "GPU-b"],
            "local_rank": 0,
            "logical_cuda_device": 0,
            "rank": 0,
            "world_size": 1,
        }
    ]
    assert policy["pinned_runtime_baseline_sha256"] == "a" * 64


def test_runtime_determinism_is_established_before_model_free_owner_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    config = SimpleNamespace(
        runtime=SimpleNamespace(
            seed=17,
            determinism=SimpleNamespace(mode="legacy"),
        ),
    )
    resolved = SimpleNamespace(config=config)
    monkeypatch.setattr(execution_plan, "load_train_config", lambda _path: resolved)
    monkeypatch.setattr(
        execution_plan,
        "_resolve_model_free_launch_identity",
        lambda: (0, 1),
    )
    monkeypatch.setattr(
        control_plane,
        "_build_model_free_preflight_gatherer",
        lambda _world_size: None,
    )
    monkeypatch.setattr(
        cache_workflow,
        "_establish_converged_runtime_determinism",
        lambda *args, **kwargs: order.append("determinism") or object(),
    )
    monkeypatch.setattr(
        pipeline,
        "_initialize_model_free_run_owner",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("owner setup reached after determinism")
        ),
    )

    with pytest.raises(RuntimeError, match="owner setup reached after determinism"):
        pipeline.run_training_pipeline(tmp_path / "config.yaml")

    assert order == ["determinism"]


def test_checkpoint_handler_binds_exact_state_callback_to_scheduled_step(
    tmp_path: Path,
) -> None:
    exact_state_calls: list[tuple[int, Path]] = []
    writer_calls: list[dict[str, object]] = []

    class CheckpointWriter:
        def write_checkpoint(self, **kwargs: object) -> None:
            writer_calls.append(dict(kwargs))
            callback = kwargs["exact_training_state_callback"]
            assert callable(callback)
            callback(tmp_path / "checkpoints" / "step-3")

    handler = pipeline._checkpoint_handler(
        CheckpointWriter(),
        model=object(),
        runtime=_Runtime(),
        adapter_name="default",
        special_token_result=None,
        schedule=SimpleNamespace(resolved_max_steps=5),
        base_model_path=tmp_path / "model",
        base_config_sha256="base",
        tokenizer_sha256="tokenizer",
        writer=None,
        eval_by_step={},
        committed_steps=set(),
        lifecycle={"checkpoint_event_count": 0},
        save_final=True,
        exact_training_state_callback_factory=lambda step,
        path: exact_state_calls.append((step, path)),
    )

    handler(SimpleNamespace(planned_step_id=3), _observation(3))

    assert len(writer_calls) == 1
    assert exact_state_calls == [(3, tmp_path / "checkpoints" / "step-3")]


def test_exact_resume_cursor_round_trips_next_rank_local_pack() -> None:
    micro_steps = tuple(
        SimpleNamespace(
            pack=SimpleNamespace(
                pack_index=index,
                input_ids=tuple(range(index + 2)),
                segments=(
                    SimpleNamespace(example_id=f"example-{index}-a"),
                    SimpleNamespace(example_id=f"example-{index}-b"),
                ),
            )
        )
        for index in range(6)
    )
    runtime = SimpleNamespace(
        optimizer_step_count=2,
        scheduler_step_count=2,
        zero_grad_count=2,
    )
    schedule = SimpleNamespace(
        resolved_max_steps=3,
        runtime_batch=SimpleNamespace(resolved_grad_accum_steps=2),
    )

    cursor, next_index = pipeline._build_exact_resume_cursor(
        checkpoint_step=2,
        consumed_micro_steps=4,
        train_micro_steps=micro_steps,
        train_cache={"format_version": "v3", "fingerprint": "f" * 64},
        schedule=schedule,
        runtime=runtime,
        rank=0,
        world_size=1,
    )

    assert next_index == 4
    assert cursor["data"]["next_rank_local_micro_step"] == 4
    assert cursor["data"]["runtime_counters"] == {
        "optimizer_step_count": 2,
        "scheduler_step_count": 2,
        "zero_grad_count": 2,
    }
    assert cursor["pack"]["next_pack"] == {
        "example_ids": ["example-4-a", "example-4-b"],
        "pack_index": 4,
        "sequence_length": 6,
    }

    position = pipeline._validate_exact_resume_cursor(
        cursor,
        next_rank_local_micro_step=next_index,
        checkpoint_step=2,
        train_micro_steps=micro_steps,
        train_cache={"format_version": "v3", "fingerprint": "f" * 64},
        schedule=schedule,
        rank=0,
        world_size=1,
    )
    assert position == {
        "next_rank_local_micro_step": 4,
        "start_planned_step_id": 3,
        "runtime_counters": cursor["data"]["runtime_counters"],
    }


def test_exact_resume_cursor_rejects_consumption_drift() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._build_exact_resume_cursor(
            checkpoint_step=2,
            consumed_micro_steps=3,
            train_micro_steps=(object(),) * 6,
            train_cache={"format_version": "v3", "fingerprint": "f" * 64},
            schedule=SimpleNamespace(
                resolved_max_steps=3,
                runtime_batch=SimpleNamespace(resolved_grad_accum_steps=2),
            ),
            runtime=SimpleNamespace(
                optimizer_step_count=2,
                scheduler_step_count=2,
                zero_grad_count=2,
            ),
            rank=0,
            world_size=1,
        )

    assert exc_info.value.code == "training.resume_cursor_position_mismatch"


def test_five_train_and_two_eval_callbacks_write_exact_wide_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {}
    runtime = _Runtime()
    train = reporting.CompletedStepReporter(writer=writer, lifecycle=lifecycle, runtime=runtime)

    class FakeEvalRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, *, planned_step_id: int, trigger_reasons: object) -> object:
            return SimpleNamespace(
                to_logging_row=lambda: {
                    "step": planned_step_id,
                    "split": "eval",
                    "trigger_reasons": list(trigger_reasons),
                    "example_count": 2,
                    "pack_count": 1,
                    "acc_top1": 0.6,
                }
            )

    monkeypatch.setattr(pipeline, "ForwardEvalRunner", FakeEvalRunner)
    eval_handler = pipeline._eval_forward_handler(
        model=object(),
        runtime=runtime,
        eval_micro_steps=(),
        loss_runner=object(),
        writer=writer,
        eval_source={"path": "eval.jsonl"},
        eval_by_step={},
    )
    for step in range(1, 6):
        train(_observation(step))
        if step in {2, 4}:
            eval_handler(
                SimpleNamespace(planned_step_id=step, trigger_reasons=("scheduled",)),
                _observation(step),
            )

    rows = [json.loads(line) for line in writer.logging_path.read_text().splitlines()]
    assert [row["split"] for row in rows].count("train") == 5
    assert [row["split"] for row in rows].count("eval") == 2
    assert lifecycle["completed_steps"] == 5
    assert lifecycle["consumed_packs"] == 10
    assert all("non_finite_fields" in row for row in rows)


def test_five_step_lifecycle_sums_only_steps_three_to_five_and_eval_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {
        "active_phase": None,
        "measurement_warmup_steps": 2,
        "resolved_max_steps": 5,
        "expected_measured_steps": 3,
        "accepted_measured_steps": 0,
        "steady_state_duration_seconds": 0.0,
        "evaluation_event_count": 0,
        "evaluation_duration_seconds": 0.0,
        "evaluation_resource_high_water": None,
        "evaluation_summary_recorded": False,
    }
    gather_calls: list[dict[str, object]] = []

    class Runtime(_Runtime):
        rank = 0

        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            values = dict(metrics)  # type: ignore[arg-type]
            gather_calls.append({"metrics": values, **kwargs})
            return {"metrics": values, "per_rank_metrics": {"0": values}}

    class FakeEvalRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, *, planned_step_id: int, trigger_reasons: object) -> object:
            return SimpleNamespace(
                to_logging_row=lambda: {
                    "step": planned_step_id,
                    "split": "eval",
                    "trigger_reasons": list(trigger_reasons),
                    "example_count": 2,
                    "pack_count": 1,
                    "acc_top1": 0.6,
                }
            )

    monotonic_values = iter((10.0, 11.5, 20.0, 22.0, 100.0, 200.0))
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(pipeline, "ForwardEvalRunner", FakeEvalRunner)
    payload_identity = {
        "schema": "coordexp-swift-inference-checkpoint-payload-publication",
        "schema_version": 2,
        "manifest_relative_path": "inference_payload_manifest.json",
        "manifest_file_sha256": "a" * 64,
        "aggregate_digest": "b" * 64,
    }
    monkeypatch.setattr(
        pipeline,
        "build_inference_checkpoint_payload_identity",
        lambda checkpoint_dir: payload_identity,
    )
    monkeypatch.setattr(
        "src.artifacts.run_writer.admit_inference_checkpoint_payload_identity",
        lambda checkpoint_dir, expected: dict(expected),
    )
    resource_values = iter(
        (
            {
                "cpu": {
                    "max_rss_bytes": 100,
                    "io_read_bytes": 10,
                    "io_write_bytes": 20,
                },
                "gpu": {"initialized": False},
            },
            {
                "cpu": {
                    "max_rss_bytes": 150,
                    "io_read_bytes": 30,
                    "io_write_bytes": 25,
                },
                "gpu": {"initialized": False},
            },
        )
    )
    runtime = Runtime()
    train_handler = reporting.CompletedStepReporter(writer=writer, lifecycle=lifecycle, runtime=runtime)
    eval_handler = pipeline._eval_forward_handler(
        model=object(),
        runtime=runtime,
        eval_micro_steps=(),
        loss_runner=object(),
        writer=writer,
        eval_source={"path": "eval.jsonl"},
        eval_by_step={},
        lifecycle=lifecycle,
        resource_collector=lambda: next(resource_values),
    )
    checkpoint_calls: list[int] = []
    checkpoint_handler = pipeline._checkpoint_handler(
        SimpleNamespace(
            write_checkpoint=lambda **kwargs: checkpoint_calls.append(
                int(kwargs["step"])
            )
        ),
        model=object(),
        runtime=runtime,
        adapter_name="default",
        special_token_result=object(),
        schedule=SimpleNamespace(resolved_max_steps=5),
        base_model_path=tmp_path / "model",
        base_config_sha256="base",
        tokenizer_sha256="tokenizer",
        writer=writer,
        eval_by_step={},
        committed_steps=set(),
        lifecycle=lifecycle,
        save_final=True,
    )

    for step, duration in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), start=1):
        observation = CompletedStepObservation(
            planned_step_id=step,
            micro_step_count=1,
            loss_bundle_artifact={"metrics": {"loss/total": 1.0}},
            optimizer_update_status="applied",
            finite_status="finite",
            step_duration_seconds=duration,
            input_build_seconds=0.1,
            input_wait_seconds=0.0,
        )
        train_handler(observation)
        if step in {3, 5}:
            eval_handler(
                SimpleNamespace(
                    planned_step_id=step,
                    trigger_reasons=("scheduled" if step == 3 else "final",),
                ),
                observation,
            )
        if step == 5:
            checkpoint_handler(
                SimpleNamespace(planned_step_id=5),
                observation,
            )

    pipeline._record_terminal_measurement_summaries(writer, lifecycle)
    measurement = writer.read_run()["measurement"]
    steady = measurement["phases"]["steady_state"]
    assert steady["duration_scope"] == ("sum_of_accepted_all_rank_max_step_durations")
    assert steady["duration_seconds"] == pytest.approx(3.0 + 4.0 + 5.0)
    assert measurement["phases"]["checkpoint_publication"][
        "duration_seconds"
    ] == pytest.approx(100.0)
    assert checkpoint_calls == [5]
    assert steady["accepted_measured_steps"] == 3
    assert steady["expected_measured_steps"] == 3
    assert measurement["steady_state_eligible"] is True
    evaluation = measurement["phases"]["evaluation_execution"]
    assert evaluation["event_count"] == 2
    assert evaluation["duration_seconds"] == pytest.approx(3.5)
    assert evaluation["resource_observation_scope"] == (
        "process_lifetime_high_water_observed_after_evaluation"
    )
    assert evaluation["resource_high_water_observed_after_events"] == {
        "resource/cpu_io_read_bytes": 30.0,
        "resource/cpu_io_write_bytes": 25.0,
        "resource/cpu_max_rss_bytes": 150.0,
    }
    assert measurement["terminal_phase"] == "evaluation_execution"
    assert measurement["terminal_phase_status"] == "completed"
    assert measurement["last_completed_phase"] == "evaluation_execution"
    assert measurement["failure_phase"] is None
    eval_rows = [
        json.loads(line)
        for line in writer.logging_path.read_text().splitlines()
        if json.loads(line)["split"] == "eval"
    ]
    assert [row["eval_duration_seconds"] for row in eval_rows] == [1.5, 2.0]
    assert {row["resource_observation_scope"] for row in eval_rows} == {
        "process_lifetime_high_water_observed_after_evaluation"
    }
    assert all("per_rank_measurement" in row for row in eval_rows)
    assert [
        call["split"] for call in gather_calls if call["split"] == "eval.measurement"
    ] == ["eval.measurement", "eval.measurement"]


def test_eval_exception_persists_failed_terminal_phase_and_ineligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {
        "evaluation_event_count": 0,
        "evaluation_summary_recorded": False,
    }

    class FailingEvalRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            raise ValueError("secret-shaped detail must not enter run.json")

    runtime = _Runtime()
    gather_calls: list[object] = []
    runtime.gather_metrics = lambda *args, **kwargs: gather_calls.append(kwargs)  # type: ignore[method-assign]
    monotonic_values = iter((30.0, 31.25))
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(pipeline, "ForwardEvalRunner", FailingEvalRunner)
    handler = pipeline._eval_forward_handler(
        model=object(),
        runtime=runtime,
        eval_micro_steps=(),
        loss_runner=object(),
        writer=writer,
        eval_source={"path": "eval.jsonl"},
        eval_by_step={},
        lifecycle=lifecycle,
    )

    with pytest.raises(ValueError, match="secret-shaped"):
        handler(
            SimpleNamespace(planned_step_id=3, trigger_reasons=("scheduled",)),
            _observation(3),
        )

    measurement = writer.read_run()["measurement"]
    failed = measurement["phases"]["evaluation_execution"]
    assert failed["status"] == "failed"
    assert failed["duration_seconds"] == pytest.approx(1.25)
    assert failed["resource_observation_scope"] == (
        "process_lifetime_high_water_observed_after_evaluation"
    )
    assert measurement["terminal_phase"] == "evaluation_execution"
    assert measurement["steady_state_eligible"] is False
    assert "secret-shaped" not in json.dumps(failed)
    assert gather_calls == []


def test_success_without_eval_records_explicit_not_run_summary(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    pipeline._record_terminal_measurement_summaries(
        writer,
        {
            "expected_measured_steps": 0,
            "accepted_measured_steps": 0,
            "steady_state_duration_seconds": 0.0,
            "evaluation_event_count": 0,
        },
    )

    phases = writer.read_run()["measurement"]["phases"]
    assert phases["steady_state"]["status"] == "not_run"
    assert phases["steady_state"]["reason"] == "no_post_warmup_steps"
    assert phases["evaluation_execution"]["status"] == "not_run"
    assert phases["evaluation_execution"]["reason"] == (
        "no_scheduled_evaluation_executed"
    )


def test_peer_artifact_initialization_returns_no_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = SimpleNamespace(is_main_process=False, num_processes=2)
    monkeypatch.setattr(
        pipeline,
        "broadcast_object_list",
        lambda values, from_process=0: values.__setitem__(0, {"ok": True}),
    )
    resolved = SimpleNamespace(fingerprint="fp", to_artifact_dict=lambda: {})
    result = pipeline._initialize_artifact_owner(
        accelerator=accelerator,
        run_directory=RunDirectory("run", tmp_path, tmp_path / "run", "created"),
        run_id="run",
        created_at="now",
        resolved_config=resolved,
    )
    assert result is None
    assert not (tmp_path / "run").exists()


def test_rank_zero_writer_initialization_failure_is_shared_and_preserves_collision_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    existing = run_dir / "existing.txt"
    existing.write_text("keep")
    resolved = SimpleNamespace(fingerprint="fp", to_artifact_dict=lambda: {})
    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._initialize_artifact_owner(
            accelerator=_Accelerator(),
            run_directory=RunDirectory("run", tmp_path, run_dir, "fail"),
            run_id="run",
            created_at="now",
            resolved_config=resolved,
        )
    assert exc_info.value.code == "runtime.artifact_initialization_failed"
    assert existing.read_text() == "keep"


def test_rank_zero_invalid_post_init_handshake_finalizes_initialized_writer(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    resolved = SimpleNamespace(fingerprint="fp", to_artifact_dict=lambda: {})
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=2,
        broadcast_object_list=lambda values, from_process=0: [None],
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._initialize_artifact_owner(
            accelerator=accelerator,
            run_directory=RunDirectory("run", tmp_path, run_dir, "created"),
            run_id="run",
            created_at="now",
            resolved_config=resolved,
        )

    assert exc_info.value.code == "runtime.artifact_initialization_failed"
    run_files = list(tmp_path.rglob("run.json"))
    assert run_files == [run_dir / "run.json"]
    state = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["completed_steps"] == 0
    assert state["consumed_packs"] == 0
    assert state["checkpoint_event_count"] == 0
    assert state["terminal_error"] == (
        "artifact initialization handshake failed: RuntimeContractError"
    )


def test_failed_lifecycle_state_keeps_progress_before_original_error(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {"checkpoint_event_count": 1}
    callback = reporting.CompletedStepReporter(writer=writer, lifecycle=lifecycle, runtime=_Runtime())
    callback(_observation(1))
    callback(_observation(2))
    original = RuntimeError("injected after two steps")
    writer.finalize(
        status="failed",
        updated_at="later",
        completed_steps=int(lifecycle["completed_steps"]),
        consumed_packs=int(lifecycle["consumed_packs"]),
        checkpoint_event_count=int(lifecycle["checkpoint_event_count"]),
        optimizer_update_status=str(lifecycle["optimizer_update_status"]),
        finite_status=str(lifecycle["finite_status"]),
        terminal_error=str(original),
    )
    state = writer.read_run()
    assert state["completed_steps"] == 2 and state["consumed_packs"] == 4
    assert state["checkpoint_event_count"] == 1
    assert state["terminal_error"] == "injected after two steps"


def test_train_logging_uses_all_rank_reduced_scalars_and_preserves_nonfinite(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    calls: list[dict[str, object]] = []

    class Runtime(_Runtime):
        world_size = 2

        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            calls.append(dict(metrics))
            return {
                "metrics": {
                    "loss/total": 2.0,
                    "acc_top1": 0.5,
                    "acc_top5": 1.0,
                    "diagnostic/nonfinite": float("inf"),
                },
                "accuracy_stats": {
                    "top1_correct": 2,
                    "top5_correct": 4,
                    "atom_count": 4,
                },
            }

    reporting.CompletedStepReporter(writer=writer, lifecycle={}, runtime=Runtime())(_observation(1))
    row = json.loads(writer.logging_path.read_text())
    # _observation() does not measure timing (production-dead batch-path
    # shape): the timing fields must be entirely absent, not fabricated 0.0.
    assert calls == [
        {
            "lr/group_0": 1e-5,
            "loss/total": 1.0,
            "acc_top1": 0.5,
            "acc_top5": 1.0,
        }
    ]
    assert row["loss/total"] == 2.0
    assert row["acc_top1"] == 0.5
    assert row["diagnostic/nonfinite"] is None
    assert row["non_finite_fields"] == ["diagnostic/nonfinite"]


def test_train_row_carries_timing_fields_additively(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    observation = CompletedStepObservation(
        planned_step_id=1,
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
        step_duration_seconds=0.42,
        input_build_seconds=0.11,
        input_wait_seconds=0.0,
    )

    class Runtime(_Runtime):
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            return {
                "metrics": dict(metrics),
                "accuracy_stats": dict(kwargs["accuracy_stats"]),
            }  # type: ignore[arg-type]

    reporting.CompletedStepReporter(writer=writer, lifecycle={}, runtime=Runtime())(observation)
    row = json.loads(writer.logging_path.read_text())

    # Presence: the three new timing scalars appear in the row.
    assert row["step_duration_seconds"] == pytest.approx(0.42)
    assert row["input_build_seconds"] == pytest.approx(0.11)
    assert row["input_wait_seconds"] == pytest.approx(0.0)
    # Additive-only: every pre-existing field is still present, unrenamed.
    assert row["loss/total"] == 1.0
    assert row["acc_top1"] == 0.5
    assert row["acc_top5"] == 1.0
    assert row["step"] == 1
    assert row["split"] == "train"


def test_train_row_normalizes_non_finite_timing_fields(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    observation = CompletedStepObservation(
        planned_step_id=1,
        micro_step_count=1,
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
        step_duration_seconds=float("inf"),
        input_build_seconds=0.1,
        input_wait_seconds=0.0,
    )

    class Runtime(_Runtime):
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            return {
                "metrics": dict(metrics),
                "accuracy_stats": dict(kwargs["accuracy_stats"]),
            }  # type: ignore[arg-type]

    reporting.CompletedStepReporter(writer=writer, lifecycle={}, runtime=Runtime())(observation)
    row = json.loads(writer.logging_path.read_text())

    assert row["step_duration_seconds"] is None
    assert "step_duration_seconds" in row["non_finite_fields"]
    assert row["input_build_seconds"] == pytest.approx(0.1)


def test_train_row_records_resources_without_rewriting_run_high_water_per_step(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    snapshot = {
        "schema_version": 1,
        "cpu": {
            "scope": "current_process",
            "max_rss_bytes": 1024,
            "io_read_bytes": 2048,
            "io_write_bytes": 4096,
        },
        "gpu": {
            "scope": "current_process_current_device",
            "initialized": True,
            "device_index": 0,
            "max_memory_allocated_bytes": 8192,
            "max_memory_reserved_bytes": 16384,
        },
    }
    observation = CompletedStepObservation(
        planned_step_id=1,
        micro_step_count=1,
        loss_bundle_artifact={"metrics": {"loss/total": 1.0}},
        optimizer_update_status="applied",
        finite_status="finite",
        step_duration_seconds=0.4,
        input_build_seconds=0.1,
        input_wait_seconds=0.02,
    )

    class Runtime(_Runtime):
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            values = dict(metrics)  # type: ignore[arg-type]
            return {
                "metrics": values,
                "per_rank_metrics": {"0": values},
            }

    reporting.CompletedStepReporter(
        writer=writer,
        lifecycle={},
        runtime=Runtime(),
        resource_collector=lambda: snapshot,
    )(observation)

    row = json.loads(writer.logging_path.read_text())
    assert row["resource/cpu_max_rss_bytes"] == 1024.0
    assert row["resource/gpu_max_memory_reserved_bytes"] == 16384.0
    assert row["per_rank_measurement"]["0"]["input_wait_seconds"] == 0.02
    assert (
        row["per_rank_measurement"]["0"]["resource/gpu_max_memory_allocated_bytes"]
        == 8192.0
    )
    assert writer.read_run()["measurement"]["resource_high_water"] is None


@pytest.mark.parametrize(
    ("environment", "expected_enabled", "expected_source"),
    [
        ({}, False, "default"),
        (
            {"COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS": "1"},
            True,
            "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS",
        ),
        (
            {"COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS": "not-enabled"},
            False,
            "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS",
        ),
    ],
)
def test_profile_sync_selector_records_only_resolved_boolean_and_source(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected_enabled: bool,
    expected_source: str,
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert pipeline._resolve_profile_sync_timing_selector() == {
        "enabled": expected_enabled,
        "source": expected_source,
    }


def test_profile_sync_selector_requires_one_exact_receipt_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", raising=False)

    def gather(report: object) -> tuple[object, object]:
        peer = {**dict(report), "rank": 1}  # type: ignore[arg-type]
        peer_details = dict(peer["rank_details"])
        peer_details["resolution"] = {
            "enabled": True,
            "source": "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS",
        }
        peer["rank_details"] = peer_details
        return report, peer

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._resolve_converged_profile_sync_timing_selector(
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
        )

    assert exc_info.value.code == "runtime.profile_sync_timing_resolution_mismatch"


def test_profile_sync_selector_accepts_one_exact_environment_receipt_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", "1")

    def gather(report: object) -> tuple[object, object]:
        return report, {**dict(report), "rank": 1}  # type: ignore[arg-type]

    assert pipeline._resolve_converged_profile_sync_timing_selector(
        rank=0,
        world_size=2,
        rank_report_gatherer=gather,
    ) == {
        "enabled": True,
        "source": "COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS",
    }


def test_pack_cache_root_selector_records_resolved_root_and_allowlisted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", raising=False)
    root, receipt = cache_workflow._resolve_pack_cache_root(tmp_path)
    assert root == tmp_path / ".cache" / "coordexp_swift" / "packing"
    assert receipt == {"resolved_root": str(root.resolve()), "source": "default"}

    monkeypatch.setenv("COORDEXP_SWIFT_PACK_CACHE_ROOT", "relative-cache")
    root, receipt = cache_workflow._resolve_pack_cache_root(tmp_path)
    assert root == Path("relative-cache")
    assert receipt == {
        "resolved_root": str(Path("relative-cache").resolve()),
        "source": "COORDEXP_SWIFT_PACK_CACHE_ROOT",
    }


def _packing_config(
    policy: str,
    *,
    window_size: int | None = None,
    lookahead: int | None = None,
    max_packs_per_fragment: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        packing=SimpleNamespace(
            global_max_length=7,
            policy=policy,
            window_size=window_size,
            lookahead=lookahead,
            seed=17,
            worker_count=1,
            cursor_byte_budget=65_536,
            max_packs_per_fragment=max_packs_per_fragment,
            fragment_item_budget=1_024,
            fragment_byte_budget=4_194_304,
        )
    )


def _encoded_pack_examples() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            example_id=f"example-{index}",
            input_ids=tuple(range(index * 10, index * 10 + length)),
            supervised_token_spans=(),
            ignored_token_spans=(),
        )
        for index, length in enumerate((4, 3, 5))
    )


def test_production_source_order_pack_plan_replays_legacy_membership_exactly() -> None:
    examples = _encoded_pack_examples()

    packs, receipt, fragment_by_pack = cache_workflow._materialize_pack_plan(
        _packing_config("source_order_next_fit"),
        examples,
    )

    assert packs == plan_packed_sequences(examples, global_max_length=7)
    assert receipt["mode"] == "complete_plan"
    assert receipt["policy_identity"]["policy"] == "source_order_next_fit"
    assert receipt["source_input_count"] == 3
    assert set(fragment_by_pack) == {pack.pack_index for pack in packs}


@pytest.mark.parametrize(
    ("config", "expected_mode"),
    [
        (_packing_config("window_binpack", window_size=3), "complete_plan"),
        (
            _packing_config(
                "online_window_binpack",
                lookahead=2,
                max_packs_per_fragment=1,
            ),
            "bounded_online_fragments",
        ),
    ],
)
def test_production_pack_plan_policies_preserve_every_atomic_example_once(
    config: SimpleNamespace,
    expected_mode: str,
) -> None:
    examples = _encoded_pack_examples()

    packs, receipt, fragment_by_pack = cache_workflow._materialize_pack_plan(
        config,
        examples,
    )

    observed_ordinals = [
        segment.example_index for pack in packs for segment in pack.segments
    ]
    assert sorted(observed_ordinals) == [0, 1, 2]
    assert len(observed_ordinals) == len(set(observed_ordinals))
    assert receipt["mode"] == expected_mode
    assert receipt["emitted_pack_count"] == len(packs)
    assert set(fragment_by_pack) == {pack.pack_index for pack in packs}


@pytest.mark.parametrize(
    "name",
    [
        "COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE",
        "COORDEXP_SWIFT_EVAL_REDUCTION_MODE",
    ],
)
def test_policy_selector_source_records_only_default_or_allowlisted_name(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(name, raising=False)
    assert cache_workflow._environment_selector_source(name) == "default"
    monkeypatch.setenv(name, "arbitrary-value-not-recorded")
    assert cache_workflow._environment_selector_source(name) == name

    with pytest.raises(RuntimeContractError) as exc_info:
        cache_workflow._environment_selector_source("SECRET_TOKEN")
    assert exc_info.value.code == "runtime.environment_selector_unsupported"


def test_rank_zero_logging_failure_is_broadcast_as_shared_named_error(
    tmp_path: Path,
) -> None:
    shared: dict[str, object] = {}

    class Collective:
        num_processes = 2
        is_main_process = True

        def broadcast_object_list(
            self, values: list[object], from_process: int = 0
        ) -> None:
            if self.is_main_process:
                shared["status"] = values[0]
            else:
                values[0] = shared["status"]

    accelerator = Collective()
    main_runtime = SimpleNamespace(
        accelerator=accelerator,
        is_main_process=True,
        world_size=2,
    )
    peer_accelerator = Collective()
    peer_accelerator.is_main_process = False
    peer_runtime = SimpleNamespace(
        accelerator=peer_accelerator,
        is_main_process=False,
        world_size=2,
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


@pytest.mark.parametrize(("save_final", "expected_calls"), [(False, 0), (True, 1)])
def test_final_handler_respects_save_final(
    save_final: bool, expected_calls: int
) -> None:
    calls: list[int] = []
    handler = pipeline._final_handler(
        checkpoint_handler=lambda event, observation: calls.append(
            event.planned_step_id
        ),
        committed_steps=set(),
        save_final=save_final,
    )
    handler(SimpleNamespace(planned_step_id=5), _observation(5))
    assert len(calls) == expected_calls


def test_final_handler_deduplicates_same_step_explicit_checkpoint() -> None:
    calls: list[int] = []
    handler = pipeline._final_handler(
        checkpoint_handler=lambda event, observation: calls.append(
            event.planned_step_id
        ),
        committed_steps={5},
        save_final=True,
    )
    handler(SimpleNamespace(planned_step_id=5), _observation(5))
    assert calls == []


@pytest.mark.parametrize(
    ("failure_location", "expected_last_completed", "expected_failure_phase"),
    [
        (
            "optimizer_runtime_assembly",
            "model_loading",
            "optimizer_runtime_assembly",
        ),
        (
            "provider_checkpoint_trainer_assembly_gap",
            "evaluation_hydration",
            "unphased_failure",
        ),
    ],
)
def test_pretrainer_failure_finalizes_truthful_terminal_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_location: str,
    expected_last_completed: str,
    expected_failure_phase: str,
) -> None:
    run_dir = tmp_path / "run"
    config = SimpleNamespace(
        training=SimpleNamespace(precision="no"),
        model=SimpleNamespace(attn_implementation="flash_attention_2"),
        runtime=SimpleNamespace(
            seed=7,
            determinism=SimpleNamespace(mode="legacy"),
        ),
        run=SimpleNamespace(name="run"),
    )
    resolved = SimpleNamespace(
        config=config,
        fingerprint="fp",
        entry_config_path=tmp_path / "config.yaml",
    )
    accelerator = _Accelerator()
    holder: dict[str, RunWriter] = {}
    gatherer_closed: list[bool] = []
    gatherer = SimpleNamespace(close=lambda: gatherer_closed.append(True))
    monkeypatch.setattr(execution_plan, "load_train_config", lambda path: resolved)
    _patch_shared_cache_import(
        monkeypatch,
        "collect_execution_provenance",
        lambda **kwargs: {"schema_version": 1},
    )
    _patch_shared_cache_import(
        monkeypatch,
        "require_pinned_runtime_baseline",
        lambda **kwargs: {
            "schema_version": 3,
            "baseline_sha256": "a" * 64,
            "attention_backend": "flash_attention_2",
            "admitted": True,
            "mismatches": [],
            "reference_only": {},
        },
    )
    monkeypatch.setattr(pipeline, "_build_accelerator", lambda precision: accelerator)
    monkeypatch.setattr(
        pipeline, "validate_accelerator_runtime", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_shared_run_directory",
        lambda *args, **kwargs: RunDirectory("run", tmp_path, run_dir, "created"),
    )

    def initialize(**kwargs: object) -> RunWriter:
        writer = RunWriter.initialize(
            run_dir=run_dir,
            run_id="run",
            run_name="run",
            artifact_root=tmp_path,
            collision_outcome="created",
            created_at="now",
            config_fingerprint="fp",
            resolved_config={},
            world_size=1,
            measurement_context=kwargs["measurement_context"],  # type: ignore[arg-type]
        )
        holder["writer"] = writer
        return writer

    monkeypatch.setattr(pipeline, "_initialize_artifact_owner", initialize)
    monkeypatch.setattr(
        cache_workflow,
        "_resolve_model_free_training_preflight",
        lambda **kwargs: {"duration_seconds": 0.1},
    )
    monkeypatch.setattr(
        control_plane, "_build_rank_report_gatherer", lambda world_size: gatherer
    )

    def fail_provider_assembly(mode: str) -> object:
        raise ValueError(f"{failure_location} failure")

    if failure_location == "provider_checkpoint_trainer_assembly_gap":
        monkeypatch.setattr(
            pipeline,
            "build_forward_input_provider",
            fail_provider_assembly,
        )

    def fail_before_trainer(**kwargs: object) -> object:
        assert kwargs["rank_report_gatherer"] is gatherer
        pipeline._begin_run_phase(
            kwargs["writer"],  # type: ignore[arg-type]
            kwargs["lifecycle"],  # type: ignore[arg-type]
            "model_loading",
        )
        pipeline._finish_run_phase(
            kwargs["writer"],  # type: ignore[arg-type]
            kwargs["lifecycle"],  # type: ignore[arg-type]
            "model_loading",
        )
        if failure_location == "optimizer_runtime_assembly":
            pipeline._begin_run_phase(
                kwargs["writer"],  # type: ignore[arg-type]
                kwargs["lifecycle"],  # type: ignore[arg-type]
                "optimizer_runtime_assembly",
            )
            raise ValueError(f"{failure_location} failure")
        for phase in ("optimizer_runtime_assembly", "evaluation_hydration"):
            pipeline._begin_run_phase(
                kwargs["writer"],  # type: ignore[arg-type]
                kwargs["lifecycle"],  # type: ignore[arg-type]
                phase,
            )
            pipeline._finish_run_phase(
                kwargs["writer"],  # type: ignore[arg-type]
                kwargs["lifecycle"],  # type: ignore[arg-type]
                phase,
            )
        return pipeline.build_forward_input_provider("synchronous")

    monkeypatch.setattr(pipeline, "_run_initialized_training", fail_before_trainer)
    with pytest.raises(ValueError, match=rf"{failure_location} failure"):
        pipeline.run_training_pipeline(tmp_path / "config.yaml")
    state = holder["writer"].read_run()
    assert state["status"] == "failed"
    assert state["completed_steps"] == 0 and state["consumed_packs"] == 0
    assert state["terminal_error"] == f"ValueError: {failure_location} failure"
    measurement = state["measurement"]
    assert measurement["phases"]["model_loading"]["status"] == "completed"
    assert measurement["last_completed_phase"] == expected_last_completed
    assert measurement["terminal_phase"] == expected_failure_phase
    assert measurement["terminal_phase_status"] == "failed"
    assert measurement["failure_phase"] == expected_failure_phase
    assert measurement["phases"][expected_failure_phase]["status"] == "failed"
    if failure_location == "provider_checkpoint_trainer_assembly_gap":
        assert measurement["phases"]["optimizer_runtime_assembly"]["status"] == (
            "completed"
        )
        assert measurement["phases"]["evaluation_hydration"]["status"] == ("completed")
        assert measurement["phases"]["unphased_failure"]["reason"] == (
            "run_failed_outside_named_phase"
        )
    assert state["measurement"]["steady_state_eligible"] is False
    assert state["measurement"]["context"]["profile_sync_timings"] == {
        "enabled": False,
        "source": "default",
    }
    assert gatherer_closed == [True]


def test_prepare_training_pack_caches_is_model_free_and_covers_train_and_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        runtime=SimpleNamespace(
            seed=17,
            determinism=SimpleNamespace(mode="legacy"),
        ),
        model=SimpleNamespace(attn_implementation="flash_attention_2"),
        packing=_packing_config("source_order_next_fit").packing,
        data=SimpleNamespace(train=object(), eval=object(), train_order="source_order"),
        template=SimpleNamespace(object_ordering="geo_sorted"),
    )
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=tmp_path / "config.yaml",
    )
    components = SimpleNamespace(token_identity=object(), tokenizer=object())
    load_model_values: list[bool] = []
    seed_phases: list[str] = []
    train_verification_levels: list[str] = []
    preparation_order: list[str] = []
    train_cache = {
        "status": "complete",
        "build_status": "built",
        "cache_dir": tmp_path / "train-cache",
        "format_version": "v2",
        "fingerprint": "train-fingerprint",
        "manifest_path": tmp_path / "train-cache" / "manifest.json",
        "manifest_sha256": "train-manifest",
        "micro_step_count": 11,
        "phase_receipt": {
            phase: {"status": "completed", "duration_seconds": 0.1}
            for phase in (
                "cache_preparation",
                "cache_publication",
                "cache_admission",
            )
        },
    }
    eval_cache = {
        **train_cache,
        "cache_dir": tmp_path / "eval-cache",
        "fingerprint": "eval-fingerprint",
        "manifest_path": tmp_path / "eval-cache" / "manifest.json",
        "manifest_sha256": "eval-manifest",
        "micro_step_count": 3,
    }

    monkeypatch.setattr(cache_workflow, "load_train_config", lambda path: resolved)
    _patch_shared_cache_import(
        monkeypatch,
        "collect_execution_provenance",
        lambda **kwargs: preparation_order.append("provenance")
        or {"schema_version": 1},
    )
    _patch_shared_cache_import(
        monkeypatch,
        "require_pinned_runtime_baseline",
        lambda **kwargs: preparation_order.append("baseline")
        or {
            "schema_version": 3,
            "baseline_sha256": "a" * 64,
            "attention_backend": "flash_attention_2",
            "admitted": True,
            "mismatches": [],
            "reference_only": {},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "require_mapped_native_execution_attestation",
        lambda **kwargs: pytest.fail(
            "model-free cache preparation attempted mapped-native attestation"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cache_workflow,
        "seed_training_runtime",
        lambda seed, determinism_mode, phase: (
            seed_phases.append(phase),
            preparation_order.append("seed"),
            seed_training_runtime(
                seed,
                determinism_mode=determinism_mode,
                phase=phase,
            ),
        )[-1],
    )

    def load_components(config: object, *, load_model: bool) -> object:
        load_model_values.append(load_model)
        preparation_order.append("components")
        return components

    _patch_shared_cache_import(monkeypatch, "load_qwen_components", load_components)
    _patch_shared_cache_import(
        monkeypatch, "build_token_vocabulary_groups", lambda *args, **kwargs: object()
    )

    def resolve_train_cache(*args: object, **kwargs: object) -> dict[str, object]:
        train_verification_levels.append(str(kwargs["verification_level"]))
        return train_cache

    monkeypatch.setattr(
        cache_workflow, "_resolve_or_build_train_pack_cache", resolve_train_cache
    )
    monkeypatch.setattr(
        cache_workflow, "_resolve_eval_pack_cache", lambda *args, **kwargs: eval_cache
    )

    result = pipeline.prepare_training_pack_caches(tmp_path / "config.yaml")

    assert load_model_values == [False]
    assert seed_phases == ["pack_cache_preparation"]
    assert train_verification_levels == ["payloads"]
    assert preparation_order[:4] == [
        "seed",
        "provenance",
        "baseline",
        "components",
    ]
    assert result["model_loaded"] is False
    assert result["train"]["fingerprint"] == "train-fingerprint"
    assert result["eval"]["fingerprint"] == "eval-fingerprint"
    assert result["train"]["micro_step_count"] == 11
    assert result["policy_identities"]["cache"]["root"] == {
        "resolved_root": str(
            (Path.cwd() / ".cache" / "coordexp_swift" / "packing").resolve()
        ),
        "source": "default",
    }


def test_resolve_eval_pack_cache_hardcodes_payloads_verification_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(data=SimpleNamespace(eval=object()))
    captured_kwargs: dict[str, object] = {}

    def fake_resolve_or_build_pack_cache(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {"status": "complete", "fingerprint": "eval-fingerprint"}

    monkeypatch.setattr(
        cache_workflow, "_resolve_or_build_pack_cache", fake_resolve_or_build_pack_cache
    )

    result = cache_workflow._resolve_eval_pack_cache(
        config,
        components=object(),
        vocab_groups=object(),
        repo_root=tmp_path,
        accelerator=None,
    )

    assert captured_kwargs["verification_level"] == "payloads"
    assert result == {"status": "complete", "fingerprint": "eval-fingerprint"}


def test_prepare_training_pack_caches_marks_all_hit_aggregate_phases_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        runtime=SimpleNamespace(
            seed=17,
            determinism=SimpleNamespace(mode="legacy"),
        ),
        model=SimpleNamespace(attn_implementation="flash_attention_2"),
        packing=_packing_config("source_order_next_fit").packing,
        data=SimpleNamespace(train=object(), eval=object(), train_order="source_order"),
        template=SimpleNamespace(object_ordering="geo_sorted"),
    )
    resolved = SimpleNamespace(
        config=config,
        fingerprint="config-fingerprint",
        entry_config_path=tmp_path / "config.yaml",
    )
    hit_cache = {
        "status": "complete",
        "build_status": "hit",
        "cache_dir": tmp_path / "cache",
        "format_version": "v2",
        "fingerprint": "fingerprint",
        "manifest_path": tmp_path / "cache" / "manifest.json",
        "manifest_sha256": "manifest",
        "micro_step_count": 3,
        "phase_receipt": {
            "cache_preparation": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_publication": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_admission": {"status": "completed", "duration_seconds": 0.1},
        },
    }
    monkeypatch.setattr(cache_workflow, "load_train_config", lambda path: resolved)
    _patch_shared_cache_import(
        monkeypatch,
        "collect_execution_provenance",
        lambda **kwargs: {"schema_version": 1},
    )
    _patch_shared_cache_import(
        monkeypatch,
        "require_pinned_runtime_baseline",
        lambda **kwargs: {
            "schema_version": 3,
            "baseline_sha256": "a" * 64,
            "attention_backend": "flash_attention_2",
            "admitted": True,
            "mismatches": [],
            "reference_only": {},
        },
    )
    monkeypatch.setattr(cache_workflow, "seed_training_runtime", seed_training_runtime)
    _patch_shared_cache_import(
        monkeypatch,
        "load_qwen_components",
        lambda *args, **kwargs: SimpleNamespace(
            token_identity=object(), tokenizer=object()
        ),
    )
    _patch_shared_cache_import(
        monkeypatch, "build_token_vocabulary_groups", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        cache_workflow,
        "_resolve_or_build_train_pack_cache",
        lambda *args, **kwargs: hit_cache,
    )
    monkeypatch.setattr(
        cache_workflow, "_resolve_eval_pack_cache", lambda *args, **kwargs: hit_cache
    )

    result = pipeline.prepare_training_pack_caches(tmp_path / "config.yaml")

    for phase in ("cache_preparation", "cache_publication"):
        assert result["measurement"]["phases"][phase] == {
            "status": "not_run",
            "reason": "all_cache_hits",
            "duration_seconds": 0.0,
        }
    assert result["measurement"]["phases"]["cache_admission"] == {
        "status": "completed",
        "duration_seconds": pytest.approx(0.2),
    }


def test_cache_phase_aggregate_marks_partial_hit_as_completed_mixed() -> None:
    caches = (
        {
            "phase_receipt": {
                "cache_preparation": {
                    "status": "not_run_cache_hit",
                    "duration_seconds": 0.0,
                }
            }
        },
        {
            "phase_receipt": {
                "cache_preparation": {
                    "status": "completed",
                    "duration_seconds": 1.25,
                }
            }
        },
    )

    assert cache_workflow._aggregate_cache_phase(caches, "cache_preparation") == {
        "status": "completed",
        "reason": "mixed_cache_hits_and_builds",
        "duration_seconds": 1.25,
    }


def test_eval_hydration_fewer_packs_uses_exact_replicated_ordinals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", raising=False)
    step = SimpleNamespace()
    selective_loads: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []

    def load_rank_eval(
        path: Path,
        *,
        cache_root: Path,
        expected_fingerprint: str,
        rank: int,
        world_size: int,
    ) -> object:
        selective_loads.append(
            {
                "path": path,
                "cache_root": cache_root,
                "fingerprint": expected_fingerprint,
                "rank": rank,
                "world_size": world_size,
            }
        )
        return SimpleNamespace(
            micro_steps=(step,),
            canonical_ordinals=(0,),
            total_ordinal_count=1,
        )

    def gather(report: object) -> tuple[object, ...]:
        return tuple({**dict(report), "rank": rank} for rank in range(4))  # type: ignore[arg-type]

    monkeypatch.setattr(
        cache_workflow, "load_rank_eval_micro_steps_from_cache", load_rank_eval
    )
    micro_steps, reduction_mode, pack_count = (
        cache_workflow._hydrate_eval_micro_steps_from_cache(
            {
                "cache_dir": tmp_path / "eval-cache",
                "fingerprint": "eval-fp",
                "micro_step_count": 1,
            },
            cache_root=tmp_path / "cache-root",
            rank=3,
            world_size=4,
            rank_report_gatherer=gather,
            receipt_sink=lambda receipt: receipts.append(dict(receipt)),
        )
    )

    assert micro_steps == (step,)
    assert reduction_mode == "replicated"
    assert pack_count == 1
    assert selective_loads == [
        {
            "path": tmp_path / "eval-cache",
            "cache_root": tmp_path / "cache-root",
            "fingerprint": "eval-fp",
            "rank": 0,
            "world_size": 1,
        }
    ]
    assert receipts[0]["rank_resources"]["world_size"] == 4
    details = receipts[0]["rank_details"]
    assert details["3"]["canonical_ordinal_total_count"] == 1
    assert details["3"]["canonical_ordinal_assigned_count"] == 1
    assert details["3"]["selective_decode"]["decoded_micro_step_count"] == 1
    assert details["3"]["selective_decode"]["payload_bytes_read"] == {
        "status": "unavailable",
        "reason": "selective_loader_byte_counter_not_exposed",
    }


def test_eval_hydration_reuses_the_model_free_reduction_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", "replicated")
    selective_loads: list[tuple[int, int]] = []
    monkeypatch.setattr(
        cache_workflow,
        "load_rank_eval_micro_steps_from_cache",
        lambda *args, **kwargs: (
            selective_loads.append((kwargs["rank"], kwargs["world_size"]))
            or SimpleNamespace(
                micro_steps=(SimpleNamespace(),),
                canonical_ordinals=(0,),
                total_ordinal_count=2,
            )
        ),
    )

    _, reduction_mode, pack_count = cache_workflow._hydrate_eval_micro_steps_from_cache(
        {
            "cache_dir": tmp_path / "eval-cache",
            "fingerprint": "eval-fp",
            "micro_step_count": 2,
        },
        cache_root=tmp_path / "cache-root",
        rank=0,
        world_size=2,
        rank_report_gatherer=lambda report: (
            report,
            {**dict(report), "rank": 1},
        ),
        reduction_receipt={
            "control": "auto",
            "effective_mode": "disjoint_shard",
            "source": "default",
            "pack_count": 2,
            "world_size": 2,
        },
    )

    assert reduction_mode == "disjoint_shard"
    assert pack_count == 2
    assert selective_loads == [(0, 2)]


def test_eval_hydration_rejects_noncanonical_selective_ordinals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", raising=False)
    monkeypatch.setattr(
        cache_workflow,
        "load_rank_eval_micro_steps_from_cache",
        lambda *args, **kwargs: SimpleNamespace(
            micro_steps=(SimpleNamespace(),),
            canonical_ordinals=(1,),
            total_ordinal_count=1,
        ),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        cache_workflow._hydrate_eval_micro_steps_from_cache(
            {
                "cache_dir": tmp_path / "eval-cache",
                "fingerprint": "eval-fp",
                "micro_step_count": 1,
            },
            cache_root=tmp_path / "cache-root",
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
        )

    assert exc_info.value.code == "training.eval_hydration_ordinal_mismatch"


def test_eval_hydration_selective_failure_has_no_full_loader_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", raising=False)
    full_loads: list[object] = []

    def fail_selective(*args: object, **kwargs: object) -> object:
        raise cache_workflow.PackingCacheInvalidError("corrupt assigned eval shard")

    def gather(report: object) -> tuple[object, object]:
        peer = {
            **dict(report),  # type: ignore[arg-type]
            "rank": 1,
            "status": "completed",
            "error_type": None,
            "error_code": None,
        }
        return report, peer

    monkeypatch.setattr(
        cache_workflow, "load_rank_eval_micro_steps_from_cache", fail_selective
    )
    monkeypatch.setattr(
        cache_workflow,
        "load_all_micro_steps_from_cache",
        lambda *args, **kwargs: full_loads.append(args),
        raising=False,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        cache_workflow._hydrate_eval_micro_steps_from_cache(
            {
                "cache_dir": tmp_path / "eval-cache",
                "fingerprint": "eval-fp",
                "micro_step_count": 2,
            },
            cache_root=tmp_path / "cache-root",
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
        )

    assert exc_info.value.code == "runtime.distributed_phase_failed"
    assert exc_info.value.context["phase"] == "evaluation_hydration"
    assert exc_info.value.context["failed_ranks"] == [0]
    assert full_loads == []


def _mapped_native_receipt(
    *, driver_build_id: str = "driver-build"
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "cuda_initialized": True,
        "admitted": True,
        "mismatches": [],
        "components": {
            "libcuda": {
                "mapped_origin": "/usr/lib/libcuda.so.1",
                "build_id": driver_build_id,
            }
        },
        "mapped_cudnn_components": ["libcudnn"],
    }


def test_mapped_native_attestation_runs_after_model_cuda_use_and_persists_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    captured: list[dict[str, object]] = []

    class Model:
        def to(self, device: object) -> object:
            assert str(device) == "cuda:0"
            events.append("model_cuda_use")
            return self

    def attest(**kwargs: object) -> dict[str, object]:
        assert kwargs == {"provenance": {"schema_version": 3}}
        events.append("mapped_native_attestation")
        return _mapped_native_receipt()

    monkeypatch.setattr(
        pipeline,
        "require_mapped_native_execution_attestation",
        attest,
        raising=False,
    )

    observed = pipeline._move_model_and_resolve_mapped_native_execution(
        model=Model(),
        accelerator=SimpleNamespace(device="cuda:0"),
        provenance={"schema_version": 3},
        rank=0,
        world_size=1,
        rank_report_gatherer=None,
        receipt_sink=lambda receipt: captured.append(dict(receipt)),
    )

    assert events == ["model_cuda_use", "mapped_native_attestation"]
    assert observed == _mapped_native_receipt()
    assert captured[0]["rank_details"] == {
        "0": {
            "admission_status": "completed",
            "attestation": _mapped_native_receipt(),
        }
    }


def test_mapped_native_attestation_failure_is_persisted_before_optimizer_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []
    rejected = {
        **_mapped_native_receipt(),
        "admitted": False,
        "mismatches": ["components.libcuda.build_id"],
    }

    class Rejected(RuntimeError):
        def __init__(self) -> None:
            super().__init__("native execution rejected")
            self.result = rejected

    monkeypatch.setattr(
        pipeline,
        "require_mapped_native_execution_attestation",
        lambda **kwargs: (_ for _ in ()).throw(Rejected()),
        raising=False,
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._move_model_and_resolve_mapped_native_execution(
            model=SimpleNamespace(to=lambda device: None),
            accelerator=SimpleNamespace(device="cuda:0"),
            provenance={"schema_version": 3},
            rank=0,
            world_size=1,
            rank_report_gatherer=None,
            receipt_sink=lambda receipt: captured.append(dict(receipt)),
        )

    assert exc_info.value.code == "runtime.mapped_native_execution_rejected"
    assert captured[0]["rank_details"] == {
        "0": {
            "admission_status": "failed",
            "attestation": rejected,
            "error_code": "runtime.mapped_native_execution_rejected",
            "error_type": "Rejected",
        }
    }


def test_mapped_native_attestation_requires_exact_rank_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "require_mapped_native_execution_attestation",
        lambda **kwargs: _mapped_native_receipt(),
        raising=False,
    )

    def gather(report: object) -> tuple[object, object]:
        peer = {**dict(report), "rank": 1}  # type: ignore[arg-type]
        peer_details = dict(peer["rank_details"])
        peer_details["attestation"] = _mapped_native_receipt(
            driver_build_id="different-driver-build"
        )
        peer["rank_details"] = peer_details
        return report, peer

    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline._move_model_and_resolve_mapped_native_execution(
            model=SimpleNamespace(to=lambda device: None),
            accelerator=SimpleNamespace(device="cuda:0"),
            provenance={"schema_version": 3},
            rank=0,
            world_size=2,
            rank_report_gatherer=gather,
            receipt_sink=lambda receipt: None,
        )

    assert exc_info.value.code == "runtime.mapped_native_execution_rank_mismatch"


@pytest.mark.parametrize("provider_mode", ("legacy_fused", "synchronous", "overlapped"))
def test_same_dataset_eval_resolves_rank_selective_cache_and_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_mode: str,
) -> None:
    monkeypatch.delenv("COORDEXP_SWIFT_EVAL_REDUCTION_MODE", raising=False)
    monkeypatch.delenv("COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE", raising=False)
    resolved_provider = pipeline.resolve_forward_input_provider_mode(provider_mode)
    monkeypatch.setattr(
        pipeline,
        "resolve_forward_input_provider_mode",
        lambda mode: pytest.fail("initialized training resolved provider mode late"),
    )
    dataset = SimpleNamespace(path=tmp_path / "shared.jsonl", sample_limit=2)
    dataset.path.write_text("{}\n")
    config = SimpleNamespace(
        runtime=SimpleNamespace(
            seed=7,
            determinism=SimpleNamespace(mode="legacy"),
        ),
        training=SimpleNamespace(precision="no", max_grad_norm=1.0),
        model=SimpleNamespace(
            special_token_embeddings=object(),
            attn_implementation="flash_attention_2",
            fa2_branch_proof="first_micro_step",
        ),
        adapter=object(),
        packing=_packing_config("source_order_next_fit").packing,
        template=SimpleNamespace(object_ordering="geo_sorted"),
        losses=object(),
        optimizer=object(),
        data=SimpleNamespace(
            train=dataset,
            eval=SimpleNamespace(
                path=dataset.path,
                sample_limit=2,
                model_dump=lambda **kwargs: {
                    "path": str(dataset.path),
                    "sample_limit": 2,
                },
            ),
            train_order="source_order",
        ),
        checkpoint=SimpleNamespace(save_final=False),
    )
    admission_order: list[str] = []

    class Model:
        def to(self, device: object) -> object:
            assert str(device) == "cuda:0"
            admission_order.append("model_cuda_use")
            return self

    components = SimpleNamespace(
        model=Model(),
        token_identity=SimpleNamespace(tokenizer_vocab_size=10),
        tokenizer=object(),
        base_model_path=tmp_path / "model",
        base_config_sha256="base",
        tokenizer_sha256="tokenizer",
        processor=SimpleNamespace(image_processor=object()),
    )
    train_shard = (SimpleNamespace(split="train"),)
    rank_eval_steps = (SimpleNamespace(split="eval"),)
    cache = {
        "cache_dir": tmp_path / "cache",
        "micro_step_count": 2,
        "format_version": "v1",
        "fingerprint": "fp",
        "determinants_sha256": "digest",
        "phase_receipt": {
            "cache_preparation": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_publication": {
                "status": "not_run_cache_hit",
                "duration_seconds": 0.0,
            },
            "cache_admission": {
                "status": "completed",
                "duration_seconds": 0.1,
            },
        },
    }
    eval_cache = {
        **cache,
        "cache_dir": tmp_path / "eval-cache",
        "fingerprint": "eval-fp",
        "determinants_sha256": "eval-digest",
    }
    schedule = SimpleNamespace(resolved_max_steps=1, runtime_batch=object())
    bindings: list[tuple[str, dict[str, object]]] = []
    provider_modes: list[str] = []
    phase_events: list[tuple[str, str]] = []
    phase_summaries: list[tuple[str, dict[str, object]]] = []
    policy_identities: list[tuple[str, dict[str, object]]] = []
    writer = SimpleNamespace(
        bind_schedule=lambda **kwargs: None,
        bind_materialization=lambda split, **kwargs: bindings.append((split, kwargs)),
        bind_forward_input_provider_mode=lambda mode, **kwargs: provider_modes.append(
            mode
        ),
        bind_policy_identity=lambda name, identity: policy_identities.append(
            (name, dict(identity))
        ),
        begin_phase=lambda phase, **kwargs: phase_events.append(("begin", phase)),
        finish_phase=lambda phase, **kwargs: phase_events.append(("finish", phase)),
        record_phase_not_run=lambda phase, **kwargs: phase_events.append(
            ("not_run", phase)
        ),
        record_completed_phase=lambda phase, **kwargs: phase_events.append(
            ("completed", phase)
        ),
        record_phase_summary=lambda phase, **kwargs: phase_summaries.append(
            (phase, kwargs)
        ),
        finalize=lambda **kwargs: None,
    )
    selective_loads: list[dict[str, object]] = []
    train_verification_levels: list[str] = []
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=2,
        process_index=0,
        device="cuda:0",
    )

    monkeypatch.setattr(cache_workflow, "seed_training_runtime", lambda *args, **kwargs: None)
    _patch_shared_cache_import(
        monkeypatch, "load_qwen_components", lambda *args, **kwargs: components
    )
    _patch_shared_cache_import(
        monkeypatch, "resolve_qwen_runtime_controls", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline, "load_default_adapter_source_gate_evidence", lambda root: object()
    )
    monkeypatch.setattr(
        pipeline,
        "build_adapter_setup_plan",
        lambda *args, **kwargs: SimpleNamespace(mode="fresh"),
    )
    monkeypatch.setattr(
        pipeline,
        "setup_dora_adapter",
        lambda model, plan: SimpleNamespace(
            model=model, receipt=SimpleNamespace(adapter_name="default")
        ),
    )
    monkeypatch.setattr(
        pipeline, "build_default_special_token_selection", lambda *args: object()
    )
    monkeypatch.setattr(
        pipeline,
        "load_default_special_token_embedding_source_gate_evidence",
        lambda root: object(),
    )
    monkeypatch.setattr(
        pipeline,
        "install_special_token_embedding_deltas",
        lambda model, selection, source_gate: SimpleNamespace(
            model=model, receipt=object()
        ),
    )
    monkeypatch.setattr(pipeline, "enable_training_memory_savers", lambda model: None)
    monkeypatch.setattr(
        pipeline,
        "require_mapped_native_execution_attestation",
        lambda **kwargs: admission_order.append("mapped_native_attestation")
        or _mapped_native_receipt(),
        raising=False,
    )
    _patch_shared_cache_import(
        monkeypatch, "build_token_vocabulary_groups", lambda *args, **kwargs: object()
    )

    def resolve_train_cache(*args: object, **kwargs: object) -> dict[str, object]:
        train_verification_levels.append(str(kwargs["verification_level"]))
        return cache

    monkeypatch.setattr(
        cache_workflow, "_resolve_or_build_train_pack_cache", resolve_train_cache
    )
    _patch_shared_cache_import(
        monkeypatch, "resolve_planned_step_schedule", lambda *args, **kwargs: schedule
    )
    _patch_shared_cache_import(
        monkeypatch,
        "load_rank_micro_steps_from_cache",
        lambda *args, **kwargs: train_shard,
    )
    monkeypatch.setattr(
        cache_workflow,
        "_attach_image_processors_to_micro_steps",
        lambda steps, **kwargs: tuple(steps),
    )
    monkeypatch.setattr(
        cache_workflow, "_apply_fa2_branch_proof_policy", lambda steps, config: tuple(steps)
    )
    monkeypatch.setattr(pipeline.LossRunner, "from_config", lambda config: object())
    monkeypatch.setattr(
        pipeline,
        "build_optimizer_group_plan",
        lambda *args, **kwargs: admission_order.append("optimizer_admission")
        or object(),
    )
    monkeypatch.setattr(
        pipeline, "build_scheduler_plan", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline,
        "build_optimizer_and_scheduler",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        pipeline, "build_trainable_surface_receipt", lambda *args, **kwargs: object()
    )
    closed: list[str] = []
    runtime_kwargs: list[dict[str, object]] = []
    consensus_calls: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        model=object(),
        accelerator=accelerator,
        is_main_process=True,
        world_size=2,
        validate_eval_reduction_consensus=lambda **kwargs: consensus_calls.append(
            kwargs
        ),
    )

    def gather(report: object) -> tuple[object, object]:
        peer = {**dict(report), "rank": 1}  # type: ignore[arg-type]
        return report, peer

    gather.close = lambda: closed.append("rank-report")  # type: ignore[attr-defined]

    def build_runtime(**kwargs: object) -> object:
        runtime_kwargs.append(kwargs)
        runtime.rank_report_gatherer = kwargs["rank_report_gatherer"]
        return runtime

    monkeypatch.setattr(pipeline, "TrainRuntime", build_runtime)
    monkeypatch.setattr(
        cache_workflow, "_resolve_eval_pack_cache", lambda *args, **kwargs: eval_cache
    )

    def load_rank_eval(
        path: Path,
        *,
        cache_root: Path,
        expected_fingerprint: str,
        rank: int,
        world_size: int,
    ) -> object:
        assert cache_root == tmp_path / ".cache" / "coordexp_swift" / "packing"
        selective_loads.append(
            {
                "path": path,
                "fingerprint": expected_fingerprint,
                "rank": rank,
                "world_size": world_size,
            }
        )
        return SimpleNamespace(
            micro_steps=rank_eval_steps,
            canonical_ordinals=(0,),
            total_ordinal_count=2,
        )

    monkeypatch.setattr(
        cache_workflow, "load_rank_eval_micro_steps_from_cache", load_rank_eval
    )
    monkeypatch.setattr(
        cache_workflow,
        "load_all_micro_steps_from_cache",
        lambda *args, **kwargs: pytest.fail("production used the full eval loader"),
        raising=False,
    )
    monkeypatch.setattr(
        cache_workflow,
        "partition_eval_micro_steps_for_rank",
        lambda *args, **kwargs: pytest.fail("selective eval was partitioned twice"),
        raising=False,
    )
    monkeypatch.setattr(pipeline, "CheckpointWriter", lambda run_dir: object())
    result = SimpleNamespace(
        completed_steps=1,
        consumed_micro_steps=1,
        scheduled_event_counts={},
        latest_observation=None,
    )
    monkeypatch.setattr(
        pipeline,
        "SupervisedTrainer",
        lambda **kwargs: SimpleNamespace(run=lambda: result),
    )

    lifecycle = {
        "completed_steps": 0,
        "consumed_packs": 0,
        "checkpoint_event_count": 0,
        "optimizer_update_status": None,
        "finite_status": None,
    }
    pipeline._run_initialized_training(
        repo_root=tmp_path,
        resolved_config=SimpleNamespace(
            entry_config_path=tmp_path / "config.yaml", fingerprint="config-fp"
        ),
        config=config,
        accelerator=accelerator,
        run_directory=RunDirectory("run", tmp_path, tmp_path / "run", "created"),
        run_id="run",
        run_segment_id="segment-run",
        writer=writer,
        lifecycle=lifecycle,
        rank_report_gatherer=gather,
        resolved_forward_input_provider=resolved_provider,
        provenance={"schema_version": 3},
    )

    assert selective_loads == [
        {
            "path": eval_cache["cache_dir"],
            "fingerprint": "eval-fp",
            "rank": 0,
            "world_size": 2,
        }
    ]
    assert train_verification_levels == ["manifest"]
    assert rank_eval_steps != train_shard
    assert [split for split, _ in bindings] == ["train", "eval"]
    assert bindings[0][1]["semantic_fingerprint"] == "fp"
    assert bindings[1][1]["semantic_fingerprint"] == "eval-fp"
    assert closed == []
    assert runtime_kwargs[0]["rank_report_gatherer"] is gather
    assert set(bindings[1][1]) == {
        "cache_format_version",
        "semantic_fingerprint",
        "determinant_digest",
    }
    assert provider_modes == [provider_mode]
    assert consensus_calls == [{"reduction_mode": "disjoint_shard", "pack_count": 2}]
    identities = dict(policy_identities)
    assert identities["cache"]["root"] == {
        "resolved_root": str(
            (tmp_path / ".cache" / "coordexp_swift" / "packing").resolve()
        ),
        "source": "default",
    }
    assert identities["eval_reduction"] == {
        "schema_version": 1,
        "control": "auto",
        "effective_mode": "disjoint_shard",
        "source": "default",
        "pack_count": 2,
        "world_size": 2,
    }
    assert identities["input_provider"] == {
        "schema_version": 1,
        "mode": provider_mode,
        **resolved_provider.to_receipt_dict(),
    }
    assert ("begin", "optimizer_runtime_assembly") in phase_events
    assert ("finish", "optimizer_runtime_assembly") in phase_events
    assert admission_order[:3] == [
        "model_cuda_use",
        "mapped_native_attestation",
        "optimizer_admission",
    ]
    model_loading_details = lifecycle["phase_rank_receipts"]["model_loading"]
    assert model_loading_details["rank_details"] == {
        "0": {
            "admission_status": "completed",
            "attestation": _mapped_native_receipt(),
        },
        "1": {
            "admission_status": "completed",
            "attestation": _mapped_native_receipt(),
        },
    }


def test_three_provider_modes_have_exact_cpu_autograd_and_adam_state_equivalence() -> (
    None
):
    image_token_id = 151655
    micro_steps: list[SupervisedMicroStep] = []
    for index in range(2):
        image_encoding = SimpleNamespace(
            image_grid_thw=(1, 4, 6),
            pixel_values=torch.full((24, 8), float(index + 1)),
            plan=SimpleNamespace(merge_size=2),
        )
        encoded = SimpleNamespace(
            example_id=f"optimizer-equivalence-{index}",
            input_ids=(10, 11, 12, *([image_token_id] * 6), 13, 14),
            image_pad_physical_start=3,
            image_pad_physical_end=9,
            image_encoding=image_encoding,
        )
        pack = plan_packed_sequences((encoded,), global_max_length=32)[0]
        micro_steps.append(
            SupervisedMicroStep(
                pack=pack,
                encoded_examples=(encoded,),
                position_inputs=build_qwen_position_inputs(pack, (encoded,)),
                # A real atom-free TokenSequence selects no causal logit
                # positions, exactly like the duck-typed stub this replaced when
                # selection moved onto TokenSequence (design decision 6).
                token_sequence=TokenSequence(
                    pack_index=pack.pack_index,
                    input_ids=pack.input_ids,
                    segments=pack.segments,
                    atoms=(),
                    spans=(),
                ),
                vocab_groups=None,
            )
        )

    def run_mode(mode: str) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        torch.manual_seed(91)
        model = torch.nn.Linear(2, 1, bias=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        provider = build_forward_input_provider(mode)
        try:
            if provider is not None:
                provider.begin_planned_step(1, tuple(micro_steps))
            for ordinal, micro_step in enumerate(micro_steps):
                forward_inputs = (
                    build_qwen_forward_inputs(
                        micro_step.pack,
                        micro_step.encoded_examples,
                        micro_step.position_inputs,
                    )
                    if provider is None
                    else provider.take(ordinal, micro_step)
                )
                features = torch.stack(
                    (
                        forward_inputs.input_ids.float().sum(),
                        forward_inputs.pixel_values.float().sum(),
                    )
                ).unsqueeze(0)
                loss = torch.nn.functional.mse_loss(
                    model(features), torch.tensor([[float(ordinal + 1)]])
                )
                loss.backward()
            optimizer.step()
            optimizer_state = optimizer.state_dict()
            return (
                {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                },
                optimizer_state,
            )
        finally:
            if provider is not None:
                provider.end_planned_step()
                provider.close()

    results = {
        mode: run_mode(mode) for mode in ("legacy_fused", "synchronous", "overlapped")
    }
    reference_parameters, reference_optimizer = results["legacy_fused"]
    for mode in ("synchronous", "overlapped"):
        parameters, optimizer_state = results[mode]
        assert parameters.keys() == reference_parameters.keys()
        for name in parameters:
            assert torch.equal(parameters[name], reference_parameters[name])
        assert optimizer_state["param_groups"] == reference_optimizer["param_groups"]
        assert optimizer_state["state"].keys() == reference_optimizer["state"].keys()
        for parameter_id, state in optimizer_state["state"].items():
            reference_state = reference_optimizer["state"][parameter_id]
            assert state.keys() == reference_state.keys()
            for name, value in state.items():
                reference_value = reference_state[name]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, reference_value)
                else:
                    assert value == reference_value


def _fake_micro_step() -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack="pack",
        encoded_examples=(),
        position_inputs="positions",
        token_sequence="tokens",
        vocab_groups="vocab",
        fa2_branch_evidence={"stale": True},
        capture_fa2_branch=False,
        require_fa2_branch_proof=False,
        fa2_branch_proof_policy=None,
    )


def test_fa2_branch_proof_policy_first_micro_step_captures_only_first_step() -> None:
    steps = (_fake_micro_step(), _fake_micro_step(), _fake_micro_step())
    config = SimpleNamespace(model=SimpleNamespace(fa2_branch_proof="first_micro_step"))

    configured = cache_workflow._apply_fa2_branch_proof_policy(steps, config)

    assert [step.capture_fa2_branch for step in configured] == [True, False, False]
    assert [step.require_fa2_branch_proof for step in configured] == [
        True,
        False,
        False,
    ]
    assert all(
        step.fa2_branch_proof_policy == "first_micro_step" for step in configured
    )
    assert all(step.fa2_branch_evidence is None for step in configured)


def test_fa2_branch_proof_policy_every_forward_captures_every_step() -> None:
    steps = (_fake_micro_step(), _fake_micro_step())
    config = SimpleNamespace(model=SimpleNamespace(fa2_branch_proof="every_forward"))

    configured = cache_workflow._apply_fa2_branch_proof_policy(steps, config)

    assert [step.capture_fa2_branch for step in configured] == [True, True]
    assert [step.require_fa2_branch_proof for step in configured] == [True, True]
    assert all(step.fa2_branch_proof_policy == "every_forward" for step in configured)


def _real_loss_context() -> LossContext:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    segment = PackedSegment(
        pack_index=0,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        start=0,
        end=2,
    )
    atom = TokenAtom(
        pack_index=0,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        target_position=1,
        token_id=7,
        token_type="desc_text",
        text="x",
        logical_target_position=1,
    )
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=0,
            input_ids=(0, 0),
            segments=(segment,),
            atoms=(atom,),
            spans=(),
        ),
        vocab_groups=TokenVocabularyGroups(
            vocab_size=8,
            desc_text=(7,),
            schema=(1, 2),
            coordinate=(3, 4),
            eos=(5,),
            blocked=(0, 6),
        ),
        logits_position_ids=None,
    )


def test_train_logging_persists_real_loss_runner_accuracy_stats(
    tmp_path: Path,
) -> None:
    # Production wiring, end to end with the real LossRunner (no fakes):
    # LossRunner.finalize_planned_step -> CompletedStepObservation ->
    # _train_logging_handler -> runtime.gather_metrics. Proves accuracy_stats
    # is forwarded as the exact integers LossRunner computed and persisted
    # as one strict nested JSON structure in the durable logging row.
    context = _real_loss_context()
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.1,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )
    plan = runner.prepare_planned_step((context.token_sequence,))
    micro_bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    artifact = runner.finalize_planned_step((micro_bundle.to_artifact_dict(),), plan)

    assert artifact["accuracy_stats"] == {
        "top1_correct": 1,
        "top5_correct": 1,
        "atom_count": 1,
    }

    writer = _writer(tmp_path)
    calls: list[dict[str, object]] = []

    class Runtime(_Runtime):
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            calls.append({"metrics": dict(metrics), **kwargs})
            return {
                "metrics": dict(metrics),
                "accuracy_stats": dict(kwargs["accuracy_stats"]),
            }

    observation = CompletedStepObservation(
        planned_step_id=1,
        micro_step_count=1,
        loss_bundle_artifact=artifact,
        optimizer_update_status="applied",
        finite_status="finite",
    )
    reporting.CompletedStepReporter(writer=writer, lifecycle={}, runtime=Runtime())(observation)

    assert calls[0]["accuracy_stats"] == artifact["accuracy_stats"]
    row = json.loads(writer.logging_path.read_text())
    assert row["acc_top1"] == pytest.approx(1.0)
    assert row["accuracy_stats"] == artifact["accuracy_stats"]
    assert "top1_correct" not in row
    assert "top5_correct" not in row
    assert "atom_count" not in row


def test_train_row_key_set_gains_exactly_the_three_timing_keys_and_keeps_accuracy_stats(
    tmp_path: Path,
) -> None:
    """Frozen key-set proof, driven through the real production handler and
    a real `LossRunner` artifact (not a hardcoded literal key list, which
    would be brittle against loss-runner-owned metric names the row schema
    contract does not itself own): capture the row's key set with timing
    fields unmeasured (the pre-existing baseline this row producer already
    wrote before task 1.3), then again with real timing values supplied,
    and assert the ONLY difference is the three additive timing keys -- no
    other key appears, disappears, or is renamed, and `accuracy_stats`
    remains the same strict nested structure in both rows.
    """

    context = _real_loss_context()
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.1,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )
    plan = runner.prepare_planned_step((context.token_sequence,))
    micro_bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    artifact = runner.finalize_planned_step((micro_bundle.to_artifact_dict(),), plan)

    class Runtime(_Runtime):
        def gather_metrics(self, metrics: object, **kwargs: object) -> object:
            return {
                "metrics": dict(metrics),
                "accuracy_stats": dict(kwargs["accuracy_stats"]),
            }  # type: ignore[arg-type]

    baseline_writer = _writer(tmp_path / "baseline")
    baseline_observation = CompletedStepObservation(
        planned_step_id=1,
        micro_step_count=1,
        loss_bundle_artifact=artifact,
        optimizer_update_status="applied",
        finite_status="finite",
        # step_duration_seconds/input_build_seconds/input_wait_seconds left
        # at their default (unmeasured, `None`) -- the pre-timing-fields
        # baseline shape this row producer already wrote.
    )
    reporting.CompletedStepReporter(
        writer=baseline_writer, lifecycle={}, runtime=Runtime()
    )(baseline_observation)
    baseline_row = json.loads(baseline_writer.logging_path.read_text())

    timed_writer = _writer(tmp_path / "timed")
    timed_observation = CompletedStepObservation(
        planned_step_id=1,
        micro_step_count=1,
        loss_bundle_artifact=artifact,
        optimizer_update_status="applied",
        finite_status="finite",
        step_duration_seconds=0.42,
        input_build_seconds=0.11,
        input_wait_seconds=0.03,
    )
    reporting.CompletedStepReporter(writer=timed_writer, lifecycle={}, runtime=Runtime())(
        timed_observation
    )
    timed_row = json.loads(timed_writer.logging_path.read_text())

    timing_keys = {"step_duration_seconds", "input_build_seconds", "input_wait_seconds"}
    assert timing_keys.isdisjoint(baseline_row)
    assert set(timed_row) == set(baseline_row) | timing_keys
    for key in baseline_row:
        assert baseline_row[key] == timed_row[key]
    assert baseline_row["accuracy_stats"] == artifact["accuracy_stats"]
    assert timed_row["accuracy_stats"] == artifact["accuracy_stats"]


# ---------------------------------------------------------------------------
# Wave-0 pre-move characterization for
# `decompose-coordexp-swift-training-orchestration`.
#
# `src/training/pipeline.py` is the facade Waves 2-5 reduce.  These additions
# freeze its public contract and record which private helpers it still owns at
# the baseline, so a later wave that moves one of them must delete it here rather
# than leave a forwarding layer behind.
# ---------------------------------------------------------------------------


WAVE0_ORCHESTRATION_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "training_orchestration"
)

WAVE0_FACADE_RESULT_KEYS = (
    "completed_steps",
    "consumed_micro_steps",
    "resolved_config_fingerprint",
    "run_dir",
    "run_id",
    "scheduled_event_counts",
)

WAVE0_TRAIN_ROW_KEYS = (
    "acc_top1",
    "acc_top5",
    "accuracy_stats",
    "finite_status",
    "input_build_seconds",
    "input_wait_seconds",
    "loss/total",
    "lr/group_0",
    "micro_step_count",
    "non_finite_fields",
    "optimizer_update_status",
    "split",
    "step",
    "step_duration_seconds",
)

#: Private helpers `pipeline.py` owned at Wave 0, with the wave that takes each
#: away and the module that owns it once that wave has landed.  ``None`` means
#: the helper is still owned by ``pipeline.py``; a module means the declared
#: wave moved it and ``pipeline.py`` must no longer expose it, because a
#: forwarding layer would keep the historical owner alive by another name.
#:
#: Wave 4 declares two flips (manifest wave-4 ``declared_flips_in_scope``):
#: ``_append_logging_row_shared`` moves to ``reporting`` under its historical
#: name (both the train and eval callbacks call it, so it survives as a
#: shared module-level function). ``_train_logging_handler`` has no entry
#: here any more: design decision 9 replaces the factory function with
#: ``reporting.CompletedStepReporter``, an architecturally different symbol,
#: so there is no same-named successor to assert against; the parametrized
#: node for that historical key no longer collects, which is the declared
#: flip landing rather than a boundary regression.
WAVE0_PIPELINE_OWNED_HELPERS: dict[str, tuple[int, object | None]] = {
    "_build_model_free_preflight_gatherer": (2, control_plane),
    "_build_rank_report_gatherer": (2, control_plane),
    "_run_rank_converged_phase": (2, control_plane),
    "_validate_phase_status_reports": (2, control_plane),
    "_normalize_bounded_phase_details": (2, control_plane),
    "_all_gather_cpu_bytes": (2, control_plane),
    "_resolve_model_free_launch_identity": (2, execution_plan),
    "_admit_model_free_pack_cache": (3, cache_workflow),
    "_resolve_model_free_training_preflight": (3, cache_workflow),
    "_resolve_or_build_pack_cache": (3, cache_workflow),
    "_resolve_or_build_train_pack_cache": (3, cache_workflow),
    "_resolve_eval_pack_cache": (3, cache_workflow),
    "_hydrate_eval_micro_steps_from_cache": (3, cache_workflow),
    "_pack_cache_preparation_receipt": (3, cache_workflow),
    "_aggregate_cache_phase": (3, cache_workflow),
    "_append_logging_row_shared": (4, reporting),
    "_run_initialized_training": (5, None),
    "_checkpoint_handler": (5, None),
    "_eval_forward_handler": (5, None),
    "_final_handler": (5, None),
}


def test_wave0_facade_signature_and_compatibility_reexport_are_frozen() -> None:
    import inspect

    signature = inspect.signature(pipeline.run_training_pipeline)

    assert list(signature.parameters) == ["config_path", "measurement_context"]
    assert (
        signature.parameters["measurement_context"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["measurement_context"].default is None
    assert callable(pipeline.prepare_training_pack_caches)
    assert pipeline.prepare_training_pack_caches.__module__ == "src.training.pipeline"


def test_wave0_facade_result_key_set_matches_the_frozen_fixture() -> None:
    frozen = json.loads(
        WAVE0_ORCHESTRATION_FIXTURE_ROOT.joinpath("pipeline_result.json").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(sorted(frozen)) == WAVE0_FACADE_RESULT_KEYS
    assert frozen["scheduled_event_counts"] == {"checkpoint": 0, "eval": 0}


@pytest.mark.parametrize(
    ("helper", "ownership"),
    sorted(WAVE0_PIPELINE_OWNED_HELPERS.items()),
    ids=sorted(WAVE0_PIPELINE_OWNED_HELPERS),
)
def test_wave0_pipeline_still_owns_helper_until_its_declared_wave(
    helper: str, ownership: tuple[int, object | None]
) -> None:
    clears_at_wave, new_owner = ownership
    if new_owner is None:
        assert hasattr(pipeline, helper), (
            f"pipeline.{helper} is owned by src/training/pipeline.py until wave "
            f"{clears_at_wave} moves it; a wave that moves it must delete it here "
            "and revise this node under the frozen manifest's revision rule"
        )
        return
    owner_name = getattr(new_owner, "__name__", str(new_owner))
    assert hasattr(new_owner, helper), (
        f"wave {clears_at_wave} moved {helper} to {owner_name}, which must own it"
    )
    assert not hasattr(pipeline, helper), (
        f"wave {clears_at_wave} moved {helper} to {owner_name}; "
        "src/training/pipeline.py must delete it rather than forward through it"
    )


def test_wave0_completed_step_row_key_set_matches_the_frozen_fixture(
    tmp_path: Path,
) -> None:
    frozen = json.loads(
        WAVE0_ORCHESTRATION_FIXTURE_ROOT.joinpath(
            "completed_step_rows.json"
        ).read_text(encoding="utf-8")
    )
    writer = _writer(tmp_path)
    lifecycle: dict[str, object] = {"consumed_packs": 0}
    handle = reporting.CompletedStepReporter(writer=writer, lifecycle=lifecycle, runtime=_Runtime())

    handle(_observation(1))

    rows = [
        json.loads(line)
        for line in writer.logging_path.read_text(encoding="utf-8").splitlines()
    ]
    assert tuple(sorted(frozen["rows"][0])) == WAVE0_TRAIN_ROW_KEYS
    assert set(rows[0]).issubset(set(WAVE0_TRAIN_ROW_KEYS))
    assert rows[0]["split"] == "train"
    assert rows[0]["non_finite_fields"] == []
