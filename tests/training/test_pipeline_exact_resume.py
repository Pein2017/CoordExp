from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import src.artifacts.run_writer as run_writer_module
from src.artifacts.checkpoints import CheckpointWriteResult
from src.artifacts.run_writer import RunWriter
from src.artifacts.training_state import TrainingStatePublicationPlan
from src.common.errors import RuntimeContractError
from src.training import pipeline, reporting
from src.training.exact_resume import RankCudaDeviceBinding
from src.training.pipeline import (
    _apply_exact_resume_cursor_state,
    _build_exact_resume_publication_plan,
    _build_pipeline_exact_resume_identities,
    _exact_resume_dependency_identity,
    _exact_resume_policy_payload,
    _read_only_admit_pipeline_exact_resume,
    _restored_exact_resume_cursor_state,
)
from src.training.supervised_trainer import CompletedStepObservation


@dataclass(frozen=True)
class _Segment:
    example_id: str


@dataclass(frozen=True)
class _Pack:
    pack_index: int
    input_ids: tuple[int, ...]
    segments: tuple[_Segment, ...]


@dataclass(frozen=True)
class _MicroStep:
    pack: _Pack


_MISSING = object()


def _decoded_rank(
    cursor: dict[str, Any],
    *,
    next_rank_local_micro_step: Any = 4,
) -> SimpleNamespace:
    cursor_envelope = {
        "data": {
            "next_rank_local_micro_step": 4,
            "owner": "data",
            "schema": "coordexp-swift-data-cursor",
            "schema_version": 1,
            "state": cursor["data"],
        },
        "pack": {
            "next_rank_local_micro_step": 4,
            "owner": "pack",
            "schema": "coordexp-swift-pack-cursor",
            "schema_version": 1,
            "state": cursor["pack"],
        },
        "schema": "coordexp-swift-exact-rank-cursor",
        "schema_version": 1,
    }
    if next_rank_local_micro_step is not _MISSING:
        cursor_envelope["next_rank_local_micro_step"] = next_rank_local_micro_step
    return SimpleNamespace(
        cursor=cursor_envelope,
        cuda_device_topology=("cuda:0",),
        cuda_device_count=1,
    )


def _micro_steps(count: int) -> tuple[_MicroStep, ...]:
    return tuple(
        _MicroStep(
            pack=_Pack(
                pack_index=index,
                input_ids=(index, index + 1),
                segments=(_Segment(example_id=f"example-{index}"),),
            )
        )
        for index in range(count)
    )


def _schedule(*, max_steps: int = 4, grad_accum: int = 2) -> Any:
    return SimpleNamespace(
        resolved_max_steps=max_steps,
        runtime_batch=SimpleNamespace(resolved_grad_accum_steps=grad_accum),
    )


def _record_exact_publication(
    writer: RunWriter,
    *,
    checkpoint_dir: Path,
    checkpoint_step: int,
    manifest_bytes: bytes,
    aggregate_digest: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    payload_identity = {
        "schema": "coordexp-swift-inference-checkpoint-payload-publication",
        "schema_version": 2,
        "manifest_relative_path": "inference_payload_manifest.json",
        "manifest_file_sha256": "c" * 64,
        "aggregate_digest": "d" * 64,
    }
    monkeypatch.setattr(
        run_writer_module,
        "admit_inference_checkpoint_payload_identity",
        lambda _checkpoint_dir, _expected: dict(payload_identity),
    )
    writer.record_checkpoint_publication_event(
        step=checkpoint_step,
        status="completed",
        started_at="2026-08-11T00:00:00+00:00",
        completed_at="2026-08-11T00:00:01+00:00",
        duration_seconds=1.0,
        is_final=False,
        exact_training_state_enabled=True,
        checkpoint_identity={
            "checkpoint_step": checkpoint_step,
            "resolved_path": str(checkpoint_dir.resolve()),
            "training_state_manifest_file_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "training_state_aggregate_digest": aggregate_digest,
        },
        inference_payload_identity=payload_identity,
        committed_progress={
            "schema": "coordexp-swift-checkpoint-committed-progress",
            "schema_version": 1,
            "completed_steps": checkpoint_step,
            "consumed_packs": checkpoint_step * 2,
            "optimizer_update_status": "applied",
            "finite_status": "finite",
        },
        failure_code=None,
    )
    return payload_identity


def test_exact_resume_policy_identity_excludes_resume_selection() -> None:
    common = {
        "packing": {"algorithm": "source_order_next_fit-v1"},
        "input_provider": {"mode": "synchronous"},
        "attention": {"implementation": "flash_attention_2"},
        "profile_sync": {"enabled": False},
        "eval_reduction": {"mode": "disjoint_shard"},
    }

    disabled = _exact_resume_policy_payload(
        **common,
        resume={"mode": "disabled", "checkpoint_dir": None},
    )
    continuation = _exact_resume_policy_payload(
        **common,
        resume={"mode": "exact_same_world_size", "checkpoint_dir": "/parent"},
    )

    assert disabled == continuation
    assert "resume" not in disabled


def test_train_logging_persists_validated_global_integer_accuracy_stats(
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=8,
        resolved_max_steps=5,
    )
    global_accuracy_stats = {
        "top1_correct": 28,
        "top5_correct": 36,
        "atom_count": 44,
    }

    class Runtime:
        world_size = 1
        is_main_process = True
        accelerator = SimpleNamespace(is_main_process=True, num_processes=1)

        def gather_metrics(self, metrics: Any, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["accuracy_stats"] == {
                "top1_correct": 3,
                "top5_correct": 4,
                "atom_count": 5,
            }
            return {
                "metrics": dict(metrics),
                "accuracy_stats": dict(global_accuracy_stats),
            }

    observation = CompletedStepObservation(
        planned_step_id=3,
        micro_step_count=3,
        loss_bundle_artifact={
            "metrics": {"acc_top1": 3 / 5, "acc_top5": 4 / 5},
            "accuracy_stats": {
                "top1_correct": 3,
                "top5_correct": 4,
                "atom_count": 5,
            },
        },
        optimizer_update_status="applied",
        finite_status="finite",
    )

    reporting.CompletedStepReporter(writer=writer, lifecycle={}, runtime=Runtime())(
        observation
    )

    row = json.loads(writer.logging_path.read_text())
    assert row["accuracy_stats"] == global_accuracy_stats
    assert all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in row["accuracy_stats"].values()
    )
    assert json.loads(json.dumps(row)) == row


def test_checkpoint_handler_persists_step3_and_final_exact_publication_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=5,
    )
    manifest_bytes_by_step = {
        3: b'{"checkpoint_step":3}\n',
        5: b'{"checkpoint_step":5}\n',
    }
    aggregate_by_step = {3: "3" * 64, 5: "5" * 64}
    payload_identity_by_step = {
        step: {
            "schema": "coordexp-swift-inference-checkpoint-payload-publication",
            "schema_version": 2,
            "manifest_relative_path": "inference_payload_manifest.json",
            "manifest_file_sha256": str(step) * 64,
            "aggregate_digest": hex(step)[2:] * 64,
        }
        for step in (3, 5)
    }

    class CheckpointWriter:
        def write_checkpoint(self, **kwargs: Any) -> CheckpointWriteResult:
            step = int(kwargs["step"])
            checkpoint_dir = writer.checkpoints_dir / f"step-{step}"
            checkpoint_dir.mkdir(parents=True)
            callback = kwargs["exact_training_state_callback"]
            assert callable(callback)
            callback(checkpoint_dir)
            return CheckpointWriteResult(
                step=step,
                checkpoint_dir=checkpoint_dir,
                final_updated=bool(kwargs["is_final"]),
                best_updated=False,
            )

    def publish_exact_state(step: int, checkpoint_dir: Path) -> None:
        state_dir = checkpoint_dir / "training_state"
        state_dir.mkdir()
        (state_dir / "manifest.json").write_bytes(manifest_bytes_by_step[step])

    monkeypatch.setattr(
        pipeline,
        "load_training_state_manifest",
        lambda checkpoint_dir: SimpleNamespace(
            aggregate_digest=aggregate_by_step[int(Path(checkpoint_dir).name[5:])]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_inference_checkpoint_payload_identity",
        lambda checkpoint_dir: payload_identity_by_step[
            int(Path(checkpoint_dir).name[5:])
        ],
    )
    monkeypatch.setattr(
        run_writer_module,
        "admit_inference_checkpoint_payload_identity",
        lambda checkpoint_dir, expected: dict(expected),
    )
    lifecycle: dict[str, Any] = {
        "active_phase": None,
        "checkpoint_event_count": 0,
        "phase_rank_receipts": {},
        "phase_started_monotonic": None,
    }
    handler = pipeline._checkpoint_handler(
        CheckpointWriter(),
        model=object(),
        runtime=SimpleNamespace(accelerator=SimpleNamespace()),
        adapter_name="default",
        special_token_result=None,
        schedule=SimpleNamespace(resolved_max_steps=5),
        base_model_path=tmp_path / "model",
        base_config_sha256="base",
        tokenizer_sha256="tokenizer",
        writer=writer,
        eval_by_step={},
        committed_steps=set(),
        lifecycle=lifecycle,
        exact_training_state_callback_factory=publish_exact_state,
    )
    observation = SimpleNamespace(
        optimizer_update_status="applied",
        finite_status="finite",
    )

    lifecycle.update(completed_steps=3, consumed_packs=9)
    handler(SimpleNamespace(planned_step_id=3), observation)
    lifecycle.update(completed_steps=5, consumed_packs=15)
    handler(SimpleNamespace(planned_step_id=5), observation)
    writer.finalize(
        status="completed",
        updated_at="2026-08-11T00:00:01+00:00",
        completed_steps=5,
        consumed_packs=15,
        checkpoint_event_count=2,
        optimizer_update_status="applied",
        finite_status="finite",
    )

    events = writer.read_run()["measurement"]["checkpoint_publication_events"]
    assert [event["step"] for event in events] == [3, 5]
    assert [event["is_final"] for event in events] == [False, True]
    for event in events:
        step = event["step"]
        checkpoint_dir = writer.checkpoints_dir / f"step-{step}"
        assert event["status"] == "completed"
        assert event["duration_clock"] == "monotonic"
        assert event["duration_seconds"] >= 0.0
        assert event["started_at"].endswith("+00:00")
        assert event["completed_at"].endswith("+00:00")
        assert event["checkpoint_path"] == f"checkpoints/step-{step}"
        assert event["exact_training_state_enabled"] is True
        assert event["failure_code"] is None
        assert event["schema"] == "coordexp-swift-checkpoint-publication-event"
        assert event["schema_version"] == 2
        assert event["inference_payload_identity"] == payload_identity_by_step[step]
        assert event["committed_progress"] == {
            "schema": "coordexp-swift-checkpoint-committed-progress",
            "schema_version": 1,
            "completed_steps": step,
            "consumed_packs": 9 if step == 3 else 15,
            "optimizer_update_status": "applied",
            "finite_status": "finite",
        }
        assert event["checkpoint_identity"] == {
            "checkpoint_step": step,
            "resolved_path": str(checkpoint_dir.resolve()),
            "training_state_aggregate_digest": aggregate_by_step[step],
            "training_state_manifest_file_sha256": hashlib.sha256(
                manifest_bytes_by_step[step]
            ).hexdigest(),
        }


def test_checkpoint_handler_records_failed_publication_without_complete_identity(
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=5,
    )

    class FailingCheckpointWriter:
        def write_checkpoint(self, **kwargs: Any) -> CheckpointWriteResult:
            raise RuntimeContractError(
                "injected publication failure",
                code="test.checkpoint_publication_failed",
            )

    handler = pipeline._checkpoint_handler(
        FailingCheckpointWriter(),
        model=object(),
        runtime=SimpleNamespace(accelerator=SimpleNamespace()),
        adapter_name="default",
        special_token_result=None,
        schedule=SimpleNamespace(resolved_max_steps=5),
        base_model_path=tmp_path / "model",
        base_config_sha256="base",
        tokenizer_sha256="tokenizer",
        writer=writer,
        eval_by_step={},
        committed_steps=set(),
        lifecycle={"checkpoint_event_count": 0},
        exact_training_state_callback_factory=lambda step, path: None,
    )

    with pytest.raises(RuntimeContractError, match="injected publication failure"):
        handler(
            SimpleNamespace(planned_step_id=3),
            SimpleNamespace(
                optimizer_update_status="applied",
                finite_status="finite",
            ),
        )

    event = writer.read_run()["measurement"]["checkpoint_publication_events"][0]
    assert event["status"] == "failed"
    assert event["step"] == 3
    assert event["checkpoint_identity"] is None
    assert event["inference_payload_identity"] is None
    assert event["committed_progress"] is None
    assert event["exact_training_state_enabled"] is True
    assert event["failure_code"] == "test.checkpoint_publication_failed"
    assert event["duration_clock"] == "monotonic"
    assert event["duration_seconds"] >= 0.0


def test_event_sink_failure_leaves_durable_checkpoint_unadmitted_before_restore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="parent-run",
        run_name="parent",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=4,
        segment_id="parent-segment",
    )
    manifest_bytes = b'{"durable":"but-event-sink-failed"}\n'

    class CheckpointWriter:
        def write_checkpoint(self, **kwargs: Any) -> CheckpointWriteResult:
            step = int(kwargs["step"])
            checkpoint_dir = writer.checkpoints_dir / f"step-{step}"
            checkpoint_dir.mkdir(parents=True)
            callback = kwargs["exact_training_state_callback"]
            assert callable(callback)
            callback(checkpoint_dir)
            return CheckpointWriteResult(
                step=step,
                checkpoint_dir=checkpoint_dir,
                final_updated=False,
                best_updated=False,
            )

    def publish_exact_state(step: int, checkpoint_dir: Path) -> None:
        assert step == 2
        state_dir = checkpoint_dir / "training_state"
        state_dir.mkdir()
        (state_dir / "manifest.json").write_bytes(manifest_bytes)

    monkeypatch.setattr(
        pipeline,
        "build_inference_checkpoint_payload_identity",
        lambda checkpoint_dir: {"uncommitted": str(checkpoint_dir)},
    )
    monkeypatch.setattr(
        pipeline,
        "load_training_state_manifest",
        lambda checkpoint_dir: SimpleNamespace(aggregate_digest="a" * 64),
    )
    original_record = RunWriter.record_checkpoint_publication_event

    def inject_completed_event_failure(self: RunWriter, **kwargs: Any) -> None:
        if kwargs["status"] == "completed":
            raise RuntimeContractError(
                "injected completed-event sink failure",
                code="test.checkpoint_event_sink_failed",
            )
        original_record(self, **kwargs)

    monkeypatch.setattr(
        RunWriter,
        "record_checkpoint_publication_event",
        inject_completed_event_failure,
    )
    handler = pipeline._checkpoint_handler(
        CheckpointWriter(),
        model=object(),
        runtime=SimpleNamespace(accelerator=SimpleNamespace()),
        adapter_name="default",
        special_token_result=None,
        schedule=SimpleNamespace(resolved_max_steps=4),
        base_model_path=tmp_path / "model",
        base_config_sha256="base",
        tokenizer_sha256="tokenizer",
        writer=writer,
        eval_by_step={},
        committed_steps=set(),
        lifecycle={"checkpoint_event_count": 0, "consumed_packs": 4},
        exact_training_state_callback_factory=publish_exact_state,
    )

    with pytest.raises(RuntimeContractError) as publication_error:
        handler(
            SimpleNamespace(planned_step_id=2),
            SimpleNamespace(
                optimizer_update_status="applied",
                finite_status="finite",
            ),
        )
    assert publication_error.value.code == "test.checkpoint_event_sink_failed"

    checkpoint_dir = writer.checkpoints_dir / "step-2"
    assert (checkpoint_dir / "training_state" / "manifest.json").is_file()
    events = writer.read_run()["measurement"]["checkpoint_publication_events"]
    assert [event["status"] for event in events] == ["failed"]

    admitted = SimpleNamespace(
        manifest=SimpleNamespace(
            aggregate_digest="a" * 64,
            checkpoint_step=2,
            continuation_index=0,
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
        ),
        decoded_rank=_decoded_rank({"data": {}, "pack": {}}),
    )
    monkeypatch.setattr(
        pipeline, "admit_training_state", lambda *args, **kwargs: admitted
    )
    monkeypatch.setattr(
        pipeline, "capture_runtime_state_expectations", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline, "_validate_exact_resume_cursor", lambda *args, **kwargs: {}
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(Exception) as admission_error:
        _read_only_admit_pipeline_exact_resume(
            checkpoint_dir,
            checkpoint_step=2,
            rank=0,
            world_size=1,
            identities={
                name: str(index) * 64
                for index, name in enumerate(
                    (
                        "base_model",
                        "cache",
                        "dependencies",
                        "policy",
                        "resolved_config",
                        "resume_compatibility",
                        "topology",
                        "trainable_surface",
                    ),
                    start=1,
                )
            },
            resolved_config={
                "config": {"run": {}, "resume": {}, "training": {}},
                "resolution": {},
            },
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            train_micro_steps=(),
            train_cache={"fingerprint": "b" * 64, "format_version": "v3"},
            schedule=_schedule(),
        )

    assert getattr(admission_error.value, "code", None) == (
        "run_writer.exact_resume_publication_invalid"
    )
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in model_before.items()
    )


def test_restored_cursor_seeds_absolute_counters_and_slices_only_trainer_input() -> (
    None
):
    train_micro_steps = _micro_steps(8)
    schedule = _schedule()
    train_cache = {"fingerprint": "a" * 64, "format_version": "v3"}
    counters = {
        "optimizer_step_count": 2,
        "scheduler_step_count": 2,
        "zero_grad_count": 2,
    }
    from src.training.pipeline import _exact_resume_cursor_from_counters

    cursor = _exact_resume_cursor_from_counters(
        checkpoint_step=2,
        consumed_micro_steps=4,
        train_micro_steps=train_micro_steps,
        train_cache=train_cache,
        schedule=schedule,
        counters=counters,
        rank=0,
        world_size=1,
    )
    runtime = SimpleNamespace(
        optimizer_step_count=0,
        scheduler_step_count=0,
        zero_grad_count=0,
    )
    lifecycle = {"completed_steps": 0, "consumed_packs": 0}

    result = _apply_exact_resume_cursor_state(
        cursor,
        next_rank_local_micro_step=4,
        checkpoint_step=2,
        train_micro_steps=train_micro_steps,
        train_cache=train_cache,
        schedule=schedule,
        runtime=runtime,
        lifecycle=lifecycle,
        rank=0,
        world_size=1,
    )

    assert result.start_planned_step_id == 3
    assert result.trainer_micro_steps == train_micro_steps[4:]
    assert train_micro_steps[4].pack.pack_index == 4
    assert runtime.optimizer_step_count == 2
    assert runtime.scheduler_step_count == 2
    assert runtime.zero_grad_count == 2
    assert lifecycle["completed_steps"] == 2
    assert lifecycle["consumed_packs"] == 4
    assert train_micro_steps == _micro_steps(8)


def test_terminal_cursor_rejected_before_any_counter_mutation() -> None:
    train_micro_steps = _micro_steps(8)
    schedule = _schedule()
    train_cache = {"fingerprint": "a" * 64, "format_version": "v3"}
    from src.training.pipeline import _exact_resume_cursor_from_counters

    cursor = _exact_resume_cursor_from_counters(
        checkpoint_step=4,
        consumed_micro_steps=8,
        train_micro_steps=train_micro_steps,
        train_cache=train_cache,
        schedule=schedule,
        counters={
            "optimizer_step_count": 4,
            "scheduler_step_count": 4,
            "zero_grad_count": 4,
        },
        rank=0,
        world_size=1,
    )
    runtime = SimpleNamespace(
        optimizer_step_count=99,
        scheduler_step_count=99,
        zero_grad_count=99,
    )
    lifecycle = {"completed_steps": 99, "consumed_packs": 99}

    with pytest.raises(Exception) as error:
        _apply_exact_resume_cursor_state(
            cursor,
            next_rank_local_micro_step=8,
            checkpoint_step=4,
            train_micro_steps=train_micro_steps,
            train_cache=train_cache,
            schedule=schedule,
            runtime=runtime,
            lifecycle=lifecycle,
            rank=0,
            world_size=1,
        )

    assert getattr(error.value, "code", None) == "training.resume_terminal_checkpoint"
    assert runtime.optimizer_step_count == 99
    assert lifecycle == {"completed_steps": 99, "consumed_packs": 99}


def test_publication_plan_names_the_current_publishing_segment() -> None:
    resolved_config = {
        "config": {
            "run": {"name": "child"},
            "resume": {"mode": "exact_same_world_size"},
            "training": {"seed": 17},
        },
        "resolution": {},
    }
    plan = _build_exact_resume_publication_plan(
        checkpoint_step=3,
        run_id="run-child",
        run_segment_id="segment-child",
        continuation_index=2,
        world_size=4,
        identities={
            "base_model": "1" * 64,
            "cache": "2" * 64,
            "dependencies": "3" * 64,
            "policy": "4" * 64,
            "resolved_config": "5" * 64,
            "resume_compatibility": "6" * 64,
            "topology": "7" * 64,
            "trainable_surface": "8" * 64,
        },
        resolved_config=resolved_config,
        scheduler_applicable=True,
        scaler_applicable=False,
    )

    assert isinstance(plan, TrainingStatePublicationPlan)
    assert plan.parent_run_id == "run-child"
    assert plan.parent_segment_id == "segment-child"
    assert plan.continuation_index == 2
    assert plan.accumulation_microstep == 0


def test_pipeline_identities_bind_physical_topology_but_project_resume_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "base_model_weight_identity",
        lambda path: {"aggregate_sha256": "1" * 64},
    )
    cache = {
        "determinants_sha256": "2" * 64,
        "fingerprint": "3" * 64,
        "format_version": "coordexp-swift-pack-cache-v3",
        "manifest_sha256": "4" * 64,
    }
    parent = {
        "config": {
            "run": {"name": "parent"},
            "resume": {"mode": "disabled", "checkpoint_dir": None},
            "training": {"seed": 17},
        },
        "resolution": {},
    }
    child = {
        "config": {
            **parent["config"],
            "run": {"name": "child"},
            "resume": {
                "mode": "exact_same_world_size",
                "checkpoint_dir": "/checkpoint",
            },
        },
        "resolution": {},
    }
    bindings = (RankCudaDeviceBinding(0, 1, 0, "GPU-a"),)
    pinned_runtime_baseline = {
        "schema_version": 3,
        "baseline_sha256": "9" * 64,
        "attention_backend": "flash_attention_2",
        "admitted": True,
        "mismatches": [],
        "reference_only": {},
    }

    parent_identities = _build_pipeline_exact_resume_identities(
        base_model_path=tmp_path,
        train_cache=cache,
        eval_cache=None,
        pinned_runtime_baseline=pinned_runtime_baseline,
        policy={"mode": "stable"},
        resolved_config=parent,
        topology_bindings=bindings,
        trainable_surface={"names": ["adapter.weight"]},
    )
    child_identities = _build_pipeline_exact_resume_identities(
        base_model_path=tmp_path,
        train_cache=cache,
        eval_cache=None,
        pinned_runtime_baseline=pinned_runtime_baseline,
        policy={"mode": "stable"},
        resolved_config=child,
        topology_bindings=bindings,
        trainable_surface={"names": ["adapter.weight"]},
    )

    assert parent_identities["resolved_config"] != child_identities["resolved_config"]
    assert (
        parent_identities["resume_compatibility"]
        == child_identities["resume_compatibility"]
    )
    assert parent_identities["policy"] == child_identities["policy"]
    assert parent_identities["topology"] == child_identities["topology"]


def test_restored_cursor_owner_envelopes_are_unwrapped_without_metadata_leakage() -> (
    None
):
    assert _restored_exact_resume_cursor_state(
        {
            "data": {"owner": "data", "state": {"position": 4}},
            "pack": {"owner": "pack", "state": {"pack_index": 4}},
        }
    ) == {
        "data": {"position": 4},
        "pack": {"pack_index": 4},
    }


def test_dependency_identity_excludes_reference_only_ms_swift_observation() -> None:
    admitted = {
        "schema_version": 3,
        "baseline_sha256": "a" * 64,
        "attention_backend": "flash_attention_2",
        "admitted": True,
        "mismatches": [],
        "reference_only": {
            "ms-swift": {"matches_recorded_reference": True, "mismatches": []}
        },
    }
    drifted_reference = {
        **admitted,
        "reference_only": {
            "ms-swift": {
                "matches_recorded_reference": False,
                "mismatches": ["commit"],
            }
        },
    }

    assert _exact_resume_dependency_identity(admitted) == (
        _exact_resume_dependency_identity(drifted_reference)
    )
    rejected_runtime = {**admitted, "admitted": False, "mismatches": ["torch.sha256"]}
    with pytest.raises(Exception) as error:
        _exact_resume_dependency_identity(rejected_runtime)
    assert (
        getattr(error.value, "code", None) == "training.resume_dependency_not_admitted"
    )


def test_wrong_cursor_is_rejected_by_read_only_admission_before_state_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer_before = optimizer.state_dict()
    scheduler_before = scheduler.state_dict()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    train_micro_steps = _micro_steps(8)
    wrong_cursor = {
        "data": {
            "checkpoint_step": 2,
            "next_rank_local_micro_step": 4,
            "rank": 0,
            "resolved_grad_accum_steps": 2,
            "resolved_max_steps": 4,
            "runtime_counters": {
                "optimizer_step_count": 2,
                "scheduler_step_count": 2,
                "zero_grad_count": 2,
            },
            "schema": "coordexp-swift-rank-data-cursor-v1",
            "total_rank_local_micro_steps": 8,
            "world_size": 1,
        },
        "pack": {
            "cache": {"fingerprint": "a" * 64, "format_version": "v3"},
            "checkpoint_step": 2,
            "next_pack": {
                "example_ids": ["wrong-example"],
                "pack_index": 4,
                "sequence_length": 2,
            },
            "next_rank_local_micro_step": 4,
            "rank": 0,
            "resolved_grad_accum_steps": 2,
            "resolved_max_steps": 4,
            "schema": "coordexp-swift-rank-pack-cursor-v1",
            "total_rank_local_micro_steps": 8,
            "world_size": 1,
        },
    }
    admitted = SimpleNamespace(
        manifest=SimpleNamespace(aggregate_digest="f" * 64),
        decoded_rank=_decoded_rank(wrong_cursor),
    )
    monkeypatch.setattr(
        pipeline, "admit_training_state", lambda *args, **kwargs: admitted
    )
    monkeypatch.setattr(
        pipeline, "capture_runtime_state_expectations", lambda **kwargs: object()
    )
    manifest_snapshot = SimpleNamespace(file_sha256="e" * 64)
    monkeypatch.setattr(
        pipeline,
        "_snapshot_training_state_manifest",
        lambda checkpoint_dir: manifest_snapshot,
    )

    with pytest.raises(Exception) as error:
        _read_only_admit_pipeline_exact_resume(
            "/checkpoint",
            checkpoint_step=2,
            rank=0,
            world_size=1,
            identities={
                "base_model": "1" * 64,
                "cache": "2" * 64,
                "dependencies": "3" * 64,
                "policy": "4" * 64,
                "resolved_config": "5" * 64,
                "resume_compatibility": "6" * 64,
                "topology": "7" * 64,
                "trainable_surface": "8" * 64,
            },
            resolved_config={
                "config": {
                    "run": {},
                    "resume": {},
                    "training": {"seed": 17},
                },
                "resolution": {},
            },
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            train_micro_steps=train_micro_steps,
            train_cache={"fingerprint": "a" * 64, "format_version": "v3"},
            schedule=_schedule(),
        )

    assert getattr(error.value, "code", None) == "training.resume_cursor_mismatch"
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in model_before.items()
    )
    assert optimizer.state_dict() == optimizer_before
    assert scheduler.state_dict() == scheduler_before
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_run",
        "corrupt_run",
        "duplicate_event",
        "checkpoint_path",
        "checkpoint_identity",
        "inference_payload_identity",
        "committed_progress",
        "top_level_progress",
    ],
)
def test_read_only_admission_rejects_uncommitted_parent_run_state_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "parent-run",
        run_id="parent-run",
        run_name="parent",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=4,
        segment_id="parent-segment",
    )
    checkpoint_dir = writer.checkpoints_dir / "step-2"
    manifest_path = checkpoint_dir / "training_state" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_bytes = b'{"authenticated":"manifest"}\n'
    manifest_path.write_bytes(manifest_bytes)
    live_payload_identity = _record_exact_publication(
        writer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=2,
        manifest_bytes=manifest_bytes,
        aggregate_digest="a" * 64,
        monkeypatch=monkeypatch,
    )
    state = writer.read_run()
    events = state["measurement"]["checkpoint_publication_events"]
    if mutation == "missing_run":
        writer.run_path.unlink()
    elif mutation == "corrupt_run":
        writer.run_path.write_text("{not-json", encoding="utf-8")
    elif mutation == "duplicate_event":
        events.append(dict(events[0]))
        writer.run_path.write_text(json.dumps(state), encoding="utf-8")
    elif mutation == "checkpoint_path":
        events[0]["checkpoint_path"] = "checkpoints/step-3"
        writer.run_path.write_text(json.dumps(state), encoding="utf-8")
    elif mutation == "checkpoint_identity":
        events[0]["checkpoint_identity"]["training_state_aggregate_digest"] = "b" * 64
        writer.run_path.write_text(json.dumps(state), encoding="utf-8")
    elif mutation == "inference_payload_identity":
        events[0]["inference_payload_identity"]["aggregate_digest"] = "e" * 64
        writer.run_path.write_text(json.dumps(state), encoding="utf-8")
    elif mutation == "committed_progress":
        events[0]["committed_progress"]["consumed_packs"] = 5
        writer.run_path.write_text(json.dumps(state), encoding="utf-8")
    elif mutation == "top_level_progress":
        state["consumed_packs"] = 5
        writer.run_path.write_text(json.dumps(state), encoding="utf-8")
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    monkeypatch.setattr(
        run_writer_module,
        "admit_inference_checkpoint_payload_identity",
        lambda _checkpoint_dir, _expected: dict(live_payload_identity),
    )
    admitted = SimpleNamespace(
        manifest=SimpleNamespace(
            aggregate_digest="a" * 64,
            checkpoint_step=2,
            continuation_index=3,
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
        ),
        decoded_rank=_decoded_rank({"data": {}, "pack": {}}),
    )
    monkeypatch.setattr(
        pipeline, "admit_training_state", lambda *args, **kwargs: admitted
    )
    monkeypatch.setattr(
        pipeline, "capture_runtime_state_expectations", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline, "_validate_exact_resume_cursor", lambda *args, **kwargs: {}
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer_before = optimizer.state_dict()

    with pytest.raises(Exception) as error:
        _read_only_admit_pipeline_exact_resume(
            checkpoint_dir,
            checkpoint_step=2,
            rank=0,
            world_size=1,
            identities={
                name: str(index) * 64
                for index, name in enumerate(
                    (
                        "base_model",
                        "cache",
                        "dependencies",
                        "policy",
                        "resolved_config",
                        "resume_compatibility",
                        "topology",
                        "trainable_surface",
                    ),
                    start=1,
                )
            },
            resolved_config={
                "config": {"run": {}, "resume": {}, "training": {}},
                "resolution": {},
            },
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            train_micro_steps=(),
            train_cache={"fingerprint": "b" * 64, "format_version": "v3"},
            schedule=_schedule(),
        )

    assert getattr(error.value, "code", None) == (
        "run_writer.exact_resume_publication_invalid"
    )
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in model_before.items()
    )
    assert optimizer.state_dict() == optimizer_before


def test_parent_run_cannot_roll_back_to_an_older_completed_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "parent-run",
        run_id="parent-run",
        run_name="parent",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=4,
        segment_id="parent-segment",
    )
    manifests: dict[int, bytes] = {}
    for step in (2, 3):
        checkpoint_dir = writer.checkpoints_dir / f"step-{step}"
        manifest_path = checkpoint_dir / "training_state" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_bytes = f'{{"checkpoint_step":{step}}}\n'.encode()
        manifest_path.write_bytes(manifest_bytes)
        manifests[step] = manifest_bytes
        _record_exact_publication(
            writer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=step,
            manifest_bytes=manifest_bytes,
            aggregate_digest=str(step) * 64,
            monkeypatch=monkeypatch,
        )

    state = writer.read_run()
    older_progress = state["measurement"]["checkpoint_publication_events"][0][
        "committed_progress"
    ]
    state.update(
        completed_steps=older_progress["completed_steps"],
        consumed_packs=older_progress["consumed_packs"],
        checkpoint_event_count=1,
        final_optimizer_update_status=older_progress["optimizer_update_status"],
        final_finite_status=older_progress["finite_status"],
    )
    writer.run_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(Exception) as error:
        run_writer_module.admit_exact_resume_checkpoint_publication(
            writer.checkpoints_dir / "step-2",
            checkpoint_step=2,
            training_state_manifest_file_sha256=hashlib.sha256(
                manifests[2]
            ).hexdigest(),
            training_state_aggregate_digest="2" * 64,
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
        )

    assert getattr(error.value, "code", None) == (
        "run_writer.exact_resume_publication_invalid"
    )
    preserved = writer.read_run()["measurement"]["checkpoint_publication_events"]
    assert [event["step"] for event in preserved] == [2, 3]


def test_failed_event_after_latest_completion_does_not_supersede_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "parent-run",
        run_id="parent-run",
        run_name="parent",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=4,
        segment_id="parent-segment",
    )
    checkpoint_dir = writer.checkpoints_dir / "step-2"
    manifest_path = checkpoint_dir / "training_state" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_bytes = b'{"checkpoint_step":2}\n'
    manifest_path.write_bytes(manifest_bytes)
    _record_exact_publication(
        writer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=2,
        manifest_bytes=manifest_bytes,
        aggregate_digest="2" * 64,
        monkeypatch=monkeypatch,
    )
    writer.record_checkpoint_publication_event(
        step=3,
        status="failed",
        started_at="2026-08-11T00:00:01+00:00",
        completed_at="2026-08-11T00:00:02+00:00",
        duration_seconds=1.0,
        is_final=False,
        exact_training_state_enabled=True,
        checkpoint_identity=None,
        inference_payload_identity=None,
        committed_progress=None,
        failure_code="checkpoint.injected_failure",
    )

    admitted = run_writer_module.admit_exact_resume_checkpoint_publication(
        checkpoint_dir,
        checkpoint_step=2,
        training_state_manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        training_state_aggregate_digest="2" * 64,
        parent_run_id="parent-run",
        parent_segment_id="parent-segment",
    )

    assert admitted["event_index"] == 0
    assert [
        event["status"]
        for event in writer.read_run()["measurement"]["checkpoint_publication_events"]
    ] == ["completed", "failed"]


def test_read_only_admission_builds_lineage_from_the_authenticated_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "parent-run",
        run_id="parent-run",
        run_name="parent",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-08-11T00:00:00+00:00",
        config_fingerprint="f" * 64,
        resolved_config={},
        world_size=1,
        resolved_max_steps=4,
        segment_id="parent-segment",
    )
    checkpoint_dir = writer.checkpoints_dir / "step-2"
    manifest_path = checkpoint_dir / "training_state" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_bytes = b'{"authenticated":"manifest"}\n'
    manifest_path.write_bytes(manifest_bytes)
    _record_exact_publication(
        writer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=2,
        manifest_bytes=manifest_bytes,
        aggregate_digest="a" * 64,
        monkeypatch=monkeypatch,
    )
    manifest = SimpleNamespace(
        aggregate_digest="a" * 64,
        checkpoint_step=2,
        continuation_index=3,
        parent_run_id="parent-run",
        parent_segment_id="parent-segment",
    )
    admitted = SimpleNamespace(
        manifest=manifest,
        decoded_rank=_decoded_rank({"data": {}, "pack": {}}),
    )
    monkeypatch.setattr(
        pipeline, "admit_training_state", lambda *args, **kwargs: admitted
    )
    monkeypatch.setattr(
        pipeline, "capture_runtime_state_expectations", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline, "_validate_exact_resume_cursor", lambda *args, **kwargs: {}
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    result = _read_only_admit_pipeline_exact_resume(
        checkpoint_dir,
        checkpoint_step=2,
        rank=0,
        world_size=1,
        identities={
            name: str(index) * 64
            for index, name in enumerate(
                (
                    "base_model",
                    "cache",
                    "dependencies",
                    "policy",
                    "resolved_config",
                    "resume_compatibility",
                    "topology",
                    "trainable_surface",
                ),
                start=1,
            )
        },
        resolved_config={
            "config": {"run": {}, "resume": {}, "training": {}},
            "resolution": {},
        },
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        train_micro_steps=(),
        train_cache={"fingerprint": "b" * 64, "format_version": "v3"},
        schedule=_schedule(),
    )

    assert result.continuation_lineage == {
        "parent_run_id": "parent-run",
        "parent_segment_id": "parent-segment",
        "parent_checkpoint_identity": {
            "resolved_path": str(checkpoint_dir.resolve()),
            "checkpoint_step": 2,
            "training_state_manifest_file_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "training_state_aggregate_digest": "a" * 64,
        },
        "parent_continuation_index": 3,
        "continuation_index": 4,
    }
    assert result.next_rank_local_micro_step == 4


@pytest.mark.parametrize(
    "next_rank_local_micro_step",
    [_MISSING, True, 4.0, "4", -1],
    ids=["missing", "bool", "float", "string", "negative"],
)
def test_read_only_admission_rejects_invalid_canonical_cursor_position(
    monkeypatch: pytest.MonkeyPatch,
    next_rank_local_micro_step: Any,
) -> None:
    admitted = SimpleNamespace(
        manifest=SimpleNamespace(aggregate_digest="f" * 64),
        decoded_rank=_decoded_rank(
            {"data": {}, "pack": {}},
            next_rank_local_micro_step=next_rank_local_micro_step,
        ),
    )
    monkeypatch.setattr(
        pipeline, "admit_training_state", lambda *args, **kwargs: admitted
    )
    monkeypatch.setattr(
        pipeline, "capture_runtime_state_expectations", lambda **kwargs: object()
    )
    manifest_snapshot = SimpleNamespace(file_sha256="e" * 64)
    monkeypatch.setattr(
        pipeline,
        "_snapshot_training_state_manifest",
        lambda checkpoint_dir: manifest_snapshot,
    )
    monkeypatch.setattr(
        pipeline,
        "_restored_exact_resume_cursor_state",
        lambda cursor: pytest.fail("cursor state restored before position validation"),
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with pytest.raises(RuntimeContractError) as error:
        _read_only_admit_pipeline_exact_resume(
            "/checkpoint",
            checkpoint_step=2,
            rank=0,
            world_size=1,
            identities={
                name: str(index) * 64
                for index, name in enumerate(
                    (
                        "base_model",
                        "cache",
                        "dependencies",
                        "policy",
                        "resolved_config",
                        "resume_compatibility",
                        "topology",
                        "trainable_surface",
                    ),
                    start=1,
                )
            },
            resolved_config={
                "config": {"run": {}, "resume": {}, "training": {}},
                "resolution": {},
            },
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            train_micro_steps=(),
            train_cache={"fingerprint": "b" * 64, "format_version": "v3"},
            schedule=_schedule(),
        )

    assert error.value.code == "training.resume_cursor_invalid"


def test_read_only_admission_rejects_manifest_path_replacement_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    manifest_path = checkpoint_dir / "training_state" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"first-manifest")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    def replace_manifest(*args: object, **kwargs: object) -> object:
        manifest_path.replace(manifest_path.with_suffix(".retired"))
        manifest_path.write_bytes(b"replacement-manifest")
        return SimpleNamespace(
            manifest=SimpleNamespace(
                aggregate_digest="a" * 64,
                checkpoint_step=2,
                continuation_index=0,
                parent_run_id="parent-run",
                parent_segment_id="parent-segment",
            ),
            decoded_rank=_decoded_rank({"data": {}, "pack": {}}),
        )

    monkeypatch.setattr(pipeline, "admit_training_state", replace_manifest)
    monkeypatch.setattr(
        pipeline, "capture_runtime_state_expectations", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        pipeline, "_validate_exact_resume_cursor", lambda *args, **kwargs: {}
    )

    with pytest.raises(Exception) as error:
        _read_only_admit_pipeline_exact_resume(
            checkpoint_dir,
            checkpoint_step=2,
            rank=0,
            world_size=1,
            identities={
                name: str(index) * 64
                for index, name in enumerate(
                    (
                        "base_model",
                        "cache",
                        "dependencies",
                        "policy",
                        "resolved_config",
                        "resume_compatibility",
                        "topology",
                        "trainable_surface",
                    ),
                    start=1,
                )
            },
            resolved_config={
                "config": {"run": {}, "resume": {}, "training": {}},
                "resolution": {},
            },
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            train_micro_steps=(),
            train_cache={"fingerprint": "b" * 64, "format_version": "v3"},
            schedule=_schedule(),
        )

    assert (
        getattr(error.value, "code", None) == "training.resume_manifest_path_replaced"
    )
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in model_before.items()
    )
