from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import src.training.supervised_trainer as trainer_module
from src.common.errors import RuntimeContractError
from src.config.models import DeepSpeedConfig, RuntimeBatchResolution, RuntimeConfig
from src.runtime import GateDecision
from src.runtime.train_runtime import TrainRuntime
from src.training.schedule import ResolvedStepSchedule, StepScheduleEvent
from src.training.supervised_trainer import (
    LossContextFactory,
    LossRunnerBoundary,
    ScheduledTrainerEvent,
    ScheduledEventHandler,
    SupervisedMicroStep,
    SupervisedTrainer,
    TrainerEventSink,
    QwenForwardFn,
    RuntimeBoundary,
)


def test_supervised_trainer_orchestrates_accumulation_and_runtime_boundaries() -> None:
    log: list[str] = []
    schedule = _schedule(
        resolved_max_steps=2,
        grad_accum_steps=2,
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(4, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=FakeLossRunner(log),
        runtime=FakeRuntime(log),
        event_sink=lambda event: log.append(f"event:{event.event_type}:{event.planned_step_id}"),
    )

    result = trainer.run()

    assert result.completed_steps == 2
    assert result.consumed_micro_steps == 4
    assert result.step_results[0].micro_step_count == 2
    assert result.step_results[0].optimizer_update_status == "applied"
    assert result.step_results[0].loss_bundle_artifact == {"total_loss": 1.0}
    assert result.step_results[0].scheduler_artifact == {
        "scheduler_step_count": 1,
        "learning_rates": [{"group_index": 0, "lr": 0.01}],
    }
    assert result.step_results[0].to_artifact_dict()["scheduler"] == {
        "scheduler_step_count": 1,
        "learning_rates": [{"group_index": 0, "lr": 0.01}],
    }
    assert not hasattr(result.step_results[0], "loss_bundle")
    assert log == [
        "event:planned_step.started:1",
        "stream:0",
        "runtime.move:1:0",
        "forward:0",
        "context:0",
        "event:micro_step.forward:1",
        "stream:1",
        "runtime.move:1:1",
        "forward:1",
        "context:1",
        "event:micro_step.forward:1",
        "loss:2",
        "event:planned_step.loss:1",
        "runtime.pre:1",
        "event:planned_step.pre_backward_gate:1",
        "runtime.backward:1.0:sync=True",
        "runtime.post:1",
        "event:planned_step.post_backward_gate:1",
        "runtime.clip:1",
        "runtime.optimizer:1",
        "runtime.scheduler:1",
        "runtime.zero:1",
        "event:planned_step.completed:1",
        "event:planned_step.started:2",
        "stream:2",
        "runtime.move:2:0",
        "forward:2",
        "context:2",
        "event:micro_step.forward:2",
        "stream:3",
        "runtime.move:2:1",
        "forward:3",
        "context:3",
        "event:micro_step.forward:2",
        "loss:2",
        "event:planned_step.loss:2",
        "runtime.pre:2",
        "event:planned_step.pre_backward_gate:2",
        "runtime.backward:1.0:sync=True",
        "runtime.post:2",
        "event:planned_step.post_backward_gate:2",
        "runtime.clip:2",
        "runtime.optimizer:2",
        "runtime.scheduler:2",
        "runtime.zero:2",
        "event:planned_step.completed:2",
    ]


def test_supervised_trainer_triggers_scheduled_eval_checkpoint_and_final_events() -> None:
    scheduled_calls: list[tuple[str, int, tuple[str, ...], str]] = []
    schedule = _schedule(
        resolved_max_steps=2,
        grad_accum_steps=1,
        events={
            "eval.forward": (
                _event(1, "eval.forward", ("explicit_step",)),
            ),
            "checkpoint": (
                _event(2, "checkpoint", ("save_final",)),
            ),
            "training.logging": (
                _event(1, "training.logging", ("explicit_step",)),
            ),
            "final": (
                _event(2, "final", ("final",), required=True),
            ),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(2),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=FakeRuntime([]),
        scheduled_event_handlers={
            "eval.forward": lambda event: scheduled_calls.append(_scheduled_tuple(event)),
            "checkpoint": lambda event: scheduled_calls.append(_scheduled_tuple(event)),
            "final": lambda event: scheduled_calls.append(_scheduled_tuple(event)),
        },
    )

    result = trainer.run()

    assert result.completed_steps == 2
    assert result.scheduled_event_counts == {
        "checkpoint": 1,
        "eval.forward": 1,
        "final": 1,
        "training.logging": 1,
    }
    assert scheduled_calls == [
        ("eval.forward", 1, ("explicit_step",), "applied"),
        ("checkpoint", 2, ("save_final",), "applied"),
        ("final", 2, ("final",), "applied"),
    ]


def test_supervised_trainer_runs_same_step_eval_before_checkpoint() -> None:
    scheduled_calls: list[str] = []
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        events={
            "checkpoint": (
                _event(1, "checkpoint", ("every_fraction:1.0",)),
            ),
            "eval.forward": (
                _event(1, "eval.forward", ("explicit_step",)),
            ),
            "training.logging": (
                _event(1, "training.logging", ("explicit_step",)),
            ),
            "final": (
                _event(1, "final", ("final",), required=True),
            ),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=FakeRuntime([]),
        scheduled_event_handlers={
            "eval.forward": lambda event: scheduled_calls.append(event.scheduled_event.event),
            "checkpoint": lambda event: scheduled_calls.append(event.scheduled_event.event),
            "final": lambda event: scheduled_calls.append(event.scheduled_event.event),
        },
    )

    result = trainer.run()

    assert result.scheduled_event_counts == {
        "checkpoint": 1,
        "eval.forward": 1,
        "final": 1,
        "training.logging": 1,
    }
    assert scheduled_calls == ["eval.forward", "checkpoint", "final"]


def test_supervised_trainer_rejects_required_scheduled_event_without_handler() -> None:
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        events={
            "checkpoint": (),
            "eval.forward": (),
            "training.logging": (),
            "final": (_event(1, "final", ("final",), required=True),),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        trainer.run()

    assert exc_info.value.code == "trainer.required_event_handler_missing"
    assert exc_info.value.context["event"] == "final"
    assert exc_info.value.context["planned_step_id"] == 1


def test_supervised_trainer_allows_optional_unhandled_scheduled_events() -> None:
    observed_events: list[str] = []
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        events={
            "checkpoint": (),
            "eval.forward": (),
            "training.logging": (
                _event(1, "training.logging", ("explicit_step",), required=False),
            ),
            "final": (),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=FakeRuntime([]),
        event_sink=lambda event: observed_events.append(event.event_type),
    )

    result = trainer.run()

    assert result.scheduled_event_counts["training.logging"] == 1
    assert "schedule.training.logging" in observed_events


def test_supervised_trainer_skips_backward_and_update_when_scalar_gate_is_unsafe() -> None:
    log: list[str] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=FakeLossRunner(log),
        runtime=FakeRuntime(log, unsafe_pre_steps={1}),
    )

    result = trainer.run()

    assert result.step_results[0].optimizer_update_status == "skipped_non_finite_scalar"
    assert not any(item.startswith("runtime.backward:1.0") for item in log)
    assert "runtime.post:1" not in log
    assert "runtime.optimizer:1" not in log
    assert "runtime.scheduler:1" in log
    assert log[-2:] == ["runtime.scheduler:1", "runtime.zero:1"]


def test_supervised_trainer_advances_scheduler_when_post_backward_gate_skips_update() -> None:
    log: list[str] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=FakeLossRunner(log),
        runtime=FakeRuntime(log, unsafe_post_steps={1}),
    )

    result = trainer.run()

    assert result.step_results[0].optimizer_update_status == "skipped_gradient_or_overflow"
    assert "runtime.backward:1.0:sync=True" in log
    assert "runtime.optimizer:1" not in log
    assert log[-3:] == ["runtime.post:1", "runtime.scheduler:1", "runtime.zero:1"]


def test_default_qwen_forward_uses_runtime_selected_forward_device(monkeypatch) -> None:
    log: list[str] = []
    observed_devices: list[str | None] = []

    def fake_build_qwen_forward_inputs(
        pack: Any,
        encoded_examples: tuple[Any, ...],
        position_inputs: Any,
        *,
        logits_to_keep_positions: tuple[int, ...] | None = None,
        device: str | None = None,
        fa2_branch_proof_policy: str | None = None,
    ) -> str:
        observed_devices.append(device)
        assert logits_to_keep_positions is None
        assert fa2_branch_proof_policy is None
        assert pack == "pack-0"
        assert encoded_examples == ("example-0",)
        assert position_inputs == "positions-0"
        return "prepared-inputs"

    def fake_run_qwen_forward(
        model: object,
        forward_inputs: str,
        **kwargs: Any,
    ) -> FakeForwardResult:
        assert forward_inputs == "prepared-inputs"
        assert kwargs["expected_vocab_size"] is None
        return FakeForwardResult(
            pack_index=0,
            logits=torch.zeros(1, 2, 3),
            receipt={"device": observed_devices[-1]},
        )

    monkeypatch.setattr(
        "src.training.supervised_trainer.build_qwen_forward_inputs",
        fake_build_qwen_forward_inputs,
    )
    monkeypatch.setattr(
        "src.training.supervised_trainer.run_qwen_forward",
        fake_run_qwen_forward,
    )

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1, log),
        loss_context_factory=_loss_context(log),
        loss_runner=FakeLossRunner(log),
        runtime=FakeRuntime(log, forward_device="cuda:7"),
    )

    result = trainer.run()

    assert observed_devices == ["cuda:7"]
    assert result.step_results[0].qwen_forward_receipts == ({"device": "cuda:7"},)


def test_supervised_trainer_forwards_with_runtime_owned_model() -> None:
    original_model = object()
    prepared_model = object()
    observed_models: list[object] = []

    def qwen_forward(model: object, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        observed_models.append(model)
        return FakeForwardResult(
            pack_index=int(str(micro_step.pack).split("-")[1]),
            logits=torch.zeros(1, 2, 3),
            receipt={"prepared": model is prepared_model},
        )

    trainer = SupervisedTrainer(
        model=original_model,
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1),
        qwen_forward=qwen_forward,
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=RuntimeWithPreparedModel([], prepared_model),
    )

    result = trainer.run()

    assert observed_models == [prepared_model]
    assert result.step_results[0].qwen_forward_receipts == ({"prepared": True},)


def test_supervised_trainer_runs_deepspeed_with_runtime_owned_model() -> None:
    original_model = object()
    prepared_deepspeed_model = object()
    runtime = RuntimeWithPreparedModel([], prepared_deepspeed_model)
    observed_models: list[object] = []

    def qwen_forward(observed_model: object, _micro_step: SupervisedMicroStep) -> FakeForwardResult:
        observed_models.append(observed_model)
        return FakeForwardResult(
            pack_index=0,
            logits=torch.zeros(1, 2, 3),
            receipt={"runtime_owned_model": observed_model is runtime.model},
        )

    trainer = SupervisedTrainer(
        model=original_model,
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1),
        qwen_forward=qwen_forward,
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=runtime,
    )

    result = trainer.run()

    assert result.completed_steps == 1
    assert observed_models == [runtime.model]
    assert result.step_results[0].qwen_forward_receipts == (
        {"runtime_owned_model": True},
    )


def test_supervised_training_result_contains_no_live_tensors() -> None:
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    result = trainer.run()

    assert not _contains_tensor(result.to_artifact_dict())
    assert not _contains_tensor(result.step_results[0].loss_bundle_artifact)


def test_training_package_exports_integration_boundary_types() -> None:
    from src.training import (
        LossContextFactory as PackageLossContextFactory,
        LossRunnerBoundary as PackageLossRunnerBoundary,
        QwenForwardFn as PackageQwenForwardFn,
        RuntimeBoundary as PackageRuntimeBoundary,
        ScheduledEventHandler as PackageScheduledEventHandler,
        TrainerEventSink as PackageTrainerEventSink,
    )

    assert PackageLossContextFactory is LossContextFactory
    assert PackageLossRunnerBoundary is LossRunnerBoundary
    assert PackageQwenForwardFn is QwenForwardFn
    assert PackageRuntimeBoundary is RuntimeBoundary
    assert PackageScheduledEventHandler is ScheduledEventHandler
    assert PackageTrainerEventSink is TrainerEventSink


def test_supervised_trainer_fails_if_pack_stream_cannot_fill_planned_window() -> None:
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=FakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        trainer.run()

    assert exc_info.value.code == "trainer.pack_stream_exhausted"
    assert exc_info.value.context["planned_step_id"] == 1
    assert exc_info.value.context["local_micro_step_index"] == 1


def test_supervised_trainer_streams_backward_before_next_forward_when_supported() -> None:
    log: list[str] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(2, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=StreamingFakeLossRunner(log),
        runtime=FakeRuntime(log),
    )

    result = trainer.run()

    assert result.completed_steps == 1
    assert result.consumed_micro_steps == 2
    assert result.step_results[0].loss_bundle_artifact["total_loss"] == 1.0
    assert result.step_results[0].loss_bundle_artifact.get("partial") is None
    assert (
        log.index("runtime.accumulation:False:enter")
        < log.index("forward:0")
        < log.index("runtime.accumulation:False:exit")
    )
    assert log.index("runtime.backward:0.5:sync=False") < log.index("forward:1")
    assert log == [
        "stream:0",
        "runtime.move:1:0",
        "stream:1",
        "runtime.move:1:1",
        "streaming.prepare:2",
        "runtime.accumulation:False:enter",
        "forward:0",
        "context:0",
        "streaming.loss:0",
        "runtime.pre:1",
        "runtime.backward:0.5:sync=False",
        "runtime.accumulation:False:exit",
        "runtime.accumulation:True:enter",
        "forward:1",
        "context:1",
        "streaming.loss:1",
        "runtime.pre:1",
        "runtime.backward:0.5:sync=True",
        "runtime.accumulation:True:exit",
        "runtime.post:1",
        "runtime.clip:1",
        "runtime.optimizer:1",
        "runtime.scheduler:1",
        "runtime.zero:1",
    ]


def test_streaming_multirank_loss_plan_uses_runtime_denominator_gatherer() -> None:
    log: list[str] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1, world_size=2),
        pack_stream=_micro_steps(1, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=StreamingGlobalDenominatorLossRunner(log),
        runtime=FakeRuntime(log, rank=0, world_size=2),
    )

    result = trainer.run()

    assert result.completed_steps == 1
    assert result.step_results[0].loss_bundle_artifact["diagnostics"][
        "denominator_scope"
    ] == "planned_step_global"
    assert "runtime.gather_denominators:1" in log
    assert "streaming.prepare_global:1:2:0:3" in log


def test_streaming_micro_step_events_expose_sync_gradients_boundary() -> None:
    events: list[SupervisedTrainerEvent] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(2),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        event_sink=events.append,
    )

    trainer.run()

    by_type = {
        event_type: [
            event.payload["sync_gradients"]
            for event in events
            if event.event_type == event_type
        ]
        for event_type in (
            "micro_step.forward",
            "micro_step.loss",
            "micro_step.pre_backward_gate",
        )
    }
    assert by_type == {
        "micro_step.forward": [False, True],
        "micro_step.loss": [False, True],
        "micro_step.pre_backward_gate": [False, True],
    }


def test_trainer_profile_sync_helper_is_exact_env_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trainer_module.torch.cuda,
        "synchronize",
        lambda device: calls.append(str(device)),
    )

    monkeypatch.delenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", raising=False)
    trainer_module._sync_device_if_requested(torch.device("cuda:3"))
    monkeypatch.setenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", "true")
    trainer_module._sync_device_if_requested(torch.device("cuda:3"))
    monkeypatch.setenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", "1")
    trainer_module._sync_device_if_requested(torch.device("cuda:3"))

    assert calls == ["cuda:3"]


def test_streaming_scalar_gate_partial_window_marks_metrics_unavailable() -> None:
    log: list[str] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(2, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=StreamingFakeLossRunner(log),
        runtime=FakeRuntime(log, unsafe_pre_call_indices={0}),
    )

    result = trainer.run()

    artifact = result.step_results[0].loss_bundle_artifact
    assert artifact["partial"] is True
    assert artifact["metrics"] == {}
    assert artifact["diagnostics"]["normalizer_scope"] == "partial_planned_step"
    assert artifact["diagnostics"]["processed_micro_step_count"] == 1
    assert artifact["diagnostics"]["planned_micro_step_count"] == 2
    assert result.step_results[0].micro_step_count == 1
    assert result.step_results[0].optimizer_update_status == "skipped_non_finite_scalar"
    assert not any(item.startswith("runtime.backward:0.5") for item in log)
    assert "runtime.accumulation:False:enter" in log
    assert "runtime.accumulation:False:exit" in log
    assert "forward:1" not in log
    assert log[-2:] == ["runtime.scheduler:1", "runtime.zero:1"]


@dataclass(frozen=True)
class FakeForwardResult:
    pack_index: int
    logits: torch.Tensor
    receipt: Any


@dataclass(frozen=True)
class FakeLossBundle:
    total_loss: torch.Tensor

    def to_artifact_dict(self) -> dict[str, Any]:
        return {"total_loss": float(self.total_loss.detach().cpu())}


class FakeLossRunner:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def compute(self, contexts: tuple[Any, ...]) -> FakeLossBundle:
        self.log.append(f"loss:{len(contexts)}")
        return FakeLossBundle(total_loss=torch.tensor(1.0, requires_grad=True))


class StreamingFakeLossRunner(FakeLossRunner):
    def compute(self, contexts: tuple[Any, ...]) -> FakeLossBundle:
        raise AssertionError("streaming trainer path must not retain all contexts")

    def prepare_planned_step(self, micro_steps: tuple[SupervisedMicroStep, ...]) -> dict[str, int]:
        self.log.append(f"streaming.prepare:{len(micro_steps)}")
        return {"micro_step_count": len(micro_steps)}

    def compute_micro_step(
        self,
        context: Any,
        plan: dict[str, int],
        *,
        local_micro_step_index: int,
    ) -> FakeLossBundle:
        del context
        self.log.append(f"streaming.loss:{local_micro_step_index}")
        loss = 1.0 / float(plan["micro_step_count"])
        return FakeLossBundle(total_loss=torch.tensor(loss, requires_grad=True))

    def finalize_planned_step(
        self,
        micro_loss_artifacts: tuple[dict[str, Any], ...],
        plan: dict[str, int],
    ) -> dict[str, float]:
        return {
            "total_loss": sum(float(item["total_loss"]) for item in micro_loss_artifacts),
            "metrics": {"loss/total": sum(float(item["total_loss"]) for item in micro_loss_artifacts)},
            "diagnostics": {"normalizer_scope": "planned_step_streaming"},
        }


class StreamingGlobalDenominatorLossRunner(StreamingFakeLossRunner):
    def prepare_planned_step(
        self,
        micro_steps: tuple[SupervisedMicroStep, ...],
        *,
        denominator_gatherer,
        world_size: int,
        rank: int,
    ) -> dict[str, Any]:
        payload = {
            "base_ce": {
                "term_name": "base_ce",
                "denominator_scope": "planned_step",
                "eligible_segment_count": 1,
                "selected_atom_count": 1,
                "skipped_segment_count": 0,
                "context_count": 1,
            },
            "token_type_gate": {
                "term_name": "token_type_gate",
                "denominator_scope": "planned_step",
                "eligible_segment_count": 1,
                "selected_atom_count": 1,
                "skipped_segment_count": 0,
                "context_count": 1,
            },
        }
        gathered = tuple(denominator_gatherer(payload))
        eligible = sum(
            int(rank_payload["base_ce"]["eligible_segment_count"])
            for rank_payload in gathered
        )
        self.log.append(
            f"streaming.prepare_global:{len(micro_steps)}:{world_size}:{rank}:{eligible}"
        )
        return {
            "micro_step_count": len(micro_steps),
            "denominator_scope": "planned_step_global",
        }

    def finalize_planned_step(
        self,
        micro_loss_artifacts: tuple[dict[str, Any], ...],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        artifact = super().finalize_planned_step(micro_loss_artifacts, plan)
        artifact["diagnostics"]["denominator_scope"] = plan["denominator_scope"]
        return artifact


class FakeRuntime:
    def __init__(
        self,
        log: list[str],
        *,
        unsafe_pre_steps: set[int] | None = None,
        unsafe_pre_call_indices: set[int] | None = None,
        unsafe_post_steps: set[int] | None = None,
        forward_device: str | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.log = log
        self.unsafe_pre_steps = unsafe_pre_steps or set()
        self.unsafe_pre_call_indices = unsafe_pre_call_indices or set()
        self.unsafe_post_steps = unsafe_post_steps or set()
        self.forward_device = forward_device
        self.pre_call_count = 0
        self.rank = int(rank)
        self.world_size = int(world_size)

    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        self.log.append(f"runtime.move:{planned_step_id}:{local_micro_step_index}")
        if self.forward_device is not None:
            return replace(micro_step, forward_device=self.forward_device)
        return micro_step

    def pre_backward(
        self,
        bundle: FakeLossBundle,
        *,
        planned_step_id: int,
    ) -> GateDecision:
        self.log.append(f"runtime.pre:{planned_step_id}")
        call_index = self.pre_call_count
        self.pre_call_count += 1
        if planned_step_id in self.unsafe_pre_steps or call_index in self.unsafe_pre_call_indices:
            return _gate(
                planned_step_id,
                stage="pre_backward_scalar",
                optimizer_update_status="skipped_non_finite_scalar",
                finite_status="non_finite",
                backward=False,
                optimizer=False,
                clear=True,
            )
        return _gate(
            planned_step_id,
            stage="pre_backward_scalar",
            optimizer_update_status="pending_backward",
            finite_status="finite",
            backward=True,
            optimizer=False,
            clear=False,
        )

    def backward(
        self,
        loss: torch.Tensor,
        *,
        planned_step_id: int,
        sync_gradients: bool = True,
    ) -> None:
        self.log.append(
            f"runtime.backward:{float(loss.detach().cpu())}:sync={sync_gradients}"
        )

    @contextmanager
    def accumulation_context(self, *, sync_gradients: bool):
        self.log.append(f"runtime.accumulation:{sync_gradients}:enter")
        try:
            yield
        finally:
            self.log.append(f"runtime.accumulation:{sync_gradients}:exit")

    def post_backward(self, *, planned_step_id: int) -> GateDecision:
        self.log.append(f"runtime.post:{planned_step_id}")
        if planned_step_id in self.unsafe_post_steps:
            return _gate(
                planned_step_id,
                stage="post_backward_gradient",
                optimizer_update_status="skipped_gradient_or_overflow",
                finite_status="non_finite",
                backward=False,
                optimizer=False,
                clear=True,
            )
        return _gate(
            planned_step_id,
            stage="post_backward_gradient",
            optimizer_update_status="ready_to_step",
            finite_status="finite",
            backward=False,
            optimizer=True,
            clear=False,
        )

    def clip_gradients(self, *, planned_step_id: int) -> None:
        self.log.append(f"runtime.clip:{planned_step_id}")

    def optimizer_step(self, *, planned_step_id: int) -> None:
        self.log.append(f"runtime.optimizer:{planned_step_id}")

    def scheduler_step(self, *, planned_step_id: int) -> None:
        self.log.append(f"runtime.scheduler:{planned_step_id}")
        return {
            "scheduler_step_count": planned_step_id,
            "learning_rates": [{"group_index": 0, "lr": 0.01}],
        }

    def zero_gradients(self, *, planned_step_id: int) -> None:
        self.log.append(f"runtime.zero:{planned_step_id}")

    def gather_loss_denominators(
        self,
        denominators: dict[str, dict[str, Any]],
        *,
        planned_step_id: int,
    ) -> tuple[dict[str, dict[str, Any]], ...]:
        self.log.append(f"runtime.gather_denominators:{planned_step_id}")
        peer = {
            name: {
                **dict(payload),
                "eligible_segment_count": 2,
                "selected_atom_count": 2,
                "skipped_segment_count": 0,
                "context_count": 2,
            }
            for name, payload in denominators.items()
        }
        return (denominators, peer)


class RuntimeWithPreparedModel(FakeRuntime):
    def __init__(self, log: list[str], model: object) -> None:
        super().__init__(log)
        self.model = model


def _micro_steps(count: int, log: list[str] | None = None):
    for index in range(count):
        if log is not None:
            log.append(f"stream:{index}")
        yield SupervisedMicroStep(
            pack=f"pack-{index}",
            encoded_examples=(f"example-{index}",),
            position_inputs=f"positions-{index}",
            token_sequence=f"tokens-{index}",
            vocab_groups=f"vocab-{index}",
            metadata={},
        )


def _forward(log: list[str]):
    def run(_model: object, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        pack_index = int(str(micro_step.pack).split("-")[1])
        log.append(f"forward:{pack_index}")
        return FakeForwardResult(
            pack_index=pack_index,
            logits=torch.zeros(1, 2, 3),
            receipt={"pack_index": pack_index},
        )

    return run


def _loss_context(log: list[str]):
    def build(
        micro_step: SupervisedMicroStep,
        forward_result: FakeForwardResult,
    ) -> str:
        pack_index = int(str(micro_step.pack).split("-")[1])
        assert forward_result.pack_index == pack_index
        log.append(f"context:{pack_index}")
        return f"context-{pack_index}"

    return build


def _scheduled_tuple(event: ScheduledTrainerEvent) -> tuple[str, int, tuple[str, ...], str]:
    return (
        event.scheduled_event.event,
        event.scheduled_event.planned_step_id,
        event.scheduled_event.trigger_reasons,
        event.step_result.optimizer_update_status,
    )


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _schedule(
    *,
    resolved_max_steps: int,
    grad_accum_steps: int,
    world_size: int = 1,
    events: dict[str, tuple[StepScheduleEvent, ...]] | None = None,
) -> ResolvedStepSchedule:
    return ResolvedStepSchedule(
        resolved_max_steps=resolved_max_steps,
        packs_per_epoch=100,
        requested_pack_presentations=resolved_max_steps * grad_accum_steps,
        actual_pack_presentations=resolved_max_steps * grad_accum_steps,
        tail_fill_pack_count=0,
        runtime_batch=RuntimeBatchResolution(
            world_size=world_size,
            effective_batch_size=grad_accum_steps * world_size,
            resolved_grad_accum_steps=grad_accum_steps,
        ),
        events=events
        or {
            "checkpoint": (),
            "eval.forward": (),
            "training.logging": (),
            "final": (),
        },
    )


def _event(
    planned_step_id: int,
    event: str,
    trigger_reasons: tuple[str, ...],
    *,
    required: bool = False,
) -> StepScheduleEvent:
    return StepScheduleEvent(
        planned_step_id=planned_step_id,
        event=event,
        trigger_reasons=trigger_reasons,
        source_config_path=None,
        deduped_from=(),
        required=required,
    )


def _gate(
    planned_step_id: int,
    *,
    stage: str,
    optimizer_update_status: str,
    finite_status: str,
    backward: bool,
    optimizer: bool,
    clear: bool,
) -> GateDecision:
    return GateDecision(
        stage=stage,
        planned_step_id=planned_step_id,
        world_size=1,
        ranks=(0,),
        all_ranks_safe=finite_status == "finite",
        should_call_backward=backward,
        should_call_optimizer_step=optimizer,
        should_clear_gradients=clear,
        optimizer_update_status=optimizer_update_status,
        finite_status=finite_status,
        reason_codes=(),
        rank_diagnostics=(),
        diagnostics={},
    )
