from __future__ import annotations

from contextlib import contextmanager
from dataclasses import MISSING, fields, replace
from dataclasses import dataclass
import inspect
from typing import Any

import pytest
import torch

import src.training.supervised_trainer as trainer_module
from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution
from src.losses import LossRunner, TokenVocabularyGroups
from src.packing.planner import PackedSegment
from src.runtime import GateDecision
from src.supervision import TokenAtom, TokenSequence
from src.training.schedule import ResolvedStepSchedule, StepScheduleEvent
from src.training.supervised_trainer import (
    CompletedStepObservation,
    LossContextFactory,
    LossRunnerBoundary,
    ScheduledStepHandler,
    SupervisedMicroStep,
    SupervisedTrainer,
    QwenForwardFn,
    RuntimeBoundary,
)


def test_supervised_trainer_orchestrates_accumulation_and_runtime_boundaries() -> None:
    log: list[str] = []
    observations: list[CompletedStepObservation] = []
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
        loss_runner=StreamingFakeLossRunner(log),
        runtime=FakeRuntime(log),
        on_completed_step=observations.append,
    )

    result = trainer.run()

    assert result.completed_steps == 2
    assert result.consumed_micro_steps == 4
    assert [item.planned_step_id for item in observations] == [1, 2]
    assert observations[0].micro_step_count == 2
    assert observations[0].optimizer_update_status == "applied"
    assert observations[0].loss_bundle_artifact["total_loss"] == 1.0
    assert observations[0].scheduler_artifact == {
        "scheduler_step_count": 1,
        "learning_rates": [{"group_index": 0, "lr": 0.01}],
    }
    assert observations[0].to_artifact_dict()["scheduler"] == {
        "scheduler_step_count": 1,
        "learning_rates": [{"group_index": 0, "lr": 0.01}],
    }
    assert not hasattr(observations[0], "loss_bundle")
    assert result.latest_observation is observations[-1]
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
        "stream:2",
        "runtime.move:2:0",
        "stream:3",
        "runtime.move:2:1",
        "streaming.prepare:2",
        "runtime.accumulation:False:enter",
        "forward:2",
        "context:2",
        "streaming.loss:0",
        "runtime.pre:2",
        "runtime.backward:0.5:sync=False",
        "runtime.accumulation:False:exit",
        "runtime.accumulation:True:enter",
        "forward:3",
        "context:3",
        "streaming.loss:1",
        "runtime.pre:2",
        "runtime.backward:0.5:sync=True",
        "runtime.accumulation:True:exit",
        "runtime.post:2",
        "runtime.clip:2",
        "runtime.optimizer:2",
        "runtime.scheduler:2",
        "runtime.zero:2",
    ]


def test_supervised_trainer_triggers_scheduled_eval_checkpoint_and_final_events() -> (
    None
):
    scheduled_calls: list[tuple[str, int, tuple[str, ...], str]] = []
    schedule = _schedule(
        resolved_max_steps=2,
        grad_accum_steps=1,
        events={
            "eval.forward": (_event(1, "eval.forward", ("explicit_step",)),),
            "checkpoint": (_event(2, "checkpoint", ("save_final",)),),
            "final": (_event(2, "final", ("final",), required=True),),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(2),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        on_eval=lambda event, observation: scheduled_calls.append(
            _scheduled_tuple(event, observation)
        ),
        on_checkpoint=lambda event, observation: scheduled_calls.append(
            _scheduled_tuple(event, observation)
        ),
        on_final=lambda event, observation: scheduled_calls.append(
            _scheduled_tuple(event, observation)
        ),
    )

    result = trainer.run()

    assert result.completed_steps == 2
    assert result.scheduled_event_counts == {
        "checkpoint": 1,
        "eval.forward": 1,
        "final": 1,
    }
    assert scheduled_calls == [
        ("eval.forward", 1, ("explicit_step",), "applied"),
        ("checkpoint", 2, ("save_final",), "applied"),
        ("final", 2, ("final",), "applied"),
    ]


def test_supervised_trainer_resume_starts_at_exact_global_planned_step() -> None:
    observations: list[int] = []
    scheduled_calls: list[tuple[str, int]] = []
    schedule = _schedule(
        resolved_max_steps=3,
        grad_accum_steps=1,
        events={
            "eval.forward": (_event(1, "eval.forward", ("before_resume",)),),
            "checkpoint": (_event(2, "checkpoint", ("resume_segment",)),),
            "final": (_event(3, "final", ("final",), required=True),),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(2),
        start_planned_step_id=2,
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        on_completed_step=lambda observation: observations.append(
            observation.planned_step_id
        ),
        on_eval=lambda event, _observation: scheduled_calls.append(
            (event.event, event.planned_step_id)
        ),
        on_checkpoint=lambda event, _observation: scheduled_calls.append(
            (event.event, event.planned_step_id)
        ),
        on_final=lambda event, _observation: scheduled_calls.append(
            (event.event, event.planned_step_id)
        ),
    )

    result = trainer.run()

    assert observations == [2, 3]
    assert result.completed_steps == 3
    assert result.consumed_micro_steps == 2
    assert result.scheduled_event_counts == {
        "checkpoint": 1,
        "eval.forward": 0,
        "final": 1,
    }
    assert scheduled_calls == [("checkpoint", 2), ("final", 3)]


@pytest.mark.parametrize("start_planned_step_id", (0, 4))
def test_supervised_trainer_rejects_resume_step_outside_schedule(
    start_planned_step_id: int,
) -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        SupervisedTrainer(
            model=object(),
            schedule=_schedule(resolved_max_steps=3, grad_accum_steps=1),
            pack_stream=_micro_steps(3),
            start_planned_step_id=start_planned_step_id,
            qwen_forward=_forward([]),
            loss_context_factory=_loss_context([]),
            loss_runner=StreamingFakeLossRunner([]),
            runtime=FakeRuntime([]),
        )

    assert exc_info.value.code == "trainer.start_planned_step_invalid"


def test_supervised_trainer_runs_same_step_eval_before_checkpoint() -> None:
    scheduled_calls: list[str] = []
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        events={
            "checkpoint": (_event(1, "checkpoint", ("every_fraction:1.0",)),),
            "eval.forward": (_event(1, "eval.forward", ("explicit_step",)),),
            "final": (_event(1, "final", ("final",), required=True),),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        on_eval=lambda event, _observation: scheduled_calls.append(event.event),
        on_checkpoint=lambda event, _observation: scheduled_calls.append(event.event),
        on_final=lambda event, _observation: scheduled_calls.append(event.event),
    )

    result = trainer.run()

    assert result.scheduled_event_counts == {
        "checkpoint": 1,
        "eval.forward": 1,
        "final": 1,
    }
    assert scheduled_calls == ["eval.forward", "checkpoint", "final"]


def test_completed_callback_precedes_direct_same_step_handlers() -> None:
    calls: list[tuple[str, int]] = []
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        events={
            "eval.forward": (_event(1, "eval.forward", ("explicit_step",)),),
            "checkpoint": (_event(1, "checkpoint", ("save_final",)),),
            "final": (_event(1, "final", ("final",), required=True),),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        on_completed_step=lambda observation: calls.append(
            ("completed", observation.planned_step_id)
        ),
        on_eval=lambda _event, observation: calls.append(
            ("eval", observation.planned_step_id)
        ),
        on_checkpoint=lambda _event, observation: calls.append(
            ("checkpoint", observation.planned_step_id)
        ),
        on_final=lambda _event, observation: calls.append(
            ("final", observation.planned_step_id)
        ),
    )

    trainer.run()

    assert calls == [
        ("completed", 1),
        ("eval", 1),
        ("checkpoint", 1),
        ("final", 1),
    ]


@pytest.mark.parametrize("step_count", [1, 25])
def test_supervised_training_result_is_bounded_independent_of_step_count(
    step_count: int,
) -> None:
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=step_count, grad_accum_steps=1),
        pack_stream=_micro_steps(step_count),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    result = trainer.run()
    artifact = result.to_artifact_dict()

    assert result.completed_steps == step_count
    assert result.latest_observation.planned_step_id == step_count
    assert "step_results" not in artifact
    assert set(artifact) == {
        "completed_steps",
        "consumed_micro_steps",
        "scheduled_event_counts",
        "latest_observation",
    }


def test_supervised_trainer_rejects_required_scheduled_event_without_handler() -> None:
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        events={
            "checkpoint": (),
            "eval.forward": (),
            "final": (_event(1, "final", ("final",), required=True),),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        trainer.run()

    assert exc_info.value.code == "trainer.required_event_handler_missing"
    assert exc_info.value.context["event"] == "final"
    assert exc_info.value.context["planned_step_id"] == 1


def test_supervised_trainer_allows_optional_unhandled_scheduled_events() -> None:
    schedule = _schedule(
        resolved_max_steps=1,
        grad_accum_steps=1,
        events={
            "checkpoint": (),
            "eval.forward": (
                _event(1, "eval.forward", ("explicit_step",), required=False),
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
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    result = trainer.run()

    assert result.scheduled_event_counts["eval.forward"] == 1


def test_supervised_trainer_has_no_generic_scheduled_event_dispatch_residue() -> None:
    trainer_source = inspect.getsource(trainer_module.SupervisedTrainer)

    assert "_scheduled" + "_handler" not in trainer_source
    assert "event_" + "name" not in trainer_source
    assert "scheduled_event" + "_handlers" not in trainer_source
    assert "training." + "logging" not in inspect.getsource(trainer_module)


def test_supervised_trainer_skips_backward_and_update_when_scalar_gate_is_unsafe() -> (
    None
):
    log: list[str] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=StreamingFakeLossRunner(log),
        runtime=FakeRuntime(log, unsafe_pre_steps={1}),
    )

    result = trainer.run()

    assert (
        result.latest_observation.optimizer_update_status == "skipped_non_finite_scalar"
    )
    assert not any(item.startswith("runtime.backward:1.0") for item in log)
    assert "runtime.post:1" not in log
    assert "runtime.optimizer:1" not in log
    assert "runtime.scheduler:1" in log
    assert log[-2:] == ["runtime.scheduler:1", "runtime.zero:1"]


def test_supervised_trainer_advances_scheduler_when_post_backward_gate_skips_update() -> (
    None
):
    log: list[str] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1, log),
        qwen_forward=_forward(log),
        loss_context_factory=_loss_context(log),
        loss_runner=StreamingFakeLossRunner(log),
        runtime=FakeRuntime(log, unsafe_post_steps={1}),
    )

    result = trainer.run()

    assert (
        result.latest_observation.optimizer_update_status
        == "skipped_gradient_or_overflow"
    )
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
        loss_runner=StreamingFakeLossRunner(log),
        runtime=FakeRuntime(log, forward_device="cuda:7"),
    )

    result = trainer.run()

    assert observed_devices == ["cuda:7"]
    assert result.latest_observation.planned_step_id == 1


def test_default_qwen_forward_keeps_exactly_the_token_sequence_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Qwen ``logits_to_keep`` input is the domain selection, unchanged."""

    sequence = _token_sequence(0, target_positions=(3, 1, 2))
    micro_step = replace(next(_micro_steps(1)), token_sequence=sequence)
    observed_positions: list[tuple[int, ...] | None] = []

    def fake_build_qwen_forward_inputs(*args: Any, **kwargs: Any) -> str:
        del args
        observed_positions.append(kwargs["logits_to_keep_positions"])
        return "prepared-inputs"

    def fake_run_qwen_forward(
        _model: object, _forward_inputs: str, **_kwargs: Any
    ) -> FakeForwardResult:
        return FakeForwardResult(
            pack_index=0, logits=torch.zeros(1, 2, 3), receipt={}
        )

    monkeypatch.setattr(
        trainer_module, "build_qwen_forward_inputs", fake_build_qwen_forward_inputs
    )
    monkeypatch.setattr(trainer_module, "run_qwen_forward", fake_run_qwen_forward)

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=iter((micro_step,)),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )
    trainer.run()

    assert observed_positions == [(0, 1, 2)]
    assert observed_positions == [sequence.causal_logits_positions()]


def test_supervised_trainer_forwards_with_runtime_owned_model() -> None:
    original_model = object()
    prepared_model = object()
    observed_models: list[object] = []

    def qwen_forward(
        model: object, micro_step: SupervisedMicroStep
    ) -> FakeForwardResult:
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
        loss_runner=StreamingFakeLossRunner([]),
        runtime=RuntimeWithPreparedModel([], prepared_model),
    )

    result = trainer.run()

    assert observed_models == [prepared_model]
    assert result.latest_observation.planned_step_id == 1


def test_supervised_trainer_runs_with_accelerate_prepared_runtime_model() -> None:
    original_model = object()
    prepared_model = object()
    runtime = RuntimeWithPreparedModel([], prepared_model)
    observed_models: list[object] = []

    def qwen_forward(
        observed_model: object, _micro_step: SupervisedMicroStep
    ) -> FakeForwardResult:
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
        loss_runner=StreamingFakeLossRunner([]),
        runtime=runtime,
    )

    result = trainer.run()

    assert result.completed_steps == 1
    assert observed_models == [runtime.model]
    assert result.latest_observation.planned_step_id == 1


def test_supervised_training_result_contains_no_live_tensors() -> None:
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    result = trainer.run()

    assert not _contains_tensor(result.to_artifact_dict())
    assert not _contains_tensor(result.latest_observation.loss_bundle_artifact)


def test_training_package_exports_integration_boundary_types() -> None:
    from src.training import (
        LossContextFactory as PackageLossContextFactory,
        LossRunnerBoundary as PackageLossRunnerBoundary,
        QwenForwardFn as PackageQwenForwardFn,
        RuntimeBoundary as PackageRuntimeBoundary,
        ScheduledStepHandler as PackageScheduledStepHandler,
    )

    assert PackageLossContextFactory is LossContextFactory
    assert PackageLossRunnerBoundary is LossRunnerBoundary
    assert PackageQwenForwardFn is QwenForwardFn
    assert PackageRuntimeBoundary is RuntimeBoundary
    assert PackageScheduledStepHandler is ScheduledStepHandler


def test_supervised_trainer_fails_if_pack_stream_cannot_fill_planned_window() -> None:
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(1),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        trainer.run()

    assert exc_info.value.code == "trainer.pack_stream_exhausted"
    assert exc_info.value.context["planned_step_id"] == 1
    assert exc_info.value.context["local_micro_step_index"] == 1


def test_supervised_trainer_streams_backward_before_next_forward_when_supported() -> (
    None
):
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
    assert result.latest_observation.loss_bundle_artifact["total_loss"] == 1.0
    assert result.latest_observation.loss_bundle_artifact.get("partial") is None
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


def test_supervised_trainer_zero_weight_gate_passes_base_only_objective() -> None:
    projection = _real_loss_projection()
    forward_calls = 0

    def qwen_forward(
        model: torch.nn.Module,
        _micro_step: SupervisedMicroStep,
    ) -> RealLossForwardResult:
        nonlocal forward_calls
        forward_calls += 1
        features = torch.tensor(
            (((1.0, -0.5), (0.25, 2.0)),),
            dtype=torch.float32,
        )
        return RealLossForwardResult(
            logits=model(features),
            receipt={"pack_index": 0},
        )

    runtime = RealLossRuntime(
        [],
        parameters=tuple(projection.parameters()),
        gate_weight=0.0,
    )
    trainer = SupervisedTrainer(
        model=projection,
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=(_real_loss_micro_step(),),
        qwen_forward=qwen_forward,
        loss_runner=LossRunner(
            base_ce_weight=1.0,
            token_type_gate_weight=0.0,
            token_type_gate_groups=("desc_text",),
        ),
        runtime=runtime,
    )

    result = trainer.run()

    assert forward_calls == 1
    assert result.completed_steps == 1
    assert result.consumed_micro_steps == 1
    assert result.latest_observation.micro_step_count == 1
    assert runtime.bundle is not None
    assert runtime.backward_loss is runtime.bundle.total_loss
    base_term = runtime.bundle.term_by_name("base_ce")
    gate_term = runtime.bundle.term_by_name("token_type_gate")
    assert torch.equal(runtime.backward_loss, base_term.weighted_loss)
    assert torch.equal(runtime.backward_loss, runtime.reference_objective)
    assert gate_term.raw_loss.dtype == torch.float32
    assert not gate_term.raw_loss.requires_grad
    assert gate_term.raw_loss.grad_fn is None
    assert not gate_term.weighted_loss.requires_grad
    assert gate_term.weighted_loss.grad_fn is None
    assert not gate_term.segment_mean_numerator.requires_grad
    assert gate_term.segment_mean_numerator.grad_fn is None
    assert not gate_term.token_weighted_diagnostic.requires_grad
    assert gate_term.token_weighted_diagnostic.grad_fn is None
    for parameter, expected_gradient in zip(
        projection.parameters(),
        runtime.reference_gradients,
        strict=True,
    ):
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, expected_gradient)


def test_supervised_trainer_nonzero_gate_keeps_differentiable_objective() -> None:
    projection = _real_loss_projection()
    forward_calls = 0

    def qwen_forward(
        model: torch.nn.Module,
        _micro_step: SupervisedMicroStep,
    ) -> RealLossForwardResult:
        nonlocal forward_calls
        forward_calls += 1
        features = torch.tensor(
            (((0.75, -1.25), (1.5, 0.5)),),
            dtype=torch.float32,
        )
        return RealLossForwardResult(
            logits=model(features),
            receipt={"pack_index": 0},
        )

    runtime = RealLossRuntime(
        [],
        parameters=tuple(projection.parameters()),
        gate_weight=0.4,
    )
    trainer = SupervisedTrainer(
        model=projection,
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=(_real_loss_micro_step(),),
        qwen_forward=qwen_forward,
        loss_runner=LossRunner(
            base_ce_weight=1.0,
            token_type_gate_weight=0.4,
            token_type_gate_groups=("desc_text",),
        ),
        runtime=runtime,
    )

    result = trainer.run()

    assert forward_calls == 1
    assert result.consumed_micro_steps == 1
    assert result.latest_observation.micro_step_count == 1
    assert runtime.bundle is not None
    assert runtime.backward_loss is runtime.bundle.total_loss
    gate_term = runtime.bundle.term_by_name("token_type_gate")
    assert gate_term.raw_loss.requires_grad
    assert gate_term.raw_loss.grad_fn is not None
    assert gate_term.weighted_loss.requires_grad
    assert gate_term.weighted_loss.grad_fn is not None
    assert torch.equal(runtime.backward_loss, runtime.reference_objective)
    for parameter, expected_gradient in zip(
        projection.parameters(),
        runtime.reference_gradients,
        strict=True,
    ):
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, expected_gradient)


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
    assert (
        result.latest_observation.loss_bundle_artifact["diagnostics"][
            "denominator_scope"
        ]
        == "planned_step_global"
    )
    assert "runtime.gather_denominators:1" in log
    assert "streaming.prepare_global:1:2:0:3" in log


def test_streaming_completion_callback_does_not_expose_micro_or_gate_events() -> None:
    observations: list[CompletedStepObservation] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(2),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        on_completed_step=observations.append,
    )

    trainer.run()

    assert len(observations) == 1
    assert observations[0].micro_step_count == 2


def test_streaming_step_records_input_build_seconds_from_forward_receipt() -> None:
    def qwen_forward(
        _model: object, micro_step: SupervisedMicroStep
    ) -> FakeForwardResult:
        pack_index = int(str(micro_step.pack).split("-")[1])
        return FakeForwardResult(
            pack_index=pack_index,
            logits=torch.zeros(1, 2, 3),
            receipt={"timings_ns": {"total_build_inputs_ns": 250_000_000}},
        )

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(2),
        qwen_forward=qwen_forward,
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    result = trainer.run()

    observation = result.latest_observation
    assert observation is not None
    # Two micro-steps, each contributing 0.25s of honest input-build time
    # from the existing Qwen forward receipt (never invented GPU time).
    assert observation.input_build_seconds == pytest.approx(0.5)
    assert observation.input_wait_seconds == 0.0
    assert observation.step_duration_seconds >= 0.0
    assert observation.to_artifact_dict()["input_build_seconds"] == pytest.approx(0.5)


def test_streaming_step_duration_excludes_completed_step_and_scheduled_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A monotonically incrementing fake clock (one fixed tick unit per call,
    # regardless of caller) is shared by the trainer's own boundary timing
    # AND by the completed-step/scheduled handlers below, which each
    # explicitly consume one tick of their own. If the compute/optimizer
    # boundary wrongly extended to include handler time, the measured
    # `step_duration_seconds` would inflate to a larger, precisely
    # predictable wrong value (asserted against explicitly below) rather
    # than merely raising on a finite iterator.
    tick_count = 0

    def fake_monotonic() -> float:
        nonlocal tick_count
        value = float(tick_count)
        tick_count += 1
        return value

    monkeypatch.setattr(trainer_module.time, "monotonic", fake_monotonic)

    observations: list[CompletedStepObservation] = []
    calls: list[str] = []

    def on_completed_step(observation: CompletedStepObservation) -> None:
        trainer_module.time.monotonic()  # consumes a tick; must not count toward the boundary
        observations.append(observation)
        calls.append(f"completed:{observation.planned_step_id}")

    def on_eval(
        event: StepScheduleEvent, observation: CompletedStepObservation
    ) -> None:
        del event, observation
        trainer_module.time.monotonic()  # consumes a tick; must not count toward the boundary
        calls.append("eval")

    def on_checkpoint(
        event: StepScheduleEvent, observation: CompletedStepObservation
    ) -> None:
        del event, observation
        trainer_module.time.monotonic()  # consumes a tick; must not count toward the boundary
        calls.append("checkpoint")

    schedule = _schedule(
        resolved_max_steps=2,
        grad_accum_steps=1,
        events={
            "eval.forward": (_event(1, "eval.forward", ("explicit_step",)),),
            "checkpoint": (_event(2, "checkpoint", ("save_final",)),),
            "final": (),
        },
    )
    trainer = SupervisedTrainer(
        model=object(),
        schedule=schedule,
        pack_stream=_micro_steps(2),
        qwen_forward=_forward([]),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        on_completed_step=on_completed_step,
        on_eval=on_eval,
        on_checkpoint=on_checkpoint,
    )

    trainer.run()

    assert calls == ["completed:1", "eval", "completed:2", "checkpoint"]
    # Each step's own compute/optimizer boundary consumes exactly the two
    # ticks bracketing it (start, end): a fixed 1-tick duration per step,
    # regardless of how many ticks the intervening handlers consumed.
    assert [obs.step_duration_seconds for obs in observations] == pytest.approx(
        [1.0, 1.0]
    )
    # Explicitly rule out the boundary-inclusive-of-handlers bug: had the
    # timer wrongly captured its end tick after on_completed_step/on_eval
    # ran, step 1's duration would read 3.0 (3 intervening ticks), not 1.0.
    assert observations[0].step_duration_seconds != pytest.approx(3.0)


def test_streaming_planned_step_uses_forward_input_provider_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeForwardInputProvider(wait_seconds_sequence=[0.01, 0.02])
    observed_forward_inputs: list[str] = []

    def fake_run_qwen_forward(
        model: object, forward_inputs: str, **kwargs: Any
    ) -> FakeForwardResult:
        del model, kwargs
        observed_forward_inputs.append(forward_inputs)
        pack_index = int(forward_inputs.split("-")[-1])
        return FakeForwardResult(
            pack_index=pack_index,
            logits=torch.zeros(1, 2, 3),
            receipt={"timings_ns": {"total_build_inputs_ns": 100_000_000}},
        )

    monkeypatch.setattr(trainer_module, "run_qwen_forward", fake_run_qwen_forward)

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(2),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        forward_input_provider=provider,
    )

    result = trainer.run()

    assert provider.calls == ["begin", "take:0", "take:1", "end"]
    assert provider.begin_args[0][0] == 1
    assert observed_forward_inputs == ["forward-inputs-0", "forward-inputs-1"]
    observation = result.latest_observation
    assert observation is not None
    assert observation.input_wait_seconds == pytest.approx(0.03)
    assert observation.input_build_seconds == pytest.approx(0.2)
    assert provider.closed is False  # trainer does not own close(); pipeline does


def test_first_micro_step_proof_is_a_run_local_one_shot_for_replayed_step() -> None:
    step = replace(
        next(_micro_steps(1)),
        capture_fa2_branch=True,
        require_fa2_branch_proof=True,
        fa2_branch_proof_policy="first_micro_step",
    )
    observed_controls: list[tuple[bool, bool]] = []

    def qwen_forward(
        _model: object,
        micro_step: SupervisedMicroStep,
    ) -> FakeForwardResult:
        observed_controls.append(
            (micro_step.capture_fa2_branch, micro_step.require_fa2_branch_proof)
        )
        return FakeForwardResult(
            pack_index=0,
            logits=torch.zeros(1, 2, 3),
            receipt=FakeProofReceipt(
                fa2_branch_proof=(object() if micro_step.capture_fa2_branch else None)
            ),
        )

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=2, grad_accum_steps=1),
        pack_stream=(step, step),
        qwen_forward=qwen_forward,
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    trainer.run()

    assert observed_controls == [(True, True), (False, False)]
    assert trainer._fa2_first_micro_step_admitted is True

    trainer.pack_stream = iter((step, step))
    trainer.run()

    assert observed_controls == [
        (True, True),
        (False, False),
        (True, True),
        (False, False),
    ]


def test_first_micro_step_proof_missing_fails_closed_without_admission() -> None:
    step = replace(
        next(_micro_steps(1)),
        capture_fa2_branch=True,
        require_fa2_branch_proof=True,
        fa2_branch_proof_policy="first_micro_step",
    )

    def qwen_forward(
        _model: object,
        _micro_step: SupervisedMicroStep,
    ) -> FakeForwardResult:
        return FakeForwardResult(
            pack_index=0,
            logits=torch.zeros(1, 2, 3),
            receipt=FakeProofReceipt(fa2_branch_proof=None),
        )

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=(step,),
        qwen_forward=qwen_forward,
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        trainer.run()

    assert exc_info.value.code == "trainer.fa2_first_micro_step_proof_missing"
    assert exc_info.value.context == {
        "planned_step_id": 1,
        "local_micro_step_index": 0,
        "policy": "first_micro_step",
        "receipt_type": "FakeProofReceipt",
    }
    assert trainer._fa2_first_micro_step_admitted is False


@pytest.mark.parametrize(
    ("policy", "stored_controls", "expected_controls"),
    [
        ("every_forward", (False, False), (True, True)),
        ("disabled", (True, True), (False, False)),
    ],
)
def test_explicit_fa2_proof_policies_override_recycled_step_flags(
    policy: str,
    stored_controls: tuple[bool, bool],
    expected_controls: tuple[bool, bool],
) -> None:
    step = replace(
        next(_micro_steps(1)),
        capture_fa2_branch=stored_controls[0],
        require_fa2_branch_proof=stored_controls[1],
        fa2_branch_proof_policy=policy,
    )
    observed_controls: list[tuple[bool, bool]] = []

    def qwen_forward(
        _model: object,
        micro_step: SupervisedMicroStep,
    ) -> FakeForwardResult:
        observed_controls.append(
            (micro_step.capture_fa2_branch, micro_step.require_fa2_branch_proof)
        )
        return FakeForwardResult(
            pack_index=0,
            logits=torch.zeros(1, 2, 3),
            receipt=FakeProofReceipt(fa2_branch_proof=object()),
        )

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=2, grad_accum_steps=1),
        pack_stream=(step, step),
        qwen_forward=qwen_forward,
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    trainer.run()

    assert observed_controls == [expected_controls, expected_controls]
    assert trainer._fa2_first_micro_step_admitted is False


def test_first_micro_step_policy_does_not_arm_unflagged_later_step() -> None:
    step = replace(
        next(_micro_steps(1)),
        capture_fa2_branch=False,
        require_fa2_branch_proof=False,
        fa2_branch_proof_policy="first_micro_step",
    )
    observed_controls: list[tuple[bool, bool]] = []

    def qwen_forward(
        _model: object,
        micro_step: SupervisedMicroStep,
    ) -> FakeForwardResult:
        observed_controls.append(
            (micro_step.capture_fa2_branch, micro_step.require_fa2_branch_proof)
        )
        return FakeForwardResult(
            pack_index=0,
            logits=torch.zeros(1, 2, 3),
            receipt=FakeProofReceipt(fa2_branch_proof=None),
        )

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=2, grad_accum_steps=1),
        pack_stream=(step, step),
        qwen_forward=qwen_forward,
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
    )

    trainer.run()

    assert observed_controls == [(False, False), (False, False)]
    assert trainer._fa2_first_micro_step_admitted is False


def test_default_and_provider_paths_receive_same_effective_fa2_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = replace(
        next(_micro_steps(1)),
        capture_fa2_branch=True,
        require_fa2_branch_proof=True,
        fa2_branch_proof_policy="first_micro_step",
    )
    observed_controls: list[tuple[bool, bool]] = []

    def fake_build_qwen_forward_inputs(*args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "default-forward-inputs"

    def fake_run_qwen_forward(
        _model: object,
        _forward_inputs: str,
        **kwargs: Any,
    ) -> FakeForwardResult:
        observed_controls.append(
            (
                kwargs["capture_fa2_branch"],
                kwargs["require_fa2_branch_proof"],
            )
        )
        return FakeForwardResult(
            pack_index=0,
            logits=torch.zeros(1, 2, 3),
            receipt=FakeProofReceipt(fa2_branch_proof=object()),
        )

    monkeypatch.setattr(
        trainer_module,
        "build_qwen_forward_inputs",
        fake_build_qwen_forward_inputs,
    )
    monkeypatch.setattr(trainer_module, "run_qwen_forward", fake_run_qwen_forward)

    for provider in (None, FakeForwardInputProvider()):
        trainer = SupervisedTrainer(
            model=object(),
            schedule=_schedule(resolved_max_steps=2, grad_accum_steps=1),
            pack_stream=(step, step),
            loss_context_factory=_loss_context([]),
            loss_runner=StreamingFakeLossRunner([]),
            runtime=FakeRuntime([]),
            forward_input_provider=provider,
        )
        trainer.run()

    assert observed_controls == [
        (True, True),
        (False, False),
        (True, True),
        (False, False),
    ]


def test_streaming_planned_step_ends_provider_cleanly_on_finite_gate_early_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeForwardInputProvider()

    def fake_run_qwen_forward(
        model: object, forward_inputs: str, **kwargs: Any
    ) -> FakeForwardResult:
        del model, kwargs
        pack_index = int(forward_inputs.split("-")[-1])
        return FakeForwardResult(
            pack_index=pack_index, logits=torch.zeros(1, 2, 3), receipt={}
        )

    monkeypatch.setattr(trainer_module, "run_qwen_forward", fake_run_qwen_forward)

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=2),
        pack_stream=_micro_steps(2),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([], unsafe_pre_steps={1}),
        forward_input_provider=provider,
    )

    result = trainer.run()

    # Finite-gate early break stops after ordinal 0: provider never sees
    # ordinal 1, and end_planned_step still runs to discard prepared work.
    assert provider.calls == ["begin", "take:0", "end"]
    assert (
        result.latest_observation.optimizer_update_status == "skipped_non_finite_scalar"
    )


def test_streaming_planned_step_ends_provider_on_consumer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeForwardInputProvider()

    def failing_run_qwen_forward(
        model: object, forward_inputs: str, **kwargs: Any
    ) -> Any:
        del model, forward_inputs, kwargs
        raise RuntimeError("consumer forward boom")

    monkeypatch.setattr(trainer_module, "run_qwen_forward", failing_run_qwen_forward)

    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        forward_input_provider=provider,
    )

    with pytest.raises(RuntimeError, match="consumer forward boom"):
        trainer.run()

    # end_planned_step must still run (bounded join / cancellation-aware
    # cleanup) even though the consumer raised mid-step.
    assert provider.calls == ["begin", "take:0", "end"]


def test_forward_input_provider_rejects_combination_with_custom_qwen_forward() -> None:
    provider = FakeForwardInputProvider()

    with pytest.raises(RuntimeContractError) as exc_info:
        SupervisedTrainer(
            model=object(),
            schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
            pack_stream=_micro_steps(1),
            qwen_forward=_forward([]),
            loss_context_factory=_loss_context([]),
            loss_runner=StreamingFakeLossRunner([]),
            runtime=FakeRuntime([]),
            forward_input_provider=provider,
        )

    assert (
        exc_info.value.code
        == "trainer.forward_input_provider_conflicts_with_custom_qwen_forward"
    )
    # No lifecycle calls happened at all: the trainer never got constructed.
    assert provider.calls == []


def test_supervised_trainer_rejects_non_streaming_loss_runner() -> None:
    # The non-streaming batch loss path has been deleted: construction must
    # fail closed for any loss runner lacking the streaming protocol,
    # regardless of whether a forward_input_provider is present.
    with pytest.raises(RuntimeContractError) as exc_info:
        SupervisedTrainer(
            model=object(),
            schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
            pack_stream=_micro_steps(1),
            loss_context_factory=_loss_context([]),
            loss_runner=FakeLossRunner([]),  # non-streaming: only .compute
            runtime=FakeRuntime([]),
        )

    assert exc_info.value.code == "trainer.loss_runner_requires_streaming_protocol"


def test_forward_input_provider_rejects_non_streaming_loss_runner() -> None:
    provider = FakeForwardInputProvider()

    with pytest.raises(RuntimeContractError) as exc_info:
        SupervisedTrainer(
            model=object(),
            schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
            pack_stream=_micro_steps(1),
            loss_context_factory=_loss_context([]),
            loss_runner=FakeLossRunner([]),  # non-streaming: only .compute
            runtime=FakeRuntime([]),
            forward_input_provider=provider,
        )

    assert exc_info.value.code == "trainer.loss_runner_requires_streaming_protocol"
    assert provider.calls == []


def test_streaming_planned_step_duration_excludes_provider_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same incrementing-fake-clock technique as the completed-step/scheduled
    # handler boundary test: the provider's end_planned_step() (teardown)
    # consumes its own extra ticks, so a boundary that wrongly extended
    # through teardown would read a precisely wrong, explicitly asserted
    # value here.
    tick_count = 0

    def fake_monotonic() -> float:
        nonlocal tick_count
        value = float(tick_count)
        tick_count += 1
        return value

    monkeypatch.setattr(trainer_module.time, "monotonic", fake_monotonic)

    def fake_run_qwen_forward(
        model: object, forward_inputs: str, **kwargs: Any
    ) -> FakeForwardResult:
        del model, kwargs
        pack_index = int(forward_inputs.split("-")[-1])
        return FakeForwardResult(
            pack_index=pack_index, logits=torch.zeros(1, 2, 3), receipt={}
        )

    monkeypatch.setattr(trainer_module, "run_qwen_forward", fake_run_qwen_forward)

    class SlowTeardownProvider(FakeForwardInputProvider):
        def end_planned_step(self) -> None:
            super().end_planned_step()
            trainer_module.time.monotonic()  # consumes a tick; must not count toward the boundary
            trainer_module.time.monotonic()  # a second tick, for emphasis

    provider = SlowTeardownProvider()
    observations: list[CompletedStepObservation] = []
    trainer = SupervisedTrainer(
        model=object(),
        schedule=_schedule(resolved_max_steps=1, grad_accum_steps=1),
        pack_stream=_micro_steps(1),
        loss_context_factory=_loss_context([]),
        loss_runner=StreamingFakeLossRunner([]),
        runtime=FakeRuntime([]),
        on_completed_step=observations.append,
        forward_input_provider=provider,
    )

    trainer.run()

    # Ticks 0 (boundary start) through the end of zero_gradients define a
    # fixed, explicitly asserted duration; the two teardown ticks consumed
    # by end_planned_step() must not be included.
    assert observations[0].step_duration_seconds == pytest.approx(1.0)
    assert observations[0].step_duration_seconds != pytest.approx(3.0)


def test_trainer_profile_sync_helper_is_exact_env_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_trainer_profile_sync_policy_is_frozen_against_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trainer_module.torch.cuda,
        "synchronize",
        lambda device: calls.append(str(device)),
    )
    monkeypatch.setenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", "1")

    trainer_module.set_profile_sync_timing_policy(False)
    try:
        trainer_module._sync_device_if_requested(torch.device("cuda:2"))
        monkeypatch.setenv("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS", "0")
        trainer_module.set_profile_sync_timing_policy(True)
        trainer_module._sync_device_if_requested(torch.device("cuda:3"))
    finally:
        trainer_module.set_profile_sync_timing_policy(None)

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

    artifact = result.latest_observation.loss_bundle_artifact
    assert artifact["partial"] is True
    assert artifact["metrics"] == {}
    assert artifact["diagnostics"]["normalizer_scope"] == "partial_planned_step"
    assert artifact["diagnostics"]["processed_micro_step_count"] == 1
    assert artifact["diagnostics"]["planned_micro_step_count"] == 2
    assert result.latest_observation.micro_step_count == 1
    assert (
        result.latest_observation.optimizer_update_status == "skipped_non_finite_scalar"
    )
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
class RealLossForwardResult:
    logits: torch.Tensor
    receipt: Any
    logits_position_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FakeProofReceipt:
    fa2_branch_proof: Any | None


class FakeForwardInputProvider:
    """Trainer-wiring test double: records lifecycle calls, never builds
    real Qwen tensors. Paired with a monkeypatched `run_qwen_forward` in
    tests that exercise the provider-driven trainer branch."""

    def __init__(self, wait_seconds_sequence: list[float] | None = None) -> None:
        self.calls: list[str] = []
        self.begin_args: list[tuple[int, tuple[Any, ...]]] = []
        self.take_args: list[tuple[int, Any]] = []
        self._wait_seconds_sequence = list(wait_seconds_sequence or [])
        self.last_take_wait_seconds = 0.0
        self.closed = False

    def begin_planned_step(
        self, planned_step_id: int, moved_micro_steps: tuple[Any, ...]
    ) -> None:
        self.calls.append("begin")
        self.begin_args.append((planned_step_id, tuple(moved_micro_steps)))

    def take(self, ordinal: int, micro_step: Any) -> str:
        self.calls.append(f"take:{ordinal}")
        self.take_args.append((ordinal, micro_step))
        self.last_take_wait_seconds = (
            self._wait_seconds_sequence.pop(0) if self._wait_seconds_sequence else 0.0
        )
        return f"forward-inputs-{ordinal}"

    def end_planned_step(self) -> None:
        self.calls.append("end")

    def close(self) -> None:
        self.closed = True


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

    def prepare_planned_step(
        self, micro_steps: tuple[SupervisedMicroStep, ...]
    ) -> dict[str, int]:
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
            "total_loss": sum(
                float(item["total_loss"]) for item in micro_loss_artifacts
            ),
            "metrics": {
                "loss/total": sum(
                    float(item["total_loss"]) for item in micro_loss_artifacts
                )
            },
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
        if (
            planned_step_id in self.unsafe_pre_steps
            or call_index in self.unsafe_pre_call_indices
        ):
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


class RealLossRuntime(FakeRuntime):
    def __init__(
        self,
        log: list[str],
        *,
        parameters: tuple[torch.nn.Parameter, ...],
        gate_weight: float,
    ) -> None:
        super().__init__(log)
        self.parameters = parameters
        self.gate_weight = float(gate_weight)
        self.bundle: Any | None = None
        self.backward_loss: torch.Tensor | None = None
        self.reference_objective: torch.Tensor | None = None
        self.reference_gradients: tuple[torch.Tensor, ...] = ()

    def pre_backward(
        self,
        bundle: Any,
        *,
        planned_step_id: int,
    ) -> GateDecision:
        self.bundle = bundle
        base_term = bundle.term_by_name("base_ce")
        gate_term = bundle.term_by_name("token_type_gate")
        self.reference_objective = base_term.weighted_loss
        if self.gate_weight != 0.0:
            self.reference_objective = (
                self.reference_objective + gate_term.weighted_loss
            )
        self.reference_gradients = torch.autograd.grad(
            self.reference_objective,
            self.parameters,
            retain_graph=True,
        )
        return super().pre_backward(bundle, planned_step_id=planned_step_id)

    def backward(
        self,
        loss: torch.Tensor,
        *,
        planned_step_id: int,
        sync_gradients: bool = True,
    ) -> None:
        self.backward_loss = loss
        super().backward(
            loss,
            planned_step_id=planned_step_id,
            sync_gradients=sync_gradients,
        )
        loss.backward()


class RuntimeWithPreparedModel(FakeRuntime):
    def __init__(self, log: list[str], model: object) -> None:
        super().__init__(log)
        self.model = model


def _token_sequence(index: int, *, target_positions: tuple[int, ...] = ()) -> TokenSequence:
    """One real ``TokenSequence``; an empty atom set selects no logit positions."""

    return TokenSequence(
        pack_index=index,
        input_ids=(10, 11, 12, 13),
        segments=(
            PackedSegment(
                pack_index=index,
                segment_index=0,
                example_index=0,
                example_id=f"ex-{index}",
                start=0,
                end=4,
            ),
        ),
        atoms=tuple(
            TokenAtom(
                pack_index=index,
                segment_index=0,
                example_index=0,
                example_id=f"ex-{index}",
                target_position=position,
                token_id=10 + position,
                token_type="schema",
                text="x",
                logical_target_position=position,
                source="unit",
            )
            for position in target_positions
        ),
        spans=(),
    )


def _micro_steps(count: int, log: list[str] | None = None):
    for index in range(count):
        if log is not None:
            log.append(f"stream:{index}")
        yield SupervisedMicroStep(
            pack=f"pack-{index}",
            encoded_examples=(f"example-{index}",),
            position_inputs=f"positions-{index}",
            token_sequence=_token_sequence(index),
            vocab_groups=f"vocab-{index}",
            metadata={},
        )


def _real_loss_projection() -> torch.nn.Linear:
    projection = torch.nn.Linear(2, 8)
    with torch.no_grad():
        projection.weight.copy_(torch.linspace(-0.4, 0.6, steps=16).reshape(8, 2))
        projection.bias.copy_(torch.linspace(0.3, -0.2, steps=8))
    return projection


def _real_loss_micro_step() -> SupervisedMicroStep:
    token_sequence = TokenSequence(
        pack_index=0,
        input_ids=(0, 7),
        segments=(
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=2,
            ),
        ),
        atoms=(
            TokenAtom(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                target_position=1,
                token_id=7,
                token_type="desc_text",
                text="x",
                logical_target_position=1,
                source="unit",
            ),
        ),
        spans=(),
    )
    vocab_groups = TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )
    return SupervisedMicroStep(
        pack="pack-0",
        encoded_examples=("example-0",),
        position_inputs="positions-0",
        token_sequence=token_sequence,
        vocab_groups=vocab_groups,
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


def _scheduled_tuple(
    event: StepScheduleEvent,
    observation: CompletedStepObservation,
) -> tuple[str, int, tuple[str, ...], str]:
    return (
        event.event,
        event.planned_step_id,
        event.trigger_reasons,
        observation.optimizer_update_status,
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


# ---------------------------------------------------------------------------
# Canonical SupervisedMicroStep owner (design decision 4)
# ---------------------------------------------------------------------------


def test_micro_step_has_one_canonical_owner_and_compatibility_exports() -> None:
    import src.training as public_training
    from src.training import micro_steps

    assert micro_steps.SupervisedMicroStep.__module__ == "src.training.micro_steps"
    assert public_training.SupervisedMicroStep is micro_steps.SupervisedMicroStep
    assert trainer_module.SupervisedMicroStep is micro_steps.SupervisedMicroStep
    assert SupervisedMicroStep is micro_steps.SupervisedMicroStep


def test_micro_step_record_schema_is_field_order_annotation_and_default_exact() -> None:
    from src.training import micro_steps

    record = micro_steps.SupervisedMicroStep
    observed = [
        (item.name, item.type, MISSING if item.default is MISSING else item.default)
        for item in fields(record)
    ]

    assert record.__dataclass_params__.frozen is True
    assert observed == [
        ("pack", "Any", MISSING),
        ("encoded_examples", "Sequence[Any]", MISSING),
        ("position_inputs", "Any", MISSING),
        ("token_sequence", "TokenSequence | Any", MISSING),
        ("vocab_groups", "TokenVocabularyGroups | Any", MISSING),
        ("metadata", "Mapping[str, Any] | None", None),
        ("forward_device", "torch.device | str | None", None),
        ("expected_vocab_size", "int | None", None),
        ("extra_model_kwargs", "Mapping[str, Any] | None", None),
        ("fa2_branch_evidence", "Mapping[str, Any] | None", None),
        ("fa2_model_dtype", "str | None", None),
        ("capture_fa2_branch", "bool", False),
        ("require_fa2_branch_proof", "bool", False),
        ("fa2_branch_proof_policy", "str | None", None),
    ]


def test_micro_step_schema_identity_owner_reports_the_exact_record_schema() -> None:
    from src.training import micro_steps

    identity = micro_steps.supervised_micro_step_schema_identity()

    assert identity["class"] == "SupervisedMicroStep"
    assert identity["frozen"] is True
    assert [entry["name"] for entry in identity["fields"]] == [
        item.name for item in fields(micro_steps.SupervisedMicroStep)
    ]
    assert [entry["annotation"] for entry in identity["fields"]] == [
        str(item.type) for item in fields(micro_steps.SupervisedMicroStep)
    ]
    assert [
        (entry["has_default"], entry["default"]) for entry in identity["fields"]
    ] == [
        (item.default is not MISSING, None if item.default is MISSING else item.default)
        for item in fields(micro_steps.SupervisedMicroStep)
    ]
