"""Supervised training loop orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from src.common.errors import RuntimeContractError
from src.losses.context import LossContext
from src.losses.runner import LossBundle
from src.losses.vocab import TokenVocabularyGroups
from src.qwen.forward import build_qwen_forward_inputs, run_qwen_forward
from src.runtime.finite_gates import GateDecision
from src.supervision import TokenSequence
from src.training.schedule import ResolvedStepSchedule, StepScheduleEvent


@dataclass(frozen=True)
class SupervisedMicroStep:
    pack: Any
    encoded_examples: Sequence[Any]
    position_inputs: Any
    token_sequence: TokenSequence | Any
    vocab_groups: TokenVocabularyGroups | Any
    metadata: Mapping[str, Any] | None = None
    forward_device: torch.device | str | None = None
    expected_vocab_size: int | None = None
    extra_model_kwargs: Mapping[str, Any] | None = None
    fa2_branch_evidence: Mapping[str, Any] | None = None
    fa2_model_dtype: str | None = None


@dataclass(frozen=True)
class SupervisedTrainerEvent:
    event_type: str
    planned_step_id: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class PlannedStepResult:
    planned_step_id: int
    micro_step_count: int
    loss_bundle_artifact: Mapping[str, Any]
    pre_backward_decision: GateDecision
    post_backward_decision: GateDecision | None
    qwen_forward_receipts: tuple[Mapping[str, Any], ...]
    optimizer_update_status: str
    finite_status: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "planned_step_id": self.planned_step_id,
            "micro_step_count": self.micro_step_count,
            "loss_bundle": dict(self.loss_bundle_artifact),
            "pre_backward_decision": self.pre_backward_decision.to_artifact_dict(),
            "post_backward_decision": (
                None
                if self.post_backward_decision is None
                else self.post_backward_decision.to_artifact_dict()
            ),
            "qwen_forward_receipts": [dict(item) for item in self.qwen_forward_receipts],
            "optimizer_update_status": self.optimizer_update_status,
            "finite_status": self.finite_status,
        }


@dataclass(frozen=True)
class ScheduledTrainerEvent:
    scheduled_event: StepScheduleEvent
    step_result: PlannedStepResult


@dataclass(frozen=True)
class SupervisedTrainingResult:
    completed_steps: int
    consumed_micro_steps: int
    step_results: tuple[PlannedStepResult, ...]
    scheduled_event_counts: dict[str, int]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "consumed_micro_steps": self.consumed_micro_steps,
            "scheduled_event_counts": dict(self.scheduled_event_counts),
            "step_results": [result.to_artifact_dict() for result in self.step_results],
        }


class RuntimeBoundary(Protocol):
    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep: ...

    def pre_backward(self, bundle: Any, *, planned_step_id: int) -> GateDecision: ...

    def backward(self, loss: torch.Tensor, *, planned_step_id: int) -> None: ...

    def post_backward(self, *, planned_step_id: int) -> GateDecision: ...

    def clip_gradients(self, *, planned_step_id: int) -> None: ...

    def optimizer_step(self, *, planned_step_id: int) -> None: ...

    def scheduler_step(self, *, planned_step_id: int) -> None: ...

    def zero_gradients(self, *, planned_step_id: int) -> None: ...


class LossRunnerBoundary(Protocol):
    def compute(self, contexts: Sequence[Any]) -> Any: ...


QwenForwardFn = Callable[[Any, SupervisedMicroStep], Any]
LossContextFactory = Callable[[SupervisedMicroStep, Any], Any]
TrainerEventSink = Callable[[SupervisedTrainerEvent], None]
ScheduledEventHandler = Callable[[ScheduledTrainerEvent], None]


class SupervisedTrainer:
    def __init__(
        self,
        *,
        model: Any,
        schedule: ResolvedStepSchedule,
        pack_stream: Iterable[SupervisedMicroStep],
        qwen_forward: QwenForwardFn | None = None,
        loss_context_factory: LossContextFactory | None = None,
        loss_runner: LossRunnerBoundary,
        runtime: RuntimeBoundary,
        event_sink: TrainerEventSink | None = None,
        scheduled_event_handlers: Mapping[str, ScheduledEventHandler] | None = None,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.pack_stream = iter(pack_stream)
        self.qwen_forward = qwen_forward or _default_qwen_forward
        self.loss_context_factory = loss_context_factory or _default_loss_context
        self.loss_runner = loss_runner
        self.runtime = runtime
        self.event_sink = event_sink
        self.scheduled_event_handlers = dict(scheduled_event_handlers or {})

    def run(self) -> SupervisedTrainingResult:
        step_results: list[PlannedStepResult] = []
        scheduled_event_counts = {
            name: 0 for name in sorted(self.schedule.events)
        }
        consumed_micro_steps = 0
        micro_steps_per_planned_step = (
            self.schedule.runtime_batch.resolved_grad_accum_steps
        )

        for planned_step_id in range(1, self.schedule.resolved_max_steps + 1):
            self._emit(
                "planned_step.started",
                planned_step_id,
                {"micro_steps_per_planned_step": micro_steps_per_planned_step},
            )
            contexts: list[Any] = []
            qwen_receipts: list[Mapping[str, Any]] = []
            for local_micro_step_index in range(micro_steps_per_planned_step):
                micro_step = self._next_micro_step(
                    planned_step_id=planned_step_id,
                    local_micro_step_index=local_micro_step_index,
                )
                consumed_micro_steps += 1
                micro_step = self.runtime.move_micro_step(
                    micro_step,
                    planned_step_id=planned_step_id,
                    local_micro_step_index=local_micro_step_index,
                )
                forward_result = self.qwen_forward(
                    _runtime_model(self.runtime, self.model),
                    micro_step,
                )
                contexts.append(
                    self.loss_context_factory(micro_step, forward_result)
                )
                qwen_receipts.append(_receipt_artifact(forward_result))
                self._emit(
                    "micro_step.forward",
                    planned_step_id,
                    {
                        "local_micro_step_index": local_micro_step_index,
                        "receipt": qwen_receipts[-1],
                    },
                )

            loss_bundle = self.loss_runner.compute(tuple(contexts))
            self._emit(
                "planned_step.loss",
                planned_step_id,
                {"loss_bundle": _artifact(loss_bundle)},
            )
            pre_decision = self.runtime.pre_backward(
                loss_bundle,
                planned_step_id=planned_step_id,
            )
            self._emit(
                "planned_step.pre_backward_gate",
                planned_step_id,
                pre_decision.to_artifact_dict(),
            )
            post_decision: GateDecision | None = None
            optimizer_update_status = pre_decision.optimizer_update_status
            finite_status = pre_decision.finite_status

            if pre_decision.should_call_backward:
                self.runtime.backward(
                    _total_loss(loss_bundle),
                    planned_step_id=planned_step_id,
                )
                post_decision = self.runtime.post_backward(
                    planned_step_id=planned_step_id,
                )
                self._emit(
                    "planned_step.post_backward_gate",
                    planned_step_id,
                    post_decision.to_artifact_dict(),
                )
                optimizer_update_status = post_decision.optimizer_update_status
                finite_status = post_decision.finite_status
                if post_decision.should_call_optimizer_step:
                    self.runtime.clip_gradients(planned_step_id=planned_step_id)
                    self.runtime.optimizer_step(planned_step_id=planned_step_id)
                    optimizer_update_status = "applied"

            self.runtime.scheduler_step(planned_step_id=planned_step_id)
            self.runtime.zero_gradients(planned_step_id=planned_step_id)

            step_result = PlannedStepResult(
                planned_step_id=planned_step_id,
                micro_step_count=len(contexts),
                loss_bundle_artifact=_artifact(loss_bundle),
                pre_backward_decision=pre_decision,
                post_backward_decision=post_decision,
                qwen_forward_receipts=tuple(qwen_receipts),
                optimizer_update_status=optimizer_update_status,
                finite_status=finite_status,
            )
            step_results.append(step_result)
            self._emit(
                "planned_step.completed",
                planned_step_id,
                step_result.to_artifact_dict(),
            )
            self._trigger_scheduled_events(
                planned_step_id=planned_step_id,
                step_result=step_result,
                scheduled_event_counts=scheduled_event_counts,
            )

        return SupervisedTrainingResult(
            completed_steps=len(step_results),
            consumed_micro_steps=consumed_micro_steps,
            step_results=tuple(step_results),
            scheduled_event_counts=scheduled_event_counts,
        )

    def _next_micro_step(
        self,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        try:
            return next(self.pack_stream)
        except StopIteration as exc:
            raise RuntimeContractError(
                "pack stream ended before the planned optimizer-step window was filled",
                code="trainer.pack_stream_exhausted",
                context={
                    "planned_step_id": planned_step_id,
                    "local_micro_step_index": local_micro_step_index,
                    "resolved_max_steps": self.schedule.resolved_max_steps,
                    "resolved_grad_accum_steps": (
                        self.schedule.runtime_batch.resolved_grad_accum_steps
                    ),
                },
                cause=exc,
            ) from exc

    def _trigger_scheduled_events(
        self,
        *,
        planned_step_id: int,
        step_result: PlannedStepResult,
        scheduled_event_counts: dict[str, int],
    ) -> None:
        for event_name in _scheduled_event_order(self.schedule.events):
            for scheduled_event in self.schedule.events[event_name]:
                if scheduled_event.planned_step_id != planned_step_id:
                    continue
                scheduled_event_counts[event_name] += 1
                payload = scheduled_event.to_artifact_dict()
                payload["optimizer_update_status"] = step_result.optimizer_update_status
                self._emit(
                    f"schedule.{scheduled_event.event}",
                    planned_step_id,
                    payload,
                )
                handler = self.scheduled_event_handlers.get(scheduled_event.event)
                if handler is None and scheduled_event.required:
                    raise RuntimeContractError(
                        "required scheduled trainer event has no handler",
                        code="trainer.required_event_handler_missing",
                        context={
                            "event": scheduled_event.event,
                            "planned_step_id": planned_step_id,
                            "trigger_reasons": list(scheduled_event.trigger_reasons),
                        },
                    )
                if handler is not None:
                    handler(
                        ScheduledTrainerEvent(
                            scheduled_event=scheduled_event,
                            step_result=step_result,
                        )
                    )

    def _emit(
        self,
        event_type: str,
        planned_step_id: int,
        payload: Mapping[str, Any],
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink(
            SupervisedTrainerEvent(
                event_type=event_type,
                planned_step_id=planned_step_id,
                payload=dict(payload),
            )
        )


def _default_qwen_forward(model: Any, micro_step: SupervisedMicroStep) -> Any:
    forward_inputs = build_qwen_forward_inputs(
        micro_step.pack,
        micro_step.encoded_examples,
        micro_step.position_inputs,
        logits_to_keep_positions=_logits_positions_to_keep(micro_step),
        device=micro_step.forward_device,
    )
    return run_qwen_forward(
        model,
        forward_inputs,
        expected_vocab_size=micro_step.expected_vocab_size,
        extra_model_kwargs=micro_step.extra_model_kwargs,
        fa2_branch_evidence=micro_step.fa2_branch_evidence,
        fa2_model_dtype=micro_step.fa2_model_dtype,
    )


def _runtime_model(runtime: RuntimeBoundary, fallback_model: Any) -> Any:
    return getattr(runtime, "model", fallback_model)


def _default_loss_context(
    micro_step: SupervisedMicroStep,
    forward_result: Any,
) -> LossContext:
    return LossContext(
        logits=forward_result.logits,
        token_sequence=micro_step.token_sequence,
        vocab_groups=micro_step.vocab_groups,
        logits_position_ids=forward_result.logits_position_ids,
    )


def _logits_positions_to_keep(micro_step: SupervisedMicroStep) -> tuple[int, ...] | None:
    atoms = getattr(micro_step.token_sequence, "atoms", None)
    if atoms is None:
        return None
    positions = tuple(
        sorted({int(atom.causal_logits_position) for atom in atoms})
    )
    return positions or None


def _total_loss(loss_bundle: LossBundle | Any) -> torch.Tensor:
    total_loss = getattr(loss_bundle, "total_loss", None)
    if not isinstance(total_loss, torch.Tensor):
        raise RuntimeContractError(
            "loss runner must return a bundle with tensor total_loss",
            code="trainer.loss_bundle_total_loss",
            context={"bundle_type": type(loss_bundle).__name__},
        )
    return total_loss


def _receipt_artifact(forward_result: Any) -> Mapping[str, Any]:
    receipt = getattr(forward_result, "receipt", None)
    if receipt is None:
        return {}
    return _artifact(receipt)


def _artifact(value: Any) -> Mapping[str, Any]:
    to_artifact_dict = getattr(value, "to_artifact_dict", None)
    if callable(to_artifact_dict):
        artifact = to_artifact_dict()
        if isinstance(artifact, Mapping):
            return artifact
    if isinstance(value, Mapping):
        return value
    return {"repr": repr(value)}


def _scheduled_event_order(
    events: Mapping[str, Sequence[StepScheduleEvent]],
) -> tuple[str, ...]:
    priority = {
        "eval.forward": 0,
        "checkpoint": 1,
        "training.logging": 2,
        "final": 3,
    }
    return tuple(sorted(events, key=lambda name: (priority.get(name, 100), name)))


__all__ = [
    "LossContextFactory",
    "LossRunnerBoundary",
    "PlannedStepResult",
    "QwenForwardFn",
    "RuntimeBoundary",
    "ScheduledEventHandler",
    "ScheduledTrainerEvent",
    "SupervisedMicroStep",
    "SupervisedTrainer",
    "SupervisedTrainerEvent",
    "SupervisedTrainingResult",
    "TrainerEventSink",
]
