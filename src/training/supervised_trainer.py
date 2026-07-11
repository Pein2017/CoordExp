"""Supervised training loop orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import inspect
import os
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
    capture_fa2_branch: bool = False
    require_fa2_branch_proof: bool = False
    fa2_branch_proof_policy: str | None = None


@dataclass(frozen=True)
class CompletedStepObservation:
    planned_step_id: int
    micro_step_count: int
    loss_bundle_artifact: Mapping[str, Any]
    optimizer_update_status: str
    finite_status: str
    scheduler_artifact: Mapping[str, Any] = field(default_factory=dict)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "planned_step_id": self.planned_step_id,
            "micro_step_count": self.micro_step_count,
            "loss_bundle": dict(self.loss_bundle_artifact),
            "optimizer_update_status": self.optimizer_update_status,
            "finite_status": self.finite_status,
            "scheduler": dict(self.scheduler_artifact),
        }


@dataclass(frozen=True)
class SupervisedTrainingResult:
    completed_steps: int
    consumed_micro_steps: int
    scheduled_event_counts: dict[str, int]
    latest_observation: CompletedStepObservation | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "consumed_micro_steps": self.consumed_micro_steps,
            "scheduled_event_counts": dict(self.scheduled_event_counts),
            "latest_observation": (
                None
                if self.latest_observation is None
                else self.latest_observation.to_artifact_dict()
            ),
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

    def accumulation_context(self, *, sync_gradients: bool) -> Any: ...

    def backward(
        self,
        loss: torch.Tensor,
        *,
        planned_step_id: int,
        sync_gradients: bool = True,
    ) -> None: ...

    def post_backward(self, *, planned_step_id: int) -> GateDecision: ...

    def clip_gradients(self, *, planned_step_id: int) -> None: ...

    def optimizer_step(self, *, planned_step_id: int) -> None: ...

    def scheduler_step(self, *, planned_step_id: int) -> Mapping[str, Any] | None: ...

    def zero_gradients(self, *, planned_step_id: int) -> None: ...


class LossRunnerBoundary(Protocol):
    def compute(self, contexts: Sequence[Any]) -> Any: ...


QwenForwardFn = Callable[[Any, SupervisedMicroStep], Any]
LossContextFactory = Callable[[SupervisedMicroStep, Any], Any]
CompletedStepHandler = Callable[[CompletedStepObservation], None]
ScheduledStepHandler = Callable[[StepScheduleEvent, CompletedStepObservation], None]


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
        on_completed_step: CompletedStepHandler | None = None,
        on_eval: ScheduledStepHandler | None = None,
        on_checkpoint: ScheduledStepHandler | None = None,
        on_final: ScheduledStepHandler | None = None,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.pack_stream = iter(pack_stream)
        self.qwen_forward = qwen_forward or _default_qwen_forward
        self.loss_context_factory = loss_context_factory or _default_loss_context
        self.loss_runner = loss_runner
        self.runtime = runtime
        self.on_completed_step = on_completed_step
        self.on_eval = on_eval
        self.on_checkpoint = on_checkpoint
        self.on_final = on_final

    def run(self) -> SupervisedTrainingResult:
        latest_observation: CompletedStepObservation | None = None
        scheduled_event_counts = {
            name: 0 for name in sorted(self.schedule.events)
        }
        consumed_micro_steps = 0
        micro_steps_per_planned_step = (
            self.schedule.runtime_batch.resolved_grad_accum_steps
        )

        for planned_step_id in range(1, self.schedule.resolved_max_steps + 1):
            if _supports_streaming_loss(self.loss_runner):
                observation, consumed_count = self._run_streaming_planned_step(
                    planned_step_id=planned_step_id,
                    micro_steps_per_planned_step=micro_steps_per_planned_step,
                )
                consumed_micro_steps += consumed_count
                latest_observation = observation
                self._notify_completed_step(observation)
                self._trigger_scheduled_events(
                    planned_step_id=planned_step_id,
                    observation=observation,
                    scheduled_event_counts=scheduled_event_counts,
                )
                continue

            contexts: list[Any] = []
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
                del forward_result, micro_step

            loss_bundle = self.loss_runner.compute(tuple(contexts))
            pre_decision = self.runtime.pre_backward(
                loss_bundle,
                planned_step_id=planned_step_id,
            )
            post_decision: GateDecision | None = None
            optimizer_update_status = pre_decision.optimizer_update_status
            finite_status = pre_decision.finite_status

            if pre_decision.should_call_backward:
                self.runtime.backward(
                    _total_loss(loss_bundle),
                    planned_step_id=planned_step_id,
                    sync_gradients=True,
                )
                post_decision = self.runtime.post_backward(
                    planned_step_id=planned_step_id,
                )
                optimizer_update_status = post_decision.optimizer_update_status
                finite_status = post_decision.finite_status
                if post_decision.should_call_optimizer_step:
                    self.runtime.clip_gradients(planned_step_id=planned_step_id)
                    self.runtime.optimizer_step(planned_step_id=planned_step_id)
                    optimizer_update_status = "applied"

            scheduler_artifact = _optional_artifact(
                self.runtime.scheduler_step(planned_step_id=planned_step_id)
            )
            self.runtime.zero_gradients(planned_step_id=planned_step_id)

            observation = CompletedStepObservation(
                planned_step_id=planned_step_id,
                micro_step_count=len(contexts),
                loss_bundle_artifact=_artifact(loss_bundle),
                optimizer_update_status=optimizer_update_status,
                finite_status=finite_status,
                scheduler_artifact=scheduler_artifact,
            )
            del contexts, loss_bundle, pre_decision, post_decision, scheduler_artifact
            latest_observation = observation
            self._notify_completed_step(observation)
            self._trigger_scheduled_events(
                planned_step_id=planned_step_id,
                observation=observation,
                scheduled_event_counts=scheduled_event_counts,
            )

        return SupervisedTrainingResult(
            completed_steps=self.schedule.resolved_max_steps,
            consumed_micro_steps=consumed_micro_steps,
            scheduled_event_counts=scheduled_event_counts,
            latest_observation=latest_observation,
        )

    def _run_streaming_planned_step(
        self,
        *,
        planned_step_id: int,
        micro_steps_per_planned_step: int,
    ) -> tuple[CompletedStepObservation, int]:
        moved_micro_steps: list[SupervisedMicroStep] = []
        for local_micro_step_index in range(micro_steps_per_planned_step):
            micro_step = self._next_micro_step(
                planned_step_id=planned_step_id,
                local_micro_step_index=local_micro_step_index,
            )
            moved_micro_steps.append(
                self.runtime.move_micro_step(
                    micro_step,
                    planned_step_id=planned_step_id,
                    local_micro_step_index=local_micro_step_index,
                )
            )

        plan = _prepare_streaming_loss_plan(
            self.loss_runner,
            tuple(moved_micro_steps),
            runtime=self.runtime,
            schedule=self.schedule,
            planned_step_id=planned_step_id,
        )
        micro_loss_artifacts: list[Mapping[str, Any]] = []
        pre_decision: GateDecision | None = None
        post_decision: GateDecision | None = None
        optimizer_update_status = "not_started"
        finite_status = "unavailable"

        for local_micro_step_index, micro_step in enumerate(moved_micro_steps):
            sync_gradients = local_micro_step_index == len(moved_micro_steps) - 1
            with self.runtime.accumulation_context(sync_gradients=sync_gradients):
                _sync_device_if_requested(micro_step.forward_device)
                forward_result = self.qwen_forward(
                    _runtime_model(self.runtime, self.model),
                    micro_step,
                )
                _sync_forward_result_if_requested(forward_result)
                _sync_forward_result_if_requested(forward_result)
                context = self.loss_context_factory(micro_step, forward_result)
                _sync_forward_result_if_requested(forward_result)
                _sync_forward_result_if_requested(forward_result)
                loss_bundle = self.loss_runner.compute_micro_step(
                    context,
                    plan,
                    local_micro_step_index=local_micro_step_index,
                )
                _sync_loss_bundle_if_requested(loss_bundle)
                micro_loss_artifact = _artifact(loss_bundle)
                micro_loss_artifacts.append(micro_loss_artifact)
                pre_decision = self.runtime.pre_backward(
                    loss_bundle,
                    planned_step_id=planned_step_id,
                )
                optimizer_update_status = pre_decision.optimizer_update_status
                finite_status = pre_decision.finite_status
                if not pre_decision.should_call_backward:
                    break
                _sync_loss_bundle_if_requested(loss_bundle)
                self.runtime.backward(
                    _total_loss(loss_bundle),
                    planned_step_id=planned_step_id,
                    sync_gradients=sync_gradients,
                )
                _sync_loss_bundle_if_requested(loss_bundle)
            del loss_bundle, context, forward_result

        if pre_decision is None:
            raise RuntimeContractError(
                "streaming planned step did not produce any micro-step decisions",
                code="trainer.streaming_empty_step",
                context={"planned_step_id": planned_step_id},
            )

        loss_bundle_artifact = self.loss_runner.finalize_planned_step(
            tuple(micro_loss_artifacts),
            plan,
        )
        if len(micro_loss_artifacts) != len(moved_micro_steps):
            loss_bundle_artifact = _partial_loss_artifact(
                loss_bundle_artifact,
                processed_micro_step_count=len(micro_loss_artifacts),
                planned_micro_step_count=len(moved_micro_steps),
            )
        if pre_decision.should_call_backward:
            post_decision = self.runtime.post_backward(
                planned_step_id=planned_step_id,
            )
            optimizer_update_status = post_decision.optimizer_update_status
            finite_status = post_decision.finite_status
            if post_decision.should_call_optimizer_step:
                self.runtime.clip_gradients(planned_step_id=planned_step_id)
                self.runtime.optimizer_step(planned_step_id=planned_step_id)
                optimizer_update_status = "applied"

        scheduler_artifact = _optional_artifact(
            self.runtime.scheduler_step(planned_step_id=planned_step_id)
        )
        self.runtime.zero_gradients(planned_step_id=planned_step_id)

        consumed_count = len(moved_micro_steps)
        observation = CompletedStepObservation(
            planned_step_id=planned_step_id,
            micro_step_count=len(micro_loss_artifacts),
            loss_bundle_artifact=dict(loss_bundle_artifact),
            optimizer_update_status=optimizer_update_status,
            finite_status=finite_status,
            scheduler_artifact=scheduler_artifact,
        )
        del moved_micro_steps, micro_loss_artifacts, plan
        del pre_decision, post_decision, scheduler_artifact, loss_bundle_artifact
        return observation, consumed_count

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
        observation: CompletedStepObservation,
        scheduled_event_counts: dict[str, int],
    ) -> None:
        self._run_scheduled_group(
            scheduled_events=self.schedule.events["eval.forward"],
            handler=self.on_eval,
            planned_step_id=planned_step_id,
            observation=observation,
            scheduled_event_counts=scheduled_event_counts,
        )
        self._run_scheduled_group(
            scheduled_events=self.schedule.events["checkpoint"],
            handler=self.on_checkpoint,
            planned_step_id=planned_step_id,
            observation=observation,
            scheduled_event_counts=scheduled_event_counts,
        )
        self._run_scheduled_group(
            scheduled_events=self.schedule.events["final"],
            handler=self.on_final,
            planned_step_id=planned_step_id,
            observation=observation,
            scheduled_event_counts=scheduled_event_counts,
        )

    @staticmethod
    def _run_scheduled_group(
        *,
        scheduled_events: Sequence[StepScheduleEvent],
        handler: ScheduledStepHandler | None,
        planned_step_id: int,
        observation: CompletedStepObservation,
        scheduled_event_counts: dict[str, int],
    ) -> None:
        for scheduled_event in scheduled_events:
            if scheduled_event.planned_step_id != planned_step_id:
                continue
            scheduled_event_counts[scheduled_event.event] += 1
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
                handler(scheduled_event, observation)

    def _notify_completed_step(self, observation: CompletedStepObservation) -> None:
        if self.on_completed_step is not None:
            self.on_completed_step(observation)


def _default_qwen_forward(model: Any, micro_step: SupervisedMicroStep) -> Any:
    forward_inputs = build_qwen_forward_inputs(
        micro_step.pack,
        micro_step.encoded_examples,
        micro_step.position_inputs,
        logits_to_keep_positions=_logits_positions_to_keep(micro_step),
        device=micro_step.forward_device,
        fa2_branch_proof_policy=micro_step.fa2_branch_proof_policy,
    )
    return run_qwen_forward(
        model,
        forward_inputs,
        expected_vocab_size=micro_step.expected_vocab_size,
        extra_model_kwargs=micro_step.extra_model_kwargs,
        fa2_branch_evidence=micro_step.fa2_branch_evidence,
        fa2_model_dtype=micro_step.fa2_model_dtype,
        capture_fa2_branch=micro_step.capture_fa2_branch,
        require_fa2_branch_proof=micro_step.require_fa2_branch_proof,
    )


def _runtime_model(runtime: RuntimeBoundary, fallback_model: Any) -> Any:
    return getattr(runtime, "model", fallback_model)


def _supports_streaming_loss(loss_runner: Any) -> bool:
    return all(
        callable(getattr(loss_runner, name, None))
        for name in (
            "prepare_planned_step",
            "compute_micro_step",
            "finalize_planned_step",
        )
    )


def _prepare_streaming_loss_plan(
    loss_runner: Any,
    moved_micro_steps: tuple[SupervisedMicroStep, ...],
    *,
    runtime: RuntimeBoundary,
    schedule: ResolvedStepSchedule,
    planned_step_id: int,
) -> Any:
    prepare_planned_step = loss_runner.prepare_planned_step
    world_size = int(schedule.runtime_batch.world_size)
    if not _call_accepts_keyword(prepare_planned_step, "denominator_gatherer"):
        if world_size > 1:
            raise RuntimeContractError(
                "multi-rank streaming loss runner must support global denominator gathering",
                code="trainer.loss_denominator_gather_unsupported",
                context={
                    "planned_step_id": planned_step_id,
                    "world_size": world_size,
                    "loss_runner": type(loss_runner).__name__,
                },
            )
        return prepare_planned_step(moved_micro_steps)

    denominator_gatherer = _runtime_loss_denominator_gatherer(
        runtime,
        planned_step_id=planned_step_id,
        world_size=world_size,
    )
    return prepare_planned_step(
        moved_micro_steps,
        denominator_gatherer=denominator_gatherer,
        world_size=world_size,
        rank=int(getattr(runtime, "rank", 0)),
    )


def _runtime_loss_denominator_gatherer(
    runtime: RuntimeBoundary,
    *,
    planned_step_id: int,
    world_size: int,
) -> Callable[[Mapping[str, Mapping[str, Any]]], Sequence[Mapping[str, Mapping[str, Any]]]] | None:
    gather_loss_denominators = getattr(runtime, "gather_loss_denominators", None)
    if callable(gather_loss_denominators):
        return lambda payload: gather_loss_denominators(
            payload,
            planned_step_id=planned_step_id,
        )
    if world_size > 1:
        raise RuntimeContractError(
            "multi-rank streaming loss requires runtime denominator gathering",
            code="trainer.loss_denominator_gather_unavailable",
            context={"planned_step_id": planned_step_id, "world_size": world_size},
        )
    return None


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


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


def _profile_sync_enabled() -> bool:
    return os.environ.get("COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS") == "1"


def _sync_forward_result_if_requested(forward_result: Any) -> None:
    logits = getattr(forward_result, "logits", None)
    if isinstance(logits, torch.Tensor):
        _sync_device_if_requested(logits.device)


def _sync_loss_bundle_if_requested(loss_bundle: Any) -> None:
    total_loss = getattr(loss_bundle, "total_loss", None)
    if isinstance(total_loss, torch.Tensor):
        _sync_device_if_requested(total_loss.device)


def _sync_device_if_requested(device: torch.device | str | None) -> None:
    if device is None or not _profile_sync_enabled():
        return
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def _artifact(value: Any) -> Mapping[str, Any]:
    to_artifact_dict = getattr(value, "to_artifact_dict", None)
    if callable(to_artifact_dict):
        artifact = to_artifact_dict()
        if isinstance(artifact, Mapping):
            return artifact
    if isinstance(value, Mapping):
        return value
    return {"repr": repr(value)}


def _optional_artifact(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _artifact(value)


def _partial_loss_artifact(
    artifact: Mapping[str, Any],
    *,
    processed_micro_step_count: int,
    planned_micro_step_count: int,
) -> dict[str, Any]:
    result = dict(artifact)
    diagnostics_value = result.get("diagnostics")
    diagnostics = (
        dict(diagnostics_value)
        if isinstance(diagnostics_value, Mapping)
        else {}
    )
    diagnostics.update(
        {
            "normalizer_scope": "partial_planned_step",
            "processed_micro_step_count": int(processed_micro_step_count),
            "planned_micro_step_count": int(planned_micro_step_count),
            "metrics_status": "unavailable_partial_planned_step",
        }
    )
    result["partial"] = True
    result["metrics"] = {}
    result["diagnostics"] = diagnostics
    return result


__all__ = [
    "CompletedStepHandler",
    "CompletedStepObservation",
    "LossContextFactory",
    "LossRunnerBoundary",
    "QwenForwardFn",
    "RuntimeBoundary",
    "ScheduledStepHandler",
    "SupervisedMicroStep",
    "SupervisedTrainer",
    "SupervisedTrainingResult",
]
