"""Artifact-independent packed forward evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import torch

from src.common.errors import RuntimeContractError
from src.training.supervised_trainer import (
    LossContextFactory,
    LossRunnerBoundary,
    QwenForwardFn,
    SupervisedMicroStep,
    _default_loss_context,
    _default_qwen_forward,
)


EVAL_FORWARD_SPLIT = "eval"


class EvalRuntimeBoundary(Protocol):
    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep: ...

    def gather_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        planned_step_id: int,
        split: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ForwardEvalObservation:
    """One completed eval invocation, ready for a canonical wide logging row."""

    planned_step_id: int
    split: str
    trigger_reasons: tuple[str, ...]
    example_count: int
    pack_count: int
    scalars: Mapping[str, float | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))

    def to_logging_row(self) -> dict[str, Any]:
        """Return writer input without normalizing non-finite scalar values."""

        return {
            "step": self.planned_step_id,
            "split": self.split,
            "trigger_reasons": list(self.trigger_reasons),
            "example_count": self.example_count,
            "pack_count": self.pack_count,
            **self.scalars,
        }


class ForwardEvalRunner:
    def __init__(
        self,
        *,
        model: Any,
        micro_step_stream: Iterable[SupervisedMicroStep],
        loss_runner: LossRunnerBoundary,
        eval_source: Mapping[str, Any] | None,
        qwen_forward: QwenForwardFn | None = None,
        loss_context_factory: LossContextFactory | None = None,
        runtime: EvalRuntimeBoundary | None = None,
    ) -> None:
        self.model = model
        self.micro_step_stream = micro_step_stream
        self.loss_runner = loss_runner
        self.eval_source = None if eval_source is None else dict(eval_source)
        self.qwen_forward = qwen_forward or _default_qwen_forward
        self.loss_context_factory = loss_context_factory or _default_loss_context
        self.runtime = runtime

    def run(
        self,
        *,
        planned_step_id: int,
        trigger_reasons: Sequence[str],
    ) -> ForwardEvalObservation:
        if planned_step_id <= 0:
            raise RuntimeContractError(
                "eval.forward planned_step_id must be positive",
                code="eval_forward.planned_step_id",
                context={"planned_step_id": planned_step_id},
            )
        if not self.eval_source:
            raise RuntimeContractError(
                "eval.forward requires an explicit eval source",
                code="eval_forward.source_required",
            )

        was_training = getattr(self.model, "training", None)
        eval_method = getattr(self.model, "eval", None)
        train_method = getattr(self.model, "train", None)
        try:
            if callable(eval_method):
                eval_method()
            with torch.no_grad():
                example_count, pack_count, scalars = self._run_forward_only(
                    planned_step_id=planned_step_id
                )
        finally:
            if was_training is not None and callable(train_method):
                train_method(bool(was_training))

        return ForwardEvalObservation(
            planned_step_id=planned_step_id,
            split=EVAL_FORWARD_SPLIT,
            trigger_reasons=tuple(str(reason) for reason in trigger_reasons),
            example_count=example_count,
            pack_count=pack_count,
            scalars=scalars,
        )

    def _run_forward_only(
        self, *, planned_step_id: int
    ) -> tuple[int, int, dict[str, float | None]]:
        micro_steps = tuple(self.micro_step_stream)
        if not micro_steps:
            raise RuntimeContractError(
                "eval.forward requires at least one eval micro-step",
                code="eval_forward.empty_stream",
                context={"planned_step_id": planned_step_id},
            )
        if _supports_streaming_loss(self.loss_runner):
            return self._run_streaming_forward_only(
                micro_steps, planned_step_id=planned_step_id
            )

        contexts: list[Any] = []
        example_count = 0
        for local_index, micro_step in enumerate(micro_steps):
            example_count += len(tuple(micro_step.encoded_examples))
            moved = self._move_micro_step(
                micro_step,
                planned_step_id=planned_step_id,
                local_micro_step_index=local_index,
            )
            forward_result = self.qwen_forward(
                _runtime_model(self.runtime, self.model), moved
            )
            contexts.append(self.loss_context_factory(moved, forward_result))
        loss_bundle = self.loss_runner.compute(tuple(contexts))
        scalars = _metric_scalars(loss_bundle, _artifact(loss_bundle))
        return example_count, len(micro_steps), self._gather_scalars(
            scalars, planned_step_id=planned_step_id
        )

    def _run_streaming_forward_only(
        self,
        micro_steps: Sequence[SupervisedMicroStep],
        *,
        planned_step_id: int,
    ) -> tuple[int, int, dict[str, float | None]]:
        plan = self.loss_runner.prepare_planned_step(tuple(micro_steps))
        micro_loss_artifacts: list[Mapping[str, Any]] = []
        example_count = 0
        for local_index, micro_step in enumerate(micro_steps):
            example_count += len(tuple(micro_step.encoded_examples))
            moved = self._move_micro_step(
                micro_step,
                planned_step_id=planned_step_id,
                local_micro_step_index=local_index,
            )
            forward_result = self.qwen_forward(
                _runtime_model(self.runtime, self.model), moved
            )
            context = self.loss_context_factory(moved, forward_result)
            loss_bundle = self.loss_runner.compute_micro_step(
                context, plan, local_micro_step_index=local_index
            )
            micro_loss_artifacts.append(_artifact(loss_bundle))
            del loss_bundle, context, forward_result, moved

        loss_artifact = self.loss_runner.finalize_planned_step(
            tuple(dict(item) for item in micro_loss_artifacts), plan
        )
        scalars = _metric_scalars(None, loss_artifact)
        return example_count, len(micro_steps), self._gather_scalars(
            scalars, planned_step_id=planned_step_id
        )

    def _move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        if self.runtime is None:
            return micro_step
        return self.runtime.move_micro_step(
            micro_step,
            planned_step_id=planned_step_id,
            local_micro_step_index=local_micro_step_index,
        )

    def _gather_scalars(
        self,
        scalars: Mapping[str, float | None],
        *,
        planned_step_id: int,
    ) -> dict[str, float | None]:
        gather = getattr(self.runtime, "gather_metrics", None)
        if not callable(gather):
            return dict(scalars)
        finite_or_nonfinite = {
            name: float(value) for name, value in scalars.items() if value is not None
        }
        gathered = gather(
            finite_or_nonfinite,
            planned_step_id=planned_step_id,
            split=EVAL_FORWARD_SPLIT,
        )
        reduced = gathered.get("metrics") if isinstance(gathered, Mapping) else None
        if not isinstance(reduced, Mapping):
            raise RuntimeContractError(
                "eval.forward runtime metric reduction returned no scalar mapping",
                code="eval_forward.metric_reduction",
                context={"planned_step_id": planned_step_id},
            )
        result = {str(name): _optional_float(value) for name, value in reduced.items()}
        for name, value in scalars.items():
            if value is None:
                result.setdefault(name, None)
        return result


def _runtime_model(runtime: EvalRuntimeBoundary | None, fallback_model: Any) -> Any:
    return getattr(runtime, "model", fallback_model)


def _artifact(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_artifact_dict"):
        return value.to_artifact_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {"type": type(value).__name__}


def _metric_scalars(
    loss_bundle: Any, loss_artifact: Mapping[str, Any]
) -> dict[str, float | None]:
    metrics = getattr(loss_bundle, "metrics", None)
    if not isinstance(metrics, Mapping):
        metrics = loss_artifact.get("metrics")
    if isinstance(metrics, Mapping):
        return {str(name): _optional_float(value) for name, value in metrics.items()}
    return {"loss/total": _optional_float(loss_artifact.get("total_loss"))}


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _supports_streaming_loss(loss_runner: Any) -> bool:
    return all(
        callable(getattr(loss_runner, name, None))
        for name in (
            "prepare_planned_step",
            "compute_micro_step",
            "finalize_planned_step",
        )
    )


__all__ = ["EVAL_FORWARD_SPLIT", "ForwardEvalObservation", "ForwardEvalRunner"]
