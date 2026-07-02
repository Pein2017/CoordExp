"""Packed forward-only evaluation runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from src.artifacts import MetricStreamEvent, RunArtifactManager
from src.common.errors import RuntimeContractError
from src.training.supervised_trainer import (
    LossContextFactory,
    LossRunnerBoundary,
    QwenForwardFn,
    SupervisedMicroStep,
    _default_loss_context,
    _default_qwen_forward,
)


EVAL_FORWARD_SPLIT = "eval.forward"


class EvalRuntimeBoundary(Protocol):
    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep: ...


@dataclass(frozen=True)
class ForwardEvalResult:
    planned_step_id: int
    summary_path: Path
    summary: Mapping[str, Any]
    metric_events: tuple[MetricStreamEvent, ...]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "planned_step_id": self.planned_step_id,
            "summary_path": str(self.summary_path),
            "summary": dict(self.summary),
            "metric_events": [event.to_record() for event in self.metric_events],
        }


class ForwardEvalRunner:
    def __init__(
        self,
        *,
        model: Any,
        micro_step_stream: Iterable[SupervisedMicroStep],
        loss_runner: LossRunnerBoundary,
        artifact_manager: RunArtifactManager,
        eval_source: Mapping[str, Any] | None,
        qwen_forward: QwenForwardFn | None = None,
        loss_context_factory: LossContextFactory | None = None,
        runtime: EvalRuntimeBoundary | None = None,
    ) -> None:
        self.model = model
        self.micro_step_stream = micro_step_stream
        self.loss_runner = loss_runner
        self.artifact_manager = artifact_manager
        self.eval_source = None if eval_source is None else dict(eval_source)
        self.qwen_forward = qwen_forward or _default_qwen_forward
        self.loss_context_factory = loss_context_factory or _default_loss_context
        self.runtime = runtime

    def run(
        self,
        *,
        planned_step_id: int,
        trigger_reasons: Sequence[str],
        optimizer_update_status: str,
        finite_status: str,
        warning_status: str,
    ) -> ForwardEvalResult:
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
                summary, metric_events = self._run_forward_only(
                    planned_step_id=planned_step_id,
                    trigger_reasons=trigger_reasons,
                    optimizer_update_status=optimizer_update_status,
                    finite_status=finite_status,
                    warning_status=warning_status,
                )
        finally:
            if was_training is not None and callable(train_method):
                train_method(bool(was_training))
        summary_path = self.artifact_manager.write_eval_forward_summary(
            planned_step_id=planned_step_id,
            summary=summary,
        )
        for event in metric_events:
            self.artifact_manager.append_metric_event(event)
        return ForwardEvalResult(
            planned_step_id=planned_step_id,
            summary_path=summary_path,
            summary=summary,
            metric_events=metric_events,
        )

    def _run_forward_only(
        self,
        *,
        planned_step_id: int,
        trigger_reasons: Sequence[str],
        optimizer_update_status: str,
        finite_status: str,
        warning_status: str,
    ) -> tuple[dict[str, Any], tuple[MetricStreamEvent, ...]]:
        micro_steps = tuple(self.micro_step_stream)
        if not micro_steps:
            raise RuntimeContractError(
                "eval.forward requires at least one eval micro-step",
                code="eval_forward.empty_stream",
                context={"planned_step_id": planned_step_id},
            )
        if _supports_streaming_loss(self.loss_runner):
            return self._run_streaming_forward_only(
                micro_steps,
                planned_step_id=planned_step_id,
                trigger_reasons=trigger_reasons,
                optimizer_update_status=optimizer_update_status,
                finite_status=finite_status,
                warning_status=warning_status,
            )

        contexts: list[Any] = []
        qwen_receipts: list[Mapping[str, Any]] = []
        example_count = 0
        for local_index, micro_step in enumerate(micro_steps):
            example_count += len(tuple(micro_step.encoded_examples))
            moved = (
                self.runtime.move_micro_step(
                    micro_step,
                    planned_step_id=planned_step_id,
                    local_micro_step_index=local_index,
                )
                if self.runtime is not None
                else micro_step
            )
            forward_result = self.qwen_forward(
                _runtime_model(self.runtime, self.model),
                moved,
            )
            contexts.append(self.loss_context_factory(moved, forward_result))
            qwen_receipts.append(_receipt_artifact(forward_result))
        loss_bundle = self.loss_runner.compute(tuple(contexts))
        loss_artifact = _artifact(loss_bundle)
        metric_summary = _metric_summary(loss_bundle, loss_artifact)
        metric_events = tuple(
            MetricStreamEvent(
                event_type="metric",
                planned_step_id=planned_step_id,
                split=EVAL_FORWARD_SPLIT,
                name=name,
                value=None if metric_summary[name] is None else float(metric_summary[name]),
                trigger_reasons=trigger_reasons,
                optimizer_update_status=optimizer_update_status,
                finite_status=finite_status,
                warning_status=warning_status,
            )
            for name in sorted(metric_summary)
        )
        summary_relative_path = f"eval/forward/step-{planned_step_id}.json"
        metric_stream_path = f"metrics/{EVAL_FORWARD_SPLIT}.jsonl"
        summary = {
            "planned_step_id": planned_step_id,
            "split": EVAL_FORWARD_SPLIT,
            "trigger_reasons": [str(reason) for reason in trigger_reasons],
            "example_count": example_count,
            "pack_count": len(contexts),
            "eval_source": dict(self.eval_source or {}),
            "loss_summary": loss_artifact,
            "metric_summary": metric_summary,
            "artifact_links": {
                "summary": summary_relative_path,
                "metric_stream": metric_stream_path,
            },
            "optimizer_update_status": optimizer_update_status,
            "finite_status": finite_status,
            "warning_status": warning_status,
            "qwen_forward_receipts": [dict(receipt) for receipt in qwen_receipts],
        }
        return summary, metric_events

    def _run_streaming_forward_only(
        self,
        micro_steps: Sequence[SupervisedMicroStep],
        *,
        planned_step_id: int,
        trigger_reasons: Sequence[str],
        optimizer_update_status: str,
        finite_status: str,
        warning_status: str,
    ) -> tuple[dict[str, Any], tuple[MetricStreamEvent, ...]]:
        plan = self.loss_runner.prepare_planned_step(tuple(micro_steps))
        micro_loss_artifacts: list[Mapping[str, Any]] = []
        qwen_receipts: list[Mapping[str, Any]] = []
        example_count = 0
        for local_index, micro_step in enumerate(micro_steps):
            example_count += len(tuple(micro_step.encoded_examples))
            moved = (
                self.runtime.move_micro_step(
                    micro_step,
                    planned_step_id=planned_step_id,
                    local_micro_step_index=local_index,
                )
                if self.runtime is not None
                else micro_step
            )
            forward_result = self.qwen_forward(
                _runtime_model(self.runtime, self.model),
                moved,
            )
            context = self.loss_context_factory(moved, forward_result)
            loss_bundle = self.loss_runner.compute_micro_step(
                context,
                plan,
                local_micro_step_index=local_index,
            )
            micro_loss_artifacts.append(_artifact(loss_bundle))
            qwen_receipts.append(_receipt_artifact(forward_result))
            del loss_bundle, context, forward_result, moved

        loss_artifact = self.loss_runner.finalize_planned_step(
            tuple(dict(item) for item in micro_loss_artifacts),
            plan,
        )
        metric_summary = _metric_summary_from_artifact(loss_artifact)
        metric_events = _metric_events(
            metric_summary,
            planned_step_id=planned_step_id,
            trigger_reasons=trigger_reasons,
            optimizer_update_status=optimizer_update_status,
            finite_status=finite_status,
            warning_status=warning_status,
        )
        summary = _summary(
            planned_step_id=planned_step_id,
            trigger_reasons=trigger_reasons,
            example_count=example_count,
            pack_count=len(micro_steps),
            eval_source=self.eval_source,
            loss_artifact=loss_artifact,
            metric_summary=metric_summary,
            optimizer_update_status=optimizer_update_status,
            finite_status=finite_status,
            warning_status=warning_status,
            qwen_receipts=qwen_receipts,
        )
        return summary, metric_events


def _runtime_model(runtime: EvalRuntimeBoundary | None, fallback_model: Any) -> Any:
    return getattr(runtime, "model", fallback_model)


def _receipt_artifact(forward_result: Any) -> Mapping[str, Any]:
    receipt = getattr(forward_result, "receipt", None)
    if hasattr(receipt, "to_artifact_dict"):
        return receipt.to_artifact_dict()
    if isinstance(receipt, Mapping):
        return dict(receipt)
    return {"receipt_type": type(receipt).__name__}


def _artifact(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_artifact_dict"):
        return value.to_artifact_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {"type": type(value).__name__}


def _metric_summary(loss_bundle: Any, loss_artifact: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = getattr(loss_bundle, "metrics", None)
    if isinstance(metrics, Mapping):
        return {
            str(name): (None if value is None else float(value))
            for name, value in metrics.items()
        }
    artifact_metrics = loss_artifact.get("metrics")
    if isinstance(artifact_metrics, Mapping):
        return {
            str(name): (None if value is None else float(value))
            for name, value in artifact_metrics.items()
        }
    total_loss = loss_artifact.get("total_loss")
    return {"loss/total": None if total_loss is None else float(total_loss)}


def _metric_summary_from_artifact(loss_artifact: Mapping[str, Any]) -> dict[str, float | None]:
    artifact_metrics = loss_artifact.get("metrics")
    if isinstance(artifact_metrics, Mapping):
        return {
            str(name): (None if value is None else float(value))
            for name, value in artifact_metrics.items()
        }
    total_loss = loss_artifact.get("total_loss")
    return {"loss/total": None if total_loss is None else float(total_loss)}


def _metric_events(
    metric_summary: Mapping[str, float | None],
    *,
    planned_step_id: int,
    trigger_reasons: Sequence[str],
    optimizer_update_status: str,
    finite_status: str,
    warning_status: str,
) -> tuple[MetricStreamEvent, ...]:
    return tuple(
        MetricStreamEvent(
            event_type="metric",
            planned_step_id=planned_step_id,
            split=EVAL_FORWARD_SPLIT,
            name=name,
            value=None if metric_summary[name] is None else float(metric_summary[name]),
            trigger_reasons=trigger_reasons,
            optimizer_update_status=optimizer_update_status,
            finite_status=finite_status,
            warning_status=warning_status,
        )
        for name in sorted(metric_summary)
    )


def _summary(
    *,
    planned_step_id: int,
    trigger_reasons: Sequence[str],
    example_count: int,
    pack_count: int,
    eval_source: Mapping[str, Any] | None,
    loss_artifact: Mapping[str, Any],
    metric_summary: Mapping[str, float | None],
    optimizer_update_status: str,
    finite_status: str,
    warning_status: str,
    qwen_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary_relative_path = f"eval/forward/step-{planned_step_id}.json"
    metric_stream_path = f"metrics/{EVAL_FORWARD_SPLIT}.jsonl"
    return {
        "planned_step_id": planned_step_id,
        "split": EVAL_FORWARD_SPLIT,
        "trigger_reasons": [str(reason) for reason in trigger_reasons],
        "example_count": int(example_count),
        "pack_count": int(pack_count),
        "eval_source": dict(eval_source or {}),
        "loss_summary": dict(loss_artifact),
        "metric_summary": dict(metric_summary),
        "artifact_links": {
            "summary": summary_relative_path,
            "metric_stream": metric_stream_path,
        },
        "optimizer_update_status": optimizer_update_status,
        "finite_status": finite_status,
        "warning_status": warning_status,
        "qwen_forward_receipts": [dict(receipt) for receipt in qwen_receipts],
    }


def _supports_streaming_loss(loss_runner: Any) -> bool:
    return all(
        callable(getattr(loss_runner, name, None))
        for name in (
            "prepare_planned_step",
            "compute_micro_step",
            "finalize_planned_step",
        )
    )


__all__ = ["EVAL_FORWARD_SPLIT", "ForwardEvalResult", "ForwardEvalRunner"]
