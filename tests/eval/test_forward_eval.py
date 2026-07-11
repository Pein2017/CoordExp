from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.eval.forward import ForwardEvalObservation, ForwardEvalRunner
from src.training.supervised_trainer import SupervisedMicroStep


def test_forward_eval_returns_one_pure_wide_observation_and_restores_mode(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    model = FakeModel(log)
    runtime = FakeEvalRuntime(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(
            _micro_step(0, encoded_examples=("a", "b")),
            _micro_step(1, encoded_examples=("c",)),
        ),
        loss_runner=FakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
        runtime=runtime,
    )

    result = runner.run(planned_step_id=4, trigger_reasons=("milestone_40pct",))

    assert isinstance(result, ForwardEvalObservation)
    assert result.planned_step_id == 4
    assert result.split == "eval"
    assert result.trigger_reasons == ("milestone_40pct",)
    assert result.example_count == 3
    assert result.pack_count == 2
    assert result.scalars == {
        "acc_top1": 0.5,
        "acc_top5": 0.75,
        "loss/total": 1.25,
    }
    assert result.to_logging_row() == {
        "step": 4,
        "split": "eval",
        "trigger_reasons": ["milestone_40pct"],
        "example_count": 3,
        "pack_count": 2,
        "acc_top1": 0.5,
        "acc_top5": 0.75,
        "loss/total": 1.25,
    }
    assert model.training is True
    assert runtime.gathered == [(4, "eval", dict(FakeLossBundle.metrics))]
    assert list(tmp_path.iterdir()) == []
    assert log == [
        "model.eval",
        "runtime.move:4:0",
        "forward:0:grad=False:training=False",
        "context:0",
        "runtime.move:4:1",
        "forward:1:grad=False:training=False",
        "context:1",
        "loss:2:grad=False",
        "runtime.gather:4:eval",
        "model.train:True",
    ]


def test_forward_eval_streaming_counts_and_wide_scalars() -> None:
    log: list[str] = []
    result = ForwardEvalRunner(
        model=FakeModel(log),
        micro_step_stream=(
            _micro_step(0, encoded_examples=("a", "b")),
            _micro_step(1, encoded_examples=("c",)),
            _micro_step(2, encoded_examples=("d",)),
        ),
        loss_runner=StreamingFakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
    ).run(planned_step_id=136, trigger_reasons=("every_fraction:0.4",))

    assert result.example_count == 4
    assert result.pack_count == 3
    assert result.scalars["loss/total"] == pytest.approx(1.25)
    assert "streaming.prepare:3" in log
    assert "streaming.finalize:3" in log


def test_forward_eval_leaves_nonfinite_values_for_writer_normalization() -> None:
    runner = ForwardEvalRunner(
        model=object(),
        micro_step_stream=(_micro_step(0),),
        loss_runner=NonfiniteLossRunner([]),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )

    row = runner.run(planned_step_id=2, trigger_reasons=("scheduled",)).to_logging_row()

    assert math.isnan(row["loss/total"])
    assert math.isinf(row["diagnostic/max_logit"])
    assert "non_finite_fields" not in row


def test_forward_eval_has_no_writer_or_event_coupling() -> None:
    import inspect
    import src.eval.forward as module

    source = inspect.getsource(module)
    assert "RunArtifactManager" not in source
    assert "MetricStreamEvent" not in source
    assert "write_" not in source
    assert "summary_path" not in source


def test_forward_eval_requires_explicit_source_before_consuming_stream() -> None:
    consumed = False

    def stream() -> Any:
        nonlocal consumed
        consumed = True
        yield _micro_step(0)

    runner = ForwardEvalRunner(
        model=object(),
        micro_step_stream=stream(),
        loss_runner=FakeLossRunner([]),
        eval_source=None,
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        runner.run(planned_step_id=1, trigger_reasons=("scheduled",))
    assert exc_info.value.code == "eval_forward.source_required"
    assert consumed is False


def test_forward_eval_restores_model_mode_when_forward_raises() -> None:
    log: list[str] = []
    model = FakeModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(_micro_step(0),),
        loss_runner=FakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_raising_forward(log),
        loss_context_factory=_loss_context(log),
    )

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        runner.run(planned_step_id=4, trigger_reasons=("scheduled",))
    assert model.training is True
    assert log == [
        "model.eval",
        "forward_raises:0:grad=False:training=False",
        "model.train:True",
    ]


def test_forward_eval_restores_model_mode_when_eval_raises() -> None:
    log: list[str] = []
    model = EvalRaisesModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(_micro_step(0),),
        loss_runner=FakeLossRunner(log),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
    )

    with pytest.raises(RuntimeError, match="synthetic eval failure"):
        runner.run(planned_step_id=4, trigger_reasons=("scheduled",))
    assert model.training is True
    assert log == ["model.eval_raises", "model.train:True"]


@dataclass
class FakeForwardResult:
    logits: torch.Tensor
    receipt: dict[str, Any]


class FakeModel:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.training = True

    def eval(self) -> None:
        self.log.append("model.eval")
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.log.append(f"model.train:{mode}")
        self.training = mode


class EvalRaisesModel(FakeModel):
    def eval(self) -> None:
        self.log.append("model.eval_raises")
        self.training = False
        raise RuntimeError("synthetic eval failure")


class FakeEvalRuntime:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.gathered: list[tuple[int, str, dict[str, float]]] = []

    def move_micro_step(self, micro_step: SupervisedMicroStep, *, planned_step_id: int, local_micro_step_index: int) -> SupervisedMicroStep:
        self.log.append(f"runtime.move:{planned_step_id}:{local_micro_step_index}")
        return micro_step

    def gather_metrics(self, metrics: dict[str, float], *, planned_step_id: int, split: str) -> dict[str, Any]:
        self.log.append(f"runtime.gather:{planned_step_id}:{split}")
        self.gathered.append((planned_step_id, split, dict(metrics)))
        return {"metrics": dict(metrics), "reduction": "single_rank"}


class FakeLossRunner:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def compute(self, contexts: tuple[Any, ...]) -> "FakeLossBundle":
        self.log.append(f"loss:{len(contexts)}:grad={torch.is_grad_enabled()}")
        return FakeLossBundle()


class FakeLossBundle:
    metrics = {"acc_top1": 0.5, "acc_top5": 0.75, "loss/total": 1.25}

    def to_artifact_dict(self) -> dict[str, Any]:
        return {"total_loss": 1.25, "metrics": dict(self.metrics)}


class NonfiniteLossRunner(FakeLossRunner):
    def compute(self, contexts: tuple[Any, ...]) -> dict[str, Any]:
        del contexts
        return {"metrics": {"loss/total": float("nan"), "diagnostic/max_logit": float("inf")}}


class StreamingFakeLossRunner(FakeLossRunner):
    def compute(self, contexts: tuple[Any, ...]) -> FakeLossBundle:
        del contexts
        raise AssertionError("streaming eval path must not retain all contexts")

    def prepare_planned_step(self, micro_steps: tuple[SupervisedMicroStep, ...]) -> dict[str, int]:
        self.log.append(f"streaming.prepare:{len(micro_steps)}")
        return {"micro_step_count": len(micro_steps)}

    def compute_micro_step(self, context: Any, plan: dict[str, int], *, local_micro_step_index: int) -> FakeLossBundle:
        del context, plan
        self.log.append(f"streaming.loss:{local_micro_step_index}")
        return FakeLossBundle()

    def finalize_planned_step(self, micro_loss_artifacts: tuple[dict[str, Any], ...], plan: dict[str, int]) -> dict[str, Any]:
        del plan
        self.log.append(f"streaming.finalize:{len(micro_loss_artifacts)}")
        return {"metrics": dict(FakeLossBundle.metrics)}


def _qwen_forward(log: list[str]):
    def forward(model: FakeModel, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        index = int(str(micro_step.pack).split("-")[1])
        log.append(f"forward:{index}:grad={torch.is_grad_enabled()}:training={getattr(model, 'training', None)}")
        return FakeForwardResult(torch.zeros(1, 2, 3), {"pack": index})
    return forward


def _raising_forward(log: list[str]):
    def forward(model: FakeModel, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        index = int(str(micro_step.pack).split("-")[1])
        log.append(f"forward_raises:{index}:grad={torch.is_grad_enabled()}:training={getattr(model, 'training', None)}")
        raise RuntimeError("synthetic forward failure")
    return forward


def _loss_context(log: list[str]):
    def factory(micro_step: SupervisedMicroStep, forward_result: FakeForwardResult) -> dict[str, Any]:
        index = int(str(micro_step.pack).split("-")[1])
        log.append(f"context:{index}")
        return {"pack": micro_step.pack, "shape": tuple(forward_result.logits.shape)}
    return factory


def _micro_step(index: int, *, encoded_examples: tuple[str, ...] = ("example",)) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=encoded_examples,
        position_inputs=f"positions-{index}",
        token_sequence=object(),
        vocab_groups=object(),
    )
