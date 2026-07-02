from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from src.artifacts import RunArtifactManager
from src.common.errors import ArtifactContractError, RuntimeContractError
from src.config.models import RunDirectory
from src.eval.forward import ForwardEvalRunner
from src.training.supervised_trainer import SupervisedMicroStep


def test_forward_eval_writes_summary_metric_stream_and_restores_model_mode(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    manager = _manager(tmp_path)
    model = FakeModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(
            _micro_step(0, encoded_examples=("example-a", "example-b")),
            _micro_step(1, encoded_examples=("example-c",)),
        ),
        loss_runner=FakeLossRunner(log),
        artifact_manager=manager,
        eval_source={"path": "tests/fixtures/eval.jsonl", "sample_limit": 3},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
        runtime=FakeEvalRuntime(log),
    )

    result = runner.run(
        planned_step_id=4,
        trigger_reasons=("milestone_40pct",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )

    summary_path = tmp_path / "run-a" / "eval" / "forward" / "step-4.json"
    assert result.summary_path == summary_path
    assert summary_path.exists()
    assert not (tmp_path / "run-a" / "eval" / "forward" / "step-000004.json").exists()
    assert model.training is True
    assert log == [
        "model.eval",
        "runtime.move:4:0",
        "forward:0:grad=False:training=False",
        "context:0",
        "runtime.move:4:1",
        "forward:1:grad=False:training=False",
        "context:1",
        "loss:2:grad=False",
        "model.train:True",
    ]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["planned_step_id"] == 4
    assert summary["split"] == "eval.forward"
    assert summary["trigger_reasons"] == ["milestone_40pct"]
    assert summary["example_count"] == 3
    assert summary["pack_count"] == 2
    assert summary["eval_source"] == {
        "path": "tests/fixtures/eval.jsonl",
        "sample_limit": 3,
    }
    assert summary["loss_summary"]["total_loss"] == 1.25
    assert summary["metric_summary"]["acc_top1"] == 0.5
    assert summary["metric_summary"]["acc_top5"] == 0.75
    assert summary["artifact_links"]["summary"] == "eval/forward/step-4.json"
    assert summary["artifact_links"]["metric_stream"] == "metrics/eval.forward.jsonl"
    assert summary["qwen_forward_receipts"] == [
        {"pack": 0, "forward_only": True},
        {"pack": 1, "forward_only": True},
    ]

    metric_records = [
        json.loads(line)
        for line in (tmp_path / "run-a" / "metrics" / "eval.forward.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["name"] for record in metric_records] == [
        "acc_top1",
        "acc_top5",
        "loss/total",
    ]
    assert metric_records[0]["split"] == "eval.forward"
    assert metric_records[0]["planned_step_id"] == 4
    assert metric_records[0]["optimizer_update_status"] == "applied"
    assert metric_records[0]["selector_eligible"] is True

    manifest = manager.read_manifest()
    assert manifest["eval"]["forward"]["step-4"] == "eval/forward/step-4.json"


def test_forward_eval_streams_loss_without_retaining_all_contexts(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    manager = _manager(tmp_path)
    model = FakeModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(
            _micro_step(0, encoded_examples=("example-a", "example-b")),
            _micro_step(1, encoded_examples=("example-c",)),
            _micro_step(2, encoded_examples=("example-d",)),
        ),
        loss_runner=StreamingFakeLossRunner(log),
        artifact_manager=manager,
        eval_source={"path": "tests/fixtures/eval.jsonl", "sample_limit": None},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
        runtime=FakeEvalRuntime(log),
    )

    result = runner.run(
        planned_step_id=136,
        trigger_reasons=("every_fraction:0.4",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )

    assert model.training is True
    assert log == [
        "model.eval",
        "streaming.prepare:3",
        "runtime.move:136:0",
        "forward:0:grad=False:training=False",
        "context:0",
        "streaming.loss:0",
        "runtime.move:136:1",
        "forward:1:grad=False:training=False",
        "context:1",
        "streaming.loss:1",
        "runtime.move:136:2",
        "forward:2:grad=False:training=False",
        "context:2",
        "streaming.loss:2",
        "streaming.finalize:3",
        "model.train:True",
    ]

    assert result.summary["example_count"] == 4
    assert result.summary["pack_count"] == 3
    assert result.summary["metric_summary"]["loss/total"] == pytest.approx(1.25)
    assert result.summary["loss_summary"]["diagnostics"] == {
        "normalizer_scope": "planned_step_streaming"
    }
    assert [receipt["pack"] for receipt in result.summary["qwen_forward_receipts"]] == [
        0,
        1,
        2,
    ]

    metric_records = [
        json.loads(line)
        for line in (tmp_path / "run-a" / "metrics" / "eval.forward.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["name"] for record in metric_records] == [
        "acc_top1",
        "acc_top5",
        "loss/total",
    ]


def test_forward_eval_requires_explicit_eval_source_before_consuming_stream(
    tmp_path: Path,
) -> None:
    consumed = False

    def stream() -> Any:
        nonlocal consumed
        consumed = True
        yield _micro_step(0)

    runner = ForwardEvalRunner(
        model=object(),
        micro_step_stream=stream(),
        loss_runner=FakeLossRunner([]),
        artifact_manager=_manager(tmp_path),
        eval_source=None,
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        runner.run(
            planned_step_id=1,
            trigger_reasons=("milestone_40pct",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )

    assert exc_info.value.code == "eval_forward.source_required"
    assert consumed is False


def test_forward_eval_restores_model_mode_when_forward_raises(tmp_path: Path) -> None:
    log: list[str] = []
    model = FakeModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(_micro_step(0),),
        loss_runner=FakeLossRunner(log),
        artifact_manager=_manager(tmp_path),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_raising_forward(log),
        loss_context_factory=_loss_context(log),
    )

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        runner.run(
            planned_step_id=4,
            trigger_reasons=("milestone_40pct",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )

    assert model.training is True
    assert log == [
        "model.eval",
        "forward_raises:0:grad=False:training=False",
        "model.train:True",
    ]
    assert not (tmp_path / "run-a" / "eval" / "forward" / "step-4.json").exists()


def test_forward_eval_restores_model_mode_when_eval_raises(tmp_path: Path) -> None:
    log: list[str] = []
    model = EvalRaisesModel(log)
    runner = ForwardEvalRunner(
        model=model,
        micro_step_stream=(_micro_step(0),),
        loss_runner=FakeLossRunner(log),
        artifact_manager=_manager(tmp_path),
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward(log),
        loss_context_factory=_loss_context(log),
    )

    with pytest.raises(RuntimeError, match="synthetic eval failure"):
        runner.run(
            planned_step_id=4,
            trigger_reasons=("milestone_40pct",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )

    assert model.training is True
    assert log == ["model.eval_raises", "model.train:True"]
    assert not (tmp_path / "run-a" / "eval" / "forward" / "step-4.json").exists()


def test_forward_eval_repairs_summary_manifest_failure_without_duplicate_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    runner = ForwardEvalRunner(
        model=object(),
        micro_step_stream=(_micro_step(0),),
        loss_runner=FakeLossRunner([]),
        artifact_manager=manager,
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )
    _fail_next_manifest_write(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        runner.run(
            planned_step_id=4,
            trigger_reasons=("milestone_40pct",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )

    assert exc_info.value.code == "test.manifest_write_failed"
    assert (tmp_path / "run-a" / "eval" / "forward" / "step-4.json").exists()
    assert not (tmp_path / "run-a" / "metrics" / "eval.forward.jsonl").exists()

    retry = ForwardEvalRunner(
        model=object(),
        micro_step_stream=(_micro_step(0),),
        loss_runner=FakeLossRunner([]),
        artifact_manager=manager,
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )
    retry.run(
        planned_step_id=4,
        trigger_reasons=("milestone_40pct",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )

    metric_records = (
        tmp_path / "run-a" / "metrics" / "eval.forward.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(metric_records) == 3
    assert manager.read_manifest()["eval"]["forward"]["step-4"] == (
        "eval/forward/step-4.json"
    )


def test_forward_eval_successful_retry_does_not_duplicate_metrics(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)

    _runner_for_retry(manager).run(
        planned_step_id=4,
        trigger_reasons=("milestone_40pct",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )
    _runner_for_retry(manager).run(
        planned_step_id=4,
        trigger_reasons=("milestone_40pct",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )

    records = (
        tmp_path / "run-a" / "metrics" / "eval.forward.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(records) == 3
    assert sorted(json.loads(line)["name"] for line in records) == [
        "acc_top1",
        "acc_top5",
        "loss/total",
    ]


def test_forward_eval_metric_append_retry_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    _fail_second_metric_append_once(monkeypatch)

    with pytest.raises(ArtifactContractError) as exc_info:
        _runner_for_retry(manager).run(
            planned_step_id=4,
            trigger_reasons=("milestone_40pct",),
            optimizer_update_status="applied",
            finite_status="finite",
            warning_status="none",
        )

    assert exc_info.value.code == "test.metric_append_failed"
    partial_records = (
        tmp_path / "run-a" / "metrics" / "eval.forward.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(partial_records) == 1

    _runner_for_retry(manager).run(
        planned_step_id=4,
        trigger_reasons=("milestone_40pct",),
        optimizer_update_status="applied",
        finite_status="finite",
        warning_status="none",
    )

    records = (
        tmp_path / "run-a" / "metrics" / "eval.forward.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(records) == 3
    assert sorted(json.loads(line)["name"] for line in records) == [
        "acc_top1",
        "acc_top5",
        "loss/total",
    ]


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

    def move_micro_step(
        self,
        micro_step: SupervisedMicroStep,
        *,
        planned_step_id: int,
        local_micro_step_index: int,
    ) -> SupervisedMicroStep:
        self.log.append(f"runtime.move:{planned_step_id}:{local_micro_step_index}")
        return micro_step


class FakeLossRunner:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def compute(self, contexts: tuple[Any, ...]) -> "FakeLossBundle":
        self.log.append(f"loss:{len(contexts)}:grad={torch.is_grad_enabled()}")
        return FakeLossBundle()


class FakeLossBundle:
    metrics = {"acc_top1": 0.5, "acc_top5": 0.75, "loss/total": 1.25}

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "total_loss": 1.25,
            "metrics": dict(self.metrics),
            "counts": {"eligible_tokens": 8},
        }


class StreamingFakeLossRunner(FakeLossRunner):
    def compute(self, contexts: tuple[Any, ...]) -> "FakeLossBundle":
        del contexts
        raise AssertionError("streaming eval path must not retain all contexts")

    def prepare_planned_step(
        self,
        micro_steps: tuple[SupervisedMicroStep, ...],
    ) -> dict[str, int]:
        self.log.append(f"streaming.prepare:{len(micro_steps)}")
        return {"micro_step_count": len(micro_steps)}

    def compute_micro_step(
        self,
        context: Any,
        plan: dict[str, int],
        *,
        local_micro_step_index: int,
    ) -> "FakeLossBundle":
        del context, plan
        self.log.append(f"streaming.loss:{local_micro_step_index}")
        return FakeLossBundle()

    def finalize_planned_step(
        self,
        micro_loss_artifacts: tuple[dict[str, Any], ...],
        plan: dict[str, int],
    ) -> dict[str, Any]:
        del plan
        self.log.append(f"streaming.finalize:{len(micro_loss_artifacts)}")
        return {
            "total_loss": 1.25,
            "metrics": {"acc_top1": 0.5, "acc_top5": 0.75, "loss/total": 1.25},
            "counts": {"eligible_tokens": 8},
            "diagnostics": {"normalizer_scope": "planned_step_streaming"},
        }


def _qwen_forward(log: list[str]):
    def forward(model: FakeModel, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        pack_index = int(str(micro_step.pack).split("-")[1])
        log.append(
            f"forward:{pack_index}:grad={torch.is_grad_enabled()}:"
            f"training={getattr(model, 'training', None)}"
        )
        return FakeForwardResult(
            logits=torch.zeros(1, 2, 3),
            receipt={"pack": pack_index, "forward_only": True},
        )

    return forward


def _raising_forward(log: list[str]):
    def forward(model: FakeModel, micro_step: SupervisedMicroStep) -> FakeForwardResult:
        pack_index = int(str(micro_step.pack).split("-")[1])
        log.append(
            f"forward_raises:{pack_index}:grad={torch.is_grad_enabled()}:"
            f"training={getattr(model, 'training', None)}"
        )
        raise RuntimeError("synthetic forward failure")

    return forward


def _loss_context(log: list[str]):
    def factory(
        micro_step: SupervisedMicroStep,
        forward_result: FakeForwardResult,
    ) -> dict[str, Any]:
        pack_index = int(str(micro_step.pack).split("-")[1])
        log.append(f"context:{pack_index}")
        return {"pack": micro_step.pack, "logits_shape": tuple(forward_result.logits.shape)}

    return factory


def _micro_step(
    index: int,
    *,
    encoded_examples: tuple[str, ...] = ("example",),
) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"pack-{index}",
        encoded_examples=encoded_examples,
        position_inputs=f"positions-{index}",
        token_sequence=object(),
        vocab_groups=object(),
    )


def _runner_for_retry(manager: RunArtifactManager) -> ForwardEvalRunner:
    return ForwardEvalRunner(
        model=object(),
        micro_step_stream=(_micro_step(0),),
        loss_runner=FakeLossRunner([]),
        artifact_manager=manager,
        eval_source={"path": "tests/fixtures/eval.jsonl"},
        qwen_forward=_qwen_forward([]),
        loss_context_factory=_loss_context([]),
    )


def _manager(tmp_path: Path) -> RunArtifactManager:
    return RunArtifactManager.initialize(
        run_directory=RunDirectory(
            run_name="run-a",
            artifact_root=tmp_path,
            run_dir=tmp_path / "run-a",
            collision_policy="fail",
        ),
        run_id="run-a",
        created_at="2026-06-30T00:00:00Z",
        runtime_identity={},
        backend_status={"single": ["active"], "accelerate": [], "deepspeed": []},
    )


def _fail_second_metric_append_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = RunArtifactManager.append_metric_event
    call_count = 0
    remaining_failures = 1

    def fail_second_once(self: RunArtifactManager, event: Any) -> Path:
        nonlocal call_count, remaining_failures
        call_count += 1
        if call_count == 2 and remaining_failures:
            remaining_failures -= 1
            raise ArtifactContractError(
                "synthetic metric append failure",
                code="test.metric_append_failed",
            )
        return original(self, event)

    monkeypatch.setattr(RunArtifactManager, "append_metric_event", fail_second_once)


def _fail_next_manifest_write(monkeypatch: pytest.MonkeyPatch) -> None:
    original = RunArtifactManager._write_manifest
    remaining_failures = 1

    def fail_once(self: RunArtifactManager, payload: dict[str, object]) -> None:
        nonlocal remaining_failures
        if remaining_failures:
            remaining_failures -= 1
            raise ArtifactContractError(
                "synthetic manifest write failure",
                code="test.manifest_write_failed",
            )
        original(self, payload)

    monkeypatch.setattr(RunArtifactManager, "_write_manifest", fail_once)
