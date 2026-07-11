from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime.seeding import seed_training_runtime
from src.runtime.train_runtime import TrainRuntime, validate_accelerator_runtime
from src.training import SupervisedMicroStep


def _config(*, seed: int = 17) -> RuntimeConfig:
    return RuntimeConfig(seed=seed)


def _batch(*, world_size: int = 1, accumulation: int = 1) -> RuntimeBatchResolution:
    return RuntimeBatchResolution(
        world_size=world_size,
        effective_batch_size=world_size * accumulation,
        resolved_grad_accum_steps=accumulation,
    )


def _runtime(
    *,
    accelerator: FakeAccelerator | None = None,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    world_size: int = 1,
    accumulation: int = 1,
    gatherer: Any | None = None,
    max_grad_norm: float | None = None,
    expected_mixed_precision: str = "bf16",
) -> TrainRuntime:
    accelerator = accelerator or FakeAccelerator(num_processes=world_size)
    model = model or torch.nn.Linear(1, 1)
    return TrainRuntime(
        runtime_config=_config(),
        runtime_batch=_batch(world_size=world_size, accumulation=accumulation),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_mixed_precision=expected_mixed_precision,
        max_grad_norm=max_grad_norm,
        accelerator=accelerator,
        rank_report_gatherer=gatherer,
    )


def test_seed_training_runtime_delegates_to_transformers_set_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "src.runtime.seeding.set_seed",
        lambda seed, deterministic=False: calls.append((seed, deterministic)),
    )
    receipt = seed_training_runtime(17, phase="pipeline_assembly").to_artifact_dict()
    assert calls == [(17, False)]
    assert receipt["phase"] == "pipeline_assembly"


def test_runtime_identity_comes_from_constructed_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool, str]] = []
    monkeypatch.setattr(
        "src.runtime.train_runtime.seed_training_runtime",
        lambda seed, deterministic=False, phase="": calls.append(
            (seed, deterministic, phase)
        ),
    )
    accelerator = FakeAccelerator(
        process_index=1,
        num_processes=2,
        device="cpu",
        is_main_process=False,
        distributed_type="MULTI_GPU",
    )
    runtime = _runtime(
        accelerator=accelerator,
        world_size=2,
        gatherer=FakeReportGatherer(),
    )
    assert (runtime.rank, runtime.world_size, runtime.device) == (
        1,
        2,
        torch.device("cpu"),
    )
    assert runtime.is_main_process is False
    assert calls == [(17, False, "runtime_setup_reapplied")]


@pytest.mark.parametrize("distributed_type", ["FSDP", "DEEPSPEED", "TP", "MULTI_CPU"])
def test_unsupported_distributed_type_is_rejected_before_prepare(
    distributed_type: str,
) -> None:
    accelerator = FakeAccelerator(distributed_type=distributed_type)
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(accelerator=accelerator)
    assert exc_info.value.code == "runtime.distributed_type_unsupported"
    assert exc_info.value.context["distributed_type"] == distributed_type
    assert accelerator.prepare_calls == 0


def test_accelerator_accumulation_must_be_neutral() -> None:
    accelerator = FakeAccelerator(gradient_accumulation_steps=2)
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(accelerator=accelerator, accumulation=2)
    assert exc_info.value.code == "runtime.accelerator_accumulation_non_neutral"
    assert accelerator.prepare_calls == 0


def test_accelerator_mixed_precision_matches_resolved_training_precision() -> None:
    accelerator = FakeAccelerator(mixed_precision="bf16")
    validate_accelerator_runtime(
        accelerator,
        expected_mixed_precision="BF16",
    )
    runtime = _runtime(accelerator=accelerator, expected_mixed_precision="bf16")
    assert runtime.expected_mixed_precision == "bf16"
    assert accelerator.prepare_calls == 1


@pytest.mark.parametrize("observed_precision", ["fp16", "no", None])
def test_accelerator_mixed_precision_mismatch_is_rejected_before_prepare(
    observed_precision: str | None,
) -> None:
    accelerator = FakeAccelerator(mixed_precision=observed_precision)
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(accelerator=accelerator, expected_mixed_precision="bf16")
    assert exc_info.value.code == "runtime.mixed_precision_mismatch"
    assert exc_info.value.context == {
        "expected_mixed_precision": "bf16",
        "observed_mixed_precision": "no" if observed_precision is None else observed_precision,
    }
    assert accelerator.prepare_calls == 0


def test_prepare_owns_model_and_optimizer_but_not_scheduler() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    accelerator = FakeAccelerator()
    runtime = _runtime(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert accelerator.prepared_objects == (model, optimizer)
    assert runtime.scheduler is scheduler
    assert runtime.setup_receipt.to_artifact_dict()["scheduler_semantics"] == {
        "scheduler_owner": "coordexp_runtime",
        "scheduler_prepared_by_accelerate": False,
        "step_policy": "once_per_planned_step",
    }


def test_runtime_preserves_accumulation_backward_clip_optimizer_scheduler_order() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = RecordingOptimizer(model.parameters())
    scheduler = RecordingScheduler(optimizer)
    accelerator = FakeAccelerator(events=optimizer.events)
    runtime = _runtime(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=2,
        max_grad_norm=0.5,
    )
    scheduler.events = optimizer.events
    with runtime.accumulation_context(sync_gradients=False):
        runtime.backward(runtime.model(torch.tensor([[1.0]])).sum(), planned_step_id=1)
    with runtime.accumulation_context(sync_gradients=True):
        runtime.backward(runtime.model(torch.tensor([[2.0]])).sum(), planned_step_id=1)
    decision = runtime.post_backward(planned_step_id=1)
    runtime.clip_gradients(planned_step_id=1)
    runtime.optimizer_step(planned_step_id=1)
    runtime.scheduler_step(planned_step_id=1)
    runtime.zero_gradients(planned_step_id=1)
    assert decision.should_call_optimizer_step
    assert accelerator.no_sync_models == [runtime.model]
    assert optimizer.events == [
        "backward",
        "backward",
        "clip",
        "optimizer_step",
        "scheduler_step",
        "zero_grad",
    ]


def test_all_rank_scalar_and_gradient_decisions_are_preserved() -> None:
    gatherer = FakeReportGatherer(peer_scalar_total=float("nan"))
    runtime = _runtime(world_size=2, gatherer=gatherer)
    pre = runtime.pre_backward(
        FakeLossBundle(total_loss=torch.tensor(1.0)), planned_step_id=6
    )
    assert not pre.should_call_backward
    assert pre.reason_codes == ("rank1:non_finite_scalar",)

    gatherer.peer_scalar_total = 1.0
    gatherer.peer_backend_overflow = True
    for parameter in runtime.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    post = runtime.post_backward(planned_step_id=7)
    assert not post.should_call_optimizer_step
    assert post.reason_codes == ("rank1:backend_overflow",)


def test_all_rank_loss_denominators_are_preserved() -> None:
    runtime = _runtime(world_size=2, gatherer=DenominatorGatherer())
    gathered = runtime.gather_loss_denominators(
        {"base_ce": {"eligible_segment_count": 1}}, planned_step_id=9
    )
    assert gathered[0]["base_ce"]["eligible_segment_count"] == 1
    assert gathered[1]["base_ce"]["eligible_segment_count"] == 3


def test_multirank_runtime_requires_report_gatherer() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        _runtime(world_size=2)
    assert exc_info.value.code == "runtime.report_gather_unavailable"


def test_move_metrics_and_rank_safe_save(tmp_path: Path) -> None:
    runtime = _runtime()
    moved = runtime.move_micro_step(
        SupervisedMicroStep(
            pack="pack",
            encoded_examples=(),
            position_inputs="positions",
            token_sequence="tokens",
            vocab_groups="vocab",
        ),
        planned_step_id=1,
        local_micro_step_index=0,
    )
    assert moved.forward_device == torch.device("cpu")
    assert runtime.gather_metrics(
        {"loss/total": 1.5}, planned_step_id=1, split="train"
    )["reduction"] == "single_rank"
    path = tmp_path / "payload.json"
    assert runtime.safe_save_json({"rank": 0}, path) == path
    assert json.loads(path.read_text()) == {"rank": 0}


class FakeLossBundle:
    def __init__(self, *, total_loss: torch.Tensor) -> None:
        self.total_loss = total_loss
        self.terms = ()


class FakeReportGatherer:
    def __init__(
        self,
        *,
        peer_scalar_total: float = 1.0,
        peer_backend_overflow: bool = False,
    ) -> None:
        self.peer_scalar_total = peer_scalar_total
        self.peer_backend_overflow = peer_backend_overflow

    def __call__(self, local_report: Any) -> tuple[Any, Any]:
        from src.runtime import RankGradientFiniteReport, RankScalarFiniteReport

        if isinstance(local_report, RankScalarFiniteReport):
            peer = RankScalarFiniteReport.from_loss_bundle(
                FakeLossBundle(total_loss=torch.tensor(self.peer_scalar_total)),
                planned_step_id=local_report.planned_step_id,
                rank=1,
                world_size=2,
            )
            return local_report, peer
        if isinstance(local_report, RankGradientFiniteReport):
            peer = RankGradientFiniteReport(
                planned_step_id=local_report.planned_step_id,
                rank=1,
                world_size=2,
                gradients_finite=True,
                backend_overflow=self.peer_backend_overflow,
                grad_norm=1.0,
            )
            return local_report, peer
        raise AssertionError(type(local_report).__name__)


class DenominatorGatherer:
    def __call__(self, payload: dict[str, Any]) -> tuple[Any, Any]:
        peer = {
            **payload,
            "rank": 1,
            "denominators": {
                name: {**value, "eligible_segment_count": 3}
                for name, value in payload["denominators"].items()
            },
        }
        return payload, peer


class FakeAccelerator:
    def __init__(
        self,
        *,
        process_index: int = 0,
        num_processes: int = 1,
        device: str = "cpu",
        is_main_process: bool = True,
        distributed_type: str = "NO",
        gradient_accumulation_steps: int = 1,
        mixed_precision: str | None = "bf16",
        events: list[str] | None = None,
    ) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.device = torch.device(device)
        self.is_main_process = is_main_process
        self.distributed_type = SimpleNamespace(name=distributed_type)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.events = events if events is not None else []
        self.prepare_calls = 0
        self.prepared_objects: tuple[Any, ...] = ()
        self.no_sync_models: list[Any] = []
        self.no_sync_depth = 0
        self.scaler = None

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        self.prepare_calls += 1
        self.prepared_objects = tuple(objects)
        return self.prepared_objects

    @contextmanager
    def no_sync(self, model: Any) -> Any:
        self.no_sync_models.append(model)
        self.no_sync_depth += 1
        try:
            yield
        finally:
            self.no_sync_depth -= 1

    def backward(self, loss: torch.Tensor) -> None:
        self.events.append("backward")
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
        self.events.append("clip")
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class RecordingOptimizer(torch.optim.SGD):
    def __init__(self, parameters: Any) -> None:
        super().__init__(parameters, lr=0.1)
        self.events: list[str] = []

    def step(self, closure: Any | None = None) -> Any:
        self.events.append("optimizer_step")
        return super().step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.events.append("zero_grad")
        super().zero_grad(set_to_none=set_to_none)


class RecordingScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.events: list[str] = []
        self.last_epoch = 0

    def step(self) -> None:
        self.events.append("scheduler_step")
        self.last_epoch += 1

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]
