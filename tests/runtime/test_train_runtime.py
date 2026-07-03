from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from src.common.errors import RuntimeContractError
from src.config.models import (
    AccelerateConfig,
    DeepSpeedConfig,
    RuntimeBatchResolution,
    RuntimeConfig,
)
from src.runtime.train_runtime import TrainRuntime
from src.runtime.seeding import seed_training_runtime
from src.training import SupervisedMicroStep


def test_seed_training_runtime_delegates_to_transformers_set_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "src.runtime.seeding.set_seed",
        lambda seed, deterministic=False: calls.append((seed, deterministic)),
    )

    receipt = seed_training_runtime(
        17,
        deterministic=False,
        phase="pipeline_assembly",
    ).to_artifact_dict()

    assert calls == [(17, False)]
    assert receipt["seed"] == 17
    assert receipt["phase"] == "pipeline_assembly"
    assert receipt["helper"] == "transformers.trainer_utils.set_seed"
    assert receipt["deterministic_algorithms"] is False
    assert receipt["applied_before"] == [
        "qwen_model_load",
        "adapter_setup",
        "special_token_embedding_setup",
        "optimizer_setup",
        "runtime_setup",
    ]


def test_train_runtime_reapplies_configured_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool, str]] = []
    monkeypatch.setattr(
        "src.runtime.train_runtime.seed_training_runtime",
        lambda seed, deterministic=False, phase="pipeline_assembly": calls.append(
            (seed, deterministic, phase)
        ),
    )

    TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=23),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(2, 1),
        optimizer=None,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=1,
    )

    assert calls == [(23, False, "runtime_setup_reapplied")]


def test_train_runtime_single_backend_receipt_and_device_movement() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=2,
            resolved_grad_accum_steps=2,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        rank=0,
        world_size=1,
        max_grad_norm=0.5,
    )

    receipt = runtime.setup_receipt.to_artifact_dict()

    assert receipt["backend"] == "single"
    assert receipt["rank"] == 0
    assert receipt["world_size"] == 1
    assert receipt["device"] == "cpu"
    assert receipt["backend_status"] == {
        "single": ["active"],
        "accelerate": [],
        "deepspeed": [],
    }
    assert receipt["runtime_batch"]["resolved_grad_accum_steps"] == 2
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


def test_train_runtime_scalar_and_gradient_boundaries_drive_optimizer_step() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        rank=0,
        world_size=1,
        max_grad_norm=0.1,
    )
    inputs = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[0.25]])
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    bundle = FakeLossBundle(total_loss=loss)

    pre = runtime.pre_backward(bundle, planned_step_id=3)
    runtime.backward(loss, planned_step_id=3)
    post = runtime.post_backward(planned_step_id=3)
    runtime.clip_gradients(planned_step_id=3)
    runtime.optimizer_step(planned_step_id=3)
    runtime.scheduler_step(planned_step_id=3)
    runtime.zero_gradients(planned_step_id=3)

    assert pre.should_call_backward is True
    assert post.should_call_optimizer_step is True
    assert runtime.optimizer_step_count == 1
    assert runtime.scheduler_step_count == 1
    assert runtime.zero_grad_count == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_runtime_unsafe_scalar_skips_backward_and_clears_gradients() -> None:
    model = torch.nn.Linear(2, 1)
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=1,
    )
    bundle = FakeLossBundle(total_loss=torch.tensor(float("nan")))

    decision = runtime.pre_backward(bundle, planned_step_id=4)

    assert not decision.should_call_backward
    assert decision.should_clear_gradients
    assert decision.optimizer_update_status == "skipped_non_finite_scalar"


def test_train_runtime_rank_safe_save_only_writes_on_rank_zero(tmp_path: Path) -> None:
    gatherer = FakeReportGatherer()
    runtime_rank0 = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=2,
            effective_batch_size=2,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=2,
        rank_report_gatherer=gatherer,
    )
    runtime_rank1 = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=2,
            effective_batch_size=2,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        device="cpu",
        rank=1,
        world_size=2,
        rank_report_gatherer=gatherer,
    )
    output_path = tmp_path / "payload.json"

    assert runtime_rank1.safe_save_json({"rank": 1}, output_path) is None
    assert not output_path.exists()
    assert runtime_rank0.safe_save_json({"rank": 0}, output_path) == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"rank": 0}


def test_train_runtime_multirank_scalar_gate_uses_report_gatherer() -> None:
    model = torch.nn.Linear(1, 1)
    gatherer = FakeReportGatherer()
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=2,
            effective_batch_size=2,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=None,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=2,
        rank_report_gatherer=gatherer,
    )
    gatherer.peer_scalar_total = torch.tensor(float("nan"))

    decision = runtime.pre_backward(
        FakeLossBundle(total_loss=torch.tensor(1.0)),
        planned_step_id=6,
    )

    assert not decision.should_call_backward
    assert decision.optimizer_update_status == "skipped_non_finite_scalar"
    assert decision.ranks == (0, 1)
    assert decision.reason_codes == ("rank1:non_finite_scalar",)


def test_train_runtime_multirank_post_backward_uses_report_gatherer() -> None:
    model = torch.nn.Linear(1, 1)
    gatherer = FakeReportGatherer(peer_backend_overflow=True)
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=2,
            effective_batch_size=2,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=None,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=2,
        rank_report_gatherer=gatherer,
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    decision = runtime.post_backward(planned_step_id=7)

    assert not decision.should_call_optimizer_step
    assert decision.optimizer_update_status == "skipped_gradient_or_overflow"
    assert decision.ranks == (0, 1)
    assert decision.reason_codes == ("rank1:backend_overflow",)


def test_train_runtime_multirank_requires_report_gatherer() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        TrainRuntime(
            runtime_config=RuntimeConfig(backend="single", seed=17),
            runtime_batch=RuntimeBatchResolution(
                world_size=2,
                effective_batch_size=2,
                resolved_grad_accum_steps=1,
            ),
            model=torch.nn.Linear(1, 1),
            optimizer=None,
            scheduler=None,
            device="cpu",
            rank=0,
            world_size=2,
        )

    assert exc_info.value.code == "runtime.report_gather_unavailable"


def test_train_runtime_gathers_metrics_with_rank_context() -> None:
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=1,
    )

    payload = runtime.gather_metrics(
        {"loss/total": 1.5, "acc_top1": 0.25},
        planned_step_id=5,
        split="train",
    )

    assert payload == {
        "planned_step_id": 5,
        "split": "train",
        "rank": 0,
        "world_size": 1,
        "metrics": {"acc_top1": 0.25, "loss/total": 1.5},
        "reduction": "single_rank",
    }


def test_train_runtime_multirank_metrics_are_explicitly_rank_local() -> None:
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(backend="single", seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=2,
            effective_batch_size=2,
            resolved_grad_accum_steps=1,
        ),
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        device="cpu",
        rank=1,
        world_size=2,
        rank_report_gatherer=FakeReportGatherer(),
    )

    payload = runtime.gather_metrics(
        {"loss/total": 2.5},
        planned_step_id=8,
        split="train",
    )

    assert payload["reduction"] == "rank_local"
    assert payload["rank"] == 1
    assert payload["world_size"] == 2


def test_train_runtime_accelerate_backend_requires_accelerator() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        TrainRuntime(
            runtime_config=RuntimeConfig(
                backend="accelerate",
                seed=17,
                accelerate=AccelerateConfig(gradient_accumulation_steps=1),
            ),
            runtime_batch=RuntimeBatchResolution(
                world_size=1,
                effective_batch_size=1,
                resolved_grad_accum_steps=1,
            ),
            model=torch.nn.Linear(1, 1),
            optimizer=None,
            scheduler=None,
            device="cpu",
            rank=0,
            world_size=1,
        )

    assert exc_info.value.code == "runtime.accelerator_required"


def test_train_runtime_deepspeed_backend_requires_accelerator() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        TrainRuntime(
            runtime_config=RuntimeConfig(
                backend="deepspeed",
                seed=17,
                deepspeed=DeepSpeedConfig(
                    config_path="ds.json",
                    gradient_accumulation_steps=1,
                    train_batch_size=1,
                ),
            ),
            runtime_batch=RuntimeBatchResolution(
                world_size=1,
                effective_batch_size=1,
                resolved_grad_accum_steps=1,
            ),
            model=torch.nn.Linear(1, 1),
            optimizer=None,
            scheduler=None,
            device="cpu",
            rank=0,
            world_size=1,
        )

    assert exc_info.value.code == "runtime.deepspeed_accelerator_required"


def test_train_runtime_accelerate_backend_prepares_owned_objects() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    accelerator = FakeAccelerator()

    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(
            backend="accelerate",
            seed=17,
            accelerate=AccelerateConfig(gradient_accumulation_steps=1),
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        rank=0,
        world_size=1,
        accelerator=accelerator,
    )

    assert accelerator.prepare_calls == 1
    assert runtime.model is accelerator.prepared_objects[0]
    assert runtime.optimizer is accelerator.prepared_objects[1]
    assert runtime.scheduler is accelerator.prepared_objects[2]
    assert runtime.setup_receipt.backend_status["accelerate"] == (
        "schema_accepted",
        "prepared",
        "active",
    )


def test_train_runtime_accelerate_backward_uses_no_sync_when_requested() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = FakeAccelerator()

    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(
            backend="accelerate",
            seed=17,
            accelerate=AccelerateConfig(gradient_accumulation_steps=None),
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=3,
            resolved_grad_accum_steps=3,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=1,
        accelerator=accelerator,
    )

    with runtime.accumulation_context(sync_gradients=False):
        first_loss = runtime.model(torch.tensor([[1.0]])).sum()
        runtime.backward(first_loss, planned_step_id=1, sync_gradients=False)
    with runtime.accumulation_context(sync_gradients=True):
        second_loss = runtime.model(torch.tensor([[2.0]])).sum()
        runtime.backward(second_loss, planned_step_id=1, sync_gradients=True)

    assert accelerator.no_sync_models == [runtime.model]
    assert accelerator.backward_sync_states == [False, True]


def test_train_runtime_deepspeed_backend_prepares_owned_objects() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    accelerator = FakeAccelerator()

    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(
            backend="deepspeed",
            seed=17,
            deepspeed=DeepSpeedConfig(
                config_path="ds.json",
                gradient_accumulation_steps=1,
                train_batch_size=1,
            ),
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        rank=0,
        world_size=1,
        accelerator=accelerator,
    )

    assert accelerator.prepare_calls == 1
    assert runtime.model is accelerator.prepared_objects[0]
    assert runtime.optimizer is accelerator.prepared_objects[1]
    assert runtime.scheduler is accelerator.prepared_objects[2]
    assert runtime.setup_receipt.backend_status["deepspeed"] == (
        "schema_accepted",
        "conflict_validation_implemented",
        "prepared",
        "active",
    )


def test_train_runtime_deepspeed_uses_accelerator_backward() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = FakeAccelerator()
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(
            backend="deepspeed",
            seed=17,
            deepspeed=DeepSpeedConfig(
                config_path="ds.json",
                gradient_accumulation_steps=1,
                train_batch_size=1,
            ),
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=1,
        accelerator=accelerator,
    )

    runtime.backward(runtime.model(torch.tensor([[1.0]])).sum(), planned_step_id=1)

    assert next(runtime.model.parameters()).grad is not None
    assert accelerator.backward_kwargs == [{"scale_wrt_gas": False}]


def test_train_runtime_deepspeed_backward_sets_sync_boundary() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = FakeDeepSpeedAccelerator(global_grad_norm=2.5)
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(
            backend="deepspeed",
            seed=17,
            deepspeed=DeepSpeedConfig(
                config_path="ds.json",
                gradient_accumulation_steps=2,
                train_batch_size=2,
            ),
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=2,
            resolved_grad_accum_steps=2,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=1,
        accelerator=accelerator,
    )

    runtime.backward(
        runtime.model(torch.tensor([[1.0]])).sum(),
        planned_step_id=1,
        sync_gradients=False,
    )
    runtime.backward(
        runtime.model(torch.tensor([[2.0]])).sum(),
        planned_step_id=1,
        sync_gradients=True,
    )

    assert accelerator.backward_sync_states == [False, True]
    assert accelerator.backward_kwargs == [
        {"scale_wrt_gas": False},
        {"scale_wrt_gas": False},
    ]
    assert accelerator.sync_gradients is True


def test_train_runtime_deepspeed_post_backward_uses_engine_global_grad_norm() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = FakeDeepSpeedAccelerator(global_grad_norm=2.5)
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(
            backend="deepspeed",
            seed=17,
            deepspeed=DeepSpeedConfig(
                config_path="ds.json",
                gradient_accumulation_steps=1,
                train_batch_size=1,
            ),
        ),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        rank=0,
        world_size=1,
        accelerator=accelerator,
    )

    loss = runtime.model(torch.tensor([[1.0]])).sum()
    runtime.backward(loss, planned_step_id=1)
    decision = runtime.post_backward(planned_step_id=1)

    assert next(runtime.model.parameters()).grad is None
    assert decision.should_call_optimizer_step is True
    assert decision.finite_status == "finite"
    assert decision.rank_diagnostics[0]["grad_norm"] == 2.5


def test_train_runtime_rejects_backend_batch_conflict() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        TrainRuntime(
            runtime_config=RuntimeConfig(
                backend="accelerate",
                seed=17,
                accelerate=AccelerateConfig(gradient_accumulation_steps=2),
            ),
            runtime_batch=RuntimeBatchResolution(
                world_size=1,
                effective_batch_size=1,
                resolved_grad_accum_steps=1,
            ),
            model=torch.nn.Linear(1, 1),
            optimizer=None,
            scheduler=None,
            device="cpu",
            rank=0,
            world_size=1,
        )

    assert exc_info.value.code == "runtime.accumulation_conflict"


class FakeLossBundle:
    def __init__(self, *, total_loss: torch.Tensor) -> None:
        self.total_loss = total_loss
        self.terms = ()


class FakeReportGatherer:
    def __init__(self, *, peer_backend_overflow: bool = False) -> None:
        self.peer_scalar_total = torch.tensor(1.0)
        self.peer_backend_overflow = peer_backend_overflow

    def __call__(self, local_report: Any) -> tuple[Any, Any]:
        from src.runtime import RankGradientFiniteReport, RankScalarFiniteReport

        if isinstance(local_report, RankScalarFiniteReport):
            peer = RankScalarFiniteReport.from_loss_bundle(
                FakeLossBundle(total_loss=self.peer_scalar_total),
                planned_step_id=local_report.planned_step_id,
                rank=1,
                world_size=2,
            )
            return (local_report, peer)
        if isinstance(local_report, RankGradientFiniteReport):
            peer = RankGradientFiniteReport(
                planned_step_id=local_report.planned_step_id,
                rank=1,
                world_size=2,
                gradients_finite=True,
                backend_overflow=self.peer_backend_overflow,
                grad_norm=1.0,
            )
            return (local_report, peer)
        raise AssertionError(f"unexpected report type: {type(local_report).__name__}")


class FakeAccelerator:
    is_main_process = True

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.prepared_objects: tuple[Any, ...] = ()
        self.no_sync_models: list[Any] = []
        self.no_sync_depth = 0
        self.backward_sync_states: list[bool] = []
        self.backward_kwargs: list[dict[str, Any]] = []
        self.sync_gradients = True

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

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:
        self.backward_sync_states.append(self.no_sync_depth == 0)
        self.backward_kwargs.append(dict(kwargs))
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class FakeDeepSpeedEngineWrapper:
    def __init__(self, *, global_grad_norm: float) -> None:
        self.global_grad_norm = global_grad_norm

    def get_global_grad_norm(self) -> float:
        return self.global_grad_norm


class FakeDeepSpeedAccelerator(FakeAccelerator):
    def __init__(self, *, global_grad_norm: float) -> None:
        super().__init__()
        self.deepspeed_engine_wrapped = FakeDeepSpeedEngineWrapper(
            global_grad_norm=global_grad_norm,
        )

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:
        self.backward_sync_states.append(bool(self.sync_gradients))
        self.backward_kwargs.append(dict(kwargs))
        loss.backward()
        for parameter in self.prepared_objects[0].parameters():
            parameter.grad = None
