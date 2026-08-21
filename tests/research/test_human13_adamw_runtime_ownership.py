from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import get_cosine_schedule_with_warmup

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer

from scripts.research.human13_live_model import (
    Human13LiveModelError,
    build_human13_adamw_runtime_ownership,
)
from src.config.models import RuntimeBatchResolution, RuntimeConfig
from src.runtime import TrainRuntime


def _real_cpu_runtime(
    *,
    dtype: torch.dtype = torch.bfloat16,
    optimizer_type: type[torch.optim.AdamW] = torch.optim.AdamW,
) -> tuple[TrainRuntime, torch.optim.AdamW, torch.nn.Module]:
    model = torch.nn.Linear(3, 2).to(dtype=dtype)
    base_optimizer = optimizer_type(
        model.parameters(),
        lr=3.0e-6,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    base_optimizer.param_groups[0]["name"] = "adapter.language"
    scheduler = get_cosine_schedule_with_warmup(
        base_optimizer,
        num_warmup_steps=0,
        num_training_steps=1,
    )
    accelerator = Accelerator(cpu=True, mixed_precision="bf16")
    runtime = TrainRuntime(
        runtime_config=RuntimeConfig(seed=17),
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
        model=model,
        optimizer=base_optimizer,
        scheduler=scheduler,
        expected_mixed_precision="bf16",
        max_grad_norm=1.0,
        accelerator=accelerator,
        rank_report_gatherer=None,
    )
    return runtime, base_optimizer, model


def _runtime_copy(runtime: TrainRuntime, **overrides: object) -> SimpleNamespace:
    values = {
        "model": runtime.model,
        "optimizer": runtime.optimizer,
        "scheduler": runtime.scheduler,
        "accelerator": runtime.accelerator,
        "world_size": runtime.world_size,
        "optimizer_step_count": runtime.optimizer_step_count,
        "scheduler_step_count": runtime.scheduler_step_count,
        "zero_grad_count": runtime.zero_grad_count,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_real_accelerate_cpu_vertical_admits_one_layer_fresh_adamw() -> None:
    runtime, base_optimizer, model = _real_cpu_runtime()
    context = build_human13_adamw_runtime_ownership(
        runtime,
        tuple(model.named_parameters()),
        expected_learning_rate=3.0e-6,
        expected_betas=(0.9, 0.999),
        expected_epsilon=1.0e-8,
        expected_weight_decay=0.0,
        capture_cuda=True,
    )

    assert type(runtime.optimizer) is AcceleratedOptimizer
    assert type(context.base_optimizer) is torch.optim.AdamW
    assert context.base_optimizer is base_optimizer
    assert context.execution_optimizer is runtime.optimizer
    assert context.scheduler_optimizer is getattr(runtime.scheduler, "optimizer", None)
    assert context.parameter_object_ids == tuple(
        id(parameter) for _, parameter in model.named_parameters()
    )
    assert context.runtime_optimizer_step_count == 0
    assert context.scheduler_step_count == 0
    assert context.capture_cuda is True
    assert len(context.content_sha256) == 64


def test_transaction_binds_base_while_runtime_keeps_wrapper_execution_handle() -> None:
    runtime, _, model = _real_cpu_runtime()
    context = build_human13_adamw_runtime_ownership(
        runtime,
        tuple(model.named_parameters()),
        expected_learning_rate=3.0e-6,
        expected_betas=(0.9, 0.999),
        expected_epsilon=1.0e-8,
        expected_weight_decay=0.0,
        capture_cuda=True,
    )
    from scripts.research.human13_training_transaction import (
        TrainingStateTransaction,
        UpdateCounter,
    )

    transaction = TrainingStateTransaction(
        tuple(model.named_parameters()),
        optimizer=context.base_optimizer,
        scheduler=context.scheduler,
        update_counter=UpdateCounter(),
        runtime=runtime,
        capture_cuda=True,
    )
    assert transaction._optimizer is context.base_optimizer
    assert context.execution_optimizer is runtime.optimizer
    assert context.scheduler_optimizer is context.base_optimizer


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda runtime, base, model: _runtime_copy(
                runtime,
                optimizer=SimpleNamespace(optimizer=runtime.optimizer),
            ),
            "AcceleratedOptimizer",
        ),
        (
            lambda runtime, base, model: _runtime_copy(
                runtime,
                scheduler=torch.optim.lr_scheduler.LambdaLR(
                    torch.optim.AdamW(model.parameters(), lr=3.0e-6),
                    lambda _: 1.0,
                ),
            ),
            "scheduler",
        ),
        (
            lambda runtime, base, model: _runtime_copy(
                runtime,
                optimizer_step_count=1,
            ),
            "counter",
        ),
        (
            lambda runtime, base, model: _runtime_copy(runtime),
            "fresh AdamW",
        ),
    ],
)
def test_runtime_ownership_rejects_deterministic_pre_acquisition_drift(
    mutation: Callable[[Any, torch.optim.AdamW, torch.nn.Module], Any],
    match: str,
) -> None:
    runtime, base_optimizer, model = _real_cpu_runtime()
    mutated = mutation(runtime, base_optimizer, model)
    if match == "fresh AdamW":
        base_optimizer.param_groups[0]["lr"] = 1.0e-5
    with pytest.raises(Human13LiveModelError, match=match):
        build_human13_adamw_runtime_ownership(
            mutated,
            tuple(model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )


def test_runtime_ownership_rejects_populated_state_param_order_and_dtype() -> None:
    runtime, base_optimizer, model = _real_cpu_runtime()
    first_parameter = next(iter(base_optimizer.state)) if base_optimizer.state else next(
        iter(base_optimizer.param_groups[0]["params"])
    )
    base_optimizer.state[first_parameter]["step"] = torch.tensor(1.0)
    with pytest.raises(Human13LiveModelError, match="state"):
        build_human13_adamw_runtime_ownership(
            runtime,
            tuple(model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )


def test_runtime_ownership_rejects_parameter_order_and_adamw_subclass() -> None:
    runtime, base_optimizer, model = _real_cpu_runtime()
    with pytest.raises(Human13LiveModelError, match="order or identity"):
        build_human13_adamw_runtime_ownership(
            runtime,
            tuple(reversed(tuple(model.named_parameters()))),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )

    class AdamWSubclass(torch.optim.AdamW):
        pass

    subclass_runtime, _, subclass_model = _real_cpu_runtime(
        optimizer_type=AdamWSubclass,
    )
    with pytest.raises(Human13LiveModelError, match="exact fresh AdamW"):
        build_human13_adamw_runtime_ownership(
            subclass_runtime,
            tuple(subclass_model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )


def test_runtime_ownership_rejects_foreign_scheduler_and_group_hparam_drift() -> None:
    runtime, base_optimizer, model = _real_cpu_runtime()
    foreign_scheduler = torch.optim.lr_scheduler.StepLR(
        base_optimizer,
        step_size=1,
    )
    foreign_runtime = _runtime_copy(runtime, scheduler=foreign_scheduler)
    with pytest.raises(Human13LiveModelError, match="scheduler type"):
        build_human13_adamw_runtime_ownership(
            foreign_runtime,
            tuple(model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )

    runtime, base_optimizer, model = _real_cpu_runtime()
    constant_scheduler = torch.optim.lr_scheduler.LambdaLR(
        base_optimizer,
        lambda _step: 1.0,
    )
    constant_runtime = _runtime_copy(runtime, scheduler=constant_scheduler)
    with pytest.raises(Human13LiveModelError, match="cosine semantics"):
        build_human13_adamw_runtime_ownership(
            constant_runtime,
            tuple(model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )

    runtime, base_optimizer, model = _real_cpu_runtime()
    base_optimizer.param_groups[0]["betas"] = (0.8, 0.9)
    with pytest.raises(Human13LiveModelError, match="group hyperparameters"):
        build_human13_adamw_runtime_ownership(
            runtime,
            tuple(model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )

    runtime, base_optimizer, model = _real_cpu_runtime()
    base_optimizer.param_groups[0]["eps"] = 1.0e-6
    with pytest.raises(Human13LiveModelError, match="group hyperparameters"):
        build_human13_adamw_runtime_ownership(
            runtime,
            tuple(model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )


def test_runtime_ownership_rejects_sync_scaler_device_and_dtype_drift() -> None:
    runtime, _, model = _real_cpu_runtime(dtype=torch.float32)
    with pytest.raises(Human13LiveModelError, match="bf16 device surface"):
        build_human13_adamw_runtime_ownership(
            runtime,
            tuple(model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )

    bf16_runtime, _, bf16_model = _real_cpu_runtime()
    drifted_accelerator = SimpleNamespace(
        mixed_precision="bf16",
        distributed_type=SimpleNamespace(name="MULTI_GPU"),
        scaler=None,
        gradient_state=SimpleNamespace(sync_gradients=True),
        device="cuda:0",
    )
    drifted = _runtime_copy(
        bf16_runtime,
        model=bf16_model,
        accelerator=drifted_accelerator,
    )
    with pytest.raises(Human13LiveModelError, match="sync-neutral"):
        build_human13_adamw_runtime_ownership(
            drifted,
            tuple(bf16_model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )

    drifted_accelerator.distributed_type = SimpleNamespace(name="NO")
    drifted_accelerator.scaler = object()
    with pytest.raises(Human13LiveModelError, match="gradient scaler"):
        build_human13_adamw_runtime_ownership(
            drifted,
            tuple(bf16_model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )

    drifted_accelerator.scaler = None
    with pytest.raises(Human13LiveModelError, match="device"):
        build_human13_adamw_runtime_ownership(
            drifted,
            tuple(bf16_model.named_parameters()),
            expected_learning_rate=3.0e-6,
            expected_betas=(0.9, 0.999),
            expected_epsilon=1.0e-8,
            expected_weight_decay=0.0,
            capture_cuda=True,
        )
