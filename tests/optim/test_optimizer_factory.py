from __future__ import annotations

import pytest
import torch

from src.config.models import OptimizerConfig, OptimizerGroupConfig, SchedulerConfig
from src.optim.factory import build_optimizer_and_scheduler
from src.optim.parameter_groups import OptimizerGroupAssignment, OptimizerGroupPlan


def test_optimizer_factory_builds_adamw_groups_and_cosine_scheduler() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    plan = OptimizerGroupPlan(
        groups=(
            OptimizerGroupAssignment(
                group_name="adapter.language",
                lr=2.0e-4,
                weight_decay=0.01,
                parameter_names=("adapter.weight",),
            ),
        ),
        parameters_by_name={"adapter.weight": parameter},
    )
    config = _optimizer_config(warmup_ratio=0.2)

    optimizer, scheduler = build_optimizer_and_scheduler(
        config,
        plan,
        total_training_steps=10,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["name"] == "adapter.language"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(2.0e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert scheduler is not None
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0


def test_optimizer_factory_rejects_empty_group_plan() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_optimizer_and_scheduler(
            _optimizer_config(warmup_steps=0),
            OptimizerGroupPlan(groups=(), parameters_by_name={}),
            total_training_steps=1,
        )


def _optimizer_config(
    *,
    warmup_ratio: float | None = None,
    warmup_steps: int | None = None,
) -> OptimizerConfig:
    return OptimizerConfig(
        name="adamw_torch",
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        kwargs={},
        groups={
            "adapters": {"language": {"lr": 1.0e-4, "weight_decay": 0.0}},
            "token_embeddings": {"lr": 1.0e-4, "weight_decay": 0.0},
        },
        scheduler=SchedulerConfig(
            name="cosine_with_warmup",
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            kwargs={},
        ),
    )
