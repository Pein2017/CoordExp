"""Optimizer and scheduler construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from transformers import get_cosine_schedule_with_warmup

from src.config.models import OptimizerConfig
from src.optim.parameter_groups import OptimizerGroupPlan


@dataclass(frozen=True)
class SchedulerPlan:
    name: str
    total_training_steps: int
    warmup_ratio: float | None
    warmup_steps: int | None
    resolved_warmup_steps: int
    kwargs: dict[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_training_steps": self.total_training_steps,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "resolved_warmup_steps": self.resolved_warmup_steps,
            "kwargs": dict(self.kwargs),
        }


def build_optimizer_and_scheduler(
    config: OptimizerConfig,
    group_plan: OptimizerGroupPlan,
    *,
    total_training_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    if not group_plan.groups:
        raise ValueError("optimizer group plan must contain at least one group")
    if total_training_steps <= 0:
        raise ValueError("total_training_steps must be positive")
    if config.name != "adamw_torch":
        raise ValueError(f"unsupported optimizer: {config.name}")
    scheduler_plan = build_scheduler_plan(
        config,
        total_training_steps=total_training_steps,
    )

    optimizer = torch.optim.AdamW(
        group_plan.to_torch_param_groups(),
        betas=config.betas,
        eps=config.epsilon,
        **config.kwargs,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_plan.resolved_warmup_steps,
        num_training_steps=total_training_steps,
        **config.scheduler.kwargs,
    )
    return optimizer, scheduler


def build_scheduler_plan(
    config: OptimizerConfig,
    *,
    total_training_steps: int,
) -> SchedulerPlan:
    if total_training_steps <= 0:
        raise ValueError("total_training_steps must be positive")
    if config.scheduler.name != "cosine_with_warmup":
        raise ValueError(f"unsupported scheduler: {config.scheduler.name}")
    return SchedulerPlan(
        name=config.scheduler.name,
        total_training_steps=total_training_steps,
        warmup_ratio=config.scheduler.warmup_ratio,
        warmup_steps=config.scheduler.warmup_steps,
        resolved_warmup_steps=_warmup_steps(config, total_training_steps),
        kwargs=dict(config.scheduler.kwargs),
    )


def _warmup_steps(config: OptimizerConfig, total_training_steps: int) -> int:
    if config.scheduler.warmup_steps is not None:
        return config.scheduler.warmup_steps
    assert config.scheduler.warmup_ratio is not None
    return int(math.ceil(config.scheduler.warmup_ratio * total_training_steps))
