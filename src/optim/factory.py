"""Optimizer and scheduler construction."""

from __future__ import annotations

import math

import torch
from transformers import get_cosine_schedule_with_warmup

from src.config.models import OptimizerConfig
from src.optim.parameter_groups import OptimizerGroupPlan


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
    if config.scheduler.name != "cosine_with_warmup":
        raise ValueError(f"unsupported scheduler: {config.scheduler.name}")

    optimizer = torch.optim.AdamW(
        group_plan.to_torch_param_groups(),
        betas=config.betas,
        eps=config.epsilon,
        **config.kwargs,
    )
    warmup_steps = _warmup_steps(config, total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        **config.scheduler.kwargs,
    )
    return optimizer, scheduler


def _warmup_steps(config: OptimizerConfig, total_training_steps: int) -> int:
    if config.scheduler.warmup_steps is not None:
        return config.scheduler.warmup_steps
    assert config.scheduler.warmup_ratio is not None
    return int(math.ceil(config.scheduler.warmup_ratio * total_training_steps))
