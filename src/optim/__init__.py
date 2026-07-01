"""Optimizer setup contracts."""

from src.optim.factory import build_optimizer_and_scheduler
from src.optim.parameter_groups import (
    OptimizerGroupAssignment,
    OptimizerGroupPlan,
    build_optimizer_group_plan,
)
from src.optim.trainable_surface import (
    FrozenReasonSummary,
    TrainableSurfaceReceipt,
    build_trainable_surface_receipt,
    write_trainable_surface_receipt,
)

__all__ = [
    "FrozenReasonSummary",
    "OptimizerGroupAssignment",
    "OptimizerGroupPlan",
    "TrainableSurfaceReceipt",
    "build_optimizer_and_scheduler",
    "build_optimizer_group_plan",
    "build_trainable_surface_receipt",
    "write_trainable_surface_receipt",
]
