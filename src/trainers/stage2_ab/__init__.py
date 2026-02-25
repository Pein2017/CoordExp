"""Deprecated compatibility package for Stage-2 two-channel helpers.

Canonical package: `src.trainers.stage2_two_channel`.
"""

from src.trainers.stage2_two_channel.executors import Stage2ABChannelExecutorsMixin
from src.trainers.stage2_two_channel.scheduler import Stage2ABSchedulerMixin

__all__ = ["Stage2ABChannelExecutorsMixin", "Stage2ABSchedulerMixin"]
