"""Stage-2 AB training components.

This package decomposes Stage-2 AB internals by concern:
- scheduler: deterministic channel selection and realized-ratio tracking
- async_queue: async actor/learner ready-pack queue + sync gating
- executors: per-channel execution helpers

The public trainer entrypoint remains `src.trainers.stage2_ab_training.Stage2ABTrainingTrainer`.
"""

from .async_queue import Stage2ABAsyncQueueManagerMixin, Stage2AsyncReadyPack
from .executors import Stage2ABChannelExecutorsMixin
from .scheduler import Stage2ABSchedulerMixin

__all__ = [
    "Stage2ABAsyncQueueManagerMixin",
    "Stage2AsyncReadyPack",
    "Stage2ABChannelExecutorsMixin",
    "Stage2ABSchedulerMixin",
]
