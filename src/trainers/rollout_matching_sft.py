"""Deprecated compatibility shim for Stage-2 rollout-aligned trainer.

Canonical module: `src.trainers.stage2_rollout_aligned`.
"""

from .stage2_rollout_aligned import RolloutMatchingSFTTrainer, Stage2RolloutAlignedTrainer

__all__ = ["RolloutMatchingSFTTrainer", "Stage2RolloutAlignedTrainer"]
