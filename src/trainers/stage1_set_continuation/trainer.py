from __future__ import annotations

from swift.trainers import TrainerFactory


class Stage1SetContinuationTrainer(TrainerFactory.get_cls(
    type("Args", (), {"task_type": "causal_lm"})(),
    TrainerFactory.TRAINER_MAPPING,
)):
    """Dedicated routing stub for the Stage-1 set-continuation variant."""


__all__ = ["Stage1SetContinuationTrainer"]
