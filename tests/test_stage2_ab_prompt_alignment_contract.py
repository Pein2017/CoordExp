from __future__ import annotations

import inspect

from src.trainers.stage2_rollout_aligned import RolloutMatchingSFTTrainer
from src.trainers.stage2_two_channel import Stage2ABTrainingTrainer


def test_stage2_ab_enforces_prompt_tokenization_alignment() -> None:
    src = inspect.getsource(Stage2ABTrainingTrainer._prepare_batch_inputs_b)
    assert "prompt tokenization mismatch" in src


def test_rollout_matching_rebuilds_prompts_with_active_object_ordering() -> None:
    src = inspect.getsource(RolloutMatchingSFTTrainer._prepare_samples_for_rollout)
    assert "ordering = self._object_ordering()" in src
