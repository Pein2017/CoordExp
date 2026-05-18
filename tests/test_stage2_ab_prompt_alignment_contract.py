from __future__ import annotations

import inspect

from src.trainers.stage2_rollout_runtime import Stage2RolloutRuntime
from src.trainers.stage2_two_channel import Stage2TwoChannelTrainer


def test_stage2_ab_enforces_prompt_tokenization_alignment() -> None:
    src = inspect.getsource(Stage2TwoChannelTrainer._prepare_batch_inputs_b_impl)
    assert "prompt tokenization mismatch" in src


def test_rollout_matching_rebuilds_prompts_with_active_object_ordering() -> None:
    src = inspect.getsource(Stage2RolloutRuntime._prepare_samples_for_rollout)
    assert "ordering = self._object_ordering()" in src
