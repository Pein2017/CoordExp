from __future__ import annotations

import inspect

from src.trainers.stage2_rollout_runtime import Stage2RolloutRuntime
from src.trainers.stage2_rollout_correction import Stage2RolloutCorrectionTrainer


def test_stage2_rollout_correction_enforces_prompt_tokenization_alignment() -> None:
    src = inspect.getsource(Stage2RolloutCorrectionTrainer._prepare_rollout_correction_inputs_impl)
    assert "prompt tokenization mismatch" in src


def test_rollout_matching_rebuilds_prompts_with_active_object_ordering() -> None:
    src = inspect.getsource(Stage2RolloutRuntime._prepare_samples_for_rollout)
    assert "ordering = self._object_ordering()" in src
