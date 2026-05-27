from __future__ import annotations

import inspect

from src.infer.prompt import prepare_rollout_prompt_samples_from_owner
from src.trainers.stage2_rollout_correction import Stage2RolloutCorrectionTrainer


def test_stage2_rollout_correction_enforces_prompt_tokenization_alignment() -> None:
    src = inspect.getsource(Stage2RolloutCorrectionTrainer._prepare_rollout_correction_inputs_impl)
    assert "require_verified_prompt_token_parity" in src
    assert "backend_prompt_token_ids=prompt_ids" in src


def test_rollout_matching_rebuilds_prompts_with_active_object_ordering() -> None:
    src = inspect.getsource(prepare_rollout_prompt_samples_from_owner)
    assert "object_ordering=owner._object_ordering()" in src
