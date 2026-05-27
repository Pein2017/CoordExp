from __future__ import annotations

import inspect

from src.infer.prompt import (
    prepare_rollout_prompt_samples_from_facts,
    prepare_rollout_prompt_samples_from_owner,
)
from src.trainers.stage2_rollout_correction import Stage2RolloutCorrectionTrainer


def test_stage2_rollout_correction_enforces_prompt_tokenization_alignment() -> None:
    src = inspect.getsource(Stage2RolloutCorrectionTrainer._prepare_rollout_correction_inputs_impl)
    assert "require_verified_prompt_token_parity" in src
    assert "backend_prompt_token_ids=prompt_ids" in src


def test_rollout_matching_rebuilds_prompts_with_active_object_ordering() -> None:
    owner_src = inspect.getsource(prepare_rollout_prompt_samples_from_owner)
    facts_src = inspect.getsource(prepare_rollout_prompt_samples_from_facts)
    assert "prepare_rollout_prompt_samples_from_owner" in (
        prepare_rollout_prompt_samples_from_owner.__name__
    )
    assert "RolloutPromptPolicyFacts(" in owner_src
    assert "object_ordering=str(owner._object_ordering())" in owner_src
    assert "object_ordering=facts.object_ordering" in facts_src
