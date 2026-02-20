import types

import pytest

from src.trainers.rollout_matching_sft import RolloutMatchingSFTTrainer


def test_hf_length_gate_raises_when_prompt_plus_generation_exceeds_context() -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.model = types.SimpleNamespace(
        config=types.SimpleNamespace(max_position_embeddings=100)
    )

    with pytest.raises(ValueError, match=r"max_position_embeddings"):
        t._enforce_hf_rollout_max_position_embeddings(
            prompt_pad_len=60,
            max_new_tokens=50,
        )


def test_hf_length_gate_allows_within_context() -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.model = types.SimpleNamespace(
        config=types.SimpleNamespace(max_position_embeddings=100)
    )

    t._enforce_hf_rollout_max_position_embeddings(prompt_pad_len=60, max_new_tokens=40)


def test_hf_length_gate_skips_when_model_has_no_max_position_embeddings() -> None:
    t = object.__new__(RolloutMatchingSFTTrainer)
    t.model = types.SimpleNamespace(config=types.SimpleNamespace())

    t._enforce_hf_rollout_max_position_embeddings(prompt_pad_len=999, max_new_tokens=999)
