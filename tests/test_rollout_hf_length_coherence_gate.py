import types

import pytest

from src.infer.backend import enforce_hf_rollout_max_position_embeddings


def test_hf_length_gate_raises_when_prompt_plus_generation_exceeds_context() -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=100))

    with pytest.raises(ValueError, match=r"max_position_embeddings"):
        enforce_hf_rollout_max_position_embeddings(
            model=model,
            prompt_pad_len=60,
            max_new_tokens=50,
        )


def test_hf_length_gate_allows_within_context() -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=100))

    enforce_hf_rollout_max_position_embeddings(
        model=model, prompt_pad_len=60, max_new_tokens=40
    )


def test_hf_length_gate_skips_when_model_has_no_max_position_embeddings() -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace())

    enforce_hf_rollout_max_position_embeddings(
        model=model, prompt_pad_len=999, max_new_tokens=999
    )
