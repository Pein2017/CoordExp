from __future__ import annotations

import pytest

from src.config.schema import PromptOverrides, TrainingConfig


def _base_training_payload() -> dict:
    # Minimal payload that passes TrainingConfig.from_mapping.
    return {
        "template": {"truncation_strategy": "raise"},
        "custom": {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
        },
    }


def test_vllm_max_model_len_must_cover_global_max_length() -> None:
    payload = _base_training_payload()
    payload["global_max_length"] = 1024
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "max_new_tokens": 64,
        "vllm": {
            "max_model_len": 512,
            "mode": "server",
            "server": {
                "servers": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "group_port": 51216,
                    }
                ]
            },
        },
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "global_max_length" in msg
    assert "rollout_matching.vllm.max_model_len" in msg or "vllm.max_model_len" in msg


def test_rollout_max_new_tokens_must_be_less_than_vllm_max_model_len() -> None:
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "max_new_tokens": 2048,
        "vllm": {
            "max_model_len": 2048,
            "mode": "server",
            "server": {
                "servers": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "group_port": 51216,
                    }
                ]
            },
        },
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "rollout_matching.max_new_tokens" in msg
    assert "rollout_matching.vllm.max_model_len" in msg or "vllm.max_model_len" in msg

