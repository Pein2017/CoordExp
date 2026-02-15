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


@pytest.mark.parametrize(
    "section",
    [
        "model",
        "quantization",
        "template",
        "data",
        "tuner",
        "training",
        "rlhf",
    ],
)
def test_unknown_key_fails_fast_with_dotted_path(section: str):
    payload = _base_training_payload()

    if section == "template":
        payload[section] = {"truncation_strategy": "raise", "unknown_key": 1}
    else:
        payload[section] = {"unknown_key": 1}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    # Dotted-path reporting is required for strict parsing.
    assert f"{section}.unknown_key" in str(exc.value)


def test_training_internal_packing_keys_are_allowed():
    payload = _base_training_payload()
    payload["training"] = {
        "packing": True,
        "packing_buffer": 128,
        "packing_min_fill_ratio": 0.5,
        "effective_batch_size": 8,
        "save_delay_steps": 10,
        "save_last_epoch": True,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.training["packing"] is True
    assert cfg.training["packing_buffer"] == 128


def test_unknown_nested_rollout_key_fails_before_trainer_init():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "vllm": {
            "mode": "server",
            "server": {
                "servers": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "group_port": 51216,
                        "unknown_flag": True,
                    }
                ]
            },
        },
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.vllm.server.servers[0].unknown_flag" in str(exc.value)


def test_legacy_rollout_server_paired_list_shape_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "vllm": {
            "mode": "server",
            "server": {
                "base_url": "http://127.0.0.1:8000",
                "group_port": 51216,
            },
        },
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.vllm.server.base_url/group_port" in str(exc.value)


def test_top_level_extra_presence_fails_fast():
    payload = _base_training_payload()
    payload["extra"] = {}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "Top-level extra:" in str(exc.value)
