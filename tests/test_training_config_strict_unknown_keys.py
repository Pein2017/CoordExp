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
            "object_field_order": "desc_first",
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
        "packing_mode": "static",
        "packing_buffer": 128,
        "packing_min_fill_ratio": 0.5,
        "packing_wait_timeout_s": 7200,
        "packing_length_cache_persist_every": 2048,
        "effective_batch_size": 8,
        "save_delay_steps": 10,
        "save_last_epoch": True,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.training["packing"] is True
    assert cfg.training["packing_mode"] == "static"
    assert cfg.training["packing_buffer"] == 128
    assert cfg.training["packing_wait_timeout_s"] == 7200
    assert cfg.training["packing_length_cache_persist_every"] == 2048


def test_training_removed_packing_exact_key_fails_fast():
    payload = _base_training_payload()
    payload["training"] = {
        "packing": True,
        "packing_mode": "static",
        "packing_require_exact_effective_batch": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "training.packing_require_exact_effective_batch" in str(exc.value)


def test_rollout_eval_detection_and_eval_prompt_variant_keys_are_accepted():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "decode_batch_size": 2,
        "eval_prompt_variant": "coco_80",
        "eval_detection": {
            "enabled": True,
            "metrics": "coco",
            "score_mode": "constant",
            "constant_score": 1.0,
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
        },
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.rollout_matching is not None
    assert cfg.rollout_matching.eval_prompt_variant == "coco_80"
    assert cfg.rollout_matching.eval_detection is not None
    assert cfg.rollout_matching.eval_detection.enabled is True


def test_training_packing_length_deprecated_fails_fast():
    payload = _base_training_payload()
    payload["training"] = {"packing": True, "packing_length": 128}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "training.packing_length" in msg
    assert "deprecated" in msg.lower()


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


def test_unknown_rollout_decoding_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "decoding": {"unknown": True},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "rollout_matching.decoding.unknown" in msg
    assert "Migration guidance" in msg


def test_unknown_rollout_monitor_dump_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "monitor_dump": {"unknown": True},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.monitor_dump.unknown" in str(exc.value)


def test_unknown_rollout_vllm_sync_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "vllm": {"sync": {"unknown": True}},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.vllm.sync.unknown" in str(exc.value)


def test_unknown_stage2_ab_schedule_key_fails_fast():
    payload = _base_training_payload()
    payload["stage2_ab"] = {"schedule": {"b_ratio": 0.5, "unknown_flag": 1}}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "stage2_ab.schedule" in msg
    assert "unknown_flag" in msg


def test_unknown_stage2_ab_key_fails_fast():
    payload = _base_training_payload()
    payload["stage2_ab"] = {"schedule": {"b_ratio": 0.5}, "unknown_top": 1}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "stage2_ab.unknown_top" in msg
    assert "Migration guidance" in msg


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


def test_custom_visual_kd_unknown_section_key_fails():
    payload = _base_training_payload()
    payload["custom"]["visual_kd"] = {
        "enabled": True,
        "vit": {"enabled": True, "weight": 1.0},
        "extra_flag": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.visual_kd.extra_flag" in str(exc.value)


def test_custom_visual_kd_target_unknown_key_fails():
    payload = _base_training_payload()
    payload["custom"]["visual_kd"] = {
        "enabled": True,
        "vit": {"enabled": True, "weight": 1.0, "unknown": 5},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.visual_kd.vit.unknown" in str(exc.value)


def test_custom_visual_kd_target_unknown_key_fails_even_when_disabled():
    payload = _base_training_payload()
    payload["custom"]["visual_kd"] = {
        "enabled": False,
        "vit": {"unknown": 5},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.visual_kd.vit.unknown" in str(exc.value)


def test_top_level_extra_presence_fails_fast():
    payload = _base_training_payload()
    payload["extra"] = {}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "Top-level extra:" in str(exc.value)


def test_custom_object_field_order_is_required():
    payload = _base_training_payload()
    payload["custom"].pop("object_field_order", None)

    with pytest.raises(ValueError, match="custom.object_field_order must be provided"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_unknown_top_level_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["bogus_top"] = 1

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "Unknown top-level config keys" in msg
    assert "bogus_top" in msg
    assert "Migration guidance" in msg
