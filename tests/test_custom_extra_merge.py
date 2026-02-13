from __future__ import annotations

from src.config.schema import CustomConfig, PromptOverrides


def test_custom_extra_mapping_is_merged_into_customconfig_extra():
    cfg = CustomConfig.from_mapping(
        {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            # Legacy style: unknown custom.* keys end up in CustomConfig.extra.
            "instability_monitor": {"enabled": True},
            # New style: namespaced knobs under custom.extra.*
            "extra": {"rollout_matching": {"rollout_backend": "hf", "decode_batch_size": 2}},
        },
        prompts=PromptOverrides(),
    )

    assert cfg.extra["instability_monitor"]["enabled"] is True
    assert cfg.extra["rollout_matching"]["rollout_backend"] == "hf"
    assert cfg.extra["rollout_matching"]["decode_batch_size"] == 2

