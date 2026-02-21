from __future__ import annotations

import pytest

from src.config.schema import CustomConfig, PromptOverrides


def _base_custom_payload() -> dict:
    return {
        "train_jsonl": "train.jsonl",
        "user_prompt": "prompt",
        "emit_norm": "none",
        "json_format": "standard",
        "object_field_order": "desc_first",
    }


def test_custom_extra_bucket_accepts_minor_keys():
    cfg = CustomConfig.from_mapping(
        {**_base_custom_payload(), "extra": {"some_minor_toggle": True}},
        prompts=PromptOverrides(),
    )

    assert cfg.extra["some_minor_toggle"] is True


def test_unknown_custom_keys_fail_fast_instead_of_merging_into_custom_extra():
    with pytest.raises(ValueError) as exc:
        CustomConfig.from_mapping(
            {**_base_custom_payload(), "instability_monitor": {"enabled": True}},
            prompts=PromptOverrides(),
        )

    # Dotted-path reporting is required for strict config parsing.
    assert "Unknown custom keys" in str(exc.value)
    assert "custom.instability_monitor" in str(exc.value)


def test_custom_extra_rollout_matching_is_rejected():
    with pytest.raises(ValueError) as exc:
        CustomConfig.from_mapping(
            {
                **_base_custom_payload(),
                "extra": {
                    "rollout_matching": {"rollout_backend": "hf", "decode_batch_size": 2}
                },
            },
            prompts=PromptOverrides(),
        )

    assert "custom.extra.rollout_matching is unsupported" in str(exc.value)


def test_custom_object_field_order_accepts_geometry_first():
    cfg = CustomConfig.from_mapping(
        {**_base_custom_payload(), "object_field_order": "geometry_first"},
        prompts=PromptOverrides(),
    )

    assert cfg.object_field_order == "geometry_first"


def test_custom_object_field_order_invalid_fails_fast():
    with pytest.raises(ValueError, match="custom.object_field_order"):
        CustomConfig.from_mapping(
            {**_base_custom_payload(), "object_field_order": "bbox_first"},
            prompts=PromptOverrides(),
        )


def test_custom_object_field_order_required():
    payload = _base_custom_payload()
    payload.pop("object_field_order", None)
    with pytest.raises(ValueError, match="custom.object_field_order must be provided"):
        CustomConfig.from_mapping(payload, prompts=PromptOverrides())
