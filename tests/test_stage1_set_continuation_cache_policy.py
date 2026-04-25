from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.schema import PromptOverrides, TrainingConfig
from src.sft import (
    EncodedSampleCacheRuntimeConfig,
    PackingRuntimeConfig,
    _build_effective_runtime_payload,
)


def _stage1_set_continuation_payload() -> dict:
    return {
        "template": {"truncation_strategy": "raise"},
        "custom": {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage1_set_continuation",
            "coord_tokens": {
                "enabled": True,
                "skip_bbox_norm": True,
            },
            "stage1_set_continuation": {
                "subset_sampling": {
                    "empty_prefix_ratio": 0.30,
                    "random_subset_ratio": 0.45,
                    "leave_one_out_ratio": 0.20,
                    "full_prefix_ratio": 0.05,
                },
            },
        },
    }


def test_set_continuation_encoded_cache_error_policy_fails_fast() -> None:
    payload = _stage1_set_continuation_payload()
    payload["training"] = {
        "encoded_sample_cache": {
            "enabled": True,
            "ineligible_policy": "error",
        }
    }

    with pytest.raises(
        ValueError,
        match="stage1_set_continuation.*encoded_sample_cache.*ineligible_policy=bypass",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_set_continuation_encoded_cache_bypass_policy_parses() -> None:
    payload = _stage1_set_continuation_payload()
    payload["training"] = {
        "encoded_sample_cache": {
            "enabled": True,
            "ineligible_policy": "bypass",
        }
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.training["encoded_sample_cache"]["enabled"] is True
    assert cfg.training["encoded_sample_cache"]["ineligible_policy"] == "bypass"


def test_set_continuation_encoded_cache_bypass_records_runtime_reason() -> None:
    runtime = _build_effective_runtime_payload(
        training_config=SimpleNamespace(template={}, global_max_length=None),
        train_args=SimpleNamespace(
            output_dir="out",
            logging_dir="logs",
            run_name="unit",
            seed=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
        ),
        trainer_variant="stage1_set_continuation",
        dataset_seed=7,
        checkpoint_mode="artifact_only",
        packing_cfg=PackingRuntimeConfig(enabled=False),
        encoded_sample_cache_cfg=EncodedSampleCacheRuntimeConfig(
            enabled=True,
            root_dir="/tmp/cache",
            ineligible_policy="bypass",
        ),
        train_jsonl="train.jsonl",
        val_jsonl=None,
        pipeline_manifest=None,
        train_encoded_sample_cache_info={
            "enabled": True,
            "status": "bypassed",
            "policy": "bypass",
            "reason": "stage1_set_continuation_branch_sampling",
        },
    )

    cache = runtime["encoded_sample_cache"]
    assert cache["enabled"] is True
    assert cache["status"] == "bypassed"
    assert cache["policy"] == "bypass"
    assert "stage1_set_continuation" in cache["reason"]
    assert cache["train"]["status"] == "bypassed"
