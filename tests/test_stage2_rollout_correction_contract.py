from __future__ import annotations

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import TrainingConfig
from src.training_runtime.plan import resolve_training_runtime_plan
from src.training_runtime.profile import resolve_training_runtime_profile


def _residual_set_config() -> dict:
    return {
        "expected_num_rollouts": 4,
        "base_seed": 17,
        "lambda_type": 1.0,
        "lambda_inner": 1.0,
        "fallback_loss_weight": 1.0,
        "lambda_ul_promoted": 0.5,
        "label_conflict_weight": 0.25,
        "commit_iou_threshold": 0.75,
        "duplicate_burst_iou_threshold": 0.95,
        "ul_cluster_iou_threshold": 0.9,
        "ul_gray_iou_low": 0.30,
        "ul_consensus_ratio": 1.0,
        "min_ul_valid_rollouts": 4,
        "clean_gt_sft_mix": 0,
        "strict_builder_invariants": True,
    }


def _rollout_correction_pipeline(
    *,
    name: str = "residual_set_correction",
    application_preset: str = "rollout_self_prefix",
    include_channels: bool = False,
) -> dict:
    spec = {
        "name": name,
        "enabled": True,
        "weight": 1.0,
        "application": {"preset": application_preset},
        "config": _residual_set_config(),
    }
    if include_channels:
        spec["channels"] = ["B"]
    return {"objective": [spec], "diagnostics": []}


def _base_payload() -> dict:
    return {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_rollout_correction",
        },
        "training": {
            "per_device_train_batch_size": 1,
            "effective_batch_size": 1,
        },
        "rollout_matching": {
            "rollout_backend": "hf",
            "rollout_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_rollout_correction": {
            "pipeline": _rollout_correction_pipeline(),
            "correction": {},
        },
    }


def _load(payload: dict) -> TrainingConfig:
    prompts = ConfigLoader.resolve_prompts(payload)
    return TrainingConfig.from_mapping(payload, prompts)


def test_stage2_rollout_correction_runtime_plan_is_canonical() -> None:
    plan = resolve_training_runtime_plan("stage2_rollout_correction")
    profile = resolve_training_runtime_profile("stage2_rollout_correction")

    assert plan.preserve_raw_sample_metadata is True
    assert plan.dataset_static_packing_allowed is False
    assert plan.post_rollout_packing_owner == "trainer"
    assert plan.collator_family == "identity"
    assert plan.ordinary_stage1_mixins_allowed is False
    assert plan.required_pipeline_namespace == "stage2_rollout_correction.pipeline"
    assert plan.requires_top_level_rollout_matching is True
    assert profile.manifest_family == "stage2_rollout_correction"


def test_stage2_two_channel_variant_is_hard_cut() -> None:
    raw = _base_payload()
    raw["custom"]["trainer_variant"] = "stage2_two_channel"
    raw["stage2_ab"] = {
        "schedule": {"b_ratio": 1.0},
        "pipeline": _rollout_correction_pipeline(include_channels=True),
        "channel_b": {},
    }
    raw.pop("stage2_rollout_correction")

    with pytest.raises(
        ValueError,
        match="stage2_two_channel.*removed.*stage2_rollout_correction",
    ):
        _load(raw)


def test_stage2_ab_namespace_is_hard_cut_even_with_new_variant() -> None:
    raw = _base_payload()
    raw["stage2_ab"] = {
        "schedule": {"b_ratio": 1.0},
        "pipeline": _rollout_correction_pipeline(include_channels=True),
        "channel_b": {},
    }

    with pytest.raises(ValueError, match="stage2_ab.*removed.*stage2_rollout_correction"):
        _load(raw)


@pytest.mark.parametrize(
    "patch, expected",
    [
        ({"schedule": {"b_ratio": 1.0}}, "schedule.*removed"),
        ({"b_ratio": 1.0}, "b_ratio.*removed"),
        (
            {"pipeline": _rollout_correction_pipeline(include_channels=True)},
            "channels.*removed",
        ),
        (
            {"pipeline": _rollout_correction_pipeline(name="token_ce")},
            "token_ce.*removed",
        ),
        (
            {"pipeline": _rollout_correction_pipeline(name="hard_sft")},
            "hard_sft.*removed",
        ),
        (
            {"pipeline": _rollout_correction_pipeline(name="stage2_trie_ce")},
            "stage2_trie_ce.*removed",
        ),
        (
            {"correction": {"pseudo_positive": {"enabled": True}}},
            "pseudo_positive.*removed",
        ),
    ],
)
def test_stage2_rollout_correction_rejects_ab_era_keys(
    patch: dict,
    expected: str,
) -> None:
    raw = _base_payload()
    raw["stage2_rollout_correction"].update(patch)

    with pytest.raises(ValueError, match=expected):
        _load(raw)


def test_minimal_stage2_rollout_correction_config_loads() -> None:
    cfg = _load(_base_payload())

    assert cfg.custom.trainer_variant == "stage2_rollout_correction"
    assert cfg.stage2_rollout_correction.pipeline.objective[0].name == (
        "residual_set_correction"
    )
    assert cfg.stage2_rollout_correction.correction.rollout_template_family == (
        "coordjson"
    )
