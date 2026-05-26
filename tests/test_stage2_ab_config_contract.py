from __future__ import annotations

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import TrainingConfig


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


def _pipeline(*, name: str = "residual_set_correction") -> dict:
    return {
        "objective": [
            {
                "name": name,
                "enabled": True,
                "weight": 1.0,
                "application": {"preset": "rollout_self_prefix"},
                "config": _residual_set_config(),
            }
        ],
        "diagnostics": [],
    }


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
            "pipeline": _pipeline(),
            "correction": {},
        },
    }


def _load(payload: dict) -> TrainingConfig:
    prompts = ConfigLoader.resolve_prompts(payload)
    return TrainingConfig.from_mapping(payload, prompts)


def test_stage2_ab_top_level_namespace_is_removed() -> None:
    payload = _base_payload()
    payload["stage2_ab"] = {
        "schedule": {"b_ratio": 1.0},
        "pipeline": {"objective": []},
    }

    with pytest.raises(ValueError, match="stage2_ab.*removed.*stage2_rollout_correction"):
        _load(payload)


def test_stage2_two_channel_variant_is_removed() -> None:
    payload = _base_payload()
    payload["custom"]["trainer_variant"] = "stage2_two_channel"

    with pytest.raises(
        ValueError,
        match="stage2_two_channel.*removed.*stage2_rollout_correction",
    ):
        _load(payload)


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"schedule": {"b_ratio": 1.0}}, "schedule.*removed"),
        ({"b_ratio": 1.0}, "b_ratio.*removed"),
        ({"channel_b": {}}, "channel_b.*removed"),
        (
            {
                "pipeline": {
                    "objective": [
                        {
                            "name": "residual_set_correction",
                            "enabled": True,
                            "weight": 1.0,
                            "channels": ["B"],
                            "application": {"preset": "rollout_self_prefix"},
                            "config": _residual_set_config(),
                        }
                    ],
                    "diagnostics": [],
                }
            },
            "channels.*removed",
        ),
        ({"pipeline": _pipeline(name="token_ce")}, "token_ce.*removed"),
        ({"pipeline": _pipeline(name="hard_sft")}, "hard_sft.*removed"),
        ({"pipeline": _pipeline(name="stage2_trie_ce")}, "stage2_trie_ce.*removed"),
        ({"correction": {"pseudo_positive": {"enabled": True}}}, "pseudo_positive.*removed"),
    ],
)
def test_removed_stage2_ab_contract_keys_fail_fast(
    patch: dict,
    expected: str,
) -> None:
    payload = _base_payload()
    payload["stage2_rollout_correction"].update(patch)

    with pytest.raises(ValueError, match=expected):
        _load(payload)


def test_stage2_rollout_correction_minimal_contract_loads() -> None:
    cfg = _load(_base_payload())

    assert cfg.custom.trainer_variant == "stage2_rollout_correction"
    assert cfg.stage2_rollout_correction is not None
    assert cfg.stage2_rollout_correction.pipeline.objective[0].name == (
        "residual_set_correction"
    )
