from __future__ import annotations

import types

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import Stage2ABChannelBConfig, TrainingConfig


class _FakeTrainArguments:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.training_args = types.SimpleNamespace()


def _make_stage2_training_config(training_section: dict) -> TrainingConfig:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "trainer_variant": "stage2_ab_training",
        },
        "training": dict(training_section),
        "rollout_matching": {"rollout_backend": "hf"},
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "channel_b": {},
        },
    }
    prompts = ConfigLoader.resolve_prompts(raw)
    return TrainingConfig.from_mapping(raw, prompts)


def _patch_loader_runtime(monkeypatch: pytest.MonkeyPatch, *, world_size: int) -> None:
    monkeypatch.setattr(
        "src.config.loader.get_dist_setting",
        lambda: (0, 0, int(world_size), 0),
    )
    monkeypatch.setattr("src.config.loader.TrainArguments", _FakeTrainArguments)
    monkeypatch.setattr("src.config.loader.RLHFArguments", _FakeTrainArguments)


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        ({"mode": "step"}, r"stage2_ab\.channel_b\.mode has been removed"),
        ({"async": {"enabled": True}}, r"stage2_ab\.channel_b\.async has been removed"),
        ({"rollouts_per_step": 16}, r"rollouts_per_step has been removed"),
        ({"enable_pipeline": True}, r"enable_pipeline has been removed"),
        ({"rollout_decode_batch_size": 4}, r"rollout_decode_batch_size has been removed"),
        ({"reordered_gt_sft": False}, r"reordered_gt_sft has been removed"),
        ({"desc_ce_weight_matched": 0.0}, r"desc_ce_weight_matched has been removed"),
        ({"semantic_desc_gate": {"enabled": False}}, r"semantic_desc_gate has been removed"),
    ],
)
def test_stage2_ab_channel_b_removed_keys_fail_fast(payload: dict, expected_msg: str):
    with pytest.raises(ValueError, match=expected_msg):
        Stage2ABChannelBConfig.from_mapping(payload)


def test_stage2_ab_requires_effective_batch_size(monkeypatch: pytest.MonkeyPatch):
    _patch_loader_runtime(monkeypatch, world_size=2)
    cfg = _make_stage2_training_config({"per_device_train_batch_size": 2})

    with pytest.raises(ValueError, match=r"requires training\.effective_batch_size"):
        ConfigLoader.build_train_arguments(cfg)


def test_stage2_ab_enforces_divisibility(monkeypatch: pytest.MonkeyPatch):
    _patch_loader_runtime(monkeypatch, world_size=2)
    cfg = _make_stage2_training_config(
        {
            "per_device_train_batch_size": 2,
            "effective_batch_size": 10,
        }
    )

    with pytest.raises(ValueError, match=r"must be divisible"):
        ConfigLoader.build_train_arguments(cfg)


def test_stage2_ab_rejects_conflicting_gradient_accumulation(monkeypatch: pytest.MonkeyPatch):
    _patch_loader_runtime(monkeypatch, world_size=2)
    cfg = _make_stage2_training_config(
        {
            "per_device_train_batch_size": 2,
            "effective_batch_size": 8,
            "gradient_accumulation_steps": 3,
        }
    )

    with pytest.raises(ValueError, match=r"must not conflict"):
        ConfigLoader.build_train_arguments(cfg)


def test_stage2_ab_derives_gradient_accumulation_from_effective_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_loader_runtime(monkeypatch, world_size=2)
    cfg = _make_stage2_training_config(
        {
            "per_device_train_batch_size": 2,
            "effective_batch_size": 8,
        }
    )

    args = ConfigLoader.build_train_arguments(cfg)
    assert args.kwargs["gradient_accumulation_steps"] == 2
