from __future__ import annotations

import types
from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import (
    Stage2ABChannelBConfig,
    Stage2ABChannelBV3K2Config,
    TrainingConfig,
)


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
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": dict(training_section),
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(),
            "channel_b": {},
        },
    }
    prompts = ConfigLoader.resolve_prompts(raw)
    return TrainingConfig.from_mapping(raw, prompts)


def _canonical_stage2_pipeline(
    *,
    token_ce_cfg: dict | None = None,
    duplicate_ul_cfg: dict | None = None,
    duplicate_ul_channels: list[str] | None = None,
    bbox_geo_cfg: dict | None = None,
    coord_reg_cfg: dict | None = None,
) -> dict:
    if token_ce_cfg is None:
        token_ce_cfg = {
            "desc_ce_weight": 1.0,
            "self_context_struct_ce_weight": 0.1,
            "rollout_fn_desc_weight": 1.0,
            "rollout_matched_prefix_struct_weight": 1.0,
        }
    if duplicate_ul_cfg is None:
        duplicate_ul_cfg = {}
    if duplicate_ul_channels is None:
        duplicate_ul_channels = ["B"]
    if bbox_geo_cfg is None:
        bbox_geo_cfg = {
            "smoothl1_weight": 0.0,
            "ciou_weight": 0.0,
            "a1_smoothl1_weight": 0.0,
            "a1_ciou_weight": 0.0,
        }
    if coord_reg_cfg is None:
        coord_reg_cfg = {
            "coord_ce_weight": 0.0,
            "coord_el1_weight": 0.0,
            "coord_ehuber_weight": 0.0,
            "coord_huber_delta": 0.001,
            "coord_entropy_weight": 0.0,
            "coord_gate_weight": 0.0,
            "text_gate_weight": 0.0,
            "soft_ce_weight": 0.0,
            "self_context_soft_ce_weight": 0.0,
            "w1_weight": 0.0,
            "a1_soft_ce_weight": 0.0,
            "a1_w1_weight": 0.0,
            "temperature": 1.0,
            "target_sigma": 2.0,
            "target_truncate": None,
        }
    return {
        "objective": [
            {
                "name": "token_ce",
                "enabled": True,
                "weight": 1.0,
                "channels": ["A", "B"],
                "config": dict(token_ce_cfg),
            },
            {
                "name": "duplicate_ul",
                "enabled": True,
                "weight": 1.0,
                "channels": list(duplicate_ul_channels),
                "config": dict(duplicate_ul_cfg),
            },
            {
                "name": "bbox_geo",
                "enabled": True,
                "weight": 0.0,
                "channels": ["A", "B"],
                "config": dict(bbox_geo_cfg),
            },
            {
                "name": "coord_reg",
                "enabled": True,
                "weight": 0.0,
                "channels": ["A", "B"],
                "config": dict(coord_reg_cfg),
            },
        ],
        "diagnostics": [],
    }


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
        (
            {"enable_pipeline": True},
            r"enable_pipeline has been removed.*runtime-managed.*under DDP it may be disabled",
        ),
        ({"rollout_decode_batch_size": 4}, r"rollout_decode_batch_size has been removed"),
        ({"reordered_gt_sft": False}, r"reordered_gt_sft has been removed"),
        ({"desc_ce_weight_matched": 0.0}, r"desc_ce_weight_matched has been removed"),
        ({"semantic_desc_gate": {"enabled": False}}, r"semantic_desc_gate has been removed"),
    ],
)
def test_stage2_ab_channel_b_removed_keys_fail_fast(payload: dict, expected_msg: str):
    with pytest.raises(ValueError, match=expected_msg):
        Stage2ABChannelBConfig.from_mapping(payload)


def test_stage2_ab_channel_b_timeout_keys_are_supported() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping(
        {
            "duplicate_iou_threshold": 0.9,
            "producer_wait_timeout_s": 0,
            "ddp_phase_timeout_s": 600,
        }
    )
    assert cfg.duplicate_iou_threshold == pytest.approx(0.9)
    assert cfg.producer_wait_timeout_s == pytest.approx(0.0)
    assert cfg.ddp_phase_timeout_s == pytest.approx(600.0)
    assert cfg.v3_k2 == Stage2ABChannelBV3K2Config()


def test_stage2_ab_channel_b_v3_k2_keys_are_supported() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping(
        {
            "v3_k2": {
                "explorer_temperature": 0.6,
                "explorer_top_p": 0.95,
                "explorer_top_k": 32,
                "consistent_iou_threshold": 0.8,
                "recovered_fn_weight": 1.5,
            }
        }
    )
    assert cfg.v3_k2.explorer_temperature == pytest.approx(0.6)
    assert cfg.v3_k2.explorer_top_p == pytest.approx(0.95)
    assert cfg.v3_k2.explorer_top_k == 32
    assert cfg.v3_k2.consistent_iou_threshold == pytest.approx(0.8)
    assert cfg.v3_k2.recovered_fn_weight == pytest.approx(1.5)


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (
            {"explorer_temperature": "oops"},
            r"stage2_ab\.channel_b\.v3_k2\.explorer_temperature must be a float/int",
        ),
        (
            {"explorer_top_p": 0.0},
            r"stage2_ab\.channel_b\.v3_k2\.explorer_top_p must be in \(0, 1\]",
        ),
        (
            {"explorer_top_k": 0},
            r"stage2_ab\.channel_b\.v3_k2\.explorer_top_k must be -1 \(disabled\) or >= 1",
        ),
        (
            {"consistent_iou_threshold": 1.1},
            r"stage2_ab\.channel_b\.v3_k2\.consistent_iou_threshold must be in \[0, 1\]",
        ),
        (
            {"recovered_fn_weight": 0.9},
            r"stage2_ab\.channel_b\.v3_k2\.recovered_fn_weight must be >= 1.0",
        ),
        (
            {"unknown_key": 1.0},
            r"Unknown stage2_ab\.channel_b\.v3_k2 keys",
        ),
    ],
)
def test_stage2_ab_channel_b_v3_k2_invalid_values_fail_fast(
    payload: dict, expected_msg: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=expected_msg):
        Stage2ABChannelBV3K2Config.from_mapping(payload)


def test_stage2_ab_channel_b_timeout_keys_invalid_values_fail_fast() -> None:
    with pytest.raises(
        TypeError,
        match=r"stage2_ab\.channel_b\.producer_wait_timeout_s must be a float/int when set",
    ):
        Stage2ABChannelBConfig.from_mapping({"producer_wait_timeout_s": "oops"})

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.producer_wait_timeout_s must be >= 0",
    ):
        Stage2ABChannelBConfig.from_mapping({"producer_wait_timeout_s": -1})

    with pytest.raises(
        TypeError,
        match=r"stage2_ab\.channel_b\.ddp_phase_timeout_s must be a float/int when set",
    ):
        Stage2ABChannelBConfig.from_mapping({"ddp_phase_timeout_s": "oops"})

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.ddp_phase_timeout_s must be > 0",
    ):
        Stage2ABChannelBConfig.from_mapping({"ddp_phase_timeout_s": 0})

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.ddp_phase_timeout_s must be > 0",
    ):
        Stage2ABChannelBConfig.from_mapping({"ddp_phase_timeout_s": -1})

    with pytest.raises(
        TypeError,
        match=r"stage2_ab\.channel_b\.duplicate_iou_threshold must be a float/int",
    ):
        Stage2ABChannelBConfig.from_mapping({"duplicate_iou_threshold": "oops"})

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.duplicate_iou_threshold must be in \[0, 1\]",
    ):
        Stage2ABChannelBConfig.from_mapping({"duplicate_iou_threshold": -0.1})

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.duplicate_iou_threshold must be in \[0, 1\]",
    ):
        Stage2ABChannelBConfig.from_mapping({"duplicate_iou_threshold": 1.1})


def test_stage2_ab_channel_b_timeout_keys_parse_in_training_config() -> None:
    cfg = _make_stage2_training_config(
        {"per_device_train_batch_size": 1, "effective_batch_size": 1}
    )
    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.channel_b.ddp_phase_timeout_s is None
    assert cfg.stage2_ab.channel_b.producer_wait_timeout_s is None
    assert cfg.stage2_ab.channel_b.v3_k2 == Stage2ABChannelBV3K2Config()

    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(),
            "channel_b": {
                "duplicate_iou_threshold": 0.85,
                "producer_wait_timeout_s": 120.0,
                "ddp_phase_timeout_s": 900.0,
                "v3_k2": {
                    "explorer_temperature": 0.5,
                    "explorer_top_p": 0.92,
                    "explorer_top_k": 16,
                    "consistent_iou_threshold": 0.82,
                    "recovered_fn_weight": 1.75,
                },
            },
        },
    }
    prompts = ConfigLoader.resolve_prompts(raw)
    parsed = TrainingConfig.from_mapping(raw, prompts)
    assert parsed.stage2_ab is not None
    assert parsed.stage2_ab.channel_b.duplicate_iou_threshold == pytest.approx(0.85)
    assert parsed.stage2_ab.channel_b.producer_wait_timeout_s == pytest.approx(120.0)
    assert parsed.stage2_ab.channel_b.ddp_phase_timeout_s == pytest.approx(900.0)
    assert parsed.stage2_ab.channel_b.v3_k2.explorer_temperature == pytest.approx(0.5)
    assert parsed.stage2_ab.channel_b.v3_k2.explorer_top_p == pytest.approx(0.92)
    assert parsed.stage2_ab.channel_b.v3_k2.explorer_top_k == 16
    assert parsed.stage2_ab.channel_b.v3_k2.consistent_iou_threshold == pytest.approx(
        0.82
    )
    assert parsed.stage2_ab.channel_b.v3_k2.recovered_fn_weight == pytest.approx(
        1.75
    )


def test_stage2_pipeline_rejects_channel_b_drop_invalid_struct_multiplier() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(),
            "channel_b": {"drop_invalid_struct_ce_multiplier": 2.0},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.drop_invalid_struct_ce_multiplier",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_token_ce_legacy_invalid_multiplier() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(
                token_ce_cfg={
                    "desc_ce_weight": 1.0,
                    "self_context_struct_ce_weight": 0.1,
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_matched_prefix_struct_weight": 1.0,
                    "rollout_drop_invalid_struct_ce_multiplier": 1.0,
                }
            ),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.pipeline\.objective\[0\]\.config\.rollout_drop_invalid_struct_ce_multiplier",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_requires_duplicate_ul_in_canonical_order() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": {
                "objective": _canonical_stage2_pipeline()["objective"][:1],
                "diagnostics": [],
            },
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"canonical module order",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_requires_duplicate_ul_channels_b_only() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(duplicate_ul_channels=["A", "B"]),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"duplicate_ul must declare channels \['B'\]",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_requires_empty_duplicate_ul_config() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(
                duplicate_ul_cfg={"unknown_weight": 1.0}
            ),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"Unknown stage2_ab\.pipeline\.objective\[1\]\.config keys",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_custom_coord_soft_ce_w1_surface() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
            "coord_soft_ce_w1": {"enabled": True, "soft_ce_weight": 0.25},
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=r"custom\.coord_soft_ce_w1"):
        TrainingConfig.from_mapping(raw, prompts)


def test_rollout_pipeline_rejects_custom_coord_soft_ce_w1_surface() -> None:
    token_ce_cfg = {
        "desc_ce_weight": 1.0,
        "self_context_struct_ce_weight": 0.0,
        "rollout_fn_desc_weight": 1.0,
        "rollout_matched_prefix_struct_weight": 1.0,
    }
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_rollout_aligned",
            "coord_soft_ce_w1": {"enabled": True, "soft_ce_weight": 0.25},
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
            "pipeline": {
                "objective": [
                    {
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "config": dict(token_ce_cfg),
                    }
                ],
                "diagnostics": [],
            },
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=r"custom\.coord_soft_ce_w1"):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_unknown_module_config_keys() -> None:
    coord_reg_cfg = {
        "coord_ce_weight": 0.0,
        "coord_el1_weight": 0.0,
        "coord_ehuber_weight": 0.0,
        "coord_huber_delta": 0.001,
        "coord_entropy_weight": 0.0,
        "coord_gate_weight": 0.0,
        "text_gate_weight": 0.25,
        "soft_ce_weight": 0.0,
        "self_context_soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "a1_soft_ce_weight": 0.0,
        "a1_w1_weight": 0.0,
        "temperature": 1.0,
        "target_sigma": 2.0,
        "target_truncate": None,
    }
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(
                coord_reg_cfg={**coord_reg_cfg, "unknown_weight": 1.0}
            ),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"Unknown stage2_ab\.pipeline\.objective\[3\]\.config keys.*unknown_weight",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_rollout_pipeline_rejects_unknown_module_config_keys() -> None:
    coord_reg_cfg = {
        "coord_ce_weight": 0.0,
        "coord_el1_weight": 0.0,
        "coord_ehuber_weight": 0.0,
        "coord_huber_delta": 0.001,
        "coord_entropy_weight": 0.0,
        "coord_gate_weight": 0.0,
        "text_gate_weight": 0.25,
        "soft_ce_weight": 0.0,
        "self_context_soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "a1_soft_ce_weight": 0.0,
        "a1_w1_weight": 0.0,
        "temperature": 1.0,
        "target_sigma": 2.0,
        "target_truncate": None,
    }
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "fusion_config": "toy/fusion.yaml",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_rollout_aligned",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "rollout_matching": {
            "rollout_backend": "hf",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
            "pipeline": {
                "objective": [
                    {
                        "name": "bbox_geo",
                        "enabled": True,
                        "weight": 0.0,
                        "channels": ["A", "B"],
                        "config": {
                            "smoothl1_weight": 0.0,
                            "ciou_weight": 0.0,
                            "a1_smoothl1_weight": 0.0,
                            "a1_ciou_weight": 0.0,
                        },
                    },
                    {
                        "name": "coord_reg",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "config": {
                            **coord_reg_cfg,
                            "unknown_weight": 1.0,
                        },
                    }
                ],
                "diagnostics": [],
            },
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"Unknown rollout_matching\.pipeline\.objective\[1\]\.config keys.*unknown_weight",
    ):
        TrainingConfig.from_mapping(raw, prompts)


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


def test_resolve_prompts_geometry_first_keeps_random_object_ordering_wording():
    raw = {
        "custom": {
            "object_ordering": "random",
            "object_field_order": "geometry_first",
            "coord_tokens": {"enabled": True},
        }
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    assert "before desc" in prompts.user
    assert "any ordering is acceptable" in prompts.user
    assert "before desc" in str(prompts.system)


def test_resolve_prompts_desc_first_remains_baseline_wording():
    raw = {
        "custom": {
            "object_ordering": "sorted",
            "object_field_order": "desc_first",
            "coord_tokens": {"enabled": True},
        }
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    assert "desc before one geometry" in prompts.user
    assert "desc before exactly one geometry key" in str(prompts.system)
    assert "before desc" not in prompts.user
    assert "before desc" not in str(prompts.system)


def test_resolve_prompts_invalid_object_field_order_fails_fast():
    raw = {
        "custom": {
            "object_field_order": "bbox_first",
        }
    }
    with pytest.raises(ValueError, match="custom.object_field_order"):
        ConfigLoader.resolve_prompts(raw)



def test_rollout_eval_decode_batch_size_overrides_eval_batch_size_when_mismatched():
    from src.sft import _apply_rollout_decode_batch_size_override

    train_args = types.SimpleNamespace(
        trainer_variant="stage2_two_channel",
        training_args=types.SimpleNamespace(per_device_eval_batch_size=1),
    )
    training_config = types.SimpleNamespace(
        rollout_matching={
            "channel_b_decode_batch_size": 4,
            "eval_decode_batch_size": 3,
        }
    )

    resolved = _apply_rollout_decode_batch_size_override(
        train_args=train_args,
        training_config=training_config,
    )

    assert resolved == 3
    assert train_args.training_args.per_device_eval_batch_size == 3
    assert train_args.per_device_eval_batch_size == 3


def test_scope_logging_dir_under_run_name_keeps_tensorboard_runs_named() -> None:
    from src.sft import _scope_logging_dir_under_run_name

    train_args = types.SimpleNamespace(
        run_name="tb_named_run",
        logging_dir="tb/stage2_ab/prod",
        add_version=True,
        training_args=types.SimpleNamespace(
            run_name="tb_named_run",
            logging_dir="tb/stage2_ab/prod",
            add_version=True,
        ),
    )

    resolved = _scope_logging_dir_under_run_name(train_args)

    assert resolved == "tb/stage2_ab/prod/tb_named_run"
    assert train_args.logging_dir == resolved
    assert train_args.training_args.logging_dir == resolved


def test_scope_logging_dir_under_run_name_is_idempotent() -> None:
    from src.sft import _scope_logging_dir_under_run_name

    train_args = types.SimpleNamespace(
        run_name="tb_named_run",
        logging_dir="tb/stage2_ab/prod/tb_named_run",
        training_args=types.SimpleNamespace(
            run_name="tb_named_run",
            logging_dir="tb/stage2_ab/prod/tb_named_run",
        ),
    )

    resolved = _scope_logging_dir_under_run_name(train_args)

    assert resolved == "tb/stage2_ab/prod/tb_named_run"
    assert train_args.logging_dir == resolved
    assert train_args.training_args.logging_dir == resolved


def test_stage2_build_pipeline_manifest_requires_explicit_pipeline():
    from src.sft import _build_pipeline_manifest

    cfg = {
        "desc_ce_weight": 0.7,
        "bbox_smoothl1_weight": 2.0,
        "bbox_ciou_weight": 0.5,
        "coord_gate_weight": 1.0,
        "text_gate_weight": 0.2,
    }
    coord_soft_cfg = {
        "enabled": True,
        "soft_ce_weight": 0.3,
        "w1_weight": 0.4,
        "temperature": 0.9,
        "target_sigma": 1.7,
        "target_truncate": 8,
    }

    with pytest.raises(ValueError, match=r"requires an explicit pipeline config"):
        _build_pipeline_manifest(
            cfg,
            default_objective=["token_ce", "duplicate_ul", "bbox_geo", "coord_reg"],
            default_diagnostics=["coord_diag"],
            trainer_variant="stage2_two_channel",
            config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
            run_name="smoke_ab_mixed_pipeline_explicit",
            seed=17,
            coord_soft_cfg=coord_soft_cfg,
        )


def test_pipeline_manifest_respects_authored_sequence_and_empty_diagnostics():
    from src.sft import _build_pipeline_manifest

    # Dataclass materialization may emit tuples for pipeline sections.
    cfg = {
        "pipeline": {
            "objective": (
                {
                    "name": "token_ce",
                    "enabled": True,
                    "weight": 1.0,
                    "channels": ("A", "B"),
                    "config": {},
                },
                {
                    "name": "duplicate_ul",
                    "enabled": True,
                    "weight": 1.0,
                    "channels": ("B",),
                    "config": {},
                },
            ),
            "diagnostics": (),
        }
    }

    manifest = _build_pipeline_manifest(
        cfg,
        default_objective=["token_ce", "duplicate_ul", "bbox_geo", "coord_reg"],
        default_diagnostics=["coord_diag"],
        trainer_variant="stage2_two_channel",
        config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
        run_name="smoke_manifest_sequence",
        seed=17,
        coord_soft_cfg=None,
    )

    assert [m["name"] for m in manifest["objective"]] == ["token_ce", "duplicate_ul"]
    assert manifest["diagnostics"] == []


def test_stage2_profile_kind_detects_live_two_channel_tree() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert (
        ConfigLoader._canonical_stage2_profile_kind(
            str(repo_root / "configs/stage2_two_channel/prod/ab_mixed.yaml")
        )
        == "prod"
    )
    assert (
        ConfigLoader._canonical_stage2_profile_kind(
            str(repo_root / "configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml")
        )
        == "smoke"
    )


def test_stage2_leaf_contract_accepts_live_prod_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ConfigLoader._validate_stage2_leaf_contract(
        str(repo_root / "configs/stage2_two_channel/prod/ab_mixed.yaml")
    )


def test_stage2_leaf_contract_rejects_live_tree_profile_without_extends() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bad_path = (
        repo_root
        / "configs"
        / "stage2_two_channel"
        / "smoke"
        / "temp_invalid_stage2_profile_for_test.yaml"
    )
    bad_path.write_text("training:\n  run_name: temp_invalid\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match=r"must declare extends/inherit"):
            ConfigLoader._validate_stage2_leaf_contract(str(bad_path))
    finally:
        bad_path.unlink(missing_ok=True)
