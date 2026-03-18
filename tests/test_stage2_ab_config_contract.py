from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import (
    Stage2ABChannelBConfig,
    Stage2ABChannelBTriagePosteriorConfig,
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
    dead_anchor_suppression_cfg: dict | None = None,
    dead_anchor_suppression_channels: list[str] | None = None,
    bbox_geo_cfg: dict | None = None,
    bbox_size_aux_cfg: dict | None = None,
    coord_reg_cfg: dict | None = None,
) -> dict:
    if token_ce_cfg is None:
        token_ce_cfg = {
            "desc_ce_weight": 1.0,
            "struct_ce_weight": 0.1,
            "rollout_fn_desc_weight": 1.0,
            "rollout_matched_prefix_struct_weight": 1.0,
        }
    if dead_anchor_suppression_cfg is None:
        dead_anchor_suppression_cfg = {}
    if dead_anchor_suppression_channels is None:
        dead_anchor_suppression_channels = ["B"]
    if bbox_geo_cfg is None:
        bbox_geo_cfg = {
            "smoothl1_weight": 0.0,
            "ciou_weight": 0.0,
        }
    if bbox_size_aux_cfg is None:
        bbox_size_aux_cfg = {
            "log_wh_weight": 0.0,
            "oversize_penalty_weight": 0.0,
            "oversize_area_frac_threshold": None,
            "oversize_log_w_threshold": None,
            "oversize_log_h_threshold": None,
            "eps": 1e-6,
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
            "w1_weight": 0.0,
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
                "application": {"preset": "anchor_text_plus_final_struct"},
                "config": dict(token_ce_cfg),
            },
            {
                "name": "loss_dead_anchor_suppression",
                "enabled": True,
                "weight": 1.0,
                "channels": list(dead_anchor_suppression_channels),
                "application": {"preset": "rollout_only"},
                "config": dict(dead_anchor_suppression_cfg),
            },
            {
                "name": "bbox_geo",
                "enabled": True,
                "weight": 0.0,
                "channels": ["A", "B"],
                "application": {"preset": "anchor_if_single_iter_else_final"},
                "config": dict(bbox_geo_cfg),
            },
            {
                "name": "bbox_size_aux",
                "enabled": True,
                "weight": 0.0,
                "channels": ["A", "B"],
                "application": {"preset": "anchor_if_single_iter_else_final"},
                "config": dict(bbox_size_aux_cfg),
            },
            {
                "name": "coord_reg",
                "enabled": True,
                "weight": 0.0,
                "channels": ["A", "B"],
                "application": {"preset": "anchor_if_single_iter_else_final"},
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
    assert cfg.triage_posterior == Stage2ABChannelBTriagePosteriorConfig()


def test_stage2_ab_channel_b_triage_posterior_keys_are_supported() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping(
        {
            "triage_posterior": {
                "explorer_temperature": 0.6,
                "explorer_top_p": 0.95,
                "explorer_top_k": 32,
                "unlabeled_consistent_iou_threshold": 0.8,
                "recovered_ground_truth_weight_multiplier": 1.5,
            }
        }
    )
    assert cfg.triage_posterior.explorer_temperature == pytest.approx(0.6)
    assert cfg.triage_posterior.explorer_top_p == pytest.approx(0.95)
    assert cfg.triage_posterior.explorer_top_k == 32
    assert cfg.triage_posterior.unlabeled_consistent_iou_threshold == pytest.approx(0.8)
    assert cfg.triage_posterior.recovered_ground_truth_weight_multiplier == pytest.approx(1.5)


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (
            {"explorer_temperature": "oops"},
            r"stage2_ab\.channel_b\.triage_posterior\.explorer_temperature must be a float/int",
        ),
        (
            {"explorer_top_p": 0.0},
            r"stage2_ab\.channel_b\.triage_posterior\.explorer_top_p must be in \(0, 1\]",
        ),
        (
            {"explorer_top_k": 0},
            r"stage2_ab\.channel_b\.triage_posterior\.explorer_top_k must be -1 \(disabled\) or >= 1",
        ),
        (
            {"unlabeled_consistent_iou_threshold": 1.1},
            r"stage2_ab\.channel_b\.triage_posterior\.unlabeled_consistent_iou_threshold must be in \[0, 1\]",
        ),
        (
            {"recovered_ground_truth_weight_multiplier": 0.9},
            r"stage2_ab\.channel_b\.triage_posterior\.recovered_ground_truth_weight_multiplier must be >= 1.0",
        ),
        (
            {"unknown_key": 1.0},
            r"Unknown stage2_ab\.channel_b\.triage_posterior keys",
        ),
    ],
)
def test_stage2_ab_channel_b_triage_posterior_invalid_values_fail_fast(
    payload: dict, expected_msg: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=expected_msg):
        Stage2ABChannelBTriagePosteriorConfig.from_mapping(payload)


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
    assert cfg.stage2_ab.channel_b.triage_posterior == Stage2ABChannelBTriagePosteriorConfig()

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
                "triage_posterior": {
                    "explorer_temperature": 0.5,
                    "explorer_top_p": 0.92,
                    "explorer_top_k": 16,
                    "unlabeled_consistent_iou_threshold": 0.82,
                    "recovered_ground_truth_weight_multiplier": 1.75,
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
    assert parsed.stage2_ab.channel_b.triage_posterior.explorer_temperature == pytest.approx(0.5)
    assert parsed.stage2_ab.channel_b.triage_posterior.explorer_top_p == pytest.approx(0.92)
    assert parsed.stage2_ab.channel_b.triage_posterior.explorer_top_k == 16
    assert parsed.stage2_ab.channel_b.triage_posterior.unlabeled_consistent_iou_threshold == pytest.approx(
        0.82
    )
    assert parsed.stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier == pytest.approx(
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
                    "struct_ce_weight": 0.1,
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


def test_stage2_pipeline_requires_loss_dead_anchor_suppression_in_canonical_order() -> None:
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


def test_stage2_pipeline_requires_dead_anchor_suppression_channels_b_only() -> None:
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
            "pipeline": _canonical_stage2_pipeline(dead_anchor_suppression_channels=["A", "B"]),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"loss_dead_anchor_suppression must declare channels \['B'\]",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_requires_empty_loss_dead_anchor_suppression_config() -> None:
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
                dead_anchor_suppression_cfg={"unknown_weight": 1.0}
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
        "struct_ce_weight": 0.0,
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
                        "application": {"preset": "anchor_text_plus_final_struct"},
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
        "w1_weight": 0.0,
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
        match=r"Unknown stage2_ab\.pipeline\.objective\[4\]\.config keys.*unknown_weight",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_accepts_token_ce_stop_signal_damping_defaults() -> None:
    cfg = _make_stage2_training_config(
        {
            "per_device_train_batch_size": 1,
            "effective_batch_size": 1,
        }
    )

    token_ce_cfg = dict(cfg.stage2_ab.pipeline.objective[0].config)
    assert token_ce_cfg["stop_signal_damping"] == {
        "enabled": False,
        "min_weight": 0.2,
        "max_weight": 1.0,
        "branch_temperature": 1.0,
        "curve_gamma": 2.0,
        "detach_gate": True,
    }


def test_stage2_pipeline_rejects_unknown_stop_signal_damping_key() -> None:
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
                    "struct_ce_weight": 0.1,
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_matched_prefix_struct_weight": 1.0,
                    "stop_signal_damping": {"unknown_key": 1.0},
                }
            ),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.pipeline\.objective\[0\]\.config\.stop_signal_damping",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_invalid_stop_signal_damping_range() -> None:
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
                    "struct_ce_weight": 0.1,
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_matched_prefix_struct_weight": 1.0,
                    "stop_signal_damping": {
                        "enabled": True,
                        "min_weight": 0.9,
                        "max_weight": 0.2,
                    },
                }
            ),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"min_weight must be <= .*max_weight",
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
        "w1_weight": 0.0,
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
                        "application": {"preset": "anchor_if_single_iter_else_final"},
                        "config": {
                            "smoothl1_weight": 0.0,
                            "ciou_weight": 0.0,
                        },
                    },
                    {
                        "name": "coord_reg",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "application": {"preset": "anchor_if_single_iter_else_final"},
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


def test_rollout_pipeline_rejects_unknown_stop_signal_damping_key() -> None:
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
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "application": {"preset": "anchor_text_plus_final_struct"},
                        "config": {
                            "desc_ce_weight": 1.0,
                            "struct_ce_weight": 0.1,
                            "rollout_fn_desc_weight": 1.0,
                            "rollout_matched_prefix_struct_weight": 1.0,
                            "stop_signal_damping": {"oops": 1.0},
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
        match=r"rollout_matching\.pipeline\.objective\[0\]\.config\.stop_signal_damping",
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


def test_strip_trailing_trainer_state_logging_row_removes_final_status_append(
    tmp_path: Path,
) -> None:
    from src.sft import _strip_trailing_trainer_state_logging_row

    log_path = tmp_path / "logging.jsonl"
    metric_row = {"loss": 0.42, "global_step/max_steps": "10/100"}
    eval_row = {"eval/detection/f1": 0.71, "step": 300}
    final_status = {
        "last_model_checkpoint": None,
        "best_model_checkpoint": None,
        "best_metric": None,
        "global_step": 300,
        "log_history": [metric_row, eval_row],
        "memory": 72.5,
    }
    log_path.write_text(
        "\n".join(
            json.dumps(row) for row in (metric_row, eval_row, final_status)
        )
        + "\n",
        encoding="utf-8",
    )

    removed = _strip_trailing_trainer_state_logging_row(log_path)

    assert removed is True
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [metric_row, eval_row]


def test_strip_trailing_trainer_state_logging_row_keeps_flat_metric_rows(
    tmp_path: Path,
) -> None:
    from src.sft import _strip_trailing_trainer_state_logging_row

    log_path = tmp_path / "logging.jsonl"
    rows = [
        {"loss": 0.42, "global_step/max_steps": "10/100"},
        {"eval/detection/f1": 0.71, "step": 300},
    ]
    expected = "\n".join(json.dumps(row) for row in rows) + "\n"
    log_path.write_text(expected, encoding="utf-8")

    removed = _strip_trailing_trainer_state_logging_row(log_path)

    assert removed is False
    assert log_path.read_text(encoding="utf-8") == expected


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
            default_objective=[
                "token_ce",
                "loss_dead_anchor_suppression",
                "bbox_geo",
                "bbox_size_aux",
                "coord_reg",
            ],
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
                    "name": "loss_dead_anchor_suppression",
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
        default_objective=[
            "token_ce",
            "loss_dead_anchor_suppression",
            "bbox_geo",
            "bbox_size_aux",
            "coord_reg",
        ],
        default_diagnostics=["coord_diag"],
        trainer_variant="stage2_two_channel",
        config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
        run_name="smoke_manifest_sequence",
        seed=17,
        coord_soft_cfg=None,
    )

    assert [m["name"] for m in manifest["objective"]] == ["token_ce", "loss_dead_anchor_suppression"]
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
    assert (
        ConfigLoader._canonical_stage2_profile_kind(
            str(
                repo_root
                / "configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml"
            )
        )
        == "ablation"
    )
    assert (
        ConfigLoader._canonical_stage2_profile_kind(
            str(
                repo_root
                / "configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml"
            )
        )
        == "ablation"
    )


def test_stage2_leaf_contract_accepts_live_prod_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ConfigLoader._validate_stage2_leaf_contract(
        str(repo_root / "configs/stage2_two_channel/prod/ab_mixed.yaml")
    )


def test_stage2_leaf_contract_accepts_live_ablation_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ConfigLoader._validate_stage2_leaf_contract(
        str(
            repo_root
            / "configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml"
        )
    )


def test_stage2_leaf_contract_accepts_live_ablation_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ConfigLoader._validate_stage2_leaf_contract(
        str(
            repo_root
            / "configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml"
        )
    )


@pytest.mark.parametrize(
    ("config_rel", "expected_ordering"),
    [
        ("ablation/a_only_iter1-res_1024.yaml", "random"),
        ("smoke/a_only_iter1-res_1024_random_order.yaml", "random"),
    ],
)
def test_stage2_ablation_leaves_pin_ordering_cache_seed_and_names(
    config_rel: str,
    expected_ordering: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs" / "stage2_two_channel" / config_rel)
    )

    training = cfg.training
    custom = cfg.custom

    assert custom.object_ordering == expected_ordering
    assert training["encoded_sample_cache"]["enabled"] is False
    assert training["seed"] == 17
    assert expected_ordering in str(training["run_name"])
    assert expected_ordering in str(training["output_dir"])
    assert expected_ordering in str(training["logging_dir"])

    if config_rel.startswith("smoke/"):
        data = cfg.data
        assert training["max_steps"] == 2
        assert training["eval_strategy"] == "no"
        assert training["save_strategy"] == "no"
        assert custom.train_sample_limit == 128
        assert custom.val_sample_limit == 8
        assert data["dataloader_num_workers"] == 0


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
