from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import (
    Stage2ABChannelBAssignmentConfig,
    Stage2ABChannelBConfig,
    Stage2ABChannelBDuplicateControlConfig,
    Stage2ABChannelBPseudoPositiveConfig,
    Stage2ABChannelBTriagePosteriorConfig,
    TrainingConfig,
)


class _FakeTrainArguments:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.training_args = types.SimpleNamespace()


def _make_stage2_training_payload(training_section: dict | None = None) -> dict:
    if training_section is None:
        training_section = {
            "per_device_train_batch_size": 1,
            "effective_batch_size": 1,
        }

    return {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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


def _make_stage2_training_config(training_section: dict) -> TrainingConfig:
    raw = _make_stage2_training_payload(training_section)
    prompts = ConfigLoader.resolve_prompts(raw)
    return TrainingConfig.from_mapping(raw, prompts)


def _canonical_stage2_pipeline(
    *,
    token_ce_cfg: dict | None = None,
) -> dict:
    if token_ce_cfg is None:
        token_ce_cfg = {
            "desc_ce_weight": 1.0,
            "rollout_fn_desc_weight": 1.0,
            "rollout_global_prefix_struct_ce_weight": 1.0,
        }
    return {
        "objective": [
            {
                "name": "token_ce",
                "enabled": True,
                "weight": 1.0,
                "channels": ["A", "B"],
                "application": {"preset": "anchor_text_only"},
                "config": dict(token_ce_cfg),
            },
        ],
        "diagnostics": [],
    }


def _stage2_pipeline_with_channel_b_trie_ce() -> dict:
    pipeline = _canonical_stage2_pipeline()
    pipeline["objective"][0]["channels"] = ["A"]
    pipeline["objective"].insert(
        1,
        {
            "name": "stage2_trie_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_trie_hard_ce"},
            "config": _residual_set_config(),
        },
    )
    return pipeline


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


def _make_raw_with_residual_set_config(config: dict) -> dict:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": dict(config),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}
    raw["stage2_ab"]["channel_b"].pop("triage_posterior", None)
    return raw


def _teacher_forcing_objective() -> dict:
    return {
        "id": "teacher_forcing",
        "profile": "pure_valid_set_marginal",
        "target_ir": {
            "rollin_policy": {
                "name": "random_permutation",
                "base_seed": 17,
            },
        },
        "modules": {
            "token_type_mass": {"enabled": True},
            "conditional_valid_set_likelihood": {"enabled": True},
            "within_valid_coverage": {
                "enabled": False,
                "coverage_strength": 0.0,
            },
            "continuation_margin": {"enabled": False},
        },
    }


def _hard_sft_teacher_forcing_objective() -> dict:
    objective = _teacher_forcing_objective()
    objective["profile"] = "hard_sft"
    objective["modules"] = {
        "token_type_mass": {"enabled": False},
        "conditional_valid_set_likelihood": {"enabled": False},
        "within_valid_coverage": {
            "enabled": False,
            "coverage_strength": 0.0,
        },
        "continuation_margin": {"enabled": False},
    }
    return objective


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
            "duplicate_control": {
                "iou_threshold": 0.9,
                "center_radius_scale": 0.7,
            },
            "producer_wait_timeout_s": 0,
            "ddp_phase_timeout_s": 600,
        }
    )
    assert cfg.duplicate_control == Stage2ABChannelBDuplicateControlConfig(
        iou_threshold=0.9,
        center_radius_scale=0.7,
    )
    assert cfg.producer_wait_timeout_s == pytest.approx(0.0)
    assert cfg.ddp_phase_timeout_s == pytest.approx(600.0)
    assert cfg.triage_posterior == Stage2ABChannelBTriagePosteriorConfig()


def test_stage2_ab_channel_b_rollout_template_defaults_to_explicit_legacy() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping({})

    assert cfg.rollout_template_family == "coordjson"
    assert cfg.rollout_decode_policy == "legacy_coordjson"
    assert cfg.invalid_rollout_policy == "abort"
    assert cfg.fallback_loss_weight == pytest.approx(1.0)
    assert cfg.fp_policy.mode == "zero_loss_context"
    assert cfg.fp_policy.weak_positive_weight == pytest.approx(0.05)
    assert cfg.fp_policy.require_explorer_support is True
    assert cfg.fp_policy.min_support_count == 1
    assert cfg.fp_policy.require_token_score is False


def test_stage2_ab_channel_b_assignment_defaults_to_greedy_iou() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping({})

    assert cfg.assignment == Stage2ABChannelBAssignmentConfig(
        strategy="greedy_iou",
        iou_threshold=None,
    )


def test_stage2_ab_channel_b_assignment_accepts_greedy_iou() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping(
        {"assignment": {"strategy": "greedy-iou", "iou_threshold": 0.55}}
    )

    assert cfg.assignment.strategy == "greedy_iou"
    assert cfg.assignment.iou_threshold == pytest.approx(0.55)


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (
            {"strategy": "legacy_hungarian_mask_iou"},
            r"legacy_hungarian_mask_iou has been removed; use greedy_iou",
        ),
        (
            {"strategy": "oops"},
            r"stage2_ab\.channel_b\.assignment\.strategy must be one of",
        ),
        (
            {"strategy": "greedy_iou", "iou_threshold": 1.5},
            r"stage2_ab\.channel_b\.assignment\.iou_threshold must be in \[0, 1\]",
        ),
        (
            {"unexpected": True},
            r"Unknown stage2_ab\.channel_b\.assignment keys",
        ),
    ],
)
def test_stage2_ab_channel_b_assignment_invalid_values_fail_fast(
    payload: dict,
    expected_msg: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected_msg):
        Stage2ABChannelBAssignmentConfig.from_mapping(payload)


def test_stage2_ab_channel_b_compact_full_derives_runtime_policy_defaults() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping({"rollout_template_family": "compact-full"})

    assert cfg.rollout_template_family == "compact_full"
    assert cfg.rollout_decode_policy == "unconstrained"
    assert cfg.invalid_rollout_policy == "fallback_gt_fn_append_only"
    assert cfg.fallback_loss_weight == pytest.approx(1.0)


def test_stage2_ab_channel_b_rejects_unknown_fp_policy() -> None:
    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.fp_policy\.mode must be one of",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"fp_policy": {"mode": "score_threshold_context"}}
        )


@pytest.mark.parametrize("min_support_count", [True, 1.5])
def test_stage2_ab_channel_b_rejects_invalid_fp_policy_min_support_count(
    min_support_count: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
        match=r"stage2_ab\.channel_b\.fp_policy\.min_support_count",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"fp_policy": {"min_support_count": min_support_count}}
        )


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (
            {"rollout_template_family": "compact_full", "rollout_decode_policy": "legacy_coordjson"},
            r"stage2_ab\.channel_b\.rollout_decode_policy.*compact_full",
        ),
        (
            {"rollout_template_family": "coordjson", "rollout_decode_policy": "unconstrained"},
            r"stage2_ab\.channel_b\.rollout_decode_policy.*coordjson",
        ),
        (
            {"rollout_template_family": "coordjson", "rollout_decode_policy": "compact_grammar"},
            r"stage2_ab\.channel_b\.rollout_decode_policy.*coordjson",
        ),
        (
            {"rollout_template_family": "coordjson", "invalid_rollout_policy": "fallback_gt_fn_append_only"},
            r"fallback_gt_fn_append_only.*coordjson",
        ),
        (
            {"rollout_template_family": "compact_full", "invalid_rollout_policy": "abort"},
            r"invalid_rollout_policy for compact_full",
        ),
        (
            {"rollout_template_family": "compact_full", "invalid_rollout_policy": "dump_and_continue"},
            r"invalid_rollout_policy for compact_full",
        ),
        (
            {"rollout_template_family": "compact_full", "fallback_loss_weight": -0.1},
            r"stage2_ab\.channel_b\.fallback_loss_weight must be >= 0",
        ),
        (
            {"rollout_template_family": "compact_full", "fallback_loss_weight": float("inf")},
            r"stage2_ab\.channel_b\.fallback_loss_weight must be finite",
        ),
    ],
)
def test_stage2_ab_channel_b_rollout_template_invalid_values_fail_fast(
    payload: dict, expected_msg: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=expected_msg):
        Stage2ABChannelBConfig.from_mapping(payload)


def test_stage2_ab_rejects_compact_full_detection_with_default_coordjson_rollout_surface() -> None:
    raw = _make_stage2_training_payload()
    raw["custom"]["detection_sequence_format"] = "compact_full"

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"detection_sequence_format=compact_full.*rollout_template_family=compact_full",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_ab_rejects_coordjson_detection_with_compact_full_rollout_surface() -> None:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["channel_b"] = {"rollout_template_family": "compact_full"}

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"detection_sequence_format=coordjson.*rollout_template_family=coordjson",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_ab_accepts_compact_full_detection_with_compact_full_rollout_surface() -> None:
    raw = _make_stage2_training_payload()
    raw["custom"]["detection_sequence_format"] = "compact_full"
    raw["stage2_ab"]["channel_b"] = {"rollout_template_family": "compact_full"}

    prompts = ConfigLoader.resolve_prompts(raw)
    cfg = TrainingConfig.from_mapping(raw, prompts)

    assert cfg.custom.detection_sequence_format == "compact_full"
    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.channel_b.rollout_template_family == "compact_full"


def test_stage2_ab_channel_b_pseudo_positive_keys_are_supported() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping(
        {
            "pseudo_positive": {
                "enabled": True,
                "coord_weight": 0.6,
            }
        }
    )
    assert cfg.pseudo_positive.enabled is True
    assert cfg.pseudo_positive.coord_weight == pytest.approx(0.6)
    assert cfg.triage_posterior.num_rollouts == 4


def test_stage2_ab_channel_b_triage_posterior_keys_are_supported() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping(
        {
            "pseudo_positive": {"enabled": True},
            "triage_posterior": {
                "num_rollouts": 4,
                "explorer_temperature": 0.6,
                "rollout_temperatures": [0.0, 0.4, 0.6, 0.8],
                "explorer_top_p": 0.95,
                "explorer_top_k": 32,
                "unlabeled_consistent_iou_threshold": 0.8,
                "recovered_ground_truth_weight_multiplier": 1.5,
            }
        }
    )
    assert cfg.triage_posterior.num_rollouts == 4
    assert cfg.triage_posterior.explorer_temperature == pytest.approx(0.6)
    assert cfg.triage_posterior.rollout_temperatures == pytest.approx(
        (0.0, 0.4, 0.6, 0.8)
    )
    assert cfg.triage_posterior.explorer_top_p == pytest.approx(0.95)
    assert cfg.triage_posterior.explorer_top_k == 32
    assert cfg.triage_posterior.unlabeled_consistent_iou_threshold == pytest.approx(0.8)
    assert cfg.triage_posterior.recovered_ground_truth_weight_multiplier == pytest.approx(1.5)


def test_stage2_ab_channel_b_num_rollouts_follow_pseudo_positive_mode() -> None:
    disabled_cfg = Stage2ABChannelBConfig.from_mapping({})
    assert disabled_cfg.pseudo_positive == Stage2ABChannelBPseudoPositiveConfig()
    assert disabled_cfg.triage_posterior.num_rollouts == 2

    k4_cfg = Stage2ABChannelBConfig.from_mapping(
        {
            "pseudo_positive": {"enabled": False},
            "triage_posterior": {
                "num_rollouts": 4,
                "rollout_temperatures": [0.0, 0.4, 0.7, 1.0],
            },
        }
    )
    assert k4_cfg.pseudo_positive.enabled is False
    assert k4_cfg.triage_posterior.num_rollouts == 4
    assert k4_cfg.triage_posterior.rollout_temperatures == pytest.approx(
        (0.0, 0.4, 0.7, 1.0)
    )

    enabled_cfg = Stage2ABChannelBConfig.from_mapping(
        {"pseudo_positive": {"enabled": True}}
    )
    assert enabled_cfg.triage_posterior.num_rollouts == 4

    with pytest.raises(
        ValueError,
        match=(
            r"stage2_ab\.channel_b\.triage_posterior\.num_rollouts must be >= 4 "
            r"when stage2_ab\.channel_b\.pseudo_positive\.enabled=true"
        ),
    ):
        Stage2ABChannelBConfig.from_mapping(
            {
                "pseudo_positive": {"enabled": True},
                "triage_posterior": {"num_rollouts": 2},
            }
        )


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (
            {"num_rollouts": 1},
            r"stage2_ab\.channel_b\.triage_posterior\.num_rollouts must be >= 2",
        ),
        (
            {"explorer_temperature": "oops"},
            r"stage2_ab\.channel_b\.triage_posterior\.explorer_temperature must be a float/int",
        ),
        (
            {"num_rollouts": 4, "rollout_temperatures": [0.0, 0.4]},
            r"stage2_ab\.channel_b\.triage_posterior\.rollout_temperatures length must be 1 or match num_rollouts",
        ),
        (
            {"num_rollouts": 4, "rollout_temperatures": [0.0, -0.1, 0.6, 0.8]},
            r"stage2_ab\.channel_b\.triage_posterior\.rollout_temperatures must contain only values >= 0",
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


@pytest.mark.parametrize(
    "payload, expected_msg",
    [
        (
            {"enabled": "maybe"},
            r"stage2_ab\.channel_b\.pseudo_positive\.enabled string value 'maybe' is not a recognized boolean representation",
        ),
        (
            {"coord_weight": "oops"},
            r"stage2_ab\.channel_b\.pseudo_positive\.coord_weight must be a float/int",
        ),
        (
            {"coord_weight": 1.0},
            r"stage2_ab\.channel_b\.pseudo_positive\.coord_weight must be in \(0, 1\)",
        ),
        (
            {"coord_weight_v1": 0.5},
            r"Versioned pseudo-positive knob aliases are unsupported",
        ),
        (
            {"unknown_key": 1.0},
            r"Unknown stage2_ab\.channel_b\.pseudo_positive keys",
        ),
    ],
)
def test_stage2_ab_channel_b_pseudo_positive_invalid_values_fail_fast(
    payload: dict, expected_msg: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=expected_msg):
        Stage2ABChannelBPseudoPositiveConfig.from_mapping(payload)


@pytest.mark.parametrize("key", ["pseudo_positive_v1", "v1_pseudo_positive"])
def test_stage2_ab_channel_b_rejects_versioned_pseudo_positive_top_level_aliases(
    key: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"Versioned pseudo-positive knob aliases are unsupported",
    ):
        Stage2ABChannelBConfig.from_mapping({key: {"enabled": True}})


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
        match=r"stage2_ab\.channel_b\.duplicate_control\.iou_threshold must be a float/int",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"duplicate_control": {"iou_threshold": "oops"}}
        )

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.duplicate_control\.iou_threshold must be in \[0, 1\]",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"duplicate_control": {"iou_threshold": -0.1}}
        )

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.duplicate_control\.iou_threshold must be in \[0, 1\]",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"duplicate_control": {"iou_threshold": 1.1}}
        )

    with pytest.raises(
        TypeError,
        match=r"stage2_ab\.channel_b\.duplicate_control\.center_radius_scale must be a float/int",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"duplicate_control": {"center_radius_scale": "oops"}}
        )

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.duplicate_control\.center_radius_scale must be >= 0",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"duplicate_control": {"center_radius_scale": -0.1}}
        )

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.duplicate_iou_threshold has been removed",
    ):
        Stage2ABChannelBConfig.from_mapping({"duplicate_iou_threshold": 0.9})

    with pytest.raises(
        ValueError,
        match=r"Unknown stage2_ab\.channel_b\.duplicate_control keys",
    ):
        Stage2ABChannelBConfig.from_mapping(
            {"duplicate_control": {"unexpected": 1}}
        )


def test_stage2_ab_channel_b_timeout_keys_parse_in_training_config() -> None:
    cfg = _make_stage2_training_config(
        {"per_device_train_batch_size": 1, "effective_batch_size": 1}
    )
    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.channel_b.ddp_phase_timeout_s is None
    assert cfg.stage2_ab.channel_b.producer_wait_timeout_s is None
    assert cfg.stage2_ab.channel_b.pseudo_positive == Stage2ABChannelBPseudoPositiveConfig()
    assert cfg.stage2_ab.channel_b.triage_posterior == Stage2ABChannelBTriagePosteriorConfig()

    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
                "duplicate_control": {
                    "iou_threshold": 0.85,
                    "center_radius_scale": 0.65,
                },
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
    assert parsed.stage2_ab.channel_b.duplicate_control == (
        Stage2ABChannelBDuplicateControlConfig(
            iou_threshold=0.85,
            center_radius_scale=0.65,
        )
    )
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


def test_stage2_ab_pseudo_positive_keys_parse_in_training_config() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
                "pseudo_positive": {
                    "enabled": True,
                    "coord_weight": 0.55,
                },
            },
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    parsed = TrainingConfig.from_mapping(raw, prompts)
    assert parsed.stage2_ab is not None
    assert parsed.stage2_ab.channel_b.pseudo_positive.enabled is True
    assert parsed.stage2_ab.channel_b.pseudo_positive.coord_weight == pytest.approx(0.55)
    assert parsed.stage2_ab.channel_b.triage_posterior.num_rollouts == 4


def test_stage2_pipeline_rejects_channel_b_drop_invalid_struct_multiplier() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_global_prefix_struct_ce_weight": 1.0,
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


def test_stage2_pipeline_uses_canonical_objective_without_duplicate_burst_unlikelihood() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    parsed = TrainingConfig.from_mapping(raw, prompts)

    assert [
        module.name for module in parsed.stage2_ab.pipeline.objective
    ] == ["token_ce"]


def test_stage2_pipeline_rejects_legacy_modules_under_teacher_forcing() -> None:
    raw = _make_stage2_training_payload()
    raw["objective"] = _teacher_forcing_objective()

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"teacher_forcing.*stage2_ab\.pipeline\.objective.*token_ce",
    ):
        TrainingConfig.from_mapping(raw, prompts)


@pytest.mark.parametrize(
    "legacy_module",
    ["bbox_geo", "bbox_size_aux", "coord_reg", "soft_ce", "w1", "token_ce"],
)
def test_stage2_pipeline_rejects_each_legacy_module_under_teacher_forcing(
    legacy_module: str,
) -> None:
    raw = _make_stage2_training_payload()
    raw["objective"] = _teacher_forcing_objective()
    raw["stage2_ab"]["pipeline"] = {
        "objective": [
            {
                "name": legacy_module,
                "enabled": True,
                "weight": 1.0,
                "channels": ["A", "B"],
                "application": {"preset": "anchor_only"},
                "config": {},
            }
        ],
        "diagnostics": [],
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=rf"teacher_forcing.*stage2_ab\.pipeline\.objective.*{legacy_module}",
    ):
        TrainingConfig.from_mapping(raw, prompts)


@pytest.mark.parametrize(
    "legacy_key",
    ["coord_gate", "text_gate", "coord_gate_weight", "text_gate_weight"],
)
def test_stage2_pipeline_rejects_legacy_gate_configs_under_teacher_forcing(
    legacy_key: str,
) -> None:
    raw = _make_stage2_training_payload()
    raw["objective"] = _teacher_forcing_objective()
    raw["stage2_ab"]["pipeline"] = {
        "objective": [
            {
                "name": "hard_sft",
                "enabled": True,
                "weight": 1.0,
                "channels": ["A", "B"],
                "application": {"preset": "target_ir"},
                "config": {legacy_key: 1.0},
            }
        ],
        "diagnostics": [],
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=rf"teacher_forcing.*{legacy_key}"):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_compiles_empty_pipeline_from_hard_sft_objective() -> None:
    raw = _make_stage2_training_payload()
    raw["objective"] = _hard_sft_teacher_forcing_objective()
    raw["stage2_ab"]["pipeline"] = {"objective": [], "diagnostics": []}

    prompts = ConfigLoader.resolve_prompts(raw)
    cfg = TrainingConfig.from_mapping(raw, prompts)

    assert cfg.objective is not None
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "hard_sft"
    assert cfg.stage2_ab is not None
    assert [module.name for module in cfg.stage2_ab.pipeline.objective] == ["hard_sft"]


def test_stage2_pipeline_accepts_channel_b_stage2_trie_ce() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "pipeline": _stage2_pipeline_with_channel_b_trie_ce(),
            "channel_b": {
                "fallback_loss_weight": 0.25,
                "insertion_order": "fn_slot_shuffle",
                "fp_policy": {
                    "mode": "weak_positive_context",
                    "weak_positive_weight": 0.05,
                    "require_explorer_support": True,
                    "min_support_count": 1,
                    "require_token_score": False,
                },
                "triage_posterior": {"num_rollouts": 4},
            },
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    parsed = TrainingConfig.from_mapping(raw, prompts)

    assert parsed.stage2_ab is not None
    assert [module.name for module in parsed.stage2_ab.pipeline.objective] == [
        "token_ce",
        "stage2_trie_ce",
    ]
    trie_objective = parsed.stage2_ab.pipeline.objective[1]
    assert trie_objective.application["preset"] == "rollout_trie_hard_ce"
    assert "prepared_rollout_jsonl" not in trie_objective.config
    assert trie_objective.config["min_ul_valid_rollouts"] == 4
    assert parsed.stage2_ab.channel_b.fallback_loss_weight == pytest.approx(0.25)
    assert parsed.stage2_ab.channel_b.insertion_order == "fn_slot_shuffle"
    assert parsed.stage2_ab.channel_b.fp_policy.mode == "weak_positive_context"
    assert parsed.stage2_ab.channel_b.triage_posterior.num_rollouts == 4


def test_stage2_pipeline_accepts_schema_format_ce_with_stage2_trie_ce() -> None:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"] = _stage2_pipeline_with_channel_b_trie_ce()
    raw["stage2_ab"]["pipeline"]["objective"].append(
        {
            "name": "schema_format_ce",
            "enabled": True,
            "weight": 0.25,
            "channels": ["B"],
            "application": {"preset": "rollout_schema_format"},
            "config": {"schema_ce_weight": 1.0},
        }
    )

    prompts = ConfigLoader.resolve_prompts(raw)
    loaded = TrainingConfig.from_mapping(raw, prompts)

    assert [module.name for module in loaded.stage2_ab.pipeline.objective] == [
        "token_ce",
        "stage2_trie_ce",
        "schema_format_ce",
    ]
    schema_objective = loaded.stage2_ab.pipeline.objective[2]
    assert schema_objective.channels == ("B",)
    assert schema_objective.application["preset"] == "rollout_schema_format"
    assert schema_objective.config["schema_ce_weight"] == pytest.approx(1.0)


def test_stage2_pipeline_accepts_residual_set_correction_objective() -> None:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}
    raw["stage2_ab"]["channel_b"].pop("triage_posterior", None)

    prompts = ConfigLoader.resolve_prompts(raw)
    loaded = TrainingConfig.from_mapping(raw, prompts)

    objective = loaded.stage2_ab.pipeline.objective[1]
    assert objective.name == "residual_set_correction"
    assert objective.application["preset"] == "rollout_self_prefix"
    assert "prepared_rollout_jsonl" not in objective.config
    assert objective.config["base_seed"] == 17
    assert objective.config["expected_num_rollouts"] == 4
    assert objective.config["lambda_type"] == pytest.approx(1.0)
    assert objective.config["lambda_inner"] == pytest.approx(1.0)
    assert objective.config["lambda_ul_promoted"] == 0.5
    assert loaded.stage2_ab.channel_b.triage_posterior.num_rollouts == 2


def test_residual_set_expected_num_rollouts_does_not_own_channel_b_rollout_count() -> None:
    raw = _make_stage2_training_payload()
    residual_cfg = _residual_set_config()
    residual_cfg["expected_num_rollouts"] = 5
    residual_cfg["min_ul_valid_rollouts"] = 5
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": residual_cfg,
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}
    raw["stage2_ab"]["channel_b"].pop("triage_posterior", None)

    loaded = TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))

    assert loaded.stage2_ab.pipeline.objective[1].config["expected_num_rollouts"] == 5
    assert loaded.stage2_ab.channel_b.triage_posterior.num_rollouts == 2


@pytest.mark.parametrize(
    "key, bad_value, expected_msg",
    [
        ("expected_num_rollouts", True, r"expected_num_rollouts.*positive integer"),
        ("expected_num_rollouts", 1.5, r"expected_num_rollouts.*positive integer"),
        ("expected_num_rollouts", 0, r"expected_num_rollouts.*positive integer"),
        ("min_ul_valid_rollouts", False, r"min_ul_valid_rollouts.*positive integer"),
        ("min_ul_valid_rollouts", "2", r"min_ul_valid_rollouts.*positive integer"),
        ("min_ul_valid_rollouts", -1, r"min_ul_valid_rollouts.*positive integer"),
        ("lambda_type", -0.1, r"lambda_type.*nonnegative"),
        ("lambda_inner", float("inf"), r"lambda_inner.*finite"),
        ("fallback_loss_weight", float("nan"), r"fallback_loss_weight.*finite"),
        ("lambda_ul_promoted", -0.1, r"lambda_ul_promoted.*nonnegative"),
        ("label_conflict_weight", -0.1, r"label_conflict_weight.*nonnegative"),
        ("commit_iou_threshold", -0.01, r"commit_iou_threshold.*\[0, 1\]"),
        ("duplicate_burst_iou_threshold", 1.01, r"duplicate_burst_iou_threshold.*\[0, 1\]"),
        ("ul_cluster_iou_threshold", -0.01, r"ul_cluster_iou_threshold.*\[0, 1\]"),
        ("ul_gray_iou_low", 1.01, r"ul_gray_iou_low.*\[0, 1\]"),
        ("ul_consensus_ratio", 0.5, r"ul_consensus_ratio.*1\.0"),
    ],
)
def test_residual_set_rejects_invalid_v1_config_values(
    key: str,
    bad_value: object,
    expected_msg: str,
) -> None:
    config = _residual_set_config()
    config[key] = bad_value
    raw = _make_raw_with_residual_set_config(config)

    with pytest.raises((TypeError, ValueError), match=expected_msg):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


def test_residual_set_rejects_gray_iou_above_cluster_threshold() -> None:
    config = _residual_set_config()
    config["ul_gray_iou_low"] = 0.91
    config["ul_cluster_iou_threshold"] = 0.90
    raw = _make_raw_with_residual_set_config(config)

    with pytest.raises(ValueError, match=r"ul_gray_iou_low.*ul_cluster_iou_threshold"):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


def test_residual_set_rejects_legacy_channel_b_trie_double_supervision() -> None:
    raw = _make_stage2_training_payload()
    trie_cfg = _stage2_pipeline_with_channel_b_trie_ce()["objective"][1]
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        trie_cfg,
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}

    with pytest.raises(ValueError, match=r"aliases.*residual-state trie.*select exactly one"):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


def test_residual_set_allows_stage2_trie_ce_outside_channel_b() -> None:
    raw = _make_stage2_training_payload()
    trie_cfg = _stage2_pipeline_with_channel_b_trie_ce()["objective"][1]
    trie_cfg["channels"] = ["A"]
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        trie_cfg,
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}

    loaded = TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))

    assert loaded.stage2_ab.pipeline.objective[1].channels == ("A",)
    assert loaded.stage2_ab.pipeline.objective[2].name == "residual_set_correction"


def test_residual_set_outside_channel_b_allows_stage2_trie_ce_channel_b() -> None:
    raw = _make_stage2_training_payload()
    trie_cfg = _stage2_pipeline_with_channel_b_trie_ce()["objective"][1]
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        trie_cfg,
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}

    loaded = TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))

    assert loaded.stage2_ab.pipeline.objective[1].channels == ("B",)
    assert loaded.stage2_ab.pipeline.objective[2].channels == ("A",)


def test_residual_set_rejects_pseudo_positive_double_supervision() -> None:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": True}

    with pytest.raises(ValueError, match=r"residual trie.*pseudo_positive"):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


def test_residual_set_rejects_channel_b_token_ce_double_supervision() -> None:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A", "B"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}

    with pytest.raises(ValueError, match=r"residual trie.*token_ce"):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


@pytest.mark.parametrize(
    "removed_key, removed_value",
    [
        ("num_rollouts", 4),
        ("coord_span_policy", "bbox_tail_from_anchor"),
        ("coverage_strength", 0.0),
        ("ul_geometry", {"iou_min": 0.75}),
        ("artifact_policy", {"ul_clusters": "monitor_debug_smoke"}),
    ],
)
def test_residual_set_rejects_removed_config_keys(
    removed_key: str,
    removed_value: object,
) -> None:
    raw = _make_stage2_training_payload()
    config = _residual_set_config()
    config[removed_key] = removed_value
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": config,
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}

    with pytest.raises(ValueError, match=rf"residual_set_correction.*{removed_key}"):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


@pytest.mark.parametrize(
    "removed_name",
    [
        "loss_duplicate_burst_unlikelihood",
        "bbox_geo",
        "bbox_size_aux",
        "coord_reg",
        "coord_gate",
        "text_gate",
    ],
)
def test_stage2_pipeline_rejects_removed_live_objective_module_names(
    removed_name: str,
) -> None:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": removed_name,
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_only"},
            "config": {},
        },
    ]

    with pytest.raises(ValueError, match=removed_name):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


@pytest.mark.parametrize(
    "legacy_key, value",
    [
        ("support_weight", 1.0),
        ("balance_weight", 1.0),
        ("struct_weight", 1.0),
        ("desc_weight", 1.0),
        ("coord_hard_ce_weight", 1.0),
        ("eos_weight", 1.0),
        ("normalization", "token_mean"),
    ],
)
def test_stage2_pipeline_rejects_legacy_candidate_trie_config_keys(
    legacy_key: str,
    value: object,
) -> None:
    pipeline = _stage2_pipeline_with_channel_b_trie_ce()
    pipeline["objective"][1]["config"][legacy_key] = value
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "pipeline": pipeline,
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=rf"Unknown.*stage2_trie_ce.*{legacy_key}"):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_channel_b_double_token_supervision() -> None:
    pipeline = _stage2_pipeline_with_channel_b_trie_ce()
    pipeline["objective"][0]["channels"] = ["A", "B"]
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "pipeline": pipeline,
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"Channel-B.*token_ce.*stage2_trie_ce",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_removed_duplicate_burst_unlikelihood_objective() -> None:
    pipeline = _canonical_stage2_pipeline()
    pipeline["objective"].insert(
        1,
        {
            "name": "loss_duplicate_burst_unlikelihood",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_only"},
            "config": {},
        },
    )
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "pipeline": pipeline,
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"loss_duplicate_burst_unlikelihood",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_custom_coord_soft_ce_w1_surface() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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


def test_stage2_pipeline_rejects_unknown_module_config_keys() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_global_prefix_struct_ce_weight": 1.0,
                    "unknown_weight": 1.0,
                }
            ),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"Unknown stage2_ab\.pipeline\.objective\[0\]\.config keys.*unknown_weight",
    ):
        TrainingConfig.from_mapping(raw, prompts)


@pytest.mark.parametrize("legacy_module", ["bbox_geo", "bbox_size_aux", "coord_reg"])
def test_stage2_pipeline_rejects_removed_geometry_modules(legacy_module: str) -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
                "objective": [
                    *_canonical_stage2_pipeline()["objective"],
                    {
                        "name": legacy_module,
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "application": {"preset": "anchor_only"},
                        "config": {},
                    },
                ],
                "diagnostics": [],
            },
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=rf"stage2_ab\.pipeline\.objective\[1\]\.name.*{legacy_module}",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_pipeline_rejects_deprecated_struct_ce_weight() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
                    "rollout_fn_desc_weight": 1.0,
                    "rollout_global_prefix_struct_ce_weight": 1.0,
                    "struct_ce_weight": 0.1,
                }
            ),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.pipeline\.objective\[0\]\.config\.struct_ce_weight is deprecated and unsupported",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_ab_rejects_deprecated_decode_toggle() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "coord_decode_mode": "st",
            "pipeline": _canonical_stage2_pipeline(),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"Deprecated Stage-2 self-context knobs are unsupported.*stage2_ab\.coord_decode_mode",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_rollout_matching_rejects_deprecated_decode_toggle() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
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
            "coord_decode_mode": "st",
        },
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": _canonical_stage2_pipeline(),
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"rollout_matching\.coord_decode_mode is deprecated and unsupported",
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


def test_stage2_ab_rejects_authored_gradient_accumulation_with_effective_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_loader_runtime(monkeypatch, world_size=2)
    cfg = _make_stage2_training_config(
        {
            "per_device_train_batch_size": 2,
            "effective_batch_size": 8,
            "gradient_accumulation_steps": 3,
        }
    )

    with pytest.raises(ValueError, match=r"derived from training\.effective_batch_size"):
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
            default_objective=["token_ce"],
            default_diagnostics=[],
            trainer_variant="stage2_two_channel",
            config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
            run_name="smoke_ab_mixed_pipeline_explicit",
            seed=17,
            coord_soft_cfg=coord_soft_cfg,
        )


@pytest.mark.parametrize(
    ("pipeline_section", "exc_type", "match"),
    [
        (
            {"objective": ["not-a-mapping"], "diagnostics": []},
            TypeError,
            r"pipeline\.objective\[0\] must be a mapping module spec",
        ),
        (
            {"objective": [{"name": ""}], "diagnostics": []},
            ValueError,
            r"pipeline\.objective\[0\]\.name must be non-empty",
        ),
        (
            {"objective": [{"name": "token_ce"}], "diagnostics": [{}]},
            ValueError,
            r"pipeline\.diagnostics\[0\]\.name must be non-empty",
        ),
    ],
)
def test_stage2_build_pipeline_manifest_rejects_malformed_explicit_modules(
    pipeline_section,
    exc_type,
    match,
) -> None:
    from src.sft import _build_pipeline_manifest

    with pytest.raises(exc_type, match=match):
        _build_pipeline_manifest(
            {"pipeline": pipeline_section},
            default_objective=["token_ce"],
            default_diagnostics=[],
            trainer_variant="stage2_two_channel",
            config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
            run_name="smoke_ab_mixed_pipeline_explicit",
            seed=17,
            coord_soft_cfg=None,
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
            ),
            "diagnostics": (),
        }
    }

    manifest = _build_pipeline_manifest(
        cfg,
        default_objective=["token_ce"],
        default_diagnostics=[],
        trainer_variant="stage2_two_channel",
        config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
        run_name="smoke_manifest_sequence",
        seed=17,
        coord_soft_cfg=None,
    )

    assert [m["name"] for m in manifest["objective"]] == ["token_ce"]
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


def test_stage2_leaf_contract_accepts_live_smoke_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ConfigLoader._validate_stage2_leaf_contract(
        str(repo_root / "configs/stage2_two_channel/smoke/a_only.yaml")
    )


def test_stage2_leaf_contract_accepts_live_mixed_smoke_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ConfigLoader._validate_stage2_leaf_contract(
        str(repo_root / "configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml")
    )


def test_stage2_leaf_contract_accepts_live_ablation_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ConfigLoader._validate_stage2_leaf_contract(
        str(repo_root / "configs/stage2_two_channel/prod/a_only.yaml")
    )


@pytest.mark.parametrize(
    ("config_rel", "expected_ordering"),
    [
        ("smoke/a_only.yaml", "sorted"),
        ("smoke/ab_mixed_20steps.yaml", "sorted"),
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
    assert training["seed"] == 17

    if config_rel.startswith("smoke/"):
        data = cfg.data
        assert training["max_steps"] == 20
        assert training["eval_strategy"] == "no"
        assert training["save_strategy"] == "no"
        assert custom.train_sample_limit == 128
        assert custom.val_sample_limit == 8
        assert data["dataloader_num_workers"] == 0


def test_stage2_compact_full_a2_smoke_config_pins_unconstrained_fallback_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(
            repo_root
            / "configs"
            / "stage2_two_channel"
            / "smoke"
            / "compact_full_et_rmp_ce_ckpt3664_hf_1step.yaml"
        )
    )

    assert cfg.custom.trainer_variant == "stage2_two_channel"
    assert cfg.custom.detection_sequence_format == "compact_full"
    assert cfg.custom.object_ordering == "random"

    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.schedule.b_ratio == 1.0
    assert cfg.stage2_ab.channel_b.rollout_template_family == "compact_full"
    assert cfg.stage2_ab.channel_b.rollout_decode_policy == "unconstrained"
    assert (
        cfg.stage2_ab.channel_b.invalid_rollout_policy
        == "fallback_gt_fn_append_only"
    )
    assert cfg.stage2_ab.channel_b.fallback_loss_weight == 1.0
    assert cfg.stage2_ab.channel_b.assignment.strategy == "greedy_iou"
    assert cfg.stage2_ab.channel_b.triage_posterior.num_rollouts == 4

    assert cfg.rollout_matching.rollout_backend == "hf"
    assert cfg.rollout_matching.eval_rollout_backend == "hf"
    assert cfg.rollout_matching.eval_detection.enabled is True
    assert cfg.rollout_matching.eval_detection.materialize_artifacts is True

    assert cfg.training["effective_batch_size"] == 8
    assert "gradient_accumulation_steps" not in cfg.training
    assert cfg.training["max_steps"] == 1
    assert cfg.custom.train_sample_limit == 8
    assert cfg.custom.val_sample_limit == 8
    assert "checkpoint-3664" in str(cfg.model["adapters"][0])


def test_stage2_compact_full_residual_set_smoke_config_uses_v1_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "configs"
        / "stage2_two_channel"
        / "smoke"
        / "compact_full_residual_set_ckpt3664_hf_1step.yaml"
    )

    cfg = ConfigLoader.load_materialized_training_config(str(config_path))
    raw = ConfigLoader.load_yaml_with_extends(str(config_path))

    assert "checkpoint-3664" in str(cfg.model["adapters"][0])
    assert cfg.training["eval_strategy"] == "no"
    assert cfg.custom.val_sample_limit == 0
    assert cfg.rollout_matching.eval_monitor_dump.enabled is False
    assert cfg.rollout_matching.eval_detection.enabled is False
    assert cfg.rollout_matching.eval_detection.materialize_artifacts is False

    assert cfg.stage2_ab is not None
    objective_by_name = {
        module.name: module for module in cfg.stage2_ab.pipeline.objective
    }
    residual_config = objective_by_name["residual_set_correction"].config
    assert "prepared_rollout_jsonl" not in residual_config
    assert residual_config["expected_num_rollouts"] == 4
    assert residual_config["base_seed"] == 17
    assert residual_config["lambda_type"] == pytest.approx(1.0)
    assert residual_config["lambda_inner"] == pytest.approx(1.0)
    assert {
        "num_rollouts",
        "coord_span_policy",
        "coverage_strength",
        "ul_geometry",
        "artifact_policy",
    }.isdisjoint(residual_config)

    raw_objective = raw["stage2_ab"]["pipeline"]["objective"]
    raw_residual = next(
        item for item in raw_objective if item["name"] == "residual_set_correction"
    )
    assert {
        "num_rollouts",
        "coord_span_policy",
        "coverage_strength",
        "ul_geometry",
        "artifact_policy",
    }.isdisjoint(raw_residual["config"])


def test_stage2_compact_full_a2_gate2_smoke_config_keeps_compact_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(
            repo_root
            / "configs"
            / "stage2_two_channel"
            / "smoke"
            / "compact_full_et_rmp_ce_ckpt3664_hf_gate2_16sample.yaml"
        )
    )

    assert cfg.custom.trainer_variant == "stage2_two_channel"
    assert cfg.custom.detection_sequence_format == "compact_full"

    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.channel_b.rollout_template_family == "compact_full"
    assert cfg.stage2_ab.channel_b.rollout_decode_policy == "unconstrained"
    assert (
        cfg.stage2_ab.channel_b.invalid_rollout_policy
        == "fallback_gt_fn_append_only"
    )
    assert cfg.stage2_ab.channel_b.assignment.strategy == "greedy_iou"
    assert cfg.stage2_ab.channel_b.triage_posterior.num_rollouts == 4

    assert cfg.training["max_steps"] == 16
    assert cfg.training["eval_steps"] == 16
    assert cfg.custom.train_sample_limit == 16
    assert cfg.custom.val_sample_limit == 16


def test_stage2_decode4_train128_val64_gate_config_is_worldsize8_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "configs"
        / "stage2_two_channel"
        / "smoke"
        / "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_decode4_hf_gate.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(config_path))

    assert cfg.custom.trainer_variant == "stage2_two_channel"
    assert cfg.custom.detection_sequence_format == "compact_full"
    assert cfg.global_max_length == 20000
    assert cfg.custom.train_jsonl == "public_data/coco/views/coco80/full/train.jsonl"
    assert cfg.custom.train_sample_limit == 128
    assert cfg.custom.val_sample_limit == 64
    assert cfg.custom.object_ordering == "random"

    assert cfg.training["max_steps"] == 4
    assert cfg.training["eval_strategy"] == "steps"
    assert cfg.training["eval_steps"] == 2
    assert cfg.training["save_strategy"] == "no"
    assert cfg.training["per_device_train_batch_size"] == 1
    assert cfg.training["effective_batch_size"] == 16
    assert cfg.training["dataloader_drop_last"] is True

    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.schedule.b_ratio == pytest.approx(1.0)
    assert cfg.stage2_ab.channel_b.ddp_phase_timeout_s == pytest.approx(600.0)
    assert cfg.stage2_ab.channel_b.rollout_template_family == "compact_full"
    assert cfg.stage2_ab.channel_b.rollout_decode_policy == "unconstrained"
    assert (
        cfg.stage2_ab.channel_b.invalid_rollout_policy
        == "fallback_gt_fn_append_only"
    )
    assert cfg.stage2_ab.channel_b.triage_posterior.num_rollouts == 4
    assert cfg.stage2_ab.channel_b.triage_posterior.rollout_temperatures == pytest.approx(
        (0.0, 0.3, 0.5, 0.7)
    )
    assert cfg.stage2_ab.channel_b.ddp_phase_timeout_s == pytest.approx(600.0)

    assert cfg.rollout_matching.rollout_backend == "hf"
    assert cfg.rollout_matching.eval_rollout_backend == "hf"
    assert cfg.rollout_matching.channel_b_decode_batch_size == 4
    assert cfg.rollout_matching.eval_decode_batch_size == 4
    assert cfg.rollout_matching.eval_detection.enabled is True
    assert cfg.rollout_matching.eval_detection.materialize_artifacts is True

    _patch_loader_runtime(monkeypatch, world_size=8)
    args = ConfigLoader.build_train_arguments(cfg)
    assert args.kwargs["gradient_accumulation_steps"] == 2


@pytest.mark.parametrize(
    ("config_name", "expected_insertion_order"),
    [
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4.yaml",
            "tail_append",
        ),
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_online_residual_trie_sorted_zero_fp_lr1e5_decode4.yaml",
            "sorted",
        ),
    ],
)
def test_online_residual_trie_train128_val64_configs_use_live_rollouts(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected_insertion_order: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs" / "stage2_two_channel" / "smoke" / config_name)
    )

    assert cfg.custom.trainer_variant == "stage2_two_channel"
    assert cfg.custom.detection_sequence_format == "compact_full"
    assert cfg.custom.train_sample_limit == 128
    assert cfg.custom.val_sample_limit == 64
    assert cfg.training["max_steps"] == 32
    assert cfg.training["eval_steps"] == 16
    assert cfg.training["per_device_train_batch_size"] == 2
    assert cfg.training["effective_batch_size"] == 16
    assert cfg.training["learning_rate"] == pytest.approx(1.0e-5)
    assert cfg.training["aligner_lr"] == pytest.approx(2.5e-5)

    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.channel_b.rollout_template_family == "compact_full"
    assert cfg.stage2_ab.channel_b.rollout_decode_policy == "compact_grammar"
    assert cfg.stage2_ab.channel_b.insertion_order == expected_insertion_order
    assert cfg.stage2_ab.channel_b.pseudo_positive.enabled is False
    assert cfg.stage2_ab.channel_b.fp_policy.mode == "zero_loss_context"
    assert cfg.stage2_ab.channel_b.triage_posterior.num_rollouts == 4
    assert cfg.stage2_ab.channel_b.triage_posterior.rollout_temperatures == pytest.approx(
        (0.0, 0.3, 0.5, 0.7)
    )

    objective_by_name = {
        module.name: module for module in cfg.stage2_ab.pipeline.objective
    }
    trie_config = objective_by_name["stage2_trie_ce"].config
    assert "rollout_source" not in trie_config
    assert "prepared_rollout_jsonl" not in trie_config
    assert trie_config["expected_num_rollouts"] == 4
    assert trie_config["min_ul_valid_rollouts"] == 4
    assert "require_real_prepared_rollouts" not in trie_config

    assert cfg.rollout_matching.channel_b_decode_batch_size == 4
    assert cfg.rollout_matching.eval_decode_batch_size == 4
    assert cfg.rollout_matching.train_monitor_dump.enabled is True
    assert cfg.rollout_matching.eval_detection.materialize_artifacts is True

    _patch_loader_runtime(monkeypatch, world_size=4)
    args = ConfigLoader.build_train_arguments(cfg)
    assert args.kwargs["gradient_accumulation_steps"] == 2


@pytest.mark.parametrize(
    "config_name, expected_insertion_order, expected_train_limit, expected_max_steps",
    [
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_1step_stage2_trie_tail_append_zero_fp.yaml",
            "tail_append",
            8,
            1,
        ),
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_tail_append_zero_fp.yaml",
            "tail_append",
            8,
            64,
        ),
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml",
            "fn_slot_shuffle",
            8,
            64,
        ),
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_sorted_zero_fp.yaml",
            "sorted",
            8,
            64,
        ),
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_tail_append_zero_fp.yaml",
            "tail_append",
            64,
            128,
        ),
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml",
            "fn_slot_shuffle",
            64,
            128,
        ),
        (
            "compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_sorted_zero_fp.yaml",
            "sorted",
            64,
            128,
        ),
    ],
)
def test_stage2_trie_ce_coco80_overfit_smoke_configs_are_training_only(
    config_name: str,
    expected_insertion_order: str,
    expected_train_limit: int,
    expected_max_steps: int,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs" / "stage2_two_channel" / "smoke" / config_name)
    )

    assert cfg.custom.trainer_variant == "stage2_two_channel"
    assert cfg.custom.detection_sequence_format == "compact_full"
    assert cfg.custom.train_jsonl == "public_data/coco/views/coco80/full/train.jsonl"
    assert cfg.custom.train_sample_limit == expected_train_limit
    assert cfg.custom.val_sample_limit == 0
    assert cfg.custom.object_ordering == "random"

    assert cfg.training["max_steps"] == expected_max_steps
    assert cfg.training["eval_strategy"] == "no"
    assert cfg.training["save_strategy"] == "no"
    assert "coco80_view_stage2_trie_ce_" in str(cfg.training["output_dir"])
    if expected_max_steps == 1:
        assert "startup" in str(cfg.training["output_dir"])
    elif expected_train_limit == 8:
        assert "overfit_probe" in str(cfg.training["output_dir"])
    else:
        assert "preflight" in str(cfg.training["output_dir"])
    assert expected_insertion_order in str(cfg.training["run_name"])

    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.channel_b.rollout_template_family == "compact_full"
    assert cfg.stage2_ab.channel_b.rollout_decode_policy == "unconstrained"
    assert (
        cfg.stage2_ab.channel_b.invalid_rollout_policy
        == "fallback_gt_fn_append_only"
    )
    assert cfg.stage2_ab.channel_b.assignment.strategy == "greedy_iou"
    assert cfg.stage2_ab.channel_b.fallback_loss_weight == pytest.approx(0.25)
    assert cfg.stage2_ab.channel_b.insertion_order == expected_insertion_order
    assert cfg.stage2_ab.channel_b.fp_policy.mode == "zero_loss_context"
    assert cfg.stage2_ab.channel_b.triage_posterior.num_rollouts == 4

    objective_by_name = {
        module.name: module for module in cfg.stage2_ab.pipeline.objective
    }
    assert [module.name for module in cfg.stage2_ab.pipeline.objective] == [
        "token_ce",
        "stage2_trie_ce",
    ]
    assert objective_by_name["token_ce"].enabled is True
    assert objective_by_name["token_ce"].channels == ("A",)
    assert objective_by_name["stage2_trie_ce"].enabled is True
    assert objective_by_name["stage2_trie_ce"].channels == ("B",)
    assert (
        objective_by_name["stage2_trie_ce"].application["preset"]
        == "rollout_trie_hard_ce"
    )
    trie_config = objective_by_name["stage2_trie_ce"].config
    assert "prepared_rollout_jsonl" not in trie_config
    assert trie_config["expected_num_rollouts"] == 4
    assert trie_config["min_ul_valid_rollouts"] == 4
    assert {
        "support_weight",
        "balance_weight",
        "struct_weight",
        "desc_weight",
        "coord_hard_ce_weight",
        "eos_weight",
        "normalization",
    }.isdisjoint(trie_config)
    assert cfg.stage2_ab.pipeline.diagnostics == ()

    assert cfg.rollout_matching.rollout_backend == "hf"
    assert cfg.rollout_matching.eval_rollout_backend == "hf"
    assert cfg.rollout_matching.train_monitor_dump.enabled is True
    assert cfg.rollout_matching.eval_monitor_dump.enabled is False
    assert cfg.rollout_matching.eval_detection.enabled is False
    assert cfg.rollout_matching.eval_detection.materialize_artifacts is False
    assert "lvis" not in cfg.custom.train_jsonl.lower()


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
