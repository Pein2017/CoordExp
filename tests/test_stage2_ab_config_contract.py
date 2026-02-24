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
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
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
        "rollout_matching": {"rollout_backend": "hf"},
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": {
                "objective": [
                    {
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "config": {},
                    }
                ],
                "diagnostics": [],
            },
            "channel_b": {"drop_invalid_struct_ce_multiplier": 2.0},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.drop_invalid_struct_ce_multiplier",
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
        "rollout_matching": {"rollout_backend": "hf"},
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": {
                "objective": [
                    {
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "config": {},
                    }
                ],
                "diagnostics": [],
            },
            "channel_b": {},
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=r"custom\.coord_soft_ce_w1"):
        TrainingConfig.from_mapping(raw, prompts)


def test_rollout_pipeline_rejects_custom_coord_soft_ce_w1_surface() -> None:
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
            "pipeline": {
                "objective": [
                    {
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "config": {},
                    }
                ],
                "diagnostics": [],
            },
        },
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=r"custom\.coord_soft_ce_w1"):
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



def test_rollout_decode_batch_size_overrides_eval_batch_size_when_mismatched():
    from src.sft import _apply_rollout_decode_batch_size_override

    train_args = types.SimpleNamespace(
        trainer_variant="stage2_two_channel",
        training_args=types.SimpleNamespace(per_device_eval_batch_size=1),
    )
    training_config = types.SimpleNamespace(rollout_matching={"decode_batch_size": 4})

    resolved = _apply_rollout_decode_batch_size_override(
        train_args=train_args,
        training_config=training_config,
    )

    assert resolved == 4
    assert train_args.training_args.per_device_eval_batch_size == 4
    assert train_args.per_device_eval_batch_size == 4


def test_default_pipeline_manifest_resolution_and_checksum_golden():
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

    manifest = _build_pipeline_manifest(
        cfg,
        default_objective=["token_ce", "bbox_geo", "coord_reg"],
        default_diagnostics=["coord_diag"],
        trainer_variant="stage2_two_channel",
        config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
        run_name="smoke_ab_mixed_pipeline_explicit",
        seed=17,
        coord_soft_cfg=coord_soft_cfg,
    )

    assert [m["name"] for m in manifest["objective"]] == [
        "token_ce",
        "bbox_geo",
        "coord_reg",
    ]
    assert [m["name"] for m in manifest["diagnostics"]] == ["coord_diag"]

    token_ce_cfg = manifest["objective"][0]["config"]
    bbox_geo_cfg = manifest["objective"][1]["config"]
    coord_reg_cfg = manifest["objective"][2]["config"]

    assert token_ce_cfg["desc_ce_weight"] == pytest.approx(0.7)
    assert token_ce_cfg["rollout_fn_desc_weight"] == pytest.approx(0.7)
    assert token_ce_cfg["rollout_matched_prefix_struct_weight"] == pytest.approx(1.0)

    assert bbox_geo_cfg["smoothl1_weight"] == pytest.approx(2.0)
    assert bbox_geo_cfg["ciou_weight"] == pytest.approx(0.5)

    assert coord_reg_cfg["coord_gate_weight"] == pytest.approx(1.0)
    assert coord_reg_cfg["text_gate_weight"] == pytest.approx(0.2)
    assert coord_reg_cfg["soft_ce_weight"] == pytest.approx(0.3)
    assert coord_reg_cfg["w1_weight"] == pytest.approx(0.4)
    assert coord_reg_cfg["temperature"] == pytest.approx(0.9)
    assert coord_reg_cfg["target_sigma"] == pytest.approx(1.7)
    assert coord_reg_cfg["target_truncate"] == 8
    assert manifest["extra"]["coord_ctx_embed_mode"] == "soft"
    assert manifest["extra"]["coord_decode_mode"] == "exp"

    # Checksum is full SHA256 and must be independent of run context.
    assert len(manifest["checksum"]) == 64

    manifest_same_pipeline_other_run = _build_pipeline_manifest(
        cfg,
        default_objective=["token_ce", "bbox_geo", "coord_reg"],
        default_diagnostics=["coord_diag"],
        trainer_variant="stage2_two_channel",
        config_path="configs/stage2_two_channel/prod/ab_mixed.yaml",
        run_name="different_run_name",
        seed=999,
        coord_soft_cfg=coord_soft_cfg,
    )
    assert (
        manifest_same_pipeline_other_run["checksum"] == manifest["checksum"]
    ), "pipeline checksum must be run-context invariant"


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
        default_objective=["token_ce", "bbox_geo", "coord_reg"],
        default_diagnostics=["coord_diag"],
        trainer_variant="stage2_two_channel",
        config_path="configs/stage2_two_channel/smoke/ab_mixed_pipeline_explicit.yaml",
        run_name="smoke_manifest_sequence",
        seed=17,
        coord_soft_cfg=None,
    )

    assert [m["name"] for m in manifest["objective"]] == ["token_ce"]
    assert manifest["diagnostics"] == []
