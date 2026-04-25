from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import PromptOverrides, TrainingConfig


def _base_training_payload() -> dict:
    # Minimal payload that passes TrainingConfig.from_mapping.
    return {
        "template": {"truncation_strategy": "raise"},
        "custom": {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
        },
    }


@pytest.mark.parametrize(
    "section",
    [
        "model",
        "quantization",
        "template",
        "data",
        "tuner",
        "training",
        "rlhf",
    ],
)
def test_unknown_key_fails_fast_with_dotted_path(section: str):
    payload = _base_training_payload()

    if section == "template":
        payload[section] = {"truncation_strategy": "raise", "unknown_key": 1}
    else:
        payload[section] = {"unknown_key": 1}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    # Dotted-path reporting is required for strict parsing.
    assert f"{section}.unknown_key" in str(exc.value)


def test_training_internal_packing_keys_are_allowed():
    payload = _base_training_payload()
    payload["training"] = {
        "output_root": "./output",
        "logging_root": "./tb",
        "artifact_subdir": "stage1/example",
        "packing": True,
        "packing_mode": "static",
        "packing_buffer": 128,
        "packing_min_fill_ratio": 0.5,
        "packing_wait_timeout_s": 7200,
        "packing_length_cache_persist_every": 2048,
        "packing_length_precompute_workers": 8,
        "effective_batch_size": 8,
        "save_delay_steps": 10,
        "save_last_epoch": True,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.training["packing"] is True
    assert cfg.training["packing_mode"] == "static"
    assert cfg.training["packing_buffer"] == 128
    assert cfg.training["packing_wait_timeout_s"] == 7200
    assert cfg.training["packing_length_cache_persist_every"] == 2048
    assert cfg.training["packing_length_precompute_workers"] == 8
    assert cfg.training["output_root"] == "./output"
    assert cfg.training["logging_root"] == "./tb"
    assert cfg.training["artifact_subdir"] == "stage1/example"


def test_experiment_unknown_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["experiment"] = {
        "purpose": "Validate authored experiment intent",
        "unknown_key": 1,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "experiment.unknown_key" in str(exc.value)


def test_experiment_section_parses_and_serializes() -> None:
    payload = _base_training_payload()
    payload["experiment"] = {
        "title": "Stage-2 smoke",
        "purpose": "Validate the new manifest path.",
        "hypothesis": "Authored metadata survives strict loading.",
        "key_deviations": ["Uses a concise run name."],
        "runtime_settings": ["Runs as a smoke profile."],
        "comments": ["Synthetic test payload."],
        "tags": ["smoke", "metadata"],
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.experiment.title == "Stage-2 smoke"
    assert cfg.experiment.purpose == "Validate the new manifest path."
    assert cfg.experiment.key_deviations == ("Uses a concise run name.",)
    assert cfg.experiment.runtime_settings == ("Runs as a smoke profile.",)
    assert cfg.experiment.tags == ("smoke", "metadata")
    assert cfg.experiment.to_mapping()["comments"] == ["Synthetic test payload."]


def test_benchmark_section_parses_and_serializes() -> None:
    payload = _base_training_payload()
    payload["benchmark"] = {
        "group_id": "stage1-set-continuation-a",
        "control_group_id": "stage1-baseline",
        "intended_variable": "candidate-set construction",
        "comparability_label": "accuracy-comparable",
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.benchmark.group_id == "stage1-set-continuation-a"
    assert cfg.benchmark.control_group_id == "stage1-baseline"
    assert cfg.benchmark.intended_variable == "candidate-set construction"
    assert cfg.benchmark.comparability_label == "accuracy-comparable"


def test_benchmark_unknown_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["benchmark"] = {
        "group_id": "stage1-set-continuation-a",
        "unknown_key": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "benchmark.unknown_key" in str(exc.value)


def test_benchmark_text_fields_reject_non_strings() -> None:
    payload = _base_training_payload()
    payload["benchmark"] = {"group_id": 0}

    with pytest.raises(TypeError, match="benchmark.group_id"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("output_root", 42),
        ("logging_root", True),
        ("artifact_subdir", ["stage1", "bad"]),
    ],
)
def test_materialized_training_artifact_paths_reject_non_string_components(
    key: str,
    bad_value: object,
) -> None:
    config = {
        "training": {
            "output_root": "./output",
            "logging_root": "./tb",
            "artifact_subdir": "stage1/example",
        }
    }
    config["training"][key] = bad_value

    with pytest.raises(TypeError, match=rf"training\.{key} must be authored as a string"):
        ConfigLoader._materialize_training_artifact_paths(config)


def test_disabled_custom_fusion_config_fails_fast() -> None:
    payload = _base_training_payload()
    payload["custom"]["fusion_config"] = "toy/fusion.yaml"

    with pytest.raises(ValueError, match=r"custom\.fusion_config is temporarily disabled"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_global_and_stage_bases_no_longer_hide_dataset_or_prompt_identity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_base = ConfigLoader.load_yaml_with_extends(str(repo_root / "configs/base.yaml"))
    raw_stage1 = ConfigLoader.load_yaml_with_extends(
        str(repo_root / "configs/stage1/sft_base.yaml")
    )
    raw_stage2 = ConfigLoader.load_yaml_with_extends(
        str(repo_root / "configs/stage2_two_channel/base.yaml")
    )

    for payload in (raw_base, raw_stage1, raw_stage2):
        custom = payload.get("custom", {}) or {}
        assert "train_jsonl" not in custom
        assert "val_jsonl" not in custom
        assert "object_field_order" not in custom
        assert "object_ordering" not in custom
        extra = custom.get("extra", {}) or {}
        assert "prompt_variant" not in extra


def test_loader_rejects_overlapping_list_owners_across_reusable_parents(
    tmp_path: Path,
) -> None:
    (tmp_path / "base_a.yaml").write_text(
        "training:\n  report_to: [tensorboard]\n", encoding="utf-8"
    )
    (tmp_path / "base_b.yaml").write_text(
        "training:\n  report_to: [wandb]\n", encoding="utf-8"
    )
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text(
        "extends:\n"
        "  - ./base_a.yaml\n"
        "  - ./base_b.yaml\n"
        "training:\n"
        "  run_name: overlap\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"training\.report_to"):
        ConfigLoader.load_yaml_with_extends(str(leaf))


def test_loader_allows_child_branch_to_replace_parent_list_bundle(
    tmp_path: Path,
) -> None:
    (tmp_path / "base.yaml").write_text(
        "training:\n  report_to: [tensorboard]\n", encoding="utf-8"
    )
    (tmp_path / "mid.yaml").write_text(
        "extends: ./base.yaml\ntraining:\n  report_to: [wandb]\n",
        encoding="utf-8",
    )
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("extends: ./mid.yaml\n", encoding="utf-8")

    merged = ConfigLoader.load_yaml_with_extends(str(leaf))
    assert merged["training"]["report_to"] == ["wandb"]


def test_custom_eval_detection_lvis_metrics_are_accepted() -> None:
    payload = _base_training_payload()
    payload["custom"]["eval_detection"] = {
        "enabled": True,
        "metrics": "lvis",
        "lvis_max_dets": 300,
        "batch_size": 1,
        "max_new_tokens": 512,
        "lvis_annotations_json": "public_data/lvis/raw/annotations/lvis_v1_val.json",
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.custom.eval_detection.enabled is True
    assert cfg.custom.eval_detection.metrics == "lvis"
    assert cfg.custom.eval_detection.lvis_max_dets == 300
    assert (
        cfg.custom.eval_detection.lvis_annotations_json
        == "public_data/lvis/raw/annotations/lvis_v1_val.json"
    )


def test_custom_bbox_size_aux_unknown_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_size_aux"] = {
        "enabled": True,
        "unknown_flag": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "bbox_size_aux.unknown_flag" in str(exc.value)


def test_custom_bbox_size_aux_requires_explicit_keys() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_size_aux"] = {
        "enabled": True,
        "log_wh_weight": 0.05,
    }

    with pytest.raises(ValueError, match=r"bbox_size_aux requires explicit keys"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_custom_bbox_geo_unknown_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_geo"] = {
        "enabled": True,
        "unknown_flag": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "bbox_geo.unknown_flag" in str(exc.value)


def test_custom_bbox_geo_requires_explicit_keys() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_geo"] = {
        "enabled": True,
        "ciou_weight": 1.0,
    }

    with pytest.raises(ValueError, match=r"bbox_geo requires explicit keys"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_custom_bbox_geo_accepts_center_size_keys() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_geo"] = {
        "enabled": True,
        "smoothl1_weight": 0.5,
        "ciou_weight": 0.25,
        "parameterization": "center_size",
        "center_weight": 1.0,
        "size_weight": 0.25,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.custom.bbox_geo.parameterization == "center_size"
    assert cfg.custom.bbox_geo.center_weight == pytest.approx(1.0)
    assert cfg.custom.bbox_geo.size_weight == pytest.approx(0.25)


def test_custom_bbox_geo_legacy_surface_defaults_to_xyxy_parameterization() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_geo"] = {
        "enabled": True,
        "smoothl1_weight": 0.0,
        "ciou_weight": 1.0,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.custom.bbox_geo.parameterization == "xyxy"
    assert cfg.custom.bbox_geo.center_weight == pytest.approx(1.0)
    assert cfg.custom.bbox_geo.size_weight == pytest.approx(1.0)


def test_custom_bbox_geo_center_size_requires_nonzero_center_or_size_weight() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_geo"] = {
        "enabled": True,
        "smoothl1_weight": 0.5,
        "ciou_weight": 0.25,
        "parameterization": "center_size",
        "center_weight": 0.0,
        "size_weight": 0.0,
    }

    with pytest.raises(
        ValueError,
        match=r"bbox_geo\.parameterization=center_size requires center_weight > 0 or size_weight > 0",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_custom_coord_soft_ce_w1_unknown_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "unknown_flag": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "coord_soft_ce_w1.unknown_flag" in str(exc.value)


def test_cxcy_logw_logh_profile_accepts_pure_ce_with_positive_gates() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcy_logw_logh"
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
        "temperature": 1.0,
        "target_sigma": 2.0,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.custom.bbox_format == "cxcy_logw_logh"
    assert cfg.custom.coord_soft_ce_w1.text_gate_weight == 1.0


def test_cxcy_logw_logh_profile_rejects_non_pure_soft_ce() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcy_logw_logh"
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.5,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
    }

    with pytest.raises(ValueError, match=r"soft_ce_weight = 0"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_cxcy_logw_logh_profile_rejects_coord_tokens_disabled() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcy_logw_logh"
    payload["custom"]["coord_tokens"] = {
        "enabled": False,
        "skip_bbox_norm": True,
    }
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
        "temperature": 1.0,
        "target_sigma": 2.0,
    }

    with pytest.raises(
        ValueError,
        match=r"requires custom\.coord_tokens\.enabled=true",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_cxcy_logw_logh_profile_rejects_skip_bbox_norm_false() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcy_logw_logh"
    payload["custom"]["coord_tokens"] = {
        "enabled": True,
        "skip_bbox_norm": False,
    }
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
        "temperature": 1.0,
        "target_sigma": 2.0,
    }

    with pytest.raises(
        ValueError,
        match=r"custom\.coord_tokens\.skip_bbox_norm must be true",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_xyxy_raw_text_profile_accepts_pure_ce_norm1000_training_surface() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "xyxy"
    payload["custom"]["coord_tokens"] = {
        "enabled": False,
        "skip_bbox_norm": True,
    }
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": False,
    }
    payload["custom"]["train_jsonl"] = "public_data/coco/demo/train.norm.jsonl"
    payload["custom"]["val_jsonl"] = "public_data/coco/demo/val.norm.jsonl"

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.custom.coord_tokens.enabled is False
    assert cfg.custom.coord_tokens.skip_bbox_norm is True
    assert cfg.custom.coord_soft_ce_w1.enabled is False
    assert str(cfg.custom.train_jsonl).endswith("train.norm.jsonl")


def test_cxcywh_profile_accepts_pure_ce_with_positive_gates() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcywh"
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
        "temperature": 1.0,
        "target_sigma": 2.0,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.custom.bbox_format == "cxcywh"
    assert cfg.custom.coord_soft_ce_w1.text_gate_weight == 1.0


def test_cxcywh_profile_rejects_non_pure_soft_ce() -> None:
    payload = _base_training_payload()
    payload["custom"]["bbox_format"] = "cxcywh"
    payload["custom"]["coord_soft_ce_w1"] = {
        "enabled": True,
        "ce_weight": 1.0,
        "soft_ce_weight": 0.5,
        "w1_weight": 0.0,
        "gate_weight": 1.0,
        "text_gate_weight": 1.0,
    }

    with pytest.raises(ValueError, match=r"soft_ce_weight = 0"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_training_encoded_sample_cache_keys_are_allowed_and_normalized() -> None:
    payload = _base_training_payload()
    payload["training"] = {
        "encoded_sample_cache": {
            "enabled": True,
            "wait_timeout_s": 42,
        }
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    cache_cfg = cfg.training["encoded_sample_cache"]
    assert cache_cfg["enabled"] is True
    assert cache_cfg["ineligible_policy"] == "error"
    assert cache_cfg["wait_timeout_s"] == 42


def test_training_encoded_sample_cache_unknown_nested_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["training"] = {
        "encoded_sample_cache": {
            "enabled": True,
            "unknown_flag": True,
        }
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "training.encoded_sample_cache.unknown_flag" in str(exc.value)


def test_training_static_packing_cache_keys_are_allowed_and_normalized() -> None:
    payload = _base_training_payload()
    payload["training"] = {
        "static_packing_cache": {
            "root_dir": "/tmp/static-packing-cache",
        }
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    cache_cfg = cfg.training["static_packing_cache"]
    assert cache_cfg["root_dir"] == "/tmp/static-packing-cache"


def test_training_static_packing_cache_unknown_nested_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["training"] = {
        "static_packing_cache": {
            "unknown_flag": True,
        }
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "training.static_packing_cache.unknown_flag" in str(exc.value)


def test_training_removed_packing_exact_key_fails_fast():
    payload = _base_training_payload()
    payload["training"] = {
        "packing": True,
        "packing_mode": "static",
        "packing_require_exact_effective_batch": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "training.packing_require_exact_effective_batch" in str(exc.value)


def test_rollout_eval_detection_and_eval_prompt_variant_keys_are_accepted():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_rollout_backend": "vllm",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
        "eval_prompt_variant": "coco_80",
        "eval_detection": {
            "enabled": True,
            "metrics": "coco",
            "score_mode": "constant",
            "constant_score": 1.0,
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
        },
        "vllm": {
            "mode": "server",
            "max_model_len": 4096,
            "server": {
                "servers": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "group_port": 51216,
                    }
                ]
            },
        },
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.rollout_matching is not None
    assert cfg.rollout_matching.eval_rollout_backend == "vllm"
    assert cfg.rollout_matching.eval_prompt_variant == "coco_80"
    assert cfg.rollout_matching.eval_detection is not None
    assert cfg.rollout_matching.eval_detection.enabled is True


def test_rollout_eval_detection_lvis_metrics_are_accepted():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
        "eval_detection": {
            "enabled": True,
            "metrics": "lvis",
            "lvis_max_dets": 300,
            "score_mode": "constant",
            "constant_score": 1.0,
            "pred_score_source": "eval_rollout_constant",
            "pred_score_version": 2,
        },
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.rollout_matching is not None
    assert cfg.rollout_matching.eval_detection is not None
    assert cfg.rollout_matching.eval_detection.metrics == "lvis"
    assert cfg.rollout_matching.eval_detection.lvis_max_dets == 300


def test_rollout_server_base_url_rejects_0_0_0_0() -> None:
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "vllm",
        "eval_rollout_backend": "vllm",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
        "vllm": {
            "mode": "server",
            "max_model_len": 4096,
            "server": {
                "servers": [
                    {
                        "base_url": "http://0.0.0.0:8000",
                        "group_port": 51216,
                    }
                ]
            },
        },
    }

    with pytest.raises(ValueError, match=r"must not use 0\.0\.0\.0"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_rollout_eval_detection_defaults_to_enabled_coco_when_omitted():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.rollout_matching is not None
    assert cfg.rollout_matching.eval_detection is not None
    assert cfg.rollout_matching.eval_detection.enabled is True
    assert cfg.rollout_matching.eval_detection.metrics == "coco"


def test_rollout_eval_rollout_backend_hf_is_accepted() -> None:
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_rollout_backend": "hf",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.rollout_matching is not None
    assert cfg.rollout_matching.rollout_backend == "hf"
    assert cfg.rollout_matching.eval_rollout_backend == "hf"


def test_rollout_eval_rollout_backend_null_inherits() -> None:
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_rollout_backend": None,
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "rollout_matching.eval_rollout_backend" in msg
    assert "must be one of" in msg


def test_rollout_eval_rollout_backend_invalid_value_fails_fast() -> None:
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_rollout_backend": "bogus",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "rollout_matching.eval_rollout_backend" in msg
    assert "vllm" in msg.lower()


def test_training_packing_length_deprecated_fails_fast():
    payload = _base_training_payload()
    payload["training"] = {"packing": True, "packing_length": 128}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "training.packing_length" in msg
    assert "deprecated" in msg.lower()


def test_unknown_nested_rollout_key_fails_before_trainer_init():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "vllm": {
            "mode": "server",
            "server": {
                "servers": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "group_port": 51216,
                        "unknown_flag": True,
                    }
                ]
            },
        },
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.vllm.server.servers[0].unknown_flag" in str(exc.value)


def test_unknown_rollout_decoding_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "decoding": {"unknown": True},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "rollout_matching.decoding.unknown" in msg
    assert "Migration guidance" in msg


def test_unknown_rollout_monitor_dump_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "monitor_dump": {"unknown": True},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.monitor_dump.unknown" in str(exc.value)


def test_unknown_rollout_train_monitor_dump_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "train_monitor_dump": {"unknown": True},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.train_monitor_dump.unknown" in str(exc.value)


def test_rollout_train_monitor_every_channel_b_steps_is_accepted():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_rollout_backend": "vllm",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
        "train_monitor_dump": {
            "enabled": True,
            "every_channel_b_steps": 3,
        },
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.rollout_matching is not None
    assert cfg.rollout_matching.train_monitor_dump is not None
    assert cfg.rollout_matching.train_monitor_dump.every_channel_b_steps == 3


@pytest.mark.parametrize("value", [0, -3, "x"])
def test_rollout_train_monitor_every_channel_b_steps_invalid_values_fail_fast(value):
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_rollout_backend": "vllm",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
        "train_monitor_dump": {
            "enabled": True,
            "every_channel_b_steps": value,
        },
    }

    with pytest.raises((TypeError, ValueError)) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.train_monitor_dump.every_channel_b_steps" in str(exc.value)


@pytest.mark.parametrize("value", [0, -3, "x"])
def test_rollout_train_monitor_every_steps_invalid_values_fail_fast(value):
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_rollout_backend": "vllm",
        "channel_b_decode_batch_size": 2,
        "eval_decode_batch_size": 2,
        "train_monitor_dump": {
            "enabled": True,
            "every_steps": value,
        },
    }

    with pytest.raises((TypeError, ValueError)) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.train_monitor_dump.every_steps" in str(exc.value)


def test_unknown_rollout_eval_monitor_dump_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "eval_monitor_dump": {"unknown": True},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.eval_monitor_dump.unknown" in str(exc.value)


def test_unknown_rollout_vllm_sync_key_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "vllm": {"sync": {"unknown": True}},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.vllm.sync.unknown" in str(exc.value)


def test_unknown_stage2_ab_schedule_key_fails_fast():
    payload = _base_training_payload()
    payload["stage2_ab"] = {"schedule": {"b_ratio": 0.5, "unknown_flag": 1}}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "stage2_ab.schedule" in msg
    assert "unknown_flag" in msg


def test_unknown_stage2_ab_key_fails_fast():
    payload = _base_training_payload()
    payload["stage2_ab"] = {
        "schedule": {"b_ratio": 0.5},
        "pipeline": {
            "objective": _canonical_stage2_two_channel_objective(),
            "diagnostics": [],
        },
        "unknown_top": 1,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "stage2_ab.unknown_top" in msg
    assert "Migration guidance" in msg


def test_unknown_stage2_ab_channel_b_duplicate_control_key_fails_fast() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["channel_b"] = {
        "duplicate_control": {
            "iou_threshold": 0.9,
            "unexpected": 1,
        }
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "stage2_ab.channel_b.duplicate_control.unexpected" in str(exc.value)


def test_legacy_stage2_ab_channel_b_duplicate_iou_threshold_fails_fast() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["channel_b"] = {"duplicate_iou_threshold": 0.9}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "stage2_ab.channel_b.duplicate_iou_threshold has been removed" in str(
        exc.value
    )


def test_legacy_rollout_server_paired_list_shape_fails_fast():
    payload = _base_training_payload()
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "vllm": {
            "mode": "server",
            "server": {
                "base_url": "http://127.0.0.1:8000",
                "group_port": 51216,
            },
        },
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "rollout_matching.vllm.server.base_url/group_port" in str(exc.value)


def test_custom_visual_kd_unknown_section_key_fails():
    payload = _base_training_payload()
    payload["custom"]["visual_kd"] = {
        "enabled": True,
        "vit": {"enabled": True, "weight": 1.0},
        "extra_flag": True,
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.visual_kd.extra_flag" in str(exc.value)


def test_custom_visual_kd_target_unknown_key_fails():
    payload = _base_training_payload()
    payload["custom"]["visual_kd"] = {
        "enabled": True,
        "vit": {"enabled": True, "weight": 1.0, "unknown": 5},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.visual_kd.vit.unknown" in str(exc.value)


def test_custom_visual_kd_target_unknown_key_fails_even_when_disabled():
    payload = _base_training_payload()
    payload["custom"]["visual_kd"] = {
        "enabled": False,
        "vit": {"unknown": 5},
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.visual_kd.vit.unknown" in str(exc.value)


def test_top_level_extra_presence_fails_fast():
    payload = _base_training_payload()
    payload["extra"] = {}

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "Top-level extra:" in str(exc.value)


def test_custom_object_field_order_is_required():
    payload = _base_training_payload()
    payload["custom"].pop("object_field_order", None)

    with pytest.raises(ValueError, match="custom.object_field_order must be provided"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_unknown_top_level_key_fails_fast() -> None:
    payload = _base_training_payload()
    payload["bogus_top"] = 1

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    msg = str(exc.value)
    assert "Unknown top-level config keys" in msg
    assert "bogus_top" in msg
    assert "Migration guidance" in msg


def _base_stage2_two_channel_payload() -> dict:
    payload = _base_training_payload()
    payload["custom"]["trainer_variant"] = "stage2_two_channel"
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "channel_b_decode_batch_size": 1,
        "eval_decode_batch_size": 1,
    }
    payload["stage2_ab"] = {
        "schedule": {"b_ratio": 0.5},
        "pipeline": {
            "objective": _canonical_stage2_two_channel_objective(),
            "diagnostics": [],
        },
        "channel_b": {},
    }
    return payload


def _base_stage2_rollout_aligned_payload() -> dict:
    payload = _base_training_payload()
    payload["custom"]["trainer_variant"] = "stage2_rollout_aligned"
    payload["rollout_matching"] = {
        "rollout_backend": "hf",
        "channel_b_decode_batch_size": 1,
        "eval_decode_batch_size": 1,
        "pipeline": {"objective": [_pipeline_token_ce_spec()], "diagnostics": []},
    }
    return payload


def _pipeline_token_ce_spec(
    *, channels: list[str] | None = None, config: dict | None = None
) -> dict:
    token_ce_cfg = {
        "desc_ce_weight": 1.0,
        "rollout_fn_desc_weight": 1.0,
        "rollout_global_prefix_struct_ce_weight": 1.0,
    }
    if isinstance(config, dict):
        token_ce_cfg.update(dict(config))
    return {
        "name": "token_ce",
        "enabled": True,
        "weight": 1.0,
        "channels": list(channels) if channels is not None else ["A", "B"],
        "application": {"preset": "anchor_text_only"},
        "config": token_ce_cfg,
    }


def _pipeline_loss_duplicate_burst_unlikelihood_spec(
    *, channels: list[str] | None = None, config: dict | None = None
) -> dict:
    return {
        "name": "loss_duplicate_burst_unlikelihood",
        "enabled": True,
        "weight": 1.0,
        "channels": list(channels) if channels is not None else ["B"],
        "application": {"preset": "rollout_only"},
        "config": dict(config or {}),
    }


def _pipeline_bbox_geo_spec(*, config: dict | None = None) -> dict:
    bbox_geo_cfg = {
        "smoothl1_weight": 0.0,
        "ciou_weight": 0.0,
    }
    if isinstance(config, dict):
        bbox_geo_cfg.update(dict(config))
    return {
        "name": "bbox_geo",
        "enabled": True,
        "weight": 0.0,
        "channels": ["A", "B"],
        "application": {"preset": "anchor_only"},
        "config": bbox_geo_cfg,
    }


def _pipeline_bbox_size_aux_spec(*, config: dict | None = None) -> dict:
    bbox_size_aux_cfg = {
        "log_wh_weight": 0.0,
        "oversize_penalty_weight": 0.0,
        "oversize_area_frac_threshold": None,
        "oversize_log_w_threshold": None,
        "oversize_log_h_threshold": None,
        "eps": 1e-6,
    }
    if isinstance(config, dict):
        bbox_size_aux_cfg.update(dict(config))
    return {
        "name": "bbox_size_aux",
        "enabled": True,
        "weight": 0.0,
        "channels": ["A", "B"],
        "application": {"preset": "anchor_only"},
        "config": bbox_size_aux_cfg,
    }


def _pipeline_coord_reg_spec(*, config: dict | None = None) -> dict:
    coord_reg_cfg = {
        "coord_ce_weight": 0.0,
        "coord_gate_weight": 0.0,
        "text_gate_weight": 0.0,
        "soft_ce_weight": 0.0,
        "w1_weight": 0.0,
        "temperature": 1.0,
        "target_sigma": 2.0,
        "target_truncate": None,
        "adjacent_repulsion_weight": 0.0,
        "adjacent_repulsion_filter_mode": "same_desc",
        "adjacent_repulsion_margin_ratio": 0.05,
        "adjacent_repulsion_copy_margin": 0.8,
    }
    if isinstance(config, dict):
        coord_reg_cfg.update(dict(config))
    return {
        "name": "coord_reg",
        "enabled": True,
        "weight": 0.0,
        "channels": ["A", "B"],
        "application": {"preset": "anchor_only"},
        "config": coord_reg_cfg,
    }


def _canonical_stage2_two_channel_objective() -> list[dict]:
    return [
        _pipeline_token_ce_spec(),
        _pipeline_loss_duplicate_burst_unlikelihood_spec(),
        _pipeline_bbox_geo_spec(),
        _pipeline_bbox_size_aux_spec(),
        _pipeline_coord_reg_spec(),
    ]


def test_stage2_pipeline_unknown_module_name_fails_fast():
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["pipeline"] = {
        "objective": [
            _pipeline_token_ce_spec(),
            {
                "name": "unknown_module",
                "enabled": True,
                "weight": 1.0,
                "channels": ["B"],
                "config": {},
            },
            _pipeline_bbox_geo_spec(),
            _pipeline_bbox_size_aux_spec(),
            _pipeline_coord_reg_spec(),
        ],
    }

    with pytest.raises(ValueError, match=r"stage2_ab\.pipeline\.objective\[1\]\.name"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_duplicate_module_name_fails_fast():
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["pipeline"] = {
        "objective": [
            _pipeline_token_ce_spec(),
            _pipeline_token_ce_spec(),
            _pipeline_bbox_geo_spec(),
            _pipeline_bbox_size_aux_spec(),
            _pipeline_coord_reg_spec(),
        ],
    }

    with pytest.raises(
        ValueError, match=r"Duplicate module name in stage2_ab\.pipeline\.objective"
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_canonical_channels_scope_parses():
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["pipeline"] = {
        "objective": _canonical_stage2_two_channel_objective()
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.pipeline is not None
    assert cfg.stage2_ab.pipeline.objective[0].channels == ("A", "B")
    assert cfg.stage2_ab.pipeline.objective[1].channels == ("B",)
    assert cfg.stage2_ab.channel_b.insertion_order == "tail_append"


def test_stage2_channel_b_insertion_order_accepts_sorted() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["channel_b"] = {"insertion_order": "sorted"}
    payload["stage2_ab"]["pipeline"] = {
        "objective": _canonical_stage2_two_channel_objective()
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.channel_b.insertion_order == "sorted"


def test_stage2_channel_b_insertion_order_rejects_unknown_value() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["channel_b"] = {"insertion_order": "middle"}
    payload["stage2_ab"]["pipeline"] = {
        "objective": _canonical_stage2_two_channel_objective()
    }

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.channel_b\.insertion_order",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_disallows_custom_bbox_geo_knobs() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["custom"]["bbox_geo"] = {
        "enabled": True,
        "smoothl1_weight": 0.0,
        "ciou_weight": 1.0,
    }
    payload["stage2_ab"]["pipeline"] = {
        "objective": _canonical_stage2_two_channel_objective()
    }

    with pytest.raises(ValueError, match=r"custom\.bbox_geo"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_bbox_geo_unknown_alias_fails_fast() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["pipeline"] = {
        "objective": [
            _pipeline_token_ce_spec(),
            _pipeline_loss_duplicate_burst_unlikelihood_spec(),
            _pipeline_bbox_geo_spec(config={"center_wt": 1.0}),
            _pipeline_bbox_size_aux_spec(),
            _pipeline_coord_reg_spec(),
        ]
    }

    with pytest.raises(
        ValueError,
        match=r"Unknown stage2_ab\.pipeline\.objective\[2\]\.config keys.*center_wt",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_bbox_geo_rejects_zero_center_and_size_weights() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["pipeline"] = {
        "objective": [
            _pipeline_token_ce_spec(),
            _pipeline_loss_duplicate_burst_unlikelihood_spec(),
            _pipeline_bbox_geo_spec(
                config={
                    "smoothl1_weight": 0.5,
                    "ciou_weight": 0.25,
                    "parameterization": "center_size",
                    "center_weight": 0.0,
                    "size_weight": 0.0,
                }
            ),
            _pipeline_bbox_size_aux_spec(),
            _pipeline_coord_reg_spec(),
        ]
    }

    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.pipeline\.objective\[2\]\.config\.parameterization=center_size requires center_weight > 0 or size_weight > 0",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_disallows_custom_bbox_size_aux_knobs() -> None:
    payload = _base_stage2_two_channel_payload()
    payload["custom"]["bbox_size_aux"] = {
        "enabled": True,
        "log_wh_weight": 0.05,
        "oversize_penalty_weight": 0.0,
        "oversize_area_frac_threshold": None,
        "oversize_log_w_threshold": None,
        "oversize_log_h_threshold": None,
        "eps": 1e-6,
    }
    payload["stage2_ab"]["pipeline"] = {
        "objective": _canonical_stage2_two_channel_objective()
    }

    with pytest.raises(ValueError, match=r"custom\.bbox_size_aux"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_module_config_unknown_key_fails_fast():
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["pipeline"] = {
        "objective": [
            _pipeline_token_ce_spec(config={"unknown_knob": 1.0}),
            _pipeline_loss_duplicate_burst_unlikelihood_spec(),
            _pipeline_bbox_geo_spec(),
            _pipeline_bbox_size_aux_spec(),
            _pipeline_coord_reg_spec(),
        ]
    }

    with pytest.raises(ValueError, match=r"Unknown stage2_ab\.pipeline\.objective"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_legacy_matched_prefix_struct_knob_fails_fast():
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"]["pipeline"] = {
        "objective": [
            _pipeline_token_ce_spec(
                config={"rollout_matched_prefix_struct_weight": 1.0}
            ),
            _pipeline_loss_duplicate_burst_unlikelihood_spec(),
            _pipeline_bbox_geo_spec(),
            _pipeline_bbox_size_aux_spec(),
            _pipeline_coord_reg_spec(),
        ]
    }

    with pytest.raises(
        ValueError,
        match=r"rollout_matched_prefix_struct_weight",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_guardrail_stage2_two_channel_rejects_rollout_pipeline():
    payload = _base_stage2_two_channel_payload()
    payload["rollout_matching"]["pipeline"] = {
        "objective": [_pipeline_token_ce_spec()],
        "diagnostics": [],
    }

    with pytest.raises(ValueError, match=r"rollout_matching\.pipeline is not allowed"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_guardrail_rollout_aligned_rejects_stage2_pipeline():
    payload = _base_stage2_rollout_aligned_payload()
    payload["stage2_ab"] = {
        "schedule": {"b_ratio": 0.5},
        "pipeline": {
            "objective": _canonical_stage2_two_channel_objective(),
            "diagnostics": [],
        },
    }

    with pytest.raises(ValueError, match=r"stage2_ab\.pipeline is not allowed"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_rollout_pipeline_bbox_geo_rejects_zero_center_and_size_weights() -> None:
    payload = _base_stage2_rollout_aligned_payload()
    payload["rollout_matching"]["pipeline"] = {
        "objective": [
            _pipeline_bbox_geo_spec(
                config={
                    "smoothl1_weight": 0.5,
                    "ciou_weight": 0.25,
                    "parameterization": "center_size",
                    "center_weight": 0.0,
                    "size_weight": 0.0,
                }
            )
        ],
        "diagnostics": [],
    }

    with pytest.raises(
        ValueError,
        match=r"rollout_matching\.pipeline\.objective\[0\]\.config\.parameterization=center_size requires center_weight > 0 or size_weight > 0",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage2_pipeline_disallows_flat_objective_knobs():
    payload = _base_stage2_two_channel_payload()
    payload["stage2_ab"].update(
        {
            "desc_ce_weight": 0.5,
            "pipeline": {"objective": [_pipeline_token_ce_spec()], "diagnostics": []},
        }
    )

    with pytest.raises(
        ValueError, match=r"Flat stage2_ab objective knobs have been removed"
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())
