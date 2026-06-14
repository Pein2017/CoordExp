from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.config.schema import DetectionTrainingConfig
from src.detection.runtime import (
    assert_detection_runtime_supported,
    detection_mode,
)


def _prefix_denoising_payload() -> dict[str, object]:
    return {
        "model": {
            "model": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp",
        },
        "template": {
            "template": "qwen3_vl",
            "truncation_strategy": "raise",
            "max_length": 12000,
            "max_pixels": 1048576,
        },
        "training": {
            "run_name": "prefix-denoising-test",
            "num_train_epochs": 1,
            "packing": True,
            "eval_packing": False,
            "encoded_sample_cache": {"enabled": False},
        },
        "data": {
            "train_jsonl": (
                "public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl"
            ),
            "val_jsonl": (
                "public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl"
            ),
            "image_root": "public_data/coco/rescale_32_1024_bbox_max60",
            "object_ordering": "sorted",
        },
        "prompt": {
            "system_variant": "stage1_detection",
            "user_variant": "compact_detection",
            "include_template_summary": True,
            "prompt_variant_enabled": True,
        },
        "detection_template": {
            "id": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "strict_parse": True,
        },
        "token_rows": {
            "enabled": True,
            "tie_head": True,
            "groups": {
                "coord_geometry": {
                    "role": "coord_geometry",
                    "start_token": "<|coord_0|>",
                    "end_token": "<|coord_999|>",
                    "expected_start": 151670,
                    "expected_end": 152669,
                },
                "compact_structure": {
                    "role": "structural_ce_only",
                    "tokens": ["<|object_ref_start|>", "<|box_start|>"],
                    "expected_ids": {
                        "<|object_ref_start|>": 151646,
                        "<|box_start|>": 151648,
                    },
                },
            },
            "embed_lr": 5.0e-5,
            "weight_decay": 0.0,
        },
        "objective": {
            "id": "teacher_forcing",
            "profile": "hard_sft",
            "modules": {
                "token_type_mass": {"enabled": False},
                "conditional_valid_set_likelihood": {"enabled": False},
                "within_valid_coverage": {
                    "enabled": False,
                    "coverage_strength": 0.0,
                },
                "continuation_margin": {"enabled": False},
            },
        },
        "prefix_denoising": {
            "enabled": True,
            "noise": {
                "center_shift_frac": 0.08,
                "uniform_scale_range": [0.92, 1.08],
            },
            "current_object_kl": {
                "weight": 0.05,
                "window_radius": 8,
                "num_objects_per_image": 1,
            },
        },
        "packing": {
            "static_packing": True,
            "padding_free_packed": False,
        },
        "evaluation": {
            "expected_template": "compact_full",
            "parser_mode": "strict_expected",
        },
        "validation": {
            "validate_span_alignment": True,
            "validate_template_capabilities": True,
            "fail_fast": True,
        },
    }


def _load(payload: dict[str, object]) -> DetectionTrainingConfig:
    return DetectionTrainingConfig.from_mapping(copy.deepcopy(payload))


def test_prefix_denoising_accepts_hard_sft_sorted_teacher_forcing() -> None:
    cfg = _load(_prefix_denoising_payload())

    assert cfg.prefix_denoising.enabled is True
    assert cfg.prefix_denoising.noise.center_shift_frac == pytest.approx(0.08)
    assert cfg.prefix_denoising.noise.uniform_scale_range == pytest.approx(
        (0.92, 1.08)
    )
    assert cfg.prefix_denoising.current_object_kl.weight == pytest.approx(0.05)
    assert cfg.prefix_denoising.current_object_kl.window_radius == 8
    assert cfg.prefix_denoising.current_object_kl.num_objects_per_image == 1
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "hard_sft"
    assert cfg.data.object_ordering == "sorted"
    assert cfg.training["packing"] is True
    assert cfg.packing.static_packing is True


def test_prefix_denoising_runtime_preflight_allows_static_training_packing() -> None:
    cfg = _load(_prefix_denoising_payload())

    assert_detection_runtime_supported(
        cfg,
        encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
        tokenizer=None,
    )


def test_prefix_denoising_detection_mode_is_distinct_from_random_order_sft() -> None:
    cfg = _load(_prefix_denoising_payload())

    assert detection_mode(cfg) == "prefix_denoising_sft"


def test_teacher_forcing_runtime_preflight_still_rejects_packing_without_prefix_denoising() -> None:
    payload = _prefix_denoising_payload()
    payload.pop("prefix_denoising")
    payload["data"] = {
        **payload["data"],  # type: ignore[arg-type]
        "object_ordering": "random_permutation",
    }
    payload["training"] = {
        **payload["training"],  # type: ignore[arg-type]
        "packing": False,
    }
    payload["packing"] = {
        **payload["packing"],  # type: ignore[arg-type]
        "static_packing": False,
    }
    cfg = _load(payload)

    with pytest.raises(ValueError, match=r"training\.packing=false"):
        assert_detection_runtime_supported(
            replace(cfg, training={**cfg.training, "packing": True}),
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
            tokenizer=None,
        )

    with pytest.raises(ValueError, match=r"packing\.static_packing=false"):
        assert_detection_runtime_supported(
            replace(cfg, packing=replace(cfg.packing, static_packing=True)),
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
            tokenizer=None,
        )


def test_prefix_denoising_omitted_defaults_to_disabled() -> None:
    payload = _prefix_denoising_payload()
    payload.pop("prefix_denoising")
    payload["data"] = {
        **payload["data"],  # type: ignore[arg-type]
        "object_ordering": "random_permutation",
    }
    payload["training"] = {
        **payload["training"],  # type: ignore[arg-type]
        "packing": False,
    }
    payload["packing"] = {
        **payload["packing"],  # type: ignore[arg-type]
        "static_packing": False,
    }

    cfg = _load(payload)

    assert cfg.prefix_denoising.enabled is False
    assert cfg.prefix_denoising.noise.center_shift_frac == pytest.approx(0.08)
    assert cfg.prefix_denoising.current_object_kl.weight == pytest.approx(0.05)


def test_prefix_denoising_accepts_zero_current_object_kl_weight_as_ce_only() -> None:
    payload = _prefix_denoising_payload()
    prefix_denoising = payload["prefix_denoising"]
    assert isinstance(prefix_denoising, dict)
    current_object_kl = prefix_denoising["current_object_kl"]
    assert isinstance(current_object_kl, dict)
    current_object_kl["weight"] = 0.0

    cfg = _load(payload)

    assert cfg.prefix_denoising.enabled is True
    assert cfg.prefix_denoising.current_object_kl.weight == pytest.approx(0.0)


def test_prefix_denoising_rejects_random_object_ordering() -> None:
    payload = _prefix_denoising_payload()
    payload["data"] = {
        **payload["data"],  # type: ignore[arg-type]
        "object_ordering": "random_permutation",
    }

    with pytest.raises(
        ValueError,
        match=r"prefix_denoising.*data\.object_ordering.*sorted",
    ):
        _load(payload)


def test_prefix_denoising_rejects_non_hard_sft_profile() -> None:
    payload = _prefix_denoising_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "profile": "pure_valid_set_marginal",
    }

    with pytest.raises(
        ValueError,
        match=r"prefix_denoising.*hard clean-label CE.*hard_sft",
    ):
        _load(payload)


def test_prefix_denoising_rejects_enabled_teacher_forcing_modules() -> None:
    payload = _prefix_denoising_payload()
    objective = payload["objective"]
    assert isinstance(objective, dict)
    modules = objective["modules"]
    assert isinstance(modules, dict)
    modules["token_type_mass"] = {"enabled": True}

    with pytest.raises(
        ValueError,
        match=r"prefix_denoising.*teacher-forcing modules disabled.*token_type_mass",
    ):
        _load(payload)


def test_prefix_denoising_rejects_explicit_target_ir_rollin_policy() -> None:
    payload = _prefix_denoising_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "target_ir": {
            "rollin_policy": {
                "name": "random_permutation",
                "base_seed": 17,
            },
        },
    }

    with pytest.raises(
        ValueError,
        match=r"prefix_denoising.*target_ir\.rollin_policy",
    ):
        _load(payload)


def test_prefix_denoising_rejects_coord_soft_ce() -> None:
    payload = _prefix_denoising_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "iou_gibbs_v0",
            "tau": 0.01,
            "tau_source": "train_one_token_iou_median_v0",
        },
    }

    with pytest.raises(
        ValueError,
        match=r"prefix_denoising.*SoftCE",
    ):
        _load(payload)


def test_prefix_denoising_rejects_encoded_sample_cache() -> None:
    payload = _prefix_denoising_payload()
    payload["training"] = {
        **payload["training"],  # type: ignore[arg-type]
        "encoded_sample_cache": {"enabled": True},
    }

    with pytest.raises(
        ValueError,
        match=r"prefix_denoising.*encoded_sample_cache",
    ):
        _load(payload)


def test_prefix_denoising_rejects_unknown_nested_key() -> None:
    payload = _prefix_denoising_payload()
    payload["prefix_denoising"] = {
        **payload["prefix_denoising"],  # type: ignore[arg-type]
        "sampler": {"name": "current_object"},
    }

    with pytest.raises(
        ValueError,
        match=r"Unknown prefix_denoising keys.*prefix_denoising\.sampler",
    ):
        _load(payload)
