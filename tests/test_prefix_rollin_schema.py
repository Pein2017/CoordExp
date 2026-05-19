from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_prefix_rollin_payload() -> dict[str, object]:
    return {
        "model": {"model": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"},
        "template": {"truncation_strategy": "raise"},
        "training": {
            "run_name": "test-prefix-rollin",
            "num_train_epochs": 1,
        },
        "data": {
            "train_jsonl": "public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl",
            "val_jsonl": "public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl",
            "image_root": "public_data/coco",
            "object_ordering": "random_permutation",
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
            "id": "recursive_detection_ce",
            "variant": "prefix_rollin_et_rmp_ce",
            "state_weighting": "uniform_permutation",
            "normalization": "semantic_image_bucket_balanced",
            "rollin": {
                "enabled": True,
                "source": "ground_truth",
                "prefix_loss": "masked",
                "suffix_order": "same_sampled_permutation",
                "k_distribution": {
                    "type": "uniform_inclusive",
                    "min_k": 0,
                    "max_k": "object_count",
                },
            },
            "target": {
                "type": "entry_trie_support_balance",
                "trie_scope": "object_entry",
                "q_weighting": "object_multiplicity_uniform",
                "singleton": "hard_ce",
                "control_tokens": "hard_ce",
                "support_weight": 1.0,
                "balance_weight": 2.0,
            },
            "type_gate": {
                "enabled": True,
                "mode": "allowed_type_mass",
                "weights": {
                    "struct": 2.0,
                    "coord": 1.0,
                    "desc": 0.2,
                    "eos": 0.5,
                },
            },
        },
        "packing": {
            "static_packing": False,
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
        "experiment": {"surface": "ablation"},
    }


def _teacher_forcing_payload() -> dict[str, object]:
    payload = _legacy_prefix_rollin_payload()
    payload["objective"] = {
        "id": "teacher_forcing",
        "profile": "pure_valid_set_marginal",
        "target_ir": {
            "rollin_policy": {
                "name": "random_permutation",
                "base_seed": 17,
            },
            "exact_packing_mapping": {"enabled": False},
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
    return payload


def _parse(payload: dict[str, object]) -> LatestDetectionTrainingConfig:
    return LatestDetectionTrainingConfig.from_mapping(payload)


def _set_path(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    cursor: object = payload
    for key in path[:-1]:
        assert isinstance(cursor, dict)
        cursor = cursor[key]
    assert isinstance(cursor, dict)
    cursor[path[-1]] = value


@pytest.mark.parametrize(
    "objective",
    [
        {"id": "recursive_detection_ce", "variant": "prefix_rollin_et_rmp_ce"},
        {"id": "recursive_detection_ce", "variant": "random_permutation_et_rmp_ce"},
        {"id": "prefix_rollin_et_rmp_ce"},
        {"id": "sft", "variant": "sorted_sft"},
    ],
)
def test_legacy_prefix_rollin_objectives_fail_at_migration_boundary(
    objective: dict[str, object],
) -> None:
    payload = _legacy_prefix_rollin_payload()
    payload["objective"] = {**payload["objective"], **objective}  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        _parse(payload)


def test_prefix_rollin_e1_ablation_config_is_historical_fail_fast() -> None:
    cfg_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml"
    )

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        ConfigLoader.load_materialized_training_config(str(cfg_path))


def test_prefix_rollin_a3_ablation_config_is_historical_fail_fast() -> None:
    cfg_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2_a3_bsz1_ebs128.yaml"
    )

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        ConfigLoader.load_materialized_training_config(str(cfg_path))


@pytest.mark.parametrize("claim_scope", ["paper", "production"])
def test_latest_ablation_rejects_evidence_bearing_claim_scope(
    claim_scope: str,
) -> None:
    payload = _teacher_forcing_payload()
    _set_path(payload, ("experiment", "surface"), "ablation")
    _set_path(payload, ("experiment", "claim_scope"), claim_scope)

    with pytest.raises(ValueError, match=r"experiment\.claim_scope"):
        _parse(payload)


def test_latest_smoke_surface_rejects_production_claim_scope() -> None:
    payload = _teacher_forcing_payload()
    _set_path(payload, ("experiment", "surface"), "smoke")
    _set_path(payload, ("experiment", "claim_scope"), "production")

    with pytest.raises(ValueError, match=r"experiment\.claim_scope"):
        _parse(payload)


@pytest.mark.parametrize(
    "custom",
    [
        {"stage1_set_continuation": {}},
        {"trainer_variant": "stage1_set_continuation"},
    ],
)
def test_removed_set_continuation_custom_keys_still_fail_fast(
    custom: dict[str, object],
) -> None:
    payload = _teacher_forcing_payload()
    payload["custom"] = custom

    with pytest.raises(ValueError, match="stage1_set_continuation"):
        _parse(payload)
