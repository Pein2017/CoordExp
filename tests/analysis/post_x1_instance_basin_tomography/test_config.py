from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.analysis.post_x1_instance_basin_tomography.config import load_config


CONFIG = Path(
    "configs/analysis/post_x1_instance_basin_tomography/"
    "three_ckpt_phase_a3_3_smoke.yaml"
)

EXPECTED_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
    "et_rmp_ce_ckpt3664",
)


def test_a3_3_config_resolves_three_roles_and_contracts() -> None:
    config = load_config(CONFIG, validate_paths=False)

    assert config.project_id == "post_x1_instance_basin_tomography"
    assert config.phase_id == "phase_a3_3"
    assert config.schema_version == "a3.3.v1"
    assert config.run_id == "three_ckpt_phase_a3_3_smoke"
    assert tuple(config.checkpoints) == EXPECTED_ROLES

    random = config.checkpoints["fullobj_random_pure_ce_ckpt3668"]
    sorted_ = config.checkpoints["fullobj_sorted_pure_ce_ckpt3668"]
    et = config.checkpoints["et_rmp_ce_ckpt3664"]

    assert random.enabled is True
    assert sorted_.enabled is True
    assert et.enabled is True
    assert random.comparison_role == "clean_pair"
    assert sorted_.comparison_role == "clean_pair"
    assert et.comparison_role == "reference_anchor"
    assert random.controlled_comparison_group == "pure_ce_sorted_vs_random_no_newline"
    assert sorted_.controlled_comparison_group == "pure_ce_sorted_vs_random_no_newline"
    assert et.controlled_comparison_group == "reference_anchor_not_controlled"

    assert random.chat_template_variant == "compact_full_no_newline_native_v1"
    assert sorted_.chat_template_variant == "compact_full_no_newline_native_v1"
    assert et.chat_template_variant == "compact_full_newline_native_v1"
    assert random.template_contract.row_separator == "none"
    assert sorted_.template_contract.row_separator == "none"
    assert et.template_contract.row_separator == "newline"
    assert et.template_contract.contract_provenance == "legacy_compact_full_default_inferred"

    assert config.case_sampling.max_images == 64
    assert config.case_sampling.max_target_instances == 256
    assert config.case_sampling.min_same_desc_count == 3
    assert config.case_sampling.split_quotas["train"] == 48
    assert config.case_sampling.split_quotas["val"] == 16
    assert config.case_sampling.desc_cap_per_split == 12
    assert config.case_sampling.desc_count_caps["train"] == 12
    assert config.case_sampling.desc_count_caps["val"] == 12
    assert config.case_sampling.object_count_buckets == ("6-9", "10-19", "20+")

    assert config.prefix.prefix_modes_requested == (
        "minimal_or_empty_prefix",
        "canonical_sorted_gt_prefix_before_target",
        "clean_non_target_same_desc_prefix",
        "duplicate_same_desc_prefix",
    )
    assert config.prefix.rollout_prefix_missing_policy_smoke == "skip_with_manifest"
    assert config.prefix.rollout_prefix_missing_policy_full == "fail"
    assert config.posterior.coord_mass_low_threshold == pytest.approx(0.01)
    assert config.posterior.r95_anchor_policy == "target_axis_fraction_cap"
    assert config.posterior.strict_r95_axis_fraction == pytest.approx(0.04)
    assert config.posterior.strict_r95_cap_bins == 8
    assert config.runtime.num_shards == 8
    assert config.greedy.sample_fraction == pytest.approx(0.10)


def test_a3_3_full_config_uses_same_roles_with_larger_sampling() -> None:
    config = load_config(
        "configs/analysis/post_x1_instance_basin_tomography/"
        "three_ckpt_phase_a3_3.yaml",
        validate_paths=False,
    )

    assert config.run_id == "three_ckpt_phase_a3_3"
    assert tuple(config.checkpoints) == EXPECTED_ROLES
    assert config.case_sampling.max_images == 1024
    assert config.case_sampling.max_target_instances == 4096
    assert "rollout_native_prefix_with_quality_label" in config.prefix.prefix_modes_requested
    assert config.greedy.sample_fraction == pytest.approx(0.20)


def test_config_rejects_missing_et_template_contract(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    raw = _base_config_data(tmp_path)
    raw["checkpoints"]["et_rmp_ce_ckpt3664"]["template_contract"][
        "row_separator"
    ] = "none"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="et_rmp_ce_ckpt3664.*row_separator"):
        load_config(path)


def test_config_rejects_global_template_contract(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    raw = _base_config_data(tmp_path)
    raw["template_contract"] = {"row_separator": "none"}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="per-checkpoint template_contract"):
        load_config(path)


def test_config_rejects_missing_checkpoint_roles(tmp_path: Path) -> None:
    raw = _base_config_data(tmp_path)
    del raw["checkpoints"]["fullobj_sorted_pure_ce_ckpt3668"]
    config_path = _write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="checkpoint roles must be ordered exactly"):
        load_config(config_path)


def test_config_rejects_relative_paths(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(_base_config_data(tmp_path), ("artifact_root",), "relative/artifacts"),
    )

    with pytest.raises(ValueError, match="artifact_root must be an absolute path"):
        load_config(config_path)


def test_config_validates_checkpoint_paths_by_default(tmp_path: Path) -> None:
    raw = _base_config_data(tmp_path)
    missing = tmp_path / "missing" / "checkpoint-3668"
    raw["checkpoints"]["fullobj_random_pure_ce_ckpt3668"]["checkpoint_path"] = str(
        missing
    )
    config_path = _write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="fullobj_random_pure_ce_ckpt3668.*checkpoint_path"):
        load_config(config_path)


def test_config_can_skip_checkpoint_path_validation(tmp_path: Path) -> None:
    raw = _base_config_data(tmp_path)
    raw["checkpoints"]["fullobj_random_pure_ce_ckpt3668"][
        "checkpoint_path"
    ] = "/tmp/does-not-exist/checkpoint-3668"
    config_path = _write_config(tmp_path, raw)

    config = load_config(config_path, validate_paths=False)

    assert config.checkpoints["fullobj_random_pure_ce_ckpt3668"].checkpoint_path == Path(
        "/tmp/does-not-exist/checkpoint-3668"
    )


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (
            (
                "checkpoints",
                "fullobj_random_pure_ce_ckpt3668",
                "controlled_comparison_group",
            ),
            "reference_anchor_not_controlled",
            "pure-CE controlled comparison group",
        ),
        (
            (
                "checkpoints",
                "fullobj_sorted_pure_ce_ckpt3668",
                "template_contract",
                "row_separator",
            ),
            "newline",
            "fullobj_sorted_pure_ce_ckpt3668.*row_separator",
        ),
        (("runtime", "num_shards"), 4, "runtime.num_shards must be 8"),
        (
            ("posterior", "coord_mass_low_threshold"),
            -0.1,
            "posterior.coord_mass_low_threshold must be non-negative",
        ),
        (
            ("posterior", "r95_anchor_policy"),
            "loose_neighbor_radius",
            "posterior.r95_anchor_policy must be target_axis_fraction_cap",
        ),
        (
            ("prefix", "rollout_prefix_missing_policy_smoke"),
            "fail",
            "prefix.rollout_prefix_missing_policy_smoke must be skip_with_manifest",
        ),
        (
            ("prefix", "prefix_modes_requested"),
            [],
            "prefix.prefix_modes_requested must not be empty",
        ),
        (
            ("case_sampling", "split_quotas"),
            {"train": 64},
            "case_sampling.split_quotas must include all configured splits",
        ),
    ],
)
def test_config_rejects_invalid_contract_values(
    tmp_path: Path,
    keys: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(_base_config_data(tmp_path), keys, value),
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def _write_config(tmp_path: Path, raw: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def _with_update(
    raw: dict[str, Any],
    keys: tuple[str, ...],
    value: Any,
) -> dict[str, Any]:
    result = deepcopy(raw)
    cursor: dict[str, Any] = result
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value
    return result


def _base_config_data(tmp_path: Path) -> dict[str, Any]:
    random_checkpoint = tmp_path / "random" / "checkpoint-3668"
    sorted_checkpoint = tmp_path / "sorted" / "checkpoint-3668"
    et_checkpoint = tmp_path / "et" / "checkpoint-3664"
    random_checkpoint.mkdir(parents=True)
    sorted_checkpoint.mkdir(parents=True)
    et_checkpoint.mkdir(parents=True)

    return {
        "project_id": "post_x1_instance_basin_tomography",
        "phase_id": "phase_a3_3",
        "schema_version": "a3.3.v1",
        "run_id": "three_ckpt_phase_a3_3_smoke",
        "artifact_root": "/tmp/a3_3/artifacts",
        "train_jsonl": "/tmp/a3_3/train.coord.jsonl",
        "val_jsonl": "/tmp/a3_3/val.coord.jsonl",
        "image_root": "/tmp/a3_3/rescale_32_1024_bbox",
        "case_sampling": {
            "max_images": 64,
            "max_target_instances": 256,
            "min_same_desc_count": 3,
            "min_object_count": 6,
            "easy_sanity_max_fraction": 0.10,
            "seed": 333,
            "splits": ["train", "val"],
            "split_quotas": {"train": 48, "val": 16},
            "desc_cap_per_split": 12,
            "desc_count_caps": {"train": 12, "val": 12},
            "object_count_buckets": ["6-9", "10-19", "20+"],
        },
        "prefix": {
            "prefix_modes_requested": [
                "minimal_or_empty_prefix",
                "canonical_sorted_gt_prefix_before_target",
                "clean_non_target_same_desc_prefix",
                "duplicate_same_desc_prefix",
            ],
            "rollout_prefix_source_jsonl": None,
            "rollout_prefix_missing_policy_smoke": "skip_with_manifest",
            "rollout_prefix_missing_policy_full": "fail",
        },
        "posterior": {
            "strict_r95_axis_fraction": 0.04,
            "strict_r95_cap_bins": 8,
            "peak_mass_floor": 0.002,
            "low_margin_threshold": 0.05,
            "coord_mass_low_threshold": 0.01,
            "r95_anchor_policy": "target_axis_fraction_cap",
        },
        "greedy": {
            "enabled": True,
            "sample_fraction": 0.10,
            "decode_policy": "free_text_unconstrained_greedy_temp0",
            "constraint_policy": "none",
        },
        "runtime": {
            "num_shards": 8,
            "max_new_tokens": 64,
            "torch_dtype": "bfloat16",
            "device_map": "single_gpu",
        },
        "checkpoints": {
            "fullobj_random_pure_ce_ckpt3668": {
                "enabled": True,
                "checkpoint_path": str(random_checkpoint),
                "training_ordering": "random_permutation",
                "comparison_role": "clean_pair",
                "controlled_comparison_group": "pure_ce_sorted_vs_random_no_newline",
                "chat_template_variant": "compact_full_no_newline_native_v1",
                "template_contract": {
                    "template_contract_id": "compact_full_no_newline_native_v1",
                    "detection_sequence_format": "compact_full",
                    "coordinate_surface": "coord_token",
                    "bbox_format": "xyxy",
                    "row_separator": "none",
                    "contract_provenance": "user_reported_training_contract",
                },
            },
            "fullobj_sorted_pure_ce_ckpt3668": {
                "enabled": True,
                "checkpoint_path": str(sorted_checkpoint),
                "training_ordering": "sorted",
                "comparison_role": "clean_pair",
                "controlled_comparison_group": "pure_ce_sorted_vs_random_no_newline",
                "chat_template_variant": "compact_full_no_newline_native_v1",
                "template_contract": {
                    "template_contract_id": "compact_full_no_newline_native_v1",
                    "detection_sequence_format": "compact_full",
                    "coordinate_surface": "coord_token",
                    "bbox_format": "xyxy",
                    "row_separator": "none",
                    "contract_provenance": "user_reported_training_contract",
                },
            },
            "et_rmp_ce_ckpt3664": {
                "enabled": True,
                "checkpoint_path": str(et_checkpoint),
                "training_ordering": "random_permutation",
                "comparison_role": "reference_anchor",
                "controlled_comparison_group": "reference_anchor_not_controlled",
                "chat_template_variant": "compact_full_newline_native_v1",
                "template_contract": {
                    "template_contract_id": "compact_full_newline_native_v1",
                    "detection_sequence_format": "compact_full",
                    "coordinate_surface": "coord_token",
                    "bbox_format": "xyxy",
                    "row_separator": "newline",
                    "contract_provenance": "legacy_compact_full_default_inferred",
                },
            },
        },
    }
