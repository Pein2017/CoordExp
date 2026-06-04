from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.analysis.sorted_random_no_newline_phenotype.config import load_config


CONFIG_PATH = Path(
    "configs/analysis/sorted_random_no_newline_phenotype/"
    "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml"
)
EXPECTED_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
)


def test_a3_2_config_resolves_semantic_roles() -> None:
    config = load_config(CONFIG_PATH)

    assert config.project_id == "sorted_random_no_newline_phenotype"
    assert config.phase_id == "phase_a3_2"
    assert config.schema_version == "a3.2.v1"
    assert config.run_id == "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2"
    assert tuple(config.checkpoints) == EXPECTED_ROLES
    assert config.template_contract.row_separator == "none"
    assert config.template_contract.detection_sequence_format == "compact_full"
    assert config.template_contract.coordinate_surface == "coord_token"
    assert config.template_contract.bbox_format == "xyxy"
    assert config.sampling.max_prefix_states == 4096
    assert config.sampling.num_shards == 8
    assert config.rollout.limit_images == 1024
    assert config.rollout.decode_policy == "free_text_unconstrained_greedy_temp0"
    assert config.rollout.constraint_policy == "none"
    assert config.fn_probe.max_fn_objects_per_checkpoint == 512


def test_load_config_rejects_invalid_row_separator(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(
            _base_config_data(tmp_path),
            ("template_contract", "row_separator"),
            "newline",
        ),
    )

    with pytest.raises(ValueError, match="row_separator must be none"):
        load_config(config_path)


def test_load_config_rejects_missing_checkpoint_roles(tmp_path: Path) -> None:
    raw = _base_config_data(tmp_path)
    del raw["checkpoints"]["fullobj_sorted_pure_ce_ckpt3668"]
    config_path = _write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="checkpoint roles must be exactly"):
        load_config(config_path)


def test_load_config_rejects_non_absolute_paths(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(_base_config_data(tmp_path), ("artifact_root",), "relative/artifacts"),
    )

    with pytest.raises(ValueError, match="artifact_root must be an absolute path"):
        load_config(config_path)


def test_load_config_rejects_image_root_pointing_to_len12000(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(
            _base_config_data(tmp_path),
            ("image_root",),
            "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000",
        ),
    )

    with pytest.raises(ValueError, match="image_root must not point to len12000"):
        load_config(config_path)


def test_load_config_rejects_wrong_num_shards(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(_base_config_data(tmp_path), ("sampling", "num_shards"), 4),
    )

    with pytest.raises(ValueError, match="sampling.num_shards must be 8"):
        load_config(config_path)


def test_load_config_rejects_nonexistent_checkpoint_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing" / "checkpoint-3668"
    config_path = _write_config(
        tmp_path,
        _with_update(
            _base_config_data(tmp_path),
            ("checkpoints", "fullobj_random_pure_ce_ckpt3668", "checkpoint_path"),
            str(missing_path),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"checkpoints\.fullobj_random_pure_ce_ckpt3668\.checkpoint_path",
    ):
        load_config(config_path)


def test_load_config_rejects_constrained_decode_policy(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(
            _base_config_data(tmp_path),
            ("rollout", "decode_policy"),
            "greedy_temp0",
        ),
    )

    with pytest.raises(
        ValueError,
        match="rollout.decode_policy must be free_text_unconstrained_greedy_temp0",
    ):
        load_config(config_path)


def test_load_config_rejects_non_none_constraint_policy(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        _with_update(
            _base_config_data(tmp_path),
            ("rollout", "constraint_policy"),
            "compact_grammar",
        ),
    )

    with pytest.raises(ValueError, match="rollout.constraint_policy must be none"):
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
    random_checkpoint.mkdir(parents=True)
    sorted_checkpoint.mkdir(parents=True)

    return {
        "project_id": "sorted_random_no_newline_phenotype",
        "phase_id": "phase_a3_2",
        "schema_version": "a3.2.v1",
        "run_id": "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2",
        "artifact_root": "/tmp/a3_2/artifacts",
        "train_jsonl": "/tmp/a3_2/train.coord.jsonl",
        "val_jsonl": "/tmp/a3_2/val.coord.jsonl",
        "image_root": "/tmp/a3_2/rescale_32_1024_bbox",
        "template_contract": {
            "detection_sequence_format": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "row_separator": "none",
        },
        "sampling": {
            "max_prefix_states": 4096,
            "num_shards": 8,
            "seed": 3668,
            "easy_sanity_max_fraction": 0.20,
        },
        "rollout": {
            "limit_images": 1024,
            "decode_policy": "free_text_unconstrained_greedy_temp0",
            "native_prompt_ordering": True,
            "constraint_policy": "none",
        },
        "fn_probe": {
            "max_fn_objects_per_checkpoint": 512,
            "hint_policy_id": "desc_x1_r95_ladder_v1",
            "strict_r95_axis_fraction": 0.04,
            "strict_r95_cap_bins": 8,
            "broad_x1_radius": 24,
        },
        "peak": {
            "absolute_mass_floor": 0.002,
            "relative_floor": 0.10,
            "primary_merge_radius": 24,
            "gt_x1_neighborhood_radius": 24,
            "raw_topk_k": 32,
        },
        "checkpoints": {
            "fullobj_random_pure_ce_ckpt3668": {
                "training_ordering": "random_permutation",
                "readout_prompt_ordering": "sorted",
                "checkpoint_path": str(random_checkpoint),
            },
            "fullobj_sorted_pure_ce_ckpt3668": {
                "training_ordering": "sorted",
                "readout_prompt_ordering": "sorted",
                "checkpoint_path": str(sorted_checkpoint),
            },
        },
    }
