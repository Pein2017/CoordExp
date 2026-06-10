from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.prefix_state_transition_tomography.config import load_config


def test_load_config_parses_phase_a3_pair(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_id: prefix_state_transition_tomography
artifact_root: /tmp/a3
train_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
val_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
stages: [prefix_state_index]
checkpoints:
  et_rmp_ce:
    checkpoint_path: /ckpts/et/checkpoint-3664
  pure_ce:
    checkpoint_path: /ckpts/pure/checkpoint-3664
sampling:
  max_prefix_states: 4096
  num_shards: 8
  seed: 3664
peak:
  absolute_mass_floor: 0.002
  relative_floor: 0.10
  primary_merge_radius: 24
  gt_x1_neighborhood_radius: 24
  raw_topk_k: 32
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.project_id == "prefix_state_transition_tomography"
    assert cfg.artifact_root == Path("/tmp/a3")
    assert cfg.checkpoints["et_rmp_ce"].checkpoint_path == Path("/ckpts/et/checkpoint-3664")
    assert cfg.checkpoints["pure_ce"].checkpoint_path == Path("/ckpts/pure/checkpoint-3664")
    assert cfg.sampling.max_prefix_states == 4096
    assert cfg.sampling.num_shards == 8
    assert cfg.peak.primary_merge_radius == 24
    assert cfg.image_root is None


def test_load_config_rejects_wrong_project_id(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
project_id: candidate_field_cardinality_tomography
artifact_root: /tmp/a3
train_jsonl: /tmp/train.jsonl
val_jsonl: /tmp/val.jsonl
checkpoints:
  et_rmp_ce:
    checkpoint_path: /ckpts/et
  pure_ce:
    checkpoint_path: /ckpts/pure
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="project_id must be prefix_state_transition_tomography"):
        load_config(config_path)


def test_load_config_rejects_unknown_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-stage.yaml"
    config_path.write_text(
        """
project_id: prefix_state_transition_tomography
artifact_root: /tmp/a3
train_jsonl: /tmp/train.jsonl
val_jsonl: /tmp/val.jsonl
stages: [prefix_state_index, launch_training]
checkpoints:
  et_rmp_ce:
    checkpoint_path: /ckpts/et
  pure_ce:
    checkpoint_path: /ckpts/pure
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown stage"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("stages: []", "stages must not be empty"),
        ("sampling:\n  max_prefix_states: 0", "sampling.max_prefix_states must be positive"),
        ("sampling:\n  num_shards: 0", "sampling.num_shards must be positive"),
        (
            "sampling:\n  easy_sanity_max_fraction: 1.5",
            "sampling.easy_sanity_max_fraction must be between 0 and 1",
        ),
        ("peak:\n  primary_merge_radius: -1", "peak.primary_merge_radius must be non-negative"),
        ("peak:\n  raw_topk_k: 0", "peak.raw_topk_k must be positive"),
    ],
)
def test_load_config_rejects_invalid_contract_values(
    tmp_path: Path,
    override: str,
    message: str,
) -> None:
    config_path = tmp_path / "bad-contract.yaml"
    config_path.write_text(
        _base_config(tmp_path) + "\n" + override + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_load_config_rejects_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "relative.yaml"
    config_path.write_text(
        """
project_id: prefix_state_transition_tomography
artifact_root: relative/artifacts
train_jsonl: /tmp/train.jsonl
val_jsonl: /tmp/val.jsonl
checkpoints:
  et_rmp_ce:
    checkpoint_path: /ckpts/et
  pure_ce:
    checkpoint_path: /ckpts/pure
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact_root must be an absolute path"):
        load_config(config_path)


def _base_config(tmp_path: Path) -> str:
    return f"""
project_id: prefix_state_transition_tomography
artifact_root: {tmp_path / "artifacts"}
train_jsonl: {tmp_path / "train.coord.jsonl"}
val_jsonl: {tmp_path / "val.coord.jsonl"}
stages: [prefix_state_index]
checkpoints:
  et_rmp_ce:
    checkpoint_path: /ckpts/et
  pure_ce:
    checkpoint_path: /ckpts/pure
"""
