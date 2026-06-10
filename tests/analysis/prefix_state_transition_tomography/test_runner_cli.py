from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from src.analysis.prefix_state_transition_tomography.runner import run_from_config


def _write_config(path: Path, artifact_root: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "project_id": "prefix_state_transition_tomography",
                "artifact_root": str(artifact_root),
                "train_jsonl": str(path.parent / "train.coord.jsonl"),
                "val_jsonl": str(path.parent / "val.coord.jsonl"),
                "stages": ["prefix_state_index", "validate"],
                "checkpoints": {
                    "et_rmp_ce": {"checkpoint_path": "/ckpts/et/checkpoint-3664"},
                    "pure_ce": {"checkpoint_path": "/ckpts/pure/checkpoint-3664"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_runner_dry_run_returns_json_safe_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_config(config_path, artifact_root)

    result = run_from_config(
        config_path,
        stages=("prefix_state_index",),
        dry_run=True,
        allow_overwrite=True,
        shard_id=0,
    )

    assert result["project_id"] == "prefix_state_transition_tomography"
    assert result["artifact_root"] == str(artifact_root)
    assert result["stages"] == ["prefix_state_index"]
    assert result["checkpoints"]["et_rmp_ce"] == "/ckpts/et/checkpoint-3664"
    assert result["checkpoints"]["pure_ce"] == "/ckpts/pure/checkpoint-3664"
    assert result["allow_overwrite"] is True
    assert result["shard_id"] == 0
    assert not artifact_root.exists()


def test_runner_rejects_out_of_range_shard_id(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_config(config_path, artifact_root)

    with pytest.raises(ValueError, match="shard_id must satisfy"):
        run_from_config(config_path, dry_run=True, shard_id=8)


def test_runner_cli_dry_run_does_not_create_artifact_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_config(config_path, artifact_root)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/prefix_state_transition_tomography/run.py",
            "--config",
            str(config_path),
            "--stages",
            "prefix_state_index",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    result = json.loads(proc.stdout)
    assert result["project_id"] == "prefix_state_transition_tomography"
    assert result["stages"] == ["prefix_state_index"]
    assert result["sampling"]["num_shards"] == 8
    assert not artifact_root.exists()
