from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.analysis.candidate_field_cardinality_tomography.config import load_config


def _write_config(path: Path, *, stages: list[str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "project_id": "candidate_field_cardinality_tomography",
                "artifact_root": str(path.parent / "artifacts"),
                "checkpoint_path": "/tmp/checkpoint-3664",
                "train_jsonl": "/tmp/train.coord.jsonl",
                "val_jsonl": "/tmp/val.coord.jsonl",
                "fn_rescue_overlay_root": None,
                "phase5_overlay_root": None,
                "stages": stages,
                "peak": {},
                "sampling": {},
                "policies": {},
            }
        ),
        encoding="utf-8",
    )


def test_load_config_accepts_known_stages(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, stages=["case_index", "probe_plan", "validate"])

    config = load_config(path)

    assert config.project_id == "candidate_field_cardinality_tomography"
    assert config.stages == ("case_index", "probe_plan", "validate")


def test_load_config_rejects_unknown_stage(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, stages=["case_index", "launch_training"])

    with pytest.raises(ValueError, match="unknown stage"):
        load_config(path)
