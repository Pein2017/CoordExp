from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.artifacts.json_values import validate_json_value

from scripts.research.materialize_human13_row_contrast_successor import (
    SuccessorMaterializationError,
    load_successor_config,
    materialize_plans,
    successor_model_config,
)


ROOT = Path("configs/coordexp_swift/research/human13_row_contrast_successor")


def test_r1_r2_configs_are_distinct_low_dose_exact_plans() -> None:
    receipt = materialize_plans((ROOT / "01_r1.yaml", ROOT / "02_r2.yaml"))
    assert receipt["actions"] == {
        "model_loads": 0,
        "forwards": 0,
        "backwards": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "gpu_allocations": 0,
    }
    assert [plan["arm_id"] for plan in receipt["plans"]] == ["R1", "R2"]
    assert len({plan["output_root"] for plan in receipt["plans"]}) == 2
    assert all(plan["milestones"] == [0, 1, 2] for plan in receipt["plans"])
    validate_json_value(receipt)


def test_successor_model_projection_changes_no_live_surface_setting() -> None:
    config = load_successor_config(ROOT / "02_r2.yaml")
    projected = successor_model_config(config, repo_root=Path.cwd())
    assert projected.arm_id == "R2"
    assert projected.unit_id == config.unit_id
    assert projected.milestones == (0, 1, 2)
    assert projected.optimizer.learning_rate == 1.0e-5
    assert projected.trainable_surface.language_tower_dora is True
    assert projected.trainable_surface.vision_tower is False


def test_config_validation_fails_closed_on_scientific_drift() -> None:
    config = load_successor_config(ROOT / "01_r1.yaml")
    with pytest.raises(SuccessorMaterializationError):
        from scripts.research.materialize_human13_row_contrast_successor import (
            _validate_config,
        )

        _validate_config(replace(config, rectangle_margin=0.1), repo_root=Path.cwd())
