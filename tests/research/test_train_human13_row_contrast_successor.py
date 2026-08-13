from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.build_human13_row_contrast_successor import load_ledger
from scripts.research.materialize_human13_row_contrast_successor import (
    load_successor_config,
)
from scripts.research.train_human13_row_contrast_successor import (
    SuccessorLiveError,
    build_successor_schedule,
    execute_cli,
    project_ledger,
    select_training_ledger,
)


ROOT = Path("configs/coordexp_swift/research/human13_row_contrast_successor")


def test_two_exposure_schedule_repeats_one_complete_pack_window() -> None:
    schedule = build_successor_schedule(pack_count=7, max_updates=2)
    assert schedule.resolved_max_steps == 2
    assert schedule.runtime_batch.resolved_grad_accum_steps == 7
    assert [event.planned_step_id for event in schedule.events["checkpoint"]] == [
        1,
        2,
    ]


def test_dry_run_binds_r2_without_model_or_gpu_actions() -> None:
    receipt = execute_cli(
        config_path=ROOT / "02_r2.yaml",
        repo_root=Path.cwd(),
        execute=False,
        authority=False,
        max_updates=2,
        vertical_image_id=None,
    )
    assert receipt["arm_id"] == "R2"
    assert receipt["actions"]["model_loads"] == 0
    assert receipt["actions"]["gpu_allocations"] == 0
    assert receipt["model_plan"]["milestones"] == (0, 1, 2)


def test_vertical_projection_is_fixed_to_image_14038_events() -> None:
    config = load_successor_config(ROOT / "01_r1.yaml")
    training = select_training_ledger(load_ledger(config.ledger_path), config)
    assert len(training.events) == 12
    assert all(event.source_kind == "manifest" for event in training.events)
    assert all(
        len(group.rows) == 1
        for event in training.events
        for group in event.candidate_groups
    )
    ledger = project_ledger(training, image_id=14038)
    assert ledger.events
    assert {event.image_id for event in ledger.events} == {14038}
    assert all(item.owner_id.startswith("gt:14038:") for item in ledger.positive_rows)


def test_live_execution_requires_explicit_authority() -> None:
    with pytest.raises(SuccessorLiveError, match="authority"):
        execute_cli(
            config_path=ROOT / "01_r1.yaml",
            repo_root=Path.cwd(),
            execute=True,
            authority=False,
            max_updates=1,
            vertical_image_id=14038,
        )


def test_direct_cli_dry_run_is_runtime_free() -> None:
    script = Path("scripts/research/train_human13_row_contrast_successor.py").resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str((ROOT / "01_r1.yaml").resolve()),
            "--repo-root",
            str(Path.cwd()),
            "--max-updates",
            "1",
            "--vertical-image-id",
            "14038",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["mode"] == "dry_run"
    assert not any(receipt["actions"].values())
