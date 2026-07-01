from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.errors import ConfigContractError
from src.config.loader import load_train_config
from src.training.schedule import (
    resolve_planned_step_schedule,
    write_resolved_step_schedule,
)

FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_debug_max_steps_define_static_schedule(tmp_path: Path) -> None:
    config = load_train_config(FIXTURE_CONFIG).config

    schedule = resolve_planned_step_schedule(
        config,
        packs_per_epoch=100,
        world_size=1,
        source_config_path=str(FIXTURE_CONFIG),
    )

    assert schedule.resolved_max_steps == 5
    assert schedule.requested_pack_presentations == 10
    assert schedule.actual_pack_presentations == 10
    assert schedule.tail_fill_pack_count == 0
    assert schedule.runtime_batch.resolved_grad_accum_steps == 2
    assert _steps(schedule, "eval.forward") == [2, 4]
    assert _steps(schedule, "training.logging") == [1, 2, 3, 4, 5]
    assert _steps(schedule, "checkpoint") == [2, 4, 5]
    assert _steps(schedule, "final") == [5]

    checkpoint_final = schedule.events["checkpoint"][-1]
    assert checkpoint_final.required is True
    assert checkpoint_final.trigger_reasons == (
        "every_fraction:0.4:clamped_final",
        "save_final",
    )
    assert checkpoint_final.deduped_from == ("save_final",)


def test_epoch_led_schedule_tail_fills_complete_optimizer_window(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "training": {
                "max_steps": None,
                "epochs": 1,
                "effective_batch_size": 4,
                "logging": {"every_fraction": None, "steps": [1, 2, 3]},
            },
            "runtime": {"backend": "single", "seed": 17},
            "checkpoint": {"every_fraction": None, "steps": [], "save_final": True},
            "eval": {"forward": {"every_fraction": None, "steps": []}},
        },
    )
    config = load_train_config(config_path).config

    schedule = resolve_planned_step_schedule(config, packs_per_epoch=10, world_size=2)

    assert schedule.resolved_max_steps == 3
    assert schedule.requested_pack_presentations == 10
    assert schedule.actual_pack_presentations == 12
    assert schedule.tail_fill_pack_count == 2
    assert schedule.runtime_batch.resolved_grad_accum_steps == 2
    assert _steps(schedule, "checkpoint") == [3]
    assert schedule.events["checkpoint"][0].trigger_reasons == ("save_final",)


def test_schedule_rejects_invalid_inputs(tmp_path: Path) -> None:
    config = load_train_config(FIXTURE_CONFIG).config

    with pytest.raises(ConfigContractError, match="packs_per_epoch"):
        resolve_planned_step_schedule(config, packs_per_epoch=0, world_size=1)

    config_path = _write_config(
        tmp_path,
        {"eval": {"forward": {"every_fraction": None, "steps": [9]}}},
    )
    with pytest.raises(ConfigContractError, match="outside"):
        resolve_planned_step_schedule(
            load_train_config(config_path).config,
            packs_per_epoch=100,
            world_size=1,
        )


def test_schedule_rejects_forward_eval_without_explicit_eval_source(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "data": {"eval": None},
            "eval": {"forward": {"every_fraction": None, "steps": [2]}},
        },
    )
    config = load_train_config(config_path).config

    with pytest.raises(ConfigContractError) as exc_info:
        resolve_planned_step_schedule(config, packs_per_epoch=10, world_size=1)

    assert exc_info.value.code == "schedule.eval_forward_source_required"


def test_schedule_artifact_write_refuses_overwrite(tmp_path: Path) -> None:
    config = load_train_config(FIXTURE_CONFIG).config
    schedule = resolve_planned_step_schedule(config, packs_per_epoch=100, world_size=1)

    output_path = write_resolved_step_schedule(schedule, tmp_path)
    payload = json.loads(output_path.read_text())

    assert payload["resolved_max_steps"] == 5
    assert payload["events"]["final"][0]["planned_step_id"] == 5
    with pytest.raises(ConfigContractError, match="already exists"):
        write_resolved_step_schedule(schedule, tmp_path)


def _steps(schedule: Any, name: str) -> list[int]:
    return [event.planned_step_id for event in schedule.events[name]]


def _write_config(tmp_path: Path, overrides: dict[str, Any]) -> Path:
    payload = yaml.safe_load(FIXTURE_CONFIG.read_text())
    _deep_update(payload, overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _deep_update(payload: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            _deep_update(payload[key], value)
        else:
            payload[key] = value
