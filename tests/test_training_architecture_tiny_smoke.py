from __future__ import annotations

import shlex
from pathlib import Path

from src.config.loader import ConfigLoader

from helpers.training_architecture_fixture_builder import (
    load_fixture,
    run_tiny_fake_backward_smoke,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "training_architecture"
REAL_BACKEND_SMOKE_COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 "
    "config=configs/stage1/detection_teacher_forcing/smoke/compact_tiny.yaml "
    "gpus=0 conda run -n ms bash scripts/train.sh"
)


def test_tiny_fake_forward_backward_smoke_has_finite_loss_and_gradient() -> None:
    source = load_fixture(FIXTURE_DIR / "compact_full_stage1_source.json")

    result = run_tiny_fake_backward_smoke(source)

    assert result["loss_is_finite"] is True
    assert result["loss"] > 0.0
    assert result["gradient_nonzero"] is True
    assert result["objective_ids"] == ["coord_soft_ce", "trie_ce"]
    assert result["objective_span_counts"] == {
        "coord_soft_ce": 1,
        "trie_ce": 1,
    }
    assert result["row_gradient_sums"]["coordinate"] > 0.0
    assert result["row_gradient_sums"]["free_text"] > 0.0


def test_real_backend_smoke_command_is_documented_and_loadable() -> None:
    assert "conda run -n ms bash scripts/train.sh" in REAL_BACKEND_SMOKE_COMMAND
    assert "configs/stage1/detection_teacher_forcing/smoke/compact_tiny.yaml" in (
        REAL_BACKEND_SMOKE_COMMAND
    )
    assert "compact_tiny.yaml" in REAL_BACKEND_SMOKE_COMMAND
    command_parts = shlex.split(REAL_BACKEND_SMOKE_COMMAND)
    config_arg = next(part for part in command_parts if part.startswith("config="))
    config_path = Path(config_arg.removeprefix("config="))

    assert config_path.is_file()
    ConfigLoader.load_materialized_training_config(str(config_path))
