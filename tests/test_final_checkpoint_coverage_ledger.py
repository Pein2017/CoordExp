from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.trainers.final_checkpoint import _validate_adapter_checkpoint


def _write_adapter_config(checkpoint_dir: Path, *, modules_to_save: list[str]) -> None:
    (checkpoint_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "unit-base",
                "peft_type": "LORA",
                "modules_to_save": modules_to_save,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_adapter_safetensors(checkpoint_dir: Path, keys: set[str]) -> None:
    from safetensors.torch import save_file

    save_file(
        {key: torch.ones((1, 1), dtype=torch.float32) for key in keys},
        str(checkpoint_dir / "adapter_model.safetensors"),
    )


def test_adapter_checkpoint_validation_requires_complete_coverage_ledger_head(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    _write_adapter_config(
        checkpoint_dir,
        modules_to_save=["coverage_ledger_head"],
    )
    _write_adapter_safetensors(
        checkpoint_dir,
        {
            "base_model.model.coverage_ledger_head.modules_to_save.default.state_projection.weight",
            "base_model.model.coverage_ledger_head.modules_to_save.default.object_projection.weight",
        },
    )

    with pytest.raises(ValueError, match="coverage_ledger_head.*region_anchor"):
        _validate_adapter_checkpoint(checkpoint_dir)

    _write_adapter_safetensors(
        checkpoint_dir,
        {
            "base_model.model.coverage_ledger_head.modules_to_save.default.state_projection.weight",
            "base_model.model.coverage_ledger_head.modules_to_save.default.region_anchor_state_projection.weight",
            "base_model.model.coverage_ledger_head.modules_to_save.default.object_projection.weight",
        },
    )

    _validate_adapter_checkpoint(checkpoint_dir)
