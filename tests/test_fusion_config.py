from __future__ import annotations

from pathlib import Path

import pytest

from src.config.schema import PromptOverrides, TrainingConfig
from src.datasets.fusion import FusionConfig


def _base_training_payload() -> dict:
    return {
        "template": {"truncation_strategy": "raise"},
        "custom": {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
        },
    }


def test_dormant_fusion_examples_remain_parseable_for_future_reactivation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_paths = [
        repo_root / "configs/fusion/examples/lvis_vg.yaml",
        repo_root / "configs/fusion/lvis_bbox_only_vs_poly_prefer_1to1.yaml",
    ]

    for path in example_paths:
        cfg = FusionConfig.from_file(str(path))
        assert cfg.datasets, f"Expected at least one dataset entry in {path}"


def test_dormant_fusion_training_yaml_is_present_but_not_supported() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "configs/fusion/sft_lvis_vg.yaml").read_text(encoding="utf-8")

    assert "DORMANT LEGACY EXAMPLE" in text
    assert "custom.fusion_config" in text


def test_training_config_still_rejects_disabled_fusion_surface() -> None:
    payload = _base_training_payload()
    payload["custom"]["fusion_config"] = "configs/fusion/examples/lvis_vg.yaml"

    with pytest.raises(ValueError, match=r"custom\.fusion_config is temporarily disabled"):
        TrainingConfig.from_mapping(payload, PromptOverrides())
