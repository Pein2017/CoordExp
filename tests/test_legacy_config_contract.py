from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from src.config.schema import PromptOverrides, TrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_payload() -> dict[str, object]:
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


def test_legacy_custom_coord_loss_is_hard_error() -> None:
    payload = _legacy_payload()
    custom = payload["custom"]
    assert isinstance(custom, dict)
    custom["coord_loss"] = {"enabled": True}

    with pytest.raises(
        ValueError,
        match=(
            r"custom\.coord_loss is no longer supported.*"
            r"custom\.coord_soft_ce_w1.*objective\.\*"
        ),
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_active_legacy_configs_do_not_rely_on_custom_coord_loss() -> None:
    config_root = REPO_ROOT / "configs"
    config_paths = sorted(config_root.rglob("*.yaml"))
    assert config_paths

    offenders: list[str] = []
    for path in config_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            continue
        custom = payload.get("custom")
        if isinstance(custom, Mapping) and "coord_loss" in custom:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_legacy_custom_extra_rollout_matching_remains_rejected() -> None:
    payload = _legacy_payload()
    custom = payload["custom"]
    assert isinstance(custom, dict)
    custom["extra"] = {"rollout_matching": {"enabled": True}}

    with pytest.raises(
        ValueError,
        match=r"custom\.extra\.rollout_matching is unsupported",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_legacy_top_level_extra_remains_rejected() -> None:
    payload = _legacy_payload()
    payload["extra"] = {"owner": "legacy"}

    with pytest.raises(
        ValueError,
        match=r"Top-level extra: is unsupported",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_legacy_training_packing_length_remains_rejected_with_guidance() -> None:
    payload = _legacy_payload()
    payload["training"] = {"packing_length": 2048}

    with pytest.raises(
        ValueError,
        match=r"training\.packing_length is deprecated.*global_max_length/template\.max_length",
    ):
        TrainingConfig.from_mapping(payload, PromptOverrides())
