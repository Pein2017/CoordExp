from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig
from src.sft import (
    _assert_latest_detection_runtime_supported,
    _latest_detection_runtime_custom_shim,
    _resolve_recursive_detection_ce_cfg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = REPO_ROOT / "src" / "sft.py"


def test_sft_resolves_recursive_detection_ce_runtime_cfg_from_latest_objective() -> None:
    cfg = _resolve_recursive_detection_ce_cfg(
        SimpleNamespace(
            objective=SimpleNamespace(
                id="recursive_detection_ce",
                variant="random_permutation_et_rmp_ce",
                trie_support_weight=2.0,
                trie_balance_weight=1.0,
            )
        )
    )

    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.trie_support_weight == pytest.approx(2.0)
    assert cfg.trie_balance_weight == pytest.approx(1.0)


def test_sft_rejects_unsupported_recursive_detection_ce_variant() -> None:
    with pytest.raises(ValueError, match="random_permutation_et_rmp_ce"):
        _resolve_recursive_detection_ce_cfg(
            SimpleNamespace(
                objective=SimpleNamespace(
                    id="recursive_detection_ce",
                    variant="trie_disabled_full_suffix_ce",
                    trie_support_weight=0.0,
                    trie_balance_weight=0.0,
                )
            )
        )


def test_sft_live_bootstrap_passes_recursive_ce_cfg_to_composition() -> None:
    tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    compose_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compose_trainer_class"
    ]

    assert compose_calls
    assert any(
        kw.arg == "recursive_detection_ce_cfg"
        for call in compose_calls
        for kw in call.keywords
    )


def test_sft_live_bootstrap_attaches_recursive_ce_cfg_to_trainer() -> None:
    tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    setattr_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setattr"
    ]

    assert any(
        len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "recursive_detection_ce_cfg"
        for call in setattr_calls
    )


def test_sft_fails_fast_if_coord_offset_hooks_are_missing_after_peft_wrap() -> None:
    tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    raise_messages: list[str] = []
    for ast_node in ast.walk(tree):
        if not isinstance(ast_node, ast.Raise):
            continue
        exc = ast_node.exc
        if not isinstance(exc, ast.Call):
            continue
        if not isinstance(exc.func, ast.Name) or exc.func.id != "RuntimeError":
            continue
        if not exc.args:
            continue
        message = exc.args[0]
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            raise_messages.append(message.value)

    assert any(
        "coord_offset_adapter not found after prepare_model" in message
        for message in raise_messages
    )


def test_sft_live_bootstrap_can_construct_latest_detection_dataset() -> None:
    tree = ast.parse(SFT_PATH.read_text(encoding="utf-8"))
    from_jsonl_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_jsonl"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DetectionTrainingDataset"
    ]

    assert from_jsonl_calls
    assert any(
        kw.arg == "swift_template"
        for call in from_jsonl_calls
        for kw in call.keywords
    )


def test_latest_detection_runtime_shim_preserves_trainable_token_rows() -> None:
    config_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(config_path))

    assert isinstance(cfg, LatestDetectionTrainingConfig)
    custom_config = _latest_detection_runtime_custom_shim(cfg)

    assert custom_config.trainable_token_rows is cfg.token_rows
    assert custom_config.trainable_token_rows.enabled is True
    assert "coord_geometry" in custom_config.trainable_token_rows.groups
    assert getattr(custom_config.coord_offset, "enabled", None) is False


def test_sft_rejects_latest_recursive_detection_packing_preflight_config() -> None:
    config_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_packing_unsupported.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(config_path))

    assert isinstance(cfg, LatestDetectionTrainingConfig)
    with pytest.raises(ValueError, match="packing=false"):
        _assert_latest_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
        )
