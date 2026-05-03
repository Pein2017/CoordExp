from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.sft import _resolve_recursive_detection_ce_cfg


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
