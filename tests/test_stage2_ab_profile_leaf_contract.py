from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader


def test_stage2_ab_canonical_profiles_are_one_hop_and_explicit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stage2_root = repo_root / "configs" / "stage2_ab"

    profiles: list[Path] = []
    for kind in ("prod", "smoke"):
        profiles.extend(sorted((stage2_root / kind).glob("*.yaml")))

    assert profiles, "Expected stage2_ab canonical profile leaves under prod/ and smoke/."

    for path in profiles:
        # `load_materialized_training_config` is intentionally side-effect free.
        ConfigLoader.load_materialized_training_config(str(path))


def test_stage2_ab_leaf_contract_missing_required_keys_lists_dotted_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This file lives outside configs/stage2_two_channel/* so we must force the contract on.
    monkeypatch.setattr(
        ConfigLoader,
        "_canonical_stage2_profile_kind",
        lambda _path: "prod",
    )

    bad_cfg = {
        "extends": "../base.yaml",
        "model": {},
        "training": {"run_name": "x"},
        "stage2_ab": {"schedule": {"b_ratio": 0.5}},
    }
    cfg_path = tmp_path / "bad_stage2_leaf.yaml"
    cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        ConfigLoader.load_materialized_training_config(str(cfg_path))

    msg = str(exc.value)
    assert "Stage-2 canonical profile leaves must explicitly define required keys" in msg
    # A few representative dotted paths from the spec-required list.
    assert "model.model" in msg
    assert "training.output_dir" in msg
    assert "training.logging_dir" in msg
    assert "training.learning_rate" in msg
    assert "stage2_ab.n_softctx_iter" in msg


def test_stage2_ab_leaf_contract_extends_chain_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ConfigLoader,
        "_canonical_stage2_profile_kind",
        lambda _path: "prod",
    )

    bad_cfg = {
        "extends": ["../base.yaml", "../something_else.yaml"],
        "model": {"model": "x"},
        "training": {
            "run_name": "x",
            "output_dir": "out",
            "logging_dir": "tb",
            "learning_rate": 1e-5,
            "vit_lr": 1e-5,
            "aligner_lr": 1e-5,
            "effective_batch_size": 8,
            "eval_strategy": "steps",
            "eval_steps": 10,
            "save_strategy": "steps",
            "save_steps": 10,
        },
        "stage2_ab": {"schedule": {"b_ratio": 1.0}, "n_softctx_iter": 1},
    }
    cfg_path = tmp_path / "bad_stage2_extends.yaml"
    cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        ConfigLoader.load_materialized_training_config(str(cfg_path))

    msg = str(exc.value)
    assert "Stage-2 canonical profile leaves must extend exactly one parent" in msg
    assert "../base.yaml" in msg

