from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader


def test_stage2_ab_canonical_profiles_load_under_current_hierarchy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stage2_root = repo_root / "configs" / "stage2_two_channel"

    profiles: list[Path] = []
    for kind in ("prod", "smoke"):
        profiles.extend(sorted((stage2_root / kind).glob("*.yaml")))

    assert profiles, "Expected stage2_two_channel canonical profile leaves under prod/ and smoke/."

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

    (tmp_path / "base.yaml").write_text("{}\n", encoding="utf-8")
    bad_cfg = {"extends": "./base.yaml", "model": {}, "training": {"run_name": "x"}}
    cfg_path = tmp_path / "bad_stage2_leaf.yaml"
    cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        ConfigLoader.load_materialized_training_config(str(cfg_path))

    msg = str(exc.value)
    assert "Stage-2 canonical prod/smoke profiles must resolve the required training keys" in msg
    # A few representative dotted paths from the spec-required list.
    assert "model.model" in msg
    assert "training.output_dir" in msg
    assert "training.logging_dir" in msg
    assert "training.learning_rate" in msg
    assert "stage2_ab.n_softctx_iter" in msg


def test_stage2_ab_leaf_contract_allows_multi_hop_when_fields_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ConfigLoader,
        "_canonical_stage2_profile_kind",
        lambda _path: "prod",
    )

    base_cfg = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "public_data/coco/rescale_32_768_bbox_max60/train.coord.jsonl",
            "val_jsonl": "public_data/coco/rescale_32_768_bbox_max60/val.coord.jsonl",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage2_two_channel",
        },
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
        "stage2_ab": {
            "schedule": {"b_ratio": 1.0},
            "pipeline": {
                "objective": [
                    {
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "config": {
                            "desc_ce_weight": 1.0,
                            "self_context_struct_ce_weight": 0.1,
                            "rollout_fn_desc_weight": 1.0,
                            "rollout_matched_prefix_struct_weight": 1.0,
                        },
                    },
                    {
                        "name": "duplicate_ul",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["B"],
                        "config": {},
                    },
                    {
                        "name": "bbox_geo",
                        "enabled": True,
                        "weight": 0.0,
                        "channels": ["A", "B"],
                        "config": {
                            "smoothl1_weight": 0.0,
                            "ciou_weight": 0.0,
                            "a1_smoothl1_weight": 0.0,
                            "a1_ciou_weight": 0.0,
                        },
                    },
                    {
                        "name": "coord_reg",
                        "enabled": True,
                        "weight": 0.0,
                        "channels": ["A", "B"],
                        "config": {
                            "coord_ce_weight": 0.0,
                            "coord_el1_weight": 0.0,
                            "coord_ehuber_weight": 0.0,
                            "coord_huber_delta": 0.001,
                            "coord_entropy_weight": 0.0,
                            "coord_gate_weight": 0.0,
                            "text_gate_weight": 0.0,
                            "soft_ce_weight": 0.0,
                            "self_context_soft_ce_weight": 0.0,
                            "w1_weight": 0.0,
                            "a1_soft_ce_weight": 0.0,
                            "a1_w1_weight": 0.0,
                            "temperature": 1.0,
                            "target_sigma": 2.0,
                            "target_truncate": None,
                        },
                    },
                ],
                "diagnostics": [],
            },
            "n_softctx_iter": 1,
        },
        "rollout_matching": {
            "rollout_backend": "hf",
            "eval_rollout_backend": "vllm",
            "channel_b_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
        },
        "model": {"model": "x"},
    }
    mid_cfg = {"extends": "./base.yaml"}
    leaf_cfg = {"extends": "./mid.yaml"}

    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    (tmp_path / "mid.yaml").write_text(yaml.safe_dump(mid_cfg), encoding="utf-8")
    cfg_path = tmp_path / "leaf.yaml"
    cfg_path.write_text(yaml.safe_dump(leaf_cfg), encoding="utf-8")

    cfg = ConfigLoader.load_materialized_training_config(str(cfg_path))

    assert cfg.model.get("model") == "x"
    assert cfg.training.get("run_name") == "x"
