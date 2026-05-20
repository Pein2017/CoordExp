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
        profiles.extend(
            sorted(
                path
                for path in (stage2_root / kind).glob("*.yaml")
                if not path.name.startswith("common_")
            )
        )

    assert profiles, "Expected stage2_two_channel canonical profile leaves under prod/ and smoke/."

    for path in profiles:
        # `load_materialized_training_config` is intentionally side-effect free.
        ConfigLoader.load_materialized_training_config(str(path))


def test_stage2_pseudo_positive_prod_profile_materializes_default_k4_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prod_cfg = ConfigLoader.load_materialized_training_config(
        str(
            repo_root
            / "configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml"
        )
    )

    stage2_ab = prod_cfg.stage2_ab
    assert stage2_ab is not None
    assert stage2_ab.channel_b.pseudo_positive.enabled is True
    assert stage2_ab.channel_b.pseudo_positive.coord_weight == pytest.approx(0.3)
    assert stage2_ab.channel_b.triage_posterior.num_rollouts == 4
    assert stage2_ab.channel_b.duplicate_control.iou_threshold == pytest.approx(0.95)
    assert stage2_ab.channel_b.duplicate_control.center_radius_scale == pytest.approx(0.8)
    assert stage2_ab.schedule.b_ratio == pytest.approx(0.85)
    assert (
        stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier
        == pytest.approx(3.0)
    )

    prod_objective = {m.name: m for m in prod_cfg.stage2_ab.pipeline.objective}
    assert list(prod_objective) == ["token_ce", "bbox_geo", "bbox_size_aux", "coord_reg"]
    assert prod_objective["token_ce"].config["rollout_fn_desc_weight"] == pytest.approx(1.5)

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
    assert "Stage-2 canonical prod/smoke/ablation profiles must resolve the required training keys" in msg
    # A few representative dotted paths from the spec-required list.
    assert "model.model" in msg
    assert "training.output_dir" in msg
    assert "training.logging_dir" in msg
    assert "training.learning_rate" in msg


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
            "output_root": "./output",
            "logging_root": "./tb",
            "artifact_subdir": "stage2/test",
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
                        "application": {"preset": "anchor_text_only"},
                        "config": {
                            "desc_ce_weight": 1.0,
                            "rollout_fn_desc_weight": 1.0,
                            "rollout_global_prefix_struct_ce_weight": 1.0,
                        },
                    },
                    {
                        "name": "bbox_geo",
                        "enabled": True,
                        "weight": 0.0,
                        "channels": ["A", "B"],
                        "application": {"preset": "anchor_only"},
                        "config": {
                            "smoothl1_weight": 0.0,
                            "ciou_weight": 0.0,
                        },
                    },
                    {
                        "name": "bbox_size_aux",
                        "enabled": True,
                        "weight": 0.0,
                        "channels": ["A", "B"],
                        "application": {"preset": "anchor_only"},
                        "config": {
                            "log_wh_weight": 0.0,
                            "oversize_penalty_weight": 0.0,
                            "oversize_area_frac_threshold": None,
                            "oversize_log_w_threshold": None,
                            "oversize_log_h_threshold": None,
                            "eps": 1e-6,
                        },
                    },
                    {
                        "name": "coord_reg",
                        "enabled": True,
                        "weight": 0.0,
                        "channels": ["A", "B"],
                        "application": {"preset": "anchor_only"},
                        "config": {
                            "coord_ce_weight": 0.0,
                            "coord_gate_weight": 0.0,
                            "text_gate_weight": 0.0,
                            "soft_ce_weight": 0.0,
                            "w1_weight": 0.0,
                            "temperature": 1.0,
                            "target_sigma": 2.0,
                            "target_truncate": None,
                        },
                    },
                ],
                "diagnostics": [],
            },
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
