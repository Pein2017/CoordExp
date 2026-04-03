from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader


def test_stage2_ab_canonical_profiles_load_under_current_hierarchy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stage2_root = repo_root / "configs" / "stage2_two_channel"

    profiles: list[Path] = []
    for kind in ("prod", "smoke", "ablation"):
        profiles.extend(
            sorted(
                path
                for path in (stage2_root / kind).glob("*.yaml")
                if not path.name.startswith("common_")
            )
        )

    assert profiles, "Expected stage2_two_channel canonical profile leaves under prod/, smoke/, and ablation/."

    for path in profiles:
        # `load_materialized_training_config` is intentionally side-effect free.
        ConfigLoader.load_materialized_training_config(str(path))


@pytest.mark.parametrize(
    ("config_relpath", "expected_ordering"),
    [
        (
            "configs/stage2_two_channel/ablation/a_only_iter1-res_1024.yaml",
            "random",
        ),
    ],
)
def test_stage2_ablation_profiles_pin_cache_parity_and_ordering(
    config_relpath: str,
    expected_ordering: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(str(repo_root / config_relpath))

    assert cfg.training["seed"] == 17
    assert cfg.training["encoded_sample_cache"]["enabled"] is False
    assert cfg.custom.object_ordering == expected_ordering
    assert expected_ordering in cfg.training["run_name"]
    assert expected_ordering in cfg.training["output_dir"]
    assert expected_ordering in cfg.training["logging_dir"]

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
    assert prod_objective["token_ce"].config["rollout_fn_desc_weight"] == pytest.approx(1.5)
    assert prod_objective["loss_duplicate_burst_unlikelihood"].weight == pytest.approx(2.0)


def test_lvis_stage2_entry_config_uses_federated_prompt_and_sorted_desc_first_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader.load_materialized_training_config(
        str(repo_root / "configs/stage2_two_channel/lvis_bbox_max60_1024.yaml")
    )

    assert (
        cfg.model["model"]
        == "output/stage1/lvis_bbox_max60_1024/hard_ce_soft_ce_w1_ciou_bbox_size-ckpt_232-merged"
    )
    assert cfg.custom.train_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/train.coord.jsonl"
    assert cfg.custom.val_jsonl == "public_data/lvis/rescale_32_1024_bbox_max60/val.coord.jsonl"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.custom.object_ordering == "sorted"
    assert cfg.custom.extra["prompt_variant"] == "lvis_stage2_federated"
    assert cfg.rollout_matching.eval_detection.metrics == "f1ish"
    assert (
        cfg.training["artifact_subdir"]
        == "stage2_ab/lvis_bbox_max60_1024_continued_to_ckpt_232"
    )
    assert (
        cfg.training["output_dir"]
        == "./output/stage2_ab/lvis_bbox_max60_1024_continued_to_ckpt_232"
    )
    assert (
        cfg.training["logging_dir"]
        == "./tb/stage2_ab/lvis_bbox_max60_1024_continued_to_ckpt_232"
    )


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
                        "name": "loss_duplicate_burst_unlikelihood",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["B"],
                        "application": {"preset": "rollout_only"},
                        "config": {},
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
                            "adjacent_repulsion_weight": 0.0,
                            "adjacent_repulsion_filter_mode": "same_desc",
                            "adjacent_repulsion_margin_ratio": 0.05,
                            "adjacent_repulsion_copy_margin": 0.8,
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
