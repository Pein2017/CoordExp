from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_ROOT = REPO_ROOT / "configs" / "stage2_rollout_correction"
CKPT3664_ADAPTER = (
    "/data/CoordExp/temp/baidudisk-union-sync/download-staging/outputs/CoordExp/"
    "outputs/stage1_2b/recursive_detection_ce_latest/"
    "compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/"
    "compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/"
    "v0-20260504-071356/checkpoint-3664"
)


def _stage2_profile_leaves() -> list[Path]:
    leaves: list[Path] = []
    for kind in ("prod", "smoke", "ablation"):
        root = STAGE2_ROOT / kind
        if root.is_dir():
            leaves.extend(sorted(root.glob("*.yaml")))
    return leaves


def test_stage2_rollout_correction_profiles_load_under_current_hierarchy() -> None:
    profiles = _stage2_profile_leaves()
    assert profiles, "Expected stage2_rollout_correction profile leaves."

    for path in profiles:
        ConfigLoader.load_materialized_training_config(str(path))


@pytest.mark.parametrize(
    "config_relpath",
    [
        "configs/stage2_rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml",
        "configs/stage2_rollout_correction/smoke/compact_full_hf_1step.yaml",
    ],
)
def test_stage2_rollout_correction_profiles_pin_residual_objective_only(
    config_relpath: str,
) -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        str(REPO_ROOT / config_relpath)
    )

    assert cfg.custom.trainer_variant == "stage2_rollout_correction"
    assert cfg.stage2_rollout_correction is not None
    assert getattr(cfg, "stage2_ab", None) is None
    assert "stage2_rollout_correction" in cfg.training["output_dir"]
    assert "stage2_rollout_correction" in cfg.training["logging_dir"]

    pipeline = cfg.stage2_rollout_correction.pipeline
    assert [module.name for module in pipeline.objective] == [
        "residual_set_correction"
    ]
    residual = pipeline.objective[0]
    assert residual.enabled is True
    assert residual.application["preset"] == "rollout_self_prefix"
    assert residual.config["clean_gt_sft_mix"] == 0
    assert not hasattr(cfg.stage2_rollout_correction, "schedule")


@pytest.mark.parametrize(
    "config_relpath",
    [
        "configs/stage2_rollout_correction/smoke/compact_full_hf_1step.yaml",
        "configs/stage2_rollout_correction/smoke/compact_full_vllm_train64_val32_base_control_lr0_1step_coco80_prompt.yaml",
        "configs/stage2_rollout_correction/smoke/compact_full_vllm_train64_val32_12steps_coco80_pilot.yaml",
    ],
)
def test_compact_full_ckpt3664_coco80_readiness_profiles_pin_same_surface(
    config_relpath: str,
) -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        str(REPO_ROOT / config_relpath)
    )

    assert cfg.custom.detection_sequence_format == "compact_full"
    assert cfg.custom.object_field_order == "desc_first"
    assert cfg.model.get("model") == (
        "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
    )
    assert cfg.model.get("adapters") == [CKPT3664_ADAPTER]
    assert cfg.rollout_matching.prompt_variant == "coco_80"
    assert cfg.rollout_matching.eval_prompt_variant == "coco_80"
    assert cfg.stage2_rollout_correction.correction.rollout_template_family == (
        "compact_full"
    )
    assert cfg.stage2_rollout_correction.correction.rollout_decode_policy == (
        "unconstrained"
    )
    assert cfg.stage2_rollout_correction.correction.invalid_rollout_policy == (
        "fallback_gt_fn_append_only"
    )


def test_stage2_rollout_correction_leaf_contract_missing_required_keys_lists_dotted_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert "model.model" in msg
    assert "training.output_dir" in msg
    assert "training.logging_dir" in msg
    assert "training.learning_rate" in msg
    assert "stage2_rollout_correction.pipeline.objective" in msg


def test_stage2_rollout_correction_leaf_contract_allows_multi_hop_when_fields_resolve(
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
            "trainer_variant": "stage2_rollout_correction",
        },
        "training": {
            "run_name": "x",
            "output_root": "./output",
            "logging_root": "./tb",
            "artifact_subdir": "stage2_rollout_correction/test",
            "learning_rate": 1e-5,
            "vit_lr": 1e-5,
            "aligner_lr": 1e-5,
            "effective_batch_size": 8,
            "per_device_train_batch_size": 1,
            "eval_strategy": "steps",
            "eval_steps": 10,
            "save_strategy": "steps",
            "save_steps": 10,
        },
        "stage2_rollout_correction": {
            "pipeline": {
                "objective": [
                    {
                        "name": "residual_set_correction",
                        "enabled": True,
                        "weight": 1.0,
                        "application": {"preset": "rollout_self_prefix"},
                        "config": {"clean_gt_sft_mix": 0},
                    },
                ],
                "diagnostics": [],
            },
            "correction": {},
        },
        "rollout_matching": {
            "rollout_backend": "hf",
            "eval_rollout_backend": "vllm",
            "rollout_decode_batch_size": 1,
            "eval_decode_batch_size": 1,
            "vllm": {"enable_lora": True, "sync": {"mode": "adapter"}},
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
    assert cfg.stage2_rollout_correction.pipeline.objective[0].name == (
        "residual_set_correction"
    )
