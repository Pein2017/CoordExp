from __future__ import annotations

from src.config.loader import ConfigLoader


def _assert_placeholder_datasets_present(config_path: str) -> None:
    cfg = ConfigLoader.load_materialized_training_config(config_path)
    dataset = cfg.data.get("dataset")
    val_dataset = cfg.data.get("val_dataset")

    # ms-swift validates these are non-empty, even when the real JSONL source is
    # custom.train_jsonl/custom.val_jsonl (CoordExp path).
    assert isinstance(dataset, list) and dataset
    assert isinstance(val_dataset, list) and val_dataset

    assert isinstance(cfg.custom.train_jsonl, str) and cfg.custom.train_jsonl
    assert isinstance(cfg.custom.val_jsonl, str) and cfg.custom.val_jsonl


def test_placeholder_datasets_present_for_stage1_and_stage2_anchored_configs() -> None:
    _assert_placeholder_datasets_present(
        "configs/stage1/ablation/geometry_first_coco80.yaml"
    )
    _assert_placeholder_datasets_present("configs/stage2_ab/prod/ab_mixed.yaml")

