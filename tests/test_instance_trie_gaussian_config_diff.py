from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs/stage1/recursive_detection_ce_latest"
BASE_CONFIG = CONFIG_ROOT / "prod/compact_full_support2.yaml"
PROD_CONFIG = (
    CONFIG_ROOT
    / "prod/compact_full_support2_instance_trie_gaussian_softce_a5.yaml"
)
TINY_CONFIG = (
    CONFIG_ROOT
    / "smoke/compact_full_support2_instance_trie_gaussian_softce_a5_tiny.yaml"
)
DDP8_CONFIG = (
    CONFIG_ROOT
    / "smoke/compact_full_support2_instance_trie_gaussian_softce_a5_ddp8_preflight.yaml"
)

PROD_ALLOWED_CHANGED_PATHS = {
    "/objective/coord_soft_ce/enabled",
    "/objective/coord_soft_ce/target_distribution",
    "/training/run_name",
    "/training/artifact_subdir",
    "/training/save_model_only",
    "/output_dir",
    "/paths/output_root",
    "/experiment/name",
    "/experiment/tag",
    "/experiment/surface",
    "/experiment/ablation_id",
    "/experiment/claim_scope",
}

SMOKE_ALLOWED_CHANGED_PATHS = PROD_ALLOWED_CHANGED_PATHS | {
    "/training/max_steps",
    "/training/output_root",
    "/training/logging_root",
    "/training/save_strategy",
    "/training/save_last_epoch",
    "/training/eval_strategy",
    "/training/logging_steps",
    "/training/logging_first_step",
    "/training/per_device_train_batch_size",
    "/training/effective_batch_size",
    "/training/per_device_eval_batch_size",
    "/training/eval_steps",
    "/debug/enabled",
    "/debug/train_sample_limit",
    "/debug/val_sample_limit",
}

TINY_ALLOWED_CHANGED_PATHS = SMOKE_ALLOWED_CHANGED_PATHS
DDP8_ALLOWED_CHANGED_PATHS = SMOKE_ALLOWED_CHANGED_PATHS


def _resolve_config(path: Path) -> dict[str, Any]:
    resolved = ConfigLoader.load_yaml_with_extends(str(path))
    assert isinstance(resolved, dict)
    cfg = ConfigLoader.load_materialized_training_config(str(path))
    assert isinstance(cfg, LatestDetectionTrainingConfig)
    return resolved


def _escape_pointer_part(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def _join_pointer(parent: str, key: str) -> str:
    suffix = _escape_pointer_part(key)
    return f"/{suffix}" if parent == "" else f"{parent}/{suffix}"


def _leaf_paths(value: Any, parent: str) -> set[str]:
    if isinstance(value, Mapping):
        paths: set[str] = set()
        for key, child in value.items():
            paths.update(_leaf_paths(child, _join_pointer(parent, str(key))))
        return paths or {parent or "/"}
    if isinstance(value, list):
        paths = set()
        for index, child in enumerate(value):
            paths.update(_leaf_paths(child, _join_pointer(parent, str(index))))
        return paths or {parent or "/"}
    return {parent or "/"}


def _changed_paths(left: Any, right: Any, parent: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right)):
            path = _join_pointer(parent, str(key))
            if key not in left:
                paths.update(_leaf_paths(right[key], path))
            elif key not in right:
                paths.update(_leaf_paths(left[key], path))
            else:
                paths.update(_changed_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return set()
        paths = set()
        for index in range(max(len(left), len(right))):
            path = _join_pointer(parent, str(index))
            if index >= len(left):
                paths.update(_leaf_paths(right[index], path))
            elif index >= len(right):
                paths.update(_leaf_paths(left[index], path))
            else:
                paths.update(_changed_paths(left[index], right[index], path))
        return paths
    return set() if left == right else {parent or "/"}


@pytest.mark.parametrize(
    ("config_path", "allowed_paths"),
    [
        (PROD_CONFIG, PROD_ALLOWED_CHANGED_PATHS),
        (TINY_CONFIG, TINY_ALLOWED_CHANGED_PATHS),
        (DDP8_CONFIG, DDP8_ALLOWED_CHANGED_PATHS),
    ],
)
def test_instance_trie_gaussian_configs_only_change_whitelisted_paths(
    config_path: Path,
    allowed_paths: set[str],
) -> None:
    base = _resolve_config(BASE_CONFIG)
    candidate = _resolve_config(config_path)

    changed_paths = _changed_paths(base, candidate)

    assert changed_paths <= allowed_paths
    assert candidate["objective"]["coord_soft_ce"] == {
        "enabled": True,
        "target_distribution": "instance_trie_gaussian",
    }
    assert candidate["experiment"]["ablation_id"] == "A5-instance-trie-gaussian"
    assert "instance_trie_gaussian_softce_a5" in candidate["training"]["run_name"]
    assert (
        "instance_trie_gaussian_softce_a5"
        in candidate["training"]["artifact_subdir"]
    )


def test_instance_trie_gaussian_prod_keeps_fair_comparison_surfaces_equal() -> None:
    base = _resolve_config(BASE_CONFIG)
    prod = _resolve_config(PROD_CONFIG)

    for path in (
        "model",
        "template",
        "data",
        "prompt",
        "detection_template",
        "token_rows",
        "packing",
        "evaluation",
        "validation",
        "deepspeed",
    ):
        assert prod[path] == base[path], path

    for path in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "effective_batch_size",
        "per_device_eval_batch_size",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "lr_scheduler_type",
        "optimizer",
        "optim",
        "train_type",
        "packing",
        "eval_packing",
        "encoded_sample_cache",
        "eval_strategy",
        "eval_steps",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "save_delay_steps",
        "logging_steps",
        "logging_first_step",
        "metric_for_best_model",
        "greater_is_better",
        "ddp_find_unused_parameters",
        "ddp_broadcast_buffers",
    ):
        assert prod["training"][path] == base["training"][path], path
