from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skip(
    reason="Archived recursive_detection_ce configs are historical after DetectionScene clean-break"
)

from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce"
BASE_CONFIG = CONFIG_ROOT / "prod/compact_full_support2.yaml"
PROD_CONFIG = (
    CONFIG_ROOT
    / "prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1.yaml"
)
SLOPE_PROD_CONFIG = (
    CONFIG_ROOT
    / "prod/compact_full_support2_instance_trie_focused_cap8_frac0p06_mix0p1.yaml"
)
STRENGTH_PROD_CONFIG = (
    CONFIG_ROOT
    / "prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2.yaml"
)
TINY_CONFIG = (
    CONFIG_ROOT
    / "smoke/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1_tiny.yaml"
)

PROD_ALLOWED_CHANGED_PATHS = {
    "/objective/coord_soft_ce/enabled",
    "/objective/coord_soft_ce/target_distribution",
    "/objective/coord_soft_ce/gaussian_mixture_weight",
    "/objective/coord_soft_ce/gaussian_r95_axis_fraction",
    "/objective/coord_soft_ce/gaussian_r95_cap_bins",
    "/objective/type_gate/enabled",
    "/objective/type_gate/mode",
    "/objective/type_gate/weights/struct",
    "/objective/type_gate/weights/desc",
    "/objective/type_gate/weights/coord",
    "/objective/type_gate/weights/eos",
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
    assert isinstance(cfg, DetectionTrainingConfig)
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
        (SLOPE_PROD_CONFIG, PROD_ALLOWED_CHANGED_PATHS),
        (STRENGTH_PROD_CONFIG, PROD_ALLOWED_CHANGED_PATHS),
        (TINY_CONFIG, TINY_ALLOWED_CHANGED_PATHS),
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
    coord_soft_ce = candidate["objective"]["coord_soft_ce"]
    assert coord_soft_ce["enabled"] is True
    assert coord_soft_ce["target_distribution"] == "instance_trie_gaussian"
    type_gate = candidate["objective"]["type_gate"]
    assert type_gate["enabled"] is True
    assert type_gate["mode"] == "allowed_type_mass"
    assert type_gate["weights"] == {
        "struct": 1.0,
        "desc": 1.0,
        "coord": 1.0,
        "eos": 0.5,
    }
    assert coord_soft_ce["gaussian_r95_cap_bins"] == 8
    if "frac0p06_mix0p1" in config_path.name:
        assert coord_soft_ce["gaussian_mixture_weight"] == pytest.approx(0.1)
        assert coord_soft_ce["gaussian_r95_axis_fraction"] == pytest.approx(0.06)
        assert candidate["experiment"]["ablation_id"] == "cap8_frac0p06_mix0p1"
        assert "cap8-frac0p06-mix0p1" in candidate["training"]["run_name"]
        assert "cap8_frac0p06_mix0p1" in candidate["training"]["artifact_subdir"]
    elif "frac0p04_mix0p2" in config_path.name:
        assert coord_soft_ce["gaussian_mixture_weight"] == pytest.approx(0.2)
        assert coord_soft_ce["gaussian_r95_axis_fraction"] == pytest.approx(0.04)
        assert candidate["experiment"]["ablation_id"] == "cap8_frac0p04_mix0p2"
        assert "cap8-frac0p04-mix0p2" in candidate["training"]["run_name"]
        assert "cap8_frac0p04_mix0p2" in candidate["training"]["artifact_subdir"]
    else:
        assert coord_soft_ce == {
            "enabled": True,
            "target_distribution": "instance_trie_gaussian",
            "gaussian_mixture_weight": 0.1,
            "gaussian_r95_axis_fraction": 0.04,
            "gaussian_r95_cap_bins": 8,
        }
        assert candidate["experiment"]["ablation_id"] == "cap8_frac0p04_mix0p1"
        assert "cap8-frac0p04-mix0p1" in candidate["training"]["run_name"]
        assert "cap8_frac0p04_mix0p1" in candidate["training"]["artifact_subdir"]


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
