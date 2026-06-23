from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import src.config.loader as config_loader
from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig
from src.config.strict_dataclass import dataclass_asdict_no_none
from src.detection.dataset import DetectionTrainingDataset
from src.detection.dataset_selection import select_dataset_row_indices
from src.training.coverage_ledger.preflight import (
    resolve_coverage_ledger_train_selection,
)
from test_detection_training_dataset import FakeSwiftTemplate, _raw_row, _write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "configs/stage1/detection_teacher_forcing/smoke"
BASELINE_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128_baseline.yaml"
LEDGER_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128.yaml"

STRUCTURAL_TOKENS = (
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
)

EXPECTED_STRUCTURAL_IDS = {
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
}

ALLOWED_BASELINE_LEDGER_DIFFS = {
    "/objective/terms/coverage_ledger/enabled",
    "/objective/terms/coverage_ledger/coverage_weight",
    "/objective/terms/coverage_ledger/region_anchor_weight",
    "/objective/terms/coverage_ledger/ledger_projection_dim",
    "/objective/terms/coverage_ledger/temperature",
    "/objective/terms/coverage_ledger/normalize_eps",
    "/objective/terms/coverage_ledger/pos_weight",
    "/objective/terms/coverage_ledger/log_auc",
    "/objective/terms/coverage_ledger/log_accuracy",
    "/objective/terms/coverage_ledger/overlay_sample_count",
    "/objective/terms/coverage_ledger/smoke_sample_count",
    "/objective/terms/coverage_ledger/smoke_sample_seed",
    "/training/run_name",
    "/training/artifact_subdir",
    "/training/output_dir",
    "/training/logging_dir",
    "/debug/train_artifact_subdir",
    "/debug/val_artifact_subdir",
    "/debug/preflight_artifact_subdir",
}

EXPECTED_TRAIN_SAMPLE_SELECTION = {
    "algorithm": "seeded_random_without_replacement_v0",
    "count": 128,
    "seed": 20260623,
}


def _load_resolved(path: Path) -> dict[str, Any]:
    resolved = ConfigLoader.load_yaml_with_extends(str(path))
    assert isinstance(resolved, dict)
    cfg = ConfigLoader.load_materialized_training_config(str(path))
    assert isinstance(cfg, DetectionTrainingConfig)
    return resolved


def _load_materialized_payload(path: Path) -> dict[str, Any]:
    cfg = ConfigLoader.load_materialized_training_config(str(path))
    assert isinstance(cfg, DetectionTrainingConfig)
    return dataclass_asdict_no_none(cfg)


@dataclass(init=False)
class _FakeTrainArguments:
    train_type: str | None = None
    tuner_type: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.training_args = self


def _load_runtime_args(path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeTrainArguments:
    monkeypatch.setattr(config_loader, "TrainArguments", _FakeTrainArguments)
    cfg = ConfigLoader.load_materialized_training_config(str(path))
    assert isinstance(cfg, DetectionTrainingConfig)
    args = ConfigLoader.build_train_arguments(cfg)
    assert isinstance(args, _FakeTrainArguments)
    return args


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


def _assert_shared_closed_hard_sft_smoke_config(resolved: dict[str, Any]) -> None:
    assert resolved["pipeline"]["id"] == "stage1_research_teacher_forcing"
    assert resolved["detection_template"]["id"] == "compact_object_box_closed"
    assert resolved["evaluation"]["expected_template"] == "compact_object_box_closed"
    assert resolved["objective"]["id"] == "research_teacher_forcing"
    assert resolved["objective"]["profile"] == "hard_sft"

    assert resolved["training"]["packing"] is False
    assert resolved["training"]["eval_packing"] is False
    assert resolved["packing"]["static_packing"] is False
    assert resolved["packing"]["padding_free_packed"] is False

    assert resolved["debug"]["train_sample_limit"] == 128
    assert resolved["debug"]["val_sample_limit"] == 128
    assert resolved["debug"]["train_sample_selection"] == EXPECTED_TRAIN_SAMPLE_SELECTION
    assert resolved["training"]["seed"] == 20260623
    assert resolved["training"]["max_steps"] == 256
    assert resolved["training"]["per_device_train_batch_size"] == 1
    assert resolved["training"]["effective_batch_size"] == 1

    group = resolved["token_embeddings_adapter"]["groups"]["compact_structure"]
    assert tuple(group["tokens"]) == STRUCTURAL_TOKENS
    assert group["expected_ids"] == EXPECTED_STRUCTURAL_IDS


def test_coverage_ledger_smoke_config_pair_loads_closed_hard_sft_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _load_resolved(BASELINE_CONFIG)
    ledger = _load_resolved(LEDGER_CONFIG)

    _assert_shared_closed_hard_sft_smoke_config(baseline)
    _assert_shared_closed_hard_sft_smoke_config(ledger)

    assert baseline["objective"]["terms"]["coverage_ledger"]["enabled"] is False
    assert ledger["objective"]["terms"]["coverage_ledger"] == {
        "enabled": True,
        "coverage_weight": 0.1,
        "region_anchor_weight": 0.1,
        "ledger_projection_dim": 256,
        "temperature": 0.2,
        "normalize_eps": 1.0e-6,
        "pos_weight": 1.0,
        "log_auc": True,
        "log_accuracy": True,
        "overlay_sample_count": 16,
        "smoke_sample_count": 128,
        "smoke_sample_seed": 20260623,
    }

    baseline_args = _load_runtime_args(BASELINE_CONFIG, monkeypatch)
    ledger_args = _load_runtime_args(LEDGER_CONFIG, monkeypatch)
    assert baseline_args.gradient_accumulation_steps == 1
    assert ledger_args.gradient_accumulation_steps == 1


def test_coverage_ledger_smoke_config_pair_only_differs_on_allowlisted_fields() -> None:
    baseline = _load_materialized_payload(BASELINE_CONFIG)
    ledger = _load_materialized_payload(LEDGER_CONFIG)

    changed_paths = _changed_paths(baseline, ledger)

    assert changed_paths <= ALLOWED_BASELINE_LEDGER_DIFFS

    for top_level_key in (
        "model",
        "template",
        "pipeline",
        "sample_factory",
        "prompt",
        "detection_template",
        "token_embeddings_adapter",
        "packing",
        "evaluation",
        "validation",
        "data",
        "deepspeed",
    ):
        assert ledger[top_level_key] == baseline[top_level_key], top_level_key

    for training_key in (
        "per_device_train_batch_size",
        "effective_batch_size",
        "max_steps",
        "seed",
        "packing",
        "eval_packing",
    ):
        assert ledger["training"][training_key] == baseline["training"][training_key]


def test_coverage_ledger_smoke_training_selection_matches_preflight_manifest_indices(
    tmp_path: Path,
) -> None:
    baseline_cfg = ConfigLoader.load_materialized_training_config(str(BASELINE_CONFIG))
    ledger_cfg = ConfigLoader.load_materialized_training_config(str(LEDGER_CONFIG))
    assert isinstance(baseline_cfg, DetectionTrainingConfig)
    assert isinstance(ledger_cfg, DetectionTrainingConfig)

    baseline_selection = resolve_coverage_ledger_train_selection(baseline_cfg)
    ledger_selection = resolve_coverage_ledger_train_selection(ledger_cfg)
    assert baseline_selection == ledger_selection

    selected_row_indices = select_dataset_row_indices(
        total_rows=256,
        selection=ledger_selection,
    )
    assert len(selected_row_indices) == 128
    assert selected_row_indices != tuple(range(128))

    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row() for _ in range(256)])
    dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        image_root=tmp_path / "image-root",
        detection_template_id="compact_object_box_closed",
        mode="random_order_sft",
        object_ordering="sorted",
        user_prompt="Detect every object.",
        system_prompt="You are a detector.",
        seed=20260623,
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        object_field_order="desc_first",
        teacher_forcing_profile="hard_sft",
        teacher_forcing_rollin_base_seed=17,
        coverage_ledger_enabled=True,
        sample_limit=128,
        sample_selection=ledger_selection,
        dataset_name="detection_train",
    )

    assert dataset.source_row_indices == selected_row_indices
