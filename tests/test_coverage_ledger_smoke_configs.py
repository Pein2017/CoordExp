from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import src.config.loader as config_loader
from src.config.loader import ConfigLoader
from src.common.model_paths import canonical_coordexp_repo_root
from src.config.schema import DetectionTrainingConfig
from src.config.strict_dataclass import dataclass_asdict_no_none
from src.detection.dataset import DetectionTrainingDataset
from src.detection.dataset_selection import select_dataset_row_indices
from src.detection.runtime import (
    build_detection_runtime_custom_shim,
    detection_mode,
    resolve_detection_prompts,
)
from src.training.coverage_ledger.preflight import (
    build_coverage_ledger_preflight_training_dataset,
    resolve_coverage_ledger_train_selection,
)
from src.training.coverage_ledger.sidecars import CoverageLedgerSidecar
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY
from test_detection_training_dataset import (
    FakeSwiftTemplate,
    ImageExpandingSwiftTemplate,
    _ensure_image,
    _raw_row,
    _write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "configs/stage1/detection_teacher_forcing/smoke"
PROD_ROOT = REPO_ROOT / "configs/stage1/detection_teacher_forcing/prod"
BASELINE_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128_baseline.yaml"
LEDGER_CONFIG = SMOKE_ROOT / "coverage_ledger_closed_hard_sft_128.yaml"
PROD_LEDGER_CONFIG = PROD_ROOT / "coverage_ledger_closed_hard_sft.yaml"

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


class _ImageGridExpandingSwiftTemplate(ImageExpandingSwiftTemplate):
    def encode(
        self,
        payload: Mapping[str, Any],
        *,
        return_length: bool,
    ) -> dict[str, Any]:
        encoded = super().encode(payload, return_length=return_length)
        encoded["image_grid_thw"] = (1, 8, 8)
        return encoded


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
    assert resolved["objective"]["target_ir"]["rollin_policy"]["name"] == "sorted"
    assert resolved["sample_factory"]["target_sequence"]["object_ordering"] == "sorted"

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
    assert resolved["training"]["save_strategy"] == "steps"
    assert resolved["training"]["save_steps"] == 128
    assert resolved["training"]["eval_strategy"] == "steps"
    assert resolved["training"]["eval_steps"] == 128

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
    assert baseline_args.model_type == "qwen3_vl"
    assert ledger_args.model_type == "qwen3_vl"
    assert baseline_args.model == ledger_args.model
    assert baseline_args.model == str(
        canonical_coordexp_repo_root()
        / "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
    )


def test_coverage_ledger_smoke_effective_batch_one_requires_single_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_loader, "TrainArguments", _FakeTrainArguments)
    monkeypatch.setattr(config_loader, "get_dist_setting", lambda: (0, 0, 8, 0))

    for path in (BASELINE_CONFIG, LEDGER_CONFIG):
        cfg = ConfigLoader.load_materialized_training_config(str(path))
        assert isinstance(cfg, DetectionTrainingConfig)
        with pytest.raises(ValueError, match=r"effective_batch_size.*divisible"):
            ConfigLoader.build_train_arguments(cfg)


def test_coverage_ledger_prod_config_resolves_batch32_on_eight_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_loader, "TrainArguments", _FakeTrainArguments)
    monkeypatch.setattr(config_loader, "get_dist_setting", lambda: (0, 0, 8, 0))

    cfg = ConfigLoader.load_materialized_training_config(str(PROD_LEDGER_CONFIG))
    assert isinstance(cfg, DetectionTrainingConfig)
    args = ConfigLoader.build_train_arguments(cfg)

    assert cfg.training["per_device_train_batch_size"] == 1
    assert cfg.training["effective_batch_size"] == 32
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert cfg.objective.terms.coverage_ledger.enabled is True


def test_coverage_ledger_prod_config_fails_for_impossible_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_loader, "TrainArguments", _FakeTrainArguments)
    monkeypatch.setattr(config_loader, "get_dist_setting", lambda: (0, 0, 6, 0))

    cfg = ConfigLoader.load_materialized_training_config(str(PROD_LEDGER_CONFIG))
    assert isinstance(cfg, DetectionTrainingConfig)
    with pytest.raises(ValueError, match=r"effective_batch_size.*world_size"):
        ConfigLoader.build_train_arguments(cfg)


def test_coverage_ledger_smoke_configs_load_real_swift_train_arguments() -> None:
    expected_model = str(
        canonical_coordexp_repo_root()
        / "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
    )
    for path in (BASELINE_CONFIG, LEDGER_CONFIG):
        args, cfg = ConfigLoader.load_training_config(str(path))

        assert isinstance(cfg, DetectionTrainingConfig)
        assert args.model == expected_model
        assert args.model_type == "qwen3_vl"
        assert args.gradient_accumulation_steps == 1
        assert cfg.detection_template.id == "compact_object_box_closed"
        assert cfg.sample_factory.target_sequence.object_ordering == "sorted"
        assert cfg.objective.target_ir.rollin_policy.name == "sorted"


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


def test_coverage_ledger_preflight_dataset_matches_smoke_training_samples(
    tmp_path: Path,
) -> None:
    ledger_cfg = ConfigLoader.load_materialized_training_config(str(LEDGER_CONFIG))
    assert isinstance(ledger_cfg, DetectionTrainingConfig)
    selection = resolve_coverage_ledger_train_selection(ledger_cfg)
    selected_row_indices = select_dataset_row_indices(
        total_rows=256,
        selection=selection,
    )
    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row() for _ in range(256)])
    _ensure_image(tmp_path)

    preflight_template = _ImageGridExpandingSwiftTemplate()
    training_template = _ImageGridExpandingSwiftTemplate()
    preflight_dataset = build_coverage_ledger_preflight_training_dataset(
        ledger_cfg,
        train_jsonl_path=jsonl_path,
        swift_template=preflight_template,
        selection=selection,
        image_root=tmp_path / "image-root",
    )

    system_prompt, _user_prompt = resolve_detection_prompts(ledger_cfg)
    custom_config = build_detection_runtime_custom_shim(ledger_cfg)
    training_dataset = DetectionTrainingDataset.from_jsonl(
        jsonl_path,
        swift_template=training_template,
        image_root=tmp_path / "image-root",
        detection_template_id=ledger_cfg.detection_template.id,
        mode=detection_mode(ledger_cfg),
        object_ordering=ledger_cfg.sample_factory.target_sequence.object_ordering,
        user_prompt=custom_config.user_prompt,
        system_prompt=system_prompt,
        seed=int(ledger_cfg.training["seed"]),
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        object_field_order=str(
            ledger_cfg.sample_factory.target_sequence.object_field_order
        ),
        teacher_forcing_profile=str(ledger_cfg.objective.profile),
        teacher_forcing_rollin_policy=str(
            ledger_cfg.objective.target_ir.rollin_policy.name
        ),
        teacher_forcing_rollin_base_seed=int(
            ledger_cfg.objective.target_ir.rollin_policy.base_seed
        ),
        coverage_ledger_enabled=True,
        sample_limit=selection.count,
        sample_selection=selection,
        dataset_name="detection_train",
    )

    assert preflight_dataset.source_row_indices == selected_row_indices
    assert preflight_dataset.source_row_indices == training_dataset.source_row_indices
    assert preflight_dataset.config.teacher_forcing_rollin_policy == "sorted"
    assert training_dataset.config.teacher_forcing_rollin_policy == "sorted"
    preflight_sample = preflight_dataset[0]
    training_sample = training_dataset[0]

    assert preflight_sample["dataset"] == training_sample["dataset"] == "detection_train"
    assert preflight_sample["base_idx"] == training_sample["base_idx"]
    assert preflight_sample["sample_id"] == training_sample["sample_id"]
    assert (
        preflight_sample["detection_metadata"]["object_ordering_seed"]
        == training_sample["detection_metadata"]["object_ordering_seed"]
    )
    assert (
        preflight_sample["detection_metadata"]["realized_source_object_indices"]
        == training_sample["detection_metadata"]["realized_source_object_indices"]
    )

    preflight_ir = preflight_sample[TEACHER_FORCING_TARGET_IR_KEY]
    training_ir = training_sample[TEACHER_FORCING_TARGET_IR_KEY]
    assert preflight_ir.metadata["stable_sample_id"] == training_ir.metadata[
        "stable_sample_id"
    ]
    assert preflight_ir.metadata["rollin_seed"] == training_ir.metadata["rollin_seed"]
    assert (
        preflight_ir.metadata["selected_source_object_indices"]
        == training_ir.metadata["selected_source_object_indices"]
    )

    preflight_sidecar = preflight_sample["training_sidecars"].supervision.payloads[0]
    training_sidecar = training_sample["training_sidecars"].supervision.payloads[0]
    assert isinstance(preflight_sidecar, CoverageLedgerSidecar)
    assert isinstance(training_sidecar, CoverageLedgerSidecar)
    assert preflight_sidecar.sample_id == training_sidecar.sample_id
    assert preflight_sidecar.prompt_end_position == training_sidecar.prompt_end_position
    assert preflight_sidecar.object_entries == training_sidecar.object_entries
