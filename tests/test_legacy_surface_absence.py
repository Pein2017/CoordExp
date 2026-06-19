from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.config import schema
from src.config.schema import PromptOverrides, TrainingConfig
from src.training_runtime.plan import resolve_training_runtime_plan


REPO_ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_ACTIVE_PATTERNS = (
    "configs/stage1/set_continuation",
    "custom.trainer_variant: stage1_set_continuation",
    "stage1_set_continuation_et_rmp_ce",
    "src/trainers/stage1_set_continuation",
    "src/data_collators/stage1_set_continuation_collator.py",
    "branch_support_weight",
    "branch_balance_weight",
    "stage1_set_continuation_metrics_v3",
    "current-continuation",
)


ACTIVE_DOCS = (
    "docs/IMPLEMENTATION_MAP.md",
    "docs/training/README.md",
    "docs/training/STAGE1_OBJECTIVE.md",
    "docs/training/METRICS.md",
    "docs/data/PACKING.md",
    "docs/catalog.yaml",
    "docs/AGENT_INDEX.md",
)


REMOVED_IMPORT_PATHS = (
    "src.trainers.stage1_set_continuation",
    "src.data_collators.stage1_set_continuation_collator",
)


REMOVED_PUBLIC_SCHEMA_NAMES = (
    "Stage1SetContinuationConfig",
    "Stage1SetContinuationSubsetSamplingConfig",
    "Stage1SetContinuationCandidatesConfig",
    "Stage1SetContinuationObjectiveConfig",
    "Stage1SetContinuationTrainForwardConfig",
)


def _minimal_training_payload() -> dict:
    return {
        "template": {"truncation_strategy": "raise"},
        "custom": {
            "train_jsonl": "train.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
        },
    }


def test_stage1_set_continuation_is_not_active_training_variant() -> None:
    with pytest.raises(ValueError, match=r"stage1_set_continuation.*removed"):
        resolve_training_runtime_plan("stage1_set_continuation")


def test_training_config_rejects_legacy_trainer_variant() -> None:
    payload = _minimal_training_payload()
    payload["custom"]["trainer_variant"] = "stage1_set_continuation"

    with pytest.raises(ValueError, match=r"stage1_set_continuation.*removed"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_training_config_rejects_legacy_custom_block_without_materializing_config() -> None:
    payload = _minimal_training_payload()
    payload["custom"]["stage1_set_continuation"] = {}

    with pytest.raises(ValueError, match=r"custom\.stage1_set_continuation.*removed"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_legacy_set_continuation_schema_classes_are_not_public() -> None:
    for name in REMOVED_PUBLIC_SCHEMA_NAMES:
        assert not hasattr(schema, name), f"{name} remains reachable from src.config.schema"


def test_legacy_trainer_and_collator_import_paths_are_gone() -> None:
    for module_name in REMOVED_IMPORT_PATHS:
        with pytest.raises(ModuleNotFoundError, match="stage1_set_continuation"):
            importlib.import_module(module_name)


def test_active_docs_do_not_recommend_legacy_set_continuation_surface() -> None:
    joined = "\n".join(
        (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in ACTIVE_DOCS
    )

    for pattern in FORBIDDEN_ACTIVE_PATTERNS:
        assert pattern not in joined


def test_executable_legacy_stage1_set_continuation_tests_are_removed() -> None:
    legacy_named_tests = sorted(
        path.name
        for path in (REPO_ROOT / "tests").glob("*stage1_set_continuation*.py")
    )

    assert legacy_named_tests == []
