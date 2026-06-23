from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.training.pipeline_registry import (
    PIPELINE_IDS,
    TrainingPipelineRegistry,
)
from src.training.pipelines.stage1_compact_trie_ce import Stage1CompactTrieCEPipeline
from src.training.pipelines.stage1_json_ce import Stage1JsonCEPipeline
from src.training.pipelines.stage2_rollout_correction import (
    Stage2RolloutCorrectionPipeline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFACE_DOT_ID = "surface" + ".id"


def _pipeline_config(pipeline_id: str = "stage1_standard_sft") -> dict[str, object]:
    supervision: dict[str, object]
    objectives: dict[str, dict[str, object]]
    if pipeline_id == "stage2_rollout_correction":
        supervision = {
            "mode": "rollout_correction",
            "assignment": {"strategy": "greedy_iou"},
            "duplicate_filter": {"strategy": "rollout_correction_duplicate_control"},
            "target_ir": {"required": True},
        }
        objectives = {
            "residual_set_correction": {"enabled": True, "weight": 1.0},
        }
    elif pipeline_id == "stage1_research_teacher_forcing":
        supervision = {"mode": "compact_trie"}
        objectives = {
            "research_teacher_forcing": {
                "enabled": True,
                "weight": 1.0,
                "config": {
                    "terms": {
                        "trie_ce": {"weight": 1.0},
                        "coord_soft_ce": {"enabled": False, "weight": 0.25},
                    }
                },
            },
        }
    else:
        supervision = {"mode": "json_ce"}
        objectives = {"standard_ce": {"enabled": True, "weight": 1.0}}

    return {
        "run": {"id": "pipeline-smoke", "scope": "smoke"},
        "pipeline": {"id": pipeline_id},
        "data": {"train_jsonl": "train.jsonl", "validation_jsonl": "val.jsonl"},
        "template": {"id": "json_chat"},
        "supervision": supervision,
        "objectives": objectives,
        "observability": {"level": "minimal"},
        "artifacts": {"output_root": "output/pipeline"},
        "runtime": {"trainer": "pipeline"},
    }


def test_public_pipeline_ids_are_closed_and_ordered() -> None:
    assert PIPELINE_IDS == (
        "stage1_standard_sft",
        "stage1_research_teacher_forcing",
        "stage2_rollout_correction",
    )


@pytest.mark.parametrize(
    ("pipeline_id", "pipeline_type", "implementation_id"),
    [
        ("stage1_standard_sft", Stage1JsonCEPipeline, "stage1_json_ce"),
        (
            "stage1_research_teacher_forcing",
            Stage1CompactTrieCEPipeline,
            "stage1_compact_trie_ce",
        ),
        (
            "stage2_rollout_correction",
            Stage2RolloutCorrectionPipeline,
            "stage2_rollout_correction",
        ),
    ],
)
def test_pipeline_id_selects_pipeline_descriptor(
    pipeline_id: str,
    pipeline_type: type[object],
    implementation_id: str,
) -> None:
    resolved = TrainingPipelineRegistry().resolve(_pipeline_config(pipeline_id))

    assert isinstance(resolved.pipeline, pipeline_type)
    assert resolved.pipeline_config["id"] == pipeline_id
    assert resolved.pipeline.identity.pipeline_id == pipeline_id
    assert resolved.pipeline.identity.implementation_id == implementation_id


@pytest.mark.parametrize(
    ("relative_path", "pipeline_id", "objective_id"),
    [
        (
            "configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml",
            "stage1_research_teacher_forcing",
            "research_teacher_forcing",
        ),
        (
            "configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml",
            "stage2_rollout_correction",
            "residual_set_correction",
        ),
    ],
)
def test_registry_resolves_real_target_hierarchy_config_leaves(
    relative_path: str,
    pipeline_id: str,
    objective_id: str,
) -> None:
    payload = ConfigLoader.load_yaml_with_extends(str(REPO_ROOT / relative_path))
    assert isinstance(payload, dict)

    resolved = TrainingPipelineRegistry().resolve(payload)

    assert resolved.pipeline_config["id"] == pipeline_id
    assert [entry.objective_id for entry in resolved.objectives.enabled_objectives] == [
        objective_id
    ]


def test_public_registry_rejects_surface_domain_authoring() -> None:
    cfg = _pipeline_config()
    cfg["surface"] = {"id": "stage1_standard_sft"}
    del cfg["pipeline"]

    with pytest.raises(ValueError, match=r"surface\.id.*pipeline\.id"):
        TrainingPipelineRegistry().resolve(cfg)


def test_stage2_registry_uses_residual_set_correction_not_teacher_forcing() -> None:
    resolved = TrainingPipelineRegistry().resolve(
        _pipeline_config("stage2_rollout_correction")
    )

    assert [entry.objective_id for entry in resolved.objectives.enabled_objectives] == [
        "residual_set_correction",
    ]

    cfg = _pipeline_config("stage2_rollout_correction")
    cfg["objectives"] = {"teacher_forcing": {"enabled": True, "weight": 1.0}}

    with pytest.raises(ValueError, match=r"teacher_forcing.*residual_set_correction"):
        TrainingPipelineRegistry().resolve(cfg)


def test_registry_rejects_trainer_variant_in_runtime_domain() -> None:
    cfg = _pipeline_config("stage2_rollout_correction")
    cfg["runtime"] = {
        "trainer": "pipeline",
        "trainer_variant": "stage2_rollout_correction",
    }

    with pytest.raises(ValueError, match=r"runtime\.trainer_variant"):
        TrainingPipelineRegistry().resolve(cfg)


def test_stage2_registry_rejects_non_greedy_iou_assignment_strategy() -> None:
    cfg = _pipeline_config("stage2_rollout_correction")
    cfg["supervision"] = {
        "mode": "rollout_correction",
        "assignment": {"strategy": "hungarian"},
        "duplicate_filter": {"strategy": "rollout_correction_duplicate_control"},
        "target_ir": {"required": True},
    }

    with pytest.raises(ValueError, match=r"assignment\.strategy.*greedy_iou"):
        TrainingPipelineRegistry().resolve(cfg)


def test_stage2_registry_rejects_open_nested_supervision_shapes() -> None:
    cfg = _pipeline_config("stage2_rollout_correction")
    cfg["supervision"] = {
        "mode": "rollout_correction",
        "assignment": {"strategy": "greedy_iou", "tie_break": "mystery"},
        "duplicate_filter": {"strategy": "rollout_correction_duplicate_control"},
        "target_ir": {"required": True},
    }

    with pytest.raises(ValueError, match=r"supervision\.assignment\.tie_break"):
        TrainingPipelineRegistry().resolve(cfg)


def test_standard_sft_registry_rejects_internal_token_ce_as_public_objective() -> None:
    cfg = _pipeline_config("stage1_standard_sft")
    cfg["objectives"] = {"token_ce": {"enabled": True, "weight": 1.0}}

    with pytest.raises(ValueError, match=r"token_ce|unknown objective"):
        TrainingPipelineRegistry().resolve(cfg)


def test_research_teacher_forcing_registry_keeps_terms_internal() -> None:
    resolved = TrainingPipelineRegistry().resolve(
        _pipeline_config("stage1_research_teacher_forcing")
    )

    enabled = resolved.objectives.enabled_objectives
    assert [entry.objective_id for entry in enabled] == ["research_teacher_forcing"]
    assert enabled[0].config["terms"]["coord_soft_ce"]["enabled"] is False


def test_surfaces_module_is_not_a_long_lived_compatibility_shim() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.training.surfaces")


def test_active_public_docs_and_fixtures_do_not_author_surface_id() -> None:
    active_paths = [
        "docs/SYSTEM_OVERVIEW.md",
        "docs/AGENT_INDEX.md",
        "docs/IMPLEMENTATION_MAP.md",
        "docs/ARTIFACTS.md",
        "docs/training/README.md",
        "docs/training/STAGE1_OBJECTIVE.md",
        "docs/training/STAGE2_RUNBOOK.md",
        "docs/catalog.yaml",
        "tests/helpers/training_architecture_fixture_builder.py",
        "tests/fixtures/training_architecture/compact_full_stage1_source.json",
        "tests/fixtures/training_architecture/stage2_rollout_source.json",
        "tests/fixtures/training_architecture/compact_full_stage1_expected.json",
        "tests/fixtures/training_architecture/stage2_rollout_expected.json",
    ]

    offenders = [
        relative_path
        for relative_path in active_paths
        if SURFACE_DOT_ID
        in (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_active_training_docs_do_not_teach_internal_terms_as_public_profile_order() -> None:
    active_training_docs = [
        REPO_ROOT / "docs/training/README.md",
        REPO_ROOT / "docs/training/STAGE1_OBJECTIVE.md",
    ]

    for path in active_training_docs:
        normalized_text = " ".join(path.read_text(encoding="utf-8").split())
        assert "canonical order `token_ce`, `trie_ce`, `coord_soft_ce`" not in (
            normalized_text
        )
        assert "standard_ce" in normalized_text
        assert "research_teacher_forcing" in normalized_text
        assert "residual_set_correction" in normalized_text
