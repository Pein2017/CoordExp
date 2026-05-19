from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig, TrainingConfig

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_latest_training_config_contract import _latest_payload
from test_stage2_ab_config_contract import (
    _canonical_stage2_pipeline,
    _make_stage2_training_payload,
)


OLD_OBJECTIVE_IDS = (
    "recursive_detection_ce",
    "random_permutation_et_rmp_ce",
    "prefix_rollin_et_rmp_ce",
    "ET_RMP_CE",
    "et_rmp_like",
    "typed_trie_alpha0_random_rollin",
    "typed_trie_alpha0p1_random_rollin",
    "support_balance",
)


def _teacher_forcing_objective(
    *,
    profile: str = "pure_valid_set_marginal",
    coverage_strength: float = 0.0,
    coverage_enabled: bool = False,
    exact_packing_mapping: bool = False,
) -> dict:
    return {
        "id": "teacher_forcing",
        "profile": profile,
        "target_ir": {
            "rollin_policy": {
                "name": "random_permutation",
                "base_seed": 17,
            },
            "exact_packing_mapping": {
                "enabled": exact_packing_mapping,
            },
        },
        "modules": {
            "token_type_mass": {"enabled": True},
            "conditional_valid_set_likelihood": {"enabled": True},
            "within_valid_coverage": {
                "enabled": coverage_enabled,
                "coverage_strength": coverage_strength,
            },
            "continuation_margin": {"enabled": False},
        },
    }


def _latest_teacher_payload(**objective_updates: object) -> dict:
    payload = _latest_payload()
    objective = _teacher_forcing_objective()
    objective.update(objective_updates)
    payload["objective"] = objective
    return payload


def _stage2_teacher_payload(
    *,
    objective: dict | None = None,
    training: dict | None = None,
    pipeline: dict | None = None,
) -> dict:
    payload = _make_stage2_training_payload(training)
    payload["objective"] = objective or _teacher_forcing_objective()
    if pipeline is not None:
        payload["stage2_ab"]["pipeline"] = pipeline
    return payload


def _migrated_teacher_forcing_pipeline() -> dict:
    return {"objective": [], "diagnostics": []}


@pytest.mark.parametrize(
    "profile",
    [
        "hard_sft",
        "pure_valid_set_marginal",
        "coverage_regularized_valid_set_marginal",
    ],
)
def test_latest_teacher_forcing_accepts_supported_profiles(profile: str) -> None:
    coverage_enabled = profile == "coverage_regularized_valid_set_marginal"
    payload = _latest_teacher_payload(
        profile=profile,
        modules={
            **_teacher_forcing_objective()["modules"],
            "within_valid_coverage": {
                "enabled": coverage_enabled,
                "coverage_strength": 0.2 if coverage_enabled else 0.0,
            },
        },
    )

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == profile
    assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"
    assert cfg.objective.target_ir.rollin_policy.base_seed == 17


def test_latest_teacher_forcing_accepts_minimal_hard_sft_profile() -> None:
    payload = _latest_teacher_payload(
        profile="hard_sft",
        modules={
            "token_type_mass": {"enabled": False},
            "conditional_valid_set_likelihood": {"enabled": False},
        },
    )

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.modules.within_valid_coverage.enabled is False
    assert cfg.objective.modules.within_valid_coverage.coverage_strength == 0.0


def test_coverage_profile_requires_explicit_positive_coverage_strength() -> None:
    payload = _latest_teacher_payload(
        profile="coverage_regularized_valid_set_marginal",
        modules={
            **_teacher_forcing_objective()["modules"],
            "within_valid_coverage": {
                "enabled": True,
                "coverage_strength": 0.0,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=r"coverage_regularized_valid_set_marginal.*coverage_strength.*> 0",
    ):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_pure_profile_rejects_positive_coverage_strength() -> None:
    payload = _latest_teacher_payload(
        modules={
            **_teacher_forcing_objective()["modules"],
            "within_valid_coverage": {
                "enabled": True,
                "coverage_strength": 0.1,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=r"pure_valid_set_marginal.*coverage_strength=0",
    ):
        LatestDetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize("old_id", OLD_OBJECTIVE_IDS)
def test_latest_teacher_forcing_rejects_legacy_objective_ids(old_id: str) -> None:
    payload = _latest_payload()
    payload["objective"] = {"id": old_id}

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        LatestDetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize("old_id", OLD_OBJECTIVE_IDS)
def test_training_config_rejects_legacy_objective_ids(old_id: str) -> None:
    raw = _make_stage2_training_payload()
    raw["objective"] = {"id": old_id}

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        TrainingConfig.from_mapping(raw, prompts)


@pytest.mark.parametrize(
    "module_name",
    ["bbox_geo", "bbox_size_aux", "coord_reg", "token_ce", "coord_gate", "text_gate"],
)
def test_stage2_teacher_forcing_rejects_legacy_pipeline_modules(
    module_name: str,
) -> None:
    pipeline = _canonical_stage2_pipeline()
    pipeline["objective"][0]["name"] = module_name
    raw = _stage2_teacher_payload(pipeline=pipeline)

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=rf"teacher_forcing.*stage2_ab\.pipeline\.objective.*{module_name}",
    ):
        TrainingConfig.from_mapping(raw, prompts)


@pytest.mark.parametrize(
    "legacy_key",
    ["coord_gate_weight", "text_gate_weight", "soft_ce_weight", "w1_weight"],
)
def test_stage2_teacher_forcing_rejects_legacy_gate_config_keys(
    legacy_key: str,
) -> None:
    pipeline = _canonical_stage2_pipeline()
    pipeline["objective"][3]["config"][legacy_key] = 0.25
    raw = _stage2_teacher_payload(pipeline=pipeline)

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=rf"teacher_forcing.*stage2_ab\.pipeline\.objective.*{legacy_key}",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_teacher_forcing_rejects_packing_without_exact_mapping() -> None:
    raw = _stage2_teacher_payload(
        training={
            "per_device_train_batch_size": 1,
            "effective_batch_size": 1,
            "packing": True,
        }
    )

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=r"teacher_forcing.*training\.packing=true.*exact_packing_mapping",
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_stage2_teacher_forcing_exact_mapping_bypasses_packing_guard() -> None:
    objective = _teacher_forcing_objective(exact_packing_mapping=True)
    raw = _stage2_teacher_payload(
        objective=objective,
        training={
            "per_device_train_batch_size": 1,
            "effective_batch_size": 1,
            "packing": True,
        },
        pipeline=_migrated_teacher_forcing_pipeline(),
    )

    prompts = ConfigLoader.resolve_prompts(raw)
    cfg = TrainingConfig.from_mapping(raw, prompts)

    assert cfg.objective is not None
    assert cfg.objective.target_ir.exact_packing_mapping.enabled is True
    assert cfg.training["packing"] is True
    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.pipeline.objective == ()


def test_stage2_teacher_forcing_accepts_migrated_pipeline_without_legacy_modules() -> None:
    raw = _stage2_teacher_payload(pipeline=_migrated_teacher_forcing_pipeline())

    prompts = ConfigLoader.resolve_prompts(raw)
    cfg = TrainingConfig.from_mapping(raw, prompts)

    assert cfg.objective is not None
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.modules.conditional_valid_set_likelihood.enabled is True
    assert cfg.stage2_ab is not None
    assert cfg.stage2_ab.pipeline.objective == ()


def test_training_config_accepts_teacher_forcing_objective_without_stage2() -> None:
    raw = {
        "template": {"template": "qwen3_vl"},
        "custom": {
            "train_jsonl": "toy/train.jsonl",
            "val_jsonl": "toy/val.jsonl",
            "user_prompt": "{bbox}",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
        },
        "training": {"per_device_train_batch_size": 1, "effective_batch_size": 1},
        "objective": copy.deepcopy(_teacher_forcing_objective()),
    }

    prompts = ConfigLoader.resolve_prompts(raw)
    cfg = TrainingConfig.from_mapping(raw, prompts)

    assert cfg.objective is not None
    assert cfg.objective.id == "teacher_forcing"
