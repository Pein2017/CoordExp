from __future__ import annotations

import copy
from dataclasses import asdict, replace
import sys
from pathlib import Path

import pytest

from src.bootstrap.pipeline_manifest import build_pipeline_manifest
from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig, TrainingConfig
from src.detection.runtime import (
    build_latest_detection_dataset,
    build_latest_detection_runtime_custom_shim,
    resolve_latest_detection_prompts,
)

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_latest_training_config_contract import _latest_payload
from test_detection_training_dataset import (
    FakeSwiftTemplate,
    _ensure_image,
    _raw_row,
    _write_jsonl,
)
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


def _teacher_forcing_stage2_objective_names(cfg: TrainingConfig) -> list[str]:
    assert cfg.stage2_ab is not None
    return [module.name for module in cfg.stage2_ab.pipeline.objective]


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


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("rollin_policy", []),
        ("rollin_policy", False),
        ("exact_packing_mapping", []),
        ("exact_packing_mapping", False),
    ],
)
def test_teacher_forcing_target_ir_nested_sections_reject_falsy_non_mappings(
    section: str,
    value: object,
) -> None:
    payload = _latest_teacher_payload()
    target_ir = payload["objective"]["target_ir"]  # type: ignore[index]
    assert isinstance(target_ir, dict)
    target_ir[section] = value

    with pytest.raises(
        TypeError,
        match=rf"objective\.target_ir\.{section} must be a mapping",
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
    assert _teacher_forcing_stage2_objective_names(cfg) == [
        "conditional_valid_set_likelihood"
    ]


def test_stage2_teacher_forcing_accepts_migrated_pipeline_without_legacy_modules() -> None:
    raw = _stage2_teacher_payload(pipeline=_migrated_teacher_forcing_pipeline())

    prompts = ConfigLoader.resolve_prompts(raw)
    cfg = TrainingConfig.from_mapping(raw, prompts)

    assert cfg.objective is not None
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.modules.conditional_valid_set_likelihood.enabled is True
    assert cfg.stage2_ab is not None
    assert _teacher_forcing_stage2_objective_names(cfg) == [
        "conditional_valid_set_likelihood"
    ]


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


def test_checked_in_latest_teacher_forcing_smoke_config_materializes() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml"
    )

    cfg = ConfigLoader.load_materialized_training_config(str(config_path))

    assert isinstance(cfg, LatestDetectionTrainingConfig)
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"
    assert cfg.objective.modules.within_valid_coverage.enabled is False
    assert cfg.training["packing"] is False


def test_checked_in_latest_teacher_forcing_smoke_reaches_dataset_runtime(
    tmp_path: Path,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(config_path))
    assert isinstance(cfg, LatestDetectionTrainingConfig)

    jsonl_path = tmp_path / "train.coord.jsonl"
    _write_jsonl(jsonl_path, [_raw_row()])
    image_root = _ensure_image(tmp_path).parents[2]
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            train_jsonl=str(jsonl_path),
            image_root=str(image_root),
        ),
    )

    system_prompt, _user_prompt = resolve_latest_detection_prompts(cfg)
    custom_config = build_latest_detection_runtime_custom_shim(cfg)
    dataset = build_latest_detection_dataset(
        jsonl_path,
        swift_template=FakeSwiftTemplate(),
        training_config=cfg,
        custom_config=custom_config,
        system_prompt=system_prompt,
        seed=17,
        sample_limit=1,
        dataset_name="teacher_forcing_smoke",
    )
    sample = dataset[0]

    assert sample["detection_metadata"]["mode"] == "random_order_sft"
    assert "recursive_detection_targets" not in sample


def test_checked_in_stage2_teacher_forcing_smoke_config_materializes() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/stage2_two_channel/teacher_forcing/pure_valid_set_marginal_smoke.yaml"
    )

    cfg = ConfigLoader.load_materialized_training_config(str(config_path))

    assert isinstance(cfg, TrainingConfig)
    assert cfg.objective is not None
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "pure_valid_set_marginal"
    assert cfg.stage2_ab is not None
    assert _teacher_forcing_stage2_objective_names(cfg) == [
        "conditional_valid_set_likelihood"
    ]


def test_checked_in_stage2_teacher_forcing_smoke_builds_nonempty_manifest() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/stage2_two_channel/teacher_forcing/pure_valid_set_marginal_smoke.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(config_path))
    assert isinstance(cfg, TrainingConfig)
    assert cfg.stage2_ab is not None

    manifest = build_pipeline_manifest(
        asdict(cfg.stage2_ab),
        default_objective=[
            "token_ce",
            "bbox_geo",
            "bbox_size_aux",
            "coord_reg",
        ],
        default_diagnostics=["coord_diag"],
        trainer_variant="stage2_two_channel",
        config_path=str(config_path),
        run_name=str(cfg.training.get("run_name", "")),
        seed=17,
    )

    objective_names = [module["name"] for module in manifest["objective"]]
    assert objective_names == ["conditional_valid_set_likelihood"]
    assert not {
        "bbox_geo",
        "bbox_size_aux",
        "coord_reg",
        "token_ce",
        "coord_gate",
        "text_gate",
    }.intersection(objective_names)
