from __future__ import annotations

import copy
from dataclasses import replace
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig, TrainingConfig
from src.detection.runtime import (
    assert_detection_runtime_supported,
    build_detection_dataset,
    build_detection_runtime_custom_shim,
    detection_mode,
    resolve_detection_prompts,
)
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_detection_training_config_contract import _detection_payload
from test_detection_training_dataset import (
    FakeSwiftTemplate,
    _ensure_image,
    _raw_row,
    _write_jsonl,
)
from test_stage2_rollout_correction_contract import (
    _base_payload as _stage2_rollout_correction_payload,
    _load as _load_stage2_rollout_correction,
    _rollout_correction_pipeline,
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
    payload = _detection_payload()
    objective = _teacher_forcing_objective()
    objective.update(objective_updates)
    payload["objective"] = objective
    return payload


def _hard_sft_objective() -> dict:
    return _teacher_forcing_objective(
        profile="hard_sft",
        coverage_strength=0.0,
        coverage_enabled=False,
    ) | {
        "modules": {
            "token_type_mass": {"enabled": False},
            "conditional_valid_set_likelihood": {"enabled": False},
            "within_valid_coverage": {
                "enabled": False,
                "coverage_strength": 0.0,
            },
            "continuation_margin": {"enabled": False},
        }
    }

@pytest.mark.parametrize(
    "profile",
    [
        "hard_sft",
        "pure_valid_set_marginal",
    ],
)
def test_latest_teacher_forcing_accepts_supported_profiles(profile: str) -> None:
    base_modules = (
        _hard_sft_objective()["modules"]
        if profile == "hard_sft"
        else _teacher_forcing_objective()["modules"]
    )
    payload = _latest_teacher_payload(
        profile=profile,
        modules={
            **base_modules,
            "within_valid_coverage": {
                "enabled": False,
                "coverage_strength": 0.0,
            },
        },
    )

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == profile
    assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"
    assert cfg.objective.target_ir.rollin_policy.base_seed == 17


@pytest.mark.parametrize(
    ("module_key", "module_payload", "expected_key"),
    [
        (
            "token_type_mass",
            {"enabled": True},
            r"objective\.modules\.token_type_mass\.enabled",
        ),
        (
            "conditional_valid_set_likelihood",
            {"enabled": True},
            r"objective\.modules\.conditional_valid_set_likelihood\.enabled",
        ),
        (
            "within_valid_coverage",
            {"enabled": True, "coverage_strength": 0.0},
            r"objective\.modules\.within_valid_coverage\.enabled",
        ),
        (
            "within_valid_coverage",
            {"enabled": False, "coverage_strength": 0.1},
            r"objective\.modules\.within_valid_coverage\.coverage_strength",
        ),
        (
            "continuation_margin",
            {"enabled": True},
            r"objective\.modules\.continuation_margin\.enabled",
        ),
    ],
)
def test_hard_sft_rejects_target_ir_only_modules(
    module_key: str,
    module_payload: dict[str, object],
    expected_key: str,
) -> None:
    objective = _hard_sft_objective()
    objective["modules"] = {
        **objective["modules"],
        module_key: module_payload,
    }
    payload = _latest_teacher_payload(
        profile="hard_sft",
        modules=objective["modules"],
    )

    with pytest.raises(
        ValueError,
        match=rf"objective\.profile=hard_sft.*{expected_key}",
    ):
        DetectionTrainingConfig.from_mapping(payload)


def test_training_config_hard_sft_rejects_enabled_valid_set_likelihood_module() -> None:
    objective = _hard_sft_objective()
    objective["modules"] = {
        **objective["modules"],
        "conditional_valid_set_likelihood": {"enabled": True},
    }
    raw = _stage2_rollout_correction_payload()
    raw["objective"] = objective

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=(
            r"objective\.profile=hard_sft.*"
            r"objective\.modules\.conditional_valid_set_likelihood\.enabled"
        ),
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_latest_teacher_forcing_accepts_minimal_hard_sft_profile() -> None:
    payload = _latest_teacher_payload(
        profile="hard_sft",
        modules={
            "token_type_mass": {"enabled": False},
            "conditional_valid_set_likelihood": {"enabled": False},
        },
    )

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.modules.within_valid_coverage.enabled is False
    assert cfg.objective.modules.within_valid_coverage.coverage_strength == 0.0


def test_hybrid_valid_set_marginal_profile_is_schema_rejected() -> None:
    payload = _latest_teacher_payload(
        profile="hybrid_valid_set_marginal",
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
        match=r"hybrid_valid_set_marginal is unsupported",
    ):
        DetectionTrainingConfig.from_mapping(payload)


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
        DetectionTrainingConfig.from_mapping(payload)


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
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize("old_id", OLD_OBJECTIVE_IDS)
def test_latest_teacher_forcing_rejects_legacy_objective_ids(old_id: str) -> None:
    payload = _detection_payload()
    payload["objective"] = {"id": old_id}

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize("old_id", OLD_OBJECTIVE_IDS)
def test_training_config_rejects_legacy_objective_ids(old_id: str) -> None:
    raw = _stage2_rollout_correction_payload()
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
    raw = _stage2_rollout_correction_payload()
    raw["stage2_rollout_correction"]["pipeline"] = _rollout_correction_pipeline(  # type: ignore[index]
        name=module_name
    )

    with pytest.raises(
        ValueError,
        match=rf"{module_name}.*removed.*residual_set_correction",
    ):
        _load_stage2_rollout_correction(raw)


@pytest.mark.parametrize(
    "legacy_key",
    ["coord_gate_weight", "text_gate_weight", "soft_ce_weight", "w1_weight"],
)
def test_stage2_teacher_forcing_rejects_legacy_gate_config_keys(
    legacy_key: str,
) -> None:
    raw = _stage2_rollout_correction_payload()
    pipeline = raw["stage2_rollout_correction"]["pipeline"]  # type: ignore[index]
    pipeline["objective"][0]["config"][legacy_key] = 0.25  # type: ignore[index]

    with pytest.raises(
        ValueError,
        match=rf"residual_set_correction.*{legacy_key}",
    ):
        _load_stage2_rollout_correction(raw)


def test_stage2_rollout_correction_rejects_hard_sft_pipeline_module() -> None:
    raw = _stage2_rollout_correction_payload()
    raw["stage2_rollout_correction"]["pipeline"] = _rollout_correction_pipeline(  # type: ignore[index]
        name="hard_sft"
    )

    with pytest.raises(ValueError, match=r"hard_sft.*removed.*residual_set_correction"):
        _load_stage2_rollout_correction(raw)


def test_stage2_rollout_correction_rejects_top_level_stage2_ab_namespace() -> None:
    raw = _stage2_rollout_correction_payload()
    raw["stage2_ab"] = {"pipeline": {"objective": []}}

    with pytest.raises(ValueError, match=r"stage2_ab.*removed.*stage2_rollout_correction"):
        _load_stage2_rollout_correction(raw)


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

    assert isinstance(cfg, DetectionTrainingConfig)
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
    assert isinstance(cfg, DetectionTrainingConfig)

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

    system_prompt, _user_prompt = resolve_detection_prompts(cfg)
    custom_config = build_detection_runtime_custom_shim(cfg)
    dataset = build_detection_dataset(
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
    assert TEACHER_FORCING_TARGET_IR_KEY in sample


def test_latest_teacher_forcing_pure_valid_set_profile_reaches_runtime() -> None:
    payload = _latest_teacher_payload()
    payload["objective"] = _teacher_forcing_objective()
    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert detection_mode(cfg) == "random_order_sft"


def test_latest_teacher_forcing_hybrid_profile_is_rejected_before_runtime() -> None:
    payload = _latest_teacher_payload()
    payload["objective"] = _teacher_forcing_objective(
        profile="hybrid_valid_set_marginal",
        coverage_enabled=True,
        coverage_strength=0.1,
    )

    with pytest.raises(
        ValueError,
        match=r"hybrid_valid_set_marginal is unsupported",
    ):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("section", "key", "match"),
    [
        ("training", "packing", r"training\.packing=false"),
        ("training", "eval_packing", r"training\.eval_packing=false"),
        ("packing", "static_packing", r"packing\.static_packing=false"),
        ("packing", "padding_free_packed", r"packing\.padding_free_packed=false"),
    ],
)
def test_latest_teacher_forcing_runtime_rejects_packing_surfaces(
    section: str,
    key: str,
    match: str,
) -> None:
    cfg = DetectionTrainingConfig.from_mapping(_latest_teacher_payload())
    if section == "training":
        cfg = replace(cfg, training={**cfg.training, key: True})
    else:
        cfg = replace(cfg, packing=replace(cfg.packing, **{key: True}))

    with pytest.raises(ValueError, match=match):
        assert_detection_runtime_supported(
            cfg,
            encoded_sample_cache_cfg=SimpleNamespace(enabled=False),
            tokenizer=None,
        )


def test_removed_stage2_teacher_forcing_config_tree_is_absent() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "configs/stage2_two_channel").exists()
    assert not (
        repo_root / "configs/stage2/rollout_correction/teacher_forcing"
    ).exists()
