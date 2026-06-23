from __future__ import annotations

import copy
from dataclasses import replace
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import (
    DetectionObjectiveConfig,
    DetectionTrainingConfig,
    TrainingConfig,
)
from src.detection.runtime import (
    assert_detection_runtime_supported,
    build_detection_dataset,
    build_detection_runtime_custom_shim,
    detection_mode,
    resolve_detection_prompts,
)
from src.detection.template_contracts import resolve_detection_template_contract
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
    "teacher_forcing",
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
        "id": "research_teacher_forcing",
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
        "terms": {
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
    payload["pipeline"] = {"id": "stage1_research_teacher_forcing"}
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
        "terms": {
            "token_type_mass": {"enabled": False},
            "conditional_valid_set_likelihood": {"enabled": False},
            "within_valid_coverage": {
                "enabled": False,
                "coverage_strength": 0.0,
            },
            "continuation_margin": {"enabled": False},
        }
    }


def _set_detection_template(payload: dict[str, object], template_id: str) -> None:
    payload["detection_template"] = {"id": template_id}
    payload["evaluation"] = {
        "expected_template": template_id,
        "parser_mode": "strict_expected",
    }
    try:
        contract = resolve_detection_template_contract(template_id)
    except ValueError:
        return
    token_rows = dict(payload["token_embeddings_adapter"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    groups["compact_structure"] = {
        "role": "structural_ce_only",
        "tokens": list(contract.required_structural_tokens),
        "expected_ids": dict(
            zip(
                contract.required_structural_tokens,
                contract.required_structural_token_ids,
            )
        ),
    }
    token_rows["groups"] = groups
    payload["token_embeddings_adapter"] = token_rows


def _coverage_ledger_payload(
    *,
    template_id: str = "compact_object_box_closed",
    coverage_ledger: dict[str, object] | None = None,
) -> dict:
    objective = _hard_sft_objective()
    objective["terms"] = {
        **objective["terms"],
        "coverage_ledger": (
            {"enabled": True}
            if coverage_ledger is None
            else {"enabled": True, **coverage_ledger}
        ),
    }
    payload = _latest_teacher_payload(
        profile="hard_sft",
        terms=objective["terms"],
    )
    training = payload["training"]
    assert isinstance(training, dict)
    training["per_device_train_batch_size"] = 1
    training["effective_batch_size"] = 1
    _set_detection_template(payload, template_id)
    return payload


@pytest.mark.parametrize(
    "profile",
    [
        "hard_sft",
        "pure_valid_set_marginal",
        "hybrid_valid_set_marginal",
    ],
)
def test_latest_teacher_forcing_accepts_supported_profiles(profile: str) -> None:
    coverage_enabled = profile == "hybrid_valid_set_marginal"
    base_terms = (
        _hard_sft_objective()["terms"]
        if profile == "hard_sft"
        else _teacher_forcing_objective()["terms"]
    )
    payload = _latest_teacher_payload(
        profile=profile,
        terms={
            **base_terms,
            "within_valid_coverage": {
                "enabled": coverage_enabled,
                "coverage_strength": 0.2 if coverage_enabled else 0.0,
            },
        },
    )

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.id == "research_teacher_forcing"
    assert cfg.objective.profile == profile
    assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"
    assert cfg.objective.target_ir.rollin_policy.base_seed == 17


def test_sft_runtime_payload_preserves_public_teacher_forcing_sidecars() -> None:
    from src.sft import _detection_objective_runtime_payload

    cfg = DetectionTrainingConfig.from_mapping(_latest_teacher_payload())

    payload = _detection_objective_runtime_payload(cfg)

    assert payload is not None
    assert payload["id"] == "research_teacher_forcing"
    assert payload["target_ir"]["rollin_policy"]["name"] == "random_permutation"
    assert payload["target_ir"]["rollin_policy"]["base_seed"] == 17
    assert payload["terms"]["conditional_valid_set_likelihood"]["enabled"] is True


def test_research_teacher_forcing_rejects_retired_modules_authoring() -> None:
    payload = _latest_teacher_payload()
    objective = payload["objective"]
    assert isinstance(objective, dict)
    objective["modules"] = objective.pop("terms")

    with pytest.raises(ValueError, match=r"objective\.modules.*objective\.terms"):
        DetectionTrainingConfig.from_mapping(payload)


def test_detection_objective_config_direct_construction_rejects_sft_id() -> None:
    with pytest.raises(ValueError, match=r"objective\.id.*standard_ce"):
        DetectionObjectiveConfig(id="sft")


@pytest.mark.parametrize(
    ("term_key", "term_payload", "expected_path"),
    [
        (
            "token_type_mass",
            {"enabled": True},
            r"objective\.terms\.token_type_mass\.enabled",
        ),
        (
            "conditional_valid_set_likelihood",
            {"enabled": True},
            r"objective\.terms\.conditional_valid_set_likelihood\.enabled",
        ),
        (
            "within_valid_coverage",
            {"enabled": True, "coverage_strength": 0.0},
            r"objective\.terms\.within_valid_coverage\.enabled",
        ),
        (
            "within_valid_coverage",
            {"enabled": False, "coverage_strength": 0.1},
            r"objective\.terms\.within_valid_coverage\.coverage_strength",
        ),
        (
            "continuation_margin",
            {"enabled": True},
            r"objective\.terms\.continuation_margin\.enabled",
        ),
    ],
)
def test_hard_sft_rejects_target_ir_only_terms(
    term_key: str,
    term_payload: dict[str, object],
    expected_path: str,
) -> None:
    objective = _hard_sft_objective()
    objective["terms"] = {
        **objective["terms"],
        term_key: term_payload,
    }
    payload = _latest_teacher_payload(
        profile="hard_sft",
        terms=objective["terms"],
    )

    with pytest.raises(
        ValueError,
        match=rf"objective\.profile=hard_sft.*{expected_path}",
    ):
        DetectionTrainingConfig.from_mapping(payload)


def test_training_config_hard_sft_rejects_enabled_valid_set_likelihood_module() -> None:
    objective = _hard_sft_objective()
    objective["terms"] = {
        **objective["terms"],
        "conditional_valid_set_likelihood": {"enabled": True},
    }
    raw = _stage2_rollout_correction_payload()
    raw["objective"] = objective

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(
        ValueError,
        match=(
            r"objective\.profile=hard_sft.*"
            r"objective\.terms\.conditional_valid_set_likelihood\.enabled"
        ),
    ):
        TrainingConfig.from_mapping(raw, prompts)


def test_latest_teacher_forcing_accepts_minimal_hard_sft_profile() -> None:
    payload = _latest_teacher_payload(
        profile="hard_sft",
        terms={
            "token_type_mass": {"enabled": False},
            "conditional_valid_set_likelihood": {"enabled": False},
        },
    )

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.terms.within_valid_coverage.enabled is False
    assert cfg.objective.terms.within_valid_coverage.coverage_strength == 0.0


def test_hard_sft_accepts_coverage_ledger_with_required_template() -> None:
    cfg = DetectionTrainingConfig.from_mapping(_coverage_ledger_payload())

    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.terms.coverage_ledger.enabled is True
    assert cfg.objective.terms.coverage_ledger.coverage_weight == 0.1
    assert cfg.objective.terms.coverage_ledger.region_anchor_weight == 0.1
    assert cfg.objective.terms.coverage_ledger.ledger_projection_dim == 256
    assert cfg.objective.terms.coverage_ledger.temperature == 0.2
    assert cfg.objective.terms.coverage_ledger.normalize_eps == 1.0e-6
    assert cfg.objective.terms.coverage_ledger.pos_weight == 1.0
    assert cfg.objective.terms.coverage_ledger.log_auc is True
    assert cfg.objective.terms.coverage_ledger.log_accuracy is True
    assert cfg.objective.terms.coverage_ledger.overlay_sample_count == 16
    assert cfg.objective.terms.coverage_ledger.smoke_sample_count == 128
    assert cfg.objective.terms.coverage_ledger.smoke_sample_seed == 20260623


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("enabled", 1, r"objective\.terms\.coverage_ledger\.enabled.*boolean"),
        (
            "coverage_weight",
            "0.1",
            r"objective\.terms\.coverage_ledger\.coverage_weight.*numeric",
        ),
        (
            "coverage_weight",
            -0.1,
            r"objective\.terms\.coverage_ledger\.coverage_weight.*>= 0",
        ),
        (
            "coverage_weight",
            float("inf"),
            r"objective\.terms\.coverage_ledger\.coverage_weight.*finite",
        ),
        (
            "region_anchor_weight",
            "0.1",
            r"objective\.terms\.coverage_ledger\.region_anchor_weight.*numeric",
        ),
        (
            "region_anchor_weight",
            -0.1,
            r"objective\.terms\.coverage_ledger\.region_anchor_weight.*>= 0",
        ),
        (
            "region_anchor_weight",
            float("inf"),
            r"objective\.terms\.coverage_ledger\.region_anchor_weight.*finite",
        ),
        (
            "ledger_projection_dim",
            0,
            r"objective\.terms\.coverage_ledger\.ledger_projection_dim"
            r".*positive integer",
        ),
        (
            "ledger_projection_dim",
            1.5,
            r"objective\.terms\.coverage_ledger\.ledger_projection_dim"
            r".*positive integer",
        ),
        ("temperature", "0.2", r"objective\.terms\.coverage_ledger\.temperature.*numeric"),
        ("temperature", 0.049, r"objective\.terms\.coverage_ledger\.temperature.*>= 0\.05"),
        ("temperature", float("inf"), r"objective\.terms\.coverage_ledger\.temperature.*finite"),
        ("normalize_eps", "1e-6", r"objective\.terms\.coverage_ledger\.normalize_eps.*numeric"),
        ("normalize_eps", 9.0e-9, r"objective\.terms\.coverage_ledger\.normalize_eps.*>= 1e-8"),
        (
            "normalize_eps",
            float("inf"),
            r"objective\.terms\.coverage_ledger\.normalize_eps.*finite",
        ),
        ("pos_weight", "1.0", r"objective\.terms\.coverage_ledger\.pos_weight.*numeric"),
        ("pos_weight", 0.0, r"objective\.terms\.coverage_ledger\.pos_weight.*> 0"),
        ("pos_weight", float("inf"), r"objective\.terms\.coverage_ledger\.pos_weight.*finite"),
        ("log_auc", 1, r"objective\.terms\.coverage_ledger\.log_auc.*boolean"),
        ("log_accuracy", 1, r"objective\.terms\.coverage_ledger\.log_accuracy.*boolean"),
        (
            "overlay_sample_count",
            15,
            r"objective\.terms\.coverage_ledger\.overlay_sample_count"
            r".*exactly 16",
        ),
        (
            "smoke_sample_count",
            127,
            r"objective\.terms\.coverage_ledger\.smoke_sample_count.*exactly 128",
        ),
        (
            "smoke_sample_seed",
            20260622,
            r"objective\.terms\.coverage_ledger\.smoke_sample_seed"
            r".*exactly 20260623",
        ),
    ],
)
def test_coverage_ledger_rejects_invalid_field_values(
    field_name: str,
    value: object,
    match: str,
) -> None:
    payload = _coverage_ledger_payload(coverage_ledger={field_name: value})

    with pytest.raises((TypeError, ValueError), match=match):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "template_id",
    [
        "compact",
        "compact_box_closed",
        "compact_object_closed",
    ],
)
def test_coverage_ledger_requires_compact_object_box_closed_template(
    template_id: str,
) -> None:
    payload = _coverage_ledger_payload(template_id=template_id)

    with pytest.raises(
        ValueError,
        match=rf"coverage_ledger.*compact_object_box_closed.*{template_id}",
    ):
        DetectionTrainingConfig.from_mapping(payload)


def test_coverage_ledger_rejects_compact_full_with_legacy_message() -> None:
    payload = _coverage_ledger_payload(template_id="compact_full")

    with pytest.raises(
        ValueError,
        match=r"coverage_ledger.*compact_full.*old chat-template/schema",
    ):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("section", "key", "match"),
    [
        ("training", "packing", r"coverage_ledger.*training\.packing=false"),
        (
            "packing",
            "static_packing",
            r"coverage_ledger.*packing\.static_packing=false",
        ),
        (
            "packing",
            "padding_free_packed",
            r"coverage_ledger.*packing\.padding_free_packed=false",
        ),
    ],
)
def test_coverage_ledger_rejects_packing_modes(
    section: str,
    key: str,
    match: str,
) -> None:
    payload = _coverage_ledger_payload()
    nested = payload[section]
    assert isinstance(nested, dict)
    nested[key] = True

    with pytest.raises(ValueError, match=match):
        DetectionTrainingConfig.from_mapping(payload)


def test_coverage_ledger_rejects_resolved_train_batch_size_above_one() -> None:
    payload = _coverage_ledger_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["per_device_train_batch_size"] = 2

    with pytest.raises(
        ValueError,
        match=r"coverage_ledger.*per_device_train_batch_size.*1",
    ):
        DetectionTrainingConfig.from_mapping(payload)


def test_coverage_profile_requires_explicit_positive_coverage_strength() -> None:
    payload = _latest_teacher_payload(
        profile="hybrid_valid_set_marginal",
        terms={
            **_teacher_forcing_objective()["terms"],
            "within_valid_coverage": {
                "enabled": True,
                "coverage_strength": 0.0,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=r"hybrid_valid_set_marginal.*coverage_strength.*> 0",
    ):
        DetectionTrainingConfig.from_mapping(payload)


def test_pure_profile_rejects_positive_coverage_strength() -> None:
    payload = _latest_teacher_payload(
        terms={
            **_teacher_forcing_objective()["terms"],
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

    with pytest.raises(ValueError, match=r"objective\.id.*research_teacher_forcing"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize("old_id", OLD_OBJECTIVE_IDS)
def test_training_config_rejects_legacy_objective_ids(old_id: str) -> None:
    raw = _stage2_rollout_correction_payload()
    raw["objective"] = {"id": old_id}

    prompts = ConfigLoader.resolve_prompts(raw)
    with pytest.raises(ValueError, match=r"objective\.id.*research_teacher_forcing"):
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
    assert cfg.objective.id == "research_teacher_forcing"


def test_target_hierarchy_teacher_forcing_smoke_config_materializes() -> None:
    cfg = DetectionTrainingConfig.from_mapping(
        _latest_teacher_payload(
            profile="hard_sft",
            terms={
                "token_type_mass": {"enabled": False},
                "conditional_valid_set_likelihood": {"enabled": False},
            },
        )
    )

    assert isinstance(cfg, DetectionTrainingConfig)
    assert cfg.objective.id == "research_teacher_forcing"
    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"
    assert cfg.objective.terms.within_valid_coverage.enabled is False
    assert cfg.training["packing"] is False


def test_target_hierarchy_teacher_forcing_smoke_reaches_dataset_runtime(
    tmp_path: Path,
) -> None:
    cfg = DetectionTrainingConfig.from_mapping(
        _latest_teacher_payload(
            profile="hard_sft",
            terms={
                "token_type_mass": {"enabled": False},
                "conditional_valid_set_likelihood": {"enabled": False},
            },
        )
    )
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


def test_detection_dataset_runtime_uses_configured_compact_field_order(
    tmp_path: Path,
) -> None:
    payload = _latest_teacher_payload()
    payload["detection_template"] = {
        "id": "compact_object_box_closed",
    }
    sample_factory = payload["sample_factory"]
    assert isinstance(sample_factory, dict)
    target_sequence = sample_factory["target_sequence"]
    assert isinstance(target_sequence, dict)
    target_sequence["object_field_order"] = "geometry_first"
    payload["evaluation"] = {
        "expected_template": "compact_object_box_closed",
        "parser_mode": "strict_expected",
    }
    contract = resolve_detection_template_contract("compact_object_box_closed")
    token_rows = dict(payload["token_embeddings_adapter"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    groups["compact_structure"] = {
        "role": "structural_ce_only",
        "tokens": list(contract.required_structural_tokens),
        "expected_ids": dict(
            zip(
                contract.required_structural_tokens,
                contract.required_structural_token_ids,
            )
        ),
    }
    token_rows["groups"] = groups
    payload["token_embeddings_adapter"] = token_rows
    cfg = DetectionTrainingConfig.from_mapping(payload)

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
        dataset_name="geometry_first_teacher_forcing",
    )

    sample = dataset[0]
    assistant_text = sample["messages"][2]["content"]

    assert assistant_text.startswith("<|box_start|>")
    assert "<|box_end|><|object_ref_start|>" in assistant_text
    assert not assistant_text.startswith("<|object_ref_start|>")
    assert {
        str(obj["desc"])
        for obj in sample["assistant_payload"]["objects"]
    } == {"cat", "dog", "bus"}
    assert sample["detection_metadata"]["object_field_order"] == "geometry_first"
    assert (
        sample[TEACHER_FORCING_TARGET_IR_KEY].metadata["object_field_order"]
        == "geometry_first"
    )
    assert dataset.encoded_length_for_row(0) == len(sample["input_ids"])


def test_latest_teacher_forcing_pure_valid_set_profile_reaches_runtime() -> None:
    payload = _latest_teacher_payload()
    payload["objective"] = _teacher_forcing_objective()
    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert detection_mode(cfg) == "random_order_sft"


def test_latest_teacher_forcing_coverage_profile_still_requires_runtime_wiring() -> None:
    payload = _latest_teacher_payload()
    payload["objective"] = _teacher_forcing_objective(
        profile="hybrid_valid_set_marginal",
        coverage_enabled=True,
        coverage_strength=0.1,
    )
    cfg = DetectionTrainingConfig.from_mapping(payload)

    with pytest.raises(
        ValueError,
        match=r"currently supports objective\.profile",
    ):
        detection_mode(cfg)


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


def test_latest_teacher_forcing_schema_rejects_training_packing_with_public_id_message() -> None:
    payload = _latest_teacher_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["packing"] = True

    with pytest.raises(
        ValueError,
        match=r"objective\.id=research_teacher_forcing.*training\.packing=true",
    ):
        DetectionTrainingConfig.from_mapping(payload)


def test_removed_stage2_teacher_forcing_config_tree_is_absent() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "configs/stage2_two_channel").exists()
    assert not (
        repo_root / "configs/stage2/rollout_correction/teacher_forcing"
    ).exists()
