from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from src.config.schema import LatestDetectionTrainingConfig


def _prefix_rollin_payload() -> dict[str, object]:
    return {
        "model": {"model": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"},
        "template": {"truncation_strategy": "raise"},
        "training": {
            "run_name": "test-prefix-rollin",
            "num_train_epochs": 1,
        },
        "data": {
            "train_jsonl": "public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl",
            "val_jsonl": "public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl",
            "image_root": "public_data/coco",
            "max_objects": 60,
            "object_ordering": "random_permutation",
        },
        "prompt": {
            "system_variant": "stage1_detection",
            "user_variant": "compact_detection",
            "include_template_summary": True,
            "prompt_variant_enabled": True,
        },
        "detection_template": {
            "id": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "strict_parse": True,
        },
        "token_rows": {
            "enabled": True,
            "tie_head": True,
            "groups": {
                "coord_geometry": {
                    "role": "coord_geometry",
                    "start_token": "<|coord_0|>",
                    "end_token": "<|coord_999|>",
                    "expected_start": 151670,
                    "expected_end": 152669,
                },
                "compact_structure": {
                    "role": "structural_ce_only",
                    "tokens": ["<|object_ref_start|>", "<|box_start|>"],
                    "expected_ids": {
                        "<|object_ref_start|>": 151646,
                        "<|box_start|>": 151648,
                    },
                },
            },
            "embed_lr": 5.0e-5,
            "weight_decay": 0.0,
        },
        "objective": {
            "id": "recursive_detection_ce",
            "variant": "prefix_rollin_et_rmp_ce",
            "state_weighting": "uniform_permutation",
            "normalization": "semantic_image_bucket_balanced",
            "rollin": {
                "enabled": True,
                "source": "ground_truth",
                "prefix_loss": "masked",
                "suffix_order": "same_sampled_permutation",
                "k_distribution": {
                    "type": "uniform_inclusive",
                    "min_k": 0,
                    "max_k": "object_count",
                },
            },
            "target": {
                "type": "entry_trie_support_balance",
                "trie_scope": "object_entry",
                "q_weighting": "object_multiplicity_uniform",
                "singleton": "hard_ce",
                "control_tokens": "hard_ce",
                "support_weight": 1.0,
                "balance_weight": 2.0,
            },
            "boundary": {
                "type": "compact_full_append_boundary",
                "separator_continue_weight": 2.0,
                "eos_stop_weight": 0.5,
                "component_weight": 0.3,
            },
            "type_gate": {
                "enabled": True,
                "mode": "allowed_type_mass",
                "weights": {
                    "struct": 2.0,
                    "coord": 1.0,
                    "desc": 0.2,
                    "eos": 0.5,
                },
            },
            "eos": {
                "eos_token": "<|im_end|>",
                "policy": "missing_label_prior_weighted_ce",
                "eos_trust_weight": {
                    "source": "empirical_unlabeled_poisson_v0",
                    "expected_unlabeled_count": {
                        "intercept": -0.35,
                        "slope": 0.43,
                        "floor": 0.0,
                    },
                    "trust_mapping": {
                        "type": "log_linear_missing_count_penalty",
                        "penalty_per_missing": 1.0,
                        "temperature": 1.0,
                        "min_weight": 0.0,
                        "max_weight": 1.0,
                    },
                },
            },
        },
        "packing": {
            "static_packing": False,
            "padding_free_packed": False,
        },
        "evaluation": {
            "expected_template": "compact_full",
            "parser_mode": "strict_expected",
        },
        "validation": {
            "validate_span_alignment": True,
            "validate_template_capabilities": True,
            "fail_fast": True,
        },
        "experiment": {"surface": "ablation"},
    }


def _random_permutation_eos_payload() -> dict[str, object]:
    payload = copy.deepcopy(_prefix_rollin_payload())
    prefix_objective = payload["objective"]
    assert isinstance(prefix_objective, dict)
    eos = copy.deepcopy(prefix_objective["eos"])
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "eos": eos,
    }
    payload["experiment"] = {
        "surface": "ablation",
        "ablation_id": "A2E",
        "claim_scope": "none",
    }
    return payload


def _parse(payload: dict[str, object]) -> LatestDetectionTrainingConfig:
    return LatestDetectionTrainingConfig.from_mapping(payload)


def _set_path(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    cursor: object = payload
    for key in path[:-1]:
        assert isinstance(cursor, dict)
        cursor = cursor[key]
    assert isinstance(cursor, dict)
    cursor[path[-1]] = value


def _delete_path(payload: dict[str, object], path: tuple[str, ...]) -> None:
    cursor: object = payload
    for key in path[:-1]:
        assert isinstance(cursor, dict)
        cursor = cursor[key]
    assert isinstance(cursor, dict)
    del cursor[path[-1]]


def test_prefix_rollin_schema_accepts_compact_full_empirical_eos_ablation() -> None:
    cfg = _parse(_prefix_rollin_payload())

    assert cfg.objective.variant == "prefix_rollin_et_rmp_ce"
    assert cfg.objective.state_weighting == "uniform_permutation"
    assert cfg.objective.normalization == "semantic_image_bucket_balanced"
    assert cfg.objective.rollin.enabled is True
    assert cfg.objective.rollin.k_distribution.max_k == "object_count"
    assert cfg.objective.target.support_weight == pytest.approx(1.0)
    assert cfg.objective.target.balance_weight == pytest.approx(2.0)
    assert cfg.objective.boundary.separator_continue_weight == pytest.approx(2.0)
    assert cfg.objective.boundary.eos_stop_weight == pytest.approx(0.5)
    assert cfg.objective.boundary.component_weight == pytest.approx(0.3)
    assert cfg.objective.type_gate.weights.struct == pytest.approx(2.0)
    assert cfg.objective.eos.eos_token == "<|im_end|>"
    assert cfg.objective.eos.eos_trust_weight.source == "empirical_unlabeled_poisson_v0"
    assert cfg.experiment.surface == "ablation"

    as_mapping = cfg.to_mapping()
    assert as_mapping["objective"]["variant"] == "prefix_rollin_et_rmp_ce"
    assert as_mapping["objective"]["state_weighting"] == "uniform_permutation"
    assert as_mapping["objective"]["normalization"] == "semantic_image_bucket_balanced"
    for flat_alias in (
        "branch_support_weight",
        "branch_balance_weight",
        "support_weight",
        "balance_weight",
        "trie_support_weight",
        "trie_balance_weight",
    ):
        assert flat_alias not in as_mapping["objective"]
    assert as_mapping["objective"]["target"]["support_weight"] == pytest.approx(1.0)
    assert as_mapping["objective"]["boundary"][
        "separator_continue_weight"
    ] == pytest.approx(2.0)
    assert as_mapping["experiment"]["surface"] == "ablation"
    reparsed = LatestDetectionTrainingConfig.from_mapping(as_mapping)
    assert reparsed.objective.target.balance_weight == pytest.approx(2.0)
    assert reparsed.objective.boundary.eos_stop_weight == pytest.approx(0.5)


def test_prefix_rollin_e1_ablation_config_materializes_from_repo_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (
        repo_root
        / "configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml"
    )

    cfg = ConfigLoader.load_materialized_training_config(str(cfg_path))

    assert isinstance(cfg, LatestDetectionTrainingConfig)
    assert cfg.experiment is not None
    assert cfg.experiment.surface == "ablation"
    assert cfg.experiment.ablation_id == "E1"
    assert cfg.experiment.claim_scope == "none"
    assert cfg.detection_template.id == "compact_full"
    assert cfg.objective.variant == "prefix_rollin_et_rmp_ce"
    assert cfg.objective.state_weighting == "uniform_permutation"
    assert cfg.objective.normalization == "semantic_image_bucket_balanced"
    assert cfg.objective.rollin.k_distribution.min_k == 0
    assert cfg.objective.rollin.k_distribution.max_k == "object_count"
    assert cfg.objective.target.support_weight == pytest.approx(1.0)
    assert cfg.objective.target.balance_weight == pytest.approx(2.0)
    assert cfg.objective.boundary.separator_continue_weight == pytest.approx(0.5)
    assert cfg.objective.boundary.eos_stop_weight == pytest.approx(0.5)
    assert cfg.objective.boundary.component_weight == pytest.approx(0.3)
    assert cfg.objective.type_gate.weights.struct == pytest.approx(2.0)
    assert cfg.objective.type_gate.weights.coord == pytest.approx(1.0)
    assert cfg.objective.type_gate.weights.desc == pytest.approx(0.2)
    assert cfg.objective.type_gate.weights.eos == pytest.approx(0.5)
    assert cfg.objective.eos.eos_token == "<|im_end|>"
    assert cfg.objective.eos.eos_trust_weight.source == "empirical_unlabeled_poisson_v0"
    eos_trust = cfg.objective.eos.eos_trust_weight
    assert eos_trust.expected_unlabeled_count.intercept == pytest.approx(-0.35)
    assert eos_trust.expected_unlabeled_count.slope == pytest.approx(0.43)
    assert eos_trust.trust_mapping.penalty_per_missing == pytest.approx(1.0)
    assert eos_trust.trust_mapping.temperature == pytest.approx(1.0)
    assert eos_trust.trust_mapping.min_weight == pytest.approx(0.0)
    assert eos_trust.trust_mapping.max_weight == pytest.approx(1.0)
    assert cfg.training["packing"] is False
    assert cfg.training["eval_packing"] is False
    assert cfg.training["encoded_sample_cache"]["enabled"] is False
    assert cfg.packing.static_packing is False
    assert cfg.packing.padding_free_packed is False


def test_prefix_rollin_separator2_ablation_overrides_only_boundary_weights() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (
        repo_root
        / "configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_separator2.yaml"
    )

    cfg = ConfigLoader.load_materialized_training_config(str(cfg_path))

    assert isinstance(cfg, LatestDetectionTrainingConfig)
    assert cfg.experiment is not None
    assert cfg.experiment.surface == "ablation"
    assert cfg.experiment.ablation_id == "E2-separator2"
    assert cfg.objective.variant == "prefix_rollin_et_rmp_ce"
    assert cfg.objective.target.support_weight == pytest.approx(1.0)
    assert cfg.objective.target.balance_weight == pytest.approx(2.0)
    assert cfg.objective.boundary.separator_continue_weight == pytest.approx(2.0)
    assert cfg.objective.boundary.eos_stop_weight == pytest.approx(0.5)
    assert cfg.objective.boundary.component_weight == pytest.approx(0.3)


def test_prefix_rollin_requires_objectized_boundary_section() -> None:
    payload = _prefix_rollin_payload()
    _delete_path(payload, ("objective", "boundary"))

    with pytest.raises(
        ValueError, match=r"objective\.variant=prefix_rollin_et_rmp_ce.*boundary"
    ):
        _parse(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("separator_continue_weight", 0.0),
        ("separator_continue_weight", -1.0),
        ("eos_stop_weight", 0.0),
        ("component_weight", 0.0),
        ("component_weight", "0.3"),
    ],
)
def test_prefix_rollin_boundary_weights_must_be_positive_numbers(
    field: str,
    value: object,
) -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("objective", "boundary", field), value)

    error_type = TypeError if isinstance(value, str) else ValueError
    with pytest.raises(error_type, match=rf"objective\.boundary\.{field}"):
        _parse(payload)


def test_prefix_rollin_production_accepts_calibrated_formula_ref_with_artifact() -> (
    None
):
    payload = _prefix_rollin_payload()
    _set_path(
        payload,
        ("objective", "eos", "eos_trust_weight"),
        {
            "source": "calibrated_formula_ref",
            "calibration_artifact_ref": "artifacts/eos_calibration_formula.v1.json",
        },
    )
    _set_path(payload, ("experiment", "surface"), "production")

    cfg = _parse(payload)

    assert cfg.experiment.surface == "production"
    assert cfg.objective.eos.eos_trust_weight.source == "calibrated_formula_ref"
    assert (
        cfg.objective.eos.eos_trust_weight.calibration_artifact_ref
        == "artifacts/eos_calibration_formula.v1.json"
    )


@pytest.mark.parametrize("claim_scope", ["paper", "production"])
def test_prefix_rollin_ablation_rejects_evidence_bearing_claim_scope(
    claim_scope: str,
) -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("experiment", "surface"), "ablation")
    _set_path(payload, ("experiment", "claim_scope"), claim_scope)

    with pytest.raises(ValueError, match=r"experiment\.claim_scope"):
        _parse(payload)


def test_latest_smoke_surface_rejects_production_claim_scope() -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("experiment", "surface"), "smoke")
    _set_path(payload, ("experiment", "claim_scope"), "production")

    with pytest.raises(ValueError, match=r"experiment\.claim_scope"):
        _parse(payload)


def test_existing_random_permutation_latest_schema_still_accepts_flat_trie_weights() -> (
    None
):
    payload = _prefix_rollin_payload()
    payload.pop("experiment")
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "legacy_row_mean_prefix_mixture_equivalence",
        "normalization": "legacy_row_mean_equivalence",
    }

    cfg = _parse(payload)

    assert cfg.objective.variant == "random_permutation_et_rmp_ce"
    assert cfg.objective.trie_support_weight == pytest.approx(2.0)
    assert cfg.experiment is None


def test_random_permutation_schema_accepts_eos_ablation_without_prefix_rollin() -> None:
    cfg = _parse(_random_permutation_eos_payload())

    assert cfg.objective.variant == "random_permutation_et_rmp_ce"
    assert cfg.objective.trie_support_weight == pytest.approx(2.0)
    assert cfg.objective.trie_balance_weight == pytest.approx(1.0)
    assert cfg.objective.rollin is None
    assert cfg.objective.target is None
    assert cfg.objective.boundary is None
    assert cfg.objective.type_gate is None
    assert cfg.objective.eos is not None
    assert cfg.objective.eos.eos_token == "<|im_end|>"
    assert cfg.objective.eos.eos_trust_weight.source == "empirical_unlabeled_poisson_v0"
    assert cfg.experiment is not None
    assert cfg.experiment.ablation_id == "A2E"

    as_mapping = cfg.to_mapping()
    assert as_mapping["objective"]["variant"] == "random_permutation_et_rmp_ce"
    assert as_mapping["objective"]["trie_support_weight"] == pytest.approx(2.0)
    assert "eos" in as_mapping["objective"]
    for prefix_only_section in ("rollin", "target", "boundary", "type_gate"):
        assert prefix_only_section not in as_mapping["objective"]
    reparsed = LatestDetectionTrainingConfig.from_mapping(as_mapping)
    assert reparsed.objective.variant == "random_permutation_et_rmp_ce"
    assert reparsed.objective.eos is not None


def test_random_permutation_eos_ablation_requires_experiment_surface() -> None:
    payload = _random_permutation_eos_payload()
    payload.pop("experiment")

    with pytest.raises(
        ValueError,
        match=r"experiment\.surface is required.*random_permutation_et_rmp_ce",
    ):
        _parse(payload)


def test_random_permutation_eos_ablation_configs_materialize_from_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prod_path = (
        repo_root
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_eos_loosen.yaml"
    )
    smoke_path = (
        repo_root
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_eos_loosen_ddp8_preflight.yaml"
    )

    prod_cfg = ConfigLoader.load_materialized_training_config(str(prod_path))
    smoke_cfg = ConfigLoader.load_materialized_training_config(str(smoke_path))

    for cfg in (prod_cfg, smoke_cfg):
        assert isinstance(cfg, LatestDetectionTrainingConfig)
        assert cfg.objective.variant == "random_permutation_et_rmp_ce"
        assert cfg.objective.trie_support_weight == pytest.approx(2.0)
        assert cfg.objective.trie_balance_weight == pytest.approx(1.0)
        assert cfg.objective.rollin is None
        assert cfg.objective.target is None
        assert cfg.objective.boundary is None
        assert cfg.objective.type_gate is None
        assert cfg.objective.eos is not None
        assert (
            cfg.objective.eos.eos_trust_weight.source
            == "empirical_unlabeled_poisson_v0"
        )
    assert prod_cfg.experiment is not None
    assert prod_cfg.experiment.surface == "ablation"
    assert prod_cfg.experiment.ablation_id == "A2E-support2-eos-loosen"
    assert smoke_cfg.experiment is not None
    assert smoke_cfg.experiment.surface == "smoke"
    assert smoke_cfg.training["max_steps"] == 4


@pytest.mark.parametrize(
    "objective",
    [
        {
            "id": "recursive_detection_ce",
            "variant": "random_permutation_et_rmp_ce",
            "trie_support_weight": 2.0,
            "trie_balance_weight": 1.0,
            "state_weighting": "legacy_row_mean_prefix_mixture_equivalence",
            "normalization": "legacy_row_mean_equivalence",
        },
        {
            "id": "sft",
            "variant": "sorted_sft",
            "state_weighting": "none",
            "normalization": "token_mean",
        },
    ],
)
def test_non_prefix_variants_reject_objectized_boundary_section(
    objective: dict[str, object],
) -> None:
    payload = _prefix_rollin_payload()
    payload.pop("experiment")
    objective = dict(objective)
    objective["boundary"] = {
        "type": "compact_full_append_boundary",
        "separator_continue_weight": 2.0,
        "eos_stop_weight": 0.5,
        "component_weight": 0.3,
    }
    payload["objective"] = objective

    with pytest.raises(
        ValueError,
        match=(
            "objectized objective sections are only supported for "
            "objective.variant=prefix_rollin_et_rmp_ce"
        ),
    ):
        _parse(payload)


def test_prefix_rollin_rejects_non_compact_full_template() -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("detection_template", "id"), "stage1_json_pretty")
    _set_path(payload, ("detection_template", "object_field_order"), "desc_first")
    _set_path(payload, ("evaluation", "expected_template"), "stage1_json_pretty")

    with pytest.raises(ValueError, match="prefix_rollin_et_rmp_ce.*compact_full"):
        _parse(payload)


@pytest.mark.parametrize(
    "custom",
    [
        {"stage1_set_continuation": {}},
        {"trainer_variant": "stage1_set_continuation"},
    ],
)
def test_prefix_rollin_rejects_legacy_set_continuation_custom_keys(
    custom: dict[str, object],
) -> None:
    payload = _prefix_rollin_payload()
    payload["custom"] = custom

    with pytest.raises(ValueError, match="stage1_set_continuation"):
        _parse(payload)


@pytest.mark.parametrize(
    "path",
    [
        ("objective", "branch_support_weight"),
        ("objective", "branch_balance_weight"),
        ("objective", "trie_support_weight"),
        ("objective", "trie_balance_weight"),
        ("objective", "support_weight"),
        ("objective", "balance_weight"),
    ],
)
def test_prefix_rollin_rejects_old_flat_weight_aliases(path: tuple[str, ...]) -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, path, 1.0)

    with pytest.raises(ValueError, match=r"objective\..*weight"):
        _parse(payload)


def test_prefix_rollin_rejects_non_qwen_chat_eos_token() -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("objective", "eos", "eos_token"), "<|endoftext|>")

    with pytest.raises(ValueError, match=r"objective\.eos\.eos_token.*<\\|im_end\\|>"):
        _parse(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_weighting", "none"),
        ("normalization", "token_mean"),
    ],
)
def test_prefix_rollin_requires_config_truthful_weighting_and_normalization(
    field: str,
    value: str,
) -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("objective", field), value)

    with pytest.raises(
        ValueError, match=rf"objective\.{field}.*prefix_rollin_et_rmp_ce"
    ):
        _parse(payload)


def test_prefix_rollin_requires_experiment_surface() -> None:
    payload = _prefix_rollin_payload()
    _delete_path(payload, ("experiment", "surface"))

    with pytest.raises(
        ValueError,
        match=r"experiment\.surface is required.*prefix_rollin_et_rmp_ce",
    ):
        _parse(payload)


@pytest.mark.parametrize("surface", ["smoke", "ablation", "production"])
def test_prefix_rollin_rejects_deferred_user_formula_for_every_surface(
    surface: str,
) -> None:
    payload = _prefix_rollin_payload()
    _set_path(
        payload,
        ("objective", "eos", "eos_trust_weight"),
        {"source": "deferred_user_formula"},
    )
    _set_path(payload, ("experiment", "surface"), surface)

    with pytest.raises(ValueError, match="deferred_user_formula"):
        _parse(payload)


@pytest.mark.parametrize(
    "source",
    [
        "constant_ablation",
        "disabled_ablation",
        "empirical_unlabeled_poisson_v0",
    ],
)
def test_prefix_rollin_production_rejects_ablation_eos_sources(source: str) -> None:
    payload = _prefix_rollin_payload()
    if source in {"constant_ablation", "disabled_ablation"}:
        eos_trust_weight = {"source": source}
    else:
        eos_trust_weight = copy.deepcopy(
            payload["objective"]["eos"]["eos_trust_weight"]  # type: ignore[index]
        )
        assert isinstance(eos_trust_weight, dict)
        eos_trust_weight["source"] = source
    _set_path(payload, ("objective", "eos", "eos_trust_weight"), eos_trust_weight)
    _set_path(payload, ("experiment", "surface"), "production")

    with pytest.raises(ValueError, match=source):
        _parse(payload)


@pytest.mark.parametrize(
    "source",
    ["constant_ablation", "disabled_ablation"],
)
def test_prefix_rollin_ablation_eos_sources_reject_stale_source_specific_fields(
    source: str,
) -> None:
    payload = _prefix_rollin_payload()
    eos_trust_weight = copy.deepcopy(
        payload["objective"]["eos"]["eos_trust_weight"]  # type: ignore[index]
    )
    assert isinstance(eos_trust_weight, dict)
    eos_trust_weight["source"] = source
    eos_trust_weight["calibration_artifact_ref"] = "artifacts/eos_calibration.v1.json"
    _set_path(payload, ("objective", "eos", "eos_trust_weight"), eos_trust_weight)

    with pytest.raises(
        ValueError,
        match=(
            rf"source={source}.*does not accept expected_unlabeled_count"
            r".*trust_mapping.*calibration_artifact_ref"
        ),
    ):
        _parse(payload)


def test_prefix_rollin_empirical_eos_source_rejects_calibration_artifact_ref() -> None:
    payload = _prefix_rollin_payload()
    _set_path(
        payload,
        ("objective", "eos", "eos_trust_weight", "calibration_artifact_ref"),
        "artifacts/eos_calibration.v1.json",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"source=empirical_unlabeled_poisson_v0.*does not accept "
            r"calibration_artifact_ref"
        ),
    ):
        _parse(payload)


def test_prefix_rollin_calibrated_eos_source_rejects_empirical_prior_fields() -> None:
    payload = _prefix_rollin_payload()
    eos_trust_weight = copy.deepcopy(
        payload["objective"]["eos"]["eos_trust_weight"]  # type: ignore[index]
    )
    assert isinstance(eos_trust_weight, dict)
    eos_trust_weight["source"] = "calibrated_formula_ref"
    eos_trust_weight["calibration_artifact_ref"] = (
        "artifacts/eos_calibration_formula.v1.json"
    )
    _set_path(payload, ("objective", "eos", "eos_trust_weight"), eos_trust_weight)

    with pytest.raises(
        ValueError,
        match=(
            r"source=calibrated_formula_ref.*does not accept "
            r"expected_unlabeled_count.*trust_mapping"
        ),
    ):
        _parse(payload)


@pytest.mark.parametrize("field", ["support_weight", "balance_weight"])
def test_prefix_rollin_target_weights_must_be_positive(field: str) -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("objective", "target", field), 0.0)

    with pytest.raises(ValueError, match=rf"objective\.target\.{field}.*> 0"):
        _parse(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "poisson"),
        ("min_k", 1),
        ("min_k", False),
        ("min_k", 0.0),
        ("min_k", "0"),
        ("max_k", "remaining_count"),
    ],
)
def test_prefix_rollin_k_distribution_must_be_uniform_inclusive_zero_to_object_count(
    field: str,
    value: object,
) -> None:
    payload = _prefix_rollin_payload()
    _set_path(payload, ("objective", "rollin", "k_distribution", field), value)

    with pytest.raises(
        ValueError,
        match=r"objective\.rollin\.k_distribution.*uniform_inclusive.*0.*object_count",
    ):
        _parse(payload)
