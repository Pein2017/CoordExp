from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.schema import LatestDetectionTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _latest_payload() -> dict[str, object]:
    return {
        "model": {
            "model": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
        },
        "template": {"truncation_strategy": "raise"},
        "training": {
            "run_name": "test-latest-detection",
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
        "objective": {
            "id": "recursive_detection_ce",
            "variant": "random_permutation_et_rmp_ce",
            "trie_support_weight": 2.0,
            "trie_balance_weight": 1.0,
            "state_weighting": "legacy_row_mean_prefix_mixture_equivalence",
            "normalization": "legacy_row_mean_equivalence",
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
    }


def test_latest_config_parses_and_exposes_typed_sections() -> None:
    cfg = LatestDetectionTrainingConfig.from_mapping(_latest_payload())

    assert cfg.model["model"].endswith("Qwen3-VL-2B-Instruct-coordexp")
    assert cfg.template["truncation_strategy"] == "raise"
    assert cfg.data.max_objects == 60
    assert cfg.data.object_ordering == "random_permutation"
    assert cfg.prompt.prompt_variant_enabled is True
    assert cfg.detection_template.id == "compact_full"
    assert cfg.objective.id == "recursive_detection_ce"
    assert cfg.objective.trie_support_weight == 2.0
    assert cfg.objective.trie_balance_weight == 1.0
    assert cfg.packing.static_packing is False
    assert cfg.evaluation.expected_template == "compact_full"
    assert cfg.validation.fail_fast is True
    assert cfg.to_mapping()["objective"]["trie_support_weight"] == 2.0


def test_custom_is_rejected_with_latest_schema_message() -> None:
    payload = _latest_payload()
    payload["custom"] = {"trainer_variant": "stage1_set_continuation"}

    with pytest.raises(ValueError, match="custom is obsolete for latest detection configs"):
        LatestDetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("trainer_variant",), "stage1_set_continuation"),
        (("objective", "branch_support_weight"), 1.0),
        (("objective", "branch_balance_weight"), 1.0),
        (("objective", "prefix_sampling"), True),
        (("objective", "candidate_balanced"), True),
        (("objective", "positive_evidence_margin"), {"enabled": True}),
        (("objective", "candidate_energy"), {"logZ": True}),
        (("objective", "branch_energy"), True),
        (("objective", "branch_energy_weight"), 1.0),
        (("objective", "legacy_candidate_branch"), True),
    ],
)
def test_obsolete_keys_fail_with_dotted_path(
    path: tuple[str, ...], value: object
) -> None:
    payload = _latest_payload()
    cursor = payload
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    cursor[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError) as exc:
        LatestDetectionTrainingConfig.from_mapping(payload)

    assert ".".join(path) in str(exc.value)


def test_debug_section_does_not_use_global_obsolete_key_scan() -> None:
    payload = _latest_payload()
    payload["debug"] = {"pem": "debug-pass-through"}

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.debug["pem"] == "debug-pass-through"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("training", "suffix"), "runtime-unknown", "Unknown training keys"),
        (("training", "energy"), "runtime-unknown", "Unknown training keys"),
        (("deepspeed", "margin"), "runtime-unknown", "Unknown deepspeed keys"),
        (("model", "not_a_train_argument"), True, "Unknown model keys"),
    ],
)
def test_framework_runtime_sections_preserve_strict_key_validation(
    path: tuple[str, ...], value: object, match: str
) -> None:
    payload = _latest_payload()
    if path[0] == "deepspeed":
        payload["deepspeed"] = {"enabled": False}
    cursor = payload
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    cursor[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=match):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_unknown_keys_fail_with_dotted_path() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "unknown_knob": True,
    }

    with pytest.raises(ValueError) as exc:
        LatestDetectionTrainingConfig.from_mapping(payload)

    assert "objective.unknown_knob" in str(exc.value)


def test_latest_trie_weight_names_are_accepted() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "trie_support_weight": 0.5,
        "trie_balance_weight": 0.25,
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.trie_support_weight == 0.5
    assert cfg.objective.trie_balance_weight == 0.25


def test_et_rmp_weights_must_be_non_negative_and_nonzero() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "trie_support_weight": 0.0,
        "trie_balance_weight": 0.0,
    }

    with pytest.raises(ValueError, match="trie_support_weight.*trie_balance_weight"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_sft_objective_uses_neutral_defaults() -> None:
    payload = _latest_payload()
    payload["data"] = {
        **payload["data"],  # type: ignore[arg-type]
        "object_ordering": "sorted",
    }
    payload["objective"] = {
        "id": "sft",
        "variant": "sorted_sft",
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.trie_support_weight == 0.0
    assert cfg.objective.trie_balance_weight == 0.0
    assert cfg.objective.state_weighting == "none"
    assert cfg.objective.normalization == "token_mean"


def test_random_order_sft_accepts_random_permutation_ordering() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        "id": "sft",
        "variant": "random_order_sft",
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.data.object_ordering == "random_permutation"
    assert cfg.objective.variant == "random_order_sft"


@pytest.mark.parametrize(
    ("object_ordering", "objective"),
    [
        (
            "random_permutation",
            {
                "id": "sft",
                "variant": "sorted_sft",
            },
        ),
        (
            "sorted",
            {
                "id": "sft",
                "variant": "random_order_sft",
            },
        ),
        (
            "sorted",
            {
                "id": "recursive_detection_ce",
                "variant": "random_permutation_et_rmp_ce",
                "trie_support_weight": 2.0,
                "trie_balance_weight": 1.0,
                "state_weighting": "uniform_permutation",
                "normalization": "semantic_image_bucket_balanced",
            },
        ),
        (
            "sorted",
            {
                "id": "recursive_detection_ce",
                "variant": "trie_disabled_full_suffix_ce",
            },
        ),
    ],
)
def test_object_ordering_must_match_objective_variant(
    object_ordering: str, objective: dict[str, object]
) -> None:
    payload = _latest_payload()
    payload["data"] = {
        **payload["data"],  # type: ignore[arg-type]
        "object_ordering": object_ordering,
    }
    payload["objective"] = objective

    with pytest.raises(ValueError, match="data.object_ordering"):
        LatestDetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "override",
    [
        {"trie_support_weight": 1.0},
        {"trie_balance_weight": 1.0},
        {"state_weighting": "uniform_permutation"},
        {"normalization": "semantic_image_bucket_balanced"},
    ],
)
def test_sft_objective_rejects_recursive_knobs(
    override: dict[str, object],
) -> None:
    payload = _latest_payload()
    payload["objective"] = {
        "id": "sft",
        "variant": "random_order_sft",
        **override,
    }

    with pytest.raises(ValueError, match="SFT objective variants"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_trie_disabled_full_suffix_ce_rejects_trie_weights() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "trie_disabled_full_suffix_ce",
        "trie_support_weight": 0.1,
        "trie_balance_weight": 0.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
    }

    with pytest.raises(
        ValueError,
        match="trie_disabled_full_suffix_ce.*trie_support_weight=0",
    ):
        LatestDetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "override",
    [
        {"state_weighting": "uniform_permutation"},
        {"normalization": "semantic_image_bucket_balanced"},
    ],
)
def test_trie_disabled_full_suffix_ce_requires_neutral_profile(
    override: dict[str, object],
) -> None:
    payload = _latest_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "trie_disabled_full_suffix_ce",
        **override,
    }

    with pytest.raises(
        ValueError,
        match="trie_disabled_full_suffix_ce",
    ):
        LatestDetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("state_weighting", "typo_profile"),
        ("normalization", "typo_norm"),
    ],
)
def test_objective_strategy_ids_are_strictly_validated(
    field_name: str, bad_value: str
) -> None:
    payload = _latest_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        field_name: bad_value,
    }

    with pytest.raises(ValueError, match=rf"objective\.{field_name}"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_compact_full_template_must_not_require_json_field_order() -> None:
    payload = _latest_payload()
    payload["detection_template"] = {
        **payload["detection_template"],  # type: ignore[arg-type]
        "object_field_order": "desc_first",
    }

    with pytest.raises(ValueError, match="detection_template.object_field_order"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_stage1_json_pretty_template_requires_desc_first_field_order() -> None:
    payload = _latest_payload()
    payload["detection_template"] = {
        "id": "stage1_json_pretty",
        "coordinate_surface": "coord_token",
        "bbox_format": "xyxy",
        "strict_parse": True,
    }
    payload["evaluation"] = {
        "expected_template": "stage1_json_pretty",
        "parser_mode": "strict_expected",
    }

    with pytest.raises(ValueError, match="stage1_json_pretty.*desc_first"):
        LatestDetectionTrainingConfig.from_mapping(payload)

    payload["detection_template"] = {
        **payload["detection_template"],  # type: ignore[arg-type]
        "object_field_order": "desc_first",
    }
    cfg = LatestDetectionTrainingConfig.from_mapping(payload)
    assert cfg.detection_template.object_field_order == "desc_first"


def test_recursive_detection_ce_fixture_parses() -> None:
    path = REPO_ROOT / "configs/stage1/recursive_detection_ce.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.model["model"] == "schema-contract://model-placeholder"
    assert cfg.data.train_jsonl == "schema-contract://train.coord.jsonl"
    assert cfg.training["run_name"] == "schema-contract-do-not-launch"
    assert cfg.detection_template.id == "compact_full"
    assert cfg.objective.variant == "random_permutation_et_rmp_ce"
    assert cfg.objective.trie_support_weight == 2.0
    assert cfg.objective.trie_balance_weight == 1.0
    assert cfg.objective.state_weighting == "uniform_permutation"
    assert cfg.objective.normalization == "semantic_image_bucket_balanced"
