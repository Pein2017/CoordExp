from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.config.loader import ConfigLoader
from src.config.schema import DebugConfig, LatestDetectionTrainingConfig
from src.detection.runtime import resolve_recursive_detection_ce_runtime_cfg


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
            "id": "teacher_forcing",
            "profile": "pure_valid_set_marginal",
            "target_ir": {
                "rollin_policy": {
                    "name": "random_permutation",
                    "base_seed": 17,
                },
                "exact_packing_mapping": {"enabled": False},
            },
            "modules": {
                "token_type_mass": {"enabled": True},
                "conditional_valid_set_likelihood": {"enabled": True},
                "within_valid_coverage": {
                    "enabled": False,
                    "coverage_strength": 0.0,
                },
                "continuation_margin": {"enabled": False},
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
    }


def _update_section(
    payload: dict[str, object],
    section: str,
    **updates: object,
) -> None:
    current = payload[section]
    assert isinstance(current, dict)
    payload[section] = {**current, **updates}


def test_latest_config_parses_and_exposes_typed_sections() -> None:
    cfg = LatestDetectionTrainingConfig.from_mapping(_latest_payload())

    assert cfg.model["model"].endswith("Qwen3-VL-2B-Instruct-coordexp")
    assert cfg.template["truncation_strategy"] == "raise"
    assert cfg.data.object_ordering == "random_permutation"
    assert cfg.prompt.prompt_variant_enabled is True
    assert cfg.detection_template.id == "compact_full"
    assert cfg.token_rows.enabled is True
    assert cfg.token_rows.tie_head is True
    assert cfg.token_rows.embed_lr == pytest.approx(5.0e-5)
    assert cfg.token_rows.groups["coord_geometry"].role.value == "coord_geometry"
    assert (
        cfg.token_rows.groups["compact_structure"].expected_ids["<|box_start|>"]
        == 151648
    )
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "pure_valid_set_marginal"
    assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"
    assert cfg.objective.target_ir.rollin_policy.base_seed == 17
    assert cfg.objective.modules.token_type_mass.enabled is True
    assert cfg.objective.modules.conditional_valid_set_likelihood.enabled is True
    assert cfg.objective.modules.within_valid_coverage.enabled is False
    assert cfg.objective.modules.within_valid_coverage.coverage_strength == 0.0
    assert cfg.packing.static_packing is False
    assert cfg.evaluation.expected_template == "compact_full"
    assert cfg.validation.fail_fast is True
    assert isinstance(cfg.debug, DebugConfig)
    assert cfg.debug.enabled is False
    assert cfg.to_mapping()["objective"]["id"] == "teacher_forcing"


def test_latest_config_rejects_legacy_data_max_objects_key() -> None:
    payload = _latest_payload()
    data = dict(payload["data"])
    data["max_objects"] = 60
    payload["data"] = data

    with pytest.raises(ValueError, match=r"Unknown keys.*data\.max_objects"):
        LatestDetectionTrainingConfig.from_mapping(payload)


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


def test_latest_debug_section_parses_through_debug_config() -> None:
    payload = _latest_payload()
    payload["debug"] = {
        "enabled": True,
        "output_dir": "temp/latest-debug",
        "train_sample_limit": 3,
        "val_sample_limit": 4,
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert isinstance(cfg.debug, DebugConfig)
    assert cfg.debug.enabled is True
    assert cfg.debug.output_dir == "temp/latest-debug"
    assert cfg.debug.train_sample_limit == 3
    assert cfg.debug.val_sample_limit == 4


def test_latest_debug_unknown_keys_fail_fast() -> None:
    payload = _latest_payload()
    payload["debug"] = {"pem": "debug-pass-through"}

    with pytest.raises(ValueError, match=r"Unknown debug keys.*debug\.pem"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_static_packing_requires_training_packing_owner() -> None:
    payload = _latest_payload()
    _update_section(payload, "packing", static_packing=True)
    _update_section(payload, "training", packing=False)

    with pytest.raises(
        ValueError,
        match=r"packing\.static_packing=true.*training\.packing=true",
    ):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_teacher_forcing_rejects_runtime_packing_without_exact_mapping() -> None:
    payload = _latest_payload()
    _update_section(payload, "training", packing=True)

    with pytest.raises(
        ValueError,
        match=r"teacher_forcing.*training\.packing=true.*exact_packing_mapping",
    ):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_teacher_forcing_rejects_static_packing_without_exact_mapping() -> None:
    payload = _latest_payload()
    _update_section(payload, "packing", static_packing=True)
    _update_section(payload, "training", packing=True)

    with pytest.raises(
        ValueError,
        match=r"teacher_forcing.*training\.packing=true.*exact_packing_mapping",
    ):
        LatestDetectionTrainingConfig.from_mapping(payload)


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


@pytest.mark.parametrize(
    "legacy_objective",
    [
        {
            "id": "recursive_detection_ce",
            "variant": "random_permutation_et_rmp_ce",
            "trie_support_weight": 2.0,
            "trie_balance_weight": 1.0,
            "state_weighting": "uniform_permutation",
            "normalization": "semantic_image_bucket_balanced",
        },
        {
            "id": "recursive_detection_ce",
            "variant": "prefix_rollin_et_rmp_ce",
        },
        {
            "id": "recursive_detection_ce",
            "variant": "trie_disabled_full_suffix_ce",
        },
        {
            "id": "sft",
            "variant": "random_order_sft",
        },
        {
            "id": "sft",
            "variant": "sorted_sft",
        },
    ],
)
def test_latest_config_rejects_legacy_objective_payloads(
    legacy_objective: dict[str, object],
) -> None:
    payload = _latest_payload()
    payload["objective"] = legacy_objective

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_instance_trie_gaussian_runtime_uses_token_row_coordinate_id_offset() -> None:
    cfg = SimpleNamespace(
        objective=SimpleNamespace(
            id="recursive_detection_ce",
            variant="random_permutation_et_rmp_ce",
            trie_support_weight=2.0,
            trie_balance_weight=1.0,
            coord_soft_ce=SimpleNamespace(
                enabled=True,
                target_distribution="instance_trie_gaussian",
            ),
        ),
        token_rows=SimpleNamespace(
            groups={
                "coord_geometry": SimpleNamespace(
                    role="coord_geometry",
                    expected_start=42000,
                    expected_end=42999,
                )
            }
        ),
    )

    runtime_cfg = resolve_recursive_detection_ce_runtime_cfg(cfg)

    assert runtime_cfg is not None
    assert runtime_cfg.coord_soft_ce is not None
    assert runtime_cfg.coord_soft_ce.target_distribution == "instance_trie_gaussian"
    assert runtime_cfg.coord_soft_ce.coord_token_start == 42000
    assert runtime_cfg.coord_soft_ce.coord_token_end == 42999
    assert runtime_cfg.coord_soft_ce.coord_value_to_token_id(0) == 42000
    assert runtime_cfg.coord_soft_ce.coord_value_to_token_id(999) == 42999
    assert runtime_cfg.coord_soft_ce.gaussian_mixture_weight == pytest.approx(0.1)
    assert runtime_cfg.coord_soft_ce.gaussian_r95_axis_fraction == pytest.approx(0.04)
    assert runtime_cfg.coord_soft_ce.gaussian_r95_cap_bins == 8


def test_recursive_detection_metrics_do_not_require_coord_softce_tau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.trainers.batch_extras import RECURSIVE_DETECTION_TARGETS_KEY
    from src.trainers.metrics import recursive_detection as metrics_module
    from src.trainers.metrics.recursive_detection import RecursiveDetectionCEMixin

    logged: dict[str, float] = {}

    class _Reporter:
        def __init__(self, trainer: object) -> None:
            self.trainer = trainer

        def update_many(self, updates: dict[str, float]) -> None:
            logged.update(updates)

    class _Model:
        training = True

        def __call__(self, **inputs: object) -> SimpleNamespace:
            input_ids = inputs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            return SimpleNamespace(
                logits=torch.zeros(
                    (*input_ids.shape, 8),
                    dtype=torch.float32,
                    device=input_ids.device,
                )
            )

    def _fake_loss(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            loss=torch.tensor(1.25, dtype=torch.float32),
            metrics={"batch_size": 1.0},
            metric_events=(),
        )

    monkeypatch.setattr(
        "src.metrics.reporter.SwiftMetricReporter",
        _Reporter,
    )
    monkeypatch.setattr(
        metrics_module,
        "compute_recursive_detection_ce_batch_loss",
        _fake_loss,
    )

    trainer = RecursiveDetectionCEMixin()
    trainer.model = _Model()
    trainer.recursive_detection_ce_cfg = SimpleNamespace(
        trie_support_weight=2.0,
        trie_balance_weight=1.0,
        type_gate=SimpleNamespace(
            enabled=True,
            weights=SimpleNamespace(
                struct=1.0,
                coord=1.0,
                desc=1.0,
                eos=0.5,
            ),
        ),
        coord_soft_ce=SimpleNamespace(
            target_distribution="instance_trie_gaussian",
            coord_token_start=42000,
            coord_token_end=42999,
            gaussian_mixture_weight=0.1,
            gaussian_r95_axis_fraction=0.04,
            gaussian_r95_cap_bins=8,
        ),
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "labels": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        RECURSIVE_DETECTION_TARGETS_KEY: (SimpleNamespace(token_targets=()),),
    }

    loss = trainer.compute_loss(trainer.model, inputs)

    assert loss.item() == pytest.approx(1.25)
    assert "recursive_detection_ce/coord_soft_ce/tau" not in logged
    assert logged["recursive_detection_ce/coord_soft_ce/config_enabled"] == pytest.approx(
        1.0
    )
    assert logged[
        "recursive_detection_ce/coord_soft_ce/is_instance_trie_gaussian"
    ] == pytest.approx(1.0)
    assert logged["recursive_detection_ce/coord_soft_ce/coord_token_start"] == pytest.approx(
        42000.0
    )
    assert logged["recursive_detection_ce/coord_soft_ce/coord_token_end"] == pytest.approx(
        42999.0
    )
    assert logged[
        "recursive_detection_ce/coord_soft_ce/gaussian_mixture_weight"
    ] == pytest.approx(0.1)
    assert logged[
        "recursive_detection_ce/coord_soft_ce/exact_ce_anchor_weight"
    ] == pytest.approx(0.9)
    assert logged[
        "recursive_detection_ce/coord_soft_ce/gaussian_r95_axis_fraction"
    ] == pytest.approx(0.04)
    assert logged[
        "recursive_detection_ce/coord_soft_ce/gaussian_r95_cap_bins"
    ] == pytest.approx(8.0)
    assert logged["recursive_detection_ce/type_gate/config_enabled"] == pytest.approx(
        1.0
    )
    assert logged["recursive_detection_ce/type_gate/struct_weight"] == pytest.approx(
        1.0
    )
    assert logged["recursive_detection_ce/type_gate/coord_weight"] == pytest.approx(
        1.0
    )
    assert logged["recursive_detection_ce/type_gate/desc_weight"] == pytest.approx(
        1.0
    )
    assert logged["recursive_detection_ce/type_gate/eos_weight"] == pytest.approx(0.5)


def test_hard_sft_profile_is_minimal_teacher_forcing_baseline() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        "id": "teacher_forcing",
        "profile": "hard_sft",
        "target_ir": {
            "rollin_policy": {
                "name": "random_permutation",
                "base_seed": 17,
            }
        },
        "modules": {
            "token_type_mass": {"enabled": False},
            "conditional_valid_set_likelihood": {"enabled": False},
        },
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.modules.token_type_mass.enabled is False
    assert cfg.objective.modules.conditional_valid_set_likelihood.enabled is False
    assert cfg.objective.modules.within_valid_coverage.enabled is False
    assert cfg.objective.modules.within_valid_coverage.coverage_strength == 0.0


def test_latest_detection_requires_token_rows_section() -> None:
    payload = _latest_payload()
    payload.pop("token_rows")

    with pytest.raises(ValueError, match="Missing latest detection config sections"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_detection_requires_coord_geometry_token_rows() -> None:
    payload = _latest_payload()
    payload["token_rows"] = {
        "enabled": True,
        "tie_head": True,
        "groups": {
            "compact_structure": {
                "role": "structural_ce_only",
                "tokens": ["<|object_ref_start|>", "<|box_start|>"],
            }
        },
    }

    with pytest.raises(ValueError, match="token_rows.*coord_geometry"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_detection_rejects_disabled_token_rows() -> None:
    payload = _latest_payload()
    payload["token_rows"] = {
        **payload["token_rows"],  # type: ignore[arg-type]
        "enabled": False,
    }

    with pytest.raises(ValueError, match="token_rows.enabled"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_detection_requires_tied_token_rows() -> None:
    payload = _latest_payload()
    payload["token_rows"] = {
        **payload["token_rows"],  # type: ignore[arg-type]
        "tie_head": False,
    }

    with pytest.raises(ValueError, match="token_rows.tie_head.*true"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_detection_requires_exact_compact_structural_rows() -> None:
    payload = _latest_payload()
    token_rows = dict(payload["token_rows"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    groups.pop("compact_structure")
    token_rows["groups"] = groups
    payload["token_rows"] = token_rows

    with pytest.raises(ValueError, match="object_ref_start.*box_start"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_detection_requires_exact_coord_row_range() -> None:
    payload = _latest_payload()
    token_rows = dict(payload["token_rows"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    coord = dict(groups["coord_geometry"])  # type: ignore[index]
    coord["end_token"] = "<|coord_998|>"
    coord["expected_end"] = 152668
    groups["coord_geometry"] = coord
    token_rows["groups"] = groups
    payload["token_rows"] = token_rows

    with pytest.raises(ValueError, match="coord_0.*coord_999"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_latest_detection_rejects_extra_trainable_token_rows() -> None:
    payload = _latest_payload()
    token_rows = dict(payload["token_rows"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    groups["natural_language_leak"] = {
        "role": "structural_ce_only",
        "tokens": ["the"],
    }
    token_rows["groups"] = groups
    payload["token_rows"] = token_rows

    with pytest.raises(ValueError, match="exactly.*1002|natural-language"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_teacher_forcing_rollin_policy_accepts_random_permutation_ordering() -> None:
    payload = _latest_payload()

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.data.object_ordering == "random_permutation"
    assert cfg.objective.id == "teacher_forcing"
    assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"


@pytest.mark.parametrize(
    ("object_ordering", "objective"),
    [
        ("sorted", _latest_payload()["objective"]),
    ],
)
def test_object_ordering_must_match_teacher_forcing_rollin_policy(
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
    "legacy_objective",
    [
        {"id": "sft", "variant": "random_order_sft", "trie_support_weight": 1.0},
        {
            "id": "recursive_detection_ce",
            "variant": "trie_disabled_full_suffix_ce",
            "trie_support_weight": 0.1,
        },
    ],
)
def test_legacy_objective_knobs_fail_at_objective_id_boundary(
    legacy_objective: dict[str, object],
) -> None:
    payload = _latest_payload()
    payload["objective"] = legacy_objective

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
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
        "profile" if field_name == "state_weighting" else field_name: bad_value,
    }

    expected_field = "profile" if field_name == "state_weighting" else field_name
    with pytest.raises(ValueError, match=rf"objective\.{expected_field}"):
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


def test_recursive_detection_ce_fixture_fails_objective_migration() -> None:
    path = REPO_ROOT / "configs/stage1/recursive_detection_ce.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        LatestDetectionTrainingConfig.from_mapping(payload)


def test_config_loader_materializes_latest_detection_config_without_custom(
    tmp_path: Path,
) -> None:
    payload = _latest_payload()
    payload["training"] = {
        "run_name": "latest-loader",
        "num_train_epochs": 1,
        "output_root": str(tmp_path / "runs"),
        "logging_root": str(tmp_path / "logs"),
        "artifact_subdir": "latest-loader",
    }
    config_path = tmp_path / "latest.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = ConfigLoader.load_materialized_training_config(str(config_path))

    assert isinstance(cfg, LatestDetectionTrainingConfig)
    assert not hasattr(cfg, "custom")
    assert cfg.data.train_jsonl.endswith("train.coord.jsonl")


def test_config_loader_builds_train_arguments_from_latest_runtime_sections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _latest_payload()
    payload["training"] = {
        "run_name": "latest-train-args",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "effective_batch_size": 4,
        "output_root": str(tmp_path / "runs"),
        "logging_root": str(tmp_path / "logs"),
        "artifact_subdir": "latest-train-args",
        "save_model_only": True,
    }
    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    captured: dict[str, object] = {}

    class FakeTrainArguments:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.training_args = SimpleNamespace()
            for key, value in kwargs.items():
                setattr(self, key, value)

    monkeypatch.setattr("src.config.loader.TrainArguments", FakeTrainArguments)
    monkeypatch.setattr("src.config.loader.RLHFArguments", FakeTrainArguments)
    monkeypatch.setattr("src.config.loader.get_dist_setting", lambda: (0, 0, 1, 1))

    train_args = ConfigLoader.build_train_arguments(cfg)

    assert isinstance(train_args, FakeTrainArguments)
    assert captured["model"] == payload["model"]["model"]  # type: ignore[index]
    assert "train_jsonl" not in captured
    assert "val_jsonl" not in captured
    assert "image_root" not in captured
    assert "save_model_only" not in captured
    assert captured["gradient_accumulation_steps"] == 2


def test_latest_detection_rejects_upstream_save_only_model_knob() -> None:
    payload = _latest_payload()
    payload["training"] = {
        "run_name": "latest-save-only-model-rejected",
        "save_only_model": True,
    }

    with pytest.raises(ValueError) as exc:
        LatestDetectionTrainingConfig.from_mapping(payload)

    assert "training.save_only_model" in str(exc.value)
    assert "training.save_model_only" in str(exc.value)


def test_latest_detection_rejects_deprecated_checkpoint_mode_knob() -> None:
    payload = _latest_payload()
    payload["training"] = {
        "run_name": "latest-checkpoint-mode-rejected",
        "checkpoint_mode": "restartable",
    }

    with pytest.raises(ValueError) as exc:
        LatestDetectionTrainingConfig.from_mapping(payload)

    assert "training.checkpoint_mode" in str(exc.value)
    assert "training.save_model_only" in str(exc.value)


def test_latest_detection_save_model_only_requires_boolean() -> None:
    payload = _latest_payload()
    payload["training"] = {
        "run_name": "latest-save-model-only-null-rejected",
        "save_model_only": None,
    }

    with pytest.raises(ValueError) as exc:
        LatestDetectionTrainingConfig.from_mapping(payload)

    assert "training.save_model_only must be a boolean" in str(exc.value)


def test_config_loader_rejects_authored_gradient_accumulation_when_effective_batch_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _latest_payload()
    payload["training"] = {
        "run_name": "latest-train-args",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "effective_batch_size": 4,
        "gradient_accumulation_steps": 2,
    }
    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    monkeypatch.setattr("src.config.loader.get_dist_setting", lambda: (0, 0, 1, 1))

    with pytest.raises(
        ValueError,
        match=r"training\.gradient_accumulation_steps.*effective_batch_size",
    ):
        ConfigLoader.build_train_arguments(cfg)


def test_legacy_effective_batch_config_fails_objective_migration() -> None:
    cfg_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_random_sft_chatfix_max12k_bsz1.yaml"
    )

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        ConfigLoader.load_materialized_training_config(str(cfg_path))


def test_latest_recursive_detection_launch_configs_fail_objective_migration() -> None:
    config_paths = [
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_focused_cap8_frac0p06_mix0p1.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_len12000_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_lvis_proxy_all_len12000_parse.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_prodlike_single_gpu.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_ddp8_preflight.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_focused_cap8_frac0p06_mix0p1_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_ddp4_preflight.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_ddp4_preflight.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1_ddp8_preflight.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_focused_cap8_frac0p06_mix0p1_ddp8_preflight.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2_ddp8_preflight.yaml",
    ]

    for config_path in config_paths:
        with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
            ConfigLoader.load_materialized_training_config(str(config_path))


def test_coco80_len12000_smoke_configs_fail_objective_migration() -> None:
    config_expectations = {
        "compact_full_coco80_len12000_tiny.yaml": (
            "public_data/coco/views/coco80/len-12000/train.jsonl",
            "public_data/coco/views/coco80/len-12000/val.jsonl",
        ),
        "compact_full_coco80_lvis_proxy_all_len12000_parse.yaml": (
            "public_data/coco/views/coco80-lvis-proxy/len-12000/train.jsonl",
            "public_data/coco/views/coco80-lvis-proxy/len-12000/val.jsonl",
        ),
    }

    for filename, (train_jsonl, val_jsonl) in config_expectations.items():
        assert train_jsonl.endswith("train.jsonl")
        assert val_jsonl.endswith("val.jsonl")
        config_path = (
            REPO_ROOT
            / "configs/stage1/recursive_detection_ce_latest/smoke"
            / filename
        )
        with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
            ConfigLoader.load_materialized_training_config(str(config_path))


def test_latest_recursive_detection_1p0_control_config_fails_objective_migration() -> None:
    config_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_ddp8_et_rmp_1p0_1p0.yaml"
    )

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        ConfigLoader.load_materialized_training_config(str(config_path))


def test_latest_recursive_detection_adapter_smoke_configs_fail_objective_migration() -> None:
    expected = {
        "compact_full_prefix_rollin_adapter_tiny.yaml": "prefix_rollin_et_rmp_ce",
        "compact_full_random_sft_adapter_tiny.yaml": "random_order_sft",
    }

    for file_name, objective_variant in expected.items():
        assert objective_variant
        config_path = (
            REPO_ROOT
            / "configs/stage1/recursive_detection_ce_latest/smoke"
            / file_name
        )
        with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
            ConfigLoader.load_materialized_training_config(str(config_path))


def test_latest_compact_sft_smoke_configs_fail_objective_migration() -> None:
    expected = {
        "compact_full_sorted_sft.yaml": ("sorted", "sorted_sft", True),
        "compact_full_random_sft.yaml": (
            "random_permutation",
            "random_order_sft",
            False,
        ),
        "compact_full_random_sft_prompt_variant.yaml": (
            "random_permutation",
            "random_order_sft",
            True,
        ),
    }

    for file_name, (ordering, variant, prompt_variant) in expected.items():
        assert ordering
        assert variant
        assert isinstance(prompt_variant, bool)
        config_path = (
            REPO_ROOT
            / "configs/stage1/recursive_detection_ce_latest/smoke"
            / file_name
        )
        with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
            ConfigLoader.load_materialized_training_config(str(config_path))


def test_latest_recursive_detection_packing_preflight_config_is_failfast_only() -> None:
    config_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml"
    )

    with pytest.raises(ValueError, match=r"objective\.id.*teacher_forcing"):
        ConfigLoader.load_materialized_training_config(str(config_path))
