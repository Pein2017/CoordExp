from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.config.loader import ConfigLoader
from src.config.schema import DebugConfig, DetectionTrainingConfig
from src.detection.runtime import resolve_recursive_detection_ce_runtime_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]


def _detection_payload() -> dict[str, object]:
    return {
        "model": {
            "model": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
        },
        "template": {"truncation_strategy": "raise"},
        "training": {
            "run_name": "test-detection-training",
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


def _update_section(
    payload: dict[str, object],
    section: str,
    **updates: object,
) -> None:
    current = payload[section]
    assert isinstance(current, dict)
    payload[section] = {**current, **updates}


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_config_parses_and_exposes_typed_sections() -> None:
    cfg = DetectionTrainingConfig.from_mapping(_detection_payload())

    assert cfg.model["model"].endswith("Qwen3-VL-2B-Instruct-coordexp")
    assert cfg.template["truncation_strategy"] == "raise"
    assert cfg.data.max_objects == 60
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
    assert cfg.objective.id == "recursive_detection_ce"
    assert cfg.objective.trie_support_weight == 2.0
    assert cfg.objective.trie_balance_weight == 1.0
    assert cfg.packing.static_packing is False
    assert cfg.evaluation.expected_template == "compact_full"
    assert cfg.validation.fail_fast is True
    assert isinstance(cfg.debug, DebugConfig)
    assert cfg.debug.enabled is False
    assert cfg.to_mapping()["objective"]["trie_support_weight"] == 2.0


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_custom_is_rejected_with_current_schema_message() -> None:
    payload = _detection_payload()
    payload["custom"] = {"trainer_variant": "stage1_set_continuation"}

    with pytest.raises(ValueError, match="custom is obsolete for detection configs"):
        DetectionTrainingConfig.from_mapping(payload)


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

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_obsolete_keys_fail_with_dotted_path(
    path: tuple[str, ...], value: object
) -> None:
    payload = _detection_payload()
    cursor = payload
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    cursor[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError) as exc:
        DetectionTrainingConfig.from_mapping(payload)

    assert ".".join(path) in str(exc.value)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_debug_section_parses_through_debug_config() -> None:
    payload = _detection_payload()
    payload["debug"] = {
        "enabled": True,
        "output_dir": "temp/detection-debug",
        "train_sample_limit": 3,
        "val_sample_limit": 4,
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert isinstance(cfg.debug, DebugConfig)
    assert cfg.debug.enabled is True
    assert cfg.debug.output_dir == "temp/detection-debug"
    assert cfg.debug.train_sample_limit == 3
    assert cfg.debug.val_sample_limit == 4


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_debug_unknown_keys_fail_fast() -> None:
    payload = _detection_payload()
    payload["debug"] = {"pem": "debug-pass-through"}

    with pytest.raises(ValueError, match=r"Unknown debug keys.*debug\.pem"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_static_packing_requires_training_packing_owner() -> None:
    payload = _detection_payload()
    _update_section(payload, "packing", static_packing=True)
    _update_section(payload, "training", packing=False)

    with pytest.raises(
        ValueError,
        match=r"packing\.static_packing=true.*training\.packing=true",
    ):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_rejects_runtime_packing_without_static_owner() -> None:
    payload = _detection_payload()
    _update_section(payload, "training", packing=True)

    with pytest.raises(
        ValueError,
        match=r"recursive_detection_ce.*training\.packing=false",
    ):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_rejects_static_packing_when_adapter_matches() -> None:
    payload = _detection_payload()
    _update_section(payload, "packing", static_packing=True)
    _update_section(payload, "training", packing=True)

    with pytest.raises(ValueError, match=r"recursive_detection_ce.*static packing"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_rejects_padding_free_packing() -> None:
    payload = _detection_payload()
    _update_section(payload, "packing", padding_free_packed=True)

    with pytest.raises(ValueError, match=r"recursive_detection_ce.*padding_free_packed"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_rejects_use_logits_to_keep() -> None:
    payload = _detection_payload()
    _update_section(payload, "training", use_logits_to_keep=True)

    with pytest.raises(ValueError, match=r"recursive_detection_ce.*use_logits_to_keep"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_rejects_training_loss_scale() -> None:
    payload = _detection_payload()
    _update_section(payload, "training", loss_scale="default")

    with pytest.raises(ValueError, match=r"recursive_detection_ce.*loss_scale"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_rejects_training_left_padding() -> None:
    payload = _detection_payload()
    _update_section(payload, "training", padding_side="left")

    with pytest.raises(ValueError, match=r"recursive_detection_ce.*padding_side='right'"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("training", "suffix"), "runtime-unknown", "Unknown training keys"),
        (("training", "energy"), "runtime-unknown", "Unknown training keys"),
        (("deepspeed", "margin"), "runtime-unknown", "Unknown deepspeed keys"),
        (("model", "not_a_train_argument"), True, "Unknown model keys"),
    ],
)

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_framework_runtime_sections_preserve_strict_key_validation(
    path: tuple[str, ...], value: object, match: str
) -> None:
    payload = _detection_payload()
    if path[0] == "deepspeed":
        payload["deepspeed"] = {"enabled": False}
    cursor = payload
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    cursor[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=match):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_unknown_keys_fail_with_dotted_path() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "unknown_knob": True,
    }

    with pytest.raises(ValueError) as exc:
        DetectionTrainingConfig.from_mapping(payload)

    assert "objective.unknown_knob" in str(exc.value)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_trie_weight_names_are_accepted() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "trie_support_weight": 0.5,
        "trie_balance_weight": 0.25,
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.trie_support_weight == 0.5
    assert cfg.objective.trie_balance_weight == 0.25


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_random_permutation_accepts_iou_gibbs_coord_softce() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "iou_gibbs_v0",
            "tau": 0.0090909091,
            "tau_source": "train_one_token_iou_median_v0",
            "weighting": "preserve_recursive_support_balance",
            "replace_coord_hard_ce": True,
            "apply_to_multi_positive": "support_mixture",
        },
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.coord_soft_ce is not None
    assert cfg.objective.coord_soft_ce.enabled is True
    assert cfg.objective.coord_soft_ce.tau == pytest.approx(0.0090909091)
    assert cfg.objective.coord_soft_ce.target_distribution == "iou_gibbs_v0"
    assert cfg.objective.coord_soft_ce.weighting == "preserve_recursive_support_balance"


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_random_permutation_accepts_ciou_gibbs_coord_softce() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "ciou_gibbs_v0",
            "tau": 0.0090909091,
            "tau_source": "train_one_token_iou_median_v0",
            "weighting": "preserve_recursive_support_balance",
            "replace_coord_hard_ce": True,
            "apply_to_multi_positive": "support_mixture",
        },
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.coord_soft_ce is not None
    assert cfg.objective.coord_soft_ce.enabled is True
    assert cfg.objective.coord_soft_ce.target_distribution == "ciou_gibbs_v0"


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_random_permutation_accepts_instance_trie_gaussian_coord_softce() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "instance_trie_gaussian",
        },
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.coord_soft_ce is not None
    assert cfg.objective.coord_soft_ce.enabled is True
    assert cfg.objective.coord_soft_ce.target_distribution == "instance_trie_gaussian"
    assert cfg.objective.coord_soft_ce.gaussian_mixture_weight == pytest.approx(0.1)
    assert cfg.objective.coord_soft_ce.gaussian_r95_axis_fraction == pytest.approx(0.04)
    assert cfg.objective.coord_soft_ce.gaussian_r95_cap_bins == 8
    assert not hasattr(cfg.objective.coord_soft_ce, "tau")
    assert not hasattr(cfg.objective.coord_soft_ce, "weighting")


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_random_permutation_et_rmp_accepts_type_gate_section() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "type_gate": {
            "enabled": True,
            "mode": "allowed_type_mass",
            "weights": {
                "struct": 1.0,
                "coord": 1.0,
                "desc": 1.0,
                "eos": 0.5,
            },
        },
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.type_gate is not None
    assert cfg.objective.type_gate.enabled is True
    assert cfg.objective.type_gate.weights.coord == pytest.approx(1.0)

    runtime_cfg = resolve_recursive_detection_ce_runtime_cfg(cfg)
    assert runtime_cfg is not None
    assert runtime_cfg.type_gate is not None
    assert runtime_cfg.type_gate.enabled is True
    assert runtime_cfg.type_gate.weights.struct == pytest.approx(1.0)
    assert runtime_cfg.type_gate.weights.desc == pytest.approx(1.0)
    assert runtime_cfg.type_gate.weights.coord == pytest.approx(1.0)
    assert runtime_cfg.type_gate.weights.eos == pytest.approx(0.5)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_random_permutation_accepts_ce_anchored_instance_trie_gaussian_coord_softce() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "instance_trie_gaussian",
            "gaussian_mixture_weight": 0.2,
            "gaussian_r95_axis_fraction": 0.06,
            "gaussian_r95_cap_bins": 8,
        },
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.coord_soft_ce is not None
    assert cfg.objective.coord_soft_ce.enabled is True
    assert cfg.objective.coord_soft_ce.target_distribution == "instance_trie_gaussian"
    assert cfg.objective.coord_soft_ce.gaussian_mixture_weight == pytest.approx(0.2)
    assert cfg.objective.coord_soft_ce.gaussian_r95_axis_fraction == pytest.approx(0.06)
    assert cfg.objective.coord_soft_ce.gaussian_r95_cap_bins == 8


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_recursive_detection_runtime_resolves_coord_softce_token_range() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[dict-item]
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "ciou_gibbs_v0",
            "tau": 0.0090909091,
            "tau_source": "train_one_token_iou_median_v0",
            "weighting": "preserve_recursive_support_balance",
            "replace_coord_hard_ce": True,
            "apply_to_multi_positive": "support_mixture",
        },
    }
    cfg = DetectionTrainingConfig.from_mapping(payload)

    runtime_cfg = resolve_recursive_detection_ce_runtime_cfg(cfg)

    assert runtime_cfg is not None
    assert runtime_cfg.coord_soft_ce is not None
    assert runtime_cfg.coord_soft_ce.target_distribution == "ciou_gibbs_v0"
    assert runtime_cfg.coord_soft_ce.coord_token_start == 151670
    assert runtime_cfg.coord_soft_ce.coord_token_end == 152669


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
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


@pytest.mark.parametrize(
    "deprecated_key",
    ["sigma", "truncate", "target_sigma", "target_truncate", "window", "radius"],
)

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_coord_softce_rejects_fixed_gaussian_knobs(deprecated_key: str) -> None:
    payload = _detection_payload()
    payload["objective"]["coord_soft_ce"] = {
        "enabled": True,
        "target_distribution": "iou_gibbs_v0",
        "tau": 0.0090909091,
        "tau_source": "train_one_token_iou_median_v0",
        deprecated_key: 2.0,
    }

    with pytest.raises(ValueError, match=rf"objective\.coord_soft_ce\.{deprecated_key}"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_coord_softce_requires_positive_data_derived_tau() -> None:
    payload = _detection_payload()
    payload["objective"]["coord_soft_ce"] = {
        "enabled": True,
        "target_distribution": "iou_gibbs_v0",
        "tau": 0.0,
        "tau_source": "train_one_token_iou_median_v0",
    }

    with pytest.raises(ValueError, match=r"objective\.coord_soft_ce\.tau.*> 0"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "stale_key",
    [
        "tau",
        "tau_source",
        "weighting",
        "replace_coord_hard_ce",
        "apply_to_multi_positive",
        "sigma",
        "truncate",
        "target_sigma",
        "target_truncate",
    ],
)

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_instance_trie_gaussian_coord_softce_rejects_stale_knobs(
    stale_key: str,
) -> None:
    payload = _detection_payload()
    payload["objective"]["coord_soft_ce"] = {
        "enabled": True,
        "target_distribution": "instance_trie_gaussian",
        stale_key: 0.01,
    }

    with pytest.raises(ValueError, match=rf"objective\.coord_soft_ce\.{stale_key}"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("key", "value", "pattern"),
    [
        ("gaussian_r95_axis_fraction", 0.0, "gaussian_r95_axis_fraction"),
        ("gaussian_r95_axis_fraction", -0.01, "gaussian_r95_axis_fraction"),
        ("gaussian_r95_axis_fraction", 1.5, "gaussian_r95_axis_fraction"),
        ("gaussian_r95_axis_fraction", "0.04", "gaussian_r95_axis_fraction"),
        ("gaussian_r95_cap_bins", -1, "gaussian_r95_cap_bins"),
        ("gaussian_r95_cap_bins", 1000, "gaussian_r95_cap_bins"),
        ("gaussian_r95_cap_bins", 8.0, "gaussian_r95_cap_bins"),
        ("gaussian_r95_cap_bins", True, "gaussian_r95_cap_bins"),
    ],
)

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_instance_trie_gaussian_coord_softce_rejects_invalid_focused_policy(
    key: str,
    value: object,
    pattern: str,
) -> None:
    payload = _detection_payload()
    payload["objective"]["coord_soft_ce"] = {
        "enabled": True,
        "target_distribution": "instance_trie_gaussian",
        "gaussian_mixture_weight": 0.1,
        "gaussian_r95_axis_fraction": 0.04,
        "gaussian_r95_cap_bins": 8,
        key: value,
    }

    with pytest.raises((TypeError, ValueError), match=pattern):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
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


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_et_rmp_weights_must_be_non_negative_and_nonzero() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        "trie_support_weight": 0.0,
        "trie_balance_weight": 0.0,
    }

    with pytest.raises(ValueError, match="trie_support_weight.*trie_balance_weight"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_sft_objective_uses_neutral_defaults() -> None:
    payload = _detection_payload()
    payload["data"] = {
        **payload["data"],  # type: ignore[arg-type]
        "object_ordering": "sorted",
    }
    payload["objective"] = {
        "id": "sft",
        "variant": "sorted_sft",
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.trie_support_weight == 0.0
    assert cfg.objective.trie_balance_weight == 0.0
    assert cfg.objective.state_weighting == "none"
    assert cfg.objective.normalization == "token_mean"


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_requires_token_rows_section() -> None:
    payload = _detection_payload()
    payload.pop("token_rows")

    with pytest.raises(ValueError, match="Missing detection config sections"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_requires_coord_geometry_token_rows() -> None:
    payload = _detection_payload()
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
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_rejects_disabled_token_rows() -> None:
    payload = _detection_payload()
    payload["token_rows"] = {
        **payload["token_rows"],  # type: ignore[arg-type]
        "enabled": False,
    }

    with pytest.raises(ValueError, match="token_rows.enabled"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_requires_tied_token_rows() -> None:
    payload = _detection_payload()
    payload["token_rows"] = {
        **payload["token_rows"],  # type: ignore[arg-type]
        "tie_head": False,
    }

    with pytest.raises(ValueError, match="token_rows.tie_head.*true"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_requires_exact_compact_structural_rows() -> None:
    payload = _detection_payload()
    token_rows = dict(payload["token_rows"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    groups.pop("compact_structure")
    token_rows["groups"] = groups
    payload["token_rows"] = token_rows

    with pytest.raises(ValueError, match="object_ref_start.*box_start"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_requires_exact_coord_row_range() -> None:
    payload = _detection_payload()
    token_rows = dict(payload["token_rows"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    coord = dict(groups["coord_geometry"])  # type: ignore[index]
    coord["end_token"] = "<|coord_998|>"
    coord["expected_end"] = 152668
    groups["coord_geometry"] = coord
    token_rows["groups"] = groups
    payload["token_rows"] = token_rows

    with pytest.raises(ValueError, match="coord_0.*coord_999"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_rejects_extra_trainable_token_rows() -> None:
    payload = _detection_payload()
    token_rows = dict(payload["token_rows"])  # type: ignore[arg-type]
    groups = dict(token_rows["groups"])  # type: ignore[index]
    groups["natural_language_leak"] = {
        "role": "structural_ce_only",
        "tokens": ["the"],
    }
    token_rows["groups"] = groups
    payload["token_rows"] = token_rows

    with pytest.raises(ValueError, match="exactly.*1002|natural-language"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_random_order_sft_accepts_random_permutation_ordering() -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "sft",
        "variant": "random_order_sft",
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

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

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_object_ordering_must_match_objective_variant(
    object_ordering: str, objective: dict[str, object]
) -> None:
    payload = _detection_payload()
    payload["data"] = {
        **payload["data"],  # type: ignore[arg-type]
        "object_ordering": object_ordering,
    }
    payload["objective"] = objective

    with pytest.raises(ValueError, match="data.object_ordering"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "override",
    [
        {"trie_support_weight": 1.0},
        {"trie_balance_weight": 1.0},
        {"state_weighting": "uniform_permutation"},
        {"normalization": "semantic_image_bucket_balanced"},
    ],
)

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_sft_objective_rejects_recursive_knobs(
    override: dict[str, object],
) -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "sft",
        "variant": "random_order_sft",
        **override,
    }

    with pytest.raises(ValueError, match="SFT objective variants"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_trie_disabled_full_suffix_ce_rejects_trie_weights() -> None:
    payload = _detection_payload()
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
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "override",
    [
        {"state_weighting": "uniform_permutation"},
        {"normalization": "semantic_image_bucket_balanced"},
    ],
)

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_trie_disabled_full_suffix_ce_requires_neutral_profile(
    override: dict[str, object],
) -> None:
    payload = _detection_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "trie_disabled_full_suffix_ce",
        **override,
    }

    with pytest.raises(
        ValueError,
        match="trie_disabled_full_suffix_ce",
    ):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("state_weighting", "typo_profile"),
        ("normalization", "typo_norm"),
    ],
)

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_objective_strategy_ids_are_strictly_validated(
    field_name: str, bad_value: str
) -> None:
    payload = _detection_payload()
    payload["objective"] = {
        **payload["objective"],  # type: ignore[arg-type]
        field_name: bad_value,
    }

    with pytest.raises(ValueError, match=rf"objective\.{field_name}"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_compact_full_template_must_not_require_json_field_order() -> None:
    payload = _detection_payload()
    payload["detection_template"] = {
        **payload["detection_template"],  # type: ignore[arg-type]
        "object_field_order": "desc_first",
    }

    with pytest.raises(ValueError, match="detection_template.object_field_order"):
        DetectionTrainingConfig.from_mapping(payload)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_stage1_json_pretty_template_requires_desc_first_field_order() -> None:
    payload = _detection_payload()
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
        DetectionTrainingConfig.from_mapping(payload)

    payload["detection_template"] = {
        **payload["detection_template"],  # type: ignore[arg-type]
        "object_field_order": "desc_first",
    }
    cfg = DetectionTrainingConfig.from_mapping(payload)
    assert cfg.detection_template.object_field_order == "desc_first"


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_recursive_detection_ce_fixture_parses() -> None:
    path = (
        REPO_ROOT
        / "tests/fixtures/configs/stage1/recursive_detection_ce_schema_contract.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.model["model"] == "schema-contract://model-placeholder"
    assert cfg.data.train_jsonl == "schema-contract://train.coord.jsonl"
    assert cfg.training["run_name"] == "schema-contract-do-not-launch"
    assert cfg.detection_template.id == "compact_full"
    assert cfg.objective.variant == "random_permutation_et_rmp_ce"
    assert cfg.objective.trie_support_weight == 2.0
    assert cfg.objective.trie_balance_weight == 1.0
    assert cfg.objective.state_weighting == "uniform_permutation"
    assert cfg.objective.normalization == "semantic_image_bucket_balanced"


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_config_loader_materializes_detection_config_without_custom(
    tmp_path: Path,
) -> None:
    payload = _detection_payload()
    payload["training"] = {
        "run_name": "detection-loader",
        "num_train_epochs": 1,
        "output_root": str(tmp_path / "runs"),
        "logging_root": str(tmp_path / "logs"),
        "artifact_subdir": "detection-loader",
    }
    config_path = tmp_path / "detection.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = ConfigLoader.load_materialized_training_config(str(config_path))

    assert isinstance(cfg, DetectionTrainingConfig)
    assert not hasattr(cfg, "custom")
    assert cfg.data.train_jsonl.endswith("train.coord.jsonl")


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_config_loader_builds_train_arguments_from_detection_runtime_sections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _detection_payload()
    payload["training"] = {
        "run_name": "detection-train-args",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "effective_batch_size": 4,
        "output_root": str(tmp_path / "runs"),
        "logging_root": str(tmp_path / "logs"),
        "artifact_subdir": "detection-train-args",
        "save_model_only": True,
    }
    cfg = DetectionTrainingConfig.from_mapping(payload)

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


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_rejects_upstream_save_only_model_knob() -> None:
    payload = _detection_payload()
    payload["training"] = {
        "run_name": "detection-save-only-model-rejected",
        "save_only_model": True,
    }

    with pytest.raises(ValueError) as exc:
        DetectionTrainingConfig.from_mapping(payload)

    assert "training.save_only_model" in str(exc.value)
    assert "training.save_model_only" in str(exc.value)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_rejects_deprecated_checkpoint_mode_knob() -> None:
    payload = _detection_payload()
    payload["training"] = {
        "run_name": "detection-checkpoint-mode-rejected",
        "checkpoint_mode": "restartable",
    }

    with pytest.raises(ValueError) as exc:
        DetectionTrainingConfig.from_mapping(payload)

    assert "training.checkpoint_mode" in str(exc.value)
    assert "training.save_model_only" in str(exc.value)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_save_model_only_requires_boolean() -> None:
    payload = _detection_payload()
    payload["training"] = {
        "run_name": "detection-save-model-only-null-rejected",
        "save_model_only": None,
    }

    with pytest.raises(ValueError) as exc:
        DetectionTrainingConfig.from_mapping(payload)

    assert "training.save_model_only must be a boolean" in str(exc.value)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_config_loader_rejects_authored_gradient_accumulation_when_effective_batch_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _detection_payload()
    payload["training"] = {
        "run_name": "detection-train-args",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "effective_batch_size": 4,
        "gradient_accumulation_steps": 2,
    }
    cfg = DetectionTrainingConfig.from_mapping(payload)

    monkeypatch.setattr("src.config.loader.get_dist_setting", lambda: (0, 0, 1, 1))

    with pytest.raises(
        ValueError,
        match=r"training\.gradient_accumulation_steps.*effective_batch_size",
    ):
        ConfigLoader.build_train_arguments(cfg)


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_effective_batch_config_names_do_not_bake_derived_accumulation() -> None:
    cfg_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/prod/compact_full_random_sft.yaml"
    )
    cfg = ConfigLoader.load_materialized_training_config(str(cfg_path))

    assert cfg.training.get("effective_batch_size") == 128
    assert "gradient_accumulation_steps" not in cfg.training
    assert "accum" not in str(cfg.training.get("artifact_subdir", ""))
    assert "accum" not in str(cfg.training.get("run_name", ""))


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_launch_configs_parse_without_custom() -> None:
    config_paths = [
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/prod/compact_full_support2.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/prod/compact_full_support2_instance_trie_focused_cap8_frac0p06_mix0p1.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/smoke/compact_full_tiny.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/smoke/compact_full_ddp8_preflight.yaml",
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce/smoke/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p1_tiny.yaml",
    ]

    for config_path in config_paths:
        cfg = ConfigLoader.load_materialized_training_config(str(config_path))
        assert isinstance(cfg, DetectionTrainingConfig)
        assert not hasattr(cfg, "custom")
        assert cfg.model["model"] == "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
        assert cfg.data.object_ordering == "random_permutation"
        assert cfg.detection_template.id == "compact_full"
        assert cfg.token_rows.enabled is True
        assert cfg.token_rows.tie_head is True
        assert "coord_geometry" in cfg.token_rows.groups
        token_to_id = {
            "<|object_ref_start|>": 151646,
            "<|box_start|>": 151648,
            **{f"<|coord_{idx}|>": 151670 + idx for idx in range(1000)},
        }

        class _FakeTokenizer:
            def convert_tokens_to_ids(self, token: str) -> int:
                return token_to_id[token]

        role_sets = cfg.token_rows.resolve_role_sets(_FakeTokenizer())
        assert set(role_sets.trainable_row_ids) == {
            151646,
            151648,
            *range(151670, 152670),
        }
        assert len(role_sets.trainable_row_ids) == 1002
        assert cfg.objective.variant == "random_permutation_et_rmp_ce"
        assert cfg.objective.trie_support_weight == 2.0
        assert cfg.objective.trie_balance_weight == 1.0
        if "instance_trie_focused_cap8_frac0p04_mix0p1" in config_path.name:
            assert cfg.objective.coord_soft_ce is not None
            assert (
                cfg.objective.coord_soft_ce.target_distribution
                == "instance_trie_gaussian"
            )
            assert cfg.objective.coord_soft_ce.gaussian_mixture_weight == pytest.approx(
                0.1
            )
            assert cfg.objective.coord_soft_ce.gaussian_r95_axis_fraction == pytest.approx(
                0.04
            )
            assert cfg.objective.coord_soft_ce.gaussian_r95_cap_bins == 8
        if "instance_trie_focused_cap8_frac0p06_mix0p1" in config_path.name:
            assert cfg.objective.coord_soft_ce is not None
            assert (
                cfg.objective.coord_soft_ce.target_distribution
                == "instance_trie_gaussian"
            )
            assert cfg.objective.coord_soft_ce.gaussian_mixture_weight == pytest.approx(
                0.1
            )
            assert cfg.objective.coord_soft_ce.gaussian_r95_axis_fraction == pytest.approx(
                0.06
            )
            assert cfg.objective.coord_soft_ce.gaussian_r95_cap_bins == 8
        if "instance_trie_focused_cap8_frac0p04_mix0p2" in config_path.name:
            assert cfg.objective.coord_soft_ce is not None
            assert (
                cfg.objective.coord_soft_ce.target_distribution
                == "instance_trie_gaussian"
            )
            assert cfg.objective.coord_soft_ce.gaussian_mixture_weight == pytest.approx(
                0.2
            )
            assert cfg.objective.coord_soft_ce.gaussian_r95_axis_fraction == pytest.approx(
                0.04
            )
            assert cfg.objective.coord_soft_ce.gaussian_r95_cap_bins == 8
        if (
            "instance_trie_focused_cap8_frac0p04_mix0p1" in config_path.name
            or "instance_trie_focused_cap8_frac0p06_mix0p1" in config_path.name
            or "instance_trie_focused_cap8_frac0p04_mix0p2" in config_path.name
        ):
            assert cfg.objective.type_gate is not None
            assert cfg.objective.type_gate.enabled is True
            assert cfg.objective.type_gate.weights.struct == pytest.approx(1.0)
            assert cfg.objective.type_gate.weights.desc == pytest.approx(1.0)
            assert cfg.objective.type_gate.weights.coord == pytest.approx(1.0)
            assert cfg.objective.type_gate.weights.eos == pytest.approx(0.5)
        assert cfg.packing.static_packing is False
        assert cfg.packing.padding_free_packed is False
        assert cfg.training["packing"] is False
        assert cfg.training["optimizer"] == "multimodal_coord_offset"


def test_stage1_detection_teacher_forcing_canonical_launch_configs_parse() -> None:
    canonical_route = REPO_ROOT / "configs/stage1/detection_teacher_forcing"
    assert "stage1_detection_teacher_forcing" in (
        canonical_route / "README.md"
    ).read_text()
    config_paths = [
        canonical_route / "prod/compact_full_support2.yaml",
        canonical_route / "smoke/compact_full_tiny.yaml",
    ]

    for config_path in config_paths:
        cfg = ConfigLoader.load_materialized_training_config(str(config_path))
        assert isinstance(cfg, DetectionTrainingConfig)
        assert cfg.objective.id == "teacher_forcing"
        assert config_path.is_relative_to(canonical_route)
        assert cfg.objective.profile == "pure_valid_set_marginal"
        assert cfg.objective.target_ir.rollin_policy.name == "random_permutation"
        assert cfg.data.object_ordering == "random_permutation"
        assert cfg.detection_template.id == "compact_full"
        assert cfg.packing.static_packing is False
        assert cfg.packing.padding_free_packed is False
        assert cfg.training["packing"] is False
        assert cfg.training["eval_packing"] is False
        assert cfg.training["encoded_sample_cache"]["enabled"] is False
        assert "detection_teacher_forcing" in str(cfg.training["output_dir"])
        assert "detection_teacher_forcing" in str(cfg.training["logging_dir"])

@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_prefix_rollin_smoke_config_parses() -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        str(
            REPO_ROOT
            / "configs/stage1/recursive_detection_ce/smoke/compact_full_prefix_rollin_tiny.yaml"
        )
    )

    assert isinstance(cfg, DetectionTrainingConfig)
    assert cfg.training["max_steps"] == 1
    assert cfg.training["per_device_train_batch_size"] == 1
    assert cfg.training["effective_batch_size"] == 1
    assert "gradient_accumulation_steps" not in cfg.training
    assert cfg.objective.variant == "prefix_rollin_et_rmp_ce"
    assert cfg.detection_template.id == "compact_full"
    assert cfg.packing.static_packing is False
    assert cfg.packing.padding_free_packed is False


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_compact_sft_smoke_configs_parse_with_hard_ce_objectives() -> None:
    expected = {
        "compact_full_random_sft.yaml": (
            "random_permutation",
            "random_order_sft",
            False,
        ),
    }

    for file_name, (ordering, variant, prompt_variant) in expected.items():
        cfg = ConfigLoader.load_materialized_training_config(
            str(
                REPO_ROOT
                / "configs/stage1/recursive_detection_ce/smoke"
                / file_name
            )
        )
        assert isinstance(cfg, DetectionTrainingConfig)
        assert cfg.objective.id == "sft"
        assert cfg.objective.variant == variant
        assert cfg.objective.trie_support_weight == 0.0
        assert cfg.objective.trie_balance_weight == 0.0
        assert cfg.objective.normalization == "token_mean"
        assert cfg.data.object_ordering == ordering
        assert cfg.prompt.prompt_variant_enabled is prompt_variant


@pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")
def test_detection_recursive_detection_packing_preflight_config_is_failfast_only() -> None:
    config_path = (
        REPO_ROOT
        / "tests/fixtures/configs/stage1/recursive_detection_ce_static_packing_should_fail.yaml"
    )

    with pytest.raises(ValueError, match=r"recursive_detection_ce.*static packing"):
        ConfigLoader.load_materialized_training_config(str(config_path))
