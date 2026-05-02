from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.bootstrap.trainer_setup import compose_trainer_class
from src.config.schema import PromptOverrides, TrainingConfig
from src.sft import (
    _inject_stage1_set_continuation_trainer_config,
    resolve_trainer_cls,
)
from src.trainers.metrics.mixins import (
    CoordSoftCEW1LossMixin,
    GradAccumLossScaleMixin,
)


def _base_training_payload() -> dict:
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


def _stage1_set_continuation_payload() -> dict:
    payload = _base_training_payload()
    payload["custom"].update(
        {
            "trainer_variant": "stage1_set_continuation",
            "coord_tokens": {
                "enabled": True,
                "skip_bbox_norm": True,
            },
            "stage1_set_continuation": {
                "subset_sampling": {
                    "empty_prefix_ratio": 0.30,
                    "random_subset_ratio": 0.45,
                    "leave_one_out_ratio": 0.20,
                    "full_prefix_ratio": 0.05,
                    "prefix_order": "random",
                },
                "candidates": {
                    "mode": "uniform_subsample",
                    "max_candidates": 8,
                    "tail_positive_count": 1,
                },
                "structural_close": {
                    "close_start_suppression_weight": 0.1,
                    "final_schema_close_weight": 0.2,
                    "json_structural_weight": 0.05,
                    "annotation_completeness_weight": {
                        "enabled": True,
                        "source": "original_ckpt1332_val200_fp_as_unlabeled",
                        "by_max_gt": {
                            1: 0.9500,
                            3: 0.8306,
                        },
                    },
                },
                "bidirectional_token_gate": {
                    "enabled": True,
                    "coord_gate_weight": 0.5,
                    "text_gate_weight": 0.1,
                    "temperature": 1.0,
                    "scope": "objective_tokens",
                },
                "positive_evidence_margin": {
                    "objective": "threshold_loss",
                    "threshold_space": "full_entry_logZ",
                    "log_rho": -4.2,
                    "threshold_calibration": "calib-v1",
                },
            },
        }
    )
    return payload


def test_stage1_set_continuation_parses_successfully() -> None:
    cfg = TrainingConfig.from_mapping(
        _stage1_set_continuation_payload(), PromptOverrides()
    )

    stage1_cfg = cfg.custom.stage1_set_continuation
    assert stage1_cfg.subset_sampling.empty_prefix_ratio == pytest.approx(0.30)
    assert stage1_cfg.subset_sampling.random_subset_ratio == pytest.approx(0.45)
    assert stage1_cfg.subset_sampling.leave_one_out_ratio == pytest.approx(0.20)
    assert stage1_cfg.subset_sampling.full_prefix_ratio == pytest.approx(0.05)
    assert stage1_cfg.subset_sampling.prefix_order == "random"
    assert stage1_cfg.candidates.mode == "uniform_subsample"
    assert stage1_cfg.candidates.max_candidates == 8
    assert stage1_cfg.candidates.tail_positive_count == 1
    assert stage1_cfg.structural_close.close_start_suppression_weight == pytest.approx(
        0.1
    )
    assert stage1_cfg.structural_close.final_schema_close_weight == pytest.approx(0.2)
    assert stage1_cfg.structural_close.json_structural_weight == pytest.approx(0.05)
    completeness = stage1_cfg.structural_close.annotation_completeness_weight
    assert completeness.enabled is True
    assert completeness.source == "original_ckpt1332_val200_fp_as_unlabeled"
    assert set(completeness.by_max_gt) == {1, 3}
    assert completeness.by_max_gt[1] == pytest.approx(0.9500)
    assert completeness.by_max_gt[3] == pytest.approx(0.8306)
    gate = stage1_cfg.bidirectional_token_gate
    assert gate.enabled is True
    assert gate.coord_gate_weight == pytest.approx(0.5)
    assert gate.text_gate_weight == pytest.approx(0.1)
    assert gate.temperature == pytest.approx(1.0)
    assert gate.scope == "objective_tokens"
    assert stage1_cfg.positive_evidence_margin.objective == "threshold_loss"
    assert stage1_cfg.positive_evidence_margin.threshold_space == "full_entry_logZ"
    assert stage1_cfg.positive_evidence_margin.rho is None
    assert stage1_cfg.positive_evidence_margin.log_rho == pytest.approx(-4.2)
    assert stage1_cfg.objective.mode == "candidate_balanced"
    assert stage1_cfg.objective.suffix_order == "random"
    assert stage1_cfg.positive_evidence_margin.threshold_calibration == "calib-v1"
    assert stage1_cfg.metric_schema_version == "stage1_set_continuation_metrics_v2"


def test_training_config_accepts_compact_detection_sequence_format() -> None:
    payload = _base_training_payload()
    payload["custom"].update(
        {
            "coord_tokens": {
                "enabled": True,
                "skip_bbox_norm": True,
            },
            "detection_sequence_format": "compact_full",
        }
    )

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    assert cfg.custom.detection_sequence_format == "compact_full"


def test_stage1_set_continuation_parses_entry_trie_rmp_objective() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["objective"] = {
        "mode": "entry_trie_rmp_ce",
        "suffix_order": "dataset",
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    objective = cfg.custom.stage1_set_continuation.objective
    assert objective.mode == "entry_trie_rmp_ce"
    assert objective.suffix_order == "dataset"
    assert objective.branch_support_weight == pytest.approx(1.0)
    assert objective.branch_balance_weight == pytest.approx(1.0)


def test_stage1_set_continuation_parses_entry_trie_rmp_branch_weights() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["objective"] = {
        "mode": "entry_trie_rmp_ce",
        "suffix_order": "random",
        "branch_support_weight": 2.0,
        "branch_balance_weight": 1.0,
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    objective = cfg.custom.stage1_set_continuation.objective
    assert objective.mode == "entry_trie_rmp_ce"
    assert objective.branch_support_weight == pytest.approx(2.0)
    assert objective.branch_balance_weight == pytest.approx(1.0)


def test_stage1_set_continuation_rejects_negative_entry_trie_rmp_branch_weights() -> (
    None
):
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["objective"] = {
        "mode": "entry_trie_rmp_ce",
        "branch_support_weight": -1.0,
    }

    with pytest.raises(ValueError, match="branch_support_weight"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage1_set_continuation_rejects_invalid_objective_mode() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["objective"] = {
        "mode": "chunk_logz",
    }

    with pytest.raises(ValueError, match="objective.mode"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage1_set_continuation_accepts_legacy_close_and_pem_names() -> None:
    payload = _stage1_set_continuation_payload()
    stage1_payload = payload["custom"]["stage1_set_continuation"]
    stage1_payload["structural_close"] = {
        "anti_close_weight": 0.3,
        "final_close_weight": 0.4,
    }
    stage1_payload["positive_evidence_margin"] = {
        "mode": "replace_mp",
        "threshold_space": "full_entry_logZ",
        "log_rho": -3.0,
        "threshold_calibration": "legacy-calib",
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())

    stage1_cfg = cfg.custom.stage1_set_continuation
    assert stage1_cfg.structural_close.close_start_suppression_weight == pytest.approx(
        0.3
    )
    assert stage1_cfg.structural_close.final_schema_close_weight == pytest.approx(0.4)
    assert stage1_cfg.positive_evidence_margin.objective == "threshold_loss"
    assert stage1_cfg.positive_evidence_margin.log_rho == pytest.approx(-3.0)


def test_stage1_set_continuation_unknown_nested_key_reports_dotted_path() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["subset_sampling"]["unknown_key"] = 1

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.stage1_set_continuation.subset_sampling.unknown_key" in str(
        exc.value
    )


def test_stage1_set_continuation_rejects_unimplemented_canonical_prefix_order() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["subset_sampling"]["prefix_order"] = (
        "canonical"
    )

    with pytest.raises(ValueError, match="prefix_order"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


@pytest.mark.parametrize(
    ("coord_tokens_patch", "error_text"),
    [
        ({"enabled": False, "skip_bbox_norm": True}, "custom.coord_tokens.enabled"),
        (
            {"enabled": True, "skip_bbox_norm": False},
            "custom.coord_tokens.skip_bbox_norm",
        ),
    ],
)
def test_stage1_set_continuation_requires_coord_token_contract(
    coord_tokens_patch: dict[str, bool], error_text: str
) -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["coord_tokens"] = coord_tokens_patch

    with pytest.raises(ValueError, match=error_text):
        TrainingConfig.from_mapping(payload, PromptOverrides())


@pytest.mark.parametrize(
    ("training_patch", "error_text"),
    [
        ({"packing": True, "eval_packing": False}, "training\\.packing=false"),
        ({"packing": False, "eval_packing": True}, "training\\.eval_packing=false"),
    ],
)
def test_stage1_set_continuation_rejects_packing_flags(
    training_patch: dict[str, bool], error_text: str
) -> None:
    payload = _stage1_set_continuation_payload()
    payload["training"] = dict(training_patch)

    with pytest.raises(ValueError, match=error_text):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage1_set_continuation_positive_evidence_margin_requires_exactly_one_threshold() -> (
    None
):
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["positive_evidence_margin"] = {
        "objective": "threshold_loss",
        "threshold_space": "full_entry_logZ",
        "rho": 0.75,
        "log_rho": -0.28,
    }

    with pytest.raises(ValueError, match=r"exactly one of rho/log_rho"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


@pytest.mark.parametrize(
    ("gate_patch", "error_text"),
    [
        (
            {"enabled": True, "coord_gate_weight": 0.0, "text_gate_weight": 0.0},
            "both 0",
        ),
        (
            {"enabled": True, "coord_gate_weight": -0.1, "text_gate_weight": 0.1},
            "coord_gate_weight",
        ),
        (
            {"enabled": True, "coord_gate_weight": 0.1, "text_gate_weight": -0.1},
            "text_gate_weight",
        ),
        (
            {
                "enabled": True,
                "coord_gate_weight": 0.1,
                "text_gate_weight": 0.1,
                "temperature": 0.0,
            },
            "temperature",
        ),
        (
            {
                "enabled": True,
                "coord_gate_weight": 0.1,
                "text_gate_weight": 0.1,
                "scope": "candidate_tokens",
            },
            "scope",
        ),
    ],
)
def test_stage1_set_continuation_bidirectional_gate_validates_config(
    gate_patch: dict[str, object], error_text: str
) -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["bidirectional_token_gate"] = (
        gate_patch
    )

    with pytest.raises(ValueError, match=error_text):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage1_set_continuation_positive_evidence_margin_requires_calibration_note() -> (
    None
):
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["positive_evidence_margin"] = {
        "objective": "threshold_loss",
        "threshold_space": "full_entry_logZ",
        "rho": 0.75,
    }

    with pytest.raises(ValueError, match="threshold_calibration"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage1_set_continuation_rejects_fixed_rho_full_entry_logz_threshold() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["positive_evidence_margin"] = {
        "objective": "threshold_loss",
        "threshold_space": "full_entry_logZ",
        "rho": 0.75,
        "threshold_calibration": "fixed-rho-ablation",
    }

    with pytest.raises(ValueError, match="log_rho"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_resolve_trainer_cls_routes_stage1_set_continuation_variant() -> None:
    trainer_cls = resolve_trainer_cls(
        SimpleNamespace(trainer_variant="stage1_set_continuation")
    )

    assert trainer_cls is not None
    assert any(
        base.__name__ == "Stage1SetContinuationTrainer" for base in trainer_cls.__mro__
    )


def test_inject_stage1_set_continuation_trainer_config_sets_runtime_attrs() -> None:
    cfg = TrainingConfig.from_mapping(
        _stage1_set_continuation_payload(), PromptOverrides()
    )
    trainer = SimpleNamespace()

    _inject_stage1_set_continuation_trainer_config(
        trainer=trainer,
        training_config=cfg,
    )

    assert trainer.stage1_set_continuation_cfg is cfg.custom.stage1_set_continuation
    assert trainer.object_field_order == str(cfg.custom.object_field_order)
    assert trainer.object_ordering == str(cfg.custom.object_ordering)


def test_compose_trainer_class_skips_ordinary_stage1_mixins_for_set_continuation() -> (
    None
):
    class _BaseTrainer:
        pass

    trainer_cls = compose_trainer_class(
        trainer_cls=_BaseTrainer,
        trainer_variant="stage1_set_continuation",
        instability_monitor_cfg={"enabled": True},
        token_type_cfg=SimpleNamespace(enabled=True),
        bbox_geo_cfg=SimpleNamespace(enabled=True),
        bbox_size_aux_cfg=SimpleNamespace(enabled=True),
        coord_soft_ce_w1_cfg=SimpleNamespace(enabled=True),
    )

    assert trainer_cls is _BaseTrainer
    assert not issubclass(trainer_cls, CoordSoftCEW1LossMixin)
    assert not issubclass(trainer_cls, GradAccumLossScaleMixin)
