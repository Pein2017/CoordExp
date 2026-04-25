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
                },
                "structural_close": {
                    "anti_close_weight": 0.1,
                    "final_close_weight": 0.2,
                },
                "positive_evidence_margin": {
                    "mode": "replace_mp",
                    "threshold_space": "full_entry_logZ",
                    "rho": 0.75,
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
    assert stage1_cfg.structural_close.anti_close_weight == pytest.approx(0.1)
    assert stage1_cfg.structural_close.final_close_weight == pytest.approx(0.2)
    assert stage1_cfg.positive_evidence_margin.mode == "replace_mp"
    assert (
        stage1_cfg.positive_evidence_margin.threshold_space == "full_entry_logZ"
    )
    assert stage1_cfg.positive_evidence_margin.rho == pytest.approx(0.75)
    assert stage1_cfg.positive_evidence_margin.log_rho is None
    assert stage1_cfg.positive_evidence_margin.threshold_calibration == "calib-v1"
    assert (
        stage1_cfg.metric_schema_version == "stage1_set_continuation_metrics_v1"
    )


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
    payload["custom"]["stage1_set_continuation"]["subset_sampling"][
        "prefix_order"
    ] = "canonical"

    with pytest.raises(ValueError, match="prefix_order"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


@pytest.mark.parametrize(
    ("coord_tokens_patch", "error_text"),
    [
        ({"enabled": False, "skip_bbox_norm": True}, "custom.coord_tokens.enabled"),
        ({"enabled": True, "skip_bbox_norm": False}, "custom.coord_tokens.skip_bbox_norm"),
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


def test_stage1_set_continuation_positive_evidence_margin_requires_exactly_one_threshold() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["positive_evidence_margin"] = {
        "mode": "replace_mp",
        "threshold_space": "full_entry_logZ",
        "rho": 0.75,
        "log_rho": -0.28,
    }

    with pytest.raises(ValueError, match=r"exactly one of rho/log_rho"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_stage1_set_continuation_positive_evidence_margin_requires_calibration_note() -> None:
    payload = _stage1_set_continuation_payload()
    payload["custom"]["stage1_set_continuation"]["positive_evidence_margin"] = {
        "mode": "replace_mp",
        "threshold_space": "full_entry_logZ",
        "rho": 0.75,
    }

    with pytest.raises(ValueError, match="threshold_calibration"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_resolve_trainer_cls_routes_stage1_set_continuation_variant() -> None:
    trainer_cls = resolve_trainer_cls(
        SimpleNamespace(trainer_variant="stage1_set_continuation")
    )

    assert trainer_cls is not None
    assert any(
        base.__name__ == "Stage1SetContinuationTrainer"
        for base in trainer_cls.__mro__
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


def test_compose_trainer_class_skips_ordinary_stage1_mixins_for_set_continuation() -> None:
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
