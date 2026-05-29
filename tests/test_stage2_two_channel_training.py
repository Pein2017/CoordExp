"""Canonical Stage-2 rollout-correction import-surface tests."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from test_stage2_ab_training import *  # noqa: F401,F403
from src.trainers.stage2_rollout_correction import Stage2RolloutCorrectionTrainer


@pytest.mark.parametrize(
    "module_name",
    [
        "src.trainers.stage2_ab",
        "src.trainers.stage2_ab_training",
    ],
)
def test_legacy_stage2_ab_import_paths_are_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_stage2_rollout_correction_target_builder_helper_import_surface_remains_available() -> None:
    from src.trainers.rollout_correction.target_builder import (
        _apply_rollout_correction_duplicate_control,
        _build_canonical_prefix_data,
        _build_canonical_prefix_text_data,
        _build_duplicate_control_divergence_diagnostics,
        _compute_duplicate_diagnostics,
    )

    assert callable(_build_canonical_prefix_text_data)
    assert callable(_build_canonical_prefix_data)
    assert callable(_build_duplicate_control_divergence_diagnostics)
    assert callable(_compute_duplicate_diagnostics)
    assert callable(_apply_rollout_correction_duplicate_control)


def test_stage2_trainer_keeps_teacher_forcing_packing_guard_before_forward() -> None:
    class _Model:
        def __call__(self, **kwargs):  # pragma: no cover - guard must fire first
            raise AssertionError("model forward should not run for unsupported packing")

    trainer = object.__new__(Stage2RolloutCorrectionTrainer)
    trainer.teacher_forcing_objective_cfg = SimpleNamespace(enabled=True)
    trainer._packing_enabled = lambda: True

    with pytest.raises(
        ValueError,
        match=r"teacher_forcing.*stage2_rollout_correction.*packing",
    ):
        trainer.compute_loss(
            _Model(),
            {
                "_rollout_matching_meta": [],
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            },
        )


@pytest.mark.parametrize(
    "helper_name",
    [
        "_apply_rollout_correction_duplicate_control",
        "_bbox_iou_norm1000_xyxy",
        "_build_canonical_prefix_data",
        "_build_canonical_prefix_text_data",
        "_build_rollout_correction_meta_entry",
        "_build_rollout_correction_supervision_targets",
        "_build_rollout_correction_triage",
        "_correction_targets",
        "_compute_duplicate_diagnostics",
        "_build_duplicate_control_divergence_diagnostics",
        "_sequential_dedup_bbox_objects",
    ],
)
def test_stage2_rollout_correction_package_does_not_export_target_builder_helpers(
    helper_name: str,
) -> None:
    import src.trainers.rollout_correction as rollout_correction

    assert not hasattr(rollout_correction, helper_name)
    with pytest.raises(ImportError):
        exec(f"from src.trainers.rollout_correction import {helper_name}", {})
