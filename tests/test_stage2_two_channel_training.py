"""Canonical Stage-2 two-channel import-surface tests."""

import importlib

import pytest

from test_stage2_ab_training import *  # noqa: F401,F403


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


def test_stage2_two_channel_target_builder_helper_import_surface_remains_available() -> None:
    from src.trainers.stage2_two_channel.target_builder import (
        _apply_channel_b_duplicate_control,
        _build_canonical_prefix_data,
        _build_canonical_prefix_text_data,
        _build_duplicate_control_divergence_diagnostics,
        _compute_duplicate_diagnostics,
    )

    assert callable(_build_canonical_prefix_text_data)
    assert callable(_build_canonical_prefix_data)
    assert callable(_build_duplicate_control_divergence_diagnostics)
    assert callable(_compute_duplicate_diagnostics)
    assert callable(_apply_channel_b_duplicate_control)


@pytest.mark.parametrize(
    "helper_name",
    [
        "_apply_channel_b_duplicate_control",
        "_bbox_iou_norm1000_xyxy",
        "_build_canonical_prefix_data",
        "_build_canonical_prefix_text_data",
        "_build_channel_b_meta_entry",
        "_build_channel_b_supervision_targets",
        "_build_channel_b_triage",
        "_channel_b_targets",
        "_compute_duplicate_diagnostics",
        "_build_duplicate_control_divergence_diagnostics",
        "_sequential_dedup_bbox_objects",
    ],
)
def test_stage2_two_channel_package_does_not_export_target_builder_helpers(
    helper_name: str,
) -> None:
    import src.trainers.stage2_two_channel as stage2_two_channel

    assert not hasattr(stage2_two_channel, helper_name)
    with pytest.raises(ImportError):
        exec(f"from src.trainers.stage2_two_channel import {helper_name}", {})
