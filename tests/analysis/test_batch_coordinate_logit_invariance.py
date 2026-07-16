from __future__ import annotations

import copy
import math

import pytest
import torch

from scripts.research.run_batch_coordinate_logit_invariance import (
    COMMON_COORDINATE_RECIPIENT_SUFFIX,
    EQUAL_LENGTH_MIXED_ROTATION_LAYOUT,
    EXPECTED_BATCH_FOUR_COORDINATE_BIN,
    EXPECTED_SINGLE_COORDINATE_BIN,
    HOMOGENEOUS_TARGET_COPIES_LAYOUT,
    MIXED_COMPANION_CONDITIONS,
    ORANGE_BOWL_WINDOW,
    PREDECESSOR_BATCH_CONDITIONS,
    PREDECESSOR_MIXED_LENGTH_ROTATION_LAYOUT,
    SINGLE_TARGET_LAYOUT,
    TARGET_ROW_WITH_TARGET_GEOMETRY,
    WHITE_BOWL_WINDOW,
    _trust_gate,
    build_cross_execution_summary,
    build_layouts,
    compare_coordinate_summaries,
    coordinate_token_ids,
    summarize_logits,
)


def _full_logits(*, white: float, orange: float, outside: float = 0.0) -> torch.Tensor:
    logits = torch.zeros(152700, dtype=torch.float32)
    logits[151670 + 700] = outside
    logits[151670 + EXPECTED_SINGLE_COORDINATE_BIN] = white
    logits[151670 + EXPECTED_BATCH_FOUR_COORDINATE_BIN] = orange
    return logits


def _row(summary: dict[str, object], coordinate_bin: int) -> dict[str, object]:
    return {
        "batch_position": 0,
        "condition_name": TARGET_ROW_WITH_TARGET_GEOMETRY,
        "is_target_recipient": True,
        "selected_coordinate_bin": coordinate_bin,
        "logit_summary": summary,
    }


def _execution(
    *,
    layout_name: str,
    layout_instance: str,
    repeat_index: int,
    coordinate_bins: list[int],
    summary: dict[str, object],
) -> dict[str, object]:
    rows = []
    for position, coordinate_bin in enumerate(coordinate_bins):
        row = _row(copy.deepcopy(summary), coordinate_bin)
        row["batch_position"] = position
        rows.append(row)
    direct_rows = copy.deepcopy(rows)
    return {
        "layout_name": layout_name,
        "layout_instance": layout_instance,
        "repeat_index": repeat_index,
        "cached_path": {"rows": rows},
        "direct_path": {"rows": direct_rows},
    }


def test_layouts_freeze_predecessor_homogeneous_and_equal_length_rotations() -> None:
    layouts = build_layouts()
    assert len(layouts) == 10
    assert layouts[0]["layout_name"] == SINGLE_TARGET_LAYOUT
    assert layouts[0]["condition_names"] == [TARGET_ROW_WITH_TARGET_GEOMETRY]
    predecessor_rotations = layouts[1:5]
    assert all(
        row["layout_name"] == PREDECESSOR_MIXED_LENGTH_ROTATION_LAYOUT
        for row in predecessor_rotations
    )
    assert {
        tuple(row["condition_names"]) for row in predecessor_rotations
    } == {
        tuple(
            [
                *PREDECESSOR_BATCH_CONDITIONS[left_rotation:],
                *PREDECESSOR_BATCH_CONDITIONS[:left_rotation],
            ]
        )
        for left_rotation in range(4)
    }
    assert {
        row["target_positions"][0] for row in predecessor_rotations
    } == {0, 1, 2, 3}
    assert layouts[5]["layout_name"] == HOMOGENEOUS_TARGET_COPIES_LAYOUT
    assert layouts[5]["condition_names"] == [TARGET_ROW_WITH_TARGET_GEOMETRY] * 4
    rotations = layouts[6:]
    assert all(row["layout_name"] == EQUAL_LENGTH_MIXED_ROTATION_LAYOUT for row in rotations)
    for target_position, layout in enumerate(rotations):
        assert layout["target_positions"] == [target_position]
        assert layout["condition_names"][target_position] == TARGET_ROW_WITH_TARGET_GEOMETRY
        companions = [
            name
            for index, name in enumerate(layout["condition_names"])
            if index != target_position
        ]
        assert tuple(companions) == MIXED_COMPANION_CONDITIONS


def test_suffix_and_coordinate_contract_are_exact() -> None:
    assert COMMON_COORDINATE_RECIPIENT_SUFFIX == (
        151646,
        65,
        9605,
        151647,
        151648,
    )
    identifiers = coordinate_token_ids()
    assert len(identifiers) == 1000
    assert identifiers[0] == 151670
    assert identifiers[-1] == 152669
    assert WHITE_BOWL_WINDOW == (149, 213)
    assert ORANGE_BOWL_WINDOW == (406, 470)
    assert WHITE_BOWL_WINDOW[1] < ORANGE_BOWL_WINDOW[0]


def test_summarize_logits_serializes_complete_coordinate_evidence() -> None:
    summary = summarize_logits(_full_logits(white=6.0, orange=4.0))
    assert summary["top_one_coordinate_bin"] == EXPECTED_SINGLE_COORDINATE_BIN
    assert summary["top_one_coordinate_token_id"] == 151670 + EXPECTED_SINGLE_COORDINATE_BIN
    assert len(summary["coordinate_raw_logits_float32"]) == 1000
    assert len(summary["coordinate_conditional_log_probabilities_float32"]) == 1000
    assert len(summary["coordinate_full_vocabulary_log_probabilities_float32"]) == 1000
    assert summary["white_bowl_conditional_probability_mass_float32"] > summary[
        "orange_bowl_conditional_probability_mass_float32"
    ]
    assert summary["white_minus_orange_log_probability_mass_margin_float32"] > 0
    assert len(summary["coordinate_raw_logits_float32_sha256"]) == 64


def test_centered_comparison_ignores_uniform_coordinate_logit_shift() -> None:
    reference = summarize_logits(_full_logits(white=6.0, orange=4.0))
    shifted = copy.deepcopy(reference)
    shifted["coordinate_raw_logits_float32"] = [
        value + 3.0 for value in reference["coordinate_raw_logits_float32"]
    ]
    comparison = compare_coordinate_summaries(reference, shifted)
    assert comparison["centered_maximum_absolute_difference_float32"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert comparison["centered_root_mean_square_difference_float32"] == pytest.approx(
        0.0, abs=1e-7
    )


def test_comparison_reports_window_margin_and_outside_shift_separately() -> None:
    reference = summarize_logits(_full_logits(white=6.0, orange=4.0, outside=0.0))
    candidate = summarize_logits(_full_logits(white=4.0, orange=6.0, outside=2.0))
    comparison = compare_coordinate_summaries(reference, candidate)
    assert comparison["white_minus_orange_margin_shift_float32"] < 0
    assert comparison["centered_root_mean_square_difference_float32"] > 0
    assert comparison[
        "outside_windows_centered_root_mean_square_difference_float32"
    ] > 0
    assert comparison["jensen_shannon_divergence_float32"] > 0


def test_trust_gate_requires_single_and_exact_predecessor_modes_on_both_repeats() -> None:
    white = summarize_logits(_full_logits(white=6.0, orange=4.0))
    orange = summarize_logits(_full_logits(white=4.0, orange=6.0))
    executions = []
    for repeat_index in range(2):
        executions.append(
            _execution(
                layout_name=SINGLE_TARGET_LAYOUT,
                layout_instance="single-target",
                repeat_index=repeat_index,
                coordinate_bins=[EXPECTED_SINGLE_COORDINATE_BIN],
                summary=white,
            )
        )
        executions.append(
            _execution(
                layout_name=PREDECESSOR_MIXED_LENGTH_ROTATION_LAYOUT,
                layout_instance="predecessor-mixed-length-target-position-1",
                repeat_index=repeat_index,
                coordinate_bins=[EXPECTED_BATCH_FOUR_COORDINATE_BIN],
                summary=orange,
            )
        )
    assert _trust_gate(executions)["passed"] is True
    executions[-1]["cached_path"]["rows"][0]["selected_coordinate_bin"] = 999
    failed = _trust_gate(executions)
    assert failed["passed"] is False
    assert failed["failure_disposition"] == "execution_state_mismatch_stop_interpretation"


def test_cross_execution_summary_uses_single_repeat_zero_as_reference() -> None:
    white = summarize_logits(_full_logits(white=6.0, orange=4.0))
    orange = summarize_logits(_full_logits(white=4.0, orange=6.0))
    executions = []
    for repeat_index in range(2):
        executions.append(
            _execution(
                layout_name=SINGLE_TARGET_LAYOUT,
                layout_instance="single-target",
                repeat_index=repeat_index,
                coordinate_bins=[EXPECTED_SINGLE_COORDINATE_BIN],
                summary=white,
            )
        )
        executions.append(
            _execution(
                layout_name=HOMOGENEOUS_TARGET_COPIES_LAYOUT,
                layout_instance="homogeneous-target-copies",
                repeat_index=repeat_index,
                coordinate_bins=[EXPECTED_BATCH_FOUR_COORDINATE_BIN] * 4,
                summary=orange,
            )
        )
    summary = build_cross_execution_summary(executions)
    assert summary["cached_path"]["reference"] == {
        "layout_instance": "single-target",
        "repeat_index": 0,
        "batch_position": 0,
    }
    homogeneous = [
        row
        for row in summary["cached_path"]["comparisons"]
        if row["layout_name"] == HOMOGENEOUS_TARGET_COPIES_LAYOUT
    ]
    assert homogeneous
    assert all(row["white_minus_orange_margin_shift_float32"] < 0 for row in homogeneous)
    assert all(math.isfinite(row["absolute_margin_shift_divided_by_repeat_noise"]) for row in homogeneous)
