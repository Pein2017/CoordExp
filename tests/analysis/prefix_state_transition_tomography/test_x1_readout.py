from __future__ import annotations

import pytest

from src.analysis.prefix_state_transition_tomography.x1_readout import (
    partition_x1_peaks,
    probe_forced_desc_pre_x1,
    summarize_x1_partitions,
)


def test_partition_x1_peaks_assigns_residual_emitted_and_artifact() -> None:
    peaks = [
        {"x1_bin": 100, "mass": 0.1},
        {"x1_bin": 250, "mass": 0.05},
        {"x1_bin": 999, "mass": 0.03},
        {"x1_bin": 400, "mass": 0.02},
    ]
    emitted = [{"gt_idx": 0, "bbox_xyxy": [100, 10, 150, 80]}]
    residual = [{"gt_idx": 1, "bbox_xyxy": [252, 20, 300, 90]}]

    assigned = partition_x1_peaks(peaks, emitted, residual, radius=24)

    assert [peak["partition"] for peak in assigned] == [
        "emitted_same_desc_x1_peak",
        "residual_same_desc_x1_peak",
        "boundary_artifact_x1_peak",
        "unmatched_x1_peak",
    ]
    assert assigned[1]["matched_residual_gt_indices"] == [1]


def test_partition_x1_peaks_marks_collision_when_near_emitted_and_residual() -> None:
    peaks = [{"x1_bin": 110, "mass": 0.1}]
    emitted = [{"gt_idx": 0, "bbox_xyxy": [100, 10, 150, 80]}]
    residual = [{"gt_idx": 1, "bbox_xyxy": [120, 20, 180, 90]}]

    assigned = partition_x1_peaks(peaks, emitted, residual, radius=24)

    assert assigned[0]["partition"] == "ambiguous_x1_collision"
    assert assigned[0]["matched_emitted_gt_indices"] == [0]
    assert assigned[0]["matched_residual_gt_indices"] == [1]


def test_summarize_x1_partitions_reports_coverage_and_rates() -> None:
    residual = [
        {"gt_idx": 1, "bbox_xyxy": [252, 20, 300, 90]},
        {"gt_idx": 2, "bbox_xyxy": [500, 20, 550, 90]},
    ]
    partitioned = [
        {"partition": "residual_same_desc_x1_peak", "matched_residual_gt_indices": [1]},
        {"partition": "emitted_same_desc_x1_peak", "matched_residual_gt_indices": []},
        {"partition": "unmatched_x1_peak", "matched_residual_gt_indices": []},
        {"partition": "boundary_artifact_x1_peak", "matched_residual_gt_indices": []},
    ]

    summary = summarize_x1_partitions(partitioned, residual)

    assert summary["residual_gt_covered_count"] == 1
    assert summary["forced_x1_residual_coverage"] == 0.5
    assert summary["emitted_attraction_rate"] == 0.25
    assert summary["unmatched_x1_peak_rate"] == 0.25
    assert summary["boundary_artifact_x1_peak_rate"] == 0.25


def test_probe_forced_desc_pre_x1_runtime_is_explicitly_deferred() -> None:
    with pytest.raises(NotImplementedError, match="paired_probe"):
        probe_forced_desc_pre_x1()

