from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.x1_candidate_field import (
    _write_progress,
    detect_projection_collision,
    extract_x1_peaks,
)


def test_extract_x1_peaks_merges_nearby_modes_and_reports_coverage() -> None:
    probs = {10: 0.30, 12: 0.20, 100: 0.25, 180: 0.20, 181: 0.11, 400: 0.001}

    result = extract_x1_peaks(probs, gt_x1_values=[10, 12, 100, 180, 181], merge_radius=24)

    assert result["merged_peak_count"] == 3
    assert result["gt_instance_coverage_count"] == 5


def test_projection_collision_flags_vertical_same_x1_instances() -> None:
    boxes = [(100, 10, 140, 50), (105, 200, 145, 250)]

    assert detect_projection_collision(boxes, primary_merge_radius=24)


def test_write_progress_records_json_safe_shard_heartbeat(tmp_path) -> None:
    progress_path = tmp_path / "x1_candidate_field_progress.json"

    _write_progress(
        progress_path,
        shard_id=3,
        planned_cases=100,
        emitted_rows=25,
        valid_rows=24,
        checkpoint_path=tmp_path / "checkpoint-3664",
        started_at=0.0,
        last_probe_plan_row_id="pp-000025",
        status="running",
    )

    import json

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["stage"] == "x1_candidate_field"
    assert progress["status"] == "running"
    assert progress["shard_id"] == 3
    assert progress["planned_cases"] == 100
    assert progress["emitted_rows"] == 25
    assert progress["valid_rows"] == 24
    assert progress["remaining_cases"] == 75
    assert progress["progress_fraction"] == 0.25
    assert progress["last_probe_plan_row_id"] == "pp-000025"
    assert progress["checkpoint_path"].endswith("checkpoint-3664")
