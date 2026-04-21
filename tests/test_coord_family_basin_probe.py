from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.coord_family_basin_probe import (
    BasinProbeRow,
    summarize_canonical_basin_rows,
    run_basin_probe,
    summarize_basin_rows,
)


def test_summarize_basin_rows_reports_mass_at_4_and_local_error() -> None:
    rows = [
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            probe_id="case-a",
            slot="cx",
            center_value=500,
            target_value=500,
            candidate_value=500,
            score_mean=-0.1,
            abs_distance_to_target=0,
        ),
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            probe_id="case-a",
            slot="cx",
            center_value=500,
            target_value=500,
            candidate_value=504,
            score_mean=-0.2,
            abs_distance_to_target=4,
        ),
    ]

    summary = summarize_basin_rows(rows)

    assert summary[0]["family_alias"] == "cxcywh_pure_ce"
    assert summary[0]["probe_id"] == "case-a"
    assert summary[0]["slot"] == "cx"
    assert summary[0]["bbox_format"] == "cxcywh"
    assert summary[0]["pred_coord_mode"] == "norm1000"
    assert summary[0]["canonical_projection_kind"] == "cxcywh_to_xyxy"
    assert summary[0]["native_slots"] == ["cx", "cy", "w", "h"]
    assert "mass_at_4" in summary[0]
    assert "local_expected_abs_error" in summary[0]


def test_summarize_basin_rows_rejects_slot_not_in_family_registry() -> None:
    rows = [
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            probe_id="case-a",
            slot="x1",
            center_value=500,
            target_value=500,
            candidate_value=500,
            score_mean=-0.1,
            abs_distance_to_target=0,
        )
    ]

    with pytest.raises(ValueError, match="slot"):
        summarize_basin_rows(rows)


def test_summarize_canonical_basin_rows_projects_when_native_bbox_context_is_available() -> None:
    rows = [
        BasinProbeRow(
            family_alias="base_xyxy_merged",
            probe_id="case-base",
            slot="x1",
            center_value=100,
            target_value=100,
            candidate_value=100,
            score_mean=-0.05,
            abs_distance_to_target=0,
            native_target_values=(100, 200, 300, 400),
            native_center_values=(100, 200, 300, 400),
            image_width=1000,
            image_height=1000,
        ),
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            probe_id="case-a",
            slot="cx",
            center_value=500,
            target_value=500,
            candidate_value=500,
            score_mean=-0.1,
            abs_distance_to_target=0,
            native_target_values=(500, 500, 200, 100),
            native_center_values=(500, 500, 200, 100),
        ),
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            probe_id="case-a",
            slot="cx",
            center_value=500,
            target_value=500,
            candidate_value=504,
            score_mean=-0.2,
            abs_distance_to_target=4,
            native_target_values=(500, 500, 200, 100),
            native_center_values=(500, 500, 200, 100),
        ),
        BasinProbeRow(
            family_alias="center_parameterization",
            probe_id="case-b",
            slot="cx",
            center_value=300,
            target_value=300,
            candidate_value=300,
            score_mean=-0.4,
            abs_distance_to_target=0,
            native_target_values=(300, 300, 50, 40),
            native_center_values=(300, 300, 50, 40),
        ),
    ]

    summary = summarize_canonical_basin_rows(rows)

    assert [row["family_alias"] for row in summary["probe_metrics"]] == [
        "base_xyxy_merged",
        "cxcywh_pure_ce",
    ]
    assert summary["probe_metrics"][0]["canonical_compare_group"] == "canonical_xyxy_norm1000"
    assert summary["probe_metrics"][1]["canonical_compare_group"] == "canonical_xyxy_norm1000"
    assert summary["probe_metrics"][1]["pred_coord_mode"] == "norm1000"
    assert summary["probe_metrics"][1]["target_bbox_metrics"]["mass_at_4"] > 0.0
    assert [row["family_alias"] for row in summary["family_rollup"]] == [
        "base_xyxy_merged",
        "cxcywh_pure_ce",
    ]
    assert summary["skipped"] == [
        {
            "family_alias": "center_parameterization",
            "probe_id": "case-b",
            "reason": "missing_image_size_for_pixel_projection",
        }
    ]


def test_run_basin_probe_materializes_summary_bundle(tmp_path: Path) -> None:
    config_path = tmp_path / "basin.yaml"
    output_dir = tmp_path / "analysis"
    config_path.write_text(
        f"""
run:
  name: coord-family-basin-smoke
  output_dir: {output_dir.as_posix()}

probe_rows:
  - family_alias: base_xyxy_merged
    probe_id: case-base
    slot: x1
    center_value: 100
    target_value: 100
    candidate_value: 100
    score_mean: -0.05
    abs_distance_to_target: 0
    native_target_values: [100, 200, 300, 400]
    native_center_values: [100, 200, 300, 400]
    image_width: 1000
    image_height: 1000
  - family_alias: cxcywh_pure_ce
    probe_id: case-a
    slot: cx
    center_value: 500
    target_value: 500
    candidate_value: 500
    score_mean: -0.1
    abs_distance_to_target: 0
    native_target_values: [500, 500, 200, 100]
    native_center_values: [500, 500, 200, 100]
  - family_alias: cxcywh_pure_ce
    probe_id: case-a
    slot: cx
    center_value: 500
    target_value: 500
    candidate_value: 504
    score_mean: -0.2
    abs_distance_to_target: 4
    native_target_values: [500, 500, 200, 100]
    native_center_values: [500, 500, 200, 100]
        """.strip(),
        encoding="utf-8",
    )

    result = run_basin_probe(config_path)

    run_dir = output_dir / "coord-family-basin-smoke"
    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "basin_rows.jsonl"
    assert result["run_dir"] == str(run_dir)
    assert summary_path.exists()
    assert rows_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_name"] == "coord-family-basin-smoke"
    assert len(summary["family_native_slot_metrics"]) == 2
    assert [row["family_alias"] for row in summary["canonical_comparison_view"]["probe_metrics"]] == [
        "base_xyxy_merged",
        "cxcywh_pure_ce",
    ]
    assert result["family_native_metric_count"] == 2
    assert result["canonical_metric_count"] == 2


def test_run_basin_probe_prefers_workspace_root_for_worktree_outputs(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / ".worktrees" / "feature"
    repo_root.mkdir(parents=True)
    config_path = repo_root / "basin.yaml"
    config_path.write_text(
        """
run:
  name: coord-family-basin-smoke
  output_dir: output/analysis

probe_rows:
  - family_alias: base_xyxy_merged
    probe_id: case-a
    slot: x1
    center_value: 500
    target_value: 500
    candidate_value: 500
    score_mean: -0.1
    abs_distance_to_target: 0
        """.strip(),
        encoding="utf-8",
    )

    result = run_basin_probe(config_path, repo_root=repo_root)

    assert result["run_dir"] == str(
        workspace_root / "output/analysis/coord-family-basin-smoke"
    )
