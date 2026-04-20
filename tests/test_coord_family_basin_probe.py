from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.coord_family_basin_probe import (
    BasinProbeRow,
    run_basin_probe,
    summarize_basin_rows,
)


def test_summarize_basin_rows_reports_mass_at_4_and_local_error() -> None:
    rows = [
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
            slot="cx",
            center_value=500,
            target_value=500,
            candidate_value=500,
            score_mean=-0.1,
            abs_distance_to_target=0,
        ),
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
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
    assert summary[0]["slot"] == "cx"
    assert "mass_at_4" in summary[0]
    assert "local_expected_abs_error" in summary[0]


def test_summarize_basin_rows_rejects_slot_not_in_family_registry() -> None:
    rows = [
        BasinProbeRow(
            family_alias="cxcywh_pure_ce",
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


def test_run_basin_probe_materializes_summary_bundle(tmp_path: Path) -> None:
    config_path = tmp_path / "basin.yaml"
    output_dir = tmp_path / "analysis"
    config_path.write_text(
        f"""
run:
  name: coord-family-basin-smoke
  output_dir: {output_dir.as_posix()}

probe_rows:
  - family_alias: cxcywh_pure_ce
    slot: cx
    center_value: 500
    target_value: 500
    candidate_value: 500
    score_mean: -0.1
    abs_distance_to_target: 0
  - family_alias: cxcywh_pure_ce
    slot: cx
    center_value: 500
    target_value: 500
    candidate_value: 504
    score_mean: -0.2
    abs_distance_to_target: 4
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
    assert summary["slot_metrics"][0]["family_alias"] == "cxcywh_pure_ce"
    assert summary["slot_metrics"][0]["slot"] == "cx"


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
