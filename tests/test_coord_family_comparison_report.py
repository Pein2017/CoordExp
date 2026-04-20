from __future__ import annotations

import json
from pathlib import Path

from src.analysis.coord_family_comparison_report import (
    build_comparison_report,
    derive_family_verdicts,
)


def test_derive_family_verdicts_flags_family_with_high_bad_basin_as_risky() -> None:
    verdicts = derive_family_verdicts(
        basin_rows=[
            {
                "family_alias": "cxcywh_pure_ce",
                "mass_at_4": 0.82,
                "wrong_anchor_advantage_at_4": 0.22,
            }
        ],
        recall_rows=[
            {
                "family_alias": "cxcywh_pure_ce",
                "suppressed_fn_rate": 0.10,
                "competitive_fn_rate": 0.35,
            }
        ],
        vision_rows=[{"family_alias": "cxcywh_pure_ce", "vision_lift": 4.0}],
    )

    assert "cxcywh_pure_ce" in verdicts
    assert verdicts["cxcywh_pure_ce"]["family_health"] == "risky"


def test_build_comparison_report_materializes_summary_bundle(tmp_path: Path) -> None:
    basin_summary = tmp_path / "basin_summary.json"
    recall_summary = tmp_path / "recall_summary.json"
    config_path = tmp_path / "report.yaml"
    output_dir = tmp_path / "analysis"
    basin_summary.write_text(
        json.dumps(
            {
                "slot_metrics": [
                    {
                        "family_alias": "cxcywh_pure_ce",
                        "slot": "cx",
                        "mass_at_4": 0.82,
                        "wrong_anchor_advantage_at_4": 0.22,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    recall_summary.write_text(
        json.dumps(
            {
                "family_metrics": [
                    {
                        "family_alias": "cxcywh_pure_ce",
                        "suppressed_fn_rate": 0.10,
                        "competitive_fn_rate": 0.35,
                        "weak_visual_fn_rate": 0.55,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        f"""
run:
  name: coord-family-report-smoke
  output_dir: {output_dir.as_posix()}

inputs:
  basin_summary_json: {basin_summary.as_posix()}
  recall_summary_json: {recall_summary.as_posix()}
  vision_rows:
    - family_alias: cxcywh_pure_ce
      vision_lift: 4.0
        """.strip(),
        encoding="utf-8",
    )

    result = build_comparison_report(config_path)

    run_dir = output_dir / "coord-family-report-smoke"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    assert result["run_dir"] == str(run_dir)
    assert summary_path.exists()
    assert report_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_name"] == "coord-family-report-smoke"
    assert summary["verdicts"]["cxcywh_pure_ce"]["family_health"] == "risky"


def test_build_comparison_report_prefers_workspace_root_for_worktree_outputs(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / ".worktrees" / "feature"
    repo_root.mkdir(parents=True)
    basin_summary = workspace_root / "temp" / "basin_summary.json"
    recall_summary = workspace_root / "temp" / "recall_summary.json"
    basin_summary.parent.mkdir(parents=True)
    basin_summary.write_text(json.dumps({"slot_metrics": []}), encoding="utf-8")
    recall_summary.write_text(json.dumps({"family_metrics": []}), encoding="utf-8")
    config_path = repo_root / "report.yaml"
    config_path.write_text(
        f"""
run:
  name: coord-family-report-smoke
  output_dir: output/analysis

inputs:
  basin_summary_json: {basin_summary.as_posix()}
  recall_summary_json: {recall_summary.as_posix()}
  vision_rows: []
        """.strip(),
        encoding="utf-8",
    )

    result = build_comparison_report(config_path, repo_root=repo_root)

    assert result["run_dir"] == str(
        workspace_root / "output/analysis/coord-family-report-smoke"
    )
