from __future__ import annotations

import json
from pathlib import Path

from src.analysis.coord_family_recall_probe import (
    FnProbeRow,
    classify_fn_mechanism,
    run_recall_probe,
    summarize_fn_rows,
)


def test_classify_fn_mechanism_marks_suppressed_when_support_and_recovery_are_high() -> None:
    label = classify_fn_mechanism(
        teacher_forced_support=0.75,
        proposal_support=0.80,
        oracle_k_recovered=True,
        competitor_margin=-0.05,
    )

    assert label == "suppressed_fn"


def test_classify_fn_mechanism_marks_competitive_when_competitor_margin_is_large() -> None:
    label = classify_fn_mechanism(
        teacher_forced_support=0.45,
        proposal_support=0.50,
        oracle_k_recovered=False,
        competitor_margin=0.30,
    )

    assert label == "competitive_fn"


def test_summarize_fn_rows_reports_per_family_mechanism_rates() -> None:
    rows = [
        FnProbeRow(
            family_alias="cxcywh_pure_ce",
            teacher_forced_support=0.75,
            proposal_support=0.80,
            oracle_k_recovered=True,
            competitor_margin=-0.05,
        ),
        FnProbeRow(
            family_alias="cxcywh_pure_ce",
            teacher_forced_support=0.45,
            proposal_support=0.50,
            oracle_k_recovered=False,
            competitor_margin=0.30,
        ),
    ]

    summary = summarize_fn_rows(rows)

    assert summary[0]["family_alias"] == "cxcywh_pure_ce"
    assert summary[0]["suppressed_fn_rate"] == 0.5
    assert summary[0]["competitive_fn_rate"] == 0.5
    assert summary[0]["weak_visual_fn_rate"] == 0.0


def test_run_recall_probe_materializes_summary_bundle(tmp_path: Path) -> None:
    config_path = tmp_path / "recall.yaml"
    output_dir = tmp_path / "analysis"
    config_path.write_text(
        f"""
run:
  name: coord-family-recall-smoke
  output_dir: {output_dir.as_posix()}

fn_rows:
  - family_alias: cxcywh_pure_ce
    teacher_forced_support: 0.75
    proposal_support: 0.80
    oracle_k_recovered: true
    competitor_margin: -0.05
  - family_alias: cxcywh_pure_ce
    teacher_forced_support: 0.45
    proposal_support: 0.50
    oracle_k_recovered: false
    competitor_margin: 0.30
        """.strip(),
        encoding="utf-8",
    )

    result = run_recall_probe(config_path)

    run_dir = output_dir / "coord-family-recall-smoke"
    summary_path = run_dir / "summary.json"
    rows_path = run_dir / "fn_rows.jsonl"
    assert result["run_dir"] == str(run_dir)
    assert summary_path.exists()
    assert rows_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_name"] == "coord-family-recall-smoke"
    assert summary["family_metrics"][0]["family_alias"] == "cxcywh_pure_ce"


def test_run_recall_probe_prefers_workspace_root_for_worktree_outputs(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / ".worktrees" / "feature"
    repo_root.mkdir(parents=True)
    config_path = repo_root / "recall.yaml"
    config_path.write_text(
        """
run:
  name: coord-family-recall-smoke
  output_dir: output/analysis

fn_rows:
  - family_alias: base_xyxy_merged
    teacher_forced_support: 0.30
    proposal_support: 0.25
    oracle_k_recovered: false
    competitor_margin: 0.05
        """.strip(),
        encoding="utf-8",
    )

    result = run_recall_probe(config_path, repo_root=repo_root)

    assert result["run_dir"] == str(
        workspace_root / "output/analysis/coord-family-recall-smoke"
    )
