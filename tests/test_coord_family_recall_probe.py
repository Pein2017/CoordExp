from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.coord_family_recall_probe import (
    ArtifactRecallSource,
    build_fn_rows_from_artifact_sources,
    FnProbeRow,
    classify_fn_mechanism,
    label_fn_rows,
    load_recall_probe_config,
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
    assert summary[0]["suppressed_fn_count"] == 1
    assert summary[0]["competitive_fn_count"] == 1
    assert summary[0]["weak_visual_fn_count"] == 0
    assert summary[0]["oracle_k_recovery_rate"] == 0.5
    assert summary[0]["teacher_forced_support_mean"] == 0.6
    assert summary[0]["proposal_support_mean"] == 0.65
    assert summary[0]["competitor_margin_mean"] == 0.125


def test_label_fn_rows_preserves_per_row_mechanism_labels_and_order() -> None:
    rows = [
        FnProbeRow(
            family_alias="base_xyxy_merged",
            teacher_forced_support=0.30,
            proposal_support=0.25,
            oracle_k_recovered=False,
            competitor_margin=0.05,
        ),
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

    labeled_rows = label_fn_rows(rows)

    assert [row["fn_row_index"] for row in labeled_rows] == [0, 1, 2]
    assert [row["fn_mechanism"] for row in labeled_rows] == [
        "weak_visual_fn",
        "suppressed_fn",
        "competitive_fn",
    ]
    assert labeled_rows[1]["family_alias"] == "cxcywh_pure_ce"


def test_load_recall_probe_config_accepts_artifact_sources_without_manual_rows(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "recall.yaml"
    config_path.write_text(
        """
run:
  name: coord-family-recall-smoke
  output_dir: output/analysis

artifact_sources:
  - family_alias: cxcywh_pure_ce
    oracle_fn_objects_jsonl: oracle.jsonl
    gt_proxy_scores_jsonl: gt.jsonl
    proposal_proxy_scores_jsonl: proposal.jsonl
        """.strip(),
        encoding="utf-8",
    )

    config = load_recall_probe_config(config_path)

    assert config.fn_rows == ()
    assert len(config.artifact_sources) == 1
    assert config.artifact_sources[0].family_alias == "cxcywh_pure_ce"


def test_build_fn_rows_from_artifact_sources_joins_oracle_gt_and_proposals(
    tmp_path: Path,
) -> None:
    oracle_path = tmp_path / "oracle.jsonl"
    gt_path = tmp_path / "gt.jsonl"
    proposal_path = tmp_path / "proposal.jsonl"
    oracle_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_idx": 7,
                        "gt_idx": 3,
                        "gt_desc": "teddy bear",
                        "baseline_fn_loc": True,
                        "ever_recovered_loc": True,
                    }
                ),
                json.dumps(
                    {
                        "record_idx": 7,
                        "gt_idx": 4,
                        "gt_desc": "teddy bear",
                        "baseline_fn_loc": False,
                        "ever_recovered_loc": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gt_path.write_text(
        json.dumps(
            {
                "source_image_idx": 7,
                "source_gt_idx": 3,
                "combined_linear": 0.7,
                "scoring_status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposal_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_idx": 7,
                        "nearest_gt_idx": 3,
                        "desc": "teddy bear",
                        "combined_linear": 0.8,
                        "is_unmatched": 1,
                        "collection_valid": True,
                        "scoring_status": "ok",
                    }
                ),
                json.dumps(
                    {
                        "image_idx": 7,
                        "nearest_gt_idx": 2,
                        "desc": "teddy bear",
                        "combined_linear": 1.1,
                        "is_unmatched": 1,
                        "collection_valid": True,
                        "scoring_status": "ok",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_fn_rows_from_artifact_sources(
        (
            ArtifactRecallSource(
                family_alias="cxcywh_pure_ce",
                oracle_fn_objects_jsonl=str(oracle_path),
                gt_proxy_scores_jsonl=str(gt_path),
                proposal_proxy_scores_jsonl=str(proposal_path),
            ),
        ),
        config_dir=tmp_path,
        repo_root=tmp_path,
    )

    assert len(rows) == 1
    assert rows[0].family_alias == "cxcywh_pure_ce"
    assert rows[0].teacher_forced_support == 0.7
    assert rows[0].proposal_support == 0.8
    assert rows[0].oracle_k_recovered is True
    assert rows[0].competitor_margin == pytest.approx(0.3)


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
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["run_name"] == "coord-family-recall-smoke"
    assert summary["family_metrics"][0]["family_alias"] == "cxcywh_pure_ce"
    assert summary["family_metrics"][0]["suppressed_fn_count"] == 1
    assert summary["family_metrics"][0]["competitive_fn_count"] == 1
    assert summary["family_mechanism_rows"] == [
        {
            "family_alias": "cxcywh_pure_ce",
            "fn_mechanism": "competitive_fn",
            "fn_count": 1,
            "fn_rate": 0.5,
            "teacher_forced_support_mean": 0.45,
            "proposal_support_mean": 0.5,
            "oracle_k_recovery_rate": 0.0,
            "competitor_margin_mean": 0.3,
        },
        {
            "family_alias": "cxcywh_pure_ce",
            "fn_mechanism": "suppressed_fn",
            "fn_count": 1,
            "fn_rate": 0.5,
            "teacher_forced_support_mean": 0.75,
            "proposal_support_mean": 0.8,
            "oracle_k_recovery_rate": 1.0,
            "competitor_margin_mean": -0.05,
        },
    ]
    assert summary["mechanism_summary"]["fn_count"] == 2
    assert summary["mechanism_summary"]["family_count"] == 1
    assert summary["mechanism_summary"]["suppressed_fn_count"] == 1
    assert summary["mechanism_summary"]["competitive_fn_count"] == 1
    assert summary["mechanism_summary"]["weak_visual_fn_count"] == 0
    assert summary["mechanism_summary"]["suppressed_fn_rate"] == 0.5
    assert summary["mechanism_summary"]["competitive_fn_rate"] == 0.5
    assert summary["mechanism_summary"]["weak_visual_fn_rate"] == 0.0
    assert summary["mechanism_summary"]["mechanism_counts"] == {
        "suppressed_fn": 1,
        "competitive_fn": 1,
        "weak_visual_fn": 0,
    }
    assert summary["mechanism_summary"]["mechanism_rates"] == {
        "suppressed_fn": 0.5,
        "competitive_fn": 0.5,
        "weak_visual_fn": 0.0,
    }
    assert summary["artifacts"] == {
        "fn_rows_jsonl": str(rows_path),
        "artifact_sources": [],
    }
    assert rows[0]["fn_row_index"] == 0
    assert rows[0]["fn_mechanism"] == "suppressed_fn"
    assert rows[1]["fn_row_index"] == 1
    assert rows[1]["fn_mechanism"] == "competitive_fn"


def test_run_recall_probe_materializes_rows_from_artifact_sources(tmp_path: Path) -> None:
    oracle_path = tmp_path / "oracle.jsonl"
    gt_path = tmp_path / "gt.jsonl"
    proposal_path = tmp_path / "proposal.jsonl"
    output_dir = tmp_path / "analysis"
    oracle_path.write_text(
        json.dumps(
            {
                "record_idx": 9,
                "gt_idx": 1,
                "gt_desc": "person",
                "baseline_fn_loc": True,
                "ever_recovered_loc": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gt_path.write_text(
        json.dumps(
            {
                "source_image_idx": 9,
                "source_gt_idx": 1,
                "combined_linear": 0.2,
                "scoring_status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposal_path.write_text(
        json.dumps(
            {
                "image_idx": 9,
                "nearest_gt_idx": 1,
                "desc": "person",
                "combined_linear": 0.1,
                "is_unmatched": 1,
                "collection_valid": True,
                "scoring_status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "recall_from_artifacts.yaml"
    config_path.write_text(
        f"""
run:
  name: coord-family-recall-artifact-smoke
  output_dir: {output_dir.as_posix()}

artifact_sources:
  - family_alias: hard_soft_ce_2b
    oracle_fn_objects_jsonl: {oracle_path.as_posix()}
    gt_proxy_scores_jsonl: {gt_path.as_posix()}
    proposal_proxy_scores_jsonl: {proposal_path.as_posix()}
        """.strip(),
        encoding="utf-8",
    )

    result = run_recall_probe(config_path)

    rows_path = output_dir / "coord-family-recall-artifact-smoke" / "fn_rows.jsonl"
    summary_path = output_dir / "coord-family-recall-artifact-smoke" / "summary.json"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result["labeled_fn_row_count"] == 1
    assert rows[0]["family_alias"] == "hard_soft_ce_2b"
    assert rows[0]["teacher_forced_support"] == 0.2
    assert rows[0]["proposal_support"] == 0.1
    assert rows[0]["oracle_k_recovered"] is False
    assert summary["family_metrics"][0]["family_alias"] == "hard_soft_ce_2b"
    assert summary["artifacts"]["artifact_sources"][0]["family_alias"] == "hard_soft_ce_2b"


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
