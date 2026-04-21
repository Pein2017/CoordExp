from __future__ import annotations

import json
from pathlib import Path

from src.analysis.coord_family_recall_progress import build_recall_progress


def test_build_recall_progress_merges_verifier_oracle_and_probe_summaries(tmp_path: Path) -> None:
    verifier_path = tmp_path / "verifier.json"
    oracle_path = tmp_path / "oracle.json"
    probe_path = tmp_path / "probe.json"
    output_dir = tmp_path / "analysis"

    verifier_path.write_text(
        json.dumps(
            {
                "collection_health": {
                    "pred_count_total": 12,
                    "matched_count": 7,
                    "unmatched_count": 5,
                    "duplicate_like_rate": 0.25,
                    "invalid_rollout_count": 1,
                    "collection_valid": True,
                    "parser_failure_counts": {"invalid_geometry": 2},
                },
                "gt_vs_hard_negative": {"commitment": 0.5},
                "matched_vs_unmatched": {"combined_linear": 0.7},
                "commitment_counterfactual_correlation": -0.2,
            }
        ),
        encoding="utf-8",
    )
    oracle_path.write_text(
        json.dumps(
            {
                "iou_thresholds": {
                    "0.50": {
                        "baseline": {"recall_loc": 0.4},
                        "oracle_k": {"recall_loc": 0.6},
                    }
                },
                "primary_recovery": {
                    "baseline_fn_count_loc": 30,
                    "recoverable_fn_count_loc": 12,
                    "systematic_fn_count_loc": 18,
                    "recover_fraction_loc": 0.19,
                },
            }
        ),
        encoding="utf-8",
    )
    probe_path.write_text(
        json.dumps(
            {
                "family_metrics": [
                    {
                        "family_alias": "raw_text_xyxy_pure_ce",
                        "suppressed_fn_rate": 0.2,
                        "competitive_fn_rate": 0.3,
                        "weak_visual_fn_rate": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "progress.yaml"
    config_path.write_text(
        f"""
run:
  name: coord-family-recall-progress-smoke
  output_dir: {output_dir.as_posix()}

families:
  - family_alias: raw_text_xyxy_pure_ce
    sample_size: 64
    verifier_summary_json: {verifier_path.as_posix()}
    oracle_summary_json: {oracle_path.as_posix()}
    recall_probe_summary_json: {probe_path.as_posix()}
        """.strip(),
        encoding="utf-8",
    )

    result = build_recall_progress(config_path, repo_root=tmp_path)

    summary_path = output_dir / "coord-family-recall-progress-smoke" / "summary.json"
    report_path = output_dir / "coord-family-recall-progress-smoke" / "report.md"
    assert result["summary_json"] == str(summary_path)
    assert report_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    family = summary["families"]["raw_text_xyxy_pure_ce"]
    assert family["status"] == "oracle_and_verifier_complete"
    assert family["baseline_recall_loc"] == 0.4
    assert family["oracle_k_recall_loc"] == 0.6
    assert family["recoverable_fraction_of_baseline_fn_loc"] == 0.19
    assert family["verifier"]["pred_count_total"] == 12
    assert family["recall_probe"]["competitive_fn_rate"] == 0.3


def test_build_recall_progress_marks_pending_when_oracle_and_probe_are_missing(
    tmp_path: Path,
) -> None:
    verifier_path = tmp_path / "verifier.json"
    output_dir = tmp_path / "analysis"
    verifier_path.write_text(
        json.dumps(
            {
                "collection_health": {
                    "pred_count_total": 8,
                    "matched_count": 2,
                    "unmatched_count": 6,
                    "duplicate_like_rate": 0.0,
                    "invalid_rollout_count": 0,
                    "collection_valid": True,
                },
                "gt_vs_hard_negative": {"commitment": 0.51},
                "matched_vs_unmatched": {"combined_linear": 0.84},
                "commitment_counterfactual_correlation": 0.08,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "progress_pending.yaml"
    config_path.write_text(
        f"""
run:
  name: coord-family-recall-progress-pending
  output_dir: {output_dir.as_posix()}

families:
  - family_alias: hard_soft_ce_2b
    sample_size: 64
    verifier_summary_json: {verifier_path.as_posix()}
    oracle_summary_json: oracle_missing.json
    recall_probe_summary_json: probe_missing.json
        """.strip(),
        encoding="utf-8",
    )

    build_recall_progress(config_path, repo_root=tmp_path)

    summary_path = output_dir / "coord-family-recall-progress-pending" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    family = summary["families"]["hard_soft_ce_2b"]
    assert family["status"] == "verifier_complete_oracle_pending"
    assert "baseline_recall_loc" not in family
    assert family["verifier"]["unmatched_count"] == 6
