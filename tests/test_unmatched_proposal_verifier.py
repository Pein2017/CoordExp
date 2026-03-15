from __future__ import annotations

import json
from pathlib import Path

from src.analysis.unmatched_proposal_verifier import (
    CollectionGateConfig,
    _binary_metrics,
    _find_subsequence,
    _pseudo_label_ready,
    _render_report,
    build_gt_and_negative_tables,
    build_rollout_proposal_table,
    load_study_config,
    resolve_checkpoint_path,
    summarize_checkpoint,
    summarize_collection_health,
    summarize_manual_audit,
)


def test_find_subsequence_smoke() -> None:
    haystack = [7, 8, 1, 2, 3, 9]
    assert _find_subsequence(haystack, [1, 2, 3], start_hint=0) == 2
    assert _find_subsequence(haystack, [2, 3], start_hint=3) == 3


def test_build_gt_and_negative_tables_emits_expected_families() -> None:
    subset_records = [
        {
            "images": ["fake/a.jpg"],
            "width": 1000,
            "height": 1000,
            "objects": [
                {"desc": "cup", "bbox_2d": [100, 100, 220, 220]},
                {"desc": "cup", "bbox_2d": [500, 500, 620, 620]},
            ],
        }
    ]

    positives, negatives = build_gt_and_negative_tables(subset_records)

    assert len(positives) == 2
    families = {str(row["row_family"]) for row in negatives}
    assert "same_desc_wrong_location_jitter" in families
    assert "desc_box_cross_swap" in families
    assert "same_class_wrong_location" in families


def test_binary_metrics_smoke() -> None:
    rows = [
        {"label": 1, "commitment": 0.9, "scoring_status": "ok"},
        {"label": 1, "commitment": 0.8, "scoring_status": "ok"},
        {"label": 0, "commitment": 0.2, "scoring_status": "ok"},
        {"label": 0, "commitment": 0.1, "scoring_status": "ok"},
    ]
    summary = _binary_metrics(rows, label_key="label", score_key="commitment")
    assert summary["count"] == 4
    assert float(summary["auroc"]) > 0.99
    assert float(summary["auprc"]) > 0.99


def test_build_rollout_proposal_table_tags_match_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)

    gt_vs_pred_row = {
        "image": "fake/a.jpg",
        "width": 1000,
        "height": 1000,
        "gt": [
            {"desc": "cup", "bbox": [100, 100, 200, 200]},
            {"desc": "book", "bbox": [500, 500, 700, 700]},
        ],
        "pred": [
            {"desc": "cup", "bbox": [102, 102, 198, 198]},
            {"desc": "plate", "bbox": [300, 300, 420, 420]},
        ],
    }
    matches_row = {
        "image_id": 0,
        "matches": [{"pred_idx": 0, "gt_idx": 0, "iou": 0.9}],
        "unmatched_pred_indices": [1],
        "ignored_pred_indices": [],
    }
    (run_dir / "gt_vs_pred.jsonl").write_text(
        json.dumps(gt_vs_pred_row) + "\n", encoding="utf-8"
    )
    (eval_dir / "matches.jsonl").write_text(
        json.dumps(matches_row) + "\n", encoding="utf-8"
    )

    rows = build_rollout_proposal_table(
        checkpoint_name="ckpt",
        run_dir=run_dir,
        temperature=0.3,
    )
    by_index = {int(row["proposal_index"]): row for row in rows}
    assert by_index[0]["match_status"] == "matched"
    assert by_index[1]["match_status"] == "unmatched"
    assert by_index[0]["proposal_uid"].startswith("ckpt__t0p3__")
    assert float(by_index[0]["nearest_gt_iou"]) > 0.9


def test_summarize_collection_health_marks_invalid_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    gt_rows = [
        {
            "image": "fake/a.jpg",
            "width": 100,
            "height": 100,
            "pred": [],
            "gt": [],
            "errors": ["invalid_json"],
            "raw_output_json": None,
        },
        {
            "image": "fake/b.jpg",
            "width": 100,
            "height": 100,
            "pred": [{"desc": "cup", "bbox": [1, 1, 10, 10]}],
            "gt": [],
            "errors": [],
            "raw_output_json": {"objects": []},
        },
    ]
    (run_dir / "gt_vs_pred.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in gt_rows), encoding="utf-8"
    )
    proposal_rows = [
        {
            "match_status": "matched",
            "is_matched": 1,
            "is_unmatched": 0,
            "is_ignored": 0,
            "duplicate_like_any_desc_iou90": 0,
        }
    ]
    summary = summarize_collection_health(
        checkpoint_name="ckpt",
        temperature=0.3,
        run_dir=run_dir,
        proposal_rows=proposal_rows,
        gate=CollectionGateConfig(
            nonempty_pred_image_rate_min=0.9,
            pred_count_total_min=5,
            unmatched_count_min=2,
        ),
    )
    assert summary["collection_valid"] is False
    assert "raw_output_missing" in summary["parser_failure_counts"]
    assert summary["collection_invalid_reason"] is not None


def test_load_study_config_reads_authority_first_surfaces(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
run:
  name: test
  output_dir: output/analysis
  stages: [prepare, gate, report]

dataset:
  jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  sample_count: 8

collection:
  backend_mode: stage2_parity_vllm
  temperature: 0.5
  max_num_seqs: 32
  enforce_eager: true

collection_gate:
  nonempty_pred_image_rate_min: 0.4
  pred_count_total_min: 12
  unmatched_count_min: 3

report:
  authoritative_temperatures: [0.0, 0.3, 0.5, 0.7]
  audit_labels_path: temp/manual_audit_labels.jsonl

manual_audit:
  sample_count: 64
  score_key: counterfactual

checkpoints:
  - name: ckpt
    path: output/stage2_ab/prod/ul-res_1024-ckpt_300_merged
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_study_config(config_path)
    assert config.run.stages == ("prepare", "gate", "report")
    assert config.collection.backend_mode == "stage2_parity_vllm"
    assert config.collection.max_num_seqs == 32
    assert config.collection.enforce_eager is True
    assert config.collection_gate.pred_count_total_min == 12
    assert config.report.authoritative_temperatures == (0.0, 0.3, 0.5, 0.7)
    assert config.manual_audit.label_path == "temp/manual_audit_labels.jsonl"
    assert config.manual_audit.score_key == "counterfactual"


def test_resolve_checkpoint_path_labels_common_root() -> None:
    path, source = resolve_checkpoint_path(
        "output/stage2_ab/prod/ul-res_1024-ckpt_300_merged"
    )
    assert path.exists()
    assert source in {"config_path_common_root", "config_path_worktree"}


def test_render_report_includes_authority_first_downgrade() -> None:
    report = _render_report(
        config=type(
            "Cfg",
            (),
            {
                "run": type("Run", (), {"stages": ("prepare", "collection", "scoring", "audit", "report")})(),
                "collection": type(
                    "Collection",
                    (),
                    {
                        "backend_mode": "hf",
                        "temperature": 0.3,
                        "repetition_penalty": 1.1,
                        "batch_size": 1,
                        "gpu_memory_utilization": 0.9,
                    },
                )(),
                "report": type(
                    "Report",
                    (),
                    {"authoritative_temperatures": (0.0, 0.3, 0.5, 0.7)},
                )(),
            },
        )(),
        subset_meta={
            "output_path": "subset.jsonl",
            "input_path": "input.jsonl",
            "num_samples": 8,
            "seed": 42,
            "root_image_dir": "images",
        },
        manifests=[
            {
                "checkpoint_name": "ckpt",
                "checkpoint_path_resolved": "/abs/ckpt",
                "prompt_variant": "coco_80",
                "object_field_order": "desc_first",
                "prompt_control_source": "study_default",
            }
        ],
        summaries=[
            {
                "checkpoint": "ckpt",
                "temperature": 0.3,
                "collection_health": {
                    "collection_valid": False,
                    "pred_count_total": 12,
                    "unmatched_count": 1,
                    "nonempty_pred_image_rate": 0.2,
                    "collection_invalid_reason": "low_unmatched_count",
                },
                "gt_vs_hard_negative": {
                    "commitment": {"auroc": 0.6, "auprc": 0.5, "count": 10, "excluded_by_reason": {}},
                    "counterfactual": {"auroc": 0.8, "auprc": 0.7, "count": 10, "excluded_by_reason": {}},
                    "combined_linear": {"auroc": 0.75, "auprc": 0.69, "count": 10, "excluded_by_reason": {}},
                },
                "matched_vs_unmatched": {
                    "commitment": {"auroc": None, "auprc": None, "count": 0, "excluded_by_reason": {}},
                    "counterfactual": {"auroc": None, "auprc": None, "count": 0, "excluded_by_reason": {}},
                    "combined_linear": {"auroc": None, "auprc": None, "count": 0, "excluded_by_reason": {}},
                },
                "commitment_counterfactual_correlation": None,
                "calibration_skipped_reason": "raw",
                "audit_pack": {"count": 3, "index_jsonl": "audit.jsonl"},
            }
        ],
        manual_audit_summary={
            "labels_loaded": False,
            "labeled_count": 0,
            "label_counts": {},
            "precision_at_k": {},
        },
        aggregate_paths={
            "gt_clean_summary_csv": "gt.csv",
            "collection_health_csv": "health.csv",
            "rollout_summary_csv": "rollout.csv",
            "manual_audit_summary_json": "audit.json",
        },
    )
    assert "Layer B1: Rollout Collection Health" in report
    assert "Layer C: Manual Audit" in report
    assert "promising but not yet promotion-ready" in report


def test_summarize_manual_audit_reports_precision() -> None:
    summary = summarize_manual_audit(
        [
            {"audit_label": "real_visible_object", "combined_linear": 1.0},
            {"audit_label": "wrong_location", "combined_linear": 0.8},
            {"audit_label": "real_visible_object", "combined_linear": 0.7},
            {"audit_label": "uncertain", "combined_linear": 0.6},
        ],
        score_key="combined_linear",
        top_k_values=(2, 4),
    )
    assert int(summary["labeled_count"]) == 4
    assert float((summary["precision_at_k"]["2"] or {})["real_visible_object_rate"]) == 0.5
    assert int((summary["label_counts"] or {})["real_visible_object"]) == 2


def test_summarize_checkpoint_excludes_invalid_rollout_metrics() -> None:
    summary = summarize_checkpoint(
        checkpoint_name="ckpt",
        temperature=0.3,
        collection_health={
            "collection_valid": False,
            "collection_invalid_reason": "low_unmatched_count",
        },
        gt_rows=[
            {
                "label": 1,
                "commitment": 0.9,
                "counterfactual": 0.2,
                "combined_linear": 1.1,
                "scoring_status": "ok",
            },
            {
                "label": 0,
                "commitment": 0.1,
                "counterfactual": -0.1,
                "combined_linear": 0.0,
                "scoring_status": "ok",
            },
        ],
        proposal_rows=[
            {
                "match_status": "matched",
                "scoring_status": "skipped",
                "failure_reason": "collection_invalid:low_unmatched_count",
            }
        ],
        manual_audit_summary=None,
        histogram_bins=8,
        top_k_values=(5,),
    )
    rollout_metrics = summary["matched_vs_unmatched"]["commitment"]
    assert rollout_metrics["count"] == 0
    assert rollout_metrics["rollout_interpretation_valid"] is False
    assert "collection_invalid:low_unmatched_count" in rollout_metrics["excluded_by_reason"]


def test_pseudo_label_ready_requires_manual_audit_support() -> None:
    summaries = [
        {
            "checkpoint": "ckpt",
            "collection_health": {"collection_valid": True},
            "gt_vs_hard_negative": {
                "counterfactual": {"auroc": 0.7, "auprc": 0.6},
                "commitment": {"auroc": 0.5, "auprc": 0.4},
                "combined_linear": {"auroc": 0.65, "auprc": 0.55},
            },
            "matched_vs_unmatched": {
                "counterfactual": {"count": 20, "auroc": 0.7, "auprc": 0.6},
                "commitment": {"count": 20, "auroc": 0.5, "auprc": 0.4},
                "combined_linear": {"count": 20, "auroc": 0.65, "auprc": 0.55},
            },
        }
    ]
    assert (
        _pseudo_label_ready(
            summaries,
            audit_summary={
                "labels_loaded": False,
                "labeled_count": 0,
                "precision_at_k": {},
            },
        )
        is False
    )
