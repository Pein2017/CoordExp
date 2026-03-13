from __future__ import annotations

import json
from pathlib import Path

from src.analysis.unmatched_proposal_verifier import (
    _binary_metrics,
    _find_subsequence,
    _render_report,
    build_gt_and_negative_tables,
    build_rollout_proposal_table,
    resolve_checkpoint_path,
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

    rows = build_rollout_proposal_table(checkpoint_name="ckpt", run_dir=run_dir)
    by_index = {int(row["proposal_index"]): row for row in rows}
    assert by_index[0]["match_status"] == "matched"
    assert by_index[1]["match_status"] == "unmatched"
    assert float(by_index[0]["nearest_gt_iou"]) > 0.9


def test_resolve_checkpoint_path_labels_common_root() -> None:
    path, source = resolve_checkpoint_path(
        "output/stage2_ab/prod/ul-res_1024-ckpt_300_merged"
    )
    assert path.exists()
    assert source == "config_path_common_root"


def test_render_report_includes_recommendation_section() -> None:
    report = _render_report(
        config=type(
            "Cfg",
            (),
            {
                "collection": type(
                    "Collection",
                    (),
                    {
                        "backend_mode": "hf",
                        "temperature": 0.1,
                        "repetition_penalty": 1.1,
                        "batch_size": 1,
                        "gpu_memory_utilization": 0.9,
                    },
                )()
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
                "gt_vs_hard_negative": {
                    "commitment": {"auroc": 0.6, "auprc": 0.5, "count": 10, "excluded_by_reason": {}},
                    "counterfactual": {"auroc": 0.8, "auprc": 0.7, "count": 10, "excluded_by_reason": {}},
                    "combined_linear": {"auroc": 0.75, "auprc": 0.69, "count": 10, "excluded_by_reason": {}},
                },
                "matched_vs_unmatched": {
                    "commitment": {"auroc": 0.55, "auprc": 0.6, "count": 10, "excluded_by_reason": {}},
                    "counterfactual": {"auroc": 0.65, "auprc": 0.7, "count": 10, "excluded_by_reason": {}},
                    "combined_linear": {"auroc": 0.7, "auprc": 0.72, "count": 10, "excluded_by_reason": {}},
                },
                "commitment_counterfactual_correlation": -0.2,
                "calibration_skipped_reason": "raw",
                "audit_pack": {"count": 3, "index_jsonl": "audit.jsonl"},
            }
        ],
    )
    assert "## Recommendation" in report
    assert "strongest single proxy" in report
