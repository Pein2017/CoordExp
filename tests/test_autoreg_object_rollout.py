from __future__ import annotations

import json
from pathlib import Path

from src.analysis.autoreg_object_rollout import run_phase1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture_artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_root = tmp_path / "artifact"
    dataset_jsonl = tmp_path / "val.coord.jsonl"
    output_dir = tmp_path / "analysis"

    dataset_rows = [
        {
            "image_id": 139,
            "image": "images/val2017/000000000139.jpg",
            "images": ["images/val2017/000000000139.jpg"],
            "width": 640,
            "height": 480,
            "objects": [
                {"desc": "cat", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]},
                {"desc": "dog", "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"]},
            ],
        },
        {
            "image_id": 140,
            "image": "images/val2017/000000000140.jpg",
            "images": ["images/val2017/000000000140.jpg"],
            "width": 640,
            "height": 480,
            "objects": [
                {"desc": "bus", "bbox_2d": ["<|coord_9|>", "<|coord_10|>", "<|coord_11|>", "<|coord_12|>"]},
            ],
        },
    ]
    _write_jsonl(dataset_jsonl, dataset_rows)

    base_rows = [
        {
            "image": "images/val2017/000000000139.jpg",
            "image_id": 139,
            "width": 640,
            "height": 480,
            "gt": dataset_rows[0]["objects"],
            "pred": [
                {"desc": "cat", "points": [1, 2, 3, 4], "type": "bbox_2d"},
                {"desc": "cat", "points": [1, 2, 3, 4], "type": "bbox_2d"},
            ],
            "raw_output_json": {"objects": [{"desc": "cat"}, {"desc": "cat"}]},
            "raw_ends_with_im_end": False,
            "errors": [],
            "error_entries": [],
        },
        {
            "image": "images/val2017/000000000140.jpg",
            "image_id": 140,
            "width": 640,
            "height": 480,
            "gt": dataset_rows[1]["objects"],
            "pred": [],
            "raw_output_json": {"objects": []},
            "raw_ends_with_im_end": True,
            "errors": [],
            "error_entries": [],
        },
    ]
    for rel in [
        "gt_vs_pred.jsonl",
        "gt_vs_pred_scored.jsonl",
    ]:
        _write_jsonl(artifact_root / rel, base_rows)
    guarded_rows = [dict(base_rows[0], pred=[base_rows[0]["pred"][0]]), base_rows[1]]
    _write_jsonl(artifact_root / "gt_vs_pred_scored_guarded.jsonl", guarded_rows)
    _write_jsonl(artifact_root / "pred_confidence.jsonl", [{"line_idx": 0, "objects": []}, {"line_idx": 1, "objects": []}])
    _write_jsonl(
        artifact_root / "pred_token_trace.jsonl",
        [
            {
                "line_idx": 0,
                "generated_token_text": [
                    "<|object_ref_start|>",
                    "cat",
                    "<|box_start|>",
                    "<|coord_1|>",
                    "<|im_end|>",
                    "<|endoftext|>",
                    "<|endoftext|>",
                ],
                "token_logprobs": [-0.1, -0.2, -0.01, -0.3, -0.4, -0.5, -0.6],
            },
            {
                "line_idx": 1,
                "generated_token_text": ["<|im_end|>"],
                "token_logprobs": [-0.05],
            },
        ],
    )
    _write_jsonl(
        artifact_root / "eval/matches.jsonl",
        [
            {
                "image_id": 0,
                "file_name": "images/val2017/000000000139.jpg",
                "gt_count": 2,
                "pred_count": 2,
                "matches": [{"pred_idx": 0, "gt_idx": 0, "iou": 0.9}],
                "unmatched_pred_indices": [1],
                "ignored_pred_indices": [],
                "unmatched_gt_indices": [1],
            },
            {
                "image_id": 1,
                "file_name": "images/val2017/000000000140.jpg",
                "gt_count": 1,
                "pred_count": 0,
                "matches": [],
                "unmatched_pred_indices": [],
                "ignored_pred_indices": [],
                "unmatched_gt_indices": [0],
            },
        ],
    )
    _write_jsonl(
        artifact_root / "eval/matches_guarded.jsonl",
        [
            {
                "image_id": 0,
                "file_name": "images/val2017/000000000139.jpg",
                "gt_count": 2,
                "pred_count": 1,
                "matches": [{"pred_idx": 0, "gt_idx": 0, "iou": 0.9}],
                "unmatched_pred_indices": [],
                "ignored_pred_indices": [],
                "unmatched_gt_indices": [1],
            },
            {
                "image_id": 1,
                "file_name": "images/val2017/000000000140.jpg",
                "gt_count": 1,
                "pred_count": 0,
                "matches": [],
                "unmatched_pred_indices": [],
                "ignored_pred_indices": [],
                "unmatched_gt_indices": [0],
            },
        ],
    )
    _write_json(
        artifact_root / "eval/duplicate_guard_report.json",
        {
            "total_records": 2,
            "total_predictions_inspected": 2,
            "total_predictions_suppressed": 1,
            "records": [
                {
                    "record_index": 0,
                    "image_id": 139,
                    "image": "images/val2017/000000000139.jpg",
                    "inspected_predictions": 2,
                    "suppressed_predictions": 1,
                    "kept_indices": [0],
                    "suppressed_indices": [1],
                },
                {
                    "record_index": 1,
                    "image_id": 140,
                    "image": "images/val2017/000000000140.jpg",
                    "inspected_predictions": 0,
                    "suppressed_predictions": 0,
                    "kept_indices": [],
                    "suppressed_indices": [],
                },
            ],
        },
    )
    _write_json(artifact_root / "eval/metrics.json", {"f1ish@0.50_recall_loc_micro": 0.5})
    _write_json(artifact_root / "eval/metrics_guarded.json", {"f1ish@0.50_recall_loc_micro": 0.5})
    _write_json(
        artifact_root / "summary.json",
        {
            "total_read": 2,
            "total_emitted": 2,
            "errors_total": 0,
            "generation": {
                "temperature": 0.0,
                "repetition_penalty": 1.1,
                "max_new_tokens": 1024,
                "compact_grammar": {"enabled": True},
            },
            "infer": {
                "prompt_template_hash": "hash",
                "object_ordering": "random",
                "limit": 2,
            },
        },
    )
    _write_json(
        artifact_root / "resolved_config.json",
        {
            "config_path": "unit.yaml",
            "artifacts": {"run_dir": str(artifact_root)},
            "infer": {
                "prompt_template_hash": "hash",
                "object_ordering": "random",
                "resolved_base_model_checkpoint": "/ckpt",
            },
        },
    )
    return artifact_root, dataset_jsonl, output_dir


def test_phase1_contract_mapping_and_outputs(tmp_path: Path) -> None:
    artifact_root, dataset_jsonl, output_dir = _fixture_artifact(tmp_path)
    result = run_phase1(
        {
            "run": {
                "output_dir": str(output_dir),
                "scope_label": "val200_first_200",
                "checkpoint": "/ckpt",
            },
            "inputs": {
                "artifact_root": str(artifact_root),
                "dataset_jsonl": str(dataset_jsonl),
                "dataset_slice": {"kind": "first_n", "n": 2},
            },
            "analysis": {
                "truncate_at_first_im_end": True,
                "require_scored_provenance": False,
                "metric_family": "guarded",
            },
        }
    )

    assert result["gate_report"]["gate0"]["status"] == "ok"
    assert result["gate_report"]["gate0b"]["status"] == "warning"
    assert result["summary"]["counts"]["images"] == 2
    assert result["summary"]["counts"]["gt_objects"] == 3
    assert result["summary"]["counts"]["raw_predictions"] == 2
    assert result["summary"]["counts"]["guarded_predictions"] == 1
    assert result["summary"]["token_summary"]["used_token_count"] == 6

    per_row_lines = (output_dir / "rollout_anatomy/per_row.jsonl").read_text().splitlines()
    per_rows = [json.loads(line) for line in per_row_lines]
    assert per_rows[0]["source_line_idx"] == 0
    assert per_rows[0]["coco_image_id"] == 139
    assert per_rows[0]["raw_pred_idx"] == 0
    assert per_rows[0]["guarded_pred_idx"] == 0
    assert per_rows[0]["raw_match_label"] == "tp_like"
    assert per_rows[0]["guarded_match_label"] == "tp_like"
    assert per_rows[1]["raw_pred_idx"] == 1
    assert per_rows[1]["guarded_pred_idx"] is None
    assert per_rows[1]["suppressed_by_guard"] is True
    assert per_rows[1]["row_label"] == "duplicate_suppressed"

    per_image = [
        json.loads(line)
        for line in (output_dir / "rollout_anatomy/per_image.jsonl").read_text().splitlines()
    ]
    assert per_image[0]["first_im_end_index"] == 4
    assert per_image[0]["tokens_after_first_im_end"] == 2
    assert per_image[0]["endoftext_after_im_end_count"] == 2
    assert per_image[0]["used_token_count"] == 5
    assert per_image[0]["raw_ends_with_im_end"] is False

    report = (output_dir / "rollout_anatomy/report.md").read_text()
    assert "production_training_recommendation: none" in report
