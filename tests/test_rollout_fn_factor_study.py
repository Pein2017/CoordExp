from __future__ import annotations

import json
from pathlib import Path

from src.analysis.rollout_fn_factor_study import (
    _aggregate_sample_health,
    _bootstrap_image_health_reason,
    _canonicalize_eval_objects,
    _build_recovery_table,
    _build_rollout_health,
    _close_prefix_rollout_text,
    _continuation_only_rows,
    _load_stage_manifest_if_present,
    _serialize_objects_to_prefix_text,
    _worker_assignments,
    load_study_config,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_study_config_reads_canonical_surfaces(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
run:
  name: test-study
  output_dir: output/analysis
  stages: [bootstrap, baseline, report]

prompts:
  prompt_variant: coco_80
  object_field_order: desc_first
  do_resize: false

bootstrap:
  candidate_pool_limit: 64
  hard32_size: 32
  hard16_size: 16

baseline_decode:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 1024
  repetition_penalty: 1.05
  batch_size: 2

sampling:
  k: 4
  temperature: 0.6
  top_p: 0.95
  max_new_tokens: 1024
  repetition_penalty: 1.0
  batch_size: 1
  seed: 7

prefix:
  lengths: [1, 2, 4]
  random_seed: 19

stress:
  prefix_length: 4
  mutations: [delete, adjacent_swap]

length:
  extended_max_new_tokens: 1536

eval:
  semantic_device: cuda:0

health_gate:
  min_parse_valid_rate: 1.0
  min_nonempty_rate: 1.0
  min_pred_count_total: 1
  max_duplicate_like_rate: 0.3

execution:
  gpu_ids: [0, 1, 2, 3]

checkpoints:
  - alias: original
    path: output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332
  - alias: a_only
    path: output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900

splits:
  train:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
  val:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_study_config(config_path)
    assert config.run.name == "test-study"
    assert config.bootstrap.candidate_pool_limit == 64
    assert config.prefix.lengths == (1, 2, 4)
    assert config.stress.mutations == ("delete", "adjacent_swap")
    assert tuple(split.name for split in config.splits) == ("train", "val")


def test_continuation_only_rows_excludes_prefix_predictions() -> None:
    rows = [
        {
            "index": 0,
            "prefix_pred_count": 1,
            "pred": [
                {"index": 0, "desc": "person", "bbox": [0, 0, 10, 10]},
                {"index": 1, "desc": "dog", "bbox": [10, 10, 20, 20]},
            ],
            "matching": {"pairs": []},
        }
    ]
    continuation = _continuation_only_rows(rows)
    assert len(continuation) == 1
    assert continuation[0]["pred"] == [{"index": 0, "desc": "dog", "bbox": [10, 10, 20, 20]}]
    assert continuation[0]["prefix_pred_count"] == 0
    assert continuation[0]["continuation_pred_start_index"] == 0
    assert "matching" not in continuation[0]


def test_serialize_objects_to_prefix_text_stays_append_ready() -> None:
    prefix_text, compact = _serialize_objects_to_prefix_text(
        [
            {
                "desc": "person",
                "bbox_2d": [
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_3|>",
                    "<|coord_4|>",
                ],
            }
        ],
        width=100,
        height=200,
        object_field_order="desc_first",
    )
    assert prefix_text == (
        '{"objects": [{"desc": "person", "bbox_2d": [<|coord_1|>, <|coord_2|>, '
        '<|coord_3|>, <|coord_4|>]}, '
    )
    assert compact == [
        {
            "type": "bbox_2d",
            "points": [0.1, 0.4, 0.3, 0.4, 0.3, 0.8, 0.1, 0.8],
            "desc": "person",
        }
    ]


def test_serialize_objects_to_prefix_text_converts_pixel_bbox_rows_from_baseline() -> None:
    prefix_text, compact = _serialize_objects_to_prefix_text(
        [
            {
                "desc": "wine glass",
                "bbox_2d": [1000, 100, 1100, 300],
                "coord_mode": "pixel",
            }
        ],
        width=2000,
        height=1000,
        object_field_order="desc_first",
    )
    assert prefix_text == (
        '{"objects": [{"desc": "wine glass", "bbox_2d": [<|coord_500|>, <|coord_100|>, '
        '<|coord_550|>, <|coord_300|>]}, '
    )
    assert compact == [
        {
            "type": "bbox_2d",
            "points": [1000.0, 100.0, 1100.0, 100.0, 1100.0, 300.0, 1000.0, 300.0],
            "desc": "wine glass",
        }
    ]


def test_close_prefix_rollout_text_repairs_open_container_with_optional_im_end() -> None:
    prefix_text = '{"objects": [{"desc": "person", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}, '
    assert (
        _close_prefix_rollout_text(
            prefix_text,
            "",
            object_field_order="desc_first",
        )
        == '{"objects": [{"desc": "person", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}'
    )
    assert (
        _close_prefix_rollout_text(
            prefix_text,
            "<|im_end|>",
            object_field_order="desc_first",
        )
        == '{"objects": [{"desc": "person", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}<|im_end|>'
    )
    assert (
        _close_prefix_rollout_text(
            prefix_text,
            ', <|im_end|>',
            object_field_order="desc_first",
        )
        == '{"objects": [{"desc": "person", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}]}<|im_end|>'
    )


def test_canonicalize_eval_objects_converts_point_boxes_to_bbox_xyxy() -> None:
    canonical = _canonicalize_eval_objects(
        [
            {
                "type": "bbox_2d",
                "points": [10, 20, 40, 20, 40, 60, 10, 60],
                "desc": "person",
            }
        ]
    )
    assert canonical == [
        {
            "index": 0,
            "desc": "person",
            "bbox_2d": [10, 20, 40, 60],
            "coord_mode": "pixel",
        }
    ]


def test_bootstrap_image_health_reason_accepts_parsed_predictions_without_raw_json() -> None:
    assert (
        _bootstrap_image_health_reason(
            {"pred": [{"desc": "person", "bbox_2d": [0, 0, 10, 10]}], "errors": [], "raw_output_json": None}
        )
        is None
    )
    assert _bootstrap_image_health_reason({"pred": [], "errors": [], "raw_output_json": None}) == "invalid_rollout"


def test_aggregate_sample_health_uses_invalid_reason_precedence() -> None:
    health = _aggregate_sample_health(
        [
            {
                "rollout_health": {
                    "num_images": 4,
                    "parse_valid_rate": 1.0,
                    "invalid_rollout_count": 1,
                    "nonempty_pred_rate": 1.0,
                    "pred_count_total": 10,
                    "duplicate_like_rate": 0.0,
                    "truncation_anomaly_count": 0,
                    "rollout_health_valid": False,
                    "rollout_health_invalid_reason": "invalid_rollout",
                }
            },
            {
                "rollout_health": {
                    "num_images": 4,
                    "parse_valid_rate": 0.5,
                    "invalid_rollout_count": 2,
                    "nonempty_pred_rate": 0.5,
                    "pred_count_total": 4,
                    "duplicate_like_rate": 0.1,
                    "truncation_anomaly_count": 1,
                    "rollout_health_valid": False,
                    "rollout_health_invalid_reason": "parse_invalid",
                }
            },
        ]
    )
    assert health["num_samples"] == 2
    assert health["num_images"] == 8
    assert health["rollout_health_valid"] is False
    assert health["rollout_health_invalid_reason"] == "parse_invalid"


def test_rollout_health_allows_nonempty_prediction_without_raw_json(tmp_path: Path) -> None:
    gt_vs_pred = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        gt_vs_pred,
        [
            {
                "index": 0,
                "pred": [{"desc": "person", "bbox_2d": [0, 0, 10, 10]}],
                "errors": [],
                "raw_output_json": None,
                "finish_reason": "length",
            }
        ],
    )
    health = _build_rollout_health(
        gt_vs_pred_path=gt_vs_pred,
        proposal_rows=[],
        gate=load_study_config(
            Path("configs/analysis/rollout_fn_factor_study/smoke.yaml")
        ).health_gate,
    )
    assert health["rollout_health_valid"] is True
    assert health["invalid_rollout_count"] == 0


def test_build_recovery_table_uses_minimal_intervention_precedence(tmp_path: Path) -> None:
    gt_row = {
        "index": 0,
        "record_idx": 0,
        "source_image_id": 139,
        "file_name": "images/val2017/000000000139.jpg",
        "width": 1000,
        "height": 1000,
        "gt": [
            {"desc": "person", "bbox": [0, 0, 100, 100]},
            {"desc": "dog", "bbox": [200, 200, 260, 260]},
            {"desc": "chair", "bbox": [400, 400, 460, 460]},
        ],
        "pred": [],
        "errors": [],
        "raw_output_json": {"objects": []},
    }

    def write_eval(run_dir: Path, *, matches_050: list[dict], matches_030: list[dict]) -> None:
        _write_jsonl(run_dir / "gt_vs_pred.jsonl", [gt_row])
        _write_jsonl(run_dir / "eval" / "matches.jsonl", matches_050)
        _write_jsonl(run_dir / "eval" / "matches@0.30.jsonl", matches_030)

    # Baseline misses all 3 objects at 0.50 but has a relaxed 0.30 match for gt_idx=2.
    baseline_dir = tmp_path / "baseline"
    write_eval(
        baseline_dir,
        matches_050=[],
        matches_030=[
            {
                "image_id": 0,
                "matches": [{"pred_idx": 0, "gt_idx": 2, "iou": 0.35, "pred_desc": "chair", "gt_desc": "chair", "sem_ok": True}],
                "unmatched_pred_indices": [],
                "ignored_pred_indices": [],
                "unmatched_gt_indices": [0, 1],
            }
        ],
    )
    baseline_manifest = {"samples": [{"sample_dir": str(baseline_dir)}]}

    # Sampling recovers gt_idx=0, so it must become decode_selection_miss.
    sampling_dirs = []
    for sample_idx in range(2):
        sample_dir = tmp_path / f"sampling_{sample_idx}"
        sampling_dirs.append(sample_dir)
        if sample_idx == 0:
            matches = []
        else:
            matches = [
                {
                    "image_id": 0,
                    "matches": [{"pred_idx": 0, "gt_idx": 0, "iou": 0.8, "pred_desc": "person", "gt_desc": "person", "sem_ok": True}],
                    "unmatched_pred_indices": [],
                    "ignored_pred_indices": [],
                    "unmatched_gt_indices": [1, 2],
                }
            ]
        write_eval(sample_dir, matches_050=matches, matches_030=matches)
    sampling_manifest = {
        "samples": [{"sample_dir": str(path)} for path in sampling_dirs]
    }

    # Prefix continuation recovers gt_idx=1, so it becomes prefix_sensitive_miss.
    prefix_dir = tmp_path / "prefix"
    _write_jsonl(
        prefix_dir / "gt_vs_pred.jsonl",
        [{**gt_row, "prefix_pred_count": 1, "pred": []}],
    )
    _write_jsonl(prefix_dir / "eval" / "matches.jsonl", [])
    _write_jsonl(prefix_dir / "eval" / "matches@0.30.jsonl", [])
    _write_jsonl(
        prefix_dir / "gt_vs_pred_continuation.jsonl",
        [{**gt_row, "pred": []}],
    )
    _write_jsonl(
        prefix_dir / "eval_continuation" / "matches.jsonl",
        [
            {
                "image_id": 0,
                "matches": [{"pred_idx": 0, "gt_idx": 1, "iou": 0.8, "pred_desc": "dog", "gt_desc": "dog", "sem_ok": True}],
                "unmatched_pred_indices": [],
                "ignored_pred_indices": [],
                "unmatched_gt_indices": [0, 2],
            }
        ],
    )
    _write_jsonl(
        prefix_dir / "eval_continuation" / "matches@0.30.jsonl",
        [
            {
                "image_id": 0,
                "matches": [{"pred_idx": 0, "gt_idx": 1, "iou": 0.8, "pred_desc": "dog", "gt_desc": "dog", "sem_ok": True}],
                "unmatched_pred_indices": [],
                "ignored_pred_indices": [],
                "unmatched_gt_indices": [0, 2],
            }
        ],
    )
    prefix_manifest = {
        "logical_cell_id": "prefix-cell",
        "samples": [{"sample_dir": str(prefix_dir)}],
    }

    # Length recovers nothing.
    length_dir = tmp_path / "length"
    write_eval(length_dir, matches_050=[], matches_030=[])
    length_manifest = {"samples": [{"sample_dir": str(length_dir)}]}

    rows, summary = _build_recovery_table(
        split_name="val",
        checkpoint_alias="original",
        baseline_manifest=baseline_manifest,
        sampling_manifest=sampling_manifest,
        prefix_manifests=[prefix_manifest],
        length_manifest=length_manifest,
    )

    by_gt = {int(row["gt_idx"]): row for row in rows}
    assert by_gt[0]["status"] == "decode_selection_miss"
    assert by_gt[1]["status"] == "prefix_sensitive_miss"
    assert by_gt[2]["status"] == "persistent_unrecovered"
    assert by_gt[2]["annotation_mismatch_candidate"] is True
    assert summary["status_counts"]["decode_selection_miss"] == 1
    assert summary["status_counts"]["prefix_sensitive_miss"] == 1


def test_load_stage_manifest_if_present_returns_empty_for_missing_stage(tmp_path: Path) -> None:
    assert _load_stage_manifest_if_present(tmp_path, "prefix") == []

    stage_dir = tmp_path / "prefix_stage"
    stage_dir.mkdir(parents=True)
    (stage_dir / "stage_manifest.json").write_text(
        json.dumps({"stage": "prefix", "cells": [{"logical_cell_id": "prefix-cell"}]}),
        encoding="utf-8",
    )
    assert _load_stage_manifest_if_present(tmp_path, "prefix") == [
        {"logical_cell_id": "prefix-cell"}
    ]


def test_worker_assignments_round_robin_when_fewer_gpus_than_pairs(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
run:
  name: rr
  output_dir: output/analysis
  stages: [report]

prompts:
  prompt_variant: coco_80
  object_field_order: desc_first
  do_resize: false

bootstrap:
  candidate_pool_limit: 16
  hard32_size: 16
  hard16_size: 16

baseline_decode:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 64
  repetition_penalty: 1.0
  batch_size: 1

sampling:
  k: 1
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 64
  repetition_penalty: 1.0
  batch_size: 1

execution:
  gpu_ids: [0, 1]

checkpoints:
  - alias: original
    path: output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332
  - alias: a_only
    path: output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900

splits:
  train:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
  val:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_study_config(config_path)
    assignments = _worker_assignments(config)
    assert assignments == {
        ("train", "original"): 0,
        ("train", "a_only"): 1,
        ("val", "original"): 0,
        ("val", "a_only"): 1,
    }
