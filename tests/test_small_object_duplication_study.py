from __future__ import annotations

from pathlib import Path

from src.analysis.small_object_duplication_study import (
    _duplicate_metrics,
    _extract_focus_match,
    _select_cohorts,
    load_study_config,
)


def test_load_study_config_reads_core_surfaces(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
run:
  name: duplication-study
  output_dir: output/analysis
  stages: [cohort, decode]

monitor_dumps:
  paths: [output/example/monitor_dumps]
  top_duplication_cases: 5
  top_control_cases: 3
  min_gt_objects: 4
  min_pred_objects: 4
  crowded_min_gt_objects: 8
  small_area_max_ratio: 0.004
  duplicate_iou_threshold: 0.75
  center_radius_scale: 0.9
  control_max_duplication_score: 0.5

checkpoint:
  alias: test-ckpt
  path: output/stage2_ab/prod/pseudo_positive-ckpt_300_merged-v1

execution:
  device: cuda:0
  cuda_visible_devices: "0"
  decode_batch_size: 2
  score_batch_size: 3

decode:
  temperatures: [0.0, 0.1]
  top_p: 0.95
  max_new_tokens: 1024
  repetition_penalty: 1.05
  sample_seeds: [11, 12]

prefix:
  max_cases: 4
  focus_policy: earliest_matched_small_or_first_matched
  sources: [pred, gt]
  jitter_offsets: [[-2, 0], [2, 0]]
  match_iou_threshold: 0.5

scoring:
  device: cuda:0
  attn_implementation: auto
  max_cases: 4
  match_iou_threshold: 0.5
  max_remaining_gt_candidates: 3
  duplicate_jitter_offsets: [[-2, 0], [2, 0]]
  include_close_candidate: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_study_config(config_path)
    assert config.run.name == "duplication-study"
    assert config.monitor_dumps.top_duplication_cases == 5
    assert config.execution.cuda_visible_devices == "0"
    assert config.decode.temperatures == (0.0, 0.1)
    assert config.prefix.jitter_offsets == ((-2, 0), (2, 0))


def test_duplicate_metrics_detects_small_shifted_cluster() -> None:
    objects = [
        {
            "index": 0,
            "desc_norm": "laptop",
            "bbox_norm1000": [100, 100, 130, 130],
            "center_norm1000": [115.0, 115.0],
            "area_bins": 900.0,
            "area_ratio": 0.0009,
            "anchor": [100.0, 100.0],
        },
        {
            "index": 1,
            "desc_norm": "laptop",
            "bbox_norm1000": [102, 102, 132, 132],
            "center_norm1000": [117.0, 117.0],
            "area_bins": 900.0,
            "area_ratio": 0.0009,
            "anchor": [102.0, 102.0],
        },
        {
            "index": 2,
            "desc_norm": "laptop",
            "bbox_norm1000": [104, 104, 134, 134],
            "center_norm1000": [119.0, 119.0],
            "area_bins": 900.0,
            "area_ratio": 0.0009,
            "anchor": [104.0, 104.0],
        },
        {
            "index": 3,
            "desc_norm": "person",
            "bbox_norm1000": [700, 100, 900, 900],
            "center_norm1000": [800.0, 500.0],
            "area_bins": 160000.0,
            "area_ratio": 0.16,
            "anchor": [100.0, 700.0],
        },
    ]
    metrics = _duplicate_metrics(
        objects,
        small_area_max_ratio=0.003,
        duplicate_iou_threshold=0.7,
        center_radius_scale=0.8,
    )
    assert metrics["duplicate_like_pair_count"] == 3
    assert metrics["small_duplicate_like_pair_count"] == 3
    assert metrics["duplicate_like_max_cluster_size"] == 3
    assert metrics["small_duplicate_like_max_cluster_size"] == 3
    assert metrics["largest_cluster_linearity"] is not None
    assert metrics["largest_cluster_linearity"] > 0.9


def test_extract_focus_match_prefers_small_matched_prediction() -> None:
    sample = {
        "match": {
            "matched_pair_details": [
                {"pred_i": 0, "gt_i": 0, "bbox_iou_norm1000": 0.95},
                {"pred_i": 1, "gt_i": 1, "bbox_iou_norm1000": 0.92},
            ]
        }
    }
    pred_objects = [
        {
            "index": 0,
            "desc_norm": "person",
            "bbox_norm1000": [0, 0, 500, 900],
            "area_ratio": 0.45,
        },
        {
            "index": 1,
            "desc_norm": "laptop",
            "bbox_norm1000": [100, 100, 130, 130],
            "area_ratio": 0.0009,
        },
    ]
    gt_objects = [
        {
            "index": 0,
            "desc_norm": "person",
            "bbox_norm1000": [0, 0, 500, 900],
            "area_ratio": 0.45,
        },
        {
            "index": 1,
            "desc_norm": "laptop",
            "bbox_norm1000": [101, 101, 131, 131],
            "area_ratio": 0.0009,
        },
    ]
    focus = _extract_focus_match(
        sample,
        pred_objects=pred_objects,
        gt_objects=gt_objects,
        small_area_max_ratio=0.003,
        match_iou_threshold=0.5,
        focus_policy="earliest_matched_small_or_first_matched",
    )
    assert focus is not None
    assert focus["pred_i"] == 1
    assert focus["gt_i"] == 1


def test_select_cohorts_separates_duplication_cases_and_controls(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
run:
  name: duplication-study
  output_dir: output/analysis

monitor_dumps:
  paths: [output/example/monitor_dumps]
  top_duplication_cases: 1
  top_control_cases: 1
  min_gt_objects: 4
  min_pred_objects: 4
  crowded_min_gt_objects: 8
  small_area_max_ratio: 0.003
  duplicate_iou_threshold: 0.7
  center_radius_scale: 0.8
  control_max_duplication_score: 0.0

checkpoint:
  alias: test-ckpt
  path: output/stage2_ab/prod/pseudo_positive-ckpt_300_merged-v1

execution:
  device: cuda:0
  decode_batch_size: 2
  score_batch_size: 2

decode:
  temperatures: [0.0]
  sample_seeds: [11]
  top_p: 0.95
  max_new_tokens: 512
  repetition_penalty: 1.05

prefix:
  max_cases: 2
  focus_policy: earliest_matched_small_or_first_matched
  sources: [pred]
  jitter_offsets: [[-2, 0]]
  match_iou_threshold: 0.5

scoring:
  device: cuda:0
  attn_implementation: auto
  max_cases: 2
  match_iou_threshold: 0.5
  max_remaining_gt_candidates: 2
  duplicate_jitter_offsets: [[-2, 0]]
  include_close_candidate: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_study_config(config_path)
    rows = [
        {
            "case_key": "dup",
            "gt_count": 20,
            "small_gt_count": 6,
            "duplication_score": 40.0,
        },
        {
            "case_key": "control",
            "gt_count": 22,
            "small_gt_count": 7,
            "duplication_score": 0.0,
        },
    ]
    duplication_cases, controls = _select_cohorts(rows, cfg=cfg)
    assert [row["case_key"] for row in duplication_cases] == ["dup"]
    assert [row["case_key"] for row in controls] == ["control"]
