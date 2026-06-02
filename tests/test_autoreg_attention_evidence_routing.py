from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.analysis.autoreg_attention_evidence_routing import (
    aggregate_attention_for_query,
    attention_record_selected,
    attention_shard_label,
    box_iou_xyxy,
    build_attention_dry_run_plan,
    build_candidate_region_rows,
    build_patch_region_membership,
    classify_extra_prediction,
    find_visual_token_spans,
    load_attention_config,
    merge_attention_shards,
    normalize_attention_shard,
    select_attention_cases_from_rows,
    select_teacher_forced_anchor_cases_from_dataset_rows,
)
from src.datasets.geometry import box_iou_xyxy as geometry_box_iou_xyxy


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_attention_config(path: Path, *, artifact_root: Path | None = None) -> None:
    root = path.parent / "attention_root" if artifact_root is None else artifact_root
    path.write_text(
        f"""
paths:
  artifact_root: {root}
  checkpoint: {path.parent / "checkpoint"}
  dataset_jsonl: {path.parent / "val.coord.jsonl"}
  lane_a_rollout_root: {path.parent / "rollout_anatomy"}
  lane_c_per_case: {path.parent / "x1_basin_attribution" / "per_case.jsonl"}
  lane_c_study_config: {path.parent / "lane_c.yaml"}
  lane_d_selected_cases: {path.parent / "hidden_state_probe" / "selected_cases.jsonl"}
  self_rollout_root: {path.parent / "self_rollout"}
selection:
  sample_limit: 200
  max_cases: 512
  prefer_prefix_mode: self_prefix
  min_remaining_gt: 1
regions:
  context_expansion_norm1000: 64
  duplicate_iou_threshold: 0.95
execution:
  batch_size: 1
  attn_implementation: eager
  torch_dtype: bfloat16
  max_feasibility_cases: 4
""".lstrip(),
        encoding="utf-8",
    )


def test_attention_shard_selection_keeps_images_together() -> None:
    assert attention_shard_label(0, 8) == "shard_000-of-008"
    assert attention_record_selected(16, shard_index=0, num_shards=8)
    assert not attention_record_selected(17, shard_index=0, num_shards=8)


def test_normalize_attention_shard_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="provided together"):
        normalize_attention_shard(shard_index=0, num_shards=None)
    with pytest.raises(ValueError, match="num_shards"):
        normalize_attention_shard(shard_index=0, num_shards=0)
    with pytest.raises(ValueError, match="shard_index"):
        normalize_attention_shard(shard_index=8, num_shards=8)


def test_box_iou_xyxy_uses_geometry_helper_closed_open_area() -> None:
    assert box_iou_xyxy is geometry_box_iou_xyxy
    assert box_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)
    assert box_iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1.0 / 3.0)


def test_classify_extra_prediction_only_marks_same_desc_iou_gt_0p95_duplicate() -> None:
    previous = [
        {"desc": "chair", "bbox_xyxy": [10, 10, 100, 100]},
        {"desc": "person", "bbox_xyxy": [300, 300, 500, 900]},
    ]
    assert (
        classify_extra_prediction(
            {"desc": "chair", "bbox_xyxy": [11, 10, 100, 100]},
            previous,
        )
        == "same_desc_iou_gt_0p95_duplicate"
    )
    assert (
        classify_extra_prediction(
            {"desc": "table", "bbox_xyxy": [10, 10, 100, 100]},
            previous,
        )
        == "other_extra_prediction"
    )
    assert (
        classify_extra_prediction(
            {"desc": "chair", "bbox_xyxy": [200, 200, 280, 280]},
            previous,
        )
        == "other_extra_prediction"
    )
    assert (
        classify_extra_prediction(
            {"desc": "chair", "bbox_xyxy": [100, 10, 10, 100]},
            previous,
        )
        == "format_or_geometry_invalid_extra"
    )


def test_load_attention_config_and_dry_run_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "attention.yaml"
    _write_attention_config(config_path)
    config = load_attention_config(config_path)
    assert config.paths.artifact_root == tmp_path / "attention_root"
    assert config.selection.sample_limit == 200
    assert config.selection.case_source == "lane_d_self_prefix_remaining_gt"
    plan = build_attention_dry_run_plan(
        config,
        stages=("select_cases", "feasibility"),
        shard_index=0,
        num_shards=8,
    )
    assert plan["artifact_root"] == str(tmp_path / "attention_root")
    assert plan["shard_label"] == "shard_000-of-008"
    assert plan["stages"] == ["select_cases", "feasibility"]
    assert plan["attn_implementation"] == "eager"


def test_select_attention_cases_prefers_self_prefix_missed_gt() -> None:
    lane_d_cases = [
        {
            "case_id": "row0:self_prefix:depth2:gt3",
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 2,
            "prefix_quality": "clean_prefix",
            "intended_target_gt_idx": 3,
            "target_desc": "vase",
            "x1_target_rank": 350,
            "x1_top_peak_attribution": "no_local_object_diffuse",
        },
        {
            "case_id": "row1:teacher_forced:depth0:gt0",
            "source_line_idx": 1,
            "prefix_mode": "teacher_forced",
            "prefix_depth": 0,
            "prefix_quality": "gt_prefix",
            "intended_target_gt_idx": 0,
            "target_desc": "person",
            "x1_target_rank": 1,
            "x1_top_peak_attribution": "target_gt_object",
        },
    ]
    rows = select_attention_cases_from_rows(
        lane_d_cases,
        shard_index=0,
        num_shards=8,
        max_cases=8,
        prefer_prefix_mode="self_prefix",
    )
    assert [row["case_id"] for row in rows] == ["row0:self_prefix:depth2:gt3"]
    assert rows[0]["attention_case_family"] == "missed_gt_evidence_routing"


def test_select_teacher_forced_anchor_cases_uses_all_gt_objects() -> None:
    dataset_rows = [
        {
            "objects": [
                {"desc": "person", "bbox_2d": [10, 20, 100, 200]},
                {"desc": "vase", "bbox_2d": [300, 400, 350, 500]},
            ],
            "width": 1000,
            "height": 1000,
        },
        {
            "objects": [
                {"desc": "chair", "bbox_2d": [100, 100, 200, 300]},
            ],
            "width": 1000,
            "height": 1000,
        },
    ]
    cases = select_teacher_forced_anchor_cases_from_dataset_rows(
        dataset_rows,
        lane_c_config=None,
        shard_index=0,
        num_shards=8,
        max_cases=8,
        scope_label="train200_teacher_forced_anchor",
    )
    assert [case["case_id"] for case in cases] == [
        "row0:teacher_forced:depth0:gt0",
        "row0:teacher_forced:depth1:gt1",
    ]
    assert cases[0]["prefix_mode"] == "teacher_forced"
    assert cases[0]["attention_case_family"] == "train_teacher_forced_anchor"


def test_build_candidate_region_rows_emits_target_same_desc_and_context() -> None:
    selected_case = {
        "case_id": "row0:self_prefix:depth2:gt1",
        "source_line_idx": 0,
        "intended_target_gt_idx": 1,
        "target_desc": "vase",
        "prefix_depth": 2,
        "prefix_mode": "self_prefix",
        "prefix_quality": "fp_prefix",
    }
    dataset_row = {
        "objects": [
            {"desc": "chair", "bbox_2d": [0, 0, 100, 100]},
            {"desc": "vase", "bbox_2d": [200, 200, 260, 300]},
            {"desc": "vase", "bbox_2d": [700, 700, 760, 820]},
        ],
        "width": 1000,
        "height": 1000,
    }
    rows = build_candidate_region_rows(
        selected_case,
        dataset_row,
        context_expansion_norm1000=64,
        shard_index=0,
        num_shards=8,
    )
    kinds = {(row["region_kind"], row.get("gt_idx")) for row in rows}
    assert ("target_gt", 1) in kinds
    assert ("same_desc_gt", 2) in kinds
    assert ("context_ring", 1) in kinds
    assert all(row["case_id"] == selected_case["case_id"] for row in rows)


def test_build_candidate_region_rows_accepts_coord_token_boxes() -> None:
    selected_case = {
        "case_id": "row0:teacher_forced:depth0:gt0",
        "source_line_idx": 0,
        "intended_target_gt_idx": 0,
        "target_desc": "vase",
        "prefix_depth": 0,
        "prefix_mode": "teacher_forced",
        "prefix_quality": "gt_prefix",
    }
    dataset_row = {
        "objects": [
            {
                "desc": "vase",
                "bbox_2d": [
                    "<|coord_200|>",
                    "<|coord_200|>",
                    "<|coord_260|>",
                    "<|coord_300|>",
                ],
            }
        ]
    }
    rows = build_candidate_region_rows(
        selected_case,
        dataset_row,
        context_expansion_norm1000=64,
        shard_index=0,
        num_shards=8,
    )
    target = next(row for row in rows if row["region_kind"] == "target_gt")
    assert target["bbox_xyxy"] == [200, 200, 260, 300]


def test_find_visual_token_spans_groups_contiguous_image_pad_tokens() -> None:
    spans = find_visual_token_spans([1, 9, 9, 2, 9, 9, 9, 3], image_token_id=9)
    assert spans == [(1, 3), (4, 7)]


def test_build_patch_region_membership_maps_grid_centers_to_regions() -> None:
    regions = [
        {
            "case_id": "case",
            "region_kind": "target_gt",
            "bbox_xyxy": [0, 0, 500, 500],
        },
        {
            "case_id": "case",
            "region_kind": "far_background",
            "bbox_xyxy": [0, 0, 999, 999],
            "exclude_bbox_xyxy": [0, 0, 500, 500],
        },
    ]
    membership = build_patch_region_membership(
        visual_token_start=10,
        grid_h=2,
        grid_w=2,
        region_rows=regions,
    )
    assert membership["target_gt"] == [10]
    assert sorted(membership["far_background"]) == [11, 12, 13]


def test_aggregate_attention_for_query_sums_region_mass_per_head() -> None:
    attention = torch.zeros(1, 2, 6, 6)
    attention[0, 0, 5, 1] = 0.25
    attention[0, 0, 5, 2] = 0.25
    attention[0, 0, 5, 3] = 0.50
    attention[0, 1, 5, 1] = 0.10
    attention[0, 1, 5, 4] = 0.90
    rows = aggregate_attention_for_query(
        attention,
        batch_idx=0,
        query_index=5,
        layer_index=3,
        role="pre_x1",
        region_membership={"target_gt": [1, 2], "far_background": [4]},
        base_row={"case_id": "case", "source_line_idx": 0},
    )
    by_key = {(row["head"], row["region_kind"]): row for row in rows}
    assert by_key[(0, "target_gt")]["attention_mass"] == pytest.approx(0.5)
    assert by_key[(1, "far_background")]["attention_mass"] == pytest.approx(0.9)


def test_merge_attention_shards_requires_expected_shards(tmp_path: Path) -> None:
    root = tmp_path / "attention"
    for shard_idx in range(2):
        label = f"shard_{shard_idx:03d}-of-002"
        shard = root / "shards" / label
        _write_jsonl(
            shard / "selected_cases.jsonl",
            [{"case_id": f"case-{shard_idx}", "source_line_idx": shard_idx}],
        )
        _write_jsonl(shard / "candidate_region_rows.jsonl", [])
        _write_jsonl(shard / "feasibility_rows.jsonl", [])
        _write_jsonl(shard / "attention_region_rows.jsonl", [])
        _write_jsonl(shard / "decision_context_rows.jsonl", [])
        (shard / "summary.json").write_text(
            json.dumps({"shard_label": label, "row_counts": {"selected_cases": 1}}),
            encoding="utf-8",
        )
    summary = merge_attention_shards(root, expected_shards=2)
    assert summary["analysis_name"] == "autoreg_attention_evidence_routing"
    assert summary["expected_shards"] == 2
    assert summary["row_counts"]["selected_cases"] == 2
    assert (root / "merge_summary.json").exists()
