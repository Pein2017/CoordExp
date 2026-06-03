from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.autoreg_fn_rescue_desc_x1_phase2 import (
    _norm1000_bbox_to_pixel_box,
    _select_intervention_smoke_rows,
    build_phase2_dry_run_plan,
    load_generation_outcomes,
    load_phase2_config,
    materialize_intervention_plan,
    materialize_attention_mining,
    mine_attention_rows,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_config(path: Path, *, artifact_root: Path, fn_rescue_root: Path) -> None:
    path.write_text(
        f"""
paths:
  artifact_root: {artifact_root}
  fn_rescue_root: {fn_rescue_root}
evidence_scope: unit_phase2
mining:
  max_attention_rows: null
  ranking_limit: 4
  min_rows_per_head: 1
  aggregation_scopes: [union]
  region_kinds:
    - target_gt
    - same_desc_competitor_gt_object
    - far_background
""".lstrip(),
        encoding="utf-8",
    )


def _generation_rows() -> list[dict[str, object]]:
    return [
        {
            "case_id": "case-a",
            "rescue_tier": "desc_only",
            "primary_rescue_success": True,
            "success_iou50": True,
            "binding_bucket": "same_desc_competitor",
        },
        {
            "case_id": "case-b",
            "rescue_tier": "desc_only",
            "primary_rescue_success": False,
            "success_iou50": False,
            "binding_bucket": "same_desc_competitor",
        },
    ]


def _attention_rows() -> list[dict[str, object]]:
    base = {
        "rescue_tier": "desc_only",
        "role": "pre_x1",
        "layer": 2,
        "head": 3,
        "aggregation_scope": "union",
    }
    return [
        {
            **base,
            "case_id": "case-a",
            "region_kind": "target_gt",
            "attention_mass_normalized": 0.70,
        },
        {
            **base,
            "case_id": "case-a",
            "region_kind": "same_desc_competitor_gt_object",
            "attention_mass_normalized": 0.10,
        },
        {
            **base,
            "case_id": "case-a",
            "region_kind": "far_background",
            "attention_mass_normalized": 0.20,
        },
        {
            **base,
            "case_id": "case-b",
            "region_kind": "target_gt",
            "attention_mass_normalized": 0.20,
        },
        {
            **base,
            "case_id": "case-b",
            "region_kind": "same_desc_competitor_gt_object",
            "attention_mass_normalized": 0.50,
        },
        {
            **base,
            "case_id": "case-b",
            "region_kind": "far_background",
            "attention_mass_normalized": 0.30,
        },
        {
            **base,
            "case_id": "case-c",
            "region_kind": "target_gt",
            "attention_mass_normalized": 0.99,
        },
        {
            **base,
            "case_id": "case-a",
            "aggregation_scope": "instance",
            "region_kind": "target_gt",
            "attention_mass_normalized": 0.99,
        },
    ]


def test_load_phase2_config_and_dry_run_plan(tmp_path: Path) -> None:
    fn_root = tmp_path / "fn"
    (fn_root / "rescue_attention_region_rows.jsonl").parent.mkdir(parents=True)
    (fn_root / "rescue_attention_region_rows.jsonl").write_text("", encoding="utf-8")
    (fn_root / "rescue_generation_rows.jsonl").write_text("", encoding="utf-8")
    config_path = tmp_path / "phase2.yaml"
    _write_config(config_path, artifact_root=tmp_path / "out", fn_rescue_root=fn_root)

    config = load_phase2_config(config_path)
    plan = build_phase2_dry_run_plan(config, stages=("attention_mining",))

    assert config.paths.artifact_root == tmp_path / "out"
    assert config.evidence_scope == "unit_phase2"
    assert plan["attention_rows_exists"] is True
    assert plan["generation_rows_exists"] is True
    assert plan["aggregation_scopes"] == ["union"]


def test_load_phase2_config_resolves_relative_fn_rescue_config_path(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "analysis" / "autoreg_fn_rescue_desc_x1_phase2"
    continuation = config_dir.parent / "autoreg_fn_rescue_continuation" / "linked.yaml"
    continuation.parent.mkdir(parents=True, exist_ok=True)
    continuation.write_text("paths: {}\n", encoding="utf-8")
    config_path = config_dir / "phase2.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
paths:
  artifact_root: {tmp_path / "out"}
  fn_rescue_root: {tmp_path / "fn"}
intervention_plan:
  fn_rescue_config_path: ../autoreg_fn_rescue_continuation/linked.yaml
""".lstrip(),
        encoding="utf-8",
    )

    config = load_phase2_config(config_path)

    assert config.intervention_plan.fn_rescue_config_path == continuation.resolve(strict=False)


def test_mine_attention_rows_ranks_target_and_background(tmp_path: Path) -> None:
    generation_path = tmp_path / "rescue_generation_rows.jsonl"
    attention_path = tmp_path / "rescue_attention_region_rows.jsonl"
    _write_jsonl(generation_path, _generation_rows())
    _write_jsonl(attention_path, _attention_rows())

    outcomes = load_generation_outcomes(generation_path)
    rows, rankings, summary = mine_attention_rows(
        attention_path=attention_path,
        generation_outcomes=outcomes,
        max_attention_rows=None,
        aggregation_scopes=("union",),
        region_kinds=(
            "target_gt",
            "same_desc_competitor_gt_object",
            "far_background",
        ),
        ranking_limit=4,
        min_rows_per_head=1,
    )

    target_row = next(row for row in rows if row["region_kind"] == "target_gt")
    assert target_row["attention_mass_mean"] == pytest.approx(0.45)
    assert target_row["success_attention_mass_mean"] == pytest.approx(0.70)
    assert target_row["failure_attention_mass_mean"] == pytest.approx(0.20)
    assert target_row["success_minus_failure_attention"] == pytest.approx(0.50)
    assert summary["processed_attention_rows"] == 8
    assert summary["joined_attention_rows"] == 6
    assert summary["skipped_scope_rows"] == 1
    assert summary["missing_generation_rows"] == 1
    assert rankings["target_margin_top"][0]["target_minus_competitor"] == pytest.approx(0.15)
    assert rankings["background_sink_top"][0]["far_background_mean"] == pytest.approx(0.25)


def test_materialize_attention_mining_writes_artifacts(tmp_path: Path) -> None:
    fn_root = tmp_path / "fn"
    _write_jsonl(fn_root / "rescue_generation_rows.jsonl", _generation_rows())
    _write_jsonl(fn_root / "rescue_attention_region_rows.jsonl", _attention_rows())
    config_path = tmp_path / "phase2.yaml"
    artifact_root = tmp_path / "out"
    _write_config(config_path, artifact_root=artifact_root, fn_rescue_root=fn_root)
    config = load_phase2_config(config_path)

    summary = materialize_attention_mining(config)

    assert summary["joined_attention_rows"] == 6
    assert (artifact_root / "attention_mining" / "summary.json").exists()
    assert (artifact_root / "attention_mining" / "head_layer_rankings.json").exists()
    assert (artifact_root / "attention_mining" / "head_layer_region_summary.jsonl").exists()
    report = (artifact_root / "report.md").read_text(encoding="utf-8")
    assert "Interpretation Boundary" in report


def test_materialize_intervention_plan_writes_region_controls(tmp_path: Path) -> None:
    fn_root = tmp_path / "fn"
    _write_jsonl(
        fn_root / "rescue_generation_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "source_line_idx": 0,
                "rescue_tier": "desc_x1",
                "primary_rescue_success": True,
                "binding_bucket": "same_desc_competitor",
                "target_desc": "vase",
            },
            {
                "case_id": "case-b",
                "source_line_idx": 1,
                "rescue_tier": "desc_x1_wrong_control",
                "primary_rescue_success": False,
                "binding_bucket": "same_desc_competitor",
                "target_desc": "vase",
            },
            {
                "case_id": "case-c",
                "source_line_idx": 2,
                "rescue_tier": "desc_only",
                "primary_rescue_success": False,
                "binding_bucket": "other",
                "target_desc": "chair",
            },
        ],
    )
    _write_jsonl(
        fn_root / "rescue_candidate_region_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "planned_rescue_tier": "desc_x1",
                "region_kind": "target_gt",
                "region_instance_id": "gt:1",
                "bbox_xyxy": [10, 20, 30, 40],
            },
            {
                "case_id": "case-a",
                "planned_rescue_tier": "desc_x1",
                "region_kind": "same_desc_competitor_gt_object",
                "region_instance_id": "gt:2",
                "bbox_xyxy": [100, 120, 130, 140],
            },
            {
                "case_id": "case-a",
                "planned_rescue_tier": "desc_x1",
                "region_kind": "far_background",
                "region_instance_id": "far_background:0",
                "bbox_xyxy": [0, 0, 999, 999],
            },
            {
                "case_id": "case-b",
                "planned_rescue_tier": "desc_x1_wrong_control",
                "region_kind": "wrong_control_source_region",
                "region_instance_id": "raw_pred:7",
                "bbox_xyxy": [200, 220, 230, 240],
            },
        ],
    )
    config_path = tmp_path / "phase2.yaml"
    artifact_root = tmp_path / "out"
    _write_config(config_path, artifact_root=artifact_root, fn_rescue_root=fn_root)
    config = load_phase2_config(config_path)

    summary = materialize_intervention_plan(config)

    rows = [
        json.loads(line)
        for line in (artifact_root / "intervention_plan" / "selected_interventions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["planned_intervention_rows"] == len(rows)
    assert summary["case_count"] == 2
    assert {
        (row["case_id"], row["intervention_kind"])
        for row in rows
    } >= {
        ("case-a", "no_op_control"),
        ("case-a", "target_gt_mask"),
        ("case-a", "same_desc_competitor_mask"),
        ("case-a", "far_background_sink_mask"),
        ("case-b", "wrong_control_source_region_mask"),
    }
    assert all(row["causal_status"] == "planned_not_executed" for row in rows)


def test_select_intervention_smoke_rows_requires_paired_controls() -> None:
    plan_rows = [
        {
            "case_id": "case-a",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "no_op_control",
        },
        {
            "case_id": "case-a",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "target_gt_mask",
        },
        {
            "case_id": "case-b",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "no_op_control",
        },
        {
            "case_id": "case-c",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "target_gt_mask",
        },
        {
            "case_id": "case-c",
            "rescue_tier": "desc_x1",
            "selection_kind": "desc_x1_success_same_desc_competitor",
            "intervention_kind": "no_op_control",
        },
    ]

    selected = _select_intervention_smoke_rows(
        plan_rows,
        intervention_kinds=("no_op_control", "target_gt_mask"),
        max_cases=1,
    )

    assert [(row["case_id"], row["intervention_kind"]) for row in selected] == [
        ("case-a", "no_op_control"),
        ("case-a", "target_gt_mask"),
    ]


def test_norm1000_bbox_to_pixel_box_clips_and_rejects_invalid_boxes() -> None:
    assert _norm1000_bbox_to_pixel_box([0, 0, 999, 999], width=200, height=100) == (
        0,
        0,
        199,
        99,
    )
    assert _norm1000_bbox_to_pixel_box([250, 100, 750, 900], width=200, height=100) == (
        50,
        10,
        149,
        89,
    )
    assert _norm1000_bbox_to_pixel_box([20, 20, 10, 30], width=200, height=100) is None
