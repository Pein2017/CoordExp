from __future__ import annotations

import json
from pathlib import Path

from src.analysis.autoreg_fn_rescue_desc_x1_phase4 import (
    build_phase4_dry_run_plan,
    load_phase4_config,
    materialize_desc_x1_probe_linkage,
    materialize_instance_attention_binding,
    write_phase4_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    *,
    artifact_root: Path,
    phase3_root: Path,
    fn_root: Path,
    lane_d_root: Path,
) -> None:
    path.write_text(
        f"""
paths:
  artifact_root: {artifact_root}
  phase3_root: {phase3_root}
  fn_rescue_root: {fn_root}
  lane_d_root: {lane_d_root}
evidence_scope: unit_phase4
instance_attention:
  max_attention_rows: null
  roles: [pre_x1, pre_y1]
  region_kinds:
    - target_gt
    - same_desc_competitor_gt_object
    - same_desc_rollout_prediction
    - wrong_control_source_region
    - context_ring
    - far_background
""".lstrip(),
        encoding="utf-8",
    )


def test_load_phase4_config_and_dry_run_plan(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase3_root = tmp_path / "phase3"
    fn_root = tmp_path / "fn"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(phase3_root / "case_linked" / "case_mechanism_rows.jsonl", [])
    _write_jsonl(fn_root / "rescue_attention_region_rows.jsonl", [])
    config_path = tmp_path / "phase4.yaml"
    _write_config(
        config_path,
        artifact_root=artifact_root,
        phase3_root=phase3_root,
        fn_root=fn_root,
        lane_d_root=lane_d_root,
    )

    config = load_phase4_config(config_path)
    plan = build_phase4_dry_run_plan(
        config,
        stages=("instance_attention_binding", "desc_x1_probe_linkage", "report"),
    )

    assert config.paths.artifact_root == artifact_root
    assert plan["phase3_case_rows_exists"] is True
    assert plan["attention_rows_exists"] is True
    assert plan["probe_rows_exists"] is False
    assert plan["roles"] == ["pre_x1", "pre_y1"]


def test_materialize_instance_attention_binding(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase3_root = tmp_path / "phase3"
    fn_root = tmp_path / "fn"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(
        phase3_root / "case_linked" / "case_mechanism_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "intervention_kind": "target_gt_mask",
                "mechanism_bucket": "target_dependent",
                "target_iou_delta": -0.4,
                "primary_success_changed": True,
                "valid_paired_row": True,
            },
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "intervention_kind": "same_desc_competitor_mask",
                "mechanism_bucket": "competitor_dependent",
                "target_iou_delta": 0.2,
                "primary_success_changed": False,
                "valid_paired_row": True,
            },
        ],
    )
    _write_jsonl(
        fn_root / "rescue_attention_region_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 6,
                "aggregation_scope": "instance",
                "region_kind": "target_gt",
                "region_instance_id": "gt:1",
                "attention_mass_normalized": 0.8,
            },
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 13,
                "head": 6,
                "aggregation_scope": "instance",
                "region_kind": "same_desc_competitor_gt_object",
                "region_instance_id": "gt:2",
                "attention_mass_normalized": 0.2,
            },
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 16,
                "head": 1,
                "aggregation_scope": "instance",
                "region_kind": "target_gt",
                "region_instance_id": "gt:3",
                "attention_mass_normalized": 0.1,
            },
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 16,
                "head": 1,
                "aggregation_scope": "instance",
                "region_kind": "same_desc_competitor_gt_object",
                "region_instance_id": "gt:4",
                "attention_mass_normalized": 0.7,
            },
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "role": "pre_y1",
                "layer": 16,
                "head": 1,
                "aggregation_scope": "union",
                "region_kind": "target_gt",
                "attention_mass_normalized": 0.99,
            },
        ],
    )
    config_path = tmp_path / "phase4.yaml"
    _write_config(
        config_path,
        artifact_root=artifact_root,
        phase3_root=phase3_root,
        fn_root=fn_root,
        lane_d_root=lane_d_root,
    )

    summary = materialize_instance_attention_binding(load_phase4_config(config_path))

    rows = [
        json.loads(line)
        for line in (artifact_root / "instance_attention_binding" / "rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert summary["row_count"] == 2
    assert rows[0]["target_attention_mass"] == 0.8
    assert rows[0]["competitor_attention_mass"] == 0.2
    assert rows[0]["target_minus_competitor_instance_attention"] == 0.6000000000000001
    assert rows[0]["top_instance_attention_heads"] == [
        {"aggregation_scope": "instance", "head": 6, "layer": 13, "role": "pre_y1"}
    ]
    assert summary["bucket_summaries"]["target_dependent"]["mean_target_minus_competitor_instance_attention"] == 0.6000000000000001
    assert (artifact_root / "instance_attention_binding" / "summary.json").exists()
    assert (artifact_root / "instance_attention_binding" / "report.md").exists()


def test_materialize_desc_x1_probe_linkage_missing_probe_rows(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase3_root = tmp_path / "phase3"
    fn_root = tmp_path / "fn"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(phase3_root / "case_linked" / "case_mechanism_rows.jsonl", [{"case_id": "case-a"}])
    config_path = tmp_path / "phase4.yaml"
    _write_config(
        config_path,
        artifact_root=artifact_root,
        phase3_root=phase3_root,
        fn_root=fn_root,
        lane_d_root=lane_d_root,
    )

    summary = materialize_desc_x1_probe_linkage(load_phase4_config(config_path))

    assert summary["status"] == "blocked_missing_probe_rows"
    assert summary["phase3_case_rows"] == 1
    assert (artifact_root / "desc_x1_probe_linkage" / "summary.json").exists()
    assert (artifact_root / "desc_x1_probe_linkage" / "report.md").exists()


def test_materialize_desc_x1_probe_linkage_available_probe_rows(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase3_root = tmp_path / "phase3"
    fn_root = tmp_path / "fn"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(
        phase3_root / "case_linked" / "case_mechanism_rows.jsonl",
        [{"case_id": "case-a"}, {"case_id": "case-b"}],
    )
    _write_jsonl(
        lane_d_root / "probe_rows.jsonl",
        [
            {"case_id": "case-a", "role": "desc_end", "layer_group": "middle"},
            {"case_id": "case-c", "role": "pre_x1", "layer_group": "late"},
        ],
    )
    config_path = tmp_path / "phase4.yaml"
    _write_config(
        config_path,
        artifact_root=artifact_root,
        phase3_root=phase3_root,
        fn_root=fn_root,
        lane_d_root=lane_d_root,
    )

    summary = materialize_desc_x1_probe_linkage(load_phase4_config(config_path))

    assert summary["status"] == "linked_probe_rows_available"
    assert summary["probe_rows"] == 2
    assert summary["joined_case_count"] == 1
    assert summary["available_roles"] == ["desc_end", "pre_x1"]
    assert summary["available_layer_groups"] == ["late", "middle"]


def test_write_phase4_report(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase3_root = tmp_path / "phase3"
    fn_root = tmp_path / "fn"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(phase3_root / "case_linked" / "case_mechanism_rows.jsonl", [])
    _write_jsonl(fn_root / "rescue_attention_region_rows.jsonl", [])
    config_path = tmp_path / "phase4.yaml"
    _write_config(
        config_path,
        artifact_root=artifact_root,
        phase3_root=phase3_root,
        fn_root=fn_root,
        lane_d_root=lane_d_root,
    )
    config = load_phase4_config(config_path)
    materialize_instance_attention_binding(config)
    materialize_desc_x1_probe_linkage(config)

    report_path = write_phase4_report(config)

    assert report_path == artifact_root / "report.md"
    assert (artifact_root / "summary.json").exists()
    assert "Phase 4" in report_path.read_text(encoding="utf-8")
