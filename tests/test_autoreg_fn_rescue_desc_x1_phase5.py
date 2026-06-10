from __future__ import annotations

import json
from pathlib import Path

from src.analysis.autoreg_fn_rescue_desc_x1_phase5 import (
    build_phase5_dry_run_plan,
    load_phase5_config,
    materialize_coord_slot_logit_binding_probe,
    materialize_x1_logit_binding_probe,
    write_phase5_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_config(path: Path, *, artifact_root: Path, phase4_root: Path, lane_d_root: Path) -> None:
    path.write_text(
        f"""
paths:
  artifact_root: {artifact_root}
  phase4_root: {phase4_root}
  lane_d_root: {lane_d_root}
evidence_scope: unit_phase5
x1_logit_probe:
  max_probe_rows: null
  requested_roles: [desc_end, box_start, pre_x1, post_x1, pre_y1]
  layer_groups: [middle, late]
  poor_rank_threshold: 100
""".lstrip(),
        encoding="utf-8",
    )


def test_load_phase5_config_and_dry_run_plan(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase4_root = tmp_path / "phase4"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(phase4_root / "instance_attention_binding" / "rows.jsonl", [])
    _write_jsonl(lane_d_root / "probe_rows.jsonl", [])
    config_path = tmp_path / "phase5.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase4_root=phase4_root, lane_d_root=lane_d_root)

    config = load_phase5_config(config_path)
    plan = build_phase5_dry_run_plan(config, stages=("x1_logit_binding_probe", "report"))

    assert config.paths.artifact_root == artifact_root
    assert plan["phase4_rows_exists"] is True
    assert plan["probe_rows_exists"] is True
    assert plan["requested_roles"] == ["desc_end", "box_start", "pre_x1", "post_x1", "pre_y1"]
    assert plan["layer_groups"] == ["middle", "late"]


def test_materialize_x1_logit_binding_probe(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase4_root = tmp_path / "phase4"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(
        phase4_root / "instance_attention_binding" / "rows.jsonl",
        [
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "mechanism_bucket": "target_dependent",
                "valid_paired_row": True,
                "target_minus_competitor_instance_attention": 0.4,
                "target_attention_mass": 0.5,
                "competitor_attention_mass": 0.1,
            },
            {
                "case_id": "case-b",
                "rescue_tier": "desc_x1",
                "mechanism_bucket": "competitor_dependent",
                "valid_paired_row": True,
                "target_minus_competitor_instance_attention": 0.2,
                "target_attention_mass": 0.3,
                "competitor_attention_mass": 0.1,
            },
        ],
    )
    _write_jsonl(
        lane_d_root / "probe_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "role": "desc_end",
                "layer_group": "middle",
                "layer": 12,
                "x1_logit_lens_available": True,
                "x1_logit_lens_rank": 5,
                "x1_logit_lens_target_minus_top1": -0.2,
                "x1_logit_lens_top1_bin": 111,
                "target_x1_bin": 100,
                "x1_target_rank": 20,
                "x1_top_peak_attribution": "same_desc_competitor_gt_object",
            },
            {
                "case_id": "case-a",
                "role": "pre_x1",
                "layer_group": "late",
                "layer": 17,
                "x1_logit_lens_available": True,
                "x1_logit_lens_rank": 150,
                "x1_logit_lens_target_minus_top1": -1.5,
                "x1_logit_lens_top1_bin": 900,
                "target_x1_bin": 100,
                "x1_target_rank": 20,
                "x1_top_peak_attribution": "same_desc_competitor_gt_object",
            },
            {
                "case_id": "case-b",
                "role": "post_x1",
                "layer_group": "late",
                "layer": 17,
                "x1_logit_lens_available": False,
                "x1_logit_lens_rank": None,
                "x1_logit_lens_target_minus_top1": None,
            },
            {
                "case_id": "case-c",
                "role": "desc_end",
                "layer_group": "middle",
                "layer": 12,
                "x1_logit_lens_available": True,
                "x1_logit_lens_rank": 1,
                "x1_logit_lens_target_minus_top1": 0.0,
            },
        ],
    )
    config_path = tmp_path / "phase5.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase4_root=phase4_root, lane_d_root=lane_d_root)

    summary = materialize_x1_logit_binding_probe(load_phase5_config(config_path))

    rows = [
        json.loads(line)
        for line in (artifact_root / "x1_logit_binding_probe" / "rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert summary["row_count"] == 3
    assert summary["available_logit_rows"] == 2
    assert summary["unique_available_probe_rows"] == 2
    assert summary["joined_case_count"] == 2
    assert summary["missing_requested_roles"] == ["box_start", "post_x1", "pre_y1"]
    assert rows[0]["case_id"] == "case-a"
    assert rows[0]["mechanism_bucket"] == "target_dependent"
    assert rows[0]["target_attention_positive"] is True
    assert rows[1]["poor_x1_logit_rank"] is True
    assert rows[1]["positive_attention_but_poor_x1_logit"] is True
    assert summary["critical_slice"]["positive_attention_but_poor_x1_logit_rows"] == 2
    assert summary["bucket_summaries"]["target_dependent"]["available_rows"] == 2
    assert summary["unique_role_layer_summaries"]["desc_end::middle"]["available_rows"] == 1
    assert (artifact_root / "x1_logit_binding_probe" / "summary.json").exists()
    assert (artifact_root / "x1_logit_binding_probe" / "report.md").exists()


def test_materialize_coord_slot_logit_binding_probe(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase4_root = tmp_path / "phase4"
    lane_d_root = tmp_path / "lane_d"
    _write_jsonl(
        phase4_root / "instance_attention_binding" / "rows.jsonl",
        [
            {
                "case_id": "case-a",
                "rescue_tier": "desc_x1",
                "mechanism_bucket": "target_dependent",
                "valid_paired_row": True,
                "target_minus_competitor_instance_attention": 0.4,
            }
        ],
    )
    _write_jsonl(
        lane_d_root / "probe_rows.jsonl",
        [
            {
                "case_id": "case-a",
                "role": "pre_x1",
                "layer_group": "middle",
                "layer": 12,
                "coord_slot_logit_lens_available": True,
                "coord_slot_logit_lens_target_slot": "x1",
                "coord_slot_logit_lens_target_bin": 100,
                "coord_slot_logit_lens_rank": 3,
                "coord_slot_logit_lens_target_minus_top1": -0.1,
                "coord_slot_logit_lens_top1_bin": 110,
            },
            {
                "case_id": "case-a",
                "role": "post_x1",
                "layer_group": "middle",
                "layer": 12,
                "coord_slot_logit_lens_available": True,
                "coord_slot_logit_lens_target_slot": "y1",
                "coord_slot_logit_lens_target_bin": 200,
                "coord_slot_logit_lens_rank": 1,
                "coord_slot_logit_lens_target_minus_top1": 0.0,
                "coord_slot_logit_lens_top1_bin": 200,
            },
            {
                "case_id": "case-a",
                "role": "post_y1",
                "layer_group": "late",
                "layer": 24,
                "coord_slot_logit_lens_available": False,
                "coord_slot_logit_lens_target_slot": "x2",
            },
        ],
    )
    config_path = tmp_path / "phase5.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase4_root=phase4_root, lane_d_root=lane_d_root)

    summary = materialize_coord_slot_logit_binding_probe(load_phase5_config(config_path))

    rows = [
        json.loads(line)
        for line in (artifact_root / "coord_slot_logit_binding_probe" / "rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert summary["row_count"] == 3
    assert summary["available_logit_rows"] == 2
    assert summary["role_slot_summaries"]["post_x1::y1"]["mean_coord_slot_logit_lens_rank"] == 1.0
    assert summary["role_slot_summaries"]["pre_x1::x1"]["mean_coord_slot_logit_lens_rank"] == 3.0
    assert rows[1]["target_slot"] == "y1"
    assert rows[1]["poor_coord_slot_logit_rank"] is False
    assert (artifact_root / "coord_slot_logit_binding_probe" / "summary.json").exists()
    assert (artifact_root / "coord_slot_logit_binding_probe" / "report.md").exists()


def test_write_phase5_report(tmp_path: Path) -> None:
    artifact_root = tmp_path / "out"
    phase4_root = tmp_path / "phase4"
    lane_d_root = tmp_path / "lane_d"
    config_path = tmp_path / "phase5.yaml"
    _write_config(config_path, artifact_root=artifact_root, phase4_root=phase4_root, lane_d_root=lane_d_root)
    (artifact_root / "x1_logit_binding_probe").mkdir(parents=True)
    (artifact_root / "x1_logit_binding_probe" / "summary.json").write_text(
        json.dumps(
            {
                "stage": "x1_logit_binding_probe",
                "row_count": 0,
                "available_logit_rows": 0,
                "missing_requested_roles": [],
                "critical_slice": {},
                "bucket_summaries": {},
            }
        ),
        encoding="utf-8",
    )

    report = write_phase5_report(load_phase5_config(config_path))

    assert report == artifact_root / "report.md"
    assert (artifact_root / "summary.json").exists()
    assert "Phase 5" in report.read_text(encoding="utf-8")
