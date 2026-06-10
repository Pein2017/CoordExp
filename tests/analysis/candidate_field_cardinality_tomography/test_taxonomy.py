from __future__ import annotations

import json
from pathlib import Path

from src.analysis.candidate_field_cardinality_tomography.artifacts import read_jsonl
from src.analysis.candidate_field_cardinality_tomography.taxonomy import (
    assign_primary_bucket,
    materialize_phase_a_taxonomy_rows,
)


def test_eos_high_routes_tail_failure_to_a2_not_a3() -> None:
    bucket = assign_primary_bucket(
        {
            "candidate_modes_cover_target": True,
            "residual_continuation_viable": False,
            "eos_dominant": True,
            "teacher_tail_loses": True,
        }
    )

    assert bucket == "A2_readout_or_coverage_failure"


def test_viable_residual_tail_loss_routes_to_a3b() -> None:
    bucket = assign_primary_bucket(
        {
            "candidate_modes_cover_target": True,
            "residual_continuation_viable": True,
            "eos_dominant": False,
            "teacher_tail_loses": True,
        }
    )

    assert bucket == "A3b_tail_representation_failure"


def test_taxonomy_materialization_assigns_x1_cardinality_evidence(
    tmp_path: Path,
    minimal_base_row: dict[str, object],
) -> None:
    rows = [
        {
            **minimal_base_row,
            "probe_status": "ok",
            "case_id": "case-collapse",
            "case_index_row_id": "case-collapse:gt0",
            "posterior_snapshot_id": "pp-000001:pre_x1",
            "same_desc_gt_count_annotated": 3,
            "gt_instance_coverage_count": 2,
            "coord_vocab_mass": 0.94,
            "x1_projection_collision": False,
        },
        {
            **minimal_base_row,
            "probe_status": "ok",
            "case_id": "case-collision",
            "case_index_row_id": "case-collision:gt0",
            "posterior_snapshot_id": "pp-000002:pre_x1",
            "same_desc_gt_count_annotated": 3,
            "gt_instance_coverage_count": 3,
            "coord_vocab_mass": 0.93,
            "x1_projection_collision": True,
        },
    ]
    root = tmp_path / "artifact"

    output_rows, summary = materialize_phase_a_taxonomy_rows(root, rows)

    assert summary["taxonomy_rows"] == 2
    assert summary["primary_bucket_counts"]["A1_cardinality_collapse"] == 1
    assert summary["primary_bucket_counts"]["unassigned_or_inconclusive"] == 1
    assert output_rows[0]["primary_bucket"] == "A1_cardinality_collapse"
    assert output_rows[0]["evidence_flags"]["candidate_modes_cover_target"] is False
    assert output_rows[1]["evidence_flags"]["projection_collision_unresolved"] is True
    assert json.loads((root / "taxonomy_summary.json").read_text()) == summary
    assert len(read_jsonl(root / "phase_a_case_taxonomy_rows.jsonl")) == 2
