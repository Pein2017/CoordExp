from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .artifacts import write_json, write_jsonl


def assign_primary_bucket(evidence: Mapping[str, object]) -> str:
    if evidence.get("policy_mismatch") or evidence.get("schema_mismatch"):
        return "invalid_or_uninterpretable"
    if evidence.get("coordinate_channel_low_mass_or_leakage"):
        return "unassigned_or_inconclusive"
    if evidence.get("sensitivity_unstable"):
        return "unassigned_or_inconclusive"
    if evidence.get("partial_label_ambiguous"):
        return "unassigned_or_inconclusive"
    if evidence.get("projection_collision_unresolved"):
        return "unassigned_or_inconclusive"
    if not evidence.get("candidate_modes_cover_target"):
        return "A1_cardinality_collapse"
    residual_viable = bool(evidence.get("residual_continuation_viable"))
    eos_dominant = bool(evidence.get("eos_dominant"))
    if not residual_viable or eos_dominant:
        return "A2_readout_or_coverage_failure"
    if evidence.get("teacher_tail_loses"):
        return "A3b_tail_representation_failure"
    if evidence.get("greedy_tail_fails"):
        return "A3a_decode_basin_failure"
    return "unassigned_or_inconclusive"


def materialize_phase_a_taxonomy_rows(
    artifact_root: Path,
    x1_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows = [_taxonomy_row_from_x1(row) for row in x1_rows]
    bucket_counts = Counter(str(row["primary_bucket"]) for row in output_rows)
    summary = {
        "stage": "taxonomy",
        "taxonomy_rows": len(output_rows),
        "primary_bucket_counts": dict(sorted(bucket_counts.items())),
    }
    write_jsonl(artifact_root / "phase_a_case_taxonomy_rows.jsonl", output_rows)
    write_json(artifact_root / "taxonomy_summary.json", summary)
    return output_rows, summary


def _taxonomy_row_from_x1(row: Mapping[str, Any]) -> dict[str, Any]:
    same_desc_count = int(row.get("same_desc_gt_count_annotated") or 0)
    covered_count = int(row.get("gt_instance_coverage_count") or 0)
    probe_ok = row.get("probe_status") == "ok"
    coord_vocab_mass = float(row.get("coord_vocab_mass") or 0.0)
    evidence_flags = {
        "schema_mismatch": False,
        "policy_mismatch": False,
        "candidate_modes_cover_target": probe_ok and covered_count >= same_desc_count,
        "coordinate_channel_low_mass_or_leakage": probe_ok and coord_vocab_mass < 0.05,
        "projection_collision_unresolved": bool(row.get("x1_projection_collision")),
        "partial_label_ambiguous": False,
        "sensitivity_unstable": False,
        "residual_continuation_viable": True,
        "eos_dominant": False,
        "teacher_tail_loses": False,
        "greedy_tail_fails": False,
    }
    if not probe_ok:
        evidence_flags["schema_mismatch"] = True
    primary_bucket = assign_primary_bucket(evidence_flags)
    return {
        "schema_version": row.get("schema_version"),
        "project_id": row.get("project_id"),
        "phase_id": row.get("phase_id"),
        "run_id": row.get("run_id"),
        "checkpoint_id": row.get("checkpoint_id"),
        "case_id": row.get("case_id"),
        "case_index_row_id": row.get("case_index_row_id"),
        "split": row.get("split"),
        "pool_role": row.get("pool_role"),
        "source_dataset_jsonl": row.get("source_dataset_jsonl"),
        "dataset_manifest_id": row.get("dataset_manifest_id"),
        "dataset_manifest_sha256": row.get("dataset_manifest_sha256"),
        "fn_rescue_overlay_membership": row.get("fn_rescue_overlay_membership"),
        "desc_text_canonical": row.get("desc_text_canonical"),
        "same_desc_gt_count_annotated": same_desc_count,
        "prefix_condition": row.get("prefix_condition"),
        "posterior_snapshot_id": row.get("posterior_snapshot_id"),
        "primary_bucket": primary_bucket,
        "evidence_flags": evidence_flags,
        "evidence_metrics": {
            "same_desc_gt_count_annotated": same_desc_count,
            "gt_instance_coverage_count": covered_count,
            "coverage_fraction": covered_count / same_desc_count if same_desc_count else 0.0,
            "coord_vocab_mass": coord_vocab_mass,
            "merged_peak_count": row.get("merged_peak_count"),
            "x1_top1_bin": row.get("x1_top1_bin"),
            "x1_target_rank": row.get("x1_target_rank"),
        },
        "taxonomy_policy_id": "phase_a_x1_posterior_v1",
    }
