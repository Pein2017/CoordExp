from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis.post_x1_instance_basin_tomography.jsonl import write_jsonl
from src.analysis.post_x1_instance_basin_tomography.status import evaluate_status


ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
    "et_rmp_ce_ckpt3664",
)


def _contracts() -> dict[str, dict[str, str]]:
    return {
        "fullobj_random_pure_ce_ckpt3668": {
            "template_contract_id": "compact_full_no_newline_native_v1",
            "detection_sequence_format": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "row_separator": "none",
            "contract_provenance": "user_reported_training_contract",
        },
        "fullobj_sorted_pure_ce_ckpt3668": {
            "template_contract_id": "compact_full_no_newline_native_v1",
            "detection_sequence_format": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "row_separator": "none",
            "contract_provenance": "user_reported_training_contract",
        },
        "et_rmp_ce_ckpt3664": {
            "template_contract_id": "compact_full_newline_native_v1",
            "detection_sequence_format": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "row_separator": "newline",
            "contract_provenance": "legacy_compact_full_default_inferred",
        },
    }


def _slot_row(role: str, *, case_id: str = "case-0", prefix_state_id: str = "state-0") -> dict[str, Any]:
    return {
        "artifact_schema_version": "a3.3.v1",
        "row_schema_version": "slot_posterior.v1",
        "case_id": case_id,
        "prefix_state_id": prefix_state_id,
        "checkpoint_role": role,
        "comparison_role": "reference_anchor" if role == "et_rmp_ce_ckpt3664" else "clean_pair",
        "controlled_comparison_group": (
            "reference_anchor_not_controlled"
            if role == "et_rmp_ce_ckpt3664"
            else "pure_ce_sorted_vs_random_no_newline"
        ),
        "slot": "y1",
        "winner_bucket": "target_instance",
        "primary_basin_label_source": "same_desc_gt_instances",
        "template_contract": _contracts()[role],
        "system_prompt_sha256": "a" * 64,
        "user_prompt_sha256": "b" * 64,
        "template_prompt_hash": "c" * 64,
        "assistant_prefix_sha256": "d" * 64,
        "forced_prompt_sha256": "e" * 64,
        "runtime_kind": "mock_cpu_slot_posterior_v1",
        "mock_runtime": True,
        "boundary_extreme_flag": False,
    }


def _downstream_row(role: str, schema: str, *, case_id: str = "case-0") -> dict[str, Any]:
    return {
        "artifact_schema_version": "a3.3.v1",
        "row_schema_version": schema,
        "case_id": case_id,
        "checkpoint_role": role,
        "template_contract": _contracts()[role],
        "primary_basin_label_source": "same_desc_gt_instances",
    }


def _write_complete_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config_resolved.json").write_text(
        json.dumps({"artifact_schema_version": "a3.3.v1", "config_path": "config.yaml"}),
        encoding="utf-8",
    )
    (root / "template_contracts.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": "a3.3.v1",
                "checkpoint_roles": list(ROLES),
                "template_contracts": _contracts(),
            },
        ),
        encoding="utf-8",
    )
    (root / "data_root_audit.json").write_text(
        json.dumps({"artifact_schema_version": "a3.3.v1", "status": "mock"}),
        encoding="utf-8",
    )
    write_jsonl(root / "case_universe.jsonl", [{"artifact_schema_version": "a3.3.v1", "case_id": "case-0"}])
    write_jsonl(root / "prefix_states.jsonl", [{"artifact_schema_version": "a3.3.v1", "prefix_state_id": "state-0"}])

    shard_root = root / "slot_posterior_shards"
    slot_rows = [_slot_row(role) for role in ROLES]
    for shard_id in range(8):
        rows = slot_rows if shard_id == 0 else []
        write_jsonl(shard_root / f"shard_{shard_id}.jsonl", rows)
    write_jsonl(
        root / "slot_posterior_shard_summaries.jsonl",
        [
            {"artifact_schema_version": "a3.3.v1", "shard_id": shard_id, "row_count": 3 if shard_id == 0 else 0}
            for shard_id in range(8)
        ],
    )
    write_jsonl(root / "slot_posterior_rows.jsonl", slot_rows)
    (root / "merge_manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": "a3.3.v1",
                "input_shards": [
                    {
                        "path": f"slot_posterior_shards/shard_{shard_id}.jsonl",
                        "sha256": "f" * 64,
                        "row_count": 3 if shard_id == 0 else 0,
                    }
                    for shard_id in range(8)
                ],
                "merged_row_count": 3,
                "merge_timestamp": "2026-06-05T00:00:00Z",
            },
        ),
        encoding="utf-8",
    )
    for filename, schema in (
        ("trajectory_rows.jsonl", "trajectory.v1"),
        ("basin_attraction_matrix.jsonl", "basin_attraction_matrix.v1"),
        ("prefix_sensitivity_rows.jsonl", "prefix_sensitivity.v1"),
        ("greedy_continuation_rows.jsonl", "greedy_continuation.v1"),
    ):
        write_jsonl(root / filename, [_downstream_row(role, schema) for role in ROLES])
    (root / "summary.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": "a3.3.v1",
                "boundary_extreme_counts_by_checkpoint": {role: 0 for role in ROLES},
                "row_counts": {"slot_posterior_rows": 3},
                "template_contracts": _contracts(),
            },
        ),
        encoding="utf-8",
    )
    (root / "report.md").write_text(
        "# A3.3 Report\n\nThis mechanism study is not by final detector accuracy.\n\nreference_anchor template_objective_confounded_reference\n",
        encoding="utf-8",
    )
    gallery = root / "gallery"
    gallery.mkdir()
    (gallery / "index.md").write_text("# Gallery\n\nreference_anchor template_objective_confounded_reference\n", encoding="utf-8")
    (gallery / "gallery_summary.json").write_text(
        json.dumps({"artifact_schema_version": "a3.3.v1", "gallery_rows": 1}),
        encoding="utf-8",
    )


def test_status_rejects_missing_template_contracts(tmp_path: Path) -> None:
    status = evaluate_status(tmp_path)

    assert status["status"] == "incomplete"
    assert "template_contracts_present" in status["failed_gates"]


def test_status_accepts_complete_mock_artifact_tree(tmp_path: Path) -> None:
    _write_complete_artifacts(tmp_path)

    status = evaluate_status(tmp_path)

    assert status["status"] == "final_artifacts_present"
    assert status["final_artifacts_present"] is True
    assert status["failed_gates"] == []
    assert status["row_counts"]["slot_posterior_rows"] == 3


def test_status_rejects_shard_total_mismatch(tmp_path: Path) -> None:
    _write_complete_artifacts(tmp_path)
    write_jsonl(
        tmp_path / "slot_posterior_shard_summaries.jsonl",
        [{"artifact_schema_version": "a3.3.v1", "shard_id": shard_id, "row_count": 0} for shard_id in range(8)],
    )

    status = evaluate_status(tmp_path)

    assert status["status"] == "incomplete"
    assert "slot_shard_row_counts_match" in status["failed_gates"]


def test_status_rejects_role_template_contract_mismatch(tmp_path: Path) -> None:
    _write_complete_artifacts(tmp_path)
    rows = [_slot_row(role) for role in ROLES]
    rows[0]["template_contract"] = dict(rows[0]["template_contract"], row_separator="newline")
    write_jsonl(tmp_path / "slot_posterior_rows.jsonl", rows)
    write_jsonl(tmp_path / "slot_posterior_shards" / "shard_0.jsonl", rows)

    status = evaluate_status(tmp_path)

    assert "checkpoint_role_template_contracts_match" in status["failed_gates"]


def test_status_rejects_uncaveated_detector_ranking_language(tmp_path: Path) -> None:
    _write_complete_artifacts(tmp_path)
    (tmp_path / "report.md").write_text(
        "# Bad\n\nsorted is best and wins on AP.\n",
        encoding="utf-8",
    )

    status = evaluate_status(tmp_path)

    assert "report_language_no_uncaveated_detector_ranking" in status["failed_gates"]
