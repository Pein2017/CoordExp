from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.candidate_field_cardinality_tomography.artifacts import (
    validate_manifest,
    validate_probe_row_against_plan,
    validate_required_fields,
    write_manifest,
)


def test_base_row_requires_dataset_manifest_identity(minimal_base_row) -> None:
    row = dict(minimal_base_row)
    row.pop("dataset_manifest_sha256")

    with pytest.raises(ValueError, match="dataset_manifest_sha256"):
        validate_required_fields(row)


def test_probe_row_must_reference_matching_probe_plan(minimal_base_row) -> None:
    row = dict(minimal_base_row)
    row.update(
        {
            "probe_plan_row_id": "pp-1",
            "sampling_policy_id": "stratified_v1",
            "sampling_policy_sha256": "a" * 64,
            "strata_key": "val|headline",
            "shard_id": 0,
        }
    )
    plan = {
        "probe_plan_row_id": "pp-1",
        "probe_sampled": True,
        "sampling_policy_id": "stratified_v1",
        "sampling_policy_sha256": "b" * 64,
        "strata_key": "val|headline",
        "planned_shard_id": 0,
    }

    with pytest.raises(ValueError, match="sampling_policy_sha256"):
        validate_probe_row_against_plan(row, plan)


def test_manifest_validation_detects_missing_and_changed_files(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    first = root / "first.json"
    second = root / "nested" / "second.jsonl"
    second.parent.mkdir()
    first.write_text('{"ok": true}\n', encoding="utf-8")
    second.write_text('{"row": 1}\n', encoding="utf-8")

    write_manifest(root, [first, second], {"stage": "test"})
    result = validate_manifest(root)

    assert result["validation_status"] == "ok"
    assert result["file_count"] == 2

    second.write_text('{"row": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_manifest(root)

    second.unlink()
    with pytest.raises(FileNotFoundError, match="nested/second.jsonl"):
        validate_manifest(root)
