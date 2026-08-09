from __future__ import annotations

import json
from pathlib import Path

from scripts.research import build_natural_boundary_owner_admission_census as census


def test_real_inputs_materialize_exact_784_rows_and_distinct_scopes(tmp_path: Path) -> None:
    result = census.build_census(
        output=tmp_path / "census.json",
        records_output=tmp_path / "census.records.jsonl",
        receipt_output=tmp_path / "census.receipt.json",
    )
    document = result["census"]
    census.validate_census(document)
    assert len(document["rows"]) == 784
    assert len(document["support_completion_candidates"]) == 200
    assert document["assessment_scope"] == {
        "S": "expanded_in_unit",
        "A3": "frozen_at_prior_32",
    }
    assert document["summary"]["S"]["target_B_support_assessed"] == 20
    assert document["summary"]["S"]["calibration_control_assessed"] == 172
    assert document["summary"]["S"]["support_unassessed"] == 200
    assert document["summary"]["S"]["eligible_verified_pair"] == 1
    assert document["summary"]["A"]["target_B_support_assessed"] == 22
    assert document["summary"]["A"]["calibration_control_assessed"] == 149
    assert document["summary"]["A"]["support_unassessed"] == 221
    assert document["summary"]["A"]["eligible_verified_pair"] == 1
    assert document["summary"]["S"]["target_B_support_assessed"] != document["summary"]["S"]["calibration_control_assessed"]
    assert document["self_sha256"] == census.document_self_sha256(document)
    assert json.loads((tmp_path / "census.receipt.json").read_text())["row_count"] == 784


def test_support_completion_candidates_are_unique_and_native_fn_only() -> None:
    document = census.build_census()["census"]
    candidates = document["support_completion_candidates"]
    assert len({item["gt_owner_id"] for item in candidates}) == 200
    assert all(item["checkpoint"] == "S" for item in candidates)
    rows = {(row["checkpoint"], row["gt_owner_id"]): row for row in document["rows"]}
    assert all(rows[("S", item["gt_owner_id"])]["native_fn"] for item in candidates)
    assert all(rows[("S", item["gt_owner_id"])]["disposition"] == "support_unassessed" for item in candidates)


def test_geometry_hash_binds_final_disposition_and_upper_bound_label() -> None:
    document = census.build_census()["census"]
    for row in document["rows"]:
        assert row["eligible_except_support_semantics"] == (
            "support_state_dependent_upper_bound_before_verified_support"
        )
        geometry = row.get("geometry")
        if geometry is None:
            continue
        payload = {key: value for key, value in geometry.items() if key != "geometry_sha256"}
        assert geometry["geometry_sha256"] == census.sha256_json(payload)
        assert {"status", "launch_eligible", "reason"} <= set(payload)
