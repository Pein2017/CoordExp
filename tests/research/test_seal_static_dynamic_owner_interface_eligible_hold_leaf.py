from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.research import seal_static_dynamic_owner_interface_eligible_hold_leaf as sealer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_attempt(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def test_build_real_s_ordinal11_hold_leaf_is_exact_and_non_scored(tmp_path: Path) -> None:
    receipt_path = tmp_path / "eligible_hold_leaf.receipt.json"
    leaf, inputs = sealer.build_leaf_document(receipt_path=receipt_path)

    assert leaf["kind"] == "eligible_pre_actuator_technical_hold"
    assert leaf["event_binding"] == {
        "checkpoint": "S",
        "cohort_eligibility": "eligible_verified_pair",
        "cohort_sha256": sealer.EXPECTED_COHORT_SHA256,
        "event_id": "gt:5001:15",
        "expected_h0_exact_prefix_sha256": sealer.EXPECTED_PREFIX_SHA256,
        "geometry_disposition": "eligible_verified_pair_regions",
        "geometry_status": "available",
        "image_id": 5001,
        "natural_boundary": 1,
        "ordinal": 11,
        "source_panel_object_index": 15,
    }
    assert leaf["classification"] == {
        "actuators_called": None,
        "complete_non_scored": False,
        "matrix_status": "eligible_pre_actuator_hold",
        "scored": False,
    }
    execution = leaf["execution_evidence"]
    assert execution["receipt_bearing_actuator_cell_count"] == 0
    assert execution["receipt_bearing_scientific_cell_count"] == 0
    assert execution["actual_actuator_invocation_count"] is None
    assert execution["actual_model_forward_count"] is None
    assert execution["persisted_failure_log"] == {
        "path": None,
        "sha256": None,
        "status": "unavailable",
    }
    assert [attempt["role"] for attempt in leaf["attempt_lineage"]] == ["initial", "repair1"]
    assert all(attempt["cause"]["evidence_level"] == "lead_observed_unattested" for attempt in leaf["attempt_lineage"])
    assert all(attempt["cause"]["verbatim_stderr"] is None for attempt in leaf["attempt_lineage"])
    matrix = leaf["matrix_dispositions"]
    assert set(matrix["p1"]["cells"]) == set(sealer.P1_CELLS)
    assert set(matrix["p2"]["cells"]) == set(sealer.P2_CELLS)
    assert set(matrix["p3"]["cells"]) == set(sealer.P3_CELLS)
    assert set(matrix["p4"]["cells"]) == set(sealer.P4_CELLS)
    assert [len(matrix[stage]["cells"]) for stage in ("p1", "p2", "p3", "p4")] == [16, 14, 8, 3]
    assert all(
        cell == sealer.invalid_cell_disposition()
        for stage in matrix.values()
        for cell in stage["cells"].values()
    )
    assert leaf["self_sha256"] == sealer.document_self_sha256(leaf)
    assert any(item["role"] == "attempt:initial:runtime_identity" for item in inputs)


def test_materialize_real_leaf_and_receipt_are_self_verifying(tmp_path: Path) -> None:
    output = tmp_path / "sealed"
    result = sealer.seal(output)
    leaf_path = output / sealer.LEAF_FILENAME
    receipt_path = output / sealer.RECEIPT_FILENAME

    assert result["leaf_path"] == str(leaf_path.resolve())
    assert result["receipt_path"] == str(receipt_path.resolve())
    leaf = json.loads(leaf_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    sealer.validate_leaf_document(leaf)
    sealer.validate_receipt_document(receipt, leaf_path=leaf_path, leaf=leaf)
    assert receipt["leaf_sha256"] == _sha256(leaf_path)
    assert receipt["self_sha256"] == sealer.document_self_sha256(receipt)
    with pytest.raises(FileExistsError, match="output collision"):
        sealer.seal(output)


def test_attempt_census_rejects_any_extra_entry(tmp_path: Path) -> None:
    copied = _copy_attempt(sealer.INITIAL_ATTEMPT_DIR, tmp_path / "initial")
    (copied / "invented.log").write_text("not historical evidence\n", encoding="utf-8")
    with pytest.raises(sealer.HoldSealError, match="exactly runtime_identity.json"):
        sealer.validate_attempt_directory(
            copied,
            expected_directory=copied,
            spec=sealer.INITIAL_ATTEMPT,
        )


def test_attempt_rejects_altered_hash_and_semantically_invalid_attestation(tmp_path: Path) -> None:
    altered = _copy_attempt(sealer.INITIAL_ATTEMPT_DIR, tmp_path / "altered")
    identity_path = altered / "runtime_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["checkpoint"] = "A"
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(sealer.HoldSealError, match="SHA-256"):
        sealer.validate_attempt_directory(
            altered,
            expected_directory=altered,
            spec=sealer.INITIAL_ATTEMPT,
        )

    invalid = _copy_attempt(sealer.INITIAL_ATTEMPT_DIR, tmp_path / "invalid-attestation")
    invalid_path = invalid / "runtime_identity.json"
    payload = json.loads(invalid_path.read_text(encoding="utf-8"))
    payload["runtime_attestation"]["passed"] = False
    invalid_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    invalid_hash = _sha256(invalid_path)
    invalid_size = invalid_path.stat().st_size
    custom_spec = sealer.AttemptSpec(
        **{
            **sealer.INITIAL_ATTEMPT.__dict__,
            "runtime_identity_sha256": invalid_hash,
            "runtime_identity_size": invalid_size,
            "file_census_sha256": sealer.sha256_json(
                [
                    {
                        "relative_path": "runtime_identity.json",
                        "sha256": invalid_hash,
                        "size_bytes": invalid_size,
                    }
                ]
            ),
        }
    )
    with pytest.raises(sealer.HoldSealError, match="attestation is not validated"):
        sealer.validate_attempt_directory(
            invalid,
            expected_directory=invalid,
            spec=custom_spec,
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda leaf: leaf["classification"].update({"scored": True}), "classification"),
        (lambda leaf: leaf["classification"].update({"complete_non_scored": True}), "classification"),
        (lambda leaf: leaf["event_binding"].update({"cohort_eligibility": "no_verified_B"}), "event binding"),
        (
            lambda leaf: leaf["matrix_dispositions"]["p1"]["cells"]["K00"].update(
                {"scientific_observation": 0.0}
            ),
            "administrative cell",
        ),
        (
            lambda leaf: leaf["execution_evidence"].update({"actual_model_forward_count": 0}),
            "actual execution counts",
        ),
    ],
)
def test_leaf_validation_rejects_scored_confusion_or_numeric_observation(
    tmp_path: Path,
    mutate: object,
    match: str,
) -> None:
    leaf, _inputs = sealer.build_leaf_document(receipt_path=tmp_path / "receipt.json")
    mutate(leaf)  # type: ignore[operator]
    leaf["self_sha256"] = sealer.document_self_sha256(leaf)
    with pytest.raises(sealer.HoldSealError, match=match):
        sealer.validate_leaf_document(leaf)


def test_frozen_contract_hashes_fail_closed(tmp_path: Path) -> None:
    changed = tmp_path / "unit.md"
    changed.write_bytes(sealer.UNIT_PATH.read_bytes() + b"\nchanged\n")
    with pytest.raises(sealer.HoldSealError, match="frozen unit SHA-256"):
        sealer.validate_frozen_contract_file(
            changed,
            expected_sha256=sealer.EXPECTED_UNIT_SHA256,
            label="frozen unit",
        )
