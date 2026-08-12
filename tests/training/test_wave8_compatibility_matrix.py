from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/probes/coordexp_swift/wave8_compatibility_matrix.py"
)


@pytest.fixture
def matrix() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wave8_compatibility_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _write_signed(
    path: Path,
    payload: dict[str, object],
    *,
    hash_field: str = "receipt_payload_sha256",
) -> dict[str, object]:
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    signed = {
        **unsigned,
        hash_field: hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(signed) + b"\n")
    return signed


def _input_artifacts(matrix: ModuleType, root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, contract in matrix.INPUT_CONTRACTS.items():
        payload: dict[str, object] = {
            "schema": contract.schema,
            contract.status_field: contract.passed_status,
            "identity": {
                "name": name,
                "digest": hashlib.sha256(name.encode()).hexdigest(),
            },
        }
        path = root / "inputs" / f"{name}.json"
        _write_signed(path, payload, hash_field=contract.hash_field)
        result[name] = {"path": path, "schema": contract.schema}
    return result


def _prepare(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Path]]:
    inputs = _input_artifacts(matrix, tmp_path)
    cell_targets = {
        name: tmp_path / "cells" / f"{name}.json" for name in matrix.CELL_NAMES
    }
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    wave7_root = tmp_path / "wave7"
    wave7_root.mkdir()
    monkeypatch.setattr(
        matrix,
        "_validate_wave7_evidence",
        lambda **_: {"status": "passed", "validator": "fixture"},
    )
    monkeypatch.setattr(
        matrix,
        "_validate_native_runtime_evidence",
        lambda _: {"status": "passed", "validator": "fixture"},
    )
    matrix.prepare_plan(
        plan_path=plan_path,
        receipt_path=receipt_path,
        wave7_root=wave7_root,
        inputs=inputs,
        cell_targets=cell_targets,
    )
    return plan_path, receipt_path, cell_targets


def _write_cell(
    matrix: ModuleType,
    *,
    plan: dict[str, object],
    name: str,
    path: Path,
) -> None:
    _write_signed(
        path,
        {
            "schema": matrix.CELL_SCHEMAS[name],
            "status": "passed",
            "cell": name,
            "plan_payload_sha256": plan["plan_payload_sha256"],
            "baseline_sha256": matrix.PINNED_BASELINE_SHA256,
            "identity_bundle_sha256": plan["identity_bundle_sha256"],
            "checks": {check: True for check in matrix.CELL_CHECKS[name]},
            "claim_scope": matrix.NONPERFORMANCE_CLAIM,
        },
    )


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"schema":"x","schema":"y"}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
    ],
)
def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(
    matrix: ModuleType, encoded: bytes
) -> None:
    with pytest.raises(matrix.Wave8MatrixError, match="strict JSON"):
        matrix.load_strict_json_bytes(encoded, owner="fixture")


def test_prepare_plan_binds_every_input_and_is_absent_only(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _, _ = _prepare(matrix, tmp_path, monkeypatch)

    plan = matrix.validate_plan(plan_path)

    assert plan["schema"] == matrix.PLAN_SCHEMA
    assert plan["status"] == "prepared"
    assert set(plan["inputs"]) == set(matrix.INPUT_CONTRACTS)
    for binding in plan["inputs"].values():
        assert set(binding) == {"path", "file_sha256", "payload_sha256", "schema"}
    assert plan["route"] == ["wave7", "wave8", "wave9"]
    assert plan["runtime_baseline"] == {
        "schema_version": 3,
        "baseline_sha256": matrix.PINNED_BASELINE_SHA256,
        "attention_backend": "flash_attention_2",
    }
    assert plan["claim_scope"] == matrix.NONPERFORMANCE_CLAIM
    with pytest.raises(matrix.Wave8MatrixError, match="already exists"):
        matrix.prepare_plan(
            plan_path=plan_path,
            receipt_path=Path(plan["receipt_target"]),
            wave7_root=Path(plan["wave7_root"]),
            inputs={
                name: {"path": binding["path"], "schema": binding["schema"]}
                for name, binding in plan["inputs"].items()
            },
            cell_targets={
                name: Path(spec["path"]) for name, spec in plan["cells"].items()
            },
        )


def test_validate_plan_rejects_noncanonical_reencoding(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _, _ = _prepare(matrix, tmp_path, monkeypatch)
    plan = json.loads(plan_path.read_text())
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    with pytest.raises(matrix.Wave8MatrixError, match="canonical"):
        matrix.validate_plan(plan_path)


def test_prepare_rejects_symlinked_input(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _input_artifacts(matrix, tmp_path)
    target = Path(inputs["config_identity"]["path"])
    alias = tmp_path / "config-identity-link.json"
    alias.symlink_to(target)
    inputs["config_identity"]["path"] = alias
    monkeypatch.setattr(matrix, "_validate_wave7_evidence", lambda **_: {})
    monkeypatch.setattr(matrix, "_validate_native_runtime_evidence", lambda _: {})

    with pytest.raises(matrix.Wave8MatrixError, match="non-symlink"):
        matrix.prepare_plan(
            plan_path=tmp_path / "plan.json",
            receipt_path=tmp_path / "receipt.json",
            wave7_root=tmp_path,
            inputs=inputs,
            cell_targets={
                name: tmp_path / "cells" / f"{name}.json" for name in matrix.CELL_NAMES
            },
        )


def test_aggregate_missing_cells_is_blocked_not_passed(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, receipt_path, _ = _prepare(matrix, tmp_path, monkeypatch)

    receipt = matrix.aggregate(plan_path)

    assert receipt["status"] == "blocked"
    assert receipt["passed"] is False
    assert {row["status"] for row in receipt["cells"]} == {"missing"}
    assert matrix.validate_receipt(receipt_path)["status"] == "blocked"


def test_failed_receipt_remains_valid_evidence_after_bound_input_drift(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, receipt_path, _ = _prepare(matrix, tmp_path, monkeypatch)
    plan = matrix.validate_plan(plan_path)
    config_path = Path(plan["inputs"]["config_identity"]["path"])
    payload = json.loads(config_path.read_text())
    payload["identity"]["digest"] = "f" * 64
    _write_signed(
        config_path,
        payload,
        hash_field=matrix.INPUT_CONTRACTS["config_identity"].hash_field,
    )

    receipt = matrix.aggregate(plan_path)

    assert receipt["status"] == "failed"
    assert matrix.validate_receipt(receipt_path)["status"] == "failed"


def test_aggregate_passes_only_with_all_five_bound_cells_and_wave7(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, receipt_path, cell_targets = _prepare(matrix, tmp_path, monkeypatch)
    plan = matrix.validate_plan(plan_path)
    for name, path in cell_targets.items():
        _write_cell(matrix, plan=plan, name=name, path=path)

    receipt = matrix.aggregate(plan_path)

    assert receipt["status"] == "passed"
    assert receipt["passed"] is True
    assert [row["cell"] for row in receipt["cells"]] == list(matrix.CELL_NAMES)
    assert {row["status"] for row in receipt["cells"]} == {"passed"}
    assert receipt["claim_scope"] == matrix.NONPERFORMANCE_CLAIM
    assert matrix.validate_receipt(receipt_path)["passed"] is True


def test_aggregate_mutated_cell_is_failed_not_passed(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _, cell_targets = _prepare(matrix, tmp_path, monkeypatch)
    plan = matrix.validate_plan(plan_path)
    for name, path in cell_targets.items():
        _write_cell(matrix, plan=plan, name=name, path=path)
    mutated = cell_targets["protected_loss"]
    mutated.write_bytes(mutated.read_bytes().replace(b'"passed"', b'"failed"'))

    receipt = matrix.aggregate(plan_path)

    assert receipt["status"] == "failed"
    assert receipt["passed"] is False
    assert any(
        row["cell"] == "protected_loss" and row["status"] == "failed"
        for row in receipt["cells"]
    )


@pytest.mark.parametrize(
    ("schema", "status"),
    [
        ("coordexp-swift-wave7-exact-resume-sequence-receipt-v4", "passed"),
        ("coordexp-swift-wave7-exact-resume-sequence-receipt-v5", "failed"),
    ],
)
def test_prepare_rejects_legacy_or_failed_wave7_sequence(
    matrix: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    status: str,
) -> None:
    inputs = _input_artifacts(matrix, tmp_path)
    sequence = Path(inputs["wave7_sequence"]["path"])
    _write_signed(
        sequence,
        {"schema": schema, "status": status, "identity": {"name": "wave7"}},
    )
    monkeypatch.setattr(matrix, "_validate_wave7_evidence", lambda **_: {})

    with pytest.raises(matrix.Wave8MatrixError, match="Wave 7"):
        matrix.prepare_plan(
            plan_path=tmp_path / "plan.json",
            receipt_path=tmp_path / "receipt.json",
            wave7_root=tmp_path,
            inputs=inputs,
            cell_targets={
                name: tmp_path / "cells" / f"{name}.json" for name in matrix.CELL_NAMES
            },
        )
