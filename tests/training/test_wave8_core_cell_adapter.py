from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/probes/coordexp_swift/wave8_core_cell_adapter.py"
)


@pytest.fixture
def adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wave8_core_cell_adapter", SCRIPT)
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


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload) + b"\n")
    return path


def _plan(
    adapter: ModuleType,
    tmp_path: Path,
    *,
    cell: str,
    wave7_paths: dict[str, Path] | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    target = tmp_path / "cells" / f"{cell}.json"
    target.parent.mkdir(parents=True)
    plan = {
        "plan_payload_sha256": "a" * 64,
        "identity_bundle_sha256": "b" * 64,
        "runtime_baseline": {
            "schema_version": 3,
            "baseline_sha256": adapter.matrix.PINNED_BASELINE_SHA256,
            "attention_backend": "flash_attention_2",
        },
        "claim_scope": adapter.matrix.NONPERFORMANCE_CLAIM,
        "wave7_root": str(tmp_path / "wave7"),
        "inputs": {
            name: {"path": str(path)} for name, path in (wave7_paths or {}).items()
        },
        "cells": {
            cell: {
                "path": str(target),
                "schema": adapter.matrix.CELL_SCHEMAS[cell],
            }
        },
    }
    plan_path = _write(tmp_path / "matrix-plan.json", {"fixture": True})
    return plan_path, plan, target


def _native_artifact(path: Path, schema: str) -> Path:
    payload: dict[str, object] = {"schema": schema, "fixture": True}
    if schema.endswith("packed-parity-receipt-v3"):
        payload["terminal_status"] = "passed"
    elif "receipt" in schema:
        payload["status"] = "passed"
    return _write(path, payload)


def _assert_exact_cell(
    adapter: ModuleType,
    *,
    receipt: dict[str, Any],
    plan: dict[str, Any],
    cell: str,
    target: Path,
) -> None:
    assert set(receipt) == {
        "schema",
        "status",
        "cell",
        "plan_payload_sha256",
        "baseline_sha256",
        "identity_bundle_sha256",
        "checks",
        "claim_scope",
        "receipt_payload_sha256",
    }
    assert receipt == json.loads(target.read_text())
    assert target.read_bytes() == _canonical(receipt) + b"\n"
    assert receipt["schema"] == adapter.matrix.CELL_SCHEMAS[cell]
    assert receipt["status"] == "passed"
    assert receipt["cell"] == cell
    assert receipt["plan_payload_sha256"] == plan["plan_payload_sha256"]
    assert receipt["baseline_sha256"] == adapter.matrix.PINNED_BASELINE_SHA256
    assert receipt["identity_bundle_sha256"] == plan["identity_bundle_sha256"]
    assert receipt["checks"] == {
        check: True for check in adapter.matrix.CELL_CHECKS[cell]
    }
    assert receipt["claim_scope"] == adapter.matrix.NONPERFORMANCE_CLAIM
    unsigned = dict(receipt)
    observed = unsigned.pop("receipt_payload_sha256")
    assert observed == hashlib.sha256(_canonical(unsigned)).hexdigest()


def test_packed_cell_reuses_wave2_plan_and_receipt_validators(
    adapter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = "packed_parity_all_layer_fa2"
    plan_path, plan, target = _plan(adapter, tmp_path, cell=cell)
    native_plan_path = _native_artifact(
        tmp_path / "wave2-plan.json", adapter.wave2.PARITY_PLAN_SCHEMA
    )
    native_receipt_path = _native_artifact(
        tmp_path / "wave2-receipt.json", adapter.wave2.PARITY_RECEIPT_SCHEMA
    )
    calls: list[tuple[str, object]] = []
    validated_native_plan = {"plan_sha256": "c" * 64}
    monkeypatch.setattr(adapter.matrix, "validate_plan", lambda _: plan)
    monkeypatch.setattr(
        adapter.wave2,
        "validate_parity_plan",
        lambda payload: calls.append(("plan", payload)) or validated_native_plan,
    )
    monkeypatch.setattr(
        adapter.wave2,
        "validate_parity_receipt",
        lambda payload, **kwargs: calls.append(("receipt", (payload, kwargs)))
        or payload,
    )

    receipt = adapter.publish_cell(
        plan_path=plan_path,
        cell=cell,
        native_receipts=[native_receipt_path, native_plan_path],
    )

    assert [name for name, _ in calls] == ["plan", "receipt"]
    _, (_, kwargs) = calls[1]
    assert kwargs == {
        "expected_plan": validated_native_plan,
        "expected_receipt_target": native_receipt_path,
    }
    _assert_exact_cell(
        adapter, receipt=receipt, plan=plan, cell=cell, target=target
    )


def test_protected_loss_cell_reuses_wave3_plan_and_receipt_validators(
    adapter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = "protected_loss"
    plan_path, plan, target = _plan(adapter, tmp_path, cell=cell)
    native_plan_path = _native_artifact(
        tmp_path / "wave3-plan.json", adapter.wave3.PLAN_SCHEMA
    )
    native_receipt_path = _native_artifact(
        tmp_path / "wave3-receipt.json", adapter.wave3.RECEIPT_SCHEMA
    )
    calls: list[tuple[str, object]] = []
    validated_native_plan = {"plan_sha256": "d" * 64}
    monkeypatch.setattr(adapter.matrix, "validate_plan", lambda _: plan)
    monkeypatch.setattr(
        adapter.wave3,
        "validate_plan",
        lambda payload: calls.append(("plan", payload)) or validated_native_plan,
    )
    monkeypatch.setattr(
        adapter.wave3,
        "validate_receipt",
        lambda payload, **kwargs: calls.append(("receipt", (payload, kwargs)))
        or payload,
    )

    receipt = adapter.publish_cell(
        plan_path=plan_path,
        cell=cell,
        native_receipts=[native_plan_path, native_receipt_path],
    )

    assert [name for name, _ in calls] == ["plan", "receipt"]
    _, (_, kwargs) = calls[1]
    assert kwargs == {"expected_plan": validated_native_plan}
    _assert_exact_cell(
        adapter, receipt=receipt, plan=plan, cell=cell, target=target
    )


def test_exact_resume_cell_requires_current_plan_bound_receipt_paths(
    adapter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schemas = {
        "wave7_sequence": adapter.wave7_sequence.RECEIPT_SCHEMA,
        "wave7_comparison": adapter.wave7_postrun.FINAL_COMPARISON_SCHEMA,
        "wave7_postrun": adapter.wave7_postrun.RECEIPT_SCHEMA,
    }
    paths = {
        name: _native_artifact(tmp_path / f"{name}.json", schema)
        for name, schema in schemas.items()
    }
    cell = "exact_resume"
    plan_path, plan, target = _plan(
        adapter, tmp_path, cell=cell, wave7_paths=paths
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(adapter.matrix, "validate_plan", lambda _: plan)
    monkeypatch.setattr(
        adapter.matrix,
        "_validate_wave7_evidence",
        lambda **kwargs: calls.append(kwargs) or {"status": "passed"},
    )

    receipt = adapter.publish_cell(
        plan_path=plan_path,
        cell=cell,
        native_receipts=list(reversed(paths.values())),
    )

    assert calls == [
        {
            "wave7_root": Path(plan["wave7_root"]),
            "sequence_path": paths["wave7_sequence"],
            "comparison_path": paths["wave7_comparison"],
            "postrun_path": paths["wave7_postrun"],
        }
    ]
    _assert_exact_cell(
        adapter, receipt=receipt, plan=plan, cell=cell, target=target
    )


def test_exact_resume_rejects_receipt_not_bound_by_matrix_plan(
    adapter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schemas = {
        "wave7_sequence": adapter.wave7_sequence.RECEIPT_SCHEMA,
        "wave7_comparison": adapter.wave7_postrun.FINAL_COMPARISON_SCHEMA,
        "wave7_postrun": adapter.wave7_postrun.RECEIPT_SCHEMA,
    }
    paths = {
        name: _native_artifact(tmp_path / f"{name}.json", schema)
        for name, schema in schemas.items()
    }
    plan_path, plan, target = _plan(
        adapter, tmp_path, cell="exact_resume", wave7_paths=paths
    )
    monkeypatch.setattr(adapter.matrix, "validate_plan", lambda _: plan)
    replacement = _native_artifact(
        tmp_path / "other-postrun.json", adapter.wave7_postrun.RECEIPT_SCHEMA
    )

    with pytest.raises(adapter.Wave8CoreCellError, match="bound"):
        adapter.publish_cell(
            plan_path=plan_path,
            cell="exact_resume",
            native_receipts=[paths["wave7_sequence"], paths["wave7_comparison"], replacement],
        )

    assert not target.exists()


def test_validator_failure_never_publishes_a_cell(
    adapter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = "protected_loss"
    plan_path, plan, target = _plan(adapter, tmp_path, cell=cell)
    native_plan_path = _native_artifact(
        tmp_path / "wave3-plan.json", adapter.wave3.PLAN_SCHEMA
    )
    native_receipt_path = _native_artifact(
        tmp_path / "wave3-receipt.json", adapter.wave3.RECEIPT_SCHEMA
    )
    monkeypatch.setattr(adapter.matrix, "validate_plan", lambda _: plan)
    monkeypatch.setattr(adapter.wave3, "validate_plan", lambda payload: payload)

    def reject(*_: object, **__: object) -> None:
        raise RuntimeError("native receipt did not pass")

    monkeypatch.setattr(adapter.wave3, "validate_receipt", reject)

    with pytest.raises(RuntimeError, match="did not pass"):
        adapter.publish_cell(
            plan_path=plan_path,
            cell=cell,
            native_receipts=[native_plan_path, native_receipt_path],
        )

    assert not target.exists()


def test_publish_is_absent_only(
    adapter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = "protected_loss"
    plan_path, plan, target = _plan(adapter, tmp_path, cell=cell)
    native_plan_path = _native_artifact(
        tmp_path / "wave3-plan.json", adapter.wave3.PLAN_SCHEMA
    )
    native_receipt_path = _native_artifact(
        tmp_path / "wave3-receipt.json", adapter.wave3.RECEIPT_SCHEMA
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("owned\n")
    monkeypatch.setattr(adapter.matrix, "validate_plan", lambda _: plan)
    monkeypatch.setattr(adapter.wave3, "validate_plan", lambda payload: payload)
    monkeypatch.setattr(
        adapter.wave3, "validate_receipt", lambda payload, **_: payload
    )

    with pytest.raises(Exception):
        adapter.publish_cell(
            plan_path=plan_path,
            cell=cell,
            native_receipts=[native_plan_path, native_receipt_path],
        )

    assert target.read_text() == "owned\n"
