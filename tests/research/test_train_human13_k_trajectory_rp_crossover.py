from __future__ import annotations

import hashlib
import io
import json

import pytest

from scripts.research.human13_rp_crossover_matrix_contracts import (
    AcquisitionKey,
    CellKey,
    CellReceipt,
    CellSpec,
    DRY_RUN_COUNTER_KEYS,
    SharedEvidenceRef,
)
from scripts.research import train_human13_k_trajectory_rp_crossover as cli


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _spec() -> CellSpec:
    shared = SharedEvidenceRef(
        source_sha256=_digest("source"),
        manifest_sha256=_digest("manifest"),
        acquisition_path="shared/acquisition.json",
        acquisition_sha256=_digest("acquisition"),
        trajectory_credit_acquisition_sha256=_digest("acquisition"),
        credit_ledger_sha256=_digest("credit"),
        compiler_ledger_sha256=_digest("compiler"),
        policy_contract_sha256=_digest("policy"),
    )
    return CellSpec(
        cell_key=CellKey(AcquisitionKey(1.0, "matrix_a", "matrix"), "A"),
        shared_evidence=shared,
        leaf_config_sha256=_digest("leaf"),
        source_checkpoint_sha256=_digest("checkpoint"),
        expected_objective_components=("trajectory",),
        fresh_adamw_fingerprint_sha256=_digest("adamw"),
        evaluation_rps=(1.0, 1.10),
        output_root="cells/A",
    )


def _write_spec(tmp_path) -> str:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(_spec().to_dict()), encoding="utf-8")
    return str(path)


def test_dry_run_emits_a_zero_action_plan_without_opening_runtime(tmp_path) -> None:
    output = io.StringIO()

    exit_code = cli.run_cli(
        ["--cell-spec", _write_spec(tmp_path)],
        services_factory=lambda _: pytest.fail("dry-run opened runtime"),
        receipt_writer=lambda _: pytest.fail("dry-run wrote a receipt"),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["mode"] == "dry_run"
    assert payload["cell_spec_sha256"] == _spec().content_sha256
    assert payload["execution_ready"] is False
    assert payload["actions"] == {key: 0 for key in DRY_RUN_COUNTER_KEYS}


def test_execute_requires_explicit_model_and_gpu_authority(tmp_path) -> None:
    with pytest.raises(PermissionError, match="explicit model/GPU authority"):
        cli.run_cli(
            ["--cell-spec", _write_spec(tmp_path), "--execute"],
            services_factory=lambda _: object(),
            receipt_writer=lambda _: None,
            stdout=io.StringIO(),
        )


def test_execute_fails_closed_without_a_production_runtime_adapter(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="production runtime adapter"):
        cli.run_cli(
            [
                "--cell-spec",
                _write_spec(tmp_path),
                "--execute",
                "--authorize-model-gpu-use",
            ],
            stdout=io.StringIO(),
        )


def test_execute_passes_the_validated_spec_to_the_injected_runtime(
    tmp_path, monkeypatch
) -> None:
    output = io.StringIO()
    sentinel_services = object()
    observed = {}
    expected = CellReceipt(
        cell_key=_spec().cell_key,
        shared_evidence=_spec().shared_evidence,
        objective_components=("trajectory",),
        before_transaction_digest=_digest("state"),
        after_transaction_digest=_digest("state"),
        status="failed",
        failure_reason="injected terminal receipt",
    )

    def fake_run_cell(spec, *, services, receipt_writer):
        observed.update(spec=spec, services=services, writer=receipt_writer)
        receipt_writer(expected)
        return expected

    monkeypatch.setattr(cli, "run_cell", fake_run_cell)

    def writer(_):
        return None

    exit_code = cli.run_cli(
        [
            "--cell-spec",
            _write_spec(tmp_path),
            "--execute",
            "--authorize-model-gpu-use",
        ],
        services_factory=lambda spec: sentinel_services,
        receipt_writer=writer,
        stdout=output,
    )

    assert exit_code == 0
    assert observed == {
        "spec": _spec(),
        "services": sentinel_services,
        "writer": writer,
    }
    assert json.loads(output.getvalue()) == expected.to_dict()
