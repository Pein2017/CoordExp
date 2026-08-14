from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from typing import Any, cast

import pytest

from scripts.research.human13_rp_crossover_matrix_contracts import (
    AcquisitionKey,
    AuditRef,
    CANONICAL_IMAGE_IDS,
    CellKey,
    CellReceipt,
    CellSpec,
    DRY_RUN_COUNTER_KEYS,
    SharedEvidenceRef,
)
from scripts.research import launch_human13_k_trajectory_rp_crossover as launcher
from scripts.research import train_human13_k_trajectory_rp_crossover as cli
from scripts.research.human13_rp_crossover_runtime import CellRuntimeServices


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _shared() -> SharedEvidenceRef:
    return SharedEvidenceRef(
        source_sha256=_digest("source"),
        manifest_sha256=_digest("manifest"),
        acquisition_path="shared/acquisition.json",
        acquisition_sha256=_digest("acquisition"),
        trajectory_credit_acquisition_sha256=_digest("acquisition"),
        credit_ledger_sha256=_digest("credit"),
        compiler_ledger_sha256=_digest("compiler"),
        policy_contract_sha256=_digest("policy"),
    )


def _spec(
    arm_id: str = "A",
    *,
    acquisition_key: AcquisitionKey | None = None,
    output_root: str | None = None,
) -> CellSpec:
    components = {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }[arm_id]
    return CellSpec(
        cell_key=CellKey(
            acquisition_key or AcquisitionKey(1.0, "matrix_a", "matrix"), arm_id
        ),
        shared_evidence=_shared(),
        leaf_config_sha256=_digest(f"leaf-{arm_id}"),
        source_checkpoint_sha256=_digest("checkpoint"),
        expected_objective_components=components,
        fresh_adamw_fingerprint_sha256=_digest("adamw"),
        evaluation_rps=(1.0, 1.10),
        output_root=output_root or f"cells/{arm_id}",
    )


def _write_spec(tmp_path) -> str:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(_spec().to_dict()), encoding="utf-8")
    return str(path)


def _write_dag_plan(tmp_path) -> tuple[str, dict[str, Any]]:
    plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id="integration",
        output_root=tmp_path / "artifacts",
    )
    path = tmp_path / "dag-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return str(path), plan


def _success_receipt(spec: CellSpec) -> CellReceipt:
    checkpoint = _digest(f"proposal-checkpoint-{spec.cell_key.arm_id}")
    audits = tuple(
        AuditRef(
            evaluation_rp=rp,
            evaluated_checkpoint_sha256=checkpoint,
            output_path=f"audit/{spec.cell_key.arm_id}/{rp}.jsonl",
            output_sha256=_digest(f"audit-{spec.cell_key.arm_id}-{rp}"),
            row_count=13,
            image_ids=CANONICAL_IMAGE_IDS,
            generation_policy_receipt_sha256=_digest(f"policy-{rp}"),
        )
        for rp in (1.0, 1.10)
    )
    return CellReceipt(
        cell_key=spec.cell_key,
        shared_evidence=spec.shared_evidence,
        objective_components=spec.expected_objective_components,
        before_transaction_digest=_digest("state"),
        after_transaction_digest=_digest("state"),
        status="succeeded",
        audits=audits,
        adamw_proposal_sha256=_digest(f"proposal-{spec.cell_key.arm_id}"),
        projection_receipt_sha256=(
            _digest("projection-C") if spec.cell_key.arm_id == "C" else None
        ),
        apply_receipt_sha256=_digest(f"apply-{spec.cell_key.arm_id}"),
    )


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
            services_factory=lambda _: cast(CellRuntimeServices, object()),
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
                "--user-model-gpu-authority",
            ],
            stdout=io.StringIO(),
        )


def test_execute_passes_the_validated_spec_to_the_injected_runtime(
    tmp_path, monkeypatch
) -> None:
    output = io.StringIO()
    sentinel_services = cast(CellRuntimeServices, object())
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
            "--user-model-gpu-authority",
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


def test_node_execute_acquires_once_runs_independent_cells_and_writes_one_terminal(
    tmp_path, monkeypatch
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = next(
        item for item in plan["acquisitions"] if item["node_id"] == "rp100:matrix_a"
    )
    observed_services: list[CellRuntimeServices] = []

    class NodeRuntime:
        def __init__(self) -> None:
            self.acquire_calls = 0
            self.cell_receipts: list[CellReceipt] = []
            self.terminals: list[tuple[str, Mapping[str, Any]]] = []

        def acquire_cell_specs(
            self, selected_node: Mapping[str, Any]
        ) -> tuple[CellSpec, ...]:
            self.acquire_calls += 1
            acquisition_key = AcquisitionKey.from_dict(selected_node["acquisition_key"])
            return tuple(
                _spec(
                    cell["cell_key"]["arm_id"],
                    acquisition_key=acquisition_key,
                    output_root=cell["output_root"],
                )
                for cell in selected_node["cells"]
            )

        def services_for_cell(self, spec: CellSpec) -> CellRuntimeServices:
            return cast(CellRuntimeServices, object())

        def write_cell_receipt(self, receipt: CellReceipt) -> None:
            self.cell_receipts.append(receipt)

        def write_node_terminal_receipt(
            self, path: str, payload: Mapping[str, Any]
        ) -> None:
            self.terminals.append((path, payload))

    runtime = NodeRuntime()

    def fake_run_cell(spec, *, services, receipt_writer):
        observed_services.append(services)
        receipt = _success_receipt(spec)
        receipt_writer(receipt)
        return receipt

    monkeypatch.setattr(cli, "run_cell", fake_run_cell)
    output = io.StringIO()

    exit_code = cli.run_cli(
        [
            "--dag-plan",
            dag_path,
            "--node-id",
            node["node_id"],
            "--receipt-path",
            node["receipt_path"],
            "--repo-root",
            str(launcher.REPO_ROOT),
            "--max-updates",
            "1",
            "--execute",
            "--user-model-gpu-authority",
        ],
        node_runtime_factory=cast(
            cli.NodeRuntimeFactory, lambda selected_node: runtime
        ),
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert runtime.acquire_calls == 1
    assert len(observed_services) == len(set(map(id, observed_services))) == 3
    assert [receipt.cell_key.arm_id for receipt in runtime.cell_receipts] == [
        "A",
        "B",
        "C",
    ]
    assert len({receipt.shared_evidence for receipt in runtime.cell_receipts}) == 1
    assert len(runtime.terminals) == 1
    assert runtime.terminals[0][0] == node["receipt_path"]
    assert payload == runtime.terminals[0][1]
    assert payload["status"] == "succeeded"
    assert payload["retry_policy"] == "none"


def test_node_execute_fails_closed_without_production_runtime_factory(tmp_path) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = next(
        item
        for item in plan["acquisitions"]
        if item["node_id"] == "rp110:qualification"
    )

    with pytest.raises(RuntimeError, match="production node runtime factory"):
        cli.run_cli(
            [
                "--dag-plan",
                dag_path,
                "--node-id",
                node["node_id"],
                "--receipt-path",
                node["receipt_path"],
                "--repo-root",
                str(launcher.REPO_ROOT),
                "--max-updates",
                "1",
                "--execute",
                "--user-model-gpu-authority",
            ],
            stdout=io.StringIO(),
        )
