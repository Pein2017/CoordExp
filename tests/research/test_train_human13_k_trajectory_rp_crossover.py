from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

from scripts.research.human13_rp_crossover_matrix_contracts import (
    AcquisitionKey,
    AggregateResourceReceipt,
    AuditRef,
    CANONICAL_IMAGE_IDS,
    PROPOSAL_COMPONENTS_BY_ARM,
    CellKey,
    CellReceipt,
    CellSpec,
    DoseMechanicalReceipt,
    DRY_RUN_COUNTER_KEYS,
    NodeTerminalReceipt,
    PHASE_QUALIFICATION,
    QUALIFICATION_LEARNING_RATE_RAY,
    SharedEvidenceRef,
    canonical_seeds,
)
from scripts.research import launch_human13_k_trajectory_rp_crossover as launcher
from scripts.research import train_human13_k_trajectory_rp_crossover as cli
from scripts.research.human13_rp_crossover_runtime import (
    CellRuntimeError,
    CellRuntimeServices,
)
from scripts.research.human13_rp_crossover_production import (
    GlobalLearningRateDecision,
    select_global_learning_rate,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _resource_receipt() -> AggregateResourceReceipt:
    return AggregateResourceReceipt(
        measurement_scope="live",
        wall_time_seconds=1.0,
        peak_host_rss_bytes=1,
        cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,
        acquisition_request_count=1,
        acquisition_batch_count=1,
        acquisition_token_count=1,
        decode_request_count=2,
        decode_batch_count=2,
        decode_token_count=2,
        packed_token_count=1,
        logical_token_count=1,
        forward_count=1,
        backward_count=1,
        row_bytes=1,
        artifact_bytes=1,
        update_count=1,
        audit_count=2,
        rollback_count=1,
    )


def _global_learning_rate_decision() -> GlobalLearningRateDecision:
    receipts = []
    for rp in (1.0, 1.10):
        acquisition = AcquisitionKey(rp, "qualification", PHASE_QUALIFICATION)
        for dose in QUALIFICATION_LEARNING_RATE_RAY:
            checkpoint = _digest(f"checkpoint:{rp}:{dose}")
            receipts.append(
                DoseMechanicalReceipt(
                    cell_key=CellKey(acquisition, "C", dose),
                    proposal_sha256=_digest(f"proposal:{rp}:{dose}"),
                    private_checkpoint_sha256=checkpoint,
                    audit_checkpoint_sha256s=(checkpoint, checkpoint),
                    greedy_decision_change_count=1,
                    malformed_output_delta_count=0,
                    cap_terminated_output_delta_count=0,
                    unparseable_output_delta_count=0,
                    active_witness_count=1,
                    jvp_fd_max_abs_error=0.0,
                    jvp_fd_tolerance=1.0e-6,
                    median_abs_decision_margin_displacement=0.1,
                    median_abs_source_decision_margin=1.0,
                    rollback_reproduced=True,
                    resources=_resource_receipt(),
                )
            )
    return select_global_learning_rate(receipts)


def _shared(
    training_rp: float = 1.0, seed_group_id: str = "matrix_a"
) -> SharedEvidenceRef:
    tag = f"{training_rp}:{seed_group_id}"
    return SharedEvidenceRef(
        source_sha256=_digest("source-checkpoint"),
        manifest_sha256=_digest("manifest"),
        acquisition_path=f"shared/{tag}/acquisition.json",
        acquisition_sha256=_digest(f"acquisition:{tag}"),
        trajectory_credit_acquisition_sha256=_digest(f"credit-acq:{tag}"),
        credit_ledger_sha256=_digest(f"credit:{tag}"),
        compiler_ledger_sha256=_digest(f"compiler:{tag}"),
        policy_contract_sha256=_digest(f"policy:{training_rp}"),
        native_receipts_sha256=_digest(f"native-receipts:{tag}"),
        training_rp=training_rp,
        seed_group_id=seed_group_id,
        seeds=canonical_seeds(seed_group_id),
    )


def _component_hashes(
    arm_id: str, tag: str = "1.0:matrix_a"
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (component, _digest(f"objective:{component}:{tag}"))
        for component in PROPOSAL_COMPONENTS_BY_ARM[arm_id]
    )


def _spec(
    arm_id: str = "A",
    *,
    cell_key: CellKey | None = None,
    acquisition_key: AcquisitionKey | None = None,
    output_root: str | None = None,
    adamw_config_sha256: str | None = None,
    fresh_optimizer_identity_sha256: str | None = None,
    learning_rate: float = 3.0e-6,
    global_learning_rate_decision_sha256: str | None = None,
    resolved_leaf_config_sha256: str | None = None,
    source_leaf_config_sha256: str | None = None,
) -> CellSpec:
    components = {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }[arm_id]
    acquisition = acquisition_key or AcquisitionKey(1.0, "matrix_a", "matrix")
    selected_cell_key = cell_key or CellKey(acquisition, arm_id)
    if (
        selected_cell_key.acquisition_key != acquisition
        or selected_cell_key.arm_id != arm_id
    ):
        raise ValueError("test CellKey differs from the requested acquisition/arm")
    tag = f"{acquisition.training_rp}:{acquisition.seed_group_id}"
    return CellSpec(
        cell_key=selected_cell_key,
        shared_evidence=_shared(acquisition.training_rp, acquisition.seed_group_id),
        leaf_config_sha256=(source_leaf_config_sha256 or _digest(f"leaf-{arm_id}")),
        source_checkpoint_sha256=_digest("source-checkpoint"),
        expected_objective_components=components,
        objective_component_hashes=_component_hashes(arm_id, tag),
        adamw_config_sha256=adamw_config_sha256 or _digest("frozen-adamw-config"),
        fresh_optimizer_identity_sha256=(
            fresh_optimizer_identity_sha256 or _digest(f"optimizer:{tag}:{arm_id}")
        ),
        evaluation_rps=(1.0, 1.10),
        output_root=output_root or f"cells/{arm_id}",
        learning_rate=learning_rate,
        global_learning_rate_decision_sha256=(
            None
            if acquisition.phase == PHASE_QUALIFICATION
            else global_learning_rate_decision_sha256
            or _digest("test-global-lr-decision")
        ),
        resolved_leaf_config_sha256=(
            resolved_leaf_config_sha256 or _digest(f"resolved-leaf:{tag}:{arm_id}")
        ),
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
        global_learning_rate_decision=_global_learning_rate_decision(),
    )
    path = tmp_path / "dag-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return str(path), plan


def _success_receipt(spec: CellSpec) -> CellReceipt:
    checkpoint = _digest(f"proposal-checkpoint-{spec.cell_key.arm_id}")
    proposal_sha = _digest(f"proposal-{spec.cell_key.arm_id}")
    projection_sha = _digest("projection-C") if spec.cell_key.arm_id == "C" else None
    apply_sha = _digest(f"apply-{spec.cell_key.arm_id}")
    witness_sha = _digest("witness-C") if spec.cell_key.arm_id == "C" else None
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
        objective_component_hashes=spec.objective_component_hashes,
        adamw_config_sha256=spec.adamw_config_sha256,
        fresh_optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
        transaction_id=_digest(f"transaction-{spec.cell_key.arm_id}")[:32],
        before_transaction_digest=_digest("state"),
        after_transaction_digest=_digest("state"),
        status="succeeded",
        audits=audits,
        adamw_proposal_sha256=proposal_sha,
        proposal_delta_sha256=_digest(
            "proposal-delta-a" if spec.cell_key.arm_id == "A" else "proposal-delta-bc"
        ),
        projection_receipt_sha256=projection_sha,
        apply_receipt_sha256=apply_sha,
        adamw_proposal_artifact_path=f"/immutable/proposal/{proposal_sha}.json",
        witness_bank_artifact_path=(
            f"/immutable/witness/{witness_sha}" if witness_sha is not None else None
        ),
        witness_bank_sha256=witness_sha,
        projection_receipt_artifact_path=(
            f"/immutable/projection/{projection_sha}.json"
            if projection_sha is not None
            else None
        ),
        apply_receipt_artifact_path=(
            f"/immutable/apply/{apply_sha}.json"
            if spec.cell_key.arm_id == "C"
            else None
        ),
        learning_rate=spec.learning_rate,
        global_learning_rate_decision_sha256=(
            spec.global_learning_rate_decision_sha256
        ),
        resolved_leaf_config_sha256=spec.resolved_leaf_config_sha256,
    )


def _failed_receipt(
    spec: CellSpec, reason: str = "injected dose failure"
) -> CellReceipt:
    return replace(
        _success_receipt(spec),
        status="failed",
        failure_reason=reason,
    )


def _declared_factory(factory) -> cli.NodeRuntimeFactory:
    """Wrap a test double in the frozen, runtime-free node factory contract."""

    return cast(cli.NodeRuntimeFactory, cli.declare_node_runtime_factory(factory))


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
        adamw_config_sha256=_spec().adamw_config_sha256,
        fresh_optimizer_identity_sha256=_spec().fresh_optimizer_identity_sha256,
        transaction_id=_digest("transaction-failed")[:32],
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
                    adamw_config_sha256=cell["adamw_config_sha256"],
                    fresh_optimizer_identity_sha256=cell[
                        "fresh_optimizer_identity_sha256"
                    ],
                    learning_rate=cell["learning_rate"],
                    global_learning_rate_decision_sha256=cell[
                        "global_learning_rate_decision_sha256"
                    ],
                    resolved_leaf_config_sha256=cell["resolved_leaf_config_sha256"],
                    source_leaf_config_sha256=cell["source_leaf_config_sha256"],
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
        node_runtime_factory=_declared_factory(lambda selected_node: runtime),
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
    terminal = NodeTerminalReceipt.from_dict(json.loads(json.dumps(payload)))
    assert [spec.cell_key.arm_id for spec in terminal.cell_specs] == ["A", "B", "C"]
    assert [item.cell_key.arm_id for item in terminal.cell_receipts] == ["A", "B", "C"]
    assert len({spec.adamw_config_sha256 for spec in terminal.cell_specs}) == 1
    assert (
        len({spec.fresh_optimizer_identity_sha256 for spec in terminal.cell_specs}) == 3
    )


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


def _specs_for_node(node: Mapping[str, Any]) -> tuple[CellSpec, ...]:
    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    return tuple(
        _spec(
            cell["cell_key"]["arm_id"],
            cell_key=CellKey.from_dict(cell["cell_key"]),
            acquisition_key=acquisition_key,
            output_root=cell["output_root"],
            adamw_config_sha256=cell["adamw_config_sha256"],
            fresh_optimizer_identity_sha256=cell["fresh_optimizer_identity_sha256"],
            learning_rate=cell["learning_rate"],
            global_learning_rate_decision_sha256=cell[
                "global_learning_rate_decision_sha256"
            ],
            resolved_leaf_config_sha256=cell["resolved_leaf_config_sha256"],
            source_leaf_config_sha256=cell["source_leaf_config_sha256"],
        )
        for cell in node["cells"]
    )


def _node_runtime(plan_node: Mapping[str, Any], specs: tuple[CellSpec, ...]):
    class NodeRuntime:
        def __init__(self) -> None:
            self.terminals: list[tuple[str, Mapping[str, Any]]] = []

        def acquire_cell_specs(self, selected_node: Mapping[str, Any]):
            return specs

        def services_for_cell(self, spec: CellSpec) -> CellRuntimeServices:
            return cast(CellRuntimeServices, object())

        def write_cell_receipt(self, receipt: CellReceipt) -> None:
            return None

        def write_node_terminal_receipt(
            self, path: str, payload: Mapping[str, Any]
        ) -> None:
            self.terminals.append((path, payload))

    return NodeRuntime()


def _node_argv(dag_path: str, node: Mapping[str, Any]) -> list[str]:
    return [
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
    ]


def _matrix_node(plan: dict[str, Any]) -> Mapping[str, Any]:
    return next(
        item for item in plan["acquisitions"] if item["node_id"] == "rp100:matrix_a"
    )


def _write_forged_dag_plan(tmp_path, mutate) -> tuple[str, dict[str, Any]]:
    plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id="forged",
        output_root=tmp_path / "forged-artifacts",
        global_learning_rate_decision=_global_learning_rate_decision(),
    )
    mutate(plan)
    path = tmp_path / "forged-dag-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return str(path), plan


def test_node_rejects_a_second_declared_adamw_config_in_its_dag(tmp_path) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        _matrix_node(plan)["cells"][1]["adamw_config_sha256"] = _digest("second-config")

    dag_path, plan = _write_forged_dag_plan(tmp_path, mutate)
    node = _matrix_node(plan)

    with pytest.raises(ValueError, match="resolved config hash differs"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(
                lambda _: pytest.fail("opened a runtime for a forged node")
            ),
            stdout=io.StringIO(),
        )


def test_node_rejects_two_dag_cells_sharing_one_optimizer_identity(tmp_path) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        cells = _matrix_node(plan)["cells"]
        cells[1]["fresh_optimizer_identity_sha256"] = cells[0][
            "fresh_optimizer_identity_sha256"
        ]

    dag_path, plan = _write_forged_dag_plan(tmp_path, mutate)
    node = _matrix_node(plan)

    with pytest.raises(ValueError, match="independent fresh optimizer identity"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(
                lambda _: pytest.fail("opened a runtime for a forged node")
            ),
            stdout=io.StringIO(),
        )


def test_node_accepts_one_shared_config_with_independent_optimizer_identities(
    tmp_path, monkeypatch
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = _matrix_node(plan)
    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    specs = tuple(
        _spec(
            cell["cell_key"]["arm_id"],
            acquisition_key=acquisition_key,
            output_root=cell["output_root"],
            adamw_config_sha256=cell["adamw_config_sha256"],
            fresh_optimizer_identity_sha256=cell["fresh_optimizer_identity_sha256"],
            learning_rate=cell["learning_rate"],
            global_learning_rate_decision_sha256=cell[
                "global_learning_rate_decision_sha256"
            ],
            resolved_leaf_config_sha256=cell["resolved_leaf_config_sha256"],
            source_leaf_config_sha256=cell["source_leaf_config_sha256"],
        )
        for cell in node["cells"]
    )
    runtime = _node_runtime(node, specs)
    monkeypatch.setattr(cli, "run_cell", lambda spec, **_: _success_receipt(spec))
    output = io.StringIO()

    assert (
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(lambda _: runtime),
            stdout=output,
        )
        == 0
    )
    terminal = NodeTerminalReceipt.from_dict(json.loads(output.getvalue()))
    assert len({spec.adamw_config_sha256 for spec in terminal.cell_specs}) == 1
    assert (
        len({spec.fresh_optimizer_identity_sha256 for spec in terminal.cell_specs}) == 3
    )


def test_node_rejects_a_trajectory_objective_that_differs_across_its_arms(
    tmp_path, monkeypatch
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = _matrix_node(plan)
    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    specs = []
    for cell in node["cells"]:
        arm_id = cell["cell_key"]["arm_id"]
        spec = _spec(
            arm_id,
            acquisition_key=acquisition_key,
            output_root=cell["output_root"],
            adamw_config_sha256=cell["adamw_config_sha256"],
            fresh_optimizer_identity_sha256=cell["fresh_optimizer_identity_sha256"],
            learning_rate=cell["learning_rate"],
            global_learning_rate_decision_sha256=cell[
                "global_learning_rate_decision_sha256"
            ],
            resolved_leaf_config_sha256=cell["resolved_leaf_config_sha256"],
            source_leaf_config_sha256=cell["source_leaf_config_sha256"],
        )
        if arm_id == "C":
            spec = replace(
                spec,
                objective_component_hashes=_component_hashes("C", "drifted"),
            )
        specs.append(spec)
    runtime = _node_runtime(node, tuple(specs))
    monkeypatch.setattr(cli, "run_cell", lambda spec, **_: _success_receipt(spec))

    with pytest.raises(ValueError, match="trajectory objective"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(lambda _: runtime),
            stdout=io.StringIO(),
        )


def test_node_execute_rejects_a_factory_without_the_frozen_contract(tmp_path) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = _matrix_node(plan)

    with pytest.raises(ValueError, match="frozen node runtime contract"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=cast(cli.NodeRuntimeFactory, lambda _: object()),
            stdout=io.StringIO(),
        )


def test_node_execute_rejects_a_declared_but_absent_factory_reference(
    tmp_path,
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = _matrix_node(plan)

    with pytest.raises(ModuleNotFoundError):
        cli.run_cli(
            [
                *_node_argv(dag_path, node),
                "--runtime-factory",
                "project.runtime:create_node_runtime",
            ],
            stdout=io.StringIO(),
        )


def test_factory_contract_inspection_is_declared_only_for_a_placeholder() -> None:
    assert cli.inspect_runtime_factory(None)["status"] == "absent"
    declared = cli.inspect_runtime_factory("project.runtime:create_node_runtime")
    assert declared["status"] == "factory_declared"
    assert "ModuleNotFoundError" in declared["detail"]


def test_failed_node_terminal_still_persists_its_acquired_specs(
    tmp_path, monkeypatch
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = _matrix_node(plan)
    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    specs = tuple(
        _spec(
            cell["cell_key"]["arm_id"],
            acquisition_key=acquisition_key,
            output_root=cell["output_root"],
            adamw_config_sha256=cell["adamw_config_sha256"],
            fresh_optimizer_identity_sha256=cell["fresh_optimizer_identity_sha256"],
            learning_rate=cell["learning_rate"],
            global_learning_rate_decision_sha256=cell[
                "global_learning_rate_decision_sha256"
            ],
            resolved_leaf_config_sha256=cell["resolved_leaf_config_sha256"],
            source_leaf_config_sha256=cell["source_leaf_config_sha256"],
        )
        for cell in node["cells"]
    )
    runtime = _node_runtime(node, specs)

    def failing_run_cell(spec, **_kwargs):
        if spec.cell_key.arm_id == "B":
            raise RuntimeError("injected cell failure")
        return _success_receipt(spec)

    monkeypatch.setattr(cli, "run_cell", failing_run_cell)

    with pytest.raises(RuntimeError, match="injected cell failure"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(lambda _: runtime),
            stdout=io.StringIO(),
        )

    terminal = NodeTerminalReceipt.from_dict(runtime.terminals[0][1])
    assert terminal.status == "failed"
    assert [spec.cell_key.arm_id for spec in terminal.cell_specs] == ["A", "B", "C"]
    assert [item.cell_key.arm_id for item in terminal.cell_receipts] == ["A"]
    assert "injected cell failure" in (terminal.failure_reason or "")


def test_failed_first_qualification_dose_receipt_survives_node_handoff(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = next(
        item
        for item in plan["acquisitions"]
        if item["node_id"] == "rp100:qualification"
    )
    specs = _specs_for_node(node)
    runtime = _node_runtime(node, specs)

    def fail_first(spec, **_kwargs):
        receipt = _failed_receipt(spec)
        raise CellRuntimeError("injected first-dose failure", receipt=receipt)

    monkeypatch.setattr(cli, "run_cell", fail_first)

    with pytest.raises(CellRuntimeError, match="first-dose failure"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(lambda _: runtime),
            stdout=io.StringIO(),
        )

    terminal = NodeTerminalReceipt.from_dict(runtime.terminals[0][1])
    assert terminal.status == "failed"
    assert [receipt.status for receipt in terminal.cell_receipts] == ["failed"]
    assert [receipt.learning_rate for receipt in terminal.cell_receipts] == [
        specs[0].learning_rate
    ]


def test_failed_after_success_receipt_survives_a_cell_writer_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = next(
        item
        for item in plan["acquisitions"]
        if item["node_id"] == "rp100:qualification"
    )
    specs = _specs_for_node(node)
    runtime = _node_runtime(node, specs)
    calls = 0

    def writer(receipt: CellReceipt) -> None:
        if receipt.status == "failed":
            raise OSError("injected standalone receipt write failure")

    runtime.write_cell_receipt = writer

    def fail_second(spec, *, receipt_writer, **_kwargs):
        nonlocal calls
        calls += 1
        receipt = _success_receipt(spec) if calls == 1 else _failed_receipt(spec)
        receipt_writer(receipt)
        return receipt

    monkeypatch.setattr(cli, "run_cell", fail_second)

    with pytest.raises(OSError, match="standalone receipt write failure"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(lambda _: runtime),
            stdout=io.StringIO(),
        )

    terminal = NodeTerminalReceipt.from_dict(runtime.terminals[0][1])
    assert terminal.status == "failed"
    assert [receipt.status for receipt in terminal.cell_receipts] == [
        "succeeded",
        "failed",
    ]
    assert [receipt.learning_rate for receipt in terminal.cell_receipts] == [
        spec.learning_rate for spec in specs[:2]
    ]


def test_node_terminal_rejects_a_failed_receipt_from_another_dose(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag_path, plan = _write_dag_plan(tmp_path)
    node = next(
        item
        for item in plan["acquisitions"]
        if item["node_id"] == "rp100:qualification"
    )
    specs = _specs_for_node(node)
    runtime = _node_runtime(node, specs)

    def forged_failure(_spec, **_kwargs):
        raise CellRuntimeError(
            "forged dose receipt",
            receipt=_failed_receipt(specs[1]),
        )

    monkeypatch.setattr(cli, "run_cell", forged_failure)

    with pytest.raises(ValueError, match="acquired cell order"):
        cli.run_cli(
            _node_argv(dag_path, node),
            node_runtime_factory=_declared_factory(lambda _: runtime),
            stdout=io.StringIO(),
        )

    terminal = NodeTerminalReceipt.from_dict(runtime.terminals[0][1])
    assert terminal.status == "failed"
    assert terminal.cell_receipts == ()
    assert "acquired cell order" in (terminal.failure_reason or "")
