from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest

from scripts.research import human13_rp_crossover_production as production
from scripts.research import launch_human13_k_trajectory_rp_crossover as launcher
from scripts.research import train_human13_k_trajectory_rp_crossover as runner
from scripts.research.human13_rp_crossover_matrix_contracts import (
    AggregateResourceReceipt,
    CANONICAL_IMAGE_IDS,
    AcquisitionKey,
    AuditRef,
    CellKey,
    CellReceipt,
    CellSpec,
    DoseMechanicalReceipt,
    SharedEvidenceRef,
    SourceBaselineRef,
)
from scripts.research.human13_rp_crossover_runtime import CellRuntimeServices


REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _qualification_node(tmp_path: Path, *, rp: float = 1.0) -> dict[str, Any]:
    plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id=f"production-rp-{rp}",
        output_root=tmp_path / "artifacts",
    )
    node_id = "rp100:qualification" if rp == 1.0 else "rp110:qualification"
    return next(item for item in plan["acquisitions"] if item["node_id"] == node_id)


def _qualification_spec(node: Mapping[str, Any], index: int = 0) -> CellSpec:
    acquisition = AcquisitionKey.from_dict(node["acquisition_key"])
    planned = node["cells"][index]
    trajectory = _digest(f"trajectory:{acquisition.training_rp}")
    compiler = _digest(f"compiler:{acquisition.training_rp}")
    source = production.SOURCE_CHECKPOINT_PAYLOAD_SHA256
    shared = SharedEvidenceRef(
        source_sha256=source,
        manifest_sha256=production.MANIFEST_SHA256,
        acquisition_path=str(Path(planned["output_root"]).parent / "acquisition"),
        acquisition_sha256=_digest(f"acquisition:{acquisition.training_rp}"),
        trajectory_credit_acquisition_sha256=_digest(
            f"trajectory-acquisition:{acquisition.training_rp}"
        ),
        credit_ledger_sha256=trajectory,
        compiler_ledger_sha256=compiler,
        policy_contract_sha256=_digest(f"policy:{acquisition.training_rp}"),
        native_receipts_sha256=_digest(f"native:{acquisition.training_rp}"),
        training_rp=acquisition.training_rp,
        seed_group_id="qualification",
        seeds=tuple(range(30001, 30017)),
    )
    return CellSpec(
        cell_key=CellKey(acquisition, "C", planned["learning_rate"]),
        shared_evidence=shared,
        leaf_config_sha256=planned["source_leaf_config_sha256"],
        source_checkpoint_sha256=source,
        expected_objective_components=("trajectory", "compiler", "preservation"),
        objective_component_hashes=(
            ("trajectory", trajectory),
            ("compiler", compiler),
        ),
        adamw_config_sha256=planned["adamw_config_sha256"],
        fresh_optimizer_identity_sha256=planned["fresh_optimizer_identity_sha256"],
        evaluation_rps=(1.0, 1.10),
        output_root=planned["output_root"],
        learning_rate=planned["learning_rate"],
        resolved_leaf_config_sha256=planned["resolved_leaf_config_sha256"],
    )


def _mechanical_receipts(
    selected: float = production.DEFAULT_QUALIFICATION_LEARNING_RATE,
    *,
    measurement_scope: str = "live",
) -> tuple[DoseMechanicalReceipt, ...]:
    if selected not in production.QUALIFICATION_LEARNING_RATE_RAY:
        raise ValueError(
            "selected learning rate must lie on the exact sealed qualification dose ray"
        )
    receipts = []
    default = production.DEFAULT_QUALIFICATION_LEARNING_RATE
    for rp in (1.0, 1.10):
        acquisition = AcquisitionKey(rp, "qualification", "qualification")
        for dose in production.QUALIFICATION_LEARNING_RATE_RAY:
            if selected == default:
                floor_passed = ceiling_passed = True
            elif selected > default:
                floor_passed = dose >= selected
                ceiling_passed = True
            else:
                floor_passed = True
                ceiling_passed = dose <= selected
            checkpoint = _digest(f"private:{rp}:{dose}")
            resources = AggregateResourceReceipt(
                measurement_scope=measurement_scope,
                wall_time_seconds=0.1,
                peak_host_rss_bytes=1024,
                cuda_peak_allocated_bytes=None,
                cuda_peak_reserved_bytes=None,
                acquisition_request_count=208,
                acquisition_batch_count=52,
                acquisition_token_count=4096,
                decode_request_count=26,
                decode_batch_count=26,
                decode_token_count=2048,
                packed_token_count=4096,
                logical_token_count=3900,
                forward_count=13,
                backward_count=13,
                row_bytes=8192,
                artifact_bytes=4096,
                update_count=1,
                audit_count=2,
                rollback_count=1,
            )
            receipts.append(
                DoseMechanicalReceipt(
                    cell_key=CellKey(acquisition, "C", dose),
                    proposal_sha256=_digest(f"proposal:{rp}:{dose}"),
                    private_checkpoint_sha256=checkpoint,
                    audit_checkpoint_sha256s=(checkpoint, checkpoint),
                    greedy_decision_change_count=int(floor_passed),
                    malformed_output_delta_count=int(not ceiling_passed),
                    cap_terminated_output_delta_count=0,
                    unparseable_output_delta_count=0,
                    active_witness_count=int(floor_passed),
                    jvp_fd_max_abs_error=1.0e-6,
                    jvp_fd_tolerance=1.0e-4,
                    median_abs_decision_margin_displacement=0.1,
                    median_abs_source_decision_margin=1.0,
                    rollback_reproduced=True,
                    resources=resources,
                )
            )
    return tuple(receipts)


def _learning_rate_decision(
    selected: float = production.DEFAULT_QUALIFICATION_LEARNING_RATE,
) -> production.GlobalLearningRateDecision:
    return production.select_global_learning_rate(_mechanical_receipts(selected))


def _validate_forged_decision(**changes: Any) -> None:
    source = _learning_rate_decision()
    forged = object.__new__(production.GlobalLearningRateDecision)
    for field in source.__dataclass_fields__:
        object.__setattr__(
            forged,
            field,
            changes.get(field, getattr(source, field)),
        )
    forged.__post_init__()


def _success_receipt(spec: CellSpec) -> CellReceipt:
    checkpoint = _digest("private-checkpoint")
    return CellReceipt(
        cell_key=spec.cell_key,
        shared_evidence=spec.shared_evidence,
        objective_components=spec.expected_objective_components,
        objective_component_hashes=spec.objective_component_hashes,
        adamw_config_sha256=spec.adamw_config_sha256,
        fresh_optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
        transaction_id=_digest("transaction")[:32],
        before_transaction_digest=_digest("source-state"),
        after_transaction_digest=_digest("source-state"),
        status="succeeded",
        audits=tuple(
            AuditRef(
                evaluation_rp=rp,
                evaluated_checkpoint_sha256=checkpoint,
                output_path=f"audit/rp-{rp}.jsonl",
                output_sha256=_digest(f"audit:{rp}"),
                row_count=13,
                image_ids=CANONICAL_IMAGE_IDS,
                generation_policy_receipt_sha256=_digest(f"audit-policy:{rp}"),
            )
            for rp in (1.0, 1.10)
        ),
        adamw_proposal_sha256=_digest("proposal"),
        proposal_delta_sha256=_digest("proposal-delta"),
        projection_receipt_sha256=_digest("projection"),
        apply_receipt_sha256=_digest("apply"),
        learning_rate=spec.learning_rate,
        global_learning_rate_decision_sha256=(
            spec.global_learning_rate_decision_sha256
        ),
        resolved_leaf_config_sha256=spec.resolved_leaf_config_sha256,
    )


def test_factory_import_and_plan_contract_are_runtime_free(tmp_path) -> None:
    program = """
import json
import sys
from scripts.research import human13_rp_crossover_production as production

print(json.dumps({
    "heavy_imports": [name for name in ("torch", "vllm", "transformers") if name in sys.modules],
    "contract": production.create_node_runtime.plan_contract(),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO_ROOT)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "heavy_imports": [],
        "contract": {
            "schema_version": "human13_rp_crossover_node_runtime_factory.v1",
            "arms": ["A", "B", "C"],
            "evaluation_rps": [1.0, 1.1],
            "max_updates": 1,
            "retry_policy": "none",
            "world_size": 1,
            "requires_user_model_gpu_authority": True,
            "actions": {
                "model_loads": 0,
                "gpu_allocations": 0,
                "subprocess_launches": 0,
                "output_roots_created": 0,
            },
        },
    }
    assert list(tmp_path.iterdir()) == []
    assert (
        runner.inspect_runtime_factory(
            "scripts.research.human13_rp_crossover_production:create_node_runtime"
        )["status"]
        == "execution_ready"
    )


def test_public_factory_owns_one_concrete_lazy_live_composition(tmp_path) -> None:
    """The public factory must be execution-ready without importing a runtime."""

    live_module = "scripts.research.human13_rp_crossover_live_composition"
    previous_live_module = sys.modules.pop(live_module, None)
    before = {name for name in ("torch", "vllm", "transformers") if name in sys.modules}
    try:
        runtime = production.create_node_runtime(_qualification_node(tmp_path))
        after = {
            name for name in ("torch", "vllm", "transformers") if name in sys.modules
        }

        assert type(runtime._composition).__name__ == (
            "LazyHuman13RPCrossoverLiveComposition"
        )
        assert type(runtime._composition._backend).__name__ == (
            "Human13RPCrossoverProductionBackend"
        )
        assert live_module not in sys.modules
        assert after == before
        assert not (tmp_path / "artifacts").exists()
    finally:
        if previous_live_module is not None:
            sys.modules[live_module] = previous_live_module


def test_qualification_closes_acquisition_before_real_node_requests_cell_service(
    tmp_path, monkeypatch
) -> None:
    node = _qualification_node(tmp_path)
    specs = tuple(_qualification_spec(node, index) for index in range(5))
    events: list[str] = []

    class Composition:
        def acquire_qualification(
            self,
            selected_node: Mapping[str, Any],
            frozen: production.FrozenProductionInputs,
        ) -> production.QualificationAcquisition:
            events.append("acquire")
            assert selected_node["node_id"] == "rp100:qualification"
            assert frozen.manifest_sha256 == production.MANIFEST_SHA256
            component_hashes = dict(specs[0].objective_component_hashes)
            return production.QualificationAcquisition(
                cell_specs=specs,
                source_baselines=tuple(
                    SourceBaselineRef(
                        evaluation_rp=rp,
                        output_a_sha256=_digest(f"source-baseline:{rp}"),
                        output_b_sha256=_digest(f"source-baseline:{rp}"),
                        checkpoint_sha256=(production.SOURCE_CHECKPOINT_PAYLOAD_SHA256),
                        image_ids=CANONICAL_IMAGE_IDS,
                    )
                    for rp in (1.0, 1.10)
                ),
                training_rp=1.0,
                seeds=tuple(range(30001, 30017)),
                image_ids=CANONICAL_IMAGE_IDS,
                native_request_count=208,
                native_batch_count=52,
                manifest_sha256=production.MANIFEST_SHA256,
                tokenizer_sha256=production.TOKENIZER_SHA256,
                prompt_policy_fingerprint=production.PROMPT_POLICY_FINGERPRINT,
                alias_bank_sha256=production.ALIAS_BANK_SHA256,
                nested_objective_hashes={
                    "A": (("trajectory", component_hashes["trajectory"]),),
                    "B": (
                        ("trajectory", component_hashes["trajectory"]),
                        ("compiler", component_hashes["compiler"]),
                    ),
                    "C": specs[0].objective_component_hashes,
                },
                streaming_mode="per_image_or_pack",
            )

        def close_acquisition(self) -> production.AcquisitionReleaseReceipt:
            events.append("close")
            return production.AcquisitionReleaseReceipt(
                engine_closed=True,
                model_released=True,
                panel_wide_logits_retained=False,
            )

        def services_for_cell(self, selected_spec: CellSpec) -> CellRuntimeServices:
            events.append("services")
            assert selected_spec in specs
            return cast(CellRuntimeServices, object())

    def fake_run_cell(selected_spec, *, services, receipt_writer):
        assert services is not None
        receipt = _success_receipt(selected_spec)
        receipt_writer(receipt)
        events.append("cell")
        return receipt

    monkeypatch.setattr(runner, "run_cell", fake_run_cell)
    composition = Composition()
    factory = runner.declare_node_runtime_factory(
        lambda selected: production.create_node_runtime(
            selected,
            _acquisition_owner=composition,
            _cell_services_owner=composition,
        )
    )

    terminal = runner._execute_node(
        node,
        receipt_path=node["receipt_path"],
        runtime_factory=factory,
    )

    events.append("terminal")
    assert events == ["acquire", "close", *(["services", "cell"] * 5), "terminal"]
    assert terminal["status"] == "succeeded"
    assert [item["cell_key"]["arm_id"] for item in terminal["cell_specs"]] == ["C"] * 5
    assert Path(node["receipt_path"]).is_file()


@pytest.mark.parametrize("selected", [2.0e-6, 0.0, float("nan")])
def test_global_learning_rate_decision_rejects_arbitrary_off_grid_doses(
    selected: float,
) -> None:
    with pytest.raises(ValueError, match="exact sealed qualification dose ray"):
        _learning_rate_decision(selected)


def test_global_learning_rate_decision_defaults_to_three_e_minus_six() -> None:
    decision = _learning_rate_decision()

    assert production.DEFAULT_QUALIFICATION_LEARNING_RATE == 3.0e-6
    assert decision.selected_learning_rate == 3.0e-6


def test_global_learning_rate_decision_rejects_an_arbitrary_receipt_digest() -> None:
    """A caller-chosen digest must not masquerade as ten mechanical receipts."""

    with pytest.raises(ValueError, match="mechanical qualification receipts"):
        production.GlobalLearningRateDecision.sealed(
            _digest("caller-asserted-not-a-qualification-receipt"),
            selected_learning_rate=1.0e-6,
        )


def test_global_selector_rejects_even_one_missing_mechanical_receipt() -> None:
    receipts = _mechanical_receipts()

    with pytest.raises(ValueError, match="exactly ten"):
        production.select_global_learning_rate(receipts[:-1])


def test_injected_cpu_selector_output_cannot_admit_matrix_planning(
    tmp_path: Path,
) -> None:
    decision = production.select_global_learning_rate(
        _mechanical_receipts(measurement_scope="injected_cpu")
    )
    live_decision = production.select_global_learning_rate(
        _mechanical_receipts(measurement_scope="live")
    )

    assert decision.measurement_scope == "injected_cpu"
    assert decision.production_admitted is False
    assert live_decision.production_admitted is True
    assert decision.content_sha256 != live_decision.content_sha256

    with pytest.raises(launcher.LaunchContractError, match="live measurement"):
        launcher.build_dag_plan(
            launcher.load_leaf_configs(),
            run_id="cpu-non-admitting",
            output_root=tmp_path / "artifacts",
            global_learning_rate_decision=decision,
        )


def test_global_selector_rejects_mixed_resource_measurement_scopes() -> None:
    receipts = list(_mechanical_receipts(measurement_scope="live"))
    resources = replace(receipts[0].resources, measurement_scope="injected_cpu")
    receipts[0] = replace(receipts[0], resources=resources)

    with pytest.raises(ValueError, match="one resource measurement scope"):
        production.select_global_learning_rate(receipts)


def test_dose_mechanical_schema_excludes_owner_outcomes_gains_and_losses() -> None:
    payload = _mechanical_receipts()[0].to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert "owner" not in encoded
    assert "gain" not in encoded
    assert "loss" not in encoded
    assert payload["greedy_decision_change_count"] == 1
    assert payload["rollback_reproduced"] is True
    assert payload["resources"]["update_count"] == 1


@pytest.mark.parametrize("selected", production.QUALIFICATION_LEARNING_RATE_RAY)
def test_every_sealed_qualification_dose_can_be_content_bound(
    selected: float,
) -> None:
    decision = _learning_rate_decision(selected)

    assert decision.selected_learning_rate == selected
    assert len(decision.content_sha256) == 64


def test_selector_resolved_nondefault_lr_binds_every_matrix_cell_hash(
    tmp_path: Path,
) -> None:
    decision = _learning_rate_decision(1.0e-6)
    plan = launcher.build_dag_plan(
        launcher.load_leaf_configs(),
        run_id="resolved-nondefault",
        output_root=tmp_path / "artifacts",
        global_learning_rate_decision=decision,
    )

    matrix_cells = [
        cell
        for node in plan["acquisitions"]
        if node["phase"] == "matrix"
        for cell in node["cells"]
    ]
    assert plan["matrix_execution_ready"] is True
    assert plan["selected_learning_rate"] == 1.0e-6
    assert plan["global_learning_rate_decision_sha256"] == decision.content_sha256
    assert len(matrix_cells) == 18
    assert {cell["learning_rate"] for cell in matrix_cells} == {1.0e-6}
    assert {cell["global_learning_rate_decision_sha256"] for cell in matrix_cells} == {
        decision.content_sha256
    }
    assert all(cell["resolved_leaf_config_sha256"] for cell in matrix_cells)
    assert all(cell["execution_blocked"] is None for cell in matrix_cells)


@pytest.mark.parametrize(
    ("field", "assignments", "message"),
    [
        (
            "training_rp_learning_rates",
            ((1.0, 3.0e-6), (1.10, 1.0e-6)),
            "both training RP contracts",
        ),
        (
            "arm_learning_rates",
            (("A", 3.0e-6), ("B", 1.0e-6), ("C", 3.0e-6)),
            "every A/B/C arm",
        ),
        (
            "seed_group_learning_rates",
            (("matrix_a", 3.0e-6), ("matrix_b", 3.0e-6), ("matrix_c", 1.0e-6)),
            "every matrix seed group",
        ),
    ],
)
def test_global_learning_rate_decision_rejects_per_contract_drift(
    field: str,
    assignments: tuple[tuple[str | float, float], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_forged_decision(**{field: assignments})


def test_global_learning_rate_decision_forbids_online_norm_adaptation() -> None:
    with pytest.raises(ValueError, match="grad/delta norms are covariates only"):
        _validate_forged_decision(online_adaptation=True)


def test_default_backend_preaction_failure_writes_a_durable_node_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    from scripts.research import human13_rp_crossover_production_backend as backend

    node = _qualification_node(tmp_path, rp=1.10)

    def fail_before_action(*_args, **_kwargs):
        raise backend.ProductionBackendError("injected pre-action failure")

    monkeypatch.setattr(
        backend.Human13RPCrossoverProductionBackend,
        "source_surface",
        fail_before_action,
    )
    monkeypatch.setattr(production, "_validate_frozen_inputs", lambda _rp: object())

    with pytest.raises(
        RuntimeError,
        match="acquisition must release vLLM",
    ):
        runner._execute_node(
            node,
            receipt_path=node["receipt_path"],
            runtime_factory=production.create_node_runtime,
        )

    payload = json.loads(Path(node["receipt_path"]).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["cell_specs"] == []
    assert "RuntimeError" in payload["failure_reason"]
    assert not Path(node["cells"][0]["output_root"]).exists()


def test_factory_rejects_forged_qualification_seeds_before_acquisition(
    tmp_path,
) -> None:
    node = _qualification_node(tmp_path)
    node["seeds"][0] = 30000

    with pytest.raises(ValueError, match="exact qualification node"):
        production.create_node_runtime(node).acquire_cell_specs(node)

    assert not Path(node["cells"][0]["output_root"]).exists()
