from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import pytest
import torch

from scripts.research.human13_adamw_proposal_preservation import (
    FrozenWitnessBank,
    OwnerWitness,
    ParameterLayout,
    SOURCE_MEMBERSHIPS,
    TRUSTED_OWNER_CLASS,
    WEAKEST_MARGIN_SELECTION,
    WitnessBinding,
    jacobian_sha256,
    load_exact_adamw_proposal,
    load_frozen_witness_bank,
    load_projected_apply_receipt,
    load_projection_receipt,
)
from scripts.research.human13_rp_crossover_matrix_contracts import (
    AggregateResourceReceipt,
    AcquisitionKey,
    AuditRef,
    CANONICAL_IMAGE_IDS,
    PROPOSAL_COMPONENTS_BY_ARM,
    CellKey,
    CellSpec,
    DoseMechanicalReceipt,
    SharedEvidenceRef,
    canonical_seeds,
)
from scripts.research.human13_rp_crossover_runtime import (
    CellExecutionState,
    CellRuntimeError,
    ObjectiveBackwardReceipt,
    PrivateCheckpointRef,
    run_cell,
)
from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)


UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
_TEST_OUTPUT_ROOT = Path(tempfile.mkdtemp(prefix="human13-rp-runtime-tests-"))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _shared() -> SharedEvidenceRef:
    return SharedEvidenceRef(
        source_sha256=_digest("source-checkpoint"),
        manifest_sha256=_digest("manifest"),
        acquisition_path="shared/acquisition.json",
        acquisition_sha256=_digest("acquisition"),
        trajectory_credit_acquisition_sha256=_digest("acquisition"),
        credit_ledger_sha256=_digest("credit"),
        compiler_ledger_sha256=_digest("compiler"),
        policy_contract_sha256=_digest("policy"),
        native_receipts_sha256=_digest("native-receipts"),
        training_rp=1.0,
        seed_group_id="matrix_a",
        seeds=canonical_seeds("matrix_a"),
    )


def _component_hashes(arm_id: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (component, _digest(f"objective-component:{component}"))
        for component in PROPOSAL_COMPONENTS_BY_ARM[arm_id]
    )


def _spec(arm_id: str = "A", *, output_root: str | Path | None = None) -> CellSpec:
    components = {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }[arm_id]
    return CellSpec(
        cell_key=CellKey(
            AcquisitionKey(1.0, "matrix_a", "matrix"),
            arm_id,
        ),
        shared_evidence=_shared(),
        leaf_config_sha256=_digest(f"leaf-{arm_id}"),
        source_checkpoint_sha256=_digest("source-checkpoint"),
        expected_objective_components=components,
        objective_component_hashes=_component_hashes(arm_id),
        adamw_config_sha256=_digest("frozen-adamw-config"),
        fresh_optimizer_identity_sha256=_digest(f"fresh-optimizer-{arm_id}"),
        evaluation_rps=(1.0, 1.10),
        output_root=str(output_root or (_TEST_OUTPUT_ROOT / arm_id)),
        learning_rate=3.0e-6,
        global_learning_rate_decision_sha256=_digest("global-lr-decision"),
        resolved_leaf_config_sha256=_digest(f"resolved-leaf-{arm_id}"),
    )


def _bank(state: CellExecutionState) -> FrozenWitnessBank:
    layout = ParameterLayout.from_named_parameters(state.named_trainable_parameters)
    row = torch.ones(layout.total_numel, dtype=torch.float64)
    witness = OwnerWitness(
        image_id="image_00",
        owner_id="owner_00",
        source_membership=SOURCE_MEMBERSHIPS[0],
        owner_class=TRUSTED_OWNER_CLASS,
        token_id=123,
        margin_selection=WEAKEST_MARGIN_SELECTION,
        margin_value=0.5,
        detached=True,
        jacobian_sha256=jacobian_sha256(row),
    )
    return FrozenWitnessBank.from_witnesses(
        (witness,),
        jacobians={witness.canonical_key: row},
        binding=WitnessBinding(
            unit_id=UNIT_ID,
            source_checkpoint_sha256=_digest("source-checkpoint"),
            manifest_sha256=_digest("manifest"),
            frozen_before_acquisition=True,
        ),
        layout=layout,
    )


class FakeServices:
    def __init__(self, *, audit_failure_rp: float | None = None) -> None:
        self.audit_failure_rp = audit_failure_rp
        self.states: list[CellExecutionState] = []
        self.audit_calls: list[tuple[str, float]] = []
        self.cleaned: list[str] = []
        self.backward_calls = 0
        self.fresh_source = True
        self.shared_evidence_sha256: str | None = None
        self.component_hashes: tuple[tuple[str, str], ...] | None = None
        self.close_calls = 0

    def open_cell(self, spec: CellSpec) -> CellExecutionState:
        parameter = torch.nn.Parameter(torch.tensor((1.0, -1.0)))
        named = (("adapter.weight", parameter),)
        optimizer = torch.optim.AdamW(
            (parameter,),
            lr=spec.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        counter = UpdateCounter()
        state = CellExecutionState(
            named_trainable_parameters=named,
            optimizer=optimizer,
            scheduler=None,
            update_counter=counter,
            transaction=TrainingStateTransaction(
                named,
                optimizer=optimizer,
                scheduler=None,
                update_counter=counter,
                runtime=None,
                capture_cuda=False,
            ),
            world_size=1,
            source_checkpoint_sha256=spec.source_checkpoint_sha256,
            shared_evidence_sha256=(
                self.shared_evidence_sha256 or spec.shared_evidence.content_sha256
            ),
            adamw_config_sha256=spec.adamw_config_sha256,
            optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
            fresh_source=self.fresh_source,
        )
        self.states.append(state)
        return state

    def backward_objective(
        self, state: CellExecutionState, spec: CellSpec
    ) -> ObjectiveBackwardReceipt:
        self.backward_calls += 1
        state.named_trainable_parameters[0][1].grad = torch.tensor((0.25, -0.5))
        return ObjectiveBackwardReceipt(
            proposal_components=(
                ("trajectory",)
                if spec.cell_key.arm_id == "A"
                else ("trajectory", "compiler")
            ),
            component_hashes=(
                self.component_hashes
                if self.component_hashes is not None
                else spec.objective_component_hashes
            ),
            objective_ledger_sha256=_digest(
                ":".join(
                    ("trajectory",)
                    if spec.cell_key.arm_id == "A"
                    else ("trajectory", "compiler")
                )
            ),
            shared_evidence_sha256=spec.shared_evidence.content_sha256,
            trajectory_credit_acquisition_sha256=(
                spec.shared_evidence.trajectory_credit_acquisition_sha256
            ),
            compiler_ledger_sha256=spec.shared_evidence.compiler_ledger_sha256,
            image_ids=CANONICAL_IMAGE_IDS,
            backward_count=13,
            optimizer_step_count=0,
            trajectory_denominator=208,
            compiler_image_denominator=(None if spec.cell_key.arm_id == "A" else 13),
            released_graph_count=13,
        )

    def witness_bank(
        self, state: CellExecutionState, spec: CellSpec
    ) -> FrozenWitnessBank:
        return _bank(state)

    def realized_margin_probe(self, state, spec, bank):
        return lambda: {
            witness.canonical_key: witness.margin_value + 0.1
            for witness in bank.constraints
        }

    def write_private_checkpoint(
        self, state: CellExecutionState, spec: CellSpec
    ) -> PrivateCheckpointRef:
        return PrivateCheckpointRef(
            path=f"private/{spec.cell_key.arm_id}",
            checkpoint_sha256=_digest(f"proposal-checkpoint-{spec.cell_key.arm_id}"),
            private=True,
        )

    def audit_checkpoint(
        self,
        state: CellExecutionState,
        checkpoint: PrivateCheckpointRef,
        repetition_penalty: float,
    ) -> AuditRef:
        self.audit_calls.append((checkpoint.checkpoint_sha256, repetition_penalty))
        if repetition_penalty == self.audit_failure_rp:
            raise RuntimeError("injected audit failure")
        return AuditRef(
            evaluation_rp=repetition_penalty,
            evaluated_checkpoint_sha256=checkpoint.checkpoint_sha256,
            output_path=f"audit/rp-{repetition_penalty}.jsonl",
            output_sha256=_digest(f"audit-{repetition_penalty}"),
            row_count=13,
            image_ids=CANONICAL_IMAGE_IDS,
            generation_policy_receipt_sha256=_digest(f"policy-{repetition_penalty}"),
        )

    def cleanup_private_checkpoint(self, checkpoint: PrivateCheckpointRef) -> None:
        self.cleaned.append(checkpoint.path)

    def close_cell(self, state: CellExecutionState, spec: CellSpec) -> None:
        assert state is self.states[-1]
        self.close_calls += 1

    def aggregate_resource_receipt(
        self, state, spec, backward, audits
    ) -> AggregateResourceReceipt:
        return AggregateResourceReceipt(
            measurement_scope="injected_cpu",
            wall_time_seconds=0.01,
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
            backward_count=backward.backward_count,
            row_bytes=8192,
            artifact_bytes=4096,
            update_count=1,
            audit_count=len(audits),
            rollback_count=1,
        )

    def dose_mechanical_receipt(self, *args, **kwargs) -> DoseMechanicalReceipt:
        raise AssertionError("matrix tests must not request qualification mechanics")


@pytest.mark.parametrize("arm_id", ["A", "B"])
def test_unprojected_arms_take_one_exact_adamw_step_then_dual_audit_and_rollback(
    arm_id: str,
) -> None:
    services = FakeServices()
    written = []

    receipt = run_cell(_spec(arm_id), services=services, receipt_writer=written.append)

    state = services.states[0]
    assert receipt.status == "succeeded"
    assert receipt.objective_components == _spec(arm_id).expected_objective_components
    assert receipt.projection_receipt_sha256 is None
    assert [rp for _, rp in services.audit_calls] == [1.0, 1.10]
    assert len({checkpoint for checkpoint, _ in services.audit_calls}) == 1
    assert services.backward_calls == 1
    assert state.update_counter.value == 0
    assert len(state.optimizer.state) == 0
    assert all(
        parameter.grad is None for _, parameter in state.named_trainable_parameters
    )
    assert state.transaction.state_digest() == receipt.before_transaction_digest
    assert written == [receipt]
    assert services.cleaned == [f"private/{arm_id}"]
    assert services.close_calls == 1


def test_preservation_arm_projects_the_exact_b_proposal_without_optimizer_moments() -> (
    None
):
    services = FakeServices()

    receipt = run_cell(_spec("C"), services=services, receipt_writer=lambda _: None)

    state = services.states[0]
    assert receipt.status == "succeeded"
    assert receipt.projection_receipt_sha256 is not None
    assert state.update_counter.value == 0
    assert len(state.optimizer.state) == 0
    assert all(
        parameter.grad is None for _, parameter in state.named_trainable_parameters
    )
    assert tuple(audit.evaluation_rp for audit in receipt.audits) == (1.0, 1.10)


def test_decision_evidence_is_reloadable_after_success_and_audit_failure(
    tmp_path: Path,
) -> None:
    success_services = FakeServices()
    success = run_cell(
        _spec("C", output_root=tmp_path / "success"),
        services=success_services,
        receipt_writer=lambda _: None,
    )

    assert success.adamw_proposal_artifact_path is not None
    assert success.witness_bank_artifact_path is not None
    assert success.witness_bank_sha256 is not None
    assert success.projection_receipt_artifact_path is not None
    assert success.apply_receipt_artifact_path is not None
    proposal = load_exact_adamw_proposal(Path(success.adamw_proposal_artifact_path))
    bank = load_frozen_witness_bank(Path(success.witness_bank_artifact_path))
    projection = load_projection_receipt(
        Path(success.projection_receipt_artifact_path),
        expected_sha256=success.projection_receipt_sha256,
    )
    applied = load_projected_apply_receipt(
        Path(success.apply_receipt_artifact_path),
        expected_sha256=success.apply_receipt_sha256,
    )
    assert proposal.proposal_sha256 == success.adamw_proposal_sha256
    assert bank.bank_sha256 == success.witness_bank_sha256
    assert projection.receipt_sha256 == success.projection_receipt_sha256
    assert applied.receipt_sha256 == success.apply_receipt_sha256
    assert projection.predicted_changes
    assert applied.realized_changes
    assert success_services.cleaned == ["private/C"]

    failure_services = FakeServices(audit_failure_rp=1.10)
    with pytest.raises(CellRuntimeError) as error:
        run_cell(
            _spec("C", output_root=tmp_path / "failure"),
            services=failure_services,
            receipt_writer=lambda _: None,
        )
    failure = error.value.receipt
    assert failure is not None
    assert failure.status == "failed"
    assert failure.adamw_proposal_artifact_path is not None
    assert failure.witness_bank_artifact_path is not None
    assert failure.projection_receipt_artifact_path is not None
    assert failure.apply_receipt_artifact_path is not None
    assert Path(failure.adamw_proposal_artifact_path).is_file()
    assert Path(failure.witness_bank_artifact_path).is_dir()
    assert Path(failure.projection_receipt_artifact_path).is_file()
    assert Path(failure.apply_receipt_artifact_path).is_file()
    assert failure_services.cleaned == ["private/C"]


def test_audit_failure_rolls_back_and_writes_a_typed_failure_receipt() -> None:
    services = FakeServices(audit_failure_rp=1.10)
    written = []

    with pytest.raises(CellRuntimeError, match="injected audit failure") as error:
        run_cell(_spec("B"), services=services, receipt_writer=written.append)

    state = services.states[0]
    assert error.value.receipt is written[0]
    assert written[0].status == "failed"
    assert written[0].rollback_confirmed is True
    assert tuple(audit.evaluation_rp for audit in written[0].audits) == (1.0,)
    assert state.update_counter.value == 0
    assert len(state.optimizer.state) == 0
    assert state.transaction.state_digest() == written[0].before_transaction_digest
    assert services.cleaned == ["private/B"]
    assert services.close_calls == 1


def test_checkpoint_write_failure_rolls_back_and_writes_failure_receipt() -> None:
    class FailingCheckpointServices(FakeServices):
        def write_private_checkpoint(self, state, spec):
            raise OSError("injected private checkpoint write failure")

    services = FailingCheckpointServices()
    written = []

    with pytest.raises(CellRuntimeError, match="checkpoint write failure"):
        run_cell(_spec("A"), services=services, receipt_writer=written.append)

    assert written[0].status == "failed"
    assert written[0].audits == ()
    assert services.states[0].update_counter.value == 0
    assert all(
        parameter.grad is None
        for _, parameter in services.states[0].named_trainable_parameters
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda service, spec: setattr(service, "fresh_source", False), "fresh Source"),
        (
            lambda service, spec: setattr(
                service, "shared_evidence_sha256", _digest("wrong-shared")
            ),
            "shared evidence",
        ),
    ],
)
def test_runtime_rejects_nonindependent_or_wrong_evidence_before_backward(
    mutation, message: str
) -> None:
    services = FakeServices()
    spec = _spec("A")
    mutation(services, spec)

    with pytest.raises(CellRuntimeError, match=message):
        run_cell(spec, services=services, receipt_writer=lambda _: None)

    assert services.backward_calls == 0


def test_runtime_rejects_a_mislabeled_dual_rp_audit_and_rolls_back() -> None:
    class MislabeledAuditServices(FakeServices):
        def audit_checkpoint(self, state, checkpoint, repetition_penalty):
            audit = super().audit_checkpoint(state, checkpoint, repetition_penalty)
            if repetition_penalty == 1.10:
                return AuditRef(
                    evaluation_rp=1.0,
                    evaluated_checkpoint_sha256=audit.evaluated_checkpoint_sha256,
                    output_path=audit.output_path,
                    output_sha256=audit.output_sha256,
                    row_count=audit.row_count,
                    image_ids=audit.image_ids,
                    generation_policy_receipt_sha256=(
                        audit.generation_policy_receipt_sha256
                    ),
                )
            return audit

    services = MislabeledAuditServices()
    written = []

    with pytest.raises(CellRuntimeError, match="evaluation RP"):
        run_cell(_spec("A"), services=services, receipt_writer=written.append)

    assert written[0].status == "failed"
    assert services.states[0].transaction.state_digest() == (
        written[0].before_transaction_digest
    )


def test_receipt_binds_per_component_objective_bytes_and_transaction_identity() -> None:
    services = FakeServices()

    receipt = run_cell(_spec("B"), services=services, receipt_writer=lambda _: None)

    assert receipt.objective_component_hashes == _component_hashes("B")
    assert receipt.adamw_config_sha256 == _spec("B").adamw_config_sha256
    assert receipt.fresh_optimizer_identity_sha256 == (
        _spec("B").fresh_optimizer_identity_sha256
    )
    assert len(receipt.transaction_id) == 32
    assert receipt.proposal_delta_sha256 is not None


def test_preservation_arm_records_the_same_base_proposal_delta_as_arm_b() -> None:
    b_receipt = run_cell(
        _spec("B"), services=FakeServices(), receipt_writer=lambda _: None
    )
    c_receipt = run_cell(
        _spec("C"), services=FakeServices(), receipt_writer=lambda _: None
    )

    assert c_receipt.proposal_delta_sha256 == b_receipt.proposal_delta_sha256
    assert c_receipt.transaction_id != b_receipt.transaction_id


def test_runtime_rejects_objective_component_bytes_outside_the_cell_spec() -> None:
    services = FakeServices()
    services.component_hashes = (
        ("trajectory", _digest("forged-trajectory-objective")),
        ("compiler", _digest("objective-component:compiler")),
    )

    with pytest.raises(CellRuntimeError, match="objective component"):
        run_cell(_spec("B"), services=services, receipt_writer=lambda _: None)


@pytest.mark.parametrize(
    ("image_ids", "backward_count"),
    [
        (CANONICAL_IMAGE_IDS[:-1], 12),
        ((*CANONICAL_IMAGE_IDS[:-1], CANONICAL_IMAGE_IDS[0]), 13),
    ],
)
def test_runtime_rejects_partial_or_duplicate_backward_image_coverage_before_proposal(
    image_ids: tuple[int, ...], backward_count: int
) -> None:
    class IncompletePanelServices(FakeServices):
        def backward_objective(self, state, spec):
            receipt = super().backward_objective(state, spec)
            object.__setattr__(receipt, "image_ids", image_ids)
            object.__setattr__(receipt, "backward_count", backward_count)
            object.__setattr__(receipt, "released_graph_count", backward_count)
            return receipt

        def witness_bank(self, state, spec):
            raise AssertionError("proposal admission advanced past backward coverage")

    with pytest.raises(CellRuntimeError, match="canonical 13-image"):
        run_cell(
            _spec("C"),
            services=IncompletePanelServices(),
            receipt_writer=lambda _: None,
        )


def test_runtime_rejects_an_optimizer_identity_outside_the_cell_spec() -> None:
    class WrongOptimizerIdentityServices(FakeServices):
        def open_cell(self, spec):
            state = super().open_cell(spec)
            state.optimizer_identity_sha256 = _digest("another-cell-optimizer")
            return state

    with pytest.raises(CellRuntimeError, match="optimizer identity"):
        run_cell(
            _spec("A"),
            services=WrongOptimizerIdentityServices(),
            receipt_writer=lambda _: None,
        )


def test_runtime_rejects_a_succeeded_cell_without_aggregate_resource_receipt() -> None:
    """Success must not omit the cross-phase resource/count evidence."""

    class MissingResourceServices(FakeServices):
        def aggregate_resource_receipt(
            self, *args, **kwargs
        ) -> AggregateResourceReceipt:
            return None  # type: ignore[return-value]

    with pytest.raises(CellRuntimeError, match="aggregate resource receipt"):
        run_cell(
            _spec("C"),
            services=MissingResourceServices(),
            receipt_writer=lambda _: None,
        )
