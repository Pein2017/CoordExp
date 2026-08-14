#!/usr/bin/env python3
"""One-cell runtime for the Human13 K-trajectory RP crossover screen.

The orchestration boundary is intentionally injected.  The current public
Human13 model builder does not yet admit this research unit or its A/B/C arms,
so this module owns the transaction and exact-update semantics while callers
must supply the production model/loss/checkpoint/evaluation adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Protocol

import torch

from scripts.research.human13_adamw_proposal_preservation import (
    AdamWProposalConfig,
    FrozenWitnessBank,
    ProposalBinding,
    apply_projected_delta,
    capture_exact_adamw_proposal,
    parameter_state_sha256,
    project_adamw_proposal,
)
from scripts.research.human13_rp_crossover_matrix_contracts import (
    PROPOSAL_COMPONENTS_BY_ARM,
    AuditRef,
    CellReceipt,
    CellSpec,
)
from scripts.research.human13_training_transaction import (
    TrainingStateSnapshot,
    TrainingStateTransaction,
    UpdateCounter,
)


UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ObjectiveBackwardReceipt:
    """Proof that one injected loss pass populated, but did not step, gradients.

    ``component_hashes`` binds the exact bytes of each proposal loss component,
    so a nested arm cannot silently consume a different trajectory or compiler
    objective than the one its acquisition planned.
    """

    proposal_components: tuple[str, ...]
    component_hashes: tuple[tuple[str, str], ...]
    objective_ledger_sha256: str
    shared_evidence_sha256: str
    trajectory_credit_acquisition_sha256: str
    compiler_ledger_sha256: str
    backward_count: int
    optimizer_step_count: int


@dataclass(frozen=True)
class PrivateCheckpointRef:
    """A proposal-only checkpoint that can be audited but never promoted."""

    path: str
    checkpoint_sha256: str
    private: bool

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("private checkpoint path must be nonempty")
        if _SHA256.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("private checkpoint identity must be a SHA-256 digest")
        if self.private is not True:
            raise ValueError("proposal checkpoint must be explicitly private")


@dataclass
class CellExecutionState:
    """Fresh, cell-private training state assembled by the injected adapter."""

    named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]]
    optimizer: torch.optim.Optimizer
    scheduler: object | None
    update_counter: UpdateCounter
    transaction: TrainingStateTransaction
    world_size: int
    source_checkpoint_sha256: str
    shared_evidence_sha256: str
    adamw_config_sha256: str
    optimizer_identity_sha256: str
    fresh_source: bool


class CellRuntimeServices(Protocol):
    """Production composition seam for model/loss/checkpoint/HF evaluation."""

    def open_cell(self, spec: CellSpec) -> CellExecutionState: ...

    def backward_objective(
        self, state: CellExecutionState, spec: CellSpec
    ) -> ObjectiveBackwardReceipt: ...

    def witness_bank(
        self, state: CellExecutionState, spec: CellSpec
    ) -> FrozenWitnessBank: ...

    def realized_margin_probe(
        self,
        state: CellExecutionState,
        spec: CellSpec,
        bank: FrozenWitnessBank,
    ) -> Callable[[], Mapping[str, float]]: ...

    def write_private_checkpoint(
        self, state: CellExecutionState, spec: CellSpec
    ) -> PrivateCheckpointRef: ...

    def audit_checkpoint(
        self,
        state: CellExecutionState,
        checkpoint: PrivateCheckpointRef,
        repetition_penalty: float,
    ) -> AuditRef: ...

    def cleanup_private_checkpoint(self, checkpoint: PrivateCheckpointRef) -> None: ...


class CellRuntimeError(RuntimeError):
    """A fail-closed cell outcome, optionally carrying its durable receipt."""

    def __init__(self, message: str, *, receipt: CellReceipt | None = None) -> None:
        super().__init__(message)
        self.receipt = receipt


def _validate_state(spec: CellSpec, state: CellExecutionState) -> None:
    if not isinstance(state, CellExecutionState):
        raise ValueError("runtime adapter must return CellExecutionState")
    if state.fresh_source is not True:
        raise ValueError("each cell requires an independently loaded fresh Source")
    if state.world_size != 1:
        raise ValueError("each cell requires an independent world-size-1 transaction")
    if state.source_checkpoint_sha256 != spec.source_checkpoint_sha256:
        raise ValueError("fresh Source checkpoint identity differs from CellSpec")
    if state.shared_evidence_sha256 != spec.shared_evidence.content_sha256:
        raise ValueError("cell shared evidence differs from the planned acquisition")
    if state.adamw_config_sha256 != spec.adamw_config_sha256:
        raise ValueError("declared AdamW configuration differs from CellSpec")
    if state.optimizer_identity_sha256 != spec.fresh_optimizer_identity_sha256:
        raise ValueError("this cell's fresh optimizer identity differs from CellSpec")
    if not isinstance(state.optimizer, torch.optim.AdamW):
        raise ValueError("cell optimizer must be torch.optim.AdamW")
    if state.optimizer.state or state.update_counter.value != 0:
        raise ValueError("each cell must start with fresh AdamW and zero updates")
    if any(
        parameter.grad is not None for _, parameter in state.named_trainable_parameters
    ):
        raise ValueError("fresh Source parameters must not carry stale gradients")
    if not isinstance(state.transaction, TrainingStateTransaction):
        raise ValueError("cell must use TrainingStateTransaction")


def _validate_backward(spec: CellSpec, receipt: ObjectiveBackwardReceipt) -> None:
    expected = PROPOSAL_COMPONENTS_BY_ARM[spec.cell_key.arm_id]
    if not isinstance(receipt, ObjectiveBackwardReceipt):
        raise ValueError("backward adapter must return ObjectiveBackwardReceipt")
    if tuple(receipt.proposal_components) != expected:
        raise ValueError(
            "proposal loss components violate the exact nested-arm contract"
        )
    if (
        tuple(tuple(item) for item in receipt.component_hashes)
        != spec.objective_component_hashes
    ):
        raise ValueError(
            "consumed objective component bytes differ from the planned CellSpec"
        )
    if receipt.backward_count != 1 or receipt.optimizer_step_count != 0:
        raise ValueError("cell requires exactly one backward and no adapter-owned step")
    if receipt.shared_evidence_sha256 != spec.shared_evidence.content_sha256:
        raise ValueError("backward pass did not consume the planned shared evidence")
    if (
        receipt.trajectory_credit_acquisition_sha256
        != spec.shared_evidence.trajectory_credit_acquisition_sha256
    ):
        raise ValueError(
            "trajectory credit is not byte-identical to shared acquisition"
        )
    if receipt.compiler_ledger_sha256 != spec.shared_evidence.compiler_ledger_sha256:
        raise ValueError("compiler ledger differs from shared acquisition")
    if (
        not receipt.objective_ledger_sha256
        or len(receipt.objective_ledger_sha256) != 64
    ):
        raise ValueError("objective ledger must be content-addressed")


def _binding(spec: CellSpec, backward: ObjectiveBackwardReceipt) -> ProposalBinding:
    training_rp = spec.cell_key.acquisition_key.training_rp
    return ProposalBinding(
        unit_id=UNIT_ID,
        arm_id=spec.cell_key.arm_id,
        training_rp="1.0" if training_rp == 1.0 else "1.10",
        seed_group=spec.cell_key.acquisition_key.seed_group_id,
        source_checkpoint_sha256=spec.source_checkpoint_sha256,
        manifest_sha256=spec.shared_evidence.manifest_sha256,
        objective_ledger_sha256=backward.objective_ledger_sha256,
    )


def _unprojected_apply_sha256(proposal_sha256: str, post_sha256: str) -> str:
    return _sha256(
        {
            "schema_version": "human13_unprojected_adamw_apply.v1",
            "proposal_sha256": proposal_sha256,
            "post_parameter_sha256": post_sha256,
            "update_count": 1,
        }
    )


def _failure_reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def run_cell(
    spec: CellSpec,
    *,
    services: CellRuntimeServices,
    receipt_writer: Callable[[CellReceipt], None],
) -> CellReceipt:
    """Execute one proposal cell, audit it twice, and unconditionally restore Source."""

    try:
        if not isinstance(spec, CellSpec) or CellSpec.from_dict(spec.to_dict()) != spec:
            raise ValueError("run_cell requires a validated canonical CellSpec")
        if not callable(receipt_writer):
            raise ValueError("run_cell requires an injected receipt writer")
        state = services.open_cell(spec)
        _validate_state(spec, state)
    except Exception as error:
        raise CellRuntimeError(str(error)) from error

    before_digest = state.transaction.state_digest()
    snapshot: TrainingStateSnapshot | None = None
    checkpoint: PrivateCheckpointRef | None = None
    proposal_sha256: str | None = None
    proposal_delta_sha256: str | None = None
    projection_sha256: str | None = None
    apply_sha256: str | None = None
    component_hashes: tuple[tuple[str, str], ...] = ()
    audits: list[AuditRef] = []
    update_attempted = False
    failure: Exception | None = None

    try:
        backward = services.backward_objective(state, spec)
        _validate_backward(spec, backward)
        component_hashes = spec.objective_component_hashes
        proposal = capture_exact_adamw_proposal(
            state.named_trainable_parameters,
            optimizer=state.optimizer,
            transaction=state.transaction,
            config=AdamWProposalConfig.frozen(),
            binding=_binding(spec, backward),
        )
        proposal_sha256 = proposal.proposal_sha256
        # The arm-independent identity of the measured proposal: arm C projects
        # exactly this base delta, so B and C of one acquisition must agree.
        proposal_delta_sha256 = proposal.delta_sha256
        snapshot = state.transaction.begin()
        update_attempted = True

        if spec.cell_key.arm_id in {"A", "B"}:
            state.optimizer.step()
            state.update_counter.value += 1
            post_sha256 = parameter_state_sha256(
                state.named_trainable_parameters, proposal.layout
            )
            if post_sha256 != proposal.post_parameter_sha256:
                raise ValueError("re-stepped AdamW proposal differs from exact capture")
            apply_sha256 = _unprojected_apply_sha256(proposal_sha256, post_sha256)
        else:
            bank = services.witness_bank(state, spec)
            projection = project_adamw_proposal(proposal=proposal, witness_bank=bank)
            projection_sha256 = projection.receipt_sha256
            applied = apply_projected_delta(
                state.named_trainable_parameters,
                proposal=proposal,
                witness_bank=bank,
                projection=projection,
                optimizer=state.optimizer,
                transaction=state.transaction,
                update_counter=state.update_counter,
                realized_margin_probe=services.realized_margin_probe(state, spec, bank),
            )
            apply_sha256 = _sha256(applied.to_dict())

        if state.update_counter.value != 1:
            raise ValueError("cell must contain exactly one update")
        checkpoint = services.write_private_checkpoint(state, spec)
        if (
            not isinstance(checkpoint, PrivateCheckpointRef)
            or checkpoint.private is not True
        ):
            raise ValueError("audits require one typed private proposal checkpoint")
        for repetition_penalty in spec.evaluation_rps:
            audit = services.audit_checkpoint(state, checkpoint, repetition_penalty)
            if not isinstance(audit, AuditRef):
                raise ValueError("audit adapter must return AuditRef")
            if audit.evaluation_rp != repetition_penalty:
                raise ValueError("audit evaluation RP differs from requested policy")
            if audit.evaluated_checkpoint_sha256 != checkpoint.checkpoint_sha256:
                raise ValueError(
                    "both audits must evaluate the same private checkpoint"
                )
            audits.append(audit)
    except Exception as error:
        failure = error
    finally:
        try:
            if snapshot is not None:
                state.transaction.reject(snapshot)
            state.optimizer.zero_grad(set_to_none=True)
            if state.transaction.state_digest() != before_digest:
                raise RuntimeError(
                    "cell rollback did not restore its full training state"
                )
        except Exception as rollback_error:
            failure = rollback_error
        try:
            if checkpoint is not None:
                services.cleanup_private_checkpoint(checkpoint)
        except Exception as cleanup_error:
            failure = cleanup_error

    if failure is not None:
        if not update_attempted:
            raise CellRuntimeError(str(failure)) from failure
        assert snapshot is not None
        receipt = CellReceipt(
            cell_key=spec.cell_key,
            shared_evidence=spec.shared_evidence,
            objective_components=spec.expected_objective_components,
            objective_component_hashes=component_hashes,
            adamw_config_sha256=spec.adamw_config_sha256,
            fresh_optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
            transaction_id=snapshot.transaction_id,
            before_transaction_digest=before_digest,
            after_transaction_digest=state.transaction.state_digest(),
            status="failed",
            audits=tuple(audits),
            adamw_proposal_sha256=proposal_sha256,
            proposal_delta_sha256=proposal_delta_sha256,
            projection_receipt_sha256=projection_sha256,
            apply_receipt_sha256=apply_sha256,
            failure_reason=_failure_reason(failure),
        )
        receipt_writer(receipt)
        raise CellRuntimeError(str(failure), receipt=receipt) from failure

    assert snapshot is not None
    receipt = CellReceipt(
        cell_key=spec.cell_key,
        shared_evidence=spec.shared_evidence,
        objective_components=spec.expected_objective_components,
        objective_component_hashes=component_hashes,
        adamw_config_sha256=spec.adamw_config_sha256,
        fresh_optimizer_identity_sha256=spec.fresh_optimizer_identity_sha256,
        transaction_id=snapshot.transaction_id,
        before_transaction_digest=before_digest,
        after_transaction_digest=state.transaction.state_digest(),
        status="succeeded",
        audits=tuple(audits),
        adamw_proposal_sha256=proposal_sha256,
        proposal_delta_sha256=proposal_delta_sha256,
        projection_receipt_sha256=projection_sha256,
        apply_receipt_sha256=apply_sha256,
    )
    receipt_writer(receipt)
    return receipt


__all__ = [
    "CellExecutionState",
    "CellRuntimeError",
    "CellRuntimeServices",
    "ObjectiveBackwardReceipt",
    "PrivateCheckpointRef",
    "UNIT_ID",
    "run_cell",
]
