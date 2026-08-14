"""Task-5 CPU contract tests for exact AdamW proposals and owner preservation.

These tests never load a real model.  They bind the frozen scientific optimizer
contract, the bias-corrected metric, the owner-wise projection equation, the
exact projected apply, and the full-state transaction rollback with real toy
optimizers only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import errno
import json
import math
import os

import pytest
import torch

from scripts.research import human13_adamw_proposal_preservation as preservation
from scripts.research.human13_adamw_proposal_preservation import (
    FROZEN_BETAS,
    FROZEN_EPSILON,
    FROZEN_LEARNING_RATE,
    FROZEN_WEIGHT_DECAY,
    LEGACY_M_MEMBERSHIP,
    LEGACY_M_OWNER_CLASS,
    SOURCE_MEMBERSHIPS,
    TRUSTED_OWNER_CLASS,
    WEAKEST_MARGIN_SELECTION,
    WITNESS_FIRST_ORDER_TOLERANCE,
    AdamWProposalConfig,
    ExactAdamWProposal,
    FrozenWitnessBank,
    OwnerWitness,
    ParameterLayout,
    ProjectedApplyError,
    ProposalAdmissionError,
    ProposalBinding,
    ProposalCaptureError,
    ProposalProjectionError,
    WitnessBinding,
    _hash_flat_tensors,
    _representation_ulp,
    _solve_metric_projection,
    _solve_working_set,
    admit_preservation_evidence,
    apply_projected_delta,
    capture_exact_adamw_proposal,
    jacobian_sha256,
    load_exact_adamw_proposal,
    load_frozen_witness_bank,
    load_projected_apply_receipt,
    load_projection_receipt,
    parameter_state_sha256,
    project_adamw_proposal,
    write_exact_adamw_proposal,
    write_frozen_witness_bank,
    write_projected_apply_receipt,
    write_projection_receipt,
)
from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)


_UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
_SOURCE_SHA = "a" * 64
_MANIFEST_SHA = "b" * 64
_LEDGER_SHA = "c" * 64
_WEIGHT = "adapter.language.weight"
_BIAS = "adapter.language.bias"

_DENSE_GRADIENTS = {
    _WEIGHT: torch.tensor(((0.5, -0.25, 2.0), (1e-3, 4.0, -7.5))),
    _BIAS: torch.tensor((0.125, -1.5)),
}
_ZERO_GRADIENTS = {
    _WEIGHT: torch.zeros((2, 3)),
    _BIAS: torch.zeros((2,)),
}
_MIXED_GRADIENTS = {
    _WEIGHT: torch.tensor(((-3.0, 0.0, 0.75), (12.5, -0.0, 1e-6))),
    _BIAS: torch.tensor((0.0, 9.75)),
}


class _FailingPublicationFileSystem:
    """Inject one real partial staging write, then fail before publication."""

    def __init__(self, *, write_number: int, error: BaseException) -> None:
        self._delegate = preservation._LocalArtifactPublicationFileSystem()
        self._write_number = write_number
        self._error = error
        self._writes = 0

    def write_payload(self, descriptor: int, payload: bytes) -> None:
        self._writes += 1
        if self._writes != self._write_number:
            self._delegate.write_payload(descriptor, payload)
            return
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload[: max(1, len(payload) // 2)])
            handle.flush()
        raise self._error

    def publish_file_exclusive(self, temporary_path, final_path) -> None:
        self._delegate.publish_file_exclusive(temporary_path, final_path)

    def publish_directory_exclusive(self, temporary_path, final_path) -> None:
        self._delegate.publish_directory_exclusive(temporary_path, final_path)

    def fsync_directory(self, directory) -> None:
        self._delegate.fsync_directory(directory)


class _FailingDirectoryFsyncPublicationFileSystem:
    def __init__(self, *, fsync_number: int) -> None:
        self._delegate = preservation._LocalArtifactPublicationFileSystem()
        self._fsync_number = fsync_number
        self._fsyncs = 0

    def write_payload(self, descriptor: int, payload: bytes) -> None:
        self._delegate.write_payload(descriptor, payload)

    def publish_file_exclusive(self, temporary_path, final_path) -> None:
        self._delegate.publish_file_exclusive(temporary_path, final_path)

    def publish_directory_exclusive(self, temporary_path, final_path) -> None:
        self._delegate.publish_directory_exclusive(temporary_path, final_path)

    def fsync_directory(self, directory) -> None:
        self._fsyncs += 1
        if self._fsyncs == self._fsync_number:
            raise OSError(errno.EIO, "injected directory fsync failure")
        self._delegate.fsync_directory(directory)


def test_frozen_adamw_capture_accepts_a_selected_nondefault_ray_dose() -> None:
    """Task 5 must consume the globally selected dose, not hard-fix 3e-6."""

    config = AdamWProposalConfig.frozen(learning_rate=1.0e-6)

    assert config.learning_rate == 1.0e-6
    assert config.betas == FROZEN_BETAS


def _binding(*, arm_id: str = "C_trajectory_compiler_preservation") -> ProposalBinding:
    return ProposalBinding(
        unit_id=_UNIT_ID,
        arm_id=arm_id,
        training_rp="1.10",
        seed_group="31001..31016",
        source_checkpoint_sha256=_SOURCE_SHA,
        manifest_sha256=_MANIFEST_SHA,
        objective_ledger_sha256=_LEDGER_SHA,
    )


def _witness_binding(*, source_checkpoint_sha256: str = _SOURCE_SHA) -> WitnessBinding:
    return WitnessBinding(
        unit_id=_UNIT_ID,
        source_checkpoint_sha256=source_checkpoint_sha256,
        manifest_sha256=_MANIFEST_SHA,
        frozen_before_acquisition=True,
    )


def _make_stack(
    seed: int = 11,
    *,
    optimizer_factory: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
    learning_rate: float = FROZEN_LEARNING_RATE,
):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2)
    parameters = dict(model.named_parameters())
    named = ((_WEIGHT, parameters["weight"]), (_BIAS, parameters["bias"]))
    optimizer = optimizer_factory(
        [parameter for _, parameter in named],
        lr=learning_rate,
        betas=FROZEN_BETAS,
        eps=FROZEN_EPSILON,
        weight_decay=FROZEN_WEIGHT_DECAY,
    )
    counter = UpdateCounter()
    transaction = TrainingStateTransaction(
        named,
        optimizer=optimizer,
        scheduler=None,
        update_counter=counter,
        runtime=None,
        capture_cuda=False,
    )
    return model, named, optimizer, counter, transaction


def _assign(named, gradients: Mapping[str, torch.Tensor]) -> None:
    for name, parameter in named:
        parameter.grad = gradients[name].clone()


def _capture(named, optimizer, transaction, **kwargs) -> ExactAdamWProposal:
    return capture_exact_adamw_proposal(
        named,
        optimizer=optimizer,
        transaction=transaction,
        config=AdamWProposalConfig.frozen(),
        binding=kwargs.pop("binding", _binding()),
        **kwargs,
    )


def _reference_step(gradients: Mapping[str, torch.Tensor], seed: int = 11):
    _, named, optimizer, _, _ = _make_stack(seed)
    _assign(named, gradients)
    pre = {name: parameter.detach().clone() for name, parameter in named}
    optimizer.step()
    return named, optimizer, pre


def _witness(
    owner_id: str,
    *,
    image_id: str = "image_00",
    membership: str = SOURCE_MEMBERSHIPS[0],
    owner_class: str = TRUSTED_OWNER_CLASS,
    token_id: int = 1234,
    margin_value: float = 0.5,
    jacobian: torch.Tensor | None = None,
    margin_selection: str = WEAKEST_MARGIN_SELECTION,
    detached: bool = True,
) -> OwnerWitness:
    return OwnerWitness(
        image_id=image_id,
        owner_id=owner_id,
        source_membership=membership,
        owner_class=owner_class,
        token_id=token_id,
        margin_selection=margin_selection,
        margin_value=margin_value,
        detached=detached,
        jacobian_sha256=None if jacobian is None else jacobian_sha256(jacobian),
    )


def _bank(
    layout: ParameterLayout,
    rows: Mapping[str, torch.Tensor],
    *,
    extra: tuple[OwnerWitness, ...] = (),
    binding: WitnessBinding | None = None,
    order: tuple[str, ...] | None = None,
) -> FrozenWitnessBank:
    keys = tuple(rows) if order is None else order
    witnesses = []
    jacobians = {}
    for owner_id in keys:
        witness = _witness(owner_id, jacobian=rows[owner_id])
        witnesses.append(witness)
        jacobians[witness.canonical_key] = rows[owner_id]
    witnesses.extend(extra)
    return FrozenWitnessBank.from_witnesses(
        tuple(witnesses),
        jacobians=jacobians,
        binding=binding if binding is not None else _witness_binding(),
        layout=layout,
    )


def _active_row(proposal: ExactAdamWProposal, scale: float = 1000.0) -> torch.Tensor:
    return -scale * torch.sign(proposal.flat_delta())


def _probe(bank: FrozenWitnessBank, changes: Mapping[str, float]):
    def probe() -> Mapping[str, float]:
        return {
            witness.canonical_key: witness.margin_value
            + changes.get(witness.owner_id, 0.0)
            for witness in bank.constraints
        }

    return probe


# --- exact optimizer proposal -------------------------------------------------


def test_capture_reproduces_a_real_fresh_adamw_step_for_dense_gradients() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    before_digest = transaction.state_digest()
    pre_hash = parameter_state_sha256(
        named, ParameterLayout.from_named_parameters(named)
    )

    proposal = _capture(named, optimizer, transaction)

    reference_named, reference_optimizer, reference_pre = _reference_step(
        _DENSE_GRADIENTS
    )
    for index, (name, parameter) in enumerate(reference_named):
        expected_delta = (
            parameter.detach().to(torch.float64) - reference_pre[name].to(torch.float64)
        ).reshape(-1)
        assert torch.equal(proposal.delta[index], expected_delta)
        state = reference_optimizer.state[parameter]
        bias_correction2 = 1.0 - FROZEN_BETAS[1] ** float(state["step"])
        expected_metric = (
            (state["exp_avg_sq"].to(torch.float64) / bias_correction2).sqrt()
            + FROZEN_EPSILON
        ).reshape(-1)
        assert torch.equal(proposal.metric_denominator[index], expected_metric)

    assert proposal.pre_parameter_sha256 == pre_hash
    assert proposal.post_parameter_sha256 == parameter_state_sha256(
        reference_named, proposal.layout
    )
    assert proposal.post_parameter_sha256 != proposal.pre_parameter_sha256
    assert proposal.trust_radius == pytest.approx(
        float(
            (
                proposal.flat_metric_denominator()
                * proposal.flat_delta()
                * proposal.flat_delta()
            ).sum()
        ),
        rel=0.0,
        abs=0.0,
    )
    assert (
        proposal.delta_reconstruction_residual
        <= proposal.delta_reconstruction_allowance
    )

    # The capture is private: nothing in the live stack advanced.
    assert len(optimizer.state) == 0
    assert counter.value == 0
    assert (
        transaction.state_digest()
        == before_digest
        == proposal.transaction_before_digest
    )
    assert proposal.transaction_after_digest == before_digest
    assert proposal.optimizer_step_count_before == 0
    assert proposal.update_count_before == 0
    assert parameter_state_sha256(named, proposal.layout) == pre_hash


@pytest.mark.parametrize("gradients", (_ZERO_GRADIENTS, _MIXED_GRADIENTS))
def test_capture_matches_real_steps_for_zero_and_mixed_sign_gradients(
    gradients: Mapping[str, torch.Tensor],
) -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, gradients)

    proposal = _capture(named, optimizer, transaction)

    reference_named, _, reference_pre = _reference_step(gradients)
    for index, (name, parameter) in enumerate(reference_named):
        expected_delta = (
            parameter.detach().to(torch.float64) - reference_pre[name].to(torch.float64)
        ).reshape(-1)
        assert torch.equal(proposal.delta[index], expected_delta)
    assert bool(torch.isfinite(proposal.flat_delta()).all())
    assert float(proposal.flat_metric_denominator().min()) >= FROZEN_EPSILON
    if gradients is _ZERO_GRADIENTS:
        assert float(proposal.flat_delta().abs().max()) == 0.0
        assert proposal.trust_radius == 0.0
        assert proposal.post_parameter_sha256 == proposal.pre_parameter_sha256


def test_metric_denominator_is_bias_corrected_and_epsilon_bearing() -> None:
    tiny = {
        _WEIGHT: torch.full((2, 3), 1e-9),
        _BIAS: torch.full((2,), -1e-9),
    }
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, tiny)

    proposal = _capture(named, optimizer, transaction)

    reference_named, reference_optimizer, _ = _reference_step(tiny)
    metric = proposal.flat_metric_denominator()
    uncorrected = torch.cat(
        [
            (reference_optimizer.state[parameter]["exp_avg_sq"].to(torch.float64))
            .sqrt()
            .reshape(-1)
            for _, parameter in reference_named
        ]
    )
    without_epsilon = torch.cat(
        [
            (
                reference_optimizer.state[parameter]["exp_avg_sq"].to(torch.float64)
                / (1.0 - FROZEN_BETAS[1])
            )
            .sqrt()
            .reshape(-1)
            for _, parameter in reference_named
        ]
    )
    # Bias correction is a ~31.6x effect; epsilon dominates a 1e-9 gradient.
    assert float((metric / (uncorrected + FROZEN_EPSILON)).min()) > 1.05
    assert float((metric / without_epsilon).min()) > 5.0
    assert float(metric.min()) > FROZEN_EPSILON


def test_capture_fails_closed_on_non_finite_gradient_without_touching_state() -> None:
    poisoned = {
        _WEIGHT: torch.tensor(((0.5, float("nan"), 2.0), (1e-3, 4.0, -7.5))),
        _BIAS: torch.tensor((0.125, float("inf"))),
    }
    reference_named, _, _ = _reference_step(poisoned)
    assert not bool(torch.isfinite(dict(reference_named)[_WEIGHT].detach()).all()), (
        "a real AdamW step really does poison the parameters"
    )

    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, poisoned)
    before_digest = transaction.state_digest()

    with pytest.raises(ProposalCaptureError) as error:
        _capture(named, optimizer, transaction)

    assert error.value.disposition == "non_finite_gradient"
    assert transaction.state_digest() == before_digest
    assert len(optimizer.state) == 0
    assert counter.value == 0


def test_capture_rejects_a_non_adamw_step_and_restores_the_full_transaction() -> None:
    # RAdam shares the AdamW configuration and state keys but its first update
    # is not lr * mhat / (sqrt(vhat) + eps).
    _, named, optimizer, counter, transaction = _make_stack(
        optimizer_factory=torch.optim.RAdam
    )
    _assign(named, _DENSE_GRADIENTS)
    before_digest = transaction.state_digest()
    pre_hash = parameter_state_sha256(
        named, ParameterLayout.from_named_parameters(named)
    )

    with pytest.raises(ProposalCaptureError) as error:
        _capture(named, optimizer, transaction)

    assert error.value.disposition == "delta_reconstruction_uncertified"
    assert transaction.state_digest() == before_digest
    assert (
        parameter_state_sha256(named, ParameterLayout.from_named_parameters(named))
        == pre_hash
    )
    assert len(optimizer.state) == 0
    assert counter.value == 0


def test_capture_requires_the_frozen_config_a_fresh_optimizer_and_bound_layout() -> (
    None
):
    _, named, optimizer, _, transaction = _make_stack(learning_rate=1e-5)
    _assign(named, _DENSE_GRADIENTS)
    with pytest.raises(ProposalCaptureError) as error:
        capture_exact_adamw_proposal(
            named,
            optimizer=optimizer,
            transaction=transaction,
            config=AdamWProposalConfig.frozen(),
            binding=_binding(),
        )
    assert error.value.disposition == "optimizer_config_mismatch"

    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)
    _assign(named, _DENSE_GRADIENTS)
    with pytest.raises(ProposalCaptureError) as error:
        _capture(named, optimizer, transaction)
    assert error.value.disposition == "optimizer_is_not_fresh"

    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    with pytest.raises(ProposalCaptureError) as error:
        _capture(tuple(reversed(named)), optimizer, transaction)
    assert error.value.disposition == "parameter_layout_mismatch"

    _, named, optimizer, _, transaction = _make_stack()
    named[0][1].grad = None
    named[1][1].grad = _DENSE_GRADIENTS[_BIAS].clone()
    with pytest.raises(ProposalCaptureError) as error:
        _capture(named, optimizer, transaction)
    assert error.value.disposition == "missing_gradient"


# --- owner-wise projection ----------------------------------------------------


def test_projection_is_identity_when_no_witness_constraint_is_active() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    inactive = -_active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": inactive})

    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)

    assert receipt.disposition == "projected_certified"
    assert receipt.active_witnesses == ()
    assert receipt.multipliers == ()
    assert torch.equal(receipt.flat_projected_delta(), proposal.flat_delta())
    assert receipt.projected_delta_sha256 == proposal.delta_sha256
    assert receipt.correction_metric_norm == 0.0
    # delta_0 sits exactly on the trust-radius boundary by construction.
    assert receipt.projected_trust_radius_value == proposal.trust_radius
    assert receipt.trust_radius_active is True
    assert receipt.minimum_predicted_change > -WITNESS_FIRST_ORDER_TOLERANCE
    assert receipt.certified_first_order is True


def test_projection_one_active_constraint_matches_the_closed_form() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})

    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)

    delta0 = proposal.flat_delta()
    metric = proposal.flat_metric_denominator()
    scaled = row / metric
    multiplier = float(row @ delta0 + WITNESS_FIRST_ORDER_TOLERANCE) / float(
        scaled @ row
    )
    expected = delta0 - multiplier * scaled
    assert receipt.active_witnesses == (f"{SOURCE_MEMBERSHIPS[0]}|image_00|owner_a",)
    torch.testing.assert_close(
        receipt.flat_projected_delta(), expected, rtol=1e-12, atol=0.0
    )
    assert float(row @ receipt.flat_projected_delta()) == pytest.approx(
        -WITNESS_FIRST_ORDER_TOLERANCE, rel=1e-9
    )
    assert receipt.multipliers[0] > 0.0
    assert receipt.correction_metric_norm > 0.0
    assert receipt.projected_trust_radius_value < proposal.trust_radius
    assert receipt.trust_radius_active is False
    assert receipt.unprojected_changes[0].change < -WITNESS_FIRST_ORDER_TOLERANCE


def test_projection_multiple_active_constraints_are_permutation_invariant() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    # Disjoint supports keep both witnesses binding at the optimum.
    row_a = _active_row(proposal)
    row_a[4:] = 0.0
    row_b = _active_row(proposal, scale=1500.0)
    row_b[:4] = 0.0
    inactive = -_active_row(proposal, scale=250.0)
    rows = {"owner_a": row_a, "owner_b": row_b, "owner_c": inactive}

    forward = project_adamw_proposal(
        proposal=proposal,
        witness_bank=_bank(
            proposal.layout, rows, order=("owner_a", "owner_b", "owner_c")
        ),
    )
    permuted_bank = _bank(
        proposal.layout, rows, order=("owner_c", "owner_b", "owner_a")
    )
    permuted = project_adamw_proposal(proposal=proposal, witness_bank=permuted_bank)

    assert len(forward.active_witnesses) == 2
    assert forward.receipt_sha256 == permuted.receipt_sha256
    assert torch.equal(forward.flat_projected_delta(), permuted.flat_projected_delta())
    assert forward.to_dict() == permuted.to_dict()

    # Independent equality-constrained reference for the certified active set.
    metric = proposal.flat_metric_denominator()
    delta0 = proposal.flat_delta()
    matrix = torch.stack((row_a, row_b))
    scaled = matrix / metric
    lagrange = torch.linalg.solve(
        scaled @ matrix.T, matrix @ delta0 + WITNESS_FIRST_ORDER_TOLERANCE
    )
    expected = delta0 - scaled.T @ lagrange
    torch.testing.assert_close(
        forward.flat_projected_delta(), expected, rtol=1e-10, atol=0.0
    )
    assert all(multiplier > 0.0 for multiplier in forward.multipliers)
    assert forward.iterations <= forward.iteration_bound


def test_projection_uses_the_bias_corrected_metric_not_euclidean_or_squared() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    # Strongly anisotropic gradients make D, D^2 and D^-1 disagree.
    _assign(
        named,
        {
            _WEIGHT: torch.tensor(((1e-4, 1.0, 50.0), (2e-3, 8.0, 1e3))),
            _BIAS: torch.tensor((1e-5, 400.0)),
        },
    )
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    row[0] *= 40.0
    bank = _bank(proposal.layout, {"owner_a": row})

    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)

    delta0 = proposal.flat_delta()
    metric = proposal.flat_metric_denominator()
    offset = float(row @ delta0 + WITNESS_FIRST_ORDER_TOLERANCE)
    for candidate_metric in (metric, metric.square(), 1.0 / metric):
        scaled = row / candidate_metric
        candidate = delta0 - (offset / float(scaled @ row)) * scaled
        close = bool(
            torch.allclose(
                receipt.flat_projected_delta(), candidate, rtol=1e-9, atol=0.0
            )
        )
        assert close is (candidate_metric is metric)


def test_projection_certifies_the_trust_radius_and_fails_closed_on_a_breach() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    metric = proposal.flat_metric_denominator()
    delta0 = proposal.flat_delta()

    def rows():
        yield 0, row

    result = _solve_metric_projection(
        delta0,
        rows,
        metric=metric,
        tolerance=WITNESS_FIRST_ORDER_TOLERANCE,
        radius=proposal.trust_radius,
        max_iterations=16,
    )
    assert result.metric_radius_value <= proposal.trust_radius

    with pytest.raises(ProposalProjectionError) as error:
        _solve_metric_projection(
            delta0,
            rows,
            metric=metric,
            tolerance=WITNESS_FIRST_ORDER_TOLERANCE,
            radius=result.metric_radius_value * 0.5,
            max_iterations=16,
        )
    assert error.value.disposition == "trust_radius_breach"

    # A Euclidean radius is not the declared constraint.
    euclidean = float((delta0 * delta0).sum())
    assert not math.isclose(euclidean, proposal.trust_radius, rel_tol=1e-3)


def test_projection_feasible_set_always_contains_the_zero_delta() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _MIXED_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    generator = torch.Generator().manual_seed(5)
    for _ in range(8):
        row = torch.randn(
            proposal.layout.total_numel, generator=generator, dtype=torch.float64
        )
        row *= 1e4
        assert float(row @ torch.zeros_like(row)) >= -WITNESS_FIRST_ORDER_TOLERANCE
    zero = torch.zeros(proposal.layout.total_numel, dtype=torch.float64)
    assert float((proposal.flat_metric_denominator() * zero * zero).sum()) <= (
        proposal.trust_radius
    )


def test_projection_fails_closed_on_singular_dual_and_iteration_bound() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    metric = proposal.flat_metric_denominator()
    delta0 = proposal.flat_delta()
    row = _active_row(proposal)

    with pytest.raises(ProposalProjectionError) as error:
        _solve_working_set(
            delta0,
            (row, 2.0 * row),
            inverse_metric=1.0 / metric,
            tolerance=WITNESS_FIRST_ORDER_TOLERANCE,
        )
    assert error.value.disposition == "singular_dual_system"

    # The public screening order resolves that dependent pair without ever
    # forming the singular working set.
    dependent = _bank(proposal.layout, {"owner_a": row, "owner_b": 2.0 * row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=dependent)
    assert len(receipt.active_witnesses) == 1
    assert receipt.certified_first_order is True

    row_b = _active_row(proposal, scale=1500.0)
    row_b[:4] = 0.0

    def rows():
        yield 0, row
        yield 1, row_b

    with pytest.raises(ProposalProjectionError) as error:
        _solve_metric_projection(
            delta0,
            rows,
            metric=metric,
            tolerance=WITNESS_FIRST_ORDER_TOLERANCE,
            radius=proposal.trust_radius,
            max_iterations=1,
        )
    assert error.value.disposition == "projection_iteration_bound"


def test_projection_fails_closed_on_non_finite_jacobians_and_denominators() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    poisoned = row.clone()
    poisoned[2] = float("nan")
    bank = _bank(proposal.layout, {"owner_a": poisoned})

    with pytest.raises(ProposalProjectionError) as error:
        project_adamw_proposal(proposal=proposal, witness_bank=bank)
    assert error.value.disposition == "non_finite_jacobian"

    payload = proposal.to_dict()
    broken = json.loads(json.dumps(payload))
    with pytest.raises(ProposalAdmissionError):
        ExactAdamWProposal.from_dict(
            {**broken, "trust_radius": broken["trust_radius"] * 2.0}
        )


def test_frozen_witness_bank_membership_keeps_legacy_m_audit_only() -> None:
    _, named, optimizer, _, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    legacy = _witness(
        "owner_m",
        membership=LEGACY_M_MEMBERSHIP,
        owner_class=LEGACY_M_OWNER_CLASS,
        jacobian=None,
    )
    bank = _bank(proposal.layout, {"owner_a": row}, extra=(legacy,))

    assert tuple(witness.owner_id for witness in bank.constraints) == ("owner_a",)
    assert tuple(witness.owner_id for witness in bank.audit_only) == ("owner_m",)
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    assert len(receipt.predicted_changes) == 1
    assert receipt.audit_only_owner_count == 1

    with pytest.raises(ProposalAdmissionError):
        _bank(
            proposal.layout,
            {"owner_a": row},
            extra=(
                _witness(
                    "owner_m",
                    membership=SOURCE_MEMBERSHIPS[0],
                    owner_class=LEGACY_M_OWNER_CLASS,
                ),
            ),
        )
    with pytest.raises(ProposalAdmissionError):
        _bank(
            proposal.layout,
            {"owner_a": row},
            extra=(
                _witness(
                    "owner_m",
                    membership=LEGACY_M_MEMBERSHIP,
                    owner_class=LEGACY_M_OWNER_CLASS,
                    jacobian=row,
                ),
            ),
        )
    with pytest.raises(ProposalAdmissionError):
        _bank(
            proposal.layout,
            {"owner_a": row},
            extra=(_witness("owner_d", margin_selection="strongest_token_margin"),),
        )
    with pytest.raises(ProposalAdmissionError):
        _bank(proposal.layout, {"owner_a": row}, extra=(_witness("owner_e"),))


def test_single_aggregate_admission_governs_reload_projection_and_apply(
    tmp_path,
) -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})

    proposal_path = write_exact_adamw_proposal(proposal, tmp_path / "proposal.json")
    bank_directory = write_frozen_witness_bank(bank, tmp_path / "witnesses")
    reloaded_proposal = load_exact_adamw_proposal(proposal_path)
    reloaded_bank = load_frozen_witness_bank(bank_directory)
    assert reloaded_proposal.proposal_sha256 == proposal.proposal_sha256
    assert reloaded_bank.bank_sha256 == bank.bank_sha256
    evidence = admit_preservation_evidence(reloaded_proposal, reloaded_bank)
    assert (
        evidence.evidence_sha256
        == admit_preservation_evidence(proposal, bank).evidence_sha256
    )
    reloaded_receipt = project_adamw_proposal(
        proposal=reloaded_proposal, witness_bank=reloaded_bank
    )
    assert reloaded_receipt.receipt_sha256 == (
        project_adamw_proposal(proposal=proposal, witness_bank=bank).receipt_sha256
    )

    # A locally valid bank frozen from another Source cannot enter any path.
    foreign = _bank(
        proposal.layout,
        {"owner_a": row},
        binding=_witness_binding(source_checkpoint_sha256="d" * 64),
    )
    assert foreign.bank_sha256 != bank.bank_sha256
    with pytest.raises(ProposalAdmissionError):
        admit_preservation_evidence(proposal, foreign)
    with pytest.raises(ProposalAdmissionError):
        project_adamw_proposal(proposal=proposal, witness_bank=foreign)
    with pytest.raises(ProposalAdmissionError):
        apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=foreign,
            projection=reloaded_receipt,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {}),
        )

    # Equal-length Jacobians frozen against another parameter ordering are
    # refused by the same choke point.
    permuted_layout = ParameterLayout(entries=tuple(reversed(proposal.layout.entries)))
    permuted = _bank(permuted_layout, {"owner_a": row})
    assert permuted_layout.total_numel == proposal.layout.total_numel
    with pytest.raises(ProposalAdmissionError) as error:
        admit_preservation_evidence(proposal, permuted)
    assert error.value.disposition == "layout_mismatch"

    # A jacobian whose bytes no longer match its frozen digest is rejected.
    tampered = (
        bank_directory / "jacobians" / f"{bank.constraints[0].jacobian_sha256}.f64"
    )
    tampered.write_bytes((row * 1.5).numpy().tobytes())
    with pytest.raises(ProposalAdmissionError):
        project_adamw_proposal(
            proposal=proposal, witness_bank=load_frozen_witness_bank(bank_directory)
        )


@pytest.mark.parametrize(
    ("injected", "expected_error"),
    (
        (OSError(errno.ENOSPC, "injected staging ENOSPC"), ProposalAdmissionError),
        (KeyboardInterrupt("injected staging interruption"), KeyboardInterrupt),
    ),
)
def test_atomic_file_publication_never_exposes_a_failed_staging_write(
    tmp_path, injected: BaseException, expected_error: type[BaseException]
) -> None:
    _, named, optimizer, _counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    target = tmp_path / f"{proposal.proposal_sha256}.json"
    filesystem = _FailingPublicationFileSystem(write_number=1, error=injected)

    with pytest.raises(expected_error):
        write_exact_adamw_proposal(proposal, target, filesystem=filesystem)

    assert not target.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_atomic_witness_bank_publication_never_exposes_a_partial_directory(
    tmp_path,
) -> None:
    _, named, optimizer, _counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    bank = _bank(proposal.layout, {"owner_a": _active_row(proposal)})
    target = tmp_path / bank.bank_sha256
    filesystem = _FailingPublicationFileSystem(
        write_number=2,
        error=OSError(errno.EIO, "injected bank write failure"),
    )

    with pytest.raises(ProposalAdmissionError, match="publication did not complete"):
        write_frozen_witness_bank(bank, target, filesystem=filesystem)

    assert not target.exists()
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize("fsync_number", (1, 2, 3))
def test_atomic_witness_bank_fsyncs_nested_root_and_parent_directories(
    tmp_path, fsync_number: int
) -> None:
    _, named, optimizer, _counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    bank = _bank(proposal.layout, {"owner_a": _active_row(proposal)})
    target = tmp_path / bank.bank_sha256
    filesystem = _FailingDirectoryFsyncPublicationFileSystem(fsync_number=fsync_number)

    with pytest.raises(ProposalAdmissionError, match="publication did not complete"):
        write_frozen_witness_bank(bank, target, filesystem=filesystem)

    assert not target.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_identical_existing_witness_bank_requires_full_stream_reload(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, named, optimizer, _counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    bank = _bank(proposal.layout, {"owner_a": _active_row(proposal)})
    target = write_frozen_witness_bank(bank, tmp_path / bank.bank_sha256)
    assert write_frozen_witness_bank(bank, target) == target

    class ReloadedMetadataOnly:
        def to_dict(self):
            return bank.to_dict()

        def stream_constraints(self):
            raise ProposalAdmissionError(
                "injected incomplete Jacobian reload",
                disposition="invalid_artifact",
            )

    monkeypatch.setattr(
        preservation,
        "load_frozen_witness_bank",
        lambda _path: ReloadedMetadataOnly(),
    )

    with pytest.raises(ProposalAdmissionError, match="incomplete Jacobian reload"):
        write_frozen_witness_bank(bank, target)

    assert target.is_dir()


def test_projection_and_apply_artifacts_reload_and_reject_tampering(tmp_path) -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})
    projection = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    applied = apply_projected_delta(
        named,
        proposal=proposal,
        witness_bank=bank,
        projection=projection,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {"owner_a": 0.25}),
    )

    projection_path = write_projection_receipt(projection, tmp_path / "projection.json")
    apply_path = write_projected_apply_receipt(applied, tmp_path / "apply.json")
    assert (
        load_projection_receipt(
            projection_path, expected_sha256=projection.receipt_sha256
        ).to_dict()
        == projection.to_dict()
    )
    assert (
        load_projected_apply_receipt(
            apply_path, expected_sha256=applied.receipt_sha256
        ).to_dict()
        == applied.to_dict()
    )

    projection_payload = json.loads(projection_path.read_text(encoding="utf-8"))
    projection_payload["projected_delta_sha256"] = "0" * 64
    tampered_projection = tmp_path / "tampered-projection.json"
    tampered_projection.write_text(json.dumps(projection_payload), encoding="utf-8")
    with pytest.raises(ProposalAdmissionError, match="content address"):
        load_projection_receipt(tampered_projection)

    apply_payload = json.loads(apply_path.read_text(encoding="utf-8"))
    apply_payload["realized_minimum_change"] = 123.0
    tampered_apply = tmp_path / "tampered-apply.json"
    tampered_apply.write_text(json.dumps(apply_payload), encoding="utf-8")
    with pytest.raises(ProposalAdmissionError, match="durable receipt binding"):
        load_projected_apply_receipt(
            tampered_apply, expected_sha256=applied.receipt_sha256
        )


def test_apply_rehashes_the_actual_parameter_change_and_rejects_wrong_deltas() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    pre_hash = parameter_state_sha256(named, proposal.layout)

    # Applying the unprojected delta through a foreign but internally valid
    # receipt is refused before any parameter changes.
    empty_bank = _bank(proposal.layout, {"owner_a": -row})
    unprojected = project_adamw_proposal(proposal=proposal, witness_bank=empty_bank)
    assert torch.equal(unprojected.flat_projected_delta(), proposal.flat_delta())
    with pytest.raises(ProjectedApplyError) as error:
        apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=bank,
            projection=unprojected,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {}),
        )
    assert error.value.disposition == "projection_evidence_mismatch"
    assert parameter_state_sha256(named, proposal.layout) == pre_hash

    # An in-place rounded copy of the certified delta breaks its digest.
    rounded = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    rounded.projected_delta[0][0] = float(
        torch.round(rounded.projected_delta[0][0] * 1e6) / 1e6
    )
    with pytest.raises(ProjectedApplyError) as error:
        apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=bank,
            projection=rounded,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {}),
        )
    assert error.value.disposition == "projected_delta_digest_mismatch"
    assert parameter_state_sha256(named, proposal.layout) == pre_hash
    assert counter.value == 0

    applied = apply_projected_delta(
        named,
        proposal=proposal,
        witness_bank=bank,
        projection=receipt,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {"owner_a": 0.125}),
    )

    assert applied.disposition == "applied_certified"
    assert applied.pre_parameter_sha256 == pre_hash
    assert applied.post_parameter_sha256 == parameter_state_sha256(
        named, proposal.layout
    )
    assert applied.post_parameter_sha256 != proposal.post_parameter_sha256
    assert applied.applied_delta_sha256 != proposal.delta_sha256
    assert applied.realized_witness_violation is False
    assert applied.eligible_for_behavioral_audits is True
    assert applied.update_count_before == 0 and applied.update_count_after == 1
    assert counter.value == 1
    assert len(optimizer.state) == 0
    realized = torch.cat(
        [
            (parameter.detach().to(torch.float64) - saved.to(torch.float64)).reshape(-1)
            for (_, parameter), saved in zip(
                named,
                [
                    tensor.reshape(entry.shape)
                    for tensor, entry in zip(proposal.delta, proposal.layout.entries)
                ],
            )
        ]
    )
    assert realized.shape == receipt.flat_projected_delta().shape

    # Applying a second time is refused: the Source parameter hash no longer
    # matches the captured proposal.
    with pytest.raises(ProjectedApplyError) as error:
        apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=bank,
            projection=receipt,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {}),
        )
    assert error.value.disposition == "source_parameter_mismatch"


def test_apply_requires_the_transaction_that_owns_the_live_parameters() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    pre_hash = parameter_state_sha256(named, proposal.layout)

    # A transaction bound to another stack cannot roll this apply back.
    _, _, _, _, foreign_transaction = _make_stack()
    with pytest.raises(ProjectedApplyError) as error:
        apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=bank,
            projection=receipt,
            optimizer=optimizer,
            transaction=foreign_transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {}),
        )

    assert error.value.disposition == "transaction_binding_uncertified"
    assert parameter_state_sha256(named, proposal.layout) == pre_hash
    assert counter.value == 0


def _forge_receipt(
    genuine, proposal: ExactAdamWProposal, rows: Mapping[str, torch.Tensor], delta
):
    """Build a fully self-consistent same-evidence receipt for ``delta``."""

    layout = proposal.layout
    pieces = []
    offset = 0
    for entry in layout.entries:
        pieces.append(delta[offset : offset + entry.numel].clone())
        offset += entry.numel
    metric = proposal.flat_metric_denominator()
    correction = delta - proposal.flat_delta()
    objective = float((metric * correction * correction).sum())
    predicted = tuple(
        dataclasses.replace(item, change=float(rows[item.owner_id] @ delta))
        for item in genuine.predicted_changes
    )
    return dataclasses.replace(
        genuine,
        projected_delta=tuple(pieces),
        projected_delta_sha256=_hash_flat_tensors(
            tuple(pieces), layout=layout, context="delta"
        ),
        active_witnesses=(),
        multipliers=(),
        objective_value=objective,
        correction_metric_norm=math.sqrt(objective),
        projected_trust_radius_value=float((metric * delta * delta).sum()),
        trust_radius_active=False,
        predicted_changes=predicted,
        minimum_predicted_change=min(item.change for item in predicted),
    )


def test_apply_rejects_a_self_consistent_non_optimal_projection_receipt() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    rows = {"owner_a": row}
    bank = _bank(proposal.layout, rows)
    genuine = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    pre_hash = parameter_state_sha256(named, proposal.layout)
    certified = genuine.flat_projected_delta()

    # Both alternates are feasible for every witness and inside the trust
    # radius; neither is the minimum-change projection.
    zero = torch.zeros_like(certified)
    halved = 0.5 * certified
    for alternate in (zero, halved):
        assert float(row @ alternate) >= -WITNESS_FIRST_ORDER_TOLERANCE
        assert (
            float((proposal.flat_metric_denominator() * alternate * alternate).sum())
            <= proposal.trust_radius
        )
        forged = _forge_receipt(genuine, proposal, rows, alternate)
        assert forged.evidence_sha256 == genuine.evidence_sha256
        assert forged.disposition == "projected_certified"
        assert (
            _hash_flat_tensors(
                forged.projected_delta, layout=proposal.layout, context="delta"
            )
            == forged.projected_delta_sha256
        )
        with pytest.raises(ProjectedApplyError) as error:
            apply_projected_delta(
                named,
                proposal=proposal,
                witness_bank=bank,
                projection=forged,
                optimizer=optimizer,
                transaction=transaction,
                update_counter=counter,
                realized_margin_probe=_probe(bank, {}),
            )
        assert error.value.disposition == "projection_not_canonical"
        assert parameter_state_sha256(named, proposal.layout) == pre_hash
        assert counter.value == 0
        assert len(optimizer.state) == 0


def test_capture_requires_the_transaction_that_owns_the_live_stack() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    _, foreign_named, foreign_optimizer, foreign_counter, foreign_transaction = (
        _make_stack()
    )
    _assign(foreign_named, _DENSE_GRADIENTS)
    # The foreign stack is value-identical: only object ownership differs.
    assert parameter_state_sha256(
        named, ParameterLayout.from_named_parameters(named)
    ) == parameter_state_sha256(
        foreign_named, ParameterLayout.from_named_parameters(foreign_named)
    )
    digest = transaction.state_digest()
    foreign_digest = foreign_transaction.state_digest()
    rng = torch.get_rng_state().clone()

    with pytest.raises(ProposalCaptureError) as error:
        capture_exact_adamw_proposal(
            named,
            optimizer=optimizer,
            transaction=foreign_transaction,
            config=AdamWProposalConfig.frozen(),
            binding=_binding(),
        )
    assert error.value.disposition == "transaction_binding_uncertified"

    # An optimizer that is not the transaction's own optimizer is refused too,
    # even though it holds exactly the bound parameters.
    detached_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=FROZEN_LEARNING_RATE,
        betas=FROZEN_BETAS,
        eps=FROZEN_EPSILON,
        weight_decay=FROZEN_WEIGHT_DECAY,
    )
    with pytest.raises(ProposalCaptureError) as error:
        capture_exact_adamw_proposal(
            named,
            optimizer=detached_optimizer,
            transaction=transaction,
            config=AdamWProposalConfig.frozen(),
            binding=_binding(),
        )
    assert error.value.disposition == "transaction_binding_uncertified"

    assert transaction.state_digest() == digest
    assert foreign_transaction.state_digest() == foreign_digest
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(optimizer.state) == 0
    assert len(foreign_optimizer.state) == 0
    assert len(detached_optimizer.state) == 0
    assert counter.value == 0 and foreign_counter.value == 0
    # Neither transaction was ever opened.
    for candidate in (transaction, foreign_transaction):
        candidate.reject(candidate.begin())


def test_apply_certifies_the_physical_parameter_change() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    with torch.no_grad():
        named[1][1].fill_(1e5)
    _assign(
        named,
        {
            _WEIGHT: torch.tensor(((0.5, -0.25, 2.0), (1.0, 4.0, -7.5))),
            _BIAS: torch.tensor((1e-4, -1e-4)),
        },
    )
    proposal = _capture(named, optimizer, transaction)
    delta0 = proposal.flat_delta()
    # The 1e5-valued parameters cannot even represent their own AdamW step.
    assert float(delta0[6:].abs().max()) == 0.0
    row = torch.zeros(proposal.layout.total_numel, dtype=torch.float64)
    row[:6] = -1000.0 * torch.sign(delta0[:6])
    row[6:] = 10.0
    bank = _bank(proposal.layout, {"owner_a": row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    pre_hash = parameter_state_sha256(named, proposal.layout)
    # The certified projection puts a materially sized correction on the
    # unrepresentable coordinates.
    assert float(receipt.flat_projected_delta()[6:].abs().min()) > 1e-5

    with pytest.raises(ProjectedApplyError) as error:
        apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=bank,
            projection=receipt,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {}),
        )

    assert error.value.disposition == "applied_delta_mismatch"
    assert parameter_state_sha256(named, proposal.layout) == pre_hash
    assert counter.value == 0

    # An honest float32 apply stays far inside the declared realization
    # allowance and hashes the physical change, not the intended one.
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    pre_values = tuple(parameter.detach().clone() for _, parameter in named)
    applied = apply_projected_delta(
        named,
        proposal=proposal,
        witness_bank=bank,
        projection=receipt,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {}),
    )
    physical = tuple(
        (parameter.detach() - saved).to(torch.float64).reshape(-1)
        for (_, parameter), saved in zip(named, pre_values)
    )
    assert applied.applied_delta_sha256 == _hash_flat_tensors(
        physical, layout=proposal.layout, context="delta"
    )
    assert (
        0.0
        < applied.applied_representation_error
        <= (applied.applied_representation_allowance)
    )
    assert applied.applied_undeliverable_dose == 0.0
    assert applied.applied_minimum_first_order_change >= (
        -WITNESS_FIRST_ORDER_TOLERANCE * 1.000001
    )

    # A proposal whose exact step is zero must physically change nothing.
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _ZERO_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    assert proposal.trust_radius == 0.0
    bank = _bank(proposal.layout, {"owner_a": torch.ones(8, dtype=torch.float64)})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    applied = apply_projected_delta(
        named,
        proposal=proposal,
        witness_bank=bank,
        projection=receipt,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {}),
    )
    assert applied.applied_representation_error == 0.0
    assert applied.applied_undeliverable_dose == 0.0
    assert applied.post_parameter_sha256 == applied.pre_parameter_sha256


def _underdosed_stack(scale: float, jacobian_scale: float):
    """A stack whose largest parameters cannot represent their own update."""

    _, named, optimizer, counter, transaction = _make_stack()
    with torch.no_grad():
        named[1][1].fill_(1e5)
    _assign(
        named,
        {
            _WEIGHT: torch.tensor(((0.5, -0.25, 2.0), (1.0, 4.0, -7.5))),
            _BIAS: torch.tensor((1e-4, -1e-4)),
        },
    )
    proposal = _capture(named, optimizer, transaction)
    delta0 = proposal.flat_delta()
    row = torch.zeros(proposal.layout.total_numel, dtype=torch.float64)
    row[:6] = -scale * torch.sign(delta0[:6])
    row[6:] = jacobian_scale
    bank = _bank(proposal.layout, {"owner_a": row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    return named, optimizer, counter, transaction, proposal, bank, receipt


def test_physical_apply_allowance_is_derived_from_the_representation() -> None:
    # A ten percent under-dose: the old fixed 0.1 relative allowance admitted
    # this apply even though a tenth of the certified update never reached the
    # parameters.
    named, optimizer, counter, transaction, proposal, bank, receipt = _underdosed_stack(
        100.0, 10.0
    )
    intended = receipt.flat_projected_delta()
    metric = proposal.flat_metric_denominator()
    pre = torch.cat([parameter.detach().reshape(-1) for _, parameter in named])
    physical = ((pre + intended.to(torch.float32)) - pre).to(torch.float64)
    error = float(
        (metric * (physical - intended) * (physical - intended)).sum()
    ) / float((metric * intended * intended).sum())
    assert 0.05 < error**0.5 < 0.1, "counterexample must pass a fixed 0.1 rule"
    # The lost dose sits on coordinates whose certified correction is below
    # half an ulp of their own parameter, so it is the deliverability gate and
    # not the rounding gate that must own this rejection: the physical change
    # is exactly what fp32 addition can do here.
    pre_bias = named[1][1].detach().reshape(-1).to(torch.float64)
    assert float(intended[6:].abs().max()) < float(
        0.5 * _representation_ulp(pre_bias, torch.float32).min()
    )
    assert torch.equal(physical[6:], torch.zeros_like(physical[6:]))
    pre_hash = parameter_state_sha256(named, proposal.layout)

    with pytest.raises(ProjectedApplyError) as failure:
        apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=bank,
            projection=receipt,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {}),
        )
    assert failure.value.disposition == "applied_delta_mismatch"
    assert parameter_state_sha256(named, proposal.layout) == pre_hash
    assert counter.value == 0

    # A bfloat16 surface cannot represent a 3e-6 step at all, so the exact
    # proposal is a certified zero step and its apply changes nothing.
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    parameters = dict(model.named_parameters())
    wide = ((_WEIGHT, parameters["weight"]), (_BIAS, parameters["bias"]))
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in wide],
        lr=FROZEN_LEARNING_RATE,
        betas=FROZEN_BETAS,
        eps=FROZEN_EPSILON,
        weight_decay=FROZEN_WEIGHT_DECAY,
    )
    counter = UpdateCounter()
    transaction = TrainingStateTransaction(
        wide,
        optimizer=optimizer,
        scheduler=None,
        update_counter=counter,
        runtime=None,
        capture_cuda=False,
    )
    _assign(
        wide,
        {
            _WEIGHT: _DENSE_GRADIENTS[_WEIGHT].to(torch.bfloat16),
            _BIAS: _DENSE_GRADIENTS[_BIAS].to(torch.bfloat16),
        },
    )
    proposal = _capture(wide, optimizer, transaction)
    assert float(proposal.flat_delta().abs().max()) == 0.0
    bank = _bank(proposal.layout, {"owner_a": torch.ones(8, dtype=torch.float64)})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
    applied = apply_projected_delta(
        wide,
        proposal=proposal,
        witness_bank=bank,
        projection=receipt,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {}),
    )
    assert applied.applied_representation_error == 0.0
    assert applied.applied_undeliverable_dose == 0.0
    assert applied.post_parameter_sha256 == applied.pre_parameter_sha256


def test_apply_labels_any_finite_negative_realized_change_as_a_violation() -> None:
    for change, violated in ((-1e-5, True), (0.0, False), (0.25, False)):
        _, named, optimizer, counter, transaction = _make_stack()
        _assign(named, _DENSE_GRADIENTS)
        proposal = _capture(named, optimizer, transaction)
        row = _active_row(proposal)
        bank = _bank(proposal.layout, {"owner_a": row})
        receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)

        applied = apply_projected_delta(
            named,
            proposal=proposal,
            witness_bank=bank,
            projection=receipt,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=counter,
            realized_margin_probe=_probe(bank, {"owner_a": change}),
        )

        assert applied.realized_witness_violation is violated
        assert applied.eligible_for_behavioral_audits is True
        assert applied.realized_minimum_change == pytest.approx(change)
        if violated:
            assert applied.disposition == "applied_with_realized_witness_violation"
            assert applied.degraded_witnesses == (bank.constraints[0].canonical_key,)
        else:
            assert applied.disposition == "applied_certified"
            assert applied.degraded_witnesses == ()


def test_apply_records_realized_witness_violation_and_stays_audit_eligible() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row, "owner_b": -row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)

    applied = apply_projected_delta(
        named,
        proposal=proposal,
        witness_bank=bank,
        projection=receipt,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {"owner_a": -0.5, "owner_b": 0.25}),
    )

    assert applied.disposition == "applied_with_realized_witness_violation"
    assert applied.realized_witness_violation is True
    assert applied.degraded_witnesses == (f"{SOURCE_MEMBERSHIPS[0]}|image_00|owner_a",)
    assert applied.realized_minimum_change == pytest.approx(-0.5)
    assert applied.eligible_for_behavioral_audits is True
    assert counter.value == 1


def test_apply_fails_closed_on_missing_or_non_finite_realized_measurement() -> None:
    for changes, missing in ((None, True), ({"owner_a": float("nan")}, False)):
        _, named, optimizer, counter, transaction = _make_stack()
        _assign(named, _DENSE_GRADIENTS)
        proposal = _capture(named, optimizer, transaction)
        row = _active_row(proposal)
        bank = _bank(proposal.layout, {"owner_a": row})
        receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)
        pre_hash = parameter_state_sha256(named, proposal.layout)

        def probe() -> Mapping[str, float]:
            if missing:
                return {}
            return {bank.constraints[0].canonical_key: float("nan")}

        with pytest.raises(ProjectedApplyError) as error:
            apply_projected_delta(
                named,
                proposal=proposal,
                witness_bank=bank,
                projection=receipt,
                optimizer=optimizer,
                transaction=transaction,
                update_counter=counter,
                realized_margin_probe=probe,
            )
        assert error.value.disposition == "realized_measurement_uncertified"
        assert parameter_state_sha256(named, proposal.layout) == pre_hash
        assert counter.value == 0


def test_no_projected_optimizer_moment_or_continuation_handle_is_emitted() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})
    receipt = project_adamw_proposal(proposal=proposal, witness_bank=bank)

    forbidden = ("exp_avg", "moment", "state_dict", "continue", "resume")
    for payload in (proposal.to_dict(), receipt.to_dict()):
        serialized = json.dumps(payload)
        for token in forbidden:
            assert token not in serialized
    for record in (proposal, receipt):
        for attribute in dir(record):
            assert "exp_avg" not in attribute
            assert "moment" not in attribute

    applied = apply_projected_delta(
        named,
        proposal=proposal,
        witness_bank=bank,
        projection=receipt,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {}),
    )
    assert applied.optimizer_state_entries == 0
    assert len(optimizer.state) == 0

    # A live optimizer that already carries moments cannot host a projected
    # apply at all.
    _, other_named, other_optimizer, other_counter, other_transaction = _make_stack()
    _assign(other_named, _DENSE_GRADIENTS)
    other_optimizer.step()
    with pytest.raises(ProjectedApplyError) as error:
        apply_projected_delta(
            other_named,
            proposal=proposal,
            witness_bank=bank,
            projection=receipt,
            optimizer=other_optimizer,
            transaction=other_transaction,
            update_counter=other_counter,
            realized_margin_probe=_probe(bank, {}),
        )
    assert error.value.disposition == "optimizer_moment_continuation"


def test_toy_end_to_end_unprojected_projected_apply_and_rollback() -> None:
    _, named, optimizer, counter, transaction = _make_stack()
    _assign(named, _DENSE_GRADIENTS)
    source_digest = transaction.state_digest()
    source_hash = parameter_state_sha256(
        named, ParameterLayout.from_named_parameters(named)
    )

    proposal = _capture(named, optimizer, transaction)
    row = _active_row(proposal)
    bank = _bank(proposal.layout, {"owner_a": row})
    projection = project_adamw_proposal(proposal=proposal, witness_bank=bank)

    # Arm A/B: the unprojected proposal really is one accepted AdamW step.
    snapshot = transaction.begin()
    optimizer.step()
    counter.value += 1
    assert parameter_state_sha256(named, proposal.layout) == (
        proposal.post_parameter_sha256
    )
    assert len(optimizer.state) == 2
    transaction.reject(snapshot)
    assert transaction.state_digest() == source_digest
    assert parameter_state_sha256(named, proposal.layout) == source_hash
    assert counter.value == 0
    assert len(optimizer.state) == 0

    # Arm C: the projected delta is applied manually inside the transaction.
    snapshot = transaction.begin()
    applied = apply_projected_delta(
        named,
        proposal=proposal,
        witness_bank=bank,
        projection=projection,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        realized_margin_probe=_probe(bank, {"owner_a": 0.0625}),
    )
    assert applied.post_parameter_sha256 == parameter_state_sha256(
        named, proposal.layout
    )
    assert applied.post_parameter_sha256 not in {
        source_hash,
        proposal.post_parameter_sha256,
    }
    assert counter.value == 1
    assert len(optimizer.state) == 0
    assert transaction.state_digest() != source_digest

    receipt = transaction.reject(snapshot)
    assert receipt.decision == "rejected_restored"
    assert receipt.after_state_digest == source_digest
    assert transaction.state_digest() == source_digest
    assert parameter_state_sha256(named, proposal.layout) == source_hash
    assert counter.value == 0
    assert len(optimizer.state) == 0
