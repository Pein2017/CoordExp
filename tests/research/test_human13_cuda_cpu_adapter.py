from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import importlib.util
import sys
from typing import Any, cast

import pytest
import torch

import scripts.research.human13_greedy_compiler as compiler_owner
from scripts.research.human13_adamw_proposal_preservation import (
    FROZEN_BETAS,
    FROZEN_EPSILON,
    FROZEN_LEARNING_RATE,
    FROZEN_WEIGHT_DECAY,
    FrozenWitnessBank,
    OwnerWitness,
    ParameterLayout,
    ProposalBinding,
    WitnessBinding,
    jacobian_sha256,
)
from scripts.research.human13_training_transaction import (
    TrainingStateSnapshot,
    TrainingStateTransaction,
    TransactionReceipt,
    UpdateCounter,
)
from scripts.research.human13_cuda_cpu_adapter import (
    CudaAdapterError,
    CudaAdapterRollbackError,
    CudaProposalInput,
    CudaHFVerticalAdapter,
    compute_cuda_objective_binding_sha256,
    require_admitted_receipt,
)


class _FrozenLinear(torch.nn.Linear):
    frozen: torch.nn.Parameter
    frozen_buffer: torch.Tensor

    def __init__(self, *, device: str) -> None:
        super().__init__(1, 1, bias=False, device=device)
        self.register_buffer("frozen_buffer", torch.tensor([3.0], device=device))
        self.frozen = torch.nn.Parameter(
            torch.tensor([2.0], device=device), requires_grad=False
        )


def _proposal_binding() -> ProposalBinding:
    return ProposalBinding(
        unit_id="2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical",
        arm_id="C-One-Image",
        training_rp="1.0",
        seed_group="35001..35016",
        source_checkpoint_sha256="a" * 64,
        manifest_sha256="b" * 64,
        objective_ledger_sha256="c" * 64,
    )


def _witness_bank(named: tuple[tuple[str, torch.nn.Parameter], ...]) -> FrozenWitnessBank:
    layout = ParameterLayout.from_named_parameters(named)
    jacobian = torch.ones(layout.total_numel, dtype=torch.float64)
    witness = OwnerWitness(
        image_id="1584",
        owner_id="g-1",
        source_membership="u_intersect_s_1.0",
        owner_class="source_emitted_trusted_owner",
        token_id=1,
        margin_selection="weakest_detached_processed_logit_token_margin",
        margin_value=0.0,
        detached=True,
        jacobian_sha256=jacobian_sha256(jacobian),
    )
    return FrozenWitnessBank.from_witnesses(
        (witness,),
        jacobians={witness.canonical_key: jacobian},
        binding=WitnessBinding(
            unit_id=_proposal_binding().unit_id,
            source_checkpoint_sha256="a" * 64,
            manifest_sha256="b" * 64,
            frozen_before_acquisition=True,
        ),
        layout=layout,
    )


def _surface(*, device: str = "cpu", capture_cuda: bool = True) -> CudaProposalInput:
    model = _FrozenLinear(device=device)
    model.eval()
    with torch.no_grad():
        model.weight.fill_(1.0)
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=FROZEN_LEARNING_RATE,
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
        capture_cuda=capture_cuda,
    )
    objective = (model.weight.square()).sum()
    return CudaProposalInput(
        model=model,
        named_trainable_parameters=named,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        objective=objective,
        witness_bank=_witness_bank(named),
        proposal_binding=_proposal_binding(),
        realized_margin_probe=lambda: {"u_intersect_s_1.0|1584|g-1": 0.0},
    )


def _task2_surface(*, module_name: str) -> tuple[CudaProposalInput, Any]:
    fixture_path = Path(__file__).with_name("test_human13_all_hf_vertical.py")
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    if spec is None or spec.loader is None:
        raise AssertionError("vertical fixture module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fixture = module._fixture(dtype=torch.bfloat16)
    binding = ProposalBinding(
        unit_id=_proposal_binding().unit_id,
        arm_id=_proposal_binding().arm_id,
        training_rp="1.0",
        seed_group="35001..35016",
        source_checkpoint_sha256=fixture.trajectory_ledger.source_sha256,
        manifest_sha256=fixture.trajectory_ledger.manifest_sha256,
        objective_ledger_sha256="d" * 64,
    )
    surface = CudaProposalInput(
        model=fixture.model,
        named_trainable_parameters=fixture.named,
        optimizer=fixture.optimizer,
        transaction=fixture.transaction,
        update_counter=fixture.counter,
        objective=None,
        witness_bank=fixture.witness_bank,
        proposal_binding=binding,
        realized_margin_probe=lambda: {
            witness.canonical_key: witness.margin_value
            for witness in fixture.witness_bank.constraints
        },
        surface_identity=fixture.sampled_groups[0].identity,
        sampled_groups=fixture.sampled_groups,
        replay_groups=tuple(fixture.replay_groups),
        replay_logprob_tensors=fixture.replay_tensors,
        trajectory_ledger=fixture.trajectory_ledger,
        compiler_ledger=fixture.compiler_ledger,
        compiler_compact_logits=fixture.compact_logits,
    )
    surface = replace(
        surface,
        proposal_binding=replace(
            binding, objective_ledger_sha256=compute_cuda_objective_binding_sha256(surface)
        ),
    )
    return surface, fixture


def test_cuda_adapter_runs_one_update_and_exact_rollback_on_cpu_injected_surface() -> None:
    surface = _surface()
    before = tuple(parameter.detach().clone() for _, parameter in surface.named_trainable_parameters)
    receipt = CudaHFVerticalAdapter(surface).apply_and_rollback()

    assert receipt.status == "applied_and_rolled_back"
    assert receipt.update_count_before == 0
    assert receipt.update_count_after == 1
    assert receipt.rollback_decision == "rejected_restored"
    assert receipt.source_parameter_sha256 == receipt.restored_parameter_sha256
    assert receipt.source_state_digest == receipt.restored_state_digest
    assert receipt.source_version_counters == receipt.restored_version_counters
    assert receipt.source_version_sha256 == receipt.restored_version_sha256
    assert receipt.source_cuda_rng_sha256 is None
    assert require_admitted_receipt(receipt) is receipt
    forged = replace(receipt, applied_parameter_sha256="0" * 64)
    with pytest.raises(CudaAdapterError, match="not issued"):
        require_admitted_receipt(forged)
    assert all(
        torch.equal(parameter.detach(), saved)
        for (_, parameter), saved in zip(surface.named_trainable_parameters, before, strict=True)
    )
    assert surface.update_counter.value == 0
    assert not surface.optimizer.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_adapter_runs_one_update_and_exact_rollback_on_cuda_surface() -> None:
    surface = _surface(device="cuda")
    surface = replace(
        surface,
        realized_margin_probe=lambda: (
            torch.rand(1, device="cuda"),
            {"u_intersect_s_1.0|1584|g-1": 0.0},
        )[1],
    )
    receipt = CudaHFVerticalAdapter(surface).apply_and_rollback()

    assert receipt.device.startswith("cuda")
    assert receipt.status == "applied_and_rolled_back"
    assert receipt.source_parameter_sha256 == receipt.restored_parameter_sha256
    assert receipt.source_cuda_rng_sha256 == receipt.restored_cuda_rng_sha256
    assert receipt.source_state_digest == receipt.restored_state_digest
    assert surface.update_counter.value == 0
    assert not surface.optimizer.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_adapter_rejects_transaction_without_cuda_rng_capture() -> None:
    surface = _surface(device="cuda", capture_cuda=False)
    with pytest.raises(CudaAdapterError, match="capture CUDA RNG"):
        CudaHFVerticalAdapter(surface)


def test_cuda_adapter_rejects_foreign_objective_graph_before_transaction() -> None:
    surface = _surface()
    foreign = torch.nn.Parameter(torch.ones(1))
    foreign_objective = foreign.square().sum()
    invalid = CudaProposalInput(
        model=surface.model,
        named_trainable_parameters=surface.named_trainable_parameters,
        optimizer=surface.optimizer,
        transaction=surface.transaction,
        update_counter=surface.update_counter,
        objective=foreign_objective,
        witness_bank=surface.witness_bank,
        proposal_binding=surface.proposal_binding,
        realized_margin_probe=surface.realized_margin_probe,
    )
    with pytest.raises(CudaAdapterError, match="graph owner"):
        CudaHFVerticalAdapter(invalid)
    assert surface.update_counter.value == 0
    assert not surface.optimizer.state


def test_cuda_adapter_rejects_optimizer_or_transaction_substitution() -> None:
    surface = _surface()
    foreign_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in surface.named_trainable_parameters],
        lr=FROZEN_LEARNING_RATE,
        betas=FROZEN_BETAS,
        eps=FROZEN_EPSILON,
        weight_decay=FROZEN_WEIGHT_DECAY,
    )
    invalid = CudaProposalInput(
        model=surface.model,
        named_trainable_parameters=surface.named_trainable_parameters,
        optimizer=foreign_optimizer,
        transaction=surface.transaction,
        update_counter=surface.update_counter,
        objective=surface.objective,
        witness_bank=surface.witness_bank,
        proposal_binding=surface.proposal_binding,
        realized_margin_probe=surface.realized_margin_probe,
    )
    with pytest.raises(CudaAdapterError, match="optimizer"):
        CudaHFVerticalAdapter(invalid)


def test_cuda_adapter_restores_frozen_full_model_values_after_probe() -> None:
    surface = _surface()
    frozen = cast(torch.nn.Parameter, getattr(surface.model, "frozen"))
    frozen_buffer = cast(torch.Tensor, getattr(surface.model, "frozen_buffer"))
    source = frozen.detach().clone()

    def mutating_probe() -> dict[str, float]:
        frozen.add_(1.0)
        frozen_buffer.add_(1.0)
        surface.model.train()
        return {"u_intersect_s_1.0|1584|g-1": 0.0}

    mutated = CudaProposalInput(
        model=surface.model,
        named_trainable_parameters=surface.named_trainable_parameters,
        optimizer=surface.optimizer,
        transaction=surface.transaction,
        update_counter=surface.update_counter,
        objective=surface.objective,
        witness_bank=surface.witness_bank,
        proposal_binding=surface.proposal_binding,
        realized_margin_probe=mutating_probe,
    )
    receipt = CudaHFVerticalAdapter(mutated).apply_and_rollback()

    assert torch.equal(frozen.detach(), source)
    assert receipt.full_model_source_sha256 == receipt.full_model_restored_sha256
    assert receipt.full_model_applied_sha256 != receipt.full_model_source_sha256


def test_cuda_adapter_restores_full_model_after_probe_exception() -> None:
    surface = _surface()
    frozen = cast(torch.nn.Parameter, getattr(surface.model, "frozen"))
    frozen_buffer = cast(torch.Tensor, getattr(surface.model, "frozen_buffer"))
    frozen_source = frozen.detach().clone()
    buffer_source = frozen_buffer.detach().clone()

    def failing_probe() -> dict[str, float]:
        frozen.add_(1.0)
        frozen_buffer.add_(1.0)
        surface.model.train()
        raise RuntimeError("injected probe failure")

    invalid = CudaProposalInput(
        model=surface.model,
        named_trainable_parameters=surface.named_trainable_parameters,
        optimizer=surface.optimizer,
        transaction=surface.transaction,
        update_counter=surface.update_counter,
        objective=surface.objective,
        witness_bank=surface.witness_bank,
        proposal_binding=surface.proposal_binding,
        realized_margin_probe=failing_probe,
    )
    with pytest.raises(CudaAdapterError, match="injected probe failure"):
        CudaHFVerticalAdapter(invalid).apply_and_rollback()
    assert torch.equal(frozen.detach(), frozen_source)
    assert torch.equal(frozen_buffer.detach(), buffer_source)
    assert surface.update_counter.value == 0


def test_cuda_adapter_fails_typed_on_parameter_storage_dtype_drift() -> None:
    surface = _surface()

    def dtype_drifting_probe() -> dict[str, float]:
        parameter = surface.named_trainable_parameters[0][1]
        parameter.data = parameter.data.to(torch.float64)
        return {"u_intersect_s_1.0|1584|g-1": 0.0}

    invalid = CudaProposalInput(
        model=surface.model,
        named_trainable_parameters=surface.named_trainable_parameters,
        optimizer=surface.optimizer,
        transaction=surface.transaction,
        update_counter=surface.update_counter,
        objective=surface.objective,
        witness_bank=surface.witness_bank,
        proposal_binding=surface.proposal_binding,
        realized_margin_probe=dtype_drifting_probe,
    )
    with pytest.raises(CudaAdapterRollbackError) as caught:
        CudaHFVerticalAdapter(invalid).apply_and_rollback()
    assert caught.value.rollback_receipt.phase == "full_model_restore"
    assert surface.transaction._active_transaction_id is None
    assert surface.named_trainable_parameters[0][1].dtype == torch.float32
    assert surface.update_counter.value == 0


def test_cuda_adapter_rejects_full_model_parameter_registry_drift() -> None:
    surface = _surface()

    def registry_drifting_probe() -> dict[str, float]:
        surface.model.register_parameter(
            "late", torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        )
        return {"u_intersect_s_1.0|1584|g-1": 0.0}

    invalid = CudaProposalInput(
        model=surface.model,
        named_trainable_parameters=surface.named_trainable_parameters,
        optimizer=surface.optimizer,
        transaction=surface.transaction,
        update_counter=surface.update_counter,
        objective=surface.objective,
        witness_bank=surface.witness_bank,
        proposal_binding=surface.proposal_binding,
        realized_margin_probe=registry_drifting_probe,
    )
    with pytest.raises(CudaAdapterError, match="parameter registry drifted"):
        CudaHFVerticalAdapter(invalid).apply_and_rollback()
    assert surface.update_counter.value == 0


def test_cuda_adapter_emits_terminal_receipt_without_retry_when_reject_fails() -> None:
    surface = _surface()

    class RejectFails(TrainingStateTransaction):
        reject_calls = 0

        def reject(self, snapshot: TrainingStateSnapshot) -> TransactionReceipt:
            del snapshot
            self.reject_calls += 1
            raise RuntimeError("injected reject failure")

    failing_transaction = RejectFails(
        surface.named_trainable_parameters,
        optimizer=surface.optimizer,
        scheduler=None,
        update_counter=surface.update_counter,
        capture_cuda=True,
    )
    invalid = CudaProposalInput(
        model=surface.model,
        named_trainable_parameters=surface.named_trainable_parameters,
        optimizer=surface.optimizer,
        transaction=failing_transaction,
        update_counter=surface.update_counter,
        objective=surface.objective,
        witness_bank=surface.witness_bank,
        proposal_binding=surface.proposal_binding,
        realized_margin_probe=surface.realized_margin_probe,
    )
    with pytest.raises(CudaAdapterRollbackError) as caught:
        CudaHFVerticalAdapter(invalid).apply_and_rollback()
    assert failing_transaction.reject_calls == 1
    assert caught.value.rollback_receipt.phase == "transaction_reject"
    assert caught.value.rollback_receipt.source_state_digest


def test_cuda_adapter_binds_admitted_task2_groups_and_task3_ledgers() -> None:
    surface, _fixture_value = _task2_surface(module_name="_vertical_fixture_module")
    receipt = CudaHFVerticalAdapter(surface).apply_and_rollback()

    assert receipt.status == "applied_and_rolled_back"
    assert receipt.source_parameter_sha256 == receipt.restored_parameter_sha256


def test_cuda_adapter_rejects_detached_compiler_logits() -> None:
    surface, fixture = _task2_surface(module_name="_vertical_detached_fixture")
    raw_logits = fixture.compact_logits._raw_logits
    original = dict(raw_logits)
    try:
        for site_id, value in original.items():
            raw_logits[site_id] = value.detach().clone()
        with pytest.raises(CudaAdapterError, match="live graph"):
            CudaHFVerticalAdapter(surface)
    finally:
        raw_logits.clear()
        raw_logits.update(original)


def test_cuda_adapter_rejects_task2_dtype_identity_drift() -> None:
    surface, _fixture_value = _task2_surface(module_name="_vertical_dtype_drift")
    parameter = surface.named_trainable_parameters[0][1]
    parameter.data = parameter.data.to(torch.float32)
    with pytest.raises(CudaAdapterError, match="parameter state identity"):
        CudaHFVerticalAdapter(surface)


def test_cuda_adapter_rejects_task2_attention_backend_drift() -> None:
    surface, _fixture_value = _task2_surface(module_name="_vertical_backend_drift")
    config = getattr(surface.model, "config")
    setattr(config, "_attn_implementation", "sdpa")
    with pytest.raises(CudaAdapterError, match="attention backend"):
        CudaHFVerticalAdapter(surface)


def test_cuda_adapter_rejects_task2_cache_configuration_drift() -> None:
    surface, _fixture_value = _task2_surface(module_name="_vertical_cache_drift")
    config = getattr(surface.model, "config")
    setattr(config, "use_cache", True)
    with pytest.raises(CudaAdapterError, match="cache configuration"):
        CudaHFVerticalAdapter(surface)


def test_cuda_adapter_restores_task2_config_after_probe_mutation() -> None:
    surface, _fixture_value = _task2_surface(module_name="_vertical_config_probe")
    config = getattr(surface.model, "config")

    def config_drifting_probe() -> dict[str, float]:
        setattr(config, "_attn_implementation", "sdpa")
        setattr(config, "use_cache", True)
        return {witness.canonical_key: 0.0 for witness in surface.witness_bank.constraints}

    mutated = replace(surface, realized_margin_probe=config_drifting_probe)
    receipt = CudaHFVerticalAdapter(mutated).apply_and_rollback()

    assert receipt.status == "applied_and_rolled_back"
    assert getattr(config, "_attn_implementation") == "flash_attention_2"
    assert getattr(config, "use_cache") is False


def test_cuda_adapter_rejects_and_restores_task2_replay_mutation() -> None:
    surface, _fixture_value = _task2_surface(module_name="_vertical_replay_probe")
    key, tensor = next(iter(surface.replay_logprob_tensors.items()))
    source = tensor.detach().clone()

    def replay_drifting_probe() -> dict[str, float]:
        with torch.no_grad():
            tensor.add_(1.0)
        return {witness.canonical_key: 0.0 for witness in surface.witness_bank.constraints}

    mutated = replace(surface, realized_margin_probe=replay_drifting_probe)
    with pytest.raises(CudaAdapterError, match="replay evidence"):
        CudaHFVerticalAdapter(mutated).apply_and_rollback()

    assert key in surface.replay_logprob_tensors
    assert torch.equal(surface.replay_logprob_tensors[key], source)
    assert surface.transaction._active_transaction_id is None


def test_cuda_adapter_rejects_stale_compact_logits_without_compiler_sites() -> None:
    surface, fixture = _task2_surface(module_name="_vertical_stale_compact")
    source = fixture.compiler_ledger
    absent = compiler_owner._admit_compiler_ledger_for_test(
        compiler_owner.CompilerLedger(
            source_sha256=source.source_sha256,
            manifest_sha256=source.manifest_sha256,
            acquisition_sha256=source.acquisition_sha256,
            trajectory_credit_sha256=source.trajectory_credit_sha256,
            repetition_penalty=source.repetition_penalty,
            logical_image_count=1,
            frozen_alias_count=source.frozen_alias_count,
            alias_bank_sha256=source.alias_bank_sha256,
            source_panel_sha256=source.source_panel_sha256,
            images=(
                compiler_owner.CompilerImageLedger(
                    1584, "f" * 64, None, "no_premature_source_boundary"
                ),
            ),
        )
    )
    stale = replace(surface, compiler_ledger=absent)
    with pytest.raises(CudaAdapterError, match="stale"):
        CudaHFVerticalAdapter(stale)
