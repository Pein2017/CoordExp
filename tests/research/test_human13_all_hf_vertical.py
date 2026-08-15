from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

import scripts.research.human13_greedy_compiler as compiler
import scripts.research.human13_all_hf_vertical as vertical
import scripts.research.human13_trajectory_credit as credit
from scripts.research.human13_adamw_proposal_preservation import (
    FROZEN_BETAS,
    FROZEN_EPSILON,
    FROZEN_WEIGHT_DECAY,
    SOURCE_MEMBERSHIPS,
    TRUSTED_OWNER_CLASS,
    WEAKEST_MARGIN_SELECTION,
    FrozenWitnessBank,
    OwnerWitness,
    ParameterLayout,
    WitnessBinding,
    jacobian_sha256,
    parameter_state_sha256,
)
from scripts.research.human13_all_hf_vertical import (
    AllHFVerticalError,
    AllHFVerticalServices,
    PrivateProposalReceipt,
    RollbackReceipt,
)
from scripts.research.human13_greedy_compiler import (
    AliasChild,
    CompilerImageLedger,
    CompilerLedger,
    CompilerSite,
)
from scripts.research.human13_hf_shared_surface import (
    HFActiveBatchStep,
    HFReplayCausalGather,
    HFSharedSurfaceIdentity,
    SampledHFGroup,
    SampledHFRequest,
    SampledHFToken,
    admit_gradient_replay,
    admit_sampled_group,
    plan_image1584_k16,
)
from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)
from scripts.research.human13_trajectory_credit import (
    ImageCreditLedger,
    RowCredit,
    TokenCredit,
    TrajectoryCreditLedger,
    TrajectoryLedger,
)


_UNIT_ID = "2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical"
_SOURCE = "1" * 64
_MANIFEST = "2" * 64
_ACQUISITION = "3" * 64


class _TinySurface(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25, 0.75], dtype=torch.float64))
        self.frozen_base = torch.nn.Parameter(
            torch.tensor([2.0], dtype=torch.float64), requires_grad=False
        )
        self.eval()


class _FakeScheduler:
    def __init__(self) -> None:
        self.value = 0

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, value: dict[str, int]) -> None:
        self.value = value["value"]


@dataclass
class _FakeRuntime:
    optimizer_step_count: int = 0
    scheduler_step_count: int = 0


@dataclass
class _Fixture:
    model: _TinySurface
    named: tuple[tuple[str, torch.nn.Parameter], ...]
    optimizer: torch.optim.AdamW
    counter: UpdateCounter
    transaction: TrainingStateTransaction
    sampled_groups: tuple[SampledHFGroup, ...]
    replay_groups: tuple[Any, ...]
    replay_tensors: dict[str, torch.Tensor]
    trajectory_ledger: TrajectoryCreditLedger
    compiler_ledger: CompilerLedger
    compact_logits: Any
    witness_bank: FrozenWitnessBank


def _surface_evidence(
    model: _TinySurface,
    named: tuple[tuple[str, torch.nn.Parameter], ...],
) -> tuple[
    tuple[SampledHFGroup, ...],
    tuple[Any, ...],
    dict[str, torch.Tensor],
]:
    plan = plan_image1584_k16()
    layout = ParameterLayout.from_named_parameters(named)
    identity = HFSharedSurfaceIdentity(
        checkpoint_payload_sha256=_SOURCE,
        model_object_id=id(model),
        parameter_state_sha256=parameter_state_sha256(named, layout),
        adapter_sha256="4" * 64,
        embedding_delta_sha256="5" * 64,
        dtype="bfloat16",
        attention_backend="flash_attention_2",
        model_mode="eval",
        tokenizer_sha256="6" * 64,
        prompt_sha256="7" * 64,
        image_sha256="8" * 64,
        use_cache=False,
    )
    sampled_groups: list[SampledHFGroup] = []
    replay_groups: list[Any] = []
    replay_tensors: dict[str, torch.Tensor] = {}
    for group_index, seeds in enumerate(plan.seed_groups):
        live = model.weight[0].expand(4) + torch.tensor(
            [0.001 * (seed - 35000) for seed in seeds], dtype=torch.float64
        )
        requests: list[SampledHFRequest] = []
        replayed_tokens: list[SampledHFToken] = []
        gathers: list[HFReplayCausalGather] = []
        for row, seed in enumerate(seeds):
            request_id = f"image-1584:seed-{seed}"
            value = float(live[row].detach().item())
            token = SampledHFToken(
                request_id=request_id,
                token_index=0,
                history_sha256=identity.prompt_sha256,
                chosen_token_id=100 + row,
                raw_chosen_logit=value,
                processed_logp=value,
                causal_logit_index=0,
            )
            requests.append(
                SampledHFRequest(
                    request_id=request_id,
                    image_id=1584,
                    seed=seed,
                    prompt_history_sha256=identity.prompt_sha256,
                    tokens=(token,),
                    processor_order=("repetition_penalty", "temperature", "top_p"),
                    stop_reason="im_end",
                    use_cache=False,
                )
            )
            replayed_tokens.append(token)
            gathers.append(
                HFReplayCausalGather(
                    request_id=request_id,
                    token_index=0,
                    history_sha256=identity.prompt_sha256,
                    chosen_token_id=100 + row,
                    causal_logit_index=0,
                )
            )
        sampled = admit_sampled_group(
            plan=plan,
            group_index=group_index,
            expected_identity=identity,
            identity=identity,
            policy=plan.policy,
            requests=tuple(requests),
            active_batch_steps=(
                HFActiveBatchStep(
                    token_index=0,
                    active_request_ids=tuple(request.request_id for request in requests),
                    active_history_sha256s=tuple(
                        identity.prompt_sha256 for _ in requests
                    ),
                    batch_shape=(4, 1),
                    rng_before_sha256=f"{group_index + 9:064x}",
                    rng_after_sha256=f"{group_index + 13:064x}",
                ),
            ),
        )
        replay = admit_gradient_replay(
            sampled_group=sampled,
            replay_identity=identity,
            replayed_tokens=tuple(replayed_tokens),
            replay_processor_order=("repetition_penalty", "temperature", "top_p"),
            causal_gathers=tuple(gathers),
        )
        sampled_groups.append(sampled)
        replay_groups.append(replay)
        replay_tensors[replay.content_sha256] = live
    return tuple(sampled_groups), tuple(replay_groups), replay_tensors


def _trajectory_ledger(
    sampled_groups: tuple[SampledHFGroup, ...],
    *,
    source_sha256: str = _SOURCE,
) -> TrajectoryCreditLedger:
    requests = tuple(request for group in sampled_groups for request in group.requests)
    returns = tuple(-1.0 if index < 8 else 0.0 for index in range(16))
    trajectories: list[TrajectoryLedger] = []
    total = sum(returns)
    for index, (request, return_to_go) in enumerate(zip(requests, returns, strict=True)):
        baseline = (total - return_to_go) / 15
        raw_advantage = return_to_go - baseline
        advantage = min(raw_advantage, 0.0)
        token = TokenCredit(
            request_id=request.request_id,
            token_index=0,
            row_position=0,
            outcome="natural_stop",
            advantage=advantage,
            scored=True,
        )
        row = RowCredit(
            request_id=request.request_id,
            generated_order=0,
            position_index=0,
            outcome="natural_stop",
            matched_owner_id=None,
            owner_stratum=None,
            token_indices=(0,),
            immediate_credit=return_to_go,
            return_to_go=return_to_go,
            unclamped_advantage=raw_advantage,
            advantage=advantage,
            scored=True,
            tokens=(token,),
        )
        trajectories.append(
            TrajectoryLedger(
                request_id=request.request_id,
                acquisition_trajectory_sha256=f"{index + 1:064x}",
                policy_contract_sha256=f"{index + 101:064x}",
                token_count=1,
                terminal_kind="natural_stop",
                rows=(row,),
            )
        )
    image = ImageCreditLedger(
        image_id=1584,
        acquisition_group_sha256="9" * 64,
        trusted_owner_ids=("owner-g",),
        legacy_m_owner_ids=(),
        owner_weight=1.0,
        trajectories=tuple(trajectories),
        position_returns=(returns,),
        plan_sha256="a" * 64,
        native_receipts_sha256="b" * 64,
        parity_receipt_sha256="c" * 64,
        parser_projection_sha256="d" * 64,
    )
    return credit._construct_trajectory_credit_ledger(
        source_sha256=source_sha256,
        manifest_sha256=_MANIFEST,
        acquisition_sha256=_ACQUISITION,
        logical_image_count=1,
        logical_k=16,
        images=(image,),
        training_repetition_penalty=1.0,
        seed_group_id="35001..35016",
        admit_scientific=True,
    )


def _compiler_evidence(
    model: _TinySurface,
    trajectory_ledger: TrajectoryCreditLedger,
) -> tuple[CompilerLedger, Any]:
    alias_bank_sha256 = "e" * 64
    site = CompilerSite(
        site_id="source-compiler:rp1.00:1584",
        packed_segment_id="segment-1584",
        image_id=1584,
        source_decode_sha256="f" * 64,
        alias_bank_sha256=alias_bank_sha256,
        generated_token_index=0,
        local_causal_position=0,
        bad_token_id=2,
        compact_token_ids=(1, 2),
        repeated_token_ids=(),
        prompt_token_count=1,
        prompt_token_sha256="0" * 64,
        source_prefix_token_count=0,
        source_prefix_token_sha256="1" * 64,
        source_history_token_count=1,
        source_history_sha256="2" * 64,
        alias_children=(AliasChild("owner-h", "alias-h", 1),),
        valid_token_ids=(1,),
        valid_token_weights=(1.0,),
    )
    ledger = compiler._admit_compiler_ledger_for_test(
        CompilerLedger(
            source_sha256=trajectory_ledger.source_sha256,
            manifest_sha256=trajectory_ledger.manifest_sha256,
            acquisition_sha256=trajectory_ledger.acquisition_sha256,
            trajectory_credit_sha256=trajectory_ledger.content_sha256,
            repetition_penalty=1.0,
            logical_image_count=1,
            frozen_alias_count=compiler.EXACT_ALIAS_COUNT,
            alias_bank_sha256=alias_bank_sha256,
            source_panel_sha256="3" * 64,
            images=(CompilerImageLedger(1584, site.source_decode_sha256, site, None),),
        )
    )
    raw_logits = torch.stack(
        (model.weight[0] * 0.0, model.weight[0], model.weight[1])
    ).unsqueeze(0)
    row = object.__new__(compiler.PackedLogitRows)
    for field, value in (
        ("site_id", site.site_id),
        ("segment_id", site.packed_segment_id),
        ("pack_index", 0),
        ("packed_causal_position", 0),
        ("mapping_sha256", "4" * 64),
        ("compiler_ledger_sha256", ledger.content_sha256),
        ("compiler_admission_sha256", ledger.admission_sha256),
        ("admission_sha256", ""),
        ("raw_logits", raw_logits),
        ("_factory_marker", compiler._PACKED_ROW_MARKER),
    ):
        object.__setattr__(row, field, value)
    row_admission = compiler._sha256(compiler._packed_row_admission_preimage(row))
    object.__setattr__(row, "admission_sha256", row_admission)
    compiler._register_admission(compiler._PACKED_ROW_ADMISSIONS, row, row_admission)
    return ledger, compiler.admit_compiler_compact_logits((row,), ledger)


def _fixture() -> _Fixture:
    torch.manual_seed(123)
    model = _TinySurface()
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=3e-6,
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
        capture_cuda=False,
    )
    sampled, replayed, replay_tensors = _surface_evidence(model, named)
    trajectory_ledger = _trajectory_ledger(sampled)
    compiler_ledger, compact_logits = _compiler_evidence(model, trajectory_ledger)
    layout = ParameterLayout.from_named_parameters(named)
    jacobian = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    witness = OwnerWitness(
        image_id="1584",
        owner_id="owner-g",
        source_membership=SOURCE_MEMBERSHIPS[0],
        owner_class=TRUSTED_OWNER_CLASS,
        token_id=1,
        margin_selection=WEAKEST_MARGIN_SELECTION,
        margin_value=0.5,
        detached=True,
        jacobian_sha256=jacobian_sha256(jacobian),
    )
    witness_bank = FrozenWitnessBank.from_witnesses(
        (witness,),
        jacobians={witness.canonical_key: jacobian},
        binding=WitnessBinding(
            unit_id=_UNIT_ID,
            source_checkpoint_sha256=_SOURCE,
            manifest_sha256=_MANIFEST,
            frozen_before_acquisition=True,
        ),
        layout=layout,
    )
    return _Fixture(
        model=model,
        named=named,
        optimizer=optimizer,
        counter=counter,
        transaction=transaction,
        sampled_groups=sampled,
        replay_groups=replayed,
        replay_tensors=replay_tensors,
        trajectory_ledger=trajectory_ledger,
        compiler_ledger=compiler_ledger,
        compact_logits=compact_logits,
        witness_bank=witness_bank,
    )


def _prepare(fixture: _Fixture, **overrides: object):
    arguments: dict[str, object] = {
        "model": fixture.model,
        "named_trainable_parameters": fixture.named,
        "optimizer": fixture.optimizer,
        "transaction": fixture.transaction,
        "update_counter": fixture.counter,
        "sampled_groups": fixture.sampled_groups,
        "replay_groups": fixture.replay_groups,
        "replay_logprob_tensors": fixture.replay_tensors,
        "trajectory_ledger": fixture.trajectory_ledger,
        "compiler_ledger": fixture.compiler_ledger,
        "compiler_compact_logits": fixture.compact_logits,
        "witness_bank": fixture.witness_bank,
        "realized_margin_probe": lambda: {
            witness.canonical_key: witness.margin_value
            for witness in fixture.witness_bank.constraints
        },
    }
    arguments.update(overrides)
    return AllHFVerticalServices().prepare(**arguments)  # type: ignore[arg-type]


def _absent_compiler(trajectory_ledger: TrajectoryCreditLedger) -> CompilerLedger:
    return compiler._admit_compiler_ledger_for_test(
        CompilerLedger(
            source_sha256=trajectory_ledger.source_sha256,
            manifest_sha256=trajectory_ledger.manifest_sha256,
            acquisition_sha256=trajectory_ledger.acquisition_sha256,
            trajectory_credit_sha256=trajectory_ledger.content_sha256,
            repetition_penalty=1.0,
            logical_image_count=1,
            frozen_alias_count=compiler.EXACT_ALIAS_COUNT,
            alias_bank_sha256="e" * 64,
            source_panel_sha256="3" * 64,
            images=(
                CompilerImageLedger(
                    1584,
                    "f" * 64,
                    None,
                    "no_premature_source_boundary",
                ),
            ),
        )
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_complete_vertical_uses_one_global_k16_objective_and_exact_rollback() -> None:
    # Catches dropping a component, per-group normalization, an unprojected apply,
    # an optimizer retry, or releasing a live tensor into a serialized receipt.
    fixture = _fixture()
    before_parameters = tuple(parameter.detach().clone() for _, parameter in fixture.named)
    before_digest = fixture.transaction.state_digest()

    prepared = _prepare(fixture)
    proposal = prepared.backward_and_propose()

    assert proposal.component_names == (
        "trajectory_score_function",
        "greedy_compiler",
        "owner_preservation_projection",
    )
    assert proposal.global_denominator == 16
    assert proposal.compiler_coefficient == 1.0
    assert proposal.compiler_kappa == 1.0
    assert proposal.compiler_margin == 1e-4
    assert proposal.learning_rate == 3e-6
    assert proposal.source_model_mode == proposal.applied_model_mode == "eval"
    assert proposal.backward_count == 1
    assert proposal.proposal_attempt_count == 1
    assert proposal.projected_apply_attempt_count == 1
    assert proposal.update_count_before == 0
    assert proposal.update_count_after == 1
    assert proposal.gradient_all_finite
    assert proposal.unprojected_delta_all_finite
    assert proposal.projected_delta_all_finite
    assert proposal.actual_delta_all_finite
    assert proposal.projected_apply_sha256
    assert proposal.promoted_checkpoint is False
    assert tuple(proposal.request_ids) == tuple(
        request.request_id
        for group in fixture.sampled_groups
        for request in group.requests
    )
    assert not _contains_tensor(proposal.to_dict())
    assert PrivateProposalReceipt.from_dict(proposal.to_dict()).to_dict() == proposal.to_dict()
    assert fixture.counter.value == 1
    assert any(
        not torch.equal(parameter, before)
        for (_, parameter), before in zip(fixture.named, before_parameters, strict=True)
    )

    rollback = prepared.rollback()

    assert isinstance(rollback, RollbackReceipt)
    assert rollback.decision == "rejected_restored"
    assert rollback.before_state_digest == before_digest
    assert rollback.after_state_digest == before_digest
    assert rollback.rollback_count == 1
    assert rollback.source_model_mode == rollback.restored_model_mode == "eval"
    assert rollback.update_count_after == 0
    assert rollback.promoted_checkpoint is False
    assert rollback.cpu_rng_before_sha256 == rollback.cpu_rng_after_sha256
    assert not _contains_tensor(rollback.to_dict())
    assert RollbackReceipt.from_dict(rollback.to_dict()).to_dict() == rollback.to_dict()
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)
    assert all(
        torch.equal(parameter, before)
        for (_, parameter), before in zip(fixture.named, before_parameters, strict=True)
    )
    with pytest.raises(RuntimeError, match="already rolled back"):
        prepared.rollback()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("trajectory_ledger", None, "scientific ledger"),
        ("compiler_ledger", None, "compiler admission"),
        ("compiler_compact_logits", None, "compiler evidence"),
        ("witness_bank", None, "preservation evidence"),
    ),
)
def test_missing_objective_or_preservation_component_fails_before_backward(
    field: str, value: object, message: str
) -> None:
    # Catches CE, missing-compiler, and unprojected fallbacks hidden in one arm.
    fixture = _fixture()
    before = tuple(parameter.detach().clone() for _, parameter in fixture.named)
    with pytest.raises((TypeError, ValueError), match=message):
        _prepare(fixture, **{field: value})
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)
    assert all(
        torch.equal(parameter, saved)
        for (_, parameter), saved in zip(fixture.named, before, strict=True)
    )


def test_absent_compiler_site_is_an_admitted_zero_component_not_a_fallback() -> None:
    # Catches requiring fake logits for a canonically absent Source boundary or
    # dropping the compiler component/gradient path from the complete arm.
    fixture = _fixture()
    absent = _absent_compiler(fixture.trajectory_ledger)
    prepared = _prepare(
        fixture,
        compiler_ledger=absent,
        compiler_compact_logits=None,
    )
    proposal = prepared.backward_and_propose()
    assert proposal.compiler_numerator == 0.0
    assert proposal.component_names[1] == "greedy_compiler"
    assert proposal.global_denominator == 16
    prepared.rollback()


def test_group_boundaries_do_not_change_the_single_global_normalization() -> None:
    # Catches dividing each physical group or compiler component separately.
    fixture = _fixture()
    prepared = _prepare(fixture)
    proposal = prepared.backward_and_propose()
    expected = (
        proposal.trajectory_numerator
        + proposal.compiler_coefficient * proposal.compiler_numerator
    ) / 16
    assert proposal.total_loss == pytest.approx(expected, abs=1e-15, rel=0)
    assert proposal.global_denominator == (
        fixture.trajectory_ledger.logical_image_count
        * fixture.trajectory_ledger.logical_k
    )
    prepared.rollback()


def test_duplicate_group_or_replay_tensor_mismatch_fails_at_prepare() -> None:
    # Catches duplicated K evidence and a detached tensor substituted under a
    # valid replay receipt hash.
    fixture = _fixture()
    duplicate_sampled = (
        fixture.sampled_groups[0],
        fixture.sampled_groups[0],
        *fixture.sampled_groups[2:],
    )
    with pytest.raises(ValueError, match="group lineage"):
        _prepare(fixture, sampled_groups=duplicate_sampled)

    changed = dict(fixture.replay_tensors)
    first_hash = fixture.replay_groups[0].content_sha256
    changed[first_hash] = changed[first_hash] + 0.5
    with pytest.raises(ValueError, match="values differ"):
        _prepare(fixture, replay_logprob_tensors=changed)


def test_mutated_trajectory_compiler_and_compact_evidence_fail_closed() -> None:
    # Catches seal bypass, alias/site drift, or graph mutation after preparation.
    fixture = _fixture()
    prepared = _prepare(fixture)
    object.__setattr__(fixture.trajectory_ledger, "logical_k", 15)
    with pytest.raises(AllHFVerticalError, match="ledger"):
        prepared.backward_and_propose()
    assert fixture.counter.value == 0

    fixture = _fixture()
    prepared = _prepare(fixture)
    object.__setattr__(fixture.compiler_ledger, "alias_bank_sha256", "0" * 64)
    with pytest.raises(AllHFVerticalError, match="compiler admission"):
        prepared.backward_and_propose()
    assert fixture.counter.value == 0

    fixture = _fixture()
    prepared = _prepare(fixture)
    raw = next(iter(fixture.compact_logits._raw_logits.values()))
    with torch.no_grad():
        raw.add_(1.0)
    with pytest.raises(AllHFVerticalError, match="compact-logit"):
        prepared.backward_and_propose()
    assert fixture.counter.value == 0


def test_model_parameter_optimizer_and_source_state_substitution_fail_closed() -> None:
    # Catches equal-looking foreign state being used under an admitted identity.
    fixture = _fixture()
    with pytest.raises(ValueError, match="model object|model trainable"):
        _prepare(fixture, model=_TinySurface())

    fixture = _fixture()
    foreign = _TinySurface()
    with pytest.raises(ValueError, match="model trainable"):
        _prepare(
            fixture,
            named_trainable_parameters=tuple(
                (name, parameter)
                for name, parameter in foreign.named_parameters()
                if parameter.requires_grad
            ),
        )

    fixture = _fixture()
    fixture.optimizer.param_groups[0]["lr"] = 1e-6
    with pytest.raises(ValueError, match="fixed fresh AdamW"):
        _prepare(fixture)

    fixture = _fixture()
    source = tuple(parameter.detach().clone() for _, parameter in fixture.named)
    source_digest = fixture.transaction.state_digest()
    prepared = _prepare(fixture)
    with torch.no_grad():
        fixture.model.weight.add_(1.0)
    with pytest.raises(AllHFVerticalError, match="Source state drifted") as error:
        prepared.backward_and_propose()
    assert error.value.rollback_receipt is not None
    assert fixture.transaction.state_digest() == source_digest
    assert all(
        torch.equal(parameter, saved)
        for (_, parameter), saved in zip(fixture.named, source, strict=True)
    )
    assert fixture.counter.value == 0


@pytest.mark.parametrize("drift", ("train_mode", "registry_swap"))
def test_post_prepare_model_mode_or_parameter_registry_drift_fails_closed(
    drift: str,
) -> None:
    # Equal-valued model-surface substitution must be detected from the live
    # module registry before backward, not only from the originally passed tuple.
    fixture = _fixture()
    prepared = _prepare(fixture)
    if drift == "train_mode":
        fixture.model.train()
    else:
        fixture.model.weight = torch.nn.Parameter(
            fixture.model.weight.detach().clone(), requires_grad=True
        )

    with pytest.raises(AllHFVerticalError, match="eval-mode|trainable surface"):
        prepared.backward_and_propose()

    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)


def test_post_prepare_frozen_parameter_drift_rejects_and_restores_full_source() -> None:
    # The HF shared surface includes registered frozen base parameters even
    # though AdamW remains bound only to the trainable proposal surface.
    fixture = _fixture()
    source_frozen = fixture.model.frozen_base.detach().clone()
    source_transaction_digest = fixture.transaction.state_digest()
    prepared = _prepare(fixture)
    with torch.no_grad():
        fixture.model.frozen_base.add_(1.0)

    with pytest.raises(AllHFVerticalError, match="full model Source") as error:
        prepared.backward_and_propose()

    rollback = error.value.rollback_receipt
    assert rollback is not None
    assert rollback.full_model_source_sha256 == rollback.full_model_restored_sha256
    assert torch.equal(fixture.model.frozen_base, source_frozen)
    assert fixture.transaction.state_digest() == source_transaction_digest
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)


def test_equal_valued_foreign_replay_graph_is_rejected_before_proposal() -> None:
    # Detached bytes alone cannot prove that the trajectory graph belongs to
    # the admitted model surface.
    fixture = _fixture()
    foreign = torch.nn.Parameter(fixture.model.weight.detach().clone())
    foreign_replay = {
        receipt_sha256: foreign[0].expand_as(value)
        + (value.detach() - fixture.model.weight[0].detach())
        for receipt_sha256, value in fixture.replay_tensors.items()
    }
    assert all(
        torch.equal(foreign_replay[key].detach(), value.detach())
        for key, value in fixture.replay_tensors.items()
    )

    with pytest.raises(ValueError, match="trajectory graph owner"):
        _prepare(fixture, replay_logprob_tensors=foreign_replay)

    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}


def test_model_surface_is_revalidated_after_capture_and_before_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    before_digest = fixture.transaction.state_digest()
    original_project = vertical.project_adamw_proposal

    def drift_during_projection(*args: object, **kwargs: object):
        projection = original_project(*args, **kwargs)  # type: ignore[arg-type]
        fixture.model.train()
        return projection

    monkeypatch.setattr(vertical, "project_adamw_proposal", drift_during_projection)
    prepared = _prepare(fixture)

    with pytest.raises(AllHFVerticalError, match="eval-mode") as error:
        prepared.backward_and_propose()

    assert error.value.rollback_receipt is not None
    assert fixture.transaction.state_digest() == before_digest
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)


def test_equal_valued_foreign_compiler_graph_is_rejected_before_proposal() -> None:
    # A valid compact-logit admission with identical values still belongs to a
    # foreign autograd graph when its leaves are another model's parameters.
    fixture = _fixture()
    foreign_model = _TinySurface()
    foreign_ledger, foreign_compact = _compiler_evidence(
        foreign_model, fixture.trajectory_ledger
    )
    original_logits = tuple(fixture.compact_logits._raw_logits.values())
    foreign_logits = tuple(foreign_compact._raw_logits.values())
    assert all(
        torch.equal(foreign.detach(), original.detach())
        for foreign, original in zip(foreign_logits, original_logits, strict=True)
    )

    with pytest.raises(ValueError, match="compiler graph owner"):
        _prepare(
            fixture,
            compiler_ledger=foreign_ledger,
            compiler_compact_logits=foreign_compact,
        )

    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}


def test_foreign_admitted_source_is_not_joined_to_hf_surface_identity() -> None:
    # Matching trajectory/compiler K16 shape and mutual lineage are insufficient:
    # the scientific source must be the exact HF identity checkpoint payload.
    fixture = _fixture()
    foreign_trajectory = _trajectory_ledger(
        fixture.sampled_groups, source_sha256="9" * 64
    )
    foreign_compiler, foreign_compact = _compiler_evidence(
        fixture.model, foreign_trajectory
    )

    with pytest.raises(ValueError, match="HF shared-surface Source"):
        _prepare(
            fixture,
            trajectory_ledger=foreign_trajectory,
            compiler_ledger=foreign_compiler,
            compiler_compact_logits=foreign_compact,
        )

    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}


def test_post_backward_exception_restores_parameter_optimizer_counter_and_rng() -> None:
    # Catches an apply/probe exception escaping without the outer exact rollback.
    fixture = _fixture()
    before_parameters = tuple(parameter.detach().clone() for _, parameter in fixture.named)
    before_digest = fixture.transaction.state_digest()
    before_rng = torch.get_rng_state().clone()
    before_versions = tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )

    def fail_probe() -> dict[str, float]:
        torch.manual_seed(999)
        raise RuntimeError("injected realized-margin failure")

    prepared = _prepare(fixture, realized_margin_probe=fail_probe)
    with pytest.raises(AllHFVerticalError, match="injected realized-margin") as error:
        prepared.backward_and_propose()

    rollback = error.value.rollback_receipt
    assert rollback is not None
    assert rollback.before_state_digest == rollback.after_state_digest == before_digest
    assert rollback.full_model_source_versions == before_versions
    assert rollback.full_model_restored_versions == before_versions
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(parameter.grad is None for _, parameter in fixture.named)
    assert all(
        torch.equal(parameter, saved)
        for (_, parameter), saved in zip(
            fixture.named, before_parameters, strict=True
        )
    )
    assert tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    ) == before_versions
    with pytest.raises(RuntimeError, match="already rolled back"):
        prepared.rollback()


@pytest.mark.parametrize("mutation", ("train_mode", "frozen", "trainable"))
def test_realized_margin_probe_cannot_mutate_the_receipted_model_surface(
    mutation: str,
) -> None:
    # The probe is external code executed after the preservation owner measures
    # the projected delta. Its return cannot authorize unreceipted model drift.
    fixture = _fixture()
    source_values = tuple(
        parameter.detach().clone() for _, parameter in fixture.model.named_parameters()
    )
    source_digest = fixture.transaction.state_digest()

    def mutating_probe() -> dict[str, float]:
        if mutation == "train_mode":
            fixture.model.train()
        elif mutation == "frozen":
            with torch.no_grad():
                fixture.model.frozen_base.add_(1.0)
        else:
            with torch.no_grad():
                fixture.model.weight.add_(1.0)
        return {
            witness.canonical_key: witness.margin_value
            for witness in fixture.witness_bank.constraints
        }

    prepared = _prepare(fixture, realized_margin_probe=mutating_probe)
    with pytest.raises(AllHFVerticalError, match="post-probe model surface") as error:
        prepared.backward_and_propose()

    rollback = error.value.rollback_receipt
    assert rollback is not None
    assert rollback.source_model_mode == rollback.restored_model_mode == "eval"
    assert rollback.full_model_source_sha256 == rollback.full_model_restored_sha256
    assert prepared._proposal_receipt is None
    assert fixture.model.training is False
    assert fixture.transaction.state_digest() == source_digest
    assert all(
        torch.equal(parameter, saved)
        for (_, parameter), saved in zip(
            fixture.model.named_parameters(), source_values, strict=True
        )
    )
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)


def test_probe_optimizer_substitution_is_restored_and_terminally_rejected() -> None:
    fixture = _fixture()
    source_optimizer_parameters = tuple(
        parameter
        for group in fixture.optimizer.param_groups
        for parameter in group["params"]
    )

    def substituting_probe() -> dict[str, float]:
        fixture.optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(
            fixture.model.weight.detach().clone()
        )
        return {
            witness.canonical_key: witness.margin_value
            for witness in fixture.witness_bank.constraints
        }

    prepared = _prepare(fixture, realized_margin_probe=substituting_probe)
    with pytest.raises(AllHFVerticalError, match="post-probe optimizer") as error:
        prepared.backward_and_propose()

    assert error.value.rollback_receipt is not None
    assert prepared._proposal_receipt is None
    assert prepared._state == "rolled_back"
    assert tuple(
        parameter
        for group in fixture.optimizer.param_groups
        for parameter in group["params"]
    ) == source_optimizer_parameters
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)
    with pytest.raises(RuntimeError, match="already rolled back"):
        prepared.rollback()


@pytest.mark.parametrize("drift", ("optimizer_ref", "optimizer_state", "counter"))
def test_pre_transaction_ownership_drift_restores_prepare_time_source(
    drift: str,
) -> None:
    fixture = _fixture()
    source_digest = fixture.transaction.state_digest()
    source_versions = tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )
    source_optimizer_parameters = tuple(
        parameter
        for group in fixture.optimizer.param_groups
        for parameter in group["params"]
    )
    prepared = _prepare(fixture)
    if drift == "optimizer_ref":
        fixture.optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(
            fixture.model.weight.detach().clone()
        )
    elif drift == "optimizer_state":
        fixture.optimizer.state[fixture.model.weight] = {
            "step": torch.tensor(1.0, dtype=torch.float64)
        }
    else:
        fixture.counter.value = 7

    with pytest.raises(AllHFVerticalError) as error:
        prepared.backward_and_propose()

    rollback = error.value.rollback_receipt
    assert rollback is not None
    assert prepared._proposal_receipt is None
    assert prepared._state == "rolled_back"
    assert rollback.full_model_source_versions == source_versions
    assert rollback.full_model_restored_versions == source_versions
    assert tuple(
        parameter
        for group in fixture.optimizer.param_groups
        for parameter in group["params"]
    ) == source_optimizer_parameters
    assert fixture.transaction.state_digest() == source_digest
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert fixture.model.training is False
    assert tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    ) == source_versions
    assert all(parameter.grad is None for _, parameter in fixture.named)
    with pytest.raises(RuntimeError, match="already rolled back"):
        prepared.rollback()


@pytest.mark.parametrize("drift", ("registry_swap", "requires_grad"))
def test_model_registry_or_trainability_drift_restores_exact_source_objects(
    drift: str,
) -> None:
    fixture = _fixture()
    source_weight = fixture.model.weight
    source_versions = tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )
    source_digest = fixture.transaction.state_digest()
    prepared = _prepare(fixture)
    if drift == "registry_swap":
        fixture.model.weight = torch.nn.Parameter(source_weight.detach().clone())
    else:
        fixture.model.weight.requires_grad_(False)

    with pytest.raises(AllHFVerticalError) as error:
        prepared.backward_and_propose()

    rollback = error.value.rollback_receipt
    assert rollback is not None
    assert prepared._state == "rolled_back"
    assert fixture.model.weight is source_weight
    assert fixture.model.weight.requires_grad is True
    assert fixture.optimizer.param_groups[0]["params"][0] is source_weight
    assert fixture.transaction.state_digest() == source_digest
    assert tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    ) == source_versions
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)
    with pytest.raises(RuntimeError, match="already rolled back"):
        prepared.rollback()


@pytest.mark.parametrize("drift", ("runtime", "scheduler", "rng"))
def test_pre_begin_transaction_owner_or_rng_drift_uses_prepare_source_recovery(
    drift: str,
) -> None:
    fixture = _fixture()
    source_rng = torch.get_rng_state().clone()
    source_digest = fixture.transaction.state_digest()
    prepared = _prepare(fixture)
    if drift == "runtime":
        fixture.transaction._runtime = _FakeRuntime()
    elif drift == "scheduler":
        fixture.transaction._scheduler = _FakeScheduler()
    else:
        torch.manual_seed(999)

    with pytest.raises(AllHFVerticalError) as error:
        prepared.backward_and_propose()

    assert error.value.rollback_receipt is not None
    assert prepared._state == "rolled_back"
    assert fixture.transaction._runtime is None
    assert fixture.transaction._scheduler is None
    assert torch.equal(torch.get_rng_state(), source_rng)
    assert fixture.transaction.state_digest() == source_digest
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)
    with pytest.raises(RuntimeError, match="already rolled back"):
        prepared.rollback()


def test_pre_backward_dtype_drift_restores_exact_source_metadata() -> None:
    fixture = _fixture()
    source_digest = fixture.transaction.state_digest()
    source_dtypes = tuple(
        parameter.dtype for _, parameter in fixture.model.named_parameters()
    )
    source_versions = tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )
    prepared = _prepare(fixture)
    fixture.model.float()

    with pytest.raises(AllHFVerticalError) as error:
        prepared.backward_and_propose()

    assert error.value.rollback_receipt is not None
    assert tuple(
        parameter.dtype for _, parameter in fixture.model.named_parameters()
    ) == source_dtypes
    assert tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    ) == source_versions
    assert fixture.transaction.state_digest() == source_digest
    assert prepared._state == "rolled_back"


def test_realized_probe_dtype_drift_restores_exact_source_metadata() -> None:
    fixture = _fixture()
    source_digest = fixture.transaction.state_digest()
    source_dtypes = tuple(
        parameter.dtype for _, parameter in fixture.model.named_parameters()
    )
    source_versions = tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )

    def dtype_probe() -> dict[str, float]:
        fixture.model.float()
        return {
            witness.canonical_key: witness.margin_value
            for witness in fixture.witness_bank.constraints
        }

    prepared = _prepare(fixture, realized_margin_probe=dtype_probe)
    with pytest.raises(AllHFVerticalError) as error:
        prepared.backward_and_propose()

    assert error.value.rollback_receipt is not None
    assert tuple(
        parameter.dtype for _, parameter in fixture.model.named_parameters()
    ) == source_dtypes
    assert tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    ) == source_versions
    assert fixture.transaction.state_digest() == source_digest
    assert prepared._state == "rolled_back"


def test_normal_rollback_restores_exact_source_tensor_versions() -> None:
    fixture = _fixture()
    source_versions = tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )
    prepared = _prepare(fixture)
    proposal = prepared.backward_and_propose()

    assert proposal.full_model_source_versions == source_versions
    assert proposal.full_model_applied_versions == tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )
    rollback = prepared.rollback()

    assert rollback.full_model_source_versions == source_versions
    assert rollback.full_model_restored_versions == source_versions
    assert tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    ) == source_versions


def test_public_rollback_failure_is_typed_terminal_and_source_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    source_digest = fixture.transaction.state_digest()
    source_versions = tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    )
    source_parameters = tuple(
        parameter.detach().clone() for _, parameter in fixture.model.named_parameters()
    )
    prepared = _prepare(fixture)
    prepared.backward_and_propose()

    def reject_failure(_snapshot: object) -> None:
        raise RuntimeError("injected transaction reject failure")

    monkeypatch.setattr(fixture.transaction, "reject", reject_failure)
    with pytest.raises(AllHFVerticalError, match="rollback failed"):
        prepared.rollback()

    assert prepared._state == "rolled_back"
    assert fixture.model.training is False
    assert fixture.transaction.state_digest() == source_digest
    assert tuple(
        parameter._version for _, parameter in fixture.model.named_parameters()
    ) == source_versions
    assert all(
        torch.equal(parameter, source)
        for (_, parameter), source in zip(
            fixture.model.named_parameters(), source_parameters, strict=True
        )
    )
    assert fixture.counter.value == 0
    assert fixture.optimizer.state == {}
    assert all(parameter.grad is None for _, parameter in fixture.named)
    with pytest.raises(RuntimeError, match="already rolled back"):
        prepared.rollback()


def test_proposal_is_one_shot_and_receipt_seals_reject_mutation() -> None:
    # Catches adaptive retry/double apply and canonical rehash of changed evidence.
    fixture = _fixture()
    prepared = _prepare(fixture)
    proposal = prepared.backward_and_propose()
    with pytest.raises(RuntimeError, match="one-shot"):
        prepared.backward_and_propose()
    assert fixture.counter.value == 1

    payload = proposal.to_dict()
    payload["global_denominator"] = 15
    with pytest.raises(ValueError, match="constants|content hash"):
        PrivateProposalReceipt.from_dict(payload)
    object.__setattr__(proposal, "global_denominator", 15)
    with pytest.raises(ValueError, match="mutated"):
        proposal.to_dict()
    prepared.rollback()


def test_vertical_exposes_no_accept_checkpoint_retry_or_fallback_surface() -> None:
    fixture = _fixture()
    prepared = _prepare(fixture)
    for prohibited in (
        "accept",
        "write_checkpoint",
        "retry",
        "learning_rate_ray",
        "cross_entropy_fallback",
        "unprojected_apply",
    ):
        assert not hasattr(prepared, prohibited)
    prepared.backward_and_propose()
    prepared.rollback()
