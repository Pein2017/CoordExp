from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import math

import pytest
import torch

import scripts.research.collect_human13_rp_crossover as task2
import scripts.research.human13_trajectory_credit as credit
from scripts.research.analyze_human13_k_union import _manifest_sha256
from scripts.research.build_human13_k_union_manifest import (
    ArmIdentity,
    GlobalDenominatorIdentity,
    Human13KUnionManifest,
    ImageRecord,
    OwnerRecord,
    default_binding,
)
from scripts.research.human13_k_trajectory_contracts import (
    AcquisitionGroup,
    ArtifactIdentity,
    CompleteTrajectoryEvidence,
    GeneratedTokenEvidence,
    PolicyContract,
)
from scripts.research.human13_trajectory_credit import (
    _AcquisitionGroupCreditInput,
    _MalformedRowSpan,
    _ParsedCreditRow,
    _ParsedTrajectoryCreditInput,
    _TrajectoryCreditAcquisition,
    TrajectoryCreditLedger,
    build_trajectory_credit_ledger as _build_public_trajectory_credit_ledger,
    trajectory_score_function_loss,
    trajectory_score_function_numerator,
)
from scripts.research.human13_rp_policy import validate_acquisition_group_replay


STOP = 99
SOURCE = "a" * 64
TASK2_STOP = 151645
_TOKEN_TEXT = {
    100: "<|object_ref_start|>",
    101: "cat",
    102: "<|object_ref_end|>",
    103: "<|box_start|>",
    104: "<|coord_0|>",
    105: "<|coord_10|>",
    106: "<|box_end|>",
    TASK2_STOP: "<|im_end|>",
}
_ONE_CANONICAL_ROW = (100, 101, 102, 103, 104, 104, 105, 105, 106, TASK2_STOP)


class _FakeTokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        assert not skip_special_tokens
        return "".join(_TOKEN_TEXT[token_id] for token_id in token_ids)


def _task2_request_receipt(
    request: task2.AcquisitionRequest,
) -> task2.NativeRequestReceipt:
    return task2.NativeRequestReceipt(
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        sampling_params=task2.expected_native_sampling_evidence(request),
        prompt_token_ids_sha256=task2.token_ids_sha256((1, 2)),
        model_id="test-model",
        model_identity_sha256="c" * 64,
        session_identity_sha256="d" * 64,
    )


def _task2_output_receipt(
    request: task2.AcquisitionRequest,
    *,
    manifest_sha256: str,
    generated_token_ids: tuple[int, ...],
) -> task2.NativeOutputReceipt:
    native = _task2_request_receipt(request)
    return task2.NativeOutputReceipt(
        native_request_receipt_sha256=native.content_sha256,
        request_id=request.request_id,
        seed=request.seed,
        physical_batch_index=request.physical_batch_index,
        request_order_in_batch=request.request_order_in_batch,
        prompt_token_ids=(1, 2),
        source_sha256=SOURCE,
        manifest_sha256=manifest_sha256,
        model_id="test-model",
        tokenizer_id="fake-tokenizer:v1",
        processor_id="test-processor",
        processor_order=("repetition_penalty", "temperature", "log_softmax"),
        sampler_backend_id="vllm:test",
        generated_token_ids=generated_token_ids,
        processed_logprobs=(-0.25,) * len(generated_token_ids),
        terminal_kind="natural_stop",
    )


def _admitted_publication(
    manifest: Human13KUnionManifest,
    *,
    image_id: int = 1584,
    repetition_penalty: float = 1.0,
    seed_group_id: str = "qualification",
    generated_token_ids: tuple[int, ...] = _ONE_CANONICAL_ROW,
) -> task2.AdmittedPublication:
    plan = task2.plan_acquisition_group(
        image_id=image_id,
        repetition_penalty=repetition_penalty,
        seed_group_id=seed_group_id,
    )
    manifest_sha = _manifest_sha256(manifest)

    def execute(
        batch: task2.AcquisitionBatch, params: tuple[object, ...]
    ) -> task2.NativeBatchReceipt:
        del params
        requests = tuple(_task2_request_receipt(request) for request in batch.requests)
        outputs = tuple(
            _task2_output_receipt(
                request,
                manifest_sha256=manifest_sha,
                generated_token_ids=generated_token_ids,
            )
            for request in batch.requests
        )
        return task2.NativeBatchReceipt(requests=requests, outputs=outputs)

    execution = task2.execute_acquisition_group(plan=plan, execute_batch=execute)
    parity = validate_acquisition_group_replay(
        execution.group, execution.group, task2.ReplayTolerance()
    )
    binding = task2._publication_binding(
        execution=execution,
        replayed=execution.group,
        replay_receipt=parity,
    )
    return task2.AdmittedPublication(binding, execution, execution.group, parity)


def _canonical_tokenizer_adapter() -> credit.CanonicalTokenizerDecodeAdapter:
    return credit.CanonicalTokenizerDecodeAdapter(
        tokenizer=_FakeTokenizer(),
        tokenizer_id="fake-tokenizer:v1",
        tokenizer_sha256=default_binding().surface.tokenizer_sha256,
        implementation_id="fake-hf-tokenizer-for-cpu-tests.v1",
    )


def _owner(
    owner_id: str,
    stratum: str,
    bbox: tuple[float, float, float, float],
    category: str = "cat",
) -> OwnerRecord:
    return OwnerRecord(
        owner_id=owner_id,
        category=category,
        bbox=bbox,
        source_object_index=int(owner_id.split("-")[-1]),
        stratum=stratum,  # type: ignore[arg-type]
        source_row_ids=(),
        sampled_row_ids=(),
    )


def _manifest(*owners: OwnerRecord, image_id: int = 7) -> Human13KUnionManifest:
    image = ImageRecord(
        image_id=image_id,
        panel_row_sha256=None,
        image_sha256=None,
        owners=tuple(owners),
        trajectories=(),
        duplicate_events=(),
        selected_rows=(),
        g_owner_ids=tuple(owner.owner_id for owner in owners if owner.stratum == "G"),
        h_owner_ids=tuple(owner.owner_id for owner in owners if owner.stratum == "H"),
        m_owner_ids=tuple(owner.owner_id for owner in owners if owner.stratum == "M"),
        replay_row_ids=(),
        target_row_ids=(),
        candidate_row_ids=(),
    )
    return Human13KUnionManifest(
        schema_version="human13_k_union_manifest.v1",
        binding=default_binding(),
        images=(image,),
        arms=(ArmIdentity("frozen_source", "none", True),),
        denominators=GlobalDenominatorIdentity(1, 0, 0, 0, 0, 0, 0),
        full_panel=False,
    )


def _row(
    order: int,
    bbox: tuple[float, float, float, float] | None,
    *,
    category: str = "cat",
    token_start: int | None = None,
    token_end: int | None = None,
    geometry_valid: bool = True,
) -> _ParsedCreditRow:
    start = order if token_start is None else token_start
    end = start + 1 if token_end is None else token_end
    return _ParsedCreditRow(
        generated_order=order,
        category=category,
        bbox=bbox,
        token_start=start,
        token_end=end,
        geometry_valid=geometry_valid,
    )


def _complete_trajectory(
    manifest_sha256: str,
    request_id: str,
    *,
    token_count: int,
    terminal_kind: str,
) -> CompleteTrajectoryEvidence:
    generated = tuple(range(10, 10 + token_count))
    if terminal_kind == "natural_stop":
        generated = (*generated[:-1], STOP)
    identity = ArtifactIdentity(
        source_sha256=SOURCE,
        manifest_sha256=manifest_sha256,
        request_id=request_id,
        model_id="test-model",
        tokenizer_id="test-tokenizer",
        processor_id="test-processor",
        prompt_token_ids=(1, 2),
        generated_token_ids=generated,
    )
    contract = PolicyContract(
        identity=replace(identity, generated_token_ids=()),
        repetition_penalty=1.0,
        temperature=0.4,
        natural_stop_token_id=STOP,
        max_new_tokens=token_count,
        sampler_backend_id="test-sampler",
        top_p=1.0,
        top_k=None,
        n=1,
        min_new_tokens=0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        ignore_eos=False,
    )
    tokens = tuple(
        GeneratedTokenEvidence(
            identity=identity,
            policy_contract_sha256=contract.content_sha256,
            token_index=index,
            history_token_ids=(*identity.prompt_token_ids, *generated[:index]),
            chosen_token_id=token,
            processed_logprob=-0.25,
        )
        for index, token in enumerate(generated)
    )
    return CompleteTrajectoryEvidence(identity, contract, tokens, terminal_kind)


def _acquisition(
    manifest: Human13KUnionManifest,
    trajectories: tuple[
        tuple[tuple[_ParsedCreditRow, ...], tuple[_MalformedRowSpan, ...], str], ...
    ],
    *,
    logical_k: int | None = None,
) -> _TrajectoryCreditAcquisition:
    manifest_sha = _manifest_sha256(manifest)
    complete = tuple(
        _complete_trajectory(
            manifest_sha,
            request_id,
            token_count=6,
            terminal_kind=terminal_kind,
        )
        for _, _, request_id, terminal_kind in (
            (*rows_and_spans, f"request-{index}", terminal_kind)
            for index, (*rows_and_spans, terminal_kind) in enumerate(trajectories)
        )
    )
    group_identity = replace(complete[0].identity, generated_token_ids=())
    group_contract = replace(complete[0].policy_contract, identity=group_identity)
    group = AcquisitionGroup(
        identity=group_identity,
        policy_contract=group_contract,
        trajectories=complete,
        seed_group_id="test-explicit-small-k",
    )
    parsed = tuple(
        _ParsedTrajectoryCreditInput(
            request_id=trajectory.identity.request_id,
            rows=rows,
            malformed_spans=malformed,
        )
        for trajectory, (rows, malformed, _) in zip(complete, trajectories, strict=True)
    )
    return _TrajectoryCreditAcquisition(
        groups=(
            _AcquisitionGroupCreditInput(manifest.images[0].image_id, group, parsed),
        ),
        logical_k=len(parsed) if logical_k is None else logical_k,
    )


def build_trajectory_credit_ledger(
    manifest: Human13KUnionManifest,
    acquisition: object,
    *,
    tokenizer_adapter: object | None = None,
) -> TrajectoryCreditLedger:
    """Route formula-unit fixtures through the private authored-row seam.

    Public-boundary tests below still call the admitted-publication API.  This
    keeps the original table-driven credit math tests small without making an
    authored semantic envelope an admissible production input.
    """

    if isinstance(acquisition, _TrajectoryCreditAcquisition):
        assert tokenizer_adapter is None
        return credit._build_trajectory_credit_ledger_from_parsed(manifest, acquisition)
    assert tokenizer_adapter is not None
    return _build_public_trajectory_credit_ledger(
        manifest,
        acquisition,  # type: ignore[arg-type]
        tokenizer_adapter=tokenizer_adapter,  # type: ignore[arg-type]
    )


def _outcomes(ledger: TrajectoryCreditLedger, trajectory_index: int = 0) -> list[str]:
    return [row.outcome for row in ledger.images[0].trajectories[trajectory_index].rows]


def test_first_hits_use_fixed_uniform_weights_and_nonlinear_marginal_utility() -> None:
    # Catches K-dependent weights or replacing the exponential utility by linear coverage.
    manifest = _manifest(
        _owner("owner-0", "G", (0, 0, 10, 10)),
        _owner("owner-1", "H", (20, 0, 30, 10)),
    )
    rows = (_row(0, (0, 0, 10, 10)), _row(1, (20, 0, 30, 10)))
    acquisition = _acquisition(
        manifest,
        ((rows, (), "natural_stop"), (rows, (), "natural_stop")),
    )

    ledger = build_trajectory_credit_ledger(manifest, acquisition)
    first, second, stop = ledger.images[0].trajectories[0].rows
    u_half = (math.exp(0.5) - 1.0) / (math.e - 1.0)
    assert ledger.images[0].owner_weight == pytest.approx(0.5)
    assert first.immediate_credit == pytest.approx(u_half)
    assert second.immediate_credit == pytest.approx(1.0 - u_half)
    assert stop.immediate_credit == 0.0
    assert first.matched_owner_id == "owner-0"
    assert second.matched_owner_id == "owner-1"


def test_duplicate_invalid_repeat_and_unmatched_precedence_never_stacks_costs() -> None:
    # Catches category-aware duplicate checks, invalid owner credit, and stacked burdens.
    manifest = _manifest(
        _owner("owner-0", "G", (0, 0, 10, 10)),
        _owner("owner-1", "H", (20, 0, 30, 10)),
    )
    rows = (
        _row(0, (0, 0, 10, 10)),
        _row(1, (0, 0, 10, 10), category=""),  # duplicate wins over invalid
        _row(2, (20, 0, 30, 10), geometry_valid=False),
        _row(3, (0, 0, 9, 10)),  # IoU=.9: owner repeat, not pred duplicate
        _row(4, (40, 0, 50, 10)),
    )
    empty = (_row(0, (40, 0, 50, 10)),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((rows, (), "natural_stop"), (empty, (), "natural_stop")),
        ),
    )
    actual = ledger.images[0].trajectories[0].rows
    assert [row.outcome for row in actual] == [
        "trusted_first_hit",
        "duplicate",
        "invalid",
        "trusted_owner_repeat",
        "unmatched",
        "natural_stop",
    ]
    assert [row.immediate_credit for row in actual[1:5]] == [-0.5] * 4
    assert all(row.matched_owner_id is None for row in actual[1:5])


def test_malformed_row_equivalent_span_costs_once_and_is_scored() -> None:
    # Catches dropping malformed spans or charging by token count.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    malformed = (_MalformedRowSpan(generated_order=0, token_start=0, token_end=3),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((((), malformed, "natural_stop")), (((), malformed, "natural_stop"))),
        ),
    )
    row = ledger.images[0].trajectories[0].rows[0]
    assert row.outcome == "malformed"
    assert row.immediate_credit == -1.0
    assert row.token_indices == (0, 1, 2)
    assert all(token.scored for token in row.tokens)


def test_legacy_m_row_is_neutral_and_masks_direct_tokens_despite_positive_rtg() -> None:
    # Catches direct imitation of M rows merely because a later trusted hit succeeds.
    manifest = _manifest(
        _owner("owner-0", "G", (0, 0, 10, 10)),
        _owner("owner-1", "M", (20, 0, 30, 10)),
    )
    m_then_g = (_row(0, (20, 0, 30, 10)), _row(1, (0, 0, 10, 10)))
    m_then_bad = (_row(0, (20, 0, 30, 10)), _row(1, (40, 0, 50, 10)))
    acquisition = _acquisition(
        manifest,
        ((m_then_g, (), "natural_stop"), (m_then_bad, (), "natural_stop")),
    )
    ledger = build_trajectory_credit_ledger(manifest, acquisition)
    m_row = ledger.images[0].trajectories[0].rows[0]
    assert m_row.outcome == "legacy_m"
    assert m_row.immediate_credit == 0.0
    assert m_row.return_to_go == pytest.approx(1.0)
    assert m_row.advantage > 0.0
    assert not m_row.scored
    assert all(not token.scored for token in m_row.tokens)

    logits = {
        trajectory.request_id: torch.zeros(6, requires_grad=True)
        for trajectory in ledger.images[0].trajectories
    }
    loss = trajectory_score_function_loss(logits, ledger)
    loss.backward()
    m_gradient = logits[m_row.request_id].grad
    assert m_gradient is not None
    assert m_gradient[0].item() == 0.0


def test_natural_stop_has_direct_shortfall_but_cap_only_changes_earlier_rtg() -> None:
    # Catches positive STOP imitation or synthesizing a cap token action.
    manifest = _manifest(
        _owner("owner-0", "G", (0, 0, 10, 10)),
        _owner("owner-1", "H", (20, 0, 30, 10)),
    )
    one_hit = (_row(0, (0, 0, 10, 10)),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((one_hit, (), "natural_stop"), (one_hit, (), "cap_stop")),
        ),
    )
    natural, capped = ledger.images[0].trajectories
    assert natural.rows[-1].outcome == "natural_stop"
    assert natural.rows[-1].immediate_credit == -0.5
    assert natural.rows[-1].token_indices == (5,)
    assert capped.rows[-1].outcome == "cap_shortfall"
    assert capped.rows[-1].immediate_credit == -0.5
    assert capped.rows[-1].token_indices == ()
    assert not capped.rows[-1].scored
    assert natural.rows[0].return_to_go == pytest.approx(capped.rows[0].return_to_go)


def test_terminated_paths_supply_zero_to_later_rloo_positions() -> None:
    # Catches shrinking a late-position baseline to only surviving paths.
    manifest = _manifest(
        _owner("owner-0", "G", (0, 0, 10, 10)),
        _owner("owner-1", "H", (20, 0, 30, 10)),
    )
    short = (_row(0, (0, 0, 10, 10)),)
    long = (_row(0, (0, 0, 10, 10)), _row(1, (20, 0, 30, 10)))
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((short, (), "cap_stop"), (long, (), "natural_stop")),
        ),
    )
    # Position 1 is the long path's second row; the capped path has terminated and contributes zero.
    assert ledger.images[0].position_returns[1][0] == 0.0
    assert ledger.images[0].trajectories[1].rows[1].advantage == pytest.approx(
        ledger.images[0].trajectories[1].rows[1].return_to_go
    )


def test_tied_rloo_returns_produce_exact_zero_advantages() -> None:
    # Catches normalized or rank-based fabrication when paths are tied.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    row = (_row(0, (0, 0, 10, 10)),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((row, (), "natural_stop"), (row, (), "natural_stop")),
        ),
    )
    assert all(
        credit.advantage == 0.0
        for trajectory in ledger.images[0].trajectories
        for credit in trajectory.rows
    )


def test_stop_advantage_is_one_sided_and_negative_stop_pressure_remains() -> None:
    # Catches directly increasing STOP probability on the more-complete trajectory.
    manifest = _manifest(
        _owner("owner-0", "G", (0, 0, 10, 10)),
        _owner("owner-1", "H", (20, 0, 30, 10)),
    )
    complete = (_row(0, (0, 0, 10, 10)), _row(1, (20, 0, 30, 10)))
    missing = (_row(0, (0, 0, 10, 10)), _row(1, (40, 0, 50, 10)))
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((complete, (), "natural_stop"), (missing, (), "natural_stop")),
        ),
    )
    complete_stop = ledger.images[0].trajectories[0].rows[-1]
    missing_stop = ledger.images[0].trajectories[1].rows[-1]
    assert complete_stop.unclamped_advantage > 0.0
    assert complete_stop.advantage == 0.0
    assert missing_stop.advantage < 0.0


def test_reward_and_cost_advantages_have_opposite_score_function_gradient_signs() -> (
    None
):
    # Catches omitting the leading minus or attaching the wrong row advantage.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    hit = (_row(0, (0, 0, 10, 10)),)
    miss = (_row(0, (20, 0, 30, 10)),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((hit, (), "natural_stop"), (miss, (), "natural_stop")),
        ),
    )
    logits = {
        trajectory.request_id: torch.zeros(6, requires_grad=True)
        for trajectory in ledger.images[0].trajectories
    }
    trajectory_score_function_loss(logits, ledger).backward()
    hit_grad = logits[ledger.images[0].trajectories[0].request_id].grad
    miss_grad = logits[ledger.images[0].trajectories[1].request_id].grad
    assert hit_grad is not None and miss_grad is not None
    assert hit_grad[0].item() < 0.0
    assert miss_grad[0].item() > 0.0


def test_loss_uses_detached_selectors_and_tied_group_has_zero_gradient() -> None:
    # Catches gradients through ledger advantages or nonzero tied-group pressure.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    row = (_row(0, (0, 0, 10, 10)),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((row, (), "natural_stop"), (row, (), "natural_stop")),
        ),
    )
    logits = {
        trajectory.request_id: torch.randn(6, requires_grad=True)
        for trajectory in ledger.images[0].trajectories
    }
    trajectory_score_function_loss(logits, ledger).backward()
    assert all(
        torch.count_nonzero(value.grad).item() == 0
        for value in logits.values()
        if value.grad is not None
    )
    assert all(
        not isinstance(token.advantage, torch.Tensor) for token in ledger.scored_tokens
    )


def test_unnormalized_microsteps_and_one_logical_denominator_are_partition_invariant() -> (
    None
):
    # Catches pack-local means or applying N*K once per accumulation microstep.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    hit = (_row(0, (0, 0, 10, 10), token_end=2),)
    miss = (_row(0, (20, 0, 30, 10), token_end=3),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((hit, (), "natural_stop"), (miss, (), "natural_stop")),
        ),
    )
    full_logits = {
        trajectory.request_id: torch.randn(6, requires_grad=True)
        for trajectory in ledger.images[0].trajectories
    }
    partition_logits = {
        key: value.detach().clone().requires_grad_()
        for key, value in full_logits.items()
    }

    full = trajectory_score_function_loss(full_logits, ledger)
    full.backward()
    indices = tuple(range(len(ledger.scored_tokens)))
    parts = (indices[::3], indices[1::3], indices[2::3])
    numerator = sum(
        (
            trajectory_score_function_numerator(
                partition_logits, ledger, token_indices=part
            )
            for part in parts
        ),
        start=torch.zeros(()),
    )
    split = numerator / ledger.logical_denominator
    split.backward()
    assert torch.allclose(full, split, atol=1e-6, rtol=0)
    for key in full_logits:
        full_gradient = full_logits[key].grad
        partition_gradient = partition_logits[key].grad
        assert full_gradient is not None and partition_gradient is not None
        assert torch.allclose(full_gradient, partition_gradient, atol=1e-7, rtol=0)


def test_microstep_numerator_accepts_only_the_pack_local_request_tensors() -> None:
    # Catches coupling one physical pack to unrelated trajectories' live tensors.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    hit = (_row(0, (0, 0, 10, 10)),)
    miss = (_row(0, (20, 0, 30, 10)),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(manifest, ((hit, (), "natural_stop"), (miss, (), "natural_stop"))),
    )
    request_id = ledger.images[0].trajectories[0].request_id
    local_indices = tuple(
        index
        for index, token in enumerate(ledger.scored_tokens)
        if token.request_id == request_id
    )
    local = {request_id: torch.zeros(6, requires_grad=True)}
    numerator = trajectory_score_function_numerator(
        local, ledger, token_indices=local_indices
    )
    numerator.backward()
    assert local[request_id].grad is not None


def test_ledger_round_trip_is_content_addressed_and_rejects_forgery() -> None:
    # Catches mutable/unbound detached labels being admitted after persistence.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)), image_id=1584)
    publication = _admitted_publication(manifest)
    acquisition = credit.TrajectoryCreditPanelAcquisition((publication,))
    tokenizer_adapter = _canonical_tokenizer_adapter()
    ledger = build_trajectory_credit_ledger(
        manifest,
        acquisition,
        tokenizer_adapter=tokenizer_adapter,
    )
    payload = ledger.to_dict()
    assert ledger.images[0].trajectories[0].policy_contract_sha256 == (
        publication.replayed_group.trajectories[0].policy_contract.content_sha256
    )
    loaded = TrajectoryCreditLedger.from_dict(
        payload,
        manifest=manifest,
        acquisition=acquisition,
        tokenizer_adapter=tokenizer_adapter,
    )
    assert loaded == ledger
    assert loaded.content_sha256 == ledger.content_sha256

    forged_hash = {**payload, "content_sha256": "0" * 64}
    with pytest.raises(ValueError, match="rerun canonical projection"):
        TrajectoryCreditLedger.from_dict(
            forged_hash,
            manifest=manifest,
            acquisition=acquisition,
            tokenizer_adapter=tokenizer_adapter,
        )
    forged_lineage = {**payload, "source_sha256": "b" * 64}
    with pytest.raises(ValueError, match="rerun canonical projection"):
        TrajectoryCreditLedger.from_dict(
            forged_lineage,
            manifest=manifest,
            acquisition=acquisition,
            tokenizer_adapter=tokenizer_adapter,
        )

    semantic_forgery = copy.deepcopy(payload)
    forged_row = semantic_forgery["images"][0]["trajectories"][0]["rows"][0]
    forged_row["advantage"] += 0.25
    for token in forged_row["tokens"]:
        token["advantage"] += 0.25
    preimage = {
        key: value for key, value in semantic_forgery.items() if key != "content_sha256"
    }
    encoded = (
        json.dumps(preimage, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    semantic_forgery["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="rerun canonical projection"):
        TrajectoryCreditLedger.from_dict(
            semantic_forgery,
            manifest=manifest,
            acquisition=acquisition,
            tokenizer_adapter=tokenizer_adapter,
        )


def test_manifest_lineage_and_explicit_helper_k_fail_closed() -> None:
    # Catches silently accepting a different manifest or an accidental non-K16 group.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    row = (_row(0, (0, 0, 10, 10)),)
    acquisition = _acquisition(
        manifest,
        ((row, (), "natural_stop"), (row, (), "natural_stop")),
    )
    with pytest.raises(ValueError, match="logical K"):
        replace(acquisition, logical_k=16)
    changed_manifest = replace(manifest, full_panel=True)
    with pytest.raises(ValueError, match="manifest"):
        build_trajectory_credit_ledger(changed_manifest, acquisition)


def _two_image_manifest() -> Human13KUnionManifest:
    first = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)), image_id=1584)
    second = _manifest(_owner("owner-1", "G", (0, 0, 10, 10)), image_id=2299)
    return replace(
        first,
        images=(first.images[0], second.images[0]),
        denominators=replace(first.denominators, panel_image_count=2),
    )


def test_public_build_requires_exact_admitted_publication_and_reruns_canonical_parse() -> (
    None
):
    # Catches admitting caller-authored row semantics or a bare Task-1/2 group.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)), image_id=1584)
    publication = _admitted_publication(manifest)
    panel = credit.TrajectoryCreditPanelAcquisition((publication,))
    ledger = build_trajectory_credit_ledger(
        manifest,
        panel,
        tokenizer_adapter=_canonical_tokenizer_adapter(),
    )
    assert ledger.images[0].trajectories[0].rows[0].outcome == "trusted_first_hit"
    assert ledger.images[0].plan_sha256 == publication.execution.plan_sha256
    assert (
        ledger.images[0].parity_receipt_sha256
        == publication.binding.parity_receipt_sha256
    )

    with pytest.raises(ValueError, match="AdmittedPublication"):
        build_trajectory_credit_ledger(
            manifest,
            publication.execution.group,
            tokenizer_adapter=_canonical_tokenizer_adapter(),
        )


def test_parser_projection_rejects_rehashed_bbox_category_and_token_span_forgery() -> (
    None
):
    # Catches a content-addressed but caller-authored semantic envelope over unchanged tokens.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)), image_id=1584)
    publication = _admitted_publication(manifest)
    tokenizer_adapter = _canonical_tokenizer_adapter()
    receipt = credit.build_canonical_parser_projection_receipt(
        manifest, publication, tokenizer_adapter=tokenizer_adapter
    )
    for field, replacement in (
        ("category", "dog"),
        ("bbox", [100.0, 100.0, 200.0, 200.0]),
        ("token_end", 8),
    ):
        forged = copy.deepcopy(receipt.to_dict())
        forged["trajectories"][0]["events"][0][field] = replacement
        preimage = {
            key: value for key, value in forged.items() if key != "content_sha256"
        }
        forged["content_sha256"] = hashlib.sha256(
            (
                json.dumps(preimage, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest()
        with pytest.raises(ValueError, match="canonical parser projection"):
            credit.CanonicalParserProjectionReceipt.from_dict(
                forged,
                manifest=manifest,
                publication=publication,
                tokenizer_adapter=tokenizer_adapter,
            )


def test_plan_image_owns_owner_selection_and_panel_cell_must_be_coherent() -> None:
    # Catches image-A evidence being scored against image-B owners, cross-RP, or mixed seeds.
    single_b = _manifest(_owner("owner-1", "G", (0, 0, 10, 10)), image_id=2299)
    publication_a = _admitted_publication(
        _manifest(_owner("owner-0", "G", (0, 0, 10, 10)), image_id=1584)
    )
    with pytest.raises(ValueError, match="plan image"):
        build_trajectory_credit_ledger(
            single_b,
            credit.TrajectoryCreditPanelAcquisition((publication_a,)),
            tokenizer_adapter=_canonical_tokenizer_adapter(),
        )

    panel_manifest = _two_image_manifest()
    publication_1584 = _admitted_publication(panel_manifest, image_id=1584)
    cross_rp = _admitted_publication(
        panel_manifest, image_id=2299, repetition_penalty=1.10
    )
    with pytest.raises(ValueError, match="training RP"):
        credit.TrajectoryCreditPanelAcquisition((publication_1584, cross_rp))
    mixed_seed = _admitted_publication(
        panel_manifest, image_id=2299, seed_group_id="matrix_a"
    )
    with pytest.raises(ValueError, match="seed group"):
        credit.TrajectoryCreditPanelAcquisition((publication_1584, mixed_seed))


def test_parser_tokenizer_request_order_and_parity_lineage_fail_closed() -> None:
    # Catches wrong parser/tokenizer IDs and bypassing Task-2 aggregate admission.
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)), image_id=1584)
    publication = _admitted_publication(manifest)
    wrong_tokenizer = credit.CanonicalTokenizerDecodeAdapter(
        tokenizer=_FakeTokenizer(),
        tokenizer_id="other-tokenizer",
        tokenizer_sha256=manifest.binding.surface.tokenizer_sha256,
        implementation_id="fake-hf-tokenizer-for-cpu-tests.v1",
    )
    with pytest.raises(ValueError, match="tokenizer identity"):
        build_trajectory_credit_ledger(
            manifest,
            credit.TrajectoryCreditPanelAcquisition((publication,)),
            tokenizer_adapter=wrong_tokenizer,
        )
    wrong_parser_manifest = replace(
        manifest,
        binding=replace(
            manifest.binding,
            surface=replace(manifest.binding.surface, parser="other-parser-policy"),
        ),
    )
    with pytest.raises(ValueError, match="parser policy"):
        build_trajectory_credit_ledger(
            wrong_parser_manifest,
            credit.TrajectoryCreditPanelAcquisition((publication,)),
            tokenizer_adapter=_canonical_tokenizer_adapter(),
        )

    other = _admitted_publication(manifest, seed_group_id="matrix_a")
    forged = object.__new__(task2.AdmittedPublication)
    object.__setattr__(forged, "binding", publication.binding)
    object.__setattr__(forged, "execution", publication.execution)
    object.__setattr__(forged, "replayed_group", publication.replayed_group)
    object.__setattr__(forged, "parity_receipt", other.parity_receipt)
    with pytest.raises(ValueError, match="parity"):
        credit.TrajectoryCreditPanelAcquisition((forged,))

    forged_execution = object.__new__(task2.AcquisitionExecution)
    for name in (
        "plan",
        "plan_sha256",
        "native_batch_receipts",
        "group",
    ):
        object.__setattr__(forged_execution, name, getattr(publication.execution, name))
    object.__setattr__(
        forged_execution,
        "plan_request_ids",
        tuple(reversed(publication.execution.plan_request_ids)),
    )
    forged_order = object.__new__(task2.AdmittedPublication)
    object.__setattr__(forged_order, "binding", publication.binding)
    object.__setattr__(forged_order, "execution", forged_execution)
    object.__setattr__(forged_order, "replayed_group", publication.replayed_group)
    object.__setattr__(forged_order, "parity_receipt", publication.parity_receipt)
    with pytest.raises(ValueError, match="request order"):
        credit.TrajectoryCreditPanelAcquisition((forged_order,))
