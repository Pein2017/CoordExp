from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import math

import pytest
import torch

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
    AcquisitionGroupCreditInput,
    MalformedRowSpan,
    ParsedCreditRow,
    ParsedTrajectoryCreditInput,
    TrajectoryCreditAcquisition,
    TrajectoryCreditLedger,
    build_trajectory_credit_ledger,
    trajectory_score_function_loss,
    trajectory_score_function_numerator,
)


STOP = 99
SOURCE = "a" * 64


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
) -> ParsedCreditRow:
    start = order if token_start is None else token_start
    end = start + 1 if token_end is None else token_end
    return ParsedCreditRow(
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
        tuple[tuple[ParsedCreditRow, ...], tuple[MalformedRowSpan, ...], str], ...
    ],
    *,
    logical_k: int | None = None,
) -> TrajectoryCreditAcquisition:
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
        ParsedTrajectoryCreditInput(
            request_id=trajectory.identity.request_id,
            rows=rows,
            malformed_spans=malformed,
        )
        for trajectory, (rows, malformed, _) in zip(complete, trajectories, strict=True)
    )
    return TrajectoryCreditAcquisition(
        groups=(
            AcquisitionGroupCreditInput(manifest.images[0].image_id, group, parsed),
        ),
        logical_k=len(parsed) if logical_k is None else logical_k,
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
    malformed = (MalformedRowSpan(generated_order=0, token_start=0, token_end=3),)
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
    manifest = _manifest(_owner("owner-0", "G", (0, 0, 10, 10)))
    row = (_row(0, (0, 0, 10, 10)),)
    ledger = build_trajectory_credit_ledger(
        manifest,
        _acquisition(
            manifest,
            ((row, (), "natural_stop"), (row, (), "natural_stop")),
        ),
    )
    payload = ledger.to_dict()
    assert ledger.images[0].trajectories[0].policy_contract_sha256 == (
        _acquisition(
            manifest,
            ((row, (), "natural_stop"), (row, (), "natural_stop")),
        )
        .groups[0]
        .acquisition_group.trajectories[0]
        .policy_contract.content_sha256
    )
    assert TrajectoryCreditLedger.from_dict(payload) == ledger
    assert (
        TrajectoryCreditLedger.from_dict(payload).content_sha256
        == ledger.content_sha256
    )

    forged_hash = {**payload, "content_sha256": "0" * 64}
    with pytest.raises(ValueError, match="content SHA-256"):
        TrajectoryCreditLedger.from_dict(forged_hash)
    forged_lineage = {**payload, "source_sha256": "b" * 64}
    with pytest.raises(ValueError, match="content SHA-256"):
        TrajectoryCreditLedger.from_dict(forged_lineage)

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
    with pytest.raises(ValueError, match="RLOO advantage"):
        TrajectoryCreditLedger.from_dict(semantic_forgery)


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
