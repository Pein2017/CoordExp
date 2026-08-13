from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.research.build_human13_on_policy_frontier import (
    FrontierCandidateAlias,
    FrontierDuplicateEvent,
    FrontierImage,
    FrontierRow,
)
from scripts.research.human13_frontier_selection import (
    CandidatePath,
    CandidateScore,
    TokenDecision,
)
from scripts.research.human13_on_policy_live import (
    BehaviorGateObservation,
    build_on_policy_payload,
    evaluate_behavior_gate,
    materialize_on_policy_segments,
)


class _Skeleton(SimpleNamespace):
    pass


def _skeleton() -> _Skeleton:
    return _Skeleton(
        example_id="source",
        input_ids=(100, 101, 700),
        prompt_token_count=2,
        image_encoding=SimpleNamespace(
            image_grid_thw=(1, 1, 1),
            plan=SimpleNamespace(merge_size=1),
        ),
        image_grid_thw=(1, 1, 1),
        image_pad_physical_start=0,
        image_pad_physical_end=1,
        image_token_id=151655,
    )


def _frontier(*, duplicate: bool = True) -> FrontierImage:
    rows = (
        FrontierRow(0, "person", (0, 0, 10, 10), 0, 4, (10, 11, 12, 13)),
        FrontierRow(1, "cat", (20, 0, 30, 10), 4, 8, (20, 21, 22, 23)),
        FrontierRow(2, "dog", (20.1, 0, 30.1, 10), 8, 12, (20, 21, 29, 23)),
    )
    duplicates = (FrontierDuplicateEvent(2, 1, 0.98),) if duplicate else ()
    return FrontierImage(
        image_id=7,
        trajectory_id="current",
        generated_token_ids=(10, 11, 12, 13, 20, 21, 22, 23, 20, 21, 29, 23, 999),
        parser="compact_object_box_closed_only",
        parser_status="complete",
        stop_reason="im_end",
        rows=rows,
        canonical_owner_ids=("g0", "g1"),
        constrained_protected_owner_ids=("g0", "g1"),
        covered_h_owner_ids=(),
        uncovered_h_owner_ids=("h0",),
        candidate_aliases=(
            FrontierCandidateAlias("h0", "alias", "k", 21001, 0.9, (30, 31, 32, 33)),
        ),
        duplicate_events=duplicates,
    )


def _score() -> CandidateScore:
    path = CandidatePath(7, "h0", "alias", (30, 31, 32, 33))
    sites = (
        TokenDecision(0, 30, 30, 1, 3.0, 1.0, 2.0, 1),
        TokenDecision(1, 31, 9, 9, 1.0, 2.0, -1.0, 1),
        TokenDecision(2, 32, 32, 4, 3.0, 1.0, 2.0, 1),
        TokenDecision(3, 33, 33, 5, 3.0, 1.0, 2.0, 1),
    )
    return CandidateScore(path, sites, sites, (), 1.0, 1, 1, False, 1.0, 0.0, False)


def test_materialize_uses_dedup_current_prefix_and_keeps_raw_digest() -> None:
    materialized = materialize_on_policy_segments(
        {7: _frontier()},
        {7: _skeleton()},
        selected_scores={7: _score()},
        arm_id="O-First-Safe",
        required_margin=0.25,
    )

    assert len(materialized.segments) == 2
    positive, duplicate = materialized.segments
    assert positive.encoded_example.input_ids == (
        100,
        101,
        10,
        11,
        12,
        13,
        20,
        21,
        22,
        23,
        30,
        31,
        32,
        33,
    )
    assert materialized.prefix_receipts[0].raw_natural_token_count == 13
    assert materialized.prefix_receipts[0].training_prefix_token_count == 8
    bindings = positive.encoded_example.human13_on_policy_bindings
    assert [(item.objective, item.token_indices) for item in bindings] == [
        ("first_bottleneck", (11,)),
        ("rectangle", (12, 13)),
    ]
    dup_binding = duplicate.encoded_example.human13_on_policy_bindings[0]
    assert dup_binding.objective == "duplicate_pair"
    assert dup_binding.duplicate_token_id == 20
    assert dup_binding.target_token_ids == (30,)


def test_full_arm_selects_complete_row_without_terminal() -> None:
    materialized = materialize_on_policy_segments(
        {7: _frontier(duplicate=False)},
        {7: _skeleton()},
        selected_scores={7: _score()},
        arm_id="O-Full-Safe",
        required_margin=0.25,
    )
    binding = materialized.segments[0].encoded_example.human13_on_policy_bindings[0]
    assert binding.objective == "full_row_ce"
    assert binding.token_indices == (14, 15, 16, 17)


def test_first_arm_rejects_candidate_without_hf_blocker() -> None:
    score = _score()
    score = CandidateScore(
        score.path,
        score.hf_sites,
        score.packed_sites,
        score.aligned_sites,
        0.0,
        None,
        None,
        False,
        0.0,
        0.0,
        False,
    )
    with pytest.raises(ValueError, match="no HF blocker"):
        materialize_on_policy_segments(
            {7: _frontier(duplicate=False)},
            {7: _skeleton()},
            selected_scores={7: score},
            arm_id="O-First-Safe",
            required_margin=0.25,
        )


def test_build_payload_packs_no_padding_and_dispatches_all_objectives() -> None:
    materialized = materialize_on_policy_segments(
        {7: _frontier()},
        {7: _skeleton()},
        selected_scores={7: _score()},
        arm_id="O-First-Safe",
        required_margin=0.25,
    )
    payload = build_on_policy_payload(
        materialized,
        arm_id="O-First-Safe",
        expected_vocab_size=64,
        vocab_groups=SimpleNamespace(),
        global_max_length=128,
        required_margin=0.25,
        rectangle_margin=0.25,
        duplicate_margin=0.25,
    )
    objectives = {
        site.objective for sites in payload.sites_by_pack.values() for site in sites
    }
    assert objectives == {"first_bottleneck", "rectangle", "duplicate_pair"}
    assert sum(pack.pack.length for pack in payload.packed_plan.packs) == sum(
        len(segment.encoded_example.input_ids) for segment in materialized.segments
    )


def test_behavior_gate_rejects_owner_exchange_and_burden() -> None:
    prior = BehaviorGateObservation(
        protected_owner_ids=("g0", "g1"),
        jointly_coverable_protected_owner_ids=("g0", "g1"),
        unique_owner_ids=("g0", "g1"),
        cap_hit_count=0,
        malformed_row_count=0,
        row_count=10,
        duplicate_count=2,
    )
    exchange = BehaviorGateObservation(
        protected_owner_ids=("g0", "g1"),
        jointly_coverable_protected_owner_ids=("g0",),
        unique_owner_ids=("g0", "h0"),
        cap_hit_count=0,
        malformed_row_count=0,
        row_count=10,
        duplicate_count=2,
    )
    verdict = evaluate_behavior_gate(prior, exchange)
    assert verdict.accepted is False
    assert "protected_owner_loss" in verdict.reasons

    accepted = BehaviorGateObservation(
        protected_owner_ids=("g0", "g1"),
        jointly_coverable_protected_owner_ids=("g0", "g1"),
        unique_owner_ids=("g0", "g1", "h0"),
        cap_hit_count=0,
        malformed_row_count=0,
        row_count=12,
        duplicate_count=4,
    )
    assert evaluate_behavior_gate(prior, accepted).accepted is True


def test_on_policy_loss_runner_has_finite_gradient() -> None:
    materialized = materialize_on_policy_segments(
        {7: _frontier()},
        {7: _skeleton()},
        selected_scores={7: _score()},
        arm_id="O-First-Safe",
        required_margin=0.25,
    )
    payload = build_on_policy_payload(
        materialized,
        arm_id="O-First-Safe",
        expected_vocab_size=64,
        vocab_groups=SimpleNamespace(),
        global_max_length=128,
        required_margin=0.25,
        rectangle_margin=0.25,
        duplicate_margin=0.25,
    )
    logits = torch.zeros(
        (1, sum(len(s.encoded_example.input_ids) for s in materialized.segments), 64),
        requires_grad=True,
    )
    from scripts.research.human13_on_policy_live import (
        OnPolicyPackContext,
        OnPolicyLossRunner,
    )

    sites = tuple(site for group in payload.sites_by_pack.values() for site in group)
    runner = OnPolicyLossRunner(
        payload.denominators,
        required_margin=0.25,
        rectangle_margin=0.25,
        duplicate_margin=0.25,
    )
    plan = runner.prepare_planned_step(payload.micro_steps)
    bundle = runner.compute_micro_step(
        OnPolicyPackContext(logits, None, sites), plan, local_micro_step_index=0
    )
    bundle.total_loss.backward()

    assert torch.isfinite(bundle.total_loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
