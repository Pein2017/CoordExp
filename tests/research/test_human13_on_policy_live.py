from __future__ import annotations

from dataclasses import replace
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
        FrontierRow(1, "cat", (20, 0, 30, 10), 5, 9, (20, 21, 22, 23)),
        FrontierRow(2, "dog", (20.1, 0, 30.1, 10), 9, 13, (20, 21, 29, 23)),
    )
    duplicates = (FrontierDuplicateEvent(2, 1, 0.98),) if duplicate else ()
    return FrontierImage(
        image_id=7,
        trajectory_id="current",
        generated_token_ids=(
            10,
            11,
            12,
            13,
            777,
            20,
            21,
            22,
            23,
            20,
            21,
            29,
            23,
            999,
        ),
        parser="compact_object_box_closed_only",
        parser_status="complete",
        stop_reason="im_end",
        rows=rows,
        canonical_owner_ids=("g0", "g1"),
        constrained_protected_owner_ids=("g0", "g1"),
        covered_h_owner_ids=(),
        uncovered_h_owner_ids=("h0",),
        candidate_aliases=(
            FrontierCandidateAlias(
                "h0", "alias", "k", 21001, 0.9, (151700, 151701, 151702, 151703)
            ),
        ),
        duplicate_events=duplicates,
    )


def _score() -> CandidateScore:
    path = CandidatePath(7, "h0", "alias", (151700, 151701, 151702, 151703))
    sites = (
        TokenDecision(0, 151700, 151700, 1, 3.0, 1.0, 2.0, 1),
        TokenDecision(1, 151701, 9, 9, 1.0, 2.0, -1.0, 1),
        TokenDecision(2, 151702, 151702, 4, 3.0, 1.0, 2.0, 1),
        TokenDecision(3, 151703, 151703, 5, 3.0, 1.0, 2.0, 1),
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
        777,
        20,
        21,
        22,
        23,
        151700,
        151701,
        151702,
        151703,
    )
    assert materialized.prefix_receipts[0].raw_natural_token_count == 14
    assert materialized.prefix_receipts[0].training_prefix_token_count == 9
    bindings = positive.encoded_example.human13_on_policy_bindings
    assert [(item.objective, item.token_indices) for item in bindings] == [
        ("first_bottleneck", (12,)),
        ("rectangle", (13, 14)),
    ]
    dup_binding = duplicate.encoded_example.human13_on_policy_bindings[0]
    assert dup_binding.objective == "duplicate_pair"
    assert dup_binding.duplicate_token_id == 20
    assert dup_binding.target_token_ids == (151700,)


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
    assert binding.token_indices == (15, 16, 17, 18)


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


def test_first_arm_margin_adds_candidate_aligned_surface_reserve() -> None:
    score = replace(_score(), max_surface_margin_drift=0.75)
    materialized = materialize_on_policy_segments(
        {7: _frontier(duplicate=False)},
        {7: _skeleton()},
        selected_scores={7: score},
        arm_id="O-First-Safe",
        required_margin=0.25,
    )
    binding = materialized.segments[0].encoded_example.human13_on_policy_bindings[0]
    assert binding.required_margin == 1.0


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
        expected_vocab_size=152700,
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
    assert payload.loss_config.required_margin == 0.25
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
        expected_vocab_size=152700,
        vocab_groups=SimpleNamespace(),
        global_max_length=128,
        required_margin=0.25,
        rectangle_margin=0.25,
        duplicate_margin=0.25,
    )
    logits = torch.zeros(
        (
            1,
            sum(len(s.encoded_example.input_ids) for s in materialized.segments),
            152700,
        ),
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


def test_coordinate_offsets_reject_noncanonical_row_tokens() -> None:
    frontier = _frontier(duplicate=False)
    alias = replace(frontier.candidate_aliases[0], token_ids=(30, 31, 32, 33))
    score = replace(_score(), path=replace(_score().path, token_ids=alias.token_ids))
    with pytest.raises(ValueError, match="four canonical coordinate"):
        materialize_on_policy_segments(
            {7: replace(frontier, candidate_aliases=(alias,))},
            {7: _skeleton()},
            selected_scores={7: score},
            arm_id="O-Full-Safe",
            required_margin=0.25,
        )


def test_loss_runner_rejects_payload_margin_drift() -> None:
    from scripts.research.human13_on_policy_live import OnPolicyLossRunner

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
        expected_vocab_size=152700,
        vocab_groups=SimpleNamespace(),
        global_max_length=128,
        required_margin=0.25,
        rectangle_margin=0.25,
        duplicate_margin=0.25,
    )
    runner = OnPolicyLossRunner(payload.denominators, 0.5, 0.25, 0.25)
    with pytest.raises(ValueError, match="loss configuration"):
        runner.prepare_planned_step(payload.micro_steps)


def _duplicate_raw_loss(duplicate_events: tuple[FrontierDuplicateEvent, ...]) -> float:
    from scripts.research.human13_on_policy_live import (
        OnPolicyLossRunner,
        OnPolicyPackContext,
    )

    frontier = replace(_frontier(), duplicate_events=duplicate_events)
    materialized = materialize_on_policy_segments(
        {7: frontier},
        {7: _skeleton()},
        selected_scores={7: _score()},
        arm_id="O-First-Safe",
        required_margin=0.25,
        global_max_length=20,
    )
    payload = build_on_policy_payload(
        materialized,
        arm_id="O-First-Safe",
        expected_vocab_size=152700,
        vocab_groups=SimpleNamespace(),
        global_max_length=20,
        required_margin=0.25,
        rectangle_margin=0.25,
        duplicate_margin=0.25,
    )
    runner = OnPolicyLossRunner(payload.denominators, 0.25, 0.25, 0.25)
    plan = runner.prepare_planned_step(payload.micro_steps)
    total = 0.0
    for micro_step in payload.micro_steps:
        assert micro_step.metadata is not None
        sites = micro_step.metadata["human13_on_policy_sites"]
        logits = torch.zeros((1, micro_step.pack.length, 152700))
        bundle = runner.compute_micro_step(
            OnPolicyPackContext(logits, None, sites),
            plan,
            local_micro_step_index=0,
        )
        total += sum(
            float(term.raw_loss) for term in bundle.terms if term.name == "duplicate"
        )
    return total


def test_duplicate_events_are_image_balanced_across_physical_packs() -> None:
    one = (FrontierDuplicateEvent(2, 1, 0.98),)
    two = (*one, FrontierDuplicateEvent(2, 1, 0.99))

    assert _duplicate_raw_loss(two) == pytest.approx(_duplicate_raw_loss(one))
