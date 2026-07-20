from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.common.errors import LossContractError
from src.losses import (
    CandidatePath,
    GateSiteIdentity,
    RolloutGateSite,
    first_wrong_coordinate_preference,
    grouped_entity_transition_preference,
    rollout_site_token_type_gate,
)


def test_entity_transition_groups_aliases_before_distinct_owner_smooth_max() -> None:
    owner_a = _path("a-1", "owner-a", ((0.0, 2.0, -1.0, 0.5),), (1,))
    duplicate_alias = _path("a-2", "owner-a", ((0.0, 2.0, -1.0, 0.5),), (1,))
    harmful = _path("duplicate", "covered-owner", ((0.0, -1.0, 2.0, 0.5),), (2,))

    one_alias = grouped_entity_transition_preference(
        (owner_a,),
        harmful,
        margin=0.4,
        smooth_max_temperature=0.7,
    )
    duplicate = grouped_entity_transition_preference(
        (owner_a, duplicate_alias),
        harmful,
        margin=0.4,
        smooth_max_temperature=0.7,
    )

    assert torch.allclose(duplicate.positive_score, one_alias.positive_score)
    assert torch.allclose(duplicate.raw_loss, one_alias.raw_loss)
    assert duplicate.positive_path_count == 2
    assert duplicate.distinct_owner_count == 1
    assert duplicate.duplicate_alias_count == 1
    assert duplicate.owner_scores[0].alias_count == 2


def test_entity_transition_supports_multiple_valid_owners_with_normalized_smooth_max() -> (
    None
):
    logits = ((0.0, 2.0, -1.0, 0.5),)
    positive_a = _path("a", "owner-a", logits, (1,))
    positive_b = _path("b", "owner-b", logits, (1,))
    harmful = _path("duplicate", "covered-owner", ((0.0, -1.0, 2.0, 0.5),), (2,))

    result = grouped_entity_transition_preference(
        (positive_a, positive_b),
        harmful,
        margin=0.2,
        smooth_max_temperature=0.5,
    )

    expected_owner_score = torch.log_softmax(torch.tensor(logits[0]), dim=0)[1]
    expected_harmful = torch.log_softmax(harmful.logits.float(), dim=1)[0, 2]
    assert torch.allclose(result.positive_score, expected_owner_score)
    assert torch.allclose(result.harmful_score, expected_harmful)
    assert torch.allclose(result.target_margin, expected_owner_score - expected_harmful)
    assert torch.allclose(
        result.raw_loss,
        F.softplus(torch.tensor(0.2) - expected_owner_score + expected_harmful),
    )
    assert result.distinct_owner_count == 2


def test_premature_terminal_harmful_path_is_null_owner_and_exactly_one_token() -> None:
    positive = _path(
        "rescue",
        "owner-a",
        ((0.0, 3.0, -1.0, 0.5), (0.0, -1.0, 2.0, 0.5)),
        (1, 2),
    )
    terminal_logits = torch.tensor(((0.0, -1.0, 0.5, 2.5),))
    terminal = CandidatePath(
        candidate_id="terminal",
        physical_owner_id=None,
        logits=terminal_logits,
        target_token_ids=(3,),
        premature_terminal=True,
    )

    result = grouped_entity_transition_preference(
        (positive,),
        terminal,
        margin=0.1,
        smooth_max_temperature=0.5,
    )

    expected = torch.log_softmax(terminal_logits.float(), dim=1)[0, 3]
    assert result.harmful_path.physical_owner_id is None
    assert result.harmful_path.token_count == 1
    assert torch.allclose(result.harmful_score, expected)

    invalid = CandidatePath(
        candidate_id="terminal-two-token",
        physical_owner_id=None,
        logits=torch.zeros((2, 4)),
        target_token_ids=(3, 3),
        premature_terminal=True,
    )
    with pytest.raises(LossContractError) as exc_info:
        grouped_entity_transition_preference(
            (positive,),
            invalid,
            margin=0.1,
            smooth_max_temperature=0.5,
        )
    assert exc_info.value.code == "loss.rollout_terminal_path_length"


def test_entity_and_coordinate_objectives_upcast_to_fp32_and_keep_finite_gradients() -> (
    None
):
    positive_logits = torch.tensor(
        ((0.0, 2.0, -1.0, 0.5),),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    harmful_logits = torch.tensor(
        ((0.0, -1.0, 2.0, 0.5),),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    entity = grouped_entity_transition_preference(
        (CandidatePath("positive", "owner", positive_logits, (1,)),),
        CandidatePath("harmful", "covered", harmful_logits, (2,)),
        margin=0.2,
        smooth_max_temperature=0.5,
    )
    assert entity.raw_loss.dtype == torch.float32
    assert entity.target_margin.dtype == torch.float32
    assert entity.finite.all_finite
    entity.raw_loss.backward()
    assert positive_logits.grad is not None and harmful_logits.grad is not None
    assert torch.isfinite(positive_logits.grad).all()
    assert torch.isfinite(harmful_logits.grad).all()

    coordinate_logits = torch.tensor(
        (0.0, 2.0, 1.0, -1.0, 3.0),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    coordinate = first_wrong_coordinate_preference(
        coordinate_logits,
        acceptable_token_ids=(1, 2),
        wrong_token_id=4,
        margin=0.3,
    )
    assert coordinate.raw_loss.dtype == torch.float32
    assert coordinate.finite.all_finite
    coordinate.raw_loss.backward()
    assert coordinate_logits.grad is not None
    assert torch.isfinite(coordinate_logits.grad).all()


def test_coordinate_preference_uses_discrete_resolved_acceptable_token_set() -> None:
    logits = torch.tensor((0.0, 2.0, 1.0, -1.0, 3.0), requires_grad=True)
    result = first_wrong_coordinate_preference(
        logits,
        acceptable_token_ids=(1, 2),
        wrong_token_id=4,
        margin=0.3,
    )

    expected_mass = torch.logsumexp(logits.float()[torch.tensor((1, 2))], dim=0)
    expected_margin = expected_mass - logits.float()[4]
    assert torch.allclose(result.acceptable_mass_score, expected_mass)
    assert torch.allclose(result.target_margin, expected_margin)
    assert torch.allclose(
        result.raw_loss, F.softplus(torch.tensor(0.3) - expected_margin)
    )
    assert result.acceptable_token_count == 2

    with pytest.raises(LossContractError) as duplicate_exc:
        first_wrong_coordinate_preference(
            logits,
            acceptable_token_ids=(1, 1),
            wrong_token_id=4,
            margin=0.3,
        )
    assert duplicate_exc.value.code == "loss.rollout_token_ids_duplicate"

    with pytest.raises(LossContractError) as overlap_exc:
        first_wrong_coordinate_preference(
            logits,
            acceptable_token_ids=(1, 4),
            wrong_token_id=4,
            margin=0.3,
        )
    assert overlap_exc.value.code == "loss.rollout_coordinate_wrong_is_acceptable"


def test_terminal_boundary_gate_uses_declared_schema_group_not_terminal_group() -> None:
    logits = torch.tensor((0.0, 2.0, -1.0, 4.0), requires_grad=True)
    site = RolloutGateSite(
        event_id="terminal-event",
        identity=GateSiteIdentity("terminal-candidate", 11),
        logits=logits,
        intended_token_type="schema",
        allowed_token_ids=(1,),
    )

    result = rollout_site_token_type_gate((site,))

    all_mass = torch.logsumexp(logits.float(), dim=0)
    expected_schema_loss = all_mass - logits.float()[1]
    terminal_loss = all_mass - logits.float()[3]
    assert torch.allclose(result.raw_loss, expected_schema_loss)
    assert not torch.allclose(result.raw_loss, terminal_loss)
    assert torch.allclose(result.legal_mass, torch.exp(-expected_schema_loss))
    result.raw_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_rollout_gate_deduplicates_identical_sites_and_rejects_type_conflicts() -> None:
    logits = torch.tensor((0.0, 2.0, -1.0, 4.0))
    identity = GateSiteIdentity("candidate-a", 5)
    schema = RolloutGateSite("event-a", identity, logits, "schema", (1,))

    result = rollout_site_token_type_gate((schema, schema))

    assert result.declaration_count == 2
    assert result.distinct_site_count == 1
    assert result.duplicate_declaration_count == 1
    assert result.eligible_event_count == 1

    coordinate = RolloutGateSite("event-a", identity, logits, "coordinate", (2,))
    with pytest.raises(LossContractError) as exc_info:
        rollout_site_token_type_gate((schema, coordinate))
    assert exc_info.value.code == "loss.rollout_gate_type_conflict"


def test_rollout_gate_averages_sites_within_event_then_events() -> None:
    one_site_event = _gate_site("event-one", "one", 0, (0.0, 2.0, -1.0), (1,))
    three_site_event = (
        _gate_site("event-three", "a", 0, (3.0, 0.0, -1.0), (1,)),
        _gate_site("event-three", "b", 1, (0.0, 1.0, 2.0), (1,)),
        _gate_site("event-three", "c", 2, (1.0, -2.0, 0.0), (1,)),
    )
    declarations = (one_site_event, *three_site_event)

    result = rollout_site_token_type_gate(declarations)

    site_losses = tuple(
        _gate_loss(site.logits, site.allowed_token_ids) for site in declarations
    )
    expected_event_balanced = (site_losses[0] + torch.stack(site_losses[1:]).mean()) / 2
    global_site_mean = torch.stack(site_losses).mean()
    assert torch.allclose(result.raw_loss, expected_event_balanced)
    assert not torch.allclose(result.raw_loss, global_site_mean)
    assert result.eligible_event_count == 2
    assert result.distinct_site_count == 4
    assert [event.distinct_site_count for event in result.events] == [1, 3]


def _path(
    candidate_id: str,
    owner_id: str,
    logits: tuple[tuple[float, ...], ...],
    targets: tuple[int, ...],
) -> CandidatePath:
    return CandidatePath(
        candidate_id=candidate_id,
        physical_owner_id=owner_id,
        logits=torch.tensor(logits),
        target_token_ids=targets,
    )


def _gate_site(
    event_id: str,
    segment_id: str,
    logits_position: int,
    logits: tuple[float, ...],
    allowed: tuple[int, ...],
) -> RolloutGateSite:
    return RolloutGateSite(
        event_id=event_id,
        identity=GateSiteIdentity(segment_id, logits_position),
        logits=torch.tensor(logits),
        intended_token_type="schema",
        allowed_token_ids=allowed,
    )


def _gate_loss(logits: torch.Tensor, allowed: tuple[int, ...]) -> torch.Tensor:
    row = logits.float()
    return torch.logsumexp(row, dim=0) - torch.logsumexp(
        row[torch.tensor(allowed)],
        dim=0,
    )
