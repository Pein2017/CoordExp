"""Pure loss math for reviewed rollout-calibration decision sites.

The records in this module are deliberately independent of ``TokenAtom``.
Rollout calibration scores exact model-produced candidate tokens, while its
token-type gate describes the intended semantic phase at a causal site.  In
particular, a premature terminal token is still gated as object-row schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from src.common.errors import LossContractError
from src.losses.vocab import V1_TOKEN_TYPES


@dataclass(frozen=True)
class CandidatePath:
    """One admitted candidate path and the logits that causally score it."""

    candidate_id: str
    physical_owner_id: str | None
    logits: torch.Tensor
    target_token_ids: tuple[int, ...]
    premature_terminal: bool = False
    # Optional semantic labels for the candidate-row tokens.  The transition
    # branch itself does not require them, which keeps old diagnostic callers
    # source-compatible.  Training events that request coherent-row
    # continuation provide one label per token.
    token_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CandidatePathScore:
    candidate_id: str
    physical_owner_id: str | None
    summed_log_probability: torch.Tensor
    mean_log_probability: torch.Tensor
    token_count: int


@dataclass(frozen=True)
class OwnerScore:
    physical_owner_id: str
    score: torch.Tensor
    alias_count: int


@dataclass(frozen=True)
class LossFiniteDiagnostics:
    all_finite: bool
    values: tuple[tuple[str, bool], ...]

    def is_finite(self, name: str) -> bool:
        for observed_name, finite in self.values:
            if observed_name == name:
                return finite
        raise KeyError(name)


@dataclass(frozen=True)
class EntityTransitionPreferenceResult:
    raw_loss: torch.Tensor
    target_margin: torch.Tensor
    positive_score: torch.Tensor
    harmful_score: torch.Tensor
    positive_paths: tuple[CandidatePathScore, ...]
    harmful_path: CandidatePathScore
    owner_scores: tuple[OwnerScore, ...]
    positive_path_count: int
    distinct_owner_count: int
    duplicate_alias_count: int
    first_divergence_offsets: tuple[tuple[str, int], ...]
    positive_continuation_loss: torch.Tensor
    positive_schema_description_loss: torch.Tensor
    positive_coordinate_loss: torch.Tensor
    finite: LossFiniteDiagnostics
    math_dtype: str = "float32"


@dataclass(frozen=True)
class PositivePathImitationResult:
    """Grouped exact-token continuation loss for one complete positive row."""

    raw_loss: torch.Tensor
    schema_description_loss: torch.Tensor
    coordinate_loss: torch.Tensor
    selected_token_count: int
    schema_description_token_count: int
    coordinate_token_count: int
    finite: LossFiniteDiagnostics
    math_dtype: str = "float32"


@dataclass(frozen=True)
class CoordinateBoundaryPreferenceResult:
    raw_loss: torch.Tensor
    target_margin: torch.Tensor
    acceptable_mass_score: torch.Tensor
    wrong_token_score: torch.Tensor
    acceptable_token_count: int
    finite: LossFiniteDiagnostics
    math_dtype: str = "float32"


@dataclass(frozen=True)
class GateSiteIdentity:
    """Stable causal-site identity; ``segment_id`` must be globally unique."""

    segment_id: str
    logits_position: int


@dataclass(frozen=True)
class RolloutGateSite:
    event_id: str
    identity: GateSiteIdentity
    logits: torch.Tensor
    intended_token_type: Literal["desc_text", "schema", "coordinate", "eos"]
    allowed_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class EventGateResult:
    event_id: str
    raw_loss: torch.Tensor
    legal_mass: torch.Tensor
    distinct_site_count: int


@dataclass(frozen=True)
class RolloutSiteTokenTypeGateResult:
    raw_loss: torch.Tensor
    legal_mass: torch.Tensor
    events: tuple[EventGateResult, ...]
    eligible_event_count: int
    distinct_site_count: int
    declaration_count: int
    duplicate_declaration_count: int
    finite: LossFiniteDiagnostics
    math_dtype: str = "float32"


@dataclass(frozen=True)
class _TransitionPathDetail:
    path: CandidatePath
    scored: CandidatePathScore
    first_divergence_offset: int
    positive_branch_log_probability: torch.Tensor
    harmful_branch_log_probability: torch.Tensor
    branch_margin: torch.Tensor
    continuation_loss: torch.Tensor
    schema_description_loss: torch.Tensor
    coordinate_loss: torch.Tensor


def grouped_entity_transition_preference(
    positive_paths: tuple[CandidatePath, ...],
    harmful_path: CandidatePath,
    *,
    margin: float,
    smooth_max_temperature: float,
) -> EntityTransitionPreferenceResult:
    """Compare valid owners and a harmful branch at their shared fork.

    A complete positive row and a one-token terminal branch must not be
    compared by their total sequence likelihoods.  For each positive path we
    therefore find the first token mismatch, compare only the two logits at
    that shared-history decision, and then apply a separate coherent-row
    continuation loss to the selected positive path.
    """

    checked_margin = _finite_scalar(margin, name="margin")
    # Validate the legacy public knob even though this treatment deliberately
    # uses a hard owner maximum for its at-least-one-owner semantics.
    _positive_finite_scalar(
        smooth_max_temperature,
        name="smooth_max_temperature",
    )
    if not positive_paths:
        raise LossContractError(
            "entity-transition preference requires at least one positive path",
            code="loss.rollout_entity_positive_empty",
        )

    positive_scores: list[CandidatePathScore] = []
    # Keep branch and continuation details together until aliases have been
    # reduced to one candidate per physical owner.
    positive_details: list[_TransitionPathDetail]
    for path in positive_paths:
        _validate_candidate_identity(path, role="positive")
        scored = _score_candidate_path(path)
        positive_scores.append(scored)

    _validate_candidate_identity(harmful_path, role="harmful")
    scored_harmful = _score_candidate_path(harmful_path)
    # The harmful path is paired independently with every positive path: its
    # first divergent position can differ when aliases use different prefixes.
    positive_details = [
        _transition_path_detail(path, scored, harmful_path=harmful_path)
        for path, scored in zip(positive_paths, positive_scores, strict=True)
    ]
    scores_by_owner: dict[str, list[_TransitionPathDetail]] = {}
    for detail in positive_details:
        assert detail.path.physical_owner_id is not None
        scores_by_owner.setdefault(detail.path.physical_owner_id, []).append(detail)

    owner_scores = tuple(
        OwnerScore(
            physical_owner_id=owner_id,
            score=max(
                alias_scores,
                key=lambda item: float(item.branch_margin.detach().item()),
            ).branch_margin,
            alias_count=len(alias_scores),
        )
        for owner_id, alias_scores in sorted(scores_by_owner.items())
    )
    # ``temperature`` remains part of the public configuration for backwards
    # compatibility, but the treatment's semantics are explicitly "at least
    # one owner beats harm".  A hard owner maximum avoids rewarding an event
    # merely because it has many weak aliases or owners.
    winning_owner_index = int(
        torch.stack(tuple(item.score for item in owner_scores)).argmax().item()
    )
    winning_owner = owner_scores[winning_owner_index].physical_owner_id
    winning_detail = max(
        scores_by_owner[winning_owner],
        key=lambda item: float(item.branch_margin.detach().item()),
    )
    positive_score = winning_detail.positive_branch_log_probability
    harmful_score = winning_detail.harmful_branch_log_probability
    target_margin = winning_detail.branch_margin
    branch_loss = F.softplus(
        positive_score.new_tensor(checked_margin) - positive_score + harmful_score
    )
    continuation_loss = winning_detail.continuation_loss
    raw_loss = branch_loss + continuation_loss
    duplicate_alias_count = len(positive_scores) - len(owner_scores)
    return EntityTransitionPreferenceResult(
        raw_loss=raw_loss,
        target_margin=target_margin,
        positive_score=positive_score,
        harmful_score=harmful_score,
        positive_paths=tuple(positive_scores),
        harmful_path=scored_harmful,
        owner_scores=owner_scores,
        positive_path_count=len(positive_scores),
        distinct_owner_count=len(owner_scores),
        duplicate_alias_count=duplicate_alias_count,
        first_divergence_offsets=tuple(
            (detail.path.candidate_id, detail.first_divergence_offset)
            for detail in positive_details
        ),
        positive_continuation_loss=continuation_loss,
        positive_schema_description_loss=winning_detail.schema_description_loss,
        positive_coordinate_loss=winning_detail.coordinate_loss,
        finite=_finite_diagnostics(
            raw_loss=raw_loss,
            target_margin=target_margin,
            positive_score=positive_score,
            harmful_score=harmful_score,
            branch_loss=branch_loss,
            continuation_loss=continuation_loss,
            schema_description_loss=winning_detail.schema_description_loss,
            coordinate_loss=winning_detail.coordinate_loss,
        ),
    )


def first_wrong_coordinate_preference(
    logits: torch.Tensor,
    *,
    acceptable_token_ids: tuple[int, ...],
    wrong_token_id: int,
    margin: float,
) -> CoordinateBoundaryPreferenceResult:
    """Prefer a reviewed discrete acceptable set to the actual wrong token."""

    row = _checked_logits_row(logits, context="coordinate_boundary")
    checked_margin = _finite_scalar(margin, name="margin")
    accepted = _checked_token_ids(
        acceptable_token_ids,
        vocab_size=int(row.shape[0]),
        name="acceptable_token_ids",
        require_nonempty=True,
        require_unique=True,
    )
    wrong = _checked_token_id(
        wrong_token_id,
        vocab_size=int(row.shape[0]),
        name="wrong_token_id",
    )
    if wrong in accepted:
        raise LossContractError(
            "the actual wrong coordinate token must be outside the acceptable set",
            code="loss.rollout_coordinate_wrong_is_acceptable",
            context={"wrong_token_id": wrong},
        )

    acceptable_index = torch.tensor(accepted, dtype=torch.long, device=row.device)
    acceptable_mass_score = torch.logsumexp(
        row.index_select(0, acceptable_index),
        dim=0,
    )
    wrong_token_score = row[wrong]
    target_margin = acceptable_mass_score - wrong_token_score
    raw_loss = F.softplus(
        row.new_tensor(checked_margin) - acceptable_mass_score + wrong_token_score
    )
    return CoordinateBoundaryPreferenceResult(
        raw_loss=raw_loss,
        target_margin=target_margin,
        acceptable_mass_score=acceptable_mass_score,
        wrong_token_score=wrong_token_score,
        acceptable_token_count=len(accepted),
        finite=_finite_diagnostics(
            raw_loss=raw_loss,
            target_margin=target_margin,
            acceptable_mass_score=acceptable_mass_score,
            wrong_token_score=wrong_token_score,
        ),
    )


def positive_path_imitation_loss(
    path: CandidatePath,
) -> PositivePathImitationResult:
    """Teach one sampled complete row from its exact generated token ids.

    The row is scored from offset zero.  Schema and description tokens form
    one mean-normalized group, while trusted coordinate tokens form a second
    group.  Coordinates converted to ``untrusted_coordinate`` by the existing
    training seam are intentionally omitted and therefore receive no exact
    token gradient.  No branch, terminal, or KL term is involved.
    """

    _validate_candidate_identity(path, role="positive")
    if path.token_types is None:
        raise LossContractError(
            "positive-path imitation requires intended token types",
            code="loss.rollout_positive_path_token_types_missing",
            context={"candidate_id": path.candidate_id},
        )
    logits = _checked_path_logits(path.logits, candidate_id=path.candidate_id)
    target_ids = _checked_token_ids(
        path.target_token_ids,
        vocab_size=int(logits.shape[1]),
        name="target_token_ids",
        require_nonempty=True,
        require_unique=False,
    )
    if len(path.token_types) != len(target_ids):
        raise LossContractError(
            "positive-path imitation token types must cover the complete row",
            code="loss.rollout_positive_path_token_type_length",
            context={
                "candidate_id": path.candidate_id,
                "token_count": len(target_ids),
                "token_type_count": len(path.token_types),
            },
        )
    allowed_types = {"desc_text", "schema", "coordinate", "untrusted_coordinate"}
    unknown_types = sorted(set(path.token_types) - allowed_types)
    if unknown_types:
        raise LossContractError(
            "positive-path imitation contains an unsupported token type",
            code="loss.rollout_positive_path_token_type_unknown",
            context={
                "candidate_id": path.candidate_id,
                "unknown_types": unknown_types,
            },
        )
    selected = -torch.log_softmax(logits, dim=-1).gather(
        1,
        torch.tensor(target_ids, dtype=torch.long, device=logits.device).unsqueeze(1),
    ).squeeze(1)
    schema_indexes = tuple(
        index
        for index, token_type in enumerate(path.token_types)
        if token_type in {"schema", "desc_text"}
    )
    coordinate_indexes = tuple(
        index
        for index, token_type in enumerate(path.token_types)
        if token_type == "coordinate"
    )
    if not schema_indexes and not coordinate_indexes:
        raise LossContractError(
            "positive-path imitation row has no trusted supervised token group",
            code="loss.rollout_positive_path_groups_empty",
            context={"candidate_id": path.candidate_id},
        )
    zero = selected.sum() * 0.0
    schema_loss = (
        selected[list(schema_indexes)].mean() if schema_indexes else zero
    )
    coordinate_loss = (
        selected[list(coordinate_indexes)].mean() if coordinate_indexes else zero
    )
    groups = [
        value
        for value, indexes in (
            (schema_loss, schema_indexes),
            (coordinate_loss, coordinate_indexes),
        )
        if indexes
    ]
    raw_loss = torch.stack(groups).mean()
    return PositivePathImitationResult(
        raw_loss=raw_loss,
        schema_description_loss=schema_loss,
        coordinate_loss=coordinate_loss,
        selected_token_count=len(schema_indexes) + len(coordinate_indexes),
        schema_description_token_count=len(schema_indexes),
        coordinate_token_count=len(coordinate_indexes),
        finite=_finite_diagnostics(
            raw_loss=raw_loss,
            schema_description_loss=schema_loss,
            coordinate_loss=coordinate_loss,
        ),
    )


# Keep a compact function alias for callers that use the objective name as a
# verb, while the explicit ``*_loss`` spelling remains the canonical API.
positive_path_imitation = positive_path_imitation_loss


def rollout_site_token_type_gate(
    declarations: tuple[RolloutGateSite, ...],
) -> RolloutSiteTokenTypeGateResult:
    """Apply an event-balanced intended-type gate after same-site deduplication."""

    if not declarations:
        raise LossContractError(
            "rollout-site token-type gate requires at least one site",
            code="loss.rollout_gate_empty",
        )

    unique_by_identity: dict[GateSiteIdentity, RolloutGateSite] = {}
    for declaration in declarations:
        _validate_gate_site(declaration)
        previous = unique_by_identity.get(declaration.identity)
        if previous is None:
            unique_by_identity[declaration.identity] = declaration
            continue
        if previous.event_id != declaration.event_id:
            raise LossContractError(
                "one rollout gate site cannot belong to conflicting events",
                code="loss.rollout_gate_event_conflict",
                context={
                    "segment_id": declaration.identity.segment_id,
                    "logits_position": declaration.identity.logits_position,
                    "first_event_id": previous.event_id,
                    "second_event_id": declaration.event_id,
                },
            )
        if previous.intended_token_type != declaration.intended_token_type:
            raise LossContractError(
                "one rollout gate site declares conflicting intended token types",
                code="loss.rollout_gate_type_conflict",
                context={
                    "event_id": declaration.event_id,
                    "segment_id": declaration.identity.segment_id,
                    "logits_position": declaration.identity.logits_position,
                    "first_type": previous.intended_token_type,
                    "second_type": declaration.intended_token_type,
                },
            )
        if tuple(sorted(previous.allowed_token_ids)) != tuple(
            sorted(declaration.allowed_token_ids)
        ):
            raise LossContractError(
                "one rollout gate site resolves different allowed token groups",
                code="loss.rollout_gate_group_conflict",
                context={
                    "event_id": declaration.event_id,
                    "segment_id": declaration.identity.segment_id,
                    "logits_position": declaration.identity.logits_position,
                },
            )

    site_values_by_event: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for site in unique_by_identity.values():
        row = _checked_logits_row(site.logits, context="rollout_token_type_gate")
        allowed = _checked_token_ids(
            site.allowed_token_ids,
            vocab_size=int(row.shape[0]),
            name="allowed_token_ids",
            require_nonempty=True,
            require_unique=True,
        )
        allowed_index = torch.tensor(allowed, dtype=torch.long, device=row.device)
        all_mass = torch.logsumexp(row, dim=0)
        allowed_mass = torch.logsumexp(row.index_select(0, allowed_index), dim=0)
        site_loss = all_mass - allowed_mass
        legal_mass = torch.exp(allowed_mass - all_mass)
        site_values_by_event.setdefault(site.event_id, []).append(
            (site_loss, legal_mass)
        )

    events: list[EventGateResult] = []
    for event_id, site_values in sorted(site_values_by_event.items()):
        site_losses = torch.stack(tuple(item[0] for item in site_values))
        legal_masses = torch.stack(tuple(item[1] for item in site_values))
        events.append(
            EventGateResult(
                event_id=event_id,
                raw_loss=site_losses.mean(),
                legal_mass=legal_masses.mean(),
                distinct_site_count=len(site_values),
            )
        )
    raw_loss = torch.stack(tuple(event.raw_loss for event in events)).mean()
    legal_mass = torch.stack(tuple(event.legal_mass for event in events)).mean()
    return RolloutSiteTokenTypeGateResult(
        raw_loss=raw_loss,
        legal_mass=legal_mass,
        events=tuple(events),
        eligible_event_count=len(events),
        distinct_site_count=len(unique_by_identity),
        declaration_count=len(declarations),
        duplicate_declaration_count=len(declarations) - len(unique_by_identity),
        finite=_finite_diagnostics(raw_loss=raw_loss, legal_mass=legal_mass),
    )


def _score_candidate_path(path: CandidatePath) -> CandidatePathScore:
    logits = _checked_path_logits(path.logits, candidate_id=path.candidate_id)
    target_ids = _checked_token_ids(
        path.target_token_ids,
        vocab_size=int(logits.shape[1]),
        name="target_token_ids",
        require_nonempty=True,
        require_unique=False,
    )
    if len(target_ids) != int(logits.shape[0]):
        raise LossContractError(
            "candidate path target count must match its causal logits rows",
            code="loss.rollout_candidate_path_length",
            context={
                "candidate_id": path.candidate_id,
                "logits_row_count": int(logits.shape[0]),
                "target_count": len(target_ids),
            },
        )
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
    selected = (
        torch.log_softmax(logits, dim=-1)
        .gather(
            1,
            target_tensor.unsqueeze(1),
        )
        .squeeze(1)
    )
    return CandidatePathScore(
        candidate_id=path.candidate_id,
        physical_owner_id=path.physical_owner_id,
        summed_log_probability=selected.sum(),
        mean_log_probability=selected.mean(),
        token_count=len(target_ids),
    )


def _transition_path_detail(
    path: CandidatePath,
    scored: CandidatePathScore,
    *,
    harmful_path: CandidatePath,
) -> _TransitionPathDetail:
    """Score one positive branch against the actual harmful branch.

    ``path`` and ``harmful_path`` are isolated replay segments, so their
    logits tensors have independent rows.  Equal target token identifiers
    before ``divergence`` prove that both rows represent the same causal
    history; only the two logits at that first mismatch are compared.
    """

    positive_ids = tuple(path.target_token_ids)
    harmful_ids = tuple(harmful_path.target_token_ids)
    divergence = _first_divergence_offset(positive_ids, harmful_ids)
    positive_logits = _checked_path_logits(path.logits, candidate_id=path.candidate_id)
    harmful_logits = _checked_path_logits(
        harmful_path.logits,
        candidate_id=harmful_path.candidate_id,
    )
    if divergence >= int(positive_logits.shape[0]) or divergence >= len(positive_ids):
        raise LossContractError(
            "positive branch has no token at its first divergent position",
            code="loss.rollout_entity_positive_divergence_missing",
            context={
                "candidate_id": path.candidate_id,
                "harmful_candidate_id": harmful_path.candidate_id,
                "divergence": divergence,
            },
        )
    if divergence >= int(harmful_logits.shape[0]) or divergence >= len(harmful_ids):
        raise LossContractError(
            "harmful branch has no token at the first divergent position",
            code="loss.rollout_entity_harmful_divergence_missing",
            context={
                "candidate_id": path.candidate_id,
                "harmful_candidate_id": harmful_path.candidate_id,
                "divergence": divergence,
            },
        )

    positive_branch = torch.log_softmax(positive_logits[divergence], dim=-1)[
        positive_ids[divergence]
    ]
    harmful_branch = torch.log_softmax(harmful_logits[divergence], dim=-1)[
        harmful_ids[divergence]
    ]
    continuation, schema_description, coordinates = _positive_continuation_loss(
        path,
        divergence,
    )
    return _TransitionPathDetail(
        path=path,
        scored=scored,
        first_divergence_offset=divergence,
        positive_branch_log_probability=positive_branch,
        harmful_branch_log_probability=harmful_branch,
        branch_margin=positive_branch - harmful_branch,
        continuation_loss=continuation,
        schema_description_loss=schema_description,
        coordinate_loss=coordinates,
    )


def _first_divergence_offset(
    positive_token_ids: tuple[int, ...],
    harmful_token_ids: tuple[int, ...],
) -> int:
    limit = min(len(positive_token_ids), len(harmful_token_ids))
    for offset in range(limit):
        if positive_token_ids[offset] != harmful_token_ids[offset]:
            return offset
    # A path that is an exact prefix of the other has no pair of tokens at a
    # shared boundary.  Such an event cannot provide the requested branch
    # comparison and must be rejected instead of silently comparing horizons.
    raise LossContractError(
        "positive and harmful branches do not have a first divergent token",
        code="loss.rollout_entity_no_divergence",
        context={
            "positive_length": len(positive_token_ids),
            "harmful_length": len(harmful_token_ids),
        },
    )


def _positive_continuation_loss(
    path: CandidatePath,
    divergence: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined, entity/schema, and coordinate continuation losses."""

    logits = _checked_path_logits(path.logits, candidate_id=path.candidate_id)
    target_ids = tuple(path.target_token_ids)
    if path.token_types is None:
        zero = logits.sum() * 0.0
        return zero, zero, zero
    if len(path.token_types) != len(target_ids):
        raise LossContractError(
            "positive continuation token types must cover the scored row",
            code="loss.rollout_continuation_token_type_length",
            context={
                "candidate_id": path.candidate_id,
                "token_count": len(target_ids),
                "token_type_count": len(path.token_types),
            },
        )
    selected = (
        -torch.log_softmax(logits[divergence:], dim=-1)
        .gather(
            1,
            torch.tensor(
                target_ids[divergence:],
                dtype=torch.long,
                device=logits.device,
            ).unsqueeze(1),
        )
        .squeeze(1)
    )
    entity_indexes = tuple(
        index
        for index, token_type in enumerate(path.token_types[divergence:])
        if token_type in {"schema", "desc_text"}
    )
    coordinate_indexes = tuple(
        index
        for index, token_type in enumerate(path.token_types[divergence:])
        if token_type == "coordinate"
    )
    group_losses: list[torch.Tensor] = []
    if entity_indexes:
        schema_description = selected[list(entity_indexes)].mean()
        group_losses.append(schema_description)
    else:
        schema_description = selected.sum() * 0.0
    if coordinate_indexes:
        coordinates = selected[list(coordinate_indexes)].mean()
        group_losses.append(coordinates)
    else:
        coordinates = selected.sum() * 0.0
    if not group_losses:
        continuation = selected.sum() * 0.0
    else:
        continuation = torch.stack(group_losses).mean()
    return continuation, schema_description, coordinates


def _validate_candidate_identity(
    path: CandidatePath,
    *,
    role: Literal["positive", "harmful"],
) -> None:
    if not path.candidate_id:
        raise LossContractError(
            "candidate path requires a non-empty identity",
            code="loss.rollout_candidate_id",
            context={"role": role},
        )
    if role == "positive":
        if path.physical_owner_id is None or not path.physical_owner_id:
            raise LossContractError(
                "positive entity-transition paths require a physical owner",
                code="loss.rollout_entity_positive_owner",
                context={"candidate_id": path.candidate_id},
            )
        if path.premature_terminal:
            raise LossContractError(
                "positive entity-transition paths cannot be premature terminal",
                code="loss.rollout_entity_positive_terminal",
                context={"candidate_id": path.candidate_id},
            )
        return
    if path.premature_terminal:
        if path.physical_owner_id is not None:
            raise LossContractError(
                "premature-terminal harmful paths must have null physical owner",
                code="loss.rollout_terminal_owner",
                context={"candidate_id": path.candidate_id},
            )
        if len(path.target_token_ids) != 1:
            raise LossContractError(
                "premature-terminal harmful paths must score exactly one token",
                code="loss.rollout_terminal_path_length",
                context={
                    "candidate_id": path.candidate_id,
                    "token_count": len(path.target_token_ids),
                },
            )
    elif path.physical_owner_id is None or not path.physical_owner_id:
        raise LossContractError(
            "nonterminal harmful paths require a duplicate physical owner",
            code="loss.rollout_harmful_owner",
            context={"candidate_id": path.candidate_id},
        )


def _validate_gate_site(site: RolloutGateSite) -> None:
    if not site.event_id or not site.identity.segment_id:
        raise LossContractError(
            "rollout gate event and segment identities must be non-empty",
            code="loss.rollout_gate_identity",
            context={
                "event_id": site.event_id,
                "segment_id": site.identity.segment_id,
            },
        )
    if site.identity.logits_position < 0:
        raise LossContractError(
            "rollout gate logits position must be non-negative",
            code="loss.rollout_gate_position",
            context={"logits_position": site.identity.logits_position},
        )
    if site.intended_token_type not in V1_TOKEN_TYPES:
        raise LossContractError(
            "rollout gate intended token type must be a closed V1 type",
            code="loss.rollout_gate_type_unknown",
            context={
                "intended_token_type": site.intended_token_type,
                "known_token_types": list(V1_TOKEN_TYPES),
            },
        )


def _checked_path_logits(logits: torch.Tensor, *, candidate_id: str) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise LossContractError(
            "candidate path logits must have shape [path_length, vocab_size]",
            code="loss.rollout_candidate_logits_shape",
            context={
                "candidate_id": candidate_id,
                "shape": (
                    None
                    if not isinstance(logits, torch.Tensor)
                    else [int(value) for value in logits.shape]
                ),
            },
        )
    if int(logits.shape[0]) <= 0 or int(logits.shape[1]) <= 0:
        raise LossContractError(
            "candidate path logits dimensions must be non-empty",
            code="loss.rollout_candidate_logits_empty",
            context={
                "candidate_id": candidate_id,
                "shape": [int(value) for value in logits.shape],
            },
        )
    if not torch.is_floating_point(logits):
        raise LossContractError(
            "candidate path logits must be floating point",
            code="loss.rollout_logits_dtype",
            context={"candidate_id": candidate_id, "dtype": str(logits.dtype)},
        )
    return logits.float()


def _checked_logits_row(logits: torch.Tensor, *, context: str) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise LossContractError(
            "rollout calibration site logits must have shape [vocab_size]",
            code="loss.rollout_site_logits_shape",
            context={
                "site_context": context,
                "shape": (
                    None
                    if not isinstance(logits, torch.Tensor)
                    else [int(value) for value in logits.shape]
                ),
            },
        )
    if int(logits.shape[0]) <= 0 or not torch.is_floating_point(logits):
        raise LossContractError(
            "rollout calibration site logits must be non-empty floating point",
            code="loss.rollout_logits_dtype",
            context={"site_context": context, "dtype": str(logits.dtype)},
        )
    return logits.float()


def _checked_token_ids(
    token_ids: tuple[int, ...],
    *,
    vocab_size: int,
    name: str,
    require_nonempty: bool,
    require_unique: bool,
) -> tuple[int, ...]:
    if require_nonempty and not token_ids:
        raise LossContractError(
            f"{name} must not be empty",
            code="loss.rollout_token_ids_empty",
            context={"field": name},
        )
    checked = tuple(
        _checked_token_id(token_id, vocab_size=vocab_size, name=name)
        for token_id in token_ids
    )
    if require_unique and len(set(checked)) != len(checked):
        raise LossContractError(
            f"{name} must contain unique token ids",
            code="loss.rollout_token_ids_duplicate",
            context={"field": name, "token_ids": list(checked)},
        )
    return checked


def _checked_token_id(token_id: int, *, vocab_size: int, name: str) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise LossContractError(
            f"{name} must contain integer token ids",
            code="loss.rollout_token_id_type",
            context={"field": name, "token_id": token_id},
        )
    if token_id < 0 or token_id >= vocab_size:
        raise LossContractError(
            f"{name} must stay inside the resolved vocabulary",
            code="loss.rollout_token_id_bounds",
            context={
                "field": name,
                "token_id": token_id,
                "vocab_size": vocab_size,
            },
        )
    return token_id


def _finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LossContractError(
            f"{name} must be numeric",
            code="loss.rollout_scalar_type",
            context={"field": name, "value_type": type(value).__name__},
        )
    checked = float(value)
    if not math.isfinite(checked):
        raise LossContractError(
            f"{name} must be finite",
            code="loss.rollout_scalar_non_finite",
            context={"field": name, "value": checked},
        )
    return checked


def _positive_finite_scalar(value: float, *, name: str) -> float:
    checked = _finite_scalar(value, name=name)
    if checked <= 0.0:
        raise LossContractError(
            f"{name} must be positive",
            code="loss.rollout_scalar_non_positive",
            context={"field": name, "value": checked},
        )
    return checked


def _finite_diagnostics(**values: torch.Tensor) -> LossFiniteDiagnostics:
    statuses = tuple(
        (name, bool(torch.isfinite(value.detach()).all().item()))
        for name, value in values.items()
    )
    return LossFiniteDiagnostics(
        all_finite=all(finite for _name, finite in statuses),
        values=statuses,
    )


__all__ = [
    "CandidatePath",
    "CandidatePathScore",
    "CoordinateBoundaryPreferenceResult",
    "EntityTransitionPreferenceResult",
    "EventGateResult",
    "GateSiteIdentity",
    "LossFiniteDiagnostics",
    "OwnerScore",
    "RolloutGateSite",
    "RolloutSiteTokenTypeGateResult",
    "first_wrong_coordinate_preference",
    "grouped_entity_transition_preference",
    "rollout_site_token_type_gate",
]
