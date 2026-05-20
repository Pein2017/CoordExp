from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from src.training.teacher_forcing.roles import TokenRole

CoordRole = Literal["x1", "y1", "x2", "y2"]
CorrectionKind = Literal[
    "transition_failure",
    "premature_stop",
    "fp_boundary",
    "repeated_object_boundary",
    "matched_object_repair",
]


@dataclass(frozen=True)
class ResidualObject:
    object_id: str
    desc_token_ids: tuple[int, ...]
    coord_token_ids: Mapping[CoordRole, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "desc_token_ids", tuple(self.desc_token_ids))


@dataclass(frozen=True)
class ValidAction:
    token_id: int
    token_role: TokenRole
    token_text: str
    candidate_ids_after: frozenset[str]
    selected_object_id: str | None = None
    coord_role: CoordRole | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids_after", frozenset(self.candidate_ids_after))


@dataclass(frozen=True)
class ResidualState:
    objects: tuple[ResidualObject, ...]
    remaining_object_ids: frozenset[str]
    active_candidate_ids: frozenset[str]
    text_prefix_token_ids: tuple[int, ...] = ()
    stop_token_id: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "remaining_object_ids", frozenset(self.remaining_object_ids))
        object.__setattr__(self, "active_candidate_ids", frozenset(self.active_candidate_ids))
        object.__setattr__(self, "text_prefix_token_ids", tuple(self.text_prefix_token_ids))

    def valid_actions_at_boundary(self) -> tuple[ValidAction, ...]:
        return enumerate_valid_actions(self, slot="boundary")


@dataclass(frozen=True)
class CorrectionAtomDraft:
    correction_kind: CorrectionKind
    target_position: int
    logit_position: int
    valid_actions: tuple[ValidAction, ...]
    selected_action: ValidAction | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_actions", tuple(self.valid_actions))


@dataclass(frozen=True)
class CorrectionEvent:
    correction_kind: CorrectionKind
    sample_id: str
    atom_drafts: tuple[CorrectionAtomDraft, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_drafts", tuple(self.atom_drafts))


def enumerate_valid_actions(state: ResidualState, *, slot: str | CoordRole = "boundary") -> tuple[ValidAction, ...]:
    if slot == "boundary":
        return _enumerate_boundary_actions(state)
    if slot in ("x1", "y1", "x2", "y2"):
        return _enumerate_coord_actions(state, slot)
    raise ValueError(f"unsupported residual-set slot: {slot!r}")


def transition_state(state: ResidualState, action: ValidAction) -> ResidualState:
    if action.token_role is TokenRole.STOP:
        return ResidualState(
            objects=state.objects,
            remaining_object_ids=state.remaining_object_ids,
            active_candidate_ids=frozenset(),
            text_prefix_token_ids=state.text_prefix_token_ids,
            stop_token_id=state.stop_token_id,
            metadata=state.metadata,
        )

    active_after = state.active_candidate_ids & action.candidate_ids_after
    if not active_after:
        raise ValueError("residual-set transition produced empty active candidates for non-STOP action")

    text_prefix_token_ids = state.text_prefix_token_ids
    if action.token_role is TokenRole.TEXT:
        text_prefix_token_ids = (*text_prefix_token_ids, action.token_id)

    return ResidualState(
        objects=state.objects,
        remaining_object_ids=state.remaining_object_ids,
        active_candidate_ids=active_after,
        text_prefix_token_ids=text_prefix_token_ids,
        stop_token_id=state.stop_token_id,
        metadata=state.metadata,
    )


def _enumerate_boundary_actions(state: ResidualState) -> tuple[ValidAction, ...]:
    if not state.remaining_object_ids:
        return (
            ValidAction(
                token_id=state.stop_token_id,
                token_role=TokenRole.STOP,
                token_text="<|im_end|>",
                candidate_ids_after=frozenset(),
            ),
        )

    cursor = len(state.text_prefix_token_ids)
    objects = _active_remaining_objects(state)
    candidate_ids_by_token: dict[int, set[str]] = {}
    token_text_by_token: dict[int, str] = {}
    for obj in objects:
        if cursor >= len(obj.desc_token_ids):
            continue
        token_id = obj.desc_token_ids[cursor]
        candidate_ids_by_token.setdefault(token_id, set()).add(obj.object_id)
        token_text_by_token.setdefault(token_id, _derive_text_token_text(obj.desc_token_ids, cursor))

    return _coalesced_actions(
        candidate_ids_by_token,
        token_role=TokenRole.TEXT,
        token_text_by_token=token_text_by_token,
        coord_role=None,
    )


def _enumerate_coord_actions(state: ResidualState, coord_role: CoordRole) -> tuple[ValidAction, ...]:
    candidate_ids_by_token: dict[int, set[str]] = {}
    for obj in _active_remaining_objects(state):
        token_id = obj.coord_token_ids.get(coord_role)
        if token_id is None:
            continue
        candidate_ids_by_token.setdefault(token_id, set()).add(obj.object_id)

    token_text_by_token = {token_id: str(token_id) for token_id in candidate_ids_by_token}
    return _coalesced_actions(
        candidate_ids_by_token,
        token_role=TokenRole.COORD,
        token_text_by_token=token_text_by_token,
        coord_role=coord_role,
    )


def _active_remaining_objects(state: ResidualState) -> tuple[ResidualObject, ...]:
    active_remaining_ids = state.remaining_object_ids & state.active_candidate_ids
    return tuple(obj for obj in state.objects if obj.object_id in active_remaining_ids)


def _coalesced_actions(
    candidate_ids_by_token: Mapping[int, set[str]],
    *,
    token_role: TokenRole,
    token_text_by_token: Mapping[int, str],
    coord_role: CoordRole | None,
) -> tuple[ValidAction, ...]:
    actions: list[ValidAction] = []
    for token_id in sorted(candidate_ids_by_token):
        candidate_ids_after = frozenset(sorted(candidate_ids_by_token[token_id]))
        selected_object_id = next(iter(candidate_ids_after)) if len(candidate_ids_after) == 1 else None
        actions.append(
            ValidAction(
                token_id=token_id,
                token_role=token_role,
                token_text=token_text_by_token[token_id],
                candidate_ids_after=candidate_ids_after,
                selected_object_id=selected_object_id,
                coord_role=coord_role,
            )
        )
    return tuple(actions)


def _derive_text_token_text(desc_token_ids: tuple[int, ...], cursor: int) -> str:
    if cursor >= len(desc_token_ids):
        return ""
    token_id = desc_token_ids[cursor]
    try:
        suffix = "".join(chr(token) for token in desc_token_ids[cursor:])
    except (TypeError, ValueError):
        return str(token_id)

    if not suffix:
        return str(token_id)
    if not suffix[0].isalnum():
        return suffix[0]

    chars: list[str] = []
    for ch in suffix:
        if not ch.isalnum():
            break
        chars.append(ch)
    return "".join(chars) or suffix[0]


__all__ = [
    "CoordRole",
    "CorrectionKind",
    "ResidualObject",
    "ValidAction",
    "ResidualState",
    "CorrectionAtomDraft",
    "CorrectionEvent",
    "enumerate_valid_actions",
    "transition_state",
]
