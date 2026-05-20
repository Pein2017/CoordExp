from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
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
_COORD_ROLES: tuple[CoordRole, ...] = ("x1", "y1", "x2", "y2")
_STOP_TOKEN_TEXT = "<|im_end|>"


def _immutable_mapping(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class ResidualObject:
    object_id: str
    desc_token_ids: tuple[int, ...]
    coord_token_ids: Mapping[CoordRole, int]
    desc_token_texts: tuple[str | None, ...] | None = None
    loss_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("ResidualObject.object_id must be nonempty")

        desc_token_ids = tuple(self.desc_token_ids)
        if not desc_token_ids:
            raise ValueError("ResidualObject.desc_token_ids must be nonempty")
        object.__setattr__(self, "desc_token_ids", desc_token_ids)

        coord_token_ids = dict(self.coord_token_ids)
        if set(coord_token_ids) != set(_COORD_ROLES):
            raise ValueError("ResidualObject.coord_token_ids must contain exactly x1/y1/x2/y2")
        object.__setattr__(self, "coord_token_ids", _immutable_mapping(coord_token_ids))

        if self.desc_token_texts is not None:
            desc_token_texts = tuple(self.desc_token_texts)
            if len(desc_token_texts) != len(desc_token_ids):
                raise ValueError("ResidualObject.desc_token_texts must match desc_token_ids length")
            object.__setattr__(self, "desc_token_texts", desc_token_texts)

        if isinstance(self.loss_weight, bool) or not isinstance(self.loss_weight, (int, float)):
            raise ValueError("ResidualObject.loss_weight must be a finite nonnegative real number")
        loss_weight = float(self.loss_weight)
        if not math.isfinite(loss_weight) or loss_weight < 0.0:
            raise ValueError("ResidualObject.loss_weight must be a finite nonnegative real number")
        object.__setattr__(self, "loss_weight", loss_weight)
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


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
        candidate_ids_after = frozenset(self.candidate_ids_after)
        object.__setattr__(self, "candidate_ids_after", candidate_ids_after)

        if self.selected_object_id is not None and self.selected_object_id not in candidate_ids_after:
            raise ValueError("ValidAction.selected_object_id must be in candidate_ids_after")
        if self.token_role is TokenRole.COORD and self.coord_role is None:
            raise ValueError("TokenRole.COORD requires coord_role")
        if self.token_role is TokenRole.TEXT and self.coord_role is not None:
            raise ValueError("TokenRole.TEXT requires coord_role is None")
        if self.token_role is TokenRole.STOP:
            if self.token_text != _STOP_TOKEN_TEXT:
                raise ValueError("TokenRole.STOP requires canonical token_text")
            if candidate_ids_after:
                raise ValueError("TokenRole.STOP requires empty candidate_ids_after")
            if self.selected_object_id is not None:
                raise ValueError("TokenRole.STOP requires selected_object_id is None")
            if self.coord_role is not None:
                raise ValueError("TokenRole.STOP requires coord_role is None")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ResidualState:
    objects: tuple[ResidualObject, ...]
    remaining_object_ids: frozenset[str]
    active_candidate_ids: frozenset[str]
    text_prefix_token_ids: tuple[int, ...] = ()
    stop_token_id: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    objects_by_id: Mapping[str, ResidualObject] = field(init=False)

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        object.__setattr__(self, "objects", objects)
        objects_by_id = {obj.object_id: obj for obj in objects}
        if len(objects_by_id) != len(objects):
            raise ValueError("ResidualState.objects_by_id keys must match each object's object_id")
        object.__setattr__(self, "objects_by_id", _immutable_mapping(objects_by_id))

        known_object_ids = frozenset(objects_by_id)
        remaining_object_ids = frozenset(self.remaining_object_ids)
        if not remaining_object_ids.issubset(known_object_ids):
            raise ValueError("ResidualState.remaining_object_ids must be a subset of known object ids")
        object.__setattr__(self, "remaining_object_ids", remaining_object_ids)

        active_candidate_ids = frozenset(self.active_candidate_ids)
        if not active_candidate_ids.issubset(remaining_object_ids):
            raise ValueError("ResidualState.active_candidate_ids must be a subset of remaining_object_ids")
        if remaining_object_ids and not active_candidate_ids:
            raise ValueError("ResidualState.active_candidate_ids must be nonempty when objects remain")
        if not remaining_object_ids and active_candidate_ids:
            raise ValueError("ResidualState.active_candidate_ids must be empty when no objects remain")
        object.__setattr__(self, "active_candidate_ids", active_candidate_ids)
        object.__setattr__(self, "text_prefix_token_ids", tuple(self.text_prefix_token_ids))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

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
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class CorrectionEvent:
    correction_kind: CorrectionKind
    sample_id: str
    atom_drafts: tuple[CorrectionAtomDraft, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_drafts", tuple(self.atom_drafts))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


def enumerate_valid_actions(state: ResidualState, *, slot: str | CoordRole = "boundary") -> tuple[ValidAction, ...]:
    if slot == "boundary":
        return _enumerate_boundary_actions(state)
    if slot in ("x1", "y1", "x2", "y2"):
        return _enumerate_coord_actions(state, slot)
    raise ValueError(f"unsupported residual-set slot: {slot!r}")


def transition_state(state: ResidualState, action: ValidAction) -> ResidualState:
    if action.token_role is TokenRole.STOP:
        if action.token_id != state.stop_token_id:
            raise ValueError("STOP transition token_id must match state.stop_token_id")
        if state.remaining_object_ids:
            raise ValueError("STOP transition requires an empty residual state")
        if action.candidate_ids_after:
            raise ValueError("STOP transition requires empty candidate_ids_after")
        if action.selected_object_id is not None:
            raise ValueError("STOP transition requires selected_object_id is None")
        if action.coord_role is not None:
            raise ValueError("STOP transition requires coord_role is None")
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
                token_text=_STOP_TOKEN_TEXT,
                candidate_ids_after=frozenset(),
            ),
        )

    cursor = len(state.text_prefix_token_ids)
    objects = _active_remaining_objects(state)
    candidate_ids_by_token: dict[int, set[str]] = {}
    token_text_by_token: dict[int, str] = {}
    token_text_supplied_by_token: dict[int, bool] = {}
    for obj in objects:
        if cursor >= len(obj.desc_token_ids):
            continue
        token_id = obj.desc_token_ids[cursor]
        candidate_ids_by_token.setdefault(token_id, set()).add(obj.object_id)
        token_text, supplied = _derive_text_token_text(obj, cursor)
        if token_id not in token_text_by_token or (supplied and not token_text_supplied_by_token[token_id]):
            token_text_by_token[token_id] = token_text
            token_text_supplied_by_token[token_id] = supplied

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


def _derive_text_token_text(obj: ResidualObject, cursor: int) -> tuple[str, bool]:
    if obj.desc_token_texts is not None:
        token_text = obj.desc_token_texts[cursor]
        if token_text is not None:
            return token_text, True
    return f"token_id={obj.desc_token_ids[cursor]}", False


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
