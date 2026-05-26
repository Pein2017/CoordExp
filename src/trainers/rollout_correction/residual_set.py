from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

from src.training.teacher_forcing.roles import TokenRole

CoordRole = Literal["x1", "y1", "x2", "y2"]
SchemaSlot = Literal["object_start", "box_start"]
CorrectionKind = Literal[
    "transition_failure",
    "premature_stop",
    "fp_boundary",
    "repeated_object_boundary",
    "matched_object_repair",
    "spatial_wrong_desc_conflict",
    "invalid_geometry",
    "malformed_span",
    "trailing_incomplete",
    "duplicate_burst",
]
RowScanDecisionKind = Literal[
    "committed",
    "duplicate_burst",
    "invalid_geometry",
    "malformed_span",
    "spatial_wrong_desc_conflict",
    "trailing_incomplete",
    "unmatched_dirty_context",
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
    next_state: ResidualState | None = None
    selected_object_id: str | None = None
    coord_role: CoordRole | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_ids_after = frozenset(self.candidate_ids_after)
        object.__setattr__(self, "candidate_ids_after", candidate_ids_after)

        if self.next_state is not None and not isinstance(self.next_state, ResidualState):
            raise ValueError("ValidAction.next_state must be a ResidualState")
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
            if self.next_state is not None and self.next_state.remaining_object_ids:
                raise ValueError("TokenRole.STOP next_state requires empty remaining_object_ids")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ResidualState:
    objects: tuple[ResidualObject, ...]
    remaining_object_ids: frozenset[str]
    active_candidate_ids: frozenset[str]
    text_prefix_token_ids: tuple[int, ...] = ()
    object_start_token_id: int | None = None
    box_start_token_id: int | None = None
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


@dataclass(frozen=True)
class ObservedResidualRow:
    desc: str | None
    bbox_norm1000: tuple[float, float, float, float] | None
    object_start: int
    object_end: int | None
    reliable_span: bool = True
    reliable_resync: bool = True
    malformed: bool = False
    incomplete: bool = False
    desc_token_ids: tuple[int, ...] | None = None
    desc_token_positions: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bbox_norm1000 is not None:
            if len(tuple(self.bbox_norm1000)) != 4:
                raise ValueError("ObservedResidualRow.bbox_norm1000 must be a 4-tuple")
            object.__setattr__(
                self,
                "bbox_norm1000",
                tuple(float(value) for value in self.bbox_norm1000),
            )
        object.__setattr__(self, "object_start", int(self.object_start))
        object.__setattr__(
            self,
            "object_end",
            None if self.object_end is None else int(self.object_end),
        )
        if self.desc_token_ids is not None:
            object.__setattr__(self, "desc_token_ids", tuple(int(value) for value in self.desc_token_ids))
        if self.desc_token_positions is not None:
            positions = tuple(int(value) for value in self.desc_token_positions)
            if self.desc_token_ids is not None and len(positions) != len(self.desc_token_ids):
                raise ValueError(
                    "ObservedResidualRow.desc_token_positions must match desc_token_ids length"
                )
            object.__setattr__(self, "desc_token_positions", positions)
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ResidualRowDecision:
    kind: RowScanDecisionKind
    remaining_before: frozenset[str]
    remaining_after: frozenset[str]
    selected_object_id: str | None = None
    iou: float | None = None
    eligible_for_ul: bool = False
    unlikelihood_eligible: bool = False
    no_atom_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "remaining_before", frozenset(self.remaining_before))
        object.__setattr__(self, "remaining_after", frozenset(self.remaining_after))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ResidualRowScanResult:
    initial_state: ResidualState
    final_state: ResidualState
    row_decisions: tuple[ResidualRowDecision, ...]
    events: tuple[CorrectionEvent, ...] = ()
    dirty_context_spans: tuple[tuple[int, int], ...] = ()
    masked_label_spans: tuple[tuple[int, int], ...] = ()
    type_loss_mask_spans: tuple[tuple[int, int], ...] = ()
    no_atom_reasons: tuple[str, ...] = ()
    retained_prefix_end: int | None = None
    dropped_sample: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_decisions", tuple(self.row_decisions))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "dirty_context_spans", tuple(self.dirty_context_spans))
        object.__setattr__(self, "masked_label_spans", tuple(self.masked_label_spans))
        object.__setattr__(self, "type_loss_mask_spans", tuple(self.type_loss_mask_spans))
        object.__setattr__(self, "no_atom_reasons", tuple(str(reason) for reason in self.no_atom_reasons))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def initial_remaining_object_ids(self) -> frozenset[str]:
        return self.initial_state.remaining_object_ids


def enumerate_valid_actions(state: ResidualState, *, slot: str | CoordRole = "boundary") -> tuple[ValidAction, ...]:
    if slot == "boundary":
        return _enumerate_boundary_actions(state)
    if slot == "object_start":
        return _enumerate_schema_marker_actions(state, slot="object_start")
    if slot == "desc":
        return _enumerate_boundary_actions(state)
    if slot == "box_start":
        return _enumerate_schema_marker_actions(state, slot="box_start")
    if slot in ("x1", "y1", "x2", "y2"):
        return _enumerate_coord_actions(state, slot)
    raise ValueError(f"unsupported residual-set slot: {slot!r}")


def transition_state(
    state: ResidualState,
    action: ValidAction,
    *,
    strict: bool = True,
) -> ResidualState:
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
        expected_next_state = _stop_next_state(state)
        if action.next_state is not None:
            if strict and not _residual_state_transition_semantics_equal(
                action.next_state,
                expected_next_state,
            ):
                raise ValueError(
                    "residual-set action.next_state does not match transition semantics"
                )
            return action.next_state
        return expected_next_state

    if action.next_state is None:
        if strict:
            raise ValueError("residual-set strict transition requires action.next_state")
    else:
        active_after = state.active_candidate_ids & action.candidate_ids_after
        if not action.next_state.remaining_object_ids:
            if not (
                action.token_role is TokenRole.COORD
                and action.coord_role == "y2"
                and action.selected_object_id is not None
                and state.remaining_object_ids == frozenset({action.selected_object_id})
            ):
                raise ValueError(
                    "residual-set transition has empty next_state while objects remain"
                )
            expected_next_state = _next_state_for_action(
                state,
                token_role=action.token_role,
                candidate_ids_after=active_after,
                selected_object_id=action.selected_object_id,
                coord_role=action.coord_role,
                token_id=action.token_id,
            )
            if strict and not _residual_state_transition_semantics_equal(
                action.next_state,
                expected_next_state,
            ):
                raise ValueError(
                    "residual-set action.next_state does not match transition semantics"
                )
            return action.next_state
        if not action.next_state.active_candidate_ids:
            raise ValueError(
                "residual-set transition produced empty active candidates for non-STOP action"
            )
        if not active_after:
            raise ValueError(
                "residual-set transition produced empty active candidates for non-STOP action"
            )
        if action.next_state.active_candidate_ids and not action.next_state.active_candidate_ids.issubset(
            action.next_state.remaining_object_ids
        ):
            raise ValueError("residual-set action.next_state has invalid active candidates")
        expected_next_state = _next_state_for_action(
            state,
            token_role=action.token_role,
            candidate_ids_after=active_after,
            selected_object_id=action.selected_object_id,
            coord_role=action.coord_role,
            token_id=action.token_id,
        )
        if strict and not _residual_state_transition_semantics_equal(
            action.next_state,
            expected_next_state,
        ):
            raise ValueError(
                "residual-set action.next_state does not match transition semantics"
            )
        return action.next_state

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
        object_start_token_id=state.object_start_token_id,
        box_start_token_id=state.box_start_token_id,
        stop_token_id=state.stop_token_id,
        metadata=state.metadata,
    )


def _residual_state_transition_semantics_equal(
    left: ResidualState,
    right: ResidualState,
) -> bool:
    return (
        left.objects == right.objects
        and left.remaining_object_ids == right.remaining_object_ids
        and left.active_candidate_ids == right.active_candidate_ids
        and left.text_prefix_token_ids == right.text_prefix_token_ids
        and left.object_start_token_id == right.object_start_token_id
        and left.box_start_token_id == right.box_start_token_id
        and left.stop_token_id == right.stop_token_id
    )


def _stop_next_state(state: ResidualState) -> ResidualState:
    return ResidualState(
        objects=state.objects,
        remaining_object_ids=state.remaining_object_ids,
        active_candidate_ids=frozenset(),
        text_prefix_token_ids=state.text_prefix_token_ids,
        object_start_token_id=state.object_start_token_id,
        box_start_token_id=state.box_start_token_id,
        stop_token_id=state.stop_token_id,
        metadata=state.metadata,
    )


def scan_dirty_prefix_rows(
    state: ResidualState,
    rows: tuple[ObservedResidualRow, ...],
    *,
    iou_threshold: float = 0.75,
    sample_id: str = "sample-1",
    rollback_duplicate_burst: bool = False,
) -> ResidualRowScanResult:
    decisions: list[ResidualRowDecision] = []
    events: list[CorrectionEvent] = []
    dirty_context_spans: list[tuple[int, int]] = []
    masked_label_spans: list[tuple[int, int]] = []
    type_loss_mask_spans: list[tuple[int, int]] = []
    no_atom_reasons: list[str] = []
    emitted_ids: list[str] = []
    current = state
    retained_prefix_end: int | None = None
    last_stable_boundary = 0
    dropped_sample = False

    for row_index, observed in enumerate(rows):
        remaining_before = current.remaining_object_ids
        row_span = _row_span(observed)

        if observed.incomplete:
            retained_prefix_end = last_stable_boundary
            masked_label_spans.append((int(observed.object_start), int(observed.object_start)))
            decisions.append(
                ResidualRowDecision(
                    kind="trailing_incomplete",
                    remaining_before=remaining_before,
                    remaining_after=remaining_before,
                    eligible_for_ul=False,
                    unlikelihood_eligible=False,
                    no_atom_reason="trailing_incomplete_object",
                )
            )
            break

        if observed.malformed:
            reason = "malformed_retained_span" if observed.reliable_resync else "unreliable_resync_boundary"
            dirty_context_spans.append(row_span)
            masked_label_spans.append(row_span)
            type_loss_mask_spans.append(row_span)
            no_atom_reasons.append(reason)
            decisions.append(
                ResidualRowDecision(
                    kind="malformed_span",
                    remaining_before=remaining_before,
                    remaining_after=remaining_before,
                    no_atom_reason=reason,
                    metadata={"reliable_resync": bool(observed.reliable_resync)},
                )
            )
            if not observed.reliable_resync:
                retained_prefix_end = last_stable_boundary
                dropped_sample = last_stable_boundary <= 0
                break
            continue

        if not _bbox_is_valid(observed.bbox_norm1000):
            dirty_context_spans.append(row_span)
            decisions.append(
                ResidualRowDecision(
                    kind="invalid_geometry",
                    remaining_before=remaining_before,
                    remaining_after=remaining_before,
                    iou=None,
                    no_atom_reason="invalid_geometry",
                )
            )
            continue

        duplicate = _duplicate_emitted_object(
            observed,
            emitted_ids=emitted_ids,
            state=state,
            iou_threshold=float(iou_threshold),
        )
        if duplicate is not None:
            metadata: dict[str, Any] = {"row_index": int(row_index)}
            if bool(rollback_duplicate_burst):
                retained_prefix_end = int(last_stable_boundary)
                metadata["prefix_rollback"] = True
            decisions.append(
                ResidualRowDecision(
                    kind="duplicate_burst",
                    remaining_before=remaining_before,
                    remaining_after=remaining_before,
                    selected_object_id=duplicate[0],
                    iou=duplicate[1],
                    eligible_for_ul=False,
                    unlikelihood_eligible=False,
                    metadata=metadata,
                )
            )
            if bool(rollback_duplicate_burst):
                break
            continue

        selected = _select_commit_candidate(
            observed,
            current,
            iou_threshold=float(iou_threshold),
        )
        if selected is not None:
            object_id, iou = selected
            remaining_after = frozenset(item_id for item_id in remaining_before if item_id != object_id)
            current = _state_after_commit(current, remaining_after)
            emitted_ids.append(object_id)
            last_stable_boundary = int(observed.object_end or observed.object_start)
            decisions.append(
                ResidualRowDecision(
                    kind="committed",
                    remaining_before=remaining_before,
                    remaining_after=remaining_after,
                    selected_object_id=object_id,
                    iou=iou,
                    eligible_for_ul=False,
                    unlikelihood_eligible=False,
                    metadata={"row_index": int(row_index)},
                )
            )
            continue

        spatial_conflict = _select_spatial_conflict(
            observed,
            current,
            iou_threshold=float(iou_threshold),
        )
        if spatial_conflict is not None:
            object_id, iou = spatial_conflict
            no_atom_reason: str | None = None
            if observed.reliable_span:
                event = _spatial_wrong_desc_event(
                    state=current,
                    target_id=object_id,
                    observed=observed,
                    sample_id=sample_id,
                    row_index=row_index,
                )
                if event is None:
                    no_atom_reason = "desc_atom_crosses_unreliable_boundary"
                    no_atom_reasons.append(no_atom_reason)
                else:
                    events.append(event)
            else:
                no_atom_reason = "unreliable_desc_divergence_span"
                no_atom_reasons.append(no_atom_reason)
            decisions.append(
                ResidualRowDecision(
                    kind="spatial_wrong_desc_conflict",
                    remaining_before=remaining_before,
                    remaining_after=remaining_before,
                    selected_object_id=None,
                    iou=iou,
                    eligible_for_ul=False,
                    unlikelihood_eligible=False,
                    no_atom_reason=no_atom_reason,
                    metadata={"conflicting_object_id": object_id},
                )
            )
            continue

        dirty_context_spans.append(row_span)
        decisions.append(
            ResidualRowDecision(
                kind="unmatched_dirty_context",
                remaining_before=remaining_before,
                remaining_after=remaining_before,
                no_atom_reason="no_matching_residual_object",
            )
        )

    return ResidualRowScanResult(
        initial_state=state,
        final_state=current,
        row_decisions=tuple(decisions),
        events=tuple(events),
        dirty_context_spans=tuple(dirty_context_spans),
        masked_label_spans=tuple(masked_label_spans),
        type_loss_mask_spans=tuple(type_loss_mask_spans),
        no_atom_reasons=tuple(no_atom_reasons),
        retained_prefix_end=retained_prefix_end,
        dropped_sample=dropped_sample,
    )


def _enumerate_schema_marker_actions(
    state: ResidualState,
    *,
    slot: SchemaSlot,
) -> tuple[ValidAction, ...]:
    if not state.remaining_object_ids:
        return _enumerate_boundary_actions(state)

    token_id = (
        state.object_start_token_id if slot == "object_start" else state.box_start_token_id
    )
    if token_id is None:
        raise ValueError(f"residual-set {slot} action requires configured schema token id")
    candidate_ids_after = frozenset(
        obj.object_id for obj in _active_remaining_objects(state)
    )
    if not candidate_ids_after:
        raise ValueError(f"residual-set {slot} action has no active candidates")
    selected_object_id = (
        next(iter(candidate_ids_after))
        if len(candidate_ids_after) == 1
        else None
    )
    return (
        ValidAction(
            token_id=int(token_id),
            token_role=TokenRole.SCHEMA,
            token_text=slot,
            candidate_ids_after=candidate_ids_after,
            next_state=_next_state_for_action(
                state,
                token_role=TokenRole.SCHEMA,
                candidate_ids_after=candidate_ids_after,
                selected_object_id=selected_object_id,
                coord_role=None,
                token_id=int(token_id),
            ),
            selected_object_id=selected_object_id,
            metadata=_action_metadata_for_candidates(state, candidate_ids_after),
        ),
    )


def _enumerate_boundary_actions(state: ResidualState) -> tuple[ValidAction, ...]:
    if not state.remaining_object_ids:
        next_state = ResidualState(
            objects=state.objects,
            remaining_object_ids=frozenset(),
            active_candidate_ids=frozenset(),
            text_prefix_token_ids=state.text_prefix_token_ids,
            object_start_token_id=state.object_start_token_id,
            box_start_token_id=state.box_start_token_id,
            stop_token_id=state.stop_token_id,
            metadata=state.metadata,
        )
        return (
            ValidAction(
                token_id=state.stop_token_id,
                token_role=TokenRole.STOP,
                token_text=_STOP_TOKEN_TEXT,
                candidate_ids_after=frozenset(),
                next_state=next_state,
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
        state=state,
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
        state=state,
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
    state: ResidualState,
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
                next_state=_next_state_for_action(
                    state,
                    token_role=token_role,
                    candidate_ids_after=candidate_ids_after,
                    selected_object_id=selected_object_id,
                    coord_role=coord_role,
                    token_id=token_id,
                ),
                selected_object_id=selected_object_id,
                coord_role=coord_role,
                metadata=_action_metadata_for_candidates(state, candidate_ids_after),
            )
        )
    return tuple(actions)


def _next_state_for_action(
    state: ResidualState,
    *,
    token_role: TokenRole,
    candidate_ids_after: frozenset[str],
    selected_object_id: str | None,
    coord_role: CoordRole | None,
    token_id: int,
) -> ResidualState:
    remaining_object_ids = state.remaining_object_ids
    active_candidate_ids = candidate_ids_after
    text_prefix_token_ids = state.text_prefix_token_ids

    if token_role is TokenRole.TEXT:
        text_prefix_token_ids = (*text_prefix_token_ids, int(token_id))

    if token_role is TokenRole.COORD and coord_role == "y2" and selected_object_id is not None:
        remaining_object_ids = frozenset(
            object_id for object_id in state.remaining_object_ids if object_id != selected_object_id
        )
        active_candidate_ids = remaining_object_ids
        text_prefix_token_ids = ()

    return ResidualState(
        objects=state.objects,
        remaining_object_ids=remaining_object_ids,
        active_candidate_ids=active_candidate_ids,
        text_prefix_token_ids=text_prefix_token_ids,
        object_start_token_id=state.object_start_token_id,
        box_start_token_id=state.box_start_token_id,
        stop_token_id=state.stop_token_id,
        metadata=state.metadata,
    )


def _with_action_metadata(
    actions: tuple[ValidAction, ...],
    metadata: Mapping[str, Any],
) -> tuple[ValidAction, ...]:
    merged: list[ValidAction] = []
    for action in actions:
        merged_metadata = dict(action.metadata)
        merged_metadata.update(metadata)
        merged.append(replace(action, metadata=merged_metadata))
    return tuple(merged)


def _row_span(row: ObservedResidualRow) -> tuple[int, int]:
    end = row.object_end if row.object_end is not None else row.object_start
    return (int(row.object_start), int(end))


def _bbox_is_valid(bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return all(math.isfinite(value) for value in bbox) and x1 < x2 and y1 < y2


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _object_bbox(obj: ResidualObject) -> tuple[float, float, float, float] | None:
    bbox = obj.metadata.get("bbox_norm1000")
    if bbox is None:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except TypeError:
        return None
    if len(values) != 4 or not _bbox_is_valid(values):
        return None
    return values


def _normalize_desc(desc: str | None) -> str:
    if desc is None:
        return ""
    return " ".join(str(desc).strip().lower().split())


def _object_desc_norm(obj: ResidualObject) -> str:
    raw = obj.metadata.get("desc_norm")
    if raw is not None:
        return _normalize_desc(str(raw))
    texts = obj.desc_token_texts
    if texts is not None and all(part is not None for part in texts):
        return _normalize_desc("".join(str(part) for part in texts))
    return _normalize_desc(obj.object_id)


def _object_source_rank(obj: ResidualObject) -> int:
    source = str(obj.metadata.get("source", "")).lower()
    support = obj.metadata.get("support_provenance", ())
    if isinstance(support, str):
        support_values = {support.lower()}
    else:
        try:
            support_values = {str(value).lower() for value in support}
        except TypeError:
            support_values = {str(support).lower()}
    if source == "ul" or "ul" in support_values:
        return 1
    return 0


def _select_commit_candidate(
    row: ObservedResidualRow,
    state: ResidualState,
    *,
    iou_threshold: float,
) -> tuple[str, float] | None:
    if row.bbox_norm1000 is None:
        return None
    row_desc = _normalize_desc(row.desc)
    candidates: list[tuple[int, float, float, str, float]] = []
    for object_id in sorted(state.remaining_object_ids):
        obj = state.objects_by_id[object_id]
        if _object_desc_norm(obj) != row_desc:
            continue
        bbox = _object_bbox(obj)
        if bbox is None:
            continue
        iou = _bbox_iou(row.bbox_norm1000, bbox)
        if iou < iou_threshold:
            continue
        candidates.append(
            (
                _object_source_rank(obj),
                -float(iou),
                _center_distance(row.bbox_norm1000, bbox),
                str(object_id),
                float(iou),
            )
        )
    if not candidates:
        return None
    _source_rank, _neg_iou, _distance, object_id, iou = min(candidates)
    return object_id, iou


def _select_spatial_conflict(
    row: ObservedResidualRow,
    state: ResidualState,
    *,
    iou_threshold: float,
) -> tuple[str, float] | None:
    if row.bbox_norm1000 is None:
        return None
    row_desc = _normalize_desc(row.desc)
    candidates: list[tuple[float, float, str, float]] = []
    for object_id in sorted(state.remaining_object_ids):
        obj = state.objects_by_id[object_id]
        if _object_desc_norm(obj) == row_desc:
            continue
        bbox = _object_bbox(obj)
        if bbox is None:
            continue
        iou = _bbox_iou(row.bbox_norm1000, bbox)
        if iou < iou_threshold:
            continue
        candidates.append(
            (
                -float(iou),
                _center_distance(row.bbox_norm1000, bbox),
                str(object_id),
                float(iou),
            )
        )
    if not candidates:
        return None
    _neg_iou, _distance, object_id, iou = min(candidates)
    return object_id, iou


def _duplicate_emitted_object(
    row: ObservedResidualRow,
    *,
    emitted_ids: list[str],
    state: ResidualState,
    iou_threshold: float,
) -> tuple[str, float] | None:
    if row.bbox_norm1000 is None:
        return None
    row_desc = _normalize_desc(row.desc)
    candidates: list[tuple[float, str, float]] = []
    for object_id in emitted_ids:
        obj = state.objects_by_id[object_id]
        if _object_desc_norm(obj) != row_desc:
            continue
        bbox = _object_bbox(obj)
        if bbox is None:
            continue
        iou = _bbox_iou(row.bbox_norm1000, bbox)
        if iou >= iou_threshold:
            candidates.append((-float(iou), str(object_id), float(iou)))
    if not candidates:
        return None
    _neg_iou, object_id, iou = min(candidates)
    return object_id, iou


def _state_after_commit(
    state: ResidualState,
    remaining_after: frozenset[str],
) -> ResidualState:
    return ResidualState(
        objects=state.objects,
        remaining_object_ids=remaining_after,
        active_candidate_ids=remaining_after,
        text_prefix_token_ids=(),
        object_start_token_id=state.object_start_token_id,
        box_start_token_id=state.box_start_token_id,
        stop_token_id=state.stop_token_id,
        metadata=state.metadata,
    )


def _spatial_wrong_desc_event(
    *,
    state: ResidualState,
    target_id: str,
    observed: ObservedResidualRow,
    sample_id: str,
    row_index: int,
) -> CorrectionEvent | None:
    target = state.objects_by_id[target_id]
    observed_tokens = observed.desc_token_ids
    if observed_tokens is None and observed.desc is not None:
        observed_tokens = tuple(ord(ch) for ch in observed.desc)
    observed_tokens = tuple(observed_tokens or ())
    divergence = _first_desc_divergence(observed_tokens, target.desc_token_ids)
    if divergence is None:
        return None
    if observed.desc_token_positions is None or int(divergence) >= len(observed.desc_token_positions):
        return None

    cursor_state = state
    for token_id in target.desc_token_ids[:divergence]:
        actions = enumerate_valid_actions(cursor_state, slot="desc")
        selected = _selected_action_for_token(actions, int(token_id))
        cursor_state = transition_state(cursor_state, selected)

    actions = enumerate_valid_actions(cursor_state, slot="desc")
    if not actions:
        return None
    actions = _with_action_metadata(
        actions,
        {"loss_weight": 0.25, "label_conflict_weight": 0.25},
    )
    selected_token_id = int(target.desc_token_ids[divergence])
    selected_action = _selected_action_for_token(actions, selected_token_id)
    target_position = int(observed.desc_token_positions[int(divergence)])
    if target_position <= int(observed.object_start):
        return None
    if observed.object_end is not None and target_position >= int(observed.object_end):
        return None

    draft = CorrectionAtomDraft(
        correction_kind="spatial_wrong_desc_conflict",
        target_position=target_position,
        logit_position=target_position - 1,
        valid_actions=actions,
        selected_action=selected_action,
        metadata={
            "slot": "desc",
            "row_index": int(row_index),
            "selected_object_id": target_id,
            "label_conflict_weight": 0.25,
            "desc_divergence": int(divergence),
            "observed_desc_token_position": int(target_position),
            "no_atom_reason": None,
        },
    )
    return CorrectionEvent(
        correction_kind="spatial_wrong_desc_conflict",
        sample_id=str(sample_id),
        atom_drafts=(draft,),
        metadata={
            "correction_builder": "stage2_residual_dirty_prefix_scan_v1",
            "row_index": int(row_index),
            "conflicting_object_id": target_id,
            "desc_divergence": int(divergence),
        },
    )


def _first_desc_divergence(
    observed: tuple[int, ...],
    target: tuple[int, ...],
) -> int | None:
    limit = min(len(observed), len(target))
    for index in range(limit):
        if int(observed[index]) != int(target[index]):
            return index
    if len(observed) != len(target):
        return limit
    return None


def _selected_action_for_token(
    actions: tuple[ValidAction, ...],
    token_id: int,
) -> ValidAction:
    for action in actions:
        if int(action.token_id) == int(token_id):
            return action
    raise ValueError("residual-set selected token is not a valid action")


def _action_metadata_for_candidates(
    state: ResidualState,
    candidate_ids: frozenset[str],
) -> Mapping[str, Any]:
    support: set[str] = set()
    weights: list[float] = []
    for candidate_id in sorted(candidate_ids):
        obj = state.objects_by_id.get(candidate_id)
        if obj is None:
            continue
        raw_support = obj.metadata.get("support_provenance", ("labeled",))
        if isinstance(raw_support, str):
            support.add(raw_support)
        else:
            try:
                support.update(str(item) for item in raw_support)
            except TypeError:
                support.add(str(raw_support))
        weights.append(float(obj.loss_weight))
    if not support:
        support.add("labeled")
    return {
        "support_provenance": tuple(sorted(support)),
        "loss_weight": max(weights) if weights else 1.0,
    }


def _derive_text_token_text(obj: ResidualObject, cursor: int) -> tuple[str, bool]:
    if obj.desc_token_texts is not None:
        token_text = obj.desc_token_texts[cursor]
        if token_text is not None:
            return token_text, True
    return f"token_id={obj.desc_token_ids[cursor]}", False


__all__ = [
    "CoordRole",
    "SchemaSlot",
    "CorrectionKind",
    "RowScanDecisionKind",
    "ResidualObject",
    "ValidAction",
    "ResidualState",
    "CorrectionAtomDraft",
    "CorrectionEvent",
    "ObservedResidualRow",
    "ResidualRowDecision",
    "ResidualRowScanResult",
    "enumerate_valid_actions",
    "scan_dirty_prefix_rows",
    "transition_state",
]
