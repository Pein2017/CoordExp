from __future__ import annotations

import pytest
import torch

from src.trainers.rollout_correction.residual_set import (
    CorrectionAtomDraft,
    CorrectionEvent,
    CorrectionKind,
    CoordRole,
    ObservedResidualRow,
    ResidualObject,
    ResidualRowScanResult,
    ResidualState,
    ValidAction,
    enumerate_valid_actions,
    scan_dirty_prefix_rows,
    transition_state,
)
from src.trainers.rollout_correction.teacher_forcing_adapter import (
    build_residual_set_target_ir,
)
from src.trainers.rollout_correction.target_builder import (
    _can_render_observed_prefix,
    _observed_prefix_objects,
    _with_selected_path_metadata,
)
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


def coord_token(value: int) -> int:
    return 10_000 + value


def make_object(
    object_id: str,
    desc: str,
    *,
    x1: int,
    y1: int = 20,
    x2: int = 30,
    y2: int = 40,
    loss_weight: float = 1.0,
    source: str = "labeled",
) -> ResidualObject:
    first_token_text = desc.split("_", 1)[0]
    return ResidualObject(
        object_id=object_id,
        desc_token_ids=tuple(ord(ch) for ch in desc),
        desc_token_texts=(first_token_text, *(None for _ in desc[1:])),
        coord_token_ids={
            "x1": coord_token(x1),
            "y1": coord_token(y1),
            "x2": coord_token(x2),
            "y2": coord_token(y2),
        },
        loss_weight=loss_weight,
        metadata={
            "bbox_norm1000": (x1, y1, x2, y2),
            "desc_norm": desc,
            "support_provenance": (source,),
            "source": source,
        },
    )


def make_state_for_objects(*objects: ResidualObject) -> ResidualState:
    object_ids = frozenset(obj.object_id for obj in objects)
    return ResidualState(
        objects=objects,
        remaining_object_ids=object_ids,
        active_candidate_ids=object_ids,
        stop_token_id=999,
    )


def only_action(actions: tuple[ValidAction, ...]) -> ValidAction:
    assert len(actions) == 1
    return actions[0]


def valid_coord_actions(state: ResidualState, role: CoordRole) -> tuple[ValidAction, ...]:
    return enumerate_valid_actions(state, slot=role)


def test_observed_prefix_with_forbidden_marker_is_not_clean_renderable() -> None:
    rows = [
        ObservedResidualRow(
            object_start=0,
            object_end=6,
            desc="person<|box_start|>",
            bbox_norm1000=(10, 20, 30, 40),
        )
    ]

    prefix_objects = _observed_prefix_objects(rows=rows, retained_prefix_end=6)

    assert prefix_objects == []
    assert not _can_render_observed_prefix(
        rows=rows,
        retained_prefix_end=6,
        prefix_objects=prefix_objects,
    )


def apply_action(state: ResidualState, action: ValidAction) -> ResidualState:
    return transition_state(state, action)


def row(
    desc: str | None,
    bbox: tuple[int, int, int, int] | None,
    *,
    object_start: int = 4,
    object_end: int | None = 12,
    reliable_span: bool = True,
    reliable_resync: bool = True,
    malformed: bool = False,
    incomplete: bool = False,
    desc_token_ids: tuple[int, ...] | None = None,
    desc_token_positions: tuple[int, ...] | None = None,
) -> ObservedResidualRow:
    return ObservedResidualRow(
        desc=desc,
        bbox_norm1000=bbox,
        object_start=object_start,
        object_end=object_end,
        reliable_span=reliable_span,
        reliable_resync=reliable_resync,
        malformed=malformed,
        incomplete=incomplete,
        desc_token_ids=desc_token_ids,
        desc_token_positions=desc_token_positions,
    )


def make_role_vocab() -> RoleVocab:
    return RoleVocab(
        schema_token_ids=frozenset({1}),
        text_token_ids=frozenset(ord(ch) for ch in "person_left_rightcar")
        | frozenset({101, 201}),
        coord_token_ids=frozenset(coord_token(value) for value in range(0, 1001)),
        stop_token_id=999,
    )


def make_text_action(
    token_id: int,
    *,
    loss_weight: float = 1.0,
    support_provenance: tuple[str, ...] = ("labeled",),
) -> ValidAction:
    return ValidAction(
        token_id=token_id,
        token_role=TokenRole.TEXT,
        token_text=f"text-{token_id}",
        candidate_ids_after=frozenset({"a"}),
        metadata={
            "loss_weight": loss_weight,
            "support_provenance": support_provenance,
        },
    )


def make_coord_action(
    token_id: int,
    coord_role: CoordRole,
    *,
    selected_object_id: str = "a",
) -> ValidAction:
    return ValidAction(
        token_id=token_id,
        token_role=TokenRole.COORD,
        token_text=str(token_id),
        candidate_ids_after=frozenset({selected_object_id}),
        selected_object_id=selected_object_id,
        coord_role=coord_role,
    )


def make_event(
    *,
    kind: CorrectionKind = "transition_failure",
    target_position: int = 8,
    logit_position: int = 7,
    valid_actions: tuple[ValidAction, ...],
    selected_action: ValidAction | None = None,
    draft_metadata: dict[str, object] | None = None,
    event_metadata: dict[str, object] | None = None,
) -> CorrectionEvent:
    draft = CorrectionAtomDraft(
        correction_kind=kind,
        target_position=target_position,
        logit_position=logit_position,
        valid_actions=valid_actions,
        selected_action=selected_action,
        metadata={"source": "unit-test", **(draft_metadata or {})},
    )
    return CorrectionEvent(
        correction_kind=kind,
        sample_id="sample-1",
        atom_drafts=(draft,),
        metadata={"adapter": "not-yet", **(event_metadata or {})},
    )


def test_shared_description_prefix_coalesces_one_valid_boundary_action() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120),
        make_object("b", "person_right", x1=640),
    )

    action = only_action(state.valid_actions_at_boundary())

    assert action.token_role is TokenRole.TEXT
    assert action.token_id == ord("p")
    assert action.token_text == "person"
    assert action.candidate_ids_after == frozenset({"a", "b"})
    assert action.selected_object_id is None


def test_shared_description_prefix_uses_supplied_token_text_for_non_character_token_ids() -> None:
    state = make_state_for_objects(
        ResidualObject(
            object_id="a",
            desc_token_ids=(50_257, 61_000),
            desc_token_texts=("person", "_left"),
            coord_token_ids={
                "x1": coord_token(120),
                "y1": coord_token(20),
                "x2": coord_token(30),
                "y2": coord_token(40),
            },
        ),
        ResidualObject(
            object_id="b",
            desc_token_ids=(50_257, 62_000),
            desc_token_texts=("person", "_right"),
            coord_token_ids={
                "x1": coord_token(640),
                "y1": coord_token(20),
                "x2": coord_token(30),
                "y2": coord_token(40),
            },
        ),
    )

    action = only_action(state.valid_actions_at_boundary())

    assert action.token_id == 50_257
    assert action.token_text == "person"
    assert action.candidate_ids_after == frozenset({"a", "b"})


def test_stop_is_valid_only_when_residual_set_is_empty() -> None:
    nonempty = make_state_for_objects(make_object("a", "person_left", x1=120))
    empty = ResidualState(
        objects=(),
        remaining_object_ids=frozenset(),
        active_candidate_ids=frozenset(),
        stop_token_id=999,
    )

    assert all(action.token_role is not TokenRole.STOP for action in nonempty.valid_actions_at_boundary())

    stop = only_action(empty.valid_actions_at_boundary())
    assert stop.token_role is TokenRole.STOP
    assert stop.token_id == 999
    assert stop.token_text == "<|im_end|>"
    assert stop.candidate_ids_after == frozenset()


def test_forged_stop_transition_is_rejected_for_nonempty_residual_state() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120))
    forged_stop = ValidAction(
        token_id=999,
        token_role=TokenRole.STOP,
        token_text="<|im_end|>",
        candidate_ids_after=frozenset(),
    )

    with pytest.raises(ValueError, match="empty residual state"):
        transition_state(state, forged_stop)


def test_strict_stop_transition_rejects_forged_next_state() -> None:
    state = ResidualState(
        objects=(),
        remaining_object_ids=frozenset(),
        active_candidate_ids=frozenset(),
        text_prefix_token_ids=(42,),
        stop_token_id=999,
    )
    forged_next_state = ResidualState(
        objects=(),
        remaining_object_ids=frozenset(),
        active_candidate_ids=frozenset(),
        text_prefix_token_ids=(123,),
        stop_token_id=999,
    )
    forged_stop = ValidAction(
        token_id=999,
        token_role=TokenRole.STOP,
        token_text="<|im_end|>",
        candidate_ids_after=frozenset(),
        next_state=forged_next_state,
    )

    with pytest.raises(ValueError, match="transition semantics"):
        transition_state(state, forged_stop)


def test_x1_valid_action_commits_subsequent_bbox_to_same_object() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120),
        make_object("b", "person_right", x1=640),
        make_object("c", "car", x1=850),
    )

    assert state.remaining_object_ids == frozenset({"a", "b", "c"})
    actions = valid_coord_actions(state, "x1")

    assert {action.token_id for action in actions} == {coord_token(120), coord_token(640), coord_token(850)}
    chosen = only_action(tuple(action for action in actions if action.token_id == coord_token(640)))
    assert chosen.next_state is not None
    assert chosen.next_state.active_candidate_ids == frozenset({"b"})
    assert chosen.next_state.remaining_object_ids == frozenset({"a", "b", "c"})

    narrowed = apply_action(state, chosen)
    assert narrowed.active_candidate_ids == frozenset({"b"})
    assert narrowed.remaining_object_ids == frozenset({"a", "b", "c"})

    for role in ("y1", "x2", "y2"):
        action = only_action(valid_coord_actions(narrowed, role))
        narrowed = apply_action(narrowed, action)

    assert narrowed.remaining_object_ids == frozenset({"a", "c"})
    assert narrowed.active_candidate_ids == frozenset({"a", "c"})


def test_shared_x1_keeps_bbox_tail_ambiguous_until_y1() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=640, y1=120),
        make_object("b", "person_right", x1=640, y1=300),
        make_object("c", "car", x1=850, y1=120),
    )

    x1 = only_action(tuple(action for action in valid_coord_actions(state, "x1") if action.token_id == coord_token(640)))
    after_x1 = apply_action(state, x1)
    y1_actions = valid_coord_actions(after_x1, "y1")

    assert after_x1.active_candidate_ids == frozenset({"a", "b"})
    assert {action.token_id for action in y1_actions} == {coord_token(120), coord_token(300)}


def test_coordinate_enumeration_supports_x2_and_y2_exact_token_roles() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120, x2=420, y2=700),
        make_object("b", "person_right", x1=640, x2=520, y2=800),
    )

    x2_actions = valid_coord_actions(state, "x2")
    y2_actions = valid_coord_actions(state, "y2")

    assert {(action.coord_role, action.token_id) for action in x2_actions} == {
        ("x2", coord_token(420)),
        ("x2", coord_token(520)),
    }
    assert {(action.coord_role, action.token_id) for action in y2_actions} == {
        ("y2", coord_token(700)),
        ("y2", coord_token(800)),
    }


def test_singleton_coordinate_action_sets_selected_object_id() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120),
        make_object("b", "person_right", x1=640),
    )

    action = only_action(tuple(action for action in valid_coord_actions(state, "x1") if action.token_id == coord_token(640)))

    assert action.candidate_ids_after == frozenset({"b"})
    assert action.selected_object_id == "b"


def test_ambiguous_desc_choice_collapses_later_object_internal_tokens_to_strict() -> None:
    left = ResidualObject(
        object_id="left",
        desc_token_ids=(101, 111),
        desc_token_texts=("left", "_tail"),
        coord_token_ids={
            "x1": coord_token(120),
            "y1": coord_token(20),
            "x2": coord_token(30),
            "y2": coord_token(40),
        },
    )
    right = ResidualObject(
        object_id="right",
        desc_token_ids=(201, 211),
        desc_token_texts=("right", "_tail"),
        coord_token_ids={
            "x1": coord_token(640),
            "y1": coord_token(20),
            "x2": coord_token(30),
            "y2": coord_token(40),
        },
    )
    state = ResidualState(
        objects=(left, right),
        remaining_object_ids=frozenset({"left", "right"}),
        active_candidate_ids=frozenset({"left", "right"}),
        object_start_token_id=1,
        box_start_token_id=2,
        stop_token_id=999,
    )

    object_start = only_action(enumerate_valid_actions(state, slot="object_start"))
    after_object_start = transition_state(state, object_start)
    first_desc_actions = enumerate_valid_actions(after_object_start, slot="desc")

    assert {action.token_id for action in first_desc_actions} == {101, 201}
    assert all(len(action.candidate_ids_after) == 1 for action in first_desc_actions)

    choose_right = only_action(
        tuple(action for action in first_desc_actions if action.token_id == 201)
    )
    after_choice = transition_state(after_object_start, choose_right)
    second_desc = only_action(enumerate_valid_actions(after_choice, slot="desc"))
    after_second_desc = transition_state(after_choice, second_desc)
    box_start = only_action(enumerate_valid_actions(after_second_desc, slot="box_start"))
    after_box_start = transition_state(after_second_desc, box_start)
    x1 = only_action(enumerate_valid_actions(after_box_start, slot="x1"))

    assert after_choice.active_candidate_ids == frozenset({"right"})
    assert second_desc.token_id == 211
    assert second_desc.candidate_ids_after == frozenset({"right"})
    assert box_start.token_id == 2
    assert box_start.candidate_ids_after == frozenset({"right"})
    assert x1.token_id == coord_token(640)
    assert x1.candidate_ids_after == frozenset({"right"})


def test_invalid_bbox_row_is_dirty_context_and_does_not_update_remaining_set() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120, x2=220))

    result: ResidualRowScanResult = scan_dirty_prefix_rows(
        state,
        (row("person_left", (220, 20, 120, 40), object_start=5, object_end=11),),
    )

    decision = result.row_decisions[0]
    assert result.initial_remaining_object_ids == frozenset({"a"})
    assert decision.remaining_before == frozenset({"a"})
    assert decision.kind == "invalid_geometry"
    assert decision.iou is None
    assert decision.remaining_after == frozenset({"a"})
    assert result.final_state.remaining_object_ids == frozenset({"a"})
    assert result.dirty_context_spans == ((5, 11),)
    assert result.events == ()


def test_trailing_incomplete_object_is_removed_to_last_stable_boundary() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120, x2=220),
        make_object("b", "car", x1=500, x2=650),
    )

    result = scan_dirty_prefix_rows(
        state,
        (
            row("person_left", (120, 20, 220, 40), object_start=2, object_end=10),
            row("car", None, object_start=10, object_end=None, incomplete=True),
        ),
    )

    assert result.row_decisions[0].remaining_before == frozenset({"a", "b"})
    assert result.row_decisions[0].remaining_after == frozenset({"b"})
    assert result.row_decisions[1].kind == "trailing_incomplete"
    assert result.row_decisions[1].remaining_before == frozenset({"b"})
    assert result.row_decisions[1].remaining_after == frozenset({"b"})
    assert result.final_state.remaining_object_ids == frozenset({"b"})
    assert result.retained_prefix_end == 10
    assert result.masked_label_spans == ((10, 10),)


def test_spatial_wrong_desc_conflict_emits_low_weight_desc_atom_when_span_reliable() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120, x2=220))

    result = scan_dirty_prefix_rows(
        state,
        (
            row(
                "person_right",
                (120, 20, 220, 40),
                object_start=5,
                object_end=19,
                reliable_span=True,
                desc_token_positions=tuple(range(6, 6 + len("person_right"))),
            ),
        ),
    )

    decision = result.row_decisions[0]
    assert decision.remaining_before == frozenset({"a"})
    assert decision.kind == "spatial_wrong_desc_conflict"
    assert decision.selected_object_id is None
    assert decision.remaining_after == frozenset({"a"})
    assert result.final_state.remaining_object_ids == frozenset({"a"})
    assert decision.eligible_for_ul is False
    assert len(result.events) == 1
    atom = result.events[0].atom_drafts[0]
    assert atom.correction_kind == "spatial_wrong_desc_conflict"
    assert atom.metadata["label_conflict_weight"] == 0.25
    assert atom.metadata["slot"] == "desc"
    assert atom.metadata["no_atom_reason"] is None


def test_spatial_wrong_desc_conflict_records_no_atom_reason_without_desc_positions() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120, x2=220))

    result = scan_dirty_prefix_rows(
        state,
        (
            row(
                "person_right",
                (120, 20, 220, 40),
                object_start=5,
                object_end=19,
                reliable_span=True,
            ),
        ),
    )

    decision = result.row_decisions[0]
    assert decision.kind == "spatial_wrong_desc_conflict"
    assert result.events == ()
    assert result.no_atom_reasons == ("desc_atom_crosses_unreliable_boundary",)
    assert decision.no_atom_reason == "desc_atom_crosses_unreliable_boundary"


def test_spatial_wrong_desc_conflict_records_no_atom_reason_when_span_unreliable() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120, x2=220))

    result = scan_dirty_prefix_rows(
        state,
        (
            row(
                "person_right",
                (120, 20, 220, 40),
                object_start=5,
                object_end=19,
                reliable_span=False,
            ),
        ),
    )

    decision = result.row_decisions[0]
    assert decision.remaining_before == frozenset({"a"})
    assert decision.kind == "spatial_wrong_desc_conflict"
    assert decision.remaining_after == frozenset({"a"})
    assert result.events == ()
    assert result.no_atom_reasons == ("unreliable_desc_divergence_span",)
    assert decision.no_atom_reason == "unreliable_desc_divergence_span"


def test_duplicate_burst_is_uncommitted_and_cannot_vote_for_ul() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120, x2=220),
        make_object("b", "car", x1=500, x2=650),
    )

    result = scan_dirty_prefix_rows(
        state,
        (
            row("person_left", (120, 20, 220, 40), object_start=2, object_end=10),
            row("person_left", (120, 20, 220, 40), object_start=10, object_end=18),
        ),
    )

    assert result.row_decisions[0].kind == "committed"
    assert result.row_decisions[0].remaining_before == frozenset({"a", "b"})
    assert result.row_decisions[0].remaining_after == frozenset({"b"})
    duplicate = result.row_decisions[1]
    assert duplicate.kind == "duplicate_burst"
    assert duplicate.remaining_before == frozenset({"b"})
    assert duplicate.remaining_after == frozenset({"b"})
    assert duplicate.eligible_for_ul is False
    assert duplicate.unlikelihood_eligible is False
    assert result.final_state.remaining_object_ids == frozenset({"b"})
    assert result.events == ()


def test_duplicate_burst_prefix_rollback_truncates_at_last_stable_boundary() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120, x2=220),
        make_object("b", "car", x1=500, x2=650),
    )

    result = scan_dirty_prefix_rows(
        state,
        (
            row("person_left", (120, 20, 220, 40), object_start=2, object_end=10),
            row("person_left", (120, 20, 220, 40), object_start=10, object_end=18),
            row("car", (500, 20, 650, 40), object_start=18, object_end=26),
        ),
        rollback_duplicate_burst=True,
    )

    assert [decision.kind for decision in result.row_decisions] == [
        "committed",
        "duplicate_burst",
    ]
    duplicate = result.row_decisions[1]
    assert duplicate.metadata["prefix_rollback"] is True
    assert result.final_state.remaining_object_ids == frozenset({"b"})
    assert result.retained_prefix_end == 10
    assert result.events == ()


def test_dirty_prefix_scan_uses_stricter_duplicate_threshold_than_commit_threshold() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=100, y1=100, x2=200, y2=200),
        make_object("b", "car", x1=500, y1=100, x2=600, y2=200),
    )

    result = scan_dirty_prefix_rows(
        state,
        (
            row("person_left", (100, 100, 200, 200), object_start=2, object_end=10),
            row("person_left", (105, 100, 205, 200), object_start=10, object_end=18),
        ),
        commit_iou_threshold=0.75,
        duplicate_burst_iou_threshold=0.95,
    )

    assert [decision.kind for decision in result.row_decisions] == [
        "committed",
        "unmatched_dirty_context",
    ]
    assert result.row_decisions[1].iou is None
    assert result.final_state.remaining_object_ids == frozenset({"b"})


def test_dirty_prefix_scan_does_not_commit_low_iou_assignment_gate_overlap() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=100, y1=100, x2=200, y2=200),
    )

    result = scan_dirty_prefix_rows(
        state,
        (row("person_left", (120, 100, 220, 200), object_start=2, object_end=10),),
        iou_threshold=0.10,
        commit_iou_threshold=0.75,
        duplicate_burst_iou_threshold=0.95,
    )

    decision = result.row_decisions[0]
    assert decision.kind == "unmatched_dirty_context"
    assert decision.remaining_before == frozenset({"a"})
    assert decision.remaining_after == frozenset({"a"})
    assert result.final_state.remaining_object_ids == frozenset({"a"})


def test_malformed_span_context_has_no_atoms_or_type_loss() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120, x2=220))

    result = scan_dirty_prefix_rows(
        state,
        (
            row(
                None,
                None,
                object_start=3,
                object_end=9,
                reliable_span=True,
                reliable_resync=True,
                malformed=True,
            ),
        ),
    )

    decision = result.row_decisions[0]
    assert decision.kind == "malformed_span"
    assert decision.remaining_before == frozenset({"a"})
    assert decision.remaining_after == frozenset({"a"})
    assert result.final_state.remaining_object_ids == frozenset({"a"})
    assert result.events == ()
    assert result.no_atom_reasons == ("malformed_retained_span",)
    assert result.type_loss_mask_spans == ((3, 9),)


def test_unreliable_malformed_resync_at_first_row_drops_sample() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120, x2=220))

    result = scan_dirty_prefix_rows(
        state,
        (
            row(
                None,
                None,
                object_start=3,
                object_end=9,
                reliable_resync=False,
                malformed=True,
            ),
        ),
    )

    decision = result.row_decisions[0]
    assert decision.kind == "malformed_span"
    assert decision.remaining_before == frozenset({"a"})
    assert decision.remaining_after == frozenset({"a"})
    assert decision.metadata["reliable_resync"] is False
    assert result.final_state.remaining_object_ids == frozenset({"a"})
    assert result.retained_prefix_end == 0
    assert result.dropped_sample is True
    assert result.no_atom_reasons == ("unreliable_resync_boundary",)
    assert result.type_loss_mask_spans == ((3, 9),)


def test_unreliable_malformed_resync_after_committed_row_retains_last_stable_boundary() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120, x2=220),
        make_object("b", "car", x1=500, x2=650),
    )

    result = scan_dirty_prefix_rows(
        state,
        (
            row("person_left", (120, 20, 220, 40), object_start=2, object_end=10),
            row(
                None,
                None,
                object_start=10,
                object_end=16,
                reliable_resync=False,
                malformed=True,
            ),
        ),
    )

    assert result.row_decisions[0].kind == "committed"
    decision = result.row_decisions[1]
    assert decision.kind == "malformed_span"
    assert decision.remaining_before == frozenset({"b"})
    assert decision.remaining_after == frozenset({"b"})
    assert result.final_state.remaining_object_ids == frozenset({"b"})
    assert result.retained_prefix_end == 10
    assert result.dropped_sample is False
    assert result.no_atom_reasons == ("unreliable_resync_boundary",)
    assert result.type_loss_mask_spans == ((10, 16),)


def test_row_commitment_uses_deterministic_gt_before_ul_tiebreak() -> None:
    state = make_state_for_objects(
        make_object("gt:2", "person_left", x1=100, y1=100, x2=200, y2=200, source="labeled"),
        make_object("ul:0", "person_left", x1=100, y1=100, x2=200, y2=200, source="ul"),
        make_object("gt:1", "person_left", x1=100, y1=100, x2=200, y2=200, source="labeled"),
    )

    result = scan_dirty_prefix_rows(
        state,
        (row("person_left", (100, 100, 200, 200), object_start=1, object_end=9),),
    )

    decision = result.row_decisions[0]
    assert decision.remaining_before == frozenset({"gt:2", "ul:0", "gt:1"})
    assert decision.kind == "committed"
    assert decision.selected_object_id == "gt:1"
    assert decision.remaining_after == frozenset({"gt:2", "ul:0"})
    assert result.final_state.remaining_object_ids == frozenset({"gt:2", "ul:0"})


def test_empty_non_stop_transition_is_guarded() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120))
    bad_action = ValidAction(
        token_id=123,
        token_role=TokenRole.TEXT,
        token_text="bad",
        candidate_ids_after=frozenset({"missing"}),
    )

    with pytest.raises(ValueError, match="empty active candidates"):
        transition_state(state, bad_action, strict=False)


def test_strict_transition_requires_materialized_next_state() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120))
    action = ValidAction(
        token_id=ord("p"),
        token_role=TokenRole.TEXT,
        token_text="person",
        candidate_ids_after=frozenset({"a"}),
    )

    with pytest.raises(ValueError, match="requires action.next_state"):
        transition_state(state, action)


def test_strict_transition_rejects_empty_next_state_while_objects_remain() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120),
        make_object("b", "person_right", x1=640),
    )
    empty_next_state = ResidualState(
        objects=state.objects,
        remaining_object_ids=frozenset(),
        active_candidate_ids=frozenset(),
        stop_token_id=999,
    )
    action = ValidAction(
        token_id=coord_token(120),
        token_role=TokenRole.COORD,
        token_text=str(coord_token(120)),
        candidate_ids_after=frozenset({"a"}),
        next_state=empty_next_state,
        selected_object_id="a",
        coord_role="x1",
    )

    with pytest.raises(ValueError, match="empty next_state"):
        transition_state(state, action)


def test_strict_transition_rejects_forged_y2_next_state_that_keeps_selected_object() -> None:
    obj_a = make_object("a", "person_left", x1=120)
    obj_b = make_object("b", "person_right", x1=640)
    state = ResidualState(
        objects=(obj_a, obj_b),
        remaining_object_ids=frozenset({"a", "b"}),
        active_candidate_ids=frozenset({"a"}),
        stop_token_id=999,
    )
    forged_next_state = ResidualState(
        objects=state.objects,
        remaining_object_ids=frozenset({"a", "b"}),
        active_candidate_ids=frozenset({"a"}),
        stop_token_id=999,
    )
    action = ValidAction(
        token_id=coord_token(40),
        token_role=TokenRole.COORD,
        token_text=str(coord_token(40)),
        candidate_ids_after=frozenset({"a"}),
        next_state=forged_next_state,
        selected_object_id="a",
        coord_role="y2",
    )

    with pytest.raises(ValueError, match="transition semantics"):
        transition_state(state, action)


def test_strict_transition_rejects_forged_next_state_that_drops_unselected_objects() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120),
        make_object("b", "person_right", x1=640),
    )
    forged_next_state = ResidualState(
        objects=state.objects,
        remaining_object_ids=frozenset({"a"}),
        active_candidate_ids=frozenset({"a"}),
        stop_token_id=999,
    )
    action = ValidAction(
        token_id=coord_token(120),
        token_role=TokenRole.COORD,
        token_text=str(coord_token(120)),
        candidate_ids_after=frozenset({"a"}),
        next_state=forged_next_state,
        selected_object_id="a",
        coord_role="x1",
    )

    with pytest.raises(ValueError, match="transition semantics"):
        transition_state(state, action)


def test_strict_transition_rejects_forged_next_state_active_candidate_mismatch() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120),
        make_object("b", "person_right", x1=640),
    )
    forged_next_state = ResidualState(
        objects=state.objects,
        remaining_object_ids=frozenset({"a", "b"}),
        active_candidate_ids=frozenset({"b"}),
        stop_token_id=999,
    )
    action = ValidAction(
        token_id=coord_token(120),
        token_role=TokenRole.COORD,
        token_text=str(coord_token(120)),
        candidate_ids_after=frozenset({"a"}),
        next_state=forged_next_state,
        selected_object_id="a",
        coord_role="x1",
    )

    with pytest.raises(ValueError, match="transition semantics"):
        transition_state(state, action)


def test_residual_object_constructor_validates_core_invariants() -> None:
    with pytest.raises(ValueError, match="object_id"):
        make_object("", "person_left", x1=120)

    with pytest.raises(ValueError, match="desc_token_ids"):
        ResidualObject(
            object_id="a",
            desc_token_ids=(),
            desc_token_texts=(),
            coord_token_ids={
                "x1": coord_token(120),
                "y1": coord_token(20),
                "x2": coord_token(30),
                "y2": coord_token(40),
            },
        )

    with pytest.raises(ValueError, match="coord_token_ids"):
        ResidualObject(
            object_id="a",
            desc_token_ids=(101,),
            desc_token_texts=("person",),
            coord_token_ids={
                "x1": coord_token(120),
                "y1": coord_token(20),
                "x2": coord_token(30),
            },
        )

    with pytest.raises(ValueError, match="loss_weight"):
        make_object("a", "person_left", x1=120, loss_weight=float("nan"))


def test_original_coord_mapping_mutation_does_not_change_residual_object() -> None:
    coord_tokens = {
        "x1": coord_token(120),
        "y1": coord_token(20),
        "x2": coord_token(30),
        "y2": coord_token(40),
    }
    obj = ResidualObject(
        object_id="a",
        desc_token_ids=(101,),
        desc_token_texts=("person",),
        coord_token_ids=coord_tokens,
    )

    coord_tokens["x1"] = coord_token(999)

    assert obj.coord_token_ids["x1"] == coord_token(120)


def test_residual_state_constructor_validates_candidate_sets() -> None:
    obj = make_object("a", "person_left", x1=120)

    with pytest.raises(ValueError, match="remaining_object_ids"):
        ResidualState(
            objects=(obj,),
            remaining_object_ids=frozenset({"missing"}),
            active_candidate_ids=frozenset(),
            stop_token_id=999,
        )

    with pytest.raises(ValueError, match="active_candidate_ids"):
        ResidualState(
            objects=(obj,),
            remaining_object_ids=frozenset({"a"}),
            active_candidate_ids=frozenset({"missing"}),
            stop_token_id=999,
        )

    with pytest.raises(ValueError, match="active_candidate_ids"):
        ResidualState(
            objects=(obj,),
            remaining_object_ids=frozenset({"a"}),
            active_candidate_ids=frozenset(),
            stop_token_id=999,
        )


def test_valid_action_constructor_validates_role_consistency() -> None:
    with pytest.raises(ValueError, match="selected_object_id"):
        ValidAction(
            token_id=101,
            token_role=TokenRole.TEXT,
            token_text="person",
            candidate_ids_after=frozenset({"a"}),
            selected_object_id="missing",
        )

    with pytest.raises(ValueError, match="COORD requires coord_role"):
        ValidAction(
            token_id=coord_token(120),
            token_role=TokenRole.COORD,
            token_text=str(coord_token(120)),
            candidate_ids_after=frozenset({"a"}),
        )

    with pytest.raises(ValueError, match="TEXT requires coord_role"):
        ValidAction(
            token_id=101,
            token_role=TokenRole.TEXT,
            token_text="person",
            candidate_ids_after=frozenset({"a"}),
            coord_role="x1",
        )

    with pytest.raises(ValueError, match="STOP requires empty"):
        ValidAction(
            token_id=999,
            token_role=TokenRole.STOP,
            token_text="<|im_end|>",
            candidate_ids_after=frozenset({"a"}),
        )

    with pytest.raises(ValueError, match="STOP requires canonical token_text"):
        ValidAction(
            token_id=999,
            token_role=TokenRole.STOP,
            token_text="<bad_stop>",
            candidate_ids_after=frozenset(),
        )


def test_correction_event_and_atom_draft_are_pure_records_without_ir_adapter() -> None:
    _ = make_role_vocab()
    state = make_state_for_objects(make_object("a", "person_left", x1=120))
    actions = state.valid_actions_at_boundary()

    event = make_event(target_position=11, logit_position=10, valid_actions=actions)

    assert event.correction_kind == "transition_failure"
    assert event.sample_id == "sample-1"
    assert event.atom_drafts[0].target_position == 11
    assert event.atom_drafts[0].logit_position == 10
    assert event.atom_drafts[0].valid_actions == actions
    assert event.atom_drafts[0].metadata == {"source": "unit-test"}
    assert event.metadata == {"adapter": "not-yet"}


def test_correction_event_to_ir_uses_next_token_logit_row() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101), make_text_action(201)),
        draft_metadata={"observed_token_id": 999, "anchor_position": 1},
        event_metadata={"rollout_index": 3},
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    assert ir.metadata["stage"] == "stage2"
    assert ir.metadata["stage2_surface"] == "rollout_correction"
    assert ir.metadata["objective"] == "residual_set_correction"
    assert len(ir.atoms) == 1
    atom = ir.atoms[0]
    assert atom.batch_index == 0
    assert atom.target_position == 2
    assert atom.logit_position == 1
    assert atom.logit_position + 1 == atom.target_position
    assert atom.selected_token_id == 101
    assert atom.valid_token_ids == frozenset({101, 201})
    assert atom.latent_valid_token_ids == frozenset({101, 201})
    assert atom.selected_token_role is TokenRole.TEXT
    assert atom.allowed_token_roles == frozenset({TokenRole.TEXT})
    assert atom.provenance["observed_token_id"] == 999
    assert atom.provenance["anchor_position"] == 1
    assert atom.provenance["rollout_index"] == 3


def test_event_to_ir_rejects_wrong_shift() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=2,
        valid_actions=(make_text_action(101),),
    )

    with pytest.raises(ValueError, match="target_position = logit_position \\+ 1"):
        build_residual_set_target_ir(
            input_ids=input_ids,
            batch_index=0,
            events=(event,),
            role_vocab=make_role_vocab(),
        )


def test_event_to_ir_corrects_wrong_live_token_without_adding_it_to_valid_set() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(201),),
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    atom = ir.atoms[0]
    assert atom.selected_token_id == 201
    assert atom.selected_token_id != int(input_ids[0, atom.target_position].item())
    assert atom.valid_token_ids == frozenset({201})
    assert atom.latent_valid_token_ids == frozenset({201})
    assert atom.provenance["live_token_id"] == 101
    assert atom.provenance["allow_target_token_mismatch"] is True
    assert atom.provenance["target_token_mismatch"] is True


def test_coordinate_event_to_ir_keeps_exact_coord_roles_without_repair_kind() -> None:
    input_ids = torch.tensor(
        [[11, 22, coord_token(120), coord_token(20), coord_token(30), coord_token(40)]]
    )
    state = make_state_for_objects(make_object("a", "person_left", x1=120))
    drafts = []
    for offset, coord_role in enumerate(("x1", "y1", "x2", "y2"), start=2):
        drafts.append(
            CorrectionAtomDraft(
                correction_kind="transition_failure",
                target_position=offset,
                logit_position=offset - 1,
                valid_actions=valid_coord_actions(state, coord_role),
                metadata={"anchor_position": 1},
            )
        )
    event = CorrectionEvent(
        correction_kind="transition_failure",
        sample_id="sample-1",
        atom_drafts=tuple(drafts),
        metadata={"rollout_index": 4},
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    assert len(ir.atoms) == 4
    assert [atom.coord_role for atom in ir.atoms] == ["x1", "y1", "x2", "y2"]
    assert [atom.selected_token_id for atom in ir.atoms] == [
        coord_token(120),
        coord_token(20),
        coord_token(30),
        coord_token(40),
    ]
    assert all(atom.logit_position + 1 == atom.target_position for atom in ir.atoms)
    assert all(
        atom.selected_token_id == int(input_ids[0, atom.target_position].item())
        for atom in ir.atoms
    )
    assert all(atom.selected_token_role is TokenRole.COORD for atom in ir.atoms)


def test_event_to_ir_rejects_mixed_valid_action_roles_in_one_draft() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    coord_action = ValidAction(
        token_id=coord_token(120),
        token_role=TokenRole.COORD,
        token_text=str(coord_token(120)),
        candidate_ids_after=frozenset({"a"}),
        selected_object_id="a",
        coord_role="x1",
    )
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101), coord_action),
    )

    with pytest.raises(ValueError, match="same token role"):
        build_residual_set_target_ir(
            input_ids=input_ids,
            batch_index=0,
            events=(event,),
            role_vocab=make_role_vocab(),
        )


def test_event_to_ir_rejects_bool_loss_weight_metadata() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101, loss_weight=True),),
    )

    with pytest.raises(ValueError, match="loss_weight"):
        build_residual_set_target_ir(
            input_ids=input_ids,
            batch_index=0,
            events=(event,),
            role_vocab=make_role_vocab(),
        )


def test_event_to_ir_keeps_provenance_compact_and_uses_live_token() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101), make_text_action(201)),
        draft_metadata={
            "observed_token_id": 999,
            "anchor_position": 1,
            "raw_bad_token": {"token": 999},
            "large_payload": list(range(100)),
        },
        event_metadata={
            "rollout_index": 7,
            "raw_rollout": [{"token_id": token_id} for token_id in range(100)],
        },
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    atom = ir.atoms[0]
    assert atom.selected_token_id == 101
    assert atom.provenance["observed_token_id"] == 999
    assert atom.provenance["rollout_index"] == 7
    assert atom.provenance["anchor_position"] == 1
    assert "raw_bad_token" not in atom.provenance
    assert "raw_rollout" not in atom.provenance
    assert "large_payload" not in atom.provenance


def test_event_to_ir_merges_identical_same_logit_atoms_with_compact_provenance() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    actions = (make_text_action(101), make_text_action(201))
    first = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=actions,
        draft_metadata={"observed_token_id": 901, "anchor_position": 1},
        event_metadata={"rollout_index": 3, "rollout_id": "r0"},
    )
    second = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=actions,
        draft_metadata={"observed_token_id": 902, "anchor_position": 4},
        event_metadata={"rollout_index": 3, "rollout_id": "r0"},
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(first, second),
        role_vocab=make_role_vocab(),
    )

    assert len(ir.atoms) == 1
    atom = ir.atoms[0]
    assert atom.logit_position + 1 == atom.target_position
    assert atom.selected_token_id == int(input_ids[0, atom.target_position].item())
    assert atom.selected_token_id in atom.valid_token_ids
    assert atom.selected_token_role in atom.allowed_token_roles
    assert atom.provenance["merged_provenance_count"] == 2
    assert atom.provenance["merged_observed_token_ids"] == (901, 902)
    assert atom.provenance["merged_anchor_positions"] == (1, 4)
    assert atom.provenance["merged_rollout_ids"] == ("r0",)


def test_event_to_ir_merges_same_logit_weights_and_canonical_support_provenance() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    labeled = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(
            make_text_action(
                101,
                loss_weight=0.25,
                support_provenance=("labeled",),
            ),
        ),
    )
    ul = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(
            make_text_action(
                101,
                loss_weight=1.0,
                support_provenance=("ul",),
            ),
        ),
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(labeled, ul),
        role_vocab=make_role_vocab(),
    )

    assert len(ir.atoms) == 1
    atom = ir.atoms[0]
    assert atom.loss_weight == 1.0
    assert atom.provenance["support_provenance"] == ("labeled", "ul")
    assert atom.provenance["merged_support_provenances"] == ("labeled", "ul")
    assert atom.provenance["merged_loss_weights"] == (0.25, 1.0)


def test_event_to_ir_merges_same_logit_atoms_with_union_valid_set() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    first = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101), make_text_action(201)),
    )
    second = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101),),
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(first, second),
        role_vocab=make_role_vocab(),
    )

    assert len(ir.atoms) == 1
    atom = ir.atoms[0]
    assert atom.selected_token_id == 101
    assert atom.valid_token_ids == frozenset({101, 201})
    assert atom.latent_valid_token_ids == frozenset({101, 201})


def test_event_to_ir_honors_explicit_selected_action_over_live_token() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101), make_text_action(201)),
        selected_action=make_text_action(201),
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    assert len(ir.atoms) == 1
    atom = ir.atoms[0]
    assert atom.selected_token_id == 201
    assert atom.valid_token_ids == frozenset({101, 201})
    assert atom.provenance["live_token_id"] == 101
    assert atom.provenance["target_token_mismatch"] is True


def test_event_to_ir_does_not_add_same_role_live_token_to_oracle_valid_set() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(201),),
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    assert len(ir.atoms) == 1
    atom = ir.atoms[0]
    assert atom.selected_token_id == 201
    assert atom.selected_token_role is TokenRole.TEXT
    assert atom.valid_token_ids == frozenset({201})
    assert atom.latent_valid_token_ids == frozenset({201})
    assert atom.provenance["live_token_id"] == 101
    assert atom.provenance["target_token_mismatch"] is True


def test_event_to_ir_uses_oracle_role_when_live_role_disagrees() -> None:
    input_ids = torch.tensor([[11, 22, coord_token(7), 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(201),),
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    assert len(ir.atoms) == 1
    atom = ir.atoms[0]
    assert atom.selected_token_id == 201
    assert atom.selected_token_role is TokenRole.TEXT
    assert atom.allowed_token_roles == frozenset({TokenRole.TEXT})
    assert atom.valid_token_ids == frozenset({201})
    assert atom.latent_valid_token_ids == frozenset({201})
    assert atom.provenance["live_token_id"] == coord_token(7)
    assert atom.provenance["target_token_mismatch"] is True


def test_event_to_ir_rejects_selected_action_not_matching_valid_action_member() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101, loss_weight=1.0),),
        selected_action=make_text_action(101, loss_weight=9.0),
    )

    with pytest.raises(ValueError, match="selected_action must be one member"):
        build_residual_set_target_ir(
            input_ids=input_ids,
            batch_index=0,
            events=(event,),
            role_vocab=make_role_vocab(),
        )


def test_event_to_ir_preserves_rollout_attempt_sequence_identity() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(make_text_action(101), make_text_action(201)),
        draft_metadata={"rollout_id": "r0", "rollout_index": 3},
        event_metadata={"rollout_id": "r0", "rollout_index": 3},
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    assert ir.metadata["sequence_scope"] == "rollout_attempt"
    assert ir.metadata["sequence_id"] == "r0"
    assert ir.metadata["rollout_id"] == "r0"
    assert ir.metadata["rollout_index"] == 3
    assert ir.metadata["sample_id"] == "sample-1"
    assert ir.atoms[0].provenance["rollout_id"] == "r0"
    assert ir.atoms[0].provenance["rollout_index"] == 3


def test_selected_path_metadata_preserves_valid_action_next_state() -> None:
    selected_object = make_object("a", "person_left", x1=120, source="ul")
    state = make_state_for_objects(selected_object)
    actions = state.valid_actions_at_boundary()
    original = only_action(actions)

    updated_actions = _with_selected_path_metadata(
        actions=actions,
        selected_token_id=int(original.token_id),
        selected_object=selected_object,
    )

    updated = only_action(updated_actions)
    assert updated.next_state == original.next_state
    assert updated.selected_object_id == "a"
    assert updated.metadata["loss_weight"] == selected_object.loss_weight
    assert updated.metadata["support_provenance"] == ("ul",)


def test_stop_draft_to_ir_uses_configured_stop_token() -> None:
    input_ids = torch.tensor([[11, 22, 999]])
    stop_action = ValidAction(
        token_id=999,
        token_role=TokenRole.STOP,
        token_text="<|im_end|>",
        candidate_ids_after=frozenset(),
    )
    event = make_event(
        kind="premature_stop",
        target_position=2,
        logit_position=1,
        valid_actions=(stop_action,),
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(),
    )

    atom = ir.atoms[0]
    assert atom.selected_token_role is TokenRole.STOP
    assert atom.allowed_token_roles == frozenset({TokenRole.STOP})
    assert atom.selected_token_id == 999
    assert atom.valid_token_ids == frozenset({999})
    assert atom.coord_role is None


def test_event_to_ir_rejects_same_role_mixed_coord_roles_in_one_draft() -> None:
    input_ids = torch.tensor([[11, 22, coord_token(120), 102, 103]])
    event = make_event(
        target_position=2,
        logit_position=1,
        valid_actions=(
            make_coord_action(coord_token(120), "x1"),
            make_coord_action(coord_token(20), "y1"),
        ),
    )

    with pytest.raises(ValueError, match="coord_role"):
        build_residual_set_target_ir(
            input_ids=input_ids,
            batch_index=0,
            events=(event,),
            role_vocab=make_role_vocab(),
        )
