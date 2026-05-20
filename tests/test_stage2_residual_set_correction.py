from __future__ import annotations

import pytest

from src.trainers.stage2_two_channel.residual_set import (
    CorrectionAtomDraft,
    CorrectionEvent,
    CorrectionKind,
    CoordRole,
    ResidualObject,
    ResidualState,
    ValidAction,
    enumerate_valid_actions,
    transition_state,
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


def apply_action(state: ResidualState, action: ValidAction) -> ResidualState:
    return transition_state(state, action)


def make_role_vocab() -> RoleVocab:
    return RoleVocab(
        schema_token_ids=frozenset({1}),
        text_token_ids=frozenset(ord(ch) for ch in "person_left_rightcar"),
        coord_token_ids=frozenset(coord_token(value) for value in range(0, 1001)),
        stop_token_id=999,
    )


def make_event(
    *,
    kind: CorrectionKind = "transition_failure",
    target_position: int = 8,
    logit_position: int = 7,
    valid_actions: tuple[ValidAction, ...],
) -> CorrectionEvent:
    draft = CorrectionAtomDraft(
        correction_kind=kind,
        target_position=target_position,
        logit_position=logit_position,
        valid_actions=valid_actions,
        metadata={"source": "unit-test"},
    )
    return CorrectionEvent(
        correction_kind=kind,
        sample_id="sample-1",
        atom_drafts=(draft,),
        metadata={"adapter": "not-yet"},
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


def test_x1_ambiguity_filters_candidates_by_exact_coord_token() -> None:
    state = make_state_for_objects(
        make_object("a", "person_left", x1=120),
        make_object("b", "person_right", x1=640),
        make_object("c", "car", x1=850),
    )

    actions = valid_coord_actions(state, "x1")

    assert {action.token_id for action in actions} == {coord_token(120), coord_token(640), coord_token(850)}
    chosen = only_action(tuple(action for action in actions if action.token_id == coord_token(640)))
    narrowed = apply_action(state, chosen)
    assert narrowed.active_candidate_ids == frozenset({"b"})


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


def test_empty_non_stop_transition_is_guarded() -> None:
    state = make_state_for_objects(make_object("a", "person_left", x1=120))
    bad_action = ValidAction(
        token_id=123,
        token_role=TokenRole.TEXT,
        token_text="bad",
        candidate_ids_after=frozenset({"missing"}),
    )

    with pytest.raises(ValueError, match="empty active candidates"):
        transition_state(state, bad_action)


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
