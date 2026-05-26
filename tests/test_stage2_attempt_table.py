from src.trainers.rollout_correction.attempt_table import (
    build_attempt_table,
    wrap_rollout_attempt_view,
)


def test_rollout_attempt_preserves_existing_rollout_id_without_copying_dict() -> None:
    view = {"parse": object(), "rollout_id": "s0:r1"}

    wrapped = wrap_rollout_attempt_view(
        sample_id="s0",
        sample_index=0,
        role="primary_attempt",
        source="live",
        rollout_index=1,
        view=view,
        rollout_result=([1, 2, 3], "abc", "sampling", [9]),
    )

    assert wrapped.view is view
    assert view["rollout_index"] == 1
    assert view["rollout_role"] == "primary_attempt"
    assert view["rollout_id"] == "s0:r1"
    assert wrapped.explicit_rollout_id == "s0:r1"
    assert wrapped.sequence_id == "s0:r1"
    assert wrapped.residual_event_identity() == {
        "rollout_index": 1,
        "rollout_id": "s0:r1",
    }


def test_live_rollout_attempt_keeps_rollout_id_out_of_view_and_identity() -> None:
    view = {"parse": object()}

    wrapped = wrap_rollout_attempt_view(
        sample_id="s0",
        sample_index=0,
        role="primary_attempt",
        source="live",
        rollout_index=2,
        view=view,
        rollout_result=([1, 2], "ab", "sampling", [9]),
    )

    assert wrapped.view is view
    assert view["rollout_index"] == 2
    assert view["rollout_role"] == "primary_attempt"
    assert "rollout_id" not in view
    assert wrapped.explicit_rollout_id is None
    assert wrapped.sequence_id == "s0:rollout_index=2"
    assert wrapped.residual_event_identity() == {"rollout_index": 2}


def test_attempt_table_preserves_ul_view_order() -> None:
    current = wrap_rollout_attempt_view(
        sample_id="s0",
        sample_index=0,
        role="primary_attempt",
        source="live",
        rollout_index=0,
        view={"name": "current"},
    )
    peer0 = wrap_rollout_attempt_view(
        sample_id="s0",
        sample_index=0,
        role="peer_attempt",
        source="live",
        rollout_index=1,
        view={"name": "peer0"},
    )
    peer1 = wrap_rollout_attempt_view(
        sample_id="s0",
        sample_index=0,
        role="peer_attempt",
        source="live",
        rollout_index=2,
        view={"name": "peer1"},
    )

    table = build_attempt_table(
        sample_id="s0",
        sample_index=0,
        primary=current,
        peers=(peer0, peer1),
    )

    assert [view["name"] for view in table.views_for_ul] == [
        "current",
        "peer0",
        "peer1",
    ]
