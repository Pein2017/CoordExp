from __future__ import annotations

import pytest

from src.training.ordering import (
    LegacyTailAppendOrdering,
    PlanningObject,
    TopLeftSpatialOrdering,
    resolve_object_ordering_strategy,
)


def test_legacy_tail_append_ordering_preserves_accepted_then_false_negative_order() -> None:
    accepted = (
        PlanningObject(object_id="accepted-late", bbox=(50.0, 50.0, 60.0, 60.0)),
        PlanningObject(object_id="accepted-early", bbox=(0.0, 0.0, 10.0, 10.0)),
    )
    false_negatives = (
        PlanningObject(object_id="fn-middle", bbox=(20.0, 20.0, 30.0, 30.0)),
        PlanningObject(object_id="fn-top", bbox=(5.0, 5.0, 15.0, 15.0)),
    )

    ordered = LegacyTailAppendOrdering().order(
        accepted_objects=accepted,
        false_negative_objects=false_negatives,
    )

    assert [item.object_id for item in ordered] == [
        "accepted-late",
        "accepted-early",
        "fn-middle",
        "fn-top",
    ]


def test_top_left_spatial_ordering_sorts_across_accepted_and_false_negatives() -> None:
    accepted = (
        PlanningObject(object_id="accepted-bottom", bbox=(0.0, 90.0, 10.0, 100.0)),
        PlanningObject(object_id="accepted-top-right", bbox=(50.0, 0.0, 60.0, 10.0)),
    )
    false_negatives = (
        PlanningObject(object_id="fn-top-left", bbox=(10.0, 0.0, 20.0, 10.0)),
        PlanningObject(object_id="fn-middle", bbox=(0.0, 30.0, 10.0, 40.0)),
    )

    ordered = TopLeftSpatialOrdering().order(
        accepted_objects=accepted,
        false_negative_objects=false_negatives,
    )

    assert [item.object_id for item in ordered] == [
        "fn-top-left",
        "accepted-top-right",
        "fn-middle",
        "accepted-bottom",
    ]


def test_resolve_object_ordering_strategy_rejects_invalid_mode() -> None:
    assert isinstance(resolve_object_ordering_strategy("tail_append"), LegacyTailAppendOrdering)
    assert isinstance(resolve_object_ordering_strategy("tail_append_legacy"), LegacyTailAppendOrdering)
    assert isinstance(resolve_object_ordering_strategy("sorted"), TopLeftSpatialOrdering)

    with pytest.raises(ValueError, match="unsupported object ordering"):
        resolve_object_ordering_strategy("random")
