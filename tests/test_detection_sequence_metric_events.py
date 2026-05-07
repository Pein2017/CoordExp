import pytest

from src.metrics.detection_sequence import (
    VALID_COORDINATE_SLOT_NAMES,
    coordinate_slot_accuracy_event,
    coordinate_token_accuracy_event,
    description_token_accuracy_event,
    object_entry_exact_match_event,
    parser_success_event,
    schema_token_accuracy_event,
)
from src.metrics.events import reduce_metric_events


def test_coordinate_slot_helpers_use_slot_denominators() -> None:
    events = [
        coordinate_slot_accuracy_event("x1", 1, 1),
        coordinate_slot_accuracy_event("x1", 1, 3),
        coordinate_slot_accuracy_event("y1", 0, 2),
        coordinate_slot_accuracy_event("x2", 2, 2),
        coordinate_slot_accuracy_event("y2", 3, 4),
    ]

    reduced = reduce_metric_events(events)

    assert {event.slot_name for event in events} == set(VALID_COORDINATE_SLOT_NAMES)
    for event in events:
        assert event.unit == "slot"
        assert event.semantic_role == "bbox_coord"
        assert event.token_role == "coord"
        assert event.coordinate_surface == "coord_token"
        assert event.geometry_type == "bbox"
        assert event.aliases == ()
    assert reduced[
        "detection_sequence/coordinate/slot_acc/bbox/coord_token/x1"
    ] == pytest.approx(2 / 4)


def test_invalid_coordinate_slot_name_raises_clearly() -> None:
    with pytest.raises(ValueError, match="Invalid coordinate slot name.*cx.*x1.*y1.*x2.*y2"):
        coordinate_slot_accuracy_event("cx", 0, 1)


def test_object_entry_exact_match_uses_object_denominator() -> None:
    event = object_entry_exact_match_event(3, 5)
    reduced = reduce_metric_events([event])

    assert event.unit == "object"
    assert event.denominator == 5
    assert event.semantic_role == "object_entry"
    assert event.object_scope == "object_entry"
    assert event.aliases == ()
    assert reduced[event.key] == pytest.approx(3 / 5)


def test_parser_success_uses_generated_sample_denominator_and_parser_fields() -> None:
    event = parser_success_event(
        7,
        10,
        parser_mode="strict_expected",
        metric_surface="strict_expected_template",
    )
    reduced = reduce_metric_events([event])

    assert event.unit == "sample"
    assert event.denominator == 10
    assert event.semantic_role == "parser"
    assert event.parser_mode == "strict_expected"
    assert event.metric_surface == "strict_expected_template"
    assert event.diagnostic_only is False
    assert event.aliases == ()
    assert reduced[event.key] == pytest.approx(7 / 10)


def test_diagnostic_parser_surface_participates_in_event_identity() -> None:
    strict_event = parser_success_event(
        7,
        10,
        parser_mode="strict_expected",
        metric_surface="strict_expected_template",
    )
    diagnostic_event = parser_success_event(
        2,
        10,
        parser_mode="diagnostic_salvage",
        metric_surface="diagnostic_only",
        diagnostic_only=True,
    )

    assert strict_event.key != diagnostic_event.key
    assert strict_event.identity != diagnostic_event.identity
    assert diagnostic_event.unit == "sample"
    assert diagnostic_event.diagnostic_only is True


def test_token_helpers_are_explicitly_token_denominated() -> None:
    events = [
        schema_token_accuracy_event(8, 10),
        description_token_accuracy_event(6, 10, top_k=5),
        coordinate_token_accuracy_event(4, 10, vocab_scope="full_vocab"),
    ]

    for event in events:
        assert event.unit == "token"
        assert event.denominator == 10
        assert event.slot_name is None
        assert event.object_scope is None
        assert event.aliases == ()

    assert events[0].semantic_role == "schema"
    assert events[0].token_role == "schema"
    assert events[1].semantic_role == "description"
    assert events[1].token_role == "description"
    assert events[1].key.endswith("/top5")
    assert events[2].semantic_role == "coordinate"
    assert events[2].token_role == "coord"
    assert events[2].coordinate_surface == "coord_token"
