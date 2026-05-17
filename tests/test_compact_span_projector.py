from __future__ import annotations

from dataclasses import replace

import pytest

from src.detection.tokenization import TokenRole, TokenSpan
from src.training.encoding.view import (
    CoordinateSlot,
    EncodedDetectionView,
    EncodedObjectEntry,
)
from src.training.span_adapters.compact_projector import CompactFullSpanProjector


def _span(start: int, end: int, label: str) -> TokenSpan:
    return TokenSpan(start=start, end=end, label=label)


def _compact_view() -> EncodedDetectionView:
    entry = EncodedObjectEntry(
        object_instance_id="obj-1",
        object_index=0,
        source_object_index=7,
        entry_span=_span(2, 12, "entry"),
        description_span=_span(3, 5, "desc"),
        coordinate_spans=(
            _span(6, 7, "x1"),
            _span(7, 8, "y1"),
            _span(8, 9, "x2"),
            _span(9, 10, "y2"),
        ),
        schema_spans=(_span(2, 3, "object_ref"), _span(5, 6, "box_start")),
    )
    coordinate_slots = tuple(
        CoordinateSlot(
            object_instance_id="obj-1",
            object_index=0,
            slot_index=index,
            slot_name=slot_name,
            token_span=entry.coordinate_spans[index],
        )
        for index, slot_name in enumerate(("x1", "y1", "x2", "y2"))
    )
    token_roles = (
        TokenRole.IGNORE,
        TokenRole.ASSISTANT,
        TokenRole.CONTROL,
        TokenRole.DESC,
        TokenRole.DESC,
        TokenRole.BBOX_START,
        TokenRole.COORD,
        TokenRole.COORD,
        TokenRole.COORD,
        TokenRole.COORD,
        TokenRole.SEPARATOR,
        TokenRole.TERMINAL,
    )

    return EncodedDetectionView(
        template_id="compact_full",
        template_version=1,
        input_ids=tuple(range(100, 112)),
        labels=(-100, 101, 102, 103, -100, 105, 106, 107, 108, 109, -100, 111),
        offset_mapping=tuple((index, index + 1) for index in range(12)),
        assistant_token_span=_span(1, 12, "assistant"),
        assistant_stop_token_span=_span(11, 12, "assistant_stop"),
        object_entries=(entry,),
        schema_spans=(_span(2, 3, "object_ref"), _span(5, 6, "box_start")),
        description_spans=(_span(3, 5, "desc"),),
        coordinate_slots=coordinate_slots,
        token_roles=token_roles,
        label_positions=(1, 2, 3, 5, 6, 7, 8, 9, 11),
        rendered_assistant_text=None,
    )


def _compact_view_with_unsupervised_coordinate() -> EncodedDetectionView:
    view = _compact_view()
    labels = tuple(
        -100 if position == 8 else label
        for position, label in enumerate(view.labels)
    )

    return replace(
        view,
        labels=labels,
        label_positions=tuple(
            position for position in view.label_positions if position != 8
        ),
    )


def test_projector_intersects_compact_spans_with_label_positions_only() -> None:
    projection = CompactFullSpanProjector().project(_compact_view())

    assert projection.template_id == "compact_full"
    assert projection.label_positions == (1, 2, 3, 5, 6, 7, 8, 9, 11)
    assert projection.schema_positions == (2, 5)
    assert projection.description_positions == (3,)
    assert projection.coordinate_positions == (6, 7, 8, 9)
    assert projection.stop_positions == (11,)

    projected_object = projection.objects[0]
    assert projected_object.object_instance_id == "obj-1"
    assert projected_object.object_index == 0
    assert projected_object.source_object_index == 7
    assert projected_object.entry_positions == (2, 3, 5, 6, 7, 8, 9, 11)
    assert projected_object.description_positions == (3,)
    assert projected_object.schema_positions == (2, 5)

    assert [
        (slot.slot_name, slot.label_positions)
        for slot in projected_object.coordinate_slots
    ] == [
        ("x1", (6,)),
        ("y1", (7,)),
        ("x2", (8,)),
        ("y2", (9,)),
    ]


def test_projector_excludes_unsupervised_coordinate_position() -> None:
    projection = CompactFullSpanProjector().project(
        _compact_view_with_unsupervised_coordinate()
    )

    assert projection.label_positions == (1, 2, 3, 5, 6, 7, 9, 11)
    assert projection.coordinate_positions == (6, 7, 9)

    projected_object = projection.objects[0]
    assert [
        (slot.slot_name, slot.label_positions)
        for slot in projected_object.coordinate_slots
    ] == [
        ("x1", (6,)),
        ("y1", (7,)),
        ("x2", ()),
        ("y2", (9,)),
    ]


def test_projector_uses_terminal_role_when_stop_span_is_absent() -> None:
    view = replace(_compact_view(), assistant_stop_token_span=None)

    projection = CompactFullSpanProjector().project(view)

    assert projection.stop_positions == (11,)


def test_projector_rejects_non_compact_full_template() -> None:
    with pytest.raises(ValueError, match="compact_full"):
        CompactFullSpanProjector().project(replace(_compact_view(), template_id="unit"))
