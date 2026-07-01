"""Projection from compact-full encoded views to supervised token spans."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from src.detection.template_contracts import is_compact_template_id
from src.training.encoding.view import EncodedDetectionView
from src.training.supervision.spans import SupervisionSpanRole

if TYPE_CHECKING:
    from src.detection.tokenization import TokenSpan
else:
    TokenSpan = object


@dataclass(frozen=True, slots=True)
class CompactCoordinateSlotProjection:
    """Projected coordinate slot intersected with supervised label positions.

    :param object_instance_id: Stable source object instance identifier.
    :param object_index: Normalized object index in rendered order.
    :param slot_index: Coordinate slot index within the object box.
    :param slot_name: Coordinate slot name.
    :param token_span: Source token span for this compact coordinate slot.
    :param label_positions: Supervised target-token positions in this slot.
    """

    object_instance_id: str
    object_index: int
    slot_index: int
    slot_name: str
    token_span: TokenSpan
    label_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CompactObjectProjection:
    """Projected compact object entry with supervised semantic sub-spans.

    :param object_instance_id: Stable source object instance identifier.
    :param object_index: Normalized object index in rendered order.
    :param source_object_index: Source object index before normalization.
    :param entry_positions: Supervised positions inside the object entry span.
    :param description_positions: Supervised positions inside description text.
    :param schema_positions: Supervised positions inside structural schema tokens.
    :param coordinate_slots: Projected coordinate slots for this object.
    """

    object_instance_id: str
    object_index: int
    source_object_index: int
    entry_positions: tuple[int, ...]
    description_positions: tuple[int, ...]
    schema_positions: tuple[int, ...]
    coordinate_slots: tuple[CompactCoordinateSlotProjection, ...]


@dataclass(frozen=True, slots=True)
class CompactSpanProjection:
    """Compact-full projection carrying supervised target-token positions only.

    :param template_id: Detection template identifier from the encoded view.
    :param template_version: Detection template version from the encoded view.
    :param objects: Object-level compact projections in rendered order.
    :param schema_positions: Supervised structural schema target positions.
    :param description_positions: Supervised description target positions.
    :param coordinate_positions: Supervised coordinate target positions.
    :param stop_positions: Supervised compact terminal or assistant-stop targets.
    :param label_positions: All supervised target-token positions from the view.
    """

    template_id: str
    template_version: int
    objects: tuple[CompactObjectProjection, ...]
    schema_positions: tuple[int, ...]
    description_positions: tuple[int, ...]
    coordinate_positions: tuple[int, ...]
    stop_positions: tuple[int, ...]
    label_positions: tuple[int, ...]

    def contains_label_position(self, label_position: int) -> bool:
        """Return whether *label_position* is supervised by this projection."""

        return label_position in self.label_positions

    def is_coordinate_position(self, label_position: int) -> bool:
        """Return whether *label_position* belongs to a coordinate slot."""

        return label_position in self.coordinate_positions

    def role_for_label_position(
        self,
        label_position: int,
    ) -> SupervisionSpanRole | None:
        """Return the supervision role for a projected label position."""

        # prioritize semantic sub-spans over broad object-entry membership.
        if label_position in self.coordinate_positions:
            return "coordinate"
        if label_position in self.description_positions:
            return "free_text"
        if label_position in self.schema_positions:
            return "schema"
        if label_position in self.stop_positions:
            return "object_boundary"
        if any(label_position in obj.entry_positions for obj in self.objects):
            return "object_boundary"

        return None


class CompactFullSpanProjector:
    """Project semantic compact encoded views into supervised span positions."""

    def project(self, view: EncodedDetectionView) -> CompactSpanProjection:
        """Return a compact projection for *view*.

        :param view: Encoded detection view owned by the template/tokenizer layer.
        :returns: Label-position-only compact span projection.
        :raises TypeError: If *view* is not an ``EncodedDetectionView``.
        :raises ValueError: If the view was not rendered by a compact template.
        """

        # validate template ownership.
        if type(view) is not EncodedDetectionView:
            raise TypeError("compact span projection requires EncodedDetectionView")
        if not is_compact_template_id(view.template_id):
            raise ValueError("compact span projection requires a compact template_id")

        # project global semantic regions.
        label_positions = tuple(view.label_positions)
        schema_positions = _positions_for_spans(label_positions, view.schema_spans)
        description_positions = _positions_for_spans(
            label_positions,
            view.description_spans,
        )
        coordinate_positions = _positions_for_spans(
            label_positions,
            (slot.token_span for slot in view.coordinate_slots),
        )
        stop_positions = _terminal_positions(view, label_positions)

        # project object-local semantic regions.
        objects: list[CompactObjectProjection] = []
        for entry in view.object_entries:
            object_slots = tuple(
                CompactCoordinateSlotProjection(
                    object_instance_id=slot.object_instance_id,
                    object_index=slot.object_index,
                    slot_index=slot.slot_index,
                    slot_name=slot.slot_name,
                    token_span=slot.token_span,
                    label_positions=_positions_for_spans(
                        label_positions,
                        (slot.token_span,),
                    ),
                )
                for slot in view.coordinate_slots
                if slot.object_instance_id == entry.object_instance_id
                and slot.object_index == entry.object_index
            )
            objects.append(
                CompactObjectProjection(
                    object_instance_id=entry.object_instance_id,
                    object_index=entry.object_index,
                    source_object_index=entry.source_object_index,
                    entry_positions=_positions_for_spans(
                        label_positions,
                        (entry.entry_span,),
                    ),
                    description_positions=_positions_for_spans(
                        label_positions,
                        (entry.description_span,),
                    ),
                    schema_positions=_positions_for_spans(
                        label_positions,
                        entry.schema_spans,
                    ),
                    coordinate_slots=object_slots,
                )
            )

        return CompactSpanProjection(
            template_id=view.template_id,
            template_version=view.template_version,
            objects=tuple(objects),
            schema_positions=schema_positions,
            description_positions=description_positions,
            coordinate_positions=coordinate_positions,
            stop_positions=stop_positions,
            label_positions=label_positions,
        )


def require_projected_role(
    projection: CompactSpanProjection,
    label_position: int,
) -> SupervisionSpanRole:
    """Return a concrete role for a compact projected label position."""

    role = projection.role_for_label_position(label_position)
    if role is None:
        raise ValueError("label position is not in a compact supervision span")

    return cast(SupervisionSpanRole, role)


def _positions_for_spans(
    label_positions: Sequence[int],
    spans: Iterable[TokenSpan],
) -> tuple[int, ...]:
    """Return label positions intersecting any token span."""

    # collect candidate token indices from the compact view spans.
    span_positions: set[int] = set()
    for span in spans:
        span_positions.update(int(position) for position in span.token_indices())

    # preserve the authoritative supervised-label order from the encoded view.
    return tuple(position for position in label_positions if position in span_positions)


def _terminal_positions(
    view: EncodedDetectionView,
    label_positions: Sequence[int],
) -> tuple[int, ...]:
    """Return supervised terminal positions from stop spans and token roles."""

    # collect assistant stop marker positions when the tokenizer owner exposes them.
    terminal_positions: set[int] = set()
    if view.assistant_stop_token_span is not None:
        terminal_positions.update(view.assistant_stop_token_span.token_indices())

    # include explicit terminal roles without importing the detection token enum.
    for position, role in enumerate(view.token_roles):
        if getattr(role, "value", role) == "terminal":
            terminal_positions.add(position)

    return tuple(position for position in label_positions if position in terminal_positions)
