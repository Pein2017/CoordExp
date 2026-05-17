"""Authoritative tokenized detection encoding views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from src.detection.tokenization import (
        TokenRole,
        TokenSpan,
        TokenizedDetectionExample,
        TokenizedObjectEntry,
    )
else:
    TokenRole = Any
    TokenSpan = Any
    TokenizedDetectionExample = Any
    TokenizedObjectEntry = Any

CoordinateSlotName: TypeAlias = Literal["x1", "y1", "x2", "y2"]

_COORDINATE_SLOT_NAMES: tuple[CoordinateSlotName, ...] = ("x1", "y1", "x2", "y2")


def _token_span_type() -> type:
    """Return the detection owner's token span type."""

    from src.detection.tokenization import TokenSpan

    return TokenSpan


def _token_role_type() -> type:
    """Return the detection owner's token-role enum type."""

    from src.detection.tokenization import TokenRole

    return TokenRole


def _require_plain_str(value: object, *, field_name: str) -> None:
    """Validate an exact string field."""

    if type(value) is not str:
        raise TypeError(f"{field_name} must be a plain string")


def _require_nonnegative_int(value: object, *, field_name: str) -> None:
    """Validate an exact nonnegative integer field."""

    if type(value) is not int:
        raise TypeError(f"{field_name} must be a plain integer")
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")


def _require_exact_token_span(value: object, *, field_name: str) -> None:
    """Validate a token span from the detection owner exactly."""

    if type(value) is not _token_span_type():
        raise TypeError(f"{field_name} must be TokenSpan")


def _require_optional_token_span(value: object, *, field_name: str) -> None:
    """Validate an optional token span from the detection owner exactly."""

    if value is not None:
        _require_exact_token_span(value, field_name=field_name)


def _as_int_tuple(values: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    """Return a defensive immutable integer tuple.

    :param values: Candidate integer sequence.
    :param field_name: Field name for error reporting.
    :returns: Immutable tuple of plain integers.
    :raises TypeError: If an entry is not a plain integer.
    """

    _require_sequence_container(values, field_name=field_name)

    normalized: list[int] = []
    for value in values:
        if type(value) is not int:
            raise TypeError(f"{field_name} must contain plain integers")
        normalized.append(value)

    return tuple(normalized)


def _as_token_span_tuple(
    values: Sequence["TokenSpan"],
    *,
    field_name: str,
) -> tuple["TokenSpan", ...]:
    """Return a defensive tuple of exact detection token spans."""

    _require_sequence_container(values, field_name=field_name)

    normalized = tuple(values)
    for span in normalized:
        _require_exact_token_span(span, field_name=field_name)

    return normalized


def _require_sequence_container(value: object, *, field_name: str) -> None:
    """Validate a non-scalar sequence container before tuple conversion."""

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")


def _as_offset_tuple(
    values: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return a defensive immutable offset tuple.

    :param values: Candidate offset sequence.
    :returns: Immutable tuple of ``(start, end)`` offsets.
    :raises TypeError: If an offset boundary is not a plain integer.
    :raises ValueError: If an offset is inverted.
    """

    normalized: list[tuple[int, int]] = []
    for start, end in values:
        if type(start) is not int or type(end) is not int:
            raise TypeError("offset boundaries must be plain integers")
        if end < start:
            raise ValueError("offset end must be greater than or equal to start")
        normalized.append((start, end))

    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class EncodedObjectEntry:
    """Token-level object entry projected from the detection tokenization owner.

    :param object_instance_id: Stable source object instance identifier.
    :param object_index: Normalized object index in rendered order.
    :param source_object_index: Source object index before normalization.
    :param entry_span: Token span covering the rendered object entry.
    :param description_span: Token span covering the object description.
    :param coordinate_spans: Coordinate token spans ordered as x1, y1, x2, y2.
    :param schema_spans: Structural token spans associated with this entry.
    """

    object_instance_id: str
    object_index: int
    source_object_index: int
    entry_span: TokenSpan
    description_span: TokenSpan
    coordinate_spans: tuple[TokenSpan, ...]
    schema_spans: tuple[TokenSpan, ...]

    def __post_init__(self) -> None:
        """Validate exact object-entry scalar and span contracts."""

        _require_plain_str(
            self.object_instance_id,
            field_name="object_instance_id",
        )
        _require_nonnegative_int(self.object_index, field_name="object_index")
        _require_nonnegative_int(
            self.source_object_index,
            field_name="source_object_index",
        )
        _require_exact_token_span(self.entry_span, field_name="entry_span")
        _require_exact_token_span(
            self.description_span,
            field_name="description_span",
        )

        coordinate_spans = _as_token_span_tuple(
            self.coordinate_spans,
            field_name="coordinate_spans",
        )
        if len(coordinate_spans) != len(_COORDINATE_SLOT_NAMES):
            raise ValueError("coordinate_spans must contain exactly four spans")
        object.__setattr__(self, "coordinate_spans", coordinate_spans)
        object.__setattr__(
            self,
            "schema_spans",
            _as_token_span_tuple(self.schema_spans, field_name="schema_spans"),
        )

    @classmethod
    def from_tokenized_entry(
        cls,
        entry: TokenizedObjectEntry,
    ) -> "EncodedObjectEntry":
        """Build an encoded object entry from the existing tokenized owner."""

        return cls(
            object_instance_id=entry.object_instance_id,
            object_index=entry.object_index,
            source_object_index=entry.source_object_index,
            entry_span=entry.entry_span,
            description_span=entry.desc_span,
            coordinate_spans=tuple(entry.coord_spans),
            schema_spans=tuple(entry.control_spans),
        )


@dataclass(frozen=True, slots=True)
class CoordinateSlot:
    """Named coordinate token slot in the encoded assistant sequence.

    :param object_instance_id: Stable source object instance identifier.
    :param object_index: Normalized object index in rendered order.
    :param slot_index: Coordinate slot index within the object box.
    :param slot_name: Coordinate slot name.
    :param token_span: Token span covering the coordinate token.
    """

    object_instance_id: str
    object_index: int
    slot_index: int
    slot_name: CoordinateSlotName
    token_span: TokenSpan

    def __post_init__(self) -> None:
        """Validate exact coordinate-slot scalar and span contracts."""

        _require_plain_str(
            self.object_instance_id,
            field_name="object_instance_id",
        )
        _require_nonnegative_int(self.object_index, field_name="object_index")
        _require_nonnegative_int(self.slot_index, field_name="slot_index")
        if self.slot_index >= len(_COORDINATE_SLOT_NAMES):
            raise ValueError("slot_index must identify one of x1/y1/x2/y2")
        if type(self.slot_name) is not str:
            raise TypeError("slot_name must be a plain string")
        if self.slot_name not in _COORDINATE_SLOT_NAMES:
            raise ValueError("slot_name must be one of x1/y1/x2/y2")
        _require_exact_token_span(self.token_span, field_name="token_span")


@dataclass(frozen=True, slots=True)
class EncodedDetectionView:
    """Authoritative tokenized view for detection training examples.

    This view is intentionally limited to rendered/tokenized contract state. It
    does not carry objective, loss, assignment, duplicate-filtering, or model
    output state; those concerns belong to later bridge/runner layers.

    :param template_id: Detection template family that rendered the assistant.
    :param template_version: Template version reported by the detection owner.
    :param input_ids: Token ids for the full rendered chat.
    :param labels: Teacher-token labels for causal training.
    :param offset_mapping: Token offsets in the full rendered chat.
    :param assistant_token_span: Token span covering the assistant payload.
    :param assistant_stop_token_span: Optional token span for the chat stop marker.
    :param object_entries: Token-level object entries in rendered order.
    :param schema_spans: Structural schema spans projected to token space.
    :param description_spans: Description spans projected to token space.
    :param coordinate_slots: Named coordinate token slots in object order.
    :param token_roles: Per-token semantic roles from the detection tokenizer.
    :param label_positions: Target-token positions with supervised labels.
    :param rendered_assistant_text: Optional diagnostics/provenance text.
    """

    template_id: str
    template_version: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    assistant_token_span: TokenSpan
    assistant_stop_token_span: TokenSpan | None
    object_entries: tuple[EncodedObjectEntry, ...]
    schema_spans: tuple[TokenSpan, ...]
    description_spans: tuple[TokenSpan, ...]
    coordinate_slots: tuple[CoordinateSlot, ...]
    token_roles: tuple[TokenRole, ...]
    label_positions: tuple[int, ...]
    rendered_assistant_text: str | None = None

    def __post_init__(self) -> None:
        """Validate immutable local shape without importing tensor libraries."""

        _require_plain_str(self.template_id, field_name="template_id")
        _require_nonnegative_int(
            self.template_version,
            field_name="template_version",
        )
        _require_exact_token_span(
            self.assistant_token_span,
            field_name="assistant_token_span",
        )
        _require_optional_token_span(
            self.assistant_stop_token_span,
            field_name="assistant_stop_token_span",
        )
        if self.rendered_assistant_text is not None and type(
            self.rendered_assistant_text
        ) is not str:
            raise TypeError("rendered_assistant_text must be a string or None")

        object.__setattr__(
            self,
            "input_ids",
            _as_int_tuple(self.input_ids, field_name="input_ids"),
        )
        object.__setattr__(
            self,
            "labels",
            _as_int_tuple(self.labels, field_name="labels"),
        )
        object.__setattr__(
            self,
            "offset_mapping",
            _as_offset_tuple(self.offset_mapping),
        )
        _require_sequence_container(self.object_entries, field_name="object_entries")
        object_entries = tuple(self.object_entries)
        for entry in object_entries:
            if type(entry) is not EncodedObjectEntry:
                raise TypeError("object_entries must contain EncodedObjectEntry")
        object.__setattr__(self, "object_entries", object_entries)
        object.__setattr__(
            self,
            "schema_spans",
            _as_token_span_tuple(self.schema_spans, field_name="schema_spans"),
        )
        object.__setattr__(
            self,
            "description_spans",
            _as_token_span_tuple(
                self.description_spans,
                field_name="description_spans",
            ),
        )
        _require_sequence_container(
            self.coordinate_slots,
            field_name="coordinate_slots",
        )
        coordinate_slots = tuple(self.coordinate_slots)
        for slot in coordinate_slots:
            if type(slot) is not CoordinateSlot:
                raise TypeError("coordinate_slots must contain CoordinateSlot")
        object.__setattr__(self, "coordinate_slots", coordinate_slots)
        token_role_type = _token_role_type()
        _require_sequence_container(self.token_roles, field_name="token_roles")
        token_roles = tuple(self.token_roles)
        for role in token_roles:
            if type(role) is not token_role_type:
                raise TypeError("token_roles must contain TokenRole")
        object.__setattr__(self, "token_roles", token_roles)
        object.__setattr__(
            self,
            "label_positions",
            _as_int_tuple(self.label_positions, field_name="label_positions"),
        )

        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have the same length")
        if len(self.input_ids) != len(self.offset_mapping):
            raise ValueError("input_ids and offset_mapping must have the same length")
        if len(self.input_ids) != len(self.token_roles):
            raise ValueError("input_ids and token_roles must have the same length")
        for position in self.label_positions:
            if position <= 0 or position >= len(self.labels):
                raise ValueError("label_positions must satisfy 0 < pos < len(labels)")
            if self.labels[position] == -100:
                raise ValueError("label_positions must reference supervised labels")
        if len(self.coordinate_slots) != len(self.object_entries) * len(
            _COORDINATE_SLOT_NAMES
        ):
            raise ValueError("each object requires exactly four coordinate slots")
        for object_offset, entry in enumerate(self.object_entries):
            if len(entry.coordinate_spans) != len(_COORDINATE_SLOT_NAMES):
                raise ValueError("each object requires exactly four coordinate spans")
            slots = self.coordinate_slots[
                object_offset * len(_COORDINATE_SLOT_NAMES) : (
                    object_offset + 1
                )
                * len(_COORDINATE_SLOT_NAMES)
            ]
            for slot_index, (expected_name, span, slot) in enumerate(
                zip(_COORDINATE_SLOT_NAMES, entry.coordinate_spans, slots, strict=True)
            ):
                if slot.object_instance_id != entry.object_instance_id:
                    raise ValueError("coordinate slot object ids must match entries")
                if slot.object_index != entry.object_index:
                    raise ValueError("coordinate slot object indexes must match entries")
                if slot.slot_index != slot_index:
                    raise ValueError("coordinate slots must be ordered by slot index")
                if slot.slot_name != expected_name:
                    raise ValueError("coordinate slots must be ordered x1/y1/x2/y2")
                if slot.token_span != span:
                    raise ValueError("coordinate slots must match object coordinate spans")

    @classmethod
    def from_tokenized(
        cls,
        tokenized: TokenizedDetectionExample,
        *,
        include_rendered_assistant_text: bool = True,
    ) -> "EncodedDetectionView":
        """Project the detection tokenizer owner into the training view."""

        object_entries = tuple(
            EncodedObjectEntry.from_tokenized_entry(entry)
            for entry in tokenized.object_entries
        )

        coordinate_slots: list[CoordinateSlot] = []
        for entry in object_entries:
            if len(entry.coordinate_spans) != len(_COORDINATE_SLOT_NAMES):
                raise ValueError("encoded object entries require four coordinate slots")
            for slot_index, token_span in enumerate(entry.coordinate_spans):
                coordinate_slots.append(
                    CoordinateSlot(
                        object_instance_id=entry.object_instance_id,
                        object_index=entry.object_index,
                        slot_index=slot_index,
                        slot_name=_COORDINATE_SLOT_NAMES[slot_index],
                        token_span=token_span,
                    )
                )

        return cls(
            template_id=tokenized.rendered_assistant.template_id,
            template_version=tokenized.rendered_assistant.template_version,
            input_ids=tokenized.input_ids,
            labels=tokenized.labels,
            offset_mapping=tokenized.offset_mapping,
            assistant_token_span=tokenized.assistant_token_span,
            assistant_stop_token_span=tokenized.assistant_stop_token_span,
            object_entries=object_entries,
            schema_spans=tokenized.structural_spans,
            description_spans=tuple(entry.desc_span for entry in tokenized.object_entries),
            coordinate_slots=tuple(coordinate_slots),
            token_roles=tokenized.token_roles,
            label_positions=tokenized.supervised_label_positions,
            rendered_assistant_text=(
                tokenized.rendered_assistant.text
                if include_rendered_assistant_text
                else None
            ),
        )


__all__ = [
    "CoordinateSlot",
    "CoordinateSlotName",
    "EncodedDetectionView",
    "EncodedObjectEntry",
]
