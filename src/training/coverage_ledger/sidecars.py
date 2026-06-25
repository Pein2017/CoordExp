"""Immutable sidecar payloads for coverage-ledger supervision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from src.training.coverage_ledger.geometry import validate_norm1000_bbox_xyxy
from src.training.teacher_forcing.packing_offsets import PackedSegmentOffset


def _require_plain_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _require_non_negative_int(value: object, *, field_name: str) -> int:
    parsed = _require_plain_int(value, field_name=field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _require_positive_int(value: object, *, field_name: str) -> int:
    parsed = _require_plain_int(value, field_name=field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be strictly positive")
    return parsed


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_optional_non_empty_str(
    value: object | None,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _require_non_empty_str(value, field_name=field_name)


def _require_optional_non_negative_int(
    value: object | None,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, field_name=field_name)


def _freeze_int_tuple(
    values: Sequence[object],
    *,
    field_name: str,
    expected_len: int,
    positive: bool = False,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    frozen = tuple(values)
    if len(frozen) != expected_len:
        raise ValueError(f"{field_name} must contain exactly {expected_len} integers")
    validator = _require_positive_int if positive else _require_non_negative_int
    return tuple(
        validator(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(frozen)
    )


@dataclass(frozen=True, slots=True)
class CoverageLedgerObjectEntry:
    object_instance_id: str
    source_object_index: int
    emitted_order_index: int
    image_index: int
    bbox_norm1000_xyxy: tuple[int, int, int, int]
    box_start_position: int
    coord_label_positions: tuple[int, int, int, int]
    object_ref_end_position: int
    box_end_position: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_instance_id",
            _require_non_empty_str(
                self.object_instance_id,
                field_name="object_instance_id",
            ),
        )
        object.__setattr__(
            self,
            "source_object_index",
            _require_non_negative_int(
                self.source_object_index,
                field_name="source_object_index",
            ),
        )
        object.__setattr__(
            self,
            "emitted_order_index",
            _require_non_negative_int(
                self.emitted_order_index,
                field_name="emitted_order_index",
            ),
        )
        object.__setattr__(
            self,
            "image_index",
            _require_non_negative_int(self.image_index, field_name="image_index"),
        )
        if self.image_index != 0:
            raise ValueError("image_index must be 0 for coverage ledger v0")
        object.__setattr__(
            self,
            "bbox_norm1000_xyxy",
            _validate_bbox(self.bbox_norm1000_xyxy),
        )
        object.__setattr__(
            self,
            "box_start_position",
            _require_non_negative_int(
                self.box_start_position,
                field_name="box_start_position",
            ),
        )
        object.__setattr__(
            self,
            "coord_label_positions",
            _freeze_int_tuple(
                self.coord_label_positions,
                field_name="coord_label_positions",
                expected_len=4,
            ),
        )
        object.__setattr__(
            self,
            "object_ref_end_position",
            _require_non_negative_int(
                self.object_ref_end_position,
                field_name="object_ref_end_position",
            ),
        )
        object.__setattr__(
            self,
            "box_end_position",
            _require_non_negative_int(
                self.box_end_position,
                field_name="box_end_position",
            ),
        )


@dataclass(frozen=True, slots=True)
class CoverageLedgerSidecar:
    sample_id: str
    prompt_end_position: int
    object_entries: tuple[CoverageLedgerObjectEntry, ...]
    image_grid_thw: tuple[int, int, int]
    processed_width: int
    processed_height: int
    image_identity: str
    packed_source_sample_id: str | None = None
    packed_row_index: int | None = None
    packed_segment_index: int | None = None
    packed_token_start: int | None = None
    packed_token_end: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            _require_non_empty_str(self.sample_id, field_name="sample_id"),
        )
        object.__setattr__(
            self,
            "image_identity",
            _require_non_empty_str(self.image_identity, field_name="image_identity"),
        )
        object.__setattr__(
            self,
            "prompt_end_position",
            _require_non_negative_int(
                self.prompt_end_position,
                field_name="prompt_end_position",
            ),
        )
        object.__setattr__(
            self,
            "processed_width",
            _require_positive_int(
                self.processed_width,
                field_name="processed_width",
            ),
        )
        object.__setattr__(
            self,
            "processed_height",
            _require_positive_int(
                self.processed_height,
                field_name="processed_height",
            ),
        )
        object.__setattr__(
            self,
            "image_grid_thw",
            _freeze_int_tuple(
                self.image_grid_thw,
                field_name="image_grid_thw",
                expected_len=3,
                positive=True,
            ),
        )
        object_entries = _freeze_object_entries(self.object_entries)
        _validate_sidecar_object_entries(object_entries)
        object.__setattr__(self, "object_entries", object_entries)
        object.__setattr__(
            self,
            "packed_source_sample_id",
            _require_optional_non_empty_str(
                self.packed_source_sample_id,
                field_name="packed_source_sample_id",
            ),
        )
        object.__setattr__(
            self,
            "packed_row_index",
            _require_optional_non_negative_int(
                self.packed_row_index,
                field_name="packed_row_index",
            ),
        )
        object.__setattr__(
            self,
            "packed_segment_index",
            _require_optional_non_negative_int(
                self.packed_segment_index,
                field_name="packed_segment_index",
            ),
        )
        object.__setattr__(
            self,
            "packed_token_start",
            _require_optional_non_negative_int(
                self.packed_token_start,
                field_name="packed_token_start",
            ),
        )
        object.__setattr__(
            self,
            "packed_token_end",
            _require_optional_non_negative_int(
                self.packed_token_end,
                field_name="packed_token_end",
            ),
        )
        _validate_packed_metadata(self)


def _validate_bbox(value: Sequence[object]) -> tuple[int, int, int, int]:
    return validate_norm1000_bbox_xyxy(value)


def _freeze_object_entries(
    entries: Sequence[CoverageLedgerObjectEntry],
) -> tuple[CoverageLedgerObjectEntry, ...]:
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("object_entries must be a sequence")
    frozen = tuple(entries)
    if not frozen:
        raise ValueError("object_entries must be non-empty")
    for index, entry in enumerate(frozen):
        if type(entry) is not CoverageLedgerObjectEntry:
            raise TypeError(
                "object_entries must contain CoverageLedgerObjectEntry values; "
                f"got {type(entry).__name__} at index {index}"
            )
    return frozen


def _validate_sidecar_object_entries(
    entries: tuple[CoverageLedgerObjectEntry, ...],
) -> None:
    ids = [entry.object_instance_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("object_instance_id values must be unique inside a sidecar")

    expected_order = tuple(range(len(entries)))
    actual_order = tuple(entry.emitted_order_index for entry in entries)
    if actual_order != expected_order:
        raise ValueError(
            "emitted_order_index values must be exactly 0..N-1 in object_entries order"
        )


def _validate_packed_metadata(sidecar: CoverageLedgerSidecar) -> None:
    fields = (
        "packed_source_sample_id",
        "packed_row_index",
        "packed_segment_index",
        "packed_token_start",
        "packed_token_end",
    )
    values = tuple(getattr(sidecar, field_name) for field_name in fields)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("packed sidecar metadata fields must be all set or all unset")
    if sidecar.packed_token_end <= sidecar.packed_token_start:  # type: ignore[operator]
        raise ValueError("packed_token_end must be greater than packed_token_start")


def shift_coverage_ledger_sidecar(
    sidecar: CoverageLedgerSidecar,
    offset: PackedSegmentOffset,
) -> CoverageLedgerSidecar:
    """Shift sidecar token positions into a packed row-local span."""

    if type(sidecar) is not CoverageLedgerSidecar:
        raise TypeError("sidecar must be a CoverageLedgerSidecar")
    if type(offset) is not PackedSegmentOffset:
        raise TypeError("offset must be a PackedSegmentOffset")
    if sidecar.sample_id != offset.sample_id:
        raise ValueError(
            "coverage ledger sidecar sample_id must match packed offset sample_id; "
            f"sidecar={sidecar.sample_id!r} offset={offset.sample_id!r}"
        )

    def shift(position: int) -> int:
        return int(position) + offset.token_start

    shifted_entries = tuple(
        replace(
            entry,
            box_start_position=shift(entry.box_start_position),
            coord_label_positions=tuple(
                shift(position) for position in entry.coord_label_positions
            ),
            object_ref_end_position=shift(entry.object_ref_end_position),
            box_end_position=shift(entry.box_end_position),
        )
        for entry in sidecar.object_entries
    )
    return replace(
        sidecar,
        prompt_end_position=shift(sidecar.prompt_end_position),
        object_entries=shifted_entries,
        packed_source_sample_id=offset.sample_id,
        packed_row_index=offset.packed_row_index,
        packed_segment_index=offset.segment_index,
        packed_token_start=offset.token_start,
        packed_token_end=offset.token_end,
    )


__all__ = [
    "CoverageLedgerObjectEntry",
    "CoverageLedgerSidecar",
    "shift_coverage_ledger_sidecar",
]
