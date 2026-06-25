"""Pure helpers for static-packed teacher-forcing position offsets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR


_LOCAL_POSITION_PROVENANCE_KEYS = frozenset(
    {
        "branch_position",
    }
)


@dataclass(frozen=True, slots=True)
class PackedSegmentOffset:
    """Row-local token span for one original sample inside a packed row."""

    sample_id: str
    packed_row_index: int
    segment_index: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            _require_non_empty_str(self.sample_id, field_name="sample_id"),
        )
        object.__setattr__(
            self,
            "packed_row_index",
            _require_non_negative_int(
                self.packed_row_index,
                field_name="packed_row_index",
            ),
        )
        object.__setattr__(
            self,
            "segment_index",
            _require_non_negative_int(self.segment_index, field_name="segment_index"),
        )
        object.__setattr__(
            self,
            "token_start",
            _require_non_negative_int(self.token_start, field_name="token_start"),
        )
        object.__setattr__(
            self,
            "token_end",
            _require_non_negative_int(self.token_end, field_name="token_end"),
        )
        if self.token_end <= self.token_start:
            raise ValueError("token_end must be greater than token_start")


def build_packed_segment_offsets(
    raw_batch: Sequence[Any],
    collated: Mapping[str, Any],
) -> tuple[PackedSegmentOffset, ...]:
    """Return row-local segment offsets for a packed raw batch.

    Non-packed batches return an empty tuple. Packed batches are represented as
    ``raw_batch[row_index][segment_index]`` lists/tuples of sample mappings.
    """

    raw_rows = _require_sequence(raw_batch, field_name="raw_batch")
    if not raw_rows:
        return ()

    packed_flags = tuple(_is_pack_row(row) for row in raw_rows)
    if not any(packed_flags):
        return ()
    if not all(packed_flags):
        raise ValueError("packed raw_batch rows must all be list/tuple pack rows")

    attention_counts = _attention_non_padding_counts(
        collated,
        expected_rows=len(raw_rows),
    )
    offsets: list[PackedSegmentOffset] = []
    seen_sample_ids: set[str] = set()
    for packed_row_index, raw_pack in enumerate(raw_rows):
        pack = _require_pack_row(raw_pack, row_index=packed_row_index)
        token_start = 0
        for segment_index, sample in enumerate(pack):
            if not isinstance(sample, Mapping):
                raise TypeError(
                    "packed raw_batch segments must be mappings; "
                    f"got {type(sample).__name__} at row {packed_row_index} "
                    f"segment {segment_index}"
                )
            sample_id = _sample_id(sample, packed_row_index, segment_index)
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id in packed raw_batch: {sample_id!r}")
            seen_sample_ids.add(sample_id)

            length = _sample_length(sample, packed_row_index, segment_index)
            token_end = token_start + length
            offsets.append(
                PackedSegmentOffset(
                    sample_id=sample_id,
                    packed_row_index=packed_row_index,
                    segment_index=segment_index,
                    token_start=token_start,
                    token_end=token_end,
                )
            )
            token_start = token_end

        if token_start != attention_counts[packed_row_index]:
            raise ValueError(
                "packed raw_batch segment lengths must match attention_mask "
                "non-padding length; "
                f"row={packed_row_index} segment_total={token_start} "
                f"attention_mask_length={attention_counts[packed_row_index]}"
            )

    return tuple(offsets)


def shift_teacher_forcing_target_ir(
    ir: TeacherForcingTargetIR,
    offset: PackedSegmentOffset,
) -> TeacherForcingTargetIR:
    """Shift a single-sample target IR into its packed row-local positions."""

    if type(ir) is not TeacherForcingTargetIR:
        raise TypeError("ir must be a TeacherForcingTargetIR")
    if type(offset) is not PackedSegmentOffset:
        raise TypeError("offset must be a PackedSegmentOffset")

    shifted_atoms = tuple(
        _shift_atom(atom, offset=offset)
        for atom in ir.atoms
    )
    return replace(ir, atoms=shifted_atoms)


def _shift_atom(
    atom: SupervisionAtom,
    *,
    offset: PackedSegmentOffset,
) -> SupervisionAtom:
    if type(atom) is not SupervisionAtom:
        raise TypeError("TeacherForcingTargetIR atoms must be SupervisionAtom values")
    return replace(
        atom,
        batch_index=offset.packed_row_index,
        logit_position=int(atom.logit_position) + offset.token_start,
        target_position=int(atom.target_position) + offset.token_start,
        provenance=_shift_provenance_positions(
            atom.provenance,
            token_start=offset.token_start,
        ),
    )


def _shift_provenance_positions(
    provenance: Mapping[str, Any],
    *,
    token_start: int,
) -> dict[str, Any]:
    shifted: dict[str, Any] = {}
    for key, value in provenance.items():
        if _should_shift_provenance_field(key):
            shifted[key] = _shift_position_value(
                value,
                token_start=token_start,
                field_name=f"provenance[{key!r}]",
            )
        else:
            shifted[key] = value
    return shifted


def _should_shift_provenance_field(key: str) -> bool:
    if key in _LOCAL_POSITION_PROVENANCE_KEYS:
        return False
    if key.endswith("_token_id") or key.endswith("_token_ids"):
        return False
    if key in {"position", "positions", "logit_position", "target_position"}:
        return True
    return key.endswith("_position") or key.endswith("_positions")


def _shift_position_value(
    value: Any,
    *,
    token_start: int,
    field_name: str,
) -> Any:
    if _is_plain_int(value):
        return int(value) + token_start
    if isinstance(value, tuple):
        return tuple(
            _shift_position_scalar(
                item,
                token_start=token_start,
                field_name=f"{field_name}[{index}]",
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _shift_position_scalar(
                item,
                token_start=token_start,
                field_name=f"{field_name}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, frozenset):
        return frozenset(
            _shift_position_scalar(item, token_start=token_start, field_name=field_name)
            for item in value
        )
    raise TypeError(f"{field_name} must contain integer token positions")


def _shift_position_scalar(
    value: Any,
    *,
    token_start: int,
    field_name: str,
) -> int:
    if not _is_plain_int(value):
        raise TypeError(f"{field_name} must be an integer token position")
    return int(value) + token_start


def _attention_non_padding_counts(
    collated: Mapping[str, Any],
    *,
    expected_rows: int,
) -> tuple[int, ...]:
    if not isinstance(collated, Mapping):
        raise TypeError("collated must be a mapping")
    if "attention_mask" not in collated:
        raise ValueError("packed offset construction requires collated attention_mask")
    rows = _to_nested_rows(collated["attention_mask"], field_name="attention_mask")
    if len(rows) != expected_rows:
        raise ValueError(
            "collated attention_mask row count must match packed raw_batch row count; "
            f"attention_mask={len(rows)} raw_batch={expected_rows}"
        )
    counts: list[int] = []
    for row_index, row in enumerate(rows):
        count = 0
        for col_index, value in enumerate(row):
            if not _is_plain_int(value):
                raise TypeError(
                    "attention_mask values must be integers; "
                    f"got {type(value).__name__} at row {row_index} col {col_index}"
                )
            if int(value) != 0:
                count += 1
        counts.append(count)
    return tuple(counts)


def _to_nested_rows(value: Any, *, field_name: str) -> tuple[tuple[Any, ...], ...]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if _is_sequence(value):
        frozen = tuple(value)
        if not frozen:
            return ()
        if all(_is_sequence(row) for row in frozen):
            return tuple(tuple(row) for row in frozen)
        return (frozen,)
    raise TypeError(f"{field_name} must be a sequence")


def _sample_id(
    sample: Mapping[str, Any],
    packed_row_index: int,
    segment_index: int,
) -> str:
    if "sample_id" not in sample:
        raise ValueError(
            "packed raw_batch segment missing sample_id; "
            f"row={packed_row_index} segment={segment_index}"
        )
    sample_id = str(sample["sample_id"])
    if not sample_id:
        raise ValueError(
            "packed raw_batch segment sample_id must be non-empty; "
            f"row={packed_row_index} segment={segment_index}"
        )
    return sample_id


def _sample_length(
    sample: Mapping[str, Any],
    packed_row_index: int,
    segment_index: int,
) -> int:
    if "length" in sample:
        return _require_positive_int(
            sample["length"],
            field_name=f"raw_batch[{packed_row_index}][{segment_index}].length",
        )
    input_ids = sample.get("input_ids")
    if input_ids is None:
        raise ValueError(
            "packed raw_batch segment missing length/input_ids; "
            f"row={packed_row_index} segment={segment_index}"
        )
    length = _sequence_length(
        input_ids,
        field_name=f"raw_batch[{packed_row_index}][{segment_index}].input_ids",
    )
    if length <= 0:
        raise ValueError(
            "packed raw_batch segment input_ids must be non-empty; "
            f"row={packed_row_index} segment={segment_index}"
        )
    return length


def _sequence_length(value: Any, *, field_name: str) -> int:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if _is_sequence(value):
        return len(value)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[-1])
    raise TypeError(f"{field_name} must be a sequence")


def _require_pack_row(value: Any, *, row_index: int) -> tuple[Any, ...]:
    if not _is_pack_row(value):
        raise TypeError(f"raw_batch[{row_index}] must be a list/tuple packed row")
    pack = tuple(value)
    if not pack:
        raise ValueError(f"raw_batch[{row_index}] packed row must be non-empty")
    return pack


def _require_sequence(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if not _is_sequence(value):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(value)


def _is_pack_row(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _require_non_empty_str(value: Any, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_positive_int(value: Any, *, field_name: str) -> int:
    parsed = _require_non_negative_int(value, field_name=field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be strictly positive")
    return parsed


def _require_non_negative_int(value: Any, *, field_name: str) -> int:
    if not _is_plain_int(value):
        raise TypeError(f"{field_name} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "PackedSegmentOffset",
    "build_packed_segment_offsets",
    "shift_teacher_forcing_target_ir",
]
