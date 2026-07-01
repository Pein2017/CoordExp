"""Canonical token-wise supervision records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from src.common.errors import LossContractError
from src.packing.planner import PackedSegment, PackedSequence
from src.packing.supervision import PackedSupervision, PackedTokenAtom


DEFAULT_IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenAtom:
    pack_index: int
    segment_index: int
    example_index: int
    example_id: str
    target_position: int
    token_id: int
    token_type: str
    text: str
    logical_target_position: int
    logical_target_end: int | None = None
    logical_base_token_start: int | None = None
    logical_base_token_end: int | None = None
    object_id: str | None = None
    field: str | None = None
    source: str | None = None

    @property
    def target_end(self) -> int:
        return self.target_position + 1

    @property
    def causal_logits_position(self) -> int:
        return self.target_position - 1

    def to_artifact_dict(self) -> dict[str, int | str | None]:
        return {
            "pack_index": self.pack_index,
            "segment_index": self.segment_index,
            "example_index": self.example_index,
            "example_id": self.example_id,
            "target_position": self.target_position,
            "target_end": self.target_end,
            "causal_logits_position": self.causal_logits_position,
            "token_id": self.token_id,
            "token_type": self.token_type,
            "text": self.text,
            "logical_target_position": self.logical_target_position,
            "logical_target_end": self.logical_target_end,
            "logical_base_token_start": self.logical_base_token_start,
            "logical_base_token_end": self.logical_base_token_end,
            "object_id": self.object_id,
            "field": self.field,
            "source": self.source,
        }

    @classmethod
    def from_packed(cls, atom: PackedTokenAtom) -> TokenAtom:
        return cls(
            pack_index=atom.pack_index,
            segment_index=atom.segment_index,
            example_index=atom.example_index,
            example_id=atom.example_id,
            target_position=atom.target_position,
            token_id=atom.token_id,
            token_type=atom.token_type,
            text=atom.text,
            logical_target_position=atom.logical_target_position,
            logical_target_end=atom.logical_target_end,
            logical_base_token_start=atom.logical_base_token_start,
            logical_base_token_end=atom.logical_base_token_end,
            object_id=atom.object_id,
            field=atom.field,
            source=atom.source,
        )


@dataclass(frozen=True)
class TokenSpan:
    pack_index: int
    segment_index: int
    example_index: int
    example_id: str
    token_type: str
    atoms: tuple[TokenAtom, ...]
    text: str
    logical_base_token_start: int | None = None
    logical_base_token_end: int | None = None
    object_id: str | None = None
    field: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.atoms:
            raise LossContractError(
                "TokenSpan must contain at least one TokenAtom",
                code="supervision.empty_span",
                context={
                    "pack_index": self.pack_index,
                    "segment_index": self.segment_index,
                    "example_id": self.example_id,
                },
            )
        for atom in self.atoms:
            if (
                atom.pack_index != self.pack_index
                or atom.segment_index != self.segment_index
                or atom.example_index != self.example_index
                or atom.example_id != self.example_id
                or atom.token_type != self.token_type
                or atom.logical_base_token_start != self.logical_base_token_start
                or atom.logical_base_token_end != self.logical_base_token_end
                or atom.object_id != self.object_id
                or atom.field != self.field
                or atom.source != self.source
                or atom.text != self.text
            ):
                raise LossContractError(
                    "TokenSpan atoms must share span identity",
                    code="supervision.span_identity",
                    context={
                        "span_pack_index": self.pack_index,
                        "span_segment_index": self.segment_index,
                        "span_example_index": self.example_index,
                        "atom_pack_index": atom.pack_index,
                        "atom_segment_index": atom.segment_index,
                        "atom_example_index": atom.example_index,
                        "span_example_id": self.example_id,
                        "atom_example_id": atom.example_id,
                        "span_object_id": self.object_id,
                        "atom_object_id": atom.object_id,
                        "span_field": self.field,
                        "atom_field": atom.field,
                        "span_source": self.source,
                        "atom_source": atom.source,
                    },
                )

    @property
    def target_start(self) -> int:
        return self.atoms[0].target_position

    @property
    def target_end(self) -> int:
        return self.atoms[-1].target_end

    @property
    def target_range(self) -> tuple[int, int]:
        return (self.target_start, self.target_end)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "pack_index": self.pack_index,
            "segment_index": self.segment_index,
            "example_index": self.example_index,
            "example_id": self.example_id,
            "token_type": self.token_type,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "token_count": len(self.atoms),
            "text": self.text,
            "logical_base_token_start": self.logical_base_token_start,
            "logical_base_token_end": self.logical_base_token_end,
            "object_id": self.object_id,
            "field": self.field,
            "source": self.source,
            "atoms": [atom.to_artifact_dict() for atom in self.atoms],
        }


@dataclass(frozen=True)
class TokenSequence:
    pack_index: int
    input_ids: tuple[int, ...]
    segments: tuple[PackedSegment, ...]
    atoms: tuple[TokenAtom, ...]
    spans: tuple[TokenSpan, ...]

    def __post_init__(self) -> None:
        _validate_segments(self.pack_index, self.pack_length, self.segments)
        _validate_atoms(self)

    @property
    def pack_length(self) -> int:
        return len(self.input_ids)

    def to_dense_labels(self, *, ignore_index: int = DEFAULT_IGNORE_INDEX) -> tuple[int, ...]:
        return dense_labels_from_token_sequence(self, ignore_index=ignore_index)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "pack_index": self.pack_index,
            "pack_length": self.pack_length,
            "segment_count": len(self.segments),
            "atom_count": len(self.atoms),
            "span_count": len(self.spans),
            "segments": [segment.to_artifact_dict() for segment in self.segments],
            "atoms": [atom.to_artifact_dict() for atom in self.atoms],
            "spans": [span.to_artifact_dict() for span in self.spans],
        }


def build_token_sequence_from_packed_supervision(
    pack: PackedSequence,
    supervision: PackedSupervision | Sequence[TokenAtom] | Sequence[PackedTokenAtom],
) -> TokenSequence:
    if not isinstance(pack, PackedSequence):
        raise LossContractError(
            "TokenSequence construction requires a PackedSequence",
            code="supervision.pack_type",
            context={"value_type": type(pack).__name__},
        )
    atoms = _coerce_atoms(supervision, pack_index=pack.pack_index)
    spans = _build_spans(atoms)
    return TokenSequence(
        pack_index=pack.pack_index,
        input_ids=pack.input_ids,
        segments=pack.segments,
        atoms=atoms,
        spans=spans,
    )


def dense_labels_from_token_sequence(
    sequence: TokenSequence,
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> tuple[int, ...]:
    if not isinstance(sequence, TokenSequence):
        raise LossContractError(
            "dense label materialization requires a TokenSequence",
            code="supervision.sequence_type",
            context={"value_type": type(sequence).__name__},
        )
    labels = [int(ignore_index)] * sequence.pack_length
    for atom in sequence.atoms:
        labels[atom.target_position] = atom.token_id
    return tuple(labels)


def validate_dense_labels_match_token_sequence(
    sequence: TokenSequence,
    dense_labels: Sequence[int],
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> None:
    expected = dense_labels_from_token_sequence(sequence, ignore_index=ignore_index)
    observed = tuple(int(label) for label in dense_labels)
    if observed != expected:
        mismatches = [
            {
                "position": index,
                "expected": expected_value,
                "observed": observed_value,
            }
            for index, (expected_value, observed_value) in enumerate(zip(expected, observed))
            if expected_value != observed_value
        ]
        if len(observed) != len(expected):
            mismatches.append(
                {
                    "position": -1,
                    "expected": len(expected),
                    "observed": len(observed),
                }
            )
        raise LossContractError(
            "dense labels must be reproducible from TokenSequence",
            code="supervision.dense_label_mismatch",
            context={"mismatches": mismatches[:8]},
        )


def _coerce_atoms(
    supervision: PackedSupervision | Sequence[TokenAtom] | Sequence[PackedTokenAtom],
    *,
    pack_index: int,
) -> tuple[TokenAtom, ...]:
    raw_atoms: Iterable[Any]
    filter_by_pack = isinstance(supervision, PackedSupervision)
    if isinstance(supervision, PackedSupervision):
        raw_atoms = supervision.atoms
    else:
        raw_atoms = supervision

    atoms: list[TokenAtom] = []
    for atom in raw_atoms:
        if isinstance(atom, PackedTokenAtom):
            token_atom = TokenAtom.from_packed(atom)
        elif isinstance(atom, TokenAtom):
            token_atom = atom
        else:
            raise LossContractError(
                "TokenSequence atoms must be TokenAtom or PackedTokenAtom records",
                code="supervision.atom_type",
                context={"value_type": type(atom).__name__},
            )
        if token_atom.pack_index == pack_index:
            atoms.append(token_atom)
        elif filter_by_pack:
            continue
        else:
            raise LossContractError(
                "explicit TokenSequence atoms must match the requested pack_index",
                code="supervision.atom_pack_index",
                context={
                    "sequence_pack_index": pack_index,
                    "atom_pack_index": token_atom.pack_index,
                    "target_position": token_atom.target_position,
                    "example_id": token_atom.example_id,
                },
            )
    return tuple(sorted(atoms, key=lambda item: item.target_position))


def _build_spans(atoms: tuple[TokenAtom, ...]) -> tuple[TokenSpan, ...]:
    spans: list[TokenSpan] = []
    current: list[TokenAtom] = []
    current_key: tuple[Any, ...] | None = None
    for atom in atoms:
        key = _span_key(atom)
        adjacent = bool(current and atom.target_position == current[-1].target_end)
        if current and (key != current_key or not adjacent):
            spans.append(_span_from_atoms(tuple(current)))
            current = []
        current.append(atom)
        current_key = key
    if current:
        spans.append(_span_from_atoms(tuple(current)))
    return tuple(spans)


def _span_key(atom: TokenAtom) -> tuple[Any, ...]:
    return (
        atom.pack_index,
        atom.segment_index,
        atom.example_id,
        atom.token_type,
        atom.logical_base_token_start,
        atom.logical_base_token_end,
        atom.object_id,
        atom.field,
        atom.source,
        atom.text,
    )


def _span_from_atoms(atoms: tuple[TokenAtom, ...]) -> TokenSpan:
    first = atoms[0]
    return TokenSpan(
        pack_index=first.pack_index,
        segment_index=first.segment_index,
        example_index=first.example_index,
        example_id=first.example_id,
        token_type=first.token_type,
        atoms=atoms,
        text=first.text,
        logical_base_token_start=first.logical_base_token_start,
        logical_base_token_end=first.logical_base_token_end,
        object_id=first.object_id,
        field=first.field,
        source=first.source,
    )


def _validate_segments(
    pack_index: int,
    pack_length: int,
    segments: tuple[PackedSegment, ...],
) -> None:
    expected_start = 0
    for expected_index, segment in enumerate(segments):
        if segment.pack_index != pack_index:
            raise LossContractError(
                "TokenSequence segment pack_index mismatch",
                code="supervision.segment_pack_index",
                context={
                    "sequence_pack_index": pack_index,
                    "segment_pack_index": segment.pack_index,
                },
            )
        if segment.segment_index != expected_index:
            raise LossContractError(
                "TokenSequence segment_index values must be sequential",
                code="supervision.segment_index_sequence",
                context={
                    "pack_index": pack_index,
                    "expected_segment_index": expected_index,
                    "observed_segment_index": segment.segment_index,
                    "known_segment_indices": [
                        known.segment_index for known in segments
                    ],
                },
            )
        if segment.start != expected_start or segment.end <= segment.start:
            raise LossContractError(
                "TokenSequence segments must be contiguous",
                code="supervision.segment_boundaries",
                context={
                    "segment_index": segment.segment_index,
                    "start": segment.start,
                    "end": segment.end,
                    "expected_start": expected_start,
                },
            )
        expected_start = segment.end
    if segments and expected_start != pack_length:
        raise LossContractError(
            "TokenSequence segments must end at pack length",
            code="supervision.segment_boundaries",
            context={"pack_length": pack_length, "last_segment_end": expected_start},
        )


def _validate_atoms(sequence: TokenSequence) -> None:
    segments_by_index = {segment.segment_index: segment for segment in sequence.segments}
    seen_targets: set[int] = set()
    for atom in sequence.atoms:
        if atom.target_position in seen_targets:
            raise LossContractError(
                "TokenSequence target positions must be unique",
                code="supervision.duplicate_target_position",
                context={
                    "pack_index": atom.pack_index,
                    "target_position": atom.target_position,
                },
            )
        seen_targets.add(atom.target_position)
        _validate_atom_identity(sequence, atom, segments_by_index)
        _validate_causal_positions(atom, segments_by_index[atom.segment_index])


def _validate_atom_identity(
    sequence: TokenSequence,
    atom: TokenAtom,
    segments_by_index: dict[int, PackedSegment],
) -> None:
    if atom.pack_index != sequence.pack_index:
        raise LossContractError(
            "TokenAtom pack_index must match its TokenSequence",
            code="supervision.atom_pack_index",
            context={
                "sequence_pack_index": sequence.pack_index,
                "atom_pack_index": atom.pack_index,
            },
        )
    segment = segments_by_index.get(atom.segment_index)
    if segment is None:
        raise LossContractError(
            "TokenAtom segment_index must exist in its TokenSequence",
            code="supervision.atom_segment_index",
            context={
                "pack_index": atom.pack_index,
                "segment_index": atom.segment_index,
                "known_segments": sorted(segments_by_index),
            },
        )
    if atom.example_id != segment.example_id or atom.example_index != segment.example_index:
        raise LossContractError(
            "TokenAtom example identity must match its PackedSegment",
            code="supervision.atom_segment_identity",
            context={
                "segment_example_id": segment.example_id,
                "atom_example_id": atom.example_id,
                "segment_example_index": segment.example_index,
                "atom_example_index": atom.example_index,
            },
        )


def _validate_causal_positions(atom: TokenAtom, segment: PackedSegment) -> None:
    if atom.target_position == 0:
        raise LossContractError(
            "causal token supervision cannot target physical position 0",
            code="supervision.causal_target_position",
            context={
                "pack_index": atom.pack_index,
                "segment_index": atom.segment_index,
                "target_position": atom.target_position,
            },
        )
    if not segment.start <= atom.target_position < segment.end:
        raise LossContractError(
            "TokenAtom target_position must stay inside its segment",
            code="supervision.target_out_of_segment",
            context={
                "pack_index": atom.pack_index,
                "segment_index": atom.segment_index,
                "target_position": atom.target_position,
                "segment_start": segment.start,
                "segment_end": segment.end,
            },
        )
    if not segment.start <= atom.causal_logits_position < segment.end:
        raise LossContractError(
            "causal logits position must stay inside the same packed segment",
            code="supervision.logits_cross_segment_boundary",
            context={
                "pack_index": atom.pack_index,
                "segment_index": atom.segment_index,
                "target_position": atom.target_position,
                "logits_position": atom.causal_logits_position,
                "segment_start": segment.start,
                "segment_end": segment.end,
            },
        )


__all__ = [
    "DEFAULT_IGNORE_INDEX",
    "TokenAtom",
    "TokenSequence",
    "TokenSpan",
    "build_token_sequence_from_packed_supervision",
    "dense_labels_from_token_sequence",
    "validate_dense_labels_match_token_sequence",
]
