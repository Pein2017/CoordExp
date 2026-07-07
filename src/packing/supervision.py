"""Packed supervision position remapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.errors import PackingContractError
from src.coordinate_targets import (
    CoordinateLossTarget,
    coordinate_target_to_artifact,
)
from src.packing.planner import PackedSequence, PackedSegment


@dataclass(frozen=True)
class PackedTokenAtom:
    pack_index: int
    segment_index: int
    example_index: int
    example_id: str
    token_type: str
    token_id: int
    text: str
    logical_base_token_start: int
    logical_base_token_end: int
    logical_target_position: int
    logical_target_end: int
    target_position: int
    target_end: int
    logits_position: int
    object_id: str | None
    field: str | None
    source: str | None
    coordinate_target: CoordinateLossTarget | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = {
            "pack_index": self.pack_index,
            "segment_index": self.segment_index,
            "example_index": self.example_index,
            "example_id": self.example_id,
            "token_type": self.token_type,
            "token_id": self.token_id,
            "text": self.text,
            "logical_base_token_start": self.logical_base_token_start,
            "logical_base_token_end": self.logical_base_token_end,
            "logical_target_position": self.logical_target_position,
            "logical_target_end": self.logical_target_end,
            "target_position": self.target_position,
            "target_end": self.target_end,
            "logits_position": self.logits_position,
            "object_id": self.object_id,
            "field": self.field,
            "source": self.source,
        }
        coordinate_target = coordinate_target_to_artifact(self.coordinate_target)
        if coordinate_target is not None:
            payload["coordinate_target"] = coordinate_target
        return payload


@dataclass(frozen=True)
class OmittedPackedTokenAtom:
    pack_index: int
    segment_index: int
    example_index: int
    example_id: str
    token_type: str
    token_id: int
    text: str
    logical_base_token_start: int
    logical_base_token_end: int
    logical_target_position: int
    logical_target_end: int
    target_position: int
    target_end: int
    logits_position: int
    object_id: str | None
    field: str | None
    source: str | None
    reason: str
    coordinate_target: CoordinateLossTarget | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = {
            "pack_index": self.pack_index,
            "segment_index": self.segment_index,
            "example_index": self.example_index,
            "example_id": self.example_id,
            "token_type": self.token_type,
            "token_id": self.token_id,
            "text": self.text,
            "logical_base_token_start": self.logical_base_token_start,
            "logical_base_token_end": self.logical_base_token_end,
            "logical_target_position": self.logical_target_position,
            "logical_target_end": self.logical_target_end,
            "target_position": self.target_position,
            "target_end": self.target_end,
            "logits_position": self.logits_position,
            "object_id": self.object_id,
            "field": self.field,
            "source": self.source,
            "reason": self.reason,
        }
        coordinate_target = coordinate_target_to_artifact(self.coordinate_target)
        if coordinate_target is not None:
            payload["coordinate_target"] = coordinate_target
        return payload


@dataclass(frozen=True)
class PackedSupervision:
    atoms: tuple[PackedTokenAtom, ...]
    omitted_atoms: tuple[OmittedPackedTokenAtom, ...]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "atom_count": len(self.atoms),
            "omitted_atom_count": len(self.omitted_atoms),
            "atoms": [atom.to_artifact_dict() for atom in self.atoms],
            "omitted_atoms": [atom.to_artifact_dict() for atom in self.omitted_atoms],
        }


def build_packed_supervision(
    packs: tuple[PackedSequence, ...] | list[PackedSequence],
    encoded_examples: tuple[Any, ...] | list[Any],
) -> PackedSupervision:
    examples_by_id = _examples_by_id(encoded_examples)
    atoms: list[PackedTokenAtom] = []
    omitted: list[OmittedPackedTokenAtom] = []

    for pack in packs:
        for segment in pack.segments:
            example = examples_by_id.get(segment.example_id)
            if example is None:
                raise PackingContractError(
                    "packed segment references an encoded example that was not provided",
                    code="packing.segment_example_missing",
                    context={
                        "pack_index": pack.pack_index,
                        "segment_index": segment.segment_index,
                        "example_id": segment.example_id,
                    },
                )
            _validate_segment_matches_example(segment, example)
            for span in getattr(example, "supervised_token_spans", ()):
                remapped, omitted_atoms = _remap_span(segment, span)
                atoms.extend(remapped)
                omitted.extend(omitted_atoms)

    return PackedSupervision(atoms=tuple(atoms), omitted_atoms=tuple(omitted))


def _examples_by_id(encoded_examples: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for example in encoded_examples:
        example_id = getattr(example, "example_id", None)
        if not isinstance(example_id, str) or not example_id:
            raise PackingContractError(
                "encoded example must expose a non-empty example_id",
                code="packing.example_id",
                context={"value_type": type(example_id).__name__},
            )
        if example_id in examples:
            raise PackingContractError(
                "encoded example ids must be unique for packed supervision remapping",
                code="packing.duplicate_example_id",
                context={"example_id": example_id},
            )
        examples[example_id] = example
    return examples


def _validate_segment_matches_example(segment: PackedSegment, example: Any) -> None:
    input_length = len(getattr(example, "input_ids", ()))
    if segment.length != input_length:
        raise PackingContractError(
            "packed segment length must match encoded example length",
            code="packing.segment_length_mismatch",
            context={
                "example_id": segment.example_id,
                "segment_length": segment.length,
                "encoded_length": input_length,
            },
        )


def _remap_span(
    segment: PackedSegment,
    span: Any,
) -> tuple[list[PackedTokenAtom], list[OmittedPackedTokenAtom]]:
    token_ids = tuple(int(token_id) for token_id in getattr(span, "token_ids", ()))
    if not token_ids:
        raise PackingContractError(
            "supervised token span must expose at least one token id",
            code="packing.supervision_empty_span",
            context={"example_id": segment.example_id},
        )
    logical_start = int(getattr(span, "physical_token_start"))
    logical_end = int(getattr(span, "physical_token_end"))
    if logical_end <= logical_start or logical_end - logical_start != len(token_ids):
        raise PackingContractError(
            "supervised token span physical range must match token id count",
            code="packing.supervision_span_shape",
            context={
                "example_id": segment.example_id,
                "logical_start": logical_start,
                "logical_end": logical_end,
                "token_count": len(token_ids),
            },
        )
    if logical_start < 0 or logical_end > segment.length:
        raise PackingContractError(
            "supervised token span physical range must stay inside its packed segment",
            code="packing.supervision_span_bounds",
            context={
                "example_id": segment.example_id,
                "segment_index": segment.segment_index,
                "segment_length": segment.length,
                "logical_start": logical_start,
                "logical_end": logical_end,
            },
        )

    atoms: list[PackedTokenAtom] = []
    omitted_atoms: list[OmittedPackedTokenAtom] = []
    token_type = str(getattr(span, "token_type"))
    text = str(getattr(span, "text", ""))
    logical_base_token_start = int(getattr(span, "base_token_start"))
    logical_base_token_end = int(getattr(span, "base_token_end"))
    object_id = getattr(span, "object_id", None)
    field = getattr(span, "field", None)
    source = getattr(span, "source", None)
    coordinate_target = getattr(span, "coordinate_target", None)
    for offset, token_id in enumerate(token_ids):
        logical_target = logical_start + offset
        target_position = segment.start + logical_target
        logits_position = target_position - 1
        if logits_position < segment.start:
            omitted_atoms.append(
                OmittedPackedTokenAtom(
                    pack_index=segment.pack_index,
                    segment_index=segment.segment_index,
                    example_index=segment.example_index,
                    example_id=segment.example_id,
                    token_type=token_type,
                    token_id=token_id,
                    text=text,
                    logical_base_token_start=logical_base_token_start,
                    logical_base_token_end=logical_base_token_end,
                    logical_target_position=logical_target,
                    logical_target_end=logical_target + 1,
                    target_position=target_position,
                    target_end=target_position + 1,
                    logits_position=logits_position,
                    object_id=object_id,
                    field=field,
                    source=source,
                    reason="logits_position_crosses_segment_boundary",
                    coordinate_target=coordinate_target,
                )
            )
            continue
        atoms.append(
            PackedTokenAtom(
                pack_index=segment.pack_index,
                segment_index=segment.segment_index,
                example_index=segment.example_index,
                example_id=segment.example_id,
                token_type=token_type,
                token_id=token_id,
                text=text,
                logical_base_token_start=logical_base_token_start,
                logical_base_token_end=logical_base_token_end,
                logical_target_position=logical_target,
                logical_target_end=logical_target + 1,
                target_position=target_position,
                target_end=target_position + 1,
                logits_position=logits_position,
                object_id=object_id,
                field=field,
                source=source,
                coordinate_target=coordinate_target,
            )
        )
    return atoms, omitted_atoms


__all__ = [
    "OmittedPackedTokenAtom",
    "PackedSupervision",
    "PackedTokenAtom",
    "build_packed_supervision",
]
