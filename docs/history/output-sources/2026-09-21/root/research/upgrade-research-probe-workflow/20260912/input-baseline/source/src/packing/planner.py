"""No-padding pack planning for encoded examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.errors import PackingContractError


@dataclass(frozen=True)
class PackedSegment:
    pack_index: int
    segment_index: int
    example_index: int
    example_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_artifact_dict(self) -> dict[str, int | str]:
        return {
            "pack_index": self.pack_index,
            "segment_index": self.segment_index,
            "example_index": self.example_index,
            "example_id": self.example_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


@dataclass(frozen=True)
class PackedSequence:
    pack_index: int
    input_ids: tuple[int, ...]
    segments: tuple[PackedSegment, ...]
    global_max_length: int

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "pack_index": self.pack_index,
            "length": self.length,
            "global_max_length": self.global_max_length,
            "padding_tokens": 0,
            "segment_count": len(self.segments),
            "segments": [segment.to_artifact_dict() for segment in self.segments],
        }


def plan_packed_sequences(
    encoded_examples: list[Any] | tuple[Any, ...],
    *,
    global_max_length: int,
) -> tuple[PackedSequence, ...]:
    if global_max_length <= 0:
        raise PackingContractError(
            "packing.global_max_length must be positive",
            code="packing.global_max_length",
            context={"global_max_length": global_max_length},
        )

    packs: list[PackedSequence] = []
    current_ids: list[int] = []
    current_segments: list[PackedSegment] = []
    pack_index = 0

    for example_index, example in enumerate(encoded_examples):
        example_id = _example_id(example)
        input_ids = _input_ids(example, example_id=example_id)
        input_length = len(input_ids)
        if input_length == 0:
            raise PackingContractError(
                "encoded example must contain at least one token before packing",
                code="packing.example_empty",
                context={"example_id": example_id, "example_index": example_index},
            )
        if input_length > global_max_length:
            raise PackingContractError(
                "encoded example exceeds packing.global_max_length",
                code="packing.example_too_long",
                context={
                    "example_id": example_id,
                    "example_index": example_index,
                    "input_length": input_length,
                    "global_max_length": global_max_length,
                },
            )

        if current_ids and len(current_ids) + input_length > global_max_length:
            packs.append(
                _commit_pack(
                    pack_index=pack_index,
                    input_ids=current_ids,
                    segments=current_segments,
                    global_max_length=global_max_length,
                )
            )
            pack_index += 1
            current_ids = []
            current_segments = []

        start = len(current_ids)
        current_ids.extend(input_ids)
        current_segments.append(
            PackedSegment(
                pack_index=pack_index,
                segment_index=len(current_segments),
                example_index=example_index,
                example_id=example_id,
                start=start,
                end=start + input_length,
            )
        )

    if current_ids:
        packs.append(
            _commit_pack(
                pack_index=pack_index,
                input_ids=current_ids,
                segments=current_segments,
                global_max_length=global_max_length,
            )
        )
    return tuple(packs)


def _commit_pack(
    *,
    pack_index: int,
    input_ids: list[int],
    segments: list[PackedSegment],
    global_max_length: int,
) -> PackedSequence:
    if not segments:
        raise PackingContractError(
            "cannot commit an empty pack",
            code="packing.empty_pack",
            context={"pack_index": pack_index},
        )
    if len(input_ids) > global_max_length:
        raise PackingContractError(
            "packed sequence exceeds packing.global_max_length",
            code="packing.pack_too_long",
            context={
                "pack_index": pack_index,
                "length": len(input_ids),
                "global_max_length": global_max_length,
            },
        )
    return PackedSequence(
        pack_index=pack_index,
        input_ids=tuple(input_ids),
        segments=tuple(segments),
        global_max_length=global_max_length,
    )


def _example_id(example: Any) -> str:
    value = getattr(example, "example_id", None)
    if not isinstance(value, str) or not value:
        raise PackingContractError(
            "encoded example must expose a non-empty example_id",
            code="packing.example_id",
            context={"value_type": type(value).__name__},
        )
    return value


def _input_ids(example: Any, *, example_id: str) -> tuple[int, ...]:
    value = getattr(example, "input_ids", None)
    if not isinstance(value, tuple):
        raise PackingContractError(
            "encoded example input_ids must be a tuple",
            code="packing.input_ids_shape",
            context={"example_id": example_id, "value_type": type(value).__name__},
        )
    try:
        return tuple(int(token_id) for token_id in value)
    except (TypeError, ValueError) as exc:
        raise PackingContractError(
            "encoded example input_ids must contain integer token ids",
            code="packing.input_ids_type",
            context={"example_id": example_id},
            cause=exc,
        ) from exc


__all__ = [
    "PackedSegment",
    "PackedSequence",
    "plan_packed_sequences",
]
