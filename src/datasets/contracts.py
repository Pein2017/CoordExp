"""Shared dataset contracts and validation helpers.

Notes:
- All public CoordExp JSONLs (numeric text and coord tokens) are pre-normalized to
  norm1000 (0–999). Validation/enforcement will raise if values fall outside the
  expected range when normalization is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from src.common.schemas import (
    ConversationRecord,
    GeometryDict,
    MessageContent,
    MessageDict,
)


@dataclass(frozen=True)
class AugmentationTelemetry:
    kept_indices: Tuple[int, ...]
    coverages: Tuple[float, ...]
    allows_geometry_drops: bool
    width: Optional[int]
    height: Optional[int]
    padding_ratio: Optional[float]
    skip_reason: Optional[str]
    skip_counts: Mapping[str, int]


def validate_conversation_record(record: Mapping[str, Any]) -> ConversationRecord:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    messages = record.get("messages")
    if messages is None:
        # Raw dense-caption records (images/objects/summary) are accepted as-is.
        return cast(ConversationRecord, record)

    if not isinstance(messages, Sequence):
        raise ValueError("conversation record 'messages' must be a sequence")

    for index, turn in enumerate(messages):
        if not isinstance(turn, Mapping):
            raise ValueError(f"messages[{index}] must be a mapping")
        turn_typed = cast(MessageDict, turn)
        if "role" not in turn_typed:
            raise ValueError(f"messages[{index}] missing 'role'")
        content = turn_typed.get("content", [])
        if not isinstance(content, Sequence):
            raise ValueError(f"messages[{index}]['content'] must be a sequence")
        cast(Sequence[MessageContent], content)
    return cast(ConversationRecord, record)


def validate_geometry_sequence(
    geometries: Iterable[Mapping[str, Any]],
) -> Tuple[GeometryDict, ...]:
    validated: list[GeometryDict] = []
    for index, geom in enumerate(geometries):
        if not isinstance(geom, Mapping):
            raise ValueError(f"geometry[{index}] must be a mapping")

        if "bbox" in geom or "polygon" in geom:
            raise ValueError(
                f"geometry[{index}] uses legacy keys ('bbox'/'polygon'); expected bbox_2d|poly"
            )

        bbox_2d = geom.get("bbox_2d")
        poly = geom.get("poly")

        has_bbox = bbox_2d is not None
        has_poly = poly is not None

        if has_bbox and has_poly:
            raise ValueError(
                f"geometry[{index}] must contain exactly one geometry key (bbox_2d xor poly), got both"
            )
        if not has_bbox and not has_poly:
            raise ValueError(
                f"geometry[{index}] must contain exactly one geometry key (bbox_2d xor poly), got none"
            )

        if has_bbox:
            if not isinstance(bbox_2d, Sequence) or isinstance(bbox_2d, (str, bytes)):
                raise ValueError(
                    f"geometry[{index}]['bbox_2d'] must be a sequence if provided"
                )
            if len(bbox_2d) != 4:
                raise ValueError(
                    f"geometry[{index}]['bbox_2d'] must contain exactly 4 values; got len={len(bbox_2d)}"
                )

        if has_poly:
            if not isinstance(poly, Sequence) or isinstance(poly, (str, bytes)):
                raise ValueError(
                    f"geometry[{index}]['poly'] must be a sequence if provided"
                )
            if any(
                isinstance(v, Sequence) and not isinstance(v, (str, bytes)) for v in poly
            ):
                raise ValueError(
                    f"geometry[{index}]['poly'] must be a flat coordinate sequence"
                )
            if len(poly) < 6 or (len(poly) % 2 != 0):
                raise ValueError(
                    f"geometry[{index}]['poly'] must contain an even number of values and at least 6 coordinates; got len={len(poly)}"
                )

        validated.append(cast(GeometryDict, geom))
    return tuple(validated)


def clone_record(record: Mapping[str, Any]) -> MutableMapping[str, Any]:
    return cast(MutableMapping[str, Any], dict(record))
