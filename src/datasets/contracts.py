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
        # Raw dense-caption records (images/objects/summary).
        #
        # Even though these records are not yet rendered into chat messages, we still
        # enforce the JSONL contract shape here so failures are early + actionable.
        images = record.get("images")
        if images is None:
            raise ValueError("record is missing required key 'images'")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            raise ValueError("record['images'] must be a list of image-path strings")
        for idx, image in enumerate(images):
            if not isinstance(image, str):
                raise ValueError(f"images[{idx}] must be a string path, got {type(image)!r}")

        objects = record.get("objects")
        if objects is None:
            raise ValueError("record is missing required key 'objects'")
        if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
            raise ValueError("record['objects'] must be a list")

        def _require_positive_int(value: Any, *, name: str) -> int:
            if value is None:
                raise ValueError(f"record is missing required key '{name}'")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {type(value)!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            return int(value)

        _require_positive_int(record.get("width"), name="width")
        _require_positive_int(record.get("height"), name="height")

        # Object-level shape validation (desc + exactly one geometry field).
        for idx, obj in enumerate(objects):
            if not isinstance(obj, Mapping):
                raise ValueError(f"objects[{idx}] must be a mapping")
            desc = obj.get("desc")
            if not isinstance(desc, str):
                raise ValueError(f"objects[{idx}] must provide a string 'desc'")

            has_bbox = "bbox_2d" in obj
            has_poly = "poly" in obj
            if has_bbox and has_poly:
                raise ValueError(
                    f"objects[{idx}] must contain exactly one geometry field (bbox_2d xor poly), got both"
                )
            if not has_bbox and not has_poly:
                raise ValueError(
                    f"objects[{idx}] must contain exactly one geometry field (bbox_2d xor poly), got none"
                )
            if "bbox" in obj or "polygon" in obj:
                raise ValueError(
                    f"objects[{idx}] uses legacy geometry keys ('bbox'/'polygon'); expected bbox_2d|poly"
                )

            geom_key = "bbox_2d" if has_bbox else "poly"
            geom = obj.get(geom_key)
            if not isinstance(geom, Sequence) or isinstance(geom, (str, bytes)):
                raise ValueError(f"objects[{idx}].{geom_key} must be a sequence")
            if geom_key == "bbox_2d" and len(geom) != 4:
                raise ValueError(
                    f"objects[{idx}].bbox_2d must contain exactly 4 values; got len={len(geom)}"
                )
            if geom_key == "poly":
                if len(geom) < 6 or (len(geom) % 2 != 0):
                    raise ValueError(
                        f"objects[{idx}].poly must contain an even number of values and at least 6 coordinates; got len={len(geom)}"
                    )

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
