from __future__ import annotations

from typing import Any, MutableMapping, Sequence

from .codec import (
    ints_to_tokens,
    normalized_from_ints,
    sequence_has_coord_tokens,
    tokens_to_ints,
    value_in_coord_range,
)

GEOMETRY_KEYS = ("bbox_2d", "poly", "line")


def _looks_like_coord_value(value: Any) -> bool:
    try:
        v_float = float(value)
    except (TypeError, ValueError):
        return False
    v_int = int(round(v_float))
    return abs(v_float - v_int) < 1e-6 and value_in_coord_range(v_int)


def annotate_coord_tokens(
    record: MutableMapping[str, Any]
) -> bool:
    """Attach coord-token metadata to a record in-place.

    When coord tokens (or pre-quantized ints) are detected, the function stores:
      - obj["_coord_tokens"][<geom>]    -> list[str] tokens
      - obj["_coord_token_ints"][<geom>] -> list[int]
      - obj["_coord_token_norm"][<geom>] -> list[float] normalized (k/1000)

    Returns True if any coord-token geometry was found.
    """

    if not isinstance(record, MutableMapping):
        raise TypeError("record must be a mutable mapping")

    objects = record.get("objects") or []
    if not isinstance(objects, list):
        raise ValueError(
            "record['objects'] must be a list when coord_tokens are enabled"
        )

    width = record.get("width")
    height = record.get("height")

    found = False
    for obj_idx, obj in enumerate(objects):
        if not isinstance(obj, MutableMapping):
            raise ValueError(f"objects[{obj_idx}] must be a mapping")

        for key in GEOMETRY_KEYS:
            if key not in obj or obj[key] is None:
                continue
            values = obj[key]
            if not isinstance(values, Sequence):
                raise ValueError(f"objects[{obj_idx}].{key} must be a sequence")
            if len(values) % 2 != 0:
                raise ValueError(
                    f"objects[{obj_idx}].{key} must contain an even number of entries (x,y pairs)"
                )

            has_tokens = sequence_has_coord_tokens(values)
            looks_quantized = all(_looks_like_coord_value(v) for v in values)
            if not has_tokens and not looks_quantized:
                continue

            if width is None or height is None:
                raise ValueError(
                    "Record with coord tokens must provide width and height for pixel recovery"
                )
            if float(width) <= 0 or float(height) <= 0:
                raise ValueError(
                    "width and height must be positive when coord tokens are used"
                )

            ints = tokens_to_ints(values, require_even=True)
            tokens = ints_to_tokens(ints)
            obj.setdefault("_coord_tokens", {})[key] = tokens
            obj.setdefault("_coord_token_ints", {})[key] = ints
            obj.setdefault("_coord_token_norm", {})[key] = normalized_from_ints(ints)
            found = True

    if found:
        record["_coord_tokens_enabled"] = True
    return found


__all__ = ["annotate_coord_tokens", "GEOMETRY_KEYS"]
