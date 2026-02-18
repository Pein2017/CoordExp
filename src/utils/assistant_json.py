from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from src.coord_tokens.codec import is_coord_token

CANONICAL_JSON_SEPARATORS: tuple[str, str] = (", ", ": ")


def dumps_canonical_json(value: Any) -> str:
    """Serialize strict JSON with the repo's canonical ms-swift-compatible style."""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=None,
        separators=CANONICAL_JSON_SEPARATORS,
    )


def _render_coordjson_value(value: Any, *, coord_context: bool = False) -> str:
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            key_text = json.dumps(str(key), ensure_ascii=False)
            next_coord_context = str(key) in {"bbox_2d", "poly"}
            parts.append(
                f"{key_text}: {_render_coordjson_value(item, coord_context=next_coord_context)}"
            )
        return "{" + ", ".join(parts) + "}"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[str] = []
        for item in value:
            if coord_context and isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                raise ValueError(
                    "CoordJSON geometry arrays must be flat and CoordTok-only; nested arrays are invalid"
                )
            items.append(_render_coordjson_value(item, coord_context=coord_context))
        return "[" + ", ".join(items) + "]"

    if coord_context:
        if isinstance(value, str) and is_coord_token(value):
            return value
        raise ValueError(
            "CoordJSON geometry arrays must contain bare coord tokens like <|coord_123|>"
        )

    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False)

    if value is None or isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False)

    raise TypeError(f"Unsupported value type for CoordJSON serialization: {type(value)!r}")


def dumps_coordjson(value: Mapping[str, Any]) -> str:
    """Serialize CoordJSON text using canonical spacing and bare CoordTok literals."""

    if not isinstance(value, Mapping):
        raise TypeError("CoordJSON payload must be a mapping")
    return _render_coordjson_value(value, coord_context=False)


__all__ = [
    "CANONICAL_JSON_SEPARATORS",
    "dumps_canonical_json",
    "dumps_coordjson",
]
