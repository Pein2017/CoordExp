"""Detection sequence rendering/parsing for compact native-token variants."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, cast

from src.utils.assistant_json import dumps_coordjson

OBJECT_REF_START_TOKEN = "<|object_ref_start|>"
BOX_START_TOKEN = "<|box_start|>"
IM_END_TOKEN = "<|im_end|>"

DetectionSequenceFormat = str

COORDJSON_FORMAT = "coordjson"
COMPACT_FULL_FORMAT = "compact_full"
COMPACT_NO_DESC_FORMAT = "compact_no_desc"
COMPACT_NO_BBOX_FORMAT = "compact_no_bbox"
COMPACT_MIN_FORMAT = "compact_min"

ALLOWED_DETECTION_SEQUENCE_FORMATS = {
    COORDJSON_FORMAT,
    COMPACT_FULL_FORMAT,
    COMPACT_NO_DESC_FORMAT,
    COMPACT_NO_BBOX_FORMAT,
    COMPACT_MIN_FORMAT,
}

_COORD_TOKEN_RE = re.compile(r"<\|coord_(\d+)\|>")
_FORBIDDEN_DESC_SUBSTRINGS = (
    "\n",
    "\r",
    "\t",
    OBJECT_REF_START_TOKEN,
    BOX_START_TOKEN,
    "<|coord_",
    "<|im_start|>",
    IM_END_TOKEN,
)


def normalize_detection_sequence_format(value: Any) -> DetectionSequenceFormat:
    if value is None:
        return COORDJSON_FORMAT
    if not isinstance(value, str):
        raise TypeError("detection_sequence_format must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ALLOWED_DETECTION_SEQUENCE_FORMATS:
        allowed = ", ".join(sorted(ALLOWED_DETECTION_SEQUENCE_FORMATS))
        raise ValueError(
            f"detection_sequence_format must be one of {{{allowed}}}, got {value!r}"
        )
    return cast(DetectionSequenceFormat, normalized)


def required_special_tokens_for_detection_sequence_format(
    detection_sequence_format: str,
) -> tuple[str, ...]:
    fmt = normalize_detection_sequence_format(detection_sequence_format)
    if fmt == COMPACT_FULL_FORMAT:
        return (OBJECT_REF_START_TOKEN, BOX_START_TOKEN)
    if fmt == COMPACT_NO_DESC_FORMAT:
        return (BOX_START_TOKEN,)
    if fmt == COMPACT_NO_BBOX_FORMAT:
        return (OBJECT_REF_START_TOKEN,)
    return ()


def compact_pattern_for_detection_sequence_format(
    detection_sequence_format: str,
) -> str:
    fmt = normalize_detection_sequence_format(detection_sequence_format)
    coord_tail = "<|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>"
    if fmt == COMPACT_FULL_FORMAT:
        return f"{OBJECT_REF_START_TOKEN}{{desc}}{BOX_START_TOKEN}{coord_tail}"
    if fmt == COMPACT_NO_DESC_FORMAT:
        return f"{{desc}}{BOX_START_TOKEN}{coord_tail}"
    if fmt == COMPACT_NO_BBOX_FORMAT:
        return f"{OBJECT_REF_START_TOKEN}{{desc}}{coord_tail}"
    if fmt == COMPACT_MIN_FORMAT:
        return f"{{desc}}{coord_tail}"
    raise ValueError("coordjson does not have a compact row pattern")


def _validate_desc(desc: Any) -> str:
    if not isinstance(desc, str):
        raise ValueError("object desc must be a string")
    value = desc.strip()
    if not value:
        raise ValueError("object desc must be non-empty")
    for forbidden in _FORBIDDEN_DESC_SUBSTRINGS:
        if forbidden in value:
            raise ValueError(f"object desc contains forbidden marker {forbidden!r}")
    return value


def _validate_bbox_tokens(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("object bbox_2d must be a sequence of four coord tokens")
    tokens = [str(v) for v in value]
    if len(tokens) != 4:
        raise ValueError("object bbox_2d must contain exactly four coord tokens")
    if not all(_COORD_TOKEN_RE.fullmatch(token) for token in tokens):
        raise ValueError("object bbox_2d must contain only <|coord_N|> tokens")
    return tokens


def render_compact_detection_sequence(
    payload: Mapping[str, Any],
    *,
    detection_sequence_format: str = COORDJSON_FORMAT,
) -> str:
    """Render canonical ``{"objects": [...]}`` payload as a detection sequence."""

    fmt = normalize_detection_sequence_format(detection_sequence_format)
    if fmt == COORDJSON_FORMAT:
        return dumps_coordjson(payload)

    objects = payload.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("payload.objects must be a sequence")

    rows: list[str] = []
    for entry in objects:
        if not isinstance(entry, Mapping):
            raise ValueError("payload.objects entries must be mappings")
        desc = _validate_desc(entry.get("desc"))
        bbox_tokens = _validate_bbox_tokens(entry.get("bbox_2d"))
        bbox_text = "".join(bbox_tokens)
        if fmt == COMPACT_FULL_FORMAT:
            rows.append(f"{OBJECT_REF_START_TOKEN}{desc}{BOX_START_TOKEN}{bbox_text}")
        elif fmt == COMPACT_NO_DESC_FORMAT:
            rows.append(f"{desc}{BOX_START_TOKEN}{bbox_text}")
        elif fmt == COMPACT_NO_BBOX_FORMAT:
            rows.append(f"{OBJECT_REF_START_TOKEN}{desc}{bbox_text}")
        else:
            rows.append(f"{desc}{bbox_text}")
    return "\n".join(rows)


def _strip_generation_suffix(text: str) -> str:
    stripped = str(text).strip()
    if stripped.endswith(IM_END_TOKEN):
        stripped = stripped[: -len(IM_END_TOKEN)].rstrip()
    return stripped


def _auto_detect_format(text: str) -> DetectionSequenceFormat:
    has_object_ref = OBJECT_REF_START_TOKEN in text
    has_box = BOX_START_TOKEN in text
    if has_object_ref and has_box:
        return COMPACT_FULL_FORMAT
    if has_box:
        return COMPACT_NO_DESC_FORMAT
    if has_object_ref:
        return COMPACT_NO_BBOX_FORMAT
    return COMPACT_MIN_FORMAT


def _parse_row(row: str, *, fmt: DetectionSequenceFormat) -> dict[str, Any] | None:
    coord_matches = list(_COORD_TOKEN_RE.finditer(row))
    if len(coord_matches) != 4:
        return None
    first_coord = coord_matches[0]
    if any(match.end() != next_match.start() for match, next_match in zip(coord_matches, coord_matches[1:])):
        return None
    if coord_matches[-1].end() != len(row):
        return None

    bbox_tokens = [match.group(0) for match in coord_matches]
    prefix = row[: first_coord.start()]
    if fmt == COMPACT_FULL_FORMAT:
        if not prefix.startswith(OBJECT_REF_START_TOKEN):
            return None
        body = prefix[len(OBJECT_REF_START_TOKEN) :]
        if BOX_START_TOKEN not in body:
            return None
        desc, trailing = body.rsplit(BOX_START_TOKEN, maxsplit=1)
        if trailing:
            return None
    elif fmt == COMPACT_NO_DESC_FORMAT:
        if BOX_START_TOKEN not in prefix:
            return None
        desc, trailing = prefix.rsplit(BOX_START_TOKEN, maxsplit=1)
        if trailing:
            return None
    elif fmt == COMPACT_NO_BBOX_FORMAT:
        if not prefix.startswith(OBJECT_REF_START_TOKEN):
            return None
        desc = prefix[len(OBJECT_REF_START_TOKEN) :]
    else:
        desc = prefix

    try:
        desc = _validate_desc(desc)
    except ValueError:
        return None
    return {"desc": desc, "bbox_2d": bbox_tokens}


def parse_compact_detection_sequence(
    text: str,
    *,
    detection_sequence_format: str | None = None,
) -> dict[str, Any] | None:
    """Parse compact generated text back into canonical prediction objects."""

    stripped = _strip_generation_suffix(text)
    if not stripped:
        return {"objects": []}
    if "<|coord_" not in stripped:
        return None

    fmt = (
        _auto_detect_format(stripped)
        if detection_sequence_format is None
        else normalize_detection_sequence_format(detection_sequence_format)
    )
    if fmt == COORDJSON_FORMAT:
        return None

    objects: list[dict[str, Any]] = []
    for raw_row in stripped.splitlines():
        row = raw_row.strip()
        if not row:
            continue
        parsed = _parse_row(row, fmt=fmt)
        if parsed is None:
            return None
        objects.append(parsed)
    return {"objects": objects}


__all__ = [
    "ALLOWED_DETECTION_SEQUENCE_FORMATS",
    "BOX_START_TOKEN",
    "COMPACT_FULL_FORMAT",
    "COMPACT_MIN_FORMAT",
    "COMPACT_NO_BBOX_FORMAT",
    "COMPACT_NO_DESC_FORMAT",
    "COORDJSON_FORMAT",
    "DetectionSequenceFormat",
    "OBJECT_REF_START_TOKEN",
    "compact_pattern_for_detection_sequence_format",
    "normalize_detection_sequence_format",
    "parse_compact_detection_sequence",
    "render_compact_detection_sequence",
    "required_special_tokens_for_detection_sequence_format",
]
