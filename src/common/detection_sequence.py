"""Detection sequence rendering/parsing for compact native-token variants.

This module is the lightweight compatibility facade used by inference and
non-training callers.  Its parser intentionally strips generation suffixes and
returns ``None`` for malformed compact rows instead of raising strict template
errors.  Keep this behavior separate from ``src.detection.template`` training
parsers unless a future shared low-level row helper can preserve both contracts
without introducing an import cycle.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from src.common.detection_compact_rows import (
    BOX_START_TOKEN,
    OBJECT_REF_START_TOKEN,
    STRICT_COMPACT_ROW_COORD_TOKEN_RE,
    parse_compact_row,
    render_compact_row,
)
from src.utils.assistant_json import dumps_coordjson

IM_END_TOKEN = "<|im_end|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"

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


def _compact_marker_flags(fmt: DetectionSequenceFormat) -> tuple[bool, bool]:
    if fmt == COMPACT_FULL_FORMAT:
        return True, True
    if fmt == COMPACT_NO_DESC_FORMAT:
        return False, True
    if fmt == COMPACT_NO_BBOX_FORMAT:
        return True, False
    return False, False


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
    if fmt == COORDJSON_FORMAT:
        raise ValueError("coordjson does not have a compact row pattern")
    include_object_ref_marker, include_bbox_start_marker = _compact_marker_flags(fmt)
    return render_compact_row(
        "{desc}",
        (coord_tail,),
        include_object_ref_marker=include_object_ref_marker,
        include_bbox_start_marker=include_bbox_start_marker,
    )


def _validate_desc(desc: Any) -> str:
    if not isinstance(desc, str):
        raise ValueError("object desc must be a string")
    for forbidden in _FORBIDDEN_DESC_SUBSTRINGS:
        if forbidden in desc:
            raise ValueError(f"object desc contains forbidden marker {forbidden!r}")
    value = desc.strip()
    if not value:
        raise ValueError("object desc must be non-empty")
    return value


def _validate_bbox_tokens(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("object bbox_2d must be a sequence of four coord tokens")
    tokens = [str(v) for v in value]
    if len(tokens) != 4:
        raise ValueError("object bbox_2d must contain exactly four coord tokens")
    if not all(STRICT_COMPACT_ROW_COORD_TOKEN_RE.fullmatch(token) for token in tokens):
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
    if fmt == COMPACT_FULL_FORMAT:
        from src.detection.teacher_forcing.compact_full_policy import (
            render_compact_full,
        )

        return render_compact_full(
            payload,
            serialization_policy="marker_delimited",
        )

    objects = payload.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("payload.objects must be a sequence")

    include_object_ref_marker, include_bbox_start_marker = _compact_marker_flags(fmt)
    rows: list[str] = []
    for entry in objects:
        if not isinstance(entry, Mapping):
            raise ValueError("payload.objects entries must be mappings")
        desc = _validate_desc(entry.get("desc"))
        bbox_tokens = _validate_bbox_tokens(entry.get("bbox_2d"))
        rows.append(
            render_compact_row(
                desc,
                bbox_tokens,
                include_object_ref_marker=include_object_ref_marker,
                include_bbox_start_marker=include_bbox_start_marker,
            )
        )
    return "\n".join(rows)


def _strip_generation_suffix(text: str) -> str:
    stripped = str(text)
    terminal_positions = [
        pos
        for token in (IM_END_TOKEN, END_OF_TEXT_TOKEN)
        if (pos := stripped.find(token)) >= 0
    ]
    if terminal_positions:
        stripped = stripped[: min(terminal_positions)]
        stripped = stripped.rstrip()
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
    require_object_ref_marker, require_bbox_start_marker = _compact_marker_flags(fmt)
    parts = parse_compact_row(
        row,
        require_object_ref_marker=require_object_ref_marker,
        require_bbox_start_marker=require_bbox_start_marker,
        coord_token_re=STRICT_COMPACT_ROW_COORD_TOKEN_RE,
    )
    if parts is None:
        return None

    try:
        desc = _validate_desc(parts.desc)
    except ValueError:
        return None
    return {"desc": desc, "bbox_2d": list(parts.bbox_tokens)}


def parse_compact_detection_sequence(
    text: str,
    *,
    detection_sequence_format: str | None = None,
) -> dict[str, Any] | None:
    """Parse compact generated text back into canonical prediction objects.

    Compatibility contract: this helper accepts generated-text suffixes such as
    chat stop markers and reports malformed rows as ``None``.  Strict training
    template validity is owned by ``CompactFullTemplate.parse_assistant``.
    """

    stripped = _strip_generation_suffix(text)
    if not stripped:
        return {"objects": []}
    if "<|coord_" not in stripped:
        if stripped.isspace():
            return {"objects": []}
        return None

    fmt = (
        _auto_detect_format(stripped)
        if detection_sequence_format is None
        else normalize_detection_sequence_format(detection_sequence_format)
    )
    if fmt == COORDJSON_FORMAT:
        return None
    if fmt == COMPACT_FULL_FORMAT:
        from src.detection.teacher_forcing.compact_full_policy import parse_compact_full

        result = parse_compact_full(stripped, mode="legacy_compatible")
        if not result.ok:
            return None
        return result.to_payload()

    objects: list[dict[str, Any]] = []
    for row in stripped.split("\n"):
        if not row:
            return None
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
