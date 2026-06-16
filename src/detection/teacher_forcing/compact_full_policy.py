"""Marker-delimited ``compact_full`` render and parse policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, cast

from src.common.detection_compact_rows import (
    BOX_START_TOKEN,
    COMPACT_DESC_FORBIDDEN_SUBSTRINGS,
    END_OF_TEXT_TOKEN,
    IM_END_TOKEN,
    OBJECT_REF_START_TOKEN,
    STRICT_COMPACT_ROW_COORD_TOKEN_RE,
    render_compact_row,
    valid_xyxy_positive_area,
)

CompactFullSerializationPolicy = Literal[
    "marker_delimited",
    "legacy_newline_delimited",
]
CompactFullParseMode = Literal[
    "marker_delimited_strict",
    "marker_delimited_axis_sort_repair",
    "legacy_compatible",
]
CompactFullParseErrorCode = Literal[
    "empty_output",
    "legacy_separator_in_new_format",
    "missing_object_ref_start",
    "missing_box_start",
    "empty_description",
    "forbidden_description_token",
    "wrong_coord_arity",
    "invalid_coord_token",
    "trailing_garbage",
    "invalid_geometry",
]

_FORBIDDEN_DESCRIPTION_TOKENS = COMPACT_DESC_FORBIDDEN_SUBSTRINGS


@dataclass(frozen=True)
class CompactFullParsedObject:
    description: str
    bbox_tokens: tuple[str, str, str, str]

    @property
    def desc(self) -> str:
        return self.description

    @property
    def bbox_2d(self) -> list[str]:
        return list(self.bbox_tokens)

    def to_payload_object(self) -> dict[str, Any]:
        return {"desc": self.description, "bbox_2d": list(self.bbox_tokens)}


@dataclass(frozen=True)
class CompactFullParseResult:
    mode: CompactFullParseMode
    objects: tuple[CompactFullParsedObject, ...] = ()
    terminal_token: str | None = None
    error_code: CompactFullParseErrorCode | None = None
    error_offset: int | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None

    def to_payload(self) -> dict[str, Any]:
        return {"objects": [obj.to_payload_object() for obj in self.objects]}


def render_compact_full(
    payload: Mapping[str, Any] | Any,
    *,
    serialization_policy: str = "marker_delimited",
) -> str:
    """Render compact_full objects with the selected object separator policy."""

    policy = _normalize_serialization_policy(serialization_policy)
    rows = [_render_object(obj) for obj in _iter_objects(payload)]
    separator = "" if policy == "marker_delimited" else "\n"
    return separator.join(rows)


def parse_compact_full(
    text: str,
    *,
    mode: str = "marker_delimited_strict",
) -> CompactFullParseResult:
    """Parse ``compact_full`` text according to the requested policy mode."""

    parse_mode = _normalize_parse_mode(mode)
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if parse_mode in {"marker_delimited_strict", "marker_delimited_axis_sort_repair"}:
        return _parse_marker_delimited_strict(text, mode=parse_mode)
    return _parse_legacy_compatible(text, mode=parse_mode)


def _parse_marker_delimited_strict(
    text: str,
    *,
    mode: CompactFullParseMode,
) -> CompactFullParseResult:
    if text == "":
        return _error(mode, "empty_output", 0)
    if "\n" in text:
        return _error(mode, "legacy_separator_in_new_format", text.index("\n"))

    objects: list[CompactFullParsedObject] = []
    index = 0
    terminal_token: str | None = None

    while index < len(text):
        if text.startswith(IM_END_TOKEN, index):
            terminal_token = IM_END_TOKEN
            index += len(IM_END_TOKEN)
            if _strip_compat_padding_suffix(text[index:]):
                return _error(mode, "trailing_garbage", index)
            break
        if not text.startswith(OBJECT_REF_START_TOKEN, index):
            return _error(mode, "missing_object_ref_start", index)

        parsed = _parse_one_object(text, index, mode=mode)
        if isinstance(parsed, CompactFullParseResult):
            return parsed
        obj, index = parsed
        objects.append(obj)

        if index == len(text):
            break
        if text.startswith(OBJECT_REF_START_TOKEN, index):
            continue
        if text.startswith(IM_END_TOKEN, index):
            continue
        if text[index] == "\n":
            return _error(mode, "legacy_separator_in_new_format", index)
        if text.startswith("<|coord_", index):
            return _error(mode, "wrong_coord_arity", index)
        return _error(mode, "trailing_garbage", index)

    return CompactFullParseResult(
        mode=mode,
        objects=tuple(objects),
        terminal_token=terminal_token,
    )


def _parse_legacy_compatible(
    text: str,
    *,
    mode: CompactFullParseMode,
) -> CompactFullParseResult:
    stripped, terminal_token = _strip_generation_terminal(text)
    if stripped == "":
        return _error(mode, "empty_output", 0)

    if "\n" not in stripped:
        strict = _parse_marker_delimited_strict(stripped, mode=mode)
        if strict.ok:
            return CompactFullParseResult(
                mode=mode,
                objects=strict.objects,
                terminal_token=terminal_token or strict.terminal_token,
            )
        return strict

    objects: list[CompactFullParsedObject] = []
    offset = 0
    for row in stripped.split("\n"):
        if row == "":
            return _error(mode, "legacy_separator_in_new_format", offset)
        parsed = _parse_marker_delimited_strict(row, mode=mode)
        if not parsed.ok:
            return CompactFullParseResult(
                mode=mode,
                terminal_token=parsed.terminal_token,
                error_code=parsed.error_code,
                error_offset=offset + (parsed.error_offset or 0),
            )
        if len(parsed.objects) != 1:
            return _error(mode, "legacy_separator_in_new_format", offset)
        objects.extend(parsed.objects)
        offset += len(row) + 1

    return CompactFullParseResult(
        mode=mode,
        objects=tuple(objects),
        terminal_token=terminal_token,
    )


def _parse_one_object(
    text: str,
    start: int,
    *,
    mode: CompactFullParseMode,
) -> tuple[CompactFullParsedObject, int] | CompactFullParseResult:
    desc_start = start + len(OBJECT_REF_START_TOKEN)
    box_start = text.find(BOX_START_TOKEN, desc_start)
    if box_start < 0:
        return _error(mode, "missing_box_start", desc_start)

    desc = text[desc_start:box_start]
    desc_error = _validate_description(desc)
    if desc_error is not None:
        return _error(mode, desc_error, desc_start)

    coord_index = box_start + len(BOX_START_TOKEN)
    bbox_tokens: list[str] = []
    for _slot_index in range(4):
        match = STRICT_COMPACT_ROW_COORD_TOKEN_RE.match(text, coord_index)
        if match is None:
            if text.startswith("<|coord_", coord_index):
                return _error(mode, "invalid_coord_token", coord_index)
            return _error(mode, "wrong_coord_arity", coord_index)
        bbox_tokens.append(match.group(0))
        coord_index = match.end()

    if text.startswith("<|coord_", coord_index):
        return _error(mode, "wrong_coord_arity", coord_index)

    bbox = cast(tuple[str, str, str, str], tuple(bbox_tokens))
    if mode == "marker_delimited_axis_sort_repair":
        bbox = _axis_sort_bbox_tokens(bbox)
    if not valid_xyxy_positive_area(bbox):
        return _error(mode, "invalid_geometry", box_start + len(BOX_START_TOKEN))

    return CompactFullParsedObject(description=desc, bbox_tokens=bbox), coord_index


def _render_object(obj: Any) -> str:
    if isinstance(obj, Mapping):
        desc = obj.get("desc")
        bbox = obj.get("bbox_2d")
    else:
        desc = getattr(obj, "desc", None)
        bbox_box = getattr(obj, "bbox_2d", None)
        bbox = getattr(bbox_box, "tokens", bbox_box)

    desc_text = _validate_render_desc(desc)
    bbox_tokens = _validate_bbox_tokens(bbox)
    return render_compact_row(
        desc_text,
        bbox_tokens,
        include_object_ref_marker=True,
        include_bbox_start_marker=True,
    )


def _iter_objects(payload: Mapping[str, Any] | Any) -> Sequence[Any]:
    objects: Any
    if isinstance(payload, Mapping):
        objects = payload.get("objects")
    else:
        objects = getattr(payload, "objects", None)
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("payload.objects must be a sequence")
    return objects


def _validate_render_desc(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("object desc must be a string")
    if not value.strip():
        raise ValueError("object desc must be non-empty")
    if _validate_description(value) is not None:
        raise ValueError("object desc contains forbidden compact_full marker")
    return value


def _validate_description(desc: str) -> CompactFullParseErrorCode | None:
    if not desc.strip():
        return "empty_description"
    for forbidden in _FORBIDDEN_DESCRIPTION_TOKENS:
        if forbidden in desc:
            return "forbidden_description_token"
    return None


def _validate_bbox_tokens(value: Any) -> tuple[str, str, str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("object bbox_2d must be a sequence of four coord tokens")
    tokens = tuple(str(token) for token in value)
    if len(tokens) != 4:
        raise ValueError("object bbox_2d must contain exactly four coord tokens")
    if not all(STRICT_COMPACT_ROW_COORD_TOKEN_RE.fullmatch(token) for token in tokens):
        raise ValueError("object bbox_2d must contain only <|coord_N|> tokens")
    bbox = cast(tuple[str, str, str, str], tokens)
    if not valid_xyxy_positive_area(bbox):
        raise ValueError("object bbox_2d must be a valid xyxy positive-area box")
    return bbox


def _axis_sort_bbox_tokens(
    bbox: tuple[str, str, str, str],
) -> tuple[str, str, str, str]:
    x1, y1, x2, y2 = (_coord_token_value(token) for token in bbox)
    sx1, sx2 = sorted((x1, x2))
    sy1, sy2 = sorted((y1, y2))
    return (
        f"<|coord_{sx1}|>",
        f"<|coord_{sy1}|>",
        f"<|coord_{sx2}|>",
        f"<|coord_{sy2}|>",
    )


def _coord_token_value(token: str) -> int:
    match = STRICT_COMPACT_ROW_COORD_TOKEN_RE.fullmatch(str(token))
    if match is None:
        raise ValueError(f"invalid coordinate token: {token!r}")
    return int(match.group(1))


def _strip_generation_terminal(text: str) -> tuple[str, str | None]:
    im_end_pos = text.find(IM_END_TOKEN)
    if im_end_pos >= 0:
        return text[:im_end_pos].rstrip(), IM_END_TOKEN
    return _strip_compat_padding_suffix(text).rstrip(), None


def _strip_compat_padding_suffix(text: str) -> str:
    stripped = text.rstrip()
    while stripped.endswith(END_OF_TEXT_TOKEN):
        stripped = stripped[: -len(END_OF_TEXT_TOKEN)].rstrip()
    return stripped


def _normalize_serialization_policy(value: str) -> CompactFullSerializationPolicy:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"marker_delimited", "legacy_newline_delimited"}:
        raise ValueError(
            "compact_full serialization_policy must be one of "
            "{'marker_delimited', 'legacy_newline_delimited'}"
        )
    return cast(CompactFullSerializationPolicy, normalized)


def _normalize_parse_mode(value: str) -> CompactFullParseMode:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {
        "marker_delimited_strict",
        "marker_delimited_axis_sort_repair",
        "legacy_compatible",
    }:
        raise ValueError(
            "compact_full parse mode must be one of "
            "{'marker_delimited_strict', "
            "'marker_delimited_axis_sort_repair', "
            "'legacy_compatible'}"
        )
    return cast(CompactFullParseMode, normalized)


def _error(
    mode: CompactFullParseMode,
    code: CompactFullParseErrorCode,
    offset: int,
) -> CompactFullParseResult:
    return CompactFullParseResult(mode=mode, error_code=code, error_offset=offset)


__all__ = [
    "BOX_START_TOKEN",
    "CompactFullParseErrorCode",
    "CompactFullParseMode",
    "CompactFullParseResult",
    "CompactFullParsedObject",
    "CompactFullSerializationPolicy",
    "END_OF_TEXT_TOKEN",
    "IM_END_TOKEN",
    "OBJECT_REF_START_TOKEN",
    "parse_compact_full",
    "render_compact_full",
]
