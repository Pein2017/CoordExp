"""Low-level compact detection row rendering/parsing primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

OBJECT_REF_START_TOKEN = "<|object_ref_start|>"
BOX_START_TOKEN = "<|box_start|>"

COMPACT_ROW_COORD_TOKEN_RE: re.Pattern[str] = re.compile(r"<\|coord_\d+\|>")
STRICT_COMPACT_ROW_COORD_TOKEN_RE: re.Pattern[str] = re.compile(
    r"<\|coord_(0|[1-9]\d{0,2})\|>"
)


@dataclass(frozen=True)
class CompactRowParts:
    """Parsed compact row fields before caller-specific validation."""

    desc: str
    bbox_tokens: tuple[str, ...]


def render_compact_row(
    desc: str,
    bbox_tokens: Sequence[str],
    *,
    include_object_ref_marker: bool,
    include_bbox_start_marker: bool,
) -> str:
    """Render one compact row after the caller has validated its fields."""

    prefix = OBJECT_REF_START_TOKEN if include_object_ref_marker else ""
    bbox_start = BOX_START_TOKEN if include_bbox_start_marker else ""
    return f"{prefix}{desc}{bbox_start}{''.join(bbox_tokens)}"


def parse_compact_row(
    row: str,
    *,
    require_object_ref_marker: bool,
    require_bbox_start_marker: bool,
    coord_token_re: re.Pattern[str] | None = None,
    bbox_marker_split: Literal["first", "last"] = "last",
) -> CompactRowParts | None:
    """Split one compact row into raw fields without applying facade policy."""

    pattern = coord_token_re or COMPACT_ROW_COORD_TOKEN_RE
    body = row

    if require_object_ref_marker:
        if not body.startswith(OBJECT_REF_START_TOKEN):
            return None
        body = body[len(OBJECT_REF_START_TOKEN) :]

    if require_bbox_start_marker:
        if BOX_START_TOKEN not in body:
            return None
        if bbox_marker_split == "first":
            desc, coord_tail = body.split(BOX_START_TOKEN, maxsplit=1)
        else:
            desc, coord_tail = body.rsplit(BOX_START_TOKEN, maxsplit=1)
        bbox_tokens = _parse_coord_tail(coord_tail, pattern)
        if bbox_tokens is None:
            return None
    else:
        coord_matches = list(pattern.finditer(body))
        if len(coord_matches) != 4:
            return None
        bbox_tokens = _coord_tokens_from_matches(
            body,
            coord_matches,
            require_start=False,
        )
        if bbox_tokens is None:
            return None
        prefix = body[: coord_matches[0].start()]
        desc = prefix

    return CompactRowParts(desc=desc, bbox_tokens=bbox_tokens)


def _parse_coord_tail(
    coord_tail: str, pattern: re.Pattern[str]
) -> tuple[str, ...] | None:
    coord_matches = list(pattern.finditer(coord_tail))
    if len(coord_matches) != 4:
        return None
    return _coord_tokens_from_matches(coord_tail, coord_matches)


def _coord_tokens_from_matches(
    text: str,
    coord_matches: Sequence[re.Match[str]],
    *,
    require_start: bool = True,
) -> tuple[str, ...] | None:
    if require_start and coord_matches[0].start() != 0:
        return None
    if any(
        match.end() != next_match.start()
        for match, next_match in zip(coord_matches, coord_matches[1:])
    ):
        return None
    if coord_matches[-1].end() != len(text):
        return None
    return tuple(match.group(0) for match in coord_matches)
