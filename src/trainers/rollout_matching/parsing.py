"""Rollout parsing helpers and contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.common.object_field_order import (
    build_object_payload,
    normalize_object_field_order,
)
from src.coord_tokens.codec import get_coord_token_ids, value_in_coord_range
from src.utils.assistant_json import dumps_coordjson
from src.utils.coordjson_transpiler import parse_coordjson

from .contracts import GTObject, ParsedPredObject, RolloutParseResult

_IM_END = "<|im_end|>"
_EMPTY_PREFIX_TEXT = '{"objects": ['


def coerce_int(value: Any) -> Optional[int]:
    try:
        v = int(round(float(value)))
    except Exception:
        return None
    if not value_in_coord_range(v):
        return None
    return v


def decode_pieces(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    return [
        tokenizer.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in token_ids
    ]


def _build_prefix_from_char_cut(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    pieces: Sequence[str],
    token_start_chars: Sequence[int],
    cut_char_pos: int,
) -> List[int]:
    if cut_char_pos <= 0:
        return []
    if not token_ids:
        return []
    if cut_char_pos >= (token_start_chars[-1] + len(pieces[-1])):
        return [int(t) for t in token_ids]

    lo = 0
    hi = len(token_start_chars) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        start = token_start_chars[mid]
        next_start = token_start_chars[mid + 1]
        if start <= cut_char_pos < next_start:
            lo = mid
            break
        if cut_char_pos < start:
            hi = mid - 1
        else:
            lo = mid + 1
    i = int(lo)
    start = int(token_start_chars[i])
    offset = max(0, min(int(cut_char_pos - start), len(pieces[i])))

    prefix = [int(t) for t in token_ids[:i]]
    if offset == 0:
        return prefix
    if offset == len(pieces[i]):
        prefix.append(int(token_ids[i]))
        return prefix

    piece_prefix = pieces[i][:offset]
    new_tail = tokenizer.encode(piece_prefix, add_special_tokens=False)
    decoded = tokenizer.decode(
        new_tail, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded != piece_prefix:
        raise ValueError(
            "Failed to retokenize a token-internal cut boundary. "
            f"expected={piece_prefix!r} got={decoded!r}"
        )
    prefix.extend([int(t) for t in new_tail])
    return prefix


def _build_token_char_offsets(pieces: Sequence[str]) -> List[int]:
    offsets: List[int] = []
    cursor = 0
    for piece in pieces:
        offsets.append(int(cursor))
        cursor += int(len(piece))
    return offsets


def _non_ws_last_char(text: str) -> str:
    stripped = str(text).rstrip()
    return stripped[-1] if stripped else ""


def _is_append_ready_prefix(text: str) -> bool:
    last = _non_ws_last_char(text)
    return last in {"[", "}", ","}


def parse_rollout_for_matching(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
    object_field_order: str = "desc_first",
) -> RolloutParseResult:
    resolved_field_order = normalize_object_field_order(
        object_field_order,
        path="custom.object_field_order",
    )
    coord_token_ids = get_coord_token_ids(tokenizer)
    coord_id_set = {int(t) for t in coord_token_ids if int(t) >= 0}

    pieces = decode_pieces(tokenizer, response_token_ids)
    token_start_chars = _build_token_char_offsets(pieces)
    response_text = "".join(pieces)

    stop_pos = response_text.find(_IM_END)
    if stop_pos >= 0:
        response_token_ids = _build_prefix_from_char_cut(
            tokenizer=tokenizer,
            token_ids=response_token_ids,
            pieces=pieces,
            token_start_chars=token_start_chars,
            cut_char_pos=int(stop_pos),
        )
        pieces = decode_pieces(tokenizer, response_token_ids)
        token_start_chars = _build_token_char_offsets(pieces)
        response_text = "".join(pieces)

    parsed = parse_coordjson(
        response_text,
        mode="salvage",
        object_field_order=resolved_field_order,
    )

    if bool(parsed.parse_failed) or parsed.objects_array_open_cut is None:
        prefix_token_ids = [
            int(t) for t in tokenizer.encode(_EMPTY_PREFIX_TEXT, add_special_tokens=False)
        ]
        return RolloutParseResult(
            response_token_ids=[int(t) for t in response_token_ids],
            response_text=response_text,
            prefix_token_ids=prefix_token_ids,
            prefix_text=_EMPTY_PREFIX_TEXT,
            invalid_rollout=True,
            valid_objects=[],
            dropped_invalid=0,
            dropped_invalid_by_reason={},
            dropped_ambiguous=0,
            truncated=bool(parsed.truncated),
        )

    cut_candidates = [int(parsed.objects_array_open_cut)]
    cut_candidates.extend(int(v) for v in parsed.record_end_cuts if int(v) > 0)
    cut_char = max(cut_candidates) if cut_candidates else int(parsed.objects_array_open_cut)

    prefix_token_ids = _build_prefix_from_char_cut(
        tokenizer=tokenizer,
        token_ids=response_token_ids,
        pieces=pieces,
        token_start_chars=token_start_chars,
        cut_char_pos=int(cut_char),
    )
    prefix_text = tokenizer.decode(
        prefix_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if not _is_append_ready_prefix(prefix_text):
        fallback_ids = [
            int(t)
            for t in tokenizer.encode(_EMPTY_PREFIX_TEXT, add_special_tokens=False)
        ]
        return RolloutParseResult(
            response_token_ids=[int(t) for t in response_token_ids],
            response_text=response_text,
            prefix_token_ids=fallback_ids,
            prefix_text=_EMPTY_PREFIX_TEXT,
            invalid_rollout=True,
            valid_objects=[],
            dropped_invalid=0,
            dropped_invalid_by_reason={},
            dropped_ambiguous=0,
            truncated=bool(parsed.truncated),
        )

    dropped_invalid = int(parsed.dropped_invalid_records)
    dropped_invalid_by_reason: Dict[str, int] = dict(parsed.dropped_invalid_by_reason)
    dropped_ambiguous = 0

    valid_objects: List[ParsedPredObject] = []

    def _coord_token_indices_for_span(span_start: int, span_end: int) -> List[int]:
        out: List[int] = []
        for idx, (tok_id, tok_start, piece) in enumerate(
            zip(response_token_ids, token_start_chars, pieces)
        ):
            tok_end = int(tok_start) + int(len(piece))
            if tok_end <= int(span_start) or int(tok_start) >= int(span_end):
                continue
            if int(tok_id) in coord_id_set:
                out.append(int(idx))
        return out

    for record in parsed.records:
        if int(record.record_span[1]) > int(cut_char):
            continue
        coord_indices = _coord_token_indices_for_span(
            int(record.geometry_span[0]),
            int(record.geometry_span[1]),
        )
        if len(coord_indices) != len(record.geometry_values):
            dropped_ambiguous += 1
            dropped_invalid += 1
            dropped_invalid_by_reason["wrong_arity"] = (
                int(dropped_invalid_by_reason.get("wrong_arity", 0)) + 1
            )
            continue
        valid_objects.append(
            ParsedPredObject(
                key=f"objects[{int(record.index)}]",
                index=int(record.index),
                desc=str(record.desc),
                geom_type=record.geometry_key,
                coord_token_indices=[int(i) for i in coord_indices],
                value_span=(int(record.record_span[0]), int(record.record_span[1])),
            )
        )

    invalid_rollout = bool(_non_ws_last_char(prefix_text) not in {"[", "}"})
    if invalid_rollout:
        fallback_ids = [
            int(t)
            for t in tokenizer.encode(_EMPTY_PREFIX_TEXT, add_special_tokens=False)
        ]
        return RolloutParseResult(
            response_token_ids=[int(t) for t in response_token_ids],
            response_text=response_text,
            prefix_token_ids=fallback_ids,
            prefix_text=_EMPTY_PREFIX_TEXT,
            invalid_rollout=True,
            valid_objects=[],
            dropped_invalid=0,
            dropped_invalid_by_reason={},
            dropped_ambiguous=0,
            truncated=bool(parsed.truncated),
        )

    return RolloutParseResult(
        response_token_ids=[int(t) for t in response_token_ids],
        response_text=response_text,
        prefix_token_ids=[int(t) for t in prefix_token_ids],
        prefix_text=str(prefix_text),
        invalid_rollout=False,
        valid_objects=valid_objects,
        dropped_invalid=int(dropped_invalid),
        dropped_invalid_by_reason=dropped_invalid_by_reason,
        dropped_ambiguous=int(dropped_ambiguous),
        truncated=bool(parsed.truncated),
    )


def points_from_coord_tokens(
    *,
    response_token_ids: Sequence[int],
    coord_token_indices: Sequence[int],
    coord_id_to_bin: Mapping[int, int],
) -> Optional[List[int]]:
    out: List[int] = []
    for idx in coord_token_indices:
        if idx < 0 or idx >= len(response_token_ids):
            return None
        tok_id = int(response_token_ids[int(idx)])
        bin_idx = coord_id_to_bin.get(tok_id)
        if bin_idx is None:
            return None
        out.append(int(bin_idx))
    return out


def serialize_append_fragment(
    *,
    fn_objects: Sequence[GTObject],
    prefix_text: str,
    object_field_order: str = "desc_first",
) -> str:
    resolved_field_order = normalize_object_field_order(
        object_field_order,
        path="custom.object_field_order",
    )
    tail = str(prefix_text).rstrip()
    if not tail:
        raise ValueError("empty rollout prefix is not append-ready")

    lead = tail.lstrip()
    objects_key_pos = lead.find('"objects"')
    objects_array_pos = lead.find("[", int(objects_key_pos) + len('"objects"'))
    if not lead.startswith("{") or objects_key_pos < 0 or objects_array_pos < 0:
        raise ValueError("rollout prefix is not append-ready for an objects array")

    last = tail[-1]
    if last not in {"[", "}", ","}:
        raise ValueError(f"rollout prefix is not append-ready; last_char={last!r}")

    try:
        parse_coordjson(
            tail,
            mode="strict",
            object_field_order=resolved_field_order,
        )
    except Exception:
        pass
    else:
        raise ValueError("rollout prefix is already a closed CoordJSON container")

    if not fn_objects:
        if last == ",":
            raise ValueError(
                "rollout prefix ends with ',' but FN set is empty; invalid JSON"
            )
        return "]}"

    leading = ""
    if last == "}":
        leading = ", "

    entries: List[str] = []
    for obj in fn_objects:
        if obj.geom_type == "bbox_2d":
            if len(obj.points_norm1000) != 4:
                continue
            geometry_key = "bbox_2d"
            geometry_value = [f"<|coord_{int(v)}|>" for v in obj.points_norm1000]
        elif obj.geom_type == "poly":
            points = list(obj.points_norm1000)
            if len(points) < 6 or len(points) % 2 != 0:
                continue
            geometry_key = "poly"
            geometry_value = [f"<|coord_{int(v)}|>" for v in points]
        else:
            continue

        payload: Dict[str, Any] = build_object_payload(
            desc=str(obj.desc),
            geometry_key=geometry_key,
            geometry_value=geometry_value,
            object_field_order=resolved_field_order,
        )
        entries.append(dumps_coordjson(payload))

    return leading + ", ".join(entries) + "]}"


def find_desc_value_char_spans(text: str) -> List[Tuple[int, int]]:
    """Return [start,end) spans of desc value content."""

    spans: List[Tuple[int, int]] = []
    i = 0
    needle = '"desc"'
    n = len(text)
    while i < n:
        k = text.find(needle, i)
        if k < 0:
            break
        j = k + len(needle)
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != ":":
            i = k + 1
            continue
        j += 1
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != '"':
            i = k + 1
            continue
        j += 1
        start = j
        esc = False
        while j < n:
            ch = text[j]
            if esc:
                esc = False
                j += 1
                continue
            if ch == "\\":
                esc = True
                j += 1
                continue
            if ch == '"':
                spans.append((start, j))
                j += 1
                break
            j += 1
        i = max(j, k + 1)
    return spans


def find_desc_value_token_positions(
    *, tokenizer: Any, token_ids: Sequence[int]
) -> List[int]:
    """Return token indices overlapping desc-value spans."""

    ids = [int(t) for t in token_ids]
    pieces = decode_pieces(tokenizer, ids)
    token_start_chars = _build_token_char_offsets(pieces)
    text = "".join(pieces)
    spans = find_desc_value_char_spans(text)
    if not spans:
        return []
    out: List[int] = []
    for ti, (start, piece) in enumerate(zip(token_start_chars, pieces)):
        end = start + len(piece)
        for s, e in spans:
            if start < e and end > s:
                out.append(int(ti))
                break
    return out


__all__ = [
    "coerce_int",
    "decode_pieces",
    "parse_rollout_for_matching",
    "points_from_coord_tokens",
    "serialize_append_fragment",
    "find_desc_value_char_spans",
    "find_desc_value_token_positions",
]

