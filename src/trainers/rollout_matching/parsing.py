"""Rollout parsing helpers and contracts.

This module is intentionally import-light (no trainer/framework imports) so Stage-2
and tests can consume strict parsing + masking helpers without depending on
`src.trainers.rollout_matching_sft` implementation details.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.common.object_field_order import (
    build_object_payload,
    normalize_object_field_order,
)
from src.common.geometry import flatten_points
from src.coord_tokens.codec import get_coord_token_ids, token_to_int, value_in_coord_range

from .contracts import GTObject, GeomType, ParsedPredObject, RolloutParseResult


_OBJECT_KEY_RE = re.compile(r"^object_(\d+)$")
_IM_END = "<|im_end|>"


def coerce_int(value: Any) -> Optional[int]:
    try:
        v = int(round(float(value)))
    except Exception:
        return None
    if not value_in_coord_range(v):
        return None
    return v


def decode_pieces(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    # Token-level decode (no cleanup) to preserve exact token boundaries.
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
    if cut_char_pos >= (token_start_chars[-1] + len(pieces[-1])):
        return list(token_ids)

    # Find token i such that token_start_chars[i] <= cut < token_start_chars[i+1]
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
    i = lo
    start = token_start_chars[i]
    offset = max(0, min(int(cut_char_pos - start), len(pieces[i])))

    prefix = list(token_ids[:i])
    if offset == 0:
        return prefix
    if offset == len(pieces[i]):
        prefix.append(int(token_ids[i]))
        return prefix

    piece_prefix = pieces[i][:offset]
    # Replace ONLY the final token with a shorter tokenization (allowed by spec).
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


def _scan_rollout_tokens(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
    coord_id_set: set[int],
) -> Tuple[List[ParsedPredObject], Optional[int], int, bool, int]:
    """Token-aligned rollout scanner.

    Returns:
      (objects_raw, max_object_index, first_open_brace_pos, truncated, last_object_end_pos)

    Notes:
    - Objects are returned in appearance order.
    - Validation is performed later using the captured value_span.
    """

    pieces = decode_pieces(tokenizer, response_token_ids)
    token_start_chars: List[int] = []
    cursor = 0
    for p in pieces:
        token_start_chars.append(cursor)
        cursor += len(p)

    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape = False
    expecting_key: Dict[int, bool] = {}

    current_string: List[str] = []
    last_key: Optional[str] = None

    pending_object_key: Optional[Tuple[str, int]] = None
    current_object: Optional[ParsedPredObject] = None
    # Geometry capture state (per current_object)
    capture_active = False
    capture_target_depth: Optional[int] = None

    objects: List[ParsedPredObject] = []
    max_index: Optional[int] = None
    first_open_brace_pos: int = -1
    last_complete_object_end: int = -1

    # Map from token index -> current capture_active state is handled by scanning
    # per token; coord tokens do not contain structural characters, so the state at
    # token-start is sufficient.

    global_char_pos = 0
    for tok_i, (tok_id, piece) in enumerate(zip(response_token_ids, pieces)):
        # Coord-token capture happens at token granularity.
        if (
            capture_active
            and int(tok_id) in coord_id_set
            and current_object is not None
        ):
            current_object.coord_token_indices.append(int(tok_i))

        for ch in piece:
            if in_string:
                if escape:
                    escape = False
                    current_string.append(ch)
                elif ch == "\\":
                    escape = True
                    current_string.append(ch)
                elif ch == '"':
                    in_string = False
                    s = "".join(current_string)
                    current_string = []
                    if expecting_key.get(brace_depth, False) and bracket_depth == 0:
                        last_key = s
                        if brace_depth == 1:
                            m = _OBJECT_KEY_RE.match(s)
                            if m:
                                n = int(m.group(1))
                                if max_index is None or n > max_index:
                                    max_index = n
                                pending_object_key = (s, n)
                        # For brace_depth==2, last_key is used to arm geometry capture.
                else:
                    current_string.append(ch)
                global_char_pos += 1
                continue

            # Not in string
            if ch == '"':
                in_string = True
                escape = False
                current_string = []
                global_char_pos += 1
                continue

            if ch == "{":
                brace_depth += 1
                expecting_key[brace_depth] = True
                if brace_depth == 1 and first_open_brace_pos < 0:
                    first_open_brace_pos = global_char_pos + 1
                elif brace_depth == 2:
                    if pending_object_key is None:
                        # Nested dict without an object key -> invalid; keep scanning.
                        current_object = None
                    else:
                        key, n = pending_object_key
                        pending_object_key = None
                        current_object = ParsedPredObject(
                            key=key,
                            index=n,
                            desc="",
                            geom_type="bbox_2d",  # placeholder until validated
                            coord_token_indices=[],
                            value_span=(global_char_pos, -1),
                        )
                        objects.append(current_object)
                        last_key = None
                        capture_active = False
                        capture_target_depth = None
                elif brace_depth > 2:
                    # Nested objects are unsupported in strict parsing.
                    # Keep scanning to find stable boundaries, but we will drop later.
                    pass
                global_char_pos += 1
                continue

            if ch == "}":
                if brace_depth == 2 and current_object is not None:
                    # End of current object value dict.
                    current_object.value_span = (
                        current_object.value_span[0],
                        global_char_pos + 1,
                    )
                    last_complete_object_end = global_char_pos + 1
                    current_object = None
                    last_key = None
                    capture_active = False
                    capture_target_depth = None
                if brace_depth > 0:
                    expecting_key.pop(brace_depth, None)
                    brace_depth -= 1
                global_char_pos += 1
                continue

            if ch == "[":
                bracket_depth += 1
                if (
                    brace_depth == 2
                    and current_object is not None
                    and last_key in {"bbox_2d", "poly"}
                    and bracket_depth >= 1
                ):
                    # Arm capture on first array after the geometry key.
                    if capture_target_depth is None:
                        capture_active = True
                        capture_target_depth = bracket_depth - 1
                global_char_pos += 1
                continue

            if ch == "]":
                if bracket_depth > 0:
                    bracket_depth -= 1
                if capture_active and capture_target_depth is not None:
                    if bracket_depth <= capture_target_depth:
                        capture_active = False
                        capture_target_depth = None
                global_char_pos += 1
                continue

            if ch == ",":
                if brace_depth >= 1 and bracket_depth == 0:
                    expecting_key[brace_depth] = True
                global_char_pos += 1
                continue

            if ch == ":":
                if brace_depth >= 1 and bracket_depth == 0:
                    expecting_key[brace_depth] = False
                global_char_pos += 1
                continue

            global_char_pos += 1

    truncated = False
    if brace_depth != 0 or bracket_depth != 0 or in_string:
        truncated = True

    return objects, max_index, first_open_brace_pos, truncated, last_complete_object_end


def _validate_objects_strict(
    *,
    tokenizer: Any,
    response_text: str,
    objects_raw: Sequence[ParsedPredObject],
    prefix_char_end: int,
) -> Tuple[List[ParsedPredObject], int, Dict[str, int]]:
    """Strict validation: drop malformed objects (no repair).

    Returns:
      valid: objects that pass strict parsing/shape constraints
      dropped: total count of dropped objects
      dropped_by_reason: coarse reason buckets (diagnostics)
    """

    valid: List[ParsedPredObject] = []
    dropped = 0
    dropped_by_reason: Dict[str, int] = {}

    def _drop(reason: str) -> None:
        nonlocal dropped
        dropped += 1
        dropped_by_reason[str(reason)] = int(dropped_by_reason.get(str(reason), 0)) + 1

    for obj in objects_raw:
        start, end = obj.value_span
        if end <= 0 or end > prefix_char_end:
            _drop("key_invalid")
            continue
        snippet = response_text[start:end]
        try:
            parsed = json.loads(snippet)
        except Exception:
            _drop("key_invalid")
            continue
        if not isinstance(parsed, dict):
            _drop("key_invalid")
            continue

        desc = parsed.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            _drop("missing_desc")
            continue

        non_desc_keys = [k for k in parsed.keys() if k != "desc"]
        if not non_desc_keys:
            _drop("missing_geom")
            continue

        known_geom = {"bbox_2d", "poly"}
        geom_keys = [k for k in non_desc_keys if k in known_geom]
        extra_keys = [k for k in parsed.keys() if k not in {"desc", *known_geom}]

        if len(geom_keys) == 0:
            # Some kind of geometry was present but not recognized.
            _drop("unknown_geom")
            continue
        if len(geom_keys) != 1:
            _drop("key_invalid")
            continue
        if extra_keys:
            _drop("key_invalid")
            continue

        geom_key = geom_keys[0]

        # Geometry values must be coord tokens (strict; no ints in rollout-matching).
        flat = flatten_points(parsed.get(geom_key))
        if flat is None or len(flat) % 2 != 0:
            _drop("wrong_arity")
            continue
        token_bins: List[int] = []
        ok = True
        for v in flat:
            if not isinstance(v, str):
                ok = False
                break
            try:
                token_bins.append(int(token_to_int(str(v).strip())))
            except Exception:
                ok = False
                break
        if not ok:
            _drop("non_coord_token")
            continue

        # Ensure we captured coord-token indices and they match geometry arity.
        coord_idx = list(obj.coord_token_indices)
        if geom_key == "bbox_2d":
            if len(coord_idx) != 4 or len(token_bins) != 4:
                _drop("wrong_arity")
                continue
            # Basic bbox sanity: must be in-range and canonical (x2>=x1, y2>=y1).
            if any((t < 0) or (t > 999) for t in token_bins):
                _drop("bbox_invalid")
                continue
            x1, y1, x2, y2 = token_bins
            if x2 < x1 or y2 < y1:
                _drop("bbox_invalid")
                continue
        else:  # poly
            if (
                len(coord_idx) < 6
                or (len(coord_idx) % 2 != 0)
                or len(token_bins) != len(coord_idx)
            ):
                _drop("wrong_arity")
                continue

        valid.append(
            ParsedPredObject(
                key=obj.key,
                index=int(obj.index),
                desc=str(desc).strip(),
                geom_type=geom_key,  # type: ignore[assignment]
                coord_token_indices=coord_idx,
                value_span=obj.value_span,
            )
        )

    return valid, dropped, dropped_by_reason


def parse_rollout_for_matching(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
) -> RolloutParseResult:
    coord_token_ids = get_coord_token_ids(tokenizer)
    coord_id_set = set(int(t) for t in coord_token_ids if int(t) >= 0)

    # Decode full response text (token-aligned, no cleanup).
    pieces = decode_pieces(tokenizer, response_token_ids)
    token_start_chars: List[int] = []
    cursor = 0
    for p in pieces:
        token_start_chars.append(cursor)
        cursor += len(p)
    response_text = "".join(pieces)

    # Treat <|im_end|> as a hard stop (common in current offline rollouts).
    # This is suffix-only trimming; tokens before the cut remain unchanged, except for a
    # possible token-internal cut on the final token.
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
        token_start_chars = []
        cursor = 0
        for p in pieces:
            token_start_chars.append(cursor)
            cursor += len(p)
        response_text = "".join(pieces)

    objects_raw, _, first_open_brace_pos, truncated, last_obj_end = (
        _scan_rollout_tokens(
            tokenizer=tokenizer,
            response_token_ids=response_token_ids,
            coord_id_set=coord_id_set,
        )
    )

    # If completely malformed (no top-level '{'), fall back to empty prefix so FN append can proceed.
    if first_open_brace_pos < 0:
        prefix_token_ids = [
            int(t) for t in tokenizer.encode("{", add_special_tokens=False)
        ]
        prefix_text = tokenizer.decode(
            prefix_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return RolloutParseResult(
            response_token_ids=list(response_token_ids),
            response_text=response_text,
            prefix_token_ids=prefix_token_ids,
            prefix_text=prefix_text,
            max_object_index_in_prefix=None,
            valid_objects=[],
            dropped_invalid=0,
            dropped_invalid_by_reason={},
            dropped_ambiguous=0,
            truncated=True,
        )

    # Determine a safe prefix cut: keep up to last complete object, or just "{".
    if last_obj_end > 0:
        cut_char = last_obj_end
    elif first_open_brace_pos > 0:
        cut_char = first_open_brace_pos
    else:
        cut_char = 0

    prefix_token_ids = _build_prefix_from_char_cut(
        tokenizer=tokenizer,
        token_ids=response_token_ids,
        pieces=pieces,
        token_start_chars=token_start_chars,
        cut_char_pos=cut_char,
    )
    prefix_text = tokenizer.decode(
        prefix_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    valid_objects, dropped_invalid, dropped_invalid_by_reason = _validate_objects_strict(
        tokenizer=tokenizer,
        response_text=response_text,
        objects_raw=objects_raw,
        prefix_char_end=cut_char,
    )

    # In this MVP implementation, coord-slot alignment ambiguity is treated as invalid.
    dropped_ambiguous = 0

    # Key allocation must reserve ids based on all retained object_N keys in prefix,
    # including malformed/non-dict values that strict object validation drops.
    _, max_in_prefix, _, _, _ = _scan_rollout_tokens(
        tokenizer=tokenizer,
        response_token_ids=prefix_token_ids,
        coord_id_set=coord_id_set,
    )
    if max_in_prefix is not None:
        max_in_prefix = int(max_in_prefix)

    return RolloutParseResult(
        response_token_ids=list(response_token_ids),
        response_text=response_text,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        max_object_index_in_prefix=max_in_prefix,
        valid_objects=valid_objects,
        dropped_invalid=int(dropped_invalid),
        dropped_invalid_by_reason=dict(dropped_invalid_by_reason),
        dropped_ambiguous=int(dropped_ambiguous),
        truncated=bool(truncated),
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
    start_index: int,
    prefix_text: str,
    object_field_order: str = "desc_first",
) -> str:
    resolved_field_order = normalize_object_field_order(
        object_field_order, path="custom.object_field_order"
    )
    tail = prefix_text.rstrip()
    if not tail:
        raise ValueError("empty rollout prefix is not append-ready")

    last = tail[-1]
    if last not in {"{", "}", ","}:
        raise ValueError(f"rollout prefix is not append-ready; last_char={last!r}")

    # Deterministically validate the retained prefix as an open top-level JSON object
    # and detect whether it already contains at least one object entry.
    def _prefix_has_object_entries(text: str) -> bool:
        depth = 0
        in_string = False
        escape = False
        expecting_key: Dict[int, bool] = {}
        current_string: List[str] = []
        saw_root_open = False
        object_keys = 0

        for ch in text:
            if in_string:
                if escape:
                    escape = False
                    current_string.append(ch)
                elif ch == "\\":
                    escape = True
                    current_string.append(ch)
                elif ch == '"':
                    in_string = False
                    s = "".join(current_string)
                    current_string = []
                    if expecting_key.get(depth, False) and depth == 1:
                        if _OBJECT_KEY_RE.match(s):
                            object_keys += 1
                else:
                    current_string.append(ch)
                continue

            if ch == '"':
                in_string = True
                escape = False
                current_string = []
                continue

            if ch == "{":
                depth += 1
                expecting_key[depth] = True
                if depth == 1:
                    saw_root_open = True
                continue

            if ch == "}":
                if depth > 0:
                    expecting_key.pop(depth, None)
                    depth -= 1
                continue

            if ch == ",":
                if depth >= 1:
                    expecting_key[depth] = True
                continue

            if ch == ":":
                if depth >= 1:
                    expecting_key[depth] = False
                continue

        if not saw_root_open:
            raise ValueError("rollout prefix missing top-level '{'")
        if depth != 1:
            raise ValueError(
                "rollout prefix must end with an open top-level object before FN append"
            )
        if in_string:
            raise ValueError("rollout prefix ends inside a quoted string")

        return bool(object_keys > 0)

    has_entries = _prefix_has_object_entries(tail)

    if not fn_objects:
        if last == ",":
            raise ValueError(
                "rollout prefix ends with ',' but FN set is empty; invalid JSON"
            )
        return "}"

    leading = ""
    if has_entries and last != ",":
        leading = ", "

    entries: List[str] = []
    n = int(start_index)
    for obj in fn_objects:
        if obj.geom_type == "bbox_2d":
            if len(obj.points_norm1000) != 4:
                continue
            geometry_key = "bbox_2d"
            geometry_value = [f"<|coord_{int(v)}|>" for v in obj.points_norm1000]
        else:
            pts = obj.points_norm1000
            geometry_key = "poly"
            geometry_value = [
                [f"<|coord_{int(pts[i])}|>", f"<|coord_{int(pts[i + 1])}|>"]
                for i in range(0, len(pts), 2)
            ]
        payload: Dict[str, Any] = build_object_payload(
            desc=str(obj.desc),
            geometry_key=geometry_key,
            geometry_value=geometry_value,
            object_field_order=resolved_field_order,
        )
        entries.append(
            f'"object_{n}": {json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))}'
        )
        n += 1

    return leading + ", ".join(entries) + "}"


def find_desc_value_char_spans(text: str) -> List[Tuple[int, int]]:
    """Return [start,end) spans of the *value content* inside `"desc": "<VALUE>"`.

    Stage_2 rollout-matching intentionally ignores desc value supervision to avoid
    amplifying GT noise; this helper supports masking those tokens from CE.
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    needle = '"desc"'
    n = len(text)
    while i < n:
        k = text.find(needle, i)
        if k < 0:
            break
        j = k + len(needle)
        # Skip whitespace, then require ':'.
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != ":":
            i = k + 1
            continue
        j += 1
        while j < n and text[j].isspace():
            j += 1
        # Require opening quote of the string value.
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
    """Return token indices (0-based, relative to token_ids) overlapping desc-value spans."""
    ids = [int(t) for t in token_ids]
    pieces = decode_pieces(tokenizer, ids)
    token_start_chars: List[int] = []
    cursor = 0
    for p in pieces:
        token_start_chars.append(cursor)
        cursor += len(p)
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
