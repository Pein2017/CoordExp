from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.trainers.rollout_matching.contracts import GTObject
from src.trainers.rollout_matching.parsing import (
    decode_pieces,
    find_desc_value_char_spans,
    find_desc_value_token_positions,
)

_SEMANTIC_STOP_PIECE = "]}"
_SEMANTIC_CONTINUE_PIECE = "]},"
_IM_END = "<|im_end|>"


def _coerce_token_id_list(token_ids: Sequence[int]) -> List[int]:
    try:
        return [int(t) for t in token_ids]
    except (TypeError, ValueError):
        return [int(t) for t in list(token_ids)]


def _build_piece_offsets(pieces: Sequence[str]) -> List[int]:
    offsets: List[int] = []
    cursor = 0
    for piece in pieces:
        offsets.append(int(cursor))
        cursor += int(len(piece))
    return offsets


def _tok_indices_overlapping(
    *,
    token_start_chars: Sequence[int],
    pieces: Sequence[str],
    char_start: int,
    char_end: int,
) -> List[int]:
    out: List[int] = []
    for ti, (start, piece) in enumerate(zip(token_start_chars, pieces)):
        end = int(start) + int(len(piece))
        if end <= int(char_start) or int(start) >= int(char_end):
            continue
        out.append(int(ti))
    return out


def _find_top_level_close_char(text: str) -> int:
    depth = 0
    in_string = False
    escape = False
    close_char: Optional[int] = None
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            if depth == 1:
                close_char = int(i)
                break
            if depth > 0:
                depth -= 1
    if close_char is None:
        raise ValueError(
            "could not locate top-level JSON closing brace for closure supervision"
        )
    return int(close_char)


def _resolve_exact_piece_token_id(tokenizer: Any, piece: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            candidate = int(convert(piece))
        except (TypeError, ValueError):
            candidate = -1
        if candidate >= 0:
            decoded = tokenizer.decode(
                [candidate],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded == str(piece):
                return int(candidate)

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = [int(t) for t in encode(str(piece), add_special_tokens=False)]
        except TypeError:
            encoded = [int(t) for t in encode(str(piece))]
        if len(encoded) == 1:
            decoded = tokenizer.decode(
                encoded,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded == str(piece):
                return int(encoded[0])

    raise ValueError(
        f"tokenizer does not expose an exact single-token piece for semantic stop marker {piece!r}"
    )


def semantic_stop_branch_metadata(
    *,
    tokenizer: Any,
    assistant_span_ids: Sequence[int],
    prefix_len: int,
) -> Dict[str, Any]:
    """Resolve tokenizer-aware semantic stop metadata for dense object-list targets.

    Returns tail-local metadata for the first terminal semantic stop token `']}'`,
    the later closure-tail markers, and the exact stop/continue token ids used to
    form the branch-local damping calculation.
    """

    prefix_len_i = max(0, int(prefix_len))
    span_ids = _coerce_token_id_list(assistant_span_ids)
    tail_cap = max(0, int(len(span_ids)) - int(prefix_len_i))
    if tail_cap <= 0:
        raise ValueError("assistant span has no tail-local region for semantic stop discovery")

    pieces = decode_pieces(tokenizer, span_ids)
    token_start_chars = _build_piece_offsets(pieces)
    text = "".join(pieces)

    stop_token_id = _resolve_exact_piece_token_id(tokenizer, _SEMANTIC_STOP_PIECE)
    continue_token_id = _resolve_exact_piece_token_id(
        tokenizer,
        _SEMANTIC_CONTINUE_PIECE,
    )

    close_char = _find_top_level_close_char(text)
    close_toks = _tok_indices_overlapping(
        token_start_chars=token_start_chars,
        pieces=pieces,
        char_start=int(close_char),
        char_end=int(close_char) + 1,
    )
    if not close_toks:
        raise ValueError("could not map top-level JSON closing brace to token indices")

    top_close_tok = min(int(t) for t in close_toks)
    stop_piece_matches = [
        int(i) for i, piece in enumerate(pieces) if str(piece) == _SEMANTIC_STOP_PIECE
    ]
    stop_candidates = [int(i) for i in stop_piece_matches if int(i) < int(top_close_tok)]
    if not stop_candidates:
        raise ValueError(
            "could not locate a semantic stop token before the top-level closure tail"
        )

    stop_tok = int(stop_candidates[-1])
    stop_end = int(token_start_chars[stop_tok]) + int(len(pieces[stop_tok]))
    gap_text = str(text[int(stop_end) : int(token_start_chars[top_close_tok])]).strip()
    if gap_text:
        raise ValueError(
            "semantic stop token is not immediately followed by the closure tail"
        )

    im_toks: List[int] = []
    im_pos = int(text.find(_IM_END, int(close_char) + 1))
    if im_pos >= 0:
        im_toks = _tok_indices_overlapping(
            token_start_chars=token_start_chars,
            pieces=pieces,
            char_start=int(im_pos),
            char_end=int(im_pos) + int(len(_IM_END)),
        )
        if not im_toks:
            raise ValueError("could not map <|im_end|> to token indices")

    tail_closure_positions: List[int] = []
    for ti in list(close_toks) + list(im_toks):
        rel = int(ti) - int(prefix_len_i)
        if 0 <= rel < int(tail_cap):
            tail_closure_positions.append(int(rel))
    tail_closure_positions = sorted({int(p) for p in tail_closure_positions})

    stop_rel = int(stop_tok) - int(prefix_len_i)
    if stop_rel < 0 or stop_rel >= int(tail_cap):
        raise ValueError(
            f"semantic stop token escaped the tail-local region: rel={int(stop_rel)} tail_cap={int(tail_cap)}"
        )

    return {
        "stop_rel_pos": int(stop_rel),
        "tail_closure_pos": [int(p) for p in tail_closure_positions],
        "stop_token_id": int(stop_token_id),
        "continue_token_id": int(continue_token_id),
    }


def bbox_groups_from_token_ids(
    *,
    token_ids: Sequence[int],
    coord_id_set: set[int],
    gt_objs: Sequence[GTObject],
) -> List[List[int]]:
    coord_pos = [i for i, tid in enumerate(token_ids) if int(tid) in coord_id_set]
    exp = int(len(gt_objs)) * 4
    if len(coord_pos) != exp:
        raise ValueError(
            f"unexpected coord-token count in teacher-forced ids: got={len(coord_pos)} expected={exp}"
        )
    groups: List[List[int]] = []
    for i in range(0, len(coord_pos), 4):
        groups.append([int(p) for p in coord_pos[i : i + 4]])
    return groups


def matched_prefix_structure_positions(
    *,
    tokenizer: Any,
    prefix_token_ids: Sequence[int],
    prefix_text: str,
    matched_pred_objects: Sequence[Any],
) -> List[int]:
    try:
        token_ids = [int(t) for t in prefix_token_ids]
    except (TypeError, ValueError):
        token_ids = [int(t) for t in list(prefix_token_ids)]

    if not token_ids or not matched_pred_objects:
        return []

    pieces = decode_pieces(tokenizer, token_ids)
    prefix_scan_text = "".join(str(p) for p in pieces)
    _ = prefix_text

    token_start_chars: List[int] = []
    cursor = 0
    for p in pieces:
        token_start_chars.append(int(cursor))
        cursor += int(len(p))

    def _tok_indices_overlapping(char_start: int, char_end: int) -> List[int]:
        out: List[int] = []
        for ti, (start, piece) in enumerate(zip(token_start_chars, pieces)):
            end = int(start) + int(len(piece))
            if end <= int(char_start) or int(start) >= int(char_end):
                continue
            out.append(int(ti))
        return out

    supervised: set[int] = set()
    prefix_len_chars = int(len(prefix_scan_text))

    for obj in matched_pred_objects:
        if obj is None:
            continue

        value_span = getattr(obj, "value_span", None)
        if not isinstance(value_span, tuple) or len(value_span) != 2:
            raise ValueError(
                "matched object is missing value_span for prefix-structure supervision"
            )

        value_start = int(value_span[0])
        value_end = int(value_span[1])
        if value_start < 0 or value_end <= value_start or value_end > prefix_len_chars:
            raise ValueError(
                f"matched object value_span is outside retained prefix: {value_span!r}"
            )

        entry_tokens = _tok_indices_overlapping(int(value_start), int(value_end))
        if not entry_tokens:
            raise ValueError("could not map matched object entry span to prefix tokens")

        desc_tokens: set[int] = set()
        value_text = str(prefix_scan_text[int(value_start) : int(value_end)])
        for ds, de in find_desc_value_char_spans(value_text):
            desc_tokens.update(
                _tok_indices_overlapping(int(value_start + ds), int(value_start + de))
            )

        for ti in entry_tokens:
            if int(ti) not in desc_tokens:
                supervised.add(int(ti))

    return sorted(int(p) for p in supervised)


def tail_closure_positions(
    *,
    tokenizer: Any,
    assistant_span_ids: Sequence[int],
    prefix_len: int,
) -> List[int]:
    """Return tail-local closure supervision positions.

    This function is intentionally strict: it raises ValueError when it cannot locate
    a top-level JSON close brace in the assistant span. Callers that enforce strict
    rollout formatting should treat this as an invalid/malformed rollout and drop the
    sample rather than training on ambiguous supervision.
    """

    prefix_len_i = max(0, int(prefix_len))

    span_ids = _coerce_token_id_list(assistant_span_ids)

    tail_cap = max(0, int(len(span_ids)) - int(prefix_len_i))
    if tail_cap <= 0:
        return []

    pieces = decode_pieces(tokenizer, span_ids)
    token_start_chars = _build_piece_offsets(pieces)
    text = "".join(pieces)

    close_char = _find_top_level_close_char(text)
    close_toks = _tok_indices_overlapping(
        token_start_chars=token_start_chars,
        pieces=pieces,
        char_start=int(close_char),
        char_end=int(close_char) + 1,
    )
    if not close_toks:
        raise ValueError("could not map JSON closing brace to token indices")

    im_toks: List[int] = []
    im_pos = int(text.find(_IM_END, int(close_char) + 1))
    if im_pos >= 0:
        im_toks = _tok_indices_overlapping(
            token_start_chars=token_start_chars,
            pieces=pieces,
            char_start=int(im_pos),
            char_end=int(im_pos) + int(len(_IM_END)),
        )
        if not im_toks:
            raise ValueError("could not map <|im_end|> to token indices")

    closure_positions: List[int] = []
    for ti in close_toks:
        rel = int(ti) - int(prefix_len_i)
        if 0 <= rel < int(tail_cap):
            closure_positions.append(int(rel))
    for ti in im_toks:
        rel = int(ti) - int(prefix_len_i)
        if 0 <= rel < int(tail_cap):
            closure_positions.append(int(rel))

    closure_positions = sorted({int(p) for p in closure_positions})
    if not closure_positions:
        raise ValueError(
            "closure supervision produced no valid tail-local positions"
        )

    return closure_positions


def tail_desc_positions(*, tokenizer: Any, token_ids: Sequence[int]) -> List[int]:
    return [int(i) for i in (find_desc_value_token_positions(tokenizer=tokenizer, token_ids=token_ids) or [])]


def shift_bbox_groups(
    *,
    groups: Sequence[Mapping[str, Any]],
    delta_prompt: int,
    lower: int,
    upper: int,
    encoded_len: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, Mapping):
            continue
        pos = g.get("pos")
        gb = g.get("gt_bins")
        if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
            continue
        if len(pos) != 4 or len(gb) != 4:
            continue

        pos_i = [int(p) + int(delta_prompt) for p in pos]
        gb_i = [int(x) for x in gb]

        if any(p < int(lower) or p >= int(upper) for p in pos_i):
            raise ValueError(
                f"bbox group pos escaped expected span after prompt shift: pos={pos_i} span=[{int(lower)},{int(upper)})"
            )
        if any(p >= int(encoded_len) for p in pos_i):
            raise ValueError(
                f"bbox group pos exceeds encoded_len after prompt shift: pos={pos_i} encoded_len={int(encoded_len)}"
            )

        out.append({"pos": pos_i, "gt_bins": gb_i})

    return out
