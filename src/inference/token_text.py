"""Map parser character evidence to literal generated tokens without retokenizing.

This pure operation owns decoding and exact boundaries, not object validity,
duplicate selection, teacher masks or loss semantics.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def decode_literal_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode ids without changing their order or passing text through encode."""

    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must expose decode(token_ids, ...) for exact history mapping")
    ids = [int(token_id) for token_id in token_ids]
    try:
        text = decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # A small fake tokenizer used by CPU tests may expose only the common
        # ``skip_special_tokens`` argument.  This fallback still decodes ids;
        # it never invokes tokenizer.encode/tokenize on generated text.
        text = decode(ids, skip_special_tokens=False)
    if not isinstance(text, str):
        text = str(text)
    return text


def exact_token_text_frame(
    action_ids: Sequence[int], tokenizer: Any,
) -> tuple[str, list[tuple[int, int]]]:
    """Return decoded text and exact character ranges for each original id.

    The parser's character evidence is meaningful only in the same frame as
    the generated token sequence.  Requiring the per-token decode join to
    equal the full decode makes a tokenizer-frame mismatch fail closed rather
    than silently assigning spans to a retokenized/history-altered sequence.
    """

    ids = [int(token_id) for token_id in action_ids]
    full_text = decode_literal_ids(tokenizer, ids)
    pieces = [decode_literal_ids(tokenizer, [token_id]) for token_id in ids]
    joined = "".join(pieces)
    if joined != full_text:
        raise ValueError(
            "tokenizer per-token decode does not reconstruct full decode; "
            "cannot map parser spans to original action-token history"
        )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        end = cursor + len(piece)
        spans.append((cursor, end))
        cursor = end
    return full_text, spans


def character_span_to_token_interval(
    char_start: Any,
    char_end: Any,
    *,
    text: str,
    token_spans: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """Map an exact parser character range to an original token interval."""

    if (
        isinstance(char_start, bool)
        or not isinstance(char_start, int)
        or isinstance(char_end, bool)
        or not isinstance(char_end, int)
        or char_start < 0
        or char_end <= char_start
        or char_end > len(text)
        or text[char_start:char_end] == ""
    ):
        raise ValueError("parser character span is invalid")

    # Empty decoded token pieces cannot provide a trustworthy boundary.  Do
    # not pick an arbitrary duplicate boundary if a tokenizer exposes one.
    start_matches = [
        index
        for index, (start, end) in enumerate(token_spans)
        if end > start and start == char_start
    ]
    end_matches = [
        index + 1
        for index, (start, end) in enumerate(token_spans)
        if end > start and end == char_end
    ]
    if len(start_matches) != 1 or len(end_matches) != 1:
        raise ValueError(
            "parser character span does not align to unique original token boundaries"
        )
    token_start, token_end = start_matches[0], end_matches[0]
    if token_end <= token_start:
        raise ValueError("parser character span maps to an empty token interval")
    if token_spans[token_start][0] != char_start or token_spans[token_end - 1][1] != char_end:
        raise ValueError("parser character span mapping is not contiguous")
    return token_start, token_end
