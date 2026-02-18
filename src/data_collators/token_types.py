"""Helpers for token-type telemetry (desc/coord/format) on assistant payloads.

Ported from Qwen3-VL and adapted for CoordExp (aggregate metrics, packing-aware).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from src.coord_tokens.codec import is_coord_token
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenType:
    IGNORE = -1
    DESC = 1
    COORD = 2
    FORMAT = 3


def _format_payload_for_text(payload: Any) -> Any:
    """Match the JSON formatting used in dataset builders for assistant_text.

    Token-type alignment must mirror the serialized assistant_text exactly;
    otherwise token ids may not match supervised labels.
    """

    if not isinstance(payload, dict):
        return payload

    formatted: dict[str, Any] = {}
    for key, entry in payload.items():
        if str(key) == "objects" and isinstance(entry, list):
            out_items: list[Any] = []
            for item in entry:
                if not isinstance(item, dict):
                    out_items.append(item)
                    continue
                formatted_entry: dict[str, Any] = {}
                for field, value in item.items():
                    if field in {"poly", "bbox_2d"} and isinstance(value, list):
                        formatted_entry[field] = list(value)
                    else:
                        formatted_entry[field] = value
                out_items.append(formatted_entry)
            formatted[str(key)] = out_items
            continue
        if not isinstance(entry, dict):
            formatted[str(key)] = entry
            continue
        formatted_entry: dict[str, Any] = {}
        for field, value in entry.items():
            if field in {"poly", "bbox_2d"} and isinstance(value, list):
                formatted_entry[field] = list(value)
            else:
                formatted_entry[field] = value
        formatted[str(key)] = formatted_entry

    return formatted


def _dumps_with_types(payload: Any) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Serialize payload to JSON text and collect typed character spans.

    Span tuple: (start, end, type_id) in character offsets.
    """

    spans: List[Tuple[int, int, int]] = []
    parts: List[str] = []
    cursor = 0

    def write(text: str, typ: int) -> None:
        nonlocal cursor
        start = cursor
        parts.append(text)
        cursor += len(text)
        end = cursor
        if end > start:
            spans.append((start, end, typ))

    def emit_value(value: Any, context: str) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            text = json.dumps(value, ensure_ascii=False)
            write(text, TokenType.COORD if context == "coord" else TokenType.FORMAT)
        elif isinstance(value, str):
            if context == "coord" and is_coord_token(value):
                # CoordJSON uses bare CoordTok literals in geometry arrays.
                write(value, TokenType.COORD)
            else:
                text = json.dumps(value, ensure_ascii=False)
                write(text, TokenType.DESC if context == "desc" else TokenType.FORMAT)
        elif isinstance(value, list):
            write("[", TokenType.FORMAT)
            for idx, item in enumerate(value):
                if idx > 0:
                    write(", ", TokenType.FORMAT)
                emit_value(item, context)
            write("]", TokenType.FORMAT)
        elif isinstance(value, dict):
            write("{", TokenType.FORMAT)
            for idx, (k, v) in enumerate(value.items()):
                if idx > 0:
                    write(", ", TokenType.FORMAT)
                key_text = json.dumps(k, ensure_ascii=False)
                write(key_text, TokenType.FORMAT)
                write(": ", TokenType.FORMAT)
                next_ctx = (
                    "desc"
                    if k == "desc"
                    else "coord"
                    if k in {"bbox_2d", "poly"}
                    else "format"
                )
                emit_value(v, next_ctx)
            write("}", TokenType.FORMAT)
        else:
            # Fallback for unexpected types (booleans/null) -> format
            text = json.dumps(value, ensure_ascii=False)
            write(text, TokenType.FORMAT)

    emit_value(payload, "format")
    text_out = "".join(parts)
    return text_out, spans


def _apply_suffix(
    text: str, suffix: Iterable[str] | None
) -> Tuple[str, List[Tuple[int, int, int]]]:
    if not suffix:
        return text, []
    suffix_text = "".join(suffix)
    if not suffix_text:
        return text, []
    start = len(text)
    end = start + len(suffix_text)
    return text + suffix_text, [(start, end, TokenType.FORMAT)]


def _char_types_from_spans(length: int, spans: Sequence[Tuple[int, int, int]]) -> List[int]:
    arr = [TokenType.FORMAT] * length
    for start, end, typ in spans:
        for i in range(start, min(end, length)):
            arr[i] = typ
    return arr


def compute_token_types(
    *,
    tokenizer: PreTrainedTokenizerBase,
    payload: Any,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    suffix_tokens: Iterable[str] | None = None,
) -> torch.Tensor | None:
    """Compute per-token types aligned to labels.

    Returns a 1D tensor (seq_len) with types or None on mismatch.
    """

    if labels.dim() != 1:
        raise ValueError("labels must be 1D for a single sample")

    supervised_mask = labels != -100
    supervised_count = int(supervised_mask.sum().detach().item())
    if supervised_count == 0:
        return torch.full(
            labels.shape, TokenType.IGNORE, dtype=torch.long, device=labels.device
        )

    payload = _format_payload_for_text(payload)

    text, spans = _dumps_with_types(payload)
    text, suffix_spans = _apply_suffix(text, suffix_tokens)
    spans = spans + suffix_spans

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=supervised_count,
    )
    offsets = enc.get("offset_mapping")
    input_ids = enc.get("input_ids")
    if offsets is None:
        return None
    if input_ids is None:
        return None
    if len(offsets) != len(input_ids):
        return None

    max_end = 0
    for start, end in offsets:
        if end is None:
            continue
        if end > max_end:
            max_end = int(end)
    char_types = _char_types_from_spans(max_end, spans)
    token_types: List[int] = []
    for start, end in offsets:
        if start is None or end is None:
            token_types.append(TokenType.FORMAT)
            continue
        if end <= start or start < 0:
            token_types.append(TokenType.FORMAT)
            continue
        slice_types = char_types[start : min(end, len(char_types))]
        # Majority vote; fallback to first if empty
        if slice_types:
            counts = {t: slice_types.count(t) for t in set(slice_types)}
            token_types.append(max(counts, key=counts.get))
        else:
            token_types.append(TokenType.FORMAT)

    supervised_ids = labels[supervised_mask].tolist()

    def _extra_supervised_ids_ok(extra: List[int]) -> bool:
        if not extra:
            return True
        # Only tolerate a small tail of special tokens (e.g. eos) beyond payload text.
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            special_ids.add(int(eos_id))
        if len(extra) > 8:
            return False
        return all(int(v) in special_ids for v in extra)

    aligned: List[int] | None = None
    if len(token_types) == supervised_count:
        if list(input_ids) == supervised_ids:
            aligned = token_types
    elif len(token_types) < supervised_count:
        # Text shorter than supervision; allow a small special-token tail.
        if supervised_ids[: len(token_types)] == list(input_ids) and _extra_supervised_ids_ok(
            supervised_ids[len(token_types) :]
        ):
            aligned = token_types + [TokenType.FORMAT] * (supervised_count - len(token_types))

    if aligned is None:
        logger.debug(
            "Token-type alignment failed (tokenized=%d supervised=%d); skipping token-type metrics.",
            len(token_types),
            supervised_count,
        )
        return None

    full_types = torch.full(
        labels.shape, TokenType.IGNORE, dtype=torch.long, device=labels.device
    )
    full_types[supervised_mask] = torch.as_tensor(
        aligned, dtype=torch.long, device=labels.device
    )
    return full_types


__all__ = ["TokenType", "compute_token_types"]
