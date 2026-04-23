"""HF stop-pressure helpers for raw-text decode interventions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

_SPECIAL_TERMINATOR_TEXTS = ("<|im_end|>", "<|endoftext|>")


def build_terminating_token_suppression_logits_processor(
    *,
    tokenizer: object,
    prompt_lengths: Sequence[int],
    suppress_structural_close_tokens: bool,
    suppress_special_terminators: bool,
    fresh_boundary_only: bool,
) -> LogitsProcessorList:
    return LogitsProcessorList(
        [
            _RawTextObjectBoundaryTerminatingTokenSuppressor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                suppress_structural_close_tokens=suppress_structural_close_tokens,
                suppress_special_terminators=suppress_special_terminators,
                fresh_boundary_only=fresh_boundary_only,
            )
        ]
    )


def build_array_branch_continuation_steering_logits_processor(
    *,
    tokenizer: object,
    prompt_lengths: Sequence[int],
    continuation_bias: float,
) -> LogitsProcessorList:
    return LogitsProcessorList(
        [
            _RawTextFreshBoundaryArrayBranchContinuationSteerer(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(continuation_bias),
            )
        ]
    )


def build_bbox_tail_closure_steering_logits_processor(
    *,
    tokenizer: object,
    prompt_lengths: Sequence[int],
    continuation_bias: float,
) -> LogitsProcessorList:
    return LogitsProcessorList(
        [
            _RawTextBBoxTailClosureToNextObjectSteerer(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(continuation_bias),
            )
        ]
    )


def build_bbox_tail_then_object_open_steering_logits_processor(
    *,
    tokenizer: object,
    prompt_lengths: Sequence[int],
    continuation_bias: float,
) -> LogitsProcessorList:
    return LogitsProcessorList(
        [
            _RawTextBBoxTailThenObjectOpenSteerer(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(continuation_bias),
            )
        ]
    )


def build_bbox_tail_then_object_open_once_steering_logits_processor(
    *,
    tokenizer: object,
    prompt_lengths: Sequence[int],
    continuation_bias: float,
) -> LogitsProcessorList:
    return LogitsProcessorList(
        [
            _RawTextBBoxTailThenObjectOpenOnceSteerer(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                continuation_bias=float(continuation_bias),
            )
        ]
    )


class _RawTextObjectBoundaryTerminatingTokenSuppressor(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer: object,
        prompt_lengths: Sequence[int],
        suppress_structural_close_tokens: bool,
        suppress_special_terminators: bool,
        fresh_boundary_only: bool,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_lengths = tuple(int(value) for value in prompt_lengths)
        self._fresh_boundary_only = bool(fresh_boundary_only)
        self._suppressed_token_ids = tuple(
            sorted(
                _resolve_terminating_token_ids(
                    tokenizer=tokenizer,
                    suppress_structural_close_tokens=suppress_structural_close_tokens,
                    suppress_special_terminators=suppress_special_terminators,
                )
            )
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not self._suppressed_token_ids:
            return scores
        for row_idx in range(int(scores.shape[0])):
            prompt_len = self._prompt_lengths[min(row_idx, len(self._prompt_lengths) - 1)]
            generated_ids = input_ids[row_idx, int(prompt_len) :].detach().cpu().tolist()
            if not generated_ids:
                continue
            history_text = _decode_ids(tokenizer=self._tokenizer, token_ids=generated_ids)
            if self._fresh_boundary_only:
                if not _at_fresh_raw_text_object_boundary(history_text):
                    continue
            elif not _at_raw_text_object_boundary(history_text):
                continue
            for token_id in self._suppressed_token_ids:
                if 0 <= int(token_id) < int(scores.shape[1]):
                    scores[row_idx, int(token_id)] = float("-inf")
        return scores


class _RawTextFreshBoundaryArrayBranchContinuationSteerer(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer: object,
        prompt_lengths: Sequence[int],
        continuation_bias: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_lengths = tuple(int(value) for value in prompt_lengths)
        self._continuation_bias = float(continuation_bias)
        self._suppressed_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_array_close_prefix_token,
                )
            )
        )
        self._boosted_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_comma_continuation_token,
                )
            )
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not self._suppressed_token_ids and not self._boosted_token_ids:
            return scores
        for row_idx in range(int(scores.shape[0])):
            prompt_len = self._prompt_lengths[min(row_idx, len(self._prompt_lengths) - 1)]
            generated_ids = input_ids[row_idx, int(prompt_len) :].detach().cpu().tolist()
            if not generated_ids:
                continue
            history_text = _decode_ids(tokenizer=self._tokenizer, token_ids=generated_ids)
            if not _at_fresh_raw_text_object_boundary(history_text):
                continue
            for token_id in self._suppressed_token_ids:
                if 0 <= int(token_id) < int(scores.shape[1]):
                    scores[row_idx, int(token_id)] = float("-inf")
            for token_id in self._boosted_token_ids:
                if 0 <= int(token_id) < int(scores.shape[1]):
                    scores[row_idx, int(token_id)] = (
                        scores[row_idx, int(token_id)] + self._continuation_bias
                    )
        return scores


class _RawTextBBoxTailClosureToNextObjectSteerer(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer: object,
        prompt_lengths: Sequence[int],
        continuation_bias: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_lengths = tuple(int(value) for value in prompt_lengths)
        self._continuation_bias = float(continuation_bias)
        self._suppressed_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_bbox_tail_noncontinuation_close_token,
                )
            )
        )
        self._boosted_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_bbox_tail_next_object_token,
                )
            )
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not self._suppressed_token_ids and not self._boosted_token_ids:
            return scores
        for row_idx in range(int(scores.shape[0])):
            prompt_len = self._prompt_lengths[min(row_idx, len(self._prompt_lengths) - 1)]
            generated_ids = input_ids[row_idx, int(prompt_len) :].detach().cpu().tolist()
            if not generated_ids:
                continue
            history_text = _decode_ids(tokenizer=self._tokenizer, token_ids=generated_ids)
            if not _at_raw_text_bbox_tail_closure_branch(history_text):
                continue
            for token_id in self._suppressed_token_ids:
                if 0 <= int(token_id) < int(scores.shape[1]):
                    scores[row_idx, int(token_id)] = float("-inf")
            for token_id in self._boosted_token_ids:
                if 0 <= int(token_id) < int(scores.shape[1]):
                    scores[row_idx, int(token_id)] = (
                        scores[row_idx, int(token_id)] + self._continuation_bias
                    )
        return scores


class _RawTextBBoxTailThenObjectOpenSteerer(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer: object,
        prompt_lengths: Sequence[int],
        continuation_bias: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_lengths = tuple(int(value) for value in prompt_lengths)
        self._continuation_bias = float(continuation_bias)
        self._bbox_suppressed_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_bbox_tail_noncontinuation_close_token,
                )
            )
        )
        self._bbox_boosted_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_bbox_tail_next_object_token,
                )
            )
        )
        self._open_object_boosted_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_nonwhitespace_object_open_token,
                )
            )
        )
        self._wrong_schema_suppressed_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_nonwhitespace_quote_token,
                )
            )
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if (
            not self._bbox_suppressed_token_ids
            and not self._bbox_boosted_token_ids
            and not self._open_object_boosted_token_ids
            and not self._wrong_schema_suppressed_token_ids
        ):
            return scores
        for row_idx in range(int(scores.shape[0])):
            prompt_len = self._prompt_lengths[min(row_idx, len(self._prompt_lengths) - 1)]
            generated_ids = input_ids[row_idx, int(prompt_len) :].detach().cpu().tolist()
            if not generated_ids:
                continue
            history_text = _decode_ids(tokenizer=self._tokenizer, token_ids=generated_ids)
            if _at_raw_text_bbox_tail_closure_branch(history_text):
                for token_id in self._bbox_suppressed_token_ids:
                    if 0 <= int(token_id) < int(scores.shape[1]):
                        scores[row_idx, int(token_id)] = float("-inf")
                for token_id in self._bbox_boosted_token_ids:
                    if 0 <= int(token_id) < int(scores.shape[1]):
                        scores[row_idx, int(token_id)] = (
                            scores[row_idx, int(token_id)] + self._continuation_bias
                        )
            elif _at_raw_text_post_bbox_tail_object_open_branch(history_text):
                for token_id in self._wrong_schema_suppressed_token_ids:
                    if 0 <= int(token_id) < int(scores.shape[1]):
                        scores[row_idx, int(token_id)] = float("-inf")
                for token_id in self._open_object_boosted_token_ids:
                    if 0 <= int(token_id) < int(scores.shape[1]):
                        scores[row_idx, int(token_id)] = (
                            scores[row_idx, int(token_id)] + self._continuation_bias
                        )
        return scores


@dataclass
class _BBoxTailThenObjectOpenOnceState:
    bbox_tail_triggered: bool = False
    followup_open_armed: bool = False
    completed: bool = False


class _RawTextBBoxTailThenObjectOpenOnceSteerer(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer: object,
        prompt_lengths: Sequence[int],
        continuation_bias: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_lengths = tuple(int(value) for value in prompt_lengths)
        self._continuation_bias = float(continuation_bias)
        self._row_states: dict[int, _BBoxTailThenObjectOpenOnceState] = {}
        self._bbox_suppressed_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_bbox_tail_noncontinuation_close_token,
                )
            )
        )
        self._bbox_boosted_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_bbox_tail_next_object_token,
                )
            )
        )
        self._open_object_boosted_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_nonwhitespace_object_open_token,
                )
            )
        )
        self._wrong_schema_suppressed_token_ids = tuple(
            sorted(
                _resolve_token_ids_matching(
                    tokenizer=tokenizer,
                    predicate=_looks_like_nonwhitespace_quote_token,
                )
            )
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if (
            not self._bbox_suppressed_token_ids
            and not self._bbox_boosted_token_ids
            and not self._open_object_boosted_token_ids
            and not self._wrong_schema_suppressed_token_ids
        ):
            return scores
        for row_idx in range(int(scores.shape[0])):
            prompt_len = self._prompt_lengths[min(row_idx, len(self._prompt_lengths) - 1)]
            generated_ids = input_ids[row_idx, int(prompt_len) :].detach().cpu().tolist()
            if not generated_ids:
                continue
            state = self._row_states.setdefault(
                row_idx, _BBoxTailThenObjectOpenOnceState()
            )
            if state.completed:
                continue
            history_text = _decode_ids(tokenizer=self._tokenizer, token_ids=generated_ids)
            if state.followup_open_armed:
                if _at_raw_text_post_bbox_tail_object_open_branch(history_text):
                    for token_id in self._wrong_schema_suppressed_token_ids:
                        if 0 <= int(token_id) < int(scores.shape[1]):
                            scores[row_idx, int(token_id)] = float("-inf")
                    for token_id in self._open_object_boosted_token_ids:
                        if 0 <= int(token_id) < int(scores.shape[1]):
                            scores[row_idx, int(token_id)] = (
                                scores[row_idx, int(token_id)] + self._continuation_bias
                            )
                    continue
                state.followup_open_armed = False
                state.completed = True
                continue
            if state.bbox_tail_triggered:
                continue
            if not _at_raw_text_bbox_tail_closure_branch(history_text):
                continue
            for token_id in self._bbox_suppressed_token_ids:
                if 0 <= int(token_id) < int(scores.shape[1]):
                    scores[row_idx, int(token_id)] = float("-inf")
            for token_id in self._bbox_boosted_token_ids:
                if 0 <= int(token_id) < int(scores.shape[1]):
                    scores[row_idx, int(token_id)] = (
                        scores[row_idx, int(token_id)] + self._continuation_bias
                    )
            state.bbox_tail_triggered = True
            state.followup_open_armed = True
        return scores


def _resolve_terminating_token_ids(
    *,
    tokenizer: object,
    suppress_structural_close_tokens: bool,
    suppress_special_terminators: bool,
) -> set[int]:
    token_ids: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if suppress_special_terminators and isinstance(eos_token_id, int):
        token_ids.add(int(eos_token_id))
    elif suppress_special_terminators and isinstance(eos_token_id, (list, tuple)):
        token_ids.update(int(value) for value in eos_token_id if isinstance(value, int))

    eos_token = getattr(tokenizer, "eos_token", None)
    if suppress_special_terminators and isinstance(eos_token, str) and eos_token.strip():
        token_ids.update(_resolve_single_token_ids_for_text(tokenizer=tokenizer, text=eos_token))

    if suppress_special_terminators:
        for special_text in _SPECIAL_TERMINATOR_TEXTS:
            token_ids.update(
                _resolve_single_token_ids_for_text(tokenizer=tokenizer, text=special_text)
            )

    for token_id in _iter_token_ids(tokenizer=tokenizer):
        decoded = _decode_ids(tokenizer=tokenizer, token_ids=[int(token_id)])
        if suppress_special_terminators and any(
            special in decoded for special in _SPECIAL_TERMINATOR_TEXTS
        ):
            token_ids.add(int(token_id))
        elif suppress_structural_close_tokens and _looks_like_close_now_token(decoded):
            token_ids.add(int(token_id))

    return token_ids


def _iter_token_ids(*, tokenizer: object) -> Iterable[int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        seen: set[int] = set()
        vocab = get_vocab()
        if isinstance(vocab, dict):
            for token_id in vocab.values():
                try:
                    token_id_int = int(token_id)
                except (TypeError, ValueError):
                    continue
                if token_id_int not in seen:
                    seen.add(token_id_int)
                    yield token_id_int
        return

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        return
    try:
        vocab_size_int = int(vocab_size)
    except (TypeError, ValueError):
        return
    for token_id in range(vocab_size_int):
        yield token_id


def _resolve_single_token_ids_for_text(*, tokenizer: object, text: str) -> set[int]:
    token_ids: set[int] = set()
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        converted = convert_tokens_to_ids(text)
        if isinstance(converted, int) and converted >= 0:
            token_ids.add(int(converted))

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = encode(text, add_special_tokens=False)
        except TypeError:
            encoded = encode(text)
        if isinstance(encoded, list) and len(encoded) == 1:
            try:
                token_ids.add(int(encoded[0]))
            except (TypeError, ValueError):
                pass
    return token_ids


def _resolve_token_ids_matching(
    *,
    tokenizer: object,
    predicate: Callable[[str], bool],
) -> set[int]:
    token_ids: set[int] = set()
    for token_id in _iter_token_ids(tokenizer=tokenizer):
        decoded = _decode_ids(tokenizer=tokenizer, token_ids=[int(token_id)])
        if predicate(decoded):
            token_ids.add(int(token_id))
    return token_ids


def _decode_ids(*, tokenizer: object, token_ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer.decode is required for stop-pressure decoding")
    return str(
        decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _looks_like_close_now_token(text: str) -> bool:
    stripped = str(text).lstrip()
    if not stripped:
        return False
    for special_text in _SPECIAL_TERMINATOR_TEXTS:
        stripped = stripped.replace(special_text, "")
    stripped = stripped.strip()
    if not stripped or not stripped.startswith("]"):
        return False
    return set(stripped).issubset({"]", "}"})


def _looks_like_array_close_prefix_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return bool(stripped) and stripped.startswith("]")


def _looks_like_comma_continuation_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return bool(stripped) and stripped.startswith(",")


def _looks_like_bbox_tail_next_object_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return bool(stripped) and stripped.startswith("]},")


def _looks_like_bbox_tail_noncontinuation_close_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return (
        bool(stripped)
        and stripped.startswith("]")
        and not _looks_like_bbox_tail_next_object_token(stripped)
    )


def _looks_like_nonwhitespace_object_open_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return bool(stripped) and stripped.startswith("{")


def _looks_like_nonwhitespace_quote_token(text: str) -> bool:
    stripped = _strip_trailing_special_terminators(text).lstrip()
    return bool(stripped) and stripped.startswith('"')


def _strip_trailing_special_terminators(text: str) -> str:
    stripped = str(text)
    changed = True
    while changed:
        changed = False
        trimmed = stripped.rstrip()
        for special_text in _SPECIAL_TERMINATOR_TEXTS:
            if trimmed.endswith(special_text):
                stripped = trimmed[: -len(special_text)]
                changed = True
                break
    return stripped


def _at_raw_text_object_boundary(text: str) -> bool:
    at_boundary, _ = _raw_text_object_boundary_status(text)
    return at_boundary


def _at_fresh_raw_text_object_boundary(text: str) -> bool:
    at_boundary, boundary_dirty = _raw_text_object_boundary_status(text)
    return at_boundary and not boundary_dirty


def _at_raw_text_bbox_tail_closure_branch(text: str) -> bool:
    parsed_text = _strip_trailing_special_terminators(text)
    if not parsed_text:
        return False

    stack: list[dict[str, object]] = []
    in_string = False
    escaped = False
    string_chars: list[str] = []
    last_string: str | None = None

    for ch in parsed_text:
        if in_string:
            if escaped:
                string_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                string_chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
                last_string = "".join(string_chars)
                string_chars = []
                continue
            string_chars.append(ch)
            continue

        bbox_array = stack[-1] if stack else None
        if (
            bbox_array
            and bbox_array["type"] == "array"
            and bbox_array.get("kind") == "bbox_2d"
        ):
            if ch in "+-0123456789":
                if not bool(bbox_array.get("in_number", False)):
                    bbox_array["value_count"] = int(bbox_array.get("value_count", 0)) + 1
                    bbox_array["in_number"] = True
                continue
            if ch in ".eE":
                if bool(bbox_array.get("in_number", False)):
                    continue
            if bool(bbox_array.get("in_number", False)):
                bbox_array["in_number"] = False

        if ch == '"':
            in_string = True
            escaped = False
            string_chars = []
            continue
        if ch.isspace():
            continue
        if ch == ":":
            if stack and stack[-1]["type"] == "object" and last_string is not None:
                stack[-1]["pending_key"] = last_string
            last_string = None
            continue
        if ch == ",":
            if stack and stack[-1]["type"] == "object":
                stack[-1]["pending_key"] = None
            continue
        if ch == "[":
            parent = stack[-1] if stack else None
            kind = None
            if (
                parent
                and parent["type"] == "object"
                and parent.get("pending_key") == "objects"
                and len(stack) == 1
            ):
                kind = "objects"
            elif (
                parent
                and parent["type"] == "object"
                and parent.get("pending_key") == "bbox_2d"
                and parent.get("kind") == "objects_element"
            ):
                kind = "bbox_2d"
            stack.append(
                {
                    "type": "array",
                    "kind": kind,
                    "value_count": 0,
                    "in_number": False,
                }
            )
            if parent and parent["type"] == "object":
                parent["pending_key"] = None
            continue
        if ch == "{":
            parent = stack[-1] if stack else None
            kind = (
                "objects_element"
                if parent and parent["type"] == "array" and parent.get("kind") == "objects"
                else None
            )
            stack.append({"type": "object", "kind": kind, "pending_key": None})
            continue
        if ch == "}":
            if stack and stack[-1]["type"] == "object":
                stack.pop()
                parent = stack[-1] if stack else None
                if parent and parent["type"] == "object":
                    parent["pending_key"] = None
            continue
        if ch == "]":
            if stack and stack[-1]["type"] == "array":
                stack.pop()
                parent = stack[-1] if stack else None
                if parent and parent["type"] == "object":
                    parent["pending_key"] = None
            continue

    if not stack:
        return False
    top = stack[-1]
    if top["type"] != "array" or top.get("kind") != "bbox_2d":
        return False
    inside_objects_element = any(
        item["type"] == "object" and item.get("kind") == "objects_element"
        for item in stack
    )
    return inside_objects_element and int(top.get("value_count", 0)) >= 4


def _at_raw_text_post_bbox_tail_object_open_branch(text: str) -> bool:
    parsed_text = _strip_trailing_special_terminators(text)
    if not parsed_text or not re.search(r"\]\s*\}\s*,\s*$", parsed_text):
        return False

    stack: list[dict[str, object]] = []
    in_string = False
    escaped = False
    string_chars: list[str] = []
    last_string: str | None = None

    for ch in parsed_text:
        if in_string:
            if escaped:
                string_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                string_chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
                last_string = "".join(string_chars)
                string_chars = []
                continue
            string_chars.append(ch)
            continue

        if ch == '"':
            in_string = True
            escaped = False
            string_chars = []
            continue
        if ch.isspace():
            continue
        if ch == ":":
            if stack and stack[-1]["type"] == "object" and last_string is not None:
                stack[-1]["pending_key"] = last_string
            last_string = None
            continue
        if ch == ",":
            if stack and stack[-1]["type"] == "object":
                stack[-1]["pending_key"] = None
            continue
        if ch == "[":
            parent = stack[-1] if stack else None
            kind = None
            if (
                parent
                and parent["type"] == "object"
                and parent.get("pending_key") == "objects"
                and len(stack) == 1
            ):
                kind = "objects"
            elif (
                parent
                and parent["type"] == "object"
                and parent.get("pending_key") == "bbox_2d"
                and parent.get("kind") == "objects_element"
            ):
                kind = "bbox_2d"
            stack.append({"type": "array", "kind": kind})
            if parent and parent["type"] == "object":
                parent["pending_key"] = None
            continue
        if ch == "{":
            parent = stack[-1] if stack else None
            kind = (
                "objects_element"
                if parent and parent["type"] == "array" and parent.get("kind") == "objects"
                else None
            )
            stack.append({"type": "object", "kind": kind, "pending_key": None})
            continue
        if ch == "}":
            if stack and stack[-1]["type"] == "object":
                stack.pop()
                parent = stack[-1] if stack else None
                if parent and parent["type"] == "object":
                    parent["pending_key"] = None
            continue
        if ch == "]":
            if stack and stack[-1]["type"] == "array":
                stack.pop()
                parent = stack[-1] if stack else None
                if parent and parent["type"] == "object":
                    parent["pending_key"] = None
            continue

    inside_objects_array = any(
        item["type"] == "array" and item.get("kind") == "objects" for item in stack
    )
    inside_objects_element = any(
        item["type"] == "object" and item.get("kind") == "objects_element"
        for item in stack
    )
    return inside_objects_array and not inside_objects_element


def _raw_text_object_boundary_status(text: str) -> tuple[bool, bool]:
    parsed_text = _strip_trailing_special_terminators(text)
    if not parsed_text:
        return False, False

    stack: list[dict[str, object]] = []
    in_string = False
    escaped = False
    string_chars: list[str] = []
    last_string: str | None = None
    completed_objects = 0
    boundary_open = False
    boundary_dirty = False

    for ch in parsed_text:
        if in_string:
            if escaped:
                string_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                string_chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
                last_string = "".join(string_chars)
                string_chars = []
                continue
            string_chars.append(ch)
            continue

        if ch == '"':
            if boundary_open:
                boundary_dirty = True
            in_string = True
            escaped = False
            string_chars = []
            continue
        if ch.isspace():
            continue
        if boundary_open:
            boundary_dirty = True
        if ch == ":":
            if stack and stack[-1]["type"] == "object" and last_string is not None:
                stack[-1]["pending_key"] = last_string
            last_string = None
            continue
        if ch == ",":
            boundary_open = False
            if stack and stack[-1]["type"] == "object":
                stack[-1]["pending_key"] = None
            continue
        if ch == "[":
            parent = stack[-1] if stack else None
            kind = None
            if (
                parent
                and parent["type"] == "object"
                and parent.get("pending_key") == "objects"
                and len(stack) == 1
            ):
                kind = "objects"
            stack.append({"type": "array", "kind": kind})
            if parent and parent["type"] == "object":
                parent["pending_key"] = None
            boundary_open = False
            continue
        if ch == "{":
            parent = stack[-1] if stack else None
            kind = (
                "objects_element"
                if parent and parent["type"] == "array" and parent.get("kind") == "objects"
                else None
            )
            stack.append({"type": "object", "kind": kind, "pending_key": None})
            boundary_open = False
            continue
        if ch == "}":
            if stack and stack[-1]["type"] == "object":
                stack.pop()
                parent = stack[-1] if stack else None
                if parent and parent["type"] == "array" and parent.get("kind") == "objects":
                    completed_objects += 1
                    boundary_open = True
                    boundary_dirty = False
                else:
                    boundary_open = False
                if parent and parent["type"] == "object":
                    parent["pending_key"] = None
            else:
                boundary_open = False
            continue
        if ch == "]":
            boundary_open = False
            if stack and stack[-1]["type"] == "array":
                stack.pop()
                parent = stack[-1] if stack else None
                if parent and parent["type"] == "object":
                    parent["pending_key"] = None
            continue
        boundary_open = False

    inside_objects_array = any(
        item["type"] == "array" and item.get("kind") == "objects" for item in stack
    )
    inside_objects_element = any(
        item["type"] == "object" and item.get("kind") == "objects_element"
        for item in stack
    )
    at_boundary = bool(
        boundary_open
        and completed_objects > 0
        and inside_objects_array
        and not inside_objects_element
    )
    return at_boundary, boundary_dirty
