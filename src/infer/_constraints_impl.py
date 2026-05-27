"""Decode-constraint implementations behind :mod:`src.infer.constraints`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from transformers import LogitsProcessor

from src.common.detection_sequence import (
    BOX_START_TOKEN,
    COMPACT_FULL_FORMAT,
    IM_END_TOKEN,
    OBJECT_REF_START_TOKEN,
    normalize_detection_sequence_format,
)


def _token_id(tokenizer: object, token: str) -> int | None:
    vocab_getter = getattr(tokenizer, "get_vocab", None)
    if callable(vocab_getter):
        vocab = vocab_getter()
        if isinstance(vocab, dict) and token in vocab:
            return int(vocab[token])
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        value = converter(token)
        if isinstance(value, int) and value >= 0:
            return int(value)
    return None


def _single_token_encoding(tokenizer: object, text: str) -> int | None:
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        return None
    ids = encoder(text, add_special_tokens=False)
    if isinstance(ids, Sequence) and len(ids) == 1:
        value = ids[0]
        if isinstance(value, int) and value >= 0:
            return int(value)
    return None


@dataclass(frozen=True)
class CompactFullGrammarIds:
    object_start_id: int
    box_start_id: int
    coord_ids: tuple[int, ...]
    newline_ids: tuple[int, ...]
    eos_ids: tuple[int, ...]


class CompactFullGrammarLogitsProcessor(LogitsProcessor):
    """Constrain compact-full rows only at structural token-type boundaries.

    The processor intentionally leaves description text unconstrained. It only
    enforces the minimal row grammar needed for parseable compact detections:

    ``<|object_ref_start|>{desc}<|box_start|>{coord}{coord}{coord}{coord}``

    After a bbox is complete it permits either a newline separator or the chat
    EOS token. At a fresh row boundary it permits a new object row or the chat
    EOS token. This is a decode-time grammar constraint, not a scoring/eval parser
    relaxation.
    """

    def __init__(
        self,
        *,
        ids: CompactFullGrammarIds,
        prompt_lengths: Sequence[int],
        force_row_start: bool = True,
    ) -> None:
        self.ids = ids
        self.prompt_lengths = tuple(max(0, int(v)) for v in prompt_lengths)
        self.force_row_start = bool(force_row_start)
        self.coord_id_set = set(int(v) for v in ids.coord_ids)
        self.newline_id_set = set(int(v) for v in ids.newline_ids)
        self.eos_id_set = set(int(v) for v in ids.eos_ids)
        self.row_start_allowed = tuple(dict.fromkeys((ids.object_start_id, *ids.eos_ids)))
        self.after_bbox_allowed = tuple(dict.fromkeys((*ids.newline_ids, *ids.eos_ids)))
        self._allowed_tensor_cache: dict[tuple[str, str], torch.LongTensor] = {}
        self._membership_mask_cache: dict[tuple[str, str, int], torch.BoolTensor] = {}
        self._row_processed_generated_lens: list[int] = []
        self._row_states: list[dict[str, int | bool | None]] = []

    def _generated_ids(self, input_ids: torch.LongTensor, row_idx: int) -> torch.LongTensor:
        prompt_len = self.prompt_lengths[min(row_idx, len(self.prompt_lengths) - 1)]
        if prompt_len >= int(input_ids.shape[-1]):
            return input_ids.new_empty((0,), dtype=torch.long)
        return input_ids[row_idx, prompt_len:]

    def _cached_allowed_tensor(
        self,
        *,
        name: str,
        ids: Sequence[int],
        device: torch.device,
    ) -> torch.LongTensor:
        key = (name, str(device))
        cached = self._allowed_tensor_cache.get(key)
        if cached is None:
            cached = torch.tensor([int(v) for v in ids], device=device, dtype=torch.long)
            self._allowed_tensor_cache[key] = cached
        return cached

    def _cached_membership_mask(
        self,
        *,
        name: str,
        ids: Sequence[int],
        device: torch.device,
        vocab_size: int,
    ) -> torch.BoolTensor:
        key = (name, str(device), int(vocab_size))
        cached = self._membership_mask_cache.get(key)
        if cached is None:
            cached = torch.zeros(int(vocab_size), device=device, dtype=torch.bool)
            ids_tensor = self._cached_allowed_tensor(
                name=f"{name}_ids",
                ids=ids,
                device=device,
            )
            valid = ids_tensor[(ids_tensor >= 0) & (ids_tensor < int(vocab_size))]
            if valid.numel() > 0:
                cached[valid] = True
            self._membership_mask_cache[key] = cached
        return cached

    def _ensure_row_state(self, row_idx: int) -> None:
        while len(self._row_states) <= row_idx:
            self._row_states.append(
                {"coord_count": None, "row_start": True, "done": False}
            )
            self._row_processed_generated_lens.append(0)

    def _reset_row_state(self, row_idx: int) -> None:
        self._ensure_row_state(row_idx)
        self._row_states[row_idx] = {
            "coord_count": None,
            "row_start": True,
            "done": False,
        }
        self._row_processed_generated_lens[row_idx] = 0

    def _advance_state_with_token(self, row_idx: int, token_id: int) -> None:
        state = self._row_states[row_idx]
        if bool(state["done"]):
            return
        token = int(token_id)
        if token in self.eos_id_set:
            state["coord_count"] = None
            state["row_start"] = False
            state["done"] = True
            return
        if token in self.newline_id_set:
            state["coord_count"] = None
            state["row_start"] = True
            return
        coord_count = state["coord_count"]
        if isinstance(coord_count, int):
            if token in self.coord_id_set and coord_count < 4:
                state["coord_count"] = coord_count + 1
                state["row_start"] = False
                return
            state["coord_count"] = None
            state["row_start"] = False
            return
        if token == self.ids.box_start_id:
            state["coord_count"] = 0
            state["row_start"] = False
            return
        state["row_start"] = False

    def _sync_row_state(self, row_idx: int, generated: torch.LongTensor) -> None:
        self._ensure_row_state(row_idx)
        generated_len = int(generated.numel())
        processed_len = int(self._row_processed_generated_lens[row_idx])
        if generated_len < processed_len:
            self._reset_row_state(row_idx)
            processed_len = 0
        if generated_len == processed_len:
            return
        new_tokens = (
            generated[processed_len:generated_len].detach().cpu().tolist()
        )
        for token in new_tokens:
            self._advance_state_with_token(row_idx, int(token))
        self._row_processed_generated_lens[row_idx] = generated_len

    def _allowed_ids_for_row_state(
        self,
        row_idx: int,
        *,
        device: torch.device,
    ) -> torch.LongTensor | None:
        state = self._row_states[row_idx]
        if bool(state["done"]):
            return None
        coord_count = state["coord_count"]
        if isinstance(coord_count, int):
            if coord_count < 4:
                return self._cached_allowed_tensor(
                    name="coord",
                    ids=self.ids.coord_ids,
                    device=device,
                )
            if coord_count == 4:
                return self._cached_allowed_tensor(
                    name="after_bbox",
                    ids=self.after_bbox_allowed,
                    device=device,
                )
            return None
        if bool(state["row_start"]) and self.force_row_start:
            return self._cached_allowed_tensor(
                name="row_start",
                ids=self.row_start_allowed,
                device=device,
            )
        return None

    def _allowed_ids_for_generated(
        self,
        generated: torch.LongTensor,
        *,
        vocab_size: int,
    ) -> torch.LongTensor | None:
        if generated.numel() == 0:
            return (
                self._cached_allowed_tensor(
                    name="row_start",
                    ids=self.row_start_allowed,
                    device=generated.device,
                )
                if self.force_row_start
                else None
            )
        eos_mask = self._cached_membership_mask(
            name="eos",
            ids=self.ids.eos_ids,
            device=generated.device,
            vocab_size=vocab_size,
        )
        newline_mask = self._cached_membership_mask(
            name="newline",
            ids=self.ids.newline_ids,
            device=generated.device,
            vocab_size=vocab_size,
        )
        last = generated[-1]
        if bool(eos_mask[last].item()):
            return None
        if bool(newline_mask[last].item()):
            return (
                self._cached_allowed_tensor(
                    name="row_start",
                    ids=self.row_start_allowed,
                    device=generated.device,
                )
                if self.force_row_start
                else None
            )

        box_positions = (generated == int(self.ids.box_start_id)).nonzero(
            as_tuple=False
        )
        if box_positions.numel() == 0:
            return None
        last_box_idx = int(box_positions[-1].item())

        tail = generated[last_box_idx + 1 :]
        if tail.numel() > 0 and bool((newline_mask[tail] | eos_mask[tail]).any().item()):
            return None
        coord_mask = self._cached_membership_mask(
            name="coord",
            ids=self.ids.coord_ids,
            device=generated.device,
            vocab_size=vocab_size,
        )
        if tail.numel() > 0 and not bool(coord_mask[tail].all().item()):
            return None
        if int(tail.numel()) < 4:
            return self._cached_allowed_tensor(
                name="coord",
                ids=self.ids.coord_ids,
                device=generated.device,
            )
        if int(tail.numel()) == 4:
            return self._cached_allowed_tensor(
                name="after_bbox",
                ids=self.after_bbox_allowed,
                device=generated.device,
            )
        return None

    @staticmethod
    def _mask_to_allowed(
        scores_row: torch.FloatTensor, allowed_ids: Iterable[int] | torch.LongTensor
    ) -> torch.FloatTensor:
        if isinstance(allowed_ids, torch.Tensor):
            allowed_tensor = allowed_ids.to(device=scores_row.device, dtype=torch.long)
            allowed_tensor = allowed_tensor[
                (allowed_tensor >= 0) & (allowed_tensor < int(scores_row.shape[-1]))
            ]
        else:
            allowed = [
                int(v) for v in allowed_ids if 0 <= int(v) < int(scores_row.shape[-1])
            ]
            allowed_tensor = torch.tensor(
                allowed, device=scores_row.device, dtype=torch.long
            )
        if allowed_tensor.numel() == 0:
            return scores_row
        masked = torch.full_like(scores_row, -float("inf"))
        masked.index_copy_(0, allowed_tensor, scores_row.index_select(0, allowed_tensor))
        return masked

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for row_idx in range(int(input_ids.shape[0])):
            generated = self._generated_ids(input_ids, row_idx)
            self._sync_row_state(row_idx, generated)
            allowed = self._allowed_ids_for_row_state(
                row_idx,
                device=scores.device,
            )
            if allowed is not None:
                scores[row_idx] = self._mask_to_allowed(scores[row_idx], allowed)
        return scores


def build_compact_full_grammar_logits_processor(
    *,
    tokenizer: object,
    prompt_lengths: Sequence[int],
    force_row_start: bool = True,
) -> CompactFullGrammarLogitsProcessor:
    object_start_id = _token_id(tokenizer, OBJECT_REF_START_TOKEN)
    box_start_id = _token_id(tokenizer, BOX_START_TOKEN)
    im_end_id = _token_id(tokenizer, IM_END_TOKEN)
    newline_id = _single_token_encoding(tokenizer, "\n")

    missing = []
    if object_start_id is None:
        missing.append(OBJECT_REF_START_TOKEN)
    if box_start_id is None:
        missing.append(BOX_START_TOKEN)
    if im_end_id is None:
        missing.append(IM_END_TOKEN)
    if newline_id is None:
        missing.append("\\n")
    if missing:
        raise ValueError(
            "compact_full grammar decoding requires tokenizer ids for: "
            + ", ".join(missing)
        )

    coord_ids: list[int] = []
    missing_coord_tokens: list[str] = []
    for idx in range(1000):
        token = f"<|coord_{idx}|>"
        token_id = _token_id(tokenizer, token)
        if token_id is None:
            missing_coord_tokens.append(token)
        else:
            coord_ids.append(int(token_id))
    if missing_coord_tokens:
        preview = ", ".join(missing_coord_tokens[:5])
        raise ValueError(
            "compact_full grammar decoding requires all 1000 coord tokens; "
            f"missing {len(missing_coord_tokens)} starting with {preview}"
        )

    ids = CompactFullGrammarIds(
        object_start_id=int(object_start_id),
        box_start_id=int(box_start_id),
        coord_ids=tuple(dict.fromkeys(coord_ids)),
        newline_ids=(int(newline_id),),
        eos_ids=(int(im_end_id),),
    )
    return CompactFullGrammarLogitsProcessor(
        ids=ids,
        prompt_lengths=prompt_lengths,
        force_row_start=force_row_start,
    )


def build_compact_grammar_logits_processor(
    *,
    tokenizer: object,
    prompt_lengths: Sequence[int],
    detection_sequence_format: str,
    force_row_start: bool = True,
) -> LogitsProcessor:
    try:
        fmt = normalize_detection_sequence_format(detection_sequence_format)
    except ValueError as exc:
        raise ValueError(
            "compact grammar decoding currently supports only "
            "detection_sequence_format=compact_full"
        ) from exc
    if fmt != COMPACT_FULL_FORMAT:
        raise ValueError(
            "compact grammar decoding currently supports only "
            f"detection_sequence_format=compact_full, got {fmt!r}"
        )
    return build_compact_full_grammar_logits_processor(
        tokenizer=tokenizer,
        prompt_lengths=prompt_lengths,
        force_row_start=force_row_start,
    )


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


__all__ = [
    "CompactFullGrammarIds",
    "CompactFullGrammarLogitsProcessor",
    "build_array_branch_continuation_steering_logits_processor",
    "build_bbox_tail_closure_steering_logits_processor",
    "build_bbox_tail_then_object_open_once_steering_logits_processor",
    "build_bbox_tail_then_object_open_steering_logits_processor",
    "build_compact_full_grammar_logits_processor",
    "build_compact_grammar_logits_processor",
    "build_terminating_token_suppression_logits_processor",
]
