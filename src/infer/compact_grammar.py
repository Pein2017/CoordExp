"""HF logits processors for compact detection sequence decoding."""

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


__all__ = [
    "CompactFullGrammarIds",
    "CompactFullGrammarLogitsProcessor",
    "build_compact_full_grammar_logits_processor",
    "build_compact_grammar_logits_processor",
]
