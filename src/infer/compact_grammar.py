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

    def _generated_ids(self, input_ids: torch.LongTensor, row_idx: int) -> list[int]:
        prompt_len = self.prompt_lengths[min(row_idx, len(self.prompt_lengths) - 1)]
        if prompt_len >= int(input_ids.shape[-1]):
            return []
        return [int(v) for v in input_ids[row_idx, prompt_len:].detach().cpu().tolist()]

    def _allowed_ids_for_generated(self, generated: Sequence[int]) -> tuple[int, ...] | None:
        if not generated:
            return self.row_start_allowed if self.force_row_start else None
        last = int(generated[-1])
        if last in self.eos_id_set:
            return None
        if last in self.newline_id_set:
            return self.row_start_allowed if self.force_row_start else None

        try:
            last_box_idx = len(generated) - 1 - list(reversed(generated)).index(
                self.ids.box_start_id
            )
        except ValueError:
            return None

        tail = [int(v) for v in generated[last_box_idx + 1 :]]
        if any(v in self.newline_id_set or v in self.eos_id_set for v in tail):
            return None
        if not all(v in self.coord_id_set for v in tail):
            return None
        if len(tail) < 4:
            return self.ids.coord_ids
        if len(tail) == 4:
            return self.after_bbox_allowed
        return None

    @staticmethod
    def _mask_to_allowed(
        scores_row: torch.FloatTensor, allowed_ids: Iterable[int]
    ) -> torch.FloatTensor:
        allowed = [int(v) for v in allowed_ids if 0 <= int(v) < int(scores_row.shape[-1])]
        if not allowed:
            return scores_row
        masked = torch.full_like(scores_row, -float("inf"))
        allowed_tensor = torch.tensor(allowed, device=scores_row.device, dtype=torch.long)
        masked.index_copy_(0, allowed_tensor, scores_row.index_select(0, allowed_tensor))
        return masked

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for row_idx in range(int(input_ids.shape[0])):
            allowed = self._allowed_ids_for_generated(
                self._generated_ids(input_ids, row_idx)
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
