"""Raw-text token-row scoring helpers for decode-bias analysis."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import torch
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor


_DESC_VALUE_PATTERN = re.compile(r'"desc"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"')
_BBOX_ARRAY_PATTERN = re.compile(r'"bbox_2d"\s*:\s*\[(?P<content>[^\]]*)\]')
_INT_PATTERN = re.compile(r"-?\d+")


def build_repetition_penalty_processors(
    *, repetition_penalty: float = 1.0
) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if float(repetition_penalty) != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(float(repetition_penalty)))
    return processors


def score_processed_span_token_rows(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    batch_idx: int,
    positions: Sequence[int],
    tokenizer: object | None = None,
    logits_processors: Sequence[object] | LogitsProcessorList | None = None,
    history_start: int = 0,
) -> list[dict[str, object]]:
    if not positions:
        raise ValueError("positions must not be empty")
    processors = _coerce_logits_processors(logits_processors)
    rows: list[dict[str, object]] = []
    for pos in positions:
        if int(pos) <= 0 or int(pos) >= int(input_ids.shape[1]):
            raise ValueError(f"position out of range: {pos}")
        raw_logits = logits[batch_idx, int(pos) - 1].float()
        history_ids = [
            int(value)
            for value in input_ids[batch_idx, int(history_start) : int(pos)]
            .detach()
            .cpu()
            .tolist()
        ]
        processed_logits = _apply_logits_processors(
            raw_logits=raw_logits,
            history_ids=history_ids,
            processors=processors,
        )
        token_id = int(input_ids[batch_idx, int(pos)].item())
        row = {
            "position": int(pos),
            "token_id": token_id,
            "token_text": _decode_token_text(tokenizer=tokenizer, token_id=token_id),
            "history_ids": history_ids,
            "raw_logprob": _token_logprob(logits=raw_logits, token_id=token_id),
            "processed_logprob": _token_logprob(
                logits=processed_logits,
                token_id=token_id,
            ),
        }
        rows.append(row)
    return rows


def group_raw_text_token_rows(
    *,
    tokenizer: object,
    candidate_assistant_text: str,
    token_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    desc_positions = _token_positions_for_matches(
        tokenizer=tokenizer,
        text=candidate_assistant_text,
        matches=[
            (match.start("value"), match.end("value"))
            for match in _DESC_VALUE_PATTERN.finditer(candidate_assistant_text)
        ],
    )
    digit_matches: list[tuple[int, int]] = []
    for bbox_match in _BBOX_ARRAY_PATTERN.finditer(candidate_assistant_text):
        content_start = int(bbox_match.start("content"))
        content = bbox_match.group("content")
        for number_match in _INT_PATTERN.finditer(content):
            digit_matches.append(
                (
                    content_start + int(number_match.start()),
                    content_start + int(number_match.end()),
                )
            )
    digit_positions = _token_positions_for_matches(
        tokenizer=tokenizer,
        text=candidate_assistant_text,
        matches=digit_matches,
    )
    grouped = {
        "desc": [],
        "digit": [],
        "structure": [],
    }
    for row in token_rows:
        position = _extract_token_row_position(row)
        copied = dict(row)
        if position in desc_positions:
            grouped["desc"].append(copied)
        elif position in digit_positions:
            grouped["digit"].append(copied)
        else:
            grouped["structure"].append(copied)
    return grouped


def _coerce_logits_processors(
    logits_processors: Sequence[object] | LogitsProcessorList | None,
) -> LogitsProcessorList:
    if isinstance(logits_processors, LogitsProcessorList):
        return logits_processors
    processors = LogitsProcessorList()
    if logits_processors is None:
        return processors
    for processor in logits_processors:
        processors.append(processor)
    return processors


def _apply_logits_processors(
    *,
    raw_logits: torch.Tensor,
    history_ids: Sequence[int],
    processors: LogitsProcessorList,
) -> torch.Tensor:
    if not processors:
        return raw_logits
    history = torch.tensor(
        [list(history_ids)],
        dtype=torch.long,
        device=raw_logits.device,
    )
    return processors(history, raw_logits.unsqueeze(0))[0]


def _token_logprob(*, logits: torch.Tensor, token_id: int) -> float:
    return float(
        logits[int(token_id)].detach().cpu().item()
        - torch.logsumexp(logits, dim=-1).detach().cpu().item()
    )


def _decode_token_text(*, tokenizer: object | None, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return None
    return str(
        decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _token_positions_for_matches(
    *,
    tokenizer: object,
    text: str,
    matches: Sequence[tuple[int, int]],
) -> set[int]:
    encode = getattr(tokenizer, "encode")
    token_positions: set[int] = set()
    for char_start, char_end in matches:
        prefix_ids = list(encode(text[: int(char_start)], add_special_tokens=False))
        through_ids = list(encode(text[: int(char_end)], add_special_tokens=False))
        token_positions.update(range(len(prefix_ids), len(through_ids)))
    return token_positions


def _extract_token_row_position(row: Mapping[str, object]) -> int:
    if "assistant_relative_position" in row:
        return int(row["assistant_relative_position"])
    if "position" in row:
        return int(row["position"])
    raise KeyError("token row missing position key")
