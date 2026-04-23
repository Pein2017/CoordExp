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


def inspect_processed_position(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    batch_idx: int,
    position: int,
    tokenizer: object | None = None,
    logits_processors: Sequence[object] | LogitsProcessorList | None = None,
    history_start: int = 0,
    top_k: int = 10,
    focus_token_ids: Mapping[str, int] | None = None,
    group_token_ids: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, object]:
    if int(position) <= 0 or int(position) >= int(input_ids.shape[1]):
        raise ValueError(f"position out of range: {position}")
    processors = _coerce_logits_processors(logits_processors)
    raw_logits = logits[batch_idx, int(position) - 1].float()
    history_ids = [
        int(value)
        for value in input_ids[batch_idx, int(history_start) : int(position)]
        .detach()
        .cpu()
        .tolist()
    ]
    processed_logits = _apply_logits_processors(
        raw_logits=raw_logits,
        history_ids=history_ids,
        processors=processors,
    )
    token_id = int(input_ids[batch_idx, int(position)].item())
    row = {
        "position": int(position),
        "token_id": token_id,
        "token_text": _decode_token_text(tokenizer=tokenizer, token_id=token_id),
        "history_ids": history_ids,
        "raw_logprob": _token_logprob(logits=raw_logits, token_id=token_id),
        "processed_logprob": _token_logprob(
            logits=processed_logits,
            token_id=token_id,
        ),
        "top_tokens": _top_token_rows(
            raw_logits=raw_logits,
            processed_logits=processed_logits,
            tokenizer=tokenizer,
            top_k=top_k,
        ),
        "focus_tokens": _focus_token_rows(
            raw_logits=raw_logits,
            processed_logits=processed_logits,
            tokenizer=tokenizer,
            focus_token_ids=focus_token_ids,
        ),
        "group_summaries": _group_token_summaries(
            raw_logits=raw_logits,
            processed_logits=processed_logits,
            tokenizer=tokenizer,
            group_token_ids=group_token_ids,
        ),
    }
    return row


def group_raw_text_token_rows(
    *,
    tokenizer: object,
    candidate_assistant_text: str,
    token_rows: Sequence[Mapping[str, object]],
    position_base: int | None = None,
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
        position = _extract_token_row_position(row, position_base=position_base)
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


def _top_token_rows(
    *,
    raw_logits: torch.Tensor,
    processed_logits: torch.Tensor,
    tokenizer: object | None,
    top_k: int,
) -> list[dict[str, object]]:
    if int(top_k) <= 0:
        raise ValueError("top_k must be >= 1")
    limit = min(int(top_k), int(raw_logits.shape[-1]))
    values, indices = torch.topk(raw_logits, k=limit)
    rows: list[dict[str, object]] = []
    for value, token_idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        token_id = int(token_idx)
        rows.append(
            {
                "token_id": token_id,
                "token_text": _decode_token_text(tokenizer=tokenizer, token_id=token_id),
                "raw_logprob": _token_logprob(logits=raw_logits, token_id=token_id),
                "processed_logprob": _token_logprob(
                    logits=processed_logits,
                    token_id=token_id,
                ),
                "raw_logit": float(value),
            }
        )
    return rows


def _focus_token_rows(
    *,
    raw_logits: torch.Tensor,
    processed_logits: torch.Tensor,
    tokenizer: object | None,
    focus_token_ids: Mapping[str, int] | None,
) -> dict[str, dict[str, object]]:
    if not focus_token_ids:
        return {}
    focus_rows: dict[str, dict[str, object]] = {}
    vocab_size = int(raw_logits.shape[-1])
    for label, token_id_raw in focus_token_ids.items():
        token_id = int(token_id_raw)
        if token_id < 0 or token_id >= vocab_size:
            continue
        focus_rows[str(label)] = {
            "token_id": token_id,
            "token_text": _decode_token_text(tokenizer=tokenizer, token_id=token_id),
            "raw_logprob": _token_logprob(logits=raw_logits, token_id=token_id),
            "processed_logprob": _token_logprob(
                logits=processed_logits,
                token_id=token_id,
            ),
        }
    return focus_rows


def _group_token_summaries(
    *,
    raw_logits: torch.Tensor,
    processed_logits: torch.Tensor,
    tokenizer: object | None,
    group_token_ids: Mapping[str, Sequence[int]] | None,
) -> dict[str, dict[str, object]]:
    if not group_token_ids:
        return {}
    raw_log_norm = torch.logsumexp(raw_logits, dim=-1)
    processed_log_norm = torch.logsumexp(processed_logits, dim=-1)
    vocab_size = int(raw_logits.shape[-1])
    summaries: dict[str, dict[str, object]] = {}
    for group_name, token_ids in group_token_ids.items():
        unique_ids = sorted(
            {
                int(token_id)
                for token_id in token_ids
                if 0 <= int(token_id) < vocab_size
            }
        )
        if not unique_ids:
            summaries[str(group_name)] = {
                "token_count": 0,
                "raw_prob_mass": 0.0,
                "raw_logprob_mass": None,
                "raw_top_token_id": None,
                "raw_top_token_text": None,
                "raw_top_token_logprob": None,
                "processed_prob_mass": 0.0,
                "processed_logprob_mass": None,
                "processed_top_token_id": None,
                "processed_top_token_text": None,
                "processed_top_token_logprob": None,
            }
            continue
        raw_group_logits = raw_logits[unique_ids]
        processed_group_logits = processed_logits[unique_ids]
        raw_mass = torch.logsumexp(raw_group_logits, dim=-1) - raw_log_norm
        processed_mass = torch.logsumexp(processed_group_logits, dim=-1) - processed_log_norm
        raw_top_offset = int(torch.argmax(raw_group_logits).detach().cpu().item())
        processed_top_offset = int(torch.argmax(processed_group_logits).detach().cpu().item())
        raw_top_token_id = int(unique_ids[raw_top_offset])
        processed_top_token_id = int(unique_ids[processed_top_offset])
        summaries[str(group_name)] = {
            "token_count": len(unique_ids),
            "raw_prob_mass": float(torch.exp(raw_mass).detach().cpu().item()),
            "raw_logprob_mass": float(raw_mass.detach().cpu().item()),
            "raw_top_token_id": raw_top_token_id,
            "raw_top_token_text": _decode_token_text(
                tokenizer=tokenizer,
                token_id=raw_top_token_id,
            ),
            "raw_top_token_logprob": _token_logprob(
                logits=raw_logits,
                token_id=raw_top_token_id,
            ),
            "processed_prob_mass": float(torch.exp(processed_mass).detach().cpu().item()),
            "processed_logprob_mass": float(processed_mass.detach().cpu().item()),
            "processed_top_token_id": processed_top_token_id,
            "processed_top_token_text": _decode_token_text(
                tokenizer=tokenizer,
                token_id=processed_top_token_id,
            ),
            "processed_top_token_logprob": _token_logprob(
                logits=processed_logits,
                token_id=processed_top_token_id,
            ),
        }
    return summaries


def _extract_token_row_position(
    row: Mapping[str, object],
    *,
    position_base: int | None,
) -> int:
    if "assistant_relative_position" in row:
        return int(row["assistant_relative_position"])
    if "position" in row and position_base is not None:
        return int(row["position"]) - int(position_base)
    if "position" in row:
        raise KeyError(
            "token row with absolute position requires position_base or "
            "assistant_relative_position"
        )
    raise KeyError("token row missing assistant_relative_position")
