"""Scoring helpers for raw-text coordinate continuity probes."""

from __future__ import annotations

import json
import re
from typing import Sequence

import torch

_BBOX_ARRAY_PATTERN = re.compile(r'"bbox_2d"\s*:\s*\[(?P<content>[^\]]*)\]')
_INT_PATTERN = re.compile(r"-?\d+")


def score_span_logprobs(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    batch_idx: int,
    positions: Sequence[int],
) -> dict[str, float | int]:
    values: list[float] = []
    for pos in positions:
        if int(pos) <= 0 or int(pos) >= int(input_ids.shape[1]):
            raise ValueError(f"position out of range: {pos}")
        prev_logits = logits[batch_idx, int(pos) - 1].float()
        target_id = int(input_ids[batch_idx, int(pos)].item())
        token_logprob = float(
            prev_logits[target_id].detach().cpu().item()
            - torch.logsumexp(prev_logits, dim=-1).detach().cpu().item()
        )
        values.append(token_logprob)
    if not values:
        raise ValueError("positions must not be empty")
    return {
        "count": len(values),
        "sum_logprob": float(sum(values)),
        "mean_logprob": float(sum(values) / len(values)),
    }


def replace_bbox_slot_value(
    *,
    assistant_text: str,
    slot: str,
    original_bbox: Sequence[int],
    candidate_value: int,
    object_index: int | None = None,
) -> str:
    bbox_match, number_match = _find_bbox_slot_text_match(
        assistant_text=assistant_text,
        slot=slot,
        original_bbox=original_bbox,
        object_index=object_index,
    )
    bbox_content = bbox_match.group("content")
    replacement = str(int(candidate_value))
    replaced_content = (
        bbox_content[: number_match.start()]
        + replacement
        + bbox_content[number_match.end() :]
    )
    return (
        assistant_text[: bbox_match.start("content")]
        + replaced_content
        + assistant_text[bbox_match.end("content") :]
    )


def _find_bbox_slot_text_match(
    *,
    assistant_text: str,
    slot: str,
    original_bbox: Sequence[int],
    object_index: int | None = None,
) -> tuple[re.Match[str], re.Match[str]]:
    payload = json.loads(assistant_text)
    if isinstance(payload, dict):
        payload_rows = payload.get("objects")
    else:
        payload_rows = payload
    if not isinstance(payload_rows, list):
        raise ValueError("assistant_payload_objects_missing")
    slot_index = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}[slot]
    bbox_row_indices: list[int] = []
    target_bbox_match_idx: int | None = None
    for row_idx, row in enumerate(payload_rows):
        bbox = list(row.get("bbox_2d") or [])
        if bbox:
            bbox_row_indices.append(int(row_idx))
        if object_index is not None and int(object_index) != int(row_idx):
            continue
        if bbox == list(original_bbox):
            target_bbox_match_idx = len(bbox_row_indices) - 1
            break
    if target_bbox_match_idx is None:
        raise ValueError("original_bbox_not_found")
    bbox_matches = list(_BBOX_ARRAY_PATTERN.finditer(assistant_text))
    if int(target_bbox_match_idx) >= len(bbox_matches):
        raise ValueError("bbox_text_not_found")
    bbox_match = bbox_matches[int(target_bbox_match_idx)]
    bbox_content = bbox_match.group("content")
    number_matches = list(_INT_PATTERN.finditer(bbox_content))
    if len(number_matches) <= int(slot_index):
        raise ValueError("bbox_text_not_found")
    return bbox_match, number_matches[int(slot_index)]


def _relative_positions_for_slot_text(
    *,
    tokenizer: object,
    assistant_text: str,
    slot: str,
    original_bbox: Sequence[int],
    object_index: int | None = None,
) -> list[int]:
    encode = getattr(tokenizer, "encode")
    bbox_match, number_match = _find_bbox_slot_text_match(
        assistant_text=assistant_text,
        slot=slot,
        original_bbox=original_bbox,
        object_index=object_index,
    )
    slot_char_start = bbox_match.start("content") + number_match.start()
    slot_char_end = bbox_match.start("content") + number_match.end()
    prefix_ids = list(encode(assistant_text[:slot_char_start], add_special_tokens=False))
    through_slot_ids = list(
        encode(assistant_text[:slot_char_end], add_special_tokens=False)
    )
    positions = list(range(len(prefix_ids), len(through_slot_ids)))
    if not positions:
        raise ValueError("slot_text_token_span_empty")
    return positions


def build_candidate_coordinate_span(
    *,
    tokenizer: object,
    assistant_text: str,
    slot: str,
    original_bbox: Sequence[int],
    candidate_value: int,
    object_index: int | None = None,
) -> dict[str, object]:
    candidate_assistant_text = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot=slot,
        original_bbox=original_bbox,
        candidate_value=candidate_value,
        object_index=object_index,
    )
    encode = getattr(tokenizer, "encode")
    original_ids = list(encode(assistant_text, add_special_tokens=False))
    candidate_ids = list(encode(candidate_assistant_text, add_special_tokens=False))
    prefix_len = 0
    for left_id, right_id in zip(original_ids, candidate_ids):
        if int(left_id) != int(right_id):
            break
        prefix_len += 1
    max_suffix = min(len(original_ids), len(candidate_ids)) - prefix_len
    suffix_len = 0
    while suffix_len < max_suffix:
        if int(original_ids[-(suffix_len + 1)]) != int(candidate_ids[-(suffix_len + 1)]):
            break
        suffix_len += 1
    changed_stop = len(candidate_ids) - suffix_len
    assistant_relative_positions = list(range(prefix_len, changed_stop))
    if not assistant_relative_positions:
        assistant_relative_positions = _relative_positions_for_slot_text(
            tokenizer=tokenizer,
            assistant_text=candidate_assistant_text,
            slot=slot,
            original_bbox=original_bbox,
            object_index=object_index,
        )
    return {
        "candidate_assistant_text": candidate_assistant_text,
        "assistant_relative_positions": assistant_relative_positions,
    }


def score_candidate_coordinate_sequence(
    *,
    scorer: object,
    image: object,
    assistant_text: str,
    slot: str,
    original_bbox: Sequence[int],
    candidate_value: int,
    prompt_variant: str,
    object_field_order: str,
    object_index: int | None = None,
) -> dict[str, object]:
    candidate = build_candidate_coordinate_span(
        tokenizer=getattr(scorer, "tokenizer"),
        assistant_text=assistant_text,
        slot=slot,
        original_bbox=original_bbox,
        candidate_value=candidate_value,
        object_index=object_index,
    )
    prepared = scorer.prepare_example(
        image=image,
        assistant_text=str(candidate["candidate_assistant_text"]),
        desc_positions_rel=[],
        prompt_variant=prompt_variant,
        object_field_order=object_field_order,
    )
    assistant_start = int(getattr(prepared, "assistant_start"))
    absolute_positions = [
        assistant_start + int(pos)
        for pos in candidate["assistant_relative_positions"]
    ]
    score = scorer.score_prepared_spans(
        prepared=prepared,
        image=image,
        spans=[absolute_positions],
    )[0]
    return {
        **candidate,
        **score,
        "candidate_value": int(candidate_value),
        "absolute_positions": absolute_positions,
    }


def lexical_features_for_candidate(
    *,
    candidate_value: int,
    center_value: int,
    gt_value: int,
    tokenizer_tokens: Sequence[str],
    center_tokens: Sequence[str],
) -> dict[str, int]:
    candidate_text = str(candidate_value)
    center_text = str(center_value)
    shared_prefix = 0
    for left, right in zip(candidate_text, center_text):
        if left != right:
            break
        shared_prefix += 1

    prev = list(range(len(center_text) + 1))
    for candidate_idx, candidate_char in enumerate(candidate_text, start=1):
        current = [candidate_idx]
        for center_idx, center_char in enumerate(center_text, start=1):
            cost = 0 if candidate_char == center_char else 1
            current.append(
                min(
                    prev[center_idx] + 1,
                    current[center_idx - 1] + 1,
                    prev[center_idx - 1] + cost,
                )
            )
        prev = current
    return {
        "numeric_distance_to_center": abs(int(candidate_value) - int(center_value)),
        "numeric_distance_to_gt": abs(int(candidate_value) - int(gt_value)),
        "char_edit_distance": int(prev[-1]),
        "token_edit_distance": abs(len(tokenizer_tokens) - len(center_tokens)),
        "digit_length_match": int(len(candidate_text) == len(center_text)),
        "token_count": int(len(tokenizer_tokens)),
        "shared_prefix_length": int(shared_prefix),
        "same_leading_digit": int(candidate_text[:1] == center_text[:1]),
    }
