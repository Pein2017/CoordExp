"""Scoring helpers for raw-text coordinate continuity probes."""

from __future__ import annotations

import json
from typing import Sequence

import torch


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
    payload = json.loads(assistant_text)
    slot_index = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}[slot]
    for row_idx, row in enumerate(payload):
        bbox = list(row.get("bbox_2d") or [])
        if object_index is not None and int(object_index) != int(row_idx):
            continue
        if bbox == list(original_bbox):
            bbox[slot_index] = int(candidate_value)
            row["bbox_2d"] = bbox
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raise ValueError("original_bbox_not_found")


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
