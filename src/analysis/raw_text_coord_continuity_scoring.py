"""Scoring helpers for raw-text coordinate continuity probes."""

from __future__ import annotations

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
