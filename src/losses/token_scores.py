"""Differentiable scores for already aligned causal token rows."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.common.errors import LossContractError


def aligned_token_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 log probabilities, one per row, on the input device.

    ``logits`` is [tokens, vocabulary]; ``target_ids`` is a same-device long
    vector. The caller owns causal alignment, selection, reduction and scaling.
    Empty selections return an empty differentiable vector. Nonfinite values
    propagate for the caller's finite gate; invalid IDs fail in torch.gather.
    """
    if logits.ndim != 2 or logits.shape[1] == 0 or not logits.is_floating_point():
        raise LossContractError(
            "aligned token logits must be floating point [tokens, vocabulary] with a nonempty vocabulary",
            code="loss.aligned_logits",
        )
    if target_ids.ndim != 1 or target_ids.shape[0] != logits.shape[0]:
        raise LossContractError(
            "aligned target IDs must have one entry per logits row",
            code="loss.aligned_targets_shape",
        )
    if target_ids.dtype != torch.long or target_ids.device != logits.device:
        raise LossContractError(
            "aligned target IDs must be long tensors on the logits device",
            code="loss.aligned_targets_dtype_device",
        )
    return logits.float().log_softmax(dim=-1).gather(1, target_ids[:, None]).squeeze(1)


def masked_active_mean_ce(logits: torch.Tensor, targets: torch.Tensor, weights: Sequence[int]) -> tuple[torch.Tensor, dict[str, float]]:
    if not (logits.ndim == 2 and logits.shape[0] == targets.numel() == len(weights)):
        raise ValueError('aligned CE shape')
    mask = torch.tensor(weights, dtype=logits.dtype, device=logits.device)
    active = int(mask.sum().item())
    if not (active > 0):
        raise ValueError('CE has no active positions')
    nll = -aligned_token_logprobs(logits.float(), targets)
    return (nll * mask).sum() / active, {"active_tokens": active, "masked_nll_sum": float((nll * mask).detach().sum())}


__all__ = ["aligned_token_logprobs", "masked_active_mean_ce"]
