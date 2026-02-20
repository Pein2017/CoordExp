"""Aggregate token metrics helpers (canonical import path).

Historically these helpers lived under `src.trainers.metrics`.
`src.trainers.metrics.aggregate_token_metrics` now remains as a compatibility shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data_collators.token_types import TokenType


@dataclass(frozen=True)
class NextTokenBatch:
    """A view of next-token prediction tensors for supervised positions."""

    logits_next: torch.Tensor
    labels_next: torch.Tensor
    supervised_mask: torch.Tensor
    supervised_count: int

    preds: torch.Tensor
    labels_masked: torch.Tensor
    preds_masked: torch.Tensor

    token_types_next: torch.Tensor | None
    types_masked: torch.Tensor | None

    topk_indices: torch.Tensor | None


def _infer_token_types_next(
    labels: torch.Tensor, token_types: Any
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (token_types_next, types_masked) or (None, None) if unavailable."""

    if not isinstance(token_types, torch.Tensor) or token_types.shape != labels.shape:
        return None, None

    token_types_next = token_types[:, 1:]
    # All-IGNORE is treated as "no types" (e.g., packed misalignment fallback).
    if not (token_types_next != TokenType.IGNORE).any().item():
        return None, None

    # types_masked is derived later once we have the supervised mask.
    return token_types_next, None


def build_next_token_batch(
    *,
    logits: Any,
    labels: Any,
    token_types: Any,
    log_top5: bool,
) -> NextTokenBatch | None:
    """Build a batch view for aggregate token metrics.

    Returns None when required inputs are missing or there are no supervised tokens.
    """

    if logits is None or labels is None or not isinstance(labels, torch.Tensor):
        return None
    if not isinstance(logits, torch.Tensor) or logits.ndim < 3 or labels.ndim < 2:
        return None

    logits_next = logits[:, :-1, :]
    labels_next = labels[:, 1:]

    supervised_mask = labels_next != -100
    supervised_count = int(supervised_mask.sum().detach().item())
    if supervised_count == 0:
        return None

    token_types_next, _ = _infer_token_types_next(labels, token_types)

    with torch.no_grad():
        preds = logits_next.argmax(dim=-1)
        labels_masked = labels_next[supervised_mask]
        preds_masked = preds[supervised_mask]

        topk_indices = None
        if log_top5:
            flat_logits_all = logits_next[supervised_mask]
            if flat_logits_all.numel() > 0:
                k = min(5, int(flat_logits_all.shape[-1]))
                topk_indices = flat_logits_all.topk(k=k, dim=-1).indices

    types_masked = None
    if token_types_next is not None:
        if token_types_next.shape != supervised_mask.shape:
            token_types_next = None
        else:
            types_masked = token_types_next[supervised_mask]

    return NextTokenBatch(
        logits_next=logits_next,
        labels_next=labels_next,
        supervised_mask=supervised_mask,
        supervised_count=supervised_count,
        preds=preds,
        labels_masked=labels_masked,
        preds_masked=preds_masked,
        token_types_next=token_types_next,
        types_masked=types_masked,
        topk_indices=topk_indices,
    )


def compute_top5_token_acc(batch: NextTokenBatch) -> float:
    if batch.topk_indices is None:
        return 0.0
    with torch.no_grad():
        acc = (
            (batch.topk_indices == batch.labels_masked.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
        )
    return float(acc.detach().item())


def compute_text_token_acc(
    batch: NextTokenBatch, *, coord_mask: torch.Tensor | None
) -> float | None:
    # Token-type metrics (optional)
    if batch.token_types_next is not None:
        text_mask = batch.supervised_mask & (batch.token_types_next != TokenType.COORD)
        text_mask = text_mask & (batch.token_types_next != TokenType.IGNORE)
    elif coord_mask is not None:
        text_mask = batch.supervised_mask & (~coord_mask)
    else:
        text_mask = batch.supervised_mask

    if text_mask is None or not text_mask.any().item():
        return None

    with torch.no_grad():
        acc = (batch.preds[text_mask] == batch.labels_next[text_mask]).float().mean()
    return float(acc.detach().item())


def compute_token_type_fracs(batch: NextTokenBatch) -> dict[str, float]:
    out: dict[str, float] = {}
    if batch.types_masked is None:
        return out

    for name, type_id in (
        ("desc", TokenType.DESC),
        ("format", TokenType.FORMAT),
        ("coord", TokenType.COORD),
    ):
        sel = batch.types_masked == type_id
        if not sel.any().item():
            continue
        out[f"{name}_token_frac"] = float(sel.float().mean().detach().item())

    return out


def compute_token_type_acc(batch: NextTokenBatch) -> dict[str, float]:
    out: dict[str, float] = {}
    if batch.types_masked is None:
        return out

    for name, type_id in (
        ("desc", TokenType.DESC),
        ("format", TokenType.FORMAT),
        ("coord", TokenType.COORD),
    ):
        type_sel = batch.types_masked == type_id
        if not type_sel.any().item():
            continue
        with torch.no_grad():
            acc = (
                (batch.preds_masked[type_sel] == batch.labels_masked[type_sel])
                .float()
                .mean()
            )
        out[f"{name}_token_acc"] = float(acc.detach().item())

        if batch.topk_indices is None:
            continue
        with torch.no_grad():
            top5 = (
                (batch.topk_indices[type_sel] == batch.labels_masked[type_sel].unsqueeze(-1))
                .any(dim=-1)
                .float()
                .mean()
            )
        out[f"{name}_token_acc_top5"] = float(top5.detach().item())

    return out
