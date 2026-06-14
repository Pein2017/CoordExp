from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

PrefixDenoisingLossBranchId = Literal["clean_full", "noisy_full"]


@dataclass(frozen=True)
class PrefixDenoisingSegmentSpan:
    batch_index: int
    token_start: int
    token_end: int
    branch_id: PrefixDenoisingLossBranchId
    segment_id: str


@dataclass(frozen=True)
class PrefixDenoisingCEResult:
    loss: torch.Tensor
    clean_ce: torch.Tensor
    noisy_ce: torch.Tensor
    token_pooled_ce: torch.Tensor
    clean_denominator: int
    noisy_denominator: int


def compute_branch_balanced_hard_ce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_spans: tuple[PrefixDenoisingSegmentSpan, ...],
) -> PrefixDenoisingCEResult:
    """Compute V1 branch-balanced hard CE with causal LM label alignment."""

    _validate_logits_and_labels(logits=logits, labels=labels)
    zero = logits.float().sum() * 0.0
    ce_sums = {"clean_full": zero, "noisy_full": zero}
    denominators = {"clean_full": 0, "noisy_full": 0}

    for span in segment_spans:
        if span.branch_id not in ce_sums:
            raise ValueError(
                f"unsupported prefix denoising branch_id: {span.branch_id!r}"
            )
        ce_sum, denominator = _segment_ce_sum(
            logits=logits,
            labels=labels,
            span=span,
        )
        ce_sums[span.branch_id] = ce_sums[span.branch_id] + ce_sum
        denominators[span.branch_id] += denominator

    clean_denominator = denominators["clean_full"]
    noisy_denominator = denominators["noisy_full"]
    if clean_denominator == 0 or noisy_denominator == 0:
        raise ValueError(
            "prefix denoising CE requires supervised labels in both "
            "clean_full and noisy_full branches"
        )

    clean_ce = ce_sums["clean_full"] / float(clean_denominator)
    noisy_ce = ce_sums["noisy_full"] / float(noisy_denominator)
    token_pooled_ce = (ce_sums["clean_full"] + ce_sums["noisy_full"]) / float(
        clean_denominator + noisy_denominator
    )
    return PrefixDenoisingCEResult(
        loss=0.5 * clean_ce + 0.5 * noisy_ce,
        clean_ce=clean_ce,
        noisy_ce=noisy_ce,
        token_pooled_ce=token_pooled_ce,
        clean_denominator=clean_denominator,
        noisy_denominator=noisy_denominator,
    )


def topk_accuracy_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_spans: tuple[PrefixDenoisingSegmentSpan, ...],
    topk: Iterable[int] = (1, 5),
) -> dict[int, float]:
    """Compute full-vocab top-k token accuracy at supervised shifted positions."""

    _validate_logits_and_labels(logits=logits, labels=labels)
    requested_topk = tuple(int(k) for k in topk)
    if not requested_topk:
        return {}
    if any(k <= 0 for k in requested_topk):
        raise ValueError("top-k accuracy requires positive k values")

    active_logits: list[torch.Tensor] = []
    active_labels: list[torch.Tensor] = []
    for span in segment_spans:
        shifted_logits, shifted_labels = _segment_shifted_logits_and_labels(
            logits=logits,
            labels=labels,
            span=span,
        )
        if shifted_labels.numel() == 0:
            continue
        active_logits.append(shifted_logits)
        active_labels.append(shifted_labels)

    if not active_labels:
        return {k: 0.0 for k in requested_topk}

    logits_tensor = torch.cat(active_logits, dim=0)
    labels_tensor = torch.cat(active_labels, dim=0)
    denominator = int(labels_tensor.numel())
    vocab_size = int(logits_tensor.shape[-1])
    max_k = min(max(requested_topk), vocab_size)
    top_indices = logits_tensor.topk(k=max_k, dim=-1).indices

    accuracies: dict[int, float] = {}
    for requested_k in requested_topk:
        k = min(requested_k, vocab_size)
        correct = top_indices[:, :k].eq(labels_tensor.unsqueeze(-1)).any(dim=-1)
        accuracies[requested_k] = float(
            correct.float().mean().detach().cpu().item()
        )
    return accuracies


def _segment_ce_sum(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    span: PrefixDenoisingSegmentSpan,
) -> tuple[torch.Tensor, int]:
    shifted_logits, shifted_labels = _segment_shifted_logits_and_labels(
        logits=logits,
        labels=labels,
        span=span,
    )
    denominator = int(shifted_labels.numel())
    if denominator == 0:
        return logits.float().sum() * 0.0, 0
    ce_sum = F.cross_entropy(
        shifted_logits.float(),
        shifted_labels,
        reduction="sum",
    )
    return ce_sum, denominator


def _segment_shifted_logits_and_labels(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    span: PrefixDenoisingSegmentSpan,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_index = int(span.batch_index)
    token_start = int(span.token_start)
    token_end = int(span.token_end)
    _validate_span(
        span=span,
        batch_size=int(labels.shape[0]),
        sequence_length=int(labels.shape[1]),
    )

    segment_labels = labels[batch_index, token_start:token_end]
    active = segment_labels.ne(-100)
    if not bool(active.any()):
        return (
            logits[batch_index, token_start:token_start].float(),
            labels[batch_index, token_start:token_start],
        )

    label_positions = torch.arange(token_start, token_end, device=labels.device)
    active_positions = label_positions[active]
    if torch.any(active_positions <= token_start):
        raise ValueError(
            "prefix denoising labels at or before segment start cannot be "
            "supervised in causal LM alignment"
        )

    return logits[batch_index, active_positions - 1], segment_labels[active]


def _validate_logits_and_labels(*, logits: torch.Tensor, labels: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("prefix denoising CE requires logits with shape [batch, time, vocab]")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("prefix denoising CE requires labels with shape [batch, time]")
    if tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            "prefix denoising labels must match logits batch/time shape; "
            f"got labels={tuple(labels.shape)} logits={tuple(logits.shape[:2])}"
        )


def _validate_span(
    *,
    span: PrefixDenoisingSegmentSpan,
    batch_size: int,
    sequence_length: int,
) -> None:
    batch_index = int(span.batch_index)
    token_start = int(span.token_start)
    token_end = int(span.token_end)
    if batch_index < 0 or batch_index >= batch_size:
        raise ValueError(
            f"prefix denoising segment {span.segment_id!r} has batch_index "
            f"{batch_index}, outside batch size {batch_size}"
        )
    if token_start < 0 or token_end > sequence_length or token_start >= token_end:
        raise ValueError(
            f"prefix denoising segment {span.segment_id!r} has invalid token span "
            f"[{token_start}, {token_end}) for sequence length {sequence_length}"
        )


__all__ = [
    "PrefixDenoisingSegmentSpan",
    "PrefixDenoisingCEResult",
    "compute_branch_balanced_hard_ce",
    "topk_accuracy_from_logits",
]
