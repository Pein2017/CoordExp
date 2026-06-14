from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

from src.detection.prefix_denoising.types import ResolvedPrefixDenoisingKLSite

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


@dataclass(frozen=True)
class PrefixDenoisingKLResult:
    loss: torch.Tensor
    raw_loss: torch.Tensor
    candidate_site_count: int
    effective_site_count: int
    identical_prefix_site_count: int
    teacher_support_mass: float
    student_support_mass: float
    teacher_gt_prob_full_coord_vocab: float
    student_gt_prob_full_coord_vocab: float
    teacher_gt_prob_conditional: float
    student_gt_prob_conditional: float
    support_bin_count: float = 0.0
    edge_truncation_rate: float = 0.0
    teacher_top1_is_gt: float = 0.0
    student_top1_is_gt: float = 0.0
    slot_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


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


def coord_support_window(
    *,
    clean_bin: int,
    radius: int,
    coord_min: int = 0,
    coord_max: int = 999,
) -> tuple[int, ...]:
    start = max(int(coord_min), int(clean_bin) - int(radius))
    end = min(int(coord_max), int(clean_bin) + int(radius))
    if start > end:
        return ()
    return tuple(range(start, end + 1))


def compute_local_coord_kl(
    *,
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    sites: tuple[ResolvedPrefixDenoisingKLSite, ...],
    coord_token_ids: torch.Tensor,
) -> PrefixDenoisingKLResult:
    """Compute local-window KL from stopgrad(clean) to noisy distributions."""

    if not sites:
        zero = noisy_logits.float().sum() * 0.0
        return PrefixDenoisingKLResult(
            loss=zero,
            raw_loss=zero,
            candidate_site_count=0,
            effective_site_count=0,
            identical_prefix_site_count=0,
            teacher_support_mass=0.0,
            student_support_mass=0.0,
            teacher_gt_prob_full_coord_vocab=0.0,
            student_gt_prob_full_coord_vocab=0.0,
            teacher_gt_prob_conditional=0.0,
            student_gt_prob_conditional=0.0,
            support_bin_count=0.0,
            edge_truncation_rate=0.0,
            teacher_top1_is_gt=0.0,
            student_top1_is_gt=0.0,
            slot_metrics={},
        )
    _validate_kl_logits(clean_logits=clean_logits, noisy_logits=noisy_logits)
    coord_token_ids = _normalize_coord_token_ids(
        coord_token_ids=coord_token_ids,
        device=noisy_logits.device,
        vocab_size=int(noisy_logits.shape[-1]),
    )

    losses: list[torch.Tensor] = []
    rows: list[dict[str, float | str]] = []
    identical_count = 0
    batch_size, sequence_length, vocab_size = (
        int(noisy_logits.shape[0]),
        int(noisy_logits.shape[1]),
        int(noisy_logits.shape[2]),
    )
    for site in sites:
        if type(site) is not ResolvedPrefixDenoisingKLSite:
            raise TypeError(
                "compute_local_coord_kl requires ResolvedPrefixDenoisingKLSite sites"
            )
        clean_row, noisy_row, support_bins, gt_support_index = _validate_kl_site(
            site=site,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        support_bin_tensor = torch.tensor(
            support_bins,
            device=noisy_logits.device,
            dtype=torch.long,
        )
        support_token_ids = coord_token_ids.index_select(0, support_bin_tensor)
        if bool(torch.any(support_token_ids < 0)) or bool(
            torch.any(support_token_ids >= vocab_size)
        ):
            raise ValueError(
                "coord_token_ids for KL support must be valid logits vocabulary ids"
            )
        gt_token_id = coord_token_ids[int(site.clean_gt_bin)]

        teacher_full = torch.softmax(
            clean_logits[int(site.clean_batch_index), clean_row].detach().float(),
            dim=-1,
        )
        student_full = torch.softmax(
            noisy_logits[int(site.noisy_batch_index), noisy_row].float(),
            dim=-1,
        )
        teacher_local_logits = (
            clean_logits[int(site.clean_batch_index), clean_row]
            .detach()
            .float()
            .index_select(0, support_token_ids)
        )
        student_local_logits = (
            noisy_logits[int(site.noisy_batch_index), noisy_row]
            .float()
            .index_select(0, support_token_ids)
        )
        teacher_prob = torch.softmax(teacher_local_logits, dim=-1)
        teacher_log_prob = torch.log_softmax(teacher_local_logits, dim=-1)
        student_log_prob = torch.log_softmax(student_local_logits, dim=-1)
        student_prob = torch.softmax(student_local_logits, dim=-1)
        losses.append(torch.sum(teacher_prob * (teacher_log_prob - student_log_prob)))

        teacher_support_mass = teacher_full.index_select(0, support_token_ids).sum()
        student_support_mass = student_full.index_select(0, support_token_ids).sum()
        teacher_gt_full = teacher_full[gt_token_id]
        student_gt_full = student_full[gt_token_id]
        teacher_gt_conditional = teacher_prob[gt_support_index]
        student_gt_conditional = student_prob[gt_support_index]
        teacher_top1_is_gt = float(
            int(torch.argmax(teacher_prob).detach().cpu().item()) == gt_support_index
        )
        student_top1_is_gt = float(
            int(torch.argmax(student_prob).detach().cpu().item()) == gt_support_index
        )
        support_min = min(support_bins)
        support_max = max(support_bins)
        edge_truncated = float(
            int(site.clean_gt_bin) - support_min != support_max - int(site.clean_gt_bin)
        )
        if bool(site.identical_prefix):
            identical_count += 1
        rows.append(
            {
                "slot": str(site.coord_slot),
                "teacher_support_mass": float(
                    teacher_support_mass.detach().cpu().item()
                ),
                "student_support_mass": float(
                    student_support_mass.detach().cpu().item()
                ),
                "teacher_gt_prob_full_coord_vocab": float(
                    teacher_gt_full.detach().cpu().item()
                ),
                "student_gt_prob_full_coord_vocab": float(
                    student_gt_full.detach().cpu().item()
                ),
                "teacher_gt_prob_conditional": float(
                    teacher_gt_conditional.detach().cpu().item()
                ),
                "student_gt_prob_conditional": float(
                    student_gt_conditional.detach().cpu().item()
                ),
                "support_bin_count": float(len(support_bins)),
                "edge_truncation_rate": edge_truncated,
                "teacher_top1_is_gt": teacher_top1_is_gt,
                "student_top1_is_gt": student_top1_is_gt,
            }
        )

    raw_loss = torch.stack(losses).mean()
    slot_metrics = {
        slot: _mean_metric_rows([row for row in rows if row["slot"] == slot])
        for slot in ("x1", "y1", "x2", "y2")
        if any(row["slot"] == slot for row in rows)
    }
    return PrefixDenoisingKLResult(
        loss=raw_loss,
        raw_loss=raw_loss,
        candidate_site_count=len(sites),
        effective_site_count=len(losses),
        identical_prefix_site_count=identical_count,
        teacher_support_mass=_mean_row_value(rows, "teacher_support_mass"),
        student_support_mass=_mean_row_value(rows, "student_support_mass"),
        teacher_gt_prob_full_coord_vocab=_mean_row_value(
            rows, "teacher_gt_prob_full_coord_vocab"
        ),
        student_gt_prob_full_coord_vocab=_mean_row_value(
            rows, "student_gt_prob_full_coord_vocab"
        ),
        teacher_gt_prob_conditional=_mean_row_value(
            rows, "teacher_gt_prob_conditional"
        ),
        student_gt_prob_conditional=_mean_row_value(
            rows, "student_gt_prob_conditional"
        ),
        support_bin_count=_mean_row_value(rows, "support_bin_count"),
        edge_truncation_rate=_mean_row_value(rows, "edge_truncation_rate"),
        teacher_top1_is_gt=_mean_row_value(rows, "teacher_top1_is_gt"),
        student_top1_is_gt=_mean_row_value(rows, "student_top1_is_gt"),
        slot_metrics=slot_metrics,
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


def _validate_kl_logits(
    *,
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
) -> None:
    if not isinstance(clean_logits, torch.Tensor) or clean_logits.ndim != 3:
        raise ValueError(
            "prefix denoising KL requires clean_logits with shape [batch, time, vocab]"
        )
    if not isinstance(noisy_logits, torch.Tensor) or noisy_logits.ndim != 3:
        raise ValueError(
            "prefix denoising KL requires noisy_logits with shape [batch, time, vocab]"
        )
    if tuple(clean_logits.shape) != tuple(noisy_logits.shape):
        raise ValueError(
            "prefix denoising KL requires clean/noisy logits with matching shapes"
        )


def _normalize_coord_token_ids(
    *,
    coord_token_ids: torch.Tensor,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    if not isinstance(coord_token_ids, torch.Tensor):
        coord_token_ids = torch.tensor(tuple(coord_token_ids), dtype=torch.long)
    coord_token_ids = coord_token_ids.to(device=device, dtype=torch.long)
    if coord_token_ids.ndim != 1 or int(coord_token_ids.numel()) != 1000:
        raise ValueError(
            "coord_token_ids for prefix denoising KL must contain exactly 1000 ids"
        )
    if bool(torch.any(coord_token_ids < 0)) or bool(
        torch.any(coord_token_ids >= vocab_size)
    ):
        raise ValueError(
            "coord_token_ids for prefix denoising KL must be valid logits vocabulary ids"
        )
    if int(torch.unique(coord_token_ids).numel()) != 1000:
        raise ValueError(
            "coord_token_ids for prefix denoising KL must contain 1000 distinct ids"
        )
    return coord_token_ids


def _validate_kl_site(
    *,
    site: ResolvedPrefixDenoisingKLSite,
    batch_size: int,
    sequence_length: int,
) -> tuple[int, int, tuple[int, ...], int]:
    clean_batch_index = int(site.clean_batch_index)
    noisy_batch_index = int(site.noisy_batch_index)
    if clean_batch_index < 0 or clean_batch_index >= batch_size:
        raise ValueError("resolved KL clean_batch_index is outside logits batch")
    if noisy_batch_index < 0 or noisy_batch_index >= batch_size:
        raise ValueError("resolved KL noisy_batch_index is outside logits batch")
    clean_label_position = int(site.clean_label_position)
    noisy_label_position = int(site.noisy_label_position)
    if clean_label_position <= 0 or clean_label_position >= sequence_length:
        raise ValueError("resolved KL clean_label_position is outside logits time range")
    if noisy_label_position <= 0 or noisy_label_position >= sequence_length:
        raise ValueError("resolved KL noisy_label_position is outside logits time range")
    clean_gt_bin = int(site.clean_gt_bin)
    if clean_gt_bin < 0 or clean_gt_bin > 999:
        raise ValueError("resolved KL clean_gt_bin must be in 0..999")
    support_bins = tuple(int(value) for value in site.support_bins)
    if not support_bins:
        raise ValueError("resolved KL support_bins must be non-empty")
    if len(set(support_bins)) != len(support_bins):
        raise ValueError("resolved KL support_bins must be unique")
    if any(value < 0 or value > 999 for value in support_bins):
        raise ValueError("resolved KL support_bins must stay in 0..999")
    if clean_gt_bin not in support_bins:
        raise ValueError("resolved KL clean_gt_bin must be present in support_bins")
    return (
        clean_label_position - 1,
        noisy_label_position - 1,
        support_bins,
        support_bins.index(clean_gt_bin),
    )


def _mean_row_value(rows: list[dict[str, float | str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / float(len(rows))


def _mean_metric_rows(rows: list[dict[str, float | str]]) -> Mapping[str, float]:
    keys = {
        key
        for row in rows
        for key, value in row.items()
        if key != "slot" and isinstance(value, (int, float))
    }
    return {
        key: sum(float(row[key]) for row in rows) / float(len(rows))
        for key in sorted(keys)
    }


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
    "PrefixDenoisingKLResult",
    "compute_branch_balanced_hard_ce",
    "compute_local_coord_kl",
    "coord_support_window",
    "topk_accuracy_from_logits",
]
