from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

AdjacentRepulsionFilterMode = Literal["same_desc", "global"]


@dataclass(frozen=True)
class AdjacentRepulsionResult:
    loss: torch.Tensor
    pair_count: int
    applied_count: int
    copy_score_mean: torch.Tensor | None


def normalize_adjacent_repulsion_filter_mode(
    value: object,
    *,
    path: str = "adjacent_repulsion_filter_mode",
) -> AdjacentRepulsionFilterMode:
    normalized = str(value or "same_desc").strip().lower()
    if normalized not in {"same_desc", "global"}:
        raise ValueError(
            f"{path} must be one of ['global', 'same_desc']; got {value!r}"
        )
    return "global" if normalized == "global" else "same_desc"


def compute_adjacent_repulsion(
    *,
    coord_logits: torch.Tensor,
    prev_target_bins: torch.Tensor,
    has_previous_mask: torch.Tensor,
    same_desc_mask: torch.Tensor | None,
    filter_mode: AdjacentRepulsionFilterMode,
    margin_ratio: float,
    copy_margin: float,
    temperature: float,
    group_weights: torch.Tensor | None = None,
) -> AdjacentRepulsionResult:
    if coord_logits.ndim != 3 or int(coord_logits.shape[1]) != 4:
        raise ValueError("coord_logits must have shape [groups, 4, coord_vocab]")
    if prev_target_bins.shape != coord_logits.shape[:2]:
        raise ValueError("prev_target_bins must align with coord_logits[:2]")
    if has_previous_mask.shape != coord_logits.shape[:1]:
        raise ValueError("has_previous_mask must align with coord_logits[:, 0]")
    if same_desc_mask is not None and same_desc_mask.shape != coord_logits.shape[:1]:
        raise ValueError("same_desc_mask must align with coord_logits[:, 0]")
    if group_weights is not None and group_weights.shape != coord_logits.shape[:1]:
        raise ValueError("group_weights must align with coord_logits[:, 0]")

    device = coord_logits.device
    dtype = coord_logits.dtype
    z = coord_logits.new_tensor(0.0)

    pair_mask = has_previous_mask.to(device=device, dtype=torch.bool)
    pair_count = int(pair_mask.sum().detach().item())
    if pair_count <= 0:
        return AdjacentRepulsionResult(
            loss=z,
            pair_count=0,
            applied_count=0,
            copy_score_mean=None,
        )

    mode = normalize_adjacent_repulsion_filter_mode(filter_mode)
    applied_mask = pair_mask.clone()
    if mode == "same_desc":
        if same_desc_mask is None:
            raise ValueError(
                "same_desc adjacent repulsion requires same_desc_mask to be provided"
            )
        applied_mask &= same_desc_mask.to(device=device, dtype=torch.bool)

    applied_count = int(applied_mask.sum().detach().item())
    if applied_count <= 0:
        return AdjacentRepulsionResult(
            loss=z,
            pair_count=pair_count,
            applied_count=0,
            copy_score_mean=None,
        )

    temp_safe = max(float(temperature), 1e-6)
    probs = torch.softmax(coord_logits.float() / temp_safe, dim=-1)

    prev_bins = prev_target_bins.to(device=device, dtype=torch.float32)
    widths = (prev_bins[:, 2] - prev_bins[:, 0]).abs().clamp(min=1.0)
    heights = (prev_bins[:, 3] - prev_bins[:, 1]).abs().clamp(min=1.0)

    margin_ratio_f = max(float(margin_ratio), 0.0)
    half_width_x = torch.round(widths * margin_ratio_f).clamp(min=1.0)
    half_width_y = torch.round(heights * margin_ratio_f).clamp(min=1.0)
    slot_half_widths = torch.stack(
        [half_width_x, half_width_y, half_width_x, half_width_y],
        dim=1,
    )

    bins = torch.arange(
        int(coord_logits.shape[-1]),
        device=device,
        dtype=torch.float32,
    ).view(1, 1, -1)
    centers = prev_bins.unsqueeze(-1)
    half_widths = slot_half_widths.unsqueeze(-1)
    band = (1.0 - ((bins - centers).abs() / half_widths)).clamp(min=0.0, max=1.0)
    overlap = (probs * band).sum(dim=-1).clamp(min=0.0, max=1.0)
    copy_scores = overlap.clamp(min=1e-12, max=1.0).prod(dim=-1).pow(0.25)

    copy_margin_f = float(copy_margin)
    penalties = torch.relu(copy_scores - copy_margin_f).pow(2.0).to(dtype=dtype)
    mask_f = applied_mask.to(device=device, dtype=dtype)
    masked_penalties = penalties * mask_f

    if group_weights is not None:
        weights = (
            group_weights.to(device=device, dtype=dtype).clamp(min=0.0) * mask_f
        )
        denom = weights.sum().clamp(min=1e-6)
        loss = masked_penalties.mul(weights).sum() / denom
    else:
        denom = mask_f.sum().clamp(min=1.0)
        loss = masked_penalties.sum() / denom

    copy_score_mean = copy_scores[applied_mask].mean().to(dtype=dtype)
    return AdjacentRepulsionResult(
        loss=loss,
        pair_count=pair_count,
        applied_count=applied_count,
        copy_score_mean=copy_score_mean,
    )


def compute_adjacent_repulsion_loss(
    *,
    coord_logits_groups: torch.Tensor,
    prev_target_bins: torch.Tensor,
    has_prev_mask: torch.Tensor,
    same_desc_prev_mask: torch.Tensor | None,
    margin_ratio: float,
    copy_margin: float,
    filter_mode: AdjacentRepulsionFilterMode | str,
    temperature: float,
    group_weights: torch.Tensor | None = None,
) -> AdjacentRepulsionResult:
    return compute_adjacent_repulsion(
        coord_logits=coord_logits_groups,
        prev_target_bins=prev_target_bins,
        has_previous_mask=has_prev_mask,
        same_desc_mask=same_desc_prev_mask,
        filter_mode=normalize_adjacent_repulsion_filter_mode(filter_mode),
        margin_ratio=margin_ratio,
        copy_margin=copy_margin,
        temperature=temperature,
        group_weights=group_weights,
    )


__all__ = [
    "AdjacentRepulsionFilterMode",
    "AdjacentRepulsionResult",
    "compute_adjacent_repulsion",
    "compute_adjacent_repulsion_loss",
    "normalize_adjacent_repulsion_filter_mode",
]
