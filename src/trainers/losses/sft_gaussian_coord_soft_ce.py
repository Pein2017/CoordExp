from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class SFTGaussianCoordSoftCEResult:
    loss: torch.Tensor
    coord_tokens: int
    target_entropy: torch.Tensor
    target_peak_prob: torch.Tensor
    target_r95_radius_mean: torch.Tensor
    target_r95_radius_max: torch.Tensor
    coord_acc_top5: torch.Tensor
    coord_p_gt_mean: torch.Tensor
    expected_bin_mae: torch.Tensor


def _r95_radius_variance(radius: torch.Tensor) -> torch.Tensor:
    sigma = radius.to(dtype=torch.float32) / 1.96
    return sigma * sigma


def _infer_coord_r95_radii(
    *,
    labels_next: torch.Tensor,
    target_bins_all: torch.Tensor,
    coord_positions_mask: torch.Tensor,
    gaussian_r95_axis_fraction: float,
    gaussian_r95_cap_bins: int,
) -> torch.Tensor:
    """Infer per-coordinate R95 radii from compact-full xyxy coord quads.

    The packed SFT path intentionally avoids recursive atom-position sidecars.
    Compact-full rows still encode each box as four adjacent coord tokens, so
    we can recover object-local x/y axis lengths from the shifted label stream.
    If a position cannot be assigned to a well-formed coord quad, fall back to
    the configured cap radius.
    """

    radius_all = torch.full_like(
        target_bins_all,
        fill_value=int(gaussian_r95_cap_bins),
        dtype=torch.long,
    )
    batch, seq_len = target_bins_all.shape
    for row in range(int(batch)):
        idxs = torch.nonzero(coord_positions_mask[row], as_tuple=False).flatten()
        if int(idxs.numel()) < 4:
            continue
        values = target_bins_all[row]
        labels_row = labels_next[row]
        start = 0
        while start < int(idxs.numel()):
            end = start + 1
            while end < int(idxs.numel()) and int(idxs[end].item()) == int(idxs[end - 1].item()) + 1:
                end += 1
            run = idxs[start:end]
            usable = (int(run.numel()) // 4) * 4
            if usable > 0:
                run = run[:usable].view(-1, 4)
                for quad in run:
                    # Require the labels really are four supervised coord tokens.
                    if not bool((labels_row[quad] != -100).all().item()):
                        continue
                    x1, y1, x2, y2 = [int(v) for v in values[quad].tolist()]
                    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
                        continue
                    x_radius = int(
                        math.floor(
                            min(
                                float(gaussian_r95_cap_bins),
                                float(gaussian_r95_axis_fraction) * float(x2 - x1),
                            )
                        )
                    )
                    y_radius = int(
                        math.floor(
                            min(
                                float(gaussian_r95_cap_bins),
                                float(gaussian_r95_axis_fraction) * float(y2 - y1),
                            )
                        )
                    )
                    radius_all[row, quad[0]] = x_radius
                    radius_all[row, quad[2]] = x_radius
                    radius_all[row, quad[1]] = y_radius
                    radius_all[row, quad[3]] = y_radius
            start = end
    return radius_all[coord_positions_mask]


def compute_sft_gaussian_coord_soft_ce_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    coord_token_ids: list[int],
    coord_id_map: torch.Tensor,
    cfg: Any,
    average_tokens_across_devices: bool,
    model_accepts_loss_kwargs: bool,
    accelerator_num_processes: int | None,
) -> SFTGaussianCoordSoftCEResult | None:
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch.Tensors")

    seq_len = min(int(logits.shape[1]), max(int(labels.shape[1]) - 1, 0))
    if seq_len <= 0:
        return None

    logits_next = logits[:, :seq_len, :]
    labels_next = labels[:, 1 : seq_len + 1]
    vocab_size = int(logits_next.shape[-1])
    if not coord_token_ids:
        raise RuntimeError(
            "sft gaussian coord soft CE enabled but no coord token ids were found"
        )
    if max(coord_token_ids) >= vocab_size:
        raise ValueError(
            "sft gaussian coord soft CE coord token id exceeds logits vocab size"
        )

    labels_safe = labels_next.clamp(min=0)
    target_bins_all = coord_id_map[labels_safe].to(dtype=torch.long)
    coord_positions_mask = (target_bins_all >= 0) & (labels_next != -100)

    denom_local = coord_positions_mask.sum().to(dtype=torch.float32)
    denom = denom_local
    if (
        average_tokens_across_devices
        and model_accepts_loss_kwargs
        and dist.is_available()
        and dist.is_initialized()
    ):
        denom = denom_local.detach().clone()
        dist.all_reduce(denom, op=dist.ReduceOp.SUM)

    if float(denom_local.detach().item()) <= 0.0:
        return None

    flat_logits_full = logits_next[coord_positions_mask]
    flat_target_bins = target_bins_all[coord_positions_mask]
    coord_ids = torch.tensor(coord_token_ids, device=logits.device, dtype=torch.long)
    flat_logits = flat_logits_full.index_select(-1, coord_ids)

    mix = float(getattr(cfg, "gaussian_mixture_weight", 0.5))
    r95_fraction = float(getattr(cfg, "gaussian_r95_axis_fraction", 0.04))
    cap_bins = int(getattr(cfg, "gaussian_r95_cap_bins", 8))

    radii = _infer_coord_r95_radii(
        labels_next=labels_next,
        target_bins_all=target_bins_all,
        coord_positions_mask=coord_positions_mask,
        gaussian_r95_axis_fraction=r95_fraction,
        gaussian_r95_cap_bins=cap_bins,
    ).to(device=flat_logits.device)

    bins = torch.arange(
        int(len(coord_token_ids)),
        device=flat_logits.device,
        dtype=torch.float32,
    )
    target_float = flat_target_bins.to(dtype=torch.float32)
    distances = bins.view(1, -1) - target_float.view(-1, 1)
    gaussian = torch.zeros_like(distances, dtype=torch.float32)
    positive = radii > 0
    if bool(positive.any().item()):
        variance = _r95_radius_variance(radii[positive]).view(-1, 1)
        gaussian_pos = torch.exp(-0.5 * distances[positive].pow(2) / variance)
        gaussian_pos = gaussian_pos / gaussian_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        gaussian[positive] = gaussian_pos
    if bool((~positive).any().item()):
        gaussian[~positive] = F.one_hot(
            flat_target_bins[~positive],
            num_classes=int(len(coord_token_ids)),
        ).to(dtype=torch.float32)

    target_probs = gaussian * mix
    target_probs.scatter_add_(
        1,
        flat_target_bins.view(-1, 1),
        torch.full(
            (int(flat_target_bins.numel()), 1),
            fill_value=1.0 - mix,
            dtype=target_probs.dtype,
            device=target_probs.device,
        ),
    )
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    coord_log_probs = F.log_softmax(flat_logits.float(), dim=-1)
    per_token = -(target_probs.to(dtype=coord_log_probs.dtype) * coord_log_probs).sum(dim=-1)
    loss_sum = torch.nan_to_num(per_token, nan=0.0, posinf=1e4, neginf=0.0).sum()
    denom = torch.where(denom > 0, denom, denom.new_tensor(1.0))
    loss = loss_sum / denom
    if average_tokens_across_devices and model_accepts_loss_kwargs:
        if dist.is_available() and dist.is_initialized():
            scale = float(dist.get_world_size())
        else:
            scale = float(accelerator_num_processes or 1)
        loss = loss * scale
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

    with torch.no_grad():
        probs = F.softmax(flat_logits.float(), dim=-1)
        k5 = min(5, int(flat_logits.shape[-1]))
        top5 = flat_logits.topk(k=k5, dim=-1).indices
        coord_acc_top5 = (top5 == flat_target_bins.view(-1, 1)).any(dim=-1).float().mean()
        coord_p_gt_mean = probs.gather(1, flat_target_bins.view(-1, 1)).mean()
        pred_expected = (probs * bins.view(1, -1)).sum(dim=-1)
        expected_bin_mae = (pred_expected - target_float).abs().mean()
        positive_probs = target_probs > 0
        entropy = -(
            target_probs[positive_probs] * target_probs[positive_probs].log()
        ).sum() / target_probs.shape[0]
        peak_prob = target_probs.max(dim=-1).values.mean()
        r95_mean = radii.to(dtype=torch.float32).mean()
        r95_max = radii.to(dtype=torch.float32).max()

    return SFTGaussianCoordSoftCEResult(
        loss=loss,
        coord_tokens=int(coord_positions_mask.sum().detach().item()),
        target_entropy=entropy,
        target_peak_prob=peak_prob,
        target_r95_radius_mean=r95_mean,
        target_r95_radius_max=r95_max,
        coord_acc_top5=coord_acc_top5,
        coord_p_gt_mean=coord_p_gt_mean,
        expected_bin_mae=expected_bin_mae,
    )
