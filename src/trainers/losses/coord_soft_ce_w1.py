from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1


@dataclass(frozen=True)
class CoordSoftCEW1Result:
    coord_loss: torch.Tensor
    softce_contrib: torch.Tensor
    w1_contrib: torch.Tensor
    ce_contrib: torch.Tensor
    gate_contrib: torch.Tensor

    coord_tokens: int

    gate_mass_mean: torch.Tensor | None
    coord_acc_top5: torch.Tensor | None
    coord_p_gt_mean: torch.Tensor | None
    coord_margin_mean: torch.Tensor | None
    coord_expected_bin_mae: torch.Tensor | None
    coord_expected_bin_abs_err_p90: torch.Tensor | None
    coord_w1_to_delta: torch.Tensor | None


def mask_coord_targets(labels: torch.Tensor, coord_token_ids: list[int]) -> torch.Tensor:
    """Mask coord-token targets to -100 for the base full-vocab CE loss."""

    if labels.numel() == 0:
        return labels

    labels_safe = labels
    if labels_safe.min().item() < 0:
        labels_safe = labels_safe.clamp(min=0)

    max_label = int(labels_safe.max().item()) if labels_safe.numel() else -1
    max_coord = max(coord_token_ids) if coord_token_ids else -1
    size = max(max_label, max_coord) + 1
    if size <= 0:
        return labels

    lookup = torch.zeros(int(size), dtype=torch.bool, device=labels.device)
    ids = torch.tensor(coord_token_ids, device=labels.device, dtype=torch.long)
    valid = (ids >= 0) & (ids < lookup.numel())
    if valid.any().item():
        lookup[ids[valid]] = True

    mask = lookup[labels_safe] & (labels != -100)
    if not mask.any().item():
        return labels

    out = labels.clone()
    out[mask] = -100
    return out


def count_supervised_tokens(labels: torch.Tensor) -> int:
    if labels.ndim < 2:
        return 0
    labels_next = labels[:, 1:]
    return int((labels_next != -100).sum().detach().item())


def build_coord_id_map(
    *, vocab_size: int, device: torch.device, coord_token_ids: list[int]
) -> torch.Tensor:
    """Map full-vocab token id -> coord-bin index (or -1 if not a coord token)."""

    id_map = torch.full((int(vocab_size),), -1, dtype=torch.long, device=device)
    if not coord_token_ids:
        return id_map
    coord_ids = torch.tensor(coord_token_ids, device=device, dtype=torch.long)
    values = torch.arange(coord_ids.numel(), device=device, dtype=torch.long)
    valid = (coord_ids >= 0) & (coord_ids < int(vocab_size))
    if valid.any().item():
        id_map[coord_ids[valid]] = values[valid]
    return id_map


def coord_vocab_gate_loss(
    logits_full: torch.Tensor,
    logits_coord: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gate loss and mean coord-vocab mass (both fp32).

    For each position:
      mass_coord = sum_{i in coord_vocab} softmax(logits_full / T)[i]
      gate_loss = -log(mass_coord) = logsumexp(all/T) - logsumexp(coord/T)

    Returns:
      (gate_loss_per_token [N], mass_mean scalar)
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    full = torch.nan_to_num(
        logits_full.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-1e4, max=1e4) / float(temperature)
    coord = torch.nan_to_num(
        logits_coord.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-1e4, max=1e4) / float(temperature)

    lse_all = torch.logsumexp(full, dim=-1)
    lse_coord = torch.logsumexp(coord, dim=-1)
    gate = (lse_all - lse_coord).clamp(min=0.0)
    mass_mean = torch.exp((-gate).clamp(min=-50.0, max=50.0)).mean()
    gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=0.0)
    mass_mean = torch.nan_to_num(mass_mean, nan=0.0, posinf=1.0, neginf=0.0)
    return gate, mass_mean


def compute_coord_soft_ce_w1_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masked_labels: torch.Tensor,
    coord_token_ids: list[int],
    coord_id_map: torch.Tensor,
    cfg: Any,
    average_tokens_across_devices: bool,
    model_accepts_loss_kwargs: bool,
    accelerator_num_processes: int | None,
) -> CoordSoftCEW1Result | None:
    """Compute the coord_soft_ce_w1 loss term from a single forward pass.

    Returns None when there are no coord-token positions in this batch.

    Raises if config is enabled but prerequisites are invalid (e.g. coord vocab missing).
    """

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
            "coord_soft_ce_w1 enabled but no coord_token_ids were provided (tokenizer missing coord vocab?)"
        )
    if max(coord_token_ids) >= vocab_size:
        raise ValueError(
            f"coord_soft_ce_w1 enabled but coord token ids exceed vocab_size={vocab_size}. "
            "Ensure the tokenizer provides a full 1000-bin coord vocab."
        )

    labels_safe = labels_next
    if labels_safe.numel() > 0 and labels_safe.min().item() < 0:
        labels_safe = labels_safe.clamp(min=0)

    target_bins_all = coord_id_map[labels_safe].to(dtype=torch.long)
    coord_positions_mask = (target_bins_all >= 0) & (labels_next != -100)
    if not coord_positions_mask.any().item():
        return None

    flat_logits_full = logits_next[coord_positions_mask]
    flat_target_bins = target_bins_all[coord_positions_mask]

    idx = torch.tensor(coord_token_ids, device=flat_logits_full.device, dtype=torch.long)
    flat_logits = flat_logits_full.index_select(-1, idx)

    temperature = float(getattr(cfg, "temperature", 1.0))
    ce_weight = float(getattr(cfg, "ce_weight", 0.0))
    soft_ce_weight = float(getattr(cfg, "soft_ce_weight", 1.0))
    w1_weight = float(getattr(cfg, "w1_weight", 1.0))
    gate_weight = float(getattr(cfg, "gate_weight", 0.0))

    out = coord_soft_ce_w1(
        flat_logits,
        flat_target_bins,
        sigma=float(getattr(cfg, "target_sigma", 2.0)),
        truncate=getattr(cfg, "target_truncate", None),
        temperature=temperature,
        # Weighting is applied at the trainer level so we can include the gate term.
        soft_ce_weight=1.0,
        w1_weight=1.0,
        normalize_w1=True,
    )

    # Coord distribution monitors (coord vocab only).
    coord_acc_top5 = None
    coord_p_gt_mean = None
    coord_margin_mean = None
    coord_expected_bin_mae = None
    coord_expected_bin_abs_err_p90 = None
    coord_w1_to_delta = None
    try:
        with torch.no_grad():
            k5 = min(5, int(flat_logits.shape[-1]))
            topk = flat_logits.topk(k=k5, dim=-1).indices
            coord_acc_top5 = (
                (topk == flat_target_bins.unsqueeze(-1)).any(dim=-1).float().mean()
            )
            coord_p_gt_mean = out.pred_probs.gather(1, flat_target_bins.view(-1, 1)).mean()

            bins = torch.arange(
                int(out.pred_probs.shape[-1]),
                device=out.pred_probs.device,
                dtype=out.pred_probs.dtype,
            )
            pred_expected = (out.pred_probs * bins.view(1, -1)).sum(dim=-1)
            abs_err = (pred_expected.float() - flat_target_bins.float()).abs()
            coord_expected_bin_mae = abs_err.mean()

            # W1(p, delta_t) in 1D bins, equivalently E_p[|k - t|].
            probs = out.pred_probs.to(dtype=torch.float32)
            bins_f = bins.to(dtype=torch.float32)
            dist_bins = (
                bins_f.view(1, -1) - flat_target_bins.to(torch.float32).view(-1, 1)
            ).abs()
            coord_w1_to_delta = (probs * dist_bins).sum(dim=-1).mean()

            if abs_err.numel() > 0:
                coord_expected_bin_abs_err_p90 = torch.quantile(
                    abs_err.to(dtype=torch.float32), 0.9
                )

            temp = float(temperature) if float(temperature) > 0 else 1.0
            logits_scaled = flat_logits.float() / temp
            gt_logit = logits_scaled.gather(1, flat_target_bins.view(-1, 1)).squeeze(1)
            max_logit = logits_scaled.max(dim=-1).values
            coord_margin_mean = (max_logit - gt_logit).mean()
    except Exception:
        coord_acc_top5 = None
        coord_p_gt_mean = None
        coord_margin_mean = None
        coord_expected_bin_mae = None
        coord_expected_bin_abs_err_p90 = None
        coord_w1_to_delta = None

    # Loss terms (token-summed before normalization).
    softce_sum = out.soft_ce_per_token.sum()
    w1_sum = out.w1_per_token.sum()
    ce_sum = softce_sum.new_tensor(0.0)
    if ce_weight != 0.0:
        ce_per_token = F.cross_entropy(flat_logits.float(), flat_target_bins, reduction="none")
        ce_per_token = torch.nan_to_num(ce_per_token, nan=0.0, posinf=1e4, neginf=0.0)
        ce_sum = ce_per_token.sum()

    gate_sum = softce_sum.new_tensor(0.0)
    gate_mass_mean = None
    if gate_weight != 0.0:
        gate_per_token, gate_mass_mean = coord_vocab_gate_loss(
            flat_logits_full, flat_logits, temperature=float(temperature) if float(temperature) > 0 else 1.0
        )
        gate_sum = gate_per_token.sum()

    denom = coord_positions_mask.sum().to(dtype=torch.float32)
    if average_tokens_across_devices and model_accepts_loss_kwargs and dist.is_available() and dist.is_initialized():
        denom_global = denom.detach().clone()
        dist.all_reduce(denom_global, op=dist.ReduceOp.SUM)
        denom = denom_global
    denom = torch.where(denom > 0, denom, denom.new_tensor(1.0))

    softce_loss = softce_sum / denom
    w1_loss = w1_sum / denom
    ce_loss = ce_sum / denom
    gate_loss = gate_sum / denom

    softce_contrib = soft_ce_weight * softce_loss
    w1_contrib = w1_weight * w1_loss
    ce_contrib = ce_weight * ce_loss
    gate_contrib = gate_weight * gate_loss
    coord_loss = ce_contrib + softce_contrib + w1_contrib + gate_contrib

    if average_tokens_across_devices and model_accepts_loss_kwargs:
        try:
            if dist.is_available() and dist.is_initialized():
                scale = float(dist.get_world_size())
            else:
                scale = float(accelerator_num_processes or 1)
        except Exception:
            scale = 1.0
        coord_loss = coord_loss * scale
        softce_contrib = softce_contrib * scale
        w1_contrib = w1_contrib * scale
        ce_contrib = ce_contrib * scale
        gate_contrib = gate_contrib * scale

    coord_loss = torch.nan_to_num(coord_loss, nan=0.0, posinf=1e4, neginf=0.0)
    softce_contrib = torch.nan_to_num(softce_contrib, nan=0.0, posinf=1e4, neginf=0.0)
    w1_contrib = torch.nan_to_num(w1_contrib, nan=0.0, posinf=1e4, neginf=0.0)
    ce_contrib = torch.nan_to_num(ce_contrib, nan=0.0, posinf=1e4, neginf=0.0)
    gate_contrib = torch.nan_to_num(gate_contrib, nan=0.0, posinf=1e4, neginf=0.0)

    coord_tokens = int(coord_positions_mask.sum().detach().item())

    return CoordSoftCEW1Result(
        coord_loss=coord_loss,
        softce_contrib=softce_contrib,
        w1_contrib=w1_contrib,
        ce_contrib=ce_contrib,
        gate_contrib=gate_contrib,
        coord_tokens=coord_tokens,
        gate_mass_mean=gate_mass_mean,
        coord_acc_top5=coord_acc_top5,
        coord_p_gt_mean=coord_p_gt_mean,
        coord_margin_mean=coord_margin_mean,
        coord_expected_bin_mae=coord_expected_bin_mae,
        coord_expected_bin_abs_err_p90=coord_expected_bin_abs_err_p90,
        coord_w1_to_delta=coord_w1_to_delta,
    )
