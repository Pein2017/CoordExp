from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1

TOKEN_TYPE_DESC = 1
TOKEN_TYPE_FORMAT = 3


@dataclass(frozen=True)
class CoordSoftCEW1Result:
    coord_loss: torch.Tensor
    softce_contrib: torch.Tensor
    w1_contrib: torch.Tensor
    ce_contrib: torch.Tensor
    gate_contrib: torch.Tensor
    text_gate_contrib: torch.Tensor
    adjacent_repulsion_contrib: torch.Tensor

    coord_tokens: int

    gate_mass_mean: torch.Tensor | None
    text_gate_coord_mass_mean: torch.Tensor | None
    coord_acc_top5: torch.Tensor | None
    coord_p_gt_mean: torch.Tensor | None
    coord_margin_mean: torch.Tensor | None
    coord_expected_bin_mae: torch.Tensor | None
    coord_expected_bin_abs_err_p90: torch.Tensor | None
    coord_w1_to_delta: torch.Tensor | None
    adjacent_repulsion_pair_count: int
    adjacent_repulsion_applied_count: int
    adjacent_repulsion_copy_score_mean: torch.Tensor | None


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


def coord_vocab_text_gate_loss(
    logits_full: torch.Tensor,
    logits_coord: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize coord-vocab mass leaking onto supervised non-coord positions."""

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
    coord_mass = torch.exp((lse_coord - lse_all).clamp(min=-50.0, max=0.0))
    non_coord_mass = (1.0 - coord_mass).clamp(min=1e-6, max=1.0)
    gate = -torch.log(non_coord_mass)
    gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=0.0)
    mass_mean = torch.nan_to_num(coord_mass.mean(), nan=0.0, posinf=1.0, neginf=0.0)
    return gate, mass_mean


def compute_coord_soft_ce_w1_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masked_labels: torch.Tensor,
    coord_token_weights: torch.Tensor | None,
    coord_token_ids: list[int],
    coord_id_map: torch.Tensor,
    tokenizer: Any | None,
    token_types: torch.Tensor | None = None,
    cfg: Any,
    average_tokens_across_devices: bool,
    model_accepts_loss_kwargs: bool,
    accelerator_num_processes: int | None,
    object_field_order: str = "desc_first",
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
    coord_position_weights = None
    if isinstance(coord_token_weights, torch.Tensor):
        if tuple(coord_token_weights.shape) != tuple(labels.shape):
            raise ValueError(
                "coord_token_weights must match labels shape when provided"
            )
        coord_weights_next = coord_token_weights[:, 1 : seq_len + 1]
        coord_position_weights = coord_weights_next[coord_positions_mask].to(
            dtype=torch.float32
        )

    # Token-averaging across devices uses distributed collectives. Even when there are
    # no coord-token positions locally, we still participate in the denom reduction so
    # callers never deadlock due to conditional collectives.
    if coord_position_weights is None:
        denom_local = coord_positions_mask.sum().to(dtype=torch.float32)
    else:
        denom_local = coord_position_weights.sum().to(dtype=torch.float32)
    denom = denom_local
    if average_tokens_across_devices and model_accepts_loss_kwargs and dist.is_available() and dist.is_initialized():
        denom_global = denom_local.detach().clone()
        dist.all_reduce(denom_global, op=dist.ReduceOp.SUM)
        denom = denom_global

    if float(denom_local.detach().item()) <= 0.0:
        return None

    flat_logits_full = logits_next[coord_positions_mask]
    flat_target_bins = target_bins_all[coord_positions_mask]
    if coord_position_weights is None:
        coord_position_weights = torch.ones(
            (int(flat_target_bins.numel()),),
            dtype=torch.float32,
            device=flat_target_bins.device,
        )
    else:
        coord_position_weights = coord_position_weights.to(device=flat_target_bins.device)

    idx = torch.tensor(coord_token_ids, device=flat_logits_full.device, dtype=torch.long)
    flat_logits = flat_logits_full.index_select(-1, idx)

    temperature = float(getattr(cfg, "temperature", 1.0))
    ce_weight = float(getattr(cfg, "ce_weight", 0.0))
    soft_ce_weight = float(getattr(cfg, "soft_ce_weight", 1.0))
    w1_weight = float(getattr(cfg, "w1_weight", 1.0))
    gate_weight = float(getattr(cfg, "gate_weight", 0.0))
    text_gate_weight = float(getattr(cfg, "text_gate_weight", 0.0))

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
    with torch.no_grad():
        k5 = min(5, int(flat_logits.shape[-1]))
        topk = flat_logits.topk(k=k5, dim=-1).indices
        coord_acc_top5 = (topk == flat_target_bins.unsqueeze(-1)).any(dim=-1).float().mean()
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

    # Loss terms (token-summed before normalization).
    token_weights = coord_position_weights.to(dtype=out.soft_ce_per_token.dtype)
    softce_sum = (out.soft_ce_per_token * token_weights).sum()
    w1_sum = (out.w1_per_token * token_weights).sum()
    ce_sum = softce_sum.new_tensor(0.0)
    if ce_weight != 0.0:
        ce_per_token = F.cross_entropy(flat_logits.float(), flat_target_bins, reduction="none")
        ce_per_token = torch.nan_to_num(ce_per_token, nan=0.0, posinf=1e4, neginf=0.0)
        ce_sum = (ce_per_token * token_weights.to(dtype=ce_per_token.dtype)).sum()

    gate_sum = softce_sum.new_tensor(0.0)
    gate_mass_mean = None
    if gate_weight != 0.0:
        gate_per_token, gate_mass_mean = coord_vocab_gate_loss(
            flat_logits_full, flat_logits, temperature=float(temperature) if float(temperature) > 0 else 1.0
        )
        gate_sum = (gate_per_token * token_weights.to(dtype=gate_per_token.dtype)).sum()

    text_gate_contrib = softce_sum.new_tensor(0.0)
    text_gate_sum = softce_sum.new_tensor(0.0)
    text_gate_coord_mass_mean = None
    if text_gate_weight != 0.0:
        text_positions_mask = masked_labels[:, 1 : seq_len + 1] != -100
        if isinstance(token_types, torch.Tensor) and tuple(token_types.shape) == tuple(
            labels.shape
        ):
            token_types_next = token_types[:, 1 : seq_len + 1]
            text_positions_mask = text_positions_mask & (
                (token_types_next == TOKEN_TYPE_DESC)
                | (token_types_next == TOKEN_TYPE_FORMAT)
            )
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            text_positions_mask = text_positions_mask & (labels_next != eos_token_id)

        if bool(text_positions_mask.any().item()):
            text_logits_full = logits_next[text_positions_mask]
            coord_ids = torch.tensor(
                coord_token_ids,
                device=text_logits_full.device,
                dtype=torch.long,
            )
            valid = (coord_ids >= 0) & (coord_ids < int(text_logits_full.shape[-1]))
            coord_ids = coord_ids[valid]
            if int(coord_ids.numel()) > 0:
                text_logits_coord = text_logits_full.index_select(dim=-1, index=coord_ids)
                text_gate_per_token, text_gate_coord_mass_mean = (
                    coord_vocab_text_gate_loss(
                    text_logits_full,
                    text_logits_coord,
                    temperature=float(temperature)
                    if float(temperature) > 0
                    else 1.0,
                    )
                )
                text_gate_sum = text_gate_per_token.sum().to(dtype=softce_sum.dtype)

                text_gate_denom_local = torch.tensor(
                    float(int(text_gate_per_token.numel())),
                    device=text_gate_sum.device,
                    dtype=text_gate_sum.dtype,
                )
                text_gate_denom = text_gate_denom_local
                if (
                    average_tokens_across_devices
                    and model_accepts_loss_kwargs
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    text_gate_denom = text_gate_denom_local.detach().clone()
                    dist.all_reduce(text_gate_denom, op=dist.ReduceOp.SUM)
                text_gate_denom = torch.where(
                    text_gate_denom > 0,
                    text_gate_denom,
                    text_gate_denom.new_tensor(1.0),
                )
                text_gate_loss = text_gate_sum / text_gate_denom
                text_gate_contrib = float(text_gate_weight) * text_gate_loss

    denom = torch.where(denom > 0, denom, denom.new_tensor(1.0))

    softce_loss = softce_sum / denom
    w1_loss = w1_sum / denom
    ce_loss = ce_sum / denom
    gate_loss = gate_sum / denom

    softce_contrib = soft_ce_weight * softce_loss
    w1_contrib = w1_weight * w1_loss
    ce_contrib = ce_weight * ce_loss
    gate_contrib = gate_weight * gate_loss
    adjacent_repulsion_contrib = softce_sum.new_tensor(0.0)
    adjacent_repulsion_pair_count = 0
    adjacent_repulsion_applied_count = 0
    adjacent_repulsion_copy_score_mean = None
    adjacent_repulsion_weight = max(
        0.0, float(getattr(cfg, "adjacent_repulsion_weight", 0.0))
    )
    if adjacent_repulsion_weight != 0.0:
        from src.trainers.teacher_forcing.adjacent_repulsion import (
            compute_adjacent_repulsion,
        )
        from src.trainers.teacher_forcing.stage1 import extract_stage1_bbox_quartets

        filter_mode = str(
            getattr(cfg, "adjacent_repulsion_filter_mode", "same_desc")
        ).strip().lower()
        quartets = extract_stage1_bbox_quartets(
            logits=logits,
            labels=labels,
            coord_token_ids=coord_token_ids,
            coord_id_map=coord_id_map,
            tokenizer=tokenizer,
            include_adjacent_metadata=True,
            require_desc_keys=(filter_mode == "same_desc"),
            object_field_order=object_field_order,
        )
        if (
            quartets is not None
            and isinstance(quartets.adjacent_prev_target_bins, torch.Tensor)
            and isinstance(quartets.adjacent_has_prev_mask, torch.Tensor)
            and isinstance(quartets.adjacent_same_desc_prev_mask, torch.Tensor)
            and int(quartets.coord_logits.numel()) > 0
        ):
            adjacent_result = compute_adjacent_repulsion(
                coord_logits=quartets.coord_logits.reshape(
                    -1, 4, flat_logits.shape[-1]
                ),
                prev_target_bins=quartets.adjacent_prev_target_bins,
                has_previous_mask=quartets.adjacent_has_prev_mask,
                same_desc_mask=quartets.adjacent_same_desc_prev_mask,
                margin_ratio=max(
                    0.0, float(getattr(cfg, "adjacent_repulsion_margin_ratio", 0.05))
                ),
                copy_margin=float(
                    getattr(cfg, "adjacent_repulsion_copy_margin", 0.8)
                ),
                filter_mode=filter_mode,
                temperature=temperature,
                group_weights=None,
            )
            adjacent_repulsion_contrib = (
                float(adjacent_repulsion_weight) * adjacent_result.loss
            )
            adjacent_repulsion_pair_count = int(adjacent_result.pair_count)
            adjacent_repulsion_applied_count = int(adjacent_result.applied_count)
            adjacent_repulsion_copy_score_mean = adjacent_result.copy_score_mean

    coord_loss = (
        ce_contrib
        + softce_contrib
        + w1_contrib
        + gate_contrib
        + text_gate_contrib
        + adjacent_repulsion_contrib
    )

    if average_tokens_across_devices and model_accepts_loss_kwargs:
        if dist.is_available() and dist.is_initialized():
            scale = float(dist.get_world_size())
        else:
            scale = float(accelerator_num_processes or 1)
        softce_contrib = softce_contrib * scale
        w1_contrib = w1_contrib * scale
        ce_contrib = ce_contrib * scale
        gate_contrib = gate_contrib * scale
        text_gate_contrib = text_gate_contrib * scale
    coord_loss = (
        ce_contrib
        + softce_contrib
        + w1_contrib
        + gate_contrib
        + text_gate_contrib
        + adjacent_repulsion_contrib
    )

    coord_loss = torch.nan_to_num(coord_loss, nan=0.0, posinf=1e4, neginf=0.0)
    softce_contrib = torch.nan_to_num(softce_contrib, nan=0.0, posinf=1e4, neginf=0.0)
    w1_contrib = torch.nan_to_num(w1_contrib, nan=0.0, posinf=1e4, neginf=0.0)
    ce_contrib = torch.nan_to_num(ce_contrib, nan=0.0, posinf=1e4, neginf=0.0)
    gate_contrib = torch.nan_to_num(gate_contrib, nan=0.0, posinf=1e4, neginf=0.0)
    text_gate_contrib = torch.nan_to_num(
        text_gate_contrib, nan=0.0, posinf=1e4, neginf=0.0
    )
    adjacent_repulsion_contrib = torch.nan_to_num(
        adjacent_repulsion_contrib, nan=0.0, posinf=1e4, neginf=0.0
    )

    coord_tokens = int(coord_positions_mask.sum().detach().item())

    return CoordSoftCEW1Result(
        coord_loss=coord_loss,
        softce_contrib=softce_contrib,
        w1_contrib=w1_contrib,
        ce_contrib=ce_contrib,
        gate_contrib=gate_contrib,
        text_gate_contrib=text_gate_contrib,
        adjacent_repulsion_contrib=adjacent_repulsion_contrib,
        coord_tokens=coord_tokens,
        gate_mass_mean=gate_mass_mean,
        text_gate_coord_mass_mean=text_gate_coord_mass_mean,
        coord_acc_top5=coord_acc_top5,
        coord_p_gt_mean=coord_p_gt_mean,
        coord_margin_mean=coord_margin_mean,
        coord_expected_bin_mae=coord_expected_bin_mae,
        coord_expected_bin_abs_err_p90=coord_expected_bin_abs_err_p90,
        coord_w1_to_delta=coord_w1_to_delta,
        adjacent_repulsion_pair_count=int(adjacent_repulsion_pair_count),
        adjacent_repulsion_applied_count=int(adjacent_repulsion_applied_count),
        adjacent_repulsion_copy_score_mean=adjacent_repulsion_copy_score_mean,
    )
