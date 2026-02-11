"""Coord monitor diagnostics for CoordExp training (canonical import path).

Historically these helpers lived under `src.trainers.metrics`.
`src.trainers.metrics.coord_monitors` now remains as a compatibility shim.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from src.coord_tokens.codec import get_coord_token_ids
from src.coord_tokens.soft_ce_w1 import coord_soft_ce_w1
from src.data_collators.token_types import TokenType


def compute_coord_flip_and_mass_metrics(
    *,
    logits_next: torch.Tensor,
    supervised_mask: torch.Tensor,
    preds_masked: torch.Tensor,
    labels_masked: torch.Tensor,
    types_masked: torch.Tensor | None,
    coord_token_ids: list[int],
    coord_monitor_mass: bool,
    coord_monitor_mass_max_tokens: int,
    temperature: float,
) -> dict[str, float]:
    """Compute coord monitor diagnostics (aggregate-only; metrics-only).

    This includes:
    - flip rates (argmax-based)
    - coord-vocab softmax mass conditioned on GT type (optional)

    Expected skips (return {}): missing coord vocab, empty masks, etc.
    """

    out: dict[str, float] = {}

    vocab_size = int(logits_next.shape[-1])
    if vocab_size <= 0:
        return out

    if not coord_token_ids:
        return out

    ids = torch.tensor(coord_token_ids, device=logits_next.device, dtype=torch.long)
    valid = (ids >= 0) & (ids < vocab_size)
    if not valid.any().item():
        return out

    coord_lookup = torch.zeros((vocab_size,), dtype=torch.bool, device=logits_next.device)
    coord_lookup[ids[valid]] = True

    # Predicted token type (coord vs noncoord) for supervised tokens.
    pred_is_coord_masked = coord_lookup[preds_masked]

    # GT token type masks over supervised tokens.
    labels_safe = labels_masked
    if labels_safe.numel() > 0 and labels_safe.min().item() < 0:
        labels_safe = labels_safe.clamp(min=0)
    gt_coord = coord_lookup[labels_safe.clamp(min=0, max=vocab_size - 1)]
    gt_text = ~gt_coord
    gt_format = None
    gt_desc = None

    if types_masked is not None:
        gt_coord = types_masked == TokenType.COORD
        gt_format = types_masked == TokenType.FORMAT
        gt_desc = types_masked == TokenType.DESC
        gt_text = (types_masked != TokenType.COORD) & (types_masked != TokenType.IGNORE)

    # Type flip rates.
    if gt_coord is not None and gt_coord.any().item():
        flip = (~pred_is_coord_masked[gt_coord]).float().mean()
        out["coord_monitor/flip_coord_to_noncoord"] = float(flip.detach().item())
    if gt_text is not None and gt_text.any().item():
        flip = (pred_is_coord_masked[gt_text]).float().mean()
        out["coord_monitor/flip_text_to_coord"] = float(flip.detach().item())
    if gt_format is not None and gt_format.any().item():
        flip = (pred_is_coord_masked[gt_format]).float().mean()
        out["coord_monitor/flip_format_to_coord"] = float(flip.detach().item())
    if gt_desc is not None and gt_desc.any().item():
        flip = (pred_is_coord_masked[gt_desc]).float().mean()
        out["coord_monitor/flip_desc_to_coord"] = float(flip.detach().item())

    if not coord_monitor_mass:
        return out

    # Coord-vocab probability mass at GT slots (mean over tokens).
    flat_logits_full = logits_next[supervised_mask]
    gt_coord_mass = gt_coord
    gt_text_mass = gt_text
    gt_format_mass = gt_format
    gt_desc_mass = gt_desc

    max_tokens = max(0, int(coord_monitor_mass_max_tokens))
    if max_tokens > 0 and flat_logits_full.shape[0] > max_tokens:
        stride = max(1, int(flat_logits_full.shape[0]) // max_tokens)
        sel = torch.arange(
            0,
            flat_logits_full.shape[0],
            stride,
            device=flat_logits_full.device,
            dtype=torch.long,
        )[:max_tokens]
        flat_logits_full = flat_logits_full.index_select(0, sel)
        if isinstance(gt_coord_mass, torch.Tensor):
            gt_coord_mass = gt_coord_mass.index_select(0, sel)
        if isinstance(gt_text_mass, torch.Tensor):
            gt_text_mass = gt_text_mass.index_select(0, sel)
        if isinstance(gt_format_mass, torch.Tensor):
            gt_format_mass = gt_format_mass.index_select(0, sel)
        if isinstance(gt_desc_mass, torch.Tensor):
            gt_desc_mass = gt_desc_mass.index_select(0, sel)

    idx = ids[valid]
    flat_logits_coord = flat_logits_full.index_select(-1, idx)

    temp = float(temperature) if float(temperature) > 0 else 1.0

    from src.trainers.losses.coord_soft_ce_w1 import coord_vocab_gate_loss

    gate_loss, _mass_mean = coord_vocab_gate_loss(
        logits_full=flat_logits_full,
        logits_coord=flat_logits_coord,
        temperature=float(temp),
    )
    mass = torch.exp((-gate_loss).clamp(min=-50.0, max=50.0))

    if isinstance(gt_coord_mass, torch.Tensor) and gt_coord_mass.any().item():
        out["coord_monitor/coord_vocab_mass_at_gt_coord"] = float(
            mass[gt_coord_mass].mean().detach().item()
        )
    if isinstance(gt_text_mass, torch.Tensor) and gt_text_mass.any().item():
        out["coord_monitor/coord_vocab_mass_at_gt_text"] = float(
            mass[gt_text_mass].mean().detach().item()
        )
    if isinstance(gt_format_mass, torch.Tensor) and gt_format_mass.any().item():
        out["coord_monitor/coord_vocab_mass_at_gt_format"] = float(
            mass[gt_format_mass].mean().detach().item()
        )
    if isinstance(gt_desc_mass, torch.Tensor) and gt_desc_mass.any().item():
        out["coord_monitor/coord_vocab_mass_at_gt_desc"] = float(
            mass[gt_desc_mass].mean().detach().item()
        )

    return out


def compute_coord_diag_metrics_for_pure_ce(
    trainer: Any,
    *,
    logits_next: torch.Tensor,
    labels_next: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute coord_diag/* metrics for pure-CE ablations (metrics-only).

    When coord_soft_ce_w1 is disabled, coord tokens are trained with pure CE.
    We still compute the distributional loss components as diagnostics to compare
    training dynamics.

    Expected skips: missing tokenizer/coord vocab, no coord positions, etc.
    """

    cfg = getattr(trainer, "coord_soft_ce_w1_cfg", None)
    if cfg is None or bool(getattr(cfg, "enabled", False)):
        return {}

    out: dict[str, float] = {"coord_diag/enabled": 0.0}

    vocab_size = int(logits_next.shape[-1])
    if vocab_size <= 0:
        return out

    # Cache coord-token ids so we don't call tokenizer.convert_tokens_to_ids() every step.
    coord_token_ids = getattr(trainer, "_coordexp_coord_token_ids", None)
    if not isinstance(coord_token_ids, list) or not coord_token_ids:
        coord_token_ids = []
        coord_ids_fn = getattr(trainer, "_get_coord_token_ids", None)
        if callable(coord_ids_fn):
            coord_token_ids = coord_ids_fn()
        else:
            tokenizer = getattr(getattr(trainer, "template", None), "tokenizer", None)
            if tokenizer is not None:
                coord_token_ids = get_coord_token_ids(tokenizer)
        try:
            setattr(trainer, "_coordexp_coord_token_ids", list(coord_token_ids))
        except Exception:
            pass

    if not coord_token_ids:
        return out

    # Cache small coord-vocab tensors per device/vocab_size.
    cache = getattr(trainer, "_coordexp_coord_vocab_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(trainer, "_coordexp_coord_vocab_cache", cache)
        except Exception:
            cache = {}

    cache_key = (str(logits_next.device), int(vocab_size))
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        # `coord_token_ids` are in *bin order* (0..999); build:
        # - ids_bin: ids in bin order (for slicing logits -> K bins)
        # - ids_sorted/bins_sorted: ids sorted by token id with bin mapping (for labels -> bin)
        valid_pairs: list[tuple[int, int]] = []
        for bin_idx, tok_id in enumerate(coord_token_ids):
            try:
                tok_id_i = int(tok_id)
            except Exception:
                continue
            if 0 <= tok_id_i < vocab_size:
                valid_pairs.append((bin_idx, tok_id_i))
        if len(valid_pairs) != len(coord_token_ids):
            entry = None
        else:
            valid_pairs.sort(key=lambda x: x[0])
            ids_bin_list = [tid for _, tid in valid_pairs]
            ids_bin = torch.tensor(
                ids_bin_list, device=logits_next.device, dtype=torch.long
            )

            sorted_pairs = sorted(
                [(tid, bin_idx) for bin_idx, tid in valid_pairs], key=lambda x: x[0]
            )
            ids_sorted = torch.tensor(
                [tid for tid, _ in sorted_pairs],
                device=logits_next.device,
                dtype=torch.long,
            )
            bins_sorted = torch.tensor(
                [bin_idx for _, bin_idx in sorted_pairs],
                device=logits_next.device,
                dtype=torch.long,
            )

            contig = False
            start = None
            k = int(ids_bin.numel())
            if k > 0:
                start_i = int(ids_bin_list[0])
                contig = all(int(ids_bin_list[i]) == start_i + i for i in range(k))
                if contig:
                    start = start_i

            entry = {
                "ids_bin": ids_bin,
                "ids_sorted": ids_sorted,
                "bins_sorted": bins_sorted,
                "contig": bool(contig),
                "start": start,
            }
            cache[cache_key] = entry

    if not isinstance(entry, dict):
        return out

    ids_sorted = entry["ids_sorted"]
    bins_sorted = entry["bins_sorted"]
    k = int(ids_sorted.numel())
    if k <= 0:
        return out

    labels_safe = labels_next
    if labels_safe.numel() > 0 and labels_safe.min().item() < 0:
        labels_safe = labels_safe.clamp(min=0)
    labels_clamped = labels_safe.clamp(min=0, max=vocab_size - 1)
    pos = torch.searchsorted(ids_sorted, labels_clamped)
    pos_safe = pos.clamp(max=k - 1)
    match = (pos < k) & (ids_sorted[pos_safe] == labels_clamped) & supervised_mask
    target_bins_all = torch.where(
        match, bins_sorted[pos_safe], pos_safe.new_full(pos_safe.shape, -1)
    )
    coord_positions_mask = target_bins_all >= 0

    if (
        not isinstance(coord_positions_mask, torch.Tensor)
        or not coord_positions_mask.any().item()
    ):
        return out

    with torch.no_grad():
        flat_logits_full = logits_next[coord_positions_mask]
        flat_target_bins = target_bins_all[coord_positions_mask].to(dtype=torch.long)

        ids_bin = entry["ids_bin"]
        start = entry.get("start", None)
        if entry.get("contig", False) and isinstance(start, int):
            flat_logits_coord = flat_logits_full[..., start : start + ids_bin.numel()]
        else:
            flat_logits_coord = flat_logits_full.index_select(-1, ids_bin)

        temperature = float(getattr(cfg, "temperature", 1.0))
        soft_ce_weight = float(getattr(cfg, "soft_ce_weight", 1.0))
        w1_weight = float(getattr(cfg, "w1_weight", 1.0))
        gate_weight = float(getattr(cfg, "gate_weight", 0.0))

        dist_out = coord_soft_ce_w1(
            flat_logits_coord,
            flat_target_bins,
            sigma=float(getattr(cfg, "target_sigma", 2.0)),
            truncate=getattr(cfg, "target_truncate", None),
            temperature=temperature,
            # Weighting is applied at the metric level so we can include the gate term.
            soft_ce_weight=1.0,
            w1_weight=1.0,
            normalize_w1=True,
        )

        # Distribution monitors (coord vocab only).
        coord_acc_top5 = None
        coord_p_gt_mean = None
        coord_margin_mean = None
        coord_expected_bin_mae = None
        coord_expected_bin_abs_err_p90 = None
        coord_w1_to_delta = None
        try:
            k5 = min(5, int(flat_logits_coord.shape[-1]))
            topk = flat_logits_coord.topk(k=k5, dim=-1).indices
            coord_acc_top5 = (
                (topk == flat_target_bins.unsqueeze(-1)).any(dim=-1).float().mean()
            )
            coord_p_gt_mean = dist_out.pred_probs.gather(
                1, flat_target_bins.view(-1, 1)
            ).mean()
            bins = torch.arange(
                int(dist_out.pred_probs.shape[-1]),
                device=dist_out.pred_probs.device,
                dtype=dist_out.pred_probs.dtype,
            )
            pred_expected = (dist_out.pred_probs * bins.view(1, -1)).sum(dim=-1)
            abs_err = (pred_expected.float() - flat_target_bins.float()).abs()
            coord_expected_bin_mae = abs_err.mean()
            # W1(p, delta_t) in 1D bins, equivalently E_p[|k - t|].
            probs = dist_out.pred_probs.to(dtype=torch.float32)
            bins_f = bins.to(dtype=torch.float32)
            dist_bins = (
                bins_f.view(1, -1)
                - flat_target_bins.to(torch.float32).view(-1, 1)
            ).abs()
            coord_w1_to_delta = (probs * dist_bins).sum(dim=-1).mean()
            if abs_err.numel() > 0:
                coord_expected_bin_abs_err_p90 = torch.quantile(
                    abs_err.to(dtype=torch.float32), 0.9
                )
            temp = float(temperature) if float(temperature) > 0 else 1.0
            logits_scaled = flat_logits_coord.float() / temp
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

        softce_sum = dist_out.soft_ce_per_token.sum()
        w1_sum = dist_out.w1_per_token.sum()
        gate_sum = softce_sum.new_tensor(0.0)
        gate_mass_mean = None
        if gate_weight != 0.0:
            temp = float(temperature) if float(temperature) > 0 else 1.0

            from src.trainers.losses.coord_soft_ce_w1 import coord_vocab_gate_loss

            gate_per_token, gate_mass_mean = coord_vocab_gate_loss(
                logits_full=flat_logits_full,
                logits_coord=flat_logits_coord,
                temperature=float(temp),
            )
            gate_sum = gate_per_token.sum()

        denom = coord_positions_mask.sum().to(dtype=torch.float32)
        if (
            getattr(getattr(trainer, "args", None), "average_tokens_across_devices", False)
            and getattr(trainer, "model_accepts_loss_kwargs", False)
            and dist.is_available()
            and dist.is_initialized()
        ):
            denom_global = denom.detach().clone()
            dist.all_reduce(denom_global, op=dist.ReduceOp.SUM)
            denom = denom_global
        denom = torch.where(denom > 0, denom, denom.new_tensor(1.0))

        softce_loss = softce_sum / denom
        w1_loss = w1_sum / denom
        gate_loss = gate_sum / denom

        softce_contrib = soft_ce_weight * softce_loss
        w1_contrib = w1_weight * w1_loss
        gate_contrib = gate_weight * gate_loss
        coord_loss = softce_contrib + w1_contrib + gate_contrib

        if (
            getattr(getattr(trainer, "args", None), "average_tokens_across_devices", False)
            and getattr(trainer, "model_accepts_loss_kwargs", False)
        ):
            try:
                if dist.is_available() and dist.is_initialized():
                    scale = float(dist.get_world_size())
                else:
                    scale = float(trainer.accelerator.num_processes)
            except Exception:
                scale = 1.0
            coord_loss = coord_loss * scale
            softce_contrib = softce_contrib * scale
            w1_contrib = w1_contrib * scale
            gate_contrib = gate_contrib * scale

        coord_loss = torch.nan_to_num(coord_loss, nan=0.0, posinf=1e4, neginf=0.0)
        softce_contrib = torch.nan_to_num(softce_contrib, nan=0.0, posinf=1e4, neginf=0.0)
        w1_contrib = torch.nan_to_num(w1_contrib, nan=0.0, posinf=1e4, neginf=0.0)
        gate_contrib = torch.nan_to_num(gate_contrib, nan=0.0, posinf=1e4, neginf=0.0)

        coord_tokens = int(coord_positions_mask.sum().detach().item())
        out["coord_diag/loss"] = float(coord_loss.detach().cpu().item())
        out["coord_diag/soft_ce"] = float(softce_contrib.detach().cpu().item())
        out["coord_diag/w1"] = float(w1_contrib.detach().cpu().item())
        out["coord_diag/gate"] = float(gate_contrib.detach().cpu().item())
        out["coord_diag/coord_tokens"] = float(coord_tokens)

        if gate_mass_mean is not None:
            out["coord_diag/coord_vocab_mass"] = float(gate_mass_mean.detach().cpu().item())
        if coord_acc_top5 is not None:
            out["coord_diag/acc_top5"] = float(coord_acc_top5.detach().cpu().item())
        if coord_p_gt_mean is not None:
            out["coord_diag/p_gt_mean"] = float(coord_p_gt_mean.detach().cpu().item())
        if coord_margin_mean is not None:
            out["coord_diag/margin_mean"] = float(coord_margin_mean.detach().cpu().item())
        if coord_expected_bin_mae is not None:
            out["coord_diag/expected_bin_mae"] = float(coord_expected_bin_mae.detach().cpu().item())
        if coord_expected_bin_abs_err_p90 is not None:
            out["coord_diag/expected_bin_abs_err_p90"] = float(coord_expected_bin_abs_err_p90.detach().cpu().item())
        if coord_w1_to_delta is not None:
            out["coord_diag/w1_to_delta"] = float(coord_w1_to_delta.detach().cpu().item())

        # Per-sample helpers (packed runs).
        try:
            total_samples = None
            pack_n = getattr(trainer, "_coordexp_pack_num_samples", None)
            if isinstance(pack_n, torch.Tensor):
                total_samples = float(pack_n.detach().sum().cpu().item())
            elif isinstance(pack_n, (list, tuple)):
                total_samples = float(sum(int(v) for v in pack_n))
            elif isinstance(pack_n, (int, float)):
                total_samples = float(pack_n)
            if total_samples is None:
                total_samples = float(labels_next.shape[0])
            total_samples = max(1.0, float(total_samples))

            out["coord_diag/coord_tokens_per_sample"] = float(coord_tokens) / float(total_samples)
            coord_loss_per_sample = float(coord_loss.detach().cpu().item()) * float(coord_tokens) / float(total_samples)
            out["coord_diag/loss_per_sample"] = float(coord_loss_per_sample)
        except Exception:
            pass

    return out
