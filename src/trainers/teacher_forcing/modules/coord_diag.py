from __future__ import annotations

from typing import Any, Mapping

import torch

from src.trainers.losses.coord_soft_ce_w1 import coord_vocab_gate_loss

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext


def run_coord_diag_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
    state: Mapping[str, Any],
) -> ModuleResult:
    coord_logits = state.get("coord_logits")
    coord_logits_full = state.get("coord_logits_full")
    target_bins = state.get("coord_target_bins")

    if not isinstance(coord_logits, torch.Tensor) or int(coord_logits.numel()) == 0:
        z = context.logits.new_tensor(0.0)
        return ModuleResult(loss=z, metrics={}, state={})

    if not isinstance(coord_logits_full, torch.Tensor) or not isinstance(target_bins, torch.Tensor):
        raise ValueError("coord_diag requires bbox_geo state with coord logits + targets")

    temp_safe = max(float(context.temperature), 1e-6)
    logits_diag = coord_logits.float()
    probs = torch.softmax(logits_diag / temp_safe, dim=-1)

    topk_k = max(1, min(5, int(probs.shape[-1])))
    topk = probs.topk(k=topk_k, dim=-1).indices
    acc_top5 = float((topk == target_bins.unsqueeze(-1)).any(dim=-1).float().mean().item())
    p_gt_mean = float(probs.gather(1, target_bins.view(-1, 1)).mean().item())

    bins_diag = torch.arange(int(probs.shape[-1]), device=probs.device, dtype=probs.dtype)
    pred_expected = (probs * bins_diag.view(1, -1)).sum(dim=-1)
    abs_err = (pred_expected.float() - target_bins.float()).abs()

    logits_scaled = logits_diag / temp_safe
    gt_logit = logits_scaled.gather(1, target_bins.view(-1, 1)).squeeze(1)
    max_logit = logits_scaled.max(dim=-1).values
    margin_mean = float((max_logit - gt_logit).mean().item())

    try:
        _gate_diag, mass_mean = coord_vocab_gate_loss(
            logits_full=coord_logits_full,
            logits_coord=coord_logits,
            temperature=float(temp_safe),
        )
        coord_vocab_mass_mean = float(mass_mean.detach().cpu().item())
    except Exception:
        coord_vocab_mass_mean = 0.0

    metrics = {
        "coord_diag/coord_tokens_total": float(int(coord_logits.shape[0])),
        "coord_diag/acc_top5": float(acc_top5),
        "coord_diag/p_gt_mean": float(p_gt_mean),
        "coord_diag/margin_mean": float(margin_mean),
        "coord_diag/expected_bin_mae": float(abs_err.mean().item()),
        "coord_diag/expected_bin_abs_err_p90": float(torch.quantile(abs_err, 0.9).item()),
        "coord_diag/coord_vocab_mass_mean": float(coord_vocab_mass_mean),
    }

    z = context.logits.new_tensor(0.0)
    return ModuleResult(loss=z, metrics=metrics, state={})
