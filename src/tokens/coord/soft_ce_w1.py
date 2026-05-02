"""Stage-1 coord-token supervision: softCE(Gaussian) + Wasserstein-1(CDF).

This module is used by Scheme A (Stage-1 retrain) to supervise `<|coord_k|>` slots
purely at the token level:
  - Build a unimodal soft target distribution over 1000 ordered bins (0..999).
  - Compute soft cross-entropy (softCE) against the model's coord-gated logits.
  - Compute 1D Wasserstein-1 via CDF differences (vectorized; fp16/bf16-safe).

All utilities are designed to be numerically robust: logits are sanitized before
softmax/log-softmax, and outputs are NaN-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

DEFAULT_COORD_BINS = 1000
LOGIT_CLAMP_ABS = 1e4


@dataclass(frozen=True)
class CoordSoftCEW1Output:
    """Per-token losses and aggregates for coord supervision."""

    loss_per_token: torch.Tensor  # [N]
    soft_ce_per_token: torch.Tensor  # [N]
    w1_per_token: torch.Tensor  # [N]
    target_probs: torch.Tensor  # [N, K]
    pred_probs: torch.Tensor  # [N, K]

    def mean(self) -> Dict[str, torch.Tensor]:
        if self.loss_per_token.numel() == 0:
            zero = self.loss_per_token.new_tensor(0.0)
            return {"loss": zero, "soft_ce": zero, "w1": zero}
        return {
            "loss": self.loss_per_token.mean(),
            "soft_ce": self.soft_ce_per_token.mean(),
            "w1": self.w1_per_token.mean(),
        }


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    """NaN/inf-safe logits sanitization (keeps gradients where possible).

    Always applies `nan_to_num` and clamp (avoids sync-inducing `.item()` checks).
    """

    logits = torch.nan_to_num(
        logits, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=-LOGIT_CLAMP_ABS
    )
    return logits.clamp(min=-LOGIT_CLAMP_ABS, max=LOGIT_CLAMP_ABS)


def gaussian_soft_targets(
    target_bins: torch.Tensor,
    *,
    num_bins: int = DEFAULT_COORD_BINS,
    sigma: float,
    truncate: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build unimodal Gaussian-kernel soft targets over ordered bins.

    Args:
        target_bins: [N] integer bin indices in [0, num_bins-1]
        num_bins: number of ordered bins (default 1000)
        sigma: Gaussian standard deviation in *bin units* (>0)
        truncate: optional radius (in bins) to zero out weights when |i - t| > truncate
        dtype: output dtype (default float32)

    Returns:
        probs: [N, num_bins] probability distribution per target bin, sums to 1.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if truncate is not None and truncate < 0:
        raise ValueError("truncate must be >= 0 or None")

    if target_bins.ndim != 1:
        raise ValueError("target_bins must be a 1D tensor of shape [N]")
    if target_bins.numel() == 0:
        return target_bins.new_zeros((0, int(num_bins)), dtype=dtype)

    device = target_bins.device
    t = target_bins.to(dtype=torch.long)
    t = t.clamp(min=0, max=int(num_bins) - 1)

    centers = torch.arange(int(num_bins), device=device, dtype=torch.float32).view(1, -1)
    diff = centers - t.to(dtype=torch.float32).unsqueeze(-1)
    if truncate is not None:
        diff = torch.where(diff.abs() <= float(truncate), diff, torch.full_like(diff, float("inf")))

    # exp(-0.5 * (diff/sigma)^2)
    inv_sigma = 1.0 / float(sigma)
    logits = -0.5 * (diff * inv_sigma) ** 2
    # exp(-inf)=0 for truncated entries
    weights = torch.exp(logits)

    # Normalize to sum=1 (safe for fp16/bf16).
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    denom = weights.sum(dim=-1, keepdim=True)
    # denom should never be 0 because diff==0 => exp(0)=1 at the center, but keep a guard.
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    probs = weights / denom
    probs = probs.to(dtype=dtype)
    return probs


def soft_cross_entropy(
    logits: torch.Tensor, target_probs: torch.Tensor, *, temperature: float = 1.0
) -> torch.Tensor:
    """Soft cross-entropy between predicted logits and a soft target distribution.

    Args:
        logits: [N, K] unnormalized logits
        target_probs: [N, K] probabilities (sum to 1)
        temperature: softmax temperature (>0); applied to logits before log-softmax

    Returns:
        loss: [N] per-token softCE
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if logits.shape != target_probs.shape:
        raise ValueError("logits and target_probs must have the same shape")

    logits_safe = _sanitize_logits(logits).float() / float(temperature)
    log_probs = F.log_softmax(logits_safe, dim=-1)

    q = torch.nan_to_num(target_probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    q_sum = q.sum(dim=-1, keepdim=True)
    q_sum = torch.where(q_sum > 0, q_sum, torch.ones_like(q_sum))
    q = q / q_sum

    loss = -(q * log_probs).sum(dim=-1)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=0.0)
    return loss


def wasserstein_1_cdf(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """1D Wasserstein-1 distance using CDF differences.

    On an ordered discrete line with unit spacing:
      W1(p, q) = sum_i |CDF_p(i) - CDF_q(i)|

    Args:
        pred_probs: [N, K] predicted distribution (sum to 1)
        target_probs: [N, K] target distribution (sum to 1)
        normalize: if True, divide by (K-1) so distances are in [0,1] for one-hot shifts

    Returns:
        w1: [N] per-token Wasserstein-1 distance
    """

    if pred_probs.shape != target_probs.shape:
        raise ValueError("pred_probs and target_probs must have the same shape")
    if pred_probs.numel() == 0:
        return pred_probs.new_zeros((0,), dtype=torch.float32)

    p = torch.nan_to_num(pred_probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    q = torch.nan_to_num(target_probs.float(), nan=0.0, posinf=0.0, neginf=0.0)

    p_sum = p.sum(dim=-1, keepdim=True)
    q_sum = q.sum(dim=-1, keepdim=True)
    p_sum = torch.where(p_sum > 0, p_sum, torch.ones_like(p_sum))
    q_sum = torch.where(q_sum > 0, q_sum, torch.ones_like(q_sum))
    p = p / p_sum
    q = q / q_sum

    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)

    w1 = (cdf_p - cdf_q).abs().sum(dim=-1)
    if normalize:
        # Normalize from "bin units" to [0,1] by dividing by the max possible shift.
        # For K ordered bins (0..K-1), the maximum distance is (K-1).
        k = int(pred_probs.shape[-1])
        if k > 1:
            w1 = w1 / float(k - 1)
    w1 = torch.nan_to_num(w1, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=0.0)
    return w1


def coord_soft_ce_w1(
    coord_logits: torch.Tensor,
    target_bins: torch.Tensor,
    *,
    sigma: float,
    truncate: int | None,
    temperature: float,
    soft_ce_weight: float,
    w1_weight: float,
    normalize_w1: bool = True,
) -> CoordSoftCEW1Output:
    """Compute coord supervision losses for coord-gated logits.

    Args:
        coord_logits: [N, K] logits restricted to ordered coord bins (K=1000)
        target_bins: [N] integer bin targets in [0, K-1]
        sigma/ truncate: Gaussian soft target parameters
        temperature: softmax temperature (>0)
        soft_ce_weight/w1_weight: weights for the combined per-token loss
        normalize_w1: normalize W1 by K (default True)

    Returns:
        CoordSoftCEW1Output with per-token losses and the constructed distributions.
    """

    if coord_logits.ndim != 2:
        raise ValueError("coord_logits must have shape [N, K]")
    if target_bins.ndim != 1:
        raise ValueError("target_bins must have shape [N]")
    if coord_logits.shape[0] != target_bins.shape[0]:
        raise ValueError("coord_logits and target_bins must align on N")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    n, k = coord_logits.shape
    if n == 0:
        empty_k = int(k)
        empty_probs = coord_logits.new_zeros((0, empty_k), dtype=torch.float32)
        empty = coord_logits.new_zeros((0,), dtype=torch.float32)
        return CoordSoftCEW1Output(
            loss_per_token=empty,
            soft_ce_per_token=empty,
            w1_per_token=empty,
            target_probs=empty_probs,
            pred_probs=empty_probs,
        )

    target_probs = gaussian_soft_targets(
        target_bins,
        num_bins=int(k),
        sigma=float(sigma),
        truncate=truncate,
        dtype=torch.float32,
    )

    logits_safe = _sanitize_logits(coord_logits).float() / float(temperature)
    pred_probs = torch.softmax(logits_safe, dim=-1)
    pred_probs = torch.nan_to_num(pred_probs, nan=0.0, posinf=0.0, neginf=0.0)
    p_sum = pred_probs.sum(dim=-1, keepdim=True)
    p_sum = torch.where(p_sum > 0, p_sum, torch.ones_like(p_sum))
    pred_probs = pred_probs / p_sum

    soft_ce = soft_cross_entropy(coord_logits, target_probs, temperature=temperature)
    w1 = wasserstein_1_cdf(pred_probs, target_probs, normalize=normalize_w1)

    try:
        w_soft = float(soft_ce_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("soft_ce_weight must be numeric") from exc
    try:
        w_w1 = float(w1_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("w1_weight must be numeric") from exc

    loss = w_soft * soft_ce + w_w1 * w1
    loss = torch.nan_to_num(loss, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=0.0)

    return CoordSoftCEW1Output(
        loss_per_token=loss,
        soft_ce_per_token=soft_ce,
        w1_per_token=w1,
        target_probs=target_probs,
        pred_probs=pred_probs,
    )


__all__ = [
    "DEFAULT_COORD_BINS",
    "CoordSoftCEW1Output",
    "gaussian_soft_targets",
    "soft_cross_entropy",
    "wasserstein_1_cdf",
    "coord_soft_ce_w1",
]
