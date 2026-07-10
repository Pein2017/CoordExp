"""Stage-1 coordinate supervision with Gaussian targets and discrete RPS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

DEFAULT_COORD_BINS = 1000
LOGIT_CLAMP_ABS = 1e4
R95_TO_SIGMA = 1.96


@dataclass(frozen=True)
class CoordGaussianRPSOutput:
    loss_per_token: torch.Tensor
    gaussian_ce_per_token: torch.Tensor
    rps_per_token: torch.Tensor
    target_probs: torch.Tensor
    pred_probs: torch.Tensor

    def mean(self) -> Dict[str, torch.Tensor]:
        if self.loss_per_token.numel() == 0:
            zero = self.loss_per_token.new_tensor(0.0)
            return {"loss": zero, "gaussian_ce": zero, "rps": zero}
        return {
            "loss": self.loss_per_token.mean(),
            "gaussian_ce": self.gaussian_ce_per_token.mean(),
            "rps": self.rps_per_token.mean(),
        }


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = torch.nan_to_num(
        logits,
        nan=0.0,
        posinf=LOGIT_CLAMP_ABS,
        neginf=-LOGIT_CLAMP_ABS,
    )
    return logits.clamp(min=-LOGIT_CLAMP_ABS, max=LOGIT_CLAMP_ABS)


def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    out = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = out.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return out / denom


def gaussian_soft_targets_from_r95(
    target_bins: torch.Tensor,
    r95_radii: torch.Tensor,
    *,
    num_bins: int = DEFAULT_COORD_BINS,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build GT-centered Gaussian coordinate targets from per-token R95 radii.

    `r95_radii` is measured in coordinate bins. A radius of zero degenerates to a
    one-hot target at the GT bin. Positive radii are converted to sigma by
    `sigma = r95 / 1.96`.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if target_bins.ndim != 1:
        raise ValueError("target_bins must be a 1D tensor")
    if r95_radii.ndim != 1:
        raise ValueError("r95_radii must be a 1D tensor")
    if int(target_bins.numel()) != int(r95_radii.numel()):
        raise ValueError("target_bins and r95_radii must have the same length")
    if target_bins.numel() == 0:
        return target_bins.new_zeros((0, int(num_bins)), dtype=dtype)

    device = target_bins.device
    targets = target_bins.to(dtype=torch.long).clamp(min=0, max=int(num_bins) - 1)
    radii = torch.nan_to_num(
        r95_radii.to(device=device, dtype=torch.float32),
        nan=0.0,
        posinf=float(num_bins),
        neginf=0.0,
    ).clamp(min=0.0)

    centers = torch.arange(int(num_bins), device=device, dtype=torch.float32).view(1, -1)
    diff = centers - targets.to(dtype=torch.float32).view(-1, 1)
    weights = torch.zeros(
        (int(target_bins.numel()), int(num_bins)),
        device=device,
        dtype=torch.float32,
    )

    positive = radii > 0
    if bool(positive.any().item()):
        sigma = (radii[positive] / R95_TO_SIGMA).view(-1, 1).clamp_min(1e-6)
        pos_weights = torch.exp(-0.5 * diff[positive].pow(2) / sigma.pow(2))
        pos_weights = _normalize_probs(pos_weights)
        weights[positive] = pos_weights

    if bool((~positive).any().item()):
        weights[~positive] = F.one_hot(
            targets[~positive],
            num_classes=int(num_bins),
        ).to(dtype=torch.float32)

    return _normalize_probs(weights).to(dtype=dtype)


def soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if logits.shape != target_probs.shape:
        raise ValueError("logits and target_probs must have the same shape")

    logits_safe = _sanitize_logits(logits).float() / float(temperature)
    log_probs = F.log_softmax(logits_safe, dim=-1)
    targets = _normalize_probs(target_probs)
    loss = -(targets * log_probs).sum(dim=-1)
    return torch.nan_to_num(loss, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=0.0)


def ranked_probability_score(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Discrete CRPS/RPS over the ordered coordinate-bin line."""

    if pred_probs.shape != target_probs.shape:
        raise ValueError("pred_probs and target_probs must have the same shape")
    if pred_probs.numel() == 0:
        return pred_probs.new_zeros((0,), dtype=torch.float32)

    pred = _normalize_probs(pred_probs)
    target = _normalize_probs(target_probs)
    cdf_pred = pred.cumsum(dim=-1)
    cdf_target = target.cumsum(dim=-1)
    rps = (cdf_pred - cdf_target).pow(2).sum(dim=-1)
    if normalize:
        bins = int(pred_probs.shape[-1])
        if bins > 1:
            rps = rps / float(bins - 1)
    return torch.nan_to_num(rps, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=0.0)


def coord_gaussian_rps(
    coord_logits: torch.Tensor,
    target_bins: torch.Tensor,
    r95_radii: torch.Tensor,
    *,
    temperature: float,
    gaussian_weight: float,
    rps_weight: float,
    normalize_rps: bool = True,
) -> CoordGaussianRPSOutput:
    if coord_logits.ndim != 2:
        raise ValueError("coord_logits must have shape [N, K]")
    if target_bins.ndim != 1:
        raise ValueError("target_bins must have shape [N]")
    if r95_radii.ndim != 1:
        raise ValueError("r95_radii must have shape [N]")
    if int(coord_logits.shape[0]) != int(target_bins.shape[0]):
        raise ValueError("coord_logits and target_bins must align on N")
    if int(target_bins.shape[0]) != int(r95_radii.shape[0]):
        raise ValueError("target_bins and r95_radii must align on N")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    n, bins = coord_logits.shape
    if n == 0:
        empty_probs = coord_logits.new_zeros((0, int(bins)), dtype=torch.float32)
        empty = coord_logits.new_zeros((0,), dtype=torch.float32)
        return CoordGaussianRPSOutput(
            loss_per_token=empty,
            gaussian_ce_per_token=empty,
            rps_per_token=empty,
            target_probs=empty_probs,
            pred_probs=empty_probs,
        )

    target_probs = gaussian_soft_targets_from_r95(
        target_bins,
        r95_radii,
        num_bins=int(bins),
        dtype=torch.float32,
    )
    logits_safe = _sanitize_logits(coord_logits).float() / float(temperature)
    pred_probs = _normalize_probs(torch.softmax(logits_safe, dim=-1))
    gaussian_ce = soft_cross_entropy(
        coord_logits,
        target_probs,
        temperature=float(temperature),
    )
    rps = ranked_probability_score(pred_probs, target_probs, normalize=normalize_rps)

    loss = float(gaussian_weight) * gaussian_ce + float(rps_weight) * rps
    loss = torch.nan_to_num(loss, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=0.0)
    return CoordGaussianRPSOutput(
        loss_per_token=loss,
        gaussian_ce_per_token=gaussian_ce,
        rps_per_token=rps,
        target_probs=target_probs,
        pred_probs=pred_probs,
    )


__all__ = [
    "DEFAULT_COORD_BINS",
    "CoordGaussianRPSOutput",
    "coord_gaussian_rps",
    "gaussian_soft_targets_from_r95",
    "ranked_probability_score",
    "soft_cross_entropy",
]
