"""Coordinate-token Gaussian soft CE plus discrete CRPS/RPS loss."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from src.common.errors import LossContractError
from src.losses.context import LossContext


LOGIT_CLAMP_ABS = 1e4
R95_TO_SIGMA = 1.96
MIN_TEMPERATURE = 1.0e-6


def gaussian_soft_targets_from_r95(
    target_bins: torch.Tensor,
    r95_radii: torch.Tensor,
    *,
    num_bins: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if int(num_bins) <= 0:
        raise ValueError("num_bins must be positive")
    if target_bins.ndim != 1:
        raise ValueError("target_bins must be a 1D tensor")
    if r95_radii.ndim != 1:
        raise ValueError("r95_radii must be a 1D tensor")
    if int(target_bins.numel()) != int(r95_radii.numel()):
        raise ValueError("target_bins and r95_radii must have the same length")
    if int(target_bins.numel()) == 0:
        return target_bins.new_zeros((0, int(num_bins)), dtype=dtype)

    device = target_bins.device
    targets = target_bins.to(device=device, dtype=torch.long).clamp(
        min=0,
        max=int(num_bins) - 1,
    )
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
        weights[positive] = torch.exp(-0.5 * diff[positive].pow(2) / sigma.pow(2))
    if bool((~positive).any().item()):
        weights[~positive] = F.one_hot(
            targets[~positive],
            num_classes=int(num_bins),
        ).to(dtype=torch.float32)
    return _normalize_probs(weights).to(dtype=dtype)


def ranked_probability_score(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    if pred_probs.shape != target_probs.shape:
        raise ValueError("pred_probs and target_probs must have the same shape")
    if pred_probs.numel() == 0:
        return pred_probs.new_zeros((0,), dtype=torch.float32)
    pred = _normalize_probs(pred_probs)
    target = _normalize_probs(target_probs)
    rps = (pred.cumsum(dim=-1) - target.cumsum(dim=-1)).pow(2).sum(dim=-1)
    if normalize and int(pred_probs.shape[-1]) > 1:
        rps = rps / float(int(pred_probs.shape[-1]) - 1)
    return torch.nan_to_num(rps, nan=0.0, posinf=LOGIT_CLAMP_ABS, neginf=0.0)


@dataclass(frozen=True)
class CoordGaussianRPSLoss:
    gaussian_weight: float
    rps_weight: float
    temperature: float
    gaussian_r95_axis_fraction: float
    gaussian_r95_cap_bins: int
    gaussian_r95_min_bins: int
    gaussian_r95_fallback_bins: int
    name: str = "coord_gaussian_rps"
    last_diagnostics: dict[str, Any] | None = field(
        default=None,
        init=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("gaussian_weight", self.gaussian_weight),
            ("rps_weight", self.rps_weight),
        ):
            if float(value) < 0.0 or not math.isfinite(float(value)):
                raise LossContractError(
                    "coord_gaussian_rps weights must be finite and non-negative",
                    code="loss.coord_gaussian_rps_weight",
                    context={"field": name, "value": value},
                )
        if (
            float(self.temperature) < MIN_TEMPERATURE
            or not math.isfinite(float(self.temperature))
        ):
            raise LossContractError(
                "coord_gaussian_rps temperature must be finite and >= 1e-6",
                code="loss.coord_gaussian_rps_temperature",
                context={
                    "temperature": self.temperature,
                    "minimum": MIN_TEMPERATURE,
                },
            )
        if (
            float(self.gaussian_r95_axis_fraction) <= 0.0
            or not math.isfinite(float(self.gaussian_r95_axis_fraction))
        ):
            raise LossContractError(
                "coord_gaussian_rps gaussian_r95_axis_fraction must be finite and > 0",
                code="loss.coord_gaussian_rps_r95_fraction",
                context={"gaussian_r95_axis_fraction": self.gaussian_r95_axis_fraction},
            )

    def per_atom_loss(self, context: LossContext) -> torch.Tensor:
        logits_fp32, target_ids, atoms = context.select_logits_fp32()
        if not atoms:
            raise LossContractError(
                "CoordGaussianRPSLoss requires at least one coordinate atom",
                code="loss.coord_gaussian_rps_empty",
                context={"term": self.name},
            )
        coordinate_ids = tuple(int(item) for item in context.vocab_groups.coordinate)
        if not coordinate_ids:
            raise LossContractError(
                "coord_gaussian_rps requires a non-empty coordinate vocabulary",
                code="loss.coord_gaussian_rps_empty_vocab",
                context={"term": self.name},
            )
        coord_index_by_token_id = {
            token_id: index for index, token_id in enumerate(coordinate_ids)
        }
        missing_targets = [
            atom.to_artifact_dict()
            for atom in atoms
            if atom.coordinate_target is None
        ]
        if missing_targets:
            raise LossContractError(
                "coordinate atoms require CoordinateLossTarget metadata",
                code="loss.coord_gaussian_rps_target_missing",
                context={
                    "term": self.name,
                    "missing_count": len(missing_targets),
                    "first_missing": missing_targets[0],
                },
            )
        target_bins = torch.tensor(
            [coord_index_by_token_id[int(target_id)] for target_id in target_ids.tolist()],
            dtype=torch.long,
            device=logits_fp32.device,
        )
        r95_radii = torch.tensor(
            [self._r95_radius(atom.coordinate_target.axis_length) for atom in atoms],
            dtype=torch.float32,
            device=logits_fp32.device,
        )
        coord_ids_tensor = torch.tensor(
            coordinate_ids,
            dtype=torch.long,
            device=logits_fp32.device,
        )
        coord_logits = logits_fp32.index_select(1, coord_ids_tensor)
        target_probs = gaussian_soft_targets_from_r95(
            target_bins,
            r95_radii,
            num_bins=len(coordinate_ids),
            dtype=torch.float32,
        )
        logits_safe = _sanitize_logits(coord_logits).float() / float(self.temperature)
        pred_probs = _normalize_probs(torch.softmax(logits_safe, dim=-1))
        log_probs = F.log_softmax(logits_safe, dim=-1)
        gaussian_ce = -(target_probs * log_probs).sum(dim=-1)
        rps = ranked_probability_score(pred_probs, target_probs)
        per_atom = float(self.gaussian_weight) * gaussian_ce + float(self.rps_weight) * rps
        if not (
            torch.isfinite(gaussian_ce).all()
            and torch.isfinite(rps).all()
            and torch.isfinite(per_atom).all()
        ):
            raise LossContractError(
                "coord_gaussian_rps produced non-finite per-atom values",
                code="loss.coord_gaussian_rps_non_finite",
                context={"term": self.name},
            )
        object.__setattr__(
            self,
            "last_diagnostics",
            _diagnostics(
                target_probs=target_probs,
                r95_radii=r95_radii,
                gaussian_ce=gaussian_ce,
                rps=rps,
                selected_count=len(atoms),
            ),
        )
        return per_atom

    def _r95_radius(self, axis_length: int) -> int:
        cap = _bounded_int(self.gaussian_r95_cap_bins, default=8, minimum=0, maximum=999)
        minimum = _bounded_int(self.gaussian_r95_min_bins, default=1, minimum=0, maximum=cap)
        fallback = _bounded_int(
            self.gaussian_r95_fallback_bins,
            default=cap,
            minimum=minimum,
            maximum=cap,
        )
        if axis_length <= 0:
            return fallback
        raw = int(math.floor(min(float(cap), float(self.gaussian_r95_axis_fraction) * float(axis_length))))
        return max(int(minimum), raw)


def _diagnostics(
    *,
    target_probs: torch.Tensor,
    r95_radii: torch.Tensor,
    gaussian_ce: torch.Tensor,
    rps: torch.Tensor,
    selected_count: int,
) -> dict[str, Any]:
    target_safe = target_probs.clamp_min(1e-30)
    entropy = -(target_probs * target_safe.log()).sum(dim=-1)
    return {
        "selected_count": int(selected_count),
        "target_entropy_mean": _float_mean(entropy),
        "target_peak_prob_mean": _float_mean(target_probs.max(dim=-1).values),
        "target_r95_radius_mean": _float_mean(r95_radii),
        "target_r95_radius_max": float(r95_radii.detach().max().cpu())
        if int(r95_radii.numel()) > 0
        else 0.0,
        "gaussian_ce_mean": _float_mean(gaussian_ce),
        "rps_mean": _float_mean(rps),
    }


def _bounded_int(value: int | float, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), min(int(maximum), parsed))


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        logits,
        nan=0.0,
        posinf=LOGIT_CLAMP_ABS,
        neginf=-LOGIT_CLAMP_ABS,
    ).clamp(min=-LOGIT_CLAMP_ABS, max=LOGIT_CLAMP_ABS)


def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    out = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = out.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return out / denom


def _float_mean(values: torch.Tensor) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.detach().float().mean().cpu())


__all__ = [
    "CoordGaussianRPSLoss",
    "gaussian_soft_targets_from_r95",
    "ranked_probability_score",
]
