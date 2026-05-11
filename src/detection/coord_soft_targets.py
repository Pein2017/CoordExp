from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

CoordSlotName = Literal["x1", "y1", "x2", "y2"]
_COORD_SLOT_NAMES: tuple[str, ...] = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class CoordSoftTargetCandidate:
    object_instance_id: str
    slot_name: CoordSlotName
    bbox_xyxy: tuple[int, int, int, int]
    probability: float

    def __post_init__(self) -> None:
        if self.slot_name not in _COORD_SLOT_NAMES:
            raise ValueError(f"unsupported coordinate slot {self.slot_name!r}")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four coord-token bins")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in self.bbox_xyxy
        ):
            raise TypeError("bbox_xyxy must contain integer coord-token bins")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
            raise ValueError(
                f"bbox_xyxy must be valid token-space xyxy; got {self.bbox_xyxy}"
            )
        if not math.isfinite(float(self.probability)) or float(self.probability) <= 0.0:
            raise ValueError("coord soft target probability must be finite and > 0")


@dataclass(frozen=True)
class CoordSoftTargetRuntimeConfig:
    target_distribution: Literal["iou_gibbs_v0"]
    tau: float
    coord_token_start: int
    coord_token_end: int
    weighting: Literal["preserve_recursive_support_balance"] = (
        "preserve_recursive_support_balance"
    )
    apply_to_multi_positive: Literal["support_mixture"] = "support_mixture"

    def __post_init__(self) -> None:
        if self.target_distribution != "iou_gibbs_v0":
            raise ValueError("coord soft target_distribution must be iou_gibbs_v0")
        if not math.isfinite(float(self.tau)) or float(self.tau) <= 0.0:
            raise ValueError("coord soft target tau must be finite and > 0")
        if not isinstance(self.coord_token_start, int) or not isinstance(
            self.coord_token_end, int
        ):
            raise TypeError("coord token bounds must be integers")
        if int(self.coord_token_end) < int(self.coord_token_start):
            raise ValueError("coord_token_end must be >= coord_token_start")
        if self.weighting != "preserve_recursive_support_balance":
            raise ValueError(
                "coord soft target weighting must be preserve_recursive_support_balance"
            )
        if self.apply_to_multi_positive != "support_mixture":
            raise ValueError(
                "coord soft target apply_to_multi_positive must be support_mixture"
            )


@dataclass(frozen=True)
class CoordSoftTargetDistribution:
    token_ids: torch.Tensor
    probs: torch.Tensor
    support_mask: torch.Tensor
    entropy: torch.Tensor
    peak_prob: torch.Tensor
    perplexity: torch.Tensor
    effective_support_size: torch.Tensor
    std: torch.Tensor
    valid_candidate_count: torch.Tensor


@dataclass(frozen=True)
class CoordSoftCELoss:
    weighted_loss: torch.Tensor
    support_loss: torch.Tensor
    support_mass: torch.Tensor
    outside_support_mass: torch.Tensor
    balance_loss: torch.Tensor
    pure_soft_ce_equiv: torch.Tensor
    target_entropy: torch.Tensor
    kl_like: torch.Tensor
    peak_prob: torch.Tensor
    perplexity: torch.Tensor
    effective_support_size: torch.Tensor
    target_std: torch.Tensor
    valid_candidate_count: torch.Tensor
    support_mixture: bool


def build_iou_gibbs_coord_target(
    candidates: Sequence[CoordSoftTargetCandidate],
    cfg: CoordSoftTargetRuntimeConfig,
    *,
    device: torch.device | str | None = None,
) -> CoordSoftTargetDistribution:
    coord_bins = int(cfg.coord_token_end) - int(cfg.coord_token_start) + 1
    if coord_bins != 1000:
        raise ValueError("IoU-Gibbs coord soft targets require exactly 1000 coord bins")
    if not candidates:
        raise ValueError("coord soft target candidates must be non-empty")
    slot_name = candidates[0].slot_name
    if any(candidate.slot_name != slot_name for candidate in candidates):
        raise ValueError(
            "coord soft target candidates must share the same coordinate slot"
        )

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    bins = torch.arange(1000, dtype=torch.float64)
    mixture = torch.zeros((1000,), dtype=torch.float64)
    support_mask = torch.zeros((1000,), dtype=torch.bool)
    weight_total = 0.0

    for candidate in candidates:
        valid_mask = _valid_replaced_slot_mask(candidate.bbox_xyxy, slot_name, bins)
        support_mask |= valid_mask
        log_weights = _candidate_iou_gibbs_log_weights(
            candidate.bbox_xyxy,
            slot_name,
            bins,
            valid_mask,
            tau=float(cfg.tau),
        )
        weight = float(candidate.probability)
        mixture += weight * torch.exp(log_weights)
        weight_total += weight

    if weight_total <= 0.0:
        raise ValueError("coord soft target candidate probability mass must be > 0")
    probs = mixture / weight_total
    total_prob = probs.sum()
    if not torch.isfinite(total_prob) or float(total_prob.item()) <= 0.0:
        raise ValueError("coord soft target probabilities are not normalizable")
    probs = probs / total_prob

    if not torch.isfinite(probs).all():
        raise ValueError("coord soft target probabilities contain non-finite values")
    if torch.any(probs[~support_mask] != 0):
        raise ValueError("coord soft target assigned mass outside geometry support")

    positive = probs > 0
    entropy = -(probs[positive] * probs[positive].log()).sum()
    perplexity = entropy.exp()
    effective_support_size = 1.0 / torch.square(probs).sum()
    mean = (probs * bins).sum()
    variance = (probs * torch.square(bins - mean)).sum()
    std = variance.clamp_min(0.0).sqrt()
    token_ids = torch.arange(
        int(cfg.coord_token_start),
        int(cfg.coord_token_end) + 1,
        dtype=torch.long,
    )

    return CoordSoftTargetDistribution(
        token_ids=token_ids.to(device=target_device),
        probs=probs.to(device=target_device),
        support_mask=support_mask.to(device=target_device),
        entropy=entropy.to(device=target_device),
        peak_prob=probs.max().to(device=target_device),
        perplexity=perplexity.to(device=target_device),
        effective_support_size=effective_support_size.to(device=target_device),
        std=std.to(device=target_device),
        valid_candidate_count=support_mask.sum()
        .to(device=target_device, dtype=torch.float32),
    )


def full_vocab_coord_support_balance_ce(
    logits: torch.Tensor,
    candidates: Sequence[CoordSoftTargetCandidate],
    cfg: CoordSoftTargetRuntimeConfig,
    *,
    support_weight: float,
    balance_weight: float,
) -> CoordSoftCELoss:
    if logits.ndim != 1:
        raise ValueError("coord softCE logits must be a 1D full-vocab tensor")
    if not torch.isfinite(logits.float()).all():
        raise ValueError("coord softCE received non-finite logits")
    if not math.isfinite(float(support_weight)) or not math.isfinite(
        float(balance_weight)
    ):
        raise ValueError("coord softCE weights must be finite")

    dist = build_iou_gibbs_coord_target(candidates, cfg, device=logits.device)
    if int(dist.token_ids.max().item()) >= int(logits.shape[-1]):
        raise ValueError("coord token id exceeds logits vocab size")

    log_probs = F.log_softmax(logits.float(), dim=-1)
    coord_log_probs = log_probs.index_select(0, dist.token_ids)
    support_mask = dist.support_mask
    if not torch.any(support_mask):
        raise ValueError("coord softCE support mask is empty")

    log_m = torch.logsumexp(coord_log_probs[support_mask], dim=0)
    support_loss = -log_m
    support_mass = log_m.exp()
    outside_support_mass = 1.0 - support_mass
    balance_loss = -(dist.probs * (coord_log_probs - log_m)).sum()
    pure_soft_ce_equiv = support_loss + balance_loss
    weighted_loss = (
        float(support_weight) * support_loss
        + float(balance_weight) * balance_loss
    )
    kl_like = pure_soft_ce_equiv - dist.entropy

    for name, value in (
        ("weighted_loss", weighted_loss),
        ("support_loss", support_loss),
        ("support_mass", support_mass),
        ("outside_support_mass", outside_support_mass),
        ("balance_loss", balance_loss),
        ("pure_soft_ce_equiv", pure_soft_ce_equiv),
        ("kl_like", kl_like),
    ):
        if not torch.isfinite(value):
            raise ValueError(f"coord softCE produced non-finite {name}")

    return CoordSoftCELoss(
        weighted_loss=weighted_loss,
        support_loss=support_loss,
        support_mass=support_mass,
        outside_support_mass=outside_support_mass,
        balance_loss=balance_loss,
        pure_soft_ce_equiv=pure_soft_ce_equiv,
        target_entropy=dist.entropy,
        kl_like=kl_like,
        peak_prob=dist.peak_prob,
        perplexity=dist.perplexity,
        effective_support_size=dist.effective_support_size,
        target_std=dist.std,
        valid_candidate_count=dist.valid_candidate_count,
        support_mixture=len(candidates) > 1,
    )


def _candidate_iou_gibbs_log_weights(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    log_scores = torch.full_like(bins, -torch.inf, dtype=torch.float64)
    iou = _replaced_slot_iou(bbox_xyxy, slot_name, bins)
    log_scores[valid_mask] = -(1.0 - iou[valid_mask]) / float(tau)
    normalizer = torch.logsumexp(log_scores[valid_mask], dim=0)
    if not torch.isfinite(normalizer):
        raise ValueError("coord soft target candidate has no finite valid scores")
    return log_scores - normalizer


def _valid_replaced_slot_mask(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
) -> torch.Tensor:
    x1, y1, x2, y2 = bbox_xyxy
    if slot_name == "x1":
        return (bins >= 0) & (bins < x2)
    if slot_name == "y1":
        return (bins >= 0) & (bins < y2)
    if slot_name == "x2":
        return (bins > x1) & (bins <= 999)
    if slot_name == "y2":
        return (bins > y1) & (bins <= 999)
    raise ValueError(f"unsupported coordinate slot {slot_name!r}")


def _replaced_slot_iou(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
) -> torch.Tensor:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    coords = (
        bins if slot_name == "x1" else torch.full_like(bins, x1),
        bins if slot_name == "y1" else torch.full_like(bins, y1),
        bins if slot_name == "x2" else torch.full_like(bins, x2),
        bins if slot_name == "y2" else torch.full_like(bins, y2),
    )
    cx1, cy1, cx2, cy2 = coords
    base_x1 = bins.new_tensor(x1)
    base_y1 = bins.new_tensor(y1)
    base_x2 = bins.new_tensor(x2)
    base_y2 = bins.new_tensor(y2)
    inter_w = (torch.minimum(cx2, base_x2) - torch.maximum(cx1, base_x1)).clamp_min(
        0.0
    )
    inter_h = (torch.minimum(cy2, base_y2) - torch.maximum(cy1, base_y1)).clamp_min(
        0.0
    )
    inter = inter_w * inter_h
    base_area = (x2 - x1) * (y2 - y1)
    cand_area = (cx2 - cx1).clamp_min(0.0) * (cy2 - cy1).clamp_min(0.0)
    union = base_area + cand_area - inter
    return torch.where(union > 0.0, inter / union, torch.zeros_like(union))
