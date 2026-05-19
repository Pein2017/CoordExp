from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

CoordSlotName = Literal["x1", "y1", "x2", "y2"]
CoordSoftTargetDistributionName = Literal[
    "iou_gibbs_v0", "ciou_gibbs_v0", "instance_trie_gaussian"
]
_COORD_SLOT_NAMES: tuple[str, ...] = ("x1", "y1", "x2", "y2")
_COORD_TARGET_DISTRIBUTIONS: tuple[str, ...] = (
    "iou_gibbs_v0",
    "ciou_gibbs_v0",
    "instance_trie_gaussian",
)
_COORD_SLOT_INDEX: dict[str, int] = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}


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
            raise ValueError("bbox_xyxy must contain integer coord-token bins")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
            raise ValueError(
                f"bbox_xyxy must be valid token-space xyxy; got {self.bbox_xyxy}"
            )
        if (
            not math.isfinite(float(self.probability))
            or float(self.probability) <= 0.0
        ):
            raise ValueError("coord soft target probability must be finite and > 0")


@dataclass(frozen=True)
class CoordSoftTargetRuntimeConfig:
    target_distribution: CoordSoftTargetDistributionName
    tau: float | None = None
    coord_token_start: int = 0
    coord_token_end: int = 999
    weighting: Literal["preserve_recursive_support_balance"] = (
        "preserve_recursive_support_balance"
    )
    apply_to_multi_positive: Literal["support_mixture"] = "support_mixture"
    gaussian_mixture_weight: float = 1.0
    gaussian_r95_axis_fraction: float = 0.04
    gaussian_r95_cap_bins: int = 8

    def __post_init__(self) -> None:
        if self.target_distribution not in _COORD_TARGET_DISTRIBUTIONS:
            raise ValueError(
                "coord soft target_distribution must be iou_gibbs_v0, "
                "ciou_gibbs_v0, or instance_trie_gaussian"
            )
        if self.target_distribution in {"iou_gibbs_v0", "ciou_gibbs_v0"}:
            if (
                self.tau is None
                or not math.isfinite(float(self.tau))
                or float(self.tau) <= 0.0
            ):
                raise ValueError("coord soft target tau must be finite and > 0")
            if float(self.gaussian_mixture_weight) != 1.0:
                raise ValueError(
                    "coord soft target gaussian_mixture_weight is only supported "
                    "for instance_trie_gaussian"
                )
        elif self.tau is not None:
            raise ValueError(
                "coord soft target tau is not supported for instance_trie_gaussian"
            )
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
        if not isinstance(self.gaussian_mixture_weight, (int, float)) or isinstance(
            self.gaussian_mixture_weight, bool
        ):
            raise TypeError("coord soft target gaussian_mixture_weight must be numeric")
        if not math.isfinite(float(self.gaussian_mixture_weight)):
            raise ValueError(
                "coord soft target gaussian_mixture_weight must be finite and within [0, 1]"
            )
        if not 0.0 <= float(self.gaussian_mixture_weight) <= 1.0:
            raise ValueError(
                "coord soft target gaussian_mixture_weight must be within [0, 1]"
            )
        if not isinstance(self.gaussian_r95_axis_fraction, (int, float)) or isinstance(
            self.gaussian_r95_axis_fraction, bool
        ):
            raise TypeError("coord soft target gaussian_r95_axis_fraction must be numeric")
        if (
            not math.isfinite(float(self.gaussian_r95_axis_fraction))
            or float(self.gaussian_r95_axis_fraction) <= 0.0
            or float(self.gaussian_r95_axis_fraction) > 1.0
        ):
            raise ValueError(
                "coord soft target gaussian_r95_axis_fraction must be finite and within (0, 1]"
            )
        if not isinstance(self.gaussian_r95_cap_bins, int) or isinstance(
            self.gaussian_r95_cap_bins, bool
        ):
            raise TypeError("coord soft target gaussian_r95_cap_bins must be an integer")
        if int(self.gaussian_r95_cap_bins) < 0 or int(self.gaussian_r95_cap_bins) > 999:
            raise ValueError(
                "coord soft target gaussian_r95_cap_bins must be within [0, 999]"
            )

    @property
    def coord_bins(self) -> int:
        return int(self.coord_token_end) - int(self.coord_token_start) + 1

    def coord_token_ids(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return torch.arange(
            int(self.coord_token_start),
            int(self.coord_token_end) + 1,
            dtype=torch.long,
            device=device,
        )

    def coord_value_to_token_id(self, coord_value: int) -> int:
        if not isinstance(coord_value, int) or isinstance(coord_value, bool):
            raise ValueError("coord value must be an integer")
        if not (0 <= int(coord_value) < self.coord_bins):
            raise ValueError(
                f"coord value must be in [0, {self.coord_bins - 1}], got {coord_value}"
            )
        return int(self.coord_token_start) + int(coord_value)


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
    target_r95_radius: torch.Tensor
    candidate_count: torch.Tensor
    support_bin_count: torch.Tensor
    valid_candidate_count: torch.Tensor
    posterior_entropy: torch.Tensor | None = None
    posterior_top1: torch.Tensor | None = None
    effective_candidate_count: torch.Tensor | None = None
    posterior: dict[str, torch.Tensor] = field(default_factory=dict)
    component_probs_by_id: dict[str, torch.Tensor] = field(default_factory=dict)


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
    target_r95_radius: torch.Tensor
    candidate_count: torch.Tensor
    support_bin_count: torch.Tensor
    valid_candidate_count: torch.Tensor
    support_mixture: bool
    posterior_entropy: torch.Tensor | None = None
    posterior_top1: torch.Tensor | None = None
    effective_candidate_count: torch.Tensor | None = None


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
        log_weights = _candidate_gibbs_log_weights(
            candidate.bbox_xyxy,
            slot_name,
            bins,
            valid_mask,
            target_distribution=cfg.target_distribution,
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

    candidate_count = torch.tensor(float(len(candidates)), dtype=torch.float32)
    support_bin_count = support_mask.sum().to(dtype=torch.float32)

    return CoordSoftTargetDistribution(
        token_ids=token_ids.to(device=target_device),
        probs=probs.to(device=target_device),
        support_mask=support_mask.to(device=target_device),
        entropy=entropy.to(device=target_device),
        peak_prob=probs.max().to(device=target_device),
        perplexity=perplexity.to(device=target_device),
        effective_support_size=effective_support_size.to(device=target_device),
        std=std.to(device=target_device),
        target_r95_radius=torch.tensor(0.0, dtype=torch.float32, device=target_device),
        candidate_count=candidate_count.to(device=target_device),
        support_bin_count=support_bin_count.to(device=target_device),
        valid_candidate_count=support_bin_count.to(device=target_device),
    )


def build_coord_soft_target(
    candidates: Sequence[CoordSoftTargetCandidate],
    cfg: CoordSoftTargetRuntimeConfig,
    *,
    current_slot: CoordSlotName | None = None,
    teacher_prefix_values: Mapping[str, int] | None = None,
    device: torch.device | str | None = None,
    return_components: bool = False,
) -> CoordSoftTargetDistribution:
    if cfg.target_distribution in {"iou_gibbs_v0", "ciou_gibbs_v0"}:
        return build_iou_gibbs_coord_target(candidates, cfg, device=device)
    if cfg.target_distribution == "instance_trie_gaussian":
        return _build_instance_trie_gaussian_coord_target(
            candidates,
            cfg,
            current_slot=current_slot,
            teacher_prefix_values=teacher_prefix_values,
            device=device,
            return_components=return_components,
        )
    raise ValueError(
        "coord soft target_distribution must be iou_gibbs_v0, "
        "ciou_gibbs_v0, or instance_trie_gaussian"
    )


def full_vocab_coord_soft_ce(
    logits: torch.Tensor,
    candidates: Sequence[CoordSoftTargetCandidate],
    cfg: CoordSoftTargetRuntimeConfig,
    *,
    current_slot: CoordSlotName | None = None,
    teacher_prefix_values: Mapping[str, int] | None = None,
    teacher_coord_value: int | None = None,
    support_weight: float | None = None,
    balance_weight: float | None = None,
) -> CoordSoftCELoss:
    if cfg.target_distribution in {"iou_gibbs_v0", "ciou_gibbs_v0"}:
        if support_weight is None or balance_weight is None:
            raise ValueError(
                "legacy coord softCE requires support_weight and balance_weight"
            )
        return full_vocab_coord_support_balance_ce(
            logits,
            candidates,
            cfg,
            support_weight=float(support_weight),
            balance_weight=float(balance_weight),
        )
    if cfg.target_distribution != "instance_trie_gaussian":
        raise ValueError(
            "coord soft target_distribution must be iou_gibbs_v0, "
            "ciou_gibbs_v0, or instance_trie_gaussian"
        )
    if logits.ndim != 1:
        raise ValueError("coord softCE logits must be a 1D full-vocab tensor")
    if not torch.isfinite(logits.float()).all():
        raise ValueError("coord softCE received non-finite logits")

    dist = build_coord_soft_target(
        candidates,
        cfg,
        current_slot=current_slot,
        teacher_prefix_values=teacher_prefix_values,
        device=logits.device,
    )
    if int(dist.token_ids.max().item()) >= int(logits.shape[-1]):
        raise ValueError("coord token id exceeds logits vocab size")

    coord_logits = logits.float().index_select(0, dist.token_ids)
    coord_log_probs = F.log_softmax(coord_logits, dim=-1)
    target_probs = dist.probs.to(dtype=coord_log_probs.dtype)
    gaussian_mixture_weight = float(cfg.gaussian_mixture_weight)
    if gaussian_mixture_weight < 1.0:
        if teacher_coord_value is None:
            raise ValueError(
                "teacher_coord_value is required when gaussian_mixture_weight < 1"
            )
        if not isinstance(teacher_coord_value, int) or isinstance(
            teacher_coord_value, bool
        ):
            raise ValueError("teacher_coord_value must be an integer")
        teacher_index = int(teacher_coord_value)
        if not (0 <= teacher_index < cfg.coord_bins):
            raise ValueError(
                f"teacher_coord_value must be in [0, {cfg.coord_bins - 1}], "
                f"got {teacher_coord_value}"
            )
        if not bool(dist.support_mask[teacher_index].item()):
            raise ValueError("teacher_coord_value is outside the coordinate support mask")
        target_probs = target_probs * gaussian_mixture_weight
        target_probs[teacher_index] = target_probs[teacher_index] + (
            1.0 - gaussian_mixture_weight
        )
        total_target_prob = target_probs.sum()
        if not torch.isfinite(total_target_prob) or float(total_target_prob.item()) <= 0:
            raise ValueError("CE-anchored coord soft target is not normalizable")
        target_probs = target_probs / total_target_prob

    weighted_loss = -(target_probs * coord_log_probs).sum()
    if not torch.isfinite(weighted_loss):
        raise ValueError("coord softCE produced non-finite weighted_loss")

    support_mask = dist.support_mask
    if not torch.any(support_mask):
        raise ValueError("coord softCE support mask is empty")
    coord_probs = coord_log_probs.exp()
    support_mass = coord_probs[support_mask].sum()
    outside_support_mass = coord_probs[~support_mask].sum()
    target_entropy = _entropy(target_probs.to(dtype=torch.float64)).to(
        dtype=weighted_loss.dtype, device=weighted_loss.device
    )
    kl_like = weighted_loss - target_entropy
    effective_support_size = 1.0 / torch.square(target_probs).sum()
    bins = torch.arange(
        cfg.coord_bins,
        dtype=target_probs.dtype,
        device=target_probs.device,
    )
    target_mean = (target_probs * bins).sum()
    target_variance = (target_probs * torch.square(bins - target_mean)).sum()
    target_std = target_variance.clamp_min(0.0).sqrt()

    return CoordSoftCELoss(
        weighted_loss=weighted_loss,
        support_loss=weighted_loss,
        support_mass=support_mass,
        outside_support_mass=outside_support_mass,
        balance_loss=torch.zeros_like(weighted_loss),
        pure_soft_ce_equiv=weighted_loss,
        target_entropy=target_entropy,
        kl_like=kl_like,
        peak_prob=target_probs.max(),
        perplexity=target_entropy.exp(),
        effective_support_size=effective_support_size,
        target_std=target_std,
        target_r95_radius=dist.target_r95_radius,
        candidate_count=dist.candidate_count,
        support_bin_count=dist.support_bin_count,
        valid_candidate_count=dist.valid_candidate_count,
        support_mixture=len(candidates) > 1,
        posterior_entropy=dist.posterior_entropy,
        posterior_top1=dist.posterior_top1,
        effective_candidate_count=dist.effective_candidate_count,
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
        target_r95_radius=dist.target_r95_radius,
        candidate_count=dist.candidate_count,
        support_bin_count=dist.support_bin_count,
        valid_candidate_count=dist.valid_candidate_count,
        support_mixture=len(candidates) > 1,
    )


def _build_instance_trie_gaussian_coord_target(
    candidates: Sequence[CoordSoftTargetCandidate],
    cfg: CoordSoftTargetRuntimeConfig,
    *,
    current_slot: CoordSlotName | None,
    teacher_prefix_values: Mapping[str, int] | None,
    device: torch.device | str | None,
    return_components: bool,
) -> CoordSoftTargetDistribution:
    if cfg.coord_bins != 1000:
        raise ValueError(
            "instance_trie_gaussian coord soft targets require exactly 1000 coord bins"
        )
    if not candidates:
        raise ValueError("coord soft target candidates must be non-empty")

    slot_name = _resolve_current_slot(candidates, current_slot)
    if any(candidate.slot_name != slot_name for candidate in candidates):
        raise ValueError(
            "coord soft target candidates must share the same coordinate slot"
        )
    object_ids = [candidate.object_instance_id for candidate in candidates]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(
            "coord soft target candidate object_instance_id values must be unique"
        )

    prefix_values = _validate_teacher_prefix_values(teacher_prefix_values or {})
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    bins = torch.arange(1000, dtype=torch.float64)
    support_mask = torch.zeros((1000,), dtype=torch.bool)
    components: list[torch.Tensor] = []
    component_r95_radii: list[torch.Tensor] = []
    posterior_log_weights: list[torch.Tensor] = []

    for candidate in candidates:
        valid_mask = _instance_slot_valid_mask(
            candidate.bbox_xyxy,
            slot_name,
            bins,
        )
        support_mask |= valid_mask
        log_component = _instance_slot_gaussian_log_component(
            candidate.bbox_xyxy,
            slot_name,
            bins,
            valid_mask,
            cfg,
        )
        components.append(torch.exp(log_component))
        component_r95_radii.append(
            torch.tensor(
                float(_slot_r95_radius(candidate.bbox_xyxy, slot_name, cfg)),
                dtype=torch.float64,
            )
        )
        posterior_log_weights.append(
            _instance_prefix_log_compatibility(
                candidate.bbox_xyxy,
                slot_name,
                prefix_values,
                cfg,
            )
        )

    stacked_components = torch.stack(components, dim=0)
    stacked_r95_radii = torch.stack(component_r95_radii, dim=0)
    posterior_logits = torch.stack(posterior_log_weights, dim=0)
    if not torch.isfinite(posterior_logits).any():
        raise ValueError(
            "coord soft target has no finite prefix-compatible instance candidates"
        )
    posterior = torch.softmax(posterior_logits, dim=0)
    probs = (posterior[:, None] * stacked_components).sum(dim=0)
    total_prob = probs.sum()
    if not torch.isfinite(total_prob) or float(total_prob.item()) <= 0.0:
        raise ValueError("coord soft target probabilities are not normalizable")
    probs = probs / total_prob
    if not torch.isfinite(probs).all():
        raise ValueError("coord soft target probabilities contain non-finite values")
    if torch.any(probs[~support_mask] != 0):
        raise ValueError("coord soft target assigned mass outside geometry support")

    entropy = _entropy(probs)
    posterior_entropy = _entropy(posterior)
    effective_support_size = 1.0 / torch.square(probs).sum()
    effective_candidate_count = 1.0 / torch.square(posterior).sum()
    mean = (probs * bins).sum()
    variance = (probs * torch.square(bins - mean)).sum()
    std = variance.clamp_min(0.0).sqrt()
    target_r95_radius = (posterior * stacked_r95_radii).sum()
    candidate_count = torch.tensor(float(len(candidates)), dtype=torch.float32)
    support_bin_count = support_mask.sum().to(dtype=torch.float32)
    valid_candidate_count = torch.tensor(float(len(candidates)), dtype=torch.float32)

    posterior_by_id = {
        candidate.object_instance_id: posterior[index].to(device=target_device)
        for index, candidate in enumerate(candidates)
    }
    component_probs_by_id = (
        {
            candidate.object_instance_id: stacked_components[index].to(
                device=target_device
            )
            for index, candidate in enumerate(candidates)
        }
        if return_components
        else {}
    )

    return CoordSoftTargetDistribution(
        token_ids=cfg.coord_token_ids(device=target_device),
        probs=probs.to(device=target_device),
        support_mask=support_mask.to(device=target_device),
        entropy=entropy.to(device=target_device),
        peak_prob=probs.max().to(device=target_device),
        perplexity=entropy.exp().to(device=target_device),
        effective_support_size=effective_support_size.to(device=target_device),
        std=std.to(device=target_device),
        target_r95_radius=target_r95_radius.to(device=target_device),
        candidate_count=candidate_count.to(device=target_device),
        support_bin_count=support_bin_count.to(device=target_device),
        valid_candidate_count=valid_candidate_count.to(device=target_device),
        posterior_entropy=posterior_entropy.to(device=target_device),
        posterior_top1=posterior.max().to(device=target_device),
        effective_candidate_count=effective_candidate_count.to(device=target_device),
        posterior=posterior_by_id,
        component_probs_by_id=component_probs_by_id,
    )


def _resolve_current_slot(
    candidates: Sequence[CoordSoftTargetCandidate],
    current_slot: CoordSlotName | None,
) -> CoordSlotName:
    if current_slot is not None:
        if current_slot not in _COORD_SLOT_NAMES:
            raise ValueError(f"unsupported coordinate slot {current_slot!r}")
        return current_slot
    slot_name = candidates[0].slot_name
    if any(candidate.slot_name != slot_name for candidate in candidates):
        raise ValueError(
            "coord soft target candidates must share the same coordinate slot"
        )
    return slot_name


def _validate_teacher_prefix_values(
    teacher_prefix_values: Mapping[str, int],
) -> dict[str, int]:
    values: dict[str, int] = {}
    for slot_name, value in teacher_prefix_values.items():
        if slot_name not in _COORD_SLOT_NAMES:
            raise ValueError(f"unsupported coordinate slot {slot_name!r}")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("teacher prefix coordinate values must be integers")
        if not (0 <= int(value) <= 999):
            raise ValueError("teacher prefix coordinate values must be in [0, 999]")
        values[str(slot_name)] = int(value)
    return values


def _instance_slot_gaussian_log_component(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: CoordSoftTargetRuntimeConfig,
) -> torch.Tensor:
    if not torch.any(valid_mask):
        raise ValueError("coord soft target candidate has no structurally legal bins")
    center = int(bbox_xyxy[_COORD_SLOT_INDEX[slot_name]])
    log_scores = torch.full_like(bins, -torch.inf, dtype=torch.float64)
    radius = _slot_r95_radius(bbox_xyxy, slot_name, cfg)
    if radius == 0:
        if not (0 <= center < int(bins.numel())) or not bool(valid_mask[center].item()):
            raise ValueError("coord soft target exact center is structurally illegal")
        log_scores[center] = 0.0
        return log_scores
    variance = _r95_radius_variance(radius)
    delta = bins - bins.new_tensor(float(center))
    log_scores[valid_mask] = -0.5 * torch.square(delta[valid_mask]) / variance
    normalizer = torch.logsumexp(log_scores[valid_mask], dim=0)
    if not torch.isfinite(normalizer):
        raise ValueError("coord soft target candidate has no finite valid scores")
    return log_scores - normalizer


def _instance_prefix_log_compatibility(
    bbox_xyxy: tuple[int, int, int, int],
    current_slot: CoordSlotName,
    teacher_prefix_values: Mapping[str, int],
    cfg: CoordSoftTargetRuntimeConfig,
) -> torch.Tensor:
    log_weight = torch.tensor(0.0, dtype=torch.float64)
    for slot_name in _causal_previous_slots(current_slot):
        if slot_name not in teacher_prefix_values:
            continue
        center = int(bbox_xyxy[_COORD_SLOT_INDEX[slot_name]])
        radius = _slot_r95_radius(bbox_xyxy, slot_name, cfg)
        if radius == 0:
            if int(teacher_prefix_values[slot_name]) != center:
                return torch.tensor(-math.inf, dtype=torch.float64)
            continue
        delta = float(teacher_prefix_values[slot_name]) - float(center)
        variance = _r95_radius_variance(radius)
        log_weight = log_weight + (-0.5 * (delta**2) / variance)
    return log_weight


def _causal_previous_slots(current_slot: CoordSlotName) -> tuple[CoordSlotName, ...]:
    index = _COORD_SLOT_NAMES.index(current_slot)
    return _COORD_SLOT_NAMES[:index]  # type: ignore[return-value]


def _slot_axis_len(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
) -> int:
    x1, y1, x2, y2 = bbox_xyxy
    if slot_name in {"x1", "x2"}:
        return int(x2 - x1)
    if slot_name in {"y1", "y2"}:
        return int(y2 - y1)
    raise ValueError(f"unsupported coordinate slot {slot_name!r}")


def _slot_r95_radius(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    cfg: CoordSoftTargetRuntimeConfig,
) -> int:
    axis_len = _slot_axis_len(bbox_xyxy, slot_name)
    cap = int(cfg.gaussian_r95_cap_bins)
    fractional_radius = float(cfg.gaussian_r95_axis_fraction) * float(axis_len)
    return int(math.floor(min(float(cap), fractional_radius)))


def _r95_radius_variance(radius: int) -> float:
    if radius <= 0:
        raise ValueError("R95 radius variance is undefined for radius <= 0")
    sigma = float(radius) / 1.96
    return sigma * sigma


def _instance_slot_valid_mask(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
) -> torch.Tensor:
    x1, y1, x2, y2 = bbox_xyxy
    if slot_name == "x1":
        right = x2
        return (bins >= 0) & (bins < right)
    if slot_name == "y1":
        bottom = y2
        return (bins >= 0) & (bins < bottom)
    if slot_name == "x2":
        return (bins > x1) & (bins <= 999)
    if slot_name == "y2":
        return (bins > y1) & (bins <= 999)
    raise ValueError(f"unsupported coordinate slot {slot_name!r}")


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    positive = probs > 0
    return -(probs[positive] * probs[positive].log()).sum()


def _candidate_gibbs_log_weights(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_distribution: CoordSoftTargetDistributionName,
    tau: float,
) -> torch.Tensor:
    log_scores = torch.full_like(bins, -torch.inf, dtype=torch.float64)
    if target_distribution == "iou_gibbs_v0":
        score = _replaced_slot_iou(bbox_xyxy, slot_name, bins)
    elif target_distribution == "ciou_gibbs_v0":
        score = _replaced_slot_ciou(bbox_xyxy, slot_name, bins)
    else:
        raise ValueError(
            "coord soft target_distribution must be iou_gibbs_v0 or ciou_gibbs_v0"
        )
    log_scores[valid_mask] = -(1.0 - score[valid_mask]) / float(tau)
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
    cx1, cy1, cx2, cy2 = _replaced_slot_coords(bbox_xyxy, slot_name, bins)
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
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


def _replaced_slot_ciou(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
) -> torch.Tensor:
    cx1, cy1, cx2, cy2 = _replaced_slot_coords(bbox_xyxy, slot_name, bins)
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    base_x1 = bins.new_tensor(x1)
    base_y1 = bins.new_tensor(y1)
    base_x2 = bins.new_tensor(x2)
    base_y2 = bins.new_tensor(y2)

    iou = _replaced_slot_iou(bbox_xyxy, slot_name, bins)
    cand_center_x = (cx1 + cx2) * 0.5
    cand_center_y = (cy1 + cy2) * 0.5
    base_center_x = (base_x1 + base_x2) * 0.5
    base_center_y = (base_y1 + base_y2) * 0.5
    center_distance_sq = torch.square(cand_center_x - base_center_x) + torch.square(
        cand_center_y - base_center_y
    )
    enclosing_w = torch.maximum(cx2, base_x2) - torch.minimum(cx1, base_x1)
    enclosing_h = torch.maximum(cy2, base_y2) - torch.minimum(cy1, base_y1)
    enclosing_diag_sq = torch.square(enclosing_w) + torch.square(enclosing_h)
    center_penalty = center_distance_sq / enclosing_diag_sq.clamp_min(1e-12)

    cand_w = (cx2 - cx1).clamp_min(1e-12)
    cand_h = (cy2 - cy1).clamp_min(1e-12)
    base_w = base_x2 - base_x1
    base_h = base_y2 - base_y1
    aspect_delta = torch.atan(base_w / base_h) - torch.atan(cand_w / cand_h)
    v = (4.0 / (math.pi**2)) * torch.square(aspect_delta)
    alpha = v / (1.0 - iou + v).clamp_min(1e-12)
    return iou - center_penalty - alpha * v


def _replaced_slot_coords(
    bbox_xyxy: tuple[int, int, int, int],
    slot_name: CoordSlotName,
    bins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return (
        bins if slot_name == "x1" else torch.full_like(bins, x1),
        bins if slot_name == "y1" else torch.full_like(bins, y1),
        bins if slot_name == "x2" else torch.full_like(bins, x2),
        bins if slot_name == "y2" else torch.full_like(bins, y2),
    )
