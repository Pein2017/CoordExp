"""Pure fp32 objectives for the Human-13 K-union overfit probe.

This module deliberately consumes already selected tensor sites.  Research
semantics such as parsing rows, matching owners, selecting targets, and arm
membership belong to the experiment-local manifest and runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from src.common.errors import LossContractError


@dataclass(frozen=True)
class OwnerMeanCrossEntropyResult:
    raw_loss: torch.Tensor
    numerator: torch.Tensor
    denominator: int
    eligible_owner_count: int
    selected_token_count: int
    all_finite: bool
    math_dtype: str = "float32"


@dataclass(frozen=True)
class PrefixFreeUnionResult:
    raw_loss: torch.Tensor
    numerator: torch.Tensor
    denominator: int
    candidate_weights: tuple[float, ...]
    effective_owner_count: float
    candidate_count: int
    selected_token_count: int
    all_finite: bool
    math_dtype: str = "float32"


@dataclass(frozen=True)
class StreamedPrefixFreeUnionResult:
    raw_loss: torch.Tensor
    numerator: torch.Tensor
    candidate_weights: tuple[float, ...]
    effective_owner_count: float
    candidate_count: int
    all_finite: bool
    math_dtype: str = "float32"


@dataclass(frozen=True)
class CoherentBottleneckResult:
    raw_loss: torch.Tensor
    numerator: torch.Tensor
    denominator: int
    competitor_token_ids: torch.Tensor
    minimum_target_margin: float | None
    violating_site_count: int
    satisfied_site_count: int
    selected_token_count: int
    all_finite: bool
    math_dtype: str = "float32"


@dataclass(frozen=True)
class DuplicateUnlikelihoodResult:
    raw_loss: torch.Tensor
    numerator: torch.Tensor
    denominator: int
    raw_event_count: int
    consumed_event_count: int
    capped_event_count: int
    eligible_image_count: int
    minimum_target_margin: float | None
    all_finite: bool
    math_dtype: str = "float32"


def owner_mean_masked_row_cross_entropy(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> OwnerMeanCrossEntropyResult:
    """Mean token CE within each row, then mean over eligible owners."""

    checked = _checked_logits(logits, ndim=3)
    _require_shape(target_token_ids, checked.shape[:2], name="target_token_ids")
    _require_shape(token_mask, checked.shape[:2], name="token_mask")
    if token_mask.dtype != torch.bool:
        raise LossContractError(
            "token_mask must be boolean",
            code="loss.human13_mask_dtype",
        )
    targets = target_token_ids.to(device=checked.device, dtype=torch.long)
    _check_target_ids(targets, vocab_size=int(checked.shape[-1]))
    mask = token_mask.to(device=checked.device)
    token_losses = F.cross_entropy(
        checked.reshape(-1, checked.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    token_counts = mask.sum(dim=1)
    eligible = token_counts > 0
    row_means = (token_losses * mask).sum(dim=1) / token_counts.clamp_min(1)
    numerator = row_means[eligible].sum()
    denominator = int(eligible.sum().item())
    raw_loss = numerator / denominator if denominator else checked.sum() * 0.0
    return OwnerMeanCrossEntropyResult(
        raw_loss=raw_loss,
        numerator=numerator,
        denominator=denominator,
        eligible_owner_count=denominator,
        selected_token_count=int(mask.sum().item()),
        all_finite=True,
    )


def prefix_free_union_negative_log_mass(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> PrefixFreeUnionResult:
    """Score one image's prefix-free union of complete native row actions."""

    checked = _checked_logits(logits, ndim=3)
    _require_shape(target_token_ids, checked.shape[:2], name="target_token_ids")
    _require_shape(token_mask, checked.shape[:2], name="token_mask")
    if token_mask.dtype != torch.bool:
        raise LossContractError(
            "token_mask must be boolean",
            code="loss.human13_mask_dtype",
        )
    targets = target_token_ids.to(device=checked.device, dtype=torch.long)
    _check_target_ids(targets, vocab_size=int(checked.shape[-1]))
    mask = token_mask.to(device=checked.device)
    candidate_count = int(checked.shape[0])
    if candidate_count == 0:
        zero = checked.sum() * 0.0
        return PrefixFreeUnionResult(
            raw_loss=zero,
            numerator=zero,
            denominator=0,
            candidate_weights=(),
            effective_owner_count=0.0,
            candidate_count=0,
            selected_token_count=0,
            all_finite=True,
        )
    _validate_prefix_free_targets(targets, mask)
    selected_log_probs = (
        torch.log_softmax(checked, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    )
    row_scores = (selected_log_probs * mask).sum(dim=1)
    numerator = -torch.logsumexp(row_scores, dim=0)
    weights = torch.softmax(row_scores, dim=0)
    effective = weights.square().sum().reciprocal()
    _require_finite_outputs(
        numerator=numerator,
        candidate_weights=weights,
        effective_owner_count=effective,
    )
    return PrefixFreeUnionResult(
        raw_loss=numerator,
        numerator=numerator,
        denominator=1,
        candidate_weights=tuple(float(value) for value in weights.detach().cpu()),
        effective_owner_count=float(effective.detach().item()),
        candidate_count=candidate_count,
        selected_token_count=int(mask.sum().item()),
        all_finite=True,
    )


def prefix_free_union_streaming_weights(row_scores: torch.Tensor) -> torch.Tensor:
    """Compute detached fp32 global candidate weights at one fixed parameter state."""

    if row_scores.ndim != 1 or row_scores.numel() == 0:
        raise LossContractError(
            "row_scores must be one nonempty vector",
            code="loss.human13_streaming_scores_shape",
        )
    checked = row_scores.float()
    if not bool(torch.isfinite(checked).all().item()):
        raise LossContractError(
            "row_scores must be finite",
            code="loss.human13_streaming_scores_nonfinite",
        )
    return torch.softmax(checked.detach(), dim=0)


def prefix_free_union_detached_weight_surrogate(
    row_scores: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> StreamedPrefixFreeUnionResult:
    """Return the exact-gradient replay surrogate for a globally scored union."""

    if row_scores.ndim != 1 or candidate_weights.ndim != 1 or (
        row_scores.shape != candidate_weights.shape
    ):
        raise LossContractError(
            "row scores and candidate weights must be aligned vectors",
            code="loss.human13_streaming_weights_shape",
        )
    if row_scores.numel() == 0:
        raise LossContractError(
            "streamed union requires at least one candidate",
            code="loss.human13_streaming_empty",
        )
    scores = row_scores.float()
    weights = candidate_weights.to(device=scores.device, dtype=torch.float32).detach()
    if not bool(torch.isfinite(scores).all().item()) or not bool(
        torch.isfinite(weights).all().item()
    ):
        raise LossContractError(
            "streamed union scores and weights must be finite",
            code="loss.human13_streaming_nonfinite",
        )
    if bool((weights < 0).any().item()) or not torch.allclose(
        weights.sum(),
        weights.new_tensor(1.0),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise LossContractError(
            "streamed union candidate weights must be nonnegative and sum to one",
            code="loss.human13_streaming_weights_normalization",
        )
    numerator = -(weights * scores).sum()
    effective = weights.square().sum().reciprocal()
    _require_finite_outputs(numerator=numerator, effective_owner_count=effective)
    return StreamedPrefixFreeUnionResult(
        raw_loss=numerator,
        numerator=numerator,
        candidate_weights=tuple(float(value) for value in weights.cpu()),
        effective_owner_count=float(effective.detach().item()),
        candidate_count=int(scores.numel()),
        all_finite=True,
    )


def coherent_full_chain_bottleneck_hinge(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    required_margin: float,
) -> CoherentBottleneckResult:
    """Apply A8-prime on one already selected coherent full-H chain."""

    checked_margin = float(required_margin)
    if not math.isfinite(checked_margin) or checked_margin <= 0.0:
        raise LossContractError(
            "required_margin must be positive and finite",
            code="loss.human13_required_margin",
        )
    checked = _checked_logits(logits, ndim=3)
    if int(checked.shape[-1]) < 2:
        raise LossContractError(
            "bottleneck logits require at least one non-target token",
            code="loss.human13_competitor_missing",
        )
    _require_shape(target_token_ids, checked.shape[:2], name="target_token_ids")
    _require_shape(token_mask, checked.shape[:2], name="token_mask")
    if token_mask.dtype != torch.bool:
        raise LossContractError(
            "token_mask must be boolean",
            code="loss.human13_mask_dtype",
        )
    targets = target_token_ids.to(device=checked.device, dtype=torch.long)
    _check_target_ids(targets, vocab_size=int(checked.shape[-1]))
    mask = token_mask.to(device=checked.device)
    target_logits = checked.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    non_target_logits = checked.scatter(
        2,
        targets.unsqueeze(-1),
        float("-inf"),
    )
    competitor_ids = non_target_logits.argmax(dim=-1).detach()
    competitor_logits = checked.gather(2, competitor_ids.unsqueeze(-1)).squeeze(-1)
    target_margins = target_logits - competitor_logits
    site_losses = F.relu(checked.new_tensor(checked_margin) - target_margins)
    token_counts = mask.sum(dim=1)
    eligible = token_counts > 0
    row_means = (site_losses * mask).sum(dim=1) / token_counts.clamp_min(1)
    numerator = row_means[eligible].sum()
    denominator = int(eligible.sum().item())
    raw_loss = numerator / denominator if denominator else checked.sum() * 0.0
    selected_margins = target_margins[mask]
    _require_finite_outputs(raw_loss=raw_loss, numerator=numerator)
    selected_count = int(mask.sum().item())
    return CoherentBottleneckResult(
        raw_loss=raw_loss,
        numerator=numerator,
        denominator=denominator,
        competitor_token_ids=torch.where(
            mask,
            competitor_ids,
            torch.full_like(competitor_ids, -1),
        ).detach(),
        minimum_target_margin=(
            float(selected_margins.detach().min().item()) if selected_count else None
        ),
        violating_site_count=int(
            ((target_margins < checked_margin) & mask).sum().item()
        ),
        satisfied_site_count=int(
            ((target_margins >= checked_margin) & mask).sum().item()
        ),
        selected_token_count=selected_count,
        all_finite=True,
    )


def image_balanced_duplicate_token_unlikelihood(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    image_ids: torch.Tensor,
) -> DuplicateUnlikelihoodResult:
    """Consume every duplicate site, averaging events within image first."""

    checked = _checked_logits(logits, ndim=2)
    event_count = int(checked.shape[0])
    expected_shape = torch.Size((event_count,))
    _require_shape(target_token_ids, expected_shape, name="target_token_ids")
    _require_shape(image_ids, expected_shape, name="image_ids")
    if int(checked.shape[-1]) < 2:
        raise LossContractError(
            "duplicate unlikelihood requires at least one non-target token",
            code="loss.human13_competitor_missing",
        )
    if image_ids.dtype == torch.bool or image_ids.is_floating_point():
        raise LossContractError(
            "image_ids must be integer tensor identifiers",
            code="loss.human13_image_id_dtype",
        )
    if event_count == 0:
        zero = checked.sum() * 0.0
        return DuplicateUnlikelihoodResult(
            raw_loss=zero,
            numerator=zero,
            denominator=0,
            raw_event_count=0,
            consumed_event_count=0,
            capped_event_count=0,
            eligible_image_count=0,
            minimum_target_margin=None,
            all_finite=True,
        )
    targets = target_token_ids.to(device=checked.device, dtype=torch.long)
    _check_target_ids(targets, vocab_size=int(checked.shape[-1]))
    images = image_ids.to(device=checked.device, dtype=torch.long)
    target_logits = checked.gather(1, targets.unsqueeze(1)).squeeze(1)
    non_target_logits = checked.scatter(1, targets.unsqueeze(1), float("-inf"))
    non_target_logsumexp = torch.logsumexp(non_target_logits, dim=1)
    target_margins = target_logits - non_target_logsumexp
    event_losses = F.softplus(target_margins)
    image_means = torch.stack(
        tuple(
            event_losses[images == image_id].mean() for image_id in torch.unique(images)
        )
    )
    numerator = image_means.sum()
    denominator = int(image_means.numel())
    raw_loss = numerator / denominator
    _require_finite_outputs(
        raw_loss=raw_loss,
        numerator=numerator,
        event_losses=event_losses,
        target_margins=target_margins,
    )
    return DuplicateUnlikelihoodResult(
        raw_loss=raw_loss,
        numerator=numerator,
        denominator=denominator,
        raw_event_count=event_count,
        consumed_event_count=event_count,
        capped_event_count=0,
        eligible_image_count=denominator,
        minimum_target_margin=float(target_margins.detach().min().item()),
        all_finite=True,
    )


def _checked_logits(logits: torch.Tensor, *, ndim: int) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != ndim:
        raise LossContractError(
            f"logits must be a rank-{ndim} tensor",
            code="loss.human13_logits_shape",
        )
    if not logits.is_floating_point():
        raise LossContractError(
            "logits must be floating point",
            code="loss.human13_logits_dtype",
        )
    checked = logits.float()
    if not bool(torch.isfinite(checked.detach()).all().item()):
        raise LossContractError(
            "logits must be finite",
            code="loss.human13_nonfinite_logits",
        )
    return checked


def _require_shape(value: torch.Tensor, shape: torch.Size, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise LossContractError(
            f"{name} shape must match logits sites",
            code="loss.human13_site_shape",
            context={"name": name, "expected": tuple(shape)},
        )


def _check_target_ids(targets: torch.Tensor, *, vocab_size: int) -> None:
    if targets.numel() and (
        bool((targets < 0).any().item()) or bool((targets >= vocab_size).any().item())
    ):
        raise LossContractError(
            "target token id is outside the logits vocabulary",
            code="loss.human13_target_id",
            context={"vocab_size": vocab_size},
        )


def _validate_prefix_free_targets(
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    sequences: list[tuple[int, ...]] = []
    for row_index in range(int(targets.shape[0])):
        row_mask = mask[row_index]
        selected_count = int(row_mask.sum().item())
        expected_mask = (
            torch.arange(row_mask.numel(), device=row_mask.device) < selected_count
        )
        if selected_count == 0 or not torch.equal(row_mask, expected_mask):
            raise LossContractError(
                "candidate masks must be nonempty contiguous prefixes",
                code="loss.human13_candidate_mask",
                context={"candidate_index": row_index},
            )
        sequences.append(
            tuple(int(value) for value in targets[row_index, :selected_count].tolist())
        )
    for left_index, left in enumerate(sequences):
        for right_index, right in enumerate(sequences):
            if left_index == right_index:
                continue
            if len(left) <= len(right) and right[: len(left)] == left:
                raise LossContractError(
                    "candidate actions must be exact-token distinct and prefix-free",
                    code="loss.human13_candidates_not_prefix_free",
                    context={
                        "prefix_candidate_index": left_index,
                        "candidate_index": right_index,
                    },
                )


def _require_finite_outputs(**values: torch.Tensor) -> None:
    if not all(
        bool(torch.isfinite(value.detach()).all().item()) for value in values.values()
    ):
        raise LossContractError(
            "objective math produced a nonfinite value",
            code="loss.human13_nonfinite_output",
            context={
                "values": tuple(
                    name
                    for name, value in values.items()
                    if not bool(torch.isfinite(value.detach()).all().item())
                )
            },
        )


__all__ = [
    "CoherentBottleneckResult",
    "DuplicateUnlikelihoodResult",
    "OwnerMeanCrossEntropyResult",
    "PrefixFreeUnionResult",
    "StreamedPrefixFreeUnionResult",
    "coherent_full_chain_bottleneck_hinge",
    "image_balanced_duplicate_token_unlikelihood",
    "owner_mean_masked_row_cross_entropy",
    "prefix_free_union_negative_log_mass",
    "prefix_free_union_detached_weight_surrogate",
    "prefix_free_union_streaming_weights",
]
