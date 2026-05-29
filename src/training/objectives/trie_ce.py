"""Multi-positive trie cross-entropy objective."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

import torch
import torch.nn.functional as F

from src.training.objectives.types import (
    DEFAULT_PRECISION_POLICY,
    ObjectiveResult,
    ObjectiveSpec,
    ResolvedObjectiveSpan,
    loss_float,
    make_sum_event,
    make_weighted_mean_event,
    metadata_float,
    precision_context,
)
from src.training.supervision.distributions import MultiPositiveTokenDistribution


def support_balance_loss(
    logits_row: torch.Tensor,
    positive_token_ids: Sequence[int] | torch.Tensor,
    positive_weights: Sequence[float] | torch.Tensor | None = None,
    *,
    support_weight: float = 1.0,
    balance_weight: float = 1.0,
) -> torch.Tensor:
    """Return sparse support/balance loss for one resolved logit row."""

    # validate row-local support and probability weights before math.
    if logits_row.ndim != 1:
        raise ValueError("support_balance_loss expects a 1D logits row")
    _raise_non_finite(logits_row, name="support_balance_loss logits")
    positive_ids = _positive_ids_tensor(
        positive_token_ids,
        device=logits_row.device,
        vocab_size=int(logits_row.shape[-1]),
    )
    q = _positive_weights_tensor(
        positive_weights,
        device=logits_row.device,
        count=int(positive_ids.numel()),
    )
    _validate_loss_weight("support_weight", support_weight)
    _validate_loss_weight("balance_weight", balance_weight)

    # compute support mass and balance CE in fp32 with autocast disabled.
    with precision_context(logits_row):
        log_probs = F.log_softmax(loss_float(logits_row), dim=-1)
        valid_log_probs = log_probs.index_select(dim=-1, index=positive_ids)
        log_valid_mass = torch.logsumexp(valid_log_probs, dim=-1)
        support = -log_valid_mass
        balance = -(q * (valid_log_probs - log_valid_mass)).sum(dim=-1)
        loss = float(support_weight) * support + float(balance_weight) * balance
        _raise_non_finite(loss, name="support_balance_loss")

    return loss.to(dtype=torch.float32)


class TrieCEObjective:
    """Multi-positive support/balance CE over resolved logit rows."""

    objective_id = "trie_ce"

    def run(
        self,
        *,
        spec: ObjectiveSpec,
        spans: tuple[ResolvedObjectiveSpan, ...],
        logits: torch.Tensor,
    ) -> ObjectiveResult:
        """Compute objective-local normalized trie CE."""

        # return a graph-anchored zero for batches with no compatible spans.
        if len(spans) == 0:
            return ObjectiveResult.zero(
                objective_id=spec.objective_id,
                weight=spec.weight,
                logits=logits,
            )

        # accumulate state-normalized support/balance losses.
        numerators: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        for resolved in spans:
            distribution = resolved.span.distribution
            if type(distribution) is not MultiPositiveTokenDistribution:
                raise TypeError(
                    "trie_ce spans must carry MultiPositiveTokenDistribution"
                )

            state_weight = metadata_float(
                resolved.span.metadata,
                "state_weight",
                default=1.0,
            )
            loss_weight = metadata_float(
                resolved.span.metadata,
                "loss_weight",
                default=1.0,
            )
            support_weight = metadata_float(
                resolved.span.metadata,
                "support_weight",
                default=1.0,
            )
            balance_weight = metadata_float(
                resolved.span.metadata,
                "balance_weight",
                default=1.0,
            )

            row_losses = [
                support_balance_loss(
                    row_logits,
                    distribution.token_ids,
                    positive_weights=distribution.token_weights,
                    support_weight=support_weight,
                    balance_weight=balance_weight,
                )
                for row_logits in resolved.logits
            ]
            losses = torch.stack(row_losses).to(dtype=torch.float32)
            weights = losses.new_full(losses.shape, state_weight)
            numerator = (losses * weights * float(loss_weight)).sum()
            denominator = weights.sum()

            numerators.append(numerator)
            denominators.append(denominator)

        numerator = torch.stack(numerators).sum().to(dtype=torch.float32)
        denominator = torch.stack(denominators).sum().to(dtype=torch.float32)
        loss = _safe_normalize(numerator, denominator)
        _raise_non_finite(loss, name="trie_ce loss")
        weighted_loss = loss * float(spec.weight)

        # publish canonical metric events for aggregation.
        metric_events = (
            make_weighted_mean_event(
                key="training/objectives/trie_ce/loss",
                value=loss,
                weight=denominator,
                objective_id=self.objective_id,
            ),
            make_sum_event(
                key="training/objectives/trie_ce/span_count",
                value=len(spans),
                objective_id=self.objective_id,
                diagnostic_only=True,
            ),
        )

        return ObjectiveResult(
            objective_id=spec.objective_id,
            loss=loss,
            weighted_loss=weighted_loss,
            numerator=numerator,
            denominator=denominator,
            span_count=len(spans),
            weight=spec.weight,
            precision_policy=DEFAULT_PRECISION_POLICY,
            metric_events=metric_events,
            state=MappingProxyType({}),
        )


def _positive_ids_tensor(
    positive_token_ids: Sequence[int] | torch.Tensor,
    *,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    """Return validated positive token ids as a tensor."""

    if isinstance(positive_token_ids, torch.Tensor):
        if positive_token_ids.dtype == torch.bool:
            raise TypeError("positive_token_ids must contain integer token ids")
        if torch.is_floating_point(positive_token_ids) or torch.is_complex(
            positive_token_ids,
        ):
            raise TypeError("positive_token_ids must contain integer token ids")
        ids = positive_token_ids.to(device=device, dtype=torch.long).reshape(-1)
    else:
        raw_ids = tuple(positive_token_ids)
        for token_id in raw_ids:
            if type(token_id) is not int:
                raise TypeError("positive_token_ids must contain integer token ids")
        ids = torch.tensor(raw_ids, device=device, dtype=torch.long)
    if int(ids.numel()) == 0:
        raise ValueError("positive_token_ids must be non-empty")
    if int(ids.unique().numel()) != int(ids.numel()):
        raise ValueError("positive_token_ids must contain unique token ids")
    if bool(((ids < 0) | (ids >= vocab_size)).any().item()):
        raise ValueError("positive_token_ids contains an id outside logits vocab")

    return ids


def _positive_weights_tensor(
    positive_weights: Sequence[float] | torch.Tensor | None,
    *,
    device: torch.device,
    count: int,
) -> torch.Tensor:
    """Return normalized positive weights in fp32."""

    if positive_weights is None:
        q = torch.ones((count,), device=device, dtype=torch.float32)
    elif isinstance(positive_weights, torch.Tensor):
        if positive_weights.dtype == torch.bool:
            raise TypeError("positive_weights must contain numeric scalar weights")
        q = positive_weights.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        raw_weights = tuple(positive_weights)
        for weight in raw_weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError("positive_weights must contain numeric scalar weights")
        q = torch.tensor(raw_weights, device=device, dtype=torch.float32)
    if int(q.numel()) != count:
        raise ValueError("positive_weights must align with positive_token_ids")
    if not bool(torch.isfinite(q).all().item()) or bool((q < 0).any().item()):
        raise ValueError("positive_weights must be finite and non-negative")
    q_sum = q.sum()
    if float(q_sum.detach().cpu().item()) <= 0.0:
        raise ValueError("positive_weights must have positive mass")

    return q / q_sum.clamp(min=1e-12)


def _validate_loss_weight(name: str, value: float) -> None:
    """Validate a non-negative support/balance weight."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite numeric scalar")
    parsed = float(value)
    if parsed < 0.0 or not torch.isfinite(torch.tensor(parsed)):
        raise ValueError(f"{name} must be finite and >= 0")


def _safe_normalize(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return numerator divided by denominator, or zero when empty."""

    if float(denominator.detach().cpu().item()) <= 0.0:
        return numerator * 0.0

    return (numerator / denominator.clamp(min=1e-6)).to(dtype=torch.float32)


def _raise_non_finite(tensor: torch.Tensor, *, name: str) -> None:
    """Raise when objective math receives or produces non-finite values."""

    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
