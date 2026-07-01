"""Hard-token cross-entropy objective."""

from __future__ import annotations

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
from src.training.supervision.distributions import HardTokenDistribution


class TokenCEObjective:
    """Hard-token cross-entropy over resolved logit rows."""

    objective_id = "token_ce"

    def run(
        self,
        *,
        spec: ObjectiveSpec,
        spans: tuple[ResolvedObjectiveSpan, ...],
        logits: torch.Tensor,
    ) -> ObjectiveResult:
        """Compute objective-local normalized hard-token CE."""

        # return a graph-anchored zero for batches with no compatible spans.
        if len(spans) == 0:
            return ObjectiveResult.zero(
                objective_id=spec.objective_id,
                weight=spec.weight,
                logits=logits,
            )

        # accumulate weighted per-row CE with a state-weight denominator.
        numerators: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        role_numerators: dict[str, torch.Tensor] = {}
        role_denominators: dict[str, torch.Tensor] = {}
        for resolved in spans:
            distribution = resolved.span.distribution
            if type(distribution) is not HardTokenDistribution:
                raise TypeError("token_ce spans must carry HardTokenDistribution")
            _validate_token_id(distribution.token_id, vocab_size=int(logits.shape[-1]))

            with precision_context(resolved.logits):
                row_logits = loss_float(resolved.logits)
                _raise_non_finite(row_logits, name="token_ce logits")
                targets = torch.full(
                    (int(row_logits.shape[0]),),
                    int(distribution.token_id),
                    dtype=torch.long,
                    device=row_logits.device,
                )
                ce = F.cross_entropy(row_logits, targets, reduction="none")
                _raise_non_finite(ce, name="token_ce row losses")

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
            weight = ce.new_full(ce.shape, state_weight)
            weighted = ce * weight * float(loss_weight)
            denom = weight.sum()
            numerator = weighted.sum()

            numerators.append(numerator)
            denominators.append(denom)
            role_numerators[resolved.span.role] = (
                role_numerators.get(resolved.span.role, numerator.new_tensor(0.0))
                + numerator
            )
            role_denominators[resolved.span.role] = (
                role_denominators.get(resolved.span.role, denom.new_tensor(0.0))
                + denom
            )

        numerator = torch.stack(numerators).sum().to(dtype=torch.float32)
        denominator = torch.stack(denominators).sum().to(dtype=torch.float32)
        loss = _safe_normalize(numerator, denominator)
        _raise_non_finite(loss, name="token_ce loss")
        weighted_loss = loss * float(spec.weight)

        # publish canonical metric events and lightweight diagnostics.
        metric_events = (
            make_weighted_mean_event(
                key="training/objectives/token_ce/loss",
                value=loss,
                weight=denominator,
                objective_id=self.objective_id,
            ),
            make_sum_event(
                key="training/objectives/token_ce/span_count",
                value=len(spans),
                objective_id=self.objective_id,
                diagnostic_only=True,
            ),
        )
        state = {
            "role_numerators": MappingProxyType(role_numerators),
            "role_denominators": MappingProxyType(role_denominators),
        }

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
            state=MappingProxyType(state),
        )


def _safe_normalize(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return numerator divided by denominator, or zero when empty."""

    if float(denominator.detach().cpu().item()) <= 0.0:
        return numerator * 0.0

    return (numerator / denominator.clamp(min=1e-6)).to(dtype=torch.float32)


def _validate_token_id(token_id: int, *, vocab_size: int) -> None:
    """Validate a token id against the logits vocabulary."""

    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"token id {token_id} is outside vocab size {vocab_size}")


def _raise_non_finite(tensor: torch.Tensor, *, name: str) -> None:
    """Raise when objective math receives or produces non-finite values."""

    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
