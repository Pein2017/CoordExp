"""Coordinate soft-token objective."""

from __future__ import annotations

import math
from types import MappingProxyType

import torch
import torch.nn.functional as F

from src.training.objectives.types import (
    CoordinateVocabulary,
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
from src.training.supervision.distributions import CoordinateSoftTokenDistribution


class CoordinateSoftCEObjective:
    """Coordinate soft CE over resolved coordinate-token rows."""

    objective_id = "coord_soft_ce"
    _ALLOWED_CONFIG_KEYS = frozenset(("coord_token_ids",))
    _DEFERRED_CONFIG_KEYS = frozenset(
        (
            "coord_gate_weight",
            "text_gate_weight",
            "soft_ce_weight",
            "w1_weight",
            "coord_ce_weight",
            "target_sigma",
            "target_truncate",
        )
    )

    def run(
        self,
        *,
        spec: ObjectiveSpec,
        spans: tuple[ResolvedObjectiveSpan, ...],
        logits: torch.Tensor,
    ) -> ObjectiveResult:
        """Compute objective-local normalized coordinate soft-token loss."""

        # validate config before any spans can silently no-op unsupported knobs.
        self._validate_config(spec.config)

        # return a graph-anchored zero for batches with no compatible spans.
        if len(spans) == 0:
            return ObjectiveResult.zero(
                objective_id=spec.objective_id,
                weight=spec.weight,
                logits=logits,
            )

        # load full coordinate vocabulary only for modes that need bin ordering.
        requires_coord_vocab = any(
            resolved.span.distribution.loss_mode in {"coord_vocab_ce", "w1_distance"}
            for resolved in spans
            if type(resolved.span.distribution) is CoordinateSoftTokenDistribution
        )
        coord_token_ids = None
        if requires_coord_vocab:
            coord_token_ids = CoordinateVocabulary.from_config(
                spec.config,
                require_1000_bins=False,
            ).as_tensor(device=logits.device)
            if int(coord_token_ids.max().detach().cpu().item()) >= int(logits.shape[-1]):
                raise ValueError("coord_token_ids exceed logits vocab size")

        # accumulate soft coordinate targets with state denominators.
        numerators: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        entropies: list[float] = []
        support_counts: list[int] = []
        for resolved in spans:
            distribution = resolved.span.distribution
            if type(distribution) is not CoordinateSoftTokenDistribution:
                raise TypeError(
                    "coord_soft_ce spans must carry CoordinateSoftTokenDistribution"
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
            token_ids, probabilities = _target_support(
                distribution,
                device=resolved.logits.device,
                vocab_size=int(logits.shape[-1]),
            )

            row_losses = [
                _coordinate_row_loss(
                    row_logits,
                    token_ids=token_ids,
                    probabilities=probabilities,
                    loss_mode=distribution.loss_mode,
                    coord_token_ids=coord_token_ids,
                )
                for row_logits in resolved.logits
            ]
            losses = torch.stack(row_losses).to(dtype=torch.float32)
            weights = losses.new_full(losses.shape, state_weight)
            numerator = (losses * weights * float(loss_weight)).sum()
            denominator = weights.sum()

            numerators.append(numerator)
            denominators.append(denominator)
            entropies.append(_target_entropy(probabilities))
            support_counts.append(int(token_ids.numel()))

        numerator = torch.stack(numerators).sum().to(dtype=torch.float32)
        denominator = torch.stack(denominators).sum().to(dtype=torch.float32)
        loss = _safe_normalize(numerator, denominator)
        weighted_loss = loss * float(spec.weight)
        support_count = max(support_counts) if support_counts else 0
        entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0

        # publish canonical metric events and deterministic diagnostics.
        metric_events = (
            make_weighted_mean_event(
                key="training/objectives/coord_soft_ce/loss",
                value=loss,
                weight=denominator,
                objective_id=self.objective_id,
            ),
            make_sum_event(
                key="training/objectives/coord_soft_ce/span_count",
                value=len(spans),
                objective_id=self.objective_id,
                diagnostic_only=True,
            ),
        )
        state = {
            "support_token_count": support_count,
            "target_entropy": entropy,
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

    def _validate_config(self, config: object) -> None:
        """Reject unsupported coordinate objective config keys."""

        # keep the skeleton strict so legacy knobs cannot become silent no-ops.
        if not isinstance(config, dict) and not isinstance(config, MappingProxyType):
            return
        keys = set(config.keys())
        deferred = keys & self._DEFERRED_CONFIG_KEYS
        if deferred:
            raise NotImplementedError(
                "deferred/unsupported coord_soft_ce config keys: "
                f"{', '.join(sorted(deferred))}"
            )
        unsupported = keys - self._ALLOWED_CONFIG_KEYS
        if unsupported:
            raise ValueError(
                "deferred/unsupported coord_soft_ce config keys: "
                f"{', '.join(sorted(unsupported))}"
            )


def _coordinate_row_loss(
    row_logits: torch.Tensor,
    *,
    token_ids: torch.Tensor,
    probabilities: torch.Tensor,
    loss_mode: str,
    coord_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Return one coordinate soft-token loss in fp32."""

    # compute the configured coordinate loss mode with autocast disabled.
    with precision_context(row_logits):
        row = loss_float(row_logits)
        if loss_mode == "full_vocab_ce":
            log_probs = F.log_softmax(row, dim=-1)
            selected = log_probs.index_select(dim=-1, index=token_ids)
            return -(probabilities * selected).sum().to(dtype=torch.float32)
        if loss_mode == "coord_vocab_ce":
            coord_ids = _require_coord_token_ids(coord_token_ids, loss_mode=loss_mode)
            target_bins = _target_bins(
                token_ids,
                coord_token_ids=coord_ids,
            )
            coord_logits = row.index_select(dim=-1, index=coord_ids)
            coord_log_probs = F.log_softmax(coord_logits, dim=-1)
            selected = coord_log_probs.index_select(dim=-1, index=target_bins)
            return -(probabilities * selected).sum().to(dtype=torch.float32)
        if loss_mode == "w1_distance":
            coord_ids = _require_coord_token_ids(coord_token_ids, loss_mode=loss_mode)
            target_bins = _target_bins(
                token_ids,
                coord_token_ids=coord_ids,
            )
            coord_logits = row.index_select(dim=-1, index=coord_ids)
            pred_probs = F.softmax(coord_logits, dim=-1)
            target_probs = pred_probs.new_zeros(pred_probs.shape)
            target_probs.scatter_add_(dim=-1, index=target_bins, src=probabilities)
            cdf_delta = torch.cumsum(pred_probs - target_probs, dim=-1).abs()
            normalizer = max(int(coord_ids.numel()) - 1, 1)
            return (cdf_delta.sum() / float(normalizer)).to(dtype=torch.float32)

    raise ValueError(f"unsupported coordinate soft loss mode: {loss_mode!r}")


def _require_coord_token_ids(
    coord_token_ids: torch.Tensor | None,
    *,
    loss_mode: str,
) -> torch.Tensor:
    """Return the configured coordinate vocabulary required by a loss mode."""

    if coord_token_ids is None:
        raise ValueError(f"{loss_mode} requires config['coord_token_ids']")

    return coord_token_ids


def _target_bins(
    token_ids: torch.Tensor,
    *,
    coord_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return target coordinate bin indexes for configured coord token ids."""

    # map semantic token ids into ordered coordinate-vocabulary bin positions.
    bins: list[int] = []
    coord_ids = [int(token_id) for token_id in coord_token_ids.detach().cpu().tolist()]
    bin_by_token_id = {token_id: index for index, token_id in enumerate(coord_ids)}
    for token_id in token_ids.detach().cpu().tolist():
        token = int(token_id)
        try:
            bins.append(bin_by_token_id[token])
        except KeyError as exc:
            raise ValueError(
                f"coordinate target token id {token} is absent from coord_token_ids"
            ) from exc

    return torch.tensor(bins, device=token_ids.device, dtype=torch.long)


def _target_support(
    distribution: CoordinateSoftTokenDistribution,
    *,
    device: torch.device,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target token ids and normalized probabilities."""

    token_ids = torch.tensor(
        [entry.token_id for entry in distribution.token_weights],
        device=device,
        dtype=torch.long,
    )
    if bool(((token_ids < 0) | (token_ids >= vocab_size)).any().item()):
        raise ValueError("coordinate soft token id outside logits vocab")
    weights = torch.tensor(
        [entry.weight for entry in distribution.token_weights],
        device=device,
        dtype=torch.float32,
    )
    total = weights.sum()
    if float(total.detach().cpu().item()) <= 0.0:
        raise ValueError("coordinate soft target weights must have positive mass")

    return token_ids, weights / total.clamp(min=1e-12)


def _target_entropy(probabilities: torch.Tensor) -> float:
    """Return target entropy as a Python diagnostic scalar."""

    probs = probabilities.detach().to(dtype=torch.float32).cpu()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    if not math.isfinite(float(entropy)):
        return 0.0

    return float(entropy)


def _safe_normalize(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return numerator divided by denominator, or zero when empty."""

    if float(denominator.detach().cpu().item()) <= 0.0:
        return numerator * 0.0

    return (numerator / denominator.clamp(min=1e-6)).to(dtype=torch.float32)
