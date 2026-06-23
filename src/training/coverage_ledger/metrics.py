"""MetricEvent producers for the coverage-ledger auxiliary objective."""

from __future__ import annotations

import math

import torch

from src.metrics.events import MetricEvent, ratio_event, sum_event, weighted_mean_event
from src.training.coverage_ledger.loss import CoverageLedgerLossResult

COVERAGE_LEDGER_OBJECTIVE_ID = "coverage_ledger"
COVERAGE_LEDGER_STAGE = "teacher_forcing"
COVERAGE_LEDGER_SURFACE = "coverage_ledger_auxiliary"

WEIGHTED_LOSS_KEY = "teacher_forcing/loss/coverage_ledger_auxiliary_weighted"
COVERAGE_BCE_KEY = "teacher_forcing/ledger/coverage_bce"
REGION_ANCHOR_KEY = "teacher_forcing/ledger/region_anchor_positive"
COVERAGE_AUC_KEY = "teacher_forcing/ledger/coverage_auc"
COVERAGE_ACCURACY_KEY = "teacher_forcing/ledger/coverage_accuracy"
COVERAGE_STATE_COUNT_KEY = "teacher_forcing/ledger/coverage_state_count"
COVERAGE_PAIR_COUNT_KEY = "teacher_forcing/ledger/coverage_pair_count"
OBJECT_COUNT_KEY = "teacher_forcing/ledger/object_count"
REGION_ANCHOR_PAIR_COUNT_KEY = "teacher_forcing/ledger/region_anchor_pair_count"


def coverage_ledger_metric_events(
    result: CoverageLedgerLossResult,
) -> tuple[MetricEvent, ...]:
    """Return canonical typed events for one coverage-ledger loss result."""

    if not isinstance(result, CoverageLedgerLossResult):
        raise TypeError("result must be a CoverageLedgerLossResult")

    debug_rows = result.debug_rows
    events: list[MetricEvent] = []

    total_loss_count = (
        int(debug_rows.coverage_pair_count)
        + int(debug_rows.region_anchor_pair_count)
    )
    _append_weighted_mean(
        events,
        WEIGHTED_LOSS_KEY,
        result.weighted_loss,
        total_loss_count,
        field_name="weighted_loss",
        semantic_role="auxiliary_weighted_loss",
        diagnostic_only=False,
    )
    _append_weighted_mean(
        events,
        COVERAGE_BCE_KEY,
        result.coverage_loss,
        int(debug_rows.coverage_pair_count),
        field_name="coverage_loss",
        semantic_role="coverage_bce_loss",
        diagnostic_only=True,
    )
    _append_weighted_mean(
        events,
        REGION_ANCHOR_KEY,
        result.region_anchor_loss,
        int(debug_rows.region_anchor_pair_count),
        field_name="region_anchor_loss",
        semantic_role="region_anchor_positive_loss",
        diagnostic_only=True,
    )

    auc = _coverage_auc_event(result)
    if auc is not None:
        events.append(auc)
    accuracy = _coverage_accuracy_event(result)
    if accuracy is not None:
        events.append(accuracy)

    events.extend(
        (
            _count_event(
                COVERAGE_STATE_COUNT_KEY,
                int(debug_rows.coverage_state_count),
                unit="span",
                semantic_role="coverage_state_count",
            ),
            _count_event(
                COVERAGE_PAIR_COUNT_KEY,
                int(debug_rows.coverage_pair_count),
                unit="object",
                semantic_role="coverage_pair_count",
            ),
            _count_event(
                OBJECT_COUNT_KEY,
                int(debug_rows.object_count),
                unit="object",
                semantic_role="object_count",
            ),
            _count_event(
                REGION_ANCHOR_PAIR_COUNT_KEY,
                int(debug_rows.region_anchor_pair_count),
                unit="object",
                semantic_role="region_anchor_pair_count",
            ),
        )
    )
    return tuple(events)


def _append_weighted_mean(
    events: list[MetricEvent],
    key: str,
    value: torch.Tensor,
    weight: int,
    *,
    field_name: str,
    semantic_role: str,
    diagnostic_only: bool,
) -> None:
    if weight <= 0:
        return
    events.append(
        weighted_mean_event(
            key,
            _scalar(value, field_name=field_name),
            weight,
            unit="object",
            semantic_role=semantic_role,
            metric_surface=COVERAGE_LEDGER_SURFACE,
            stage=COVERAGE_LEDGER_STAGE,
            objective_id=COVERAGE_LEDGER_OBJECTIVE_ID,
            diagnostic_only=diagnostic_only,
        )
    )


def _coverage_auc_event(result: CoverageLedgerLossResult) -> MetricEvent | None:
    logits, targets = _coverage_logits_and_targets(result)
    if logits.numel() == 0:
        return None
    positive_logits = logits[targets >= 0.5]
    negative_logits = logits[targets < 0.5]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return None

    comparisons = positive_logits[:, None] - negative_logits[None, :]
    wins = (comparisons > 0).to(dtype=torch.float64).sum()
    ties = (comparisons == 0).to(dtype=torch.float64).sum()
    numerator = float((wins + 0.5 * ties).item())
    denominator = int(positive_logits.numel() * negative_logits.numel())
    if denominator <= 0:
        return None
    return ratio_event(
        COVERAGE_AUC_KEY,
        numerator,
        denominator,
        unit="object",
        semantic_role="coverage_auc",
        metric_surface=COVERAGE_LEDGER_SURFACE,
        stage=COVERAGE_LEDGER_STAGE,
        objective_id=COVERAGE_LEDGER_OBJECTIVE_ID,
        diagnostic_only=True,
    )


def _coverage_accuracy_event(result: CoverageLedgerLossResult) -> MetricEvent | None:
    logits, targets = _coverage_logits_and_targets(result)
    denominator = int(logits.numel())
    if denominator <= 0:
        return None
    predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=targets.dtype)
    expected = (targets >= 0.5).to(dtype=targets.dtype)
    correct = float((predictions == expected).to(dtype=torch.float64).sum().item())
    return ratio_event(
        COVERAGE_ACCURACY_KEY,
        correct,
        denominator,
        unit="object",
        semantic_role="coverage_accuracy",
        metric_surface=COVERAGE_LEDGER_SURFACE,
        stage=COVERAGE_LEDGER_STAGE,
        objective_id=COVERAGE_LEDGER_OBJECTIVE_ID,
        diagnostic_only=True,
    )


def _coverage_logits_and_targets(
    result: CoverageLedgerLossResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = result.debug_rows.coverage_logits.detach().reshape(-1).to(
        dtype=torch.float64,
        device="cpu",
    )
    targets = result.debug_rows.coverage_targets.detach().reshape(-1).to(
        dtype=torch.float64,
        device="cpu",
    )
    if int(logits.numel()) != int(targets.numel()):
        raise ValueError("coverage ledger logits and targets must have matching shape")
    return logits, targets


def _count_event(
    key: str,
    value: int,
    *,
    unit: str,
    semantic_role: str,
) -> MetricEvent:
    return sum_event(
        key,
        value,
        unit=unit,
        semantic_role=semantic_role,
        metric_surface=COVERAGE_LEDGER_SURFACE,
        stage=COVERAGE_LEDGER_STAGE,
        objective_id=COVERAGE_LEDGER_OBJECTIVE_ID,
        diagnostic_only=True,
    )


def _scalar(value: torch.Tensor, *, field_name: str) -> float:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if value.numel() != 1:
        raise ValueError(f"{field_name} must be scalar")
    scalar = float(value.detach().to(dtype=torch.float64, device="cpu").item())
    if not math.isfinite(scalar):
        raise FloatingPointError(f"coverage ledger {field_name} is non-finite")
    return scalar


__all__ = [
    "COVERAGE_ACCURACY_KEY",
    "COVERAGE_AUC_KEY",
    "COVERAGE_BCE_KEY",
    "COVERAGE_LEDGER_OBJECTIVE_ID",
    "COVERAGE_PAIR_COUNT_KEY",
    "COVERAGE_STATE_COUNT_KEY",
    "OBJECT_COUNT_KEY",
    "REGION_ANCHOR_KEY",
    "REGION_ANCHOR_PAIR_COUNT_KEY",
    "WEIGHTED_LOSS_KEY",
    "coverage_ledger_metric_events",
]
