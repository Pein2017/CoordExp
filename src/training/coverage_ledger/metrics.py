"""MetricEvent producers for the coverage-ledger auxiliary objective."""

from __future__ import annotations

import math

import torch

from src.metrics.events import (
    MetricEvent,
    last_event,
    ratio_event,
    sum_event,
    weighted_mean_event,
)
from src.training.coverage_ledger.loss import CoverageLedgerLossResult

COVERAGE_LEDGER_OBJECTIVE_ID = "coverage_ledger"
COVERAGE_LEDGER_STAGE = "teacher_forcing"
COVERAGE_LEDGER_SURFACE = "coverage_ledger_auxiliary"

WEIGHTED_LOSS_KEY = "teacher_forcing/loss/coverage_ledger_auxiliary_weighted"
AUXILIARY_PAIR_NORMALIZED_KEY = (
    "teacher_forcing/ledger/coverage_ledger_auxiliary_pair_normalized"
)
COVERAGE_BCE_KEY = "teacher_forcing/ledger/coverage_bce"
ROW_OBJECT_BINDING_BCE_KEY = "teacher_forcing/ledger/row_object_binding_bce"
COVERAGE_AUC_KEY = "teacher_forcing/ledger/coverage_auc"
COVERAGE_ACCURACY_KEY = "teacher_forcing/ledger/coverage_accuracy"
ROW_OBJECT_BINDING_AUC_KEY = "teacher_forcing/ledger/row_object_binding_auc"
ROW_OBJECT_BINDING_ACCURACY_KEY = "teacher_forcing/ledger/row_object_binding_accuracy"
COVERAGE_STATE_COUNT_KEY = "teacher_forcing/ledger/coverage_state_count"
COVERAGE_PAIR_COUNT_KEY = "teacher_forcing/ledger/coverage_pair_count"
OBJECT_COUNT_KEY = "teacher_forcing/ledger/object_count"
ROW_OBJECT_BINDING_PAIR_COUNT_KEY = "teacher_forcing/ledger/row_object_binding_pair_count"


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
    if total_loss_count > 0:
        _append_weighted_auxiliary_loss(
            events,
            result,
        )
    _append_pair_normalized_auxiliary_loss(
        events,
        result,
        coverage_pair_count=int(debug_rows.coverage_pair_count),
        region_anchor_pair_count=int(debug_rows.region_anchor_pair_count),
        total_pair_count=total_loss_count,
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
        ROW_OBJECT_BINDING_BCE_KEY,
        result.region_anchor_loss,
        int(debug_rows.region_anchor_pair_count),
        field_name="region_anchor_loss",
        semantic_role="row_object_binding_bce_loss",
        diagnostic_only=True,
    )

    auc = _coverage_auc_event(result)
    if auc is not None:
        events.append(auc)
    accuracy = _coverage_accuracy_event(result)
    if accuracy is not None:
        events.append(accuracy)
    binding_auc = _row_object_binding_auc_event(result)
    if binding_auc is not None:
        events.append(binding_auc)
    binding_accuracy = _row_object_binding_accuracy_event(result)
    if binding_accuracy is not None:
        events.append(binding_accuracy)

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
                ROW_OBJECT_BINDING_PAIR_COUNT_KEY,
                int(debug_rows.region_anchor_pair_count),
                unit="object",
                semantic_role="row_object_binding_pair_count",
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


def _append_weighted_auxiliary_loss(
    events: list[MetricEvent],
    result: CoverageLedgerLossResult,
) -> None:
    events.append(
        last_event(
            WEIGHTED_LOSS_KEY,
            _scalar(result.weighted_loss, field_name="weighted_loss"),
            unit="batch",
            semantic_role="auxiliary_weighted_loss",
            metric_surface=COVERAGE_LEDGER_SURFACE,
            stage=COVERAGE_LEDGER_STAGE,
            objective_id=COVERAGE_LEDGER_OBJECTIVE_ID,
            diagnostic_only=False,
        )
    )


def _append_pair_normalized_auxiliary_loss(
    events: list[MetricEvent],
    result: CoverageLedgerLossResult,
    *,
    coverage_pair_count: int,
    region_anchor_pair_count: int,
    total_pair_count: int,
) -> None:
    if total_pair_count <= 0:
        return
    numerator = (
        _finite_float(result.coverage_weight, field_name="coverage_weight")
        * _scalar(result.coverage_loss, field_name="coverage_loss")
        * float(coverage_pair_count)
        + _finite_float(
            result.region_anchor_weight,
            field_name="region_anchor_weight",
        )
        * _scalar(result.region_anchor_loss, field_name="region_anchor_loss")
        * float(region_anchor_pair_count)
    )
    events.append(
        weighted_mean_event(
            AUXILIARY_PAIR_NORMALIZED_KEY,
            numerator / float(total_pair_count),
            total_pair_count,
            unit="object",
            semantic_role="auxiliary_weighted_loss_pair_normalized",
            metric_surface=COVERAGE_LEDGER_SURFACE,
            stage=COVERAGE_LEDGER_STAGE,
            objective_id=COVERAGE_LEDGER_OBJECTIVE_ID,
            diagnostic_only=True,
        )
    )


def _coverage_auc_event(result: CoverageLedgerLossResult) -> MetricEvent | None:
    logits, targets = _coverage_logits_and_targets(result)
    if logits.numel() == 0:
        return None
    auc_counts = _binary_auc_counts(logits, targets)
    if auc_counts is None:
        return None
    numerator, denominator = auc_counts
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


def _row_object_binding_auc_event(
    result: CoverageLedgerLossResult,
) -> MetricEvent | None:
    logits, targets = _row_object_binding_logits_and_targets(result)
    if logits.numel() == 0:
        return None
    auc_counts = _binary_auc_counts(logits, targets)
    if auc_counts is None:
        return None
    numerator, denominator = auc_counts
    return ratio_event(
        ROW_OBJECT_BINDING_AUC_KEY,
        numerator,
        denominator,
        unit="object",
        semantic_role="row_object_binding_auc",
        metric_surface=COVERAGE_LEDGER_SURFACE,
        stage=COVERAGE_LEDGER_STAGE,
        objective_id=COVERAGE_LEDGER_OBJECTIVE_ID,
        diagnostic_only=True,
    )


def _binary_auc_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, int] | None:
    positive_mask = targets >= 0.5
    positive_count = int(positive_mask.sum().item())
    total_count = int(positive_mask.numel())
    negative_count = total_count - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    order = torch.argsort(logits)
    sorted_logits = logits.index_select(0, order)
    sorted_positive = positive_mask.index_select(0, order)
    _, group_counts = torch.unique_consecutive(sorted_logits, return_counts=True)

    numerator = 0.0
    lower_negative_count = 0
    group_start = 0
    for group_count in group_counts.tolist():
        group_end = group_start + int(group_count)
        group_positive_count = int(
            sorted_positive[group_start:group_end].sum().item()
        )
        group_negative_count = int(group_count) - group_positive_count
        numerator += (
            group_positive_count * lower_negative_count
            + 0.5 * group_positive_count * group_negative_count
        )
        lower_negative_count += group_negative_count
        group_start = group_end

    return numerator, positive_count * negative_count


def _coverage_accuracy_event(result: CoverageLedgerLossResult) -> MetricEvent | None:
    logits, targets = _coverage_logits_and_targets(result)
    return _accuracy_event(
        key=COVERAGE_ACCURACY_KEY,
        semantic_role="coverage_accuracy",
        logits=logits,
        targets=targets,
    )


def _row_object_binding_accuracy_event(
    result: CoverageLedgerLossResult,
) -> MetricEvent | None:
    logits, targets = _row_object_binding_logits_and_targets(result)
    return _accuracy_event(
        key=ROW_OBJECT_BINDING_ACCURACY_KEY,
        semantic_role="row_object_binding_accuracy",
        logits=logits,
        targets=targets,
    )


def _accuracy_event(
    *,
    key: str,
    semantic_role: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> MetricEvent | None:
    denominator = int(logits.numel())
    if denominator <= 0:
        return None
    predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=targets.dtype)
    expected = (targets >= 0.5).to(dtype=targets.dtype)
    correct = float((predictions == expected).to(dtype=torch.float64).sum().item())
    return ratio_event(
        key,
        correct,
        denominator,
        unit="object",
        semantic_role=semantic_role,
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


def _row_object_binding_logits_and_targets(
    result: CoverageLedgerLossResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = result.debug_rows.region_anchor_logits.detach().reshape(-1).to(
        dtype=torch.float64,
        device="cpu",
    )
    targets = result.debug_rows.region_anchor_targets.detach().reshape(-1).to(
        dtype=torch.float64,
        device="cpu",
    )
    if int(logits.numel()) != int(targets.numel()):
        raise ValueError(
            "coverage ledger row-object binding logits and targets must have matching shape"
        )
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


def _finite_float(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{field_name} must be numeric")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(f"coverage ledger {field_name} is non-finite")
    return scalar


__all__ = [
    "COVERAGE_ACCURACY_KEY",
    "AUXILIARY_PAIR_NORMALIZED_KEY",
    "COVERAGE_AUC_KEY",
    "COVERAGE_BCE_KEY",
    "COVERAGE_LEDGER_OBJECTIVE_ID",
    "COVERAGE_PAIR_COUNT_KEY",
    "COVERAGE_STATE_COUNT_KEY",
    "OBJECT_COUNT_KEY",
    "ROW_OBJECT_BINDING_ACCURACY_KEY",
    "ROW_OBJECT_BINDING_AUC_KEY",
    "ROW_OBJECT_BINDING_BCE_KEY",
    "ROW_OBJECT_BINDING_PAIR_COUNT_KEY",
    "WEIGHTED_LOSS_KEY",
    "coverage_ledger_metric_events",
]
