"""Teacher-forcing objective metric helpers and compact summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.metrics.events import (
    MetricEvent,
    last_event,
    ratio_event,
    reduce_metric_events,
    sum_event,
    weighted_mean_event,
)

TEACHER_FORCING_OBJECTIVE_ID = "teacher_forcing"

TEACHER_FORCING_BUILDER_REJECTION_PREFIX = (
    "teacher_forcing/builder/rejection_reason"
)
COMPACT_FULL_PARSE_ERROR_PREFIX = "infer/parse/compact_full/error"

REQUIRED_TEACHER_FORCING_METRIC_KEYS: tuple[str, ...] = (
    "teacher_forcing/loss/total",
    "teacher_forcing/loss/token_type_mass",
    "teacher_forcing/loss/conditional_valid_set_likelihood",
    "teacher_forcing/loss/within_valid_coverage",
    "teacher_forcing/valid_set/mass",
    "teacher_forcing/coverage/kl",
    "teacher_forcing/continuation/eos_margin",
    "teacher_forcing/ambiguity/coordinate_onset_count",
    "teacher_forcing/ambiguity/mixed_role_count",
    "teacher_forcing/builder/rejected_samples",
    "teacher_forcing/builder/rejection_reason/<code>",
    "teacher_forcing/branch_coherence/rate",
    "teacher_forcing/residual_set/remaining_count",
    "teacher_forcing/permutation_probe/nll_std",
    "teacher_forcing/decode/object_coherence_rate",
    "teacher_forcing/decode/duplicate_rate",
    "teacher_forcing/decode/missed_object_rate",
    "teacher_forcing/decode/malformed_sequence_rate",
    "infer/parse/compact_full/error/<code>",
)


def teacher_forcing_loss_events(
    *,
    loss: float,
    denominator: float,
    span_count: int,
    atom_count: int,
) -> tuple[MetricEvent, ...]:
    """Return canonical metric events for the teacher-forcing objective loss."""

    return (
        _weighted_mean(
            "teacher_forcing/loss/total",
            _numeric_scalar(loss, field_name="loss"),
            _numeric_scalar(denominator, field_name="denominator"),
            unit="token",
            metric_surface="objective_loss",
        ),
        _sum(
            "teacher_forcing/loss/span_count",
            span_count,
            unit="span",
            metric_surface="objective_loss",
            diagnostic_only=True,
        ),
        _sum(
            "teacher_forcing/loss/atom_count",
            atom_count,
            unit="token",
            metric_surface="objective_loss",
            diagnostic_only=True,
        ),
    )


def teacher_forcing_diagnostic_events(
    *,
    token_type_mass: float | None = None,
    conditional_valid_set_likelihood: float | None = None,
    within_valid_coverage: float | None = None,
    valid_set_mass: float | None = None,
    coverage_kl: float | None = None,
    eos_margin: float | None = None,
    coordinate_onset_count: int | None = None,
    mixed_role_count: int | None = None,
    branch_coherent: int | None = None,
    branch_total: int | None = None,
    residual_remaining_count: int | None = None,
    permutation_nll_std: float | None = None,
) -> tuple[MetricEvent, ...]:
    """Return optional teacher-forcing diagnostic metric events."""

    events: list[MetricEvent] = []
    _append_weighted(events, "teacher_forcing/loss/token_type_mass", token_type_mass)
    _append_weighted(
        events,
        "teacher_forcing/loss/conditional_valid_set_likelihood",
        conditional_valid_set_likelihood,
    )
    _append_weighted(
        events,
        "teacher_forcing/loss/within_valid_coverage",
        within_valid_coverage,
    )
    _append_weighted(events, "teacher_forcing/valid_set/mass", valid_set_mass)
    _append_weighted(events, "teacher_forcing/coverage/kl", coverage_kl)
    _append_weighted(events, "teacher_forcing/continuation/eos_margin", eos_margin)
    _append_sum(
        events,
        "teacher_forcing/ambiguity/coordinate_onset_count",
        coordinate_onset_count,
        unit="token",
    )
    _append_sum(
        events,
        "teacher_forcing/ambiguity/mixed_role_count",
        mixed_role_count,
        unit="token",
    )
    if (branch_coherent is None) != (branch_total is None):
        raise ValueError(
            "branch_coherent and branch_total must be provided together"
        )
    if branch_coherent is not None and branch_total is not None:
        events.append(
            _ratio(
                "teacher_forcing/branch_coherence/rate",
                branch_coherent,
                branch_total,
                unit="sample",
                metric_surface="branch_coherence",
            )
        )
    _append_sum(
        events,
        "teacher_forcing/residual_set/remaining_count",
        residual_remaining_count,
        unit="token",
    )
    _append_weighted(
        events,
        "teacher_forcing/permutation_probe/nll_std",
        permutation_nll_std,
        metric_surface="permutation_probe",
    )
    return tuple(events)


def builder_rejection_events(reason_counts: Mapping[str, int]) -> tuple[MetricEvent, ...]:
    """Return builder rejection count events while preserving exact reason codes."""

    normalized = _normalize_count_codes(
        reason_counts,
        family="teacher_forcing/builder/rejection_reason",
    )
    total = sum(normalized.values())
    events = [
        _sum(
            "teacher_forcing/builder/rejected_samples",
            total,
            unit="sample",
            metric_surface="target_realizer",
        )
    ]
    for reason, count in sorted(normalized.items()):
        events.append(
            _sum(
                f"{TEACHER_FORCING_BUILDER_REJECTION_PREFIX}/{reason}",
                count,
                unit="sample",
                metric_surface="target_realizer",
            )
        )
    return tuple(events)


def decode_quality_events(
    *,
    artifact_present: bool,
    object_coherent: int,
    duplicate: int,
    missed_object: int,
    malformed_sequence: int,
    total: int,
) -> tuple[MetricEvent, ...]:
    """Return decode-quality rates only when generation artifacts are present."""

    if not artifact_present:
        return (
            last_event(
                "teacher_forcing/decode/artifact_status",
                "absent",
                unit="sample",
                objective_id=TEACHER_FORCING_OBJECTIVE_ID,
                metric_surface="decode_generation",
                diagnostic_only=True,
            ),
        )

    return (
        _ratio(
            "teacher_forcing/decode/object_coherence_rate",
            object_coherent,
            total,
            unit="object",
            metric_surface="decode_generation",
        ),
        _ratio(
            "teacher_forcing/decode/duplicate_rate",
            duplicate,
            total,
            unit="object",
            metric_surface="decode_generation",
        ),
        _ratio(
            "teacher_forcing/decode/missed_object_rate",
            missed_object,
            total,
            unit="object",
            metric_surface="decode_generation",
        ),
        _ratio(
            "teacher_forcing/decode/malformed_sequence_rate",
            malformed_sequence,
            total,
            unit="sample",
            metric_surface="decode_generation",
        ),
    )


def compact_full_parse_error_events(
    error_counts: Mapping[str, int],
) -> tuple[MetricEvent, ...]:
    """Return compact-full parse error events keyed by exact parser error code."""

    normalized = _normalize_count_codes(
        error_counts,
        family="infer/parse/compact_full/error",
    )
    events: list[MetricEvent] = []
    for code, count_int in sorted(normalized.items()):
        events.append(
            _sum(
                f"{COMPACT_FULL_PARSE_ERROR_PREFIX}/{code}",
                count_int,
                unit="sample",
                parser_mode="compact_full",
                metric_surface="inference_parse",
                diagnostic_only=True,
            )
        )
    return tuple(events)


def summarize_metric_events(events: Iterable[MetricEvent]) -> dict[str, Any]:
    """Reduce events into a JSON-friendly diagnostic summary.

    String-valued status events stay under ``artifact_status`` so trainer flat
    logs do not fabricate numeric zero rates when optional artifacts are absent.
    """

    reduced = reduce_metric_events(tuple(events))
    metrics: dict[str, float] = {}
    artifact_status: dict[str, str] = {}
    for key, value in reduced.items():
        if key.endswith("/artifact_status"):
            owner = key.removesuffix("/artifact_status")
            artifact_status[owner] = str(value)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[key] = float(value)
    return {"metrics": metrics, "artifact_status": artifact_status}


def _append_weighted(
    events: list[MetricEvent],
    key: str,
    value: float | None,
    *,
    metric_surface: str = "objective_diagnostic",
) -> None:
    if value is None:
        return
    events.append(
        _weighted_mean(
            key,
            value,
            1.0,
            unit="token",
            metric_surface=metric_surface,
            diagnostic_only=True,
        )
    )


def _append_sum(
    events: list[MetricEvent],
    key: str,
    value: int | None,
    *,
    unit: str,
) -> None:
    if value is None:
        return
    events.append(
        _sum(
            key,
            value,
            unit=unit,
            metric_surface="objective_diagnostic",
            diagnostic_only=True,
        )
    )


def _weighted_mean(
    key: str,
    value: float,
    weight: float,
    *,
    unit: str,
    metric_surface: str,
    diagnostic_only: bool = False,
) -> MetricEvent:
    return weighted_mean_event(
        key,
        value,
        weight,
        unit=unit,
        objective_id=TEACHER_FORCING_OBJECTIVE_ID,
        metric_surface=metric_surface,
        diagnostic_only=diagnostic_only,
    )


def _ratio(
    key: str,
    numerator: float,
    denominator: float,
    *,
    unit: str,
    metric_surface: str,
    diagnostic_only: bool = True,
) -> MetricEvent:
    return ratio_event(
        key,
        numerator,
        denominator,
        unit=unit,
        objective_id=TEACHER_FORCING_OBJECTIVE_ID,
        metric_surface=metric_surface,
        diagnostic_only=diagnostic_only,
    )


def _sum(
    key: str,
    value: float,
    *,
    unit: str,
    metric_surface: str,
    parser_mode: str | None = None,
    diagnostic_only: bool = True,
) -> MetricEvent:
    return sum_event(
        key,
        value,
        unit=unit,
        objective_id=TEACHER_FORCING_OBJECTIVE_ID,
        parser_mode=parser_mode,
        metric_surface=metric_surface,
        diagnostic_only=diagnostic_only,
    )


def _normalize_code(code: object) -> str:
    text = str(code).strip()
    if not text:
        raise ValueError("metric reason/error code must be non-empty")
    return text.replace("/", "_")


def _normalize_count_codes(
    counts: Mapping[str, int],
    *,
    family: str,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    source_by_normalized: dict[str, str] = {}
    for source_code, raw_count in counts.items():
        count = int(raw_count)
        if count == 0:
            continue
        normalized_code = _normalize_code(source_code)
        previous_source = source_by_normalized.get(normalized_code)
        if previous_source is not None and previous_source != str(source_code):
            raise ValueError(
                f"{family} code collision after normalization: "
                f"{previous_source!r} and {str(source_code)!r} both map to "
                f"{normalized_code!r}"
            )
        source_by_normalized[normalized_code] = str(source_code)
        normalized[normalized_code] = normalized.get(normalized_code, 0) + count
    return normalized


def _numeric_scalar(value: Any, *, field_name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    return float(value)


__all__ = [
    "COMPACT_FULL_PARSE_ERROR_PREFIX",
    "REQUIRED_TEACHER_FORCING_METRIC_KEYS",
    "TEACHER_FORCING_BUILDER_REJECTION_PREFIX",
    "builder_rejection_events",
    "compact_full_parse_error_events",
    "decode_quality_events",
    "summarize_metric_events",
    "teacher_forcing_diagnostic_events",
    "teacher_forcing_loss_events",
]
