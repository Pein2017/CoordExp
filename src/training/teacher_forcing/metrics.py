"""Minimal teacher-forcing objective metric helpers."""

from __future__ import annotations

from src.metrics.events import MetricEvent
from src.training.objectives.types import make_sum_event, make_weighted_mean_event


def teacher_forcing_loss_events(
    *,
    loss,
    denominator,
    span_count: int,
    atom_count: int,
) -> tuple[MetricEvent, ...]:
    """Return the minimal metric surface for the teacher-forcing objective."""

    return (
        make_weighted_mean_event(
            key="training/objectives/teacher_forcing/loss",
            value=loss,
            weight=denominator,
            objective_id="teacher_forcing",
        ),
        make_sum_event(
            key="training/objectives/teacher_forcing/span_count",
            value=span_count,
            objective_id="teacher_forcing",
            diagnostic_only=True,
        ),
        make_sum_event(
            key="training/objectives/teacher_forcing/atom_count",
            value=atom_count,
            objective_id="teacher_forcing",
            diagnostic_only=True,
        ),
    )
