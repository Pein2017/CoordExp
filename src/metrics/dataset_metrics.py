"""Compatibility shim for trainer-side metrics mixins.

This module re-exports the active implementation from `src.trainers.metrics.mixins`
while preserving the historical import path.

This module is kept to avoid breaking older imports:
  `from src.metrics.dataset_metrics import ...`
"""

from __future__ import annotations

from src.trainers.metrics.mixins import (  # noqa: F401
    AggregateTokenTypeMetricsMixin,
    CoordSoftCEW1LossMixin,
    GradAccumLossScaleMixin,
    InstabilityMonitorMixin,
)

__all__ = [
    "AggregateTokenTypeMetricsMixin",
    "CoordSoftCEW1LossMixin",
    "GradAccumLossScaleMixin",
    "InstabilityMonitorMixin",
]
