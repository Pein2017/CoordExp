"""Compatibility shim for trainer-side metrics mixins.

The canonical implementation lives in `src.trainers.metrics.mixins`.

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
