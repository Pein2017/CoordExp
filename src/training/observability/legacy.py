from __future__ import annotations

from dataclasses import dataclass

from src.metrics.events import MetricReducer, MetricValue
from src.training.observability.contracts import (
    REMOVED_TRAINING_METRIC_KEYS,
    legacy_reducer_for_key,
)


@dataclass(frozen=True)
class LegacyMetricRecord:
    """Typed view of a tolerated historical flat metric key."""

    original_key: str
    value: MetricValue
    canonical_key: str
    reducer: MetricReducer
    legacy: bool = True
    removed: bool = False


def adapt_legacy_metric(key: str, value: MetricValue) -> LegacyMetricRecord:
    """Return a tolerant read record without authorizing new writer usage."""

    return LegacyMetricRecord(
        original_key=key,
        value=value,
        canonical_key=key,
        reducer=legacy_reducer_for_key(key),
        removed=key in REMOVED_TRAINING_METRIC_KEYS,
    )


__all__ = ["LegacyMetricRecord", "adapt_legacy_metric"]
