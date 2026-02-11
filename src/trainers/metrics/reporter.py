"""Compatibility shim: trainer metrics reporter now lives under `src.metrics`."""

from src.metrics.reporter import (
    SwiftMetricReporter,
    best_effort,
    best_effort_value,
    warn_once,
)

__all__ = [
    "SwiftMetricReporter",
    "best_effort",
    "best_effort_value",
    "warn_once",
]
