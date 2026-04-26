from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.metrics.reporter import SwiftMetricReporter


STAGE1_SET_CONTINUATION_METRIC_SCHEMA_VERSION = "stage1_set_continuation_metrics_v1"

LOGZ_ESTIMATOR_CODES = {
    "exact": 0.0,
    "sampled_raw": 1.0,
    "uniform_importance": 2.0,
}

CANDIDATE_SCORING_MODE_CODES = {
    "exact": 0.0,
    "uniform_subsample": 1.0,
}

PREFIX_ATTACH_MODE_CODES = {
    "repeated_forward": 0.0,
}

BRANCH_ISOLATION_CODES = {
    "independent_forward": 0.0,
}

PREFIX_GRADIENT_CODES = {
    "non_detached_recomputed_per_branch": 0.0,
}

BRANCH_RUNTIME_MODE_CODES = {
    "retained_graph": 0.0,
    "checkpointed_exact": 1.0,
    "smart_batched_exact": 2.0,
}

DDP_CANDIDATE_PADDING_POLICY_CODES = {
    "max_count": 0.0,
    "none": 1.0,
}

BRANCH_BATCH_SCHEDULER_CODES = {
    "disabled": 0.0,
    "constant_volume": 1.0,
    "deterministic_fallback": 2.0,
}


def metric_code(value: str, mapping: Mapping[str, float], *, metric_name: str) -> float:
    key = str(value or "").strip()
    if key not in mapping:
        raise ValueError(f"{metric_name} has no numeric code for value: {key!r}")
    return float(mapping[key])


def numeric_metric_payload(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep trainer custom metrics scalar-only.

    The provenance layer records string-valued fields such as candidate scoring
    mode; ms-swift's `custom_metrics` surface stores floats.
    """

    payload: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            payload[str(key)] = float(value)
            continue
        if isinstance(value, int | float):
            numeric = float(value)
            if math.isfinite(numeric):
                payload[str(key)] = numeric
    return payload


def mean_numeric_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        for key, value in numeric_metric_payload(row).items():
            buckets.setdefault(key, []).append(float(value))
    return {
        key: float(sum(values) / len(values))
        for key, values in sorted(buckets.items())
        if values
    }


def emit_stage1_set_continuation_metrics(
    trainer: Any,
    metrics: Mapping[str, Any],
) -> None:
    SwiftMetricReporter(trainer).update_many(numeric_metric_payload(metrics))


__all__ = [
    "BRANCH_ISOLATION_CODES",
    "BRANCH_BATCH_SCHEDULER_CODES",
    "BRANCH_RUNTIME_MODE_CODES",
    "CANDIDATE_SCORING_MODE_CODES",
    "DDP_CANDIDATE_PADDING_POLICY_CODES",
    "LOGZ_ESTIMATOR_CODES",
    "PREFIX_ATTACH_MODE_CODES",
    "PREFIX_GRADIENT_CODES",
    "STAGE1_SET_CONTINUATION_METRIC_SCHEMA_VERSION",
    "emit_stage1_set_continuation_metrics",
    "metric_code",
    "mean_numeric_metrics",
    "numeric_metric_payload",
]
