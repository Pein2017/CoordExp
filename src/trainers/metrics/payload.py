"""Trainer-metrics payload contracts.

This module defines a neutral, versioned payload used to move step-level metrics
between trainer surfaces and metric reporters.

Contract (v1):
- schema_version: int major version (initial 1)
- mode: "train" | "eval"
- global_step: optimizer-step index
- metrics: mapping[str, float]

Within a major version, the contract is additive-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping

TRAINER_METRICS_SCHEMA_VERSION = 1

TrainerMetricsMode = Literal["train", "eval"]


@dataclass(frozen=True)
class TrainerMetricsPayload:
    schema_version: int
    mode: TrainerMetricsMode
    global_step: int
    metrics: Dict[str, float]


def parse_trainer_metrics_payload(payload: Mapping[str, Any]) -> TrainerMetricsPayload:
    if not isinstance(payload, Mapping):
        raise TypeError("trainer-metrics payload must be a mapping")

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError(
            "trainer-metrics payload schema_version must be an int major version"
        )
    if schema_version != TRAINER_METRICS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trainer-metrics payload schema_version={schema_version}; "
            f"supported={TRAINER_METRICS_SCHEMA_VERSION}"
        )

    mode = payload.get("mode")
    if mode not in ("train", "eval"):
        raise ValueError(
            f"trainer-metrics payload mode must be 'train' or 'eval'; got {mode!r}"
        )

    global_step = payload.get("global_step")
    if not isinstance(global_step, int):
        raise ValueError("trainer-metrics payload global_step must be an int")
    if global_step < 0:
        raise ValueError("trainer-metrics payload global_step must be >= 0")

    metrics_raw = payload.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        raise ValueError("trainer-metrics payload metrics must be a mapping")

    metrics: Dict[str, float] = {}
    for k, v in metrics_raw.items():
        key = str(k)
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(
                f"trainer-metrics payload metric '{key}' must be numeric"
            ) from exc
        if not math.isfinite(fv):
            raise ValueError(
                f"trainer-metrics payload metric '{key}' must be finite; got {fv}"
            )
        metrics[key] = float(fv)

    return TrainerMetricsPayload(
        schema_version=int(schema_version),
        mode=mode,
        global_step=int(global_step),
        metrics=metrics,
    )


def build_trainer_metrics_payload(
    *,
    mode: TrainerMetricsMode,
    global_step: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": TRAINER_METRICS_SCHEMA_VERSION,
        "mode": str(mode),
        "global_step": int(global_step),
        "metrics": {str(k): float(v) for k, v in metrics.items()},
    }

    # Validate upfront so producers and consumers share strict behavior.
    _ = parse_trainer_metrics_payload(payload)
    return payload
