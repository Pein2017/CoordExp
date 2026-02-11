from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TRAINER_METRICS_SCHEMA_VERSION = 1
_SUPPORTED_TRAINER_METRICS_SCHEMA_MAJORS = {1}


def _is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_metric_map(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("trainer_metrics.metrics must be a mapping of metric_name -> numeric value")

    out: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError("trainer_metrics.metrics keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"trainer_metrics.metrics[{key!r}] must be numeric; got {type(value).__name__}"
            )
        out[key] = float(value)
    return out


def validate_trainer_metrics_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the trainer-metrics payload contract.

    Contract requirements:
    - schema_version: integer major version (initial major 1)
    - mode: "train" | "eval"
    - global_step: integer step index
    - metrics: mapping[str, float]
    Optional sections are additive only and may be omitted.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("trainer_metrics payload must be a mapping")

    if "schema_version" not in payload:
        raise ValueError("trainer_metrics.schema_version is required")
    schema_version = payload.get("schema_version")
    if not _is_int_like(schema_version):
        raise ValueError("trainer_metrics.schema_version must be an integer major version")
    if int(schema_version) not in _SUPPORTED_TRAINER_METRICS_SCHEMA_MAJORS:
        raise ValueError(
            "Unsupported trainer_metrics.schema_version "
            f"{schema_version}; supported majors: {sorted(_SUPPORTED_TRAINER_METRICS_SCHEMA_MAJORS)}"
        )

    mode = payload.get("mode")
    if mode not in {"train", "eval"}:
        raise ValueError("trainer_metrics.mode must be 'train' or 'eval'")

    global_step = payload.get("global_step")
    if not _is_int_like(global_step):
        raise ValueError("trainer_metrics.global_step must be an integer")

    metrics = _coerce_metric_map(payload.get("metrics"))

    normalized: dict[str, Any] = {
        "schema_version": int(schema_version),
        "mode": str(mode),
        "global_step": int(global_step),
        "metrics": metrics,
    }

    for optional_key in ("batch_extras", "token_coord_summary"):
        if optional_key not in payload:
            continue
        optional_val = payload.get(optional_key)
        if optional_val is None:
            continue
        if not isinstance(optional_val, Mapping):
            raise ValueError(f"trainer_metrics.{optional_key} must be a mapping when provided")
        normalized[optional_key] = dict(optional_val)

    return normalized


def build_trainer_metrics_payload(
    *,
    mode: str,
    global_step: int,
    metrics: Mapping[str, float],
    schema_version: int = TRAINER_METRICS_SCHEMA_VERSION,
    batch_extras: Mapping[str, Any] | None = None,
    token_coord_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": int(schema_version),
        "mode": str(mode),
        "global_step": int(global_step),
        "metrics": dict(metrics),
    }
    if batch_extras is not None:
        payload["batch_extras"] = dict(batch_extras)
    if token_coord_summary is not None:
        payload["token_coord_summary"] = dict(token_coord_summary)
    return validate_trainer_metrics_payload(payload)
