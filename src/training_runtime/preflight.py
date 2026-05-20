"""Config-aware training runtime preflight contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from src.training_runtime.plan import (
    TrainingRuntimePlan,
    resolve_training_runtime_plan,
)

TEACHER_FORCING_EPOCH_VARYING_ROLLIN_BYPASS_REASON = (
    "teacher_forcing_epoch_varying_rollin"
)


@dataclass(frozen=True, slots=True)
class EncodedCachePreflightResult:
    """Encoded-cache eligibility diagnostics resolved from config and plan."""

    enabled: bool
    allowed: bool
    ineligible_policy: Literal["error", "bypass"]
    bypass_reason: str | None = None
    namespace: Literal["encoded_sample_cache", "encoded_cache"] | None = None


@dataclass(frozen=True, slots=True)
class TrainingRuntimePreflightResult:
    """Config-aware runtime preflight diagnostics."""

    runtime_plan: TrainingRuntimePlan
    encoded_cache: EncodedCachePreflightResult


def collect_training_runtime_preflight(
    config: Any,
    *,
    runtime_plan: TrainingRuntimePlan | None = None,
) -> TrainingRuntimePreflightResult:
    """Collect runtime preflight diagnostics without raising eligibility errors."""

    plan = runtime_plan or resolve_training_runtime_plan(
        _read_path(config, ("custom", "trainer_variant"))
    )
    encoded_cache_cfg = _encoded_cache_config(config)
    bypass_reason = None
    allowed = encoded_cache_cfg.enabled

    if encoded_cache_cfg.enabled and _is_epoch_varying_teacher_forcing_rollin(config):
        allowed = False
        bypass_reason = TEACHER_FORCING_EPOCH_VARYING_ROLLIN_BYPASS_REASON

    return TrainingRuntimePreflightResult(
        runtime_plan=plan,
        encoded_cache=EncodedCachePreflightResult(
            enabled=encoded_cache_cfg.enabled,
            allowed=allowed,
            ineligible_policy=encoded_cache_cfg.ineligible_policy,
            bypass_reason=bypass_reason,
            namespace=encoded_cache_cfg.namespace,
        ),
    )


def validate_training_runtime_preflight(
    config: Any,
    *,
    runtime_plan: TrainingRuntimePlan | None = None,
) -> TrainingRuntimePreflightResult:
    """Validate runtime preflight contracts and return collected diagnostics."""

    result = collect_training_runtime_preflight(
        config,
        runtime_plan=runtime_plan,
    )
    if (
        result.encoded_cache.enabled
        and not result.encoded_cache.allowed
        and result.encoded_cache.bypass_reason
        == TEACHER_FORCING_EPOCH_VARYING_ROLLIN_BYPASS_REASON
        and result.encoded_cache.ineligible_policy == "error"
    ):
        raise ValueError(
            "teacher_forcing encoded training cache is unsupported for v1 "
            "epoch-varying random roll-in because atom positions change across "
            "epochs; bypass_reason="
            f"{TEACHER_FORCING_EPOCH_VARYING_ROLLIN_BYPASS_REASON}"
        )
    return result


def _is_epoch_varying_teacher_forcing_rollin(config: Any) -> bool:
    objective_id = _read_path(config, ("objective", "id"))
    if objective_id != "teacher_forcing":
        return False
    rollin_policy = _read_path(
        config,
        ("objective", "target_ir", "rollin_policy", "name"),
    )
    return rollin_policy == "random_permutation"


@dataclass(frozen=True, slots=True)
class _EncodedCacheConfig:
    enabled: bool
    ineligible_policy: Literal["error", "bypass"]
    namespace: Literal["encoded_sample_cache", "encoded_cache"] | None


def _encoded_cache_config(config: Any) -> _EncodedCacheConfig:
    training = _read_path(config, ("training",))
    for field_name in ("encoded_sample_cache", "encoded_cache"):
        cache_cfg = _read_value(training, field_name)
        enabled = _read_value(cache_cfg, "enabled")
        if enabled is not None:
            return _EncodedCacheConfig(
                enabled=bool(enabled),
                ineligible_policy=_encoded_cache_ineligible_policy(cache_cfg),
                namespace=field_name,
            )
    return _EncodedCacheConfig(
        enabled=False,
        ineligible_policy="error",
        namespace=None,
    )


def _encoded_cache_ineligible_policy(
    cache_cfg: Any,
) -> Literal["error", "bypass"]:
    policy = _read_value(cache_cfg, "ineligible_policy")
    if policy is None or policy == "":
        return "error"
    normalized = str(policy).strip().lower()
    if normalized not in {"error", "bypass"}:
        raise ValueError(
            "training.encoded_sample_cache.ineligible_policy must be one of "
            "{'error', 'bypass'}"
        )
    if normalized == "error":
        return "error"
    return "bypass"


def _read_path(root: Any, path: tuple[str, ...]) -> Any:
    value = root
    for part in path:
        value = _read_value(value, part)
        if value is None:
            return None
    return value


def _read_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


__all__ = [
    "EncodedCachePreflightResult",
    "TEACHER_FORCING_EPOCH_VARYING_ROLLIN_BYPASS_REASON",
    "TrainingRuntimePreflightResult",
    "collect_training_runtime_preflight",
    "validate_training_runtime_preflight",
]
