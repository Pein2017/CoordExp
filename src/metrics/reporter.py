from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.metrics.payload_contract import (
    build_trainer_metrics_payload,
    validate_trainer_metrics_payload,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


_WARNED_ONCE_ATTR = "_coordexp_warned_once"
_DISABLED_DIAGNOSTICS_ATTR = "_coordexp_disabled_diagnostics"


def warn_once(trainer: Any, *, key: str, message: str, exc_info: bool = False) -> None:
    """Emit a warning once per trainer instance (best-effort)."""

    warned = getattr(trainer, _WARNED_ONCE_ATTR, None)
    if not isinstance(warned, set):
        warned = set()
        try:
            setattr(trainer, _WARNED_ONCE_ATTR, warned)
        except (AttributeError, TypeError):
            # If we can't stash state, fall back to always warning.
            logger.warning(message, exc_info=exc_info)
            return

    if key in warned:
        return
    warned.add(key)
    logger.warning(message, exc_info=exc_info)


def _is_disabled(trainer: Any, name: str) -> bool:
    disabled = getattr(trainer, _DISABLED_DIAGNOSTICS_ATTR, None)
    return isinstance(disabled, set) and name in disabled


def _disable(trainer: Any, name: str) -> None:
    disabled = getattr(trainer, _DISABLED_DIAGNOSTICS_ATTR, None)
    if not isinstance(disabled, set):
        disabled = set()
        try:
            setattr(trainer, _DISABLED_DIAGNOSTICS_ATTR, disabled)
        except (AttributeError, TypeError):
            return
    disabled.add(name)


def _resolve_global_step(trainer: Any) -> int:
    state = getattr(trainer, "state", None)
    step = getattr(state, "global_step", None)
    if isinstance(step, int) and not isinstance(step, bool):
        return int(step)

    step = getattr(trainer, "global_step", None)
    if isinstance(step, int) and not isinstance(step, bool):
        return int(step)

    return 0


def best_effort(
    trainer: Any,
    *,
    name: str,
    fn: Callable[[], None],
    disable_on_error: bool = True,
) -> None:
    """Run a diagnostics-only callback without blocking training.

    - Expected skips should be handled inside `fn` via pre-checks/early returns.
    - Unexpected exceptions are warned once, and the diagnostic may be disabled.
    """

    if _is_disabled(trainer, name):
        return

    try:
        fn()
    except Exception:
        warn_once(
            trainer,
            key=f"diagnostic_failed:{name}",
            message=(
                f"Diagnostic '{name}' failed (best-effort); skipping. "
                f"This diagnostic may be disabled for the remainder of the run."
            ),
            exc_info=True,
        )
        if disable_on_error:
            _disable(trainer, name)


def best_effort_value(
    trainer: Any,
    *,
    name: str,
    fn: Callable[[], Any],
    default: Any,
    disable_on_error: bool = True,
) -> Any:
    """Best-effort wrapper for diagnostics that return a value."""

    if _is_disabled(trainer, name):
        return default

    value = default
    try:
        value = fn()
    except Exception:
        warn_once(
            trainer,
            key=f"diagnostic_failed:{name}",
            message=(
                f"Diagnostic '{name}' failed (best-effort); skipping. "
                f"This diagnostic may be disabled for the remainder of the run."
            ),
            exc_info=True,
        )
        if disable_on_error:
            _disable(trainer, name)
        value = default

    return value


class SwiftMetricReporter:
    """Thin adapter around ms-swift `Trainer.custom_metrics`.

    Metric updates are first normalized through a schema-versioned neutral payload
    so metric logic is decoupled from trainer internals.
    """

    def __init__(self, trainer: Any):
        self._trainer = trainer

    def mode(self) -> str:
        model = getattr(self._trainer, "model", None)
        return "train" if model is None or bool(getattr(model, "training", True)) else "eval"

    def _metrics_dict_for_mode(self, mode: str) -> Any:
        custom_metrics = getattr(self._trainer, "custom_metrics", None)
        if custom_metrics is None or not isinstance(custom_metrics, dict):
            return None
        return custom_metrics.get(mode)

    def _metrics_dict(self) -> Any:
        return self._metrics_dict_for_mode(self.mode())

    def emit_payload(self, payload: Mapping[str, Any]) -> None:
        try:
            normalized = validate_trainer_metrics_payload(payload)
        except (TypeError, ValueError):
            warn_once(
                self._trainer,
                key="metric_payload_invalid",
                message="Invalid trainer-metrics payload (best-effort); skipping metric update.",
                exc_info=True,
            )
            return

        metrics = self._metrics_dict_for_mode(str(normalized["mode"]))
        if metrics is None:
            return

        for key, value in normalized["metrics"].items():
            try:
                metrics[key].update(float(value))
            except Exception:
                warn_once(
                    self._trainer,
                    key=f"metric_update_failed:{key}",
                    message=f"Failed to update metric '{key}' (best-effort).",
                    exc_info=True,
                )

    def update(self, key: str, value: float) -> None:
        self.update_many({key: float(value)})

    def update_many(self, updates: Mapping[str, float]) -> None:
        if not updates:
            return

        try:
            payload = build_trainer_metrics_payload(
                mode=self.mode(),
                global_step=_resolve_global_step(self._trainer),
                metrics=updates,
            )
        except (TypeError, ValueError):
            warn_once(
                self._trainer,
                key="metric_payload_build_failed",
                message=(
                    "Failed to build trainer-metrics payload (best-effort); "
                    "skipping metric update."
                ),
                exc_info=True,
            )
            return

        self.emit_payload(payload)
