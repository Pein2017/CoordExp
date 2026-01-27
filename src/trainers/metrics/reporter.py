from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
        except Exception:
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
        except Exception:
            return
    disabled.add(name)


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
    """Best-effort wrapper for diagnostics that return a value.

    This is useful for monitors that may choose to replace the loss (guard), but
    should never block training when they fail.
    """

    if _is_disabled(trainer, name):
        return default

    try:
        return fn()
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
        return default


class SwiftMetricReporter:
    """Thin adapter around ms-swift `Trainer.custom_metrics`.

    This intentionally stays ms-swift specific; it's a glue layer so metric computation
    can be pure (dict[str, float]) and logging remains best-effort.
    """

    def __init__(self, trainer: Any):
        self._trainer = trainer

    def mode(self) -> str:
        model = getattr(self._trainer, "model", None)
        return "train" if model is None or bool(getattr(model, "training", True)) else "eval"

    def _metrics_dict(self) -> Any:
        custom_metrics = getattr(self._trainer, "custom_metrics", None)
        if custom_metrics is None:
            return None
        mode = self.mode()
        return custom_metrics.get(mode) if isinstance(custom_metrics, dict) else None

    def update(self, key: str, value: float) -> None:
        metrics = self._metrics_dict()
        if metrics is None:
            return
        try:
            metrics[key].update(float(value))
        except Exception:
            warn_once(
                self._trainer,
                key=f"metric_update_failed:{key}",
                message=f"Failed to update metric '{key}' (best-effort).",
                exc_info=True,
            )

    def update_many(self, updates: dict[str, float]) -> None:
        for k, v in updates.items():
            self.update(k, v)
