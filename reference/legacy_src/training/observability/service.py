from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.metrics.events import (
    MetricEvent,
    MetricUnit,
    flatten_metric_events,
    sum_event,
    weighted_mean_event,
)
from src.training.observability.contracts import REMOVED_TRAINING_METRIC_KEYS
from src.training.observability.events import (
    DiagnosticEvent,
    DiagnosticPayloadValue,
    DiagnosticProfile,
)

_SCALAR_DIAGNOSTIC_TYPES = (str, int, float, bool, type(None))
_TRUNCATED_SENTINEL = "<truncated>"
_CYCLE_SENTINEL = "<cycle>"
_TRUNCATION_SUFFIX = "..."


class ObservabilityService:
    """Factory and publication boundary for canonical training observability."""

    def __init__(
        self,
        *,
        diagnostic_profile: DiagnosticProfile = "standard",
        max_diagnostic_entries: int = 16,
        max_diagnostic_items: int = 16,
        max_diagnostic_depth: int = 4,
        max_diagnostic_scalar_chars: int = 4096,
    ) -> None:
        if diagnostic_profile not in {"off", "standard", "debug"}:
            raise ValueError(f"Unsupported diagnostic profile: {diagnostic_profile!r}")
        if max_diagnostic_entries < 0:
            raise ValueError("max_diagnostic_entries must be >= 0")
        if max_diagnostic_items < 0:
            raise ValueError("max_diagnostic_items must be >= 0")
        if max_diagnostic_depth < 0:
            raise ValueError("max_diagnostic_depth must be >= 0")
        if max_diagnostic_scalar_chars < 0:
            raise ValueError("max_diagnostic_scalar_chars must be >= 0")

        self._diagnostic_profile = diagnostic_profile
        self._max_diagnostic_entries = max_diagnostic_entries
        self._max_diagnostic_items = max_diagnostic_items
        self._max_diagnostic_depth = max_diagnostic_depth
        self._max_diagnostic_scalar_chars = max_diagnostic_scalar_chars

    def flatten_for_ms_swift(self, events: Iterable[MetricEvent]) -> dict[str, float]:
        """Reduce canonical metric events into flat trainer reporting keys."""

        return flatten_metric_events(events)

    def objective_metric(
        self,
        *,
        key: str,
        value: float | int,
        weight: float | int,
        stage: str,
        channel: str | None,
        objective_id: str,
        unit: MetricUnit,
        provenance: str,
    ) -> MetricEvent:
        """Create an objective-local weighted-mean metric event."""

        self._reject_removed_writer_key(key)
        return weighted_mean_event(
            key,
            value,
            weight,
            unit=unit,
            stage=stage,
            channel=channel,
            objective_id=objective_id,
            provenance=provenance,
        )

    def duplicate_count(
        self,
        key: str,
        value: float | int,
        *,
        stage: str,
        channel: str | None,
        provenance: str,
    ) -> MetricEvent:
        """Create a duplicate diagnostic count that aggregates by summation."""

        self._reject_removed_writer_key(key)
        return sum_event(
            key,
            value,
            unit="object",
            stage=stage,
            channel=channel,
            provenance=provenance,
            diagnostic_only=True,
        )

    def duplicate_gauge(
        self,
        key: str,
        *,
        value: float | int,
        weight: float | int,
        stage: str,
        channel: str | None,
        provenance: str,
    ) -> MetricEvent:
        """Create a duplicate diagnostic gauge as an explicitly weighted mean."""

        self._reject_removed_writer_key(key)
        return weighted_mean_event(
            key,
            value,
            weight,
            unit="object",
            stage=stage,
            channel=channel,
            provenance=provenance,
            diagnostic_only=True,
        )

    def diagnostic_event(
        self,
        key: str,
        payload: Mapping[str, object],
    ) -> DiagnosticEvent | None:
        """Return a bounded diagnostic event according to the active profile."""

        if self._diagnostic_profile == "off":
            return None

        if self._diagnostic_profile == "standard":
            bounded_payload, truncated = self._standard_payload(payload)
        else:
            bounded_payload, truncated = self._debug_payload(payload)

        return DiagnosticEvent(
            key=key,
            payload=bounded_payload,
            profile=self._diagnostic_profile,
            truncated=truncated,
        )

    def _reject_removed_writer_key(self, key: str) -> None:
        if key in REMOVED_TRAINING_METRIC_KEYS:
            raise ValueError(f"Metric writer key uses a removed training mechanism: {key}")

    def _standard_payload(
        self,
        payload: Mapping[str, object],
    ) -> tuple[dict[str, DiagnosticPayloadValue], bool]:
        bounded: dict[str, DiagnosticPayloadValue] = {}
        truncated = len(payload) > self._max_diagnostic_entries

        for key, value in payload.items():
            if len(bounded) >= self._max_diagnostic_entries:
                truncated = True
                break
            if isinstance(value, _SCALAR_DIAGNOSTIC_TYPES):
                bounded_value, value_truncated = self._bound_scalar_value(value)
                bounded[key] = bounded_value
                truncated = truncated or value_truncated
            else:
                truncated = True

        return bounded, truncated

    def _debug_payload(
        self,
        payload: Mapping[str, object],
    ) -> tuple[dict[str, DiagnosticPayloadValue], bool]:
        bounded: dict[str, DiagnosticPayloadValue] = {}
        truncated = len(payload) > self._max_diagnostic_entries

        for key, value in payload.items():
            if len(bounded) >= self._max_diagnostic_entries:
                truncated = True
                break
            bounded_value, value_truncated = self._bound_debug_value(
                value,
                depth=1,
                seen=set(),
            )
            bounded[key] = bounded_value
            truncated = truncated or value_truncated

        return bounded, truncated

    def _bound_debug_value(
        self,
        value: object,
        *,
        depth: int,
        seen: set[int],
    ) -> tuple[DiagnosticPayloadValue, bool]:
        if isinstance(value, _SCALAR_DIAGNOSTIC_TYPES):
            return self._bound_scalar_value(value)
        if depth >= self._max_diagnostic_depth:
            return _TRUNCATED_SENTINEL, True

        value_id = id(value)
        if value_id in seen:
            return _CYCLE_SENTINEL, True
        next_seen = set(seen)
        next_seen.add(value_id)

        if isinstance(value, Mapping):
            bounded: dict[str, DiagnosticPayloadValue] = {}
            truncated = len(value) > self._max_diagnostic_items
            for nested_key, nested_value in value.items():
                if len(bounded) >= self._max_diagnostic_items:
                    truncated = True
                    break
                bounded_value, value_truncated = self._bound_debug_value(
                    nested_value,
                    depth=depth + 1,
                    seen=next_seen,
                )
                bounded[str(nested_key)] = bounded_value
                truncated = truncated or value_truncated
            return bounded, truncated
        if isinstance(value, (list, tuple)):
            bounded_items: list[DiagnosticPayloadValue] = []
            truncated = len(value) > self._max_diagnostic_items
            for item in value[: self._max_diagnostic_items]:
                bounded_item, item_truncated = self._bound_debug_value(
                    item,
                    depth=depth + 1,
                    seen=next_seen,
                )
                bounded_items.append(bounded_item)
                truncated = truncated or item_truncated
            return bounded_items, truncated

        bounded_repr, _ = self._truncate_text(repr(value))
        return bounded_repr, True

    def _bound_scalar_value(
        self,
        value: object,
    ) -> tuple[DiagnosticPayloadValue, bool]:
        if isinstance(value, str):
            return self._truncate_text(value)

        return value, False

    def _truncate_text(self, value: str) -> tuple[str, bool]:
        if len(value) <= self._max_diagnostic_scalar_chars:
            return value, False

        return (
            value[: self._max_diagnostic_scalar_chars] + _TRUNCATION_SUFFIX,
            True,
        )


__all__ = ["ObservabilityService"]
