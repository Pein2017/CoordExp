from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from src.metrics.events import MetricEvent, MetricIdentity

DiagnosticProfile: TypeAlias = Literal["off", "standard", "debug"]
DiagnosticPayloadValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["DiagnosticPayloadValue"]
    | dict[str, "DiagnosticPayloadValue"]
)


@dataclass(frozen=True)
class DiagnosticEvent:
    """Bounded non-metric diagnostic payload emitted by training observability."""

    key: str
    payload: dict[str, DiagnosticPayloadValue]
    profile: DiagnosticProfile
    truncated: bool = False


__all__ = [
    "DiagnosticEvent",
    "DiagnosticPayloadValue",
    "DiagnosticProfile",
    "MetricEvent",
    "MetricIdentity",
]
