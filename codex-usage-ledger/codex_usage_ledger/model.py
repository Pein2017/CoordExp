from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def _int(value: Any) -> int:
    """Coerce persisted numeric fields without allowing malformed data to crash a scan."""

    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "Usage":
        if not isinstance(value, Mapping):
            return cls()
        return cls(**{field: _int(value.get(field)) for field in USAGE_FIELDS})

    def subtract(self, baseline: "Usage") -> "Usage":
        return Usage(
            **{
                field: max(getattr(self, field) - getattr(baseline, field), 0)
                for field in USAGE_FIELDS
            }
        )

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in USAGE_FIELDS
            }
        )

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, field) == 0 for field in USAGE_FIELDS)

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in USAGE_FIELDS}


@dataclass(frozen=True)
class TurnContext:
    timestamp: str | None
    turn_id: str | None
    model: str | None
    effort: str | None
    multi_agent_version: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "timestamp": self.timestamp,
            "turn_id": self.turn_id,
            "model": self.model,
            "effort": self.effort,
            "multi_agent_version": self.multi_agent_version,
        }


@dataclass(frozen=True)
class TokenEvent:
    timestamp: str | None
    context_index: int
    total: Usage
    last: Usage
