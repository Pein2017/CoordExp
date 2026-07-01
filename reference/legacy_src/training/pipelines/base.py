"""Lightweight training pipeline descriptor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class PipelineLifecycle(str, Enum):
    """Lifecycle state for a training pipeline descriptor."""

    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class TrainingPipelineIdentity:
    """Stable identity for a selectable training pipeline.

    :param pipeline_id: Public training pipeline identifier selected by config.
    :param implementation_id: Concrete descriptor or implementation handle.
    :param lifecycle: Migration lifecycle for the pipeline descriptor.
    :param summary: Human-readable one-line descriptor.
    """

    pipeline_id: str
    implementation_id: str
    lifecycle: PipelineLifecycle
    summary: str

    def __post_init__(self) -> None:
        """Validate pipeline identity fields."""

        # validate stable string identity.
        for field_name in ("pipeline_id", "implementation_id", "summary"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

        # validate lifecycle enum normalization.
        if not isinstance(self.lifecycle, PipelineLifecycle):
            object.__setattr__(
                self,
                "lifecycle",
                PipelineLifecycle(str(self.lifecycle)),
            )


@runtime_checkable
class TrainingPipeline(Protocol):
    """Minimal protocol implemented by shadow pipeline descriptors."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Stable identity for resolver and artifact metadata."""

        ...
