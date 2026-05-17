"""Stage-2 two-channel shadow pipeline descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipelineIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage2TwoChannelPipeline:
    """Descriptor for the Stage-2 two-channel training surface."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Return the stable Stage-2 two-channel pipeline identity."""

        return TrainingPipelineIdentity(
            surface_id="stage2_two_channel",
            pipeline_id="stage2_two_channel",
            lifecycle=PipelineLifecycle.SHADOW,
            summary="Stage-2 two-channel shadow pipeline.",
        )
