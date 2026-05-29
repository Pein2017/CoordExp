"""Stage-1 JSON cross-entropy shadow pipeline descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipelineIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage1JsonCEPipeline:
    """Descriptor for the Stage-1 JSON CE training surface."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Return the stable Stage-1 JSON CE pipeline identity."""

        return TrainingPipelineIdentity(
            surface_id="stage1_json_ce",
            pipeline_id="stage1_json_ce",
            lifecycle=PipelineLifecycle.SHADOW,
            summary="Stage-1 JSON chat-template token CE shadow pipeline.",
        )
