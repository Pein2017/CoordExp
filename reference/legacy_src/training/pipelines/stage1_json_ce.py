"""Stage-1 standard SFT pipeline descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipelineIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage1JsonCEPipeline:
    """Descriptor for the Stage-1 standard SFT training pipeline."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Return the stable Stage-1 standard SFT pipeline identity."""

        return TrainingPipelineIdentity(
            pipeline_id="stage1_standard_sft",
            implementation_id="stage1_json_ce",
            lifecycle=PipelineLifecycle.ACTIVE,
            summary="Stage-1 standard SFT pipeline backed by JSON chat-template CE.",
        )
