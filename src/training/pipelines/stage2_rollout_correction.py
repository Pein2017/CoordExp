"""Stage-2 rollout-correction shadow pipeline descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipelineIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage2RolloutCorrectionPipeline:
    """Descriptor for the Stage-2 rollout prefix plus GT correction surface."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Return the stable Stage-2 rollout-correction pipeline identity."""

        return TrainingPipelineIdentity(
            surface_id="stage2_rollout_correction",
            pipeline_id="stage2_rollout_correction",
            lifecycle=PipelineLifecycle.SHADOW,
            summary="Stage-2 rollout prefix plus GT correction shadow pipeline.",
        )
