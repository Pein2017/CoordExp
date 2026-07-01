"""Stage-2 rollout-correction pipeline descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipelineIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage2RolloutCorrectionPipeline:
    """Descriptor for the Stage-2 rollout prefix plus GT correction pipeline."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Return the stable Stage-2 rollout-correction pipeline identity."""

        return TrainingPipelineIdentity(
            pipeline_id="stage2_rollout_correction",
            implementation_id="stage2_rollout_correction",
            lifecycle=PipelineLifecycle.ACTIVE,
            summary="Stage-2 rollout prefix plus GT correction pipeline.",
        )
