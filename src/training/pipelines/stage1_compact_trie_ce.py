"""Stage-1 research teacher-forcing pipeline descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipelineIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage1CompactTrieCEPipeline:
    """Descriptor for the Stage-1 research teacher-forcing pipeline."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Return the stable Stage-1 research teacher-forcing pipeline identity."""

        return TrainingPipelineIdentity(
            pipeline_id="stage1_research_teacher_forcing",
            implementation_id="stage1_compact_trie_ce",
            lifecycle=PipelineLifecycle.ACTIVE,
            summary=(
                "Stage-1 research teacher-forcing pipeline backed by compact-full "
                "trie/coordinate objectives."
            ),
        )
