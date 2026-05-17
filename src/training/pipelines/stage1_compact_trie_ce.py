"""Stage-1 compact-full trie CE shadow pipeline descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipelineIdentity,
)


@dataclass(frozen=True, slots=True)
class Stage1CompactTrieCEPipeline:
    """Descriptor for the Stage-1 compact-full trie CE training surface."""

    @property
    def identity(self) -> TrainingPipelineIdentity:
        """Return the stable Stage-1 compact trie CE pipeline identity."""

        return TrainingPipelineIdentity(
            surface_id="stage1_compact_trie_ce",
            pipeline_id="stage1_compact_trie_ce",
            lifecycle=PipelineLifecycle.SHADOW,
            summary="Stage-1 compact-full trie CE shadow pipeline.",
        )
