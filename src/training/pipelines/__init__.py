"""Shadow training pipeline descriptors."""

from src.training.pipelines.base import (
    PipelineLifecycle,
    TrainingPipeline,
    TrainingPipelineIdentity,
)
from src.training.pipelines.stage1_compact_trie_ce import Stage1CompactTrieCEPipeline
from src.training.pipelines.stage1_json_ce import Stage1JsonCEPipeline
from src.training.pipelines.stage2_two_channel import Stage2TwoChannelPipeline

__all__ = [
    "PipelineLifecycle",
    "Stage1CompactTrieCEPipeline",
    "Stage1JsonCEPipeline",
    "Stage2TwoChannelPipeline",
    "TrainingPipeline",
    "TrainingPipelineIdentity",
]
