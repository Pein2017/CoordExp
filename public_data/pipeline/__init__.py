"""Unified public-data pipeline/factory internals.

This package centralizes dataset adapters, deterministic stage planning,
and shared artifact writing for public dataset preparation.
"""

from .planner import PipelinePlanner, PipelineResult
from .types import PipelineConfig

__all__ = ["PipelinePlanner", "PipelineResult", "PipelineConfig"]
