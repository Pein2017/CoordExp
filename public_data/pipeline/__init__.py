"""Unified public-data pipeline/factory internals.

This package centralizes dataset adapters, deterministic stage planning,
and shared artifact writing for public dataset preparation.
"""

from public_data.defaults import DEFAULT_NUM_WORKERS

from .planner import PipelinePlanner, PipelineResult
from .types import PipelineConfig

__all__ = ["DEFAULT_NUM_WORKERS", "PipelinePlanner", "PipelineResult", "PipelineConfig"]
