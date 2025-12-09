"""Inference utilities for CoordExp.

This package hosts the centralized inference engine that supports both
coord-token checkpoints (normalized 0-999 outputs) and pure-text checkpoints
that emit pixel or normalized coordinates. The public surface is exposed via
``src.infer.engine`` and reused by CLI tools and visualizers.
"""

from .engine import GenerationConfig, InferenceConfig, InferenceEngine

__all__ = ["GenerationConfig", "InferenceConfig", "InferenceEngine"]
