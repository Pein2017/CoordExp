"""Training-facing Qwen loading adapter.

The runtime implementation lives in :mod:`src.qwen.runtime_loading` so
inference can use the same owner code without importing training config types.
"""

from __future__ import annotations

from src.config.models import TrainConfig
from src.qwen.runtime_loading import (
    QwenComponents,
    QwenLoadOptions,
    QwenModelIdentity,
    QwenProcessorIdentity,
    load_qwen_components_from_options,
)


def load_qwen_components(
    config: TrainConfig,
    *,
    load_model: bool = False,
) -> QwenComponents:
    """Load Qwen components from a training config via neutral runtime options."""

    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=config.model.base_model,
            dtype=config.training.precision,
            attn_implementation=config.model.attn_implementation,
            patch_embed_linearization=config.model.runtime_patches.patch_embed_linearization,
            load_model=load_model,
        )
    )


__all__ = [
    "QwenComponents",
    "QwenLoadOptions",
    "QwenModelIdentity",
    "QwenProcessorIdentity",
    "load_qwen_components",
    "load_qwen_components_from_options",
]
