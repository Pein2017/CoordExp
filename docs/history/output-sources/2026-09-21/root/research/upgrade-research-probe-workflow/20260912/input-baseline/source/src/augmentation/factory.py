"""Factory for split-scoped augmentation processors."""

from __future__ import annotations

from typing import Any

from src.augmentation.processor import (
    GeometryFlipAugmentationProcessor,
    NoopAugmentationProcessor,
)


def build_augmentation_processor(
    config: Any,
    *,
    split: str,
) -> GeometryFlipAugmentationProcessor | NoopAugmentationProcessor:
    runtime_seed = int(config.runtime.seed)
    if split != "train":
        return NoopAugmentationProcessor(runtime_seed=runtime_seed)
    geometry_flips = config.data.augmentation.train.geometry_flips
    if not geometry_flips.enabled:
        return NoopAugmentationProcessor(runtime_seed=runtime_seed)
    return GeometryFlipAugmentationProcessor(
        config=geometry_flips,
        runtime_seed=runtime_seed,
    )


__all__ = ["build_augmentation_processor"]
