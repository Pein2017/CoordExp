"""Shared dataclasses for fusion datasets.

These dataclasses are intentionally compatible with the upstream Qwen3-VL
fusion config shapes, but CoordExp may choose simpler semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

FALLBACK_OPTIONS = ("off", "bbox_2d")


@dataclass(frozen=True)
class DatasetSpec:
    """Normalized dataset description used by fusion builders."""

    key: str
    name: str
    train_jsonl: Path
    template: str
    domain: Literal["target", "source"]
    supports_augmentation: bool
    supports_curriculum: bool
    # Optional explicit mode; defaults applied elsewhere when None.
    mode: Optional[Literal["dense", "summary"]] = None
    poly_fallback: Literal["off", "bbox_2d"] = "off"
    poly_max_points: Optional[int] = None
    val_jsonl: Optional[Path] = None
    # Optional object cap applied at load/epoch time (random downsample of objects).
    max_objects_per_image: Optional[int] = None
    # Optional prompt overrides (per-dataset) applied on top of domain/default prompts.
    prompt_user: Optional[str] = None
    prompt_system: Optional[str] = None
    # Optional deterministic seed for per-epoch sampling/shuffling.
    seed: Optional[int] = None
    # Optional per-dataset sampling policy: when true and quota <= pool, sample
    # without replacement. CoordExp does not guarantee determinism; this only
    # controls whether duplicates can appear when downsampling.
    sample_without_replacement: bool = False


@dataclass(frozen=True)
class FusionDatasetSpec(DatasetSpec):
    """Dataset entry used by CoordExp fusion training.

    CoordExp treats every dataset entry uniformly (no target/source semantics).
    Each entry contributes `round(len(pool) * ratio)` samples per epoch.
    """

    ratio: float = 1.0


__all__ = [
    "DatasetSpec",
    "FusionDatasetSpec",
    "FALLBACK_OPTIONS",
]
