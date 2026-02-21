from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

SPLITS: tuple[str, ...] = ("train", "val")


@dataclass(frozen=True)
class PipelineConfig:
    dataset_id: str
    dataset_dir: Path
    raw_dir: Path
    preset: str
    max_objects: Optional[int] = None
    image_factor: int = 32
    max_pixels: int = 32 * 32 * 768
    min_pixels: int = 32 * 32 * 4
    num_workers: int = 1
    relative_images: bool = True
    assume_normalized: bool = False
    compact_json: bool = False
    skip_image_check: bool = False
    run_validation_stage: bool = False


@dataclass(frozen=True)
class SplitArtifactPaths:
    split: str
    raw: Path
    norm: Path
    coord: Path
    filter_stats: Path


@dataclass
class PipelineState:
    config: PipelineConfig
    effective_preset: str
    base_preset: str
    base_preset_dir: Path
    is_derived_preset: bool
    preset_dir: Path
    split_inputs: Dict[str, Path]
    split_artifacts: Dict[str, SplitArtifactPaths]
    split_raw_sources: Dict[str, Path] = field(default_factory=dict)
    stage_stats: Dict[str, dict] = field(default_factory=dict)
    records_seen: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    dataset_id: str
    preset: str
    preset_dir: Path
    split_artifacts: Dict[str, SplitArtifactPaths]
    stage_stats: Dict[str, dict]
