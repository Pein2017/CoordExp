"""Datasets package for dense captioning.

This package provides a modular, ms-swift-aligned architecture for training
dense caption models with configurable builders, preprocessors, and augmentation.

Submodules:
- utils: shared utilities (load_jsonl, extract_geometry, etc.)
- geometry: geometry ops (scaling, affine transformations)
- augment: image + geometry augmentations
- collators: data collation utilities
- preprocessors: row-level transformations (ms-swift style)
- builders: message builders for different output formats
- dense_caption: high-level dataset class for dense captioning

Main exports:
- DenseCaptionDataset: Primary dataset class
- Builders: JSONLinesBuilder
- Preprocessors: DenseCaptionPreprocessor, AugmentationPreprocessor

Note: Fusion (multi-JSONL) training is supported via `custom.fusion_config` and
is implemented in `datasets/fusion.py` + `datasets/unified_fusion_dataset.py`.
"""

from .contracts import (
    MessageContent,
    MessageDict,
    ConversationRecord,
    GeometryDict,
    AugmentationTelemetry,
    validate_conversation_record,
    validate_geometry_sequence,
)
from .dense_caption import BaseCaptionDataset, DenseCaptionDataset
from .utils import load_jsonl, extract_object_points, extract_geometry

# AugmentationConfig removed with v1 API; use Compose pipelines directly
from .augmentation.base import Compose
from .augmentation.registry import register, get, available
from .builders import (
    BaseBuilder,
    JSONLinesBuilder,
)
from .preprocessors import (
    BasePreprocessor,
    DenseCaptionPreprocessor,
    AugmentationPreprocessor,
)
from .wrappers.packed_caption import PackedCaptionDataset, build_packed_dataset
from .wrappers.random_sample import RandomSampleDataset

__all__ = [
    # Primary dataset
    "BaseCaptionDataset",
    "DenseCaptionDataset",
    "PackedCaptionDataset",
    "build_packed_dataset",
    "RandomSampleDataset",
    # Utilities
    "load_jsonl",
    "extract_object_points",
    "extract_geometry",
    # Contracts
    "MessageContent",
    "MessageDict",
    "ConversationRecord",
    "GeometryDict",
    "AugmentationTelemetry",
    "validate_conversation_record",
    "validate_geometry_sequence",
    # Augmentation
    "Compose",
    "register",
    "get",
    "available",
    # Builders
    "BaseBuilder",
    "JSONLinesBuilder",
    # Preprocessors
    "BasePreprocessor",
    "DenseCaptionPreprocessor",
    "AugmentationPreprocessor",
]
