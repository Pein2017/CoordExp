"""Compatibility shim for the batch-extras collator wrapper.

Canonical implementation lives in `src.data_collators.batch_extras_collator`.
"""

from __future__ import annotations

from .batch_extras_collator import (  # noqa: F401
    build_batch_extras_collator,
    build_dataset_metrics_collator,
)

__all__ = [
    "build_batch_extras_collator",
    "build_dataset_metrics_collator",
]
