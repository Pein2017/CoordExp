"""Standalone COCO-80 refinement service primitives."""

from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import (
    CanonicalDraft,
    DraftSnapshotBinding,
    NativeObject,
    NativeTaskIdentity,
)

__all__ = [
    "CanonicalDraft",
    "DraftSnapshotBinding",
    "NativeObject",
    "NativeTaskIdentity",
    "canonicalize_objects",
]
