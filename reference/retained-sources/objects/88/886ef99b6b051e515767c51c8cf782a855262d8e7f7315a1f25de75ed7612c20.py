"""Lane B: fixed-grid visual-detail dependence preparation and qualification."""

from .runtime import (
    BASELINE_MAX_PIXELS,
    HIGH_MAX_PIXELS,
    prepare_manifest,
    qualify_one,
)

__all__ = ["BASELINE_MAX_PIXELS", "HIGH_MAX_PIXELS", "prepare_manifest", "qualify_one"]
