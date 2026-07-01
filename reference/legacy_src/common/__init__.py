"""Shared, low-risk utilities and type surfaces for CoordExp.

This package is intentionally thin: it re-exports common types/helpers without
changing behavior, so upstream modules can import a single canonical location.
"""

from __future__ import annotations

# Re-export core shared modules
from . import schemas  # noqa: F401
from . import geometry  # noqa: F401
from .coord_standardizer import CoordinateStandardizer  # noqa: F401
from .io import load_jsonl  # noqa: F401

__all__ = ["schemas", "geometry", "load_jsonl", "CoordinateStandardizer"]
