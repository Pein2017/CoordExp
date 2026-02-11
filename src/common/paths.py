"""Shared path-resolution helpers.

These utilities are intentionally small and dependency-light so they can be
reused across datasets, inference, evaluation, and visualization without
introducing import cycles.

IMPORTANT: Some callsites require different strictness:
- Best-effort resolution: produce a deterministic absolute Path even if missing.
- Strict resolution: return a Path only when it exists; otherwise return None.

This module encodes both behaviors explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_image_path_best_effort(
    image_field: str,
    *,
    jsonl_dir: Path | None,
    root_image_dir: Path | None = None,
    env_root_var: str | None = "ROOT_IMAGE_DIR",
) -> Path:
    """Resolve an image path deterministically without existence checks.

    Resolution precedence for relative paths:
      1) `root_image_dir` (explicit)
      2) environment variable `env_root_var` (if provided and set)
      3) `jsonl_dir` (typically JSONL parent directory)
      4) cwd

    Absolute paths are returned as-is.
    """

    p = Path(str(image_field))
    if p.is_absolute():
        return p

    base: Path | None = None
    if root_image_dir is not None:
        base = root_image_dir

    if base is None and env_root_var:
        root = os.environ.get(str(env_root_var))
        if root:
            base = Path(root)

    if base is None:
        base = jsonl_dir

    if base is None:
        base = Path.cwd()

    return (base / p).resolve()


def resolve_image_path_strict(
    image_field: Optional[str],
    *,
    jsonl_dir: Path | None,
    root_image_dir: Path | None = None,
) -> Path | None:
    """Resolve an image path and return it only if it exists.

    Resolution precedence:
      1) absolute existing paths
      2) `root_image_dir / image_field` if provided and exists
      3) `jsonl_dir / image_field` if provided and exists

    Returns None when resolution fails.
    """

    if not image_field:
        return None

    p = Path(str(image_field))
    if p.is_absolute() and p.exists():
        return p

    if root_image_dir is not None:
        cand = (root_image_dir / p)
        if cand.exists():
            return cand

    if jsonl_dir is not None:
        cand = (jsonl_dir / p)
        if cand.exists():
            return cand

    return None
