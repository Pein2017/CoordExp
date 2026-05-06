"""Model path normalization helpers for portable CoordExp artifacts."""

from __future__ import annotations

from pathlib import Path


def canonical_coordexp_repo_root() -> Path:
    """Return the durable CoordExp root, collapsing repo worktree paths."""

    root = Path(__file__).resolve().parents[2]
    parts = root.parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        return Path(*parts[:idx])
    return root


def normalize_coordexp_base_model_path(path: str | None) -> str | None:
    """Normalize ``model_cache/**/*-coordexp`` paths to the durable repo root."""

    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return raw

    candidate = Path(raw).expanduser()
    parts = candidate.parts
    if "model_cache" not in parts:
        return raw
    if not candidate.name.endswith("-coordexp"):
        return raw

    idx = parts.index("model_cache")
    suffix = Path(*parts[idx:])
    return str(canonical_coordexp_repo_root() / suffix)


__all__ = [
    "canonical_coordexp_repo_root",
    "normalize_coordexp_base_model_path",
]
