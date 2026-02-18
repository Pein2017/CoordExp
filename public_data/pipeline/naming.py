from __future__ import annotations

import re
from pathlib import Path

_MAX_SUFFIX_RE = re.compile(r"(?:_max_?(\d+))+$")


def max_token(max_objects: int) -> str:
    return f"max_{int(max_objects)}"


def canonical_suffix(max_objects: int) -> str:
    return f"_max_{int(max_objects)}"


def legacy_suffix(max_objects: int) -> str:
    return f"_max{int(max_objects)}"


def strip_trailing_max_suffix(name: str) -> str:
    return _MAX_SUFFIX_RE.sub("", name)


def _extract_trailing_max_value(name: str) -> int | None:
    match = _MAX_SUFFIX_RE.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def apply_max_suffix(base_preset: str, max_objects: int | None) -> str:
    """Apply canonical `_max_<N>` suffix if max filtering is configured.

    Existing equivalent suffixes (`_max<N>` or `_max_<N>`) are deduplicated.
    """
    if max_objects is None:
        return base_preset

    max_n = int(max_objects)
    if max_n <= 0:
        raise ValueError("max_objects must be > 0 when provided")

    existing = _extract_trailing_max_value(base_preset)
    root = strip_trailing_max_suffix(base_preset)
    if existing == max_n:
        return f"{root}{canonical_suffix(max_n)}"
    return f"{root}{canonical_suffix(max_n)}"


def equivalent_legacy_name(canonical_name: str, max_objects: int) -> str:
    root = strip_trailing_max_suffix(canonical_name)
    return f"{root}{legacy_suffix(max_objects)}"


def resolve_effective_preset(dataset_dir: Path, base_preset: str, max_objects: int | None) -> str:
    """Resolve emitted preset while preserving legacy `max<N>` reuse semantics."""
    if max_objects is None:
        return base_preset

    canonical = apply_max_suffix(base_preset, max_objects)
    legacy = equivalent_legacy_name(canonical, int(max_objects))
    canonical_path = dataset_dir / canonical
    legacy_path = dataset_dir / legacy

    # Reuse legacy artifacts when they already exist and canonical does not.
    if legacy_path.exists() and not canonical_path.exists():
        return legacy
    return canonical
