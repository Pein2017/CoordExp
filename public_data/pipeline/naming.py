from __future__ import annotations

import re

_CANONICAL_MAX_SUFFIX_RE = re.compile(r"(?:_max(\d+))+$")
_LEGACY_MAX_SUFFIX_RE = re.compile(r"(?:_max_(\d+))+$")


def max_token(max_objects: int) -> str:
    return f"max{int(max_objects)}"


def canonical_suffix(max_objects: int) -> str:
    return f"_max{int(max_objects)}"


def _extract_trailing_max_value(name: str) -> int | None:
    match = _CANONICAL_MAX_SUFFIX_RE.search(name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _strip_trailing_canonical_suffix(name: str) -> str:
    return _CANONICAL_MAX_SUFFIX_RE.sub("", name)


def _require_no_legacy_suffix(name: str) -> None:
    legacy_match = _LEGACY_MAX_SUFFIX_RE.search(name)
    if legacy_match is None:
        return

    legacy_suffix = legacy_match.group(0)
    canonical_name = _LEGACY_MAX_SUFFIX_RE.sub(
        canonical_suffix(int(legacy_match.group(1))),
        name,
        count=1,
    )
    raise ValueError(
        "Legacy max-object suffix is not supported: "
        f"'{legacy_suffix}' in preset '{name}'. "
        f"Use canonical preset naming (for example '{canonical_name}')."
    )


def apply_max_suffix(base_preset: str, max_objects: int | None) -> str:
    """Apply canonical `_max{N}` suffix if max filtering is configured."""
    _require_no_legacy_suffix(base_preset)

    if max_objects is None:
        return base_preset

    max_n = int(max_objects)
    if max_n <= 0:
        raise ValueError("max_objects must be > 0 when provided")

    existing = _extract_trailing_max_value(base_preset)
    if existing is not None and existing != max_n:
        raise ValueError(
            "Conflicting max_objects sources: "
            f"preset '{base_preset}' encodes {existing}, but max_objects={max_n}."
        )

    root = _strip_trailing_canonical_suffix(base_preset)
    if existing == max_n:
        return f"{root}{canonical_suffix(max_n)}"
    return f"{root}{canonical_suffix(max_n)}"


def resolve_effective_preset(base_preset: str, max_objects: int | None) -> str:
    """Resolve emitted preset using canonical naming only."""
    return apply_max_suffix(base_preset, max_objects)
