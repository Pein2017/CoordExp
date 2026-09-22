"""Image reference helpers for V1 raw examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.errors import DataContractError


def resolve_image_path(
    value: Any,
    *,
    root: Path,
    field: str,
    require_under_root: bool = False,
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(
            "image path must be a non-empty string",
            code="data.image_path",
            context={"field": field, "value_type": type(value).__name__},
        )
    declared = value
    raw_path = Path(value)
    resolved_root = root.resolve()
    resolved = (raw_path if raw_path.is_absolute() else resolved_root / raw_path).resolve()
    if require_under_root:
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise DataContractError(
                "canonical image path must stay under the JSONL directory",
                code="data.image_path_escape",
                context={
                    "field": field,
                    "declared_path": declared,
                    "jsonl_root": str(resolved_root),
                    "resolved_path": str(resolved),
                },
                cause=exc,
            ) from exc
    if not resolved.exists():
        raise DataContractError(
            "image path does not exist",
            code="data.image_missing",
            context={"field": field, "declared_path": declared, "resolved_path": str(resolved)},
        )
    if not resolved.is_file():
        raise DataContractError(
            "image path is not a file",
            code="data.image_not_file",
            context={"field": field, "declared_path": declared, "resolved_path": str(resolved)},
        )
    return declared, resolved


def validate_declared_dimension(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(
            "image dimension must be a positive integer",
            code="data.image_dimension_type",
            context={"field": field, "value": value, "value_type": type(value).__name__},
        )
    if value <= 0:
        raise DataContractError(
            "image dimension must be positive",
            code="data.image_dimension_range",
            context={"field": field, "value": value},
        )
    return value


def image_stat_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


__all__ = [
    "image_stat_fingerprint",
    "resolve_image_path",
    "validate_declared_dimension",
]
