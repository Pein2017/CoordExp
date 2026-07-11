"""Path resolution helpers for authored config fields."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.errors import ConfigContractError
from src.config.models import PathOrigin, RunDirectory, TrainConfig


PATH_FIELDS = (
    "model.base_model",
    "adapter.path",
    "data.train.path",
    "data.eval.path",
)


def get_nested(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if current is None:
            return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested(payload: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        current = child
    current[parts[-1]] = value


def resolve_path_fields(
    payload: dict[str, Any],
    leaf_origins: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, PathOrigin]]:
    resolved = _deep_copy_dict(payload)
    path_origins: dict[str, PathOrigin] = {}
    for field in PATH_FIELDS:
        declared = get_nested(resolved, field)
        if declared is None:
            continue
        if not isinstance(declared, str):
            raise ConfigContractError(
                "path field must be a string",
                code="config.path_type",
                context={"field": field, "value_type": type(declared).__name__},
            )
        declaring_file = leaf_origins.get(field)
        if declaring_file is None:
            continue
        declared_path = Path(declared)
        resolved_path = (
            declared_path
            if declared_path.is_absolute()
            else declaring_file.parent / declared_path
        ).resolve()
        set_nested(resolved, field, str(resolved_path))
        path_origins[field] = PathOrigin(
            field=field,
            declared_path=declared,
            declaring_config_path=declaring_file,
            resolved_path=resolved_path,
        )
    return resolved, path_origins


def resolve_run_directory(
    config: TrainConfig,
    *,
    cwd: Path | None = None,
    timestamp: str | None = None,
) -> RunDirectory:
    root_base = Path(config.run.artifact_root)
    root = root_base if root_base.is_absolute() else (cwd or Path.cwd()) / root_base
    root = root.resolve()

    run_dir_name = config.run.output_dir or config.run.name
    run_dir_path = Path(run_dir_name)
    if run_dir_path.is_absolute() or ".." in run_dir_path.parts:
        raise ConfigContractError(
            "run.output_dir must stay under run.artifact_root",
            code="config.run_output_dir_escape",
            context={"output_dir": run_dir_name},
        )
    run_dir = (root / run_dir_path).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ConfigContractError(
            "run output directory escaped artifact root",
            code="config.run_output_dir_escape",
            context={"artifact_root": str(root), "run_dir": str(run_dir)},
            cause=exc,
        ) from exc

    if run_dir.exists():
        if config.run.collision_policy == "fail":
            raise ConfigContractError(
                "run output directory already exists",
                code="config.run_dir_exists",
                context={"run_dir": str(run_dir), "collision_policy": "fail"},
            )
        suffix = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = run_dir.with_name(f"{run_dir.name}-{suffix}")
        if run_dir.exists():
            raise ConfigContractError(
                "timestamped run output directory already exists",
                code="config.timestamped_run_dir_exists",
                context={"run_dir": str(run_dir), "collision_policy": "timestamp"},
            )

    return RunDirectory(
        run_name=config.run.name,
        artifact_root=root,
        run_dir=run_dir.resolve(),
        collision_policy=config.run.collision_policy,
    )


def _deep_copy_dict(payload: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            copied[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            copied[key] = [_deep_copy_value(item) for item in value]
        else:
            copied[key] = value
    return copied


def _deep_copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _deep_copy_dict(value)
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    return value
