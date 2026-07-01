"""Strict YAML config loading for CoordExp-swift."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.common.errors import ConfigContractError
from src.config.fingerprint import sha256_file, sha256_json
from src.config.models import (
    CONFIG_LOADER_VERSION,
    ConfigSource,
    ResolvedTrainConfig,
    TrainConfig,
)
from src.config.paths import resolve_path_fields
from src.config.resolve import validate_static_qwen_runtime_controls


def load_train_config(path: str | Path) -> ResolvedTrainConfig:
    entry_path = Path(path).expanduser().resolve()
    entry_payload = _load_yaml_mapping(entry_path)
    if entry_payload.get("schema_version") != 1:
        raise ConfigContractError(
            "runnable root config must declare schema_version: 1",
            code="config.schema_version",
            context={"path": str(entry_path)},
        )
    merged, origins, sources = _load_with_extends(entry_path, stack=())
    _reject_required_placeholders(merged)
    resolved_payload, path_origins = resolve_path_fields(merged, origins)
    try:
        config = TrainConfig.model_validate(resolved_payload)
    except ValidationError as exc:
        _raise_validation_error(exc, entry_path)
    validate_static_qwen_runtime_controls(config)
    config_dict = config.model_dump(mode="json")
    fingerprint = sha256_json(config_dict)
    return ResolvedTrainConfig(
        config=config,
        config_dict=config_dict,
        fingerprint=fingerprint,
        schema_version=config.schema_version,
        loader_version=CONFIG_LOADER_VERSION,
        entry_config_path=entry_path,
        sources=sources,
        path_origins=path_origins,
    )


def _load_with_extends(
    path: Path,
    *,
    stack: tuple[Path, ...],
) -> tuple[dict[str, Any], dict[str, Path], tuple[ConfigSource, ...]]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigContractError(
            "config extends cycle detected",
            code="config.extends_cycle",
            context={"cycle": cycle},
        )
    payload = _load_yaml_mapping(path)
    _reject_nested_extends(payload, path)
    parent_ref = payload.pop("extends", None)
    if parent_ref is None:
        merged: dict[str, Any] = {}
        origins: dict[str, Path] = {}
        sources: tuple[ConfigSource, ...] = ()
    else:
        if not isinstance(parent_ref, str):
            raise ConfigContractError(
                "extends must be a single parent path string",
                code="config.extends_shape",
                context={"path": str(path), "value_type": type(parent_ref).__name__},
            )
        parent_path = (path.parent / parent_ref).resolve()
        merged, origins, sources = _load_with_extends(
            parent_path,
            stack=(*stack, path),
        )
    current_origins = _leaf_origins(payload, path)
    merged_payload, merged_origins = _deep_merge(
        merged,
        payload,
        origins,
        current_origins,
    )
    return (
        merged_payload,
        merged_origins,
        (*sources, ConfigSource(path=path, sha256=sha256_file(path))),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigContractError(
            "config file does not exist",
            code="config.missing_file",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ConfigContractError(
            "config file must contain a YAML mapping",
            code="config.mapping",
            context={"path": str(path), "value_type": type(payload).__name__},
        )
    return payload


def _reject_nested_extends(value: Any, path: Path, dotted: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{dotted}.{key}" if dotted else str(key)
            if key == "extends" and dotted:
                raise ConfigContractError(
                    "extends is allowed only at the YAML top level",
                    code="config.nested_extends",
                    context={"path": str(path), "field": child_path},
                )
            _reject_nested_extends(child, path, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{dotted}[{index}]"
            _reject_nested_extends(child, path, child_path)


def _leaf_origins(payload: dict[str, Any], path: Path) -> dict[str, Path]:
    origins: dict[str, Path] = {}

    def visit(value: Any, parts: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*parts, str(key)))
        else:
            origins[".".join(parts)] = path

    visit(payload, ())
    return origins


def _deep_merge(
    parent: dict[str, Any],
    child: dict[str, Any],
    parent_origins: dict[str, Path],
    child_origins: dict[str, Path],
    *,
    prefix: str = "",
) -> tuple[dict[str, Any], dict[str, Path]]:
    merged = _copy_mapping(parent)
    origins = dict(parent_origins)
    for key, child_value in child.items():
        field_path = f"{prefix}.{key}" if prefix else key
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(child_value, dict):
            sub_payload, sub_origins = _deep_merge(
                parent_value,
                child_value,
                _sub_origins(origins, key),
                _sub_origins(child_origins, key),
                prefix=field_path,
            )
            merged[key] = sub_payload
            _remove_origin_prefix(origins, key)
            origins.update(
                {
                    f"{key}.{sub_key}": sub_path
                    for sub_key, sub_path in sub_origins.items()
                }
            )
            continue
        if child_value is None and key in merged:
            raise ConfigContractError(
                "null cannot delete an inherited config value",
                code="config.null_inherited_delete",
                context={
                    "field": field_path,
                    "declaring_config_path": str(child_origins.get(key, "")),
                },
            )
        merged[key] = _copy_value(child_value)
        _remove_origin_prefix(origins, key)
        origins.update(
            {
                path: origin
                for path, origin in child_origins.items()
                if path == key or path.startswith(f"{key}.")
            }
        )
    return merged, origins


def _sub_origins(origins: dict[str, Path], key: str) -> dict[str, Path]:
    prefix = f"{key}."
    return {
        path.removeprefix(prefix): origin
        for path, origin in origins.items()
        if path.startswith(prefix)
    }


def _remove_origin_prefix(origins: dict[str, Path], key: str) -> None:
    prefix = f"{key}."
    for path in list(origins):
        if path == key or path.startswith(prefix):
            del origins[path]


def _copy_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _copy_value(value) for key, value in payload.items()}


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _copy_mapping(value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _reject_required_placeholders(value: Any, dotted: str = "") -> None:
    if value == "REQUIRED":
        raise ConfigContractError(
            "REQUIRED placeholder survived into runnable config",
            code="config.required_placeholder",
            context={"field": dotted or "<root>"},
        )
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{dotted}.{key}" if dotted else str(key)
            _reject_required_placeholders(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_required_placeholders(child, f"{dotted}[{index}]")


def _raise_validation_error(exc: ValidationError, path: Path) -> None:
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    selected_error = next(
        (
            error
            for error in errors
            if "V1 uses adapter.type: dora" in str(error.get("msg", ""))
        ),
        first_error,
    )
    location = ".".join(str(part) for part in selected_error.get("loc", ()))
    validation_message = str(selected_error.get("msg", str(exc)))
    message = (
        validation_message
        if "V1 uses adapter.type: dora" in validation_message
        else "config schema validation failed"
    )
    raise ConfigContractError(
        message,
        code="config.schema_validation",
        context={
            "path": str(path),
            "field": location,
            "message": validation_message,
            "error_count": len(errors),
        },
        cause=exc,
    ) from exc
