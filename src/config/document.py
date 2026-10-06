"""Shared YAML inheritance mechanics; typed train/inference policy stays at its owner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.common.errors import ConfigContractError
from src.config.fingerprint import sha256_file
from src.config.models import ConfigSource


def load_inherited_mapping(
    path: Path, *, replace_discriminated: bool = False,
) -> tuple[dict[str, Any], dict[str, Path], tuple[ConfigSource, ...]]:
    """Resolve one parent chain, retaining leaf authorship and ordered source hashes.

    Inference replaces objects when their type changes. Training deliberately
    keeps ordinary recursive merging; selecting a different policy is not an alias.
    Schema-version and final placeholder checks remain with the typed loader.
    """
    return _load_with_extends(path, stack=(), replace_discriminated=replace_discriminated)


def _load_with_extends(
    path: Path,
    *,
    stack: tuple[Path, ...],
    replace_discriminated: bool,
) -> tuple[dict[str, Any], dict[str, Path], tuple[ConfigSource, ...]]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigContractError(
            "config extends cycle detected",
            code="config.extends_cycle",
            context={"cycle": cycle},
        )
    payload = read_yaml_mapping(path)
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
            replace_discriminated=replace_discriminated,
        )
    current_origins = _leaf_origins(payload, path)
    merged_payload, merged_origins = _deep_merge(
        merged,
        payload,
        origins,
        current_origins,
        replace_discriminated=replace_discriminated,
    )
    return (
        merged_payload,
        merged_origins,
        (*sources, ConfigSource(path=path, sha256=sha256_file(path))),
    )


def read_yaml_mapping(path: Path) -> dict[str, Any]:
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
    replace_discriminated: bool,
) -> tuple[dict[str, Any], dict[str, Path]]:
    merged = copy_mapping(parent)
    origins = dict(parent_origins)
    for key, child_value in child.items():
        field_path = f"{prefix}.{key}" if prefix else key
        parent_value = merged.get(key)
        if (
            isinstance(parent_value, dict)
            and isinstance(child_value, dict)
            and not (replace_discriminated and _discriminator_changed(parent_value, child_value))
        ):
            sub_payload, sub_origins = _deep_merge(
                parent_value,
                child_value,
                _sub_origins(origins, key),
                _sub_origins(child_origins, key),
                prefix=field_path,
                replace_discriminated=replace_discriminated,
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


def copy_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy each occurrence independently, including containers shared by YAML aliases."""
    return {key: _copy_value(value) for key, value in payload.items()}


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return copy_mapping(value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def reject_required_placeholders(value: Any, dotted: str = "") -> None:
    if value == "REQUIRED":
        raise ConfigContractError(
            "REQUIRED placeholder survived into runnable config",
            code="config.required_placeholder",
            context={"field": dotted or "<root>"},
        )
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{dotted}.{key}" if dotted else str(key)
            reject_required_placeholders(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_required_placeholders(child, f"{dotted}[{index}]")


def _discriminator_changed(
    parent: dict[str, Any],
    child: dict[str, Any],
) -> bool:
    """Replace a discriminated config object when its selected variant changes."""

    return (
        "type" in parent
        and "type" in child
        and parent["type"] != child["type"]
    )
