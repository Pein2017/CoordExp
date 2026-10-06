"""Strict YAML config loading for coordexp-infras."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.common.errors import ConfigContractError
from src.config.document import load_inherited_mapping, read_yaml_mapping, reject_required_placeholders
from src.config.fingerprint import sha256_json
from src.config.models import (
    CONFIG_LOADER_VERSION,
    ResolvedTrainConfig,
    TrainConfig,
)
from src.config.paths import resolve_path_fields
from src.config.resolve import validate_static_qwen_runtime_controls


def load_train_config(path: str | Path) -> ResolvedTrainConfig:
    entry_path = Path(path).expanduser().resolve()
    entry_payload = read_yaml_mapping(entry_path)
    if entry_payload.get("schema_version") != 1:
        raise ConfigContractError(
            "runnable root config must declare schema_version: 1",
            code="config.schema_version",
            context={"path": str(entry_path)},
        )
    merged, origins, sources = load_inherited_mapping(entry_path)
    reject_required_placeholders(merged)
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
