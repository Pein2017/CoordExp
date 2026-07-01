"""Resolved config artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.common.errors import ConfigContractError
from src.config.models import ResolvedConfigArtifacts, ResolvedTrainConfig


def write_resolved_config_artifacts(
    resolved_config: ResolvedTrainConfig,
    run_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ResolvedConfigArtifacts:
    config_dir = Path(run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = resolved_config.to_artifact_dict()
    json_path = config_dir / "resolved.json"
    yaml_path = config_dir / "resolved.yaml"
    existing = [path for path in (json_path, yaml_path) if path.exists()]
    if existing and not overwrite:
        raise ConfigContractError(
            "resolved config artifacts already exist",
            code="config.resolved_artifact_exists",
            context={"paths": [str(path) for path in existing]},
        )

    try:
        json_payload = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    except ValueError as exc:
        raise ConfigContractError(
            "resolved config artifact contains non-finite JSON values",
            code="config.resolved_artifact_non_finite",
            cause=exc,
        ) from exc
    json_path.write_text(json_payload + "\n", encoding="utf-8")
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    return ResolvedConfigArtifacts(
        yaml_path=yaml_path,
        json_path=json_path,
        fingerprint=resolved_config.fingerprint,
    )
