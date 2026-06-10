from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import PROJECT_ID, RUN_ID, SCHEMA_VERSION


@dataclass(frozen=True)
class CheckpointRoleConfig:
    objective_policy: str
    training_ordering: str
    template_contract_id: str
    comparison_group: str


@dataclass(frozen=True)
class ArtifactSourceConfig:
    source_id: str
    root: Path
    evidence_scope: str
    required: bool = False


@dataclass(frozen=True)
class ComparisonConfig:
    project_id: str
    schema_version: str
    run_id: str
    artifact_root: Path
    checkpoint_roles: dict[str, CheckpointRoleConfig]
    artifact_sources: dict[str, ArtifactSourceConfig]


def load_config(path: str | Path) -> ComparisonConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("policy/objective comparison config must be a mapping")
    project_id = str(raw.get("project_id") or "")
    schema_version = str(raw.get("schema_version") or "")
    run_id = str(raw.get("run_id") or "")
    if project_id != PROJECT_ID:
        raise ValueError(f"project_id must be {PROJECT_ID}")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if run_id != RUN_ID:
        raise ValueError(f"run_id must be {RUN_ID}")
    return ComparisonConfig(
        project_id=project_id,
        schema_version=schema_version,
        run_id=run_id,
        artifact_root=_required_path(raw, "artifact_root"),
        checkpoint_roles=_load_roles(_required_mapping(raw, "checkpoint_roles")),
        artifact_sources=_load_sources(_required_mapping(raw, "artifact_sources")),
    )


def _load_roles(raw: Mapping[str, Any]) -> dict[str, CheckpointRoleConfig]:
    roles: dict[str, CheckpointRoleConfig] = {}
    for role, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoint_roles.{role} must be a mapping")
        roles[str(role)] = CheckpointRoleConfig(
            objective_policy=_required(value, "objective_policy"),
            training_ordering=_required(value, "training_ordering"),
            template_contract_id=_required(value, "template_contract_id"),
            comparison_group=_required(value, "comparison_group"),
        )
    if len(roles) != 5:
        raise ValueError("checkpoint_roles must contain exactly five roles")
    return roles


def _load_sources(raw: Mapping[str, Any]) -> dict[str, ArtifactSourceConfig]:
    sources: dict[str, ArtifactSourceConfig] = {}
    for source_id, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"artifact_sources.{source_id} must be a mapping")
        source_id_text = str(source_id)
        sources[source_id_text] = ArtifactSourceConfig(
            source_id=source_id_text,
            root=_required_path(value, "root"),
            evidence_scope=str(value.get("evidence_scope", source_id_text)),
            required=bool(value.get("required", False)),
        )
    if not sources:
        raise ValueError("artifact_sources must not be empty")
    return sources


def _required(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or value == "":
        raise ValueError(f"missing config key: {key}")
    return str(value)


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_path(raw: Mapping[str, Any], key: str) -> Path:
    path = Path(_required(raw, key))
    if not path.is_absolute():
        raise ValueError(f"{key} must be an absolute path")
    return path
