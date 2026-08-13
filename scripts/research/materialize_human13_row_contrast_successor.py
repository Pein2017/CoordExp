#!/usr/bin/env python3
"""Validate and materialize the two bounded Human-13 successor arms."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "human13_row_contrast_arm.v1"
PLAN_SCHEMA_VERSION = "human13_row_contrast_resolved_plan.v1"
UNIT_ID = "2026-08-13-human13-row-contrast-geometry-preservation-successor"
CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_row_contrast_successor")
MILESTONES = (0, 1, 2)
MANIFEST_SHA256 = "a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb"
FAMILIES = ("union", "replay", "row_contrast", "fallback", "rectangle")
ZERO_ACTIONS = {
    "model_loads": 0,
    "forwards": 0,
    "backwards": 0,
    "optimizer_steps": 0,
    "checkpoint_writes": 0,
    "gpu_allocations": 0,
}


class SuccessorMaterializationError(ValueError):
    """Raised when a successor plan is not exactly frozen."""


@dataclass(frozen=True)
class ProjectionConfig:
    epsilon: float
    tolerance: float


@dataclass(frozen=True)
class SuccessorArmConfig:
    schema_version: str
    unit_id: str
    arm_id: str
    arm_name: str
    updates: bool
    base_arm_config: str
    manifest_path: str
    manifest_sha256: str
    ledger_path: str
    ledger_sha256: str
    duplicate_event_sources: str
    candidate_alias_policy: str
    output_root: str
    global_max_length: int
    max_updates: int
    milestones: tuple[int, ...]
    duplicate_margin: float
    rectangle_margin: float
    loss_coefficients: tuple[tuple[str, float], ...]
    gradient_projection: bool
    projection: ProjectionConfig

    def to_artifact_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["loss_coefficients"] = dict(self.loss_coefficients)
        value["schema_version"] = PLAN_SCHEMA_VERSION
        value["source_schema_version"] = self.schema_version
        value["actions"] = dict(ZERO_ACTIONS)
        return value


def load_successor_config(
    path: str | Path, *, repo_root: str | Path | None = None
) -> SuccessorArmConfig:
    target = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SuccessorMaterializationError("successor config must be an object")
    required = {
        "schema_version",
        "unit_id",
        "arm_id",
        "arm_name",
        "updates",
        "base_arm_config",
        "manifest_path",
        "manifest_sha256",
        "ledger_path",
        "ledger_sha256",
        "duplicate_event_sources",
        "candidate_alias_policy",
        "output_root",
        "global_max_length",
        "max_updates",
        "milestones",
        "duplicate_margin",
        "rectangle_margin",
        "loss_coefficients",
        "gradient_projection",
        "projection",
    }
    if set(raw) != required:
        raise SuccessorMaterializationError("successor config fields differ")
    coefficients = raw["loss_coefficients"]
    projection = raw["projection"]
    if not isinstance(coefficients, Mapping) or set(coefficients) != set(FAMILIES):
        raise SuccessorMaterializationError("loss coefficient families differ")
    if not isinstance(projection, Mapping) or set(projection) != {
        "epsilon",
        "tolerance",
    }:
        raise SuccessorMaterializationError("projection fields differ")
    config = SuccessorArmConfig(
        schema_version=str(raw["schema_version"]),
        unit_id=str(raw["unit_id"]),
        arm_id=str(raw["arm_id"]),
        arm_name=str(raw["arm_name"]),
        updates=bool(raw["updates"]),
        base_arm_config=str(raw["base_arm_config"]),
        manifest_path=str(raw["manifest_path"]),
        manifest_sha256=str(raw["manifest_sha256"]),
        ledger_path=str(raw["ledger_path"]),
        ledger_sha256=str(raw["ledger_sha256"]),
        duplicate_event_sources=str(raw["duplicate_event_sources"]),
        candidate_alias_policy=str(raw["candidate_alias_policy"]),
        output_root=str(raw["output_root"]),
        global_max_length=int(raw["global_max_length"]),
        max_updates=int(raw["max_updates"]),
        milestones=tuple(int(value) for value in raw["milestones"]),
        duplicate_margin=float(raw["duplicate_margin"]),
        rectangle_margin=float(raw["rectangle_margin"]),
        loss_coefficients=tuple((name, float(coefficients[name])) for name in FAMILIES),
        gradient_projection=bool(raw["gradient_projection"]),
        projection=ProjectionConfig(
            epsilon=float(projection["epsilon"]),
            tolerance=float(projection["tolerance"]),
        ),
    )
    _validate_config(config, repo_root=repo_root or target.parents[4])
    return config


def successor_model_config(config: SuccessorArmConfig, *, repo_root: str | Path):
    """Reuse the frozen A4 live-model surface with successor identities only."""

    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    root = Path(repo_root).resolve()
    path = Path(config.base_arm_config)
    path = path if path.is_absolute() else root / path
    base = load_arm_config(path.resolve(strict=True))
    return replace(
        base,
        unit_id=config.unit_id,
        arm_id=config.arm_id,
        arm_name=config.arm_name,
        milestones=config.milestones,
    )


def materialize_plans(config_paths: Sequence[str | Path]) -> dict[str, Any]:
    configs = tuple(load_successor_config(path) for path in config_paths)
    if tuple(config.arm_id for config in configs) != ("R1", "R2"):
        raise SuccessorMaterializationError("plans must contain ordered R1/R2")
    if len({config.output_root for config in configs}) != 2:
        raise SuccessorMaterializationError("R1/R2 output roots must be distinct")
    return {
        "schema_version": "human13_row_contrast_plans.v1",
        "unit_id": UNIT_ID,
        "plans": [config.to_artifact_dict() for config in configs],
        "actions": dict(ZERO_ACTIONS),
    }


def _validate_config(config: SuccessorArmConfig, *, repo_root: str | Path) -> None:
    if (
        config.schema_version != SCHEMA_VERSION
        or config.unit_id != UNIT_ID
        or config.arm_id not in {"R1", "R2"}
        or config.updates is not True
        or config.manifest_sha256 != MANIFEST_SHA256
        or config.global_max_length != 12_000
        or config.max_updates != 2
        or config.milestones != MILESTONES
        or config.gradient_projection != (config.arm_id == "R2")
        or config.duplicate_event_sources != "sealed_manifest_only"
        or config.candidate_alias_policy != "first_trajectory_row_per_owner"
    ):
        raise SuccessorMaterializationError("successor frozen contract differs")
    root = Path(repo_root).resolve()
    base = Path(config.base_arm_config)
    base = base if base.is_absolute() else root / base
    if not base.resolve(strict=True).is_file():
        raise SuccessorMaterializationError("base A4 config is unavailable")
    for label, path_value, expected in (
        ("manifest", config.manifest_path, config.manifest_sha256),
        ("ledger", config.ledger_path, config.ledger_sha256),
    ):
        path = Path(path_value).expanduser().resolve(strict=True)
        if _sha256_file(path) != expected:
            raise SuccessorMaterializationError(f"{label} SHA-256 drifted")
    root_output = Path(config.output_root).expanduser()
    if not root_output.is_absolute():
        raise SuccessorMaterializationError("output root must be absolute")
    for name, value in config.loss_coefficients:
        if name not in FAMILIES or not math.isfinite(value) or value != 1.0:
            raise SuccessorMaterializationError("loss coefficients differ")
    if (
        not math.isfinite(config.duplicate_margin)
        or config.duplicate_margin != 0.0
        or not math.isfinite(config.rectangle_margin)
        or config.rectangle_margin != 1.0e-4
        or not math.isfinite(config.projection.epsilon)
        or config.projection.epsilon != 1.0e-12
        or not math.isfinite(config.projection.tolerance)
        or config.projection.tolerance != 1.0e-5
    ):
        raise SuccessorMaterializationError("successor numeric contract differs")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(materialize_plans(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProjectionConfig",
    "SuccessorArmConfig",
    "SuccessorMaterializationError",
    "load_successor_config",
    "materialize_plans",
    "successor_model_config",
]
