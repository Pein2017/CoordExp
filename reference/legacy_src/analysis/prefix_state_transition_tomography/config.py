from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import PROJECT_ID


KNOWN_STAGES = {
    "prefix_state_index",
    "paired_checkpoint_probe",
    "merge",
    "report",
    "gallery",
    "validate",
}


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_path: Path
    objective_policy: str = ""
    training_ordering: str = ""
    template_contract_id: str = ""
    comparison_group: str = ""


@dataclass(frozen=True)
class SamplingConfig:
    max_prefix_states: int = 4096
    num_shards: int = 8
    seed: int = 3664
    easy_sanity_max_fraction: float = 0.20


@dataclass(frozen=True)
class PeakConfig:
    absolute_mass_floor: float = 0.002
    relative_floor: float = 0.10
    primary_merge_radius: int = 24
    gt_x1_neighborhood_radius: int = 24
    raw_topk_k: int = 32


@dataclass(frozen=True)
class PrefixStateTransitionConfig:
    project_id: str
    run_id: str
    index_checkpoint_id: str
    index_checkpoint_role: str
    artifact_root: Path
    train_jsonl: Path
    val_jsonl: Path
    image_root: Path | None
    stages: tuple[str, ...]
    checkpoints: dict[str, CheckpointConfig]
    sampling: SamplingConfig
    peak: PeakConfig


def load_config(path: str | Path) -> PrefixStateTransitionConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("prefix-state transition config must be a mapping")

    project_id = str(raw.get("project_id") or "")
    if project_id != PROJECT_ID:
        raise ValueError(f"project_id must be {PROJECT_ID}")

    stages = _string_tuple(raw.get("stages", ("prefix_state_index", "validate")), "stages")
    unknown = sorted(set(stages) - KNOWN_STAGES)
    if unknown:
        raise ValueError(f"unknown stage: {unknown[0]}")

    checkpoints_raw = _mapping_or_empty(raw.get("checkpoints"), "checkpoints")
    checkpoints = _load_checkpoints(checkpoints_raw)
    sampling_raw = _mapping_or_empty(raw.get("sampling"), "sampling")
    peak_raw = _mapping_or_empty(raw.get("peak"), "peak")

    config = PrefixStateTransitionConfig(
        project_id=project_id,
        run_id=str(raw.get("run_id", "phase_a3_1_ckpt3664_4096")),
        index_checkpoint_id=str(raw.get("index_checkpoint_id", "paired_ckpt3664")),
        index_checkpoint_role=str(raw.get("index_checkpoint_role", "paired_index")),
        artifact_root=_required_path(raw, "artifact_root"),
        train_jsonl=_required_path(raw, "train_jsonl"),
        val_jsonl=_required_path(raw, "val_jsonl"),
        image_root=_optional_path(raw, "image_root"),
        stages=stages,
        checkpoints=checkpoints,
        sampling=SamplingConfig(
            max_prefix_states=int(sampling_raw.get("max_prefix_states", 4096)),
            num_shards=int(sampling_raw.get("num_shards", 8)),
            seed=int(sampling_raw.get("seed", 3664)),
            easy_sanity_max_fraction=float(sampling_raw.get("easy_sanity_max_fraction", 0.20)),
        ),
        peak=PeakConfig(
            absolute_mass_floor=float(peak_raw.get("absolute_mass_floor", 0.002)),
            relative_floor=float(peak_raw.get("relative_floor", 0.10)),
            primary_merge_radius=int(peak_raw.get("primary_merge_radius", 24)),
            gt_x1_neighborhood_radius=int(peak_raw.get("gt_x1_neighborhood_radius", 24)),
            raw_topk_k=int(peak_raw.get("raw_topk_k", 32)),
        ),
    )
    _validate_config(config)
    return config


def _load_checkpoints(raw: Mapping[str, Any]) -> dict[str, CheckpointConfig]:
    if len(raw) < 2:
        raise ValueError("checkpoints must include at least two roles")
    checkpoints: dict[str, CheckpointConfig] = {}
    for role, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoints.{role} must be a mapping")
        checkpoints[str(role)] = CheckpointConfig(
            checkpoint_path=_required_path(value, "checkpoint_path"),
            objective_policy=str(value.get("objective_policy", "")),
            training_ordering=str(value.get("training_ordering", "")),
            template_contract_id=str(value.get("template_contract_id", "")),
            comparison_group=str(value.get("comparison_group", "")),
        )
    return checkpoints


def _validate_config(config: PrefixStateTransitionConfig) -> None:
    if not config.stages:
        raise ValueError("stages must not be empty")
    if config.sampling.max_prefix_states <= 0:
        raise ValueError("sampling.max_prefix_states must be positive")
    if config.sampling.num_shards <= 0:
        raise ValueError("sampling.num_shards must be positive")
    if not 0.0 <= config.sampling.easy_sanity_max_fraction <= 1.0:
        raise ValueError("sampling.easy_sanity_max_fraction must be between 0 and 1")
    if config.peak.absolute_mass_floor < 0.0:
        raise ValueError("peak.absolute_mass_floor must be non-negative")
    if not 0.0 <= config.peak.relative_floor <= 1.0:
        raise ValueError("peak.relative_floor must be between 0 and 1")
    if config.peak.primary_merge_radius < 0:
        raise ValueError("peak.primary_merge_radius must be non-negative")
    if config.peak.gt_x1_neighborhood_radius < 0:
        raise ValueError("peak.gt_x1_neighborhood_radius must be non-negative")
    if config.peak.raw_topk_k <= 0:
        raise ValueError("peak.raw_topk_k must be positive")


def _required(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or value == "":
        raise ValueError(f"missing config key: {key}")
    return str(value)


def _required_path(raw: Mapping[str, Any], key: str) -> Path:
    path = Path(_required(raw, key))
    if not path.is_absolute():
        raise ValueError(f"{key} must be an absolute path")
    return path


def _optional_path(raw: Mapping[str, Any], key: str) -> Path | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"{key} must be an absolute path")
    return path


def _mapping_or_empty(value: Any, key: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a sequence")
    return tuple(str(item) for item in value)
