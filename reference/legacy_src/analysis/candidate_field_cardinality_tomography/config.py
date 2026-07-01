from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import PROJECT_ID


KNOWN_STAGES = {
    "case_index",
    "probe_plan",
    "x1_candidate_field",
    "residual_row_scoring",
    "basin_attraction",
    "attention_components",
    "taxonomy",
    "merge",
    "report",
    "gallery",
    "validate",
}


@dataclass(frozen=True)
class PeakConfig:
    absolute_mass_floor: float = 0.002
    relative_floor: float = 0.10
    primary_merge_radius: int = 24
    gt_x1_neighborhood_radius: int = 24
    raw_topk_k: int = 32


@dataclass(frozen=True)
class SamplingConfig:
    num_shards: int = 8
    max_cases: int | None = None
    seed: int = 3664


@dataclass(frozen=True)
class PolicyConfig:
    desc_normalization_policy_id: str = "lower_strip_collapse_ws_v1"
    row_score_policy_id: str = "residual_row_mean_v1"
    decode_policy_id: str = "greedy_y1x2y2_v1"


@dataclass(frozen=True)
class CandidateFieldConfig:
    project_id: str
    artifact_root: Path
    checkpoint_path: Path
    train_jsonl: Path
    val_jsonl: Path
    fn_rescue_overlay_root: Path | None
    phase5_overlay_root: Path | None
    stages: tuple[str, ...]
    peak: PeakConfig
    sampling: SamplingConfig
    policies: PolicyConfig


def load_config(path: str | Path) -> CandidateFieldConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("candidate-field config must be a mapping")
    project_id = str(raw.get("project_id") or "")
    if project_id != PROJECT_ID:
        raise ValueError(f"project_id must be {PROJECT_ID}")
    stages = _string_tuple(raw.get("stages", ("case_index", "probe_plan", "validate")), "stages")
    unknown = sorted(set(stages) - KNOWN_STAGES)
    if unknown:
        raise ValueError(f"unknown stage: {unknown[0]}")

    peak_raw = _mapping_or_empty(raw.get("peak"), "peak")
    sampling_raw = _mapping_or_empty(raw.get("sampling"), "sampling")
    policies_raw = _mapping_or_empty(raw.get("policies"), "policies")
    return CandidateFieldConfig(
        project_id=project_id,
        artifact_root=Path(_required(raw, "artifact_root")),
        checkpoint_path=Path(_required(raw, "checkpoint_path")),
        train_jsonl=Path(_required(raw, "train_jsonl")),
        val_jsonl=Path(_required(raw, "val_jsonl")),
        fn_rescue_overlay_root=_optional_path(raw.get("fn_rescue_overlay_root")),
        phase5_overlay_root=_optional_path(raw.get("phase5_overlay_root")),
        stages=stages,
        peak=PeakConfig(
            absolute_mass_floor=float(peak_raw.get("absolute_mass_floor", 0.002)),
            relative_floor=float(peak_raw.get("relative_floor", 0.10)),
            primary_merge_radius=int(peak_raw.get("primary_merge_radius", 24)),
            gt_x1_neighborhood_radius=int(peak_raw.get("gt_x1_neighborhood_radius", 24)),
            raw_topk_k=int(peak_raw.get("raw_topk_k", 32)),
        ),
        sampling=SamplingConfig(
            num_shards=int(sampling_raw.get("num_shards", 8)),
            max_cases=(
                None if sampling_raw.get("max_cases") is None else int(sampling_raw["max_cases"])
            ),
            seed=int(sampling_raw.get("seed", 3664)),
        ),
        policies=PolicyConfig(
            desc_normalization_policy_id=str(
                policies_raw.get("desc_normalization_policy_id", "lower_strip_collapse_ws_v1")
            ),
            row_score_policy_id=str(policies_raw.get("row_score_policy_id", "residual_row_mean_v1")),
            decode_policy_id=str(policies_raw.get("decode_policy_id", "greedy_y1x2y2_v1")),
        ),
    )


def _required(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or value == "":
        raise ValueError(f"missing config key: {key}")
    return str(value)


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


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
