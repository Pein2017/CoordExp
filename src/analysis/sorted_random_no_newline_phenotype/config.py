from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import PHASE_ID, POLICY_OBJECTIVE_RUN_IDS, PROJECT_ID, RUN_ID, SCHEMA_VERSION


LEGACY_CHECKPOINT_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
)
EXPECTED_CHECKPOINT_ROLES = LEGACY_CHECKPOINT_ROLES


@dataclass(frozen=True)
class TemplateContractConfig:
    detection_sequence_format: str
    coordinate_surface: str
    bbox_format: str
    row_separator: str


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_path: Path
    training_ordering: str
    readout_prompt_ordering: str
    objective_policy: str = ""
    comparison_group: str = ""
    template_contract_id: str = ""


@dataclass(frozen=True)
class SamplingConfig:
    max_prefix_states: int = 4096
    num_shards: int = 8
    seed: int = 3668
    easy_sanity_max_fraction: float = 0.20


@dataclass(frozen=True)
class RolloutConfig:
    limit_images: int = 1024
    decode_policy: str = "free_text_unconstrained_greedy_temp0"
    native_prompt_ordering: bool = True
    constraint_policy: str = "none"


@dataclass(frozen=True)
class FNProbeConfig:
    max_fn_objects_per_checkpoint: int = 512
    hint_policy_id: str = "desc_x1_r95_ladder_v1"
    strict_r95_axis_fraction: float = 0.04
    strict_r95_cap_bins: int = 8
    broad_x1_radius: int = 24


@dataclass(frozen=True)
class PeakConfig:
    absolute_mass_floor: float = 0.002
    relative_floor: float = 0.10
    primary_merge_radius: int = 24
    gt_x1_neighborhood_radius: int = 24
    raw_topk_k: int = 32


@dataclass(frozen=True)
class A32Config:
    project_id: str
    phase_id: str
    schema_version: str
    run_id: str
    artifact_root: Path
    train_jsonl: Path
    val_jsonl: Path
    image_root: Path
    checkpoints: dict[str, CheckpointConfig]
    template_contract: TemplateContractConfig
    sampling: SamplingConfig
    rollout: RolloutConfig
    fn_probe: FNProbeConfig
    peak: PeakConfig


def load_config(path: str | Path) -> A32Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("A3.2 config must be a mapping")

    project_id = _required(raw, "project_id")
    phase_id = _required(raw, "phase_id")
    schema_version = _required(raw, "schema_version")
    run_id = _required(raw, "run_id")
    if project_id != PROJECT_ID:
        raise ValueError(f"project_id must be {PROJECT_ID}")
    if phase_id != PHASE_ID:
        raise ValueError(f"phase_id must be {PHASE_ID}")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    accepted_run_ids = (RUN_ID, *POLICY_OBJECTIVE_RUN_IDS)
    if run_id not in accepted_run_ids:
        raise ValueError(f"run_id must be one of {', '.join(accepted_run_ids)}")

    template_raw = _required_mapping(raw, "template_contract")
    sampling_raw = _mapping_or_empty(raw.get("sampling"), "sampling")
    rollout_raw = _mapping_or_empty(raw.get("rollout"), "rollout")
    fn_probe_raw = _mapping_or_empty(raw.get("fn_probe"), "fn_probe")
    peak_raw = _mapping_or_empty(raw.get("peak"), "peak")
    checkpoints = _load_checkpoints(_required_mapping(raw, "checkpoints"))

    config = A32Config(
        project_id=project_id,
        phase_id=phase_id,
        schema_version=schema_version,
        run_id=run_id,
        artifact_root=_required_path(raw, "artifact_root"),
        train_jsonl=_required_path(raw, "train_jsonl"),
        val_jsonl=_required_path(raw, "val_jsonl"),
        image_root=_required_path(raw, "image_root"),
        checkpoints=checkpoints,
        template_contract=TemplateContractConfig(
            detection_sequence_format=_required(
                template_raw,
                "detection_sequence_format",
            ),
            coordinate_surface=_required(template_raw, "coordinate_surface"),
            bbox_format=_required(template_raw, "bbox_format"),
            row_separator=_required(template_raw, "row_separator"),
        ),
        sampling=SamplingConfig(
            max_prefix_states=int(sampling_raw.get("max_prefix_states", 4096)),
            num_shards=int(sampling_raw.get("num_shards", 8)),
            seed=int(sampling_raw.get("seed", 3668)),
            easy_sanity_max_fraction=float(
                sampling_raw.get("easy_sanity_max_fraction", 0.20)
            ),
        ),
        rollout=RolloutConfig(
            limit_images=int(rollout_raw.get("limit_images", 1024)),
            decode_policy=str(
                rollout_raw.get(
                    "decode_policy",
                    "free_text_unconstrained_greedy_temp0",
                )
            ),
            native_prompt_ordering=bool(
                rollout_raw.get("native_prompt_ordering", True)
            ),
            constraint_policy=str(rollout_raw.get("constraint_policy", "none")),
        ),
        fn_probe=FNProbeConfig(
            max_fn_objects_per_checkpoint=int(
                fn_probe_raw.get("max_fn_objects_per_checkpoint", 512)
            ),
            hint_policy_id=str(
                fn_probe_raw.get("hint_policy_id", "desc_x1_r95_ladder_v1")
            ),
            strict_r95_axis_fraction=float(
                fn_probe_raw.get("strict_r95_axis_fraction", 0.04)
            ),
            strict_r95_cap_bins=int(fn_probe_raw.get("strict_r95_cap_bins", 8)),
            broad_x1_radius=int(fn_probe_raw.get("broad_x1_radius", 24)),
        ),
        peak=PeakConfig(
            absolute_mass_floor=float(peak_raw.get("absolute_mass_floor", 0.002)),
            relative_floor=float(peak_raw.get("relative_floor", 0.10)),
            primary_merge_radius=int(peak_raw.get("primary_merge_radius", 24)),
            gt_x1_neighborhood_radius=int(
                peak_raw.get("gt_x1_neighborhood_radius", 24)
            ),
            raw_topk_k=int(peak_raw.get("raw_topk_k", 32)),
        ),
    )
    _validate_config(config)
    return config


def _load_checkpoints(raw: Mapping[str, Any]) -> dict[str, CheckpointConfig]:
    roles = tuple(raw)
    if len(roles) < 2:
        raise ValueError(
            "checkpoint roles must be exactly a configured comparison cohort "
            "or include at least two checkpoint roles"
        )
    if len(set(str(role) for role in roles)) != len(roles):
        raise ValueError("checkpoint roles must be unique")

    checkpoints: dict[str, CheckpointConfig] = {}
    for role in roles:
        value = raw[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoints.{role} must be a mapping")
        role_text = str(role)
        checkpoints[role_text] = CheckpointConfig(
            checkpoint_path=_required_path(value, "checkpoint_path"),
            training_ordering=_required(value, "training_ordering"),
            readout_prompt_ordering=_required(value, "readout_prompt_ordering"),
            objective_policy=str(value.get("objective_policy", "")),
            comparison_group=str(value.get("comparison_group", "")),
            template_contract_id=str(value.get("template_contract_id", "")),
        )
    return checkpoints


def _validate_config(config: A32Config) -> None:
    template = config.template_contract
    if template.row_separator != "none":
        raise ValueError("template_contract.row_separator must be none")
    if template.detection_sequence_format != "compact_full":
        raise ValueError("template_contract.detection_sequence_format must be compact_full")
    if template.coordinate_surface != "coord_token":
        raise ValueError("template_contract.coordinate_surface must be coord_token")
    if template.bbox_format != "xyxy":
        raise ValueError("template_contract.bbox_format must be xyxy")
    if "len12000" in config.image_root.name:
        raise ValueError("image_root must not point to len12000")
    if config.sampling.max_prefix_states <= 0:
        raise ValueError("sampling.max_prefix_states must be positive")
    if config.sampling.num_shards != 8:
        raise ValueError("sampling.num_shards must be 8")
    for role, checkpoint in config.checkpoints.items():
        if (
            checkpoint.template_contract_id
            and checkpoint.template_contract_id != "compact_full_no_newline_native_v1"
        ):
            raise ValueError(
                f"{role} template_contract_id must be compact_full_no_newline_native_v1"
            )
        if not checkpoint.checkpoint_path.is_dir():
            raise ValueError(
                f"checkpoints.{role}.checkpoint_path must exist and be a directory: "
                f"{checkpoint.checkpoint_path}"
            )
    if not 0.0 <= config.sampling.easy_sanity_max_fraction <= 1.0:
        raise ValueError(
            "sampling.easy_sanity_max_fraction must be between 0 and 1"
        )
    if config.rollout.limit_images <= 0:
        raise ValueError("rollout.limit_images must be positive")
    if config.rollout.decode_policy != "free_text_unconstrained_greedy_temp0":
        raise ValueError(
            "rollout.decode_policy must be free_text_unconstrained_greedy_temp0"
        )
    if config.rollout.constraint_policy != "none":
        raise ValueError("rollout.constraint_policy must be none")
    if config.fn_probe.max_fn_objects_per_checkpoint <= 0:
        raise ValueError("fn_probe.max_fn_objects_per_checkpoint must be positive")
    if config.fn_probe.strict_r95_axis_fraction <= 0.0:
        raise ValueError("fn_probe.strict_r95_axis_fraction must be positive")
    if config.fn_probe.strict_r95_cap_bins <= 0:
        raise ValueError("fn_probe.strict_r95_cap_bins must be positive")
    if config.fn_probe.broad_x1_radius < 0:
        raise ValueError("fn_probe.broad_x1_radius must be non-negative")
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


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None:
        raise ValueError(f"missing config key: {key}")
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mapping_or_empty(value: Any, key: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value
