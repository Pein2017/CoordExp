from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import (
    CHECKPOINT_ROLES,
    ET_RMP_ROLE,
    FULL_RUN_ID,
    PHASE_ID,
    PROJECT_ID,
    PURE_CE_ROLES,
    SCHEMA_VERSION,
    SMOKE_RUN_ID,
)


@dataclass(frozen=True)
class TemplateContractConfig:
    template_contract_id: str
    detection_sequence_format: str
    coordinate_surface: str
    bbox_format: str
    row_separator: str
    contract_provenance: str


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool
    checkpoint_path: Path
    training_ordering: str
    comparison_role: str
    controlled_comparison_group: str
    chat_template_variant: str
    template_contract: TemplateContractConfig


@dataclass(frozen=True)
class CaseSamplingConfig:
    max_images: int
    max_target_instances: int
    min_same_desc_count: int
    min_object_count: int
    easy_sanity_max_fraction: float
    seed: int
    splits: tuple[str, ...]
    split_quotas: dict[str, int]
    desc_cap_per_split: int
    desc_count_caps: dict[str, int]
    object_count_buckets: tuple[str, ...]


@dataclass(frozen=True)
class PrefixConfig:
    prefix_modes_requested: tuple[str, ...]
    rollout_prefix_source_jsonl: Path | None
    rollout_prefix_missing_policy_smoke: str
    rollout_prefix_missing_policy_full: str


@dataclass(frozen=True)
class PosteriorConfig:
    strict_r95_axis_fraction: float
    strict_r95_cap_bins: int
    peak_mass_floor: float
    low_margin_threshold: float
    coord_mass_low_threshold: float
    r95_anchor_policy: str


@dataclass(frozen=True)
class GreedyConfig:
    enabled: bool
    sample_fraction: float
    decode_policy: str
    constraint_policy: str


@dataclass(frozen=True)
class RuntimeConfig:
    num_shards: int
    max_new_tokens: int
    torch_dtype: str
    device_map: str


@dataclass(frozen=True)
class A33Config:
    project_id: str
    phase_id: str
    schema_version: str
    run_id: str
    artifact_root: Path
    train_jsonl: Path
    val_jsonl: Path
    image_root: Path
    checkpoints: dict[str, CheckpointConfig]
    case_sampling: CaseSamplingConfig
    prefix: PrefixConfig
    posterior: PosteriorConfig
    greedy: GreedyConfig
    runtime: RuntimeConfig


def load_config(path: str | Path, *, validate_paths: bool = True) -> A33Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("A3.3 config must be a mapping")

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
    if run_id not in {SMOKE_RUN_ID, FULL_RUN_ID}:
        raise ValueError(f"run_id must be {SMOKE_RUN_ID} or {FULL_RUN_ID}")

    config = A33Config(
        project_id=project_id,
        phase_id=phase_id,
        schema_version=schema_version,
        run_id=run_id,
        artifact_root=_required_path(raw, "artifact_root"),
        train_jsonl=_required_path(raw, "train_jsonl"),
        val_jsonl=_required_path(raw, "val_jsonl"),
        image_root=_required_path(raw, "image_root"),
        checkpoints=_load_checkpoints(_required_mapping(raw, "checkpoints")),
        case_sampling=_load_case_sampling(
            _required_mapping(raw, "case_sampling")
        ),
        prefix=_load_prefix(_required_mapping(raw, "prefix")),
        posterior=_load_posterior(_required_mapping(raw, "posterior")),
        greedy=_load_greedy(_required_mapping(raw, "greedy")),
        runtime=_load_runtime(_required_mapping(raw, "runtime")),
    )
    _validate_config(config, raw, validate_paths=validate_paths)
    return config


def _load_checkpoints(raw: Mapping[str, Any]) -> dict[str, CheckpointConfig]:
    roles = tuple(str(role) for role in raw)
    if roles != CHECKPOINT_ROLES:
        expected = ", ".join(CHECKPOINT_ROLES)
        actual = ", ".join(roles) or "<none>"
        raise ValueError(
            f"checkpoint roles must be ordered exactly as {expected}; got {actual}"
        )

    checkpoints: dict[str, CheckpointConfig] = {}
    for role in CHECKPOINT_ROLES:
        value = raw[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoints.{role} must be a mapping")
        template_raw = _required_mapping(value, "template_contract")
        checkpoints[role] = CheckpointConfig(
            enabled=bool(value.get("enabled", True)),
            checkpoint_path=_required_path(value, "checkpoint_path"),
            training_ordering=_required(value, "training_ordering"),
            comparison_role=_required(value, "comparison_role"),
            controlled_comparison_group=_required(
                value,
                "controlled_comparison_group",
            ),
            chat_template_variant=_required(value, "chat_template_variant"),
            template_contract=TemplateContractConfig(
                template_contract_id=_required(
                    template_raw,
                    "template_contract_id",
                ),
                detection_sequence_format=_required(
                    template_raw,
                    "detection_sequence_format",
                ),
                coordinate_surface=_required(template_raw, "coordinate_surface"),
                bbox_format=_required(template_raw, "bbox_format"),
                row_separator=_required(template_raw, "row_separator"),
                contract_provenance=_required(
                    template_raw,
                    "contract_provenance",
                ),
            ),
        )
    return checkpoints


def _load_case_sampling(raw: Mapping[str, Any]) -> CaseSamplingConfig:
    splits = _string_tuple(_required_sequence(raw, "splits"), "case_sampling.splits")
    desc_cap_per_split = int(raw.get("desc_cap_per_split", 0))
    desc_count_caps_raw = raw.get("desc_count_caps")
    if desc_count_caps_raw is None:
        desc_count_caps = {split: desc_cap_per_split for split in splits}
    else:
        desc_count_caps = _int_dict(
            _required_mapping(raw, "desc_count_caps"),
            "case_sampling.desc_count_caps",
        )
    return CaseSamplingConfig(
        max_images=int(_required(raw, "max_images")),
        max_target_instances=int(_required(raw, "max_target_instances")),
        min_same_desc_count=int(_required(raw, "min_same_desc_count")),
        min_object_count=int(_required(raw, "min_object_count")),
        easy_sanity_max_fraction=float(
            _required(raw, "easy_sanity_max_fraction")
        ),
        seed=int(_required(raw, "seed")),
        splits=splits,
        split_quotas=_int_dict(
            _required_mapping(raw, "split_quotas"),
            "case_sampling.split_quotas",
        ),
        desc_cap_per_split=desc_cap_per_split,
        desc_count_caps=desc_count_caps,
        object_count_buckets=_string_tuple(
            _required_sequence(raw, "object_count_buckets"),
            "case_sampling.object_count_buckets",
        ),
    )


def _load_prefix(raw: Mapping[str, Any]) -> PrefixConfig:
    modes_raw = raw.get("prefix_modes_requested", raw.get("modes"))
    if modes_raw is None:
        raise ValueError("missing config key: prefix.prefix_modes_requested")
    return PrefixConfig(
        prefix_modes_requested=_string_tuple(
            modes_raw,
            "prefix.prefix_modes_requested",
        ),
        rollout_prefix_source_jsonl=_optional_path(
            raw,
            "rollout_prefix_source_jsonl",
        ),
        rollout_prefix_missing_policy_smoke=_required(
            raw,
            "rollout_prefix_missing_policy_smoke",
        ),
        rollout_prefix_missing_policy_full=_required(
            raw,
            "rollout_prefix_missing_policy_full",
        ),
    )


def _load_posterior(raw: Mapping[str, Any]) -> PosteriorConfig:
    return PosteriorConfig(
        strict_r95_axis_fraction=float(_required(raw, "strict_r95_axis_fraction")),
        strict_r95_cap_bins=int(_required(raw, "strict_r95_cap_bins")),
        peak_mass_floor=float(_required(raw, "peak_mass_floor")),
        low_margin_threshold=float(_required(raw, "low_margin_threshold")),
        coord_mass_low_threshold=float(_required(raw, "coord_mass_low_threshold")),
        r95_anchor_policy=_required(raw, "r95_anchor_policy"),
    )


def _load_greedy(raw: Mapping[str, Any]) -> GreedyConfig:
    return GreedyConfig(
        enabled=bool(raw.get("enabled", True)),
        sample_fraction=float(_required(raw, "sample_fraction")),
        decode_policy=_required(raw, "decode_policy"),
        constraint_policy=_required(raw, "constraint_policy"),
    )


def _load_runtime(raw: Mapping[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        num_shards=int(_required(raw, "num_shards")),
        max_new_tokens=int(_required(raw, "max_new_tokens")),
        torch_dtype=_required(raw, "torch_dtype"),
        device_map=_required(raw, "device_map"),
    )


def _validate_config(
    config: A33Config,
    raw: Mapping[str, Any],
    *,
    validate_paths: bool,
) -> None:
    if "template_contract" in raw:
        raise ValueError(
            "A3.3 requires per-checkpoint template_contract, not a global "
            "template_contract"
        )
    if tuple(config.checkpoints) != CHECKPOINT_ROLES:
        raise ValueError("checkpoint roles must be ordered exactly as CHECKPOINT_ROLES")
    if "len12000" in config.image_root.name:
        raise ValueError("image_root must not point to the len12000 JSONL directory")

    _validate_case_sampling(config.case_sampling)
    _validate_prefix(config.prefix)
    _validate_posterior(config.posterior)
    _validate_greedy(config.greedy)
    _validate_runtime(config.runtime)
    _validate_checkpoints(config.checkpoints, validate_paths=validate_paths)


def _validate_checkpoints(
    checkpoints: Mapping[str, CheckpointConfig],
    *,
    validate_paths: bool,
) -> None:
    for role, checkpoint in checkpoints.items():
        contract = checkpoint.template_contract
        if contract.detection_sequence_format != "compact_full":
            raise ValueError(f"{role} detection_sequence_format must be compact_full")
        if contract.coordinate_surface != "coord_token":
            raise ValueError(f"{role} coordinate_surface must be coord_token")
        if contract.bbox_format != "xyxy":
            raise ValueError(f"{role} bbox_format must be xyxy")
        if checkpoint.chat_template_variant != contract.template_contract_id:
            raise ValueError(
                f"{role} chat_template_variant must match template_contract_id"
            )

        if role == ET_RMP_ROLE:
            if checkpoint.comparison_role != "reference_anchor":
                raise ValueError(f"{role} comparison_role must be reference_anchor")
            if checkpoint.controlled_comparison_group != "reference_anchor_not_controlled":
                raise ValueError(
                    "et_rmp_ce_ckpt3664 must be labeled as a non-controlled "
                    "reference anchor"
                )
            if contract.template_contract_id != "compact_full_newline_native_v1":
                raise ValueError(
                    "et_rmp_ce_ckpt3664 template_contract_id must be "
                    "compact_full_newline_native_v1"
                )
            if contract.row_separator != "newline":
                raise ValueError(
                    "et_rmp_ce_ckpt3664 template_contract.row_separator must be newline"
                )
            if contract.contract_provenance != "legacy_compact_full_default_inferred":
                raise ValueError(
                    "et_rmp_ce_ckpt3664 contract_provenance must be "
                    "legacy_compact_full_default_inferred"
                )
        elif role in PURE_CE_ROLES:
            if checkpoint.comparison_role != "clean_pair":
                raise ValueError(f"{role} comparison_role must be clean_pair")
            if (
                checkpoint.controlled_comparison_group
                != "pure_ce_sorted_vs_random_no_newline"
            ):
                raise ValueError(
                    f"{role} must stay in the pure-CE controlled comparison group"
                )
            if contract.template_contract_id != "compact_full_no_newline_native_v1":
                raise ValueError(
                    f"{role} template_contract_id must be "
                    "compact_full_no_newline_native_v1"
                )
            if contract.row_separator != "none":
                raise ValueError(f"{role} template_contract.row_separator must be none")
            if contract.contract_provenance != "user_reported_training_contract":
                raise ValueError(
                    f"{role} contract_provenance must be "
                    "user_reported_training_contract"
                )
        else:
            raise ValueError(f"unknown checkpoint role: {role}")

        if validate_paths and not checkpoint.checkpoint_path.is_dir():
            raise ValueError(
                f"{role} checkpoint_path must exist and be a directory: "
                f"{checkpoint.checkpoint_path}"
            )


def _validate_case_sampling(config: CaseSamplingConfig) -> None:
    if config.max_images <= 0:
        raise ValueError("case_sampling.max_images must be positive")
    if config.max_target_instances <= 0:
        raise ValueError("case_sampling.max_target_instances must be positive")
    if config.min_same_desc_count < 2:
        raise ValueError("case_sampling.min_same_desc_count must be at least 2")
    if config.min_object_count <= 0:
        raise ValueError("case_sampling.min_object_count must be positive")
    if not 0.0 <= config.easy_sanity_max_fraction <= 1.0:
        raise ValueError(
            "case_sampling.easy_sanity_max_fraction must be between 0 and 1"
        )
    if not config.splits:
        raise ValueError("case_sampling.splits must not be empty")
    if set(config.split_quotas) != set(config.splits):
        raise ValueError(
            "case_sampling.split_quotas must include all configured splits"
        )
    if any(value < 0 for value in config.split_quotas.values()):
        raise ValueError("case_sampling.split_quotas values must be non-negative")
    if config.desc_cap_per_split <= 0:
        raise ValueError("case_sampling.desc_cap_per_split must be positive")
    if set(config.desc_count_caps) != set(config.splits):
        raise ValueError(
            "case_sampling.desc_count_caps must include all configured splits"
        )
    if any(value <= 0 for value in config.desc_count_caps.values()):
        raise ValueError("case_sampling.desc_count_caps values must be positive")
    if not config.object_count_buckets:
        raise ValueError("case_sampling.object_count_buckets must not be empty")


def _validate_prefix(config: PrefixConfig) -> None:
    if not config.prefix_modes_requested:
        raise ValueError("prefix.prefix_modes_requested must not be empty")
    if config.rollout_prefix_missing_policy_smoke != "skip_with_manifest":
        raise ValueError(
            "prefix.rollout_prefix_missing_policy_smoke must be skip_with_manifest"
        )
    if config.rollout_prefix_missing_policy_full != "fail":
        raise ValueError("prefix.rollout_prefix_missing_policy_full must be fail")


def _validate_posterior(config: PosteriorConfig) -> None:
    if config.strict_r95_axis_fraction <= 0.0:
        raise ValueError("posterior.strict_r95_axis_fraction must be positive")
    if config.strict_r95_cap_bins <= 0:
        raise ValueError("posterior.strict_r95_cap_bins must be positive")
    if config.peak_mass_floor < 0.0:
        raise ValueError("posterior.peak_mass_floor must be non-negative")
    if config.low_margin_threshold < 0.0:
        raise ValueError("posterior.low_margin_threshold must be non-negative")
    if config.coord_mass_low_threshold < 0.0:
        raise ValueError(
            "posterior.coord_mass_low_threshold must be non-negative"
        )
    if config.r95_anchor_policy != "target_axis_fraction_cap":
        raise ValueError(
            "posterior.r95_anchor_policy must be target_axis_fraction_cap"
        )


def _validate_greedy(config: GreedyConfig) -> None:
    if not 0.0 <= config.sample_fraction <= 1.0:
        raise ValueError("greedy.sample_fraction must be between 0 and 1")
    if config.decode_policy != "free_text_unconstrained_greedy_temp0":
        raise ValueError(
            "greedy.decode_policy must be free_text_unconstrained_greedy_temp0"
        )
    if config.constraint_policy != "none":
        raise ValueError("greedy.constraint_policy must be none")


def _validate_runtime(config: RuntimeConfig) -> None:
    if config.num_shards != 8:
        raise ValueError("runtime.num_shards must be 8")
    if config.max_new_tokens <= 0:
        raise ValueError("runtime.max_new_tokens must be positive")
    if config.torch_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("runtime.torch_dtype must be bfloat16, float16, or float32")
    if config.device_map != "single_gpu":
        raise ValueError("runtime.device_map must be single_gpu")


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


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None:
        raise ValueError(f"missing config key: {key}")
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_sequence(raw: Mapping[str, Any], key: str) -> Any:
    value = raw.get(key)
    if value is None:
        raise ValueError(f"missing config key: {key}")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a sequence")
    return tuple(str(item) for item in value)


def _int_dict(value: Mapping[str, Any], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        result[str(raw_key)] = int(raw_value)
    if not result:
        raise ValueError(f"{key} must not be empty")
    return result
