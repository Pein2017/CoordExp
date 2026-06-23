"""Typed configuration schemas for training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
import math
from pathlib import Path
from typing import (
    Any,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    cast,
)

from src.common.object_field_order import (
    ObjectFieldOrder,
    ObjectOrdering,
    normalize_object_field_order,
    normalize_object_ordering,
)
from src.common.detection_sequence import (
    COMPACT_FULL_FORMAT,
    COORDJSON_FORMAT,
    normalize_detection_sequence_format,
)
from src.common.geometry.bbox_parameterization import (
    AllowedBBoxFormat,
    DEFAULT_BBOX_FORMAT,
    normalize_bbox_format,
)
from src.tokens.roles import (
    TokenRole,
    TokenRoleSets,
    normalize_token_role,
    unique_stable_ids,
)
from src.tokens.qwen_native import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    COORD_END_TOKEN,
    COORD_START_TOKEN,
    EXPECTED_BOX_END_ID,
    EXPECTED_BOX_START_ID,
    EXPECTED_COORD_END_ID,
    EXPECTED_COORD_START_ID,
    EXPECTED_OBJECT_REF_END_ID,
    EXPECTED_OBJECT_REF_START_ID,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.detection.template_contracts import (
    COMPACT_TEMPLATE_IDS,
    SUPPORTED_DETECTION_TEMPLATE_IDS,
    resolve_detection_template_contract,
)
from src.trainers.teacher_forcing.module_registry import (
    ALLOWED_DIAGNOSTIC_MODULES,
    ALLOWED_OBJECTIVE_MODULES,
    DIAGNOSTIC_CONFIG_ALLOWLIST,
    OBJECTIVE_APPLICATION_PRESET_ALLOWLIST,
    OBJECTIVE_CONFIG_ALLOWLIST,
    OBJECTIVE_OPTIONAL_CONFIG_KEYS,
)
from src.training.stage2.rollout_codec import (
    resolve_stage2_rollout_template_policy,
)
from src.objective_ids import (
    LEGACY_TEACHER_FORCING_OBJECTIVE_ID,
    RESEARCH_TEACHER_FORCING_OBJECTIVE_ID,
    STANDARD_CE_OBJECTIVE_ID,
    TEACHER_FORCING_OBJECTIVE_ID,
)

from .eval_monitor_dump_schema import EvalMonitorDumpConfig
from .rollout_matching_schema import RolloutMatchingConfig
from .strict_dataclass import dataclass_asdict_no_none, parse_dataclass_strict


AllowedNorm = Literal["none", "norm100", "norm1000"]
AllowedVisualDistance = Literal["mse", "cosine"]
AllowedJsonFormat = Literal["standard"]

ALLOWED_JSON_FORMATS: set[str] = {"standard"}
STAGE2_ROLLOUT_CORRECTION_FP_POLICIES: set[str] = {
    "zero_loss_context",
    "weak_positive_context",
}
STAGE2_TRIE_CE_MODULE_NAME = "stage2_trie_ce"
STAGE2_TRIE_CE_APPLICATION_PRESETS: set[str] = {"rollout_trie_hard_ce"}
STAGE2_SCHEMA_FORMAT_CE_MODULE_NAME = "schema_format_ce"
STAGE2_RESIDUAL_SET_MODULE_NAME = "residual_set_correction"
STAGE2_RESIDUAL_SET_APPLICATION_PRESETS: set[str] = {"rollout_self_prefix"}
STAGE2_RESIDUAL_TRIE_MODULE_NAMES: set[str] = {
    STAGE2_TRIE_CE_MODULE_NAME,
    STAGE2_RESIDUAL_SET_MODULE_NAME,
}
STAGE2_RESIDUAL_SET_DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "expected_num_rollouts": 4,
    "base_seed": 17,
    "lambda_type": 1.0,
    "lambda_inner": 1.0,
    "fallback_loss_weight": 1.0,
    "lambda_ul_promoted": 0.5,
    "label_conflict_weight": 0.25,
    "commit_iou_threshold": 0.75,
    "duplicate_burst_iou_threshold": 0.95,
    "duplicate_burst_prefix_rollback": False,
    "ul_cluster_iou_threshold": 0.9,
    "ul_gray_iou_low": 0.30,
    "ul_consensus_ratio": 1.0,
    "min_ul_valid_rollouts": 4,
    "clean_gt_sft_mix": 0,
    "strict_builder_invariants": True,
}
STAGE2_RESIDUAL_SET_REQUIRED_CONFIG_KEYS: set[str] = set()
STAGE2_RESIDUAL_SET_OPTIONAL_CONFIG_KEYS: set[str] = set()
STAGE2_RESIDUAL_SET_CONFIG_KEYS: set[str] = (
    STAGE2_RESIDUAL_SET_REQUIRED_CONFIG_KEYS
    | STAGE2_RESIDUAL_SET_OPTIONAL_CONFIG_KEYS
    | set(STAGE2_RESIDUAL_SET_DEFAULT_CONFIG_VALUES)
)
STAGE2_RESIDUAL_SET_POSITIVE_INT_CONFIG_KEYS: set[str] = {
    "expected_num_rollouts",
    "min_ul_valid_rollouts",
}
STAGE2_RESIDUAL_SET_NONNEGATIVE_FLOAT_CONFIG_KEYS: set[str] = {
    "lambda_type",
    "lambda_inner",
    "fallback_loss_weight",
    "lambda_ul_promoted",
    "label_conflict_weight",
}
STAGE2_RESIDUAL_SET_THRESHOLD_CONFIG_KEYS: set[str] = {
    "commit_iou_threshold",
    "duplicate_burst_iou_threshold",
    "ul_cluster_iou_threshold",
    "ul_gray_iou_low",
    "ul_consensus_ratio",
}
STAGE2_RESIDUAL_SET_BOOL_CONFIG_KEYS: set[str] = {
    "strict_builder_invariants",
    "duplicate_burst_prefix_rollback",
}
TEACHER_FORCING_PROFILES: set[str] = {
    "hard_sft",
    "pure_valid_set_marginal",
    "hybrid_valid_set_marginal",
}
LEGACY_TEACHER_FORCING_OBJECTIVE_IDS: set[str] = {
    LEGACY_TEACHER_FORCING_OBJECTIVE_ID,
    "recursive_detection_ce",
    "random_permutation_et_rmp_ce",
    "prefix_rollin_et_rmp_ce",
    "ET_RMP_CE",
    "et_rmp_like",
    "typed_trie_alpha0_random_rollin",
    "typed_trie_alpha0p1_random_rollin",
    "support_balance",
}
LEGACY_STAGE2_TEACHER_FORCING_MODULES: set[str] = {
    "bbox_geo",
    "bbox_size_aux",
    "coord_reg",
    "soft_ce",
    "w1",
    "token_ce",
    "coord_gate",
    "text_gate",
}
LEGACY_STAGE2_TEACHER_FORCING_CONFIG_KEYS: set[str] = {
    "coord_gate",
    "coord_gate_weight",
    "text_gate",
    "text_gate_weight",
    "soft_ce_weight",
    "w1_weight",
}


def _normalize_json_format(value: Any) -> AllowedJsonFormat:
    if not isinstance(value, str):
        raise TypeError("custom.json_format must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ALLOWED_JSON_FORMATS:
        raise ValueError("custom.json_format must be 'standard'")
    return cast(AllowedJsonFormat, normalized)


def _as_dict(value: Any, *, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(value)!r}")
    return value


def _validate_section_keys_strict(
    section: str, payload: Mapping[str, Any], *, allowed: set[str]
) -> None:
    """Fail fast on unknown keys within a top-level config section.

    This enforces schema-derived strictness for sections that are ultimately
    flattened into ms-swift TrainArguments / RLHFArguments keyword arguments.

    Unknown keys are reported as dotted paths (e.g., `training.foo`).
    """

    if not payload:
        return

    unknown: list[Any] = []
    for k in payload.keys():
        if not isinstance(k, str) or k not in allowed:
            unknown.append(k)

    if unknown:
        rendered = [
            f"{section}.{str(k)}" for k in sorted(unknown, key=lambda x: str(x))
        ]
        raise ValueError(f"Unknown {section} keys: {rendered}")


def _is_versioned_alias_for(name: str, canonical: str) -> bool:
    normalized = name.strip().lower()
    canonical = canonical.strip().lower()
    if normalized == canonical:
        return False

    suffix_prefix = f"{canonical}_v"
    if (
        normalized.startswith(suffix_prefix)
        and normalized[len(suffix_prefix) :].isdigit()
    ):
        return True

    prefix_suffix = f"_{canonical}"
    if normalized.startswith("v") and normalized.endswith(prefix_suffix):
        version = normalized[1 : -len(prefix_suffix)]
        return version.isdigit()

    return False


def _stage2_residual_set_config_path(key: str) -> str:
    return (
        "stage2_rollout_correction.pipeline.objective[name=residual_set_correction]"
        f".config.{key}"
    )


def _coerce_stage2_residual_set_positive_int(value: Any, *, key: str) -> int:
    path = _stage2_residual_set_config_path(key)
    if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return int(value)


def _coerce_stage2_residual_set_nonnegative_float(value: Any, *, key: str) -> float:
    path = _stage2_residual_set_config_path(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if result < 0.0:
        raise ValueError(f"{path} must be nonnegative")
    return result


def _coerce_stage2_residual_set_threshold(value: Any, *, key: str) -> float:
    path = _stage2_residual_set_config_path(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{path} must be in [0, 1]")
    return result


def _validate_stage2_residual_set_config(
    config: MutableMapping[str, Any],
) -> None:
    for key in sorted(STAGE2_RESIDUAL_SET_POSITIVE_INT_CONFIG_KEYS):
        config[key] = _coerce_stage2_residual_set_positive_int(
            config.get(key),
            key=key,
        )

    base_seed = config.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError(
            f"{_stage2_residual_set_config_path('base_seed')} must be an integer"
        )
    config["base_seed"] = int(base_seed)

    for key in sorted(STAGE2_RESIDUAL_SET_NONNEGATIVE_FLOAT_CONFIG_KEYS):
        config[key] = _coerce_stage2_residual_set_nonnegative_float(
            config.get(key),
            key=key,
        )

    for key in sorted(STAGE2_RESIDUAL_SET_THRESHOLD_CONFIG_KEYS):
        config[key] = _coerce_stage2_residual_set_threshold(
            config.get(key),
            key=key,
        )

    if float(config["ul_consensus_ratio"]) != 1.0:
        raise ValueError(
            f"{_stage2_residual_set_config_path('ul_consensus_ratio')} must be 1.0 "
            "because mine_ul_consensus currently supports only consensus_ratio == 1.0"
        )
    if float(config["ul_gray_iou_low"]) > float(config["ul_cluster_iou_threshold"]):
        raise ValueError(
            f"{_stage2_residual_set_config_path('ul_gray_iou_low')} must be <= "
            f"{_stage2_residual_set_config_path('ul_cluster_iou_threshold')}"
        )

    clean_gt_sft_mix = config.get("clean_gt_sft_mix")
    if isinstance(clean_gt_sft_mix, bool) or not isinstance(clean_gt_sft_mix, int):
        raise ValueError(
            f"{_stage2_residual_set_config_path('clean_gt_sft_mix')} must be an integer"
        )
    if int(clean_gt_sft_mix) < 0:
        raise ValueError(
            f"{_stage2_residual_set_config_path('clean_gt_sft_mix')} must be nonnegative"
        )
    config["clean_gt_sft_mix"] = int(clean_gt_sft_mix)

    for key in sorted(STAGE2_RESIDUAL_SET_BOOL_CONFIG_KEYS):
        if not isinstance(config.get(key), bool):
            raise TypeError(f"{_stage2_residual_set_config_path(key)} must be bool")


@lru_cache(maxsize=1)
def _train_arguments_allowed_keys() -> set[str]:
    # Schema-derived strict key acceptance for ms-swift TrainArguments-driven sections.
    try:
        from swift.llm.argument import TrainArguments
    except ImportError:
        from swift.arguments import SftArguments as TrainArguments

    allowed = {f.name for f in fields(TrainArguments)}
    if "tuner_type" in allowed:
        allowed.add("train_type")
    return allowed


_TRAINING_INTERNAL_KEYS: set[str] = {
    # CoordExp-only training knobs (not ms-swift args).
    "effective_batch_size",
    "save_delay_steps",
    "save_delay_epochs",
    "save_last_epoch",
    "output_root",
    "logging_root",
    "artifact_subdir",
    # Packing-only knobs consumed by our runner (not ms-swift args).
    "packing",
    "packing_mode",
    "packing_buffer",
    "packing_min_fill_ratio",
    "packing_drop_last",
    "packing_allow_single_long",
    "eval_packing",
    "packing_avg_samples",
    "packing_wait_timeout_s",
    "packing_length_cache_persist_every",
    "packing_length_precompute_workers",
    "encoded_sample_cache",
    "static_packing_cache",
    "save_model_only",
}


@lru_cache(maxsize=1)
def _training_allowed_keys() -> set[str]:
    return set(_train_arguments_allowed_keys()) | set(_TRAINING_INTERNAL_KEYS)


def _validate_training_checkpoint_keys(data: Mapping[str, Any]) -> None:
    if "save_only_model" in data:
        raise ValueError(
            "training.save_only_model is an upstream/internal knob and is unsupported "
            "in CoordExp YAML. Use training.save_model_only=true for restartable "
            "checkpoints or false for inference-only checkpoints."
        )
    if "checkpoint_mode" in data:
        raise ValueError(
            "training.checkpoint_mode is deprecated and unsupported in CoordExp YAML. "
            "Use training.save_model_only=true for restartable checkpoints or false "
            "for inference-only checkpoints."
        )
    if "save_model_only" in data and not isinstance(data.get("save_model_only"), bool):
        raise ValueError(
            "training.save_model_only must be a boolean true/false value, "
            f"got {data.get('save_model_only')!r}"
        )


@lru_cache(maxsize=1)
def _rlhf_arguments_allowed_keys() -> set[str]:
    try:
        from swift.llm.argument import RLHFArguments
    except ImportError:
        from swift.arguments import RLHFArguments

    allowed = {f.name for f in fields(RLHFArguments)}
    # Local knob (popped before RLHFArguments/TrainArguments init).
    allowed.add("llm_kd_weight")
    return allowed


@dataclass(frozen=True)
class TokenTypeMetricsConfig:
    enabled: bool = False
    include: tuple[str, ...] = ("lvis",)
    exclude: tuple[str, ...] = ()
    # Metric compute knobs (diagnostics-only; does not affect training loss)
    log_top5: bool = True
    coord_monitor_mass: bool = True
    # Optional deterministic downsampling cap for expensive coord-vocab mass diagnostics.
    # 0 means "no cap" (compute on all supervised tokens).
    coord_monitor_mass_max_tokens: int = 0

    def __post_init__(self) -> None:
        inc = tuple(str(v).strip().lower() for v in self.include)
        exc = tuple(str(v).strip().lower() for v in self.exclude)
        object.__setattr__(self, "include", inc)
        object.__setattr__(self, "exclude", exc)
        max_tokens = int(self.coord_monitor_mass_max_tokens or 0)
        object.__setattr__(self, "coord_monitor_mass_max_tokens", max(0, max_tokens))

    @classmethod
    def from_mapping(cls, payload: Any) -> "TokenTypeMetricsConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.token_type_metrics must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        include_raw = payload.get("include", cls.include)
        exclude_raw = payload.get("exclude", cls.exclude)
        log_top5 = bool(payload.get("log_top5", cls.log_top5))
        coord_monitor_mass = bool(
            payload.get("coord_monitor_mass", cls.coord_monitor_mass)
        )
        coord_monitor_mass_max_tokens_raw = payload.get(
            "coord_monitor_mass_max_tokens", cls.coord_monitor_mass_max_tokens
        )
        try:
            coord_monitor_mass_max_tokens = int(coord_monitor_mass_max_tokens_raw or 0)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "custom.token_type_metrics.coord_monitor_mass_max_tokens must be an int"
            ) from exc

        def _to_tuple(value: Any) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, (list, tuple)):
                return tuple(str(v).strip() for v in value)
            return (str(value).strip(),)

        include = _to_tuple(include_raw)
        exclude = _to_tuple(exclude_raw)

        return cls(
            enabled=enabled,
            include=include,
            exclude=exclude,
            log_top5=log_top5,
            coord_monitor_mass=coord_monitor_mass,
            coord_monitor_mass_max_tokens=coord_monitor_mass_max_tokens,
        )


@dataclass(frozen=True)
class CoordTokensConfig:
    enabled: bool = True
    skip_bbox_norm: bool = True

    def __post_init__(self) -> None:
        if not self.skip_bbox_norm:
            raise ValueError(
                "Pre-normalized geometry contract: custom.coord_tokens.skip_bbox_norm must be true to avoid double normalization."
            )

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CoordTokensConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("coord_tokens section must be a mapping when provided")

        enabled = bool(payload.get("enabled", True))
        skip_bbox_norm = bool(payload.get("skip_bbox_norm", True))
        return cls(
            enabled=enabled,
            skip_bbox_norm=skip_bbox_norm,
        )


@dataclass(frozen=True)
class CoordSoftCEW1Config:
    """Coord-token supervision: CE (optional) + softCE(Gaussian) + W1(CDF) + coord-vocab gate."""

    enabled: bool = False
    # Optional hard CE (on coord-only logits, at coord-token positions).
    ce_weight: float = 0.0
    soft_ce_weight: float = 1.0
    w1_weight: float = 1.0
    gate_weight: float = 1.0
    text_gate_weight: float = 0.0
    temperature: float = 1.0
    target_sigma: float = 2.0
    target_truncate: Optional[int] = None

    @classmethod
    def from_mapping(
        cls,
        payload: Optional[Mapping[str, Any]],
        *,
        path: str = "coord_soft_ce_w1",
    ) -> "CoordSoftCEW1Config":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path} section must be a mapping when provided")

        allowed_keys = {
            "enabled",
            "ce_weight",
            "soft_ce_weight",
            "w1_weight",
            "gate_weight",
            "text_gate_weight",
            "temperature",
            "target_sigma",
            "target_truncate",
        }
        unknown = sorted(str(k) for k in payload.keys() if str(k) not in allowed_keys)
        if unknown:
            raise ValueError(
                f"Unknown {path} keys: {[f'{path}.{key}' for key in unknown]}"
            )

        enabled = bool(payload.get("enabled", False))

        def _parse_float(key: str, default: float) -> float:
            raw = payload.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}.{key} must be numeric") from exc

        ce_weight = _parse_float("ce_weight", cls.ce_weight)
        soft_ce_weight = _parse_float("soft_ce_weight", cls.soft_ce_weight)
        w1_weight = _parse_float("w1_weight", cls.w1_weight)
        gate_weight = _parse_float("gate_weight", cls.gate_weight)
        text_gate_weight = _parse_float("text_gate_weight", cls.text_gate_weight)
        temperature = _parse_float("temperature", cls.temperature)
        target_sigma = _parse_float("target_sigma", cls.target_sigma)

        target_truncate_raw = payload.get("target_truncate", cls.target_truncate)
        target_truncate: Optional[int]
        if target_truncate_raw is None:
            target_truncate = None
        else:
            try:
                target_truncate = int(target_truncate_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}.target_truncate must be an integer or null"
                ) from exc

        if ce_weight < 0:
            raise ValueError(f"{path}.ce_weight must be >= 0")
        if soft_ce_weight < 0:
            raise ValueError(f"{path}.soft_ce_weight must be >= 0")
        if w1_weight < 0:
            raise ValueError(f"{path}.w1_weight must be >= 0")
        if gate_weight < 0:
            raise ValueError(f"{path}.gate_weight must be >= 0")
        if text_gate_weight < 0:
            raise ValueError(f"{path}.text_gate_weight must be >= 0")
        if (
            enabled
            and ce_weight == 0
            and soft_ce_weight == 0
            and w1_weight == 0
            and gate_weight == 0
            and text_gate_weight == 0
        ):
            raise ValueError(
                f"{path} is enabled but ce_weight, soft_ce_weight, w1_weight, gate_weight, and text_gate_weight are all 0"
            )
        if temperature <= 0:
            raise ValueError(f"{path}.temperature must be > 0")
        if target_sigma <= 0:
            raise ValueError(f"{path}.target_sigma must be > 0")
        if target_truncate is not None and target_truncate < 0:
            raise ValueError(f"{path}.target_truncate must be >= 0 or null")

        return cls(
            enabled=enabled,
            ce_weight=ce_weight,
            soft_ce_weight=soft_ce_weight,
            w1_weight=w1_weight,
            gate_weight=gate_weight,
            text_gate_weight=text_gate_weight,
            temperature=temperature,
            target_sigma=target_sigma,
            target_truncate=target_truncate,
        )


@dataclass(frozen=True)
class BBoxGeoConfig:
    enabled: bool = False
    smoothl1_weight: float = 0.0
    ciou_weight: float = 1.0
    parameterization: str = "xyxy"
    center_weight: float = 1.0
    size_weight: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "BBoxGeoConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("bbox_geo section must be a mapping when provided")
        raise ValueError(
            "custom.bbox_geo has been removed from active training configs; "
            "use the unified teacher-forcing objective or standard hard SFT "
            "without bbox geometry auxiliaries."
        )


@dataclass(frozen=True)
class BBoxSizeAuxConfig:
    enabled: bool = False
    log_wh_weight: float = 0.05
    oversize_penalty_weight: float = 0.0
    oversize_area_frac_threshold: Optional[float] = None
    oversize_log_w_threshold: Optional[float] = None
    oversize_log_h_threshold: Optional[float] = None
    eps: float = 1e-6

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "BBoxSizeAuxConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("bbox_size_aux section must be a mapping when provided")
        raise ValueError(
            "custom.bbox_size_aux has been removed from active training configs; "
            "use the unified teacher-forcing objective or standard hard SFT "
            "without bbox size auxiliaries."
        )


@dataclass(frozen=True)
class TrainableTokenRowGroupConfig:
    role: TokenRole
    start_token: Optional[str] = None
    end_token: Optional[str] = None
    tokens: tuple[str, ...] = ()
    expected_start: Optional[int] = None
    expected_end: Optional[int] = None
    expected_ids: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, path: str
    ) -> "TrainableTokenRowGroupConfig":
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path} must be a mapping")

        role = normalize_token_role(payload.get("role"), path=f"{path}.role")
        start_token_raw = payload.get("start_token")
        end_token_raw = payload.get("end_token")
        tokens_raw = payload.get("tokens")
        has_range = start_token_raw is not None or end_token_raw is not None
        has_tokens = tokens_raw is not None
        if has_range == has_tokens:
            raise ValueError(
                f"{path} must provide exactly one of start_token/end_token or tokens"
            )

        start_token: Optional[str] = None
        end_token: Optional[str] = None
        tokens: tuple[str, ...] = ()
        if has_range:
            if not isinstance(start_token_raw, str) or not start_token_raw:
                raise TypeError(f"{path}.start_token must be a non-empty string")
            if not isinstance(end_token_raw, str) or not end_token_raw:
                raise TypeError(f"{path}.end_token must be a non-empty string")
            start_token = start_token_raw
            end_token = end_token_raw
        else:
            if not isinstance(tokens_raw, Sequence) or isinstance(
                tokens_raw, (str, bytes)
            ):
                raise TypeError(f"{path}.tokens must be a list of token strings")
            parsed_tokens = tuple(str(token) for token in tokens_raw)
            if not parsed_tokens or any(not token for token in parsed_tokens):
                raise ValueError(
                    f"{path}.tokens must contain at least one non-empty token"
                )
            tokens = parsed_tokens

        def _parse_expected_int(key: str) -> Optional[int]:
            raw = payload.get(key)
            if raw is None:
                return None
            try:
                return int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}.{key} must be an integer") from exc

        expected_start = _parse_expected_int("expected_start")
        expected_end = _parse_expected_int("expected_end")
        if has_range and ((expected_start is None) != (expected_end is None)):
            raise ValueError(
                f"{path}.expected_start and {path}.expected_end must be provided together"
            )
        if (
            has_range
            and expected_start is not None
            and expected_end is not None
            and expected_end < expected_start
        ):
            raise ValueError(f"{path}.expected_end must be >= expected_start")

        expected_ids_raw = payload.get("expected_ids", {})
        if expected_ids_raw is None:
            expected_ids: Mapping[str, int] = {}
        elif not isinstance(expected_ids_raw, Mapping):
            raise TypeError(f"{path}.expected_ids must be a mapping")
        else:
            expected_ids = {
                str(token): int(token_id)
                for token, token_id in expected_ids_raw.items()
            }

        return cls(
            role=role,
            start_token=start_token,
            end_token=end_token,
            tokens=tokens,
            expected_start=expected_start,
            expected_end=expected_end,
            expected_ids=expected_ids,
        )

    def resolve_ids(self, tokenizer: Any, *, path: str) -> tuple[int, ...]:
        if self.start_token is not None and self.end_token is not None:
            start_id = int(tokenizer.convert_tokens_to_ids(self.start_token))
            end_id = int(tokenizer.convert_tokens_to_ids(self.end_token))
            if self.expected_start is not None and start_id != self.expected_start:
                raise ValueError(
                    f"{path}.start_token {self.start_token!r} resolved to id {start_id}, "
                    f"expected id {self.expected_start}"
                )
            if self.expected_end is not None and end_id != self.expected_end:
                raise ValueError(
                    f"{path}.end_token {self.end_token!r} resolved to id {end_id}, "
                    f"expected id {self.expected_end}"
                )
            if end_id < start_id:
                raise ValueError(
                    f"{path} resolved end id {end_id} before start id {start_id}"
                )
            return tuple(range(start_id, end_id + 1))

        resolved: list[int] = []
        for token in self.tokens:
            token_id = int(tokenizer.convert_tokens_to_ids(token))
            expected = self.expected_ids.get(token)
            if expected is not None and token_id != expected:
                raise ValueError(
                    f"{path}.tokens token {token!r} resolved to id {token_id}, "
                    f"expected id {expected}"
                )
            resolved.append(token_id)
        return tuple(resolved)


@dataclass(frozen=True)
class TokenEmbeddingsAdapterConfig:
    enabled: bool = False
    tie_head: bool = True
    groups: Mapping[str, TrainableTokenRowGroupConfig] = field(default_factory=dict)
    embed_lr: Optional[float] = None
    head_lr: Optional[float] = None
    weight_decay: float = 0.0
    dtype: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weight_decay < 0:
            raise ValueError("custom.token_embeddings_adapter.weight_decay must be >= 0")

    @classmethod
    def from_mapping(
        cls,
        payload: Optional[Mapping[str, Any]],
        *,
        path: str = "custom.token_embeddings_adapter",
    ) -> "TokenEmbeddingsAdapterConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path} section must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        tie_head_raw = payload.get("tie_head", True)
        if tie_head_raw is None:
            tie_head = True
        elif isinstance(tie_head_raw, bool):
            tie_head = tie_head_raw
        else:
            raise TypeError(f"{path}.tie_head must be a boolean when provided")

        groups_raw = payload.get("groups", {})
        if not isinstance(groups_raw, Mapping):
            raise TypeError(f"{path}.groups must be a mapping when provided")
        groups = {
            str(name): TrainableTokenRowGroupConfig.from_mapping(
                group_payload,
                path=f"{path}.groups.{name}",
            )
            for name, group_payload in groups_raw.items()
        }
        if enabled and not groups:
            raise ValueError(f"{path}.enabled=true requires at least one group")

        def _parse_lr(key: str) -> Optional[float]:
            raw = payload.get(key)
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}.{key} must be numeric") from exc

        weight_decay_raw = payload.get("weight_decay", 0.0)
        try:
            weight_decay = float(weight_decay_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.weight_decay must be numeric") from exc
        if weight_decay < 0:
            raise ValueError(f"{path}.weight_decay must be >= 0")

        dtype_raw = payload.get("dtype")

        return cls(
            enabled=enabled,
            tie_head=tie_head,
            groups=groups,
            embed_lr=_parse_lr("embed_lr"),
            head_lr=_parse_lr("head_lr"),
            weight_decay=weight_decay,
            dtype=str(dtype_raw) if dtype_raw is not None else None,
        )

    def resolve_role_sets(self, tokenizer: Any) -> TokenRoleSets:
        coord_geometry: list[int] = []
        structural_ce_only: list[int] = []
        for name, group in self.groups.items():
            ids = group.resolve_ids(
                tokenizer, path=f"token_embeddings_adapter.groups.{name}"
            )
            if group.role is TokenRole.COORD_GEOMETRY:
                coord_geometry.extend(ids)
            elif group.role is TokenRole.STRUCTURAL_CE_ONLY:
                structural_ce_only.extend(ids)
            else:  # pragma: no cover - enum exhaustiveness guard
                raise ValueError(f"Unsupported token role: {group.role}")

        coord_geometry_ids = unique_stable_ids(coord_geometry)
        structural_ce_only_ids = unique_stable_ids(structural_ce_only)
        return TokenRoleSets(
            coord_geometry_ids=coord_geometry_ids,
            structural_ce_only_ids=structural_ce_only_ids,
            trainable_row_ids=unique_stable_ids(
                (*coord_geometry_ids, *structural_ce_only_ids)
            ),
            coord_loss_ids=coord_geometry_ids,
        )

    def resolve_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return self.resolve_role_sets(tokenizer).trainable_row_ids


@dataclass(frozen=True)
class PromptOverrides:
    system: Optional[str] = None
    user: Optional[str] = None
    output_variant: Literal["dense", "summary"] = "dense"


@dataclass(frozen=True)
class DeepSpeedConfig:
    enabled: bool
    config: Any

    @classmethod
    def from_mapping(
        cls, payload: Optional[Mapping[str, Any]]
    ) -> Optional["DeepSpeedConfig"]:
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("deepspeed section must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)
        if "enabled" not in data:
            raise ValueError("deepspeed.enabled must be explicitly set")

        enabled = bool(data.pop("enabled"))

        if enabled:
            if "config" not in data:
                raise ValueError(
                    "deepspeed.config must be provided when deepspeed.enabled is true"
                )
            config_value = data.pop("config")
        else:
            config_value = data.pop("config", None)

        if enabled and (config_value is None or config_value == ""):
            raise ValueError(
                "deepspeed.config must be a non-empty value when deepspeed.enabled is true"
            )

        if data:
            unknown = sorted(str(k) for k in data.keys())
            rendered = [f"deepspeed.{k}" for k in unknown]
            raise ValueError(f"Unknown deepspeed keys: {rendered}")

        return cls(enabled=enabled, config=config_value)


@dataclass(frozen=True)
class SaveDelayConfig:
    steps: Optional[int] = None
    epochs: Optional[float] = None

    @classmethod
    def from_raw(cls, steps: Any, epochs: Any) -> "SaveDelayConfig":
        parsed_steps: Optional[int] = None
        if steps is not None:
            try:
                value = int(steps)
            except (TypeError, ValueError) as exc:
                raise ValueError("save_delay_steps must be an integer") from exc
            if value > 0:
                parsed_steps = value

        parsed_epochs: Optional[float] = None
        if epochs is not None:
            try:
                value = float(epochs)
            except (TypeError, ValueError) as exc:
                raise ValueError("save_delay_epochs must be numeric") from exc
            if value > 0:
                parsed_epochs = value

        return cls(steps=parsed_steps, epochs=parsed_epochs)

    @property
    def active(self) -> bool:
        return (self.steps or 0) > 0 or (self.epochs or 0.0) > 0

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "SaveDelayConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("save_delay section must be a mapping")
        steps = payload.get("steps")
        epochs = payload.get("epochs")
        return cls.from_raw(steps, epochs)


@dataclass(frozen=True)
class EncodedSampleCacheConfig:
    enabled: bool = False
    root_dir: Optional[str] = None
    ineligible_policy: Literal["error", "bypass"] = "error"
    wait_timeout_s: int = 7200
    max_resident_shards: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("training.encoded_sample_cache.enabled must be a boolean")

        root_dir = self.root_dir
        if root_dir is not None:
            root_dir = str(root_dir).strip()
            if not root_dir:
                raise ValueError(
                    "training.encoded_sample_cache.root_dir must be a non-empty string when provided"
                )
            object.__setattr__(self, "root_dir", root_dir)

        policy = str(self.ineligible_policy or "").strip().lower()
        if policy not in {"error", "bypass"}:
            raise ValueError(
                "training.encoded_sample_cache.ineligible_policy must be one of "
                "{'error', 'bypass'}"
            )
        object.__setattr__(
            self, "ineligible_policy", cast(Literal["error", "bypass"], policy)
        )

        wait_timeout_raw = self.wait_timeout_s
        if isinstance(wait_timeout_raw, bool):
            raise TypeError(
                "training.encoded_sample_cache.wait_timeout_s must be an integer"
            )
        try:
            wait_timeout = int(wait_timeout_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "training.encoded_sample_cache.wait_timeout_s must be an integer"
            ) from exc
        if wait_timeout < 0:
            raise ValueError(
                "training.encoded_sample_cache.wait_timeout_s must be >= 0 "
                "(set 0 to wait indefinitely)"
            )
        object.__setattr__(self, "wait_timeout_s", wait_timeout)

        max_resident_raw = self.max_resident_shards
        if isinstance(max_resident_raw, bool):
            raise TypeError(
                "training.encoded_sample_cache.max_resident_shards must be an integer"
            )
        try:
            max_resident = int(max_resident_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "training.encoded_sample_cache.max_resident_shards must be an integer"
            ) from exc
        if max_resident <= 0:
            raise ValueError(
                "training.encoded_sample_cache.max_resident_shards must be > 0"
            )
        object.__setattr__(self, "max_resident_shards", max_resident)

    @classmethod
    def from_mapping(cls, payload: Any) -> "EncodedSampleCacheConfig":
        if payload is None:
            return cls()
        return parse_dataclass_strict(
            cls, payload, path="training.encoded_sample_cache"
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclass_asdict_no_none(self)


@dataclass(frozen=True)
class StaticPackingCacheConfig:
    root_dir: Optional[str] = None

    def __post_init__(self) -> None:
        root_dir = self.root_dir
        if root_dir is not None:
            root_dir = str(root_dir).strip()
            if not root_dir:
                raise ValueError(
                    "training.static_packing_cache.root_dir must be a non-empty string when provided"
                )
            object.__setattr__(self, "root_dir", root_dir)

    @classmethod
    def from_mapping(cls, payload: Any) -> "StaticPackingCacheConfig":
        if payload is None:
            return cls()
        return parse_dataclass_strict(
            cls, payload, path="training.static_packing_cache"
        )

    def to_mapping(self) -> dict[str, Any]:
        return dataclass_asdict_no_none(self)


# warnings: this is deprecated and not used
@dataclass(frozen=True)
class VisualKDTargetConfig:
    enabled: bool = False
    weight: float = 0.0
    distance: AllowedVisualDistance = "mse"

    def __post_init__(self) -> None:
        if self.enabled and self.weight <= 0:
            raise ValueError("visual_kd.*.weight must be > 0 when enabled")
        if self.distance not in {"mse", "cosine"}:
            raise ValueError("visual_kd.*.distance must be one of {mse, cosine}")


# warnings: this is deprecated and not used

_ALLOWED_VISUAL_KD_KEYS = {"enabled", "vit", "aligner", "deepstack"}
_ALLOWED_VISUAL_KD_TARGET_KEYS = {"enabled", "weight", "distance"}


@dataclass(frozen=True)
class VisualKDConfig:
    enabled: bool
    vit: VisualKDTargetConfig = field(default_factory=VisualKDTargetConfig)
    aligner: VisualKDTargetConfig = field(default_factory=VisualKDTargetConfig)
    deepstack: VisualKDTargetConfig = field(default_factory=VisualKDTargetConfig)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not (self.vit.enabled or self.aligner.enabled or self.deepstack.enabled):
            raise ValueError(
                "custom.visual_kd must enable at least one of vit/aligner/deepstack "
                "when visual_kd.enabled is true"
            )

    @classmethod
    def disabled(cls) -> "VisualKDConfig":
        return cls(enabled=False)

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "VisualKDConfig":
        if payload is None:
            return cls.disabled()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.visual_kd must be a mapping when provided")

        _validate_section_keys_strict(
            "custom.visual_kd", payload, allowed=_ALLOWED_VISUAL_KD_KEYS
        )

        def _validate_target_mapping(
            name: str, raw: Optional[Mapping[str, Any]]
        ) -> Optional[Mapping[str, Any]]:
            if raw is None:
                return None
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"custom.visual_kd.{name} must be a mapping when provided"
                )

            _validate_section_keys_strict(
                f"custom.visual_kd.{name}",
                raw,
                allowed=_ALLOWED_VISUAL_KD_TARGET_KEYS,
            )
            return raw

        # Validate nested target keys even when visual_kd.enabled is false, so
        # typos in disabled subtrees are still caught early.
        vit_raw = _validate_target_mapping("vit", payload.get("vit"))
        aligner_raw = _validate_target_mapping("aligner", payload.get("aligner"))
        deepstack_raw = _validate_target_mapping("deepstack", payload.get("deepstack"))

        enabled = bool(payload.get("enabled", False))
        if not enabled:
            return cls.disabled()

        def parse_target(
            name: str, raw: Optional[Mapping[str, Any]]
        ) -> VisualKDTargetConfig:
            if raw is None:
                return VisualKDTargetConfig()

            target_enabled = bool(raw.get("enabled", False))
            raw_weight = raw.get("weight", 0.0)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"custom.visual_kd.{name}.weight must be numeric"
                ) from exc

            raw_distance = raw.get("distance", "mse")
            if not isinstance(raw_distance, str):
                raise TypeError(f"custom.visual_kd.{name}.distance must be a string")
            distance = raw_distance.lower()

            if distance not in {"mse", "cosine"}:
                raise ValueError(
                    f"custom.visual_kd.{name}.distance must be one of {{mse, cosine}}"
                )

            return VisualKDTargetConfig(
                enabled=target_enabled,
                weight=weight,
                distance=distance,  # type: ignore[arg-type]
            )

        vit_cfg = parse_target("vit", vit_raw)
        aligner_cfg = parse_target("aligner", aligner_raw)
        deepstack_cfg = parse_target("deepstack", deepstack_raw)

        if not (vit_cfg.enabled or aligner_cfg.enabled or deepstack_cfg.enabled):
            raise ValueError(
                "custom.visual_kd.enabled is true but all per-target configs are disabled; "
                "enable at least one of vit/aligner/deepstack"
            )

        return cls(
            enabled=True,
            vit=vit_cfg,
            aligner=aligner_cfg,
            deepstack=deepstack_cfg,
        )


@dataclass(frozen=True)
class Stage1EvalDetectionConfig:
    enabled: bool = False
    metrics: str = "f1ish"  # f1ish | coco | lvis | both
    use_segm: bool = False
    strict_parse: bool = True
    iou_thrs: Optional[list[float]] = None
    lvis_max_dets: int = 300
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_threshold: float = 0.6
    semantic_device: str = "auto"
    semantic_batch_size: int = 64
    f1ish_iou_thrs: list[float] = field(default_factory=lambda: [0.3, 0.5])
    f1ish_pred_scope: str = "annotated"  # annotated | all
    score_mode: str = "constant"  # constant | confidence_postop
    pred_score_source: str = "stage1_eval_constant"
    pred_score_version: int = 1
    constant_score: float = 1.0
    batch_size: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    limit: Optional[int] = None
    distributed: bool = True
    lvis_annotations_json: Optional[str] = None

    def __post_init__(self) -> None:
        metrics_mode = str(self.metrics or "").strip().lower()
        if metrics_mode not in {"f1ish", "coco", "lvis", "both"}:
            raise ValueError(
                "custom.eval_detection.metrics must be one of {'f1ish', 'coco', 'lvis', 'both'}"
            )
        if self.iou_thrs is not None:
            object.__setattr__(
                self, "iou_thrs", [float(value) for value in list(self.iou_thrs)]
            )
        object.__setattr__(
            self,
            "f1ish_iou_thrs",
            [float(value) for value in list(self.f1ish_iou_thrs)],
        )
        if str(self.f1ish_pred_scope or "").strip().lower() not in {
            "annotated",
            "all",
        }:
            raise ValueError(
                "custom.eval_detection.f1ish_pred_scope must be one of {'annotated', 'all'}"
            )
        score_mode = str(self.score_mode or "").strip().lower()
        if score_mode not in {"constant", "confidence_postop"}:
            raise ValueError(
                "custom.eval_detection.score_mode must be one of {'constant', 'confidence_postop'}"
            )
        object.__setattr__(self, "score_mode", score_mode)
        if int(self.lvis_max_dets) <= 0:
            raise ValueError("custom.eval_detection.lvis_max_dets must be > 0")
        if int(self.semantic_batch_size) <= 0:
            raise ValueError("custom.eval_detection.semantic_batch_size must be > 0")
        if int(self.batch_size) <= 0:
            raise ValueError("custom.eval_detection.batch_size must be > 0")
        if int(self.max_new_tokens) <= 0:
            raise ValueError("custom.eval_detection.max_new_tokens must be > 0")
        if not (0.0 <= float(self.constant_score) <= 1.0):
            raise ValueError(
                "custom.eval_detection.constant_score must satisfy 0.0 <= score <= 1.0"
            )
        if not str(self.pred_score_source or "").strip():
            raise ValueError(
                "custom.eval_detection.pred_score_source must be non-empty"
            )
        if int(self.pred_score_version) <= 0:
            raise ValueError("custom.eval_detection.pred_score_version must be > 0")
        if self.limit is not None and int(self.limit) <= 0:
            raise ValueError("custom.eval_detection.limit must be > 0 when provided")
        if (
            self.lvis_annotations_json is not None
            and not str(self.lvis_annotations_json).strip()
        ):
            raise ValueError(
                "custom.eval_detection.lvis_annotations_json must be non-empty when provided"
            )

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage1EvalDetectionConfig":
        if payload is None:
            return cls()
        return parse_dataclass_strict(cls, payload, path="custom.eval_detection")


@dataclass(frozen=True)
class Stage1SFTStructuralCloseConfig:
    enabled: bool = False
    final_close_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("custom.sft_structural_close.enabled must be a boolean")
        weight = float(self.final_close_weight)
        if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
            raise ValueError(
                "custom.sft_structural_close.final_close_weight must satisfy 0 <= weight <= 1"
            )
        object.__setattr__(self, "final_close_weight", weight)

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage1SFTStructuralCloseConfig":
        if payload is None:
            return cls()
        return parse_dataclass_strict(
            cls,
            payload,
            path="custom.sft_structural_close",
        )


@dataclass(frozen=True)
class CustomConfig:
    train_jsonl: str
    user_prompt: str
    emit_norm: AllowedNorm
    json_format: AllowedJsonFormat
    object_field_order: ObjectFieldOrder
    bbox_format: AllowedBBoxFormat = DEFAULT_BBOX_FORMAT
    object_ordering: ObjectOrdering = "sorted"
    detection_sequence_format: str = COORDJSON_FORMAT
    detection_template_id: Optional[str] = None
    coord_tokens: CoordTokensConfig = field(default_factory=CoordTokensConfig)
    token_embeddings_adapter: TokenEmbeddingsAdapterConfig = field(
        default_factory=TokenEmbeddingsAdapterConfig
    )
    coord_soft_ce_w1: CoordSoftCEW1Config = field(default_factory=CoordSoftCEW1Config)
    bbox_geo: BBoxGeoConfig = field(default_factory=BBoxGeoConfig)
    bbox_size_aux: BBoxSizeAuxConfig = field(default_factory=BBoxSizeAuxConfig)
    sft_structural_close: Stage1SFTStructuralCloseConfig = field(
        default_factory=Stage1SFTStructuralCloseConfig
    )
    use_summary: bool = False
    system_prompt_summary: Optional[str] = None
    augmentation: Optional[Mapping[str, Any]] = None
    augmentation_curriculum: Optional[Mapping[str, Any]] = None
    bypass_prob: float = 0.0
    trainer_variant: Optional[str] = None
    train_sample_limit: Optional[Any] = None
    val_sample_limit: Optional[Any] = None
    val_sample_with_replacement: bool = False
    eval_monitor_dump: EvalMonitorDumpConfig = field(
        default_factory=EvalMonitorDumpConfig
    )
    dump_conversation_text: bool = False
    dump_conversation_path: Optional[str] = None
    val_jsonl: Optional[str] = None
    offline_max_pixels: Optional[int] = None
    eval_detection: Stage1EvalDetectionConfig = field(
        default_factory=Stage1EvalDetectionConfig
    )
    output_variant: Literal["dense", "summary"] = "dense"
    visual_kd: VisualKDConfig = field(default_factory=VisualKDConfig.disabled)
    token_type_metrics: TokenTypeMetricsConfig = field(
        default_factory=TokenTypeMetricsConfig
    )
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train_jsonl:
            raise ValueError("custom.train_jsonl must be provided")
        if not self.user_prompt:
            raise ValueError("custom.user_prompt must be provided")
        if self.emit_norm != "none":
            raise ValueError(
                "Pre-normalized data is required; set custom.emit_norm: none (runtime normalization is disabled)."
            )
        normalize_object_ordering(self.object_ordering, path="custom.object_ordering")
        if self.object_field_order not in {"desc_first", "geometry_first"}:
            raise ValueError(
                "custom.object_field_order must be one of {'desc_first', 'geometry_first'}"
            )
        normalize_bbox_format(self.bbox_format, path="custom.bbox_format")
        if not isinstance(self.use_summary, bool):
            raise TypeError("custom.use_summary must be a boolean value")
        if not isinstance(self.val_sample_with_replacement, bool):
            raise TypeError(
                "custom.val_sample_with_replacement must be a boolean value"
            )
        if self.json_format not in ALLOWED_JSON_FORMATS:
            raise ValueError("custom.json_format must be 'standard'")
        normalize_detection_sequence_format(self.detection_sequence_format)
        if self.detection_template_id is not None:
            resolve_detection_template_contract(self.detection_template_id)
        normalize_bbox_format(self.bbox_format, path="custom.bbox_format")
        if self.offline_max_pixels is not None and int(self.offline_max_pixels) <= 0:
            raise ValueError("custom.offline_max_pixels must be > 0 when provided")
        # NOTE: Coord tokens can be supervised either via distribution losses
        # (custom.coord_soft_ce_w1) or via the base CE objective (ablations).
        # NOTE: We intentionally do not validate val_sample_with_replacement sizing here
        # because runtime may override sample limits (e.g. via debug.*). The runner
        # performs the strict check after resolving the active sample-limit namespace.

    @classmethod
    def from_mapping(
        cls, payload: Optional[Mapping[str, Any]], *, prompts: PromptOverrides
    ) -> "CustomConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("custom section must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)
        if "sample_limit" in data:
            raise ValueError(
                "custom.sample_limit has been removed. "
                "Use custom.train_sample_limit and/or custom.val_sample_limit instead."
            )
        train_jsonl = data.pop("train_jsonl", data.pop("jsonl", None))
        user_prompt = data.pop("user_prompt", prompts.user)
        emit_norm = data.pop("emit_norm", None)

        if isinstance(user_prompt, str) and user_prompt.endswith(".txt"):
            path = Path(user_prompt)
            if path.is_file():
                user_prompt = path.read_text(encoding="utf-8").strip("\n")

        if "summary_ratio" in data:
            raise ValueError(
                "custom.summary_ratio has been removed; use custom.use_summary instead."
            )

        def _parse_bool(value: Any, field_name: str) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                if value in (0, 1, 0.0, 1.0):
                    return bool(value)
                raise ValueError(
                    f"{field_name} must be boolean (0 or 1), got {value!r}."
                )
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError(
                    f"{field_name} string value '{value}' is not a recognized boolean representation."
                )
            raise TypeError(
                f"{field_name} must be a boolean value, got {type(value)!r}."
            )

        use_summary_raw = data.pop("use_summary", None)
        val_sample_with_replacement_raw = data.pop("val_sample_with_replacement", False)
        use_summary = (
            False
            if use_summary_raw is None
            else _parse_bool(use_summary_raw, "custom.use_summary")
        )
        val_sample_with_replacement = _parse_bool(
            val_sample_with_replacement_raw, "custom.val_sample_with_replacement"
        )
        coord_tokens_raw = data.pop("coord_tokens", None)
        system_prompt_summary = data.pop("system_prompt_summary", None)
        if "images_per_user_turn" in data:
            raise ValueError(
                "custom.images_per_user_turn is no longer supported; remove the field to use single-image turns."
            )
        augmentation = data.pop("augmentation", None)
        augmentation_curriculum = data.pop("augmentation_curriculum", None)
        bypass_prob = float(data.pop("bypass_prob", 0.0))
        trainer_variant = data.pop("trainer_variant", None)
        if str(trainer_variant or "") == "stage1_set_continuation":
            raise ValueError(
                "custom.trainer_variant=stage1_set_continuation has been removed; "
                "use prefix_rollin_et_rmp_ce under the latest compact detection schema"
            )
        train_sample_limit = data.pop("train_sample_limit", None)
        val_sample_limit = data.pop("val_sample_limit", None)
        eval_monitor_dump_raw = data.pop("eval_monitor_dump", None)
        dump_conversation_text = bool(data.pop("dump_conversation_text", False))
        dump_conversation_path = data.pop("dump_conversation_path", None)
        object_ordering_raw = data.pop("object_ordering", "sorted")
        object_field_order_raw = data.pop("object_field_order", None)
        if object_field_order_raw is None:
            raise ValueError("custom.object_field_order must be provided")
        val_jsonl = data.pop("val_jsonl", None)
        offline_max_pixels_raw = data.pop("offline_max_pixels", None)
        if offline_max_pixels_raw in (None, "", False):
            offline_max_pixels = None
        else:
            if isinstance(offline_max_pixels_raw, bool):
                raise TypeError(
                    "custom.offline_max_pixels must be an int when provided"
                )
            try:
                offline_max_pixels = int(offline_max_pixels_raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "custom.offline_max_pixels must be an int when provided"
                ) from exc
            if offline_max_pixels <= 0:
                raise ValueError("custom.offline_max_pixels must be > 0 when provided")
        if "fusion_config" in data:
            raise ValueError(
                "custom.fusion_config has been removed. "
                "CoordExp supports offline-prepared single-dataset JSONL training configs; "
                "merge JSONLs offline before training when dataset mixing is needed."
            )
        if eval_monitor_dump_raw is None:
            eval_monitor_dump = EvalMonitorDumpConfig()
        else:
            eval_monitor_dump = parse_dataclass_strict(
                EvalMonitorDumpConfig,
                eval_monitor_dump_raw,
                path="custom.eval_monitor_dump",
            )
        visual_kd_raw = data.pop("visual_kd", None)
        visual_kd = VisualKDConfig.from_mapping(visual_kd_raw)
        hsm_raw = data.pop("hard_sample_mining", None)
        if hsm_raw is not None:
            raise ValueError(
                "custom.hard_sample_mining is deprecated and unsupported. Remove this section to continue."
            )
        token_type_metrics_raw = data.pop("token_type_metrics", None)
        token_type_metrics = TokenTypeMetricsConfig.from_mapping(token_type_metrics_raw)
        if "coord_expectation_metrics" in data:
            raise ValueError(
                "custom.coord_expectation_metrics has been removed. "
                "Decoded-coordinate diagnostics (expectation/argmax) are not supported; "
                "use the distribution-based coord losses/logs instead."
            )
        json_format_raw = data.pop("json_format", None)
        if json_format_raw is None:
            raise ValueError("custom.json_format must be provided")
        json_format = _normalize_json_format(json_format_raw)
        bbox_format = normalize_bbox_format(
            data.pop("bbox_format", DEFAULT_BBOX_FORMAT),
            path="custom.bbox_format",
        )
        detection_sequence_format = normalize_detection_sequence_format(
            data.pop("detection_sequence_format", COORDJSON_FORMAT)
        )
        detection_template_id_raw = data.pop("detection_template_id", None)
        detection_template_id = (
            None if detection_template_id_raw is None else str(detection_template_id_raw)
        )
        detection_template_contract = (
            None
            if detection_template_id is None
            else resolve_detection_template_contract(detection_template_id)
        )

        # `custom.extra` is the only intentional extension bucket.
        nested_extra_raw = data.pop("extra", None)
        if nested_extra_raw is None:
            nested_extra: Mapping[str, Any] = {}
        elif not isinstance(nested_extra_raw, Mapping):
            raise TypeError("custom.extra must be a mapping when provided")
        else:
            nested_extra = dict(nested_extra_raw)

        if "rollout_matching" in nested_extra:
            raise ValueError(
                "custom.extra.rollout_matching is unsupported. "
                "Move rollout settings to top-level rollout_matching.*."
            )

        if emit_norm is None:
            raise ValueError("custom.emit_norm must be provided")
        if not isinstance(emit_norm, str):
            raise TypeError("custom.emit_norm must be a string")
        emit_norm_value = emit_norm.strip()
        if emit_norm_value != "none":
            raise ValueError(
                "Pre-normalized data is required; set custom.emit_norm: none (runtime normalization is disabled)."
            )

        coord_tokens = CoordTokensConfig.from_mapping(coord_tokens_raw)
        if (
            (
                detection_sequence_format != COORDJSON_FORMAT
                or (
                    detection_template_contract is not None
                    and detection_template_contract.is_compact
                )
            )
            and not coord_tokens.enabled
        ):
            raise ValueError(
                "compact detection rendering requires custom.coord_tokens.enabled=true"
            )
        if "coord_offset" in data:
            raise ValueError(
                "custom.coord_offset has been removed; use "
                "custom.token_embeddings_adapter instead."
            )
        if "trainable_token_rows" in data:
            raise ValueError(
                "custom.trainable_token_rows has been removed; use "
                "custom.token_embeddings_adapter instead."
            )
        token_embeddings_adapter_raw = data.pop("token_embeddings_adapter", None)
        token_embeddings_adapter = TokenEmbeddingsAdapterConfig.from_mapping(
            token_embeddings_adapter_raw
        )
        if "coord_loss" in data:
            raise ValueError(
                "custom.coord_loss is no longer supported; use custom.coord_soft_ce_w1 "
                "for legacy Stage-1 SFT losses or latest objective.* for DetectionTrainingConfig."
            )
        coord_soft_ce_w1_raw = data.pop("coord_soft_ce_w1", None)
        coord_soft_ce_w1 = CoordSoftCEW1Config.from_mapping(coord_soft_ce_w1_raw)
        bbox_geo_raw = data.pop("bbox_geo", None)
        bbox_geo = BBoxGeoConfig.from_mapping(bbox_geo_raw)
        bbox_size_aux_raw = data.pop("bbox_size_aux", None)
        bbox_size_aux = BBoxSizeAuxConfig.from_mapping(bbox_size_aux_raw)
        sft_structural_close_raw = data.pop("sft_structural_close", None)
        sft_structural_close = Stage1SFTStructuralCloseConfig.from_mapping(
            sft_structural_close_raw
        )
        eval_detection_raw = data.pop("eval_detection", None)
        eval_detection = Stage1EvalDetectionConfig.from_mapping(eval_detection_raw)
        if "stage1_set_continuation" in data:
            raise ValueError(
                "custom.stage1_set_continuation has been removed; "
                "use prefix_rollin_et_rmp_ce under the latest compact detection schema"
            )

        object_ordering = normalize_object_ordering(
            object_ordering_raw,
            path="custom.object_ordering",
        )
        object_field_order = normalize_object_field_order(
            object_field_order_raw, path="custom.object_field_order"
        )

        if data:
            unknown = sorted(str(k) for k in data.keys())
            rendered = [f"custom.{k}" for k in unknown]
            raise ValueError(f"Unknown custom keys: {rendered}")

        return cls(
            train_jsonl=str(train_jsonl) if train_jsonl is not None else "",
            user_prompt=str(user_prompt) if user_prompt is not None else "",
            emit_norm=cast("AllowedNorm", emit_norm_value),
            json_format=json_format,
            object_field_order=object_field_order,
            bbox_format=bbox_format,
            object_ordering=object_ordering,
            detection_sequence_format=detection_sequence_format,
            detection_template_id=detection_template_id,
            coord_tokens=coord_tokens,
            token_embeddings_adapter=token_embeddings_adapter,
            coord_soft_ce_w1=coord_soft_ce_w1,
            bbox_geo=bbox_geo,
            bbox_size_aux=bbox_size_aux,
            sft_structural_close=sft_structural_close,
            use_summary=use_summary,
            system_prompt_summary=system_prompt_summary,
            augmentation=augmentation
            if isinstance(augmentation, Mapping)
            else augmentation,
            augmentation_curriculum=augmentation_curriculum
            if isinstance(augmentation_curriculum, Mapping)
            else augmentation_curriculum,
            bypass_prob=bypass_prob,
            trainer_variant=str(trainer_variant)
            if trainer_variant is not None
            else None,
            train_sample_limit=train_sample_limit,
            val_sample_limit=val_sample_limit,
            val_sample_with_replacement=val_sample_with_replacement,
            eval_monitor_dump=eval_monitor_dump,
            dump_conversation_text=dump_conversation_text,
            dump_conversation_path=str(dump_conversation_path)
            if dump_conversation_path is not None
            else None,
            val_jsonl=str(val_jsonl) if val_jsonl is not None else None,
            offline_max_pixels=offline_max_pixels,
            eval_detection=eval_detection,
            output_variant=prompts.output_variant,
            visual_kd=visual_kd,
            token_type_metrics=token_type_metrics,
            extra=dict(nested_extra),
        )


@dataclass(frozen=True)
class DebugConfig:
    """Optional debug overrides (e.g. tiny JSONLs for smoke tests).

    This is intentionally separate from `custom.*` so we can grow debug-only knobs
    without polluting the dataset contract.
    """

    enabled: bool = False
    # When set, overrides both training.output_dir and training.logging_dir so that
    # checkpoints + tensorboard logs land in the same folder (easy cleanup).
    output_dir: Optional[str] = None
    # Optional: override dataset sampling for smoke tests (does NOT change dataset paths).
    # When debug.enabled=true, these replace custom.{train,val}_sample_limit in the runner.
    train_sample_limit: Optional[Any] = None
    val_sample_limit: Optional[Any] = None

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "DebugConfig":
        if payload is None:
            return cls(enabled=False)
        if not isinstance(payload, Mapping):
            raise TypeError("debug section must be a mapping when provided")

        data: MutableMapping[str, Any] = dict(payload)
        # Hard error on removed keys to avoid silently ignoring old configs.
        if "train_jsonl" in data or "val_jsonl" in data:
            raise ValueError(
                "debug.train_jsonl/debug.val_jsonl have been removed. "
                "Use custom.train_jsonl/custom.val_jsonl for dataset paths, and "
                "debug.train_sample_limit/debug.val_sample_limit for smoke-test sizing."
            )

        def _parse_bool(value: Any, field_name: str) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                if value in (0, 1, 0.0, 1.0):
                    return bool(value)
                raise ValueError(
                    f"{field_name} must be boolean (0 or 1), got {value!r}."
                )
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError(
                    f"{field_name} string value '{value}' is not a recognized boolean representation."
                )
            raise TypeError(
                f"{field_name} must be a boolean value, got {type(value)!r}."
            )

        enabled_raw = data.pop("enabled", False)
        enabled = _parse_bool(enabled_raw, "debug.enabled")

        output_dir_raw = data.pop("output_dir", None)
        output_dir = (
            None if output_dir_raw in (None, "", False) else str(output_dir_raw)
        )

        train_sample_limit = data.pop("train_sample_limit", None)
        val_sample_limit = data.pop("val_sample_limit", None)

        if data:
            unknown = sorted(str(k) for k in data.keys())
            rendered = [f"debug.{k}" for k in unknown]
            raise ValueError(f"Unknown debug keys: {rendered}")

        return cls(
            enabled=enabled,
            output_dir=output_dir,
            train_sample_limit=train_sample_limit,
            val_sample_limit=val_sample_limit,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Structured, human-authored run context for retrospective analysis."""

    title: Optional[str] = None
    purpose: Optional[str] = None
    hypothesis: Optional[str] = None
    baseline: Optional[str] = None
    key_deviations: tuple[str, ...] = ()
    runtime_settings: tuple[str, ...] = ()
    comments: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @staticmethod
    def _coerce_optional_text(value: Any, field_name: str) -> Optional[str]:
        if value in (None, "", False):
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string when provided")
        text = value.strip()
        return text or None

    @staticmethod
    def _coerce_text_list(value: Any, field_name: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items: Sequence[Any] = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = value
        else:
            raise TypeError(
                f"{field_name} must be a string or a list of strings when provided"
            )

        normalized: list[str] = []
        for idx, item in enumerate(items):
            if not isinstance(item, str):
                raise TypeError(f"{field_name}[{idx}] must be a string")
            text = item.strip()
            if not text:
                raise ValueError(f"{field_name}[{idx}] must not be empty")
            normalized.append(text)
        return tuple(normalized)

    def has_authored_content(self) -> bool:
        return any(
            (
                self.title,
                self.purpose,
                self.hypothesis,
                self.baseline,
                self.key_deviations,
                self.runtime_settings,
                self.comments,
                self.tags,
            )
        )

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.title is not None:
            out["title"] = self.title
        if self.purpose is not None:
            out["purpose"] = self.purpose
        if self.hypothesis is not None:
            out["hypothesis"] = self.hypothesis
        if self.baseline is not None:
            out["baseline"] = self.baseline
        if self.key_deviations:
            out["key_deviations"] = list(self.key_deviations)
        if self.runtime_settings:
            out["runtime_settings"] = list(self.runtime_settings)
        if self.comments:
            out["comments"] = list(self.comments)
        if self.tags:
            out["tags"] = list(self.tags)
        return out

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "ExperimentConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("experiment section must be a mapping when provided")

        data: MutableMapping[str, Any] = dict(payload)
        title = cls._coerce_optional_text(data.pop("title", None), "experiment.title")
        purpose = cls._coerce_optional_text(
            data.pop("purpose", None), "experiment.purpose"
        )
        hypothesis = cls._coerce_optional_text(
            data.pop("hypothesis", None), "experiment.hypothesis"
        )
        baseline = cls._coerce_optional_text(
            data.pop("baseline", None), "experiment.baseline"
        )
        key_deviations = cls._coerce_text_list(
            data.pop("key_deviations", None),
            "experiment.key_deviations",
        )
        runtime_settings = cls._coerce_text_list(
            data.pop("runtime_settings", None),
            "experiment.runtime_settings",
        )
        comments = cls._coerce_text_list(
            data.pop("comments", None),
            "experiment.comments",
        )
        tags = cls._coerce_text_list(data.pop("tags", None), "experiment.tags")

        if data:
            unknown = sorted(str(k) for k in data.keys())
            rendered = [f"experiment.{k}" for k in unknown]
            raise ValueError(f"Unknown experiment keys: {rendered}")

        return cls(
            title=title,
            purpose=purpose,
            hypothesis=hypothesis,
            baseline=baseline,
            key_deviations=key_deviations,
            runtime_settings=runtime_settings,
            comments=comments,
            tags=tags,
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    group_id: Optional[str] = None
    control_group_id: Optional[str] = None
    intended_variable: Optional[str] = None
    comparability_label: Optional[
        Literal["accuracy-comparable", "throughput-comparable", "not-comparable"]
    ] = None

    def __post_init__(self) -> None:
        if self.comparability_label is not None and self.comparability_label not in {
            "accuracy-comparable",
            "throughput-comparable",
            "not-comparable",
        }:
            raise ValueError(
                "benchmark.comparability_label must be one of {'accuracy-comparable', 'throughput-comparable', 'not-comparable'}"
            )

    @staticmethod
    def _coerce_optional_text(value: Any, field_name: str) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string when provided")
        text = value.strip()
        return text or None

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "BenchmarkConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("benchmark section must be a mapping when provided")
        parsed = parse_dataclass_strict(cls, payload, path="benchmark")
        return cls(
            group_id=cls._coerce_optional_text(parsed.group_id, "benchmark.group_id"),
            control_group_id=cls._coerce_optional_text(
                parsed.control_group_id, "benchmark.control_group_id"
            ),
            intended_variable=cls._coerce_optional_text(
                parsed.intended_variable, "benchmark.intended_variable"
            ),
            comparability_label=parsed.comparability_label,
        )


@dataclass(frozen=True)
class Stage2RolloutCorrectionTriagePosteriorConfig:
    num_rollouts: int = 2
    explorer_temperature: float = 0.7
    rollout_temperatures: tuple[float, ...] | None = None
    explorer_top_p: float = 1.0
    explorer_top_k: int = -1
    unlabeled_consistent_iou_threshold: float = 0.85
    recovered_ground_truth_weight_multiplier: float = 2.0

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        default_num_rollouts: Optional[int] = None,
    ) -> "Stage2RolloutCorrectionTriagePosteriorConfig":
        if payload is None:
            return cls(
                num_rollouts=cls.num_rollouts
                if default_num_rollouts is None
                else default_num_rollouts
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                "stage2_rollout_correction.correction.triage_posterior must be a mapping when provided"
            )

        data: MutableMapping[str, Any] = dict(payload)

        default_rollouts = (
            cls.num_rollouts if default_num_rollouts is None else default_num_rollouts
        )
        num_rollouts_raw = data.pop("num_rollouts", default_rollouts)
        try:
            num_rollouts = int(num_rollouts_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.triage_posterior.num_rollouts must be an int"
            ) from exc
        if num_rollouts < 2:
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.num_rollouts must be >= 2"
            )

        explorer_temperature_raw = data.pop(
            "explorer_temperature",
            cls.explorer_temperature,
        )
        try:
            explorer_temperature = float(explorer_temperature_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_temperature must be a float/int"
            ) from exc
        if not math.isfinite(explorer_temperature):
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_temperature must be finite"
            )
        if explorer_temperature < 0.0:
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_temperature must be >= 0"
            )

        rollout_temperatures_raw = data.pop("rollout_temperatures", None)
        rollout_temperatures: tuple[float, ...] | None = None
        if rollout_temperatures_raw is not None:
            if isinstance(rollout_temperatures_raw, (str, bytes)) or not isinstance(
                rollout_temperatures_raw, Sequence
            ):
                raise TypeError(
                    "stage2_rollout_correction.correction.triage_posterior.rollout_temperatures "
                    "must be a sequence of float/int values"
                )
            parsed_temperatures = []
            for idx, raw_value in enumerate(rollout_temperatures_raw):
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "stage2_rollout_correction.correction.triage_posterior.rollout_temperatures "
                        f"must contain only float/int values; bad index={int(idx)}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        "stage2_rollout_correction.correction.triage_posterior.rollout_temperatures "
                        f"must contain only finite values; bad index={int(idx)}"
                    )
                if value < 0.0:
                    raise ValueError(
                        "stage2_rollout_correction.correction.triage_posterior.rollout_temperatures "
                        f"must contain only values >= 0; bad index={int(idx)}"
                    )
                parsed_temperatures.append(float(value))
            if len(parsed_temperatures) not in {1, int(num_rollouts)}:
                raise ValueError(
                    "stage2_rollout_correction.correction.triage_posterior.rollout_temperatures "
                    "length must be 1 or match num_rollouts"
                )
            rollout_temperatures = tuple(float(v) for v in parsed_temperatures)

        explorer_top_p_raw = data.pop("explorer_top_p", cls.explorer_top_p)
        try:
            explorer_top_p = float(
                cls.explorer_top_p if explorer_top_p_raw is None else explorer_top_p_raw
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_top_p must be a float/int"
            ) from exc
        if not math.isfinite(explorer_top_p):
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_top_p must be finite"
            )
        if not (0.0 < explorer_top_p <= 1.0):
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_top_p must be in (0, 1]"
            )

        explorer_top_k_raw = data.pop("explorer_top_k", cls.explorer_top_k)
        try:
            explorer_top_k = int(explorer_top_k_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_top_k must be an int"
            ) from exc
        if explorer_top_k != -1 and explorer_top_k < 1:
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.explorer_top_k must be -1 (disabled) or >= 1"
            )

        unlabeled_consistent_iou_threshold_raw = data.pop(
            "unlabeled_consistent_iou_threshold",
            cls.unlabeled_consistent_iou_threshold,
        )
        try:
            unlabeled_consistent_iou_threshold = float(
                unlabeled_consistent_iou_threshold_raw
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.triage_posterior.unlabeled_consistent_iou_threshold must be a float/int"
            ) from exc
        if not math.isfinite(unlabeled_consistent_iou_threshold):
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.unlabeled_consistent_iou_threshold must be finite"
            )
        if (
            unlabeled_consistent_iou_threshold < 0.0
            or unlabeled_consistent_iou_threshold > 1.0
        ):
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.unlabeled_consistent_iou_threshold must be in [0, 1]"
            )

        recovered_ground_truth_weight_multiplier_raw = data.pop(
            "recovered_ground_truth_weight_multiplier",
            cls.recovered_ground_truth_weight_multiplier,
        )
        try:
            recovered_ground_truth_weight_multiplier = float(
                recovered_ground_truth_weight_multiplier_raw
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.triage_posterior.recovered_ground_truth_weight_multiplier must be a float/int"
            ) from exc
        if not math.isfinite(recovered_ground_truth_weight_multiplier):
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.recovered_ground_truth_weight_multiplier must be finite"
            )
        if recovered_ground_truth_weight_multiplier < 1.0:
            raise ValueError(
                "stage2_rollout_correction.correction.triage_posterior.recovered_ground_truth_weight_multiplier must be >= 1.0"
            )

        if data:
            unknown = [
                f"stage2_rollout_correction.correction.triage_posterior.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_rollout_correction.correction.triage_posterior keys: {unknown}"
            )

        return cls(
            num_rollouts=num_rollouts,
            explorer_temperature=explorer_temperature,
            rollout_temperatures=rollout_temperatures,
            explorer_top_p=explorer_top_p,
            explorer_top_k=explorer_top_k,
            unlabeled_consistent_iou_threshold=unlabeled_consistent_iou_threshold,
            recovered_ground_truth_weight_multiplier=recovered_ground_truth_weight_multiplier,
        )


@dataclass(frozen=True)
class Stage2RolloutCorrectionPseudoPositiveConfig:
    enabled: bool = False
    coord_weight: float = 0.5

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2RolloutCorrectionPseudoPositiveConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError(
                "stage2_rollout_correction.correction.pseudo_positive must be a mapping when provided"
            )

        data: MutableMapping[str, Any] = dict(payload)

        versioned = [
            f"stage2_rollout_correction.correction.pseudo_positive.{str(key)}"
            for key in sorted(data.keys(), key=lambda x: str(x))
            if isinstance(key, str)
            and (
                _is_versioned_alias_for(key, "enabled")
                or _is_versioned_alias_for(key, "coord_weight")
            )
        ]
        if versioned:
            raise ValueError(
                "Versioned pseudo-positive knob aliases are unsupported; "
                f"use unversioned keys instead: {versioned}"
            )

        enabled_raw = data.pop("enabled", cls.enabled)
        if isinstance(enabled_raw, bool):
            enabled = enabled_raw
        elif isinstance(enabled_raw, (int, float)):
            if enabled_raw in (0, 1, 0.0, 1.0):
                enabled = bool(enabled_raw)
            else:
                raise ValueError(
                    "stage2_rollout_correction.correction.pseudo_positive.enabled must be boolean (0 or 1)"
                )
        elif isinstance(enabled_raw, str):
            normalized = enabled_raw.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                enabled = True
            elif normalized in {"false", "0", "no", "n", "off"}:
                enabled = False
            else:
                raise ValueError(
                    "stage2_rollout_correction.correction.pseudo_positive.enabled string value "
                    f"'{enabled_raw}' is not a recognized boolean representation."
                )
        else:
            raise TypeError(
                "stage2_rollout_correction.correction.pseudo_positive.enabled must be a boolean value"
            )

        coord_weight_raw = data.pop("coord_weight", cls.coord_weight)
        try:
            coord_weight = float(coord_weight_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.pseudo_positive.coord_weight must be a float/int"
            ) from exc
        if not math.isfinite(coord_weight):
            raise ValueError(
                "stage2_rollout_correction.correction.pseudo_positive.coord_weight must be finite"
            )
        if not (0.0 < coord_weight < 1.0):
            raise ValueError(
                "stage2_rollout_correction.correction.pseudo_positive.coord_weight must be in (0, 1)"
            )

        if data:
            unknown = [
                f"stage2_rollout_correction.correction.pseudo_positive.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_rollout_correction.correction.pseudo_positive keys: {unknown}"
            )

        return cls(enabled=enabled, coord_weight=coord_weight)


@dataclass(frozen=True)
class Stage2RolloutCorrectionDuplicateControlConfig:
    iou_threshold: float = 0.90
    center_radius_scale: float = 0.80

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2RolloutCorrectionDuplicateControlConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError(
                "stage2_rollout_correction.correction.duplicate_control must be a mapping when provided"
            )

        data: MutableMapping[str, Any] = dict(payload)

        iou_threshold_raw = data.pop("iou_threshold", cls.iou_threshold)
        try:
            iou_threshold = float(iou_threshold_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.duplicate_control.iou_threshold must be a float/int"
            ) from exc
        if not math.isfinite(iou_threshold):
            raise ValueError(
                "stage2_rollout_correction.correction.duplicate_control.iou_threshold must be finite"
            )
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(
                "stage2_rollout_correction.correction.duplicate_control.iou_threshold must be in [0, 1]"
            )

        center_radius_scale_raw = data.pop(
            "center_radius_scale",
            cls.center_radius_scale,
        )
        try:
            center_radius_scale = float(center_radius_scale_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.duplicate_control.center_radius_scale must be a float/int"
            ) from exc
        if not math.isfinite(center_radius_scale):
            raise ValueError(
                "stage2_rollout_correction.correction.duplicate_control.center_radius_scale must be finite"
            )
        if center_radius_scale < 0.0:
            raise ValueError(
                "stage2_rollout_correction.correction.duplicate_control.center_radius_scale must be >= 0"
            )

        if data:
            unknown = [
                f"stage2_rollout_correction.correction.duplicate_control.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_rollout_correction.correction.duplicate_control keys: {unknown}"
            )

        return cls(
            iou_threshold=iou_threshold,
            center_radius_scale=center_radius_scale,
        )


@dataclass(frozen=True)
class Stage2RolloutCorrectionAssignmentConfig:
    strategy: str = "greedy_iou"
    iou_threshold: Optional[float] = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2RolloutCorrectionAssignmentConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_rollout_correction.correction.assignment must be a mapping when provided")

        data: MutableMapping[str, Any] = dict(payload)
        strategy_raw = data.pop("strategy", cls.strategy)
        strategy = str(strategy_raw).strip().lower().replace("-", "_")
        if strategy == "legacy_hungarian_mask_iou":
            raise ValueError(
                "stage2_rollout_correction.correction.assignment.strategy=legacy_hungarian_mask_iou "
                "has been removed; use greedy_iou"
            )
        if strategy != "greedy_iou":
            raise ValueError(
                "stage2_rollout_correction.correction.assignment.strategy must be one of "
                "{'greedy_iou'}"
            )

        iou_threshold_raw = data.pop("iou_threshold", None)
        iou_threshold: Optional[float] = None
        if iou_threshold_raw is not None:
            try:
                iou_threshold = float(iou_threshold_raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "stage2_rollout_correction.correction.assignment.iou_threshold must be a float/int when set"
                ) from exc
            if not math.isfinite(iou_threshold):
                raise ValueError(
                    "stage2_rollout_correction.correction.assignment.iou_threshold must be finite"
                )
            if iou_threshold < 0.0 or iou_threshold > 1.0:
                raise ValueError(
                    "stage2_rollout_correction.correction.assignment.iou_threshold must be in [0, 1]"
                )

        if data:
            unknown = [
                f"stage2_rollout_correction.correction.assignment.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_rollout_correction.correction.assignment keys: {unknown}"
            )

        return cls(
            strategy=strategy,
            iou_threshold=iou_threshold,
        )


@dataclass(frozen=True)
class Stage2RolloutCorrectionFalsePositivePolicyConfig:
    mode: str = "zero_loss_context"
    weak_positive_weight: float = 0.05
    require_explorer_support: bool = True
    min_support_count: int = 1
    require_token_score: bool = False

    @classmethod
    def from_mapping(
        cls, payload: Any
    ) -> "Stage2RolloutCorrectionFalsePositivePolicyConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_rollout_correction.correction.fp_policy must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)

        mode_raw = data.pop("mode", cls.mode)
        mode = str(mode_raw).strip().lower().replace("-", "_")
        if mode not in STAGE2_ROLLOUT_CORRECTION_FP_POLICIES:
            raise ValueError(
                "stage2_rollout_correction.correction.fp_policy.mode must be one of "
                f"{sorted(STAGE2_ROLLOUT_CORRECTION_FP_POLICIES)}"
            )

        weak_positive_weight_raw = data.pop(
            "weak_positive_weight", cls.weak_positive_weight
        )
        try:
            weak_positive_weight = float(weak_positive_weight_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_rollout_correction.correction.fp_policy.weak_positive_weight must be a float/int"
            ) from exc
        if isinstance(weak_positive_weight_raw, bool):
            raise TypeError(
                "stage2_rollout_correction.correction.fp_policy.weak_positive_weight must be a float/int, not bool"
            )
        if not math.isfinite(weak_positive_weight):
            raise ValueError(
                "stage2_rollout_correction.correction.fp_policy.weak_positive_weight must be finite"
            )
        if weak_positive_weight < 0.0:
            raise ValueError(
                "stage2_rollout_correction.correction.fp_policy.weak_positive_weight must be >= 0"
            )

        def _parse_bool(value: Any, *, path: str) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                if value in (0, 1, 0.0, 1.0):
                    return bool(value)
                raise ValueError(f"{path} must be boolean (0 or 1)")
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError(
                    f"{path} string value '{value}' is not a recognized boolean representation."
                )
            raise TypeError(f"{path} must be a boolean value")

        require_explorer_support = _parse_bool(
            data.pop("require_explorer_support", cls.require_explorer_support),
            path="stage2_rollout_correction.correction.fp_policy.require_explorer_support",
        )
        require_token_score = _parse_bool(
            data.pop("require_token_score", cls.require_token_score),
            path="stage2_rollout_correction.correction.fp_policy.require_token_score",
        )

        min_support_count_raw = data.pop(
            "min_support_count", cls.min_support_count
        )
        if isinstance(min_support_count_raw, bool):
            raise TypeError(
                "stage2_rollout_correction.correction.fp_policy.min_support_count must be an int, not bool"
            )
        if isinstance(min_support_count_raw, float):
            if not math.isfinite(min_support_count_raw):
                raise ValueError(
                    "stage2_rollout_correction.correction.fp_policy.min_support_count must be finite"
                )
            if not min_support_count_raw.is_integer():
                raise ValueError(
                    "stage2_rollout_correction.correction.fp_policy.min_support_count must be an integer"
                )
            min_support_count = int(min_support_count_raw)
        elif isinstance(min_support_count_raw, int):
            min_support_count = min_support_count_raw
        elif isinstance(min_support_count_raw, str):
            normalized_count = min_support_count_raw.strip()
            if not normalized_count or not normalized_count.lstrip("+-").isdigit():
                raise ValueError(
                    "stage2_rollout_correction.correction.fp_policy.min_support_count must be an int"
                )
            min_support_count = int(normalized_count)
        else:
            raise TypeError(
                "stage2_rollout_correction.correction.fp_policy.min_support_count must be an int"
            )
        if min_support_count < 1:
            raise ValueError(
                "stage2_rollout_correction.correction.fp_policy.min_support_count must be >= 1"
            )

        if data:
            unknown = [
                f"stage2_rollout_correction.correction.fp_policy.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_rollout_correction.correction.fp_policy keys: {unknown}"
            )

        return cls(
            mode=mode,
            weak_positive_weight=weak_positive_weight,
            require_explorer_support=require_explorer_support,
            min_support_count=min_support_count,
            require_token_score=require_token_score,
        )








@dataclass(frozen=True)
class Stage2RolloutCorrectionModuleSpec:
    name: str
    enabled: bool = True
    weight: float = 1.0
    application: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Any, *, path: str) -> "Stage2RolloutCorrectionModuleSpec":
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path} must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)

        name = str(data.pop("name", "") or "").strip()
        if not name:
            raise ValueError(f"{path}.name must be a non-empty string")
        if name != STAGE2_RESIDUAL_SET_MODULE_NAME:
            raise ValueError(
                f"{path}.name={name!r} has been removed from unified Stage-2; "
                "use residual_set_correction with application.preset=rollout_self_prefix."
            )

        if "channels" in data:
            raise ValueError(
                f"{path}.channels has been removed from unified Stage-2; "
                "stage2_rollout_correction has no per-channel split."
            )

        if "enabled" not in data:
            raise ValueError(f"{path}.enabled must be provided")
        enabled = bool(data.pop("enabled"))

        try:
            weight = float(data.pop("weight", cls.weight))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{path}.weight must be numeric") from exc
        if weight < 0.0:
            raise ValueError(f"{path}.weight must be >= 0")

        app_raw = data.pop("application", None)
        if not isinstance(app_raw, Mapping):
            raise TypeError(f"{path}.application must be a mapping")
        application = dict(app_raw)
        app_unknown = set(application.keys()) - {"preset"}
        if app_unknown:
            raise ValueError(
                f"Unknown {path}.application keys: {sorted(str(k) for k in app_unknown)}"
            )
        preset = str(application.get("preset", "") or "").strip()
        if preset != "rollout_self_prefix":
            raise ValueError(
                f"{path}.application.preset must be 'rollout_self_prefix'; got {preset!r}"
            )

        cfg_raw = data.pop("config", None)
        if cfg_raw is None:
            cfg_raw = {}
        if not isinstance(cfg_raw, Mapping):
            raise TypeError(f"{path}.config must be a mapping")
        config = {
            **STAGE2_RESIDUAL_SET_DEFAULT_CONFIG_VALUES,
            **dict(cfg_raw),
        }
        unknown_cfg = set(config.keys()) - STAGE2_RESIDUAL_SET_CONFIG_KEYS
        if unknown_cfg:
            raise ValueError(
                f"Unknown {path}.config keys for residual_set_correction: "
                f"{sorted(str(k) for k in unknown_cfg)}"
            )
        _validate_stage2_residual_set_config(config)
        missing_cfg = (
            STAGE2_RESIDUAL_SET_REQUIRED_CONFIG_KEYS
            - set(config.keys())
            - STAGE2_RESIDUAL_SET_OPTIONAL_CONFIG_KEYS
        )
        if missing_cfg:
            raise ValueError(
                f"Missing required {path}.config keys for residual_set_correction: "
                f"{sorted(str(k) for k in missing_cfg)}"
            )

        if data:
            unknown = [f"{path}.{str(k)}" for k in sorted(data.keys(), key=lambda x: str(x))]
            raise ValueError(f"Unknown rollout-correction module keys: {unknown}")

        return cls(
            name=name,
            enabled=enabled,
            weight=weight,
            application=application,
            config=config,
        )


@dataclass(frozen=True)
class Stage2RolloutCorrectionPipelineConfig:
    objective: tuple[Stage2RolloutCorrectionModuleSpec, ...] = field(default_factory=tuple)
    diagnostics: tuple[Stage2RolloutCorrectionModuleSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2RolloutCorrectionPipelineConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_rollout_correction.pipeline must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)

        objective_raw = data.pop("objective", None)
        diagnostics_raw = data.pop("diagnostics", [])
        if not isinstance(objective_raw, Sequence) or isinstance(objective_raw, (str, bytes)):
            raise TypeError("stage2_rollout_correction.pipeline.objective must be a list")
        if diagnostics_raw is None:
            diagnostics_raw = []
        if not isinstance(diagnostics_raw, Sequence) or isinstance(diagnostics_raw, (str, bytes)):
            raise TypeError("stage2_rollout_correction.pipeline.diagnostics must be a list")
        if diagnostics_raw:
            raise ValueError(
                "stage2_rollout_correction.pipeline.diagnostics must be empty; "
                "legacy Stage-2 diagnostic modules are removed from the unified contract."
            )

        objective = tuple(
            Stage2RolloutCorrectionModuleSpec.from_mapping(
                item,
                path=f"stage2_rollout_correction.pipeline.objective[{idx}]",
            )
            for idx, item in enumerate(objective_raw)
        )
        enabled_objective = [spec for spec in objective if bool(spec.enabled)]
        if len(enabled_objective) != 1:
            raise ValueError(
                "stage2_rollout_correction.pipeline.objective must contain exactly "
                "one enabled residual_set_correction module"
            )

        if data:
            unknown = [
                f"stage2_rollout_correction.pipeline.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown stage2_rollout_correction.pipeline keys: {unknown}")

        return cls(objective=objective, diagnostics=())


@dataclass(frozen=True)
class Stage2RolloutCorrectionRuntimeConfig:
    assignment: Stage2RolloutCorrectionAssignmentConfig = field(
        default_factory=Stage2RolloutCorrectionAssignmentConfig
    )
    duplicate_control: Stage2RolloutCorrectionDuplicateControlConfig = field(
        default_factory=Stage2RolloutCorrectionDuplicateControlConfig
    )
    producer_wait_timeout_s: Optional[float] = None
    ddp_phase_timeout_s: Optional[float] = None
    rollout_template_family: str = "coordjson"
    fallback_loss_weight: float = 1.0
    invalid_rollout_policy: str = "abort"
    strict_rollout_preflight: bool = False
    insertion_order: str = "tail_append"
    fp_policy: Stage2RolloutCorrectionFalsePositivePolicyConfig = field(
        default_factory=Stage2RolloutCorrectionFalsePositivePolicyConfig
    )
    triage_posterior: Stage2RolloutCorrectionTriagePosteriorConfig = field(
        default_factory=Stage2RolloutCorrectionTriagePosteriorConfig
    )

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2RolloutCorrectionRuntimeConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_rollout_correction.correction must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)

        for removed_key in (
            "pseudo_positive",
            "channel_b",
            "mode",
            "async",
            "rollouts_per_step",
            "enable_pipeline",
            "rollout_decode_batch_size",
            "reordered_gt_sft",
            "desc_ce_weight_matched",
            "semantic_desc_gate",
        ):
            if removed_key in data:
                raise ValueError(
                    f"stage2_rollout_correction.correction.{removed_key} has been removed "
                    "from unified Stage-2 rollout correction."
                )

        assignment = Stage2RolloutCorrectionAssignmentConfig.from_mapping(data.pop("assignment", None))
        duplicate_control = Stage2RolloutCorrectionDuplicateControlConfig.from_mapping(
            data.pop("duplicate_control", None)
        )

        rollout_template_policy = resolve_stage2_rollout_template_policy(
            data.pop("rollout_template_family", cls.rollout_template_family),
            invalid_rollout_policy=data.pop("invalid_rollout_policy", None),
            fallback_loss_weight=data.pop("fallback_loss_weight", cls.fallback_loss_weight),
            strict_rollout_preflight=data.pop(
                "strict_rollout_preflight",
                cls.strict_rollout_preflight,
            ),
        )

        insertion_order = str(data.pop("insertion_order", cls.insertion_order)).strip().lower()
        if insertion_order not in {"tail_append", "sorted", "fn_slot_shuffle"}:
            raise ValueError(
                "stage2_rollout_correction.correction.insertion_order must be one of "
                "{'tail_append', 'sorted', 'fn_slot_shuffle'}"
            )

        fp_policy = Stage2RolloutCorrectionFalsePositivePolicyConfig.from_mapping(
            data.pop("fp_policy", None)
        )
        triage_default_rollouts = (
            4
            if fp_policy.mode == "weak_positive_context"
            else Stage2RolloutCorrectionTriagePosteriorConfig.num_rollouts
        )
        triage_posterior = Stage2RolloutCorrectionTriagePosteriorConfig.from_mapping(
            data.pop("triage_posterior", None),
            default_num_rollouts=triage_default_rollouts,
        )

        producer_wait_timeout_s_raw = data.pop("producer_wait_timeout_s", None)
        producer_wait_timeout_s: Optional[float] = None
        if producer_wait_timeout_s_raw is not None:
            producer_wait_timeout_s = float(producer_wait_timeout_s_raw)
            if producer_wait_timeout_s < 0.0:
                raise ValueError(
                    "stage2_rollout_correction.correction.producer_wait_timeout_s must be >= 0"
                )

        ddp_phase_timeout_s_raw = data.pop("ddp_phase_timeout_s", None)
        ddp_phase_timeout_s: Optional[float] = None
        if ddp_phase_timeout_s_raw is not None:
            ddp_phase_timeout_s = float(ddp_phase_timeout_s_raw)
            if ddp_phase_timeout_s <= 0.0:
                raise ValueError(
                    "stage2_rollout_correction.correction.ddp_phase_timeout_s must be > 0"
                )

        if data:
            unknown = [
                f"stage2_rollout_correction.correction.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown stage2_rollout_correction.correction keys: {unknown}")

        return cls(
            assignment=assignment,
            duplicate_control=duplicate_control,
            producer_wait_timeout_s=producer_wait_timeout_s,
            ddp_phase_timeout_s=ddp_phase_timeout_s,
            rollout_template_family=rollout_template_policy.template_family,
            fallback_loss_weight=float(rollout_template_policy.fallback_loss_weight),
            invalid_rollout_policy=rollout_template_policy.invalid_rollout_policy,
            strict_rollout_preflight=bool(rollout_template_policy.strict_rollout_preflight),
            insertion_order=insertion_order,
            fp_policy=fp_policy,
            triage_posterior=triage_posterior,
        )


@dataclass(frozen=True)
class Stage2RolloutCorrectionConfig:
    pipeline: Stage2RolloutCorrectionPipelineConfig
    correction: Stage2RolloutCorrectionRuntimeConfig = field(
        default_factory=Stage2RolloutCorrectionRuntimeConfig
    )

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2RolloutCorrectionConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_rollout_correction section must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)

        for removed_key in ("schedule", "b_ratio", "channel_b"):
            if removed_key in data:
                raise ValueError(
                    f"stage2_rollout_correction.{removed_key} has been removed; "
                    "unified Stage-2 has no scheduler or per-channel namespace."
                )

        pipeline_raw = data.pop("pipeline", None)
        if pipeline_raw is None:
            raise ValueError("stage2_rollout_correction.pipeline must be provided")
        pipeline = Stage2RolloutCorrectionPipelineConfig.from_mapping(pipeline_raw)
        correction = Stage2RolloutCorrectionRuntimeConfig.from_mapping(
            data.pop("correction", None)
        )

        if data:
            unknown = [
                f"stage2_rollout_correction.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown stage2_rollout_correction keys: {unknown}")

        return cls(pipeline=pipeline, correction=correction)




def _validate_teacher_forcing_training_packing_contract(
    *,
    objective: TeacherForcingObjectiveConfig | None,
    training: Mapping[str, Any],
) -> None:
    if objective is None:
        return
    packing_raw = training.get("packing", False)
    if packing_raw in (None, ""):
        return
    if not isinstance(packing_raw, bool):
        raise TypeError(
            f"training.packing must be boolean when objective.id={TEACHER_FORCING_OBJECTIVE_ID}"
        )
    if packing_raw:
        raise ValueError(
            f"objective.id={TEACHER_FORCING_OBJECTIVE_ID} currently rejects training.packing=true; "
            "exact atom-position packing mapping is not implemented"
        )





_DETECTION_REQUIRED_SECTIONS: set[str] = {
    "data",
    "pipeline",
    "sample_factory",
    "prompt",
    "detection_template",
    "token_embeddings_adapter",
    "objective",
    "packing",
    "evaluation",
    "validation",
}

_DETECTION_RUNTIME_SECTIONS: set[str] = {
    "model",
    "template",
    "training",
    "deepspeed",
    "rlhf",
    "tuner",
    "quantization",
}

_DETECTION_OPTIONAL_SECTIONS: set[str] = {
    "debug",
    "experiment",
    "global_max_length",
    "stage2_rollout_correction",
    "rollout_matching",
}

_DETECTION_OBSOLETE_KEYS: set[str] = {
    "trainer_variant",
    "stage1_set_continuation",
    "prefix_conditioning",
    "legacy_candidate_branch",
    "candidate_balanced",
    "branch_support_weight",
    "branch_balance_weight",
    "support_weight",
    "balance_weight",
    "prefix_sampling",
    "prefix_sampling_mode",
    "prefix_sampling_count",
    "prefix_sampling_prob",
    "prefix_sample_count",
    "prefix_min_objects",
    "prefix_max_objects",
    "suffix",
    "suffix_row",
    "suffix_rows",
    "candidate_energy",
    "branch_energy",
    "branch_energy_weight",
    "energy",
    "energy_weight",
    "logz",
    "log_z",
    "logz_weight",
    "log_z_weight",
    "margin",
    "margin_weight",
    "margin_ranking",
    "positive_evidence_margin",
    "pem",
    "pem_weight",
}

_DETECTION_STATE_WEIGHTINGS: set[str] = {
    "none",
    "legacy_row_mean_prefix_mixture_equivalence",
    "uniform_permutation",
}

_DETECTION_NORMALIZATIONS: set[str] = {
    "token_mean",
    "legacy_row_mean_equivalence",
    "semantic_image_bucket_balanced",
}

_DETECTION_OBSOLETE_SCAN_SECTIONS: set[str] = (
    _DETECTION_REQUIRED_SECTIONS - {"data", "prompt"}
)


def _detection_join_path(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}.{child}"


def _detection_find_obsolete_keys(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            key_path = _detection_join_path(path, key)
            normalized = key.strip().lower().replace("-", "_")
            # `target.support_weight` and `target.balance_weight` are the
            # canonical objectized paths for prefix-rollin; only the flat
            # objective-level aliases remain obsolete.
            if normalized in _DETECTION_OBSOLETE_KEYS and key_path not in {
                "objective.target.support_weight",
                "objective.target.balance_weight",
            }:
                found.append(key_path)
            found.extend(_detection_find_obsolete_keys(raw_value, path=key_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(
                _detection_find_obsolete_keys(item, path=f"{path}[{index}]")
            )
    return found


def _detection_find_obsolete_keys_on_latest_surface(
    payload: Mapping[str, Any],
) -> list[str]:
    found: list[str] = []
    for raw_key in payload.keys():
        key = str(raw_key)
        normalized = key.strip().lower().replace("-", "_")
        if normalized in _DETECTION_OBSOLETE_KEYS:
            found.append(key)

    for section in sorted(_DETECTION_OBSOLETE_SCAN_SECTIONS):
        if section in payload:
            found.extend(
                _detection_find_obsolete_keys(payload[section], path=section)
            )
    return found


def _detection_validate_choice(
    value: str, *, path: str, allowed: set[str]
) -> None:
    if value not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}, got {value!r}")


def _detection_validate_bool(value: bool, *, path: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")


def _detection_validate_runtime_mapping(
    value: Any, *, path: str
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _detection_validate_framework_mapping(
    value: Any, *, path: str, allowed: set[str]
) -> dict[str, Any]:
    data = _detection_validate_runtime_mapping(value, path=path)
    _validate_section_keys_strict(path, data, allowed=allowed)
    return data


def _detection_validate_training_mapping(value: Any) -> dict[str, Any]:
    data = _detection_validate_runtime_mapping(value, path="training")
    _validate_training_checkpoint_keys(data)
    _validate_section_keys_strict("training", data, allowed=_training_allowed_keys())
    if "packing_length" in data:
        raise ValueError(
            "training.packing_length is deprecated and unsupported. "
            "Remove it and set global_max_length/template.max_length instead."
        )
    if "encoded_sample_cache" in data:
        encoded_sample_cache = EncodedSampleCacheConfig.from_mapping(
            data.get("encoded_sample_cache")
        )
        data["encoded_sample_cache"] = encoded_sample_cache.to_mapping()
    if "static_packing_cache" in data:
        static_packing_cache = StaticPackingCacheConfig.from_mapping(
            data.get("static_packing_cache")
        )
        data["static_packing_cache"] = static_packing_cache.to_mapping()
    return data


def _detection_runtime_bool(
    training: Mapping[str, Any],
    key: str,
) -> bool:
    value = training.get(key, False)
    if value in (None, ""):
        return False
    if not isinstance(value, bool):
        raise TypeError(
            f"training.{key} must be boolean when used with detection packing guardrails"
        )
    return value


def _detection_validate_packing_runtime_contract(
    *,
    objective: "DetectionObjectiveConfig | TeacherForcingObjectiveConfig | None",
    packing: "DetectionPackingConfig",
    training: Mapping[str, Any],
) -> None:
    training_packing = _detection_runtime_bool(training, "packing")
    training_eval_packing = _detection_runtime_bool(training, "eval_packing")

    if getattr(objective, "id", None) == TEACHER_FORCING_OBJECTIVE_ID:
        if training_packing:
            raise ValueError(
                f"objective.id={TEACHER_FORCING_OBJECTIVE_ID} currently rejects training.packing=true; "
                "exact atom-position packing mapping is not implemented"
            )
        if training_eval_packing:
            raise ValueError(
                f"objective.id={TEACHER_FORCING_OBJECTIVE_ID} currently rejects "
                "training.eval_packing=true; exact atom-position packing mapping "
                "is not implemented"
            )
        if packing.static_packing:
            raise ValueError(
                f"objective.id={TEACHER_FORCING_OBJECTIVE_ID} currently rejects "
                "packing.static_packing=true; exact atom-position packing mapping "
                "is not implemented"
            )
        if packing.padding_free_packed:
            raise ValueError(
                f"objective.id={TEACHER_FORCING_OBJECTIVE_ID} currently rejects "
                "packing.padding_free_packed=true; exact atom-position packing "
                "mapping is not implemented"
            )
        return

    if packing.static_packing and not training_packing:
        raise ValueError(
            "packing.static_packing=true requires training.packing=true for detection runtime materialization."
        )

    if objective is None:
        return

    if objective.id != "recursive_detection_ce":
        return

    if packing.padding_free_packed:
        raise ValueError(
            "objective.id=recursive_detection_ce does not support packing.padding_free_packed=true "
            "until sidecar offset rewriting is implemented."
        )
    if not packing.static_packing and training_packing:
        raise ValueError(
            "objective.id=recursive_detection_ce requires training.packing=false when packing.static_packing=false."
        )
    if packing.static_packing:
        raise ValueError(
            "objective.id=recursive_detection_ce does not support static packing yet; "
            "set training.packing=false and packing.static_packing=false."
        )
    if training_eval_packing:
        raise ValueError(
            "objective.id=recursive_detection_ce requires training.eval_packing=false."
        )
    if bool(training.get("use_logits_to_keep", False)):
        raise ValueError(
            "objective.id=recursive_detection_ce requires training.use_logits_to_keep=false "
            "because recursive sidecar target positions require full sequence logits."
        )
    if "loss_scale" in training and training.get("loss_scale") not in (None, ""):
        raise ValueError(
            "objective.id=recursive_detection_ce does not support training.loss_scale; "
            "recursive_detection_ce owns the token loss and metric scale."
        )
    padding_side = training.get("padding_side")
    if padding_side not in (None, "", "right"):
        raise ValueError(
            "objective.id=recursive_detection_ce requires training.padding_side='right' "
            "until sidecar offset rewriting is implemented."
        )


def _detection_validate_teacher_forcing_coverage_ledger_contract(
    *,
    objective: "DetectionObjectiveConfig | TeacherForcingObjectiveConfig | None",
    detection_template: "DetectionTemplateConfig",
    packing: "DetectionPackingConfig",
    training: Mapping[str, Any],
    raw_detection_template_id: Any,
) -> None:
    if getattr(objective, "id", None) != TEACHER_FORCING_OBJECTIVE_ID:
        return
    coverage_ledger = getattr(
        getattr(objective, "terms", None),
        "coverage_ledger",
        None,
    )
    if not bool(getattr(coverage_ledger, "enabled", False)):
        return

    if raw_detection_template_id == "compact_full":
        raise ValueError(
            "objective.terms.coverage_ledger.enabled=true rejects "
            "detection_template.id=compact_full from old chat-template/schema "
            "usage; use detection_template.id=compact_object_box_closed"
        )

    contract = resolve_detection_template_contract(detection_template.id)
    if (
        contract.template_id != "compact_object_box_closed"
        or contract.include_object_ref_end is not True
        or contract.include_box_end is not True
    ):
        raise ValueError(
            "objective.terms.coverage_ledger.enabled=true requires "
            "detection_template.id=compact_object_box_closed with "
            "include_object_ref_end=true and include_box_end=true; "
            f"got detection_template.id={detection_template.id!r}"
        )

    if _detection_runtime_bool(training, "packing"):
        raise ValueError(
            "objective.terms.coverage_ledger.enabled=true requires "
            "training.packing=false"
        )
    if _detection_runtime_bool(training, "eval_packing"):
        raise ValueError(
            "objective.terms.coverage_ledger.enabled=true requires "
            "training.eval_packing=false"
        )
    if packing.static_packing:
        raise ValueError(
            "objective.terms.coverage_ledger.enabled=true requires "
            "packing.static_packing=false"
        )
    if packing.padding_free_packed:
        raise ValueError(
            "objective.terms.coverage_ledger.enabled=true requires "
            "packing.padding_free_packed=false"
        )

    batch_size = training.get("per_device_train_batch_size")
    if batch_size not in (None, ""):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size != 1
        ):
            raise ValueError(
                "objective.terms.coverage_ledger.enabled=true requires "
                "training.per_device_train_batch_size == 1 when provided"
            )


def _detection_validate_deepspeed_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    data = _detection_validate_runtime_mapping(value, path="deepspeed")
    DeepSpeedConfig.from_mapping(data)
    return data


def _detection_validate_order_matches_objective(
    data: DetectionDataConfig,
    objective: DetectionObjectiveConfig | TeacherForcingObjectiveConfig,
) -> None:
    if getattr(objective, "id", None) == TEACHER_FORCING_OBJECTIVE_ID:
        required_order = objective.target_ir.rollin_policy.name
        accepted_orders = (
            {"random", "random_permutation"}
            if required_order == "random_permutation"
            else {required_order}
        )
        if data.object_ordering not in accepted_orders:
            raise ValueError(
                "sample_factory.target_sequence.object_ordering must be "
                f"one of {sorted(accepted_orders)!r} for "
                "objective.id='research_teacher_forcing', "
                f"got {data.object_ordering!r}"
            )
        return

    required_order = (
        "sorted" if objective.variant == "sorted_sft" else "random_permutation"
    )
    accepted_orders = (
        {"random", "random_permutation"}
        if required_order == "random_permutation"
        else {required_order}
    )
    if data.object_ordering not in accepted_orders:
        raise ValueError(
            "sample_factory.target_sequence.object_ordering must be "
            f"one of {sorted(accepted_orders)!r} for "
            f"objective.variant={objective.variant!r}, "
            f"got {data.object_ordering!r}"
        )


def _detection_validate_prefix_rollin_contract(
    *,
    detection_template: "DetectionTemplateConfig",
    objective: "DetectionObjectiveConfig | TeacherForcingObjectiveConfig | None",
    experiment: "DetectionExperimentConfig | None",
) -> None:
    if objective is None:
        return
    if getattr(objective, "id", None) == TEACHER_FORCING_OBJECTIVE_ID:
        return
    if objective.variant != "prefix_rollin_et_rmp_ce":
        return

    template_contract = resolve_detection_template_contract(detection_template.id)
    if objective.variant == "prefix_rollin_et_rmp_ce" and not template_contract.is_compact:
        raise ValueError(
            "objective.variant=prefix_rollin_et_rmp_ce requires "
            f"detection_template.id to be one of {COMPACT_TEMPLATE_IDS}"
        )
    if experiment is None:
        raise ValueError(
            f"experiment.surface is required for objective.variant={objective.variant}"
        )


def _detection_validate_token_rows(
    detection_template: "DetectionTemplateConfig",
    token_rows: TokenEmbeddingsAdapterConfig,
) -> None:
    if not token_rows.enabled:
        raise ValueError(
            "token_embeddings_adapter.enabled must be true for detection coord-token training; "
            "otherwise coordinate special-token rows stay frozen and cannot be saved "
            "in the adapter"
        )
    if not token_rows.tie_head:
        raise ValueError(
            "token_embeddings_adapter.tie_head must be true for the current tied-head "
            "Qwen3-VL token-row adapter contract"
        )
    template_contract = resolve_detection_template_contract(detection_template.id)
    if detection_template.coordinate_surface == "coord_token":
        has_coord_geometry = any(
            group.role is TokenRole.COORD_GEOMETRY
            for group in token_rows.groups.values()
        )
        if not has_coord_geometry:
            raise ValueError(
                "token_embeddings_adapter must include at least one group with "
                "role=coord_geometry for coord-token detection"
            )
        coord_groups = [
            group
            for group in token_rows.groups.values()
            if group.role is TokenRole.COORD_GEOMETRY
        ]
        structural_groups = [
            group
            for group in token_rows.groups.values()
            if group.role is TokenRole.STRUCTURAL_CE_ONLY
        ]
        expected_group_count = 2 if template_contract.is_compact else 1
        if len(token_rows.groups) != expected_group_count:
            expected_count = len(template_contract.required_structural_token_ids) + 1000
            raise ValueError(
                "token_embeddings_adapter for coord-token detection must contain exactly the "
                f"{expected_count} allowed trainable rows for "
                f"detection_template.id={detection_template.id!r}: "
                f"{', '.join(template_contract.required_structural_tokens)}"
                f"{', and ' if template_contract.required_structural_tokens else ''}"
                f"{COORD_START_TOKEN}..{COORD_END_TOKEN}; "
                "extra natural-language rows are not allowed"
            )
        if len(coord_groups) != 1:
            raise ValueError(
                "token_embeddings_adapter must contain exactly one coord_geometry group for "
                f"{COORD_START_TOKEN}..{COORD_END_TOKEN}"
            )
        coord_group = coord_groups[0]
        if (
            coord_group.start_token != COORD_START_TOKEN
            or coord_group.end_token != COORD_END_TOKEN
            or coord_group.tokens
            or coord_group.expected_start != EXPECTED_COORD_START_ID
            or coord_group.expected_end != EXPECTED_COORD_END_ID
        ):
            raise ValueError(
                "token_embeddings_adapter coord_geometry must be exactly "
                f"{COORD_START_TOKEN}..{COORD_END_TOKEN} with expected ids "
                f"{EXPECTED_COORD_START_ID}..{EXPECTED_COORD_END_ID}"
            )
        if not template_contract.is_compact:
            if structural_groups:
                raise ValueError(
                    "stage1_json_pretty must not configure compact structural "
                    "token_embeddings_adapter rows"
                )
            return
        if len(structural_groups) != 1:
            raise ValueError(
                "token_embeddings_adapter must include exactly the compact structural rows "
                f"{', '.join(template_contract.required_structural_tokens)}"
            )
        structural_group = structural_groups[0]
        expected_structural_ids = dict(
            zip(
                template_contract.required_structural_tokens,
                template_contract.required_structural_token_ids,
            )
        )
        if (
            structural_group.start_token is not None
            or structural_group.end_token is not None
            or structural_group.tokens != template_contract.required_structural_tokens
            or dict(structural_group.expected_ids) != expected_structural_ids
        ):
            raise ValueError(
                "token_embeddings_adapter structural group must be exactly "
                f"{', '.join(template_contract.required_structural_tokens)} "
                f"with expected ids {expected_structural_ids}"
            )


@dataclass(frozen=True)
class DetectionPipelineConfig:
    id: Literal[
        "stage1_standard_sft",
        "stage1_research_teacher_forcing",
        "stage2_rollout_correction",
    ]

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.id,
            path="pipeline.id",
            allowed={
                "stage1_standard_sft",
                "stage1_research_teacher_forcing",
                "stage2_rollout_correction",
            },
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionPipelineConfig":
        return parse_dataclass_strict(cls, payload, path="pipeline")


@dataclass(frozen=True)
class DetectionTargetSequenceConfig:
    task_family: Literal["detection"]
    object_ordering: Literal["sorted", "random", "random_permutation"] = "sorted"
    object_field_order: Literal["desc_first", "geometry_first"] = "desc_first"
    bbox_format: Literal["xyxy"] = "xyxy"
    coordinate_surface: Literal["coord_token"] = "coord_token"
    strict_parse: bool = True

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.task_family,
            path="sample_factory.target_sequence.task_family",
            allowed={"detection"},
        )
        _detection_validate_choice(
            self.object_ordering,
            path="sample_factory.target_sequence.object_ordering",
            allowed={"sorted", "random", "random_permutation"},
        )
        _detection_validate_choice(
            self.object_field_order,
            path="sample_factory.target_sequence.object_field_order",
            allowed={"desc_first", "geometry_first"},
        )
        _detection_validate_choice(
            self.bbox_format,
            path="sample_factory.target_sequence.bbox_format",
            allowed={"xyxy"},
        )
        _detection_validate_choice(
            self.coordinate_surface,
            path="sample_factory.target_sequence.coordinate_surface",
            allowed={"coord_token"},
        )
        _detection_validate_bool(
            self.strict_parse,
            path="sample_factory.target_sequence.strict_parse",
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionTargetSequenceConfig":
        return parse_dataclass_strict(
            cls, payload, path="sample_factory.target_sequence"
        )


@dataclass(frozen=True)
class DetectionSampleFactoryConfig:
    id: Literal["detection_sequence"]
    target_sequence: DetectionTargetSequenceConfig

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.id,
            path="sample_factory.id",
            allowed={"detection_sequence"},
        )
        if self.target_sequence.task_family != "detection":
            raise ValueError(
                "sample_factory.id=detection_sequence requires "
                "sample_factory.target_sequence.task_family='detection'"
            )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionSampleFactoryConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("sample_factory must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        if "target_sequence" in data:
            data["target_sequence"] = DetectionTargetSequenceConfig.from_mapping(
                data["target_sequence"]
            )
        return parse_dataclass_strict(cls, data, path="sample_factory")


@dataclass(frozen=True)
class DetectionDataConfig:
    train_jsonl: str
    val_jsonl: str
    image_root: str | None = None
    object_ordering: Literal["sorted", "random", "random_permutation"] = "sorted"

    def __post_init__(self) -> None:
        for field_name in ("train_jsonl", "val_jsonl"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"data.{field_name} must be a string")
        if self.image_root is not None and not isinstance(self.image_root, str):
            raise TypeError("data.image_root must be a string when provided")
        _detection_validate_choice(
            self.object_ordering,
            path="data.object_ordering",
            allowed={"sorted", "random", "random_permutation"},
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionDataConfig":
        return parse_dataclass_strict(cls, payload, path="data")


@dataclass(frozen=True)
class DetectionPromptConfig:
    system_variant: str
    user_variant: str
    include_template_summary: bool = True
    variant: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("system_variant", "user_variant"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"prompt.{field_name} must be a string")
        _detection_validate_bool(
            self.include_template_summary,
            path="prompt.include_template_summary",
        )
        if self.variant is not None and not isinstance(self.variant, str):
            raise TypeError("prompt.variant must be a string when provided")

    @property
    def prompt_variant_enabled(self) -> bool:
        return self.variant is not None

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionPromptConfig":
        return parse_dataclass_strict(cls, payload, path="prompt")


@dataclass(frozen=True)
class DetectionTemplateConfig:
    id: Literal[
        "stage1_json_pretty",
        "compact",
        "compact_box_closed",
        "compact_object_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    ]

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.id,
            path="detection_template.id",
            allowed=set(SUPPORTED_DETECTION_TEMPLATE_IDS),
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionTemplateConfig":
        if isinstance(payload, Mapping) and payload.get("id") == "compact_full":
            payload = {**dict(payload), "id": "compact"}
        return parse_dataclass_strict(cls, payload, path="detection_template")

    @property
    def coordinate_surface(self) -> Literal["coord_token"]:
        return "coord_token"

    @property
    def bbox_format(self) -> Literal["xyxy"]:
        return "xyxy"

    @property
    def object_field_order(self) -> None:
        return None

    @property
    def strict_parse(self) -> bool:
        return True


@dataclass(frozen=True)
class UniformInclusiveKConfig:
    type: Literal["uniform_inclusive"]
    min_k: int
    max_k: Literal["object_count"]

    def __post_init__(self) -> None:
        if (
            self.type != "uniform_inclusive"
            or type(self.min_k) is not int
            or self.min_k != 0
            or self.max_k != "object_count"
        ):
            raise ValueError(
                "objective.rollin.k_distribution must be exactly "
                "uniform_inclusive over 0..object_count"
            )


@dataclass(frozen=True)
class PrefixRollinConfig:
    enabled: bool
    source: Literal["ground_truth"]
    prefix_loss: Literal["masked"]
    suffix_order: Literal["same_sampled_permutation"]
    k_distribution: UniformInclusiveKConfig

    def __post_init__(self) -> None:
        _detection_validate_bool(self.enabled, path="objective.rollin.enabled")
        _detection_validate_choice(
            self.source,
            path="objective.rollin.source",
            allowed={"ground_truth"},
        )
        _detection_validate_choice(
            self.prefix_loss,
            path="objective.rollin.prefix_loss",
            allowed={"masked"},
        )
        _detection_validate_choice(
            self.suffix_order,
            path="objective.rollin.suffix_order",
            allowed={"same_sampled_permutation"},
        )


@dataclass(frozen=True)
class EntryTrieSupportBalanceConfig:
    type: Literal["entry_trie_support_balance"]
    trie_scope: Literal["object_entry"]
    q_weighting: Literal["object_multiplicity_uniform"]
    singleton: Literal["hard_ce"]
    control_tokens: Literal["hard_ce"]
    support_weight: float
    balance_weight: float

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.type,
            path="objective.target.type",
            allowed={"entry_trie_support_balance"},
        )
        _detection_validate_choice(
            self.trie_scope,
            path="objective.target.trie_scope",
            allowed={"object_entry"},
        )
        _detection_validate_choice(
            self.q_weighting,
            path="objective.target.q_weighting",
            allowed={"object_multiplicity_uniform"},
        )
        _detection_validate_choice(
            self.singleton,
            path="objective.target.singleton",
            allowed={"hard_ce"},
        )
        _detection_validate_choice(
            self.control_tokens,
            path="objective.target.control_tokens",
            allowed={"hard_ce"},
        )
        for field_name in ("support_weight", "balance_weight"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"objective.target.{field_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"objective.target.{field_name} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"objective.target.{field_name} must be > 0")


@dataclass(frozen=True)
class CompactTypeGateWeights:
    struct: float
    coord: float
    desc: float
    eos: float

    def __post_init__(self) -> None:
        for field_name in ("struct", "coord", "desc", "eos"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"objective.type_gate.weights.{field_name} must be numeric"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"objective.type_gate.weights.{field_name} must be finite"
                )
            if float(value) < 0.0:
                raise ValueError(
                    f"objective.type_gate.weights.{field_name} must be >= 0"
                )


@dataclass(frozen=True)
class CompactTypeGateConfig:
    enabled: bool
    mode: Literal["allowed_type_mass"]
    weights: CompactTypeGateWeights

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled, path="objective.type_gate.enabled"
        )
        _detection_validate_choice(
            self.mode,
            path="objective.type_gate.mode",
            allowed={"allowed_type_mass"},
        )


@dataclass(frozen=True)
class DetectionExperimentConfig:
    surface: Literal["smoke", "ablation", "production"]
    ablation_id: Optional[str] = None
    claim_scope: Optional[Literal["none", "smoke", "paper", "production"]] = None
    title: Optional[str] = None
    purpose: Optional[str] = None
    hypothesis: Optional[str] = None
    key_deviations: tuple[str, ...] = ()
    runtime_settings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.surface,
            path="experiment.surface",
            allowed={"smoke", "ablation", "production"},
        )
        if self.claim_scope is not None:
            _detection_validate_choice(
                self.claim_scope,
                path="experiment.claim_scope",
                allowed={"none", "smoke", "paper", "production"},
            )
        allowed_claim_scopes = {
            "ablation": {None, "none"},
            "smoke": {None, "none", "smoke"},
            "production": {None, "paper", "production"},
        }[self.surface]
        if self.claim_scope not in allowed_claim_scopes:
            allowed_s = ", ".join(
                str(value)
                for value in sorted(v for v in allowed_claim_scopes if v is not None)
            )
            raise ValueError(
                f"experiment.claim_scope={self.claim_scope!r} is not allowed for "
                f"experiment.surface={self.surface!r}; allowed: {allowed_s}"
            )
        for field_name in ("ablation_id", "title", "purpose", "hypothesis"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"experiment.{field_name} must be a string when provided"
                )
        for field_name in ("key_deviations", "runtime_settings"):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"experiment.{field_name} must be a list of strings")
            normalized = tuple(str(item) for item in value)
            object.__setattr__(self, field_name, normalized)

    @classmethod
    def from_mapping(
        cls,
        payload: Optional[Mapping[str, Any]],
        *,
        required_for_variant: Optional[str] = None,
    ) -> Optional["DetectionExperimentConfig"]:
        if payload is None:
            if required_for_variant is not None:
                raise ValueError(
                    "experiment.surface is required for "
                    f"objective.variant={required_for_variant}"
                )
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("experiment section must be a mapping when provided")
        if "surface" not in payload and required_for_variant is not None:
            raise ValueError(
                "experiment.surface is required for "
                f"objective.variant={required_for_variant}"
            )
        return parse_dataclass_strict(cls, payload, path="experiment")


@dataclass(frozen=True)
class CoordSoftCEConfig:
    enabled: bool
    target_distribution: Literal[
        "iou_gibbs_v0",
        "ciou_gibbs_v0",
        "instance_trie_gaussian",
    ]

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.coord_soft_ce.enabled",
        )
        _detection_validate_choice(
            self.target_distribution,
            path="objective.coord_soft_ce.target_distribution",
            allowed={"iou_gibbs_v0", "ciou_gibbs_v0", "instance_trie_gaussian"},
        )


@dataclass(frozen=True)
class InstanceTrieGaussianCoordSoftCEConfig(CoordSoftCEConfig):
    target_distribution: Literal["instance_trie_gaussian"]
    gaussian_mixture_weight: float = 0.1
    gaussian_r95_axis_fraction: float = 0.04
    gaussian_r95_cap_bins: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        _detection_validate_choice(
            self.target_distribution,
            path="objective.coord_soft_ce.target_distribution",
            allowed={"instance_trie_gaussian"},
        )
        if not isinstance(self.gaussian_mixture_weight, (int, float)) or isinstance(
            self.gaussian_mixture_weight, bool
        ):
            raise TypeError(
                "objective.coord_soft_ce.gaussian_mixture_weight must be numeric"
            )
        if not math.isfinite(float(self.gaussian_mixture_weight)):
            raise ValueError(
                "objective.coord_soft_ce.gaussian_mixture_weight must be finite and within [0, 1]"
            )
        if not 0.0 <= float(self.gaussian_mixture_weight) <= 1.0:
            raise ValueError(
                "objective.coord_soft_ce.gaussian_mixture_weight must be within [0, 1]"
            )
        if not isinstance(self.gaussian_r95_axis_fraction, (int, float)) or isinstance(
            self.gaussian_r95_axis_fraction, bool
        ):
            raise TypeError(
                "objective.coord_soft_ce.gaussian_r95_axis_fraction must be numeric"
            )
        if (
            not math.isfinite(float(self.gaussian_r95_axis_fraction))
            or float(self.gaussian_r95_axis_fraction) <= 0.0
            or float(self.gaussian_r95_axis_fraction) > 1.0
        ):
            raise ValueError(
                "objective.coord_soft_ce.gaussian_r95_axis_fraction must be finite and within (0, 1]"
            )
        if not isinstance(self.gaussian_r95_cap_bins, int) or isinstance(
            self.gaussian_r95_cap_bins, bool
        ):
            raise TypeError(
                "objective.coord_soft_ce.gaussian_r95_cap_bins must be an integer"
            )
        if int(self.gaussian_r95_cap_bins) < 0 or int(self.gaussian_r95_cap_bins) > 999:
            raise ValueError(
                "objective.coord_soft_ce.gaussian_r95_cap_bins must be within [0, 999]"
            )

@dataclass(frozen=True)
class GibbsCoordSoftCEConfig(CoordSoftCEConfig):
    target_distribution: Literal["iou_gibbs_v0", "ciou_gibbs_v0"]
    tau: float
    tau_source: Literal["train_one_token_iou_median_v0"]
    weighting: Literal["preserve_recursive_support_balance"] = (
        "preserve_recursive_support_balance"
    )
    replace_coord_hard_ce: bool = True
    apply_to_multi_positive: Literal["support_mixture"] = "support_mixture"

    def __post_init__(self) -> None:
        super().__post_init__()
        _detection_validate_choice(
            self.target_distribution,
            path="objective.coord_soft_ce.target_distribution",
            allowed={"iou_gibbs_v0", "ciou_gibbs_v0"},
        )
        if not isinstance(self.tau, (int, float)) or isinstance(self.tau, bool):
            raise TypeError("objective.coord_soft_ce.tau must be numeric")
        if not math.isfinite(float(self.tau)) or float(self.tau) <= 0.0:
            raise ValueError("objective.coord_soft_ce.tau must be finite and > 0")
        _detection_validate_choice(
            self.tau_source,
            path="objective.coord_soft_ce.tau_source",
            allowed={"train_one_token_iou_median_v0"},
        )
        _detection_validate_choice(
            self.weighting,
            path="objective.coord_soft_ce.weighting",
            allowed={"preserve_recursive_support_balance"},
        )
        _detection_validate_bool(
            self.replace_coord_hard_ce,
            path="objective.coord_soft_ce.replace_coord_hard_ce",
        )
        if not self.replace_coord_hard_ce:
            raise ValueError(
                "objective.coord_soft_ce.replace_coord_hard_ce=false is unsupported"
            )
        _detection_validate_choice(
            self.apply_to_multi_positive,
            path="objective.coord_soft_ce.apply_to_multi_positive",
            allowed={"support_mixture"},
        )


@dataclass(frozen=True)
class TeacherForcingRollinPolicyConfig:
    name: Literal["random_permutation"] = "random_permutation"
    base_seed: int = 17

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.name,
            path="objective.target_ir.rollin_policy.name",
            allowed={"random_permutation"},
        )
        if not isinstance(self.base_seed, int) or isinstance(self.base_seed, bool):
            raise TypeError(
                "objective.target_ir.rollin_policy.base_seed must be an integer"
            )
        if int(self.base_seed) < 0:
            raise ValueError("objective.target_ir.rollin_policy.base_seed must be >= 0")

    @classmethod
    def from_mapping(cls, payload: Any) -> "TeacherForcingRollinPolicyConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError("objective.target_ir.rollin_policy must be a mapping")
        return parse_dataclass_strict(
            cls,
            payload,
            path="objective.target_ir.rollin_policy",
        )


@dataclass(frozen=True)
class TeacherForcingExactPackingMappingConfig:
    enabled: bool = False

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.target_ir.exact_packing_mapping.enabled",
        )
        if self.enabled:
            raise ValueError(
                "objective.target_ir.exact_packing_mapping.enabled=true is "
                "unsupported until exact atom-position mapping is implemented"
            )

    @classmethod
    def from_mapping(cls, payload: Any) -> "TeacherForcingExactPackingMappingConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError(
                "objective.target_ir.exact_packing_mapping must be a mapping"
            )
        return parse_dataclass_strict(
            cls,
            payload,
            path="objective.target_ir.exact_packing_mapping",
        )


@dataclass(frozen=True)
class TeacherForcingTargetIRConfig:
    rollin_policy: TeacherForcingRollinPolicyConfig = field(
        default_factory=TeacherForcingRollinPolicyConfig
    )
    exact_packing_mapping: TeacherForcingExactPackingMappingConfig = field(
        default_factory=TeacherForcingExactPackingMappingConfig
    )

    @classmethod
    def from_mapping(cls, payload: Any) -> "TeacherForcingTargetIRConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError("objective.target_ir must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        rollin_policy = TeacherForcingRollinPolicyConfig.from_mapping(
            data.pop("rollin_policy", {})
        )
        exact_packing_mapping = TeacherForcingExactPackingMappingConfig.from_mapping(
            data.pop("exact_packing_mapping", {})
        )
        if data:
            unknown = [
                f"objective.target_ir.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown objective.target_ir keys: {unknown}")
        return cls(
            rollin_policy=rollin_policy,
            exact_packing_mapping=exact_packing_mapping,
        )


@dataclass(frozen=True)
class TeacherForcingEnabledModuleConfig:
    enabled: bool = False

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.terms.*.enabled",
        )


@dataclass(frozen=True)
class TeacherForcingWithinValidCoverageConfig:
    enabled: bool = False
    coverage_strength: float = 0.0

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.terms.within_valid_coverage.enabled",
        )
        if not isinstance(self.coverage_strength, (int, float)) or isinstance(
            self.coverage_strength, bool
        ):
            raise TypeError(
                "objective.terms.within_valid_coverage.coverage_strength must be numeric"
            )
        value = float(self.coverage_strength)
        if not math.isfinite(value):
            raise ValueError(
                "objective.terms.within_valid_coverage.coverage_strength must be finite"
            )
        if value < 0.0:
            raise ValueError(
                "objective.terms.within_valid_coverage.coverage_strength must be >= 0"
            )
        object.__setattr__(self, "coverage_strength", value)


def _teacher_forcing_finite_float(
    value: Any,
    *,
    path: str,
    minimum: float,
    minimum_label: str,
    inclusive: bool = True,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{path} must be finite")
    if inclusive:
        if parsed < minimum:
            raise ValueError(f"{path} must be >= {minimum_label}")
    elif parsed <= minimum:
        raise ValueError(f"{path} must be > {minimum_label}")
    return parsed


@dataclass(frozen=True)
class TeacherForcingCoverageLedgerConfig:
    enabled: bool = False
    coverage_weight: float = 0.1
    region_anchor_weight: float = 0.1
    ledger_projection_dim: int = 256
    temperature: float = 0.2
    normalize_eps: float = 1.0e-6
    pos_weight: float = 1.0
    log_auc: bool = True
    log_accuracy: bool = True
    overlay_sample_count: int = 16
    smoke_sample_count: int = 128
    smoke_sample_seed: int = 20260623

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.terms.coverage_ledger.enabled",
        )
        for field_name in ("coverage_weight", "region_anchor_weight"):
            object.__setattr__(
                self,
                field_name,
                _teacher_forcing_finite_float(
                    getattr(self, field_name),
                    path=f"objective.terms.coverage_ledger.{field_name}",
                    minimum=0.0,
                    minimum_label="0",
                ),
            )
        if (
            not isinstance(self.ledger_projection_dim, int)
            or isinstance(self.ledger_projection_dim, bool)
            or self.ledger_projection_dim <= 0
        ):
            raise ValueError(
                "objective.terms.coverage_ledger.ledger_projection_dim "
                "must be a positive integer"
            )
        object.__setattr__(
            self,
            "temperature",
            _teacher_forcing_finite_float(
                self.temperature,
                path="objective.terms.coverage_ledger.temperature",
                minimum=0.05,
                minimum_label="0.05",
            ),
        )
        object.__setattr__(
            self,
            "normalize_eps",
            _teacher_forcing_finite_float(
                self.normalize_eps,
                path="objective.terms.coverage_ledger.normalize_eps",
                minimum=1.0e-8,
                minimum_label="1e-8",
            ),
        )
        object.__setattr__(
            self,
            "pos_weight",
            _teacher_forcing_finite_float(
                self.pos_weight,
                path="objective.terms.coverage_ledger.pos_weight",
                minimum=0.0,
                minimum_label="0",
                inclusive=False,
            ),
        )
        _detection_validate_bool(
            self.log_auc,
            path="objective.terms.coverage_ledger.log_auc",
        )
        _detection_validate_bool(
            self.log_accuracy,
            path="objective.terms.coverage_ledger.log_accuracy",
        )
        expected_exact_counts = {
            "overlay_sample_count": 16,
            "smoke_sample_count": 128,
            "smoke_sample_seed": 20260623,
        }
        for field_name, expected in expected_exact_counts.items():
            if (
                not isinstance(getattr(self, field_name), int)
                or isinstance(getattr(self, field_name), bool)
                or getattr(self, field_name) != expected
            ):
                raise ValueError(
                    f"objective.terms.coverage_ledger.{field_name} "
                    f"must be exactly {expected}"
                )


@dataclass(frozen=True)
class TeacherForcingModulesConfig:
    token_type_mass: TeacherForcingEnabledModuleConfig = field(
        default_factory=TeacherForcingEnabledModuleConfig
    )
    conditional_valid_set_likelihood: TeacherForcingEnabledModuleConfig = field(
        default_factory=TeacherForcingEnabledModuleConfig
    )
    within_valid_coverage: TeacherForcingWithinValidCoverageConfig = field(
        default_factory=TeacherForcingWithinValidCoverageConfig
    )
    continuation_margin: TeacherForcingEnabledModuleConfig = field(
        default_factory=TeacherForcingEnabledModuleConfig
    )
    coverage_ledger: TeacherForcingCoverageLedgerConfig = field(
        default_factory=TeacherForcingCoverageLedgerConfig
    )

    @classmethod
    def from_mapping(cls, payload: Any) -> "TeacherForcingModulesConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError("objective.terms must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        token_type_mass = parse_dataclass_strict(
            TeacherForcingEnabledModuleConfig,
            data.pop("token_type_mass", {}),
            path="objective.terms.token_type_mass",
        )
        conditional_valid_set_likelihood = parse_dataclass_strict(
            TeacherForcingEnabledModuleConfig,
            data.pop("conditional_valid_set_likelihood", {}),
            path="objective.terms.conditional_valid_set_likelihood",
        )
        within_valid_coverage = parse_dataclass_strict(
            TeacherForcingWithinValidCoverageConfig,
            data.pop("within_valid_coverage", {}),
            path="objective.terms.within_valid_coverage",
        )
        continuation_margin = parse_dataclass_strict(
            TeacherForcingEnabledModuleConfig,
            data.pop("continuation_margin", {}),
            path="objective.terms.continuation_margin",
        )
        coverage_ledger = parse_dataclass_strict(
            TeacherForcingCoverageLedgerConfig,
            data.pop("coverage_ledger", {}),
            path="objective.terms.coverage_ledger",
        )
        if data:
            unknown = [
                f"objective.terms.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown objective.terms keys: {unknown}")
        return cls(
            token_type_mass=token_type_mass,
            conditional_valid_set_likelihood=conditional_valid_set_likelihood,
            within_valid_coverage=within_valid_coverage,
            continuation_margin=continuation_margin,
            coverage_ledger=coverage_ledger,
        )


@dataclass(frozen=True)
class TeacherForcingObjectiveConfig:
    id: Literal["research_teacher_forcing"]
    profile: Literal[
        "hard_sft",
        "pure_valid_set_marginal",
        "hybrid_valid_set_marginal",
    ] = "hard_sft"
    target_ir: TeacherForcingTargetIRConfig = field(
        default_factory=TeacherForcingTargetIRConfig
    )
    terms: TeacherForcingModulesConfig = field(default_factory=TeacherForcingModulesConfig)

    @property
    def modules(self) -> TeacherForcingModulesConfig:
        return self.terms

    def __post_init__(self) -> None:
        if self.id != TEACHER_FORCING_OBJECTIVE_ID:
            raise ValueError(
                "objective.id must be exactly 'research_teacher_forcing'; "
                f"got {self.id!r}"
            )
        _detection_validate_choice(
            self.profile,
            path="objective.profile",
            allowed=TEACHER_FORCING_PROFILES,
        )
        coverage = self.terms.within_valid_coverage
        coverage_strength = float(coverage.coverage_strength)
        if self.profile == "hybrid_valid_set_marginal":
            if not bool(coverage.enabled) or coverage_strength <= 0.0:
                raise ValueError(
                    "objective.profile=hybrid_valid_set_marginal requires "
                    "objective.terms.within_valid_coverage.coverage_strength > 0"
                )
        elif coverage_strength > 0.0:
            raise ValueError(
                f"objective.profile={self.profile} requires "
                "objective.terms.within_valid_coverage.coverage_strength=0"
            )
        if self.profile == "hard_sft":
            hard_sft_module_checks = (
                (
                    "objective.terms.token_type_mass.enabled",
                    bool(self.terms.token_type_mass.enabled),
                ),
                (
                    "objective.terms.conditional_valid_set_likelihood.enabled",
                    bool(self.terms.conditional_valid_set_likelihood.enabled),
                ),
                (
                    "objective.terms.within_valid_coverage.enabled",
                    bool(coverage.enabled),
                ),
                (
                    "objective.terms.within_valid_coverage.coverage_strength",
                    coverage_strength > 0.0,
                ),
                (
                    "objective.terms.continuation_margin.enabled",
                    bool(self.terms.continuation_margin.enabled),
                ),
            )
            for module_key, is_enabled in hard_sft_module_checks:
                if is_enabled:
                    raise ValueError(
                            "objective.profile=hard_sft does not support "
                            f"{module_key}; target-IR teacher-forcing modules "
                            "require a valid-set runtime path"
                    )
    @classmethod
    def from_mapping(cls, payload: Any) -> "TeacherForcingObjectiveConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("objective must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        raw_id = data.get("id")
        if raw_id != TEACHER_FORCING_OBJECTIVE_ID:
            if raw_id == LEGACY_TEACHER_FORCING_OBJECTIVE_ID:
                raise ValueError(
                    "objective.id='teacher_forcing' has been retired for active "
                    "target-hierarchy configs; use objective.id: "
                    "'research_teacher_forcing'"
                )
            raise ValueError(
                "objective.id must be exactly 'research_teacher_forcing'; "
                f"legacy objective ids are unsupported, got {raw_id!r}"
            )
        if "target_ir" in data:
            data["target_ir"] = TeacherForcingTargetIRConfig.from_mapping(
                data["target_ir"]
            )
        if "modules" in data:
            raise ValueError(
                "objective.modules is retired for active research_teacher_forcing "
                "configs; use objective.terms"
            )
        if "terms" in data:
            data["terms"] = TeacherForcingModulesConfig.from_mapping(data["terms"])
        return parse_dataclass_strict(cls, data, path="objective")


@dataclass(frozen=True)
class StandardCECoordSoftAuxiliaryConfig(CoordSoftCEW1Config):
    """Optional coord-token auxiliary under the public Standard SFT objective."""

    soft_ce_weight: float = 0.0
    w1_weight: float = 0.0
    gate_weight: float = 0.0


@dataclass(frozen=True)
class StandardCEGeometryAuxiliaryConfig:
    enabled: bool = False

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.auxiliaries.geometry.enabled",
        )
        if self.enabled:
            raise ValueError(
                "objective.auxiliaries.geometry.enabled=true is not implemented "
                "for standard_ce; keep objective.auxiliaries.geometry.enabled=false "
                "until a geometry auxiliary has an audited loss/runtime path"
            )


@dataclass(frozen=True)
class StandardCEAuxiliariesConfig:
    coord_soft_ce: StandardCECoordSoftAuxiliaryConfig = field(
        default_factory=StandardCECoordSoftAuxiliaryConfig
    )
    geometry: StandardCEGeometryAuxiliaryConfig = field(
        default_factory=StandardCEGeometryAuxiliaryConfig
    )

    @classmethod
    def from_mapping(cls, payload: Any) -> "StandardCEAuxiliariesConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError("objective.auxiliaries must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        coord_soft_ce = StandardCECoordSoftAuxiliaryConfig.from_mapping(
            data.pop("coord_soft_ce", None),
            path="objective.auxiliaries.coord_soft_ce",
        )
        geometry = parse_dataclass_strict(
            StandardCEGeometryAuxiliaryConfig,
            data.pop("geometry", {}),
            path="objective.auxiliaries.geometry",
        )
        if data:
            unknown = [
                f"objective.auxiliaries.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown objective.auxiliaries keys: {unknown}")
        return cls(coord_soft_ce=coord_soft_ce, geometry=geometry)


@dataclass(frozen=True)
class DetectionObjectiveConfig:
    id: Literal["standard_ce", "recursive_detection_ce"]
    variant: Literal[
        "sorted_sft",
        "random_order_sft",
        "random_permutation_et_rmp_ce",
        "trie_disabled_full_suffix_ce",
        "prefix_rollin_et_rmp_ce",
    ] = "random_order_sft"
    trie_support_weight: float = 0.0
    trie_balance_weight: float = 0.0
    state_weighting: str = "none"
    normalization: str = "token_mean"
    rollin: Optional[PrefixRollinConfig] = None
    target: Optional[EntryTrieSupportBalanceConfig] = None
    type_gate: Optional[CompactTypeGateConfig] = None
    coord_soft_ce: Optional[CoordSoftCEConfig] = None
    auxiliaries: Optional[StandardCEAuxiliariesConfig] = None

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.id,
            path="objective.id",
            allowed={STANDARD_CE_OBJECTIVE_ID, "recursive_detection_ce"},
        )
        _detection_validate_choice(
            self.variant,
            path="objective.variant",
            allowed={
                "sorted_sft",
                "random_order_sft",
                "random_permutation_et_rmp_ce",
                "trie_disabled_full_suffix_ce",
                "prefix_rollin_et_rmp_ce",
            },
        )
        for field_name in ("trie_support_weight", "trie_balance_weight"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"objective.{field_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"objective.{field_name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"objective.{field_name} must be >= 0")
        if self.variant in {"sorted_sft", "random_order_sft"}:
            if (
                float(self.trie_support_weight) != 0.0
                or float(self.trie_balance_weight) != 0.0
            ):
                raise ValueError(
                    "SFT objective variants require objective.trie_support_weight=0 "
                    "and objective.trie_balance_weight=0"
                )
            if self.state_weighting != "none":
                raise ValueError(
                    "SFT objective variants require objective.state_weighting=none"
                )
            if self.normalization != "token_mean":
                raise ValueError(
                    "SFT objective variants require objective.normalization=token_mean"
                )
        if self.variant == "trie_disabled_full_suffix_ce":
            if (
                float(self.trie_support_weight) != 0.0
                or float(self.trie_balance_weight) != 0.0
            ):
                raise ValueError(
                    "objective.variant=trie_disabled_full_suffix_ce requires "
                    "objective.trie_support_weight=0 and objective.trie_balance_weight=0"
                )
            if self.state_weighting != "none":
                raise ValueError(
                    "objective.variant=trie_disabled_full_suffix_ce requires "
                    "objective.state_weighting=none"
                )
            if self.normalization != "token_mean":
                raise ValueError(
                    "objective.variant=trie_disabled_full_suffix_ce requires "
                    "objective.normalization=token_mean"
                )
        if self.variant == "random_permutation_et_rmp_ce":
            total = float(self.trie_support_weight) + float(self.trie_balance_weight)
            if total <= 0.0:
                raise ValueError(
                    "objective.trie_support_weight and objective.trie_balance_weight "
                    "must sum to > 0 for random_permutation_et_rmp_ce"
                )
        if self.variant == "prefix_rollin_et_rmp_ce":
            if float(self.trie_support_weight) != 0.0:
                raise ValueError(
                    "objective.trie_support_weight is an obsolete flat weight alias "
                    "for prefix_rollin_et_rmp_ce; use objective.target.support_weight"
                )
            if float(self.trie_balance_weight) != 0.0:
                raise ValueError(
                    "objective.trie_balance_weight is an obsolete flat weight alias "
                    "for prefix_rollin_et_rmp_ce; use objective.target.balance_weight"
                )
            object.__setattr__(self, "trie_support_weight", None)
            object.__setattr__(self, "trie_balance_weight", None)
            if self.state_weighting != "uniform_permutation":
                raise ValueError(
                    "objective.state_weighting must be uniform_permutation for "
                    "prefix_rollin_et_rmp_ce"
                )
            if self.normalization != "semantic_image_bucket_balanced":
                raise ValueError(
                    "objective.normalization must be semantic_image_bucket_balanced "
                    "for prefix_rollin_et_rmp_ce"
                )
            missing = [
                name
                for name in ("rollin", "target", "type_gate")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "objective.variant=prefix_rollin_et_rmp_ce requires "
                    f"objectized objective sections: {missing}"
                )
        else:
            unexpected = [
                name
                for name in ("rollin", "target")
                if getattr(self, name) is not None
            ]
            if unexpected:
                raise ValueError(
                    "objectized objective sections are only supported for "
                    "objective.variant=prefix_rollin_et_rmp_ce, except "
                    "objective.type_gate on "
                    "random_permutation_et_rmp_ce: "
                    f"{unexpected}"
                )
            if (
                self.type_gate is not None
                and self.variant != "random_permutation_et_rmp_ce"
            ):
                raise ValueError(
                    "objective.type_gate is only supported for latest "
                    "recursive_detection_ce ET-RMP variants"
                )
        if self.id == STANDARD_CE_OBJECTIVE_ID and self.variant not in {
            "sorted_sft",
            "random_order_sft",
        }:
            raise ValueError(
                f"objective.id={self.id} requires an SFT objective.variant"
            )
        if self.id == STANDARD_CE_OBJECTIVE_ID and self.auxiliaries is None:
            object.__setattr__(
                self,
                "auxiliaries",
                StandardCEAuxiliariesConfig(),
            )
        if self.id == "recursive_detection_ce" and self.variant in {
            "sorted_sft",
            "random_order_sft",
        }:
            raise ValueError(
                "objective.id=recursive_detection_ce requires a recursive detection "
                "objective.variant"
            )
        if self.id != STANDARD_CE_OBJECTIVE_ID and self.auxiliaries is not None:
            raise ValueError(
                "objective.auxiliaries is only supported for objective.id=standard_ce"
            )
        if self.coord_soft_ce is not None:
            if self.id != "recursive_detection_ce" or self.variant not in {
                "random_permutation_et_rmp_ce",
                "prefix_rollin_et_rmp_ce",
            }:
                raise ValueError(
                    "objective.coord_soft_ce is only supported for latest "
                    "recursive_detection_ce ET-RMP variants"
                )
        for field_name in ("state_weighting", "normalization"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"objective.{field_name} must be a string")
        _detection_validate_choice(
            self.state_weighting,
            path="objective.state_weighting",
            allowed=_DETECTION_STATE_WEIGHTINGS,
        )
        _detection_validate_choice(
            self.normalization,
            path="objective.normalization",
            allowed=_DETECTION_NORMALIZATIONS,
        )

    @classmethod
    def from_mapping(
        cls, payload: Any
    ) -> "DetectionObjectiveConfig | TeacherForcingObjectiveConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("objective must be a mapping")
        raw_id = payload.get("id")
        if raw_id == TEACHER_FORCING_OBJECTIVE_ID:
            return TeacherForcingObjectiveConfig.from_mapping(payload)
        if raw_id == STANDARD_CE_OBJECTIVE_ID:
            data: MutableMapping[str, Any] = dict(payload)
            data["auxiliaries"] = StandardCEAuxiliariesConfig.from_mapping(
                data.get("auxiliaries")
            )
            return parse_dataclass_strict(cls, data, path="objective")
        if raw_id == LEGACY_TEACHER_FORCING_OBJECTIVE_ID:
            raise ValueError(
                "objective.id='teacher_forcing' has been retired for active "
                "target-hierarchy configs; use objective.id: "
                "'research_teacher_forcing'"
            )
        if raw_id == "token_ce":
            raise ValueError(
                "objective.id='token_ce' is internal implementation/metric "
                "vocabulary; use objective.id: 'standard_ce'"
            )
        if raw_id in LEGACY_TEACHER_FORCING_OBJECTIVE_IDS or raw_id in {"sft"}:
            raise ValueError(
                "objective.id must be 'standard_ce' or 'research_teacher_forcing'; "
                f"legacy objective ids are unsupported, got {raw_id!r}"
            )
        raise ValueError(
            "objective.id must be 'standard_ce' or 'research_teacher_forcing'; "
            f"got {raw_id!r}"
        )

@dataclass(frozen=True)
class DetectionPackingConfig:
    static_packing: bool = False
    padding_free_packed: bool = False

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.static_packing,
            path="packing.static_packing",
        )
        _detection_validate_bool(
            self.padding_free_packed,
            path="packing.padding_free_packed",
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionPackingConfig":
        return parse_dataclass_strict(cls, payload, path="packing")


@dataclass(frozen=True)
class DetectionEvaluationConfig:
    expected_template: Literal[
        "stage1_json_pretty",
        "compact",
        "compact_box_closed",
        "compact_object_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    ]
    parser_mode: Literal["strict_expected", "diagnostic_salvage"] = "strict_expected"

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.expected_template,
            path="evaluation.expected_template",
            allowed=set(SUPPORTED_DETECTION_TEMPLATE_IDS),
        )
        _detection_validate_choice(
            self.parser_mode,
            path="evaluation.parser_mode",
            allowed={"strict_expected", "diagnostic_salvage"},
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionEvaluationConfig":
        if isinstance(payload, Mapping) and payload.get("expected_template") == "compact_full":
            payload = {**dict(payload), "expected_template": "compact"}
        return parse_dataclass_strict(cls, payload, path="evaluation")


@dataclass(frozen=True)
class DetectionValidationConfig:
    validate_span_alignment: bool = True
    validate_template_capabilities: bool = True
    fail_fast: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "validate_span_alignment",
            "validate_template_capabilities",
            "fail_fast",
        ):
            _detection_validate_bool(
                getattr(self, field_name),
                path=f"validation.{field_name}",
            )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionValidationConfig":
        return parse_dataclass_strict(cls, payload, path="validation")


def _detection_reject_removed_target_hierarchy_paths(payload: Mapping[str, Any]) -> None:
    if "pipeline_id" in payload:
        raise ValueError("pipeline_id is retired; use top-level pipeline.id")
    surface = payload.get("surface")
    if isinstance(surface, Mapping) and "id" in surface:
        raise ValueError("surface.id is not public config; use top-level pipeline.id")
    if "token_rows" in payload:
        raise ValueError(
            "flat token_rows is retired; use top-level token_embeddings_adapter"
        )

    data = payload.get("data")
    if isinstance(data, Mapping) and "object_ordering" in data:
        raise ValueError(
            "data.object_ordering is retired for target-hierarchy configs; use "
            "sample_factory.target_sequence.object_ordering"
        )

    custom = payload.get("custom")
    if custom is not None:
        if not isinstance(custom, Mapping):
            raise TypeError("custom must be a mapping when provided")
        guidance = {
            "trainer_variant": "pipeline.id",
            "object_ordering": "sample_factory.target_sequence.object_ordering",
            "object_field_order": "sample_factory.target_sequence.object_field_order",
            "detection_template_id": "detection_template.id",
            "detection_sequence_format": "sample_factory.id",
            "token_embeddings_adapter": "token_embeddings_adapter",
        }
        for key, new_path in guidance.items():
            if key in custom:
                raise ValueError(f"custom.{key} is retired; use {new_path}")
        raise ValueError("custom is obsolete for target-hierarchy detection configs")

    prompt = payload.get("prompt")
    if isinstance(prompt, Mapping) and "prompt_variant_enabled" in prompt:
        raise ValueError(
            "prompt.prompt_variant_enabled is retired; use prompt.variant"
        )

    sample_factory = payload.get("sample_factory")
    target_sequence = None
    if isinstance(sample_factory, Mapping):
        target_sequence = sample_factory.get("target_sequence")
    if isinstance(target_sequence, Mapping) and "template_id" in target_sequence:
        raise ValueError(
            "sample_factory.target_sequence.template_id is retired; use "
            "top-level detection_template.id"
        )

    detection_template = payload.get("detection_template")
    if (
        isinstance(detection_template, Mapping)
        and "strict_parse" in detection_template
        and isinstance(target_sequence, Mapping)
        and "strict_parse" in target_sequence
    ):
        raise ValueError(
            "detection_template.strict_parse is retired for active training configs; "
            "use sample_factory.target_sequence.strict_parse"
        )


def _detection_validate_pipeline_objective_pairing(
    pipeline: "DetectionPipelineConfig",
    objective: "DetectionObjectiveConfig | TeacherForcingObjectiveConfig | None",
) -> None:
    objective_id = getattr(objective, "id", None)
    if pipeline.id == "stage1_standard_sft":
        if objective_id != STANDARD_CE_OBJECTIVE_ID:
            raise ValueError(
                "pipeline.id=stage1_standard_sft requires "
                f"objective.id={STANDARD_CE_OBJECTIVE_ID}; got {objective_id!r}"
            )
        return

    if pipeline.id == "stage1_research_teacher_forcing":
        if objective_id != TEACHER_FORCING_OBJECTIVE_ID:
            raise ValueError(
                "pipeline.id=stage1_research_teacher_forcing requires "
                f"objective.id={TEACHER_FORCING_OBJECTIVE_ID}; got {objective_id!r}"
            )
        return

    if pipeline.id == "stage2_rollout_correction":
        if objective_id in {STANDARD_CE_OBJECTIVE_ID, TEACHER_FORCING_OBJECTIVE_ID}:
            raise ValueError(
                "pipeline.id=stage2_rollout_correction does not accept "
                f"objective.id={objective_id} in Slice 1A; use the Stage-2 "
                "rollout_correction runtime objective namespace instead of "
                "top-level Stage-1 objective.id"
            )


@dataclass(frozen=True)
class DetectionTrainingConfig:
    pipeline: DetectionPipelineConfig
    sample_factory: DetectionSampleFactoryConfig
    data: DetectionDataConfig
    prompt: DetectionPromptConfig
    detection_template: DetectionTemplateConfig
    token_embeddings_adapter: TokenEmbeddingsAdapterConfig
    objective: DetectionObjectiveConfig | TeacherForcingObjectiveConfig | None
    packing: DetectionPackingConfig
    evaluation: DetectionEvaluationConfig
    validation: DetectionValidationConfig
    experiment: Optional[DetectionExperimentConfig] = None
    debug: DebugConfig = field(default_factory=DebugConfig)
    model: Mapping[str, Any] = field(default_factory=dict)
    template: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    deepspeed: Mapping[str, Any] = field(default_factory=dict)
    rlhf: Mapping[str, Any] = field(default_factory=dict)
    tuner: Mapping[str, Any] = field(default_factory=dict)
    quantization: Mapping[str, Any] = field(default_factory=dict)
    global_max_length: Optional[int] = None
    stage2_rollout_correction: Optional[Stage2RolloutCorrectionConfig] = None
    rollout_matching: Optional[RolloutMatchingConfig] = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionTrainingConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("detection config payload must be a mapping")
        _detection_reject_removed_target_hierarchy_paths(payload)

        obsolete_paths = _detection_find_obsolete_keys_on_latest_surface(payload)
        if obsolete_paths:
            rendered = sorted(obsolete_paths)
            raise ValueError(f"Obsolete detection config keys: {rendered}")

        known_top_level = (
            _DETECTION_REQUIRED_SECTIONS
            | _DETECTION_RUNTIME_SECTIONS
            | _DETECTION_OPTIONAL_SECTIONS
        )
        unknown_top_level = sorted(
            str(k)
            for k in payload.keys()
            if not isinstance(k, str) or k not in known_top_level
        )
        if unknown_top_level:
            raise ValueError(
                f"Unknown detection config top-level keys: {unknown_top_level}"
            )

        required_sections = set(_DETECTION_REQUIRED_SECTIONS)
        required_sections.discard("objective")
        missing_sections = sorted(
            section for section in required_sections if section not in payload
        )
        if missing_sections:
            raise ValueError(
                f"Missing detection config sections: {missing_sections}"
            )

        global_max_length = payload.get("global_max_length")
        if global_max_length is not None:
            if (
                not isinstance(global_max_length, int)
                or isinstance(global_max_length, bool)
                or global_max_length <= 0
            ):
                raise ValueError("global_max_length must be a positive integer")

        pipeline = DetectionPipelineConfig.from_mapping(payload["pipeline"])
        objective = (
            DetectionObjectiveConfig.from_mapping(payload["objective"])
            if "objective" in payload
            else None
        )
        _detection_validate_pipeline_objective_pairing(pipeline, objective)
        if pipeline.id != "stage2_rollout_correction" and objective is None:
            raise ValueError(
                f"pipeline.id={pipeline.id} requires top-level objective.id"
            )
        stage2_rollout_correction = None
        rollout_matching = None
        if pipeline.id == "stage2_rollout_correction":
            stage2_rollout_correction_raw = payload.get("stage2_rollout_correction")
            if stage2_rollout_correction_raw is None:
                raise ValueError(
                    "pipeline.id=stage2_rollout_correction requires "
                    "stage2_rollout_correction.pipeline"
                )
            stage2_rollout_correction = Stage2RolloutCorrectionConfig.from_mapping(
                stage2_rollout_correction_raw
            )
            rollout_matching_raw = payload.get("rollout_matching")
            if rollout_matching_raw is None:
                raise ValueError(
                    "pipeline.id=stage2_rollout_correction requires rollout_matching"
                )
            if not isinstance(rollout_matching_raw, Mapping):
                raise TypeError("rollout_matching must be a mapping when provided")
            if "pipeline" in rollout_matching_raw:
                raise ValueError(
                    "rollout_matching.pipeline has been removed. Use "
                    "stage2_rollout_correction.pipeline with "
                    "pipeline.id=stage2_rollout_correction instead."
                )
            rollout_matching = parse_dataclass_strict(
                RolloutMatchingConfig,
                dict(rollout_matching_raw),
                path="rollout_matching",
            )
        sample_factory = DetectionSampleFactoryConfig.from_mapping(
            payload["sample_factory"]
        )
        target_sequence = sample_factory.target_sequence
        raw_detection_template = payload["detection_template"]
        raw_detection_template_id = (
            raw_detection_template.get("id")
            if isinstance(raw_detection_template, Mapping)
            else None
        )
        detection_template = DetectionTemplateConfig.from_mapping(
            raw_detection_template
        )
        if (
            detection_template.id == "stage1_json_pretty"
            and target_sequence.object_field_order != "desc_first"
        ):
            raise ValueError(
                "detection_template.id=stage1_json_pretty requires "
                "sample_factory.target_sequence.object_field_order=desc_first"
            )
        evaluation = DetectionEvaluationConfig.from_mapping(payload["evaluation"])
        if evaluation.expected_template != detection_template.id:
            raise ValueError(
                "evaluation.expected_template must match detection_template.id "
                f"({evaluation.expected_template!r} != {detection_template.id!r})"
            )
        data_config = replace(
            DetectionDataConfig.from_mapping(payload["data"]),
            object_ordering=target_sequence.object_ordering,
        )
        if objective is not None:
            _detection_validate_order_matches_objective(data_config, objective)
        if (
            objective is not None
            and getattr(objective, "id", None) != TEACHER_FORCING_OBJECTIVE_ID
            and objective.variant == "prefix_rollin_et_rmp_ce"
        ):
            required_experiment_variant = "prefix_rollin_et_rmp_ce"
        else:
            required_experiment_variant = None
        experiment = DetectionExperimentConfig.from_mapping(
            payload.get("experiment"),
            required_for_variant=required_experiment_variant,
        )
        _detection_validate_prefix_rollin_contract(
            detection_template=detection_template,
            objective=objective,
            experiment=experiment,
        )
        packing = DetectionPackingConfig.from_mapping(payload["packing"])
        training = _detection_validate_training_mapping(payload.get("training"))
        _detection_validate_teacher_forcing_coverage_ledger_contract(
            objective=objective,
            detection_template=detection_template,
            packing=packing,
            training=training,
            raw_detection_template_id=raw_detection_template_id,
        )
        token_rows = TokenEmbeddingsAdapterConfig.from_mapping(
            payload["token_embeddings_adapter"],
            path="token_embeddings_adapter",
        )
        _detection_validate_token_rows(detection_template, token_rows)
        _detection_validate_packing_runtime_contract(
            objective=objective,
            packing=packing,
            training=training,
        )

        return cls(
            pipeline=pipeline,
            sample_factory=sample_factory,
            data=data_config,
            prompt=DetectionPromptConfig.from_mapping(payload["prompt"]),
            detection_template=detection_template,
            token_embeddings_adapter=token_rows,
            objective=objective,
            packing=packing,
            evaluation=evaluation,
            validation=DetectionValidationConfig.from_mapping(payload["validation"]),
            experiment=experiment,
            debug=DebugConfig.from_mapping(payload.get("debug")),
            model=_detection_validate_framework_mapping(
                payload.get("model"),
                path="model",
                allowed=_train_arguments_allowed_keys(),
            ),
            template=_detection_validate_framework_mapping(
                payload.get("template"),
                path="template",
                allowed=_train_arguments_allowed_keys(),
            ),
            training=training,
            deepspeed=_detection_validate_deepspeed_mapping(
                payload.get("deepspeed")
            ),
            rlhf=_detection_validate_framework_mapping(
                payload.get("rlhf"),
                path="rlhf",
                allowed=_rlhf_arguments_allowed_keys(),
            ),
            tuner=_detection_validate_framework_mapping(
                payload.get("tuner"),
                path="tuner",
                allowed=_train_arguments_allowed_keys(),
            ),
            quantization=_detection_validate_framework_mapping(
                payload.get("quantization"),
                path="quantization",
                allowed=_train_arguments_allowed_keys(),
            ),
            global_max_length=global_max_length,
            stage2_rollout_correction=stage2_rollout_correction,
            rollout_matching=rollout_matching,
        )

    @property
    def token_rows(self) -> TokenEmbeddingsAdapterConfig:
        return self.token_embeddings_adapter

    def to_mapping(self) -> dict[str, Any]:
        payload = dataclass_asdict_no_none(self)
        for section in _DETECTION_RUNTIME_SECTIONS:
            if payload.get(section) == {}:
                payload.pop(section, None)
        data_payload = payload.get("data")
        if isinstance(data_payload, dict):
            data_payload.pop("object_ordering", None)
        token_groups = payload.get("token_embeddings_adapter", {}).get("groups", {})
        if isinstance(token_groups, dict):
            for group in token_groups.values():
                if not isinstance(group, dict):
                    continue
                if group.get("tokens") in ((), []):
                    group.pop("tokens", None)
                if group.get("expected_ids") == {}:
                    group.pop("expected_ids", None)
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    template: Mapping[str, Any]
    custom: CustomConfig
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    model: Mapping[str, Any] = field(default_factory=dict)
    quantization: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)
    tuner: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    objective: Optional[TeacherForcingObjectiveConfig] = None
    stage2_rollout_correction: Optional[Stage2RolloutCorrectionConfig] = None
    rollout_matching: Optional[RolloutMatchingConfig] = None
    rlhf: Mapping[str, Any] = field(default_factory=dict)
    prompts: PromptOverrides = field(default_factory=PromptOverrides)
    deepspeed: Optional[DeepSpeedConfig] = None
    global_max_length: Optional[int] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], prompts: PromptOverrides
    ) -> "TrainingConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("config payload must be a mapping")

        # Strict parsing policy:
        # - Unknown keys fail fast at load time with dotted-path reporting.
        # - Each top-level section is validated against schema-derived accepted keys.
        # - Top-level extra: is reserved/rejected; custom.extra is the only escape hatch.
        if "extra" in payload:
            raise ValueError(
                "Top-level extra: is unsupported under strict config parsing. "
                "Use custom.extra for minor residual knobs; unknown keys elsewhere are rejected."
            )

        data = dict(payload)

        model = dict(_as_dict(data.pop("model", None), path="model"))
        _validate_section_keys_strict(
            "model", model, allowed=_train_arguments_allowed_keys()
        )

        quantization = dict(
            _as_dict(data.pop("quantization", None), path="quantization")
        )
        _validate_section_keys_strict(
            "quantization", quantization, allowed=_train_arguments_allowed_keys()
        )

        template_raw = _as_dict(data.pop("template", None), path="template")
        template = dict(template_raw)
        _validate_section_keys_strict(
            "template", template, allowed=_train_arguments_allowed_keys()
        )

        data_section = dict(_as_dict(data.pop("data", None), path="data"))
        _validate_section_keys_strict(
            "data", data_section, allowed=_train_arguments_allowed_keys()
        )

        tuner = dict(_as_dict(data.pop("tuner", None), path="tuner"))
        _validate_section_keys_strict(
            "tuner", tuner, allowed=_train_arguments_allowed_keys()
        )

        training = dict(_as_dict(data.pop("training", None), path="training"))
        if "packing_length" in training:
            raise ValueError(
                "training.packing_length is deprecated and unsupported. "
                "Remove it and set global_max_length/template.max_length instead."
            )
        _validate_training_checkpoint_keys(training)
        _validate_section_keys_strict(
            "training", training, allowed=_training_allowed_keys()
        )
        if "encoded_sample_cache" in training:
            encoded_sample_cache = EncodedSampleCacheConfig.from_mapping(
                training.get("encoded_sample_cache")
            )
            training["encoded_sample_cache"] = encoded_sample_cache.to_mapping()
        if "static_packing_cache" in training:
            static_packing_cache = StaticPackingCacheConfig.from_mapping(
                training.get("static_packing_cache")
            )
            training["static_packing_cache"] = static_packing_cache.to_mapping()

        stage2_ab_raw = data.pop("stage2_ab", None)
        stage2_rollout_correction_raw = data.pop("stage2_rollout_correction", None)
        rollout_matching_raw = data.pop("rollout_matching", None)
        objective_raw = data.pop("objective", None)

        rlhf = dict(_as_dict(data.pop("rlhf", None), path="rlhf"))
        _validate_section_keys_strict(
            "rlhf", rlhf, allowed=_rlhf_arguments_allowed_keys()
        )
        custom_raw = data.pop("custom", None)
        custom_coord_soft_ce_w1_present = bool(
            isinstance(custom_raw, Mapping) and "coord_soft_ce_w1" in custom_raw
        )
        custom_bbox_geo_present = bool(
            isinstance(custom_raw, Mapping) and "bbox_geo" in custom_raw
        )
        custom_bbox_size_aux_present = bool(
            isinstance(custom_raw, Mapping) and "bbox_size_aux" in custom_raw
        )
        experiment = ExperimentConfig.from_mapping(data.pop("experiment", None))
        benchmark = BenchmarkConfig.from_mapping(data.pop("benchmark", None))
        debug = DebugConfig.from_mapping(data.pop("debug", None))
        deepspeed = DeepSpeedConfig.from_mapping(data.pop("deepspeed", None))
        global_max_length = data.pop("global_max_length", None)
        objective = None
        if objective_raw is not None:
            objective = TeacherForcingObjectiveConfig.from_mapping(objective_raw)

        if data:
            unknown = sorted(str(k) for k in data.keys())
            raise ValueError(
                "Unknown top-level config keys: "
                f"{unknown}. "
                "Migration guidance: keep only documented top-level sections; "
                "move residual experiment knobs under custom.extra."
            )

        if global_max_length is not None:
            if not isinstance(global_max_length, int) or global_max_length <= 0:
                raise ValueError(
                    "global_max_length must be a positive integer when provided"
                )

        if not template:
            raise ValueError("template section must be provided in the config")

        if prompts.system and "system" not in template:
            template["system"] = prompts.system

        custom = CustomConfig.from_mapping(custom_raw, prompts=prompts)
        trainer_variant = str(custom.trainer_variant or "")
        removed_two_channel_variant = "stage2_" "two_channel"
        removed_ab_training_variant = "stage2_" "ab_training"
        if trainer_variant == removed_two_channel_variant:
            raise ValueError(
                f"custom.trainer_variant={removed_two_channel_variant} has been removed; "
                "use stage2_rollout_correction"
            )
        if trainer_variant == removed_ab_training_variant:
            raise ValueError(
                f"custom.trainer_variant={removed_ab_training_variant} has been removed; "
                "use stage2_rollout_correction"
            )
        if trainer_variant == "rollout_matching_sft":
            raise ValueError(
                "custom.trainer_variant=rollout_matching_sft has been removed; "
                "use stage2_rollout_correction"
            )
        if trainer_variant in {"stage2_rollout_aligned", "stage2_rollout_runtime"}:
            raise ValueError(
                f"custom.trainer_variant={trainer_variant} has been removed; "
                "use stage2_rollout_correction"
            )
        if bool(getattr(custom.sft_structural_close, "enabled", False)):
            if bool(training.get("packing", False)):
                raise ValueError(
                    "custom.sft_structural_close requires training.packing=false "
                    "because final global-close token weights are sequence-local."
                )
            if bool(training.get("eval_packing", False)):
                raise ValueError(
                    "custom.sft_structural_close requires training.eval_packing=false "
                    "because final global-close token weights are sequence-local."
                )

        _validate_teacher_forcing_training_packing_contract(
            objective=objective,
            training=training,
        )

        if stage2_ab_raw is not None:
            raise ValueError(
                "stage2_ab has been removed from active Stage-2 configs; "
                "use stage2_rollout_correction."
            )

        stage2_rollout_correction = None
        if stage2_rollout_correction_raw is not None:
            stage2_rollout_correction = Stage2RolloutCorrectionConfig.from_mapping(
                stage2_rollout_correction_raw
            )
        elif trainer_variant == "stage2_rollout_correction":
            raise ValueError(
                "stage2_rollout_correction section must be provided when "
                "custom.trainer_variant=stage2_rollout_correction"
            )
        rollout_matching = None
        if rollout_matching_raw is not None:
            if not isinstance(rollout_matching_raw, Mapping):
                raise TypeError("rollout_matching must be a mapping when provided")
            if "coord_decode_mode" in rollout_matching_raw:
                raise ValueError(
                    "rollout_matching.coord_decode_mode is deprecated and unsupported in active/training "
                    "configs. Remove it; Stage-2 geometry decode now uses the fixed expectation-decode baseline."
                )
            if "pipeline" in rollout_matching_raw:
                raise ValueError(
                    "rollout_matching.pipeline has been removed. "
                    "Use stage2_rollout_correction.pipeline with "
                    "pipeline.id=stage2_rollout_correction instead."
                )

            # Preserve prior strictness: an explicitly empty mapping counts as "missing".
            if not rollout_matching_raw:
                rollout_matching = None
            else:
                # BREAKING: legacy paired-list server form is removed.
                vllm_raw = rollout_matching_raw.get("vllm")
                if isinstance(vllm_raw, Mapping):
                    server_raw = vllm_raw.get("server")
                    if isinstance(server_raw, Mapping) and (
                        "base_url" in server_raw or "group_port" in server_raw
                    ):
                        raise ValueError(
                            "Legacy rollout server config has been removed: "
                            "rollout_matching.vllm.server.base_url/group_port. "
                            "Use rollout_matching.vllm.server.servers[] (list of {base_url, group_port})."
                        )

                rollout_matching = parse_dataclass_strict(
                    RolloutMatchingConfig,
                    rollout_matching_raw,
                    path="rollout_matching",
                )

        if trainer_variant == "stage2_rollout_correction":
            if rollout_matching is None:
                raise ValueError(
                    "rollout_matching section must be provided for stage2_rollout_correction"
                )

        stage2_pipeline_present = bool(
            stage2_rollout_correction is not None
            and getattr(stage2_rollout_correction, "pipeline", None) is not None
        )

        if stage2_pipeline_present and custom_coord_soft_ce_w1_present:
            raise ValueError(
                "stage2_rollout_correction.pipeline is provided; custom.coord_soft_ce_w1.* is disallowed. "
                "Coordinate regularizers have been removed from the active Stage-2 pipeline."
            )
        if stage2_pipeline_present and custom_bbox_geo_present:
            raise ValueError(
                "stage2_rollout_correction.pipeline is provided; custom.bbox_geo.* is disallowed. "
                "bbox geometry auxiliaries have been removed from the active Stage-2 pipeline."
            )
        if stage2_pipeline_present and custom_bbox_size_aux_present:
            raise ValueError(
                "stage2_rollout_correction.pipeline is provided; custom.bbox_size_aux.* is disallowed. "
                "bbox size auxiliaries have been removed from the active Stage-2 pipeline."
            )
        # Length-coherence guardrails (fail-fast). These settings affect whether the
        # eval-step vLLM backend will truncate/error on long prompts, which is
        # objective-changing.
        if rollout_matching is not None:
            backend = (
                str(getattr(rollout_matching, "rollout_backend", "") or "")
                .strip()
                .lower()
            )
            if backend not in {"hf", "vllm"}:
                raise ValueError(
                    "rollout_matching.rollout_backend must be one of {'hf', 'vllm'}."
                )

            effective_eval_backend = (
                str(getattr(rollout_matching, "eval_rollout_backend", "") or "")
                .strip()
                .lower()
            )
            if effective_eval_backend not in {"hf", "vllm"}:
                raise ValueError(
                    "rollout_matching.eval_rollout_backend must be one of {'hf', 'vllm'}."
                )

            if backend == "vllm" or effective_eval_backend == "vllm":
                vllm_cfg = getattr(rollout_matching, "vllm", None)
                enable_lora = bool(getattr(vllm_cfg, "enable_lora", False))
                sync_cfg = getattr(vllm_cfg, "sync", None)
                sync_mode = (
                    str(getattr(sync_cfg, "mode", "full") or "full")
                    .strip()
                    .lower()
                )
                if sync_mode != "adapter":
                    raise ValueError(
                        "vLLM rollouts require official adapter sync: set "
                        "rollout_matching.vllm.sync.mode=adapter."
                    )
                if not enable_lora:
                    raise ValueError(
                        "vLLM rollouts require official adapter sync: set "
                        "rollout_matching.vllm.enable_lora=true."
                    )
                if enable_lora and sync_mode != "adapter":
                    raise ValueError(
                        "rollout_matching.vllm.enable_lora=true requires "
                        "rollout_matching.vllm.sync.mode=adapter."
                    )
                if sync_mode == "adapter" and not enable_lora:
                    raise ValueError(
                        "rollout_matching.vllm.sync.mode=adapter requires "
                        "rollout_matching.vllm.enable_lora=true."
                    )
                vllm_max_model_len_raw = getattr(vllm_cfg, "max_model_len", None)
                max_new_tokens_raw = getattr(rollout_matching, "max_new_tokens", None)

                if vllm_max_model_len_raw is not None:
                    vllm_max_model_len = int(vllm_max_model_len_raw)
                    if vllm_max_model_len <= 0:
                        raise ValueError(
                            "rollout_matching.vllm.max_model_len must be > 0 when provided."
                        )

                    if max_new_tokens_raw is not None:
                        max_new_tokens = int(max_new_tokens_raw)
                        if max_new_tokens >= vllm_max_model_len:
                            raise ValueError(
                                "rollout_matching.max_new_tokens must be < rollout_matching.vllm.max_model_len "
                                f"to avoid truncation/overflow. Got max_new_tokens={max_new_tokens} "
                                f"vllm.max_model_len={vllm_max_model_len}."
                            )

                    if global_max_length is not None and vllm_max_model_len < int(
                        global_max_length
                    ):
                        raise ValueError(
                            "rollout_matching.vllm.max_model_len must be >= global_max_length to avoid "
                            "silent truncation drift between training and rollouts. "
                            f"Got global_max_length={int(global_max_length)} vllm.max_model_len={vllm_max_model_len}."
                        )

        if custom.bbox_format in {"cxcy_logw_logh", "cxcywh"}:
            bbox_format_label = str(custom.bbox_format)
            if trainer_variant == removed_two_channel_variant:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} is Stage-1-only in V1 and is unsupported for stage2 trainer variants."
                )
            if stage2_pipeline_present:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} is Stage-1-only in V1 and cannot be combined with stage2_rollout_correction.pipeline."
                )
            if not bool(getattr(custom.coord_tokens, "enabled", False)):
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_tokens.enabled=true."
                )
            if not bool(getattr(custom.coord_tokens, "skip_bbox_norm", False)):
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_tokens.skip_bbox_norm=true."
                )

            coord_cfg = custom.coord_soft_ce_w1
            if not bool(getattr(coord_cfg, "enabled", False)):
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.enabled=true."
                )
            if float(getattr(coord_cfg, "ce_weight", 0.0)) <= 0.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.ce_weight > 0."
                )
            if float(getattr(coord_cfg, "soft_ce_weight", 0.0)) != 0.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.soft_ce_weight = 0."
                )
            if float(getattr(coord_cfg, "w1_weight", 0.0)) != 0.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.w1_weight = 0."
                )
            if float(getattr(coord_cfg, "gate_weight", 0.0)) <= 0.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.gate_weight > 0."
                )
            if float(getattr(coord_cfg, "text_gate_weight", 0.0)) <= 0.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.text_gate_weight > 0."
                )
            if float(getattr(coord_cfg, "temperature", 1.0)) != 1.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.temperature = 1.0."
                )
            if float(getattr(coord_cfg, "target_sigma", 2.0)) != 2.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.target_sigma = 2.0."
                )
            if getattr(coord_cfg, "target_truncate", None) is not None:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.target_truncate = null."
                )
            if custom_bbox_geo_present or bool(
                getattr(custom.bbox_geo, "enabled", False)
            ):
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} rejects custom.bbox_geo in V1."
                )
            if custom_bbox_size_aux_present or bool(
                getattr(custom.bbox_size_aux, "enabled", False)
            ):
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} rejects custom.bbox_size_aux in V1."
                )

        return cls(
            template=template,
            custom=custom,
            experiment=experiment,
            benchmark=benchmark,
            debug=debug,
            model=model,
            quantization=quantization,
            data=data_section,
            tuner=tuner,
            training=training,
            objective=objective,
            stage2_rollout_correction=stage2_rollout_correction,
            rollout_matching=rollout_matching,
            rlhf=rlhf,
            prompts=prompts,
            deepspeed=deepspeed,
            global_max_length=global_max_length,
            extra={},
        )
