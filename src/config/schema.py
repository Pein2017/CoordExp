"""Typed configuration schemas for training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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
    BOX_START_TOKEN,
    COORD_END_TOKEN,
    COORD_START_TOKEN,
    EXPECTED_BOX_START_ID,
    EXPECTED_COORD_END_ID,
    EXPECTED_COORD_START_ID,
    EXPECTED_OBJECT_REF_START_ID,
    OBJECT_REF_START_TOKEN,
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

from .eval_monitor_dump_schema import EvalMonitorDumpConfig
from .rollout_matching_schema import RolloutMatchingConfig
from .strict_dataclass import dataclass_asdict_no_none, parse_dataclass_strict


AllowedNorm = Literal["none", "norm100", "norm1000"]
AllowedVisualDistance = Literal["mse", "cosine"]
AllowedJsonFormat = Literal["standard"]

ALLOWED_JSON_FORMATS: set[str] = {"standard"}
STAGE2_CHANNEL_B_FP_POLICIES: set[str] = {
    "zero_loss_context",
    "weak_positive_context",
}
STAGE2_TRIE_CE_MODULE_NAME = "stage2_trie_ce"
STAGE2_TRIE_CE_CONFIG_KEYS: set[str] = {
    "support_weight",
    "balance_weight",
    "struct_weight",
    "desc_weight",
    "coord_hard_ce_weight",
    "eos_weight",
    "normalization",
}
STAGE2_TRIE_CE_NORMALIZATIONS: set[str] = {
    "token_mean",
}
STAGE2_TRIE_CE_RESERVED_WEIGHT_KEYS: set[str] = (
    STAGE2_TRIE_CE_CONFIG_KEYS - {"normalization"}
)
STAGE2_TRIE_CE_APPLICATION_PRESETS: set[str] = {"rollout_trie_hard_ce"}
STAGE2_RESIDUAL_SET_MODULE_NAME = "residual_set_correction"
STAGE2_RESIDUAL_SET_APPLICATION_PRESETS: set[str] = {"rollout_self_prefix"}
STAGE2_RESIDUAL_SET_CONFIG_KEYS: set[str] = {
    "rollin_policy",
    "rollin_resample_policy",
    "base_seed",
    "coord_span_policy",
    "strict_builder_invariants",
    "lambda_ul_promoted",
    "lambda_continue_margin",
    "continue_margin_m",
    "coverage_strength",
    "num_rollouts",
    "min_ul_valid_rollouts",
    "ul_consensus_ratio",
    "ul_geometry",
    "artifact_policy",
}
STAGE2_RESIDUAL_SET_UL_GEOMETRY_KEYS: set[str] = {
    "iou_min",
    "center_distance_scale_max",
    "area_ratio_max",
    "aspect_ratio_max",
    "consumed_overlap_iou_min",
}
STAGE2_RESIDUAL_SET_ARTIFACT_POLICY_KEYS: set[str] = {
    "ul_clusters",
}
TEACHER_FORCING_OBJECTIVE_ID = "teacher_forcing"
TEACHER_FORCING_PROFILES: set[str] = {
    "hard_sft",
    "pure_valid_set_marginal",
    "hybrid_valid_set_marginal",
}
LEGACY_TEACHER_FORCING_OBJECTIVE_IDS: set[str] = {
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


@lru_cache(maxsize=1)
def _train_arguments_allowed_keys() -> set[str]:
    # Schema-derived strict key acceptance for ms-swift TrainArguments-driven sections.
    from swift.llm.argument import TrainArguments

    return {f.name for f in fields(TrainArguments)}


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
    from swift.llm.argument import RLHFArguments

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
        cls, payload: Optional[Mapping[str, Any]]
    ) -> "CoordSoftCEW1Config":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("coord_soft_ce_w1 section must be a mapping when provided")

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
                f"Unknown coord_soft_ce_w1 keys: {[f'coord_soft_ce_w1.{key}' for key in unknown]}"
            )

        enabled = bool(payload.get("enabled", False))

        def _parse_float(key: str, default: float) -> float:
            raw = payload.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"coord_soft_ce_w1.{key} must be numeric") from exc

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
                    "coord_soft_ce_w1.target_truncate must be an integer or null"
                ) from exc

        if ce_weight < 0:
            raise ValueError("coord_soft_ce_w1.ce_weight must be >= 0")
        if soft_ce_weight < 0:
            raise ValueError("coord_soft_ce_w1.soft_ce_weight must be >= 0")
        if w1_weight < 0:
            raise ValueError("coord_soft_ce_w1.w1_weight must be >= 0")
        if gate_weight < 0:
            raise ValueError("coord_soft_ce_w1.gate_weight must be >= 0")
        if text_gate_weight < 0:
            raise ValueError("coord_soft_ce_w1.text_gate_weight must be >= 0")
        if (
            enabled
            and ce_weight == 0
            and soft_ce_weight == 0
            and w1_weight == 0
            and gate_weight == 0
            and text_gate_weight == 0
        ):
            raise ValueError(
                "coord_soft_ce_w1 is enabled but ce_weight, soft_ce_weight, w1_weight, gate_weight, and text_gate_weight are all 0"
            )
        if temperature <= 0:
            raise ValueError("coord_soft_ce_w1.temperature must be > 0")
        if target_sigma <= 0:
            raise ValueError("coord_soft_ce_w1.target_sigma must be > 0")
        if target_truncate is not None and target_truncate < 0:
            raise ValueError("coord_soft_ce_w1.target_truncate must be >= 0 or null")

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
class CoordOffsetConfig:
    enabled: bool = False
    tie_head: bool = True
    ids: tuple[int, ...] = ()
    embed_lr: Optional[float] = None
    head_lr: Optional[float] = None
    weight_decay: float = 0.0
    dtype: Optional[str] = None  # "auto"/None defaults to model dtype

    def __post_init__(self) -> None:
        if self.weight_decay < 0:
            raise ValueError("coord_offset.weight_decay must be >= 0")

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CoordOffsetConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("coord_offset section must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))

        tie_head_raw = payload.get("tie_head", True)
        if tie_head_raw is None:
            tie_head = True
        elif isinstance(tie_head_raw, bool):
            tie_head = tie_head_raw
        else:
            raise TypeError("coord_offset.tie_head must be a boolean when provided")

        ids_raw = payload.get("ids")
        ids: tuple[int, ...]
        if ids_raw is None:
            ids = ()
        elif isinstance(ids_raw, (list, tuple, set)):
            try:
                ids = tuple(int(v) for v in ids_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("coord_offset.ids must be a list of integers") from exc
        elif isinstance(ids_raw, Mapping):
            start = ids_raw.get("start")
            end = ids_raw.get("end")
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "coord_offset.ids mapping must provide integer start/end"
                ) from exc
            if end_i < start_i:
                raise ValueError("coord_offset.ids.end must be >= start")
            ids = tuple(range(start_i, end_i + 1))
        else:
            raise TypeError(
                "coord_offset.ids must be a list, mapping with start/end, or omitted"
            )

        def _parse_lr(key: str) -> Optional[float]:
            raw = payload.get(key)
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"coord_offset.{key} must be numeric") from exc

        embed_lr = _parse_lr("embed_lr")
        head_lr = _parse_lr("head_lr")

        weight_decay_raw = payload.get("weight_decay", 0.0)
        try:
            weight_decay = float(weight_decay_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("coord_offset.weight_decay must be numeric") from exc
        if weight_decay < 0:
            raise ValueError("coord_offset.weight_decay must be >= 0")

        dtype_raw = payload.get("dtype")
        dtype = str(dtype_raw) if dtype_raw is not None else None

        return cls(
            enabled=enabled,
            tie_head=tie_head,
            ids=ids,
            embed_lr=embed_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
            dtype=dtype,
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
class TrainableTokenRowsConfig:
    enabled: bool = False
    tie_head: bool = True
    groups: Mapping[str, TrainableTokenRowGroupConfig] = field(default_factory=dict)
    embed_lr: Optional[float] = None
    head_lr: Optional[float] = None
    weight_decay: float = 0.0
    dtype: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weight_decay < 0:
            raise ValueError("custom.trainable_token_rows.weight_decay must be >= 0")

    @classmethod
    def from_mapping(
        cls,
        payload: Optional[Mapping[str, Any]],
        *,
        path: str = "custom.trainable_token_rows",
    ) -> "TrainableTokenRowsConfig":
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
                tokenizer, path=f"trainable_token_rows.groups.{name}"
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
    coord_tokens: CoordTokensConfig = field(default_factory=CoordTokensConfig)
    coord_offset: CoordOffsetConfig = field(default_factory=CoordOffsetConfig)
    trainable_token_rows: TrainableTokenRowsConfig = field(
        default_factory=TrainableTokenRowsConfig
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
        if detection_sequence_format != COORDJSON_FORMAT and not coord_tokens.enabled:
            raise ValueError(
                "custom.detection_sequence_format="
                f"{detection_sequence_format} requires custom.coord_tokens.enabled=true"
            )
        coord_offset_raw = data.pop("coord_offset", None)
        coord_offset = CoordOffsetConfig.from_mapping(coord_offset_raw)
        trainable_token_rows_raw = data.pop("trainable_token_rows", None)
        trainable_token_rows = TrainableTokenRowsConfig.from_mapping(
            trainable_token_rows_raw
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
            coord_tokens=coord_tokens,
            coord_offset=coord_offset,
            trainable_token_rows=trainable_token_rows,
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
class Stage2ABScheduleConfig:
    """Deterministic Stage-2 AB channel schedule."""

    b_ratio: float

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2ABScheduleConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab.schedule must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)

        if "pattern" in data:
            raise ValueError(
                "stage2_ab.schedule.pattern is not supported. "
                "Use stage2_ab.schedule.b_ratio (float in [0,1]) instead."
            )

        if "b_ratio" not in data:
            raise ValueError(
                "stage2_ab.schedule.b_ratio must be provided (float in [0,1]); "
                "e.g. 0.0=A-only, 1.0=B-only, 0.05=~5% B."
            )
        b_ratio_raw = data.pop("b_ratio", None)
        try:
            b_ratio = float(b_ratio_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.schedule.b_ratio must be a float in [0,1]"
            ) from exc

        if not (0.0 <= b_ratio <= 1.0):
            raise ValueError(
                f"stage2_ab.schedule.b_ratio must be in [0,1], got {b_ratio!r}"
            )

        if data:
            raise ValueError(
                f"Unknown stage2_ab.schedule keys: {sorted(str(k) for k in data.keys())}"
            )

        return cls(b_ratio=b_ratio)


@dataclass(frozen=True)
class Stage2ABChannelBTriagePosteriorConfig:
    num_rollouts: int = 2
    explorer_temperature: float = 0.7
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
    ) -> "Stage2ABChannelBTriagePosteriorConfig":
        if payload is None:
            return cls(
                num_rollouts=cls.num_rollouts
                if default_num_rollouts is None
                else default_num_rollouts
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                "stage2_ab.channel_b.triage_posterior must be a mapping when provided"
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
                "stage2_ab.channel_b.triage_posterior.num_rollouts must be an int"
            ) from exc
        if num_rollouts < 2:
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.num_rollouts must be >= 2"
            )

        explorer_temperature_raw = data.pop(
            "explorer_temperature",
            cls.explorer_temperature,
        )
        try:
            explorer_temperature = float(explorer_temperature_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.triage_posterior.explorer_temperature must be a float/int"
            ) from exc
        if not math.isfinite(explorer_temperature):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.explorer_temperature must be finite"
            )
        if explorer_temperature < 0.0:
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.explorer_temperature must be >= 0"
            )

        explorer_top_p_raw = data.pop("explorer_top_p", cls.explorer_top_p)
        try:
            explorer_top_p = float(
                cls.explorer_top_p if explorer_top_p_raw is None else explorer_top_p_raw
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.triage_posterior.explorer_top_p must be a float/int"
            ) from exc
        if not math.isfinite(explorer_top_p):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.explorer_top_p must be finite"
            )
        if not (0.0 < explorer_top_p <= 1.0):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.explorer_top_p must be in (0, 1]"
            )

        explorer_top_k_raw = data.pop("explorer_top_k", cls.explorer_top_k)
        try:
            explorer_top_k = int(explorer_top_k_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.triage_posterior.explorer_top_k must be an int"
            ) from exc
        if explorer_top_k != -1 and explorer_top_k < 1:
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.explorer_top_k must be -1 (disabled) or >= 1"
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
                "stage2_ab.channel_b.triage_posterior.unlabeled_consistent_iou_threshold must be a float/int"
            ) from exc
        if not math.isfinite(unlabeled_consistent_iou_threshold):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.unlabeled_consistent_iou_threshold must be finite"
            )
        if (
            unlabeled_consistent_iou_threshold < 0.0
            or unlabeled_consistent_iou_threshold > 1.0
        ):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.unlabeled_consistent_iou_threshold must be in [0, 1]"
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
                "stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier must be a float/int"
            ) from exc
        if not math.isfinite(recovered_ground_truth_weight_multiplier):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier must be finite"
            )
        if recovered_ground_truth_weight_multiplier < 1.0:
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier must be >= 1.0"
            )

        if data:
            unknown = [
                f"stage2_ab.channel_b.triage_posterior.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_ab.channel_b.triage_posterior keys: {unknown}"
            )

        return cls(
            num_rollouts=num_rollouts,
            explorer_temperature=explorer_temperature,
            explorer_top_p=explorer_top_p,
            explorer_top_k=explorer_top_k,
            unlabeled_consistent_iou_threshold=unlabeled_consistent_iou_threshold,
            recovered_ground_truth_weight_multiplier=recovered_ground_truth_weight_multiplier,
        )


@dataclass(frozen=True)
class Stage2ABChannelBPseudoPositiveConfig:
    enabled: bool = False
    coord_weight: float = 0.5

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2ABChannelBPseudoPositiveConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError(
                "stage2_ab.channel_b.pseudo_positive must be a mapping when provided"
            )

        data: MutableMapping[str, Any] = dict(payload)

        versioned = [
            f"stage2_ab.channel_b.pseudo_positive.{str(key)}"
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
                    "stage2_ab.channel_b.pseudo_positive.enabled must be boolean (0 or 1)"
                )
        elif isinstance(enabled_raw, str):
            normalized = enabled_raw.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                enabled = True
            elif normalized in {"false", "0", "no", "n", "off"}:
                enabled = False
            else:
                raise ValueError(
                    "stage2_ab.channel_b.pseudo_positive.enabled string value "
                    f"'{enabled_raw}' is not a recognized boolean representation."
                )
        else:
            raise TypeError(
                "stage2_ab.channel_b.pseudo_positive.enabled must be a boolean value"
            )

        coord_weight_raw = data.pop("coord_weight", cls.coord_weight)
        try:
            coord_weight = float(coord_weight_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.pseudo_positive.coord_weight must be a float/int"
            ) from exc
        if not math.isfinite(coord_weight):
            raise ValueError(
                "stage2_ab.channel_b.pseudo_positive.coord_weight must be finite"
            )
        if not (0.0 < coord_weight < 1.0):
            raise ValueError(
                "stage2_ab.channel_b.pseudo_positive.coord_weight must be in (0, 1)"
            )

        if data:
            unknown = [
                f"stage2_ab.channel_b.pseudo_positive.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_ab.channel_b.pseudo_positive keys: {unknown}"
            )

        return cls(enabled=enabled, coord_weight=coord_weight)


@dataclass(frozen=True)
class Stage2ABChannelBDuplicateControlConfig:
    iou_threshold: float = 0.90
    center_radius_scale: float = 0.80

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2ABChannelBDuplicateControlConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError(
                "stage2_ab.channel_b.duplicate_control must be a mapping when provided"
            )

        data: MutableMapping[str, Any] = dict(payload)

        iou_threshold_raw = data.pop("iou_threshold", cls.iou_threshold)
        try:
            iou_threshold = float(iou_threshold_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.duplicate_control.iou_threshold must be a float/int"
            ) from exc
        if not math.isfinite(iou_threshold):
            raise ValueError(
                "stage2_ab.channel_b.duplicate_control.iou_threshold must be finite"
            )
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(
                "stage2_ab.channel_b.duplicate_control.iou_threshold must be in [0, 1]"
            )

        center_radius_scale_raw = data.pop(
            "center_radius_scale",
            cls.center_radius_scale,
        )
        try:
            center_radius_scale = float(center_radius_scale_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.duplicate_control.center_radius_scale must be a float/int"
            ) from exc
        if not math.isfinite(center_radius_scale):
            raise ValueError(
                "stage2_ab.channel_b.duplicate_control.center_radius_scale must be finite"
            )
        if center_radius_scale < 0.0:
            raise ValueError(
                "stage2_ab.channel_b.duplicate_control.center_radius_scale must be >= 0"
            )

        if data:
            unknown = [
                f"stage2_ab.channel_b.duplicate_control.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_ab.channel_b.duplicate_control keys: {unknown}"
            )

        return cls(
            iou_threshold=iou_threshold,
            center_radius_scale=center_radius_scale,
        )


@dataclass(frozen=True)
class Stage2ABChannelBAssignmentConfig:
    strategy: str = "greedy_iou"
    iou_threshold: Optional[float] = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2ABChannelBAssignmentConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab.channel_b.assignment must be a mapping when provided")

        data: MutableMapping[str, Any] = dict(payload)
        strategy_raw = data.pop("strategy", cls.strategy)
        strategy = str(strategy_raw).strip().lower().replace("-", "_")
        if strategy == "legacy_hungarian_mask_iou":
            raise ValueError(
                "stage2_ab.channel_b.assignment.strategy=legacy_hungarian_mask_iou "
                "has been removed; use greedy_iou"
            )
        if strategy != "greedy_iou":
            raise ValueError(
                "stage2_ab.channel_b.assignment.strategy must be one of "
                "{'greedy_iou'}"
            )

        iou_threshold_raw = data.pop("iou_threshold", None)
        iou_threshold: Optional[float] = None
        if iou_threshold_raw is not None:
            try:
                iou_threshold = float(iou_threshold_raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "stage2_ab.channel_b.assignment.iou_threshold must be a float/int when set"
                ) from exc
            if not math.isfinite(iou_threshold):
                raise ValueError(
                    "stage2_ab.channel_b.assignment.iou_threshold must be finite"
                )
            if iou_threshold < 0.0 or iou_threshold > 1.0:
                raise ValueError(
                    "stage2_ab.channel_b.assignment.iou_threshold must be in [0, 1]"
                )

        if data:
            unknown = [
                f"stage2_ab.channel_b.assignment.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_ab.channel_b.assignment keys: {unknown}"
            )

        return cls(
            strategy=strategy,
            iou_threshold=iou_threshold,
        )


@dataclass(frozen=True)
class Stage2ABChannelBFalsePositivePolicyConfig:
    mode: str = "zero_loss_context"
    weak_positive_weight: float = 0.05
    require_explorer_support: bool = True
    min_support_count: int = 1
    require_token_score: bool = False

    @classmethod
    def from_mapping(
        cls, payload: Any
    ) -> "Stage2ABChannelBFalsePositivePolicyConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab.channel_b.fp_policy must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)

        mode_raw = data.pop("mode", cls.mode)
        mode = str(mode_raw).strip().lower().replace("-", "_")
        if mode not in STAGE2_CHANNEL_B_FP_POLICIES:
            raise ValueError(
                "stage2_ab.channel_b.fp_policy.mode must be one of "
                f"{sorted(STAGE2_CHANNEL_B_FP_POLICIES)}"
            )

        weak_positive_weight_raw = data.pop(
            "weak_positive_weight", cls.weak_positive_weight
        )
        try:
            weak_positive_weight = float(weak_positive_weight_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.fp_policy.weak_positive_weight must be a float/int"
            ) from exc
        if isinstance(weak_positive_weight_raw, bool):
            raise TypeError(
                "stage2_ab.channel_b.fp_policy.weak_positive_weight must be a float/int, not bool"
            )
        if not math.isfinite(weak_positive_weight):
            raise ValueError(
                "stage2_ab.channel_b.fp_policy.weak_positive_weight must be finite"
            )
        if weak_positive_weight < 0.0:
            raise ValueError(
                "stage2_ab.channel_b.fp_policy.weak_positive_weight must be >= 0"
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
            path="stage2_ab.channel_b.fp_policy.require_explorer_support",
        )
        require_token_score = _parse_bool(
            data.pop("require_token_score", cls.require_token_score),
            path="stage2_ab.channel_b.fp_policy.require_token_score",
        )

        min_support_count_raw = data.pop(
            "min_support_count", cls.min_support_count
        )
        if isinstance(min_support_count_raw, bool):
            raise TypeError(
                "stage2_ab.channel_b.fp_policy.min_support_count must be an int, not bool"
            )
        if isinstance(min_support_count_raw, float):
            if not math.isfinite(min_support_count_raw):
                raise ValueError(
                    "stage2_ab.channel_b.fp_policy.min_support_count must be finite"
                )
            if not min_support_count_raw.is_integer():
                raise ValueError(
                    "stage2_ab.channel_b.fp_policy.min_support_count must be an integer"
                )
            min_support_count = int(min_support_count_raw)
        elif isinstance(min_support_count_raw, int):
            min_support_count = min_support_count_raw
        elif isinstance(min_support_count_raw, str):
            normalized_count = min_support_count_raw.strip()
            if not normalized_count or not normalized_count.lstrip("+-").isdigit():
                raise ValueError(
                    "stage2_ab.channel_b.fp_policy.min_support_count must be an int"
                )
            min_support_count = int(normalized_count)
        else:
            raise TypeError(
                "stage2_ab.channel_b.fp_policy.min_support_count must be an int"
            )
        if min_support_count < 1:
            raise ValueError(
                "stage2_ab.channel_b.fp_policy.min_support_count must be >= 1"
            )

        if data:
            unknown = [
                f"stage2_ab.channel_b.fp_policy.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                f"Unknown stage2_ab.channel_b.fp_policy keys: {unknown}"
            )

        return cls(
            mode=mode,
            weak_positive_weight=weak_positive_weight,
            require_explorer_support=require_explorer_support,
            min_support_count=min_support_count,
            require_token_score=require_token_score,
        )


@dataclass(frozen=True)
class Stage2ABChannelBConfig:
    assignment: Stage2ABChannelBAssignmentConfig = field(
        default_factory=Stage2ABChannelBAssignmentConfig
    )
    duplicate_control: Stage2ABChannelBDuplicateControlConfig = field(
        default_factory=Stage2ABChannelBDuplicateControlConfig
    )
    producer_wait_timeout_s: Optional[float] = None
    ddp_phase_timeout_s: Optional[float] = None
    rollout_template_family: str = "coordjson"
    rollout_decode_policy: str = "legacy_coordjson"
    fallback_loss_weight: float = 1.0
    invalid_rollout_policy: str = "abort"
    insertion_order: str = "tail_append"
    fp_policy: Stage2ABChannelBFalsePositivePolicyConfig = field(
        default_factory=Stage2ABChannelBFalsePositivePolicyConfig
    )
    pseudo_positive: Stage2ABChannelBPseudoPositiveConfig = field(
        default_factory=Stage2ABChannelBPseudoPositiveConfig
    )
    triage_posterior: Stage2ABChannelBTriagePosteriorConfig = field(
        default_factory=Stage2ABChannelBTriagePosteriorConfig
    )

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        validate_legacy_rollouts: bool = True,
    ) -> "Stage2ABChannelBConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab.channel_b must be a mapping when provided")

        data: MutableMapping[str, Any] = dict(payload)

        # Removed keys (single step-budgeted pathway; no legacy knobs).
        if "mode" in data:
            raise ValueError(
                "stage2_ab.channel_b.mode has been removed. "
                "Remove this key (Channel-B is always step-budgeted)."
            )
        if "async" in data:
            raise ValueError(
                "stage2_ab.channel_b.async has been removed. "
                "Remove this key (async actor-learner is unsupported)."
            )
        if "rollouts_per_step" in data:
            raise ValueError(
                "stage2_ab.channel_b.rollouts_per_step has been removed. "
                "Use training.effective_batch_size to control raw rollouts per optimizer step."
            )
        if "enable_pipeline" in data:
            raise ValueError(
                "stage2_ab.channel_b.enable_pipeline has been removed. "
                "Pipeline overlap is runtime-managed under vLLM server mode; "
                "under DDP it may be disabled for safety."
            )
        if "rollout_decode_batch_size" in data:
            raise ValueError(
                "stage2_ab.channel_b.rollout_decode_batch_size has been removed. "
                "Use rollout_matching.channel_b_decode_batch_size instead."
            )
        for key in sorted(data.keys(), key=lambda x: str(x)):
            if isinstance(key, str) and _is_versioned_alias_for(
                key, "invalid_rollout_policy"
            ):
                raise ValueError(
                    "Versioned invalid-rollout policy aliases are unsupported; "
                    "use stage2_ab.channel_b.invalid_rollout_policy instead."
                )

        if "drop_invalid_struct_ce_multiplier" in data:
            raise ValueError(
                "stage2_ab.channel_b.drop_invalid_struct_ce_multiplier has been removed. "
                "Legacy raw-prefix invalid-structure amplification is not part of the "
                "canonical clean-prefix Channel-B contract."
            )

        # Removed keys (legacy/ablation-only behavior is now deleted).
        if "reordered_gt_sft" in data:
            raise ValueError(
                "stage2_ab.channel_b.reordered_gt_sft has been removed. "
                "Remove this key (Channel-B is unified rollout-prefix + FN-append)."
            )
        if "desc_ce_weight_matched" in data:
            raise ValueError(
                "stage2_ab.channel_b.desc_ce_weight_matched has been removed. "
                "Remove this key (matched-object desc CE is always disabled in Channel-B)."
            )
        if "semantic_desc_gate" in data:
            raise ValueError(
                "stage2_ab.channel_b.semantic_desc_gate has been removed. "
                "Remove this key (training-time semantic gating is unsupported)."
            )
        for key in sorted(data.keys(), key=lambda x: str(x)):
            if isinstance(key, str) and _is_versioned_alias_for(key, "pseudo_positive"):
                raise ValueError(
                    "Versioned pseudo-positive knob aliases are unsupported; "
                    "use stage2_ab.channel_b.pseudo_positive instead."
                )

        if "duplicate_iou_threshold" in data:
            raise ValueError(
                "stage2_ab.channel_b.duplicate_iou_threshold has been removed. "
                "Use stage2_ab.channel_b.duplicate_control.iou_threshold instead."
            )
        if "center_radius_scale" in data:
            raise ValueError(
                "stage2_ab.channel_b.center_radius_scale has been removed. "
                "Use stage2_ab.channel_b.duplicate_control.center_radius_scale instead."
            )

        assignment = Stage2ABChannelBAssignmentConfig.from_mapping(
            data.pop("assignment", None)
        )
        duplicate_control = Stage2ABChannelBDuplicateControlConfig.from_mapping(
            data.pop("duplicate_control", None)
        )

        rollout_template_family_raw = data.pop(
            "rollout_template_family", cls.rollout_template_family
        )
        rollout_decode_policy_raw = data.pop("rollout_decode_policy", None)
        invalid_rollout_policy_raw = data.pop("invalid_rollout_policy", None)
        fallback_loss_weight_raw = data.pop(
            "fallback_loss_weight", cls.fallback_loss_weight
        )
        rollout_template_policy = resolve_stage2_rollout_template_policy(
            rollout_template_family_raw,
            rollout_decode_policy=rollout_decode_policy_raw,
            invalid_rollout_policy=invalid_rollout_policy_raw,
            fallback_loss_weight=fallback_loss_weight_raw,
        )
        rollout_template_family = rollout_template_policy.template_family
        rollout_decode_policy = rollout_template_policy.decode_policy
        invalid_rollout_policy = rollout_template_policy.invalid_rollout_policy
        fallback_loss_weight = float(rollout_template_policy.fallback_loss_weight)

        insertion_order_raw = data.pop(
            "insertion_order",
            cls.insertion_order,
        )
        insertion_order = str(insertion_order_raw).strip().lower()
        if insertion_order not in {"tail_append", "sorted", "fn_slot_shuffle"}:
            raise ValueError(
                "stage2_ab.channel_b.insertion_order must be one of "
                "{'tail_append', 'sorted', 'fn_slot_shuffle'}"
            )

        fp_policy = Stage2ABChannelBFalsePositivePolicyConfig.from_mapping(
            data.pop("fp_policy", None)
        )

        producer_wait_timeout_s_raw = data.pop("producer_wait_timeout_s", None)
        producer_wait_timeout_s: Optional[float] = None
        if producer_wait_timeout_s_raw is not None:
            try:
                producer_wait_timeout_s = float(producer_wait_timeout_s_raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "stage2_ab.channel_b.producer_wait_timeout_s must be a float/int when set"
                ) from exc
            if producer_wait_timeout_s < 0.0:
                raise ValueError(
                    "stage2_ab.channel_b.producer_wait_timeout_s must be >= 0 when set "
                    "(use 0 for automatic timeout selection)"
                )

        ddp_phase_timeout_s_raw = data.pop("ddp_phase_timeout_s", None)
        ddp_phase_timeout_s: Optional[float] = None
        if ddp_phase_timeout_s_raw is not None:
            try:
                ddp_phase_timeout_s = float(ddp_phase_timeout_s_raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "stage2_ab.channel_b.ddp_phase_timeout_s must be a float/int when set"
                ) from exc
            if ddp_phase_timeout_s <= 0.0:
                raise ValueError(
                    "stage2_ab.channel_b.ddp_phase_timeout_s must be > 0 when set "
                    "(bounded DDP phase barriers are required)"
                )

        pseudo_positive = Stage2ABChannelBPseudoPositiveConfig.from_mapping(
            data.pop("pseudo_positive", None)
        )
        triage_default_rollouts = (
            4
            if pseudo_positive.enabled or fp_policy.mode == "weak_positive_context"
            else Stage2ABChannelBTriagePosteriorConfig.num_rollouts
        )
        triage_posterior = Stage2ABChannelBTriagePosteriorConfig.from_mapping(
            data.pop("triage_posterior", None),
            default_num_rollouts=triage_default_rollouts,
        )
        if (
            validate_legacy_rollouts
            and not pseudo_positive.enabled
            and fp_policy.mode != "weak_positive_context"
            and triage_posterior.num_rollouts
            != Stage2ABChannelBTriagePosteriorConfig.num_rollouts
        ):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.num_rollouts must be 2 when "
                "stage2_ab.channel_b.pseudo_positive.enabled=false"
            )

        if data:
            raise ValueError(
                f"Unknown stage2_ab.channel_b keys: {sorted(str(k) for k in data.keys())}"
            )

        return cls(
            assignment=assignment,
            duplicate_control=duplicate_control,
            producer_wait_timeout_s=producer_wait_timeout_s,
            ddp_phase_timeout_s=ddp_phase_timeout_s,
            rollout_template_family=rollout_template_family,
            rollout_decode_policy=rollout_decode_policy,
            fallback_loss_weight=fallback_loss_weight,
            invalid_rollout_policy=invalid_rollout_policy,
            insertion_order=insertion_order,
            fp_policy=fp_policy,
            pseudo_positive=pseudo_positive,
            triage_posterior=triage_posterior,
        )


@dataclass(frozen=True)
class Stage2PipelineModuleSpec:
    name: str
    enabled: bool = True
    weight: float = 1.0
    channels: tuple[str, ...] = ("A", "B")
    application: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        path: str,
        allowed_names: set[str],
    ) -> "Stage2PipelineModuleSpec":
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path} must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)

        name_raw = data.pop("name", None)
        name = str(name_raw or "").strip()
        if not name:
            raise ValueError(f"{path}.name must be a non-empty string")
        if name not in allowed_names:
            raise ValueError(
                f"{path}.name must be one of {sorted(allowed_names)}; got {name!r}"
            )

        if "enabled" not in data:
            raise ValueError(
                f"{path}.enabled must be provided (explicit pipeline spec; no defaults)"
            )
        enabled_raw = data.pop("enabled")
        enabled = bool(enabled_raw)

        if "weight" not in data:
            raise ValueError(
                f"{path}.weight must be provided (explicit pipeline spec; no defaults)"
            )
        weight_raw = data.pop("weight")
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{path}.weight must be numeric") from exc
        if weight < 0.0:
            raise ValueError(f"{path}.weight must be >= 0")

        if "channels" not in data:
            raise ValueError(
                f"{path}.channels must be provided (explicit pipeline spec; no defaults)"
            )
        channels_raw = data.pop("channels")
        if not isinstance(channels_raw, Sequence) or isinstance(
            channels_raw, (str, bytes)
        ):
            raise TypeError(f"{path}.channels must be a sequence of 'A'/'B'")
        channels_list: list[str] = []
        for idx, ch in enumerate(channels_raw):
            ch_s = str(ch).strip().upper()
            if ch_s not in {"A", "B"}:
                raise ValueError(f"{path}.channels[{idx}] must be 'A' or 'B'")
            channels_list.append(ch_s)
        if not channels_list:
            raise ValueError(f"{path}.channels must not be empty")
        channels = tuple(dict.fromkeys(channels_list).keys())

        application_raw = data.pop("application", {})
        if not isinstance(application_raw, Mapping):
            raise TypeError(f"{path}.application must be a mapping")
        application = dict(application_raw)

        if "config" not in data:
            raise ValueError(
                f"{path}.config must be provided (explicit pipeline spec; no defaults)"
            )
        cfg_raw = data.pop("config")
        if cfg_raw is None:
            cfg_raw = {}
        if not isinstance(cfg_raw, Mapping):
            raise TypeError(f"{path}.config must be a mapping")
        config = dict(cfg_raw)

        if data:
            unknown = [
                f"{path}.{str(k)}" for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown pipeline module keys: {unknown}")

        return cls(
            name=name,
            enabled=enabled,
            weight=weight,
            channels=channels,
            application=application,
            config=config,
        )


@dataclass(frozen=True)
class Stage2PipelineConfig:
    objective: tuple[Stage2PipelineModuleSpec, ...] = field(default_factory=tuple)
    diagnostics: tuple[Stage2PipelineModuleSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        allow_empty_objective: bool = False,
    ) -> "Stage2PipelineConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab.pipeline must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)

        objective_raw = data.pop("objective", [])
        diagnostics_raw = data.pop("diagnostics", [])
        if diagnostics_raw is None:
            diagnostics_raw = []

        if not isinstance(objective_raw, Sequence) or isinstance(
            objective_raw, (str, bytes)
        ):
            raise TypeError("stage2_ab.pipeline.objective must be a list")
        if not isinstance(diagnostics_raw, Sequence) or isinstance(
            diagnostics_raw, (str, bytes)
        ):
            raise TypeError("stage2_ab.pipeline.diagnostics must be a list")

        objective_specs = [
            Stage2PipelineModuleSpec.from_mapping(
                item,
                path=f"stage2_ab.pipeline.objective[{idx}]",
                allowed_names=ALLOWED_OBJECTIVE_MODULES
                | {STAGE2_TRIE_CE_MODULE_NAME, STAGE2_RESIDUAL_SET_MODULE_NAME},
            )
            for idx, item in enumerate(objective_raw)
        ]
        diagnostics_specs = [
            Stage2PipelineModuleSpec.from_mapping(
                item,
                path=f"stage2_ab.pipeline.diagnostics[{idx}]",
                allowed_names=ALLOWED_DIAGNOSTIC_MODULES,
            )
            for idx, item in enumerate(diagnostics_raw)
        ]
        if not objective_specs and allow_empty_objective:
            if data:
                unknown = [
                    f"stage2_ab.pipeline.{str(k)}"
                    for k in sorted(data.keys(), key=lambda x: str(x))
                ]
                raise ValueError(f"Unknown stage2_ab.pipeline keys: {unknown}")
            return cls(
                objective=(),
                diagnostics=tuple(diagnostics_specs),
            )
        if not objective_specs:
            raise ValueError("stage2_ab.pipeline.objective must be non-empty")

        def _assert_no_duplicates(
            items: list[Stage2PipelineModuleSpec], *, path: str
        ) -> None:
            seen: set[str] = set()
            for spec in items:
                if spec.name in seen:
                    raise ValueError(f"Duplicate module name in {path}: {spec.name}")
                seen.add(spec.name)

        _assert_no_duplicates(objective_specs, path="stage2_ab.pipeline.objective")
        _assert_no_duplicates(diagnostics_specs, path="stage2_ab.pipeline.diagnostics")

        canonical_objective_order = ["token_ce"]
        trie_ce_objective_order = ["token_ce", STAGE2_TRIE_CE_MODULE_NAME]
        residual_set_objective_order = ["token_ce", STAGE2_RESIDUAL_SET_MODULE_NAME]
        trie_ce_residual_set_objective_order = [
            "token_ce",
            STAGE2_TRIE_CE_MODULE_NAME,
            STAGE2_RESIDUAL_SET_MODULE_NAME,
        ]
        hard_sft_objective_order = ["hard_sft"]
        authored_objective_order = [str(spec.name) for spec in objective_specs]
        if authored_objective_order not in (
            canonical_objective_order,
            trie_ce_objective_order,
            residual_set_objective_order,
            trie_ce_residual_set_objective_order,
            hard_sft_objective_order,
        ):
            raise ValueError(
                "stage2_ab.pipeline.objective must use the canonical module order "
                f"{canonical_objective_order}, {trie_ce_objective_order}, "
                f"{residual_set_objective_order}, "
                f"{trie_ce_residual_set_objective_order}, "
                f"or {hard_sft_objective_order}; "
                f"got {authored_objective_order}"
            )

        for idx, spec in enumerate(objective_specs):
            if not isinstance(spec.application, Mapping):
                raise TypeError(
                    f"stage2_ab.pipeline.objective[{idx}].application must be a mapping"
                )
            app_unknown = set(spec.application.keys()) - {"preset"}
            if app_unknown:
                raise ValueError(
                    "Unknown stage2_ab.pipeline.objective"
                    f"[{idx}].application keys for module {spec.name!r}: "
                    f"{sorted(str(k) for k in app_unknown)}"
                )
            preset = str(spec.application.get("preset", "") or "").strip()
            if not preset:
                raise ValueError(
                    f"stage2_ab.pipeline.objective[{idx}].application.preset must be provided"
                )
            allowed_presets = OBJECTIVE_APPLICATION_PRESET_ALLOWLIST.get(
                str(spec.name), set()
            )
            if str(spec.name) == STAGE2_TRIE_CE_MODULE_NAME:
                allowed_presets = STAGE2_TRIE_CE_APPLICATION_PRESETS
            if str(spec.name) == STAGE2_RESIDUAL_SET_MODULE_NAME:
                allowed_presets = STAGE2_RESIDUAL_SET_APPLICATION_PRESETS
            if preset not in allowed_presets:
                if preset in {
                    "anchor_text_plus_final_struct",
                    "anchor_if_single_iter_else_final",
                    "final_only",
                    "anchor_and_final",
                }:
                    replacement = (
                        "anchor_text_only"
                        if str(spec.name) == "token_ce"
                        else "anchor_only"
                    )
                    raise ValueError(
                        "stage2_ab.pipeline.objective"
                        f"[{idx}].application.preset for module {spec.name!r} uses deprecated "
                        f"self-context-era routing {preset!r}. Use {replacement!r} for the "
                        "single-pass Channel-A contract."
                    )
                raise ValueError(
                    "stage2_ab.pipeline.objective"
                    f"[{idx}].application.preset for module {spec.name!r} must be one of "
                    f"{sorted(str(x) for x in allowed_presets)}; got {preset!r}"
                )
            if (
                str(spec.name) == "token_ce"
                and "rollout_drop_invalid_struct_ce_multiplier" in spec.config
            ):
                raise ValueError(
                    "stage2_ab.pipeline.objective[%d].config.rollout_drop_invalid_struct_ce_multiplier "
                    "has been removed. Legacy raw-prefix invalid-structure amplification "
                    "is not part of the canonical clean-prefix Channel-B contract."
                    % int(idx)
                )
            if str(spec.name) == "token_ce" and "struct_ce_weight" in spec.config:
                raise ValueError(
                    "stage2_ab.pipeline.objective"
                    f"[{idx}].config.struct_ce_weight is deprecated and unsupported. "
                    "Remove the self-context struct/EOS stabilizer; active Stage-2 Channel-A "
                    "training uses only the single-pass anchor_text_only contract."
                )
            allowed_cfg = OBJECTIVE_CONFIG_ALLOWLIST.get(str(spec.name), set())
            if str(spec.name) == STAGE2_TRIE_CE_MODULE_NAME:
                allowed_cfg = STAGE2_TRIE_CE_CONFIG_KEYS
            if str(spec.name) == STAGE2_RESIDUAL_SET_MODULE_NAME:
                allowed_cfg = STAGE2_RESIDUAL_SET_CONFIG_KEYS
            unknown_cfg = set(spec.config.keys()) - allowed_cfg
            if unknown_cfg:
                raise ValueError(
                    "Unknown stage2_ab.pipeline.objective"
                    f"[{idx}].config keys for module {spec.name!r}: "
                    f"{sorted(str(k) for k in unknown_cfg)}"
                )
            if str(spec.name) == STAGE2_RESIDUAL_SET_MODULE_NAME:
                if "ul_geometry" in spec.config:
                    ul_geometry = spec.config["ul_geometry"]
                    if not isinstance(ul_geometry, Mapping):
                        raise TypeError(
                            "stage2_ab.pipeline.objective"
                            f"[{idx}].config.ul_geometry must be a mapping"
                        )
                    ul_geometry_unknown = (
                        set(ul_geometry.keys())
                        - STAGE2_RESIDUAL_SET_UL_GEOMETRY_KEYS
                    )
                    if ul_geometry_unknown:
                        raise ValueError(
                            "Unknown stage2_ab.pipeline.objective"
                            f"[{idx}].config.ul_geometry keys for module {spec.name!r}: "
                            f"{sorted(str(k) for k in ul_geometry_unknown)}"
                        )
                if "artifact_policy" in spec.config:
                    artifact_policy = spec.config["artifact_policy"]
                    if not isinstance(artifact_policy, Mapping):
                        raise TypeError(
                            "stage2_ab.pipeline.objective"
                            f"[{idx}].config.artifact_policy must be a mapping"
                        )
                    artifact_policy_unknown = (
                        set(artifact_policy.keys())
                        - STAGE2_RESIDUAL_SET_ARTIFACT_POLICY_KEYS
                    )
                    if artifact_policy_unknown:
                        raise ValueError(
                            "Unknown stage2_ab.pipeline.objective"
                            f"[{idx}].config.artifact_policy keys for module {spec.name!r}: "
                            f"{sorted(str(k) for k in artifact_policy_unknown)}"
                        )
            optional_cfg = OBJECTIVE_OPTIONAL_CONFIG_KEYS.get(str(spec.name), set())
            missing_cfg = allowed_cfg - set(spec.config.keys()) - set(optional_cfg)
            if missing_cfg:
                raise ValueError(
                    "Missing required stage2_ab.pipeline.objective"
                    f"[{idx}].config keys for module {spec.name!r}: "
                    f"{sorted(str(k) for k in missing_cfg)}"
                )
            if str(spec.name) == STAGE2_TRIE_CE_MODULE_NAME:
                for weight_key in sorted(STAGE2_TRIE_CE_RESERVED_WEIGHT_KEYS):
                    weight_raw = spec.config.get(weight_key)
                    if isinstance(weight_raw, bool):
                        raise TypeError(
                            "stage2_ab.pipeline.objective"
                            f"[{idx}].config.{weight_key} must be numeric, not bool"
                        )
                    try:
                        weight_value = float(weight_raw)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            "stage2_ab.pipeline.objective"
                            f"[{idx}].config.{weight_key} must be numeric"
                        ) from exc
                    if not math.isfinite(weight_value):
                        raise ValueError(
                            "stage2_ab.pipeline.objective"
                            f"[{idx}].config.{weight_key} must be finite"
                        )
                    if weight_value < 0.0:
                        raise ValueError(
                            "stage2_ab.pipeline.objective"
                            f"[{idx}].config.{weight_key} must be >= 0"
                        )
                    if weight_value != 1.0:
                        raise ValueError(
                            "stage2_ab.pipeline.objective"
                            f"[{idx}].config.{weight_key} must be 1.0 because "
                            "Stage-2 trie CE pure hard CE v0 does not apply "
                            "reserved future weight knobs."
                        )
                normalization = str(
                    spec.config.get("normalization", "") or ""
                ).strip().lower()
                if normalization not in STAGE2_TRIE_CE_NORMALIZATIONS:
                    raise ValueError(
                        "stage2_ab.pipeline.objective"
                        f"[{idx}].config.normalization: Stage-2 trie CE pure "
                        "hard CE v0 requires token_mean; semantic bucket "
                        "balancing is a reserved future knob."
                    )
        for idx, spec in enumerate(diagnostics_specs):
            allowed_cfg = DIAGNOSTIC_CONFIG_ALLOWLIST.get(str(spec.name), set())
            unknown_cfg = set(spec.config.keys()) - allowed_cfg
            if unknown_cfg:
                raise ValueError(
                    "Unknown stage2_ab.pipeline.diagnostics"
                    f"[{idx}].config keys for module {spec.name!r}: "
                    f"{sorted(str(k) for k in unknown_cfg)}"
                )

        specs_by_name = {spec.name: spec for spec in objective_specs}
        token_ce = specs_by_name.get("token_ce")
        stage2_trie_ce = specs_by_name.get(STAGE2_TRIE_CE_MODULE_NAME)
        residual_set = specs_by_name.get(STAGE2_RESIDUAL_SET_MODULE_NAME)

        if (
            residual_set is not None
            and bool(residual_set.enabled)
            and stage2_trie_ce is not None
            and bool(stage2_trie_ce.enabled)
        ):
            raise ValueError(
                "residual_set_correction cannot be enabled together with "
                "stage2_trie_ce on Channel-B; remove the legacy trie CE path."
            )

        if (
            token_ce is not None
            and stage2_trie_ce is not None
            and bool(token_ce.enabled)
            and bool(stage2_trie_ce.enabled)
            and "B" in token_ce.channels
            and "B" in stage2_trie_ce.channels
        ):
            raise ValueError(
                "Channel-B objective supervision cannot enable both token_ce and "
                "stage2_trie_ce on Channel-B; remove Channel-B from token_ce.channels "
                "or disable stage2_trie_ce."
            )

        if data:
            unknown = [
                f"stage2_ab.pipeline.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown stage2_ab.pipeline keys: {unknown}")

        return cls(
            objective=tuple(objective_specs),
            diagnostics=tuple(diagnostics_specs),
        )


@dataclass(frozen=True)
class Stage2ABConfig:
    schedule: Stage2ABScheduleConfig
    pipeline: Stage2PipelineConfig
    channel_b: Stage2ABChannelBConfig = field(default_factory=Stage2ABChannelBConfig)

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        allow_teacher_forcing_pipeline: bool = False,
    ) -> "Stage2ABConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab section must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)

        disallowed_flat = [
            k
            for k in (
                "desc_ce_weight",
                "fmt_struct_ce_weight",
                "bbox_smoothl1_weight",
                "bbox_ciou_weight",
                "text_gate_weight",
                "coord_ce_weight",
                "coord_gate_weight",
            )
            if k in data
        ]
        if disallowed_flat:
            raise ValueError(
                "Flat stage2_ab objective knobs have been removed. "
                "Express objective weights via stage2_ab.pipeline.objective[*].config instead: "
                f"{sorted(disallowed_flat)}"
            )

        schedule_raw = data.pop("schedule", None)
        if schedule_raw is None:
            raise ValueError("stage2_ab.schedule must be provided")
        schedule = Stage2ABScheduleConfig.from_mapping(schedule_raw)

        pipeline_raw = data.pop("pipeline", None)
        if pipeline_raw is None:
            raise ValueError(
                "stage2_ab.pipeline must be provided (no implicit default objective manifest)."
            )
        pipeline = Stage2PipelineConfig.from_mapping(
            pipeline_raw,
            allow_empty_objective=allow_teacher_forcing_pipeline,
        )

        if "bbox_l1_weight" in data or "bbox_giou_weight" in data:
            raise ValueError(
                "stage2_ab.bbox_l1_weight/bbox_giou_weight are deprecated. "
                "bbox geometry auxiliaries have been removed from the active "
                "Stage-2 pipeline."
            )

        deprecated_keys = [
            f"stage2_ab.{key}"
            for key in (
                "n_softctx_iter",
                "softctx_grad_mode",
                "softctx_temperature",
                "coord_ctx_embed_mode",
                "coord_decode_mode",
            )
            if key in data
        ]
        if deprecated_keys:
            raise ValueError(
                "Deprecated Stage-2 self-context knobs are unsupported in active/training "
                "configs. Remove them and use the single-pass Channel-A contract "
                "(token_ce: anchor_text_only; optional stage2_trie_ce: rollout_text_only). "
                f"Found: {sorted(deprecated_keys)}"
            )

        channel_b = Stage2ABChannelBConfig.from_mapping(
            data.pop("channel_b", None),
            validate_legacy_rollouts=False,
        )

        residual_set = next(
            (
                spec
                for spec in pipeline.objective
                if spec.name == STAGE2_RESIDUAL_SET_MODULE_NAME and bool(spec.enabled)
            ),
            None,
        )
        if residual_set is not None:
            residual_set_on_channel_b = "B" in residual_set.channels
            if residual_set_on_channel_b:
                for conflicting_module_name in ("token_ce", "hard_sft"):
                    conflicting_module = next(
                        (
                            spec
                            for spec in pipeline.objective
                            if spec.name == conflicting_module_name
                            and bool(spec.enabled)
                            and "B" in spec.channels
                        ),
                        None,
                    )
                    if conflicting_module is not None:
                        raise ValueError(
                            "residual_set_correction is mutually exclusive with "
                            f"{conflicting_module_name} on Channel-B; remove "
                            f"Channel-B from {conflicting_module_name}.channels "
                            "or disable one objective."
                        )
            if channel_b.pseudo_positive.enabled:
                raise ValueError(
                    "residual_set_correction is mutually exclusive with "
                    "stage2_ab.channel_b.pseudo_positive; disable pseudo_positive "
                    "to avoid double supervision."
                )
            if STAGE2_TRIE_CE_MODULE_NAME in {
                spec.name for spec in pipeline.objective if bool(spec.enabled)
            }:
                raise ValueError(
                    "residual_set_correction is mutually exclusive with "
                    "stage2_trie_ce; remove the legacy Channel-B trie CE objective."
                )
            num_rollouts_raw = residual_set.config.get("num_rollouts")
            if isinstance(num_rollouts_raw, bool) or not isinstance(
                num_rollouts_raw, int
            ):
                raise TypeError(
                    "stage2_ab.pipeline.objective residual_set_correction "
                    "config.num_rollouts must be an int"
                )
            if num_rollouts_raw < 2:
                raise ValueError(
                    "stage2_ab.pipeline.objective residual_set_correction "
                    "config.num_rollouts must be >= 2"
                )
        elif (
            not channel_b.pseudo_positive.enabled
            and channel_b.fp_policy.mode != "weak_positive_context"
            and channel_b.triage_posterior.num_rollouts
            != Stage2ABChannelBTriagePosteriorConfig.num_rollouts
        ):
            raise ValueError(
                "stage2_ab.channel_b.triage_posterior.num_rollouts must be 2 when "
                "stage2_ab.channel_b.pseudo_positive.enabled=false"
            )

        if data:
            unknown = [
                f"stage2_ab.{str(k)}" for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(
                "Unknown stage2_ab keys: "
                f"{unknown}. "
                "Migration guidance: remove unsupported keys or move them into "
                "the current stage2_ab schema (for Channel-B options use stage2_ab.channel_b.*)."
            )

        return cls(
            schedule=schedule,
            pipeline=pipeline,
            channel_b=channel_b,
        )


def _compile_teacher_forcing_stage2_ab(
    stage2_ab: Stage2ABConfig,
    objective: TeacherForcingObjectiveConfig,
) -> Stage2ABConfig:
    if stage2_ab.pipeline.objective:
        return stage2_ab
    if objective.profile != "hard_sft":
        raise ValueError(
            "teacher_forcing Stage-2 valid-set profiles require target IR runtime "
            "wiring before they can compile to a runtime objective manifest"
        )

    objective_spec = Stage2PipelineModuleSpec(
        name="hard_sft",
        enabled=True,
        weight=1.0,
        channels=("A", "B"),
        application={"preset": "hard_sft"},
        config={
            "desc_ce_weight": 1.0,
            "rollout_fn_desc_weight": 1.0,
            "rollout_global_prefix_struct_ce_weight": 1.0,
        },
    )

    return Stage2ABConfig(
        schedule=stage2_ab.schedule,
        pipeline=Stage2PipelineConfig(
            objective=(objective_spec,),
            diagnostics=stage2_ab.pipeline.diagnostics,
        ),
        channel_b=stage2_ab.channel_b,
    )


def _validate_stage2_ab_rollout_surface_alignment(
    *,
    custom: CustomConfig,
    stage2_ab: Stage2ABConfig | None,
) -> None:
    if stage2_ab is None or custom.trainer_variant != "stage2_two_channel":
        return

    detection_sequence_format = normalize_detection_sequence_format(
        custom.detection_sequence_format
    )
    rollout_template_family = str(stage2_ab.channel_b.rollout_template_family)

    if detection_sequence_format == COORDJSON_FORMAT:
        expected_rollout_template_family = "coordjson"
    elif detection_sequence_format == COMPACT_FULL_FORMAT:
        expected_rollout_template_family = COMPACT_FULL_FORMAT
    else:
        raise ValueError(
            "custom.trainer_variant=stage2_two_channel supports "
            "custom.detection_sequence_format values {'coordjson', 'compact_full'}; "
            f"got {detection_sequence_format!r}."
        )

    if rollout_template_family != expected_rollout_template_family:
        raise ValueError(
            "custom.trainer_variant=stage2_two_channel requires aligned prompt and "
            "rollout parser surfaces: "
            f"custom.detection_sequence_format={detection_sequence_format} requires "
            "stage2_ab.channel_b.rollout_template_family="
            f"{expected_rollout_template_family}; got {rollout_template_family!r}."
        )


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
            "training.packing must be boolean when objective.id=teacher_forcing"
        )
    if packing_raw:
        raise ValueError(
            "objective.id=teacher_forcing currently rejects training.packing=true; "
            "exact atom-position packing mapping is not implemented"
        )


def _validate_teacher_forcing_stage2_migration_raw(stage2_ab_raw: Any) -> None:
    if stage2_ab_raw is None:
        return
    if not isinstance(stage2_ab_raw, Mapping):
        raise TypeError("stage2_ab section must be a mapping")
    pipeline_raw = stage2_ab_raw.get("pipeline")
    if not isinstance(pipeline_raw, Mapping):
        return
    objective_raw = pipeline_raw.get("objective", [])
    if not isinstance(objective_raw, Sequence) or isinstance(
        objective_raw, (str, bytes)
    ):
        return

    violations: list[str] = []
    for idx, item in enumerate(objective_raw):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "") or "").strip()
        if name in LEGACY_STAGE2_TEACHER_FORCING_MODULES:
            violations.append(f"stage2_ab.pipeline.objective[{idx}].name={name}")
        config = item.get("config")
        if isinstance(config, Mapping):
            for key in sorted(
                set(str(k) for k in config.keys())
                & LEGACY_STAGE2_TEACHER_FORCING_CONFIG_KEYS
            ):
                violations.append(
                    f"stage2_ab.pipeline.objective[{idx}].config.{key}"
                )

    if violations:
        raise ValueError(
            "objective.id=teacher_forcing rejects legacy "
            "stage2_ab.pipeline.objective modules/config keys: "
            f"{violations}. Move active loss selection to objective.modules."
        )


_DETECTION_REQUIRED_SECTIONS: set[str] = {
    "data",
    "prompt",
    "detection_template",
    "token_rows",
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
    objective: "DetectionObjectiveConfig | TeacherForcingObjectiveConfig",
    packing: "DetectionPackingConfig",
    training: Mapping[str, Any],
) -> None:
    training_packing = _detection_runtime_bool(training, "packing")
    training_eval_packing = _detection_runtime_bool(training, "eval_packing")

    if getattr(objective, "id", None) == TEACHER_FORCING_OBJECTIVE_ID:
        if training_packing:
            raise ValueError(
                "objective.id=teacher_forcing currently rejects training.packing=true; "
                "exact atom-position packing mapping is not implemented"
            )
        if training_eval_packing:
            raise ValueError(
                "objective.id=teacher_forcing currently rejects "
                "training.eval_packing=true; exact atom-position packing mapping "
                "is not implemented"
            )
        if packing.static_packing:
            raise ValueError(
                "objective.id=teacher_forcing currently rejects "
                "packing.static_packing=true; exact atom-position packing mapping "
                "is not implemented"
            )
        if packing.padding_free_packed:
            raise ValueError(
                "objective.id=teacher_forcing currently rejects "
                "packing.padding_free_packed=true; exact atom-position packing "
                "mapping is not implemented"
            )
        return

    if packing.static_packing and not training_packing:
        raise ValueError(
            "packing.static_packing=true requires training.packing=true for detection runtime materialization."
        )

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
        if data.object_ordering != required_order:
            raise ValueError(
                "data.object_ordering must be "
                f"{required_order!r} for objective.id='teacher_forcing', "
                f"got {data.object_ordering!r}"
            )
        return

    required_order = (
        "sorted" if objective.variant == "sorted_sft" else "random_permutation"
    )
    if data.object_ordering != required_order:
        raise ValueError(
            "data.object_ordering must be "
            f"{required_order!r} for objective.variant={objective.variant!r}, "
            f"got {data.object_ordering!r}"
        )


def _detection_validate_prefix_rollin_contract(
    *,
    detection_template: "DetectionTemplateConfig",
    objective: "DetectionObjectiveConfig | TeacherForcingObjectiveConfig",
    experiment: "DetectionExperimentConfig | None",
) -> None:
    if getattr(objective, "id", None) == TEACHER_FORCING_OBJECTIVE_ID:
        return
    if objective.variant != "prefix_rollin_et_rmp_ce":
        return

    if (
        objective.variant == "prefix_rollin_et_rmp_ce"
        and detection_template.id != "compact_full"
    ):
        raise ValueError(
            "objective.variant=prefix_rollin_et_rmp_ce requires "
            "detection_template.id=compact_full"
        )
    if experiment is None:
        raise ValueError(
            f"experiment.surface is required for objective.variant={objective.variant}"
        )


def _detection_validate_token_rows(
    detection_template: "DetectionTemplateConfig",
    token_rows: TrainableTokenRowsConfig,
) -> None:
    if not token_rows.enabled:
        raise ValueError(
            "token_rows.enabled must be true for detection coord-token training; "
            "otherwise coordinate special-token rows stay frozen and cannot be saved "
            "in the adapter"
        )
    if not token_rows.tie_head:
        raise ValueError(
            "token_rows.tie_head must be true for the current tied-head "
            "Qwen3-VL token-row adapter contract"
        )
    if detection_template.coordinate_surface == "coord_token":
        has_coord_geometry = any(
            group.role is TokenRole.COORD_GEOMETRY
            for group in token_rows.groups.values()
        )
        if not has_coord_geometry:
            raise ValueError(
                "token_rows must include at least one group with "
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
        if len(token_rows.groups) != 2:
            raise ValueError(
                "token_rows for coord-token detection must contain exactly the "
                "1002 allowed trainable rows: "
                f"{OBJECT_REF_START_TOKEN}, {BOX_START_TOKEN}, and "
                f"{COORD_START_TOKEN}..{COORD_END_TOKEN}; "
                "extra natural-language rows are not allowed"
            )
        if len(coord_groups) != 1:
            raise ValueError(
                "token_rows must contain exactly one coord_geometry group for "
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
                "token_rows coord_geometry must be exactly "
                f"{COORD_START_TOKEN}..{COORD_END_TOKEN} with expected ids "
                f"{EXPECTED_COORD_START_ID}..{EXPECTED_COORD_END_ID}"
            )
        if len(structural_groups) != 1:
            raise ValueError(
                "token_rows must include exactly the compact structural rows "
                f"{OBJECT_REF_START_TOKEN} and {BOX_START_TOKEN}"
            )
        structural_group = structural_groups[0]
        expected_structural_ids = {
            OBJECT_REF_START_TOKEN: EXPECTED_OBJECT_REF_START_ID,
            BOX_START_TOKEN: EXPECTED_BOX_START_ID,
        }
        if (
            structural_group.start_token is not None
            or structural_group.end_token is not None
            or structural_group.tokens != (OBJECT_REF_START_TOKEN, BOX_START_TOKEN)
            or dict(structural_group.expected_ids) != expected_structural_ids
        ):
            raise ValueError(
                "token_rows structural group must be exactly "
                f"{OBJECT_REF_START_TOKEN} and {BOX_START_TOKEN} with expected "
                f"ids {expected_structural_ids}"
            )


@dataclass(frozen=True)
class DetectionDataConfig:
    train_jsonl: str
    val_jsonl: str
    image_root: str | None = None
    object_ordering: Literal["sorted", "random_permutation"] = "sorted"

    def __post_init__(self) -> None:
        for field_name in ("train_jsonl", "val_jsonl"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"data.{field_name} must be a string")
        if self.image_root is not None and not isinstance(self.image_root, str):
            raise TypeError("data.image_root must be a string when provided")
        _detection_validate_choice(
            self.object_ordering,
            path="data.object_ordering",
            allowed={"sorted", "random_permutation"},
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionDataConfig":
        return parse_dataclass_strict(cls, payload, path="data")


@dataclass(frozen=True)
class DetectionPromptConfig:
    system_variant: str
    user_variant: str
    include_template_summary: bool = True
    prompt_variant_enabled: bool = False

    def __post_init__(self) -> None:
        for field_name in ("system_variant", "user_variant"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"prompt.{field_name} must be a string")
        _detection_validate_bool(
            self.include_template_summary,
            path="prompt.include_template_summary",
        )
        _detection_validate_bool(
            self.prompt_variant_enabled,
            path="prompt.prompt_variant_enabled",
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionPromptConfig":
        return parse_dataclass_strict(cls, payload, path="prompt")


@dataclass(frozen=True)
class DetectionTemplateConfig:
    id: Literal["stage1_json_pretty", "compact_full"]
    coordinate_surface: Literal["coord_token"]
    bbox_format: Literal["xyxy"]
    object_field_order: Optional[Literal["desc_first"]] = None
    strict_parse: bool = True

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.id,
            path="detection_template.id",
            allowed={"stage1_json_pretty", "compact_full"},
        )
        _detection_validate_choice(
            self.coordinate_surface,
            path="detection_template.coordinate_surface",
            allowed={"coord_token"},
        )
        _detection_validate_choice(
            self.bbox_format,
            path="detection_template.bbox_format",
            allowed={"xyxy"},
        )
        if self.object_field_order is not None:
            _detection_validate_choice(
                self.object_field_order,
                path="detection_template.object_field_order",
                allowed={"desc_first"},
            )
        _detection_validate_bool(
            self.strict_parse,
            path="detection_template.strict_parse",
        )
        if self.id == "stage1_json_pretty" and self.object_field_order != "desc_first":
            raise ValueError(
                "detection_template.id=stage1_json_pretty requires "
                "detection_template.object_field_order=desc_first"
            )
        if self.id == "compact_full" and self.object_field_order is not None:
            raise ValueError(
                "detection_template.object_field_order must be omitted for "
                "detection_template.id=compact_full"
            )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionTemplateConfig":
        return parse_dataclass_strict(cls, payload, path="detection_template")


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
        for field_name in ("ablation_id",):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"experiment.{field_name} must be a string when provided"
                )

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
            path="objective.modules.*.enabled",
        )


@dataclass(frozen=True)
class TeacherForcingWithinValidCoverageConfig:
    enabled: bool = False
    coverage_strength: float = 0.0

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.modules.within_valid_coverage.enabled",
        )
        if not isinstance(self.coverage_strength, (int, float)) or isinstance(
            self.coverage_strength, bool
        ):
            raise TypeError(
                "objective.modules.within_valid_coverage.coverage_strength must be numeric"
            )
        value = float(self.coverage_strength)
        if not math.isfinite(value):
            raise ValueError(
                "objective.modules.within_valid_coverage.coverage_strength must be finite"
            )
        if value < 0.0:
            raise ValueError(
                "objective.modules.within_valid_coverage.coverage_strength must be >= 0"
            )
        object.__setattr__(self, "coverage_strength", value)


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

    @classmethod
    def from_mapping(cls, payload: Any) -> "TeacherForcingModulesConfig":
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise TypeError("objective.modules must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        token_type_mass = parse_dataclass_strict(
            TeacherForcingEnabledModuleConfig,
            data.pop("token_type_mass", {}),
            path="objective.modules.token_type_mass",
        )
        conditional_valid_set_likelihood = parse_dataclass_strict(
            TeacherForcingEnabledModuleConfig,
            data.pop("conditional_valid_set_likelihood", {}),
            path="objective.modules.conditional_valid_set_likelihood",
        )
        within_valid_coverage = parse_dataclass_strict(
            TeacherForcingWithinValidCoverageConfig,
            data.pop("within_valid_coverage", {}),
            path="objective.modules.within_valid_coverage",
        )
        continuation_margin = parse_dataclass_strict(
            TeacherForcingEnabledModuleConfig,
            data.pop("continuation_margin", {}),
            path="objective.modules.continuation_margin",
        )
        if data:
            unknown = [
                f"objective.modules.{str(k)}"
                for k in sorted(data.keys(), key=lambda x: str(x))
            ]
            raise ValueError(f"Unknown objective.modules keys: {unknown}")
        return cls(
            token_type_mass=token_type_mass,
            conditional_valid_set_likelihood=conditional_valid_set_likelihood,
            within_valid_coverage=within_valid_coverage,
            continuation_margin=continuation_margin,
        )


@dataclass(frozen=True)
class TeacherForcingObjectiveConfig:
    id: Literal["teacher_forcing"]
    profile: Literal[
        "hard_sft",
        "pure_valid_set_marginal",
        "hybrid_valid_set_marginal",
    ] = "hard_sft"
    target_ir: TeacherForcingTargetIRConfig = field(
        default_factory=TeacherForcingTargetIRConfig
    )
    modules: TeacherForcingModulesConfig = field(default_factory=TeacherForcingModulesConfig)

    def __post_init__(self) -> None:
        if self.id != TEACHER_FORCING_OBJECTIVE_ID:
            raise ValueError(
                "objective.id must be exactly 'teacher_forcing'; "
                f"got {self.id!r}"
            )
        _detection_validate_choice(
            self.profile,
            path="objective.profile",
            allowed=TEACHER_FORCING_PROFILES,
        )
        coverage = self.modules.within_valid_coverage
        coverage_strength = float(coverage.coverage_strength)
        if self.profile == "hard_sft":
            hard_sft_module_checks = (
                (
                    "objective.modules.token_type_mass.enabled",
                    bool(self.modules.token_type_mass.enabled),
                ),
                (
                    "objective.modules.conditional_valid_set_likelihood.enabled",
                    bool(self.modules.conditional_valid_set_likelihood.enabled),
                ),
                (
                    "objective.modules.within_valid_coverage.enabled",
                    bool(coverage.enabled),
                ),
                (
                    "objective.modules.within_valid_coverage.coverage_strength",
                    coverage_strength > 0.0,
                ),
                (
                    "objective.modules.continuation_margin.enabled",
                    bool(self.modules.continuation_margin.enabled),
                ),
            )
            for module_key, is_enabled in hard_sft_module_checks:
                if is_enabled:
                    raise ValueError(
                        "objective.profile=hard_sft does not support "
                        f"{module_key}; target-IR teacher-forcing modules "
                        "require a valid-set runtime path"
                    )
        if self.profile == "hybrid_valid_set_marginal":
            if not bool(coverage.enabled) or coverage_strength <= 0.0:
                raise ValueError(
                    "objective.profile=hybrid_valid_set_marginal "
                    "requires objective.modules.within_valid_coverage.enabled=true "
                    "and coverage_strength > 0"
                )
        elif coverage_strength > 0.0:
            raise ValueError(
                f"objective.profile={self.profile} requires "
                "objective.modules.within_valid_coverage.coverage_strength=0"
            )

    @classmethod
    def from_mapping(cls, payload: Any) -> "TeacherForcingObjectiveConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("objective must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)
        raw_id = data.get("id")
        if raw_id != TEACHER_FORCING_OBJECTIVE_ID:
            raise ValueError(
                "objective.id must be exactly 'teacher_forcing'; "
                f"legacy objective ids are unsupported, got {raw_id!r}"
            )
        if "target_ir" in data:
            data["target_ir"] = TeacherForcingTargetIRConfig.from_mapping(
                data["target_ir"]
            )
        if "modules" in data:
            data["modules"] = TeacherForcingModulesConfig.from_mapping(data["modules"])
        return parse_dataclass_strict(cls, data, path="objective")


@dataclass(frozen=True)
class DetectionObjectiveConfig:
    id: Literal["sft", "recursive_detection_ce"]
    variant: Literal[
        "sorted_sft",
        "random_order_sft",
        "random_permutation_et_rmp_ce",
        "trie_disabled_full_suffix_ce",
        "prefix_rollin_et_rmp_ce",
    ]
    trie_support_weight: float = 0.0
    trie_balance_weight: float = 0.0
    state_weighting: str = "none"
    normalization: str = "token_mean"
    rollin: Optional[PrefixRollinConfig] = None
    target: Optional[EntryTrieSupportBalanceConfig] = None
    type_gate: Optional[CompactTypeGateConfig] = None
    coord_soft_ce: Optional[CoordSoftCEConfig] = None

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.id,
            path="objective.id",
            allowed={"sft", "recursive_detection_ce"},
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
        if self.id == "sft" and self.variant not in {"sorted_sft", "random_order_sft"}:
            raise ValueError("objective.id=sft requires an SFT objective.variant")
        if self.id == "recursive_detection_ce" and self.variant in {
            "sorted_sft",
            "random_order_sft",
        }:
            raise ValueError(
                "objective.id=recursive_detection_ce requires a recursive detection "
                "objective.variant"
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
        if raw_id in LEGACY_TEACHER_FORCING_OBJECTIVE_IDS or raw_id in {"sft"}:
            raise ValueError(
                "objective.id must be exactly 'teacher_forcing'; "
                f"legacy objective ids are unsupported, got {raw_id!r}"
            )
        raise ValueError(
            "objective.id must be exactly 'teacher_forcing'; "
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
    expected_template: Literal["stage1_json_pretty", "compact_full"]
    parser_mode: Literal["strict_expected", "diagnostic_salvage"] = "strict_expected"

    def __post_init__(self) -> None:
        _detection_validate_choice(
            self.expected_template,
            path="evaluation.expected_template",
            allowed={"stage1_json_pretty", "compact_full"},
        )
        _detection_validate_choice(
            self.parser_mode,
            path="evaluation.parser_mode",
            allowed={"strict_expected", "diagnostic_salvage"},
        )

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionEvaluationConfig":
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


@dataclass(frozen=True)
class DetectionTrainingConfig:
    data: DetectionDataConfig
    prompt: DetectionPromptConfig
    detection_template: DetectionTemplateConfig
    token_rows: TrainableTokenRowsConfig
    objective: DetectionObjectiveConfig | TeacherForcingObjectiveConfig
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

    @classmethod
    def from_mapping(cls, payload: Any) -> "DetectionTrainingConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("detection config payload must be a mapping")
        if "custom" in payload:
            custom_raw = payload.get("custom")
            if isinstance(custom_raw, Mapping) and (
                "stage1_set_continuation" in custom_raw
                or custom_raw.get("trainer_variant") == "stage1_set_continuation"
            ):
                raise ValueError(
                    "custom is obsolete for detection configs; "
                    "custom.trainer_variant=stage1_set_continuation and "
                    "custom.stage1_set_continuation have been removed"
                )
            raise ValueError("custom is obsolete for detection configs")

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

        missing_sections = sorted(
            section
            for section in _DETECTION_REQUIRED_SECTIONS
            if section not in payload
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

        detection_template = DetectionTemplateConfig.from_mapping(
            payload["detection_template"]
        )
        evaluation = DetectionEvaluationConfig.from_mapping(payload["evaluation"])
        if evaluation.expected_template != detection_template.id:
            raise ValueError(
                "evaluation.expected_template must match detection_template.id "
                f"({evaluation.expected_template!r} != {detection_template.id!r})"
            )
        data_config = DetectionDataConfig.from_mapping(payload["data"])
        objective = DetectionObjectiveConfig.from_mapping(payload["objective"])
        _detection_validate_order_matches_objective(data_config, objective)
        if (
            getattr(objective, "id", None) != TEACHER_FORCING_OBJECTIVE_ID
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
        token_rows = TrainableTokenRowsConfig.from_mapping(
            payload["token_rows"],
            path="token_rows",
        )
        _detection_validate_token_rows(detection_template, token_rows)
        packing = DetectionPackingConfig.from_mapping(payload["packing"])
        training = _detection_validate_training_mapping(payload.get("training"))
        _detection_validate_packing_runtime_contract(
            objective=objective,
            packing=packing,
            training=training,
        )

        return cls(
            data=data_config,
            prompt=DetectionPromptConfig.from_mapping(payload["prompt"]),
            detection_template=detection_template,
            token_rows=token_rows,
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
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = dataclass_asdict_no_none(self)
        for section in _DETECTION_RUNTIME_SECTIONS:
            if payload.get(section) == {}:
                payload.pop(section, None)
        token_groups = payload.get("token_rows", {}).get("groups", {})
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
    stage2_ab: Optional[Stage2ABConfig] = None
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
        if trainer_variant == "stage2_ab_training":
            raise ValueError(
                "custom.trainer_variant=stage2_ab_training has been removed; use stage2_two_channel"
            )
        if trainer_variant == "rollout_matching_sft":
            raise ValueError(
                "custom.trainer_variant=rollout_matching_sft has been removed; use stage2_two_channel"
            )
        if trainer_variant in {"stage2_rollout_aligned", "stage2_rollout_runtime"}:
            raise ValueError(
                f"custom.trainer_variant={trainer_variant} has been removed; use stage2_two_channel"
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

        if objective is not None and stage2_ab_raw is not None:
            _validate_teacher_forcing_stage2_migration_raw(stage2_ab_raw)

        stage2_ab = None
        if stage2_ab_raw is not None:
            stage2_ab = Stage2ABConfig.from_mapping(
                stage2_ab_raw,
                allow_teacher_forcing_pipeline=objective is not None,
            )
            if objective is not None:
                stage2_ab = _compile_teacher_forcing_stage2_ab(stage2_ab, objective)
        elif trainer_variant == "stage2_two_channel":
            raise ValueError(
                "stage2_ab section must be provided when custom.trainer_variant=stage2_two_channel"
            )
        _validate_stage2_ab_rollout_surface_alignment(
            custom=custom,
            stage2_ab=stage2_ab,
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
                    "Use stage2_ab.pipeline with custom.trainer_variant=stage2_two_channel instead."
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

        if trainer_variant == "stage2_two_channel":
            if rollout_matching is None:
                raise ValueError(
                    "rollout_matching section must be provided for stage2_two_channel"
                )

        stage2_pipeline_present = bool(
            stage2_ab is not None and getattr(stage2_ab, "pipeline", None) is not None
        )

        if stage2_pipeline_present and custom_coord_soft_ce_w1_present:
            raise ValueError(
                "stage2_ab.pipeline is provided; custom.coord_soft_ce_w1.* is disallowed. "
                "Coordinate regularizers have been removed from the active Stage-2 pipeline."
            )
        if stage2_pipeline_present and custom_bbox_geo_present:
            raise ValueError(
                "stage2_ab.pipeline is provided; custom.bbox_geo.* is disallowed. "
                "bbox geometry auxiliaries have been removed from the active Stage-2 pipeline."
            )
        if stage2_pipeline_present and custom_bbox_size_aux_present:
            raise ValueError(
                "stage2_ab.pipeline is provided; custom.bbox_size_aux.* is disallowed. "
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
                if bool(getattr(vllm_cfg, "enable_lora", False)):
                    raise ValueError(
                        "vLLM rollouts require full merged-weight sync in this stack: "
                        "set rollout_matching.vllm.enable_lora=false."
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
            if trainer_variant == "stage2_two_channel":
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} is Stage-1-only in V1 and is unsupported for stage2 trainer variants."
                )
            if stage2_pipeline_present:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} is Stage-1-only in V1 and cannot be combined with stage2_ab.pipeline."
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
            stage2_ab=stage2_ab,
            rollout_matching=rollout_matching,
            rlhf=rlhf,
            prompts=prompts,
            deepspeed=deepspeed,
            global_max_length=global_max_length,
            extra={},
        )
