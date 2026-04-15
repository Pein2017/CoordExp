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
from src.common.geometry.bbox_parameterization import (
    AllowedBBoxFormat,
    DEFAULT_BBOX_FORMAT,
    normalize_bbox_format,
)
from src.trainers.teacher_forcing.module_registry import (
    ALLOWED_DIAGNOSTIC_MODULES,
    ALLOWED_OBJECTIVE_MODULES,
    DIAGNOSTIC_CONFIG_ALLOWLIST,
    OBJECTIVE_APPLICATION_PRESET_ALLOWLIST,
    OBJECTIVE_CONFIG_ALLOWLIST,
    OBJECTIVE_OPTIONAL_CONFIG_KEYS,
    validate_bbox_geo_config_values,
    normalize_token_ce_stop_signal_damping_config,
    validate_adjacent_repulsion_config_values,
)

from .eval_monitor_dump_schema import EvalMonitorDumpConfig
from .rollout_matching_schema import RolloutMatchingConfig
from .strict_dataclass import dataclass_asdict_no_none, parse_dataclass_strict


AllowedNorm = Literal["none", "norm100", "norm1000"]
AllowedVisualDistance = Literal["mse", "cosine"]
AllowedJsonFormat = Literal["standard"]

ALLOWED_JSON_FORMATS: set[str] = {"standard"}


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
    "checkpoint_mode",
}


@lru_cache(maxsize=1)
def _training_allowed_keys() -> set[str]:
    return set(_train_arguments_allowed_keys()) | set(_TRAINING_INTERNAL_KEYS)


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
        if not self.enabled:
            raise ValueError(
                "Coord-token-only contract: custom.coord_tokens.enabled must be true."
            )
        if not self.skip_bbox_norm:
            raise ValueError(
                "Coord-token-only contract: custom.coord_tokens.skip_bbox_norm must be true to avoid double normalization."
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
    adjacent_repulsion_weight: float = 0.0
    adjacent_repulsion_filter_mode: str = "same_desc"
    adjacent_repulsion_margin_ratio: float = 0.05
    adjacent_repulsion_copy_margin: float = 0.8

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
            "adjacent_repulsion_weight",
            "adjacent_repulsion_filter_mode",
            "adjacent_repulsion_margin_ratio",
            "adjacent_repulsion_copy_margin",
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
        adjacent_repulsion_weight = _parse_float(
            "adjacent_repulsion_weight", cls.adjacent_repulsion_weight
        )
        adjacent_repulsion_margin_ratio = _parse_float(
            "adjacent_repulsion_margin_ratio", cls.adjacent_repulsion_margin_ratio
        )
        adjacent_repulsion_copy_margin = _parse_float(
            "adjacent_repulsion_copy_margin", cls.adjacent_repulsion_copy_margin
        )
        adjacent_repulsion_filter_mode = str(
            payload.get(
                "adjacent_repulsion_filter_mode",
                cls.adjacent_repulsion_filter_mode,
            )
            or cls.adjacent_repulsion_filter_mode
        ).strip().lower()

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
        if adjacent_repulsion_weight < 0:
            raise ValueError(
                "coord_soft_ce_w1.adjacent_repulsion_weight must be >= 0"
            )
        if adjacent_repulsion_margin_ratio < 0:
            raise ValueError(
                "coord_soft_ce_w1.adjacent_repulsion_margin_ratio must be >= 0"
            )
        if not 0.0 <= adjacent_repulsion_copy_margin <= 1.0:
            raise ValueError(
                "coord_soft_ce_w1.adjacent_repulsion_copy_margin must be within [0, 1]"
            )
        if adjacent_repulsion_filter_mode not in {"same_desc", "global"}:
            raise ValueError(
                "coord_soft_ce_w1.adjacent_repulsion_filter_mode must be one of ['global', 'same_desc']"
            )
        if (
            enabled
            and ce_weight == 0
            and soft_ce_weight == 0
            and w1_weight == 0
            and gate_weight == 0
            and text_gate_weight == 0
            and adjacent_repulsion_weight == 0
        ):
            raise ValueError(
                "coord_soft_ce_w1 is enabled but ce_weight, soft_ce_weight, w1_weight, gate_weight, text_gate_weight, and adjacent_repulsion_weight are all 0"
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
            adjacent_repulsion_weight=adjacent_repulsion_weight,
            adjacent_repulsion_filter_mode=adjacent_repulsion_filter_mode,
            adjacent_repulsion_margin_ratio=adjacent_repulsion_margin_ratio,
            adjacent_repulsion_copy_margin=adjacent_repulsion_copy_margin,
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

        data: MutableMapping[str, Any] = dict(payload)
        required = {
            "enabled",
            "smoothl1_weight",
            "ciou_weight",
        }
        optional = {
            "parameterization",
            "center_weight",
            "size_weight",
        }
        allowed = required | optional
        unknown = sorted(str(k) for k in data.keys() if k not in allowed)
        if unknown:
            raise ValueError(
                f"Unknown bbox_geo keys: {[f'bbox_geo.{k}' for k in unknown]}"
            )
        missing = sorted(str(k) for k in required if k not in data)
        if missing:
            raise ValueError(
                f"bbox_geo requires explicit keys: {[f'bbox_geo.{k}' for k in missing]}"
            )

        enabled = bool(data.pop("enabled"))

        def _parse_float(key: str, default: float) -> float:
            raw = data.pop(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bbox_geo.{key} must be numeric") from exc

        smoothl1_weight = _parse_float("smoothl1_weight", cls.smoothl1_weight)
        ciou_weight = _parse_float("ciou_weight", cls.ciou_weight)
        parameterization = (
            str(data.pop("parameterization", cls.parameterization) or cls.parameterization)
            .strip()
            .lower()
        )
        center_weight = _parse_float("center_weight", cls.center_weight)
        size_weight = _parse_float("size_weight", cls.size_weight)

        if smoothl1_weight < 0:
            raise ValueError("bbox_geo.smoothl1_weight must be >= 0")
        if ciou_weight < 0:
            raise ValueError("bbox_geo.ciou_weight must be >= 0")
        validate_bbox_geo_config_values(
            {
                "parameterization": parameterization,
                "center_weight": center_weight,
                "size_weight": size_weight,
            },
            path="bbox_geo",
        )
        if enabled and smoothl1_weight == 0 and ciou_weight == 0:
            raise ValueError(
                "bbox_geo is enabled but smoothl1_weight and ciou_weight are both 0"
            )

        return cls(
            enabled=enabled,
            smoothl1_weight=smoothl1_weight,
            ciou_weight=ciou_weight,
            parameterization=parameterization,
            center_weight=center_weight,
            size_weight=size_weight,
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

        data: MutableMapping[str, Any] = dict(payload)
        required = {
            "enabled",
            "log_wh_weight",
            "oversize_penalty_weight",
            "oversize_area_frac_threshold",
            "oversize_log_w_threshold",
            "oversize_log_h_threshold",
            "eps",
        }
        unknown = sorted(str(k) for k in data.keys() if k not in required)
        if unknown:
            raise ValueError(
                f"Unknown bbox_size_aux keys: {[f'bbox_size_aux.{k}' for k in unknown]}"
            )
        missing = sorted(str(k) for k in required if k not in data)
        if missing:
            raise ValueError(
                f"bbox_size_aux requires explicit keys: {[f'bbox_size_aux.{k}' for k in missing]}"
            )

        enabled = bool(data.pop("enabled"))

        def _parse_float(key: str, default: float) -> float:
            raw = data.pop(key)
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bbox_size_aux.{key} must be numeric") from exc

        def _parse_optional_float(key: str) -> Optional[float]:
            raw = data.pop(key)
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"bbox_size_aux.{key} must be numeric or null"
                ) from exc

        log_wh_weight = _parse_float("log_wh_weight", cls.log_wh_weight)
        oversize_penalty_weight = _parse_float(
            "oversize_penalty_weight", cls.oversize_penalty_weight
        )
        oversize_area_frac_threshold = _parse_optional_float(
            "oversize_area_frac_threshold"
        )
        oversize_log_w_threshold = _parse_optional_float("oversize_log_w_threshold")
        oversize_log_h_threshold = _parse_optional_float("oversize_log_h_threshold")
        eps = _parse_float("eps", cls.eps)

        if log_wh_weight < 0:
            raise ValueError("bbox_size_aux.log_wh_weight must be >= 0")
        if oversize_penalty_weight < 0:
            raise ValueError("bbox_size_aux.oversize_penalty_weight must be >= 0")
        if enabled and (log_wh_weight == 0 and oversize_penalty_weight == 0):
            raise ValueError(
                "bbox_size_aux is enabled but log_wh_weight and oversize_penalty_weight are both 0"
            )
        if eps <= 0:
            raise ValueError("bbox_size_aux.eps must be > 0")
        if (
            oversize_area_frac_threshold is not None
            and oversize_area_frac_threshold < 0
        ):
            raise ValueError(
                "bbox_size_aux.oversize_area_frac_threshold must be >= 0 or null"
            )
        return cls(
            enabled=enabled,
            log_wh_weight=log_wh_weight,
            oversize_penalty_weight=oversize_penalty_weight,
            oversize_area_frac_threshold=oversize_area_frac_threshold,
            oversize_log_w_threshold=oversize_log_w_threshold,
            oversize_log_h_threshold=oversize_log_h_threshold,
            eps=eps,
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
    pred_score_source: str = "stage1_eval_constant"
    pred_score_version: int = 1
    constant_score: float = 1.0
    batch_size: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    limit: Optional[int] = None
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
class CustomConfig:
    train_jsonl: str
    user_prompt: str
    emit_norm: AllowedNorm
    json_format: AllowedJsonFormat
    object_field_order: ObjectFieldOrder
    bbox_format: AllowedBBoxFormat = DEFAULT_BBOX_FORMAT
    object_ordering: ObjectOrdering = "sorted"
    coord_tokens: CoordTokensConfig = field(default_factory=CoordTokensConfig)
    coord_offset: CoordOffsetConfig = field(default_factory=CoordOffsetConfig)
    coord_soft_ce_w1: CoordSoftCEW1Config = field(default_factory=CoordSoftCEW1Config)
    bbox_geo: BBoxGeoConfig = field(default_factory=BBoxGeoConfig)
    bbox_size_aux: BBoxSizeAuxConfig = field(default_factory=BBoxSizeAuxConfig)
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
                "custom.fusion_config is temporarily disabled. "
                "CoordExp now supports only the canonical single-dataset training configs; "
                "merge JSONLs offline if you need dataset mixing for now. "
                "Legacy fusion examples remain in-tree for future reactivation."
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
        coord_offset_raw = data.pop("coord_offset", None)
        coord_offset = CoordOffsetConfig.from_mapping(coord_offset_raw)
        # Deprecated legacy knob: ignore to ease config refactors.
        # (Stage-2 AB contract refactor requires this to be non-fatal.)
        data.pop("coord_loss", None)
        coord_soft_ce_w1_raw = data.pop("coord_soft_ce_w1", None)
        coord_soft_ce_w1 = CoordSoftCEW1Config.from_mapping(coord_soft_ce_w1_raw)
        bbox_geo_raw = data.pop("bbox_geo", None)
        bbox_geo = BBoxGeoConfig.from_mapping(bbox_geo_raw)
        bbox_size_aux_raw = data.pop("bbox_size_aux", None)
        bbox_size_aux = BBoxSizeAuxConfig.from_mapping(bbox_size_aux_raw)
        eval_detection_raw = data.pop("eval_detection", None)
        eval_detection = Stage1EvalDetectionConfig.from_mapping(eval_detection_raw)

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
            coord_tokens=coord_tokens,
            coord_offset=coord_offset,
            coord_soft_ce_w1=coord_soft_ce_w1,
            bbox_geo=bbox_geo,
            bbox_size_aux=bbox_size_aux,
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
class Stage2ABChannelBConfig:
    duplicate_control: Stage2ABChannelBDuplicateControlConfig = field(
        default_factory=Stage2ABChannelBDuplicateControlConfig
    )
    producer_wait_timeout_s: Optional[float] = None
    ddp_phase_timeout_s: Optional[float] = None
    invalid_rollout_policy: str = "abort"
    insertion_order: str = "tail_append"
    pseudo_positive: Stage2ABChannelBPseudoPositiveConfig = field(
        default_factory=Stage2ABChannelBPseudoPositiveConfig
    )
    triage_posterior: Stage2ABChannelBTriagePosteriorConfig = field(
        default_factory=Stage2ABChannelBTriagePosteriorConfig
    )

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2ABChannelBConfig":
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

        duplicate_control = Stage2ABChannelBDuplicateControlConfig.from_mapping(
            data.pop("duplicate_control", None)
        )

        invalid_rollout_policy_raw = data.pop(
            "invalid_rollout_policy",
            cls.invalid_rollout_policy,
        )
        invalid_rollout_policy = str(invalid_rollout_policy_raw).strip().lower()
        if invalid_rollout_policy not in {"abort", "dump_and_continue"}:
            raise ValueError(
                "stage2_ab.channel_b.invalid_rollout_policy must be one of "
                "{'abort', 'dump_and_continue'}"
            )

        insertion_order_raw = data.pop(
            "insertion_order",
            cls.insertion_order,
        )
        insertion_order = str(insertion_order_raw).strip().lower()
        if insertion_order not in {"tail_append", "sorted"}:
            raise ValueError(
                "stage2_ab.channel_b.insertion_order must be one of "
                "{'tail_append', 'sorted'}"
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
            if pseudo_positive.enabled
            else Stage2ABChannelBTriagePosteriorConfig.num_rollouts
        )
        triage_posterior = Stage2ABChannelBTriagePosteriorConfig.from_mapping(
            data.pop("triage_posterior", None),
            default_num_rollouts=triage_default_rollouts,
        )
        if (
            not pseudo_positive.enabled
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
            duplicate_control=duplicate_control,
            producer_wait_timeout_s=producer_wait_timeout_s,
            ddp_phase_timeout_s=ddp_phase_timeout_s,
            invalid_rollout_policy=invalid_rollout_policy,
            insertion_order=insertion_order,
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
    def from_mapping(cls, payload: Any) -> "Stage2PipelineConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab.pipeline must be a mapping")
        data: MutableMapping[str, Any] = dict(payload)

        objective_raw = data.pop("objective", [])
        diagnostics_raw = data.pop("diagnostics", [])

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
                allowed_names=ALLOWED_OBJECTIVE_MODULES,
            )
            for idx, item in enumerate(objective_raw)
        ]
        if not objective_specs:
            raise ValueError("stage2_ab.pipeline.objective must be non-empty")
        diagnostics_specs = [
            Stage2PipelineModuleSpec.from_mapping(
                item,
                path=f"stage2_ab.pipeline.diagnostics[{idx}]",
                allowed_names=ALLOWED_DIAGNOSTIC_MODULES,
            )
            for idx, item in enumerate(diagnostics_raw)
        ]

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

        canonical_objective_order = [
            "token_ce",
            "loss_duplicate_burst_unlikelihood",
            "bbox_geo",
            "bbox_size_aux",
            "coord_reg",
        ]
        authored_objective_order = [str(spec.name) for spec in objective_specs]
        if authored_objective_order != canonical_objective_order:
            raise ValueError(
                "stage2_ab.pipeline.objective must use the canonical module order "
                f"{canonical_objective_order}; got {authored_objective_order}"
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
            if str(spec.name) == "token_ce" and "stop_signal_damping" in spec.config:
                normalize_token_ce_stop_signal_damping_config(
                    spec.config.get("stop_signal_damping"),
                    path=(
                        "stage2_ab.pipeline.objective"
                        f"[{idx}].config.stop_signal_damping"
                    ),
                )
            if str(spec.name) == "token_ce" and "struct_ce_weight" in spec.config:
                raise ValueError(
                    "stage2_ab.pipeline.objective"
                    f"[{idx}].config.struct_ce_weight is deprecated and unsupported. "
                    "Remove the self-context struct/EOS stabilizer; active Stage-2 Channel-A "
                    "training uses only the single-pass anchor_text_only contract."
                )
            allowed_cfg = OBJECTIVE_CONFIG_ALLOWLIST.get(str(spec.name), set())
            unknown_cfg = set(spec.config.keys()) - allowed_cfg
            if unknown_cfg:
                raise ValueError(
                    "Unknown stage2_ab.pipeline.objective"
                    f"[{idx}].config keys for module {spec.name!r}: "
                    f"{sorted(str(k) for k in unknown_cfg)}"
                )
            optional_cfg = OBJECTIVE_OPTIONAL_CONFIG_KEYS.get(str(spec.name), set())
            missing_cfg = allowed_cfg - set(spec.config.keys()) - set(optional_cfg)
            if missing_cfg:
                raise ValueError(
                    "Missing required stage2_ab.pipeline.objective"
                    f"[{idx}].config keys for module {spec.name!r}: "
                    f"{sorted(str(k) for k in missing_cfg)}"
                )
            if str(spec.name) == "bbox_geo":
                validate_bbox_geo_config_values(
                    spec.config,
                    path=f"stage2_ab.pipeline.objective[{idx}].config",
                )
                if isinstance(spec.config, dict):
                    spec.config.setdefault("parameterization", "xyxy")
                    spec.config.setdefault("center_weight", 1.0)
                    spec.config.setdefault("size_weight", 1.0)
            if str(spec.name) == "coord_reg":
                validate_adjacent_repulsion_config_values(
                    spec.config,
                    path=f"stage2_ab.pipeline.objective[{idx}].config",
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
        loss_duplicate_burst_unlikelihood = specs_by_name.get(
            "loss_duplicate_burst_unlikelihood"
        )
        if loss_duplicate_burst_unlikelihood is None:
            raise ValueError(
                "stage2_ab.pipeline.objective requires loss_duplicate_burst_unlikelihood in the canonical "
                "clean-prefix Channel-B contract."
            )
        if tuple(str(ch) for ch in loss_duplicate_burst_unlikelihood.channels) != (
            "B",
        ):
            raise ValueError(
                "stage2_ab.pipeline.objective loss_duplicate_burst_unlikelihood must declare channels ['B'] "
                f"for the canonical clean-prefix Channel-B contract; got {list(loss_duplicate_burst_unlikelihood.channels)!r}"
            )

        bbox_geo = specs_by_name.get("bbox_geo")
        bbox_size_aux = specs_by_name.get("bbox_size_aux")
        coord_reg = specs_by_name.get("coord_reg")

        def _coord_targets_for_preset(preset: str) -> set[str]:
            if preset == "anchor_only":
                return {"coord"}
            raise ValueError(f"Unsupported coord/bbox application preset: {preset!r}")

        if bbox_size_aux is not None and bool(bbox_size_aux.enabled):
            if bbox_geo is None or not bool(bbox_geo.enabled):
                raise ValueError(
                    "stage2_ab.pipeline.objective requires bbox_geo to be present+enabled when bbox_size_aux is enabled "
                    "(bbox_size_aux depends on bbox_geo state: decoded bbox tensors + weights)."
                )
            missing_channels = set(bbox_size_aux.channels) - set(bbox_geo.channels)
            if missing_channels:
                raise ValueError(
                    "stage2_ab.pipeline.objective bbox_size_aux channels must be a subset of bbox_geo channels; "
                    f"missing={sorted(missing_channels)}"
                )
        if coord_reg is not None and bool(coord_reg.enabled):
            if bbox_geo is None or not bool(bbox_geo.enabled):
                raise ValueError(
                    "stage2_ab.pipeline.objective requires bbox_geo to be present+enabled when coord_reg is enabled "
                    "(coord_reg depends on bbox_geo state: coord logits + targets)."
                )
            missing_channels = set(coord_reg.channels) - set(bbox_geo.channels)
            if missing_channels:
                raise ValueError(
                    "stage2_ab.pipeline.objective coord_reg channels must be a subset of bbox_geo channels; "
                    f"missing={sorted(missing_channels)}"
                )

        if bbox_geo is not None and bool(bbox_geo.enabled):
            bbox_geo_preset = str(bbox_geo.application.get("preset", "") or "").strip()
            bbox_geo_targets = _coord_targets_for_preset(bbox_geo_preset)
            for dep_name, dep_spec in (
                ("bbox_size_aux", bbox_size_aux),
                ("coord_reg", coord_reg),
            ):
                if dep_spec is None or not bool(dep_spec.enabled):
                    continue
                dep_preset = str(dep_spec.application.get("preset", "") or "").strip()
                dep_targets = _coord_targets_for_preset(dep_preset)
                if not dep_targets.issubset(bbox_geo_targets):
                    raise ValueError(
                        "stage2_ab.pipeline.objective "
                        f"{dep_name} application must be a subset of bbox_geo; "
                        f"bbox_geo={sorted(bbox_geo_targets)} {dep_name}={sorted(dep_targets)}"
                    )

        for dspec in diagnostics_specs:
            if not bool(dspec.enabled):
                continue
            if dspec.name != "coord_diag":
                continue
            if bbox_geo is None or not bool(bbox_geo.enabled):
                raise ValueError(
                    "stage2_ab.pipeline.diagnostics requires bbox_geo to be present+enabled when coord_diag is enabled "
                    "(coord_diag depends on bbox_geo state)."
                )
            missing_channels = set(dspec.channels) - set(bbox_geo.channels)
            if missing_channels:
                raise ValueError(
                    "stage2_ab.pipeline.diagnostics coord_diag channels must be a subset of bbox_geo channels; "
                    f"missing={sorted(missing_channels)}"
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
    def from_mapping(cls, payload: Any) -> "Stage2ABConfig":
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
        pipeline = Stage2PipelineConfig.from_mapping(pipeline_raw)

        if "bbox_l1_weight" in data or "bbox_giou_weight" in data:
            raise ValueError(
                "stage2_ab.bbox_l1_weight/bbox_giou_weight are deprecated. "
                "Move bbox weights into stage2_ab.pipeline.objective[*].config for the bbox_geo module "
                "using smoothl1_weight/ciou_weight."
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
                "(token_ce: anchor_text_only; bbox_geo/bbox_size_aux/coord_reg: anchor_only). "
                f"Found: {sorted(deprecated_keys)}"
            )

        channel_b = Stage2ABChannelBConfig.from_mapping(data.pop("channel_b", None))

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


@dataclass(frozen=True)
class TrainingConfig:
    template: Mapping[str, Any]
    custom: CustomConfig
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    model: Mapping[str, Any] = field(default_factory=dict)
    quantization: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)
    tuner: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
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
        debug = DebugConfig.from_mapping(data.pop("debug", None))
        deepspeed = DeepSpeedConfig.from_mapping(data.pop("deepspeed", None))
        global_max_length = data.pop("global_max_length", None)

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
                "custom.trainer_variant=rollout_matching_sft has been removed; use stage2_rollout_aligned"
            )

        stage2_ab = None
        if stage2_ab_raw is not None:
            stage2_ab = Stage2ABConfig.from_mapping(stage2_ab_raw)
        elif trainer_variant == "stage2_two_channel":
            raise ValueError(
                "stage2_ab section must be provided when custom.trainer_variant=stage2_two_channel"
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

        if trainer_variant in {"stage2_rollout_aligned", "stage2_two_channel"}:
            if rollout_matching is None:
                raise ValueError(
                    "rollout_matching section must be provided for stage2_rollout_aligned/stage2_two_channel"
                )

        stage2_pipeline_present = bool(
            stage2_ab is not None and getattr(stage2_ab, "pipeline", None) is not None
        )
        rollout_pipeline_present = bool(
            rollout_matching is not None
            and getattr(rollout_matching, "pipeline", None) is not None
        )

        if trainer_variant == "stage2_rollout_aligned" and not rollout_pipeline_present:
            raise ValueError(
                "rollout_matching.pipeline must be provided when custom.trainer_variant=stage2_rollout_aligned "
                "(no implicit default objective manifest)."
            )

        if stage2_pipeline_present and custom_coord_soft_ce_w1_present:
            raise ValueError(
                "stage2_ab.pipeline is provided; custom.coord_soft_ce_w1.* is disallowed and must be moved into "
                "stage2_ab.pipeline.objective[*].config for the coord_reg module"
            )
        if stage2_pipeline_present and custom_bbox_geo_present:
            raise ValueError(
                "stage2_ab.pipeline is provided; custom.bbox_geo.* is disallowed and must be moved into "
                "stage2_ab.pipeline.objective[*].config for the bbox_geo module"
            )
        if stage2_pipeline_present and custom_bbox_size_aux_present:
            raise ValueError(
                "stage2_ab.pipeline is provided; custom.bbox_size_aux.* is disallowed and must be moved into "
                "stage2_ab.pipeline.objective[*].config for the bbox_size_aux module"
            )
        if rollout_pipeline_present and custom_coord_soft_ce_w1_present:
            raise ValueError(
                "rollout_matching.pipeline is provided; custom.coord_soft_ce_w1.* is disallowed and must be moved into "
                "rollout_matching.pipeline.objective[*].config for the coord_reg module"
            )
        if rollout_pipeline_present and custom_bbox_geo_present:
            raise ValueError(
                "rollout_matching.pipeline is provided; custom.bbox_geo.* is disallowed and must be moved into "
                "rollout_matching.pipeline.objective[*].config for the bbox_geo module"
            )
        if rollout_pipeline_present and custom_bbox_size_aux_present:
            raise ValueError(
                "rollout_matching.pipeline is provided; custom.bbox_size_aux.* is disallowed and must be moved into "
                "rollout_matching.pipeline.objective[*].config for the bbox_size_aux module"
            )

        if trainer_variant == "stage2_two_channel" and rollout_pipeline_present:
            raise ValueError(
                "rollout_matching.pipeline is not allowed when custom.trainer_variant=stage2_two_channel. "
                "Use stage2_ab.pipeline instead."
            )
        if trainer_variant == "stage2_rollout_aligned" and stage2_pipeline_present:
            raise ValueError(
                "stage2_ab.pipeline is not allowed when custom.trainer_variant=stage2_rollout_aligned. "
                "Use rollout_matching.pipeline instead."
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
            if trainer_variant in {"stage2_two_channel", "stage2_rollout_aligned"}:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} is Stage-1-only in V1 and is unsupported for stage2 trainer variants."
                )
            if stage2_pipeline_present or rollout_pipeline_present:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} is Stage-1-only in V1 and cannot be combined with stage2_ab.pipeline or rollout_matching.pipeline."
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
            if float(getattr(coord_cfg, "adjacent_repulsion_weight", 0.0)) != 0.0:
                raise ValueError(
                    f"custom.bbox_format={bbox_format_label} requires custom.coord_soft_ce_w1.adjacent_repulsion_weight = 0."
                )
            if custom_bbox_geo_present or bool(getattr(custom.bbox_geo, "enabled", False)):
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
            debug=debug,
            model=model,
            quantization=quantization,
            data=data_section,
            tuner=tuner,
            training=training,
            stage2_ab=stage2_ab,
            rollout_matching=rollout_matching,
            rlhf=rlhf,
            prompts=prompts,
            deepspeed=deepspeed,
            global_max_length=global_max_length,
            extra={},
        )
