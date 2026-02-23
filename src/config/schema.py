"""Typed configuration schemas for training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    cast,
)

from src.common.object_field_order import (
    ObjectFieldOrder,
    normalize_object_field_order,
)

from .rollout_matching_schema import RolloutMatchingConfig
from .strict_dataclass import parse_dataclass_strict


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
        rendered = [f"{section}.{str(k)}" for k in sorted(unknown, key=lambda x: str(x))]
        raise ValueError(f"Unknown {section} keys: {rendered}")


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
        coord_monitor_mass = bool(payload.get("coord_monitor_mass", cls.coord_monitor_mass))
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
    temperature: float = 1.0
    target_sigma: float = 2.0
    target_truncate: Optional[int] = None

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CoordSoftCEW1Config":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("coord_soft_ce_w1 section must be a mapping when provided")

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
                raise ValueError("coord_soft_ce_w1.target_truncate must be an integer or null") from exc

        if ce_weight < 0:
            raise ValueError("coord_soft_ce_w1.ce_weight must be >= 0")
        if soft_ce_weight < 0:
            raise ValueError("coord_soft_ce_w1.soft_ce_weight must be >= 0")
        if w1_weight < 0:
            raise ValueError("coord_soft_ce_w1.w1_weight must be >= 0")
        if gate_weight < 0:
            raise ValueError("coord_soft_ce_w1.gate_weight must be >= 0")
        if enabled and ce_weight == 0 and soft_ce_weight == 0 and w1_weight == 0 and gate_weight == 0:
            raise ValueError(
                "coord_soft_ce_w1 is enabled but ce_weight, soft_ce_weight, w1_weight, and gate_weight are all 0"
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
            temperature=temperature,
            target_sigma=target_sigma,
            target_truncate=target_truncate,
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
class CustomConfig:
    train_jsonl: str
    user_prompt: str
    emit_norm: AllowedNorm
    json_format: AllowedJsonFormat
    object_field_order: ObjectFieldOrder
    object_ordering: Literal["sorted", "random"] = "sorted"
    coord_tokens: CoordTokensConfig = field(default_factory=CoordTokensConfig)
    coord_offset: CoordOffsetConfig = field(default_factory=CoordOffsetConfig)
    coord_soft_ce_w1: CoordSoftCEW1Config = field(default_factory=CoordSoftCEW1Config)
    use_summary: bool = False
    system_prompt_summary: Optional[str] = None
    augmentation: Optional[Mapping[str, Any]] = None
    augmentation_curriculum: Optional[Mapping[str, Any]] = None
    bypass_prob: float = 0.0
    trainer_variant: Optional[str] = None
    train_sample_limit: Optional[Any] = None
    val_sample_limit: Optional[Any] = None
    val_sample_with_replacement: bool = False
    dump_conversation_text: bool = False
    dump_conversation_path: Optional[str] = None
    val_jsonl: Optional[str] = None
    output_variant: Literal["dense", "summary"] = "dense"
    visual_kd: VisualKDConfig = field(default_factory=VisualKDConfig.disabled)
    token_type_metrics: TokenTypeMetricsConfig = field(default_factory=TokenTypeMetricsConfig)
    extra: Mapping[str, Any] = field(default_factory=dict)
    # Optional path to a fusion config (YAML/JSON) describing multiple datasets.
    # When set, training/eval datasets are built from the fusion config and
    # `train_jsonl`/`val_jsonl` are ignored by the runner.
    fusion_config: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.fusion_config and not self.train_jsonl:
            raise ValueError(
                "custom.train_jsonl must be provided when custom.fusion_config is not set"
            )
        if not self.user_prompt:
            raise ValueError("custom.user_prompt must be provided")
        if self.emit_norm != "none":
            raise ValueError(
                "Pre-normalized data is required; set custom.emit_norm: none (runtime normalization is disabled)."
            )
        if self.object_ordering not in {"sorted", "random"}:
            raise ValueError(
                "custom.object_ordering must be one of {'sorted', 'random'}"
            )
        if self.object_field_order not in {"desc_first", "geometry_first"}:
            raise ValueError(
                "custom.object_field_order must be one of {'desc_first', 'geometry_first'}"
            )
        if not isinstance(self.use_summary, bool):
            raise TypeError("custom.use_summary must be a boolean value")
        if not isinstance(self.val_sample_with_replacement, bool):
            raise TypeError("custom.val_sample_with_replacement must be a boolean value")
        if self.json_format not in ALLOWED_JSON_FORMATS:
            raise ValueError("custom.json_format must be 'standard'")
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
        val_sample_with_replacement_raw = data.pop(
            "val_sample_with_replacement", False
        )
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
        dump_conversation_text = bool(data.pop("dump_conversation_text", False))
        dump_conversation_path = data.pop("dump_conversation_path", None)
        object_ordering_raw = data.pop("object_ordering", "sorted")
        object_field_order_raw = data.pop("object_field_order", None)
        if object_field_order_raw is None:
            raise ValueError("custom.object_field_order must be provided")
        val_jsonl = data.pop("val_jsonl", None)
        fusion_config_raw = data.pop("fusion_config", None)
        fusion_config: Optional[str]
        if fusion_config_raw in (None, "", False):
            fusion_config = None
        elif isinstance(fusion_config_raw, str):
            fusion_config = fusion_config_raw.strip() or None
        else:
            raise TypeError("custom.fusion_config must be a string path when provided")
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

        object_ordering = str(object_ordering_raw).lower()
        if object_ordering not in {"sorted", "random"}:
            raise ValueError(
                "custom.object_ordering must be 'sorted' or 'random' when provided"
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
            object_ordering=cast(Literal["sorted", "random"], object_ordering),
            object_field_order=object_field_order,
            coord_tokens=coord_tokens,
            coord_offset=coord_offset,
            coord_soft_ce_w1=coord_soft_ce_w1,
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
            dump_conversation_text=dump_conversation_text,
            dump_conversation_path=str(dump_conversation_path)
            if dump_conversation_path is not None
            else None,
            val_jsonl=str(val_jsonl) if val_jsonl is not None else None,
            fusion_config=fusion_config,
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
                raise ValueError(f"{field_name} must be boolean (0 or 1), got {value!r}.")
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError(
                    f"{field_name} string value '{value}' is not a recognized boolean representation."
                )
            raise TypeError(f"{field_name} must be a boolean value, got {type(value)!r}.")

        enabled_raw = data.pop("enabled", False)
        enabled = _parse_bool(enabled_raw, "debug.enabled")

        output_dir_raw = data.pop("output_dir", None)
        output_dir = None if output_dir_raw in (None, "", False) else str(output_dir_raw)

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
            raise TypeError("stage2_ab.schedule.b_ratio must be a float in [0,1]") from exc

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
class Stage2ABChannelBConfig:
    drop_invalid_struct_ce_multiplier: float = 1.0

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
                "Pipeline overlap is automatic under vLLM server mode."
            )
        if "rollout_decode_batch_size" in data:
            raise ValueError(
                "stage2_ab.channel_b.rollout_decode_batch_size has been removed. "
                "Use rollout_matching.decode_batch_size instead."
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

        drop_invalid_raw = data.pop(
            "drop_invalid_struct_ce_multiplier", cls.drop_invalid_struct_ce_multiplier
        )
        try:
            drop_invalid_struct_ce_multiplier = float(drop_invalid_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "stage2_ab.channel_b.drop_invalid_struct_ce_multiplier must be a float"
            ) from exc
        if not (1.0 <= drop_invalid_struct_ce_multiplier <= 4.0):
            raise ValueError(
                "stage2_ab.channel_b.drop_invalid_struct_ce_multiplier must be in [1.0, 4.0]"
            )

        if data:
            raise ValueError(
                f"Unknown stage2_ab.channel_b keys: {sorted(str(k) for k in data.keys())}"
            )

        return cls(drop_invalid_struct_ce_multiplier=drop_invalid_struct_ce_multiplier)


@dataclass(frozen=True)
class Stage2ABConfig:
    schedule: Stage2ABScheduleConfig
    n_softctx_iter: int = 2
    softctx_grad_mode: Literal["unroll", "em_detach"] = "unroll"
    softctx_temperature: float = 1.0
    desc_ce_weight: float = 1.0
    bbox_smoothl1_weight: float = 1.0
    bbox_ciou_weight: float = 1.0

    coord_ce_weight: float = 0.0
    coord_el1_weight: float = 0.0
    coord_ehuber_weight: float = 0.0
    coord_huber_delta: float = 0.001
    coord_entropy_weight: float = 0.0
    coord_gate_weight: float = 0.0

    channel_b: Stage2ABChannelBConfig = field(default_factory=Stage2ABChannelBConfig)

    @classmethod
    def from_mapping(cls, payload: Any) -> "Stage2ABConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("stage2_ab section must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)

        schedule_raw = data.pop("schedule", None)
        if schedule_raw is None:
            raise ValueError("stage2_ab.schedule must be provided")
        schedule = Stage2ABScheduleConfig.from_mapping(schedule_raw)

        if "bbox_l1_weight" in data or "bbox_giou_weight" in data:
            raise ValueError(
                "stage2_ab.bbox_l1_weight/bbox_giou_weight are deprecated. "
                "Use stage2_ab.bbox_smoothl1_weight and stage2_ab.bbox_ciou_weight instead."
            )

        n_softctx_iter_raw = data.pop("n_softctx_iter", cls.n_softctx_iter)
        try:
            n_softctx_iter = int(n_softctx_iter_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError("stage2_ab.n_softctx_iter must be an int") from exc
        if n_softctx_iter < 1:
            raise ValueError("stage2_ab.n_softctx_iter must be >= 1")

        grad_mode_raw = data.pop("softctx_grad_mode", cls.softctx_grad_mode)
        softctx_grad_mode = str(grad_mode_raw).strip().lower()
        if softctx_grad_mode not in {"unroll", "em_detach"}:
            raise ValueError(
                "stage2_ab.softctx_grad_mode must be one of {'unroll','em_detach'}"
            )

        temp_raw = data.pop("softctx_temperature", cls.softctx_temperature)
        try:
            softctx_temperature = float(temp_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError("stage2_ab.softctx_temperature must be a float") from exc
        if softctx_temperature <= 0:
            raise ValueError("stage2_ab.softctx_temperature must be > 0")

        desc_ce_raw = data.pop("desc_ce_weight", cls.desc_ce_weight)
        try:
            desc_ce_weight = float(desc_ce_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError("stage2_ab.desc_ce_weight must be a float") from exc
        if desc_ce_weight < 0:
            raise ValueError("stage2_ab.desc_ce_weight must be >= 0")

        bbox_smoothl1_raw = data.pop("bbox_smoothl1_weight", cls.bbox_smoothl1_weight)
        try:
            bbox_smoothl1_weight = float(bbox_smoothl1_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError("stage2_ab.bbox_smoothl1_weight must be a float") from exc
        if bbox_smoothl1_weight < 0:
            raise ValueError("stage2_ab.bbox_smoothl1_weight must be >= 0")

        bbox_ciou_raw = data.pop("bbox_ciou_weight", cls.bbox_ciou_weight)
        try:
            bbox_ciou_weight = float(bbox_ciou_raw)
        except (TypeError, ValueError) as exc:
            raise TypeError("stage2_ab.bbox_ciou_weight must be a float") from exc
        if bbox_ciou_weight < 0:
            raise ValueError("stage2_ab.bbox_ciou_weight must be >= 0")

        coord_ce_weight = float(data.pop("coord_ce_weight", cls.coord_ce_weight) or 0.0)
        coord_ce_weight = max(0.0, coord_ce_weight)
        coord_el1_weight = float(data.pop("coord_el1_weight", cls.coord_el1_weight) or 0.0)
        coord_el1_weight = max(0.0, coord_el1_weight)
        coord_ehuber_weight = float(
            data.pop("coord_ehuber_weight", cls.coord_ehuber_weight) or 0.0
        )
        coord_ehuber_weight = max(0.0, coord_ehuber_weight)
        coord_huber_delta = float(data.pop("coord_huber_delta", cls.coord_huber_delta) or 0.001)
        coord_huber_delta = max(1e-6, coord_huber_delta)
        coord_entropy_weight = float(
            data.pop("coord_entropy_weight", cls.coord_entropy_weight) or 0.0
        )

        coord_gate_weight = float(
            data.pop("coord_gate_weight", cls.coord_gate_weight) or 0.0
        )
        coord_gate_weight = max(0.0, coord_gate_weight)

        channel_b = Stage2ABChannelBConfig.from_mapping(data.pop("channel_b", None))

        if data:
            unknown = [f"stage2_ab.{str(k)}" for k in sorted(data.keys(), key=lambda x: str(x))]
            raise ValueError(
                "Unknown stage2_ab keys: "
                f"{unknown}. "
                "Migration guidance: remove unsupported keys or move them into "
                "the current stage2_ab schema (for Channel-B options use stage2_ab.channel_b.*)."
            )

        return cls(
            schedule=schedule,
            n_softctx_iter=n_softctx_iter,
            softctx_grad_mode=cast(Literal["unroll", "em_detach"], softctx_grad_mode),
            softctx_temperature=softctx_temperature,
            desc_ce_weight=desc_ce_weight,
            bbox_smoothl1_weight=bbox_smoothl1_weight,
            bbox_ciou_weight=bbox_ciou_weight,
            coord_ce_weight=coord_ce_weight,
            coord_el1_weight=coord_el1_weight,
            coord_ehuber_weight=coord_ehuber_weight,
            coord_huber_delta=coord_huber_delta,
            coord_entropy_weight=coord_entropy_weight,
            coord_gate_weight=coord_gate_weight,
            channel_b=channel_b,
        )


@dataclass(frozen=True)
class TrainingConfig:
    template: Mapping[str, Any]
    custom: CustomConfig
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

        stage2_ab_raw = data.pop("stage2_ab", None)
        rollout_matching_raw = data.pop("rollout_matching", None)

        rlhf = dict(_as_dict(data.pop("rlhf", None), path="rlhf"))
        _validate_section_keys_strict(
            "rlhf", rlhf, allowed=_rlhf_arguments_allowed_keys()
        )
        custom_raw = data.pop("custom", None)
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

        stage2_ab = None
        if stage2_ab_raw is not None:
            stage2_ab = Stage2ABConfig.from_mapping(stage2_ab_raw)
        elif trainer_variant == "stage2_ab_training":
            raise ValueError(
                "stage2_ab section must be provided when custom.trainer_variant=stage2_ab_training"
            )

        rollout_matching = None
        if rollout_matching_raw is not None:
            if not isinstance(rollout_matching_raw, Mapping):
                raise TypeError("rollout_matching must be a mapping when provided")

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

        if trainer_variant in {"rollout_matching_sft", "stage2_ab_training"}:
            if rollout_matching is None:
                raise ValueError(
                    "rollout_matching section must be provided for rollout_matching_sft/stage2_ab_training"
                )

        # Length-coherence guardrails (fail-fast). These settings affect whether the
        # rollout backend will truncate/error on long prompts, which is objective-changing.
        if rollout_matching is not None:
            backend = str(getattr(rollout_matching, "rollout_backend", "") or "").strip()
            if backend == "vllm":
                vllm_cfg = getattr(rollout_matching, "vllm", None)
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

        return cls(
            template=template,
            custom=custom,
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
