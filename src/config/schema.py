"""Typed configuration schemas for training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    cast,
)

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


def _as_dict(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section must be a mapping, got {type(value)!r}")
    return value


@dataclass(frozen=True)
class TokenTypeMetricsConfig:
    enabled: bool = False
    include: tuple[str, ...] = ("lvis",)
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        inc = tuple(str(v).strip().lower() for v in self.include)
        exc = tuple(str(v).strip().lower() for v in self.exclude)
        object.__setattr__(self, "include", inc)
        object.__setattr__(self, "exclude", exc)

    @classmethod
    def from_mapping(cls, payload: Any) -> "TokenTypeMetricsConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("custom.token_type_metrics must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        include_raw = payload.get("include", cls.include)
        exclude_raw = payload.get("exclude", cls.exclude)

        def _to_tuple(value: Any) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, (list, tuple)):
                return tuple(str(v).strip() for v in value)
            return (str(value).strip(),)

        include = _to_tuple(include_raw)
        exclude = _to_tuple(exclude_raw)

        return cls(enabled=enabled, include=include, exclude=exclude)


@dataclass(frozen=True)
class CoordTokensConfig:
    enabled: bool = False
    skip_bbox_norm: bool = True

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CoordTokensConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("coord_tokens section must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))
        skip_bbox_norm = bool(payload.get("skip_bbox_norm", True))
        return cls(
            enabled=enabled,
            skip_bbox_norm=skip_bbox_norm,
        )


@dataclass(frozen=True)
class CoordLossConfig:
    enabled: bool = False
    l1_weight: float = 0.0
    giou_weight: float = 0.0
    coord_ce_weight: float = 1.0
    non_coord_ce_weight: float = 1.0
    top_k: float = 0.1
    temperature: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CoordLossConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("coord_loss section must be a mapping when provided")

        enabled = bool(payload.get("enabled", False))

        def _parse_float(key: str, default: float) -> float:
            raw = payload.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"coord_loss.{key} must be numeric") from exc

        l1_weight = _parse_float("l1_weight", cls.l1_weight)
        giou_weight = _parse_float("giou_weight", cls.giou_weight)
        coord_ce_weight = _parse_float("coord_ce_weight", cls.coord_ce_weight)
        non_coord_ce_weight = _parse_float(
            "non_coord_ce_weight", cls.non_coord_ce_weight
        )
        top_k = _parse_float("top_k", cls.top_k)
        temperature = _parse_float("temperature", cls.temperature)

        if l1_weight < 0:
            raise ValueError("coord_loss.l1_weight must be >= 0")
        if giou_weight < 0:
            raise ValueError("coord_loss.giou_weight must be >= 0")
        if coord_ce_weight < 0:
            raise ValueError("coord_loss.coord_ce_weight must be >= 0")
        if non_coord_ce_weight < 0:
            raise ValueError("coord_loss.non_coord_ce_weight must be >= 0")
        if top_k <= 0:
            raise ValueError("coord_loss.top_k must be > 0")
        if temperature <= 0:
            raise ValueError("coord_loss.temperature must be > 0")

        return cls(
            enabled=enabled,
            l1_weight=l1_weight,
            giou_weight=giou_weight,
            coord_ce_weight=coord_ce_weight,
            non_coord_ce_weight=non_coord_ce_weight,
            top_k=top_k,
            temperature=temperature,
        )


@dataclass(frozen=True)
class CoordOffsetConfig:
    enabled: bool = False
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

        ids_raw = payload.get("ids")
        ids: tuple[int, ...]
        if ids_raw is None:
            ids = ()
        elif isinstance(ids_raw, (list, tuple, set)):
            try:
                ids = tuple(int(v) for v in ids_raw)
            except Exception as exc:
                raise ValueError("coord_offset.ids must be a list of integers") from exc
        elif isinstance(ids_raw, Mapping):
            start = ids_raw.get("start")
            end = ids_raw.get("end")
            try:
                start_i = int(start)
                end_i = int(end)
            except Exception as exc:
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
        if "enabled" not in payload:
            raise ValueError("deepspeed.enabled must be explicitly set")

        enabled = bool(payload["enabled"])

        if enabled:
            if "config" not in payload:
                raise ValueError(
                    "deepspeed.config must be provided when deepspeed.enabled is true"
                )
            config_value = payload["config"]
        else:
            config_value = payload.get("config")

        if enabled and (config_value is None or config_value == ""):
            raise ValueError(
                "deepspeed.config must be a non-empty value when deepspeed.enabled is true"
            )
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
class VisualKDTargetConfig:
    enabled: bool = False
    weight: float = 0.0
    distance: AllowedVisualDistance = "mse"

    def __post_init__(self) -> None:
        if self.enabled and self.weight <= 0:
            raise ValueError("visual_kd.*.weight must be > 0 when enabled")
        if self.distance not in {"mse", "cosine"}:
            raise ValueError("visual_kd.*.distance must be one of {mse, cosine}")


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

        enabled = bool(payload.get("enabled", False))
        if not enabled:
            return cls.disabled()

        def parse_target(
            name: str, raw: Optional[Mapping[str, Any]]
        ) -> VisualKDTargetConfig:
            if raw is None:
                return VisualKDTargetConfig()
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"custom.visual_kd.{name} must be a mapping when provided"
                )

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

        vit_cfg = parse_target("vit", payload.get("vit"))
        aligner_cfg = parse_target("aligner", payload.get("aligner"))
        deepstack_cfg = parse_target("deepstack", payload.get("deepstack"))

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
    object_ordering: Literal["sorted", "random"] = "sorted"
    coord_tokens: CoordTokensConfig = field(default_factory=CoordTokensConfig)
    coord_offset: CoordOffsetConfig = field(default_factory=CoordOffsetConfig)
    coord_loss: CoordLossConfig = field(default_factory=CoordLossConfig)
    use_summary: bool = False
    system_prompt_summary: Optional[str] = None
    augmentation: Optional[Mapping[str, Any]] = None
    augmentation_curriculum: Optional[Mapping[str, Any]] = None
    bypass_prob: float = 0.0
    trainer_variant: Optional[str] = None
    sample_limit: Optional[Any] = None
    train_sample_limit: Optional[Any] = None
    val_sample_limit: Optional[Any] = None
    val_sample_with_replacement: bool = False
    dump_conversation_text: bool = False
    dump_conversation_path: Optional[str] = None
    val_jsonl: Optional[str] = None
    output_variant: Literal["dense", "summary"] = "dense"
    visual_kd: VisualKDConfig = field(default_factory=VisualKDConfig.disabled)
    hard_sample_mining: Optional["HardSampleMiningConfig"] = None  # Deprecated: not wired; will error if provided
    token_type_metrics: TokenTypeMetricsConfig = field(default_factory=TokenTypeMetricsConfig)
    extra: Mapping[str, Any] = field(default_factory=dict)
    fusion_config: Optional[str] = None  # Deprecated: fusion disabled

    def __post_init__(self) -> None:
        if not self.train_jsonl:
            raise ValueError("custom.train_jsonl must be provided")
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
        if not isinstance(self.use_summary, bool):
            raise TypeError("custom.use_summary must be a boolean value")
        if not isinstance(self.val_sample_with_replacement, bool):
            raise TypeError("custom.val_sample_with_replacement must be a boolean value")
        if self.json_format not in ALLOWED_JSON_FORMATS:
            raise ValueError("custom.json_format must be 'standard'")
        if (
            self.val_sample_with_replacement
            and self.val_sample_limit is None
            and self.sample_limit is None
        ):
            raise ValueError(
                "custom.val_sample_with_replacement requires custom.val_sample_limit "
                "or custom.sample_limit to specify the eval sample size"
            )

    @classmethod
    def from_mapping(
        cls, payload: Optional[Mapping[str, Any]], *, prompts: PromptOverrides
    ) -> "CustomConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("custom section must be a mapping")

        data: MutableMapping[str, Any] = dict(payload)
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
        sample_limit = data.pop("sample_limit", None)
        train_sample_limit = data.pop("train_sample_limit", None)
        val_sample_limit = data.pop("val_sample_limit", None)
        dump_conversation_text = bool(data.pop("dump_conversation_text", False))
        dump_conversation_path = data.pop("dump_conversation_path", None)
        object_ordering_raw = data.pop("object_ordering", "sorted")
        val_jsonl = data.pop("val_jsonl", None)
        fusion_config_raw = data.pop("fusion_config", None)
        if fusion_config_raw not in (None, "", False):
            raise ValueError(
                "custom.fusion_config is deprecated while the pipeline focuses on a single LVIS dataset. "
                "Remove this field to continue."
            )
        fusion_config = None
        visual_kd_raw = data.pop("visual_kd", None)
        visual_kd = VisualKDConfig.from_mapping(visual_kd_raw)
        hsm_raw = data.pop("hard_sample_mining", None)
        if hsm_raw is not None:
            raise ValueError(
                "custom.hard_sample_mining is deprecated and unsupported. Remove this section to continue."
            )
        hsm_cfg = None
        token_type_metrics_raw = data.pop("token_type_metrics", None)
        token_type_metrics = TokenTypeMetricsConfig.from_mapping(token_type_metrics_raw)
        json_format_raw = data.pop("json_format", None)
        if json_format_raw is None:
            raise ValueError("custom.json_format must be provided")
        json_format = _normalize_json_format(json_format_raw)

        extra = dict(data)

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
        coord_loss_raw = data.pop("coord_loss", None)
        coord_loss = CoordLossConfig.from_mapping(coord_loss_raw)

        object_ordering = str(object_ordering_raw).lower()
        if object_ordering not in {"sorted", "random"}:
            raise ValueError(
                "custom.object_ordering must be 'sorted' or 'random' when provided"
            )

        return cls(
            train_jsonl=str(train_jsonl) if train_jsonl is not None else "",
            user_prompt=str(user_prompt) if user_prompt is not None else "",
            emit_norm=cast("AllowedNorm", emit_norm_value),
            json_format=json_format,
            object_ordering=cast(Literal["sorted", "random"], object_ordering),
            coord_tokens=coord_tokens,
            coord_offset=coord_offset,
            coord_loss=coord_loss,
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
            sample_limit=sample_limit,
            train_sample_limit=train_sample_limit,
            val_sample_limit=val_sample_limit,
            val_sample_with_replacement=val_sample_with_replacement,
            dump_conversation_text=dump_conversation_text,
            dump_conversation_path=str(dump_conversation_path)
            if dump_conversation_path is not None
            else None,
            val_jsonl=str(val_jsonl) if val_jsonl is not None else None,
            fusion_config=str(fusion_config) if fusion_config is not None else None,
            output_variant=prompts.output_variant,
            visual_kd=visual_kd,
            hard_sample_mining=hsm_cfg,
            token_type_metrics=token_type_metrics,
            extra=extra,
        )


@dataclass(frozen=True)
class TrainingConfig:
    template: Mapping[str, Any]
    custom: CustomConfig
    model: Mapping[str, Any] = field(default_factory=dict)
    quantization: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)
    tuner: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
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

        data = dict(payload)

        model = dict(_as_dict(data.pop("model", None)))
        quantization = dict(_as_dict(data.pop("quantization", None)))
        template_raw = _as_dict(data.pop("template", None))
        template = dict(template_raw)
        data_section = dict(_as_dict(data.pop("data", None)))
        tuner = dict(_as_dict(data.pop("tuner", None)))
        training = dict(_as_dict(data.pop("training", None)))
        rlhf = dict(_as_dict(data.pop("rlhf", None)))
        custom_raw = data.pop("custom", None)
        deepspeed = DeepSpeedConfig.from_mapping(data.pop("deepspeed", None))
        global_max_length = data.pop("global_max_length", None)

        extra = dict(data)

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

        return cls(
            template=template,
            custom=custom,
            model=model,
            quantization=quantization,
            data=data_section,
            tuner=tuner,
            training=training,
            rlhf=rlhf,
            prompts=prompts,
            deepspeed=deepspeed,
            global_max_length=global_max_length,
            extra=extra,
        )

@dataclass(frozen=True)
class HardSampleMiningConfig:
    """Deprecated configuration placeholder for hard sample mining."""
    enabled: bool = False
    start_epoch: int = 0
    hard_sample_size: int = 500
    regular_sample_size: int = 150
    ema_decay: float = 0.9
    mine_clean: bool = False
    recompute_full_pass: bool = False

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> Optional["HardSampleMiningConfig"]:
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("custom.hard_sample_mining must be a mapping when provided")

        data = dict(payload)
        enabled = bool(data.pop("enabled", False))
        if not enabled:
            return cls(enabled=False)

        start_epoch = int(data.pop("start_epoch", 0))
        hard_sample_size = int(data.pop("hard_sample_size", 500))
        if hard_sample_size <= 0:
            raise ValueError("custom.hard_sample_mining.hard_sample_size must be >0")
        regular_sample_size = int(data.pop("regular_sample_size", 150))
        if regular_sample_size < 0:
            raise ValueError("custom.hard_sample_mining.regular_sample_size must be >=0")

        ema_decay = float(data.pop("ema_decay", 0.9))
        if not (0 < ema_decay <= 1):
            raise ValueError("custom.hard_sample_mining.ema_decay must be in (0,1]")
        mine_clean = bool(data.pop("mine_clean", False))
        recompute_full_pass = bool(data.pop("recompute_full_pass", False))

        return cls(
            enabled=True,
            start_epoch=start_epoch,
            hard_sample_size=hard_sample_size,
            regular_sample_size=regular_sample_size,
            ema_decay=ema_decay,
            mine_clean=mine_clean,
            recompute_full_pass=recompute_full_pass,
        )
