"""Typed V1 training configuration models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CONFIG_LOADER_VERSION = "coordexp-swift-config-v1"

ForwardInputProviderMode = Literal["synchronous", "overlapped"]
PackingPolicy = Literal[
    "source_order_next_fit",
    "window_binpack",
    "online_window_binpack",
]


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CadenceConfig(StrictConfigModel):
    every_fraction: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    steps: tuple[int, ...] = ()

    @field_validator("steps")
    @classmethod
    def _steps_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(step <= 0 for step in value):
            raise ValueError("cadence steps must be positive planned-step ids")
        return value


class ObservabilityConfig(StrictConfigModel):
    """Rank-zero presentation cadence for console and TensorBoard projections.

    ``steps`` is a required positive planned-step interval with no schema
    default. It decides only how often already-canonical observations are
    projected to the rank-zero console and to TensorBoard. It never suppresses,
    samples, delays, or otherwise changes the canonical one-row-per-completed-
    step ``logging.jsonl`` stream, which remains the sole authoritative durable
    scalar record, and it carries no research meaning.
    """

    steps: int = Field(gt=0, strict=True)


class RunConfig(StrictConfigModel):
    name: str
    artifact_root: str
    output_dir: str | None = None
    collision_policy: Literal["fail", "timestamp"] = "fail"


class ProcessorConfig(StrictConfigModel):
    do_resize: bool = False
    max_raw_pixels: int = Field(gt=0)
    max_merged_visual_tokens: int = Field(gt=0)


class SpecialTokenEmbeddingGroupsConfig(StrictConfigModel):
    coordinate_tokens: Literal["default_coord_0_999"]
    wrapper_tokens: Literal["default_object_box_wrappers"]


class SpecialTokenEmbeddingsConfig(StrictConfigModel):
    groups: SpecialTokenEmbeddingGroupsConfig


class QwenRuntimePatchesConfig(StrictConfigModel):
    patch_embed_linearization: Literal["enabled", "disabled"] = "enabled"


class ModelConfig(StrictConfigModel):
    base_model: str
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"]
    fa2_branch_proof: Literal["every_forward", "first_micro_step", "disabled"] = (
        "first_micro_step"
    )
    logits_memory_budget_bytes: int = Field(gt=0)
    processor: ProcessorConfig
    special_token_embeddings: SpecialTokenEmbeddingsConfig
    runtime_patches: QwenRuntimePatchesConfig = Field(
        default_factory=QwenRuntimePatchesConfig
    )


class AdapterConfig(StrictConfigModel):
    type: Literal["dora"]
    seed_mode: (
        Literal["initialize_new", "load_existing", "warm_start_expand_dora"] | None
    ) = None
    path: str | None = None
    source_adapter_path: str | None = None
    repaired_embedding_payload_path: str | None = None
    target_towers: tuple[Literal["language", "aligner", "vision"], ...]
    target_modules: Literal["all_linear"]
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    bias: Literal["none", "all", "lora_only"] = "none"

    @field_validator("type", mode="before")
    @classmethod
    def _type_uses_public_v1_dora_name(cls, value: object) -> object:
        if value == "dlora":
            raise ValueError(
                "V1 uses adapter.type: dora; adapter.type: dlora is unsupported"
            )
        return value

    @field_validator("target_towers")
    @classmethod
    def _target_towers_not_empty(
        cls, value: tuple[Literal["language", "aligner", "vision"], ...]
    ) -> tuple[Literal["language", "aligner", "vision"], ...]:
        if not value:
            raise ValueError("adapter.target_towers must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("adapter.target_towers must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _seed_mode_contract(self) -> "AdapterConfig":
        seed_mode = self.seed_mode
        if seed_mode is None:
            if (
                self.source_adapter_path is not None
                or self.repaired_embedding_payload_path is not None
            ):
                raise ValueError(
                    "adapter source/payload seed paths require adapter.seed_mode: warm_start_expand_dora"
                )
            return self
        if seed_mode == "initialize_new":
            if self.path is not None:
                raise ValueError(
                    "adapter.seed_mode=initialize_new must not set adapter.path"
                )
            if (
                self.source_adapter_path is not None
                or self.repaired_embedding_payload_path is not None
            ):
                raise ValueError(
                    "adapter.seed_mode=initialize_new must not set warm-start source paths"
                )
            return self
        if seed_mode == "load_existing":
            if self.path is None:
                raise ValueError(
                    "adapter.seed_mode=load_existing requires adapter.path"
                )
            if (
                self.source_adapter_path is not None
                or self.repaired_embedding_payload_path is not None
            ):
                raise ValueError(
                    "adapter.seed_mode=load_existing must not set warm-start source paths"
                )
            return self
        if self.path is not None:
            raise ValueError(
                "adapter.seed_mode=warm_start_expand_dora must not set adapter.path"
            )
        if self.source_adapter_path is None:
            raise ValueError(
                "adapter.seed_mode=warm_start_expand_dora requires adapter.source_adapter_path"
            )
        if self.repaired_embedding_payload_path is None:
            raise ValueError(
                "adapter.seed_mode=warm_start_expand_dora requires adapter.repaired_embedding_payload_path"
            )
        return self


class DatasetSplitConfig(StrictConfigModel):
    path: str
    sample_limit: int | None = Field(default=None, gt=0)


class GeometryFlipsAugmentationConfig(StrictConfigModel):
    enabled: bool = False
    horizontal_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    vertical_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    @field_validator("horizontal_prob", "vertical_prob", mode="before")
    @classmethod
    def _probability_is_numeric(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("geometry flip probabilities must be numeric")
        return value


class TrainAugmentationConfig(StrictConfigModel):
    geometry_flips: GeometryFlipsAugmentationConfig = Field(
        default_factory=GeometryFlipsAugmentationConfig
    )


class DataAugmentationConfig(StrictConfigModel):
    train: TrainAugmentationConfig = Field(default_factory=TrainAugmentationConfig)


class DataConfig(StrictConfigModel):
    train: DatasetSplitConfig
    eval: DatasetSplitConfig | None = None
    train_order: Literal["source_order"] = "source_order"
    augmentation: DataAugmentationConfig = Field(default_factory=DataAugmentationConfig)


class TemplatePromptConfig(StrictConfigModel):
    system: str | None = None
    user: str


class TemplateConfig(StrictConfigModel):
    object_field_order: Literal["desc_first", "geometry_first"]
    object_ordering: Literal["source_order", "geo_sorted", "random"]
    assistant_format: Literal["object_box_closed"]
    prompt: TemplatePromptConfig


class PackingConfig(StrictConfigModel):
    global_max_length: int = Field(gt=0)
    policy: PackingPolicy = "source_order_next_fit"
    window_size: int | None = Field(default=None, gt=0, strict=True)
    lookahead: int | None = Field(default=None, gt=0, strict=True)
    seed: int = Field(default=0, strict=True)
    worker_count: int = Field(default=1, gt=0, strict=True)
    cursor_byte_budget: int = Field(default=65_536, gt=0, strict=True)
    max_packs_per_fragment: int | None = Field(default=None, gt=0, strict=True)
    fragment_item_budget: int = Field(default=1_024, gt=0, strict=True)
    fragment_byte_budget: int = Field(default=4_194_304, gt=0, strict=True)

    @model_validator(mode="after")
    def _policy_parameters_are_exact(self) -> "PackingConfig":
        if self.policy == "source_order_next_fit":
            if (
                self.window_size is not None
                or self.lookahead is not None
                or self.max_packs_per_fragment is not None
            ):
                raise ValueError(
                    "source_order_next_fit does not accept window, lookahead, "
                    "or fragment-pack parameters"
                )
            return self
        if self.policy == "window_binpack":
            if self.window_size is None:
                raise ValueError("window_binpack requires packing.window_size")
            if self.lookahead is not None or self.max_packs_per_fragment is not None:
                raise ValueError(
                    "window_binpack does not accept lookahead or fragment-pack parameters"
                )
            return self
        if self.window_size is not None:
            raise ValueError(
                "online_window_binpack does not accept packing.window_size"
            )
        if self.lookahead is None:
            raise ValueError("online_window_binpack requires packing.lookahead")
        if self.max_packs_per_fragment is None:
            raise ValueError(
                "online_window_binpack requires packing.max_packs_per_fragment"
            )
        return self


TokenTypeGroupName = Literal["desc_text", "schema", "coordinate", "eos"]
TokenTypeGateMode = Literal["enabled", "zero_weight_ablation"]

# Canonical ordered gate group tuple. Pinned to `src.losses.vocab.V1_TOKEN_TYPES`
# by an equality test; the constant is duplicated here (rather than imported at
# module scope) because `src.losses` imports this module.
CANONICAL_TOKEN_TYPE_GROUPS: tuple[TokenTypeGroupName, ...] = (
    "desc_text",
    "schema",
    "coordinate",
    "eos",
)
# The only supported (mode, weight) pairs for the protected token-type gate.
TOKEN_TYPE_GATE_MODE_WEIGHTS: dict[str, float] = {
    "enabled": 0.1,
    "zero_weight_ablation": 0.0,
}
BASE_CE_WEIGHT = 1.0


class WeightedLossConfig(StrictConfigModel):
    weight: float = Field(ge=0.0, allow_inf_nan=False)


class BaseCELossConfig(WeightedLossConfig):
    @model_validator(mode="after")
    def _weight_is_exactly_one(self) -> "BaseCELossConfig":
        if self.weight != BASE_CE_WEIGHT:
            raise ValueError(
                "losses.protected.base_ce.weight must be exactly 1.0; "
                f"got {self.weight}. Base CE is a protected term and cannot be "
                "reweighted or omitted."
            )
        return self


class TokenTypeGateLossConfig(WeightedLossConfig):
    mode: TokenTypeGateMode
    groups: tuple[TokenTypeGroupName, ...]

    @model_validator(mode="before")
    @classmethod
    def _mode_is_authored_explicitly(cls, value: Any) -> Any:
        if isinstance(value, dict) and "mode" not in value:
            raise ValueError(
                "losses.protected.token_type_gate.mode is required: author "
                "mode: enabled with weight 0.1, or mode: zero_weight_ablation "
                "with weight 0.0. A gate weight without a mode identity is no "
                "longer accepted."
            )
        return value

    @model_validator(mode="after")
    def _mode_pairs_with_weight_and_canonical_groups(
        self,
    ) -> "TokenTypeGateLossConfig":
        from src.losses.vocab import V1_TOKEN_TYPES

        canonical = tuple(V1_TOKEN_TYPES)
        if tuple(self.groups) != canonical:
            raise ValueError(
                "losses.protected.token_type_gate.groups must be exactly the "
                f"ordered tuple {list(canonical)}; got {list(self.groups)}. "
                "Omitting, duplicating, adding, or reordering a group is not "
                "supported."
            )
        expected_weight = TOKEN_TYPE_GATE_MODE_WEIGHTS[self.mode]
        if self.weight != expected_weight:
            raise ValueError(
                f"losses.protected.token_type_gate.mode: {self.mode} pairs only "
                f"with weight exactly {expected_weight}; got {self.weight}. "
                "Use mode: enabled with weight 0.1 or mode: zero_weight_ablation "
                "with weight 0.0."
            )
        return self


class CoordGaussianRPSLossConfig(WeightedLossConfig):
    weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    gaussian_weight: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    rps_weight: float = Field(default=0.2, ge=0.0, allow_inf_nan=False)
    temperature: float = Field(default=1.0, ge=1.0e-6, allow_inf_nan=False)
    gaussian_r95_axis_fraction: float = Field(
        default=0.04,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    gaussian_r95_cap_bins: int = Field(default=8, ge=0, le=999)
    gaussian_r95_min_bins: int = Field(default=1, ge=0, le=999)
    gaussian_r95_fallback_bins: int = Field(default=8, ge=0, le=999)

    @model_validator(mode="after")
    def _enabled_term_has_differentiable_weight(self) -> "CoordGaussianRPSLossConfig":
        if self.gaussian_r95_min_bins > self.gaussian_r95_cap_bins:
            raise ValueError(
                "coord_gaussian_rps.gaussian_r95_min_bins must be <= gaussian_r95_cap_bins"
            )
        if self.gaussian_r95_fallback_bins > self.gaussian_r95_cap_bins:
            raise ValueError(
                "coord_gaussian_rps.gaussian_r95_fallback_bins must be <= gaussian_r95_cap_bins"
            )
        if self.weight > 0.0 and self.gaussian_weight == 0.0 and self.rps_weight == 0.0:
            raise ValueError(
                "coord_gaussian_rps.weight > 0 requires gaussian_weight or rps_weight > 0"
            )
        return self


class ProtectedLossesConfig(StrictConfigModel):
    base_ce: BaseCELossConfig
    token_type_gate: TokenTypeGateLossConfig

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_coordinate_placement(cls, value: Any) -> Any:
        if isinstance(value, dict) and "coord_gaussian_rps" in value:
            raise ValueError(
                "coord_gaussian_rps is not a protected loss: move the block to "
                "losses.auxiliary.coord_gaussian_rps. There is no compatibility "
                "alias for losses.protected.coord_gaussian_rps."
            )
        return value


class AuxiliaryLossesConfig(StrictConfigModel):
    coord_gaussian_rps: CoordGaussianRPSLossConfig | None = None


class LossesConfig(StrictConfigModel):
    normalizer: Literal["segment_balanced"]
    protected: ProtectedLossesConfig
    auxiliary: AuxiliaryLossesConfig | None = None


class OptimizerGroupConfig(StrictConfigModel):
    lr: float = Field(gt=0.0, allow_inf_nan=False)
    weight_decay: float = Field(ge=0.0, allow_inf_nan=False)


class AdapterOptimizerGroupsConfig(StrictConfigModel):
    language: OptimizerGroupConfig | None = None
    aligner: OptimizerGroupConfig | None = None
    vision: OptimizerGroupConfig | None = None


class OptimizerGroupsConfig(StrictConfigModel):
    adapters: AdapterOptimizerGroupsConfig
    token_embeddings: OptimizerGroupConfig


class SchedulerConfig(StrictConfigModel):
    name: Literal["cosine_with_warmup"]
    warmup_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    warmup_steps: int | None = Field(default=None, ge=0)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_warmup_surface(self) -> "SchedulerConfig":
        if (self.warmup_ratio is None) == (self.warmup_steps is None):
            raise ValueError("exactly one of warmup_ratio or warmup_steps must be set")
        return self


class OptimizerConfig(StrictConfigModel):
    name: Literal["adamw_torch"]
    betas: tuple[float, float]
    epsilon: float = Field(gt=0.0, allow_inf_nan=False)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    groups: OptimizerGroupsConfig
    scheduler: SchedulerConfig

    @field_validator("betas")
    @classmethod
    def _betas_are_finite(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(not math.isfinite(beta) for beta in value):
            raise ValueError("optimizer betas must be finite")
        return value


class TrainingConfig(StrictConfigModel):
    mode: Literal["supervised"]
    epochs: int = Field(gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    effective_batch_size: int = Field(gt=0)
    precision: Literal["bf16", "fp16"]
    max_grad_norm: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    forward_input_provider_mode: ForwardInputProviderMode = "synchronous"


class RuntimeDeterminismConfig(StrictConfigModel):
    mode: Literal["legacy", "strict_cuda_replay_v1"] = "legacy"


class RuntimeConfig(StrictConfigModel):
    seed: int = 17
    determinism: RuntimeDeterminismConfig = Field(
        default_factory=RuntimeDeterminismConfig
    )


class EvalForwardConfig(CadenceConfig):
    pass


class EvalInferenceConfig(StrictConfigModel):
    enabled: bool = False


class EvalConfig(StrictConfigModel):
    forward: EvalForwardConfig
    inference: EvalInferenceConfig


class CheckpointConfig(CadenceConfig):
    save_final: bool = True


class ResumeConfig(StrictConfigModel):
    mode: Literal["disabled", "exact_same_world_size"] = "disabled"
    checkpoint_dir: str | None = None

    @model_validator(mode="after")
    def _disabled_mode_has_no_checkpoint(self) -> "ResumeConfig":
        if self.mode == "disabled" and self.checkpoint_dir is not None:
            raise ValueError(
                "resume.checkpoint_dir requires resume.mode: exact_same_world_size"
            )
        return self


class TrainConfig(StrictConfigModel):
    schema_version: Literal[1]
    run: RunConfig
    model: ModelConfig
    adapter: AdapterConfig
    data: DataConfig
    template: TemplateConfig
    packing: PackingConfig
    losses: LossesConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig
    observability: ObservabilityConfig
    resume: ResumeConfig = Field(default_factory=ResumeConfig)

    @model_validator(mode="after")
    def _exact_resume_requires_strict_cuda_replay(self) -> "TrainConfig":
        if (
            self.resume.mode == "exact_same_world_size"
            and self.runtime.determinism.mode != "strict_cuda_replay_v1"
        ):
            raise ValueError(
                "resume.mode=exact_same_world_size requires "
                "runtime.determinism.mode=strict_cuda_replay_v1"
            )
        return self


@dataclass(frozen=True)
class ConfigSource:
    path: Path
    sha256: str

    def to_artifact_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class PathOrigin:
    field: str
    declared_path: str
    declaring_config_path: Path
    resolved_path: Path

    def to_artifact_dict(self) -> dict[str, str]:
        return {
            "declared_path": self.declared_path,
            "declaring_config_path": str(self.declaring_config_path),
            "resolved_path": str(self.resolved_path),
        }


@dataclass(frozen=True)
class ResolvedTrainConfig:
    config: TrainConfig
    config_dict: dict[str, Any]
    fingerprint: str
    schema_version: int
    loader_version: str
    entry_config_path: Path
    sources: tuple[ConfigSource, ...]
    path_origins: dict[str, PathOrigin]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "config": self.config_dict,
            "resolution": {
                "schema_version": self.schema_version,
                "loader_version": self.loader_version,
                "fingerprint": self.fingerprint,
                "entry_config_path": str(self.entry_config_path),
                "sources": [source.to_artifact_dict() for source in self.sources],
                "path_origins": {
                    field: origin.to_artifact_dict()
                    for field, origin in sorted(self.path_origins.items())
                },
            },
        }


@dataclass(frozen=True)
class ResolvedConfigArtifacts:
    yaml_path: Path
    json_path: Path
    fingerprint: str


@dataclass(frozen=True)
class RunDirectory:
    run_name: str
    artifact_root: Path
    run_dir: Path
    collision_policy: str


@dataclass(frozen=True)
class RuntimeBatchResolution:
    world_size: int
    effective_batch_size: int
    resolved_grad_accum_steps: int

    def to_artifact_dict(self) -> dict[str, int]:
        return {
            "world_size": self.world_size,
            "effective_batch_size": self.effective_batch_size,
            "resolved_grad_accum_steps": self.resolved_grad_accum_steps,
        }


@dataclass(frozen=True)
class QwenRuntimeControls:
    attn_implementation: str
    compute_precision: str
    model_logits_dtype: str
    global_max_length: int
    tokenizer_vocab_size: int
    estimated_logits_bytes: int
    logits_memory_budget_bytes: int

    def to_artifact_dict(self) -> dict[str, int | str]:
        return {
            "attn_implementation": self.attn_implementation,
            "compute_precision": self.compute_precision,
            "model_logits_dtype": self.model_logits_dtype,
            "global_max_length": self.global_max_length,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "estimated_logits_bytes": self.estimated_logits_bytes,
            "logits_memory_budget_bytes": self.logits_memory_budget_bytes,
        }
