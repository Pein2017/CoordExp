"""Typed schema contracts for `rollout_matching.*`.

This module defines the YAML-visible rollout-matching configuration surface as a
set of dataclasses. It is used for schema-derived strict parsing (unknown-key
fail-fast with dotted paths) at config-load time.

Notes:
- This is intentionally dependency-light (no torch/vllm imports).
- Runtime-only validations (world-size, backend readiness, etc.) remain in the
  trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

from src.trainers.teacher_forcing.module_registry import (
    ALLOWED_DIAGNOSTIC_MODULES,
    ALLOWED_OBJECTIVE_MODULES,
    DIAGNOSTIC_CONFIG_ALLOWLIST,
    OBJECTIVE_APPLICATION_PRESET_ALLOWLIST,
    OBJECTIVE_CONFIG_ALLOWLIST,
    OBJECTIVE_OPTIONAL_CONFIG_KEYS,
    normalize_token_ce_stop_signal_damping_config,
)


@dataclass(frozen=True)
class RolloutDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass(frozen=True)
class RolloutOffloadConfig:
    enabled: bool = False
    # Missing values default at runtime:
    # - enabled=true  -> offload_model/offload_optimizer default to true
    # - enabled=false -> both default to false
    offload_model: Optional[bool] = None
    offload_optimizer: Optional[bool] = None


@dataclass(frozen=True)
class RolloutMonitorDumpConfig:
    enabled: bool = False
    every_steps: Optional[int] = None
    every_channel_b_steps: Optional[int] = None
    dump_first_step: Optional[bool] = None
    only_world_process_zero: bool = True
    max_events: int = 20
    max_samples: int = 1
    max_text_chars: int = 4000
    # Safety rails: dumps are diagnostics only and should not destabilize training.
    # - async_write avoids blocking the training step on slow filesystems.
    # - min_free_gb skips dumps when disk is low (prevents "disk full" surprises).
    # - max_pending_writes bounds in-flight async dump tasks to avoid memory growth.
    async_write: bool = True
    max_pending_writes: int = 2
    min_free_gb: float = 2.0
    out_dir: Optional[str] = None
    write_markdown: bool = True

    def __post_init__(self) -> None:
        def _validate_positive_optional_int(raw: Any, *, field_name: str) -> None:
            if raw is None:
                return
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"rollout_matching.train_monitor_dump.{field_name} must be an int"
                ) from exc
            if value <= 0:
                raise ValueError(
                    f"rollout_matching.train_monitor_dump.{field_name} must be > 0"
                )

        _validate_positive_optional_int(self.every_steps, field_name="every_steps")
        _validate_positive_optional_int(
            self.every_channel_b_steps, field_name="every_channel_b_steps"
        )


@dataclass(frozen=True)
class RolloutEvalMonitorDumpConfig:
    enabled: bool = False
    every_evals: int = 1
    only_world_process_zero: bool = True
    max_events: int = 20
    max_samples: int = 1
    max_text_chars: int = 4000
    async_write: bool = True
    max_pending_writes: int = 2
    min_free_gb: float = 2.0
    out_dir: Optional[str] = None
    write_markdown: bool = True


@dataclass(frozen=True)
class RolloutDescMonitorConfig:
    enabled: bool = False
    every_steps: int = 0
    max_pairs: int = 64
    semantic_threshold: float = 0.6
    mode: str = "semantic"  # semantic|both|exact

    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_device: str = "cpu"
    semantic_batch_size: int = 64
    semantic_max_length: int = 64


@dataclass(frozen=True)
class RolloutEvalConfidencePostOpConfig:
    """Confidence post-op knobs reused inside trainer eval_step.

    Mirrors `src.eval.confidence_postop.ConfidencePostOpOptions` defaults.
    """

    fusion_w_geom: float = 0.7
    fusion_w_desc: float = 0.3
    desc_span_policy: str = "best_effort"  # best_effort|strict
    empty_desc_policy: str = "geom_only"  # geom_only|drop


@dataclass(frozen=True)
class RolloutEvalDetectionConfig:
    # Enable official detection AP during trainer eval_step.
    enabled: bool = True
    metrics: str = (
        "coco"  # coco | lvis | both (both includes evaluator f1-ish in addition to official AP)
    )

    # Official evaluator knobs.
    use_segm: bool = False
    strict_parse: bool = True
    iou_thrs: Optional[list[float]] = None
    lvis_max_dets: int = 300

    # Semantic mapping for open-vocab desc -> dataset categories.
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_threshold: float = 0.6
    semantic_device: str = "auto"
    semantic_batch_size: int = 64

    # Optional f1-ish knobs (used only when metrics=both).
    f1ish_iou_thrs: list[float] = field(default_factory=lambda: [0.3, 0.5])
    f1ish_pred_scope: str = "annotated"  # annotated | all

    # Score provenance attached to eval-step prediction records for COCO contract.
    pred_score_source: str = "eval_rollout_constant"
    pred_score_version: int = 2

    # Score policy used inside trainer eval-step.
    score_mode: str = "constant"  # constant | confidence_postop
    constant_score: float = 1.0
    confidence: RolloutEvalConfidencePostOpConfig = field(
        default_factory=RolloutEvalConfidencePostOpConfig
    )


@dataclass(frozen=True)
class VllmServerDebugDumpConfig:
    enabled: bool = False
    every_steps: Optional[int] = None
    dump_first_step: Optional[bool] = None
    only_world_process_zero: bool = True
    max_events: int = 3
    max_samples: int = 1
    max_chars: int = 4000
    async_write: bool = True
    max_pending_writes: int = 2
    min_free_gb: float = 2.0
    out_dir: Optional[str] = None


@dataclass(frozen=True)
class VllmServerEntryConfig:
    base_url: str
    group_port: int

    def __post_init__(self) -> None:
        base_url = str(self.base_url or "").strip()
        if not base_url:
            raise ValueError(
                "rollout_matching.vllm.server.servers[*].base_url must be non-empty"
            )

        parsed = urlparse(base_url)
        scheme = str(parsed.scheme or "").strip().lower()
        host = str(parsed.hostname or "").strip().lower()

        if scheme not in {"http", "https"}:
            raise ValueError(
                "rollout_matching.vllm.server.servers[*].base_url must use http:// or https://"
            )
        if not host:
            raise ValueError(
                "rollout_matching.vllm.server.servers[*].base_url must include a hostname"
            )
        if host == "0.0.0.0":
            raise ValueError(
                "rollout_matching.vllm.server.servers[*].base_url must not use 0.0.0.0; "
                "use 127.0.0.1 or a routable host/IP instead"
            )

        try:
            group_port = int(self.group_port)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.vllm.server.servers[*].group_port must be an int"
            ) from exc
        if group_port <= 0:
            raise ValueError(
                "rollout_matching.vllm.server.servers[*].group_port must be > 0"
            )


@dataclass(frozen=True)
class VllmServerConfig:
    timeout_s: float = 240.0
    infer_timeout_s: Optional[float] = None
    allow_infinite_infer_timeout: bool = False
    servers: list[VllmServerEntryConfig] = field(default_factory=list)
    debug_dump: VllmServerDebugDumpConfig = field(
        default_factory=VllmServerDebugDumpConfig
    )


@dataclass(frozen=True)
class VllmSyncConfig:
    mode: str = "full"  # full (only supported mode in this stack)
    # Deprecated/ignored: kept for backwards-compatible config parsing only.
    fallback_to_full: bool = True


@dataclass(frozen=True)
class VllmConfig:
    # server|colocate
    mode: Optional[str] = None

    tensor_parallel_size: Optional[int] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.45

    enable_lora: bool = False
    load_format: Optional[str] = None

    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = True

    # Rebuild colocate vLLM engine at each eval-step window boundary.
    reinit_each_eval: bool = False

    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None

    distributed_executor_backend: Optional[str] = None

    # Optional micro-batching for colocate /infer.
    infer_batch_size: Optional[int] = None

    # Optional vLLM EngineArgs passthrough (kept explicit to catch typos).
    enable_chunked_prefill: Optional[bool] = None
    disable_chunked_mm_input: Optional[bool] = None
    kv_cache_dtype: Optional[str] = None
    cpu_offload_gb: Optional[float] = None
    swap_space: Optional[float] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    mm_encoder_tp_mode: Optional[str] = None
    skip_mm_profiling: Optional[bool] = None

    # Optional HF multimodal processor kwargs forwarded into vLLM.
    # Use this to keep server-side prompt tokenization consistent with teacher-forcing
    # encode (e.g., disable smart-resize when images are already offline-rescaled).
    mm_processor_kwargs: Optional[Mapping[str, Any]] = None

    server: Optional[VllmServerConfig] = None
    sync: Optional[VllmSyncConfig] = None


@dataclass(frozen=True)
class RolloutPipelineModuleSpec:
    name: str
    enabled: bool
    weight: float
    channels: tuple[str, ...]
    config: Mapping[str, Any]
    application: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutPipelineConfig:
    objective: tuple[RolloutPipelineModuleSpec, ...] = field(default_factory=tuple)
    diagnostics: tuple[RolloutPipelineModuleSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RolloutMatchingConfig:
    # Core backend selection.
    rollout_backend: str = "hf"
    eval_rollout_backend: str = "vllm"

    # Decode / generation.
    # Explicit per-context rollout decode batch sizes (required).
    channel_b_decode_batch_size: Optional[int] = None
    eval_decode_batch_size: Optional[int] = None
    decode_mode: str = "greedy"
    max_new_tokens: int = 512
    num_beams: int = 1
    num_return_sequences: Optional[int] = None
    repetition_penalty: float = 1.0

    decoding: Optional[RolloutDecodingConfig] = None
    # Matching knobs.
    candidate_top_k: int = 10
    maskiou_gate: float = 0.3
    maskiou_resolution: int = 256
    fp_cost: float = 1.0
    fn_cost: float = 1.0

    ot_cost: str = "l2"
    ot_epsilon: float = 10.0
    ot_iters: int = 30

    # Nested namespaces.
    offload: Optional[RolloutOffloadConfig] = None
    # Legacy shared namespace retained as a fallback for older configs.
    monitor_dump: Optional[RolloutMonitorDumpConfig] = None
    train_monitor_dump: Optional[RolloutMonitorDumpConfig] = None
    eval_monitor_dump: Optional[RolloutEvalMonitorDumpConfig] = None
    desc_monitor: Optional[RolloutDescMonitorConfig] = None
    pipeline: Optional[RolloutPipelineConfig] = None
    eval_detection: RolloutEvalDetectionConfig = field(
        default_factory=RolloutEvalDetectionConfig
    )
    vllm: Optional[VllmConfig] = None
    # Optional override applied only to eval-step rollouts.
    eval_prompt_variant: Optional[str] = None

    # Optional coord-loss override knobs (advanced; default to custom.coord_soft_ce_w1).
    target_sigma: Optional[float] = None
    target_truncate: Optional[int] = None
    temperature_coord: Optional[float] = None
    soft_ce_weight: Optional[float] = None
    w1_weight: Optional[float] = None
    gate_weight: Optional[float] = None

    def __post_init__(self) -> None:
        # Lightweight semantic checks that do not require runtime state.
        def _normalize_backend(
            raw: Any, *, field_name: str, allow_none: bool
        ) -> Optional[str]:
            if raw is None:
                if allow_none:
                    return None
                raise ValueError(
                    f"rollout_matching.{field_name} must be one of {{hf,vllm}}"
                )
            value = str(raw).strip().lower()
            if allow_none and value in {"", "null", "none"}:
                return None
            if value not in {"hf", "vllm"}:
                allowed = "{null,hf,vllm}" if allow_none else "{hf,vllm}"
                raise ValueError(
                    f"rollout_matching.{field_name} must be one of {allowed}; got {raw!r}"
                )
            return value

        rollout_backend = _normalize_backend(
            self.rollout_backend,
            field_name="rollout_backend",
            allow_none=False,
        )
        eval_backend = _normalize_backend(
            self.eval_rollout_backend,
            field_name="eval_rollout_backend",
            allow_none=False,
        )
        effective_eval_backend = eval_backend

        if self.channel_b_decode_batch_size is None:
            raise ValueError(
                "rollout_matching.channel_b_decode_batch_size must be provided explicitly"
            )
        try:
            channel_b_decode_bs = int(self.channel_b_decode_batch_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.channel_b_decode_batch_size must be an int"
            ) from exc
        if channel_b_decode_bs <= 0:
            raise ValueError("rollout_matching.channel_b_decode_batch_size must be > 0")

        if self.eval_decode_batch_size is None:
            raise ValueError(
                "rollout_matching.eval_decode_batch_size must be provided explicitly"
            )
        try:
            eval_decode_bs = int(self.eval_decode_batch_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout_matching.eval_decode_batch_size must be an int"
            ) from exc
        if eval_decode_bs <= 0:
            raise ValueError("rollout_matching.eval_decode_batch_size must be > 0")

        if self.decoding is not None:
            dec = self.decoding
            try:
                temperature = float(getattr(dec, "temperature", 0.0) or 0.0)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.decoding.temperature must be a float"
                ) from exc
            if temperature < 0.0:
                raise ValueError("rollout_matching.decoding.temperature must be >= 0")

            try:
                top_p = float(
                    getattr(dec, "top_p", 1.0) if dec.top_p is not None else 1.0
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.decoding.top_p must be a float"
                ) from exc
            if not (0.0 < top_p <= 1.0):
                raise ValueError("rollout_matching.decoding.top_p must be in (0, 1]")

            try:
                top_k = int(getattr(dec, "top_k", -1))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.decoding.top_k must be an int"
                ) from exc
            if top_k != -1 and top_k < 1:
                raise ValueError(
                    "rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
                )

        if self.vllm is not None and self.vllm.sync is not None:
            mode = str(self.vllm.sync.mode or "full").strip().lower()
            if mode != "full":
                raise ValueError(
                    "rollout_matching.vllm.sync.mode must be 'full' in this stack "
                    "(adapter/auto sync modes are unsupported)."
                )

        if (
            rollout_backend == "vllm" or effective_eval_backend == "vllm"
        ) and self.vllm is not None:
            if bool(getattr(self.vllm, "enable_lora", False)):
                raise ValueError(
                    "vLLM rollouts require full merged-weight sync in this stack: "
                    "set rollout_matching.vllm.enable_lora=false."
                )

        if self.pipeline is not None:
            if not isinstance(self.pipeline, RolloutPipelineConfig):
                raise TypeError(
                    "rollout_matching.pipeline must be a RolloutPipelineConfig"
                )

            if not self.pipeline.objective:
                raise ValueError(
                    "rollout_matching.pipeline.objective must be non-empty"
                )

            def _validate_specs(
                specs: tuple[RolloutPipelineModuleSpec, ...],
                *,
                allowed_names: set[str],
                config_allowlist_by_name: Mapping[str, set[str]],
                path: str,
            ) -> None:
                seen: set[str] = set()
                for idx, spec in enumerate(specs):
                    if not isinstance(spec, RolloutPipelineModuleSpec):
                        raise TypeError(
                            f"{path}[{idx}] must be RolloutPipelineModuleSpec"
                        )
                    name = str(spec.name or "").strip()
                    if not name:
                        raise ValueError(f"{path}[{idx}].name must be non-empty")
                    if name not in allowed_names:
                        raise ValueError(
                            f"{path}[{idx}].name must be one of {sorted(allowed_names)}; got {name!r}"
                        )
                    if name in seen:
                        raise ValueError(f"Duplicate module name in {path}: {name}")
                    seen.add(name)

                    try:
                        weight = float(spec.weight)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            f"{path}[{idx}].weight must be numeric"
                        ) from exc
                    if weight < 0.0:
                        raise ValueError(f"{path}[{idx}].weight must be >= 0")

                    if not isinstance(spec.channels, Sequence) or isinstance(
                        spec.channels, (str, bytes)
                    ):
                        raise TypeError(f"{path}[{idx}].channels must be a sequence")
                    if not spec.channels:
                        raise ValueError(f"{path}[{idx}].channels must not be empty")
                    for cidx, ch in enumerate(spec.channels):
                        ch_s = str(ch).strip().upper()
                        if ch_s not in {"A", "B"}:
                            raise ValueError(
                                f"{path}[{idx}].channels[{cidx}] must be 'A' or 'B'"
                            )

                    if not isinstance(spec.config, Mapping):
                        raise TypeError(f"{path}[{idx}].config must be a mapping")

                    if path.endswith(".objective"):
                        if not isinstance(spec.application, Mapping):
                            raise TypeError(
                                f"{path}[{idx}].application must be a mapping"
                            )
                        app_unknown = set(spec.application.keys()) - {"preset"}
                        if app_unknown:
                            raise ValueError(
                                f"Unknown {path}[{idx}].application keys for module {name!r}: "
                                f"{sorted(str(k) for k in app_unknown)}"
                            )
                        preset = str(spec.application.get("preset", "") or "").strip()
                        if not preset:
                            raise ValueError(
                                f"{path}[{idx}].application.preset must be provided"
                            )
                        allowed_presets = OBJECTIVE_APPLICATION_PRESET_ALLOWLIST.get(
                            name, set()
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
                                    if name == "token_ce"
                                    else "anchor_only"
                                )
                                raise ValueError(
                                    f"{path}[{idx}].application.preset for module {name!r} "
                                    f"uses deprecated self-context-era routing {preset!r}. "
                                    f"Use {replacement!r} for the single-pass contract."
                                )
                            raise ValueError(
                                f"{path}[{idx}].application.preset for module {name!r} "
                                f"must be one of {sorted(str(x) for x in allowed_presets)}; got {preset!r}"
                            )

                        if name == "token_ce" and "stop_signal_damping" in spec.config:
                            normalize_token_ce_stop_signal_damping_config(
                                spec.config.get("stop_signal_damping"),
                                path=f"{path}[{idx}].config.stop_signal_damping",
                            )
                        if name == "token_ce" and "struct_ce_weight" in spec.config:
                            raise ValueError(
                                f"{path}[{idx}].config.struct_ce_weight is deprecated and unsupported. "
                                "Remove the self-context struct/EOS stabilizer; active training uses "
                                "only the single-pass anchor_text_only contract."
                            )
                    allowed_cfg = config_allowlist_by_name.get(name, set())
                    unknown_cfg = set(spec.config.keys()) - set(allowed_cfg)
                    if unknown_cfg:
                        raise ValueError(
                            f"Unknown {path}[{idx}].config keys for module {name!r}: "
                            f"{sorted(str(k) for k in unknown_cfg)}"
                        )
                    optional_cfg = OBJECTIVE_OPTIONAL_CONFIG_KEYS.get(name, set())
                    missing_cfg = set(allowed_cfg) - set(spec.config.keys()) - set(optional_cfg)
                    if missing_cfg:
                        raise ValueError(
                            f"Missing required {path}[{idx}].config keys for module {name!r}: "
                            f"{sorted(str(k) for k in missing_cfg)}"
                        )

            _validate_specs(
                self.pipeline.objective,
                allowed_names=ALLOWED_OBJECTIVE_MODULES,
                config_allowlist_by_name=OBJECTIVE_CONFIG_ALLOWLIST,
                path="rollout_matching.pipeline.objective",
            )
            _validate_specs(
                self.pipeline.diagnostics,
                allowed_names=ALLOWED_DIAGNOSTIC_MODULES,
                config_allowlist_by_name=DIAGNOSTIC_CONFIG_ALLOWLIST,
                path="rollout_matching.pipeline.diagnostics",
            )

            obj_by_name = {str(spec.name): spec for spec in self.pipeline.objective}
            bbox_geo = obj_by_name.get("bbox_geo")
            bbox_size_aux = obj_by_name.get("bbox_size_aux")
            coord_reg = obj_by_name.get("coord_reg")
            if bbox_size_aux is not None and bool(getattr(bbox_size_aux, "enabled", False)):
                if bbox_geo is None or not bool(getattr(bbox_geo, "enabled", False)):
                    raise ValueError(
                        "rollout_matching.pipeline.objective requires bbox_geo to be present+enabled when bbox_size_aux is enabled "
                        "(bbox_size_aux depends on bbox_geo state)."
                    )
                missing_channels = set(bbox_size_aux.channels) - set(bbox_geo.channels)
                if missing_channels:
                    raise ValueError(
                        "rollout_matching.pipeline.objective bbox_size_aux channels must be a subset of bbox_geo channels; "
                        f"missing={sorted(missing_channels)}"
                    )
            if coord_reg is not None and bool(getattr(coord_reg, "enabled", False)):
                if bbox_geo is None or not bool(getattr(bbox_geo, "enabled", False)):
                    raise ValueError(
                        "rollout_matching.pipeline.objective requires bbox_geo to be present+enabled when coord_reg is enabled "
                        "(coord_reg depends on bbox_geo state)."
                    )
                missing_channels = set(coord_reg.channels) - set(bbox_geo.channels)
                if missing_channels:
                    raise ValueError(
                        "rollout_matching.pipeline.objective coord_reg channels must be a subset of bbox_geo channels; "
                        f"missing={sorted(missing_channels)}"
                    )

            for dspec in self.pipeline.diagnostics:
                if not bool(getattr(dspec, "enabled", False)):
                    continue
                if str(getattr(dspec, "name", "") or "") != "coord_diag":
                    continue
                if bbox_geo is None or not bool(getattr(bbox_geo, "enabled", False)):
                    raise ValueError(
                        "rollout_matching.pipeline.diagnostics requires bbox_geo to be present+enabled when coord_diag is enabled "
                        "(coord_diag depends on bbox_geo state)."
                    )
                missing_channels = set(dspec.channels) - set(bbox_geo.channels)
                if missing_channels:
                    raise ValueError(
                        "rollout_matching.pipeline.diagnostics coord_diag channels must be a subset of bbox_geo channels; "
                        f"missing={sorted(missing_channels)}"
                    )

        if self.eval_prompt_variant is not None and not isinstance(
            self.eval_prompt_variant, str
        ):
            raise TypeError(
                "rollout_matching.eval_prompt_variant must be a string when provided"
            )

        if self.eval_detection is not None:
            eval_det = self.eval_detection
            metrics = (
                str(getattr(eval_det, "metrics", "coco") or "coco").strip().lower()
            )
            if metrics not in {"coco", "lvis", "both", "f1ish"}:
                raise ValueError(
                    "rollout_matching.eval_detection.metrics must be one of {'coco', 'lvis', 'both', 'f1ish'}"
                )

            score_mode = str(getattr(eval_det, "score_mode", "constant") or "constant")
            score_mode = score_mode.strip().lower()
            if score_mode not in {"constant", "confidence_postop"}:
                raise ValueError(
                    "rollout_matching.eval_detection.score_mode must be one of {'constant', 'confidence_postop'}"
                )

            try:
                const_score = float(getattr(eval_det, "constant_score", 1.0))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.eval_detection.constant_score must be numeric"
                ) from exc
            if const_score < 0.0 or const_score > 1.0:
                raise ValueError(
                    "rollout_matching.eval_detection.constant_score must satisfy 0.0 <= score <= 1.0"
                )

            source = str(getattr(eval_det, "pred_score_source", "") or "").strip()
            if not source:
                raise ValueError(
                    "rollout_matching.eval_detection.pred_score_source must be non-empty"
                )

            try:
                int(getattr(eval_det, "pred_score_version", 0))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.eval_detection.pred_score_version must be int-compatible"
                ) from exc
