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


@dataclass(frozen=True)
class RolloutDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass(frozen=True)
class RolloutOffloadConfig:
    enabled: bool = False
    offload_model: bool = False
    offload_optimizer: bool = False


@dataclass(frozen=True)
class RolloutMonitorDumpConfig:
    enabled: bool = False
    every_steps: Optional[int] = None
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
    # Enable COCO-style AP/mAP during trainer eval_step.
    enabled: bool = True
    metrics: str = "coco"  # coco | both (both includes evaluator f1-ish in addition to COCO)

    # COCO evaluator knobs.
    use_segm: bool = False
    strict_parse: bool = True
    iou_thrs: Optional[list[float]] = None

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


@dataclass(frozen=True)
class VllmServerConfig:
    timeout_s: float = 240.0
    infer_timeout_s: Optional[float] = None
    servers: list[VllmServerEntryConfig] = field(default_factory=list)
    debug_dump: VllmServerDebugDumpConfig = field(default_factory=VllmServerDebugDumpConfig)


@dataclass(frozen=True)
class VllmSyncConfig:
    mode: str = "full"  # full|adapter|auto
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
    sleep_level: int = 0
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = True

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

    server: Optional[VllmServerConfig] = None
    sync: Optional[VllmSyncConfig] = None


_ALLOWED_ROLLOUT_OBJECTIVE_MODULES: set[str] = {"token_ce", "bbox_geo", "coord_reg"}
_ALLOWED_ROLLOUT_DIAGNOSTIC_MODULES: set[str] = {"coord_diag"}


@dataclass(frozen=True)
class RolloutPipelineModuleSpec:
    name: str
    enabled: bool = True
    weight: float = 1.0
    channels: tuple[str, ...] = ("A", "B")
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutPipelineConfig:
    objective: tuple[RolloutPipelineModuleSpec, ...] = field(default_factory=tuple)
    diagnostics: tuple[RolloutPipelineModuleSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RolloutMatchingConfig:
    # Core backend selection.
    rollout_backend: str = "vllm"

    # Decode / generation.
    decode_batch_size: int = 1
    decode_mode: str = "greedy"
    max_new_tokens: int = 512
    num_beams: int = 1
    num_return_sequences: Optional[int] = None
    repetition_penalty: float = 1.0

    decoding: Optional[RolloutDecodingConfig] = None
    coord_decode_mode: str = "exp"  # exp|st
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
    monitor_dump: Optional[RolloutMonitorDumpConfig] = None
    desc_monitor: Optional[RolloutDescMonitorConfig] = None
    pipeline: Optional[RolloutPipelineConfig] = None
    eval_detection: RolloutEvalDetectionConfig = field(default_factory=RolloutEvalDetectionConfig)
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
        try:
            decode_bs = int(self.decode_batch_size)
        except (TypeError, ValueError) as exc:
            raise TypeError("rollout_matching.decode_batch_size must be an int") from exc
        if decode_bs <= 0:
            raise ValueError("rollout_matching.decode_batch_size must be > 0")

        coord_decode_mode = str(self.coord_decode_mode or "exp").strip().lower()
        if coord_decode_mode not in {"exp", "st"}:
            raise ValueError(
                "rollout_matching.coord_decode_mode must be one of {'exp', 'st'}"
            )

        if self.decoding is not None:
            dec = self.decoding
            try:
                temperature = float(getattr(dec, "temperature", 0.0) or 0.0)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "rollout_matching.decoding.temperature must be a float"
                ) from exc
            if temperature < 0.0:
                raise ValueError(
                    "rollout_matching.decoding.temperature must be >= 0"
                )

            try:
                top_p = float(getattr(dec, "top_p", 1.0) if dec.top_p is not None else 1.0)
            except (TypeError, ValueError) as exc:
                raise TypeError("rollout_matching.decoding.top_p must be a float") from exc
            if not (0.0 < top_p <= 1.0):
                raise ValueError("rollout_matching.decoding.top_p must be in (0, 1]")

            try:
                top_k = int(getattr(dec, "top_k", -1))
            except (TypeError, ValueError) as exc:
                raise TypeError("rollout_matching.decoding.top_k must be an int") from exc
            if top_k != -1 and top_k < 1:
                raise ValueError(
                    "rollout_matching.decoding.top_k must be -1 (disabled) or >= 1"
                )

        if self.vllm is not None and self.vllm.sync is not None:
            mode = str(self.vllm.sync.mode or "full").strip().lower()
            if mode not in {"full", "adapter", "auto"}:
                raise ValueError(
                    "rollout_matching.vllm.sync.mode must be one of: full|adapter|auto"
                )

        if self.pipeline is not None:
            if not isinstance(self.pipeline, RolloutPipelineConfig):
                raise TypeError("rollout_matching.pipeline must be a RolloutPipelineConfig")

            def _validate_specs(
                specs: tuple[RolloutPipelineModuleSpec, ...],
                *,
                allowed_names: set[str],
                path: str,
            ) -> None:
                seen: set[str] = set()
                for idx, spec in enumerate(specs):
                    if not isinstance(spec, RolloutPipelineModuleSpec):
                        raise TypeError(f"{path}[{idx}] must be RolloutPipelineModuleSpec")
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
                        raise TypeError(f"{path}[{idx}].weight must be numeric") from exc
                    if weight < 0.0:
                        raise ValueError(f"{path}[{idx}].weight must be >= 0")

                    if not isinstance(spec.channels, Sequence) or isinstance(spec.channels, (str, bytes)):
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

            _validate_specs(
                self.pipeline.objective,
                allowed_names=_ALLOWED_ROLLOUT_OBJECTIVE_MODULES,
                path="rollout_matching.pipeline.objective",
            )
            _validate_specs(
                self.pipeline.diagnostics,
                allowed_names=_ALLOWED_ROLLOUT_DIAGNOSTIC_MODULES,
                path="rollout_matching.pipeline.diagnostics",
            )

        if self.eval_prompt_variant is not None and not isinstance(
            self.eval_prompt_variant, str
        ):
            raise TypeError(
                "rollout_matching.eval_prompt_variant must be a string when provided"
            )

        if self.eval_detection is not None:
            eval_det = self.eval_detection
            metrics = str(getattr(eval_det, "metrics", "coco") or "coco").strip().lower()
            if metrics not in {"coco", "both"}:
                raise ValueError(
                    "rollout_matching.eval_detection.metrics must be one of {'coco', 'both'}"
                )

            score_mode = str(getattr(eval_det, "score_mode", "constant") or "constant")
            score_mode = score_mode.strip().lower()
            if score_mode != "constant":
                raise ValueError(
                    "rollout_matching.eval_detection.score_mode must be 'constant'"
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
