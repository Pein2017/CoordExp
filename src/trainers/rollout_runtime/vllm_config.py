from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class VllmEngineConfig:
    vcfg: Dict[str, Any]
    dist: Any
    world_size: int
    rank: int
    tp_size: int
    max_model_len: int
    enable_lora: bool
    load_format: str
    gpu_mem: float
    enable_prefix_caching: bool
    enable_sleep_mode: bool
    sleep_level: int
    enforce_eager: bool
    disable_custom_all_reduce: bool
    decode_bs_per_rank: int
    max_num_seqs: Optional[int]
    limit_mm_per_prompt: Optional[Dict[str, int]]
    vllm_engine_kwargs: Dict[str, Any]
    dist_backend: str


def resolve_vllm_engine_config(owner: Any) -> VllmEngineConfig:
    vcfg_raw = owner._cfg("vllm", {}) or {}
    if not isinstance(vcfg_raw, Mapping):
        raise ValueError("rollout_matching.vllm must be a mapping")
    vcfg = dict(vcfg_raw)

    try:
        import torch.distributed as dist
    except (TypeError, ValueError):
        dist = None  # type: ignore[assignment]

    world_size = 1
    rank = 0
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        rank = int(dist.get_rank())

    default_tp = 4 if world_size == 4 else 1
    tp_size = int(vcfg.get("tensor_parallel_size", default_tp))
    if tp_size <= 0:
        raise ValueError("rollout_matching.vllm.tensor_parallel_size must be > 0")
    if world_size % tp_size != 0:
        raise ValueError(
            f"vLLM colocate requires world_size % tp == 0; world_size={world_size} tp={tp_size}"
        )

    max_model_len_raw = vcfg.get("max_model_len", None)
    if max_model_len_raw is None:
        raise ValueError(
            "rollout_matching.vllm.max_model_len is required when rollout_backend=vllm "
            "(it must cover prompt_len + max_new_tokens)."
        )
    max_model_len = int(max_model_len_raw)
    if max_model_len <= 0:
        raise ValueError("rollout_matching.vllm.max_model_len must be > 0")

    enable_lora = bool(vcfg.get("enable_lora", False))
    if enable_lora:
        raise RuntimeError(
            "vLLM rollouts require full merged-weight sync in this stack: "
            "set rollout_matching.vllm.enable_lora=false."
        )

    load_format = vcfg.get("load_format", None)
    if load_format is None:
        load_format = "dummy"
    if not isinstance(load_format, str):
        raise ValueError("rollout_matching.vllm.load_format must be a string")
    load_format = load_format.strip()

    gpu_mem = float(vcfg.get("gpu_memory_utilization", 0.45))
    enable_prefix_caching = bool(vcfg.get("enable_prefix_caching", True))
    enable_sleep_mode = False
    sleep_level = 0
    enforce_eager = bool(vcfg.get("enforce_eager", False))
    disable_custom_all_reduce = bool(vcfg.get("disable_custom_all_reduce", True))

    decode_bs_per_rank = max(
        1,
        int(
            owner._rollout_decode_batch_size_per_rank(
                rollout_backend="vllm",
                rollout_context="eval",
            )
        ),
    )
    default_max_num_seqs = max(8, int(decode_bs_per_rank) * 4)

    max_num_seqs_raw = vcfg.get("max_num_seqs", default_max_num_seqs)
    max_num_seqs: Optional[int] = None
    if max_num_seqs_raw is not None:
        try:
            max_num_seqs = int(max_num_seqs_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.vllm.max_num_seqs must be an int"
            ) from exc
        if max_num_seqs <= 0:
            raise ValueError("rollout_matching.vllm.max_num_seqs must be > 0")

    vllm_engine_kwargs: Dict[str, Any] = {}
    limit_mm_per_prompt: Optional[Dict[str, int]] = None
    max_num_batched_tokens_raw = vcfg.get("max_num_batched_tokens", None)
    if max_num_batched_tokens_raw is not None:
        try:
            max_num_batched_tokens = int(max_num_batched_tokens_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.vllm.max_num_batched_tokens must be an int"
            ) from exc
        if max_num_batched_tokens <= 0:
            raise ValueError(
                "rollout_matching.vllm.max_num_batched_tokens must be > 0"
            )
        vllm_engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

    if "enable_chunked_prefill" in vcfg:
        vllm_engine_kwargs["enable_chunked_prefill"] = bool(
            vcfg.get("enable_chunked_prefill")
        )
    if "disable_chunked_mm_input" in vcfg:
        vllm_engine_kwargs["disable_chunked_mm_input"] = bool(
            vcfg.get("disable_chunked_mm_input")
        )
    if "kv_cache_dtype" in vcfg and vcfg.get("kv_cache_dtype") is not None:
        kv_cache_dtype = vcfg.get("kv_cache_dtype")
        if not isinstance(kv_cache_dtype, str):
            raise ValueError("rollout_matching.vllm.kv_cache_dtype must be a string")
        vllm_engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if "cpu_offload_gb" in vcfg and vcfg.get("cpu_offload_gb") is not None:
        try:
            cpu_offload_gb = float(vcfg.get("cpu_offload_gb"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rollout_matching.vllm.cpu_offload_gb must be a float"
            ) from exc
        if cpu_offload_gb < 0:
            raise ValueError("rollout_matching.vllm.cpu_offload_gb must be >= 0")
        vllm_engine_kwargs["cpu_offload_gb"] = cpu_offload_gb
    if "swap_space" in vcfg and vcfg.get("swap_space") is not None:
        try:
            swap_space = float(vcfg.get("swap_space"))
        except (TypeError, ValueError) as exc:
            raise ValueError("rollout_matching.vllm.swap_space must be a float") from exc
        if swap_space < 0:
            raise ValueError("rollout_matching.vllm.swap_space must be >= 0")
        vllm_engine_kwargs["swap_space"] = swap_space
    if "limit_mm_per_prompt" in vcfg and vcfg.get("limit_mm_per_prompt") is not None:
        limit_raw = vcfg.get("limit_mm_per_prompt")
        if not isinstance(limit_raw, Mapping):
            raise ValueError(
                "rollout_matching.vllm.limit_mm_per_prompt must be a mapping"
            )
        limit_parsed: Dict[str, int] = {}
        for k, v in limit_raw.items():
            if not isinstance(k, str):
                raise ValueError(
                    "rollout_matching.vllm.limit_mm_per_prompt keys must be strings"
                )
            try:
                limit_parsed[k] = int(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout_matching.vllm.limit_mm_per_prompt values must be ints"
                ) from exc
        limit_mm_per_prompt = limit_parsed

    if "mm_encoder_tp_mode" in vcfg and vcfg.get("mm_encoder_tp_mode") is not None:
        mm_encoder_tp_mode = vcfg.get("mm_encoder_tp_mode")
        if not isinstance(mm_encoder_tp_mode, str):
            raise ValueError(
                "rollout_matching.vllm.mm_encoder_tp_mode must be a string"
            )
        mm_encoder_tp_mode = mm_encoder_tp_mode.strip().lower()
        if mm_encoder_tp_mode not in {"weights", "data"}:
            raise ValueError(
                "rollout_matching.vllm.mm_encoder_tp_mode must be 'weights' or 'data'"
            )
        vllm_engine_kwargs["mm_encoder_tp_mode"] = mm_encoder_tp_mode

    if "skip_mm_profiling" in vcfg:
        vllm_engine_kwargs["skip_mm_profiling"] = bool(vcfg.get("skip_mm_profiling"))

    dist_backend_raw = vcfg.get("distributed_executor_backend")
    if dist_backend_raw is None:
        dist_backend = "mp" if world_size == 1 and tp_size == 1 else "external_launcher"
    else:
        dist_backend = str(dist_backend_raw).strip() or (
            "mp" if world_size == 1 and tp_size == 1 else "external_launcher"
        )

    return VllmEngineConfig(
        vcfg=vcfg,
        dist=dist,
        world_size=int(world_size),
        rank=int(rank),
        tp_size=int(tp_size),
        max_model_len=int(max_model_len),
        enable_lora=bool(enable_lora),
        load_format=str(load_format),
        gpu_mem=float(gpu_mem),
        enable_prefix_caching=bool(enable_prefix_caching),
        enable_sleep_mode=bool(enable_sleep_mode),
        sleep_level=int(sleep_level),
        enforce_eager=bool(enforce_eager),
        disable_custom_all_reduce=bool(disable_custom_all_reduce),
        decode_bs_per_rank=int(decode_bs_per_rank),
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt=limit_mm_per_prompt,
        vllm_engine_kwargs=vllm_engine_kwargs,
        dist_backend=str(dist_backend),
    )

