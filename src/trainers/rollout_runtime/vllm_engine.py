from __future__ import annotations

from copy import copy as shallow_copy
from typing import Any

import torch

from .vllm_config import VllmEngineConfig


def instantiate_vllm_engine(
    *,
    owner: Any,
    engine_cfg: VllmEngineConfig,
    model_dir: str,
    torch_dtype: Any,
    logger: Any,
) -> Any:
    dist = engine_cfg.dist
    world_size = int(engine_cfg.world_size)
    tp_size = int(engine_cfg.tp_size)

    max_lora_rank = 16

    if tp_size > 1:
        if dist is None or not dist.is_initialized():
            raise RuntimeError(
                "vLLM tensor parallel requires torch.distributed to be initialized"
            )
        owner._vllm_tp_group, _ = dist.new_subgroups_by_enumeration(
            [
                list(range(i * tp_size, (i + 1) * tp_size))
                for i in range(world_size // tp_size)
            ]
        )
    owner._vllm_tp_size = int(tp_size)

    vllm_template = shallow_copy(owner.template)
    vllm_template.packing = False
    vllm_template.padding_free = False
    vllm_template.set_mode("vllm")

    owner._vllm_saved_cuda_allocator = None
    try:
        if torch.cuda.is_available():
            import torch.cuda.memory as cuda_mem

            owner._vllm_saved_cuda_allocator = cuda_mem._get_current_allocator()
    except Exception:
        owner._vllm_saved_cuda_allocator = None

    try:
        from swift.llm import VllmEngine

        engine = VllmEngine(
            model_dir,
            torch_dtype=torch_dtype,
            template=vllm_template,
            tensor_parallel_size=int(engine_cfg.tp_size),
            gpu_memory_utilization=float(engine_cfg.gpu_mem),
            max_model_len=int(engine_cfg.max_model_len),
            max_num_seqs=engine_cfg.max_num_seqs,
            enforce_eager=bool(engine_cfg.enforce_eager),
            disable_custom_all_reduce=bool(engine_cfg.disable_custom_all_reduce),
            limit_mm_per_prompt=engine_cfg.limit_mm_per_prompt,
            load_format=str(engine_cfg.load_format),
            enable_lora=bool(engine_cfg.enable_lora),
            max_loras=1,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=bool(engine_cfg.enable_prefix_caching),
            engine_kwargs=dict(engine_cfg.vllm_engine_kwargs) or None,
            distributed_executor_backend=str(engine_cfg.dist_backend),
        )
    except (TypeError, ValueError) as exc:
        logger.exception(
            "vLLM engine init failed (backend=%s): %s",
            str(engine_cfg.dist_backend),
            exc,
        )
        raise RuntimeError(
            "Failed to initialize vLLM engine for rollout generation. "
            "Set rollout_backend: hf to bypass vLLM."
        ) from exc

    owner._vllm_engine = engine
    return engine
