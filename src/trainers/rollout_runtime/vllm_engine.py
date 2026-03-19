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


def shutdown_vllm_colocate_engine(
    *,
    owner: Any,
    logger: Any,
    wake_before_release: bool = True,
) -> None:
    engine = getattr(owner, "_vllm_engine", None)
    if engine is None:
        return

    raw_engine: Any = None

    if bool(wake_before_release):
        try:
            raw_engine = owner._vllm_raw_engine_or_raise(engine)
            is_sleeping_fn = getattr(raw_engine, "is_sleeping", None)
            is_sleeping = bool(is_sleeping_fn()) if callable(is_sleeping_fn) else False
            if is_sleeping:
                owner._wake_vllm_engine(engine)
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Failed to wake colocate vLLM engine during shutdown: %s", exc
            )

    if raw_engine is None:
        try:
            raw_engine = owner._vllm_raw_engine_or_raise(engine)
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            raw_engine = None

    owner._cuda_memory_drain(synchronize=True)

    def _maybe_invoke_shutdown(obj: Any) -> None:
        if obj is None:
            return
        try:
            shutdown_fn = getattr(obj, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()
                return
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            pass

        try:
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                close_fn()
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            pass

    if raw_engine is not None:
        try:
            _maybe_invoke_shutdown(raw_engine)
            _maybe_invoke_shutdown(getattr(raw_engine, "engine_core", None))
            _maybe_invoke_shutdown(getattr(raw_engine, "model_executor", None))
            _maybe_invoke_shutdown(getattr(raw_engine, "executor", None))
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Failed to shutdown colocate vLLM engine cleanly: %s", exc
            )

    for obj in (engine, raw_engine):
        if obj is None:
            continue
        for attr in (
            "engine",
            "engine_core",
            "model_executor",
            "executor",
            "model",
            "llm_engine",
        ):
            try:
                if hasattr(obj, attr):
                    setattr(obj, attr, None)
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                pass

    owner._vllm_engine = None
    owner._vllm_last_loaded_step = -1
    owner._vllm_tp_group = None
    owner._vllm_tp_size = 1
    owner._eval_vllm_window_active = False

    try:
        del engine
    except (AttributeError, NameError, OSError, RuntimeError, TypeError, ValueError):
        pass
    try:
        del raw_engine
    except (AttributeError, NameError, OSError, RuntimeError, TypeError, ValueError):
        pass

    owner._best_effort_cleanup_vllm_sleep_mode_pools()
    owner._cuda_memory_drain(synchronize=True)
