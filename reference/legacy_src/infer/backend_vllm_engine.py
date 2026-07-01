from __future__ import annotations

import inspect
from contextlib import contextmanager
from copy import copy as shallow_copy
from typing import Any, Literal, Mapping

import torch

from src.infer.backend_vllm_config import VllmEngineConfig
from src.infer.runtime import rollout_owner_cfg, vllm_mode_from_rollout_owner


def _instance_override(owner: Any, name: str) -> Any:
    try:
        raw = getattr(owner, "__dict__", {}).get(name)
    except (AttributeError, TypeError):
        return None
    return raw if callable(raw) else None


def _vllm_cfg(owner: Any) -> Mapping[str, Any]:
    vcfg_raw = rollout_owner_cfg(owner, "vllm", {}) or {}
    if not isinstance(vcfg_raw, Mapping):
        raise ValueError("rollout_matching.vllm must be a mapping")
    return vcfg_raw


def vllm_sleep_mode_enabled(owner: Any) -> bool:
    """Whether colocate vLLM should use sleep-mode lifecycle hooks."""

    override = _instance_override(owner, "_vllm_sleep_mode_enabled")
    if override is not None:
        return bool(override())
    return bool(_vllm_cfg(owner).get("enable_sleep_mode", False))


def vllm_reinit_each_eval(owner: Any) -> bool:
    """Whether colocate vLLM engine should be rebuilt every eval window."""

    override = _instance_override(owner, "_vllm_reinit_each_eval")
    if override is not None:
        return bool(override())
    raw_reinit = _vllm_cfg(owner).get("reinit_each_eval", False)
    if not isinstance(raw_reinit, bool):
        raise ValueError("rollout_matching.vllm.reinit_each_eval must be a bool")
    return bool(raw_reinit)


def vllm_sleep_level(owner: Any, *, default: int = 0) -> int:
    """Normalized sleep level for optional vLLM sleep-mode lifecycle."""

    override = _instance_override(owner, "_vllm_sleep_level")
    if override is not None:
        return int(override(default=default))
    raw_level = _vllm_cfg(owner).get("sleep_level", default)
    try:
        level = int(default if raw_level is None else raw_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("rollout_matching.vllm.sleep_level must be an int >= 0") from exc
    if level < 0:
        raise ValueError("rollout_matching.vllm.sleep_level must be >= 0")
    return level


def validate_vllm_eval_lifecycle_preflight() -> None:
    try:
        from vllm import EngineArgs
    except Exception as exc:
        raise RuntimeError(
            "Eval-time colocate vLLM lifecycle requires a vLLM runtime with sleep/wake "
            "support. Failed to import vllm.EngineArgs."
        ) from exc

    try:
        ctor_sig = inspect.signature(EngineArgs.__init__)
    except Exception as exc:
        raise RuntimeError(
            "Unable to inspect vLLM EngineArgs for sleep-mode support preflight."
        ) from exc
    if "enable_sleep_mode" not in ctor_sig.parameters:
        raise RuntimeError(
            "Eval-time colocate vLLM requires EngineArgs(enable_sleep_mode=...). "
            "Upgrade vLLM to a runtime that supports sleep mode."
        )

    llm_engine_cls = None
    for module_name in (
        "vllm.v1.engine.llm_engine",
        "vllm.engine.llm_engine",
    ):
        try:
            module = __import__(module_name, fromlist=["LLMEngine"])
            candidate = getattr(module, "LLMEngine", None)
        except (AttributeError, ImportError, ModuleNotFoundError):
            continue
        if candidate is not None:
            llm_engine_cls = candidate
            break
    if llm_engine_cls is None:
        raise RuntimeError(
            "Unable to locate vLLM LLMEngine class for sleep/wake preflight checks."
        )
    if not callable(getattr(llm_engine_cls, "sleep", None)) or not callable(
        getattr(llm_engine_cls, "wake_up", None)
    ):
        raise RuntimeError(
            "Eval-time colocate vLLM requires LLMEngine.sleep(level=...) and "
            "LLMEngine.wake_up() APIs."
        )


def vllm_raw_engine_or_raise(engine_wrapper: Any) -> Any:
    raw_engine = getattr(engine_wrapper, "engine", None)
    if raw_engine is None:
        raise RuntimeError(
            "vLLM engine wrapper is missing an underlying `engine` handle."
        )
    return raw_engine


def wake_vllm_engine(engine_wrapper: Any) -> None:
    raw_engine = vllm_raw_engine_or_raise(engine_wrapper)
    wake_fn = getattr(raw_engine, "wake_up", None)
    if not callable(wake_fn):
        raise RuntimeError(
            "vLLM runtime does not expose LLMEngine.wake_up(); "
            "cannot satisfy eval lifecycle requirements."
        )
    try:
        wake_fn()
    except Exception as exc:
        raise RuntimeError("Failed to wake vLLM engine for evaluation.") from exc


def sleep_vllm_engine(engine_wrapper: Any, *, level: int) -> None:
    raw_engine = vllm_raw_engine_or_raise(engine_wrapper)
    sleep_fn = getattr(raw_engine, "sleep", None)
    if not callable(sleep_fn):
        raise RuntimeError(
            "vLLM runtime does not expose LLMEngine.sleep(level=...); "
            "cannot satisfy eval lifecycle requirements."
        )
    try:
        sleep_fn(int(level))
        return
    except TypeError:
        try:
            sleep_fn(level=int(level))
            return
        except Exception as exc:
            raise RuntimeError(
                f"Failed to sleep vLLM engine at level={int(level)}."
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to sleep vLLM engine at level={int(level)}."
        ) from exc


def best_effort_fix_vllm_nccl_allocator_atexit_order() -> None:
    """Best-effort mitigation for vLLM CUDAPluggableAllocator teardown crashes."""

    try:
        import atexit

        from vllm.distributed.device_communicators import (
            pynccl_allocator as _pynccl_alloc,
        )

        mem_cleanup = getattr(_pynccl_alloc, "_cleanup_nccl_mem_pool", None)
        alloc_cleanup = getattr(_pynccl_alloc, "_cleanup_nccl_allocator_wrapper", None)
        if not callable(mem_cleanup) or not callable(alloc_cleanup):
            return

        try:
            atexit.unregister(mem_cleanup)
        except (RuntimeError, TypeError, ValueError):
            pass
        try:
            atexit.unregister(alloc_cleanup)
        except (RuntimeError, TypeError, ValueError):
            pass

        # Register allocator cleanup first, then MemPool cleanup, so MemPool
        # runs first at exit.
        atexit.register(alloc_cleanup)
        atexit.register(mem_cleanup)
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return


def best_effort_patch_vllm_cumem_sleep_no_empty_cache() -> None:
    """Best-effort mitigation for CUDAPluggableAllocator teardown aborts."""

    try:
        from vllm.device_allocator import cumem as _cumem

        CuMemAllocator = getattr(_cumem, "CuMemAllocator", None)
        if CuMemAllocator is None:
            return

        orig_sleep = getattr(CuMemAllocator, "sleep", None)
        if not callable(orig_sleep):
            return

        if bool(getattr(orig_sleep, "_coordexp_no_empty_cache", False)):
            return

        def _sleep_no_empty_cache(self, *args, **kwargs):
            orig_empty_cache = torch.cuda.empty_cache
            try:
                torch.cuda.empty_cache = lambda: None
                return orig_sleep(self, *args, **kwargs)
            finally:
                torch.cuda.empty_cache = orig_empty_cache

        setattr(_sleep_no_empty_cache, "_coordexp_no_empty_cache", True)
        CuMemAllocator.sleep = _sleep_no_empty_cache  # type: ignore[assignment]
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return


def best_effort_cleanup_vllm_sleep_mode_pools() -> None:
    """Best-effort cleanup for vLLM sleep-mode pluggable allocator pools."""

    try:
        import gc

        # vLLM CuMemAllocator (sleep mode pools for weights/kv_cache).
        # IMPORTANT: do not clear `pointer_to_data` here. CUDAPluggableAllocator
        # frees may invoke the Python free callback during MemPool teardown,
        # which expects pointer bookkeeping to still be present.
        try:
            from vllm.device_allocator.cumem import CuMemAllocator

            inst = getattr(CuMemAllocator, "instance", None)
            if inst is not None:
                try:
                    getattr(inst, "allocator_and_pools", {}).clear()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
        except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError):
            pass

        # vLLM NCCL symmetric-memory allocator (if enabled).
        try:
            import sys

            pynccl_allocator = sys.modules.get(
                "vllm.distributed.device_communicators.pynccl_allocator"
            )
            if pynccl_allocator is None:
                from vllm.distributed.device_communicators import pynccl_allocator

            # Ensure pool is dropped before wrapper (MemPool depends on it).
            if getattr(pynccl_allocator, "_mem_pool", None) is not None:
                pynccl_allocator._mem_pool = None
            if getattr(pynccl_allocator, "_allocator_wrapper", None) is not None:
                pynccl_allocator._allocator_wrapper = None
            if getattr(pynccl_allocator, "_allocator", None) is not None:
                pynccl_allocator._allocator = None
        except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError):
            pass

        gc.collect()
    except (
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return


@contextmanager
def maybe_eval_vllm_colocate_window(
    *,
    owner: Any,
    rollout_backend: Literal["hf", "vllm"],
) -> Any:
    if rollout_backend != "vllm" or vllm_mode_from_rollout_owner(owner) != "colocate":
        yield
        return

    reinit_each_eval = bool(vllm_reinit_each_eval(owner))

    with owner._maybe_rollout_offload_context(
        rollout_backend=rollout_backend,
        force_enable=True,
        force_offload_model=True,
        force_offload_optimizer=True,
        require_cuda_drain=True,
    ):
        if reinit_each_eval:
            owner._shutdown_vllm_colocate_engine(wake_before_release=False)
            owner._cuda_memory_drain(synchronize=True)
        _ = owner._ensure_vllm_engine()
        prev = bool(getattr(owner, "_eval_vllm_window_active", False))
        owner._eval_vllm_window_active = True
        try:
            yield
        finally:
            owner._eval_vllm_window_active = prev
            if reinit_each_eval:
                owner._shutdown_vllm_colocate_engine(wake_before_release=False)
                owner._cuda_memory_drain(synchronize=True)


def ensure_vllm_engine(*, owner: Any, logger: Any) -> Any:
    """Lazily initialize and return the colocated vLLM engine for a rollout owner."""

    engine = getattr(owner, "_vllm_engine", None)
    if engine is not None:
        return engine

    from src.infer.backend_vllm_config import resolve_vllm_engine_config

    engine_cfg = resolve_vllm_engine_config(owner)
    dist = engine_cfg.dist
    world_size = int(engine_cfg.world_size)
    tp_size = int(engine_cfg.tp_size)

    model_dir = getattr(owner.model, "model_dir", None) or getattr(
        getattr(owner.model, "model", None), "model_dir", None
    )
    if not model_dir:
        raise RuntimeError(
            "vLLM rollout backend requires a ms-swift model wrapper with `model_dir`. "
            "Set rollout_backend: hf to disable vLLM rollouts."
        )
    model_info = getattr(owner.model, "model_info", None)
    torch_dtype = (
        getattr(model_info, "torch_dtype", None) if model_info is not None else None
    )

    logger.info(
        "Initializing vLLM rollout engine: tp=%s world_size=%s max_model_len=%s gpu_memory_utilization=%.2f "
        "decode_batch_size_per_rank=%s max_num_seqs=%s sleep_mode=%s limit_mm_per_prompt=%s engine_kwargs=%s",
        tp_size,
        world_size,
        int(engine_cfg.max_model_len),
        float(engine_cfg.gpu_mem),
        int(engine_cfg.decode_bs_per_rank),
        engine_cfg.max_num_seqs,
        bool(engine_cfg.enable_sleep_mode),
        engine_cfg.limit_mm_per_prompt,
        dict(engine_cfg.vllm_engine_kwargs) or {},
    )

    if tp_size > 1 and (dist is None or not dist.is_initialized()):
        raise RuntimeError(
            "vLLM tensor parallel requires torch.distributed to be initialized"
        )

    return instantiate_vllm_engine(
        owner=owner,
        engine_cfg=engine_cfg,
        model_dir=str(model_dir),
        torch_dtype=torch_dtype,
        logger=logger,
    )


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
            raw_engine = vllm_raw_engine_or_raise(engine)
            is_sleeping_fn = getattr(raw_engine, "is_sleeping", None)
            is_sleeping = bool(is_sleeping_fn()) if callable(is_sleeping_fn) else False
            if is_sleeping:
                wake_vllm_engine(engine)
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
            raw_engine = vllm_raw_engine_or_raise(engine)
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

    best_effort_cleanup_vllm_sleep_mode_pools()
    owner._cuda_memory_drain(synchronize=True)
