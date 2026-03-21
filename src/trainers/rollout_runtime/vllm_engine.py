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


def sync_vllm_full_weights_if_needed(
    *,
    owner: Any,
) -> None:
    step = int(getattr(getattr(owner, "state", None), "global_step", 0) or 0)
    if step == owner._vllm_last_loaded_step:
        return

    engine = owner._ensure_vllm_engine()

    from contextlib import nullcontext

    unwrap_model = getattr(getattr(owner, "accelerator", None), "unwrap_model", None)
    train_model = unwrap_model(owner.model) if callable(unwrap_model) else owner.model

    try:
        from accelerate.utils import is_peft_model
    except (TypeError, ValueError):
        is_peft_model = None  # type: ignore[assignment]

    is_peft = bool(is_peft_model(train_model)) if is_peft_model is not None else False

    merge_cm = nullcontext()
    unmerge_cm = nullcontext()
    if is_peft:
        try:
            from swift.trainers.rlhf_trainer.utils import (
                patch_lora_merge,
                patch_lora_unmerge,
            )

            merge_cm = patch_lora_merge(train_model)
            unmerge_cm = patch_lora_unmerge(train_model)
        except (TypeError, ValueError):
            merge_cm = nullcontext()
            unmerge_cm = nullcontext()

    from swift.trainers.rlhf_trainer.utils import get_gather_if_zero3_context

    params = [p for _, p in train_model.named_parameters()]
    gather_if_zero3 = get_gather_if_zero3_context(owner)

    with gather_if_zero3(params), merge_cm, torch.no_grad():
        merged = False
        try:
            if is_peft:
                try:
                    train_model.merge_adapter()
                    merged = True
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Failed to merge adapter weights from the training model for vLLM full sync. "
                        "Mitigations: switch rollout backend to HF or ensure your PEFT stack supports "
                        "merge_adapter/unmerge_adapter."
                    ) from exc

            state_dict = train_model.state_dict()
            if is_peft:
                prefix_removed = {
                    k.removeprefix("base_model.model."): v for k, v in state_dict.items()
                }
                state_dict = {
                    k.replace(".base_layer", ""): v for k, v in prefix_removed.items()
                }
                prefix = getattr(train_model, "prefix", None)
                if isinstance(prefix, str) and prefix:
                    state_dict = {
                        k: v for k, v in state_dict.items() if prefix not in k
                    }
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
                state_dict = {k: v for k, v in state_dict.items() if "lora_" not in k}

            engine.inner_model.load_weights(state_dict.items())
        finally:
            if is_peft and merged:
                with unmerge_cm:
                    train_model.unmerge_adapter()

    engine.engine.reset_prefix_cache()
    owner._vllm_last_loaded_step = step
