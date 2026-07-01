"""Runtime-independent config resolution checks."""

from __future__ import annotations

from src.common.errors import ConfigContractError
from src.config.models import QwenRuntimeControls, RuntimeBatchResolution, TrainConfig


_DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


def resolve_effective_batch_runtime(
    config: TrainConfig,
    *,
    world_size: int,
) -> RuntimeBatchResolution:
    if world_size <= 0:
        raise ConfigContractError(
            "world size must be positive",
            code="config.world_size",
            context={"world_size": world_size},
        )
    effective_batch_size = config.training.effective_batch_size
    if effective_batch_size < world_size:
        raise ConfigContractError(
            "effective batch size is smaller than world size",
            code="config.effective_batch_world_size",
            context={
                "effective_batch_size": effective_batch_size,
                "world_size": world_size,
            },
        )
    if effective_batch_size % world_size != 0:
        raise ConfigContractError(
            "effective batch size must divide evenly by world size",
            code="config.effective_batch_divisibility",
            context={
                "effective_batch_size": effective_batch_size,
                "world_size": world_size,
            },
        )
    resolved = effective_batch_size // world_size
    _validate_backend_accumulation(config, resolved)
    return RuntimeBatchResolution(
        world_size=world_size,
        effective_batch_size=effective_batch_size,
        resolved_grad_accum_steps=resolved,
    )


def estimate_full_logits_bytes(
    *,
    global_max_length: int,
    vocab_size: int,
    dtype: str,
) -> int:
    if vocab_size <= 0:
        raise ConfigContractError(
            "vocab size must be positive",
            code="config.vocab_size",
            context={"vocab_size": vocab_size},
        )
    if dtype not in _DTYPE_BYTES:
        raise ConfigContractError(
            "unsupported logits dtype for memory estimate",
            code="config.logits_dtype",
            context={"dtype": dtype},
        )
    return global_max_length * vocab_size * _DTYPE_BYTES[dtype]


def validate_static_qwen_runtime_controls(config: TrainConfig) -> None:
    if config.model.attn_implementation != "flash_attention_2":
        raise ConfigContractError(
            "packed training requires FlashAttention 2",
            code="config.qwen_attention",
            context={"attn_implementation": config.model.attn_implementation},
        )
    if config.training.precision not in {"bf16", "fp16"}:
        raise ConfigContractError(
            "FlashAttention packed training requires bf16 or fp16 precision",
            code="config.qwen_precision",
            context={"precision": config.training.precision},
        )
    if config.model.processor.do_resize:
        raise ConfigContractError(
            "Qwen processor resize must be disabled for CoordExp training",
            code="config.qwen_resize",
            context={"do_resize": True},
        )


def resolve_qwen_runtime_controls(
    config: TrainConfig,
    *,
    tokenizer_vocab_size: int,
    model_logits_dtype: str,
) -> QwenRuntimeControls:
    validate_static_qwen_runtime_controls(config)
    estimate = estimate_full_logits_bytes(
        global_max_length=config.packing.global_max_length,
        vocab_size=tokenizer_vocab_size,
        dtype=model_logits_dtype,
    )
    if estimate > config.model.logits_memory_budget_bytes:
        raise ConfigContractError(
            "estimated full-logits tensor exceeds configured budget",
            code="config.logits_memory_budget",
            context={
                "estimated_bytes": estimate,
                "budget_bytes": config.model.logits_memory_budget_bytes,
                "global_max_length": config.packing.global_max_length,
                "vocab_size": tokenizer_vocab_size,
                "dtype": model_logits_dtype,
                "model_logits_dtype": model_logits_dtype,
                "compute_precision": config.training.precision,
            },
        )
    return QwenRuntimeControls(
        attn_implementation=config.model.attn_implementation,
        compute_precision=config.training.precision,
        model_logits_dtype=model_logits_dtype,
        global_max_length=config.packing.global_max_length,
        tokenizer_vocab_size=tokenizer_vocab_size,
        estimated_logits_bytes=estimate,
        logits_memory_budget_bytes=config.model.logits_memory_budget_bytes,
    )


def _validate_backend_accumulation(config: TrainConfig, resolved: int) -> None:
    accelerate = config.runtime.accelerate
    if accelerate and accelerate.gradient_accumulation_steps not in (None, resolved):
        raise ConfigContractError(
            "runtime.accelerate gradient accumulation conflicts with derived value",
            code="config.accelerate_accumulation_conflict",
            context={
                "authored": accelerate.gradient_accumulation_steps,
                "resolved_grad_accum_steps": resolved,
            },
        )
    deepspeed = config.runtime.deepspeed
    if deepspeed and deepspeed.gradient_accumulation_steps not in (None, resolved):
        raise ConfigContractError(
            "runtime.deepspeed gradient accumulation conflicts with derived value",
            code="config.deepspeed_accumulation_conflict",
            context={
                "authored": deepspeed.gradient_accumulation_steps,
                "resolved_grad_accum_steps": resolved,
            },
        )
    if deepspeed and deepspeed.train_batch_size not in (
        None,
        config.training.effective_batch_size,
    ):
        raise ConfigContractError(
            "runtime.deepspeed train batch size conflicts with effective batch",
            code="config.deepspeed_batch_conflict",
            context={
                "authored": deepspeed.train_batch_size,
                "effective_batch_size": config.training.effective_batch_size,
            },
        )
