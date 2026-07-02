"""Qwen component loading and setup receipts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from transformers import AutoConfig, AutoProcessor

from src.common.errors import QwenForwardContractError
from src.config.models import TrainConfig
from src.qwen.patches import (
    apply_qwen3_vl_patch_embed_linearization,
    model_not_loaded_patch_receipts,
)
from src.qwen.tokens import QwenTokenIdentity, validate_qwen_token_identity


@dataclass(frozen=True)
class QwenProcessorIdentity:
    processor_class: str
    tokenizer_class: str
    image_processor_class: str
    patch_size: int
    merge_size: int
    temporal_patch_size: int

    def to_artifact_dict(self) -> dict[str, int | str]:
        return {
            "processor_class": self.processor_class,
            "tokenizer_class": self.tokenizer_class,
            "image_processor_class": self.image_processor_class,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "temporal_patch_size": self.temporal_patch_size,
        }


@dataclass(frozen=True)
class QwenModelIdentity:
    config_class: str
    model_type: str
    architectures: tuple[str, ...]
    tie_word_embeddings: bool
    text_vocab_size: int
    text_hidden_size: int | None
    config_dtype: str | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "config_class": self.config_class,
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "tie_word_embeddings": self.tie_word_embeddings,
            "text_vocab_size": self.text_vocab_size,
            "text_hidden_size": self.text_hidden_size,
            "config_dtype": self.config_dtype,
        }


@dataclass(frozen=True)
class QwenComponents:
    base_model_path: Path
    processor: Any
    tokenizer: Any
    config: Any
    model: Any | None
    processor_identity: QwenProcessorIdentity
    model_identity: QwenModelIdentity
    token_identity: QwenTokenIdentity
    attn_implementation: str
    load_model: bool
    package_versions: dict[str, str]
    runtime_patches: dict[str, dict[str, Any]]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "base_model_path": str(self.base_model_path),
            "load_model": self.load_model,
            "attn_implementation": self.attn_implementation,
            "processor": self.processor_identity.to_artifact_dict(),
            "model": self.model_identity.to_artifact_dict(),
            "tokens": self.token_identity.to_artifact_dict(),
            "package_versions": dict(self.package_versions),
            "runtime_patches": dict(self.runtime_patches),
        }


@dataclass(frozen=True)
class QwenLoadOptions:
    base_model: str
    dtype: Literal["bf16", "fp16", "fp32"]
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"]
    patch_embed_linearization: Literal["enabled", "disabled"] = "enabled"
    load_model: bool = False


def load_qwen_components(
    config: TrainConfig,
    *,
    load_model: bool = False,
) -> QwenComponents:
    """Load Qwen processor/tokenizer/config and optionally the model.

    The default path is intentionally processor/config-only so preflight and
    tests can run without allocating the 2B model. Training code can request
    `load_model=True` once adapter and runtime setup are ready.
    """

    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=config.model.base_model,
            dtype=config.training.precision,
            attn_implementation=config.model.attn_implementation,
            patch_embed_linearization=config.model.runtime_patches.patch_embed_linearization,
            load_model=load_model,
        )
    )


def load_qwen_components_from_options(options: QwenLoadOptions) -> QwenComponents:
    """Load Qwen components from owner-neutral runtime options."""

    base_model_path = Path(options.base_model).expanduser().resolve()
    processor = AutoProcessor.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise QwenForwardContractError(
            "loaded Qwen processor does not expose a tokenizer",
            code="qwen.processor_tokenizer_missing",
            context={"base_model_path": str(base_model_path), "processor_class": type(processor).__name__},
        )
    hf_config = AutoConfig.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    token_identity = validate_qwen_token_identity(tokenizer)
    processor_identity = _processor_identity(processor, tokenizer)
    model_identity = _model_identity(hf_config)
    model = _load_model_from_options(options, base_model_path) if options.load_model else None
    patch_policy = options.patch_embed_linearization
    runtime_patches = (
        _apply_runtime_patches(model, patch_policy=patch_policy)
        if model is not None
        else model_not_loaded_patch_receipts(patch_policy=patch_policy)
    )

    return QwenComponents(
        base_model_path=base_model_path,
        processor=processor,
        tokenizer=tokenizer,
        config=hf_config,
        model=model,
        processor_identity=processor_identity,
        model_identity=model_identity,
        token_identity=token_identity,
        attn_implementation=options.attn_implementation,
        load_model=options.load_model,
        package_versions=_package_versions(("transformers", "tokenizers", "torch")),
        runtime_patches=runtime_patches,
    )


def _processor_identity(processor: Any, tokenizer: Any) -> QwenProcessorIdentity:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise QwenForwardContractError(
            "loaded Qwen processor does not expose an image_processor",
            code="qwen.image_processor_missing",
            context={"processor_class": type(processor).__name__},
        )
    return QwenProcessorIdentity(
        processor_class=type(processor).__name__,
        tokenizer_class=type(tokenizer).__name__,
        image_processor_class=type(image_processor).__name__,
        patch_size=_required_positive_int(image_processor, "patch_size"),
        merge_size=_required_positive_int(image_processor, "merge_size"),
        temporal_patch_size=_required_positive_int(image_processor, "temporal_patch_size"),
    )


def _model_identity(hf_config: Any) -> QwenModelIdentity:
    text_config = getattr(hf_config, "text_config", None)
    text_vocab_size = getattr(text_config, "vocab_size", None)
    if text_vocab_size is None:
        text_vocab_size = getattr(hf_config, "vocab_size", None)
    if text_vocab_size is None:
        raise QwenForwardContractError(
            "Qwen config does not expose text vocab size",
            code="qwen.config_vocab_size_missing",
            context={"config_class": type(hf_config).__name__},
        )
    hidden_size = getattr(text_config, "hidden_size", None)
    dtype = getattr(hf_config, "torch_dtype", None)
    if dtype is None:
        dtype = getattr(hf_config, "dtype", None)
    return QwenModelIdentity(
        config_class=type(hf_config).__name__,
        model_type=str(getattr(hf_config, "model_type", "")),
        architectures=tuple(str(item) for item in (getattr(hf_config, "architectures", None) or ())),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        text_vocab_size=int(text_vocab_size),
        text_hidden_size=None if hidden_size is None else int(hidden_size),
        config_dtype=None if dtype is None else str(dtype),
    )


def _load_model(config: TrainConfig, base_model_path: Path) -> Any:
    return _load_model_from_options(
        QwenLoadOptions(
            base_model=str(base_model_path),
            dtype=config.training.precision,
            attn_implementation=config.model.attn_implementation,
            patch_embed_linearization=config.model.runtime_patches.patch_embed_linearization,
            load_model=True,
        ),
        base_model_path,
    )


def _load_model_from_options(options: QwenLoadOptions, base_model_path: Path) -> Any:
    from transformers import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        dtype=_torch_dtype(options.dtype),
        attn_implementation=options.attn_implementation,
        device_map=None,
        local_files_only=True,
    )


def _apply_runtime_patches(
    model: Any,
    *,
    patch_policy: str,
) -> dict[str, dict[str, Any]]:
    patch_embed_receipt = apply_qwen3_vl_patch_embed_linearization(
        model,
        policy=patch_policy,
    )
    return {patch_embed_receipt.name: patch_embed_receipt.to_artifact_dict()}


def _torch_dtype(precision: str) -> Any:
    import torch

    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise QwenForwardContractError(
        "unsupported Qwen model precision",
        code="qwen.unsupported_precision",
        context={"precision": precision},
    )


def _required_positive_int(owner: Any, name: str) -> int:
    value = getattr(owner, name, None)
    if value is None:
        raise QwenForwardContractError(
            "Qwen processor image field is missing",
            code="qwen.processor_image_field_missing",
            context={"field": name, "owner_class": type(owner).__name__},
        )
    int_value = int(value)
    if int_value <= 0:
        raise QwenForwardContractError(
            "Qwen processor image field must be positive",
            code="qwen.processor_image_field_invalid",
            context={"field": name, "value": value, "owner_class": type(owner).__name__},
        )
    return int_value


def _package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions
