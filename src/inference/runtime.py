"""Inference runtime assembly for dry setup and identity recording."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.config.inference import InferConfig
from src.adapters.dora import load_inference_dora_adapter
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.special_token_embeddings import (
    validate_inference_embedding_delta_identity,
)


@dataclass(frozen=True)
class InferenceRuntime:
    qwen: Any
    adapter_receipt: Mapping[str, Any] | None
    embedding_delta_receipt: Mapping[str, Any] | None
    model_identity: dict[str, Any]


def assemble_runtime(
    config: InferConfig,
) -> InferenceRuntime:
    qwen = _load_qwen(config)
    adapter_receipt = None
    if config.adapter is not None:
        adapter_receipt = load_inference_dora_adapter(config=config, qwen=qwen)
    embedding_delta_receipt = None
    if config.embedding_delta is not None:
        embedding_delta_receipt = validate_inference_embedding_delta_identity(
            config=config,
            qwen=qwen,
        )

    return InferenceRuntime(
        qwen=qwen,
        adapter_receipt=adapter_receipt,
        embedding_delta_receipt=embedding_delta_receipt,
        model_identity=_model_identity(
            config=config,
            qwen=qwen,
            adapter_receipt=adapter_receipt,
            embedding_delta_receipt=embedding_delta_receipt,
        ),
    )


def _load_qwen(config: InferConfig) -> Any:
    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=config.model.base_model,
            dtype=config.model.dtype,
            attn_implementation=config.model.attn_implementation,
            patch_embed_linearization=config.model.runtime_patches.patch_embed_linearization,
            load_model=config.adapter is not None,
        )
    )


def _model_identity(
    *,
    config: InferConfig,
    qwen: Any,
    adapter_receipt: Mapping[str, Any] | None,
    embedding_delta_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config.adapter is not None and config.embedding_delta is not None:
        family = "base-plus-adapter-plus-delta"
    elif config.adapter is not None:
        family = "base-plus-adapter"
    elif config.embedding_delta is not None:
        family = "base-plus-delta"
    else:
        family = "base-only"
    return {
        "family": family,
        "base": {"path": _base_model_path(config, qwen)},
        "adapter": None if adapter_receipt is None else dict(adapter_receipt),
        "embedding_delta": None
        if embedding_delta_receipt is None
        else dict(embedding_delta_receipt),
    }


def _base_model_path(config: InferConfig, qwen: Any) -> str:
    if isinstance(qwen, Mapping):
        return str(qwen.get("base_model_path", config.model.base_model))
    return str(getattr(qwen, "base_model_path", config.model.base_model))
