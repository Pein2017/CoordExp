"""Inference runtime assembly for dry setup and identity recording."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.common.errors import RuntimeContractError
from src.config.inference import InferConfig
from src.qwen.loading import QwenLoadOptions, load_qwen_components_from_options


@dataclass(frozen=True)
class InferenceRuntime:
    qwen: Any
    adapter_receipt: Mapping[str, Any] | None
    embedding_delta_receipt: Mapping[str, Any] | None
    model_identity: dict[str, Any]


def assemble_runtime(
    config: InferConfig,
    *,
    load_qwen: Callable[[InferConfig], Any] | None = None,
    adapter_loader: Callable[[InferConfig, Any], Mapping[str, Any]] | None = None,
    delta_validator: Callable[[InferConfig, Any], Mapping[str, Any]] | None = None,
) -> InferenceRuntime:
    qwen = load_qwen(config) if load_qwen is not None else _load_qwen(config)
    adapter_receipt = None
    if config.adapter is not None:
        if adapter_loader is None:
            raise RuntimeContractError(
                "adapter inference runtime requires an adapter-owner loader",
                code="runtime.adapter_loader_required",
            )
        adapter_receipt = adapter_loader(config, qwen)
    embedding_delta_receipt = None
    if config.embedding_delta is not None:
        if delta_validator is None:
            raise RuntimeContractError(
                "embedding-delta inference runtime requires an owner validator",
                code="runtime.embedding_delta_validator_required",
            )
        embedding_delta_receipt = delta_validator(config, qwen)

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
            load_model=False,
        )
    )


def _model_identity(
    *,
    config: InferConfig,
    qwen: Any,
    adapter_receipt: Mapping[str, Any] | None,
    embedding_delta_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config.adapter is None:
        family = "base-only"
    elif config.embedding_delta is None:
        family = "base-plus-adapter"
    else:
        family = "base-plus-adapter-plus-delta"
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
