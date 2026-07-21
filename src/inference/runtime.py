"""Processor-only inference frontend and backend launch assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.errors import RuntimeContractError
from src.config.inference import InferConfig
from src.inference.backend import BackendLaunch
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options


@dataclass(frozen=True)
class InferenceFrontend:
    """Shared prompt/image frontend; it never owns an executable model."""

    qwen: Any
    launch: BackendLaunch


def assemble_frontend(
    config: InferConfig,
    *,
    generation_config_fingerprint: str,
    execution_model: Mapping[str, Any] | None = None,
) -> InferenceFrontend:
    qwen = load_qwen_components_from_options(_frontend_load_options(config))
    if getattr(qwen, "model", None) is not None:
        raise RuntimeContractError(
            "the shared inference frontend must not load an executable model",
            code="inference.frontend_model_loaded",
        )
    return InferenceFrontend(
        qwen=qwen,
        launch=prepare_backend_launch(
            config,
            generation_config_fingerprint=generation_config_fingerprint,
            execution_model=execution_model,
        ),
    )


def prepare_backend_launch(
    config: InferConfig,
    *,
    generation_config_fingerprint: str,
    execution_model: Mapping[str, Any] | None = None,
) -> BackendLaunch:
    """Project strict config into a serializable backend launch contract."""

    execution_identity = None if execution_model is None else dict(execution_model)
    if config.backend.type == "hf" and execution_identity is not None:
        raise RuntimeContractError(
            "HF inference must use the configured dynamic base, adapter, and embedding delta",
            code="inference.hf_execution_model_forbidden",
        )
    model_path = config.model.base_model
    if execution_identity is not None:
        candidate = execution_identity.get("model_path") or execution_identity.get(
            "snapshot_root"
        )
        if not isinstance(candidate, str) or not candidate:
            raise RuntimeContractError(
                "execution-model identity is missing its executable model path",
                code="inference.execution_model_path_missing",
            )
        model_path = candidate

    if config.backend.type == "hf":
        backend_options = {"hf": config.backend.hf.model_dump(mode="json")}
        adapter = (
            None if config.adapter is None else config.adapter.model_dump(mode="json")
        )
        embedding_delta = (
            None
            if config.embedding_delta is None
            else config.embedding_delta.model_dump(mode="json")
        )
    else:
        if execution_identity is None and (
            config.adapter is not None or config.embedding_delta is not None
        ):
            raise RuntimeContractError(
                "composed vLLM inference requires a resolved execution model",
                code="inference.execution_model_required",
            )
        backend_options = {"vllm": config.backend.vllm.model_dump(mode="json")}
        adapter = None
        embedding_delta = None

    return BackendLaunch(
        backend=config.backend.type,
        model_path=str(Path(model_path).expanduser().resolve()),
        model_dtype=config.model.dtype,
        batch_size=config.generation.batch_size,
        generation_config_fingerprint=generation_config_fingerprint,
        backend_options=backend_options,
        execution_model_identity=execution_identity,
        adapter=adapter,
        embedding_delta=embedding_delta,
    )


def _frontend_load_options(config: InferConfig) -> QwenLoadOptions:
    if config.backend.type == "hf":
        attention = config.backend.hf.attn_implementation
        patch_policy = config.backend.hf.patch_embed_linearization
    else:
        # These values are inert when no model is loaded; public vLLM config
        # remains free of HF execution controls.
        attention = "eager"
        patch_policy = "disabled"
    return QwenLoadOptions(
        base_model=config.model.base_model,
        dtype=config.model.dtype,
        attn_implementation=attention,
        patch_embed_linearization=patch_policy,
        load_model=False,
    )
