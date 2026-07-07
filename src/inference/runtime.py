"""Inference runtime assembly for dry setup and identity recording."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.artifacts.checkpoint_handoff import validate_checkpoint_handoff
from src.common.errors import RuntimeContractError
from src.config.inference import InferConfig
from src.adapters.dora import load_inference_dora_adapter
from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
from src.qwen.special_token_embeddings import (
    load_inference_embedding_delta,
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
        embedding_delta_receipt = load_inference_embedding_delta(
            config=config,
            qwen=qwen,
        )
    checkpoint_handoff = _resolve_checkpoint_handoff(config)
    _prepare_model_for_generation(qwen)

    return InferenceRuntime(
        qwen=qwen,
        adapter_receipt=adapter_receipt,
        embedding_delta_receipt=embedding_delta_receipt,
        model_identity=_model_identity(
            config=config,
            qwen=qwen,
            adapter_receipt=adapter_receipt,
            embedding_delta_receipt=embedding_delta_receipt,
            checkpoint_handoff=checkpoint_handoff,
        ),
    )


def _load_qwen(config: InferConfig) -> Any:
    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=config.model.base_model,
            dtype=config.model.dtype,
            attn_implementation=config.model.attn_implementation,
            patch_embed_linearization=config.model.runtime_patches.patch_embed_linearization,
            load_model=True,
        )
    )


def _prepare_model_for_generation(qwen: Any) -> None:
    model = _qwen_model(qwen)
    if model is None:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    move = getattr(model, "to", None)
    if callable(move):
        move(device)
    eval_model = getattr(model, "eval", None)
    if callable(eval_model):
        eval_model()


def _model_identity(
    *,
    config: InferConfig,
    qwen: Any,
    adapter_receipt: Mapping[str, Any] | None,
    embedding_delta_receipt: Mapping[str, Any] | None,
    checkpoint_handoff: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config.adapter is not None and config.embedding_delta is not None:
        family = "base-plus-adapter-plus-delta"
    elif config.adapter is not None:
        family = "base-plus-adapter"
    elif config.embedding_delta is not None:
        family = "base-plus-delta"
    else:
        family = "base-only"
    identity = {
        "family": family,
        "base": {"path": _base_model_path(config, qwen)},
        "adapter": None if adapter_receipt is None else dict(adapter_receipt),
        "embedding_delta": None
        if embedding_delta_receipt is None
        else dict(embedding_delta_receipt),
    }
    if checkpoint_handoff is not None:
        identity["checkpoint_handoff"] = dict(checkpoint_handoff)
    return identity


def _resolve_checkpoint_handoff(config: InferConfig) -> dict[str, Any] | None:
    candidate_dirs = _checkpoint_dirs_from_payload_paths(config)
    if not candidate_dirs:
        return None
    if len(candidate_dirs) > 1:
        raise RuntimeContractError(
            "inference adapter and embedding-delta payloads resolve to different checkpoints",
            code="inference.checkpoint_handoff_payload_mismatch",
            context={"checkpoint_dirs": [str(path) for path in sorted(candidate_dirs)]},
        )
    checkpoint_dir = next(iter(candidate_dirs))
    handoff_path = checkpoint_dir / "checkpoint_handoff.json"
    if not handoff_path.exists():
        return None
    run_dir = checkpoint_dir.parent.parent
    verdict = validate_checkpoint_handoff(
        run_root=run_dir,
        checkpoint_ref=handoff_path,
        gate="handoff",
    )
    if verdict["status"] != "pass":
        raise RuntimeContractError(
            "inference checkpoint handoff identity validation failed",
            code="inference.checkpoint_handoff_hold",
            context=verdict,
        )
    return {
        "path": verdict["handoff_path"],
        "fingerprint": verdict["handoff_fingerprint"],
        "adapter_identity": verdict["adapter_identity"],
        "special_token_embedding_identity": verdict[
            "special_token_embedding_identity"
        ],
    }


def _checkpoint_dirs_from_payload_paths(config: InferConfig) -> set[Path]:
    dirs: set[Path] = set()
    if config.adapter is not None:
        adapter_path = Path(config.adapter.path).expanduser().resolve()
        if adapter_path.name == "adapter" and adapter_path.parent.parent.name == "checkpoints":
            dirs.add(adapter_path.parent)
    if config.embedding_delta is not None:
        delta_path = Path(config.embedding_delta.path).expanduser().resolve()
        if (
            delta_path.name == "special_token_embeddings"
            and delta_path.parent.parent.name == "checkpoints"
        ):
            dirs.add(delta_path.parent)
    return dirs


def _base_model_path(config: InferConfig, qwen: Any) -> str:
    if isinstance(qwen, Mapping):
        return str(qwen.get("base_model_path", config.model.base_model))
    return str(getattr(qwen, "base_model_path", config.model.base_model))


def _qwen_model(qwen: Any) -> Any | None:
    if isinstance(qwen, Mapping):
        return qwen.get("model")
    return getattr(qwen, "model", None)
