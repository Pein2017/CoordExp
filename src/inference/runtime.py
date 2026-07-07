"""Inference runtime assembly for dry setup and identity recording."""

from __future__ import annotations

import json
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
    SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS,
    load_inference_embedding_delta,
)


INFERENCE_CONFIG_FAMILY = "configs/coordexp_swift/infer"


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
    if checkpoint_handoff is not None:
        _validate_handoff_runtime_identity(
            config=config,
            qwen=qwen,
            handoff=checkpoint_handoff,
        )
    adapter_loader_receipt = adapter_receipt
    embedding_delta_loader_receipt = embedding_delta_receipt
    if checkpoint_handoff is not None:
        adapter_receipt = checkpoint_handoff.get("adapter_identity")
        embedding_delta_receipt = checkpoint_handoff.get("special_token_embedding_identity")
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
            adapter_loader_receipt=adapter_loader_receipt,
            embedding_delta_loader_receipt=embedding_delta_loader_receipt,
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
    adapter_loader_receipt: Mapping[str, Any] | None,
    embedding_delta_loader_receipt: Mapping[str, Any] | None,
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
        "adapter_loader_receipt": None
        if adapter_loader_receipt is None
        else dict(adapter_loader_receipt),
        "embedding_delta_loader_receipt": None
        if embedding_delta_loader_receipt is None
        else dict(embedding_delta_loader_receipt),
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
    handoff_manifest = _read_json_object(
        _resolve_run_relative(run_dir, str(verdict["handoff_path"])),
        field="checkpoint_handoff",
    )
    handoff = {
        "path": verdict["handoff_path"],
        "fingerprint": verdict["handoff_fingerprint"],
        "adapter_identity": verdict["adapter_identity"],
        "special_token_embedding_identity": verdict[
            "special_token_embedding_identity"
        ],
        "base_model": dict(handoff_manifest.get("base_model") or {}),
        "processor_identity": dict(handoff_manifest.get("processor_identity") or {}),
        "template_identity": dict(handoff_manifest.get("template_identity") or {}),
        "resolved_config_fingerprint": handoff_manifest.get("resolved_config_fingerprint"),
        "intended_inference_config_family": handoff_manifest.get(
            "intended_inference_config_family"
        ),
    }
    _validate_configured_payloads_match_handoff(
        config=config,
        run_dir=run_dir,
        handoff=handoff,
    )
    return handoff


def _validate_handoff_runtime_identity(
    *,
    config: InferConfig,
    qwen: Any,
    handoff: Mapping[str, Any],
) -> None:
    base_model = handoff.get("base_model")
    if not isinstance(base_model, Mapping):
        _raise_handoff_identity_mismatch(
            field="base_model",
            expected={"path": _base_model_path(config, qwen)},
            observed=base_model,
        )
    else:
        _require_same_identity_path(
            field="base_model.path",
            expected=Path(_base_model_path(config, qwen)),
            observed=base_model.get("path"),
        )
        _require_same_identity_value(
            field="base_model.base_config_sha256",
            expected=_qwen_identity_value(qwen, "base_config_sha256"),
            observed=base_model.get("base_config_sha256"),
        )
        _require_same_identity_value(
            field="base_model.tokenizer_sha256",
            expected=_qwen_identity_value(qwen, "tokenizer_sha256"),
            observed=base_model.get("tokenizer_sha256"),
        )
    _require_same_identity_value(
        field="processor_identity",
        expected=_qwen_processor_identity(qwen),
        observed=handoff.get("processor_identity"),
    )
    _require_same_identity_value(
        field="template_identity",
        expected=_template_identity(config),
        observed=handoff.get("template_identity"),
    )
    _require_same_identity_value(
        field="intended_inference_config_family",
        expected=INFERENCE_CONFIG_FAMILY,
        observed=handoff.get("intended_inference_config_family"),
    )


def _validate_configured_payloads_match_handoff(
    *,
    config: InferConfig,
    run_dir: Path,
    handoff: Mapping[str, Any],
) -> None:
    adapter_identity = handoff.get("adapter_identity")
    if config.adapter is not None and not isinstance(adapter_identity, Mapping):
        _raise_handoff_payload_mismatch(
            field="adapter_identity",
            configured=Path(config.adapter.path),
            declared=None,
        )
    if config.adapter is not None and isinstance(adapter_identity, Mapping):
        payload_path = adapter_identity.get("payload_path")
        if isinstance(payload_path, str) and payload_path:
            _require_same_path(
                configured=Path(config.adapter.path),
                declared=_resolve_run_relative(run_dir, payload_path),
                field="adapter_identity.payload_path",
            )
    embedding_identity = handoff.get("special_token_embedding_identity")
    if config.embedding_delta is not None and not isinstance(embedding_identity, Mapping):
        _raise_handoff_payload_mismatch(
            field="special_token_embedding_identity",
            configured=_embedding_delta_payload_dir(config.embedding_delta.path),
            declared=None,
        )
    if config.embedding_delta is not None and isinstance(embedding_identity, Mapping):
        configured_delta_dir = _embedding_delta_payload_dir(config.embedding_delta.path)
        for field in ("metadata_path", "tensor_path"):
            rel_path = embedding_identity.get(field)
            if not isinstance(rel_path, str) or not rel_path:
                continue
            _require_same_path(
                configured=configured_delta_dir,
                declared=_resolve_run_relative(run_dir, rel_path).parent,
                field=f"special_token_embedding_identity.{field}",
            )


def _require_same_path(
    *,
    configured: Path,
    declared: Path,
    field: str,
) -> None:
    if configured.expanduser().resolve() == declared.expanduser().resolve():
        return
    _raise_handoff_payload_mismatch(
        field=field,
        configured=configured,
        declared=declared,
    )


def _raise_handoff_payload_mismatch(
    *,
    field: str,
    configured: Path,
    declared: Path | None,
) -> None:
    raise RuntimeContractError(
        "inference payload path disagrees with checkpoint handoff identity",
        code="inference.checkpoint_handoff_payload_mismatch",
        context={
            "field": field,
            "configured_path": str(configured),
            "declared_path": None if declared is None else str(declared),
        },
    )


def _require_same_identity_path(
    *,
    field: str,
    expected: Path,
    observed: Any,
) -> None:
    if isinstance(observed, str) and Path(observed).expanduser().resolve() == expected.expanduser().resolve():
        return
    _raise_handoff_identity_mismatch(
        field=field,
        expected=str(expected),
        observed=observed,
    )


def _require_same_identity_value(
    *,
    field: str,
    expected: Any,
    observed: Any,
) -> None:
    if observed == expected:
        return
    _raise_handoff_identity_mismatch(
        field=field,
        expected=expected,
        observed=observed,
    )


def _raise_handoff_identity_mismatch(
    *,
    field: str,
    expected: Any,
    observed: Any,
) -> None:
    raise RuntimeContractError(
        "inference runtime identity disagrees with checkpoint handoff",
        code="inference.checkpoint_handoff_identity_mismatch",
        context={
            "field": field,
            "expected": expected,
            "observed": observed,
        },
    )


def _qwen_identity_value(qwen: Any, field: str) -> Any:
    if isinstance(qwen, Mapping):
        return qwen.get(field)
    return getattr(qwen, field, None)


def _qwen_processor_identity(qwen: Any) -> dict[str, Any] | None:
    processor_identity = _qwen_identity_value(qwen, "processor_identity")
    if processor_identity is None:
        return None
    if hasattr(processor_identity, "to_artifact_dict"):
        return dict(processor_identity.to_artifact_dict())
    if isinstance(processor_identity, Mapping):
        return dict(processor_identity)
    return None


def _template_identity(config: InferConfig) -> dict[str, Any]:
    template = config.template
    if hasattr(template, "model_dump"):
        return dict(template.model_dump(mode="json"))
    return dict(template)


def _embedding_delta_payload_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.name == SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS:
        return path.parent
    return path


def _resolve_run_relative(run_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return run_dir / path


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
        elif (
            delta_path.name == SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
            and delta_path.parent.name == "special_token_embeddings"
            and delta_path.parent.parent.parent.name == "checkpoints"
        ):
            dirs.add(delta_path.parent.parent)
    return dirs


def _read_json_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeContractError(
            "inference checkpoint handoff JSON is not readable",
            code="inference.checkpoint_handoff_read_failed",
            context={"field": field, "path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeContractError(
            "inference checkpoint handoff JSON must be an object",
            code="inference.checkpoint_handoff_shape",
            context={"field": field, "path": str(path), "type": type(payload).__name__},
        )
    return payload


def _base_model_path(config: InferConfig, qwen: Any) -> str:
    if isinstance(qwen, Mapping):
        return str(qwen.get("base_model_path", config.model.base_model))
    return str(getattr(qwen, "base_model_path", config.model.base_model))


def _qwen_model(qwen: Any) -> Any | None:
    if isinstance(qwen, Mapping):
        return qwen.get("model")
    return getattr(qwen, "model", None)
