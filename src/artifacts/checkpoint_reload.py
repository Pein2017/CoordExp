"""Checkpoint reload payload verification."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from src.artifacts.checkpoints import ADAPTER_CONFIG_NAME, ADAPTER_WEIGHT_NAMES
from src.common.errors import ArtifactContractError
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
)


RELOAD_CONTRACT_NAME = "base_model_plus_dora_adapter_plus_token_embed_delta"


@dataclass(frozen=True)
class CheckpointReloadPlan:
    checkpoint_id: str
    run_dir: Path
    checkpoint_metadata_path: Path
    checkpoint_metadata: Mapping[str, Any]
    base_model_path: Path
    adapter_dir: Path | None
    adapter_config_path: Path | None
    adapter_weight_paths: tuple[Path, ...]
    special_token_payload_dir: Path | None
    special_token_tensor_path: Path | None
    special_token_metadata_path: Path | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_dir": str(self.run_dir),
            "checkpoint_metadata_path": str(self.checkpoint_metadata_path),
            "base_model_path": str(self.base_model_path),
            "adapter_dir": None if self.adapter_dir is None else str(self.adapter_dir),
            "adapter_config_path": (
                None if self.adapter_config_path is None else str(self.adapter_config_path)
            ),
            "adapter_weight_paths": [str(path) for path in self.adapter_weight_paths],
            "special_token_payload_dir": (
                None
                if self.special_token_payload_dir is None
                else str(self.special_token_payload_dir)
            ),
            "special_token_tensor_path": (
                None
                if self.special_token_tensor_path is None
                else str(self.special_token_tensor_path)
            ),
            "special_token_metadata_path": (
                None
                if self.special_token_metadata_path is None
                else str(self.special_token_metadata_path)
            ),
            "reload_contract": RELOAD_CONTRACT_NAME,
        }


def build_checkpoint_reload_plan(
    checkpoint_path: str | Path | None,
) -> CheckpointReloadPlan:
    if checkpoint_path is None:
        raise ArtifactContractError(
            "checkpoint reload requires a checkpoint metadata or alias path",
            code="checkpoint_reload.path_missing",
        )
    path = Path(checkpoint_path).expanduser()
    payload = _read_json(path)
    if "metadata_path" in payload and "adapter" not in payload:
        run_dir = _infer_run_dir_from_alias(path)
        path = _resolve_run_relative(run_dir, payload["metadata_path"])
        payload = _read_json(path)
    run_dir = _infer_run_dir_from_metadata(path)
    checkpoint_id = str(payload.get("checkpoint_id") or "")
    if not checkpoint_id:
        raise ArtifactContractError(
            "checkpoint metadata must include checkpoint_id",
            code="checkpoint_reload.checkpoint_id_missing",
            context={"path": str(path)},
        )

    adapter = _mapping(payload.get("adapter"), field="adapter", path=path)
    special = _mapping(
        payload.get("special_token_embeddings"),
        field="special_token_embeddings",
        path=path,
    )
    adapter_dir = _enabled_payload_dir(
        adapter,
        field="adapter",
        run_dir=run_dir,
        path=path,
    )
    special_tensor_path = _enabled_payload_path(
        special,
        field="special_token_embeddings.tensor_path",
        run_dir=run_dir,
        path=path,
    )
    special_metadata_path = _enabled_payload_path(
        special,
        field="special_token_embeddings.metadata_path",
        run_dir=run_dir,
        path=path,
    )
    special_metadata = _read_json(special_metadata_path)
    base_model_path = _base_model_path(
        checkpoint_metadata=payload,
        special_token_metadata=special_metadata,
        path=path,
    )
    adapter_weight_paths = _adapter_weight_paths(adapter, run_dir=run_dir)
    return CheckpointReloadPlan(
        checkpoint_id=checkpoint_id,
        run_dir=run_dir,
        checkpoint_metadata_path=path,
        checkpoint_metadata=payload,
        base_model_path=base_model_path,
        adapter_dir=adapter_dir,
        adapter_config_path=None if adapter_dir is None else adapter_dir / ADAPTER_CONFIG_NAME,
        adapter_weight_paths=adapter_weight_paths,
        special_token_payload_dir=special_metadata_path.parent,
        special_token_tensor_path=special_tensor_path,
        special_token_metadata_path=special_metadata_path,
    )


def verify_checkpoint_reload_payloads(
    plan: CheckpointReloadPlan,
) -> dict[str, Any]:
    if not isinstance(plan, CheckpointReloadPlan):
        raise ArtifactContractError(
            "checkpoint reload verification requires a CheckpointReloadPlan",
            code="checkpoint_reload.plan_type",
            context={"value_type": type(plan).__name__},
        )
    adapter_receipt = _verify_adapter_payload(plan)
    special_receipt = _verify_special_token_payload(plan)
    return {
        "checkpoint_id": plan.checkpoint_id,
        "checkpoint_metadata_path": str(plan.checkpoint_metadata_path),
        "base_model_path": str(plan.base_model_path),
        "adapter": adapter_receipt,
        "special_token_embeddings": special_receipt,
        "reload_contract": RELOAD_CONTRACT_NAME,
    }


def _verify_adapter_payload(plan: CheckpointReloadPlan) -> dict[str, Any]:
    if plan.adapter_dir is None or plan.adapter_config_path is None:
        raise ArtifactContractError(
            "checkpoint reload requires an enabled adapter payload",
            code="checkpoint_reload.adapter_disabled",
            context={"checkpoint_id": plan.checkpoint_id},
        )
    if not plan.adapter_dir.is_dir():
        raise ArtifactContractError(
            "checkpoint adapter payload directory is missing",
            code="checkpoint_reload.adapter_dir_missing",
            context={"path": str(plan.adapter_dir)},
        )
    adapter_config = _read_json(plan.adapter_config_path)
    if adapter_config.get("use_dora") is not True:
        raise ArtifactContractError(
            "checkpoint adapter payload must preserve use_dora: true",
            code="checkpoint_reload.adapter_not_dora",
            context={
                "adapter_config_path": str(plan.adapter_config_path),
                "use_dora": adapter_config.get("use_dora"),
            },
        )
    missing_weights = [path for path in plan.adapter_weight_paths if not path.exists()]
    if missing_weights:
        raise ArtifactContractError(
            "checkpoint adapter payload is missing declared weight files",
            code="checkpoint_reload.adapter_weight_missing",
            context={"missing": [str(path) for path in missing_weights]},
        )
    return {
        "enabled": True,
        "path": str(plan.adapter_dir),
        "config_path": str(plan.adapter_config_path),
        "config": adapter_config,
        "weight_files": [
            _relative_to_run(path, run_dir=plan.run_dir)
            for path in plan.adapter_weight_paths
        ],
    }


def _verify_special_token_payload(plan: CheckpointReloadPlan) -> dict[str, Any]:
    if (
        plan.special_token_payload_dir is None
        or plan.special_token_tensor_path is None
        or plan.special_token_metadata_path is None
    ):
        raise ArtifactContractError(
            "checkpoint reload requires selected-token embedding deltas",
            code="checkpoint_reload.special_token_payload_disabled",
            context={"checkpoint_id": plan.checkpoint_id},
        )
    metadata = _read_json(plan.special_token_metadata_path)
    if metadata.get("semantics") != SPECIAL_TOKEN_EMBEDDING_SEMANTICS:
        raise ArtifactContractError(
            "special-token embedding payload records unsupported semantics",
            code="checkpoint_reload.special_token_semantics",
            context={
                "metadata_path": str(plan.special_token_metadata_path),
                "semantics": metadata.get("semantics"),
            },
        )
    tensors = load_file(str(plan.special_token_tensor_path), device="cpu")
    if set(tensors) != {DEFAULT_EMBED_DELTA_TENSOR_KEY}:
        raise ArtifactContractError(
            "special-token embedding payload must contain only compact delta tensors",
            code="checkpoint_reload.special_token_tensor_keys",
            context={
                "tensor_path": str(plan.special_token_tensor_path),
                "actual_keys": sorted(tensors),
            },
        )
    tensor = tensors[DEFAULT_EMBED_DELTA_TENSOR_KEY]
    expected_shape = tuple(int(item) for item in metadata.get("tensor_shape", ()))
    observed_shape = tuple(int(item) for item in tensor.shape)
    if observed_shape != expected_shape:
        raise ArtifactContractError(
            "special-token embedding tensor shape does not match metadata",
            code="checkpoint_reload.special_token_tensor_shape",
            context={
                "expected_shape": list(expected_shape),
                "observed_shape": list(observed_shape),
            },
        )
    tensor_dtype = str(tensor.dtype).replace("torch.", "")
    if tensor_dtype != metadata.get("tensor_dtype"):
        raise ArtifactContractError(
            "special-token embedding tensor dtype does not match metadata",
            code="checkpoint_reload.special_token_tensor_dtype",
            context={
                "expected_dtype": metadata.get("tensor_dtype"),
                "observed_dtype": tensor_dtype,
            },
        )
    return {
        "enabled": True,
        "path": str(plan.special_token_payload_dir),
        "tensor_path": str(plan.special_token_tensor_path),
        "metadata_path": str(plan.special_token_metadata_path),
        "tensor_key": DEFAULT_EMBED_DELTA_TENSOR_KEY,
        "tensor_shape": list(observed_shape),
        "tensor_dtype": tensor_dtype,
        "metadata": metadata,
    }


def _mapping(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(
            "checkpoint metadata field must be an object",
            code="checkpoint_reload.metadata_field_shape",
            context={"path": str(path), "field": field, "value_type": type(value).__name__},
        )
    return value


def _enabled_payload_dir(
    payload: Mapping[str, Any],
    *,
    field: str,
    run_dir: Path,
    path: Path,
) -> Path:
    if payload.get("enabled") is not True:
        raise ArtifactContractError(
            "checkpoint reload payload must be enabled",
            code="checkpoint_reload.payload_disabled",
            context={"path": str(path), "field": field},
        )
    payload_path = payload.get("payload_path")
    if not isinstance(payload_path, str) or not payload_path:
        raise ArtifactContractError(
            "checkpoint reload payload is missing payload_path",
            code="checkpoint_reload.payload_path_missing",
            context={"path": str(path), "field": field},
        )
    return _resolve_run_relative(run_dir, payload_path)


def _enabled_payload_path(
    payload: Mapping[str, Any],
    *,
    field: str,
    run_dir: Path,
    path: Path,
) -> Path:
    if payload.get("enabled") is not True:
        raise ArtifactContractError(
            "checkpoint reload payload must be enabled",
            code="checkpoint_reload.payload_disabled",
            context={"path": str(path), "field": field},
        )
    payload_path = payload.get(field.rsplit(".", maxsplit=1)[-1])
    if not isinstance(payload_path, str) or not payload_path:
        raise ArtifactContractError(
            "checkpoint reload payload path is missing",
            code="checkpoint_reload.payload_path_missing",
            context={"path": str(path), "field": field},
        )
    return _resolve_run_relative(run_dir, payload_path)


def _adapter_weight_paths(adapter: Mapping[str, Any], *, run_dir: Path) -> tuple[Path, ...]:
    files = adapter.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ArtifactContractError(
            "checkpoint adapter payload must list saved files",
            code="checkpoint_reload.adapter_files_missing",
        )
    paths = tuple(
        _resolve_run_relative(run_dir, str(file_path))
        for file_path in files
        if Path(str(file_path)).name in ADAPTER_WEIGHT_NAMES
    )
    if not paths:
        raise ArtifactContractError(
            "checkpoint adapter payload must include adapter weight files",
            code="checkpoint_reload.adapter_weight_missing",
            context={"files": [str(file_path) for file_path in files]},
        )
    return paths


def _base_model_path(
    *,
    checkpoint_metadata: Mapping[str, Any],
    special_token_metadata: Mapping[str, Any],
    path: Path,
) -> Path:
    candidates = [
        special_token_metadata.get("base_model_path"),
        _nested(checkpoint_metadata, ("adapter", "receipt", "base_model_path")),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return Path(candidate).expanduser()
    raise ArtifactContractError(
        "checkpoint reload metadata does not identify the base model path",
        code="checkpoint_reload.base_model_path_missing",
        context={"path": str(path)},
    )


def _nested(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _infer_run_dir_from_alias(path: Path) -> Path:
    if path.parent.name != "checkpoints":
        raise ArtifactContractError(
            "checkpoint alias must live under a checkpoints directory",
            code="checkpoint_reload.alias_path",
            context={"path": str(path)},
        )
    return path.parent.parent


def _infer_run_dir_from_metadata(path: Path) -> Path:
    if path.name != "checkpoint.json" or path.parent.parent.name != "checkpoints":
        raise ArtifactContractError(
            "checkpoint metadata must be run_dir/checkpoints/step-N/checkpoint.json",
            code="checkpoint_reload.metadata_path",
            context={"path": str(path)},
        )
    return path.parent.parent.parent


def _resolve_run_relative(run_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return run_dir / path


def _relative_to_run(path: Path, *, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactContractError(
            "checkpoint reload JSON file is missing",
            code="checkpoint_reload.json_missing",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            "checkpoint reload JSON file is invalid",
            code="checkpoint_reload.json_invalid",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(
            "checkpoint reload JSON file must contain an object",
            code="checkpoint_reload.json_shape",
            context={"path": str(path), "value_type": type(payload).__name__},
        )
    return payload


__all__ = [
    "CheckpointReloadPlan",
    "RELOAD_CONTRACT_NAME",
    "build_checkpoint_reload_plan",
    "verify_checkpoint_reload_payloads",
]
