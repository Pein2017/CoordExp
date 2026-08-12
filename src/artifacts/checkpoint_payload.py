"""Self-authenticating inference-payload manifests for committed checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.adapters.dora import inspect_dora_adapter_payload
from src.common.errors import ArtifactContractError
from src.qwen.special_token_embeddings import (
    inspect_special_token_embedding_delta_payload,
)


INFERENCE_PAYLOAD_MANIFEST_NAME = "inference_payload_manifest.json"
INFERENCE_PAYLOAD_MANIFEST_SCHEMA = (
    "coordexp-swift-inference-checkpoint-payload-manifest"
)
INFERENCE_PAYLOAD_PUBLICATION_SCHEMA = (
    "coordexp-swift-inference-checkpoint-payload-publication"
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "adapter",
        "special_token_embedding_delta",
        "aggregate_digest",
    }
)
_PUBLICATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "manifest_relative_path",
        "manifest_file_sha256",
        "aggregate_digest",
    }
)


def write_inference_checkpoint_payload_manifest(
    checkpoint_dir: str | Path,
    *,
    expected_base_model_path: str | Path | None = None,
    expected_base_config_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> Path:
    """Write the authoritative manifest before the checkpoint directory rename."""

    root = Path(checkpoint_dir).expanduser().resolve()
    manifest_path = root / INFERENCE_PAYLOAD_MANIFEST_NAME
    if manifest_path.exists():
        raise ArtifactContractError(
            "inference checkpoint payload manifest already exists",
            code="checkpoint.inference_payload_manifest_exists",
            context={"path": str(manifest_path)},
        )
    manifest = _build_stable_manifest(
        root,
        expected_base_model_path=expected_base_model_path,
        expected_base_config_sha256=expected_base_config_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def load_inference_checkpoint_payload_manifest(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Load and self-authenticate the manifest without trusting its payload scan."""

    root = Path(checkpoint_dir).expanduser().resolve()
    path = root / INFERENCE_PAYLOAD_MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(
            "inference checkpoint payload manifest is missing or invalid",
            code="checkpoint.inference_payload_manifest_invalid",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    return _validate_manifest(value)


def build_inference_checkpoint_payload_identity(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Re-scan one committed checkpoint and return its compact event binding."""

    root = Path(checkpoint_dir).expanduser().resolve()
    manifest = load_inference_checkpoint_payload_manifest(root)
    observed = _build_stable_manifest(root)
    if observed != manifest:
        raise ArtifactContractError(
            "committed inference payload differs from its manifest",
            code="checkpoint.inference_payload_identity_mismatch",
            context={"checkpoint_dir": str(root)},
        )
    manifest_path = root / INFERENCE_PAYLOAD_MANIFEST_NAME
    return {
        "schema": INFERENCE_PAYLOAD_PUBLICATION_SCHEMA,
        "schema_version": 2,
        "manifest_relative_path": INFERENCE_PAYLOAD_MANIFEST_NAME,
        "manifest_file_sha256": _sha256_file(manifest_path),
        "aggregate_digest": manifest["aggregate_digest"],
    }


def admit_inference_checkpoint_payload_identity(
    checkpoint_dir: str | Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the committed payload and exact-compare its event identity."""

    root = Path(checkpoint_dir).expanduser().resolve()
    try:
        expected = _validate_publication_identity(expected_identity)
        observed = build_inference_checkpoint_payload_identity(root)
    except BaseException as exc:
        if isinstance(exc, ArtifactContractError) and exc.code == (
            "checkpoint.inference_payload_identity_mismatch"
        ):
            raise
        raise ArtifactContractError(
            "committed inference payload cannot be admitted",
            code="checkpoint.inference_payload_identity_mismatch",
            context={
                "checkpoint_dir": str(root),
                "observed_error_code": getattr(exc, "code", type(exc).__name__),
            },
            cause=exc,
        ) from exc
    if observed != expected:
        raise ArtifactContractError(
            "committed inference payload identity differs from the expected binding",
            code="checkpoint.inference_payload_identity_mismatch",
            context={"checkpoint_dir": str(root)},
        )
    return observed


def _build_stable_manifest(
    root: Path,
    *,
    expected_base_model_path: str | Path | None = None,
    expected_base_config_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    first = _build_manifest_body(
        root,
        expected_base_model_path=expected_base_model_path,
        expected_base_config_sha256=expected_base_config_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
    )
    second = _build_manifest_body(
        root,
        expected_base_model_path=expected_base_model_path,
        expected_base_config_sha256=expected_base_config_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
    )
    if first != second:
        raise ArtifactContractError(
            "inference checkpoint payload changed while its manifest was built",
            code="checkpoint.inference_payload_changed",
            context={"checkpoint_dir": str(root)},
        )
    return {**first, "aggregate_digest": _sha256_json(first)}


def _build_manifest_body(
    root: Path,
    *,
    expected_base_model_path: str | Path | None,
    expected_base_config_sha256: str | None,
    expected_tokenizer_sha256: str | None,
) -> dict[str, Any]:
    if not root.is_dir():
        raise ArtifactContractError(
            "inference checkpoint payload root is not a directory",
            code="checkpoint.inference_payload_missing",
            context={"checkpoint_dir": str(root)},
        )
    adapter_root = root / "adapter"
    adapter_identity = inspect_dora_adapter_payload(
        adapter_root,
        expected_base_model_path=expected_base_model_path,
    )
    adapter = {
        "status": "present",
        "relative_root": "adapter",
        "files": _inventory_files(adapter_root),
        "inspector_identity": _without_root(adapter_identity),
    }

    embedding_root = root / "special_token_embeddings"
    if embedding_root.is_symlink():
        raise ArtifactContractError(
            "inference payload component root cannot be a symlink",
            code="checkpoint.inference_payload_component_invalid",
            context={"path": str(embedding_root)},
        )
    if embedding_root.exists():
        embedding_identity = inspect_special_token_embedding_delta_payload(
            embedding_root,
            expected_base_model_path=expected_base_model_path,
            expected_base_config_sha256=expected_base_config_sha256,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
        )
        embedding = {
            "status": "present",
            "relative_root": "special_token_embeddings",
            "files": _inventory_files(embedding_root),
            "inspector_identity": _without_root(embedding_identity),
        }
    else:
        embedding = {
            "status": "absent",
            "relative_root": "special_token_embeddings",
            "files": [],
            "inspector_identity": None,
        }
    return {
        "schema": INFERENCE_PAYLOAD_MANIFEST_SCHEMA,
        "schema_version": 1,
        "adapter": adapter,
        "special_token_embedding_delta": embedding,
    }


def _inventory_files(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ArtifactContractError(
            "inference payload component must be a real directory",
            code="checkpoint.inference_payload_component_invalid",
            context={"path": str(root)},
        )
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArtifactContractError(
                "inference payload components cannot contain symlinks",
                code="checkpoint.inference_payload_component_invalid",
                context={"path": str(path)},
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactContractError(
                "inference payload components can contain only ordinary files",
                code="checkpoint.inference_payload_component_invalid",
                context={"path": str(path)},
            )
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ArtifactContractError(
            "inference payload component contains no ordinary files",
            code="checkpoint.inference_payload_component_invalid",
            context={"path": str(root)},
        )
    return files


def _validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
        raise ArtifactContractError(
            "inference payload manifest has an invalid field set",
            code="checkpoint.inference_payload_manifest_invalid",
        )
    normalized = dict(value)
    if (
        normalized["schema"] != INFERENCE_PAYLOAD_MANIFEST_SCHEMA
        or normalized["schema_version"] != 1
        or not _is_sha256(normalized["aggregate_digest"])
    ):
        raise ArtifactContractError(
            "inference payload manifest has an invalid schema or digest",
            code="checkpoint.inference_payload_manifest_invalid",
        )
    body = {key: normalized[key] for key in normalized if key != "aggregate_digest"}
    if _sha256_json(body) != normalized["aggregate_digest"]:
        raise ArtifactContractError(
            "inference payload manifest aggregate digest does not match its body",
            code="checkpoint.inference_payload_manifest_invalid",
        )
    return normalized


def _validate_publication_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PUBLICATION_FIELDS:
        raise ArtifactContractError(
            "inference payload publication identity has an invalid field set",
            code="checkpoint.inference_payload_identity_invalid",
        )
    normalized = dict(value)
    if (
        normalized["schema"] != INFERENCE_PAYLOAD_PUBLICATION_SCHEMA
        or normalized["schema_version"] != 2
        or normalized["manifest_relative_path"] != INFERENCE_PAYLOAD_MANIFEST_NAME
        or not _is_sha256(normalized["manifest_file_sha256"])
        or not _is_sha256(normalized["aggregate_digest"])
    ):
        raise ArtifactContractError(
            "inference payload publication identity is malformed",
            code="checkpoint.inference_payload_identity_invalid",
        )
    return normalized


def _without_root(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "root"}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "INFERENCE_PAYLOAD_MANIFEST_NAME",
    "INFERENCE_PAYLOAD_MANIFEST_SCHEMA",
    "INFERENCE_PAYLOAD_PUBLICATION_SCHEMA",
    "admit_inference_checkpoint_payload_identity",
    "build_inference_checkpoint_payload_identity",
    "load_inference_checkpoint_payload_manifest",
    "write_inference_checkpoint_payload_manifest",
]
