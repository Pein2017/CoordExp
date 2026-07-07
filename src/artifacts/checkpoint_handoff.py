"""Read-only checkpoint handoff identity validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


HANDOFF_GATES = frozenset({"handoff", "eval", "production"})


def validate_checkpoint_handoff(
    *,
    run_root: str | Path,
    checkpoint_ref: str | Path | None,
    gate: str = "handoff",
) -> dict[str, Any]:
    """Validate a checkpoint handoff artifact without modifying it."""

    if gate not in HANDOFF_GATES:
        raise ValueError(f"unsupported checkpoint handoff gate: {gate!r}")
    run_dir = Path(run_root).expanduser().resolve()
    missing: list[str] = []
    mismatches: list[str] = []
    resolved = _resolve_handoff_reference(
        run_dir=run_dir,
        checkpoint_ref=checkpoint_ref,
        missing=missing,
    )
    if resolved.handoff_path is None or resolved.handoff is None:
        return _verdict(
            status="hold",
            gate=gate,
            checkpoint_id=resolved.checkpoint_id,
            composition_mode="legacy_manual",
            handoff_path=None,
            missing=missing,
            mismatches=mismatches,
        )

    handoff = resolved.handoff
    checkpoint_id = str(handoff.get("checkpoint_id") or resolved.checkpoint_id or "")
    adapter_identity = _validate_adapter_identity(
        handoff,
        run_dir=run_dir,
        missing=missing,
        mismatches=mismatches,
    )
    special_identity = _validate_special_token_embedding_identity(
        handoff,
        run_dir=run_dir,
        missing=missing,
        mismatches=mismatches,
    )
    if gate in {"eval", "production"}:
        eval_roots = handoff.get("accepted_eval_artifact_roots")
        if not isinstance(eval_roots, Sequence) or isinstance(eval_roots, (str, bytes)) or not eval_roots:
            missing.append("accepted_eval_artifact_roots")
    return _verdict(
        status="pass" if not missing and not mismatches else "hold",
        gate=gate,
        checkpoint_id=checkpoint_id,
        composition_mode="canonical_handoff",
        handoff_path=_relative_to_run(resolved.handoff_path, run_dir=run_dir),
        missing=missing,
        mismatches=mismatches,
        adapter_identity=adapter_identity,
        special_token_embedding_identity=special_identity,
        handoff_fingerprint=_sha256_file(resolved.handoff_path),
    )


def _validate_adapter_identity(
    handoff: Mapping[str, Any],
    *,
    run_dir: Path,
    missing: list[str],
    mismatches: list[str],
) -> Mapping[str, Any] | None:
    adapter = handoff.get("adapter")
    if not isinstance(adapter, Mapping) or adapter.get("enabled") is not True:
        return None
    identity = adapter.get("identity")
    if not isinstance(identity, Mapping):
        missing.append("adapter.identity")
        return None
    required = identity.get("required_files")
    if not isinstance(required, Mapping):
        missing.append("adapter.identity.required_files")
        return identity
    config_rel = required.get("adapter_config.json")
    weight_rel = (
        required.get("adapter_model.safetensors")
        or required.get("adapter_model.bin")
    )
    if not isinstance(config_rel, str) or not config_rel:
        missing.append("adapter.identity.required_files.adapter_config.json")
    if not isinstance(weight_rel, str) or not weight_rel:
        missing.append("adapter.identity.required_files.adapter_model")
    if isinstance(config_rel, str) and config_rel:
        _compare_file_hash(
            run_dir=run_dir,
            rel_path=config_rel,
            expected=identity.get("adapter_config_sha256"),
            field="adapter.adapter_config_sha256",
            missing=missing,
            mismatches=mismatches,
        )
    if isinstance(weight_rel, str) and weight_rel:
        _compare_file_hash(
            run_dir=run_dir,
            rel_path=weight_rel,
            expected=identity.get("adapter_model_sha256"),
            field="adapter.adapter_model_sha256",
            missing=missing,
            mismatches=mismatches,
        )
    if not identity.get("fingerprint"):
        missing.append("adapter.identity.fingerprint")
    return identity


def _validate_special_token_embedding_identity(
    handoff: Mapping[str, Any],
    *,
    run_dir: Path,
    missing: list[str],
    mismatches: list[str],
) -> Mapping[str, Any] | None:
    payload = handoff.get("special_token_embeddings")
    if not isinstance(payload, Mapping) or payload.get("enabled") is not True:
        return None
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        missing.append("special_token_embeddings.identity")
        return None
    metadata_rel = identity.get("metadata_path")
    tensor_rel = identity.get("tensor_path")
    if not isinstance(metadata_rel, str) or not metadata_rel:
        missing.append("special_token_embeddings.identity.metadata_path")
    if not isinstance(tensor_rel, str) or not tensor_rel:
        missing.append("special_token_embeddings.identity.tensor_path")
    if isinstance(metadata_rel, str) and metadata_rel:
        _compare_file_hash(
            run_dir=run_dir,
            rel_path=metadata_rel,
            expected=identity.get("metadata_sha256"),
            field="special_token_embeddings.metadata_sha256",
            missing=missing,
            mismatches=mismatches,
        )
    if isinstance(tensor_rel, str) and tensor_rel:
        _compare_file_hash(
            run_dir=run_dir,
            rel_path=tensor_rel,
            expected=identity.get("tensor_sha256"),
            field="special_token_embeddings.tensor_sha256",
            missing=missing,
            mismatches=mismatches,
        )
    for field in (
        "tensor_key",
        "tensor_shape",
        "tensor_dtype",
        "base_config_sha256",
        "tokenizer_sha256",
        "fingerprint",
    ):
        if identity.get(field) in (None, "", []):
            missing.append(f"special_token_embeddings.identity.{field}")
    return identity


def _compare_file_hash(
    *,
    run_dir: Path,
    rel_path: str,
    expected: Any,
    field: str,
    missing: list[str],
    mismatches: list[str],
) -> None:
    path = _resolve_run_relative(run_dir, rel_path)
    if not path.exists():
        missing.append(rel_path)
        return
    if not isinstance(expected, str) or not expected:
        missing.append(field)
        return
    if _sha256_file(path) != expected:
        mismatches.append(field)


class _ResolvedHandoff:
    def __init__(
        self,
        *,
        handoff_path: Path | None,
        handoff: Mapping[str, Any] | None,
        checkpoint_id: str | None = None,
    ) -> None:
        self.handoff_path = handoff_path
        self.handoff = handoff
        self.checkpoint_id = checkpoint_id


def _resolve_handoff_reference(
    *,
    run_dir: Path,
    checkpoint_ref: str | Path | None,
    missing: list[str],
) -> _ResolvedHandoff:
    if checkpoint_ref is None:
        missing.append("checkpoint_ref")
        return _ResolvedHandoff(handoff_path=None, handoff=None)
    path = _resolve_run_relative(run_dir, str(checkpoint_ref))
    if path.is_dir():
        handoff_path = path / "checkpoint_handoff.json"
        if handoff_path.exists():
            return _ResolvedHandoff(handoff_path=handoff_path, handoff=_read_json(handoff_path))
        metadata_path = path / "checkpoint.json"
        if metadata_path.exists():
            return _resolve_from_metadata(run_dir=run_dir, metadata_path=metadata_path, missing=missing)
        missing.append("checkpoint_handoff")
        return _ResolvedHandoff(handoff_path=None, handoff=None)
    if path.name == "checkpoint_handoff.json":
        if not path.exists():
            missing.append("checkpoint_handoff")
            return _ResolvedHandoff(handoff_path=None, handoff=None)
        return _ResolvedHandoff(handoff_path=path, handoff=_read_json(path))
    if path.name == "checkpoint.json":
        return _resolve_from_metadata(run_dir=run_dir, metadata_path=path, missing=missing)
    payload = _read_json_if_exists(path)
    if payload is None:
        missing.append("checkpoint_ref")
        return _ResolvedHandoff(handoff_path=None, handoff=None)
    handoff_ref = payload.get("handoff_path")
    if isinstance(handoff_ref, str) and handoff_ref:
        handoff_path = _resolve_run_relative(run_dir, handoff_ref)
        if handoff_path.exists():
            return _ResolvedHandoff(handoff_path=handoff_path, handoff=_read_json(handoff_path))
    metadata_ref = payload.get("metadata_path")
    if isinstance(metadata_ref, str) and metadata_ref:
        return _resolve_from_metadata(
            run_dir=run_dir,
            metadata_path=_resolve_run_relative(run_dir, metadata_ref),
            missing=missing,
        )
    missing.append("checkpoint_handoff")
    return _ResolvedHandoff(
        handoff_path=None,
        handoff=None,
        checkpoint_id=None if not isinstance(payload.get("checkpoint_id"), str) else payload["checkpoint_id"],
    )


def _resolve_from_metadata(
    *,
    run_dir: Path,
    metadata_path: Path,
    missing: list[str],
) -> _ResolvedHandoff:
    metadata = _read_json_if_exists(metadata_path)
    checkpoint_id = None if metadata is None else str(metadata.get("checkpoint_id") or "")
    if metadata is None:
        missing.append("checkpoint_metadata")
        return _ResolvedHandoff(handoff_path=None, handoff=None)
    handoff_ref = metadata.get("checkpoint_handoff")
    if not isinstance(handoff_ref, str) or not handoff_ref:
        missing.append("checkpoint_handoff")
        return _ResolvedHandoff(
            handoff_path=None,
            handoff=None,
            checkpoint_id=checkpoint_id,
        )
    handoff_path = _resolve_run_relative(run_dir, handoff_ref)
    if not handoff_path.exists():
        missing.append("checkpoint_handoff")
        return _ResolvedHandoff(
            handoff_path=None,
            handoff=None,
            checkpoint_id=checkpoint_id,
        )
    return _ResolvedHandoff(
        handoff_path=handoff_path,
        handoff=_read_json(handoff_path),
        checkpoint_id=checkpoint_id,
    )


def _verdict(
    *,
    status: str,
    gate: str,
    checkpoint_id: str | None,
    composition_mode: str,
    handoff_path: str | None,
    missing: Sequence[str],
    mismatches: Sequence[str],
    adapter_identity: Mapping[str, Any] | None = None,
    special_token_embedding_identity: Mapping[str, Any] | None = None,
    handoff_fingerprint: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "gate": gate,
        "checkpoint_id": checkpoint_id,
        "composition_mode": composition_mode,
        "handoff_path": handoff_path,
        "handoff_fingerprint": handoff_fingerprint,
        "adapter_identity": None if adapter_identity is None else dict(adapter_identity),
        "special_token_embedding_identity": (
            None
            if special_token_embedding_identity is None
            else dict(special_token_embedding_identity)
        ),
        "missing": sorted(set(missing)),
        "mismatches": sorted(set(mismatches)),
    }


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


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint handoff JSON must contain an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["HANDOFF_GATES", "validate_checkpoint_handoff"]
