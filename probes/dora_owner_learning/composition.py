"""Shared effective-model composition evidence; frozen receipt semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

def _resolved_path(value: Any) -> str | None:
    return str(Path(value).resolve()) if isinstance(value, str) and value else None

def loaded_composition_evidence(
    *,
    loaded_identity: Mapping[str, Any],
    expected_base: str,
    expected_adapter: str,
    expected_embedding: Mapping[str, Any],
    inspected_embedding: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare effective composition across the two embedding receipt schemas.

    The Swift and infras inspectors name their receipt versions differently and
    therefore compute different envelope fingerprints. Payload files plus the
    complete semantic identity are the shared, exact identity surface.
    """

    model_identity = loaded_identity.get("model_identity", {})
    model_identity = model_identity if isinstance(model_identity, Mapping) else {}
    base = model_identity.get("base", {})
    base = base if isinstance(base, Mapping) else {}
    adapter = model_identity.get("adapter", {})
    adapter = adapter if isinstance(adapter, Mapping) else {}
    adapter_state = adapter.get("adapter_state_evidence", {})
    adapter_state = adapter_state if isinstance(adapter_state, Mapping) else {}
    embedding_delta = model_identity.get("embedding_delta", {})
    embedding_delta = embedding_delta if isinstance(embedding_delta, Mapping) else {}
    embedding_load = embedding_delta.get("load", {})
    embedding_load = embedding_load if isinstance(embedding_load, Mapping) else {}
    settings = loaded_identity.get("effective_settings", {})
    settings = settings if isinstance(settings, Mapping) else {}
    dtype = settings.get("observed_model_dtype", {})
    dtype = dtype if isinstance(dtype, Mapping) else {}

    actual_files = inspected_embedding.get("files")
    expected_files = expected_embedding.get("files")
    actual_semantics = inspected_embedding.get("semantic_identity")
    expected_semantics = expected_embedding.get("semantic_identity")
    checks = {
        "base_path": _resolved_path(base.get("path")) == _resolved_path(expected_base),
        "adapter_path": _resolved_path(adapter.get("adapter_path"))
        == _resolved_path(expected_adapter),
        "adapter_unmerged": adapter.get("merged_adapters") == [],
        "adapter_state_checked": adapter_state.get("state_checked") is True,
        "embedding_loaded": embedding_delta.get("status") == "loaded"
        and embedding_load.get("loaded") is True,
        "embedding_root": _resolved_path(inspected_embedding.get("root"))
        == _resolved_path(expected_embedding.get("root")),
        "embedding_kind": inspected_embedding.get("kind")
        == expected_embedding.get("kind"),
        "embedding_file_count": inspected_embedding.get("file_count")
        == expected_embedding.get("file_count"),
        "embedding_payload_files": isinstance(actual_files, list)
        and actual_files == expected_files,
        "embedding_semantic_identity": isinstance(actual_semantics, Mapping)
        and actual_semantics == expected_semantics,
        "attention_implementation": settings.get("observed_attn_implementation") == "sdpa",
        "parameter_dtype": dtype.get("parameter_dtype_names") == ["torch.float32"],
    }

    def embedding_summary(identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "root": _resolved_path(identity.get("root")),
            "kind": identity.get("kind"),
            "version": identity.get("version"),
            "fingerprint": identity.get("fingerprint"),
            "file_count": identity.get("file_count"),
            "files_sha256": digest_json(identity.get("files")),
            "semantic_identity_sha256": digest_json(identity.get("semantic_identity")),
        }

    return {
        "schema": "repeat_recovery_train.loaded_composition_check.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "actual": {
            "base_path": _resolved_path(base.get("path")),
            "adapter_path": _resolved_path(adapter.get("adapter_path")),
            "merged_adapters": adapter.get("merged_adapters"),
            "adapter_state_checked": adapter_state.get("state_checked"),
            "embedding_load_status": embedding_delta.get("status"),
            "embedding_load_confirmed": embedding_load.get("loaded"),
            "embedding": embedding_summary(inspected_embedding),
            "attention_implementation": settings.get("observed_attn_implementation"),
            "parameter_dtype_names": dtype.get("parameter_dtype_names"),
        },
        "expected": {
            "base_path": _resolved_path(expected_base),
            "adapter_path": _resolved_path(expected_adapter),
            "merged_adapters": [],
            "adapter_state_checked": True,
            "embedding_load_status": "loaded",
            "embedding_load_confirmed": True,
            "embedding": embedding_summary(expected_embedding),
            "attention_implementation": "sdpa",
            "parameter_dtype_names": ["torch.float32"],
        },
        "embedding_identity_basis": (
            "exact root, kind, file count, payload file identities, and complete semantic "
            "identity; receipt version labels and their envelope fingerprints are recorded "
            "but are not cross-schema comparable"
        ),
    }
