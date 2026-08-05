#!/usr/bin/env python3
"""Freeze the production runtime identity used by sorted owner-basin inputs.

This is deliberately a metadata-only builder.  It validates the final Task-0
receipt and its bound HF manifest and checkpoint files without importing torch,
transformers, or loading a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "sorted-owner-basin-runtime-identity.v1"
TASK0_SCHEMA_VERSION = "sorted-owner-basin-task0-execution-receipt.v2"
COORDINATE_MIN = 0
COORDINATE_MAX = 999
COORDINATE_TOKEN_ID_START = 151670
COORDINATE_TOKEN_ID_END_EXCLUSIVE = 152670
MODEL_VOCAB_SIZE = 152670
WRAPPER_TOKEN_IDS = {
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
}
SCHEMA_TOKENS = {
    "object_ref_start_token_id": 151646,
    "object_ref_end_token_id": 151647,
    "box_start_token_id": 151648,
    "box_end_token_id": 151649,
}
_COMPONENTS = frozenset({"base_model_path", "adapter_path", "embedding_delta_path"})
_REQUIRED_COMPONENT_FILES = {
    "base_model_path": frozenset(
        {
            "added_tokens.json",
            "config.json",
            "coord_tokens.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    ),
    "adapter_path": frozenset({"adapter_config.json", "adapter_model.safetensors"}),
    "embedding_delta_path": frozenset(
        {
            "special_token_embeddings.json",
            "special_token_embeddings.safetensors",
        }
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeIdentityError(ValueError):
    """Raised before output when production identity evidence is not admissible."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeIdentityError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeIdentityError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeIdentityError(f"{label} must be a non-empty trimmed string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeIdentityError(f"{label} must be an integer")
    return value


def _digest(value: Any, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        raise RuntimeIdentityError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise RuntimeIdentityError(f"{label} is not valid JSON") from exc


def _resolved_file(path: str | Path, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeIdentityError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise RuntimeIdentityError(f"{label} must be a regular file")
    return resolved


def _content_digest(document: Mapping[str, Any]) -> str:
    declared = _digest(
        document.get("execution_receipt_content_sha256"),
        "execution receipt.execution_receipt_content_sha256",
    )
    content = {
        key: value
        for key, value in document.items()
        if key != "execution_receipt_content_sha256"
    }
    if sha256_json(content) != declared:
        raise RuntimeIdentityError(
            "execution receipt content digest is stale or has been tampered with"
        )
    return declared


def _file_record(value: Any, label: str) -> dict[str, Any]:
    record = _mapping(value, label)
    path = _resolved_file(_string(record.get("path"), f"{label}.path"), label)
    size = _integer(record.get("bytes"), f"{label}.bytes")
    digest = _digest(record.get("sha256"), f"{label}.sha256")
    if path.stat().st_size != size:
        raise RuntimeIdentityError(f"{label} byte size is stale")
    if sha256_file(path) != digest:
        raise RuntimeIdentityError(
            f"{label} SHA-256 is stale or file was tampered with"
        )
    return {"bytes": size, "path": str(path), "sha256": digest}


def _validate_manifest_binding(
    receipt: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    inputs = _mapping(receipt.get("inputs"), "execution receipt.inputs")
    bound_files = _sequence(inputs.get("bound_files"), "execution receipt bound_files")
    matches = [
        item
        for item in bound_files
        if isinstance(item, Mapping)
        and item.get("role") == "production_rp_1_10_manifest"
    ]
    if len(matches) != 1:
        raise RuntimeIdentityError(
            "execution receipt must bind exactly one production HF manifest"
        )
    bound = _file_record(matches[0], "bound production manifest")
    if bound["path"] != str(manifest_path):
        raise RuntimeIdentityError(
            "supplied production manifest is not the one bound by Task-0"
        )
    return bound


def _validate_component_files(
    receipt: Mapping[str, Any], components: Mapping[str, Any]
) -> list[dict[str, Any]]:
    identity = _mapping(
        receipt.get("model_tokenizer_processor_identity"),
        "execution receipt.model_tokenizer_processor_identity",
    )
    raw_records = _sequence(
        identity.get("model_component_files"), "model component files"
    )
    bound_files = _sequence(
        _mapping(receipt.get("inputs"), "execution receipt.inputs").get("bound_files"),
        "execution receipt bound_files",
    )
    bound_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in bound_files:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        if isinstance(role, str) and role.startswith("model_component:"):
            key = (role.removeprefix("model_component:"), str(item.get("path")))
            if key in bound_by_key:
                raise RuntimeIdentityError(f"duplicate bound component file {key!r}")
            bound_by_key[key] = item

    observed_relative: dict[str, set[str]] = {key: set() for key in _COMPONENTS}
    validated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_records):
        label = f"model component file {index}"
        record = _mapping(raw, label)
        component = _string(record.get("component"), f"{label}.component")
        if component not in _COMPONENTS:
            raise RuntimeIdentityError(
                f"{label} has unsupported component {component!r}"
            )
        relative = _string(record.get("relative_path"), f"{label}.relative_path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeIdentityError(f"{label}.relative_path is unsafe")
        root = Path(_string(components.get(component), f"model_components.{component}"))
        try:
            root = root.expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeIdentityError(
                f"model component {component} is missing"
            ) from exc
        expected_path = (root / relative_path).resolve(strict=False)
        path_text = _string(record.get("path"), f"{label}.path")
        if str(expected_path) != str(
            Path(path_text).expanduser().resolve(strict=False)
        ):
            raise RuntimeIdentityError(
                f"{label} path is outside its declared component"
            )
        validated_file = _file_record(record, label)
        if validated_file["path"] in seen_paths:
            raise RuntimeIdentityError(f"duplicate model component path {path_text!r}")
        seen_paths.add(validated_file["path"])
        observed_relative[component].add(relative)

        bound = bound_by_key.get((component, path_text))
        if bound is None:
            raise RuntimeIdentityError(
                f"{label} is not bound by execution receipt inputs"
            )
        bound_projection = {
            "bytes": bound.get("bytes"),
            "path": bound.get("path"),
            "sha256": bound.get("sha256"),
        }
        if bound_projection != {
            "bytes": record.get("bytes"),
            "path": record.get("path"),
            "sha256": record.get("sha256"),
        }:
            raise RuntimeIdentityError(f"{label} disagrees with its bound input record")
        validated.append(
            {
                **validated_file,
                "component": component,
                "relative_path": relative,
            }
        )

    if len(bound_by_key) != len(validated):
        raise RuntimeIdentityError(
            "bound component files disagree with identity inventory"
        )
    for component, required in _REQUIRED_COMPONENT_FILES.items():
        missing = required - observed_relative[component]
        if missing:
            raise RuntimeIdentityError(
                f"model component {component} is missing required files {sorted(missing)}"
            )
    return sorted(
        validated, key=lambda item: (item["component"], item["relative_path"])
    )


def _validate_tokenizer_files(
    components: Mapping[str, Any], component_files: Sequence[Mapping[str, Any]]
) -> None:
    base = Path(str(components["base_model_path"])).resolve()
    coord_tokens = json.loads((base / "coord_tokens.json").read_text(encoding="utf-8"))
    expected_coord_strings = [f"<|coord_{index}|>" for index in range(1000)]
    if coord_tokens != ["<|coord_*|>", *expected_coord_strings]:
        raise RuntimeIdentityError(
            "coord_tokens.json is not the exact 0..999 vocabulary"
        )
    added_tokens = _mapping(
        json.loads((base / "added_tokens.json").read_text(encoding="utf-8")),
        "added_tokens.json",
    )
    expected_added = {
        **WRAPPER_TOKEN_IDS,
        **{
            token: COORDINATE_TOKEN_ID_START + index
            for index, token in enumerate(expected_coord_strings)
        },
    }
    for token, token_id in expected_added.items():
        if added_tokens.get(token) != token_id:
            raise RuntimeIdentityError(
                f"added_tokens.json has wrong token id for {token!r}"
            )

    files_by_key = {
        (item["component"], item["relative_path"]): item for item in component_files
    }
    delta = Path(str(components["embedding_delta_path"])).resolve()
    metadata = _mapping(
        json.loads(
            (delta / "special_token_embeddings.json").read_text(encoding="utf-8")
        ),
        "special token embedding metadata",
    )
    expected_ids = [*WRAPPER_TOKEN_IDS.values(), *range(151670, 152670)]
    expected_strings = [*WRAPPER_TOKEN_IDS.keys(), *expected_coord_strings]
    if metadata.get("token_ids") != expected_ids:
        raise RuntimeIdentityError(
            "embedding delta token_ids are not the required 1004 ids"
        )
    if metadata.get("token_strings") != expected_strings:
        raise RuntimeIdentityError(
            "embedding delta token_strings are not the required 1004 tokens"
        )
    if (
        metadata.get("tokenizer_sha256")
        != files_by_key[("base_model_path", "tokenizer.json")]["sha256"]
    ):
        raise RuntimeIdentityError("embedding delta tokenizer SHA-256 is stale")
    if (
        metadata.get("base_config_sha256")
        != files_by_key[("base_model_path", "config.json")]["sha256"]
    ):
        raise RuntimeIdentityError("embedding delta base config SHA-256 is stale")


def _validate_token_identity(tokenizer: Mapping[str, Any]) -> None:
    expected = {
        "coord_token_count": 1000,
        "coord_token_id_min": COORDINATE_TOKEN_ID_START,
        "coord_token_id_max": COORDINATE_TOKEN_ID_END_EXCLUSIVE - 1,
        "coord_token_ids_contiguous": True,
        "required_token_count": 1004,
        "tokenizer_vocab_size": MODEL_VOCAB_SIZE,
        "wrapper_token_ids": WRAPPER_TOKEN_IDS,
    }
    for field, expected_value in expected.items():
        if tokenizer.get(field) != expected_value:
            raise RuntimeIdentityError(
                f"production tokenizer identity has wrong {field}"
            )


def _validate_model_identity(
    manifest: Mapping[str, Any], components: Mapping[str, Any]
) -> Mapping[str, Any]:
    model = _mapping(manifest.get("model_identity"), "manifest.model_identity")
    if model.get("family") != "base-plus-adapter-plus-delta":
        raise RuntimeIdentityError(
            "production model family must be base-plus-adapter-plus-delta"
        )
    base = _mapping(model.get("base"), "model identity.base")
    adapter = _mapping(model.get("adapter"), "model identity.adapter")
    embedding = _mapping(model.get("embedding_delta"), "model identity.embedding_delta")
    qwen = _mapping(model.get("qwen"), "model identity.qwen")
    if base.get("path") != components["base_model_path"]:
        raise RuntimeIdentityError("base model path disagrees across sources")
    if (
        adapter.get("enabled") is not True
        or adapter.get("status") != "validated"
        or adapter.get("adapter_path") != components["adapter_path"]
        or adapter.get("base_model_path") != components["base_model_path"]
        or adapter.get("active_adapters") != ["default"]
        or adapter.get("missing_keys") != []
        or adapter.get("unexpected_keys") != []
    ):
        raise RuntimeIdentityError("production adapter identity is not fully validated")
    embedding_identity = _mapping(
        embedding.get("identity"), "model identity.embedding_delta.identity"
    )
    metadata = _mapping(embedding_identity.get("metadata"), "embedding delta metadata")
    load = _mapping(embedding.get("load"), "embedding delta load")
    if (
        embedding.get("status") != "loaded"
        or embedding_identity.get("base_model_path") != components["base_model_path"]
        or embedding_identity.get("delta_path") != components["embedding_delta_path"]
        or load.get("loaded") is not True
    ):
        raise RuntimeIdentityError("production embedding delta is not loaded")
    if any(
        value != "float32"
        for value in (
            metadata.get("tensor_dtype"),
            load.get("runtime_tensor_dtype"),
            load.get("source_tensor_dtype"),
            load.get("tensor_dtype"),
        )
    ):
        raise RuntimeIdentityError("only an FP32 embedding delta is supported")
    if metadata.get("tensor_shape") != [1004, 2048] or load.get("tensor_shape") != [
        1004,
        2048,
    ]:
        raise RuntimeIdentityError("embedding delta shape must be [1004, 2048]")
    if qwen.get("config_dtype") != "torch.float32":
        raise RuntimeIdentityError("only the FP32 production checkpoint is supported")
    if (
        qwen.get("model_type") != "qwen3_vl"
        or qwen.get("text_vocab_size") != MODEL_VOCAB_SIZE
        or qwen.get("text_hidden_size") != 2048
        or qwen.get("tie_word_embeddings") is not True
    ):
        raise RuntimeIdentityError("Qwen model identity is incompatible")
    return model


def _validate_sources(
    receipt_path: Path, manifest_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]], dict[str, Any]]:
    receipt = _read_json(receipt_path, "execution receipt")
    manifest = _read_json(manifest_path, "production manifest")
    if receipt.get("schema_version") != TASK0_SCHEMA_VERSION:
        raise RuntimeIdentityError("execution receipt schema_version is not recognized")
    receipt_content_digest = _content_digest(receipt)
    if receipt.get("execution_status") != "completed":
        raise RuntimeIdentityError("Task-0 execution receipt is not completed")
    bound_manifest = _validate_manifest_binding(receipt, manifest_path)
    if manifest.get("terminal_status") != "completed":
        raise RuntimeIdentityError("production manifest is not completed")
    if (
        manifest.get("backend") != "hf"
        or manifest.get("backend_mode") != "generate"
        or manifest.get("response_family") != "hf"
    ):
        raise RuntimeIdentityError(
            "only the production HF generate backend is supported"
        )

    identity = _mapping(
        receipt.get("model_tokenizer_processor_identity"),
        "execution receipt.model_tokenizer_processor_identity",
    )
    components = _mapping(identity.get("model_components"), "model components")
    if set(components) != _COMPONENTS:
        raise RuntimeIdentityError("model components must be base, adapter, and delta")
    component_files = _validate_component_files(receipt, components)
    _validate_tokenizer_files(components, component_files)

    session = _mapping(manifest.get("backend_session"), "manifest.backend_session")
    expected_exact = dict(session)
    expected_settings = dict(
        _mapping(expected_exact.get("effective_settings"), "backend effective settings")
    )
    expected_settings.pop("performance", None)
    expected_exact["effective_settings"] = expected_settings
    exact = _mapping(identity.get("exact_identity"), "execution exact identity")
    if exact != expected_exact:
        raise RuntimeIdentityError(
            "Task-0 exact runtime identity disagrees with production manifest"
        )
    exact_digest = sha256_json(exact)
    if identity.get("model_identity_sha256") != exact_digest:
        raise RuntimeIdentityError("Task-0 exact identity digest is stale")

    for field in ("model_identity", "tokenizer_identity", "processor_identity"):
        if manifest.get(field) != session.get(field):
            raise RuntimeIdentityError(
                f"manifest {field} disagrees with backend session"
            )
    if manifest.get("adapter_identity") != manifest.get("model_identity", {}).get(
        "adapter"
    ):
        raise RuntimeIdentityError("manifest adapter identity is stale")
    if manifest.get("embedding_delta_identity") != manifest.get(
        "model_identity", {}
    ).get("embedding_delta"):
        raise RuntimeIdentityError("manifest embedding delta identity is stale")

    tokenizer = _mapping(manifest.get("tokenizer_identity"), "tokenizer identity")
    _validate_token_identity(tokenizer)
    if identity.get("tokenizer_identity") != tokenizer:
        raise RuntimeIdentityError("Task-0 tokenizer identity is stale")
    processor = _mapping(manifest.get("processor_identity"), "processor identity")
    if identity.get("processor_identity") != processor:
        raise RuntimeIdentityError("Task-0 processor identity is stale")
    model = _validate_model_identity(manifest, components)
    model_fingerprint = sha256_json(model)
    if manifest.get("model_identity_fingerprint") != model_fingerprint:
        raise RuntimeIdentityError("manifest model identity fingerprint is stale")
    processor_fingerprint = sha256_json(processor)
    if manifest.get("processor_identity_fingerprint") != processor_fingerprint:
        raise RuntimeIdentityError("manifest processor identity fingerprint is stale")
    metadata_path = Path(str(components["embedding_delta_path"])) / (
        "special_token_embeddings.json"
    )
    metadata = _mapping(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        "special token embedding metadata",
    )
    manifest_metadata = _mapping(
        _mapping(
            _mapping(model.get("embedding_delta"), "embedding delta").get("identity"),
            "embedding delta identity",
        ).get("metadata"),
        "embedding delta metadata",
    )
    if metadata != manifest_metadata:
        raise RuntimeIdentityError("embedding delta metadata disagrees with manifest")

    coordinate_contract = _mapping(
        receipt.get("coordinate_contract"), "execution coordinate contract"
    )
    if (
        coordinate_contract.get("coordinate_bin_min") != COORDINATE_MIN
        or coordinate_contract.get("coordinate_bin_max") != COORDINATE_MAX
        or coordinate_contract.get("source_token_grammar")
        != "<|coord_N|> with integer N in [0, 999]"
    ):
        raise RuntimeIdentityError("Task-0 coordinate contract is incompatible")
    source_binding = {
        "task0_execution_receipt": {
            "content_sha256": receipt_content_digest,
            "file_sha256": sha256_file(receipt_path),
            "path": str(receipt_path),
            "schema_version": TASK0_SCHEMA_VERSION,
        },
        "production_run_manifest": {
            "file_sha256": bound_manifest["sha256"],
            "path": str(manifest_path),
            "terminal_status": "completed",
        },
    }
    return receipt, manifest, component_files, source_binding


def build_sorted_owner_basin_runtime_identity(
    *,
    execution_receipt: str | Path,
    production_manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate production evidence and write one deterministic identity receipt."""

    receipt_path = _resolved_file(execution_receipt, "execution receipt")
    manifest_path = _resolved_file(production_manifest, "production manifest")
    receipt, manifest, component_files, sources = _validate_sources(
        receipt_path, manifest_path
    )
    source_identity = _mapping(
        receipt.get("model_tokenizer_processor_identity"),
        "execution identity",
    )
    components = dict(_mapping(source_identity.get("model_components"), "components"))
    tokenizer_files = [
        item for item in component_files if item["component"] == "base_model_path"
    ]
    tokenizer_source = {
        "component_files": tokenizer_files,
        "path": components["base_model_path"],
        "tokenizer_identity": manifest["tokenizer_identity"],
    }
    model_source = {
        "component_files": component_files,
        "model_identity": manifest["model_identity"],
        "model_identity_fingerprint": manifest["model_identity_fingerprint"],
    }
    exact = _mapping(source_identity.get("exact_identity"), "exact identity")
    runtime_source = {
        "backend": exact["backend"],
        "backend_mode": exact["backend_mode"],
        "backend_version": exact["backend_version"],
        "effective_settings": exact["effective_settings"],
        "generation_config_fingerprint": exact["generation_config_fingerprint"],
        "likelihood_semantics": exact["likelihood_semantics"],
        "precision": "float32",
        "processor_identity": exact["processor_identity"],
        "processor_identity_fingerprint": manifest["processor_identity_fingerprint"],
        "resolved_config_fingerprints": manifest["resolved_config_fingerprints"],
        "response_family": exact["response_family"],
    }
    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "sources": sources,
        "tokenizer": {
            "path": components["base_model_path"],
            "identity_sha256": sha256_json(tokenizer_source),
            "identity_source": tokenizer_source,
        },
        "model": {
            "identity_sha256": sha256_json(model_source),
            "identity_source": model_source,
        },
        "runtime": {
            "identity_sha256": sha256_json(runtime_source),
            "identity_source": runtime_source,
        },
        "coordinate_vocabulary": {
            "coordinate_min": COORDINATE_MIN,
            "coordinate_max": COORDINATE_MAX,
            "token_id_start": COORDINATE_TOKEN_ID_START,
            "token_id_end_exclusive": COORDINATE_TOKEN_ID_END_EXCLUSIVE,
        },
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "schema_tokens": SCHEMA_TOKENS,
    }
    document = {**content, "receipt_digest": sha256_json(content)}
    output_path = Path(output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(document) + b"\n"
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if not output_path.is_file() or output_path.read_bytes() != encoded:
            raise RuntimeIdentityError(
                "output already exists with different content; refusing to overwrite"
            ) from None
    return document


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", required=True)
    parser.add_argument("--production-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    document = build_sorted_owner_basin_runtime_identity(
        execution_receipt=args.execution_receipt,
        production_manifest=args.production_manifest,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve(strict=False)),
                "receipt_digest": document["receipt_digest"],
                "schema_version": document["schema_version"],
                "status": document["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
