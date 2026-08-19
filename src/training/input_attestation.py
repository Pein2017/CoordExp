"""Model-free, content-bound inputs for the Wave 7 r5 request producer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

from src.artifacts.identity import (
    MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
    assert_model_weight_identity_equal,
    base_model_weight_identity_with_execution_policy,
    canonical_json_bytes,
    validate_model_weight_identity,
)
from src.config.loader import load_train_config
from src.losses import build_token_vocabulary_groups
from src.qwen import load_qwen_components
from src.training.pack_cache import (
    PACKING_CACHE_VERSION,
    build_packing_cache_determinants,
    cache_dir_for_fingerprint,
    load_cache_manifest,
    packing_cache_fingerprint_from_determinants,
)


CACHE_ATTESTATION_SCHEMA = "coordexp-swift-wave7-r5-cache-input-attestation-v1"
MODEL_ATTESTATION_SCHEMA = "coordexp-swift-wave7-r5-model-input-attestation-v2"
PREPARATION_RECEIPT_SCHEMA = "coordexp-swift-pack-cache-preparation-receipt-v1"
RUN_ROLES = ("uninterrupted", "interrupted_parent", "resume_child")
SPLITS = ("train", "eval.forward")
_SPLIT_RECEIPT_KEYS = {"train": "train", "eval.forward": "eval"}
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _native_preparation_receipt_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_bound(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, owner: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{owner} fields are not exact: missing={sorted(expected - observed)!r} "
            f"unknown={sorted(observed - expected)!r}"
        )


def _strict_json(path: Path, *, owner: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    info = resolved.lstat()
    if not stat.S_ISREG(info.st_mode) or resolved.is_symlink():
        raise ValueError(f"{owner} must be a regular non-symlink file")
    if info.st_size <= 0 or info.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{owner} exceeds its strict JSON byte bound")
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{owner} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{owner} root must be an object")
    canonical_json_bytes(payload)
    return payload


def _resolved_configs(
    config_paths: Mapping[str, str | Path], *, expected_model_root: str | Path | None
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if set(config_paths) != set(RUN_ROLES):
        raise ValueError("config_paths must contain exactly the three Wave 7 run roles")
    resolved = {role: load_train_config(config_paths[role]) for role in RUN_ROLES}
    projections = {
        role: {
            key: deepcopy(value)
            for key, value in resolved[role].config_dict.items()
            if key not in {"run", "resume"}
        }
        for role in RUN_ROLES
    }
    reference = projections[RUN_ROLES[0]]
    if any(projections[role] != reference for role in RUN_ROLES[1:]):
        raise ValueError(
            "training configs have semantic drift outside accepted run/resume differences"
        )
    roots = {
        Path(item.config.model.base_model).expanduser().resolve()
        for item in resolved.values()
    }
    if len(roots) != 1:
        raise ValueError("training configs do not resolve one explicit model root")
    model_root = next(iter(roots))
    if (
        expected_model_root is not None
        and model_root != Path(expected_model_root).expanduser().resolve()
    ):
        raise ValueError("training config model root differs from expected_model_root")
    inventory = {
        "paths": {role: str(resolved[role].entry_config_path) for role in RUN_ROLES},
        "fingerprints": {role: resolved[role].fingerprint for role in RUN_ROLES},
        "semantic_projection_sha256": _sha256_json(reference),
    }
    return resolved, inventory, model_root


def _derive_cache_inputs(
    resolved: Mapping[str, Any], *, cache_root: Path, model_root: Path
) -> dict[str, dict[str, Any]]:
    config = resolved[RUN_ROLES[0]].config
    if config.data.eval is None:
        raise ValueError("Wave 7 r5 input attestation requires eval.forward data")
    components = load_qwen_components(config, load_model=False)
    if (
        getattr(components, "model", None) is not None
        or getattr(components, "load_model", False) is not False
        or Path(components.base_model_path).expanduser().resolve() != model_root
    ):
        raise ValueError(
            "Qwen input components are not model-free or use the wrong root"
        )
    vocab_groups = build_token_vocabulary_groups(
        components.token_identity, tokenizer=components.tokenizer
    )
    datasets = {"train": config.data.train, "eval.forward": config.data.eval}
    result: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        determinants = build_packing_cache_determinants(
            config,
            components,
            dataset=datasets[split],
            split=split,
            vocab_groups=vocab_groups,
        )
        fingerprint = packing_cache_fingerprint_from_determinants(determinants)
        result[split] = {
            "determinants": determinants,
            "fingerprint": fingerprint,
            "cache_dir": cache_dir_for_fingerprint(cache_root, fingerprint),
        }
    return result


def _validate_preparation_receipt(
    path: Path,
    *,
    resolved: Mapping[str, Any],
    cache_root: Path,
    derived: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _strict_json(path, owner="cache preparation receipt")
    _exact_keys(
        payload,
        {
            "schema",
            "terminal_status",
            "config_path",
            "result",
            "failure",
            "receipt_sha256",
        },
        owner="cache preparation receipt",
    )
    body = dict(payload)
    observed_digest = body.pop("receipt_sha256")
    if not isinstance(
        observed_digest, str
    ) or observed_digest != _native_preparation_receipt_sha256(body):
        raise ValueError("cache preparation receipt hash is invalid")
    expected_config = str(resolved[RUN_ROLES[0]].entry_config_path)
    result = payload["result"]
    if (
        payload["schema"] != PREPARATION_RECEIPT_SCHEMA
        or payload["terminal_status"] != "completed"
        or payload["config_path"] != expected_config
        or payload["failure"] is not None
        or not isinstance(result, Mapping)
        or result.get("model_loaded") is not False
        or result.get("entry_config_path") != expected_config
        or result.get("resolved_config_fingerprint")
        != resolved[RUN_ROLES[0]].fingerprint
    ):
        raise ValueError(
            "cache preparation receipt is not the completed model-free run"
        )
    policies = result.get("policy_identities")
    cache_policy = policies.get("cache") if isinstance(policies, Mapping) else None
    root_receipt = (
        cache_policy.get("root") if isinstance(cache_policy, Mapping) else None
    )
    if not isinstance(root_receipt, Mapping) or root_receipt.get(
        "resolved_root"
    ) != str(cache_root):
        raise ValueError("cache preparation receipt binds the wrong cache root")
    if (
        cache_policy.get("train_fingerprint") != derived["train"]["fingerprint"]
        or cache_policy.get("eval_fingerprint")
        != derived["eval.forward"]["fingerprint"]
    ):
        raise ValueError("cache preparation receipt binds the wrong split fingerprint")
    for split in SPLITS:
        row = result.get(_SPLIT_RECEIPT_KEYS[split])
        expected = derived[split]
        if not isinstance(row, Mapping):
            raise ValueError(f"cache preparation receipt omits {split}")
        if (
            row.get("status") != "complete"
            or row.get("build_status") != "built"
            or row.get("format_version") != PACKING_CACHE_VERSION
            or row.get("fingerprint") != expected["fingerprint"]
            or Path(str(row.get("cache_dir", ""))).resolve() != expected["cache_dir"]
            or Path(str(row.get("manifest_path", ""))).resolve()
            != expected["cache_dir"] / "manifest.json"
            or isinstance(row.get("micro_step_count"), bool)
            or not isinstance(row.get("micro_step_count"), int)
            or int(row["micro_step_count"]) <= 0
        ):
            raise ValueError(f"cache preparation receipt {split} identity is invalid")
        phases = row.get("phase_receipt")
        if (
            not isinstance(phases, Mapping)
            or phases.get("cache_preparation", {}).get("status") != "completed"
            or phases.get("cache_publication", {}).get("status") != "completed"
            or phases.get("cache_admission", {}).get("status") != "completed"
            or phases.get("cache_admission", {}).get("verification_level") != "payloads"
        ):
            raise ValueError(
                f"cache preparation receipt {split} did not perform a fresh payload build"
            )
    binding = {
        "path": str(path.expanduser().resolve()),
        "file_sha256": _sha256_file(path.expanduser().resolve()),
        "receipt_sha256": observed_digest,
        "schema": PREPARATION_RECEIPT_SCHEMA,
        "terminal_status": "completed",
        "config_path": expected_config,
    }
    return dict(result), binding


def _chunk_payload_precheck(
    derived: Mapping[str, Mapping[str, Any]], *, max_cache_payload_bytes: int
) -> tuple[dict[str, dict[str, Any]], int]:
    total = 0
    raw: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        cache_dir = Path(derived[split]["cache_dir"])
        manifest = _strict_json(
            cache_dir / "manifest.json", owner=f"{split} cache manifest"
        )
        _validated_r5_materialization_policy(
            manifest.get("materialization"), split=split
        )
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"{split} cache manifest has no chunk declarations")
        split_bytes = 0
        for chunk in chunks:
            if not isinstance(chunk, Mapping) or not isinstance(chunk.get("path"), str):
                raise ValueError(f"{split} cache manifest has an invalid chunk")
            relative = Path(chunk["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"{split} cache chunk path escapes its cache directory"
                )
            chunk_path = cache_dir / relative
            current = chunk_path.parent
            while current != cache_dir:
                if current.is_symlink():
                    raise ValueError(
                        f"{split} cache chunk path traverses a symlink directory"
                    )
                if cache_dir not in current.parents:
                    raise ValueError(
                        f"{split} cache chunk path escapes its cache directory"
                    )
                current = current.parent
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(chunk_path, flags)
            try:
                info = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"{split} cache chunk is not a regular no-follow file")
            split_bytes += int(info.st_size)
            total += int(info.st_size)
            if total > max_cache_payload_bytes:
                raise ValueError("aggregate cache payload byte bound exceeded")
        raw[split] = {"manifest": manifest, "payload_bytes": split_bytes}
    return raw, total


def _native_split_identity(
    split: str,
    *,
    derived: Mapping[str, Any],
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    validation_seconds: float,
) -> dict[str, Any]:
    determinants = manifest["determinants"]
    materialization = _validated_r5_materialization_policy(
        manifest.get("materialization"), split=split
    )
    determinant_hashes = [
        {
            "name": row["name"],
            "content_identity_sha256": _sha256_json(row["content_identity"]),
            "owner_source_sha256": row["owner_source_identity"]["sha256"],
        }
        for row in determinants["determinants"]
    ]
    cache_dir = Path(derived["cache_dir"])
    return {
        "split": split,
        "cache_dir": str(cache_dir),
        "fingerprint": derived["fingerprint"],
        "manifest_path": str(cache_dir / "manifest.json"),
        "manifest_sha256": _sha256_file(cache_dir / "manifest.json"),
        "determinants_sha256": _sha256_json(determinants),
        "determinant_hashes": determinant_hashes,
        "chunk_sha256s": [str(row["sha256"]) for row in manifest["chunks"]],
        "chunk_count": len(manifest["chunks"]),
        "micro_step_count": int(manifest["micro_step_count"]),
        "chunk_size": int(manifest["chunk_size"]),
        "payload_bytes": int(raw["payload_bytes"]),
        "payload_validation_seconds": validation_seconds,
        "materialization": materialization,
        "augmentation": deepcopy(manifest.get("augmentation")),
    }


def build_training_input_attestations(
    *,
    config_paths: Mapping[str, str | Path],
    cache_root: str | Path,
    cache_preparation_receipt_path: str | Path,
    expected_model_root: str | Path,
    max_cache_payload_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build native train/eval cache and complete base-weight attestations."""

    bound = _positive_bound(max_cache_payload_bytes, field="max_cache_payload_bytes")
    root = Path(cache_root).expanduser().resolve()
    resolved, config_identity, model_root = _resolved_configs(
        config_paths, expected_model_root=expected_model_root
    )
    derived = _derive_cache_inputs(resolved, cache_root=root, model_root=model_root)
    preparation, preparation_binding = _validate_preparation_receipt(
        Path(cache_preparation_receipt_path),
        resolved=resolved,
        cache_root=root,
        derived=derived,
    )
    raw, measured_bytes = _chunk_payload_precheck(
        derived, max_cache_payload_bytes=bound
    )
    split_identities: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        started = time.monotonic()
        manifest = load_cache_manifest(
            derived[split]["cache_dir"],
            cache_root=root,
            expected_fingerprint=derived[split]["fingerprint"],
            level="payloads",
        )
        split_identities[split] = _native_split_identity(
            split,
            derived=derived[split],
            raw=raw[split],
            manifest=manifest,
            validation_seconds=max(0.0, time.monotonic() - started),
        )
        receipt_row = preparation[_SPLIT_RECEIPT_KEYS[split]]
        if (
            receipt_row["manifest_sha256"] != split_identities[split]["manifest_sha256"]
            or int(receipt_row["micro_step_count"])
            != split_identities[split]["micro_step_count"]
        ):
            raise ValueError(
                f"cache preparation receipt {split} manifest/count drifted"
            )
    cache_body = {
        "schema": CACHE_ATTESTATION_SCHEMA,
        "status": "passed",
        "model_loaded": False,
        "config_identity": config_identity,
        "cache_root": str(root),
        "max_cache_payload_bytes": bound,
        "measured_payload_bytes": measured_bytes,
        "preparation_duration_seconds": float(
            preparation.get("measurement", {}).get("duration_seconds", 0.0)
        ),
        "preparation_receipt": preparation_binding,
        "splits": split_identities,
    }
    cache_attestation = {
        **cache_body,
        "attestation_sha256": _sha256_json(cache_body),
    }
    weight_identity, weight_hash_execution_policy = (
        base_model_weight_identity_with_execution_policy(model_root)
    )
    weight_hash_execution_policy = _validated_weight_hash_execution_policy(
        weight_hash_execution_policy,
        weight_identity=weight_identity,
    )
    model_body = {
        "schema": MODEL_ATTESTATION_SCHEMA,
        "status": "passed",
        "model_loaded": False,
        "config_identity": config_identity,
        "model_root": str(model_root),
        "base_model_weight_identity": weight_identity,
        "weight_hash_execution_policy": weight_hash_execution_policy,
    }
    model_attestation = {
        **model_body,
        "attestation_sha256": _sha256_json(model_body),
    }
    return cache_attestation, model_attestation


def _validate_attestation_digest(
    value: Mapping[str, Any], *, schema: str, owner: str
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    observed = receipt.pop("attestation_sha256", None)
    if receipt.get("schema") != schema or receipt.get("status") != "passed":
        raise ValueError(f"{owner} schema/status is unsupported")
    if not isinstance(observed, str) or observed != _sha256_json(receipt):
        raise ValueError(f"{owner} attestation hash is invalid")
    return dict(value)


def _validate_cache_attestation_shape(cache: Mapping[str, Any]) -> None:
    _exact_keys(
        cache,
        {
            "schema",
            "status",
            "model_loaded",
            "config_identity",
            "cache_root",
            "max_cache_payload_bytes",
            "measured_payload_bytes",
            "preparation_duration_seconds",
            "preparation_receipt",
            "splits",
            "attestation_sha256",
        },
        owner="cache input attestation",
    )
    splits = cache.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise ValueError("cache input split inventory is not exact")
    expected_split_fields = {
        "split",
        "cache_dir",
        "fingerprint",
        "manifest_path",
        "manifest_sha256",
        "determinants_sha256",
        "determinant_hashes",
        "chunk_sha256s",
        "chunk_count",
        "micro_step_count",
        "chunk_size",
        "payload_bytes",
        "payload_validation_seconds",
        "materialization",
        "augmentation",
    }
    for split in SPLITS:
        row = splits[split]
        if not isinstance(row, Mapping):
            raise ValueError(f"{split} cache input identity must be an object")
        _exact_keys(row, expected_split_fields, owner=f"{split} cache input identity")
        if row.get("split") != split:
            raise ValueError(f"{split} cache input identity has the wrong split")
        _validated_r5_materialization_policy(row.get("materialization"), split=split)


def _validated_r5_materialization_policy(value: Any, *, split: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"strategy", "workers"}:
        raise ValueError(
            f"{split} cache lacks the authenticated materialization policy"
        )
    strategy = value.get("strategy")
    workers = value.get("workers")
    if (
        strategy != "fork_process_pool"
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 1
    ):
        raise ValueError(
            f"{split} cache lacks the authenticated materialization policy"
        )
    return {"strategy": strategy, "workers": workers}


def _validated_weight_hash_execution_policy(
    value: Any, *, weight_identity: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("model weight hash execution policy must be an object")
    _exact_keys(
        value,
        {"schema", "strategy", "resolved_workers", "payload_file_count"},
        owner="model weight hash execution policy",
    )
    workers = value.get("resolved_workers")
    payload_file_count = value.get("payload_file_count")
    if (
        value.get("schema") != MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA
        or value.get("strategy") != "thread_pool_file_sha256"
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
        or isinstance(payload_file_count, bool)
        or not isinstance(payload_file_count, int)
        or payload_file_count <= 0
        or workers > payload_file_count
        or payload_file_count != weight_identity.get("shard_count")
    ):
        raise ValueError("model weight hash execution policy is invalid")
    if payload_file_count > 1 and workers <= 1:
        raise ValueError("multi-file model weight hashing must execute in parallel")
    return {
        "schema": MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": workers,
        "payload_file_count": payload_file_count,
    }


def _validate_model_attestation_shape(model: Mapping[str, Any]) -> None:
    _exact_keys(
        model,
        {
            "schema",
            "status",
            "model_loaded",
            "config_identity",
            "model_root",
            "base_model_weight_identity",
            "weight_hash_execution_policy",
            "attestation_sha256",
        },
        owner="model input attestation",
    )


def validate_training_input_attestations(
    *,
    cache_attestation: Mapping[str, Any],
    model_attestation: Mapping[str, Any],
    config_paths: Mapping[str, str | Path],
    validate_cache_payloads: bool,
    rehash_model_weights: bool,
    max_cache_payload_bytes: int,
) -> dict[str, Any]:
    """Revalidate attested inputs without loading model tensors."""

    if not isinstance(validate_cache_payloads, bool) or not isinstance(
        rehash_model_weights, bool
    ):
        raise ValueError("validation controls must be booleans")
    bound = _positive_bound(max_cache_payload_bytes, field="max_cache_payload_bytes")
    _validate_cache_attestation_shape(cache_attestation)
    _validate_model_attestation_shape(model_attestation)
    cache = _validate_attestation_digest(
        cache_attestation, schema=CACHE_ATTESTATION_SCHEMA, owner="cache input"
    )
    model = _validate_attestation_digest(
        model_attestation, schema=MODEL_ATTESTATION_SCHEMA, owner="model input"
    )
    root = Path(str(cache.get("cache_root", ""))).expanduser().resolve()
    resolved, config_identity, model_root = _resolved_configs(
        config_paths, expected_model_root=model.get("model_root")
    )
    if (
        cache.get("config_identity") != config_identity
        or model.get("config_identity") != config_identity
    ):
        raise ValueError("input attestation config identity drifted")
    if cache.get("model_loaded") is not False or model.get("model_loaded") is not False:
        raise ValueError("input attestations must remain model-free")
    derived = _derive_cache_inputs(resolved, cache_root=root, model_root=model_root)
    preparation, preparation_binding = _validate_preparation_receipt(
        Path(cache["preparation_receipt"]["path"]),
        resolved=resolved,
        cache_root=root,
        derived=derived,
    )
    if preparation_binding != cache["preparation_receipt"] or float(
        preparation.get("measurement", {}).get("duration_seconds", 0.0)
    ) != float(cache["preparation_duration_seconds"]):
        raise ValueError("cache preparation receipt binding drifted")
    raw, measured_bytes = _chunk_payload_precheck(
        derived, max_cache_payload_bytes=bound
    )
    if measured_bytes != cache.get("measured_payload_bytes") or bound != cache.get(
        "max_cache_payload_bytes"
    ):
        raise ValueError("cache input payload byte identity or bound drifted")
    expected_splits = cache["splits"]
    level = "payloads" if validate_cache_payloads else "manifest"
    for split in SPLITS:
        started = time.monotonic()
        manifest = load_cache_manifest(
            derived[split]["cache_dir"],
            cache_root=root,
            expected_fingerprint=derived[split]["fingerprint"],
            level=level,
        )
        current = _native_split_identity(
            split,
            derived=derived[split],
            raw=raw[split],
            manifest=manifest,
            validation_seconds=float(
                expected_splits[split]["payload_validation_seconds"]
            ),
        )
        if current != expected_splits[split]:
            raise ValueError(f"{split} cache input attestation drifted")
        _ = time.monotonic() - started
    expected_weights = validate_model_weight_identity(
        model.get("base_model_weight_identity", {})
    )
    expected_weight_hash_policy = _validated_weight_hash_execution_policy(
        model.get("weight_hash_execution_policy"),
        weight_identity=expected_weights,
    )
    if expected_weights["root"] != str(model_root):
        raise ValueError("model input attestation binds the wrong root")
    if rehash_model_weights:
        observed_weights, observed_weight_hash_policy = (
            base_model_weight_identity_with_execution_policy(model_root)
        )
        assert_model_weight_identity_equal(expected_weights, observed_weights)
        observed_weight_hash_policy = _validated_weight_hash_execution_policy(
            observed_weight_hash_policy,
            weight_identity=observed_weights,
        )
        if observed_weight_hash_policy != expected_weight_hash_policy:
            raise ValueError("model weight hash execution policy drifted")
    return {
        "status": "passed",
        "cache_attestation_sha256": cache["attestation_sha256"],
        "model_attestation_sha256": model["attestation_sha256"],
        "cache_payloads_validated": validate_cache_payloads,
        "model_weights_rehashed": rehash_model_weights,
        "measured_cache_payload_bytes": measured_bytes,
        "materialization_policy": {
            split: _validated_r5_materialization_policy(
                expected_splits[split]["materialization"], split=split
            )
            for split in SPLITS
        },
        "weight_hash_execution_policy": expected_weight_hash_policy,
    }


__all__ = [
    "CACHE_ATTESTATION_SCHEMA",
    "MODEL_ATTESTATION_SCHEMA",
    "build_training_input_attestations",
    "validate_training_input_attestations",
]
