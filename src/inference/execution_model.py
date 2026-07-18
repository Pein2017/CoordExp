"""Immutable execution models for backends that cannot compose CoordExp payloads."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, Literal

from src.common.errors import RuntimeContractError
from src.inference.model_assets import (
    build_model_snapshot_manifest,
    validate_model_snapshot_manifest,
)


EXECUTION_MODEL_RECEIPT_VERSION = "coordexp-swift-execution-model-v1"
MATERIALIZATION_ALGORITHM_VERSION = "coordexp-swift-dora-delta-fold-v3"
MATERIALIZATION_RECEIPT_NAME = "coordexp_materialization.json"
DEFAULT_EXECUTION_MODEL_CACHE_ROOT = Path(
    "model_cache/coordexp_swift/vllm_materialized"
)

MaterializeSnapshot = Callable[[Path], Mapping[str, object] | None]


def build_execution_model_composition_key(
    *,
    base_manifest: Mapping[str, object],
    adapter_identity: Mapping[str, object] | None,
    embedding_delta_identity: Mapping[str, object] | None,
    target_dtype: Literal["bf16", "fp16", "fp32"],
    package_versions: Mapping[str, str] | None = None,
) -> str:
    """Build the path-independent identity of one requested composition."""

    _validate_target_dtype(target_dtype)
    payload = {
        "algorithm_version": MATERIALIZATION_ALGORITHM_VERSION,
        "base": _without_provenance_paths(base_manifest),
        "adapter": _without_provenance_paths(adapter_identity),
        "embedding_delta": _without_provenance_paths(embedding_delta_identity),
        "target_dtype": target_dtype,
        "package_versions": dict(
            sorted((package_versions or _materialization_package_versions()).items())
        ),
    }
    return _sha256_json(payload)


def resolve_execution_model(
    *,
    base_model_path: str | Path,
    target_dtype: Literal["bf16", "fp16", "fp32"],
    adapter_path: str | Path | None = None,
    adapter_name: str = "default",
    embedding_delta_path: str | Path | None = None,
    adapter_identity: Mapping[str, object] | None = None,
    embedding_delta_identity: Mapping[str, object] | None = None,
    cache_root: str | Path | None = None,
    materialize_snapshot: MaterializeSnapshot | None = None,
) -> dict[str, Any]:
    """Resolve a content-hashed base snapshot or atomically materialize a composition."""

    _validate_target_dtype(target_dtype)
    base_root = Path(base_model_path).expanduser().resolve()
    base_manifest = build_model_snapshot_manifest(base_root)
    packages = _materialization_package_versions()

    if adapter_identity is None and adapter_path is not None:
        from src.adapters.dora import inspect_dora_adapter_payload

        adapter_identity = inspect_dora_adapter_payload(
            adapter_path,
            expected_base_model_path=base_root,
        )
    if embedding_delta_identity is None and embedding_delta_path is not None:
        from src.qwen.special_token_embeddings import (
            inspect_special_token_embedding_delta_payload,
        )

        embedding_delta_identity = inspect_special_token_embedding_delta_payload(
            embedding_delta_path,
            expected_base_model_path=base_root,
            expected_base_config_sha256=_manifest_file_sha256(
                base_manifest,
                "config.json",
            ),
            expected_tokenizer_sha256=_manifest_file_sha256(
                base_manifest,
                "tokenizer.json",
            ),
        )

    composed = adapter_identity is not None or embedding_delta_identity is not None
    composition_key = build_execution_model_composition_key(
        base_manifest=base_manifest,
        adapter_identity=adapter_identity,
        embedding_delta_identity=embedding_delta_identity,
        target_dtype=target_dtype,
        package_versions=packages,
    )
    source_identity = {
        "base": base_manifest,
        "adapter": None if adapter_identity is None else dict(adapter_identity),
        "embedding_delta": (
            None
            if embedding_delta_identity is None
            else dict(embedding_delta_identity)
        ),
    }
    if not composed:
        receipt = _build_receipt(
            mode="base_only",
            model_path=base_root,
            composition_key=composition_key,
            target_dtype=target_dtype,
            package_versions=packages,
            source_identity=source_identity,
            snapshot_manifest=base_manifest,
            receipt_path=None,
            materialization={"status": "direct_base_snapshot"},
        )
        return validate_execution_model_receipt(receipt)

    root = Path(cache_root or DEFAULT_EXECUTION_MODEL_CACHE_ROOT).expanduser().resolve()
    final_root = root / composition_key
    receipt_path = final_root / MATERIALIZATION_RECEIPT_NAME
    snapshot_root = final_root / "snapshot"
    lock_path = root / ".locks" / f"{composition_key}.lock"

    with _exclusive_lock(lock_path):
        if final_root.exists():
            receipt = _load_receipt(receipt_path)
            _validate_expected_composition(
                receipt,
                composition_key=composition_key,
                source_identity=source_identity,
                target_dtype=target_dtype,
            )
            return _bind_existing_parity_if_present(
                validate_execution_model_receipt(receipt)
            )

        staging_parent = root / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging_root = staging_parent / f"{composition_key}.{os.getpid()}.{uuid.uuid4().hex}"
        staging_snapshot = staging_root / "snapshot"
        staging_root.mkdir(parents=False, exist_ok=False)
        try:
            builder = materialize_snapshot or _default_materializer(
                base_model_path=base_root,
                target_dtype=target_dtype,
                adapter_path=adapter_path,
                adapter_name=adapter_name,
                embedding_delta_path=embedding_delta_path,
                adapter_identity=adapter_identity,
                embedding_delta_identity=embedding_delta_identity,
            )
            evidence = dict(builder(staging_snapshot) or {})
            _validate_standard_snapshot(staging_snapshot)
            staged_manifest = build_model_snapshot_manifest(staging_snapshot)
            published_manifest = {
                **staged_manifest,
                "root": str(snapshot_root),
            }
            receipt = _build_receipt(
                mode="materialized",
                model_path=snapshot_root,
                composition_key=composition_key,
                target_dtype=target_dtype,
                package_versions=packages,
                source_identity=source_identity,
                snapshot_manifest=published_manifest,
                receipt_path=receipt_path,
                materialization={"status": "built", **evidence},
            )
            _write_json(staging_root / MATERIALIZATION_RECEIPT_NAME, receipt)
            root.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root, final_root)
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

    return _bind_existing_parity_if_present(
        validate_execution_model_receipt(_load_receipt(receipt_path))
    )


def validate_execution_model_receipt(
    receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Revalidate receipt integrity and every executable snapshot byte."""

    payload = dict(receipt)
    required = {
        "version",
        "mode",
        "model_path",
        "composition_key",
        "target_dtype",
        "algorithm_version",
        "package_versions",
        "source_identity",
        "snapshot_manifest",
        "snapshot_fingerprint",
        "materialization",
        "receipt_fingerprint",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeContractError(
            "execution-model receipt is missing required identity fields",
            code="inference.execution_model_receipt_invalid",
            context={"missing_fields": missing},
        )
    if payload["version"] != EXECUTION_MODEL_RECEIPT_VERSION:
        _receipt_error("version", payload)
    if payload["algorithm_version"] != MATERIALIZATION_ALGORITHM_VERSION:
        _receipt_error("algorithm_version", payload)
    if payload["mode"] not in ("base_only", "materialized"):
        _receipt_error("mode", payload)
    _validate_target_dtype(str(payload["target_dtype"]))

    expected_receipt_fingerprint = _receipt_fingerprint(payload)
    if payload["receipt_fingerprint"] != expected_receipt_fingerprint:
        raise RuntimeContractError(
            "execution-model receipt identity does not match its semantic payload",
            code="inference.execution_model_receipt_identity_mismatch",
            context={
                "expected_fingerprint": expected_receipt_fingerprint,
                "actual_fingerprint": payload["receipt_fingerprint"],
            },
        )

    snapshot_manifest = payload["snapshot_manifest"]
    if not isinstance(snapshot_manifest, dict):
        _receipt_error("snapshot_manifest", payload)
    observed = validate_model_snapshot_manifest(snapshot_manifest)
    if observed["fingerprint"] != payload["snapshot_fingerprint"]:
        raise RuntimeContractError(
            "execution-model snapshot fingerprint disagrees with its manifest",
            code="inference.execution_model_snapshot_identity_mismatch",
            context={
                "receipt_fingerprint": payload["snapshot_fingerprint"],
                "observed_fingerprint": observed["fingerprint"],
            },
        )
    if str(Path(str(payload["model_path"])).resolve()) != str(
        Path(str(snapshot_manifest["root"])).resolve()
    ):
        _receipt_error("model_path", payload)
    _validate_standard_snapshot(Path(str(payload["model_path"])))

    qualification = payload.get("qualification")
    if qualification is not None:
        _validate_qualification(qualification, execution_model=payload)

    if payload["mode"] == "materialized":
        receipt_path = payload.get("receipt_path")
        if not isinstance(receipt_path, str) or not receipt_path:
            _receipt_error("receipt_path", payload)
        on_disk = _load_receipt(Path(receipt_path))
        on_disk_comparable = {
            key: value for key, value in payload.items() if key != "qualification"
        }
        if on_disk != on_disk_comparable:
            raise RuntimeContractError(
                "execution-model receipt differs from the published receipt",
                code="inference.execution_model_receipt_disk_mismatch",
                context={"receipt_path": receipt_path},
            )
    elif payload.get("receipt_path") is not None:
        _receipt_error("receipt_path", payload)

    return payload


def bind_execution_model_parity(
    receipt: Mapping[str, object],
    parity_receipt: Mapping[str, object],
    *,
    parity_path: str | Path | None = None,
) -> dict[str, Any]:
    from src.inference.execution_model_parity import (
        validate_execution_model_parity_receipt,
    )

    execution_model = validate_execution_model_receipt(receipt)
    parity = validate_execution_model_parity_receipt(
        parity_receipt,
        execution_model=execution_model,
    )
    qualified = dict(execution_model)
    qualified["qualification"] = {
        "parity_digest": parity["digest"],
        "parity_path": None if parity_path is None else str(Path(parity_path).resolve()),
        "parity_receipt": parity,
    }
    return validate_execution_model_receipt(qualified)


def load_execution_model_receipt(path: str | Path) -> dict[str, Any]:
    return _bind_existing_parity_if_present(
        validate_execution_model_receipt(_load_receipt(Path(path)))
    )


def _bind_existing_parity_if_present(receipt: Mapping[str, object]) -> dict[str, Any]:
    if receipt.get("mode") != "materialized":
        return dict(receipt)
    from src.inference.execution_model_parity import (
        EXECUTION_MODEL_PARITY_NAME,
        load_execution_model_parity_receipt,
    )

    receipt_path = Path(str(receipt["receipt_path"]))
    parity_path = receipt_path.with_name(EXECUTION_MODEL_PARITY_NAME)
    if not parity_path.is_file():
        return dict(receipt)
    parity = load_execution_model_parity_receipt(parity_path)
    return bind_execution_model_parity(
        receipt,
        parity,
        parity_path=parity_path,
    )


def _default_materializer(
    *,
    base_model_path: Path,
    target_dtype: str,
    adapter_path: str | Path | None,
    adapter_name: str,
    embedding_delta_path: str | Path | None,
    adapter_identity: Mapping[str, object] | None,
    embedding_delta_identity: Mapping[str, object] | None,
) -> MaterializeSnapshot:
    if adapter_identity is not None and adapter_path is None:
        raise RuntimeContractError(
            "adapter identity requires an adapter path for materialization",
            code="inference.execution_model_adapter_path_missing",
        )
    if embedding_delta_identity is not None and embedding_delta_path is None:
        raise RuntimeContractError(
            "embedding-delta identity requires a payload path for materialization",
            code="inference.execution_model_delta_path_missing",
        )

    def build(snapshot_root: Path) -> Mapping[str, object]:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_model_path),
            dtype=_torch_dtype(target_dtype),
            attn_implementation="eager",
            device_map="cpu",
            local_files_only=True,
        )
        _validate_cpu_target_dtype(model, target_dtype=target_dtype)
        merge_receipt: Mapping[str, object] | None = None
        delta_receipt: Mapping[str, object] | None = None
        if adapter_path is not None:
            from src.adapters.dora import merge_dora_adapter_for_execution

            model, merge_receipt = merge_dora_adapter_for_execution(
                model,
                adapter_path,
                adapter_name=adapter_name,
                expected_identity=adapter_identity,
            )
            _validate_cpu_target_dtype(model, target_dtype=target_dtype)
        if embedding_delta_path is not None:
            from src.qwen.special_token_embeddings import (
                fold_special_token_embedding_delta_for_execution,
            )

            delta_receipt = fold_special_token_embedding_delta_for_execution(
                model,
                embedding_delta_path,
                expected_identity=embedding_delta_identity,
            )
            _validate_cpu_target_dtype(model, target_dtype=target_dtype)
        _validate_tied_weights(model)
        snapshot_root.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(snapshot_root, safe_serialization=True)
        processor = AutoProcessor.from_pretrained(
            str(base_model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        processor.save_pretrained(snapshot_root)
        return {
            "adapter_merge": None if merge_receipt is None else dict(merge_receipt),
            "embedding_delta_fold": (
                None if delta_receipt is None else dict(delta_receipt)
            ),
            "tied_input_output": True,
        }

    return build


def _build_receipt(
    *,
    mode: str,
    model_path: Path,
    composition_key: str,
    target_dtype: str,
    package_versions: Mapping[str, str],
    source_identity: Mapping[str, object],
    snapshot_manifest: Mapping[str, object],
    receipt_path: Path | None,
    materialization: Mapping[str, object],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": EXECUTION_MODEL_RECEIPT_VERSION,
        "mode": mode,
        "model_path": str(model_path.resolve()),
        "composition_key": composition_key,
        "target_dtype": target_dtype,
        "algorithm_version": MATERIALIZATION_ALGORITHM_VERSION,
        "package_versions": dict(sorted(package_versions.items())),
        "source_identity": _jsonable(source_identity),
        "snapshot_manifest": _jsonable(snapshot_manifest),
        "snapshot_fingerprint": snapshot_manifest["fingerprint"],
        "receipt_path": None if receipt_path is None else str(receipt_path.resolve()),
        "materialization": _jsonable(materialization),
    }
    payload["receipt_fingerprint"] = _receipt_fingerprint(payload)
    return payload


def _validate_expected_composition(
    receipt: Mapping[str, object],
    *,
    composition_key: str,
    source_identity: Mapping[str, object],
    target_dtype: str,
) -> None:
    if receipt.get("composition_key") != composition_key:
        _receipt_error("composition_key", receipt)
    if receipt.get("target_dtype") != target_dtype:
        _receipt_error("target_dtype", receipt)
    if _without_provenance_paths(receipt.get("source_identity")) != (
        _without_provenance_paths(source_identity)
    ):
        raise RuntimeContractError(
            "published execution-model source identity differs from the request",
            code="inference.execution_model_source_identity_mismatch",
            context={"composition_key": composition_key},
        )


def _receipt_fingerprint(receipt: Mapping[str, object]) -> str:
    semantic = {
        key: value
        for key, value in receipt.items()
        if key not in {
            "model_path",
            "receipt_path",
            "receipt_fingerprint",
            "qualification",
        }
    }
    return _sha256_json(_without_provenance_paths(semantic))


def _validate_standard_snapshot(root: Path) -> None:
    required = ("config.json", "tokenizer.json", "preprocessor_config.json")
    missing = [name for name in required if not (root / name).is_file()]
    weight_files = [
        path
        for path in root.glob("*")
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".bin")
            or path.name.endswith(".index.json")
        )
        and path.name.startswith(("model", "pytorch_model"))
    ]
    if missing or not weight_files:
        raise RuntimeContractError(
            "execution-model snapshot is missing required standard HF files",
            code="inference.execution_model_snapshot_incomplete",
            context={
                "root": str(root),
                "missing_files": missing,
                "weight_files": [path.name for path in weight_files],
            },
        )


def _validate_qualification(
    qualification: object,
    *,
    execution_model: Mapping[str, object],
) -> None:
    from src.inference.execution_model_parity import (
        validate_execution_model_parity_receipt,
    )

    if not isinstance(qualification, Mapping):
        _receipt_error("qualification", execution_model)
    parity = qualification.get("parity_receipt")
    if not isinstance(parity, Mapping):
        _receipt_error("qualification.parity_receipt", execution_model)
    validated = validate_execution_model_parity_receipt(
        parity,
        execution_model=execution_model,
    )
    if qualification.get("parity_digest") != validated["digest"]:
        _receipt_error("qualification.parity_digest", execution_model)


def _validate_tied_weights(model: Any) -> None:
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    if getattr(embedding, "weight", None) is not getattr(output, "weight", None):
        raise RuntimeContractError(
            "materialized execution model does not preserve tied input/output weights",
            code="inference.execution_model_untied",
        )


def _validate_cpu_target_dtype(model: Any, *, target_dtype: str) -> None:
    expected_dtype = _torch_dtype(target_dtype)
    invalid: list[dict[str, str]] = []
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            wrong_device = value.device.type != "cpu"
            wrong_dtype = (
                kind == "parameter"
                and value.dtype.is_floating_point
                and value.dtype != expected_dtype
            )
            if wrong_device or wrong_dtype:
                invalid.append(
                    {
                        "kind": kind,
                        "name": name,
                        "device": str(value.device),
                        "dtype": str(value.dtype),
                    }
                )
                if len(invalid) >= 20:
                    break
        if len(invalid) >= 20:
            break
    if invalid:
        raise RuntimeContractError(
            "execution-model materialization requires CPU state and target-dtype parameters",
            code="inference.execution_model_cpu_dtype_mismatch",
            context={"target_dtype": target_dtype, "invalid": invalid},
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeContractError(
            "published execution-model receipt is missing",
            code="inference.execution_model_receipt_missing",
            context={"receipt_path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeContractError(
            "execution-model receipt must be a JSON object",
            code="inference.execution_model_receipt_invalid",
            context={"receipt_path": str(path)},
        )
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _manifest_file_sha256(manifest: Mapping[str, object], relative_path: str) -> str:
    files = manifest.get("files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, Mapping) and item.get("relative_path") == relative_path:
                value = item.get("sha256")
                if isinstance(value, str) and value:
                    return value
    raise RuntimeContractError(
        "base snapshot identity is missing a required file hash",
        code="inference.execution_model_base_identity_incomplete",
        context={"relative_path": relative_path},
    )


def _without_provenance_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_provenance_paths(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key)
            not in {
                "root",
                "path",
                "adapter_path",
                "delta_path",
                "tensor_path",
                "metadata_path",
                "config_path",
                "model_path",
                "receipt_path",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_without_provenance_paths(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _materialization_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("transformers", "peft", "torch", "safetensors"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _validate_target_dtype(value: str) -> None:
    if value not in {"bf16", "fp16", "fp32"}:
        raise RuntimeContractError(
            "execution-model target dtype is unsupported",
            code="inference.execution_model_dtype",
            context={"target_dtype": value},
        )


def _receipt_error(field: str, receipt: Mapping[str, object]) -> None:
    raise RuntimeContractError(
        "execution-model receipt contains an invalid identity field",
        code="inference.execution_model_receipt_invalid",
        context={"field": field, "value": receipt.get(field)},
    )


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_EXECUTION_MODEL_CACHE_ROOT",
    "EXECUTION_MODEL_RECEIPT_VERSION",
    "MATERIALIZATION_ALGORITHM_VERSION",
    "MATERIALIZATION_RECEIPT_NAME",
    "build_execution_model_composition_key",
    "bind_execution_model_parity",
    "load_execution_model_receipt",
    "resolve_execution_model",
    "validate_execution_model_receipt",
]
