"""Domain-neutral content-identity owner for production artifacts.

Design decision 7 of ``decompose-coordexp-swift-training-orchestration`` moves
the generic identity machinery here so production owners
(``src/prepare_train_cache.py``, ``src/training/input_attestation.py``, and the
training assembly) no longer depend on the historical packed-parity module.

This is a *move*, not a rewrite.  Payload schemas, sorting/encoding, byte
bounds, file-stability checks, symlink rejection, aggregate digests, worker
bounds, artifact bytes, the ``ParityContractError`` type, and the literal
``qwen.parity.*`` contract codes are preserved exactly so historical readers
that catch these errors and compare these digests keep working.
``src/qwen/parity.py`` re-exports every public symbol below under its existing
name and keeps its parity-only comparison, tolerance, plan, receipt, gradient,
and Qwen attestation logic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import subprocess
from typing import Any

from src.common.errors import RuntimeContractError


MODEL_WEIGHT_IDENTITY_SCHEMA = "coordexp-swift-base-model-weights-v1"
MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA = (
    "coordexp-swift-base-model-weight-hash-execution-policy-v1"
)

MAX_WEIGHT_INDEX_BYTES = 16 * 1024 * 1024
MAX_WEIGHT_DECLARATIONS = 100_000
MAX_WEIGHT_SHARDS = 256
MAX_WEIGHT_SHARD_BYTES = 16 * 1024 * 1024 * 1024
MAX_WEIGHT_TOTAL_BYTES = 64 * 1024 * 1024 * 1024


class ParityContractError(RuntimeContractError):
    """Fail-closed packed-parity contract violation."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict deterministic JSON and reject non-finite values."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ParityContractError(
            "parity artifact is not strict JSON",
            code="qwen.parity.strict_json",
            context={"value_type": type(value).__name__, "error": str(exc)},
            cause=exc,
        ) from exc
    return text.encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ParityContractError(
            "parity identity file is unreadable",
            code="qwen.parity.identity_file",
            context={"path": str(source), "error": type(exc).__name__},
            cause=exc,
        ) from exc
    return digest.hexdigest()


def write_strict_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    on_linked: Callable[[], None] | None = None,
) -> Path:
    """Publish one strict JSON artifact without exposing partial bytes."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ParityContractError(
                "parity artifact target already exists",
                code="qwen.parity.artifact_collision",
                context={"path": str(target)},
                cause=exc,
            ) from exc
        if on_linked is not None:
            on_linked()
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError as exc:
            raise ParityContractError(
                "parity artifact directory cannot be opened for durability sync",
                code="qwen.parity.artifact_directory_sync",
                context={
                    "path": str(target.parent),
                    "error_type": type(exc).__name__,
                },
                cause=exc,
            ) from exc
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def assert_absent_artifact_target(path: str | Path) -> Path:
    """Fail before expensive work unless an artifact can be published safely."""

    requested = Path(path).expanduser()
    target = requested.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise ParityContractError(
            "parity artifact target already exists",
            code="qwen.parity.artifact_collision",
            context={"path": str(target)},
        )
    parent = target.parent
    if not parent.is_dir():
        raise ParityContractError(
            "parity artifact parent must already exist",
            code="qwen.parity.artifact_parent",
            context={"path": str(parent)},
        )
    current = parent
    while current != current.parent:
        if current.is_symlink():
            raise ParityContractError(
                "parity artifact path must not traverse symlinks",
                code="qwen.parity.artifact_symlink",
                context={"path": str(current)},
            )
        current = current.parent
    return target


def base_model_weight_identity(
    model_root: str | Path,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Content-bind the bounded base-model safetensors payload."""

    identity, _ = base_model_weight_identity_with_execution_policy(
        model_root,
        max_workers=max_workers,
    )
    return identity


def base_model_weight_identity_with_execution_policy(
    model_root: str | Path,
    *,
    max_workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Content-bind base weights and return the schema-neutral hash policy."""

    _validate_requested_weight_hash_workers(max_workers)

    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise ParityContractError(
            "base-model weight root is not a directory",
            code="qwen.parity.weight_root",
            context={"path": str(root)},
        )
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index_identity, index_bytes = _stable_file_identity(
            index_path,
            root=root,
            max_bytes=MAX_WEIGHT_INDEX_BYTES,
            return_bytes=True,
        )
        try:
            index_payload = json.loads(
                index_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ParityContractError(
                "base-model weight index is not strict UTF-8 JSON",
                code="qwen.parity.weight_index_json",
                context={"path": str(index_path)},
                cause=exc,
            ) from exc
        if not isinstance(index_payload, dict):
            raise ParityContractError(
                "base-model weight index root must be a mapping",
                code="qwen.parity.weight_index_shape",
                context={"path": str(index_path)},
            )
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ParityContractError(
                "base-model weight index must declare a non-empty weight_map",
                code="qwen.parity.weight_index_shape",
                context={"path": str(index_path)},
            )
        if len(weight_map) > MAX_WEIGHT_DECLARATIONS:
            raise ParityContractError(
                "base-model weight index exceeds the declaration bound",
                code="qwen.parity.weight_declaration_bound",
                context={
                    "count": len(weight_map),
                    "maximum": MAX_WEIGHT_DECLARATIONS,
                },
            )
        shard_names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            _non_empty_text(tensor_name, "weight_map tensor name")
            shard_names.add(_safe_relative_weight_path(shard_name))
        if len(shard_names) > MAX_WEIGHT_SHARDS:
            raise ParityContractError(
                "base-model weight index exceeds the shard bound",
                code="qwen.parity.weight_shard_bound",
                context={"count": len(shard_names), "maximum": MAX_WEIGHT_SHARDS},
            )
        shard_paths = [root / shard_name for shard_name in sorted(shard_names)]
        _preflight_weight_file_set(shard_paths)
        resolved_workers = _resolve_weight_hash_workers(
            max_workers,
            payload_file_count=len(shard_paths),
        )
        shards = _hash_weight_shards(
            shard_paths,
            root=root,
            resolved_workers=resolved_workers,
        )
        mode = "indexed_safetensors"
        index_artifact: dict[str, Any] | None = index_identity
        declaration_count = len(weight_map)
    else:
        standalone = root / "model.safetensors"
        discovered: list[Path] = []
        try:
            for candidate in root.iterdir():
                if candidate.name.endswith(".safetensors"):
                    discovered.append(candidate)
                    if len(discovered) > MAX_WEIGHT_SHARDS:
                        raise ParityContractError(
                            "unindexed base-model weight discovery exceeds its bound",
                            code="qwen.parity.weight_shard_bound",
                            context={
                                "count": len(discovered),
                                "maximum": MAX_WEIGHT_SHARDS,
                            },
                        )
        except OSError as exc:
            raise ParityContractError(
                "base-model weight root cannot be inventoried",
                code="qwen.parity.weight_root",
                context={"path": str(root), "error": type(exc).__name__},
                cause=exc,
            ) from exc
        discovered.sort()
        if not standalone.is_file() or discovered != [standalone]:
            raise ParityContractError(
                "base-model weights require an index or one standalone model.safetensors",
                code="qwen.parity.weight_layout",
                context={
                    "root": str(root),
                    "discovered_count": len(discovered),
                    "discovered": [item.name for item in discovered[:16]],
                },
            )
        _preflight_weight_file_set([standalone])
        resolved_workers = _resolve_weight_hash_workers(
            max_workers,
            payload_file_count=1,
        )
        shards = _hash_weight_shards(
            [standalone],
            root=root,
            resolved_workers=resolved_workers,
        )
        mode = "standalone_safetensors"
        index_artifact = None
        declaration_count = None
    total_bytes = sum(int(item["size_bytes"]) for item in shards)
    if total_bytes > MAX_WEIGHT_TOTAL_BYTES:
        raise ParityContractError(
            "base-model weight payload exceeds the total byte bound",
            code="qwen.parity.weight_total_bound",
            context={"total_bytes": total_bytes, "maximum": MAX_WEIGHT_TOTAL_BYTES},
        )
    determinants = {
        "schema": MODEL_WEIGHT_IDENTITY_SCHEMA,
        "root": str(root),
        "mode": mode,
        "index": index_artifact,
        "declaration_count": declaration_count,
        "shards": shards,
        "shard_count": len(shards),
        "total_bytes": total_bytes,
        "bounds": {
            "max_index_bytes": MAX_WEIGHT_INDEX_BYTES,
            "max_declarations": MAX_WEIGHT_DECLARATIONS,
            "max_shards": MAX_WEIGHT_SHARDS,
            "max_shard_bytes": MAX_WEIGHT_SHARD_BYTES,
            "max_total_bytes": MAX_WEIGHT_TOTAL_BYTES,
        },
    }
    identity = {**determinants, "aggregate_sha256": sha256_json(determinants)}
    execution_policy = {
        "schema": MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": resolved_workers,
        "payload_file_count": len(shards),
    }
    return validate_model_weight_identity(identity), execution_policy


def validate_model_weight_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one bounded indexed or standalone safetensors identity."""

    receipt = dict(identity)
    _expect_exact_keys(
        receipt,
        {
            "schema",
            "root",
            "mode",
            "index",
            "declaration_count",
            "shards",
            "shard_count",
            "total_bytes",
            "bounds",
            "aggregate_sha256",
        },
        owner="model_weight_identity",
    )
    if receipt["schema"] != MODEL_WEIGHT_IDENTITY_SCHEMA:
        raise ParityContractError(
            "base-model weight identity schema is unsupported",
            code="qwen.parity.weight_schema",
            context={"schema": receipt["schema"]},
        )
    _non_empty_text(receipt["root"], "model_weight_identity.root")
    mode = receipt["mode"]
    if mode not in {"indexed_safetensors", "standalone_safetensors"}:
        raise ParityContractError(
            "base-model weight identity mode is unsupported",
            code="qwen.parity.weight_layout",
            context={"mode": mode},
        )
    expected_bounds = {
        "max_index_bytes": MAX_WEIGHT_INDEX_BYTES,
        "max_declarations": MAX_WEIGHT_DECLARATIONS,
        "max_shards": MAX_WEIGHT_SHARDS,
        "max_shard_bytes": MAX_WEIGHT_SHARD_BYTES,
        "max_total_bytes": MAX_WEIGHT_TOTAL_BYTES,
    }
    bounds = _mapping(receipt["bounds"], "model_weight_identity.bounds")
    if dict(bounds) != expected_bounds:
        raise ParityContractError(
            "base-model weight identity bounds drifted",
            code="qwen.parity.weight_bounds",
            context={"expected": expected_bounds, "observed": dict(bounds)},
        )
    shards = _list_of_mappings(receipt["shards"], "model_weight_identity.shards")
    shard_count = _positive_int(receipt["shard_count"], "shard_count")
    if shard_count != len(shards) or shard_count > MAX_WEIGHT_SHARDS:
        raise ParityContractError(
            "base-model weight shard count is invalid",
            code="qwen.parity.weight_shard_bound",
            context={"declared": shard_count, "observed": len(shards)},
        )
    shard_paths: list[str] = []
    observed_total = 0
    for shard in shards:
        _expect_exact_keys(
            shard, {"path", "size_bytes", "sha256"}, owner="weight_shard"
        )
        shard_path = _safe_relative_weight_path(shard["path"])
        shard_paths.append(shard_path)
        size_bytes = _positive_int(shard["size_bytes"], "weight_shard.size_bytes")
        if size_bytes > MAX_WEIGHT_SHARD_BYTES:
            raise ParityContractError(
                "base-model weight shard exceeds the byte bound",
                code="qwen.parity.weight_file_bound",
                context={"path": shard_path, "size_bytes": size_bytes},
            )
        observed_total += size_bytes
        _sha256_text(shard["sha256"], "weight_shard.sha256")
    if shard_paths != sorted(set(shard_paths)):
        raise ParityContractError(
            "base-model weight shard paths must be unique and sorted",
            code="qwen.parity.weight_shard_order",
            context={"paths": shard_paths},
        )
    total_bytes = _positive_int(receipt["total_bytes"], "total_bytes")
    if total_bytes != observed_total or total_bytes > MAX_WEIGHT_TOTAL_BYTES:
        raise ParityContractError(
            "base-model weight total bytes are inconsistent",
            code="qwen.parity.weight_total_bound",
            context={"declared": total_bytes, "observed": observed_total},
        )
    if mode == "indexed_safetensors":
        index = _mapping(receipt["index"], "model_weight_identity.index")
        _expect_exact_keys(
            index, {"path", "size_bytes", "sha256"}, owner="weight_index"
        )
        if index["path"] != "model.safetensors.index.json":
            raise ParityContractError(
                "base-model weight index path is invalid",
                code="qwen.parity.weight_index_path",
                context={"path": index["path"]},
            )
        index_size = _positive_int(index["size_bytes"], "weight_index.size_bytes")
        if index_size > MAX_WEIGHT_INDEX_BYTES:
            raise ParityContractError(
                "base-model weight index exceeds its byte bound",
                code="qwen.parity.weight_file_bound",
                context={"size_bytes": index_size},
            )
        _sha256_text(index["sha256"], "weight_index.sha256")
        declarations = _positive_int(receipt["declaration_count"], "declaration_count")
        if declarations > MAX_WEIGHT_DECLARATIONS:
            raise ParityContractError(
                "base-model weight declaration count exceeds its bound",
                code="qwen.parity.weight_declaration_bound",
                context={"count": declarations},
            )
    elif (
        receipt["index"] is not None
        or receipt["declaration_count"] is not None
        or shard_paths != ["model.safetensors"]
    ):
        raise ParityContractError(
            "standalone safetensors identity has indexed-layout residue",
            code="qwen.parity.weight_layout",
            context={"shard_paths": shard_paths},
        )
    fingerprint = _sha256_text(
        receipt["aggregate_sha256"], "model_weight_identity.aggregate_sha256"
    )
    body = dict(receipt)
    del body["aggregate_sha256"]
    observed_fingerprint = sha256_json(body)
    if fingerprint != observed_fingerprint:
        raise ParityContractError(
            "base-model weight identity fingerprint is invalid",
            code="qwen.parity.weight_fingerprint",
            context={"expected": fingerprint, "observed": observed_fingerprint},
        )
    canonical_json_bytes(receipt)
    return receipt


def assert_model_weight_identity_equal(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    expected_receipt = validate_model_weight_identity(expected)
    observed_receipt = validate_model_weight_identity(observed)
    if expected_receipt != observed_receipt:
        raise ParityContractError(
            "base-model weight identity changed after plan preparation",
            code="qwen.parity.weight_identity_drift",
            context={
                "expected_aggregate_sha256": expected_receipt.get("aggregate_sha256"),
                "observed_aggregate_sha256": observed_receipt.get("aggregate_sha256"),
            },
        )


def repo_identity(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    head = _git(root, "rev-parse", "HEAD").strip()
    # Probe artifacts are intentionally untracked and may be written between
    # prepare and run.  Bind tracked repository state here; every harness and
    # production owner file (including untracked new owners) is content-bound
    # separately through ``source_owner_identity``.
    status = _git(root, "status", "--short", "--untracked-files=no")
    diff = _git(root, "diff", "--binary", "HEAD", "--", ".")
    return {
        "root": str(root),
        "head": _git_object_id(head, "git HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def source_owner_identity(
    repo_root: str | Path, paths: Sequence[str]
) -> list[dict[str, str]]:
    root = Path(repo_root).expanduser().resolve()
    rows = []
    for relative in paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ParityContractError(
                "source owner escaped repository root",
                code="qwen.parity.source_owner_path",
                context={"path": str(path)},
                cause=exc,
            ) from exc
        rows.append({"path": relative, "sha256": sha256_file(path)})
    return rows


def _safe_relative_weight_path(value: Any) -> str:
    text = _non_empty_text(value, "weight shard path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or not text.endswith(".safetensors")
    ):
        raise ParityContractError(
            "weight shard declaration is not a safe relative safetensors path",
            code="qwen.parity.weight_shard_path",
            context={"path": text[:240]},
        )
    return path.as_posix()


def _preflight_weight_file_set(paths: Sequence[Path]) -> None:
    total_bytes = 0
    for path in paths:
        try:
            file_stat = path.lstat()
        except OSError as exc:
            raise ParityContractError(
                "declared weight identity file is unavailable",
                code="qwen.parity.weight_file_unavailable",
                context={"path": str(path), "error": type(exc).__name__},
                cause=exc,
            ) from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise ParityContractError(
                "declared weight identity file must be a regular non-symlink",
                code="qwen.parity.weight_file_type",
                context={"path": str(path)},
            )
        if file_stat.st_size > MAX_WEIGHT_SHARD_BYTES:
            raise ParityContractError(
                "declared weight identity file exceeds its byte bound",
                code="qwen.parity.weight_file_bound",
                context={
                    "path": str(path),
                    "size_bytes": file_stat.st_size,
                    "maximum": MAX_WEIGHT_SHARD_BYTES,
                },
            )
        total_bytes += int(file_stat.st_size)
        if total_bytes > MAX_WEIGHT_TOTAL_BYTES:
            raise ParityContractError(
                "base-model weight payload exceeds the total byte bound",
                code="qwen.parity.weight_total_bound",
                context={
                    "total_bytes": total_bytes,
                    "maximum": MAX_WEIGHT_TOTAL_BYTES,
                },
            )


def _validate_requested_weight_hash_workers(max_workers: int | None) -> None:
    if max_workers is None:
        return
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ParityContractError(
            "base-model weight hash workers must be an integer",
            code="qwen.parity.weight_hash_workers",
            context={"value_type": type(max_workers).__name__},
        )
    if max_workers <= 0:
        raise ParityContractError(
            "base-model weight hash workers must be positive",
            code="qwen.parity.weight_hash_workers",
            context={"max_workers": max_workers},
        )


def _resolve_weight_hash_workers(
    max_workers: int | None,
    *,
    payload_file_count: int,
) -> int:
    _validate_requested_weight_hash_workers(max_workers)
    if payload_file_count <= 0:
        raise ParityContractError(
            "base-model weight hash payload inventory is empty",
            code="qwen.parity.weight_layout",
            context={"payload_file_count": payload_file_count},
        )
    if max_workers is not None:
        return min(max_workers, payload_file_count, 4)
    available_cpus = os.cpu_count() or 1
    if available_cpus <= 0:
        available_cpus = 1
    return min(payload_file_count, 4, available_cpus)


def _weight_hash_executor_error(
    exc: Exception,
    *,
    stage: str,
    path: Path | None = None,
) -> ParityContractError:
    context: dict[str, Any] = {
        "stage": stage,
        "error": type(exc).__name__,
    }
    if path is not None:
        context["path"] = path.name
    return ParityContractError(
        "base-model weight hash executor failed",
        code="qwen.parity.weight_hash_executor",
        context=context,
        cause=exc,
    )


def _hash_weight_shards(
    shard_paths: Sequence[Path],
    *,
    root: Path,
    resolved_workers: int,
) -> list[dict[str, Any]]:
    try:
        executor = ThreadPoolExecutor(max_workers=resolved_workers)
    except Exception as exc:
        raise _weight_hash_executor_error(exc, stage="create") from exc

    futures: list[Future[tuple[dict[str, Any], bytes]]] = []
    primary_error: BaseException | None = None
    try:
        submit_error: Exception | None = None
        for shard_path in shard_paths:
            try:
                futures.append(
                    executor.submit(
                        _stable_file_identity,
                        shard_path,
                        root=root,
                        max_bytes=MAX_WEIGHT_SHARD_BYTES,
                    )
                )
            except Exception as exc:
                submit_error = exc
                break

        results: list[dict[str, Any] | None] = [None] * len(futures)
        contract_errors: list[tuple[int, ParityContractError]] = []
        operational_errors: list[tuple[int, Exception]] = []
        failure_seen = submit_error is not None
        for index, future in enumerate(futures):
            if failure_seen:
                future.cancel()
            try:
                results[index] = future.result()[0]
            except ParityContractError as exc:
                contract_errors.append((index, exc))
                failure_seen = True
                for pending in futures[index + 1 :]:
                    pending.cancel()
            except Exception as exc:
                operational_errors.append((index, exc))
                failure_seen = True
                for pending in futures[index + 1 :]:
                    pending.cancel()

        if contract_errors:
            raise min(contract_errors, key=lambda item: item[0])[1]
        if submit_error is not None:
            raise _weight_hash_executor_error(submit_error, stage="submit")
        if operational_errors:
            failed_index, operational_error = min(
                operational_errors,
                key=lambda item: item[0],
            )
            raise _weight_hash_executor_error(
                operational_error,
                stage="result",
                path=shard_paths[failed_index],
            )
        if len(results) != len(shard_paths) or any(item is None for item in results):
            raise ParityContractError(
                "base-model weight hash executor returned an incomplete inventory",
                code="qwen.parity.weight_hash_executor",
                context={
                    "stage": "result",
                    "expected": len(shard_paths),
                    "observed": sum(item is not None for item in results),
                },
            )
        return [item for item in results if item is not None]
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception as exc:
            if primary_error is None:
                raise _weight_hash_executor_error(exc, stage="shutdown") from exc


def _stable_file_identity(
    path: Path,
    *,
    root: Path,
    max_bytes: int,
    return_bytes: bool = False,
) -> tuple[dict[str, Any], bytes]:
    candidate = path
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ParityContractError(
            "weight identity path escaped the model root",
            code="qwen.parity.weight_shard_path",
            context={"path": str(candidate), "root": str(root)},
            cause=exc,
        ) from exc
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                raise ParityContractError(
                    "weight identity path contains a symlink",
                    code="qwen.parity.weight_file_type",
                    context={"path": relative, "symlink_component": str(cursor)},
                )
        except OSError as exc:
            raise ParityContractError(
                "weight identity path component is unavailable",
                code="qwen.parity.weight_file_unavailable",
                context={"path": relative, "error": type(exc).__name__},
                cause=exc,
            ) from exc
    try:
        path_stat = candidate.lstat()
    except OSError as exc:
        raise ParityContractError(
            "declared weight identity file is unavailable",
            code="qwen.parity.weight_file_unavailable",
            context={"path": relative, "error": type(exc).__name__},
            cause=exc,
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ParityContractError(
            "declared weight identity file must be a regular non-symlink",
            code="qwen.parity.weight_file_type",
            context={"path": relative},
        )
    if path_stat.st_size > max_bytes:
        raise ParityContractError(
            "declared weight identity file exceeds its byte bound",
            code="qwen.parity.weight_file_bound",
            context={
                "path": relative,
                "size_bytes": path_stat.st_size,
                "maximum": max_bytes,
            },
        )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    observed_bytes = 0
    try:
        with candidate.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                observed_bytes += len(chunk)
                if return_bytes:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ParityContractError(
            "declared weight identity file could not be read",
            code="qwen.parity.weight_file_unavailable",
            context={"path": relative, "error": type(exc).__name__},
            cause=exc,
        ) from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ParityContractError(
            "declared weight identity file changed while hashing",
            code="qwen.parity.weight_file_drift",
            context={"path": relative},
        )
    if observed_bytes != before.st_size:
        raise ParityContractError(
            "declared weight identity file produced a short snapshot",
            code="qwen.parity.weight_file_drift",
            context={
                "path": relative,
                "expected": before.st_size,
                "observed": observed_bytes,
            },
        )
    payload = b"".join(chunks) if return_bytes else b""
    return (
        {
            "path": relative,
            "size_bytes": int(before.st_size),
            "sha256": digest.hexdigest(),
        },
        payload,
    )


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParityContractError(
            "repository identity command failed",
            code="qwen.parity.git_identity",
            context={"args": list(args), "error": type(exc).__name__},
            cause=exc,
        ) from exc
    return result.stdout


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, owner: str
) -> None:
    if not isinstance(value, Mapping):
        raise ParityContractError(
            "parity object must be a mapping",
            code="qwen.parity.mapping",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    observed = set(value)
    if observed != expected:
        raise ParityContractError(
            "parity object fields differ from strict schema",
            code="qwen.parity.fields",
            context={
                "owner": owner,
                "missing": sorted(expected - observed),
                "unknown": sorted(observed - expected),
            },
        )


def _mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParityContractError(
            "parity field must be a mapping",
            code="qwen.parity.mapping",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    return value


def _list_of_mappings(value: Any, owner: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ParityContractError(
            "parity field must be a list of mappings",
            code="qwen.parity.mapping_list",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    return value


def _non_empty_text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityContractError(
            "parity field must be a non-empty string",
            code="qwen.parity.text",
            context={"owner": owner, "value_type": type(value).__name__},
        )
    return value


def _sha256_text(value: Any, owner: str) -> str:
    text = _non_empty_text(value, owner)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ParityContractError(
            "parity identity must be a lowercase SHA-256 digest",
            code="qwen.parity.sha256",
            context={"owner": owner},
        )
    return text


def _git_object_id(value: Any, owner: str) -> str:
    text = _non_empty_text(value, owner)
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ParityContractError(
            "repository HEAD must be a lowercase Git object id",
            code="qwen.parity.git_object_id",
            context={"owner": owner, "length": len(text)},
        )
    return text


def _non_negative_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParityContractError(
            "parity field must be a non-negative integer",
            code="qwen.parity.integer",
            context={"owner": owner, "value": value},
        )
    return value


def _positive_int(value: Any, owner: str) -> int:
    integer = _non_negative_int(value, owner)
    if integer <= 0:
        raise ParityContractError(
            "parity field must be a positive integer",
            code="qwen.parity.integer",
            context={"owner": owner, "value": value},
        )
    return integer


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


__all__ = [
    "MAX_WEIGHT_DECLARATIONS",
    "MAX_WEIGHT_INDEX_BYTES",
    "MAX_WEIGHT_SHARDS",
    "MAX_WEIGHT_SHARD_BYTES",
    "MAX_WEIGHT_TOTAL_BYTES",
    "MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA",
    "MODEL_WEIGHT_IDENTITY_SCHEMA",
    "ParityContractError",
    "assert_absent_artifact_target",
    "assert_model_weight_identity_equal",
    "base_model_weight_identity",
    "base_model_weight_identity_with_execution_policy",
    "canonical_json_bytes",
    "repo_identity",
    "sha256_file",
    "sha256_json",
    "source_owner_identity",
    "validate_model_weight_identity",
    "write_strict_json_atomic",
]
