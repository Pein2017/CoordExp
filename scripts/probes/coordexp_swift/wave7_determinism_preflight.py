#!/usr/bin/env python3
"""Two-launch, eight-rank CUDA determinism plumbing preflight.

The receipt produced by this probe is plumbing-only evidence.  It does not load
the training model and cannot establish training quality, throughput, or exact
resume correctness.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Protocol, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


PLAN_SCHEMA = "coordexp-swift-wave7-determinism-preflight-plan-v4"
RUNTIME_ADMISSION_SCHEMA = "coordexp-swift-wave7-r5-runtime-admission-v1"
NATIVE_REFERENCE_SCHEMA = "coordexp-swift-wave8-native-runtime-attestation-v1"
ATTEMPT_MARKER_SCHEMA = (
    "coordexp-swift-wave7-determinism-preflight-attempt-start-marker-v4"
)
RANK_RECEIPT_SCHEMA = "coordexp-swift-wave7-determinism-preflight-rank-receipt-v1"
TERMINAL_RECEIPT_SCHEMA = "coordexp-swift-wave7-r5-determinism-preflight-v4"

WORLD_SIZE = 8
LAUNCH_IDS = ("launch-a", "launch-b")
STRICT_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
}
SEED = 17
HEX_LENGTH = 64
MAX_CAPTURE_BYTES = 64 * 1024
GPU_BASELINE_SAMPLE_COUNT = 2
GPU_BASELINE_SAMPLE_INTERVAL_SECONDS = 2.0
GPU_REQUIRED_TOTAL_MEMORY_MIB = 81920
GPU_BASELINE_MAX_MEMORY_USED_MIB = 49152
GPU_BASELINE_MIN_MEMORY_HEADROOM_MIB = 32768
_CONTROLLER_TOKEN_ENV = "COORDEXP_WAVE7_CONTROLLER_TOKEN"
_CONTROLLER_PID_ENV = "COORDEXP_WAVE7_CONTROLLER_PID"
_CANONICAL_NATIVE_REFERENCE_RELATIVE_PATH = Path(
    "outputs/probes/coordexp_swift/wave8_native_runtime/2026-08-10-r2/receipt.json"
)
_CANONICAL_NATIVE_REFERENCE_FILE_SHA256 = (
    "d88c9c4ded7c698786c54721f8fbbffdad1467e467f845f72dbdb8cfe9eb75ac"
)
_CANONICAL_NATIVE_REFERENCE_PAYLOAD_SHA256 = (
    "829d86ec8d977f30d37e8e979acba6527b3640180825e551785078a4876b1043"
)
_LATE_NATIVE_RELATIVE_PATHS = {
    "libcublas": ("nvidia-cublas-cu12", "nvidia/cublas/lib/libcublas.so.12"),
    "libnccl": ("nvidia-nccl-cu12", "nvidia/nccl/lib/libnccl.so.2"),
}


class PreflightError(RuntimeError):
    """Fail-closed preflight contract error."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        process_record: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.process_record = None if process_record is None else dict(process_record)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreflightError(
            "artifact is not strict JSON", code="preflight.strict_json"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PreflightError(
            f"identity file is unreadable: {path}",
            code="preflight.identity_file",
        ) from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def finalize_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict self-authenticating immutable-receipt payload."""

    value = dict(payload)
    if "receipt_payload_sha256" in value:
        raise PreflightError(
            "receipt is already finalized", code="preflight.receipt_finalized"
        )
    value["receipt_payload_sha256"] = _sha256_bytes(_canonical_json_bytes(value))
    return value


def _finalize_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "plan_payload_sha256" in value:
        raise PreflightError(
            "plan is already finalized", code="preflight.plan_finalized"
        )
    value["plan_payload_sha256"] = _sha256_bytes(_canonical_json_bytes(value))
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise PreflightError(
            f"strict JSON artifact is unreadable: {path}",
            code="preflight.json_read",
        ) from exc
    return _load_json_bytes(encoded, path=path)


def _load_json_bytes(encoded: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PreflightError(
            f"strict JSON artifact is unreadable: {path}",
            code="preflight.json_read",
        ) from exc
    if not isinstance(value, dict):
        raise PreflightError(
            "strict JSON artifact root must be an object",
            code="preflight.json_shape",
        )
    _canonical_json_bytes(value)
    return value


def _read_exact_regular_file(path: str | Path, *, owner: str) -> tuple[Path, bytes]:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise PreflightError(
            f"{owner} must be an absolute path", code="preflight.path_absolute"
        )
    try:
        initial_stat = requested.lstat()
    except OSError as exc:
        raise PreflightError(
            f"{owner} file is unavailable: {requested}",
            code="preflight.plan_file",
        ) from exc
    if stat.S_ISLNK(initial_stat.st_mode) or not stat.S_ISREG(initial_stat.st_mode):
        raise PreflightError(
            f"{owner} must be an exact regular non-symlink file: {requested}",
            code="preflight.plan_file",
        )
    try:
        resolved = requested.resolve(strict=True)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(requested, flags)
    except (OSError, RuntimeError) as exc:
        raise PreflightError(
            f"{owner} file is unavailable: {requested}",
            code="preflight.plan_file",
        ) from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise PreflightError(
                f"{owner} must be an exact regular non-symlink file: {requested}",
                code="preflight.plan_file",
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    except OSError as exc:
        raise PreflightError(
            f"{owner} file is unreadable: {requested}",
            code="preflight.plan_file",
        ) from exc
    finally:
        os.close(descriptor)
    return resolved, b"".join(chunks)


def _read_bound_plan_file(
    path: str | Path, *, expected_file_sha256: str
) -> tuple[Path, bytes]:
    if not _is_sha256(expected_file_sha256):
        raise PreflightError(
            "expected plan file SHA-256 is malformed",
            code="preflight.plan_file_sha",
        )
    resolved, encoded = _read_exact_regular_file(path, owner="plan")
    if _sha256_bytes(encoded) != expected_file_sha256:
        raise PreflightError(
            "plan file SHA-256 does not match",
            code="preflight.plan_file_sha",
        )
    return resolved, encoded


def _validate_finalized(
    value: Mapping[str, Any], *, digest_field: str, owner: str
) -> dict[str, Any]:
    payload = dict(value)
    observed = payload.pop(digest_field, None)
    expected = _sha256_bytes(_canonical_json_bytes(payload))
    if not _is_sha256(observed) or observed != expected:
        raise PreflightError(
            f"{owner} self-authentication digest is invalid",
            code=f"preflight.{owner}_digest",
        )
    return dict(value)


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, owner: str) -> None:
    if set(value) != expected:
        raise PreflightError(
            f"{owner} fields do not match its schema",
            code=f"preflight.{owner}_schema",
        )


def _absolute_path(value: str | Path, *, owner: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise PreflightError(
            f"{owner} must be an absolute path", code="preflight.path_absolute"
        )
    return requested.resolve(strict=False)


def _assert_absent(path: Path, *, directory: bool = False) -> None:
    if path.exists() or path.is_symlink():
        raise PreflightError(
            f"artifact target already exists: {path}",
            code="preflight.artifact_collision",
        )
    parent = path.parent
    if not parent.is_dir():
        raise PreflightError(
            f"artifact parent does not exist: {parent}",
            code="preflight.artifact_parent",
        )
    if parent.is_symlink():
        raise PreflightError(
            f"artifact parent must not be a symlink: {parent}",
            code="preflight.artifact_symlink",
        )
    if directory and path.suffix:
        # A suffix is allowed, but calling out this branch keeps directory intent
        # explicit for static reviewers.
        return


def _publish_absent(path: Path, payload: Mapping[str, Any]) -> None:
    _assert_absent(path)
    encoded = _canonical_json_bytes(dict(payload)) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PreflightError(
                f"artifact target already exists: {path}",
                code="preflight.artifact_collision",
            ) from exc
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


def _file_binding(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "payload_sha256": receipt.get("receipt_payload_sha256"),
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
    }


def _regular_file_identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise PreflightError(
            f"regular file identity is unavailable: {path}",
            code="preflight.identity_file",
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise PreflightError(
            f"identity target is not a regular file: {resolved}",
            code="preflight.identity_file",
        )
    return {
        "path": str(resolved),
        "size_bytes": file_stat.st_size,
        "file_sha256": _sha256_file(resolved),
    }


def _resolve_accelerate_identity(requested: str) -> dict[str, Any]:
    resolved_text = shutil.which(requested) if os.sep not in requested else requested
    if not resolved_text:
        raise PreflightError(
            "accelerate executable cannot be resolved",
            code="preflight.accelerate_identity",
        )
    identity = _regular_file_identity(Path(resolved_text))
    path = Path(identity["path"])
    if not os.access(path, os.X_OK):
        raise PreflightError(
            "accelerate executable is not executable",
            code="preflight.accelerate_identity",
        )
    try:
        version = distribution("accelerate").version
    except PackageNotFoundError as exc:
        raise PreflightError(
            "accelerate distribution metadata is unavailable",
            code="preflight.accelerate_identity",
        ) from exc
    if not isinstance(version, str) or not version:
        raise PreflightError(
            "accelerate distribution version identity is empty",
            code="preflight.accelerate_identity",
        )
    return {
        **identity,
        "distribution_name": "accelerate",
        "distribution_version": version,
    }


def _validate_accelerate_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError(
            "accelerate identity is malformed",
            code="preflight.accelerate_identity",
        )
    _exact_fields(
        value,
        {
            "path",
            "size_bytes",
            "file_sha256",
            "distribution_name",
            "distribution_version",
        },
        owner="accelerate_identity",
    )
    try:
        observed = _resolve_accelerate_identity(value.get("path", ""))
    except PreflightError as exc:
        raise PreflightError(
            "accelerate executable identity drifted",
            code="preflight.accelerate_identity",
        ) from exc
    if observed != value:
        raise PreflightError(
            "accelerate executable identity drifted",
            code="preflight.accelerate_identity",
        )
    return dict(value)


def _select_available_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            handle.bind(("127.0.0.1", 0))
            handle.listen(1)
            sockets.append(handle)
        ports = [int(handle.getsockname()[1]) for handle in sockets]
    except OSError as exc:
        raise PreflightError(
            "available launch ports cannot be selected",
            code="controller.port_unavailable",
        ) from exc
    finally:
        for handle in sockets:
            handle.close()
    if len(set(ports)) != count:
        raise PreflightError(
            "selected launch ports are not unique",
            code="controller.port_unavailable",
        )
    return ports


def _check_launch_ports_available(plan: Mapping[str, Any]) -> None:
    sockets: list[socket.socket] = []
    try:
        for launch in plan["launch_contract"]:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            handle.bind(("127.0.0.1", launch["main_process_port"]))
            handle.listen(1)
            sockets.append(handle)
    except OSError as exc:
        raise PreflightError(
            "a bound launch port is unavailable before attempt publication",
            code="controller.port_unavailable",
        ) from exc
    finally:
        for handle in sockets:
            handle.close()


def _validate_native_reference_receipt(
    path: Path, *, expected_file_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_path = (
        Path(__file__).resolve().parents[3] / _CANONICAL_NATIVE_REFERENCE_RELATIVE_PATH
    ).resolve(strict=False)
    if (
        path.resolve(strict=False) != canonical_path
        or expected_file_sha256 != _CANONICAL_NATIVE_REFERENCE_FILE_SHA256
        or _sha256_file(path) != _CANONICAL_NATIVE_REFERENCE_FILE_SHA256
    ):
        raise PreflightError(
            "native runtime reference file SHA-256 does not match",
            code="preflight.native_reference",
        )
    receipt = _load_json(path)
    _exact_fields(
        receipt,
        {
            "schema",
            "created_at",
            "repository_root",
            "source_identity",
            "attention_backend",
            "pinned_runtime_baseline",
            "cuda",
            "mapped_native_execution",
            "terminal_status",
            "receipt_sha256",
        },
        owner="native_runtime_reference",
    )
    _validate_finalized(
        receipt,
        digest_field="receipt_sha256",
        owner="native_runtime_reference",
    )
    from src.artifacts import provenance as provenance_module

    source_path = Path(provenance_module.__file__).resolve()
    expected_source = {"path": str(source_path), "sha256": _sha256_file(source_path)}
    baseline = receipt.get("pinned_runtime_baseline")
    mapped = receipt.get("mapped_native_execution")
    if (
        receipt.get("schema") != NATIVE_REFERENCE_SCHEMA
        or receipt.get("receipt_sha256") != _CANONICAL_NATIVE_REFERENCE_PAYLOAD_SHA256
        or receipt.get("terminal_status") != "passed"
        or receipt.get("source_identity") != expected_source
        or receipt.get("attention_backend") != "flash_attention_2"
        or not isinstance(baseline, dict)
        or baseline
        != {
            "schema_version": 3,
            "baseline_sha256": provenance_module.PINNED_RUNTIME_BASELINE_SHA256,
            "admitted": True,
        }
        or not isinstance(mapped, dict)
        or mapped.get("schema_version") != 1
        or mapped.get("cuda_initialized") is not True
        or mapped.get("admitted") is not True
        or mapped.get("mismatches") != []
    ):
        raise PreflightError(
            "native runtime reference is not the exact passed reference",
            code="preflight.native_reference",
        )
    return receipt, {
        "path": str(path),
        "file_sha256": _CANONICAL_NATIVE_REFERENCE_FILE_SHA256,
        "payload_sha256": receipt["receipt_sha256"],
        "schema": receipt["schema"],
        "status": receipt["terminal_status"],
    }


def _late_native_expectations(provenance: Mapping[str, Any]) -> dict[str, Any]:
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, dict):
        raise PreflightError(
            "runtime provenance dependencies are malformed",
            code="preflight.runtime_admission",
        )
    flash = dependencies.get("flash_attn_2_cuda")
    origin = flash.get("imported_origin") if isinstance(flash, dict) else None
    flash_path_value = origin.get("value") if isinstance(origin, dict) else None
    if origin.get("status") != "available" or not isinstance(flash_path_value, str):
        raise PreflightError(
            "flash-attn native origin is unavailable in runtime provenance",
            code="preflight.runtime_admission",
        )
    paths = {
        "flash_attention_2": (
            "flash-attn",
            Path(flash_path_value),
        )
    }
    for component, (
        distribution_name,
        relative_path,
    ) in _LATE_NATIVE_RELATIVE_PATHS.items():
        try:
            target = Path(distribution(distribution_name).locate_file(relative_path))
        except PackageNotFoundError as exc:
            raise PreflightError(
                f"late native distribution is unavailable: {distribution_name}",
                code="preflight.runtime_admission",
            ) from exc
        paths[component] = (distribution_name, target)
    expectations: dict[str, Any] = {}
    for component, (distribution_name, path) in sorted(paths.items()):
        identity = _regular_file_identity(path)
        expectations[component] = {
            "distribution": distribution_name,
            "soname": Path(identity["path"]).name,
            **identity,
        }
    return expectations


def _validate_late_native_expectations(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "flash_attention_2",
        "libcublas",
        "libnccl",
    }:
        raise PreflightError(
            "late native expectation inventory is not exact",
            code="preflight.runtime_admission",
        )
    for component, identity in value.items():
        if not isinstance(identity, dict):
            raise PreflightError(
                f"late native expectation is malformed: {component}",
                code="preflight.runtime_admission",
            )
        _exact_fields(
            identity,
            {"distribution", "soname", "path", "size_bytes", "file_sha256"},
            owner="late_native_expectation",
        )
        if (
            not isinstance(identity["distribution"], str)
            or not identity["distribution"]
            or not isinstance(identity["soname"], str)
            or not identity["soname"]
            or Path(identity["path"]).name != identity["soname"]
            or not isinstance(identity["size_bytes"], int)
            or isinstance(identity["size_bytes"], bool)
            or identity["size_bytes"] < 0
            or not _is_sha256(identity["file_sha256"])
        ):
            raise PreflightError(
                f"late native expectation is rejected: {component}",
                code="preflight.runtime_admission",
            )
        _absolute_path(identity["path"], owner="late native expectation")
    return dict(value)


def _validate_runtime_admission_receipt(
    path: Path, *, expected_file_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, binding = _validate_admission_receipt(
        path,
        expected_file_sha256=expected_file_sha256,
        owner="runtime_admission",
    )
    _exact_fields(
        receipt,
        {
            "schema",
            "status",
            "model_loaded",
            "cuda_initialized",
            "repository_root",
            "source_identity",
            "native_reference",
            "attention_backend",
            "provenance",
            "pinned_runtime_baseline",
            "late_native_expectations",
            "receipt_payload_sha256",
        },
        owner="runtime_admission",
    )
    _validate_finalized(
        receipt, digest_field="receipt_payload_sha256", owner="runtime_admission"
    )
    from src.artifacts import provenance as provenance_module

    source_path = Path(provenance_module.__file__).resolve()
    source_identity = {
        "path": str(source_path),
        "file_sha256": _sha256_file(source_path),
    }
    provenance = receipt.get("provenance")
    native_reference = receipt.get("native_reference")
    if (
        receipt.get("schema") != RUNTIME_ADMISSION_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("model_loaded") is not False
        or receipt.get("cuda_initialized") is not False
        or receipt.get("repository_root") != str(Path(__file__).resolve().parents[3])
        or receipt.get("source_identity") != source_identity
        or receipt.get("attention_backend") != "flash_attention_2"
        or not isinstance(provenance, dict)
        or set(provenance)
        != {"schema_version", "repository", "dependencies", "runtime"}
        or provenance.get("schema_version") != 1
        or not isinstance(native_reference, dict)
    ):
        raise PreflightError(
            "runtime admission identity is rejected",
            code="preflight.runtime_admission",
        )
    _exact_fields(
        native_reference,
        {"path", "file_sha256", "payload_sha256", "schema", "status"},
        owner="runtime_native_reference_binding",
    )
    if (
        native_reference.get("schema") != NATIVE_REFERENCE_SCHEMA
        or native_reference.get("status") != "passed"
        or not _is_sha256(native_reference.get("file_sha256"))
        or not _is_sha256(native_reference.get("payload_sha256"))
    ):
        raise PreflightError(
            "runtime admission native reference binding is rejected",
            code="preflight.runtime_admission",
        )
    baseline = provenance_module.require_pinned_runtime_baseline(
        provenance=provenance,
        attention_backend="flash_attention_2",
    )
    if receipt.get("pinned_runtime_baseline") != baseline:
        raise PreflightError(
            "runtime admission baseline comparison drifted",
            code="preflight.runtime_admission",
        )
    _validate_late_native_expectations(receipt.get("late_native_expectations"))
    return receipt, binding


def admit_runtime(
    *,
    receipt_path: str | Path,
    native_reference_path: str | Path,
    expected_native_reference_file_sha256: str,
) -> dict[str, Any]:
    """Publish model-free full provenance admitted against the pinned runtime."""

    target = _absolute_path(receipt_path, owner="runtime admission receipt")
    _assert_absent(target)
    native_path = _absolute_path(native_reference_path, owner="native reference")
    _native, native_binding = _validate_native_reference_receipt(
        native_path,
        expected_file_sha256=expected_native_reference_file_sha256,
    )
    import torch
    from src.artifacts import provenance as provenance_module

    if torch.cuda.is_initialized():
        raise PreflightError(
            "CUDA is already initialized before runtime admission",
            code="preflight.runtime_admission",
        )
    repository_root = Path(__file__).resolve().parents[3]
    provenance = provenance_module.collect_execution_provenance(
        repository_root=repository_root
    )
    baseline = provenance_module.require_pinned_runtime_baseline(
        provenance=provenance,
        attention_backend="flash_attention_2",
    )
    if torch.cuda.is_initialized():
        raise PreflightError(
            "runtime admission initialized CUDA",
            code="preflight.runtime_admission",
        )
    source_path = Path(provenance_module.__file__).resolve()
    receipt = finalize_receipt(
        {
            "schema": RUNTIME_ADMISSION_SCHEMA,
            "status": "passed",
            "model_loaded": False,
            "cuda_initialized": False,
            "repository_root": str(repository_root),
            "source_identity": {
                "path": str(source_path),
                "file_sha256": _sha256_file(source_path),
            },
            "native_reference": native_binding,
            "attention_backend": "flash_attention_2",
            "provenance": provenance,
            "pinned_runtime_baseline": baseline,
            "late_native_expectations": _late_native_expectations(provenance),
        }
    )
    _publish_absent(target, receipt)
    return receipt


def _validate_admission_receipt(
    path: Path, *, expected_file_sha256: str, owner: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_sha256(expected_file_sha256):
        raise PreflightError(
            f"{owner} expected file SHA-256 is malformed",
            code="preflight.admission_sha",
        )
    if _sha256_file(path) != expected_file_sha256:
        raise PreflightError(
            f"{owner} file SHA-256 does not match",
            code="preflight.admission_sha",
        )
    receipt = _load_json(path)
    if receipt.get("status") != "passed":
        raise PreflightError(
            f"{owner} is not passed", code="preflight.admission_status"
        )
    if "receipt_payload_sha256" in receipt:
        _validate_finalized(receipt, digest_field="receipt_payload_sha256", owner=owner)
    return receipt, _file_binding(path, receipt)


def _claim_scope() -> dict[str, Any]:
    return {
        "classification": "plumbing_only",
        "establishes": [
            "two_fresh_eight_rank_launches_execute_the_fixed_synthetic_workload",
            "fixed_rank_digests_are_exactly_equal_across_launches",
            "active_cuda_native_mappings_match_the_pinned_admission",
        ],
        "does_not_establish": [
            "model_loading",
            "training_quality",
            "throughput",
            "performance_under_shared_gpu_occupancy",
            "resource_comparisons_under_shared_gpu_occupancy",
            "exact_resume_correctness",
        ],
    }


def _expected_policy_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "strict_cuda_replay_v1",
        "seed": SEED,
        "deterministic_algorithms": {
            "enabled": True,
            "managed": True,
            "warn_only": False,
        },
        "cudnn": {"benchmark": False, "deterministic": True, "managed": True},
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
        },
        "phase": "pipeline_entry",
        "helper": "transformers.trainer_utils.set_seed",
        "required_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
        },
        "observed_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
        },
        "cuda_initialized": False,
        "surfaces": [
            "python.random",
            "numpy",
            "torch.cpu",
            "torch.cuda_all",
            "torch.other_supported_devices",
        ],
        "applied_before": [
            "cache_preflight",
            "accelerator_setup",
            "qwen_model_load",
            "adapter_setup",
            "special_token_embedding_setup",
            "optimizer_setup",
            "runtime_setup",
        ],
    }


def _check_outer_environment() -> None:
    conflicts = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in sorted(STRICT_ENVIRONMENT.items())
        if os.environ.get(name) not in (None, expected)
    }
    if conflicts:
        raise PreflightError(
            f"outer environment conflicts with the exact launch policy: {conflicts}",
            code="controller.outer_environment",
        )


def prepare_plan(
    *,
    plan_path: str | Path,
    attempt_marker_path: str | Path,
    terminal_receipt_path: str | Path,
    rank_receipt_root: str | Path,
    runtime_receipt_path: str | Path,
    expected_runtime_file_sha256: str,
    native_runtime_receipt_path: str | Path,
    expected_native_runtime_file_sha256: str,
    accelerate_executable: str = "accelerate",
) -> dict[str, Any]:
    """Validate immutable inputs and publish one absent-only execution plan."""

    _check_outer_environment()
    plan_target = _absolute_path(plan_path, owner="plan")
    marker_target = _absolute_path(attempt_marker_path, owner="attempt marker")
    terminal_target = _absolute_path(terminal_receipt_path, owner="terminal receipt")
    rank_root = _absolute_path(rank_receipt_root, owner="rank receipt root")
    runtime_path = _absolute_path(runtime_receipt_path, owner="runtime receipt")
    native_path = _absolute_path(
        native_runtime_receipt_path, owner="native runtime receipt"
    )
    targets = (plan_target, marker_target, terminal_target, rank_root)
    if len(set(targets)) != len(targets):
        raise PreflightError(
            "artifact targets must be mutually exclusive",
            code="preflight.artifact_targets",
        )
    for target in targets:
        _assert_absent(target, directory=target == rank_root)

    runtime_receipt, runtime_binding = _validate_runtime_admission_receipt(
        runtime_path,
        expected_file_sha256=expected_runtime_file_sha256,
    )
    _native, native_binding = _validate_native_reference_receipt(
        native_path,
        expected_file_sha256=expected_native_runtime_file_sha256,
    )
    if runtime_receipt["native_reference"] != native_binding:
        raise PreflightError(
            "runtime admission does not bind the supplied native reference",
            code="preflight.runtime_admission",
        )
    accelerate_identity = _resolve_accelerate_identity(accelerate_executable)
    ports = _select_available_ports(len(LAUNCH_IDS))
    script_path = Path(__file__).resolve()
    launch_contract = []
    for launch_id, port in zip(LAUNCH_IDS, ports, strict=True):
        launch_contract.append(
            {
                "launch_id": launch_id,
                "num_processes": WORLD_SIZE,
                "main_process_port": port,
                "argv_template": [
                    accelerate_identity["path"],
                    "launch",
                    "--multi_gpu",
                    "--num_processes",
                    str(WORLD_SIZE),
                    "--main_process_port",
                    str(port),
                    str(script_path),
                    "_worker",
                    "--plan",
                    str(plan_target),
                    "--launch-id",
                    launch_id,
                    "--controller-pid",
                    "${CONTROLLER_PID}",
                ],
            }
        )
    payload = {
        "schema": PLAN_SCHEMA,
        "status": "prepared",
        "claim_scope": _claim_scope(),
        "source_identity": {
            "path": str(script_path),
            "file_sha256": _sha256_file(script_path),
        },
        "runtime_admission": runtime_binding,
        "native_runtime_reference": native_binding,
        "accelerate_identity": accelerate_identity,
        "environment": dict(STRICT_ENVIRONMENT),
        "launch_contract": launch_contract,
        "workload_contract": {
            "seed": SEED,
            "cublas": {
                "dtype": "bfloat16",
                "matrix_shape": [32, 32],
                "operation": "matmul_forward_backward",
            },
            "flash_attention_2": {
                "batch_total_tokens": 8,
                "causal": True,
                "cu_seqlens": [0, 3, 8],
                "deterministic": True,
                "dropout_p": 0.0,
                "head_dim": 64,
                "heads": 2,
                "max_seqlen": 5,
            },
            "nccl_all_reduce": {
                "backend": "nccl",
                "dtype": "float32",
                "input": "rank",
                "expected_sum": 28.0,
            },
        },
        "gpu_shared_preexisting_baseline_contract": {
            "device_indices": list(range(WORLD_SIZE)),
            "sample_count": GPU_BASELINE_SAMPLE_COUNT,
            "sample_interval_seconds": GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
            "utilization_percent_range": [0, 100],
            "required_total_memory_mib": GPU_REQUIRED_TOTAL_MEMORY_MIB,
            "max_prelaunch_memory_used_mib": GPU_BASELINE_MAX_MEMORY_USED_MIB,
            "min_prelaunch_memory_headroom_mib": (GPU_BASELINE_MIN_MEMORY_HEADROOM_MIB),
            "compute_process_identity_fields": ["gpu_uuid", "pid"],
            "require_stable_device_inventory": True,
            "require_stable_compute_process_inventory": True,
            "postlaunch_compute_process_policy": "subset_of_preexisting_baseline",
        },
        "artifact_targets": {
            "attempt_marker": str(marker_target),
            "terminal_receipt": str(terminal_target),
            "rank_receipt_root": str(rank_root),
        },
        "process_contract": {
            "launch_timeout_seconds": 180,
            "terminate_grace_seconds": 10,
            "kill_grace_seconds": 10,
            "start_new_session": True,
            "retry_count": 0,
        },
    }
    plan = _finalize_plan(payload)
    _publish_absent(plan_target, plan)
    return plan


def _validate_plan(plan: Mapping[str, Any], *, plan_path: Path) -> dict[str, Any]:
    _exact_fields(
        plan,
        {
            "schema",
            "status",
            "claim_scope",
            "source_identity",
            "runtime_admission",
            "native_runtime_reference",
            "accelerate_identity",
            "environment",
            "launch_contract",
            "workload_contract",
            "gpu_shared_preexisting_baseline_contract",
            "artifact_targets",
            "process_contract",
            "plan_payload_sha256",
        },
        owner="plan",
    )
    _validate_finalized(plan, digest_field="plan_payload_sha256", owner="plan")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "prepared":
        raise PreflightError(
            "plan schema or status is unsupported", code="preflight.plan_schema"
        )
    if plan.get("environment") != STRICT_ENVIRONMENT:
        raise PreflightError(
            "plan environment is not exact", code="preflight.plan_environment"
        )
    launches = plan.get("launch_contract")
    if (
        not isinstance(launches, list)
        or [item.get("launch_id") for item in launches if isinstance(item, dict)]
        != list(LAUNCH_IDS)
        or any(item.get("num_processes") != WORLD_SIZE for item in launches)
    ):
        raise PreflightError(
            "plan launch inventory is not exact",
            code="preflight.plan_launch_inventory",
        )
    observed_ports: list[int] = []
    for launch, expected_id in zip(launches, LAUNCH_IDS, strict=True):
        _exact_fields(
            launch,
            {"launch_id", "num_processes", "main_process_port", "argv_template"},
            owner="plan_launch",
        )
        if (
            launch["launch_id"] != expected_id
            or launch["num_processes"] != WORLD_SIZE
            or not isinstance(launch["main_process_port"], int)
            or isinstance(launch["main_process_port"], bool)
            or launch["main_process_port"] not in range(1024, 65536)
            or not isinstance(launch["argv_template"], list)
            or any(
                not isinstance(part, str) or not part
                for part in launch["argv_template"]
            )
        ):
            raise PreflightError(
                "plan launch command is malformed",
                code="preflight.plan_launch_inventory",
            )
        observed_ports.append(launch["main_process_port"])
    if len(set(observed_ports)) != len(LAUNCH_IDS):
        raise PreflightError(
            "plan launch ports are not unique",
            code="preflight.plan_launch_inventory",
        )
    accelerate_identity = _validate_accelerate_identity(plan.get("accelerate_identity"))
    if any(
        launch["argv_template"][0] != accelerate_identity["path"] for launch in launches
    ):
        raise PreflightError(
            "plan launch executable is not the frozen identity",
            code="preflight.accelerate_identity",
        )
    source = plan.get("source_identity")
    if not isinstance(source, dict):
        raise PreflightError(
            "plan source identity is malformed", code="preflight.plan_source"
        )
    _exact_fields(source, {"path", "file_sha256"}, owner="plan_source")
    source_path = _absolute_path(source.get("path", ""), owner="source")
    if source_path != Path(__file__).resolve() or _sha256_file(
        source_path
    ) != source.get("file_sha256"):
        raise PreflightError(
            "preflight source changed after plan preparation",
            code="preflight.plan_source",
        )
    if plan_path != plan_path.resolve():
        raise PreflightError("plan path must be resolved", code="preflight.plan_path")
    for field in ("runtime_admission", "native_runtime_reference"):
        binding = plan.get(field)
        if not isinstance(binding, dict):
            raise PreflightError(
                f"{field} binding is malformed", code="preflight.admission_binding"
            )
        _exact_fields(
            binding,
            {"path", "file_sha256", "payload_sha256", "schema", "status"},
            owner="admission_binding",
        )
        path = _absolute_path(binding.get("path", ""), owner=field)
        if _sha256_file(path) != binding.get("file_sha256"):
            raise PreflightError(
                f"{field} changed after preparation",
                code="preflight.admission_binding",
            )
        if field == "runtime_admission":
            receipt, observed_binding = _validate_runtime_admission_receipt(
                path, expected_file_sha256=binding["file_sha256"]
            )
        else:
            receipt, observed_binding = _validate_native_reference_receipt(
                path, expected_file_sha256=binding["file_sha256"]
            )
        if binding != observed_binding:
            raise PreflightError(
                f"{field} semantic binding changed",
                code="preflight.admission_binding",
            )
    if plan.get("claim_scope") != _claim_scope():
        raise PreflightError(
            "plan claim scope drifted", code="preflight.plan_claim_scope"
        )
    workload = plan.get("workload_contract")
    if not isinstance(workload, dict):
        raise PreflightError(
            "plan workload contract is malformed", code="preflight.plan_workload"
        )
    _exact_fields(
        workload,
        {"seed", "cublas", "flash_attention_2", "nccl_all_reduce"},
        owner="plan_workload",
    )
    expected_workload = {
        "seed": SEED,
        "cublas": {
            "dtype": "bfloat16",
            "matrix_shape": [32, 32],
            "operation": "matmul_forward_backward",
        },
        "flash_attention_2": {
            "batch_total_tokens": 8,
            "causal": True,
            "cu_seqlens": [0, 3, 8],
            "deterministic": True,
            "dropout_p": 0.0,
            "head_dim": 64,
            "heads": 2,
            "max_seqlen": 5,
        },
        "nccl_all_reduce": {
            "backend": "nccl",
            "dtype": "float32",
            "input": "rank",
            "expected_sum": 28.0,
        },
    }
    if workload != expected_workload:
        raise PreflightError(
            "plan workload contract drifted", code="preflight.plan_workload"
        )
    gpu_baseline = plan.get("gpu_shared_preexisting_baseline_contract")
    process = plan.get("process_contract")
    artifacts = plan.get("artifact_targets")
    if not all(isinstance(item, dict) for item in (gpu_baseline, process, artifacts)):
        raise PreflightError(
            "plan controller contracts are malformed", code="preflight.plan_schema"
        )
    _exact_fields(
        gpu_baseline,
        {
            "device_indices",
            "sample_count",
            "sample_interval_seconds",
            "utilization_percent_range",
            "required_total_memory_mib",
            "max_prelaunch_memory_used_mib",
            "min_prelaunch_memory_headroom_mib",
            "compute_process_identity_fields",
            "require_stable_device_inventory",
            "require_stable_compute_process_inventory",
            "postlaunch_compute_process_policy",
        },
        owner="plan_gpu_shared_baseline",
    )
    _exact_fields(
        process,
        {
            "launch_timeout_seconds",
            "terminate_grace_seconds",
            "kill_grace_seconds",
            "start_new_session",
            "retry_count",
        },
        owner="plan_process",
    )
    _exact_fields(
        artifacts,
        {"attempt_marker", "terminal_receipt", "rank_receipt_root"},
        owner="plan_artifacts",
    )
    if (
        gpu_baseline
        != {
            "device_indices": list(range(WORLD_SIZE)),
            "sample_count": GPU_BASELINE_SAMPLE_COUNT,
            "sample_interval_seconds": GPU_BASELINE_SAMPLE_INTERVAL_SECONDS,
            "utilization_percent_range": [0, 100],
            "required_total_memory_mib": GPU_REQUIRED_TOTAL_MEMORY_MIB,
            "max_prelaunch_memory_used_mib": GPU_BASELINE_MAX_MEMORY_USED_MIB,
            "min_prelaunch_memory_headroom_mib": (GPU_BASELINE_MIN_MEMORY_HEADROOM_MIB),
            "compute_process_identity_fields": ["gpu_uuid", "pid"],
            "require_stable_device_inventory": True,
            "require_stable_compute_process_inventory": True,
            "postlaunch_compute_process_policy": "subset_of_preexisting_baseline",
        }
        or process["start_new_session"] is not True
        or process["retry_count"] != 0
    ):
        raise PreflightError(
            "plan controller contracts drifted", code="preflight.plan_schema"
        )
    artifact_paths = [
        _absolute_path(artifacts[name], owner=name) for name in sorted(artifacts)
    ]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise PreflightError(
            "plan artifact targets overlap", code="preflight.artifact_targets"
        )
    return dict(plan)


def _command_for_launch(
    plan: Mapping[str, Any],
    *,
    launch_id: str,
    controller_pid: int,
) -> list[str]:
    launch = next(
        item for item in plan["launch_contract"] if item["launch_id"] == launch_id
    )
    replacements = {
        "${CONTROLLER_PID}": str(controller_pid),
    }
    return [replacements.get(value, value) for value in launch["argv_template"]]


def _redacted_command_for_launch(
    plan: Mapping[str, Any], *, launch_id: str, controller_pid: int
) -> list[str]:
    return _command_for_launch(
        plan,
        launch_id=launch_id,
        controller_pid=controller_pid,
    )


def _parse_csv_rows(output: str) -> list[list[str]]:
    return [
        [part.strip() for part in line.split(",")]
        for line in output.splitlines()
        if line.strip()
    ]


def _collect_gpu_sample(
    contract: Mapping[str, Any], sample_index: int
) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(
            "GPU baseline sampling failed", code="controller.gpu_sampling"
        ) from exc
    rows = _parse_csv_rows(result.stdout)
    compute_rows = _parse_csv_rows(process_result.stdout)
    expected_indices = contract["device_indices"]
    observed: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != 5:
            raise PreflightError(
                "GPU sample is malformed", code="controller.gpu_sampling"
            )
        try:
            memory_used_mib = int(row[3])
            memory_total_mib = int(row[4])
            observed.append(
                {
                    "index": int(row[0]),
                    "gpu_uuid": row[1],
                    "utilization_percent": int(row[2]),
                    "memory_used_mib": memory_used_mib,
                    "memory_total_mib": memory_total_mib,
                    "memory_headroom_mib": memory_total_mib - memory_used_mib,
                }
            )
        except ValueError as exc:
            raise PreflightError(
                "GPU sample numeric field is malformed",
                code="controller.gpu_sampling",
            ) from exc
    compute_processes: list[dict[str, Any]] = []
    for row in compute_rows:
        if len(row) != 2:
            raise PreflightError(
                "GPU compute-process sample is malformed",
                code="controller.gpu_sampling",
            )
        try:
            compute_processes.append({"gpu_uuid": row[0], "pid": int(row[1])})
        except ValueError as exc:
            raise PreflightError(
                "GPU compute-process PID is malformed",
                code="controller.gpu_sampling",
            ) from exc
    if [item["index"] for item in observed] != expected_indices:
        raise PreflightError(
            "GPU index inventory is not exact", code="controller.gpu_inventory"
        )
    gpu_uuids = [item["gpu_uuid"] for item in observed]
    if (
        len(set(gpu_uuids)) != WORLD_SIZE
        or any(
            not isinstance(value, str) or not value.startswith("GPU-")
            for value in gpu_uuids
        )
        or any(
            item["utilization_percent"] not in range(0, 101)
            or item["memory_used_mib"] < 0
            for item in observed
        )
    ):
        raise PreflightError(
            "GPU sample values are outside the bounded contract",
            code="controller.gpu_sampling",
        )
    process_keys = [(item["gpu_uuid"], item["pid"]) for item in compute_processes]
    if len(set(process_keys)) != len(process_keys) or any(
        gpu_uuid not in gpu_uuids
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        for gpu_uuid, pid in process_keys
    ):
        raise PreflightError(
            "GPU compute-process inventory is not exact",
            code="controller.gpu_process_inventory",
        )
    return {
        "sample_index": sample_index,
        "sample_monotonic_ns": time.monotonic_ns(),
        "gpus": observed,
        "compute_processes": sorted(
            compute_processes, key=lambda value: (value["gpu_uuid"], value["pid"])
        ),
    }


def _sample_device_inventory(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"index": item["index"], "gpu_uuid": item["gpu_uuid"]}
        for item in sample["gpus"]
    ]


def _sample_compute_processes(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(item) for item in sample["compute_processes"]],
        key=lambda value: (value["gpu_uuid"], value["pid"]),
    )


def _validate_gpu_sample_evidence(
    value: Any, *, contract: Mapping[str, Any], sample_index: int
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError(
            "GPU sample evidence is malformed", code="controller.gpu_sampling"
        )
    _exact_fields(
        value,
        {
            "sample_index",
            "sample_monotonic_ns",
            "gpus",
            "compute_processes",
        },
        owner="gpu_sample",
    )
    gpus = value.get("gpus")
    processes = value.get("compute_processes")
    if (
        value.get("sample_index") != sample_index
        or not isinstance(value.get("sample_monotonic_ns"), int)
        or isinstance(value.get("sample_monotonic_ns"), bool)
        or value["sample_monotonic_ns"] < 0
        or not isinstance(gpus, list)
        or not isinstance(processes, list)
        or len(gpus) != WORLD_SIZE
    ):
        raise PreflightError(
            "GPU sample evidence is malformed", code="controller.gpu_sampling"
        )
    for gpu in gpus:
        if not isinstance(gpu, dict):
            raise PreflightError("GPU row is malformed", code="controller.gpu_sampling")
        _exact_fields(
            gpu,
            {
                "index",
                "gpu_uuid",
                "utilization_percent",
                "memory_used_mib",
                "memory_total_mib",
                "memory_headroom_mib",
            },
            owner="gpu_sample_row",
        )
        if (
            not isinstance(gpu["index"], int)
            or isinstance(gpu["index"], bool)
            or not isinstance(gpu["gpu_uuid"], str)
            or not gpu["gpu_uuid"].startswith("GPU-")
            or not isinstance(gpu["utilization_percent"], int)
            or isinstance(gpu["utilization_percent"], bool)
            or gpu["utilization_percent"] not in range(0, 101)
            or not isinstance(gpu["memory_used_mib"], int)
            or isinstance(gpu["memory_used_mib"], bool)
            or gpu["memory_used_mib"] < 0
        ):
            raise PreflightError(
                "GPU row is outside the bounded contract",
                code="controller.gpu_sampling",
            )
        if (
            not isinstance(gpu["memory_total_mib"], int)
            or isinstance(gpu["memory_total_mib"], bool)
            or gpu["memory_total_mib"] != contract["required_total_memory_mib"]
            or not isinstance(gpu["memory_headroom_mib"], int)
            or isinstance(gpu["memory_headroom_mib"], bool)
            or gpu["memory_headroom_mib"]
            != gpu["memory_total_mib"] - gpu["memory_used_mib"]
        ):
            raise PreflightError(
                "GPU memory total/headroom geometry is not exact",
                code="controller.gpu_memory_geometry",
            )
    if [gpu["index"] for gpu in gpus] != contract["device_indices"] or len(
        {gpu["gpu_uuid"] for gpu in gpus}
    ) != WORLD_SIZE:
        raise PreflightError(
            "GPU UUID/index inventory is not exact",
            code="controller.gpu_inventory",
        )
    gpu_uuids = {gpu["gpu_uuid"] for gpu in gpus}
    process_keys: list[tuple[str, int]] = []
    for process in processes:
        if not isinstance(process, dict):
            raise PreflightError(
                "GPU compute-process row is malformed",
                code="controller.gpu_process_inventory",
            )
        _exact_fields(process, {"gpu_uuid", "pid"}, owner="gpu_compute_process")
        if (
            process["gpu_uuid"] not in gpu_uuids
            or not isinstance(process["pid"], int)
            or isinstance(process["pid"], bool)
            or process["pid"] <= 0
        ):
            raise PreflightError(
                "GPU compute-process row is outside the exact inventory",
                code="controller.gpu_process_inventory",
            )
        process_keys.append((process["gpu_uuid"], process["pid"]))
    if len(set(process_keys)) != len(process_keys):
        raise PreflightError(
            "GPU compute-process rows are duplicated",
            code="controller.gpu_process_inventory",
        )
    return dict(value)


def _collect_stable_gpu_idle_samples(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Collect and admit the exact shared pre-existing GPU baseline."""

    contract = plan["gpu_shared_preexisting_baseline_contract"]
    samples: list[dict[str, Any]] = []
    for index in range(contract["sample_count"]):
        if index:
            time.sleep(contract["sample_interval_seconds"])
        samples.append(
            _validate_gpu_sample_evidence(
                _collect_gpu_sample(contract, index),
                contract=contract,
                sample_index=index,
            )
        )
    if len(samples) != GPU_BASELINE_SAMPLE_COUNT:
        raise PreflightError(
            "GPU baseline sample count is not exact",
            code="controller.gpu_baseline_changed",
        )
    device_inventories = [_sample_device_inventory(sample) for sample in samples]
    process_inventories = [_sample_compute_processes(sample) for sample in samples]
    if device_inventories[0] != device_inventories[1]:
        raise PreflightError(
            "GPU UUID/index inventory changed across baseline samples",
            code="controller.gpu_baseline_changed",
        )
    if process_inventories[0] != process_inventories[1]:
        raise PreflightError(
            "GPU compute-process inventory changed across baseline samples",
            code="controller.gpu_baseline_changed",
        )
    observed_interval_ns = (
        samples[1]["sample_monotonic_ns"] - samples[0]["sample_monotonic_ns"]
    )
    required_interval_ns = int(contract["sample_interval_seconds"] * 1_000_000_000)
    if observed_interval_ns < required_interval_ns:
        raise PreflightError(
            "GPU baseline samples are less than two seconds apart",
            code="controller.gpu_baseline_interval",
        )
    if any(
        gpu["memory_used_mib"] > contract["max_prelaunch_memory_used_mib"]
        or gpu["memory_headroom_mib"] < contract["min_prelaunch_memory_headroom_mib"]
        for sample in samples
        for gpu in sample["gpus"]
    ):
        raise PreflightError(
            "GPU prelaunch memory violates the shared-baseline headroom",
            code="controller.gpu_baseline_headroom",
        )
    return {
        "samples": samples,
        "device_inventory": device_inventories[0],
        "preexisting_compute_processes": process_inventories[0],
    }


def _validate_gpu_baseline(value: Any, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError(
            "shared GPU baseline is malformed", code="preflight.marker_schema"
        )
    _exact_fields(
        value,
        {"samples", "device_inventory", "preexisting_compute_processes"},
        owner="gpu_shared_baseline",
    )
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != GPU_BASELINE_SAMPLE_COUNT:
        raise PreflightError(
            "shared GPU baseline sample count is not exact",
            code="preflight.marker_schema",
        )
    contract = plan["gpu_shared_preexisting_baseline_contract"]
    validated = [
        _validate_gpu_sample_evidence(
            sample, contract=contract, sample_index=sample_index
        )
        for sample_index, sample in enumerate(samples)
    ]
    device_inventories = [_sample_device_inventory(sample) for sample in validated]
    process_inventories = [_sample_compute_processes(sample) for sample in validated]
    interval_ns = (
        validated[1]["sample_monotonic_ns"] - validated[0]["sample_monotonic_ns"]
    )
    if (
        device_inventories[0] != device_inventories[1]
        or process_inventories[0] != process_inventories[1]
        or value["device_inventory"] != device_inventories[0]
        or value["preexisting_compute_processes"] != process_inventories[0]
        or interval_ns < int(contract["sample_interval_seconds"] * 1_000_000_000)
        or any(
            gpu["memory_used_mib"] > contract["max_prelaunch_memory_used_mib"]
            or gpu["memory_headroom_mib"]
            < contract["min_prelaunch_memory_headroom_mib"]
            for sample in validated
            for gpu in sample["gpus"]
        )
    ):
        raise PreflightError(
            "shared GPU baseline is not the exact admitted inventory",
            code="preflight.marker_schema",
        )
    return dict(value)


def _build_postlaunch_gpu_sweep(
    *,
    plan: Mapping[str, Any],
    baseline: Mapping[str, Any],
    phase: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = plan["gpu_shared_preexisting_baseline_contract"]
    if len(samples) != contract["sample_count"]:
        raise PreflightError(
            "postlaunch GPU sweep sample count is not exact",
            code="controller.gpu_postlaunch_samples",
        )
    validated = [
        _validate_gpu_sample_evidence(
            sample, contract=contract, sample_index=sample_index
        )
        for sample_index, sample in enumerate(samples)
    ]
    observed_interval_ns = (
        validated[1]["sample_monotonic_ns"] - validated[0]["sample_monotonic_ns"]
    )
    required_interval_ns = int(contract["sample_interval_seconds"] * 1_000_000_000)
    if observed_interval_ns < required_interval_ns:
        raise PreflightError(
            "postlaunch GPU sweep samples are less than two seconds apart",
            code="controller.gpu_postlaunch_interval",
        )
    device_inventory_matches = all(
        _sample_device_inventory(sample) == baseline["device_inventory"]
        for sample in validated
    )
    observed_by_key = {
        (process["gpu_uuid"], process["pid"]): process
        for sample in validated
        for process in _sample_compute_processes(sample)
    }
    baseline_keys = {
        (process["gpu_uuid"], process["pid"])
        for process in baseline["preexisting_compute_processes"]
    }
    observed = [dict(observed_by_key[key]) for key in sorted(observed_by_key)]
    new_processes = [
        dict(observed_by_key[key])
        for key in sorted(set(observed_by_key).difference(baseline_keys))
    ]
    return {
        "phase": phase,
        "samples": validated,
        "device_inventory_matches_baseline": device_inventory_matches,
        "observed_compute_processes": observed,
        "new_compute_processes": new_processes,
        "admitted": device_inventory_matches and not new_processes,
    }


def _validate_postlaunch_gpu_sweep(
    value: Any,
    *,
    plan: Mapping[str, Any],
    baseline: Mapping[str, Any],
    require_admitted: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError(
            "postlaunch GPU sweep is malformed",
            code="controller.gpu_postlaunch_schema",
        )
    _exact_fields(
        value,
        {
            "phase",
            "samples",
            "device_inventory_matches_baseline",
            "observed_compute_processes",
            "new_compute_processes",
            "admitted",
        },
        owner="gpu_postlaunch_sweep",
    )
    if value.get("phase") not in {"post-launch-a", "post-launch-b", "preterminal"}:
        raise PreflightError(
            "postlaunch GPU sweep phase is unsupported",
            code="controller.gpu_postlaunch_schema",
        )
    rebuilt = _build_postlaunch_gpu_sweep(
        plan=plan,
        baseline=baseline,
        phase=value["phase"],
        samples=value.get("samples", []),
    )
    if rebuilt != value:
        raise PreflightError(
            "postlaunch GPU sweep derivation is inconsistent",
            code="controller.gpu_postlaunch_schema",
        )
    if require_admitted and value["device_inventory_matches_baseline"] is not True:
        raise PreflightError(
            "postlaunch GPU UUID/index inventory changed",
            code="controller.gpu_inventory",
        )
    if require_admitted and value["new_compute_processes"]:
        raise PreflightError(
            "a new GPU compute-process row appeared after launch",
            code="controller.gpu_new_compute_process",
        )
    if require_admitted and value["admitted"] is not True:
        raise PreflightError(
            "postlaunch GPU sweep is not admitted",
            code="controller.gpu_postlaunch_schema",
        )
    return dict(value)


def _collect_postlaunch_gpu_sweep(
    *, plan: Mapping[str, Any], baseline: Mapping[str, Any], phase: str
) -> dict[str, Any]:
    contract = plan["gpu_shared_preexisting_baseline_contract"]
    samples: list[dict[str, Any]] = []
    for index in range(contract["sample_count"]):
        if index:
            time.sleep(contract["sample_interval_seconds"])
        samples.append(_collect_gpu_sample(contract, index))
    return _build_postlaunch_gpu_sweep(
        plan=plan, baseline=baseline, phase=phase, samples=samples
    )


def _bounded_text(value: str | None) -> str:
    if value is None:
        return ""
    encoded = value.encode("utf-8", errors="replace")[-MAX_CAPTURE_BYTES:]
    return encoded.decode("utf-8", errors="replace")


def _scrub_controller_token(value: str | None, controller_token: str) -> str:
    bounded = _bounded_text(value)
    if not controller_token:
        return bounded
    return bounded.replace(controller_token, "<redacted-controller-token>")


def _proc_identity(pid: int) -> dict[str, int | str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    end = raw.rfind(")")
    if end < 0:
        return None
    fields = raw[end + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return {
            "pid": pid,
            "state": fields[0],
            "parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "session_id": int(fields[3]),
            "start_time_ticks": int(fields[19]),
        }
    except ValueError:
        return None


def _session_members(session_id: int) -> list[dict[str, int | str]]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = _proc_identity(int(entry.name))
        if identity is not None and identity["session_id"] == session_id:
            members.append(identity)
    return sorted(members, key=lambda value: int(value["pid"]))


def _identity_key(value: Mapping[str, Any]) -> tuple[int, int]:
    return int(value["pid"]), int(value["start_time_ticks"])


def _wait_session_empty(session_id: int, timeout: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        remaining = [
            value for value in _session_members(session_id) if value["state"] != "Z"
        ]
        if not remaining or time.monotonic() >= deadline:
            return [dict(value) for value in remaining]
        time.sleep(0.05)


def _signal_process_scope(
    members: Sequence[Mapping[str, Any]], signum: signal.Signals
) -> None:
    process_groups = sorted({int(value["process_group_id"]) for value in members})
    for process_group in process_groups:
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            pass
    for value in members:
        identity = _proc_identity(int(value["pid"]))
        if identity is None or _identity_key(identity) != _identity_key(value):
            continue
        try:
            os.kill(int(value["pid"]), signum)
        except ProcessLookupError:
            pass


def _cleanup_process_scope(
    process: subprocess.Popen[str],
    *,
    terminate_grace: float,
    kill_grace: float,
    initial_members: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, dict[str, Any]]:
    session_id = process.pid
    observed: dict[tuple[int, int], dict[str, Any]] = {}

    def capture() -> list[dict[str, Any]]:
        members = [dict(value) for value in _session_members(session_id)]
        for member in members:
            observed[_identity_key(member)] = member
        return members

    for member in initial_members:
        observed[_identity_key(member)] = dict(member)
    members = capture()
    term_sent = bool(members)
    kill_sent = False
    termination = "exited"
    remaining: list[dict[str, Any]] = []
    if members:
        _signal_process_scope(members, signal.SIGTERM)
        termination = "terminated"
        remaining = _wait_session_empty(session_id, terminate_grace)
        for member in remaining:
            observed[_identity_key(member)] = member
        if remaining:
            kill_sent = True
            _signal_process_scope(remaining, signal.SIGKILL)
            termination = "killed"
            remaining = _wait_session_empty(session_id, kill_grace)
    survivors_at_bound = [dict(value) for value in remaining]
    leader_reaped = True
    try:
        process.wait(timeout=kill_grace)
    except subprocess.TimeoutExpired:
        leader_reaped = False
    current_remaining = [value for value in capture() if value["state"] != "Z"]
    remaining_by_key = {
        _identity_key(value): dict(value)
        for value in [*survivors_at_bound, *current_remaining]
    }
    bounded_remaining = sorted(
        remaining_by_key.values(), key=lambda value: value["pid"]
    )
    if survivors_at_bound or current_remaining:
        cleanup_failure = "surviving_members_after_sigkill"
    elif not leader_reaped:
        cleanup_failure = "leader_reap_timeout"
    else:
        cleanup_failure = None
    cleanup_verified = cleanup_failure is None
    scope = {
        "session_id": session_id,
        "leader": {
            "pid": process.pid,
            "start_time_ticks": next(
                (
                    value["start_time_ticks"]
                    for value in observed.values()
                    if value["pid"] == process.pid
                ),
                None,
            ),
        },
        "observed_members": sorted(observed.values(), key=lambda value: value["pid"]),
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "remaining_members": bounded_remaining,
        "leader_reaped": leader_reaped,
        "cleanup_failure": cleanup_failure,
        "cleanup_verified": cleanup_verified,
    }
    return termination, scope


def _terminate_process_group(
    process: subprocess.Popen[str], *, terminate_grace: float, kill_grace: float
) -> str:
    termination, scope = _cleanup_process_scope(
        process, terminate_grace=terminate_grace, kill_grace=kill_grace
    )
    if scope["cleanup_verified"] is not True:
        raise PreflightError(
            "accelerate process scope cleanup is incomplete",
            code="controller.cleanup_failed",
        )
    return termination


def _run_accelerate_launch(
    *,
    plan: Mapping[str, Any],
    launch_id: str,
    controller_pid: int,
    controller_token: str,
) -> dict[str, Any]:
    argv = _command_for_launch(
        plan,
        launch_id=launch_id,
        controller_pid=controller_pid,
    )
    child_environment = os.environ.copy()
    child_environment.update(plan["environment"])
    child_environment[_CONTROLLER_TOKEN_ENV] = controller_token
    child_environment[_CONTROLLER_PID_ENV] = str(controller_pid)
    contract = plan["process_contract"]
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        try:
            process = subprocess.Popen(
                argv,
                env=child_environment,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except OSError as exc:
            raise PreflightError(
                f"accelerate launch could not start: {launch_id}",
                code="controller.launch_start",
            ) from exc
        observed: dict[tuple[int, int], dict[str, Any]] = {}
        primary_exception: BaseException | None = None
        timed_out = False
        try:
            leader_identity = _proc_identity(process.pid)
            if leader_identity is not None:
                observed[_identity_key(leader_identity)] = dict(leader_identity)
            deadline = time.monotonic() + contract["launch_timeout_seconds"]
            while process.poll() is None and time.monotonic() < deadline:
                for member in _session_members(process.pid):
                    observed[_identity_key(member)] = dict(member)
                time.sleep(0.05)
            for member in _session_members(process.pid):
                observed[_identity_key(member)] = dict(member)
            timed_out = process.poll() is None
        except BaseException as exc:
            primary_exception = exc
        termination, process_scope = _cleanup_process_scope(
            process,
            terminate_grace=contract["terminate_grace_seconds"],
            kill_grace=contract["kill_grace_seconds"],
            initial_members=list(observed.values()),
        )

        def tail(handle: Any) -> str:
            handle.flush()
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - MAX_CAPTURE_BYTES), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")

        stdout = tail(stdout_file)
        stderr = tail(stderr_file)
    process_record = {
        "launch_id": launch_id,
        "pid": process.pid,
        "returncode": process.returncode,
        "termination": termination,
        "process_scope": process_scope,
        "cleanup_verified": process_scope["cleanup_verified"],
        "stdout_tail": _scrub_controller_token(stdout, controller_token),
        "stderr_tail": _scrub_controller_token(stderr, controller_token),
    }
    if primary_exception is not None:
        raise PreflightError(
            f"accelerate launch was interrupted: {launch_id}",
            code=_failure_code(primary_exception),
            process_record=process_record,
        ) from primary_exception
    if process_record["cleanup_verified"] is not True:
        raise PreflightError(
            f"accelerate process scope cleanup failed: {launch_id}",
            code="controller.cleanup_failed",
            process_record=process_record,
        )
    if timed_out:
        raise PreflightError(
            f"accelerate launch timed out and was {termination}: {launch_id}",
            code="controller.launch_timeout",
            process_record=process_record,
        )
    if process.returncode != 0:
        raise PreflightError(
            f"accelerate launch failed: {launch_id}",
            code="controller.launch_failed",
            process_record=process_record,
        )
    return process_record


def _mapped_regular_file_origins() -> dict[str, list[Path]]:
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PreflightError(
            "process mappings are unreadable", code="worker.late_native"
        ) from exc
    origins: dict[str, set[Path]] = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            name = Path(raw_path.removesuffix(" (deleted)")).name
            if name.startswith("flash_attn_2_cuda") or name.startswith(
                ("libcublas.so", "libnccl.so")
            ):
                raise PreflightError(
                    f"late native mapping was deleted: {name}",
                    code="worker.late_native",
                )
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            continue
        try:
            resolved = path.resolve(strict=True)
            file_stat = resolved.stat()
        except (OSError, RuntimeError):
            continue
        if stat.S_ISREG(file_stat.st_mode):
            origins.setdefault(resolved.name, set()).add(resolved)
    return {name: sorted(paths, key=str) for name, paths in sorted(origins.items())}


def _attest_late_native_mappings(provenance: Mapping[str, Any]) -> dict[str, Any]:
    expectations = provenance.get("late_native_expectations")
    expected = _validate_late_native_expectations(expectations)
    mappings = _mapped_regular_file_origins()
    components: dict[str, Any] = {}
    mismatches: list[str] = []
    for component, identity in sorted(expected.items()):
        candidates = mappings.get(identity["soname"], [])
        if len(candidates) != 1:
            mismatches.append(f"components.{component}.mapped_origin")
            continue
        observed = _regular_file_identity(candidates[0])
        matches = observed == {
            "path": identity["path"],
            "size_bytes": identity["size_bytes"],
            "file_sha256": identity["file_sha256"],
        }
        components[component] = {
            **observed,
            "soname": candidates[0].name,
            "origin_matches_admission": matches,
        }
        if not matches:
            mismatches.append(f"components.{component}.identity")
    unique = sorted(set(mismatches))
    return {
        "schema_version": 1,
        "cuda_initialized": True,
        "admitted": not unique,
        "mismatches": unique,
        "components": components,
    }


def _validate_device_identity(value: Any, *, rank: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError("device identity is malformed", code="rank.device")
    _exact_fields(
        value,
        {"current_device", "visible_index", "gpu_uuid", "pci_bus_id"},
        owner="device_identity",
    )
    if (
        value.get("current_device") != rank
        or value.get("visible_index") != rank
        or not isinstance(value.get("gpu_uuid"), str)
        or not value["gpu_uuid"].startswith("GPU-")
        or not isinstance(value.get("pci_bus_id"), str)
        or not value["pci_bus_id"]
    ):
        raise PreflightError(
            "declared rank does not match the observed CUDA device",
            code="rank.device",
        )
    return dict(value)


def _validate_rank_receipt(
    receipt: Mapping[str, Any], *, plan: Mapping[str, Any], launch_id: str, rank: int
) -> dict[str, Any]:
    _exact_fields(
        receipt,
        {
            "schema",
            "status",
            "plan_sha256",
            "launch_id",
            "rank",
            "local_rank",
            "world_size",
            "device_index",
            "device_identity",
            "pre_cuda",
            "determinism_policy",
            "runtime_admission",
            "native_runtime_reference",
            "mapped_native_execution",
            "workloads",
            "cleanup",
            "claim_scope",
            "receipt_payload_sha256",
        },
        owner="rank_receipt",
    )
    _validate_finalized(
        receipt, digest_field="receipt_payload_sha256", owner="rank_receipt"
    )
    if (
        receipt.get("schema") != RANK_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("plan_sha256") != plan["plan_payload_sha256"]
        or receipt.get("launch_id") != launch_id
        or receipt.get("rank") != rank
        or receipt.get("local_rank") != rank
        or receipt.get("world_size") != WORLD_SIZE
        or receipt.get("device_index") != rank
        or receipt.get("device_identity") is None
        or receipt.get("runtime_admission") != plan["runtime_admission"]
        or receipt.get("native_runtime_reference") != plan["native_runtime_reference"]
        or receipt.get("claim_scope") != plan["claim_scope"]
    ):
        raise PreflightError("rank receipt identity is rejected", code="rank.identity")
    _validate_device_identity(receipt.get("device_identity"), rank=rank)
    pre_cuda = receipt.get("pre_cuda")
    if (
        not isinstance(pre_cuda, dict)
        or pre_cuda.get("cuda_initialized") is not False
        or pre_cuda.get("environment") != plan["environment"]
        or pre_cuda.get("rank_environment")
        != {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "8"}
    ):
        raise PreflightError("rank pre-CUDA evidence is rejected", code="rank.pre_cuda")
    if receipt.get("determinism_policy") != _expected_policy_receipt():
        raise PreflightError("rank determinism policy is rejected", code="rank.policy")
    native = receipt.get("mapped_native_execution")
    if (
        not isinstance(native, dict)
        or set(native)
        != {
            "schema_version",
            "cuda_initialized",
            "admitted",
            "mismatches",
            "pre_workload_attestation_sha256",
            "post_workload_attestation_sha256",
            "late_native_mappings",
        }
        or native.get("schema_version") != 2
        or native.get("cuda_initialized") is not True
        or native.get("admitted") is not True
        or native.get("mismatches") != []
        or not _is_sha256(native.get("pre_workload_attestation_sha256"))
        or not _is_sha256(native.get("post_workload_attestation_sha256"))
    ):
        raise PreflightError(
            "rank live native attestation is rejected", code="rank.native"
        )
    late = native.get("late_native_mappings")
    if (
        not isinstance(late, dict)
        or set(late)
        != {
            "schema_version",
            "cuda_initialized",
            "admitted",
            "mismatches",
            "components",
        }
        or late.get("schema_version") != 1
        or late.get("cuda_initialized") is not True
        or late.get("admitted") is not True
        or late.get("mismatches") != []
        or set(late.get("components", {}))
        != {"flash_attention_2", "libcublas", "libnccl"}
    ):
        raise PreflightError(
            "rank late native attestation is rejected", code="rank.native"
        )
    runtime_receipt = _load_json(Path(plan["runtime_admission"]["path"]))
    expectations = runtime_receipt["late_native_expectations"]
    for component, expected in expectations.items():
        observed = late["components"].get(component)
        if not isinstance(observed, dict):
            raise PreflightError(
                "rank late native component is missing", code="rank.native"
            )
        _exact_fields(
            observed,
            {
                "path",
                "size_bytes",
                "file_sha256",
                "soname",
                "origin_matches_admission",
            },
            owner="rank_late_native_component",
        )
        if observed != {
            "path": expected["path"],
            "size_bytes": expected["size_bytes"],
            "file_sha256": expected["file_sha256"],
            "soname": expected["soname"],
            "origin_matches_admission": True,
        }:
            raise PreflightError(
                "rank late native component drifted", code="rank.native"
            )
    workloads = receipt.get("workloads")
    required_workloads = {
        "cublas",
        "flash_attention_2",
        "nccl_all_reduce",
        "aggregate_sha256",
    }
    if (
        not isinstance(workloads, dict)
        or set(workloads) != required_workloads
        or not _is_sha256(workloads.get("aggregate_sha256"))
        or any(
            not isinstance(workloads.get(name), dict)
            or not _is_sha256(workloads[name].get("digest"))
            for name in ("cublas", "flash_attention_2", "nccl_all_reduce")
        )
        or workloads["nccl_all_reduce"].get("expected_sum") != 28.0
        or workloads.get("aggregate_sha256")
        != _sha256_bytes(
            _canonical_json_bytes(
                {
                    name: workloads[name]
                    for name in (
                        "cublas",
                        "flash_attention_2",
                        "nccl_all_reduce",
                    )
                }
            )
        )
        or receipt.get("cleanup") != {"process_group_destroyed": True}
    ):
        raise PreflightError(
            "rank workload or cleanup evidence is rejected", code="rank.workload"
        )
    return dict(receipt)


def _load_rank_receipts(
    plan: Mapping[str, Any], launch_id: str
) -> list[dict[str, Any]]:
    root = Path(plan["artifact_targets"]["rank_receipt_root"])
    launch_root = root / launch_id
    expected = {f"rank-{rank:05d}.json" for rank in range(WORLD_SIZE)}
    observed = (
        {path.name for path in launch_root.iterdir()} if launch_root.is_dir() else set()
    )
    if observed != expected:
        raise PreflightError(
            f"rank receipt inventory is not exact for {launch_id}",
            code="rank.inventory",
        )
    if any(
        not (launch_root / name).is_file() or (launch_root / name).is_symlink()
        for name in expected
    ):
        raise PreflightError(
            f"rank receipt entries are not regular files for {launch_id}",
            code="rank.inventory",
        )
    receipts = []
    for rank in range(WORLD_SIZE):
        receipt = _load_json(launch_root / f"rank-{rank:05d}.json")
        receipts.append(
            _validate_rank_receipt(receipt, plan=plan, launch_id=launch_id, rank=rank)
        )
    digests = {receipt["workloads"]["aggregate_sha256"] for receipt in receipts}
    if len(digests) != 1:
        raise PreflightError(
            f"rank workload digests differ within {launch_id}",
            code="rank.digest",
        )
    uuid_inventory = [receipt["device_identity"]["gpu_uuid"] for receipt in receipts]
    pci_inventory = [receipt["device_identity"]["pci_bus_id"] for receipt in receipts]
    if len(set(uuid_inventory)) != WORLD_SIZE or len(set(pci_inventory)) != WORLD_SIZE:
        raise PreflightError(
            f"GPU identity inventory is not one-to-one for {launch_id}",
            code="rank.device",
        )
    return receipts


_load_rank_receipts_original = _load_rank_receipts


def _partial_rank_file_inventory(
    plan: Mapping[str, Any],
) -> tuple[tuple[str, int, int, str], ...]:
    root = Path(plan["artifact_targets"]["rank_receipt_root"])
    if not root.exists() and not root.is_symlink():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise PreflightError(
            "partial rank receipt root is not an exact directory",
            code="rank.inventory",
        )
    root_entries = {path.name: path for path in root.iterdir()}
    if not set(root_entries).issubset(LAUNCH_IDS):
        raise PreflightError(
            "partial rank receipt launch inventory is not exact",
            code="rank.inventory",
        )
    inventory: list[tuple[str, int, int, str]] = []
    expected_names = {f"rank-{rank:05d}.json" for rank in range(WORLD_SIZE)}
    for launch_id in LAUNCH_IDS:
        launch_root = root_entries.get(launch_id)
        if launch_root is None:
            continue
        if launch_root.is_symlink() or not launch_root.is_dir():
            raise PreflightError(
                f"partial rank receipt launch root is invalid: {launch_id}",
                code="rank.inventory",
            )
        entries = {path.name: path for path in launch_root.iterdir()}
        if not set(entries).issubset(expected_names):
            raise PreflightError(
                f"partial rank receipt inventory is not exact for {launch_id}",
                code="rank.inventory",
            )
        for name in sorted(entries):
            path = entries[name]
            if path.is_symlink() or not path.is_file():
                raise PreflightError(
                    f"partial rank receipt entry is invalid: {path}",
                    code="rank.inventory",
                )
            file_stat = path.stat()
            inventory.append(
                (
                    str(path),
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                    _sha256_file(path),
                )
            )
    return tuple(inventory)


def _load_quiesced_partial_rank_receipts(
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 1.0
    previous = _partial_rank_file_inventory(plan)
    while True:
        time.sleep(0.05)
        observed = _partial_rank_file_inventory(plan)
        if observed == previous:
            break
        previous = observed
        if time.monotonic() >= deadline:
            raise PreflightError(
                "partial rank receipt inventory did not quiesce",
                code="rank.inventory",
            )
    receipts: list[dict[str, Any]] = []
    for path_text, _size, _mtime_ns, _file_sha256 in observed:
        path = Path(path_text)
        launch_id = path.parent.name
        rank = int(path.stem.removeprefix("rank-"))
        receipt = _load_json(path)
        receipts.append(
            _validate_rank_receipt(
                receipt,
                plan=plan,
                launch_id=launch_id,
                rank=rank,
            )
        )
    if _partial_rank_file_inventory(plan) != observed:
        raise PreflightError(
            "partial rank receipt inventory changed while binding",
            code="rank.inventory",
        )
    return receipts


def _compare_launch_receipts(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mismatch_ranks = [
        rank
        for rank in range(WORLD_SIZE)
        if left[rank]["workloads"] != right[rank]["workloads"]
        or left[rank]["mapped_native_execution"]
        != right[rank]["mapped_native_execution"]
        or left[rank]["device_identity"] != right[rank]["device_identity"]
    ]
    if mismatch_ranks:
        raise PreflightError(
            f"launch A/B evidence differs at ranks {mismatch_ranks}",
            code="comparison.launch_digest",
        )
    return {
        "launch_digest_equal": True,
        "device_mapping_equal": True,
        "native_mapping_equal": True,
        "rank_count_per_launch": WORLD_SIZE,
        "mismatch_ranks": [],
        "launch_a_aggregate_sha256": left[0]["workloads"]["aggregate_sha256"],
        "launch_b_aggregate_sha256": right[0]["workloads"]["aggregate_sha256"],
    }


def _receipt_binding(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "payload_sha256": receipt["receipt_payload_sha256"],
        "schema": receipt["schema"],
        "status": receipt["status"],
    }


def _validate_marker(marker: Mapping[str, Any], *, plan: Mapping[str, Any]) -> None:
    _exact_fields(
        marker,
        {
            "schema",
            "status",
            "plan_sha256",
            "controller",
            "environment",
            "gpu_shared_preexisting_baseline",
            "commands",
            "artifact_targets",
            "receipt_payload_sha256",
        },
        owner="attempt_marker",
    )
    _validate_finalized(
        marker, digest_field="receipt_payload_sha256", owner="attempt_marker"
    )
    controller = marker.get("controller")
    if not isinstance(controller, dict):
        raise PreflightError(
            "attempt marker controller is malformed", code="preflight.marker_schema"
        )
    _exact_fields(controller, {"pid", "token_sha256"}, owner="marker_controller")
    baseline = marker.get("gpu_shared_preexisting_baseline")
    commands = marker.get("commands")
    if (
        marker.get("schema") != ATTEMPT_MARKER_SCHEMA
        or marker.get("status") != "started"
        or marker.get("plan_sha256") != plan["plan_payload_sha256"]
        or marker.get("environment") != plan["environment"]
        or marker.get("artifact_targets") != plan["artifact_targets"]
        or not isinstance(controller["pid"], int)
        or isinstance(controller["pid"], bool)
        or controller["pid"] <= 0
        or not _is_sha256(controller["token_sha256"])
        or not isinstance(baseline, dict)
        or not isinstance(commands, list)
        or len(commands) != 2
    ):
        raise PreflightError(
            "attempt marker is rejected", code="preflight.marker_schema"
        )
    _validate_gpu_baseline(baseline, plan=plan)
    for command, launch_id in zip(commands, LAUNCH_IDS, strict=True):
        if not isinstance(command, dict):
            raise PreflightError(
                "attempt command is malformed", code="preflight.marker_schema"
            )
        _exact_fields(
            command, {"launch_id", "argv", "environment"}, owner="marker_command"
        )
        if (
            command["launch_id"] != launch_id
            or command["environment"] != plan["environment"]
            or command["argv"]
            != _redacted_command_for_launch(
                plan, launch_id=launch_id, controller_pid=controller["pid"]
            )
        ):
            raise PreflightError(
                "attempt command is rejected", code="preflight.marker_schema"
            )


def _validate_process_record(value: Mapping[str, Any]) -> None:
    _exact_fields(
        value,
        {
            "launch_id",
            "pid",
            "returncode",
            "termination",
            "process_scope",
            "cleanup_verified",
            "stdout_tail",
            "stderr_tail",
        },
        owner="terminal_process",
    )
    if (
        value["launch_id"] not in LAUNCH_IDS
        or not isinstance(value["pid"], int)
        or isinstance(value["pid"], bool)
        or value["pid"] <= 0
        or (
            value["returncode"] is not None and not isinstance(value["returncode"], int)
        )
        or value["termination"] not in {"exited", "terminated", "killed"}
        or not isinstance(value["cleanup_verified"], bool)
        or not isinstance(value["stdout_tail"], str)
        or not isinstance(value["stderr_tail"], str)
    ):
        raise PreflightError(
            "terminal process record is rejected",
            code="preflight.terminal_process",
        )
    scope = value.get("process_scope")
    if not isinstance(scope, dict):
        raise PreflightError(
            "terminal process scope is malformed",
            code="preflight.terminal_process",
        )
    _exact_fields(
        scope,
        {
            "session_id",
            "leader",
            "observed_members",
            "term_sent",
            "kill_sent",
            "remaining_members",
            "leader_reaped",
            "cleanup_failure",
            "cleanup_verified",
        },
        owner="terminal_process_scope",
    )
    leader = scope.get("leader")
    if (
        scope.get("session_id") != value["pid"]
        or not isinstance(leader, dict)
        or set(leader) != {"pid", "start_time_ticks"}
        or leader.get("pid") != value["pid"]
        or not isinstance(leader.get("start_time_ticks"), int)
        or isinstance(leader.get("start_time_ticks"), bool)
        or not isinstance(scope.get("observed_members"), list)
        or not scope["observed_members"]
        or not isinstance(scope.get("remaining_members"), list)
        or not isinstance(scope.get("term_sent"), bool)
        or not isinstance(scope.get("kill_sent"), bool)
        or not isinstance(scope.get("leader_reaped"), bool)
        or scope.get("cleanup_failure")
        not in {
            None,
            "leader_reap_timeout",
            "surviving_members_after_sigkill",
        }
        or not isinstance(scope.get("cleanup_verified"), bool)
        or value["cleanup_verified"] is not scope["cleanup_verified"]
    ):
        raise PreflightError(
            "terminal process scope is rejected",
            code="preflight.terminal_process",
        )
    member_keys = []
    for member in scope["observed_members"]:
        if not isinstance(member, dict):
            raise PreflightError(
                "terminal process member is malformed",
                code="preflight.terminal_process",
            )
        _exact_fields(
            member,
            {
                "pid",
                "state",
                "parent_pid",
                "process_group_id",
                "session_id",
                "start_time_ticks",
            },
            owner="terminal_process_member",
        )
        if (
            member["session_id"] != value["pid"]
            or not isinstance(member["pid"], int)
            or isinstance(member["pid"], bool)
            or not isinstance(member["start_time_ticks"], int)
            or isinstance(member["start_time_ticks"], bool)
            or not isinstance(member["state"], str)
            or len(member["state"]) != 1
        ):
            raise PreflightError(
                "terminal process member is rejected",
                code="preflight.terminal_process",
            )
        member_keys.append(_identity_key(member))
    if (
        len(set(member_keys)) != len(member_keys)
        or (value["pid"], leader["start_time_ticks"]) not in member_keys
    ):
        raise PreflightError(
            "terminal process identity inventory is incomplete",
            code="preflight.terminal_process",
        )
    remaining_keys = []
    for member in scope["remaining_members"]:
        if not isinstance(member, dict):
            raise PreflightError(
                "terminal remaining process member is malformed",
                code="preflight.terminal_process",
            )
        _exact_fields(
            member,
            {
                "pid",
                "state",
                "parent_pid",
                "process_group_id",
                "session_id",
                "start_time_ticks",
            },
            owner="terminal_remaining_process_member",
        )
        key = _identity_key(member)
        if member["session_id"] != value["pid"] or key not in member_keys:
            raise PreflightError(
                "terminal remaining process identity is rejected",
                code="preflight.terminal_process",
            )
        remaining_keys.append(key)
    if len(set(remaining_keys)) != len(remaining_keys):
        raise PreflightError(
            "terminal remaining process identities are duplicated",
            code="preflight.terminal_process",
        )
    cleanup_verified = scope["cleanup_verified"]
    cleanup_failure = scope["cleanup_failure"]
    if cleanup_verified:
        if (
            scope["leader_reaped"] is not True
            or cleanup_failure is not None
            or scope["remaining_members"] != []
        ):
            raise PreflightError(
                "successful cleanup scope is inconsistent",
                code="preflight.terminal_process",
            )
    elif cleanup_failure == "leader_reap_timeout":
        if scope["leader_reaped"] is not False:
            raise PreflightError(
                "leader reap failure scope is inconsistent",
                code="preflight.terminal_process",
            )
    elif (
        cleanup_failure != "surviving_members_after_sigkill"
        or not scope["remaining_members"]
    ):
        raise PreflightError(
            "failed cleanup scope lacks bounded survivor evidence",
            code="preflight.terminal_process",
        )


def _validate_terminal_receipt(
    receipt: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    _exact_fields(
        receipt,
        {
            "schema",
            "status",
            "world_size",
            "launch_count",
            "mismatches",
            "plan_sha256",
            "attempt_marker",
            "processes",
            "rank_receipts",
            "comparisons",
            "gpu_shared_occupancy_sweeps",
            "cleanup",
            "claim_scope",
            "receipt_payload_sha256",
        },
        owner="terminal_receipt",
    )
    _validate_finalized(
        receipt, digest_field="receipt_payload_sha256", owner="terminal_receipt"
    )
    processes = receipt.get("processes")
    mismatches = receipt.get("mismatches")
    rank_receipts = receipt.get("rank_receipts")
    cleanup = receipt.get("cleanup")
    gpu_sweeps = receipt.get("gpu_shared_occupancy_sweeps")
    if (
        receipt.get("schema") != TERMINAL_RECEIPT_SCHEMA
        or receipt.get("status") not in {"passed", "failed"}
        or receipt.get("world_size") != WORLD_SIZE
        or receipt.get("plan_sha256") != plan["plan_payload_sha256"]
        or receipt.get("claim_scope") != plan["claim_scope"]
        or not isinstance(processes, list)
        or receipt.get("launch_count") != len(processes)
        or not isinstance(mismatches, list)
        or any(not isinstance(item, str) or not item for item in mismatches)
        or not isinstance(rank_receipts, list)
        or not isinstance(cleanup, dict)
        or not isinstance(gpu_sweeps, list)
        or set(cleanup) != {"all_process_groups_exited", "bounded"}
        or cleanup["bounded"]
        is not (receipt.get("attempt_marker") is None or bool(processes))
    ):
        raise PreflightError(
            "terminal receipt is rejected", code="preflight.terminal_schema"
        )
    for process in processes:
        if not isinstance(process, dict):
            raise PreflightError(
                "terminal process record is malformed",
                code="preflight.terminal_process",
            )
        _validate_process_record(process)
    marker_binding = receipt.get("attempt_marker")
    marker_baseline = None
    if marker_binding is not None:
        marker_path = Path(plan["artifact_targets"]["attempt_marker"])
        marker_payload = _load_json(marker_path)
        _validate_marker(marker_payload, plan=plan)
        if marker_binding != _receipt_binding(marker_path, marker_payload):
            raise PreflightError(
                "terminal marker binding is not live",
                code="preflight.terminal_schema",
            )
        marker_baseline = marker_payload["gpu_shared_preexisting_baseline"]
    expected_sweep_phases = ["post-launch-a", "post-launch-b", "preterminal"]
    observed_sweep_phases = []
    for sweep in gpu_sweeps:
        if marker_baseline is None:
            raise PreflightError(
                "GPU sweep cannot exist without its marker baseline",
                code="preflight.terminal_schema",
            )
        _validate_postlaunch_gpu_sweep(
            sweep,
            plan=plan,
            baseline=marker_baseline,
            require_admitted=receipt["status"] == "passed",
        )
        observed_sweep_phases.append(sweep["phase"])
    if observed_sweep_phases != expected_sweep_phases[: len(gpu_sweeps)]:
        raise PreflightError(
            "terminal GPU sweep sequence is not an exact prefix",
            code="preflight.terminal_schema",
        )
    if [process["launch_id"] for process in processes] != list(LAUNCH_IDS)[
        : len(processes)
    ]:
        raise PreflightError(
            "terminal launch sequence is not an exact prefix",
            code="preflight.terminal_process",
        )
    for binding in rank_receipts:
        if not isinstance(binding, dict):
            raise PreflightError(
                "terminal rank binding is malformed",
                code="preflight.terminal_schema",
            )
        _exact_fields(
            binding,
            {"path", "file_sha256", "payload_sha256", "schema", "status"},
            owner="terminal_rank_binding",
        )
        if (
            not _is_sha256(binding["file_sha256"])
            or not _is_sha256(binding["payload_sha256"])
            or binding["schema"] != RANK_RECEIPT_SCHEMA
            or binding["status"] != "passed"
        ):
            raise PreflightError(
                "terminal rank binding is rejected",
                code="preflight.terminal_schema",
            )
        _absolute_path(binding["path"], owner="rank binding")
    if len({binding["path"] for binding in rank_receipts}) != len(rank_receipts):
        raise PreflightError(
            "terminal rank bindings are duplicated",
            code="preflight.terminal_schema",
        )
    expected_all_exited = (
        None
        if not processes
        else all(process["cleanup_verified"] is True for process in processes)
    )
    if cleanup["all_process_groups_exited"] is not expected_all_exited:
        raise PreflightError(
            "terminal cleanup summary is inconsistent",
            code="preflight.terminal_process",
        )
    if receipt["status"] == "passed":
        comparisons = receipt.get("comparisons")
        if (
            mismatches != []
            or receipt["launch_count"] != 2
            or cleanup["all_process_groups_exited"] is not True
            or len(rank_receipts) != 16
            or not isinstance(comparisons, dict)
            or set(comparisons)
            != {
                "launch_digest_equal",
                "device_mapping_equal",
                "native_mapping_equal",
                "rank_count_per_launch",
                "mismatch_ranks",
                "launch_a_aggregate_sha256",
                "launch_b_aggregate_sha256",
            }
            or comparisons.get("launch_digest_equal") is not True
            or comparisons.get("device_mapping_equal") is not True
            or comparisons.get("native_mapping_equal") is not True
            or comparisons.get("rank_count_per_launch") != WORLD_SIZE
            or comparisons.get("mismatch_ranks") != []
            or comparisons.get("launch_a_aggregate_sha256")
            != comparisons.get("launch_b_aggregate_sha256")
            or not _is_sha256(comparisons.get("launch_a_aggregate_sha256"))
            or not isinstance(marker_binding, dict)
            or observed_sweep_phases != expected_sweep_phases
        ):
            raise PreflightError(
                "passed terminal evidence is incomplete",
                code="preflight.terminal_schema",
            )
        _exact_fields(
            marker_binding,
            {"path", "file_sha256", "payload_sha256", "schema", "status"},
            owner="terminal_marker_binding",
        )
        if (
            marker_binding["schema"] != ATTEMPT_MARKER_SCHEMA
            or marker_binding["status"] != "started"
            or not _is_sha256(marker_binding["file_sha256"])
            or not _is_sha256(marker_binding["payload_sha256"])
        ):
            raise PreflightError(
                "terminal marker binding is rejected",
                code="preflight.terminal_schema",
            )
    elif not mismatches:
        raise PreflightError(
            "failed terminal must name a mismatch",
            code="preflight.terminal_schema",
        )
    return dict(receipt)


def _terminal_payload(
    *,
    plan: Mapping[str, Any],
    status: str,
    mismatches: list[str],
    marker: Mapping[str, Any] | None,
    processes: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any] | None,
    gpu_shared_occupancy_sweeps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(plan["artifact_targets"]["rank_receipt_root"])
    rank_bindings = []
    for receipt in receipts:
        path = root / receipt["launch_id"] / f"rank-{receipt['rank']:05d}.json"
        rank_bindings.append(_receipt_binding(path, receipt))
    marker_binding = None
    if marker is not None:
        marker_binding = _receipt_binding(
            Path(plan["artifact_targets"]["attempt_marker"]), marker
        )
    return {
        "schema": TERMINAL_RECEIPT_SCHEMA,
        "status": status,
        "world_size": WORLD_SIZE,
        "launch_count": len(processes),
        "mismatches": list(mismatches),
        "plan_sha256": plan["plan_payload_sha256"],
        "attempt_marker": marker_binding,
        "processes": [dict(value) for value in processes],
        "rank_receipts": rank_bindings,
        "comparisons": None if comparisons is None else dict(comparisons),
        "gpu_shared_occupancy_sweeps": [
            dict(value) for value in gpu_shared_occupancy_sweeps
        ],
        "cleanup": {
            "all_process_groups_exited": (
                None
                if not processes
                else all(item.get("cleanup_verified") is True for item in processes)
            ),
            "bounded": marker is None or bool(processes),
        },
        "claim_scope": dict(plan["claim_scope"]),
    }


def _failure_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    return str(code) if isinstance(code, str) and code else type(exc).__name__


def run_plan(
    plan_path: str | Path, *, expected_plan_file_sha256: str
) -> dict[str, Any]:
    """Consume the prepared plan once and publish exactly one terminal receipt."""

    resolved_plan, encoded_plan = _read_bound_plan_file(
        plan_path, expected_file_sha256=expected_plan_file_sha256
    )
    raw_plan = _load_json_bytes(encoded_plan, path=resolved_plan)
    plan = _validate_plan(raw_plan, plan_path=resolved_plan)
    targets = plan["artifact_targets"]
    marker_path = Path(targets["attempt_marker"])
    terminal_path = Path(targets["terminal_receipt"])
    rank_root = Path(targets["rank_receipt_root"])
    _assert_absent(terminal_path)
    marker: dict[str, Any] | None = None
    processes: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    comparisons: dict[str, Any] | None = None
    gpu_shared_occupancy_sweeps: list[dict[str, Any]] = []
    try:
        _assert_absent(marker_path)
        _assert_absent(rank_root, directory=True)
        _check_outer_environment()
        _check_launch_ports_available(plan)
        gpu_baseline = _collect_stable_gpu_idle_samples(plan)
        controller_pid = os.getpid()
        controller_token = secrets.token_hex(32)
        commands = [
            {
                "launch_id": launch_id,
                "argv": _redacted_command_for_launch(
                    plan,
                    launch_id=launch_id,
                    controller_pid=controller_pid,
                ),
                "environment": dict(plan["environment"]),
            }
            for launch_id in LAUNCH_IDS
        ]
        marker_candidate = finalize_receipt(
            {
                "schema": ATTEMPT_MARKER_SCHEMA,
                "status": "started",
                "plan_sha256": plan["plan_payload_sha256"],
                "controller": {
                    "pid": controller_pid,
                    "token_sha256": _sha256_bytes(controller_token.encode()),
                },
                "environment": dict(plan["environment"]),
                "gpu_shared_preexisting_baseline": gpu_baseline,
                "commands": commands,
                "artifact_targets": dict(targets),
            }
        )
        _validate_marker(marker_candidate, plan=plan)
        _read_bound_plan_file(
            resolved_plan, expected_file_sha256=expected_plan_file_sha256
        )
        _publish_absent(marker_path, marker_candidate)
        marker = marker_candidate

        launches: dict[str, list[dict[str, Any]]] = {}
        for launch_id in LAUNCH_IDS:
            _validate_accelerate_identity(plan["accelerate_identity"])
            try:
                process = _run_accelerate_launch(
                    plan=plan,
                    launch_id=launch_id,
                    controller_pid=controller_pid,
                    controller_token=controller_token,
                )
            except PreflightError as exc:
                if exc.process_record is not None:
                    processes.append(exc.process_record)
                raise
            processes.append(process)
            sweep = _collect_postlaunch_gpu_sweep(
                plan=plan,
                baseline=gpu_baseline,
                phase=f"post-{launch_id}",
            )
            gpu_shared_occupancy_sweeps.append(sweep)
            _validate_postlaunch_gpu_sweep(sweep, plan=plan, baseline=gpu_baseline)
            launch_receipts = _load_rank_receipts(plan, launch_id)
            launches[launch_id] = launch_receipts
            receipts.extend(launch_receipts)

        sweep = _collect_postlaunch_gpu_sweep(
            plan=plan, baseline=gpu_baseline, phase="preterminal"
        )
        gpu_shared_occupancy_sweeps.append(sweep)
        _validate_postlaunch_gpu_sweep(sweep, plan=plan, baseline=gpu_baseline)
        marker = _load_json(marker_path)
        _validate_marker(marker, plan=plan)
        launches = {
            launch_id: _load_rank_receipts(plan, launch_id) for launch_id in LAUNCH_IDS
        }
        receipts = [
            receipt for launch_id in LAUNCH_IDS for receipt in launches[launch_id]
        ]
        comparisons = _compare_launch_receipts(
            launches["launch-a"], launches["launch-b"]
        )
        terminal = finalize_receipt(
            _terminal_payload(
                plan=plan,
                status="passed",
                mismatches=[],
                marker=marker,
                processes=processes,
                receipts=receipts,
                comparisons=comparisons,
                gpu_shared_occupancy_sweeps=gpu_shared_occupancy_sweeps,
            )
        )
        _validate_terminal_receipt(terminal, plan=plan)
        _publish_absent(terminal_path, terminal)
        return terminal
    except BaseException as exc:
        failure_mismatches = [_failure_code(exc)]
        if marker is not None:
            try:
                receipts = _load_quiesced_partial_rank_receipts(plan)
            except BaseException:
                receipts = []
                failure_mismatches.append("rank.partial_evidence_binding")
        failure = finalize_receipt(
            _terminal_payload(
                plan=plan,
                status="failed",
                mismatches=failure_mismatches,
                marker=marker,
                processes=processes,
                receipts=receipts,
                comparisons=comparisons,
                gpu_shared_occupancy_sweeps=gpu_shared_occupancy_sweeps,
            )
        )
        _validate_terminal_receipt(failure, plan=plan)
        try:
            _publish_absent(terminal_path, failure)
        except PreflightError:
            if not terminal_path.exists():
                raise
        if isinstance(exc, PreflightError):
            raise
        raise PreflightError(
            "preflight controller failed", code=_failure_code(exc)
        ) from exc


class WorkerHooks(Protocol):
    def cuda_is_initialized(self) -> bool: ...

    def apply_seed_policy(self, seed: int) -> Mapping[str, Any]: ...

    def initialize_cuda(self, local_rank: int) -> None: ...

    def observe_device_identity(self, local_rank: int) -> Mapping[str, Any]: ...

    def attest_native(self, provenance: dict[str, Any]) -> Mapping[str, Any]: ...

    def attest_late_native(self, provenance: dict[str, Any]) -> Mapping[str, Any]: ...

    def run_workloads(self, rank: int, world_size: int) -> Mapping[str, Any]: ...

    def cleanup(self) -> bool: ...


def _tensor_digest(tensor: Any) -> str:
    import torch

    value = tensor.detach().contiguous().cpu()
    byte_view = value.view(torch.uint8)
    return _sha256_bytes(byte_view.numpy().tobytes())


class _CudaWorkerHooks:
    def cuda_is_initialized(self) -> bool:
        import torch

        return bool(torch.cuda.is_initialized())

    def apply_seed_policy(self, seed: int) -> Mapping[str, Any]:
        from src.runtime.seeding import seed_training_runtime

        return seed_training_runtime(
            seed,
            determinism_mode="strict_cuda_replay_v1",
            phase="pipeline_entry",
        ).to_artifact_dict()

    def initialize_cuda(self, local_rank: int) -> None:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        torch.empty(1, device=f"cuda:{local_rank}")

    def observe_device_identity(self, local_rank: int) -> Mapping[str, Any]:
        import torch

        current_device = int(torch.cuda.current_device())
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PreflightError(
                "worker GPU identity query failed", code="worker.device"
            ) from exc
        rows = _parse_csv_rows(result.stdout)
        matches = [row for row in rows if len(row) == 3 and row[0] == str(local_rank)]
        if len(matches) != 1:
            raise PreflightError(
                "worker GPU identity mapping is ambiguous", code="worker.device"
            )
        return {
            "current_device": current_device,
            "visible_index": local_rank,
            "gpu_uuid": matches[0][1],
            "pci_bus_id": matches[0][2],
        }

    def attest_native(self, provenance: dict[str, Any]) -> Mapping[str, Any]:
        from src.artifacts.provenance import (
            require_mapped_native_execution_attestation,
        )

        native_provenance = provenance.get("provenance", provenance)
        if not isinstance(native_provenance, dict):
            raise PreflightError(
                "native provenance payload is malformed", code="worker.native"
            )
        return require_mapped_native_execution_attestation(provenance=native_provenance)

    def attest_late_native(self, provenance: dict[str, Any]) -> Mapping[str, Any]:
        return _attest_late_native_mappings(provenance)

    def run_workloads(self, rank: int, world_size: int) -> Mapping[str, Any]:
        import torch
        import torch.distributed as dist

        device = torch.device("cuda", rank)
        base = torch.linspace(
            -1.0, 1.0, 32 * 32, device=device, dtype=torch.float32
        ).reshape(32, 32)
        left = base.to(torch.bfloat16).detach().requires_grad_(True)
        right = base.flip(0).to(torch.bfloat16).detach().requires_grad_(True)
        output = left @ right
        output.float().sum().backward()
        cublas_digest = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "output": _tensor_digest(output),
                    "left_grad": _tensor_digest(left.grad),
                    "right_grad": _tensor_digest(right.grad),
                }
            )
        )

        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func

        values = torch.linspace(
            -0.5,
            0.5,
            8 * 2 * 64,
            device=device,
            dtype=torch.float32,
        ).reshape(8, 2, 64)
        query = values.to(torch.bfloat16).detach().requires_grad_(True)
        key = values.flip(0).to(torch.bfloat16).detach().requires_grad_(True)
        value = values.roll(1, 0).to(torch.bfloat16).detach().requires_grad_(True)
        cu_seqlens = torch.tensor([0, 3, 8], device=device, dtype=torch.int32)
        attention = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            5,
            5,
            dropout_p=0.0,
            causal=True,
            deterministic=True,
        )
        attention.float().sum().backward()
        flash_digest = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "output": _tensor_digest(attention),
                    "query_grad": _tensor_digest(query.grad),
                    "key_grad": _tensor_digest(key.grad),
                    "value_grad": _tensor_digest(value.grad),
                }
            )
        )

        collective = torch.tensor([float(rank)], device=device, dtype=torch.float32)
        dist.all_reduce(collective, op=dist.ReduceOp.SUM)
        expected_sum = float(sum(range(world_size)))
        if collective.item() != expected_sum:
            raise PreflightError(
                "NCCL all-reduce result is not exact", code="worker.nccl"
            )
        nccl_digest = _tensor_digest(collective)
        workloads = {
            "cublas": {"digest": cublas_digest},
            "flash_attention_2": {"digest": flash_digest},
            "nccl_all_reduce": {
                "digest": nccl_digest,
                "expected_sum": expected_sum,
            },
        }
        return {
            **workloads,
            "aggregate_sha256": _sha256_bytes(_canonical_json_bytes(workloads)),
        }

    def cleanup(self) -> bool:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        return not dist.is_initialized()


def _validate_worker_guard(
    *,
    plan: Mapping[str, Any],
    launch_id: str,
    controller_pid: int,
    controller_token: str,
) -> None:
    if launch_id not in LAUNCH_IDS:
        raise PreflightError("worker launch id is invalid", code="worker.launch_id")
    if (
        os.environ.get(_CONTROLLER_PID_ENV) != str(controller_pid)
        or os.environ.get(_CONTROLLER_TOKEN_ENV) != controller_token
    ):
        raise PreflightError(
            "worker controller environment is invalid", code="worker.guard"
        )
    marker = _load_json(Path(plan["artifact_targets"]["attempt_marker"]))
    _validate_marker(marker, plan=plan)
    controller = marker.get("controller")
    if (
        marker.get("schema") != ATTEMPT_MARKER_SCHEMA
        or marker.get("status") != "started"
        or marker.get("plan_sha256") != plan["plan_payload_sha256"]
        or not isinstance(controller, dict)
        or controller.get("pid") != controller_pid
        or controller.get("token_sha256") != _sha256_bytes(controller_token.encode())
    ):
        raise PreflightError("worker attempt marker is invalid", code="worker.guard")


def run_worker(
    plan_path: str | Path,
    *,
    launch_id: str,
    controller_pid: int,
    controller_token: str,
    hooks: WorkerHooks | None = None,
) -> dict[str, Any]:
    """Execute one rank workload and publish its immutable rank receipt."""

    resolved_plan = _absolute_path(plan_path, owner="plan")
    plan = _validate_plan(_load_json(resolved_plan), plan_path=resolved_plan)
    _validate_worker_guard(
        plan=plan,
        launch_id=launch_id,
        controller_pid=controller_pid,
        controller_token=controller_token,
    )
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as exc:
        raise PreflightError(
            "worker rank environment is malformed", code="worker.rank"
        ) from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE) or local_rank != rank:
        raise PreflightError("worker rank mapping is not exact", code="worker.rank")
    observed_environment = {
        name: os.environ.get(name) for name in sorted(STRICT_ENVIRONMENT)
    }
    if observed_environment != STRICT_ENVIRONMENT:
        raise PreflightError(
            "worker launch environment is not exact", code="worker.environment"
        )
    worker_hooks = hooks or _CudaWorkerHooks()
    if worker_hooks.cuda_is_initialized():
        raise PreflightError(
            "worker CUDA was initialized before policy admission",
            code="worker.pre_cuda",
        )
    pre_cuda = {
        "cuda_initialized": False,
        "environment": dict(observed_environment),
        "rank_environment": {
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
        },
    }
    policy = dict(worker_hooks.apply_seed_policy(SEED))
    if policy != _expected_policy_receipt():
        raise PreflightError(
            "worker determinism policy receipt is rejected", code="worker.policy"
        )
    runtime_path = Path(plan["runtime_admission"]["path"])
    runtime_provenance = _load_json(runtime_path)
    cleaned = False
    try:
        worker_hooks.initialize_cuda(local_rank)
        device_identity = dict(worker_hooks.observe_device_identity(local_rank))
        _validate_device_identity(device_identity, rank=rank)
        native_pre = dict(worker_hooks.attest_native(runtime_provenance))
        if (
            native_pre.get("schema_version") != 1
            or native_pre.get("cuda_initialized") is not True
            or native_pre.get("admitted") is not True
            or native_pre.get("mismatches") != []
        ):
            raise PreflightError(
                "live native execution attestation is rejected",
                code="worker.native",
            )
        workloads = dict(worker_hooks.run_workloads(rank, world_size))
        if not worker_hooks.cuda_is_initialized():
            raise PreflightError(
                "CUDA became uninitialized after workloads",
                code="worker.native",
            )
        native_post = dict(worker_hooks.attest_native(runtime_provenance))
        late_native = dict(worker_hooks.attest_late_native(runtime_provenance))
        if (
            native_post.get("schema_version") != 1
            or native_post.get("cuda_initialized") is not True
            or native_post.get("admitted") is not True
            or native_post.get("mismatches") != []
            or late_native.get("schema_version") != 1
            or late_native.get("cuda_initialized") is not True
            or late_native.get("admitted") is not True
            or late_native.get("mismatches") != []
        ):
            raise PreflightError(
                "post-workload native execution attestation is rejected",
                code="worker.native",
            )
    finally:
        cleaned = bool(worker_hooks.cleanup())
    if not cleaned:
        raise PreflightError(
            "worker process group cleanup failed", code="worker.cleanup"
        )
    payload = {
        "schema": RANK_RECEIPT_SCHEMA,
        "status": "passed",
        "plan_sha256": plan["plan_payload_sha256"],
        "launch_id": launch_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device_index": local_rank,
        "device_identity": device_identity,
        "pre_cuda": pre_cuda,
        "determinism_policy": policy,
        "runtime_admission": dict(plan["runtime_admission"]),
        "native_runtime_reference": dict(plan["native_runtime_reference"]),
        "mapped_native_execution": {
            "schema_version": 2,
            "cuda_initialized": True,
            "admitted": True,
            "mismatches": [],
            "pre_workload_attestation_sha256": _sha256_bytes(
                _canonical_json_bytes(native_pre)
            ),
            "post_workload_attestation_sha256": _sha256_bytes(
                _canonical_json_bytes(native_post)
            ),
            "late_native_mappings": late_native,
        },
        "workloads": workloads,
        "cleanup": {"process_group_destroyed": cleaned},
        "claim_scope": dict(plan["claim_scope"]),
    }
    receipt = finalize_receipt(payload)
    _validate_rank_receipt(receipt, plan=plan, launch_id=launch_id, rank=rank)
    rank_root = Path(plan["artifact_targets"]["rank_receipt_root"])
    launch_root = rank_root / launch_id
    launch_root.mkdir(parents=True, exist_ok=True)
    target = launch_root / f"rank-{rank:05d}.json"
    _publish_absent(target, receipt)
    return receipt


def _verify_passed_terminal_evidence(
    receipt: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    terminal = _validate_terminal_receipt(receipt, plan=plan)
    if terminal["status"] != "passed":
        raise PreflightError(
            "terminal receipt is not passed", code="preflight.verify_status"
        )
    marker_path = Path(plan["artifact_targets"]["attempt_marker"])
    marker = _load_json(marker_path)
    _validate_marker(marker, plan=plan)
    if terminal["attempt_marker"] != _receipt_binding(marker_path, marker):
        raise PreflightError(
            "terminal marker binding is not live",
            code="preflight.verify_marker",
        )
    rank_root = Path(plan["artifact_targets"]["rank_receipt_root"])
    if (
        not rank_root.is_dir()
        or rank_root.is_symlink()
        or {path.name for path in rank_root.iterdir()} != set(LAUNCH_IDS)
        or any(
            not (rank_root / launch_id).is_dir() or (rank_root / launch_id).is_symlink()
            for launch_id in LAUNCH_IDS
        )
    ):
        raise PreflightError(
            "rank receipt root inventory is not exact",
            code="preflight.verify_rank_inventory",
        )
    launches = {
        launch_id: _load_rank_receipts(plan, launch_id) for launch_id in LAUNCH_IDS
    }
    receipts = [
        rank_receipt for launch_id in LAUNCH_IDS for rank_receipt in launches[launch_id]
    ]
    live_bindings = [
        _receipt_binding(
            rank_root
            / rank_receipt["launch_id"]
            / f"rank-{rank_receipt['rank']:05d}.json",
            rank_receipt,
        )
        for rank_receipt in receipts
    ]
    if terminal["rank_receipts"] != live_bindings:
        raise PreflightError(
            "terminal rank receipt bindings are not live",
            code="preflight.verify_rank_binding",
        )
    comparison = _compare_launch_receipts(launches["launch-a"], launches["launch-b"])
    if terminal["comparisons"] != comparison:
        raise PreflightError(
            "terminal A/B comparison is not reproducible",
            code="preflight.verify_comparison",
        )
    if terminal["attempt_marker"] != _receipt_binding(
        marker_path, _load_json(marker_path)
    ):
        raise PreflightError(
            "attempt marker changed during verification",
            code="preflight.verify_marker",
        )
    return terminal


def verify_receipt(
    path: str | Path,
    *,
    expected_file_sha256: str,
    plan_path: str | Path,
    expected_plan_file_sha256: str,
) -> dict[str, Any]:
    receipt_path = _absolute_path(path, owner="receipt")
    resolved_plan = _absolute_path(plan_path, owner="plan")
    if (
        not _is_sha256(expected_plan_file_sha256)
        or _sha256_file(resolved_plan) != expected_plan_file_sha256
    ):
        raise PreflightError(
            "plan file SHA-256 does not match", code="preflight.verify_plan_sha"
        )
    plan = _validate_plan(_load_json(resolved_plan), plan_path=resolved_plan)
    if (
        not _is_sha256(expected_file_sha256)
        or _sha256_file(receipt_path) != expected_file_sha256
    ):
        raise PreflightError(
            "receipt file SHA-256 does not match", code="preflight.verify_sha"
        )
    receipt = _load_json(receipt_path)
    if receipt.get("schema") != TERMINAL_RECEIPT_SCHEMA:
        raise PreflightError(
            "terminal receipt schema is unsupported", code="preflight.verify_schema"
        )
    terminal = _verify_passed_terminal_evidence(receipt, plan=plan)
    if _sha256_file(receipt_path) != expected_file_sha256:
        raise PreflightError(
            "terminal receipt changed during verification",
            code="preflight.verify_sha",
        )
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    admit = subparsers.add_parser("admit-runtime")
    admit.add_argument("--receipt", required=True)
    admit.add_argument("--native-reference", required=True)
    admit.add_argument("--expected-native-reference-file-sha256", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--attempt-marker", required=True)
    prepare.add_argument("--terminal-receipt", required=True)
    prepare.add_argument("--rank-receipt-root", required=True)
    prepare.add_argument("--runtime-receipt", required=True)
    prepare.add_argument("--expected-runtime-file-sha256", required=True)
    prepare.add_argument("--native-runtime-receipt", required=True)
    prepare.add_argument("--expected-native-runtime-file-sha256", required=True)
    prepare.add_argument("--accelerate-executable", default="accelerate")
    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--expected-plan-file-sha256", required=True)
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--launch-id", required=True, choices=LAUNCH_IDS)
    worker.add_argument("--controller-pid", required=True, type=int)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--expected-plan-file-sha256", required=True)
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--expected-file-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "admit-runtime":
            value = admit_runtime(
                receipt_path=args.receipt,
                native_reference_path=args.native_reference,
                expected_native_reference_file_sha256=(
                    args.expected_native_reference_file_sha256
                ),
            )
        elif args.command == "prepare":
            value = prepare_plan(
                plan_path=args.plan,
                attempt_marker_path=args.attempt_marker,
                terminal_receipt_path=args.terminal_receipt,
                rank_receipt_root=args.rank_receipt_root,
                runtime_receipt_path=args.runtime_receipt,
                expected_runtime_file_sha256=args.expected_runtime_file_sha256,
                native_runtime_receipt_path=args.native_runtime_receipt,
                expected_native_runtime_file_sha256=(
                    args.expected_native_runtime_file_sha256
                ),
                accelerate_executable=args.accelerate_executable,
            )
        elif args.command == "run":
            value = run_plan(
                args.plan,
                expected_plan_file_sha256=args.expected_plan_file_sha256,
            )
        elif args.command == "_worker":
            value = run_worker(
                args.plan,
                launch_id=args.launch_id,
                controller_pid=args.controller_pid,
                controller_token=os.environ.get(_CONTROLLER_TOKEN_ENV, ""),
            )
        else:
            value = verify_receipt(
                args.receipt,
                expected_file_sha256=args.expected_file_sha256,
                plan_path=args.plan,
                expected_plan_file_sha256=args.expected_plan_file_sha256,
            )
    except PreflightError as exc:
        print(
            _canonical_json_bytes(
                {"status": "failed", "code": exc.code, "message": str(exc)}
            ).decode(),
            file=sys.stderr,
        )
        return 2
    print(_canonical_json_bytes(value).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
