"""One-shot executor for a frozen two-rank exact-resume qualification packet.

This is experiment-local qualification tooling.  It deliberately owns only
one schema, one six-command order, one exclusive attempt marker, and one
immutable outer terminal receipt.  Numeric execution bounds are consumed from
the manifest's machine-readable ``execution_contract``; the launch packet is
hash-bound evidence and is never parsed as an execution policy.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


MANIFEST_SCHEMA = "coordexp-swift-reconcile-resume-probe-command-manifest-v2"
MARKER_SCHEMA = "coordexp-swift-reconcile-resume-probe-attempt-marker-v1"
OUTER_RECEIPT_SCHEMA = "coordexp-swift-reconcile-resume-probe-outer-terminal-receipt-v1"
INNER_RECEIPT_SCHEMA = "coordexp-swift-reconcile-resume-probe-terminal-receipt-v1"
REVIEW_RECEIPT_SCHEMA = "coordexp-swift-reconcile-resume-probe-pre-cost-review-v1"
COMMAND_ORDER = (
    "setup",
    "success.uninterrupted_control",
    "success.resumed_child",
    "rank_failure",
    "interruption",
    "verification",
)
MODEL_COMMANDS = frozenset({"success.uninterrupted_control", "success.resumed_child"})
REQUIRED_RESOURCE_OBSERVATIONS = (
    "wall_time",
    "cpu_rss_per_rank",
    "gpu_memory_per_rank",
    "artifact_bytes",
)
REQUIRED_OUTER_BINDINGS = (
    "implementation_commit",
    "manifest_sha256",
    "packet_sha256",
    "marker_sha256",
    "argv_observations",
    "resource_maxima",
    "pre_cost_review",
    "launcher_identity",
    "runtime_identity",
    "command_process_groups",
    "artifact_tree_summaries",
    "cleanup_outcomes",
    "stop_outcome",
)
REQUIRED_INNER_BINDINGS = (
    "schema",
    "status",
    "commit",
    "artifact_root",
    "world_size",
    "receipt_payload_sha256",
)


class PacketExecutorError(RuntimeError):
    """Strict, bounded failure owned by this packet executor."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "packet_executor.invalid",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


class PopenLike(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PacketExecutorError(
            "value is not strict JSON", code="packet_executor.non_strict_json"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "receipt_payload_sha256": _sha256_bytes(_canonical_json_bytes(body)),
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _strict_json_load_bytes(raw: bytes, *, path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PacketExecutorError(
                    f"duplicate JSON key: {key}",
                    code="packet_executor.duplicate_json_key",
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except PacketExecutorError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PacketExecutorError(
            f"cannot read strict JSON: {path}",
            code="packet_executor.invalid_json",
            context={"path": str(path)},
        ) from exc
    if not isinstance(value, dict):
        raise PacketExecutorError(
            f"JSON root must be an object: {path}",
            code="packet_executor.invalid_json",
        )
    return value


def _strict_json_load(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PacketExecutorError(
            f"cannot read strict JSON: {path}",
            code="packet_executor.invalid_json",
            context={"path": str(path)},
        ) from exc
    return _strict_json_load_bytes(raw, path=path)


def _open_immutable_review(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PacketExecutorError(
            "independent pre-cost review must be an immutable regular file",
            code="packet_executor.review_missing_or_mutable",
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
            raise PacketExecutorError(
                "independent pre-cost review must be an immutable regular file",
                code="packet_executor.review_missing_or_mutable",
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        payload = _strict_json_load_bytes(raw, path=path)
        return payload, {
            "descriptor": descriptor,
            "path": path,
            "device": info.st_dev,
            "inode": info.st_ino,
            "sha256": _sha256_bytes(raw),
        }
    except BaseException:
        os.close(descriptor)
        raise


def _review_path_matches(guard: Mapping[str, Any]) -> bool:
    try:
        descriptor_info = os.fstat(int(guard["descriptor"]))
        path_info = os.lstat(Path(guard["path"]))
    except (OSError, KeyError, TypeError, ValueError):
        return False
    expected = (int(guard["device"]), int(guard["inode"]))
    return (
        stat.S_ISREG(path_info.st_mode)
        and not stat.S_ISLNK(path_info.st_mode)
        and not path_info.st_mode & 0o222
        and (descriptor_info.st_dev, descriptor_info.st_ino) == expected
        and (path_info.st_dev, path_info.st_ino) == expected
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PacketExecutorError(
            f"{field} must be an object", code="packet_executor.contract"
        )
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PacketExecutorError(
            f"{field} must be a non-empty string", code="packet_executor.contract"
        )
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PacketExecutorError(
            f"{field} must be an integer >= {minimum}",
            code="packet_executor.contract",
        )
    return value


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PacketExecutorError(
            f"{field} must be numeric", code="packet_executor.contract"
        )
    result = float(value)
    if not result >= minimum or result == float("inf"):
        raise PacketExecutorError(
            f"{field} must be finite and >= {minimum}",
            code="packet_executor.contract",
        )
    return result


def _absolute_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field)).expanduser()
    if not path.is_absolute():
        raise PacketExecutorError(
            f"{field} must be absolute", code="packet_executor.path_not_absolute"
        )
    return path


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _assert_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    rows = [path, *path.parents] if include_leaf else list(path.parents)
    for candidate in rows:
        info = _lstat(candidate)
        if info is not None and stat.S_ISLNK(info.st_mode):
            raise PacketExecutorError(
                f"path contains a symlink component: {candidate}",
                code="packet_executor.symlink_path",
            )


def _assert_absent(path: Path, field: str) -> None:
    _assert_no_symlink_components(path, include_leaf=True)
    if _lstat(path) is not None:
        raise PacketExecutorError(
            f"{field} must be absent: {path}",
            code="packet_executor.target_drift",
            context={"field": field, "path": str(path)},
        )


def _write_new_file(path: Path, payload: bytes) -> None:
    _assert_no_symlink_components(path, include_leaf=True)
    parent_info = _lstat(path.parent)
    if parent_info is None or not stat.S_ISDIR(parent_info.st_mode):
        raise PacketExecutorError(
            f"target parent is not an existing directory: {path.parent}",
            code="packet_executor.target_parent",
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        parent_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError:
        pass


def _verify_signed(
    payload: Mapping[str, Any], *, field: str, code: str
) -> dict[str, Any]:
    recorded = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    expected = _sha256_bytes(_canonical_json_bytes(body))
    if not isinstance(recorded, str) or recorded != expected:
        raise PacketExecutorError("signed receipt digest mismatch", code=code)
    return dict(payload)


def _git_output(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PacketExecutorError(
            "Git identity query failed",
            code="packet_executor.git_query",
            context={"args": list(args), "stderr": result.stderr[-1000:]},
        )
    return result.stdout.strip()


def _command_argv(manifest: Mapping[str, Any], name: str) -> list[str]:
    if name == "setup":
        value = manifest.get("setup_command")
    elif name == "verification":
        value = manifest.get("verification_command")
    elif name.startswith("success."):
        value = _mapping(manifest.get("commands"), "commands").get("success")
        value = _mapping(value, "commands.success").get(name.split(".", 1)[1])
    else:
        value = _mapping(manifest.get("commands"), "commands").get(name)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise PacketExecutorError(
            f"frozen argv for {name} is invalid", code="packet_executor.contract"
        )
    return list(value)


def _flatten_config_files(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    config_files = _mapping(manifest.get("config_files"), "config_files")
    success = _mapping(config_files.get("success"), "config_files.success")
    control = _mapping(
        success.get("uninterrupted_control"),
        "config_files.success.uninterrupted_control",
    )
    resumed = _mapping(
        success.get("resumed_child"), "config_files.success.resumed_child"
    )
    parent = _mapping(
        resumed.get("setup_parent"),
        "config_files.success.resumed_child.setup_parent",
    )
    child = _mapping(
        resumed.get("resumed_child"),
        "config_files.success.resumed_child.resumed_child",
    )
    rows = {
        "uninterrupted_control": control,
        "resumed_parent": parent,
        "resumed_child": child,
    }
    result: dict[str, dict[str, str]] = {}
    for role, row in rows.items():
        path = _absolute_path(row.get("path"), f"config_files.{role}.path")
        digest = _string(
            row.get("expected_sha256"), f"config_files.{role}.expected_sha256"
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise PacketExecutorError(
                f"config digest for {role} is invalid",
                code="packet_executor.contract",
            )
        result[role] = {"path": str(path), "expected_sha256": digest}
    return result


def _validate_contract(
    manifest: Mapping[str, Any],
    *,
    packet_path: Path,
    packet_sha256: str,
    attempt_marker_path: Path,
    terminal_receipt_path: Path,
) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PacketExecutorError(
            "only the schema-v2 command manifest is accepted",
            code="packet_executor.schema",
        )
    if manifest.get("world_size") != 2 or manifest.get("arms") != [
        "success",
        "rank_failure",
        "interruption",
    ]:
        raise PacketExecutorError(
            "manifest is not the fixed two-rank, three-arm packet",
            code="packet_executor.contract",
        )
    contract = dict(_mapping(manifest.get("execution_contract"), "execution_contract"))
    exact_values = {
        "command_order": list(COMMAND_ORDER),
        "marker_creation": "O_EXCL",
        "stop_on_first_failure": True,
        "retry_count": 0,
        "required_resource_observations": list(REQUIRED_RESOURCE_OBSERVATIONS),
        "required_outer_receipt_bindings": list(REQUIRED_OUTER_BINDINGS),
        "required_inner_receipt_bindings": list(REQUIRED_INNER_BINDINGS),
    }
    for field, expected in exact_values.items():
        if contract.get(field) != expected:
            raise PacketExecutorError(
                f"execution_contract.{field} drifted",
                code="packet_executor.contract",
            )

    bindings = _mapping(contract.get("bindings"), "execution_contract.bindings")
    if (
        _absolute_path(bindings.get("packet_path"), "bindings.packet_path")
        != packet_path
    ):
        raise PacketExecutorError(
            "packet path binding drifted", code="packet_executor.binding"
        )
    if bindings.get("packet_sha256") != packet_sha256:
        raise PacketExecutorError(
            "packet digest binding drifted", code="packet_executor.binding"
        )

    review = _mapping(
        contract.get("pre_cost_review"), "execution_contract.pre_cost_review"
    )
    review_path = _absolute_path(review.get("path"), "pre_cost_review.path")
    if review.get("schema") != REVIEW_RECEIPT_SCHEMA:
        raise PacketExecutorError(
            "pre-cost review schema drifted", code="packet_executor.review_contract"
        )
    if review.get("required_status") != "READY":
        raise PacketExecutorError(
            "pre-cost review status contract drifted",
            code="packet_executor.review_contract",
        )
    packet_author_identity = _string(
        review.get("packet_author_identity"),
        "pre_cost_review.packet_author_identity",
    )

    artifact_root = _absolute_path(manifest.get("artifact_root"), "artifact_root")
    targets = _mapping(contract.get("targets"), "execution_contract.targets")
    expected_targets = {
        "artifact_root": artifact_root,
        "attempt_marker_path": attempt_marker_path,
        "terminal_receipt_path": terminal_receipt_path,
    }
    for field, expected in expected_targets.items():
        if _absolute_path(targets.get(field), f"targets.{field}") != expected:
            raise PacketExecutorError(
                f"execution target {field} drifted", code="packet_executor.target_drift"
            )
    absent_values = targets.get("must_be_absent")
    if not isinstance(absent_values, list) or any(
        not isinstance(value, str) for value in absent_values
    ):
        raise PacketExecutorError(
            "targets.must_be_absent is invalid", code="packet_executor.contract"
        )
    absent_paths = [
        _absolute_path(value, "targets.must_be_absent") for value in absent_values
    ]
    if len(absent_paths) != len(set(absent_paths)):
        raise PacketExecutorError(
            "targets.must_be_absent contains duplicates",
            code="packet_executor.contract",
        )
    for required in (artifact_root, attempt_marker_path, terminal_receipt_path):
        if required not in absent_paths:
            raise PacketExecutorError(
                "required immutable target is not preflighted absent",
                code="packet_executor.contract",
            )

    preflight = _mapping(contract.get("preflight"), "execution_contract.preflight")
    filesystem_path = _absolute_path(
        preflight.get("filesystem_path"), "preflight.filesystem_path"
    )
    if not filesystem_path.is_dir():
        raise PacketExecutorError(
            "preflight filesystem path is not a directory",
            code="packet_executor.contract",
        )
    _integer(preflight.get("required_free_disk_bytes"), "required_free_disk_bytes")
    _integer(preflight.get("max_gpu_occupancy_bytes"), "max_gpu_occupancy_bytes")
    devices = preflight.get("gpu_devices")
    if not isinstance(devices, list) or len(devices) != 2:
        raise PacketExecutorError(
            "exactly two GPU device bindings are required",
            code="packet_executor.contract",
        )
    device_bindings: list[dict[str, str]] = []
    for index, device in enumerate(devices):
        row = _mapping(device, f"gpu_devices[{index}]")
        device_bindings.append(
            {
                "physical_index": _string(
                    row.get("physical_index"),
                    f"gpu_devices[{index}].physical_index",
                ),
                "uuid": _string(row.get("uuid"), f"gpu_devices[{index}].uuid"),
            }
        )
    device_uuids = [row["uuid"] for row in device_bindings]
    if len(set(device_uuids)) != 2:
        raise PacketExecutorError(
            "GPU UUID bindings must be unique", code="packet_executor.contract"
        )
    if len({row["physical_index"] for row in device_bindings}) != 2:
        raise PacketExecutorError(
            "GPU physical-index bindings must be unique",
            code="packet_executor.contract",
        )

    bounds = _mapping(
        contract.get("resource_bounds"), "execution_contract.resource_bounds"
    )
    if set(bounds) != set(COMMAND_ORDER):
        raise PacketExecutorError(
            "resource bounds must name exactly the six commands",
            code="packet_executor.contract",
        )
    normalized_bounds: dict[str, dict[str, Any]] = {}
    for name in COMMAND_ORDER:
        row = _mapping(bounds[name], f"resource_bounds.{name}")
        required_cpu_ranks = row.get("required_cpu_ranks")
        required_gpu_ranks = row.get("required_gpu_ranks")
        expected_cpu_ranks = [0, 1] if name in MODEL_COMMANDS else []
        expected_gpu_ranks = [0, 1] if name in MODEL_COMMANDS else []
        expected_cpu_mode = (
            "per_rank" if name in MODEL_COMMANDS else "command_tree_aggregate"
        )
        if row.get("cpu_measurement_mode") != expected_cpu_mode:
            raise PacketExecutorError(
                f"resource_bounds.{name}.cpu_measurement_mode drifted",
                code="packet_executor.contract",
            )
        if required_cpu_ranks != expected_cpu_ranks:
            raise PacketExecutorError(
                f"resource_bounds.{name}.required_cpu_ranks drifted",
                code="packet_executor.contract",
            )
        if required_gpu_ranks != expected_gpu_ranks:
            raise PacketExecutorError(
                f"resource_bounds.{name}.required_gpu_ranks drifted",
                code="packet_executor.contract",
            )
        normalized_bound = {
            "wall_time_seconds": _number(
                row.get("wall_time_seconds"),
                f"resource_bounds.{name}.wall_time_seconds",
            ),
            "cpu_measurement_mode": expected_cpu_mode,
            "max_gpu_memory_bytes_per_rank": _integer(
                row.get("max_gpu_memory_bytes_per_rank"),
                f"resource_bounds.{name}.max_gpu_memory_bytes_per_rank",
            ),
            "max_new_artifact_bytes": _integer(
                row.get("max_new_artifact_bytes"),
                f"resource_bounds.{name}.max_new_artifact_bytes",
            ),
            "required_cpu_ranks": list(required_cpu_ranks),
            "required_gpu_ranks": list(required_gpu_ranks),
        }
        if expected_cpu_mode == "per_rank":
            if "max_cpu_rss_command_tree_bytes" in row:
                raise PacketExecutorError(
                    f"resource_bounds.{name} mixes CPU measurement modes",
                    code="packet_executor.contract",
                )
            normalized_bound["max_cpu_rss_bytes_per_rank"] = _integer(
                row.get("max_cpu_rss_bytes_per_rank"),
                f"resource_bounds.{name}.max_cpu_rss_bytes_per_rank",
            )
        else:
            if "max_cpu_rss_bytes_per_rank" in row:
                raise PacketExecutorError(
                    f"resource_bounds.{name} mixes CPU measurement modes",
                    code="packet_executor.contract",
                )
            normalized_bound["max_cpu_rss_command_tree_bytes"] = _integer(
                row.get("max_cpu_rss_command_tree_bytes"),
                f"resource_bounds.{name}.max_cpu_rss_command_tree_bytes",
            )
        normalized_bounds[name] = normalized_bound
    total = _mapping(contract.get("total_bounds"), "execution_contract.total_bounds")
    normalized_total = {
        "wall_time_seconds": _number(
            total.get("wall_time_seconds"), "total_bounds.wall_time_seconds"
        ),
        "max_new_artifact_bytes": _integer(
            total.get("max_new_artifact_bytes"), "total_bounds.max_new_artifact_bytes"
        ),
    }

    setup = _mapping(
        contract.get("setup_validation"), "execution_contract.setup_validation"
    )
    setup_paths = {
        field: _absolute_path(setup.get(field), f"setup_validation.{field}")
        for field in (
            "prepare_receipt_path",
            "pack_cache_root",
            "pack_cache_receipt_path",
        )
    }
    if setup_paths["prepare_receipt_path"] != artifact_root / "prepare-receipt.json":
        raise PacketExecutorError(
            "prepare receipt target drifted", code="packet_executor.contract"
        )
    for field in ("pack_cache_root", "pack_cache_receipt_path"):
        if setup_paths[field] not in absent_paths:
            raise PacketExecutorError(
                f"{field} is not preflighted absent", code="packet_executor.contract"
            )

    inner = _mapping(contract.get("inner_receipt"), "execution_contract.inner_receipt")
    inner_path = _absolute_path(inner.get("path"), "inner_receipt.path")
    if inner_path != artifact_root / "terminal-receipt.json":
        raise PacketExecutorError(
            "inner receipt path drifted", code="packet_executor.contract"
        )
    if (
        inner.get("schema") != INNER_RECEIPT_SCHEMA
        or inner.get("required_status") != "verified"
    ):
        raise PacketExecutorError(
            "inner receipt contract drifted", code="packet_executor.contract"
        )

    summary = _mapping(
        contract.get("artifact_tree_summary"),
        "execution_contract.artifact_tree_summary",
    )
    summary_roots = [
        _absolute_path(value, "artifact_tree_summary.roots")
        for value in summary.get("roots", [])
    ]
    expected_summary_roots = [
        artifact_root,
        setup_paths["pack_cache_root"],
        setup_paths["pack_cache_receipt_path"],
    ]
    if summary_roots != expected_summary_roots:
        raise PacketExecutorError(
            "artifact-tree summary roots drifted",
            code="packet_executor.contract",
        )
    summary_bounds = {
        "roots": summary_roots,
        "max_entries": _integer(
            summary.get("max_entries"), "artifact_tree_summary.max_entries", minimum=1
        ),
        "max_depth": _integer(
            summary.get("max_depth"), "artifact_tree_summary.max_depth"
        ),
        "max_path_bytes": _integer(
            summary.get("max_path_bytes"),
            "artifact_tree_summary.max_path_bytes",
            minimum=1,
        ),
        "max_total_bytes": _integer(
            summary.get("max_total_bytes"),
            "artifact_tree_summary.max_total_bytes",
        ),
    }

    commands = {name: _command_argv(manifest, name) for name in COMMAND_ORDER}
    configs = _flatten_config_files(manifest)
    return {
        **contract,
        "artifact_root": artifact_root,
        "absent_paths": absent_paths,
        "commands": commands,
        "configs": configs,
        "filesystem_path": filesystem_path,
        "device_bindings": device_bindings,
        "device_uuids": device_uuids,
        "resource_bounds": normalized_bounds,
        "total_bounds": normalized_total,
        "setup_paths": setup_paths,
        "inner_path": inner_path,
        "review_path": review_path,
        "packet_author_identity": packet_author_identity,
        "artifact_summary": summary_bounds,
    }


def _pre_marker_validate(
    *,
    manifest_path: Path,
    packet_path: Path,
    expected_manifest_sha256: str,
    expected_packet_sha256: str,
    attempt_marker_path: Path,
    terminal_receipt_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    for path, field in ((manifest_path, "manifest"), (packet_path, "packet")):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise PacketExecutorError(
                f"{field} must be an absolute regular file",
                code="packet_executor.binding",
            )
    manifest_sha = _sha256_file(manifest_path)
    packet_sha = _sha256_file(packet_path)
    if manifest_sha != expected_manifest_sha256:
        raise PacketExecutorError(
            "manifest digest differs from the expected frozen digest",
            code="packet_executor.manifest_digest",
        )
    if packet_sha != expected_packet_sha256:
        raise PacketExecutorError(
            "packet digest differs from the expected frozen digest",
            code="packet_executor.packet_digest",
        )
    manifest = _strict_json_load(manifest_path)
    cwd = _absolute_path(manifest.get("cwd"), "cwd")
    if Path.cwd().resolve() != cwd.resolve():
        raise PacketExecutorError(
            "current working directory differs from the frozen cwd",
            code="packet_executor.cwd_drift",
        )
    head = _git_output(cwd, "rev-parse", "HEAD")
    commit = _string(manifest.get("implementation_commit"), "implementation_commit")
    if head != commit:
        raise PacketExecutorError(
            "HEAD differs from the frozen implementation commit",
            code="packet_executor.head_drift",
        )
    tracked = _git_output(cwd, "status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise PacketExecutorError(
            "tracked worktree differs from HEAD",
            code="packet_executor.tracked_diff",
            context={"status": tracked.splitlines()[:20]},
        )
    contract = _validate_contract(
        manifest,
        packet_path=packet_path,
        packet_sha256=packet_sha,
        attempt_marker_path=attempt_marker_path,
        terminal_receipt_path=terminal_receipt_path,
    )
    review_path: Path = contract["review_path"]
    for index, target in enumerate(contract["absent_paths"]):
        _assert_absent(target, f"targets.must_be_absent[{index}]")
    review_payload, review_guard = _open_immutable_review(review_path)
    try:
        review = _verify_signed(
            review_payload,
            field="receipt_payload_sha256",
            code="packet_executor.review_digest",
        )
    except PacketExecutorError as exc:
        os.close(int(review_guard["descriptor"]))
        raise PacketExecutorError(
            "independent pre-cost review digest mismatched",
            code="packet_executor.review_digest",
        ) from exc
    expected_review = {
        "schema": REVIEW_RECEIPT_SCHEMA,
        "status": "READY",
        "implementation_commit": commit,
        "manifest_sha256": manifest_sha,
        "packet_sha256": packet_sha,
    }
    for field, expected in expected_review.items():
        if review.get(field) != expected:
            os.close(int(review_guard["descriptor"]))
            raise PacketExecutorError(
                f"independent pre-cost review {field} mismatched",
                code="packet_executor.review_mismatch",
            )
    try:
        reviewer_identity = _string(
            review.get("reviewer_identity"), "review.reviewer_identity"
        )
        if reviewer_identity == contract["packet_author_identity"]:
            raise PacketExecutorError(
                "pre-cost reviewer is not independent from the packet author",
                code="packet_executor.review_not_independent",
            )
    except BaseException:
        os.close(int(review_guard["descriptor"]))
        raise
    return (
        manifest,
        contract,
        manifest_sha,
        packet_sha,
        {
            "path": str(review_path),
            "sha256": review_guard["sha256"],
            "device": review_guard["device"],
            "inode": review_guard["inode"],
            "payload": review,
        },
        review_guard,
    )


def _pid_starttime(pid: int) -> int | None:
    identity = _proc_identity(pid)
    return None if identity is None else identity[2]


def _proc_identity(pid: int) -> tuple[int, int, int] | None:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
        tail = text[text.rfind(")") + 2 :].split()
        return int(tail[1]), int(tail[2]), int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def _pid_rank(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    values: dict[bytes, bytes] = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            values[key] = value
    for key in (b"LOCAL_RANK", b"RANK"):
        try:
            return int(values[key])
        except (KeyError, ValueError):
            continue
    return None


def _default_process_sampler() -> list[dict[str, Any]]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    records: dict[int, tuple[int, int, int]] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / "stat").read_text(encoding="ascii")
            tail = text[text.rfind(")") + 2 :].split()
            pid = int(path.name)
            records[pid] = (int(tail[1]), int(tail[19]), int(tail[21]) * page_size)
        except (OSError, ValueError, IndexError):
            continue
    descendants: set[int] = set()
    frontier = {os.getpid()}
    while frontier:
        children = {pid for pid, (ppid, _, _) in records.items() if ppid in frontier}
        children -= descendants
        descendants.update(children)
        frontier = children
    return [
        {
            "pid": pid,
            "ppid": records[pid][0],
            "starttime": records[pid][1],
            "rank": _pid_rank(pid),
            "rss_bytes": records[pid][2],
        }
        for pid in sorted(descendants)
    ]


def _default_gpu_sampler() -> list[dict[str, Any]]:
    device_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if device_result.returncode != 0:
        raise PacketExecutorError(
            "nvidia-smi device sampling failed", code="packet_executor.gpu_sample"
        )
    rows: list[dict[str, Any]] = []
    for line in device_result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            raise PacketExecutorError(
                "malformed nvidia-smi device row", code="packet_executor.gpu_sample"
            )
        rows.append(
            {
                "kind": "device",
                "physical_index": parts[0],
                "gpu_uuid": parts[1],
                "memory_used_bytes": int(parts[2]) * 1024 * 1024,
            }
        )
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process_result.returncode not in (0, 9):
        raise PacketExecutorError(
            "nvidia-smi process sampling failed", code="packet_executor.gpu_sample"
        )
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            raise PacketExecutorError(
                "malformed nvidia-smi process row", code="packet_executor.gpu_sample"
            )
        pid = int(parts[1])
        starttime = _pid_starttime(pid)
        if starttime is None:
            continue
        rows.append(
            {
                "kind": "process",
                "gpu_uuid": parts[0],
                "pid": pid,
                "starttime": starttime,
                "rank": _pid_rank(pid),
                "gpu_memory_bytes": int(parts[2]) * 1024 * 1024,
            }
        )
    return rows


def _default_launch(argv: Sequence[str], *, cwd: Path) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _normalize_process_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise PacketExecutorError(
            "process sampler must return a list",
            code="packet_executor.process_sample",
        )
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PacketExecutorError(
                "invalid process sample", code="packet_executor.process_sample"
            )
        pid = _integer(row.get("pid"), "process.pid", minimum=1)
        starttime = _integer(row.get("starttime"), "process.starttime", minimum=1)
        rank = row.get("rank")
        if rank is not None:
            rank = _integer(rank, "process.rank")
        rss = _integer(row.get("rss_bytes"), "process.rss_bytes")
        ppid = row.get("ppid")
        if ppid is not None:
            ppid = _integer(ppid, "process.ppid")
        result.append(
            {
                "pid": pid,
                "ppid": ppid,
                "starttime": starttime,
                "rank": rank,
                "rss_bytes": rss,
            }
        )
    return result


def _normalize_gpu_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise PacketExecutorError(
            "GPU sampler must return a list", code="packet_executor.gpu_sample"
        )
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PacketExecutorError(
                "invalid GPU sample", code="packet_executor.gpu_sample"
            )
        kind = row.get("kind")
        uuid = _string(row.get("gpu_uuid"), "gpu.gpu_uuid")
        if kind == "device":
            result.append(
                {
                    "kind": "device",
                    "physical_index": _string(
                        row.get("physical_index"), "gpu.physical_index"
                    ),
                    "gpu_uuid": uuid,
                    "memory_used_bytes": _integer(
                        row.get("memory_used_bytes"), "gpu.memory_used_bytes"
                    ),
                }
            )
        elif kind == "process":
            rank = row.get("rank")
            if rank is not None:
                rank = _integer(rank, "gpu.rank")
            result.append(
                {
                    "kind": "process",
                    "gpu_uuid": uuid,
                    "pid": _integer(row.get("pid"), "gpu.pid", minimum=1),
                    "starttime": _integer(
                        row.get("starttime"), "gpu.starttime", minimum=1
                    ),
                    "rank": rank,
                    "gpu_memory_bytes": _integer(
                        row.get("gpu_memory_bytes"), "gpu.gpu_memory_bytes"
                    ),
                }
            )
        else:
            raise PacketExecutorError(
                "invalid GPU sample kind", code="packet_executor.gpu_sample"
            )
    return result


def _summarize_artifact_tree(bounds: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for root in bounds["roots"]:
        info = _lstat(root)
        if info is None:
            continue
        paths = [root]
        if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            paths.extend(sorted(root.rglob("*"), key=str))
        for path in paths:
            relative = "." if path == root else path.relative_to(root).as_posix()
            depth = 0 if relative == "." else len(Path(relative).parts)
            if depth > int(bounds["max_depth"]):
                raise PacketExecutorError(
                    "artifact-tree depth exceeds its frozen bound",
                    code="packet_executor.artifact_summary_bound",
                )
            if len(relative.encode("utf-8")) > int(bounds["max_path_bytes"]):
                raise PacketExecutorError(
                    "artifact-tree path exceeds its frozen bound",
                    code="packet_executor.artifact_summary_bound",
                )
            child = _lstat(path)
            if child is None:
                continue
            kind = (
                "symlink"
                if stat.S_ISLNK(child.st_mode)
                else "directory"
                if stat.S_ISDIR(child.st_mode)
                else "file"
                if stat.S_ISREG(child.st_mode)
                else "other"
            )
            size = child.st_size if kind == "file" else 0
            total_bytes += size
            entries.append(
                {
                    "root": str(root),
                    "path": relative,
                    "kind": kind,
                    "size_bytes": size,
                }
            )
            if len(entries) > int(bounds["max_entries"]):
                raise PacketExecutorError(
                    "artifact-tree entry count exceeds its frozen bound",
                    code="packet_executor.artifact_summary_bound",
                )
            if total_bytes > int(bounds["max_total_bytes"]):
                raise PacketExecutorError(
                    "artifact-tree bytes exceed their frozen bound",
                    code="packet_executor.artifact_summary_bound",
                )
    return {
        "roots": [str(path) for path in bounds["roots"]],
        "entry_count": len(entries),
        "total_bytes": total_bytes,
        "inventory_sha256": _sha256_bytes(_canonical_json_bytes(entries)),
        "entries": entries,
        "bounds": {
            key: int(bounds[key])
            for key in ("max_entries", "max_depth", "max_path_bytes", "max_total_bytes")
        },
    }


def _sample_resources(
    process_sampler: Callable[[], list[dict[str, Any]]],
    gpu_sampler: Callable[[], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        _normalize_process_rows(process_sampler()),
        _normalize_gpu_rows(gpu_sampler()),
    )


def _owned_process_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    leader_pid: int,
    leader_starttime: int,
    retained_identities: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    retained = retained_identities or set()
    leader = next(
        (
            row
            for row in rows
            if int(row["pid"]) == leader_pid
            and int(row["starttime"]) == leader_starttime
        ),
        None,
    )
    owned_identities = {
        (int(row["pid"]), int(row["starttime"]))
        for row in rows
        if (int(row["pid"]), int(row["starttime"])) in retained
    }
    if leader is not None:
        owned_identities.add((leader_pid, leader_starttime))
    if not owned_identities:
        return []
    frontier = {pid for pid, _ in owned_identities}
    while frontier:
        children = [
            row
            for row in rows
            if row.get("ppid") in frontier
            and (int(row["pid"]), int(row["starttime"])) not in owned_identities
        ]
        owned_identities.update(
            (int(row["pid"]), int(row["starttime"])) for row in children
        )
        frontier = {int(row["pid"]) for row in children}
    return [
        dict(row)
        for row in rows
        if (int(row["pid"]), int(row["starttime"])) in owned_identities
    ]


def _owned_gpu_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    owned_processes: Sequence[Mapping[str, Any]],
    selected_uuids: set[str],
) -> list[dict[str, Any]]:
    identities = {(int(row["pid"]), int(row["starttime"])) for row in owned_processes}
    return [
        dict(row)
        for row in rows
        if row.get("kind") == "process"
        and str(row["gpu_uuid"]) in selected_uuids
        and (int(row["pid"]), int(row["starttime"])) in identities
    ]


def _merge_process_samples(
    target: dict[tuple[int, int], dict[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> None:
    for row in rows:
        key = (int(row["pid"]), int(row["starttime"]))
        existing = target.get(key)
        if existing is None or int(row["rss_bytes"]) > int(existing["rss_bytes"]):
            target[key] = dict(row)


def _merge_gpu_samples(
    target: dict[tuple[str, int, int], dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in rows:
        if row.get("kind") != "process":
            continue
        key = (str(row["gpu_uuid"]), int(row["pid"]), int(row["starttime"]))
        existing = target.get(key)
        if existing is None or int(row["gpu_memory_bytes"]) > int(
            existing["gpu_memory_bytes"]
        ):
            target[key] = dict(row)


def _rank_maxima(rows: Sequence[Mapping[str, Any]], value_field: str) -> dict[str, int]:
    maxima: dict[str, int] = {}
    for row in rows:
        rank = row.get("rank")
        if rank is None:
            continue
        key = str(int(rank))
        maxima[key] = max(maxima.get(key, 0), int(row[value_field]))
    return maxima


def _check_command_resources(
    *,
    name: str,
    bound: Mapping[str, Any],
    elapsed: float,
    new_artifact_bytes: int,
    process_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    cpu_rss_command_tree_max_bytes: int | None,
) -> tuple[dict[str, int], dict[str, int], int | None]:
    if elapsed > float(bound["wall_time_seconds"]):
        raise PacketExecutorError(
            f"{name} exceeded its wall-time bound",
            code="packet_executor.wall_bound",
        )
    if new_artifact_bytes > int(bound["max_new_artifact_bytes"]):
        raise PacketExecutorError(
            f"{name} exceeded its artifact-byte bound",
            code="packet_executor.artifact_bound",
        )
    cpu_mode = str(bound["cpu_measurement_mode"])
    cpu = _rank_maxima(process_rows, "rss_bytes")
    gpu = _rank_maxima(gpu_rows, "gpu_memory_bytes")
    required_cpu = {str(rank) for rank in bound["required_cpu_ranks"]}
    required_gpu = {str(rank) for rank in bound["required_gpu_ranks"]}
    if cpu_mode == "command_tree_aggregate":
        cpu = {}
    if not required_gpu:
        gpu = {}
    missing_cpu = sorted(required_cpu - set(cpu))
    if missing_cpu:
        raise PacketExecutorError(
            f"{name} is missing CPU RSS measurements for ranks {missing_cpu}",
            code="packet_executor.missing_process_rank",
        )
    missing_gpu = sorted(required_gpu - set(gpu))
    if missing_gpu:
        raise PacketExecutorError(
            f"{name} is missing GPU measurements for ranks {missing_gpu}",
            code="packet_executor.missing_gpu_rank",
        )
    if cpu_mode == "per_rank":
        if any(value > int(bound["max_cpu_rss_bytes_per_rank"]) for value in cpu.values()):
            raise PacketExecutorError(
                f"{name} exceeded its CPU RSS bound", code="packet_executor.rss_bound"
            )
        cpu_rss_command_tree_max_bytes = None
    else:
        if cpu_rss_command_tree_max_bytes is None:
            raise PacketExecutorError(
                f"{name} has no owned command-tree CPU RSS sample",
                code="packet_executor.missing_owned_process_sample",
            )
        if cpu_rss_command_tree_max_bytes > int(
            bound["max_cpu_rss_command_tree_bytes"]
        ):
            raise PacketExecutorError(
                f"{name} exceeded its CPU RSS bound", code="packet_executor.rss_bound"
            )
    if any(
        value > int(bound["max_gpu_memory_bytes_per_rank"]) for value in gpu.values()
    ):
        raise PacketExecutorError(
            f"{name} exceeded its GPU-memory bound", code="packet_executor.gpu_bound"
        )
    return cpu, gpu, cpu_rss_command_tree_max_bytes


def _is_popen_like(process: Any) -> bool:
    return isinstance(getattr(process, "pid", None), int) and all(
        callable(getattr(process, name, None))
        for name in ("poll", "wait", "terminate", "kill")
    )


def _process_group_absent(
    *,
    leader_pid: int,
    leader_starttime: int,
    process_group_id: int,
    trusted_group: bool,
) -> bool:
    del leader_pid, leader_starttime
    if not trusted_group:
        return False
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / "stat").read_text(encoding="ascii")
            tail = text[text.rfind(")") + 2 :].split()
            if int(tail[2]) == process_group_id:
                return False
        except (OSError, ValueError, IndexError):
            continue
    return True


def _process_identity(process: PopenLike) -> tuple[int, int, int, bool]:
    leader_pid = int(process.pid)
    declared_starttime = getattr(process, "starttime", None)
    declared_group = getattr(process, "process_group_id", None)
    live_identity = _proc_identity(leader_pid)
    if live_identity is not None:
        if live_identity[0] != os.getpid():
            raise PacketExecutorError(
                "launched process PID is not a direct child of the executor",
                code="packet_executor.foreign_process",
            )
        leader_starttime = live_identity[2]
        process_group_id = live_identity[1]
        trusted_group = True
    elif isinstance(declared_starttime, int) and isinstance(declared_group, int):
        leader_starttime = declared_starttime
        process_group_id = declared_group
        trusted_group = False
    else:
        raise PacketExecutorError(
            "launched process PID is not a direct child of the executor",
            code="packet_executor.foreign_process",
        )
    if not isinstance(leader_starttime, int) or leader_starttime < 1:
        raise PacketExecutorError(
            "launched process starttime is unavailable",
            code="packet_executor.launch_identity",
        )
    if not isinstance(process_group_id, int) or process_group_id < 1:
        raise PacketExecutorError(
            "launched process group is unavailable",
            code="packet_executor.launch_identity",
        )
    if process_group_id == os.getpgrp():
        raise PacketExecutorError(
            "launched process uses the executor's own process group",
            code="packet_executor.own_process_group",
        )
    return leader_pid, leader_starttime, process_group_id, trusted_group


def _current_group_identities(
    process_group_id: int, *, trusted_group: bool
) -> set[tuple[int, int]]:
    if not trusted_group:
        return set()
    identities: set[tuple[int, int]] = set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        identity = _proc_identity(int(path.name))
        if identity is not None and identity[1] == process_group_id:
            identities.add((int(path.name), identity[2]))
    return identities


def _identities_absent(identities: set[tuple[int, int]]) -> bool:
    return all(_pid_starttime(pid) != starttime for pid, starttime in identities)


def _signal_exact_identities(
    identities: set[tuple[int, int]],
    sig: signal.Signals,
    errors: list[str],
    *,
    trusted_group: bool,
) -> bool:
    if not trusted_group:
        return False
    sent = False
    for pid, starttime in sorted(identities):
        if pid == os.getpid() or _pid_starttime(pid) != starttime:
            continue
        try:
            os.kill(pid, sig)
            sent = True
        except ProcessLookupError:
            continue
        except BaseException as exc:
            errors.append(f"{sig.name.lower()}:{pid}:{exc}")
    return sent


def _cleanup_process(
    process: PopenLike,
    *,
    leader_starttime: int,
    process_group_id: int,
    trusted_group: bool,
    retained_identities: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if not trusted_group:
        cleanup = _cleanup_unidentified_process(process)
        cleanup["errors"][0] = (
            "process-group identity was not launch-trusted; "
            "OS group was not inspected or signalled"
        )
        return cleanup
    leader_pid = int(process.pid)
    retained = set(retained_identities or set())
    retained.add((leader_pid, leader_starttime))
    sent: list[str] = []
    errors: list[str] = []
    current_leader_starttime = _pid_starttime(leader_pid)
    group_is_owned = current_leader_starttime in (None, leader_starttime)
    group_identities = (
        _current_group_identities(process_group_id, trusted_group=trusted_group)
        if group_is_owned
        else set()
    )
    confirmed_identities = retained | group_identities
    term_targets = {
        identity for identity in retained if _pid_starttime(identity[0]) == identity[1]
    } | group_identities
    term_sent = _signal_exact_identities(
        term_targets, signal.SIGTERM, errors, trusted_group=trusted_group
    )
    try:
        running = process.poll() is None
    except BaseException as exc:
        running = True
        errors.append(f"poll:{exc}")
    if running and not term_sent:
        try:
            process.terminate()
            term_sent = True
        except BaseException as exc:
            errors.append(f"term:{exc}")
    if term_sent:
        sent.append("TERM")

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        group_absent = _process_group_absent(
            leader_pid=leader_pid,
            leader_starttime=leader_starttime,
            process_group_id=process_group_id,
            trusted_group=trusted_group,
        )
        descendants_absent = _identities_absent(confirmed_identities)
        try:
            reaped = process.poll() is not None
        except BaseException as exc:
            reaped = False
            errors.append(f"poll:{exc}")
        if group_absent and descendants_absent and reaped:
            break
        time.sleep(0.05)

    group_absent = _process_group_absent(
        leader_pid=leader_pid,
        leader_starttime=leader_starttime,
        process_group_id=process_group_id,
        trusted_group=trusted_group,
    )
    descendants_absent = _identities_absent(confirmed_identities)
    try:
        reaped = process.poll() is not None
    except BaseException as exc:
        reaped = False
        errors.append(f"poll:{exc}")
    if not (group_absent and descendants_absent and reaped):
        kill_targets = {
            identity
            for identity in confirmed_identities
            if _pid_starttime(identity[0]) == identity[1]
        }
        if group_is_owned:
            current_group = _current_group_identities(
                process_group_id, trusted_group=trusted_group
            )
            confirmed_identities.update(current_group)
            kill_targets.update(current_group)
        kill_sent = _signal_exact_identities(
            kill_targets, signal.SIGKILL, errors, trusted_group=trusted_group
        )
        try:
            if process.poll() is None and not kill_sent:
                process.kill()
                kill_sent = True
        except BaseException as exc:
            errors.append(f"kill:{exc}")
        if kill_sent:
            sent.append("KILL")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            group_absent = _process_group_absent(
                leader_pid=leader_pid,
                leader_starttime=leader_starttime,
                process_group_id=process_group_id,
                trusted_group=trusted_group,
            )
            descendants_absent = _identities_absent(confirmed_identities)
            try:
                reaped = process.poll() is not None
            except BaseException:
                reaped = False
            if group_absent and descendants_absent and reaped:
                break
            time.sleep(0.05)
    try:
        process.wait(timeout=0)
    except subprocess.TimeoutExpired:
        pass
    except BaseException as exc:
        errors.append(f"reap:{exc}")
    try:
        group_absent = _process_group_absent(
            leader_pid=leader_pid,
            leader_starttime=leader_starttime,
            process_group_id=process_group_id,
            trusted_group=trusted_group,
        )
        descendants_absent = _identities_absent(confirmed_identities)
    except BaseException as exc:
        group_absent = False
        descendants_absent = False
        errors.append(f"absence:{exc}")
    try:
        reaped = process.poll() is not None
    except BaseException as exc:
        reaped = False
        errors.append(f"final_poll:{exc}")
    return {
        "status": "confirmed_absent"
        if group_absent and descendants_absent and reaped and not errors
        else "failed",
        "sent_signals": sent,
        "reaped": reaped,
        "group_absent": group_absent,
        "descendants_absent": descendants_absent,
        "retained_descendant_identities": [
            {"pid": pid, "starttime": starttime}
            for pid, starttime in sorted(confirmed_identities)
            if (pid, starttime) != (leader_pid, leader_starttime)
        ],
        "errors": errors,
    }


def _cleanup_unidentified_process(
    process: PopenLike, *, allow_process_signals: bool = True
) -> dict[str, Any]:
    sent: list[str] = []
    errors = ["process-group identity unavailable"]
    if not allow_process_signals:
        return {
            "status": "failed",
            "sent_signals": sent,
            "reaped": False,
            "group_absent": False,
            "descendants_absent": False,
            "errors": errors + ["unsafe process identity; no signal sent"],
        }
    try:
        if process.poll() is None:
            process.terminate()
            sent.append("TERM")
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            sent.append("KILL")
            process.wait(timeout=5)
        except BaseException as exc:
            errors.append(f"kill_or_reap:{exc}")
    except BaseException as exc:
        errors.append(f"term_or_reap:{exc}")
    try:
        reaped = process.poll() is not None
    except BaseException as exc:
        reaped = False
        errors.append(f"final_poll:{exc}")
    return {
        "status": "failed",
        "sent_signals": sent,
        "reaped": reaped,
        "group_absent": False,
        "descendants_absent": False,
        "errors": errors,
    }


def _run_command(
    *,
    name: str,
    argv: Sequence[str],
    cwd: Path,
    bound: Mapping[str, Any],
    artifact_summary_bounds: Mapping[str, Any],
    launch: Callable[..., Any],
    gpu_sampler: Callable[[], list[dict[str, Any]]],
    process_sampler: Callable[[], list[dict[str, Any]]],
    selected_uuids: set[str],
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_before: dict[str, Any] | None = None
    artifact_after: dict[str, Any] | None = None
    process_observations: dict[tuple[int, int], dict[str, Any]] = {}
    retained_process_identities: set[tuple[int, int]] = set()
    gpu_observations: dict[tuple[str, int, int], dict[str, Any]] = {}
    cpu_rss_command_tree_max_bytes: int | None = None
    launched: PopenLike | None = None
    process_group: dict[str, int | bool] | None = None
    cleanup: dict[str, Any] | None = None
    error: BaseException | None = None
    returncode: int | None = None
    try:
        artifact_before = _summarize_artifact_tree(artifact_summary_bounds)
        try:
            candidate = launch(list(argv), cwd=cwd)
        except BaseException as exc:
            raise PacketExecutorError(
                f"{name} launch raised: {exc}",
                code="packet_executor.launch_exception",
            ) from exc
        if not _is_popen_like(candidate):
            raise PacketExecutorError(
                "launch must return exactly one Popen-like process",
                code="packet_executor.launch_protocol",
            )
        launched = candidate
        leader_pid, leader_starttime, process_group_id, trusted_group = (
            _process_identity(launched)
        )
        process_group = {
            "leader_pid": leader_pid,
            "leader_starttime": leader_starttime,
            "process_group_id": process_group_id,
            "trusted_group": trusted_group,
        }
        while True:
            returncode = launched.poll()
            process_rows, gpu_rows = _sample_resources(process_sampler, gpu_sampler)
            owned_processes = _owned_process_rows(
                process_rows,
                leader_pid=leader_pid,
                leader_starttime=leader_starttime,
                retained_identities=retained_process_identities,
            )
            retained_process_identities.update(
                (int(row["pid"]), int(row["starttime"])) for row in owned_processes
            )
            if (
                bound["cpu_measurement_mode"] == "command_tree_aggregate"
                and owned_processes
            ):
                snapshot_sum = sum(int(row["rss_bytes"]) for row in owned_processes)
                cpu_rss_command_tree_max_bytes = max(
                    cpu_rss_command_tree_max_bytes or 0, snapshot_sum
                )
            owned_gpu = _owned_gpu_rows(
                gpu_rows,
                owned_processes=owned_processes,
                selected_uuids=selected_uuids,
            )
            _merge_process_samples(process_observations, owned_processes)
            _merge_gpu_samples(gpu_observations, owned_gpu)
            if returncode is not None:
                break
            if time.monotonic() - started > float(bound["wall_time_seconds"]):
                raise PacketExecutorError(
                    f"{name} exceeded its wall-time bound while running",
                    code="packet_executor.wall_timeout",
                )
            time.sleep(0.25)
    except BaseException as exc:
        error = exc
    finally:
        if launched is not None and process_group is not None:
            cleanup = _cleanup_process(
                launched,
                leader_starttime=int(process_group["leader_starttime"]),
                process_group_id=int(process_group["process_group_id"]),
                trusted_group=bool(process_group["trusted_group"]),
                retained_identities=retained_process_identities,
            )
            try:
                returncode = launched.poll()
            except BaseException:
                pass
            if cleanup["status"] != "confirmed_absent":
                error = PacketExecutorError(
                    "process-group cleanup could not be confirmed",
                    code="packet_executor.cleanup_failed",
                    context={"cleanup": cleanup},
                )
        elif launched is not None:
            unsafe_identity = isinstance(error, PacketExecutorError) and error.code in {
                "packet_executor.foreign_process",
                "packet_executor.own_process_group",
            }
            cleanup = _cleanup_unidentified_process(
                launched, allow_process_signals=not unsafe_identity
            )
            error = PacketExecutorError(
                "process-group cleanup could not be confirmed",
                code="packet_executor.cleanup_failed",
                context={"cleanup": cleanup},
            )
        try:
            artifact_after = _summarize_artifact_tree(artifact_summary_bounds)
        except BaseException as exc:
            if error is None:
                error = exc
    elapsed = max(0.0, time.monotonic() - started)
    new_bytes = max(
        0,
        int((artifact_after or {}).get("total_bytes", 0))
        - int((artifact_before or {}).get("total_bytes", 0)),
    )
    cpu = _rank_maxima(list(process_observations.values()), "rss_bytes")
    gpu = _rank_maxima(list(gpu_observations.values()), "gpu_memory_bytes")
    if error is None:
        try:
            cpu, gpu, cpu_rss_command_tree_max_bytes = _check_command_resources(
                name=name,
                bound=bound,
                elapsed=elapsed,
                new_artifact_bytes=new_bytes,
                process_rows=list(process_observations.values()),
                gpu_rows=list(gpu_observations.values()),
                cpu_rss_command_tree_max_bytes=cpu_rss_command_tree_max_bytes,
            )
        except BaseException as exc:
            error = exc
    return {
        "name": name,
        "argv": list(argv),
        "argv_sha256": _sha256_bytes(_canonical_json_bytes(list(argv))),
        "attempt": 1,
        "returncode": returncode,
        "wall_time_seconds": elapsed,
        "new_artifact_bytes": new_bytes,
        "cpu_measurement_mode": bound["cpu_measurement_mode"],
        "cpu_rss_bytes_per_rank": cpu,
        "cpu_rss_command_tree_max_bytes": cpu_rss_command_tree_max_bytes,
        "max_cpu_rss_bytes_per_rank": bound.get("max_cpu_rss_bytes_per_rank"),
        "max_cpu_rss_command_tree_bytes": bound.get("max_cpu_rss_command_tree_bytes"),
        "gpu_memory_bytes_per_rank": gpu,
        "required_cpu_ranks": list(bound["required_cpu_ranks"]),
        "required_gpu_ranks": list(bound["required_gpu_ranks"]),
        "process_observations": sorted(
            process_observations.values(),
            key=lambda row: (row["pid"], row["starttime"]),
        ),
        "gpu_observations": sorted(
            gpu_observations.values(),
            key=lambda row: (row["gpu_uuid"], row["pid"], row["starttime"]),
        ),
        "process_group": process_group,
        "cleanup": cleanup,
        "artifact_tree": {"before": artifact_before, "after": artifact_after},
        "execution_error": None
        if error is None
        else _error_record(error, command=name),
    }


def _preflight_resources(
    contract: Mapping[str, Any],
    gpu_sampler: Callable[[], list[dict[str, Any]]],
    *,
    phase: str,
) -> dict[str, Any]:
    free = shutil.disk_usage(contract["filesystem_path"]).free
    required_free = int(
        _mapping(contract["preflight"], "preflight")["required_free_disk_bytes"]
    )
    if free < required_free:
        raise PacketExecutorError(
            "free disk is below the frozen bound",
            code="packet_executor.free_disk_bound",
        )
    rows = _normalize_gpu_rows(gpu_sampler())
    device_rows = {
        (row["physical_index"], row["gpu_uuid"]): row
        for row in rows
        if row["kind"] == "device"
    }
    required_pairs = {
        (row["physical_index"], row["uuid"]) for row in contract["device_bindings"]
    }
    if not required_pairs <= set(device_rows):
        raise PacketExecutorError(
            "selected physical-index to GPU-UUID inventory drifted",
            code="packet_executor.gpu_uuid_drift",
        )
    maximum = int(
        _mapping(contract["preflight"], "preflight")["max_gpu_occupancy_bytes"]
    )
    occupied = {
        f"{index}:{uuid}": int(device_rows[(index, uuid)]["memory_used_bytes"])
        for index, uuid in sorted(required_pairs)
    }
    if any(value > maximum for value in occupied.values()):
        raise PacketExecutorError(
            "selected GPU occupancy exceeds the frozen bound",
            code="packet_executor.gpu_occupancy_bound",
        )
    return {
        "phase": phase,
        "filesystem_path": str(contract["filesystem_path"]),
        "free_disk_bytes": free,
        "required_free_disk_bytes": required_free,
        "gpu_occupancy_bytes": occupied,
        "max_gpu_occupancy_bytes": maximum,
    }


def _preflight_gpu_mapping(
    contract: Mapping[str, Any],
    gpu_sampler: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = _normalize_gpu_rows(gpu_sampler())
    observed = {
        (row["physical_index"], row["gpu_uuid"])
        for row in rows
        if row["kind"] == "device"
    }
    expected = {
        (row["physical_index"], row["uuid"]) for row in contract["device_bindings"]
    }
    if not expected <= observed:
        raise PacketExecutorError(
            "selected physical-index to GPU-UUID inventory drifted",
            code="packet_executor.gpu_uuid_drift",
        )
    return {
        "phase": "before_marker",
        "physical_index_uuid_pairs": [
            {"physical_index": index, "uuid": uuid} for index, uuid in sorted(expected)
        ],
    }


def _validate_setup(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    paths = contract["setup_paths"]
    prepare_path: Path = paths["prepare_receipt_path"]
    if not prepare_path.is_file() or prepare_path.is_symlink():
        raise PacketExecutorError(
            "setup did not publish the prepare receipt",
            code="packet_executor.setup_prepare_missing",
        )
    prepare = _verify_signed(
        _strict_json_load(prepare_path),
        field="receipt_payload_sha256",
        code="packet_executor.setup_prepare_digest",
    )
    expected_prepare = {
        "schema": "coordexp-swift-reconcile-resume-probe-prepare-receipt-v1",
        "status": "prepared",
        "commit": manifest["implementation_commit"],
        "world_size": 2,
        "artifact_root": str(contract["artifact_root"]),
    }
    for field, expected in expected_prepare.items():
        if prepare.get(field) != expected:
            raise PacketExecutorError(
                f"prepare receipt {field} mismatched",
                code="packet_executor.setup_prepare_mismatch",
            )
    prepare_configs = _mapping(prepare.get("configs"), "prepare.configs")
    config_observations: dict[str, Any] = {}
    for role, expected in contract["configs"].items():
        path = Path(expected["path"])
        if not path.is_file() or path.is_symlink():
            raise PacketExecutorError(
                f"generated config is missing: {role}",
                code="packet_executor.setup_config_missing",
            )
        observed_sha = _sha256_file(path)
        prepare_row = _mapping(prepare_configs.get(role), f"prepare.configs.{role}")
        if (
            observed_sha != expected["expected_sha256"]
            or prepare_row.get("path") != str(path)
            or prepare_row.get("file_sha256") != observed_sha
            or not isinstance(prepare_row.get("resolved_config_fingerprint"), str)
        ):
            raise PacketExecutorError(
                f"generated config binding mismatched: {role}",
                code="packet_executor.setup_config_mismatch",
            )
        config_observations[role] = {
            "path": str(path),
            "sha256": observed_sha,
            "resolved_config_fingerprint": prepare_row["resolved_config_fingerprint"],
        }
    cache_root: Path = paths["pack_cache_root"]
    cache_receipt_path: Path = paths["pack_cache_receipt_path"]
    if (
        not cache_root.is_dir()
        or cache_root.is_symlink()
        or not cache_receipt_path.is_file()
    ):
        raise PacketExecutorError(
            "setup did not publish the private cache and receipt",
            code="packet_executor.setup_cache_missing",
        )
    cache_binding = _mapping(prepare.get("pack_cache"), "prepare.pack_cache")
    cache_file_sha = _sha256_file(cache_receipt_path)
    if (
        cache_binding.get("root") != str(cache_root)
        or cache_binding.get("status") != "prepared"
        or cache_binding.get("receipt_path") != str(cache_receipt_path)
        or cache_binding.get("receipt_sha256") != cache_file_sha
    ):
        raise PacketExecutorError(
            "private cache binding mismatched",
            code="packet_executor.setup_cache_mismatch",
        )
    cache_receipt = _verify_signed(
        _strict_json_load(cache_receipt_path),
        field="receipt_sha256",
        code="packet_executor.setup_cache_digest",
    )
    cache_result = _mapping(cache_receipt.get("result"), "cache.result")
    if cache_receipt.get("terminal_status") != "completed" or cache_result.get(
        "resolved_config_fingerprint"
    ) != cache_binding.get("resolved_config_fingerprint"):
        raise PacketExecutorError(
            "private cache receipt mismatched",
            code="packet_executor.setup_cache_mismatch",
        )
    return {
        "prepare_receipt_path": str(prepare_path),
        "prepare_receipt_sha256": _sha256_file(prepare_path),
        "configs": config_observations,
        "pack_cache_root": str(cache_root),
        "pack_cache_receipt_path": str(cache_receipt_path),
        "pack_cache_receipt_sha256": cache_file_sha,
    }


def _validate_inner(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    path: Path = contract["inner_path"]
    if not path.is_file() or path.is_symlink():
        raise PacketExecutorError(
            "verifier did not publish its inner terminal receipt",
            code="packet_executor.inner_missing",
        )
    receipt = _verify_signed(
        _strict_json_load(path),
        field="receipt_payload_sha256",
        code="packet_executor.inner_digest",
    )
    expected = {
        "schema": INNER_RECEIPT_SCHEMA,
        "status": "verified",
        "commit": manifest["implementation_commit"],
        "artifact_root": str(contract["artifact_root"]),
        "world_size": 2,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise PacketExecutorError(
                f"inner terminal receipt {field} mismatched",
                code="packet_executor.inner_mismatch",
            )
    if receipt.get("missing_inputs") or receipt.get("bounded_mismatches"):
        raise PacketExecutorError(
            "inner terminal receipt contains failed comparisons",
            code="packet_executor.inner_mismatch",
        )
    return {"path": str(path), "sha256": _sha256_file(path), **receipt}


def _error_record(exc: BaseException, *, command: str | None) -> dict[str, Any]:
    return {
        "code": str(getattr(exc, "code", "packet_executor.unexpected")),
        "command": command,
        "message": str(exc),
        "context": dict(getattr(exc, "context", {})),
    }


def _launcher_identity(launch: Callable[..., Any]) -> dict[str, Any]:
    target: Any = launch
    try:
        source = inspect.getsourcefile(target) or inspect.getfile(target)
    except (TypeError, OSError):
        target = type(launch)
        source = inspect.getsourcefile(target) or inspect.getfile(target)
    source_path = Path(source).resolve(strict=True)
    return {
        "module": str(getattr(launch, "__module__", type(launch).__module__)),
        "qualname": str(getattr(launch, "__qualname__", type(launch).__qualname__)),
        "source_realpath": str(source_path),
        "source_sha256": _sha256_file(source_path),
    }


def execute(
    *,
    manifest_path: Path,
    packet_path: Path,
    expected_manifest_sha256: str,
    expected_packet_sha256: str,
    attempt_marker_path: Path,
    terminal_receipt_path: Path,
    launch: Callable[..., Any] = _default_launch,
    gpu_sampler: Callable[[], list[dict[str, Any]]] = _default_gpu_sampler,
    process_sampler: Callable[[], list[dict[str, Any]]] = _default_process_sampler,
) -> dict[str, Any]:
    """Execute one validated packet attempt and return its durable outer receipt.

    Validation through target absence is pre-marker and raises on failure.  Once
    the marker is owned, every command/resource/verifier outcome is converted to
    ``status: stopped`` or ``status: verified`` and one immutable signed outer
    receipt is attempted.
    """

    manifest_path = Path(manifest_path)
    packet_path = Path(packet_path)
    attempt_marker_path = Path(attempt_marker_path)
    terminal_receipt_path = Path(terminal_receipt_path)
    (
        manifest,
        contract,
        manifest_sha,
        packet_sha,
        pre_cost_review,
        review_guard,
    ) = _pre_marker_validate(
        manifest_path=manifest_path,
        packet_path=packet_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_packet_sha256=expected_packet_sha256,
        attempt_marker_path=attempt_marker_path,
        terminal_receipt_path=terminal_receipt_path,
    )
    try:
        mapping_preflight = _preflight_gpu_mapping(contract, gpu_sampler)
        if not _review_path_matches(review_guard):
            raise PacketExecutorError(
                "independent pre-cost review was replaced before marker creation",
                code="packet_executor.review_replaced",
            )
        marker_body = {
            "schema": MARKER_SCHEMA,
            "created_at": _utc_now(),
            "implementation_commit": manifest["implementation_commit"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "packet_path": str(packet_path),
            "packet_sha256": packet_sha,
            "attempt_marker_path": str(attempt_marker_path),
            "terminal_receipt_path": str(terminal_receipt_path),
            "pre_cost_review_device": review_guard["device"],
            "pre_cost_review_inode": review_guard["inode"],
        }
        marker = _signed(marker_body)
        marker_bytes = _canonical_json_bytes(marker) + b"\n"
        _write_new_file(attempt_marker_path, marker_bytes)
        marker_sha = _sha256_bytes(marker_bytes)
        if not _review_path_matches(review_guard):
            try:
                if _sha256_file(attempt_marker_path) == marker_sha:
                    attempt_marker_path.unlink()
            except OSError:
                pass
            raise PacketExecutorError(
                "independent pre-cost review was replaced during marker creation",
                code="packet_executor.review_replaced",
            )
    finally:
        os.close(int(review_guard["descriptor"]))

    started_at = _utc_now()
    started_monotonic = time.monotonic()
    observations: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = [mapping_preflight]
    attempted_order: list[str] = []
    completed_order: list[str] = []
    setup_observation: dict[str, Any] | None = None
    inner_receipt: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    current_command: str | None = None
    initial_artifact_summary: dict[str, Any] | None = None
    final_artifact_summary: dict[str, Any] | None = None
    try:
        initial_artifact_summary = _summarize_artifact_tree(
            contract["artifact_summary"]
        )
        preflights.append(
            _preflight_resources(contract, gpu_sampler, phase="before_setup")
        )
        for name in COMMAND_ORDER:
            current_command = name
            attempted_order.append(name)
            observation = _run_command(
                name=name,
                argv=contract["commands"][name],
                cwd=Path(manifest["cwd"]),
                bound=contract["resource_bounds"][name],
                artifact_summary_bounds=contract["artifact_summary"],
                launch=launch,
                gpu_sampler=gpu_sampler,
                process_sampler=process_sampler,
                selected_uuids=set(contract["device_uuids"]),
            )
            observations.append(observation)
            if observation["execution_error"] is not None:
                error = observation["execution_error"]
                raise PacketExecutorError(
                    error["message"],
                    code=error["code"],
                    context=error["context"],
                )
            if observation["returncode"] != 0:
                raise PacketExecutorError(
                    f"{name} exited nonzero",
                    code="packet_executor.command_nonzero",
                    context={"returncode": observation["returncode"]},
                )
            if name == "setup":
                setup_observation = _validate_setup(manifest, contract)
                preflights.append(
                    _preflight_resources(
                        contract, gpu_sampler, phase="before_first_gpu_command"
                    )
                )
            if name == "verification":
                inner_receipt = _validate_inner(manifest, contract)
            completed_order.append(name)
            total_elapsed = time.monotonic() - started_monotonic
            current_summary = _summarize_artifact_tree(contract["artifact_summary"])
            total_new_bytes = max(
                0,
                current_summary["total_bytes"]
                - initial_artifact_summary["total_bytes"],
            )
            if total_elapsed > float(contract["total_bounds"]["wall_time_seconds"]):
                raise PacketExecutorError(
                    "packet exceeded its total wall-time bound",
                    code="packet_executor.total_wall_bound",
                )
            if total_new_bytes > int(
                contract["total_bounds"]["max_new_artifact_bytes"]
            ):
                raise PacketExecutorError(
                    "packet exceeded its total artifact-byte bound",
                    code="packet_executor.total_artifact_bound",
                )
    except BaseException as exc:
        failure = _error_record(exc, command=current_command)

    try:
        final_artifact_summary = _summarize_artifact_tree(contract["artifact_summary"])
    except BaseException as exc:
        if failure is None:
            failure = _error_record(exc, command=current_command)

    cpu_max: dict[str, int] = {}
    gpu_max: dict[str, int] = {}
    cpu_tree_max: dict[str, int] = {}
    for observation in observations:
        for rank, value in observation["cpu_rss_bytes_per_rank"].items():
            cpu_max[rank] = max(cpu_max.get(rank, 0), int(value))
        for rank, value in observation["gpu_memory_bytes_per_rank"].items():
            gpu_max[rank] = max(gpu_max.get(rank, 0), int(value))
        tree_value = observation["cpu_rss_command_tree_max_bytes"]
        if tree_value is not None:
            cpu_tree_max[observation["name"]] = int(tree_value)
    total_new_bytes = max(
        0,
        int((final_artifact_summary or {}).get("total_bytes", 0))
        - int((initial_artifact_summary or {}).get("total_bytes", 0)),
    )
    verified = (
        failure is None
        and attempted_order == list(COMMAND_ORDER)
        and completed_order == list(COMMAND_ORDER)
        and inner_receipt is not None
        and inner_receipt.get("status") == "verified"
    )
    stop_outcome = (
        {
            "code": "packet_executor.verified",
            "command": "verification",
            "message": "all six frozen commands and the signed inner verifier passed",
            "context": {},
        }
        if verified
        else failure
        or {
            "code": "packet_executor.incomplete",
            "command": current_command,
            "message": "packet did not complete its exact contract",
            "context": {},
        }
    )
    body = {
        "schema": OUTER_RECEIPT_SCHEMA,
        "status": "verified" if verified else "stopped",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_time_seconds": max(0.0, time.monotonic() - started_monotonic),
        "implementation_commit": manifest["implementation_commit"],
        "manifest_sha256": manifest_sha,
        "packet_sha256": packet_sha,
        "marker_sha256": marker_sha,
        "cwd": manifest["cwd"],
        "world_size": 2,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "packet": {"path": str(packet_path), "sha256": packet_sha},
        "pre_cost_review": pre_cost_review,
        "attempt_marker": {
            "path": str(attempt_marker_path),
            "sha256": marker_sha,
            "payload": marker,
        },
        "terminal_receipt_path": str(terminal_receipt_path),
        "attempted_order": attempted_order,
        "completed_order": completed_order,
        "argv_observations": [
            {
                "name": name,
                "argv": list(contract["commands"][name]),
                "argv_sha256": _sha256_bytes(
                    _canonical_json_bytes(list(contract["commands"][name]))
                ),
            }
            for name in attempted_order
        ],
        "commands": observations,
        "launcher_identity": _launcher_identity(launch),
        "runtime_identity": {
            "python_executable": sys.executable,
            "python_executable_realpath": os.path.realpath(sys.executable),
            "python_implementation": sys.implementation.name,
            "python_version": list(sys.version_info[:3]),
        },
        "command_process_groups": [
            {"name": row["name"], **(row["process_group"] or {})}
            for row in observations
        ],
        "cleanup_outcomes": [
            {"name": row["name"], "cleanup": row["cleanup"]} for row in observations
        ],
        "artifact_tree": {
            "initial": initial_artifact_summary,
            "final": final_artifact_summary,
        },
        "artifact_tree_summaries": {
            "initial": initial_artifact_summary,
            "final": final_artifact_summary,
        },
        "preflight_observations": preflights,
        "setup_validation": setup_observation,
        "resource_maxima": {
            "cpu_rss_bytes_per_rank": cpu_max,
            "cpu_rss_command_tree_max_bytes": cpu_tree_max,
            "gpu_memory_bytes_per_rank": gpu_max,
            "new_artifact_bytes": total_new_bytes,
        },
        "stop_outcome": stop_outcome,
        "inner_receipt": inner_receipt,
    }
    missing_outer_bindings = set(REQUIRED_OUTER_BINDINGS) - set(body)
    if missing_outer_bindings:
        raise PacketExecutorError(
            "outer receipt is missing required top-level bindings",
            code="packet_executor.outer_receipt_binding",
            context={"missing": sorted(missing_outer_bindings)},
        )
    receipt = _signed(body)
    _write_new_file(terminal_receipt_path, _canonical_json_bytes(receipt) + b"\n")
    return _strict_json_load(terminal_receipt_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--manifest", type=Path, required=True)
    execute_parser.add_argument("--packet", type=Path, required=True)
    execute_parser.add_argument("--manifest-sha256", required=True)
    execute_parser.add_argument("--packet-sha256", required=True)
    execute_parser.add_argument("--attempt-marker", type=Path, required=True)
    execute_parser.add_argument("--terminal-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        receipt = execute(
            manifest_path=args.manifest,
            packet_path=args.packet,
            expected_manifest_sha256=args.manifest_sha256,
            expected_packet_sha256=args.packet_sha256,
            attempt_marker_path=args.attempt_marker,
            terminal_receipt_path=args.terminal_receipt,
        )
    except BaseException as exc:
        print(
            f"{getattr(exc, 'code', 'packet_executor.unexpected')}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0 if receipt.get("status") == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
