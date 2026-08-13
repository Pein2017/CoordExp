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
from typing import Any


MANIFEST_SCHEMA = "coordexp-swift-reconcile-resume-probe-command-manifest-v2"
MARKER_SCHEMA = "coordexp-swift-reconcile-resume-probe-attempt-marker-v1"
OUTER_RECEIPT_SCHEMA = (
    "coordexp-swift-reconcile-resume-probe-outer-terminal-receipt-v1"
)
INNER_RECEIPT_SCHEMA = "coordexp-swift-reconcile-resume-probe-terminal-receipt-v1"
COMMAND_ORDER = (
    "setup",
    "success.uninterrupted_control",
    "success.resumed_child",
    "rank_failure",
    "interruption",
    "verification",
)
MODEL_COMMANDS = frozenset(
    {"success.uninterrupted_control", "success.resumed_child"}
)
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


def _strict_json_load(path: Path) -> dict[str, Any]:
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
            path.read_text(encoding="utf-8"),
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
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
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
    if manifest.get("authorization_status") != "READY":
        raise PacketExecutorError(
            "frozen packet does not have READY authorization",
            code="packet_executor.authorization",
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
    if _absolute_path(bindings.get("packet_path"), "bindings.packet_path") != packet_path:
        raise PacketExecutorError("packet path binding drifted", code="packet_executor.binding")
    if bindings.get("packet_sha256") != packet_sha256:
        raise PacketExecutorError("packet digest binding drifted", code="packet_executor.binding")

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
    absent_paths = [_absolute_path(value, "targets.must_be_absent") for value in absent_values]
    if len(absent_paths) != len(set(absent_paths)):
        raise PacketExecutorError(
            "targets.must_be_absent contains duplicates", code="packet_executor.contract"
        )
    for required in (artifact_root, attempt_marker_path, terminal_receipt_path):
        if required not in absent_paths:
            raise PacketExecutorError(
                "required immutable target is not preflighted absent",
                code="packet_executor.contract",
            )

    preflight = _mapping(contract.get("preflight"), "execution_contract.preflight")
    filesystem_path = _absolute_path(preflight.get("filesystem_path"), "preflight.filesystem_path")
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
    device_uuids: list[str] = []
    for index, device in enumerate(devices):
        row = _mapping(device, f"gpu_devices[{index}]")
        _string(row.get("physical_index"), f"gpu_devices[{index}].physical_index")
        device_uuids.append(_string(row.get("uuid"), f"gpu_devices[{index}].uuid"))
    if len(set(device_uuids)) != 2:
        raise PacketExecutorError("GPU UUID bindings must be unique", code="packet_executor.contract")

    bounds = _mapping(contract.get("resource_bounds"), "execution_contract.resource_bounds")
    if set(bounds) != set(COMMAND_ORDER):
        raise PacketExecutorError(
            "resource bounds must name exactly the six commands",
            code="packet_executor.contract",
        )
    normalized_bounds: dict[str, dict[str, Any]] = {}
    for name in COMMAND_ORDER:
        row = _mapping(bounds[name], f"resource_bounds.{name}")
        required_ranks = row.get("required_ranks")
        expected_ranks = [0, 1] if name in MODEL_COMMANDS else []
        if required_ranks != expected_ranks:
            raise PacketExecutorError(
                f"resource_bounds.{name}.required_ranks drifted",
                code="packet_executor.contract",
            )
        normalized_bounds[name] = {
            "wall_time_seconds": _number(
                row.get("wall_time_seconds"), f"resource_bounds.{name}.wall_time_seconds"
            ),
            "max_cpu_rss_bytes_per_rank": _integer(
                row.get("max_cpu_rss_bytes_per_rank"),
                f"resource_bounds.{name}.max_cpu_rss_bytes_per_rank",
            ),
            "max_gpu_memory_bytes_per_rank": _integer(
                row.get("max_gpu_memory_bytes_per_rank"),
                f"resource_bounds.{name}.max_gpu_memory_bytes_per_rank",
            ),
            "max_new_artifact_bytes": _integer(
                row.get("max_new_artifact_bytes"),
                f"resource_bounds.{name}.max_new_artifact_bytes",
            ),
            "required_ranks": list(required_ranks),
        }
    total = _mapping(contract.get("total_bounds"), "execution_contract.total_bounds")
    normalized_total = {
        "wall_time_seconds": _number(
            total.get("wall_time_seconds"), "total_bounds.wall_time_seconds"
        ),
        "max_new_artifact_bytes": _integer(
            total.get("max_new_artifact_bytes"), "total_bounds.max_new_artifact_bytes"
        ),
    }

    setup = _mapping(contract.get("setup_validation"), "execution_contract.setup_validation")
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
        raise PacketExecutorError("inner receipt path drifted", code="packet_executor.contract")
    if inner.get("schema") != INNER_RECEIPT_SCHEMA or inner.get("required_status") != "verified":
        raise PacketExecutorError("inner receipt contract drifted", code="packet_executor.contract")

    commands = {name: _command_argv(manifest, name) for name in COMMAND_ORDER}
    configs = _flatten_config_files(manifest)
    return {
        **contract,
        "artifact_root": artifact_root,
        "absent_paths": absent_paths,
        "commands": commands,
        "configs": configs,
        "filesystem_path": filesystem_path,
        "device_uuids": device_uuids,
        "resource_bounds": normalized_bounds,
        "total_bounds": normalized_total,
        "setup_paths": setup_paths,
        "inner_path": inner_path,
    }


def _pre_marker_validate(
    *,
    manifest_path: Path,
    packet_path: Path,
    expected_manifest_sha256: str,
    expected_packet_sha256: str,
    attempt_marker_path: Path,
    terminal_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
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
    for index, target in enumerate(contract["absent_paths"]):
        _assert_absent(target, f"targets.must_be_absent[{index}]")
    return manifest, contract, manifest_sha, packet_sha


def _pid_starttime(pid: int) -> int | None:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
        tail = text[text.rfind(")") + 2 :].split()
        return int(tail[19])
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
            raise PacketExecutorError("invalid process sample", code="packet_executor.process_sample")
        pid = _integer(row.get("pid"), "process.pid", minimum=1)
        starttime = _integer(row.get("starttime"), "process.starttime", minimum=1)
        rank = row.get("rank")
        if rank is not None:
            rank = _integer(rank, "process.rank")
        rss = _integer(row.get("rss_bytes"), "process.rss_bytes")
        result.append({"pid": pid, "starttime": starttime, "rank": rank, "rss_bytes": rss})
    return result


def _normalize_gpu_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise PacketExecutorError(
            "GPU sampler must return a list", code="packet_executor.gpu_sample"
        )
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PacketExecutorError("invalid GPU sample", code="packet_executor.gpu_sample")
        kind = row.get("kind")
        uuid = _string(row.get("gpu_uuid"), "gpu.gpu_uuid")
        if kind == "device":
            result.append(
                {
                    "kind": "device",
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
            raise PacketExecutorError("invalid GPU sample kind", code="packet_executor.gpu_sample")
    return result


def _tree_bytes(paths: Sequence[Path]) -> int:
    total = 0
    for root in paths:
        info = _lstat(root)
        if info is None:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
            continue
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            continue
        for directory, names, filenames in os.walk(root, followlinks=False):
            del names
            for filename in filenames:
                path = Path(directory) / filename
                child = _lstat(path)
                if child is not None and stat.S_ISREG(child.st_mode):
                    total += child.st_size
    return total


def _sample_resources(
    process_sampler: Callable[[], list[dict[str, Any]]],
    gpu_sampler: Callable[[], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        _normalize_process_rows(process_sampler()),
        _normalize_gpu_rows(gpu_sampler()),
    )


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


def _rank_maxima(
    rows: Sequence[Mapping[str, Any]], value_field: str
) -> dict[str, int]:
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
) -> tuple[dict[str, int], dict[str, int]]:
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
    cpu = _rank_maxima(process_rows, "rss_bytes")
    gpu = _rank_maxima(gpu_rows, "gpu_memory_bytes")
    required = {str(rank) for rank in bound["required_ranks"]}
    # Model-free commands have no required rank inventory.  Their sampled
    # host/device rows remain in the receipt, but unrelated shared-host ranks
    # are not attributed to that command for per-rank limit enforcement.
    if not required:
        cpu = {}
        gpu = {}
    missing_cpu = sorted(required - set(cpu))
    if missing_cpu:
        raise PacketExecutorError(
            f"{name} is missing CPU RSS measurements for ranks {missing_cpu}",
            code="packet_executor.missing_process_rank",
        )
    missing_gpu = sorted(required - set(gpu))
    if missing_gpu:
        raise PacketExecutorError(
            f"{name} is missing GPU measurements for ranks {missing_gpu}",
            code="packet_executor.missing_gpu_rank",
        )
    if any(value > int(bound["max_cpu_rss_bytes_per_rank"]) for value in cpu.values()):
        raise PacketExecutorError(
            f"{name} exceeded its CPU RSS bound", code="packet_executor.rss_bound"
        )
    if any(value > int(bound["max_gpu_memory_bytes_per_rank"]) for value in gpu.values()):
        raise PacketExecutorError(
            f"{name} exceeded its GPU-memory bound", code="packet_executor.gpu_bound"
        )
    return cpu, gpu


def _terminate_process(process: Any) -> None:
    try:
        pgid = os.getpgid(int(process.pid))
        os.killpg(pgid, signal.SIGTERM)
    except (AttributeError, OSError, ValueError):
        try:
            process.terminate()
        except (AttributeError, OSError):
            return
    try:
        process.wait(timeout=5)
    except (AttributeError, subprocess.TimeoutExpired):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (UnboundLocalError, OSError):
            try:
                process.kill()
            except (AttributeError, OSError):
                pass
        try:
            process.wait(timeout=5)
        except (AttributeError, subprocess.TimeoutExpired):
            pass


def _run_command(
    *,
    name: str,
    argv: Sequence[str],
    cwd: Path,
    bound: Mapping[str, Any],
    artifact_paths: Sequence[Path],
    launch: Callable[..., Any],
    gpu_sampler: Callable[[], list[dict[str, Any]]],
    process_sampler: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_before = _tree_bytes(artifact_paths)
    process_observations: dict[tuple[int, int], dict[str, Any]] = {}
    gpu_observations: dict[tuple[str, int, int], dict[str, Any]] = {}

    process_rows, gpu_rows = _sample_resources(process_sampler, gpu_sampler)
    _merge_process_samples(process_observations, process_rows)
    _merge_gpu_samples(gpu_observations, gpu_rows)
    launched: Any = None
    launch_error: BaseException | None = None
    try:
        launched = launch(list(argv), cwd=cwd)
    except BaseException as exc:
        launch_error = exc
    returncode: int | None = None
    if launch_error is not None:
        returncode = None
    elif hasattr(launched, "poll") and callable(launched.poll):
        while True:
            returncode = launched.poll()
            process_rows, gpu_rows = _sample_resources(process_sampler, gpu_sampler)
            _merge_process_samples(process_observations, process_rows)
            _merge_gpu_samples(gpu_observations, gpu_rows)
            elapsed = time.monotonic() - started
            if elapsed > float(bound["wall_time_seconds"]):
                _terminate_process(launched)
                returncode = launched.poll()
                break
            if returncode is not None:
                break
            time.sleep(0.25)
    elif isinstance(launched, int):
        returncode = launched
    elif isinstance(launched, Mapping):
        value = launched.get("returncode")
        returncode = int(value) if isinstance(value, int) else None
    else:
        value = getattr(launched, "returncode", None)
        returncode = int(value) if isinstance(value, int) else None
    process_rows, gpu_rows = _sample_resources(process_sampler, gpu_sampler)
    _merge_process_samples(process_observations, process_rows)
    _merge_gpu_samples(gpu_observations, gpu_rows)
    elapsed = max(0.0, time.monotonic() - started)
    artifact_after = _tree_bytes(artifact_paths)
    new_bytes = max(0, artifact_after - artifact_before)
    required = {str(rank) for rank in bound["required_ranks"]}
    cpu = _rank_maxima(list(process_observations.values()), "rss_bytes")
    gpu = _rank_maxima(list(gpu_observations.values()), "gpu_memory_bytes")
    if not required:
        cpu = {}
        gpu = {}
    resource_error: dict[str, Any] | None = None
    try:
        cpu, gpu = _check_command_resources(
            name=name,
            bound=bound,
            elapsed=elapsed,
            new_artifact_bytes=new_bytes,
            process_rows=list(process_observations.values()),
            gpu_rows=list(gpu_observations.values()),
        )
    except BaseException as exc:
        resource_error = _error_record(exc, command=name)
    observation = {
        "name": name,
        "argv": list(argv),
        "argv_sha256": _sha256_bytes(_canonical_json_bytes(list(argv))),
        "attempt": 1,
        "returncode": returncode,
        "wall_time_seconds": elapsed,
        "new_artifact_bytes": new_bytes,
        "cpu_rss_bytes_per_rank": cpu,
        "gpu_memory_bytes_per_rank": gpu,
        "process_observations": sorted(
            process_observations.values(), key=lambda row: (row["pid"], row["starttime"])
        ),
        "gpu_observations": sorted(
            gpu_observations.values(),
            key=lambda row: (row["gpu_uuid"], row["pid"], row["starttime"]),
        ),
        "launch_error": None
        if launch_error is None
        else _error_record(launch_error, command=name),
        "resource_error": resource_error,
    }
    return observation


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
    device_rows = {row["gpu_uuid"]: row for row in rows if row["kind"] == "device"}
    required = set(contract["device_uuids"])
    if set(device_rows) & required != required:
        raise PacketExecutorError(
            "selected GPU UUID inventory drifted",
            code="packet_executor.gpu_uuid_drift",
        )
    maximum = int(_mapping(contract["preflight"], "preflight")["max_gpu_occupancy_bytes"])
    occupied = {
        uuid: int(device_rows[uuid]["memory_used_bytes"]) for uuid in sorted(required)
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
    if not cache_root.is_dir() or cache_root.is_symlink() or not cache_receipt_path.is_file():
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
    if (
        cache_receipt.get("terminal_status") != "completed"
        or cache_result.get("resolved_config_fingerprint")
        != cache_binding.get("resolved_config_fingerprint")
    ):
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
    manifest, contract, manifest_sha, packet_sha = _pre_marker_validate(
        manifest_path=manifest_path,
        packet_path=packet_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_packet_sha256=expected_packet_sha256,
        attempt_marker_path=attempt_marker_path,
        terminal_receipt_path=terminal_receipt_path,
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
    }
    marker = _signed(marker_body)
    _write_new_file(attempt_marker_path, _canonical_json_bytes(marker) + b"\n")
    marker_sha = _sha256_file(attempt_marker_path)

    started_at = _utc_now()
    started_monotonic = time.monotonic()
    observations: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = []
    attempted_order: list[str] = []
    completed_order: list[str] = []
    setup_observation: dict[str, Any] | None = None
    inner_receipt: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    current_command: str | None = None
    artifact_paths = [
        contract["artifact_root"],
        contract["setup_paths"]["pack_cache_root"],
        contract["setup_paths"]["pack_cache_receipt_path"],
    ]
    initial_artifact_bytes = _tree_bytes(artifact_paths)
    try:
        preflights.append(_preflight_resources(contract, gpu_sampler, phase="before_setup"))
        for name in COMMAND_ORDER:
            current_command = name
            attempted_order.append(name)
            observation = _run_command(
                name=name,
                argv=contract["commands"][name],
                cwd=Path(manifest["cwd"]),
                bound=contract["resource_bounds"][name],
                artifact_paths=artifact_paths,
                launch=launch,
                gpu_sampler=gpu_sampler,
                process_sampler=process_sampler,
            )
            observations.append(observation)
            if observation["launch_error"] is not None:
                error = observation["launch_error"]
                raise PacketExecutorError(
                    f"{name} launch raised: {error['message']}",
                    code="packet_executor.launch_exception",
                    context={"error_code": error["code"]},
                )
            if observation["resource_error"] is not None:
                error = observation["resource_error"]
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
                    _preflight_resources(contract, gpu_sampler, phase="before_first_gpu_command")
                )
            if name == "verification":
                inner_receipt = _validate_inner(manifest, contract)
            completed_order.append(name)
            total_elapsed = time.monotonic() - started_monotonic
            total_new_bytes = max(0, _tree_bytes(artifact_paths) - initial_artifact_bytes)
            if total_elapsed > float(contract["total_bounds"]["wall_time_seconds"]):
                raise PacketExecutorError(
                    "packet exceeded its total wall-time bound",
                    code="packet_executor.total_wall_bound",
                )
            if total_new_bytes > int(contract["total_bounds"]["max_new_artifact_bytes"]):
                raise PacketExecutorError(
                    "packet exceeded its total artifact-byte bound",
                    code="packet_executor.total_artifact_bound",
                )
    except BaseException as exc:
        failure = _error_record(exc, command=current_command)

    cpu_max: dict[str, int] = {}
    gpu_max: dict[str, int] = {}
    for observation in observations:
        for rank, value in observation["cpu_rss_bytes_per_rank"].items():
            cpu_max[rank] = max(cpu_max.get(rank, 0), int(value))
        for rank, value in observation["gpu_memory_bytes_per_rank"].items():
            gpu_max[rank] = max(gpu_max.get(rank, 0), int(value))
    total_new_bytes = max(0, _tree_bytes(artifact_paths) - initial_artifact_bytes)
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
        "cwd": manifest["cwd"],
        "world_size": 2,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "packet": {"path": str(packet_path), "sha256": packet_sha},
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
        "preflight_observations": preflights,
        "setup_validation": setup_observation,
        "resource_maxima": {
            "cpu_rss_bytes_per_rank": cpu_max,
            "gpu_memory_bytes_per_rank": gpu_max,
            "new_artifact_bytes": total_new_bytes,
        },
        "stop_outcome": stop_outcome,
        "inner_receipt": inner_receipt,
    }
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
        print(f"{getattr(exc, 'code', 'packet_executor.unexpected')}: {exc}", file=sys.stderr)
        return 1
    return 0 if receipt.get("status") == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
