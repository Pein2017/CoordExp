"""Fail-closed launch preflight for the standalone refinement service.

Call :func:`run_launch_preflight` before creating a listening socket or starting
any background worker.  Its returned handle owns the runtime-root writer lock
until explicitly released (normally by a context manager).
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO


SUPPORTED_DEPENDENCIES: Mapping[str, str] = {
    "fastapi": "0.136.3",
    "uvicorn": "0.37.0",
    "starlette": "0.52.1",
}
WRITER_LOCK_NAME = ".writer.lock"
RUNTIME_RECEIPT_NAME = "runtime.json"

VersionResolver = Callable[[str], str]


class LaunchPreflightError(RuntimeError):
    """The local service cannot safely start with the resolved launch shape."""


class RuntimeRootBusyError(LaunchPreflightError):
    """Another process already owns the runtime-root writer lock."""


class UnsafeRuntimeRootLockError(LaunchPreflightError):
    """The writer-lock path is not one safe, private regular-file inode."""


@dataclass(frozen=True)
class LaunchPreflightReceipt:
    """JSON-safe attestation produced while the writer lock is held."""

    runtime_root: Path
    lock_path: Path
    dependency_versions: Mapping[str, str]
    reload: bool
    workers: int
    web_concurrency: int
    pid: int
    accepted_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "coco_refinement_launch_preflight",
            "status": "accepted",
            "runtime_root": str(self.runtime_root),
            "lock_path": str(self.lock_path),
            "dependencies": dict(self.dependency_versions),
            "process_shape": {
                "reload": self.reload,
                "workers": self.workers,
                "web_concurrency": self.web_concurrency,
                "pid": self.pid,
            },
            "accepted_at": self.accepted_at,
        }

    def to_json(self) -> str:
        """Return the canonical compact JSON representation of this receipt."""

        return json.dumps(
            self.as_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


class RuntimeRootLock:
    """Non-blocking, process-scoped exclusive lock for one runtime root."""

    def __init__(self, runtime_root: str | os.PathLike[str]) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.path = self.runtime_root / WRITER_LOCK_NAME
        self._handle: TextIO | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> RuntimeRootLock:
        """Acquire the sole-writer lock without waiting."""

        if self._handle is not None:
            return self
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        handle = _open_writer_lock(self.path)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeRootBusyError(
                f"runtime root already has a writer: {self.runtime_root}"
            ) from exc
        except BaseException:
            handle.close()
            raise

        try:
            _validate_writer_lock_inode(handle.fileno(), self.path)
            handle.seek(0)
            handle.truncate()
            json.dump(
                {
                    "kind": "coco_refinement_runtime_writer",
                    "pid": os.getpid(),
                },
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise
        self._handle = handle
        return self

    def release(self) -> None:
        """Release the lock idempotently while retaining its stable inode."""

        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> RuntimeRootLock:
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


@dataclass
class LaunchPreflight:
    """Accepted launch receipt plus the live sole-writer lock."""

    receipt: LaunchPreflightReceipt
    writer_lock: RuntimeRootLock
    receipt_path: Path

    def release(self) -> None:
        self.writer_lock.release()

    def __enter__(self) -> LaunchPreflight:
        if not self.writer_lock.acquired:
            raise LaunchPreflightError("launch preflight writer lock was already released")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def resolve_dependency_versions(
    version_resolver: VersionResolver | None = None,
) -> dict[str, str]:
    """Resolve and exactly validate the supported local HTTP dependency stack."""

    resolver = version_resolver or metadata.version
    resolved: dict[str, str] = {}
    for distribution, expected in SUPPORTED_DEPENDENCIES.items():
        try:
            actual = resolver(distribution)
        except (metadata.PackageNotFoundError, KeyError) as exc:
            raise LaunchPreflightError(
                f"required dependency is missing: {distribution}=={expected}"
            ) from exc
        if not isinstance(actual, str) or not actual:
            raise LaunchPreflightError(
                f"required dependency is missing: {distribution}=={expected}"
            )
        if actual != expected:
            raise LaunchPreflightError(
                f"unsupported {distribution} version: expected {expected}, got {actual}"
            )
        resolved[distribution] = actual
    return resolved


def validate_process_shape(
    *,
    reload: bool,
    workers: int,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Require one non-reloading process and return resolved WEB_CONCURRENCY."""

    if reload is not False:
        raise LaunchPreflightError("reload must be disabled")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers != 1:
        raise LaunchPreflightError("workers must equal 1")

    environ = os.environ if environment is None else environment
    raw_web_concurrency = environ.get("WEB_CONCURRENCY")
    if raw_web_concurrency is None:
        return 1
    try:
        web_concurrency = int(raw_web_concurrency)
    except (TypeError, ValueError) as exc:
        raise LaunchPreflightError("WEB_CONCURRENCY must equal 1") from exc
    if str(raw_web_concurrency).strip() != "1" or web_concurrency != 1:
        raise LaunchPreflightError("WEB_CONCURRENCY must equal 1")
    return web_concurrency


def run_launch_preflight(
    runtime_root: str | os.PathLike[str],
    *,
    reload: bool = False,
    workers: int = 1,
    environment: Mapping[str, str] | None = None,
    version_resolver: VersionResolver | None = None,
) -> LaunchPreflight:
    """Validate, lock, and attest one supported launch before bind/worker start."""

    dependency_versions = resolve_dependency_versions(version_resolver)
    web_concurrency = validate_process_shape(
        reload=reload,
        workers=workers,
        environment=environment,
    )
    writer_lock = RuntimeRootLock(runtime_root).acquire()
    try:
        receipt = LaunchPreflightReceipt(
            runtime_root=writer_lock.runtime_root,
            lock_path=writer_lock.path,
            dependency_versions=dependency_versions,
            reload=reload,
            workers=workers,
            web_concurrency=web_concurrency,
            pid=os.getpid(),
            accepted_at=datetime.now(UTC).isoformat(),
        )
        receipt_path = writer_lock.runtime_root / RUNTIME_RECEIPT_NAME
        _atomic_write_json(receipt_path, receipt.as_dict())
    except BaseException:
        writer_lock.release()
        raise
    return LaunchPreflight(
        receipt=receipt,
        writer_lock=writer_lock,
        receipt_path=receipt_path,
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _open_writer_lock(path: Path) -> TextIO:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        _validate_writer_lock_inode(descriptor, path)
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        descriptor = None
        return handle
    except UnsafeRuntimeRootLockError:
        raise
    except OSError as exc:
        raise UnsafeRuntimeRootLockError(
            f"unsafe writer lock path: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_writer_lock_inode(descriptor: int, path: Path) -> None:
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.lstat(path)
    except OSError as exc:
        raise UnsafeRuntimeRootLockError(
            f"unsafe writer lock path: {path}"
        ) from exc
    safe = (
        stat.S_ISREG(descriptor_stat.st_mode)
        and descriptor_stat.st_nlink == 1
        and path_stat.st_dev == descriptor_stat.st_dev
        and path_stat.st_ino == descriptor_stat.st_ino
    )
    if not safe:
        raise UnsafeRuntimeRootLockError(f"unsafe writer lock inode: {path}")


__all__ = [
    "LaunchPreflight",
    "LaunchPreflightError",
    "LaunchPreflightReceipt",
    "RUNTIME_RECEIPT_NAME",
    "RuntimeRootBusyError",
    "RuntimeRootLock",
    "SUPPORTED_DEPENDENCIES",
    "UnsafeRuntimeRootLockError",
    "WRITER_LOCK_NAME",
    "resolve_dependency_versions",
    "run_launch_preflight",
    "validate_process_shape",
]
