#!/usr/bin/env python3
"""Run one official Serena backend per Git worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import sys
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Iterator


class RuntimeFailure(RuntimeError):
    """Raised when the shared Serena runtime cannot satisfy its contract."""


@dataclass(frozen=True)
class Slot:
    root: Path
    key: str
    path: Path


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    executable: str
    argv: tuple[str, ...]
    project_root: Path


@dataclass(frozen=True)
class LeaseIdentity:
    pid: int
    start_ticks: int


@dataclass(frozen=True)
class RuntimeConfig:
    runtime_base: Path = Path("/data/CoordExp/.codex/serena/shared")
    serena_command: tuple[str, ...] = ("/root/.local/bin/serena",)
    context: str = "coordexp-minimal"
    startup_timeout: float = 60.0
    port_min: int = 23000
    port_count: int = 10000


@dataclass(frozen=True)
class Backend:
    slot: Slot
    identity: ProcessIdentity
    port: int


def resolve_worktree(start: Path) -> Path:
    """Return the canonical Git worktree root containing *start*."""
    result = subprocess.run(
        ["/usr/bin/git", "-C", os.fspath(start), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "not inside a Git worktree"
        raise RuntimeFailure(f"cannot resolve Git worktree from {start}: {detail}")
    root = Path(result.stdout.strip())
    if not root.is_absolute():
        raise RuntimeFailure(f"Git returned a non-absolute worktree root: {root}")
    return root.resolve(strict=True)


def slot_for(root: Path, runtime_base: Path) -> Slot:
    """Map a canonical worktree root to a collision-detectable runtime slot."""
    canonical_root = root.resolve(strict=True)
    key = hashlib.sha256(os.fsencode(canonical_root)).hexdigest()[:24]
    return Slot(root=canonical_root, key=key, path=runtime_base / key)


def read_process_identity(pid: int) -> ProcessIdentity | None:
    """Read a Serena-compatible process identity from Linux procfs."""
    if pid <= 0:
        return None
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text()
        command_end = stat_text.rfind(")")
        if command_end < 0:
            return None
        stat_fields = stat_text[command_end + 2 :].split()
        start_ticks = int(stat_fields[19])
        executable = os.path.realpath(os.readlink(proc / "exe"))
        argv = tuple(
            os.fsdecode(part)
            for part in (proc / "cmdline").read_bytes().split(b"\0")
            if part
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError, IndexError):
        return None

    project_indexes = [index for index, value in enumerate(argv) if value == "--project"]
    if len(project_indexes) != 1:
        return None
    project_index = project_indexes[0]
    if project_index + 1 >= len(argv):
        return None
    try:
        project_root = Path(argv[project_index + 1]).resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    return ProcessIdentity(
        pid=pid,
        start_ticks=start_ticks,
        executable=executable,
        argv=argv,
        project_root=project_root,
    )


def identity_matches(expected: ProcessIdentity) -> bool:
    """Return whether the exact process identity is still alive."""
    return read_process_identity(expected.pid) == expected


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Durably replace a small private JSON metadata file."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(temporary_fd, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(temporary_fd, encoded[offset:])
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_start_ticks(pid: int) -> int | None:
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text()
        command_end = stat_text.rfind(")")
        if command_end < 0:
            return None
        return int(stat_text[command_end + 2 :].split()[19])
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return None


@contextmanager
def _slot_lock(slot: Slot) -> Iterator[None]:
    slot.path.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = slot.path / "startup.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _port_allocation_lock(runtime_base: Path) -> Iterator[None]:
    runtime_base.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        runtime_base / "ports.lock",
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _bind_slot_root(slot: Slot) -> None:
    root_metadata = slot.path / "root.json"
    if root_metadata.exists():
        try:
            payload = json.loads(root_metadata.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeFailure(f"cannot read slot root metadata: {root_metadata}") from exc
        if payload != {"root": os.fspath(slot.root)}:
            raise RuntimeFailure(
                f"slot root mismatch for {slot.path}: expected {slot.root}, observed {payload.get('root')!r}"
            )
        return
    write_json_atomic(root_metadata, {"root": os.fspath(slot.root)})


def _reconcile_leases(slot: Slot) -> None:
    clients = slot.path / "clients"
    clients.mkdir(mode=0o700, exist_ok=True)
    for lease_path in clients.glob("*.json"):
        try:
            payload = json.loads(lease_path.read_text())
            pid = int(payload["pid"])
            start_ticks = int(payload["start_ticks"])
            root = payload["root"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            lease_path.unlink(missing_ok=True)
            continue
        if root != os.fspath(slot.root) or _read_start_ticks(pid) != start_ticks:
            lease_path.unlink(missing_ok=True)


@dataclass
class SlotLease:
    slot: Slot
    identity: LeaseIdentity
    path: Path
    _released: bool = False

    @classmethod
    def acquire(cls, slot: Slot) -> "SlotLease":
        start_ticks = _read_start_ticks(os.getpid())
        if start_ticks is None:
            raise RuntimeFailure("cannot read current wrapper process identity")
        identity = LeaseIdentity(pid=os.getpid(), start_ticks=start_ticks)
        path = slot.path / "clients" / f"{identity.pid}-{identity.start_ticks}.json"
        with _slot_lock(slot):
            _bind_slot_root(slot)
            _reconcile_leases(slot)
            write_json_atomic(
                path,
                {
                    "pid": identity.pid,
                    "root": os.fspath(slot.root),
                    "start_ticks": identity.start_ticks,
                },
            )
        return cls(slot=slot, identity=identity, path=path)

    def release(self) -> None:
        if self._released:
            return
        with _slot_lock(self.slot):
            self.path.unlink(missing_ok=True)
            _reconcile_leases(self.slot)
        self._released = True


def _proxy_free_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.upper().endswith("_PROXY")}


def _port_is_available(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _choose_port(slot: Slot, config: RuntimeConfig) -> int:
    if config.port_count <= 0:
        raise RuntimeFailure("port_count must be positive")
    start = int(slot.key[:8], 16) % config.port_count
    for offset in range(config.port_count):
        port = config.port_min + ((start + offset) % config.port_count)
        if _port_is_available(port):
            return port
    raise RuntimeFailure("no free loopback port is available for Serena")


def process_listens_on(pid: int, port: int) -> bool:
    """Return whether *pid* owns a loopback TCP listen socket on *port*."""
    socket_inodes: set[str] = set()
    try:
        for descriptor in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                socket_inodes.add(target[8:-1])
    except (FileNotFoundError, PermissionError, OSError):
        return False
    if not socket_inodes:
        return False
    wanted_port = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text().splitlines()[1:]
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            local_address = fields[1]
            state = fields[3]
            inode = fields[9]
            if local_address.rpartition(":")[2].upper() == wanted_port and state == "0A" and inode in socket_inodes:
                return True
    return False


def _backend_payload(backend: Backend) -> dict[str, object]:
    return {
        "argv": list(backend.identity.argv),
        "executable": backend.identity.executable,
        "key": backend.slot.key,
        "pid": backend.identity.pid,
        "port": backend.port,
        "project_root": os.fspath(backend.identity.project_root),
        "root": os.fspath(backend.slot.root),
        "start_ticks": backend.identity.start_ticks,
    }


def _read_backend(slot: Slot) -> Backend | None:
    metadata = slot.path / "backend.json"
    if not metadata.exists():
        return None
    try:
        payload = json.loads(metadata.read_text())
        identity = ProcessIdentity(
            pid=int(payload["pid"]),
            start_ticks=int(payload["start_ticks"]),
            executable=str(payload["executable"]),
            argv=tuple(str(value) for value in payload["argv"]),
            project_root=Path(str(payload["project_root"])),
        )
        backend = Backend(slot=slot, identity=identity, port=int(payload["port"]))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeFailure(f"cannot read backend metadata: {metadata}") from exc
    if payload.get("root") != os.fspath(slot.root) or payload.get("key") != slot.key:
        raise RuntimeFailure(f"backend metadata does not belong to slot {slot.path}")
    if not identity_matches(identity):
        metadata.unlink(missing_ok=True)
        return None
    if not process_listens_on(identity.pid, backend.port):
        raise RuntimeFailure(
            f"backend port ownership mismatch for PID {identity.pid} on 127.0.0.1:{backend.port}"
        )
    return backend


def _terminate_startup_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait(timeout=1)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def ensure_backend(slot: Slot, config: RuntimeConfig) -> Backend:
    """Return the exact live backend for *slot*, starting it once if absent."""
    with _slot_lock(slot):
        _bind_slot_root(slot)
        existing = _read_backend(slot)
        if existing is not None:
            return existing
        with _port_allocation_lock(config.runtime_base):
            port = _choose_port(slot, config)
            log_path = slot.path / "backend.log"
            if log_path.exists() and log_path.stat().st_size > 1_000_000:
                log_path.replace(slot.path / "backend.log.previous")
            log_descriptor = os.open(
                log_path,
                os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC,
                0o600,
            )
            argv = (
                *config.serena_command,
                "start-mcp-server",
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--project",
                os.fspath(slot.root),
                "--context",
                config.context,
                "--enable-web-dashboard",
                "false",
                "--enable-gui-log-window",
                "false",
                "--open-web-dashboard",
                "false",
                "--log-level",
                "CRITICAL",
            )
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log_descriptor,
                    stderr=log_descriptor,
                    env=_proxy_free_environment(),
                    start_new_session=True,
                )
            except OSError as exc:
                raise RuntimeFailure(f"cannot start official Serena: {exc}") from exc
            finally:
                os.close(log_descriptor)
            deadline = time.monotonic() + config.startup_timeout
            try:
                while time.monotonic() < deadline:
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeFailure(
                            f"backend exited before readiness with status {return_code}; log: {log_path}"
                        )
                    identity = read_process_identity(process.pid)
                    if (
                        identity is not None
                        and identity.project_root == slot.root
                        and process_listens_on(process.pid, port)
                    ):
                        backend = Backend(slot=slot, identity=identity, port=port)
                        write_json_atomic(slot.path / "backend.json", _backend_payload(backend))
                        return backend
                    time.sleep(0.05)
                raise RuntimeFailure(f"backend readiness timed out after {config.startup_timeout}s; log: {log_path}")
            except BaseException:
                _terminate_startup_process(process)
                (slot.path / "backend.json").unlink(missing_ok=True)
                raise


def reap_slot(slot: Slot, grace_seconds: float = 60.0) -> bool:
    """Retire an idle backend after grace; return whether one was stopped."""
    if grace_seconds < 0:
        raise RuntimeFailure("grace_seconds must not be negative")
    time.sleep(grace_seconds)
    with _slot_lock(slot):
        _bind_slot_root(slot)
        _reconcile_leases(slot)
        clients = slot.path / "clients"
        if any(clients.glob("*.json")):
            return False
        backend = _read_backend(slot)
        if backend is None:
            return False
        try:
            process_group = os.getpgid(backend.identity.pid)
        except ProcessLookupError:
            (slot.path / "backend.json").unlink(missing_ok=True)
            return False
        if process_group != backend.identity.pid:
            raise RuntimeFailure(
                f"refusing to stop backend PID {backend.identity.pid}: process group is {process_group}"
            )
        os.killpg(process_group, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while identity_matches(backend.identity) and time.monotonic() < deadline:
            time.sleep(0.05)
        if identity_matches(backend.identity):
            if os.getpgid(backend.identity.pid) != backend.identity.pid:
                raise RuntimeFailure("backend identity changed before SIGKILL")
            os.killpg(backend.identity.pid, signal.SIGKILL)
            kill_deadline = time.monotonic() + 5
            while identity_matches(backend.identity) and time.monotonic() < kill_deadline:
                time.sleep(0.05)
        if identity_matches(backend.identity):
            raise RuntimeFailure(f"backend PID {backend.identity.pid} did not exit")
        (slot.path / "backend.json").unlink(missing_ok=True)
        return True


def release_and_schedule_reap(lease: SlotLease, grace_seconds: float = 60.0) -> int:
    """Release *lease* and return the PID of a detached guarded reaper."""
    process = subprocess.Popen(
        [
            sys.executable,
            os.path.abspath(__file__),
            "reap",
            "--root",
            os.fspath(lease.slot.root),
            "--runtime-base",
            os.fspath(lease.slot.path.parent),
            "--grace-seconds",
            str(grace_seconds),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_proxy_free_environment(),
        start_new_session=True,
    )
    lease.release()
    return process.pid


def doctor_slot(slot: Slot) -> dict[str, object]:
    """Reconcile and report bounded slot status without starting a backend."""
    with _slot_lock(slot):
        _bind_slot_root(slot)
        _reconcile_leases(slot)
        backend = _read_backend(slot)
        client_count = sum(1 for _ in (slot.path / "clients").glob("*.json"))
        backend_status: dict[str, object] | None = None
        if backend is not None:
            backend_status = {
                "pid": backend.identity.pid,
                "port": backend.port,
                "start_ticks": backend.identity.start_ticks,
            }
        return {
            "backend": backend_status,
            "client_count": client_count,
            "key": slot.key,
            "root": os.fspath(slot.root),
            "slot": os.fspath(slot.path),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "reap"):
        command = subcommands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument(
            "--runtime-base",
            type=Path,
            default=RuntimeConfig.runtime_base,
        )
        if name == "reap":
            command.add_argument("--grace-seconds", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    slot = slot_for(args.root, args.runtime_base)
    if args.command == "doctor":
        print(json.dumps(doctor_slot(slot), sort_keys=True))
        return 0
    if args.command == "reap":
        reap_slot(slot, grace_seconds=args.grace_seconds)
        return 0
    raise RuntimeFailure(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeFailure as exc:
        print(f"serena-worktree-mcp: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
