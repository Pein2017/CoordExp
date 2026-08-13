from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / ".codex" / "serena" / "serena_worktree_mcp.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("serena_worktree_mcp", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_backend_command(tmp_path: Path, *, exit_early: bool = False) -> tuple[str, ...]:
    script = tmp_path / ("fake_backend_exit.py" if exit_early else "fake_backend.py")
    body = (
        "import sys\nsys.exit(23)\n"
        if exit_early
        else """
import os
import signal
import socket
import sys

count_path = sys.argv[1]
args = sys.argv[2:]
host = args[args.index("--host") + 1]
port = int(args[args.index("--port") + 1])
with open(count_path, "a", encoding="utf-8") as stream:
    stream.write(f"{os.getpid()}\\n")
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((host, port))
server.listen()
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    connection, _ = server.accept()
    connection.close()
"""
    )
    script.write_text(body)
    return (sys.executable, str(script), str(tmp_path / "starts.txt"))


def _stop_backend_process(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            waited = pid
        if waited == pid:
            return
        time.sleep(0.02)
    if Path(f"/proc/{pid}").exists():
        os.killpg(pid, signal.SIGKILL)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


def test_resolve_worktree_uses_exact_git_root(tmp_path: Path) -> None:
    subprocess.run(["/usr/bin/git", "init", "-q", str(tmp_path)], check=True)
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)

    module = _load_module()

    assert module.resolve_worktree(nested) == tmp_path.resolve()


def test_slot_records_full_root_and_stable_key(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    runtime_base = tmp_path / "runtime"
    expected_key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24]

    module = _load_module()
    slot = module.slot_for(root, runtime_base)

    assert slot.root == root.resolve()
    assert slot.key == expected_key
    assert slot.path == runtime_base / expected_key


def test_process_identity_reads_exact_project_and_rejects_pid_reuse(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "--project",
            str(root),
        ]
    )
    try:
        module = _load_module()
        identity = module.read_process_identity(process.pid)

        assert identity is not None
        assert identity.pid == process.pid
        assert identity.start_ticks > 0
        assert identity.executable == str(Path(sys.executable).resolve())
        assert identity.argv[-2:] == ("--project", str(root))
        assert identity.project_root == root.resolve()
        assert module.identity_matches(identity)
        assert not module.identity_matches(dataclasses.replace(identity, start_ticks=identity.start_ticks + 1))
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_process_identity_requires_exactly_one_project_argument(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "--project",
            str(root),
            "--project",
            str(root),
        ]
    )
    try:
        module = _load_module()
        assert module.read_process_identity(process.pid) is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_atomic_json_metadata_is_complete_and_private(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "slot" / "backend.json"

    module.write_json_atomic(target, {"root": "/repo", "pid": 123})

    assert json.loads(target.read_text()) == {"pid": 123, "root": "/repo"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert list(target.parent.glob("*.tmp")) == []


def test_lease_acquire_removes_dead_client_and_preserves_current(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=5)
    stale = slot.path / "clients" / f"{dead.pid}-1.json"
    module.write_json_atomic(
        stale,
        {"pid": dead.pid, "start_ticks": 1, "root": str(root.resolve())},
    )

    lease = module.SlotLease.acquire(slot)
    try:
        assert not stale.exists()
        assert lease.path.exists()
        payload = json.loads(lease.path.read_text())
        assert payload["pid"] == os.getpid()
        assert payload["root"] == str(root.resolve())
    finally:
        lease.release()

    assert not lease.path.exists()


def test_lease_rejects_slot_bound_to_another_root(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    module.write_json_atomic(slot.path / "root.json", {"root": str(tmp_path / "other")})

    with pytest.raises(module.RuntimeFailure, match="slot root mismatch"):
        module.SlotLease.acquire(slot)


def test_concurrent_clients_start_one_backend(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    config = module.RuntimeConfig(
        runtime_base=tmp_path / "runtime",
        serena_command=_fake_backend_command(tmp_path),
        startup_timeout=5,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        backends = list(executor.map(lambda _: module.ensure_backend(slot, config), range(2)))

    try:
        assert {backend.identity.pid for backend in backends} == {backends[0].identity.pid}
        assert (tmp_path / "starts.txt").read_text().splitlines() == [str(backends[0].identity.pid)]
        assert module.process_listens_on(backends[0].identity.pid, backends[0].port)
    finally:
        _stop_backend_process(backends[0].identity.pid)


def test_backend_exit_before_readiness_removes_attempt_metadata(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    config = module.RuntimeConfig(
        runtime_base=tmp_path / "runtime",
        serena_command=_fake_backend_command(tmp_path, exit_early=True),
        startup_timeout=2,
    )

    with pytest.raises(module.RuntimeFailure, match="backend exited before readiness"):
        module.ensure_backend(slot, config)

    assert not (slot.path / "backend.json").exists()


def test_existing_backend_refuses_port_owned_by_another_process(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "--project", str(root)],
        start_new_session=True,
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    try:
        identity = module.read_process_identity(sleeper.pid)
        assert identity is not None
        port = listener.getsockname()[1]
        module.write_json_atomic(
            slot.path / "backend.json",
            {
                "argv": list(identity.argv),
                "executable": identity.executable,
                "key": slot.key,
                "pid": identity.pid,
                "port": port,
                "project_root": str(root.resolve()),
                "root": str(root.resolve()),
                "start_ticks": identity.start_ticks,
            },
        )
        config = module.RuntimeConfig(runtime_base=tmp_path / "runtime")

        with pytest.raises(module.RuntimeFailure, match="port ownership mismatch"):
            module.ensure_backend(slot, config)

        assert sleeper.poll() is None
    finally:
        listener.close()
        _stop_backend_process(sleeper.pid)


def test_reap_slot_stops_backend_after_last_lease_grace(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    config = module.RuntimeConfig(
        runtime_base=tmp_path / "runtime",
        serena_command=_fake_backend_command(tmp_path),
        startup_timeout=5,
    )
    backend = module.ensure_backend(slot, config)
    lease = module.SlotLease.acquire(slot)
    lease.release()

    assert module.reap_slot(slot, grace_seconds=0.05)

    assert module.read_process_identity(backend.identity.pid) is None
    assert not (slot.path / "backend.json").exists()
    try:
        os.waitpid(backend.identity.pid, 0)
    except ChildProcessError:
        pass


def test_new_lease_during_grace_cancels_backend_retirement(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    config = module.RuntimeConfig(
        runtime_base=tmp_path / "runtime",
        serena_command=_fake_backend_command(tmp_path),
        startup_timeout=5,
    )
    backend = module.ensure_backend(slot, config)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(module.reap_slot, slot, 0.2)
        time.sleep(0.05)
        lease = module.SlotLease.acquire(slot)
        assert future.result(timeout=2) is False

    try:
        assert module.identity_matches(backend.identity)
    finally:
        lease.release()
        _stop_backend_process(backend.identity.pid)


def test_release_schedules_detached_reaper_that_stops_backend(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    config = module.RuntimeConfig(
        runtime_base=tmp_path / "runtime",
        serena_command=_fake_backend_command(tmp_path),
        startup_timeout=5,
    )
    backend = module.ensure_backend(slot, config)
    lease = module.SlotLease.acquire(slot)

    reaper_pid = module.release_and_schedule_reap(lease, grace_seconds=0.05)

    deadline = time.monotonic() + 5
    while module.read_process_identity(backend.identity.pid) is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    os.waitpid(reaper_pid, 0)
    assert module.read_process_identity(backend.identity.pid) is None
    assert not (slot.path / "backend.json").exists()
    try:
        os.waitpid(backend.identity.pid, 0)
    except ChildProcessError:
        pass


def test_doctor_reports_slot_without_starting_backend(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")

    status = module.doctor_slot(slot)

    assert status == {
        "backend": None,
        "client_count": 0,
        "key": slot.key,
        "root": str(root.resolve()),
        "slot": str(slot.path),
    }
    assert not (slot.path / "backend.json").exists()


def test_reaper_spawn_failure_preserves_live_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    root = tmp_path / "repo"
    root.mkdir()
    slot = module.slot_for(root, tmp_path / "runtime")
    lease = module.SlotLease.acquire(slot)

    def fail_to_spawn(*args, **kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(module.subprocess, "Popen", fail_to_spawn)
    with pytest.raises(OSError, match="injected spawn failure"):
        module.release_and_schedule_reap(lease, grace_seconds=60)

    assert lease.path.exists()
    lease.release()
    assert not lease.path.exists()
