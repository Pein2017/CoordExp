from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.runtime import owned_process as owned


def test_spawn_logs_exit_and_literal_environment(tmp_path):
    log = tmp_path / "child.log"
    process, stream, started = owned.spawn_logged_process(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['PROBE_TEST']); raise SystemExit(7)",
        ],
        cwd=tmp_path,
        log_path=log,
        env={"PROBE_TEST": "literal value"},
    )
    try:
        assert owned.wait_owned_process(process, deadline=started + 10) == 7
    finally:
        owned.terminate_owned_process(process)
        stream.close()
    assert log.read_text() == "literal value\n"
    with pytest.raises(FileExistsError):
        owned.spawn_logged_process(
            [sys.executable, "-c", "pass"], cwd=tmp_path, log_path=log, env={}
        )
    assert log.read_text() == "literal value\n"


def test_spawn_failure_closes_open_log(tmp_path, monkeypatch):
    streams = []
    original = Path.open

    def tracked_open(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        streams.append(stream)
        return stream

    def fail_spawn(*args, **kwargs):
        raise FileNotFoundError("missing executable")

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(owned.subprocess, "Popen", fail_spawn)
    with pytest.raises(FileNotFoundError, match="missing executable"):
        owned.spawn_logged_process(
            ["missing"], cwd=tmp_path, log_path=tmp_path / "failure.log", env={}
        )
    assert len(streams) == 1 and streams[0].closed


def test_live_child_is_reaped_when_deadline_expired_before_wait(tmp_path):
    process, stream, _ = owned.spawn_logged_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        log_path=tmp_path / "late.log",
        env={},
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            owned.wait_owned_process(process, deadline=time.monotonic() - 1)
        assert process.poll() is not None
    finally:
        owned.terminate_owned_process(process)
        stream.close()


def test_already_completed_child_is_accepted_when_observed_late(tmp_path):
    process, stream, _ = owned.spawn_logged_process(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        log_path=tmp_path / "completed.log",
        env={},
    )
    try:
        assert process.wait(timeout=10) == 0
        assert owned.wait_owned_process(process, deadline=time.monotonic() - 1) == 0
    finally:
        owned.terminate_owned_process(process)
        stream.close()


def test_interrupted_wait_cleans_up_the_exact_child(monkeypatch):
    class Child:
        args = ["test-only"]

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise KeyboardInterrupt

    child = Child()
    stopped = []
    monkeypatch.setattr(owned, "terminate_owned_process", stopped.append)
    with pytest.raises(KeyboardInterrupt):
        owned.wait_owned_process(child, deadline=time.monotonic() + 10)
    assert stopped == [child]


def test_termination_rejects_an_unowned_group_without_signalling(monkeypatch):
    class Child:
        pid = 123

        def poll(self):
            return None

    signals = []
    monkeypatch.setattr(owned.os, "getpgid", lambda pid: 456)
    monkeypatch.setattr(owned.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(ValueError, match="own process group"):
        owned.terminate_owned_process(Child())
    assert signals == []
