from __future__ import annotations

import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time

import pytest

from probes.training_set_completion import coco227_recover_readback as recovery
from src.runtime.process_completion import next_process_completion, start_process_waiter


def _child(source: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", source],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_deployed_python_reproduces_missing_pidfd_open():
    assert not hasattr(os, "pidfd_open")


def test_real_child_wait_threads_report_success_and_failure_without_polling():
    completions: queue.Queue[dict[str, object]] = queue.Queue()
    processes = [
        _child("import time; time.sleep(0.05)"),
        _child("import time; time.sleep(0.10); raise SystemExit(7)"),
    ]
    for process in processes:
        start_process_waiter(process, completions, thread_name_prefix="test-readback-wait")
    observed = [
        next_process_completion(completions, deadline=time.time() + 3, clock=time.time)
        for _ in processes
    ]
    assert {row["pid"] for row in observed} == {process.pid for process in processes}
    assert {row["exit_code"] for row in observed} == {0, 7}
    assert all(row["wait_error"] is None for row in observed)


def test_real_child_deadline_fails_closed_and_can_be_reaped():
    completions: queue.Queue[dict[str, object]] = queue.Queue()
    process = _child("import time; time.sleep(30)")
    start_process_waiter(process, completions, thread_name_prefix="test-readback-wait")
    try:
        with pytest.raises(TimeoutError, match="original readback phase deadline"):
            next_process_completion(
                completions,
                deadline=time.time() + 0.05,
                clock=time.time,
                timeout_message="original readback phase deadline reached",
            )
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)


def test_original_reconciliation_identity_matches_live_process(tmp_path: Path):
    marker = str(tmp_path / "identity-marker")
    process = _child("import time; time.sleep(30); # " + marker)
    try:
        identity = recovery._process_identity(process.pid)
        # Popen can return while /proc still exposes the child's pre-exec state.
        # Wait for the test child to publish a command line, not an arbitrary delay.
        deadline = time.monotonic() + 2
        while (identity is None or not identity["cmdline"]) and time.monotonic() < deadline:
            time.sleep(0.005)
            identity = recovery._process_identity(process.pid)
        assert identity is not None
        assert identity["pid"] == process.pid
        assert marker in identity["cmdline"][-1]
        assert abs(identity["create_time"] - time.time()) < 3
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)


def test_recovery_contract_keeps_original_absolute_deadline_and_no_retraining():
    assert recovery.ENDPOINTS == tuple(
        (arm, step) for step in (8, 16, 32, 64, 128, 256) for arm in ("S", "T")
    )
    assert recovery.REQUEST_COUNT == 132
    assert recovery.MAX_LIVE_WORKERS == 8
    assert recovery.PHASE_SECONDS == 7200
