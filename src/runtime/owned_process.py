"""Small lifecycle operations for explicitly owned POSIX child processes.

No resource allocation, queue, retry policy, experiment state or GPU defaults.
Use only handles to children started in their own session, never discovered PIDs.
The caller owns the returned log stream and must close it after observation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence, TextIO


def spawn_logged_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str],
) -> tuple[subprocess.Popen[Any], TextIO, float]:
    """Start one owned process group, closing its log when spawn fails."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("x")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
            start_new_session=True,
        )
    except BaseException:
        stream.close()
        raise
    return process, stream, started


def terminate_owned_process(
    process: subprocess.Popen[Any], *, grace_seconds: float = 30
) -> None:
    """Terminate and reap an owned session leader, never the caller's group."""
    if process.poll() is not None:
        return
    try:
        if os.getpgid(process.pid) != process.pid:
            raise ValueError("refusing to signal a child without its own process group")
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=grace_seconds)
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_seconds)


def wait_owned_process(process: subprocess.Popen[Any], *, deadline: float) -> int:
    """Observe exit by an absolute monotonic deadline; clean up on interruption.

    An already-exited child remains completed when observed late. This does not
    certify its historical finish time. A still-live overdue child is stopped.
    """
    try:
        code = process.poll()
        if code is not None:
            return code
        if not math.isfinite(deadline):
            raise ValueError("owned child requires a finite monotonic deadline")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, 0)
        return process.wait(timeout=remaining)
    except BaseException:
        terminate_owned_process(process)
        raise
