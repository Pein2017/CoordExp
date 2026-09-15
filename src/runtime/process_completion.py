"""Portable completion waiting for explicitly owned child processes."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any


def start_process_waiter(
    process: subprocess.Popen[Any],
    completions: queue.Queue[dict[str, Any]],
    *,
    thread_name_prefix: str = "owned-process-wait",
) -> threading.Thread:
    """Wait for one already-owned child without polling or platform-specific pidfds."""

    def wait() -> None:
        try:
            code: int | str = process.wait()
            error = None
        except BaseException as exc:
            code = "wait_error"
            error = f"{type(exc).__name__}: {exc}"
        completions.put(
            {
                "pid": process.pid,
                "exit_code": code,
                "wait_error": error,
                "completed_at_monotonic": time.monotonic(),
                "completed_at_unix": time.time(),
            }
        )

    thread = threading.Thread(
        target=wait,
        name=f"{thread_name_prefix}-{process.pid}",
        daemon=True,
    )
    thread.start()
    return thread


def next_process_completion(
    completions: queue.Queue[dict[str, Any]],
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
    timeout_message: str = "owned process deadline reached",
) -> dict[str, Any]:
    """Return the next owned-child completion before the caller's absolute deadline."""

    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError(timeout_message)
    try:
        return completions.get(timeout=remaining)
    except queue.Empty as exc:
        raise TimeoutError(timeout_message) from exc
