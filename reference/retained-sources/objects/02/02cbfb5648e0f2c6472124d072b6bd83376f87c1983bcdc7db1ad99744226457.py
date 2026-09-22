#!/usr/bin/env python3
"""Single-use owner for the explicitly granted corrected retry."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-checkpoint-history-cross")
ROOT = BASE / "retry-v2"
HERE = ROOT / "launcher"
PACKET = BASE / "preparation-v2" / "packet.json"
RUNNER = Path("/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-11-checkpoint-history-cross/run_cross.py")
PACKET_SHA = "7459968bef4da5d89c3f75d5e63cf50d2fbd7c89c64efc00e85df776678821f2"
RUNNER_SHA = "344317d4dfa66fec9e864caea95f4abc45ec1857a006e8168184b98498446e88"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def proc_identity(pid: int) -> dict[str, object]:
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    start_ticks = int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    return {"pid": pid, "cmdline": [part.decode() for part in cmdline], "start_ticks": start_ticks}


def main() -> int:
    if sha(PACKET) != PACKET_SHA or sha(RUNNER) != RUNNER_SHA:
        raise RuntimeError("corrected grant identity differs")
    if any((ROOT / f"worker-{worker}").exists() for worker in range(8)):
        raise RuntimeError("occupied corrected worker target")
    if (HERE / "group-launch.json").exists() or (HERE / "group-completion.json").exists():
        raise RuntimeError("corrected launcher already used")
    started = time.time()
    processes = []
    launch_rows = []
    for worker in range(8):
        command = ["timeout", "--signal=TERM", "--kill-after=30s", "1530s", "python", str(RUNNER),
                   "execute", "--packet", str(PACKET), "--out-root", str(ROOT), "--worker", str(worker)]
        log = (HERE / f"worker-{worker}.log").open("xb")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker)), start_new_session=True)
        identity = proc_identity(process.pid)
        if identity["cmdline"] != command:
            process.terminate()
            raise RuntimeError(f"worker {worker} cmdline differs")
        launch_rows.append({"worker": worker, "cuda_visible_devices": str(worker), "command": command, **identity})
        processes.append((worker, process, log, command))
    publish(HERE / "group-launch.json", {"schema": "checkpoint_history_cross.retry_v2_group_launch.v1",
            "status": "launched", "launcher_pid": os.getpid(), "started_unix": started,
            "packet_sha256": PACKET_SHA, "runner_sha256": RUNNER_SHA, "workers": launch_rows})
    exits = []
    for worker, process, log, command in processes:
        returncode = process.wait()
        log.close()
        row = {"schema": "checkpoint_history_cross.retry_v2_outer_exit.v1", "worker": worker,
               "pid": process.pid, "returncode": returncode, "command": command, "finished_unix": time.time()}
        publish(HERE / f"worker-{worker}-outer-exit.json", row)
        exits.append(row)
    complete = {"schema": "checkpoint_history_cross.retry_v2_group_completion.v1",
                "status": "completed" if all(row["returncode"] == 0 for row in exits) else "failed",
                "all_outer_exit_zero": all(row["returncode"] == 0 for row in exits),
                "started_unix": started, "finished_unix": time.time(), "elapsed_seconds": time.time() - started,
                "returncodes": [row["returncode"] for row in exits]}
    publish(HERE / "group-completion.json", complete)
    print(json.dumps(complete, sort_keys=True))
    return 0 if complete["all_outer_exit_zero"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
