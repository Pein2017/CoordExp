#!/usr/bin/env python3
"""Single-use concurrent owner for the granted eight-worker cross."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-checkpoint-history-cross")
PACKET = ROOT / "preparation" / "packet.json"
RUNNER = Path("/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-11-checkpoint-history-cross/run_cross.py")
HERE = ROOT / "launcher"
EXPECTED_PACKET = "e41059f2150b125a2eeb3f1f925b82e59c8d50a1d85d75d931d3cce1e79d2cdf"
EXPECTED_RUNNER = "c744e928f49379a0dacaa97b943d1bb015313abe9b0dd6286e38e9633b9d0357"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid: int) -> dict[str, object]:
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    return {"pid": pid, "cmdline": [part.decode() for part in cmdline], "start_ticks": int(stat[21])}


def main() -> int:
    if sha(PACKET) != EXPECTED_PACKET or sha(RUNNER) != EXPECTED_RUNNER:
        raise RuntimeError("granted packet/runner identity differs")
    if any((ROOT / f"worker-{worker}").exists() for worker in range(8)):
        raise RuntimeError("occupied worker target")
    if (HERE / "group-launch.json").exists() or (HERE / "group-completion.json").exists():
        raise RuntimeError("launcher already used")
    started = time.time()
    processes: list[tuple[int, subprocess.Popen[bytes], object, list[str]]] = []
    launch_rows = []
    for worker in range(8):
        command = ["timeout", "--signal=TERM", "--kill-after=30s", "1530s", "python", str(RUNNER),
                   "execute", "--packet", str(PACKET), "--out-root", str(ROOT), "--worker", str(worker)]
        log = (HERE / f"worker-{worker}.log").open("xb")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker))
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                   start_new_session=True)
        observed = identity(process.pid)
        expected = command
        if observed["cmdline"] != expected:
            process.terminate()
            raise RuntimeError(f"worker {worker} outer cmdline differs")
        row = {"worker": worker, "cuda_visible_devices": str(worker), "command": command, **observed}
        launch_rows.append(row)
        processes.append((worker, process, log, command))
    publish(HERE / "group-launch.json", {"schema": "checkpoint_history_cross.group_launch.v1",
            "status": "launched", "started_unix": started, "launcher_pid": os.getpid(),
            "packet_sha256": EXPECTED_PACKET, "runner_sha256": EXPECTED_RUNNER, "workers": launch_rows})
    exits = []
    for worker, process, log, command in processes:
        returncode = process.wait()
        log.close()
        receipt = {"schema": "checkpoint_history_cross.outer_exit.v1", "worker": worker,
                   "pid": process.pid, "returncode": returncode, "command": command,
                   "finished_unix": time.time()}
        publish(HERE / f"worker-{worker}-outer-exit.json", receipt)
        exits.append(receipt)
    completed = {"schema": "checkpoint_history_cross.group_completion.v1",
                 "status": "completed" if all(row["returncode"] == 0 for row in exits) else "failed",
                 "all_outer_exit_zero": all(row["returncode"] == 0 for row in exits),
                 "started_unix": started, "finished_unix": time.time(),
                 "elapsed_seconds": time.time() - started,
                 "returncodes": [row["returncode"] for row in exits]}
    publish(HERE / "group-completion.json", completed)
    print(json.dumps(completed, sort_keys=True))
    return 0 if completed["all_outer_exit_zero"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
