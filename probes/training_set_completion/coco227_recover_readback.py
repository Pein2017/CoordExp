"""Recover only the interrupted COCO227 saved-checkpoint readback phase.

The original paired training is immutable and already complete.  This scoped
controller reuses the frozen endpoint worker/collector, retains valid rows, and
waits for owned children with blocking ``Popen.wait`` threads plus a completion
queue.  It has no ``pidfd`` dependency and never polls child state.
"""

from __future__ import annotations

from src.runtime.owned_process import terminate_owned_process

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback
from typing import Any, Mapping

from probes.training_set_completion import coco227_trial as frozen
from probes.training_set_completion import training
from src.runtime.process_completion import next_process_completion, start_process_waiter


SCHEMA = "training_set_completion.coco227_readback_recovery.v1"
REPO = Path(__file__).resolve().parents[2]
ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-15-coco227-ce-normalization"
)
TRIAL = ROOT / "trial-v1/trial.json"
RECOVERY_ROOT = ROOT / "readback-recovery-v1"
ORIGINAL_FAILURE = ROOT / "trial-v1/controller-failures/attempt-001.json"
ORIGINAL_RECONCILIATION = RECOVERY_ROOT / "original-worker-reconciliation.json"
MECHANICS_REPRODUCTION = RECOVERY_ROOT / "mechanics-reproduction.json"
TMUX_SESSION = "coordexp-coco227-readback-recovery"
ENDPOINTS = tuple(
    (arm, step) for step in frozen.CHECKPOINT_STEPS for arm in frozen.ARMS
)
MAX_LIVE_WORKERS = 8
REQUEST_COUNT = 132
PHASE_SECONDS = 7_200


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _verify(binding: Mapping[str, Any], label: str) -> None:
    require(training.binding(binding["path"]) == dict(binding), f"{label} changed")


def _process_identity(pid: int) -> dict[str, Any] | None:
    """Read one Linux process identity without waiting or repeated probes."""
    proc = Path("/proc") / str(pid)
    try:
        cmdline = [
            os.fsdecode(item)
            for item in (proc / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        stat = (proc / "stat").read_text()
        after_name = stat[stat.rfind(")") + 2 :].split()
        start_ticks = int(after_name[19])
        boot_time = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {
        "pid": pid,
        "cmdline": cmdline,
        "create_time": boot_time + start_ticks / os.sysconf("SC_CLK_TCK"),
    }


def _live_trial_endpoint_processes(
    *, trial_path: Path, readback_output: Path
) -> list[dict[str, Any]]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = _process_identity(int(entry.name))
        if identity is None:
            continue
        command = identity["cmdline"]
        if (
            "probes.training_set_completion.coco227_readback" in command
            and "endpoint-worker" in command
            and "--trial" in command
            and command[command.index("--trial") + 1] == str(trial_path)
            and "--output" in command
            and command[command.index("--output") + 1] == str(readback_output)
        ):
            result.append(identity)
    return result


def _checkpoint_adapter(
    *, trial: Mapping[str, Any], trial_root: Path, arm: str, step: int
) -> Path:
    terminal = read(trial_root / arm / "training/terminal.json")
    entry = next(item for item in terminal["checkpoints"] if item["step"] == step)
    return Path(entry["adapter"]["root"])


def _expected_original_command(
    *, trial: Mapping[str, Any], trial_path: Path, reconciliation: Mapping[str, Any]
) -> list[str]:
    arm, step = reconciliation["arm"], int(reconciliation["step"])
    trial_root = trial_path.parent
    return frozen._readback_command(
        trial=trial,
        trial_path=trial_path,
        training_manifest=Path(trial["arms"][arm]["training_manifest"]["path"]),
        training_terminal=trial_root / arm / "training/terminal.json",
        adapter=_checkpoint_adapter(
            trial=trial, trial_root=trial_root, arm=arm, step=step
        ),
        arm=arm,
        step=step,
        output=trial_root / "readback",
        gpu=int(reconciliation["physical_gpu"]),
        qualification_result=Path(trial["readback"]["qualification_result"]["path"]),
        attempt=str(reconciliation["attempt"]),
    )


def prepare(
    *, trial_path: Path, reconciliation_path: Path, output: Path
) -> dict[str, Any]:
    trial = frozen.validate_trial(read(trial_path))
    reconciliation = read(reconciliation_path)
    require(
        reconciliation.get("status") == "original_S8_worker_live_do_not_duplicate"
        and reconciliation.get("arm") == "S"
        and reconciliation.get("step") == 8,
        "original worker reconciliation",
    )
    start = float(reconciliation["readback_phase_start_unix"])
    deadline = float(reconciliation["readback_phase_deadline_unix"])
    require(deadline - start == PHASE_SECONDS, "original phase deadline")
    require(
        reconciliation.get("cmdline")
        == _expected_original_command(
            trial=trial,
            trial_path=trial_path,
            reconciliation=reconciliation,
        ),
        "original worker command",
    )
    require(not hasattr(os, "pidfd_open"), "pidfd absence reproduction changed")
    mechanics = read(MECHANICS_REPRODUCTION)
    require(
        mechanics.get("status") == "passed"
        and mechanics.get("runtime_has_pidfd_open") is False
        and mechanics.get("success_exit_code") == 0
        and mechanics.get("failure_exit_code") == 7
        and mechanics.get("deadline_failed_closed") is True,
        "portable wait mechanics reproduction",
    )
    manifest_path = output / "manifest.json"
    require(not manifest_path.exists(), "recovery manifest exists")
    value = {
        "schema": f"{SCHEMA}.manifest",
        "status": "candidate_ready_release_pending",
        "trial": training.binding(trial_path),
        "original_failure": training.binding(ORIGINAL_FAILURE),
        "original_worker_reconciliation": training.binding(reconciliation_path),
        "mechanics_reproduction": training.binding(MECHANICS_REPRODUCTION),
        "implementation": training.binding(Path(__file__)),
        "readback_output": str(trial_path.parent / "readback"),
        "endpoints": [{"arm": arm, "step": step} for arm, step in ENDPOINTS],
        "limits": {
            "logical_request_count": REQUEST_COUNT,
            "maximum_live_workers": MAX_LIVE_WORKERS,
            "physical_gpus": list(range(MAX_LIVE_WORKERS)),
            "batch_size": 3,
            "original_phase_start_unix": start,
            "original_phase_deadline_unix": deadline,
            "phase_seconds": PHASE_SECONDS,
            "retrain": False,
        },
        "mechanism": {
            "child_wait": "blocking Popen.wait threads feeding one completion queue",
            "polling": False,
            "pidfd_required": False,
            "runtime_has_pidfd_open": hasattr(os, "pidfd_open"),
            "recovery": "validate retained rows and generate missing rows only",
        },
    }
    value["content_sha256"] = training.digest(value)
    output.mkdir(parents=True, exist_ok=True)
    training.publish(manifest_path, value)
    return value


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    require(
        value.get("schema") == f"{SCHEMA}.manifest"
        and value.get("status") == "candidate_ready_release_pending"
        and value.get("content_sha256") == training.digest(content),
        "recovery manifest",
    )
    require(
        value.get("limits")
        == {
            "logical_request_count": 132,
            "maximum_live_workers": 8,
            "physical_gpus": list(range(8)),
            "batch_size": 3,
            "original_phase_start_unix": 1789468942.59,
            "original_phase_deadline_unix": 1789476142.59,
            "phase_seconds": 7200,
            "retrain": False,
        },
        "recovery limits/deadline",
    )
    require(
        value.get("endpoints")
        == [{"arm": arm, "step": step} for arm, step in ENDPOINTS],
        "recovery endpoints",
    )
    require(
        value.get("mechanism", {}).get("polling") is False
        and value["mechanism"].get("pidfd_required") is False,
        "portable wait mechanism",
    )
    for key in (
        "trial",
        "original_failure",
        "original_worker_reconciliation",
        "mechanics_reproduction",
        "implementation",
    ):
        _verify(value[key], key)
    trial = frozen.validate_trial(read(value["trial"]["path"]))
    reconciliation = read(value["original_worker_reconciliation"]["path"])
    require(
        reconciliation["cmdline"]
        == _expected_original_command(
            trial=trial,
            trial_path=Path(value["trial"]["path"]),
            reconciliation=reconciliation,
        ),
        "bound original worker command",
    )
    return dict(value)


def _publish_attempt(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"attempt receipt collision: {path}")
    training.publish(path, dict(value))


def controller(
    *, manifest_path: Path, release_path: Path, output: Path, attempt: str
) -> dict[str, Any]:
    from probes.training_set_completion import coco227_readback

    manifest = validate_manifest(read(manifest_path))
    release = read(release_path)
    require(
        release.get("status") == "released"
        and release.get("recovery_manifest_sha256")
        == training.file_hash(manifest_path),
        "explicit readback recovery release",
    )
    require(attempt and "/" not in attempt, "recovery attempt ID")
    require(not (output / "terminal.json").exists(), "recovery already completed")
    tmux_session = subprocess.check_output(
        ["tmux", "display-message", "-p", "#S"], text=True
    ).strip()
    require(tmux_session == TMUX_SESSION, "recovery tmux session")
    trial_path = Path(manifest["trial"]["path"])
    trial = frozen.validate_trial(read(trial_path))
    trial_root = trial_path.parent
    readback_output = Path(manifest["readback_output"])
    deadline = float(manifest["limits"]["original_phase_deadline_unix"])
    require(time.time() < deadline, "original readback deadline already elapsed")
    attempt_root = output / "controller" / attempt
    identity = {
        "schema": f"{SCHEMA}.controller_identity",
        "status": "running",
        "pid": os.getpid(),
        "attempt": attempt,
        "tmux_session": tmux_session,
        "manifest": training.binding(manifest_path),
        "release": training.binding(release_path),
        "started_at": time.time(),
        "deadline_unix": deadline,
    }
    _publish_attempt(attempt_root / "identity.json", identity)
    log_path = attempt_root / "controller.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x")

    def emit(message: str) -> None:
        log.write(message + "\n")
        log.flush()
        os.fsync(log.fileno())

    completions: queue.Queue[dict[str, Any]] = queue.Queue()
    active: dict[int, dict[str, Any]] = {}
    exits: list[dict[str, Any]] = []
    collections: list[dict[str, Any]] = []
    started = time.monotonic()
    emit(f"COCO227_READBACK_RECOVERY pid={os.getpid()} attempt={attempt}")
    try:
        for arm in frozen.ARMS:
            frozen.validate_training_terminal(
                trial_root / arm / "training",
                manifest_path=Path(trial["arms"][arm]["training_manifest"]["path"]),
            )
        live = _live_trial_endpoint_processes(
            trial_path=trial_path, readback_output=readback_output
        )
        reconciliation = read(manifest["original_worker_reconciliation"]["path"])
        original = next(
            (item for item in live if item["pid"] == reconciliation["pid"]), None
        )
        if original is not None:
            require(
                original["cmdline"] == reconciliation["cmdline"]
                and abs(original["create_time"] - reconciliation["create_time"])
                <= 0.02,
                "original worker PID identity changed",
            )
        require(not live, "an endpoint worker is still live; refusing duplicate")

        pending: list[tuple[str, int, Path]] = []
        for arm, step in ENDPOINTS:
            adapter = _checkpoint_adapter(
                trial=trial, trial_root=trial_root, arm=arm, step=step
            )
            collection = frozen._collect_endpoint_if_complete(
                readback_module=coco227_readback,
                trial=trial,
                trial_path=trial_path,
                output=trial_root,
                arm=arm,
                step=step,
                adapter=adapter,
            )
            if collection is not None:
                collections.append(collection)
                exits.append(
                    {
                        "name": f"readback-{arm}-{step}",
                        "status": "recovered_completed",
                        "exit_code": 0,
                    }
                )
            else:
                pending.append((arm, step, adapter))

        available = list(range(MAX_LIVE_WORKERS))
        sequence = 0
        while pending or active:
            while pending and available:
                require(time.time() < deadline, "original readback phase deadline")
                arm, step, adapter = pending.pop(0)
                gpu = available.pop(0)
                sequence += 1
                worker_attempt = f"{attempt}-job-{sequence:03d}"
                command = frozen._readback_command(
                    trial=trial,
                    trial_path=trial_path,
                    training_manifest=Path(
                        trial["arms"][arm]["training_manifest"]["path"]
                    ),
                    training_terminal=trial_root / arm / "training/terminal.json",
                    adapter=adapter,
                    arm=arm,
                    step=step,
                    output=readback_output,
                    gpu=gpu,
                    qualification_result=Path(
                        trial["readback"]["qualification_result"]["path"]
                    ),
                    attempt=worker_attempt,
                )
                spawned_unix = time.time()
                process, stream, spawned_monotonic = frozen._spawn(
                    command,
                    visible_devices=str(gpu),
                    log_path=attempt_root / "workers" / f"{arm}-step-{step:05d}.log",
                )
                # Register ownership before starting any fallible wait machinery.
                active[process.pid] = {
                    "process": process,
                    "stream": stream,
                    "command": command,
                    "arm": arm,
                    "step": step,
                    "adapter": adapter,
                    "gpu": gpu,
                    "spawned_unix": spawned_unix,
                    "spawned_monotonic": spawned_monotonic,
                    "deadline_unix": deadline,
                }
                start_process_waiter(
                    process,
                    completions,
                    thread_name_prefix="coco227-readback-recovery-wait",
                )
                emit(
                    f"SPAWN pid={process.pid} arm={arm} step={step} gpu={gpu} "
                    f"deadline={deadline}"
                )
            completion = next_process_completion(
                completions,
                deadline=deadline,
                clock=time.time,
                timeout_message="original readback phase deadline reached",
            )
            item = active.pop(int(completion["pid"]))
            item["stream"].close()
            exit_row = {
                "name": f"readback-{item['arm']}-{item['step']}",
                "pid": completion["pid"],
                "gpu": item["gpu"],
                "exit_code": completion["exit_code"],
                "wait_error": completion["wait_error"],
                "command": item["command"],
                "spawned_at_unix": item["spawned_unix"],
                "completed_at_unix": completion["completed_at_unix"],
                "deadline_unix": deadline,
            }
            exits.append(exit_row)
            available.append(int(item["gpu"]))
            available.sort()
            require(
                completion["exit_code"] == 0,
                f"readback {item['arm']}/{item['step']} failed",
            )
            collection = frozen._collect_endpoint_if_complete(
                readback_module=coco227_readback,
                trial=trial,
                trial_path=trial_path,
                output=trial_root,
                arm=item["arm"],
                step=item["step"],
                adapter=item["adapter"],
            )
            require(collection is not None, "worker left an incomplete endpoint")
            collections.append(collection)
            emit(
                f"EXIT pid={completion['pid']} arm={item['arm']} "
                f"step={item['step']} code=0"
            )

        result = frozen.collect_readbacks(
            trial_path=trial_path, output=trial_root, collections=collections
        )
        require(
            result.get("request_count") == REQUEST_COUNT
            and result.get("endpoint_count") == len(ENDPOINTS),
            "final readback cardinality",
        )
        terminal = {
            "schema": f"{SCHEMA}.controller_terminal",
            "status": "completed_unscored",
            "exit_code": 0,
            "pid": os.getpid(),
            "attempt": attempt,
            "tmux_session": tmux_session,
            "identity": training.binding(attempt_root / "identity.json"),
            "result": training.binding(trial_root / "readback/result.json"),
            "endpoint_count": len(collections),
            "logical_request_count": REQUEST_COUNT,
            "exits": exits,
            "original_phase_deadline_unix": deadline,
            "completed_at_unix": time.time(),
            "elapsed_seconds": time.monotonic() - started,
        }
        require(
            terminal["completed_at_unix"] <= deadline,
            "readback completed after original phase deadline",
        )
        training.publish(output / "terminal.json", terminal)
        emit("COCO227_READBACK_RECOVERY_COMPLETED_UNSCORED")
        return terminal
    except BaseException as exc:
        for item in active.values():
            process = item["process"]
            if process.poll() is None:
                terminate_owned_process(process)
            item["stream"].close()
        failure = {
            "schema": f"{SCHEMA}.controller_failure",
            "status": "failed",
            "exit_code": 1,
            "pid": os.getpid(),
            "attempt": attempt,
            "identity": training.binding(attempt_root / "identity.json"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "exits": exits,
            "original_phase_deadline_unix": deadline,
            "elapsed_seconds": time.monotonic() - started,
        }
        training.publish(attempt_root / "failure.json", failure)
        emit(f"COCO227_READBACK_RECOVERY_FAILED {failure['error']}")
        raise
    finally:
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--trial", type=Path, default=TRIAL)
    p.add_argument("--reconciliation", type=Path, default=ORIGINAL_RECONCILIATION)
    p.add_argument("--output", type=Path, default=RECOVERY_ROOT)
    p = sub.add_parser("verify")
    p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("controller")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--output", type=Path, default=RECOVERY_ROOT)
    p.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare(
                    trial_path=args.trial,
                    reconciliation_path=args.reconciliation,
                    output=args.output,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "verify":
        validate_manifest(read(args.manifest))
    else:
        print(
            json.dumps(
                controller(
                    manifest_path=args.manifest,
                    release_path=args.release,
                    output=args.output,
                    attempt=args.attempt,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
