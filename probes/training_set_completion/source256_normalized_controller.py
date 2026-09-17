"""Durable tmux controller for the one B-normalized Source256 main course.

This controller has one write surface: the successor runtime root.  It never
launches before a lead release, retains compatible completed training and
readback shards, and records a separate failure receipt so a scoring repair can
resume without retraining a valid checkpoint.
"""
from __future__ import annotations

from src.runtime.owned_process import spawn_logged_process, terminate_owned_process, wait_owned_process

import argparse
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from probes.training_set_completion import source256_normalized_evaluation as evaluation
from probes.training_set_completion import source256_normalized_readback as readback
from probes.training_set_completion import training
from src.runtime.process_completion import next_process_completion, start_process_waiter


SCHEMA = "training_set_completion.source256_completion_ce_normalization_controller.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _runtime_root(packet_path: Path) -> Path:
    # Packets are immutable at runtime/main-v1/evaluation/packet.json.
    return packet_path.resolve(strict=True).parent.parent


def _verified_binding(value: Mapping[str, Any], name: str) -> Path:
    require(set(value) == {"path", "sha256", "size_bytes"}, f"{name} binding fields")
    path = Path(str(value["path"])).resolve(strict=True)
    require(training.binding(path) == dict(value), f"{name} binding changed")
    return path


def _validate_release(*, release_path: Path, packet_path: Path, packet: Mapping[str, Any]) -> dict[str, Any]:
    # The controller consumes a lead-owned envelope, not the training
    # backend's release receipt directly.  Check the immutable packet binding
    # and the separate training receipt before touching deeper packet fields so
    # malformed envelopes fail closed with an actionable error.
    runtime = evaluation._normalized_runtime()
    release_path = release_path.resolve(strict=True)
    envelope = read(release_path)
    require(isinstance(envelope, Mapping), "main release envelope")
    require(envelope.get("schema") == f"{SCHEMA}.main_release", "main release envelope schema")
    require(envelope.get("status") == "released", "main release envelope status")
    require(
        envelope.get("packet") == training.binding(packet_path),
        "explicit accepted packet binding",
    )
    sources = packet.get("sources")
    require(isinstance(sources, Mapping), "normalized packet sources")
    require(
        envelope.get("actual_entry_qualification") == sources.get("actual_entry_qualification"),
        "explicit accepted qualification binding",
    )
    training_release = envelope.get("training_release")
    require(isinstance(training_release, Mapping), "separate training release required")
    require(envelope.get("unit_id") == runtime.UNIT_ID, "main release envelope unit")
    training_release_path = _verified_binding(training_release, "training release")

    manifest_path = _verified_binding(
        packet["sources"]["main_training_manifest"], "main training manifest"
    )
    manifest = runtime.validate_training_manifest(read(manifest_path))
    release = runtime.validate_release(training_release_path, manifest=manifest)
    value = release["value"]
    require(
        value.get("main_training_manifest") == training.binding(manifest_path)
        and value.get("qualification") == packet["sources"]["actual_entry_qualification"],
        "lead main release scientific bindings",
    )
    qualification_path = _verified_binding(
        value["qualification"], "lead qualification"
    )
    qualification = read(qualification_path)
    qualification_manifest_path = _verified_binding(
        qualification["qualification_manifest"], "lead qualification manifest"
    )
    qualification_terminal_path = _verified_binding(
        qualification["training_terminal"], "lead qualification terminal"
    )
    cold_reload_path = _verified_binding(value["cold_reload"], "lead cold reload")
    require(cold_reload_path.is_file(), "lead cold reload receipt")
    readback_path = _verified_binding(
        value["natural_batch4_readback"], "lead natural batch4 readback"
    )
    readback.validate_qualification_checkpoint(
        value=read(readback_path),
        control_reuse_path=Path(packet["sources"]["control_reuse"]["path"]),
        plan_path=Path(packet["control_identity"]["readback_plan"]["path"]),
        qualification_path=Path(packet["control_identity"]["batch4_qualification"]["path"]),
        training_manifest_path=qualification_manifest_path,
        terminal_path=qualification_terminal_path,
    )
    expected_release_path = Path(packet["training_launch"]["release_receipt"]).resolve()
    require(
        release["binding"] == training.binding(expected_release_path),
        "lead main release must be the packet-bound receipt",
    )
    return value


def _require_named_tmux(session: str) -> None:
    observed = subprocess.check_output(["tmux", "display-message", "-p", "#S"], text=True).strip()
    require(observed == session, "controller must run in its packet-named tmux session")


def controller_argv(*, packet_path: Path, release_path: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "probes.training_set_completion.source256_normalized_controller",
        "controller",
        "--packet",
        str(packet_path),
        "--release",
        str(release_path),
        "--output",
        str(output),
    ]


def tmux_launch_command(*, packet_path: Path, release_path: Path, output: Path) -> str:
    packet_path = packet_path.resolve(strict=True)
    packet = evaluation._validate_packet(read(packet_path))["packet"]
    _validate_release(release_path=release_path, packet_path=packet_path, packet=packet)
    session = packet["orchestration"]["tmux_session"]
    return shlex.join(
        ["tmux", "new-session", "-d", "-s", session, shlex.join(controller_argv(packet_path=packet_path, release_path=release_path, output=output))]
    )


def _spawn(command: list[str], *, visible_devices: str, log_path: Path) -> tuple[subprocess.Popen[Any], Any, float]:
    return spawn_logged_process(
        command, cwd=Path(__file__).resolve().parents[2], log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": visible_devices, "OMP_NUM_THREADS": "2",
             "TOKENIZERS_PARALLELISM": "false"},
    )


def _wait_one(
    *,
    process: subprocess.Popen[Any],
    stream: Any,
    command: list[str],
    name: str,
    gpu: Any,
    started: float,
    wall_seconds: float,
) -> dict[str, Any]:
    deadline = started + wall_seconds
    try:
        code = wait_owned_process(process, deadline=deadline)
        return {
            "name": name,
            "pid": process.pid,
            "gpu": gpu,
            "exit_code": code,
            "command": command,
            "spawned_at_monotonic": started,
            "deadline_monotonic": deadline,
        }
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{name} wall deadline") from exc
    finally:
        stream.close()


def _validate_controller_terminal(value: Mapping[str, Any], *, packet_path: Path, release_path: Path) -> dict[str, Any]:
    require(
        value.get("schema") == f"{SCHEMA}.terminal"
        and value.get("status") == "completed"
        and value.get("packet") == training.binding(packet_path)
        and value.get("release") == training.binding(release_path)
        and isinstance(value.get("result"), Mapping),
        "completed normalized controller terminal",
    )
    evaluation.validate_result(read(value["result"]["path"]), packet_path=packet_path)
    require(training.binding(value["result"]["path"]) == value["result"], "controller result binding")
    return dict(value)


def _new_shard_jobs(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(job, endpoint={key: endpoint[key] for key in ("label", "arm", "step")}, terminal=endpoint["training_terminal"])
            for endpoint in packet["endpoints"] for job in endpoint["jobs"]]


def _validate_or_pending_shard(
    *,
    job: Mapping[str, Any],
    packet: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> bool:
    path = Path(job["output"])
    if not path.exists():
        return False
    require(path.is_file() and not path.is_symlink(), "existing normalized shard must be a regular file")
    readback.validate_endpoint_shard(
        value=read(path),
        control_reuse_path=paths["control_reuse"],
        plan_path=Path(packet["control_identity"]["readback_plan"]["path"]),
        qualification_path=Path(packet["control_identity"]["batch4_qualification"]["path"]),
        training_manifest_path=paths["manifest"],
        terminal_path=Path(str(job["terminal"])),
        label=str(job["endpoint"]["label"]),
        step=int(job["endpoint"]["step"]),
        split=str(job["split"]),
        shard=int(job["shard"]),
    )
    return True


def controller(*, packet_path: Path, release_path: Path, output: Path) -> dict[str, Any]:
    """Run or recover the released course from durable artifacts only."""

    packet_path = packet_path.resolve(strict=True)
    checked = evaluation._validate_packet(read(packet_path))
    packet = checked["packet"]
    release_path = release_path.resolve(strict=True)
    _validate_release(release_path=release_path, packet_path=packet_path, packet=packet)
    output = output.resolve()
    require(output == _runtime_root(packet_path), "controller output must be the packet runtime root")
    terminal_path = output / "controller-terminal.json"
    if terminal_path.is_file():
        return _validate_controller_terminal(read(terminal_path), packet_path=packet_path, release_path=release_path)
    _require_named_tmux(packet["orchestration"]["tmux_session"])
    # Admit all immutable reused raw controls before any new GPU process starts.
    evaluation.validate_control_reuse(read(checked["paths"]["control_reuse"]))
    identity_root = output / "controller-identities"
    identity_root.mkdir(parents=True, exist_ok=True)
    attempt = len(list(identity_root.glob("attempt-*.json"))) + 1
    identity_path = identity_root / f"attempt-{attempt:03d}.json"
    identity = {
        "schema": f"{SCHEMA}.identity",
        "status": "running",
        "attempt": attempt,
        "pid": os.getpid(),
        "tmux_session": packet["orchestration"]["tmux_session"],
        "packet": training.binding(packet_path),
        "release": training.binding(release_path),
        "started_at_unix": time.time(),
    }
    training.publish(identity_path, identity)
    log_path = output / "logs" / f"controller-attempt-{attempt:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x")

    def emit(line: str) -> None:
        log.write(line + "\n")
        log.flush()
        os.fsync(log.fileno())

    exits: list[dict[str, Any]] = []
    active: dict[int, dict[str, Any]] = {}
    started = time.monotonic()
    paths = checked["paths"]
    try:
        emit(f"SOURCE256_NORMALIZED_CONTROLLER pid={os.getpid()} attempt={attempt}")
        training_launch = packet["training_launch"]
        training_root = Path(training_launch["output"])
        if (training_root / "terminal.json").is_file():
            readback.validate_training_terminal(
                training_manifest_path=paths["manifest"], terminal_path=training_root / "terminal.json"
            )
            exits.append({"name": "train-B-normalized", "status": "recovered_completed", "exit_code": 0})
            emit("TRAINING_RECOVERED_COMPLETED")
        else:
            require(not training_root.exists(), "incomplete normalized training requires explicit repair; refusing retrain")
            manifest = read(paths["manifest"])
            process, stream, spawned = _spawn(
                list(training_launch["command"]),
                visible_devices=",".join(str(item) for item in training_launch["visible_devices"]),
                log_path=output / "logs" / "train-B-normalized.log",
            )
            exit_row = _wait_one(
                process=process,
                stream=stream,
                command=list(training_launch["command"]),
                name="train-B-normalized",
                gpu=training_launch["visible_devices"],
                started=spawned,
                wall_seconds=float(manifest["runtime"]["wall_seconds"]),
            )
            exits.append(exit_row)
            require(exit_row["exit_code"] == 0, "B-normalized training process failed")
            readback.validate_training_terminal(
                training_manifest_path=paths["manifest"], terminal_path=training_root / "terminal.json"
            )
            emit("TRAINING_COMPLETED")
        attempt_root = output / "controller-attempts" / f"attempt-{attempt:03d}"
        training.publish(attempt_root / "training-exits.json", {"schema": f"{SCHEMA}.training_exits", "exits": list(exits)})

        pending = []
        for job in _new_shard_jobs(packet):
            if _validate_or_pending_shard(job=job, packet=packet, paths=paths):
                exits.append({
                    "name": f"readback-{job['endpoint']['label']}-{job['split']}-{job['shard']:02d}",
                    "status": "recovered_completed", "exit_code": 0,
                })
            else:
                pending.append(job)
        emit(f"READBACK_PENDING count={len(pending)}")
        completions: queue.Queue[dict[str, Any]] = queue.Queue()
        available = list(range(8))
        phase_deadline = time.monotonic() + 14_400
        while pending or active:
            while pending and available:
                require(time.monotonic() < phase_deadline, "normalized readback phase deadline")
                job = pending.pop(0)
                gpu = available.pop(0)
                command = list(job["command"])
                process, stream, spawned = _spawn(
                    command,
                    visible_devices=str(gpu),
                    log_path=output / "logs" / f"readback-{job['endpoint']['label']}-{job['split']}-{job['shard']:02d}-attempt-{attempt:03d}.log",
                )
                active[process.pid] = {
                    "process": process, "stream": stream, "job": job, "gpu": gpu,
                    "command": command, "spawned": spawned,
                }
                start_process_waiter(process, completions, thread_name_prefix="source256-normalized-readback-wait")
                emit(f"READBACK_SPAWN pid={process.pid} endpoint={job['endpoint']['label']} split={job['split']} shard={job['shard']} gpu={gpu}")
            completion = next_process_completion(
                completions,
                deadline=phase_deadline,
                timeout_message="normalized readback phase deadline",
            )
            item = active.pop(int(completion["pid"]))
            item["stream"].close()
            job = item["job"]
            exit_row = {
                "name": f"readback-{job['endpoint']['label']}-{job['split']}-{job['shard']:02d}",
                "pid": completion["pid"], "gpu": item["gpu"],
                "exit_code": completion["exit_code"], "wait_error": completion["wait_error"],
                "command": item["command"], "spawned_at_monotonic": item["spawned"],
                "completed_at_monotonic": completion["completed_at_monotonic"],
            }
            exits.append(exit_row)
            available.append(int(item["gpu"]))
            available.sort()
            require(exit_row["exit_code"] == 0, f"{exit_row['name']} failed")
            require(_validate_or_pending_shard(job=job, packet=packet, paths=paths), "readback worker left invalid shard")
            emit(f"READBACK_EXIT pid={completion['pid']} endpoint={job['endpoint']['label']} split={job['split']} shard={job['shard']} code=0")
        training.publish(attempt_root / "readback-exits.json", {"schema": f"{SCHEMA}.readback_exits", "exits": list(exits)})

        result_path = Path(packet["reducer"]["output"])
        if result_path.is_file():
            evaluation.validate_result(read(result_path), packet_path=packet_path)
            emit("REDUCER_RECOVERED_COMPLETED")
        else:
            evaluation.reduce(packet_path=packet_path, output=result_path)
            evaluation.validate_result(read(result_path), packet_path=packet_path)
            emit("REDUCER_COMPLETED")
        terminal = {
            "schema": f"{SCHEMA}.terminal",
            "status": "completed",
            "pid": os.getpid(),
            "attempt": attempt,
            "tmux_session": identity["tmux_session"],
            "identity": training.binding(identity_path),
            "packet": identity["packet"],
            "release": identity["release"],
            "result": training.binding(result_path),
            "exits": exits,
            "elapsed_seconds": time.monotonic() - started,
        }
        training.publish(terminal_path, terminal)
        emit(f"SOURCE256_NORMALIZED_COMPLETED result={result_path}")
        return terminal
    except BaseException as exc:
        cleanup_errors = []
        for item in active.values():
            try:
                terminate_owned_process(item["process"])
            except BaseException as cleanup_exc:
                cleanup_errors.append(f"{type(cleanup_exc).__name__}: {cleanup_exc}")
            finally:
                item["stream"].close()
        failure = {
            "schema": f"{SCHEMA}.failure",
            "status": "failed",
            "pid": os.getpid(),
            "attempt": attempt,
            "identity": training.binding(identity_path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "exits": exits,
            "cleanup_errors": cleanup_errors,
            "elapsed_seconds": time.monotonic() - started,
        }
        training.publish(output / "controller-failures" / f"attempt-{attempt:03d}.json", failure)
        emit(f"SOURCE256_NORMALIZED_FAILED {failure['error']}")
        raise
    finally:
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    control = sub.add_parser("controller")
    control.add_argument("--packet", type=Path, required=True)
    control.add_argument("--release", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    launch = sub.add_parser("tmux-command")
    launch.add_argument("--packet", type=Path, required=True)
    launch.add_argument("--release", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "controller":
        result = controller(packet_path=args.packet, release_path=args.release, output=args.output)
        print(json.dumps(result, sort_keys=True))
    else:
        print(tmux_launch_command(packet_path=args.packet, release_path=args.release, output=args.output))


if __name__ == "__main__":
    main()
