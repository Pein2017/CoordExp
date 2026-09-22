#!/usr/bin/env python3
"""Durable launcher for the frozen seven-worker natural endpoint.

This controller owns process durability and outer time limits only.  Endpoint
materialization, generation, receipts, merging, and scoring remain in the
committed row-feedback evaluation modules.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


WORKTREE = Path("/data/CoordExp/.worktrees/row-feedback-pilot-20260913")
OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-13-row-feedback-pilot/evaluation"
)
SELECTION = OUTPUT_ROOT / "selection-v2.json"
SELECTION_SHA256 = "22471903a08d9f075f352703cf4bdfe79ee9e1419abe86fa74756fc4255c693d"
NATURAL_ROOT = OUTPUT_ROOT / "natural"
OUTER_TIMEOUT_SECONDS = 6300
KILL_GRACE_SECONDS = 60
MAX_ALLOCATED_GPU_HOURS = 12.25
EXPECTED_IMAGES = 32
EXPECTED_ANNOTATIONS = 280
EXPECTED_DOSE = 64

SHARDS = (
    *(("S", index, 4, index) for index in range(4)),
    *(("F", index, 3, index + 4) for index in range(3)),
)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def publish(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def label(arm: str, index: int) -> str:
    return f"{arm}-shard-{index}"


def output_dir(natural_root: Path, arm: str, index: int) -> Path:
    return natural_root / arm / f"shard-{index}"


def _load_and_validate_selection(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(WORKTREE))
    from probes.row_feedback.evaluation import validate_selection

    value = read(path)
    validate_selection(value, verify_files=True)
    require(sha256(path) == SELECTION_SHA256, "selection-v2 file digest changed")
    require(len(value.get("image_ids", [])) == EXPECTED_IMAGES, "endpoint image count changed")
    require(len(value.get("records", [])) == EXPECTED_IMAGES, "endpoint record count changed")
    annotation_count = sum(int(row["annotated_object_count"]) for row in value["records"])
    require(annotation_count == EXPECTED_ANNOTATIONS, "endpoint annotation count changed")
    require(value.get("model_calls") == 0, "selection must remain pre-output")
    return value


def _load_and_validate_adapter(path: Path, arm: str) -> dict[str, Any]:
    sys.path.insert(0, str(WORKTREE))
    from probes.row_feedback.evaluation_run import validate_adapter_binding

    value = validate_adapter_binding(path, arm=arm)
    require(value["dose"] == EXPECTED_DOSE, f"{arm} adapter is not the frozen dose 64")
    return value


def plan(selection_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(WORKTREE))
    from probes.row_feedback.evaluation_run import shard_ids

    selection = _load_and_validate_selection(selection_path)
    rows = []
    for arm, index, count, gpu in SHARDS:
        ids = shard_ids(selection["image_ids"], shard_index=index, shard_count=count)
        rows.append(
            {
                "label": label(arm, index),
                "arm": arm,
                "shard_index": index,
                "shard_count": count,
                "physical_gpu": gpu,
                "image_ids": ids,
                "images": len(ids),
            }
        )
    require(
        sorted(item for row in rows if row["arm"] == "S" for item in row["image_ids"])
        == sorted(selection["image_ids"]),
        "S shard coverage changed",
    )
    require(
        sorted(item for row in rows if row["arm"] == "F" for item in row["image_ids"])
        == sorted(selection["image_ids"]),
        "F shard coverage changed",
    )
    return {
        "schema": "row_feedback.endpoint_launch_plan.v1",
        "status": "preflight_only_no_processes_started",
        "selection": binding(selection_path),
        "images_per_arm": EXPECTED_IMAGES,
        "annotations": EXPECTED_ANNOTATIONS,
        "max_visible_tokens": 3084,
        "decode": {"temperature": 0, "top_p": 1, "repetition_penalty": 1, "history": "empty"},
        "outer_timeout_seconds_per_shard": OUTER_TIMEOUT_SECONDS,
        "kill_grace_seconds": KILL_GRACE_SECONDS,
        "max_allocated_gpu_hours": MAX_ALLOCATED_GPU_HOURS,
        "automatic_retries": 0,
        "shards": rows,
    }


def append_event(controller: Path, value: Mapping[str, Any]) -> None:
    path = controller / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _settlement_path(controller: Path, shard_label: str) -> Path | None:
    for suffix in ("outer-terminal.json", "outer-failure.json"):
        candidate = controller / "processes" / shard_label / suffix
        if candidate.is_file():
            return candidate
    return None


def publish_rollup_events(controller: Path) -> None:
    """Have the last settling wrapper emit stable per-arm and suite events."""

    events_path = controller / "events.jsonl"
    with events_path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        existing = [json.loads(line) for line in stream if line.strip()]
        emitted = {(row.get("event"), row.get("arm")) for row in existing}
        settlements: dict[str, dict[str, Any]] = {}
        for arm, index, _count, _gpu in SHARDS:
            shard_label = label(arm, index)
            path = _settlement_path(controller, shard_label)
            if path is not None:
                settlements[shard_label] = read(path)

        additions = []
        for arm, expected in (("S", 4), ("F", 3)):
            arm_rows = [value for key, value in settlements.items() if key.startswith(f"{arm}-")]
            if len(arm_rows) == expected and ("arm_terminal", arm) not in emitted:
                completed = all(row.get("status") == "completed" for row in arm_rows)
                additions.append(
                    {
                        "schema": "row_feedback.endpoint_controller_event.v1",
                        "event": "arm_terminal",
                        "arm": arm,
                        "status": "completed" if completed else "failed",
                        "timestamp_utc": utc_now(),
                        "producer_pids": sorted(int(row["producer_pid"]) for row in arm_rows if row.get("producer_pid")),
                    }
                )
        if len(settlements) == len(SHARDS) and ("suite_terminal", None) not in emitted:
            completed = all(row.get("status") == "completed" for row in settlements.values())
            additions.append(
                {
                    "schema": "row_feedback.endpoint_controller_event.v1",
                    "event": "suite_terminal",
                    "status": "completed" if completed else "failed",
                    "timestamp_utc": utc_now(),
                    "producer_pids": sorted(
                        int(row["producer_pid"]) for row in settlements.values() if row.get("producer_pid")
                    ),
                }
            )
        stream.seek(0, os.SEEK_END)
        for item in additions:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
        if additions:
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def wrap_worker(args: argparse.Namespace) -> int:
    controller = args.natural_root / "controller"
    shard_label = label(args.arm, args.shard_index)
    process_root = controller / "processes" / shard_label
    process_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    producer: subprocess.Popen[bytes] | None = None
    try:
        command = [
            sys.executable,
            "-m",
            "probes.row_feedback.evaluation_run",
            "worker",
            "--selection",
            str(args.selection),
            "--adapter-binding",
            str(args.adapter_binding),
            "--arm",
            args.arm,
            "--shard-index",
            str(args.shard_index),
            "--shard-count",
            str(args.shard_count),
            "--physical-gpu",
            str(args.physical_gpu),
            "--output",
            str(args.output),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
        producer = subprocess.Popen(
            command,
            cwd=WORKTREE,
            env=environment,
            start_new_session=True,
        )
        outer_launch = {
            "schema": "row_feedback.endpoint_outer_launch.v1",
            "status": "running",
            "timestamp_utc": utc_now(),
            "label": shard_label,
            "wrapper_pid": os.getpid(),
            "wrapper_pgid": os.getpgrp(),
            "producer_pid": producer.pid,
            "producer_pgid": producer.pid,
            "arm": args.arm,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "physical_gpu": args.physical_gpu,
            "outer_timeout_seconds": OUTER_TIMEOUT_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS,
            "command": command,
            "cwd": str(WORKTREE),
            "output": str(args.output),
        }
        publish(process_root / "outer-launch.json", outer_launch)
        append_event(
            controller,
            {
                "schema": "row_feedback.endpoint_controller_event.v1",
                "event": "producer_started",
                "timestamp_utc": utc_now(),
                "label": shard_label,
                "arm": args.arm,
                "producer_pid": producer.pid,
                "physical_gpu": args.physical_gpu,
            },
        )
        timed_out = False
        try:
            exit_code = producer.wait(timeout=OUTER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(producer.pid, signal.SIGTERM)
            try:
                producer.wait(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(producer.pid, signal.SIGKILL)
                producer.wait()
            exit_code = 124

        worker_terminal = args.output / "terminal.json"
        worker_failure = args.output / "failure.json"
        completed = exit_code == 0 and worker_terminal.is_file() and not timed_out
        terminal = {
            "schema": "row_feedback.endpoint_outer_terminal.v1",
            "status": "completed" if completed else "failed",
            "timestamp_utc": utc_now(),
            "label": shard_label,
            "wrapper_pid": os.getpid(),
            "producer_pid": producer.pid,
            "producer_pgid": producer.pid,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started,
            "worker_terminal": binding(worker_terminal) if worker_terminal.is_file() else None,
            "worker_failure": binding(worker_failure) if worker_failure.is_file() else None,
        }
        publish(process_root / "outer-terminal.json", terminal)
        append_event(
            controller,
            {
                "schema": "row_feedback.endpoint_controller_event.v1",
                "event": "producer_terminal",
                "timestamp_utc": utc_now(),
                "label": shard_label,
                "arm": args.arm,
                "status": terminal["status"],
                "producer_pid": producer.pid,
                "exit_code": exit_code,
                "timed_out": timed_out,
            },
        )
        publish_rollup_events(controller)
        return 0 if completed else int(exit_code or 1)
    except BaseException as exc:
        failure = {
            "schema": "row_feedback.endpoint_outer_failure.v1",
            "status": "failed",
            "timestamp_utc": utc_now(),
            "label": shard_label,
            "wrapper_pid": os.getpid(),
            "producer_pid": None if producer is None else producer.pid,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        publish(process_root / "outer-failure.json", failure)
        append_event(
            controller,
            {
                "schema": "row_feedback.endpoint_controller_event.v1",
                "event": "producer_terminal",
                "timestamp_utc": utc_now(),
                "label": shard_label,
                "arm": args.arm,
                "status": "failed",
                "producer_pid": failure["producer_pid"],
                "error": failure["error"],
            },
        )
        publish_rollup_events(controller)
        return 1


def launch(args: argparse.Namespace) -> dict[str, Any]:
    launch_plan = plan(args.selection)
    s_binding = _load_and_validate_adapter(args.s_binding, "S")
    f_binding = _load_and_validate_adapter(args.f_binding, "F")
    require(not args.natural_root.exists(), "natural endpoint root already exists")
    controller = args.natural_root / "controller"
    logs = controller / "logs"
    logs.mkdir(parents=True)
    publish(controller / "plan.json", launch_plan)

    wrappers = []
    try:
        for arm, index, count, gpu in SHARDS:
            shard_label = label(arm, index)
            adapter_path = args.s_binding if arm == "S" else args.f_binding
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "wrap-worker",
                "--selection",
                str(args.selection.resolve()),
                "--adapter-binding",
                str(adapter_path.resolve()),
                "--arm",
                arm,
                "--shard-index",
                str(index),
                "--shard-count",
                str(count),
                "--physical-gpu",
                str(gpu),
                "--output",
                str(output_dir(args.natural_root, arm, index).resolve()),
                "--natural-root",
                str(args.natural_root.resolve()),
            ]
            log_path = logs / f"{shard_label}.log"
            log_stream = log_path.open("xb", buffering=0)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=WORKTREE,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_stream.close()
            wrappers.append(
                {
                    "label": shard_label,
                    "arm": arm,
                    "shard_index": index,
                    "shard_count": count,
                    "physical_gpu": gpu,
                    "wrapper_pid": process.pid,
                    "wrapper_pgid": process.pid,
                    "command": command,
                    "log": str(log_path.resolve()),
                    "output": str(output_dir(args.natural_root, arm, index).resolve()),
                }
            )
    except BaseException as exc:
        failure = {
            "schema": "row_feedback.endpoint_controller_launch_failure.v1",
            "status": "partial_launch_failure",
            "timestamp_utc": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "started_wrappers": wrappers,
            "automatic_retry": False,
        }
        publish(controller / "launch-failure.json", failure)
        raise

    deadline = time.monotonic() + 30
    outer_launches: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        for item in wrappers:
            path = controller / "processes" / item["label"] / "outer-launch.json"
            if item["label"] not in outer_launches and path.is_file():
                outer_launches[item["label"]] = read(path)
        if len(outer_launches) == len(wrappers):
            break
        time.sleep(0.05)
    require(len(outer_launches) == len(wrappers), "not all worker producers published launch identity")

    processes = []
    plan_by_label = {row["label"]: row for row in launch_plan["shards"]}
    for item in wrappers:
        outer = outer_launches[item["label"]]
        processes.append(
            {
                **item,
                "producer_pid": outer["producer_pid"],
                "producer_pgid": outer["producer_pgid"],
                "expected_image_ids": plan_by_label[item["label"]]["image_ids"],
                "outer_launch": binding(
                    controller / "processes" / item["label"] / "outer-launch.json"
                ),
            }
        )
    receipt = {
        "schema": "row_feedback.endpoint_controller_launch_receipt.v1",
        "status": "launched",
        "timestamp_utc": utc_now(),
        "controller_pid": os.getpid(),
        "worktree": str(WORKTREE),
        "launcher": binding(Path(__file__)),
        "selection": binding(args.selection),
        "adapter_bindings": {"S": binding(args.s_binding), "F": binding(args.f_binding)},
        "adapter_fingerprints": {
            "S": s_binding["adapter"]["fingerprint"],
            "F": f_binding["adapter"]["fingerprint"],
        },
        "dose": EXPECTED_DOSE,
        "processes": processes,
        "durability": {
            "detached_wrapper_sessions": True,
            "detached_producer_sessions": True,
            "outer_timeout_seconds_per_shard": OUTER_TIMEOUT_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS,
            "automatic_retries": 0,
            "max_allocated_gpu_hours": MAX_ALLOCATED_GPU_HOURS,
            "events_log": str((controller / "events.jsonl").resolve()),
        },
        "acceptance": {
            "required_images_per_arm": EXPECTED_IMAGES,
            "required_annotations": EXPECTED_ANNOTATIONS,
            "required_shard_terminals": len(SHARDS),
            "merge_and_consume_are_manual_after_all_shards_complete": True,
        },
    }
    publish(controller / "launch-receipt.json", receipt)
    return receipt


def _process_state(pid: int | None) -> str:
    if not pid:
        return "not_started"
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return "exited"
    try:
        state = stat.read_text().split()[2]
    except (OSError, IndexError):
        return "unknown"
    return "zombie" if state == "Z" else "running"


def status(natural_root: Path) -> dict[str, Any]:
    controller = natural_root / "controller"
    receipt = read(controller / "launch-receipt.json")
    rows = []
    for item in receipt["processes"]:
        shard_label = item["label"]
        settlement = _settlement_path(controller, shard_label)
        value = None if settlement is None else read(settlement)
        rows.append(
            {
                "label": shard_label,
                "physical_gpu": item["physical_gpu"],
                "wrapper_pid": item["wrapper_pid"],
                "wrapper_state": _process_state(item["wrapper_pid"]),
                "producer_pid": item["producer_pid"],
                "producer_state": _process_state(item["producer_pid"]),
                "settlement": None if settlement is None else binding(settlement),
                "status": "running" if value is None else value["status"],
                "log": item["log"],
            }
        )
    return {
        "schema": "row_feedback.endpoint_controller_status.v1",
        "timestamp_utc": utc_now(),
        "launch_receipt": binding(controller / "launch-receipt.json"),
        "processes": rows,
        "settled": sum(row["settlement"] is not None for row in rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "events_log": str((controller / "events.jsonl").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--selection", type=Path, default=SELECTION)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--selection", type=Path, default=SELECTION)
    launch_parser.add_argument("--s-binding", type=Path, required=True)
    launch_parser.add_argument("--f-binding", type=Path, required=True)
    launch_parser.add_argument("--natural-root", type=Path, default=NATURAL_ROOT)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--natural-root", type=Path, default=NATURAL_ROOT)
    wrapper = commands.add_parser("wrap-worker", help=argparse.SUPPRESS)
    wrapper.add_argument("--selection", type=Path, required=True)
    wrapper.add_argument("--adapter-binding", type=Path, required=True)
    wrapper.add_argument("--arm", choices=("S", "F"), required=True)
    wrapper.add_argument("--shard-index", type=int, required=True)
    wrapper.add_argument("--shard-count", type=int, required=True)
    wrapper.add_argument("--physical-gpu", type=int, required=True)
    wrapper.add_argument("--output", type=Path, required=True)
    wrapper.add_argument("--natural-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.selection)
    elif args.command == "launch":
        result = launch(args)
    elif args.command == "status":
        result = status(args.natural_root)
    else:
        return wrap_worker(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
