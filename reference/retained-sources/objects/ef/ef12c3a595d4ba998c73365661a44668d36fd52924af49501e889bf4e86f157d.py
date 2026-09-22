#!/root/miniconda3/envs/ms/bin/python3
"""Bounded concurrent paired fit launcher for the frozen row-feedback packet."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-row-feedback-pilot/training")
WORKTREE = Path("/data/CoordExp/.worktrees/row-feedback-pilot-20260913")
PACKET = BASE / "fit-packet-v1.json"
LAUNCH_RECEIPT = BASE / "fit-pair-launch-receipt-v1.json"
TERMINAL_RECEIPT = BASE / "fit-pair-terminal-receipt-v1.json"
PAIR_LOG = BASE / "fit-pair.log"
LAUNCHER_LOG = BASE / "fit-pair-launcher.log"
ARMS = ("S", "F")
GPUS = {"S": 0, "F": 1}
TIMEOUT_SECONDS = 36000.0
FROZEN_COMMIT = "2eb4d10802bb68036576529bc0c7f68946de352a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_pair_log(event: str, payload: dict[str, Any]) -> None:
    with PAIR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(event + " " + json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def terminate_group(process: subprocess.Popen[Any], reason: str) -> None:
    if process.poll() is not None:
        return
    append_pair_log("FIT_CHILD_TERMINATE", {
        "arm": process._row_feedback_arm, "producer_pid": process.pid, "reason": reason,
    })
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=60.0)


def fit_receipt_status(path: Path, packet_hash: str, arm: str) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "status": "missing"}
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        updates = receipt.get("updates", [])
        valid = (
            receipt.get("schema") == "row_feedback.training_receipt.v1"
            and receipt.get("status") == "fit_complete_serial_arm_saved"
            and receipt.get("mode") == "fit"
            and receipt.get("arm") == arm
            and receipt.get("packet", {}).get("sha256") == packet_hash
            and len(updates) == 64
        )
        return {
            "exists": True, "status": receipt.get("status"), "valid_64_update_receipt": valid,
            "sha256": sha256(path), "updates": len(updates),
        }
    except Exception as exc:  # receipt corruption is a failed completion
        return {"exists": True, "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=PACKET)
    args = parser.parse_args()
    packet_path = args.packet.resolve()
    packet_bytes = packet_path.read_bytes()
    packet_hash = hashlib.sha256(packet_bytes).hexdigest()
    packet = json.loads(packet_bytes)
    if packet_hash != "6591ae6fa8fc2dc1ef4c48c60ea884ce8c2885421e00018ad67a712d01fcab8a":
        raise SystemExit(f"unexpected fit packet hash: {packet_hash}")
    if packet.get("schema") != "row_feedback.training_packet.v1":
        raise SystemExit("unexpected fit packet schema")
    execution = packet.get("execution", {})
    if packet.get("status") != "root_frozen_ready_for_fit":
        raise SystemExit("fit packet is not root-frozen")
    if execution.get("optimizer_updates_per_arm") != 64 or execution.get("physical_gpus") != GPUS:
        raise SystemExit("fit packet topology or dose changed")
    if execution.get("maximum_process_wall_seconds_per_arm") != int(TIMEOUT_SECONDS):
        raise SystemExit("fit packet timeout changed")
    if not execution.get("concurrent_arms") or execution.get("stop_on_first_failure") is not True:
        raise SystemExit("fit packet concurrency or stop rule changed")

    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True,
    ).strip()
    bindings = packet["code_bindings"]
    source_hashes = {
        name: {"path": binding["path"], "sha256": sha256(Path(binding["path"])),
               "packet_sha256": binding["sha256"]}
        for name, binding in bindings.items()
    }
    mismatches = [name for name, value in source_hashes.items()
                  if value["sha256"] != value["packet_sha256"]]
    if mismatches:
        raise SystemExit(f"packet source binding changed: {mismatches}")

    # The detached parent opens LAUNCHER_LOG before this script starts.
    required_absent = [LAUNCH_RECEIPT, TERMINAL_RECEIPT, PAIR_LOG]
    for arm in ARMS:
        required_absent.extend([
            BASE / f"fit-{arm}-v1",
            BASE / f"fit-{arm}-v1.outer.log",
            BASE / f"fit-{arm}-v1.outer-receipt.json",
        ])
    for path in required_absent:
        if path.exists():
            raise SystemExit(f"refuse preexisting fit artifact: {path}")

    BASE.mkdir(parents=True, exist_ok=True)
    PAIR_LOG.touch(mode=0o644, exist_ok=False)
    append_pair_log("FIT_PAIR_LAUNCHING", {
        "packet": {"path": str(packet_path), "sha256": packet_hash},
        "frozen_commit": FROZEN_COMMIT, "current_git_head": current_head,
    })

    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"
    # Keep the detached launcher CPU-only; each child receives one explicit GPU.
    env_base["CUDA_VISIBLE_DEVICES"] = ""
    children: dict[str, subprocess.Popen[Any]] = {}
    logs: dict[str, Any] = {}
    launch_started = time.time()
    launch_mono = time.monotonic()
    commands: dict[str, list[str]] = {}
    try:
        for arm in ARMS:
            output = BASE / f"fit-{arm}-v1"
            arm_log = BASE / f"fit-{arm}-v1.outer.log"
            command = [
                sys.executable, "-m", "probes.row_feedback.training",
                "--packet", str(packet_path), "--output", str(output),
                "--arm", arm, "--mode", "fit",
            ]
            commands[arm] = command
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(GPUS[arm])
            log_handle = arm_log.open("x", encoding="utf-8", buffering=1)
            log_handle.write(json.dumps({
                "schema": "row_feedback.arm_outer_launch.v1", "arm": arm,
                "command": command, "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "cwd": str(WORKTREE), "packet": {"path": str(packet_path), "sha256": packet_hash},
                "output": str(output), "started_unix": time.time(),
            }, sort_keys=True) + "\n")
            log_handle.flush()
            process = subprocess.Popen(
                command, cwd=str(WORKTREE), env=env, stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
            process._row_feedback_arm = arm  # type: ignore[attr-defined]
            children[arm] = process
            logs[arm] = log_handle
            append_pair_log("FIT_CHILD_STARTED", {
                "arm": arm, "producer_pid": process.pid,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "command": command,
            })
    except BaseException as exc:
        for process in children.values():
            terminate_group(process, f"launch failure: {type(exc).__name__}: {exc}")
        for handle in logs.values():
            handle.close()
        append_pair_log("FIT_PAIR_FAILED", {"phase": "launch", "error": f"{type(exc).__name__}: {exc}"})
        raise

    launch_receipt = {
        "schema": "row_feedback.fit_pair_launch_receipt.v1", "status": "launched_waiting",
        "parent_pid": os.getpid(), "frozen_commit": FROZEN_COMMIT,
        "current_git_head": current_head,
        "worktree": str(WORKTREE), "cwd": str(WORKTREE),
        "packet": {"path": str(packet_path), "sha256": packet_hash},
        "code_bindings": source_hashes,
        "arms": {
            arm: {"physical_gpu": GPUS[arm], "producer_pid": children[arm].pid,
                  "command": commands[arm],
                  "outer_log": str(BASE / f"fit-{arm}-v1.outer.log"),
                  "output": str(BASE / f"fit-{arm}-v1")}
            for arm in ARMS
        },
        "pair_log": str(PAIR_LOG), "launcher_log": str(LAUNCHER_LOG),
        "started_unix": launch_started, "timeout_seconds_per_arm": TIMEOUT_SECONDS,
        "stop_on_first_failure": True, "automatic_retries": False,
    }
    write_exclusive(LAUNCH_RECEIPT, launch_receipt)
    append_pair_log("FIT_PAIR_HANDSHAKE_READY", {
        "parent_pid": os.getpid(),
        "producer_pids": {arm: children[arm].pid for arm in ARMS},
        "launch_receipt": str(LAUNCH_RECEIPT),
    })

    def wait_one(arm: str) -> dict[str, Any]:
        process = children[arm]
        started = time.time()
        timed_out = False
        try:
            exit_code = process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group(process, "per-arm timeout")
            exit_code = process.returncode
        finished = time.time()
        receipt_path = BASE / f"fit-{arm}-v1/training-receipt.json"
        result = {
            "arm": arm, "producer_pid": process.pid, "physical_gpu": GPUS[arm],
            "exit_code": exit_code, "timed_out": timed_out,
            "started_unix": started, "finished_unix": finished,
            "wall_seconds": finished - started,
            "output": str(BASE / f"fit-{arm}-v1"),
            "outer_log": str(BASE / f"fit-{arm}-v1.outer.log"),
            "training_receipt": fit_receipt_status(receipt_path, packet_hash, arm),
        }
        with (BASE / f"fit-{arm}-v1.outer-receipt.json").open("x", encoding="utf-8") as handle:
            json.dump({
                "schema": "row_feedback.fit_arm_outer_receipt.v1",
                "status": "complete" if exit_code == 0 and result["training_receipt"].get("valid_64_update_receipt") else "failed",
                "packet": {"path": str(packet_path), "sha256": packet_hash},
                "command": commands[arm], "cuda_visible_devices": str(GPUS[arm]),
                "cwd": str(WORKTREE), **result,
            }, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logs[arm].flush()
        return result

    outcomes: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="fit-wait") as pool:
        futures = {pool.submit(wait_one, arm): arm for arm in ARMS}
        for future in concurrent.futures.as_completed(futures):
            arm = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                result = {"arm": arm, "producer_pid": children[arm].pid,
                          "exit_code": None, "timed_out": False,
                          "error": f"{type(exc).__name__}: {exc}"}
            outcomes[arm] = result
            failed = result.get("exit_code") != 0 or not result.get("training_receipt", {}).get("valid_64_update_receipt", False)
            if failed:
                append_pair_log("FIT_PAIR_FIRST_FAILURE", result)
                for other_arm, process in children.items():
                    if other_arm != arm:
                        terminate_group(process, f"paired arm {arm} failed")

    for handle in logs.values():
        handle.close()
    finished = time.time()
    success = (
        len(outcomes) == 2
        and all(result.get("exit_code") == 0 for result in outcomes.values())
        and all(result.get("training_receipt", {}).get("valid_64_update_receipt", False)
                for result in outcomes.values())
    )
    terminal = {
        "schema": "row_feedback.fit_pair_terminal_receipt.v1",
        "status": "complete" if success else "failed",
        "claim_boundary": "64-update paired fit execution receipt only; no endpoint quality claim.",
        "parent_pid": os.getpid(), "frozen_commit": FROZEN_COMMIT,
        "current_git_head": current_head, "worktree": str(WORKTREE),
        "packet": {"path": str(packet_path), "sha256": packet_hash},
        "code_bindings": source_hashes, "pair_log": str(PAIR_LOG),
        "launch_receipt": {"path": str(LAUNCH_RECEIPT), "sha256": sha256(LAUNCH_RECEIPT)},
        "arms": outcomes,
        "started_unix": launch_started, "finished_unix": finished,
        "pair_wall_seconds": finished - launch_started,
        "pair_allocated_gpu_hours": 2.0 * (finished - launch_started) / 3600.0,
        "maximum_pair_allocated_gpu_hours": 20.0,
    }
    write_exclusive(TERMINAL_RECEIPT, terminal)
    append_pair_log("FIT_PAIR_COMPLETE" if success else "FIT_PAIR_FAILED", {
        "terminal_receipt": str(TERMINAL_RECEIPT), "terminal_receipt_sha256": sha256(TERMINAL_RECEIPT),
        "status": terminal["status"], "pair_wall_seconds": terminal["pair_wall_seconds"],
        "outcomes": outcomes,
    })
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
