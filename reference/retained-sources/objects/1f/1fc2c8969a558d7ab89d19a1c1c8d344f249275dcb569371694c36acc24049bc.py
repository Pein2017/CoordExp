#!/root/miniconda3/envs/ms/bin/python
"""Durable one-shot launcher for the authorized repaired content retry."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from typing import Any, Mapping


WORKTREE = Path("/data/CoordExp/.worktrees/row-feedback-pilot-20260913")
ROOT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-row-feedback-pilot")
CONTENT = ROOT / "content"
PACKET = CONTENT / "packet-v2.json"
FIT_ACCEPTANCE = ROOT / "training/fit-acceptance-v1.json"
F_BINDING = ROOT / "evaluation/F-fit-binding-v1.json"
RETRY_AUTHORIZATION = CONTENT / "retry-authorization-v1.json"
ADAPTER = ROOT / "training/fit-F-v1/adapter"
OUTPUT = CONTENT / "run-v2"
LOG = CONTENT / "run-v2.log"
LAUNCH_RECEIPT = CONTENT / "run-v2-launch-receipt.json"
TERMINAL_RECEIPT = CONTENT / "run-v2-terminal-receipt.json"
TIMEOUT_SECONDS = 13_425.0
KILL_GRACE_SECONDS = 60.0
PHYSICAL_GPU = "7"
EXPECTED = {
    PACKET: "014c06f70e57032ed6daa8a344270be77dc61dafa19f71e5cc3a23167ad0dfdd",
    FIT_ACCEPTANCE: "5db237af435bcca212dfd9fd5cba8874d698ecd688d8b98cd3c5065bc4ae24ff",
    F_BINDING: "3c662391a986cf1e9958e209c0bed4f3d15bcb8affc670500e8d5c84efb84ee1",
    RETRY_AUTHORIZATION: "fd98a5f702cfda206a86ec11b03cc6236a344a08e6d89fa82ce3d231ffa7468a",
    WORKTREE / "probes/row_feedback/content.py": "383028378e27b33e49d76f1eeae218ddd617b69887f8cecbeba224147778ed1d",
    WORKTREE / "probes/row_feedback/content_run.py": "c2e0c5829c302cf78ca26857f1864458dff5612cab9a6a0b150ac117b88c3e04",
    WORKTREE / "probes/row_feedback/runtime.py": "94e9a2f472af2a1e60f06f215f72a2a362c9b2773918f98a7f85f983f1ef57f6",
}
F_FINGERPRINT = "3d9b4063ff9728b05828d742bf41cfdc59d60ee52e236e3f8dd323b1e0e7683e"


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


def append_log(handle: Any, event: str, payload: Mapping[str, Any]) -> None:
    handle.write(event + " " + json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def terminate(process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + KILL_GRACE_SECONDS
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=55.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        remaining = max(0.0, deadline - time.monotonic())
        process.wait(timeout=remaining)


def aggregate_case_counters(receipt: Mapping[str, Any]) -> dict[str, int]:
    totals = {
        "cases": 0,
        "donor_replays": 0,
        "generations": 0,
        "visible_generated_tokens": 0,
        "internal_slots": 0,
        "model_forwards": 0,
        "image_forwards": 0,
    }
    for binding in receipt.get("case_receipts", []):
        case_path = OUTPUT / binding["path"]
        if not case_path.is_file() or sha256(case_path) != binding["sha256"]:
            continue
        case = read_json(case_path)
        totals["cases"] += 1
        totals["donor_replays"] += 1
        for arm in ("correct_f", "exact_self_replay", "wrong_owner"):
            result = case["arms"][arm]
            totals["generations"] += 1
            totals["visible_generated_tokens"] += int(result["visible_generated_tokens"])
            totals["internal_slots"] += int(result["internal_slot_count"])
            totals["model_forwards"] += int(result["model_forwards"])
            totals["image_forwards"] += int(result["image_forwards"])
    return totals


def main() -> int:
    for path in (OUTPUT, LOG, LAUNCH_RECEIPT, TERMINAL_RECEIPT):
        if path.exists():
            raise SystemExit(f"refuse preexisting content artifact: {path}")
    actual = {str(path): sha256(path) for path in EXPECTED}
    mismatches = [str(path) for path, expected in EXPECTED.items() if actual[str(path)] != expected]
    if mismatches:
        raise SystemExit(f"frozen launch binding changed: {mismatches}")
    acceptance = read_json(FIT_ACCEPTANCE)
    binding = read_json(F_BINDING)
    authorization = read_json(RETRY_AUTHORIZATION)
    if acceptance.get("status") != "lead_accepted_fixed64_paired_fit":
        raise SystemExit("paired fit is not lead accepted")
    if binding.get("arm") != "F" or binding.get("adapter", {}).get("fingerprint") != F_FINGERPRINT:
        raise SystemExit("canonical F binding changed")
    if Path(binding["adapter"]["root"]).resolve() != ADAPTER.resolve():
        raise SystemExit("canonical F adapter path changed")
    if (
        authorization.get("physical_gpu") != int(PHYSICAL_GPU)
        or authorization.get("retry_wait_timeout_seconds") != int(TIMEOUT_SECONDS)
        or Path(authorization["output"]).resolve() != OUTPUT.resolve()
        or authorization.get("automatic_retries") != 0
    ):
        raise SystemExit("retry authorization contract changed")
    for declared in binding["adapter"]["files"]:
        path = ADAPTER / declared["relative_path"]
        if path.stat().st_size != declared["size_bytes"] or sha256(path) != declared["sha256"]:
            raise SystemExit(f"canonical F adapter payload changed: {path}")

    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True,
    ).strip()
    command = [
        sys.executable,
        "-m",
        "probes.row_feedback.content_run",
        "--packet",
        str(PACKET),
        "--adapter",
        str(ADAPTER),
        "--output",
        str(OUTPUT),
        "--device",
        "cuda:0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
    env["PYTHONUNBUFFERED"] = "1"
    started_unix = time.time()
    started = time.monotonic()
    timed_out = False
    with LOG.open("x", encoding="utf-8", buffering=1) as log:
        append_log(log, "CONTENT_RUN_LAUNCHING", {
            "command": command,
            "cwd": str(WORKTREE),
            "cuda_visible_devices": PHYSICAL_GPU,
            "timeout_seconds": TIMEOUT_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS,
        })
        process = subprocess.Popen(
            command,
            cwd=WORKTREE,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        launch = {
            "schema": "row_feedback.content_launch_receipt.v1",
            "status": "launched_waiting",
            "launcher_pid": os.getpid(),
            "producer_pid": process.pid,
            "physical_gpu": int(PHYSICAL_GPU),
            "command": command,
            "cwd": str(WORKTREE),
            "timeout_seconds": TIMEOUT_SECONDS,
            "kill_grace_seconds": KILL_GRACE_SECONDS,
            "automatic_retries": False,
            "started_unix": started_unix,
            "output": str(OUTPUT),
            "log": str(LOG),
            "terminal_receipt": str(TERMINAL_RECEIPT),
            "current_git_head": current_head,
            "launcher_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "bindings": {
                "packet": {"path": str(PACKET), "sha256": EXPECTED[PACKET]},
                "fit_acceptance": {"path": str(FIT_ACCEPTANCE), "sha256": EXPECTED[FIT_ACCEPTANCE]},
                "F_fit_binding": {"path": str(F_BINDING), "sha256": EXPECTED[F_BINDING]},
                "retry_authorization": {
                    "path": str(RETRY_AUTHORIZATION),
                    "sha256": EXPECTED[RETRY_AUTHORIZATION],
                },
                "adapter": binding["adapter"],
                "code": {
                    str(path): {"sha256": actual[str(path)]}
                    for path in EXPECTED
                    if str(path).startswith(str(WORKTREE))
                },
            },
        }
        write_exclusive(LAUNCH_RECEIPT, launch)
        append_log(log, "CONTENT_RUN_STARTED", {
            "producer_pid": process.pid,
            "launch_receipt": str(LAUNCH_RECEIPT),
            "launch_receipt_sha256": sha256(LAUNCH_RECEIPT),
        })
        try:
            exit_code = process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            append_log(log, "CONTENT_RUN_TIMEOUT", {"producer_pid": process.pid})
            terminate(process)
            exit_code = process.returncode

        finished_unix = time.time()
        run_receipt_path = OUTPUT / "receipt.json"
        run_receipt = read_json(run_receipt_path) if run_receipt_path.is_file() else {}
        counters = aggregate_case_counters(run_receipt)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        success = (
            exit_code == 0
            and not timed_out
            and run_receipt.get("status") in {
                "completed_non_gating_content_diagnostic",
                "completed_with_technical_invalid_cases",
            }
            and run_receipt.get("case_count") == 3
            and counters["cases"] == 3
            and counters["donor_replays"] == 3
            and counters["generations"] == 9
        )
        terminal = {
            "schema": "row_feedback.content_terminal_receipt.v1",
            "status": "complete" if success else "failed",
            "claim_boundary": "Exposed non-gating diagnostic execution only; no native-capacity or architecture-quality conclusion.",
            "launcher_pid": os.getpid(),
            "producer_pid": process.pid,
            "physical_gpu": int(PHYSICAL_GPU),
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "started_unix": started_unix,
            "finished_unix": finished_unix,
            "wall_seconds": time.monotonic() - started,
            "allocated_gpu_hours": (time.monotonic() - started) / 3600.0,
            "maximum_allocated_gpu_hours": TIMEOUT_SECONDS / 3600.0,
            "resources": {
                "child_user_cpu_seconds": usage.ru_utime,
                "child_system_cpu_seconds": usage.ru_stime,
                "peak_child_rss_bytes": int(usage.ru_maxrss) * 1024,
            },
            "counters": counters,
            "content_receipt": None if not run_receipt_path.is_file() else {
                "path": str(run_receipt_path),
                "sha256": sha256(run_receipt_path),
                "status": run_receipt.get("status"),
                "loaded_identity": run_receipt.get("loaded_identity"),
                "technical_invalid_cases": run_receipt.get("technical_invalid_cases"),
            },
            "launch_receipt": {"path": str(LAUNCH_RECEIPT), "sha256": sha256(LAUNCH_RECEIPT)},
            "future_completion_caveat": "W feedback is captured after h+c+w and transplanted to C; it is diagnostic, not a deployable causal write.",
        }
        write_exclusive(TERMINAL_RECEIPT, terminal)
        append_log(log, "CONTENT_RUN_COMPLETE" if success else "CONTENT_RUN_FAILED", {
            "status": terminal["status"],
            "exit_code": exit_code,
            "terminal_receipt": str(TERMINAL_RECEIPT),
            "terminal_receipt_sha256": sha256(TERMINAL_RECEIPT),
            "counters": counters,
        })
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
