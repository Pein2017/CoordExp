#!/usr/bin/env python3
"""One-shot Wave 7 parent interrupter for exact same-world-size resume evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts.training_state import (  # noqa: E402
    TrainingStateManifest,
    load_training_state_manifest,
)
from src.config.loader import load_train_config  # noqa: E402
from src.config.paths import resolve_run_directory  # noqa: E402
from src.qwen.parity import (  # noqa: E402
    ParityContractError,
    assert_absent_artifact_target,
    canonical_json_bytes,
    write_strict_json_atomic,
)


MARKER_SCHEMA = "coordexp-swift-wave7-interrupt-marker-v1"
RECEIPT_SCHEMA = "coordexp-swift-wave7-interrupt-receipt-v1"
EXPECTED_WORLD_SIZE = 8
MAX_ARGV_ITEMS = 512
MAX_ARG_CHARS = 16_384
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_GRAPH_PROCESSES = 4_096
MAX_TREE_ENTRIES = 100_000
MAX_OBSERVED_FILE_BYTES = 16 * 1024 * 1024
PR_SET_CHILD_SUBREAPER = 36
_TERMINAL_STATES = frozenset({"Z", "X", "x"})


class Wave7InterruptError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.context = dict(context or {})
        super().__init__(message)


class ProcessGraphOverflow(Wave7InterruptError):
    def __init__(
        self,
        *,
        graph: tuple[ProcessRecord, ...],
        discovered: tuple[ProcessRecord, ...],
        observed: int,
    ) -> None:
        self.graph = graph
        self.discovered = discovered
        super().__init__(
            "expanded process graph exceeds the bound",
            code="wave7.process_inventory",
            context={"maximum": MAX_GRAPH_PROCESSES, "observed": observed},
        )


@dataclass(frozen=True)
class ControllerRequest:
    launcher_argv: tuple[str, ...]
    parent_run_dir: Path
    expected_checkpoint_step: int
    marker: Path
    receipt: Path
    timeout_seconds: float
    term_grace_seconds: float
    kill_grace_seconds: float
    stability_seconds: float
    poll_seconds: float
    sources: tuple[Path, ...]
    configs: tuple[Path, ...]


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    pgid: int
    session_id: int
    state: str
    start_time_ticks: int
    depth: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_loads(encoded: bytes, *, owner: str) -> Any:
    if len(encoded) > MAX_JSON_BYTES:
        raise Wave7InterruptError(
            f"{owner} exceeds the JSON size bound",
            code="wave7.json_oversize",
            context={"owner": owner, "size": len(encoded)},
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant: {value}")

    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise Wave7InterruptError(
            f"{owner} is not strict JSON",
            code="wave7.json_malformed",
            context={"owner": owner, "error_type": type(exc).__name__},
        ) from exc


def _load_launcher_json(path: Path) -> tuple[str, ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Wave7InterruptError(
            "launcher JSON is unreadable",
            code="wave7.launcher_json",
            context={"path": str(path)},
        ) from exc
    value = _strict_json_loads(raw, owner="launcher argv")
    if not isinstance(value, list):
        raise Wave7InterruptError(
            "launcher JSON must contain one argv list", code="wave7.launcher_argv"
        )
    return _validate_argv(value)


def _validate_argv(value: Sequence[Any]) -> tuple[str, ...]:
    if not value or len(value) > MAX_ARGV_ITEMS:
        raise Wave7InterruptError(
            "launcher argv count is invalid",
            code="wave7.launcher_argv",
            context={"count": len(value)},
        )
    result: list[str] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, str)
            or not item
            or len(item) > MAX_ARG_CHARS
            or "\x00" in item
        ):
            raise Wave7InterruptError(
                "launcher argv contains an invalid item",
                code="wave7.launcher_argv",
                context={"index": index},
            )
        result.append(item)
    return tuple(result)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not (parsed > 0.0 and parsed < 24 * 60 * 60):
        raise argparse.ArgumentTypeError("must be positive and less than one day")
    return parsed


def parse_request(argv: Sequence[str] | None = None) -> ControllerRequest:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-json", type=Path)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, default=3)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=_positive_float, required=True)
    parser.add_argument("--term-grace-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--kill-grace-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--stability-seconds", type=_positive_float, default=1.0)
    parser.add_argument("--poll-seconds", type=_positive_float, default=0.05)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("launcher", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    remainder = list(args.launcher)
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if (args.launcher_json is None) == (not remainder):
        parser.error(
            "provide exactly one of --launcher-json or remainder launcher argv"
        )
    launcher = (
        _load_launcher_json(args.launcher_json.expanduser().resolve())
        if args.launcher_json is not None
        else _validate_argv(remainder)
    )
    if args.expected_checkpoint_step <= 0:
        parser.error("--expected-checkpoint-step must be positive")
    parent = args.parent_run_dir.expanduser().resolve()
    marker = args.marker.expanduser().resolve(strict=False)
    receipt = args.receipt.expanduser().resolve(strict=False)
    if marker == receipt:
        parser.error("marker and receipt targets must differ")
    return ControllerRequest(
        launcher_argv=launcher,
        parent_run_dir=parent,
        expected_checkpoint_step=args.expected_checkpoint_step,
        marker=marker,
        receipt=receipt,
        timeout_seconds=args.timeout_seconds,
        term_grace_seconds=args.term_grace_seconds,
        kill_grace_seconds=args.kill_grace_seconds,
        stability_seconds=args.stability_seconds,
        poll_seconds=args.poll_seconds,
        sources=tuple(path.expanduser().resolve() for path in args.source),
        configs=tuple(path.expanduser().resolve() for path in args.config),
    )


def _identity_files(paths: Sequence[Path], *, owner: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise Wave7InterruptError(
                f"{owner} identity file is unreadable",
                code="wave7.identity_file",
                context={"path": str(path)},
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise Wave7InterruptError(
                f"{owner} identity must be a non-symlink regular file",
                code="wave7.identity_file",
                context={"path": str(path)},
            )
        rows.append(
            {"path": str(path), "sha256": _sha256_file(path), "size": metadata.st_size}
        )
    return rows


def _enable_subreaper() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (AttributeError, OSError) as exc:
        raise Wave7InterruptError(
            "child-subreaper support is unavailable", code="wave7.subreaper"
        ) from exc
    if result != 0:
        error_number = ctypes.get_errno()
        raise Wave7InterruptError(
            "child-subreaper activation failed",
            code="wave7.subreaper",
            context={"errno": error_number},
        )


def _read_process(pid: int) -> ProcessRecord | None:
    try:
        encoded = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, UnicodeError) as exc:
        raise Wave7InterruptError(
            "process identity is unreadable",
            code="wave7.process_inventory",
            context={"pid": pid},
        ) from exc
    close = encoded.rfind(")")
    fields = encoded[close + 2 :].split() if close >= 0 else []
    if len(fields) <= 19:
        raise Wave7InterruptError(
            "process identity is malformed",
            code="wave7.process_inventory",
            context={"pid": pid},
        )
    try:
        return ProcessRecord(
            pid=pid,
            ppid=int(fields[1]),
            pgid=int(fields[2]),
            session_id=int(fields[3]),
            state=fields[0],
            start_time_ticks=int(fields[19]),
            depth=0,
        )
    except (IndexError, ValueError) as exc:
        raise Wave7InterruptError(
            "process identity is malformed",
            code="wave7.process_inventory",
            context={"pid": pid},
        ) from exc


def _process_snapshot() -> dict[int, ProcessRecord]:
    snapshot: dict[int, ProcessRecord] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise Wave7InterruptError(
            "process inventory is unavailable", code="wave7.process_inventory"
        ) from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        record = _read_process(int(entry.name))
        if record is not None:
            snapshot[record.pid] = record
    return snapshot


def _capture_process_graph(root_pid: int) -> tuple[ProcessRecord, ...]:
    snapshot = _process_snapshot()
    root = snapshot.get(root_pid)
    if root is None:
        raise Wave7InterruptError(
            "launcher exited before its process graph was captured",
            code="wave7.launcher_exited",
            context={"pid": root_pid},
        )
    children: dict[int, list[int]] = defaultdict(list)
    for record in snapshot.values():
        children[record.ppid].append(record.pid)
    pending = [(root_pid, 0)]
    seen: set[int] = set()
    result: list[ProcessRecord] = []
    while pending:
        pid, depth = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        record = snapshot.get(pid)
        if record is None:
            continue
        result.append(replace(record, depth=depth))
        if len(result) > MAX_GRAPH_PROCESSES:
            raise Wave7InterruptError(
                "launcher process graph exceeds the bound",
                code="wave7.process_inventory",
                context={"maximum": MAX_GRAPH_PROCESSES},
            )
        pending.extend((child, depth + 1) for child in children.get(pid, []))
    return tuple(sorted(result, key=lambda item: (item.depth, item.pid)))


def _remaining_captured(
    graph: tuple[ProcessRecord, ...],
) -> tuple[list[dict[str, Any]], list[int]]:
    remaining_pids: list[dict[str, Any]] = []
    for captured in graph:
        current = _read_process(captured.pid)
        if current is None:
            continue
        remaining_pids.append(
            {
                "identity_match": current.start_time_ticks == captured.start_time_ticks,
                "pid": captured.pid,
                "state": current.state,
            }
        )
    remaining_pgids: list[int] = []
    for pgid in sorted({item.pgid for item in graph}):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        remaining_pgids.append(pgid)
    return remaining_pids, remaining_pgids


def _controller_child_pids() -> dict[int, int]:
    controller_pid = os.getpid()
    return {
        record.pid: record.start_time_ticks
        for record in _process_snapshot().values()
        if record.ppid == controller_pid
    }


def _expand_related_processes(
    graph: tuple[ProcessRecord, ...],
    *,
    baseline_controller_children: Mapping[int, int],
) -> tuple[tuple[ProcessRecord, ...], tuple[ProcessRecord, ...]]:
    snapshot = _process_snapshot()
    known = {record.pid: record for record in graph}
    if len(known) > MAX_GRAPH_PROCESSES:
        bounded = tuple(
            sorted(known.values(), key=lambda item: (item.depth, item.pid))[
                :MAX_GRAPH_PROCESSES
            ]
        )
        raise ProcessGraphOverflow(
            graph=bounded,
            discovered=(),
            observed=len(known),
        )
    discovered: dict[int, ProcessRecord] = {}
    changed = True
    while changed:
        changed = False
        for current in sorted(snapshot.values(), key=lambda item: item.pid):
            if current.pid in known or current.pid in discovered:
                continue
            parent = discovered.get(current.ppid)
            if parent is None:
                captured_parent = known.get(current.ppid)
                live_parent = snapshot.get(current.ppid)
                if (
                    captured_parent is not None
                    and live_parent is not None
                    and live_parent.start_time_ticks == captured_parent.start_time_ticks
                    and live_parent.session_id == captured_parent.session_id
                ):
                    parent = captured_parent
            adopted = (
                current.ppid == os.getpid()
                and baseline_controller_children.get(current.pid)
                != current.start_time_ticks
            )
            if parent is None and not adopted:
                continue
            depth = (
                parent.depth + 1
                if parent is not None
                else max((item.depth for item in known.values()), default=0) + 1
            )
            observed = len(known) + len(discovered) + 1
            if observed > MAX_GRAPH_PROCESSES:
                bounded_discovered = tuple(
                    sorted(discovered.values(), key=lambda item: (item.depth, item.pid))
                )
                bounded_graph = tuple(
                    sorted(
                        (*known.values(), *bounded_discovered),
                        key=lambda item: (item.depth, item.pid),
                    )
                )
                raise ProcessGraphOverflow(
                    graph=bounded_graph,
                    discovered=bounded_discovered,
                    observed=observed,
                )
            discovered[current.pid] = replace(current, depth=depth)
            changed = True
    combined = tuple(
        sorted(
            (*known.values(), *discovered.values()),
            key=lambda item: (item.depth, item.pid),
        )
    )
    return combined, tuple(
        sorted(discovered.values(), key=lambda item: (item.depth, item.pid))
    )


def _reap_children(process: subprocess.Popen[Any]) -> list[dict[str, int]]:
    process.poll()
    reaped: list[dict[str, int]] = []
    while True:
        try:
            pid, status_value = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        reaped.append({"pid": pid, "wait_status": status_value})
    return reaped


def _signal_graph(
    graph: tuple[ProcessRecord, ...],
    signum: int,
    *,
    events: list[dict[str, Any]],
) -> None:
    signal_name = signal.Signals(signum).name
    controller_pgid = os.getpgrp()
    snapshot = _process_snapshot()
    known = {record.pid: record for record in graph}
    group_depth: dict[int, int] = {}
    unproven_groups: set[int] = set()
    for record in graph:
        if record.pgid == controller_pgid:
            continue
        members = [item for item in snapshot.values() if item.pgid == record.pgid]
        leader = known.get(record.pgid)
        proven = (
            leader is not None
            and all(
                member.pid in known
                and member.start_time_ticks == known[member.pid].start_time_ticks
                for member in members
            )
            and snapshot.get(leader.pid) is not None
            and snapshot[leader.pid].start_time_ticks == leader.start_time_ticks
        )
        if proven:
            group_depth[record.pgid] = min(
                item.depth for item in graph if item.pgid == record.pgid
            )
        else:
            unproven_groups.add(record.pgid)

    for depth in sorted({item.depth for item in graph}, reverse=True):
        for captured in sorted(
            (item for item in graph if item.depth == depth),
            key=lambda item: -item.pid,
        ):
            current = _read_process(captured.pid)
            if (
                current is None
                or current.start_time_ticks != captured.start_time_ticks
                or current.state in _TERMINAL_STATES
            ):
                outcome = "absent_or_terminal"
            else:
                try:
                    os.kill(captured.pid, signum)
                    outcome = "sent"
                except ProcessLookupError:
                    outcome = "absent"
            events.append(
                {
                    "depth": captured.depth,
                    "kind": f"signal_pid_{signal_name.lower()}",
                    "monotonic": time.monotonic(),
                    "outcome": outcome,
                    "pid": captured.pid,
                    "timestamp": _utc_now(),
                }
            )
        for pgid in sorted(
            (
                group
                for group, minimum_depth in group_depth.items()
                if minimum_depth == depth
            ),
            reverse=True,
        ):
            try:
                os.killpg(pgid, signum)
                outcome = "sent"
            except ProcessLookupError:
                outcome = "absent"
            events.append(
                {
                    "depth": depth,
                    "kind": f"signal_group_{signal_name.lower()}",
                    "monotonic": time.monotonic(),
                    "outcome": outcome,
                    "pgid": pgid,
                    "timestamp": _utc_now(),
                }
            )
    for pgid in sorted(unproven_groups):
        events.append(
            {
                "depth": None,
                "kind": f"signal_group_{signal_name.lower()}",
                "monotonic": time.monotonic(),
                "outcome": "unproven_identity",
                "pgid": pgid,
                "timestamp": _utc_now(),
            }
        )


def _wait_for_absence(
    process: subprocess.Popen[Any],
    graph: tuple[ProcessRecord, ...],
    *,
    seconds: float,
    poll_seconds: float,
    signum: int,
    events: list[dict[str, Any]],
    baseline_controller_children: Mapping[int, int],
) -> tuple[
    tuple[ProcessRecord, ...],
    list[dict[str, Any]],
    list[int],
    list[dict[str, int]],
    list[dict[str, Any]],
]:
    deadline = time.monotonic() + seconds
    all_reaped: list[dict[str, int]] = []
    cleanup_errors: list[dict[str, Any]] = []
    current_graph = graph
    while True:
        try:
            current_graph, discovered = _expand_related_processes(
                current_graph,
                baseline_controller_children=baseline_controller_children,
            )
        except ProcessGraphOverflow as exc:
            current_graph = exc.graph
            discovered = exc.discovered
            if not cleanup_errors:
                cleanup_errors.append(_error_record(exc))
        if discovered:
            _signal_graph(discovered, signum, events=events)
        all_reaped.extend(_reap_children(process))
        remaining_pids, remaining_pgids = _remaining_captured(current_graph)
        if not remaining_pids and not remaining_pgids:
            return (
                current_graph,
                remaining_pids,
                remaining_pgids,
                all_reaped,
                cleanup_errors,
            )
        if time.monotonic() >= deadline:
            return (
                current_graph,
                remaining_pids,
                remaining_pgids,
                all_reaped,
                cleanup_errors,
            )
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _terminate_captured(
    process: subprocess.Popen[Any],
    graph: tuple[ProcessRecord, ...],
    *,
    request: ControllerRequest,
    capture_completed_monotonic: float,
    events: list[dict[str, Any]],
    baseline_controller_children: Mapping[int, int],
) -> dict[str, Any]:
    started = time.monotonic()
    initial_graph = graph
    _signal_graph(graph, signal.SIGTERM, events=events)
    graph, remaining_pids, remaining_pgids, reaped, cleanup_errors = _wait_for_absence(
        process,
        graph,
        seconds=request.term_grace_seconds,
        poll_seconds=request.poll_seconds,
        signum=signal.SIGTERM,
        events=events,
        baseline_controller_children=baseline_controller_children,
    )
    if remaining_pids or remaining_pgids:
        _signal_graph(graph, signal.SIGKILL, events=events)
        graph, final_pids, final_pgids, killed_reaped, killed_errors = (
            _wait_for_absence(
                process,
                graph,
                seconds=request.kill_grace_seconds,
                poll_seconds=request.poll_seconds,
                signum=signal.SIGKILL,
                events=events,
                baseline_controller_children=baseline_controller_children,
            )
        )
        remaining_pids, remaining_pgids = final_pids, final_pgids
        reaped.extend(killed_reaped)
        if not cleanup_errors:
            cleanup_errors.extend(killed_errors)
    process.poll()
    return {
        "capture_completed_monotonic": capture_completed_monotonic,
        "captured_process_graph": [asdict(item) for item in graph],
        "cleanup_errors": cleanup_errors,
        "captured_pgids": sorted({item.pgid for item in initial_graph}),
        "captured_pids": [item.pid for item in initial_graph],
        "duration_seconds": max(0.0, time.monotonic() - started),
        "launcher_exited": process.poll() is not None,
        "launcher_returncode": process.poll(),
        "reaped": reaped,
        "remaining_pgids": remaining_pgids,
        "remaining_pids": remaining_pids,
        "post_marker_discovered_pids": sorted(
            {item.pid for item in graph} - {item.pid for item in initial_graph}
        ),
    }


def _manifest_path(request: ControllerRequest) -> Path:
    return (
        request.parent_run_dir
        / "checkpoints"
        / f"step-{request.expected_checkpoint_step}"
        / "training_state"
        / "manifest.json"
    )


def _parse_manifest_bytes(
    encoded: bytes, *, request: ControllerRequest
) -> TrainingStateManifest:
    try:
        value = _strict_json_loads(encoded, owner="training-state manifest")
    except Wave7InterruptError as exc:
        raise Wave7InterruptError(
            "training-state manifest is malformed or uncommitted",
            code="wave7.manifest_malformed",
            context={"error_code": exc.code},
        ) from exc
    if not isinstance(value, Mapping):
        raise Wave7InterruptError(
            "training-state manifest must be an object",
            code="wave7.manifest_malformed",
        )
    try:
        manifest = TrainingStateManifest.from_dict(value)
    except BaseException as exc:
        raise Wave7InterruptError(
            "training-state manifest is malformed or uncommitted",
            code="wave7.manifest_malformed",
            context={"error_code": str(getattr(exc, "code", type(exc).__name__))},
        ) from exc
    mismatches: dict[str, Any] = {}
    if manifest.checkpoint_step != request.expected_checkpoint_step:
        mismatches["checkpoint_step"] = manifest.checkpoint_step
    if manifest.world_size != EXPECTED_WORLD_SIZE:
        mismatches["world_size"] = manifest.world_size
    if tuple(rank.rank for rank in manifest.ranks) != tuple(range(EXPECTED_WORLD_SIZE)):
        mismatches["ranks"] = [rank.rank for rank in manifest.ranks]
    if mismatches:
        raise Wave7InterruptError(
            "training-state manifest does not identify the expected exact boundary",
            code="wave7.manifest_mismatch",
            context=mismatches,
        )
    return manifest


def _checkpoint_publication_event(
    encoded: bytes,
    *,
    request: ControllerRequest,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = _strict_json_loads(encoded, owner="parent run.json")
    if not isinstance(value, Mapping):
        raise Wave7InterruptError(
            "parent run.json must be an object",
            code="wave7.checkpoint_event_malformed",
        )
    measurement = value.get("measurement")
    if measurement is None:
        return None
    if not isinstance(measurement, Mapping):
        raise Wave7InterruptError(
            "parent run measurement must be an object",
            code="wave7.checkpoint_event_malformed",
        )
    events = measurement.get("checkpoint_publication_events")
    if events is None:
        return None
    if not isinstance(events, list) or any(
        not isinstance(event, Mapping) for event in events
    ):
        raise Wave7InterruptError(
            "checkpoint publication events must be an object list",
            code="wave7.checkpoint_event_malformed",
        )
    matching = [
        dict(event)
        for event in events
        if event.get("step") == request.expected_checkpoint_step
        and not isinstance(event.get("step"), bool)
    ]
    if not matching:
        if any(
            isinstance(event.get("step"), int)
            and not isinstance(event.get("step"), bool)
            and int(event["step"]) > request.expected_checkpoint_step
            for event in events
        ):
            raise Wave7InterruptError(
                "checkpoint publication events skipped the expected step",
                code="wave7.checkpoint_event_mismatch",
            )
        return None
    if len(matching) != 1:
        raise Wave7InterruptError(
            "checkpoint publication event step is duplicated",
            code="wave7.checkpoint_event_mismatch",
            context={"count": len(matching)},
        )
    event = matching[0]
    if event.get("status") == "failed":
        raise Wave7InterruptError(
            "expected checkpoint publication reported failure",
            code="wave7.checkpoint_event_failed",
            context={"failure_code": event.get("failure_code")},
        )
    checkpoint_dir = Path(str(checkpoint["path"])).parents[1].resolve()
    expected_identity = {
        "checkpoint_step": request.expected_checkpoint_step,
        "resolved_path": str(checkpoint_dir),
        "training_state_aggregate_digest": checkpoint["aggregate_digest"],
        "training_state_manifest_file_sha256": checkpoint["file_sha256"],
    }
    mismatches: dict[str, Any] = {}
    expected_path = f"checkpoints/step-{request.expected_checkpoint_step}"
    if event.get("status") != "completed":
        mismatches["status"] = event.get("status")
    if event.get("checkpoint_path") != expected_path:
        mismatches["checkpoint_path"] = event.get("checkpoint_path")
    if event.get("exact_training_state_enabled") is not True:
        mismatches["exact_training_state_enabled"] = event.get(
            "exact_training_state_enabled"
        )
    if event.get("failure_code") is not None:
        mismatches["failure_code"] = event.get("failure_code")
    if event.get("checkpoint_identity") != expected_identity:
        mismatches["checkpoint_identity"] = event.get("checkpoint_identity")
    if mismatches:
        raise Wave7InterruptError(
            "checkpoint publication event does not match the committed manifest",
            code="wave7.checkpoint_event_mismatch",
            context=mismatches,
        )
    return event


def _read_boundary_files(request: ControllerRequest) -> tuple[bytes, bytes]:
    manifest_path = _manifest_path(request)
    run_path = request.parent_run_dir / "run.json"
    try:
        return manifest_path.read_bytes(), run_path.read_bytes()
    except OSError as exc:
        raise Wave7InterruptError(
            "durable checkpoint boundary files are unreadable",
            code="wave7.checkpoint_event_changed",
            context={"error_type": type(exc).__name__},
        ) from exc


def _revalidate_durable_boundary(
    request: ControllerRequest,
    *,
    checkpoint: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> None:
    manifest_bytes, run_bytes = _read_boundary_files(request)
    manifest = _parse_manifest_bytes(manifest_bytes, request=request)
    try:
        event = _checkpoint_publication_event(
            run_bytes,
            request=request,
            checkpoint=checkpoint,
        )
    except Wave7InterruptError as exc:
        raise Wave7InterruptError(
            "durable checkpoint boundary changed after admission",
            code="wave7.checkpoint_event_changed",
            context={"error_code": exc.code},
        ) from exc
    if (
        _sha256_bytes(manifest_bytes) != checkpoint["file_sha256"]
        or manifest.aggregate_digest != checkpoint["aggregate_digest"]
        or _sha256_bytes(run_bytes) != publication["run_file_sha256"]
        or event != publication["event"]
    ):
        raise Wave7InterruptError(
            "durable checkpoint boundary changed after admission",
            code="wave7.checkpoint_event_changed",
        )


def _wait_for_durable_boundary(
    process: subprocess.Popen[Any], request: ControllerRequest
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _manifest_path(request)
    run_path = request.parent_run_dir / "run.json"
    deadline = time.monotonic() + request.timeout_seconds
    manifest_observed = False
    while True:
        if (path.exists() or path.is_symlink()) and (
            run_path.exists() or run_path.is_symlink()
        ):
            try:
                first_manifest_bytes = path.read_bytes()
                first_run_bytes = run_path.read_bytes()
            except OSError as exc:
                raise Wave7InterruptError(
                    "durable checkpoint boundary is unreadable",
                    code="wave7.manifest_malformed",
                    context={"path": str(path)},
                ) from exc
            first_manifest = _parse_manifest_bytes(
                first_manifest_bytes, request=request
            )
            manifest_observed = True
            checkpoint = {
                "aggregate_digest": first_manifest.aggregate_digest,
                "checkpoint_step": first_manifest.checkpoint_step,
                "file_sha256": _sha256_bytes(first_manifest_bytes),
                "path": str(path),
                "rank_count": len(first_manifest.ranks),
                "size": len(first_manifest_bytes),
                "world_size": first_manifest.world_size,
            }
            event = _checkpoint_publication_event(
                first_run_bytes,
                request=request,
                checkpoint=checkpoint,
            )
            if event is not None:
                time.sleep(request.poll_seconds)
                try:
                    second_manifest_bytes = path.read_bytes()
                    second_run_bytes = run_path.read_bytes()
                except OSError:
                    second_manifest_bytes = b""
                    second_run_bytes = b""
            else:
                second_manifest_bytes = b""
                second_run_bytes = b""
            if (
                event is not None
                and first_manifest_bytes == second_manifest_bytes
                and first_run_bytes == second_run_bytes
            ):
                checkpoint_dir = path.parents[1]
                try:
                    reloaded = load_training_state_manifest(checkpoint_dir)
                except BaseException as exc:
                    raise Wave7InterruptError(
                        "training-state manifest did not reload strictly",
                        code="wave7.manifest_malformed",
                        context={
                            "error_code": str(getattr(exc, "code", type(exc).__name__))
                        },
                    ) from exc
                third_manifest_bytes, third_run_bytes = _read_boundary_files(request)
                if (
                    third_manifest_bytes == first_manifest_bytes
                    and third_run_bytes == first_run_bytes
                    and reloaded.aggregate_digest == first_manifest.aggregate_digest
                ):
                    return checkpoint, {
                        "event": event,
                        "run_file_sha256": _sha256_bytes(first_run_bytes),
                        "run_path": str(run_path),
                        "run_size": len(first_run_bytes),
                    }
        if process.poll() is not None:
            raise Wave7InterruptError(
                "launcher exited before the committed checkpoint became durable",
                code="wave7.launcher_exited",
                context={"returncode": process.returncode},
            )
        if time.monotonic() >= deadline:
            if manifest_observed:
                raise Wave7InterruptError(
                    "timed out waiting for the matching checkpoint publication event",
                    code="wave7.checkpoint_event_timeout",
                    context={"path": str(run_path)},
                )
            raise Wave7InterruptError(
                "timed out waiting for the committed checkpoint",
                code="wave7.checkpoint_timeout",
                context={"path": str(path)},
            )
        time.sleep(request.poll_seconds)


def _snapshot_run_tree(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise Wave7InterruptError(
            "parent run directory is unavailable",
            code="wave7.parent_run",
            context={"path": str(run_dir)},
        )
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if len(rows) >= MAX_TREE_ENTRIES:
            raise Wave7InterruptError(
                "parent run tree exceeds the stability inventory bound",
                code="wave7.run_tree_oversize",
            )
        metadata = path.lstat()
        relative = path.relative_to(run_dir).as_posix()
        if stat.S_ISREG(metadata.st_mode):
            kind = "file"
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
        else:
            kind = "other"
        row: dict[str, Any] = {
            "inode": metadata.st_ino,
            "kind": kind,
            "mode": stat.S_IMODE(metadata.st_mode),
            "mtime_ns": metadata.st_mtime_ns,
            "path": relative,
            "size": metadata.st_size,
        }
        if (
            kind == "file"
            and metadata.st_size <= MAX_OBSERVED_FILE_BYTES
            and (
                relative in {"run.json", "logging.jsonl"}
                or relative.endswith("/training_state/manifest.json")
                or relative == "checkpoints/final.json"
            )
        ):
            row["sha256"] = _sha256_file(path)
        rows.append(row)
    return {
        "entry_count": len(rows),
        "fingerprint": _sha256_bytes(canonical_json_bytes(rows)),
        "rows": rows,
    }


def _max_logged_train_step(path: Path) -> int:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise Wave7InterruptError(
            "parent logging artifact is unreadable",
            code="wave7.logging",
            context={"path": str(path)},
        ) from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise Wave7InterruptError(
            "parent logging artifact exceeds the bound", code="wave7.logging"
        )
    maximum = 0
    for index, line in enumerate(encoded.splitlines(), start=1):
        if not line.strip():
            continue
        value = _strict_json_loads(line, owner=f"logging row {index}")
        if not isinstance(value, Mapping):
            raise Wave7InterruptError(
                "parent logging row must be an object", code="wave7.logging"
            )
        if value.get("split") != "train":
            continue
        step = value.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise Wave7InterruptError(
                "parent train logging step is invalid", code="wave7.logging"
            )
        maximum = max(maximum, step)
    return maximum


def _run_status(path: Path) -> str:
    try:
        value = _strict_json_loads(path.read_bytes(), owner="parent run.json")
    except OSError as exc:
        raise Wave7InterruptError(
            "parent run.json is unreadable", code="wave7.run_status"
        ) from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("status"), str):
        raise Wave7InterruptError(
            "parent run status is malformed", code="wave7.run_status"
        )
    return value["status"]


def _sample_nvidia_compute_apps() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Wave7InterruptError(
            "nvidia compute-process inventory is unavailable",
            code="wave7.nvidia_inventory",
        ) from exc
    if result.returncode != 0:
        raise Wave7InterruptError(
            "nvidia compute-process inventory failed",
            code="wave7.nvidia_inventory",
            context={"returncode": result.returncode},
        )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if (
            len(fields) != 3
            or not fields[0]
            or not fields[1].isdigit()
            or not fields[2]
        ):
            raise Wave7InterruptError(
                "nvidia compute-process inventory is malformed",
                code="wave7.nvidia_inventory",
            )
        rows.append(
            {"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": fields[2]}
        )
    return rows


def _error_record(exc: BaseException) -> dict[str, Any]:
    context = getattr(exc, "context", {})
    try:
        safe_context = json.loads(json.dumps(context, allow_nan=False, default=str))
    except (TypeError, ValueError):
        safe_context = {"unavailable": True}
    return {
        "code": str(getattr(exc, "code", "wave7.unexpected"))[:128],
        "context": safe_context,
        "message": str(exc)[:2_048],
        "type": type(exc).__name__[:128],
    }


def _publish_absent(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    if len(encoded) > MAX_JSON_BYTES:
        raise Wave7InterruptError(
            "Wave 7 artifact exceeds the JSON size bound",
            code="wave7.receipt_oversize",
            context={"path": str(path), "size": len(encoded)},
        )
    try:
        write_strict_json_atomic(path, payload)
    except ParityContractError as exc:
        raise Wave7InterruptError(
            "Wave 7 artifact publication failed",
            code="wave7.artifact_publication",
            context={"path": str(path), "publisher_code": exc.code},
        ) from exc
    persisted = path.read_bytes()
    if persisted != encoded or _strict_json_loads(persisted, owner=str(path)) != dict(
        payload
    ):
        raise Wave7InterruptError(
            "Wave 7 artifact did not reload exactly",
            code="wave7.artifact_reload",
            context={"path": str(path)},
        )


def _verify_terminal_marker(
    path: Path,
    *,
    expected_bytes: bytes,
    expected_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[BaseException]]:
    expected_sha256 = _sha256_bytes(expected_bytes)
    observation: dict[str, Any] = {
        "expected_file_sha256": expected_sha256,
        "final_file_sha256": None,
        "strict_payload_equal": False,
        "unchanged": False,
    }
    try:
        final_bytes = path.read_bytes()
    except OSError as exc:
        return observation, [
            Wave7InterruptError(
                "published marker is unavailable at terminal verification",
                code="wave7.marker_changed",
                context={"path": str(path), "error_type": type(exc).__name__},
            )
        ]
    observation["final_file_sha256"] = _sha256_bytes(final_bytes)
    try:
        final_payload = _strict_json_loads(final_bytes, owner="terminal marker")
    except Wave7InterruptError as exc:
        return observation, [
            Wave7InterruptError(
                "published marker no longer reloads as strict JSON",
                code="wave7.marker_changed",
                context={"error_code": exc.code, "path": str(path)},
            )
        ]
    observation["strict_payload_equal"] = final_payload == dict(expected_payload)
    observation["unchanged"] = (
        final_bytes == expected_bytes
        and observation["final_file_sha256"] == expected_sha256
        and observation["strict_payload_equal"]
    )
    if observation["unchanged"]:
        return observation, []
    return observation, [
        Wave7InterruptError(
            "published marker changed before terminal receipt publication",
            code="wave7.marker_changed",
            context={
                "expected_file_sha256": expected_sha256,
                "final_file_sha256": observation["final_file_sha256"],
                "path": str(path),
            },
        )
    ]


def _preflight_target(path: Path, *, collision_code: str) -> None:
    try:
        assert_absent_artifact_target(path)
    except ParityContractError as exc:
        raise Wave7InterruptError(
            "Wave 7 artifact target is not absent",
            code=collision_code,
            context={"path": str(path), "publisher_code": exc.code},
        ) from exc


def _launcher_config_path(argv: Sequence[str]) -> Path:
    values: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--config":
            if index + 1 >= len(argv):
                raise Wave7InterruptError(
                    "launcher --config has no value",
                    code="wave7.launcher_config",
                )
            values.append(argv[index + 1])
            index += 2
            continue
        if item.startswith("--config="):
            values.append(item.partition("=")[2])
        index += 1
    if len(values) != 1 or not values[0]:
        raise Wave7InterruptError(
            "launcher must contain exactly one --config path",
            code="wave7.launcher_config",
            context={"count": len(values)},
        )
    declared = Path(values[0]).expanduser()
    return (declared if declared.is_absolute() else REPO_ROOT / declared).resolve()


def _resolve_parent_target_binding(request: ControllerRequest) -> dict[str, Any]:
    launcher_config = _launcher_config_path(request.launcher_argv)
    if launcher_config not in request.configs:
        raise Wave7InterruptError(
            "launcher config is not among the explicitly hashed config inputs",
            code="wave7.launcher_config_binding",
            context={
                "launcher_config": str(launcher_config),
                "supplied_configs": [str(path) for path in request.configs],
            },
        )
    try:
        resolved = load_train_config(launcher_config)
        selected = resolve_run_directory(resolved.config, cwd=REPO_ROOT)
    except BaseException as exc:
        raise Wave7InterruptError(
            "launcher config did not resolve a valid training artifact target",
            code="wave7.launcher_config",
            context={
                "error_code": str(getattr(exc, "code", type(exc).__name__)),
                "launcher_config": str(launcher_config),
            },
        ) from exc
    if selected.run_dir != request.parent_run_dir:
        raise Wave7InterruptError(
            "launcher config resolves a different parent run target",
            code="wave7.parent_target_mismatch",
            context={
                "expected": str(request.parent_run_dir),
                "resolved": str(selected.run_dir),
            },
        )
    return {
        "artifact_root": str(selected.artifact_root),
        "launcher_config": str(launcher_config),
        "resolved_config_fingerprint": resolved.fingerprint,
        "resolved_config_sources": [
            {"path": str(source.path), "sha256": source.sha256}
            for source in resolved.sources
        ],
        "run_dir": str(selected.run_dir),
        "run_name": selected.run_name,
    }


def _validate_prospective_parent_run(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise Wave7InterruptError(
            "parent run target must be absent before the one-shot launch",
            code="wave7.parent_run_exists",
            context={"path": str(path)},
        )
    ancestor = path.parent
    while not (ancestor.exists() or ancestor.is_symlink()):
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    if not ancestor.is_dir() or ancestor.is_symlink():
        raise Wave7InterruptError(
            "parent run path has no safe existing directory ancestor",
            code="wave7.parent_run",
            context={"ancestor": str(ancestor), "path": str(path)},
        )


def _postconditions(
    request: ControllerRequest,
    *,
    checkpoint: Mapping[str, Any] | None,
    checkpoint_publication: Mapping[str, Any] | None,
    gpu_baseline: Sequence[Mapping[str, Any]],
    gpu_sampler: Callable[[], list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[BaseException]]:
    failures: list[BaseException] = []
    result: dict[str, Any] = {}
    before = _snapshot_run_tree(request.parent_run_dir)
    time.sleep(request.stability_seconds)
    after = _snapshot_run_tree(request.parent_run_dir)
    result["stability_before"] = {
        "entry_count": before["entry_count"],
        "fingerprint": before["fingerprint"],
    }
    result["stability_after"] = {
        "entry_count": after["entry_count"],
        "fingerprint": after["fingerprint"],
    }
    result["late_write_detected"] = before["fingerprint"] != after["fingerprint"]
    if result["late_write_detected"]:
        failures.append(
            Wave7InterruptError(
                "parent run changed during the post-termination stability window",
                code="wave7.late_write",
            )
        )
    maximum = _max_logged_train_step(request.parent_run_dir / "logging.jsonl")
    result["max_logged_train_step"] = maximum
    if maximum > request.expected_checkpoint_step:
        failures.append(
            Wave7InterruptError(
                "parent logged a train step beyond the interruption boundary",
                code="wave7.logged_step",
                context={"maximum": maximum},
            )
        )
    step_five = request.parent_run_dir / "checkpoints/step-5"
    final = request.parent_run_dir / "checkpoints/final.json"
    result["step_5_absent"] = not (step_five.exists() or step_five.is_symlink())
    result["final_json_absent"] = not (final.exists() or final.is_symlink())
    if not result["step_5_absent"]:
        failures.append(
            Wave7InterruptError("step-5 exists", code="wave7.step_5_present")
        )
    if not result["final_json_absent"]:
        failures.append(
            Wave7InterruptError("final.json exists", code="wave7.final_present")
        )
    status_value = _run_status(request.parent_run_dir / "run.json")
    result["run_status"] = status_value
    result["run_not_completed"] = status_value != "completed"
    if not result["run_not_completed"]:
        failures.append(
            Wave7InterruptError(
                "parent run reports completed", code="wave7.run_completed"
            )
        )
    apps = gpu_sampler()
    baseline_keys = {
        (str(row["gpu_uuid"]), int(row["pid"]), str(row["process_name"]))
        for row in gpu_baseline
    }
    added_apps = [
        row
        for row in apps
        if (str(row["gpu_uuid"]), int(row["pid"]), str(row["process_name"]))
        not in baseline_keys
    ]
    result["nvidia_compute_apps"] = apps
    result["nvidia_compute_apps_added"] = added_apps
    result["nvidia_compute_apps_baseline"] = list(gpu_baseline)
    if added_apps:
        failures.append(
            Wave7InterruptError(
                "new nvidia compute processes remain after interruption",
                code="wave7.gpu_process_survivor",
                context={"count": len(added_apps), "rows": added_apps},
            )
        )
    if checkpoint is not None and checkpoint_publication is not None:
        final_bytes, final_run_bytes = _read_boundary_files(request)
        final_manifest = _parse_manifest_bytes(final_bytes, request=request)
        event_error: BaseException | None = None
        try:
            final_event = _checkpoint_publication_event(
                final_run_bytes,
                request=request,
                checkpoint=checkpoint,
            )
        except BaseException as exc:
            final_event = None
            event_error = exc
        result["manifest_unchanged"] = (
            _sha256_bytes(final_bytes) == checkpoint["file_sha256"]
            and final_manifest.aggregate_digest == checkpoint["aggregate_digest"]
        )
        result["manifest_final_file_sha256"] = _sha256_bytes(final_bytes)
        result["manifest_final_aggregate_digest"] = final_manifest.aggregate_digest
        result["checkpoint_publication_event_unchanged"] = (
            event_error is None
            and _sha256_bytes(final_run_bytes)
            == checkpoint_publication["run_file_sha256"]
            and final_event == checkpoint_publication["event"]
        )
        result["checkpoint_publication_run_final_file_sha256"] = _sha256_bytes(
            final_run_bytes
        )
        if not result["manifest_unchanged"]:
            failures.append(
                Wave7InterruptError(
                    "step-3 training-state manifest changed after interruption",
                    code="wave7.manifest_changed",
                )
            )
        if not result["checkpoint_publication_event_unchanged"]:
            failures.append(
                Wave7InterruptError(
                    "checkpoint publication event changed after interruption",
                    code="wave7.checkpoint_event_changed",
                    context={
                        "error_code": None
                        if event_error is None
                        else str(
                            getattr(event_error, "code", type(event_error).__name__)
                        )
                    },
                )
            )
    else:
        result["manifest_unchanged"] = False
        result["checkpoint_publication_event_unchanged"] = False
    return result, failures


def execute(
    request: ControllerRequest,
    *,
    gpu_sampler: Callable[[], list[dict[str, Any]]] = _sample_nvidia_compute_apps,
) -> int:
    started_monotonic = time.monotonic()
    started_at = _utc_now()
    events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] | None = None
    checkpoint_publication: dict[str, Any] | None = None
    graph: tuple[ProcessRecord, ...] = ()
    process: subprocess.Popen[Any] | None = None
    termination: dict[str, Any] = {
        "capture_completed_monotonic": None,
        "captured_process_graph": [],
        "cleanup_errors": [],
        "captured_pgids": [],
        "captured_pids": [],
        "duration_seconds": 0.0,
        "launcher_exited": False,
        "launcher_returncode": None,
        "reaped": [],
        "remaining_pgids": [],
        "remaining_pids": [],
        "post_marker_discovered_pids": [],
    }
    launch: dict[str, Any] = {
        "attempted": False,
        "argv": list(request.launcher_argv),
        "argv_sha256": _sha256_bytes(canonical_json_bytes(list(request.launcher_argv))),
        "pgid": None,
        "pid": None,
        "returncode": None,
    }
    marker_observation: dict[str, Any] = {
        "expected_file_sha256": None,
        "file_sha256": None,
        "final_file_sha256": None,
        "path": str(request.marker),
        "published": False,
        "strict_payload_equal": False,
        "unchanged": False,
    }
    postconditions: dict[str, Any] = {}
    script_hash = _sha256_file(Path(__file__).resolve())
    source_hashes: list[dict[str, Any]] = []
    config_hashes: list[dict[str, Any]] = []
    baseline_controller_children: dict[int, int] = {}
    gpu_baseline: list[dict[str, Any]] = []
    marker_expected_bytes: bytes | None = None
    marker_expected_payload: dict[str, Any] | None = None
    target_binding: dict[str, Any] = {}

    try:
        _preflight_target(request.receipt, collision_code="wave7.receipt_collision")
    except BaseException as exc:
        print(f"{getattr(exc, 'code', 'wave7.unexpected')}: {exc}", file=sys.stderr)
        return 1

    try:
        _preflight_target(request.marker, collision_code="wave7.marker_collision")
        if request.expected_checkpoint_step != 3:
            raise Wave7InterruptError(
                "Wave 7 interruption is fixed to optimizer step 3",
                code="wave7.expected_step",
                context={"observed": request.expected_checkpoint_step, "required": 3},
            )
        _validate_prospective_parent_run(request.parent_run_dir)
        target_binding = _resolve_parent_target_binding(request)
        source_hashes = _identity_files(request.sources, owner="source")
        config_hashes = _identity_files(request.configs, owner="config")
        gpu_baseline = gpu_sampler()
        _enable_subreaper()
        baseline_controller_children = _controller_child_pids()
        process = subprocess.Popen(
            list(request.launcher_argv),
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        launch.update(
            {
                "attempted": True,
                "pgid": os.getpgid(process.pid),
                "pid": process.pid,
            }
        )
        events.append(
            {
                "kind": "launcher_started",
                "monotonic": time.monotonic(),
                "pid": process.pid,
                "timestamp": _utc_now(),
            }
        )
        checkpoint, checkpoint_publication = _wait_for_durable_boundary(
            process, request
        )
        graph = _capture_process_graph(process.pid)
        _revalidate_durable_boundary(
            request,
            checkpoint=checkpoint,
            publication=checkpoint_publication,
        )
        capture_completed = time.monotonic()
        graph_payload = [asdict(item) for item in graph]
        marker_payload = {
            "captured_pgid_set": sorted({item.pgid for item in graph}),
            "captured_pid_set": [item.pid for item in graph],
            "captured_process_graph": graph_payload,
            "checkpoint": checkpoint,
            "checkpoint_publication_event": checkpoint_publication["event"],
            "checkpoint_publication_run": {
                "file_sha256": checkpoint_publication["run_file_sha256"],
                "path": checkpoint_publication["run_path"],
                "size": checkpoint_publication["run_size"],
            },
            "config_hashes": config_hashes,
            "gpu_compute_apps_baseline": gpu_baseline,
            "launch": {"pgid": launch["pgid"], "pid": launch["pid"]},
            "launcher": {
                "argv": list(request.launcher_argv),
                "argv_sha256": launch["argv_sha256"],
            },
            "marker_target": str(request.marker),
            "receipt_target": str(request.receipt),
            "schema": MARKER_SCHEMA,
            "script": {"path": str(Path(__file__).resolve()), "sha256": script_hash},
            "source_hashes": source_hashes,
            "target_binding": target_binding,
            "timestamp": _utc_now(),
        }
        _publish_absent(request.marker, marker_payload)
        marker_expected_payload = marker_payload
        marker_expected_bytes = canonical_json_bytes(marker_payload) + b"\n"
        marker_sha256 = _sha256_bytes(marker_expected_bytes)
        marker_observation.update(
            {
                "expected_file_sha256": marker_sha256,
                "file_sha256": _sha256_file(request.marker),
                "published": True,
            }
        )
        events.append(
            {
                "kind": "marker_published",
                "monotonic": time.monotonic(),
                "timestamp": _utc_now(),
            }
        )
        termination = _terminate_captured(
            process,
            graph,
            request=request,
            capture_completed_monotonic=capture_completed,
            events=events,
            baseline_controller_children=baseline_controller_children,
        )
    except BaseException as exc:
        errors.append(_error_record(exc))
        if process is not None:
            try:
                if not graph:
                    try:
                        graph = _capture_process_graph(process.pid)
                    except Wave7InterruptError:
                        try:
                            graph, _ = _expand_related_processes(
                                (),
                                baseline_controller_children=baseline_controller_children,
                            )
                        except ProcessGraphOverflow as overflow:
                            graph = overflow.graph
                            errors.append(_error_record(overflow))
                capture_completed = time.monotonic()
                termination = _terminate_captured(
                    process,
                    graph,
                    request=request,
                    capture_completed_monotonic=capture_completed,
                    events=events,
                    baseline_controller_children=baseline_controller_children,
                )
            except BaseException as termination_exc:
                errors.append(_error_record(termination_exc))
    finally:
        if process is not None:
            launch["returncode"] = process.poll()

    if process is not None:
        errors.extend(termination["cleanup_errors"])
        if not termination["launcher_exited"]:
            errors.append(
                _error_record(
                    Wave7InterruptError(
                        "launcher did not exit after termination",
                        code="wave7.launcher_survivor",
                    )
                )
            )
        if termination["remaining_pids"] or termination["remaining_pgids"]:
            errors.append(
                _error_record(
                    Wave7InterruptError(
                        "captured process identities remain after bounded reap",
                        code="wave7.termination_survivor",
                        context={
                            "remaining_pids": termination["remaining_pids"],
                            "remaining_pgids": termination["remaining_pgids"],
                        },
                    )
                )
            )
        try:
            postconditions, post_failures = _postconditions(
                request,
                checkpoint=checkpoint,
                checkpoint_publication=checkpoint_publication,
                gpu_baseline=gpu_baseline,
                gpu_sampler=gpu_sampler,
            )
            errors.extend(_error_record(exc) for exc in post_failures)
        except BaseException as exc:
            errors.append(_error_record(exc))
        if marker_expected_bytes is not None and marker_expected_payload is not None:
            marker_final, marker_failures = _verify_terminal_marker(
                request.marker,
                expected_bytes=marker_expected_bytes,
                expected_payload=marker_expected_payload,
            )
            marker_observation.update(marker_final)
            errors.extend(_error_record(exc) for exc in marker_failures)

    finished_at = _utc_now()
    status_value = "passed" if process is not None and not errors else "failed"
    receipt_payload = {
        "checkpoint": checkpoint,
        "checkpoint_publication": checkpoint_publication,
        "config_hashes": config_hashes,
        "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        "errors": errors,
        "events": events,
        "finished_at": finished_at,
        "launch": launch,
        "marker": marker_observation,
        "postconditions": postconditions,
        "request": {
            "expected_checkpoint_step": request.expected_checkpoint_step,
            "parent_run_dir": str(request.parent_run_dir),
            "timeout_seconds": request.timeout_seconds,
        },
        "schema": RECEIPT_SCHEMA,
        "script": {"path": str(Path(__file__).resolve()), "sha256": script_hash},
        "source_hashes": source_hashes,
        "started_at": started_at,
        "status": status_value,
        "termination": termination,
        "target_binding": target_binding,
    }
    try:
        _publish_absent(request.receipt, receipt_payload)
    except BaseException as exc:
        print(f"{getattr(exc, 'code', 'wave7.unexpected')}: {exc}", file=sys.stderr)
        return 1
    if status_value != "passed":
        for error in errors:
            print(f"{error['code']}: {error['message']}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        request = parse_request(argv)
    except Wave7InterruptError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    return execute(request)


if __name__ == "__main__":
    raise SystemExit(main())
