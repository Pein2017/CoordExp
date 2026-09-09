#!/usr/bin/env python3
"""Verify wake-MCP frontend reclamation without touching production wake state.

This check launches the supplied Core binary on an isolated Unix WebSocket
control socket, configures the installed wake frontend as an opted-in stdio
MCP server, and starts the installed wake daemon against the same isolated
``CODEX_HOME``.  It arms one real, future ``time`` monitor and then unsubscribes
the synthetic task.  A candidate must reclaim the frontend while the daemon and
the durable monitor remain intact, and a later status call must reconstruct the
frontend.  The stock binary is expected to produce ``HOLD``.

No model turn, delivery, production socket, shared ledger, or user task is
created.  Every process terminated by cleanup was first observed as an exact
PID/creation-time identity under the isolated Core or daemon root.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Callable

import aiohttp
import psutil


REQUESTED_PLUGIN_ROOT = Path(
    "/data/CoordExp/.codex/plugins/cache/coordexp-local/codex-wake-me-up/"
    "0.1.0+codex.20260908151430"
)
PLUGIN_CACHE_ROOT = REQUESTED_PLUGIN_ROOT.parent
DEFAULT_MAIN_IDLE = 0.8
DEFAULT_CHILD_IDLE = 0.4
WAIT_TIMEOUT = 6.0
RPC_TIMEOUT = 25.0


def _same_process(identity: tuple[int, float] | None) -> bool:
    if identity is None:
        return False
    pid, create_time = identity
    try:
        process = psutil.Process(pid)
        return (
            process.create_time() == create_time
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _identity(process: psutil.Process) -> tuple[int, float]:
    return process.pid, process.create_time()


def _details(identity: tuple[int, float]) -> dict[str, object]:
    result: dict[str, object] = {
        "pid": identity[0],
        "create_time": identity[1],
    }
    try:
        process = psutil.Process(identity[0])
        if process.create_time() != identity[1]:
            result["reused"] = True
            return result
        result.update(
            {
                "ppid": process.ppid(),
                "name": process.name(),
                "status": process.status(),
                "cmdline": process.cmdline(),
            }
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
        result["error"] = repr(exc)
    return result


async def _wait_until(
    predicate: Callable[[], bool], timeout: float, *, interval: float = 0.05
) -> tuple[bool, float]:
    started = time.monotonic()
    deadline = started + timeout
    while True:
        if predicate():
            return True, time.monotonic() - started
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, time.monotonic() - started
        await asyncio.sleep(min(interval, remaining))


class ExactTree:
    """Track and terminate only one process root and identities seen below it."""

    def __init__(self, root: tuple[int, float]):
        self.root = root
        self.identities: set[tuple[int, float]] = {root}
        self.snapshots: list[list[dict[str, object]]] = []

    def snapshot(self) -> list[tuple[int, float]]:
        current: list[tuple[int, float]] = []
        if _same_process(self.root):
            try:
                root = psutil.Process(self.root[0])
                current = [_identity(root)] + [
                    _identity(child) for child in root.children(recursive=True)
                ]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                current = []
        self.identities.update(current)
        self.snapshots.append([_details(item) for item in current])
        return current

    def live(self) -> list[tuple[int, float]]:
        return [item for item in self.identities if _same_process(item)]

    def details(self) -> list[dict[str, object]]:
        return [_details(item) for item in sorted(self.identities)]

    def terminate(self) -> dict[str, object]:
        self.snapshot()
        before = self.live()
        processes: list[psutil.Process] = []
        for identity in before:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                processes.append(psutil.Process(identity[0]))

        def depth(process: psutil.Process) -> int:
            result = 0
            try:
                parent = process.ppid()
                while parent and parent != 1 and result < 64:
                    result += 1
                    parent = psutil.Process(parent).ppid()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            return result

        for process in sorted(processes, key=depth, reverse=True):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                process.terminate()
        _, survivors = psutil.wait_procs(processes, timeout=2.0)
        for process in survivors:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                process.kill()
        if survivors:
            psutil.wait_procs(survivors, timeout=2.0)
        self.snapshot()
        after = self.live()
        return {
            "before": before,
            "after": after,
            "details": self.details(),
        }


class UnixRpc:
    """Small request-id client for Core's Unix WebSocket app-server transport."""

    def __init__(self, socket_path: Path, trace: list[dict[str, object]]):
        self.socket_path = socket_path
        self.trace = trace
        self.session: aiohttp.ClientSession | None = None
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.next_id = 0
        self.notifications: list[dict[str, object]] = []

    async def connect(self) -> None:
        self.session = aiohttp.ClientSession(
            connector=aiohttp.UnixConnector(path=str(self.socket_path))
        )
        self.websocket = await self.session.ws_connect(
            "http://localhost/", max_msg_size=0
        )

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def notify(
        self, method: str, params: dict[str, object] | None = None
    ) -> None:
        if self.websocket is None:
            raise RuntimeError("Core WebSocket is not connected")
        await self.websocket.send_json(
            {"jsonrpc": "2.0", "method": method, "params": params or {}}
        )

    async def request(
        self,
        method: str,
        params: dict[str, object] | None = None,
        *,
        timeout: float = RPC_TIMEOUT,
    ) -> dict[str, object]:
        if self.websocket is None:
            raise RuntimeError("Core WebSocket is not connected")
        self.next_id += 1
        request_id = self.next_id
        await self.websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {method}")
            message = await asyncio.wait_for(self.websocket.receive_json(), remaining)
            if not isinstance(message, dict):
                continue
            if message.get("id") == request_id:
                self.trace.append(
                    {"id": request_id, "method": method, "response": message}
                )
                return message
            self.notifications.append(message)


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _render_config(
    home: Path,
    plugin_root: Path,
    idle_timeout: float,
    child_idle_timeout: float,
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    lines = [
        'model = "probe-model"',
        'model_provider = "probe"',
        'approval_policy = "never"',
        'sandbox_mode = "danger-full-access"',
        "",
        "[features]",
        "plugins = false",
        "",
        "[model_providers.probe]",
        'name = "isolated no-model probe"',
        'base_url = "http://127.0.0.1:9/v1"',
        'wire_api = "responses"',
        "requires_openai_auth = false",
        "",
        "[mcp_servers.wake]",
        f"command = {_toml_string(str((Path('/root/miniconda3/envs/ms/bin/python') if Path('/root/miniconda3/envs/ms/bin/python').is_file() else Path(sys.executable)).resolve()))}",
        f"args = [{_toml_string(str(plugin_root / 'scripts/mcp_server.py'))}]",
        f"cwd = {_toml_string(str(plugin_root))}",
        "startup_timeout_sec = 15",
        "tool_timeout_sec = 25",
        f"idle_timeout_sec = {idle_timeout:.6g}",
        f"idle_timeout_completed_subagent_sec = {child_idle_timeout:.6g}",
        'idle_recovery = "stateless"',
        "",
        "[mcp_servers.wake.env]",
        f"PYTHONPATH = {_toml_string(str(plugin_root / 'src'))}",
        f"CODEX_HOME = {_toml_string(str(home))}",
        'PYTHONDONTWRITEBYTECODE = "1"',
    ]
    path = home / "config.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _find_plugin(explicit: str | None) -> tuple[Path | None, str | None]:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        return (
            (candidate, None)
            if (candidate / "scripts/mcp_server.py").is_file()
            else (None, str(candidate))
        )
    if (REQUESTED_PLUGIN_ROOT / "scripts/mcp_server.py").is_file():
        return REQUESTED_PLUGIN_ROOT, None
    candidates = sorted(
        (
            item
            for item in PLUGIN_CACHE_ROOT.glob("*/scripts/mcp_server.py")
            if item.is_file()
        ),
        reverse=True,
    )
    if candidates:
        selected = candidates[0].parent.parent
        return selected, str(REQUESTED_PLUGIN_ROOT)
    return None, str(REQUESTED_PLUGIN_ROOT)


def _frontend_identity(
    core_tree: ExactTree, plugin_root: Path
) -> tuple[int, float] | None:
    core_tree.snapshot()
    script = str(plugin_root / "scripts/mcp_server.py")
    for identity in core_tree.identities:
        if not _same_process(identity) or identity == core_tree.root:
            continue
        try:
            command_line = " ".join(psutil.Process(identity[0]).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if script in command_line:
            return identity
    return None


def _result(response: dict[str, object]) -> dict[str, object]:
    if "error" in response:
        raise RuntimeError(json.dumps(response["error"], sort_keys=True))
    value = response.get("result")
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON-RPC result is not an object: {response!r}")
    return value


def _structured_result(response: dict[str, object]) -> dict[str, object]:
    result = _result(response)
    value = result.get("structuredContent")
    if isinstance(value, dict):
        return value
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            try:
                parsed = json.loads(item["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise RuntimeError(f"MCP result has no structured object: {response!r}")


def _stable_monitor(value: dict[str, object]) -> dict[str, object]:
    """Keep durable identity/semantic fields while excluding evaluator counters."""

    keys = (
        "monitor_id",
        "idempotency_key",
        "target",
        "condition",
        "state",
        "mode",
        "idle_barrier",
        "allow_heuristic_continuation",
        "expires_at",
        "delivery_kind",
        "delivery",
        "delivery_state",
        "armed_at",
        "created_at",
        "rearm_of",
    )
    return {key: value.get(key) for key in keys if key in value}


def _read_raw_monitor(db_path: Path, monitor_id: str) -> dict[str, object]:
    if not db_path.is_file():
        return {"exists": False, "row_count": 0}
    uri = f"file:{db_path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=2.0) as database:
            columns = [
                row[1] for row in database.execute("PRAGMA table_info(monitors)")
            ]
            rows = database.execute(
                "SELECT * FROM monitors ORDER BY created_at"
            ).fetchall()
    except (sqlite3.Error, OSError) as exc:
        return {"exists": True, "error": repr(exc)}
    row = None
    for candidate in rows:
        item = dict(zip(columns, candidate))
        if item.get("monitor_id") == monitor_id:
            row = item
            break
    if row is None:
        return {"exists": True, "row_count": len(rows), "monitor": None}
    stable_columns = (
        "monitor_id",
        "idempotency_key",
        "state",
        "mode",
        "condition_json",
        "semantic_json",
        "delivery_json",
        "delivery_state",
        "expires_at",
        "armed_at",
        "created_at",
        "rearm_of",
    )
    stable = {key: row.get(key) for key in stable_columns if key in row}
    return {"exists": True, "row_count": len(rows), "monitor": stable}


def _heartbeat(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _stable_heartbeat(value: dict[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        key: value.get(key)
        for key in (
            "pid",
            "accepting_work",
            "event_capability_epoch",
            "delivery_capability_epoch",
            "loaded_source_identity",
        )
    }


async def run(args: argparse.Namespace) -> int:
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"candidate binary is not executable: {binary}")
    plugin_root, requested_missing = _find_plugin(args.plugin_root)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("run-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    run_dir = output / run_name
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = output / f"{run_name}-{suffix}"
    run_dir.mkdir(mode=0o700)
    report: dict[str, object] = {
        "schema_version": 1,
        "harness": str(Path(__file__).resolve()),
        "binary": {
            "path": str(binary),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        "run_label": args.run_label,
        "plugin_requested": str(REQUESTED_PLUGIN_ROOT),
        "plugin_selected": str(plugin_root) if plugin_root else None,
        "plugin_requested_missing": requested_missing,
        "output_dir": str(output),
        "run_dir": str(run_dir),
        "timers": {
            "idle_timeout_sec": args.idle_timeout,
            "idle_timeout_completed_subagent_sec": args.child_idle_timeout,
            "wait_timeout_sec": WAIT_TIMEOUT,
        },
        "no_model_or_gpu": True,
        "delivery_attempted": False,
        "status": "HOLD",
        "gaps": [],
    }
    if plugin_root is None:
        report["boundary"] = "installed wake frontend is unavailable"
        report["gaps"].append(
            {
                "area": "wake_frontend",
                "status": "UNPROVEN",
                "reason": f"no usable plugin root; requested={REQUESTED_PLUGIN_ROOT}",
            }
        )
        receipt = run_dir / "wake-mcp-idle-acceptance-receipt.json"
        receipt.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps({"status": report["status"], "receipt": str(receipt)}))
        return 1

    home = run_dir / "core-home"
    process_home = run_dir / "process-home"
    workspace = run_dir / "workspace"
    runtime_root = home / "runtime" / "codex-wake-me-up"
    for path in (home, process_home, workspace, runtime_root):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    (workspace / "probe.txt").write_text("isolated wake idle probe\n", encoding="utf-8")
    config = _render_config(
        home, plugin_root, args.idle_timeout, args.child_idle_timeout
    )
    report.update(
        {
            "codex_home": str(home),
            "runtime_root": str(runtime_root),
            "workspace": str(workspace),
            "config": str(config),
            "monitor_db": str(runtime_root / "monitors.sqlite3"),
        }
    )

    env = dict(os.environ)
    env.update(
        {
            "CODEX_HOME": str(home),
            "HOME": str(process_home),
            "NO_COLOR": "1",
            "RUST_BACKTRACE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    plugin_env = dict(env)
    plugin_env["PYTHONPATH"] = str(plugin_root / "src")

    core_process: subprocess.Popen[bytes] | None = None
    daemon_process: subprocess.Popen[bytes] | None = None
    core_tree: ExactTree | None = None
    daemon_tree: ExactTree | None = None
    rpc: UnixRpc | None = None
    trace: list[dict[str, object]] = []
    # Linux AF_UNIX paths are capped by SUN_LEN.  Keep the actual listener
    # short, then expose it through the isolated CODEX_HOME path expected by
    # the installed frontend's AppServerClient.
    socket_path = Path(f"/tmp/codex-wake-mcp-{os.getpid()}.sock")
    control_dir = home / "app-server-control"
    control_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    control_link = control_dir / "app-server-control.sock"
    if socket_path.exists() or control_link.exists() or control_link.is_symlink():
        raise SystemExit(
            f"refusing to reuse an existing isolated socket: {socket_path}"
        )
    heartbeat_path = runtime_root / "daemon-heartbeat.json"
    daemon_stderr = (run_dir / "daemon.stderr.log").open("w", encoding="utf-8")
    core_stderr = (run_dir / "core.stderr.log").open("w", encoding="utf-8")
    monitor_id: str | None = None
    frontend_identities: list[tuple[int, float]] = []
    try:
        core_process = subprocess.Popen(
            [str(binary), "app-server", "--listen", f"unix://{socket_path}"],
            cwd=str(workspace),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=core_stderr,
            start_new_session=True,
        )
        core_tree = ExactTree(_identity(psutil.Process(core_process.pid)))
        core_tree.snapshot()
        socket_ready, socket_wait = await _wait_until(socket_path.exists, 10.0)
        report["socket_wait_seconds"] = round(socket_wait, 3)
        if not socket_ready:
            raise RuntimeError(f"isolated Core socket did not appear: {socket_path}")
        # Core removes stale default control paths during startup.  Create the
        # compatibility link only after the real listener is ready so the
        # installed frontend resolves this exact isolated socket.
        control_link.symlink_to(socket_path)
        report["control_link_before_daemon"] = {
            "path": str(control_link),
            "is_symlink": control_link.is_symlink(),
            "exists": control_link.exists(),
            "target": os.readlink(control_link),
            "target_exists": socket_path.exists(),
        }

        daemon_process = subprocess.Popen(
            [
                str(Path(sys.executable).resolve()),
                "-m",
                "codex_wake_me_up.daemon",
                "--runtime-root",
                str(runtime_root),
            ],
            cwd=str(plugin_root),
            env=plugin_env,
            stdin=subprocess.DEVNULL,
            stdout=(run_dir / "daemon.stdout.log").open("w", encoding="utf-8"),
            stderr=daemon_stderr,
            start_new_session=True,
        )
        daemon_tree = ExactTree(_identity(psutil.Process(daemon_process.pid)))
        daemon_tree.snapshot()
        daemon_ready, daemon_wait = await _wait_until(
            lambda: (
                daemon_process is not None
                and daemon_process.poll() is None
                and (_heartbeat(heartbeat_path) or {}).get("pid") == daemon_process.pid
                and (_heartbeat(heartbeat_path) or {}).get("accepting_work") is True
            ),
            10.0,
        )
        report["daemon_wait_seconds"] = round(daemon_wait, 3)
        report["daemon_pid"] = daemon_process.pid
        report["daemon_identity"] = daemon_tree.root
        report["daemon_heartbeat_before"] = _heartbeat(heartbeat_path)
        if not daemon_ready:
            raise RuntimeError(
                f"isolated daemon did not become healthy: {daemon_stderr.name}"
            )

        rpc = UnixRpc(socket_path, trace)
        await rpc.connect()
        initialized = await rpc.request(
            "initialize",
            {
                "clientInfo": {"name": "wake-mcp-idle-acceptance", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        initialized_result = _result(initialized)
        report["core_initialize"] = initialized
        if initialized_result.get("codexHome") != str(home):
            raise RuntimeError(
                f"Core reported unexpected CODEX_HOME: {initialized_result.get('codexHome')!r}"
            )
        await rpc.notify("initialized")
        started = await rpc.request(
            "thread/start",
            {
                "cwd": str(workspace),
                "ephemeral": False,
                "model": "probe-model",
                "modelProvider": "probe",
                "approvalPolicy": "never",
            },
        )
        thread = _result(started).get("thread")
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
            raise RuntimeError(f"thread/start returned no durable thread: {started!r}")
        thread_id = str(thread["id"])
        report["thread_id"] = thread_id
        report["control_link_before_frontend"] = {
            "is_symlink": control_link.is_symlink(),
            "exists": control_link.exists(),
            "target_exists": socket_path.exists(),
        }

        arm = await rpc.request(
            "mcpServer/tool/call",
            {
                "threadId": thread_id,
                "server": "wake",
                "tool": "wait_for_event",
                "arguments": {
                    "condition": {"type": "time", "after_seconds": 3600},
                    "expires_in_seconds": 300,
                    "idempotency_key": "codex-mcp-idle-wake-acceptance",
                },
                "_meta": {"threadId": thread_id},
            },
        )
        arm_value = _structured_result(arm)
        report["arm_response"] = arm
        if arm_value.get("state") != "armed":
            report["boundary"] = "real frontend did not return armed"
            report["gaps"].append(
                {
                    "area": "native_delivery_registration",
                    "status": "HOLD",
                    "reason": "wait_for_event reached the installed frontend but did not return state=armed; no monitor-preservation claim is made",
                }
            )
            return 1
        monitor_id_value = arm_value.get("monitor_id")
        if not isinstance(monitor_id_value, str) or not monitor_id_value:
            raise RuntimeError(f"armed response omitted monitor_id: {arm_value!r}")
        monitor_id = monitor_id_value
        report["monitor_id"] = monitor_id
        initial_frontend = _frontend_identity(core_tree, plugin_root)
        if initial_frontend is None:
            raise RuntimeError("wake frontend process was not found under Core")
        frontend_identities.append(initial_frontend)
        report["frontend_identity_before"] = initial_frontend

        status_before_response = await rpc.request(
            "mcpServer/tool/call",
            {
                "threadId": thread_id,
                "server": "wake",
                "tool": "wake_me_up_status",
                "arguments": {"monitor_id": monitor_id, "view": "audit"},
            },
        )
        status_before = _structured_result(status_before_response)
        report["monitor_before"] = status_before
        report["monitor_before_stable"] = _stable_monitor(status_before)
        raw_before = _read_raw_monitor(runtime_root / "monitors.sqlite3", monitor_id)
        report["raw_monitor_before"] = raw_before
        report["daemon_heartbeat_at_reclaim"] = _heartbeat(heartbeat_path)
        report["daemon_stable_at_reclaim"] = _stable_heartbeat(
            report["daemon_heartbeat_at_reclaim"]
        )

        unsubscribe = await rpc.request("thread/unsubscribe", {"threadId": thread_id})
        report["unsubscribe"] = unsubscribe
        reclaimed, reclaim_wait = await _wait_until(
            lambda: not _same_process(initial_frontend),
            max(WAIT_TIMEOUT, args.idle_timeout * 6),
        )
        report["frontend_reclaim_wait_seconds"] = round(reclaim_wait, 3)
        report["frontend_reclaimed"] = reclaimed
        report["daemon_alive_after_reclaim"] = _same_process(daemon_tree.root)
        report["daemon_heartbeat_after_reclaim"] = _heartbeat(heartbeat_path)
        report["daemon_stable_after_reclaim"] = _stable_heartbeat(
            report["daemon_heartbeat_after_reclaim"]
        )
        report["raw_monitor_during_reclaim"] = _read_raw_monitor(
            runtime_root / "monitors.sqlite3", monitor_id
        )

        if not reclaimed:
            report["status"] = "HOLD"
            report["boundary"] = (
                "candidate did not reclaim the wake frontend within the bounded grace"
            )
            report["gaps"].append(
                {
                    "area": "wake_frontend_reclamation",
                    "status": "HOLD",
                    "reason": "frontend remained resident; stock 0.153.4 is expected to land here",
                }
            )
            return 1

        if not _same_process(daemon_tree.root):
            raise RuntimeError("independent wake daemon exited during frontend reclaim")
        raw_during = report["raw_monitor_during_reclaim"]
        if raw_during.get("monitor") != raw_before.get("monitor"):
            raise RuntimeError(
                "durable monitor identity/semantic row changed during frontend reclaim"
            )
        if (
            _stable_heartbeat(report["daemon_heartbeat_after_reclaim"])
            != report["daemon_stable_at_reclaim"]
        ):
            raise RuntimeError("wake daemon identity/capability heartbeat changed")

        reconnect_response = await rpc.request(
            "mcpServer/tool/call",
            {
                "threadId": thread_id,
                "server": "wake",
                "tool": "wake_me_up_status",
                "arguments": {"monitor_id": monitor_id, "view": "audit"},
            },
        )
        reconnect_status = _structured_result(reconnect_response)
        report["monitor_after_reconnect"] = reconnect_status
        report["monitor_after_reconnect_stable"] = _stable_monitor(reconnect_status)
        replacement_frontend = _frontend_identity(core_tree, plugin_root)
        if replacement_frontend is not None:
            frontend_identities.append(replacement_frontend)
        report["frontend_identity_after_reconnect"] = replacement_frontend
        report["frontend_identities_seen"] = frontend_identities
        raw_after = _read_raw_monitor(runtime_root / "monitors.sqlite3", monitor_id)
        report["raw_monitor_after_reconnect"] = raw_after
        _stable_after = _stable_monitor(reconnect_status)
        if reconnect_status.get("monitor_id") != monitor_id:
            raise RuntimeError(
                "reconnected status returned a different monitor identity"
            )
        if reconnect_status.get("state") != "armed":
            raise RuntimeError(
                f"reconnected monitor was not still armed: {reconnect_status!r}"
            )
        if _stable_after != report["monitor_before_stable"]:
            raise RuntimeError(
                "durable monitor fields changed after frontend reconnect"
            )
        if raw_after.get("monitor") != raw_before.get("monitor"):
            raise RuntimeError(
                "raw durable monitor row changed after frontend reconnect"
            )
        if replacement_frontend is None or replacement_frontend == initial_frontend:
            raise RuntimeError(
                "status demand did not create one distinct frontend replacement"
            )
        if len(set(frontend_identities)) != 2:
            raise RuntimeError(
                f"expected exactly one frontend replacement, saw {frontend_identities!r}"
            )
        report["status"] = "PASS"
        report["claim"] = (
            "The real wake frontend was reclaimed and reconstructed while one future armed monitor and the independent daemon remained unchanged."
        )
        return 0
    except Exception as exc:
        report["status"] = "HOLD"
        report["error"] = repr(exc)
        report.setdefault("gaps", []).append(
            {
                "area": "wake_frontend_acceptance",
                "status": "HOLD",
                "reason": repr(exc),
            }
        )
        return 1
    finally:
        report["rpc_trace"] = trace
        if rpc is not None:
            with contextlib.suppress(Exception):
                await rpc.close()
        if core_tree is not None:
            report["core_cleanup"] = core_tree.terminate()
        if daemon_tree is not None:
            report["daemon_cleanup"] = daemon_tree.terminate()
        core_stderr.close()
        daemon_stderr.close()
        with contextlib.suppress(FileNotFoundError):
            control_link.unlink()
        with contextlib.suppress(FileNotFoundError):
            socket_path.unlink()
        receipt = run_dir / "wake-mcp-idle-acceptance-receipt.json"
        receipt.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        report["receipt"] = str(receipt)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "receipt": str(receipt),
                    "boundary": report.get("boundary"),
                    "frontend_reclaimed": report.get("frontend_reclaimed"),
                    "daemon_alive_after_reclaim": report.get(
                        "daemon_alive_after_reclaim"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", required=True, help="Exact candidate codex executable."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Dedicated isolated output directory."
    )
    parser.add_argument(
        "--run-label", default="candidate", help="Receipt label, e.g. stock-baseline."
    )
    parser.add_argument("--plugin-root", help="Exact installed wake plugin root.")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_MAIN_IDLE,
        help="Positive main grace in seconds.",
    )
    parser.add_argument(
        "--child-idle-timeout",
        type=float,
        default=DEFAULT_CHILD_IDLE,
        help="Positive completed-child grace recorded in the config.",
    )
    args = parser.parse_args()
    if args.idle_timeout <= 0 or args.child_idle_timeout <= 0:
        parser.error("idle timers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(_parse_args())))
