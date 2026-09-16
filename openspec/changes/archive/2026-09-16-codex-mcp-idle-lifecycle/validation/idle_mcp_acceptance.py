#!/usr/bin/env python3
"""Bounded real-entry acceptance checks for the opt-in MCP idle lifecycle.

The checks launch the supplied ``codex`` binary as an app-server over stdio,
then launch real child MCP processes through Core's configured stdio transport.
The fixture server is intentionally tiny; its JSONL receipt is the source of
truth for process generations and business-call counts.  No model, GPU, or
production Core home is used.

The stock 0.153.4 binary is expected to produce RED because it does not know
the idle policy fields or reclaim behavior.  A later candidate is expected to
produce PASS for the required fixture scenarios.  Optional Serena and
completed-subagent paths are reported as gaps when they cannot be exercised;
they are never converted into a synthetic PASS.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import contextlib
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable

import psutil


FIXTURE = Path(__file__).resolve().with_name("idle_mcp_fixture_server.py")
DEFAULT_MAIN_IDLE = 0.8
DEFAULT_CHILD_IDLE = 0.4
RPC_TIMEOUT = 20.0
STARTUP_TIMEOUT = 8.0
WAIT_TIMEOUT = 5.0


def _json_text(value: object) -> str:
    """Render a value as a TOML basic string or a simple array."""

    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_json_text(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".6g")
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"unsupported TOML value: {type(value)!r}")


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


def _process_identity(process: psutil.Process) -> tuple[int, float]:
    return process.pid, process.create_time()


def _safe_process_details(identity: tuple[int, float]) -> dict[str, object]:
    details: dict[str, object] = {"pid": identity[0], "create_time": identity[1]}
    try:
        process = psutil.Process(identity[0])
        if process.create_time() != identity[1]:
            details["reused"] = True
            return details
        details.update(
            {
                "ppid": process.ppid(),
                "name": process.name(),
                "status": process.status(),
                "cmdline": process.cmdline(),
            }
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
        details["error"] = repr(exc)
    return details


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    events: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return events


async def _wait_for_event(
    path: Path,
    predicate: Callable[[dict[str, object]], bool],
    timeout: float = STARTUP_TIMEOUT,
) -> dict[str, object] | None:
    found: dict[str, object] | None = None

    def check() -> bool:
        nonlocal found
        for event in _read_jsonl(path):
            if predicate(event):
                found = event
        return found is not None

    await _wait_until(check, timeout)
    return found


class OwnedProcessTree:
    """Track only the app-server and descendants started by this harness."""

    def __init__(self, root: tuple[int, float]):
        self.root = root
        self.identities: set[tuple[int, float]] = {root}
        self.snapshots: list[list[dict[str, object]]] = []

    def snapshot(self) -> list[tuple[int, float]]:
        current: list[tuple[int, float]] = []
        if _same_process(self.root):
            try:
                root = psutil.Process(self.root[0])
                current = [_process_identity(root)]
                current.extend(
                    _process_identity(child) for child in root.children(recursive=True)
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                current = []
        for identity in current:
            self.identities.add(identity)
        self.snapshots.append([_safe_process_details(item) for item in current])
        return current

    def details(self) -> list[dict[str, object]]:
        return [_safe_process_details(identity) for identity in sorted(self.identities)]

    async def terminate_exact(self) -> list[tuple[int, float]]:
        """Terminate tracked identities, descendants first, then force-kill survivors."""

        self.snapshot()
        live: list[psutil.Process] = []
        for identity in self.identities:
            if not _same_process(identity):
                continue
            try:
                live.append(psutil.Process(identity[0]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        def depth(process: psutil.Process) -> int:
            count = 0
            try:
                parent = process.ppid()
                while parent and parent != 1 and count < 64:
                    count += 1
                    parent = psutil.Process(parent).ppid()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            return count

        for process in sorted(live, key=depth, reverse=True):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                process.terminate()
        _, survivors = psutil.wait_procs(live, timeout=2.0)
        for process in survivors:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                process.kill()
        if survivors:
            psutil.wait_procs(survivors, timeout=2.0)
        self.snapshot()
        return [identity for identity in self.identities if _same_process(identity)]


class RpcClient:
    """Line-delimited JSON-RPC client for the app-server stdio entrypoint."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        stderr_path: Path,
        trace: list[dict[str, object]],
    ):
        self.process = process
        self.stderr_path = stderr_path
        self.trace = trace
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, object]]] = {}
        self._write_lock = asyncio.Lock()
        self._notification_event = asyncio.Event()
        self.notifications: deque[dict[str, object]] = deque(maxlen=512)
        self.reader_error: str | None = None
        self.reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            assert self.process.stdout is not None
            while line := await self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self.notifications.append({"raw": line.decode(errors="replace")})
                    self._notification_event.set()
                    continue
                if not isinstance(message, dict):
                    continue
                request_id = message.get("id")
                if isinstance(request_id, int) and request_id in self._pending:
                    future = self._pending.pop(request_id)
                    if not future.done():
                        future.set_result(message)
                else:
                    self.notifications.append(message)
                    self._notification_event.set()
        except asyncio.CancelledError:
            raise
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            self.reader_error = repr(exc)
        finally:
            if self._pending:
                error = RuntimeError(self.reader_error or "app-server stdout closed")
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(error)
                self._pending.clear()
            self._notification_event.set()

    async def _send(self, payload: dict[str, object]) -> None:
        async with self._write_lock:
            if self.process.stdin is None:
                raise RuntimeError("app-server stdin is closed")
            self.process.stdin.write((json.dumps(payload) + "\n").encode())
            await self.process.stdin.drain()

    async def request(
        self,
        method: str,
        params: dict[str, object] | None = None,
        *,
        timeout: float = RPC_TIMEOUT,
    ) -> dict[str, object]:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, object]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        try:
            response = await asyncio.wait_for(future, timeout)
        except BaseException:
            self._pending.pop(request_id, None)
            raise
        self.trace.append(
            {
                "id": request_id,
                "method": method,
                "response": response,
            }
        )
        return response

    async def notify(
        self, method: str, params: dict[str, object] | None = None
    ) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def wait_notification(
        self,
        method: str,
        *,
        predicate: Callable[[dict[str, object]], bool] | None = None,
        timeout: float = RPC_TIMEOUT,
    ) -> dict[str, object]:
        async def find() -> dict[str, object] | None:
            for message in self.notifications:
                if message.get("method") != method:
                    continue
                if predicate is None or predicate(message):
                    return message
            return None

        async def wait() -> dict[str, object]:
            while True:
                self._notification_event.clear()
                message = await find()
                if message is not None:
                    return message
                await self._notification_event.wait()

        return await asyncio.wait_for(wait(), timeout)


class AppServer:
    def __init__(
        self,
        binary: Path,
        home: Path,
        workspace: Path,
        stderr_path: Path,
    ):
        self.binary = binary
        self.home = home
        self.workspace = workspace
        self.stderr_path = stderr_path
        self.process: asyncio.subprocess.Process | None = None
        self.tree: OwnedProcessTree | None = None
        self.rpc: RpcClient | None = None
        self.stderr_file: Any = None

    async def start(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_file = self.stderr_path.open("w", encoding="utf-8")
        process_home = self.home.parent / "process-home"
        process_home.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            {
                "CODEX_HOME": str(self.home),
                "HOME": str(process_home),
                "XDG_CONFIG_HOME": str(process_home / "config"),
                "XDG_CACHE_HOME": str(process_home / "cache"),
                "XDG_DATA_HOME": str(process_home / "data"),
                "TMPDIR": str(process_home / "tmp"),
                "NO_COLOR": "1",
                "RUST_BACKTRACE": "1",
            }
        )
        for key in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "TMPDIR"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        self.process = await asyncio.create_subprocess_exec(
            str(self.binary),
            "app-server",
            "--listen",
            "stdio://",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr_file,
            cwd=str(self.workspace),
            env=env,
            start_new_session=True,
        )
        root = psutil.Process(self.process.pid)
        self.tree = OwnedProcessTree(_process_identity(root))
        self.tree.snapshot()
        trace: list[dict[str, object]] = []
        self.rpc = RpcClient(self.process, self.stderr_path, trace)

    async def close(self) -> dict[str, object]:
        process = self.process
        tree = self.tree
        if process is None:
            return {}
        if tree is not None:
            tree.snapshot()
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        forced = False
        try:
            await asyncio.wait_for(process.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            forced = True
            if tree is not None:
                await tree.terminate_exact()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=3.0)
        # Core can exit cleanly while a language-server grandchild has already
        # been reparented.  Reconcile the exact identities we observed before
        # declaring cleanup complete; never scan or kill by process name.
        if tree is not None:
            tree.snapshot()
            survivors_before_cleanup = [
                identity for identity in tree.identities if _same_process(identity)
            ]
            if survivors_before_cleanup:
                forced = True
                await tree.terminate_exact()
        else:
            survivors_before_cleanup = []
        if self.rpc is not None:
            self.rpc.reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.rpc.reader_task
        if self.stderr_file is not None:
            self.stderr_file.close()
        if tree is not None:
            tree.snapshot()
            survivors_after_cleanup = [
                identity for identity in tree.identities if _same_process(identity)
            ]
        else:
            survivors_after_cleanup = []
        return {
            "pid": process.pid,
            "exit_code": process.returncode,
            "forced_cleanup": forced,
            "survivors_before_cleanup": survivors_before_cleanup,
            "survivors_after_cleanup": survivors_after_cleanup,
            "owned_processes": tree.details() if tree is not None else [],
            "stderr_log": str(self.stderr_path),
        }


class LocalResponsesFixture:
    """One deterministic Responses API turn, used only for history continuity."""

    def __init__(self, text: str):
        self.text = text
        self.requests = 0
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                fixture.requests += 1
                item = {
                    "id": f"idle-history-message-{fixture.requests}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": fixture.text}],
                }
                events = [
                    {
                        "type": "response.created",
                        "response": {"id": "idle-history-response"},
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "idle-history-response",
                            "status": "completed",
                            "output": [item],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    },
                ]
                body = "".join(
                    f"data: {json.dumps(event)}\n\n" for event in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _write_workspace(workspace: Path, *, serena: bool = False) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "sample.py").write_text(
        "class Widget:\n"
        "    def run(self, value: int) -> int:\n"
        "        return value + 7\n",
        encoding="utf-8",
    )
    if serena:
        project = workspace / ".serena"
        project.mkdir(parents=True, exist_ok=True)
        (project / "project.yml").write_text(
            "encoding: utf-8\n"
            "language_servers:\n"
            "- python\n"
            "project_name: codex-idle-acceptance\n"
            "read_only: true\n"
            "default_modes: []\n"
            "added_modes: []\n"
            "excluded_tools: []\n"
            "included_optional_tools: []\n"
            "fixed_tools: []\n",
            encoding="utf-8",
        )


def _fixture_spec(
    name: str,
    scenario_dir: Path,
    *,
    idle_timeout: float,
    child_idle_timeout: float,
    recovery: str = "stateless",
    opted_in: bool = True,
    startup_delay: float = 0.0,
) -> dict[str, object]:
    receipt = scenario_dir / f"{name}.receipt.jsonl"
    command = Path(sys.executable).resolve()
    args = [str(FIXTURE), "--receipt", str(receipt), "--instance", name]
    if startup_delay:
        args.extend(["--startup-delay", str(startup_delay)])
    spec: dict[str, object] = {
        "name": name,
        "command": str(command),
        "args": args,
        "receipt": receipt,
        "startup_timeout": 8.0,
        "tool_timeout": 8.0,
        "opted_in": opted_in,
    }
    if opted_in:
        spec.update(
            {
                "idle_timeout": idle_timeout,
                "child_idle_timeout": child_idle_timeout,
                "recovery": recovery,
            }
        )
    return spec


def _serena_spec(
    name: str,
    scenario_dir: Path,
    workspace: Path,
    command: str,
    *,
    idle_timeout: float,
    child_idle_timeout: float,
) -> dict[str, object]:
    receipt = scenario_dir / f"{name}.process-receipt.jsonl"
    serena_home = scenario_dir / "serena-home"
    return {
        "name": name,
        "command": command,
        "args": [
            "start-mcp-server",
            "--transport",
            "stdio",
            "--context",
            "idle-acceptance",
            "--project",
            str(workspace),
            "--enable-web-dashboard",
            "false",
            "--enable-gui-log-window",
            "false",
            "--open-web-dashboard",
            "false",
            "--log-level",
            "ERROR",
        ],
        "env": {
            "SERENA_HOME": str(serena_home),
            "PYTHONUNBUFFERED": "1",
        },
        "receipt": receipt,
        "startup_timeout": 12.0,
        "tool_timeout": 15.0,
        "idle_timeout": idle_timeout,
        "child_idle_timeout": child_idle_timeout,
        "recovery": "serena_startup_project",
        "opted_in": True,
        "workspace": workspace,
    }


def _prepare_serena_home(home: Path, workspace: Path) -> None:
    """Create only the Serena config/context needed by the isolated probe."""

    home.mkdir(parents=True, exist_ok=True)
    pyright = Path(
        "/data/CoordExp/.codex/serena/runtime/language-servers/pyright/bin/pyright-langserver"
    )
    config = (
        "language_backend: LSP\n"
        "line_ending: native\n"
        "gui_log_window: false\n"
        "web_dashboard: false\n"
        "web_dashboard_open_on_launch: false\n"
        "web_dashboard_interface: null\n"
        "web_dashboard_listen_address: 127.0.0.1\n"
        "log_level: 40\n"
        "trace_lsp_communication: false\n"
        "ls_specific_settings:\n"
        "  python:\n"
        f"    ls_path: {pyright}\n"
        "ignored_paths: []\n"
        "read_only_memory_patterns: []\n"
        "ignored_memory_patterns: []\n"
        "tool_timeout: 30\n"
        "excluded_tools: []\n"
        "included_optional_tools: []\n"
        "fixed_tools: []\n"
        "base_modes: []\n"
        "default_modes: null\n"
        "symbol_info_budget: 10\n"
        f"trusted_project_path_patterns:\n- {workspace}/**\n"
        f"projects:\n- {workspace}\n"
    )
    (home / "serena_config.yml").write_text(config, encoding="utf-8")
    context = (
        "description: Isolated context for the Core MCP idle acceptance probe.\n"
        "prompt: |\n"
        "  Use only the startup project supplied by the test. Keep all state in the isolated Serena home.\n"
        "fixed_tools:\n"
        "  - initial_instructions\n"
        "  - activate_project\n"
        "  - get_current_config\n"
        "  - get_symbols_overview\n"
        "  - find_symbol\n"
        "  - search_for_pattern\n"
        "single_project: false\n"
        "structured_tool_output: true\n"
    )
    contexts = home / "contexts"
    contexts.mkdir(parents=True, exist_ok=True)
    (contexts / "idle-acceptance.yml").write_text(context, encoding="utf-8")


def _render_config(
    home: Path,
    specs: Iterable[dict[str, object]],
    *,
    responses_port: int | None = None,
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    providers_url = (
        f"http://127.0.0.1:{responses_port}/v1"
        if responses_port is not None
        else "http://127.0.0.1:9/v1"
    )
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
        'name = "isolated local Responses fixture"',
        f"base_url = {_json_text(providers_url)}",
        'wire_api = "responses"',
        "requires_openai_auth = false",
        "",
    ]
    for spec in specs:
        name = str(spec["name"])
        lines.extend(
            [
                f"[mcp_servers.{name}]",
                f"command = {_json_text(spec['command'])}",
                f"args = {_json_text(spec['args'])}",
                f"startup_timeout_sec = {_json_text(float(spec['startup_timeout']))}",
                f"tool_timeout_sec = {_json_text(float(spec['tool_timeout']))}",
            ]
        )
        if bool(spec.get("opted_in")):
            lines.extend(
                [
                    f"idle_timeout_sec = {_json_text(float(spec['idle_timeout']))}",
                    f"idle_timeout_completed_subagent_sec = {_json_text(float(spec['child_idle_timeout']))}",
                    f"idle_recovery = {_json_text(str(spec['recovery']))}",
                ]
            )
        env = spec.get("env")
        if isinstance(env, dict) and env:
            lines.append("")
            lines.append(f"[mcp_servers.{name}.env]")
            for key, value in env.items():
                lines.append(f"{key} = {_json_text(str(value))}")
        lines.append("")
    path = home / "config.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _response_result(response: dict[str, object]) -> dict[str, object]:
    if "error" in response:
        raise RuntimeError(json.dumps(response["error"], sort_keys=True))
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"missing JSON-RPC result: {response!r}")
    return result


def _result_text(response: dict[str, object]) -> str:
    result = response.get("result")
    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(item.get("text", "")) for item in content if isinstance(item, dict)
    )


def _record_check(
    report: dict[str, object], key: str, condition: bool, detail: object = None
) -> None:
    assertions = report.setdefault("assertions", {})
    assert isinstance(assertions, dict)
    assertions[key] = {"passed": bool(condition), "detail": detail}
    if not condition:
        failures = report.setdefault("failures", [])
        assert isinstance(failures, list)
        failures.append(key)


def _start_events(spec: dict[str, object]) -> list[dict[str, object]]:
    return [
        event
        for event in _read_jsonl(Path(spec["receipt"]))
        if event.get("kind") == "started"
    ]


def _call_events(spec: dict[str, object]) -> list[dict[str, object]]:
    return [
        event
        for event in _read_jsonl(Path(spec["receipt"]))
        if event.get("kind") == "call_started"
    ]


def _latest_identity(spec: dict[str, object]) -> tuple[int, float] | None:
    events = _start_events(spec)
    if not events:
        return None
    event = events[-1]
    try:
        pid = int(event["pid"])
        if event.get("create_time") is not None:
            return pid, float(event["create_time"])
        # Resolve the creation time immediately after the fixture receipt.  A
        # generation is then safe to compare even after the process exits or a
        # later Core replacement reuses the numeric PID.
        return pid, psutil.Process(pid).create_time()
    except (KeyError, TypeError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _fixture_identity_from_text(text: str) -> tuple[str | None, int | None]:
    values: dict[str, str] = {}
    for part in text.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    try:
        pid = int(values["pid"])
    except (KeyError, TypeError, ValueError):
        pid = None
    return values.get("instance"), pid


async def _start_core(
    binary: Path,
    scenario_dir: Path,
    specs: list[dict[str, object]],
    *,
    responses_port: int | None = None,
) -> tuple[AppServer, str, dict[str, object]]:
    home = scenario_dir / "core-home"
    workspace = scenario_dir / "workspace"
    _write_workspace(workspace)
    config = _render_config(home, specs, responses_port=responses_port)
    app = AppServer(binary, home, workspace, scenario_dir / "core.stderr.log")
    await app.start()
    assert app.rpc is not None
    await app.rpc.request(
        "initialize",
        {
            "clientInfo": {"name": "codex-mcp-idle-acceptance", "version": "1"},
            "capabilities": {"experimentalApi": True},
        },
    )
    await app.rpc.notify("initialized")
    response = await app.rpc.request(
        "thread/start",
        {
            "cwd": str(workspace),
            "ephemeral": False,
            "model": "probe-model",
            "modelProvider": "probe",
            "approvalPolicy": "never",
        },
    )
    result = _response_result(response)
    thread = result.get("thread")
    if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
        raise RuntimeError(f"thread/start did not return a thread id: {response!r}")
    return (
        app,
        str(thread["id"]),
        {
            "config": str(config),
            "home": str(home),
            "workspace": str(workspace),
            "server_receipts": {
                str(spec["name"]): str(spec["receipt"])
                for spec in specs
                if spec.get("receipt") is not None
            },
            "thread_start": response,
        },
    )


async def _call_tool(
    app: AppServer, thread_id: str, server: str, tool: str, arguments: dict[str, object]
) -> dict[str, object]:
    assert app.rpc is not None
    return await app.rpc.request(
        "mcpServer/tool/call",
        {
            "threadId": thread_id,
            "server": server,
            "tool": tool,
            "arguments": arguments,
        },
    )


async def _unsubscribe(app: AppServer, thread_id: str) -> dict[str, object]:
    assert app.rpc is not None
    return await app.rpc.request("thread/unsubscribe", {"threadId": thread_id})


async def _passive_status(
    app: AppServer, thread_id: str, count: int = 3
) -> list[dict[str, object]]:
    assert app.rpc is not None
    responses = []
    for _ in range(count):
        responses.append(
            await app.rpc.request(
                "mcpServerStatus/list",
                {"threadId": thread_id, "detail": "full"},
            )
        )
        await asyncio.sleep(0.12)
    return responses


async def _finish_scenario(
    report: dict[str, object],
    app: AppServer | None,
    fixture: LocalResponsesFixture | None,
) -> None:
    if app is not None:
        core = await app.close()
        report["core"] = core
        _record_check(
            report,
            "owned_process_cleanup_complete",
            not core.get("survivors_after_cleanup"),
            {
                "survivors_before_cleanup": core.get("survivors_before_cleanup"),
                "survivors_after_cleanup": core.get("survivors_after_cleanup"),
            },
        )
        _record_check(report, "core_exited_cleanly", core.get("exit_code") == 0, core)
        if app.rpc is not None:
            report["rpc_notifications"] = list(app.rpc.notifications)
            report["rpc_trace"] = app.rpc.trace
    if fixture is not None:
        report["responses_fixture_requests"] = fixture.requests
        fixture.stop()
    failures = report.get("failures", [])
    if not isinstance(failures, list):
        failures = []
    report["status"] = "PASS" if not failures else "RED"


async def scenario_reclaim_and_reuse(
    binary: Path, root: Path, main_idle: float, child_idle: float
) -> dict[str, object]:
    name = "reclaim_and_reuse"
    scenario_dir = root / name
    scenario_dir.mkdir(parents=True)
    report: dict[str, object] = {
        "name": name,
        "required": True,
        "claim": "A same-task MCP demand reconstructs one intentionally suspended stdio client.",
        "policy": {
            "idle_timeout_sec": main_idle,
            "idle_timeout_completed_subagent_sec": child_idle,
            "idle_recovery": "stateless",
        },
    }
    spec = _fixture_spec(
        "eligible",
        scenario_dir,
        idle_timeout=main_idle,
        child_idle_timeout=child_idle,
    )
    history_fixture = LocalResponsesFixture("idle history fixture turn completed")
    fixture_port = history_fixture.start()
    app: AppServer | None = None
    try:
        app, thread_id, setup = await _start_core(
            binary, scenario_dir, [spec], responses_port=fixture_port
        )
        report.update(setup)
        report["thread_id_before"] = thread_id
        initial = await _call_tool(app, thread_id, "eligible", "identity", {})
        _response_result(initial)
        initial_text = _result_text(initial)
        report["initial_identity"] = initial_text
        initial_identity = _latest_identity(spec)
        _record_check(report, "initial_call_succeeded", bool(initial_text), initial)
        _record_check(report, "initial_process_receipt", initial_identity is not None)

        turn_response = await app.rpc.request(  # type: ignore[union-attr]
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": "record one local fixture turn"}],
            },
        )
        _response_result(turn_response)
        completed = await app.rpc.wait_notification(  # type: ignore[union-attr]
            "turn/completed",
            predicate=lambda msg: msg.get("params", {}).get("threadId") == thread_id,
        )
        report["turn_completed"] = completed
        report["fixture_turns"] = 1
        _record_check(report, "synthetic_turn_completed", True, completed)

        await _unsubscribe(app, thread_id)
        suspended, elapsed = await _wait_until(
            lambda: initial_identity is not None
            and not _same_process(initial_identity),
            max(WAIT_TIMEOUT, main_idle * 4),
        )
        report["suspension_wait_seconds"] = round(elapsed, 3)
        report["initial_process_dead"] = suspended
        _record_check(
            report,
            "eligible_process_exited_after_idle",
            suspended,
            _safe_process_details(initial_identity) if initial_identity else None,
        )

        statuses = await _passive_status(app, thread_id)
        report["passive_status_responses"] = statuses
        after_status_starts = len(_start_events(spec))
        _record_check(
            report,
            "passive_status_does_not_wake",
            suspended and after_status_starts == 1,
            {"started_processes": after_status_starts},
        )

        recovered = await _call_tool(app, thread_id, "eligible", "identity", {})
        recovered_result = _response_result(recovered)
        recovered_text = _result_text(recovered)
        report["recovered_identity"] = recovered_text
        recovered_identity = _latest_identity(spec)
        recovered_instance, recovered_pid_from_text = _fixture_identity_from_text(
            recovered_text
        )
        _record_check(
            report, "same_task_call_succeeded", bool(recovered_text), recovered_result
        )
        _record_check(
            report,
            "replacement_pid_is_distinct",
            suspended
            and initial_identity is not None
            and recovered_identity is not None
            and recovered_identity != initial_identity,
            {
                "initial": initial_identity,
                "replacement": recovered_identity,
                "pid_in_response": recovered_pid_from_text,
            },
        )
        _record_check(
            report,
            "one_replacement_and_no_replay",
            len(_start_events(spec)) == 2 and len(_call_events(spec)) == 2,
            {
                "started_processes": len(_start_events(spec)),
                "business_calls": len(_call_events(spec)),
            },
        )

        read = await app.rpc.request(  # type: ignore[union-attr]
            "thread/read", {"threadId": thread_id, "includeTurns": True}
        )
        read_result = _response_result(read)
        read_thread = read_result.get("thread")
        history_blob = json.dumps(read_thread, sort_keys=True)
        report["thread_read"] = read
        _record_check(
            report,
            "same_thread_and_workspace",
            isinstance(read_thread, dict)
            and read_thread.get("id") == thread_id
            and read_thread.get("cwd") == setup["workspace"],
            {"thread_id": thread_id, "read_thread": read_thread},
        )
        _record_check(
            report,
            "persisted_fixture_history_survives",
            "idle history fixture turn completed" in history_blob,
            {
                "history_contains_fixture": "idle history fixture turn completed"
                in history_blob
            },
        )
    except Exception as exc:
        report["error"] = repr(exc)
        _record_check(report, "scenario_execution", False, repr(exc))
    finally:
        await _finish_scenario(report, app, history_fixture)
    return report


async def scenario_neighbor_preserved(
    binary: Path, root: Path, main_idle: float, child_idle: float
) -> dict[str, object]:
    name = "neighbor_preserved"
    scenario_dir = root / name
    scenario_dir.mkdir(parents=True)
    report: dict[str, object] = {
        "name": name,
        "required": True,
        "claim": "A second active task using an opted-out stdio server stays connected while the eligible peer is reclaimed.",
    }
    eligible = _fixture_spec(
        "eligible", scenario_dir, idle_timeout=main_idle, child_idle_timeout=child_idle
    )
    neighbor = _fixture_spec(
        "neighbor",
        scenario_dir,
        idle_timeout=main_idle,
        child_idle_timeout=child_idle,
        opted_in=False,
    )
    app: AppServer | None = None
    try:
        app, thread_id, setup = await _start_core(
            binary, scenario_dir, [eligible, neighbor]
        )
        report.update(setup)
        second_start = await app.rpc.request(  # type: ignore[union-attr]
            "thread/start",
            {
                "cwd": str(setup["workspace"]),
                "ephemeral": False,
                "model": "probe-model",
                "modelProvider": "probe",
                "approvalPolicy": "never",
            },
        )
        second_thread = _response_result(second_start).get("thread")
        if not isinstance(second_thread, dict) or not isinstance(
            second_thread.get("id"), str
        ):
            raise RuntimeError(
                f"neighbor thread/start did not return an id: {second_start!r}"
            )
        neighbor_thread_id = str(second_thread["id"])
        report["neighbor_thread_id"] = neighbor_thread_id
        _record_check(
            report, "neighbor_task_is_distinct", neighbor_thread_id != thread_id
        )
        first_eligible = await _call_tool(app, thread_id, "eligible", "identity", {})
        first_neighbor = await _call_tool(
            app, neighbor_thread_id, "neighbor", "identity", {}
        )
        _response_result(first_eligible)
        _response_result(first_neighbor)
        eligible_identity = _latest_identity(eligible)
        neighbor_identity = _latest_identity(neighbor)
        await asyncio.sleep(0.2)
        neighbor_initial_identities = [
            (int(event["pid"]), float(event["create_time"]))
            for event in _start_events(neighbor)
            if event.get("create_time") is not None
        ]
        _record_check(
            report,
            "both_initial_calls_succeeded",
            bool(_result_text(first_eligible)) and bool(_result_text(first_neighbor)),
        )
        _record_check(
            report,
            "neighbor_initial_process_receipt",
            bool(neighbor_initial_identities),
        )
        await _unsubscribe(app, thread_id)
        eligible_dead, _ = await _wait_until(
            lambda: eligible_identity is not None
            and not _same_process(eligible_identity),
            max(WAIT_TIMEOUT, main_idle * 4),
        )
        neighbor_alive = all(
            _same_process(identity) for identity in neighbor_initial_identities
        )
        report["eligible_initial_process_dead"] = eligible_dead
        report["neighbor_alive_while_eligible_dead"] = neighbor_alive
        _record_check(report, "eligible_reclaimed", eligible_dead)
        _record_check(
            report,
            "neighbor_processes_untouched",
            neighbor_alive,
            [
                _safe_process_details(identity)
                for identity in neighbor_initial_identities
            ],
        )
        second_neighbor = await _call_tool(
            app, neighbor_thread_id, "neighbor", "identity", {}
        )
        _response_result(second_neighbor)
        second_neighbor_identity = _latest_identity(neighbor)
        _record_check(
            report,
            "neighbor_call_reuses_original_pid",
            bool(_result_text(second_neighbor))
            and neighbor_identity is not None
            and second_neighbor_identity is not None
            and second_neighbor_identity == neighbor_identity,
            {"initial": neighbor_identity, "after": second_neighbor_identity},
        )
        _record_check(
            report,
            "neighbor_has_no_replacement",
            len(_start_events(neighbor)) == len(neighbor_initial_identities)
            and len(_call_events(neighbor)) == 2,
            {
                "started_processes": len(_start_events(neighbor)),
                "initial_instances": len(neighbor_initial_identities),
                "business_calls": len(_call_events(neighbor)),
            },
        )
    except Exception as exc:
        report["error"] = repr(exc)
        _record_check(report, "scenario_execution", False, repr(exc))
    finally:
        await _finish_scenario(report, app, None)
    return report


async def scenario_concurrent_cold_demand(
    binary: Path, root: Path, main_idle: float, child_idle: float
) -> dict[str, object]:
    name = "concurrent_cold_demand"
    scenario_dir = root / name
    scenario_dir.mkdir(parents=True)
    report: dict[str, object] = {
        "name": name,
        "required": True,
        "claim": "Two cold same-task calls share one MCP replacement and each receives a valid response.",
    }
    spec = _fixture_spec(
        "eligible",
        scenario_dir,
        idle_timeout=main_idle,
        child_idle_timeout=child_idle,
        startup_delay=0.35,
    )
    app: AppServer | None = None
    try:
        app, thread_id, setup = await _start_core(binary, scenario_dir, [spec])
        report.update(setup)
        first = await _call_tool(app, thread_id, "eligible", "identity", {})
        _response_result(first)
        initial_identity = _latest_identity(spec)
        await _unsubscribe(app, thread_id)
        suspended, _ = await _wait_until(
            lambda: initial_identity is not None
            and not _same_process(initial_identity),
            max(WAIT_TIMEOUT, main_idle * 4),
        )
        report["initial_process_dead"] = suspended
        _record_check(report, "eligible_suspended_before_cold_demand", suspended)
        cold_responses = await asyncio.gather(
            _call_tool(app, thread_id, "eligible", "identity", {}),
            _call_tool(app, thread_id, "eligible", "identity", {}),
        )
        cold_texts = [_result_text(response) for response in cold_responses]
        report["cold_identities"] = cold_texts
        cold_pids = [_fixture_identity_from_text(text)[1] for text in cold_texts]
        all_ok = all(
            "result" in response and text
            for response, text in zip(cold_responses, cold_texts)
        )
        _record_check(report, "both_cold_calls_succeeded", all_ok, cold_responses)
        _record_check(
            report,
            "both_cold_calls_share_pid",
            len(set(pid for pid in cold_pids if pid is not None)) == 1,
            cold_pids,
        )
        _record_check(
            report,
            "single_replacement_and_no_replay",
            len(_start_events(spec)) == 2 and len(_call_events(spec)) == 3,
            {
                "started_processes": len(_start_events(spec)),
                "business_calls": len(_call_events(spec)),
            },
        )
    except Exception as exc:
        report["error"] = repr(exc)
        _record_check(report, "scenario_execution", False, repr(exc))
    finally:
        await _finish_scenario(report, app, None)
    return report


async def scenario_inflight_and_background(
    binary: Path, root: Path, main_idle: float, child_idle: float
) -> dict[str, object]:
    name = "inflight_and_background"
    scenario_dir = root / name
    scenario_dir.mkdir(parents=True)
    report: dict[str, object] = {
        "name": name,
        "required": True,
        "claim": "An in-flight MCP operation and a real command/exec workload survive the idle deadline.",
    }
    spec = _fixture_spec(
        "eligible", scenario_dir, idle_timeout=main_idle, child_idle_timeout=child_idle
    )
    app: AppServer | None = None
    background_task: asyncio.Task[dict[str, object]] | None = None
    try:
        app, thread_id, setup = await _start_core(binary, scenario_dir, [spec])
        report.update(setup)
        initial = await _call_tool(app, thread_id, "eligible", "identity", {})
        _response_result(initial)
        initial_identity = _latest_identity(spec)
        _record_check(report, "initial_call_succeeded", bool(_result_text(initial)))
        assert app.rpc is not None
        marker = "background-exec-idle-acceptance-ok"
        command = [
            str(Path(sys.executable).resolve()),
            "-c",
            "import time; time.sleep(2.0); print(%r, flush=True)" % marker,
        ]
        background_task = asyncio.create_task(
            app.rpc.request(
                "command/exec",
                {
                    "command": command,
                    "processId": "idle-acceptance-background",
                    "streamStdoutStderr": False,
                },
                timeout=10.0,
            )
        )
        await asyncio.sleep(0.2)
        inflight = asyncio.create_task(
            _call_tool(app, thread_id, "eligible", "sleep", {"seconds": 2.0})
        )
        started_event = await _wait_for_event(
            Path(spec["receipt"]),
            lambda event: event.get("kind") == "call_started"
            and event.get("name") == "sleep",
            timeout=3.0,
        )
        _record_check(
            report,
            "sleep_call_entered_fixture",
            started_event is not None,
            started_event,
        )
        await _unsubscribe(app, thread_id)
        await asyncio.sleep(max(1.2, main_idle * 1.8))
        alive_during_call = _same_process(initial_identity)
        report["mcp_alive_beyond_idle_while_inflight"] = alive_during_call
        _record_check(
            report,
            "inflight_process_not_reclaimed",
            alive_during_call,
            _safe_process_details(initial_identity) if initial_identity else None,
        )
        inflight_response = await asyncio.wait_for(inflight, timeout=5.0)
        _response_result(inflight_response)
        _record_check(
            report,
            "inflight_call_succeeded",
            bool(_result_text(inflight_response)),
            inflight_response,
        )
        background_response = await asyncio.wait_for(background_task, timeout=5.0)
        background_result = _response_result(background_response)
        report["background_exec_response"] = background_response
        _record_check(
            report,
            "background_exec_completed",
            background_result.get("exitCode") == 0
            and marker in str(background_result.get("stdout", "")),
            background_result,
        )
        eventually_dead, _ = await _wait_until(
            lambda: initial_identity is not None
            and not _same_process(initial_identity),
            max(WAIT_TIMEOUT, main_idle * 4),
        )
        _record_check(report, "idle_reclaim_after_inflight_completion", eventually_dead)
    except Exception as exc:
        report["error"] = repr(exc)
        _record_check(report, "scenario_execution", False, repr(exc))
        if background_task is not None and not background_task.done():
            background_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await background_task
    finally:
        await _finish_scenario(report, app, None)
    return report


def _find_serena_identity(app: AppServer) -> tuple[int, float] | None:
    if app.tree is None:
        return None
    app.tree.snapshot()
    for identity in app.tree.identities:
        if not _same_process(identity):
            continue
        try:
            process = psutil.Process(identity[0])
            command_line = " ".join(process.cmdline()).lower()
            if (
                "serena" in command_line
                and "start-mcp-server" in command_line
                and identity != app.tree.root
            ):
                return identity
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


async def scenario_serena_context(
    binary: Path,
    root: Path,
    main_idle: float,
    child_idle: float,
    command: str | None,
) -> dict[str, object]:
    name = "serena_startup_project"
    scenario_dir = root / name
    scenario_dir.mkdir(parents=True)
    report: dict[str, object] = {
        "name": name,
        "required": False,
        "claim": "When available, Serena is restored in the exact startup worktree after MCP suspension.",
    }
    if command is None:
        report.update(
            {
                "status": "UNPROVEN",
                "gap": "installed Serena executable was not found; no substitute server is treated as Serena evidence",
            }
        )
        return report
    workspace = scenario_dir / "workspace"
    _write_workspace(workspace, serena=True)
    spec = _serena_spec(
        "serena",
        scenario_dir,
        workspace,
        command,
        idle_timeout=main_idle,
        child_idle_timeout=child_idle,
    )
    _prepare_serena_home(Path(spec["env"]["SERENA_HOME"]), workspace)
    # _start_core creates the ordinary workspace itself; replace it with the
    # exact Serena worktree before Core reads config and starts the server.
    app: AppServer | None = None
    try:
        home = scenario_dir / "core-home"
        config = _render_config(home, [spec])
        app = AppServer(binary, home, workspace, scenario_dir / "core.stderr.log")
        await app.start()
        assert app.rpc is not None
        await app.rpc.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "codex-mcp-idle-serena-acceptance",
                    "version": "1",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await app.rpc.notify("initialized")
        started = await app.rpc.request(
            "thread/start",
            {
                "cwd": str(workspace),
                "ephemeral": False,
                "model": "probe-model",
                "modelProvider": "probe",
                "approvalPolicy": "never",
            },
        )
        thread = _response_result(started).get("thread")
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
            raise RuntimeError(f"Serena thread/start did not return an id: {started!r}")
        thread_id = str(thread["id"])
        report.update(
            {
                "config": str(config),
                "workspace": str(workspace),
                "thread_id": thread_id,
                "serena_command": command,
                "serena_home": str(spec["env"]["SERENA_HOME"]),
            }
        )
        arguments = {
            "name_path_pattern": "Widget/run",
            "relative_path": "sample.py",
            "include_body": True,
        }
        initial = await _call_tool(app, thread_id, "serena", "find_symbol", arguments)
        _response_result(initial)
        first_identity = _find_serena_identity(app)
        first_blob = json.dumps(initial, sort_keys=True)
        _record_check(
            report,
            "initial_serena_query_succeeded",
            "return value + 7" in first_blob,
            initial,
        )
        _record_check(report, "initial_serena_process_seen", first_identity is not None)
        if first_identity is None:
            raise RuntimeError(
                "Serena process identity was not found among Core descendants"
            )
        await _unsubscribe(app, thread_id)
        dead, _ = await _wait_until(
            lambda: not _same_process(first_identity), max(8.0, main_idle * 5)
        )
        _record_check(
            report,
            "serena_process_reclaimed",
            dead,
            _safe_process_details(first_identity),
        )
        status_responses = await _passive_status(app, thread_id)
        report["passive_status_responses"] = status_responses
        second = await _call_tool(app, thread_id, "serena", "find_symbol", arguments)
        _response_result(second)
        second_identity = _find_serena_identity(app)
        second_blob = json.dumps(second, sort_keys=True)
        _record_check(
            report,
            "restored_serena_query_succeeded",
            "return value + 7" in second_blob,
            second,
        )
        _record_check(
            report,
            "restored_process_is_one_replacement",
            dead and second_identity is not None and second_identity != first_identity,
            {"first": first_identity, "second": second_identity},
        )
        _record_check(
            report,
            "restored_start_command_keeps_exact_worktree",
            second_identity is not None
            and str(workspace).lower()
            in " ".join(psutil.Process(second_identity[0]).cmdline()).lower(),
            {
                "workspace": str(workspace),
                "process": _safe_process_details(second_identity)
                if second_identity
                else None,
            },
        )
    except Exception as exc:
        report["error"] = repr(exc)
        _record_check(report, "scenario_execution", False, repr(exc))
    finally:
        await _finish_scenario(report, app, None)
    initial_assertion = report.get("assertions", {}).get(
        "initial_serena_query_succeeded"
    )
    if not isinstance(initial_assertion, dict) or not initial_assertion.get("passed"):
        report["status"] = "UNPROVEN"
        report["gap"] = (
            "Serena did not reach a successful initial query in this environment; "
            "the idle policy is not judged from an uninitialized server."
        )
    return report


def _binary_info(binary: Path) -> dict[str, object]:
    info: dict[str, object] = {"path": str(binary), "resolved": str(binary.resolve())}
    try:
        info["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError as exc:
        info["sha256_error"] = repr(exc)
    try:
        version = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        info["version_exit_code"] = version.returncode
        info["version_stdout"] = version.stdout.strip()
        info["version_stderr"] = version.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["version_error"] = repr(exc)
    return info


def _find_serena(command: str | None) -> str | None:
    if command:
        path = Path(command)
        return str(path) if path.exists() and os.access(path, os.X_OK) else None
    for candidate in ("/root/.local/bin/serena", shutil.which("serena")):
        if candidate and Path(candidate).exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


async def run(args: argparse.Namespace) -> int:
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"candidate binary is not executable: {binary}")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("run-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    run_dir = output / run_name
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = output / f"{run_name}-{suffix}"
    run_dir.mkdir()
    report: dict[str, object] = {
        "schema_version": 1,
        "harness": str(Path(__file__).resolve()),
        "fixture": str(FIXTURE),
        "run_label": args.run_label,
        "binary": _binary_info(binary),
        "output_dir": str(output),
        "run_dir": str(run_dir),
        "timers": {
            "idle_timeout_sec": args.idle_timeout,
            "idle_timeout_completed_subagent_sec": args.child_idle_timeout,
            "wait_timeout_sec": WAIT_TIMEOUT,
        },
        "no_model_or_gpu": True,
        "scenarios": [],
        "gaps": [
            {
                "area": "completed_subagent_grace",
                "status": "UNPROVEN",
                "reason": "This bounded harness does not synthesize native-agent spawn through a model; no completed-subagent PASS is claimed.",
            },
            {
                "area": "wake_frontend_independence",
                "status": "UNPROVEN",
                "reason": "No production wake registration or task wake is created; daemon/ledger state is intentionally untouched.",
            },
        ],
    }
    report["scenarios"].append(
        await scenario_reclaim_and_reuse(
            binary, run_dir, args.idle_timeout, args.child_idle_timeout
        )
    )
    report["scenarios"].append(
        await scenario_neighbor_preserved(
            binary, run_dir, args.idle_timeout, args.child_idle_timeout
        )
    )
    report["scenarios"].append(
        await scenario_concurrent_cold_demand(
            binary, run_dir, args.idle_timeout, args.child_idle_timeout
        )
    )
    report["scenarios"].append(
        await scenario_inflight_and_background(
            binary, run_dir, args.idle_timeout, args.child_idle_timeout
        )
    )
    serena_command = None if args.skip_serena else _find_serena(args.serena_command)
    if args.skip_serena:
        report["gaps"].append(
            {
                "area": "serena_startup_project",
                "status": "UNPROVEN",
                "reason": "Serena scenario was explicitly skipped.",
            }
        )
    report["scenarios"].append(
        await scenario_serena_context(
            binary,
            run_dir,
            args.idle_timeout,
            args.child_idle_timeout,
            serena_command,
        )
    )
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    required = [item for item in scenarios if item.get("required")]
    required_red = [item["name"] for item in required if item.get("status") != "PASS"]
    optional_red = [
        item["name"]
        for item in scenarios
        if not item.get("required") and item.get("status") == "RED"
    ]
    report["required_red"] = required_red
    report["optional_red"] = optional_red
    report["status"] = (
        "PASS_WITH_GAPS" if not required_red and not optional_red else "RED"
    )
    report["expected_stock_baseline_red"] = (
        str(args.run_label).lower().startswith("stock")
    )
    receipt = run_dir / "idle-mcp-acceptance-receipt.json"
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "receipt": str(receipt),
                "required_red": required_red,
                "optional_red": optional_red,
                "expected_stock_baseline_red": report["expected_stock_baseline_red"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["status"] != "RED" else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", required=True, help="Exact candidate codex executable to exercise."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Dedicated directory for isolated homes, logs, and receipts.",
    )
    parser.add_argument(
        "--run-label", default="candidate", help="Label stored in the receipt."
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_MAIN_IDLE,
        help=f"Short positive main-task grace in seconds (default: {DEFAULT_MAIN_IDLE}).",
    )
    parser.add_argument(
        "--child-idle-timeout",
        type=float,
        default=DEFAULT_CHILD_IDLE,
        help=f"Short positive completed-subagent grace in seconds (default: {DEFAULT_CHILD_IDLE}).",
    )
    parser.add_argument(
        "--serena-command",
        help="Optional exact Serena executable; otherwise /root/.local/bin/serena or PATH is used.",
    )
    parser.add_argument(
        "--skip-serena",
        action="store_true",
        help="Record Serena as UNPROVEN without launching it.",
    )
    args = parser.parse_args()
    if args.idle_timeout <= 0 or args.child_idle_timeout <= 0:
        parser.error("idle timers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(_parse_args())))
