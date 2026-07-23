#!/usr/bin/env python3
"""Supervise one reusable Pi RPC thread from Pi's worktree-local home.

The supervisor imports no Codex code. It binds one persistent Pi session to
one prepared sandbox/workspace, exposes a small JSONL command surface, records
the raw Pi RPC stream, and can be restarted without resetting conversation
context.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import threading
from typing import Any, BinaryIO, IO, Iterator, Mapping
from urllib.parse import urlparse
import uuid


THREAD_SCHEMA_VERSION = "pi_thread_spec.v1"
RUNTIME_SCHEMA_VERSION = "pi_thread_runtime.v1"
MANIFEST_SCHEMA_VERSION = "pi_thread_manifest.v1"
PROTOCOL_VERSION = "pi_thread_protocol.v1"
ALLOWED_REASONING = frozenset({"medium", "high", "xhigh"})
ALLOWED_EVENT_STREAMS = frozenset({"lifecycle", "all"})
LIFECYCLE_EVENTS = frozenset(
    {
        "agent_start",
        "agent_settled",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "queue_update",
        "compaction_start",
        "compaction_end",
        "auto_retry_start",
        "auto_retry_end",
        "extension_error",
    }
)
ALLOWED_COMMANDS = frozenset(
    {
        "prompt",
        "steer",
        "follow_up",
        "abort",
        "compact",
        "get_state",
        "get_messages",
        "get_stats",
        "get_last",
        "get_entries",
        "status",
        "close",
    }
)
_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")


class PiThreadError(ValueError):
    """Raised when a thread contract or lifecycle request is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_object(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_object(path: str | Path, *, schema: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PiThreadError(f"JSON document must be an object: {source}")
    result = dict(value)
    if result.get("schema_version") != schema:
        raise PiThreadError(f"{source} schema_version must be {schema!r}")
    return result


def _string(value: Any, *, field: str, identity: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PiThreadError(f"{field} must be a non-empty string")
    result = value.strip()
    if identity and not _IDENTITY_PATTERN.fullmatch(result):
        raise PiThreadError(f"{field} contains unsupported characters")
    return result


def _inside_path(value: Any, *, field: str) -> str:
    raw = _string(value, field=field)
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise PiThreadError(f"{field} must be an absolute normalized sandbox path")
    return str(path)


def _local_proxy(value: Any) -> str:
    proxy = _string(value, field="runtime.proxy_url")
    parsed = urlparse(proxy)
    if parsed.scheme not in {"http", "https", "socks5", "socks5h"}:
        raise PiThreadError("runtime.proxy_url uses an unsupported scheme")
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port is None:
        raise PiThreadError("runtime.proxy_url must use an explicit local proxy port")
    if parsed.username is not None or parsed.password is not None:
        raise PiThreadError("runtime.proxy_url must not embed credentials")
    return proxy


@dataclass(frozen=True)
class PiThreadSpec:
    thread_id: str
    provider: str
    model: str
    reasoning: str
    auto_compaction: bool
    auto_retry: bool
    event_stream: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PiThreadSpec":
        reasoning = _string(value.get("reasoning"), field="thread.reasoning")
        if reasoning not in ALLOWED_REASONING:
            raise PiThreadError(
                f"thread.reasoning must be one of {sorted(ALLOWED_REASONING)}; max is intentionally unsupported"
            )
        auto_compaction = value.get("auto_compaction", True)
        auto_retry = value.get("auto_retry", True)
        if not isinstance(auto_compaction, bool) or not isinstance(auto_retry, bool):
            raise PiThreadError("thread auto_compaction and auto_retry must be booleans")
        event_stream = value.get("event_stream", "lifecycle")
        if event_stream not in ALLOWED_EVENT_STREAMS:
            raise PiThreadError(f"thread.event_stream must be one of {sorted(ALLOWED_EVENT_STREAMS)}")
        return cls(
            thread_id=_string(value.get("thread_id"), field="thread.thread_id", identity=True),
            provider=_string(value.get("provider"), field="thread.provider", identity=True),
            model=_string(value.get("model"), field="thread.model", identity=True),
            reasoning=reasoning,
            auto_compaction=auto_compaction,
            auto_retry=auto_retry,
            event_stream=event_stream,
        )


@dataclass(frozen=True)
class PiThreadRuntime:
    sandbox_root: Path
    workspace: str
    node_path: str
    pi_cli_path: str
    home: str
    agent_dir: str
    session_dir: str
    proxy_url: str
    userspec: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PiThreadRuntime":
        root = Path(_string(value.get("sandbox_root"), field="runtime.sandbox_root"))
        if not root.is_absolute():
            raise PiThreadError("runtime.sandbox_root must be absolute")
        root = root.resolve(strict=True)
        userspec = _string(value.get("userspec", "65534:65534"), field="runtime.userspec")
        if not re.fullmatch(r"[0-9]+:[0-9]+", userspec):
            raise PiThreadError("runtime.userspec must be numeric uid:gid")
        runtime = cls(
            sandbox_root=root,
            workspace=_inside_path(value.get("workspace", "/workspace"), field="runtime.workspace"),
            node_path=_inside_path(value.get("node_path"), field="runtime.node_path"),
            pi_cli_path=_inside_path(value.get("pi_cli_path"), field="runtime.pi_cli_path"),
            home=_inside_path(value.get("home"), field="runtime.home"),
            agent_dir=_inside_path(value.get("agent_dir"), field="runtime.agent_dir"),
            session_dir=_inside_path(value.get("session_dir"), field="runtime.session_dir"),
            proxy_url=_local_proxy(value.get("proxy_url")),
            userspec=userspec,
        )
        for field, sandbox_path in (
            ("runtime.node_path", runtime.node_path),
            ("runtime.pi_cli_path", runtime.pi_cli_path),
        ):
            if not runtime.host_path(sandbox_path).is_file():
                raise PiThreadError(f"{field} does not resolve to a file inside the sandbox")
        if not runtime.host_path(runtime.workspace).is_dir():
            raise PiThreadError("runtime.workspace does not resolve to a directory inside the sandbox")
        return runtime

    def host_path(self, sandbox_path: str) -> Path:
        return self.sandbox_root.joinpath(sandbox_path.lstrip("/"))


def load_thread_spec(path: str | Path) -> tuple[PiThreadSpec, dict[str, Any]]:
    raw = _read_object(path, schema=THREAD_SCHEMA_VERSION)
    return PiThreadSpec.from_mapping(raw), raw


def load_runtime(path: str | Path) -> tuple[PiThreadRuntime, dict[str, Any]]:
    raw = _read_object(path, schema=RUNTIME_SCHEMA_VERSION)
    return PiThreadRuntime.from_mapping(raw), raw


def build_pi_rpc_command(thread: PiThreadSpec, runtime: PiThreadRuntime, *, session_id: str) -> list[str]:
    uuid.UUID(session_id)
    pi_command = [
        runtime.node_path,
        runtime.pi_cli_path,
        "--mode",
        "rpc",
        "--provider",
        thread.provider,
        "--model",
        thread.model,
        "--thinking",
        thread.reasoning,
        "--session-id",
        session_id,
        "--session-dir",
        runtime.session_dir,
        "--name",
        thread.thread_id,
        "--no-approve",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
    ]
    return [
        "/usr/sbin/chroot",
        f"--userspec={runtime.userspec}",
        str(runtime.sandbox_root),
        "/bin/bash",
        "-c",
        f"cd {shlex.quote(runtime.workspace)} && exec {shlex.join(pi_command)}",
    ]


def translate_command(command: Mapping[str, Any]) -> dict[str, Any] | None:
    command_type = _string(command.get("type"), field="command.type")
    if command_type not in ALLOWED_COMMANDS:
        raise PiThreadError(f"unsupported command type: {command_type}")
    if command_type in {"status", "close"}:
        return None
    request_id = _request_id(command)
    if request_id is not None:
        request_id = _string(request_id, field="command.id", identity=True)
    translated_type = {
        "get_stats": "get_session_stats",
        "get_last": "get_last_assistant_text",
    }.get(command_type, command_type)
    result: dict[str, Any] = {"type": translated_type}
    if request_id is not None:
        result["id"] = request_id
    if command_type in {"prompt", "steer", "follow_up"}:
        result["message"] = _string(command.get("message"), field="command.message")
        images = command.get("images")
        if images is not None:
            if not isinstance(images, list):
                raise PiThreadError("command.images must be an array")
            result["images"] = images
        if command_type == "prompt" and command.get("streaming_behavior") is not None:
            behavior = command["streaming_behavior"]
            if behavior not in {"steer", "followUp"}:
                raise PiThreadError("command.streaming_behavior must be steer or followUp")
            result["streamingBehavior"] = behavior
    elif command_type == "compact" and command.get("instructions") is not None:
        result["customInstructions"] = _string(command["instructions"], field="command.instructions")
    elif command_type == "get_entries" and command.get("since") is not None:
        result["since"] = _string(command["since"], field="command.since", identity=True)
    return result


def _request_id(command: Mapping[str, Any]) -> str | None:
    """Resolve the stable request identifier while retaining the short v0 alias."""

    request_id = command.get("request_id")
    alias = command.get("id")
    if request_id is not None and alias is not None and request_id != alias:
        raise PiThreadError("command.request_id and command.id must match when both are provided")
    value = request_id if request_id is not None else alias
    if value is None:
        return None
    return _string(value, field="command.request_id", identity=True)


def iter_lf_records(stream: IO[bytes], *, chunk_size: int = 65536) -> Iterator[bytes]:
    """Yield strict LF-delimited records without splitting on Unicode separators."""

    pending = bytearray()
    read_chunk = getattr(stream, "read1", stream.read)
    while True:
        chunk = read_chunk(chunk_size)
        if not chunk:
            break
        pending.extend(chunk)
        while True:
            offset = pending.find(b"\n")
            if offset < 0:
                break
            record = bytes(pending[:offset])
            del pending[: offset + 1]
            if record.endswith(b"\r"):
                record = record[:-1]
            if record:
                yield record
    if pending:
        yield bytes(pending)


class PiThreadSupervisor:
    def __init__(
        self,
        thread: PiThreadSpec,
        runtime: PiThreadRuntime,
        *,
        thread_dir: Path,
        thread_hash: str,
        runtime_hash: str,
    ) -> None:
        self.thread = thread
        self.runtime = runtime
        self.thread_dir = thread_dir.expanduser().resolve()
        self.thread_hash = thread_hash
        self.runtime_hash = runtime_hash
        self.output_lock = threading.Lock()
        self.stdin_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.ready_condition = threading.Condition(self.state_lock)
        self.process: subprocess.Popen[bytes] | None = None
        self.reader_thread: threading.Thread | None = None
        self.stderr_handle: BinaryIO | None = None
        self.event_handle: BinaryIO | None = None
        self.command_handle: BinaryIO | None = None
        self.last_event_type: str | None = None
        self.is_streaming = False
        self.settled_count = 0
        self.startup_response: dict[str, Any] | None = None
        self.eof = False
        self._lock_handle: BinaryIO | None = None
        self.session_id = self._open_manifest()

    def _open_manifest(self) -> str:
        self.thread_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.thread_dir, 0o700)
        lock_path = self.thread_dir / "supervisor.lock"
        self._lock_handle = lock_path.open("a+b")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PiThreadError(f"thread already has a live supervisor: {self.thread_dir}") from exc
        manifest_path = self.thread_dir / "manifest.json"
        if manifest_path.exists():
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping) or value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                raise PiThreadError("existing thread manifest is malformed")
            if value.get("thread_spec_sha256") != self.thread_hash:
                raise PiThreadError("thread spec changed for an existing persistent thread")
            if value.get("runtime_spec_sha256") != self.runtime_hash:
                raise PiThreadError("runtime spec changed for an existing persistent thread")
            session_id = _string(value.get("session_id"), field="manifest.session_id")
            uuid.UUID(session_id)
            return session_id
        session_id = str(uuid.uuid4())
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "thread_id": self.thread.thread_id,
            "session_id": session_id,
            "thread_spec_sha256": self.thread_hash,
            "runtime_spec_sha256": self.runtime_hash,
            "sandbox_root": str(self.runtime.sandbox_root),
            "workspace": self.runtime.workspace,
        }
        manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
        os.chmod(manifest_path, 0o600)
        return session_id

    def emit(self, value: Mapping[str, Any]) -> None:
        record = _canonical_json(value) + b"\n"
        with self.output_lock:
            sys.stdout.buffer.write(record)
            sys.stdout.buffer.flush()

    def start(self) -> None:
        command = build_pi_rpc_command(self.thread, self.runtime, session_id=self.session_id)
        proxy = self.runtime.proxy_url
        child_env = {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "ALL_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "all_proxy": proxy,
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
            "HOME": self.runtime.home,
            "PI_CODING_AGENT_DIR": self.runtime.agent_dir,
            "PI_CODING_AGENT_SESSION_DIR": self.runtime.session_dir,
            "PI_TELEMETRY": "0",
            "PATH": "/usr/sbin:/usr/bin:/bin",
            "TERM": "dumb",
        }
        self.stderr_handle = (self.thread_dir / "stderr.log").open("ab")
        self.event_handle = (self.thread_dir / "rpc-events.jsonl").open("ab")
        self.command_handle = (self.thread_dir / "commands.jsonl").open("ab")
        for name in ("stderr.log", "rpc-events.jsonl", "commands.jsonl"):
            os.chmod(self.thread_dir / name, 0o600)
        self.process = subprocess.Popen(
            command,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            start_new_session=True,
        )
        assert self.process.stdout is not None
        self.reader_thread = threading.Thread(target=self._read_pi, name=f"pi-rpc-{self.thread.thread_id}", daemon=True)
        self.reader_thread.start()
        self._send_pi({"id": "__startup_state__", "type": "get_state"})
        with self.ready_condition:
            if not self.ready_condition.wait_for(lambda: self.startup_response is not None or self.eof, timeout=60):
                raise PiThreadError("Pi RPC did not answer startup get_state within 60 seconds")
            if self.startup_response is None or self.startup_response.get("success") is not True:
                raise PiThreadError("Pi RPC startup get_state failed")
        self._send_pi(
            {"id": "__startup_compaction__", "type": "set_auto_compaction", "enabled": self.thread.auto_compaction}
        )
        self._send_pi({"id": "__startup_retry__", "type": "set_auto_retry", "enabled": self.thread.auto_retry})
        state = self.startup_response.get("data", {})
        self.emit(
            {
                "type": "thread_ready",
                "protocol_version": PROTOCOL_VERSION,
                "thread_id": self.thread.thread_id,
                "session_id": self.session_id,
                "resumed": bool(isinstance(state, Mapping) and state.get("messageCount", 0)),
                "state": state,
            }
        )

    def _read_pi(self) -> None:
        assert self.process is not None and self.process.stdout is not None and self.event_handle is not None
        for raw in iter_lf_records(self.process.stdout):
            self.event_handle.write(raw + b"\n")
            self.event_handle.flush()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                self.emit(
                    {
                        "type": "thread_protocol_error",
                        "thread_id": self.thread.thread_id,
                        "error": f"invalid Pi JSON: {exc.msg}",
                    }
                )
                continue
            if not isinstance(payload, Mapping):
                continue
            payload = dict(payload)
            payload_type = payload.get("type")
            with self.ready_condition:
                self.last_event_type = str(payload_type) if payload_type is not None else None
                if payload_type == "agent_start":
                    self.is_streaming = True
                elif payload_type == "agent_settled":
                    self.is_streaming = False
                    self.settled_count += 1
                if payload_type == "response" and payload.get("id") == "__startup_state__":
                    self.startup_response = payload
                self.ready_condition.notify_all()
            if payload_type == "response":
                self.emit(
                    {
                        "type": "thread_response",
                        "thread_id": self.thread.thread_id,
                        "request_id": payload.get("id"),
                        "command": payload.get("command"),
                        "success": payload.get("success"),
                        "data": payload.get("data"),
                        "error": payload.get("error"),
                        "payload": payload,
                    }
                )
            elif self.thread.event_stream == "all" or payload_type in LIFECYCLE_EVENTS:
                self.emit(
                    {
                        "type": "thread_event",
                        "thread_id": self.thread.thread_id,
                        "event_type": payload_type,
                        "payload": payload,
                    }
                )
        with self.ready_condition:
            self.eof = True
            self.ready_condition.notify_all()
        self.emit(
            {
                "type": "thread_exited",
                "thread_id": self.thread.thread_id,
                "exit_code": self.process.poll() if self.process else None,
            }
        )

    def _send_pi(self, payload: Mapping[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            raise PiThreadError("Pi RPC process is not running")
        record = _canonical_json(payload) + b"\n"
        with self.stdin_lock:
            self.process.stdin.write(record)
            self.process.stdin.flush()

    def handle(self, command: Mapping[str, Any]) -> bool:
        assert self.command_handle is not None
        self.command_handle.write(_canonical_json(command) + b"\n")
        self.command_handle.flush()
        command_type = _string(command.get("type"), field="command.type")
        request_id = _request_id(command)
        if command_type == "status":
            with self.state_lock:
                status = {
                    "type": "thread_status",
                    "thread_id": self.thread.thread_id,
                    "request_id": request_id,
                    "session_id": self.session_id,
                    "alive": self.process is not None and self.process.poll() is None,
                    "is_streaming": self.is_streaming,
                    "settled_count": self.settled_count,
                    "last_event_type": self.last_event_type,
                }
            self.emit(status)
            return True
        if command_type == "close":
            force = command.get("force", False)
            if not isinstance(force, bool):
                raise PiThreadError("command.force must be boolean")
            with self.state_lock:
                streaming = self.is_streaming
            if streaming and not force:
                self.emit(
                    {
                        "type": "thread_response",
                        "thread_id": self.thread.thread_id,
                        "request_id": request_id,
                        "command": "close",
                        "success": False,
                        "error": "thread is streaming; abort first or close with force=true",
                    }
                )
                return True
            self.emit(
                {
                    "type": "thread_response",
                    "thread_id": self.thread.thread_id,
                    "request_id": request_id,
                    "command": "close",
                    "success": True,
                }
            )
            return False
        translated = translate_command(command)
        assert translated is not None
        self._send_pi(translated)
        return True

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=30)
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=5)
        for handle in (self.stderr_handle, self.event_handle, self.command_handle, self._lock_handle):
            if handle is not None:
                handle.close()


def serve(*, thread_spec_path: Path, runtime_spec_path: Path, thread_dir: Path) -> int:
    thread, thread_raw = load_thread_spec(thread_spec_path)
    runtime, runtime_raw = load_runtime(runtime_spec_path)
    supervisor = PiThreadSupervisor(
        thread,
        runtime,
        thread_dir=thread_dir,
        thread_hash=_hash_object(thread_raw),
        runtime_hash=_hash_object(runtime_raw),
    )
    try:
        supervisor.start()
        for raw in iter_lf_records(sys.stdin.buffer):
            try:
                command = json.loads(raw)
                if not isinstance(command, Mapping):
                    raise PiThreadError("command must be a JSON object")
                if not supervisor.handle(command):
                    break
            except (json.JSONDecodeError, PiThreadError) as exc:
                supervisor.emit(
                    {
                        "type": "thread_protocol_error",
                        "thread_id": thread.thread_id,
                        "error": str(exc),
                    }
                )
        return 0
    finally:
        supervisor.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-spec", type=Path, required=True)
    parser.add_argument("--runtime-spec", type=Path, required=True)
    parser.add_argument("--thread-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return serve(thread_spec_path=args.thread_spec, runtime_spec_path=args.runtime_spec, thread_dir=args.thread_dir)
    except (OSError, PiThreadError, json.JSONDecodeError, ValueError) as exc:
        sys.stdout.buffer.write(
            _canonical_json(
                {
                    "type": "thread_start_error",
                    "protocol_version": PROTOCOL_VERSION,
                    "error": str(exc),
                }
            )
            + b"\n"
        )
        sys.stdout.buffer.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
