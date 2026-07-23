#!/usr/bin/env python3
"""Run one isolated Pi task behind a Codex-independent receipt contract.

This Pi-home-local foreman deliberately imports no Codex implementation.
Any caller that can start a process and read JSON can use it, including a
native Codex subagent, the root agent, a future MCP tool, or a batch scheduler.

The sandbox is prepared by the caller.  The foreman owns process lifecycle,
Pi JSONL validation, timeout handling, optional hidden verification, and the
final receipt.  It never treats a zero-token transport error as task success.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


TASK_SCHEMA_VERSION = "pi_foreman_task.v1"
RUNTIME_SCHEMA_VERSION = "pi_foreman_runtime.v1"
RESULT_SCHEMA_VERSION = "pi_foreman_result.v1"
ALLOWED_REASONING = frozenset({"medium", "high", "xhigh"})
ALLOWED_RESPONSE_FORMATS = frozenset({"json_object", "text"})
_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")


class PiForemanError(ValueError):
    """Raised when a task/runtime contract is unsafe or malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, expected_schema: str) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PiForemanError(f"JSON document must be an object: {source}")
    result = dict(value)
    if result.get("schema_version") != expected_schema:
        raise PiForemanError(
            f"{source} schema_version must be {expected_schema!r}, got {result.get('schema_version')!r}"
        )
    return result


def _required_string(value: Any, *, field: str, identity: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PiForemanError(f"{field} must be a non-empty string")
    result = value.strip()
    if identity and not _IDENTITY_PATTERN.fullmatch(result):
        raise PiForemanError(f"{field} contains unsupported characters: {result!r}")
    return result


def _positive_int(value: Any, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise PiForemanError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _inside_path(value: Any, *, field: str) -> str:
    raw = _required_string(value, field=field)
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise PiForemanError(f"{field} must be an absolute normalized sandbox path")
    return str(path)


def _relative_path(value: Any, *, field: str) -> str:
    raw = _required_string(value, field=field)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise PiForemanError(f"{field} must be a normalized relative path")
    return str(path)


def _local_proxy(value: Any) -> str:
    proxy = _required_string(value, field="runtime.proxy_url")
    parsed = urlparse(proxy)
    if parsed.scheme not in {"http", "https", "socks5", "socks5h"}:
        raise PiForemanError("runtime.proxy_url has an unsupported scheme")
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port is None:
        raise PiForemanError("runtime.proxy_url must name an explicit local proxy port")
    if parsed.username is not None or parsed.password is not None:
        raise PiForemanError("runtime.proxy_url must not embed credentials")
    return proxy


@dataclass(frozen=True)
class PiTaskSpec:
    task_id: str
    task_file: str
    prompt: str
    provider: str
    model: str
    reasoning: str
    timeout_seconds: int
    response_format: str
    verifier_argv: tuple[str, ...] | None
    verifier_timeout_seconds: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PiTaskSpec":
        reasoning = _required_string(value.get("reasoning"), field="task.reasoning")
        if reasoning not in ALLOWED_REASONING:
            raise PiForemanError(
                f"task.reasoning must be one of {sorted(ALLOWED_REASONING)}; max is intentionally unsupported"
            )
        response_format = _required_string(value.get("response_format"), field="task.response_format")
        if response_format not in ALLOWED_RESPONSE_FORMATS:
            raise PiForemanError(f"task.response_format must be one of {sorted(ALLOWED_RESPONSE_FORMATS)}")
        verifier = value.get("verifier_argv")
        verifier_argv: tuple[str, ...] | None
        if verifier is None:
            verifier_argv = None
        elif isinstance(verifier, list) and verifier and all(isinstance(item, str) and item for item in verifier):
            verifier_argv = tuple(verifier)
        else:
            raise PiForemanError("task.verifier_argv must be a non-empty string array or null")
        return cls(
            task_id=_required_string(value.get("task_id"), field="task.task_id", identity=True),
            task_file=_relative_path(value.get("task_file"), field="task.task_file"),
            prompt=_required_string(value.get("prompt"), field="task.prompt"),
            provider=_required_string(value.get("provider"), field="task.provider", identity=True),
            model=_required_string(value.get("model"), field="task.model", identity=True),
            reasoning=reasoning,
            timeout_seconds=_positive_int(value.get("timeout_seconds"), field="task.timeout_seconds", maximum=7200),
            response_format=response_format,
            verifier_argv=verifier_argv,
            verifier_timeout_seconds=_positive_int(
                value.get("verifier_timeout_seconds", 120),
                field="task.verifier_timeout_seconds",
                maximum=1800,
            ),
        )


@dataclass(frozen=True)
class PiRuntimeSpec:
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
    def from_mapping(cls, value: Mapping[str, Any]) -> "PiRuntimeSpec":
        sandbox_root = Path(_required_string(value.get("sandbox_root"), field="runtime.sandbox_root"))
        if not sandbox_root.is_absolute():
            raise PiForemanError("runtime.sandbox_root must be absolute")
        sandbox_root = sandbox_root.resolve(strict=True)
        userspec = _required_string(value.get("userspec", "65534:65534"), field="runtime.userspec")
        if not re.fullmatch(r"[0-9]+:[0-9]+", userspec):
            raise PiForemanError("runtime.userspec must be numeric uid:gid")
        return cls(
            sandbox_root=sandbox_root,
            workspace=_inside_path(value.get("workspace", "/workspace"), field="runtime.workspace"),
            node_path=_inside_path(value.get("node_path"), field="runtime.node_path"),
            pi_cli_path=_inside_path(value.get("pi_cli_path"), field="runtime.pi_cli_path"),
            home=_inside_path(value.get("home"), field="runtime.home"),
            agent_dir=_inside_path(value.get("agent_dir"), field="runtime.agent_dir"),
            session_dir=_inside_path(value.get("session_dir"), field="runtime.session_dir"),
            proxy_url=_local_proxy(value.get("proxy_url")),
            userspec=userspec,
        )

    def host_path(self, sandbox_path: str) -> Path:
        return self.sandbox_root.joinpath(sandbox_path.lstrip("/"))


def load_task_spec(path: str | Path) -> PiTaskSpec:
    return PiTaskSpec.from_mapping(_read_object(Path(path), expected_schema=TASK_SCHEMA_VERSION))


def load_runtime_spec(path: str | Path) -> PiRuntimeSpec:
    return PiRuntimeSpec.from_mapping(_read_object(Path(path), expected_schema=RUNTIME_SCHEMA_VERSION))


def build_pi_command(task: PiTaskSpec, runtime: PiRuntimeSpec) -> list[str]:
    task_path = runtime.host_path(f"{runtime.workspace}/{task.task_file}")
    for field, sandbox_path in (
        ("runtime.node_path", runtime.node_path),
        ("runtime.pi_cli_path", runtime.pi_cli_path),
    ):
        if not runtime.host_path(sandbox_path).is_file():
            raise PiForemanError(f"{field} does not resolve to a file inside the sandbox")
    if not task_path.is_file():
        raise PiForemanError(f"task.task_file does not exist inside the sandbox workspace: {task_path}")
    return [
        "/usr/sbin/chroot",
        f"--userspec={runtime.userspec}",
        str(runtime.sandbox_root),
        "/usr/bin/env",
        "-C",
        runtime.workspace,
        f"HOME={runtime.home}",
        f"PI_CODING_AGENT_DIR={runtime.agent_dir}",
        f"PI_CODING_AGENT_SESSION_DIR={runtime.session_dir}",
        "PI_TELEMETRY=0",
        "PATH=/opt/node/bin:/usr/bin:/bin",
        "TERM=dumb",
        runtime.node_path,
        runtime.pi_cli_path,
        "--mode",
        "json",
        "--print",
        "--no-session",
        "--provider",
        task.provider,
        "--model",
        task.model,
        "--thinking",
        task.reasoning,
        f"@{task.task_file}",
        task.prompt,
    ]


@dataclass(frozen=True)
class ParsedEvents:
    terminal_message: dict[str, Any] | None
    final_text: str
    usage: dict[str, Any]
    tool_call_count: int
    parse_errors: tuple[str, ...]


def parse_pi_events(path: str | Path) -> ParsedEvents:
    source = Path(path)
    terminal: dict[str, Any] | None = None
    tool_call_count = 0
    parse_errors: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_errors.append(f"line {line_number}: {exc.msg}")
                continue
            if not isinstance(event, Mapping):
                parse_errors.append(f"line {line_number}: event is not an object")
                continue
            if event.get("type") == "tool_execution_start":
                tool_call_count += 1
            message = event.get("message")
            if (
                event.get("type") == "message_end"
                and isinstance(message, Mapping)
                and message.get("role") == "assistant"
            ):
                terminal = dict(message)
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    if terminal is not None:
        content = terminal.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
        if isinstance(terminal.get("usage"), Mapping):
            usage = dict(terminal["usage"])
    return ParsedEvents(
        terminal_message=terminal,
        final_text="\n".join(text_parts),
        usage=usage,
        tool_call_count=tool_call_count,
        parse_errors=tuple(parse_errors),
    )


def classify_execution(*, process_exit_code: int, timed_out: bool, parsed: ParsedEvents) -> tuple[str, list[str]]:
    failures = list(parsed.parse_errors)
    if timed_out:
        failures.append("worker_timeout")
        return "timed_out", failures
    terminal = parsed.terminal_message
    if terminal is None:
        failures.append("missing_terminal_assistant_event")
        return "protocol_failed", failures
    total_tokens = parsed.usage.get("totalTokens", 0)
    if terminal.get("stopReason") == "error" or not isinstance(total_tokens, int) or total_tokens <= 0:
        failures.append(f"terminal_stop_reason:{terminal.get('stopReason')}")
        failures.append(f"terminal_total_tokens:{total_tokens}")
        return "infrastructure_failed", failures
    if process_exit_code != 0:
        failures.append(f"process_exit_code:{process_exit_code}")
        return "process_failed", failures
    if not parsed.final_text.strip():
        failures.append("empty_final_text")
        return "protocol_failed", failures
    return "completed", failures


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_verifier(task: PiTaskSpec, *, answer_path: Path, result_dir: Path) -> dict[str, Any] | None:
    if task.verifier_argv is None:
        return None
    argv = [item.replace("{answer_path}", str(answer_path)) for item in task.verifier_argv]
    completed = subprocess.run(
        argv,
        cwd=result_dir,
        capture_output=True,
        text=True,
        timeout=task.verifier_timeout_seconds,
        check=False,
    )
    return {
        "argv": argv,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_foreman(task: PiTaskSpec, runtime: PiRuntimeSpec, result_dir: str | Path) -> dict[str, Any]:
    destination = Path(result_dir).expanduser()
    if destination.exists():
        raise PiForemanError(f"result directory already exists: {destination}")
    destination.mkdir(parents=True)
    events_path = destination / "events.jsonl"
    stderr_path = destination / "stderr.log"
    answer_path = destination / "answer.txt"
    result_path = destination / "result.json"
    command = build_pi_command(task, runtime)
    child_env = {
        "HTTP_PROXY": runtime.proxy_url,
        "HTTPS_PROXY": runtime.proxy_url,
        "ALL_PROXY": runtime.proxy_url,
        "http_proxy": runtime.proxy_url,
        "https_proxy": runtime.proxy_url,
        "all_proxy": runtime.proxy_url,
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
        "PATH": "/usr/sbin:/usr/bin:/bin",
    }
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    timed_out = False
    with events_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            env=child_env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            process_exit_code = process.wait(timeout=task.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process_exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process_exit_code = process.wait(timeout=30)
    ended_at = _utc_now()
    wall_seconds = time.monotonic() - started_monotonic
    parsed = parse_pi_events(events_path)
    status, failures = classify_execution(
        process_exit_code=process_exit_code,
        timed_out=timed_out,
        parsed=parsed,
    )
    answer_path.write_text(parsed.final_text + ("\n" if parsed.final_text else ""), encoding="utf-8")

    verification: dict[str, Any] | None = None
    if status == "completed":
        if task.response_format == "json_object":
            try:
                decoded = json.loads(parsed.final_text)
                if not isinstance(decoded, Mapping):
                    raise PiForemanError("final JSON is not an object")
            except (json.JSONDecodeError, PiForemanError) as exc:
                failures.append(f"response_format:{exc}")
                status = "protocol_failed"
        if status == "completed":
            try:
                verification = _run_verifier(task, answer_path=answer_path, result_dir=destination)
            except subprocess.TimeoutExpired:
                verification = {"passed": False, "timed_out": True}
            if verification is not None:
                status = "passed" if verification.get("passed") is True else "failed_verification"

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task_id": task.task_id,
        "status": status,
        "worker": {
            "harness": "pi",
            "provider": task.provider,
            "model": task.model,
            "reasoning": task.reasoning,
        },
        "execution": {
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_seconds": wall_seconds,
            "timeout_seconds": task.timeout_seconds,
            "process_exit_code": process_exit_code,
            "timed_out": timed_out,
            "proxy_route": f"{urlparse(runtime.proxy_url).hostname}:{urlparse(runtime.proxy_url).port}",
            "command_argv": command,
        },
        "response": {
            "stop_reason": parsed.terminal_message.get("stopReason") if parsed.terminal_message else None,
            "usage": parsed.usage,
            "tool_call_count": parsed.tool_call_count,
            "answer_sha256": _sha256_file(answer_path),
        },
        "verification": verification,
        "failures": failures,
        "artifacts": {
            "events": {"path": str(events_path), "sha256": _sha256_file(events_path)},
            "stderr": {"path": str(stderr_path), "sha256": _sha256_file(stderr_path)},
            "answer": {"path": str(answer_path), "sha256": _sha256_file(answer_path)},
        },
    }
    _write_json(result_path, result)
    return result


def inspect_events(task: PiTaskSpec, events_path: str | Path) -> dict[str, Any]:
    parsed = parse_pi_events(events_path)
    status, failures = classify_execution(process_exit_code=0, timed_out=False, parsed=parsed)
    if status == "completed" and task.response_format == "json_object":
        try:
            value = json.loads(parsed.final_text)
            if not isinstance(value, Mapping):
                raise PiForemanError("final JSON is not an object")
        except (json.JSONDecodeError, PiForemanError) as exc:
            failures.append(f"response_format:{exc}")
            status = "protocol_failed"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task_id": task.task_id,
        "status": status,
        "response": {
            "stop_reason": parsed.terminal_message.get("stopReason") if parsed.terminal_message else None,
            "usage": parsed.usage,
            "tool_call_count": parsed.tool_call_count,
        },
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one Pi task and write a receipt directory")
    run_parser.add_argument("--task-spec", type=Path, required=True)
    run_parser.add_argument("--runtime-spec", type=Path, required=True)
    run_parser.add_argument("--result-dir", type=Path, required=True)
    inspect_parser = subparsers.add_parser("inspect", help="classify an existing Pi JSONL event stream")
    inspect_parser.add_argument("--task-spec", type=Path, required=True)
    inspect_parser.add_argument("--events", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        task = load_task_spec(args.task_spec)
        if args.command == "inspect":
            print(_canonical_json(inspect_events(task, args.events)))
            return 0
        runtime = load_runtime_spec(args.runtime_spec)
        result = run_foreman(task, runtime, args.result_dir)
        print(_canonical_json(result))
        return 0 if result["status"] in {"completed", "passed"} else 1
    except (OSError, PiForemanError, json.JSONDecodeError) as exc:
        print(_canonical_json({"schema_version": RESULT_SCHEMA_VERSION, "status": "foreman_error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
