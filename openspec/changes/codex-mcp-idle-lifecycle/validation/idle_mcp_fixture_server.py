#!/usr/bin/env python3
"""Small MCP stdio server used by the idle-lifecycle real-entry smoke.

The process is deliberately boring: every lifecycle and tool invocation is
written to a JSONL receipt, while protocol traffic stays on stdout.  The
receipt lets the parent prove that a cold demand used one replacement process
and that an in-flight call was not terminated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


PROTOCOL_VERSION = "2025-06-18"


def _record(path: Path, instance: str, kind: str, **fields: object) -> None:
    try:
        import psutil

        create_time: float | None = psutil.Process(os.getpid()).create_time()
    except (ImportError, OSError):
        create_time = None
    event = {
        "kind": kind,
        "instance": instance,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "create_time": create_time,
        "wall_time": time.time(),
        "monotonic": time.monotonic(),
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each event opens the file independently.  This is enough for the one
    # process per server invariant and avoids keeping a stale descriptor over
    # a Core replacement.  fcntl is optional so the fixture remains portable.
    with path.open("a", encoding="utf-8") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass


def _response(request_id: object, result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: object, code: int, message: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _tools() -> list[dict[str, object]]:
    return [
        {
            "name": "echo",
            "description": "Return a supplied value without side effects.",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
        {
            "name": "identity",
            "description": "Return the current fixture process identity.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "sleep",
            "description": "Wait briefly while retaining an active MCP call.",
            "inputSchema": {
                "type": "object",
                "properties": {"seconds": {"type": "number"}},
                "required": ["seconds"],
                "additionalProperties": False,
            },
        },
    ]


def _tool_result(text: str, *, is_error: bool = False) -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def serve(receipt: Path, instance: str, startup_delay: float = 0.0) -> int:
    call_index = 0
    _record(receipt, instance, "started", command=sys.argv)
    if startup_delay:
        time.sleep(startup_delay)
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    json.dumps(_error(None, -32700, f"invalid JSON: {exc}")), flush=True
                )
                continue

            method = request.get("method")
            request_id = request.get("id")
            params = request.get("params") or {}
            if request_id is None:
                # MCP notifications have no response.  Keep a receipt for
                # diagnostics, but do not contaminate stdout with a reply.
                _record(receipt, instance, "notification", method=method)
                continue

            if method == "initialize":
                _record(receipt, instance, "initialize", request=request)
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "codex-idle-fixture", "version": "1"},
                }
                print(json.dumps(_response(request_id, result)), flush=True)
                continue

            if method == "tools/list":
                _record(receipt, instance, "tools_list")
                print(
                    json.dumps(_response(request_id, {"tools": _tools()})), flush=True
                )
                continue

            if method == "resources/list":
                _record(receipt, instance, "resources_list")
                print(json.dumps(_response(request_id, {"resources": []})), flush=True)
                continue

            if method == "resources/templates/list":
                _record(receipt, instance, "resource_templates_list")
                print(
                    json.dumps(_response(request_id, {"resourceTemplates": []})),
                    flush=True,
                )
                continue

            if method != "tools/call":
                _record(receipt, instance, "unknown_method", method=method)
                print(
                    json.dumps(_error(request_id, -32601, f"unknown method: {method}")),
                    flush=True,
                )
                continue

            name = params.get("name")
            arguments = params.get("arguments") or {}
            call_index += 1
            _record(
                receipt,
                instance,
                "call_started",
                call_index=call_index,
                name=name,
                arguments=arguments,
            )
            if name == "echo":
                text = str(arguments.get("value", ""))
            elif name == "identity":
                text = f"instance={instance};pid={os.getpid()};call={call_index}"
            elif name == "sleep":
                try:
                    seconds = float(arguments.get("seconds", 0))
                except (TypeError, ValueError):
                    seconds = -1
                if seconds < 0 or seconds > 5:
                    text = "seconds must be between 0 and 5"
                    result = _tool_result(text, is_error=True)
                    _record(
                        receipt,
                        instance,
                        "call_finished",
                        call_index=call_index,
                        name=name,
                        is_error=True,
                    )
                    print(json.dumps(_response(request_id, result)), flush=True)
                    continue
                time.sleep(seconds)
                text = f"slept={seconds:.3f}"
            else:
                text = f"unknown tool: {name}"
                result = _tool_result(text, is_error=True)
                _record(
                    receipt,
                    instance,
                    "call_finished",
                    call_index=call_index,
                    name=name,
                    is_error=True,
                )
                print(json.dumps(_response(request_id, result)), flush=True)
                continue

            result = _tool_result(text)
            _record(
                receipt,
                instance,
                "call_finished",
                call_index=call_index,
                name=name,
                is_error=False,
            )
            print(json.dumps(_response(request_id, result)), flush=True)
    except BrokenPipeError:
        _record(receipt, instance, "broken_pipe")
    except BaseException as exc:
        _record(receipt, instance, "fatal", error=repr(exc))
        print(f"fixture server failed: {exc!r}", file=sys.stderr, flush=True)
        return 1
    finally:
        _record(receipt, instance, "exited")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    args = parser.parse_args()
    if args.startup_delay < 0 or args.startup_delay > 2:
        parser.error("startup delay must be between 0 and 2 seconds")
    return serve(args.receipt, args.instance, args.startup_delay)


if __name__ == "__main__":
    raise SystemExit(main())
