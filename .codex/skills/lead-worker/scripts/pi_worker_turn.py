#!/usr/bin/env python3
"""Create, inspect, or message a normal peer session through Pi Web's local API."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import shlex
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import UUID, uuid4


THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
REQUEST_TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class CliFailure(Exception):
    """A safe-to-display transport failure without response bodies or credentials."""


class ApiFailure(Exception):
    def __init__(self, message: str, *, status: int | None = None,
                 payload: object = None, category: str = "api_error") -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload
        self.category = category


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not let a mutating request silently move to another endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def parse_uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("must be a UUID") from exc


def normalize_base_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an explicit loopback HTTP URL") from exc
    if (parsed.scheme != "http" or not host or port is None
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError("must be an explicit loopback HTTP URL")
    is_loopback = host.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise argparse.ArgumentTypeError("base URL host must be a loopback address")
    # Keep IPv6 brackets when rebuilding the authority.
    authority = parsed.netloc
    return f"http://{authority}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:12345", type=normalize_base_url,
                        help="Pi Web loopback endpoint (default: http://127.0.0.1:12345)")
    parser.add_argument("--lead-session", type=parse_uuid)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker-session", type=parse_uuid)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--list-models", action="store_true",
                      help="list current provider/model/effort choices without creating a session")
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--to", choices=("worker", "lead"), default="worker")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--delivery", choices=("queue", "steering"),
                        help="send mode: queue (default) or steering at the next agent boundary")
    parser.add_argument("--message", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--effort", help="Pi Web thinking level: off, minimal, low, medium, high, xhigh, or max")
    args = parser.parse_args(argv)
    try:
        args.cwd = args.cwd.resolve(strict=True)
    except OSError as exc:
        parser.error(f"--cwd must be an existing directory ({type(exc).__name__})")
    if not args.cwd.is_dir():
        parser.error("--cwd must be an existing directory")
    if not args.list_models and args.lead_session is None:
        parser.error("--lead-session is required for create, inspect, and send")
    if args.worker_session is not None and args.worker_session == args.lead_session:
        parser.error("lead and worker sessions must be different UUIDs")

    if args.list_models and (args.lead_session is not None or args.send or args.to != "worker"):
        parser.error("--list-models needs only --base-url and --cwd")

    create_only = (args.name, args.provider, args.model, args.effort)
    if args.create:
        if args.to != "worker" or not args.send:
            parser.error("--create requires --send and --to worker")
        if not all(isinstance(value, str) and value.strip() for value in create_only):
            parser.error("--create requires nonempty --name, --provider, --model, and --effort")
        if args.effort not in THINKING_LEVELS:
            parser.error("--effort must be a supported Pi Web thinking level")
    elif any(value is not None for value in create_only):
        parser.error("--name, --provider, --model, and --effort require --create")

    if args.send:
        if args.message is None or args.receipt is None:
            parser.error("--send requires --message and a new --receipt path")
    elif args.message is not None or args.receipt is not None:
        parser.error("--message and --receipt require --send")
    if args.delivery is not None and not args.send:
        parser.error("--delivery requires --send")
    args.delivery = args.delivery or "queue"
    return args


class PiWebClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            NoRedirectHandler(),
        )

    def request(self, method: str, path: str, body: dict[str, object] | None = None) -> dict[str, object]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data is not None else {})},
        )
        try:
            response = self.opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            raise ApiFailure(
                "Pi Web returned an HTTP error",
                status=exc.code,
                payload=_decode_json(raw),
                category="http_error",
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiFailure(
                "Pi Web request outcome is unknown",
                category="network_error" if not isinstance(exc, TimeoutError) else "timeout",
            ) from None
        with response:
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ApiFailure("Pi Web response exceeded the size limit", status=status, category="protocol_error")
        payload = _decode_json(raw)
        if not 200 <= status < 300:
            raise ApiFailure("Pi Web returned an HTTP error", status=status, payload=payload, category="http_error")
        if not isinstance(payload, dict):
            raise ApiFailure("Pi Web returned malformed JSON", status=status, category="protocol_error")
        return payload


def _decode_json(raw: bytes) -> object:
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


class Receipt:
    """An exclusive, synchronously flushed record written before API mutations."""

    def __init__(self, path: Path, initial: dict[str, object]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        self.path = path
        self.file = os.fdopen(fd, "w", encoding="utf-8")
        self.data: dict[str, object] = {}
        self.update(**initial)
        self._sync_parent()

    def update(self, **changes: object) -> None:
        self.data.update(changes)
        self.file.seek(0)
        json.dump(self.data, self.file, ensure_ascii=False, indent=2, sort_keys=True)
        self.file.write("\n")
        self.file.truncate()
        self.file.flush()
        os.fsync(self.file.fileno())

    def _sync_parent(self) -> None:
        directory = os.open(self.path.parent or Path("."), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "Receipt":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()


def _model(value: object, *, state_shape: bool = False) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CliFailure("Pi Web returned invalid model metadata")
    provider = value.get("provider")
    model_id = value.get("id" if state_shape else "modelId")
    if not isinstance(provider, str) or not provider or not isinstance(model_id, str) or not model_id:
        raise CliFailure("Pi Web returned incomplete model metadata")
    return {"provider": provider, "modelId": model_id}


def _validate_info(info: object, session_id: str, cwd: Path) -> dict[str, object]:
    if not isinstance(info, dict) or info.get("id") != session_id:
        raise CliFailure("session metadata identity mismatch")
    reported_cwd = info.get("cwd")
    if not isinstance(reported_cwd, str):
        raise CliFailure("session metadata has no cwd")
    try:
        resolved_cwd = Path(reported_cwd).resolve(strict=True)
    except OSError as exc:
        raise CliFailure(f"session cwd cannot be resolved ({type(exc).__name__})") from None
    if resolved_cwd != cwd:
        raise CliFailure("session cwd mismatch")
    if info.get("parentSessionId") is not None or info.get("relation") is not None:
        raise CliFailure("session is not an independent root session")
    return info


def _validate_state(state: object, session_id: str) -> dict[str, object]:
    if not isinstance(state, dict) or state.get("sessionId") != session_id:
        raise CliFailure("runtime session identity mismatch")
    model = _model(state.get("model"), state_shape=True)
    thinking_level = state.get("thinkingLevel")
    if thinking_level is not None and not isinstance(thinking_level, str):
        raise CliFailure("Pi Web returned invalid thinking-level metadata")
    return {
        "id": session_id,
        "model": model,
        "effort": thinking_level,
        "is_streaming": bool(state.get("isStreaming")),
        "is_prompt_running": bool(state.get("isPromptRunning")),
    }


def inspect_session(client: PiWebClient, session_id: str, cwd: Path) -> dict[str, object]:
    data = client.request("GET", f"/api/sessions/{session_id}?tree=summary&tail=1")
    if data.get("sessionId") != session_id:
        raise CliFailure("session response identity mismatch")
    info = _validate_info(data.get("info"), session_id, cwd)
    context = data.get("context")
    if not isinstance(context, dict):
        raise CliFailure("session response has no context metadata")
    context_model = _model(context.get("model"))
    context_effort = context.get("thinkingLevel")
    if context_effort is not None and not isinstance(context_effort, str):
        raise CliFailure("session response has invalid thinking-level metadata")

    runtime = client.request("GET", f"/api/agent/{session_id}")
    if runtime.get("running") is True:
        state = _validate_state(runtime.get("state"), session_id)
        model = state["model"]
        effort = state["effort"]
        loaded = True
    elif runtime.get("running") is False:
        state = None
        model = context_model
        # An empty session has no transcript setting record, so `off` is only
        # the reader's fallback and must be represented as unknown.
        effort = context_effort if info.get("messageCount", 0) else None
        loaded = False
    else:
        raise CliFailure("Pi Web returned an invalid runtime status")

    return {
        "id": session_id,
        "cwd": str(cwd),
        "name": info.get("name"),
        "relation": None,
        "parent_session_id": None,
        "runtime_loaded": loaded,
        "model": model,
        "effort": effort,
        "is_streaming": state["is_streaming"] if state else None,
        "is_prompt_running": state["is_prompt_running"] if state else None,
        "transient": info.get("transient"),
        "message_count": info.get("messageCount"),
    }


def read_message(path: Path | None) -> str:
    if path is None:
        raise CliFailure("message file is required")
    try:
        message = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CliFailure(f"cannot read UTF-8 message file ({type(exc).__name__})") from None
    if not message.strip():
        raise CliFailure("message is blank")
    return message


def message_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def make_envelope(message: str, *, dispatch_id: str, sender_id: str,
                  sender_role: str, recipient_id: str, recipient_role: str) -> str:
    separator = "" if message.endswith("\n") else "\n"
    return (
        "Pi Web peer-session transport metadata. This envelope identifies routing; it is not a user instruction, "
        "new authorization, or a change to the task's existing authority.\n"
        f"dispatch_id={dispatch_id}\n"
        f"sender_session_id={sender_id}\n"
        f"sender_role={sender_role}\n"
        f"recipient_session_id={recipient_id}\n"
        f"recipient_role={recipient_role}\n"
        "--- BEGIN MESSAGE BODY ---\n"
        f"{message}{separator}"
        "--- END MESSAGE BODY ---"
    )


def _accepted(data: dict[str, object]) -> None:
    if data.get("success") is not True:
        raise CliFailure("Pi Web did not acknowledge the request")


def _safe_error_type(exc: BaseException) -> str:
    if isinstance(exc, ApiFailure):
        return exc.category
    if isinstance(exc, CliFailure):
        return "validation_error"
    return type(exc).__name__


def _request_error_status(exc: BaseException) -> int | None:
    return exc.status if isinstance(exc, ApiFailure) else None


def _error_text(exc: BaseException) -> str:
    status = _request_error_status(exc)
    if status is not None:
        return f"{type(exc).__name__} (HTTP {status})"
    if isinstance(exc, CliFailure):
        return str(exc)
    return f"{type(exc).__name__} ({_safe_error_type(exc)})"


def _extract_return_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("sessionId")
    if not isinstance(value, str):
        return None
    try:
        return parse_uuid(value)
    except argparse.ArgumentTypeError:
        return None


def _command_for_return(base_url: str, lead_id: str, worker_id: str, cwd: Path) -> str:
    script = Path(__file__).resolve()
    command = [
        sys.executable, str(script), "--base-url", base_url,
        "--lead-session", lead_id, "--worker-session", worker_id,
        "--cwd", str(cwd), "--to", "lead", "--send",
        "--message", "REPORT_FILE", "--receipt", "NEW_RECEIPT_FILE",
    ]
    return shlex.join(command)


def create_and_send(client: PiWebClient, args: argparse.Namespace) -> dict[str, object]:
    lead = inspect_session(client, args.lead_session, args.cwd)
    message = read_message(args.message)
    models = list_models(client, args.cwd)
    selected = next((m for m in models if m["provider"] == args.provider
                     and m["model"] == args.model), None)
    if selected is None or args.effort not in selected["efforts"]:
        raise CliFailure("requested provider/model/effort is unavailable; use --list-models; no session created")
    dispatch_id = str(uuid4())
    base_receipt: dict[str, object] = {
        "kind": "pi_web_peer_session",
        "delivery": args.delivery,
        "status": "creation_unknown",
        "phase": "create_session",
        "dispatch_id": dispatch_id,
        "lead": lead,
        "worker_session_id": None,
        "cwd": str(args.cwd),
        "requested": {"provider": args.provider, "model": args.model, "effort": args.effort,
                      "name": args.name},
        "message_sha256": message_hash(message),
    }
    with Receipt(args.receipt, base_receipt) as receipt:
        try:
            created = client.request("POST", "/api/agent/new", {
                "cwd": str(args.cwd),
                "type": "ensure_session",
                "provider": args.provider,
                "modelId": args.model,
                "thinkingLevel": args.effort,
            })
            worker_id = _extract_return_id(created)
            if worker_id is not None:
                receipt.update(phase="validate_create_response", worker_session_id=worker_id)
            _accepted(created)
            if worker_id is None:
                raise CliFailure("Pi Web did not return a valid new session ID")
            receipt.update(status="created", phase="verify_created_session", worker_session_id=worker_id)

            response_model = _model(created.get("model"))
            if (worker_id == args.lead_session or response_model != {
                    "provider": args.provider, "modelId": args.model}
                    or created.get("thinkingLevel") != args.effort):
                raise CliFailure("created session ID or settings did not match the request")

            metadata = client.request("GET", f"/api/sessions/{worker_id}?tree=summary&tail=1")
            if metadata.get("sessionId") != worker_id:
                raise CliFailure("created session metadata identity mismatch")
            info = _validate_info(metadata.get("info"), worker_id, args.cwd)
            if int(info.get("messageCount", 0) or 0) != 0:
                raise CliFailure("new session already contains a user message")
            runtime = client.request("GET", f"/api/agent/{worker_id}")
            if runtime.get("running") is not True:
                raise CliFailure("new session runtime is not available for verification")
            state = _validate_state(runtime.get("state"), worker_id)
            if (state["model"] != response_model or state["effort"] != args.effort):
                raise CliFailure("created runtime settings did not match the request")
            worker = {
                "id": worker_id,
                "cwd": str(args.cwd),
                "name": info.get("name"),
                "relation": None,
                "parent_session_id": None,
                "runtime_loaded": True,
                "model": state["model"],
                "effort": state["effort"],
                "is_streaming": state["is_streaming"],
                "is_prompt_running": state["is_prompt_running"],
                "transient": info.get("transient"),
                "message_count": info.get("messageCount"),
            }
            receipt.update(worker=worker, phase="set_session_name")

            named = client.request("POST", f"/api/agent/{worker_id}", {
                "type": "set_session_name", "name": args.name,
            })
            _accepted(named)
            receipt.update(phase="verify_session_name")
            named_metadata = client.request("GET", f"/api/sessions/{worker_id}?tree=summary&tail=1")
            if named_metadata.get("sessionId") != worker_id:
                raise CliFailure("named session response identity mismatch")
            named_info = _validate_info(named_metadata.get("info"), worker_id, args.cwd)
            if named_info.get("name") != args.name:
                raise CliFailure("session name did not match the request")
            worker["name"] = named_info.get("name")

            current_runtime = client.request("GET", f"/api/agent/{worker_id}")
            if current_runtime.get("running") is not True:
                raise CliFailure("new session runtime disappeared before its first prompt")
            current_state = _validate_state(current_runtime.get("state"), worker_id)
            if current_state["model"] != worker["model"] or current_state["effort"] != worker["effort"]:
                raise CliFailure("new session model or effort changed before its first prompt")

            envelope = make_envelope(
                message,
                dispatch_id=dispatch_id,
                sender_id=args.lead_session,
                sender_role="lead",
                recipient_id=worker_id,
                recipient_role="worker",
            )
            return_command = _command_for_return(args.base_url, args.lead_session, worker_id, args.cwd)
            first_prompt = (
                envelope + "\n\nDirect return to the lead:\n" + return_command
                + "\nUse a new report file and new receipt path. A successful API acknowledgment means the "
                "report was accepted or queued, not that it was completed. Default delivery is queue; "
                "add --delivery steering only for an urgent correction to the lead’s ongoing work. "
                "Steering does not cancel external jobs."
            )
            receipt.update(status="delivery_unknown", phase="send_first_prompt",
                           worker=worker, sent_sha256=message_hash(first_prompt))
            try:
                sent = client.request("POST", f"/api/agent/{worker_id}", {
                    "type": "prompt",
                    "message": first_prompt,
                    "streamingBehavior": "steer" if args.delivery == "steering" else "followUp",
                })
                _accepted(sent)
            except ApiFailure as exc:
                payload = exc.payload
                if isinstance(payload, dict) and payload.get("accepted") is False:
                    receipt.update(status="rejected", phase="send_first_prompt", error_type=exc.category,
                                   http_status=exc.status)
                raise
            receipt.update(status="accepted_or_queued", phase="submitted",
                           acknowledgment="api_accepted_or_queued")
            return {
                "status": "accepted_or_queued",
                "acknowledgment": "api_accepted_or_queued",
                "dispatch_id": dispatch_id,
                "delivery": args.delivery,
                "lead": lead,
                "worker": worker,
                "receipt": str(args.receipt),
            }
        except Exception as exc:
            # Preserve any ID returned with an error body; creation may have
            # reached Pi Web even when its response was not successful.
            if receipt.data.get("worker_session_id") is None and isinstance(exc, ApiFailure):
                worker_id = _extract_return_id(exc.payload)
                if worker_id is not None:
                    receipt.update(worker_session_id=worker_id)
            if receipt.data.get("status") == "creation_unknown":
                receipt.update(status="creation_unknown", error_type=_safe_error_type(exc),
                               http_status=_request_error_status(exc))
            elif receipt.data.get("status") == "delivery_unknown":
                receipt.update(error_type=_safe_error_type(exc), http_status=_request_error_status(exc))
            else:
                receipt.update(error_type=_safe_error_type(exc), http_status=_request_error_status(exc))
            raise


def _check_preserved_settings(before: dict[str, object], after: dict[str, object]) -> None:
    if after.get("model") is None or after.get("effort") not in THINKING_LEVELS:
        raise CliFailure("recipient runtime model and effort must be known before sending")
    for key in ("model", "effort"):
        expected = before.get(key)
        if expected is not None and after.get(key) != expected:
            raise CliFailure(f"recipient {key} changed while loading; no prompt was sent")


def _pair_snapshot(client: PiWebClient, args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    lead = inspect_session(client, args.lead_session, args.cwd)
    worker = inspect_session(client, args.worker_session, args.cwd)
    return lead, worker


def inspect_or_send_pair(client: PiWebClient, args: argparse.Namespace) -> dict[str, object]:
    lead, worker = _pair_snapshot(client, args)
    target_role = args.to
    target = lead if target_role == "lead" else worker
    target_id = args.lead_session if target_role == "lead" else args.worker_session
    summary: dict[str, object] = {
        "lead": lead,
        "worker": worker,
        "target": target_role,
        "target_session_id": target_id,
        "cwd": str(args.cwd),
        "status": "inspected_only",
    }
    if not args.send:
        return summary

    message = read_message(args.message)
    dispatch_id = str(uuid4())
    sender_role = "worker" if target_role == "lead" else "lead"
    sender_id = args.worker_session if target_role == "lead" else args.lead_session
    envelope = make_envelope(
        message,
        dispatch_id=dispatch_id,
        sender_id=sender_id,
        sender_role=sender_role,
        recipient_id=target_id,
        recipient_role=target_role,
    )
    receipt_data: dict[str, object] = {
        "kind": "pi_web_peer_session",
        "delivery": args.delivery,
        "status": "prepared",
        "phase": "load_target_state",
        "dispatch_id": dispatch_id,
        "lead": lead,
        "worker": worker,
        "target": target_role,
        "target_session_id": target_id,
        "cwd": str(args.cwd),
        "message_sha256": message_hash(message),
        "sent_sha256": message_hash(envelope),
    }
    with Receipt(args.receipt, receipt_data) as receipt:
        try:
            # POST get_state intentionally loads an unloaded persisted session.
            # The receipt above is durable before that API side effect.
            loaded = client.request("POST", f"/api/agent/{target_id}", {"type": "get_state"})
            _accepted(loaded)
            target_state = _validate_state(loaded.get("data"), target_id)
            _check_preserved_settings(target, target_state)
            receipt.update(status="prepared", phase="send_prompt", loaded_target={
                "id": target_id,
                "model": target_state["model"],
                "effort": target_state["effort"],
                "is_streaming": target_state["is_streaming"],
                "is_prompt_running": target_state["is_prompt_running"],
            })
            receipt.update(status="delivery_unknown", phase="send_prompt")
            try:
                sent = client.request("POST", f"/api/agent/{target_id}", {
                    "type": "prompt",
                    "message": envelope,
                    "streamingBehavior": "steer" if args.delivery == "steering" else "followUp",
                })
                _accepted(sent)
            except ApiFailure as exc:
                payload = exc.payload
                if isinstance(payload, dict) and payload.get("accepted") is False:
                    receipt.update(status="rejected", phase="send_prompt", error_type=exc.category,
                                   http_status=exc.status)
                raise
            receipt.update(status="accepted_or_queued", phase="submitted",
                           acknowledgment="api_accepted_or_queued")
        except Exception as exc:
            if receipt.data.get("status") != "rejected":
                receipt.update(error_type=_safe_error_type(exc), http_status=_request_error_status(exc))
            raise
    summary.update(status="accepted_or_queued", acknowledgment="api_accepted_or_queued",
                   dispatch_id=dispatch_id, receipt=str(args.receipt), delivery=args.delivery)
    return summary


def list_models(client: PiWebClient, cwd: Path) -> list[dict[str, object]]:
    """Return only the public selection fields; never echo backend error bodies."""
    catalog = client.request("GET", "/api/models?" + urllib.parse.urlencode({"cwd": str(cwd)}))
    if catalog.get("modelError"):
        raise CliFailure("Pi Web model catalog is unavailable; no session created")
    entries, levels = catalog.get("modelList"), catalog.get("thinkingLevels")
    if not isinstance(entries, list) or not isinstance(levels, dict):
        raise CliFailure("Pi Web model catalog has an unsupported shape; no session created")
    result = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise CliFailure("Pi Web model catalog has an invalid entry; no session created")
        provider, model = entry.get("provider"), entry.get("id")
        if not all(isinstance(v, str) and v.strip() for v in (provider, model)):
            raise CliFailure("Pi Web model catalog has an invalid identity; no session created")
        efforts = levels.get(f"{provider}:{model}")
        if (not isinstance(efforts, list) or not efforts
                or any(not isinstance(v, str) or v not in THINKING_LEVELS for v in efforts)
                or (provider, model) in seen):
            raise CliFailure("Pi Web model catalog has invalid effort metadata; no session created")
        seen.add((provider, model))
        result.append({"provider": provider, "model": model, "efforts": efforts})
    return result


def operate(args: argparse.Namespace) -> dict[str, object]:
    client = PiWebClient(args.base_url)
    if args.list_models:
        return {"status": "listed_only", "cwd": str(args.cwd), "models": list_models(client, args.cwd)}
    if args.create:
        return create_and_send(client, args)
    return inspect_or_send_pair(client, args)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = operate(args)
    except FileExistsError:
        print("pi_worker_turn: receipt path already exists; reconcile it before another send", file=sys.stderr)
        return 1
    except (CliFailure, ApiFailure, OSError, ValueError, TypeError) as exc:
        print(f"pi_worker_turn: {_error_text(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
