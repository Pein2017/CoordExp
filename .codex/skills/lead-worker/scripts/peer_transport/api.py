"""Bounded loopback HTTP transport and exact Pi session validation."""

from __future__ import annotations

import argparse
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import UUID

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
    return f"http://{parsed.netloc}"


class PiWebClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            NoRedirectHandler(),
        )

    def request(self, method: str, path: str,
                body: dict[str, object] | None = None) -> dict[str, object]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Accept": "application/json",
                     **({"Content-Type": "application/json"} if data is not None else {})},
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
            raise ApiFailure("Pi Web response exceeded the size limit", status=status,
                             category="protocol_error")
        payload = _decode_json(raw)
        if not 200 <= status < 300:
            raise ApiFailure("Pi Web returned an HTTP error", status=status, payload=payload,
                             category="http_error")
        if not isinstance(payload, dict):
            raise ApiFailure("Pi Web returned malformed JSON", status=status,
                             category="protocol_error")
        return payload


def _decode_json(raw: bytes) -> object:
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


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


def accepted(data: dict[str, object]) -> None:
    if data.get("success") is not True:
        raise CliFailure("Pi Web did not acknowledge the request")


def list_models(client: PiWebClient, cwd: Path) -> list[dict[str, object]]:
    """Return only public selection fields; never echo backend error bodies."""
    catalog = client.request("GET", "/api/models?" + urllib.parse.urlencode({"cwd": str(cwd)}))
    if catalog.get("modelError"):
        raise CliFailure("Pi Web model catalog is unavailable; no session created")
    entries, levels = catalog.get("modelList"), catalog.get("thinkingLevels")
    if not isinstance(entries, list) or not isinstance(levels, dict):
        raise CliFailure("Pi Web model catalog has an unsupported shape; no session created")
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise CliFailure("Pi Web model catalog has an invalid entry; no session created")
        provider, model = entry.get("provider"), entry.get("id")
        if not all(isinstance(value, str) and value.strip() for value in (provider, model)):
            raise CliFailure("Pi Web model catalog has an invalid identity; no session created")
        efforts = levels.get(f"{provider}:{model}")
        if (not isinstance(efforts, list) or not efforts
                or any(not isinstance(value, str) or value not in THINKING_LEVELS for value in efforts)
                or (provider, model) in seen):
            raise CliFailure("Pi Web model catalog has invalid effort metadata; no session created")
        seen.add((provider, model))
        result.append({"provider": provider, "model": model, "efforts": efforts})
    return result


def safe_error_type(exc: BaseException) -> str:
    if isinstance(exc, ApiFailure):
        return exc.category
    if isinstance(exc, CliFailure):
        return "validation_error"
    return type(exc).__name__


def request_error_status(exc: BaseException) -> int | None:
    return exc.status if isinstance(exc, ApiFailure) else None


def error_text(exc: BaseException) -> str:
    status = request_error_status(exc)
    if status is not None:
        return f"{type(exc).__name__} (HTTP {status})"
    if isinstance(exc, CliFailure):
        return str(exc)
    return f"{type(exc).__name__} ({safe_error_type(exc)})"
