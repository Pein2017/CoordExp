"""Minimal loopback HTTP boundary for the standalone refinement service."""

from __future__ import annotations

import ipaddress
import json
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from hmac import compare_digest
from typing import Final
from urllib.parse import urlsplit

from fastapi import Request, Response
from fastapi.responses import JSONResponse


SESSION_COOKIE: Final = "coordexp_session"
CSRF_HEADER: Final = "x-csrf-token"
LOCAL_OPERATOR: Final = "local-operator"
_MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class HttpSecurityError(ValueError):
    """The requested HTTP authority cannot satisfy the local-only contract."""


@dataclass(frozen=True)
class LoopbackAuthority:
    host: str
    port: int
    host_header: str
    origin: str


@dataclass(frozen=True)
class BrowserAuthority:
    host: str
    port: int
    host_header: str
    origin: str


def validate_loopback_authority(
    host: str, port: int, *, scheme: str = "http"
) -> LoopbackAuthority:
    """Require a numeric loopback address and one concrete TCP port."""

    if not isinstance(host, str):
        raise HttpSecurityError("bind host must be a numeric loopback address")
    if "%" in host:
        raise HttpSecurityError("bind host must be an unscoped numeric loopback address")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise HttpSecurityError("bind host must be a numeric loopback address") from exc
    if not address.is_loopback:
        raise HttpSecurityError("bind host must be a numeric loopback address")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise HttpSecurityError("bind port must be an integer from 1 through 65535")
    if scheme != "http":
        raise HttpSecurityError("the standalone local service requires http")
    canonical_host = address.compressed
    host_header = (
        f"[{canonical_host}]:{port}" if address.version == 6 else f"{canonical_host}:{port}"
    )
    return LoopbackAuthority(
        host=canonical_host,
        port=port,
        host_header=host_header,
        origin=f"{scheme}://{host_header}",
    )


def validate_browser_origin(origin: object) -> BrowserAuthority:
    """Validate one explicit canonical origin exposed by a local browser proxy."""

    if not isinstance(origin, str):
        raise HttpSecurityError("browser origin must be a canonical http URL")
    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise HttpSecurityError("browser origin must be a canonical http URL") from exc
    if parsed.scheme != "http":
        raise HttpSecurityError("browser origin must use http")
    if parsed.username is not None or parsed.password is not None:
        raise HttpSecurityError("browser origin must not contain userinfo")
    if parsed.path or parsed.query or parsed.fragment:
        raise HttpSecurityError(
            "browser origin must not contain a path, query, or fragment"
        )
    if hostname is None or port is None:
        raise HttpSecurityError("browser origin must include an explicit port")
    if not 1 <= port <= 65535:
        raise HttpSecurityError("browser origin port must be from 1 through 65535")
    if port == 80:
        raise HttpSecurityError(
            "browser origin default HTTP port is unsupported for exact authority matching"
        )
    if hostname == "localhost":
        canonical_host = hostname
        is_ipv6 = False
    else:
        if "%" in hostname:
            raise HttpSecurityError(
                "browser origin host must be localhost or an unscoped numeric loopback"
            )
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise HttpSecurityError(
                "browser origin host must be localhost or a numeric loopback"
            ) from exc
        if not address.is_loopback:
            raise HttpSecurityError(
                "browser origin host must be localhost or a numeric loopback"
            )
        canonical_host = address.compressed
        is_ipv6 = address.version == 6
    host_header = (
        f"[{canonical_host}]:{port}" if is_ipv6 else f"{canonical_host}:{port}"
    )
    canonical_origin = f"http://{host_header}"
    if origin != canonical_origin:
        raise HttpSecurityError(
            f"browser origin must use the canonical form {canonical_origin}"
        )
    return BrowserAuthority(
        host=canonical_host,
        port=port,
        host_header=host_header,
        origin=canonical_origin,
    )


class OpaqueSessionStore:
    """Small process-local session/CSRF registry for one local operator."""

    def __init__(self, *, max_sessions: int = 8) -> None:
        if (
            isinstance(max_sessions, bool)
            or not isinstance(max_sessions, int)
            or max_sessions < 1
            or max_sessions > 64
        ):
            raise HttpSecurityError("max_sessions must be from 1 through 64")
        self._max_sessions = max_sessions
        self._tokens: OrderedDict[str, str] = OrderedDict()

    def issue(self) -> tuple[str, str]:
        session = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        self._tokens[session] = csrf
        while len(self._tokens) > self._max_sessions:
            self._tokens.popitem(last=False)
        return session, csrf

    def csrf_for(self, session: str | None) -> str | None:
        if not session:
            return None
        value = self._tokens.get(session)
        if value is not None:
            self._tokens.move_to_end(session)
        return value


class LocalHttpSecurity:
    """Exact-authority session and CSRF policy installed as FastAPI middleware."""

    def __init__(
        self,
        authority: LoopbackAuthority,
        *,
        browser_authority: BrowserAuthority | None = None,
        allow_browser_port_remap: bool = False,
        sessions: OpaqueSessionStore | None = None,
    ) -> None:
        if not isinstance(authority, LoopbackAuthority):
            raise HttpSecurityError("authority must be a validated loopback authority")
        if browser_authority is not None and not isinstance(
            browser_authority, BrowserAuthority
        ):
            raise HttpSecurityError(
                "browser authority must be a validated browser authority"
            )
        if not isinstance(allow_browser_port_remap, bool):
            raise HttpSecurityError("browser port remap policy must be boolean")
        if allow_browser_port_remap and browser_authority is None:
            raise HttpSecurityError(
                "browser port remap requires an explicit browser authority"
            )
        self.authority = authority
        self.browser_authority = browser_authority
        self.allow_browser_port_remap = allow_browser_port_remap
        self.sessions = sessions or OpaqueSessionStore()

    async def enforce(self, request: Request, call_next: object) -> Response:
        raw_headers = tuple(request.scope.get("headers", ()))
        names = [name.lower() for name, _value in raw_headers]
        if b"forwarded" in names or any(name.startswith(b"x-forwarded-") for name in names):
            return self._error(400, "forwarded authority headers are forbidden")
        host_values = [value.decode("latin-1") for name, value in raw_headers if name.lower() == b"host"]
        request_authority: LoopbackAuthority | BrowserAuthority | None = None
        for candidate in (self.authority, self.browser_authority):
            if candidate is not None and host_values == [candidate.host_header]:
                request_authority = candidate
                break
        if request_authority is None and self.allow_browser_port_remap:
            request_authority = self._remapped_browser_authority(host_values)
        if request_authority is None:
            return self._error(400, "request Host does not match the bound authority")

        is_api = request.url.path == "/api" or request.url.path.startswith("/api/")
        is_session_bootstrap = request.method == "GET" and request.url.path == "/api/session"
        if is_api and not is_session_bootstrap:
            session_values = _cookie_values(raw_headers, SESSION_COOKIE)
            if len(session_values) > 1:
                return self._error(400, "duplicate session cookies are forbidden")
            session = session_values[0] if session_values else None
            csrf = self.sessions.csrf_for(session)
            if csrf is None:
                return self._error(401, "a valid local session is required")
            request.state.principal = LOCAL_OPERATOR
            request.state.csrf_token = csrf
            if request.method in _MUTATION_METHODS:
                origins = [
                    value.decode("latin-1")
                    for name, value in raw_headers
                    if name.lower() == b"origin"
                ]
                csrf_values = [
                    value.decode("latin-1")
                    for name, value in raw_headers
                    if name.lower() == CSRF_HEADER.encode("ascii")
                ]
                if origins != [request_authority.origin]:
                    return self._error(
                        403, "mutation Origin does not match the request authority"
                    )
                if len(csrf_values) != 1 or not compare_digest(csrf_values[0], csrf):
                    return self._error(403, "mutation CSRF token is invalid")
                content_type = (
                    request.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if content_type == "application/json" or content_type.endswith("+json"):
                    try:
                        _strict_json_loads(await request.body())
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                        return self._error(
                            422,
                            "mutation JSON must be strict and have unique keys",
                            code="coco_refinement.request_invalid",
                        )

        response = await call_next(request)  # type: ignore[operator]
        if is_api:
            _mark_no_store(response)
        return response

    def _remapped_browser_authority(
        self, host_values: list[str]
    ) -> BrowserAuthority | None:
        if len(host_values) != 1:
            return None
        try:
            candidate = validate_browser_origin(f"http://{host_values[0]}")
        except HttpSecurityError:
            return None
        allowed_hosts = {self.authority.host}
        if self.browser_authority is not None:
            allowed_hosts.add(self.browser_authority.host)
        if candidate.host not in allowed_hosts:
            return None
        return candidate

    def issue_response(self) -> JSONResponse:
        session, csrf = self.sessions.issue()
        response = JSONResponse({"csrf_token": csrf})
        response.set_cookie(
            SESSION_COOKIE,
            session,
            httponly=True,
            secure=False,
            samesite="strict",
            path="/",
        )
        _mark_no_store(response)
        return response

    @staticmethod
    def _error(
        status_code: int,
        message: str,
        *,
        code: str = "coco_refinement.http_security",
    ) -> JSONResponse:
        response = JSONResponse(
            {"error": {"code": code, "message": message}},
            status_code=status_code,
        )
        _mark_no_store(response)
        return response


def _mark_no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


def _cookie_values(
    raw_headers: tuple[tuple[bytes, bytes], ...], name: str
) -> list[str]:
    values: list[str] = []
    for header_name, raw_value in raw_headers:
        if header_name.lower() != b"cookie":
            continue
        for item in raw_value.decode("latin-1").split(";"):
            key, separator, value = item.strip().partition("=")
            if separator and key == name:
                values.append(value.strip())
    return values


def _strict_json_loads(raw: bytes) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


__all__ = [
    "BrowserAuthority",
    "CSRF_HEADER",
    "HttpSecurityError",
    "LOCAL_OPERATOR",
    "LocalHttpSecurity",
    "LoopbackAuthority",
    "OpaqueSessionStore",
    "SESSION_COOKIE",
    "validate_browser_origin",
    "validate_loopback_authority",
]
