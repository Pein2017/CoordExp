from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient
import pytest

from src.coco_refinement.http_security import (
    CSRF_HEADER,
    HttpSecurityError,
    LocalHttpSecurity,
    SESSION_COOKIE,
    validate_browser_origin,
    validate_loopback_authority,
)
from src.coco_refinement.service import create_service_app
from src.coco_refinement.task_service import TaskService


HOST = "127.0.0.1:9123"
ORIGIN = f"http://{HOST}"
BROWSER_HOST = "localhost:53662"
BROWSER_ORIGIN = f"http://{BROWSER_HOST}"


def _app():
    uncalled_service = object.__new__(TaskService)
    return _app_with_service(uncalled_service)


def _app_with_service(
    task_service: TaskService,
    *,
    browser_origin: str | None = None,
    allow_browser_port_remap: bool = False,
):
    return create_service_app(
        task_service,
        bind_host="127.0.0.1",
        port=9123,
        browser_origin=browser_origin,
        allow_browser_port_remap=allow_browser_port_remap,
    )


@pytest.mark.parametrize(
    "host",
    ["localhost", "0.0.0.0", "192.168.1.7", "example.com", "::1%lo"],
)
def test_bind_requires_numeric_loopback(host: str) -> None:
    with pytest.raises(HttpSecurityError, match="loopback"):
        validate_loopback_authority(host, 9123)


def test_ipv4_and_ipv6_numeric_loopback_authorities_are_canonical() -> None:
    assert validate_loopback_authority("127.0.0.1", 9123).host_header == HOST
    assert validate_loopback_authority("::1", 9123).host_header == "[::1]:9123"


def test_browser_origin_rejects_default_http_port() -> None:
    with pytest.raises(HttpSecurityError, match="default HTTP port is unsupported"):
        validate_browser_origin("http://localhost:80")


def test_exact_host_is_required_and_forwarded_authority_is_forbidden() -> None:
    with TestClient(_app(), base_url=ORIGIN) as client:
        wrong = client.get("/api/session", headers={"Host": "127.0.0.1:9124"})
        assert wrong.status_code == 400
        assert wrong.headers["cache-control"] == "no-store"

        forwarded = client.get("/api/session", headers={"X-Forwarded-Host": HOST})
        assert forwarded.status_code == 400
        assert "forwarded" in forwarded.json()["error"]["message"]


def test_browser_proxy_host_requires_one_explicit_origin() -> None:
    with TestClient(_app(), base_url=BROWSER_ORIGIN) as client:
        response = client.get("/api/session")

    assert response.status_code == 400
    assert "Host" in response.json()["error"]["message"]


class _MutationBoundaryTaskService(TaskService):
    mutation_calls: int

    def save_draft(self, **_kwargs: object):
        self.mutation_calls += 1
        raise RuntimeError("route boundary reached")


def test_explicit_browser_origin_allows_static_session_and_same_origin_mutation() -> None:
    service = object.__new__(_MutationBoundaryTaskService)
    service.mutation_calls = 0
    app = _app_with_service(service, browser_origin=BROWSER_ORIGIN)
    body = {
        "mutation_id": "browser-proxy",
        "expected_revision": 0,
        "expected_generation": 0,
        "expected_base_row_hash": "a" * 64,
        "objects": [],
    }
    path = "/api/splits/train/tasks/train:1/draft"

    with TestClient(
        app,
        base_url=BROWSER_ORIGIN,
        raise_server_exceptions=False,
    ) as client:
        assert client.get("/").status_code == 200
        csrf = client.get("/api/session").json()["csrf_token"]

        cross_authority = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: csrf, "Origin": ORIGIN},
        )
        assert cross_authority.status_code == 403
        assert service.mutation_calls == 0

        same_authority = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: csrf, "Origin": BROWSER_ORIGIN},
        )

    assert same_authority.status_code == 500
    assert service.mutation_calls == 1


def test_explicit_browser_origin_does_not_allow_other_localhost_ports() -> None:
    app = _app_with_service(
        object.__new__(TaskService), browser_origin=BROWSER_ORIGIN
    )
    with TestClient(app, base_url="http://localhost:53663") as client:
        response = client.get("/api/session")

    assert response.status_code == 400


def test_opted_in_browser_port_remap_keeps_loopback_origin_and_csrf_exact() -> None:
    service = object.__new__(_MutationBoundaryTaskService)
    service.mutation_calls = 0
    app = _app_with_service(
        service,
        browser_origin=BROWSER_ORIGIN,
        allow_browser_port_remap=True,
    )
    remapped_origin = "http://localhost:53663"
    body = {
        "mutation_id": "remapped-browser-proxy",
        "expected_revision": 0,
        "expected_generation": 0,
        "expected_base_row_hash": "a" * 64,
        "objects": [],
    }
    path = "/api/splits/train/tasks/train:1/draft"

    with TestClient(
        app,
        base_url=remapped_origin,
        raise_server_exceptions=False,
    ) as client:
        assert client.get("/").status_code == 200
        csrf = client.get("/api/session").json()["csrf_token"]

        missing_origin = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: csrf},
        )
        assert missing_origin.status_code == 403
        assert service.mutation_calls == 0

        cross_port = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: csrf, "Origin": BROWSER_ORIGIN},
        )
        assert cross_port.status_code == 403
        assert service.mutation_calls == 0

        wrong_csrf = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: "wrong", "Origin": remapped_origin},
        )
        assert wrong_csrf.status_code == 403
        assert service.mutation_calls == 0

        same_origin = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: csrf, "Origin": remapped_origin},
        )
        assert same_origin.status_code == 500
        assert service.mutation_calls == 1

        non_loopback = client.get(
            "/api/session", headers={"Host": "example.com:53663"}
        )
        assert non_loopback.status_code == 400


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:1",
        "http://localhost:65535",
        "http://127.0.0.1:53663",
    ],
)
def test_opted_in_browser_port_remap_accepts_only_configured_local_hosts(
    origin: str,
) -> None:
    app = _app_with_service(
        object.__new__(TaskService),
        browser_origin=BROWSER_ORIGIN,
        allow_browser_port_remap=True,
    )
    with TestClient(app, base_url=origin) as client:
        response = client.get("/api/session")

    assert response.status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "localhost:80",
        "LOCALHOST:53663",
        "localhost.:53663",
        "127.0.0.2:53663",
        "[::1]:53663",
        "192.168.1.7:53663",
        "example.com:53663",
    ],
)
def test_opted_in_browser_port_remap_rejects_other_or_noncanonical_hosts(
    host: str,
) -> None:
    app = _app_with_service(
        object.__new__(TaskService),
        browser_origin=BROWSER_ORIGIN,
        allow_browser_port_remap=True,
    )
    with TestClient(app, base_url=BROWSER_ORIGIN) as client:
        response = client.get("/api/session", headers={"Host": host})

    assert response.status_code == 400


def test_opted_in_browser_port_remap_still_rejects_forwarded_authority() -> None:
    app = _app_with_service(
        object.__new__(TaskService),
        browser_origin=BROWSER_ORIGIN,
        allow_browser_port_remap=True,
    )
    with TestClient(app, base_url="http://localhost:53663") as client:
        response = client.get(
            "/api/session", headers={"X-Forwarded-Host": "localhost:53663"}
        )

    assert response.status_code == 400
    assert "forwarded" in response.json()["error"]["message"]


def test_browser_port_remap_configuration_fails_closed() -> None:
    authority = validate_loopback_authority("127.0.0.1", 9123)
    with pytest.raises(HttpSecurityError, match="requires an explicit browser"):
        LocalHttpSecurity(authority, allow_browser_port_remap=True)
    with pytest.raises(HttpSecurityError, match="must be boolean"):
        LocalHttpSecurity(
            authority,
            browser_authority=validate_browser_origin(BROWSER_ORIGIN),
            allow_browser_port_remap=1,  # type: ignore[arg-type]
        )


def test_session_cookie_is_opaque_httponly_strict_and_csrf_is_separate() -> None:
    with TestClient(_app(), base_url=ORIGIN) as client:
        response = client.get("/api/session")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    csrf = response.json()["csrf_token"]
    assert isinstance(csrf, str) and len(csrf) >= 40
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/" in cookie
    assert csrf not in cookie
    assert "local-operator" not in response.text


def test_api_requires_session_and_marks_even_not_found_responses_no_store() -> None:
    app = _app()
    with TestClient(app, base_url=ORIGIN) as client:
        unauthenticated = client.get("/api/not-found")
        assert unauthenticated.status_code == 401

        client.get("/api/session")
        authenticated = client.get("/api/not-found")
        assert authenticated.status_code == 404
        assert authenticated.headers["cache-control"] == "no-store"


def test_duplicate_session_cookie_is_rejected_even_when_values_match() -> None:
    with TestClient(_app(), base_url=ORIGIN) as client:
        issued = client.get("/api/session")
        token = issued.cookies[SESSION_COOKIE]
        client.cookies.clear()

        response = client.get(
            "/api/not-found",
            headers={"Cookie": f"{SESSION_COOKIE}={token}; {SESSION_COOKIE}={token}"},
        )

    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"
    assert "duplicate session" in response.json()["error"]["message"]


def test_mutation_requires_exact_origin_and_csrf_before_route_dispatch() -> None:
    body = {
        "mutation_id": "never-dispatched",
        "expected_revision": 0,
        "expected_generation": 0,
        "expected_base_row_hash": "a" * 64,
        "objects": [],
    }
    path = "/api/splits/train/tasks/train:1/draft"
    with TestClient(_app(), base_url=ORIGIN) as client:
        session = client.get("/api/session").json()
        csrf = session["csrf_token"]

        missing_origin = client.put(path, json=body, headers={CSRF_HEADER: csrf})
        assert missing_origin.status_code == 403

        wrong_origin = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: csrf, "Origin": "http://127.0.0.1:9999"},
        )
        assert wrong_origin.status_code == 403

        wrong_csrf = client.put(
            path,
            json=body,
            headers={CSRF_HEADER: "wrong", "Origin": ORIGIN},
        )
        assert wrong_csrf.status_code == 403


class _ExplodingTaskService(TaskService):
    error: BaseException

    def list_tasks(self, *, split: str, cursor: int, limit: int):
        del split, cursor, limit
        raise self.error


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RuntimeError("secret /tmp/runtime path"), 500, "coco_refinement.internal_error"),
        (
            sqlite3.OperationalError("database /tmp/private is locked"),
            503,
            "coco_refinement.repository_unavailable",
        ),
        (
            sqlite3.DatabaseError("database /tmp/private is corrupt"),
            503,
            "coco_refinement.repository_unavailable",
        ),
    ],
)
def test_unexpected_and_sqlite_errors_are_safe_no_store_json(
    error: BaseException, status: int, code: str
) -> None:
    service = object.__new__(_ExplodingTaskService)
    service.error = error
    with TestClient(
        _app_with_service(service),
        base_url=ORIGIN,
        raise_server_exceptions=False,
    ) as client:
        client.get("/api/session")
        response = client.get("/api/splits/train/tasks")

    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == code
    assert "/tmp/" not in response.text
    assert "locked" not in response.text
    assert "corrupt" not in response.text
