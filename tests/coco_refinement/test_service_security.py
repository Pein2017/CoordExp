from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient
import pytest

from src.coco_refinement.http_security import (
    CSRF_HEADER,
    HttpSecurityError,
    SESSION_COOKIE,
    validate_loopback_authority,
)
from src.coco_refinement.service import create_service_app
from src.coco_refinement.task_service import TaskService


HOST = "127.0.0.1:9123"
ORIGIN = f"http://{HOST}"


def _app():
    uncalled_service = object.__new__(TaskService)
    return _app_with_service(uncalled_service)


def _app_with_service(task_service: TaskService):
    return create_service_app(
        task_service,
        bind_host="127.0.0.1",
        port=9123,
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


def test_exact_host_is_required_and_forwarded_authority_is_forbidden() -> None:
    with TestClient(_app(), base_url=ORIGIN) as client:
        wrong = client.get("/api/session", headers={"Host": "127.0.0.1:9124"})
        assert wrong.status_code == 400
        assert wrong.headers["cache-control"] == "no-store"

        forwarded = client.get("/api/session", headers={"X-Forwarded-Host": HOST})
        assert forwarded.status_code == 400
        assert "forwarded" in forwarded.json()["error"]["message"]


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
