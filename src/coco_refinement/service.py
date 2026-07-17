"""FastAPI surface for the standalone COCO refinement human loop."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from src.common.errors import DataContractError, RuntimeContractError
from src.coco_refinement.http_security import (
    LOCAL_OPERATOR,
    LocalHttpSecurity,
    OpaqueSessionStore,
    validate_loopback_authority,
)
from src.coco_refinement.repository import (
    MutationCollisionError,
    ProjectNotFoundError,
    RepositoryError,
    TaskNotFoundError,
)
from src.coco_refinement.task_service import (
    MAX_TASK_PAGE,
    CrossAuthorityConflict,
    TaskService,
    TaskServiceError,
)


class DraftPutBody(BaseModel):
    """Strict full-Draft mutation body; paths and principals are server-owned."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mutation_id: Annotated[
        str, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    ]
    expected_revision: Annotated[int, Field(ge=0)]
    expected_generation: Annotated[int, Field(ge=0)]
    expected_base_row_hash: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    objects: list[dict[str, Any]]


def create_service_app(
    task_service: TaskService,
    *,
    bind_host: str,
    port: int,
    sessions: OpaqueSessionStore | None = None,
) -> FastAPI:
    """Build the HTTP app without binding a port or changing runtime lifecycle."""

    if not isinstance(task_service, TaskService):
        raise TypeError("task_service must be a TaskService")
    authority = validate_loopback_authority(bind_host, port)
    security = LocalHttpSecurity(authority, sessions=sessions)
    app = FastAPI(
        title="COCO Refinement",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.task_service = task_service
    app.state.local_principal = LOCAL_OPERATOR
    app.state.bound_authority = authority

    @app.middleware("http")
    async def local_security(request: Request, call_next: object) -> Response:
        return await security.enforce(request, call_next)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            422,
            "coco_refinement.request_invalid",
            "request does not match the strict API contract",
        )

    @app.exception_handler(TaskNotFoundError)
    async def task_not_found(_request: Request, _exc: TaskNotFoundError) -> JSONResponse:
        return _error_response(
            404, "coco_refinement.task_not_found", "task was not found"
        )

    @app.exception_handler(ProjectNotFoundError)
    async def project_not_found(
        _request: Request, _exc: ProjectNotFoundError
    ) -> JSONResponse:
        return _error_response(
            404, "coco_refinement.project_not_found", "project was not found"
        )

    @app.exception_handler(MutationCollisionError)
    async def mutation_collision(
        _request: Request, _exc: MutationCollisionError
    ) -> JSONResponse:
        return _error_response(
            409,
            "coco_refinement.mutation_collision",
            "mutation ID is already bound to another request",
        )

    @app.exception_handler(CrossAuthorityConflict)
    async def cross_authority(
        _request: Request, exc: CrossAuthorityConflict
    ) -> JSONResponse:
        return _error_response(
            503,
            exc.code,
            "task authority is reconciling; retry the same request",
        )

    @app.exception_handler(TaskServiceError)
    async def task_service_error(
        _request: Request, exc: TaskServiceError
    ) -> JSONResponse:
        status = (
            422
            if exc.code
            in {
                "coco_refinement.split",
                "coco_refinement.task_cursor",
                "coco_refinement.task_limit",
            }
            else 503
        )
        return _error_response(status, exc.code, "task request could not be completed")

    @app.exception_handler(DataContractError)
    async def data_contract_error(
        _request: Request, exc: DataContractError
    ) -> JSONResponse:
        return _error_response(422, exc.code, "Draft objects are invalid")

    @app.exception_handler(RepositoryError)
    async def repository_error(
        _request: Request, exc: RepositoryError
    ) -> JSONResponse:
        if exc.code == "coco_refinement.task_bound_identity":
            return _error_response(422, exc.code, "Draft objects are invalid")
        status = 422 if exc.code == "coco_refinement.task_cursor" else 503
        return _error_response(status, exc.code, "repository authority is unavailable")

    @app.exception_handler(RuntimeContractError)
    async def runtime_contract_error(
        _request: Request, exc: RuntimeContractError
    ) -> JSONResponse:
        return _error_response(503, exc.code, "runtime authority is unavailable")

    @app.exception_handler(sqlite3.Error)
    async def sqlite_error(_request: Request, _exc: sqlite3.Error) -> JSONResponse:
        return _error_response(
            503,
            "coco_refinement.repository_unavailable",
            "repository authority is temporarily unavailable",
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(
            500,
            "coco_refinement.internal_error",
            "request failed; reload authoritative task state before retrying",
        )

    @app.get("/api/session")
    async def issue_session() -> JSONResponse:
        return security.issue_response()

    @app.get("/api/splits/{split}/tasks")
    async def list_tasks(
        split: Annotated[str, Path(pattern=r"^(train|val)$")],
        cursor: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=MAX_TASK_PAGE)] = 50,
    ) -> dict[str, Any]:
        return task_service.list_tasks(
            split=split, cursor=cursor, limit=limit
        ).to_dict()

    @app.get("/api/splits/{split}/tasks/{task_id}")
    async def get_task(
        split: Annotated[str, Path(pattern=r"^(train|val)$")],
        task_id: Annotated[str, Path(min_length=1, max_length=200)],
    ) -> dict[str, Any]:
        return task_service.read_task(split=split, task_id=task_id).to_dict()

    @app.get("/api/splits/{split}/tasks/{task_id}/draft")
    async def get_draft(
        split: Annotated[str, Path(pattern=r"^(train|val)$")],
        task_id: Annotated[str, Path(min_length=1, max_length=200)],
    ) -> dict[str, Any]:
        return task_service.read_task(split=split, task_id=task_id).to_dict()

    @app.put("/api/splits/{split}/tasks/{task_id}/draft")
    async def put_draft(
        split: Annotated[str, Path(pattern=r"^(train|val)$")],
        task_id: Annotated[str, Path(min_length=1, max_length=200)],
        body: DraftPutBody,
    ) -> JSONResponse:
        outcome = task_service.save_draft(
            split=split,
            task_id=task_id,
            mutation_id=body.mutation_id,
            expected_revision=body.expected_revision,
            expected_generation=body.expected_generation,
            expected_base_row_hash=body.expected_base_row_hash,
            objects=body.objects,
        )
        return JSONResponse(
            outcome.to_dict(), status_code=409 if outcome.status == "conflict" else 200
        )

    @app.get("/api/splits/{split}/tasks/{task_id}/image")
    async def get_image(
        split: Annotated[str, Path(pattern=r"^(train|val)$")],
        task_id: Annotated[str, Path(min_length=1, max_length=200)],
    ) -> Response:
        image = task_service.read_image(split=split, task_id=task_id)
        return Response(
            image.body,
            media_type=image.media_type,
            headers={"ETag": image.etag, "X-Content-Type-Options": "nosniff"},
        )

    return app


def create_runtime_service_app(
    runtime: object,
    *,
    bind_host: str,
    port: int,
    sessions: OpaqueSessionStore | None = None,
) -> FastAPI:
    """Adapt an assembled runtime to HTTP while leaving bind/start to the launcher."""

    return create_service_app(
        TaskService.from_runtime(runtime),
        bind_host=bind_host,
        port=port,
        sessions=sessions,
    )


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


__all__ = ["DraftPutBody", "create_runtime_service_app", "create_service_app"]
