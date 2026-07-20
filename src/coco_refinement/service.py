"""FastAPI surface for the standalone COCO refinement human loop."""

from __future__ import annotations

import sqlite3
from pathlib import Path as FileSystemPath
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from src.common.errors import DataContractError, RuntimeContractError
from src.coco_refinement.commit_service import CommitService, CommitServiceError
from src.coco_refinement.focus_service import FocusQueueService, FocusQueueServiceError
from src.coco_refinement.http_security import (
    LOCAL_OPERATOR,
    LocalHttpSecurity,
    OpaqueSessionStore,
    validate_browser_origin,
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
    ObjectProjectionConflict,
    TaskService,
    TaskServiceError,
)
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY


_STATIC_ROOT = FileSystemPath(__file__).with_name("static")
_STATIC_RESPONSE_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
    ),
    "Referrer-Policy": "no-referrer",
}
_STATIC_ASSETS = {
    "api-client.js": "text/javascript",
    "app.css": "text/css",
    "app.js": "text/javascript",
    "class-search.js": "text/javascript",
    "commit-controller.js": "text/javascript",
    "draft-controller.js": "text/javascript",
    "editor-geometry.js": "text/javascript",
    "svg-editor.js": "text/javascript",
}


class DraftPutBody(BaseModel):
    """Strict full-Draft mutation body; paths and principals are server-owned."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mutation_id: Annotated[
        str,
        Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
    ]
    expected_revision: Annotated[int, Field(ge=0)]
    expected_generation: Annotated[int, Field(ge=0)]
    expected_base_row_hash: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    objects: list[dict[str, Any]]


class CommitPostBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    batch_id: Annotated[
        str,
        Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
    ]


class FocusCreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    image_paths: Annotated[list[str], Field(min_length=1, max_length=1000)]


class _ObjectProjectionBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pixel_xyxy: Annotated[list[float], Field(min_length=4, max_length=4)]
    category_name: Annotated[str, Field(min_length=1, max_length=100)]
    expected_revision: Annotated[int, Field(ge=0)]
    expected_generation: Annotated[int, Field(ge=0)]
    expected_base_row_hash: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]


class CreateObjectProjectionBody(_ObjectProjectionBody):
    operation: Literal["create"]
    request_id: Annotated[
        str,
        Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
    ]


class UpdateObjectProjectionBody(_ObjectProjectionBody):
    operation: Literal["update"]
    region_key: Annotated[str, Field(min_length=1, max_length=500)]


ObjectProjectionBody = Annotated[
    CreateObjectProjectionBody | UpdateObjectProjectionBody,
    Field(discriminator="operation"),
]


def create_service_app(
    task_service: TaskService,
    *,
    bind_host: str,
    port: int,
    browser_origin: str | None = None,
    allow_browser_port_remap: bool = False,
    sessions: OpaqueSessionStore | None = None,
    commit_service: CommitService | None = None,
    focus_service: FocusQueueService | None = None,
) -> FastAPI:
    """Build the HTTP app without binding a port or changing runtime lifecycle."""

    if not isinstance(task_service, TaskService):
        raise TypeError("task_service must be a TaskService")
    authority = validate_loopback_authority(bind_host, port)
    browser_authority = (
        None if browser_origin is None else validate_browser_origin(browser_origin)
    )
    security = LocalHttpSecurity(
        authority,
        browser_authority=browser_authority,
        allow_browser_port_remap=allow_browser_port_remap,
        sessions=sessions,
    )
    app = FastAPI(
        title="COCO Refinement",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.task_service = task_service
    app.state.commit_service = commit_service
    app.state.focus_service = focus_service
    app.state.local_principal = LOCAL_OPERATOR
    app.state.bound_authority = authority
    app.state.browser_authority = browser_authority

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
    async def task_not_found(
        _request: Request, _exc: TaskNotFoundError
    ) -> JSONResponse:
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

    @app.exception_handler(ObjectProjectionConflict)
    async def object_projection_conflict(
        _request: Request, exc: ObjectProjectionConflict
    ) -> JSONResponse:
        return JSONResponse(exc.to_dict(), status_code=409)

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
                "coco_refinement.object_projection_operation",
            }
            else 503
        )
        return _error_response(status, exc.code, "task request could not be completed")

    @app.exception_handler(CommitServiceError)
    async def commit_service_error(
        _request: Request, exc: CommitServiceError
    ) -> JSONResponse:
        if exc.code == "coco_refinement.commit_not_found":
            status = 404
        elif exc.code in {
            "coco_refinement.commit_busy",
            "coco_refinement.commit_conflict",
            "coco_refinement.no_pending_drafts",
        }:
            status = 409
        elif exc.code in {
            "coco_refinement.split",
            "coco_refinement.commit_invalid_draft",
        }:
            status = 422
        else:
            status = 503
        return _error_response(
            status, exc.code, "Commit request could not be completed"
        )

    @app.exception_handler(FocusQueueServiceError)
    async def focus_service_error(
        _request: Request, exc: FocusQueueServiceError
    ) -> JSONResponse:
        if exc.code == "coco_refinement.focus_not_found":
            status = 404
        elif exc.code in {
            "coco_refinement.focus_busy",
            "coco_refinement.focus_active",
            "coco_refinement.focus_conflict",
            "coco_refinement.commit_busy",
            "coco_refinement.focus_publication_not_retryable",
            "coco_refinement.focus_publication_retry_required",
            "coco_refinement.no_pending_drafts",
        }:
            status = 409
        elif exc.code in {
            "coco_refinement.focus_paths",
            "coco_refinement.focus_tasks",
            "coco_refinement.focus_split",
            "coco_refinement.focus_locators",
            "coco_refinement.focus_locator_duplicate",
            "coco_refinement.focus_locator_resolution",
            "coco_refinement.focus_locator_scope",
        }:
            status = 422
        else:
            status = 503
        error: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.context:
            error["details"] = dict(exc.context)
        return JSONResponse(
            {"error": error},
            status_code=status,
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.exception_handler(DataContractError)
    async def data_contract_error(
        _request: Request, exc: DataContractError
    ) -> JSONResponse:
        return _error_response(422, exc.code, "Draft objects are invalid")

    @app.exception_handler(RepositoryError)
    async def repository_error(_request: Request, exc: RepositoryError) -> JSONResponse:
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

    @app.get("/api/categories")
    async def get_categories() -> dict[str, Any]:
        """Expose the frozen official registry without duplicating it in the client."""

        return {
            "fingerprint": COCO80_REGISTRY.fingerprint,
            "categories": [
                {"id": category.id, "name": category.name}
                for category in COCO80_REGISTRY.categories
            ],
        }

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

    @app.post("/api/splits/{split}/tasks/{task_id}/objects/canonicalize")
    async def canonicalize_object(
        split: Annotated[str, Path(pattern=r"^(train|val)$")],
        task_id: Annotated[str, Path(min_length=1, max_length=200)],
        body: ObjectProjectionBody,
    ) -> dict[str, Any]:
        return task_service.canonicalize_object_projection(
            split=split,
            task_id=task_id,
            operation=body.operation,
            request_id=body.request_id
            if isinstance(body, CreateObjectProjectionBody)
            else None,
            region_key=body.region_key
            if isinstance(body, UpdateObjectProjectionBody)
            else None,
            pixel_xyxy=body.pixel_xyxy,
            category_name=body.category_name,
            expected_revision=body.expected_revision,
            expected_generation=body.expected_generation,
            expected_base_row_hash=body.expected_base_row_hash,
        ).to_dict()

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

    if commit_service is not None:

        @app.post("/api/splits/{split}/commits")
        async def enqueue_commit(
            split: Annotated[str, Path(pattern=r"^(train|val)$")],
            body: CommitPostBody,
        ) -> JSONResponse:
            status = commit_service.enqueue(split=split, batch_id=body.batch_id)
            return JSONResponse(
                status.to_dict(),
                status_code=202 if status.status == "queued" else 200,
            )

        @app.get("/api/splits/{split}/commits/{batch_id}")
        async def get_commit_status(
            split: Annotated[str, Path(pattern=r"^(train|val)$")],
            batch_id: Annotated[
                str,
                Path(
                    min_length=1,
                    max_length=200,
                    pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
                ),
            ],
        ) -> dict[str, Any]:
            return commit_service.status(split=split, batch_id=batch_id).to_dict()

        @app.get("/api/splits/{split}/state")
        async def get_project_state(
            split: Annotated[str, Path(pattern=r"^(train|val)$")],
        ) -> dict[str, Any]:
            return commit_service.project_state(split=split).to_dict()

    if focus_service is not None:

        @app.get("/api/focus")
        async def get_focus_queue() -> dict[str, Any]:
            return focus_service.status()

        @app.post("/api/focus")
        async def create_focus_queue(body: FocusCreateBody) -> JSONResponse:
            return JSONResponse(focus_service.create(body.image_paths), status_code=201)

        @app.delete("/api/focus")
        async def release_focus_queue() -> dict[str, Any]:
            return focus_service.release()

        @app.post("/api/focus/commits")
        async def enqueue_focus_commit(body: CommitPostBody) -> JSONResponse:
            value = focus_service.enqueue(batch_id=body.batch_id)
            status = value.get("queue", {}).get("batch", {}).get("status")
            return JSONResponse(value, status_code=202 if status == "queued" else 200)

        @app.post("/api/focus/publication/retry")
        async def retry_focus_publication() -> JSONResponse:
            return JSONResponse(focus_service.retry_publication(), status_code=202)

    @app.get("/", include_in_schema=False)
    async def get_application_shell() -> Response:
        return _static_file("index.html", media_type="text/html")

    @app.get("/app.css", include_in_schema=False)
    async def get_application_styles() -> Response:
        return _static_file("app.css", media_type=_STATIC_ASSETS["app.css"])

    @app.get("/app.js", include_in_schema=False)
    async def get_application_module() -> Response:
        return _static_file("app.js", media_type=_STATIC_ASSETS["app.js"])

    @app.get("/class-search.js", include_in_schema=False)
    async def get_class_search_module() -> Response:
        return _static_file(
            "class-search.js", media_type=_STATIC_ASSETS["class-search.js"]
        )

    @app.get("/api-client.js", include_in_schema=False)
    async def get_api_client_module() -> Response:
        return _static_file("api-client.js", media_type=_STATIC_ASSETS["api-client.js"])

    @app.get("/draft-controller.js", include_in_schema=False)
    async def get_draft_controller_module() -> Response:
        return _static_file(
            "draft-controller.js", media_type=_STATIC_ASSETS["draft-controller.js"]
        )

    @app.get("/commit-controller.js", include_in_schema=False)
    async def get_commit_controller_module() -> Response:
        return _static_file(
            "commit-controller.js", media_type=_STATIC_ASSETS["commit-controller.js"]
        )

    @app.get("/editor-geometry.js", include_in_schema=False)
    async def get_editor_geometry_module() -> Response:
        return _static_file(
            "editor-geometry.js", media_type=_STATIC_ASSETS["editor-geometry.js"]
        )

    @app.get("/svg-editor.js", include_in_schema=False)
    async def get_svg_editor_module() -> Response:
        return _static_file("svg-editor.js", media_type=_STATIC_ASSETS["svg-editor.js"])

    return app


def create_runtime_service_app(
    runtime: object,
    *,
    bind_host: str,
    port: int,
    browser_origin: str | None = None,
    allow_browser_port_remap: bool = False,
    sessions: OpaqueSessionStore | None = None,
) -> FastAPI:
    """Adapt an assembled runtime to HTTP while leaving bind/start to the launcher."""

    commit_service = CommitService.from_runtime(runtime)
    return create_service_app(
        TaskService.from_runtime(runtime),
        bind_host=bind_host,
        port=port,
        browser_origin=browser_origin,
        allow_browser_port_remap=allow_browser_port_remap,
        sessions=sessions,
        commit_service=commit_service,
        focus_service=FocusQueueService.from_runtime(
            runtime, commit_service=commit_service
        ),
    )


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


def _static_file(filename: str, *, media_type: str) -> Response:
    """Serve one packaged asset from the fixed application allowlist."""

    if filename != "index.html" and filename not in _STATIC_ASSETS:
        return _error_response(
            404, "coco_refinement.static_not_found", "static asset was not found"
        )
    path = _STATIC_ROOT / filename
    if not path.is_file():
        return _error_response(
            404, "coco_refinement.static_not_found", "static asset was not found"
        )
    return FileResponse(
        path,
        media_type=media_type,
        headers=dict(_STATIC_RESPONSE_HEADERS),
    )


__all__ = [
    "CommitPostBody",
    "CreateObjectProjectionBody",
    "DraftPutBody",
    "FocusCreateBody",
    "UpdateObjectProjectionBody",
    "create_runtime_service_app",
    "create_service_app",
]
