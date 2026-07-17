from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from fastapi.testclient import TestClient
import pytest

import src.coco_refinement.service as service_module
from src.coco_refinement.service import create_service_app
from src.coco_refinement.task_service import TaskService
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.editor_policy import editor_policy_golden_vectors


HOST = "127.0.0.1:9143"
ORIGIN = f"http://{HOST}"


@dataclass(frozen=True)
class _TaskPage:
    def to_dict(self) -> dict[str, object]:
        return {
            "split": "train",
            "cursor": 0,
            "limit": 1,
            "total": 0,
            "next_cursor": None,
            "tasks": [],
        }


class _TaskService(TaskService):
    def list_tasks(self, *, split: str, cursor: int, limit: int) -> _TaskPage:
        assert (split, cursor, limit) == ("train", 0, 1)
        return _TaskPage()


@pytest.fixture
def app(tmp_path, monkeypatch):
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<!doctype html><title>COCO</title>")
    (static_root / "app.css").write_text("body { color: black; }")
    (static_root / "app.js").write_text("import './class-search.js';")
    (static_root / "api-client.js").write_text("export const ready = true;")
    (static_root / "class-search.js").write_text("export const ready = true;")
    (static_root / "draft-controller.js").write_text("export const ready = true;")
    (static_root / "editor-geometry.js").write_text("export const ready = true;")
    (static_root / "svg-editor.js").write_text("export const ready = true;")
    monkeypatch.setattr(service_module, "_STATIC_ROOT", static_root)

    task_service = object.__new__(_TaskService)
    return create_service_app(
        task_service,
        bind_host="127.0.0.1",
        port=9143,
    )


def test_categories_project_the_authoritative_sparse_registry(app) -> None:
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.get("/api/categories").status_code == 401
        client.get("/api/session")
        response = client.get("/api/categories")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "fingerprint": COCO80_REGISTRY.fingerprint,
        "categories": [
            {"id": category.id, "name": category.name}
            for category in COCO80_REGISTRY.categories
        ],
    }
    assert len(response.json()["categories"]) == 80
    assert response.json()["categories"][-1] == {"id": 90, "name": "toothbrush"}


@pytest.mark.parametrize(
    ("path", "media_type"),
    [
        ("/", "text/html"),
        ("/app.css", "text/css"),
        ("/app.js", "text/javascript"),
        ("/api-client.js", "text/javascript"),
        ("/class-search.js", "text/javascript"),
        ("/draft-controller.js", "text/javascript"),
        ("/editor-geometry.js", "text/javascript"),
        ("/svg-editor.js", "text/javascript"),
    ],
)
def test_exact_static_allowlist_has_safe_no_store_headers(
    app, path: str, media_type: str
) -> None:
    with TestClient(app, base_url=ORIGIN) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp


@pytest.mark.parametrize(
    "path",
    [
        "/index.html",
        "/static/app.js",
        "/unknown.js",
        "/class-search.js.map",
    ],
)
def test_unknown_static_paths_are_not_served(app, path: str) -> None:
    with TestClient(app, base_url=ORIGIN) as client:
        response = client.get(path)

    assert response.status_code == 404


def test_static_routes_do_not_shadow_existing_api(app) -> None:
    with TestClient(app, base_url=ORIGIN) as client:
        client.get("/api/session")
        response = client.get("/api/splits/train/tasks?cursor=0&limit=1")

    assert response.status_code == 200
    assert response.json() == _TaskPage().to_dict()
    assert response.headers["cache-control"] == "no-store"


def test_packaged_static_assets_match_the_exact_route_allowlist() -> None:
    static_root = Path(service_module.__file__).with_name("static")

    assert {path.name for path in static_root.iterdir() if path.is_file()} == {
        "index.html",
        "app.css",
        "app.js",
        "api-client.js",
        "class-search.js",
        "draft-controller.js",
        "editor-geometry.js",
        "svg-editor.js",
    }
    index = (static_root / "index.html").read_text()
    assert 'href="/app.css"' in index
    assert 'type="module" src="/app.js"' in index
    assert "from '/class-search.js'" in (static_root / "app.js").read_text()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional test tooling")
def test_packaged_class_search_matches_python_golden_vectors() -> None:
    """Node is test-only; the shipped application has no Node runtime dependency."""

    module_uri = (Path(service_module.__file__).with_name("static") / "class-search.js").as_uri()
    categories = [
        {"id": category.id, "name": category.name}
        for category in COCO80_REGISTRY.categories
    ]
    vectors = editor_policy_golden_vectors()["search"]
    script = """
import { rankCategories } from %s;
const categories = JSON.parse(process.argv[1]);
const vectors = JSON.parse(process.argv[2]);
const projected = vectors.map(vector => rankCategories(categories, vector.input.query)
  .slice(0, vector.input.limit)
  .map(category => ({id: category.id, name: category.name})));
process.stdout.write(JSON.stringify(projected));
""" % json.dumps(module_uri)

    completed = subprocess.run(
        [
            shutil.which("node") or "node",
            "--experimental-default-type=module",
            "--input-type=module",
            "--eval",
            script,
            json.dumps(categories, separators=(",", ":")),
            json.dumps(vectors, separators=(",", ":")),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(completed.stdout)
    expected = [
        [
            {"id": item["category_id"], "name": item["canonical_name"]}
            for item in vector["expected"]
        ]
        for vector in vectors
    ]
    assert actual == expected
