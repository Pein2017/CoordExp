from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.http_security import CSRF_HEADER
from src.coco_refinement.models import NativeTaskIdentity
from src.coco_refinement.repository import (
    CompactTaskRecord,
    ProjectRecord,
    SqliteDraftRepository,
)
from src.coco_refinement.service import create_service_app
from src.coco_refinement.task_service import TaskService
from src.label_studio_coco_refinement.geometry import pixel_xyxy_to_norm1000
from src.label_studio_coco_refinement.store import DraftRestore, sha256_json


HOST = "127.0.0.1:9144"
ORIGIN = f"http://{HOST}"
PATH = "/api/splits/train/tasks/train:7/objects/canonicalize"
LOCAL_KEY = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603938"


class _Store:
    def __init__(self, split: str, project_id: str, restore: DraftRestore) -> None:
        self.split = split
        self.project_id = project_id
        self.restore = restore
        self.mismatch = False

    def resolve_task_navigation_row_index(
        self, *, split: str, project_id: str, task_id: str, image_id: int
    ) -> int:
        assert (split, project_id, task_id, image_id) == (
            self.split,
            self.project_id,
            f"{self.split}:{self.restore.image_id}",
            self.restore.image_id,
        )
        return 0

    def restore_task_navigation(
        self, image_id: int, *, projected_generation: int, projected_row_hash: str
    ) -> DraftRestore:
        assert image_id == self.restore.image_id
        if not self.mismatch:
            return self.restore
        return DraftRestore(
            split=self.restore.split,
            image_id=self.restore.image_id,
            generation=projected_generation + 1,
            row_hash=projected_row_hash,
            row=self.restore.row,
            region_id_mapping=self.restore.region_id_mapping,
        )


def _source(split: str, object_id: int) -> dict[str, Any]:
    return {
        "region_key": f"{split}:coco:{object_id}",
        "bbox_2d": [10, 20, 300, 400],
        "category_name": "person",
        "category_id": 1,
        "coco_ann_id": object_id,
    }


def _local() -> dict[str, Any]:
    return {
        "region_key": LOCAL_KEY,
        "bbox_2d": [30, 40, 500, 600],
        "category_name": "bicycle",
        "category_id": 2,
    }


def _roi() -> dict[str, Any]:
    return {
        "region_key": "roi:receipt-1:result-1",
        "bbox_2d": [100, 110, 700, 710],
        "category_name": "car",
        "category_id": 3,
        "metadata": {
            "inference_origin": True,
            "receipt_id": "receipt-1",
            "request_id": "infer-1",
            "result_id": "result-1",
            "draft_revision": "0",
        },
    }


@pytest.fixture
def api(tmp_path: Path):
    image_root = tmp_path / "images"
    repository = SqliteDraftRepository(tmp_path / "state.sqlite3")
    stores: dict[str, _Store] = {}
    for split, image_id, object_id, color in (
        ("train", 7, 101, (255, 0, 0)),
        ("val", 8, 102, (0, 0, 255)),
    ):
        image = image_root / f"{split}2017" / f"{image_id:012d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color).save(image, format="JPEG")
        row = {
            "image_id": image_id,
            "images": [f"images/{split}2017/{image_id:012d}.jpg"],
            "objects": [
                {
                    key: value
                    for key, value in _source(split, object_id).items()
                    if key != "region_key"
                }
            ],
        }
        row_hash = sha256_json(row)
        baseline = canonicalize_objects(
            [_source(split, object_id)], split=split  # type: ignore[arg-type]
        )
        project_id = f"coco:{split}"
        task = CompactTaskRecord(
            project_id=project_id,
            identity=NativeTaskIdentity(
                split=split, image_id=image_id, source_row_index=0  # type: ignore[arg-type]
            ),
            image_locator=f"{split}2017/{image_id:012d}.jpg",
            image_width=64,
            image_height=48,
            image_fingerprint=hashlib.sha256(image.read_bytes()).hexdigest(),
            current_generation=0,
            base_row_hash=row_hash,
            committed_result_hash=baseline.result_hash,
        )
        repository.bootstrap_project(
            ProjectRecord(
                project_id=project_id,
                split=split,  # type: ignore[arg-type]
                source_fingerprint=("a" if split == "train" else "b") * 64,
                task_count=1,
            ),
            (task,),
        )
        stores[split] = _Store(
            split,
            project_id,
            DraftRestore(
                split=split,
                image_id=image_id,
                generation=0,
                row_hash=row_hash,
                row=row,
                region_id_mapping={f"{split}:coco:{object_id}": object_id},
            ),
        )
    task_service = TaskService(
        repository=repository,
        stores=stores,  # type: ignore[arg-type]
        project_ids={"train": "coco:train", "val": "coco:val"},
        image_roots={"train": image_root, "val": image_root},
    )
    app = create_service_app(task_service, bind_host="127.0.0.1", port=9144)
    with TestClient(app, base_url=ORIGIN) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        yield client, csrf, repository, stores


def _binding(client: TestClient) -> dict[str, Any]:
    task = client.get("/api/splits/train/tasks/train:7").json()
    return {
        "expected_revision": task["revision"],
        "expected_generation": task["generation"],
        "expected_base_row_hash": task["base_row_hash"],
    }


def _post(client: TestClient, csrf: str, body: dict[str, Any]):
    return client.post(
        PATH,
        json=body,
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )


def _put(
    client: TestClient,
    csrf: str,
    *,
    mutation_id: str,
    binding: dict[str, Any],
    objects: list[dict[str, Any]],
):
    return client.put(
        "/api/splits/train/tasks/train:7/draft",
        json={"mutation_id": mutation_id, **binding, "objects": objects},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )


def _create_body(client: TestClient, **updates: Any) -> dict[str, Any]:
    body = {
        "operation": "create",
        "request_id": "create-1",
        "pixel_xyxy": [0.1, 0.1, 63.1, 47.1],
        "category_name": "traffic light",
        **_binding(client),
    }
    body.update(updates)
    return body


def test_create_is_replayable_clipped_outward_and_nonpersistent(api) -> None:
    client, csrf, repository, _stores = api
    body = _create_body(client)
    first = _post(client, csrf, body)
    replay = _post(client, csrf, body)

    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    obj = first.json()["object"]
    assert obj["region_key"].startswith("local:")
    assert obj["bbox_2d"] == list(
        pixel_xyxy_to_norm1000([0.1, 0.1, 63.1, 47.1], image_width=64, image_height=48)
    )
    assert (obj["category_name"], obj["category_id"]) == ("traffic light", 10)
    assert repository.count_mutations() == 0
    assert repository.count_drafts(project_id="coco:train") == 0

    clipped = _post(
        client,
        csrf,
        _create_body(
            client,
            request_id="create-clipped",
            pixel_xyxy=[-4, -2, 70, 55],
        ),
    )
    assert clipped.status_code == 200
    assert clipped.json()["object"]["bbox_2d"] == [0, 0, 999, 999]


@pytest.mark.parametrize(
    "updates",
    [
        {"pixel_xyxy": [1.0, 1.0, 1.0, 2.0]},
        {"pixel_xyxy": [float("nan"), 1.0, 2.0, 3.0]},
        {"category_name": "bike"},
        {"category_id": 1},
        {"region_key": LOCAL_KEY},
    ],
)
def test_create_rejects_invalid_geometry_class_and_shape(api, updates) -> None:
    client, csrf, repository, _stores = api
    body = _create_body(client, **updates)
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in body.get("pixel_xyxy", [])
    ):
        response = client.post(
            PATH,
            content=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Origin": ORIGIN,
                CSRF_HEADER: csrf,
            },
        )
    else:
        response = _post(client, csrf, body)

    assert response.status_code == 422
    assert repository.count_mutations() == 0


def test_update_preserves_source_local_and_roi_identity(api) -> None:
    client, csrf, repository, _stores = api
    initial = client.get("/api/splits/train/tasks/train:7").json()
    objects = [_source("train", 101), _local(), _roi()]
    saved = _put(
        client,
        csrf,
        mutation_id="seed-identities",
        binding=_binding(client),
        objects=objects,
    )
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    before_mutations = repository.count_mutations()

    for index, original in enumerate(objects):
        response = _post(
            client,
            csrf,
            {
                "operation": "update",
                "region_key": original["region_key"],
                "pixel_xyxy": [5.0 + index, 6.0, 40.0, 41.0],
                "category_name": "dog",
                **_binding(client),
            },
        )
        assert response.status_code == 200
        projected = response.json()["object"]
        assert projected["region_key"] == original["region_key"]
        assert projected.get("coco_ann_id") == original.get("coco_ann_id")
        assert projected.get("metadata") == original.get("metadata")
        assert (projected["category_name"], projected["category_id"]) == ("dog", 18)

    assert repository.count_mutations() == before_mutations
    assert client.get("/api/splits/train/tasks/train:7").json()["objects"] == objects
    assert initial["revision"] == 0


def test_stale_or_missing_update_is_409_without_mutation(api) -> None:
    client, csrf, repository, _stores = api
    stale = _post(
        client,
        csrf,
        _create_body(client, expected_revision=99),
    )
    missing = _post(
        client,
        csrf,
        {
            "operation": "update",
            "region_key": "local:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "pixel_xyxy": [1.0, 2.0, 30.0, 40.0],
            "category_name": "person",
            **_binding(client),
        },
    )

    assert stale.status_code == 409
    assert stale.json()["conflict"]["reason"] == "revision"
    assert stale.json()["binding"]["revision"] == 0
    assert missing.status_code == 409
    assert missing.json()["conflict"]["reason"] == "region_key"
    assert repository.count_mutations() == 0


def test_projection_then_put_succeeds_but_intervening_put_conflicts(api) -> None:
    client, csrf, repository, _stores = api
    baseline = client.get("/api/splits/train/tasks/train:7").json()["objects"]
    projected = _post(client, csrf, _create_body(client)).json()
    first = _put(
        client,
        csrf,
        mutation_id="save-projected",
        binding={
            "expected_revision": projected["binding"]["revision"],
            "expected_generation": projected["binding"]["generation"],
            "expected_base_row_hash": projected["binding"]["base_row_hash"],
        },
        objects=[*baseline, projected["object"]],
    )
    assert first.status_code == 200

    stale_binding = _binding(client)
    second_projection = _post(
        client, csrf, _create_body(client, request_id="create-2")
    ).json()
    intervening_objects = [
        {**first.json()["objects"][0], "bbox_2d": [11, 20, 300, 400]},
        *first.json()["objects"][1:],
    ]
    intervening = _put(
        client,
        csrf,
        mutation_id="intervening-save",
        binding=stale_binding,
        objects=intervening_objects,
    )
    assert intervening.status_code == 200
    stale_save = _put(
        client,
        csrf,
        mutation_id="stale-projected-save",
        binding=stale_binding,
        objects=[*first.json()["objects"], second_projection["object"]],
    )
    assert stale_save.status_code == 409
    assert stale_save.json()["conflict"]["reason"] == "revision"
    assert repository.count_mutations() == 3


def test_projection_authority_failure_is_503(api) -> None:
    client, csrf, repository, stores = api
    body = _create_body(client)
    stores["train"].mismatch = True
    response = _post(client, csrf, body)

    assert response.status_code == 503
    assert repository.count_mutations() == 0
