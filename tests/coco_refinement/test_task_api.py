from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.http_security import CSRF_HEADER
from src.coco_refinement.models import NativeTaskIdentity
from src.coco_refinement.repository import (
    CompactTaskRecord,
    ProjectRecord,
    SaveDraftRequest,
    SqliteDraftRepository,
)
from src.coco_refinement.service import create_service_app
from src.coco_refinement.task_service import TaskService
from src.label_studio_coco_refinement.store import DraftRestore, sha256_json


HOST = "127.0.0.1:9131"
ORIGIN = f"http://{HOST}"
LOCAL_KEY = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603938"


class FakeStore:
    def __init__(
        self,
        *,
        split: str,
        project_id: str,
        restore: DraftRestore,
    ) -> None:
        self.split = split
        self.project_id = project_id
        self.restore = restore
        self.restore_calls = 0
        self.on_restore: Callable[[], None] | None = None
        self.mismatch_generation_after: int | None = None

    def resolve_source_row_index(
        self,
        *,
        split: str,
        project_id: str,
        task_id: str,
        image_id: int,
    ) -> int:
        assert split == self.split
        assert project_id == self.project_id
        assert task_id == f"{split}:{image_id}"
        return 0

    def resolve_task_navigation_row_index(
        self,
        *,
        split: str,
        project_id: str,
        task_id: str,
        image_id: int,
    ) -> int:
        return self.resolve_source_row_index(
            split=split,
            project_id=project_id,
            task_id=task_id,
            image_id=image_id,
        )

    def restore_draft(self, image_id: int) -> DraftRestore:
        assert image_id == self.restore.image_id
        self.restore_calls += 1
        callback = self.on_restore
        self.on_restore = None
        if callback is not None:
            callback()
        if (
            self.mismatch_generation_after is not None
            and self.restore_calls >= self.mismatch_generation_after
        ):
            return DraftRestore(
                split=self.restore.split,
                image_id=self.restore.image_id,
                generation=self.restore.generation + 1,
                row_hash=self.restore.row_hash,
                row=self.restore.row,
                region_id_mapping=self.restore.region_id_mapping,
            )
        return self.restore

    def restore_task_navigation(
        self,
        image_id: int,
        *,
        projected_generation: int,
        projected_row_hash: str,
    ) -> DraftRestore:
        assert projected_generation >= 0
        assert len(projected_row_hash) == 64
        return self.restore_draft(image_id)


def _source_object(split: str, object_id: int) -> dict[str, Any]:
    return {
        "region_key": f"{split}:coco:{object_id}",
        "bbox_2d": [10, 20, 300, 400],
        "category_name": "person",
        "category_id": 1,
        "coco_ann_id": object_id,
    }


def _local_object(*, x1: int = 30) -> dict[str, Any]:
    return {
        "region_key": LOCAL_KEY,
        "bbox_2d": [x1, 40, 500, 600],
        "category_name": "bicycle",
        "category_id": 2,
    }


def _working_row(split: str, image_id: int, object_id: int) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "images": [f"images/{split}2017/{image_id:012d}.jpg"],
        "objects": [
            {
                "bbox_2d": [10, 20, 300, 400],
                "category_name": "person",
                "category_id": 1,
                "coco_ann_id": object_id,
            }
        ],
    }


def _write_image(path: Path, *, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def api(tmp_path: Path):
    image_root = tmp_path / "images"
    train_image = image_root / "train2017" / "000000000007.jpg"
    val_image = image_root / "val2017" / "000000000008.jpg"
    train_image_hash = _write_image(train_image, color=(255, 0, 0))
    val_image_hash = _write_image(val_image, color=(0, 0, 255))

    repository = SqliteDraftRepository(tmp_path / "state.sqlite3")
    stores: dict[str, FakeStore] = {}
    for split, image_id, object_id, image_hash in (
        ("train", 7, 101, train_image_hash),
        ("val", 8, 102, val_image_hash),
    ):
        project_id = f"coco:{split}"
        row = _working_row(split, image_id, object_id)
        row_hash = sha256_json(row)
        baseline = canonicalize_objects(
            [_source_object(split, object_id)], split=split  # type: ignore[arg-type]
        )
        task = CompactTaskRecord(
            project_id=project_id,
            identity=NativeTaskIdentity(
                split=split, image_id=image_id, source_row_index=0  # type: ignore[arg-type]
            ),
            image_locator=f"{split}2017/{image_id:012d}.jpg",
            image_width=64,
            image_height=48,
            image_fingerprint=image_hash,
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
        stores[split] = FakeStore(
            split=split,
            project_id=project_id,
            restore=DraftRestore(
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
    app = create_service_app(
        task_service,
        bind_host="127.0.0.1",
        port=9131,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        yield client, csrf, repository, stores, image_root


def _put(
    client: TestClient,
    csrf: str,
    *,
    mutation_id: str,
    revision: int,
    base_row_hash: str,
    objects: list[dict[str, Any]],
):
    return client.put(
        "/api/splits/train/tasks/train:7/draft",
        json={
            "mutation_id": mutation_id,
            "expected_revision": revision,
            "expected_generation": 0,
            "expected_base_row_hash": base_row_hash,
            "objects": objects,
        },
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )


def test_bounded_task_cursor_exposes_no_paths_or_principal(api) -> None:
    client, _csrf, _repository, _stores, _image_root = api

    response = client.get("/api/splits/train/tasks?cursor=0&limit=1")

    assert response.status_code == 200
    assert response.json() == {
        "split": "train",
        "cursor": 0,
        "limit": 1,
        "total": 1,
        "next_cursor": None,
        "tasks": [
            {
                "task_id": "train:7",
                "image_id": 7,
                "source_row_index": 0,
                "image_width": 64,
                "image_height": 48,
            }
        ],
    }
    assert "path" not in response.text
    assert "local-operator" not in response.text
    assert client.get("/api/splits/train/tasks?limit=201").status_code == 422
    assert client.get("/api/splits/train/tasks?cursor=2&limit=1").status_code == 422


def test_get_returns_authoritative_committed_objects_and_validated_image(api) -> None:
    client, _csrf, _repository, _stores, _image_root = api

    task = client.get("/api/splits/train/tasks/train:7/draft")
    image = client.get("/api/splits/train/tasks/train:7/image")

    assert task.status_code == 200
    assert task.json()["authority"] == "committed"
    assert task.json()["objects"] == [_source_object("train", 101)]
    assert "image_locator" not in task.text
    assert "project_id" not in task.text
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.headers["x-content-type-options"] == "nosniff"
    assert image.headers["etag"].startswith('"sha256:')
    assert image.headers["cache-control"] == "no-store"


def test_image_route_rejects_manifest_locator_symlink_without_disclosing_path(api) -> None:
    client, _csrf, _repository, _stores, image_root = api
    image = image_root / "train2017" / "000000000007.jpg"
    outside = image_root.parent / "outside.jpg"
    outside.write_bytes(image.read_bytes())
    image.unlink()
    image.symlink_to(outside)

    response = client.get("/api/splits/train/tasks/train:7/image")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "coco_refinement.image_symlink"
    assert str(outside) not in response.text


def test_full_draft_put_replay_conflict_and_baseline_retirement(api) -> None:
    client, csrf, repository, _stores, _image_root = api
    initial = client.get("/api/splits/train/tasks/train:7/draft").json()
    changed = [_source_object("train", 101), _local_object()]

    applied = _put(
        client,
        csrf,
        mutation_id="save-response-lost",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=changed,
    )
    replayed = _put(
        client,
        csrf,
        mutation_id="save-response-lost",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=changed,
    )
    conflict = _put(
        client,
        csrf,
        mutation_id="stale-save",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=[_source_object("train", 101)],
    )
    retired = _put(
        client,
        csrf,
        mutation_id="return-to-baseline",
        revision=1,
        base_row_hash=initial["base_row_hash"],
        objects=[_source_object("train", 101)],
    )

    assert applied.status_code == 200
    assert applied.json()["status"] == "applied"
    assert applied.json()["authority"] == "draft"
    assert applied.json()["revision"] == 1
    assert replayed.status_code == 200
    assert replayed.json()["revision"] == 1
    assert repository.count_mutations() == 3
    assert conflict.status_code == 409
    assert conflict.json()["conflict"]["reason"] == "revision"
    assert conflict.json()["objects"] == changed
    assert retired.status_code == 200
    assert retired.json()["retired"] is True
    assert retired.json()["authority"] == "committed"
    assert retired.json()["revision"] == 2
    assert repository.count_drafts(project_id="coco:train") == 0


def test_strict_shape_invalid_objects_and_empty_list_behavior(api) -> None:
    client, csrf, repository, _stores, _image_root = api
    initial = client.get("/api/splits/train/tasks/train:7/draft").json()
    body = {
        "mutation_id": "caller-path-forbidden",
        "expected_revision": 0,
        "expected_generation": 0,
        "expected_base_row_hash": initial["base_row_hash"],
        "objects": [],
        "source_path": "/tmp/not-allowed",
    }
    extra = client.put(
        "/api/splits/train/tasks/train:7/draft",
        json=body,
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    invalid = _put(
        client,
        csrf,
        mutation_id="degenerate",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=[{**_source_object("train", 101), "bbox_2d": [1, 1, 1, 2]}],
    )
    empty = _put(
        client,
        csrf,
        mutation_id="empty-is-data-layer-valid",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=[],
    )

    assert extra.status_code == 422
    assert invalid.status_code == 422
    assert repository.get_task_state("coco:train", "train:7").revision == 1
    assert empty.status_code == 200
    assert empty.json()["objects"] == []
    assert empty.json()["authority"] == "draft"


def test_foreign_positive_and_invented_negative_ids_fail_before_persistence(api) -> None:
    client, csrf, repository, _stores, _image_root = api
    initial = client.get("/api/splits/train/tasks/train:7/draft").json()
    foreign = _put(
        client,
        csrf,
        mutation_id="foreign-positive",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=[
            _source_object("train", 101),
            _source_object("train", 999),
        ],
    )
    invented = _put(
        client,
        csrf,
        mutation_id="invented-negative",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=[
            _source_object("train", 101),
            {**_local_object(), "coco_ann_id": -99},
        ],
    )

    assert foreign.status_code == 422
    assert invented.status_code == 422
    assert foreign.json()["error"] == {
        "code": "coco_refinement.task_bound_identity",
        "message": "Draft objects are invalid",
    }
    assert invented.json()["error"] == foreign.json()["error"]
    assert repository.get_task_state("coco:train", "train:7").revision == 0
    assert repository.count_mutations() == 0


@pytest.mark.parametrize(
    "raw_body",
    [
        (
            '{"mutation_id":"duplicate-top","mutation_id":"shadowed",'
            '"expected_revision":0,"expected_generation":0,'
            '"expected_base_row_hash":"BASE","objects":[]}'
        ),
        (
            '{"mutation_id":"duplicate-nested","expected_revision":0,'
            '"expected_generation":0,"expected_base_row_hash":"BASE",'
            '"objects":[{"region_key":"train:coco:101",'
            '"bbox_2d":[10,20,300,400],"category_name":"person",'
            '"category_id":1,"category_id":2,"coco_ann_id":101}]}'
        ),
        (
            '{"mutation_id":"nonstandard-number","expected_revision":0,'
            '"expected_generation":0,"expected_base_row_hash":"BASE",'
            '"objects":[{"region_key":"train:coco:101",'
            '"bbox_2d":[NaN,20,300,400],"category_name":"person",'
            '"category_id":1,"coco_ann_id":101}]}'
        ),
    ],
)
def test_duplicate_json_keys_and_nonstandard_constants_never_reach_repository(
    api, raw_body: str
) -> None:
    client, csrf, repository, _stores, _image_root = api
    base_row_hash = client.get(
        "/api/splits/train/tasks/train:7/draft"
    ).json()["base_row_hash"]
    response = client.put(
        "/api/splits/train/tasks/train:7/draft",
        content=raw_body.replace("BASE", base_row_hash).encode("utf-8"),
        headers={
            "Content-Type": "Application/JSON; Charset=UTF-8",
            "Origin": ORIGIN,
            CSRF_HEADER: csrf,
        },
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["error"]["code"] == "coco_refinement.request_invalid"
    assert repository.get_task_state("coco:train", "train:7").revision == 0
    assert repository.count_mutations() == 0


def test_bounded_double_read_retries_when_sqlite_changes_during_restore(api) -> None:
    client, _csrf, repository, stores, _image_root = api
    state = repository.get_task_state("coco:train", "train:7")
    draft = canonicalize_objects(
        [_source_object("train", 101), _local_object()], split="train"
    )

    def concurrent_save() -> None:
        repository.save_draft(
            SaveDraftRequest(
                project_id="coco:train",
                task_id="train:7",
                mutation_id="concurrent-save",
                expected_revision=state.revision,
                expected_generation=state.current_generation,
                expected_base_row_hash=state.base_row_hash,
                committed=canonicalize_objects(
                    [_source_object("train", 101)], split="train"
                ),
                draft=draft,
            )
        )

    stores["train"].on_restore = concurrent_save
    response = client.get("/api/splits/train/tasks/train:7/draft")

    assert response.status_code == 200
    assert response.json()["authority"] == "draft"
    assert response.json()["revision"] == 1
    assert stores["train"].restore_calls == 2


def test_post_save_authority_failure_is_replayable_with_same_mutation(api) -> None:
    client, csrf, repository, stores, _image_root = api
    initial = client.get("/api/splits/train/tasks/train:7/draft").json()
    stores["train"].restore_calls = 0
    stores["train"].mismatch_generation_after = 2
    changed = [_source_object("train", 101), _local_object()]

    lost = _put(
        client,
        csrf,
        mutation_id="post-save-response-lost",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=changed,
    )
    assert lost.status_code == 503
    assert repository.get_task_state("coco:train", "train:7").revision == 1

    stores["train"].mismatch_generation_after = None
    replayed = _put(
        client,
        csrf,
        mutation_id="post-save-response-lost",
        revision=0,
        base_row_hash=initial["base_row_hash"],
        objects=changed,
    )
    assert replayed.status_code == 200
    assert replayed.json()["revision"] == 1
    assert replayed.json()["objects"] == changed
    assert repository.count_mutations() == 1
