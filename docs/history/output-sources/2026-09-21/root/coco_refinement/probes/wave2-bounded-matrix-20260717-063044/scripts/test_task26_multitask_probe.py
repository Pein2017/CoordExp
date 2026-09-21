from __future__ import annotations

import json
import threading
import hashlib

pytest_plugins = ["test_commit_api"]

from test_commit_api import ORIGIN, _PauseAt, _committed, _local, _post
from src.coco_refinement.adapters import SqliteTerminalReconciler
from src.coco_refinement.http_security import CSRF_HEADER


def emit(name: str, value: object) -> None:
    print(name, json.dumps(value, sort_keys=True))


def digest_entries(entries: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(entries):
        digest.update(name.encode() + b"\0" + hashlib.sha256(value).hexdigest().encode() + b"\n")
    return digest.hexdigest()


def put(client, csrf, task_id: str, mutation_id: str, current: dict, objects: list[dict]):
    return client.put(
        f"/api/splits/train/tasks/{task_id}/draft",
        json={
            "mutation_id": mutation_id,
            "expected_revision": current["revision"],
            "expected_generation": current["generation"],
            "expected_base_row_hash": current["base_row_hash"],
            "objects": objects,
        },
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )


def test_captured_and_uncaptured_later_edits(commit_api) -> None:
    client = commit_api["client"]
    csrf = commit_api["csrf"]
    store = commit_api["stores"]["train"]
    repository = commit_api["repository"]

    task3 = client.get("/api/splits/train/tasks/train:3/draft").json()
    retired3 = put(
        client,
        csrf,
        "train:3",
        "probe-retire-task3-before-capture",
        task3,
        _committed("train", 3).to_json_regions(),
    )
    assert retired3.status_code == 200 and retired3.json()["retired"] is True

    queued = _post(client, csrf, "probe-task26-multitask")
    assert queued.status_code == 202 and queued.json()["member_count"] == 1
    pause = _PauseAt("batch_working_temp_fsynced")
    store._fault_injector = pause
    reconciler = SqliteTerminalReconciler(repository, current_user_id="local-operator")
    outcomes = []
    errors = []

    def process() -> None:
        try:
            outcomes.append(
                store.process_next_batch(
                    terminal_observer=lambda request, result: reconciler.reconcile_batch(request, result)
                )
            )
        except BaseException as exc:
            errors.append(repr(exc))

    worker = threading.Thread(target=process)
    worker.start()
    assert pause.entered.wait(1.0)

    captured = client.get("/api/splits/train/tasks/train:1/draft").json()
    captured_objects = captured["objects"]
    captured_objects[-1]["bbox_2d"] = [141, 142, 741, 842]
    saved_captured = put(
        client,
        csrf,
        "train:1",
        "probe-later-captured",
        captured,
        captured_objects,
    )

    uncaptured = client.get("/api/splits/train/tasks/train:3/draft").json()
    uncaptured_objects = [
        *uncaptured["objects"],
        _local("local:3f5dd17d-46ee-43dd-9fc0-51a5fd603940", x1=55),
    ]
    saved_uncaptured = put(
        client,
        csrf,
        "train:3",
        "probe-later-uncaptured",
        uncaptured,
        uncaptured_objects,
    )
    emit("paused", {
        "queued": queued.status_code,
        "members": queued.json()["member_count"],
        "captured_put": saved_captured.status_code,
        "captured_revision": saved_captured.json().get("revision"),
        "uncaptured_put": saved_uncaptured.status_code,
        "uncaptured_revision": saved_uncaptured.json().get("revision"),
        "status": client.get("/api/splits/train/commits/probe-task26-multitask").json()["status"],
    })

    pause.release.set()
    worker.join(3.0)
    status = client.get("/api/splits/train/commits/probe-task26-multitask")
    final1 = client.get("/api/splits/train/tasks/train:1/draft")
    final3 = client.get("/api/splits/train/tasks/train:3/draft")
    sources_same = all(commit_api["source_paths"][s].read_bytes() == before for s, before in commit_api["source_before"].items())
    images_same = all(path.read_bytes() == before for path, before in commit_api["image_before"].items())
    source_before_hash = digest_entries([(split, value) for split, value in commit_api["source_before"].items()])
    source_after_hash = digest_entries([(split, commit_api["source_paths"][split].read_bytes()) for split in commit_api["source_before"]])
    image_before_hash = digest_entries([("/".join(path.parts[-2:]), value) for path, value in commit_api["image_before"].items()])
    image_after_hash = digest_entries([("/".join(path.parts[-2:]), path.read_bytes()) for path in commit_api["image_before"]])
    emit("final", {
        "worker_alive": worker.is_alive(),
        "errors": errors,
        "outcomes": len(outcomes),
        "status": status.json().get("status"),
        "captured_generation": final1.json().get("generation"),
        "captured_authority": final1.json().get("authority"),
        "captured_bbox": final1.json().get("objects", [{}])[-1].get("bbox_2d"),
        "captured_id": final1.json().get("objects", [{}])[-1].get("coco_ann_id"),
        "uncaptured_generation": final3.json().get("generation"),
        "uncaptured_authority": final3.json().get("authority"),
        "uncaptured_bbox": final3.json().get("objects", [{}])[-1].get("bbox_2d"),
        "uncaptured_has_id": "coco_ann_id" in final3.json().get("objects", [{}])[-1],
        "sources_same": sources_same,
        "images_same": images_same,
        "source_tree_before_sha256": source_before_hash,
        "source_tree_after_sha256": source_after_hash,
        "image_tree_before_sha256": image_before_hash,
        "image_tree_after_sha256": image_after_hash,
    })

    assert saved_captured.status_code == saved_uncaptured.status_code == 200
    assert not worker.is_alive() and not errors and len(outcomes) == 1
    assert status.json()["status"] == "succeeded"
    assert final1.json()["generation"] == final3.json()["generation"] == 1
    assert final1.json()["authority"] == final3.json()["authority"] == "draft"
    assert final1.json()["objects"][-1]["bbox_2d"] == [141, 142, 741, 842]
    assert final1.json()["objects"][-1]["coco_ann_id"] < 0
    assert final3.json()["objects"][-1]["bbox_2d"] == [55, 40, 500, 600]
    assert "coco_ann_id" not in final3.json()["objects"][-1]
    assert sources_same and images_same
