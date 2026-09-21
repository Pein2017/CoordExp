from __future__ import annotations

import json
import hashlib

pytest_plugins = ["test_commit_api"]

from test_commit_api import _post
from src.coco_refinement.adapters import SqliteTerminalReconciler
from src.label_studio_coco_refinement.store import BatchStatus, ValidationError


def emit(name: str, value: object) -> None:
    print(name, json.dumps(value, sort_keys=True))


def digest_entries(entries: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(entries):
        digest.update(name.encode() + b"\0" + hashlib.sha256(value).hexdigest().encode() + b"\n")
    return digest.hexdigest()


def test_one_invalid_member_is_all_or_nothing(commit_api, monkeypatch) -> None:
    client = commit_api["client"]
    csrf = commit_api["csrf"]
    store = commit_api["stores"]["train"]
    repository = commit_api["repository"]
    before_states = {
        task_id: repository.get_task_state("coco-refinement:train", task_id)
        for task_id in ("train:1", "train:3")
    }
    working_before = store.working_path.read_bytes()
    manifest_before = store.manifest_path.read_bytes()
    queued = _post(client, csrf, "probe-invalid-member")
    assert queued.status_code == 202 and queued.json()["member_count"] == 2

    original = store._validate_batch_frozen_request
    calls = 0

    def reject_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValidationError("probe invalid second member")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_validate_batch_frozen_request", reject_second)
    reconciler = SqliteTerminalReconciler(repository, current_user_id="local-operator")
    result = store.process_next_batch(
        terminal_observer=lambda request, terminal: reconciler.reconcile_batch(request, terminal)
    )
    after_states = {
        task_id: repository.get_task_state("coco-refinement:train", task_id)
        for task_id in ("train:1", "train:3")
    }
    sources_same = all(commit_api["source_paths"][s].read_bytes() == before for s, before in commit_api["source_before"].items())
    images_same = all(path.read_bytes() == before for path, before in commit_api["image_before"].items())
    source_before_hash = digest_entries([(split, value) for split, value in commit_api["source_before"].items()])
    source_after_hash = digest_entries([(split, commit_api["source_paths"][split].read_bytes()) for split in commit_api["source_before"]])
    image_before_hash = digest_entries([("/".join(path.parts[-2:]), value) for path, value in commit_api["image_before"].items()])
    image_after_hash = digest_entries([("/".join(path.parts[-2:]), path.read_bytes()) for path in commit_api["image_before"]])
    status = client.get("/api/splits/train/commits/probe-invalid-member")
    emit("invalid", {
        "queued": queued.status_code,
        "members": queued.json()["member_count"],
        "validation_calls": calls,
        "terminal": result.status.value if result else None,
        "terminal_generation": result.generation if result else None,
        "status_http": status.status_code,
        "status": status.json().get("status"),
        "working_same": store.working_path.read_bytes() == working_before,
        "manifest_same": store.manifest_path.read_bytes() == manifest_before,
        "draft1_same": after_states["train:1"] == before_states["train:1"],
        "draft3_same": after_states["train:3"] == before_states["train:3"],
        "draft_count": repository.count_drafts(project_id="coco-refinement:train"),
        "sources_same": sources_same,
        "images_same": images_same,
        "source_tree_before_sha256": source_before_hash,
        "source_tree_after_sha256": source_after_hash,
        "image_tree_before_sha256": image_before_hash,
        "image_tree_after_sha256": image_after_hash,
    })

    assert result is not None and result.status is BatchStatus.FAILED
    assert result.generation == 0
    assert store.working_path.read_bytes() == working_before
    assert store.manifest_path.read_bytes() == manifest_before
    assert after_states == before_states
    assert repository.count_drafts(project_id="coco-refinement:train") == 2
    assert status.status_code == 200 and status.json()["status"] == "failed"
    assert sources_same and images_same
