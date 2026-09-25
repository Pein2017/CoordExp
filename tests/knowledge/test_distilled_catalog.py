import copy
import hashlib
import subprocess

import pytest

from scripts.check_research_knowledge import check_catalog, local_path


@pytest.fixture
def historical_catalog(tmp_path):
    root = tmp_path / "repo"; root.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
    git("init", "-q"); git("config", "user.name", "Fixture"); git("config", "user.email", "fixture@example.invalid")
    path = root / "research/experiments/old/results.md"; path.parent.mkdir(parents=True)
    path.write_text("# Result\nSelected-panel nonpass, no transfer claim.\n")
    git("add", "."); git("commit", "-qm", "Original")
    commit = git("rev-parse", "HEAD"); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.unlink(); path.parent.rmdir()
    question = root / "research/questions/q.md"; question.parent.mkdir(parents=True); question.write_text("# Q\n")
    row = {"schema_version": 2, "id": "old", "title": "Old", "tracking": "distilled", "topics": ["q"],
           "scientific_status": "closed", "evidence": "accepted", "boundary": "selected panel", "summary": "nonpass",
           "continuation": "unsupported_historical", "locator_status": "recorded_not_revalidated", "external_locators": [],
           "recovery": {"commit": commit, "record_root": "research/experiments/old", "reading_entry": "research/experiments/old/results.md",
                        "reading_sha256": digest, "protocols": [], "results": ["research/experiments/old/results.md"]}}
    return root, row


def test_distilled_history_needs_no_live_unit_directory(historical_catalog):
    root, row = historical_catalog
    assert not (root / row["recovery"]["record_root"]).exists()
    assert check_catalog(root, [row]) == []


@pytest.mark.parametrize("mutation", ["commit", "hash", "path", "topic", "continuation", "evidence", "duplicate"])
def test_missing_identity_or_false_continuation_is_rejected(historical_catalog, mutation):
    root, original = historical_catalog; row = copy.deepcopy(original)
    if mutation == "commit": row["recovery"]["commit"] = "0" * 40
    if mutation == "hash": row["recovery"]["reading_sha256"] = "0" * 64
    if mutation == "path": row["recovery"]["reading_entry"] = "../escape"
    if mutation == "topic": row["topics"] = ["missing"]
    if mutation == "continuation": row["continuation"] = "ready"
    if mutation == "evidence": row.pop("evidence")
    assert check_catalog(root, [row, row] if mutation == "duplicate" else [row])


def test_current_unit_cannot_use_distilled_exception(historical_catalog):
    root, row = historical_catalog; row["tracking"] = "current"
    assert check_catalog(root, [row])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\n"])
def test_path_rejection(tmp_path, name):
    with pytest.raises(ValueError): local_path(tmp_path, name)


@pytest.mark.parametrize("defect", [None, "accepted_without_result", "wrong_state_owner", "missing_protocol", "orphan"])
def test_live_unit_preserves_state_integrity(historical_catalog, defect):
    import json
    root, historic = historical_catalog
    record = root / "research/experiments/new"
    record.mkdir(parents=True)
    (record / "unit.md").write_text("# Current question\n")
    (record / "results.md").write_text("# Current result\n")
    state = {"unit_id": "new", "lifecycle": "closed", "evidence": "accepted",
             "boundary": "Explicit test scope", "not_authorized": [],
             "protocol": "research/experiments/new/unit.md",
             "state_source": "research/experiments/new/results.md",
             "result": "research/experiments/new/results.md"}
    row = {"schema_version": 2, "id": "new", "title": "Current", "tracking": "current", "topics": ["q"],
           "record_root": "research/experiments/new", "reading_entry": "research/experiments/new/unit.md",
           "state": "research/experiments/new/state.json"}
    if defect == "accepted_without_result": state["result"] = None
    if defect == "wrong_state_owner": state["unit_id"] = "other"
    if defect == "missing_protocol": (record / "unit.md").unlink()
    (record / "state.json").write_text(json.dumps(state))
    if defect == "orphan":
        orphan = root / "research/experiments/orphan/state.json"
        orphan.parent.mkdir()
        orphan.write_text(json.dumps(state))
    errors = check_catalog(root, [historic, row])
    assert bool(errors) == (defect is not None)
