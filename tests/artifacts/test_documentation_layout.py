from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_documentation import check_docs, declared_docs, load_seal, read_extras, read_original

ROOT = Path(__file__).resolve().parents[2]


def test_current_documentation_is_a_small_valid_surface():
    assert check_docs(ROOT, load_seal(ROOT)) == []


def _surface(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/README.md").write_text("# Durable assets\n")
    return {"live_docs": ["docs/README.md"]}


def test_new_history_does_not_silently_enter_the_live_surface(tmp_path):
    seal = _surface(tmp_path)
    (tmp_path / "docs/history").mkdir()
    (tmp_path / "docs/history/new-plan.md").write_text("not an admitted asset")
    assert "unexpected doc: docs/history/new-plan.md" in check_docs(tmp_path, seal)


def test_broken_document_link_is_rejected(tmp_path):
    seal = _surface(tmp_path)
    (tmp_path / "docs/README.md").write_text("[missing](missing.md)\n")
    assert any("broken local link" in s for s in check_docs(tmp_path, seal))


def test_runtime_prose_dependency_is_rejected(tmp_path):
    seal = _surface(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/gate.py").write_text('decision = "docs/old-gate.md"\n')
    assert any("runtime documentation dependency" in s for s in check_docs(tmp_path, seal))


def test_original_reader_rejects_paths_outside_the_sealed_docs(tmp_path):
    with pytest.raises(ValueError, match="original docs"):
        read_original(tmp_path, {}, "../AGENTS.md")


def test_sealed_extras_detect_tampering(tmp_path):
    path = tmp_path / "manifests/documentation/extras.json"
    path.parent.mkdir(parents=True)
    raw = b"original evidence\n"
    payload = {"schema_version": 1, "status": "sealed", "entries": [{
        "path": "docs/history/evidence.md", "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "content_utf8": raw.decode(),
    }]}
    data = json.dumps(payload).encode()
    path.write_bytes(data)
    seal = {"untracked_file_count": 1, "sealed_extras": {
        "path": "manifests/documentation/extras.json", "count": 1,
        "sha256": hashlib.sha256(data).hexdigest(),
    }}
    assert read_extras(tmp_path, seal)["docs/history/evidence.md"] == raw
    path.write_bytes(data + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_extras(tmp_path, seal)


def test_readme_is_the_single_live_inventory(tmp_path):
    _surface(tmp_path)
    (tmp_path / "docs/README.md").write_text("[asset](asset.md)\n")
    (tmp_path / "docs/asset.md").write_text("# Lasting decision\n")
    assert declared_docs(tmp_path) == ["docs/README.md", "docs/asset.md"]


def test_linking_a_retired_tree_cannot_reopen_it(tmp_path):
    _surface(tmp_path)
    (tmp_path / "docs/README.md").write_text("[archive](history/new.md)\n")
    with pytest.raises(ValueError, match="cannot be readmitted"):
        declared_docs(tmp_path)
