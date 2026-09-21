import hashlib
import json

import pytest

from src.artifacts.source_archive import SourceArchive


def test_explicit_historical_lookup_does_not_restore_or_accept_changed_bytes(tmp_path):
    old = tmp_path / "outputs" / "old.py"
    old.parent.mkdir()
    archived = tmp_path / "frozen.py"
    archived.write_bytes(b"frozen\n")
    sha = hashlib.sha256(archived.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "coordexp.output_source_archive.v1", "files": [
        {"source": str(old), "archive": str(archived), "sha256": sha}
    ]}))
    archive = SourceArchive(manifest)
    assert archive.resolve(old, sha)["path"] == str(archived)
    assert not old.exists()
    old.write_bytes(b"new implementation\n")
    assert archive.resolve(old, sha)["resolution"] == "archived_exact_path"
    assert old.read_bytes() == b"new implementation\n"
    archived.write_bytes(b"tampered\n")
    with pytest.raises(FileNotFoundError):
        archive.resolve(old, sha)
    with pytest.raises(ValueError):
        archive.verify_archive()


def test_expected_hash_is_mandatory_and_size_is_checked(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"x")
    sha = hashlib.sha256(b"x").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "coordexp.output_source_archive.v1", "files": []}))
    archive = SourceArchive(manifest)
    with pytest.raises(ValueError):
        archive.resolve(source, "unknown")
    with pytest.raises(ValueError, match="size"):
        archive.verify_bindings({"producer": {"path": str(source), "sha256": sha, "size_bytes": 99}})
