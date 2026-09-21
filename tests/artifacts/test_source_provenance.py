from pathlib import Path

import pytest

from src.artifacts import source_provenance as sources


def test_source_capture_is_external_exact_and_collision_checked(tmp_path, monkeypatch):
    archive = tmp_path / "history"
    monkeypatch.setattr(sources, "SOURCE_ARCHIVE_ROOT", archive)
    run = tmp_path / "outputs" / "run-1"
    run.mkdir(parents=True)
    original = tmp_path / "producer.py"
    original.write_bytes(b"original source\n")
    target = sources.preserve_source(original, run_root=run, relative_name="code/producer.py")
    assert target.is_relative_to(archive)
    assert not target.is_relative_to(run)
    assert target.read_bytes() == original.read_bytes()
    assert sources.preserve_source(original, run_root=run, relative_name="code/producer.py") == target
    original.write_bytes(b"different source\n")
    with pytest.raises(ValueError, match="differs"):
        sources.preserve_source(original, run_root=run, relative_name="code/producer.py")
    assert target.read_bytes() == b"original source\n"


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", ""])
def test_invalid_source_name_fails_before_writing(tmp_path, name):
    with pytest.raises(ValueError):
        sources.source_snapshot_path(tmp_path / "run", name)


def test_snapshot_identity_distinguishes_run_roots(tmp_path):
    assert sources.source_snapshot_path(tmp_path / "a", "file.py") != sources.source_snapshot_path(
        tmp_path / "b", "file.py"
    )
