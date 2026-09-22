from pathlib import Path

import pytest

from src.artifacts import source_provenance as sources


def test_default_source_capture_belongs_to_reference_not_documentation():
    checkout = Path(sources.__file__).resolve().parents[2]
    assert sources.SOURCE_ARCHIVE_ROOT == checkout / "reference/retained-sources/runs"
    assert not sources.SOURCE_ARCHIVE_ROOT.is_relative_to(checkout / "docs")


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


def test_cross_run_captures_share_readonly_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "SOURCE_ARCHIVE_ROOT", tmp_path / "archive" / "runs")
    original = tmp_path / "producer.py"
    original.write_bytes(b"same bytes\n")
    first = sources.preserve_source(original, run_root=tmp_path / "a", relative_name="a.py")
    second = sources.preserve_source(original, run_root=tmp_path / "b", relative_name="b.py")
    assert first != second
    assert first.samefile(second)
    assert first.stat().st_mode & 0o222 == 0
    original.write_bytes(b"changed live source\n")
    assert first.read_bytes() == b"same bytes\n"


def test_capture_publishes_only_verified_complete_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "SOURCE_ARCHIVE_ROOT", tmp_path / "archive" / "runs")
    original = tmp_path / "producer.py"
    original.write_bytes(b"source\n")
    target = sources.source_snapshot_path(tmp_path / "a", "a.py")

    def broken_copy(handle, output, **kwargs):
        output.write(b"partial")
        output.flush()
        assert not target.exists(), "published an incomplete capture"
        raise OSError("interrupted copy")

    monkeypatch.setattr(sources.shutil, "copyfileobj", broken_copy)
    with pytest.raises(OSError, match="interrupted"):
        sources.preserve_source(original, run_root=tmp_path / "a", relative_name="a.py")
    assert not target.exists()
    assert not [p for p in (tmp_path / "archive").rglob("*") if p.is_file()]


@pytest.mark.parametrize("corruption", ["bytes", "symlink"])
def test_invalid_object_never_binds_another_run(tmp_path, monkeypatch, corruption):
    archive = tmp_path / "archive" / "runs"
    monkeypatch.setattr(sources, "SOURCE_ARCHIVE_ROOT", archive)
    original = tmp_path / "producer.py"
    original.write_bytes(b"source\n")
    digest = sources._sha256(original)
    obj = archive.parent / "objects" / digest[:2] / (digest + ".py")
    obj.parent.mkdir(parents=True)
    if corruption == "bytes":
        obj.write_bytes(b"corrupt")
    else:
        obj.symlink_to(original)
    with pytest.raises(ValueError, match="differs"):
        sources.preserve_source(original, run_root=tmp_path / "a", relative_name="a.py")
    assert not sources.source_snapshot_path(tmp_path / "a", "a.py").exists()


def test_concurrent_captures_publish_one_complete_object(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(sources, "SOURCE_ARCHIVE_ROOT", tmp_path / "archive" / "runs")
    original = tmp_path / "producer.py"
    original.write_bytes(b"source\n" * 1024)

    def capture(i):
        return sources.preserve_source(original, run_root=tmp_path / str(i % 3), relative_name="a.py")

    with ThreadPoolExecutor(max_workers=6) as pool:
        captures = list(pool.map(capture, range(12)))
    assert all(p.read_bytes() == original.read_bytes() for p in captures)
    assert all(p.samefile(captures[0]) for p in captures)
    objects = [p for p in (tmp_path / "archive" / "objects").rglob("*") if p.is_file()]
    assert len(objects) == 1


def test_source_mutation_during_capture_leaves_no_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "SOURCE_ARCHIVE_ROOT", tmp_path / "archive" / "runs")
    original = tmp_path / "producer.py"
    original.write_bytes(b"source\n")
    copy = sources.shutil.copyfileobj

    def changing_copy(handle, output, **kwargs):
        copy(handle, output, **kwargs)
        original.write_bytes(b"changed source\n")

    monkeypatch.setattr(sources.shutil, "copyfileobj", changing_copy)
    with pytest.raises(ValueError, match="source changed"):
        sources.preserve_source(original, run_root=tmp_path / "a", relative_name="a.py")
    assert not [p for p in (tmp_path / "archive").rglob("*") if p.is_file()]
