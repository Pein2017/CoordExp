import hashlib
import json
from pathlib import Path

import pytest

from src.artifacts.source_locations import INDEX, SourceLocations


def write_index(root: Path, entries: list[dict]) -> None:
    path = root / INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "coordexp.documentation_layout.v1", "files": entries}))


def test_relocated_source_is_exact_and_does_not_restore_old_docs(tmp_path):
    source = "docs/history/source.py"
    target = "reference/retained-sources/old.py"
    payload = b"original = 7\n"
    sha = hashlib.sha256(payload).hexdigest()
    path = tmp_path / target
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    write_index(tmp_path, [{"source": source, "target": target, "sha256": sha}])
    locations = SourceLocations(tmp_path)
    assert locations.target(source) == target
    assert locations.materialized(tmp_path / source, sha) == path
    assert locations.original_bytes(source, sha) == payload
    assert not (tmp_path / source).exists()
    assert locations.materialized(tmp_path / source, "0" * 64) is None
    path.write_bytes(b"changed")
    with pytest.raises(FileNotFoundError):
        locations.original_bytes(source, sha)


@pytest.mark.parametrize("target", ["../outside.py", "/absolute.py"])
def test_location_index_rejects_path_escape(tmp_path, target):
    write_index(tmp_path, [{"source": "docs/old.py", "target": target, "sha256": "a" * 64}])
    with pytest.raises(ValueError, match="checkout"):
        SourceLocations(tmp_path)


def test_location_cycles_are_not_interpreted_as_recovery(tmp_path):
    write_index(tmp_path, [
        {"source": "docs/a.md", "target": "docs/b.md", "sha256": "a" * 64},
        {"source": "docs/b.md", "target": "docs/a.md", "sha256": "b" * 64},
    ])
    with pytest.raises(ValueError, match="cyclic"):
        SourceLocations(tmp_path).target("docs/a.md")


def test_research_reader_verifies_original_master_manifest_after_local_moves():
    from src.artifacts.source_archive import SourceArchive

    master = Path("/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json")
    if not master.is_file():
        pytest.skip("retained deployment source manifest is unavailable")
    assert SourceArchive(master).verify_archive()["verified_files"] == 10877
