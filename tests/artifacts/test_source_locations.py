import json
from pathlib import Path

import pytest

from src.artifacts.source_locations import INDEX, SourceLocations


def write_index(root: Path, entries: list[dict]) -> None:
    path = root / INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "coordexp.documentation_layout.v1", "files": entries}))


def test_relocated_path_routes_to_current_owner_without_byte_identity(tmp_path):
    source = "docs/history/old-note.md"
    target = "research/experiments/example/results.md"
    write_index(tmp_path, [{"source": source, "target": target}])
    assert SourceLocations(tmp_path).target(source) == target


@pytest.mark.parametrize("target", ["../outside.py", "/absolute.py"])
def test_location_index_rejects_path_escape(tmp_path, target):
    write_index(tmp_path, [{"source": "docs/old.py", "target": target}])
    with pytest.raises(ValueError, match="checkout"):
        SourceLocations(tmp_path)


def test_location_cycles_are_not_interpreted_as_recovery(tmp_path):
    write_index(tmp_path, [
        {"source": "docs/a.md", "target": "docs/b.md"},
        {"source": "docs/b.md", "target": "docs/a.md"},
    ])
    with pytest.raises(ValueError, match="cyclic"):
        SourceLocations(tmp_path).target("docs/a.md")
