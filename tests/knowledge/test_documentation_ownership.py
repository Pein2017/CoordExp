import importlib.util
import json
from pathlib import Path

import pytest

from scripts.tools import research_knowledge as CHECK


ROOT = Path(__file__).resolve().parents[2]





@pytest.mark.parametrize("name", ["producer.py", "launch.sh", "run.yaml"])
def test_docs_rejects_code_including_frozen_source_directories(tmp_path, name):
    for root_name in CHECK.ROOT_NAMES:
        path = tmp_path / "research" / root_name
        if root_name.endswith(".md"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Fixture\n")
        else:
            path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "docs/history/run-sources" / name
    target.parent.mkdir(parents=True)
    target.write_text("pass\n")
    assert any("code/cache in docs" in error for error in CHECK.check_layout(tmp_path))


def test_catalogued_scientific_records_live_with_research_not_history():
    rows = [json.loads(line) for line in (ROOT / "research/experiments/catalog.jsonl").read_text().splitlines() if line]
    assert all(row["record_root"].startswith("research/experiments/") for row in rows)
    assert (ROOT / "research/assets.md").is_file()
    assert not (ROOT / "docs/RESEARCH_ASSETS.md").exists()


def test_all_original_duplication_notes_are_owned_by_the_declared_synthesis():
    import csv
    from scripts.tools.document_locations import DocumentLocations

    locations = DocumentLocations(ROOT)
    with (ROOT / "manifests/duplication-source-lineage.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 223
    for row in rows:
        old = "docs/history/research-intake/2026-07-01-autoregressive-binding-template-study/" + row["snapshot_path"]
        current = locations.target(old)
        unit = Path(row["absorbed_into"]).stem
        assert current.startswith(f"research/experiments/{unit}/supporting/")
        assert (ROOT / current).is_file()
