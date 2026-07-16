from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from src.common.errors import DataContractError
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "label_studio_coco_refinement"


def test_frozen_registry_exhaustively_matches_official_sparse_fixture() -> None:
    expected = json.loads((FIXTURE_ROOT / "coco80_sparse_categories.json").read_text())
    contract = _source_contract()
    registry_source = REPO_ROOT / contract["registry"]["source"]
    actual = [
        {"id": category.id, "name": category.name}
        for category in COCO80_REGISTRY.categories
    ]

    assert actual == expected
    assert json.loads(registry_source.read_text()) == expected
    assert hashlib.sha256(registry_source.read_bytes()).hexdigest() == contract["registry"]["source_sha256"]
    assert len(actual) == 80
    assert COCO80_REGISTRY.ids[-1] == 90
    assert COCO80_REGISTRY.fingerprint == _source_contract()["registry"]["fingerprint"]
    for category in COCO80_REGISTRY.categories:
        assert COCO80_REGISTRY.by_name(category.name) is category
        assert COCO80_REGISTRY.by_id(category.id) is category
        assert COCO80_REGISTRY.validate(category.name, category.id) is category


@pytest.mark.parametrize(
    ("name", "category_id", "code"),
    [
        ("traffic light", 10, None),
        ("traffic light", 9, "label_studio.category_mismatch"),
        ("stop sign", 12, "label_studio.category_mismatch"),
        ("Traffic Light", 10, "label_studio.category_name"),
        (" traffic light ", 10, "label_studio.category_name"),
        ("motorbike", 4, "label_studio.category_name"),
    ],
)
def test_registry_accepts_only_exact_canonical_name_sparse_id_pairs(
    name: str,
    category_id: int,
    code: str | None,
) -> None:
    if code is None:
        assert COCO80_REGISTRY.validate(name, category_id).name == name
        return
    with pytest.raises(DataContractError) as exc_info:
        COCO80_REGISTRY.validate(name, category_id)
    assert exc_info.value.code == code


def test_source_contract_receipts_match_exact_selected_sources() -> None:
    contract = _source_contract()
    assert (REPO_ROOT / contract["image_root"]).is_dir()
    expected_row_fields = set(contract["row_fields"])
    expected_object_fields = set(contract["object_fields"])

    for split, expected in contract["sources"].items():
        source = REPO_ROOT / expected["path"]
        digest = hashlib.sha256()
        rows = boxes = 0
        mapping: dict[str, int] = {}
        representative_raw: bytes | None = None
        with source.open("rb") as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                payload = json.loads(raw_line)
                assert set(payload) == expected_row_fields
                rows += 1
                boxes += len(payload["objects"])
                for obj in payload["objects"]:
                    assert set(obj) == expected_object_fields
                    mapping[obj["category_name"]] = obj["category_id"]
                if row_number == expected["representative_row_number"]:
                    representative_raw = raw_line.rstrip(b"\r\n")

        assert digest.hexdigest() == expected["sha256"]
        assert rows == expected["row_count"]
        assert boxes == expected["box_count"]
        assert mapping == {
            category.name: category.id for category in COCO80_REGISTRY.categories
        }
        assert representative_raw is not None
        assert hashlib.sha256(representative_raw).hexdigest() == expected["representative_row_sha256"]
        fixture_raw = (FIXTURE_ROOT / expected["representative_fixture"]).read_bytes().rstrip(b"\r\n")
        assert fixture_raw == representative_raw


def test_fixture_pins_current_label_studio_revision_and_version() -> None:
    expected = _source_contract()["label_studio"]
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT / "label-studio"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pyproject = tomllib.loads((REPO_ROOT / "label-studio" / "pyproject.toml").read_text())

    assert revision == expected["git_revision"]
    assert pyproject["project"]["version"] == expected["version"]


def _source_contract() -> dict:
    return json.loads((FIXTURE_ROOT / "source_contract.json").read_text())
