"""Decision metadata never substitutes for measured qualification receipts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adapters.source_gates import _dora_source_study_is_passed
from src.qwen.special_token_embeddings import _special_token_embedding_source_study_is_passed

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("dora-source-study.json", _dora_source_study_is_passed),
    ("selected-embedding-source-study.json", _special_token_embedding_source_study_is_passed),
]


@pytest.mark.parametrize("filename,reader", CASES)
def test_source_decision_requires_explicit_schema_and_mechanism(tmp_path, filename, reader):
    path = tmp_path / filename
    original = json.loads((ROOT / "manifests/qualification" / filename).read_text())
    path.write_text(json.dumps(original))
    assert reader(path)
    for key, value in [
        ("schema_version", True), ("schema_version", 2),
        ("evidence_kind", "measured_qualification"), ("component", "other"),
        ("mechanism", "other"), ("decision", "rejected"),
        ("roundtrip_receipt_required", False),
    ]:
        path.write_text(json.dumps({**original, key: value}))
        assert not reader(path), key
    for invalid in ("not json", "null", "[]", "{}"):
        path.write_text(invalid)
        assert not reader(path)
    path.unlink()
    assert not reader(path)
