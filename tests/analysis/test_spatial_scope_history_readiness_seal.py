from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.research.materialize_spatial_scope_history_schedule import (
    FROZEN_READINESS_ARTIFACT_DIGESTS,
    FROZEN_READINESS_CATEGORY_NAMESPACE_SHA256,
    FROZEN_READINESS_LEDGER_SEAL_SHA256,
    FROZEN_READINESS_SOURCE_DIGESTS,
    _validate_readiness_root,
)


def test_readiness_v2_resealed_reviewer_change_is_rejected(tmp_path: Path) -> None:
    """A self-consistent replacement seal is not an authorized readiness root."""

    reviewer_labels = b"reviewer-one-labels-jsonl altered after adjudication\n"
    (tmp_path / "reviewer-one-labels.jsonl").write_bytes(reviewer_labels)
    altered_source_digests = dict(FROZEN_READINESS_SOURCE_DIGESTS)
    altered_source_digests["reviewer_one_labels_jsonl"] = hashlib.sha256(
        reviewer_labels
    ).hexdigest()
    # This seal is internally consistent with the altered reviewer-label
    # source, but it is not the frozen readiness-v2 seal.
    seal = {
        "schema_version": "dense-union-51.final-ledger-seal.v1",
        "artifact_digests": FROZEN_READINESS_ARTIFACT_DIGESTS,
        "source_digests": altered_source_digests,
        "category_namespace_sha256": FROZEN_READINESS_CATEGORY_NAMESPACE_SHA256,
    }
    seal_path = tmp_path / "ledger-seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n")
    assert seal_path.read_bytes()
    assert FROZEN_READINESS_LEDGER_SEAL_SHA256 != hashlib.sha256(
        seal_path.read_bytes()
    ).hexdigest()

    with pytest.raises(RuntimeError, match="authorized readiness-v2 seal"):
        _validate_readiness_root(tmp_path)
