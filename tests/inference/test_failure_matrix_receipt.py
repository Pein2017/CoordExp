from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = (
    REPO_ROOT / "tests/inference/receipts/inference-failure-matrix.json"
)
EXPECTED_CASES = {
    "engine_startup",
    "cuda_oom",
    "worker_timeout",
    "process_tree_termination",
    "orphan_process",
    "malformed_shard",
    "backend_identity",
    "likelihood_mismatch",
}


@pytest.mark.parametrize("case", sorted(EXPECTED_CASES))
def test_failure_matrix_case_is_complete_historical_evidence(case: str) -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    rows = {row["case"]: row for row in receipt["cases"]}

    assert set(rows) == EXPECTED_CASES
    row = rows[case]
    assert row["terminal_diagnostics"] is True
    assert row["complete_cleanup"] is True
    assert row["canonical_publication"] is False
    assert row["tests"]

    for relative_path, recorded_sha256 in receipt["source_sha256"].items():
        assert (REPO_ROOT / relative_path).is_file()
        assert len(recorded_sha256) == 64
        int(recorded_sha256, 16)


def test_failure_matrix_digest_covers_complete_receipt() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    expected = receipt.pop("digest")
    observed = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert observed == expected
