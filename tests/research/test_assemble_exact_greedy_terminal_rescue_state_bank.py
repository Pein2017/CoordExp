from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.assemble_exact_greedy_terminal_rescue_state_bank import (
    AssemblyError,
    DEFAULT_REFERENCE_MANIFEST,
    merge_collector_shards,
)
from src.config.fingerprint import sha256_file
from src.rollout_calibration import load_state_bank_manifest_binding


def _write_collector(
    root: Path,
    *,
    reference: Path,
    event_ids: list[str],
    selected_count: int | None = None,
    accepted_count: int | None = None,
    review_event_ids: list[str] | None = None,
) -> None:
    root.mkdir()
    rollout_rows = [{"event_id": event_id, "candidates": []} for event_id in event_ids]
    review_ids = event_ids if review_event_ids is None else review_event_ids
    review_rows = [{"event_id": event_id, "admission_status": "rejected", "rejection_reason": "unit"} for event_id in review_ids]
    (root / "rollout_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rollout_rows),
        encoding="utf-8",
    )
    (root / "review_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    source = root / "source.txt"
    source.write_text("source evidence\n", encoding="utf-8")
    binding = load_state_bank_manifest_binding(reference)
    receipt = {
        "schema_version": "exact_greedy_terminal_rescue_collector.v1",
        "selected_record_count": len(event_ids) if selected_count is None else selected_count,
        "source_artifacts": {
            "unit-source": {"path": str(source), "sha256": sha256_file(source)},
        },
        "runtime": {
            "status": "completed",
            "accepted_event_count": len(event_ids) if accepted_count is None else accepted_count,
            "checkpoint_id": binding.source_checkpoint_id,
            "reference_state_bank_manifest": str(reference),
            "reference_state_bank_manifest_sha256": sha256_file(reference),
        },
    }
    (root / "collection-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_merge_is_sorted_and_accepts_selected_records_above_accepted_events(tmp_path: Path) -> None:
    reference = Path(DEFAULT_REFERENCE_MANIFEST)
    first = tmp_path / "shard-a"
    second = tmp_path / "shard-b"
    _write_collector(first, reference=reference, event_ids=["event-b"], selected_count=3)
    _write_collector(second, reference=reference, event_ids=["event-a"])

    rollouts, reviews, artifacts, binding = merge_collector_shards(
        [second, first], reference_manifest=reference
    )

    assert [row["event_id"] for row in rollouts] == ["event-a", "event-b"]
    assert [row["event_id"] for row in reviews] == ["event-a", "event-b"]
    assert binding.source_checkpoint_id
    assert len(artifacts) == len({item["artifact_id"] for item in artifacts})
    assert len(artifacts) >= 2 * 4


def test_merge_rejects_runtime_accepted_count_mismatch(tmp_path: Path) -> None:
    reference = Path(DEFAULT_REFERENCE_MANIFEST)
    root = tmp_path / "shard"
    _write_collector(root, reference=reference, event_ids=["event-a"], accepted_count=2)
    with pytest.raises(AssemblyError, match="accepted_event_count"):
        merge_collector_shards([root], reference_manifest=reference)


def test_merge_rejects_duplicate_review_ids(tmp_path: Path) -> None:
    reference = Path(DEFAULT_REFERENCE_MANIFEST)
    root = tmp_path / "shard"
    _write_collector(
        root,
        reference=reference,
        event_ids=["event-a", "event-b"],
        review_event_ids=["event-a", "event-a"],
    )
    with pytest.raises(AssemblyError, match="duplicate event_id inside review"):
        merge_collector_shards([root], reference_manifest=reference)


def test_merge_rejects_source_checksum_mismatch(tmp_path: Path) -> None:
    reference = Path(DEFAULT_REFERENCE_MANIFEST)
    root = tmp_path / "shard"
    _write_collector(root, reference=reference, event_ids=["event-a"])
    receipt_path = root / "collection-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source_artifacts"]["unit-source"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AssemblyError, match="checksum mismatch"):
        merge_collector_shards([root], reference_manifest=reference)
