from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest
import torch

import src.artifacts.evidence_journal as journal_module
from src.artifacts.evidence_journal import ExecutionEvidenceJournal, JOURNAL_SCHEMA_VERSION
from src.artifacts.json_values import (
    LocalFileSystemOps,
    canonical_json_bytes,
    json_sha256,
    load_canonical_json,
    validate_json_value,
)
from src.common.errors import ArtifactContractError


def test_strict_values_are_canonical_and_reject_live_values(tmp_path: Path) -> None:
    receipt = {"nested": [None, {"count": 2, "score": 0.5}], "name": "ok"}
    encoded = canonical_json_bytes(receipt)
    path = tmp_path / "receipt.json"
    path.write_bytes(encoded)

    assert load_canonical_json(path) == receipt
    assert json_sha256(receipt) == json_sha256(load_canonical_json(path))
    assert canonical_json_bytes(MappingProxyType(receipt)) == encoded
    for invalid in (
        lambda: None,
        Path("receipt"),
        {"values": {1, 2}},
        {1: "not-a-string-key"},
        {"value": math.nan},
        {"value": math.inf},
        {"value": -math.inf},
    ):
        with pytest.raises(ArtifactContractError) as exc_info:
            validate_json_value(invalid)
        assert exc_info.value.code == "artifact.invalid_json_value"
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize(
    "invalid_context",
    [
        {"callback": lambda: None},
        {"tensor": torch.tensor([1.0])},
    ],
)
def test_journal_create_rejects_live_values_before_creating_any_root(
    tmp_path: Path, invalid_context: dict[str, object]
) -> None:
    root = tmp_path / "journal"
    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.create(
            root=root,
            execution_id="execution-a",
            execution_identity={"source": "a"},
            expected_work_item_ids=["first"],
            context=invalid_context,
        )
    assert exc_info.value.code == "artifact.invalid_json_value"
    assert not root.exists()


def test_plan_is_immutable_and_uses_durable_exclusive_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    snapshot = journal.reload()

    assert snapshot.completed_work_item_ids == ()
    assert snapshot.temporary_paths == ()
    plan = load_canonical_json(root / "plan.json")
    assert plan["execution_identity_fingerprint"] == json_sha256({"source": "a"})
    assert plan["expected_work_item_ids"] == ["first", "second"]
    assert journal.plan_file_sha256 == _sha256(root / "plan.json")
    journal.close()

    with pytest.raises(ArtifactContractError) as exc_info:
        _create(root)
    assert exc_info.value.code == "journal.root_already_exists"


def test_open_missing_root_fails_with_typed_journal_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactContractError) as exc_info:
        _open(tmp_path / "missing")

    assert exc_info.value.code == "journal.root_missing"


def test_invalid_or_duplicate_planned_identities_fail_typed_before_root_creation(
    tmp_path: Path,
) -> None:
    for name, expected_ids, expected_code in (
        ("not-a-sequence", None, "journal.invalid_plan"),
        ("duplicate", ["first", "first"], "journal.duplicate_work_item"),
    ):
        root = tmp_path / name
        with pytest.raises(ArtifactContractError) as exc_info:
            ExecutionEvidenceJournal.create(
                root=root,
                execution_id="execution-a",
                execution_identity={"source": "a"},
                expected_work_item_ids=cast(Any, expected_ids),
                context={"mode": "cpu"},
            )
        assert exc_info.value.code == expected_code
        assert not root.exists()


@pytest.mark.parametrize("failure", ["file", "publish", "directory"])
def test_plan_publication_failures_never_replace_existing_final_path(
    tmp_path: Path, failure: str
) -> None:
    root = tmp_path / failure
    with pytest.raises(ArtifactContractError) as exc_info:
        _create(root, filesystem=_FailingFileSystemOps(failure))
    assert exc_info.value.code == (
        "journal.prepare_failed"
        if failure == "directory"
        else "artifact.publish_failed"
    )
    assert not (root / "plan.json").exists()
    assert (root / "records").is_dir()
    assert (root / "attempts").is_dir()


def test_post_publication_directory_sync_failure_is_reloadable_only_after_preparation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    with pytest.raises(ArtifactContractError) as exc_info:
        _create(root, filesystem=_FailOnDirectorySyncCall(2))
    assert exc_info.value.code == "artifact.publish_failed"
    assert (root / "plan.json").is_file()
    assert (root / "records").is_dir()
    assert (root / "attempts").is_dir()
    assert ExecutionEvidenceJournal.inspect(root).completed_work_item_ids == ()


def test_records_remain_durable_across_bad_later_payload_and_attempt_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    first_attempt = journal.start_attempt()
    first_path = journal.append_record(
        work_item_id="first", payload={"value": 1}, attempt_id=first_attempt
    )
    before = first_path.read_bytes()

    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="first",
            payload={"value": "replacement"},
            attempt_id=first_attempt,
        )
    assert exc_info.value.code == "journal.work_item_already_completed"
    assert first_path.read_bytes() == before

    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="second", payload={"bad": object()}, attempt_id=first_attempt
        )
    assert exc_info.value.code == "artifact.invalid_json_value"
    assert first_path.read_bytes() == before
    assert journal.completed_work_item_ids == ("first",)
    journal.record_attempt_failure(
        attempt_id=first_attempt,
        failure_code="interrupted",
        failure_message="synthetic interruption",
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="second",
            payload={"value": "late"},
            attempt_id=first_attempt,
        )
    assert exc_info.value.code == "journal.attempt_already_closed"
    journal.close()

    continued = _open(root)
    second_attempt = continued.start_attempt()
    assert second_attempt != first_attempt
    continued.append_record(
        work_item_id="second", payload={"value": 2}, attempt_id=second_attempt
    )
    terminal = continued.finalize()
    assert load_canonical_json(terminal)["status"] == "completed"
    assert ExecutionEvidenceJournal.inspect(root).completed_work_item_ids == (
        "first",
        "second",
    )
    continued.close()


def test_attempt_directory_prepare_failure_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    attempt_id = "fixed-attempt-id"
    (root / "attempts" / attempt_id).mkdir()

    class _FixedUuid:
        hex = attempt_id

    monkeypatch.setattr(journal_module.uuid, "uuid4", lambda: _FixedUuid())
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.start_attempt()

    assert exc_info.value.code == "journal.attempt_prepare_failed"
    assert not (root / "attempts" / attempt_id / "start.json").exists()
    journal.close()


def test_research_shaped_values_round_trip_opaquely_while_terminal_stays_mechanical(
    tmp_path: Path,
) -> None:
    research_context = {
        "arm": "masked",
        "cohort": ["case-a"],
        "conditioning": {"opener": "natural"},
        "claim": None,
        "stop_rule": {"kind": "caller-owned"},
    }
    research_payload = {
        "unmatched": True,
        "threshold": 0.25,
        "observation": {"tokens": [1, 2]},
    }
    root = tmp_path / "opaque-research-values"
    journal = ExecutionEvidenceJournal.create(
        root=root,
        execution_id="execution-opaque",
        execution_identity={"source": "caller"},
        expected_work_item_ids=["case-a"],
        context=research_context,
    )
    attempt = journal.start_attempt()
    record_path = journal.append_record(
        work_item_id="case-a",
        payload=research_payload,
        attempt_id=attempt,
    )
    terminal_path = journal.finalize()

    assert load_canonical_json(root / "plan.json")["context"] == research_context
    assert load_canonical_json(record_path)["payload"] == research_payload
    assert set(load_canonical_json(terminal_path)) == {
        "journal_schema_version",
        "execution_id",
        "plan_fingerprint",
        "record_digests",
        "record_digest_aggregate",
        "status",
        "content_sha256",
    }
    journal.close()


def test_diagnostics_project_validated_records_attempts_and_last_durable_immutably(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    first = journal.start_attempt()
    journal.append_record(
        work_item_id="first",
        payload={"status": "unmatched", "nested": ["opaque"]},
        attempt_id=first,
    )
    journal.record_attempt_failure(
        attempt_id=first,
        failure_code="process.interrupted",
        failure_message="operator stopped worker",
    )
    journal.close()

    continued = _open(root)
    second = continued.start_attempt()
    continued.close()
    diagnostics = ExecutionEvidenceJournal.inspect_diagnostics(root)

    assert JOURNAL_SCHEMA_VERSION == 1
    assert diagnostics.last_durable_record == diagnostics.records[0]
    assert diagnostics.records[0].sequence == 0
    assert diagnostics.records[0].work_item_id == "first"
    assert diagnostics.records[0].attempt_id == first
    assert diagnostics.records[0].payload["status"] == "unmatched"
    assert diagnostics.records[0].payload_fingerprint == json_sha256(
        {"status": "unmatched", "nested": ["opaque"]}
    )
    by_id = {attempt.attempt_id: attempt for attempt in diagnostics.attempts}
    assert by_id[first].status == "failed"
    assert by_id[second].status == "unfinished"
    with pytest.raises(TypeError):
        diagnostics.records[0].payload["status"] = "mutated"
    with pytest.raises(TypeError):
        by_id[first].failure["code"] = "mutated"  # type: ignore[index]


def test_diagnostics_rejects_corrupt_persisted_record_before_projecting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    attempt = journal.start_attempt()
    record_path = journal.append_record(
        work_item_id="first", payload={"value": 1}, attempt_id=attempt
    )
    journal.close()
    corrupted = load_canonical_json(record_path)
    corrupted["payload_fingerprint"] = "0" * 64
    _rewrite_content_digest(record_path, corrupted)

    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.inspect_diagnostics(root)
    assert exc_info.value.code == "journal.invalid_persisted_evidence"


def test_attempt_surfaces_reject_freeform_research_mappings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt-mechanics-only"
    journal = _create(root)

    with pytest.raises(TypeError):
        cast(Any, journal).start_attempt({"claim": {"effect": "supported"}})
    assert not tuple((root / "attempts").iterdir())

    attempt = journal.start_attempt()
    with pytest.raises(TypeError):
        cast(Any, journal).record_attempt_failure(
            attempt_id=attempt,
            failure={
                "scientific_validity": True,
                "stop_rule": {"decision": "promote"},
            },
        )
    assert not (root / "attempts" / attempt / "outcome.json").exists()

    outcome_path = journal.record_attempt_failure(
        attempt_id=attempt,
        failure_code="process.interrupted",
        failure_message="worker exited before completion",
        exception_type="KeyboardInterrupt",
    )
    assert load_canonical_json(outcome_path)["failure"] == {
        "code": "process.interrupted",
        "message": "worker exited before completion",
        "exception_type": "KeyboardInterrupt",
    }
    journal.close()


def test_attempt_failure_rejects_rehashed_unknown_research_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt-failure-envelope"
    path = _materialize_journal_surface(root=root, surface="attempt_outcome")
    envelope = load_canonical_json(path)
    failure = cast(dict[str, object], envelope["failure"])
    failure["scientific_validity"] = True
    _rewrite_content_digest(path, envelope)

    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.inspect(root)

    assert exc_info.value.code == "journal.invalid_persisted_evidence"


@pytest.mark.parametrize(
    "surface",
    ["plan", "attempt_start", "attempt_outcome", "record", "terminal"],
)
def test_persisted_envelopes_reject_rehashed_unknown_research_keys(
    tmp_path: Path,
    surface: str,
) -> None:
    root = tmp_path / surface
    path = _materialize_journal_surface(root=root, surface=surface)
    envelope = load_canonical_json(path)
    envelope["scientific_validity"] = "caller-claim-must-stay-opaque"
    _rewrite_content_digest(path, envelope)

    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.inspect(root)

    assert exc_info.value.code == "journal.invalid_persisted_evidence"


@pytest.mark.parametrize(
    "surface",
    ["plan", "attempt_start", "attempt_outcome", "record", "terminal"],
)
def test_persisted_envelopes_reject_boolean_schema_versions(
    tmp_path: Path,
    surface: str,
) -> None:
    root = tmp_path / surface
    path = _materialize_journal_surface(root=root, surface=surface)
    envelope = load_canonical_json(path)
    envelope["journal_schema_version"] = True
    _rewrite_content_digest(path, envelope)

    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.inspect(root)

    assert exc_info.value.code == "journal.invalid_persisted_evidence"


def test_continuation_requires_exact_identity_and_terminal_requires_full_plan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    attempt = journal.start_attempt()
    journal.append_record(work_item_id="first", payload={}, attempt_id=attempt)
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.finalize()
    assert exc_info.value.code == "journal.incomplete_plan"
    journal.close()

    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.open(
            root=root,
            execution_id="execution-a",
            execution_identity={"source": "changed"},
            expected_work_item_ids=["first", "second"],
            context={"mode": "cpu"},
        )
    assert exc_info.value.code == "journal.continuation_identity_mismatch"


def test_reload_reports_uncommitted_temporary_file_without_accepting_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    temporary = root / "records" / ".00000000-left-behind.json"
    temporary.write_text('{"not":"accepted"}', encoding="utf-8")

    snapshot = journal.reload()
    assert snapshot.completed_work_item_ids == ()
    assert snapshot.temporary_paths == (temporary,)
    journal.close()


def test_single_writer_refuses_a_second_live_writer(tmp_path: Path) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    with pytest.raises(ArtifactContractError) as exc_info:
        _open(root)
    assert exc_info.value.code == "journal.writer_already_active"
    journal.close()


def test_explicit_record_sequence_cannot_backfill_or_skip(tmp_path: Path) -> None:
    journal = _create(tmp_path / "journal")
    attempt = journal.start_attempt()
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="first", payload={}, attempt_id=attempt, sequence=1
        )
    assert exc_info.value.code == "journal.invalid_sequence"
    journal.append_record(work_item_id="first", payload={}, attempt_id=attempt)
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="second", payload={}, attempt_id=attempt, sequence=0
        )
    assert exc_info.value.code == "journal.invalid_sequence"
    journal.close()


def test_cached_append_rejects_duplicate_and_sequence_without_full_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _create(tmp_path / "journal")
    attempt = journal.start_attempt()

    def reject_full_reload(*args: object, **kwargs: object) -> object:
        raise AssertionError("append performed a full journal reload")

    monkeypatch.setattr(journal_module, "_inspect", reject_full_reload)
    journal.append_record(work_item_id="first", payload={}, attempt_id=attempt)
    assert journal.completed_work_item_ids == ("first",)
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(work_item_id="first", payload={}, attempt_id=attempt)
    assert exc_info.value.code == "journal.work_item_already_completed"
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="second", payload={}, attempt_id=attempt, sequence=2
        )
    assert exc_info.value.code == "journal.invalid_sequence"
    journal.close()


@pytest.mark.parametrize("directory_name", ["records", "attempts"])
def test_reload_rejects_missing_required_directories_after_plan_exists(
    tmp_path: Path, directory_name: str
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    journal.close()
    (root / directory_name).rmdir()
    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.inspect(root)
    assert exc_info.value.code == "journal.invalid_persisted_evidence"


def test_reload_rejects_record_sequence_gap_and_deleted_attempt_start(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    attempt = journal.start_attempt()
    first = journal.append_record(
        work_item_id="first", payload={"value": 1}, attempt_id=attempt
    )
    journal.append_record(
        work_item_id="second", payload={"value": 2}, attempt_id=attempt
    )
    first.unlink()
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.reload()
    assert exc_info.value.code == "journal.invalid_persisted_evidence"
    journal.close()

    root = tmp_path / "journal-attempt"
    journal = _create(root)
    attempt = journal.start_attempt()
    journal.append_record(work_item_id="first", payload={}, attempt_id=attempt)
    (root / "attempts" / attempt / "start.json").unlink()
    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.inspect(root)
    assert exc_info.value.code == "journal.invalid_persisted_evidence"
    journal.close()


def test_uncommitted_attempt_start_is_diagnostic_and_does_not_block_continuation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root, filesystem=_FailOnFileSyncCall(4))
    first_attempt = journal.start_attempt()
    journal.append_record(
        work_item_id="first", payload={"receipt": 1}, attempt_id=first_attempt
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        journal.start_attempt()
    assert exc_info.value.code == "artifact.publish_failed"
    snapshot = journal.reload()
    orphan_dirs = tuple(path for path in snapshot.temporary_paths if path.is_dir())
    assert len(orphan_dirs) == 1
    assert orphan_dirs[0].parent == root / "attempts"
    assert snapshot.completed_work_item_ids == ("first",)
    journal.close()

    continued = _open(root)
    assert continued.completed_work_item_ids == ("first",)
    successor_attempt = continued.start_attempt()
    continued.append_record(
        work_item_id="second", payload={"receipt": 2}, attempt_id=successor_attempt
    )
    assert load_canonical_json(continued.finalize())["status"] == "completed"
    continued.close()


def test_outcome_without_committed_attempt_start_remains_an_integrity_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    journal.close()
    orphan = root / "attempts" / "orphan"
    orphan.mkdir()
    (orphan / "outcome.json").write_bytes(b"{}")

    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.inspect(root)
    assert exc_info.value.code == "journal.invalid_persisted_evidence"


def test_directory_sync_failure_poisons_stale_writer_until_full_reload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root, filesystem=_FailOnDirectorySyncCall(5))
    attempt = journal.start_attempt()

    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="first", payload={"receipt": 1}, attempt_id=attempt
        )
    assert exc_info.value.code == "artifact.publish_failed"
    assert exc_info.value.context["published_before_failure"] is True
    assert tuple(path.name for path in (root / "records").glob("*.json")) == (
        f"00000000-{json_sha256('first')[:16]}.json",
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        journal.append_record(
            work_item_id="second", payload={"receipt": 2}, attempt_id=attempt
        )
    assert exc_info.value.code == "journal.writer_requires_reload"
    assert len(tuple((root / "records").glob("*.json"))) == 1

    assert journal.reload().completed_work_item_ids == ("first",)
    journal.append_record(
        work_item_id="second", payload={"receipt": 2}, attempt_id=attempt
    )
    assert tuple(
        sorted(path.name[:8] for path in (root / "records").glob("*.json"))
    ) == (
        "00000000",
        "00000001",
    )
    assert load_canonical_json(journal.finalize())["status"] == "completed"
    journal.close()


def test_cpu_interruption_shape_preserves_records_until_a_fresh_attempt_finalizes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root, filesystem=_TerminalFailingFileSystemOps())
    first_attempt = journal.start_attempt()
    journal.append_record(
        work_item_id="first", payload={"receipt": 1}, attempt_id=first_attempt
    )
    journal.append_record(
        work_item_id="second", payload={"receipt": 2}, attempt_id=first_attempt
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.finalize()
    assert exc_info.value.code == "artifact.publish_failed"
    assert ExecutionEvidenceJournal.inspect(root).terminal is None
    assert ExecutionEvidenceJournal.inspect(root).completed_work_item_ids == (
        "first",
        "second",
    )
    journal.close()

    continued = _open(root)
    second_attempt = continued.start_attempt()
    assert second_attempt != first_attempt
    assert load_canonical_json(continued.finalize())["status"] == "completed"
    continued.close()


def test_exact_cpu_gate_requires_an_authored_callback_receipt_and_recovers(
    tmp_path: Path,
) -> None:
    rejected_root = tmp_path / "rejected-live-callback"
    with pytest.raises(ArtifactContractError) as exc_info:
        ExecutionEvidenceJournal.create(
            root=rejected_root,
            execution_id="execution-callback-gate",
            execution_identity={"source": "cpu-gate"},
            expected_work_item_ids=["first", "second"],
            context={"finalizer": lambda: None},
        )
    assert exc_info.value.code == "artifact.invalid_json_value"
    assert not rejected_root.exists()

    receipt = {
        "finalizer": {
            "kind": "authored_callback_receipt",
            "name": "fixed-dose-score-bias",
        }
    }
    root = tmp_path / "accepted-explicit-receipt"
    journal = ExecutionEvidenceJournal.create(
        root=root,
        execution_id="execution-callback-gate",
        execution_identity={"source": "cpu-gate"},
        expected_work_item_ids=["first", "second"],
        context=receipt,
        filesystem=_TerminalFailingFileSystemOps(),
    )
    assert load_canonical_json(root / "plan.json")["context"] == receipt
    attempt = journal.start_attempt()
    journal.append_record(
        work_item_id="first", payload={"receipt": 1}, attempt_id=attempt
    )
    journal.append_record(
        work_item_id="second", payload={"receipt": 2}, attempt_id=attempt
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        journal.finalize()
    assert exc_info.value.code == "artifact.publish_failed"
    journal.close()

    continued = ExecutionEvidenceJournal.open(
        root=root,
        execution_id="execution-callback-gate",
        execution_identity={"source": "cpu-gate"},
        expected_work_item_ids=["first", "second"],
        context=receipt,
    )
    successor_attempt = continued.start_attempt()
    assert successor_attempt != attempt
    assert continued.completed_work_item_ids == ("first", "second")
    assert load_canonical_json(continued.finalize())["status"] == "completed"
    continued.close()


def test_interrupted_attempt_exposes_prior_ids_for_a_fresh_completion_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    journal = _create(root)
    first_attempt = journal.start_attempt()
    journal.append_record(
        work_item_id="first", payload={"receipt": 1}, attempt_id=first_attempt
    )
    journal.record_attempt_failure(
        attempt_id=first_attempt,
        failure_code="interrupted",
        failure_message="synthetic interruption",
    )
    journal.close()

    continued = _open(root)
    assert continued.completed_work_item_ids == ("first",)
    second_attempt = continued.start_attempt()
    continued.append_record(
        work_item_id="second", payload={"receipt": 2}, attempt_id=second_attempt
    )
    assert load_canonical_json(continued.finalize())["status"] == "completed"
    continued.close()


class _FailingFileSystemOps(LocalFileSystemOps):
    def __init__(self, failure: str) -> None:
        self.failure = failure

    def fsync_file(self, file_descriptor: int) -> None:
        if self.failure == "file":
            raise OSError("injected file sync failure")
        super().fsync_file(file_descriptor)

    def publish_exclusive(self, temporary_path: Path, final_path: Path) -> None:
        if self.failure == "publish":
            raise OSError("injected publication failure")
        super().publish_exclusive(temporary_path, final_path)

    def fsync_directory(self, directory: Path) -> None:
        if self.failure == "directory":
            raise OSError("injected directory sync failure")
        super().fsync_directory(directory)


class _TerminalFailingFileSystemOps(LocalFileSystemOps):
    def publish_exclusive(self, temporary_path: Path, final_path: Path) -> None:
        if final_path.name == "terminal.json":
            raise OSError("injected terminal publication failure")
        super().publish_exclusive(temporary_path, final_path)


class _FailOnDirectorySyncCall(LocalFileSystemOps):
    def __init__(self, failing_call: int) -> None:
        self.failing_call = failing_call
        self.calls = 0

    def fsync_directory(self, directory: Path) -> None:
        self.calls += 1
        if self.calls == self.failing_call:
            raise OSError("injected directory sync failure")
        super().fsync_directory(directory)


class _FailOnFileSyncCall(LocalFileSystemOps):
    def __init__(self, failing_call: int) -> None:
        self.failing_call = failing_call
        self.calls = 0

    def fsync_file(self, file_descriptor: int) -> None:
        self.calls += 1
        if self.calls == self.failing_call:
            raise OSError("injected file sync failure")
        super().fsync_file(file_descriptor)


def _create(
    root: Path, *, filesystem: LocalFileSystemOps | None = None
) -> ExecutionEvidenceJournal:
    return ExecutionEvidenceJournal.create(
        root=root,
        execution_id="execution-a",
        execution_identity={"source": "a"},
        expected_work_item_ids=["first", "second"],
        context={"mode": "cpu"},
        **({"filesystem": filesystem} if filesystem is not None else {}),
    )


def _open(root: Path) -> ExecutionEvidenceJournal:
    return ExecutionEvidenceJournal.open(
        root=root,
        execution_id="execution-a",
        execution_identity={"source": "a"},
        expected_work_item_ids=["first", "second"],
        context={"mode": "cpu"},
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_journal_surface(*, root: Path, surface: str) -> Path:
    journal = ExecutionEvidenceJournal.create(
        root=root,
        execution_id="execution-envelope",
        execution_identity={"source": "unit"},
        expected_work_item_ids=["only"],
        context={"mode": "cpu"},
    )
    if surface == "plan":
        journal.close()
        return root / "plan.json"
    attempt = journal.start_attempt()
    if surface == "attempt_start":
        journal.close()
        return root / "attempts" / attempt / "start.json"
    if surface == "attempt_outcome":
        path = journal.record_attempt_failure(
            attempt_id=attempt,
            failure_code="unit",
            failure_message="unit failure",
        )
        journal.close()
        return path
    record = journal.append_record(
        work_item_id="only",
        payload={"value": 1},
        attempt_id=attempt,
    )
    if surface == "record":
        journal.close()
        return record
    terminal = journal.finalize()
    journal.close()
    if surface != "terminal":
        raise AssertionError(f"unknown journal surface: {surface}")
    return terminal


def _rewrite_content_digest(path: Path, envelope: dict[str, object]) -> None:
    envelope.pop("content_sha256", None)
    envelope["content_sha256"] = json_sha256(envelope)
    path.write_bytes(canonical_json_bytes(envelope))
