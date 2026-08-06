"""Immutable execution plans with independently durable item records."""

from __future__ import annotations

import fcntl
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts.json_values import (
    DEFAULT_FILESYSTEM_OPS,
    FileSystemOps,
    bytes_sha256,
    json_sha256,
    load_canonical_json,
    publish_json_exclusive,
    validate_json_value,
)
from src.common.errors import ArtifactContractError


JOURNAL_SCHEMA_VERSION = 1
_PLAN_KEYS = {
    "journal_schema_version",
    "execution_id",
    "execution_identity",
    "execution_identity_fingerprint",
    "expected_work_item_ids",
    "context",
    "context_fingerprint",
    "plan_fingerprint",
    "content_sha256",
}
_ATTEMPT_START_KEYS = {
    "journal_schema_version",
    "execution_id",
    "plan_fingerprint",
    "attempt_id",
    "content_sha256",
}
_ATTEMPT_OUTCOME_KEYS = {
    "journal_schema_version",
    "execution_id",
    "plan_fingerprint",
    "attempt_id",
    "status",
    "failure",
    "content_sha256",
}
_ATTEMPT_FAILURE_KEYS = {
    "code",
    "message",
    "exception_type",
}
_RECORD_KEYS = {
    "journal_schema_version",
    "execution_id",
    "execution_identity_fingerprint",
    "plan_fingerprint",
    "sequence",
    "work_item_id",
    "attempt_id",
    "payload",
    "payload_fingerprint",
    "content_sha256",
}
_TERMINAL_KEYS = {
    "journal_schema_version",
    "execution_id",
    "plan_fingerprint",
    "record_digests",
    "record_digest_aggregate",
    "status",
    "content_sha256",
}


@dataclass(frozen=True)
class JournalSnapshot:
    execution_id: str
    execution_identity_fingerprint: str
    plan_fingerprint: str
    completed_work_item_ids: tuple[str, ...]
    temporary_paths: tuple[Path, ...]
    terminal: Mapping[str, Any] | None


class ExecutionEvidenceJournal:
    """One locked writer over a sealed plan and non-replacing record files."""

    def __init__(
        self,
        *,
        root: Path,
        plan: Mapping[str, Any],
        lock_descriptor: int,
        filesystem: FileSystemOps,
        completed_work_item_ids: Sequence[str],
        terminal: Mapping[str, Any] | None,
    ) -> None:
        self.root = root
        self._plan = dict(plan)
        self._lock_descriptor = lock_descriptor
        self._filesystem = filesystem
        self._expected_work_item_id_set = set(plan["expected_work_item_ids"])
        self._completed_work_item_ids = list(completed_work_item_ids)
        self._completed_work_item_id_set = set(completed_work_item_ids)
        self._next_sequence = len(completed_work_item_ids)
        self._terminal = terminal
        self._poisoned = False

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        execution_id: str,
        execution_identity: Mapping[str, Any],
        expected_work_item_ids: Sequence[str],
        context: Mapping[str, Any],
        filesystem: FileSystemOps = DEFAULT_FILESYSTEM_OPS,
    ) -> "ExecutionEvidenceJournal":
        _require_nonempty_string(execution_id, field="execution_id")
        identity = _strict_mapping(execution_identity, field="execution_identity")
        opaque_context = _strict_mapping(context, field="context")
        expected = _expected_ids(expected_work_item_ids)
        root = root.resolve()
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactContractError(
                "journal root is already occupied",
                code="journal.root_already_exists",
                context={"root": str(root)},
                cause=exc,
            ) from exc
        descriptor = -1
        try:
            # Prepare all non-evidence structure before plan publication.  A
            # failed preparation therefore leaves an occupied diagnostic root,
            # never an apparently admitted plan missing its required surfaces.
            (root / "records").mkdir()
            (root / "attempts").mkdir()
            filesystem.fsync_directory(root)
            descriptor = _acquire_writer_lock(root)
            core = {
                "journal_schema_version": JOURNAL_SCHEMA_VERSION,
                "execution_id": execution_id,
                "execution_identity": identity,
                "execution_identity_fingerprint": json_sha256(identity),
                "expected_work_item_ids": expected,
                "context": opaque_context,
                "context_fingerprint": json_sha256(opaque_context),
            }
            plan = {
                **core,
                "plan_fingerprint": json_sha256(core),
            }
            plan = _with_content_digest(plan)
            publish_json_exclusive(root / "plan.json", plan, filesystem=filesystem)
            snapshot, loaded_plan, _ = _inspect(root)
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            raise ArtifactContractError(
                "journal preparation or plan publication did not complete",
                code=(
                    "artifact.publish_failed"
                    if (root / "plan.json").exists()
                    else "journal.prepare_failed"
                ),
                context={"root": str(root)},
                cause=exc,
            ) from exc
        except BaseException:
            # A failed root remains visibly unadmitted; never remove it or its
            # diagnostics because a caller may need to inspect the failure shape.
            if descriptor >= 0:
                os.close(descriptor)
            raise
        return cls(
            root=root,
            plan=loaded_plan,
            lock_descriptor=descriptor,
            filesystem=filesystem,
            completed_work_item_ids=snapshot.completed_work_item_ids,
            terminal=snapshot.terminal,
        )

    @classmethod
    def open(
        cls,
        *,
        root: Path,
        execution_id: str,
        execution_identity: Mapping[str, Any],
        expected_work_item_ids: Sequence[str],
        context: Mapping[str, Any],
        filesystem: FileSystemOps = DEFAULT_FILESYSTEM_OPS,
    ) -> "ExecutionEvidenceJournal":
        root = root.resolve()
        if not root.is_dir():
            raise ArtifactContractError(
                "journal root does not exist or is not a directory",
                code="journal.root_missing",
                context={"root": str(root)},
            )
        descriptor = _acquire_writer_lock(root)
        try:
            snapshot, plan, _ = _inspect(root)
            if snapshot.terminal is not None:
                raise ArtifactContractError(
                    "terminal journal cannot accept another writer",
                    code="journal.already_terminal",
                    context={"root": str(root)},
                )
            requested_core = {
                "journal_schema_version": JOURNAL_SCHEMA_VERSION,
                "execution_id": execution_id,
                "execution_identity": _strict_mapping(
                    execution_identity, field="execution_identity"
                ),
                "execution_identity_fingerprint": json_sha256(execution_identity),
                "expected_work_item_ids": _expected_ids(expected_work_item_ids),
                "context": _strict_mapping(context, field="context"),
                "context_fingerprint": json_sha256(context),
            }
            requested_fingerprint = json_sha256(requested_core)
            if (
                plan["execution_id"] != execution_id
                or plan["plan_fingerprint"] != requested_fingerprint
            ):
                raise ArtifactContractError(
                    "journal continuation identity does not match its immutable plan",
                    code="journal.continuation_identity_mismatch",
                    context={"root": str(root)},
                )
        except BaseException:
            os.close(descriptor)
            raise
        return cls(
            root=root,
            plan=plan,
            lock_descriptor=descriptor,
            filesystem=filesystem,
            completed_work_item_ids=snapshot.completed_work_item_ids,
            terminal=snapshot.terminal,
        )

    @classmethod
    def inspect(cls, root: Path) -> JournalSnapshot:
        snapshot, _, _ = _inspect(root.resolve())
        return snapshot

    @property
    def execution_id(self) -> str:
        return str(self._plan["execution_id"])

    @property
    def plan_fingerprint(self) -> str:
        return str(self._plan["plan_fingerprint"])

    @property
    def plan_file_sha256(self) -> str:
        return bytes_sha256((self.root / "plan.json").read_bytes())

    @property
    def completed_work_item_ids(self) -> tuple[str, ...]:
        return tuple(self._completed_work_item_ids)

    def close(self) -> None:
        if self._lock_descriptor >= 0:
            os.close(self._lock_descriptor)
            self._lock_descriptor = -1

    def __enter__(self) -> "ExecutionEvidenceJournal":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start_attempt(self) -> str:
        self._require_open_writer()
        attempt_id = uuid.uuid4().hex
        payload = _with_content_digest(
            {
                "journal_schema_version": JOURNAL_SCHEMA_VERSION,
                "execution_id": self.execution_id,
                "plan_fingerprint": self.plan_fingerprint,
                "attempt_id": attempt_id,
            }
        )
        attempt_dir = self.root / "attempts" / attempt_id
        try:
            attempt_dir.mkdir()
        except OSError as exc:
            raise ArtifactContractError(
                "attempt directory could not be prepared",
                code="journal.attempt_prepare_failed",
                context={"path": str(attempt_dir)},
                cause=exc,
            ) from exc
        try:
            self._publish_json(
                attempt_dir / "start.json", payload, filesystem=self._filesystem
            )
            self._filesystem.fsync_directory(self.root / "attempts")
        except OSError as exc:
            self._poisoned = True
            raise ArtifactContractError(
                "attempt-directory publication did not complete",
                code="artifact.publish_failed",
                context={
                    "path": str(attempt_dir),
                    "published_before_failure": True,
                },
                cause=exc,
            ) from exc
        return attempt_id

    def record_attempt_failure(
        self,
        *,
        attempt_id: str,
        failure_code: str,
        failure_message: str,
        exception_type: str | None = None,
    ) -> Path:
        self._require_open_writer()
        self._validate_attempt_id(attempt_id)
        failure = _mechanical_failure(
            failure_code=failure_code,
            failure_message=failure_message,
            exception_type=exception_type,
        )
        payload = _with_content_digest(
            {
                "journal_schema_version": JOURNAL_SCHEMA_VERSION,
                "execution_id": self.execution_id,
                "plan_fingerprint": self.plan_fingerprint,
                "attempt_id": attempt_id,
                "status": "failed",
                "failure": _strict_mapping(failure, field="failure"),
            }
        )
        return self._publish_json(
            self.root / "attempts" / attempt_id / "outcome.json",
            payload,
            filesystem=self._filesystem,
        )

    def append_record(
        self,
        *,
        work_item_id: str,
        payload: Any,
        attempt_id: str,
        sequence: int | None = None,
    ) -> Path:
        self._require_open_writer()
        _require_nonempty_string(work_item_id, field="work_item_id")
        self._validate_attempt_id(attempt_id)
        validate_json_value(payload)
        if self._terminal is not None:
            raise ArtifactContractError(
                "terminal journal cannot accept another record",
                code="journal.already_terminal",
            )
        if work_item_id not in self._expected_work_item_id_set:
            raise ArtifactContractError(
                "record work item is not part of the immutable plan",
                code="journal.unplanned_work_item",
                context={"work_item_id": work_item_id},
            )
        if work_item_id in self._completed_work_item_id_set:
            raise ArtifactContractError(
                "record work item was already accepted",
                code="journal.work_item_already_completed",
                context={"work_item_id": work_item_id},
            )
        chosen_sequence = self._next_sequence if sequence is None else sequence
        if isinstance(chosen_sequence, bool) or not isinstance(chosen_sequence, int):
            raise ArtifactContractError(
                "record sequence must be an integer",
                code="journal.invalid_sequence",
            )
        if chosen_sequence != self._next_sequence:
            raise ArtifactContractError(
                "record sequence must be the next monotonic sequence",
                code="journal.invalid_sequence",
                context={"sequence": chosen_sequence, "expected": self._next_sequence},
            )
        record = _with_content_digest(
            {
                "journal_schema_version": JOURNAL_SCHEMA_VERSION,
                "execution_id": self.execution_id,
                "execution_identity_fingerprint": self._plan[
                    "execution_identity_fingerprint"
                ],
                "plan_fingerprint": self.plan_fingerprint,
                "sequence": chosen_sequence,
                "work_item_id": work_item_id,
                "attempt_id": attempt_id,
                "payload": payload,
                "payload_fingerprint": json_sha256(payload),
            }
        )
        record_name = f"{chosen_sequence:08d}-{json_sha256(work_item_id)[:16]}.json"
        output = self._publish_json(
            self.root / "records" / record_name,
            record,
            filesystem=self._filesystem,
        )
        try:
            published_record = load_canonical_json(output)
            _validate_record(published_record, self._plan, {attempt_id})
            if (
                published_record["sequence"] != self._next_sequence
                or published_record["work_item_id"] != work_item_id
            ):
                _invalid_journal("published record differs from the accepted append")
        except BaseException:
            self._poisoned = True
            raise
        self._completed_work_item_ids.append(work_item_id)
        self._completed_work_item_id_set.add(work_item_id)
        self._next_sequence += 1
        return output

    def finalize(self) -> Path:
        self._require_open_writer()
        snapshot, plan, records = _inspect(self.root)
        if plan["plan_fingerprint"] != self.plan_fingerprint:
            raise ArtifactContractError(
                "journal plan changed while writer was open",
                code="journal.plan_changed",
            )
        self._accept_snapshot(snapshot)
        if snapshot.terminal is not None:
            raise ArtifactContractError(
                "journal is already terminal",
                code="journal.already_terminal",
            )
        expected = tuple(self._plan["expected_work_item_ids"])
        if set(snapshot.completed_work_item_ids) != set(expected):
            raise ArtifactContractError(
                "terminal completion requires every planned work item",
                code="journal.incomplete_plan",
                context={
                    "completed": snapshot.completed_work_item_ids,
                    "expected": expected,
                },
            )
        record_digests = [record["content_sha256"] for record in records]
        terminal = _with_content_digest(
            {
                "journal_schema_version": JOURNAL_SCHEMA_VERSION,
                "execution_id": self.execution_id,
                "plan_fingerprint": self.plan_fingerprint,
                "record_digests": record_digests,
                "record_digest_aggregate": json_sha256(record_digests),
                "status": "completed",
            }
        )
        output = self._publish_json(
            self.root / "terminal.json", terminal, filesystem=self._filesystem
        )
        try:
            published_terminal = load_canonical_json(output)
            _validate_terminal(published_terminal, self._plan, records)
        except BaseException:
            self._poisoned = True
            raise
        self._terminal = published_terminal
        return output

    def reload(self) -> JournalSnapshot:
        snapshot, plan, _ = _inspect(self.root)
        if plan["plan_fingerprint"] != self.plan_fingerprint:
            raise ArtifactContractError(
                "journal plan changed while writer was open",
                code="journal.plan_changed",
            )
        self._accept_snapshot(snapshot)
        self._poisoned = False
        return snapshot

    def _accept_snapshot(self, snapshot: JournalSnapshot) -> None:
        self._completed_work_item_ids = list(snapshot.completed_work_item_ids)
        self._completed_work_item_id_set = set(snapshot.completed_work_item_ids)
        self._next_sequence = len(snapshot.completed_work_item_ids)
        self._terminal = snapshot.terminal

    def _validate_attempt_id(self, attempt_id: str) -> None:
        _require_nonempty_string(attempt_id, field="attempt_id")
        start_path = self.root / "attempts" / attempt_id / "start.json"
        if not start_path.is_file():
            raise ArtifactContractError(
                "record references an unknown process attempt",
                code="journal.unknown_attempt",
                context={"attempt_id": attempt_id},
            )
        _validate_attempt(load_canonical_json(start_path), self._plan, attempt_id)
        if (start_path.parent / "outcome.json").exists():
            raise ArtifactContractError(
                "process attempt already has a terminal outcome",
                code="journal.attempt_already_closed",
                context={"attempt_id": attempt_id},
            )

    def _publish_json(
        self,
        path: Path,
        value: Any,
        *,
        filesystem: FileSystemOps,
    ) -> Path:
        try:
            return publish_json_exclusive(path, value, filesystem=filesystem)
        except ArtifactContractError as exc:
            if exc.context.get("published_before_failure") is True:
                self._poisoned = True
            raise

    def _require_open_writer(self) -> None:
        if self._lock_descriptor < 0:
            raise ArtifactContractError(
                "journal writer is closed",
                code="journal.writer_closed",
            )
        if self._poisoned:
            raise ArtifactContractError(
                "journal writer requires full reload after uncertain publication",
                code="journal.writer_requires_reload",
            )


def _inspect(
    root: Path,
) -> tuple[JournalSnapshot, dict[str, Any], list[dict[str, Any]]]:
    plan = _load_plan(root / "plan.json")
    _validate_plan(plan)
    validated_attempt_ids, orphan_attempt_dirs = _validate_attempts(root, plan)
    records = _load_records(root, plan, validated_attempt_ids)
    completed = tuple(record["work_item_id"] for record in records)
    temporary_paths = tuple(
        sorted(
            {
                *orphan_attempt_dirs,
                *(
                    path
                    for path in root.rglob(".*")
                    if path.is_file() and path.name != ".writer.lock"
                ),
            }
        )
    )
    terminal_path = root / "terminal.json"
    terminal: Mapping[str, Any] | None = None
    if terminal_path.exists():
        loaded_terminal = load_canonical_json(terminal_path)
        if not isinstance(loaded_terminal, Mapping):
            _invalid_journal("terminal receipt must be a mapping")
        _validate_terminal(loaded_terminal, plan, records)
        terminal = loaded_terminal
    snapshot = JournalSnapshot(
        execution_id=plan["execution_id"],
        execution_identity_fingerprint=plan["execution_identity_fingerprint"],
        plan_fingerprint=plan["plan_fingerprint"],
        completed_work_item_ids=completed,
        temporary_paths=temporary_paths,
        terminal=terminal,
    )
    return snapshot, plan, records


def _load_plan(path: Path) -> dict[str, Any]:
    value = load_canonical_json(path)
    if not isinstance(value, dict):
        _invalid_journal("plan must be a mapping")
    return value


def _load_records(
    root: Path,
    plan: Mapping[str, Any],
    validated_attempt_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records_dir = root / "records"
    if not records_dir.is_dir():
        _invalid_journal("records directory is missing")
    records: list[dict[str, Any]] = []
    for path in sorted(
        path for path in records_dir.glob("*.json") if not path.name.startswith(".")
    ):
        value = load_canonical_json(path)
        _validate_record(value, plan, validated_attempt_ids)
        records.append(value)
    records.sort(key=lambda item: item["sequence"])
    sequences = [item["sequence"] for item in records]
    identifiers = [item["work_item_id"] for item in records]
    if sequences != list(range(len(records))) or len(identifiers) != len(
        set(identifiers)
    ):
        _invalid_journal("accepted records have duplicate sequence or work item")
    return records


def _validate_attempts(
    root: Path, plan: Mapping[str, Any]
) -> tuple[set[str], set[Path]]:
    attempts_dir = root / "attempts"
    if not attempts_dir.is_dir():
        _invalid_journal("attempts directory is missing")
    attempt_ids: set[str] = set()
    orphan_attempt_dirs: set[Path] = set()
    for path in attempts_dir.iterdir():
        if not path.is_dir() and not path.name.startswith("."):
            _invalid_journal("attempts directory contains an unknown final path")
    for attempt_dir in sorted(path for path in attempts_dir.iterdir() if path.is_dir()):
        attempt_id = attempt_dir.name
        start_path = attempt_dir / "start.json"
        outcome_path = attempt_dir / "outcome.json"
        if start_path.exists() and not start_path.is_file():
            _invalid_journal("attempt start path is not a file")
        if outcome_path.exists() and not outcome_path.is_file():
            _invalid_journal("attempt outcome path is not a file")
        unknown_final_paths = tuple(
            path
            for path in attempt_dir.iterdir()
            if not path.name.startswith(".")
            and path.name not in {"start.json", "outcome.json"}
        )
        if unknown_final_paths:
            _invalid_journal("attempt directory contains an unknown final path")
        if not start_path.exists():
            if outcome_path.exists():
                _invalid_journal("attempt outcome exists without a committed start")
            orphan_attempt_dirs.add(attempt_dir)
            continue
        _validate_attempt(load_canonical_json(start_path), plan, attempt_id)
        attempt_ids.add(attempt_id)
        if outcome_path.exists():
            outcome = load_canonical_json(outcome_path)
            if not isinstance(outcome, Mapping):
                _invalid_journal("attempt outcome must be a mapping")
            _verify_content_digest(outcome)
            _require_exact_envelope(
                outcome,
                expected_keys=_ATTEMPT_OUTCOME_KEYS,
                envelope="attempt outcome",
            )
            _require_journal_schema_version(outcome, envelope="attempt outcome")
            if (
                outcome.get("execution_id") != plan["execution_id"]
                or outcome.get("plan_fingerprint") != plan["plan_fingerprint"]
                or outcome.get("attempt_id") != attempt_id
                or outcome.get("status") != "failed"
            ):
                _invalid_journal("attempt outcome does not bind this plan")
            failure = outcome.get("failure")
            if not isinstance(failure, Mapping):
                _invalid_journal("attempt outcome failure must be a mapping")
            _validate_mechanical_failure(
                _strict_mapping(failure, field="attempt_failure")
            )
    return attempt_ids, orphan_attempt_dirs


def _validate_plan(plan: Mapping[str, Any]) -> None:
    _verify_content_digest(plan)
    _require_exact_envelope(plan, expected_keys=_PLAN_KEYS, envelope="plan")
    _require_journal_schema_version(plan, envelope="plan")
    _require_nonempty_string(plan["execution_id"], field="execution_id")
    identity = _strict_mapping(plan["execution_identity"], field="execution_identity")
    context = _strict_mapping(plan["context"], field="context")
    expected = _expected_ids(plan["expected_work_item_ids"])
    if plan["execution_identity_fingerprint"] != json_sha256(identity):
        _invalid_journal("plan execution identity fingerprint is invalid")
    if plan["context_fingerprint"] != json_sha256(context):
        _invalid_journal("plan context fingerprint is invalid")
    core = {
        "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        "execution_id": plan["execution_id"],
        "execution_identity": identity,
        "execution_identity_fingerprint": plan["execution_identity_fingerprint"],
        "expected_work_item_ids": expected,
        "context": context,
        "context_fingerprint": plan["context_fingerprint"],
    }
    if plan["plan_fingerprint"] != json_sha256(core):
        _invalid_journal("plan fingerprint is invalid")


def _validate_record(
    record: Any,
    plan: Mapping[str, Any],
    validated_attempt_ids: set[str] | None,
) -> None:
    if not isinstance(record, Mapping):
        _invalid_journal("record must be a mapping")
    _verify_content_digest(record)
    _require_exact_envelope(record, expected_keys=_RECORD_KEYS, envelope="record")
    _require_journal_schema_version(record, envelope="record")
    if (
        record.get("execution_id") != plan["execution_id"]
        or record.get("execution_identity_fingerprint")
        != plan["execution_identity_fingerprint"]
        or record.get("plan_fingerprint") != plan["plan_fingerprint"]
    ):
        _invalid_journal("record does not bind this plan")
    _require_nonempty_string(record.get("work_item_id"), field="work_item_id")
    _require_nonempty_string(record.get("attempt_id"), field="attempt_id")
    if (
        validated_attempt_ids is not None
        and record["attempt_id"] not in validated_attempt_ids
    ):
        _invalid_journal("record references an unvalidated process attempt")
    if record["work_item_id"] not in plan["expected_work_item_ids"]:
        _invalid_journal("record work item is not planned")
    if isinstance(record.get("sequence"), bool) or not isinstance(
        record.get("sequence"), int
    ):
        _invalid_journal("record sequence is invalid")
    validate_json_value(record.get("payload"))
    if record.get("payload_fingerprint") != json_sha256(record["payload"]):
        _invalid_journal("record payload fingerprint is invalid")


def _validate_attempt(value: Any, plan: Mapping[str, Any], attempt_id: str) -> None:
    if not isinstance(value, Mapping):
        _invalid_journal("attempt receipt must be a mapping")
    _verify_content_digest(value)
    _require_exact_envelope(
        value,
        expected_keys=_ATTEMPT_START_KEYS,
        envelope="attempt start",
    )
    _require_journal_schema_version(value, envelope="attempt start")
    if (
        value.get("execution_id") != plan["execution_id"]
        or value.get("plan_fingerprint") != plan["plan_fingerprint"]
        or value.get("attempt_id") != attempt_id
    ):
        _invalid_journal("attempt receipt does not bind this plan")


def _mechanical_failure(
    *,
    failure_code: str,
    failure_message: str,
    exception_type: str | None,
) -> dict[str, Any]:
    _require_nonempty_string(failure_code, field="failure_code")
    if not isinstance(failure_message, str):
        raise ArtifactContractError(
            "attempt failure message must be a string",
            code="journal.invalid_value",
            context={"field": "failure_message"},
        )
    if exception_type is not None:
        _require_nonempty_string(exception_type, field="exception_type")
    return {
        "code": failure_code,
        "message": failure_message,
        "exception_type": exception_type,
    }


def _validate_mechanical_failure(value: Mapping[str, Any]) -> None:
    _require_exact_envelope(
        value,
        expected_keys=_ATTEMPT_FAILURE_KEYS,
        envelope="attempt failure",
    )
    if not isinstance(value.get("code"), str) or not value["code"]:
        _invalid_journal("attempt failure code is invalid")
    if not isinstance(value.get("message"), str):
        _invalid_journal("attempt failure message is invalid")
    exception_type = value.get("exception_type")
    if exception_type is not None and (
        not isinstance(exception_type, str) or not exception_type
    ):
        _invalid_journal("attempt failure exception type is invalid")


def _validate_terminal(
    terminal: Mapping[str, Any],
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    _verify_content_digest(terminal)
    _require_exact_envelope(
        terminal,
        expected_keys=_TERMINAL_KEYS,
        envelope="terminal",
    )
    _require_journal_schema_version(terminal, envelope="terminal")
    if (
        terminal.get("execution_id") != plan["execution_id"]
        or terminal.get("plan_fingerprint") != plan["plan_fingerprint"]
        or terminal.get("status") != "completed"
    ):
        _invalid_journal("terminal receipt does not bind this plan")
    expected = set(plan["expected_work_item_ids"])
    if {record["work_item_id"] for record in records} != expected:
        _invalid_journal("terminal receipt has incomplete records")
    digests = [
        record["content_sha256"]
        for record in sorted(records, key=lambda x: x["sequence"])
    ]
    if terminal.get("record_digests") != digests or terminal.get(
        "record_digest_aggregate"
    ) != json_sha256(digests):
        _invalid_journal("terminal record aggregate is invalid")


def _require_exact_envelope(
    value: Mapping[str, Any],
    *,
    expected_keys: set[str],
    envelope: str,
) -> None:
    if set(value) != expected_keys:
        _invalid_journal(f"{envelope} schema is invalid")


def _require_journal_schema_version(value: Mapping[str, Any], *, envelope: str) -> None:
    schema_version = value.get("journal_schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != JOURNAL_SCHEMA_VERSION
    ):
        _invalid_journal(f"{envelope} schema version is invalid")


def _with_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if "content_sha256" in payload:
        _invalid_journal("content digest cannot be supplied by a caller")
    validate_json_value(payload)
    payload["content_sha256"] = json_sha256(payload)
    return payload


def _verify_content_digest(value: Mapping[str, Any]) -> None:
    if "content_sha256" not in value or not isinstance(value["content_sha256"], str):
        _invalid_journal("record content digest is missing")
    payload = dict(value)
    digest = payload.pop("content_sha256")
    if json_sha256(payload) != digest:
        _invalid_journal("record content digest is invalid")


def _strict_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(
            "journal field must be a mapping",
            code="journal.invalid_value",
            context={"field": field},
        )
    result = dict(value)
    validate_json_value(result)
    return result


def _expected_ids(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactContractError(
            "expected work item identifiers must be a sequence",
            code="journal.invalid_plan",
        )
    identifiers = list(value)
    if not identifiers or any(
        not isinstance(item, str) or not item for item in identifiers
    ):
        raise ArtifactContractError(
            "expected work item identifiers must be nonempty strings",
            code="journal.invalid_plan",
        )
    if len(identifiers) != len(set(identifiers)):
        raise ArtifactContractError(
            "expected work item identifiers must be unique",
            code="journal.duplicate_work_item",
        )
    return identifiers


def _require_nonempty_string(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactContractError(
            "journal field must be a nonempty string",
            code="journal.invalid_value",
            context={"field": field},
        )


def _acquire_writer_lock(root: Path) -> int:
    descriptor = os.open(root / ".writer.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ArtifactContractError(
            "journal already has an exclusive writer",
            code="journal.writer_already_active",
            context={"root": str(root)},
            cause=exc,
        ) from exc
    return descriptor


def _invalid_journal(message: str) -> None:
    raise ArtifactContractError(message, code="journal.invalid_persisted_evidence")
