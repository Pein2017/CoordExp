"""Immutable cohort and append-only attempt ledgers for research schedules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal

from src.common.errors import ArtifactContractError, DataContractError


COHORT_RECORD_SCHEMA_VERSION = "spatial_scope_history.cohort_record.v1"
ATTEMPT_RECORD_SCHEMA_VERSION = "spatial_scope_history.attempt_record.v4"
ATTEMPT_DEPENDENCY_SCHEMA_VERSION = "spatial_scope_history.attempt_dependency.v2"
IDENTITY_BUNDLE_SCHEMA_VERSION = "spatial_scope_history.identity_bundle.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AttemptStatus = Literal["completed", "failed", "skipped", "capped", "invalid"]
_ATTEMPT_STATUSES = frozenset({"completed", "failed", "skipped", "capped", "invalid"})


def canonical_json_text(value: Any) -> str:
    """Return the single accepted JSON representation for research identities."""

    canonical = _canonical_json_value(value, path="$")
    return json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_payload(value: Any) -> str:
    """Hash a canonical JSON-safe payload with Secure Hash Algorithm 256-bit."""

    return hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionIdentityBundle:
    """Exact code, config, reference-ledger, and runtime identities for one run."""

    code_sha256: str
    config_sha256: str
    ledger_sha256: str
    runtime_sha256: str
    schema_version: str = IDENTITY_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != IDENTITY_BUNDLE_SCHEMA_VERSION:
            _data_error(
                "identity bundle has an unsupported schema version",
                code="analysis.identity_bundle_schema",
                schema_version=self.schema_version,
            )
        for field_name in (
            "code_sha256",
            "config_sha256",
            "ledger_sha256",
            "runtime_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, str]:
        return {
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "ledger_sha256": self.ledger_sha256,
            "runtime_sha256": self.runtime_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> ExecutionIdentityBundle:
        fields = {
            "code_sha256",
            "config_sha256",
            "ledger_sha256",
            "runtime_sha256",
            "schema_version",
        }
        _require_exact_keys(
            value, fields=fields, record_name="execution identity bundle"
        )
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True)
class CohortImageRecord:
    """One source image in an immutable, explicitly ordered execution cohort."""

    image_id: int
    frozen_order: int
    source_row_index: int
    image_path: str
    image_sha256: str
    source_width: int
    source_height: int
    raw_width: int
    raw_height: int
    source_row_sha256: str
    source_dataset_sha256: str
    raw_annotation_sha256: str
    noncrowd_annotated_object_count: int
    annotated_person_count: int
    annotated_food_tableware_count: int
    source_crowd_annotation_count: int
    cohort_memberships: tuple[str, ...]
    density_tags: tuple[str, ...]
    schema_version: str = COHORT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COHORT_RECORD_SCHEMA_VERSION:
            _data_error(
                "cohort image record has an unsupported schema version",
                code="analysis.cohort_record_schema",
                schema_version=self.schema_version,
            )
        _require_nonnegative_integer(self.image_id, field="image_id")
        _require_nonnegative_integer(self.frozen_order, field="frozen_order")
        _require_nonnegative_integer(self.source_row_index, field="source_row_index")
        _require_positive_integer(self.source_width, field="source_width")
        _require_positive_integer(self.source_height, field="source_height")
        _require_positive_integer(self.raw_width, field="raw_width")
        _require_positive_integer(self.raw_height, field="raw_height")
        for field_name in (
            "noncrowd_annotated_object_count",
            "annotated_person_count",
            "annotated_food_tableware_count",
            "source_crowd_annotation_count",
        ):
            _require_nonnegative_integer(getattr(self, field_name), field=field_name)
        _require_nonempty_string(self.image_path, field="image_path")
        for field_name in (
            "image_sha256",
            "source_row_sha256",
            "source_dataset_sha256",
            "raw_annotation_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        _validate_sorted_unique_strings(
            self.cohort_memberships, field="cohort_memberships"
        )
        _validate_sorted_unique_strings(self.density_tags, field="density_tags")
        if not self.cohort_memberships:
            _data_error(
                "cohort image record requires at least one cohort membership",
                code="analysis.cohort_membership_empty",
                image_id=self.image_id,
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "annotated_food_tableware_count": self.annotated_food_tableware_count,
            "annotated_person_count": self.annotated_person_count,
            "cohort_memberships": list(self.cohort_memberships),
            "density_tags": list(self.density_tags),
            "frozen_order": self.frozen_order,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "noncrowd_annotated_object_count": self.noncrowd_annotated_object_count,
            "raw_height": self.raw_height,
            "raw_annotation_sha256": self.raw_annotation_sha256,
            "raw_width": self.raw_width,
            "schema_version": self.schema_version,
            "source_crowd_annotation_count": self.source_crowd_annotation_count,
            "source_dataset_sha256": self.source_dataset_sha256,
            "source_height": self.source_height,
            "source_row_index": self.source_row_index,
            "source_row_sha256": self.source_row_sha256,
            "source_width": self.source_width,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> CohortImageRecord:
        fields = {
            "annotated_food_tableware_count",
            "annotated_person_count",
            "cohort_memberships",
            "density_tags",
            "frozen_order",
            "image_id",
            "image_path",
            "image_sha256",
            "noncrowd_annotated_object_count",
            "raw_height",
            "raw_annotation_sha256",
            "raw_width",
            "schema_version",
            "source_crowd_annotation_count",
            "source_dataset_sha256",
            "source_height",
            "source_row_index",
            "source_row_sha256",
            "source_width",
        }
        _require_exact_keys(value, fields=fields, record_name="cohort image record")
        payload = dict(value)
        payload["cohort_memberships"] = tuple(payload["cohort_memberships"])
        payload["density_tags"] = tuple(payload["density_tags"])
        return cls(**payload)


@dataclass(frozen=True)
class CohortLedger:
    """A content-addressed cohort whose ordering cannot drift at resume time."""

    cohort_id: str
    full_name: str
    operational_meaning: str
    records: tuple[CohortImageRecord, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.cohort_id, field="cohort_id")
        _require_nonempty_string(self.full_name, field="full_name")
        _require_nonempty_string(self.operational_meaning, field="operational_meaning")
        if not self.records:
            _data_error(
                "cohort ledger must contain at least one image",
                code="analysis.cohort_empty",
            )
        image_ids = [record.image_id for record in self.records]
        if len(image_ids) != len(set(image_ids)):
            _data_error(
                "cohort ledger contains duplicate image identifiers",
                code="analysis.cohort_duplicate_image",
                image_ids=image_ids,
            )
        observed_order = [record.frozen_order for record in self.records]
        expected_order = list(range(len(self.records)))
        if observed_order != expected_order:
            _data_error(
                "cohort ledger records must follow contiguous frozen order",
                code="analysis.cohort_order",
                observed_order=observed_order,
                expected_order=expected_order,
            )

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_artifact_dict())

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "full_name": self.full_name,
            "operational_meaning": self.operational_meaning,
            "records": [record.to_artifact_dict() for record in self.records],
        }

    def to_jsonl_bytes(self) -> bytes:
        lines = []
        for record in self.records:
            lines.append(
                canonical_json_text(
                    {
                        "cohort_id": self.cohort_id,
                        "full_name": self.full_name,
                        "operational_meaning": self.operational_meaning,
                        "record": record.to_artifact_dict(),
                    }
                )
            )
        return ("\n".join(lines) + "\n").encode("utf-8")

    @classmethod
    def from_jsonl_bytes(cls, payload: bytes) -> CohortLedger:
        rows = _parse_canonical_jsonl(payload, artifact_name="cohort ledger")
        if not rows:
            _data_error("cohort ledger is empty", code="analysis.cohort_empty")
        expected = {"cohort_id", "full_name", "operational_meaning", "record"}
        for row in rows:
            _require_exact_keys(row, fields=expected, record_name="cohort ledger row")
        identities = {
            (row["cohort_id"], row["full_name"], row["operational_meaning"])
            for row in rows
        }
        if len(identities) != 1:
            _data_error(
                "cohort ledger row metadata drifted within one artifact",
                code="analysis.cohort_metadata_drift",
            )
        cohort_id, full_name, operational_meaning = identities.pop()
        return cls(
            cohort_id=cohort_id,
            full_name=full_name,
            operational_meaning=operational_meaning,
            records=tuple(
                CohortImageRecord.from_artifact_dict(row["record"]) for row in rows
            ),
        )


@dataclass(frozen=True)
class AttemptDependencyContract:
    """One request's sealed physical batch and cumulative-state requirements."""

    request_id: str
    physical_batch_plan_sha256: str
    physical_batch_sha256: str
    physical_batch_index: int
    predecessor_request_id: str | None
    initial_cumulative_state_sha256: str | None
    schema_version: str = ATTEMPT_DEPENDENCY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.request_id, field="request_id")
        _require_sha256(
            self.physical_batch_plan_sha256,
            field="physical_batch_plan_sha256",
        )
        _require_sha256(self.physical_batch_sha256, field="physical_batch_sha256")
        _require_nonnegative_integer(
            self.physical_batch_index,
            field="physical_batch_index",
        )
        if self.schema_version != ATTEMPT_DEPENDENCY_SCHEMA_VERSION:
            _artifact_error(
                "attempt dependency has an unsupported schema version",
                code="analysis.attempt_dependency_schema",
            )
        if self.predecessor_request_id is not None:
            _require_nonempty_string(
                self.predecessor_request_id, field="predecessor_request_id"
            )
        if self.initial_cumulative_state_sha256 is not None:
            _require_sha256(
                self.initial_cumulative_state_sha256,
                field="initial_cumulative_state_sha256",
            )
        if (
            self.predecessor_request_id is not None
            and self.initial_cumulative_state_sha256 is not None
        ):
            _artifact_error(
                "dependency cannot have both a predecessor and an initial state",
                code="analysis.attempt_dependency_ambiguous_origin",
                request_id=self.request_id,
            )

    @property
    def is_cumulative(self) -> bool:
        return (
            self.predecessor_request_id is not None
            or self.initial_cumulative_state_sha256 is not None
        )


@dataclass(frozen=True)
class AttemptRecord:
    """One terminal, non-repeatable attempt for one pre-materialized request."""

    run_id: str
    schedule_sha256: str
    request_id: str
    physical_batch_plan_sha256: str
    physical_batch_sha256: str
    physical_batch_index: int
    attempt_status: AttemptStatus
    started_at_utc: str
    finished_at_utc: str
    execution_identity: ExecutionIdentityBundle
    output_artifact_sha256: str | None = None
    output_artifact_path: str | None = None
    failure_code: str | None = None
    expected_cumulative_state_sha256: str | None = None
    produced_cumulative_state_sha256: str | None = None
    produced_cumulative_state_artifact_path: str | None = None
    schema_version: str = ATTEMPT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTEMPT_RECORD_SCHEMA_VERSION:
            _artifact_error(
                "attempt record has an unsupported schema version",
                code="analysis.attempt_record_schema",
                schema_version=self.schema_version,
            )
        for field_name in ("run_id", "request_id", "started_at_utc", "finished_at_utc"):
            _require_nonempty_string(getattr(self, field_name), field=field_name)
        _require_sha256(self.schedule_sha256, field="schedule_sha256")
        _require_sha256(
            self.physical_batch_plan_sha256,
            field="physical_batch_plan_sha256",
        )
        _require_sha256(self.physical_batch_sha256, field="physical_batch_sha256")
        _require_nonnegative_integer(
            self.physical_batch_index,
            field="physical_batch_index",
        )
        if self.attempt_status not in _ATTEMPT_STATUSES:
            _artifact_error(
                "attempt record has an unknown terminal status",
                code="analysis.attempt_status",
                attempt_status=self.attempt_status,
            )
        if self.output_artifact_sha256 is not None:
            _require_sha256(self.output_artifact_sha256, field="output_artifact_sha256")
        if self.output_artifact_path is not None:
            _require_nonempty_string(
                self.output_artifact_path,
                field="output_artifact_path",
            )
        if self.attempt_status == "completed" and self.output_artifact_sha256 is None:
            _artifact_error(
                "completed attempt requires an output artifact digest",
                code="analysis.attempt_completed_without_output",
                request_id=self.request_id,
            )
        if self.attempt_status != "completed" and not self.failure_code:
            _artifact_error(
                "non-completed attempt requires a failure code",
                code="analysis.attempt_failure_code_missing",
                request_id=self.request_id,
                attempt_status=self.attempt_status,
            )
        for field_name in (
            "expected_cumulative_state_sha256",
            "produced_cumulative_state_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_sha256(value, field=field_name)
        if self.produced_cumulative_state_artifact_path is not None:
            _require_nonempty_string(
                self.produced_cumulative_state_artifact_path,
                field="produced_cumulative_state_artifact_path",
            )
        if (self.produced_cumulative_state_sha256 is None) != (
            self.produced_cumulative_state_artifact_path is None
        ):
            _artifact_error(
                "produced cumulative-state fingerprint and artifact path must coexist",
                code="analysis.attempt_cumulative_state_artifact_pair",
                request_id=self.request_id,
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "attempt_status": self.attempt_status,
            "execution_identity": self.execution_identity.to_artifact_dict(),
            "expected_cumulative_state_sha256": (self.expected_cumulative_state_sha256),
            "failure_code": self.failure_code,
            "finished_at_utc": self.finished_at_utc,
            "output_artifact_sha256": self.output_artifact_sha256,
            "output_artifact_path": self.output_artifact_path,
            "physical_batch_index": self.physical_batch_index,
            "physical_batch_plan_sha256": self.physical_batch_plan_sha256,
            "physical_batch_sha256": self.physical_batch_sha256,
            "produced_cumulative_state_artifact_path": (
                self.produced_cumulative_state_artifact_path
            ),
            "produced_cumulative_state_sha256": self.produced_cumulative_state_sha256,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "schedule_sha256": self.schedule_sha256,
            "schema_version": self.schema_version,
            "started_at_utc": self.started_at_utc,
        }

    @classmethod
    def from_artifact_dict(cls, value: Mapping[str, Any]) -> AttemptRecord:
        fields = {
            "attempt_status",
            "execution_identity",
            "expected_cumulative_state_sha256",
            "failure_code",
            "finished_at_utc",
            "output_artifact_sha256",
            "output_artifact_path",
            "physical_batch_index",
            "physical_batch_plan_sha256",
            "physical_batch_sha256",
            "produced_cumulative_state_artifact_path",
            "produced_cumulative_state_sha256",
            "request_id",
            "run_id",
            "schedule_sha256",
            "schema_version",
            "started_at_utc",
        }
        _require_exact_keys(value, fields=fields, record_name="attempt record")
        payload = dict(value)
        payload["execution_identity"] = ExecutionIdentityBundle.from_artifact_dict(
            payload["execution_identity"]
        )
        return cls(**payload)


@dataclass(frozen=True)
class AttemptLedger:
    """Validated append-only terminal attempts joined to one immutable schedule."""

    run_id: str
    schedule_sha256: str
    execution_identity: ExecutionIdentityBundle
    records: tuple[AttemptRecord, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.run_id, field="run_id")
        _require_sha256(self.schedule_sha256, field="schedule_sha256")
        request_ids: set[str] = set()
        for record in self.records:
            if record.run_id != self.run_id:
                _artifact_error(
                    "attempt ledger contains a different run identity",
                    code="analysis.attempt_run_drift",
                    request_id=record.request_id,
                )
            if record.schedule_sha256 != self.schedule_sha256:
                _artifact_error(
                    "attempt ledger contains a different schedule identity",
                    code="analysis.attempt_schedule_drift",
                    request_id=record.request_id,
                )
            if record.execution_identity != self.execution_identity:
                _artifact_error(
                    "attempt ledger contains different execution identities",
                    code="analysis.attempt_execution_identity_drift",
                    request_id=record.request_id,
                )
            if record.request_id in request_ids:
                _artifact_error(
                    "attempt ledger repeats a request identity",
                    code="analysis.attempt_duplicate_request",
                    request_id=record.request_id,
                )
            request_ids.add(record.request_id)

    @property
    def attempted_request_ids(self) -> frozenset[str]:
        return frozenset(record.request_id for record in self.records)

    @property
    def successfully_completed_request_ids(self) -> frozenset[str]:
        return frozenset(
            record.request_id
            for record in self.records
            if record.attempt_status == "completed"
        )

    @property
    def records_by_request_id(self) -> Mapping[str, AttemptRecord]:
        return {record.request_id: record for record in self.records}

    def validate_request_universe(self, request_ids: Iterable[str]) -> None:
        allowed = frozenset(request_ids)
        unknown = sorted(self.attempted_request_ids - allowed)
        if unknown:
            _artifact_error(
                "attempt ledger contains request identities absent from its schedule",
                code="analysis.attempt_unknown_request",
                unknown_request_ids=unknown,
            )

    def validate_dependencies(
        self,
        dependencies: Sequence[AttemptDependencyContract],
        *,
        verify_cumulative_state_artifacts: bool = True,
    ) -> None:
        dependency_by_id = _dependency_map(dependencies)
        self.validate_request_universe(dependency_by_id)
        preceding: dict[str, AttemptRecord] = {}
        for record in self.records:
            _validate_attempt_dependency(
                record,
                dependency=dependency_by_id[record.request_id],
                preceding_records=preceding,
                verify_cumulative_state_artifacts=verify_cumulative_state_artifacts,
            )
            preceding[record.request_id] = record

    def with_appended(self, record: AttemptRecord) -> AttemptLedger:
        return AttemptLedger(
            run_id=self.run_id,
            schedule_sha256=self.schedule_sha256,
            execution_identity=self.execution_identity,
            records=(*self.records, record),
        )

    def to_jsonl_bytes(self) -> bytes:
        if not self.records:
            return b""
        return (
            "\n".join(
                canonical_json_text(record.to_artifact_dict())
                for record in self.records
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_jsonl_bytes(
        cls,
        payload: bytes,
        *,
        run_id: str,
        schedule_sha256: str,
        execution_identity: ExecutionIdentityBundle,
    ) -> AttemptLedger:
        rows = _parse_canonical_jsonl(payload, artifact_name="attempt ledger")
        return cls(
            run_id=run_id,
            schedule_sha256=schedule_sha256,
            execution_identity=execution_identity,
            records=tuple(AttemptRecord.from_artifact_dict(row) for row in rows),
        )


def append_attempt_record(
    path: Path,
    *,
    record: AttemptRecord,
    dependencies: Sequence[AttemptDependencyContract],
) -> AttemptLedger:
    """Append one unique terminal attempt after validating the complete ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = handle.read()
            ledger = AttemptLedger.from_jsonl_bytes(
                existing,
                run_id=record.run_id,
                schedule_sha256=record.schedule_sha256,
                execution_identity=record.execution_identity,
            )
            dependency_by_id = _dependency_map(dependencies)
            ledger.validate_dependencies(dependencies)
            if record.request_id not in dependency_by_id:
                _artifact_error(
                    "attempt request identity is absent from its schedule",
                    code="analysis.attempt_unknown_request",
                    request_id=record.request_id,
                )
            _validate_produced_state_artifact(record)
            updated = ledger.with_appended(record)
            updated.validate_dependencies(dependencies)
            handle.seek(0, os.SEEK_END)
            handle.write(
                (canonical_json_text(record.to_artifact_dict()) + "\n").encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
            return updated
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def dependency_skip_failure_code(
    *, predecessor_request_id: str, predecessor_status: AttemptStatus
) -> str:
    """Return the exact terminal-skip code for a non-completed predecessor."""

    return f"dependency_predecessor_not_completed:{predecessor_request_id}:{predecessor_status}"


def dependency_state_unavailable_failure_code(*, predecessor_request_id: str) -> str:
    """Return the exact terminal-skip code for unavailable predecessor state."""

    return f"dependency_predecessor_state_unavailable:{predecessor_request_id}"


def sha256_file(path: Path) -> str:
    """Hash a cumulative-state artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_map(
    dependencies: Sequence[AttemptDependencyContract],
) -> dict[str, AttemptDependencyContract]:
    result: dict[str, AttemptDependencyContract] = {}
    for dependency in dependencies:
        if dependency.request_id in result:
            _artifact_error(
                "attempt dependency universe repeats a request identity",
                code="analysis.attempt_dependency_duplicate_request",
                request_id=dependency.request_id,
            )
        result[dependency.request_id] = dependency
    return result


def _validate_attempt_dependency(
    record: AttemptRecord,
    *,
    dependency: AttemptDependencyContract,
    preceding_records: Mapping[str, AttemptRecord],
    verify_cumulative_state_artifacts: bool,
) -> None:
    if (
        record.physical_batch_plan_sha256 != dependency.physical_batch_plan_sha256
        or record.physical_batch_sha256 != dependency.physical_batch_sha256
        or record.physical_batch_index != dependency.physical_batch_index
    ):
        _artifact_error(
            "attempt physical batch identity differs from the sealed schedule",
            code="analysis.attempt_physical_batch_drift",
            request_id=record.request_id,
            expected_physical_batch_plan_sha256=(dependency.physical_batch_plan_sha256),
            expected_physical_batch_sha256=dependency.physical_batch_sha256,
            expected_physical_batch_index=dependency.physical_batch_index,
            observed_physical_batch_plan_sha256=(record.physical_batch_plan_sha256),
            observed_physical_batch_sha256=record.physical_batch_sha256,
            observed_physical_batch_index=record.physical_batch_index,
        )
    if not dependency.is_cumulative:
        if any(
            value is not None
            for value in (
                record.expected_cumulative_state_sha256,
                record.produced_cumulative_state_sha256,
                record.produced_cumulative_state_artifact_path,
            )
        ):
            _artifact_error(
                "non-cumulative request carries cumulative-state evidence",
                code="analysis.attempt_unexpected_cumulative_state",
                request_id=record.request_id,
            )
        return

    predecessor: AttemptRecord | None = None
    expected_state = dependency.initial_cumulative_state_sha256
    if dependency.predecessor_request_id is not None:
        predecessor = preceding_records.get(dependency.predecessor_request_id)
        if predecessor is None:
            _artifact_error(
                "cumulative request was appended before its predecessor",
                code="analysis.attempt_dependency_out_of_order",
                request_id=record.request_id,
                predecessor_request_id=dependency.predecessor_request_id,
            )
        if predecessor.attempt_status != "completed":
            required_failure_code = dependency_skip_failure_code(
                predecessor_request_id=predecessor.request_id,
                predecessor_status=predecessor.attempt_status,
            )
            if (
                record.attempt_status != "skipped"
                or record.failure_code != required_failure_code
                or any(
                    value is not None
                    for value in (
                        record.expected_cumulative_state_sha256,
                        record.produced_cumulative_state_sha256,
                        record.produced_cumulative_state_artifact_path,
                    )
                )
            ):
                _artifact_error(
                    "downstream request of a non-completed predecessor must be terminally skipped",
                    code="analysis.attempt_dependency_skip_required",
                    request_id=record.request_id,
                    predecessor_request_id=predecessor.request_id,
                    predecessor_status=predecessor.attempt_status,
                    required_failure_code=required_failure_code,
                )
            return
        expected_state = predecessor.produced_cumulative_state_sha256
        if (
            expected_state is None
            or predecessor.produced_cumulative_state_artifact_path is None
        ):
            _artifact_error(
                "completed cumulative predecessor lacks reconstructible state evidence",
                code="analysis.attempt_predecessor_state_missing",
                request_id=record.request_id,
                predecessor_request_id=predecessor.request_id,
            )
        if verify_cumulative_state_artifacts and not _produced_state_artifact_matches(
            predecessor
        ):
            required_failure_code = dependency_state_unavailable_failure_code(
                predecessor_request_id=predecessor.request_id
            )
            if (
                record.attempt_status != "skipped"
                or record.failure_code != required_failure_code
                or any(
                    value is not None
                    for value in (
                        record.expected_cumulative_state_sha256,
                        record.produced_cumulative_state_sha256,
                        record.produced_cumulative_state_artifact_path,
                    )
                )
            ):
                _artifact_error(
                    "downstream request of an unreconstructible predecessor must be terminally skipped",
                    code="analysis.attempt_dependency_state_unavailable",
                    request_id=record.request_id,
                    predecessor_request_id=predecessor.request_id,
                    required_failure_code=required_failure_code,
                )
            return

    if record.expected_cumulative_state_sha256 != expected_state:
        _artifact_error(
            "cumulative request expected-state fingerprint does not match its predecessor",
            code="analysis.attempt_expected_state_mismatch",
            request_id=record.request_id,
            expected_cumulative_state_sha256=expected_state,
            observed_cumulative_state_sha256=record.expected_cumulative_state_sha256,
        )
    if record.attempt_status == "completed":
        if (
            record.produced_cumulative_state_sha256 is None
            or record.produced_cumulative_state_artifact_path is None
        ):
            _artifact_error(
                "completed cumulative attempt requires reconstructible produced state",
                code="analysis.attempt_produced_state_missing",
                request_id=record.request_id,
            )
    elif (
        record.produced_cumulative_state_sha256 is not None
        or record.produced_cumulative_state_artifact_path is not None
    ):
        _artifact_error(
            "non-completed cumulative attempt cannot produce committed state",
            code="analysis.attempt_noncompleted_produced_state",
            request_id=record.request_id,
        )


def _validate_produced_state_artifact(record: AttemptRecord) -> None:
    if record.produced_cumulative_state_sha256 is None:
        return
    if not _produced_state_artifact_matches(record):
        _artifact_error(
            "produced cumulative-state artifact is missing or does not match its fingerprint",
            code="analysis.attempt_produced_state_artifact_mismatch",
            request_id=record.request_id,
            artifact_path=record.produced_cumulative_state_artifact_path,
        )


def _produced_state_artifact_matches(record: AttemptRecord) -> bool:
    if (
        record.produced_cumulative_state_sha256 is None
        or record.produced_cumulative_state_artifact_path is None
    ):
        return False
    path = Path(record.produced_cumulative_state_artifact_path)
    return (
        path.is_file() and sha256_file(path) == record.produced_cumulative_state_sha256
    )


def _parse_canonical_jsonl(
    payload: bytes, *, artifact_name: str
) -> list[Mapping[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        _artifact_error(
            f"{artifact_name} must end with a newline",
            code="analysis.jsonl_final_newline",
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactContractError(
            f"{artifact_name} must use UTF-8",
            code="analysis.jsonl_utf8",
            cause=exc,
        ) from exc
    rows: list[Mapping[str, Any]] = []
    for line_index, line in enumerate(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(
                f"{artifact_name} contains invalid JSON",
                code="analysis.jsonl_parse",
                context={"line_index": line_index},
                cause=exc,
            ) from exc
        if not isinstance(value, Mapping):
            _artifact_error(
                f"{artifact_name} rows must be JSON objects",
                code="analysis.jsonl_row_type",
                line_index=line_index,
            )
        if line != canonical_json_text(value):
            _artifact_error(
                f"{artifact_name} rows must use canonical JSON serialization",
                code="analysis.jsonl_noncanonical",
                line_index=line_index,
            )
        rows.append(value)
    return rows


def _canonical_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _artifact_error(
                "canonical JSON payload contains a non-finite number",
                code="analysis.canonical_json_nonfinite",
                path=path,
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _artifact_error(
                    "canonical JSON object keys must be strings",
                    code="analysis.canonical_json_key_type",
                    path=path,
                )
            result[key] = _canonical_json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    _artifact_error(
        "canonical JSON payload contains an unsupported value",
        code="analysis.canonical_json_value_type",
        path=path,
        value_type=type(value).__name__,
    )


def _require_exact_keys(
    value: Mapping[str, Any], *, fields: set[str], record_name: str
) -> None:
    if not isinstance(value, Mapping):
        _artifact_error(
            f"{record_name} must be a mapping",
            code="analysis.record_type",
            record_name=record_name,
        )
    observed = set(value)
    if observed != fields:
        _artifact_error(
            f"{record_name} keys do not match its schema",
            code="analysis.record_keys",
            record_name=record_name,
            missing_keys=sorted(fields - observed),
            unknown_keys=sorted(observed - fields),
        )


def _validate_sorted_unique_strings(values: tuple[str, ...], *, field: str) -> None:
    if not isinstance(values, tuple):
        _data_error(
            "immutable string collection must be a tuple",
            code="analysis.immutable_tuple_required",
            field=field,
        )
    for value in values:
        _require_nonempty_string(value, field=field)
    if list(values) != sorted(set(values)):
        _data_error(
            "string collection must be sorted and unique",
            code="analysis.sorted_unique_strings",
            field=field,
            values=list(values),
        )


def _require_sha256(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _artifact_error(
            "identity field must be a lowercase SHA-256 hexadecimal digest",
            code="analysis.sha256",
            field=field,
            value=value,
        )


def _require_nonempty_string(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _data_error(
            "field must be a nonempty string",
            code="analysis.nonempty_string",
            field=field,
            value=value,
        )


def _require_nonnegative_integer(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _data_error(
            "field must be a nonnegative integer",
            code="analysis.nonnegative_integer",
            field=field,
            value=value,
        )


def _require_positive_integer(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _data_error(
            "field must be a positive integer",
            code="analysis.positive_integer",
            field=field,
            value=value,
        )


def _data_error(message: str, *, code: str, **context: Any) -> None:
    raise DataContractError(message, code=code, context=context)


def _artifact_error(message: str, *, code: str, **context: Any) -> None:
    raise ArtifactContractError(message, code=code, context=context)
