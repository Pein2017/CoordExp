"""Mechanics-only admission for production-shaped research probes.

This module binds immutable local inputs and closes two durable evidence stages.
It deliberately does not execute work, retry failures, or interpret scientific
payloads.
"""

from __future__ import annotations

import json
import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from src.artifacts.evidence_journal import (
    JOURNAL_SCHEMA_VERSION,
    ExecutionEvidenceJournal,
    JournalInspection,
)
from src.artifacts.json_values import (
    canonical_json_bytes,
    json_sha256,
    load_canonical_json,
    publish_json_exclusive,
    validate_json_value,
)
from src.common.errors import ArtifactContractError


ADMISSION_SCHEMA_VERSION: Final = 1
STAGES: Final = ("cpu_preflight", "vertical_smoke")
MECHANICS_STATEMENT: Final = (
    "Mechanics only; no scientific interpretation or launch authorization."
)

_CPU_ASSERTIONS: Final = {
    "production_entrypoint_resolved": True,
    "consumer_validator_ran": True,
    "model_free_finalizer_ran": True,
    "downstream_validator_accepted": True,
    "model_loaded": False,
    "gpu_used": False,
}
_VERTICAL_BOOLEAN_ASSERTIONS: Final = {
    "production_entrypoint_executed": True,
    "model_runtime_loaded": True,
    "terminal_finalizer_completed": True,
    "downstream_validator_accepted": True,
}
_CLAIM_BOUNDARY: Final = {
    "mechanics_only": True,
    "scientific_validity": False,
    "model_quality": False,
    "launch_authorized": False,
    "automatic_continuation": False,
    "training_authorized": False,
    "architecture_or_decoder_change_authorized": False,
    "downstream_promotion_authorized": False,
    "publication_authorized": False,
    "deployment_authorized": False,
}


class ResearchProbeAdmissionError(ArtifactContractError):
    """A fail-closed typed admission-contract failure."""


@dataclass(frozen=True)
class RegularFileBinding:
    name: str
    path: Path


@dataclass(frozen=True)
class DirectoryTreeBinding:
    name: str
    path: Path


@dataclass(frozen=True)
class ResolvedDataFileBinding:
    name: str
    declared_path: str | Path
    base_directory: Path


@dataclass(frozen=True)
class AbsoluteExecutableBinding:
    name: str
    path: Path


@dataclass(frozen=True)
class StrictValueBinding:
    name: str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_strict_value(self.value))


@dataclass(frozen=True)
class ReservedOutputPath:
    name: str
    path: Path


BindingRequest = (
    RegularFileBinding
    | DirectoryTreeBinding
    | ResolvedDataFileBinding
    | AbsoluteExecutableBinding
    | StrictValueBinding
)


@dataclass(frozen=True)
class BindingManifest:
    """Recursively immutable canonical projection of captured bindings."""

    bindings: tuple[Mapping[str, Any], ...]
    content_fingerprint: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "admission_binding_schema_version": ADMISSION_SCHEMA_VERSION,
            "bindings": [_thaw_json_value(item) for item in self.bindings],
            "content_fingerprint": self.content_fingerprint,
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(str(item["name"]) for item in self.bindings)


@dataclass(frozen=True)
class StageEvidence:
    """Caller-supplied mechanics facts for one fixed admission stage."""

    producer_binding: str
    validator_binding: str
    output_files: tuple[RegularFileBinding, ...]
    assertions: Mapping[str, Any]
    detail: Any
    scientific_interpretation: bool = False
    mechanics_statement: str = MECHANICS_STATEMENT

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_files", tuple(self.output_files))
        object.__setattr__(
            self, "assertions", _freeze_strict_value(self.assertions)
        )
        object.__setattr__(self, "detail", _freeze_strict_value(self.detail))


@dataclass(frozen=True)
class AdmissionInspection:
    admission_id: str
    binding_manifest_fingerprint: str
    completed_stages: tuple[str, ...]
    missing_stages: tuple[str, ...]
    journal_terminal: bool
    mechanically_admitted: bool


def capture_binding_manifest(requests: Sequence[BindingRequest]) -> BindingManifest:
    """Capture a deterministic strict manifest without creating any root."""

    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        _fail("binding requests must be a sequence", "admission.invalid_bindings")
    captured: list[dict[str, Any]] = []
    names: set[str] = set()
    for request in requests:
        name = _request_name(request)
        if name in names:
            _fail(
                "binding names must be unique",
                "admission.duplicate_binding_name",
                name=name,
            )
        names.add(name)
        captured.append(_capture_binding(request))
    if not captured:
        _fail("at least one binding is required", "admission.invalid_bindings")
    core = {
        "admission_binding_schema_version": ADMISSION_SCHEMA_VERSION,
        "bindings": captured,
    }
    fingerprint = json_sha256(core)
    return BindingManifest(
        bindings=tuple(_freeze_strict_value(item) for item in captured),
        content_fingerprint=fingerprint,
    )


def revalidate_binding_manifest(manifest: BindingManifest) -> None:
    """Re-capture every live input and require the complete manifest identity."""

    _validate_manifest(manifest)
    recaptured = capture_binding_manifest(
        tuple(_request_from_binding(item) for item in manifest.bindings)
    )
    if recaptured.to_mapping() != manifest.to_mapping():
        _fail(
            "live binding inputs differ from the immutable manifest",
            "admission.binding_drift",
        )


class ResearchProbeAdmission:
    """Locked writer or terminal projection over one admission dossier."""

    def __init__(
        self,
        *,
        root: Path,
        admission_id: str,
        bindings: BindingManifest,
        reserved_output_paths: tuple[Mapping[str, Any], ...],
        context: Mapping[str, Any],
        journal: ExecutionEvidenceJournal | None,
        journal_inspection: JournalInspection,
    ) -> None:
        self.root = root
        self.admission_id = admission_id
        self.bindings = bindings
        self.reserved_output_paths = reserved_output_paths
        self.context = context
        self._journal = journal
        self._inspection = journal_inspection

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        admission_id: str,
        bindings: BindingManifest,
        reserved_output_paths: Sequence[ReservedOutputPath],
        context: Mapping[str, Any],
    ) -> "ResearchProbeAdmission":
        """Validate everything in memory, then exclusively create the dossier."""

        _require_nonempty_string(admission_id, field="admission_id")
        _validate_manifest(bindings)
        revalidate_binding_manifest(bindings)
        strict_context = _strict_mapping(context, field="context")
        root_candidate = _require_path(root, field="root")
        if os.path.lexists(root_candidate):
            _fail(
                "admission root is already occupied",
                "admission.root_already_exists",
                root=str(root_candidate),
            )
        resolved_root = root_candidate.resolve(strict=False)
        reserved = _capture_reserved_paths(
            reserved_output_paths,
            require_absent=True,
            admission_root=resolved_root,
        )
        try:
            resolved_root.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            _fail(
                "admission root could not be created",
                "admission.root_create_failed",
                cause=exc,
                root=str(resolved_root),
            )
        try:
            publish_json_exclusive(resolved_root / "bindings.json", bindings.to_mapping())
            journal = ExecutionEvidenceJournal.create(
                root=resolved_root / "journal",
                execution_id=admission_id,
                execution_identity=_execution_identity(admission_id, bindings, reserved),
                expected_work_item_ids=STAGES,
                context=strict_context,
            )
            inspection = ExecutionEvidenceJournal.inspect_diagnostics(
                resolved_root / "journal"
            )
        except BaseException:
            # A partial root is durable diagnostic evidence and is never cleaned.
            raise
        return cls(
            root=resolved_root,
            admission_id=admission_id,
            bindings=bindings,
            reserved_output_paths=reserved,
            context=_freeze_strict_value(strict_context),
            journal=journal,
            journal_inspection=inspection,
        )

    @classmethod
    def open(
        cls,
        *,
        root: Path,
        admission_id: str,
        bindings: BindingManifest,
        reserved_output_paths: Sequence[ReservedOutputPath],
        context: Mapping[str, Any],
    ) -> "ResearchProbeAdmission":
        """Open only an exact continuation, including terminal receipt recovery."""

        _require_nonempty_string(admission_id, field="admission_id")
        _validate_manifest(bindings)
        revalidate_binding_manifest(bindings)
        strict_context = _strict_mapping(context, field="context")
        root_candidate = _require_path(root, field="root")
        if root_candidate.is_symlink():
            _fail(
                "admission root must not be a symlink",
                "admission.root_missing",
                root=str(root_candidate),
            )
        resolved_root = root_candidate.resolve(strict=False)
        if not resolved_root.is_dir() or resolved_root.is_symlink():
            _fail(
                "admission root is missing or invalid",
                "admission.root_missing",
                root=str(resolved_root),
            )
        reserved = _capture_reserved_paths(
            reserved_output_paths,
            require_absent=False,
            admission_root=resolved_root,
        )
        loaded_manifest = _load_binding_manifest(resolved_root / "bindings.json")
        if loaded_manifest.to_mapping() != bindings.to_mapping():
            _fail(
                "requested bindings do not match the published manifest",
                "admission.continuation_identity_mismatch",
            )
        journal_root = resolved_root / "journal"
        inspection = ExecutionEvidenceJournal.inspect_diagnostics(journal_root)
        expected_plan_fingerprint = _journal_plan_fingerprint(
            admission_id=admission_id,
            bindings=bindings,
            reserved=reserved,
            context=strict_context,
        )
        snapshot = inspection.snapshot
        if (
            snapshot.execution_id != admission_id
            or snapshot.plan_fingerprint != expected_plan_fingerprint
            or snapshot.expected_work_item_ids != STAGES
        ):
            _fail(
                "admission continuation identity does not match its journal plan",
                "admission.continuation_identity_mismatch",
            )
        _validate_persisted_stage_records(
            inspection,
            bindings,
            admission_id,
            reserved,
        )
        journal: ExecutionEvidenceJournal | None = None
        if snapshot.terminal is None:
            journal = ExecutionEvidenceJournal.open(
                root=journal_root,
                execution_id=admission_id,
                execution_identity=_execution_identity(admission_id, bindings, reserved),
                expected_work_item_ids=STAGES,
                context=strict_context,
            )
        return cls(
            root=resolved_root,
            admission_id=admission_id,
            bindings=bindings,
            reserved_output_paths=reserved,
            context=_freeze_strict_value(strict_context),
            journal=journal,
            journal_inspection=inspection,
        )

    @classmethod
    def inspect(cls, root: Path) -> AdmissionInspection:
        """Inspect persisted closure without authorizing execution or continuation."""

        resolved_root = _resolved_candidate(root, field="root")
        manifest = _load_binding_manifest(resolved_root / "bindings.json")
        revalidate_binding_manifest(manifest)
        journal = ExecutionEvidenceJournal.inspect_diagnostics(resolved_root / "journal")
        reserved = _load_reserved_plan(resolved_root, journal)
        _validate_persisted_stage_records(
            journal,
            manifest,
            journal.snapshot.execution_id,
            reserved,
        )
        completed = tuple(record.work_item_id for record in journal.records)
        admitted = False
        admission_path = resolved_root / "admission.json"
        if admission_path.exists():
            if journal.snapshot.terminal is None:
                _fail(
                    "admission receipt exists before terminal journal closure",
                    "admission.invalid_receipt",
                )
            expected_receipt = _build_receipt_projection(
                root=resolved_root,
                admission_id=journal.snapshot.execution_id,
                bindings=manifest,
                inspection=journal,
            )
            _require_identical_receipt(admission_path, expected_receipt)
            admitted = True
        return AdmissionInspection(
            admission_id=journal.snapshot.execution_id,
            binding_manifest_fingerprint=manifest.content_fingerprint,
            completed_stages=completed,
            missing_stages=tuple(stage for stage in STAGES if stage not in completed),
            journal_terminal=journal.snapshot.terminal is not None,
            mechanically_admitted=admitted,
        )

    @property
    def completed_stages(self) -> tuple[str, ...]:
        return tuple(record.work_item_id for record in self._inspection.records)

    @property
    def missing_stages(self) -> tuple[str, ...]:
        return tuple(stage for stage in STAGES if stage not in self.completed_stages)

    def close(self) -> None:
        if self._journal is not None:
            self._journal.close()
            self._journal = None

    def __enter__(self) -> "ResearchProbeAdmission":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start_attempt(self) -> str:
        return self._require_writer().start_attempt()

    def record_attempt_failure(
        self,
        *,
        attempt_id: str,
        failure_code: str,
        failure_message: str,
        exception_type: str | None = None,
    ) -> Path:
        return self._require_writer().record_attempt_failure(
            attempt_id=attempt_id,
            failure_code=failure_code,
            failure_message=failure_message,
            exception_type=exception_type,
        )

    def append_stage(
        self,
        *,
        stage: str,
        evidence: StageEvidence,
        attempt_id: str,
    ) -> Path:
        """Validate live inputs and one fixed envelope before durable append."""

        journal = self._require_writer()
        self._refresh_inspection()
        expected = self.missing_stages[0] if self.missing_stages else None
        if stage not in STAGES:
            _fail("stage is not planned", "admission.unplanned_stage", stage=stage)
        if stage != expected:
            _fail(
                "stage must be appended in fixed admission order",
                "admission.invalid_stage_order",
                stage=stage,
                expected=expected,
            )
        revalidate_binding_manifest(self.bindings)
        payload = _capture_stage_payload(
            stage=stage,
            evidence=evidence,
            admission_id=self.admission_id,
            admission_fingerprint=journal.plan_fingerprint,
            bindings=self.bindings,
            reserved_output_paths=self.reserved_output_paths,
        )
        output = journal.append_record(
            work_item_id=stage,
            payload=payload,
            attempt_id=attempt_id,
        )
        self._refresh_inspection()
        return output

    def finalize(self) -> Path:
        """Close the journal and idempotently publish one mechanics-only receipt."""

        revalidate_binding_manifest(self.bindings)
        self._refresh_inspection()
        if tuple(record.work_item_id for record in self._inspection.records) != STAGES:
            _fail(
                "admission requires both ordered stages",
                "admission.incomplete_stages",
                completed=self.completed_stages,
            )
        _validate_persisted_stage_records(
            self._inspection,
            self.bindings,
            self.admission_id,
            self.reserved_output_paths,
        )
        if self._inspection.snapshot.terminal is None:
            self._require_writer().finalize()
            self._refresh_inspection()
        if self._inspection.snapshot.terminal is None:
            _fail(
                "journal did not become terminal",
                "admission.journal_not_terminal",
            )
        receipt = self._build_receipt()
        receipt_path = self.root / "admission.json"
        if receipt_path.exists():
            _require_identical_receipt(receipt_path, receipt)
            return receipt_path
        try:
            publish_json_exclusive(receipt_path, receipt)
        except ArtifactContractError:
            if receipt_path.exists():
                _require_identical_receipt(receipt_path, receipt)
                return receipt_path
            raise
        return receipt_path

    def _build_receipt(self) -> dict[str, Any]:
        return _build_receipt_projection(
            root=self.root,
            admission_id=self.admission_id,
            bindings=self.bindings,
            inspection=self._inspection,
        )

    def _refresh_inspection(self) -> None:
        self._inspection = ExecutionEvidenceJournal.inspect_diagnostics(
            self.root / "journal"
        )

    def _require_writer(self) -> ExecutionEvidenceJournal:
        if self._journal is None:
            _fail(
                "terminal or closed admission has no active writer",
                "admission.writer_unavailable",
            )
        return self._journal


def _capture_binding(request: BindingRequest) -> dict[str, Any]:
    if isinstance(request, RegularFileBinding):
        return _capture_regular_file(name=request.name, path=request.path)
    if isinstance(request, DirectoryTreeBinding):
        return _capture_directory_tree(name=request.name, path=request.path)
    if isinstance(request, ResolvedDataFileBinding):
        return _capture_resolved_data_file(request)
    if isinstance(request, AbsoluteExecutableBinding):
        return _capture_absolute_executable(request)
    if isinstance(request, StrictValueBinding):
        value = _thaw_json_value(request.value)
        try:
            validate_json_value(value)
        except ArtifactContractError as exc:
            _fail(
                "strict-value binding contains an unsupported live value",
                "admission.invalid_strict_value",
                cause=exc,
                name=request.name,
            )
        return {
            "kind": "strict_value",
            "name": request.name,
            "value": value,
            "value_fingerprint": json_sha256(value),
        }
    _fail(
        "unsupported binding request type",
        "admission.unsupported_binding",
        value_type=type(request).__name__,
    )


def _capture_regular_file(*, name: str, path: Path) -> dict[str, Any]:
    _require_nonempty_string(name, field="binding_name")
    resolved = _resolve_existing_leaf(path, field=name, expected="file")
    encoded = _read_stable_regular_file(resolved, field=name)
    return {
        "kind": "regular_file",
        "name": name,
        "path": str(resolved),
        "byte_count": len(encoded),
        "sha256": _bytes_sha256(encoded),
    }


def _capture_directory_tree(*, name: str, path: Path) -> dict[str, Any]:
    _require_nonempty_string(name, field="binding_name")
    resolved = _resolve_existing_leaf(path, field=name, expected="directory")
    inventory: list[dict[str, Any]] = []
    try:
        descendants = sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix())
    except OSError as exc:
        _fail(
            "directory tree cannot be inventoried",
            "admission.unreadable_directory",
            cause=exc,
            path=str(resolved),
        )
    for descendant in descendants:
        relative_path = descendant.relative_to(resolved).as_posix()
        try:
            metadata = descendant.lstat()
        except OSError as exc:
            _fail(
                "directory descendant cannot be inspected",
                "admission.unreadable_directory",
                cause=exc,
                path=str(descendant),
            )
        if stat.S_ISLNK(metadata.st_mode):
            _fail(
                "directory tree contains a symlink",
                "admission.symlink_rejected",
                path=str(descendant),
            )
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _fail(
                "directory tree contains a non-regular descendant",
                "admission.invalid_directory_descendant",
                path=str(descendant),
            )
        encoded = _read_stable_regular_file(descendant, field=name)
        inventory.append(
            {
                "relative_path": relative_path,
                "byte_count": len(encoded),
                "sha256": _bytes_sha256(encoded),
            }
        )
    return {
        "kind": "directory_tree",
        "name": name,
        "path": str(resolved),
        "files": inventory,
        "inventory_sha256": json_sha256(inventory),
    }


def _capture_resolved_data_file(request: ResolvedDataFileBinding) -> dict[str, Any]:
    _require_nonempty_string(request.name, field="binding_name")
    if not isinstance(request.declared_path, (str, Path)):
        _fail(
            "declared data path must be a string or Path",
            "admission.invalid_path",
            name=request.name,
        )
    declaration = str(request.declared_path)
    if not declaration:
        _fail("declared data path is empty", "admission.invalid_path")
    base = _require_absolute_directory(request.base_directory, field="base_directory")
    declared = Path(declaration)
    target = declared if declared.is_absolute() else base / declared
    file_identity = _capture_regular_file(name=request.name, path=target)
    file_identity.pop("name")
    file_identity.pop("kind")
    return {
        "kind": "resolved_data_file",
        "name": request.name,
        "declared_path": declaration,
        "declaration_is_absolute": declared.is_absolute(),
        "base_directory": str(base),
        "resolved_file": file_identity,
    }


def _capture_absolute_executable(request: AbsoluteExecutableBinding) -> dict[str, Any]:
    _require_nonempty_string(request.name, field="binding_name")
    path = _require_path(request.path, field=request.name)
    if not path.is_absolute():
        _fail(
            "executable path must be absolute and is never searched through PATH",
            "admission.executable_not_absolute",
            path=str(path),
        )
    identity = _capture_regular_file(name=request.name, path=path)
    if not os.access(identity["path"], os.X_OK):
        _fail(
            "bound executable is not executable",
            "admission.executable_not_executable",
            path=identity["path"],
        )
    identity["kind"] = "absolute_executable"
    return identity


def _capture_reserved_paths(
    requests: Sequence[ReservedOutputPath],
    *,
    require_absent: bool,
    admission_root: Path,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        _fail(
            "reserved output paths must be a sequence",
            "admission.invalid_reserved_paths",
        )
    captured: list[dict[str, Any]] = []
    names: set[str] = set()
    paths: set[str] = set()
    for request in requests:
        if not isinstance(request, ReservedOutputPath):
            _fail(
                "reserved output path has an unsupported type",
                "admission.invalid_reserved_paths",
            )
        _require_nonempty_string(request.name, field="reserved_output_name")
        path = _require_path(request.path, field=request.name)
        if not path.is_absolute():
            _fail(
                "reserved output paths must be absolute",
                "admission.reserved_path_not_absolute",
                path=str(path),
            )
        occupied_candidate = os.path.lexists(path)
        resolved = path.resolve(strict=False)
        if request.name in names:
            _fail(
                "reserved output names must be unique",
                "admission.duplicate_reserved_output",
                name=request.name,
            )
        if str(resolved) in paths:
            _fail(
                "reserved output paths must not alias",
                "admission.duplicate_reserved_output",
                path=str(resolved),
            )
        if _is_relative_to(admission_root, resolved):
            _fail(
                "reserved output path cannot contain the admission root",
                "admission.reserved_path_conflict",
                path=str(resolved),
            )
        if resolved in {
            admission_root / "bindings.json",
            admission_root / "journal",
            admission_root / "admission.json",
        }:
            _fail(
                "reserved output path conflicts with an admission artifact",
                "admission.reserved_path_conflict",
                path=str(resolved),
            )
        if require_absent and (occupied_candidate or os.path.lexists(resolved)):
            _fail(
                "reserved output path is already occupied",
                "admission.reserved_path_occupied",
                path=str(resolved),
            )
        names.add(request.name)
        paths.add(str(resolved))
        captured.append({"name": request.name, "path": str(resolved)})
    if not captured:
        _fail(
            "at least one reserved output path is required",
            "admission.invalid_reserved_paths",
        )
    return tuple(_freeze_strict_value(item) for item in captured)


def _capture_stage_payload(
    *,
    stage: str,
    evidence: StageEvidence,
    admission_id: str,
    admission_fingerprint: str,
    bindings: BindingManifest,
    reserved_output_paths: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(evidence, StageEvidence):
        _fail("stage evidence must be typed", "admission.invalid_stage_evidence")
    binding_by_name = {item["name"]: item for item in bindings.bindings}
    for role, name in (
        ("producer", evidence.producer_binding),
        ("validator", evidence.validator_binding),
    ):
        _require_nonempty_string(name, field=f"{role}_binding")
        binding = binding_by_name.get(name)
        if binding is None:
            _fail(
                f"{role} binding is not present in the immutable manifest",
                "admission.missing_stage_binding",
                role=role,
                name=name,
            )
        if binding["kind"] not in {
            "regular_file",
            "resolved_data_file",
            "absolute_executable",
        }:
            _fail(
                f"{role} binding does not identify an exact file artifact",
                "admission.invalid_stage_binding_kind",
                role=role,
                name=name,
            )
    if evidence.scientific_interpretation is not False:
        _fail(
            "stage evidence must make no scientific interpretation",
            "admission.scientific_interpretation_rejected",
        )
    if evidence.mechanics_statement != MECHANICS_STATEMENT:
        _fail(
            "stage evidence must carry the fixed mechanics-only statement",
            "admission.invalid_claim_boundary",
        )
    assertions = _validate_stage_assertions(stage, evidence.assertions)
    output_requests = evidence.output_files
    if not output_requests:
        _fail(
            "stage evidence must bind at least one output file",
            "admission.missing_output_file",
        )
    outputs: list[dict[str, Any]] = []
    output_names: set[str] = set()
    output_paths: set[str] = set()
    for request in output_requests:
        if not isinstance(request, RegularFileBinding):
            _fail(
                "stage outputs must use regular-file bindings",
                "admission.invalid_output_binding",
            )
        output = _capture_regular_file(name=request.name, path=request.path)
        if not _is_output_reserved(
            Path(output["path"]), reserved_output_paths=reserved_output_paths
        ):
            _fail(
                "stage output is outside every immutable reserved output path",
                "admission.output_outside_reserved_paths",
                path=output["path"],
            )
        if output["name"] in output_names or output["path"] in output_paths:
            _fail(
                "stage output identities must be unique",
                "admission.duplicate_output_binding",
            )
        output_names.add(output["name"])
        output_paths.add(output["path"])
        outputs.append(output)
    detail = _thaw_json_value(evidence.detail)
    validate_json_value(detail)
    return {
        "stage": stage,
        "admission_id": admission_id,
        "admission_fingerprint": admission_fingerprint,
        "binding_manifest_fingerprint": bindings.content_fingerprint,
        "producer_binding": evidence.producer_binding,
        "validator_binding": evidence.validator_binding,
        "output_files": outputs,
        "assertions": assertions,
        "detail": detail,
        "scientific_interpretation": False,
        "mechanics_statement": MECHANICS_STATEMENT,
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }


def _validate_stage_assertions(stage: str, value: Mapping[str, Any]) -> dict[str, Any]:
    assertions = _strict_mapping(value, field="assertions")
    if stage == "cpu_preflight":
        if set(assertions) != set(_CPU_ASSERTIONS):
            _fail(
                "CPU preflight assertions must use the fixed schema",
                "admission.invalid_stage_assertions",
            )
        for name, accepted in _CPU_ASSERTIONS.items():
            if type(assertions[name]) is not bool or assertions[name] is not accepted:
                _fail(
                    "CPU preflight assertion has an unaccepted value",
                    "admission.invalid_stage_assertions",
                    assertion=name,
                )
        return assertions
    expected = {*_VERTICAL_BOOLEAN_ASSERTIONS, "durable_work_item_count"}
    if set(assertions) != expected:
        _fail(
            "vertical smoke assertions must use the fixed schema",
            "admission.invalid_stage_assertions",
        )
    for name, accepted in _VERTICAL_BOOLEAN_ASSERTIONS.items():
        if type(assertions[name]) is not bool or assertions[name] is not accepted:
            _fail(
                "vertical smoke assertion has an unaccepted value",
                "admission.invalid_stage_assertions",
                assertion=name,
            )
    count = assertions["durable_work_item_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        _fail(
            "vertical smoke requires at least one durable work item",
            "admission.invalid_stage_assertions",
            assertion="durable_work_item_count",
        )
    return assertions


def _validate_persisted_stage_records(
    inspection: JournalInspection,
    bindings: BindingManifest,
    admission_id: str,
    reserved_output_paths: Sequence[Mapping[str, Any]],
) -> None:
    binding_by_name = {item["name"]: item for item in bindings.bindings}
    expected_admission_fingerprint = inspection.snapshot.plan_fingerprint
    for sequence, record in enumerate(inspection.records):
        if record.work_item_id != STAGES[sequence]:
            _fail(
                "persisted stages are not in fixed order",
                "admission.invalid_persisted_evidence",
            )
        payload = record.payload
        if not isinstance(payload, Mapping):
            _fail(
                "persisted stage payload is not a mapping",
                "admission.invalid_persisted_evidence",
            )
        expected_keys = {
            "stage",
            "admission_id",
            "admission_fingerprint",
            "binding_manifest_fingerprint",
            "producer_binding",
            "validator_binding",
            "output_files",
            "assertions",
            "detail",
            "scientific_interpretation",
            "mechanics_statement",
            "claim_boundary",
        }
        if set(payload) != expected_keys or (
            payload["stage"] != record.work_item_id
            or payload["admission_id"] != admission_id
            or payload["admission_fingerprint"] != expected_admission_fingerprint
            or payload["binding_manifest_fingerprint"]
            != bindings.content_fingerprint
            or payload["producer_binding"] not in binding_by_name
            or payload["validator_binding"] not in binding_by_name
            or payload["scientific_interpretation"] is not False
            or payload["mechanics_statement"] != MECHANICS_STATEMENT
            or _thaw_json_value(payload["claim_boundary"]) != _CLAIM_BOUNDARY
        ):
            _fail(
                "persisted stage envelope does not bind this admission",
                "admission.invalid_persisted_evidence",
            )
        for role in ("producer_binding", "validator_binding"):
            if binding_by_name[payload[role]]["kind"] not in {
                "regular_file",
                "resolved_data_file",
                "absolute_executable",
            }:
                _fail(
                    "persisted stage role does not resolve to a file artifact",
                    "admission.invalid_persisted_evidence",
                    role=role,
                )
        _validate_stage_assertions(record.work_item_id, payload["assertions"])
        outputs = payload["output_files"]
        if not isinstance(outputs, (list, tuple)) or not outputs:
            _fail(
                "persisted stage lacks output identities",
                "admission.invalid_persisted_evidence",
            )
        for output in outputs:
            if not isinstance(output, Mapping):
                _fail(
                    "persisted output identity is invalid",
                    "admission.invalid_persisted_evidence",
                )
            recaptured = _capture_regular_file(
                name=str(output.get("name")), path=Path(str(output.get("path")))
            )
            if not _is_output_reserved(
                Path(recaptured["path"]),
                reserved_output_paths=reserved_output_paths,
            ):
                _fail(
                    "persisted stage output is outside the reserved plan",
                    "admission.invalid_persisted_evidence",
                    path=recaptured["path"],
                )
            if recaptured != _thaw_json_value(output):
                _fail(
                    "stage output file changed after acceptance",
                    "admission.output_drift",
                    path=str(output.get("path")),
                )


def _validate_manifest(manifest: BindingManifest) -> None:
    if not isinstance(manifest, BindingManifest):
        _fail("bindings must be a captured manifest", "admission.invalid_manifest")
    mapping = manifest.to_mapping()
    core = {
        "admission_binding_schema_version": ADMISSION_SCHEMA_VERSION,
        "bindings": mapping["bindings"],
    }
    if manifest.content_fingerprint != json_sha256(core):
        _fail(
            "binding manifest fingerprint is invalid",
            "admission.invalid_manifest",
        )


def _load_binding_manifest(path: Path) -> BindingManifest:
    value = load_canonical_json(path)
    if not isinstance(value, Mapping) or set(value) != {
        "admission_binding_schema_version",
        "bindings",
        "content_fingerprint",
    }:
        _fail("published binding manifest is invalid", "admission.invalid_manifest")
    if value["admission_binding_schema_version"] != ADMISSION_SCHEMA_VERSION:
        _fail("published binding schema is unsupported", "admission.invalid_manifest")
    bindings = value["bindings"]
    if not isinstance(bindings, list):
        _fail("published bindings must be a list", "admission.invalid_manifest")
    manifest = BindingManifest(
        bindings=tuple(_freeze_strict_value(item) for item in bindings),
        content_fingerprint=str(value["content_fingerprint"]),
    )
    _validate_manifest(manifest)
    return manifest


def _request_from_binding(binding: Mapping[str, Any]) -> BindingRequest:
    kind = binding["kind"]
    name = str(binding["name"])
    if kind == "regular_file":
        return RegularFileBinding(name, Path(str(binding["path"])))
    if kind == "directory_tree":
        return DirectoryTreeBinding(name, Path(str(binding["path"])))
    if kind == "resolved_data_file":
        return ResolvedDataFileBinding(
            name,
            str(binding["declared_path"]),
            Path(str(binding["base_directory"])),
        )
    if kind == "absolute_executable":
        return AbsoluteExecutableBinding(name, Path(str(binding["path"])))
    if kind == "strict_value":
        return StrictValueBinding(name, _thaw_json_value(binding["value"]))
    _fail(
        "published manifest contains an unsupported binding kind",
        "admission.invalid_manifest",
        kind=str(kind),
    )


def _execution_identity(
    admission_id: str,
    bindings: BindingManifest,
    reserved: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "admission_schema_version": ADMISSION_SCHEMA_VERSION,
        "admission_id": admission_id,
        "binding_manifest_fingerprint": bindings.content_fingerprint,
        "reserved_output_paths": [_thaw_json_value(item) for item in reserved],
    }


def _load_reserved_plan(
    admission_root: Path,
    inspection: JournalInspection,
) -> tuple[Mapping[str, Any], ...]:
    plan = load_canonical_json(admission_root / "journal" / "plan.json")
    if not isinstance(plan, Mapping):
        _fail(
            "journal plan is not a mapping",
            "admission.invalid_persisted_evidence",
        )
    identity = plan.get("execution_identity")
    if not isinstance(identity, Mapping):
        _fail(
            "journal plan lacks an admission identity",
            "admission.invalid_persisted_evidence",
        )
    raw_reserved = identity.get("reserved_output_paths")
    if not isinstance(raw_reserved, list):
        _fail(
            "journal plan lacks reserved output paths",
            "admission.invalid_persisted_evidence",
        )
    requests: list[ReservedOutputPath] = []
    for item in raw_reserved:
        if not isinstance(item, Mapping) or set(item) != {"name", "path"}:
            _fail(
                "journal reserved output entry is invalid",
                "admission.invalid_persisted_evidence",
            )
        requests.append(
            ReservedOutputPath(name=str(item["name"]), path=Path(str(item["path"])))
        )
    reserved = _capture_reserved_paths(
        requests,
        require_absent=False,
        admission_root=admission_root,
    )
    if (
        [_thaw_json_value(item) for item in reserved] != raw_reserved
        or plan.get("execution_id") != inspection.snapshot.execution_id
    ):
        _fail(
            "journal reserved output plan identity is invalid",
            "admission.invalid_persisted_evidence",
        )
    return reserved


def _is_output_reserved(
    output_path: Path,
    *,
    reserved_output_paths: Sequence[Mapping[str, Any]],
) -> bool:
    return any(
        _is_relative_to(output_path, Path(str(item["path"])))
        for item in reserved_output_paths
    )


def _journal_plan_fingerprint(
    *,
    admission_id: str,
    bindings: BindingManifest,
    reserved: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> str:
    identity = _execution_identity(admission_id, bindings, reserved)
    core = {
        "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        "execution_id": admission_id,
        "execution_identity": identity,
        "execution_identity_fingerprint": json_sha256(identity),
        "expected_work_item_ids": list(STAGES),
        "context": dict(context),
        "context_fingerprint": json_sha256(context),
    }
    return json_sha256(core)


def _request_name(request: Any) -> str:
    if not isinstance(
        request,
        (
            RegularFileBinding,
            DirectoryTreeBinding,
            ResolvedDataFileBinding,
            AbsoluteExecutableBinding,
            StrictValueBinding,
        ),
    ):
        _fail(
            "unsupported binding request type",
            "admission.unsupported_binding",
            value_type=type(request).__name__,
        )
    _require_nonempty_string(request.name, field="binding_name")
    return request.name


def _require_absolute_directory(path: Path, *, field: str) -> Path:
    candidate = _require_path(path, field=field)
    if not candidate.is_absolute():
        _fail(
            "base directory must be absolute",
            "admission.path_not_absolute",
            field=field,
        )
    return _resolve_existing_leaf(candidate, field=field, expected="directory")


def _resolve_existing_leaf(path: Path, *, field: str, expected: str) -> Path:
    candidate = _require_path(path, field=field)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        _fail(
            "binding path is missing or unreadable",
            "admission.path_missing",
            cause=exc,
            field=field,
            path=str(candidate),
        )
    if stat.S_ISLNK(metadata.st_mode):
        _fail(
            "binding path must not be a symlink",
            "admission.symlink_rejected",
            field=field,
            path=str(candidate),
        )
    if expected == "file" and not stat.S_ISREG(metadata.st_mode):
        _fail(
            "binding path is not a regular file",
            "admission.path_wrong_kind",
            field=field,
            path=str(candidate),
        )
    if expected == "directory" and not stat.S_ISDIR(metadata.st_mode):
        _fail(
            "binding path is not a directory",
            "admission.path_wrong_kind",
            field=field,
            path=str(candidate),
        )
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        _fail(
            "binding path cannot be resolved",
            "admission.path_missing",
            cause=exc,
            field=field,
            path=str(candidate),
        )


def _read_stable_regular_file(path: Path, *, field: str) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            _fail(
                "binding path ceased to be a regular non-symlink file",
                "admission.path_wrong_kind",
                field=field,
                path=str(path),
            )
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            encoded = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
    except ResearchProbeAdmissionError:
        raise
    except OSError as exc:
        _fail(
            "binding file cannot be read",
            "admission.unreadable_file",
            cause=exc,
            field=field,
            path=str(path),
        )
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_opened_before = (
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_size,
        opened_before.st_mtime_ns,
    )
    identity_opened_after = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if not (
        identity_before
        == identity_opened_before
        == identity_opened_after
        == identity_after
    ) or len(encoded) != before.st_size:
        _fail(
            "binding file changed while its bytes were captured",
            "admission.concurrent_input_drift",
            field=field,
            path=str(path),
        )
    return encoded


def _require_identical_receipt(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        _fail(
            "existing admission receipt is unreadable",
            "admission.invalid_receipt",
            cause=exc,
        )
    if encoded != canonical_json_bytes(expected):
        _fail(
            "existing admission receipt differs from exact terminal projection",
            "admission.receipt_conflict",
            path=str(path),
        )


def _build_receipt_projection(
    *,
    root: Path,
    admission_id: str,
    bindings: BindingManifest,
    inspection: JournalInspection,
) -> dict[str, Any]:
    records = inspection.records
    stage_fingerprints = [
        {
            "stage": record.work_item_id,
            "payload_fingerprint": record.payload_fingerprint,
        }
        for record in records
    ]
    outputs = [
        {
            "stage": record.work_item_id,
            "files": _thaw_json_value(record.payload["output_files"]),
        }
        for record in records
    ]
    core = {
        "admission_schema_version": ADMISSION_SCHEMA_VERSION,
        "admission_id": admission_id,
        "status": "mechanically_admitted",
        "binding_manifest_fingerprint": bindings.content_fingerprint,
        "journal_plan_identity": {
            "plan_fingerprint": inspection.snapshot.plan_fingerprint,
            "content_sha256": load_canonical_json(root / "journal" / "plan.json")[
                "content_sha256"
            ],
        },
        "journal_terminal_identity": _capture_regular_file(
            name="journal_terminal", path=root / "journal" / "terminal.json"
        ),
        "stage_payload_fingerprints": stage_fingerprints,
        "output_files": outputs,
        "claim_boundary": dict(_CLAIM_BOUNDARY),
        "mechanics_statement": MECHANICS_STATEMENT,
    }
    return {**core, "content_sha256": json_sha256(core)}


def _strict_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(
            "admission value must be a mapping",
            "admission.invalid_strict_value",
            field=field,
        )
    try:
        validate_json_value(value)
        result = json.loads(canonical_json_bytes(value).decode("utf-8"))
    except ArtifactContractError as exc:
        _fail(
            "admission value is not recursively strict JSON",
            "admission.invalid_strict_value",
            cause=exc,
            field=field,
        )
    return result


def _freeze_strict_value(value: Any) -> Any:
    try:
        validate_json_value(value)
        copied = json.loads(canonical_json_bytes(value).decode("utf-8"))
    except ArtifactContractError as exc:
        _fail(
            "admission value is not recursively strict JSON",
            "admission.invalid_strict_value",
            cause=exc,
        )
    return _freeze_json_value(copied)


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json_value(item) for item in value]
    return value


def _resolved_candidate(path: Path, *, field: str) -> Path:
    candidate = _require_path(path, field=field)
    return candidate.resolve(strict=False)


def _require_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, Path):
        _fail(
            "admission path fields must be Path values",
            "admission.invalid_path",
            field=field,
            value_type=type(value).__name__,
        )
    return value


def _require_nonempty_string(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        _fail(
            "admission field must be a nonempty string",
            "admission.invalid_value",
            field=field,
        )


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_relative_to(path: Path, possible_parent: Path) -> bool:
    try:
        path.relative_to(possible_parent)
    except ValueError:
        return False
    return True


def _fail(
    message: str,
    code: str,
    *,
    cause: BaseException | None = None,
    **context: Any,
) -> None:
    raise ResearchProbeAdmissionError(
        message,
        code=code,
        context=context,
        cause=cause,
    )


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "MECHANICS_STATEMENT",
    "STAGES",
    "AbsoluteExecutableBinding",
    "AdmissionInspection",
    "BindingManifest",
    "DirectoryTreeBinding",
    "RegularFileBinding",
    "ResearchProbeAdmission",
    "ResearchProbeAdmissionError",
    "ReservedOutputPath",
    "ResolvedDataFileBinding",
    "StageEvidence",
    "StrictValueBinding",
    "capture_binding_manifest",
    "revalidate_binding_manifest",
]
