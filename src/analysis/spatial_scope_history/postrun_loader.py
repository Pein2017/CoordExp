"""Fail-closed disk loader for the frozen spatial-scope primary experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.analysis.spatial_scope_history.calibration import PrimaryScheduleArtifact
from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptRecord,
    CohortLedger,
    sha256_file,
)
from src.analysis.spatial_scope_history.execution_evidence import (
    ExecutionEvidenceEnvelope,
    TerminalCallOutputBundle,
)
from src.analysis.spatial_scope_history.merge import (
    NormalizedCallPrediction,
    normalize_call_prediction,
)
from src.analysis.spatial_scope_history.metrics import (
    MetricAdmissionContract,
    MetricReadinessIdentity,
    RowMetricRecord,
    admit_metric_schedule,
)
from src.analysis.spatial_scope_history.parse_score_evidence import (
    CanonicalParseScoreReceipt,
    build_canonical_parse_score_receipts,
)
from src.analysis.spatial_scope_history.schedule import (
    ResearchSchedule,
    ScheduledRequest,
)
from src.analysis.spatial_scope_history.spatial import (
    SpatialGrid,
    SpatialGridSpec,
    spatial_variant_mode_for_arm,
)
from src.common.errors import ArtifactContractError, DataContractError
from src.eval.detection_categories import COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME
from src.inference.backend import DecodeResult


@dataclass(frozen=True)
class LoadedTerminalCall:
    """One exact scheduled call rehydrated from its terminal output bundle."""

    request: ScheduledRequest
    attempt: AttemptRecord
    bundle_path: str
    bundle: TerminalCallOutputBundle
    execution_evidence: ExecutionEvidenceEnvelope | None
    decode_result: DecodeResult | None
    parse_score_receipts: tuple[CanonicalParseScoreReceipt, ...]
    normalized_predictions: tuple[NormalizedCallPrediction, ...]
    row_diagnostics: tuple[RowMetricRecord, ...]
    call_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedPostrunEvidence:
    """Complete five-arm terminal universe admitted for metric assembly."""

    schedule_path: str
    cohort_path: str
    attempt_ledger_path: str
    readiness_root: str
    primary_schedule_artifact: PrimaryScheduleArtifact
    primary_schedule_artifact_file_sha256: str
    schedule: ResearchSchedule
    cohort: CohortLedger
    attempt_ledger: AttemptLedger
    readiness: MetricReadinessIdentity
    admission: MetricAdmissionContract
    calls: tuple[LoadedTerminalCall, ...]

    @property
    def calls_by_request_id(self) -> Mapping[str, LoadedTerminalCall]:
        return {call.request.request_id: call for call in self.calls}


def load_postrun_evidence(
    *,
    schedule_path: str | Path,
    cohort_path: str | Path,
    attempt_ledger_path: str | Path,
    readiness_root: str | Path,
) -> LoadedPostrunEvidence:
    """Load only explicitly named current-run artifacts; never discover outputs."""

    resolved_schedule_path = _absolute_existing_path(schedule_path, "schedule")
    resolved_cohort_path = _absolute_existing_path(cohort_path, "cohort")
    resolved_attempt_path = _absolute_existing_path(
        attempt_ledger_path, "attempt ledger"
    )
    resolved_readiness_root = _absolute_existing_path(
        readiness_root, "readiness root", directory=True
    )
    schedule_payload = _read_json_object(
        resolved_schedule_path, artifact_name="primary schedule artifact"
    )
    primary_schedule_artifact = PrimaryScheduleArtifact.from_artifact_dict(
        schedule_payload
    )
    schedule = primary_schedule_artifact.schedule
    cohort = CohortLedger.from_jsonl_bytes(resolved_cohort_path.read_bytes())
    observed_cohort_sha256 = sha256_file(resolved_cohort_path)
    source_hashes = dict(primary_schedule_artifact.source_hashes)
    if (
        primary_schedule_artifact.cohort_artifact_sha256 != observed_cohort_sha256
        or source_hashes.get(primary_schedule_artifact.cohort_artifact_name)
        != observed_cohort_sha256
    ):
        _fail(
            "cohort artifact differs from the canonical primary schedule wrapper",
            "analysis.postrun_primary_schedule_cohort",
            cohort_artifact_name=primary_schedule_artifact.cohort_artifact_name,
        )
    attempt_ledger = AttemptLedger.from_jsonl_bytes(
        resolved_attempt_path.read_bytes(),
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
    )
    readiness = MetricReadinessIdentity.from_verified_artifacts(
        readiness_root=resolved_readiness_root,
        spatial_grid_spec=SpatialGridSpec(),
    )
    grids = {
        str(record.image_id): SpatialGrid.build(
            source_width=record.source_width,
            source_height=record.source_height,
        )
        for record in cohort.records
    }
    admission = admit_metric_schedule(
        schedule=schedule,
        cohort=cohort,
        readiness=readiness,
        attempt_ledger=attempt_ledger,
        spatial_grids_by_image_id=grids,
    )
    attempts = attempt_ledger.records_by_request_id
    calls = tuple(
        _load_terminal_call(request=request, attempt=attempts[request.request_id])
        for request in schedule.requests
    )
    return LoadedPostrunEvidence(
        schedule_path=str(resolved_schedule_path),
        cohort_path=str(resolved_cohort_path),
        attempt_ledger_path=str(resolved_attempt_path),
        readiness_root=str(resolved_readiness_root),
        primary_schedule_artifact=primary_schedule_artifact,
        primary_schedule_artifact_file_sha256=sha256_file(resolved_schedule_path),
        schedule=schedule,
        cohort=cohort,
        attempt_ledger=attempt_ledger,
        readiness=readiness,
        admission=admission,
        calls=calls,
    )


def _load_terminal_call(
    *, request: ScheduledRequest, attempt: AttemptRecord
) -> LoadedTerminalCall:
    if attempt.output_artifact_path is None or attempt.output_artifact_sha256 is None:
        _fail(
            "terminal attempt does not bind an output bundle",
            "analysis.postrun_terminal_bundle_missing",
            request_id=request.request_id,
        )
    output_path = Path(attempt.output_artifact_path)
    if not output_path.is_absolute():
        _fail(
            "terminal output bundle path must be absolute",
            "analysis.postrun_terminal_bundle_absolute",
            request_id=request.request_id,
            output_artifact_path=attempt.output_artifact_path,
        )
    resolved_output_path = output_path.resolve(strict=True)
    if str(resolved_output_path) != attempt.output_artifact_path:
        _fail(
            "terminal output bundle path was relocated or resolved through an alias",
            "analysis.postrun_terminal_bundle_relocated",
            request_id=request.request_id,
            recorded_path=attempt.output_artifact_path,
            resolved_path=str(resolved_output_path),
        )
    bundle = TerminalCallOutputBundle.from_path(resolved_output_path)
    if bundle.bundle_sha256 != attempt.output_artifact_sha256:
        _fail(
            "terminal output bundle bytes differ from the attempt ledger",
            "analysis.postrun_terminal_bundle_digest",
            request_id=request.request_id,
        )
    payload = bundle.payload
    if (
        payload.get("request_id") != request.request_id
        or payload.get("scheduled_request") != request.to_artifact_dict()
        or payload.get("attempt_status") != attempt.attempt_status
        or payload.get("failure_code") != attempt.failure_code
    ):
        _fail(
            "terminal bundle differs from its exact schedule and attempt record",
            "analysis.postrun_terminal_bundle_join",
            request_id=request.request_id,
        )
    call_diagnostics = payload.get("call_diagnostics")
    if not isinstance(call_diagnostics, Mapping):
        _fail(
            "terminal call diagnostics must be an object",
            "analysis.postrun_call_diagnostics",
            request_id=request.request_id,
        )
    row_payloads = payload.get("row_diagnostics")
    receipt_payloads = payload.get("parse_score_receipts")
    if not isinstance(row_payloads, list) or not isinstance(receipt_payloads, list):
        _fail(
            "terminal row and parse evidence must be arrays",
            "analysis.postrun_terminal_rows",
            request_id=request.request_id,
        )
    if attempt.attempt_status != "completed":
        if row_payloads or receipt_payloads:
            _fail(
                "non-completed terminal call carries metric rows",
                "analysis.postrun_noncompleted_rows",
                request_id=request.request_id,
            )
        return LoadedTerminalCall(
            request=request,
            attempt=attempt,
            bundle_path=str(resolved_output_path),
            bundle=bundle,
            execution_evidence=None,
            decode_result=None,
            parse_score_receipts=(),
            normalized_predictions=(),
            row_diagnostics=(),
            call_diagnostics=dict(call_diagnostics),
        )

    execution_payload = payload.get("execution_evidence")
    decode_payload = payload.get("decode_result")
    if not isinstance(execution_payload, Mapping) or not isinstance(
        decode_payload, Mapping
    ):
        _fail(
            "completed terminal call lacks typed execution or decode evidence",
            "analysis.postrun_completed_evidence",
            request_id=request.request_id,
        )
    execution = ExecutionEvidenceEnvelope.from_artifact_dict(execution_payload)
    if execution.scheduled_request != request:
        _fail(
            "execution envelope differs from the scheduled request",
            "analysis.postrun_execution_schedule",
            request_id=request.request_id,
        )
    decode_result = DecodeResult.from_artifact_dict(decode_payload)
    decode_result.validate_for_scored()
    if decode_result.to_artifact_dict() != dict(decode_payload):
        _fail(
            "decode result does not round-trip through its typed representation",
            "analysis.postrun_decode_roundtrip",
            request_id=request.request_id,
        )
    stored_receipts = tuple(
        CanonicalParseScoreReceipt.from_artifact_dict(item)
        for item in receipt_payloads
    )
    replayed_receipts = tuple(
        receipt
        for receipt in build_canonical_parse_score_receipts(
            execution_evidence=execution,
            decode_result=decode_result,
        )
        if receipt.normalized_category_name
        in COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME
    )
    if tuple(item.to_artifact_dict() for item in stored_receipts) != tuple(
        item.to_artifact_dict() for item in replayed_receipts
    ):
        _fail(
            "stored parse-and-score receipts differ from canonical replay",
            "analysis.postrun_parse_score_replay",
            request_id=request.request_id,
        )
    normalized = tuple(
        _normalize_receipt(execution=execution, receipt=receipt)
        for receipt in stored_receipts
    )
    rows = tuple(_row_metric_record_from_payload(item) for item in row_payloads)
    if tuple(row.to_json_dict() for row in rows) != tuple(row_payloads):
        _fail(
            "row diagnostics do not round-trip through the typed record",
            "analysis.postrun_row_diagnostic_roundtrip",
            request_id=request.request_id,
        )
    return LoadedTerminalCall(
        request=request,
        attempt=attempt,
        bundle_path=str(resolved_output_path),
        bundle=bundle,
        execution_evidence=execution,
        decode_result=decode_result,
        parse_score_receipts=stored_receipts,
        normalized_predictions=normalized,
        row_diagnostics=rows,
        call_diagnostics=dict(call_diagnostics),
    )


def _normalize_receipt(
    *,
    execution: ExecutionEvidenceEnvelope,
    receipt: CanonicalParseScoreReceipt,
) -> NormalizedCallPrediction:
    mode = spatial_variant_mode_for_arm(execution.arm)
    coordinate_receipt = None
    if mode is not None:
        if execution.canonical_cell_index is None:
            _fail(
                "spatial execution lacks a cell index",
                "analysis.postrun_spatial_cell",
                request_id=execution.request_id,
            )
        coordinate_receipt = SpatialGrid.build(
            source_width=execution.source_width,
            source_height=execution.source_height,
        ).plan(
            cell_index=execution.canonical_cell_index,
            variant_mode=mode,
        ).coordinate_receipt(receipt.coordinate_bins)
    return normalize_call_prediction(
        execution_evidence=execution,
        parse_score_receipt=receipt,
        spatial_coordinate_receipt=coordinate_receipt,
    )


def _row_metric_record_from_payload(value: object) -> RowMetricRecord:
    fields = {
        "canonical_call_id",
        "generated_row_index",
        "matched_reference_ids",
        "ownership_status",
        "parse_status",
        "prediction_id",
        "validity_status",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(
            "row diagnostic keys are not exact",
            "analysis.postrun_row_diagnostic_keys",
        )
    payload = dict(value)
    payload["matched_reference_ids"] = tuple(payload["matched_reference_ids"])
    return RowMetricRecord(**payload)


def _absolute_existing_path(
    value: str | Path, name: str, *, directory: bool = False
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        _fail(
            f"{name} must be an explicit absolute path",
            "analysis.postrun_absolute_path",
            artifact_name=name,
            path=str(path),
        )
    resolved = path.resolve(strict=True)
    if directory and not resolved.is_dir():
        _fail(
            f"{name} must be a directory",
            "analysis.postrun_artifact_type",
            path=str(resolved),
        )
    if not directory and not resolved.is_file():
        _fail(
            f"{name} must be a file",
            "analysis.postrun_artifact_type",
            path=str(resolved),
        )
    return resolved


def _read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(
            f"{artifact_name} is not readable JSON",
            code="analysis.postrun_json",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(value, dict):
        _fail(
            f"{artifact_name} must be a JSON object",
            "analysis.postrun_json_type",
            path=str(path),
        )
    return value


def _fail(message: str, code: str, **context: Any) -> None:
    raise DataContractError(message, code=code, context=context)
