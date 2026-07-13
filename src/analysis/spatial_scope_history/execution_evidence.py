"""Canonical execution-evidence envelope for normalized research predictions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from PIL import Image
import torch

from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptDependencyContract,
    AttemptRecord,
    canonical_json_text,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    GridProvenance,
    PhysicalBatchPlan,
    ResearchArmDefinition,
    RequestBatch,
    ScheduledRequest,
)
from src.analysis.spatial_scope_history.spatial import (
    ExecutedVisualTensorReceipt,
    MaterializedVisualInput,
    TensorContentReceipt,
    spatial_variant_mode_for_arm,
)
from src.common.errors import DataContractError
from src.inference.backend import (
    DecodeExecutionReceipt,
    DecodeRequest,
    DecodeResult,
    batch_request_order_fingerprint,
)
from src.inference.image_plan import ImagePlanRow
from src.inference.prompt import IMAGE_PLACEHOLDER, PromptRecord


EXECUTION_EVIDENCE_SCHEMA_VERSION = "spatial_scope_history.execution_evidence.v2"
TERMINAL_CALL_OUTPUT_BUNDLE_SCHEMA_VERSION = (
    "spatial_scope_history.terminal_call_output_bundle.v1"
)
EXECUTED_PROMPT_EVIDENCE_SCHEMA_VERSION = (
    "spatial_scope_history.executed_prompt_evidence.v1"
)
EXECUTED_REQUEST_BATCH_EVIDENCE_SCHEMA_VERSION = (
    "spatial_scope_history.executed_request_batch_evidence.v1"
)
CUMULATIVE_STATE_DEPENDENCY_EVIDENCE_SCHEMA_VERSION = (
    "spatial_scope_history.cumulative_state_dependency_evidence.v1"
)
ExecutionEvidenceKind = Literal[
    "full_image_materialization_receipt",
    "spatial_materialization_receipt",
]


@dataclass(frozen=True)
class ExecutedPromptEvidence:
    """Exact prompt text, record, and full token identifiers sent to one request."""

    prompt_record_payload_json: str
    prompt_record_sha256: str
    full_prompt_fingerprint: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str
    continuation_text_sha256: str | None
    schema_version: str = EXECUTED_PROMPT_EVIDENCE_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        prompt_record: PromptRecord,
        decode_request: DecodeRequest,
    ) -> ExecutedPromptEvidence:
        token_ids = tuple(int(value) for value in prompt_record.prompt_token_ids)
        if token_ids != tuple(int(value) for value in decode_request.prompt_token_ids):
            _fail(
                "prompt record token identifiers differ from the executed request",
                "analysis.execution_evidence_prompt_tokens",
            )
        payload = {
            **prompt_record.to_artifact_dict(),
            "full_chat_text": prompt_record.chat_text,
        }
        payload_json = canonical_json_text(payload)
        return cls(
            prompt_record_payload_json=payload_json,
            prompt_record_sha256=hashlib.sha256(
                payload_json.encode("utf-8")
            ).hexdigest(),
            full_prompt_fingerprint=prompt_record.full_prompt_fingerprint,
            prompt_token_ids=token_ids,
            prompt_token_ids_sha256=sha256_payload(list(token_ids)),
            continuation_text_sha256=prompt_record.continuation_text_sha256,
        )

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTED_PROMPT_EVIDENCE_SCHEMA_VERSION:
            _fail(
                "executed prompt evidence schema is unsupported",
                "analysis.execution_evidence_prompt_schema",
            )
        try:
            payload = json.loads(self.prompt_record_payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DataContractError(
                "executed prompt evidence payload must be canonical JSON",
                code="analysis.execution_evidence_prompt_payload",
                cause=exc,
            ) from exc
        if canonical_json_text(payload) != self.prompt_record_payload_json:
            _fail(
                "executed prompt evidence payload is not canonical",
                "analysis.execution_evidence_prompt_payload",
            )
        if (
            hashlib.sha256(self.prompt_record_payload_json.encode("utf-8")).hexdigest()
            != self.prompt_record_sha256
        ):
            _fail(
                "executed prompt evidence digest does not bind its payload",
                "analysis.execution_evidence_prompt_digest",
            )
        if payload.get("full_prompt_fingerprint") != self.full_prompt_fingerprint:
            _fail(
                "executed prompt fingerprint differs from the prompt record",
                "analysis.execution_evidence_prompt_fingerprint",
            )
        if self.full_prompt_fingerprint != _full_prompt_fingerprint(
            full_chat_text=payload.get("full_chat_text"),
            prompt_token_ids=self.prompt_token_ids,
        ):
            _fail(
                "executed prompt fingerprint does not bind full text and tokens",
                "analysis.execution_evidence_prompt_fingerprint",
            )
        if tuple(payload.get("prompt_token_ids", ())) != self.prompt_token_ids:
            _fail(
                "executed prompt token identifiers differ from the prompt record",
                "analysis.execution_evidence_prompt_tokens",
            )
        if self.prompt_token_ids_sha256 != sha256_payload(list(self.prompt_token_ids)):
            _fail(
                "executed prompt token digest does not bind the full token sequence",
                "analysis.execution_evidence_prompt_token_digest",
            )
        for field in (
            "prompt_record_sha256",
            "full_prompt_fingerprint",
            "prompt_token_ids_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.continuation_text_sha256 is not None:
            _require_sha256(
                self.continuation_text_sha256,
                field="continuation_text_sha256",
            )
            _validate_continuation_prompt_payload(
                payload=payload,
                continuation_text_sha256=self.continuation_text_sha256,
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "continuation_text_sha256": self.continuation_text_sha256,
            "full_prompt_fingerprint": self.full_prompt_fingerprint,
            "prompt_record_payload_json": self.prompt_record_payload_json,
            "prompt_record_sha256": self.prompt_record_sha256,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> ExecutedPromptEvidence:
        fields = {
            "continuation_text_sha256",
            "full_prompt_fingerprint",
            "prompt_record_payload_json",
            "prompt_record_sha256",
            "prompt_token_ids",
            "prompt_token_ids_sha256",
            "schema_version",
        }
        _require_exact_artifact_keys(
            value, fields=fields, record_name="executed prompt evidence"
        )
        payload = dict(value)
        payload["prompt_token_ids"] = tuple(payload["prompt_token_ids"])
        return cls(**payload)


@dataclass(frozen=True)
class ExecutedRequestBatchEvidence:
    """Sealed physical membership and actual backend request order for one call."""

    schedule_identity_sha256: str
    physical_batch_plan_sha256: str
    physical_batch_sha256: str
    physical_batch_index: int
    batch_request_ids: tuple[str, ...]
    batch_cardinality: int
    request_execution_index: int
    backend_batch_order_fingerprint: str
    batch_prompt_token_ids_sha256: tuple[str, ...]
    evidence_sha256: str
    schema_version: str = EXECUTED_REQUEST_BATCH_EVIDENCE_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        scheduled_request: ScheduledRequest,
        request_batch: RequestBatch,
        physical_batch_plan: PhysicalBatchPlan,
        decode_request_batch: Sequence[DecodeRequest],
        decode_receipt: DecodeExecutionReceipt,
    ) -> ExecutedRequestBatchEvidence:
        _validate_sealed_request_batch(
            scheduled_request=scheduled_request,
            request_batch=request_batch,
            physical_batch_plan=physical_batch_plan,
        )
        ordered_decode_requests = tuple(decode_request_batch)
        decode_ids = tuple(request.request_id for request in ordered_decode_requests)
        if decode_ids != request_batch.request_ids:
            _fail(
                "executed backend request order differs from the sealed physical batch",
                "analysis.execution_evidence_batch_order",
            )
        try:
            execution_index = decode_ids.index(scheduled_request.request_id)
        except ValueError:
            _fail(
                "scheduled request is absent from its executed physical batch",
                "analysis.execution_evidence_batch_membership",
            )
        backend_fingerprint = batch_request_order_fingerprint(ordered_decode_requests)
        if decode_receipt.request_execution_index != execution_index:
            _fail(
                "decode receipt execution index differs from sealed batch order",
                "analysis.execution_evidence_batch_execution_index",
            )
        if decode_receipt.batch_request_order_fingerprint != backend_fingerprint:
            _fail(
                "decode receipt batch fingerprint differs from actual ordered requests",
                "analysis.execution_evidence_backend_batch_fingerprint",
            )
        prompt_hashes = tuple(
            sha256_payload([int(value) for value in request.prompt_token_ids])
            for request in ordered_decode_requests
        )
        identity = {
            "backend_batch_order_fingerprint": backend_fingerprint,
            "batch_cardinality": len(ordered_decode_requests),
            "batch_prompt_token_ids_sha256": list(prompt_hashes),
            "batch_request_ids": list(decode_ids),
            "physical_batch_index": request_batch.batch_index,
            "physical_batch_plan_sha256": physical_batch_plan.fingerprint,
            "physical_batch_sha256": request_batch.physical_batch_sha256,
            "request_execution_index": execution_index,
            "schedule_identity_sha256": physical_batch_plan.schedule_identity_sha256,
            "schema_version": EXECUTED_REQUEST_BATCH_EVIDENCE_SCHEMA_VERSION,
        }
        return cls(
            schedule_identity_sha256=physical_batch_plan.schedule_identity_sha256,
            physical_batch_plan_sha256=physical_batch_plan.fingerprint,
            physical_batch_sha256=request_batch.physical_batch_sha256,
            physical_batch_index=request_batch.batch_index,
            batch_request_ids=decode_ids,
            batch_cardinality=len(decode_ids),
            request_execution_index=execution_index,
            backend_batch_order_fingerprint=backend_fingerprint,
            batch_prompt_token_ids_sha256=prompt_hashes,
            evidence_sha256=sha256_payload(identity),
        )

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTED_REQUEST_BATCH_EVIDENCE_SCHEMA_VERSION:
            _fail(
                "executed request-batch evidence schema is unsupported",
                "analysis.execution_evidence_batch_schema",
            )
        if self.batch_cardinality not in {3, 4} or self.batch_cardinality != len(
            self.batch_request_ids
        ):
            _fail(
                "executed request batch must be the primary four or authorized natural tail of three",
                "analysis.execution_evidence_batch_cardinality",
            )
        if len(self.batch_prompt_token_ids_sha256) != self.batch_cardinality:
            _fail(
                "executed request batch prompt evidence has the wrong cardinality",
                "analysis.execution_evidence_batch_prompt_cardinality",
            )
        if not 0 <= self.request_execution_index < self.batch_cardinality:
            _fail(
                "executed request index is outside the physical batch",
                "analysis.execution_evidence_batch_execution_index",
            )
        for field_name in (
            "schedule_identity_sha256",
            "physical_batch_plan_sha256",
            "physical_batch_sha256",
            "backend_batch_order_fingerprint",
            "evidence_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        for prompt_sha256 in self.batch_prompt_token_ids_sha256:
            _require_sha256(
                prompt_sha256,
                field="batch_prompt_token_ids_sha256",
            )
        if self.evidence_sha256 != sha256_payload(self.identity_payload()):
            _fail(
                "executed request-batch evidence digest does not bind its identity",
                "analysis.execution_evidence_batch_digest",
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "backend_batch_order_fingerprint": self.backend_batch_order_fingerprint,
            "batch_cardinality": self.batch_cardinality,
            "batch_prompt_token_ids_sha256": list(self.batch_prompt_token_ids_sha256),
            "batch_request_ids": list(self.batch_request_ids),
            "physical_batch_index": self.physical_batch_index,
            "physical_batch_plan_sha256": self.physical_batch_plan_sha256,
            "physical_batch_sha256": self.physical_batch_sha256,
            "request_execution_index": self.request_execution_index,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "schema_version": self.schema_version,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "evidence_sha256": self.evidence_sha256}

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> ExecutedRequestBatchEvidence:
        fields = {
            "backend_batch_order_fingerprint",
            "batch_cardinality",
            "batch_prompt_token_ids_sha256",
            "batch_request_ids",
            "evidence_sha256",
            "physical_batch_index",
            "physical_batch_plan_sha256",
            "physical_batch_sha256",
            "request_execution_index",
            "schedule_identity_sha256",
            "schema_version",
        }
        _require_exact_artifact_keys(
            value, fields=fields, record_name="executed request batch evidence"
        )
        payload = dict(value)
        payload["batch_prompt_token_ids_sha256"] = tuple(
            payload["batch_prompt_token_ids_sha256"]
        )
        payload["batch_request_ids"] = tuple(payload["batch_request_ids"])
        return cls(**payload)


@dataclass(frozen=True)
class CumulativeStateDependencyEvidence:
    """Expected accepted-row state and exact prompt binding for a cumulative call."""

    request_id: str
    predecessor_request_id: str | None
    expected_cumulative_state_sha256: str
    predecessor_attempt_sha256: str | None
    predecessor_state_artifact_path: str | None
    predecessor_state_artifact_sha256: str | None
    prompt_record_sha256: str
    full_prompt_fingerprint: str
    evidence_sha256: str
    schema_version: str = CUMULATIVE_STATE_DEPENDENCY_EVIDENCE_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        scheduled_request: ScheduledRequest,
        dependency_contract: AttemptDependencyContract,
        prompt_evidence: ExecutedPromptEvidence,
        predecessor_attempt: AttemptRecord | None,
    ) -> CumulativeStateDependencyEvidence:
        if not scheduled_request.arm.cumulative_dependency:
            _fail(
                "non-cumulative request cannot build cumulative-state evidence",
                "analysis.execution_evidence_unexpected_cumulative_state",
            )
        predecessor_attempt_sha256: str | None = None
        artifact_path: str | None = None
        artifact_sha256: str | None = None
        if scheduled_request.predecessor_request_id is None:
            expected_state = scheduled_request.initial_cumulative_state_sha256
            if (
                predecessor_attempt is not None
                or prompt_evidence.continuation_text_sha256 is not None
            ):
                _fail(
                    "first cumulative request must use the empty state and fresh prompt",
                    "analysis.execution_evidence_cumulative_initial_prompt",
                )
        else:
            if predecessor_attempt is None:
                _fail(
                    "later cumulative request requires its completed predecessor attempt",
                    "analysis.execution_evidence_cumulative_predecessor_missing",
                )
            if (
                predecessor_attempt.request_id
                != scheduled_request.predecessor_request_id
                or predecessor_attempt.attempt_status != "completed"
                or predecessor_attempt.produced_cumulative_state_sha256 is None
                or predecessor_attempt.produced_cumulative_state_artifact_path is None
            ):
                _fail(
                    "cumulative predecessor identity or produced state is invalid",
                    "analysis.execution_evidence_cumulative_predecessor",
                )
            path = Path(predecessor_attempt.produced_cumulative_state_artifact_path)
            if not path.is_file():
                _fail(
                    "cumulative predecessor state artifact is unavailable",
                    "analysis.execution_evidence_cumulative_artifact_missing",
                )
            observed_digest = sha256_file(path)
            if observed_digest != predecessor_attempt.produced_cumulative_state_sha256:
                _fail(
                    "cumulative predecessor state artifact bytes differ from its digest",
                    "analysis.execution_evidence_cumulative_artifact_digest",
                )
            expected_state = observed_digest
            artifact_path = str(path)
            artifact_sha256 = observed_digest
            predecessor_attempt_sha256 = sha256_payload(
                predecessor_attempt.to_artifact_dict()
            )
            accepted_row_prefix = _accepted_row_prefix_from_state_artifact(path)
            expected_continuation_sha256 = (
                hashlib.sha256(accepted_row_prefix.encode("utf-8")).hexdigest()
                if accepted_row_prefix
                else None
            )
            if prompt_evidence.continuation_text_sha256 != expected_continuation_sha256:
                _fail(
                    "cumulative prompt continuation differs from the predecessor accepted-row state",
                    "analysis.execution_evidence_cumulative_prompt_state",
                )
        if expected_state is None:
            _fail(
                "cumulative request has no expected state identity",
                "analysis.execution_evidence_cumulative_state_missing",
            )
        if (
            dependency_contract.request_id != scheduled_request.request_id
            or dependency_contract.predecessor_request_id
            != scheduled_request.predecessor_request_id
            or dependency_contract.initial_cumulative_state_sha256
            != scheduled_request.initial_cumulative_state_sha256
        ):
            _fail(
                "attempt dependency differs from the scheduled cumulative origin",
                "analysis.execution_evidence_dependency_contract",
            )
        identity = {
            "expected_cumulative_state_sha256": expected_state,
            "full_prompt_fingerprint": prompt_evidence.full_prompt_fingerprint,
            "predecessor_attempt_sha256": predecessor_attempt_sha256,
            "predecessor_request_id": scheduled_request.predecessor_request_id,
            "predecessor_state_artifact_path": artifact_path,
            "predecessor_state_artifact_sha256": artifact_sha256,
            "prompt_record_sha256": prompt_evidence.prompt_record_sha256,
            "request_id": scheduled_request.request_id,
            "schema_version": CUMULATIVE_STATE_DEPENDENCY_EVIDENCE_SCHEMA_VERSION,
        }
        return cls(
            request_id=scheduled_request.request_id,
            predecessor_request_id=scheduled_request.predecessor_request_id,
            expected_cumulative_state_sha256=expected_state,
            predecessor_attempt_sha256=predecessor_attempt_sha256,
            predecessor_state_artifact_path=artifact_path,
            predecessor_state_artifact_sha256=artifact_sha256,
            prompt_record_sha256=prompt_evidence.prompt_record_sha256,
            full_prompt_fingerprint=prompt_evidence.full_prompt_fingerprint,
            evidence_sha256=sha256_payload(identity),
        )

    def __post_init__(self) -> None:
        if self.schema_version != CUMULATIVE_STATE_DEPENDENCY_EVIDENCE_SCHEMA_VERSION:
            _fail(
                "cumulative-state dependency evidence schema is unsupported",
                "analysis.execution_evidence_cumulative_schema",
            )
        for field in (
            "expected_cumulative_state_sha256",
            "prompt_record_sha256",
            "full_prompt_fingerprint",
            "evidence_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        for field in (
            "predecessor_attempt_sha256",
            "predecessor_state_artifact_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                _require_sha256(value, field=field)
        if self.evidence_sha256 != sha256_payload(self.identity_payload()):
            _fail(
                "cumulative-state dependency evidence digest is invalid",
                "analysis.execution_evidence_cumulative_digest",
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "expected_cumulative_state_sha256": self.expected_cumulative_state_sha256,
            "full_prompt_fingerprint": self.full_prompt_fingerprint,
            "predecessor_attempt_sha256": self.predecessor_attempt_sha256,
            "predecessor_request_id": self.predecessor_request_id,
            "predecessor_state_artifact_path": self.predecessor_state_artifact_path,
            "predecessor_state_artifact_sha256": self.predecessor_state_artifact_sha256,
            "prompt_record_sha256": self.prompt_record_sha256,
            "request_id": self.request_id,
            "schema_version": self.schema_version,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "evidence_sha256": self.evidence_sha256}

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> CumulativeStateDependencyEvidence:
        fields = {
            "evidence_sha256",
            "expected_cumulative_state_sha256",
            "full_prompt_fingerprint",
            "predecessor_attempt_sha256",
            "predecessor_request_id",
            "predecessor_state_artifact_path",
            "predecessor_state_artifact_sha256",
            "prompt_record_sha256",
            "request_id",
            "schema_version",
        }
        _require_exact_artifact_keys(
            value, fields=fields, record_name="cumulative state dependency evidence"
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class ExecutionEvidenceEnvelope:
    """One immutable chain from scheduled request through executed output."""

    scheduled_request: ScheduledRequest
    request_fingerprint: str
    grid_provenance: GridProvenance
    decode_provenance: DecodeProvenance
    executed_prompt_evidence: ExecutedPromptEvidence
    executed_visual_tensor_receipt: ExecutedVisualTensorReceipt
    executed_request_batch_evidence: ExecutedRequestBatchEvidence
    attempt_dependency_payload_json: str
    attempt_dependency_sha256: str
    cumulative_state_dependency_evidence: CumulativeStateDependencyEvidence | None
    evidence_kind: ExecutionEvidenceKind
    source_image_sha256: str
    source_width: int
    source_height: int
    processor_receipt_sha256: str
    processor_receipt_payload_json: str
    spatial_image_encoding_sha256: str | None
    full_image_plan_sha256: str | None
    decode_execution_receipt: DecodeExecutionReceipt
    decode_receipt_fingerprint: str
    generated_output_token_sha256: str
    token_trace_sha256: str
    envelope_fingerprint: str | None = None
    schema_version: str = EXECUTION_EVIDENCE_SCHEMA_VERSION

    @classmethod
    def from_spatial_execution(
        cls,
        *,
        scheduled_request: ScheduledRequest,
        grid_provenance: GridProvenance,
        decode_provenance: DecodeProvenance,
        prompt_record: PromptRecord,
        decode_request_batch: Sequence[DecodeRequest],
        request_batch: RequestBatch,
        physical_batch_plan: PhysicalBatchPlan,
        attempt_dependency: AttemptDependencyContract,
        predecessor_attempt: AttemptRecord | None,
        visual_materialization: MaterializedVisualInput,
        decode_result: DecodeResult,
    ) -> ExecutionEvidenceEnvelope:
        """Build evidence for tile or masked-canvas execution."""

        receipt = _validate_common_execution(
            scheduled_request=scheduled_request,
            grid_provenance=grid_provenance,
            decode_provenance=decode_provenance,
            decode_result=decode_result,
        )
        (
            prompt_evidence,
            visual_tensor_receipt,
            batch_evidence,
            dependency_payload_json,
            dependency_sha256,
            cumulative_evidence,
        ) = _build_bound_input_evidence(
            scheduled_request=scheduled_request,
            prompt_record=prompt_record,
            decode_request_batch=decode_request_batch,
            request_batch=request_batch,
            physical_batch_plan=physical_batch_plan,
            attempt_dependency=attempt_dependency,
            predecessor_attempt=predecessor_attempt,
            decode_receipt=receipt,
            decode_result=decode_result,
        )
        spatial_image_encoding = visual_materialization.spatial_image_encoding
        if spatial_image_encoding is None:
            _fail(
                "spatial execution requires processor-owned spatial materialization",
                "analysis.execution_evidence_spatial_materialization",
            )
        expected_mode = spatial_variant_mode_for_arm(scheduled_request.arm)
        if expected_mode is None:
            _fail(
                "full-image request cannot use spatial execution evidence",
                "analysis.execution_evidence_spatial_arm",
            )
        if spatial_image_encoding.plan.variant_mode != expected_mode:
            _fail(
                "spatial encoding variant does not match the scheduled arm",
                "analysis.execution_evidence_spatial_variant",
            )
        if (
            grid_provenance.canonical_spatial_spec_sha256
            != spatial_image_encoding.plan.grid_spec.fingerprint
        ):
            _fail(
                "spatial encoding grid differs from canonical grid provenance",
                "analysis.execution_evidence_spatial_grid",
            )
        if spatial_image_encoding.plan.cell.index != scheduled_request.cell_index:
            _fail(
                "spatial encoding cell does not match the scheduled request",
                "analysis.execution_evidence_spatial_cell",
            )
        source_digest = spatial_image_encoding.source_image_sha256
        if source_digest is None:
            _fail(
                "execution evidence requires a file-backed spatial encoding",
                "analysis.execution_evidence_source_unattested",
            )
        if source_digest != scheduled_request.image_sha256:
            _fail(
                "spatial encoding source bytes differ from the scheduled image",
                "analysis.execution_evidence_source_mismatch",
            )
        if visual_materialization.receipt.spatial_image_encoding_sha256 != (
            spatial_image_encoding.fingerprint
        ):
            _fail(
                "materialization receipt is not bound to the spatial encoding",
                "analysis.execution_evidence_processor_encoding",
            )
        if visual_materialization.receipt.source_image_sha256 != source_digest:
            _fail(
                "materialization receipt is not bound to the scheduled source",
                "analysis.execution_evidence_source_mismatch",
            )
        visual_materialization.verify_model_inputs(
            decode_request_batch[receipt.request_execution_index].model_inputs
        )
        if visual_materialization.receipt.executed_visual_tensors.receipt_sha256 != (
            visual_tensor_receipt.receipt_sha256
        ):
            _fail(
                "processor-owned materialization differs from backend visual tensors",
                "analysis.execution_evidence_visual_tensor_mismatch",
            )
        envelope = cls(
            scheduled_request=scheduled_request,
            request_fingerprint=sha256_payload(scheduled_request.to_artifact_dict()),
            grid_provenance=grid_provenance,
            decode_provenance=decode_provenance,
            executed_prompt_evidence=prompt_evidence,
            executed_visual_tensor_receipt=visual_tensor_receipt,
            executed_request_batch_evidence=batch_evidence,
            attempt_dependency_payload_json=dependency_payload_json,
            attempt_dependency_sha256=dependency_sha256,
            cumulative_state_dependency_evidence=cumulative_evidence,
            evidence_kind="spatial_materialization_receipt",
            source_image_sha256=source_digest,
            source_width=spatial_image_encoding.plan.source_width,
            source_height=spatial_image_encoding.plan.source_height,
            processor_receipt_sha256=visual_materialization.receipt.receipt_sha256,
            processor_receipt_payload_json=canonical_json_text(
                visual_materialization.receipt.identity_payload()
            ),
            spatial_image_encoding_sha256=spatial_image_encoding.fingerprint,
            full_image_plan_sha256=None,
            decode_execution_receipt=receipt,
            decode_receipt_fingerprint=receipt.receipt_fingerprint,
            generated_output_token_sha256=(receipt.generated_token_identifiers_hash),
            token_trace_sha256=receipt.canonical_float32_score_trace_hash,
        )
        return envelope

    @classmethod
    def from_full_image_execution(
        cls,
        *,
        scheduled_request: ScheduledRequest,
        grid_provenance: GridProvenance,
        decode_provenance: DecodeProvenance,
        prompt_record: PromptRecord,
        decode_request_batch: Sequence[DecodeRequest],
        request_batch: RequestBatch,
        physical_batch_plan: PhysicalBatchPlan,
        attempt_dependency: AttemptDependencyContract,
        predecessor_attempt: AttemptRecord | None,
        image_plan_row: ImagePlanRow,
        visual_materialization: MaterializedVisualInput,
        decode_result: DecodeResult,
    ) -> ExecutionEvidenceEnvelope:
        """Build evidence for an executed no-resize complete-image call."""

        receipt = _validate_common_execution(
            scheduled_request=scheduled_request,
            grid_provenance=grid_provenance,
            decode_provenance=decode_provenance,
            decode_result=decode_result,
        )
        (
            prompt_evidence,
            visual_tensor_receipt,
            batch_evidence,
            dependency_payload_json,
            dependency_sha256,
            cumulative_evidence,
        ) = _build_bound_input_evidence(
            scheduled_request=scheduled_request,
            prompt_record=prompt_record,
            decode_request_batch=decode_request_batch,
            request_batch=request_batch,
            physical_batch_plan=physical_batch_plan,
            attempt_dependency=attempt_dependency,
            predecessor_attempt=predecessor_attempt,
            decode_receipt=receipt,
            decode_result=decode_result,
        )
        if spatial_variant_mode_for_arm(scheduled_request.arm) is not None:
            _fail(
                "spatial request cannot use full-image processor evidence",
                "analysis.execution_evidence_full_arm",
            )
        _validate_full_image_plan(
            image_plan_row=image_plan_row,
        )
        _validate_full_image_visual_tensors(
            image_plan_row=image_plan_row,
            decode_request=decode_request_batch[receipt.request_execution_index],
        )
        image_path = Path(image_plan_row.image_path)
        source_digest = sha256_file(image_path)
        if source_digest != scheduled_request.image_sha256:
            _fail(
                "full-image source bytes differ from the scheduled image",
                "analysis.execution_evidence_source_mismatch",
            )
        if visual_materialization.spatial_image_encoding is not None or (
            visual_materialization.receipt.input_kind != "full_image"
        ):
            _fail(
                "full-image execution requires full-image processor materialization",
                "analysis.execution_evidence_full_materialization",
            )
        if visual_materialization.receipt.source_image_sha256 != source_digest:
            _fail(
                "full-image materialization differs from scheduled source bytes",
                "analysis.execution_evidence_source_mismatch",
            )
        if (
            visual_materialization.receipt.input_width != image_plan_row.decoded_width
            or visual_materialization.receipt.input_height
            != image_plan_row.decoded_height
        ):
            _fail(
                "full-image materialization dimensions differ from image plan",
                "analysis.execution_evidence_full_dimensions",
            )
        visual_materialization.verify_model_inputs(
            decode_request_batch[receipt.request_execution_index].model_inputs
        )
        if visual_materialization.receipt.executed_visual_tensors.receipt_sha256 != (
            visual_tensor_receipt.receipt_sha256
        ):
            _fail(
                "processor-owned materialization differs from backend visual tensors",
                "analysis.execution_evidence_visual_tensor_mismatch",
            )
        image_plan_sha256 = sha256_payload(image_plan_row.to_artifact_dict())
        envelope = cls(
            scheduled_request=scheduled_request,
            request_fingerprint=sha256_payload(scheduled_request.to_artifact_dict()),
            grid_provenance=grid_provenance,
            decode_provenance=decode_provenance,
            executed_prompt_evidence=prompt_evidence,
            executed_visual_tensor_receipt=visual_tensor_receipt,
            executed_request_batch_evidence=batch_evidence,
            attempt_dependency_payload_json=dependency_payload_json,
            attempt_dependency_sha256=dependency_sha256,
            cumulative_state_dependency_evidence=cumulative_evidence,
            evidence_kind="full_image_materialization_receipt",
            source_image_sha256=source_digest,
            source_width=image_plan_row.decoded_width,
            source_height=image_plan_row.decoded_height,
            processor_receipt_sha256=visual_materialization.receipt.receipt_sha256,
            processor_receipt_payload_json=canonical_json_text(
                visual_materialization.receipt.identity_payload()
            ),
            spatial_image_encoding_sha256=None,
            full_image_plan_sha256=image_plan_sha256,
            decode_execution_receipt=receipt,
            decode_receipt_fingerprint=receipt.receipt_fingerprint,
            generated_output_token_sha256=(receipt.generated_token_identifiers_hash),
            token_trace_sha256=receipt.canonical_float32_score_trace_hash,
        )
        return envelope

    @property
    def request_id(self) -> str:
        return self.scheduled_request.request_id

    @property
    def image_id(self) -> int:
        return self.scheduled_request.image_id

    @property
    def arm(self) -> ResearchArmDefinition:
        return self.scheduled_request.arm

    @property
    def canonical_cell_index(self) -> int | None:
        return self.scheduled_request.cell_index

    @property
    def sampling_seed(self) -> int:
        return self.scheduled_request.sampling_seed

    @property
    def fingerprint(self) -> str:
        value = self.envelope_fingerprint
        if value is None:  # __post_init__ always materializes the canonical digest.
            raise AssertionError("execution evidence fingerprint was not materialized")
        return value

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_EVIDENCE_SCHEMA_VERSION:
            _fail(
                "execution evidence schema version is unsupported",
                "analysis.execution_evidence_schema",
            )
        self.scheduled_request.validate_request_id()
        if self.request_fingerprint != sha256_payload(
            self.scheduled_request.to_artifact_dict()
        ):
            _fail(
                "request fingerprint does not bind the scheduled request",
                "analysis.execution_evidence_request_fingerprint",
            )
        if self.scheduled_request.grid_sha256 != self.grid_provenance.fingerprint:
            _fail(
                "grid provenance differs from the scheduled request",
                "analysis.execution_evidence_grid_provenance",
            )
        if self.scheduled_request.decode_sha256 != self.decode_provenance.fingerprint:
            _fail(
                "decode provenance differs from the scheduled request",
                "analysis.execution_evidence_decode_provenance",
            )
        try:
            dependency_payload = json.loads(self.attempt_dependency_payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DataContractError(
                "attempt dependency payload must be canonical JSON",
                code="analysis.execution_evidence_dependency_payload",
                cause=exc,
            ) from exc
        if (
            canonical_json_text(dependency_payload)
            != self.attempt_dependency_payload_json
        ):
            _fail(
                "attempt dependency payload is not canonical",
                "analysis.execution_evidence_dependency_payload",
            )
        if (
            hashlib.sha256(
                self.attempt_dependency_payload_json.encode("utf-8")
            ).hexdigest()
            != self.attempt_dependency_sha256
        ):
            _fail(
                "attempt dependency digest does not bind its payload",
                "analysis.execution_evidence_dependency_digest",
            )
        if dependency_payload.get("request_id") != self.request_id:
            _fail(
                "attempt dependency belongs to another request",
                "analysis.execution_evidence_dependency_request",
            )
        batch_evidence = self.executed_request_batch_evidence
        expected_backend_batch_fingerprint = sha256_payload(
            [
                {
                    "decode_generation_policy_fingerprint": (
                        self.decode_provenance.canonical_generation_policy_sha256
                    ),
                    "prompt_token_identifiers_hash": prompt_sha256,
                    "request_id": request_id,
                }
                for request_id, prompt_sha256 in zip(
                    batch_evidence.batch_request_ids,
                    batch_evidence.batch_prompt_token_ids_sha256,
                    strict=True,
                )
            ]
        )
        if (
            batch_evidence.backend_batch_order_fingerprint
            != expected_backend_batch_fingerprint
        ):
            _fail(
                "executed batch fingerprint does not bind ordered requests and prompts",
                "analysis.execution_evidence_backend_batch_fingerprint",
            )
        if (
            batch_evidence.schedule_identity_sha256
            != self.scheduled_request.schedule_identity_sha256
            or batch_evidence.physical_batch_plan_sha256
            != dependency_payload.get("physical_batch_plan_sha256")
            or batch_evidence.physical_batch_sha256
            != dependency_payload.get("physical_batch_sha256")
            or batch_evidence.physical_batch_index
            != dependency_payload.get("physical_batch_index")
            or batch_evidence.batch_request_ids[batch_evidence.request_execution_index]
            != self.request_id
        ):
            _fail(
                "sealed batch, dependency, and scheduled request evidence disagree",
                "analysis.execution_evidence_batch_dependency_binding",
            )
        prompt_evidence = self.executed_prompt_evidence
        current_prompt_hash = batch_evidence.batch_prompt_token_ids_sha256[
            batch_evidence.request_execution_index
        ]
        if current_prompt_hash != prompt_evidence.prompt_token_ids_sha256:
            _fail(
                "current prompt differs from its executed batch slot",
                "analysis.execution_evidence_batch_prompt_binding",
            )
        if self.source_image_sha256 != self.scheduled_request.image_sha256:
            _fail(
                "source image digest differs from the scheduled request",
                "analysis.execution_evidence_source_mismatch",
            )
        try:
            processor_payload = json.loads(self.processor_receipt_payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DataContractError(
                "processor receipt payload must be canonical JSON",
                code="analysis.execution_evidence_processor_payload",
                cause=exc,
            ) from exc
        if not isinstance(processor_payload, dict):
            _fail(
                "processor receipt payload must be a JSON object",
                "analysis.execution_evidence_processor_payload",
            )
        if (
            canonical_json_text(processor_payload)
            != self.processor_receipt_payload_json
        ):
            _fail(
                "processor receipt payload is not canonically serialized",
                "analysis.execution_evidence_processor_payload",
            )
        if (
            hashlib.sha256(
                self.processor_receipt_payload_json.encode("utf-8")
            ).hexdigest()
            != self.processor_receipt_sha256
        ):
            _fail(
                "processor receipt digest does not bind its canonical payload",
                "analysis.execution_evidence_processor_digest",
            )
        decode_receipt = self.decode_execution_receipt
        if decode_receipt.receipt_fingerprint != self.decode_receipt_fingerprint:
            _fail(
                "decode receipt fingerprint differs from the embedded receipt",
                "analysis.execution_evidence_decode_receipt",
            )
        if decode_receipt.request_id != self.scheduled_request.request_id:
            _fail(
                "embedded decode receipt belongs to another request",
                "analysis.execution_evidence_decode_request",
            )
        if (
            decode_receipt.request_execution_index
            != batch_evidence.request_execution_index
            or decode_receipt.batch_request_order_fingerprint
            != batch_evidence.backend_batch_order_fingerprint
        ):
            _fail(
                "embedded decode receipt differs from sealed executed batch evidence",
                "analysis.execution_evidence_backend_batch_fingerprint",
            )
        if (
            decode_receipt.prompt_token_count != len(prompt_evidence.prompt_token_ids)
            or decode_receipt.prompt_token_identifiers_hash
            != prompt_evidence.prompt_token_ids_sha256
        ):
            _fail(
                "embedded decode receipt prompt differs from executed prompt evidence",
                "analysis.execution_evidence_prompt_tokens",
            )
        if decode_receipt.sampling_seed != self.scheduled_request.sampling_seed:
            _fail(
                "embedded decode receipt seed differs from the scheduled request",
                "analysis.execution_evidence_sampling_seed",
            )
        if (
            decode_receipt.decode_generation_policy_fingerprint
            != self.decode_provenance.canonical_generation_policy_sha256
        ):
            _fail(
                "embedded decode receipt policy differs from decode provenance",
                "analysis.execution_evidence_decode_policy",
            )
        if (
            decode_receipt.generated_token_identifiers_hash
            != self.generated_output_token_sha256
        ):
            _fail(
                "generated output-token digest differs from the decode receipt",
                "analysis.execution_evidence_output_tokens",
            )
        if decode_receipt.canonical_float32_score_trace_hash != self.token_trace_sha256:
            _fail(
                "token-trace digest differs from the decode receipt",
                "analysis.execution_evidence_token_trace",
            )
        if self.envelope_fingerprint is None:
            object.__setattr__(
                self,
                "envelope_fingerprint",
                sha256_payload(self.identity_payload()),
            )
        for field_name in (
            "request_fingerprint",
            "attempt_dependency_sha256",
            "source_image_sha256",
            "processor_receipt_sha256",
            "decode_receipt_fingerprint",
            "generated_output_token_sha256",
            "token_trace_sha256",
            "envelope_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        for field_name in (
            "spatial_image_encoding_sha256",
            "full_image_plan_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_sha256(value, field=field_name)
        expected_spatial_mode = spatial_variant_mode_for_arm(self.arm)
        cumulative_evidence = self.cumulative_state_dependency_evidence
        if self.arm.cumulative_dependency:
            if cumulative_evidence is None:
                _fail(
                    "cumulative request is missing predecessor-state evidence",
                    "analysis.execution_evidence_cumulative_state_missing",
                )
            if (
                cumulative_evidence.request_id != self.request_id
                or cumulative_evidence.prompt_record_sha256
                != prompt_evidence.prompt_record_sha256
                or cumulative_evidence.full_prompt_fingerprint
                != prompt_evidence.full_prompt_fingerprint
            ):
                _fail(
                    "cumulative predecessor state does not bind the executed prompt",
                    "analysis.execution_evidence_cumulative_prompt_binding",
                )
            if cumulative_evidence.predecessor_state_artifact_path is not None:
                state_path = Path(cumulative_evidence.predecessor_state_artifact_path)
                if (
                    not state_path.is_file()
                    or sha256_file(state_path)
                    != cumulative_evidence.predecessor_state_artifact_sha256
                ):
                    _fail(
                        "cumulative predecessor artifact no longer matches its evidence",
                        "analysis.execution_evidence_cumulative_artifact_digest",
                    )
        elif cumulative_evidence is not None:
            _fail(
                "reset request cannot carry cumulative predecessor-state evidence",
                "analysis.execution_evidence_unexpected_cumulative_state",
            )
        if self.evidence_kind == "spatial_materialization_receipt":
            if expected_spatial_mode is None:
                _fail(
                    "full-image arm cannot carry spatial processor evidence",
                    "analysis.execution_evidence_spatial_arm",
                )
            if (
                self.spatial_image_encoding_sha256 is None
                or self.full_image_plan_sha256 is not None
            ):
                _fail(
                    "spatial evidence must contain only its encoding fingerprint",
                    "analysis.execution_evidence_spatial_shape",
                )
            if (
                processor_payload.get("spatial_image_encoding_sha256")
                != self.spatial_image_encoding_sha256
            ):
                _fail(
                    "spatial processor payload differs from its encoding fingerprint",
                    "analysis.execution_evidence_processor_encoding",
                )
            if processor_payload.get("input_kind") != "spatial_variant":
                _fail(
                    "spatial processor payload has the wrong input kind",
                    "analysis.execution_evidence_spatial_materialization",
                )
            if (
                processor_payload.get("executed_visual_tensors")
                != self.executed_visual_tensor_receipt.to_artifact_dict()
            ):
                _fail(
                    "spatial processor payload differs from executed visual tensors",
                    "analysis.execution_evidence_visual_tensor_mismatch",
                )
        elif self.evidence_kind == "full_image_materialization_receipt":
            if expected_spatial_mode is not None:
                _fail(
                    "spatial arm cannot carry full-image processor evidence",
                    "analysis.execution_evidence_full_arm",
                )
            if (
                self.full_image_plan_sha256 is None
                or self.spatial_image_encoding_sha256 is not None
            ):
                _fail(
                    "full-image evidence must contain its source plan fingerprint",
                    "analysis.execution_evidence_full_shape",
                )
            if processor_payload.get("input_kind") != "full_image":
                _fail(
                    "full-image processor payload has the wrong input kind",
                    "analysis.execution_evidence_full_materialization",
                )
            if (
                processor_payload.get("executed_visual_tensors")
                != self.executed_visual_tensor_receipt.to_artifact_dict()
            ):
                _fail(
                    "full-image processor payload differs from executed visual tensors",
                    "analysis.execution_evidence_visual_tensor_mismatch",
                )
        else:
            _fail(
                "execution evidence kind is unsupported",
                "analysis.execution_evidence_kind",
            )
        if self.source_width <= 0 or self.source_height <= 0:
            _fail(
                "source canvas dimensions must be positive",
                "analysis.execution_evidence_source_frame",
            )
        if self.envelope_fingerprint != sha256_payload(self.identity_payload()):
            _fail(
                "execution evidence fingerprint does not bind the envelope",
                "analysis.execution_evidence_fingerprint",
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "attempt_dependency_payload_json": self.attempt_dependency_payload_json,
            "attempt_dependency_sha256": self.attempt_dependency_sha256,
            "cumulative_state_dependency_evidence": (
                None
                if self.cumulative_state_dependency_evidence is None
                else self.cumulative_state_dependency_evidence.to_artifact_dict()
            ),
            "decode_provenance": self.decode_provenance.to_artifact_dict(),
            "decode_execution_receipt": (
                self.decode_execution_receipt.to_artifact_dict()
            ),
            "decode_receipt_fingerprint": self.decode_receipt_fingerprint,
            "envelope_kind": self.evidence_kind,
            "executed_prompt_evidence": self.executed_prompt_evidence.to_artifact_dict(),
            "executed_request_batch_evidence": (
                self.executed_request_batch_evidence.to_artifact_dict()
            ),
            "executed_visual_tensor_receipt": (
                self.executed_visual_tensor_receipt.to_artifact_dict()
            ),
            "full_image_plan_sha256": self.full_image_plan_sha256,
            "generated_output_token_sha256": self.generated_output_token_sha256,
            "grid_provenance": self.grid_provenance.to_artifact_dict(),
            "processor_receipt_sha256": self.processor_receipt_sha256,
            "processor_receipt_payload_json": self.processor_receipt_payload_json,
            "request_fingerprint": self.request_fingerprint,
            "scheduled_request": self.scheduled_request.to_artifact_dict(),
            "schema_version": self.schema_version,
            "source_height": self.source_height,
            "source_image_sha256": self.source_image_sha256,
            "source_width": self.source_width,
            "spatial_image_encoding_sha256": (self.spatial_image_encoding_sha256),
            "token_trace_sha256": self.token_trace_sha256,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "arm": self.arm.to_artifact_dict(),
            "canonical_cell_index": self.canonical_cell_index,
            "decode_sha256": self.scheduled_request.decode_sha256,
            "envelope_fingerprint": self.envelope_fingerprint,
            "grid_sha256": self.scheduled_request.grid_sha256,
            "image_id": self.image_id,
            "request_id": self.request_id,
            "runtime_attestation_sha256": (
                self.decode_provenance.sampled_runtime_attestation_sha256
            ),
            "sampling_seed": self.sampling_seed,
        }

    @classmethod
    def from_artifact_dict(
        cls, value: Mapping[str, Any]
    ) -> ExecutionEvidenceEnvelope:
        """Rehydrate a sealed envelope and re-run every live invariant."""

        identity_fields = {
            "attempt_dependency_payload_json",
            "attempt_dependency_sha256",
            "cumulative_state_dependency_evidence",
            "decode_execution_receipt",
            "decode_provenance",
            "decode_receipt_fingerprint",
            "envelope_kind",
            "executed_prompt_evidence",
            "executed_request_batch_evidence",
            "executed_visual_tensor_receipt",
            "full_image_plan_sha256",
            "generated_output_token_sha256",
            "grid_provenance",
            "processor_receipt_payload_json",
            "processor_receipt_sha256",
            "request_fingerprint",
            "scheduled_request",
            "schema_version",
            "source_height",
            "source_image_sha256",
            "source_width",
            "spatial_image_encoding_sha256",
            "token_trace_sha256",
        }
        derived_fields = {
            "arm",
            "canonical_cell_index",
            "decode_sha256",
            "envelope_fingerprint",
            "grid_sha256",
            "image_id",
            "request_id",
            "runtime_attestation_sha256",
            "sampling_seed",
        }
        _require_exact_artifact_keys(
            value,
            fields=identity_fields | derived_fields,
            record_name="execution evidence envelope",
        )
        payload = dict(value)
        scheduled_request = ScheduledRequest.from_artifact_dict(
            payload["scheduled_request"]
        )
        expected_derived = {
            "arm": scheduled_request.arm.to_artifact_dict(),
            "canonical_cell_index": scheduled_request.cell_index,
            "decode_sha256": scheduled_request.decode_sha256,
            "grid_sha256": scheduled_request.grid_sha256,
            "image_id": scheduled_request.image_id,
            "request_id": scheduled_request.request_id,
            "sampling_seed": scheduled_request.sampling_seed,
        }
        mismatches = {
            field: {"expected": expected, "observed": payload[field]}
            for field, expected in expected_derived.items()
            if payload[field] != expected
        }
        decode_provenance = DecodeProvenance.from_artifact_dict(
            payload["decode_provenance"]
        )
        if (
            payload["runtime_attestation_sha256"]
            != decode_provenance.sampled_runtime_attestation_sha256
        ):
            mismatches["runtime_attestation_sha256"] = {
                "expected": decode_provenance.sampled_runtime_attestation_sha256,
                "observed": payload["runtime_attestation_sha256"],
            }
        if mismatches:
            _fail(
                "execution evidence derived fields differ from sealed identities",
                "analysis.execution_evidence_artifact_derived_fields",
                mismatches=mismatches,
            )
        visual_payload = payload["executed_visual_tensor_receipt"]
        _require_exact_artifact_keys(
            visual_payload,
            fields={
                "image_grid_thw",
                "pixel_values",
                "receipt_sha256",
                "schema_version",
            },
            record_name="executed visual tensor receipt",
        )
        tensor_fields = {
            "canonical_content_sha256",
            "dtype",
            "schema_version",
            "shape",
        }

        def load_tensor(name: str) -> TensorContentReceipt:
            tensor_payload = visual_payload[name]
            _require_exact_artifact_keys(
                tensor_payload,
                fields=tensor_fields,
                record_name=f"{name} tensor content receipt",
            )
            tensor_values = dict(tensor_payload)
            tensor_values["shape"] = tuple(tensor_values["shape"])
            return TensorContentReceipt(**tensor_values)

        visual_receipt = ExecutedVisualTensorReceipt(
            pixel_values=load_tensor("pixel_values"),
            image_grid_thw=load_tensor("image_grid_thw"),
            receipt_sha256=visual_payload["receipt_sha256"],
            schema_version=visual_payload["schema_version"],
        )
        cumulative_payload = payload["cumulative_state_dependency_evidence"]
        return cls(
            scheduled_request=scheduled_request,
            request_fingerprint=payload["request_fingerprint"],
            grid_provenance=GridProvenance.from_artifact_dict(
                payload["grid_provenance"]
            ),
            decode_provenance=decode_provenance,
            executed_prompt_evidence=ExecutedPromptEvidence.from_artifact_dict(
                payload["executed_prompt_evidence"]
            ),
            executed_visual_tensor_receipt=visual_receipt,
            executed_request_batch_evidence=(
                ExecutedRequestBatchEvidence.from_artifact_dict(
                    payload["executed_request_batch_evidence"]
                )
            ),
            attempt_dependency_payload_json=payload[
                "attempt_dependency_payload_json"
            ],
            attempt_dependency_sha256=payload["attempt_dependency_sha256"],
            cumulative_state_dependency_evidence=(
                None
                if cumulative_payload is None
                else CumulativeStateDependencyEvidence.from_artifact_dict(
                    cumulative_payload
                )
            ),
            evidence_kind=payload["envelope_kind"],
            source_image_sha256=payload["source_image_sha256"],
            source_width=payload["source_width"],
            source_height=payload["source_height"],
            processor_receipt_sha256=payload["processor_receipt_sha256"],
            processor_receipt_payload_json=payload[
                "processor_receipt_payload_json"
            ],
            spatial_image_encoding_sha256=payload[
                "spatial_image_encoding_sha256"
            ],
            full_image_plan_sha256=payload["full_image_plan_sha256"],
            decode_execution_receipt=DecodeExecutionReceipt.from_artifact_dict(
                payload["decode_execution_receipt"]
            ),
            decode_receipt_fingerprint=payload["decode_receipt_fingerprint"],
            generated_output_token_sha256=payload[
                "generated_output_token_sha256"
            ],
            token_trace_sha256=payload["token_trace_sha256"],
            envelope_fingerprint=payload["envelope_fingerprint"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class TerminalCallOutputBundle:
    """Canonical terminal artifact joining execution, decode, parse, and state."""

    payload_json: str
    bundle_sha256: str
    schema_version: str = TERMINAL_CALL_OUTPUT_BUNDLE_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        scheduled_request: ScheduledRequest,
        attempt_status: str,
        execution_evidence: ExecutionEvidenceEnvelope | None,
        decode_result: DecodeResult | None,
        parse_score_receipts: Sequence[Any],
        row_diagnostics: Sequence[Mapping[str, Any]],
        call_diagnostics: Mapping[str, Any],
        stop_reason: str | None,
        failure_code: str | None,
        cumulative_state_product: Mapping[str, Any] | None,
    ) -> TerminalCallOutputBundle:
        if attempt_status == "completed":
            if execution_evidence is None or decode_result is None or failure_code is not None:
                _fail(
                    "completed terminal bundle requires execution and decode evidence only",
                    "analysis.terminal_bundle_completed_evidence",
                )
            if execution_evidence.request_id != scheduled_request.request_id:
                _fail(
                    "terminal bundle execution evidence belongs to another request",
                    "analysis.terminal_bundle_request",
                )
            if decode_result.request_id != scheduled_request.request_id:
                _fail(
                    "terminal bundle decode result belongs to another request",
                    "analysis.terminal_bundle_request",
                )
            if stop_reason != decode_result.stop_reason:
                _fail(
                    "terminal bundle stop reason differs from decode result",
                    "analysis.terminal_bundle_stop_reason",
                )
            if scheduled_request.arm.cumulative_dependency:
                if cumulative_state_product is None:
                    _fail(
                        "completed cumulative call requires its produced state artifact",
                        "analysis.terminal_bundle_cumulative_state",
                    )
                _validate_terminal_cumulative_state_product(cumulative_state_product)
            elif cumulative_state_product is not None:
                _fail(
                    "reset call cannot claim a cumulative state product",
                    "analysis.terminal_bundle_cumulative_state",
                )
        else:
            if not failure_code:
                _fail(
                    "non-completed terminal bundle requires a failure code",
                    "analysis.terminal_bundle_failure_code",
                )
            if execution_evidence is not None or decode_result is not None:
                _fail(
                    "non-completed terminal bundle cannot claim decode evidence",
                    "analysis.terminal_bundle_failure_evidence",
                )
            if parse_score_receipts or row_diagnostics:
                _fail(
                    "non-completed terminal bundle cannot contain parsed rows",
                    "analysis.terminal_bundle_failure_rows",
                )
            if cumulative_state_product is not None:
                _fail(
                    "non-completed terminal bundle cannot contain cumulative state",
                    "analysis.terminal_bundle_failure_cumulative_state",
                )
        receipt_payloads: list[dict[str, Any]] = []
        for receipt in parse_score_receipts:
            to_artifact = getattr(receipt, "to_artifact_dict", None)
            if not callable(to_artifact):
                _fail(
                    "terminal bundle parse receipt lacks canonical serialization",
                    "analysis.terminal_bundle_parse_receipt",
                )
            payload = to_artifact()
            if payload.get("request_id") != scheduled_request.request_id:
                _fail(
                    "terminal bundle parse receipt belongs to another request",
                    "analysis.terminal_bundle_request",
                )
            if execution_evidence is not None and payload.get(
                "execution_evidence_fingerprint"
            ) != execution_evidence.fingerprint:
                _fail(
                    "terminal bundle parse receipt differs from execution evidence",
                    "analysis.terminal_bundle_parse_execution",
                )
            receipt_payloads.append(payload)
        payload = {
            "attempt_status": attempt_status,
            "call_diagnostics": dict(call_diagnostics),
            "cumulative_state_product": (
                None
                if cumulative_state_product is None
                else dict(cumulative_state_product)
            ),
            "decode_result": (
                None if decode_result is None else decode_result.to_artifact_dict()
            ),
            "execution_evidence": (
                None
                if execution_evidence is None
                else execution_evidence.to_artifact_dict()
            ),
            "failure_code": failure_code,
            "parse_score_receipts": receipt_payloads,
            "request_id": scheduled_request.request_id,
            "row_diagnostics": [dict(row) for row in row_diagnostics],
            "scheduled_request": scheduled_request.to_artifact_dict(),
            "schema_version": TERMINAL_CALL_OUTPUT_BUNDLE_SCHEMA_VERSION,
            "stop_reason": stop_reason,
        }
        payload_json = canonical_json_text(payload)
        return cls(
            payload_json=payload_json,
            bundle_sha256=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> TerminalCallOutputBundle:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataContractError(
                "terminal bundle must be UTF-8 JSON",
                code="analysis.terminal_bundle_encoding",
                cause=exc,
            ) from exc
        return cls(
            payload_json=text,
            bundle_sha256=hashlib.sha256(payload).hexdigest(),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> TerminalCallOutputBundle:
        return cls.from_json_bytes(Path(path).read_bytes())

    def __post_init__(self) -> None:
        if self.schema_version != TERMINAL_CALL_OUTPUT_BUNDLE_SCHEMA_VERSION:
            _fail(
                "terminal bundle schema is unsupported",
                "analysis.terminal_bundle_schema",
            )
        try:
            payload = json.loads(self.payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DataContractError(
                "terminal bundle payload must be JSON",
                code="analysis.terminal_bundle_payload",
                cause=exc,
            ) from exc
        if canonical_json_text(payload) != self.payload_json:
            _fail(
                "terminal bundle payload is not canonical",
                "analysis.terminal_bundle_payload",
            )
        if payload.get("schema_version") != self.schema_version:
            _fail(
                "terminal bundle payload schema differs from its wrapper",
                "analysis.terminal_bundle_schema",
            )
        _require_sha256(self.bundle_sha256, field="bundle_sha256")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != (
            self.bundle_sha256
        ):
            _fail(
                "terminal bundle digest does not bind its canonical payload",
                "analysis.terminal_bundle_digest",
            )
        scheduled = payload.get("scheduled_request")
        if not isinstance(scheduled, dict) or scheduled.get("request_id") != payload.get(
            "request_id"
        ):
            _fail(
                "terminal bundle scheduled request identity is inconsistent",
                "analysis.terminal_bundle_request",
            )
        status = payload.get("attempt_status")
        if status == "completed":
            execution = payload.get("execution_evidence")
            decode = payload.get("decode_result")
            if not isinstance(execution, dict) or not isinstance(decode, dict):
                _fail(
                    "completed terminal bundle is missing execution evidence",
                    "analysis.terminal_bundle_completed_evidence",
                )
            if execution.get("request_id") != payload.get("request_id") or decode.get(
                "request_id"
            ) != payload.get("request_id"):
                _fail(
                    "terminal bundle nested evidence belongs to another request",
                    "analysis.terminal_bundle_request",
                )
            if decode.get("stop_reason") != payload.get("stop_reason"):
                _fail(
                    "terminal bundle stop reason differs from decode result",
                    "analysis.terminal_bundle_stop_reason",
                )
            scheduled_arm = scheduled.get("arm")
            is_cumulative = isinstance(scheduled_arm, dict) and bool(
                scheduled_arm.get("cumulative_dependency")
            )
            cumulative_product = payload.get("cumulative_state_product")
            if is_cumulative:
                if not isinstance(cumulative_product, dict):
                    _fail(
                        "completed cumulative bundle lacks its state product",
                        "analysis.terminal_bundle_cumulative_state",
                    )
                _validate_terminal_cumulative_state_product(cumulative_product)
            elif cumulative_product is not None:
                _fail(
                    "reset bundle cannot claim a cumulative state product",
                    "analysis.terminal_bundle_cumulative_state",
                )
        else:
            if not payload.get("failure_code"):
                _fail(
                    "non-completed terminal bundle requires a failure code",
                    "analysis.terminal_bundle_failure_code",
                )
            forbidden_payload = {
                "execution_evidence": payload.get("execution_evidence"),
                "decode_result": payload.get("decode_result"),
                "parse_score_receipts": payload.get("parse_score_receipts"),
                "row_diagnostics": payload.get("row_diagnostics"),
                "cumulative_state_product": payload.get(
                    "cumulative_state_product"
                ),
            }
            if (
                forbidden_payload["execution_evidence"] is not None
                or forbidden_payload["decode_result"] is not None
                or forbidden_payload["parse_score_receipts"] != []
                or forbidden_payload["row_diagnostics"] != []
                or forbidden_payload["cumulative_state_product"] is not None
            ):
                _fail(
                    "non-completed terminal bundle carries forbidden output payload",
                    "analysis.terminal_bundle_failure_payload",
                )

    @property
    def payload(self) -> Mapping[str, Any]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise AssertionError("validated terminal bundle payload is not an object")
        return value

    def to_artifact_bytes(self) -> bytes:
        return self.payload_json.encode("utf-8")


def _validate_terminal_cumulative_state_product(
    value: Mapping[str, Any],
) -> None:
    artifact_path = value.get("artifact_path")
    artifact_sha256 = value.get("artifact_sha256")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        _fail(
            "cumulative state product requires an artifact path",
            "analysis.terminal_bundle_cumulative_state",
        )
    _require_sha256(artifact_sha256, field="cumulative_state_product.artifact_sha256")
    path = Path(artifact_path)
    if not path.is_file() or sha256_file(path) != artifact_sha256:
        _fail(
            "cumulative state product differs from its artifact",
            "analysis.terminal_bundle_cumulative_state_digest",
        )


def _build_bound_input_evidence(
    *,
    scheduled_request: ScheduledRequest,
    prompt_record: PromptRecord,
    decode_request_batch: Sequence[DecodeRequest],
    request_batch: RequestBatch,
    physical_batch_plan: PhysicalBatchPlan,
    attempt_dependency: AttemptDependencyContract,
    predecessor_attempt: AttemptRecord | None,
    decode_receipt: DecodeExecutionReceipt,
    decode_result: DecodeResult,
) -> tuple[
    ExecutedPromptEvidence,
    ExecutedVisualTensorReceipt,
    ExecutedRequestBatchEvidence,
    str,
    str,
    CumulativeStateDependencyEvidence | None,
]:
    ordered_requests = tuple(decode_request_batch)
    batch_evidence = ExecutedRequestBatchEvidence.build(
        scheduled_request=scheduled_request,
        request_batch=request_batch,
        physical_batch_plan=physical_batch_plan,
        decode_request_batch=ordered_requests,
        decode_receipt=decode_receipt,
    )
    decode_request = ordered_requests[batch_evidence.request_execution_index]
    if decode_result.request_id != decode_request.request_id:
        _fail(
            "decode result differs from its executed request-batch slot",
            "analysis.execution_evidence_decode_request",
        )
    prompt_evidence = ExecutedPromptEvidence.build(
        prompt_record=prompt_record,
        decode_request=decode_request,
    )
    if tuple(decode_result.prompt_token_ids) != prompt_evidence.prompt_token_ids:
        _fail(
            "decode result prompt differs from the exact executed prompt record",
            "analysis.execution_evidence_prompt_tokens",
        )
    try:
        pixel_values = decode_request.model_inputs["pixel_values"]
        image_grid_thw = decode_request.model_inputs["image_grid_thw"]
    except KeyError as exc:
        raise DataContractError(
            "executed decode request is missing visual model-input tensors",
            code="analysis.execution_evidence_visual_tensor_missing",
            context={"request_id": decode_request.request_id, "field": str(exc)},
            cause=exc,
        ) from exc
    visual_receipt = ExecutedVisualTensorReceipt.from_tensors(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    dependency_payload = _attempt_dependency_payload(attempt_dependency)
    _validate_attempt_dependency(
        scheduled_request=scheduled_request,
        request_batch=request_batch,
        physical_batch_plan=physical_batch_plan,
        attempt_dependency=attempt_dependency,
    )
    dependency_payload_json = canonical_json_text(dependency_payload)
    dependency_sha256 = hashlib.sha256(
        dependency_payload_json.encode("utf-8")
    ).hexdigest()
    cumulative_evidence = None
    if scheduled_request.arm.cumulative_dependency:
        cumulative_evidence = CumulativeStateDependencyEvidence.build(
            scheduled_request=scheduled_request,
            dependency_contract=attempt_dependency,
            prompt_evidence=prompt_evidence,
            predecessor_attempt=predecessor_attempt,
        )
    elif predecessor_attempt is not None:
        _fail(
            "reset request cannot consume cumulative predecessor evidence",
            "analysis.execution_evidence_unexpected_cumulative_state",
        )
    return (
        prompt_evidence,
        visual_receipt,
        batch_evidence,
        dependency_payload_json,
        dependency_sha256,
        cumulative_evidence,
    )


def _validate_sealed_request_batch(
    *,
    scheduled_request: ScheduledRequest,
    request_batch: RequestBatch,
    physical_batch_plan: PhysicalBatchPlan,
) -> None:
    if request_batch.physical_batch_plan_sha256 != physical_batch_plan.fingerprint:
        _fail(
            "materialized request batch differs from the sealed physical plan",
            "analysis.execution_evidence_batch_plan",
        )
    if (
        physical_batch_plan.schedule_identity_sha256
        != scheduled_request.schedule_identity_sha256
    ):
        _fail(
            "physical batch plan belongs to another schedule",
            "analysis.execution_evidence_batch_schedule",
        )
    if not 0 <= request_batch.batch_index < len(physical_batch_plan.batches):
        _fail(
            "materialized request batch index is outside the sealed plan",
            "analysis.execution_evidence_batch_index",
        )
    sealed_batch = physical_batch_plan.batches[request_batch.batch_index]
    if (
        sealed_batch.fingerprint != request_batch.physical_batch_sha256
        or sealed_batch.request_ids != request_batch.request_ids
    ):
        _fail(
            "materialized request membership differs from its sealed batch",
            "analysis.execution_evidence_batch_membership",
        )
    if request_batch.cardinality == 3:
        next_batch_index = request_batch.batch_index + 1
        if (
            next_batch_index < len(physical_batch_plan.batches)
            and physical_batch_plan.batches[next_batch_index].execution_wave_partition
            == sealed_batch.execution_wave_partition
        ):
            _fail(
                "only the final sealed physical batch of an execution-wave partition may use the natural tail of three",
                "analysis.execution_evidence_batch_tail",
            )
    elif request_batch.cardinality != 4:
        _fail(
            "primary execution requires batch size four or a final natural tail of three",
            "analysis.execution_evidence_batch_cardinality",
        )
    if scheduled_request.request_id not in request_batch.request_ids:
        _fail(
            "scheduled request is absent from its sealed physical batch",
            "analysis.execution_evidence_batch_membership",
        )
    for request in request_batch.requests:
        request.validate_request_id()
        if (
            request.schedule_identity_sha256
            != scheduled_request.schedule_identity_sha256
        ):
            _fail(
                "sealed physical batch mixes schedule identities",
                "analysis.execution_evidence_batch_schedule",
            )


def _validate_attempt_dependency(
    *,
    scheduled_request: ScheduledRequest,
    request_batch: RequestBatch,
    physical_batch_plan: PhysicalBatchPlan,
    attempt_dependency: AttemptDependencyContract,
) -> None:
    expected = {
        "request_id": scheduled_request.request_id,
        "physical_batch_plan_sha256": physical_batch_plan.fingerprint,
        "physical_batch_sha256": request_batch.physical_batch_sha256,
        "physical_batch_index": request_batch.batch_index,
        "predecessor_request_id": scheduled_request.predecessor_request_id,
        "initial_cumulative_state_sha256": (
            scheduled_request.initial_cumulative_state_sha256
        ),
    }
    observed = {field: getattr(attempt_dependency, field) for field in expected}
    if observed != expected:
        _fail(
            "attempt dependency differs from the sealed request, batch, or state origin",
            "analysis.execution_evidence_dependency_contract",
        )


def _attempt_dependency_payload(
    dependency: AttemptDependencyContract,
) -> dict[str, Any]:
    return {
        "initial_cumulative_state_sha256": dependency.initial_cumulative_state_sha256,
        "physical_batch_index": dependency.physical_batch_index,
        "physical_batch_plan_sha256": dependency.physical_batch_plan_sha256,
        "physical_batch_sha256": dependency.physical_batch_sha256,
        "predecessor_request_id": dependency.predecessor_request_id,
        "request_id": dependency.request_id,
        "schema_version": dependency.schema_version,
    }


def _full_prompt_fingerprint(
    *,
    full_chat_text: Any,
    prompt_token_ids: Sequence[int],
) -> str:
    if not isinstance(full_chat_text, str):
        _fail(
            "executed prompt payload must contain full chat text",
            "analysis.execution_evidence_prompt_payload",
        )
    encoded = json.dumps(
        {
            "full_chat_text": full_chat_text,
            "prompt_token_ids": [int(value) for value in prompt_token_ids],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_continuation_prompt_payload(
    *,
    payload: dict[str, Any],
    continuation_text_sha256: str,
) -> None:
    full_chat_text = payload.get("full_chat_text")
    if not isinstance(full_chat_text, str):
        _fail(
            "continued prompt evidence must contain exact full chat text",
            "analysis.execution_evidence_cumulative_prompt_payload",
        )
    character_span = _prompt_span(
        payload.get("continuation_character_span"),
        field="continuation_character_span",
    )
    byte_span = _prompt_span(
        payload.get("continuation_byte_span"),
        field="continuation_byte_span",
    )
    token_span = _prompt_span(
        payload.get("continuation_token_impact_span"),
        field="continuation_token_impact_span",
    )
    content_start = payload.get("open_assistant_content_start_character")
    content_start_byte = payload.get("open_assistant_content_start_byte")
    if (
        isinstance(content_start, bool)
        or not isinstance(content_start, int)
        or isinstance(content_start_byte, bool)
        or not isinstance(content_start_byte, int)
        or character_span[0] != content_start
        or byte_span[0] != content_start_byte
        or character_span[1] != len(full_chat_text)
        or byte_span[1] != len(full_chat_text.encode("utf-8"))
        or token_span[1] != len(payload.get("prompt_token_ids", ()))
        or payload.get("open_assistant_interval_verified") is not True
    ):
        _fail(
            "continued prompt spans do not bind the final open-assistant interval",
            "analysis.execution_evidence_cumulative_prompt_span",
        )
    continuation_text = full_chat_text[character_span[0] : character_span[1]]
    continuation_bytes = full_chat_text.encode("utf-8")[byte_span[0] : byte_span[1]]
    if (
        continuation_bytes != continuation_text.encode("utf-8")
        or hashlib.sha256(continuation_bytes).hexdigest() != continuation_text_sha256
    ):
        _fail(
            "continued prompt digest does not bind the exact continuation bytes",
            "analysis.execution_evidence_cumulative_prompt_digest",
        )
    image_placeholder_count = full_chat_text.count(IMAGE_PLACEHOLDER)
    if (
        payload.get("image_placeholder_count") != image_placeholder_count
        or image_placeholder_count != 1
    ):
        _fail(
            "continued prompt must attest exactly one current-image placeholder",
            "analysis.execution_evidence_cumulative_prompt_image",
        )


def _prompt_span(value: Any, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] < value[0]
    ):
        _fail(
            "continued prompt evidence contains an invalid span",
            "analysis.execution_evidence_cumulative_prompt_span",
            field=field,
        )
    return int(value[0]), int(value[1])


def _accepted_row_prefix_from_state_artifact(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            "cumulative predecessor state artifact is not valid UTF-8 JSON",
            code="analysis.execution_evidence_cumulative_artifact_payload",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "accepted_row_prefix_state.v1"
        or not isinstance(payload.get("accepted_global_coordinate_rows"), list)
        or any(
            not isinstance(row, str)
            for row in payload.get("accepted_global_coordinate_rows", ())
        )
    ):
        _fail(
            "cumulative predecessor state artifact has invalid accepted-row semantics",
            "analysis.execution_evidence_cumulative_artifact_payload",
            path=str(path),
        )
    return "".join(payload["accepted_global_coordinate_rows"])


def _validate_full_image_visual_tensors(
    *,
    image_plan_row: ImagePlanRow,
    decode_request: DecodeRequest,
) -> None:
    try:
        pixel_values = decode_request.model_inputs["pixel_values"]
        image_grid_thw = decode_request.model_inputs["image_grid_thw"]
    except KeyError as exc:
        raise DataContractError(
            "full-image request is missing executed visual tensors",
            code="analysis.execution_evidence_visual_tensor_missing",
            cause=exc,
        ) from exc
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(
        image_grid_thw, torch.Tensor
    ):
        _fail(
            "full-image executed visual inputs must be tensors",
            "analysis.execution_evidence_visual_tensor_type",
        )
    expected_pixel_shape = (
        image_plan_row.raw_patch_rows,
        3
        * image_plan_row.temporal_patch_size
        * image_plan_row.patch_size
        * image_plan_row.patch_size,
    )
    if tuple(pixel_values.shape) != expected_pixel_shape:
        _fail(
            "full-image pixel tensor shape differs from the no-resize image plan",
            "analysis.execution_evidence_visual_pixel_shape",
        )
    if tuple(image_grid_thw.shape) != (1, 3):
        _fail(
            "full-image grid tensor shape differs from the image plan",
            "analysis.execution_evidence_visual_grid_shape",
        )
    observed_grid = tuple(
        int(value) for value in image_grid_thw.detach().to("cpu")[0].tolist()
    )
    if observed_grid != tuple(image_plan_row.observed_image_grid_thw):
        _fail(
            "full-image grid tensor values differ from the image plan",
            "analysis.execution_evidence_visual_grid_values",
        )


def _validate_common_execution(
    *,
    scheduled_request: ScheduledRequest,
    grid_provenance: GridProvenance,
    decode_provenance: DecodeProvenance,
    decode_result: DecodeResult,
) -> DecodeExecutionReceipt:
    scheduled_request.validate_request_id()
    if scheduled_request.grid_sha256 != grid_provenance.fingerprint:
        _fail(
            "grid provenance differs from the scheduled request",
            "analysis.execution_evidence_grid_provenance",
        )
    if scheduled_request.decode_sha256 != decode_provenance.fingerprint:
        _fail(
            "decode provenance differs from the scheduled request",
            "analysis.execution_evidence_decode_provenance",
        )
    decode_result.validate_for_scored()
    receipt = decode_result.execution_receipt
    if receipt is None:  # validate_for_scored is the authority; narrows the type.
        _fail(
            "decode result has no execution receipt",
            "analysis.execution_evidence_decode_receipt",
        )
    if decode_result.request_id != scheduled_request.request_id:
        _fail(
            "decode result request does not match the scheduled request",
            "analysis.execution_evidence_decode_request",
        )
    if receipt.sampling_seed != scheduled_request.sampling_seed:
        _fail(
            "decode sampling seed does not match the scheduled request",
            "analysis.execution_evidence_sampling_seed",
        )
    if receipt.random_generator_initial_seed != scheduled_request.sampling_seed:
        _fail(
            "executed random generator seed does not match the scheduled request",
            "analysis.execution_evidence_generator_seed",
        )
    if (
        receipt.decode_generation_policy_fingerprint
        != decode_provenance.canonical_generation_policy_sha256
    ):
        _fail(
            "executed decode policy differs from canonical decode provenance",
            "analysis.execution_evidence_decode_policy",
        )
    return receipt


def _validate_full_image_plan(
    *,
    image_plan_row: ImagePlanRow,
) -> None:
    if image_plan_row.status != "ok" or image_plan_row.error is not None:
        _fail(
            "full-image processor plan must be successfully materialized",
            "analysis.execution_evidence_full_plan_status",
        )
    if image_plan_row.row_id != image_plan_row.example_id:
        _fail(
            "full-image processor row identity differs from its example identity",
            "analysis.execution_evidence_full_plan_row_id",
        )
    if image_plan_row.do_resize is not False:
        _fail(
            "full-image processor plan must attest no resize",
            "analysis.execution_evidence_full_plan_resize",
        )
    if image_plan_row.observed_image_grid_thw != image_plan_row.expected_image_grid_thw:
        _fail(
            "full-image observed processor grid differs from its plan",
            "analysis.execution_evidence_full_plan_grid",
        )
    if (
        image_plan_row.declared_width != image_plan_row.decoded_width
        or image_plan_row.declared_height != image_plan_row.decoded_height
    ):
        _fail(
            "full-image plan changed the declared source dimensions",
            "analysis.execution_evidence_full_plan_dimensions",
        )
    for field_name in ("patch_size", "merge_size", "temporal_patch_size"):
        value = getattr(image_plan_row, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(
                "full-image processor dimensions must be positive integers",
                "analysis.execution_evidence_full_plan_processor_identity",
                field=field_name,
            )
    if (
        image_plan_row.decoded_width % image_plan_row.patch_size != 0
        or image_plan_row.decoded_height % image_plan_row.patch_size != 0
    ):
        _fail(
            "full-image dimensions are not divisible by the processor patch size",
            "analysis.execution_evidence_full_plan_patch_divisibility",
        )
    expected_grid = [
        1,
        image_plan_row.decoded_height // image_plan_row.patch_size,
        image_plan_row.decoded_width // image_plan_row.patch_size,
    ]
    if image_plan_row.expected_image_grid_thw != expected_grid:
        _fail(
            "full-image processor grid does not derive from source dimensions",
            "analysis.execution_evidence_full_plan_grid_dimensions",
        )
    expected_raw_patch_rows = expected_grid[1] * expected_grid[2]
    expected_merged_tokens = expected_raw_patch_rows // (
        image_plan_row.merge_size * image_plan_row.merge_size
    )
    if (
        image_plan_row.raw_patch_rows != expected_raw_patch_rows
        or image_plan_row.merged_visual_tokens != expected_merged_tokens
    ):
        _fail(
            "full-image token counts do not derive from the executed processor grid",
            "analysis.execution_evidence_full_plan_token_count",
        )
    path = Path(image_plan_row.image_path)
    with Image.open(path) as image:
        actual_dimensions = image.size
    if actual_dimensions != (
        image_plan_row.decoded_width,
        image_plan_row.decoded_height,
    ):
        _fail(
            "full-image processor plan dimensions differ from the source file",
            "analysis.execution_evidence_full_plan_source_dimensions",
        )


def _require_sha256(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            "execution evidence digest must be lowercase SHA-256 hexadecimal",
            "analysis.execution_evidence_digest",
            field=field,
        )


def _require_exact_artifact_keys(
    value: Mapping[str, Any], *, fields: set[str], record_name: str
) -> None:
    if not isinstance(value, Mapping):
        _fail(
            f"{record_name} must be a JSON object",
            "analysis.execution_evidence_artifact_type",
            record_name=record_name,
        )
    observed = set(value)
    if observed != fields:
        _fail(
            f"{record_name} keys are not exact",
            "analysis.execution_evidence_artifact_keys",
            record_name=record_name,
            missing=sorted(fields - observed),
            extra=sorted(observed - fields),
        )


def _fail(message: str, code: str, **context: Any) -> None:
    raise DataContractError(message, code=code, context=context)
