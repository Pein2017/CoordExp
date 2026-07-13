"""Production execution adapter for Spatial Scope and History Disentanglement.

This module is intentionally narrow.  It turns one sealed ``RequestBatch``
into one attested Hugging Face generation call and durable per-request evidence.
Scientific scheduling, metric admission, and process lifecycle remain owned by
their existing modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from src.analysis.spatial_scope_history.calibration import (
    CalibrationSelectionReceipt,
    PrimaryScheduleArtifact,
    SourceRuntimeIdentityReceipt,
    load_attested_sampling_policy_set,
)
from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptRecord,
    CohortImageRecord,
    CohortLedger,
    append_attempt_record,
    canonical_json_text,
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
from src.analysis.spatial_scope_history.metrics import RowMetricRecord
from src.analysis.spatial_scope_history.parse_score_evidence import (
    CanonicalParseScoreReceipt,
    build_canonical_parse_score_receipts,
)
from src.analysis.spatial_scope_history.runner import (
    BatchExecutionContext,
    BatchExecutor,
    WorkerDeviceAssignment,
    WorkerExecutorFactory,
    build_cumulative_prompt_record,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    ResearchSchedule,
    ScheduledRequest,
)
from src.analysis.spatial_scope_history.spatial import (
    MaterializedVisualInput,
    SpatialGrid,
    SpatialGridSpec,
    spatial_variant_mode_for_arm,
)
from src.common.errors import (
    ArtifactContractError,
    DataContractError,
)
from src.config.fingerprint import sha256_json
from src.config.inference import InferConfig, load_infer_config
from src.data import RawExample, load_raw_examples
from src.inference.backend import (
    DecodeGenerationPolicy,
    DecodeRequest,
    DecodeResult,
    HFGenerateBackend,
    VerifiedSampledRuntimeAttestation,
    load_and_rebind_sampled_runtime_attestation_aggregate,
)
from src.inference.image_plan import ImagePlanRow
from src.inference.parsing import parse_compact_object_box_closed
from src.inference.pipeline import _template_config, _tokenizer_identity
from src.inference.prompt import PromptRecord, build_prompt_record
from src.inference.runtime import InferenceRuntime, assemble_runtime


PRODUCTION_EXECUTOR_FACTORY_CONFIG_SCHEMA_VERSION = (
    "spatial_scope_history.production_executor_factory_config.v2"
)
ACCEPTED_ROW_PREFIX_STATE_SCHEMA_VERSION = "accepted_row_prefix_state.v1"


class ProductionExecutorContractError(ArtifactContractError):
    """Fail-closed production-executor protocol violation."""


@dataclass(frozen=True)
class ProductionExecutorFactoryConfig:
    """Spawn-safe paths and immutable identities needed by one worker.

    The source, schedule, cohort, inference config, and sampled-runtime
    attestation are all reloaded inside the child process.  No deserialized
    model or process-local attestation capability crosses the spawn boundary.
    """

    infer_config_path: str
    primary_schedule_artifact_path: str
    cohort_ledger_path: str
    source_jsonl_path: str
    calibration_selection_receipt_path: str
    source_runtime_identity_receipt_path: str
    sampled_runtime_attestation_path: str
    attempt_ledger_path: str
    schema_version: str = PRODUCTION_EXECUTOR_FACTORY_CONFIG_SCHEMA_VERSION

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> ProductionExecutorFactoryConfig:
        expected = {
            "attempt_ledger_path",
            "calibration_selection_receipt_path",
            "cohort_ledger_path",
            "infer_config_path",
            "primary_schedule_artifact_path",
            "sampled_runtime_attestation_path",
            "schema_version",
            "source_jsonl_path",
            "source_runtime_identity_receipt_path",
        }
        if set(value) != expected:
            _fail(
                "production executor factory config keys are not exact",
                "production_executor.factory_config_keys",
                missing=sorted(expected - set(value)),
                extra=sorted(set(value) - expected),
            )
        return cls(**dict(value))

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_EXECUTOR_FACTORY_CONFIG_SCHEMA_VERSION:
            _fail(
                "production executor factory config schema is unsupported",
                "production_executor.factory_config_schema",
            )
        for field in (
            "infer_config_path",
            "primary_schedule_artifact_path",
            "cohort_ledger_path",
            "source_jsonl_path",
            "calibration_selection_receipt_path",
            "source_runtime_identity_receipt_path",
            "sampled_runtime_attestation_path",
            "attempt_ledger_path",
        ):
            if not isinstance(getattr(self, field), str) or not getattr(
                self, field
            ).strip():
                _fail(
                    "production executor factory path is empty",
                    "production_executor.factory_config_path",
                    field=field,
                )


@dataclass(frozen=True)
class ProductionRuntimeBinding:
    """One worker-local runtime, backend, identities, and exact capability."""

    runtime: InferenceRuntime
    backend: HFGenerateBackend
    model_identity: Mapping[str, Any]
    tokenizer_identity: Mapping[str, Any]
    generation_config_fingerprint: str
    verified_runtime_attestation: VerifiedSampledRuntimeAttestation


@dataclass(frozen=True)
class PreparedRequest:
    """One fully materialized request before the indivisible backend call."""

    scheduled_request: ScheduledRequest
    cohort_record: CohortImageRecord
    raw_example: RawExample
    prompt_record: PromptRecord
    visual_materialization: MaterializedVisualInput
    decode_request: DecodeRequest
    image_plan_row: ImagePlanRow | None
    predecessor_attempt: AttemptRecord | None
    predecessor_accepted_rows: tuple[str, ...]


@dataclass(frozen=True)
class CompletedRequestArtifacts:
    """Normalized scientific products derived from one successful decode."""

    execution_evidence: ExecutionEvidenceEnvelope
    parse_score_receipts: tuple[CanonicalParseScoreReceipt, ...]
    normalized_predictions: tuple[NormalizedCallPrediction, ...]
    row_diagnostics: tuple[RowMetricRecord, ...]
    cumulative_accepted_rows: tuple[str, ...]


class ProductionExecutorFactory(WorkerExecutorFactory):
    """Load exactly one runtime/backend and return one persistent executor."""

    def __call__(
        self,
        assignment: WorkerDeviceAssignment,
        factory_config: Mapping[str, Any],
    ) -> BatchExecutor:
        config = ProductionExecutorFactoryConfig.from_mapping(factory_config)
        loaded = _load_worker_inputs(config)
        runtime_binding = _load_runtime_binding(
            infer_config=loaded.infer_config,
            sampled_runtime_attestation_path=loaded.sampled_runtime_attestation_path,
            decode_provenance=loaded.schedule.identity.decode,
        )
        return ProductionBatchExecutor(
            assignment=assignment,
            schedule=loaded.schedule,
            cohort=loaded.cohort,
            raw_examples_by_image_id=loaded.raw_examples_by_image_id,
            infer_config=loaded.infer_config,
            runtime_binding=runtime_binding,
            attempt_ledger_path=loaded.attempt_ledger_path,
        )


@dataclass(frozen=True)
class _LoadedWorkerInputs:
    infer_config: InferConfig
    schedule: ResearchSchedule
    cohort: CohortLedger
    raw_examples_by_image_id: Mapping[int, RawExample]
    sampled_runtime_attestation_path: Path
    attempt_ledger_path: Path


class ProductionBatchExecutor(BatchExecutor):
    """Execute only complete sealed batches through an attested sampled backend."""

    def __init__(
        self,
        *,
        assignment: WorkerDeviceAssignment,
        schedule: ResearchSchedule,
        cohort: CohortLedger,
        raw_examples_by_image_id: Mapping[int, RawExample],
        infer_config: InferConfig,
        runtime_binding: ProductionRuntimeBinding,
        attempt_ledger_path: Path,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._assignment = assignment
        self._schedule = schedule
        self._cohort_by_image_id = {
            record.image_id: record for record in cohort.records
        }
        self._raw_examples_by_image_id = dict(raw_examples_by_image_id)
        self._infer_config = infer_config
        self._binding = runtime_binding
        self._attempt_ledger_path = attempt_ledger_path
        self._monotonic_clock = monotonic_clock
        self._dependency_by_request_id = {
            dependency.request_id: dependency
            for dependency in schedule.attempt_dependencies
        }
        self._request_by_id = {
            request.request_id: request for request in schedule.requests
        }
        qwen = runtime_binding.runtime.qwen
        self._qwen = qwen
        self._processor = _component(qwen, "processor")
        self._image_processor = getattr(self._processor, "image_processor", None)
        if self._image_processor is None:
            _fail(
                "Qwen runtime processor lacks an image processor",
                "production_executor.image_processor_missing",
            )
        self._processor_identity = _processor_identity_payload(qwen)
        self._generation_policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=512,
            repetition_penalty=1.0,
            temperature=schedule.identity.decode.temperature,
            top_p=0.95,
        )
        if (
            self._generation_policy.fingerprint
            != schedule.identity.decode.canonical_generation_policy_sha256
        ):
            _fail(
                "selected generation policy differs from the sealed schedule",
                "production_executor.generation_policy_drift",
            )
        if infer_config.generation.batch_size != 4:
            _fail(
                "production executor requires frozen physical batch size four",
                "production_executor.config_batch_size",
            )
        observed_generation = {
            "max_new_tokens": infer_config.generation.max_new_tokens,
            "repetition_penalty": infer_config.generation.repetition_penalty,
            "top_p": infer_config.generation.top_p,
        }
        if observed_generation != {
            "max_new_tokens": 512,
            "repetition_penalty": 1.0,
            "top_p": 0.95,
        }:
            _fail(
                "inference config differs from frozen generation factors",
                "production_executor.config_generation",
                observed=observed_generation,
            )

    def __call__(self, context: BatchExecutionContext) -> None:
        batch = context.dispatch.batch
        if batch.physical_batch_plan_sha256 != self._schedule.physical_batch_plan.fingerprint:
            _fail(
                "dispatched batch belongs to another physical batch plan",
                "production_executor.batch_plan_identity",
            )
        if batch.cardinality not in {3, 4}:
            _fail(
                "production execution accepts only sealed cardinality four or natural tail three",
                "production_executor.batch_cardinality",
                cardinality=batch.cardinality,
            )
        expected_batch = next(
            (
                item
                for item in self._schedule.batches()
                if item.physical_batch_sha256 == batch.physical_batch_sha256
            ),
            None,
        )
        if expected_batch != batch:
            _fail(
                "dispatched batch differs from the sealed schedule",
                "production_executor.batch_identity",
            )
        attempt_ledger = self._read_attempt_ledger()
        prepared = tuple(
            self._prepare_request(request, attempt_ledger=attempt_ledger)
            for request in batch.requests
        )
        decode_requests = tuple(item.decode_request for item in prepared)
        if tuple(request.request_id for request in decode_requests) != batch.request_ids:
            _fail(
                "prepared backend order differs from sealed physical membership",
                "production_executor.decode_request_order",
            )
        _reset_peak_memory_if_available()
        started = self._monotonic_clock()
        results = self._binding.backend.generate_batch_with_verified_runtime_attestation(
            decode_requests,
            model_identity=self._binding.model_identity,
            tokenizer_identity=self._binding.tokenizer_identity,
            generation_config_fingerprint=(
                self._binding.generation_config_fingerprint
            ),
            verified_runtime_attestation=(
                self._binding.verified_runtime_attestation
            ),
        )
        elapsed = self._monotonic_clock() - started
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
            _fail(
                "backend wall time is invalid",
                "production_executor.wall_time",
            )
        results_by_id = _validate_result_batch(
            decode_requests=decode_requests,
            decode_results=results,
        )
        peak_memory = _peak_memory_if_available()
        completed = tuple(
            self._derive_completed_artifacts(
                item,
                decode_result=results_by_id[item.scheduled_request.request_id],
                decode_request_batch=decode_requests,
                request_batch=batch,
            )
            for item in prepared
        )
        # Persist only after the complete backend result set and all scientific
        # transformations have validated.  This prevents a malformed peer from
        # being silently repacked into a smaller physical call.
        for item, artifacts in zip(prepared, completed, strict=True):
            self._persist_completed_request(
                context=context,
                prepared=item,
                artifacts=artifacts,
                decode_result=results_by_id[item.scheduled_request.request_id],
                elapsed_seconds=float(elapsed),
                peak_device_memory_bytes=peak_memory,
            )

    def _prepare_request(
        self,
        request: ScheduledRequest,
        *,
        attempt_ledger: AttemptLedger,
    ) -> PreparedRequest:
        canonical = self._request_by_id.get(request.request_id)
        if canonical != request:
            _fail(
                "request differs from the schedule-owned request",
                "production_executor.request_identity",
                request_id=request.request_id,
            )
        cohort_record = self._cohort_by_image_id.get(request.image_id)
        raw_example = self._raw_examples_by_image_id.get(request.image_id)
        if cohort_record is None or raw_example is None:
            _fail(
                "scheduled image is absent from cohort or source mapping",
                "production_executor.source_mapping",
                image_id=request.image_id,
            )
        predecessor_attempt, predecessor_rows = self._resolve_predecessor(
            request,
            attempt_ledger=attempt_ledger,
        )
        if request.arm.cumulative_dependency and predecessor_rows:
            prompt_record = build_cumulative_prompt_record(
                raw_example,
                _template_config(self._infer_config),
                processor=self._processor,
                row_index=request.schedule_index,
                accepted_global_coordinate_rows="".join(predecessor_rows),
            )
        else:
            prompt_record = build_prompt_record(
                raw_example,
                _template_config(self._infer_config),
                processor=self._processor,
                row_index=request.schedule_index,
            )
        mode = spatial_variant_mode_for_arm(request.arm)
        if mode is None:
            materialization = MaterializedVisualInput.from_full_image_path(
                source_image_path=cohort_record.image_path,
                expected_source_image_sha256=cohort_record.image_sha256,
                image_processor=self._image_processor,
                processor_contract_sha256=(
                    self._schedule.identity.grid.canonical_spatial_receipt_contract_sha256
                ),
            )
            image_plan_row = _full_image_plan_row(
                request=request,
                raw_example=raw_example,
                cohort_record=cohort_record,
                materialization=materialization,
                processor_identity=self._processor_identity,
            )
        else:
            if request.cell_index is None:
                _fail(
                    "spatial request lacks its cell index",
                    "production_executor.spatial_cell_missing",
                    request_id=request.request_id,
                )
            grid = SpatialGrid.build(
                source_width=cohort_record.source_width,
                source_height=cohort_record.source_height,
                spec=SpatialGridSpec(),
            )
            materialization = MaterializedVisualInput.from_spatial_plan_path(
                plan=grid.plan(cell_index=request.cell_index, variant_mode=mode),
                source_image_path=cohort_record.image_path,
                expected_source_image_sha256=cohort_record.image_sha256,
                image_processor=self._image_processor,
                processor_contract_sha256=(
                    self._schedule.identity.grid.canonical_spatial_receipt_contract_sha256
                ),
            )
            image_plan_row = None
        model_inputs = {
            "pixel_values": materialization.pixel_values,
            "image_grid_thw": materialization.image_grid_thw,
        }
        decode_request = DecodeRequest(
            request_id=request.request_id,
            prompt_token_ids=list(prompt_record.prompt_token_ids),
            model_inputs=model_inputs,
            generation_policy=self._generation_policy,
            sampling_seed=request.sampling_seed,
        )
        return PreparedRequest(
            scheduled_request=request,
            cohort_record=cohort_record,
            raw_example=raw_example,
            prompt_record=prompt_record,
            visual_materialization=materialization,
            decode_request=decode_request,
            image_plan_row=image_plan_row,
            predecessor_attempt=predecessor_attempt,
            predecessor_accepted_rows=predecessor_rows,
        )

    def _resolve_predecessor(
        self,
        request: ScheduledRequest,
        *,
        attempt_ledger: AttemptLedger,
    ) -> tuple[AttemptRecord | None, tuple[str, ...]]:
        if not request.arm.cumulative_dependency:
            return None, ()
        if request.predecessor_request_id is None:
            return None, ()
        predecessor = attempt_ledger.records_by_request_id.get(
            request.predecessor_request_id
        )
        if predecessor is None or predecessor.attempt_status != "completed":
            _fail(
                "runner dispatched cumulative request without a completed predecessor",
                "production_executor.predecessor_not_completed",
                request_id=request.request_id,
                predecessor_request_id=request.predecessor_request_id,
                predecessor_status=(
                    None if predecessor is None else predecessor.attempt_status
                ),
            )
        path_text = predecessor.produced_cumulative_state_artifact_path
        digest = predecessor.produced_cumulative_state_sha256
        if path_text is None or digest is None:
            _fail(
                "completed cumulative predecessor lacks its durable state",
                "production_executor.predecessor_state_missing",
                request_id=request.request_id,
            )
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != digest:
            _fail(
                "cumulative predecessor state is absent or mutated",
                "production_executor.predecessor_state_digest",
                request_id=request.request_id,
            )
        return predecessor, _read_accepted_row_state(path)

    def _derive_completed_artifacts(
        self,
        prepared: PreparedRequest,
        *,
        decode_result: DecodeResult,
        decode_request_batch: Sequence[DecodeRequest],
        request_batch: Any,
    ) -> CompletedRequestArtifacts:
        request = prepared.scheduled_request
        dependency = self._dependency_by_request_id[request.request_id]
        if prepared.image_plan_row is None:
            evidence = ExecutionEvidenceEnvelope.from_spatial_execution(
                scheduled_request=request,
                grid_provenance=self._schedule.identity.grid,
                decode_provenance=self._schedule.identity.decode,
                prompt_record=prepared.prompt_record,
                decode_request_batch=decode_request_batch,
                request_batch=request_batch,
                physical_batch_plan=self._schedule.physical_batch_plan,
                attempt_dependency=dependency,
                predecessor_attempt=prepared.predecessor_attempt,
                visual_materialization=prepared.visual_materialization,
                decode_result=decode_result,
            )
        else:
            evidence = ExecutionEvidenceEnvelope.from_full_image_execution(
                scheduled_request=request,
                grid_provenance=self._schedule.identity.grid,
                decode_provenance=self._schedule.identity.decode,
                prompt_record=prepared.prompt_record,
                decode_request_batch=decode_request_batch,
                request_batch=request_batch,
                physical_batch_plan=self._schedule.physical_batch_plan,
                attempt_dependency=dependency,
                predecessor_attempt=prepared.predecessor_attempt,
                image_plan_row=prepared.image_plan_row,
                visual_materialization=prepared.visual_materialization,
                decode_result=decode_result,
            )
        candidate_receipts = build_canonical_parse_score_receipts(
            execution_evidence=evidence,
            decode_result=decode_result,
        )
        normalized: list[NormalizedCallPrediction] = []
        accepted_receipts: list[CanonicalParseScoreReceipt] = []
        for receipt in candidate_receipts:
            coordinate_receipt = None
            if prepared.image_plan_row is None:
                assert request.cell_index is not None
                mode = spatial_variant_mode_for_arm(request.arm)
                assert mode is not None
                coordinate_receipt = SpatialGrid.build(
                    source_width=prepared.cohort_record.source_width,
                    source_height=prepared.cohort_record.source_height,
                ).plan(
                    cell_index=request.cell_index,
                    variant_mode=mode,
                ).coordinate_receipt(receipt.coordinate_bins)
            try:
                prediction = normalize_call_prediction(
                    execution_evidence=evidence,
                    parse_score_receipt=receipt,
                    spatial_coordinate_receipt=coordinate_receipt,
                )
            except DataContractError as exc:
                if exc.code != "analysis.spatial_merge_category_unknown":
                    raise
                continue
            accepted_receipts.append(receipt)
            normalized.append(prediction)
        parsed = parse_compact_object_box_closed(
            decode_result.parser_text,
            row_id=evidence.request_id,
            row_index=request.schedule_index,
            image_width=(
                prepared.visual_materialization.receipt.input_width
            ),
            image_height=(
                prepared.visual_materialization.receipt.input_height
            ),
        )
        prediction_by_row = {
            prediction.generated_row_index: prediction for prediction in normalized
        }
        row_records: list[RowMetricRecord] = []
        for prediction in parsed.predictions:
            row_index = int(prediction["generated_order"])
            normalized_prediction = prediction_by_row.get(row_index)
            if normalized_prediction is None:
                row_records.append(
                    RowMetricRecord(
                        evidence.request_id,
                        row_index,
                        None,
                        "parsed",
                        "invalid",
                        "not_applicable",
                    )
                )
            else:
                row_records.append(
                    RowMetricRecord(
                        evidence.request_id,
                        row_index,
                        normalized_prediction.prediction_id,
                        "parsed",
                        "valid",
                        normalized_prediction.ownership_status,
                    )
                )
        for dropped in parsed.dropped_predictions:
            row_index = dropped.get("generated_order")
            if isinstance(row_index, int) and not isinstance(row_index, bool):
                row_records.append(
                    RowMetricRecord(
                        evidence.request_id,
                        row_index,
                        None,
                        "malformed",
                        "not_applicable",
                        "not_applicable",
                    )
                )
        row_records.sort(key=lambda row: row.generated_row_index)
        if len({row.generated_row_index for row in row_records}) != len(row_records):
            _fail(
                "parser diagnostics repeat a generated row index",
                "production_executor.row_diagnostic_duplicate",
                request_id=request.request_id,
            )
        owning_rows = tuple(
            receipt.raw_span_text
            for receipt, prediction in zip(
                accepted_receipts,
                normalized,
                strict=True,
            )
            if prediction.ownership_status in {"owned", "not_applicable"}
        )
        cumulative_rows = (
            (*prepared.predecessor_accepted_rows, *owning_rows)
            if request.arm.cumulative_dependency
            else ()
        )
        return CompletedRequestArtifacts(
            execution_evidence=evidence,
            parse_score_receipts=tuple(accepted_receipts),
            normalized_predictions=tuple(normalized),
            row_diagnostics=tuple(row_records),
            cumulative_accepted_rows=tuple(cumulative_rows),
        )

    def _persist_completed_request(
        self,
        *,
        context: BatchExecutionContext,
        prepared: PreparedRequest,
        artifacts: CompletedRequestArtifacts,
        decode_result: DecodeResult,
        elapsed_seconds: float,
        peak_device_memory_bytes: int,
    ) -> None:
        request = prepared.scheduled_request
        journal = context.journals[request.request_id]
        journal.record(
            "materialized_input",
            prepared.visual_materialization.receipt.to_artifact_dict(),
        )
        journal.record(
            "execution_evidence",
            artifacts.execution_evidence.to_artifact_dict(),
        )
        journal.record("decode_result", decode_result.to_artifact_dict())
        journal.record(
            "parse_score",
            {
                "normalized_predictions": [
                    prediction.to_json_dict()
                    for prediction in artifacts.normalized_predictions
                ],
                "parse_score_receipts": [
                    receipt.to_artifact_dict()
                    for receipt in artifacts.parse_score_receipts
                ],
                "row_diagnostics": [
                    row.to_json_dict() for row in artifacts.row_diagnostics
                ],
            },
        )
        cumulative_product = None
        produced_state_path: str | None = None
        produced_state_sha256: str | None = None
        if request.arm.cumulative_dependency:
            state_path = journal.request_directory / "accepted-row-prefix-state.json"
            state_payload = {
                "accepted_global_coordinate_rows": list(
                    artifacts.cumulative_accepted_rows
                ),
                "schema_version": ACCEPTED_ROW_PREFIX_STATE_SCHEMA_VERSION,
            }
            _write_once_bytes(
                state_path,
                canonical_json_text(state_payload).encode("utf-8"),
            )
            produced_state_path = str(state_path)
            produced_state_sha256 = sha256_file(state_path)
            cumulative_product = {
                "artifact_path": produced_state_path,
                "artifact_sha256": produced_state_sha256,
            }
            journal.record("cumulative_state", state_payload)
        call_diagnostics = _call_diagnostics(
            prepared=prepared,
            artifacts=artifacts,
            decode_result=decode_result,
            elapsed_seconds=elapsed_seconds,
            peak_device_memory_bytes=peak_device_memory_bytes,
            processor_identity=self._processor_identity,
        )
        bundle = TerminalCallOutputBundle.build(
            scheduled_request=request,
            attempt_status="completed",
            execution_evidence=artifacts.execution_evidence,
            decode_result=decode_result,
            parse_score_receipts=artifacts.parse_score_receipts,
            row_diagnostics=[row.to_json_dict() for row in artifacts.row_diagnostics],
            call_diagnostics=call_diagnostics,
            stop_reason=decode_result.stop_reason,
            failure_code=None,
            cumulative_state_product=cumulative_product,
        )
        bundle_path = journal.request_directory / "terminal-output-bundle.json"
        _write_once_bytes(bundle_path, bundle.to_artifact_bytes())
        now = _utc_timestamp()
        attempt = AttemptRecord(
            run_id=self._schedule.identity.run_id,
            schedule_sha256=self._schedule.fingerprint,
            request_id=request.request_id,
            physical_batch_plan_sha256=(
                context.dispatch.batch.physical_batch_plan_sha256
            ),
            physical_batch_sha256=context.dispatch.batch.physical_batch_sha256,
            physical_batch_index=context.dispatch.batch.batch_index,
            attempt_status="completed",
            started_at_utc=now,
            finished_at_utc=now,
            execution_identity=self._schedule.identity.execution_identity,
            output_artifact_sha256=bundle.bundle_sha256,
            output_artifact_path=str(bundle_path),
            expected_cumulative_state_sha256=(
                None
                if artifacts.execution_evidence.cumulative_state_dependency_evidence
                is None
                else artifacts.execution_evidence.cumulative_state_dependency_evidence.expected_cumulative_state_sha256
            ),
            produced_cumulative_state_sha256=produced_state_sha256,
            produced_cumulative_state_artifact_path=produced_state_path,
        )
        append_attempt_record(
            self._attempt_ledger_path,
            record=attempt,
            dependencies=self._schedule.attempt_dependencies,
        )
        journal.record("terminal_attempt", attempt.to_artifact_dict())

    def _read_attempt_ledger(self) -> AttemptLedger:
        path = self._attempt_ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                handle.seek(0)
                payload = handle.read()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        ledger = AttemptLedger.from_jsonl_bytes(
            payload,
            run_id=self._schedule.identity.run_id,
            schedule_sha256=self._schedule.fingerprint,
            execution_identity=self._schedule.identity.execution_identity,
        )
        ledger.validate_dependencies(
            self._schedule.attempt_dependencies,
            verify_cumulative_state_artifacts=True,
        )
        return ledger


def _load_worker_inputs(config: ProductionExecutorFactoryConfig) -> _LoadedWorkerInputs:
    config_path = Path(config.infer_config_path).expanduser().resolve(strict=True)
    schedule_path = (
        Path(config.primary_schedule_artifact_path)
        .expanduser()
        .resolve(strict=True)
    )
    cohort_path = Path(config.cohort_ledger_path).expanduser().resolve(strict=True)
    source_path = Path(config.source_jsonl_path).expanduser().resolve(strict=True)
    selection_path = (
        Path(config.calibration_selection_receipt_path)
        .expanduser()
        .resolve(strict=True)
    )
    runtime_identity_path = (
        Path(config.source_runtime_identity_receipt_path)
        .expanduser()
        .resolve(strict=True)
    )
    attestation_path = Path(
        config.sampled_runtime_attestation_path
    ).expanduser().resolve(strict=True)
    attempt_path = Path(config.attempt_ledger_path).expanduser().resolve()
    primary_schedule_artifact = PrimaryScheduleArtifact.from_artifact_dict(
        _read_json_object(schedule_path, artifact_name="research schedule")
    )
    schedule = primary_schedule_artifact.schedule
    calibration_selection = CalibrationSelectionReceipt.from_artifact_dict(
        _read_json_object(
            selection_path,
            artifact_name="calibration selection receipt",
        )
    )
    source_runtime_identity = SourceRuntimeIdentityReceipt.from_artifact_dict(
        _read_json_object(
            runtime_identity_path,
            artifact_name="source runtime identity receipt",
        )
    )
    attested_policy_set = load_attested_sampling_policy_set(attestation_path)
    _validate_schedule_source_receipts(
        primary_schedule_artifact=primary_schedule_artifact,
        cohort_path=cohort_path,
        calibration_selection_path=selection_path,
        calibration_selection=calibration_selection,
        source_runtime_identity_path=runtime_identity_path,
        source_runtime_identity=source_runtime_identity,
        sampled_runtime_attestation_path=attestation_path,
        attested_policy_set=attested_policy_set,
    )
    if sha256_file(config_path) != source_runtime_identity.config_sha256:
        _fail(
            "inference config bytes differ from the schedule execution identity",
            "production_executor.config_identity",
        )
    cohort = CohortLedger.from_jsonl_bytes(cohort_path.read_bytes())
    source_digest = sha256_file(source_path)
    if {record.source_dataset_sha256 for record in cohort.records} != {
        source_digest
    }:
        _fail(
            "cohort records differ from the source dataset bytes",
            "production_executor.source_dataset_identity",
        )
    _validate_schedule_cohort(schedule=schedule, cohort=cohort)
    raw_examples = _load_exact_raw_examples(source_path=source_path, cohort=cohort)
    resolved = load_infer_config(config_path)
    return _LoadedWorkerInputs(
        infer_config=resolved.config,
        schedule=schedule,
        cohort=cohort,
        raw_examples_by_image_id=raw_examples,
        sampled_runtime_attestation_path=attestation_path,
        attempt_ledger_path=attempt_path,
    )


def _validate_schedule_source_receipts(
    *,
    primary_schedule_artifact: PrimaryScheduleArtifact,
    cohort_path: Path,
    calibration_selection_path: Path,
    calibration_selection: CalibrationSelectionReceipt,
    source_runtime_identity_path: Path,
    source_runtime_identity: SourceRuntimeIdentityReceipt,
    sampled_runtime_attestation_path: Path,
    attested_policy_set: Any,
) -> None:
    """Rebind production inputs to the canonical schedule-source receipts."""

    schedule = primary_schedule_artifact.schedule
    observed_file_digests = {
        primary_schedule_artifact.cohort_artifact_name: sha256_file(cohort_path),
        "calibration-selection-receipt.json": sha256_file(
            calibration_selection_path
        ),
        "sampled-runtime-attestation-aggregate.json": sha256_file(
            sampled_runtime_attestation_path
        ),
        "source-runtime-identity-receipt.json": sha256_file(
            source_runtime_identity_path
        ),
    }
    expected_source_hashes = dict(primary_schedule_artifact.source_hashes)
    mismatches = {
        name: {
            "expected": expected_source_hashes.get(name),
            "observed": observed,
        }
        for name, observed in observed_file_digests.items()
        if expected_source_hashes.get(name) != observed
    }
    direct_checks = {
        "cohort artifact digest": (
            observed_file_digests[primary_schedule_artifact.cohort_artifact_name],
            primary_schedule_artifact.cohort_artifact_sha256,
        ),
        "calibration selection artifact digest": (
            observed_file_digests["calibration-selection-receipt.json"],
            primary_schedule_artifact.calibration_selection_receipt_sha256,
        ),
        "calibration selection fingerprint": (
            calibration_selection.receipt_sha256,
            primary_schedule_artifact.calibration_selection_fingerprint,
        ),
        "source runtime identity artifact digest": (
            observed_file_digests["source-runtime-identity-receipt.json"],
            primary_schedule_artifact.source_runtime_identity_receipt_sha256,
        ),
        "source runtime identity fingerprint": (
            source_runtime_identity.receipt_sha256,
            primary_schedule_artifact.source_runtime_identity_fingerprint,
        ),
        "readiness ledger seal": (
            source_runtime_identity.ledger_seal_sha256,
            primary_schedule_artifact.readiness_ledger_seal_sha256,
        ),
        "sampled runtime attestation aggregate digest": (
            observed_file_digests[
                "sampled-runtime-attestation-aggregate.json"
            ],
            source_runtime_identity.sampled_runtime_attestation_aggregate_sha256,
        ),
        "sampled runtime attestation aggregate fingerprint": (
            attested_policy_set.aggregate_payload_fingerprint,
            source_runtime_identity.sampled_runtime_attestation_aggregate_fingerprint,
        ),
        "selected temperature": (
            calibration_selection.selected_temperature,
            schedule.identity.decode.temperature,
        ),
        "selected generation policy": (
            calibration_selection.selected_decode_generation_policy_fingerprint,
            schedule.identity.decode.canonical_generation_policy_sha256,
        ),
        "scheduled sampled-runtime attestation": (
            schedule.identity.decode.sampled_runtime_attestation_sha256,
            source_runtime_identity.sampled_runtime_attestation_aggregate_sha256,
        ),
        "processor contract": (
            schedule.identity.grid.canonical_spatial_receipt_contract_sha256,
            source_runtime_identity.processor_contract_sha256,
        ),
        "execution identity": (
            schedule.identity.execution_identity,
            source_runtime_identity.execution_identity,
        ),
        "attested policy set": (
            calibration_selection.attested_policy_set,
            attested_policy_set,
        ),
    }
    mismatches.update(
        {
            name: {"expected": expected, "observed": observed}
            for name, (observed, expected) in direct_checks.items()
            if observed != expected
        }
    )
    if mismatches:
        _fail(
            "production inputs differ from canonical schedule-source receipts",
            "production_executor.schedule_source_identity",
            mismatches=mismatches,
        )


def _load_runtime_binding(
    *,
    infer_config: InferConfig,
    sampled_runtime_attestation_path: Path,
    decode_provenance: DecodeProvenance,
) -> ProductionRuntimeBinding:
    runtime = assemble_runtime(infer_config)
    qwen = runtime.qwen
    model = _component(qwen, "model")
    tokenizer = _component(qwen, "tokenizer")
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(
        infer_config.generation.model_dump(mode="json")
    )
    backend = HFGenerateBackend(
        model=model,
        tokenizer=tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    capability = load_and_rebind_sampled_runtime_attestation_aggregate(
        sampled_runtime_attestation_path,
        decode_generation_policy_fingerprint=(
            decode_provenance.canonical_generation_policy_sha256
        ),
        backend=backend,
    )
    return ProductionRuntimeBinding(
        runtime=runtime,
        backend=backend,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
        verified_runtime_attestation=capability,
    )


def _load_exact_raw_examples(
    *, source_path: Path, cohort: CohortLedger
) -> dict[int, RawExample]:
    by_image_id: dict[int, RawExample] = {}
    for example in load_raw_examples(source_path):
        source_metadata = example.metadata.get("source", {})
        image_id = source_metadata.get("image_id")
        if isinstance(image_id, int):
            if image_id in by_image_id:
                _fail(
                    "source dataset repeats an image identifier",
                    "production_executor.source_duplicate_image",
                    image_id=image_id,
                )
            by_image_id[image_id] = example
    selected: dict[int, RawExample] = {}
    for record in cohort.records:
        example = by_image_id.get(record.image_id)
        if example is None:
            _fail(
                "cohort image is absent from the source dataset",
                "production_executor.source_image_missing",
                image_id=record.image_id,
            )
        observed = {
            "image_name": example.image.path.name,
            "source_row_index": example.source.row_number - 1,
            "source_row_sha256": example.source.row_sha256,
            "source_width": example.image.width,
            "source_height": example.image.height,
        }
        expected = {
            "image_name": Path(record.image_path).name,
            "source_row_index": record.source_row_index,
            "source_row_sha256": record.source_row_sha256,
            "source_width": record.source_width,
            "source_height": record.source_height,
        }
        if observed != expected or sha256_file(example.image.path) != record.image_sha256:
            _fail(
                "source example differs from its immutable cohort record",
                "production_executor.source_record_identity",
                image_id=record.image_id,
                observed=observed,
                expected=expected,
            )
        selected[record.image_id] = example
    return selected


def _validate_schedule_cohort(
    *, schedule: ResearchSchedule, cohort: CohortLedger
) -> None:
    baseline = tuple(
        request
        for request in schedule.requests
        if request.arm.arm_code == "FULL_SINGLE"
    )
    observed = tuple(
        (record.image_id, record.frozen_order, record.image_sha256)
        for record in cohort.records
    )
    expected = tuple(
        (request.image_id, request.image_frozen_order, request.image_sha256)
        for request in baseline
    )
    if observed != expected:
        _fail(
            "cohort order or image identity differs from the schedule baseline",
            "production_executor.cohort_schedule_identity",
        )
    if schedule.identity.grid.canonical_spatial_spec_sha256 != SpatialGridSpec().fingerprint:
        _fail(
            "schedule does not use the canonical four-by-four spatial grid",
            "production_executor.grid_identity",
        )


def _full_image_plan_row(
    *,
    request: ScheduledRequest,
    raw_example: RawExample,
    cohort_record: CohortImageRecord,
    materialization: MaterializedVisualInput,
    processor_identity: Mapping[str, Any],
) -> ImagePlanRow:
    grid = tuple(
        int(value)
        for value in materialization.image_grid_thw.detach().to("cpu")[0].tolist()
    )
    patch_size = int(processor_identity["patch_size"])
    merge_size = int(processor_identity["merge_size"])
    temporal_patch_size = int(processor_identity["temporal_patch_size"])
    raw_patch_rows = int(materialization.pixel_values.shape[0])
    return ImagePlanRow(
        row_id=raw_example.example_id,
        row_index=request.schedule_index,
        example_id=raw_example.example_id,
        image_path=str(cohort_record.image_path),
        declared_width=cohort_record.source_width,
        declared_height=cohort_record.source_height,
        decoded_width=cohort_record.source_width,
        decoded_height=cohort_record.source_height,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        expected_image_grid_thw=list(grid),
        observed_image_grid_thw=list(grid),
        raw_patch_rows=raw_patch_rows,
        merged_visual_tokens=raw_patch_rows // (merge_size * merge_size),
        do_resize=False,
        status="ok",
        error=None,
    )


def _call_diagnostics(
    *,
    prepared: PreparedRequest,
    artifacts: CompletedRequestArtifacts,
    decode_result: DecodeResult,
    elapsed_seconds: float,
    peak_device_memory_bytes: int,
    processor_identity: Mapping[str, Any],
) -> dict[str, Any]:
    rows = artifacts.row_diagnostics
    predictions = artifacts.normalized_predictions
    merge_size = int(processor_identity["merge_size"])
    image_tokens = int(prepared.visual_materialization.pixel_values.shape[0]) // (
        merge_size * merge_size
    )
    return {
        "attempted_row_count": len(rows),
        "controller_cap_count": 0,
        "error_count": 0,
        "generated_token_count": len(decode_result.generated_token_ids),
        "image_token_count": image_tokens,
        "invalid_row_count": sum(
            row.parse_status != "malformed" and row.validity_status == "invalid"
            for row in rows
        ),
        "malformed_row_count": sum(row.parse_status == "malformed" for row in rows),
        "natural_closure_count": int(decode_result.stop_reason == "im_end"),
        "non_owning_prediction_count": sum(
            prediction.ownership_status == "non_owning"
            for prediction in predictions
        ),
        "peak_device_memory_bytes": peak_device_memory_bytes,
        "prompt_token_count": len(prepared.prompt_record.prompt_token_ids),
        "token_cap_count": int(decode_result.stop_reason == "length"),
        "valid_prediction_ids": sorted(
            prediction.prediction_id for prediction in predictions
        ),
        "wall_time_seconds": elapsed_seconds,
    }


def _validate_result_batch(
    *,
    decode_requests: Sequence[DecodeRequest],
    decode_results: Sequence[DecodeResult],
) -> dict[str, DecodeResult]:
    expected_ids = tuple(request.request_id for request in decode_requests)
    observed_ids = tuple(result.request_id for result in decode_results)
    if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
        _fail(
            "backend result order differs from the sealed request batch",
            "production_executor.decode_result_order",
            expected_request_ids=list(expected_ids),
            observed_request_ids=list(observed_ids),
        )
    for request, result in zip(decode_requests, decode_results, strict=True):
        result.validate_for_scored()
        if tuple(result.prompt_token_ids) != tuple(request.prompt_token_ids):
            _fail(
                "backend result prompt differs from the executed request",
                "production_executor.decode_result_prompt",
                request_id=request.request_id,
            )
    return {result.request_id: result for result in decode_results}


def _processor_identity_payload(qwen: Any) -> dict[str, Any]:
    identity = _component(qwen, "processor_identity")
    to_artifact = getattr(identity, "to_artifact_dict", None)
    if not callable(to_artifact):
        _fail(
            "Qwen runtime lacks processor identity evidence",
            "production_executor.processor_identity",
        )
    payload = dict(to_artifact())
    for field in ("patch_size", "merge_size", "temporal_patch_size"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(
                "processor identity has invalid vision dimensions",
                "production_executor.processor_identity",
                field=field,
            )
    return payload


def _component(owner: Any, field: str) -> Any:
    value = owner.get(field) if isinstance(owner, Mapping) else getattr(owner, field, None)
    if value is None:
        _fail(
            "runtime component is absent",
            "production_executor.runtime_component",
            field=field,
        )
    return value


def _read_accepted_row_state(path: Path) -> tuple[str, ...]:
    payload = _read_json_object(path, artifact_name="accepted-row prefix state")
    if set(payload) != {"accepted_global_coordinate_rows", "schema_version"} or payload.get(
        "schema_version"
    ) != ACCEPTED_ROW_PREFIX_STATE_SCHEMA_VERSION:
        _fail(
            "accepted-row prefix state schema is invalid",
            "production_executor.cumulative_state_schema",
            path=str(path),
        )
    rows = payload.get("accepted_global_coordinate_rows")
    if not isinstance(rows, list) or any(
        not isinstance(row, str) or not row for row in rows
    ):
        _fail(
            "accepted-row prefix state rows are invalid",
            "production_executor.cumulative_state_rows",
            path=str(path),
        )
    return tuple(rows)


def _read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionExecutorContractError(
            f"{artifact_name} is not readable JSON",
            code="production_executor.json_artifact",
            context={"path": str(path), "artifact_name": artifact_name},
            cause=exc,
        ) from exc
    if not isinstance(value, dict):
        _fail(
            f"{artifact_name} must be a JSON object",
            "production_executor.json_artifact",
            path=str(path),
        )
    return value


def _write_once_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _reset_peak_memory_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_if_available() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _fail(message: str, code: str, **context: Any) -> None:
    raise ProductionExecutorContractError(message, code=code, context=context)


__all__ = [
    "CompletedRequestArtifacts",
    "PreparedRequest",
    "ProductionBatchExecutor",
    "ProductionExecutorContractError",
    "ProductionExecutorFactory",
    "ProductionExecutorFactoryConfig",
    "ProductionRuntimeBinding",
]
