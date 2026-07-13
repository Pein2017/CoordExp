"""Execution skeleton for Spatial Scope and History Disentanglement research.

The module deliberately owns orchestration rather than scientific policy.  A
``ResearchSchedule`` already seals request order, physical batch membership,
sampling seeds, arm semantics, and cumulative dependencies.  This runner only
maps whole sealed batches to persistent workers, enforces cumulative cell-wave
barriers, and makes the append-only artifact ordering executable and testable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import multiprocessing
import os
from pathlib import Path
import queue
import time
import traceback
from typing import Any, cast, Literal, Protocol

from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    canonical_json_text,
)
from src.analysis.spatial_scope_history.schedule import (
    RequestBatch,
    ResearchSchedule,
    ResumePlan,
)
from src.common.errors import ArtifactContractError
from src.config.inference import InferConfig
from src.config.models import TemplateConfig
from src.data import RawExample
from src.inference.backend import HFGenerateBackend
from src.inference.prompt import (
    AssistantContinuation,
    PromptRecord,
    build_prompt_record,
)
from src.inference.runtime import InferenceRuntime, assemble_runtime


RUNNER_PLAN_SCHEMA_VERSION = "spatial_scope_history.runner_plan.v1"
CALL_INTENT_SCHEMA_VERSION = "spatial_scope_history.call_intent.v2"


class RunnerContractError(ArtifactContractError):
    """A fail-closed violation of the research execution contract."""


@dataclass(frozen=True)
class WorkerDeviceAssignment:
    """One physical GPU token mapped to the worker-local logical ``cuda:0``.

    ``GPU`` means graphics processing unit.  One persistent worker process owns
    one assignment for its complete lifetime; batches are never split or
    repacked between assignments.
    """

    worker_index: int
    physical_gpu_token: str
    logical_device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.worker_index < 0:
            _fail("worker index must be nonnegative", "runner.worker_index")
        if not self.physical_gpu_token.strip() or "," in self.physical_gpu_token:
            _fail(
                "one worker requires exactly one physical GPU token",
                "runner.physical_gpu_token",
            )
        if self.logical_device != "cuda:0":
            _fail(
                "a GPU-isolated worker must use logical cuda:0",
                "runner.logical_device",
            )

    @property
    def environment(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": self.physical_gpu_token}


@dataclass(frozen=True)
class WorkerProcessSpec:
    """Pure launch contract for one persistent model/backend worker process."""

    assignment: WorkerDeviceAssignment
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.command or any(not item for item in self.command):
            _fail("worker command must be nonempty", "runner.worker_command")

    @property
    def environment(self) -> dict[str, str]:
        return self.assignment.environment


@dataclass(frozen=True)
class PersistentInferenceWorker:
    """One process-local Qwen runtime and Hugging Face generation backend.

    Construction calls the canonical ``assemble_runtime`` function exactly
    once.  Production launchers must create one instance per isolated worker
    process after applying ``CUDA_VISIBLE_DEVICES``; coordinator tests should
    inject a fake batch executor instead and never construct this class.
    """

    runtime: InferenceRuntime
    backend: HFGenerateBackend
    logical_device: str = "cuda:0"

    @classmethod
    def from_config(cls, config: InferConfig) -> PersistentInferenceWorker:
        runtime = assemble_runtime(config)
        qwen = runtime.qwen
        model = qwen.get("model") if isinstance(qwen, Mapping) else qwen.model
        tokenizer = (
            qwen.get("tokenizer") if isinstance(qwen, Mapping) else qwen.tokenizer
        )
        if model is None or tokenizer is None:
            _fail(
                "assembled runtime lacks model or tokenizer",
                "runner.runtime_components",
            )
        return cls(
            runtime=runtime,
            backend=HFGenerateBackend(model=model, tokenizer=tokenizer),
        )


def build_cumulative_prompt_record(
    raw_example: RawExample,
    template_config: TemplateConfig,
    *,
    processor: Any,
    row_index: int,
    accepted_global_coordinate_rows: str,
    max_prompt_tokens: int | None = None,
) -> PromptRecord:
    """Build cumulative history through the canonical open-assistant interface.

    The accepted rows remain inside the assistant turn through
    ``AssistantContinuation``.  This helper intentionally provides no manual
    token concatenation path.
    """

    return build_prompt_record(
        raw_example,
        template_config,
        processor=processor,
        row_index=row_index,
        assistant_continuation=AssistantContinuation(
            text=accepted_global_coordinate_rows
        ),
        max_prompt_tokens=max_prompt_tokens,
    )


@dataclass(frozen=True)
class BatchDispatch:
    """One indivisible sealed physical batch assigned to one worker."""

    wave_index: int
    execution_wave_partition: str
    batch: RequestBatch
    worker: WorkerDeviceAssignment
    run_id: str
    schedule_sha256: str


@dataclass(frozen=True)
class DispatchWave:
    """Batches allowed to execute before the next dependency barrier."""

    wave_index: int
    execution_wave_partition: str
    dispatches: tuple[BatchDispatch, ...]


@dataclass(frozen=True)
class CoordinatorPlan:
    """Deterministic whole-batch plan with explicit cumulative barriers."""

    schedule_sha256: str
    physical_batch_plan_sha256: str
    workers: tuple[WorkerDeviceAssignment, ...]
    waves: tuple[DispatchWave, ...]
    schema_version: str = RUNNER_PLAN_SCHEMA_VERSION

    @property
    def batches(self) -> tuple[RequestBatch, ...]:
        return tuple(
            dispatch.batch for wave in self.waves for dispatch in wave.dispatches
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "physical_batch_plan_sha256": self.physical_batch_plan_sha256,
            "schedule_sha256": self.schedule_sha256,
            "schema_version": self.schema_version,
            "waves": [
                {
                    "execution_wave_partition": wave.execution_wave_partition,
                    "dispatches": [
                        {
                            "batch_index": dispatch.batch.batch_index,
                            "cardinality": dispatch.batch.cardinality,
                            "physical_batch_sha256": (
                                dispatch.batch.physical_batch_sha256
                            ),
                            "request_ids": list(dispatch.batch.request_ids),
                            "worker_index": dispatch.worker.worker_index,
                        }
                        for dispatch in wave.dispatches
                    ],
                    "wave_index": wave.wave_index,
                }
                for wave in self.waves
            ],
            "workers": [
                {
                    "environment": worker.environment,
                    "logical_device": worker.logical_device,
                    "physical_gpu_token": worker.physical_gpu_token,
                    "worker_index": worker.worker_index,
                }
                for worker in self.workers
            ],
        }


def build_worker_assignments(
    physical_gpu_tokens: Sequence[str],
) -> tuple[WorkerDeviceAssignment, ...]:
    """Build stable one-process-per-GPU assignments in caller-provided order."""

    if not physical_gpu_tokens:
        _fail("at least one physical GPU token is required", "runner.no_workers")
    if len(set(physical_gpu_tokens)) != len(physical_gpu_tokens):
        _fail("physical GPU tokens must be unique", "runner.duplicate_gpu")
    return tuple(
        WorkerDeviceAssignment(index, token)
        for index, token in enumerate(physical_gpu_tokens)
    )


def build_coordinator_plan(
    schedule: ResearchSchedule,
    *,
    physical_gpu_tokens: Sequence[str],
    attempt_ledger: AttemptLedger | None = None,
) -> CoordinatorPlan:
    """Create a deterministic plan without loading a model or touching CUDA."""

    ledger = attempt_ledger or AttemptLedger(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
    )
    resume_plan = schedule.resume_batches(ledger)
    return _build_coordinator_plan_from_resume(
        schedule,
        resume_plan=resume_plan,
        physical_gpu_tokens=physical_gpu_tokens,
    )


def _build_coordinator_plan_from_resume(
    schedule: ResearchSchedule,
    *,
    resume_plan: ResumePlan,
    physical_gpu_tokens: Sequence[str],
) -> CoordinatorPlan:
    """Consume only the dependency-safe whole batches admitted by the schedule."""

    if (
        resume_plan.physical_batch_plan_sha256
        != schedule.physical_batch_plan.fingerprint
    ):
        _fail(
            "resume plan belongs to another physical batch plan",
            "runner.resume_plan_identity",
        )
    workers = build_worker_assignments(physical_gpu_tokens)
    full_wave_indexes = {
        partition: wave_index
        for wave_index, (partition, _) in enumerate(
            _group_batches_by_sealed_execution_wave(schedule.batches())
        )
    }
    grouped = _group_batches_by_sealed_execution_wave(resume_plan.runnable_batches)
    waves: list[DispatchWave] = []
    for partition, wave_batches in grouped:
        wave_index = full_wave_indexes[partition]
        dispatches = tuple(
            BatchDispatch(
                wave_index=wave_index,
                execution_wave_partition=partition,
                batch=batch,
                worker=workers[batch.batch_index % len(workers)],
                run_id=schedule.identity.run_id,
                schedule_sha256=schedule.fingerprint,
            )
            for batch in wave_batches
        )
        waves.append(
            DispatchWave(
                wave_index=wave_index,
                execution_wave_partition=partition,
                dispatches=dispatches,
            )
        )
    return CoordinatorPlan(
        schedule_sha256=schedule.fingerprint,
        physical_batch_plan_sha256=schedule.physical_batch_plan.fingerprint,
        workers=workers,
        waves=tuple(waves),
    )


def _group_batches_by_sealed_execution_wave(
    batches: Sequence[RequestBatch],
) -> tuple[tuple[str, tuple[RequestBatch, ...]], ...]:
    grouped: list[tuple[str, list[RequestBatch]]] = []
    for batch in batches:
        partition = batch.execution_wave_partition
        if not grouped or grouped[-1][0] != partition:
            if any(existing_partition == partition for existing_partition, _ in grouped):
                _fail(
                    "sealed execution-wave partition is not contiguous",
                    "runner.execution_wave_contiguity",
                    execution_wave_partition=partition,
                )
            grouped.append((partition, []))
        grouped[-1][1].append(batch)
    return tuple((partition, tuple(items)) for partition, items in grouped)


ArtifactStage = Literal[
    "call_intent",
    "materialized_input",
    "execution_evidence",
    "decode_result",
    "parse_score",
    "cumulative_state",
    "terminal_attempt",
]
_STAGE_ORDER: Mapping[ArtifactStage, int] = {
    "call_intent": 0,
    "materialized_input": 1,
    "execution_evidence": 2,
    "decode_result": 3,
    "parse_score": 4,
    "cumulative_state": 5,
    "terminal_attempt": 6,
}


@dataclass
class RequestArtifactJournal:
    """Write-once, fsync-backed artifact ordering for one scheduled request."""

    root: Path
    dispatch: BatchDispatch
    request_id: str
    _last_stage: int = -1
    _terminal_written: bool = False
    _terminal_status: AttemptStatus | None = None

    @property
    def request_directory(self) -> Path:
        digest = hashlib.sha256(self.request_id.encode("utf-8")).hexdigest()
        return self.root / "calls" / digest

    @property
    def terminal_written(self) -> bool:
        return self._terminal_written

    @property
    def terminal_status(self) -> AttemptStatus | None:
        return self._terminal_status

    @classmethod
    def open_after_call_intent(
        cls,
        *,
        root: Path,
        dispatch: BatchDispatch,
        request_id: str,
    ) -> RequestArtifactJournal:
        """Open child-side journal state after coordinator-fsynced intent."""

        journal = cls(root=root, dispatch=dispatch, request_id=request_id)
        intent_path = journal._artifact_path("call_intent")
        if not intent_path.is_file():
            _fail(
                "worker cannot execute before its coordinator-owned call intent",
                "runner.missing_call_intent",
                request_id=request_id,
            )
        journal._validate_existing_call_intent(intent_path)
        journal._last_stage = _STAGE_ORDER["call_intent"]
        return journal

    def write_call_intent(self) -> Path:
        if self._last_stage != -1:
            _fail(
                "call intent cannot be written twice through one journal handle",
                "runner.call_intent_handle_reuse",
                request_id=self.request_id,
            )
        path = self._artifact_path("call_intent")
        payload = self._artifact_bytes("call_intent", self._call_intent_payload())
        try:
            _write_once_fsync(path, payload)
        except FileExistsError:
            self._validate_existing_call_intent(path, expected_bytes=payload)
        self._last_stage = _STAGE_ORDER["call_intent"]
        return path

    def _request(self) -> Any:
        return next(
            item
            for item in self.dispatch.batch.requests
            if item.request_id == self.request_id
        )

    def _call_intent_payload(self) -> dict[str, Any]:
        request = self._request()
        return {
            "batch_index": self.dispatch.batch.batch_index,
            "cell_index": request.cell_index,
            "execution_identity_sha256": request.execution_identity_sha256,
            "execution_wave_partition": self.dispatch.execution_wave_partition,
            "physical_batch_plan_sha256": (
                self.dispatch.batch.physical_batch_plan_sha256
            ),
            "physical_batch_sha256": self.dispatch.batch.physical_batch_sha256,
            "request_id": self.request_id,
            "run_id": self.dispatch.run_id,
            "sampling_seed": request.sampling_seed,
            "schedule_identity_sha256": request.schedule_identity_sha256,
            "schedule_index": request.schedule_index,
            "schedule_sha256": self.dispatch.schedule_sha256,
            "schema_version": CALL_INTENT_SCHEMA_VERSION,
            "wave_index": self.dispatch.wave_index,
            "worker_environment": self.dispatch.worker.environment,
            "worker_logical_device": self.dispatch.worker.logical_device,
        }

    def _artifact_path(self, stage: ArtifactStage) -> Path:
        return self.request_directory / f"{_STAGE_ORDER[stage]:02d}-{stage}.json"

    def _artifact_bytes(
        self, stage: ArtifactStage, payload: Mapping[str, Any]
    ) -> bytes:
        body = {
            "artifact_stage": stage,
            "payload": dict(payload),
            "request_id": self.request_id,
        }
        return (canonical_json_text(body) + "\n").encode("utf-8")

    def _validate_existing_call_intent(
        self,
        path: Path,
        *,
        expected_bytes: bytes | None = None,
    ) -> None:
        canonical_bytes = expected_bytes or self._artifact_bytes(
            "call_intent", self._call_intent_payload()
        )
        if path.read_bytes() != canonical_bytes:
            _fail(
                "existing call intent differs from the canonical same-run intent",
                "runner.call_intent_drift",
                request_id=self.request_id,
            )
        later_entries = sorted(
            item.name for item in self.request_directory.iterdir() if item != path
        )
        if later_entries:
            _fail(
                "existing call intent has a partially advanced request journal",
                "runner.call_intent_partial_journal",
                request_id=self.request_id,
                later_entries=later_entries,
            )

    def record(self, stage: ArtifactStage, payload: Mapping[str, Any]) -> Path:
        """Write one stage after validating the canonical per-request order."""

        if self._terminal_written:
            _fail(
                "no artifact may follow the terminal attempt",
                "runner.artifact_after_terminal",
                request_id=self.request_id,
            )
        stage_index = _STAGE_ORDER[stage]
        if stage_index <= self._last_stage:
            _fail(
                "artifact stage is duplicated or out of order",
                "runner.artifact_order",
                request_id=self.request_id,
                stage=stage,
            )
        if stage_index != self._last_stage + 1:
            cumulative_optional = stage == "terminal_attempt" and self._last_stage == 4
            if not cumulative_optional:
                _fail(
                    "artifact stage skipped a required predecessor",
                    "runner.artifact_missing_predecessor",
                    request_id=self.request_id,
                    stage=stage,
                )
        canonical_payload = dict(payload)
        terminal_attempt: AttemptRecord | None = None
        if stage == "terminal_attempt":
            terminal_attempt = AttemptRecord.from_artifact_dict(payload)
            self._validate_terminal_attempt_identity(terminal_attempt)
            canonical_payload = terminal_attempt.to_artifact_dict()
            request = self._request()
            completed = terminal_attempt.attempt_status == "completed"
            has_cumulative_state = self._last_stage == _STAGE_ORDER["cumulative_state"]
            if completed and request.arm.cumulative_dependency != has_cumulative_state:
                _fail(
                    "completed terminal attempt has the wrong cumulative-state sequence",
                    "runner.cumulative_state_order",
                    request_id=self.request_id,
                )
        path = self._artifact_path(stage)
        _write_once_fsync(path, self._artifact_bytes(stage, canonical_payload))
        self._last_stage = stage_index
        if stage == "terminal_attempt":
            assert terminal_attempt is not None
            self._terminal_written = True
            self._terminal_status = terminal_attempt.attempt_status
        return path

    def _validate_terminal_attempt_identity(self, attempt: AttemptRecord) -> None:
        request = self._request()
        observed = {
            "run_id": attempt.run_id,
            "schedule_sha256": attempt.schedule_sha256,
            "request_id": attempt.request_id,
            "physical_batch_plan_sha256": attempt.physical_batch_plan_sha256,
            "physical_batch_sha256": attempt.physical_batch_sha256,
            "physical_batch_index": attempt.physical_batch_index,
            "execution_identity_sha256": attempt.execution_identity.fingerprint,
        }
        expected = {
            "run_id": self.dispatch.run_id,
            "schedule_sha256": self.dispatch.schedule_sha256,
            "request_id": self.request_id,
            "physical_batch_plan_sha256": (
                self.dispatch.batch.physical_batch_plan_sha256
            ),
            "physical_batch_sha256": self.dispatch.batch.physical_batch_sha256,
            "physical_batch_index": self.dispatch.batch.batch_index,
            "execution_identity_sha256": request.execution_identity_sha256,
        }
        drifted_fields = sorted(
            field for field in expected if observed[field] != expected[field]
        )
        if drifted_fields:
            _fail(
                "terminal attempt identity differs from its sealed request",
                "runner.terminal_attempt_identity",
                request_id=self.request_id,
                drifted_fields=drifted_fields,
            )


@dataclass(frozen=True)
class BatchExecutionContext:
    """Sealed batch context passed to one injected persistent-worker callback."""

    dispatch: BatchDispatch
    journals: Mapping[str, RequestArtifactJournal]


class BatchExecutor(Protocol):
    """Injected worker callback that executes one whole sealed request batch."""

    def __call__(self, context: BatchExecutionContext) -> None: ...


class WorkerExecutorFactory(Protocol):
    """Spawn-safe factory constructed once inside each persistent worker."""

    def __call__(
        self,
        assignment: WorkerDeviceAssignment,
        factory_config: Mapping[str, Any],
    ) -> BatchExecutor: ...


@dataclass(frozen=True)
class WorkerStartupReceipt:
    """Observed persistent-process identity and isolated device environment."""

    worker_index: int
    process_id: int
    physical_gpu_token: str
    visible_device_environment: str
    logical_device: str


@dataclass(frozen=True)
class RequestTerminalReceipt:
    """One request's legal scientific terminal outcome inside a sealed batch."""

    request_id: str
    attempt_status: AttemptStatus


@dataclass(frozen=True)
class BatchProcessReceipt:
    """One completed whole-batch process result with wall-clock timestamps."""

    task_id: str
    worker_index: int
    process_id: int
    physical_batch_sha256: str
    execution_wave_partition: str
    started_at_unix_nanoseconds: int
    finished_at_unix_nanoseconds: int
    request_terminal_receipts: tuple[RequestTerminalReceipt, ...]


@dataclass(frozen=True)
class ProcessPoolRunReceipt:
    """Observable lifecycle evidence returned by the persistent worker pool."""

    worker_startups: tuple[WorkerStartupReceipt, ...]
    batch_receipts: tuple[BatchProcessReceipt, ...]


class PersistentWorkerPool:
    """Spawn-based persistent worker pool with one executor per physical GPU.

    ``GPU`` means graphics processing unit.  The pool constructs each injected
    executor exactly once after inheriting ``CUDA_VISIBLE_DEVICES`` before
    spawn bootstrap imports, then reasserting it inside that child.
    ``execute_plan`` may be called repeatedly so a higher-level coordinator can
    rebuild dependency-safe ``ResumePlan`` frontiers without reloading models.
    """

    def __init__(
        self,
        *,
        workers: Sequence[WorkerDeviceAssignment],
        executor_factory: WorkerExecutorFactory,
        factory_config: Mapping[str, Any] | None = None,
        result_timeout_seconds: float = 600.0,
    ) -> None:
        if not workers:
            _fail("persistent pool requires workers", "runner.pool_no_workers")
        self._workers = tuple(workers)
        self._executor_factory = executor_factory
        self._factory_config = dict(factory_config or {})
        self._result_timeout_seconds = result_timeout_seconds
        self._context = multiprocessing.get_context("spawn")
        self._result_queue: Any = self._context.Queue()
        self._task_queues: dict[int, Any] = {}
        self._processes: dict[int, Any] = {}
        self._startup_receipts: tuple[WorkerStartupReceipt, ...] = ()
        self._closed = False
        self._started = False

    @property
    def startup_receipts(self) -> tuple[WorkerStartupReceipt, ...]:
        return self._startup_receipts

    def start(self) -> None:
        if self._started:
            return
        startup_by_worker: dict[int, WorkerStartupReceipt] = {}
        try:
            for worker in self._workers:
                task_queue = self._context.Queue()
                process = self._context.Process(
                    target=_persistent_worker_main,
                    args=(
                        worker,
                        task_queue,
                        self._result_queue,
                        self._executor_factory,
                        self._factory_config,
                    ),
                    name=f"spatial-scope-worker-{worker.worker_index}",
                )
                worker_environment = worker.environment
                missing_environment_value = object()
                parent_environment = {
                    key: os.environ.get(key, missing_environment_value)
                    for key in worker_environment
                }
                try:
                    try:
                        os.environ.update(worker_environment)
                        process.start()
                    finally:
                        for key, value in parent_environment.items():
                            if value is missing_environment_value:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = cast(str, value)
                except BaseException:
                    if process.pid is not None:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=10.0)
                    task_queue.close()
                    raise
                self._task_queues[worker.worker_index] = task_queue
                self._processes[worker.worker_index] = process
            while len(startup_by_worker) != len(self._workers):
                message = self._next_message()
                if message["kind"] == "worker_error":
                    self._raise_worker_error(message)
                if message["kind"] != "ready":
                    _fail(
                        "worker emitted a batch result before readiness",
                        "runner.worker_startup_protocol",
                    )
                receipt = WorkerStartupReceipt(**message["receipt"])
                if receipt.worker_index in startup_by_worker:
                    _fail(
                        "worker emitted duplicate readiness",
                        "runner.worker_duplicate_ready",
                        worker_index=receipt.worker_index,
                    )
                startup_by_worker[receipt.worker_index] = receipt
        except BaseException:
            self.close()
            raise
        self._startup_receipts = tuple(
            startup_by_worker[index] for index in sorted(startup_by_worker)
        )
        self._started = True

    def execute_plan(
        self,
        plan: CoordinatorPlan,
        *,
        artifact_root: Path,
        after_wave: Callable[[DispatchWave], None] | None = None,
    ) -> ProcessPoolRunReceipt:
        if not self._started:
            self.start()
        if self._closed:
            _fail("persistent pool is closed", "runner.pool_closed")
        if tuple(plan.workers) != self._workers:
            _fail(
                "coordinator plan worker identities differ from persistent pool",
                "runner.pool_worker_identity",
            )
        batch_receipts: list[BatchProcessReceipt] = []
        workers_by_index = {worker.worker_index: worker for worker in self._workers}
        if len(workers_by_index) != len(self._workers):
            _fail(
                "persistent pool worker indexes must be unique",
                "runner.pool_worker_identity",
            )
        for wave in plan.waves:
            pending_by_worker = {
                worker.worker_index: deque() for worker in self._workers
            }
            for dispatch in wave.dispatches:
                worker_index = dispatch.worker.worker_index
                if (
                    worker_index not in pending_by_worker
                    or dispatch.worker != workers_by_index[worker_index]
                ):
                    _fail(
                        "wave dispatch uses an unplanned worker",
                        "runner.wave_worker_identity",
                        worker_index=worker_index,
                    )
                pending_by_worker[worker_index].append(dispatch)
            issued_task_ids: set[str] = set()
            in_flight_by_worker: dict[int, tuple[str, BatchDispatch]] = {}

            def submit_next(worker_index: int) -> None:
                if worker_index in in_flight_by_worker:
                    _fail(
                        "worker already has an in-flight physical batch",
                        "runner.worker_in_flight_limit",
                        worker_index=worker_index,
                    )
                pending = pending_by_worker[worker_index]
                if not pending:
                    return
                dispatch = pending.popleft()
                task_id = (
                    f"wave-{wave.wave_index}:batch-{dispatch.batch.batch_index}:"
                    f"{dispatch.batch.physical_batch_sha256}"
                )
                if task_id in issued_task_ids:
                    _fail(
                        "wave contains a duplicate physical batch task",
                        "runner.worker_task_identity",
                        task_id=task_id,
                    )
                _write_dispatch_call_intents(dispatch, artifact_root=artifact_root)
                self._task_queues[worker_index].put(
                    {
                        "artifact_root": str(artifact_root),
                        "dispatch": dispatch,
                        "kind": "execute",
                        "task_id": task_id,
                    }
                )
                issued_task_ids.add(task_id)
                in_flight_by_worker[worker_index] = (task_id, dispatch)

            for worker in self._workers:
                submit_next(worker.worker_index)

            completed_count = 0
            while completed_count != len(wave.dispatches):
                message = self._next_message()
                if message["kind"] == "worker_error":
                    self._raise_worker_error(message)
                if message["kind"] != "batch_completed":
                    _fail(
                        "worker emitted an unexpected lifecycle message",
                        "runner.worker_message_protocol",
                        message_kind=message["kind"],
                    )
                receipt = BatchProcessReceipt(**message["receipt"])
                expected = in_flight_by_worker.get(receipt.worker_index)
                if expected is None:
                    _fail(
                        "worker result does not belong uniquely to the current wave",
                        "runner.worker_result_identity",
                        task_id=receipt.task_id,
                    )
                expected_task_id, expected_dispatch = expected
                if (
                    receipt.task_id != expected_task_id
                    or receipt.physical_batch_sha256
                    != expected_dispatch.batch.physical_batch_sha256
                    or receipt.execution_wave_partition != wave.execution_wave_partition
                ):
                    _fail(
                        "worker result differs from its in-flight physical batch",
                        "runner.worker_result_identity",
                        task_id=receipt.task_id,
                        expected_task_id=expected_task_id,
                    )
                del in_flight_by_worker[receipt.worker_index]
                completed_count += 1
                batch_receipts.append(receipt)
                submit_next(receipt.worker_index)
            if after_wave is not None:
                after_wave(wave)
        return ProcessPoolRunReceipt(
            worker_startups=self._startup_receipts,
            batch_receipts=tuple(batch_receipts),
        )

    def _next_message(self) -> Mapping[str, Any]:
        deadline = time.monotonic() + self._result_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail(
                    "persistent worker result timed out",
                    "runner.worker_timeout",
                    dead_worker_indexes=[],
                )
            try:
                return self._result_queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                dead_workers = [
                    index
                    for index, process in self._processes.items()
                    if not process.is_alive()
                ]
                if dead_workers:
                    _fail(
                        "persistent worker exited without a result",
                        "runner.worker_exited",
                        dead_worker_indexes=dead_workers,
                    )

    def _raise_worker_error(self, message: Mapping[str, Any]) -> None:
        _fail(
            "persistent worker failed closed",
            "runner.worker_error",
            worker_index=message.get("worker_index"),
            task_id=message.get("task_id"),
            error=message.get("error"),
            traceback=message.get("traceback"),
        )

    def close(self) -> None:
        if self._closed:
            return
        for task_queue in self._task_queues.values():
            task_queue.put({"kind": "shutdown"})
        for process in self._processes.values():
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
        for task_queue in self._task_queues.values():
            task_queue.close()
        self._result_queue.close()
        self._closed = True

    def __enter__(self) -> PersistentWorkerPool:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _persistent_worker_main(
    assignment: WorkerDeviceAssignment,
    task_queue: Any,
    result_queue: Any,
    executor_factory: WorkerExecutorFactory,
    factory_config: Mapping[str, Any],
) -> None:
    os.environ.update(assignment.environment)
    process_id = os.getpid()
    try:
        executor = executor_factory(assignment, factory_config)
        result_queue.put(
            {
                "kind": "ready",
                "receipt": {
                    "logical_device": assignment.logical_device,
                    "physical_gpu_token": assignment.physical_gpu_token,
                    "process_id": process_id,
                    "visible_device_environment": os.environ.get(
                        "CUDA_VISIBLE_DEVICES", ""
                    ),
                    "worker_index": assignment.worker_index,
                },
            }
        )
    except BaseException as exc:
        _put_worker_error(
            result_queue,
            assignment=assignment,
            task_id=None,
            exception=exc,
        )
        return
    while True:
        message = task_queue.get()
        if message["kind"] == "shutdown":
            return
        task_id = str(message["task_id"])
        try:
            dispatch: BatchDispatch = message["dispatch"]
            journals = {
                request.request_id: RequestArtifactJournal.open_after_call_intent(
                    root=Path(message["artifact_root"]),
                    dispatch=dispatch,
                    request_id=request.request_id,
                )
                for request in dispatch.batch.requests
            }
            started = time.time_ns()
            executor(BatchExecutionContext(dispatch=dispatch, journals=journals))
            unterminated = sorted(
                request_id
                for request_id, journal in journals.items()
                if not journal.terminal_written
            )
            if unterminated:
                _fail(
                    "worker batch lacks terminal attempts",
                    "runner.worker_batch_terminal",
                    request_ids=unterminated,
                )
            finished = time.time_ns()
            result_queue.put(
                {
                    "kind": "batch_completed",
                    "receipt": {
                        "execution_wave_partition": (
                            dispatch.execution_wave_partition
                        ),
                        "finished_at_unix_nanoseconds": finished,
                        "physical_batch_sha256": (
                            dispatch.batch.physical_batch_sha256
                        ),
                        "process_id": process_id,
                        "request_terminal_receipts": tuple(
                            RequestTerminalReceipt(
                                request_id=request_id,
                                attempt_status=cast(
                                    AttemptStatus, journals[request_id].terminal_status
                                ),
                            )
                            for request_id in dispatch.batch.request_ids
                        ),
                        "started_at_unix_nanoseconds": started,
                        "task_id": task_id,
                        "worker_index": assignment.worker_index,
                    },
                }
            )
        except BaseException as exc:
            _put_worker_error(
                result_queue,
                assignment=assignment,
                task_id=task_id,
                exception=exc,
            )


def _put_worker_error(
    result_queue: Any,
    *,
    assignment: WorkerDeviceAssignment,
    task_id: str | None,
    exception: BaseException,
) -> None:
    result_queue.put(
        {
            "error": f"{type(exception).__name__}: {exception}",
            "kind": "worker_error",
            "task_id": task_id,
            "traceback": traceback.format_exc(),
            "worker_index": assignment.worker_index,
        }
    )


def _write_dispatch_call_intents(
    dispatch: BatchDispatch,
    *,
    artifact_root: Path,
) -> None:
    for request in dispatch.batch.requests:
        RequestArtifactJournal(
            root=artifact_root,
            dispatch=dispatch,
            request_id=request.request_id,
        ).write_call_intent()


def execute_coordinator_plan(
    plan: CoordinatorPlan,
    *,
    artifact_root: Path,
    executor: BatchExecutor,
    after_wave: Callable[[DispatchWave], None] | None = None,
) -> None:
    """Execute waves sequentially while leaving within-wave parallelism injectable."""

    for wave in plan.waves:
        for dispatch in wave.dispatches:
            _write_dispatch_call_intents(dispatch, artifact_root=artifact_root)
            journals = {
                request.request_id: RequestArtifactJournal.open_after_call_intent(
                    root=artifact_root,
                    dispatch=dispatch,
                    request_id=request.request_id,
                )
                for request in dispatch.batch.requests
            }
            executor(BatchExecutionContext(dispatch=dispatch, journals=journals))
            missing = sorted(
                request_id
                for request_id, journal in journals.items()
                if not journal.terminal_written
            )
            if missing:
                _fail(
                    "batch executor returned without terminal attempts",
                    "runner.batch_missing_terminal",
                    request_ids=missing,
                )
        if after_wave is not None:
            after_wave(wave)


def _write_once_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _fail(message: str, code: str, **context: Any) -> None:
    raise RunnerContractError(message, code=code, context=context)
