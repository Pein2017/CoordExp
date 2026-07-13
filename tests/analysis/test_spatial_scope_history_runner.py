from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from src.analysis.spatial_scope_history.calibration import PrimaryScheduleArtifact
from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptRecord,
    CohortImageRecord,
    CohortLedger,
    ExecutionIdentityBundle,
    sha256_payload,
)
from src.analysis.spatial_scope_history.runner import (
    BatchExecutionContext,
    RunnerContractError,
    PersistentWorkerPool,
    RequestArtifactJournal,
    build_coordinator_plan,
    build_cumulative_prompt_record,
    build_worker_assignments,
    execute_coordinator_plan,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    GridProvenance,
    PhysicalBatchDefinition,
    RequestBatch,
    ResearchSchedule,
    ResumePlan,
)
from src.common.errors import ArtifactContractError


_SPAWN_BOOTSTRAP_CAPTURE_ROOT_ENVIRONMENT_VARIABLE = (
    "COORDEXP_TEST_SPAWN_BOOTSTRAP_CAPTURE_ROOT"
)


def _capture_spawn_bootstrap_cuda_visibility() -> None:
    """Record the inherited device scope while spawn imports this module."""

    capture_root = os.environ.get(
        _SPAWN_BOOTSTRAP_CAPTURE_ROOT_ENVIRONMENT_VARIABLE
    )
    if capture_root is None:
        return
    root = Path(capture_root)
    root.mkdir(parents=True, exist_ok=True)
    capture_path = root / f"bootstrap-{os.getpid()}.json"
    capture_path.write_text(
        json.dumps(
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "process_id": os.getpid(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


_capture_spawn_bootstrap_cuda_visibility()


_FAKE_EXECUTION_IDENTITY = ExecutionIdentityBundle(
    code_sha256="1" * 64,
    config_sha256="2" * 64,
    ledger_sha256="3" * 64,
    runtime_sha256="4" * 64,
)
_FAKE_SCHEDULE_SHA256 = "b" * 64


@dataclass(frozen=True)
class _Arm:
    cumulative_dependency: bool


@dataclass(frozen=True)
class _Request:
    request_id: str
    arm: _Arm
    cell_index: int | None
    sampling_seed: int
    schedule_index: int
    schedule_identity_sha256: str
    execution_identity_sha256: str
    predecessor_request_id: str | None = None


@dataclass(frozen=True)
class _PhysicalPlan:
    fingerprint: str = "a" * 64


class _Schedule:
    def __init__(self, batches: tuple[RequestBatch, ...]) -> None:
        self._batches = batches
        self.identity = SimpleNamespace(
            run_id="run-one", execution_identity=_FAKE_EXECUTION_IDENTITY
        )
        self.physical_batch_plan = _PhysicalPlan()
        self.attempt_dependencies: tuple[Any, ...] = ()
        self.fingerprint = _FAKE_SCHEDULE_SHA256
        self.resume_calls: list[Any] = []

    def batches(self) -> tuple[RequestBatch, ...]:
        return self._batches

    def resume_batches(self, attempt_ledger: Any) -> ResumePlan:
        self.resume_calls.append(attempt_ledger)
        failure_code = getattr(attempt_ledger, "force_failure_code", None)
        if failure_code is not None:
            raise ArtifactContractError(
                "forced resume failure",
                code=failure_code,
            )
        return ResumePlan(
            physical_batch_plan_sha256=self.physical_batch_plan.fingerprint,
            runnable_batches=self._batches,
            held_batch=None,
            completed_physical_batch_sha256s=(),
            blocked_dependencies=(),
            deferred_dependencies=(),
            already_attempted_request_ids=(),
        )


def _batch(
    batch_index: int,
    request_count: int,
    *,
    cumulative: bool = False,
    cell_index: int | None = None,
    execution_wave_partition: str = "independent",
) -> RequestBatch:
    requests = tuple(
        _Request(
            request_id=f"request-{batch_index}-{offset}",
            arm=_Arm(cumulative_dependency=cumulative),
            cell_index=cell_index,
            sampling_seed=1000 + batch_index * 10 + offset,
            schedule_index=batch_index * 4 + offset,
            schedule_identity_sha256="5" * 64,
            execution_identity_sha256=_FAKE_EXECUTION_IDENTITY.fingerprint,
        )
        for offset in range(request_count)
    )
    definition = PhysicalBatchDefinition(
        batch_index=batch_index,
        request_ids=tuple(request.request_id for request in requests),
        execution_wave_partition=execution_wave_partition,
    )
    return RequestBatch(
        batch_index=batch_index,
        requests=requests,  # type: ignore[arg-type]
        physical_batch_sha256=definition.fingerprint,
        physical_batch_plan_sha256="a" * 64,
        execution_wave_partition=execution_wave_partition,
    )


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


def _real_schedule() -> ResearchSchedule:
    records = tuple(
        CohortImageRecord(
            image_id=20_000 + index,
            frozen_order=index,
            source_row_index=index,
            image_path=f"/images/{20_000 + index}.jpg",
            image_sha256=_digest(f"image-{index}"),
            source_width=1024,
            source_height=768,
            raw_width=640,
            raw_height=480,
            source_row_sha256=_digest(f"row-{index}"),
            source_dataset_sha256=_digest("source"),
            raw_annotation_sha256=_digest("annotation"),
            noncrowd_annotated_object_count=12,
            annotated_person_count=8,
            annotated_food_tableware_count=0,
            source_crowd_annotation_count=0,
            cohort_memberships=("runner-test",),
            density_tags=("runner-test",),
        )
        for index in range(4)
    )
    return ResearchSchedule.build_primary(
        unit_id="runner-test-unit",
        run_id="runner-test-run",
        cohort=CohortLedger(
            cohort_id="runner-test",
            full_name="Runner Test Four-Image Cohort",
            operational_meaning="Canonical direct command-line interface fixture.",
            records=records,
        ),
        root_seed=2026071301,
        decode=DecodeProvenance(
            temperature=0.4,
            canonical_generation_policy_sha256=_digest("policy"),
            sampled_runtime_attestation_sha256=_digest("attestation"),
        ),
        execution_identity=ExecutionIdentityBundle(
            code_sha256=_digest("code"),
            config_sha256=_digest("config"),
            ledger_sha256=_digest("ledger"),
            runtime_sha256=_digest("runtime"),
        ),
        grid=GridProvenance(
            canonical_spatial_spec_sha256=_digest("spatial"),
            canonical_spatial_receipt_contract_sha256=_digest("receipt"),
        ),
    )


def _primary_schedule_artifact() -> PrimaryScheduleArtifact:
    return PrimaryScheduleArtifact(
        schedule=_real_schedule(),
        cohort_artifact_name="cohort-manifest.jsonl",
        cohort_artifact_sha256=_digest("cohort-file"),
        readiness_ledger_seal_sha256=_digest("ledger-seal"),
        calibration_selection_receipt_sha256=_digest("calibration-file"),
        calibration_selection_fingerprint=_digest("calibration"),
        source_runtime_identity_receipt_sha256=_digest("runtime-identity-file"),
        source_runtime_identity_fingerprint=_digest("runtime-identity"),
        source_hashes=(("cohort-manifest.jsonl", _digest("cohort-file")),),
    )


def _terminal_attempt_payload(
    context: BatchExecutionContext,
    *,
    request_id: str,
    attempt_status: str = "completed",
) -> dict[str, Any]:
    completed = attempt_status == "completed"
    return AttemptRecord(
        run_id=context.dispatch.run_id,
        schedule_sha256=context.dispatch.schedule_sha256,
        request_id=request_id,
        physical_batch_plan_sha256=(
            context.dispatch.batch.physical_batch_plan_sha256
        ),
        physical_batch_sha256=context.dispatch.batch.physical_batch_sha256,
        physical_batch_index=context.dispatch.batch.batch_index,
        attempt_status=attempt_status,  # type: ignore[arg-type]
        started_at_utc="2026-07-13T00:00:00+00:00",
        finished_at_utc="2026-07-13T00:00:01+00:00",
        execution_identity=_FAKE_EXECUTION_IDENTITY,
        output_artifact_sha256="6" * 64 if completed else None,
        output_artifact_path=f"/artifacts/{request_id}.json" if completed else None,
        failure_code=None if completed else f"research.{attempt_status}",
    ).to_artifact_dict()


class _ProcessFakeExecutor:
    def __init__(
        self,
        *,
        sleep_seconds: float,
        fail: bool = False,
        terminal_statuses: tuple[str, ...] = (),
    ) -> None:
        self._sleep_seconds = sleep_seconds
        self._fail = fail
        self._terminal_statuses = terminal_statuses

    def __call__(self, context: BatchExecutionContext) -> None:
        if self._fail:
            raise RuntimeError("injected process executor failure")
        time.sleep(self._sleep_seconds)
        for request_index, (request_id, journal) in enumerate(
            context.journals.items()
        ):
            journal.record("materialized_input", {"kind": "fake_rgb"})
            journal.record("execution_evidence", {"kind": "fake_execution"})
            journal.record("decode_result", {"kind": "fake_decode"})
            journal.record("parse_score", {"rows": []})
            if context.dispatch.execution_wave_partition != "independent":
                journal.record("cumulative_state", {"accepted_rows": []})
            attempt_status = (
                self._terminal_statuses[request_index]
                if self._terminal_statuses
                else "completed"
            )
            journal.record(
                "terminal_attempt",
                _terminal_attempt_payload(
                    context,
                    request_id=request_id,
                    attempt_status=attempt_status,
                ),
            )


def _process_fake_executor_factory(
    assignment: Any,
    factory_config: Any,
) -> _ProcessFakeExecutor:
    factory_root = Path(factory_config["factory_root"])
    factory_root.mkdir(parents=True, exist_ok=True)
    factory_path = factory_root / f"worker-{assignment.worker_index}-{os.getpid()}.json"
    with factory_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "logical_device": assignment.logical_device,
                    "process_id": os.getpid(),
                },
                sort_keys=True,
            )
        )
    return _ProcessFakeExecutor(
        sleep_seconds=float(factory_config["sleep_seconds"]),
        fail=bool(factory_config.get("fail", False)),
        terminal_statuses=tuple(factory_config.get("terminal_statuses", ())),
    )


def test_plan_preserves_deterministic_b4_b3_batches_without_repacking() -> None:
    batches = (_batch(0, 4), _batch(1, 4), _batch(2, 3))
    schedule = _Schedule(batches)

    first = build_coordinator_plan(  # type: ignore[arg-type]
        schedule, physical_gpu_tokens=("7", "2")
    )
    second = build_coordinator_plan(  # type: ignore[arg-type]
        schedule, physical_gpu_tokens=("7", "2")
    )

    assert [batch.request_ids for batch in first.batches] == [
        batch.request_ids for batch in batches
    ]
    assert [batch.cardinality for batch in first.batches] == [4, 4, 3]
    assert first.to_artifact_dict() == second.to_artifact_dict()
    assert [
        dispatch.worker.physical_gpu_token
        for dispatch in first.waves[0].dispatches
    ] == ["7", "2", "7"]


def test_worker_mapping_is_one_physical_token_to_logical_cuda_zero() -> None:
    workers = build_worker_assignments(("5", "MIG-instance-one"))

    assert [worker.environment for worker in workers] == [
        {"CUDA_VISIBLE_DEVICES": "5"},
        {"CUDA_VISIBLE_DEVICES": "MIG-instance-one"},
    ]
    assert {worker.logical_device for worker in workers} == {"cuda:0"}


@pytest.mark.parametrize("parent_visible_device", [None, "parent-visible-device"])
def test_persistent_pool_binds_device_scope_before_spawn_bootstrap_and_restores_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_visible_device: str | None,
) -> None:
    workers = build_worker_assignments(("physical-zero", "physical-one"))
    bootstrap_root = tmp_path / "bootstrap"
    monkeypatch.setenv(
        _SPAWN_BOOTSTRAP_CAPTURE_ROOT_ENVIRONMENT_VARIABLE,
        str(bootstrap_root),
    )
    if parent_visible_device is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", parent_visible_device)

    with PersistentWorkerPool(
        workers=workers,
        executor_factory=_process_fake_executor_factory,
        factory_config={
            "factory_root": str(tmp_path / "factory"),
            "sleep_seconds": 0.0,
        },
        result_timeout_seconds=60.0,
    ) as pool:
        assert {
            receipt.visible_device_environment for receipt in pool.startup_receipts
        } == {"physical-zero", "physical-one"}

    bootstrap_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(bootstrap_root.glob("bootstrap-*.json"))
    ]
    assert len(bootstrap_payloads) == 2
    assert {
        payload["cuda_visible_devices"] for payload in bootstrap_payloads
    } == {"physical-zero", "physical-one"}
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == parent_visible_device


def test_persistent_pool_unwinds_prior_workers_when_later_spawn_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = build_worker_assignments(("physical-zero", "physical-one"))
    pool = PersistentWorkerPool(
        workers=workers,
        executor_factory=_process_fake_executor_factory,
        factory_config={
            "factory_root": str(tmp_path / "factory"),
            "sleep_seconds": 0.0,
        },
        result_timeout_seconds=60.0,
    )
    real_context = pool._context
    task_queues: list[Any] = []
    failed_processes: list[Any] = []

    class _SynchronousStartFailureProcess:
        pid = None

        def start(self) -> None:
            raise RuntimeError("injected synchronous second-worker start failure")

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def terminate(self) -> None:
            raise AssertionError("an unstarted process must not be terminated")

    def tracking_queue() -> Any:
        task_queue = real_context.Queue()
        task_queues.append(task_queue)
        return task_queue

    def fail_second_process(*args: Any, **kwargs: Any) -> Any:
        if not failed_processes and len(pool._processes) == 1:
            failed_process = _SynchronousStartFailureProcess()
            failed_processes.append(failed_process)
            return failed_process
        return real_context.Process(*args, **kwargs)

    pool._context = SimpleNamespace(  # type: ignore[assignment]
        Process=fail_second_process,
        Queue=tracking_queue,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-visible-device")

    with pytest.raises(
        RuntimeError,
        match="injected synchronous second-worker start failure",
    ):
        pool.start()

    try:
        assert pool._closed
        assert len(pool._processes) == 1
        assert len(failed_processes) == 1
        assert all(not process.is_alive() for process in pool._processes.values())
        assert all(not process.is_alive() for process in failed_processes)
        assert len(task_queues) == 2
        assert all(task_queue._closed for task_queue in task_queues)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-visible-device"
    finally:
        pool.close()
        for task_queue in task_queues:
            if not task_queue._closed:
                task_queue.close()


def test_cumulative_batches_are_separated_by_cell_wave_barriers() -> None:
    batches = (
        _batch(0, 4),
        _batch(
            1,
            4,
            cumulative=True,
            cell_index=0,
            execution_wave_partition="cumulative-cell-00",
        ),
        _batch(
            2,
            4,
            cumulative=True,
            cell_index=0,
            execution_wave_partition="cumulative-cell-00",
        ),
        _batch(
            3,
            4,
            cumulative=True,
            cell_index=1,
            execution_wave_partition="cumulative-cell-01",
        ),
    )
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule(batches), physical_gpu_tokens=("0", "1")
    )

    assert [wave.execution_wave_partition for wave in plan.waves] == [
        "independent",
        "cumulative-cell-00",
        "cumulative-cell-01",
    ]
    assert [[item.batch.batch_index for item in wave.dispatches] for wave in plan.waves] == [
        [0],
        [1, 2],
        [3],
    ]


def test_resume_preserves_sealed_global_worker_and_wave_identity(
    tmp_path: Path,
) -> None:
    batches = (
        _batch(48, 4),
        _batch(
            49,
            4,
            cumulative=True,
            cell_index=0,
            execution_wave_partition="cumulative-cell-00",
        ),
        _batch(
            50,
            4,
            cumulative=True,
            cell_index=1,
            execution_wave_partition="cumulative-cell-01",
        ),
    )
    schedule = _Schedule(batches)
    full_plan = build_coordinator_plan(  # type: ignore[arg-type]
        schedule,
        physical_gpu_tokens=("physical-zero", "physical-one"),
    )
    orphan_batch = batches[1]

    def resume_only_orphan(attempt_ledger: Any) -> ResumePlan:
        schedule.resume_calls.append(attempt_ledger)
        return ResumePlan(
            physical_batch_plan_sha256=schedule.physical_batch_plan.fingerprint,
            runnable_batches=(orphan_batch,),
            held_batch=None,
            completed_physical_batch_sha256s=(batches[0].physical_batch_sha256,),
            blocked_dependencies=(),
            deferred_dependencies=(),
            already_attempted_request_ids=batches[0].request_ids,
        )

    schedule.resume_batches = resume_only_orphan  # type: ignore[method-assign]
    resumed_plan = build_coordinator_plan(  # type: ignore[arg-type]
        schedule,
        physical_gpu_tokens=("physical-zero", "physical-one"),
    )

    original = full_plan.waves[1].dispatches[0]
    resumed = resumed_plan.waves[0].dispatches[0]
    assert original.batch.batch_index == resumed.batch.batch_index == 49
    assert original.wave_index == resumed.wave_index == 1
    assert original.worker == resumed.worker
    assert resumed.worker.worker_index == 1
    assert resumed.worker.environment == {"CUDA_VISIBLE_DEVICES": "physical-one"}
    original_journal = RequestArtifactJournal(
        root=tmp_path,
        dispatch=original,
        request_id=orphan_batch.request_ids[0],
    )
    intent_path = original_journal.write_call_intent()
    original_intent_bytes = intent_path.read_bytes()

    resumed_journal = RequestArtifactJournal(
        root=tmp_path,
        dispatch=resumed,
        request_id=orphan_batch.request_ids[0],
    )
    assert resumed_journal.write_call_intent() == intent_path
    assert intent_path.read_bytes() == original_intent_bytes


@pytest.mark.parametrize(
    "failure_code",
    [
        "analysis.resume_partial_physical_batch",
        "analysis.resume_requires_continuation_plan.failed",
        "analysis.resume_requires_continuation_plan.capped",
        "analysis.resume_requires_continuation_plan.invalid",
        "analysis.resume_requires_continuation_plan.skipped",
        "analysis.resume_requires_continuation_plan.missing_state",
    ],
)
def test_runner_delegates_resume_failure_to_schedule_authority(
    failure_code: str,
) -> None:
    schedule = _Schedule((_batch(0, 4),))
    attempt_ledger = SimpleNamespace(force_failure_code=failure_code)
    with pytest.raises(ArtifactContractError, match="forced resume failure"):
        build_coordinator_plan(  # type: ignore[arg-type]
            schedule,
            physical_gpu_tokens=("0",),
            attempt_ledger=attempt_ledger,  # type: ignore[arg-type]
        )
    assert schedule.resume_calls == [attempt_ledger]


def test_call_intents_precede_execution_and_terminal_attempts_are_last(
    tmp_path: Path,
) -> None:
    batch = _batch(0, 4)
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule((batch,)), physical_gpu_tokens=("3",)
    )
    observed: list[tuple[str, str]] = []

    def executor(context: BatchExecutionContext) -> None:
        for request_id, journal in context.journals.items():
            assert (journal.request_directory / "00-call_intent.json").is_file()
            observed.append((request_id, "execution_started"))
            journal.record("materialized_input", {"kind": "fake_rgb"})
            journal.record("execution_evidence", {"kind": "fake_execution"})
            journal.record("decode_result", {"kind": "fake_decode"})
            journal.record("parse_score", {"rows": []})
            journal.record(
                "terminal_attempt",
                _terminal_attempt_payload(context, request_id=request_id),
            )
            with pytest.raises(RunnerContractError, match="follow the terminal"):
                journal.record("cumulative_state", {"rows": []})

    execute_coordinator_plan(plan, artifact_root=tmp_path, executor=executor)

    assert len(observed) == 4
    for request_id in batch.request_ids:
        request_dir = next(
            journal_dir
            for journal_dir in (tmp_path / "calls").iterdir()
            if request_id in (journal_dir / "06-terminal_attempt.json").read_text()
        )
        assert [path.name for path in sorted(request_dir.iterdir())] == [
            "00-call_intent.json",
            "01-materialized_input.json",
            "02-execution_evidence.json",
            "03-decode_result.json",
            "04-parse_score.json",
            "06-terminal_attempt.json",
        ]


def test_cumulative_state_is_fsynced_before_terminal_and_wave_callback(
    tmp_path: Path,
) -> None:
    batches = (
        _batch(
            0,
            4,
            cumulative=True,
            cell_index=0,
            execution_wave_partition="cumulative-cell-00",
        ),
        _batch(
            1,
            4,
            cumulative=True,
            cell_index=1,
            execution_wave_partition="cumulative-cell-01",
        ),
    )
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule(batches), physical_gpu_tokens=("0",)
    )
    events: list[str] = []

    def executor(context: BatchExecutionContext) -> None:
        events.append(f"execute-{context.dispatch.execution_wave_partition}")
        for request_id, journal in context.journals.items():
            journal.record("materialized_input", {})
            journal.record("execution_evidence", {})
            journal.record("decode_result", {})
            journal.record("parse_score", {})
            state_path = journal.record("cumulative_state", {"accepted_rows": []})
            assert state_path.is_file()
            journal.record(
                "terminal_attempt",
                _terminal_attempt_payload(context, request_id=request_id),
            )

    execute_coordinator_plan(
        plan,
        artifact_root=tmp_path,
        executor=executor,
        after_wave=lambda wave: events.append(
            f"barrier-{wave.execution_wave_partition}"
        ),
    )

    assert events == [
        "execute-cumulative-cell-00",
        "barrier-cumulative-cell-00",
        "execute-cumulative-cell-01",
        "barrier-cumulative-cell-01",
    ]


def test_orphan_call_intent_has_no_terminal_attempt_claim(tmp_path: Path) -> None:
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule((_batch(0, 4),)), physical_gpu_tokens=("0",)
    )
    dispatch = plan.waves[0].dispatches[0]
    journal = RequestArtifactJournal(
        root=tmp_path,
        dispatch=dispatch,
        request_id=dispatch.batch.request_ids[0],
    )
    journal.write_call_intent()

    assert journal.terminal_written is False
    assert not (journal.request_directory / "06-terminal_attempt.json").exists()


def test_orphan_call_intent_allows_exact_same_run_sealed_batch_replay(
    tmp_path: Path,
) -> None:
    batch = _batch(0, 4)
    schedule = _Schedule((batch,))
    initial_plan = build_coordinator_plan(  # type: ignore[arg-type]
        schedule, physical_gpu_tokens=("0",)
    )
    dispatch = initial_plan.waves[0].dispatches[0]
    orphan = RequestArtifactJournal(
        root=tmp_path,
        dispatch=dispatch,
        request_id=batch.request_ids[0],
    )
    orphan_intent_path = orphan.write_call_intent()
    original_intent_bytes = orphan_intent_path.read_bytes()
    resumed_plan = build_coordinator_plan(  # type: ignore[arg-type]
        schedule, physical_gpu_tokens=("0",)
    )
    assert resumed_plan.batches[0].physical_batch_sha256 == (
        initial_plan.batches[0].physical_batch_sha256
    )

    def executor(context: BatchExecutionContext) -> None:
        for request_id, journal in context.journals.items():
            journal.record("materialized_input", {})
            journal.record("execution_evidence", {})
            journal.record("decode_result", {})
            journal.record("parse_score", {})
            journal.record(
                "terminal_attempt",
                _terminal_attempt_payload(context, request_id=request_id),
            )

    execute_coordinator_plan(resumed_plan, artifact_root=tmp_path, executor=executor)

    assert orphan_intent_path.read_bytes() == original_intent_bytes
    assert len(list((tmp_path / "calls").glob("*/06-terminal_attempt.json"))) == 4


@pytest.mark.parametrize("failure_mode", ["drifted_intent", "partial_journal"])
def test_call_intent_retry_fails_closed_on_drift_or_partial_progress(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule((_batch(0, 4),)), physical_gpu_tokens=("0",)
    )
    dispatch = plan.waves[0].dispatches[0]
    journal = RequestArtifactJournal(
        root=tmp_path,
        dispatch=dispatch,
        request_id=dispatch.batch.request_ids[0],
    )
    intent_path = journal.write_call_intent()
    if failure_mode == "drifted_intent":
        intent_path.write_bytes(intent_path.read_bytes() + b" ")
        expected_message = "differs from the canonical"
    else:
        resumed = RequestArtifactJournal.open_after_call_intent(
            root=tmp_path,
            dispatch=dispatch,
            request_id=dispatch.batch.request_ids[0],
        )
        resumed.record("materialized_input", {})
        expected_message = "partially advanced"

    with pytest.raises(RunnerContractError, match=expected_message):
        execute_coordinator_plan(
            plan,
            artifact_root=tmp_path,
            executor=lambda context: None,
        )


def test_cumulative_prompt_uses_canonical_assistant_continuation(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_build_prompt_record(*args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "prompt-record"

    monkeypatch.setattr(
        "src.analysis.spatial_scope_history.runner.build_prompt_record",
        fake_build_prompt_record,
    )
    result = build_cumulative_prompt_record(
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        processor=object(),
        row_index=7,
        accepted_global_coordinate_rows="<row-one><row-two>",
        max_prompt_tokens=2048,
    )

    assert result == "prompt-record"
    assert captured["assistant_continuation"].text == "<row-one><row-two>"
    assert captured["row_index"] == 7
    assert captured["max_prompt_tokens"] == 2048


class _RecordingTaskQueue:
    def __init__(self, submissions: list[dict[str, Any]]) -> None:
        self._submissions = submissions

    def put(self, payload: dict[str, Any]) -> None:
        assert payload["kind"] == "execute"
        self._submissions.append(payload)


def _intent_batch_indexes(artifact_root: Path) -> set[int]:
    return {
        int(json.loads(path.read_text(encoding="utf-8"))["payload"]["batch_index"])
        for path in (artifact_root / "calls").glob("*/00-call_intent.json")
    }


def _completed_batch_message(submission: dict[str, Any]) -> dict[str, Any]:
    dispatch = submission["dispatch"]
    return {
        "kind": "batch_completed",
        "receipt": {
            "execution_wave_partition": dispatch.execution_wave_partition,
            "finished_at_unix_nanoseconds": 2,
            "physical_batch_sha256": dispatch.batch.physical_batch_sha256,
            "process_id": 10_000 + dispatch.worker.worker_index,
            "request_terminal_receipts": (),
            "started_at_unix_nanoseconds": 1,
            "task_id": submission["task_id"],
            "worker_index": dispatch.worker.worker_index,
        },
    }


def _coordinator_only_pool(
    plan: Any,
    submissions: list[dict[str, Any]],
) -> PersistentWorkerPool:
    pool = object.__new__(PersistentWorkerPool)
    pool._workers = plan.workers
    pool._started = True
    pool._closed = False
    pool._startup_receipts = ()
    pool._task_queues = {
        worker.worker_index: _RecordingTaskQueue(submissions) for worker in plan.workers
    }
    return pool


def test_persistent_pool_refills_only_completed_worker_with_bounded_intents(
    tmp_path: Path,
) -> None:
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule(tuple(_batch(index, 4) for index in range(4))),
        physical_gpu_tokens=("physical-zero", "physical-one"),
    )
    artifact_root = tmp_path / "artifacts"
    submissions: list[dict[str, Any]] = []
    pool = _coordinator_only_pool(plan, submissions)
    completion_order = (1, 0, 3, 2)
    message_index = 0

    def next_message() -> dict[str, Any]:
        nonlocal message_index
        expected_submission_order = {
            0: [0, 1],
            1: [0, 1, 3],
            2: [0, 1, 3, 2],
            3: [0, 1, 3, 2],
        }[message_index]
        assert [
            item["dispatch"].batch.batch_index for item in submissions
        ] == expected_submission_order
        assert _intent_batch_indexes(artifact_root) == set(expected_submission_order)
        batch_index = completion_order[message_index]
        message_index += 1
        submission = next(
            item
            for item in submissions
            if item["dispatch"].batch.batch_index == batch_index
        )
        return _completed_batch_message(submission)

    pool._next_message = next_message  # type: ignore[method-assign]

    receipt = pool.execute_plan(plan, artifact_root=artifact_root)

    assert [item["dispatch"].batch.batch_index for item in submissions] == [0, 1, 3, 2]
    assert [item.worker_index for item in receipt.batch_receipts] == [1, 0, 1, 0]
    assert [item.physical_batch_sha256 for item in receipt.batch_receipts] == [
        plan.waves[0].dispatches[index].batch.physical_batch_sha256
        for index in completion_order
    ]


def test_persistent_process_pool_overlaps_within_wave_and_barriers_between_waves(
    tmp_path: Path,
) -> None:
    batches = (
        _batch(0, 4),
        _batch(1, 4),
        _batch(
            2,
            4,
            cumulative=True,
            cell_index=0,
            execution_wave_partition="cumulative-cell-00",
        ),
        _batch(
            3,
            4,
            cumulative=True,
            cell_index=0,
            execution_wave_partition="cumulative-cell-00",
        ),
    )
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule(batches), physical_gpu_tokens=("physical-zero", "physical-one")
    )
    factory_root = tmp_path / "factory"
    with PersistentWorkerPool(
        workers=plan.workers,
        executor_factory=_process_fake_executor_factory,
        factory_config={
            "factory_root": str(factory_root),
            "sleep_seconds": 0.2,
        },
        result_timeout_seconds=60.0,
    ) as pool:
        receipt = pool.execute_plan(plan, artifact_root=tmp_path / "artifacts")

    assert len(receipt.worker_startups) == 2
    assert len({item.process_id for item in receipt.worker_startups}) == 2
    assert {
        (item.physical_gpu_token, item.visible_device_environment, item.logical_device)
        for item in receipt.worker_startups
    } == {
        ("physical-zero", "physical-zero", "cuda:0"),
        ("physical-one", "physical-one", "cuda:0"),
    }
    assert len(list(factory_root.glob("worker-*.json"))) == 2
    receipts_by_wave = {
        partition: [
            item
            for item in receipt.batch_receipts
            if item.execution_wave_partition == partition
        ]
        for partition in ("independent", "cumulative-cell-00")
    }
    for wave_receipts in receipts_by_wave.values():
        assert len(wave_receipts) == 2
        assert max(
            item.started_at_unix_nanoseconds for item in wave_receipts
        ) < min(item.finished_at_unix_nanoseconds for item in wave_receipts)
    assert max(
        item.finished_at_unix_nanoseconds
        for item in receipts_by_wave["independent"]
    ) <= min(
        item.started_at_unix_nanoseconds
        for item in receipts_by_wave["cumulative-cell-00"]
    )
    startup_pid_by_worker = {
        item.worker_index: item.process_id for item in receipt.worker_startups
    }
    assert all(
        item.process_id == startup_pid_by_worker[item.worker_index]
        for item in receipt.batch_receipts
    )
    assert {
        terminal.attempt_status
        for batch_receipt in receipt.batch_receipts
        for terminal in batch_receipt.request_terminal_receipts
    } == {"completed"}
    terminal_artifacts = list(
        (tmp_path / "artifacts" / "calls").glob("*/06-terminal_attempt.json")
    )
    assert len(terminal_artifacts) == 16


def test_persistent_process_pool_propagates_executor_failure_fail_closed(
    tmp_path: Path,
) -> None:
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule(tuple(_batch(index, 4) for index in range(4))),
        physical_gpu_tokens=("physical-zero", "physical-one"),
    )
    artifact_root = tmp_path / "artifacts"
    pool = PersistentWorkerPool(
        workers=plan.workers,
        executor_factory=_process_fake_executor_factory,
        factory_config={
            "factory_root": str(tmp_path / "factory"),
            "sleep_seconds": 0.0,
            "fail": True,
        },
        result_timeout_seconds=60.0,
    )
    with pytest.raises(RunnerContractError, match="failed closed"):
        with pool:
            pool.execute_plan(plan, artifact_root=artifact_root)
    assert all(not process.is_alive() for process in pool._processes.values())
    assert _intent_batch_indexes(artifact_root) == {0, 1}


def test_persistent_process_pool_propagates_legal_scientific_terminal_states(
    tmp_path: Path,
) -> None:
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule((_batch(0, 4),)), physical_gpu_tokens=("physical-zero",)
    )
    terminal_statuses = ("failed", "skipped", "capped", "invalid")
    with PersistentWorkerPool(
        workers=plan.workers,
        executor_factory=_process_fake_executor_factory,
        factory_config={
            "factory_root": str(tmp_path / "factory"),
            "sleep_seconds": 0.0,
            "terminal_statuses": terminal_statuses,
        },
        result_timeout_seconds=60.0,
    ) as pool:
        receipt = pool.execute_plan(plan, artifact_root=tmp_path / "artifacts")

    assert len(receipt.batch_receipts) == 1
    assert tuple(
        terminal.attempt_status
        for terminal in receipt.batch_receipts[0].request_terminal_receipts
    ) == terminal_statuses
    assert len(
        list((tmp_path / "artifacts" / "calls").glob("*/06-terminal_attempt.json"))
    ) == 4


def _journal_ready_for_terminal(
    tmp_path: Path,
) -> tuple[BatchExecutionContext, str, RequestArtifactJournal]:
    plan = build_coordinator_plan(  # type: ignore[arg-type]
        _Schedule((_batch(0, 4),)), physical_gpu_tokens=("physical-zero",)
    )
    dispatch = plan.waves[0].dispatches[0]
    request_id = dispatch.batch.request_ids[0]
    journal = RequestArtifactJournal(
        root=tmp_path,
        dispatch=dispatch,
        request_id=request_id,
    )
    journal.write_call_intent()
    journal.record("materialized_input", {})
    journal.record("execution_evidence", {})
    journal.record("decode_result", {})
    journal.record("parse_score", {})
    return BatchExecutionContext(dispatch=dispatch, journals={}), request_id, journal


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("missing_failure_code", "requires a failure code"),
        ("completed_without_output", "requires an output artifact digest"),
        ("unknown_status", "unknown terminal status"),
        ("unknown_key", "keys do not match"),
    ],
)
def test_request_journal_rejects_noncanonical_terminal_attempt_payload(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    context, request_id, journal = _journal_ready_for_terminal(tmp_path)
    payload = _terminal_attempt_payload(context, request_id=request_id)
    if mutation == "missing_failure_code":
        payload = _terminal_attempt_payload(
            context, request_id=request_id, attempt_status="failed"
        )
        payload["failure_code"] = None
    elif mutation == "completed_without_output":
        payload["output_artifact_sha256"] = None
    elif mutation == "unknown_status":
        payload["attempt_status"] = "unknown"
    else:
        payload["unknown_key"] = True

    with pytest.raises(ArtifactContractError, match=expected_message):
        journal.record("terminal_attempt", payload)
    assert journal.terminal_written is False
    assert not (journal.request_directory / "06-terminal_attempt.json").exists()


@pytest.mark.parametrize(
    "identity_field",
    [
        "run_id",
        "schedule_sha256",
        "request_id",
        "physical_batch_plan_sha256",
        "physical_batch_sha256",
        "physical_batch_index",
        "execution_identity",
    ],
)
def test_request_journal_rejects_terminal_attempt_identity_drift(
    tmp_path: Path,
    identity_field: str,
) -> None:
    context, request_id, journal = _journal_ready_for_terminal(tmp_path)
    payload = _terminal_attempt_payload(context, request_id=request_id)
    if identity_field == "physical_batch_index":
        payload[identity_field] += 1
    elif identity_field == "execution_identity":
        payload[identity_field] = {
            **payload[identity_field],
            "code_sha256": "7" * 64,
        }
    else:
        payload[identity_field] = (
            "wrong-run" if identity_field in {"run_id", "request_id"} else "7" * 64
        )

    with pytest.raises(RunnerContractError, match="differs from its sealed request"):
        journal.record("terminal_attempt", payload)
    assert journal.terminal_written is False
    assert not (journal.request_directory / "06-terminal_attempt.json").exists()


def test_direct_cli_help_missing_inputs_and_dry_run_plan_are_canonical(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "scripts/research/run_spatial_scope_history.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "Spatial Scope and History Disentanglement" in help_result.stdout
    assert "--physical-gpu-token" in help_result.stdout

    negative_result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert negative_result.returncode == 2
    assert "the following arguments are required" in negative_result.stderr

    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps(_primary_schedule_artifact().to_artifact_dict(), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    output_plan_path = tmp_path / "coordinator-plan.json"
    positive_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--schedule",
            str(schedule_path),
            "--physical-gpu-token",
            "physical-zero",
            "--physical-gpu-token",
            "physical-one",
            "--output-plan",
            str(output_plan_path),
            "--dry-run",
        ],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert positive_result.returncode == 0, positive_result.stderr
    output_plan = json.loads(output_plan_path.read_text(encoding="utf-8"))
    assert output_plan["schema_version"] == "spatial_scope_history.runner_plan.v1"
    assert [wave["execution_wave_partition"] for wave in output_plan["waves"]] == [
        "independent",
        "cumulative-cell-00",
    ]
    assert [
        sum(dispatch["cardinality"] for dispatch in wave["dispatches"])
        for wave in output_plan["waves"]
    ] == [196, 4]
