from __future__ import annotations

from dataclasses import replace
import json

import pytest

from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptDependencyContract,
    AttemptLedger,
    AttemptRecord,
    CohortImageRecord,
    CohortLedger,
    ExecutionIdentityBundle,
    append_attempt_record,
    canonical_json_text,
    dependency_skip_failure_code,
    sha256_file,
    sha256_payload,
)
from src.common.errors import ArtifactContractError, DataContractError


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


def _image(order: int) -> CohortImageRecord:
    return CohortImageRecord(
        image_id=1000 + order,
        frozen_order=order,
        source_row_index=order,
        image_path=f"/images/{1000 + order}.jpg",
        image_sha256=_digest(f"image-{order}"),
        source_width=1024,
        source_height=768,
        raw_width=640,
        raw_height=480,
        source_row_sha256=_digest(f"row-{order}"),
        source_dataset_sha256=_digest("dataset"),
        raw_annotation_sha256=_digest("annotations"),
        noncrowd_annotated_object_count=12,
        annotated_person_count=8,
        annotated_food_tableware_count=0,
        source_crowd_annotation_count=0,
        cohort_memberships=("validation-200",),
        density_tags=("annotated-count-density", "people-density"),
    )


def _cohort(count: int = 4) -> CohortLedger:
    return CohortLedger(
        cohort_id="validation-200",
        full_name="Validation-200",
        operational_meaning="The fixed first 200 validation images ordered by image identifier.",
        records=tuple(_image(index) for index in range(count)),
    )


def _identity() -> ExecutionIdentityBundle:
    return ExecutionIdentityBundle(
        code_sha256=_digest("code"),
        config_sha256=_digest("config"),
        ledger_sha256=_digest("ledger"),
        runtime_sha256=_digest("runtime"),
    )


def _attempt(
    *,
    request_id: str,
    status: str = "completed",
    physical_batch_index: int = 0,
) -> AttemptRecord:
    return AttemptRecord(
        run_id="run-primary",
        schedule_sha256=_digest("schedule"),
        request_id=request_id,
        physical_batch_plan_sha256=_digest("physical-batch-plan"),
        physical_batch_sha256=_digest(f"physical-batch-{physical_batch_index}"),
        physical_batch_index=physical_batch_index,
        attempt_status=status,
        started_at_utc="2026-07-13T00:00:00Z",
        finished_at_utc="2026-07-13T00:01:00Z",
        execution_identity=_identity(),
        output_artifact_sha256=_digest(f"output-{request_id}")
        if status == "completed"
        else None,
        failure_code=None if status == "completed" else f"research.{status}",
    )


def _dependency(
    *,
    request_id: str,
    predecessor_request_id: str | None = None,
    initial_cumulative_state_sha256: str | None = None,
    physical_batch_index: int = 0,
) -> AttemptDependencyContract:
    return AttemptDependencyContract(
        request_id=request_id,
        physical_batch_plan_sha256=_digest("physical-batch-plan"),
        physical_batch_sha256=_digest(f"physical-batch-{physical_batch_index}"),
        physical_batch_index=physical_batch_index,
        predecessor_request_id=predecessor_request_id,
        initial_cumulative_state_sha256=initial_cumulative_state_sha256,
    )


def _state_artifact(tmp_path, label: str) -> tuple[str, str]:
    path = tmp_path / f"{label}.json"
    path.write_text(f'{{"label":"{label}"}}\n', encoding="utf-8")
    return str(path), sha256_file(path)


def test_cohort_jsonl_round_trip_is_byte_identical_and_content_addressed() -> None:
    cohort = _cohort()
    payload = cohort.to_jsonl_bytes()

    restored = CohortLedger.from_jsonl_bytes(payload)

    assert restored == cohort
    assert restored.to_jsonl_bytes() == payload
    assert restored.fingerprint == cohort.fingerprint
    assert payload.endswith(b"\n")


def test_cohort_rejects_order_drift_duplicate_images_and_unknown_keys() -> None:
    with pytest.raises(DataContractError, match="contiguous frozen order"):
        CohortLedger(
            cohort_id="x",
            full_name="Example Cohort",
            operational_meaning="Synthetic validation fixture.",
            records=(_image(1),),
        )
    duplicate = replace(_image(1), frozen_order=1, image_id=_image(0).image_id)
    with pytest.raises(DataContractError, match="duplicate image"):
        CohortLedger(
            cohort_id="x",
            full_name="Example Cohort",
            operational_meaning="Synthetic validation fixture.",
            records=(_image(0), duplicate),
        )
    record = _image(0).to_artifact_dict()
    record["unexpected"] = True
    with pytest.raises(ArtifactContractError, match="unknown_keys"):
        CohortImageRecord.from_artifact_dict(record)


def test_canonical_json_rejects_nonfinite_and_non_string_keys() -> None:
    with pytest.raises(ArtifactContractError, match="non-finite"):
        canonical_json_text({"value": float("nan")})
    with pytest.raises(ArtifactContractError, match="keys must be strings"):
        canonical_json_text({1: "value"})


def test_attempt_ledger_is_append_only_and_rejects_duplicate_or_drift(tmp_path) -> None:
    path = tmp_path / "attempts.jsonl"
    first = _attempt(request_id="request-1")
    dependencies = tuple(
        _dependency(request_id=request_id) for request_id in ("request-1", "request-2")
    )
    updated = append_attempt_record(
        path,
        record=first,
        dependencies=dependencies,
    )
    assert updated.attempted_request_ids == frozenset({"request-1"})
    before = path.read_bytes()

    with pytest.raises(ArtifactContractError, match="repeats a request"):
        append_attempt_record(
            path,
            record=first,
            dependencies=dependencies,
        )
    assert path.read_bytes() == before

    drift = replace(_attempt(request_id="request-2"), schedule_sha256=_digest("other"))
    with pytest.raises(ArtifactContractError, match="different schedule identity"):
        append_attempt_record(
            path,
            record=drift,
            dependencies=dependencies,
        )
    assert path.read_bytes() == before


def test_attempt_ledger_rejects_unknown_request_and_noncanonical_jsonl() -> None:
    attempt = _attempt(request_id="unknown")
    ledger = AttemptLedger(
        run_id=attempt.run_id,
        schedule_sha256=attempt.schedule_sha256,
        execution_identity=attempt.execution_identity,
        records=(attempt,),
    )
    with pytest.raises(ArtifactContractError, match="absent from its schedule"):
        ledger.validate_request_universe(("request-1",))

    noncanonical = (
        json.dumps(attempt.to_artifact_dict(), sort_keys=False) + "\n"
    ).encode()
    with pytest.raises(ArtifactContractError, match="canonical JSON"):
        AttemptLedger.from_jsonl_bytes(
            noncanonical,
            run_id=attempt.run_id,
            schedule_sha256=attempt.schedule_sha256,
            execution_identity=attempt.execution_identity,
        )


def test_attempt_status_requires_terminal_evidence() -> None:
    with pytest.raises(ArtifactContractError, match="output artifact digest"):
        replace(_attempt(request_id="request-1"), output_artifact_sha256=None)
    with pytest.raises(ArtifactContractError, match="failure code"):
        replace(
            _attempt(request_id="request-2", status="failed"),
            failure_code=None,
        )


def test_attempt_rejects_physical_batch_identity_drift() -> None:
    dependency = _dependency(request_id="request-1")
    attempt = replace(
        _attempt(request_id="request-1"),
        physical_batch_sha256=_digest("wrong-physical-batch"),
    )
    ledger = AttemptLedger(
        run_id=attempt.run_id,
        schedule_sha256=attempt.schedule_sha256,
        execution_identity=attempt.execution_identity,
        records=(attempt,),
    )

    with pytest.raises(ArtifactContractError, match="sealed schedule"):
        ledger.validate_dependencies((dependency,))


def test_cumulative_attempts_require_ordered_reconstructible_state_chain(
    tmp_path,
) -> None:
    path = tmp_path / "attempts.jsonl"
    initial_state_sha256 = _digest("empty-accepted-row-prefix")
    dependencies = (
        _dependency(
            request_id="cell-00",
            initial_cumulative_state_sha256=initial_state_sha256,
        ),
        _dependency(
            request_id="cell-01",
            predecessor_request_id="cell-00",
        ),
    )
    cell_zero_path, cell_zero_sha256 = _state_artifact(tmp_path, "cell-00-state")
    cell_one_path, cell_one_sha256 = _state_artifact(tmp_path, "cell-01-state")
    cell_zero = replace(
        _attempt(request_id="cell-00"),
        expected_cumulative_state_sha256=initial_state_sha256,
        produced_cumulative_state_sha256=cell_zero_sha256,
        produced_cumulative_state_artifact_path=cell_zero_path,
    )
    cell_one = replace(
        _attempt(request_id="cell-01"),
        expected_cumulative_state_sha256=cell_zero_sha256,
        produced_cumulative_state_sha256=cell_one_sha256,
        produced_cumulative_state_artifact_path=cell_one_path,
    )

    with pytest.raises(ArtifactContractError, match="before its predecessor"):
        append_attempt_record(path, record=cell_one, dependencies=dependencies)
    assert not path.read_bytes()

    append_attempt_record(path, record=cell_zero, dependencies=dependencies)
    with pytest.raises(ArtifactContractError, match="expected-state fingerprint"):
        append_attempt_record(
            path,
            record=replace(
                cell_one,
                expected_cumulative_state_sha256=_digest("wrong-state"),
            ),
            dependencies=dependencies,
        )
    updated = append_attempt_record(path, record=cell_one, dependencies=dependencies)
    assert updated.successfully_completed_request_ids == frozenset(
        {"cell-00", "cell-01"}
    )


def test_failed_cumulative_attempt_requires_terminal_skip_propagation() -> None:
    initial_state_sha256 = _digest("empty-accepted-row-prefix")
    dependencies = (
        _dependency(
            request_id="cell-00",
            initial_cumulative_state_sha256=initial_state_sha256,
        ),
        _dependency(
            request_id="cell-01",
            predecessor_request_id="cell-00",
        ),
        _dependency(
            request_id="cell-02",
            predecessor_request_id="cell-01",
        ),
    )
    failed = replace(
        _attempt(request_id="cell-00", status="failed"),
        expected_cumulative_state_sha256=initial_state_sha256,
    )
    invalid_child = _attempt(request_id="cell-01")
    with pytest.raises(ArtifactContractError, match="terminally skipped"):
        AttemptLedger(
            run_id=failed.run_id,
            schedule_sha256=failed.schedule_sha256,
            execution_identity=failed.execution_identity,
            records=(failed, invalid_child),
        ).validate_dependencies(dependencies)

    cell_one_skip = replace(
        _attempt(request_id="cell-01", status="skipped"),
        failure_code=dependency_skip_failure_code(
            predecessor_request_id="cell-00",
            predecessor_status="failed",
        ),
    )
    cell_two_skip = replace(
        _attempt(request_id="cell-02", status="skipped"),
        failure_code=dependency_skip_failure_code(
            predecessor_request_id="cell-01",
            predecessor_status="skipped",
        ),
    )
    ledger = AttemptLedger(
        run_id=failed.run_id,
        schedule_sha256=failed.schedule_sha256,
        execution_identity=failed.execution_identity,
        records=(failed, cell_one_skip, cell_two_skip),
    )
    ledger.validate_dependencies(dependencies)
    assert ledger.attempted_request_ids == frozenset({"cell-00", "cell-01", "cell-02"})
    assert ledger.successfully_completed_request_ids == frozenset()


def test_replayed_cumulative_record_is_rejected_without_mutating_ledger(
    tmp_path,
) -> None:
    path = tmp_path / "attempts.jsonl"
    initial_state_sha256 = _digest("empty-accepted-row-prefix")
    state_path, state_sha256 = _state_artifact(tmp_path, "cell-00-state")
    dependency = _dependency(
        request_id="cell-00",
        initial_cumulative_state_sha256=initial_state_sha256,
    )
    record = replace(
        _attempt(request_id="cell-00"),
        expected_cumulative_state_sha256=initial_state_sha256,
        produced_cumulative_state_sha256=state_sha256,
        produced_cumulative_state_artifact_path=state_path,
    )
    append_attempt_record(path, record=record, dependencies=(dependency,))
    before = path.read_bytes()

    with pytest.raises(ArtifactContractError, match="repeats a request"):
        append_attempt_record(path, record=record, dependencies=(dependency,))
    assert path.read_bytes() == before
