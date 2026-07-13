from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptRecord,
    CohortImageRecord,
    CohortLedger,
    ExecutionIdentityBundle,
    dependency_skip_failure_code,
    dependency_state_unavailable_failure_code,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256,
    GridProvenance,
    PRIMARY_ARM_DEFINITIONS,
    PhysicalBatchDefinition,
    PhysicalBatchPlan,
    ResearchSchedule,
    ScheduledRequest,
    derive_sampling_seed,
)
from src.common.errors import ArtifactContractError, DataContractError


UNIT_ID = "2026-07-13-spatial-scope-history-disentanglement"


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


def _image(order: int) -> CohortImageRecord:
    return CohortImageRecord(
        image_id=10_000 + order,
        frozen_order=order,
        source_row_index=order,
        image_path=f"/images/{10_000 + order}.jpg",
        image_sha256=_digest(f"image-{order}"),
        source_width=1024,
        source_height=768,
        raw_width=640,
        raw_height=480,
        source_row_sha256=_digest(f"row-{order}"),
        source_dataset_sha256=_digest("source"),
        raw_annotation_sha256=_digest("annotations"),
        noncrowd_annotated_object_count=12,
        annotated_person_count=8,
        annotated_food_tableware_count=0,
        source_crowd_annotation_count=0,
        cohort_memberships=("dense-union-51",),
        density_tags=("annotated-count-density",),
    )


def _cohort(count: int) -> CohortLedger:
    return CohortLedger(
        cohort_id=f"synthetic-{count}",
        full_name=f"Synthetic {count}-Image Cohort",
        operational_meaning="Synthetic schedule contract fixture.",
        records=tuple(_image(index) for index in range(count)),
    )


def _execution_identity() -> ExecutionIdentityBundle:
    return ExecutionIdentityBundle(
        code_sha256=_digest("code"),
        config_sha256=_digest("config"),
        ledger_sha256=_digest("ledger"),
        runtime_sha256=_digest("runtime"),
    )


def _schedule(count: int, *, run_id: str = "run-primary") -> ResearchSchedule:
    return ResearchSchedule.build_primary(
        unit_id=UNIT_ID,
        run_id=run_id,
        cohort=_cohort(count),
        root_seed=2026071301,
        decode=DecodeProvenance(
            temperature=0.4,
            canonical_generation_policy_sha256=_digest("generation-policy"),
            sampled_runtime_attestation_sha256=_digest("sampling-attestation"),
        ),
        execution_identity=_execution_identity(),
        grid=GridProvenance(
            canonical_spatial_spec_sha256=_digest("spatial-spec"),
            canonical_spatial_receipt_contract_sha256=_digest(
                "spatial-receipt-contract"
            ),
        ),
    )


def _attempt(
    schedule: ResearchSchedule,
    request_id: str,
    *,
    status: str = "completed",
    failure_code: str | None = None,
    expected_state_sha256: str | None = None,
    produced_state: tuple[str, str] | None = None,
) -> AttemptRecord:
    dependency = next(
        dependency
        for dependency in schedule.attempt_dependencies
        if dependency.request_id == request_id
    )
    return AttemptRecord(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        request_id=request_id,
        physical_batch_plan_sha256=dependency.physical_batch_plan_sha256,
        physical_batch_sha256=dependency.physical_batch_sha256,
        physical_batch_index=dependency.physical_batch_index,
        attempt_status=status,
        started_at_utc="2026-07-13T00:00:00Z",
        finished_at_utc="2026-07-13T00:01:00Z",
        execution_identity=schedule.identity.execution_identity,
        output_artifact_sha256=_digest(request_id) if status == "completed" else None,
        failure_code=failure_code if status != "completed" else None,
        expected_cumulative_state_sha256=expected_state_sha256,
        produced_cumulative_state_sha256=(
            None if produced_state is None else produced_state[1]
        ),
        produced_cumulative_state_artifact_path=(
            None if produced_state is None else produced_state[0]
        ),
    )


def _state_artifact(tmp_path, label: str) -> tuple[str, str]:
    path = tmp_path / f"{label}.json"
    path.write_text(f'{{"label":"{label}"}}\n', encoding="utf-8")
    return str(path), sha256_file(path)


def _cumulative_requests(
    schedule: ResearchSchedule, *, image_frozen_order: int
) -> tuple[ScheduledRequest, ...]:
    return tuple(
        request
        for request in schedule.requests
        if request.arm.arm_code == "MASK_CUMULATIVE"
        and request.image_frozen_order == image_frozen_order
    )


def _ledger(
    schedule: ResearchSchedule, records: tuple[AttemptRecord, ...]
) -> AttemptLedger:
    return AttemptLedger(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
        records=records,
    )


def test_primary_schedule_has_exact_five_arm_counts_and_canonical_order() -> None:
    schedule = _schedule(4)

    assert len(schedule.requests) == 4 * 65
    assert schedule.physical_batch_plan.fingerprint
    assert all(
        batch.physical_batch_plan_sha256 == schedule.physical_batch_plan.fingerprint
        for batch in schedule.batches()
    )
    assert (
        tuple(
            request.request_id
            for batch in schedule.batches()
            for request in batch.requests
        )
        == schedule.request_ids
    )
    by_arm = {
        arm.arm_code: sum(
            request.arm.arm_code == arm.arm_code for request in schedule.requests
        )
        for arm in PRIMARY_ARM_DEFINITIONS
    }
    assert by_arm == {
        "FULL_SINGLE": 4,
        "FULL_BAG_K": 64,
        "TILE_RESET": 64,
        "MASK_RESET": 64,
        "MASK_CUMULATIVE": 64,
    }
    assert [request.image_id for request in schedule.requests[:4]] == [
        10_000,
        10_001,
        10_002,
        10_003,
    ]
    first_bag = schedule.requests[4:8]
    assert {request.cell_index for request in first_bag} == {0}
    assert [request.image_frozen_order for request in first_bag] == [0, 1, 2, 3]
    cumulative = [
        request
        for request in schedule.requests
        if request.arm.arm_code == "MASK_CUMULATIVE"
    ]
    image_zero_cells = [
        request.cell_index for request in cumulative if request.image_frozen_order == 0
    ]
    assert image_zero_cells == list(range(16))


def test_request_identity_and_seed_are_batch_order_independent_and_arm_paired() -> None:
    schedule = _schedule(4)
    original = schedule.requests[17]
    moved = replace(original, schedule_index=999)
    moved.validate_request_id()
    assert moved.request_id == original.request_id

    image_id = 10_000
    cell_zero = [
        request
        for request in schedule.requests
        if request.image_id == image_id and request.cell_index == 0
    ]
    assert {request.arm.arm_code for request in cell_zero} == {
        "FULL_BAG_K",
        "TILE_RESET",
        "MASK_RESET",
        "MASK_CUMULATIVE",
    }
    assert len({request.sampling_seed for request in cell_zero}) == 1
    baseline = next(
        request
        for request in schedule.requests
        if request.image_id == image_id and request.arm.arm_code == "FULL_SINGLE"
    )
    assert baseline.sampling_seed != cell_zero[0].sampling_seed
    assert cell_zero[0].sampling_seed == derive_sampling_seed(
        root_seed=2026071301,
        role="paired-cell",
        image_id=image_id,
        cell_or_call_label="cell-00",
    )


def test_dense_union_schedule_seals_one_natural_three_request_tail_per_wave() -> None:
    schedule = _schedule(51)
    batches = schedule.batches()

    assert len(schedule.requests) == 3_315
    assert len(batches) == 833
    assert sum(batch.cardinality == 4 for batch in batches) == 816
    assert sum(batch.cardinality == 3 for batch in batches) == 17
    partitions = {
        batch.execution_wave_partition for batch in schedule.physical_batch_plan.batches
    }
    assert len(partitions) == 17
    assert all(
        sum(
            batch.execution_wave_partition == partition and batch.cardinality == 3
            for batch in batches
        )
        == 1
        for partition in partitions
    )


def test_validation_200_schedule_has_exact_budget_and_only_four_request_batches() -> (
    None
):
    schedule = _schedule(200)

    assert len(schedule.requests) == 13_000
    assert len(schedule.batches()) == 3_250
    assert {batch.cardinality for batch in schedule.batches()} == {4}
    assert len(
        {
            batch.execution_wave_partition
            for batch in schedule.physical_batch_plan.batches
        }
    ) == 17


def test_physical_plan_rejects_multiple_three_request_tails_in_one_wave() -> None:
    with pytest.raises(ArtifactContractError, match="final batch"):
        PhysicalBatchPlan(
            schedule_identity_sha256=_digest("schedule"),
            primary_batch_size=4,
            batches=(
                PhysicalBatchDefinition(
                    batch_index=0,
                    request_ids=("a", "b", "c"),
                ),
                PhysicalBatchDefinition(
                    batch_index=1,
                    request_ids=("d", "e", "f"),
                ),
            ),
        )


def test_schedule_round_trip_rejects_unknown_keys_and_primary_matrix_drift() -> None:
    schedule = _schedule(4)
    restored = ResearchSchedule.from_artifact_dict(schedule.to_artifact_dict())
    assert restored == schedule
    assert restored.fingerprint == schedule.fingerprint

    payload = schedule.to_artifact_dict()
    payload["unknown"] = True
    with pytest.raises(ArtifactContractError, match="unknown_keys"):
        ResearchSchedule.from_artifact_dict(payload)

    drifted_plan = schedule.to_artifact_dict()
    drifted_plan["physical_batch_plan"]["batches"][0]["request_ids"][0] = (
        schedule.requests[4].request_id
    )
    with pytest.raises(ArtifactContractError, match="fingerprint"):
        ResearchSchedule.from_artifact_dict(drifted_plan)

    drifted_wave = schedule.to_artifact_dict()
    drifted_wave["physical_batch_plan"]["batches"][0][
        "execution_wave_partition"
    ] = "cumulative-cell-00"
    with pytest.raises(ArtifactContractError, match="fingerprint"):
        ResearchSchedule.from_artifact_dict(drifted_wave)

    with pytest.raises(DataContractError, match="membership"):
        ResearchSchedule(
            identity=schedule.identity,
            requests=schedule.requests[:-1],
            physical_batch_plan=schedule.physical_batch_plan,
        )


def test_resume_materializes_only_never_attempted_requests_in_canonical_order() -> None:
    schedule = _schedule(4)
    first_batch = schedule.batches()[0]
    completed_batch = AttemptLedger(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
        records=tuple(
            _attempt(schedule, request.request_id) for request in first_batch.requests
        ),
    )
    pending = schedule.resume_batches(completed_batch)
    assert pending.runnable_batches[0].batch_index == 1
    assert (
        pending.runnable_batches[0].requests[0].request_id
        == schedule.requests[4].request_id
    )
    assert all(batch.cardinality == 4 for batch in pending.runnable_batches)

    one_attempt = AttemptLedger(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
        records=(_attempt(schedule, first_batch.requests[0].request_id),),
    )
    with pytest.raises(ArtifactContractError, match="partially terminal"):
        schedule.resume_batches(one_attempt)


def test_resume_rejects_identity_drift_and_unattested_tail_cardinality() -> None:
    schedule = _schedule(51)
    empty = AttemptLedger(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        execution_identity=schedule.identity.execution_identity,
    )
    plan = schedule.resume_batches(empty)
    assert plan.physical_batch_plan_sha256 == schedule.physical_batch_plan.fingerprint
    assert plan.held_batch is not None
    assert all(batch.cardinality in {3, 4} for batch in plan.runnable_batches)
    with pytest.raises(ArtifactContractError, match="not attested"):
        schedule.resume_batches(empty, attested_batch_cardinalities=frozenset({4}))

    drift = replace(empty, run_id="other-run")
    with pytest.raises(ArtifactContractError, match="identities differ"):
        schedule.resume_batches(drift)


def test_invalid_temperature_seed_role_and_request_arm_fail_fast() -> None:
    with pytest.raises(DataContractError, match="calibration candidate"):
        DecodeProvenance(
            temperature=0.5,
            canonical_generation_policy_sha256=_digest("generation-policy"),
            sampled_runtime_attestation_sha256=_digest("sampling-attestation"),
        )
    with pytest.raises(DataContractError, match="role is unknown"):
        derive_sampling_seed(
            root_seed=2026071301,
            role="unknown",
            image_id=1,
            cell_or_call_label="cell-00",
        )

    schedule = _schedule(4)
    drifted_arm = replace(
        schedule.requests[0].arm, operational_meaning="Drifted meaning."
    )
    with pytest.raises(DataContractError, match="arm definition differs"):
        replace(schedule.requests[0], arm=drifted_arm)


def test_resume_advances_only_after_all_cell_zero_states_are_reconstructible(
    tmp_path,
) -> None:
    schedule = _schedule(4)
    records = []
    cell_one_ids = set()
    for image_order in range(4):
        chain = _cumulative_requests(schedule, image_frozen_order=image_order)
        produced_state = _state_artifact(tmp_path, f"image-{image_order}-cell-00")
        records.append(
            _attempt(
                schedule,
                chain[0].request_id,
                expected_state_sha256=EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256,
                produced_state=produced_state,
            )
        )
        cell_one_ids.add(chain[1].request_id)

    ledger = _ledger(schedule, tuple(records))
    ledger.validate_dependencies(schedule.attempt_dependencies)
    plan = schedule.resume_batches(ledger)
    runnable_cell_one = {
        request.request_id
        for batch in plan.runnable_batches
        for request in batch.requests
        if request.arm.arm_code == "MASK_CUMULATIVE"
    }
    assert runnable_cell_one == cell_one_ids
    assert len(plan.deferred_dependencies) == 56
    assert not plan.blocked_dependencies
    assert all(batch.cardinality == 4 for batch in plan.runnable_batches)


@pytest.mark.parametrize("terminal_status", ["failed", "skipped", "capped", "invalid"])
def test_resume_blocks_entire_image_chain_after_terminal_predecessor(
    terminal_status: str,
    tmp_path,
) -> None:
    schedule = _schedule(4)
    chain = _cumulative_requests(schedule, image_frozen_order=0)
    records = [
        _attempt(
            schedule,
            chain[0].request_id,
            status=terminal_status,
            failure_code=f"research.{terminal_status}",
            expected_state_sha256=EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256,
        )
    ]
    for image_order in range(1, 4):
        other_chain = _cumulative_requests(schedule, image_frozen_order=image_order)
        records.append(
            _attempt(
                schedule,
                other_chain[0].request_id,
                expected_state_sha256=EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256,
                produced_state=_state_artifact(
                    tmp_path, f"image-{image_order}-cell-00"
                ),
            )
        )
    ledger = _ledger(schedule, tuple(records))

    with pytest.raises(
        ArtifactContractError, match="explicit continuation plan"
    ) as caught:
        schedule.resume_batches(ledger)
    assert caught.value.code == "analysis.resume_requires_continuation_plan"
    assert chain[1].request_id in caught.value.context["blocked_request_ids"]
    assert chain[0].request_id in caught.value.context["blocking_request_ids"]
    assert records[0].request_id not in ledger.successfully_completed_request_ids
    assert (
        dependency_skip_failure_code(
            predecessor_request_id=chain[0].request_id,
            predecessor_status=terminal_status,
        )
        in caught.value.context["required_terminal_skip_failure_codes"]
    )


def test_resume_crash_gap_requires_new_sealed_continuation_plan(
    tmp_path,
) -> None:
    schedule = _schedule(4)
    records = []
    state_paths = []
    cell_one_ids = set()
    broken_chain = _cumulative_requests(schedule, image_frozen_order=0)
    for image_order in range(4):
        chain = _cumulative_requests(schedule, image_frozen_order=image_order)
        produced_state = _state_artifact(tmp_path, f"image-{image_order}-cell-00")
        state_paths.append(produced_state[0])
        records.append(
            _attempt(
                schedule,
                chain[0].request_id,
                expected_state_sha256=EMPTY_ACCEPTED_ROW_PREFIX_STATE_SHA256,
                produced_state=produced_state,
            )
        )
        if image_order:
            cell_one_ids.add(chain[1].request_id)
    ledger = _ledger(schedule, tuple(records))
    ledger.validate_dependencies(schedule.attempt_dependencies)
    Path(state_paths[0]).unlink()

    with pytest.raises(
        ArtifactContractError, match="explicit continuation plan"
    ) as caught:
        schedule.resume_batches(ledger)
    assert caught.value.code == "analysis.resume_requires_continuation_plan"
    assert broken_chain[1].request_id in caught.value.context["blocked_request_ids"]
    assert broken_chain[0].request_id in caught.value.context["blocking_request_ids"]
    assert cell_one_ids.isdisjoint(caught.value.context["blocked_request_ids"])
    assert (
        dependency_state_unavailable_failure_code(
            predecessor_request_id=broken_chain[0].request_id
        )
        in caught.value.context["required_terminal_skip_failure_codes"]
    )


def test_val200_partial_batch_never_becomes_same_run_three_request_batch() -> None:
    schedule = _schedule(200)
    first_batch = schedule.batches()[0]
    assert {batch.cardinality for batch in schedule.batches()} == {4}
    ledger = _ledger(
        schedule,
        (_attempt(schedule, first_batch.requests[0].request_id),),
    )

    with pytest.raises(ArtifactContractError, match="partially terminal") as caught:
        schedule.resume_batches(ledger)
    assert caught.value.code == "analysis.resume_partial_physical_batch"
    assert caught.value.context["physical_batch_sha256"] == (
        first_batch.physical_batch_sha256
    )
    assert len(caught.value.context["pending_request_ids"]) == 3
