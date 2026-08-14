from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from scripts.research.human13_rp_crossover_matrix_contracts import (
    ARM_IDS,
    CANONICAL_IMAGE_IDS,
    DRY_RUN_COUNTER_KEYS,
    EVALUATION_RPS,
    MATRIX_SEED_GROUPS,
    PHASE_MATRIX,
    PHASE_QUALIFICATION,
    QUALIFICATION_SEED_GROUP,
    TRAINING_RPS,
    AcquisitionKey,
    AuditRef,
    CellKey,
    CellReceipt,
    CellSpec,
    MatrixPlan,
    SharedEvidenceRef,
    SourceBaselineRef,
)


# ---------------------------------------------------------------------------
# Private, formula-only fixture builders.  These live only in the test file:
# scientific admission never imports test convenience from the production
# module.
# ---------------------------------------------------------------------------


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _shared_evidence(*, training_rp: float, seed_group_id: str) -> SharedEvidenceRef:
    tag = f"{training_rp}:{seed_group_id}"
    return SharedEvidenceRef(
        source_sha256=_digest(f"source:{tag}"),
        manifest_sha256=_digest(f"manifest:{tag}"),
        acquisition_path=f"/artifacts/{tag}/acquisition.json",
        acquisition_sha256=_digest(f"acquisition:{tag}"),
        trajectory_credit_acquisition_sha256=_digest(f"credit-acq:{tag}"),
        credit_ledger_sha256=_digest(f"credit-ledger:{tag}"),
        compiler_ledger_sha256=_digest(f"compiler-ledger:{tag}"),
        policy_contract_sha256=_digest(f"policy:{tag}"),
    )


def _acquisition_key(
    *, training_rp: float, seed_group_id: str, phase: str = PHASE_MATRIX
) -> AcquisitionKey:
    return AcquisitionKey(
        training_rp=training_rp, seed_group_id=seed_group_id, phase=phase
    )


def _cell_spec(
    *,
    acquisition: AcquisitionKey,
    arm_id: str,
    shared_evidence: SharedEvidenceRef,
    proposal_tag: str | None = None,
    root_tag: str | None = None,
) -> CellSpec:
    tag = f"{acquisition.training_rp}:{acquisition.seed_group_id}:{arm_id}"
    components = {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }[arm_id]
    return CellSpec(
        cell_key=CellKey(acquisition_key=acquisition, arm_id=arm_id),
        shared_evidence=shared_evidence,
        leaf_config_sha256=_digest(f"leaf-config:{tag}"),
        source_checkpoint_sha256=_digest(f"checkpoint:{tag}"),
        expected_objective_components=components,
        fresh_adamw_fingerprint_sha256=_digest(f"proposal:{proposal_tag or tag}"),
        evaluation_rps=EVALUATION_RPS,
        output_root=f"/roots/{root_tag or tag}",
    )


def _source_baseline(evaluation_rp: float) -> SourceBaselineRef:
    output = _digest(f"source-output:{evaluation_rp}")
    return SourceBaselineRef(
        evaluation_rp=evaluation_rp,
        output_a_sha256=output,
        output_b_sha256=output,
        checkpoint_sha256=_digest(f"source-checkpoint:{evaluation_rp}"),
        image_ids=CANONICAL_IMAGE_IDS,
    )


def _audit_ref(evaluation_rp: float, *, tag: str) -> AuditRef:
    return AuditRef(
        evaluation_rp=evaluation_rp,
        evaluated_checkpoint_sha256=_digest(f"evaluated-checkpoint:{tag}"),
        output_path=f"/audits/{tag}/{evaluation_rp}.jsonl",
        output_sha256=_digest(f"audit-output:{tag}:{evaluation_rp}"),
        row_count=len(CANONICAL_IMAGE_IDS),
        image_ids=CANONICAL_IMAGE_IDS,
        generation_policy_receipt_sha256=_digest(
            f"generation-policy:{tag}:{evaluation_rp}"
        ),
    )


def _canonical_acquisitions() -> tuple[AcquisitionKey, ...]:
    return tuple(
        _acquisition_key(training_rp=rp, seed_group_id=group)
        for rp in TRAINING_RPS
        for group in MATRIX_SEED_GROUPS
    )


def _canonical_cells(acquisitions: tuple[AcquisitionKey, ...]) -> tuple[CellSpec, ...]:
    cells = []
    for acquisition in acquisitions:
        shared = _shared_evidence(
            training_rp=acquisition.training_rp, seed_group_id=acquisition.seed_group_id
        )
        for arm_id in ARM_IDS:
            cells.append(
                _cell_spec(
                    acquisition=acquisition, arm_id=arm_id, shared_evidence=shared
                )
            )
    return tuple(cells)


def _canonical_matrix_plan(
    *,
    acquisitions: tuple[AcquisitionKey, ...] | None = None,
    cells: tuple[CellSpec, ...] | None = None,
    source_baselines: tuple[SourceBaselineRef, ...] | None = None,
    dependency_edges: tuple[tuple[str, str], ...] | None = None,
    concurrency_cap: int = 8,
    dry_run_counters: dict[str, int] | None = None,
) -> MatrixPlan:
    acquisitions = (
        acquisitions if acquisitions is not None else _canonical_acquisitions()
    )
    cells = cells if cells is not None else _canonical_cells(acquisitions)
    source_baselines = (
        source_baselines
        if source_baselines is not None
        else tuple(_source_baseline(rp) for rp in EVALUATION_RPS)
    )
    dependency_edges = (
        dependency_edges
        if dependency_edges is not None
        else tuple(
            (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
            for cell in cells
        )
    )
    dry_run_counters = (
        dry_run_counters
        if dry_run_counters is not None
        else {key: 0 for key in DRY_RUN_COUNTER_KEYS}
    )
    return MatrixPlan(
        acquisitions=acquisitions,
        cells=cells,
        source_baselines=source_baselines,
        dependency_edges=dependency_edges,
        concurrency_cap=concurrency_cap,
        dry_run_counters=dry_run_counters,
    )


def _canonical_cell_receipt(cell: CellSpec, *, tag: str) -> CellReceipt:
    arm_id = cell.cell_key.arm_id
    requires_projection = arm_id == "C"
    return CellReceipt(
        cell_key=cell.cell_key,
        shared_evidence=cell.shared_evidence,
        objective_components=cell.expected_objective_components,
        before_transaction_digest=_digest(f"transaction:{tag}"),
        after_transaction_digest=_digest(f"transaction:{tag}"),
        status="succeeded",
        audits=(_audit_ref(1.0, tag=tag), _audit_ref(1.10, tag=tag)),
        adamw_proposal_sha256=_digest(f"adamw-proposal:{tag}"),
        projection_receipt_sha256=_digest(f"projection:{tag}")
        if requires_projection
        else None,
        apply_receipt_sha256=_digest(f"apply:{tag}"),
    )


# ---------------------------------------------------------------------------
# AcquisitionKey
# ---------------------------------------------------------------------------


def test_acquisition_key_accepts_canonical_matrix_group() -> None:
    key = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    assert key.training_rp == 1.0
    assert key.phase == PHASE_MATRIX


def test_acquisition_key_accepts_disjoint_qualification_group() -> None:
    key = _acquisition_key(
        training_rp=1.10,
        seed_group_id=QUALIFICATION_SEED_GROUP,
        phase=PHASE_QUALIFICATION,
    )
    assert key.seed_group_id == QUALIFICATION_SEED_GROUP


def test_acquisition_key_rejects_off_canonical_training_rp() -> None:
    with pytest.raises(ValueError, match="training_rp"):
        _acquisition_key(training_rp=1.05, seed_group_id="matrix_a")


def test_acquisition_key_rejects_qualification_seed_with_matrix_phase() -> None:
    with pytest.raises(ValueError, match="matrix-phase"):
        _acquisition_key(
            training_rp=1.0, seed_group_id=QUALIFICATION_SEED_GROUP, phase=PHASE_MATRIX
        )


def test_acquisition_key_rejects_matrix_seed_with_qualification_phase() -> None:
    with pytest.raises(ValueError, match="qualification-phase"):
        _acquisition_key(
            training_rp=1.0, seed_group_id="matrix_a", phase=PHASE_QUALIFICATION
        )


def test_acquisition_key_round_trips_through_dict() -> None:
    key = _acquisition_key(training_rp=1.10, seed_group_id="matrix_b")
    assert AcquisitionKey.from_dict(key.to_dict()) == key


def test_acquisition_key_is_frozen() -> None:
    key = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    with pytest.raises(FrozenInstanceError):
        key.training_rp = 1.10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CellKey
# ---------------------------------------------------------------------------


def test_cell_key_rejects_unknown_arm() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    with pytest.raises(ValueError, match="arm_id"):
        CellKey(acquisition_key=acquisition, arm_id="D")


def test_cell_key_round_trips_through_dict() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_c")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="B")
    assert CellKey.from_dict(cell_key.to_dict()) == cell_key


# ---------------------------------------------------------------------------
# SharedEvidenceRef
# ---------------------------------------------------------------------------


def test_shared_evidence_rejects_non_digest_field() -> None:
    with pytest.raises(ValueError, match="source_sha256"):
        SharedEvidenceRef(
            source_sha256="not-a-digest",
            manifest_sha256=_digest("m"),
            acquisition_path="/a",
            acquisition_sha256=_digest("a"),
            trajectory_credit_acquisition_sha256=_digest("t"),
            credit_ledger_sha256=_digest("c"),
            compiler_ledger_sha256=_digest("g"),
            policy_contract_sha256=_digest("p"),
        )


def test_shared_evidence_round_trips_and_hashes_deterministically() -> None:
    ref = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    assert SharedEvidenceRef.from_dict(ref.to_dict()) == ref
    assert ref.content_sha256 == ref.content_sha256
    other = _shared_evidence(training_rp=1.0, seed_group_id="matrix_b")
    assert ref.content_sha256 != other.content_sha256


# ---------------------------------------------------------------------------
# SourceBaselineRef
# ---------------------------------------------------------------------------


def test_source_baseline_rejects_nondeterministic_outputs() -> None:
    with pytest.raises(ValueError, match="byte-identical"):
        SourceBaselineRef(
            evaluation_rp=1.0,
            output_a_sha256=_digest("output-a"),
            output_b_sha256=_digest("output-b"),
            checkpoint_sha256=_digest("checkpoint"),
            image_ids=CANONICAL_IMAGE_IDS,
        )


def test_source_baseline_rejects_incomplete_panel_coverage() -> None:
    with pytest.raises(ValueError, match="thirteen-image panel"):
        SourceBaselineRef(
            evaluation_rp=1.0,
            output_a_sha256=_digest("output"),
            output_b_sha256=_digest("output"),
            checkpoint_sha256=_digest("checkpoint"),
            image_ids=CANONICAL_IMAGE_IDS[:12],
        )


def test_source_baseline_rejects_off_canonical_evaluation_rp() -> None:
    with pytest.raises(ValueError, match="evaluation_rp"):
        SourceBaselineRef(
            evaluation_rp=0.9,
            output_a_sha256=_digest("output"),
            output_b_sha256=_digest("output"),
            checkpoint_sha256=_digest("checkpoint"),
            image_ids=CANONICAL_IMAGE_IDS,
        )


# ---------------------------------------------------------------------------
# CellSpec: exact nested arm delta
# ---------------------------------------------------------------------------


def test_cell_spec_rejects_compiler_component_on_arm_a() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    with pytest.raises(ValueError, match="expected_objective_components"):
        CellSpec(
            cell_key=CellKey(acquisition_key=acquisition, arm_id="A"),
            shared_evidence=shared,
            leaf_config_sha256=_digest("leaf"),
            source_checkpoint_sha256=_digest("checkpoint"),
            expected_objective_components=("trajectory", "compiler"),
            fresh_adamw_fingerprint_sha256=_digest("proposal"),
            evaluation_rps=EVALUATION_RPS,
            output_root="/roots/a",
        )


def test_cell_spec_rejects_missing_preservation_component_on_arm_c() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    with pytest.raises(ValueError, match="expected_objective_components"):
        CellSpec(
            cell_key=CellKey(acquisition_key=acquisition, arm_id="C"),
            shared_evidence=shared,
            leaf_config_sha256=_digest("leaf"),
            source_checkpoint_sha256=_digest("checkpoint"),
            expected_objective_components=("trajectory", "compiler"),
            fresh_adamw_fingerprint_sha256=_digest("proposal"),
            evaluation_rps=EVALUATION_RPS,
            output_root="/roots/c",
        )


def test_cell_spec_rejects_second_update() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    with pytest.raises(ValueError, match="max_updates"):
        CellSpec(
            cell_key=CellKey(acquisition_key=acquisition, arm_id="A"),
            shared_evidence=shared,
            leaf_config_sha256=_digest("leaf"),
            source_checkpoint_sha256=_digest("checkpoint"),
            expected_objective_components=("trajectory",),
            fresh_adamw_fingerprint_sha256=_digest("proposal"),
            evaluation_rps=EVALUATION_RPS,
            output_root="/roots/a",
            max_updates=2,
        )


def test_cell_spec_rejects_adaptive_retry_policy() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    with pytest.raises(ValueError, match="retry_policy"):
        CellSpec(
            cell_key=CellKey(acquisition_key=acquisition, arm_id="A"),
            shared_evidence=shared,
            leaf_config_sha256=_digest("leaf"),
            source_checkpoint_sha256=_digest("checkpoint"),
            expected_objective_components=("trajectory",),
            fresh_adamw_fingerprint_sha256=_digest("proposal"),
            evaluation_rps=EVALUATION_RPS,
            output_root="/roots/a",
            retry_policy="on_failure",
        )


def test_cell_spec_rejects_single_evaluation_rp() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    with pytest.raises(ValueError, match="evaluation RPs"):
        CellSpec(
            cell_key=CellKey(acquisition_key=acquisition, arm_id="A"),
            shared_evidence=shared,
            leaf_config_sha256=_digest("leaf"),
            source_checkpoint_sha256=_digest("checkpoint"),
            expected_objective_components=("trajectory",),
            fresh_adamw_fingerprint_sha256=_digest("proposal"),
            evaluation_rps=(1.0,),
            output_root="/roots/a",
        )


def test_cell_spec_round_trips_through_dict() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    spec = _cell_spec(acquisition=acquisition, arm_id="B", shared_evidence=shared)
    assert CellSpec.from_dict(spec.to_dict()) == spec


# ---------------------------------------------------------------------------
# AuditRef
# ---------------------------------------------------------------------------


def test_audit_ref_rejects_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="row_count"):
        AuditRef(
            evaluation_rp=1.0,
            evaluated_checkpoint_sha256=_digest("checkpoint"),
            output_path="/audits/x.jsonl",
            output_sha256=_digest("output"),
            row_count=12,
            image_ids=CANONICAL_IMAGE_IDS,
            generation_policy_receipt_sha256=_digest("policy"),
        )


# ---------------------------------------------------------------------------
# CellReceipt: rollback, dual-RP audit, no retry, projection-by-arm
# ---------------------------------------------------------------------------


def test_cell_receipt_rejects_transaction_asymmetry() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="restore the pre-proposal transaction state"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("before"),
            after_transaction_digest=_digest("after"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"), _audit_ref(1.10, tag="x")),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
        )


def test_cell_receipt_rejects_single_rp_audit() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="exactly one audit per evaluation RP"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"),),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
        )


def test_cell_receipt_rejects_duplicate_rp_audit() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="duplicate"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"), _audit_ref(1.0, tag="x")),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
        )


def test_cell_receipt_rejects_projection_evidence_on_non_preservation_arm() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="only the preservation arm"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"), _audit_ref(1.10, tag="x")),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
            projection_receipt_sha256=_digest("projection"),
        )


def test_cell_receipt_requires_projection_evidence_on_preservation_arm() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="C")
    with pytest.raises(ValueError, match="projection evidence"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory", "compiler", "preservation"),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"), _audit_ref(1.10, tag="x")),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
        )


def test_cell_receipt_rejects_second_update() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="update_count"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"), _audit_ref(1.10, tag="x")),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
            update_count=2,
        )


def test_cell_receipt_rejects_retry_policy_change() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="retry_policy"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"), _audit_ref(1.10, tag="x")),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
            retry_policy="retry_once",
        )


def test_cell_receipt_rejects_rollback_not_confirmed() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="rollback_confirmed"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="succeeded",
            audits=(_audit_ref(1.0, tag="x"), _audit_ref(1.10, tag="x")),
            adamw_proposal_sha256=_digest("proposal"),
            apply_receipt_sha256=_digest("apply"),
            rollback_confirmed=False,
        )


def test_cell_receipt_failed_status_requires_failure_reason() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    with pytest.raises(ValueError, match="failure_reason"):
        CellReceipt(
            cell_key=cell_key,
            shared_evidence=shared,
            objective_components=("trajectory",),
            before_transaction_digest=_digest("t"),
            after_transaction_digest=_digest("t"),
            status="failed",
        )


def test_cell_receipt_failed_status_still_requires_transaction_symmetry_and_rollback() -> (
    None
):
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell_key = CellKey(acquisition_key=acquisition, arm_id="A")
    receipt = CellReceipt(
        cell_key=cell_key,
        shared_evidence=shared,
        objective_components=("trajectory",),
        before_transaction_digest=_digest("t"),
        after_transaction_digest=_digest("t"),
        status="failed",
        failure_reason="projection infeasible",
    )
    assert receipt.rollback_confirmed is True
    assert receipt.before_transaction_digest == receipt.after_transaction_digest


def test_cell_receipt_succeeded_round_trips_through_dict() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell = _cell_spec(acquisition=acquisition, arm_id="C", shared_evidence=shared)
    receipt = _canonical_cell_receipt(cell, tag="c-arm")
    assert CellReceipt.from_dict(receipt.to_dict()) == receipt


# ---------------------------------------------------------------------------
# MatrixPlan: the aggregate admission choke point
# ---------------------------------------------------------------------------


def test_matrix_plan_admits_the_canonical_six_by_eighteen_matrix() -> None:
    plan = _canonical_matrix_plan()
    assert len(plan.acquisitions) == 6
    assert len(plan.cells) == 18
    assert len(plan.source_baselines) == 2
    assert plan.content_sha256 == plan.content_sha256


def test_matrix_plan_rejects_seventeen_cells() -> None:
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    cells.pop()
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="eighteen cells"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


def test_matrix_plan_rejects_duplicate_arm_within_one_acquisition() -> None:
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    # Replace the last cell (arm C of the last acquisition) with a second arm A,
    # so one acquisition has two A cells and zero C cells -- still eighteen total.
    last_acquisition = acquisitions[-1]
    shared = _shared_evidence(
        training_rp=last_acquisition.training_rp,
        seed_group_id=last_acquisition.seed_group_id,
    )
    duplicate = _cell_spec(
        acquisition=last_acquisition,
        arm_id="A",
        shared_evidence=shared,
        proposal_tag="duplicate-arm",
        root_tag="duplicate-arm",
    )
    cells[-1] = duplicate
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="arm"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


def test_matrix_plan_rejects_qualification_acquisition_pooled_into_matrix() -> None:
    acquisitions = list(_canonical_acquisitions())
    acquisitions[0] = _acquisition_key(
        training_rp=1.0,
        seed_group_id=QUALIFICATION_SEED_GROUP,
        phase=PHASE_QUALIFICATION,
    )
    cells = _canonical_cells(tuple(acquisitions))
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="qualification"):
        _canonical_matrix_plan(
            acquisitions=tuple(acquisitions), cells=cells, dependency_edges=edges
        )


def test_matrix_plan_rejects_duplicate_acquisition_pair() -> None:
    acquisitions = list(_canonical_acquisitions())
    acquisitions[0] = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    acquisitions[1] = _acquisition_key(
        training_rp=1.0, seed_group_id="matrix_a"
    )  # duplicate, not matrix_b
    cells = _canonical_cells(tuple(acquisitions))
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="2x3 RP/seed-group surface"):
        _canonical_matrix_plan(
            acquisitions=tuple(acquisitions), cells=cells, dependency_edges=edges
        )


def test_matrix_plan_rejects_mismatched_shared_evidence_within_one_group() -> None:
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    target_acquisition = acquisitions[0]
    divergent_shared = _shared_evidence(training_rp=1.10, seed_group_id="matrix_b")
    for index, cell in enumerate(cells):
        if (
            cell.cell_key.acquisition_key == target_acquisition
            and cell.cell_key.arm_id == "B"
        ):
            cells[index] = _cell_spec(
                acquisition=target_acquisition,
                arm_id="B",
                shared_evidence=divergent_shared,
                proposal_tag="divergent-b",
                root_tag="divergent-b",
            )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="byte-identical evidence"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


def test_matrix_plan_rejects_duplicate_output_root() -> None:
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    first_shared = _shared_evidence(
        training_rp=acquisitions[0].training_rp,
        seed_group_id=acquisitions[0].seed_group_id,
    )
    collided = _cell_spec(
        acquisition=acquisitions[0],
        arm_id="A",
        shared_evidence=first_shared,
        proposal_tag="unique-proposal",
        root_tag="collision",
    )
    cells[0] = collided
    # Force a second cell to reuse the same output_root.
    other_shared = _shared_evidence(
        training_rp=acquisitions[1].training_rp,
        seed_group_id=acquisitions[1].seed_group_id,
    )
    for index, cell in enumerate(cells):
        if (
            cell.cell_key.acquisition_key == acquisitions[1]
            and cell.cell_key.arm_id == "A"
        ):
            cells[index] = _cell_spec(
                acquisition=acquisitions[1],
                arm_id="A",
                shared_evidence=other_shared,
                proposal_tag="other-unique-proposal",
                root_tag="collision",
            )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="output_root must be unique"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


def test_matrix_plan_rejects_reused_proposal_identity_across_cells() -> None:
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    first_shared = _shared_evidence(
        training_rp=acquisitions[0].training_rp,
        seed_group_id=acquisitions[0].seed_group_id,
    )
    second_shared = _shared_evidence(
        training_rp=acquisitions[1].training_rp,
        seed_group_id=acquisitions[1].seed_group_id,
    )
    shared_proposal_tag = "reused-proposal"
    for index, cell in enumerate(cells):
        if (
            cell.cell_key.acquisition_key == acquisitions[0]
            and cell.cell_key.arm_id == "A"
        ):
            cells[index] = _cell_spec(
                acquisition=acquisitions[0],
                arm_id="A",
                shared_evidence=first_shared,
                proposal_tag=shared_proposal_tag,
                root_tag="root-a0",
            )
        if (
            cell.cell_key.acquisition_key == acquisitions[1]
            and cell.cell_key.arm_id == "A"
        ):
            cells[index] = _cell_spec(
                acquisition=acquisitions[1],
                arm_id="A",
                shared_evidence=second_shared,
                proposal_tag=shared_proposal_tag,
                root_tag="root-a1",
            )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="independent fresh AdamW proposal identity"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


def test_matrix_plan_rejects_single_source_baseline_rp() -> None:
    with pytest.raises(ValueError, match="two RP source baselines"):
        _canonical_matrix_plan(source_baselines=(_source_baseline(1.0),))


def test_matrix_plan_rejects_duplicate_source_baseline_rp() -> None:
    with pytest.raises(ValueError, match="RP 1.0 and RP 1.10 exactly once"):
        _canonical_matrix_plan(
            source_baselines=(_source_baseline(1.0), _source_baseline(1.0))
        )


def test_matrix_plan_rejects_missing_dependency_edge() -> None:
    acquisitions = _canonical_acquisitions()
    cells = _canonical_cells(acquisitions)
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )[:-1]
    with pytest.raises(ValueError, match="dependency edges"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=cells, dependency_edges=edges
        )


def test_matrix_plan_rejects_forged_dependency_edge() -> None:
    acquisitions = _canonical_acquisitions()
    cells = _canonical_cells(acquisitions)
    edges = list(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    edges[0] = (edges[0][0], "0" * 64)
    with pytest.raises(ValueError, match="dependency edges"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=cells, dependency_edges=tuple(edges)
        )


def test_matrix_plan_rejects_concurrency_cap_above_eight() -> None:
    with pytest.raises(ValueError, match="concurrency_cap"):
        _canonical_matrix_plan(concurrency_cap=9)


def test_matrix_plan_rejects_concurrency_cap_below_one() -> None:
    with pytest.raises(ValueError, match="concurrency_cap"):
        _canonical_matrix_plan(concurrency_cap=0)


def test_matrix_plan_rejects_nonzero_dry_run_counter() -> None:
    counters = {key: 0 for key in DRY_RUN_COUNTER_KEYS}
    counters["gpu_allocations"] = 1
    with pytest.raises(ValueError, match="dry_run_counters must be all zero"):
        _canonical_matrix_plan(dry_run_counters=counters)


def test_matrix_plan_rejects_incomplete_dry_run_counter_schema() -> None:
    counters = {key: 0 for key in DRY_RUN_COUNTER_KEYS if key != "gpu_allocations"}
    with pytest.raises(ValueError, match="dry_run_counters schema"):
        _canonical_matrix_plan(dry_run_counters=counters)


def test_matrix_plan_freezes_mutable_inputs_before_hashing() -> None:
    acquisitions = list(_canonical_acquisitions())
    cells = list(_canonical_cells(tuple(acquisitions)))
    edges = [
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    ]
    plan = MatrixPlan(
        acquisitions=acquisitions,  # type: ignore[arg-type]
        cells=cells,  # type: ignore[arg-type]
        source_baselines=[_source_baseline(rp) for rp in EVALUATION_RPS],  # type: ignore[arg-type]
        dependency_edges=edges,  # type: ignore[arg-type]
        concurrency_cap=8,
        dry_run_counters={key: 0 for key in DRY_RUN_COUNTER_KEYS},
    )
    assert isinstance(plan.acquisitions, tuple)
    assert isinstance(plan.cells, tuple)
    assert isinstance(plan.dependency_edges, tuple)
    assert all(isinstance(edge, tuple) for edge in plan.dependency_edges)
    digest_before_mutation = plan.content_sha256
    acquisitions.append(acquisitions[0])
    cells.append(cells[0])
    assert plan.content_sha256 == digest_before_mutation


def test_matrix_plan_is_frozen() -> None:
    plan = _canonical_matrix_plan()
    with pytest.raises(FrozenInstanceError):
        plan.concurrency_cap = 1  # type: ignore[misc]


def test_matrix_plan_round_trips_through_dict_via_the_same_admission() -> None:
    plan = _canonical_matrix_plan()
    reloaded = MatrixPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert reloaded.content_sha256 == plan.content_sha256


def test_matrix_plan_from_dict_rejects_a_locally_valid_but_cross_artifact_mixed_plan() -> (
    None
):
    plan = _canonical_matrix_plan()
    payload = plan.to_dict()
    # Sever one dependency edge so the persisted payload is locally well-typed
    # per-record but the aggregate cross-artifact relationship is mixed.
    payload["dependency_edges"][0][1] = "0" * 64
    with pytest.raises(ValueError, match="dependency edges"):
        MatrixPlan.from_dict(payload)
