from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from scripts.research.human13_rp_crossover_matrix_contracts import (
    ARM_IDS,
    CANONICAL_IMAGE_IDS,
    CANONICAL_SEED_GROUPS,
    DRY_RUN_COUNTER_KEYS,
    EVALUATION_RPS,
    MATRIX_SEED_GROUPS,
    PHASE_MATRIX,
    PHASE_QUALIFICATION,
    PROPOSAL_COMPONENTS_BY_ARM,
    QUALIFICATION_LEARNING_RATE_RAY,
    QUALIFICATION_SEED_GROUP,
    TRAINING_RPS,
    AcquisitionKey,
    AuditRef,
    CellKey,
    CellReceipt,
    CellSpec,
    MatrixPlan,
    NodeTerminalReceipt,
    SharedEvidenceRef,
    SourceBaselineRef,
    canonical_seeds,
    validate_matrix_receipts,
)


# ---------------------------------------------------------------------------
# Private, formula-only fixture builders.  These live only in the test file:
# scientific admission never imports test convenience from the production
# module.
# ---------------------------------------------------------------------------


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


SOURCE_SHA256 = _digest("source-checkpoint")
MANIFEST_SHA256 = _digest("manifest")
ADAMW_CONFIG_SHA256 = _digest("frozen-adamw-config")


def _shared_evidence(
    *,
    training_rp: float,
    seed_group_id: str,
    acquisition_tag: str | None = None,
) -> SharedEvidenceRef:
    tag = acquisition_tag or f"{training_rp}:{seed_group_id}"
    return SharedEvidenceRef(
        source_sha256=SOURCE_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        acquisition_path=f"/artifacts/{tag}/acquisition.json",
        acquisition_sha256=_digest(f"acquisition:{tag}"),
        trajectory_credit_acquisition_sha256=_digest(f"credit-acq:{tag}"),
        credit_ledger_sha256=_digest(f"credit-ledger:{tag}"),
        compiler_ledger_sha256=_digest(f"compiler-ledger:{tag}"),
        policy_contract_sha256=_digest(f"policy:{training_rp}"),
        native_receipts_sha256=_digest(f"native-receipts:{tag}"),
        training_rp=training_rp,
        seed_group_id=seed_group_id,
        seeds=canonical_seeds(seed_group_id),
    )


def _acquisition_key(
    *,
    training_rp: float,
    seed_group_id: str,
    phase: str = PHASE_MATRIX,
    seeds: tuple[int, ...] | None = None,
) -> AcquisitionKey:
    return AcquisitionKey(
        training_rp=training_rp,
        seed_group_id=seed_group_id,
        phase=phase,
        seeds=seeds,
    )


def _component_hashes(
    *,
    arm_id: str,
    training_rp: float,
    seed_group_id: str,
    trajectory_tag: str | None = None,
    compiler_tag: str | None = None,
) -> tuple[tuple[str, str], ...]:
    group = f"{training_rp}:{seed_group_id}"
    values = {
        "trajectory": _digest(f"trajectory:{trajectory_tag or group}"),
        "compiler": _digest(f"compiler:{compiler_tag or group}"),
    }
    return tuple(
        (component, values[component])
        for component in PROPOSAL_COMPONENTS_BY_ARM[arm_id]
    )


def _cell_spec(
    *,
    acquisition: AcquisitionKey,
    arm_id: str,
    shared_evidence: SharedEvidenceRef,
    optimizer_tag: str | None = None,
    root_tag: str | None = None,
    config_tag: str | None = None,
    adamw_config_sha256: str = ADAMW_CONFIG_SHA256,
    objective_component_hashes: tuple[tuple[str, str], ...] | None = None,
    learning_rate: float = 3.0e-6,
) -> CellSpec:
    tag = f"{acquisition.training_rp}:{acquisition.seed_group_id}:{arm_id}"
    components = {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }[arm_id]
    return CellSpec(
        cell_key=CellKey(
            acquisition_key=acquisition,
            arm_id=arm_id,
            qualification_learning_rate=(
                learning_rate if acquisition.phase == PHASE_QUALIFICATION else None
            ),
        ),
        shared_evidence=shared_evidence,
        leaf_config_sha256=_digest(
            f"leaf-config:{config_tag or f'{acquisition.training_rp}:{arm_id}'}"
        ),
        source_checkpoint_sha256=SOURCE_SHA256,
        expected_objective_components=components,
        objective_component_hashes=(
            objective_component_hashes
            if objective_component_hashes is not None
            else _component_hashes(
                arm_id=arm_id,
                training_rp=acquisition.training_rp,
                seed_group_id=acquisition.seed_group_id,
            )
        ),
        adamw_config_sha256=adamw_config_sha256,
        fresh_optimizer_identity_sha256=_digest(f"optimizer:{optimizer_tag or tag}"),
        evaluation_rps=EVALUATION_RPS,
        output_root=f"/roots/{root_tag or tag}",
        learning_rate=learning_rate,
        global_learning_rate_decision_sha256=(
            _digest("global-lr-decision") if acquisition.phase == PHASE_MATRIX else None
        ),
        resolved_leaf_config_sha256=(
            _digest(f"resolved-leaf:{config_tag or tag}:{learning_rate}")
            if acquisition.phase == PHASE_MATRIX
            else _digest(f"resolved-qualification-leaf:{tag}:{learning_rate}")
        ),
    )


def _source_baseline(evaluation_rp: float) -> SourceBaselineRef:
    output = _digest(f"source-output:{evaluation_rp}")
    return SourceBaselineRef(
        evaluation_rp=evaluation_rp,
        output_a_sha256=output,
        output_b_sha256=output,
        checkpoint_sha256=SOURCE_SHA256,
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


def _transaction_id(tag: str) -> str:
    return _digest(f"transaction-id:{tag}")[:32]


def _proposal_delta(cell: CellSpec) -> str:
    """B and C measure the same base AdamW proposal; A measures its own."""

    acquisition = cell.cell_key.acquisition_key
    stage = "a" if cell.cell_key.arm_id == "A" else "bc"
    return _digest(
        f"proposal-delta:{acquisition.training_rp}:{acquisition.seed_group_id}:{stage}"
    )


def _canonical_cell_receipt(
    cell: CellSpec,
    *,
    tag: str,
    proposal_delta_sha256: str | None = None,
    transaction_id: str | None = None,
) -> CellReceipt:
    arm_id = cell.cell_key.arm_id
    requires_projection = arm_id == "C"
    proposal_sha256 = _digest(f"adamw-proposal:{tag}")
    projection_sha256 = _digest(f"projection:{tag}") if requires_projection else None
    apply_sha256 = _digest(f"apply:{tag}")
    witness_sha256 = _digest(f"witness:{tag}") if requires_projection else None
    return CellReceipt(
        cell_key=cell.cell_key,
        shared_evidence=cell.shared_evidence,
        objective_components=cell.expected_objective_components,
        objective_component_hashes=cell.objective_component_hashes,
        adamw_config_sha256=cell.adamw_config_sha256,
        fresh_optimizer_identity_sha256=cell.fresh_optimizer_identity_sha256,
        transaction_id=transaction_id or _transaction_id(tag),
        before_transaction_digest=_digest(f"transaction:{tag}"),
        after_transaction_digest=_digest(f"transaction:{tag}"),
        status="succeeded",
        audits=(_audit_ref(1.0, tag=tag), _audit_ref(1.10, tag=tag)),
        adamw_proposal_sha256=proposal_sha256,
        proposal_delta_sha256=proposal_delta_sha256 or _proposal_delta(cell),
        projection_receipt_sha256=projection_sha256,
        apply_receipt_sha256=apply_sha256,
        adamw_proposal_artifact_path=(f"/immutable/proposal/{proposal_sha256}.json"),
        witness_bank_artifact_path=(
            f"/immutable/witness/{witness_sha256}"
            if witness_sha256 is not None
            else None
        ),
        witness_bank_sha256=witness_sha256,
        projection_receipt_artifact_path=(
            f"/immutable/projection/{projection_sha256}.json"
            if projection_sha256 is not None
            else None
        ),
        apply_receipt_artifact_path=(
            f"/immutable/apply/{apply_sha256}.json" if requires_projection else None
        ),
        learning_rate=cell.learning_rate,
        global_learning_rate_decision_sha256=(
            cell.global_learning_rate_decision_sha256
        ),
        resolved_leaf_config_sha256=cell.resolved_leaf_config_sha256,
    )


def _canonical_receipts(plan: MatrixPlan) -> tuple[CellReceipt, ...]:
    return tuple(
        _canonical_cell_receipt(
            cell,
            tag=(
                f"{cell.cell_key.acquisition_key.training_rp}:"
                f"{cell.cell_key.acquisition_key.seed_group_id}:{cell.cell_key.arm_id}"
            ),
        )
        for cell in plan.cells
    )


def _node_terminal(
    acquisition: AcquisitionKey,
    *,
    arm_ids: tuple[str, ...] = ARM_IDS,
    status: str = "succeeded",
    failure_reason: str | None = None,
    receipt_count: int | None = None,
) -> NodeTerminalReceipt:
    shared = _shared_evidence(
        training_rp=acquisition.training_rp, seed_group_id=acquisition.seed_group_id
    )
    if acquisition.phase == PHASE_QUALIFICATION:
        specs = tuple(
            _cell_spec(
                acquisition=acquisition,
                arm_id="C",
                shared_evidence=shared,
                learning_rate=dose,
                optimizer_tag=f"dose:{dose}",
                root_tag=f"dose:{dose}",
            )
            for dose in QUALIFICATION_LEARNING_RATE_RAY
        )
    else:
        specs = tuple(
            _cell_spec(acquisition=acquisition, arm_id=arm_id, shared_evidence=shared)
            for arm_id in arm_ids
        )
    receipts = tuple(
        _canonical_cell_receipt(
            cell,
            tag=(
                f"{acquisition.training_rp}:{acquisition.seed_group_id}:"
                f"{cell.cell_key.arm_id}"
            ),
        )
        for cell in specs
    )
    if receipt_count is not None:
        receipts = receipts[:receipt_count]
    return NodeTerminalReceipt(
        node_id=f"{acquisition.training_rp}:{acquisition.seed_group_id}",
        phase=acquisition.phase,
        acquisition_key=acquisition,
        status=status,
        cell_specs=specs,
        cell_receipts=receipts,
        failure_reason=failure_reason,
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
        replace(
            _shared_evidence(training_rp=1.0, seed_group_id="matrix_a"),
            source_sha256="not-a-digest",
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


def _cell_spec_kwargs(arm_id: str = "A") -> dict[str, object]:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    components = {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }[arm_id]
    return {
        "cell_key": CellKey(acquisition_key=acquisition, arm_id=arm_id),
        "shared_evidence": shared,
        "leaf_config_sha256": _digest("leaf"),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "expected_objective_components": components,
        "objective_component_hashes": _component_hashes(
            arm_id=arm_id, training_rp=1.0, seed_group_id="matrix_a"
        ),
        "adamw_config_sha256": ADAMW_CONFIG_SHA256,
        "fresh_optimizer_identity_sha256": _digest("optimizer"),
        "evaluation_rps": EVALUATION_RPS,
        "output_root": f"/roots/{arm_id.lower()}",
    }


def test_cell_spec_rejects_compiler_component_on_arm_a() -> None:
    with pytest.raises(ValueError, match="expected_objective_components"):
        CellSpec(
            **{
                **_cell_spec_kwargs("A"),
                "expected_objective_components": ("trajectory", "compiler"),
            }
        )


def test_cell_spec_rejects_missing_preservation_component_on_arm_c() -> None:
    with pytest.raises(ValueError, match="expected_objective_components"):
        CellSpec(
            **{
                **_cell_spec_kwargs("C"),
                "expected_objective_components": ("trajectory", "compiler"),
            }
        )


def test_cell_spec_rejects_second_update() -> None:
    with pytest.raises(ValueError, match="max_updates"):
        CellSpec(**{**_cell_spec_kwargs("A"), "max_updates": 2})


def test_cell_spec_rejects_adaptive_retry_policy() -> None:
    with pytest.raises(ValueError, match="retry_policy"):
        CellSpec(**{**_cell_spec_kwargs("A"), "retry_policy": "on_failure"})


def test_cell_spec_rejects_single_evaluation_rp() -> None:
    with pytest.raises(ValueError, match="evaluation RPs"):
        CellSpec(**{**_cell_spec_kwargs("A"), "evaluation_rps": (1.0,)})


def test_cell_spec_rejects_objective_hashes_outside_the_arm_proposal_surface() -> None:
    with pytest.raises(ValueError, match="objective_component_hashes"):
        CellSpec(
            **{
                **_cell_spec_kwargs("A"),
                "objective_component_hashes": (
                    ("trajectory", _digest("trajectory")),
                    ("compiler", _digest("compiler")),
                ),
            }
        )


def test_cell_spec_rejects_evidence_from_a_different_seed_group() -> None:
    with pytest.raises(ValueError, match="seed group"):
        CellSpec(
            **{
                **_cell_spec_kwargs("A"),
                "shared_evidence": _shared_evidence(
                    training_rp=1.0, seed_group_id="matrix_b"
                ),
            }
        )


def test_cell_spec_rejects_evidence_from_a_different_training_rp() -> None:
    with pytest.raises(ValueError, match="training RP"):
        CellSpec(
            **{
                **_cell_spec_kwargs("A"),
                "shared_evidence": _shared_evidence(
                    training_rp=1.10, seed_group_id="matrix_a"
                ),
            }
        )


def test_cell_spec_rejects_a_source_checkpoint_outside_its_evidence() -> None:
    with pytest.raises(ValueError, match="Source checkpoint"):
        CellSpec(
            **{
                **_cell_spec_kwargs("A"),
                "source_checkpoint_sha256": _digest("other-source"),
            }
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


def _cell_receipt_kwargs(arm_id: str = "A") -> dict[str, object]:
    spec = CellSpec(**_cell_spec_kwargs(arm_id))  # type: ignore[arg-type]
    proposal_sha256 = _digest("proposal")
    projection_sha256 = _digest("projection") if arm_id == "C" else None
    apply_sha256 = _digest("apply")
    return {
        "cell_key": spec.cell_key,
        "shared_evidence": spec.shared_evidence,
        "objective_components": spec.expected_objective_components,
        "objective_component_hashes": spec.objective_component_hashes,
        "adamw_config_sha256": spec.adamw_config_sha256,
        "fresh_optimizer_identity_sha256": spec.fresh_optimizer_identity_sha256,
        "transaction_id": _transaction_id("x"),
        "before_transaction_digest": _digest("t"),
        "after_transaction_digest": _digest("t"),
        "status": "succeeded",
        "audits": (_audit_ref(1.0, tag="x"), _audit_ref(1.10, tag="x")),
        "adamw_proposal_sha256": proposal_sha256,
        "proposal_delta_sha256": _digest("proposal-delta"),
        "apply_receipt_sha256": apply_sha256,
        "projection_receipt_sha256": projection_sha256,
        "adamw_proposal_artifact_path": f"/immutable/proposal/{proposal_sha256}.json",
        "witness_bank_artifact_path": (
            f"/immutable/witness/{_digest('witness')}" if arm_id == "C" else None
        ),
        "witness_bank_sha256": _digest("witness") if arm_id == "C" else None,
        "projection_receipt_artifact_path": (
            f"/immutable/projection/{projection_sha256}.json"
            if projection_sha256 is not None
            else None
        ),
        "apply_receipt_artifact_path": (
            f"/immutable/apply/{apply_sha256}.json" if arm_id == "C" else None
        ),
    }


def test_cell_receipt_rejects_transaction_asymmetry() -> None:
    with pytest.raises(ValueError, match="restore the pre-proposal transaction state"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("A"),
                "before_transaction_digest": _digest("before"),
                "after_transaction_digest": _digest("after"),
            }
        )


def test_cell_receipt_rejects_single_rp_audit() -> None:
    with pytest.raises(ValueError, match="exactly one audit per evaluation RP"):
        CellReceipt(
            **{**_cell_receipt_kwargs("A"), "audits": (_audit_ref(1.0, tag="x"),)}
        )


def test_cell_receipt_rejects_duplicate_rp_audit() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("A"),
                "audits": (_audit_ref(1.0, tag="x"), _audit_ref(1.0, tag="x")),
            }
        )


def test_cell_receipt_rejects_projection_evidence_on_non_preservation_arm() -> None:
    with pytest.raises(ValueError, match="only the preservation arm"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("A"),
                "projection_receipt_sha256": _digest("projection"),
            }
        )


def test_cell_receipt_requires_projection_evidence_on_preservation_arm() -> None:
    with pytest.raises(ValueError, match="projection evidence"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("C"),
                "projection_receipt_sha256": None,
                "projection_receipt_artifact_path": None,
            }
        )


def test_cell_receipt_rejects_second_update() -> None:
    with pytest.raises(ValueError, match="update_count"):
        CellReceipt(**{**_cell_receipt_kwargs("A"), "update_count": 2})


def test_cell_receipt_rejects_retry_policy_change() -> None:
    with pytest.raises(ValueError, match="retry_policy"):
        CellReceipt(**{**_cell_receipt_kwargs("A"), "retry_policy": "retry_once"})


def test_cell_receipt_rejects_rollback_not_confirmed() -> None:
    with pytest.raises(ValueError, match="rollback_confirmed"):
        CellReceipt(**{**_cell_receipt_kwargs("A"), "rollback_confirmed": False})


def test_cell_receipt_rejects_objective_hashes_outside_the_arm_surface() -> None:
    with pytest.raises(ValueError, match="objective_component_hashes"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("A"),
                "objective_component_hashes": (
                    ("trajectory", _digest("trajectory")),
                    ("compiler", _digest("compiler")),
                ),
            }
        )


def test_cell_receipt_requires_its_measured_proposal_delta_on_success() -> None:
    with pytest.raises(ValueError, match="proposal delta"):
        CellReceipt(**{**_cell_receipt_kwargs("A"), "proposal_delta_sha256": None})


def test_cell_receipt_rejects_an_untyped_transaction_identity() -> None:
    with pytest.raises(ValueError, match="transaction_id"):
        CellReceipt(**{**_cell_receipt_kwargs("A"), "transaction_id": "not-a-uuid"})


def test_cell_receipt_failed_status_requires_failure_reason() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("A"),
                "status": "failed",
                "audits": (),
                "adamw_proposal_sha256": None,
                "proposal_delta_sha256": None,
                "apply_receipt_sha256": None,
                "adamw_proposal_artifact_path": None,
                "objective_component_hashes": (),
            }
        )


def test_cell_receipt_failed_status_still_requires_transaction_symmetry_and_rollback() -> (
    None
):
    receipt = CellReceipt(
        **{
            **_cell_receipt_kwargs("A"),
            "status": "failed",
            "audits": (),
            "adamw_proposal_sha256": None,
            "proposal_delta_sha256": None,
            "apply_receipt_sha256": None,
            "adamw_proposal_artifact_path": None,
            "objective_component_hashes": (),
            "failure_reason": "projection infeasible",
        }
    )
    assert receipt.rollback_confirmed is True
    assert receipt.before_transaction_digest == receipt.after_transaction_digest


def test_cell_receipt_succeeded_round_trips_through_dict() -> None:
    acquisition = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    shared = _shared_evidence(training_rp=1.0, seed_group_id="matrix_a")
    cell = _cell_spec(acquisition=acquisition, arm_id="C", shared_evidence=shared)
    receipt = _canonical_cell_receipt(cell, tag="c-arm")
    reloaded = CellReceipt.from_dict(receipt.to_dict())
    assert reloaded == receipt
    assert reloaded.adamw_proposal_artifact_path == (
        receipt.adamw_proposal_artifact_path
    )
    assert reloaded.witness_bank_sha256 == receipt.witness_bank_sha256
    assert reloaded.projection_receipt_artifact_path == (
        receipt.projection_receipt_artifact_path
    )
    assert reloaded.apply_receipt_artifact_path == (receipt.apply_receipt_artifact_path)


def test_cell_receipt_fails_closed_on_missing_or_relative_decision_artifacts() -> None:
    with pytest.raises(ValueError, match="proposal artifact path/hash"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("A"),
                "adamw_proposal_artifact_path": None,
            }
        )
    with pytest.raises(ValueError, match="witness-bank artifact path/hash"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("C"),
                "witness_bank_artifact_path": None,
            }
        )
    with pytest.raises(ValueError, match="absolute immutable artifact path"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("C"),
                "projection_receipt_artifact_path": "relative/projection.json",
            }
        )
    with pytest.raises(ValueError, match="content-addressed hash"):
        CellReceipt(
            **{
                **_cell_receipt_kwargs("A"),
                "adamw_proposal_artifact_path": "/immutable/proposal/forged.json",
            }
        )


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
        optimizer_tag="duplicate-arm",
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
    with pytest.raises(ValueError, match="qualification cell keys may bind only arm C"):
        _canonical_cells(tuple(acquisitions))


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
    divergent_shared = _shared_evidence(
        training_rp=target_acquisition.training_rp,
        seed_group_id=target_acquisition.seed_group_id,
        acquisition_tag="divergent-acquisition",
    )
    for index, cell in enumerate(cells):
        if (
            cell.cell_key.acquisition_key == target_acquisition
            and cell.cell_key.arm_id == "B"
        ):
            cells[index] = _cell_spec(
                acquisition=target_acquisition,
                arm_id="B",
                shared_evidence=divergent_shared,
                optimizer_tag="divergent-b",
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
        optimizer_tag="unique-proposal",
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
                optimizer_tag="other-unique-proposal",
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


def test_matrix_plan_rejects_reused_optimizer_identity_across_cells() -> None:
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
    shared_optimizer_tag = "reused-optimizer"
    for index, cell in enumerate(cells):
        if (
            cell.cell_key.acquisition_key == acquisitions[0]
            and cell.cell_key.arm_id == "A"
        ):
            cells[index] = _cell_spec(
                acquisition=acquisitions[0],
                arm_id="A",
                shared_evidence=first_shared,
                optimizer_tag=shared_optimizer_tag,
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
                optimizer_tag=shared_optimizer_tag,
                root_tag="root-a1",
            )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="independent fresh optimizer identity"):
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


# ---------------------------------------------------------------------------
# Numeric acquisition identity: sealed seed tuples and distinct group evidence
# ---------------------------------------------------------------------------


def test_canonical_seed_groups_bind_the_exact_sealed_numeric_tuples() -> None:
    assert CANONICAL_SEED_GROUPS[QUALIFICATION_SEED_GROUP] == tuple(range(30001, 30017))
    assert CANONICAL_SEED_GROUPS["matrix_a"] == tuple(range(31001, 31017))
    assert CANONICAL_SEED_GROUPS["matrix_b"] == tuple(range(32001, 32017))
    assert CANONICAL_SEED_GROUPS["matrix_c"] == tuple(range(33001, 33017))
    seen = [seed for seeds in CANONICAL_SEED_GROUPS.values() for seed in seeds]
    assert len(set(seen)) == len(seen) == 64


def test_acquisition_key_binds_and_round_trips_its_sealed_seed_tuple() -> None:
    key = _acquisition_key(training_rp=1.10, seed_group_id="matrix_b")
    assert key.seeds == tuple(range(32001, 32017))
    assert AcquisitionKey.from_dict(key.to_dict()) == key
    other = _acquisition_key(training_rp=1.10, seed_group_id="matrix_c")
    assert key.content_sha256 != other.content_sha256


def test_acquisition_key_rejects_another_groups_seed_tuple() -> None:
    with pytest.raises(ValueError, match="seed"):
        _acquisition_key(
            training_rp=1.0,
            seed_group_id="matrix_a",
            seeds=canonical_seeds("matrix_b"),
        )


def test_acquisition_key_rejects_qualification_seeds_in_a_matrix_group() -> None:
    with pytest.raises(ValueError, match="seed"):
        _acquisition_key(
            training_rp=1.0,
            seed_group_id="matrix_a",
            seeds=canonical_seeds(QUALIFICATION_SEED_GROUP),
        )


def test_acquisition_key_payload_must_carry_its_seed_tuple() -> None:
    payload = _acquisition_key(training_rp=1.0, seed_group_id="matrix_a").to_dict()
    payload.pop("seeds")
    with pytest.raises(ValueError, match="seed"):
        AcquisitionKey.from_dict(payload)


def test_shared_evidence_rejects_a_seed_tuple_outside_its_group() -> None:
    with pytest.raises(ValueError, match="seed"):
        SharedEvidenceRef(
            source_sha256=SOURCE_SHA256,
            manifest_sha256=MANIFEST_SHA256,
            acquisition_path="/a",
            acquisition_sha256=_digest("a"),
            trajectory_credit_acquisition_sha256=_digest("t"),
            credit_ledger_sha256=_digest("c"),
            compiler_ledger_sha256=_digest("g"),
            policy_contract_sha256=_digest("p"),
            native_receipts_sha256=_digest("n"),
            training_rp=1.0,
            seed_group_id="matrix_a",
            seeds=canonical_seeds("matrix_c"),
        )


def test_matrix_plan_rejects_one_acquisition_artifact_reused_by_two_seed_groups() -> (
    None
):
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    reused = _shared_evidence(
        training_rp=acquisitions[0].training_rp,
        seed_group_id=acquisitions[0].seed_group_id,
    )
    for index, cell in enumerate(cells):
        if cell.cell_key.acquisition_key == acquisitions[1]:
            cells[index] = _cell_spec(
                acquisition=acquisitions[1],
                arm_id=cell.cell_key.arm_id,
                shared_evidence=replace(
                    reused,
                    seed_group_id=acquisitions[1].seed_group_id,
                    seeds=canonical_seeds(acquisitions[1].seed_group_id),
                ),
            )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="distinct acquisition"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


def test_matrix_plan_rejects_one_policy_contract_across_both_training_rps() -> None:
    acquisitions = _canonical_acquisitions()
    cells = []
    for acquisition in acquisitions:
        shared = replace(
            _shared_evidence(
                training_rp=acquisition.training_rp,
                seed_group_id=acquisition.seed_group_id,
            ),
            policy_contract_sha256=_digest("one-policy-for-both-rps"),
        )
        for arm_id in ARM_IDS:
            cells.append(
                _cell_spec(
                    acquisition=acquisition, arm_id=arm_id, shared_evidence=shared
                )
            )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="policy"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


# ---------------------------------------------------------------------------
# AdamW configuration identity vs per-cell optimizer identity
# ---------------------------------------------------------------------------


def test_matrix_plan_binds_one_adamw_config_and_eighteen_optimizer_identities() -> None:
    plan = _canonical_matrix_plan()

    assert len({cell.adamw_config_sha256 for cell in plan.cells}) == 1
    assert len({cell.fresh_optimizer_identity_sha256 for cell in plan.cells}) == 18


def test_matrix_plan_rejects_a_second_declared_adamw_config() -> None:
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    shared = _shared_evidence(
        training_rp=acquisitions[0].training_rp,
        seed_group_id=acquisitions[0].seed_group_id,
    )
    cells[0] = _cell_spec(
        acquisition=acquisitions[0],
        arm_id="A",
        shared_evidence=shared,
        adamw_config_sha256=_digest("second-adamw-config"),
    )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="one declared AdamW configuration"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


# ---------------------------------------------------------------------------
# Nested objective byte identity
# ---------------------------------------------------------------------------


def test_matrix_plan_rejects_a_trajectory_objective_that_differs_within_one_group() -> (
    None
):
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    shared = _shared_evidence(
        training_rp=acquisitions[0].training_rp,
        seed_group_id=acquisitions[0].seed_group_id,
    )
    cells[1] = _cell_spec(
        acquisition=acquisitions[0],
        arm_id="B",
        shared_evidence=shared,
        objective_component_hashes=_component_hashes(
            arm_id="B",
            training_rp=acquisitions[0].training_rp,
            seed_group_id=acquisitions[0].seed_group_id,
            trajectory_tag="drifted-trajectory",
        ),
    )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="trajectory objective"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


def test_matrix_plan_rejects_a_compiler_objective_that_differs_between_b_and_c() -> (
    None
):
    acquisitions = _canonical_acquisitions()
    cells = list(_canonical_cells(acquisitions))
    shared = _shared_evidence(
        training_rp=acquisitions[0].training_rp,
        seed_group_id=acquisitions[0].seed_group_id,
    )
    cells[2] = _cell_spec(
        acquisition=acquisitions[0],
        arm_id="C",
        shared_evidence=shared,
        objective_component_hashes=_component_hashes(
            arm_id="C",
            training_rp=acquisitions[0].training_rp,
            seed_group_id=acquisitions[0].seed_group_id,
            compiler_tag="drifted-compiler",
        ),
    )
    edges = tuple(
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    )
    with pytest.raises(ValueError, match="compiler objective"):
        _canonical_matrix_plan(
            acquisitions=acquisitions, cells=tuple(cells), dependency_edges=edges
        )


# ---------------------------------------------------------------------------
# The shared receipt aggregate validator
# ---------------------------------------------------------------------------


def test_validate_matrix_receipts_admits_the_canonical_eighteen_cell_outcome() -> None:
    plan = _canonical_matrix_plan()
    receipts = _canonical_receipts(plan)

    admitted = validate_matrix_receipts(plan, receipts)

    assert len(admitted) == 18
    assert len({receipt.transaction_id for receipt in receipts}) == 18


def test_validate_matrix_receipts_admits_identical_fresh_source_state_digests() -> None:
    """All eighteen cells load the same Source: state digests may coincide."""

    plan = _canonical_matrix_plan()
    shared_state = _digest("identical-fresh-source-state")
    receipts = tuple(
        replace(
            _canonical_cell_receipt(cell, tag=str(index)),
            before_transaction_digest=shared_state,
            after_transaction_digest=shared_state,
        )
        for index, cell in enumerate(plan.cells)
    )

    assert len(validate_matrix_receipts(plan, receipts)) == 18


def test_validate_matrix_receipts_rejects_a_reused_transaction_identity() -> None:
    plan = _canonical_matrix_plan()
    receipts = list(_canonical_receipts(plan))
    receipts[1] = replace(receipts[1], transaction_id=receipts[0].transaction_id)

    with pytest.raises(ValueError, match="independent transaction identity"):
        validate_matrix_receipts(plan, tuple(receipts))


def test_validate_matrix_receipts_rejects_a_c_arm_projecting_another_base_proposal() -> (
    None
):
    plan = _canonical_matrix_plan()
    receipts = list(_canonical_receipts(plan))
    receipts[2] = replace(
        receipts[2], proposal_delta_sha256=_digest("some-other-base-proposal")
    )

    with pytest.raises(ValueError, match="exact admitted arm-B proposal"):
        validate_matrix_receipts(plan, tuple(receipts))


def test_validate_matrix_receipts_rejects_a_receipt_objective_outside_its_plan() -> (
    None
):
    plan = _canonical_matrix_plan()
    receipts = list(_canonical_receipts(plan))
    receipts[0] = replace(
        receipts[0], objective_component_hashes=(("trajectory", _digest("forged")),)
    )

    with pytest.raises(ValueError, match="objective component"):
        validate_matrix_receipts(plan, tuple(receipts))


# ---------------------------------------------------------------------------
# NodeTerminalReceipt: the typed acquisition-node terminal
# ---------------------------------------------------------------------------


def test_node_terminal_persists_typed_specs_and_receipts_and_round_trips() -> None:
    terminal = _node_terminal(
        _acquisition_key(training_rp=1.0, seed_group_id="matrix_a")
    )

    reloaded = NodeTerminalReceipt.from_dict(json.loads(json.dumps(terminal.to_dict())))

    assert reloaded == terminal
    assert reloaded.content_sha256 == terminal.content_sha256
    assert [spec.cell_key.arm_id for spec in reloaded.cell_specs] == ["A", "B", "C"]
    assert [receipt.cell_key.arm_id for receipt in reloaded.cell_receipts] == [
        "A",
        "B",
        "C",
    ]


def test_node_terminal_qualification_phase_binds_five_distinct_c_doses() -> None:
    terminal = _node_terminal(
        _acquisition_key(
            training_rp=1.10,
            seed_group_id=QUALIFICATION_SEED_GROUP,
            phase=PHASE_QUALIFICATION,
        ),
    )

    assert terminal.phase == PHASE_QUALIFICATION
    assert len(terminal.cell_specs) == 5
    assert tuple(spec.learning_rate for spec in terminal.cell_specs) == (
        QUALIFICATION_LEARNING_RATE_RAY
    )


def test_node_terminal_succeeded_status_requires_every_planned_arm() -> None:
    with pytest.raises(ValueError, match="every planned arm"):
        _node_terminal(
            _acquisition_key(training_rp=1.0, seed_group_id="matrix_a"),
            receipt_count=2,
        )


def test_node_terminal_failed_status_requires_a_failure_reason() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        _node_terminal(
            _acquisition_key(training_rp=1.0, seed_group_id="matrix_a"),
            status="failed",
            receipt_count=1,
        )


def test_cell_spec_rejects_qualification_evidence_in_a_matrix_cell() -> None:
    qualification_evidence = _shared_evidence(
        training_rp=1.0, seed_group_id=QUALIFICATION_SEED_GROUP
    )

    with pytest.raises(ValueError, match="seed group"):
        CellSpec(
            **{**_cell_spec_kwargs("A"), "shared_evidence": qualification_evidence}
        )
