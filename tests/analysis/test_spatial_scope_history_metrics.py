from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.analysis.spatial_scope_history.calibration import PrimaryScheduleArtifact
from src.analysis.spatial_scope_history.cohort_ledger import (
    AttemptLedger,
    AttemptRecord,
    CohortImageRecord,
    CohortLedger,
    ExecutionIdentityBundle,
    canonical_json_text,
    dependency_skip_failure_code,
    sha256_file,
    sha256_payload,
)
from src.analysis.spatial_scope_history.merge import (
    COMPACT_OBJECT_SELECTED_TOKEN_SCORE_VERSION_1,
    NormalizedCallPrediction,
    merge_image_predictions,
)
from src.analysis.spatial_scope_history.execution_evidence import (
    TerminalCallOutputBundle,
)
from src.analysis.spatial_scope_history.metrics import (
    BootstrapImageRecord,
    CallMetricRecord,
    MetricAdmittedRequest,
    MetricReadinessIdentity,
    ReferenceMatch,
    ReferenceObject,
    RowMetricRecord,
    _build_image_metric_primitive,
    admit_metric_schedule,
    aggregate_metric_report,
    bootstrap_named_metric_report,
    build_image_metric_primitive,
    exact_reference_match,
    image_clustered_bootstrap,
    _validate_call_against_terminal_bundle,
)
from src.analysis.spatial_scope_history.schedule import (
    DecodeProvenance,
    GridProvenance,
    PRIMARY_ROOT_SEED,
    ResearchSchedule,
    derive_sampling_seed,
    primary_arm_definition,
)
from src.analysis.spatial_scope_history.spatial import SpatialGrid, SpatialGridSpec
from src.common.errors import DataContractError
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
    COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
)
from spatial_scope_history_fixtures import (
    build_test_execution_evidence_with_result,
)


class _SyntheticParseReceipt:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_artifact_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _prediction(
    prediction_id: str,
    box: tuple[float, float, float, float],
    *,
    image_id: str = "1",
    arm: str = "FULL_SINGLE",
    category: str = "person",
    score: float = 0.8,
    call_id: str = "call-00",
    row_index: int = 0,
):
    cell_index = int(call_id.rsplit("-", 1)[-1]) if arm != "FULL_SINGLE" else None
    return _formula_prediction(
        image_id=image_id,
        arm=arm,
        canonical_call_id=call_id,
        canonical_cell_index=cell_index,
        generated_row_index=row_index,
        prediction_id=prediction_id,
        category=category,
        global_bbox_xyxy=box,
        local_bbox_xyxy=None,
        score=score,
        ownership_status="not_applicable",
        owner_cell_index=None,
        spatial_coordinate_receipt=None,
    )


def _spatial_prediction(
    prediction_id: str,
    *,
    cell_index: int,
    coordinate_bins: tuple[int, int, int, int],
    row_index: int,
):
    grid = SpatialGrid.build(source_width=128, source_height=128)
    receipt = grid.plan(
        cell_index=cell_index, variant_mode="tile_reset"
    ).coordinate_receipt(coordinate_bins)
    ownership = grid.ownership(receipt.clipped_global_integer_box)
    assert ownership.owner_cell_index is not None
    return _formula_prediction(
        image_id="1",
        arm="TILE_RESET",
        canonical_call_id=f"call-{cell_index:02d}",
        canonical_cell_index=cell_index,
        generated_row_index=row_index,
        prediction_id=prediction_id,
        category="person",
        global_bbox_xyxy=tuple(float(v) for v in receipt.clipped_global_integer_box),
        local_bbox_xyxy=tuple(float(v) for v in receipt.local_integer_box),
        score=0.8,
        ownership_status=(
            "owned" if ownership.owner_cell_index == cell_index else "non_owning"
        ),
        owner_cell_index=ownership.owner_cell_index,
        spatial_coordinate_receipt=receipt,
    )


def _formula_prediction(**values) -> NormalizedCallPrediction:
    """Build a formula-only prediction without exercising evidence normalization."""

    prediction = object.__new__(NormalizedCallPrediction)
    category = values.pop("category")
    arm_code = values.pop("arm")
    payload = {
        **values,
        "arm": primary_arm_definition(arm_code),
        "normalized_category_name": category,
        "evaluator_category_id": COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[category],
        "official_coco_category_id": COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[category],
        "category_registry_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "score_source": COMPACT_OBJECT_SELECTED_TOKEN_SCORE_VERSION_1,
        "source_width": 128,
        "source_height": 128,
        "execution_evidence": None,
    }
    for field_name, value in payload.items():
        object.__setattr__(prediction, field_name, value)
    return prediction


def _reference(
    reference_id: str,
    box: tuple[float, float, float, float] | None,
    *,
    image_id: str = "1",
    scope: str = "audit_augmented",
    state: str = "accepted",
    category: str = "person",
    candidates: tuple[str, ...] = (),
    owner_cell_index: int | None = None,
    core_interior: bool = False,
) -> ReferenceObject:
    return ReferenceObject(
        image_id=image_id,
        ledger_scope=scope,  # type: ignore[arg-type]
        reference_id=reference_id,
        normalized_category_name=category,
        evaluator_category_id=COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[category],
        official_coco_category_id=COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME[category],
        category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        source_canvas_bbox_xyxy=box,
        state=state,  # type: ignore[arg-type]
        provenance=(
            "official_annotation" if scope == "official_annotation" else "test"
        ),
        candidate_category_names=candidates,
        owner_cell_index=owner_cell_index,
        core_interior_for_mask_harm=core_interior,
    )


def _primitive(
    *,
    arm: str,
    predictions=(),
    references=(),
    image_id: str = "1",
    call_records: tuple[CallMetricRecord, ...] | None = None,
    row_records: tuple[RowMetricRecord, ...] | None = None,
    configured_call_budget: int | None = None,
    density_tags: tuple[str, ...] = (),
):
    merge = merge_image_predictions(
        image_id=image_id,
        arm_identifier=arm,
        predictions=predictions,
        source_width=128,
        source_height=128,
    )
    if call_records is None or row_records is None:
        grouped: dict[str, list] = {}
        for prediction in predictions:
            grouped.setdefault(prediction.canonical_call_id, []).append(prediction)
        if not grouped:
            grouped = {"call-00": []}
        generated_rows: list[RowMetricRecord] = []
        generated_calls: list[CallMetricRecord] = []
        for call_index, (call_id, call_predictions) in enumerate(
            sorted(grouped.items())
        ):
            matched = exact_reference_match(call_predictions, references)
            generated_rows.extend(
                RowMetricRecord(
                    call_id,
                    prediction.generated_row_index,
                    prediction.prediction_id,
                    "parsed",
                    "valid",
                    prediction.ownership_status,
                    tuple(
                        match.reference_id
                        for match in matched.matches
                        if match.prediction_id == prediction.prediction_id
                    ),
                )
                for prediction in call_predictions
            )
            generated_calls.append(
                CallMetricRecord(
                    canonical_call_id=call_id,
                    canonical_cell_index=None,
                    sampling_seed=call_index,
                    raw_any_call_matched_reference_ids=(matched.matched_reference_ids),
                    raw_owning_call_matched_reference_ids=(
                        matched.matched_reference_ids
                    ),
                    valid_prediction_ids=tuple(
                        sorted(item.prediction_id for item in call_predictions)
                    ),
                    attempted_row_count=len(call_predictions),
                    malformed_row_count=0,
                    invalid_row_count=0,
                    non_owning_prediction_count=sum(
                        item.ownership_status == "non_owning"
                        for item in call_predictions
                    ),
                    natural_closure_count=1,
                    controller_cap_count=0,
                    token_cap_count=0,
                    error_count=0,
                    prompt_token_count=0,
                    image_token_count=0,
                    generated_token_count=0,
                    wall_time_seconds=0.0,
                    peak_device_memory_bytes=0,
                )
            )
        call_records = tuple(generated_calls)
        row_records = tuple(generated_rows)
    assert call_records is not None and row_records is not None
    return _build_image_metric_primitive(
        merge_result=merge,
        references=references,
        ledger_scope="audit_augmented",
        configured_call_budget=(
            len(call_records)
            if configured_call_budget is None
            else configured_call_budget
        ),
        density_tags=density_tags,
        call_records=call_records,
        row_records=row_records,
    )


def _empty_call(
    call_id: str,
    *,
    cell_index: int | None = None,
    sampling_seed: int = 0,
    attempted_row_count: int = 0,
    malformed_row_count: int = 0,
    invalid_row_count: int = 0,
    natural_closure_count: int = 1,
    controller_cap_count: int = 0,
    token_cap_count: int = 0,
    error_count: int = 0,
    peak_device_memory_bytes: int = 0,
    attempt_status: str = "completed",
    source_image_sha256: str | None = None,
    image_frozen_order: int | None = None,
    terminal_attempt_output_artifact_sha256: str | None = None,
    terminal_attempt_output_artifact_path: str | None = None,
    terminal_attempt_failure_code: str | None = None,
) -> CallMetricRecord:
    return CallMetricRecord(
        canonical_call_id=call_id,
        canonical_cell_index=cell_index,
        sampling_seed=sampling_seed,
        raw_any_call_matched_reference_ids=(),
        raw_owning_call_matched_reference_ids=(),
        valid_prediction_ids=(),
        attempted_row_count=attempted_row_count,
        malformed_row_count=malformed_row_count,
        invalid_row_count=invalid_row_count,
        non_owning_prediction_count=0,
        natural_closure_count=natural_closure_count,
        controller_cap_count=controller_cap_count,
        token_cap_count=token_cap_count,
        error_count=error_count,
        prompt_token_count=0,
        image_token_count=0,
        generated_token_count=0,
        wall_time_seconds=0.0,
        peak_device_memory_bytes=peak_device_memory_bytes,
        attempt_status=attempt_status,  # type: ignore[arg-type]
        source_image_sha256=source_image_sha256,
        image_frozen_order=image_frozen_order,
        terminal_attempt_output_artifact_sha256=(
            terminal_attempt_output_artifact_sha256
        ),
        terminal_attempt_output_artifact_path=terminal_attempt_output_artifact_path,
        terminal_attempt_failure_code=terminal_attempt_failure_code,
    )


def _empty_cell_calls(
    prefix: str, *, seed_offset: int = 0, count: int = 16
) -> tuple[CallMetricRecord, ...]:
    return tuple(
        _empty_call(
            f"{prefix}-{cell_index:02d}",
            cell_index=cell_index,
            sampling_seed=seed_offset + cell_index,
        )
        for cell_index in range(count)
    )


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


def _write_failure_attempt_ledger(
    *,
    root: Path,
    schedule: ResearchSchedule,
) -> AttemptLedger:
    schedule_sha256 = schedule.fingerprint
    dependencies = {
        dependency.request_id: dependency
        for dependency in schedule.attempt_dependencies
    }
    attempts: list[AttemptRecord] = []
    status_by_request_id: dict[str, str] = {}
    for request in schedule.requests:
        dependency = dependencies[request.request_id]
        predecessor_status = (
            status_by_request_id.get(request.predecessor_request_id)
            if request.predecessor_request_id is not None
            else None
        )
        if predecessor_status is None:
            status = "failed"
            failure_code = "fixture_execution_failure"
            expected_state = request.initial_cumulative_state_sha256
        else:
            status = "skipped"
            failure_code = dependency_skip_failure_code(
                predecessor_request_id=request.predecessor_request_id,
                predecessor_status=predecessor_status,  # type: ignore[arg-type]
            )
            expected_state = None
        terminal_bundle = TerminalCallOutputBundle.build(
            scheduled_request=request,
            attempt_status=status,
            execution_evidence=None,
            decode_result=None,
            parse_score_receipts=(),
            row_diagnostics=(),
            call_diagnostics=_terminal_call_diagnostics(
                error_count=1,
                natural_closure_count=0,
            ),
            stop_reason=None,
            failure_code=failure_code,
            cumulative_state_product=None,
        )
        terminal_path = root / f"{request.schedule_index}.json"
        terminal_path.parent.mkdir(parents=True, exist_ok=True)
        terminal_path.write_bytes(terminal_bundle.to_artifact_bytes())
        attempts.append(
            AttemptRecord(
                run_id=schedule.identity.run_id,
                schedule_sha256=schedule_sha256,
                request_id=request.request_id,
                physical_batch_plan_sha256=(dependency.physical_batch_plan_sha256),
                physical_batch_sha256=dependency.physical_batch_sha256,
                physical_batch_index=dependency.physical_batch_index,
                attempt_status=status,  # type: ignore[arg-type]
                started_at_utc="2026-07-13T00:00:00Z",
                finished_at_utc="2026-07-13T00:00:01Z",
                execution_identity=schedule.identity.execution_identity,
                output_artifact_sha256=terminal_bundle.bundle_sha256,
                output_artifact_path=str(terminal_path),
                failure_code=failure_code,
                expected_cumulative_state_sha256=expected_state,
            )
        )
        status_by_request_id[request.request_id] = status
    return AttemptLedger(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule_sha256,
        execution_identity=schedule.identity.execution_identity,
        records=tuple(attempts),
    )


def _metric_admission_fixture(tmp_path: Path):
    records = tuple(
        CohortImageRecord(
            image_id=image_index,
            frozen_order=image_index,
            source_row_index=image_index,
            image_path=f"/fixture/image-{image_index}.png",
            image_sha256=_digest(f"image-{image_index}"),
            source_width=512,
            source_height=512,
            raw_width=512,
            raw_height=512,
            source_row_sha256=_digest(f"row-{image_index}"),
            source_dataset_sha256=_digest("dataset"),
            raw_annotation_sha256=_digest("annotations"),
            noncrowd_annotated_object_count=1,
            annotated_person_count=1,
            annotated_food_tableware_count=0,
            source_crowd_annotation_count=0,
            cohort_memberships=("val200",),
            density_tags=(),
        )
        for image_index in range(4)
    )
    cohort = CohortLedger(
        cohort_id="metric-admission-fixture",
        full_name="Metric Admission Fixture",
        operational_meaning="Four images make the sixty-five-call matrix batch-even.",
        records=records,
    )
    grid_spec = SpatialGridSpec()
    readiness_root = tmp_path / "readiness-v2"
    readiness, _ = _write_readiness_fixture(
        root=readiness_root,
        cohort=cohort,
        grid_spec=grid_spec,
    )
    execution_identity = ExecutionIdentityBundle(
        code_sha256=_digest("code"),
        config_sha256=_digest("config"),
        ledger_sha256=readiness.readiness_seal_sha256,
        runtime_sha256=_digest("runtime"),
    )
    grid_provenance = GridProvenance(
        canonical_spatial_spec_sha256=grid_spec.fingerprint,
        canonical_spatial_receipt_contract_sha256=_digest("grid-receipt-contract"),
    )
    decode = DecodeProvenance(
        temperature=0.4,
        canonical_generation_policy_sha256=_digest("generation-policy"),
        sampled_runtime_attestation_sha256=_digest("sampler-attestation"),
    )
    schedule = ResearchSchedule.build_primary(
        unit_id="metric-admission-fixture",
        run_id="metric-admission-run",
        cohort=cohort,
        root_seed=PRIMARY_ROOT_SEED,
        decode=decode,
        execution_identity=execution_identity,
        grid=grid_provenance,
    )
    attempt_ledger = _write_failure_attempt_ledger(
        root=tmp_path / "terminal",
        schedule=schedule,
    )
    grids = {
        str(record.image_id): SpatialGrid.build(
            source_width=record.source_width,
            source_height=record.source_height,
            spec=grid_spec,
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
    return admission, schedule, cohort, readiness, attempt_ledger, grids


def _terminal_call_diagnostics(
    *,
    error_count: int,
    natural_closure_count: int,
) -> dict[str, object]:
    return {
        "attempted_row_count": 0,
        "controller_cap_count": 0,
        "error_count": error_count,
        "generated_token_count": 0,
        "image_token_count": 0,
        "invalid_row_count": 0,
        "malformed_row_count": 0,
        "natural_closure_count": natural_closure_count,
        "non_owning_prediction_count": 0,
        "peak_device_memory_bytes": 0,
        "prompt_token_count": 0,
        "token_cap_count": 0,
        "valid_prediction_ids": [],
        "wall_time_seconds": 0.0,
    }


def _write_readiness_fixture(
    *,
    root: Path,
    cohort: CohortLedger,
    grid_spec: SpatialGridSpec,
    cohort_artifact_name: str = "cohort-manifest.jsonl",
) -> tuple[MetricReadinessIdentity, str]:
    root.mkdir(parents=True)
    category_source = (
        Path(__file__).parents[2]
        / "research/investigations/qwen3-vl-dense-enumeration/experiments"
        / "2026-07-13-spatial-scope-history-disentanglement"
        / "coco-80-review-ontology-v1.json"
    )
    first_image = cohort.records[0]
    official_individual = {
        "coco_80_category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "evaluator_local_category_id": COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[
            "person"
        ],
        "geometry": {"clipped_source_corners_xyxy": [16, 16, 64, 64]},
        "image_id": first_image.image_id,
        "normalized_category_name": "person",
        "object_or_region_identifier": "coco-ann:1",
        "official_coco_category_id": COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["person"],
        "source_image_sha256": first_image.image_sha256,
    }
    audit_reference = {
        "adjudication_identifier": "fixture-adjudication:0",
        "category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "evaluator_local_category_id": COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[
            "person"
        ],
        "final_state": "accepted",
        "image_id": first_image.image_id,
        "image_sha256": first_image.image_sha256,
        "normalized_category_name": "person",
        "object_identifier": "core-object",
        "official_coco_category_id": COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["person"],
        "provenance": "test",
        "source_canvas_box_xyxy": [16, 16, 64, 64],
    }
    adjudication = {
        "adjudication_identifier": "fixture-adjudication:0",
        "candidate_categories": [],
        "final_state": "accepted",
    }
    cohort_artifact = cohort.to_jsonl_bytes()
    artifacts = {
        cohort_artifact_name: cohort_artifact,
        "coco-80-category-namespace.json": category_source.read_bytes(),
        "official-individual-ledger.jsonl": (
            canonical_json_text(official_individual) + "\n"
        ).encode("utf-8"),
        "official-crowd-ignore-ledger.jsonl": b"",
        "audit-augmented-ledger.jsonl": (
            canonical_json_text(audit_reference) + "\n"
        ).encode("utf-8"),
        "adjudication.jsonl": (canonical_json_text(adjudication) + "\n").encode(
            "utf-8"
        ),
    }
    for name, payload in artifacts.items():
        (root / name).write_bytes(payload)
    annotation_digests = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in artifacts.items()
        if name not in {"adjudication.jsonl", "audit-augmented-ledger.jsonl"}
    }
    annotation_seal = {"artifact_digests": annotation_digests}
    final_seal = {
        "artifact_digests": {
            "adjudication.jsonl": hashlib.sha256(
                artifacts["adjudication.jsonl"]
            ).hexdigest(),
            "audit-augmented-ledger.jsonl": hashlib.sha256(
                artifacts["audit-augmented-ledger.jsonl"]
            ).hexdigest()
        },
        "category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "source_digests": {
            "official_crowd_ledger_jsonl": annotation_digests[
                "official-crowd-ignore-ledger.jsonl"
            ],
            "official_individual_ledger_jsonl": annotation_digests[
                "official-individual-ledger.jsonl"
            ],
        },
    }
    (root / "annotation-derived-cohort-seal.json").write_text(
        canonical_json_text(annotation_seal) + "\n",
        encoding="utf-8",
    )
    (root / "ledger-seal.json").write_text(
        canonical_json_text(final_seal) + "\n",
        encoding="utf-8",
    )
    readiness = MetricReadinessIdentity.from_verified_artifacts(
        readiness_root=root,
        spatial_grid_spec=grid_spec,
        cohort_artifact_name=cohort_artifact_name,
        cohort_artifact_sha256=hashlib.sha256(cohort_artifact).hexdigest(),
    )
    return readiness, readiness.reference_ledger_sha256


def _failed_call_for_request(request, admitted_request) -> CallMetricRecord:
    return _empty_call(
        request.request_id,
        cell_index=request.cell_index,
        sampling_seed=request.sampling_seed,
        natural_closure_count=0,
        error_count=1,
        attempt_status="failed",
        source_image_sha256=request.image_sha256,
        image_frozen_order=request.image_frozen_order,
        terminal_attempt_failure_code="fixture_execution_failure",
        terminal_attempt_output_artifact_sha256=(
            admitted_request.output_artifact_sha256
        ),
        terminal_attempt_output_artifact_path=admitted_request.output_artifact_path,
    )


def test_readiness_uses_schedule_selected_alternate_sealed_cohort_for_references(
    tmp_path: Path,
) -> None:
    _, _, cohort, _, _, _ = _metric_admission_fixture(tmp_path / "base")
    readiness_root = tmp_path / "alternate-readiness"
    readiness, _ = _write_readiness_fixture(
        root=readiness_root,
        cohort=cohort,
        grid_spec=SpatialGridSpec(),
        cohort_artifact_name="dense-union-51-manifest.jsonl",
    )

    assert readiness.cohort_artifact_name == "dense-union-51-manifest.jsonl"
    assert readiness.cohort_artifact_sha256 == sha256_file(
        readiness_root / "dense-union-51-manifest.jsonl"
    )
    references = readiness.reference_objects(
        image_id=str(cohort.records[0].image_id),
        ledger_scope="official_annotation",
    )
    assert tuple(reference.reference_id for reference in references) == ("coco-ann:1",)


def test_readiness_rejects_unsealed_schedule_selected_cohort_name(
    tmp_path: Path,
) -> None:
    _, _, cohort, _, _, _ = _metric_admission_fixture(tmp_path / "base")
    readiness_root = tmp_path / "canonical-readiness"
    _write_readiness_fixture(
        root=readiness_root,
        cohort=cohort,
        grid_spec=SpatialGridSpec(),
    )

    with pytest.raises(DataContractError, match="omits required artifact"):
        MetricReadinessIdentity.from_verified_artifacts(
            readiness_root=readiness_root,
            spatial_grid_spec=SpatialGridSpec(),
            cohort_artifact_name="dense-union-51-manifest.jsonl",
            cohort_artifact_sha256=_digest("unsealed-cohort"),
        )


def test_readiness_rejects_wrong_schedule_selected_cohort_digest(
    tmp_path: Path,
) -> None:
    _, _, cohort, _, _, _ = _metric_admission_fixture(tmp_path / "base")
    readiness_root = tmp_path / "canonical-readiness"
    _write_readiness_fixture(
        root=readiness_root,
        cohort=cohort,
        grid_spec=SpatialGridSpec(),
    )

    with pytest.raises(DataContractError, match="schedule-selected cohort digest"):
        MetricReadinessIdentity.from_verified_artifacts(
            readiness_root=readiness_root,
            spatial_grid_spec=SpatialGridSpec(),
            cohort_artifact_name="cohort-manifest.jsonl",
            cohort_artifact_sha256=_digest("wrong-cohort-bytes"),
        )


def test_readiness_rejects_final_source_digest_detached_from_annotation_seal(
    tmp_path: Path,
) -> None:
    _, _, cohort, _, _, _ = _metric_admission_fixture(tmp_path / "base")
    readiness_root = tmp_path / "detached-readiness"
    _write_readiness_fixture(
        root=readiness_root,
        cohort=cohort,
        grid_spec=SpatialGridSpec(),
    )
    final_seal_path = readiness_root / "ledger-seal.json"
    final_seal = json.loads(final_seal_path.read_text(encoding="utf-8"))
    final_seal["source_digests"]["official_individual_ledger_jsonl"] = _digest(
        "detached-official-individual-ledger"
    )
    final_seal_path.write_text(
        canonical_json_text(final_seal) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match="annotation-sealed official ledgers"):
        MetricReadinessIdentity.from_verified_artifacts(
            readiness_root=readiness_root,
            spatial_grid_spec=SpatialGridSpec(),
            cohort_artifact_name="cohort-manifest.jsonl",
            cohort_artifact_sha256=sha256_file(
                readiness_root / "cohort-manifest.jsonl"
            ),
        )


def test_equal_cardinality_equal_iou_uses_exact_lexicographic_assignment() -> None:
    predictions = (
        _prediction("prediction-a", (0, 0, 10, 10)),
        _prediction("prediction-b", (0, 0, 10, 10), row_index=1),
    )
    references = (
        _reference("reference-a", (0, 0, 10, 10)),
        _reference("reference-b", (0, 0, 10, 10)),
    )
    result = exact_reference_match(predictions, references)
    assert [(match.prediction_id, match.reference_id) for match in result.matches] == [
        ("prediction-a", "reference-a"),
        ("prediction-b", "reference-b"),
    ]
    assert result.solver.startswith("custom-successive-shortest")
    assert result.objective_tolerance == 0.0


def test_matcher_enforces_cardinality_before_total_intersection_over_union() -> None:
    predictions = (
        _prediction("wide", (0, 0, 15, 10)),
        _prediction("left", (0, 0, 10, 10), row_index=1),
    )
    references = (
        _reference("left-reference", (0, 0, 10, 10)),
        _reference("right-reference", (5, 0, 15, 10)),
    )
    result = exact_reference_match(predictions, references)
    assert len(result.matches) == 2
    assert {(item.prediction_id, item.reference_id) for item in result.matches} == {
        ("left", "left-reference"),
        ("wide", "right-reference"),
    }


def test_duplicate_predictions_only_recall_one_reference() -> None:
    predictions = (
        _prediction("duplicate-a", (0, 0, 10, 10)),
        _prediction("duplicate-b", (0, 0, 10, 10), row_index=1),
    )
    result = exact_reference_match(
        predictions, (_reference("one-object", (0, 0, 10, 10)),)
    )
    assert result.matched_reference_ids == ("one-object",)
    assert len(result.unmatched_prediction_ids) == 1


def test_primitive_preserves_merge_destroyed_reference_transition() -> None:
    predictions = (
        _prediction("keeper", (0, 0, 10, 10), score=0.9),
        _prediction("suppressed", (1, 0, 11, 10), score=0.8, row_index=1),
    )
    references = (
        _reference("object-a", (0, 0, 10, 10)),
        _reference("object-b", (1, 0, 11, 10)),
    )
    primitive = _primitive(
        arm="FULL_SINGLE", predictions=predictions, references=references
    )
    assert primitive.pre_merge_match.matched_reference_ids == (
        "object-a",
        "object-b",
    )
    assert primitive.post_merge_match.matched_reference_ids == ("object-a",)
    assert primitive.merge_created_reference_ids == ()
    assert primitive.merge_destroyed_reference_ids == ("object-b",)


def test_spatial_call_separates_raw_any_from_raw_owning_matches() -> None:
    owning = _spatial_prediction(
        "owning", cell_index=0, coordinate_bins=(0, 0, 250, 250), row_index=0
    )
    non_owning = _spatial_prediction(
        "non-owning",
        cell_index=0,
        coordinate_bins=(500, 0, 999, 500),
        row_index=1,
    )
    references = (
        _reference("cell-0-object", (0, 0, 16, 16), owner_cell_index=0),
        _reference("cell-1-object", (32, 0, 64, 32), owner_cell_index=1),
    )
    call_id = owning.canonical_call_id
    assert non_owning.canonical_call_id == call_id
    call = CallMetricRecord(
        canonical_call_id=call_id,
        canonical_cell_index=0,
        sampling_seed=17,
        raw_any_call_matched_reference_ids=("cell-0-object", "cell-1-object"),
        raw_owning_call_matched_reference_ids=("cell-0-object",),
        valid_prediction_ids=("non-owning", "owning"),
        attempted_row_count=2,
        malformed_row_count=0,
        invalid_row_count=0,
        non_owning_prediction_count=1,
        natural_closure_count=1,
        controller_cap_count=0,
        token_cap_count=0,
        error_count=0,
        prompt_token_count=0,
        image_token_count=0,
        generated_token_count=0,
        wall_time_seconds=0.0,
        peak_device_memory_bytes=0,
    )
    rows = (
        RowMetricRecord(
            call_id,
            0,
            "owning",
            "parsed",
            "valid",
            "owned",
            ("cell-0-object",),
        ),
        RowMetricRecord(
            call_id,
            1,
            "non-owning",
            "parsed",
            "valid",
            "non_owning",
            ("cell-1-object",),
        ),
    )
    primitive = _primitive(
        arm="TILE_RESET",
        predictions=(owning, non_owning),
        references=references,
        call_records=(call,),
        row_records=rows,
        configured_call_budget=1,
    )
    assert primitive.raw_any_call_union_match.matched_reference_ids == (
        "cell-0-object",
        "cell-1-object",
    )
    assert primitive.raw_owning_match.matched_reference_ids == ("cell-0-object",)
    assert primitive.cell_records[0].raw_owning_call_matched_reference_ids == (
        "cell-0-object",
    )


def test_ledger_separation_and_crowd_ignore_precedes_uncertainty_ignore() -> None:
    prediction = _prediction("unmatched", (20, 20, 30, 30))
    references = (
        _reference("accepted", (0, 0, 10, 10)),
        _reference("crowd", (20, 20, 30, 30), state="crowd"),
        _reference(
            "uncertain",
            (20, 20, 30, 30),
            state="ambiguous",
            candidates=("person",),
        ),
        ReferenceObject(
            image_id="1",
            ledger_scope="audit_augmented",
            reference_id="out-of-scope",
            normalized_category_name=None,
            evaluator_category_id=None,
            official_coco_category_id=None,
            category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
            source_canvas_bbox_xyxy=None,
            state="out_of_scope",
            provenance="test",
        ),
    )
    primitive = _primitive(
        arm="FULL_SINGLE", predictions=(prediction,), references=references
    )
    assert primitive.crowd_ignored_prediction_ids == ("unmatched",)
    assert primitive.uncertainty_ignored_prediction_ids == ()
    assert primitive.unmatched_valid_prediction_ids == ()
    assert primitive.accepted_reference_ids == ("accepted",)

    official = _reference("coco-ann:123", (0, 0, 10, 10), scope="official_annotation")
    merge = merge_image_predictions(
        image_id="1", arm_identifier="FULL_SINGLE", predictions=(prediction,)
    )
    with pytest.raises(DataContractError, match="ledger_mixed"):
        _build_image_metric_primitive(
            merge_result=merge,
            references=(*references, official),
            ledger_scope="audit_augmented",
            configured_call_budget=1,
            call_records=(),
            row_records=(),
        )


def test_category_unresolved_ambiguous_reference_requires_candidates() -> None:
    with pytest.raises(DataContractError, match="ambiguous_candidate_categories"):
        ReferenceObject(
            image_id="1",
            ledger_scope="audit_augmented",
            reference_id="ambiguous-without-candidates",
            normalized_category_name=None,
            evaluator_category_id=None,
            official_coco_category_id=None,
            category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
            source_canvas_bbox_xyxy=(10, 10, 30, 30),
            state="ambiguous",
            provenance="test",
        )


def test_ambiguous_reference_ignores_only_recorded_candidate_categories() -> None:
    reference = ReferenceObject(
        image_id="1",
        ledger_scope="audit_augmented",
        reference_id="bus-or-train",
        normalized_category_name=None,
        evaluator_category_id=None,
        official_coco_category_id=None,
        category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        source_canvas_bbox_xyxy=(20, 20, 40, 40),
        state="ambiguous",
        provenance="test",
        candidate_category_names=("bus", "train"),
    )
    bus = _prediction("bus-prediction", (20, 20, 40, 40), category="bus")
    car = _prediction(
        "car-prediction",
        (20, 20, 40, 40),
        category="car",
        row_index=1,
    )
    primitive = _primitive(
        arm="FULL_SINGLE",
        predictions=(bus, car),
        references=(reference,),
    )
    assert primitive.uncertainty_ignored_prediction_ids == ("bus-prediction",)
    assert primitive.unmatched_valid_prediction_ids == ("car-prediction",)


def test_out_of_scope_reference_never_ignores_coco80_prediction() -> None:
    prediction = _prediction("bowl-prediction", (20, 20, 40, 40), category="bowl")
    reference = ReferenceObject(
        image_id="1",
        ledger_scope="audit_augmented",
        reference_id="visible-plate-outside-coco80",
        normalized_category_name=None,
        evaluator_category_id=None,
        official_coco_category_id=None,
        category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        source_canvas_bbox_xyxy=(20, 20, 40, 40),
        state="out_of_scope",
        provenance="test",
    )
    primitive = _primitive(
        arm="FULL_SINGLE",
        predictions=(prediction,),
        references=(reference,),
    )
    assert primitive.uncertainty_ignored_prediction_ids == ()
    assert primitive.crowd_ignored_prediction_ids == ()
    assert primitive.unmatched_valid_prediction_ids == ("bowl-prediction",)


def test_actual_readiness_v2_reconstructs_all_official_and_audit_references() -> None:
    readiness_root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-13-spatial-scope-history-disentanglement/readiness-v2"
    )
    if not readiness_root.is_dir():
        pytest.skip("sealed Dense-Union-51 readiness-v2 artifacts are unavailable")
    readiness = MetricReadinessIdentity.from_verified_artifacts(
        readiness_root=readiness_root,
        spatial_grid_spec=SpatialGridSpec(),
        cohort_artifact_name="cohort-manifest.jsonl",
        cohort_artifact_sha256=sha256_file(readiness_root / "cohort-manifest.jsonl"),
    )
    cohort = CohortLedger.from_jsonl_bytes(
        (readiness_root / "cohort-manifest.jsonl").read_bytes()
    )
    assert len(cohort.records) == 200
    for record in cohort.records:
        image_id = str(record.image_id)
        readiness.reference_objects(
            image_id=image_id,
            ledger_scope="official_annotation",
        )
        readiness.reference_objects(
            image_id=image_id,
            ledger_scope="audit_augmented",
        )

    ambiguous = next(
        reference
        for reference in readiness.reference_objects(
            image_id="16228",
            ledger_scope="audit_augmented",
        )
        if reference.reference_id.endswith(":16228:0010")
    )
    assert ambiguous.state == "ambiguous"
    assert ambiguous.candidate_category_names == ("bus", "train")

    out_of_scope = tuple(
        reference
        for reference in readiness.reference_objects(
            image_id="17714",
            ledger_scope="audit_augmented",
        )
        if reference.state == "out_of_scope"
    )
    assert tuple(reference.source_canvas_bbox_xyxy for reference in out_of_scope) == (
        (224.0, 384.0, 655.0, 843.0),
        (671.0, 402.0, 1000.0, 762.0),
    )


def test_actual_dense_union_51_schedule_admits_selected_sealed_cohort(
    tmp_path: Path,
) -> None:
    experiment_root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-13-spatial-scope-history-disentanglement"
    )
    readiness_root = experiment_root / "readiness-v2"
    schedule_path = (
        experiment_root
        / "schedules/dense-union-51-primary-after-contract-recovery.json"
    )
    if not readiness_root.is_dir() or not schedule_path.is_file():
        pytest.skip("sealed Dense-Union-51 admission artifacts are unavailable")
    schedule_artifact = PrimaryScheduleArtifact.from_artifact_dict(
        json.loads(schedule_path.read_text(encoding="utf-8"))
    )
    assert schedule_artifact.cohort_artifact_name == (
        "dense-union-51-manifest.jsonl"
    )
    cohort_path = readiness_root / schedule_artifact.cohort_artifact_name
    assert sha256_file(cohort_path) == schedule_artifact.cohort_artifact_sha256
    cohort = CohortLedger.from_jsonl_bytes(cohort_path.read_bytes())
    readiness = MetricReadinessIdentity.from_verified_artifacts(
        readiness_root=readiness_root,
        spatial_grid_spec=SpatialGridSpec(),
        cohort_artifact_name=schedule_artifact.cohort_artifact_name,
        cohort_artifact_sha256=schedule_artifact.cohort_artifact_sha256,
    )
    attempt_ledger = _write_failure_attempt_ledger(
        root=tmp_path / "dense-union-51-terminal",
        schedule=schedule_artifact.schedule,
    )
    grids = {
        str(record.image_id): SpatialGrid.build(
            source_width=record.source_width,
            source_height=record.source_height,
        )
        for record in cohort.records
    }

    admission = admit_metric_schedule(
        schedule=schedule_artifact.schedule,
        cohort=cohort,
        readiness=readiness,
        attempt_ledger=attempt_ledger,
        spatial_grids_by_image_id=grids,
    )

    assert len(admission.images) == 51
    assert len(admission.requests) == 3_315
    assert admission.readiness.readiness_seal_sha256 == (
        schedule_artifact.schedule.identity.execution_identity.ledger_sha256
    )
    assert admission.readiness.cohort_artifact_name == (
        schedule_artifact.cohort_artifact_name
    )


def test_zero_denominator_is_not_applicable_not_zero() -> None:
    reference = _reference("object", (0, 0, 10, 10))
    baseline = _primitive(
        arm="FULL_SINGLE",
        predictions=(_prediction("baseline", (0, 0, 10, 10)),),
        references=(reference,),
    )
    candidate = _primitive(
        arm="FULL_BAG_K",
        predictions=(_prediction("candidate", (0, 0, 10, 10), arm="FULL_BAG_K"),),
        references=(reference,),
    )
    report = aggregate_metric_report((candidate,), (baseline,), scope="test")
    rescue = next(
        metric
        for metric in report.metrics
        if metric.metric_name == "post_merge_local_rescue_rate"
    )
    assert rescue.denominator == 0
    assert rescue.estimate is None
    assert rescue.status == "not_applicable"


def test_bootstrap_resamples_paired_image_bundles() -> None:
    records = (
        BootstrapImageRecord("image-a", 1, 1, 0, 1),
        BootstrapImageRecord("image-b", 0, 1, 1, 1),
    )
    report = image_clustered_bootstrap(
        records,
        metric_name="paired-test",
        estimator="paired_rate_difference",
        scope="test",
        threshold=0.5,
        comparator="control",
        replicates=100,
        minimum_applicable_replicates=95,
        synthetic_only=True,
    )
    assert report.applicable_replicates == 100
    assert report.status == "ok"
    assert report.point_estimate.estimate == 0.0
    assert report.point_estimate.numerator == 1
    assert report.point_estimate.denominator == 2
    assert report.point_estimate.comparator_numerator == 1
    assert report.point_estimate.comparator_denominator == 2
    assert report.evidence_status == "synthetic_only"
    assert report.sampling_seed == derive_sampling_seed(
        root_seed=PRIMARY_ROOT_SEED,
        role="image-bootstrap",
        image_id=0,
        cell_or_call_label="replicates-100",
    )
    with pytest.raises(DataContractError, match="synthetic_bootstrap_evidence"):
        report.to_json_dict()
    assert all(len(record.sampled_image_indices) == 2 for record in report.records)
    assert {record.estimate for record in report.records} <= {-1.0, 0.0, 1.0}


def test_primitive_preserves_count_prompt_row_strata_and_arrays() -> None:
    reference = _reference("object", (0, 0, 10, 10), owner_cell_index=0)
    prediction = _prediction("prediction", (0, 0, 10, 10), arm="FULL_BAG_K")
    call_id = prediction.canonical_call_id
    call = CallMetricRecord(
        canonical_call_id=call_id,
        canonical_cell_index=0,
        sampling_seed=7,
        raw_any_call_matched_reference_ids=("object",),
        raw_owning_call_matched_reference_ids=("object",),
        valid_prediction_ids=("prediction",),
        attempted_row_count=3,
        malformed_row_count=1,
        invalid_row_count=1,
        non_owning_prediction_count=0,
        natural_closure_count=1,
        controller_cap_count=0,
        token_cap_count=0,
        error_count=0,
        prompt_token_count=777,
        image_token_count=64,
        generated_token_count=55,
        wall_time_seconds=1.25,
        peak_device_memory_bytes=1234,
    )
    rows = (
        RowMetricRecord(
            call_id,
            0,
            "prediction",
            "parsed",
            "valid",
            "not_applicable",
            ("object",),
        ),
        RowMetricRecord(
            call_id, 1, None, "malformed", "not_applicable", "not_applicable"
        ),
        RowMetricRecord(call_id, 2, None, "parsed", "invalid", "not_applicable"),
    )
    primitive = _primitive(
        arm="FULL_BAG_K",
        predictions=(prediction,),
        references=(reference,),
        configured_call_budget=1,
        density_tags=("people_dense",),
        call_records=(call,),
        row_records=rows,
    )
    payload = primitive.to_json_dict()
    assert payload["category_namespace_sha256"] == COCO_80_CATEGORY_NAMESPACE_SHA256
    assert payload["reference_object_count_stratum_value"] == 1
    assert payload["prompt_token_count_stratum_value"] == 777
    assert payload["attempted_row_count_stratum_value"] == 3
    assert len(payload["cell_records"]) == 1
    assert len(payload["call_records"]) == 1
    assert len(payload["row_records"]) == 3
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_reference_contract_rejects_untraceable_official_records() -> None:
    gapped = _reference("stop-sign", (0, 0, 10, 10), category="stop sign")
    assert gapped.evaluator_category_id == 12
    assert gapped.official_coco_category_id == 13
    assert "category_id" not in gapped.to_json_dict()

    with pytest.raises(DataContractError, match="reference_category"):
        replace(gapped, official_coco_category_id=12)
    with pytest.raises(DataContractError, match="category_namespace"):
        replace(gapped, category_namespace_sha256="0" * 64)

    with pytest.raises(DataContractError, match="official_identifier"):
        ReferenceObject(
            image_id="image-1",
            ledger_scope="official_annotation",
            reference_id="opaque-reference",
            normalized_category_name="person",
            evaluator_category_id=COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME["person"],
            official_coco_category_id=COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["person"],
            category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
            source_canvas_bbox_xyxy=(0, 0, 10, 10),
            state="accepted",
            provenance="official_annotation",
        )
    with pytest.raises(DataContractError, match="official_state"):
        ReferenceObject(
            image_id="image-1",
            ledger_scope="official_annotation",
            reference_id="coco-ann:7",
            normalized_category_name="person",
            evaluator_category_id=COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME["person"],
            official_coco_category_id=COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["person"],
            category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
            source_canvas_bbox_xyxy=(0, 0, 10, 10),
            state="ambiguous",
            provenance="official_annotation",
        )


def test_aggregate_rejects_mixed_candidate_and_wrong_comparator_arms() -> None:
    candidate_a = _primitive(arm="MASK_RESET", image_id="image-a")
    candidate_b = _primitive(arm="TILE_RESET", image_id="image-b")
    baseline_a = _primitive(arm="FULL_SINGLE", image_id="image-a")
    baseline_b = _primitive(arm="FULL_SINGLE", image_id="image-b")
    with pytest.raises(DataContractError, match="mixed_arms"):
        aggregate_metric_report(
            (candidate_a, candidate_b),
            (baseline_a, baseline_b),
            scope="test",
        )

    with pytest.raises(DataContractError, match="pair_contract"):
        aggregate_metric_report(
            (replace(candidate_a, source_width=256),),
            (baseline_a,),
            scope="test",
        )

    with pytest.raises(DataContractError, match="image_universe"):
        aggregate_metric_report(
            (candidate_a,),
            (baseline_b,),
            scope="test",
        )

    wrong_baseline = _primitive(arm="FULL_BAG_K", image_id="image-a")
    with pytest.raises(DataContractError, match="wrong_comparator_arm"):
        aggregate_metric_report((candidate_a,), (wrong_baseline,), scope="test")

    wrong_paired_comparator = _primitive(arm="TILE_RESET", image_id="image-a")
    with pytest.raises(DataContractError, match="wrong_comparator_arm"):
        aggregate_metric_report(
            (candidate_a,),
            (baseline_a,),
            scope="test",
            paired_full_bag=(wrong_paired_comparator,),
        )


def test_owning_seed_evidence_binds_all_cells_requests_and_sampling_seeds() -> None:
    reference = _reference("object", (0, 0, 10, 10), owner_cell_index=0)
    baseline = _primitive(arm="FULL_SINGLE", references=(reference,))
    candidate = _primitive(
        arm="MASK_RESET",
        references=(reference,),
        call_records=_empty_cell_calls("mask"),
        row_records=(),
        configured_call_budget=16,
    )
    full_bag = _primitive(
        arm="FULL_BAG_K",
        references=(reference,),
        call_records=_empty_cell_calls("bag"),
        row_records=(),
        configured_call_budget=16,
    )
    report = aggregate_metric_report(
        (candidate,),
        (baseline,),
        scope="test",
        paired_full_bag=(full_bag,),
    )
    assert len(report.owning_seed_pair_records) == 16
    first_pair = report.owning_seed_pair_records[0]
    assert first_pair.candidate_request_id == "mask-00"
    assert first_pair.full_bag_request_id == "bag-00"
    assert first_pair.sampling_seed == 0
    assert first_pair.owned_reference_ids == ("object",)
    assert first_pair.baseline_missed_reference_ids == ("object",)
    owning_rescue = next(
        metric
        for metric in report.metrics
        if metric.metric_name == "owning_seed_raw_rescue_rate"
    )
    assert (owning_rescue.numerator, owning_rescue.denominator) == (0, 1)

    missing_cell_candidate = _primitive(
        arm="MASK_RESET",
        references=(reference,),
        call_records=_empty_cell_calls("mask", count=15),
        row_records=(),
        configured_call_budget=15,
    )
    with pytest.raises(DataContractError, match="owning_seed_cells"):
        aggregate_metric_report(
            (missing_cell_candidate,),
            (baseline,),
            scope="test",
            paired_full_bag=(full_bag,),
        )

    wrong_seed_bag = _primitive(
        arm="FULL_BAG_K",
        references=(reference,),
        call_records=_empty_cell_calls("bag", seed_offset=1),
        row_records=(),
        configured_call_budget=16,
    )
    with pytest.raises(DataContractError, match="owning_seed_seed"):
        aggregate_metric_report(
            (candidate,),
            (baseline,),
            scope="test",
            paired_full_bag=(wrong_seed_bag,),
        )

    tampered_cells = (
        replace(candidate.cell_records[0], canonical_call_id="not-mask-00"),
        *candidate.cell_records[1:],
    )
    with pytest.raises(DataContractError, match="owning_seed_request"):
        aggregate_metric_report(
            (replace(candidate, cell_records=tampered_cells),),
            (baseline,),
            scope="test",
            paired_full_bag=(full_bag,),
        )

    fabricated_raw_match = replace(
        candidate.raw_owning_match,
        matches=(ReferenceMatch("fabricated", "object", "person", 1.0),),
        unmatched_prediction_ids=(),
        unmatched_reference_ids=(),
    )
    with pytest.raises(DataContractError, match="candidate_raw_owning_reconciliation"):
        aggregate_metric_report(
            (replace(candidate, raw_owning_match=fabricated_raw_match),),
            (baseline,),
            scope="test",
            paired_full_bag=(full_bag,),
        )


def test_owning_seed_evidence_requires_exact_reference_owner_partition() -> None:
    reference = _reference("object", (0, 0, 10, 10))
    baseline = _primitive(arm="FULL_SINGLE", references=(reference,))
    candidate = _primitive(
        arm="MASK_RESET",
        references=(reference,),
        call_records=_empty_cell_calls("mask"),
        row_records=(),
        configured_call_budget=16,
    )
    full_bag = _primitive(
        arm="FULL_BAG_K",
        references=(reference,),
        call_records=_empty_cell_calls("bag"),
        row_records=(),
        configured_call_budget=16,
    )
    with pytest.raises(DataContractError, match="reference_owner_partition"):
        aggregate_metric_report(
            (candidate,),
            (baseline,),
            scope="test",
            paired_full_bag=(full_bag,),
        )


def test_builder_rejects_inconsistent_nested_counts_and_terminal_outcomes() -> None:
    inconsistent_count = _empty_call("call-00", attempted_row_count=1)
    with pytest.raises(DataContractError, match="call_row_count"):
        _primitive(
            arm="FULL_SINGLE",
            call_records=(inconsistent_count,),
            row_records=(),
            configured_call_budget=1,
        )

    missing_terminal = _empty_call("call-00", natural_closure_count=0)
    with pytest.raises(DataContractError, match="terminal_outcome"):
        _primitive(
            arm="FULL_SINGLE",
            call_records=(missing_terminal,),
            row_records=(),
            configured_call_budget=1,
        )

    duplicate_row_indexes = (
        RowMetricRecord(
            "call-00", 0, None, "malformed", "not_applicable", "not_applicable"
        ),
        RowMetricRecord(
            "call-00", 0, None, "malformed", "not_applicable", "not_applicable"
        ),
    )
    duplicate_row_call = _empty_call(
        "call-00", attempted_row_count=2, malformed_row_count=2
    )
    with pytest.raises(DataContractError, match="row_index_reconciliation"):
        _primitive(
            arm="FULL_SINGLE",
            call_records=(duplicate_row_call,),
            row_records=duplicate_row_indexes,
            configured_call_budget=1,
        )

    reference = _reference("object", (0, 0, 10, 10))
    prediction = _prediction("prediction", (0, 0, 10, 10))
    prediction_call_id = prediction.canonical_call_id
    exact_call = replace(
        _empty_call(prediction_call_id, attempted_row_count=1),
        raw_any_call_matched_reference_ids=("object",),
        raw_owning_call_matched_reference_ids=("object",),
        valid_prediction_ids=("prediction",),
    )
    row_without_exact_match = RowMetricRecord(
        prediction_call_id,
        0,
        "prediction",
        "parsed",
        "valid",
        "not_applicable",
    )
    row_with_exact_postrun_match = replace(
        row_without_exact_match, matched_reference_ids=("object",)
    )
    primitive = _primitive(
        arm="FULL_SINGLE",
        predictions=(prediction,),
        references=(reference,),
        call_records=(exact_call,),
        row_records=(row_with_exact_postrun_match,),
        configured_call_budget=1,
    )
    assert primitive.row_records == (row_with_exact_postrun_match,)
    with pytest.raises(DataContractError, match="row_prediction_reconciliation"):
        _primitive(
            arm="FULL_SINGLE",
            predictions=(prediction,),
            references=(reference,),
            call_records=(exact_call,),
            row_records=(row_without_exact_match,),
            configured_call_budget=1,
        )

    multiple_terminals = _empty_call("call-00", controller_cap_count=1)
    with pytest.raises(DataContractError, match="terminal_outcome"):
        _primitive(
            arm="FULL_SINGLE",
            call_records=(multiple_terminals,),
            row_records=(),
            configured_call_budget=1,
        )


def test_aggregate_reports_mask_harm_uncertainty_and_invalid_call_views() -> None:
    core_reference = _reference("core-object", (0, 0, 10, 10), core_interior=True)
    baseline = _primitive(
        arm="FULL_SINGLE",
        predictions=(_prediction("baseline-hit", (0, 0, 10, 10)),),
        references=(core_reference,),
    )
    candidate = _primitive(arm="MASK_RESET", references=(core_reference,))
    mask_report = aggregate_metric_report((candidate,), (baseline,), scope="test")
    mask_harm = next(
        metric
        for metric in mask_report.metrics
        if metric.metric_name == "mask_harm_retention"
    )
    assert (mask_harm.numerator, mask_harm.denominator, mask_harm.estimate) == (
        0,
        1,
        0.0,
    )

    accepted = _reference("accepted", (0, 0, 10, 10))
    uncertain = _reference(
        "uncertain",
        (20, 20, 30, 30),
        state="ambiguous",
        candidates=("person",),
    )
    precision_candidate = _primitive(
        arm="FULL_BAG_K",
        predictions=(
            _prediction("accepted-hit", (0, 0, 10, 10), arm="FULL_BAG_K", row_index=0),
            _prediction(
                "uncertain-hit", (20, 20, 30, 30), arm="FULL_BAG_K", row_index=1
            ),
        ),
        references=(accepted, uncertain),
    )
    precision_baseline = _primitive(arm="FULL_SINGLE", references=(accepted, uncertain))
    precision_report = aggregate_metric_report(
        (precision_candidate,), (precision_baseline,), scope="test"
    )
    precision = next(
        metric
        for metric in precision_report.metrics
        if metric.metric_name == "manual_precision"
    )
    sensitivity = next(
        metric
        for metric in precision_report.metrics
        if metric.metric_name == "manual_precision_uncertainty_ignored_as_unmatched"
    )
    assert precision.estimate == 1.0
    assert sensitivity.estimate == 0.5

    invalid_call = _empty_call("call-00", attempted_row_count=1, malformed_row_count=1)
    invalid_row = RowMetricRecord(
        "call-00", 0, None, "malformed", "not_applicable", "not_applicable"
    )
    invalid_candidate = _primitive(
        arm="MASK_RESET",
        call_records=(invalid_call,),
        row_records=(invalid_row,),
        configured_call_budget=1,
    )
    invalid_baseline = _primitive(arm="FULL_SINGLE")
    invalid_report = aggregate_metric_report(
        (invalid_candidate,), (invalid_baseline,), scope="test"
    )
    invalid_call_rate = next(
        metric
        for metric in invalid_report.metrics
        if metric.metric_name == "invalid_call_rate"
    )
    assert invalid_call_rate.estimate == 1.0


def test_peak_device_memory_aggregates_by_maximum_not_sum() -> None:
    candidates = tuple(
        _primitive(
            arm="MASK_RESET",
            image_id=image_id,
            call_records=(
                _empty_call(
                    f"{image_id}-call",
                    peak_device_memory_bytes=peak_device_memory_bytes,
                ),
            ),
            row_records=(),
            configured_call_budget=1,
        )
        for image_id, peak_device_memory_bytes in (("image-a", 100), ("image-b", 250))
    )
    baselines = tuple(
        _primitive(arm="FULL_SINGLE", image_id=image_id)
        for image_id in ("image-a", "image-b")
    )
    report = aggregate_metric_report(candidates, baselines, scope="test")
    peak_memory = next(
        metric
        for metric in report.metrics
        if metric.metric_name == "realized_budget_peak_device_memory_bytes"
    )
    assert peak_memory.numerator == 250
    assert report.category_namespace_sha256 == COCO_80_CATEGORY_NAMESPACE_SHA256


def test_arbitrary_bootstrap_input_cannot_become_metric_evidence() -> None:
    with pytest.raises(DataContractError, match="arbitrary_bootstrap_evidence"):
        image_clustered_bootstrap(
            (BootstrapImageRecord("image-a", 1, 1),),
            metric_name="unregistered-input",
            estimator="rate",
            scope="test",
            threshold=0.5,
            comparator=None,
        )


def test_metric_bearing_bootstrap_uses_frozen_compact_contract() -> None:
    candidate = _primitive(arm="MASK_RESET")
    baseline = _primitive(arm="FULL_SINGLE")
    report = bootstrap_named_metric_report(
        (candidate,),
        (baseline,),
        metric_name="natural_closure_rate",
        scope="test",
    )
    payload = report.to_json_dict()
    assert report.total_replicates == 10_000
    assert report.minimum_applicable_replicates == 9_500
    assert report.applicable_replicates == 10_000
    assert report.records == ()
    assert report.evidence_status == "metric_bearing"
    assert payload["bootstrap_algorithm"] == "image_clustered_percentile_95_v1"
    assert payload["category_namespace_sha256"] == COCO_80_CATEGORY_NAMESPACE_SHA256
    assert "records" not in payload
    assert payload["sampling_seed"] == derive_sampling_seed(
        root_seed=PRIMARY_ROOT_SEED,
        role="image-bootstrap",
        image_id=0,
        cell_or_call_label="replicates-10000",
    )


def test_metric_admission_rejects_missing_and_extra_terminal_requests(
    tmp_path: Path,
) -> None:
    _, schedule, cohort, readiness, ledger, grids = _metric_admission_fixture(tmp_path)
    missing_ledger = replace(ledger, records=ledger.records[:-1])
    with pytest.raises(DataContractError, match="admission_terminal_universe"):
        admit_metric_schedule(
            schedule=schedule,
            cohort=cohort,
            readiness=readiness,
            attempt_ledger=missing_ledger,
            spatial_grids_by_image_id=grids,
        )

    first_dependency = schedule.attempt_dependencies[0]
    rogue = AttemptRecord(
        run_id=schedule.identity.run_id,
        schedule_sha256=schedule.fingerprint,
        request_id="rogue-request",
        physical_batch_plan_sha256=(first_dependency.physical_batch_plan_sha256),
        physical_batch_sha256=first_dependency.physical_batch_sha256,
        physical_batch_index=first_dependency.physical_batch_index,
        attempt_status="failed",
        started_at_utc="2026-07-13T00:00:00Z",
        finished_at_utc="2026-07-13T00:00:01Z",
        execution_identity=schedule.identity.execution_identity,
        failure_code="rogue",
    )
    extra_ledger = replace(ledger, records=(*ledger.records, rogue))
    with pytest.raises(DataContractError, match="admission_terminal_universe"):
        admit_metric_schedule(
            schedule=schedule,
            cohort=cohort,
            readiness=readiness,
            attempt_ledger=extra_ledger,
            spatial_grids_by_image_id=grids,
        )


def test_metric_admission_rejects_wrong_source_digest_and_rogue_schedule(
    tmp_path: Path,
) -> None:
    _, schedule, cohort, readiness, ledger, grids = _metric_admission_fixture(tmp_path)
    altered_records = (
        replace(cohort.records[0], image_sha256=_digest("wrong-source")),
        *cohort.records[1:],
    )
    altered_cohort = replace(cohort, records=altered_records)
    with pytest.raises(DataContractError, match="cohort_schedule"):
        admit_metric_schedule(
            schedule=schedule,
            cohort=altered_cohort,
            readiness=readiness,
            attempt_ledger=ledger,
            spatial_grids_by_image_id=grids,
        )

    rogue_schedule = ResearchSchedule.build_primary(
        unit_id=schedule.identity.unit_id,
        run_id="rogue-run",
        cohort=cohort,
        root_seed=PRIMARY_ROOT_SEED,
        decode=schedule.identity.decode,
        execution_identity=schedule.identity.execution_identity,
        grid=schedule.identity.grid,
    )
    with pytest.raises(DataContractError, match="attempt_schedule"):
        admit_metric_schedule(
            schedule=rogue_schedule,
            cohort=cohort,
            readiness=readiness,
            attempt_ledger=ledger,
            spatial_grids_by_image_id=grids,
        )


def test_metric_admission_rechecks_actual_readiness_seal_contents(
    tmp_path: Path,
) -> None:
    _, schedule, cohort, readiness, ledger, grids = _metric_admission_fixture(tmp_path)
    audit_ledger = Path(readiness.readiness_root) / "audit-augmented-ledger.jsonl"
    audit_ledger.write_text('{"fixture":"mutated"}\n', encoding="utf-8")

    with pytest.raises(DataContractError, match="readiness_artifact_digest"):
        admit_metric_schedule(
            schedule=schedule,
            cohort=cohort,
            readiness=readiness,
            attempt_ledger=ledger,
            spatial_grids_by_image_id=grids,
        )


def test_completed_zero_prediction_call_reconstructs_terminal_bundle_digest(
    tmp_path: Path,
) -> None:
    evidence, decode_result = build_test_execution_evidence_with_result(
        arm_code="FULL_SINGLE",
        image_id=91,
        sampling_seed=404,
        raw_generated_text="",
        parser_text="",
        generated_token_ids=(151645,),
    )
    diagnostics = _terminal_call_diagnostics(
        error_count=0,
        natural_closure_count=1,
    )
    diagnostics["generated_token_count"] = 1
    bundle = TerminalCallOutputBundle.build(
        scheduled_request=evidence.scheduled_request,
        attempt_status="completed",
        execution_evidence=evidence,
        decode_result=decode_result,
        parse_score_receipts=(),
        row_diagnostics=(),
        call_diagnostics=diagnostics,
        stop_reason=decode_result.stop_reason,
        failure_code=None,
        cumulative_state_product=None,
    )
    path = tmp_path / "zero-prediction-terminal.json"
    path.write_bytes(bundle.to_artifact_bytes())
    admitted = MetricAdmittedRequest(
        request=evidence.scheduled_request,
        attempt_status="completed",
        output_artifact_sha256=bundle.bundle_sha256,
        output_artifact_path=str(path),
        failure_code=None,
        produced_cumulative_state_sha256=None,
        produced_cumulative_state_artifact_path=None,
    )
    call = CallMetricRecord(
        canonical_call_id=evidence.request_id,
        canonical_cell_index=None,
        sampling_seed=evidence.sampling_seed,
        raw_any_call_matched_reference_ids=(),
        raw_owning_call_matched_reference_ids=(),
        valid_prediction_ids=(),
        attempted_row_count=0,
        malformed_row_count=0,
        invalid_row_count=0,
        non_owning_prediction_count=0,
        natural_closure_count=1,
        controller_cap_count=0,
        token_cap_count=0,
        error_count=0,
        prompt_token_count=0,
        image_token_count=0,
        generated_token_count=1,
        wall_time_seconds=0.0,
        peak_device_memory_bytes=0,
        attempt_status="completed",
        source_image_sha256=evidence.source_image_sha256,
        image_frozen_order=evidence.scheduled_request.image_frozen_order,
        terminal_attempt_output_artifact_sha256=bundle.bundle_sha256,
        terminal_attempt_output_artifact_path=str(path),
    )

    _validate_call_against_terminal_bundle(
        call=call,
        admitted_request=admitted,
        predictions=(),
        row_records=(),
    )
    with pytest.raises(DataContractError, match="terminal_diagnostics"):
        _validate_call_against_terminal_bundle(
            call=replace(call, generated_token_count=2),
            admitted_request=admitted,
            predictions=(),
            row_records=(),
        )


def test_terminal_bundle_is_the_exact_parse_row_and_attempt_state_join(
    tmp_path: Path,
) -> None:
    evidence, decode_result = build_test_execution_evidence_with_result(
        arm_code="FULL_SINGLE",
        image_id=92,
        sampling_seed=405,
        raw_generated_text="person",
        parser_text="person",
        generated_token_ids=(101,),
    )
    receipt_payload = {
        "execution_evidence_fingerprint": evidence.fingerprint,
        "generated_row_index": 0,
        "object_span_id": "pred-0",
        "request_id": evidence.request_id,
    }
    receipt = _SyntheticParseReceipt(receipt_payload)
    row = RowMetricRecord(
        canonical_call_id=evidence.request_id,
        generated_row_index=0,
        prediction_id="pred-0",
        parse_status="parsed",
        validity_status="valid",
        ownership_status="owned",
    )
    diagnostics = _terminal_call_diagnostics(
        error_count=0,
        natural_closure_count=1,
    )
    diagnostics.update(
        attempted_row_count=1,
        generated_token_count=1,
        valid_prediction_ids=["pred-0"],
    )
    bundle = TerminalCallOutputBundle.build(
        scheduled_request=evidence.scheduled_request,
        attempt_status="completed",
        execution_evidence=evidence,
        decode_result=decode_result,
        parse_score_receipts=(receipt,),
        row_diagnostics=(row.to_json_dict(),),
        call_diagnostics=diagnostics,
        stop_reason=decode_result.stop_reason,
        failure_code=None,
        cumulative_state_product=None,
    )
    path = tmp_path / "parse-row-terminal.json"
    path.write_bytes(bundle.to_artifact_bytes())
    admitted = MetricAdmittedRequest(
        request=evidence.scheduled_request,
        attempt_status="completed",
        output_artifact_sha256=bundle.bundle_sha256,
        output_artifact_path=str(path),
        failure_code=None,
        produced_cumulative_state_sha256=None,
        produced_cumulative_state_artifact_path=None,
    )
    call = CallMetricRecord(
        canonical_call_id=evidence.request_id,
        canonical_cell_index=None,
        sampling_seed=evidence.sampling_seed,
        raw_any_call_matched_reference_ids=(),
        raw_owning_call_matched_reference_ids=(),
        valid_prediction_ids=("pred-0",),
        attempted_row_count=1,
        malformed_row_count=0,
        invalid_row_count=0,
        non_owning_prediction_count=0,
        natural_closure_count=1,
        controller_cap_count=0,
        token_cap_count=0,
        error_count=0,
        prompt_token_count=0,
        image_token_count=0,
        generated_token_count=1,
        wall_time_seconds=0.0,
        peak_device_memory_bytes=0,
        attempt_status="completed",
        source_image_sha256=evidence.source_image_sha256,
        image_frozen_order=evidence.scheduled_request.image_frozen_order,
        terminal_attempt_output_artifact_sha256=bundle.bundle_sha256,
        terminal_attempt_output_artifact_path=str(path),
    )
    prediction = SimpleNamespace(
        canonical_call_id=evidence.request_id,
        generated_row_index=0,
        prediction_id="pred-0",
        parse_score_receipt=receipt,
    )
    enriched_row = replace(row, matched_reference_ids=("reference-1",))
    _validate_call_against_terminal_bundle(
        call=call,
        admitted_request=admitted,
        predictions=(prediction,),
        row_records=(enriched_row,),
    )

    wrong_receipt = _SyntheticParseReceipt(
        {**receipt_payload, "object_span_id": "substituted"}
    )
    with pytest.raises(DataContractError, match="terminal_parse_receipts"):
        _validate_call_against_terminal_bundle(
            call=call,
            admitted_request=admitted,
            predictions=(
                SimpleNamespace(
                    canonical_call_id=evidence.request_id,
                    generated_row_index=0,
                    prediction_id="pred-0",
                    parse_score_receipt=wrong_receipt,
                ),
            ),
            row_records=(enriched_row,),
        )
    with pytest.raises(DataContractError, match="terminal_row_diagnostics"):
        _validate_call_against_terminal_bundle(
            call=call,
            admitted_request=admitted,
            predictions=(prediction,),
            row_records=(replace(enriched_row, generated_row_index=1),),
        )
    with pytest.raises(DataContractError, match="terminal_row_diagnostics"):
        _validate_call_against_terminal_bundle(
            call=call,
            admitted_request=admitted,
            predictions=(prediction,),
            row_records=(
                replace(
                    enriched_row,
                    prediction_id=None,
                    parse_status="malformed",
                    validity_status="not_applicable",
                    ownership_status="not_applicable",
                    matched_reference_ids=(),
                ),
            ),
        )
    with pytest.raises(DataContractError, match="terminal_row_diagnostics"):
        _validate_call_against_terminal_bundle(
            call=call,
            admitted_request=admitted,
            predictions=(prediction,),
            row_records=(replace(enriched_row, ownership_status="non_owning"),),
        )

    forged_bundle = TerminalCallOutputBundle.build(
        scheduled_request=evidence.scheduled_request,
        attempt_status="completed",
        execution_evidence=evidence,
        decode_result=decode_result,
        parse_score_receipts=(receipt,),
        row_diagnostics=(enriched_row.to_json_dict(),),
        call_diagnostics=diagnostics,
        stop_reason=decode_result.stop_reason,
        failure_code=None,
        cumulative_state_product=None,
    )
    forged_path = tmp_path / "forged-matched-terminal.json"
    forged_path.write_bytes(forged_bundle.to_artifact_bytes())
    with pytest.raises(DataContractError, match="terminal_row_diagnostics"):
        _validate_call_against_terminal_bundle(
            call=replace(
                call,
                terminal_attempt_output_artifact_sha256=forged_bundle.bundle_sha256,
                terminal_attempt_output_artifact_path=str(forged_path),
            ),
            admitted_request=replace(
                admitted,
                output_artifact_sha256=forged_bundle.bundle_sha256,
                output_artifact_path=str(forged_path),
            ),
            predictions=(prediction,),
            row_records=(enriched_row,),
        )

    with pytest.raises(DataContractError, match="terminal_bundle_cumulative_state"):
        MetricAdmittedRequest(
            request=evidence.scheduled_request,
            attempt_status="completed",
            output_artifact_sha256=bundle.bundle_sha256,
            output_artifact_path=str(path),
            failure_code=None,
            produced_cumulative_state_sha256=_digest("unexpected-state"),
            produced_cumulative_state_artifact_path="unexpected-state.json",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_evidence", {"forbidden": True}),
        ("decode_result", {"forbidden": True}),
        ("parse_score_receipts", [{"forbidden": True}]),
        ("row_diagnostics", [{"forbidden": True}]),
        (
            "cumulative_state_product",
            {"artifact_path": "state.json", "artifact_sha256": "0" * 64},
        ),
    ],
)
def test_noncompleted_terminal_bundle_deserialization_rejects_output_payload(
    field: str,
    value: object,
) -> None:
    evidence, _ = build_test_execution_evidence_with_result(
        arm_code="FULL_SINGLE",
        image_id=93,
        sampling_seed=406,
        raw_generated_text="",
        parser_text="",
        generated_token_ids=(151645,),
    )
    bundle = TerminalCallOutputBundle.build(
        scheduled_request=evidence.scheduled_request,
        attempt_status="failed",
        execution_evidence=None,
        decode_result=None,
        parse_score_receipts=(),
        row_diagnostics=(),
        call_diagnostics={},
        stop_reason=None,
        failure_code="fixture_failure",
        cumulative_state_product=None,
    )
    payload = dict(bundle.payload)
    payload[field] = value
    with pytest.raises(DataContractError, match="terminal_bundle_failure_payload"):
        TerminalCallOutputBundle.from_json_bytes(
            canonical_json_text(payload).encode("utf-8")
        )

@pytest.mark.parametrize("arm_identifier", ["FULL_BAG_K", "MASK_RESET"])
def test_metric_builder_rejects_partial_sixteen_call_universe(
    arm_identifier: str,
    tmp_path: Path,
) -> None:
    admission, schedule, *_ = _metric_admission_fixture(tmp_path)
    requests = tuple(
        request
        for request in schedule.requests
        if request.image_id == 0 and request.arm.arm_code == arm_identifier
    )
    assert len(requests) == 16
    merge = merge_image_predictions(
        image_id="0",
        arm_identifier=arm_identifier,
        predictions=(),
        source_width=512,
        source_height=512,
    )
    admitted_by_id = {
        item.request.request_id: item for item in admission.requests
    }
    calls = tuple(
        _failed_call_for_request(request, admitted_by_id[request.request_id])
        for request in requests
    )
    with pytest.raises(DataContractError, match="admission_call_universe"):
        build_image_metric_primitive(
            admission=admission,
            merge_result=merge,
            references=(),
            ledger_scope="audit_augmented",
            configured_call_budget=16,
            call_records=calls[:-1],
            row_records=(),
        )

    rogue_call = _empty_call(
        "rogue-call",
        cell_index=0,
        sampling_seed=0,
        natural_closure_count=0,
        error_count=1,
        attempt_status="failed",
    )
    with pytest.raises(DataContractError, match="admission_call_universe"):
        build_image_metric_primitive(
            admission=admission,
            merge_result=merge,
            references=(),
            ledger_scope="audit_augmented",
            configured_call_budget=16,
            call_records=(*calls, rogue_call),
            row_records=(),
        )


def test_metric_builder_derives_and_enforces_reference_spatial_contract(
    tmp_path: Path,
) -> None:
    admission, schedule, *_ = _metric_admission_fixture(tmp_path)
    request = next(
        request
        for request in schedule.requests
        if request.image_id == 0 and request.arm.arm_code == "FULL_SINGLE"
    )
    merge = merge_image_predictions(
        image_id="0",
        arm_identifier="FULL_SINGLE",
        predictions=(),
        source_width=512,
        source_height=512,
    )
    admitted_request = next(
        item for item in admission.requests if item.request.request_id == request.request_id
    )
    call = _failed_call_for_request(request, admitted_request)
    correct = _reference(
        "core-object",
        (16, 16, 64, 64),
        image_id="0",
        owner_cell_index=0,
        core_interior=True,
    )
    primitive = build_image_metric_primitive(
        admission=admission,
        merge_result=merge,
        references=(correct,),
        ledger_scope="audit_augmented",
        configured_call_budget=1,
        call_records=(call,),
        row_records=(),
    )
    assert primitive.core_interior_reference_ids == ("core-object",)
    assert primitive.metric_admission_sha256 == admission.fingerprint
    assert primitive.source_image_sha256 == admission.image("0").source_image_sha256
    assert tuple(
        reference.reference_id
        for reference in admission.readiness.reference_objects(
            image_id="0",
            ledger_scope="official_annotation",
        )
    ) == ("coco-ann:1",)

    for substituted_references in (
        (),
        (replace(correct, source_canvas_bbox_xyxy=(20, 20, 64, 64)),),
        (
            correct,
            _reference(
                "arbitrary-extra",
                (80, 80, 96, 96),
                image_id="0",
            ),
        ),
        (correct, correct),
    ):
        with pytest.raises(
            DataContractError,
            match="admission_reference_(set|duplicate)",
        ):
            build_image_metric_primitive(
                admission=admission,
                merge_result=merge,
                references=substituted_references,
                ledger_scope="audit_augmented",
                configured_call_budget=1,
                call_records=(call,),
                row_records=(),
            )

    with pytest.raises(DataContractError, match="admission_call_provenance"):
        build_image_metric_primitive(
            admission=admission,
            merge_result=merge,
            references=(correct,),
            ledger_scope="audit_augmented",
            configured_call_budget=1,
            call_records=(replace(call, source_image_sha256=_digest("wrong-source")),),
            row_records=(),
        )

    with pytest.raises(DataContractError, match="reference_spatial_derivation"):
        build_image_metric_primitive(
            admission=admission,
            merge_result=merge,
            references=(replace(correct, core_interior_for_mask_harm=False),),
            ledger_scope="audit_augmented",
            configured_call_budget=1,
            call_records=(call,),
            row_records=(),
        )

    with pytest.raises(DataContractError, match="metrics_owner_cell"):
        replace(correct, owner_cell_index=99)
