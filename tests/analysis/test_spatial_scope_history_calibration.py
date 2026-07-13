from __future__ import annotations

from dataclasses import replace
import json

import pytest

from src.analysis.spatial_scope_history.calibration import (
    AttestedSamplingPolicySet,
    CalibrationCallObservation,
    CalibrationSelectionReceipt,
    _select_sampling_calibration_from_reconstructed_observations,
    build_calibration_requests,
    summarize_calibration_panel,
    write_immutable_json,
)
from src.analysis.spatial_scope_history.cohort_ledger import (
    CohortImageRecord,
    CohortLedger,
    sha256_file,
    sha256_payload,
)
from src.common.errors import ArtifactContractError, DataContractError
from src.inference.backend import DecodeGenerationPolicy


def _digest(label: str) -> str:
    return sha256_payload({"label": label})


def _cohort(cohort_id: str, count: int, *, image_id_offset: int) -> CohortLedger:
    return CohortLedger(
        cohort_id=cohort_id,
        full_name=f"Synthetic {cohort_id}",
        operational_meaning="Synthetic exact-contract fixture.",
        records=tuple(
            CohortImageRecord(
                image_id=image_id_offset + order,
                frozen_order=order,
                source_row_index=order,
                image_path=f"/images/{image_id_offset + order}.jpg",
                image_sha256=_digest(f"image-{image_id_offset + order}"),
                source_width=1024,
                source_height=768,
                raw_width=640,
                raw_height=480,
                source_row_sha256=_digest(f"row-{image_id_offset + order}"),
                source_dataset_sha256=_digest("source"),
                raw_annotation_sha256=_digest("annotation"),
                noncrowd_annotated_object_count=10,
                annotated_person_count=7,
                annotated_food_tableware_count=0,
                source_crowd_annotation_count=0,
                cohort_memberships=(cohort_id,),
                density_tags=("synthetic",),
            )
            for order in range(count)
        ),
    )


def _policy_set() -> AttestedSamplingPolicySet:
    return AttestedSamplingPolicySet(
        aggregate_artifact_sha256=_digest("aggregate-file"),
        aggregate_payload_fingerprint=_digest("aggregate-payload"),
        policy_fingerprints_by_temperature=tuple(
            (
                temperature,
                DecodeGenerationPolicy.sampled(
                    max_new_tokens=512,
                    repetition_penalty=1.0,
                    temperature=temperature,
                    top_p=0.95,
                ).fingerprint,
            )
            for temperature in (0.2, 0.4, 0.6)
        ),
    )


@pytest.fixture
def calibration_cohort() -> CohortLedger:
    return _cohort("sampling-calibration-12", 12, image_id_offset=10_000)


@pytest.fixture
def validation_cohort() -> CohortLedger:
    return _cohort("validation-200", 200, image_id_offset=20_000)


def _panel_observations(
    *,
    cohort: CohortLedger,
    temperature: float,
    panel_kind: str,
    gate_passes: bool,
    replay_mismatch: bool = False,
) -> tuple[CalibrationCallObservation, ...]:
    policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=512,
        repetition_penalty=1.0,
        temperature=temperature,
        top_p=0.95,
    )
    requests = build_calibration_requests(
        cohort=cohort,
        temperature=temperature,
        decode_generation_policy_fingerprint=policy.fingerprint,
    )
    rows: list[CalibrationCallObservation] = []
    for batch_index in range(12):
        canonical = requests[batch_index * 4 : (batch_index + 1) * 4]
        physical = (
            tuple(reversed(canonical)) if panel_kind == "reversed_order" else canonical
        )
        for execution_index, request in enumerate(physical):
            raw_label = f"raw:{request.request_id}"
            if replay_mismatch and panel_kind == "exact_replay" and batch_index == 0:
                raw_label += ":mismatch"
            rows.append(
                CalibrationCallObservation(
                    panel_kind=panel_kind,
                    request=request,
                    physical_batch_index=batch_index,
                    request_execution_index=execution_index,
                    raw_response_bytes_sha256=_digest(raw_label),
                    parse_without_call_level_failure=(gate_passes or batch_index < 8),
                    natural_closure=gate_passes or batch_index < 6,
                    detected_official_reference_object_ids=(
                        f"reference-{request.call_index % 2}",
                    ),
                    decode_receipt_fingerprint=_digest(
                        f"receipt:{panel_kind}:{request.request_id}"
                    ),
                    terminal_bundle_sha256=_digest(
                        f"bundle:{panel_kind}:{request.request_id}"
                    ),
                )
            )
    return tuple(rows)


def _selection_observations(
    cohort: CohortLedger,
) -> tuple[CalibrationCallObservation, ...]:
    return (
        *_panel_observations(
            cohort=cohort,
            temperature=0.2,
            panel_kind="initial",
            gate_passes=False,
        ),
        *_panel_observations(
            cohort=cohort,
            temperature=0.4,
            panel_kind="initial",
            gate_passes=True,
        ),
        *_panel_observations(
            cohort=cohort,
            temperature=0.4,
            panel_kind="exact_replay",
            gate_passes=True,
        ),
        *_panel_observations(
            cohort=cohort,
            temperature=0.4,
            panel_kind="reversed_order",
            gate_passes=True,
        ),
    )


def _select(
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
    observations: tuple[CalibrationCallObservation, ...] | None = None,
) -> CalibrationSelectionReceipt:
    return _select_sampling_calibration_from_reconstructed_observations(
        calibration_cohort=calibration_cohort,
        validation_cohort=validation_cohort,
        attested_policy_set=_policy_set(),
        observations=(
            _selection_observations(calibration_cohort)
            if observations is None
            else observations
        ),
    )


def test_pure_selection_core_selects_first_passing_policy_at_192_calls(
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
) -> None:
    receipt = _select(calibration_cohort, validation_cohort)

    assert receipt.selected_temperature == 0.4
    assert receipt.total_model_call_count == 192
    assert [item.temperature for item in receipt.candidate_evaluations] == [0.2, 0.4]
    assert [item.selected for item in receipt.candidate_evaluations] == [False, True]
    assert receipt.metric_eligible is False
    serialized = json.dumps(receipt.to_artifact_dict(), sort_keys=True)
    assert "average_precision" not in serialized
    assert "local_rescue_rate" not in serialized
    assert (
        CalibrationSelectionReceipt.from_artifact_dict(receipt.to_artifact_dict())
        == receipt
    )
    tampered = receipt.to_artifact_dict()
    tampered["selected_temperature"] = 0.6
    with pytest.raises(ArtifactContractError):
        CalibrationSelectionReceipt.from_artifact_dict(tampered)


def test_calibration_threshold_boundaries_recompute_exactly(
    calibration_cohort: CohortLedger,
) -> None:
    rows = list(
        _panel_observations(
            cohort=calibration_cohort,
            temperature=0.4,
            panel_kind="initial",
            gate_passes=True,
        )
    )
    observations = [
        replace(
            row,
            parse_without_call_level_failure=index < 46,
            natural_closure=index < 39,
            raw_response_bytes_sha256=_digest(
                f"raw-{row.request.image_id}-{row.request.call_index if row.request.image_frozen_order < 9 else 0}"
            ),
        )
        for index, row in enumerate(rows)
    ]
    panel = summarize_calibration_panel(observations)
    assert panel.parse_success_count == 46
    assert panel.natural_closure_count == 39
    assert panel.images_with_multiple_distinct_raw_outputs == 9
    assert panel.prediction_set_diversity >= 0.10
    assert panel.gate_passed

    below = list(observations)
    below[45] = replace(below[45], parse_without_call_level_failure=False)
    assert not summarize_calibration_panel(below).gate_passed

    empty_union = [
        replace(row, detected_official_reference_object_ids=()) for row in observations
    ]
    empty_panel = summarize_calibration_panel(empty_union)
    assert empty_panel.prediction_set_diversity == 0.0
    assert empty_panel.prediction_set_diversity_image_count == 0
    assert not empty_panel.gate_passed


def test_calibration_panel_rejects_image_order_drift(
    calibration_cohort: CohortLedger,
) -> None:
    observations = list(
        _panel_observations(
            cohort=calibration_cohort,
            temperature=0.4,
            panel_kind="initial",
            gate_passes=True,
        )
    )
    drifted = observations[4:8] + observations[0:4] + observations[8:]
    with pytest.raises(ArtifactContractError, match="image order"):
        summarize_calibration_panel(drifted)


def test_replay_or_order_dependence_is_a_hard_failure(
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
) -> None:
    observations = list(_selection_observations(calibration_cohort))
    replay_start = 96
    observations[replay_start] = replace(
        observations[replay_start],
        raw_response_bytes_sha256=_digest("mismatch"),
    )
    with pytest.raises(ArtifactContractError, match="replay or order invariance"):
        _select(calibration_cohort, validation_cohort, tuple(observations))


def test_materialized_observations_reject_post_selection_panels(
    calibration_cohort: CohortLedger,
    validation_cohort: CohortLedger,
) -> None:
    observations = _selection_observations(calibration_cohort)
    receipt = _select(calibration_cohort, validation_cohort, observations)
    assert receipt.total_model_call_count == len(observations)

    extra = _panel_observations(
        cohort=calibration_cohort,
        temperature=0.6,
        panel_kind="initial",
        gate_passes=True,
    )[0]
    with pytest.raises(ArtifactContractError, match="post-selection"):
        _select(calibration_cohort, validation_cohort, (*observations, extra))


def test_calibration_rejects_validation_overlap(
    calibration_cohort: CohortLedger,
) -> None:
    overlapping = _cohort("validation-200", 200, image_id_offset=10_000)
    with pytest.raises(DataContractError, match="overlaps"):
        _select(calibration_cohort, overlapping)


def test_immutable_writer_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    write_immutable_json(path, {"schema_version": "test.v1", "value": 1})
    first_digest = sha256_file(path)
    with pytest.raises(ArtifactContractError, match="overwrite"):
        write_immutable_json(path, {"schema_version": "test.v1", "value": 2})
    assert sha256_file(path) == first_digest
