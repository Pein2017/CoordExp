from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.rollout_calibration.state_bank as state_bank_module
from src.common.errors import ArtifactContractError
from src.config.fingerprint import sha256_json
from src.inference.backend import token_ids_sha256
from src.rollout_calibration import (
    StateBankEvent,
    assemble_state_bank,
    load_state_bank,
    load_state_bank_manifest_binding,
    validate_state_bank_token_identity,
)
from src.rollout_calibration.state_bank import (
    STATE_BANK_MANIFEST_NAME,
    STATE_BANK_RECORDS_NAME,
    StateBankCandidate,
)
from conftest import source_artifacts, synthetic_inputs


def _assemble(
    tmp_path: Path,
    checkpoint_identity,
    prompt_identity_sha256: str,
    *,
    rollouts=None,
    reviews=None,
    output_name: str = "bank",
):
    if rollouts is None or reviews is None:
        rollouts, reviews, _ = synthetic_inputs(tmp_path)
    return assemble_state_bank(
        output_dir=tmp_path / output_name,
        rollout_rows=rollouts,
        review_rows=reviews,
        source_checkpoint=checkpoint_identity,
        prompt_identity_sha256=prompt_identity_sha256,
        source_artifacts=source_artifacts(),
    )


def test_state_bank_round_trip_binds_identity_splits_and_receipt(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    manifest = _assemble(tmp_path, checkpoint_identity, prompt_identity_sha256)

    loaded = load_state_bank(
        tmp_path / "bank" / STATE_BANK_MANIFEST_NAME,
        expected_source_checkpoint=checkpoint_identity,
        expected_prompt_identity_sha256=prompt_identity_sha256,
    )

    assert loaded.manifest.bank_id == manifest.bank_id
    assert loaded.records[0].executed_prompt_token_ids == (
        10,
        151655,
        151655,
        151655,
        151655,
        11,
    )
    assert loaded.records[0].prefix_token_ids == (12, 13)
    assert loaded.records[0].prefix_object_row_count == 1
    assert loaded.records[0].prefix_coverage_status == "resolved"
    assert loaded.records[0].positive_path_imitation_eligible is False
    assert loaded.records[0].image_balanced_event_weight == 1.0
    assert loaded.validation_receipt.to_artifact_dict() == {
        "bank_id": manifest.bank_id,
        "source_checkpoint_id": manifest.source_checkpoint_id,
        "source_checkpoint": checkpoint_identity.to_artifact_dict(),
        "records_sha256": manifest.records_sha256,
        "record_count": 1,
        "split_counts": {"train": 1},
        "event_family_counts": {"coordinate_boundary": 1, "entity_transition": 1},
        "rejection_reasons": {},
        "status": "validated",
    }


def _counterfactual_admission() -> dict:
    return {
        "row_budget": {"native": 1, "counterfactual": 1},
        "generated_token_budget": {"native": 512, "counterfactual": 512},
        "target_owner_retained": True,
        "verified_owner_delta": {
            "added_owner_ids": ["entity-a"],
            "removed_owner_ids": [],
        },
        "confirmed_new_duplicate_count": 0,
        "confirmed_new_malformed_count": 0,
        "confirmed_new_unsupported_entity_count": 0,
        "unknown_suffix_neutral": True,
        "unknown_suffix_provenance": {
            "source": "synthetic_unit_test",
            "reason": "suffix was not used for admission",
        },
    }


def test_entity_transition_counterfactual_admission_round_trip_is_strict(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["counterfactual_admission"] = _counterfactual_admission()

    event = StateBankEvent.from_mapping(joined)

    assert event.counterfactual_admission is not None
    evidence = event.counterfactual_admission
    assert evidence.row_budget == {"native": 1, "counterfactual": 1}
    assert evidence.generated_token_budget["native"] == 512
    assert event.to_artifact_dict()["counterfactual_admission"] == (
        _counterfactual_admission()
    )


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda evidence: evidence["row_budget"].update(counterfactual=2),
            "state_bank.admission_row_budget_mismatch",
        ),
        (
            lambda evidence: evidence.update(target_owner_retained=False),
            "state_bank.admission_target_owner_not_retained",
        ),
        (
            lambda evidence: evidence.update(confirmed_new_duplicate_count=1),
            "state_bank.admission_confirmed_new_harm",
        ),
        (
            lambda evidence: evidence["verified_owner_delta"].update(
                added_owner_ids=[]
            ),
            "state_bank.admission_empty_owner_delta",
        ),
    ],
)
def test_entity_transition_counterfactual_admission_rejects_unsafe_evidence(
    tmp_path: Path, mutate, expected_code: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    evidence = _counterfactual_admission()
    mutate(evidence)
    joined["counterfactual_admission"] = evidence

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == expected_code


def _target_owner_noncoverage_proof() -> dict:
    return {
        "target_owner_id": "entity-a",
        "native_terminal_generated_step": 7,
        "prior_row_count": 1,
        "thresholds": {
            "target_iou_max": 0.25,
            "intersection_over_smaller_area_max": 0.5,
        },
        "prior_row_exclusions": [
            {
                "row_index": 0,
                "prediction_description": "person",
                "prediction_coord_bins": [10, 10, 90, 90],
                "same_category": True,
                "target_iou": 0.0,
                "target_center_inside_prediction": False,
                "prediction_center_inside_target": False,
                "intersection_over_smaller_area": 0.0,
                "best_same_class_owner_id": "entity-covered",
                "best_same_class_owner_margin": 0.1,
                "plausible_target_association": False,
                "evidence_reason": "prior row does not plausibly represent target",
            }
        ],
    }


@pytest.mark.parametrize("prefix_status", ["empty", "resolved", "unresolved"])
def test_positive_path_imitation_event_accepts_context_prefix_modes(
    tmp_path: Path, prefix_status: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["positive_path_imitation_eligible"] = True
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = False
    positive = joined["candidates"][0]
    joined["candidates"] = [positive]
    positive.update(
        {
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 3],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                {"candidate_token_offset": 2, "intended_token_type": "schema"},
            ],
        }
    )
    if prefix_status == "empty":
        joined["prefix_object_row_count"] = 0
        joined["prefix_coverage_status"] = "empty"
        joined["prefix_token_ids"] = []
        joined["prefix_token_ids_sha256"] = token_ids_sha256([])
        joined["prefix_covered_owner_proofs"] = []
        positive["generation_provenance"]["prefix_token_ids_sha256"] = (
            token_ids_sha256([])
        )
    elif prefix_status == "unresolved":
        joined["prefix_coverage_status"] = "unresolved"
        joined["prefix_covered_owner_proofs"] = []
    event = StateBankEvent.from_mapping(joined)

    assert event.positive_path_imitation_eligible is True
    assert event.entity_transition_eligible is False
    assert event.coordinate_boundary_eligible is False
    assert event.candidates[0].owner_resolution_interval == (0, 3)


def test_positive_path_imitation_token_identity_allows_masked_untrusted_coordinates(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["positive_path_imitation_eligible"] = True
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = False
    joined["candidates"] = [joined["candidates"][0]]
    positive = joined["candidates"][0]
    positive.update(
        {
            "token_ids": [20, 1500, 22],
            "token_ids_sha256": token_ids_sha256((20, 1500, 22)),
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 3],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                {"candidate_token_offset": 2, "intended_token_type": "schema"},
            ],
        }
    )

    event = StateBankEvent.from_mapping(joined)
    validate_state_bank_token_identity(
        SimpleNamespace(records=(event,)),
        SimpleNamespace(
            im_end_token_ids=(5,),
            coordinate_token_ids=tuple(range(1000, 2000)),
        ),
    )


@pytest.mark.parametrize(
    ("offset", "wrong_type"),
    [(0, "coordinate"), (1, "schema")],
)
def test_positive_path_imitation_rejects_selected_site_token_type_mismatch(
    tmp_path: Path,
    offset: int,
    wrong_type: str,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["positive_path_imitation_eligible"] = True
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = False
    joined["candidates"] = [joined["candidates"][0]]
    positive = joined["candidates"][0]
    selected_sites = [
        {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
        {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
        {"candidate_token_offset": 2, "intended_token_type": "schema"},
    ]
    selected_sites[offset]["intended_token_type"] = wrong_type
    positive.update(
        {
            "token_ids": [20, 1500, 22],
            "token_ids_sha256": token_ids_sha256((20, 1500, 22)),
            "geometry_review_status": "trusted",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 3],
            "selected_sites": selected_sites,
        }
    )
    event = StateBankEvent.from_mapping(joined)

    with pytest.raises(ArtifactContractError) as exc_info:
        validate_state_bank_token_identity(
            SimpleNamespace(records=(event,)),
            SimpleNamespace(
                im_end_token_ids=(5,),
                coordinate_token_ids=tuple(range(1000, 2000)),
            ),
        )
    assert exc_info.value.code == "state_bank.positive_path_token_type_identity"


def test_positive_path_imitation_rejects_harmful_or_multiple_candidates(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["positive_path_imitation_eligible"] = True
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = False
    positive = joined["candidates"][0]
    positive.update(
        {
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 3],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                {"candidate_token_offset": 2, "intended_token_type": "schema"},
            ],
        }
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)
    assert exc_info.value.code == "state_bank.positive_path_candidate_count"


def test_positive_path_imitation_rejects_partial_row_interval(tmp_path: Path) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["positive_path_imitation_eligible"] = True
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = False
    joined["candidates"] = [joined["candidates"][0]]
    positive = joined["candidates"][0]
    positive.update(
        {
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 1],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"}
            ],
        }
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)
    assert exc_info.value.code == "state_bank.positive_path_full_row_interval"


def _positive_path_collection_event(
    tmp_path: Path,
    *,
    event_id: str,
    image_id: int,
    weight: float,
) -> StateBankEvent:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["positive_path_imitation_eligible"] = True
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = False
    joined["candidates"] = [joined["candidates"][0]]
    positive = joined["candidates"][0]
    positive.update(
        {
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 3],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                {"candidate_token_offset": 2, "intended_token_type": "schema"},
            ],
        }
    )
    event = StateBankEvent.from_mapping(joined)
    image = replace(
        event.image,
        image_id=image_id,
        content_sha256=f"{image_id:064x}",
    )
    return replace(
        event,
        event_id=event_id,
        image=image,
        split_group_id=f"image:{image_id}",
        image_balanced_event_weight=weight,
    )


def test_positive_path_collection_requires_unit_mean_event_weight(
    tmp_path: Path,
) -> None:
    event = _positive_path_collection_event(
        tmp_path,
        event_id="positive-path-a",
        image_id=101,
        weight=2.0,
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        state_bank_module._validate_record_collection((event,))

    assert exc_info.value.code == "state_bank.positive_path_event_weight_mean"


def test_positive_path_collection_requires_equal_total_weight_per_image(
    tmp_path: Path,
) -> None:
    unequal = (
        _positive_path_collection_event(
            tmp_path,
            event_id="positive-path-a0",
            image_id=101,
            weight=0.5,
        ),
        _positive_path_collection_event(
            tmp_path,
            event_id="positive-path-a1",
            image_id=101,
            weight=0.5,
        ),
        _positive_path_collection_event(
            tmp_path,
            event_id="positive-path-b0",
            image_id=202,
            weight=2.0,
        ),
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        state_bank_module._validate_record_collection(unequal)

    assert exc_info.value.code == "state_bank.positive_path_image_weight_totals"

    balanced = tuple(
        replace(event, image_balanced_event_weight=weight)
        for event, weight in zip(unequal, (0.75, 0.75, 1.5), strict=True)
    )
    state_bank_module._validate_record_collection(balanced)


def test_positive_path_weight_normalization_ignores_non_positive_records(
    tmp_path: Path,
) -> None:
    positive = _positive_path_collection_event(
        tmp_path,
        event_id="positive-path-a",
        image_id=101,
        weight=2.0,
    )
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    historical = replace(
        StateBankEvent.from_mapping(_joined_event(rollouts[0], reviews[0])),
        event_id="historical-transition",
        image_balanced_event_weight=3.0,
    )

    normalized = state_bank_module._normalize_image_balanced_event_weights(
        (positive, historical)
    )

    assert normalized[0].image_balanced_event_weight == 1.0
    assert normalized[1].image_balanced_event_weight == 3.0
    state_bank_module._validate_record_collection(normalized)


def test_target_scoped_noncoverage_allows_only_premature_terminal(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["prefix_coverage_status"] = "target_scoped_noncoverage"
    joined["prefix_covered_owner_proofs"] = []
    joined["target_owner_noncoverage_proof"] = _target_owner_noncoverage_proof()
    joined["counterfactual_admission"] = _counterfactual_admission()
    harmful = next(item for item in joined["candidates"] if item["role"] == "harmful")
    harmful.update(
        {
            "harmful_kind": "premature_terminal",
            "physical_owner_id": None,
            "coverage_status": "unknown",
            "owner_resolution_interval": None,
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "schema"}
            ],
        }
    )

    event = StateBankEvent.from_mapping(joined)

    assert event.prefix_coverage_status == "target_scoped_noncoverage"
    assert event.target_owner_noncoverage_proof is not None
    assert event.target_owner_noncoverage_proof.target_owner_id == "entity-a"


def test_target_scoped_noncoverage_rejects_duplicate_harmful_branch(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["prefix_coverage_status"] = "target_scoped_noncoverage"
    joined["prefix_covered_owner_proofs"] = []
    joined["target_owner_noncoverage_proof"] = _target_owner_noncoverage_proof()
    joined["counterfactual_admission"] = _counterfactual_admission()
    harmful = next(item for item in joined["candidates"] if item["role"] == "harmful")
    harmful.update(
        {
            "physical_owner_id": None,
            "coverage_status": "unknown",
            "entity_eligible": False,
            "owner_resolution_interval": None,
            "selected_sites": [],
        }
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == "state_bank.target_noncoverage_harmful_kind"


def test_target_scoped_noncoverage_requires_counterfactual_admission(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["prefix_coverage_status"] = "target_scoped_noncoverage"
    joined["prefix_covered_owner_proofs"] = []
    joined["target_owner_noncoverage_proof"] = _target_owner_noncoverage_proof()
    harmful = next(item for item in joined["candidates"] if item["role"] == "harmful")
    harmful.update(
        {
            "harmful_kind": "premature_terminal",
            "physical_owner_id": None,
            "coverage_status": "unknown",
            "owner_resolution_interval": None,
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "schema"}
            ],
        }
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == "state_bank.target_noncoverage_admission_missing"


def test_manifest_record_counts_are_recomputed_during_full_load(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    _assemble(tmp_path, checkpoint_identity, prompt_identity_sha256)
    manifest_path = tmp_path / "bank" / STATE_BANK_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["split_counts"] = {"train": 2}
    payload["bank_id"] = sha256_json(
        {key: value for key, value in payload.items() if key != "bank_id"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactContractError) as exc_info:
        load_state_bank(
            manifest_path,
            expected_source_checkpoint=checkpoint_identity,
            expected_prompt_identity_sha256=prompt_identity_sha256,
        )

    assert exc_info.value.code == "state_bank.split_counts"


def test_state_bank_identity_is_deterministic_and_output_is_write_once(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    first = _assemble(
        tmp_path,
        checkpoint_identity,
        prompt_identity_sha256,
        rollouts=rollouts,
        reviews=reviews,
        output_name="bank-a",
    )
    second = _assemble(
        tmp_path,
        checkpoint_identity,
        prompt_identity_sha256,
        rollouts=rollouts,
        reviews=reviews,
        output_name="bank-b",
    )
    assert first.bank_id == second.bank_id
    assert first.records_sha256 == second.records_sha256
    assert (tmp_path / "bank-a" / STATE_BANK_RECORDS_NAME).read_bytes() == (
        tmp_path / "bank-b" / STATE_BANK_RECORDS_NAME
    ).read_bytes()

    with pytest.raises(ArtifactContractError) as exc_info:
        _assemble(
            tmp_path,
            checkpoint_identity,
            prompt_identity_sha256,
            rollouts=rollouts,
            reviews=reviews,
            output_name="bank-a",
        )
    assert exc_info.value.code == "state_bank.immutable_output_exists"


def test_coordinate_only_event_accepts_one_diagnostic_greedy_candidate(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = True
    candidate = joined["candidates"][0]
    candidate.update(
        {
            "role": "diagnostic",
            "harmful_kind": None,
            "entity_eligible": False,
            "owner_resolution_interval": None,
            "selected_sites": [
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"}
            ],
            "generation_provenance": {
                **candidate["generation_provenance"],
                "mode": "greedy",
                "seed": 0,
                "temperature": 0.0,
            },
        }
    )
    joined["candidates"] = [candidate]

    event = StateBankEvent.from_mapping(joined)

    assert event.entity_transition_eligible is False
    assert event.coordinate_boundary_eligible is True
    assert len(event.candidates) == 1
    assert event.candidates[0].role == "diagnostic"
    assert event.candidates[0].harmful_kind is None
    assert event.candidates[0].entity_eligible is False
    assert event.candidates[0].geometry_eligible is True
    assert event.candidates[0].coordinate_decision is not None


def test_coordinate_only_event_rejects_non_greedy_diagnostic_candidate(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = True
    candidate = joined["candidates"][0]
    candidate.update(
        {
            "role": "diagnostic",
            "harmful_kind": None,
            "entity_eligible": False,
            "owner_resolution_interval": None,
            "selected_sites": [
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"}
            ],
            "generation_provenance": {
                **candidate["generation_provenance"],
                "mode": "sampled",
                "seed": 11,
                "temperature": 0.2,
            },
        }
    )
    joined["candidates"] = [candidate]

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == "state_bank.coordinate_greedy_diagnostic"


def _coordinate_only_joined_event(tmp_path: Path) -> dict:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["entity_transition_eligible"] = False
    joined["coordinate_boundary_eligible"] = True
    candidate = joined["candidates"][0]
    candidate.update(
        {
            "role": "diagnostic",
            "harmful_kind": None,
            "coverage_status": "unknown",
            "entity_eligible": False,
            "owner_resolution_interval": None,
            "selected_sites": [
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"}
            ],
            "generation_provenance": {
                **candidate["generation_provenance"],
                "mode": "greedy",
                "seed": 0,
                "temperature": 0.0,
            },
        }
    )
    joined["candidates"] = [candidate]
    return joined


def test_empty_prefix_coverage_accepts_exactly_zero_rows_tokens_and_proofs(
    tmp_path: Path,
) -> None:
    joined = _coordinate_only_joined_event(tmp_path)
    joined["prefix_object_row_count"] = 0
    joined["prefix_coverage_status"] = "empty"
    joined["prefix_token_ids"] = []
    joined["prefix_token_ids_sha256"] = token_ids_sha256([])
    joined["prefix_covered_owner_proofs"] = []
    for candidate in joined["candidates"]:
        candidate["generation_provenance"]["prefix_token_ids_sha256"] = (
            token_ids_sha256([])
        )

    event = StateBankEvent.from_mapping(joined)

    assert event.prefix_object_row_count == 0
    assert event.prefix_coverage_status == "empty"
    assert event.prefix_token_ids == ()
    assert event.prefix_covered_owner_proofs == ()


def test_empty_prefix_coverage_rejects_entity_transition_supervision(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["prefix_object_row_count"] = 0
    joined["prefix_coverage_status"] = "empty"
    joined["prefix_token_ids"] = []
    joined["prefix_token_ids_sha256"] = token_ids_sha256([])
    joined["prefix_covered_owner_proofs"] = []
    for candidate in joined["candidates"]:
        candidate["generation_provenance"]["prefix_token_ids_sha256"] = (
            token_ids_sha256([])
        )
        if candidate["role"] == "harmful":
            candidate.update(
                harmful_kind="premature_terminal",
                physical_owner_id=None,
                coverage_status="unknown",
                owner_resolution_interval=None,
                selected_sites=[
                    {"candidate_token_offset": 0, "intended_token_type": "schema"}
                ],
            )

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == "state_bank.entity_prefix_coverage_not_resolved"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda event: event.update(
                prefix_object_row_count=0,
                prefix_coverage_status="empty",
            ),
            "state_bank.prefix_coverage_empty",
        ),
        (
            lambda event: event.update(prefix_object_row_count=2),
            "state_bank.prefix_coverage_resolved",
        ),
        (
            lambda event: event.update(
                prefix_coverage_status="unresolved",
            ),
            "state_bank.prefix_coverage_unresolved",
        ),
    ],
)
def test_prefix_coverage_shape_rejects_inconsistent_row_token_and_proof_counts(
    tmp_path: Path, mutate, expected_code: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    mutate(joined)

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == expected_code


def test_unresolved_prefix_coverage_accepts_coordinate_only_unknown_coverage(
    tmp_path: Path,
) -> None:
    joined = _coordinate_only_joined_event(tmp_path)
    joined["prefix_coverage_status"] = "unresolved"
    joined["prefix_covered_owner_proofs"] = []

    event = StateBankEvent.from_mapping(joined)

    assert event.prefix_object_row_count == 1
    assert event.prefix_coverage_status == "unresolved"
    assert event.prefix_covered_owner_proofs == ()
    assert event.entity_transition_eligible is False
    assert all(not candidate.entity_eligible for candidate in event.candidates)
    assert event.candidates[0].coverage_status == "unknown"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda event: event.update(entity_transition_eligible=True),
            "state_bank.unresolved_prefix_entity_event",
        ),
        (
            lambda event: event["candidates"][0].update(
                role="positive",
                coverage_status="uncovered",
                entity_eligible=True,
                owner_resolution_interval=[0, 1],
                selected_sites=[
                    {
                        "candidate_token_offset": 0,
                        "intended_token_type": "desc_text",
                    },
                    {
                        "candidate_token_offset": 1,
                        "intended_token_type": "coordinate",
                    },
                ],
                generation_provenance={
                    **event["candidates"][0]["generation_provenance"],
                    "mode": "sampled",
                    "seed": 11,
                    "temperature": 0.2,
                },
            ),
            "state_bank.unresolved_prefix_entity_candidate",
        ),
        (
            lambda event: event["candidates"][0].update(
                coverage_status="uncovered"
            ),
            "state_bank.unresolved_prefix_coordinate_coverage",
        ),
    ],
)
def test_unresolved_prefix_coverage_rejects_entity_use_and_known_coordinate_coverage(
    tmp_path: Path, mutate, expected_code: str
) -> None:
    joined = _coordinate_only_joined_event(tmp_path)
    joined["prefix_coverage_status"] = "unresolved"
    joined["prefix_covered_owner_proofs"] = []
    mutate(joined)

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == expected_code


def test_manifest_binding_is_strict_and_does_not_read_records(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    manifest = _assemble(tmp_path, checkpoint_identity, prompt_identity_sha256)
    records_path = tmp_path / "bank" / STATE_BANK_RECORDS_NAME
    records_path.unlink()

    binding = load_state_bank_manifest_binding(
        tmp_path / "bank" / STATE_BANK_MANIFEST_NAME
    )

    assert binding.source_checkpoint_id == sha256_json(
        checkpoint_identity.to_artifact_dict()
    )
    assert binding.bank_id == manifest.bank_id
    assert binding.fingerprint == manifest.bank_id
    assert binding.records_sha256 == manifest.records_sha256
    assert binding.record_count == 1
    assert dict(binding.split_counts) == {"train": 1}
    assert dict(binding.event_family_counts) == {
        "coordinate_boundary": 1,
        "entity_transition": 1,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.replace(
            '"bank_id":', f'"bank_id": "{"0" * 64}", "bank_id":', 1
        ),
        lambda raw: raw.replace('"record_count": 1', '"record_count": NaN', 1),
    ],
)
def test_manifest_binding_rejects_non_strict_json(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str, mutate
) -> None:
    _assemble(tmp_path, checkpoint_identity, prompt_identity_sha256)
    manifest_path = tmp_path / "bank" / STATE_BANK_MANIFEST_NAME
    manifest_path.write_text(
        mutate(manifest_path.read_text(encoding="utf-8")), encoding="utf-8"
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        load_state_bank_manifest_binding(manifest_path)

    assert exc_info.value.code == "state_bank.manifest_json"


def test_checkpoint_mismatch_fails_before_consumption(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    _assemble(tmp_path, checkpoint_identity, prompt_identity_sha256)
    wrong = copy.deepcopy(checkpoint_identity.to_artifact_dict())
    wrong["adapter_fingerprint"] = "9" * 64

    with pytest.raises(ArtifactContractError) as exc_info:
        load_state_bank(
            tmp_path / "bank" / STATE_BANK_MANIFEST_NAME,
            expected_source_checkpoint=wrong,
            expected_prompt_identity_sha256=prompt_identity_sha256,
        )
    assert exc_info.value.code == "state_bank.source_checkpoint_mismatch"


def test_exact_token_hash_corruption_is_rejected(tmp_path: Path) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    joined["prefix_token_ids"] = [12, 99]

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)
    assert exc_info.value.code == "state_bank.token_hash"


def test_candidate_rejects_selected_sites_outside_enabled_objectives(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    candidate = _joined_event(rollouts[0], reviews[0])["candidates"][0]
    candidate["selected_sites"].append(
        {"candidate_token_offset": 2, "intended_token_type": "schema"}
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankCandidate.from_mapping(candidate, field="candidate")

    assert exc_info.value.code == "state_bank.selected_site_scope"


def test_positive_candidate_metadata_retains_unknown_geometry_for_profile_validation(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    candidate = _joined_event(rollouts[0], reviews[0])["candidates"][0]
    candidate.update(
        {
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 2],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
            ],
        }
    )

    parsed = StateBankCandidate.from_mapping(candidate, field="candidate")

    assert parsed.geometry_review_status == "unknown"
    assert parsed.geometry_eligible is False
    assert parsed.owner_resolution_interval == (0, 2)


def test_entity_transition_may_use_trusted_full_row_geometry_without_coordinate_objective(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    candidate = _joined_event(rollouts[0], reviews[0])["candidates"][0]
    candidate.update(
        {
            "geometry_review_status": "trusted",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 2],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
            ],
        }
    )

    parsed = StateBankCandidate.from_mapping(candidate, field="candidate")

    assert parsed.geometry_review_status == "trusted"
    assert parsed.geometry_eligible is False


def test_entity_transition_can_resolve_before_coordinates_with_ambiguous_geometry(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    candidate = _joined_event(rollouts[0], reviews[0])["candidates"][0]
    candidate.update(
        {
            "geometry_review_status": "ambiguous",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 1],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"}
            ],
        }
    )

    parsed = StateBankCandidate.from_mapping(candidate, field="candidate")

    assert parsed.geometry_review_status == "ambiguous"
    assert parsed.owner_resolution_interval == (0, 1)


def test_token_identity_rejects_coordinate_token_declared_as_description(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    positive = joined["candidates"][0]
    positive.update(
        {
            "token_ids": [20, 1500, 22],
            "token_ids_sha256": token_ids_sha256((20, 1500, 22)),
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 2],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "desc_text"},
            ],
        }
    )
    joined["coordinate_boundary_eligible"] = False
    event = StateBankEvent.from_mapping(joined)

    with pytest.raises(ArtifactContractError) as exc_info:
        validate_state_bank_token_identity(
            SimpleNamespace(records=(event,)),
            SimpleNamespace(
                im_end_token_ids=(5,),
                coordinate_token_ids=tuple(range(1000, 2000)),
            ),
        )

    assert exc_info.value.code == "state_bank.entity_transition_geometry_untrusted"


def test_token_identity_rejects_untrusted_geometry_on_harmful_duplicate_path(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    positive = joined["candidates"][0]
    positive.update(
        {
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 1],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"}
            ],
        }
    )
    harmful = joined["candidates"][1]
    harmful.update(
        {
            "token_ids": [30, 1500, 32],
            "token_ids_sha256": token_ids_sha256((30, 1500, 32)),
            "geometry_review_status": "ambiguous",
            "owner_resolution_interval": [0, 2],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "desc_text"},
            ],
        }
    )
    joined["coordinate_boundary_eligible"] = False
    event = StateBankEvent.from_mapping(joined)

    with pytest.raises(ArtifactContractError) as exc_info:
        validate_state_bank_token_identity(
            SimpleNamespace(records=(event,)),
            SimpleNamespace(
                im_end_token_ids=(5,),
                coordinate_token_ids=tuple(range(1000, 2000)),
            ),
        )

    assert exc_info.value.code == "state_bank.entity_transition_geometry_untrusted"


def test_entity_transition_rejects_untrusted_coordinate_owner_resolution(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    positive = joined["candidates"][0]
    positive.update(
        {
            "geometry_review_status": "ambiguous",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "owner_resolution_interval": [0, 2],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "schema"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
            ],
        }
    )
    joined["coordinate_boundary_eligible"] = False

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(joined)

    assert exc_info.value.code == "state_bank.entity_transition_geometry_untrusted"


def test_diagnostic_candidate_rejects_selected_gradient_sites(tmp_path: Path) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    candidate = _joined_event(rollouts[0], reviews[0])["candidates"][0]
    candidate.update(
        {
            "role": "diagnostic",
            "harmful_kind": None,
            "physical_owner_id": None,
            "coverage_status": "unknown",
            "entity_review_status": "unknown",
            "geometry_review_status": "unknown",
            "entity_eligible": False,
            "geometry_eligible": False,
            "owner_resolution_interval": None,
            "coordinate_decision": None,
        }
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankCandidate.from_mapping(candidate, field="candidate")

    assert exc_info.value.code == "state_bank.selected_site_scope"


def test_bound_token_identity_rejects_mislabeled_premature_terminal(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    joined = _joined_event(rollouts[0], reviews[0])
    harmful = next(
        candidate
        for candidate in joined["candidates"]
        if candidate["role"] == "harmful"
    )
    harmful.update(
        {
            "harmful_kind": "premature_terminal",
            "token_ids": [99],
            "token_ids_sha256": token_ids_sha256((99,)),
            "physical_owner_id": None,
            "owner_resolution_interval": None,
            "coordinate_decision": None,
            "geometry_eligible": False,
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "schema"}
            ],
        }
    )
    positive = next(
        candidate
        for candidate in joined["candidates"]
        if candidate["role"] == "positive"
    )
    positive.update(
        {
            "geometry_review_status": "unknown",
            "geometry_eligible": False,
            "coordinate_decision": None,
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"}
            ],
        }
    )
    joined["coordinate_boundary_eligible"] = False
    event = StateBankEvent.from_mapping(joined)

    with pytest.raises(ArtifactContractError) as exc_info:
        validate_state_bank_token_identity(
            SimpleNamespace(records=(event,)),
            SimpleNamespace(
                im_end_token_ids=(5,),
                coordinate_token_ids=tuple(range(1000)),
            ),
        )

    assert exc_info.value.code == "state_bank.terminal_token_identity"


@pytest.mark.parametrize(
    ("status_field", "eligible_field"),
    [
        ("entity_review_status", "entity_eligible"),
        ("geometry_review_status", "geometry_eligible"),
    ],
)
def test_unknown_axis_cannot_receive_direct_gradient(
    tmp_path: Path, status_field: str, eligible_field: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    reviews[0]["candidates"][0][status_field] = "unknown"
    reviews[0]["candidates"][0][eligible_field] = True

    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(_joined_event(rollouts[0], reviews[0]))
    expected = (
        "state_bank.entity_unknown_eligible"
        if eligible_field == "entity_eligible"
        else "state_bank.geometry_unknown_eligible"
    )
    assert exc_info.value.code == expected


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda rollouts, reviews: rollouts[0]["candidates"][0][
                "generation_provenance"
            ].update(mode="greedy", temperature=0.0),
            "state_bank.greedy_harmful_identity",
        ),
        (
            lambda rollouts, reviews: rollouts[0]["candidates"][0][
                "generation_provenance"
            ].update(prompt_token_ids_sha256="f" * 64),
            "state_bank.candidate_prompt_provenance",
        ),
        (
            lambda rollouts, reviews: reviews[0].update(prefix_covered_owner_proofs=[]),
            "state_bank.prefix_coverage_resolved",
        ),
        (
            lambda rollouts, reviews: reviews[0]["candidates"][0][
                "coordinate_decision"
            ].update(owner_id="entity-covered"),
            "state_bank.geometry_same_owner",
        ),
        (
            lambda rollouts, reviews: reviews[0]["candidates"][0].update(
                owner_resolution_interval=[1, 2]
            ),
            "state_bank.owner_interval_start",
        ),
    ],
)
def test_adversarial_candidate_provenance_owner_and_interval_contracts(
    tmp_path: Path, mutate, expected_code: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    mutate(rollouts, reviews)
    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(_joined_event(rollouts[0], reviews[0]))
    assert exc_info.value.code == expected_code


def test_coordinate_evidence_requires_ordered_accepted_prefix_then_first_wrong(
    tmp_path: Path,
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    decision = reviews[0]["candidates"][0]["coordinate_decision"]
    decision["observations"] = [
        {
            **decision["observations"][0],
            "actual_coordinate_value": 500,
            "acceptable_coordinate_values": [490],
        },
        {
            **decision["observations"][0],
            "coordinate": "y1",
            "tolerance_axis": "vertical",
            "candidate_token_offset": 2,
            "actual_coordinate_value": 600,
            "acceptable_coordinate_values": [590],
        },
    ]
    reviews[0]["candidates"][0]["selected_sites"] = [
        {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
        {"candidate_token_offset": 2, "intended_token_type": "coordinate"},
    ]
    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(_joined_event(rollouts[0], reviews[0]))
    assert exc_info.value.code == "state_bank.coordinate_earlier_wrong"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            {"acceptable_coordinate_values": []},
            "state_bank.coordinate_acceptable_empty",
        ),
        (
            {"acceptable_coordinate_values": [490, 490]},
            "state_bank.coordinate_acceptable_duplicate",
        ),
        ({"acceptable_coordinate_values": [1000]}, "state_bank.coordinate_range"),
        ({"acceptable_coordinate_values": [True]}, "state_bank.nonnegative_int"),
        ({"tolerance_axis": "vertical"}, "state_bank.coordinate_axis"),
    ],
)
def test_coordinate_acceptable_set_and_axis_validation(
    tmp_path: Path, mutation: dict, code: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    reviews[0]["candidates"][0]["coordinate_decision"]["observations"][0].update(
        mutation
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        StateBankEvent.from_mapping(_joined_event(rollouts[0], reviews[0]))
    assert exc_info.value.code == code


def test_physical_entity_aliases_are_preserved_without_extra_owner_weight(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    alias_rollout = copy.deepcopy(rollouts[0]["candidates"][0])
    alias_rollout["candidate_id"] = "positive-a-alias"
    alias_rollout["token_ids"] = [40]
    alias_rollout["token_ids_sha256"] = token_ids_sha256([40])
    rollouts[0]["candidates"].append(alias_rollout)
    alias_review = copy.deepcopy(reviews[0]["candidates"][0])
    alias_review.update(
        candidate_id="positive-a-alias",
        geometry_review_status="unknown",
        geometry_eligible=False,
        coordinate_decision=None,
        owner_resolution_interval=[0, 1],
        selected_sites=[
            {"candidate_token_offset": 0, "intended_token_type": "desc_text"}
        ],
    )
    reviews[0]["candidates"].append(alias_review)
    _assemble(
        tmp_path,
        checkpoint_identity,
        prompt_identity_sha256,
        rollouts=rollouts,
        reviews=reviews,
    )
    loaded = load_state_bank(
        tmp_path / "bank" / STATE_BANK_MANIFEST_NAME,
        expected_source_checkpoint=checkpoint_identity,
        expected_prompt_identity_sha256=prompt_identity_sha256,
    )
    positives = [
        candidate
        for candidate in loaded.records[0].candidates
        if candidate.role == "positive"
    ]
    assert len(positives) == 2
    assert {candidate.physical_owner_id for candidate in positives} == {"entity-a"}


def test_image_grouped_split_rejects_id_and_content_alias_leakage(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    second_rollouts, second_reviews, _ = synthetic_inputs(
        tmp_path, event_id="synthetic-event-2", image_id=43, split="eval"
    )
    rollouts.extend(second_rollouts)
    reviews.extend(second_reviews)

    with pytest.raises(ArtifactContractError) as exc_info:
        _assemble(
            tmp_path,
            checkpoint_identity,
            prompt_identity_sha256,
            rollouts=rollouts,
            reviews=reviews,
        )
    assert exc_info.value.code == "state_bank.image_content_split_leakage"


def test_blind_cohort_is_rejected_during_assembly(
    tmp_path: Path,
    checkpoint_identity,
    prompt_identity_sha256: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_reserved_id = 424242
    monkeypatch.setattr(
        state_bank_module,
        "BLIND_IMAGE_IDS",
        frozenset({synthetic_reserved_id}),
    )
    rollouts, reviews, _ = synthetic_inputs(
        tmp_path,
        image_id=f"{synthetic_reserved_id:012d}",
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        _assemble(
            tmp_path,
            checkpoint_identity,
            prompt_identity_sha256,
            rollouts=rollouts,
            reviews=reviews,
        )
    assert exc_info.value.code == "state_bank.blind_cohort"


def test_loader_requires_complete_blind_contract(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    _assemble(tmp_path, checkpoint_identity, prompt_identity_sha256)
    manifest_path = tmp_path / "bank" / STATE_BANK_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text())
    payload["blind_image_ids"] = payload["blind_image_ids"][:-1]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactContractError) as exc_info:
        load_state_bank(
            manifest_path,
            expected_source_checkpoint=checkpoint_identity,
            expected_prompt_identity_sha256=prompt_identity_sha256,
        )
    assert exc_info.value.code == "state_bank.blind_cohort_contract"


def test_explicit_rejected_review_is_counted_not_reclassified(
    tmp_path: Path, checkpoint_identity, prompt_identity_sha256: str
) -> None:
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    rejected_rollouts, _rejected_reviews, _ = synthetic_inputs(
        tmp_path, event_id="synthetic-rejected", image_id=44
    )
    rollouts.extend(rejected_rollouts)
    reviews.append(
        {
            "event_id": "synthetic-rejected",
            "admission_status": "rejected",
            "rejection_reason": "explicit_human_review_rejection",
        }
    )
    manifest = _assemble(
        tmp_path,
        checkpoint_identity,
        prompt_identity_sha256,
        rollouts=rollouts,
        reviews=reviews,
    )
    assert manifest.record_count == 1
    assert dict(manifest.rejection_reasons) == {"explicit_human_review_rejection": 1}


def _joined_event(rollout: dict, review: dict) -> dict:
    reviews_by_id = {
        candidate["candidate_id"]: candidate for candidate in review["candidates"]
    }
    candidates = [
        {**candidate, **reviews_by_id[candidate["candidate_id"]]}
        for candidate in rollout["candidates"]
    ]
    return {
        **{key: value for key, value in rollout.items() if key != "candidates"},
        "physical_entities": review["physical_entities"],
        "prefix_object_row_count": review["prefix_object_row_count"],
        "prefix_coverage_status": review["prefix_coverage_status"],
        "prefix_covered_owner_proofs": review["prefix_covered_owner_proofs"],
        "entity_transition_eligible": review["entity_transition_eligible"],
        "coordinate_boundary_eligible": review["coordinate_boundary_eligible"],
        "candidates": candidates,
        "review_provenance": review["review_provenance"],
    }
