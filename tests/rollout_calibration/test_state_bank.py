from __future__ import annotations

import copy
import json
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
    reviews[0]["candidates"][0]["coordinate_decision"].update(mutation)
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
        "entity_transition_eligible": review["entity_transition_eligible"],
        "coordinate_boundary_eligible": review["coordinate_boundary_eligible"],
        "candidates": candidates,
        "review_provenance": review["review_provenance"],
    }
