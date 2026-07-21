from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import src.training.pipeline as pipeline_module
import src.training.rollout_calibration as rollout_training_module
from src.common.errors import RuntimeContractError
from src.config.fingerprint import sha256_json
from src.rollout_calibration.planning import (
    CalibrationCandidateMetadata,
    CalibrationEventMetadata,
    CalibrationSelectedSite,
)
from src.rollout_calibration.state_bank import (
    CoordinateBoundaryObservation,
    CoordinateDecision,
    ReviewProvenance,
)
from src.rollout_calibration.state_bank import (
    CheckpointIdentity,
    StateBankManifestBinding,
)
from src.training.rollout_calibration import (
    RolloutCalibrationLossRunner,
    build_calibration_micro_step_stream,
    rollout_calibration_loss_context,
)
from src.training.supervised_trainer import SupervisedMicroStep


class _VocabGroups:
    def allowed_ids(self, token_type: str) -> tuple[int, ...]:
        return {
            "desc_text": (0, 1),
            "schema": (2,),
            "coordinate": (3, 4),
            "eos": (5,),
        }[token_type]


def test_joint_runner_uses_compact_logits_and_emits_event_balanced_metrics() -> None:
    micro_step = _micro_step(_joint_metadata())
    logits = torch.tensor(
        [[[0.0, 0.0, 2.0, -1.0, -1.0, -1.0], [0.0, 0.0, -1.0, 1.5, 0.5, -1.0]]],
        requires_grad=True,
    )
    context = rollout_calibration_loss_context(
        micro_step,
        SimpleNamespace(logits=logits, logits_position_ids=(0, 2)),
    )
    runner = RolloutCalibrationLossRunner(
        profile="joint",
        entity_weight=0.5,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.5,
        coordinate_margin=0.2,
        gate_weight=0.1,
        rejection_count=3,
    )

    plan = runner.prepare_planned_step((micro_step,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    bundle.total_loss.backward()
    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)

    assert torch.isfinite(bundle.total_loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert [term.name for term in bundle.terms] == [
        "rollout_entity_transition",
        "rollout_coordinate_boundary",
        "rollout_site_token_type_gate",
    ]
    assert artifact["diagnostics"]["normalizer"] == "event_balanced"
    assert (
        artifact["metrics"][
            "calibration/rollout_entity_transition/eligible_event_count"
        ]
        == 1.0
    )
    assert (
        artifact["metrics"]["calibration/rollout_entity_transition/ignored_event_count"]
        == 0.0
    )
    assert artifact["metrics"][
        "calibration/rollout_coordinate_boundary/target_margin"
    ] == pytest.approx(-1.0)
    assert (
        0.0
        < artifact["metrics"]["calibration/rollout_site_token_type_gate/legal_mass"]
        < 1.0
    )
    assert artifact["metrics"]["calibration/rejected_record_count"] == 3.0
    assert artifact["finite_status"]["all_finite"] is True


def test_coordinate_only_diagnostic_candidate_reaches_coordinate_and_gate_losses() -> None:
    joint = _joint_metadata()
    harmful = joint.candidates[1]
    diagnostic = replace(
        harmful,
        candidate_id="diagnostic",
        role="diagnostic",
        harmful_kind=None,
        entity_eligible=False,
        owner_resolution_candidate_interval=None,
        owner_resolution_physical_target_interval=None,
    )
    metadata = CalibrationEventMetadata(
        event_id="coordinate-only-event",
        image_id=joint.image_id,
        split=joint.split,
        entity_transition_eligible=False,
        coordinate_boundary_eligible=True,
        candidates=(diagnostic,),
        selected_logits_positions=joint.selected_logits_positions,
    )
    micro_step = _micro_step(metadata)
    logits = torch.tensor(
        [[[0.0, 0.0, -1.0, -1.0, 1.5, -1.0]]], requires_grad=True
    )
    context = rollout_calibration_loss_context(
        micro_step,
        SimpleNamespace(logits=logits, logits_position_ids=(2,)),
    )
    runner = RolloutCalibrationLossRunner(
        profile="coordinate_boundary_only",
        entity_weight=0.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=1.0,
        coordinate_margin=0.2,
        gate_weight=0.1,
    )

    plan = runner.prepare_planned_step((micro_step,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    bundle.total_loss.backward()

    assert [term.name for term in bundle.terms] == [
        "rollout_coordinate_boundary",
        "rollout_site_token_type_gate",
    ]
    assert bundle.terms[0].selected_count == 1
    assert bundle.terms[1].selected_count == 1
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_joint_stream_preserves_frozen_bank_exposure_without_duplication() -> None:
    entity_step = _micro_step(_single_family_metadata("entity"))
    coordinate_step = _micro_step(_single_family_metadata("coordinate"))
    schedule = SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=2,
            resolved_grad_accum_steps=2,
        ),
    )

    stream = tuple(
        build_calibration_micro_step_stream(
            (entity_step, coordinate_step),
            schedule,
            profile="joint",
            rank=0,
            world_size=1,
        )
    )

    assert stream == (entity_step, coordinate_step)


def test_joint_stream_allows_visible_single_family_windows_without_reweighting() -> (
    None
):
    schedule = SimpleNamespace(
        resolved_max_steps=2,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
    )
    entity_step = _micro_step(_single_family_metadata("entity"))
    coordinate_step = _micro_step(_single_family_metadata("coordinate"))
    assert tuple(
        build_calibration_micro_step_stream(
            (entity_step, coordinate_step),
            schedule,
            profile="joint",
            rank=0,
            world_size=1,
        )
    ) == (entity_step, coordinate_step)


def test_joint_stream_rejects_schedule_that_duplicates_frozen_events() -> None:
    schedule = SimpleNamespace(
        resolved_max_steps=2,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=2,
            resolved_grad_accum_steps=2,
        ),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        tuple(
            build_calibration_micro_step_stream(
                (
                    _micro_step(_single_family_metadata("entity")),
                    _micro_step(_single_family_metadata("coordinate")),
                ),
                schedule,
                profile="joint",
                rank=0,
                world_size=1,
            )
        )
    assert exc_info.value.code == "training.rollout_calibration_frozen_exposure"


def test_single_objective_gate_selects_only_that_objective_sites() -> None:
    candidate = SimpleNamespace(
        entity_eligible=True,
        geometry_eligible=True,
        harmful_kind=None,
        owner_resolution_candidate_interval=(0, 1),
        coordinate_decision=SimpleNamespace(candidate_token_offset=2),
        selected_sites=(
            SimpleNamespace(candidate_token_offset=0),
            SimpleNamespace(candidate_token_offset=2),
        ),
    )
    assert [
        site.candidate_token_offset
        for site in rollout_training_module._active_selected_sites(
            candidate, "transition_only"
        )
    ] == [0]
    assert [
        site.candidate_token_offset
        for site in rollout_training_module._active_selected_sites(
            candidate, "coordinate_boundary_only"
        )
    ] == [2]
    assert [
        site.candidate_token_offset
        for site in rollout_training_module._active_selected_sites(
            candidate, "coordinate_boundary_gate_only"
        )
    ] == [2]


def test_coordinate_boundary_gate_only_has_gate_term_only_and_nonzero_gradient() -> None:
    micro_step = _micro_step(_single_family_metadata("coordinate"))
    logits = torch.tensor(
        [[[0.0, 0.0, 2.0, -1.0, -1.0, -1.0], [0.0, 0.0, -1.0, 1.5, 0.5, -1.0]]],
        requires_grad=True,
    )
    context = rollout_calibration_loss_context(
        micro_step,
        SimpleNamespace(logits=logits, logits_position_ids=(0, 2)),
    )
    runner = RolloutCalibrationLossRunner(
        profile="coordinate_boundary_gate_only",
        entity_weight=0.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.1,
    )

    plan = runner.prepare_planned_step((micro_step,))
    assert plan.enabled_terms == ("rollout_site_token_type_gate",)
    denominator = plan.denominators["rollout_site_token_type_gate"]
    assert denominator.eligible_segment_count == 1
    assert denominator.selected_atom_count == 1

    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    assert [term.name for term in bundle.terms] == ["rollout_site_token_type_gate"]
    bundle.total_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 0]).item() == 0

    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    assert artifact["counts"]["global_eligible_event_counts"] == {
        "rollout_site_token_type_gate": 1
    }
    assert artifact["diagnostics"]["profile"] == "coordinate_boundary_gate_only"
    assert artifact["metrics"]["calibration/rollout_site_token_type_gate/legal_mass"] > 0.0


def test_pipeline_counts_coordinate_boundary_gate_only_events() -> None:
    assert pipeline_module._calibration_profile_event_count(
        (
            _micro_step(_single_family_metadata("coordinate")),
            _micro_step(_single_family_metadata("entity")),
        ),
        "coordinate_boundary_gate_only",
    ) == 1


def test_coordinate_boundary_gate_only_stream_exposes_coordinate_events_once() -> None:
    coordinate_step = _micro_step(_single_family_metadata("coordinate"))
    entity_step = _micro_step(_single_family_metadata("entity"))
    schedule = SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
    )
    assert tuple(
        build_calibration_micro_step_stream(
            (entity_step, coordinate_step),
            schedule,
            profile="coordinate_boundary_gate_only",
            rank=0,
            world_size=1,
        )
    ) == (coordinate_step,)


def test_pipeline_validates_actual_warm_start_payloads_before_loading_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = SimpleNamespace(
        base_model_path="/base",
        base_config_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        token_identity=SimpleNamespace(to_artifact_dict=lambda: {"tokens": "exact"}),
        processor_identity=SimpleNamespace(
            to_artifact_dict=lambda: {"processor": "no-resize"}
        ),
    )
    selection = SimpleNamespace(to_artifact_dict=lambda: {"selection": "v1"})
    checkpoint = CheckpointIdentity(
        adapter_fingerprint="1" * 64,
        embedding_delta_fingerprint="2" * 64,
        base_config_sha256=components.base_config_sha256,
        tokenizer_sha256=components.tokenizer_sha256,
        token_identity_sha256=sha256_json(components.token_identity.to_artifact_dict()),
        special_token_identity_sha256=sha256_json(selection.to_artifact_dict()),
        processor_identity_sha256=sha256_json(
            components.processor_identity.to_artifact_dict()
        ),
    )
    binding = StateBankManifestBinding(
        source_checkpoint_id=sha256_json(checkpoint.to_artifact_dict()),
        bank_id="9" * 64,
        source_composite_fingerprint=sha256_json(checkpoint.to_artifact_dict()),
        source_checkpoint=checkpoint,
        prompt_identity_sha256="8" * 64,
        records_sha256="7" * 64,
        record_count=1,
        split_counts={"train": 1},
        event_family_counts={"entity_transition": 1},
    )
    config = SimpleNamespace(
        rollout_calibration=SimpleNamespace(
            state_bank_manifest_path="/bank/manifest.json"
        ),
        adapter=SimpleNamespace(
            source_adapter_path="/source/adapter",
            repaired_embedding_payload_path="/source/embedding",
        ),
    )
    loaded_bank = object()
    monkeypatch.setattr(
        pipeline_module,
        "load_state_bank_manifest_binding",
        lambda path: binding,
    )
    monkeypatch.setattr(
        pipeline_module,
        "inspect_dora_adapter_payload",
        lambda path, expected_base_model_path: {"fingerprint": "1" * 64},
    )
    monkeypatch.setattr(
        pipeline_module,
        "inspect_special_token_embedding_delta_payload",
        lambda path, **kwargs: {"fingerprint": "2" * 64},
    )

    def load_bank(path: str, **kwargs: object) -> object:
        assert kwargs["expected_source_checkpoint"] == checkpoint
        assert kwargs["expected_prompt_identity_sha256"] == "8" * 64
        return loaded_bank

    monkeypatch.setattr(pipeline_module, "load_state_bank", load_bank)
    validated: list[object] = []
    monkeypatch.setattr(
        pipeline_module,
        "validate_state_bank_token_identity",
        lambda bank, token_identity: validated.append(bank),
    )

    assert (
        pipeline_module._load_bound_rollout_calibration_bank(
            config,
            components=components,
            special_token_selection=selection,
        )
        is loaded_bank
    )
    assert validated == [loaded_bank]

    monkeypatch.setattr(
        pipeline_module,
        "inspect_dora_adapter_payload",
        lambda path, expected_base_model_path: {"fingerprint": "0" * 64},
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline_module._load_bound_rollout_calibration_bank(
            config,
            components=components,
            special_token_selection=selection,
        )
    assert exc_info.value.code == (
        "training.rollout_calibration_source_checkpoint_mismatch"
    )

    config.rollout_calibration.allow_off_policy_state_bank_replay = True
    assert (
        pipeline_module._load_bound_rollout_calibration_bank(
            config,
            components=components,
            special_token_selection=selection,
        )
        is loaded_bank
    )

    components.base_config_sha256 = "a" * 64
    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline_module._load_bound_rollout_calibration_bank(
            config,
            components=components,
            special_token_selection=selection,
        )
    assert exc_info.value.code == (
        "training.rollout_calibration_off_policy_identity_mismatch"
    )
    assert "base_config_sha256" in exc_info.value.context["mismatches"]


def test_source_step_zero_surface_parity_requires_exact_copy_and_frozen_delta() -> None:
    receipt = SimpleNamespace(
        warm_start={
            "initialized_target_tensors": [],
            "ignored_source_tensors": [],
            "post_copy_equality": "pass",
        }
    )
    special = SimpleNamespace(
        shared_embed_delta=torch.nn.Parameter(torch.zeros(1), requires_grad=False),
        receipt=SimpleNamespace(delta_parameter_names=("shared_embed_delta",)),
    )
    parity = pipeline_module._validate_calibration_source_step_zero_parity(
        receipt, special_token_result=special
    )
    assert parity["status"] == "pass"
    assert parity["real_fixture_logit_parity"]["status"] == "not_run"

    receipt.warm_start["initialized_target_tensors"] = ["new.target"]
    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline_module._validate_calibration_source_step_zero_parity(
            receipt, special_token_result=special
        )
    assert exc_info.value.code == "training.rollout_calibration_source_surface_mismatch"


def test_calibration_trainable_surface_rejects_embedding_or_nonlanguage_ownership() -> (
    None
):
    special = SimpleNamespace(
        shared_embed_delta=torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    )
    valid = SimpleNamespace(
        optimizer_groups=({"group_name": "adapter.language"},),
        trainable_towers=("adapter.language",),
    )
    pipeline_module._validate_calibration_trainable_surface(
        valid, special_token_result=special
    )

    invalid = SimpleNamespace(
        optimizer_groups=(
            {"group_name": "adapter.language"},
            {"group_name": "token_embeddings"},
        ),
        trainable_towers=("adapter.language", "token_embeddings"),
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        pipeline_module._validate_calibration_trainable_surface(
            invalid, special_token_result=special
        )
    assert exc_info.value.code == "training.rollout_calibration_optimizer_surface"


def _micro_step(metadata: CalibrationEventMetadata) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=SimpleNamespace(pack_index=0, input_ids=(9, 2, 9, 3, 9)),
        encoded_examples=(),
        position_inputs=None,
        token_sequence=SimpleNamespace(atoms=()),
        vocab_groups=_VocabGroups(),
        metadata={"coordinate_token_ids": (4, 3)},
        calibration_metadata=metadata,
    )


def _joint_metadata() -> CalibrationEventMetadata:
    positive_site = CalibrationSelectedSite(
        candidate_id="positive",
        segment_index=0,
        candidate_token_offset=0,
        intended_token_type="schema",
        physical_target_position=1,
        physical_logits_position=0,
    )
    harmful_site = CalibrationSelectedSite(
        candidate_id="harmful",
        segment_index=1,
        candidate_token_offset=0,
        intended_token_type="coordinate",
        physical_target_position=3,
        physical_logits_position=2,
    )
    coordinate = CoordinateDecision(
        owner_id="owner-b",
        observations=(
            CoordinateBoundaryObservation(
                coordinate="x1",
                tolerance_axis="horizontal",
                candidate_token_offset=0,
                actual_coordinate_value=1,
                acceptable_coordinate_values=(0,),
                review_provenance=ReviewProvenance(
                    source="synthetic_unit_test",
                    reviewer="synthetic",
                    confidence="fixture_only",
                    comment="not scientific evidence",
                ),
            ),
        ),
    )
    return CalibrationEventMetadata(
        event_id="joint-event",
        image_id=42,
        split="train",
        entity_transition_eligible=True,
        coordinate_boundary_eligible=True,
        candidates=(
            CalibrationCandidateMetadata(
                candidate_id="positive",
                segment_index=0,
                role="positive",
                harmful_kind=None,
                physical_owner_id="owner-a",
                coverage_status="uncovered",
                entity_review_status="trusted",
                geometry_review_status="unknown",
                entity_eligible=True,
                geometry_eligible=False,
                owner_resolution_candidate_interval=(0, 1),
                owner_resolution_physical_target_interval=(1, 2),
                coordinate_decision=None,
                coordinate_physical_target_position=None,
                coordinate_physical_logits_position=None,
                selected_sites=(positive_site,),
            ),
            CalibrationCandidateMetadata(
                candidate_id="harmful",
                segment_index=1,
                role="harmful",
                harmful_kind="duplicate",
                physical_owner_id="owner-b",
                coverage_status="covered",
                entity_review_status="trusted",
                geometry_review_status="trusted",
                entity_eligible=True,
                geometry_eligible=True,
                owner_resolution_candidate_interval=(0, 1),
                owner_resolution_physical_target_interval=(3, 4),
                coordinate_decision=coordinate,
                coordinate_physical_target_position=3,
                coordinate_physical_logits_position=2,
                selected_sites=(harmful_site,),
            ),
        ),
        selected_logits_positions=(0, 2),
    )


def _single_family_metadata(family: str) -> CalibrationEventMetadata:
    joint = _joint_metadata()
    return CalibrationEventMetadata(
        event_id=f"{family}-event",
        image_id=joint.image_id,
        split=joint.split,
        entity_transition_eligible=family == "entity",
        coordinate_boundary_eligible=family == "coordinate",
        candidates=joint.candidates,
        selected_logits_positions=joint.selected_logits_positions,
    )
