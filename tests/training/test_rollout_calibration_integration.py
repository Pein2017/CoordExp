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
        artifact["metrics"]["calibration/rollout_entity_transition/continuation_loss"]
        > 0.0
    )
    assert (
        artifact["metrics"][
            "calibration/rollout_entity_transition/schema_description_continuation_loss"
        ]
        > 0.0
    )
    assert artifact["metrics"][
        "calibration/rollout_entity_transition/coordinate_continuation_loss"
    ] == pytest.approx(0.0)
    assert (
        0.0
        < artifact["metrics"]["calibration/rollout_site_token_type_gate/legal_mass"]
        < 1.0
    )
    assert artifact["metrics"]["calibration/rejected_record_count"] == 3.0
    assert artifact["finite_status"]["all_finite"] is True


def test_complete_action_pairwise_runner_uses_summed_candidate_scores() -> None:
    micro_step = _micro_step(_single_family_metadata("entity"))
    logits = torch.tensor(
        [[[0.0, 0.0, 2.0, -1.0, -1.0, -1.0], [0.0, 0.0, -1.0, 1.5, 0.5, -1.0]]],
        requires_grad=True,
    )
    context = rollout_calibration_loss_context(
        micro_step,
        SimpleNamespace(logits=logits, logits_position_ids=(0, 2)),
    )
    runner = RolloutCalibrationLossRunner(
        profile="complete_action_pairwise",
        entity_weight=1.0,
        entity_margin=0.0,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.0,
        gate_weight=0.1,
    )

    plan = runner.prepare_planned_step((micro_step,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    bundle.total_loss.backward()
    entity = next(
        term for term in bundle.terms if term.name == "rollout_entity_transition"
    )

    assert entity.diagnostics["score_semantics"] == "summed_complete_action_log_probability"
    assert entity.diagnostics["positive_path_count"] == 1
    assert entity.diagnostics["harmful_path_count"] == 1
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_positive_path_imitation_runner_has_no_branch_and_weights_both_terms() -> None:
    positive = CalibrationCandidateMetadata(
        candidate_id="positive-row",
        segment_index=0,
        role="positive",
        harmful_kind=None,
        physical_owner_id="owner-a",
        coverage_status="uncovered",
        entity_review_status="trusted",
        geometry_review_status="unknown",
        entity_eligible=True,
        geometry_eligible=False,
        owner_resolution_candidate_interval=(0, 3),
        owner_resolution_physical_target_interval=(1, 4),
        coordinate_decision=None,
        coordinate_physical_target_position=None,
        coordinate_physical_logits_position=None,
        selected_sites=(
            CalibrationSelectedSite("positive-row", 0, 0, "schema", 1, 0),
            CalibrationSelectedSite("positive-row", 0, 1, "desc_text", 2, 1),
            CalibrationSelectedSite("positive-row", 0, 2, "coordinate", 3, 2),
        ),
    )
    metadata = CalibrationEventMetadata(
        event_id="positive-row-event",
        image_id=42,
        split="train",
        entity_transition_eligible=False,
        coordinate_boundary_eligible=False,
        candidates=(positive,),
        selected_logits_positions=(0, 1, 2),
        positive_path_imitation_eligible=True,
        image_balanced_event_weight=2.0,
    )
    micro_step = replace(
        _micro_step(metadata),
        pack=SimpleNamespace(pack_index=0, input_ids=(0, 2, 1, 3, 0)),
    )
    logits = torch.tensor(
        [[[0.0, 2.0, -1.0, -1.0, -1.0, -1.0],
          [0.0, 1.0, -1.0, -1.0, -1.0, -1.0],
          [0.0, -1.0, 1.0, -1.0, -1.0, -1.0]]],
        requires_grad=True,
    )
    context = rollout_calibration_loss_context(
        micro_step,
        SimpleNamespace(logits=logits, logits_position_ids=(0, 1, 2)),
    )
    runner = RolloutCalibrationLossRunner(
        profile="positive_path_imitation_only",
        entity_weight=1.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.1,
    )
    plan = runner.prepare_planned_step((micro_step,))
    assert plan.enabled_terms == (
        "rollout_positive_path_imitation",
        "rollout_site_token_type_gate",
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    assert [term.name for term in bundle.terms] == list(plan.enabled_terms)
    assert bundle.terms[0].selected_count == 2
    assert bundle.terms[1].selected_count == 2
    assert bundle.terms[0].diagnostics["image_balanced_event_weight"] == 2.0
    assert bundle.terms[1].diagnostics["image_balanced_event_weight"] == 2.0
    bundle.total_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[0, 2], torch.zeros_like(logits.grad[0, 2]))
    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    assert artifact["diagnostics"]["profile"] == "positive_path_imitation_only"
    assert artifact["metrics"]["calibration/image_balanced_event_weight"] == 2.0


def test_mixed_complete_row_profile_reuses_loss_and_reports_families() -> None:
    sampled_step = _micro_step(
        _complete_row_metadata("sampled-event", source_route=False)
    )
    source_step = _micro_step(
        _complete_row_metadata("source-event", source_route=True)
    )
    runner = RolloutCalibrationLossRunner(
        profile="sampled_path_and_source_route_imitation_only",
        entity_weight=1.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.1,
    )
    plan = runner.prepare_planned_step((sampled_step, source_step))
    artifacts = []
    for index, micro_step in enumerate((sampled_step, source_step)):
        logits = torch.tensor(
            [[[0.0, 0.0, 2.0, -1.0, -1.0, -1.0]]],
            requires_grad=True,
        )
        context = rollout_calibration_loss_context(
            micro_step,
            SimpleNamespace(logits=logits, logits_position_ids=(0,)),
        )
        bundle = runner.compute_micro_step(
            context,
            plan,
            local_micro_step_index=index,
        )
        bundle.total_loss.backward()
        assert torch.isfinite(bundle.total_loss)
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        artifacts.append(bundle.to_artifact_dict())

    artifact = runner.finalize_planned_step(tuple(artifacts), plan)
    assert plan.enabled_terms == (
        "rollout_positive_path_imitation",
        "rollout_site_token_type_gate",
    )
    assert artifact["counts"]["complete_row_imitation_family_counts"] == {
        "positive_path_imitation": 1,
        "source_route_imitation": 1,
    }
    assert artifact["metrics"][
        "calibration/positive_path_imitation/admitted_event_count"
    ] == 1.0
    assert artifact["metrics"][
        "calibration/source_route_imitation/admitted_event_count"
    ] == 1.0


def test_mixed_family_counts_survive_two_rank_mean_reduction() -> None:
    runner = RolloutCalibrationLossRunner(
        profile="sampled_path_and_source_route_imitation_only",
        entity_weight=1.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.1,
    )
    sampled_step = _micro_step(
        _complete_row_metadata("sampled-rank-zero", source_route=False)
    )
    source_step = _micro_step(
        _complete_row_metadata("source-rank-one", source_route=True)
    )

    def gather(local: dict) -> tuple[dict, dict]:
        return (local, local)

    rank_artifacts = []
    for rank, micro_step in enumerate((sampled_step, source_step)):
        plan = runner.prepare_planned_step(
            (micro_step,),
            denominator_gatherer=gather,
            world_size=2,
            rank=rank,
        )
        logits = torch.tensor(
            [[[0.0, 0.0, 2.0, -1.0, -1.0, -1.0]]],
            requires_grad=True,
        )
        context = rollout_calibration_loss_context(
            micro_step,
            SimpleNamespace(logits=logits, logits_position_ids=(0,)),
        )
        bundle = runner.compute_micro_step(
            context,
            plan,
            local_micro_step_index=0,
        )
        rank_artifacts.append(
            runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
        )

    mean_reduced = {
        family: sum(
            artifact["counts"]["complete_row_imitation_family_counts"][family]
            for artifact in rank_artifacts
        )
        / 2
        for family in (
            "positive_path_imitation",
            "source_route_imitation",
        )
    }
    assert mean_reduced == {
        "positive_path_imitation": 1.0,
        "source_route_imitation": 1.0,
    }


def test_complete_row_profiles_preserve_old_isolation_and_mixed_exact_once() -> None:
    sampled_step = _micro_step(
        _complete_row_metadata("sampled-event", source_route=False)
    )
    source_step = _micro_step(
        _complete_row_metadata("source-event", source_route=True)
    )
    unrelated_step = _micro_step(_single_family_metadata("entity"))

    old_schedule = SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
    )
    assert tuple(
        build_calibration_micro_step_stream(
            (sampled_step, source_step, unrelated_step),
            old_schedule,
            profile="positive_path_imitation_only",
            rank=0,
            world_size=1,
        )
    ) == (sampled_step,)

    mixed_schedule = SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=2,
            resolved_grad_accum_steps=2,
        ),
    )
    assert tuple(
        build_calibration_micro_step_stream(
            (sampled_step, source_step, unrelated_step),
            mixed_schedule,
            profile="sampled_path_and_source_route_imitation_only",
            rank=0,
            world_size=1,
        )
    ) == (sampled_step, source_step)


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


def test_pipeline_counts_positive_path_imitation_events() -> None:
    metadata = _single_family_metadata("entity")
    metadata = replace(
        metadata,
        entity_transition_eligible=False,
        coordinate_boundary_eligible=False,
        positive_path_imitation_eligible=True,
    )
    assert pipeline_module._calibration_profile_event_count(
        (_micro_step(metadata),),
        "positive_path_imitation_only",
    ) == 1


def test_pipeline_counts_both_complete_row_imitation_families() -> None:
    sampled = _micro_step(
        _complete_row_metadata("sampled-event", source_route=False)
    )
    source = _micro_step(_complete_row_metadata("source-event", source_route=True))
    unrelated = _micro_step(_single_family_metadata("coordinate"))
    assert pipeline_module._calibration_profile_event_count(
        (sampled, source, unrelated),
        "sampled_path_and_source_route_imitation_only",
    ) == 2


def test_local_duplicate_runner_weights_pairwise_loss_and_reports_margin() -> None:
    micro_step = replace(
        _micro_step(_local_duplicate_metadata()),
        pack=SimpleNamespace(
            pack_index=0,
            input_ids=(9, 2, 0, 3, 2, 0, 3, 9),
        ),
    )
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, -1.0, -1.0, -1.0],
                [0.0, -1.0, 2.0, -1.0, -1.0],
                [0.0, -1.0, -1.0, 2.0, -1.0],
                [0.0, 1.0, -1.0, -1.0, -1.0],
                [0.0, -1.0, -1.0, 1.5, -1.0],
                [0.0, -1.0, 1.5, -1.0, -1.0],
            ]
        ],
        requires_grad=True,
    )
    context = rollout_calibration_loss_context(
        micro_step,
        SimpleNamespace(logits=logits, logits_position_ids=(0, 1, 2, 3, 4, 5)),
    )
    runner = RolloutCalibrationLossRunner(
        profile="local_duplicate_rejection_and_recovery",
        entity_weight=0.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.1,
        duplicate_weight=1.0,
        duplicate_margin=0.25,
    )

    plan = runner.prepare_planned_step((micro_step,))
    assert plan.enabled_terms == (
        "rollout_duplicate_rejection",
        "rollout_site_token_type_gate",
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    assert bundle.terms[0].diagnostics["image_balanced_event_weight"] == 0.5
    assert bundle.terms[0].diagnostics["burst_credit"] == 0.5
    assert bundle.terms[0].selected_count == 4
    bundle.total_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    assert artifact["counts"]["complete_row_imitation_family_counts"] == {
        "source_route_imitation": 0,
        "local_duplicate_rejection": 1
    }
    assert "calibration/rollout_duplicate_rejection/target_margin" in artifact[
        "metrics"
    ]


@pytest.mark.parametrize(
    "profile",
    (
        "recovery_positive_only",
        "local_duplicate_rejection_and_recovery",
        "duplicate_cleaned_imitation_only",
        "combined_duplicate_rejection_and_cleaned_imitation",
    ),
)
def test_duplicate_profiles_admit_fixed_source_rows_and_apply_source_loss(
    profile: str,
) -> None:
    source_step = _micro_step(
        _complete_row_metadata("source-preservation", source_route=True)
    )
    recovery_step = _micro_step(
        replace(
            _complete_row_metadata("recovery-positive", source_route=False),
            positive_path_imitation_eligible=False,
            recovery_positive_imitation_eligible=True,
        )
    )
    local_step = _local_duplicate_micro_step("local-duplicate", weight=1.0)
    cleaned_step = _micro_step(
        replace(
            _complete_row_metadata("cleaned-positive", source_route=False),
            positive_path_imitation_eligible=False,
            duplicate_cleaned_imitation_eligible=True,
        )
    )
    treatment_steps = {
        "recovery_positive_only": (recovery_step,),
        "local_duplicate_rejection_and_recovery": (local_step,),
        "duplicate_cleaned_imitation_only": (cleaned_step,),
        "combined_duplicate_rejection_and_cleaned_imitation": (
            local_step,
            cleaned_step,
        ),
    }[profile]
    planned_steps = (source_step, *treatment_steps)
    assert (
        pipeline_module._calibration_profile_event_count(planned_steps, profile)
        == len(planned_steps)
    )
    schedule = SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=len(planned_steps),
            resolved_grad_accum_steps=len(planned_steps),
        ),
    )

    stream = tuple(
        build_calibration_micro_step_stream(
            planned_steps,
            schedule,
            profile=profile,
            rank=0,
            world_size=1,
        )
    )
    assert "source-preservation" in {
        item.calibration_metadata.event_id for item in stream
    }

    runner = RolloutCalibrationLossRunner(
        profile=profile,
        entity_weight=1.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.0,
        duplicate_weight=float(
            profile
            in {
                "local_duplicate_rejection_and_recovery",
                "combined_duplicate_rejection_and_cleaned_imitation",
            }
        ),
        duplicate_margin=0.25,
    )
    plan = runner.prepare_planned_step(planned_steps)
    source_logits = torch.zeros((1, 1, 5), requires_grad=True)
    source_context = rollout_calibration_loss_context(
        source_step,
        SimpleNamespace(logits=source_logits, logits_position_ids=(0,)),
    )
    source_bundle = runner.compute_micro_step(
        source_context,
        plan,
        local_micro_step_index=0,
    )
    source_term = next(
        term
        for term in source_bundle.terms
        if term.name == "rollout_positive_path_imitation"
    )
    assert source_term.diagnostics["complete_row_imitation_family"] == (
        "source_route_imitation"
    )
    # With uniform five-way logits, the fixed one-unit Source row contributes
    # exactly log(5) regardless of how many recovery/cleaned rows share this
    # profile's positive-path denominator.
    assert source_term.weighted_loss.item() == pytest.approx(
        torch.log(torch.tensor(5.0)).item()
    )


def test_duplicate_credit_coefficients_preserve_burst_total_and_combined_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def constant_duplicate_loss(positive_path, duplicate_path, *, margin):
        unit = positive_path.logits.sum().float() * 0.0 + 1.0
        zero = unit * 0.0
        return SimpleNamespace(
            raw_loss=unit,
            target_margin=zero,
            positive_row_score=zero,
            duplicate_row_score=zero,
            positive_description_mean_log_probability=zero,
            positive_coordinate_mean_log_probability=zero,
            duplicate_description_mean_log_probability=zero,
            duplicate_coordinate_mean_log_probability=zero,
            positive_description_token_count=1,
            positive_coordinate_token_count=1,
            duplicate_description_token_count=1,
            duplicate_coordinate_token_count=1,
            selected_token_count=4,
        )

    def constant_positive_loss(path):
        unit = path.logits.sum().float() * 0.0 + 1.0
        zero = unit * 0.0
        return SimpleNamespace(
            raw_loss=unit,
            schema_description_loss=zero,
            coordinate_loss=zero,
            selected_token_count=1,
            schema_description_token_count=1,
            coordinate_token_count=0,
        )

    monkeypatch.setattr(
        rollout_training_module,
        "field_balanced_duplicate_rejection_loss",
        constant_duplicate_loss,
    )
    monkeypatch.setattr(
        rollout_training_module,
        "positive_path_imitation_loss",
        constant_positive_loss,
    )

    local_runner = RolloutCalibrationLossRunner(
        profile="local_duplicate_rejection_and_recovery",
        entity_weight=0.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.0,
        duplicate_weight=1.0,
        duplicate_margin=0.25,
    )

    def duplicate_total(
        steps: tuple[SupervisedMicroStep, ...],
    ) -> float:
        plan = local_runner.prepare_planned_step(steps)
        total = 0.0
        for index, step in enumerate(steps):
            context = rollout_calibration_loss_context(
                step,
                SimpleNamespace(
                    logits=torch.zeros((1, 6, 5), requires_grad=True),
                    logits_position_ids=(0, 1, 2, 3, 4, 5),
                ),
            )
            bundle = local_runner.compute_micro_step(
                context,
                plan,
                local_micro_step_index=index,
            )
            total += next(
                term.weighted_loss.item()
                for term in bundle.terms
                if term.name == "rollout_duplicate_rejection"
            )
        return total

    one_row_total = duplicate_total(
        (_local_duplicate_micro_step("one-row", weight=1.0),)
    )
    two_row_total = duplicate_total(
        (
            _local_duplicate_micro_step("two-row-entry", weight=0.5),
            _local_duplicate_micro_step("two-row-later", weight=0.5),
        )
    )
    assert one_row_total == pytest.approx(1.0)
    assert two_row_total == pytest.approx(1.0)

    local_step = _local_duplicate_micro_step("combined-local", weight=0.5)
    cleaned_step = _micro_step(
        replace(
            _complete_row_metadata("combined-cleaned", source_route=False),
            positive_path_imitation_eligible=False,
            duplicate_cleaned_imitation_eligible=True,
            image_balanced_event_weight=0.5,
        )
    )
    combined_runner = replace(
        local_runner,
        profile="combined_duplicate_rejection_and_cleaned_imitation",
        entity_weight=1.0,
    )
    combined_plan = combined_runner.prepare_planned_step((local_step, cleaned_step))
    local_bundle = combined_runner.compute_micro_step(
        rollout_calibration_loss_context(
            local_step,
            SimpleNamespace(
                logits=torch.zeros((1, 6, 5), requires_grad=True),
                logits_position_ids=(0, 1, 2, 3, 4, 5),
            ),
        ),
        combined_plan,
        local_micro_step_index=0,
    )
    cleaned_bundle = combined_runner.compute_micro_step(
        rollout_calibration_loss_context(
            cleaned_step,
            SimpleNamespace(
                logits=torch.zeros((1, 1, 5), requires_grad=True),
                logits_position_ids=(0,),
            ),
        ),
        combined_plan,
        local_micro_step_index=1,
    )
    local_contribution = next(
        term.weighted_loss.item()
        for term in local_bundle.terms
        if term.name == "rollout_duplicate_rejection"
    )
    cleaned_contribution = next(
        term.weighted_loss.item()
        for term in cleaned_bundle.terms
        if term.name == "rollout_positive_path_imitation"
    )
    assert local_contribution == pytest.approx(0.5)
    assert cleaned_contribution == pytest.approx(0.5)


def test_combined_stream_is_deterministic_family_stratified_and_fails_early() -> None:
    source_a = _micro_step(_complete_row_metadata("source-a", source_route=True))
    source_b = _micro_step(_complete_row_metadata("source-b", source_route=True))
    local_a = _micro_step(
        replace(
            _complete_row_metadata("local-a", source_route=False),
            positive_path_imitation_eligible=False,
            local_duplicate_rejection_eligible=True,
        )
    )
    local_b = _micro_step(
        replace(
            _complete_row_metadata("local-b", source_route=False),
            positive_path_imitation_eligible=False,
            local_duplicate_rejection_eligible=True,
        )
    )
    cleaned_a = _micro_step(
        replace(
            _complete_row_metadata("cleaned-a", source_route=False),
            positive_path_imitation_eligible=False,
            duplicate_cleaned_imitation_eligible=True,
        )
    )
    cleaned_b = _micro_step(
        replace(
            _complete_row_metadata("cleaned-b", source_route=False),
            positive_path_imitation_eligible=False,
            duplicate_cleaned_imitation_eligible=True,
        )
    )
    schedule = SimpleNamespace(
        resolved_max_steps=2,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=3,
            resolved_grad_accum_steps=3,
        ),
    )
    stream = tuple(
        build_calibration_micro_step_stream(
            (cleaned_b, source_b, local_b, cleaned_a, source_a, local_a),
            schedule,
            profile="combined_duplicate_rejection_and_cleaned_imitation",
            rank=0,
            world_size=1,
        )
    )
    windows = (stream[:3], stream[3:])
    assert [
        tuple(item.calibration_metadata.event_id for item in window)
        for window in windows
    ] == [
        ("source-a", "local-a", "cleaned-a"),
        ("source-b", "local-b", "cleaned-b"),
    ]
    assert all(
        any(item.calibration_metadata.source_route_imitation_eligible for item in window)
        and any(item.calibration_metadata.local_duplicate_rejection_eligible for item in window)
        and any(item.calibration_metadata.duplicate_cleaned_imitation_eligible for item in window)
        for window in windows
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        tuple(
            build_calibration_micro_step_stream(
                (source_a, local_a, local_b),
                SimpleNamespace(
                    resolved_max_steps=1,
                    runtime_batch=SimpleNamespace(
                        world_size=1,
                        effective_batch_size=3,
                        resolved_grad_accum_steps=3,
                    ),
                ),
                profile="combined_duplicate_rejection_and_cleaned_imitation",
                rank=0,
                world_size=1,
            )
        )
    assert exc_info.value.code == "training.rollout_calibration_combined_family_missing"
    assert exc_info.value.context["missing_families"] == [
        "duplicate_cleaned_imitation"
    ]


def test_local_duplicate_stream_is_global_window_stratified_across_ranks_and_fails_early() -> None:
    source_steps = (
        _micro_step(_complete_row_metadata("local-source-a", source_route=True)),
        _micro_step(_complete_row_metadata("local-source-b", source_route=True)),
    )
    local_steps = tuple(
        _local_duplicate_micro_step(f"local-duplicate-{index}", weight=1.0)
        for index in range(6)
    )
    schedule = SimpleNamespace(
        resolved_max_steps=2,
        runtime_batch=SimpleNamespace(
            world_size=2,
            effective_batch_size=4,
            resolved_grad_accum_steps=2,
        ),
    )
    streams = {
        rank: tuple(
            build_calibration_micro_step_stream(
                (*local_steps, *source_steps),
                schedule,
                profile="local_duplicate_rejection_and_recovery",
                rank=rank,
                world_size=2,
            )
        )
        for rank in range(2)
    }
    assert all(len(stream) == 4 for stream in streams.values())
    runner = RolloutCalibrationLossRunner(
        profile="local_duplicate_rejection_and_recovery",
        entity_weight=1.0,
        entity_margin=0.2,
        entity_smooth_max_temperature=0.5,
        coordinate_weight=0.0,
        coordinate_margin=0.2,
        gate_weight=0.0,
        duplicate_weight=1.0,
        duplicate_margin=0.25,
    )
    for planned_index in range(2):
        rank_windows = {
            rank: streams[rank][planned_index * 2 : (planned_index + 1) * 2]
            for rank in range(2)
        }
        global_window = [
            item
            for local_accum_index in range(2)
            for rank in range(2)
            for item in (rank_windows[rank][local_accum_index],)
        ]
        assert any(
            item.calibration_metadata.source_route_imitation_eligible
            for item in global_window
        )
        assert any(
            item.calibration_metadata.local_duplicate_rejection_eligible
            for item in global_window
        )
        gathered_denominators = [
            {
                term: rollout_training_module._local_denominator(
                    term,
                    tuple(item.calibration_metadata for item in rank_window),
                    profile=runner.profile,
                ).to_artifact_dict()
                for term in runner.enabled_terms
            }
            for rank_window in rank_windows.values()
        ]
        for rank, rank_window in rank_windows.items():
            plan = runner.prepare_planned_step(
                rank_window,
                world_size=2,
                rank=rank,
                denominator_gatherer=lambda _local: gathered_denominators,
            )
            assert plan.denominators["rollout_positive_path_imitation"].eligible_segment_count > 0
            assert plan.denominators["rollout_duplicate_rejection"].eligible_segment_count > 0

    insufficient_sources = tuple(
        _local_duplicate_micro_step(f"local-insufficient-{index}", weight=1.0)
        for index in range(7)
    ) + (source_steps[0],)
    with pytest.raises(RuntimeContractError) as exc_info:
        tuple(
            build_calibration_micro_step_stream(
                insufficient_sources,
                schedule,
                profile="local_duplicate_rejection_and_recovery",
                rank=0,
                world_size=2,
            )
        )
    assert exc_info.value.code == "training.rollout_calibration_local_duplicate_family_missing"
    assert exc_info.value.context["missing_families"] == [
        "source_route_imitation"
    ]


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


def _complete_row_metadata(
    event_id: str,
    *,
    source_route: bool,
) -> CalibrationEventMetadata:
    candidate = CalibrationCandidateMetadata(
        candidate_id=f"{event_id}-candidate",
        segment_index=0,
        role="positive",
        harmful_kind=None,
        physical_owner_id="owner-a",
        coverage_status="uncovered",
        entity_review_status="trusted",
        geometry_review_status="trusted",
        entity_eligible=True,
        geometry_eligible=True,
        owner_resolution_candidate_interval=(0, 1),
        owner_resolution_physical_target_interval=(1, 2),
        coordinate_decision=None,
        coordinate_physical_target_position=None,
        coordinate_physical_logits_position=None,
        selected_sites=(
            CalibrationSelectedSite(
                candidate_id=f"{event_id}-candidate",
                segment_index=0,
                candidate_token_offset=0,
                intended_token_type="schema",
                physical_target_position=1,
                physical_logits_position=0,
            ),
        ),
    )
    return CalibrationEventMetadata(
        event_id=event_id,
        image_id=42,
        split="train",
        entity_transition_eligible=False,
        coordinate_boundary_eligible=False,
        candidates=(candidate,),
        selected_logits_positions=(0,),
        positive_path_imitation_eligible=not source_route,
        source_route_imitation_eligible=source_route,
    )


def _local_duplicate_metadata() -> CalibrationEventMetadata:
    def candidate(
        candidate_id: str,
        *,
        segment_index: int,
        role: str,
        harmful_kind: str | None,
        owner_id: str,
        coverage_status: str,
        target_start: int,
        logits_start: int,
    ) -> CalibrationCandidateMetadata:
        return CalibrationCandidateMetadata(
            candidate_id=candidate_id,
            segment_index=segment_index,
            role=role,
            harmful_kind=harmful_kind,
            physical_owner_id=owner_id,
            coverage_status=coverage_status,
            entity_review_status="trusted",
            geometry_review_status="trusted",
            entity_eligible=True,
            geometry_eligible=True,
            owner_resolution_candidate_interval=(0, 3),
            owner_resolution_physical_target_interval=(target_start, target_start + 3),
            coordinate_decision=None,
            coordinate_physical_target_position=None,
            coordinate_physical_logits_position=None,
            selected_sites=(
                CalibrationSelectedSite(
                    candidate_id=candidate_id,
                    segment_index=segment_index,
                    candidate_token_offset=0,
                    intended_token_type="schema",
                    physical_target_position=target_start,
                    physical_logits_position=logits_start,
                ),
                CalibrationSelectedSite(
                    candidate_id=candidate_id,
                    segment_index=segment_index,
                    candidate_token_offset=1,
                    intended_token_type="desc_text",
                    physical_target_position=target_start + 1,
                    physical_logits_position=logits_start + 1,
                ),
                CalibrationSelectedSite(
                    candidate_id=candidate_id,
                    segment_index=segment_index,
                    candidate_token_offset=2,
                    intended_token_type="coordinate",
                    physical_target_position=target_start + 2,
                    physical_logits_position=logits_start + 2,
                ),
            ),
        )

    return CalibrationEventMetadata(
        event_id="local-duplicate-event",
        image_id=42,
        split="train",
        entity_transition_eligible=False,
        coordinate_boundary_eligible=False,
        candidates=(
            candidate(
                "recovery",
                segment_index=0,
                role="positive",
                harmful_kind=None,
                owner_id="owner-recovery",
                coverage_status="uncovered",
                target_start=1,
                logits_start=0,
            ),
            candidate(
                "duplicate",
                segment_index=1,
                role="harmful",
                harmful_kind="duplicate",
                owner_id="owner-duplicate",
                coverage_status="covered",
                target_start=4,
                logits_start=3,
            ),
        ),
        selected_logits_positions=(0, 1, 2, 3, 4, 5),
        duplicate_trajectory_evidence=SimpleNamespace(burst_credit=0.5),
        local_duplicate_rejection_eligible=True,
        image_balanced_event_weight=0.5,
    )


def _local_duplicate_micro_step(
    event_id: str,
    *,
    weight: float,
) -> SupervisedMicroStep:
    metadata = replace(
        _local_duplicate_metadata(),
        event_id=event_id,
        duplicate_trajectory_evidence=SimpleNamespace(burst_credit=weight),
        image_balanced_event_weight=weight,
    )
    return replace(
        _micro_step(metadata),
        pack=SimpleNamespace(
            pack_index=0,
            input_ids=(9, 2, 0, 3, 2, 0, 3, 9),
        ),
    )
