from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import src.training.pipeline as pipeline_module
from src.common.errors import RuntimeContractError
from src.config.fingerprint import sha256_json
from src.rollout_calibration.planning import (
    CalibrationCandidateMetadata,
    CalibrationEventMetadata,
    CalibrationSelectedSite,
)
from src.rollout_calibration.state_bank import CoordinateDecision
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


def test_joint_stream_keeps_both_objective_families_in_every_planned_step() -> None:
    entity_step = _micro_step(_single_family_metadata("entity"))
    coordinate_step = _micro_step(_single_family_metadata("coordinate"))
    schedule = SimpleNamespace(
        resolved_max_steps=2,
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

    assert len(stream) == 4
    for offset in (0, 2):
        window = stream[offset : offset + 2]
        assert any(
            item.calibration_metadata.entity_transition_eligible for item in window
        )
        assert any(
            item.calibration_metadata.coordinate_boundary_eligible for item in window
        )


def test_joint_stream_rejects_one_slot_without_a_dual_eligible_event() -> None:
    schedule = SimpleNamespace(
        resolved_max_steps=1,
        runtime_batch=SimpleNamespace(
            world_size=1,
            effective_batch_size=1,
            resolved_grad_accum_steps=1,
        ),
    )
    with pytest.raises(RuntimeContractError, match="effective batch cannot contain"):
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
        coordinate="x1",
        tolerance_axis="horizontal",
        candidate_token_offset=0,
        actual_wrong_coordinate_value=1,
        acceptable_coordinate_values=(0,),
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
