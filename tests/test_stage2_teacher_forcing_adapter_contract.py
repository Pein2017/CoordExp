from __future__ import annotations

import torch

from src.training.teacher_forcing.ir import TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.trainers.rollout_correction.teacher_forcing_adapter import (
    Stage2TeacherForcingObjectTarget,
    build_stage2_teacher_forcing_target_ir,
)


def _coord_token_ids() -> tuple[int, ...]:
    return tuple(range(100, 1100))


def test_rollout_correction_emits_gt_context_target_ir_without_self_context() -> None:
    input_ids = torch.tensor([[10, 20, 101, 102, 103, 104, 30, 31, 2]])
    meta = {
        "stage2_surface": "rollout_correction",
        "prompt_len": 2,
        "prefix_len": 0,
        "train_len": 7,
        "encoded_len": 9,
        "tail_desc_pos": [2, 3],
        "stop_token_id": 2,
        "bbox_groups_prefix": [],
        "bbox_groups_fn": [
            {"pos": [2, 3, 4, 5], "gt_bins": [1, 2, 3, 4]},
        ],
    }

    ir = build_stage2_teacher_forcing_target_ir(
        input_ids=input_ids,
        batch_index=0,
        meta=meta,
        coord_token_ids=_coord_token_ids(),
    )

    assert type(ir) is TeacherForcingTargetIR
    assert ir.metadata["stage2_surface"] == "rollout_correction"
    assert ir.metadata["stage2_context"] == "gt_context"
    assert len(ir.atoms) == 6
    assert all(atom.batch_index == 0 for atom in ir.atoms)
    assert all(atom.provenance.get("context") != "self_context" for atom in ir.atoms)
    coord_atoms = [atom for atom in ir.atoms if atom.selected_token_role is TokenRole.COORD]
    assert [atom.selected_token_id for atom in coord_atoms] == [101, 102, 103, 104]
    assert all(atom.loss_weight == 1.0 for atom in coord_atoms)
    text_atoms = [atom for atom in ir.atoms if atom.selected_token_role is TokenRole.TEXT]
    assert [atom.target_position for atom in text_atoms] == [4, 5]
    assert [atom.logit_position for atom in text_atoms] == [3, 4]


def test_rollout_correction_converts_rollout_fn_and_recovered_fn_provenance() -> None:
    input_ids = torch.tensor(
        [[10, 20, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 2]]
    )
    meta = {
        "stage2_surface": "rollout_correction",
        "prompt_len": 2,
        "prefix_len": 4,
        "train_len": 11,
        "encoded_len": 13,
        "rollout_context": "rollout_valid_with_fn_append",
        "fn_gt_indices_final": [4, 9],
        "recovered_gt_indices": [9],
        "fn_object_weights": [1.0, 0.25],
        "stop_token_id": 2,
        "bbox_groups_prefix": [
            {"pos": [2, 3, 4, 5], "gt_bins": [1, 2, 3, 4]},
        ],
        "bbox_groups_fn": [
            {"pos": [6, 7, 8, 9], "gt_bins": [5, 6, 7, 8], "weight": 1.0},
            {"pos": [8, 9, 10, 11], "gt_bins": [7, 8, 9, 10], "weight": 0.25},
        ],
    }

    ir = build_stage2_teacher_forcing_target_ir(
        input_ids=input_ids,
        batch_index=0,
        meta=meta,
        coord_token_ids=_coord_token_ids(),
    )

    rollout_atoms = [
        atom for atom in ir.atoms if atom.provenance.get("object_source") == "rollout"
    ]
    fn_atoms = [atom for atom in ir.atoms if atom.provenance.get("object_source") == "fn"]
    recovered_atoms = [
        atom
        for atom in ir.atoms
        if atom.provenance.get("object_source") == "recovered_fn"
    ]
    assert len(rollout_atoms) == 4
    assert len(fn_atoms) == 4
    assert len(recovered_atoms) == 4
    assert all("fn" in atom.loss_tags for atom in fn_atoms)
    assert all("recovered_fn" not in atom.loss_tags for atom in fn_atoms)
    assert all(atom.loss_weight == 0.25 for atom in recovered_atoms)
    assert all("recovered_fn" in atom.loss_tags for atom in recovered_atoms)
    assert all("recovered_fn" not in atom.loss_tags for atom in rollout_atoms + fn_atoms)


def test_rollout_correction_tail_description_positions_are_relative_to_prompt_and_prefix() -> None:
    input_ids = torch.tensor([[10, 20, 101, 102, 103, 104, 77, 78, 2]])
    meta = {
        "stage2_surface": "rollout_correction",
        "prompt_len": 2,
        "prefix_len": 4,
        "train_len": 7,
        "encoded_len": 9,
        "rollout_context": "rollout_valid_with_fn_append",
        "tail_desc_pos": [0, 1],
        "stop_token_id": 2,
        "bbox_groups_prefix": [
            {"pos": [2, 3, 4, 5], "gt_bins": [1, 2, 3, 4]},
        ],
        "bbox_groups_fn": [],
    }

    ir = build_stage2_teacher_forcing_target_ir(
        input_ids=input_ids,
        batch_index=0,
        meta=meta,
        coord_token_ids=_coord_token_ids(),
    )

    text_atoms = [atom for atom in ir.atoms if atom.selected_token_role is TokenRole.TEXT]
    assert [atom.target_position for atom in text_atoms] == [6, 7]
    assert [atom.logit_position for atom in text_atoms] == [5, 6]
    assert [atom.selected_token_id for atom in text_atoms] == [77, 78]


def test_duplicate_pseudo_and_shielded_rollout_objects_emit_zero_positive_atoms() -> None:
    input_ids = torch.tensor([[10, 20, 101, 102, 103, 104]])
    common = {
        "coord_positions": (2, 3, 4, 5),
        "gt_bins": (1, 2, 3, 4),
        "loss_weight": 1.0,
    }
    for policy in ("duplicate_certified", "pseudo_positive", "shielded"):
        ir = build_stage2_teacher_forcing_target_ir(
            input_ids=input_ids,
            batch_index=0,
            meta={
                "stage2_surface": "rollout_correction",
                "prompt_len": 2,
                "prefix_len": 4,
                "train_len": 4,
                "encoded_len": 6,
            },
            coord_token_ids=_coord_token_ids(),
            object_targets=(
                Stage2TeacherForcingObjectTarget(
                    object_source="rollout",
                    positive_policy=policy,
                    **common,
                ),
            ),
        )
        assert ir.atoms == ()


def test_promoted_pseudo_positive_rollout_object_can_emit_positive_atoms() -> None:
    input_ids = torch.tensor([[10, 20, 101, 102, 103, 104]])

    ir = build_stage2_teacher_forcing_target_ir(
        input_ids=input_ids,
        batch_index=0,
        meta={
            "stage2_surface": "rollout_correction",
            "prompt_len": 2,
            "prefix_len": 4,
            "train_len": 4,
            "encoded_len": 6,
        },
        coord_token_ids=_coord_token_ids(),
        object_targets=(
            Stage2TeacherForcingObjectTarget(
                object_source="rollout",
                positive_policy="triage_promoted",
                coord_positions=(2, 3, 4, 5),
                gt_bins=(1, 2, 3, 4),
                loss_weight=0.5,
            ),
        ),
    )

    assert len(ir.atoms) == 4
    assert all(atom.loss_weight == 0.5 for atom in ir.atoms)
    assert all(atom.provenance["positive_policy"] == "triage_promoted" for atom in ir.atoms)
