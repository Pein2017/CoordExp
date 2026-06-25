from __future__ import annotations

from types import MappingProxyType

import pytest
import torch

from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.packing_offsets import (
    PackedSegmentOffset,
    build_packed_segment_offsets,
    shift_teacher_forcing_target_ir,
)
from src.training.teacher_forcing.roles import TokenRole


def _sample(
    sample_id: str,
    *,
    length: int | None = None,
    input_ids: tuple[int, ...] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {"sample_id": sample_id}
    if length is not None:
        row["length"] = int(length)
    if input_ids is not None:
        row["input_ids"] = tuple(input_ids)
    return row


def _atom() -> SupervisionAtom:
    return SupervisionAtom(
        batch_index=0,
        logit_position=2,
        target_position=3,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101, 102}),
        selected_token_id=101,
        latent_valid_token_ids=frozenset({101, 102}),
        coverage_target_weights=MappingProxyType({101: 0.75, 102: 0.25}),
        loss_tags=frozenset({"hard_sft"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={
            "target_position": 3,
            "logit_position": 2,
            "token_positions": (1, 3),
            "coord_label_positions": [4, 5, 6, 7],
            "branch_position": 8,
            "continuation_token_ids": (201, 202),
            "stop_token_id": 203,
            "bbox_positive_area_valid_token_ids": (301, 302),
            "bbox_positive_area_invalid_token_ids": frozenset({303}),
            "selected_token_id": 101,
        },
    )


def test_build_packed_segment_offsets_returns_empty_for_non_packed_raw_batch() -> None:
    collated = {"attention_mask": torch.ones((2, 4), dtype=torch.long)}

    assert build_packed_segment_offsets(
        [
            _sample("sample-a", length=4),
            _sample("sample-b", length=4),
        ],
        collated,
    ) == ()


def test_build_packed_segment_offsets_uses_row_local_lengths_and_attention_counts() -> None:
    raw_batch = [
        [
            _sample("sample-a", length=2),
            _sample("sample-b", input_ids=(11, 12, 13)),
        ],
        [
            _sample("sample-c", length=1),
        ],
    ]
    collated = {
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    }

    assert build_packed_segment_offsets(raw_batch, collated) == (
        PackedSegmentOffset(
            sample_id="sample-a",
            packed_row_index=0,
            segment_index=0,
            token_start=0,
            token_end=2,
        ),
        PackedSegmentOffset(
            sample_id="sample-b",
            packed_row_index=0,
            segment_index=1,
            token_start=2,
            token_end=5,
        ),
        PackedSegmentOffset(
            sample_id="sample-c",
            packed_row_index=1,
            segment_index=0,
            token_start=0,
            token_end=1,
        ),
    )


def test_build_packed_segment_offsets_rejects_duplicate_sample_ids() -> None:
    raw_batch = [
        [_sample("duplicate", length=2)],
        [_sample("duplicate", length=1)],
    ]
    collated = {"attention_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long)}

    with pytest.raises(ValueError, match="duplicate.*sample_id"):
        build_packed_segment_offsets(raw_batch, collated)


def test_build_packed_segment_offsets_rejects_attention_length_mismatch() -> None:
    raw_batch = [[_sample("sample-a", length=2), _sample("sample-b", length=1)]]
    collated = {"attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long)}

    with pytest.raises(ValueError, match="attention_mask.*length"):
        build_packed_segment_offsets(raw_batch, collated)


def test_shift_teacher_forcing_target_ir_offsets_atom_and_provenance_positions_only() -> None:
    ir = TeacherForcingTargetIR(
        schema_version=1,
        atoms=(_atom(),),
        metadata={"template_id": "compact_object_box_closed"},
    )
    offset = PackedSegmentOffset(
        sample_id="sample-a",
        packed_row_index=2,
        segment_index=1,
        token_start=10,
        token_end=20,
    )

    shifted = shift_teacher_forcing_target_ir(ir, offset)
    atom = shifted.atoms[0]

    assert atom.batch_index == 2
    assert atom.logit_position == 12
    assert atom.target_position == 13
    assert atom.valid_token_ids == frozenset({101, 102})
    assert atom.latent_valid_token_ids == frozenset({101, 102})
    assert atom.selected_token_id == 101
    assert atom.coverage_target_weights == {101: 0.75, 102: 0.25}
    assert atom.provenance["target_position"] == 13
    assert atom.provenance["logit_position"] == 12
    assert atom.provenance["token_positions"] == (11, 13)
    assert atom.provenance["coord_label_positions"] == [14, 15, 16, 17]
    assert atom.provenance["branch_position"] == 8
    assert atom.provenance["continuation_token_ids"] == (201, 202)
    assert atom.provenance["stop_token_id"] == 203
    assert atom.provenance["bbox_positive_area_valid_token_ids"] == (301, 302)
    assert atom.provenance["bbox_positive_area_invalid_token_ids"] == frozenset({303})
    assert atom.provenance["selected_token_id"] == 101

    assert ir.atoms[0].target_position == 3
    assert ir.atoms[0].provenance["coord_label_positions"] == [4, 5, 6, 7]
