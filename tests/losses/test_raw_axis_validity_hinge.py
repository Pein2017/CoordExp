from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.common.errors import LossContractError
from src.coordinate_targets import CoordinateLossTarget
from src.losses.context import LossContext
from src.losses.runner import LossRunner
from src.losses.raw_axis_validity_hinge import (
    RawAxisValidityHingeLoss,
    raw_axis_validity_hinge,
)
from src.losses.vocab import TokenVocabularyGroups
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


COORDINATE_IDS = tuple(range(5, 1005))
VOCAB_SIZE = 1008


def test_formula_uses_coordinate_vocab_only_and_has_correctly_directed_gradient() -> (
    None
):
    logits = torch.zeros((4, VOCAB_SIZE), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for row, bin_value in enumerate((800, 800, 200, 200)):
            logits[row, COORDINATE_IDS[bin_value]] = 8.0
        logits[:, 0] = 100.0
    box = {
        "x1_position": 0,
        "y1_position": 1,
        "x2_position": 2,
        "y2_position": 3,
    }

    before = raw_axis_validity_hinge(
        logits,
        (box,),
        coordinate_token_ids=COORDINATE_IDS,
        coordinate_bin_values=tuple(range(1000)),
        margin=1.0 / 999.0,
    )
    before.backward()

    assert torch.isfinite(before) and before.item() > 0.0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[:, 0].abs().sum().item() == 0.0
    updated = (logits.detach() - 10.0 * logits.grad).requires_grad_(True)
    after = raw_axis_validity_hinge(
        updated,
        (box,),
        coordinate_token_ids=COORDINATE_IDS,
        coordinate_bin_values=tuple(range(1000)),
        margin=1.0 / 999.0,
    )
    assert after.item() < before.item()


def test_loss_means_boxes_within_segment_then_segments_equally() -> None:
    atoms: list[TokenAtom] = []
    logits = torch.zeros((1, 44, VOCAB_SIZE), dtype=torch.float32)
    segments = (
        PackedSegment(0, 0, 0, "one-box", 0, 5),
        PackedSegment(0, 1, 1, "nine-boxes", 5, 42),
        PackedSegment(0, 2, 2, "no-box", 42, 44),
    )
    atoms.extend(
        _box_atoms(
            segment_index=0,
            example_index=0,
            example_id="one-box",
            object_id="violated",
            target_start=1,
            bbox=(800, 800, 801, 801),
        )
    )
    for box_index in range(9):
        start = 6 + 4 * box_index
        atoms.extend(
            _box_atoms(
                segment_index=1,
                example_index=1,
                example_id="nine-boxes",
                object_id=f"valid-{box_index}",
                target_start=start,
                bbox=(100, 100, 900, 900),
            )
        )
    atoms.append(_desc_atom(2, 2, "no-box", 43))
    for atom in atoms:
        if atom.coordinate_target is None:
            logits[0, atom.causal_logits_position, 1007] = 8.0
            continue
        slot = atom.coordinate_target.slot_index
        predicted_bin = (
            (800, 800, 200, 200)[slot]
            if atom.object_id == "violated"
            else atom.coordinate_target.bbox[slot]
        )
        logits[0, atom.causal_logits_position, COORDINATE_IDS[predicted_bin]] = 8.0
    context = _context(logits.requires_grad_(), segments, tuple(atoms))

    result = RawAxisValidityHingeLoss(margin=1.0 / 999.0).per_segment_loss(context)

    assert result.complete_box_count == 10
    assert result.zero_box_segment_count == 1
    assert result.eligible_segment_count == 3
    assert result.segment_losses[0].item() > 0.4
    assert result.segment_losses[1].item() == pytest.approx(0.0)
    assert result.segment_losses[2].item() == pytest.approx(0.0)
    torch.testing.assert_close(
        result.segment_losses.mean(), result.segment_losses[0] / 3
    )


def test_grouping_uses_object_identity_and_skips_only_incomplete_groups() -> None:
    segments = (PackedSegment(0, 0, 0, "same-geometry", 0, 11),)
    first = _box_atoms(
        segment_index=0,
        example_index=0,
        example_id="same-geometry",
        object_id="object-a",
        target_start=1,
        bbox=(100, 100, 900, 900),
    )
    second = _box_atoms(
        segment_index=0,
        example_index=0,
        example_id="same-geometry",
        object_id="object-b",
        target_start=5,
        bbox=(100, 100, 900, 900),
    )
    incomplete = _box_atoms(
        segment_index=0,
        example_index=0,
        example_id="same-geometry",
        object_id="partial",
        target_start=9,
        bbox=(100, 100, 900, 900),
    )[:2]
    atoms = first + second + incomplete
    logits = _target_aligned_logits(11, atoms)

    result = RawAxisValidityHingeLoss(margin=1.0 / 999.0).per_segment_loss(
        _context(logits, segments, atoms)
    )

    assert result.complete_box_count == 2
    assert result.incomplete_box_count == 1
    assert result.coordinate_atom_count == 10
    assert result.segment_losses.item() == pytest.approx(0.0)


def test_desc_only_segment_contributes_autograd_connected_zero_through_runner() -> None:
    atom = _desc_atom(0, 0, "no-box", 1)
    context = _context(
        torch.zeros((1, 2, VOCAB_SIZE), requires_grad=True),
        (PackedSegment(0, 0, 0, "no-box", 0, 2),),
        (atom,),
    )
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )

    bundle = runner.compute((context,))
    hinge = bundle.term_by_name("raw_axis_validity_hinge")

    assert hinge.raw_loss.item() == 0.0
    assert hinge.denominator.eligible_segment_count == 1
    assert hinge.denominator.complete_box_count == 0
    assert hinge.denominator.incomplete_box_count == 0
    assert hinge.denominator.zero_box_segment_count == 1
    bundle.total_loss.backward()
    assert context.logits.grad is not None
    assert torch.isfinite(context.logits.grad).all()


def test_incomplete_only_segment_streams_zero_and_reports_incomplete_group() -> None:
    atoms = _box_atoms(
        segment_index=0,
        example_index=0,
        example_id="partial-only",
        object_id="partial",
        target_start=1,
        bbox=(100, 100, 900, 900),
    )[:2]
    context = _context(
        _target_aligned_logits(3, atoms).requires_grad_(),
        (PackedSegment(0, 0, 0, "partial-only", 0, 3),),
        atoms,
    )
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )

    plan = runner.prepare_planned_step((context.token_sequence,))
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    hinge = bundle.term_by_name("raw_axis_validity_hinge")

    assert hinge.raw_loss.item() == 0.0
    assert hinge.denominator.complete_box_count == 0
    assert hinge.denominator.incomplete_box_count == 1
    assert hinge.denominator.zero_box_segment_count == 1
    assert artifact["metrics"]["loss/raw_axis_validity_hinge"] == 0.0
    assert artifact["metrics"]["loss/raw_axis_validity_hinge/incomplete_box_count"] == 1.0
    bundle.total_loss.backward()
    assert context.logits.grad is not None
    assert torch.isfinite(context.logits.grad).all()


def test_unsupervised_context_preserves_geometry_and_runner_eligibility_contracts() -> None:
    context = _context(
        torch.zeros((1, 2, VOCAB_SIZE), requires_grad=True),
        (PackedSegment(0, 0, 0, "unsupervised", 0, 2),),
        (),
    )

    result = RawAxisValidityHingeLoss().per_segment_loss(context)

    assert result.segment_losses.shape == (0,)
    assert result.eligible_segment_count == 0
    assert result.skipped_segment_count == 1
    result.segment_losses.sum().backward()
    assert context.logits.grad is not None
    assert torch.count_nonzero(context.logits.grad).item() == 0

    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )
    with pytest.raises(LossContractError) as exc_info:
        runner.compute((context,))
    assert exc_info.value.code == "loss.segment_balanced_zero_eligible"


def test_valid_coordinate_atom_missing_mapping_fails_closed() -> None:
    atom = replace(
        _box_atoms(
            segment_index=0,
            example_index=0,
            example_id="broken",
            object_id="object-a",
            target_start=1,
            bbox=(100, 100, 900, 900),
        )[0],
        coordinate_target=None,
    )
    context = _context(
        _target_aligned_logits(2, (atom,)),
        (PackedSegment(0, 0, 0, "broken", 0, 2),),
        (atom,),
    )

    with pytest.raises(LossContractError) as exc_info:
        RawAxisValidityHingeLoss(margin=1.0 / 999.0).per_segment_loss(context)

    assert exc_info.value.code == "loss.raw_axis_validity_hinge_target_missing"


def test_duplicate_slot_and_target_token_mismatch_fail_closed() -> None:
    atoms = _box_atoms(
        segment_index=0,
        example_index=0,
        example_id="broken",
        object_id="object-a",
        target_start=1,
        bbox=(100, 100, 900, 900),
    )
    segments = (PackedSegment(0, 0, 0, "broken", 0, 5),)
    duplicate = (
        atoms[0],
        replace(
            atoms[1],
            coordinate_target=CoordinateLossTarget(
                bbox=(100, 100, 900, 900), slot_index=0
            ),
        ),
        atoms[2],
        atoms[3],
    )
    with pytest.raises(LossContractError) as duplicate_exc:
        RawAxisValidityHingeLoss().per_segment_loss(
            _context(_target_aligned_logits(5, duplicate), segments, duplicate)
        )
    assert duplicate_exc.value.code == "loss.raw_axis_validity_hinge_duplicate_slot"

    mismatched = (replace(atoms[0], token_id=COORDINATE_IDS[101]), *atoms[1:])
    with pytest.raises(LossContractError) as mismatch_exc:
        RawAxisValidityHingeLoss().per_segment_loss(
            _context(_target_aligned_logits(5, mismatched), segments, mismatched)
        )
    assert mismatch_exc.value.code == "loss.raw_axis_validity_hinge_target_mismatch"


def test_full_and_compact_logits_are_equivalent_for_packed_segments() -> None:
    segments = (
        PackedSegment(0, 0, 0, "left", 0, 8),
        PackedSegment(0, 1, 1, "right", 8, 16),
    )
    atoms = (
        *_box_atoms(0, 0, "left", "left-box", 2, (100, 100, 900, 900)),
        *_box_atoms(1, 1, "right", "right-box", 10, (700, 700, 701, 701)),
    )
    full_logits = _target_aligned_logits(16, atoms)
    full = _context(full_logits, segments, atoms)
    kept = tuple(atom.causal_logits_position for atom in atoms)
    compact_logits = full_logits[:, kept, :].clone()
    compact = _context(
        compact_logits, segments, atoms, pack_length=16, logits_position_ids=kept
    )
    loss = RawAxisValidityHingeLoss(margin=1.0 / 999.0)

    full_result = loss.per_segment_loss(full)
    compact_result = loss.per_segment_loss(compact)

    torch.testing.assert_close(
        full_result.segment_losses, compact_result.segment_losses
    )
    assert full_result.complete_box_count == compact_result.complete_box_count == 2


def test_loss_runner_default_adds_weighted_hinge_and_metrics() -> None:
    segments = (
        PackedSegment(0, 0, 0, "box", 0, 5),
        PackedSegment(0, 1, 1, "no-box", 5, 7),
    )
    box_atoms = _box_atoms(0, 0, "box", "box-a", 1, (100, 100, 900, 900))
    atoms = (*box_atoms, _desc_atom(1, 1, "no-box", 6))
    logits = _target_aligned_logits(7, atoms)
    with torch.no_grad():
        for atom, predicted_bin in zip(box_atoms, (800, 800, 200, 200), strict=True):
            logits[0, atom.causal_logits_position].zero_()
            logits[0, atom.causal_logits_position, COORDINATE_IDS[predicted_bin]] = 8.0
    context = _context(logits.requires_grad_(), segments, atoms)
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )

    bundle = runner.compute((context,))
    hinge = bundle.term_by_name("raw_axis_validity_hinge")
    base = bundle.term_by_name("base_ce")

    assert hinge.weight == pytest.approx(0.01)
    assert hinge.denominator.eligible_segment_count == 2
    assert hinge.denominator.complete_box_count == 1
    assert hinge.denominator.zero_box_segment_count == 1
    torch.testing.assert_close(bundle.total_loss, base.raw_loss + 0.01 * hinge.raw_loss)
    assert bundle.metrics["loss/raw_axis_validity_hinge"] == pytest.approx(
        float(hinge.weighted_loss.detach())
    )
    assert bundle.metrics["loss/raw_axis_validity_hinge/complete_box_count"] == 1.0
    bundle.total_loss.backward()
    assert context.logits.grad is not None and torch.isfinite(context.logits.grad).all()


def test_streaming_global_geometry_denominator_matches_ddp_mean_scaling() -> None:
    segments = (PackedSegment(0, 0, 0, "local", 0, 5),)
    atoms = _box_atoms(0, 0, "local", "box-a", 1, (100, 100, 900, 900))
    logits = _target_aligned_logits(5, atoms)
    with torch.no_grad():
        for atom, predicted_bin in zip(atoms, (800, 800, 200, 200), strict=True):
            logits[0, atom.causal_logits_position].zero_()
            logits[0, atom.causal_logits_position, COORDINATE_IDS[predicted_bin]] = 8.0
    context = _context(logits.requires_grad_(), segments, atoms)
    runner = LossRunner(
        base_ce_weight=1.0,
        token_type_gate_weight=0.0,
        token_type_gate_groups=("desc_text", "schema", "coordinate", "eos"),
    )

    def gatherer(local_payload):
        peer = {
            name: {
                **dict(payload),
                "eligible_segment_count": 3,
                "selected_atom_count": 0,
                "skipped_segment_count": 0,
                "context_count": 3,
                **(
                    {
                        "complete_box_count": 0,
                        "incomplete_box_count": 0,
                        "zero_box_segment_count": 3,
                    }
                    if name == "raw_axis_validity_hinge"
                    else {}
                ),
            }
            for name, payload in local_payload.items()
        }
        return (local_payload, peer)

    plan = runner.prepare_planned_step(
        (context.token_sequence,),
        denominator_gatherer=gatherer,
        world_size=2,
        rank=0,
    )
    local_reference = (
        RawAxisValidityHingeLoss().per_segment_loss(context).segment_losses.sum()
    )
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    hinge = bundle.term_by_name("raw_axis_validity_hinge")

    assert plan.denominator_scope == "planned_step_global"
    assert hinge.denominator.eligible_segment_count == 4
    assert hinge.denominator.complete_box_count == 1
    assert hinge.denominator.zero_box_segment_count == 3
    torch.testing.assert_close(hinge.raw_loss, local_reference / 4.0 * 2.0)
    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    assert artifact["metrics"]["loss/raw_axis_validity_hinge/complete_box_count"] == 1.0


def _context(
    logits: torch.Tensor,
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
    *,
    pack_length: int | None = None,
    logits_position_ids: tuple[int, ...] | None = None,
) -> LossContext:
    resolved_length = int(logits.shape[1]) if pack_length is None else pack_length
    return LossContext(
        logits=logits,
        token_sequence=TokenSequence(
            pack_index=0,
            input_ids=tuple(0 for _ in range(resolved_length)),
            segments=segments,
            atoms=atoms,
            spans=(),
        ),
        vocab_groups=TokenVocabularyGroups(
            vocab_size=VOCAB_SIZE,
            desc_text=(1007,),
            schema=(0, 1),
            coordinate=COORDINATE_IDS,
            eos=(1006,),
            blocked=(2, 3, 4, 1005),
        ),
        logits_position_ids=logits_position_ids,
    )


def _box_atoms(
    segment_index: int,
    example_index: int,
    example_id: str,
    object_id: str,
    target_start: int,
    bbox: tuple[int, int, int, int],
) -> tuple[TokenAtom, ...]:
    return tuple(
        TokenAtom(
            pack_index=0,
            segment_index=segment_index,
            example_index=example_index,
            example_id=example_id,
            target_position=target_start + slot,
            token_id=COORDINATE_IDS[value],
            token_type="coordinate",
            text=f"<|coord_{value}|>",
            logical_target_position=target_start + slot,
            object_id=object_id,
            field=f"bbox[{slot}]",
            source=f"object_id:{object_id}",
            coordinate_target=CoordinateLossTarget(bbox=bbox, slot_index=slot),
        )
        for slot, value in enumerate(bbox)
    )


def _desc_atom(
    segment_index: int,
    example_index: int,
    example_id: str,
    target_position: int,
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=example_index,
        example_id=example_id,
        target_position=target_position,
        token_id=1007,
        token_type="desc_text",
        text="object",
        logical_target_position=target_position,
    )


def _target_aligned_logits(
    pack_length: int,
    atoms: tuple[TokenAtom, ...],
) -> torch.Tensor:
    logits = torch.zeros((1, pack_length, VOCAB_SIZE), dtype=torch.float32)
    for atom in atoms:
        logits[0, atom.causal_logits_position, atom.token_id] = 8.0
    return logits
