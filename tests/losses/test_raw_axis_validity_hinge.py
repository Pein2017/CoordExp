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


def test_current_runner_preserves_box_segment_and_ddp_gradient_normalization():
    # Unequal box/token counts, repeated object names across packed examples,
    # plus an empty-box segment. Compare gradients to independent per-box math.
    segments = (PackedSegment(0, 0, 0, 'a', 0, 5),
                PackedSegment(0, 1, 1, 'b', 5, 14),
                PackedSegment(0, 2, 2, 'c', 14, 16))
    atoms = (*_box_atoms(0, 0, 'a', 'same', 1, (100, 100, 900, 900)),
             *_box_atoms(1, 1, 'b', 'same', 6, (100, 100, 900, 900)),
             *_box_atoms(1, 1, 'b', 'other', 10, (100, 100, 900, 900)),
             _desc_atom(2, 2, 'c', 15))
    logits = torch.zeros((1, 16, VOCAB_SIZE), requires_grad=True)
    with torch.no_grad():
        for atom in atoms:
            if atom.coordinate_target:
                predicted = (800, 800, 200, 200)[atom.coordinate_target.slot_index]
                logits[0, atom.causal_logits_position, COORDINATE_IDS[predicted]] = 8
    context = _context(logits, segments, atoms)
    runner = LossRunner(1.0, 0.0, ('desc_text', 'schema', 'coordinate', 'eos'),
                       raw_axis_validity_hinge_weight=0.01,
                       raw_axis_validity_hinge=RawAxisValidityHingeLoss())
    def gather(local):
        # Second rank: two supervised no-box segments, contributing zero hinge.
        peer = {name: {**d, 'eligible_segment_count': 2,
                      'selected_atom_count': 2, 'context_count': 1,
                      'skipped_segment_count': 0} for name, d in local.items()}
        return (local, peer)
    plan = runner.prepare_planned_step((context.token_sequence,),
                                      denominator_gatherer=gather, world_size=2, rank=0)
    bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
    hinge = bundle.term_by_name('raw_axis_validity_hinge')
    box_losses = []
    for group in (atoms[:4], atoms[4:8], atoms[8:12]):
        positions = {key: atom.causal_logits_position for key, atom in zip(
            ('x1_position', 'y1_position', 'x2_position', 'y2_position'), group)}
        box_losses.append(raw_axis_validity_hinge(logits[0], (positions,),
            coordinate_token_ids=COORDINATE_IDS, coordinate_bin_values=range(1000),
            margin=1/999))
    reference = (box_losses[0] + (box_losses[1] + box_losses[2])/2) / 5
    torch.testing.assert_close(hinge.raw_loss, reference)
    torch.testing.assert_close(hinge.backward_contribution, 2 * 0.01 * reference)
    observed = torch.autograd.grad(hinge.backward_contribution, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(2 * 0.01 * reference, logits, retain_graph=True)[0]
    torch.testing.assert_close(observed, expected)
    assert observed.abs().sum() > 0
    assert observed[0, atoms[-1].causal_logits_position].count_nonzero() == 0
    torch.testing.assert_close(bundle.total_loss,
                               bundle.term_by_name('base_ce').raw_loss + 0.01*reference)
    artifact = runner.finalize_planned_step((bundle.to_artifact_dict(),), plan)
    assert artifact['metrics']['loss/raw_axis_validity_hinge/weighted'] == pytest.approx(float(0.01*reference.detach()))
    assert hinge.diagnostics['term_diagnostics']['complete_box_count'] == 3


def test_no_complete_boxes_contribute_connected_zero():
    for atoms, length in (((_desc_atom(0, 0, 'empty', 1),), 2),
                          (_box_atoms(0, 0, 'partial', 'box', 1, (100,100,900,900))[:2], 3)):
        context = _context(torch.zeros((1, length, VOCAB_SIZE), requires_grad=True),
                           (PackedSegment(0,0,0,atoms[0].example_id,0,length),), atoms)
        runner = LossRunner(1.0, 0.0, ('desc_text','schema','coordinate','eos'),
                           raw_axis_validity_hinge_weight=0.01,
                           raw_axis_validity_hinge=RawAxisValidityHingeLoss())
        plan = runner.prepare_planned_step((context.token_sequence,))
        bundle = runner.compute_micro_step(context, plan, local_micro_step_index=0)
        loss = bundle.term_by_name('raw_axis_validity_hinge').backward_contribution
        assert loss.item() == 0
        loss.backward()
        assert context.logits.grad is not None
        assert context.logits.grad.count_nonzero() == 0


def test_axis_config_opt_in_omission_and_invalid_values():
    from src.config.loader import load_train_config
    from src.config.models import RawAxisValidityHingeLossConfig
    from pydantic import ValidationError
    config = load_train_config('configs/train/geo_sorted_xy/untied_axis.yaml').config
    runner = LossRunner.from_config(config.losses)
    assert runner.raw_axis_validity_hinge_weight == 0.01
    assert runner.raw_axis_validity_hinge.margin == pytest.approx(1/999)
    assert not config.model.special_token_embeddings.tie_word_embeddings
    old = load_train_config('configs/train/geo_sorted_xy/untied.yaml').config
    assert 'raw_axis_validity_hinge' not in {b.name for b in LossRunner.from_config(old.losses).active_bindings()}
    zero = config.losses.model_copy(update={"auxiliary": config.losses.auxiliary.model_copy(update={
        "raw_axis_validity_hinge": RawAxisValidityHingeLossConfig(weight=0)
    })})
    assert LossRunner.from_config(zero).raw_axis_validity_hinge is None
    for fields in ({'weight': -1}, {'weight': float('nan')}, {'weight': .01, 'margin': -1}):
        with pytest.raises(ValidationError):
            RawAxisValidityHingeLossConfig(**fields)


def test_axis_math_stays_fp32_under_bf16_autocast():
    logits = torch.zeros((4, VOCAB_SIZE), requires_grad=True)
    box = dict(zip(('x1_position','y1_position','x2_position','y2_position'), range(4)))
    kwargs = dict(coordinate_token_ids=COORDINATE_IDS,
                  coordinate_bin_values=range(1000), margin=1/999)
    expected = raw_axis_validity_hinge(logits, (box,), **kwargs)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = raw_axis_validity_hinge(logits, (box,), **kwargs)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert logits.grad.abs().sum() > 0
