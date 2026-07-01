from __future__ import annotations

import pytest
import torch

from src.common.errors import LossContractError
from src.config.models import RuntimeBatchResolution
from src.losses import (
    LossContext,
    PlannedStepLossSlice,
    TokenVocabularyGroups,
    reduce_segment_balanced_planned_step,
    segment_balanced_contribution,
    validate_planned_step_backend_scaling,
)
from src.packing.planner import PackedSegment
from src.supervision import TokenAtom, TokenSequence


def test_segment_balanced_reducer_weights_segment_means_equally() -> None:
    context = _context(
        (
            _segment(0, 0, 4),
            _segment(1, 4, 8),
        ),
        (
            _atom(segment_index=0, target_position=1),
            _atom(segment_index=1, target_position=5),
            _atom(segment_index=1, target_position=6),
            _atom(segment_index=1, target_position=7),
        ),
    )
    losses = torch.tensor((1.0, 9.0, 9.0, 9.0), dtype=torch.float32)

    result = reduce_segment_balanced_planned_step(
        (PlannedStepLossSlice(term_name="base_ce", context=context, per_atom_losses=losses),)
    )

    assert torch.allclose(result.loss, torch.tensor(5.0))
    assert torch.allclose(result.token_balanced_diagnostic, torch.tensor(7.0))
    assert result.eligible_segment_count == 2
    assert result.selected_atom_count == 4
    assert result.skipped_segment_count == 0
    assert result.denominator_scope == "planned_step"


def test_segment_balanced_reducer_uses_complete_planned_step_window() -> None:
    first = _context(
        (
            _segment(0, 0, 4),
            _segment(1, 4, 8),
        ),
        (
            _atom(segment_index=0, target_position=1),
            _atom(segment_index=0, target_position=2),
            _atom(segment_index=1, target_position=5),
            _atom(segment_index=1, target_position=6),
        ),
    )
    second = _context(
        (_segment(0, 0, 4),),
        (_atom(segment_index=0, target_position=1),),
        pack_index=1,
    )
    slices = (
        PlannedStepLossSlice(
            term_name="base_ce",
            context=first,
            per_atom_losses=torch.tensor((1.0, 3.0, 2.0, 6.0), dtype=torch.float32),
            local_micro_step_index=0,
        ),
        PlannedStepLossSlice(
            term_name="base_ce",
            context=second,
            per_atom_losses=torch.tensor((10.0,), dtype=torch.float32),
            local_micro_step_index=1,
        ),
    )

    result = reduce_segment_balanced_planned_step(slices)
    first_pack_local_mean = torch.tensor(((1.0 + 3.0) / 2.0 + (2.0 + 6.0) / 2.0) / 2.0)
    second_pack_local_mean = torch.tensor(10.0)
    wrong_pack_mean = (first_pack_local_mean + second_pack_local_mean) / 2.0

    assert torch.allclose(result.loss, torch.tensor((2.0 + 4.0 + 10.0) / 3.0))
    assert not torch.allclose(result.loss, wrong_pack_mean)
    assert result.eligible_segment_count == 3
    assert result.context_count == 2


def test_segment_balanced_diagnostic_is_detached_from_objective_graph() -> None:
    context = _context(
        (_segment(0, 0, 4),),
        (
            _atom(segment_index=0, target_position=1),
            _atom(segment_index=0, target_position=2),
        ),
    )
    losses = torch.tensor((1.0, 3.0), dtype=torch.float32, requires_grad=True)

    result = reduce_segment_balanced_planned_step(
        (PlannedStepLossSlice(term_name="base_ce", context=context, per_atom_losses=losses),)
    )

    assert result.loss.requires_grad
    assert not result.token_balanced_diagnostic.requires_grad


def test_segment_balanced_contributions_sum_to_planned_step_reduction() -> None:
    first = _context(
        (
            _segment(0, 0, 4),
            _segment(1, 4, 8),
        ),
        (
            _atom(segment_index=0, target_position=1),
            _atom(segment_index=1, target_position=5),
            _atom(segment_index=1, target_position=6),
        ),
    )
    second = _context(
        (
            _segment(0, 0, 4),
            _segment(1, 4, 8),
        ),
        (_atom(segment_index=1, target_position=5),),
        pack_index=1,
    )
    slices = (
        PlannedStepLossSlice(
            term_name="token_type_gate",
            context=first,
            per_atom_losses=torch.tensor((2.0, 4.0, 8.0), dtype=torch.float32),
        ),
        PlannedStepLossSlice(
            term_name="token_type_gate",
            context=second,
            per_atom_losses=torch.tensor((20.0,), dtype=torch.float32),
        ),
    )

    result = reduce_segment_balanced_planned_step(slices)
    contribution_sum = sum(
        (
            segment_balanced_contribution(item, denominator=result.denominator)
            for item in slices
        ),
        torch.tensor(0.0),
    )

    assert torch.allclose(contribution_sum, result.loss)
    assert result.denominator.eligible_segment_count == 3
    assert result.denominator.skipped_segment_count == 1


def test_segment_balanced_reducer_excludes_empty_segments_and_rejects_zero_global() -> None:
    context = _context(
        (
            _segment(0, 0, 4),
            _segment(1, 4, 8),
        ),
        (_atom(segment_index=1, target_position=5),),
    )

    result = reduce_segment_balanced_planned_step(
        (
            PlannedStepLossSlice(
                term_name="base_ce",
                context=context,
                per_atom_losses=torch.tensor((4.0,), dtype=torch.float32),
            ),
        )
    )

    assert torch.allclose(result.loss, torch.tensor(4.0))
    assert result.eligible_segment_count == 1
    assert result.skipped_segment_count == 1

    empty = _context((_segment(0, 0, 4),), ())
    with pytest.raises(LossContractError) as exc_info:
        reduce_segment_balanced_planned_step(
            (
                PlannedStepLossSlice(
                    term_name="base_ce",
                    context=empty,
                    per_atom_losses=torch.empty((0,), dtype=torch.float32),
                ),
            )
        )

    assert exc_info.value.code == "loss.segment_balanced_zero_eligible"


def test_planned_step_backend_scaling_guard_rejects_extra_accumulation_divisors() -> None:
    runtime_batch = RuntimeBatchResolution(
        world_size=2,
        effective_batch_size=8,
        resolved_grad_accum_steps=4,
    )

    receipt = validate_planned_step_backend_scaling(
        runtime_batch,
        runtime_loss_divisor=1,
        backend_loss_divisor=1,
    )

    assert receipt.normalizer_scope == "planned_step"
    assert receipt.resolved_grad_accum_steps == 4
    assert receipt.runtime_loss_divisor == 1
    assert receipt.backend_loss_divisor == 1

    with pytest.raises(LossContractError) as exc_info:
        validate_planned_step_backend_scaling(
            runtime_batch,
            runtime_loss_divisor=4,
            backend_loss_divisor=1,
        )

    assert exc_info.value.code == "loss.backend_double_scaling"

    with pytest.raises(LossContractError) as exc_info:
        validate_planned_step_backend_scaling(
            runtime_batch,
            runtime_loss_divisor=1,
            backend_loss_divisor=4,
        )

    assert exc_info.value.code == "loss.backend_double_scaling"


def test_planned_step_backend_scaling_guard_rejects_non_numeric_divisors() -> None:
    runtime_batch = RuntimeBatchResolution(
        world_size=1,
        effective_batch_size=1,
        resolved_grad_accum_steps=1,
    )

    with pytest.raises(LossContractError) as exc_info:
        validate_planned_step_backend_scaling(
            runtime_batch,
            runtime_loss_divisor="1",  # type: ignore[arg-type]
        )

    assert exc_info.value.code == "loss.backend_scaling_divisor"

    with pytest.raises(LossContractError) as exc_info:
        validate_planned_step_backend_scaling(
            runtime_batch,
            backend_loss_divisor=True,
        )

    assert exc_info.value.code == "loss.backend_scaling_divisor"


def _context(
    segments: tuple[PackedSegment, ...],
    atoms: tuple[TokenAtom, ...],
    *,
    pack_index: int = 0,
) -> LossContext:
    return LossContext(
        logits=torch.zeros((1, segments[-1].end, 8), dtype=torch.bfloat16),
        token_sequence=TokenSequence(
            pack_index=pack_index,
            input_ids=tuple(0 for _ in range(segments[-1].end)),
            segments=tuple(
                PackedSegment(
                    pack_index=pack_index,
                    segment_index=segment.segment_index,
                    example_index=segment.example_index,
                    example_id=segment.example_id,
                    start=segment.start,
                    end=segment.end,
                )
                for segment in segments
            ),
            atoms=tuple(
                TokenAtom(
                    pack_index=pack_index,
                    segment_index=atom.segment_index,
                    example_index=atom.example_index,
                    example_id=atom.example_id,
                    target_position=atom.target_position,
                    token_id=atom.token_id,
                    token_type=atom.token_type,
                    text=atom.text,
                    logical_target_position=atom.logical_target_position,
                    source=atom.source,
                )
                for atom in atoms
            ),
            spans=(),
        ),
        vocab_groups=_groups(),
    )


def _segment(segment_index: int, start: int, end: int) -> PackedSegment:
    return PackedSegment(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        start=start,
        end=end,
    )


def _atom(*, segment_index: int, target_position: int) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=f"ex-{segment_index}",
        target_position=target_position,
        token_id=7,
        token_type="desc_text",
        text="x",
        logical_target_position=target_position,
        source="unit",
    )


def _groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )
