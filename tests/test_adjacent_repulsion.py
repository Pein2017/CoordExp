from __future__ import annotations

import pytest
import torch

from src.trainers.teacher_forcing.adjacent_repulsion import (
    compute_adjacent_repulsion_loss,
)


def _group_logits(boxes: list[list[int]], *, coord_vocab: int = 32) -> torch.Tensor:
    logits = torch.full((len(boxes), 4, coord_vocab), -20.0, dtype=torch.float32)
    for group_idx, box in enumerate(boxes):
        for slot_idx, coord in enumerate(box):
            logits[group_idx, slot_idx, int(coord)] = 20.0
    return logits


def test_adjacent_repulsion_penalizes_near_exact_adjacent_copy() -> None:
    logits = _group_logits([[4, 4, 10, 10]])
    result = compute_adjacent_repulsion_loss(
        coord_logits_groups=logits,
        prev_target_bins=torch.tensor([[4, 4, 10, 10]], dtype=torch.long),
        has_prev_mask=torch.tensor([True], dtype=torch.bool),
        same_desc_prev_mask=torch.tensor([True], dtype=torch.bool),
        margin_ratio=0.1,
        copy_margin=0.8,
        filter_mode="same_desc",
        temperature=1.0,
        group_weights=None,
    )

    assert result.pair_count == 1
    assert result.applied_count == 1
    assert result.copy_score_mean is not None
    assert float(result.copy_score_mean.detach().item()) > 0.95
    assert float(result.loss.detach().item()) > 0.0


def test_adjacent_repulsion_partial_edge_match_stays_below_copy_margin() -> None:
    logits = _group_logits([[4, 4, 10, 14]])
    result = compute_adjacent_repulsion_loss(
        coord_logits_groups=logits,
        prev_target_bins=torch.tensor([[4, 4, 10, 10]], dtype=torch.long),
        has_prev_mask=torch.tensor([True], dtype=torch.bool),
        same_desc_prev_mask=torch.tensor([True], dtype=torch.bool),
        margin_ratio=0.1,
        copy_margin=0.8,
        filter_mode="same_desc",
        temperature=1.0,
        group_weights=None,
    )

    assert result.copy_score_mean is not None
    assert float(result.copy_score_mean.detach().item()) < 0.8
    assert float(result.loss.detach().item()) == pytest.approx(0.0, abs=1e-6)


def test_adjacent_repulsion_larger_previous_boxes_induce_wider_bands() -> None:
    logits = _group_logits([[4, 4, 10, 10], [4, 4, 22, 22]])
    result = compute_adjacent_repulsion_loss(
        coord_logits_groups=logits,
        prev_target_bins=torch.tensor(
            [[2, 2, 8, 8], [2, 2, 20, 20]], dtype=torch.long
        ),
        has_prev_mask=torch.tensor([True, True], dtype=torch.bool),
        same_desc_prev_mask=torch.tensor([True, True], dtype=torch.bool),
        margin_ratio=0.2,
        copy_margin=0.0,
        filter_mode="same_desc",
        temperature=1.0,
        group_weights=None,
    )

    # The same absolute 2-bin offset should look more copy-like for the larger
    # previous box because its edge bands are wider.
    per_group = compute_adjacent_repulsion_loss(
        coord_logits_groups=logits[1:].clone(),
        prev_target_bins=torch.tensor([[2, 2, 20, 20]], dtype=torch.long),
        has_prev_mask=torch.tensor([True], dtype=torch.bool),
        same_desc_prev_mask=torch.tensor([True], dtype=torch.bool),
        margin_ratio=0.2,
        copy_margin=0.0,
        filter_mode="same_desc",
        temperature=1.0,
        group_weights=None,
    )
    small = compute_adjacent_repulsion_loss(
        coord_logits_groups=logits[:1].clone(),
        prev_target_bins=torch.tensor([[2, 2, 8, 8]], dtype=torch.long),
        has_prev_mask=torch.tensor([True], dtype=torch.bool),
        same_desc_prev_mask=torch.tensor([True], dtype=torch.bool),
        margin_ratio=0.2,
        copy_margin=0.0,
        filter_mode="same_desc",
        temperature=1.0,
        group_weights=None,
    )

    assert result.applied_count == 2
    assert small.copy_score_mean is not None
    assert per_group.copy_score_mean is not None
    assert float(per_group.copy_score_mean.detach().item()) > float(
        small.copy_score_mean.detach().item()
    )


def test_adjacent_repulsion_global_mode_drops_same_desc_gate() -> None:
    logits = _group_logits([[4, 4, 10, 10]])
    same_desc = compute_adjacent_repulsion_loss(
        coord_logits_groups=logits,
        prev_target_bins=torch.tensor([[4, 4, 10, 10]], dtype=torch.long),
        has_prev_mask=torch.tensor([True], dtype=torch.bool),
        same_desc_prev_mask=torch.tensor([False], dtype=torch.bool),
        margin_ratio=0.1,
        copy_margin=0.8,
        filter_mode="same_desc",
        temperature=1.0,
        group_weights=None,
    )
    global_mode = compute_adjacent_repulsion_loss(
        coord_logits_groups=logits,
        prev_target_bins=torch.tensor([[4, 4, 10, 10]], dtype=torch.long),
        has_prev_mask=torch.tensor([True], dtype=torch.bool),
        same_desc_prev_mask=torch.tensor([False], dtype=torch.bool),
        margin_ratio=0.1,
        copy_margin=0.8,
        filter_mode="global",
        temperature=1.0,
        group_weights=None,
    )

    assert same_desc.applied_count == 0
    assert float(same_desc.loss.detach().item()) == pytest.approx(0.0, abs=1e-6)
    assert global_mode.applied_count == 1
    assert float(global_mode.loss.detach().item()) > 0.0
