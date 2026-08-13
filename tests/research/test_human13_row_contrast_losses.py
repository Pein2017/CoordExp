from __future__ import annotations

import math

import pytest
import torch

from src.common.errors import LossContractError
from src.losses.human13_k_union import (
    image_balanced_four_coordinate_unlikelihood,
    image_balanced_owner_normalized_row_contrast,
    rectangle_valid_argmax_hinge,
    teacher_forced_mean_log_scores,
)


def test_teacher_forced_mean_scores_are_length_normalized_fp32() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [9.0, -9.0, -9.0]],
            [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float16,
    )
    targets = torch.tensor([[0, 1, 0], [0, 1, 2]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    scores = teacher_forced_mean_log_scores(logits, targets, mask)

    expected = torch.stack(
        (
            torch.log_softmax(logits[0, :2].float(), -1)[range(2), [0, 1]].mean(),
            torch.log_softmax(logits[1].float(), -1)[range(3), [0, 1, 2]].mean(),
        )
    )
    assert scores.dtype == torch.float32
    assert torch.allclose(scores, expected)


def test_owner_normalized_contrast_is_invariant_to_duplicate_alias() -> None:
    duplicate = torch.tensor([0.0])
    one_alias = image_balanced_owner_normalized_row_contrast(
        duplicate,
        torch.tensor([-1.0, -2.0]),
        candidate_event_indices=torch.tensor([0, 0]),
        candidate_owner_ids=torch.tensor([10, 20]),
        event_image_ids=torch.tensor([14038]),
        required_margin=0.25,
    )
    duplicated_alias = image_balanced_owner_normalized_row_contrast(
        duplicate,
        torch.tensor([-1.0, -1.0, -2.0]),
        candidate_event_indices=torch.tensor([0, 0, 0]),
        candidate_owner_ids=torch.tensor([10, 10, 20]),
        event_image_ids=torch.tensor([14038]),
        required_margin=0.25,
    )

    assert one_alias.denominator == duplicated_alias.denominator == 1
    assert (
        one_alias.candidate_owner_count == duplicated_alias.candidate_owner_count == 2
    )
    assert torch.allclose(one_alias.raw_loss, duplicated_alias.raw_loss)
    assert torch.allclose(
        one_alias.valid_owner_log_mass, duplicated_alias.valid_owner_log_mass
    )


def test_owner_normalized_contrast_balances_events_within_image() -> None:
    result = image_balanced_owner_normalized_row_contrast(
        torch.tensor([0.0, 2.0, 1.0]),
        torch.tensor([-1.0, -1.0, 0.0]),
        candidate_event_indices=torch.tensor([0, 1, 2]),
        candidate_owner_ids=torch.tensor([10, 11, 12]),
        event_image_ids=torch.tensor([1, 1, 2]),
        required_margin=0.0,
    )
    event_losses = torch.nn.functional.softplus(torch.tensor([1.0, 3.0, 1.0]))
    expected = torch.stack((event_losses[:2].mean(), event_losses[2:].mean())).mean()
    assert result.denominator == 2
    assert result.raw_event_count == result.consumed_event_count == 3
    assert torch.allclose(result.raw_loss, expected)


def test_four_coordinate_fallback_consumes_every_coordinate_and_balances_images() -> (
    None
):
    logits = torch.zeros(3, 4, 5)
    targets = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1]])
    logits[0].scatter_(1, targets[0].unsqueeze(1), 3.0)
    logits[1].scatter_(1, targets[1].unsqueeze(1), 1.0)
    logits[2].scatter_(1, targets[2].unsqueeze(1), 2.0)
    logits.requires_grad_()
    result = image_balanced_four_coordinate_unlikelihood(
        logits, targets, torch.tensor([1, 1, 2])
    )

    assert result.raw_event_count == 3
    assert result.selected_coordinate_count == 12
    assert result.denominator == 2
    assert result.all_finite
    result.raw_loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 60


def test_rectangle_gate_accepts_any_valid_coordinate_and_targets_global_bad() -> None:
    logits = torch.tensor(
        [
            [8.0, 1.0, 7.0, 9.0, 6.0],
            [1.0, 10.0, 8.0, 7.0, 0.0],
        ],
        requires_grad=True,
    )
    valid = torch.tensor(
        [
            [False, False, True, False, True],
            [False, True, True, False, False],
        ]
    )
    result = rectangle_valid_argmax_hinge(
        logits,
        valid,
        row_ids=torch.tensor([10, 11]),
        required_margin=0.5,
    )

    assert result.selected_valid_token_ids.tolist() == [2, 1]
    assert result.selected_invalid_token_ids.tolist() == [3, 3]
    assert result.violating_site_count == 1
    assert result.satisfied_site_count == 1
    assert math.isclose(float(result.raw_loss), 1.25)
    result.raw_loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 3] > 0 and logits.grad[0, 2] < 0
    assert torch.count_nonzero(logits.grad[1]).item() == 0


def test_rectangle_gate_groups_x2_y2_by_row() -> None:
    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 3.0],
            [4.0, 3.0, 1.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, False, False],
            [False, False, True],
            [False, True, False],
        ]
    )
    result = rectangle_valid_argmax_hinge(
        logits, valid, row_ids=torch.tensor([7, 7, 8]), required_margin=0.0
    )
    site = torch.tensor([2.0, 0.0, 1.0])
    assert result.denominator == 2
    assert torch.allclose(
        result.raw_loss, torch.stack((site[:2].mean(), site[2])).mean()
    )


@pytest.mark.parametrize(
    ("fn", "args", "match"),
    [
        (
            image_balanced_owner_normalized_row_contrast,
            (
                torch.tensor([0.0]),
                torch.tensor([]),
            ),
            "candidate",
        ),
        (
            image_balanced_four_coordinate_unlikelihood,
            (
                torch.zeros(1, 3, 4),
                torch.zeros(1, 3, dtype=torch.long),
                torch.tensor([1]),
            ),
            "four",
        ),
        (
            rectangle_valid_argmax_hinge,
            (
                torch.zeros(1, 4),
                torch.zeros(1, 4, dtype=torch.bool),
            ),
            "valid",
        ),
    ],
)
def test_new_losses_fail_closed(fn, args, match: str) -> None:
    kwargs = {}
    if fn is image_balanced_owner_normalized_row_contrast:
        kwargs = {
            "candidate_event_indices": torch.tensor([], dtype=torch.long),
            "candidate_owner_ids": torch.tensor([], dtype=torch.long),
            "event_image_ids": torch.tensor([1]),
            "required_margin": 0.0,
        }
    elif fn is rectangle_valid_argmax_hinge:
        kwargs = {"row_ids": torch.tensor([1]), "required_margin": 0.0}
    with pytest.raises(LossContractError, match=match):
        fn(*args, **kwargs)
