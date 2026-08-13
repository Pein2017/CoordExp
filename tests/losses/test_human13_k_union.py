from __future__ import annotations

import math

import pytest
import torch

from src.common.errors import LossContractError
from src.losses.human13_k_union import (
    coherent_full_chain_bottleneck_hinge,
    image_balanced_duplicate_token_unlikelihood,
    owner_mean_masked_row_cross_entropy,
    prefix_free_union_detached_weight_surrogate,
    prefix_free_union_negative_log_mass,
    prefix_free_union_streaming_weights,
)


def test_owner_mean_ce_masks_tokens_and_normalizes_each_owner_first() -> None:
    logits = torch.tensor(
        (
            ((0.0, 0.0), (0.0, math.log(3.0))),
            ((0.0, 0.0), (100.0, -100.0)),
        ),
        dtype=torch.float16,
        requires_grad=True,
    )
    targets = torch.zeros((2, 2), dtype=torch.long)
    mask = torch.tensor(((True, True), (True, False)))

    result = owner_mean_masked_row_cross_entropy(logits, targets, mask)

    assert result.raw_loss.dtype == torch.float32
    assert result.numerator.item() == pytest.approx(2.5 * math.log(2.0), rel=1e-4)
    assert result.denominator == 2
    assert result.raw_loss.item() == pytest.approx(1.25 * math.log(2.0), rel=1e-4)
    assert result.eligible_owner_count == 2
    assert result.selected_token_count == 3


def test_owner_mean_ce_returns_differentiable_zero_for_zero_denominator() -> None:
    logits = torch.zeros((2, 2, 3), requires_grad=True)

    result = owner_mean_masked_row_cross_entropy(
        logits,
        torch.zeros((2, 2), dtype=torch.long),
        torch.zeros((2, 2), dtype=torch.bool),
    )
    result.raw_loss.backward()

    assert result.raw_loss.item() == 0.0
    assert result.numerator.item() == 0.0
    assert result.denominator == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_owner_mean_ce_rejects_truly_nonfinite_logits() -> None:
    logits = torch.tensor([[[0.0, float("nan")]]])

    with pytest.raises(LossContractError) as exc_info:
        owner_mean_masked_row_cross_entropy(
            logits,
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1), dtype=torch.bool),
        )

    assert exc_info.value.code == "loss.human13_nonfinite_logits"


def test_prefix_free_union_uses_summed_action_probability_once_per_image() -> None:
    logits = torch.zeros((2, 2, 2), requires_grad=True)
    targets = torch.tensor(((0, 0), (1, 0)))
    mask = torch.tensor(((True, False), (True, True)))

    result = prefix_free_union_negative_log_mass(logits, targets, mask)

    assert result.denominator == 1
    assert result.candidate_count == 2
    assert result.selected_token_count == 3
    assert result.numerator.item() == pytest.approx(math.log(4.0 / 3.0))
    assert result.raw_loss.item() == pytest.approx(math.log(4.0 / 3.0))
    assert result.candidate_weights == pytest.approx((2.0 / 3.0, 1.0 / 3.0))
    assert result.effective_owner_count == pytest.approx(1.8)


def test_prefix_free_union_returns_zero_when_image_has_no_candidates() -> None:
    logits = torch.empty((0, 2, 3), requires_grad=True)

    result = prefix_free_union_negative_log_mass(
        logits,
        torch.empty((0, 2), dtype=torch.long),
        torch.empty((0, 2), dtype=torch.bool),
    )
    result.raw_loss.backward()

    assert result.numerator.item() == 0.0
    assert result.denominator == 0
    assert result.candidate_weights == ()
    assert result.effective_owner_count == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_prefix_free_union_rejects_a_candidate_that_prefixes_another() -> None:
    logits = torch.zeros((2, 2, 3))

    with pytest.raises(LossContractError) as exc_info:
        prefix_free_union_negative_log_mass(
            logits,
            torch.tensor(((1, 0), (1, 2))),
            torch.tensor(((True, False), (True, True))),
        )

    assert exc_info.value.code == "loss.human13_candidates_not_prefix_free"


def test_streamed_union_surrogate_has_exact_reference_gradient() -> None:
    reference_scores = torch.tensor((-0.2, -1.1, -2.3), requires_grad=True)
    streamed_scores = reference_scores.detach().clone().requires_grad_(True)

    reference = -torch.logsumexp(reference_scores.float(), dim=0)
    weights = prefix_free_union_streaming_weights(streamed_scores)
    streamed = prefix_free_union_detached_weight_surrogate(
        streamed_scores, weights
    )
    reference_gradient = torch.autograd.grad(reference, reference_scores)[0]
    streamed_gradient = torch.autograd.grad(streamed.raw_loss, streamed_scores)[0]

    assert not weights.requires_grad
    assert weights.dtype == torch.float32
    assert weights.sum().item() == pytest.approx(1.0)
    assert torch.allclose(streamed_gradient, reference_gradient)
    assert streamed.candidate_weights == pytest.approx(tuple(weights.tolist()))


def test_chunk_local_union_gradients_are_not_the_global_union_gradient() -> None:
    scores = torch.tensor((-0.2, -1.1, -2.3), requires_grad=True)
    global_gradient = torch.autograd.grad(-torch.logsumexp(scores, dim=0), scores)[0]
    chunked = -torch.logsumexp(scores[:2], dim=0) - scores[2]
    chunked_gradient = torch.autograd.grad(chunked, scores)[0]

    assert not torch.allclose(global_gradient, chunked_gradient)


def test_streamed_union_rejects_non_normalized_or_misaligned_weights() -> None:
    scores = torch.tensor((-0.2, -1.1))

    with pytest.raises(LossContractError, match="aligned"):
        prefix_free_union_detached_weight_surrogate(scores, torch.tensor((1.0,)))
    with pytest.raises(LossContractError, match="sum to one"):
        prefix_free_union_detached_weight_surrogate(
            scores, torch.tensor((0.2, 0.2))
        )


def test_bottleneck_hinge_averages_tokens_within_owner_then_owners() -> None:
    logits = torch.tensor(
        (
            ((2.0, 1.0, 0.0), (0.75, 0.0, -1.0)),
            ((0.0, 0.0, 1.0), (100.0, -100.0, 0.0)),
        ),
        requires_grad=True,
    )
    targets = torch.tensor(((0, 1), (2, 0)))
    mask = torch.tensor(((True, True), (True, False)))

    result = coherent_full_chain_bottleneck_hinge(
        logits,
        targets,
        mask,
        required_margin=0.5,
    )

    assert result.numerator.item() == pytest.approx(0.625)
    assert result.denominator == 2
    assert result.raw_loss.item() == pytest.approx(0.3125)
    assert result.minimum_target_margin == pytest.approx(-0.75)
    assert result.violating_site_count == 1
    assert result.satisfied_site_count == 2


def test_bottleneck_hinge_detaches_competitor_and_zeroes_satisfied_gradient() -> None:
    logits = torch.tensor(
        (((2.0, 1.0, 0.0),), ((0.0, 2.0, 1.0),)),
        requires_grad=True,
    )
    result = coherent_full_chain_bottleneck_hinge(
        logits,
        torch.tensor(((0,), (0,))),
        torch.ones((2, 1), dtype=torch.bool),
        required_margin=0.5,
    )
    result.raw_loss.backward()

    assert not result.competitor_token_ids.requires_grad
    assert result.competitor_token_ids.tolist() == [[1], [1]]
    assert torch.equal(logits.grad[0], torch.zeros_like(logits.grad[0]))
    assert logits.grad[1, 0].tolist() == pytest.approx((-0.5, 0.5, 0.0))


def test_bottleneck_hinge_returns_zero_for_no_selected_chain_sites() -> None:
    logits = torch.zeros((1, 2, 3), requires_grad=True)
    result = coherent_full_chain_bottleneck_hinge(
        logits,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.bool),
        required_margin=0.1,
    )
    result.raw_loss.backward()

    assert result.raw_loss.item() == 0.0
    assert result.denominator == 0
    assert result.minimum_target_margin is None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_duplicate_unlikelihood_consumes_every_event_and_balances_images() -> None:
    logits = torch.tensor(
        ((0.0, 0.0), (0.0, 0.0), (math.log(3.0), 0.0)),
        requires_grad=True,
    )
    result = image_balanced_duplicate_token_unlikelihood(
        logits,
        torch.zeros(3, dtype=torch.long),
        torch.tensor((10, 10, 20)),
    )

    assert result.numerator.item() == pytest.approx(3.0 * math.log(2.0))
    assert result.denominator == 2
    assert result.raw_loss.item() == pytest.approx(1.5 * math.log(2.0))
    assert result.raw_event_count == 3
    assert result.consumed_event_count == 3
    assert result.capped_event_count == 0
    assert result.eligible_image_count == 2


def test_duplicate_unlikelihood_is_finite_with_gradient_above_thirty_nats() -> None:
    logits = torch.tensor(((35.0, 0.0, -1.0),), requires_grad=True)
    result = image_balanced_duplicate_token_unlikelihood(
        logits,
        torch.tensor((0,)),
        torch.tensor((7,)),
    )
    result.raw_loss.backward()

    assert torch.isfinite(result.raw_loss)
    assert result.minimum_target_margin is not None
    assert result.minimum_target_margin >= 30.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0].item() > 0.99


def test_duplicate_unlikelihood_returns_zero_for_no_events() -> None:
    logits = torch.empty((0, 3), requires_grad=True)
    result = image_balanced_duplicate_token_unlikelihood(
        logits,
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
    )
    result.raw_loss.backward()

    assert result.raw_loss.item() == 0.0
    assert result.denominator == 0
    assert result.raw_event_count == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
