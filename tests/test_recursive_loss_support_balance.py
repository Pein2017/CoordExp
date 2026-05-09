from __future__ import annotations

import pytest
import torch

from src.detection.loss import support_balance_loss


def test_support_one_balance_one_equals_sparse_soft_ce() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32)
    positives = torch.tensor([0, 1], dtype=torch.long)
    q = torch.tensor([0.5, 0.5], dtype=torch.float32)

    got = support_balance_loss(
        logits,
        positives,
        q,
        support_weight=1.0,
        balance_weight=1.0,
    )

    log_probs = logits.log_softmax(dim=-1)[0]
    expected = -0.5 * log_probs[0] - 0.5 * log_probs[1]
    assert got.item() == pytest.approx(expected.item())


def test_higher_balance_penalizes_valid_conditional_collapse() -> None:
    flat_valid = torch.tensor([[3.0, 3.0, -5.0]], dtype=torch.float32)
    peaked_valid = torch.tensor([[6.0, 0.0, -5.0]], dtype=torch.float32)
    positives = torch.tensor([0, 1], dtype=torch.long)
    q = torch.tensor([0.5, 0.5], dtype=torch.float32)

    flat_loss = support_balance_loss(
        flat_valid,
        positives,
        q,
        support_weight=1.0,
        balance_weight=2.0,
    )
    peaked_loss = support_balance_loss(
        peaked_valid,
        positives,
        q,
        support_weight=1.0,
        balance_weight=2.0,
    )

    assert peaked_loss.item() > flat_loss.item()


def test_support_term_penalizes_low_absolute_valid_mass_with_same_conditional_distribution() -> None:
    high_mass = torch.tensor([[5.0, 5.0, -5.0]], dtype=torch.float32)
    low_mass = torch.tensor([[-5.0, -5.0, 5.0]], dtype=torch.float32)
    positives = torch.tensor([0, 1], dtype=torch.long)
    q = torch.tensor([0.5, 0.5], dtype=torch.float32)

    assert support_balance_loss(
        low_mass,
        positives,
        q,
        support_weight=1.0,
        balance_weight=2.0,
    ).item() > support_balance_loss(
        high_mass,
        positives,
        q,
        support_weight=1.0,
        balance_weight=2.0,
    ).item()
    assert support_balance_loss(
        low_mass,
        positives,
        q,
        support_weight=0.0,
        balance_weight=1.0,
    ).item() == pytest.approx(
        support_balance_loss(
            high_mass,
            positives,
            q,
            support_weight=0.0,
            balance_weight=1.0,
        ).item()
    )


def test_support_balance_rejects_duplicate_positive_token_ids() -> None:
    with pytest.raises(ValueError, match="unique vocabulary token ids"):
        support_balance_loss(
            torch.zeros(1, 10),
            torch.tensor([7, 7], dtype=torch.long),
            torch.tensor([0.5, 0.5], dtype=torch.float32),
            support_weight=1.0,
            balance_weight=2.0,
        )
