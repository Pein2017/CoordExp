from __future__ import annotations

import math

import pytest
import torch

from src.trainers.stage1_set_continuation.losses import (
    compute_candidate_full_entry_logprob,
    compute_close_sequence_nll,
    compute_close_start_nll,
    compute_mp_pem_losses,
    summarize_candidate_scores,
)


def test_candidate_score_uses_full_vocab_for_text_and_coord_vocab_for_coord_slots() -> None:
    logits = torch.full((1, 4, 8), -10.0)
    labels = torch.tensor([[0, 2, 5, 6]], dtype=torch.long)
    candidate_mask = torch.tensor([[False, True, True, True]])
    coord_mask = torch.tensor([[False, False, True, False]])
    coord_token_ids = torch.tensor([4, 5, 6], dtype=torch.long)

    logits[0, 0, 2] = 3.0
    logits[0, 1, 5] = 4.0
    logits[0, 1, 1] = 9.0
    logits[0, 2, 6] = 5.0

    result = compute_candidate_full_entry_logprob(
        logits=logits,
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )

    full_vocab_text_1 = torch.log_softmax(logits[0, 0], dim=-1)[2]
    coord_vocab_slot = torch.log_softmax(logits[0, 1, coord_token_ids], dim=-1)[1]
    full_vocab_text_2 = torch.log_softmax(logits[0, 2], dim=-1)[6]
    expected = full_vocab_text_1 + coord_vocab_slot + full_vocab_text_2

    assert torch.allclose(result.score, expected)
    assert result.coord_tokens == 1
    assert result.non_coord_tokens == 2


def test_mp_exact_logz_and_responsibility_metrics() -> None:
    scores = torch.tensor([-2.0, -3.0, -5.0])

    result = compute_mp_pem_losses(scores=scores, pem_mode="disabled")

    log_z = torch.logsumexp(scores, dim=0)
    responsibilities = torch.softmax(scores, dim=0)
    entropy = -(responsibilities * responsibilities.log()).sum()
    assert torch.allclose(result.loss_mp, -log_z)
    assert torch.allclose(result.total_objective, result.loss_mp)
    assert torch.allclose(result.log_z_remaining, log_z)
    assert result.log_z_estimator == "exact"
    assert result.metrics["mp/responsibility_entropy"] == pytest.approx(float(entropy))
    assert result.metrics["mp/max_responsibility"] == pytest.approx(
        float(responsibilities.max())
    )
    assert result.metrics["mp/min_responsibility"] == pytest.approx(
        float(responsibilities.min())
    )


def test_uniform_importance_logz_adds_population_correction() -> None:
    scores = torch.tensor([-2.0, -3.0])

    result = compute_mp_pem_losses(
        scores=scores,
        pem_mode="disabled",
        estimator="uniform_importance",
        remaining_count=8,
        scored_count=2,
    )

    expected = torch.logsumexp(scores, dim=0) + math.log(8 / 2)
    assert torch.allclose(result.log_z_remaining, expected)
    assert result.log_z_estimator == "uniform_importance"
    assert result.metrics["mp/logZ_estimator"] == "uniform_importance"


def test_uniform_importance_logz_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="positive counts"):
        compute_mp_pem_losses(
            scores=torch.tensor([-2.0, -3.0]),
            pem_mode="disabled",
            estimator="uniform_importance",
            remaining_count=0,
            scored_count=2,
        )


def test_sampled_raw_logz_omits_population_correction() -> None:
    scores = torch.tensor([-2.0, -3.0])

    result = compute_mp_pem_losses(
        scores=scores,
        pem_mode="disabled",
        estimator="sampled_raw",
        remaining_count=8,
        scored_count=2,
    )

    assert torch.allclose(result.log_z_remaining, torch.logsumexp(scores, dim=0))
    assert result.log_z_estimator == "sampled_raw"


def test_pem_threshold_loss_does_not_add_mp_loss() -> None:
    scores = torch.tensor([-2.0, -3.0])
    log_z = torch.logsumexp(scores, dim=0)

    result = compute_mp_pem_losses(
        scores=scores,
        pem_mode="threshold_loss",
        log_rho=torch.tensor(-1.0),
    )

    assert torch.allclose(
        result.loss_pem,
        torch.clamp(torch.tensor(-1.0) - log_z, min=0.0),
    )
    assert torch.allclose(result.total_objective, result.loss_pem)
    assert torch.allclose(result.loss_mp, -log_z)


def test_pem_requires_exactly_one_threshold() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        compute_mp_pem_losses(
            scores=torch.tensor([-2.0]),
            pem_mode="threshold_loss",
            rho=0.9,
            log_rho=torch.tensor(-0.1),
        )


def test_close_losses_use_start_and_sequence_masks_separately() -> None:
    logits = torch.full((1, 4, 6), -5.0)
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    start_mask = torch.tensor([[False, False, True, False]])
    sequence_mask = torch.tensor([[False, False, True, True]])
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 3.0

    start = compute_close_start_nll(
        logits=logits,
        labels=labels,
        structural_close_start_mask=start_mask,
    )
    sequence = compute_close_sequence_nll(
        logits=logits,
        labels=labels,
        structural_close_sequence_mask=sequence_mask,
    )

    expected_start = -torch.log_softmax(logits[0, 1], dim=-1)[2]
    expected_sequence = -(
        torch.log_softmax(logits[0, 1], dim=-1)[2]
        + torch.log_softmax(logits[0, 2], dim=-1)[3]
    )
    assert torch.allclose(start, expected_start)
    assert torch.allclose(sequence, expected_sequence)


def test_empty_remaining_with_zero_close_weight_contributes_zero() -> None:
    result = compute_mp_pem_losses(
        scores=torch.empty(0),
        pem_mode="disabled",
        remaining_count=0,
        scored_count=0,
    )

    assert torch.allclose(result.total_objective, torch.tensor(0.0))
    assert result.denominator == 0


def test_single_candidate_summary_has_zero_entropy_and_skips_correlation() -> None:
    summary = summarize_candidate_scores(
        scores=torch.tensor([-2.0]),
        candidate_lengths=torch.tensor([7.0]),
    )

    assert summary["mp/responsibility_entropy"] == pytest.approx(0.0)
    assert summary["mp/candidate_score_std"] == pytest.approx(0.0)
    assert summary["mp/responsibility_length_corr_valid"] == 0
