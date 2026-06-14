from __future__ import annotations

import pytest
import torch

from src.detection.prefix_denoising.loss import (
    PrefixDenoisingSegmentSpan,
    compute_branch_balanced_hard_ce,
    topk_accuracy_from_logits,
)


def test_branch_balanced_hard_ce_uses_shifted_positions_and_all_segments() -> None:
    logits = torch.full((1, 8, 4), -9.0, dtype=torch.float32)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    labels[0, 2] = 1
    labels[0, 3] = 2
    labels[0, 6] = 3
    logits[0, 1, 1] = 8.0
    logits[0, 2, 2] = 8.0
    logits[0, 5, 3] = 8.0
    logits[0, 2, 0] = 12.0
    spans = (
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=0,
            token_end=4,
            branch_id="clean_full",
            segment_id="a:clean",
        ),
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=4,
            token_end=8,
            branch_id="noisy_full",
            segment_id="a:noisy",
        ),
    )

    result = compute_branch_balanced_hard_ce(
        logits=logits,
        labels=labels,
        segment_spans=spans,
    )

    clean_ce = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, 1], logits[0, 2]]),
        torch.tensor([1, 2]),
    )
    noisy_ce = torch.nn.functional.cross_entropy(
        logits[0, 5].unsqueeze(0),
        torch.tensor([3]),
    )
    torch.testing.assert_close(result.loss, 0.5 * clean_ce + 0.5 * noisy_ce)
    assert result.clean_denominator == 2
    assert result.noisy_denominator == 1
    torch.testing.assert_close(
        result.token_pooled_ce,
        (clean_ce * 2.0 + noisy_ce) / 3.0,
    )


def test_branch_balanced_hard_ce_requires_both_branches() -> None:
    logits = torch.zeros((1, 4, 3), dtype=torch.float32)
    labels = torch.full((1, 4), -100, dtype=torch.long)
    labels[0, 2] = 1
    spans = (
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=0,
            token_end=4,
            branch_id="clean_full",
            segment_id="a:clean",
        ),
    )

    with pytest.raises(ValueError, match="both clean_full and noisy_full"):
        compute_branch_balanced_hard_ce(
            logits=logits,
            labels=labels,
            segment_spans=spans,
        )


def test_branch_balanced_hard_ce_rejects_segment_start_supervision() -> None:
    logits = torch.zeros((1, 4, 3), dtype=torch.float32)
    labels = torch.full((1, 4), -100, dtype=torch.long)
    labels[0, 0] = 1
    labels[0, 3] = 2
    spans = (
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=0,
            token_end=2,
            branch_id="clean_full",
            segment_id="a:clean",
        ),
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=2,
            token_end=4,
            branch_id="noisy_full",
            segment_id="a:noisy",
        ),
    )

    with pytest.raises(ValueError, match="segment start"):
        compute_branch_balanced_hard_ce(
            logits=logits,
            labels=labels,
            segment_spans=spans,
        )


def test_topk_accuracy_uses_shifted_causal_positions() -> None:
    logits = torch.tensor(
        [[[0.0, 5.0, 1.0], [5.0, 0.0, 1.0], [0.0, 1.0, 5.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 1, -100]], dtype=torch.long)
    spans = (
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=0,
            token_end=3,
            branch_id="clean_full",
            segment_id="a:clean",
        ),
    )

    result = topk_accuracy_from_logits(
        logits=logits,
        labels=labels,
        segment_spans=spans,
        topk=(1, 2, 5),
    )

    assert result[1] == 1.0
    assert result[2] == 1.0
    assert result[5] == 1.0
