from __future__ import annotations

import pytest
import torch

from src.detection.prefix_denoising.loss import (
    PrefixDenoisingSegmentSpan,
    compute_branch_balanced_hard_ce,
    compute_local_coord_kl,
    coord_support_window,
    topk_accuracy_from_logits,
)
from src.detection.prefix_denoising.types import (
    PrefixDenoisingKLSite,
    ResolvedPrefixDenoisingKLSite,
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


def test_branch_balanced_hard_ce_rejects_nonfinite_logits() -> None:
    logits = torch.zeros((1, 6, 4), dtype=torch.float32)
    labels = torch.full((1, 6), -100, dtype=torch.long)
    labels[0, 1] = 1
    labels[0, 4] = 1
    logits[0, 0, 1] = float("nan")
    spans = (
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=0,
            token_end=3,
            branch_id="clean_full",
            segment_id="a:clean",
        ),
        PrefixDenoisingSegmentSpan(
            batch_index=0,
            token_start=3,
            token_end=6,
            branch_id="noisy_full",
            segment_id="a:noisy",
        ),
    )

    with pytest.raises(ValueError, match="finite"):
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


def test_coord_support_window_clips_at_edges() -> None:
    assert coord_support_window(clean_bin=2, radius=4) == (0, 1, 2, 3, 4, 5, 6)
    assert coord_support_window(clean_bin=998, radius=4) == (
        994,
        995,
        996,
        997,
        998,
        999,
    )


def test_local_coord_kl_maps_bins_to_coord_token_ids_and_detaches_teacher() -> None:
    coord_token_ids = torch.tensor([1000 + i for i in range(1000)], dtype=torch.long)
    clean_logits = torch.zeros((2, 6, 2100), dtype=torch.float32, requires_grad=True)
    noisy_logits = torch.zeros((2, 6, 2100), dtype=torch.float32, requires_grad=True)
    clean_logits.data[1, 1, 1010] = 5.0
    noisy_logits.data[1, 1, 1010] = 3.0
    clean_logits.data[0, 1, 1010] = -20.0
    clean_logits.data[1, 1, 10] = 30.0
    site = ResolvedPrefixDenoisingKLSite(
        clean_batch_index=1,
        noisy_batch_index=1,
        object_index=0,
        coord_slot="y1",
        clean_label_position=2,
        noisy_label_position=2,
        clean_gt_bin=10,
        support_bins=tuple(range(8, 13)),
        identical_prefix=False,
    )

    result = compute_local_coord_kl(
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        sites=(site,),
        coord_token_ids=coord_token_ids,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert clean_logits.grad is None
    assert noisy_logits.grad is not None
    assert result.candidate_site_count == 1
    assert result.effective_site_count == 1
    assert result.teacher_gt_prob_conditional > 0.9


def test_local_coord_kl_raw_loss_is_mean_over_effective_sites() -> None:
    coord_token_ids = torch.tensor([1000 + i for i in range(1000)], dtype=torch.long)
    clean_logits = torch.zeros((1, 4, 2100), dtype=torch.float32)
    noisy_logits = torch.zeros((1, 4, 2100), dtype=torch.float32)
    clean_logits[0, 1, 1010] = 4.0
    noisy_logits[0, 1, 1011] = 4.0
    base_site = ResolvedPrefixDenoisingKLSite(
        clean_batch_index=0,
        noisy_batch_index=0,
        object_index=0,
        coord_slot="x1",
        clean_label_position=2,
        noisy_label_position=2,
        clean_gt_bin=10,
        support_bins=(9, 10, 11),
        identical_prefix=True,
    )

    one_site = compute_local_coord_kl(
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        sites=(base_site,),
        coord_token_ids=coord_token_ids,
    )
    two_sites = compute_local_coord_kl(
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        sites=(base_site, base_site),
        coord_token_ids=coord_token_ids,
    )

    torch.testing.assert_close(two_sites.raw_loss, one_site.raw_loss)
    assert two_sites.candidate_site_count == 2
    assert two_sites.effective_site_count == 2
    assert two_sites.identical_prefix_site_count == 2


def test_local_coord_kl_rejects_nonfinite_logits() -> None:
    coord_token_ids = torch.tensor([1000 + i for i in range(1000)], dtype=torch.long)
    clean_logits = torch.zeros((1, 4, 2100), dtype=torch.float32)
    noisy_logits = torch.zeros((1, 4, 2100), dtype=torch.float32)
    clean_logits[0, 1, 1010] = float("nan")
    site = ResolvedPrefixDenoisingKLSite(
        clean_batch_index=0,
        noisy_batch_index=0,
        object_index=0,
        coord_slot="x1",
        clean_label_position=2,
        noisy_label_position=2,
        clean_gt_bin=10,
        support_bins=(9, 10, 11),
        identical_prefix=False,
    )

    with pytest.raises(ValueError, match="finite"):
        compute_local_coord_kl(
            clean_logits=clean_logits,
            noisy_logits=noisy_logits,
            sites=(site,),
            coord_token_ids=coord_token_ids,
        )


def test_local_coord_kl_uses_teacher_to_student_orientation_and_gradients() -> None:
    coord_token_ids = torch.tensor([1000 + i for i in range(1000)], dtype=torch.long)
    clean_logits = torch.zeros((1, 4, 2100), dtype=torch.float32, requires_grad=True)
    noisy_logits = torch.zeros((1, 4, 2100), dtype=torch.float32, requires_grad=True)
    support_bins = (9, 10, 11)
    support_token_ids = torch.tensor([1009, 1010, 1011], dtype=torch.long)
    teacher_local_logits = torch.tensor([0.2, 2.3, -0.7], dtype=torch.float32)
    student_local_logits = torch.tensor([1.1, -0.4, 2.0], dtype=torch.float32)
    clean_logits.data[0, 1, support_token_ids] = teacher_local_logits
    noisy_logits.data[0, 1, support_token_ids] = student_local_logits
    site = ResolvedPrefixDenoisingKLSite(
        clean_batch_index=0,
        noisy_batch_index=0,
        object_index=0,
        coord_slot="x1",
        clean_label_position=2,
        noisy_label_position=2,
        clean_gt_bin=10,
        support_bins=support_bins,
        identical_prefix=False,
    )

    result = compute_local_coord_kl(
        clean_logits=clean_logits,
        noisy_logits=noisy_logits,
        sites=(site,),
        coord_token_ids=coord_token_ids,
    )
    teacher_prob = torch.softmax(teacher_local_logits, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_local_logits, dim=-1)
    student_prob = torch.softmax(student_local_logits, dim=-1)
    student_log_prob = torch.log_softmax(student_local_logits, dim=-1)
    expected_forward = torch.sum(
        teacher_prob * (torch.log(teacher_prob) - student_log_prob)
    )
    expected_reverse = torch.sum(student_prob * (student_log_prob - teacher_log_prob))

    assert expected_forward.item() != pytest.approx(expected_reverse.item())
    torch.testing.assert_close(result.raw_loss, expected_forward)
    result.raw_loss.backward()

    assert clean_logits.grad is None
    assert noisy_logits.grad is not None
    assert noisy_logits.grad[0, 1, 1010].item() < 0.0
    assert noisy_logits.grad[0, 1, 1011].item() > 0.0


def test_local_coord_kl_rejects_unresolved_and_invalid_support_sites() -> None:
    coord_token_ids = torch.tensor([1000 + i for i in range(1000)], dtype=torch.long)
    clean_logits = torch.zeros((1, 4, 2100), dtype=torch.float32)
    noisy_logits = torch.zeros((1, 4, 2100), dtype=torch.float32)
    unresolved = PrefixDenoisingKLSite(
        clean_segment_id="unit:clean",
        noisy_segment_id="unit:noisy",
        object_index=0,
        history_object_count=0,
        coord_slot="x1",
        clean_label_position=2,
        noisy_label_position=2,
        clean_gt_bin=10,
        support_bins=(9, 10, 11),
    )
    resolved_missing_gt = ResolvedPrefixDenoisingKLSite(
        clean_batch_index=0,
        noisy_batch_index=0,
        object_index=0,
        coord_slot="x1",
        clean_label_position=2,
        noisy_label_position=2,
        clean_gt_bin=10,
        support_bins=(8, 9, 11),
    )

    with pytest.raises(TypeError, match="ResolvedPrefixDenoisingKLSite"):
        compute_local_coord_kl(
            clean_logits=clean_logits,
            noisy_logits=noisy_logits,
            sites=(unresolved,),  # type: ignore[arg-type]
            coord_token_ids=coord_token_ids,
        )
    with pytest.raises(ValueError, match="clean_gt_bin"):
        compute_local_coord_kl(
            clean_logits=clean_logits,
            noisy_logits=noisy_logits,
            sites=(resolved_missing_gt,),
            coord_token_ids=coord_token_ids,
        )
