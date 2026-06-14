from __future__ import annotations

import pytest

from src.detection.prefix_denoising.metrics import (
    PREFIX_DENOISING_REQUIRED_METRIC_KEYS,
    prefix_denoising_ce_events,
    prefix_denoising_kl_events,
)
from src.detection.prefix_denoising.loss import PrefixDenoisingKLResult
from src.metrics.events import flatten_metric_events


def test_prefix_denoising_metric_keys_are_distinct_flat_keys() -> None:
    events = prefix_denoising_ce_events(
        ce_balanced=1.2,
        ce_clean=1.0,
        ce_noisy=1.4,
        ce_token_pooled=1.25,
        token_top1=0.5,
        token_top5=0.8,
        clean_denominator=10,
        noisy_denominator=12,
    )

    flat = flatten_metric_events(events)

    for key in PREFIX_DENOISING_REQUIRED_METRIC_KEYS:
        if key.startswith("prefix_denoising/kl/local_window"):
            continue
        assert key in flat
    assert flat["llm_loss"] == pytest.approx(1.2)
    assert flat["prefix_denoising/global/loss/ce_balanced"] == pytest.approx(1.2)
    assert flat["prefix_denoising/clean_full/loss/ce"] == pytest.approx(1.0)
    assert flat["prefix_denoising/noisy_full/loss/ce"] == pytest.approx(1.4)
    assert flat["prefix_denoising/global/token_acc/full_vocab/top1"] == pytest.approx(0.5)
    assert flat["prefix_denoising/global/token_acc/full_vocab/top5"] == pytest.approx(0.8)


def test_prefix_denoising_kl_metric_keys_flatten_with_slot_deltas() -> None:
    result = PrefixDenoisingKLResult(
        loss=None,  # type: ignore[arg-type]
        raw_loss=None,  # type: ignore[arg-type]
        candidate_site_count=4,
        effective_site_count=4,
        identical_prefix_site_count=1,
        teacher_support_mass=0.8,
        student_support_mass=0.6,
        teacher_gt_prob_full_coord_vocab=0.2,
        student_gt_prob_full_coord_vocab=0.1,
        teacher_gt_prob_conditional=0.5,
        student_gt_prob_conditional=0.3,
        support_bin_count=9.0,
        edge_truncation_rate=0.25,
        teacher_top1_is_gt=1.0,
        student_top1_is_gt=0.0,
        slot_metrics={
            "x1": {
                "teacher_support_mass": 0.9,
                "student_support_mass": 0.7,
                "teacher_gt_prob_full_coord_vocab": 0.3,
                "student_gt_prob_full_coord_vocab": 0.2,
                "teacher_gt_prob_conditional": 0.6,
                "student_gt_prob_conditional": 0.4,
            }
        },
    )

    flat = flatten_metric_events(
        prefix_denoising_kl_events(
            kl=result,
            raw_loss=1.25,
            weighted_loss=0.125,
        )
    )

    for key in PREFIX_DENOISING_REQUIRED_METRIC_KEYS:
        if key.startswith("prefix_denoising/kl/local_window"):
            assert key in flat
    assert flat["prefix_denoising/kl/local_window/raw"] == pytest.approx(1.25)
    assert flat["prefix_denoising/kl/local_window/weighted"] == pytest.approx(0.125)
    assert flat["prefix_denoising/kl/local_window/candidate_site_count"] == pytest.approx(4)
    assert flat["prefix_denoising/kl/local_window/site_count"] == pytest.approx(4)
    assert flat["prefix_denoising/kl/local_window/identical_prefix_site_count"] == pytest.approx(1)
    assert flat["prefix_denoising/kl/local_window/x1/teacher_minus_student/support_mass"] == pytest.approx(0.2)
    assert flat["prefix_denoising/kl/local_window/x1/teacher_minus_student/full_vocab_gt_prob"] == pytest.approx(0.1)
    assert flat["prefix_denoising/kl/local_window/x1/teacher_minus_student/local_gt_prob"] == pytest.approx(0.2)
