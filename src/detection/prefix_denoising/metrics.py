from __future__ import annotations

from collections.abc import Mapping

from src.detection.prefix_denoising.loss import PrefixDenoisingKLResult
from src.metrics.events import MetricEvent, weighted_mean_event

PREFIX_DENOISING_REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "llm_loss",
    "prefix_denoising/global/loss/ce_balanced",
    "prefix_denoising/global/loss/ce_token_pooled",
    "prefix_denoising/clean_full/loss/ce",
    "prefix_denoising/noisy_full/loss/ce",
    "prefix_denoising/global/token_acc/full_vocab/top1",
    "prefix_denoising/global/token_acc/full_vocab/top5",
    "prefix_denoising/kl/local_window/raw",
    "prefix_denoising/kl/local_window/weighted",
    "prefix_denoising/kl/local_window/candidate_site_count",
    "prefix_denoising/kl/local_window/site_count",
    "prefix_denoising/kl/local_window/identical_prefix_site_count",
    "prefix_denoising/kl/local_window/teacher_support_mass",
    "prefix_denoising/kl/local_window/student_support_mass",
    "prefix_denoising/kl/local_window/teacher_gt_prob_full_coord_vocab",
    "prefix_denoising/kl/local_window/student_gt_prob_full_coord_vocab",
    "prefix_denoising/kl/local_window/teacher_gt_prob_conditional",
    "prefix_denoising/kl/local_window/student_gt_prob_conditional",
    "prefix_denoising/kl/local_window/support_bin_count",
    "prefix_denoising/kl/local_window/edge_truncation_rate",
    "prefix_denoising/kl/local_window/teacher_top1_is_gt",
    "prefix_denoising/kl/local_window/student_top1_is_gt",
)


def prefix_denoising_ce_events(
    *,
    ce_balanced: float,
    ce_clean: float,
    ce_noisy: float,
    ce_token_pooled: float,
    token_top1: float,
    token_top5: float,
    clean_denominator: int,
    noisy_denominator: int,
    llm_loss: float | None = None,
) -> tuple[MetricEvent, ...]:
    total_denominator = float(int(clean_denominator) + int(noisy_denominator))
    optimized_loss = ce_balanced if llm_loss is None else float(llm_loss)
    return (
        weighted_mean_event(
            "llm_loss",
            optimized_loss,
            1.0,
            unit="token",
            metric_surface="prefix_denoising",
            objective_id="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/global/loss/ce_balanced",
            ce_balanced,
            1.0,
            unit="token",
            metric_surface="prefix_denoising",
            objective_id="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/global/loss/ce_token_pooled",
            ce_token_pooled,
            total_denominator,
            unit="token",
            metric_surface="prefix_denoising",
            objective_id="prefix_denoising",
            diagnostic_only=True,
        ),
        weighted_mean_event(
            "prefix_denoising/clean_full/loss/ce",
            ce_clean,
            float(clean_denominator),
            unit="token",
            metric_surface="prefix_denoising",
            channel="clean_full",
            objective_id="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/noisy_full/loss/ce",
            ce_noisy,
            float(noisy_denominator),
            unit="token",
            metric_surface="prefix_denoising",
            channel="noisy_full",
            objective_id="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/global/token_acc/full_vocab/top1",
            token_top1,
            total_denominator,
            unit="token",
            metric_surface="prefix_denoising",
            vocab_scope="full_vocab",
            objective_id="prefix_denoising",
        ),
        weighted_mean_event(
            "prefix_denoising/global/token_acc/full_vocab/top5",
            token_top5,
            total_denominator,
            unit="token",
            metric_surface="prefix_denoising",
            vocab_scope="full_vocab",
            objective_id="prefix_denoising",
        ),
    )


def prefix_denoising_kl_events(
    *,
    kl: PrefixDenoisingKLResult,
    raw_loss: float,
    weighted_loss: float,
) -> tuple[MetricEvent, ...]:
    events = [
        _kl_event("prefix_denoising/kl/local_window/raw", raw_loss),
        _kl_event("prefix_denoising/kl/local_window/weighted", weighted_loss),
        _kl_event(
            "prefix_denoising/kl/local_window/candidate_site_count",
            kl.candidate_site_count,
        ),
        _kl_event("prefix_denoising/kl/local_window/site_count", kl.effective_site_count),
        _kl_event(
            "prefix_denoising/kl/local_window/identical_prefix_site_count",
            kl.identical_prefix_site_count,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/teacher_support_mass",
            kl.teacher_support_mass,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/student_support_mass",
            kl.student_support_mass,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/teacher_gt_prob_full_coord_vocab",
            kl.teacher_gt_prob_full_coord_vocab,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/student_gt_prob_full_coord_vocab",
            kl.student_gt_prob_full_coord_vocab,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/teacher_gt_prob_conditional",
            kl.teacher_gt_prob_conditional,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/student_gt_prob_conditional",
            kl.student_gt_prob_conditional,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/support_bin_count",
            kl.support_bin_count,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/edge_truncation_rate",
            kl.edge_truncation_rate,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/teacher_top1_is_gt",
            kl.teacher_top1_is_gt,
        ),
        _kl_event(
            "prefix_denoising/kl/local_window/student_top1_is_gt",
            kl.student_top1_is_gt,
        ),
    ]
    for slot, values in kl.slot_metrics.items():
        events.extend(_slot_kl_events(slot=str(slot), values=values))
    return tuple(events)


def _kl_event(key: str, value: float) -> MetricEvent:
    return weighted_mean_event(
        key,
        float(value),
        1.0,
        unit="slot",
        metric_surface="prefix_denoising",
        objective_id="prefix_denoising",
        coordinate_surface="local_window",
    )


def _slot_kl_events(
    *,
    slot: str,
    values: Mapping[str, float],
) -> tuple[MetricEvent, ...]:
    teacher_support = float(values.get("teacher_support_mass", 0.0))
    student_support = float(values.get("student_support_mass", 0.0))
    teacher_full = float(values.get("teacher_gt_prob_full_coord_vocab", 0.0))
    student_full = float(values.get("student_gt_prob_full_coord_vocab", 0.0))
    teacher_local = float(values.get("teacher_gt_prob_conditional", 0.0))
    student_local = float(values.get("student_gt_prob_conditional", 0.0))
    prefix = f"prefix_denoising/kl/local_window/{slot}"
    return (
        _slot_kl_event(
            f"{prefix}/teacher_support_mass",
            teacher_support,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/student_support_mass",
            student_support,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/teacher_gt_prob_full_coord_vocab",
            teacher_full,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/student_gt_prob_full_coord_vocab",
            student_full,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/teacher_gt_prob_conditional",
            teacher_local,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/student_gt_prob_conditional",
            student_local,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/teacher_minus_student/support_mass",
            teacher_support - student_support,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/teacher_minus_student/full_vocab_gt_prob",
            teacher_full - student_full,
            slot=slot,
        ),
        _slot_kl_event(
            f"{prefix}/teacher_minus_student/local_gt_prob",
            teacher_local - student_local,
            slot=slot,
        ),
    )


def _slot_kl_event(key: str, value: float, *, slot: str) -> MetricEvent:
    return weighted_mean_event(
        key,
        float(value),
        1.0,
        unit="slot",
        slot_name=slot,
        metric_surface="prefix_denoising",
        objective_id="prefix_denoising",
        coordinate_surface="local_window",
    )


__all__ = [
    "PREFIX_DENOISING_REQUIRED_METRIC_KEYS",
    "prefix_denoising_ce_events",
    "prefix_denoising_kl_events",
]
