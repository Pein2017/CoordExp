from __future__ import annotations

from src.metrics.events import MetricEvent, weighted_mean_event

PREFIX_DENOISING_REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "llm_loss",
    "prefix_denoising/global/loss/ce_balanced",
    "prefix_denoising/global/loss/ce_token_pooled",
    "prefix_denoising/clean_full/loss/ce",
    "prefix_denoising/noisy_full/loss/ce",
    "prefix_denoising/global/token_acc/full_vocab/top1",
    "prefix_denoising/global/token_acc/full_vocab/top5",
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
) -> tuple[MetricEvent, ...]:
    total_denominator = float(int(clean_denominator) + int(noisy_denominator))
    return (
        weighted_mean_event(
            "llm_loss",
            ce_balanced,
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


__all__ = [
    "PREFIX_DENOISING_REQUIRED_METRIC_KEYS",
    "prefix_denoising_ce_events",
]
