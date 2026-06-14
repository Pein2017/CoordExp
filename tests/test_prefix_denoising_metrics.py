from __future__ import annotations

import pytest

from src.detection.prefix_denoising.metrics import (
    PREFIX_DENOISING_REQUIRED_METRIC_KEYS,
    prefix_denoising_ce_events,
)
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
        assert key in flat
    assert flat["llm_loss"] == pytest.approx(1.2)
    assert flat["prefix_denoising/global/loss/ce_balanced"] == pytest.approx(1.2)
    assert flat["prefix_denoising/clean_full/loss/ce"] == pytest.approx(1.0)
    assert flat["prefix_denoising/noisy_full/loss/ce"] == pytest.approx(1.4)
    assert flat["prefix_denoising/global/token_acc/full_vocab/top1"] == pytest.approx(0.5)
    assert flat["prefix_denoising/global/token_acc/full_vocab/top5"] == pytest.approx(0.8)
