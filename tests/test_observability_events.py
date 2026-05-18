from __future__ import annotations

import pytest

from src.metrics.events import MetricEvent
from src.training.observability import ObservabilityService
from src.training.observability.contracts import REMOVED_TRAINING_METRIC_KEYS
from src.training.observability.legacy import adapt_legacy_metric


def test_objective_metric_events_use_canonical_metric_event_with_axes() -> None:
    service = ObservabilityService()

    event = service.objective_metric(
        key="train/objective/token_ce",
        value=2.5,
        weight=4,
        stage="stage2",
        channel="B",
        objective_id="token_ce",
        unit="token",
        provenance="objective_runner",
    )

    assert isinstance(event, MetricEvent)
    assert event.reducer == "weighted_mean"
    assert event.value == pytest.approx(2.5)
    assert event.numerator == pytest.approx(10.0)
    assert event.denominator == pytest.approx(4.0)
    assert event.stage == "stage2"
    assert event.channel == "B"
    assert event.objective_id == "token_ce"
    assert event.provenance == "objective_runner"
    assert event.identity.stage == "stage2"
    assert event.identity.channel == "B"
    assert event.identity.objective_id == "token_ce"
    assert event.identity.provenance == "objective_runner"


def test_service_flattens_canonical_metric_events_for_ms_swift_reporting() -> None:
    service = ObservabilityService()
    events = [
        service.objective_metric(
            key="train/objective/token_ce",
            value=1.0,
            weight=2,
            stage="stage2",
            channel="A",
            objective_id="token_ce",
            unit="token",
            provenance="objective_runner",
        ),
        service.objective_metric(
            key="train/objective/token_ce",
            value=3.0,
            weight=6,
            stage="stage2",
            channel="A",
            objective_id="token_ce",
            unit="token",
            provenance="objective_runner",
        ),
    ]

    assert service.flatten_for_ms_swift(events) == {
        "train/objective/token_ce": pytest.approx(2.5)
    }


def test_flattening_rejects_same_key_with_different_observability_axes() -> None:
    service = ObservabilityService()
    events = [
        service.objective_metric(
            key="train/objective/shared",
            value=1.0,
            weight=1,
            stage="stage2",
            channel="A",
            objective_id="token_ce",
            unit="token",
            provenance="objective_runner",
        ),
        service.objective_metric(
            key="train/objective/shared",
            value=2.0,
            weight=1,
            stage="stage2",
            channel="A",
            objective_id="trie_ce",
            unit="token",
            provenance="objective_runner",
        ),
    ]

    with pytest.raises(ValueError, match="Metric key collision"):
        service.flatten_for_ms_swift(events)


def test_service_rejects_removed_mechanism_writer_keys() -> None:
    service = ObservabilityService()

    with pytest.raises(ValueError, match="removed training mechanism"):
        service.objective_metric(
            key="train/optimization/loss_duplicate_burst_unlikelihood",
            value=1.0,
            weight=1,
            stage="stage2",
            channel="B",
            objective_id="loss_duplicate_burst_unlikelihood",
            unit="token",
            provenance="objective_runner",
        )


def test_removed_metric_keys_are_rejected_by_writers_and_legacy_readers() -> None:
    service = ObservabilityService()

    for key in sorted(REMOVED_TRAINING_METRIC_KEYS):
        with pytest.raises(ValueError, match="removed training mechanism"):
            service.objective_metric(
                key=key,
                value=1.0,
                weight=1,
                stage="stage2",
                channel="B",
                objective_id="retired",
                unit="token",
                provenance="objective_runner",
            )
        with pytest.raises(ValueError, match="removed training mechanism"):
            adapt_legacy_metric(key, 1.0)


def test_duplicate_diagnostic_counters_and_gauges_use_explicit_reducers() -> None:
    service = ObservabilityService()

    count = service.duplicate_count(
        "stage2_ab/channel_b/dup/N_duplicate_control_first_divergence_boundaries",
        3,
        stage="stage2",
        channel="B",
        provenance="duplicate_diagnostics",
    )
    gauge = service.duplicate_gauge(
        "stage2_ab/channel_b/dup/raw_duplicate_iou_mean",
        value=0.75,
        weight=4,
        stage="stage2",
        channel="B",
        provenance="duplicate_diagnostics",
    )

    assert count.reducer == "sum"
    assert count.value == pytest.approx(3)
    assert count.denominator is None
    assert gauge.reducer == "weighted_mean"
    assert gauge.numerator == pytest.approx(3.0)
    assert gauge.denominator == pytest.approx(4.0)


def test_duplicate_training_loss_keys_are_rejected_but_diagnostics_are_allowed() -> None:
    service = ObservabilityService()

    with pytest.raises(ValueError, match="removed training mechanism"):
        service.duplicate_gauge(
            "loss/B_rollout_text/duplicate_burst_unlikelihood",
            value=0.5,
            weight=2,
            stage="stage2",
            channel="B",
            provenance="duplicate_diagnostics",
        )

    diagnostic = service.duplicate_count(
        "stage2_ab/channel_b/dup/N_clusters_total",
        2,
        stage="stage2",
        channel="B",
        provenance="duplicate_diagnostics",
    )

    assert diagnostic.key == "stage2_ab/channel_b/dup/N_clusters_total"
    assert diagnostic.diagnostic_only is True


def test_legacy_adapter_preserves_duplicate_reducer_semantics() -> None:
    count_keys = (
        "stage2_ab/channel_b/dup/N_raw_bbox_valid",
        "stage2_ab/channel_b/dup/N_clean_accepted",
        "stage2_ab/channel_b/dup/N_clusters_total",
        "stage2_ab/channel_b/dup/N_clusters_exempt",
        "stage2_ab/channel_b/dup/N_clusters_suppressed",
        "stage2_ab/channel_b/dup/N_objects_suppressed",
        "stage2_ab/channel_b/dup/N_duplicate_control_first_divergence_boundaries",
        "stage2_ab/channel_b/dup/N_duplicate_control_first_divergence_skipped_no_divergence",
        "dup/raw/example_count",
        "dup/raw/example_total",
        "dup/raw/example_sum",
        "dup/raw/example_num",
        "dup/raw/example_den",
        "dup/raw/near_iou90_pairs_same_desc_count",
        "dup/raw/near_iou90_pairs_any_desc_count",
    )
    gauge_keys = (
        "stage2_ab/channel_b/dup/raw_duplicate_iou_mean",
        "dup/raw/max_desc_count",
        "dup/raw/saturation_rate",
        "dup/raw/duplicate_like_max_cluster_size",
        "dup/raw/desc_entropy",
    )

    assert all(adapt_legacy_metric(key, 4).reducer == "sum" for key in count_keys)
    assert all(
        adapt_legacy_metric(key, 0.75).reducer == "weighted_mean"
        for key in gauge_keys
    )
