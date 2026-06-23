from __future__ import annotations

import pytest
import torch

from src.metrics.events import flatten_metric_events, reduce_metric_events
from src.training.coverage_ledger.loss import (
    CoverageLedgerDebugRows,
    CoverageLedgerLossResult,
)
from src.training.coverage_ledger.metrics import coverage_ledger_metric_events


WEIGHTED_LOSS_KEY = "teacher_forcing/loss/coverage_ledger_auxiliary_weighted"
COVERAGE_BCE_KEY = "teacher_forcing/ledger/coverage_bce"
REGION_ANCHOR_KEY = "teacher_forcing/ledger/region_anchor_positive"
COVERAGE_AUC_KEY = "teacher_forcing/ledger/coverage_auc"
COVERAGE_ACCURACY_KEY = "teacher_forcing/ledger/coverage_accuracy"
COVERAGE_STATE_COUNT_KEY = "teacher_forcing/ledger/coverage_state_count"
COVERAGE_PAIR_COUNT_KEY = "teacher_forcing/ledger/coverage_pair_count"
OBJECT_COUNT_KEY = "teacher_forcing/ledger/object_count"
REGION_ANCHOR_PAIR_COUNT_KEY = "teacher_forcing/ledger/region_anchor_pair_count"


def _result(
    *,
    coverage_loss: float = 0.25,
    region_anchor_loss: float = 0.5,
    weighted_loss: float = 0.75,
    coverage_weight: float = 1.0,
    region_anchor_weight: float = 1.0,
    coverage_logits: torch.Tensor | None = None,
    coverage_targets: torch.Tensor | None = None,
    region_anchor_positive_logits: torch.Tensor | None = None,
    object_count: int = 3,
) -> CoverageLedgerLossResult:
    logits = (
        coverage_logits
        if coverage_logits is not None
        else torch.tensor([[0.8, 0.8], [-0.2, -0.2]], dtype=torch.float32)
    )
    targets = (
        coverage_targets
        if coverage_targets is not None
        else torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    )
    anchor_logits = (
        region_anchor_positive_logits
        if region_anchor_positive_logits is not None
        else torch.tensor([1.5, 0.5, -0.5], dtype=torch.float32)
    )
    return CoverageLedgerLossResult(
        total_loss=torch.tensor(weighted_loss, dtype=torch.float32),
        coverage_loss=torch.tensor(coverage_loss, dtype=torch.float32),
        region_anchor_loss=torch.tensor(region_anchor_loss, dtype=torch.float32),
        weighted_loss=torch.tensor(weighted_loss, dtype=torch.float32),
        coverage_weight=coverage_weight,
        region_anchor_weight=region_anchor_weight,
        metric_events=(),
        debug_rows=CoverageLedgerDebugRows(
            coverage_state_positions=tuple(range(int(targets.shape[0]))),
            region_anchor_positions=tuple(range(int(anchor_logits.numel()))),
            region_anchor_object_indices=tuple(range(int(anchor_logits.numel()))),
            coverage_targets=targets,
            coverage_logits=logits,
            region_anchor_positive_logits=anchor_logits,
            object_count=object_count,
            coverage_state_count=int(targets.shape[0]),
            coverage_pair_count=int(targets.numel()),
            region_anchor_pair_count=int(anchor_logits.numel()),
        ),
    )


def _events_by_key(result: CoverageLedgerLossResult):
    return {event.key: event for event in coverage_ledger_metric_events(result)}


def test_coverage_ledger_metric_events_publish_canonical_keys_and_metadata() -> None:
    events = coverage_ledger_metric_events(_result())
    by_key = {event.key: event for event in events}

    assert tuple(by_key) == (
        WEIGHTED_LOSS_KEY,
        COVERAGE_BCE_KEY,
        REGION_ANCHOR_KEY,
        COVERAGE_AUC_KEY,
        COVERAGE_ACCURACY_KEY,
        COVERAGE_STATE_COUNT_KEY,
        COVERAGE_PAIR_COUNT_KEY,
        OBJECT_COUNT_KEY,
        REGION_ANCHOR_PAIR_COUNT_KEY,
    )
    assert by_key[WEIGHTED_LOSS_KEY].diagnostic_only is False
    assert by_key[WEIGHTED_LOSS_KEY].objective_id == "coverage_ledger"
    for key, event in by_key.items():
        assert event.objective_id == "coverage_ledger"
        assert event.stage == "teacher_forcing"
        if key != WEIGHTED_LOSS_KEY:
            assert event.diagnostic_only is True


def test_loss_events_use_weighted_mean_values_and_explicit_denominators() -> None:
    by_key = _events_by_key(
        _result(
            coverage_loss=2.0,
            region_anchor_loss=10.0,
            weighted_loss=999.0,
            coverage_weight=0.5,
            region_anchor_weight=2.0,
        )
    )

    coverage_bce = by_key[COVERAGE_BCE_KEY]
    assert coverage_bce.reducer == "weighted_mean"
    assert coverage_bce.value == pytest.approx(2.0)
    assert coverage_bce.denominator == pytest.approx(4.0)
    assert coverage_bce.numerator == pytest.approx(8.0)
    assert coverage_bce.diagnostic_only is True

    region_anchor = by_key[REGION_ANCHOR_KEY]
    assert region_anchor.reducer == "weighted_mean"
    assert region_anchor.value == pytest.approx(10.0)
    assert region_anchor.denominator == pytest.approx(3.0)
    assert region_anchor.numerator == pytest.approx(30.0)
    assert region_anchor.diagnostic_only is True

    weighted_loss = by_key[WEIGHTED_LOSS_KEY]
    assert weighted_loss.reducer == "weighted_mean"
    assert weighted_loss.value == pytest.approx(64.0 / 7.0)
    assert weighted_loss.denominator == pytest.approx(7.0)
    assert weighted_loss.numerator == pytest.approx(64.0)
    assert weighted_loss.diagnostic_only is False


def test_weighted_auxiliary_loss_reduces_by_component_weighted_pair_sums() -> None:
    first = _events_by_key(
        _result(
            coverage_loss=2.0,
            region_anchor_loss=10.0,
            weighted_loss=999.0,
            coverage_weight=0.5,
            region_anchor_weight=2.0,
            coverage_logits=torch.zeros((2, 2), dtype=torch.float32),
            coverage_targets=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            region_anchor_positive_logits=torch.zeros((3,), dtype=torch.float32),
            object_count=2,
        )
    )[WEIGHTED_LOSS_KEY]
    second = _events_by_key(
        _result(
            coverage_loss=1.0,
            region_anchor_loss=4.0,
            weighted_loss=777.0,
            coverage_weight=3.0,
            region_anchor_weight=0.25,
            coverage_logits=torch.zeros((1, 5), dtype=torch.float32),
            coverage_targets=torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]]),
            region_anchor_positive_logits=torch.zeros((1,), dtype=torch.float32),
            object_count=5,
        )
    )[WEIGHTED_LOSS_KEY]

    assert first.numerator == pytest.approx(64.0)
    assert first.denominator == pytest.approx(7.0)
    assert second.numerator == pytest.approx(16.0)
    assert second.denominator == pytest.approx(6.0)
    assert reduce_metric_events([first, second])[WEIGHTED_LOSS_KEY] == pytest.approx(
        80.0 / 13.0
    )


def test_auc_uses_positive_negative_pairs_and_tie_credit() -> None:
    result = _result(
        coverage_logits=torch.tensor([[0.8, 0.8], [-0.2, -0.2]]),
        coverage_targets=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )

    event = _events_by_key(result)[COVERAGE_AUC_KEY]

    assert event.reducer == "ratio"
    assert event.numerator == pytest.approx(2.0)
    assert event.denominator == pytest.approx(4.0)
    assert reduce_metric_events([event])[COVERAGE_AUC_KEY] == pytest.approx(0.5)


def test_auc_reduces_by_comparable_pairs_across_batches() -> None:
    first = _events_by_key(
        _result(
            coverage_logits=torch.tensor([[2.0, 0.0]]),
            coverage_targets=torch.tensor([[1.0, 0.0]]),
            region_anchor_positive_logits=torch.tensor([1.0]),
            object_count=1,
        )
    )[COVERAGE_AUC_KEY]
    second = _events_by_key(_result())[COVERAGE_AUC_KEY]

    reduced = reduce_metric_events([first, second])

    assert first.denominator == pytest.approx(1.0)
    assert second.denominator == pytest.approx(4.0)
    assert reduced[COVERAGE_AUC_KEY] == pytest.approx(3.0 / 5.0)


def test_auc_for_large_matrix_does_not_use_positive_negative_outer_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side = 256
    coverage_targets = torch.zeros((side, side), dtype=torch.float32)
    coverage_targets[:, ::2] = 1.0
    coverage_logits = torch.where(
        coverage_targets > 0.5,
        torch.ones_like(coverage_targets),
        torch.zeros_like(coverage_targets),
    )
    original_sub = torch.Tensor.__sub__

    def guarded_sub(self, other):
        if (
            isinstance(other, torch.Tensor)
            and self.ndim == 2
            and other.ndim == 2
            and self.shape[1] == 1
            and other.shape[0] == 1
            and self.numel() * other.numel() > 100_000
        ):
            raise AssertionError("coverage AUC used dense positive-negative broadcast")
        return original_sub(self, other)

    monkeypatch.setattr(torch.Tensor, "__sub__", guarded_sub)

    event = _events_by_key(
        _result(
            coverage_logits=coverage_logits,
            coverage_targets=coverage_targets,
            region_anchor_positive_logits=torch.ones((side,), dtype=torch.float32),
            object_count=side,
        )
    )[COVERAGE_AUC_KEY]

    positive_count = int(coverage_targets.sum().item())
    negative_count = int(coverage_targets.numel() - positive_count)
    assert event.numerator == pytest.approx(float(positive_count * negative_count))
    assert event.denominator == pytest.approx(float(positive_count * negative_count))
    assert reduce_metric_events([event])[COVERAGE_AUC_KEY] == pytest.approx(1.0)


def test_auc_is_omitted_when_coverage_targets_have_only_one_class() -> None:
    events = coverage_ledger_metric_events(
        _result(
            coverage_logits=torch.tensor([[1.0, 2.0, 3.0]]),
            coverage_targets=torch.tensor([[1.0, 1.0, 1.0]]),
            region_anchor_positive_logits=torch.tensor([1.0]),
            object_count=1,
        )
    )

    assert COVERAGE_AUC_KEY not in {event.key for event in events}


def test_accuracy_uses_sigmoid_threshold_at_half() -> None:
    event = _events_by_key(
        _result(
            coverage_logits=torch.tensor([[-0.1, 0.0, 0.1, 2.0]]),
            coverage_targets=torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
            region_anchor_positive_logits=torch.tensor([1.0]),
            object_count=1,
        )
    )[COVERAGE_ACCURACY_KEY]

    assert event.reducer == "ratio"
    assert event.numerator == pytest.approx(3.0)
    assert event.denominator == pytest.approx(4.0)
    assert flatten_metric_events([event])[COVERAGE_ACCURACY_KEY] == pytest.approx(0.75)


def test_zero_denominator_metrics_are_omitted_not_flattened_as_zero() -> None:
    events = coverage_ledger_metric_events(
        _result(
            coverage_loss=float("nan"),
            region_anchor_loss=float("nan"),
            weighted_loss=float("nan"),
            coverage_logits=torch.empty((0, 0), dtype=torch.float32),
            coverage_targets=torch.empty((0, 0), dtype=torch.float32),
            region_anchor_positive_logits=torch.empty((0,), dtype=torch.float32),
            object_count=0,
        )
    )
    flat = flatten_metric_events(events)

    assert WEIGHTED_LOSS_KEY not in flat
    assert COVERAGE_BCE_KEY not in flat
    assert REGION_ANCHOR_KEY not in flat
    assert COVERAGE_AUC_KEY not in flat
    assert COVERAGE_ACCURACY_KEY not in flat
    assert flat[COVERAGE_STATE_COUNT_KEY] == pytest.approx(0.0)
    assert flat[COVERAGE_PAIR_COUNT_KEY] == pytest.approx(0.0)
    assert flat[OBJECT_COUNT_KEY] == pytest.approx(0.0)
    assert flat[REGION_ANCHOR_PAIR_COUNT_KEY] == pytest.approx(0.0)
