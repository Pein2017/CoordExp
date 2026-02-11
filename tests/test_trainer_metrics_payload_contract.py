from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from src.metrics.payload_contract import validate_trainer_metrics_payload
from src.metrics.reporter import SwiftMetricReporter, best_effort


class _DummyMetric:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))


class _DummyModel:
    def __init__(self, training: bool) -> None:
        self.training = training


class _DummyTrainer:
    def __init__(self, *, training: bool) -> None:
        self.model = _DummyModel(training=training)
        self.state = SimpleNamespace(global_step=7)
        self.custom_metrics = {
            "train": defaultdict(_DummyMetric),
            "eval": defaultdict(_DummyMetric),
        }


def test_validate_payload_rejects_missing_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        validate_trainer_metrics_payload(
            {
                "mode": "train",
                "global_step": 0,
                "metrics": {"loss": 1.0},
            }
        )


def test_validate_payload_rejects_non_integer_schema_version() -> None:
    with pytest.raises(ValueError, match="integer"):
        validate_trainer_metrics_payload(
            {
                "schema_version": "1",
                "mode": "train",
                "global_step": 0,
                "metrics": {"loss": 1.0},
            }
        )


def test_validate_payload_rejects_unsupported_schema_major() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        validate_trainer_metrics_payload(
            {
                "schema_version": 999,
                "mode": "train",
                "global_step": 0,
                "metrics": {"loss": 1.0},
            }
        )


def test_reporter_preserves_metric_keys_and_routes_by_mode() -> None:
    trainer = _DummyTrainer(training=True)
    reporter = SwiftMetricReporter(trainer)

    reporter.update_many({"loss": 1.0, "eval_bbox_AP50": 0.5})
    assert trainer.custom_metrics["train"]["loss"].values[-1] == pytest.approx(1.0)
    assert trainer.custom_metrics["train"]["eval_bbox_AP50"].values[-1] == pytest.approx(0.5)
    assert "loss" not in trainer.custom_metrics["eval"]

    trainer.model.training = False
    reporter.update_many({"eval_bbox_AP50": 0.6})
    assert trainer.custom_metrics["eval"]["eval_bbox_AP50"].values[-1] == pytest.approx(0.6)


def test_best_effort_warns_once_and_disables_failing_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = SimpleNamespace()
    calls = {"n": 0}
    warnings: list[str] = []

    def _warn(msg: str, **kwargs: object) -> None:
        del kwargs
        warnings.append(msg)

    def _fail() -> None:
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr("src.metrics.reporter.logger.warning", _warn)

    best_effort(trainer, name="coord_diag", fn=_fail)
    best_effort(trainer, name="coord_diag", fn=_fail)

    assert calls["n"] == 1
    assert len(warnings) == 1
    assert "Diagnostic 'coord_diag' failed" in warnings[0]
