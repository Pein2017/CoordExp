from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.trainers.monitoring.loss_gradient_monitor import (
    get_loss_gradient_monitor,
)


def _t(value: float) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32, requires_grad=True)


class _ToyProbeModel(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


def _build_trainer() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(global_step=0),
        accelerator=SimpleNamespace(sync_gradients=True),
        loss_gradient_monitor_cfg={
            "enabled": True,
            "interval_steps": 1,
            "require_sync_gradients": True,
            "ema_beta": 0.9,
            "param_block": {
                "strategy": "regex",
                "include": r"^probe$",
                "max_params": 1,
                "max_numel": 8,
            },
        },
    )


def test_loss_gradient_monitor_reports_conflict_and_cos_to_total() -> None:
    model = _ToyProbeModel([2.0, -1.0])
    trainer = _build_trainer()
    monitor = get_loss_gradient_monitor(trainer)
    assert monitor is not None

    loss_terms = {
        "term_pos": 2.0 * model.probe[0],
        "term_neg": -1.0 * model.probe[0],
    }

    metrics = monitor.measure(model=model, loss_terms=loss_terms)

    assert metrics["gradmon/neg_cosine_pair_frac"] == pytest.approx(1.0)
    assert metrics["gradmon/neg_cosine_pair_pct"] == pytest.approx(100.0)
    assert metrics["gradmon/cos_to_total/term_pos"] > 0.0
    assert metrics["gradmon/cos_to_total/term_neg"] < 0.0
    assert metrics["gradmon/grad_norm/term_pos"] > 0.0
    assert metrics["gradmon/grad_norm/term_neg"] > 0.0
    assert metrics["gradmon/shared_param_count"] == pytest.approx(1.0)
    assert metrics["gradmon/shared_param_numel"] == pytest.approx(2.0)


def test_loss_gradient_monitor_does_not_change_loss_or_backward_grads() -> None:
    baseline_model = _ToyProbeModel([1.5, -0.5])
    baseline_loss_terms = {
        "left": (baseline_model.probe[0] - baseline_model.probe[1]) ** 2,
        "right": 0.5 * baseline_model.probe[0],
    }
    baseline_total = sum(baseline_loss_terms.values())
    baseline_total.backward()
    baseline_grad = baseline_model.probe.grad.detach().clone()

    monitored_model = _ToyProbeModel([1.5, -0.5])
    monitored_trainer = _build_trainer()
    monitor = get_loss_gradient_monitor(monitored_trainer)
    assert monitor is not None

    monitored_loss_terms = {
        "left": (monitored_model.probe[0] - monitored_model.probe[1]) ** 2,
        "right": 0.5 * monitored_model.probe[0],
    }
    monitored_total = sum(monitored_loss_terms.values())
    _ = monitor.measure(model=monitored_model, loss_terms=monitored_loss_terms)
    monitored_total.backward()

    assert float(monitored_total.detach().cpu().item()) == pytest.approx(
        float(baseline_total.detach().cpu().item())
    )
    assert torch.allclose(monitored_model.probe.grad, baseline_grad)


def test_loss_gradient_monitor_apply_caps_rejects_oversized_first_param() -> None:
    trainer = _build_trainer()
    monitor = get_loss_gradient_monitor(trainer)
    assert monitor is not None
    monitor.max_numel = 10

    big = torch.nn.Parameter(torch.zeros(100))
    small = torch.nn.Parameter(torch.zeros(5))

    selected = monitor._apply_caps(
        [
            ("big_param", big),
            ("small_param", small),
        ]
    )

    assert [name for name, _ in selected] == ["small_param"]
