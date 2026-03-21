from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.trainers.monitoring.loss_gradient_monitor import (
    build_stage2_coord_monitor_terms_from_pipeline,
    build_stage2_two_channel_coord_monitor_terms,
    get_loss_gradient_monitor,
)
from src.trainers.teacher_forcing.contracts import PipelineResult


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


def test_build_stage2_coord_monitor_terms_from_pipeline_excludes_text_terms() -> None:
    pipeline_result = PipelineResult(
        total_loss=_t(0.0),
        state={
            "bbox_smoothl1_contrib": _t(0.4),
            "bbox_ciou_contrib": _t(0.6),
            "bbox_log_wh_contrib": _t(0.1),
            "coord_token_ce_contrib": _t(0.2),
            "coord_soft_ce_contrib": _t(0.3),
            "coord_w1_contrib": _t(0.5),
            "coord_el1_contrib": _t(0.7),
            "coord_gate_contrib": _t(0.9),
            "text_gate_contrib": _t(1.1),
        },
    )
    objective_specs = [
        {"name": "token_ce", "weight": 7.0},
        {"name": "bbox_geo", "weight": 2.0},
        {"name": "bbox_size_aux", "weight": 4.0},
        {"name": "coord_reg", "weight": 3.0},
    ]

    terms = build_stage2_coord_monitor_terms_from_pipeline(
        pipeline_result=pipeline_result,
        objective_specs=objective_specs,
        coord_provenance="B_coord",
    )

    assert set(terms.keys()) == {
        "B_coord/bbox_smoothl1",
        "B_coord/bbox_ciou",
        "B_coord/bbox_log_wh",
        "B_coord/coord_token_ce",
        "B_coord/coord_soft_ce",
        "B_coord/coord_w1",
        "B_coord/coord_el1",
        "B_coord/coord_gate",
    }
    assert float(terms["B_coord/bbox_smoothl1"].detach().cpu().item()) == pytest.approx(0.8)
    assert float(terms["B_coord/bbox_log_wh"].detach().cpu().item()) == pytest.approx(0.4)
    assert float(terms["B_coord/coord_soft_ce"].detach().cpu().item()) == pytest.approx(0.9)
    assert "B_coord/text_gate" not in terms


def test_build_stage2_two_channel_coord_monitor_terms_uses_single_pass_coord_group_for_channel_a() -> None:
    pipeline_result = PipelineResult(
        total_loss=_t(0.0),
        state={
            "bbox_smoothl1_contrib": _t(0.4),
            "bbox_ciou_contrib": _t(0.6),
            "bbox_log_wh_contrib": _t(0.2),
            "coord_soft_ce_contrib": _t(0.2),
            "coord_w1_contrib": _t(0.3),
        },
    )
    objective_specs = [
        {"name": "bbox_geo", "weight": 2.0},
        {"name": "bbox_size_aux", "weight": 3.0},
        {"name": "coord_reg", "weight": 5.0},
    ]

    terms = build_stage2_two_channel_coord_monitor_terms(
        channel="A",
        pipeline_result=pipeline_result,
        objective_specs=objective_specs,
        bbox_module_weight=2.0,
        bbox_size_aux_module_weight=3.0,
        coord_module_weight=5.0,
    )

    assert set(terms.keys()) == {
        "coord/bbox_smoothl1",
        "coord/bbox_ciou",
        "coord/bbox_log_wh",
        "coord/coord_soft_ce",
        "coord/coord_w1",
    }
    assert float(terms["coord/bbox_smoothl1"].detach().cpu().item()) == pytest.approx(0.8)
    assert float(terms["coord/bbox_log_wh"].detach().cpu().item()) == pytest.approx(0.6)
    assert float(terms["coord/coord_w1"].detach().cpu().item()) == pytest.approx(1.5)
    assert all("text_gate" not in key for key in terms)


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
