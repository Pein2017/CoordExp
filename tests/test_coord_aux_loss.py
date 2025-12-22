from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import torch

from src.config.schema import CoordLossConfig
from src.metrics.dataset_metrics import CoordAuxLossMixin


class DummyMetric:
    def __init__(self):
        self.values = []

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        self.values.append(float(value))


class DummyTokenizer:
    def convert_tokens_to_ids(self, tokens):
        out = []
        for tok in tokens:
            if tok.startswith("<|coord_") and tok.endswith("|>"):
                out.append(int(tok[len("<|coord_") : -len("|>")] ))
            else:
                out.append(0)
        return out


class DummyTemplate:
    def __init__(self):
        self.tokenizer = DummyTokenizer()


class DummyTrainer(CoordAuxLossMixin):
    def __init__(self, coord_loss_cfg):
        self.coord_loss_cfg = coord_loss_cfg
        self.custom_metrics = {"train": defaultdict(DummyMetric)}
        self.template = DummyTemplate()
        self.model = None


def _make_logits(coord_tokens, vocab_size=1100):
    seq_len = len(coord_tokens)
    logits = torch.full((1, seq_len, vocab_size), -10.0)
    for idx, token_id in enumerate(coord_tokens):
        logits[0, idx, token_id] = 10.0
    return logits


def test_poly_mask_iou_empty_is_zero():
    empty = torch.empty((0, 2), dtype=torch.float32)
    iou = CoordAuxLossMixin._poly_mask_iou(
        empty,
        empty,
        mask_size=16,
        sigma_mask=1.5 / 16.0,
        tau_inside=0.08,
        beta_dist=100.0,
    )
    assert float(iou) == 0.0


def test_poly_mask_iou_handles_nan_inputs():
    pred = torch.tensor(
        [[float("nan"), 0.1], [0.6, float("inf")], [0.6, 0.6]],
        dtype=torch.float32,
    )
    target = torch.tensor([[0.1, 0.1], [0.6, 0.1], [0.6, 0.6]], dtype=torch.float32)
    iou = CoordAuxLossMixin._poly_mask_iou(
        pred,
        target,
        mask_size=16,
        sigma_mask=1.5 / 16.0,
        tau_inside=0.08,
        beta_dist=100.0,
    )
    assert torch.isfinite(iou).item()
    assert 0.0 <= float(iou) <= 1.0


def test_coord_aux_loss_teacher_forcing_poly_and_bbox():
    # bbox coords: (0.1,0.1)-(0.6,0.6)
    bbox_tokens = [100, 100, 600, 600]
    # poly coords: rectangle corners
    poly_tokens = [100, 100, 600, 100, 600, 600, 100, 600]
    coord_tokens = bbox_tokens + poly_tokens

    labels = torch.tensor([[0] + coord_tokens], dtype=torch.long)
    logits = _make_logits(coord_tokens)
    outputs = SimpleNamespace(logits=logits)

    coord_cfg = CoordLossConfig.from_mapping(
        {
            "enabled": True,
            "l1_weight": 1.0,
            "giou_weight": 1.0,
            "top_k": 1,
            "temperature": 1.0,
            "poly_mask_size": 32,
            "poly_tau_inside": 0.08,
            "poly_beta_dist": 100.0,
            "poly_smooth_weight": 0.05,
        }
    )

    coord_spans = [
        [
            {"geom_type": "bbox_2d", "coord_len": 4, "start": 0},
            {"geom_type": "poly", "coord_len": 8, "start": 4},
        ]
    ]

    trainer = DummyTrainer(coord_cfg)
    base_loss = torch.tensor(0.0)
    loss = trainer._maybe_add_coord_aux_loss(base_loss, outputs, labels, coord_spans)

    metrics = trainer.custom_metrics["train"]
    poly_iou = metrics["coord_loss/poly_mask_iou"].values[-1]
    poly_smooth = metrics["coord_loss/poly_smooth"].values[-1]
    giou_loss = metrics["coord_loss/giou"].values[-1]

    assert float(loss) >= 0.0
    assert poly_iou > 0.4
    assert poly_smooth > 0.0
    assert abs(giou_loss) < 1e-6

    shifted_poly_tokens = [200, 200, 700, 200, 700, 700, 200, 700]
    shifted_tokens = bbox_tokens + shifted_poly_tokens
    shifted_logits = _make_logits(shifted_tokens)
    shifted_outputs = SimpleNamespace(logits=shifted_logits)
    trainer_shifted = DummyTrainer(coord_cfg)
    _ = trainer_shifted._maybe_add_coord_aux_loss(
        base_loss, shifted_outputs, labels, coord_spans
    )
    shifted_metrics = trainer_shifted.custom_metrics["train"]
    shifted_poly_iou = shifted_metrics["coord_loss/poly_mask_iou"].values[-1]
    assert poly_iou > shifted_poly_iou
