from __future__ import annotations

import pytest
import torch

from src.config.schema import Stage2ABChannelBConfig
from src.trainers.rollout_matching.contracts import GTObject
from src.trainers.stage2_two_channel import _deterministic_max_iou_assignment
from src.trainers.teacher_forcing.contracts import PipelineModuleSpec, TeacherForcingContext
from src.trainers.teacher_forcing.modules.coord_reg import run_coord_reg_module


def test_stage2_ab_channel_b_v3_k2_config_parse() -> None:
    cfg = Stage2ABChannelBConfig.from_mapping(
        {
            "v3_k2": {
                "enabled": True,
                "recovered_fn_weight": 2.5,
                "anchor_decode": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
                "explorer_decode": {"temperature": 0.7, "top_p": 0.95, "top_k": 32},
                "pairing": {"iou_threshold": 0.9},
            }
        }
    )
    assert cfg.v3_k2.enabled is True
    assert cfg.v3_k2.recovered_fn_weight == pytest.approx(2.5)
    assert cfg.v3_k2.explorer_decode.temperature == pytest.approx(0.7)
    assert cfg.v3_k2.explorer_decode.top_p == pytest.approx(0.95)
    assert cfg.v3_k2.explorer_decode.top_k == 32


def test_deterministic_max_iou_assignment_all_equal_uses_stable_pair_order() -> None:
    anchors = [
        GTObject(index=0, geom_type="bbox_2d", points_norm1000=[0, 0, 10, 10], desc="a0"),
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[0, 0, 10, 10], desc="a1"),
    ]
    explorers = [
        GTObject(index=0, geom_type="bbox_2d", points_norm1000=[0, 0, 10, 10], desc="e0"),
        GTObject(index=1, geom_type="bbox_2d", points_norm1000=[0, 0, 10, 10], desc="e1"),
    ]
    pairs = _deterministic_max_iou_assignment(
        anchor_objects=anchors,
        explorer_objects=explorers,
        iou_threshold=0.5,
    )
    assert pairs == [(0, 0), (1, 1)]


def test_coord_reg_uses_coord_slot_weights_for_weighted_mean() -> None:
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]]], dtype=torch.float32)
    coord_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32)
    target_bins = torch.tensor([0, 0], dtype=torch.long)
    slot_weights = torch.tensor([10.0, 1.0], dtype=torch.float32)

    context = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=torch.tensor([[1, 1, 1]], dtype=torch.long),
        logits=logits,
        logits_ce=logits,
        meta=[{"prompt_len": 0, "prefix_len": 0, "train_len": 0}],
        coord_token_ids=[0, 1],
        temperature=1.0,
        decode_mode="exp",
    )
    spec = PipelineModuleSpec(
        name="coord_reg",
        enabled=True,
        weight=1.0,
        channels=("B",),
        config={"coord_ce_weight": 1.0},
    )

    out = run_coord_reg_module(
        context=context,
        spec=spec,
        state={
            "coord_logits": coord_logits,
            "coord_logits_full": coord_logits,
            "coord_target_bins": target_bins,
            "coord_slot_weights": slot_weights,
        },
    )

    ce = torch.nn.functional.cross_entropy(coord_logits, target_bins, reduction="none")
    expected = ((ce * slot_weights).sum() / slot_weights.sum()).item()
    assert out.metrics["loss/coord_token_ce"] == pytest.approx(expected)
