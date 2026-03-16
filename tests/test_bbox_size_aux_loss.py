from __future__ import annotations

import pytest
import torch

from src.config.schema import BBoxSizeAuxConfig
from src.trainers.losses.bbox_size_aux import compute_stage1_bbox_size_aux_loss
from src.trainers.teacher_forcing.contracts import TeacherForcingContext
from src.trainers.teacher_forcing.geometry import (
    compute_bbox_log_size_loss,
    compute_bbox_oversize_penalty,
)
from src.trainers.teacher_forcing.objective_atoms import project_stage2_objective_atoms
from src.trainers.teacher_forcing.objective_pipeline import run_teacher_forcing_pipeline


class _DummyTokenizer:
    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        mapping = {
            0: '{"objects":[{"desc":"x","bbox_2d":[',
            1: '{"objects":[{"desc":"x","poly":[',
            5: "text",
        }
        return "".join(mapping.get(int(t), f"<|coord_{int(t) - 100}|>") for t in token_ids)


def test_bbox_log_size_loss_exact_match_is_near_zero() -> None:
    pred = torch.tensor([[0.1, 0.2, 0.4, 0.5]], dtype=torch.float32)
    target = pred.clone()

    result = compute_bbox_log_size_loss(
        pred_boxes_xyxy=pred,
        target_boxes_xyxy=target,
    )

    assert float(result.log_wh.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)
    assert float(result.log_area.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)


def test_bbox_log_size_loss_is_positive_for_larger_predicted_box() -> None:
    pred = torch.tensor([[0.1, 0.2, 0.8, 0.9]], dtype=torch.float32)
    target = torch.tensor([[0.1, 0.2, 0.4, 0.5]], dtype=torch.float32)

    result = compute_bbox_log_size_loss(
        pred_boxes_xyxy=pred,
        target_boxes_xyxy=target,
    )

    assert float(result.log_wh.detach().cpu().item()) > 0.0
    assert float(result.log_area.detach().cpu().item()) > 0.0


def test_bbox_log_size_loss_canonicalizes_reversed_corners() -> None:
    pred = torch.tensor([[0.4, 0.5, 0.1, 0.2]], dtype=torch.float32)
    target = torch.tensor([[0.1, 0.2, 0.4, 0.5]], dtype=torch.float32)

    result = compute_bbox_log_size_loss(
        pred_boxes_xyxy=pred,
        target_boxes_xyxy=target,
    )

    assert float(result.log_wh.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)
    assert float(result.log_area.detach().cpu().item()) == pytest.approx(0.0, abs=1e-8)


def test_bbox_log_size_loss_empty_mask_returns_zero() -> None:
    pred = torch.tensor([[0.1, 0.2, 0.8, 0.9]], dtype=torch.float32)
    target = torch.tensor([[0.1, 0.2, 0.4, 0.5]], dtype=torch.float32)
    mask = torch.tensor([False], dtype=torch.bool)

    result = compute_bbox_log_size_loss(
        pred_boxes_xyxy=pred,
        target_boxes_xyxy=target,
        mask=mask,
    )

    assert float(result.log_wh.detach().cpu().item()) == pytest.approx(0.0)
    assert float(result.log_area.detach().cpu().item()) == pytest.approx(0.0)
    assert result.stats.valid_count == 0


def test_bbox_oversize_penalty_thresholds_are_zero_below_and_positive_above() -> None:
    small = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    large = torch.tensor([[0.1, 0.2, 0.95, 0.98]], dtype=torch.float32)

    penalty_small = compute_bbox_oversize_penalty(
        pred_boxes_xyxy=small,
        log_w_threshold=-0.5,
        log_h_threshold=-0.5,
    )
    penalty_large = compute_bbox_oversize_penalty(
        pred_boxes_xyxy=large,
        log_w_threshold=-0.5,
        log_h_threshold=-0.5,
    )

    assert float(penalty_small.detach().cpu().item()) == pytest.approx(0.0)
    assert float(penalty_large.detach().cpu().item()) > 0.0


def test_stage1_bbox_size_aux_loss_exact_match_is_near_zero() -> None:
    vocab = 1200
    labels = torch.tensor([[0, 5, 103, 104, 107, 108]], dtype=torch.long)
    logits = torch.full((1, labels.shape[1], vocab), -20.0, dtype=torch.float32)
    logits[0, 0, 5] = 20.0
    logits[0, 1, 103] = 20.0
    logits[0, 2, 104] = 20.0
    logits[0, 3, 107] = 20.0
    logits[0, 4, 108] = 20.0

    coord_id_map = torch.full((vocab,), -1, dtype=torch.long)
    for bin_id in range(1000):
        coord_id_map[100 + bin_id] = bin_id

    cfg = BBoxSizeAuxConfig.from_mapping(
        {
            "enabled": True,
            "log_wh_weight": 0.05,
            "log_area_weight": 0.0,
            "oversize_penalty_weight": 0.0,
            "oversize_area_frac_threshold": None,
            "oversize_log_w_threshold": None,
            "oversize_log_h_threshold": None,
            "eps": 1e-6,
        }
    )
    result = compute_stage1_bbox_size_aux_loss(
        logits=logits,
        labels=labels,
        coord_token_ids=[100 + i for i in range(1000)],
        coord_id_map=coord_id_map,
        tokenizer=_DummyTokenizer(),
        cfg=cfg,
        decode_temperature=1.0,
    )

    assert result is not None
    assert result.bbox_groups == 1
    assert float(result.log_wh_loss.detach().cpu().item()) < 1e-6


def test_stage1_bbox_size_aux_loss_rejects_non_bbox_four_n_coord_sequences() -> None:
    vocab = 1200
    labels = torch.tensor([[1, 103, 104, 105, 106, 107, 108, 109, 110]], dtype=torch.long)
    logits = torch.full((1, labels.shape[1], vocab), -20.0, dtype=torch.float32)
    logits[0, :, 103] = 20.0

    coord_id_map = torch.full((vocab,), -1, dtype=torch.long)
    for bin_id in range(1000):
        coord_id_map[100 + bin_id] = bin_id

    cfg = BBoxSizeAuxConfig.from_mapping(
        {
            "enabled": True,
            "log_wh_weight": 0.05,
            "log_area_weight": 0.0,
            "oversize_penalty_weight": 0.0,
            "oversize_area_frac_threshold": None,
            "oversize_log_w_threshold": None,
            "oversize_log_h_threshold": None,
            "eps": 1e-6,
        }
    )
    with pytest.raises(ValueError, match=r"bbox-only Stage-1 supervision"):
        _ = compute_stage1_bbox_size_aux_loss(
            logits=logits,
            labels=labels,
            coord_token_ids=[100 + i for i in range(1000)],
            coord_id_map=coord_id_map,
            tokenizer=_DummyTokenizer(),
            cfg=cfg,
            decode_temperature=1.0,
        )


def test_rollout_teacher_forcing_pipeline_emits_bbox_size_aux_atom() -> None:
    vocab = 1200
    coord_token_ids = [100 + i for i in range(1000)]
    logits = torch.full((1, 5, vocab), -20.0, dtype=torch.float32)
    logits[0, 0, 1099] = 20.0
    logits[0, 1, 1099] = 20.0
    logits[0, 2, 1099] = 20.0
    logits[0, 3, 1099] = 20.0
    input_ids = torch.tensor([[0, 100, 101, 102, 103]], dtype=torch.long)
    meta = [
        {
            "prompt_len": 0,
            "prefix_len": 0,
            "train_len": 5,
            "encoded_len": 5,
            "bbox_groups_prefix": [],
            "bbox_groups_fn": [{"pos": [1, 2, 3, 4], "gt_bins": [0, 1, 2, 3]}],
        }
    ]
    context = TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits,
        meta=meta,
        coord_token_ids=coord_token_ids,
        temperature=1.0,
        decode_mode="exp",
    )
    objective_specs = [
        {
            "name": "bbox_geo",
            "enabled": True,
            "weight": 0.0,
            "channels": ["B"],
            "application": {"preset": "anchor_if_single_iter_else_final"},
            "config": {
                "smoothl1_weight": 0.0,
                "ciou_weight": 0.0,
            },
        },
        {
            "name": "bbox_size_aux",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "anchor_if_single_iter_else_final"},
            "config": {
                "log_wh_weight": 0.05,
                "log_area_weight": 0.0,
                "oversize_penalty_weight": 0.0,
                "oversize_area_frac_threshold": None,
                "oversize_log_w_threshold": None,
                "oversize_log_h_threshold": None,
                "eps": 1.0e-6,
            },
        },
    ]

    pipeline_result = run_teacher_forcing_pipeline(
        context=context,
        objective_specs=objective_specs,
        diagnostics_specs=[],
    )
    atoms = project_stage2_objective_atoms(
        pipeline_result=pipeline_result,
        objective_specs=objective_specs,
        text_provenance="B_rollout_text",
        coord_provenance="B_coord",
        emit_text=False,
        emit_coord=True,
        require_additive=True,
    )

    assert float(pipeline_result.metrics["loss/bbox_log_wh"]) > 0.0
    assert float(atoms["loss/B_coord/bbox_log_wh"]) > 0.0
