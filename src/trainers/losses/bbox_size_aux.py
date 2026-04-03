from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.trainers.teacher_forcing.geometry import (
    BBoxSizeStats,
    compute_bbox_log_size_loss,
    compute_bbox_oversize_penalty,
    expectation_decode_coords,
    summarize_bbox_size_stats,
)
from src.trainers.teacher_forcing.stage1 import extract_stage1_bbox_quartets


@dataclass(frozen=True)
class BBoxSizeAuxResult:
    total_loss: torch.Tensor
    log_wh_loss: torch.Tensor
    oversize_loss: torch.Tensor
    log_wh_contrib: torch.Tensor
    oversize_contrib: torch.Tensor
    stats: BBoxSizeStats
    bbox_groups: int
    coord_slots: int
    skipped_incomplete_rows: int = 0
    skipped_incomplete_coord_slots: int = 0


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    raw = cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _cfg_optional_float(cfg: Any, key: str) -> float | None:
    raw = cfg.get(key, None) if isinstance(cfg, Mapping) else getattr(cfg, key, None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be numeric when provided") from None


def compute_bbox_size_aux_from_boxes(
    *,
    pred_boxes_xyxy: torch.Tensor,
    target_boxes_xyxy: torch.Tensor | None = None,
    box_weights: torch.Tensor | None = None,
    log_wh_weight: float = 0.0,
    oversize_penalty_weight: float = 0.0,
    oversize_area_frac_threshold: float | None = None,
    oversize_log_w_threshold: float | None = None,
    oversize_log_h_threshold: float | None = None,
    eps: float = 1e-6,
) -> BBoxSizeAuxResult:
    zero = pred_boxes_xyxy.new_tensor(0.0)
    stats = summarize_bbox_size_stats(
        pred_boxes_xyxy,
        weights=box_weights,
        eps=eps,
    )

    log_wh_loss = zero
    if target_boxes_xyxy is not None and float(log_wh_weight) != 0.0:
        matched = compute_bbox_log_size_loss(
            pred_boxes_xyxy=pred_boxes_xyxy,
            target_boxes_xyxy=target_boxes_xyxy,
            weights=box_weights,
            eps=eps,
        )
        log_wh_loss = matched.log_wh
        stats = matched.stats

    oversize_loss = zero
    if float(oversize_penalty_weight) != 0.0:
        oversize_loss = compute_bbox_oversize_penalty(
            pred_boxes_xyxy=pred_boxes_xyxy,
            weights=box_weights,
            area_frac_threshold=oversize_area_frac_threshold,
            log_w_threshold=oversize_log_w_threshold,
            log_h_threshold=oversize_log_h_threshold,
            eps=eps,
        )

    log_wh_contrib = float(max(0.0, log_wh_weight)) * log_wh_loss
    oversize_contrib = float(max(0.0, oversize_penalty_weight)) * oversize_loss
    total_loss = log_wh_contrib + oversize_contrib

    return BBoxSizeAuxResult(
        total_loss=torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0),
        log_wh_loss=torch.nan_to_num(log_wh_loss, nan=0.0, posinf=0.0, neginf=0.0),
        oversize_loss=torch.nan_to_num(oversize_loss, nan=0.0, posinf=0.0, neginf=0.0),
        log_wh_contrib=torch.nan_to_num(log_wh_contrib, nan=0.0, posinf=0.0, neginf=0.0),
        oversize_contrib=torch.nan_to_num(oversize_contrib, nan=0.0, posinf=0.0, neginf=0.0),
        stats=stats,
        bbox_groups=int(pred_boxes_xyxy.shape[0]) if pred_boxes_xyxy.ndim == 2 else 0,
        coord_slots=int(pred_boxes_xyxy.shape[0] * 4) if pred_boxes_xyxy.ndim == 2 else 0,
    )


def compute_stage1_bbox_size_aux_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    coord_token_ids: list[int],
    coord_id_map: torch.Tensor,
    tokenizer: Any | None,
    cfg: Any,
    decode_temperature: float,
    decode_mode: str = "exp",
    object_field_order: str = "desc_first",
) -> BBoxSizeAuxResult | None:
    quartets = extract_stage1_bbox_quartets(
        logits=logits,
        labels=labels,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        tokenizer=tokenizer,
        object_field_order=object_field_order,
    )
    if quartets is None:
        return None
    if int(quartets.bbox_groups) <= 0 or int(quartets.coord_slots) <= 0:
        zero = logits.new_zeros(())
        return BBoxSizeAuxResult(
            total_loss=zero,
            log_wh_loss=zero,
            oversize_loss=zero,
            log_wh_contrib=zero,
            oversize_contrib=zero,
            stats=compute_bbox_size_aux_from_boxes(
                pred_boxes_xyxy=logits.new_zeros((0, 4)),
                target_boxes_xyxy=logits.new_zeros((0, 4)),
                box_weights=None,
                log_wh_weight=0.0,
                oversize_penalty_weight=0.0,
                oversize_area_frac_threshold=None,
                oversize_log_w_threshold=None,
                oversize_log_h_threshold=None,
                eps=max(1e-9, _cfg_float(cfg, "eps", 1e-6)),
            ).stats,
            bbox_groups=0,
            coord_slots=0,
            skipped_incomplete_rows=int(quartets.skipped_incomplete_rows),
            skipped_incomplete_coord_slots=int(quartets.skipped_incomplete_coord_slots),
        )

    pred = expectation_decode_coords(
        coord_logits=quartets.coord_logits,
        temperature=float(max(1e-6, decode_temperature)),
        mode=str(decode_mode or "exp"),
    )
    pred_boxes = pred.reshape(-1, 4)

    result = compute_bbox_size_aux_from_boxes(
        pred_boxes_xyxy=pred_boxes,
        target_boxes_xyxy=quartets.target_boxes_xyxy,
        box_weights=None,
        log_wh_weight=_cfg_float(cfg, "log_wh_weight", 0.0),
        oversize_penalty_weight=_cfg_float(cfg, "oversize_penalty_weight", 0.0),
        oversize_area_frac_threshold=_cfg_optional_float(cfg, "oversize_area_frac_threshold"),
        oversize_log_w_threshold=_cfg_optional_float(cfg, "oversize_log_w_threshold"),
        oversize_log_h_threshold=_cfg_optional_float(cfg, "oversize_log_h_threshold"),
        eps=max(1e-9, _cfg_float(cfg, "eps", 1e-6)),
    )

    return BBoxSizeAuxResult(
        total_loss=result.total_loss,
        log_wh_loss=result.log_wh_loss,
        oversize_loss=result.oversize_loss,
        log_wh_contrib=result.log_wh_contrib,
        oversize_contrib=result.oversize_contrib,
        stats=result.stats,
        bbox_groups=int(quartets.bbox_groups),
        coord_slots=int(quartets.coord_slots),
        skipped_incomplete_rows=int(quartets.skipped_incomplete_rows),
        skipped_incomplete_coord_slots=int(quartets.skipped_incomplete_coord_slots),
    )
