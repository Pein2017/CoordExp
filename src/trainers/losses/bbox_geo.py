from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.trainers.teacher_forcing.geometry import (
    canonicalize_bbox_xyxy,
    compute_bbox_regression_loss,
    bbox_smoothl1_ciou_loss,
    expectation_decode_coords,
)
from src.trainers.teacher_forcing.stage1 import extract_stage1_bbox_quartets


@dataclass(frozen=True)
class BBoxGeoResult:
    total_loss: torch.Tensor
    smoothl1_loss: torch.Tensor
    ciou_loss: torch.Tensor
    smoothl1_contrib: torch.Tensor
    ciou_contrib: torch.Tensor
    bbox_groups: int
    coord_slots: int


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    raw = cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _cfg_str(cfg: Any, key: str, default: str) -> str:
    raw = cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)
    return str(raw or default)


def compute_stage1_bbox_geo_loss(
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
) -> BBoxGeoResult | None:
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

    pred = expectation_decode_coords(
        coord_logits=quartets.coord_logits,
        temperature=float(max(1e-6, decode_temperature)),
        mode=str(decode_mode or "exp"),
    )
    pred_boxes = canonicalize_bbox_xyxy(pred.reshape(-1, 4))
    target_boxes = canonicalize_bbox_xyxy(quartets.target_boxes_xyxy)

    parameterization = _cfg_str(cfg, "parameterization", "xyxy").strip().lower()
    center_weight = max(0.0, _cfg_float(cfg, "center_weight", 1.0))
    size_weight = max(0.0, _cfg_float(cfg, "size_weight", 1.0))
    regression = compute_bbox_regression_loss(
        pred_boxes_xyxy=pred_boxes,
        target_boxes_xyxy=target_boxes,
        parameterization=parameterization,
        center_weight=center_weight,
        size_weight=size_weight,
    )
    geo = bbox_smoothl1_ciou_loss(
        pred_xyxy=pred_boxes,
        gt_xyxy=target_boxes,
    )

    smoothl1_weight = max(0.0, _cfg_float(cfg, "smoothl1_weight", 0.0))
    ciou_weight = max(0.0, _cfg_float(cfg, "ciou_weight", 0.0))
    smoothl1_contrib = smoothl1_weight * regression.loss
    ciou_contrib = ciou_weight * geo.ciou
    total_loss = smoothl1_contrib + ciou_contrib

    return BBoxGeoResult(
        total_loss=torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0),
        smoothl1_loss=torch.nan_to_num(
            regression.loss, nan=0.0, posinf=0.0, neginf=0.0
        ),
        ciou_loss=torch.nan_to_num(geo.ciou, nan=0.0, posinf=0.0, neginf=0.0),
        smoothl1_contrib=torch.nan_to_num(
            smoothl1_contrib, nan=0.0, posinf=0.0, neginf=0.0
        ),
        ciou_contrib=torch.nan_to_num(ciou_contrib, nan=0.0, posinf=0.0, neginf=0.0),
        bbox_groups=int(quartets.bbox_groups),
        coord_slots=int(quartets.coord_slots),
    )
