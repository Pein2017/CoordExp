from __future__ import annotations

from typing import Any, Mapping

import torch

from src.trainers.losses.bbox_size_aux import compute_bbox_size_aux_from_boxes

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("bbox_size_aux threshold values must be numeric when provided") from None


def _term_weight(cfg: Mapping[str, Any], *, key: str) -> float:
    return max(0.0, _coerce_float(cfg.get(key, 0.0), 0.0))


def run_bbox_size_aux_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
    state: Mapping[str, Any],
) -> ModuleResult:
    cfg = spec.config if isinstance(spec.config, Mapping) else {}

    pred_boxes = state.get("bbox_pred_boxes_xyxy")
    target_boxes = state.get("bbox_target_boxes_xyxy")
    box_weights = state.get("bbox_group_weights")
    if not isinstance(pred_boxes, torch.Tensor) or int(pred_boxes.numel()) == 0:
        z = context.logits.new_tensor(0.0)
        metrics = {
            "loss/bbox_log_wh": 0.0,
            "loss/bbox_oversize": 0.0,
            "bbox_size_aux/groups_total": 0.0,
            "bbox_size_aux/mean_width": 0.0,
            "bbox_size_aux/mean_height": 0.0,
            "bbox_size_aux/mean_log_area": 0.0,
        }
        return ModuleResult(loss=z, metrics=metrics, state={"bbox_size_aux": z})

    result = compute_bbox_size_aux_from_boxes(
        pred_boxes_xyxy=pred_boxes,
        target_boxes_xyxy=target_boxes if isinstance(target_boxes, torch.Tensor) else None,
        box_weights=box_weights if isinstance(box_weights, torch.Tensor) else None,
        log_wh_weight=_term_weight(cfg, key="log_wh_weight"),
        oversize_penalty_weight=_term_weight(cfg, key="oversize_penalty_weight"),
        oversize_area_frac_threshold=_coerce_optional_float(cfg.get("oversize_area_frac_threshold")),
        oversize_log_w_threshold=_coerce_optional_float(cfg.get("oversize_log_w_threshold")),
        oversize_log_h_threshold=_coerce_optional_float(cfg.get("oversize_log_h_threshold")),
        eps=max(1e-9, _coerce_float(cfg.get("eps", 1e-6), 1e-6)),
    )

    metrics = {
        "loss/bbox_log_wh": float(result.log_wh_loss.detach().cpu().item()),
        "loss/bbox_oversize": float(result.oversize_loss.detach().cpu().item()),
        "bbox_size_aux/groups_total": float(result.bbox_groups),
        "bbox_size_aux/mean_width": float(result.stats.mean_width.detach().cpu().item()),
        "bbox_size_aux/mean_height": float(result.stats.mean_height.detach().cpu().item()),
        "bbox_size_aux/mean_log_area": float(result.stats.mean_log_area.detach().cpu().item()),
    }
    module_loss = result.total_loss.to(dtype=context.logits.dtype)
    state_out = {
        "bbox_size_aux": module_loss,
        "bbox_log_wh_contrib": result.log_wh_contrib.to(dtype=context.logits.dtype),
        "bbox_oversize_contrib": result.oversize_contrib.to(dtype=context.logits.dtype),
    }
    return ModuleResult(loss=module_loss, metrics=metrics, state=state_out)
