from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.common.geometry.bbox_formats import normalize_bbox_format


@dataclass(frozen=True)
class BBoxLossResult:
    smoothl1: torch.Tensor
    ciou: torch.Tensor


@dataclass(frozen=True)
class BBoxRegressionLossResult:
    loss: torch.Tensor
    center: torch.Tensor
    size: torch.Tensor
    per_box: torch.Tensor
    center_per_box: torch.Tensor
    size_per_box: torch.Tensor


@dataclass(frozen=True)
class BBoxSizeStats:
    mean_width: torch.Tensor
    mean_height: torch.Tensor
    mean_log_area: torch.Tensor
    valid_count: int


@dataclass(frozen=True)
class BBoxLogSizeLossResult:
    log_wh: torch.Tensor
    stats: BBoxSizeStats


def expectation_decode_coords(
    *,
    coord_logits: torch.Tensor,
    temperature: float,
    mode: str = "exp",
) -> torch.Tensor:
    if coord_logits.numel() == 0:
        return coord_logits.new_zeros((0,), dtype=torch.float32)

    temp = float(temperature)
    if temp <= 0:
        raise ValueError(f"temperature must be > 0; got {temp}")

    mode_s = str(mode or "exp").strip().lower()
    if mode_s not in {"exp", "st", "hard"}:
        raise ValueError(
            f"coord decode mode must be one of {{'exp','st','hard'}}; got {mode_s!r}"
        )

    probs = torch.softmax(coord_logits.float() / temp, dim=-1)
    bins = torch.arange(0, 1000, device=coord_logits.device, dtype=torch.float32)
    soft = (probs * bins).sum(dim=-1) / 999.0

    if mode_s == "exp":
        return soft

    hard = coord_logits.argmax(dim=-1).to(dtype=torch.float32) / 999.0
    if mode_s == "hard":
        return hard

    return hard + (soft - soft.detach())


def canonicalize_bbox_xyxy(box_xyxy: torch.Tensor) -> torch.Tensor:
    if box_xyxy.ndim != 2 or int(box_xyxy.shape[-1]) != 4:
        raise ValueError("box_xyxy must have shape [N, 4]")

    x1, y1, x2, y2 = box_xyxy.unbind(dim=-1)
    x_lo = torch.minimum(x1, x2).clamp(0.0, 1.0)
    y_lo = torch.minimum(y1, y2).clamp(0.0, 1.0)
    x_hi = torch.maximum(x1, x2).clamp(0.0, 1.0)
    y_hi = torch.maximum(y1, y2).clamp(0.0, 1.0)
    return torch.stack([x_lo, y_lo, x_hi, y_hi], dim=-1)


def bbox_tensor_to_xyxy(
    box_tensor: torch.Tensor,
    *,
    bbox_format: str = "xyxy",
) -> torch.Tensor:
    if box_tensor.ndim != 2 or int(box_tensor.shape[-1]) != 4:
        raise ValueError("box_tensor must have shape [N, 4]")

    bbox_format_norm = normalize_bbox_format(
        bbox_format, path="bbox_format"
    )
    boxes = box_tensor.float()
    if bbox_format_norm == "xyxy":
        return canonicalize_bbox_xyxy(boxes)

    cx, cy, w, h = boxes.unbind(dim=-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return canonicalize_bbox_xyxy(
        torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)
    )


def _reduce_weighted_mean(
    values: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(values.numel()) == 0:
        return values.new_tensor(0.0)

    values_f = values.float().reshape(-1)
    if weights is None:
        out = values_f.mean()
    else:
        weights_f = weights.float().reshape(-1)
        if int(weights_f.numel()) != int(values_f.numel()):
            raise ValueError("weights must align with values")
        positive = weights_f > 0
        if not bool(positive.any().item()):
            return values.new_tensor(0.0)
        values_f = values_f[positive]
        weights_f = weights_f[positive]
        denom = weights_f.sum().clamp(min=1e-6)
        out = (values_f * weights_f).sum() / denom

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=values.dtype)


def compute_bbox_regression_loss(
    *,
    pred_boxes_xyxy: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    parameterization: str = "xyxy",
    center_weight: float = 1.0,
    size_weight: float = 1.0,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> BBoxRegressionLossResult:
    if reduction != "mean":
        raise ValueError("compute_bbox_regression_loss only supports reduction='mean'")
    if pred_boxes_xyxy.ndim != 2 or int(pred_boxes_xyxy.shape[-1]) != 4:
        raise ValueError("pred_boxes_xyxy must have shape [N, 4]")
    if target_boxes_xyxy.ndim != 2 or int(target_boxes_xyxy.shape[-1]) != 4:
        raise ValueError("target_boxes_xyxy must have shape [N, 4]")
    if int(pred_boxes_xyxy.shape[0]) != int(target_boxes_xyxy.shape[0]):
        raise ValueError("pred_boxes_xyxy and target_boxes_xyxy must align")

    parameterization_s = str(parameterization or "xyxy").strip().lower()
    if parameterization_s not in {"xyxy", "center_size"}:
        raise ValueError(
            "bbox regression parameterization must be one of {'xyxy', 'center_size'}"
        )

    pred_raw = pred_boxes_xyxy.float()
    target_raw = target_boxes_xyxy.float()
    pred = canonicalize_bbox_xyxy(pred_raw)
    target = canonicalize_bbox_xyxy(target_raw)
    eff_weights = _coerce_box_weights(
        boxes_xyxy=pred_raw,
        mask=mask,
        weights=weights,
        require_target=target_raw,
    )

    if parameterization_s == "xyxy":
        per_box = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        zeros = per_box.new_zeros(per_box.shape)
        loss = _reduce_weighted_mean(per_box, weights=eff_weights).to(
            dtype=pred_boxes_xyxy.dtype
        )
        z = pred_boxes_xyxy.new_tensor(0.0)
        return BBoxRegressionLossResult(
            loss=loss,
            center=z,
            size=z,
            per_box=torch.nan_to_num(per_box, nan=0.0, posinf=0.0, neginf=0.0).to(
                dtype=pred_boxes_xyxy.dtype
            ),
            center_per_box=zeros.to(dtype=pred_boxes_xyxy.dtype),
            size_per_box=zeros.to(dtype=pred_boxes_xyxy.dtype),
        )

    pred_cx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_cy = (pred[:, 1] + pred[:, 3]) * 0.5
    target_cx = (target[:, 0] + target[:, 2]) * 0.5
    target_cy = (target[:, 1] + target[:, 3]) * 0.5
    center_per_box = (
        F.smooth_l1_loss(pred_cx, target_cx, reduction="none")
        + F.smooth_l1_loss(pred_cy, target_cy, reduction="none")
    ) * 0.5

    pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=float(eps))
    pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=float(eps))
    target_w = (target[:, 2] - target[:, 0]).clamp(min=float(eps))
    target_h = (target[:, 3] - target[:, 1]).clamp(min=float(eps))
    size_per_box = (
        F.smooth_l1_loss(torch.log(pred_w), torch.log(target_w), reduction="none")
        + F.smooth_l1_loss(torch.log(pred_h), torch.log(target_h), reduction="none")
    ) * 0.5

    per_box = float(center_weight) * center_per_box + float(size_weight) * size_per_box
    return BBoxRegressionLossResult(
        loss=_reduce_weighted_mean(per_box, weights=eff_weights).to(
            dtype=pred_boxes_xyxy.dtype
        ),
        center=_reduce_weighted_mean(center_per_box, weights=eff_weights).to(
            dtype=pred_boxes_xyxy.dtype
        ),
        size=_reduce_weighted_mean(size_per_box, weights=eff_weights).to(
            dtype=pred_boxes_xyxy.dtype
        ),
        per_box=torch.nan_to_num(per_box, nan=0.0, posinf=0.0, neginf=0.0).to(
            dtype=pred_boxes_xyxy.dtype
        ),
        center_per_box=torch.nan_to_num(
            center_per_box, nan=0.0, posinf=0.0, neginf=0.0
        ).to(dtype=pred_boxes_xyxy.dtype),
        size_per_box=torch.nan_to_num(
            size_per_box, nan=0.0, posinf=0.0, neginf=0.0
        ).to(dtype=pred_boxes_xyxy.dtype),
    )


def _coerce_box_weights(
    *,
    boxes_xyxy: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    require_target: torch.Tensor | None = None,
) -> torch.Tensor:
    if boxes_xyxy.ndim != 2 or int(boxes_xyxy.shape[-1]) != 4:
        raise ValueError("boxes_xyxy must have shape [N, 4]")

    n = int(boxes_xyxy.shape[0])
    out = torch.ones((n,), device=boxes_xyxy.device, dtype=torch.float32)

    if mask is not None:
        mask_t = mask.to(device=boxes_xyxy.device)
        if mask_t.ndim != 1 or int(mask_t.numel()) != n:
            raise ValueError("mask must have shape [N]")
        out = out * mask_t.to(dtype=torch.float32)

    if weights is not None:
        weights_t = weights.to(device=boxes_xyxy.device, dtype=torch.float32)
        if weights_t.ndim != 1 or int(weights_t.numel()) != n:
            raise ValueError("weights must have shape [N]")
        out = out * weights_t

    valid = torch.isfinite(boxes_xyxy.float()).all(dim=-1)
    if require_target is not None:
        valid = valid & torch.isfinite(require_target.float()).all(dim=-1)

    out = out * valid.to(dtype=torch.float32)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_bbox_size_stats(
    pred_boxes_xyxy: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> BBoxSizeStats:
    pred_raw = pred_boxes_xyxy.float()
    if pred_raw.numel() == 0:
        z = pred_boxes_xyxy.new_tensor(0.0)
        return BBoxSizeStats(mean_width=z, mean_height=z, mean_log_area=z, valid_count=0)

    eff_weights = _coerce_box_weights(
        boxes_xyxy=pred_raw,
        mask=mask,
        weights=weights,
    )
    if not bool((eff_weights > 0).any().item()):
        z = pred_boxes_xyxy.new_tensor(0.0)
        return BBoxSizeStats(mean_width=z, mean_height=z, mean_log_area=z, valid_count=0)

    pred = canonicalize_bbox_xyxy(pred_raw)
    widths = (pred[:, 2] - pred[:, 0]).clamp(min=float(eps))
    heights = (pred[:, 3] - pred[:, 1]).clamp(min=float(eps))
    log_area = torch.log((widths * heights).clamp(min=float(eps)))

    return BBoxSizeStats(
        mean_width=_reduce_weighted_mean(widths, weights=eff_weights),
        mean_height=_reduce_weighted_mean(heights, weights=eff_weights),
        mean_log_area=_reduce_weighted_mean(log_area, weights=eff_weights),
        valid_count=int((eff_weights > 0).sum().detach().item()),
    )


def compute_bbox_log_size_loss(
    *,
    pred_boxes_xyxy: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> BBoxLogSizeLossResult:
    if reduction != "mean":
        raise ValueError("compute_bbox_log_size_loss only supports reduction='mean'")
    if pred_boxes_xyxy.ndim != 2 or int(pred_boxes_xyxy.shape[-1]) != 4:
        raise ValueError("pred_boxes_xyxy must have shape [N, 4]")
    if target_boxes_xyxy.ndim != 2 or int(target_boxes_xyxy.shape[-1]) != 4:
        raise ValueError("target_boxes_xyxy must have shape [N, 4]")
    if int(pred_boxes_xyxy.shape[0]) != int(target_boxes_xyxy.shape[0]):
        raise ValueError("pred_boxes_xyxy and target_boxes_xyxy must align")

    pred_raw = pred_boxes_xyxy.float()
    target_raw = target_boxes_xyxy.float()
    eff_weights = _coerce_box_weights(
        boxes_xyxy=pred_raw,
        mask=mask,
        weights=weights,
        require_target=target_raw,
    )
    stats = summarize_bbox_size_stats(
        pred_boxes_xyxy=pred_boxes_xyxy,
        mask=mask,
        weights=weights,
        eps=eps,
    )
    if not bool((eff_weights > 0).any().item()):
        z = pred_boxes_xyxy.new_tensor(0.0)
        return BBoxLogSizeLossResult(log_wh=z, stats=stats)

    pred = canonicalize_bbox_xyxy(pred_raw)
    target = canonicalize_bbox_xyxy(target_raw)

    pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=float(eps))
    pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=float(eps))
    target_w = (target[:, 2] - target[:, 0]).clamp(min=float(eps))
    target_h = (target[:, 3] - target[:, 1]).clamp(min=float(eps))

    log_wh_per = F.smooth_l1_loss(
        torch.log(pred_w),
        torch.log(target_w),
        reduction="none",
    ) + F.smooth_l1_loss(
        torch.log(pred_h),
        torch.log(target_h),
        reduction="none",
    )
    log_wh = _reduce_weighted_mean(log_wh_per, weights=eff_weights).to(dtype=pred_boxes_xyxy.dtype)
    return BBoxLogSizeLossResult(log_wh=log_wh, stats=stats)


def compute_bbox_oversize_penalty(
    *,
    pred_boxes_xyxy: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    area_frac_threshold: float | None = None,
    log_w_threshold: float | None = None,
    log_h_threshold: float | None = None,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction != "mean":
        raise ValueError("compute_bbox_oversize_penalty only supports reduction='mean'")
    if pred_boxes_xyxy.ndim != 2 or int(pred_boxes_xyxy.shape[-1]) != 4:
        raise ValueError("pred_boxes_xyxy must have shape [N, 4]")
    if pred_boxes_xyxy.numel() == 0:
        return pred_boxes_xyxy.new_tensor(0.0)
    if area_frac_threshold is None and log_w_threshold is None and log_h_threshold is None:
        return pred_boxes_xyxy.new_tensor(0.0)

    pred_raw = pred_boxes_xyxy.float()
    eff_weights = _coerce_box_weights(
        boxes_xyxy=pred_raw,
        mask=mask,
        weights=weights,
    )
    if not bool((eff_weights > 0).any().item()):
        return pred_boxes_xyxy.new_tensor(0.0)

    pred = canonicalize_bbox_xyxy(pred_raw)
    widths = (pred[:, 2] - pred[:, 0]).clamp(min=float(eps))
    heights = (pred[:, 3] - pred[:, 1]).clamp(min=float(eps))
    penalty = pred_raw.new_zeros((int(pred.shape[0]),), dtype=torch.float32)

    if log_w_threshold is not None:
        penalty = penalty + torch.relu(torch.log(widths) - float(log_w_threshold))
    if log_h_threshold is not None:
        penalty = penalty + torch.relu(torch.log(heights) - float(log_h_threshold))
    if area_frac_threshold is not None:
        penalty = penalty + torch.relu(widths * heights - float(area_frac_threshold))

    return _reduce_weighted_mean(penalty, weights=eff_weights).to(dtype=pred_boxes_xyxy.dtype)


def bbox_smoothl1_ciou_loss(
    *,
    pred_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    eps: float = 1e-7,
) -> BBoxLossResult:
    if pred_xyxy.numel() == 0:
        z = pred_xyxy.new_tensor(0.0)
        return BBoxLossResult(smoothl1=z, ciou=z)

    pred_raw = pred_xyxy.float()
    gt_raw = gt_xyxy.float()

    smoothl1 = F.smooth_l1_loss(pred_raw, gt_raw, reduction="mean")

    pred = canonicalize_bbox_xyxy(pred_raw)
    gt = canonicalize_bbox_xyxy(gt_raw)

    px1, py1, px2, py2 = pred.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = gt.unbind(dim=-1)

    inter_x1 = torch.maximum(px1, gx1)
    inter_y1 = torch.maximum(py1, gy1)
    inter_x2 = torch.minimum(px2, gx2)
    inter_y2 = torch.minimum(py2, gy2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_p = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    area_g = (gx2 - gx1).clamp(min=0.0) * (gy2 - gy1).clamp(min=0.0)
    union = (area_p + area_g - inter).clamp(min=eps)
    iou = inter / union

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    gcx = (gx1 + gx2) * 0.5
    gcy = (gy1 + gy2) * 0.5
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    enc_x1 = torch.minimum(px1, gx1)
    enc_y1 = torch.minimum(py1, gy1)
    enc_x2 = torch.maximum(px2, gx2)
    enc_y2 = torch.maximum(py2, gy2)
    c2 = ((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2).clamp(min=eps)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)
    v = (4.0 / (math.pi**2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    ciou_loss = (1.0 - ciou).mean()

    smoothl1 = torch.nan_to_num(smoothl1, nan=0.0, posinf=0.0, neginf=0.0)
    ciou_loss = torch.nan_to_num(ciou_loss, nan=0.0, posinf=0.0, neginf=0.0)

    return BBoxLossResult(
        smoothl1=smoothl1.to(dtype=pred_xyxy.dtype),
        ciou=ciou_loss.to(dtype=pred_xyxy.dtype),
    )
