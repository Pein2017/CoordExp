from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BBoxLossResult:
    smoothl1: torch.Tensor
    ciou: torch.Tensor


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
