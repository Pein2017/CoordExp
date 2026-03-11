from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..geometry import expectation_decode_coords
from ..token_types import iter_segment_views


@dataclass(frozen=True)
class _DecodedGroups:
    smoothl1: torch.Tensor
    ciou: torch.Tensor
    n_groups: int
    n_slots: int
    logits_full: torch.Tensor
    coord_logits: torch.Tensor
    target_bins: torch.Tensor
    slot_weights: torch.Tensor


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _flatten_groups(
    *,
    key: str,
    input_ids: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int], list[int], list[float]]:
    b_list: List[int] = []
    pos_list: List[int] = []
    bins_list: List[int] = []
    weights_list: List[float] = []

    for b, seg_start, _seg_end, seg in iter_segment_views(input_ids=input_ids, meta=meta):
        groups = seg.get(key) or []
        for g in groups:
            if not isinstance(g, Mapping):
                continue
            pos = g.get("pos")
            gb = g.get("gt_bins")
            if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
                continue
            if len(pos) != 4 or len(gb) != 4:
                continue
            for p, tbin in zip(pos, gb):
                b_list.append(int(b))
                pos_list.append(int(seg_start + int(p)) if len(meta) != int(input_ids.shape[0]) else int(p))
                bins_list.append(int(tbin))
                w_raw = g.get("weight", 1.0) if isinstance(g, Mapping) else 1.0
                try:
                    w = float(w_raw)
                except (TypeError, ValueError):
                    w = 1.0
                if w <= 0.0:
                    w = 1.0
                weights_list.append(float(w))

    return b_list, pos_list, bins_list, weights_list


def _decode_groups(
    *,
    key: str,
    context: TeacherForcingContext,
    coord_ids_t: torch.Tensor,
) -> _DecodedGroups:
    input_ids = context.input_ids
    logits = context.logits
    meta = context.meta

    b_list, pos_list, bins_list, weights_list = _flatten_groups(
        key=key,
        input_ids=input_ids,
        meta=meta,
    )
    if not pos_list:
        z = logits.new_tensor(0.0)
        empty = logits.new_zeros((0, logits.shape[-1]))
        empty_coord = logits.new_zeros((0, coord_ids_t.numel()))
        empty_bins = logits.new_zeros((0,), dtype=torch.long)
        empty_w = logits.new_zeros((0,), dtype=torch.float32)
        return _DecodedGroups(
            smoothl1=z,
            ciou=z,
            n_groups=0,
            n_slots=0,
            logits_full=empty,
            coord_logits=empty_coord,
            target_bins=empty_bins,
            slot_weights=empty_w,
        )

    if len(pos_list) % 4 != 0:
        raise ValueError(f"{key} coord slots must be a multiple of 4")

    b_t = torch.tensor(b_list, device=logits.device, dtype=torch.long)
    pos_t = torch.tensor(pos_list, device=logits.device, dtype=torch.long)
    bin_t = torch.tensor(bins_list, device=logits.device, dtype=torch.long).clamp(min=0, max=999)
    weight_t = torch.tensor(weights_list, device=logits.device, dtype=torch.float32)

    b_g = b_t.reshape(-1, 4)
    pos_g = pos_t.reshape(-1, 4)
    bin_g = bin_t.reshape(-1, 4)
    weight_g = weight_t.reshape(-1, 4)

    seq_len = int(logits.shape[1])
    valid = (pos_g > 0).all(dim=1)
    valid &= (pos_g < int(seq_len)).all(dim=1)
    if not bool(valid.all().item()):
        bad_i = int((~valid).nonzero(as_tuple=False)[0].item())
        bad_pos = [int(x) for x in pos_g[bad_i].detach().cpu().tolist()]
        raise ValueError(
            f"{key} contains out-of-range bbox group positions (example={bad_pos}, seq_len={int(seq_len)})"
        )

    n_groups = int(pos_g.shape[0])
    if n_groups == 0:
        z = logits.new_tensor(0.0)
        empty = logits.new_zeros((0, logits.shape[-1]))
        empty_coord = logits.new_zeros((0, coord_ids_t.numel()))
        empty_bins = logits.new_zeros((0,), dtype=torch.long)
        empty_w = logits.new_zeros((0,), dtype=torch.float32)
        return _DecodedGroups(
            smoothl1=z,
            ciou=z,
            n_groups=0,
            n_slots=0,
            logits_full=empty,
            coord_logits=empty_coord,
            target_bins=empty_bins,
            slot_weights=empty_w,
        )

    b_t = b_g.reshape(-1)
    pos_t = pos_g.reshape(-1)
    bin_t = bin_g.reshape(-1)

    logits_prev = logits[b_t, pos_t - 1]
    coord_logits = logits_prev.index_select(dim=-1, index=coord_ids_t)

    pred = expectation_decode_coords(
        coord_logits=coord_logits,
        temperature=float(context.temperature),
        mode=str(context.decode_mode or "exp"),
    )
    gt = bin_t.float() / 999.0

    pred_box = pred.reshape(-1, 4)
    gt_box = gt.reshape(-1, 4)
    smoothl1_per_box = torch.nn.functional.smooth_l1_loss(
        pred_box,
        gt_box,
        reduction="none",
    ).mean(dim=-1)

    eps = 1e-7
    px1, py1, px2, py2 = pred_box.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = gt_box.unbind(dim=-1)
    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
    area_p = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    area_g = (gx2 - gx1).clamp(min=0.0) * (gy2 - gy1).clamp(min=0.0)
    union = (area_p + area_g - inter).clamp(min=eps)
    iou = inter / union
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    gcx = (gx1 + gx2) * 0.5
    gcy = (gy1 + gy2) * 0.5
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    ex1 = torch.minimum(px1, gx1)
    ey1 = torch.minimum(py1, gy1)
    ex2 = torch.maximum(px2, gx2)
    ey2 = torch.maximum(py2, gy2)
    c2 = ((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2).clamp(min=eps)
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)
    v = (4.0 / (math.pi**2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    alpha = v / (1.0 - iou + v + eps)
    ciou_per_box = 1.0 - (iou - (rho2 / c2 + alpha * v))
    group_weights = weight_g.mean(dim=1)
    denom = group_weights.sum().clamp(min=1e-6)
    smoothl1 = (smoothl1_per_box * group_weights).sum() / denom
    ciou = (ciou_per_box * group_weights).sum() / denom

    return _DecodedGroups(
        smoothl1=smoothl1,
        ciou=ciou,
        n_groups=int(n_groups),
        n_slots=int(pos_t.numel()),
        logits_full=logits_prev,
        coord_logits=coord_logits,
        target_bins=bin_t,
        slot_weights=weight_t,
    )


def run_bbox_geo_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    if str(context.registry_context) == "gt":
        z = context.logits.new_tensor(0.0)
        metrics = {
            "loss/geo": 0.0,
            "loss/bbox_smoothl1": 0.0,
            "loss/bbox_ciou": 0.0,
            "bbox/groups_total": 0.0,
            "bbox/slots_total": 0.0,
        }
        return ModuleResult(loss=z, metrics=metrics, state={})

    cfg = spec.config if isinstance(spec.config, Mapping) else {}

    bbox_smoothl1_w = max(
        0.0,
        _coerce_float(
            cfg.get("smoothl1_weight", 1.0),
            1.0,
        ),
    )
    bbox_ciou_w = max(
        0.0,
        _coerce_float(
            cfg.get("ciou_weight", 1.0),
            1.0,
        ),
    )

    coord_ids_t = torch.tensor(
        [int(i) for i in context.coord_token_ids],
        device=context.logits.device,
        dtype=torch.long,
    )

    dec_prefix = _decode_groups(key="bbox_groups_prefix", context=context, coord_ids_t=coord_ids_t)
    dec_fn = _decode_groups(key="bbox_groups_fn", context=context, coord_ids_t=coord_ids_t)

    n_groups_all = int(dec_prefix.n_groups + dec_fn.n_groups)
    if n_groups_all > 0:
        smoothl1 = (
            dec_prefix.smoothl1 * float(dec_prefix.n_groups)
            + dec_fn.smoothl1 * float(dec_fn.n_groups)
        ) / float(n_groups_all)
        ciou = (
            dec_prefix.ciou * float(dec_prefix.n_groups)
            + dec_fn.ciou * float(dec_fn.n_groups)
        ) / float(n_groups_all)
    else:
        smoothl1 = context.logits.new_tensor(0.0)
        ciou = context.logits.new_tensor(0.0)

    bbox_smoothl1_contrib = float(bbox_smoothl1_w) * smoothl1
    bbox_ciou_contrib = float(bbox_ciou_w) * ciou
    loss = bbox_smoothl1_contrib + bbox_ciou_contrib

    metrics = {
        "loss/geo": float(loss.detach().cpu().item()),
        "loss/bbox_smoothl1": float(smoothl1.detach().cpu().item()),
        "loss/bbox_ciou": float(ciou.detach().cpu().item()),
        "bbox/groups_total": float(n_groups_all),
        "bbox/slots_total": float(int(dec_prefix.n_slots + dec_fn.n_slots)),
    }

    logits_full = torch.cat([dec_prefix.logits_full, dec_fn.logits_full], dim=0)
    coord_logits = torch.cat([dec_prefix.coord_logits, dec_fn.coord_logits], dim=0)
    target_bins = torch.cat([dec_prefix.target_bins, dec_fn.target_bins], dim=0)
    slot_weights = torch.cat([dec_prefix.slot_weights, dec_fn.slot_weights], dim=0)

    state = {
        "bbox_geo": loss,
        "smoothl1": smoothl1,
        "ciou": ciou,
        "bbox_smoothl1_contrib": bbox_smoothl1_contrib,
        "bbox_ciou_contrib": bbox_ciou_contrib,
        "coord_logits": coord_logits,
        "coord_logits_full": logits_full,
        "coord_target_bins": target_bins,
        "coord_slot_weights": slot_weights,
        "coord_slots_total": int(dec_prefix.n_slots + dec_fn.n_slots),
    }

    return ModuleResult(loss=loss, metrics=metrics, state=state)
