from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..geometry import bbox_smoothl1_ciou_loss, expectation_decode_coords
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
) -> tuple[list[int], list[int], list[int]]:
    b_list: List[int] = []
    pos_list: List[int] = []
    bins_list: List[int] = []

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

    return b_list, pos_list, bins_list


def _decode_groups(
    *,
    key: str,
    context: TeacherForcingContext,
    coord_ids_t: torch.Tensor,
) -> _DecodedGroups:
    input_ids = context.input_ids
    logits = context.logits
    meta = context.meta

    b_list, pos_list, bins_list = _flatten_groups(key=key, input_ids=input_ids, meta=meta)
    if not pos_list:
        z = logits.new_tensor(0.0)
        empty = logits.new_zeros((0, logits.shape[-1]))
        empty_coord = logits.new_zeros((0, coord_ids_t.numel()))
        empty_bins = logits.new_zeros((0,), dtype=torch.long)
        return _DecodedGroups(
            smoothl1=z,
            ciou=z,
            n_groups=0,
            n_slots=0,
            logits_full=empty,
            coord_logits=empty_coord,
            target_bins=empty_bins,
        )

    if len(pos_list) % 4 != 0:
        raise ValueError(f"{key} coord slots must be a multiple of 4")

    b_t = torch.tensor(b_list, device=logits.device, dtype=torch.long)
    pos_t = torch.tensor(pos_list, device=logits.device, dtype=torch.long)
    bin_t = torch.tensor(bins_list, device=logits.device, dtype=torch.long).clamp(min=0, max=999)

    b_g = b_t.reshape(-1, 4)
    pos_g = pos_t.reshape(-1, 4)
    bin_g = bin_t.reshape(-1, 4)

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
        return _DecodedGroups(
            smoothl1=z,
            ciou=z,
            n_groups=0,
            n_slots=0,
            logits_full=empty,
            coord_logits=empty_coord,
            target_bins=empty_bins,
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

    bbox = bbox_smoothl1_ciou_loss(
        pred_xyxy=pred.reshape(-1, 4),
        gt_xyxy=gt.reshape(-1, 4),
    )

    return _DecodedGroups(
        smoothl1=bbox.smoothl1,
        ciou=bbox.ciou,
        n_groups=int(n_groups),
        n_slots=int(pos_t.numel()),
        logits_full=logits_prev,
        coord_logits=coord_logits,
        target_bins=bin_t,
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

    state = {
        "bbox_geo": loss,
        "smoothl1": smoothl1,
        "ciou": ciou,
        "bbox_smoothl1_contrib": bbox_smoothl1_contrib,
        "bbox_ciou_contrib": bbox_ciou_contrib,
        "coord_logits": coord_logits,
        "coord_logits_full": logits_full,
        "coord_target_bins": target_bins,
        "coord_slots_total": int(dec_prefix.n_slots + dec_fn.n_slots),
    }

    return ModuleResult(loss=loss, metrics=metrics, state=state)
