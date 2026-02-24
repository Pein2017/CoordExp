from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..token_types import build_token_type_masks, iter_segment_views


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def run_token_ce_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    input_ids = context.input_ids
    logits_ce = context.logits_ce
    meta = context.meta
    coord_id_set = context.coord_id_set

    cfg = spec.config if isinstance(spec.config, Mapping) else {}

    desc_ce_weight = max(
        0.0,
        _coerce_float(
            cfg.get("desc_ce_weight", context.extra.get("desc_ce_weight", 1.0)),
            1.0,
        ),
    )
    fn_desc_ce_weight = max(
        0.0,
        _coerce_float(
            cfg.get(
                "fn_desc_ce_weight",
                cfg.get("rollout_fn_desc_weight", desc_ce_weight),
            ),
            desc_ce_weight,
        ),
    )
    matched_prefix_struct_ce_weight = max(
        0.0,
        _coerce_float(
            cfg.get(
                "matched_prefix_struct_ce_weight",
                cfg.get("rollout_matched_prefix_struct_weight", 1.0),
            ),
            1.0,
        ),
    )
    drop_invalid_struct_ce_multiplier_cfg = max(
        1.0,
        _coerce_float(
            cfg.get("rollout_drop_invalid_struct_ce_multiplier", 1.0),
            1.0,
        ),
    )

    labels_masked = torch.full_like(input_ids, -100)
    weights = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)
    struct_weights = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)
    desc_weights = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)

    channel = str(context.channel or "").strip().upper()

    for b, seg_start, seg_end, seg in iter_segment_views(input_ids=input_ids, meta=meta):
        prompt_len = int(seg.get("prompt_len", 0) or 0)
        prefix_len = int(seg.get("prefix_len", 0) or 0)
        train_len = int(seg.get("train_len", 0) or 0)

        prefix_struct_pos = [int(p) for p in (seg.get("prefix_struct_pos") or [])]
        tail_ignore_pos = [int(p) for p in (seg.get("tail_ignore_pos") or [])]
        tail_desc_pos = [int(p) for p in (seg.get("tail_desc_pos") or [])]
        tail_closure_pos = [int(p) for p in (seg.get("tail_closure_pos") or [])]

        drop_invalid_total = int(seg.get("drop_invalid_total", 0) or 0)
        drop_invalid_struct_ce_multiplier = max(
            1.0,
            _coerce_float(seg.get("drop_invalid_struct_ce_multiplier", 1.0), 1.0),
        )

        seg_start_i = int(seg_start)
        seg_end_i = int(seg_end)

        prefix_start = max(seg_start_i + 1, min(seg_end_i, seg_start_i + prompt_len))
        prefix_end = max(
            prefix_start,
            min(seg_end_i, seg_start_i + prompt_len + prefix_len),
        )

        tail_start = max(prefix_end, min(seg_end_i, seg_start_i + prompt_len + prefix_len))
        tail_end = max(
            tail_start,
            min(seg_end_i, seg_start_i + prompt_len + train_len),
        )

        if channel == "B" and prefix_struct_pos:
            for rel in prefix_struct_pos:
                p = int(prefix_start + rel)
                if p < int(prefix_start) or p >= int(prefix_end):
                    continue
                if int(input_ids[b, p].item()) in coord_id_set:
                    continue
                labels_masked[b, p] = input_ids[b, p]
                weights[b, p] = float(matched_prefix_struct_ce_weight)
                struct_weights[b, p] = float(matched_prefix_struct_ce_weight)

        tail_ignore = {int(x) for x in tail_ignore_pos if int(x) >= 0}
        tail_desc = {int(x) for x in tail_desc_pos if int(x) >= 0}
        tail_closure = {int(x) for x in tail_closure_pos if int(x) >= 0}

        for p in range(int(tail_start), int(tail_end)):
            tok_id = int(input_ids[b, p].item())
            if tok_id in coord_id_set:
                continue
            rel = int(p - tail_start)
            if rel in tail_ignore and rel not in tail_closure:
                continue

            labels_masked[b, p] = input_ids[b, p]

            if rel in tail_desc:
                w_desc = float(fn_desc_ce_weight if channel == "B" else desc_ce_weight)
                weights[b, p] = float(w_desc)
                desc_weights[b, p] = float(w_desc)
            else:
                weights[b, p] = 1.0
                struct_weights[b, p] = 1.0

        for rel in tail_closure:
            p = int(tail_start + rel)
            if p < int(tail_start) or p >= int(tail_end):
                continue
            if int(input_ids[b, p].item()) in coord_id_set:
                continue
            labels_masked[b, p] = input_ids[b, p]
            weights[b, p] = max(float(weights[b, p].item()), 1.0)
            struct_weights[b, p] = max(float(struct_weights[b, p].item()), 1.0)

        if (
            channel == "B"
            and int(drop_invalid_total) > 0
            and float(drop_invalid_struct_ce_multiplier) != 1.0
        ):
            mult = float(drop_invalid_struct_ce_multiplier) * float(
                drop_invalid_struct_ce_multiplier_cfg
            )
            for p in range(int(tail_start), int(tail_end)):
                if int(labels_masked[b, p].item()) == -100:
                    continue
                rel = int(p - tail_start)
                if rel in tail_ignore or rel in tail_desc:
                    continue
                struct_weights[b, p] = struct_weights[b, p] * mult
                weights[b, p] = weights[b, p] * mult

    bsz, _, vocab = logits_ce.shape
    logits_next = logits_ce[:, :-1, :]
    labels_next = labels_masked[:, 1:]
    weights_next = weights[:, 1:]
    struct_weights_next = struct_weights[:, 1:]
    desc_weights_next = desc_weights[:, 1:]

    per_tok = F.cross_entropy(
        logits_next.reshape(-1, vocab),
        labels_next.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(bsz, -1)

    def _weighted_mean(w: torch.Tensor) -> torch.Tensor:
        denom = w.sum()
        if float(denom.detach().cpu().item()) <= 0.0:
            return per_tok.new_tensor(0.0)
        return (per_tok * w).sum() / denom.clamp(min=1e-6)

    token_ce = _weighted_mean(weights_next)
    struct_ce = _weighted_mean(struct_weights_next)
    desc_ce = _weighted_mean(desc_weights_next)

    token_masks = build_token_type_masks(
        input_ids=input_ids,
        meta=meta,
        coord_id_set=coord_id_set,
        channel=channel,
    )

    metrics = {
        "loss/token_ce": float(token_ce.detach().cpu().item()),
        "loss/struct_ce": float(struct_ce.detach().cpu().item()),
        "loss/desc_ce": float(desc_ce.detach().cpu().item()),
    }

    state = {
        "token_ce": token_ce,
        "struct_ce": struct_ce,
        "desc_ce": desc_ce,
        "labels_masked": labels_masked,
        "weights_masked": weights,
        "token_type_masks": token_masks,
    }

    return ModuleResult(loss=token_ce, metrics=metrics, state=state)
