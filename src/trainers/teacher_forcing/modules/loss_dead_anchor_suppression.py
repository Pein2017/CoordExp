from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..contracts import ModuleResult, PipelineModuleSpec, TeacherForcingContext
from ..token_types import iter_segment_views


def run_loss_dead_anchor_suppression_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    channel = str(context.channel or "").strip().upper()
    if channel != "B":
        raise ValueError("loss_dead_anchor_suppression only supports Channel-B teacher forcing")

    registry_context = str(context.registry_context or "").strip().lower()
    if registry_context != "rollout":
        raise ValueError(
            "loss_dead_anchor_suppression requires registry_context='rollout' for Channel-B teacher forcing"
        )

    input_ids = context.input_ids
    logits_ce = context.logits_ce
    meta = context.meta

    if input_ids.ndim != 2 or logits_ce.ndim != 3:
        raise ValueError(
            "loss_dead_anchor_suppression expects input_ids[B,T] and logits[B,T,V]"
        )
    if logits_ce.shape[:2] != input_ids.shape:
        raise ValueError(
            "loss_dead_anchor_suppression requires logits_ce and input_ids to share [B,T]"
        )

    for idx, seg in enumerate(meta):
        if not isinstance(seg, Mapping):
            raise ValueError(
                f"loss_dead_anchor_suppression meta[{idx}] must be a mapping"
            )
        if "dead_anchor_suppression_targets" not in seg:
            raise ValueError(
                "loss_dead_anchor_suppression requires dead_anchor_suppression_targets in every Channel-B rollout segment"
            )

    terms: list[torch.Tensor] = []
    boundaries: set[int] = set()

    vocab = int(logits_ce.shape[-1])
    for b, seg_start, seg_end, seg in iter_segment_views(input_ids=input_ids, meta=meta):
        prompt_len = int(seg.get("prompt_len", 0) or 0)
        train_len = int(seg.get("train_len", 0) or 0)

        assistant_start = max(
            int(seg_start + 1),
            min(int(seg_end), int(seg_start + prompt_len)),
        )
        assistant_end = max(
            assistant_start,
            min(int(seg_end), int(seg_start + prompt_len + train_len)),
        )

        raw_targets = seg.get("dead_anchor_suppression_targets") or []
        if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
            raise ValueError(
                "dead_anchor_suppression_targets must be a sequence of mappings"
            )

        for item in raw_targets:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "dead_anchor_suppression target entries must be mappings"
                )

            try:
                rel_pos = int(item.get("rel_pos"))
                bad_token_id = int(item.get("token_id"))
                boundary = int(item.get("boundary"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "dead_anchor_suppression target entries must provide int boundary/rel_pos/token_id"
                ) from exc

            token_pos = int(assistant_start + rel_pos)
            if token_pos <= int(seg_start) or token_pos >= int(assistant_end):
                raise ValueError(
                    "dead_anchor_suppression target rel_pos is outside the retained assistant span: "
                    f"rel_pos={rel_pos} token_pos={token_pos} assistant=[{assistant_start},{assistant_end})"
                )
            if bad_token_id < 0 or bad_token_id >= vocab:
                raise ValueError(
                    f"dead_anchor_suppression target token_id is outside vocabulary: {bad_token_id}"
                )

            pred_row = int(token_pos - 1)
            log_probs = F.log_softmax(logits_ce[b, pred_row, :], dim=-1)
            bad_log_prob = log_probs[int(bad_token_id)]
            bad_prob = bad_log_prob.exp()
            term = -torch.log((1.0 - bad_prob).clamp(min=1e-6))

            terms.append(term)
            boundaries.add(int(boundary))

    if terms:
        loss = torch.stack(terms).mean()
    else:
        loss = logits_ce.new_tensor(0.0)

    metrics = {
        "train/optimization/loss_dead_anchor_suppression": float(loss.detach().cpu().item()),
        "train/triage/dead_anchor_suppression_target_count": float(len(terms)),
        "train/triage/dead_anchor_suppression_boundary_count": float(len(boundaries)),
    }
    state = {
        "loss_dead_anchor_suppression": loss,
        "loss_dead_anchor_suppression_contrib": loss,
        "dead_anchor_suppression_target_count": int(len(terms)),
        "dead_anchor_suppression_boundary_count": int(len(boundaries)),
    }
    return ModuleResult(loss=loss, metrics=metrics, state=state)
