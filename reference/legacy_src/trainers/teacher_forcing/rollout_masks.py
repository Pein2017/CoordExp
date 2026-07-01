from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .token_types import iter_segment_views


def _to_rel_set(values: Sequence[Any] | None) -> set[int]:
    out: set[int] = set()
    if values is None:
        return out
    for v in values:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv >= 0:
            out.add(int(iv))
    return out


def build_rollout_subset_masks(
    *,
    input_ids: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
    coord_id_set: set[int],
) -> dict[str, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")

    shape = input_ids.shape
    matched = torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
    fp = torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
    fn = torch.zeros(shape, dtype=torch.bool, device=input_ids.device)

    for b, seg_start, seg_end, seg in iter_segment_views(input_ids=input_ids, meta=meta):
        prompt_len = int(seg.get("prompt_len", 0) or 0)
        prefix_len = int(seg.get("prefix_len", 0) or 0)
        train_len = int(seg.get("train_len", 0) or 0)

        assistant_start = max(int(seg_start + 1), min(int(seg_end), int(seg_start + prompt_len)))
        assistant_end = max(
            assistant_start,
            min(int(seg_end), int(seg_start + prompt_len + train_len)),
        )

        prefix_start = assistant_start
        prefix_end = max(prefix_start, min(assistant_end, int(prefix_start + prefix_len)))
        tail_start = prefix_end
        tail_end = assistant_end

        prefix_struct_rel = _to_rel_set(seg.get("prefix_struct_pos"))
        prefix_coord_rel = _to_rel_set(seg.get("prefix_coord_pos"))

        matched_prefix_abs: set[int] = set()
        for rel in prefix_struct_rel:
            p = int(prefix_start + rel)
            if int(prefix_start) <= p < int(prefix_end):
                matched_prefix_abs.add(int(p))
        for rel in prefix_coord_rel:
            p = int(prefix_start + rel)
            if int(prefix_start) <= p < int(prefix_end):
                matched_prefix_abs.add(int(p))

        # Matched subset in prefix.
        for p in sorted(matched_prefix_abs):
            matched[b, p] = True

        # FP subset: prefix positions not matched (non-coord only).
        for p in range(int(prefix_start), int(prefix_end)):
            if int(p) in matched_prefix_abs:
                continue
            if int(input_ids[b, p].item()) in coord_id_set:
                continue
            fp[b, p] = True

        # FN subset: full tail span.
        for p in range(int(tail_start), int(tail_end)):
            fn[b, p] = True

    return {
        "matched": matched,
        "fp": fp,
        "fn": fn,
    }
