from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class RolloutAlignedSampleTargets:
    prefix_coord_pos: List[int]
    prefix_coord_target_bins: List[int]
    prefix_struct_pos: List[int]
    tail_ignore_pos: List[int]
    tail_desc_pos: List[int]
    tail_closure_pos: List[int]
    bbox_groups_prefix: List[dict[str, Any]]
    bbox_groups_fn: List[dict[str, Any]]
    matched_gt_indices: List[int]
    fn_gt_indices: List[int]
    fn_objs: List[Any]
    append_text: str
    append_ids: List[int]
    y_train_ids: List[int]
    semantic_stop_meta: Mapping[str, Any]
    excluded_from_supervision: int


def build_labels_and_coord_targets_for_sample(
    *,
    input_ids_1d: torch.Tensor,  # [T]
    prompt_len: int,
    prefix_len: int,
    train_len: int,
    coord_id_set: set[int],
    coord_id_to_bin: Mapping[int, int],
    prefix_coord_pos: Sequence[int],
    prefix_coord_target_bins: Sequence[int],
    tail_ignore_pos: Optional[Sequence[int]] = None,
    prefix_struct_pos: Optional[Sequence[int]] = None,
    tail_desc_pos: Optional[Sequence[int]] = None,
    tail_closure_pos: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, List[int], List[int], List[bool]]:
    """Create CE labels and coord supervision targets for a single sample."""
    seq_len = int(input_ids_1d.shape[0])
    labels = torch.full((seq_len,), -100, dtype=torch.long, device=input_ids_1d.device)

    coord_pos: List[int] = []
    coord_bins: List[int] = []
    coord_is_prefix: List[bool] = []

    assistant_start = int(prompt_len)
    assistant_end = int(prompt_len) + int(train_len)
    if assistant_start < 0:
        raise ValueError(f"invalid prompt_len={prompt_len}")
    if assistant_end < assistant_start:
        raise ValueError(f"invalid train_len={train_len} for prompt_len={prompt_len}")
    assistant_end = min(assistant_end, seq_len)
    if assistant_end <= assistant_start:
        raise ValueError(
            f"invalid assistant span [{assistant_start},{assistant_end}) for seq_len={seq_len}"
        )

    prefix_start = max(1, min(int(prompt_len), seq_len))
    prefix_end = max(prefix_start, min(int(prompt_len + prefix_len), assistant_end))

    tail_start = max(1, min(int(prompt_len + prefix_len), seq_len))
    tail_end = max(tail_start, min(int(prompt_len + train_len), assistant_end))

    ignore_set = set(int(i) for i in (tail_ignore_pos or []) if int(i) >= 0)
    closure_set = set(int(i) for i in (tail_closure_pos or []) if int(i) >= 0)
    _ = set(int(i) for i in (tail_desc_pos or []) if int(i) >= 0)

    prefix_struct_set = set(int(i) for i in (prefix_struct_pos or []) if int(i) >= 0)
    for local_idx in sorted(prefix_struct_set):
        if local_idx < 0 or local_idx >= int(prefix_len):
            continue
        p = int(prompt_len + local_idx)
        if p < int(prefix_start) or p >= int(prefix_end):
            continue
        if p < int(assistant_start) or p >= int(assistant_end):
            raise ValueError(
                f"prefix struct CE index out of assistant span: p={p} span=[{assistant_start},{assistant_end})"
            )
        tok_id = int(input_ids_1d[p].item())
        if tok_id in coord_id_set:
            continue
        labels[p] = input_ids_1d[p]

    for p in range(tail_start, tail_end):
        if p < assistant_start or p >= assistant_end:
            raise ValueError(
                f"tail supervision index out of assistant span: p={p} span=[{assistant_start},{assistant_end})"
            )
        tok_id = int(input_ids_1d[p].item())
        if tok_id in coord_id_set:
            bin_idx = coord_id_to_bin.get(tok_id)
            if bin_idx is not None:
                coord_pos.append(int(p))
                coord_bins.append(int(bin_idx))
                coord_is_prefix.append(False)
            continue
        rel = int(p - tail_start)
        if rel in ignore_set and rel not in closure_set:
            continue
        labels[p] = input_ids_1d[p]

    for rel in sorted(closure_set):
        p = int(tail_start + rel)
        if p < int(tail_start) or p >= int(tail_end):
            continue
        tok_id = int(input_ids_1d[p].item())
        if tok_id in coord_id_set:
            continue
        labels[p] = input_ids_1d[p]

    if len(prefix_coord_pos) != len(prefix_coord_target_bins):
        raise ValueError(
            "prefix_coord_pos and prefix_coord_target_bins must have identical length"
        )
    for local_idx, tbin in zip(prefix_coord_pos, prefix_coord_target_bins):
        li = int(local_idx)
        if li < 0 or li >= int(prefix_len):
            continue
        p = int(prompt_len + li)
        if p <= 0 or p >= int(seq_len):
            continue
        if p < int(assistant_start) or p >= int(assistant_end):
            raise ValueError(
                f"prefix supervision index out of assistant span: p={p} span=[{assistant_start},{assistant_end})"
            )
        coord_pos.append(int(p))
        coord_bins.append(int(tbin))
        coord_is_prefix.append(True)

    return labels, coord_pos, coord_bins, coord_is_prefix


def build_labels_and_coord_targets_for_batch(
    *,
    input_ids: torch.Tensor,  # [B, T]
    meta: List[Mapping[str, Any]],
    coord_id_set: set[int],
    coord_id_to_bin: Mapping[int, int],
) -> Tuple[torch.Tensor, List[int], List[int], List[int], List[bool]]:
    """Build masked CE labels + coord supervision targets for a batch."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    bsz, seq_len = input_ids.shape

    labels_masked = torch.full_like(input_ids, -100)

    supervised_batch: List[int] = []
    supervised_pos: List[int] = []
    supervised_bin: List[int] = []
    supervised_is_prefix: List[bool] = []

    if len(meta) == bsz:
        for b in range(bsz):
            m = meta[b]
            prompt_len = int(m["prompt_len"])
            prefix_len = int(m["prefix_len"])
            train_len = int(m["train_len"])
            prompt_ids = m.get("prompt_ids")

            if prompt_len <= 0 or prompt_len >= seq_len:
                raise ValueError(
                    f"invalid prompt_len={prompt_len} for seq_len={seq_len}"
                )
            if isinstance(prompt_ids, list):
                teacher_prefix = input_ids[b, :prompt_len].detach().cpu().tolist()
                if teacher_prefix != prompt_ids:
                    raise ValueError(
                        "prompt tokenization mismatch between generation and teacher-forced encoding"
                    )

            labels_1d, cpos, cbins, cis_prefix = (
                build_labels_and_coord_targets_for_sample(
                    input_ids_1d=input_ids[b],
                    prompt_len=prompt_len,
                    prefix_len=prefix_len,
                    train_len=train_len,
                    coord_id_set=coord_id_set,
                    coord_id_to_bin=coord_id_to_bin,
                    prefix_coord_pos=m.get("prefix_coord_pos") or [],
                    prefix_coord_target_bins=m.get("prefix_coord_target_bins") or [],
                    tail_ignore_pos=m.get("tail_ignore_pos") or [],
                    prefix_struct_pos=m.get("prefix_struct_pos") or [],
                    tail_desc_pos=m.get("tail_desc_pos") or [],
                    tail_closure_pos=m.get("tail_closure_pos") or [],
                )
            )
            labels_masked[b] = labels_1d
            for p, tbin, is_pref in zip(cpos, cbins, cis_prefix):
                supervised_batch.append(int(b))
                supervised_pos.append(int(p))
                supervised_bin.append(int(tbin))
                supervised_is_prefix.append(bool(is_pref))

        return (
            labels_masked,
            supervised_batch,
            supervised_pos,
            supervised_bin,
            supervised_is_prefix,
        )

    if bsz != 1:
        raise ValueError(
            "packed-mode meta requires bsz==1; got len(meta)=%s bsz=%s"
            % (len(meta), bsz)
        )
    if not meta:
        raise ValueError("packed-mode meta must be a non-empty list")

    offset = 0
    for seg in meta:
        if not isinstance(seg, Mapping):
            raise ValueError("packed-mode meta must be a list of dict-like segments")
        encoded_len = int(seg.get("encoded_len") or 0)
        if encoded_len <= 0:
            raise ValueError("packed-mode segment missing/invalid encoded_len")
        if offset + encoded_len > seq_len:
            raise ValueError("packed-mode segments exceed packed seq_len")

        seg_input_ids = input_ids[0, offset : offset + encoded_len]
        seg_prompt_len = int(seg["prompt_len"])
        seg_prefix_len = int(seg["prefix_len"])
        seg_train_len = int(seg["train_len"])
        prompt_ids = seg.get("prompt_ids")

        if seg_prompt_len <= 0 or seg_prompt_len >= encoded_len:
            raise ValueError(
                f"invalid prompt_len={seg_prompt_len} for encoded_len={encoded_len}"
            )
        if isinstance(prompt_ids, list):
            teacher_prefix = seg_input_ids[:seg_prompt_len].detach().cpu().tolist()
            if teacher_prefix != prompt_ids:
                raise ValueError(
                    "prompt tokenization mismatch between generation and teacher-forced encoding"
                )

        labels_1d, cpos, cbins, cis_prefix = build_labels_and_coord_targets_for_sample(
            input_ids_1d=seg_input_ids,
            prompt_len=seg_prompt_len,
            prefix_len=seg_prefix_len,
            train_len=seg_train_len,
            coord_id_set=coord_id_set,
            coord_id_to_bin=coord_id_to_bin,
            prefix_coord_pos=seg.get("prefix_coord_pos") or [],
            prefix_coord_target_bins=seg.get("prefix_coord_target_bins") or [],
            tail_ignore_pos=seg.get("tail_ignore_pos") or [],
            prefix_struct_pos=seg.get("prefix_struct_pos") or [],
            tail_desc_pos=seg.get("tail_desc_pos") or [],
            tail_closure_pos=seg.get("tail_closure_pos") or [],
        )
        labels_masked[0, offset : offset + encoded_len] = labels_1d
        for p, tbin, is_pref in zip(cpos, cbins, cis_prefix):
            supervised_batch.append(0)
            supervised_pos.append(int(offset + p))
            supervised_bin.append(int(tbin))
            supervised_is_prefix.append(bool(is_pref))

        offset += encoded_len

    return (
        labels_masked,
        supervised_batch,
        supervised_pos,
        supervised_bin,
        supervised_is_prefix,
    )


def build_rollout_aligned_sample_targets(
    *,
    tokenizer: Any,
    parse: Any,
    prompt_ids: Sequence[int],
    pred_meta: Sequence[Any],
    preds: Sequence[Any],
    gts: Sequence[Any],
    match: Any,
    coord_id_set: set[int],
    object_field_order: Any,
    ot_epsilon: float,
    ot_iters: int,
    ot_cost: str,
    build_prefix_targets_fn: Callable[..., Any],
    matched_prefix_structure_positions_fn: Callable[..., Sequence[int]],
    serialize_append_fragment_fn: Callable[..., str],
    tail_desc_positions_fn: Callable[..., Sequence[int]],
    bbox_groups_from_token_ids_fn: Callable[..., Sequence[Sequence[int]]],
    tail_closure_positions_fn: Callable[..., Sequence[int]],
    semantic_stop_branch_metadata_fn: Callable[..., Mapping[str, Any]],
) -> RolloutAlignedSampleTargets:
    """Build the rollout-aligned training-target payload for one sample.

    This is the live extraction seam from `_prepare_batch_inputs`: given a
    parsed rollout plus the match result, produce the exact supervision payload
    that the teacher-forced encode and loss path consume.
    """

    prefix_coord_pos: List[int] = []
    prefix_coord_target_bins: List[int] = []
    bbox_groups_prefix: List[dict[str, Any]] = []
    excluded_from_supervision = 0

    matched_gt_for_supervision: set[int] = set()
    matched_pred_for_supervision: set[int] = set()
    for pred_i, gt_i in match.matched_pairs:
        if pred_i < 0 or pred_i >= len(preds) or pred_i >= len(pred_meta):
            continue
        if gt_i < 0 or gt_i >= len(gts):
            continue
        pobj = pred_meta[pred_i]
        pred_obj = preds[pred_i]
        gt_obj = gts[gt_i]
        try:
            targets = build_prefix_targets_fn(
                pred_obj=pred_obj,
                gt_obj=gt_obj,
                pred_coord_indices=pobj.coord_token_indices,
                ot_epsilon=ot_epsilon,
                ot_iters=ot_iters,
                ot_cost=ot_cost,
            )
        except (TypeError, ValueError):
            targets = None
        if targets is None or len(targets) != len(pobj.coord_token_indices):
            excluded_from_supervision += 1
            continue
        matched_gt_for_supervision.add(int(gt_i))
        matched_pred_for_supervision.add(int(pred_i))
        for local_idx, tbin in zip(pobj.coord_token_indices, targets):
            if local_idx < 0 or local_idx >= len(parse.prefix_token_ids):
                continue
            prefix_coord_pos.append(int(local_idx))
            prefix_coord_target_bins.append(int(tbin))

        if (
            str(pred_obj.geom_type) == "bbox_2d"
            and str(gt_obj.geom_type) == "bbox_2d"
            and len(pobj.coord_token_indices) == 4
            and len(gt_obj.points_norm1000) == 4
        ):
            bbox_groups_prefix.append(
                {
                    "pos": [
                        int(len(prompt_ids) + int(local_idx))
                        for local_idx in pobj.coord_token_indices
                    ],
                    "gt_bins": [int(x) for x in gt_obj.points_norm1000],
                }
            )

    matched_pred_objects = [
        pred_meta[int(i)]
        for i in sorted(matched_pred_for_supervision)
        if 0 <= int(i) < len(pred_meta)
    ]
    prefix_struct_pos = [
        int(p)
        for p in matched_prefix_structure_positions_fn(
            tokenizer=tokenizer,
            prefix_token_ids=parse.prefix_token_ids,
            prefix_text=parse.prefix_text,
            matched_pred_objects=matched_pred_objects,
        )
    ]

    fn_gt_indices = [
        int(i) for i in range(len(gts)) if int(i) not in matched_gt_for_supervision
    ]
    fn_objs = [gts[i] for i in fn_gt_indices]

    append_text = str(
        serialize_append_fragment_fn(
            fn_objects=fn_objs,
            prefix_text=parse.prefix_text,
            object_field_order=object_field_order,
        )
    )
    append_ids = [int(t) for t in tokenizer.encode(append_text, add_special_tokens=False)]
    tail_desc_pos = [
        int(p) for p in tail_desc_positions_fn(tokenizer=tokenizer, token_ids=append_ids)
    ]
    tail_ignore_pos: List[int] = []

    bbox_groups_fn: List[dict[str, Any]] = []
    bbox_fn_objs = [
        obj
        for obj in fn_objs
        if str(getattr(obj, "geom_type", "")) == "bbox_2d"
        and len(getattr(obj, "points_norm1000", []) or []) == 4
    ]
    if bbox_fn_objs and len(bbox_fn_objs) == len(fn_objs):
        rel_groups = bbox_groups_from_token_ids_fn(
            token_ids=append_ids,
            coord_id_set=coord_id_set,
            gt_objs=bbox_fn_objs,
        )
        for obj, rel_pos in zip(bbox_fn_objs, rel_groups):
            bbox_groups_fn.append(
                {
                    "pos": [
                        int(len(prompt_ids) + int(len(parse.prefix_token_ids)) + int(p))
                        for p in rel_pos
                    ],
                    "gt_bins": [int(x) for x in obj.points_norm1000],
                }
            )

    y_train_ids = [int(t) for t in parse.prefix_token_ids] + list(append_ids)
    tail_closure_pos = [
        int(p)
        for p in tail_closure_positions_fn(
            tokenizer=tokenizer,
            assistant_span_ids=y_train_ids,
            prefix_len=int(len(parse.prefix_token_ids)),
        )
    ]
    semantic_stop_meta = dict(
        semantic_stop_branch_metadata_fn(
            tokenizer=tokenizer,
            assistant_span_ids=y_train_ids,
            prefix_len=int(len(parse.prefix_token_ids)),
        )
    )

    return RolloutAlignedSampleTargets(
        prefix_coord_pos=prefix_coord_pos,
        prefix_coord_target_bins=prefix_coord_target_bins,
        prefix_struct_pos=prefix_struct_pos,
        tail_ignore_pos=tail_ignore_pos,
        tail_desc_pos=tail_desc_pos,
        tail_closure_pos=tail_closure_pos,
        bbox_groups_prefix=bbox_groups_prefix,
        bbox_groups_fn=bbox_groups_fn,
        matched_gt_indices=sorted(matched_gt_for_supervision),
        fn_gt_indices=fn_gt_indices,
        fn_objs=fn_objs,
        append_text=append_text,
        append_ids=append_ids,
        y_train_ids=y_train_ids,
        semantic_stop_meta=semantic_stop_meta,
        excluded_from_supervision=excluded_from_supervision,
    )
