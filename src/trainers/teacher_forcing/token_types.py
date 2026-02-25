from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

import torch


def iter_segment_views(
    *,
    input_ids: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
) -> Iterator[Tuple[int, int, int, Mapping[str, Any]]]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    bsz, seq_len = input_ids.shape

    if len(meta) == bsz:
        for b in range(int(bsz)):
            yield int(b), 0, int(seq_len), meta[b]
        return

    if int(bsz) != 1:
        raise ValueError(
            f"packed-mode meta requires bsz==1 when len(meta)!=bsz (len(meta)={len(meta)} bsz={int(bsz)})"
        )

    offset = 0
    for seg in meta:
        enc_len = int(seg.get("encoded_len") or 0)
        if enc_len <= 0:
            raise ValueError("packed segment missing encoded_len")
        seg_start = int(offset)
        seg_end = int(offset + enc_len)
        if seg_end > int(seq_len):
            raise ValueError("packed segments exceed input_ids length")
        yield 0, seg_start, seg_end, seg
        offset = seg_end


def _to_rel_set(values: Iterable[Any] | None) -> set[int]:
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


def build_token_type_masks(
    *,
    input_ids: torch.Tensor,
    meta: Sequence[Mapping[str, Any]],
    coord_id_set: set[int],
    channel: str,
) -> dict[str, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")

    shape = input_ids.shape
    mask_struct = torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
    mask_desc = torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
    mask_coord = torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
    mask_eos = torch.zeros(shape, dtype=torch.bool, device=input_ids.device)

    ch = str(channel or "").strip().upper()

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

        tail_desc_rel = _to_rel_set(seg.get("tail_desc_pos"))
        tail_ignore_rel = _to_rel_set(seg.get("tail_ignore_pos"))
        tail_closure_rel = _to_rel_set(seg.get("tail_closure_pos"))
        prefix_struct_rel = _to_rel_set(seg.get("prefix_struct_pos"))

        # Coord mask across the whole supervised assistant span.
        for p in range(int(assistant_start), int(assistant_end)):
            if int(input_ids[b, p].item()) in coord_id_set:
                mask_coord[b, p] = True

        # Prefix structure mask.
        if prefix_struct_rel:
            for rel in prefix_struct_rel:
                p = int(prefix_start + rel)
                if p < int(prefix_start) or p >= int(prefix_end):
                    continue
                if not bool(mask_coord[b, p].item()):
                    mask_struct[b, p] = True
        elif ch != "B":
            for p in range(int(prefix_start), int(prefix_end)):
                if not bool(mask_coord[b, p].item()):
                    mask_struct[b, p] = True

        # Tail masks.
        tail_cap = max(0, int(tail_end - tail_start))
        for p in range(int(tail_start), int(tail_end)):
            rel = int(p - tail_start)
            if rel < 0 or rel >= tail_cap:
                continue

            is_coord = bool(mask_coord[b, p].item())
            if rel in tail_desc_rel and not is_coord:
                mask_desc[b, p] = True
                continue

            if rel in tail_ignore_rel and rel not in tail_closure_rel:
                continue

            if not is_coord:
                mask_struct[b, p] = True

        # EOS / closure markers are always supervised.
        for rel in tail_closure_rel:
            p = int(tail_start + rel)
            if p < int(tail_start) or p >= int(tail_end):
                continue
            if bool(mask_coord[b, p].item()):
                continue
            mask_eos[b, p] = True
            mask_struct[b, p] = True

    return {
        "struct": mask_struct,
        "desc": mask_desc,
        "coord": mask_coord,
        "eos": mask_eos,
    }
