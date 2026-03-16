from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


def mask_stage1_coord_targets(
    labels: torch.Tensor,
    coord_token_ids: Sequence[int],
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")

    if not coord_token_ids:
        return labels.clone()

    out = labels.clone()
    mask = torch.zeros_like(out, dtype=torch.bool)
    for tok_id in coord_token_ids:
        mask |= out.eq(int(tok_id))
    out[mask] = -100
    return out


@dataclass(frozen=True)
class Stage1BBoxQuartets:
    coord_logits: torch.Tensor
    target_bins: torch.Tensor
    target_boxes_xyxy: torch.Tensor
    coord_slots: int
    bbox_groups: int


def _decode_pieces(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [
        tokenizer.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in token_ids
    ]


def extract_stage1_bbox_quartets(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    coord_token_ids: Sequence[int],
    coord_id_map: torch.Tensor,
    tokenizer: Any | None = None,
) -> Stage1BBoxQuartets | None:
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch.Tensors")
    if not isinstance(coord_id_map, torch.Tensor):
        raise TypeError("coord_id_map must be a torch.Tensor")
    if not coord_token_ids:
        return None

    seq_len = min(int(logits.shape[1]), max(int(labels.shape[1]) - 1, 0))
    if seq_len <= 0:
        return None

    logits_next = logits[:, :seq_len, :]
    labels_next = labels[:, 1 : seq_len + 1]
    labels_safe = labels_next
    if int(labels_safe.numel()) > 0 and int(labels_safe.min().detach().item()) < 0:
        labels_safe = labels_safe.clamp(min=0)

    target_bins_all = coord_id_map[labels_safe].to(dtype=torch.long)
    coord_mask = (target_bins_all >= 0) & (labels_next != -100)
    if not bool(coord_mask.any().item()):
        return None

    coord_counts = coord_mask.sum(dim=1)
    bad_rows = (coord_counts > 0) & ((coord_counts % 4) != 0)
    if bool(bad_rows.any().item()):
        bad_row = int(bad_rows.nonzero(as_tuple=False)[0].item())
        bad_count = int(coord_counts[bad_row].detach().item())
        raise ValueError(
            "bbox_size_aux requires bbox-only Stage-1 coord supervision with coord-token "
            f"counts divisible by 4; row={bad_row} count={bad_count}"
        )
    if tokenizer is not None:
        for row_idx in range(int(labels.shape[0])):
            coord_count = int(coord_counts[row_idx].detach().item())
            if coord_count <= 0:
                continue
            supervised = labels[row_idx][labels[row_idx] != -100]
            token_ids = [int(t) for t in supervised.detach().cpu().tolist() if int(t) >= 0]
            text = "".join(_decode_pieces(tokenizer, token_ids))
            bbox_count = int(text.count("bbox_2d"))
            if bbox_count <= 0 or "poly" in text:
                raise ValueError(
                    "bbox_size_aux requires bbox-only Stage-1 supervision with explicit "
                    f"bbox_2d fields; row={row_idx}"
                )
            if int(bbox_count * 4) != int(coord_count):
                raise ValueError(
                    "bbox_size_aux requires bbox-only Stage-1 coord supervision aligned to "
                    f"bbox_2d quartets; row={row_idx} bbox_fields={bbox_count} coord_count={coord_count}"
                )

    coord_ids = torch.tensor(coord_token_ids, device=logits.device, dtype=torch.long)
    flat_logits_full = logits_next[coord_mask]
    flat_coord_logits = flat_logits_full.index_select(dim=-1, index=coord_ids)
    flat_target_bins = target_bins_all[coord_mask]
    target_boxes = flat_target_bins.float().reshape(-1, 4) / 999.0

    return Stage1BBoxQuartets(
        coord_logits=flat_coord_logits,
        target_bins=flat_target_bins,
        target_boxes_xyxy=target_boxes,
        coord_slots=int(flat_target_bins.numel()),
        bbox_groups=int(target_boxes.shape[0]),
    )
