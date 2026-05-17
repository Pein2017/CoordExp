from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.trainers.teacher_forcing.geometry import bbox_tensor_to_xyxy


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
    skipped_incomplete_rows: int = 0
    skipped_incomplete_coord_slots: int = 0


def _decode_pieces(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [
        tokenizer.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in token_ids
    ]


def _validate_row_bbox_only(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    expected_bbox_count: int,
    row_idx: int,
) -> None:
    text = "".join(_decode_pieces(tokenizer, token_ids))
    bbox_count = int(text.count("bbox_2d"))
    if bbox_count <= 0 or "poly" in text:
        raise ValueError(
            "bbox_size_aux requires bbox-only Stage-1 supervision with explicit "
            f"bbox_2d fields; row={row_idx}"
        )
    if int(bbox_count) != int(expected_bbox_count):
        raise ValueError(
            "bbox_size_aux requires bbox-only Stage-1 coord supervision aligned to "
            f"bbox_2d quartets; row={row_idx} bbox_fields={bbox_count} coord_count={expected_bbox_count * 4}"
        )


def extract_stage1_bbox_quartets(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    coord_token_ids: Sequence[int],
    coord_id_map: torch.Tensor,
    tokenizer: Any | None = None,
    object_field_order: str = "desc_first",
    bbox_format: str = "xyxy",
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
    skipped_incomplete_rows = int(bad_rows.sum().detach().item())
    skipped_incomplete_coord_slots = (
        int(coord_counts[bad_rows].sum().detach().item())
        if skipped_incomplete_rows > 0
        else 0
    )
    if skipped_incomplete_rows > 0:
        coord_mask = coord_mask & (~bad_rows).unsqueeze(1)

    if tokenizer is not None:
        for row_idx in range(int(labels.shape[0])):
            coord_count = int(coord_counts[row_idx].detach().item())
            if coord_count <= 0 or bool(bad_rows[row_idx].detach().item()):
                continue
            supervised = labels[row_idx][labels[row_idx] != -100]
            token_ids = [int(t) for t in supervised.detach().cpu().tolist() if int(t) >= 0]
            _validate_row_bbox_only(
                tokenizer=tokenizer,
                token_ids=token_ids,
                expected_bbox_count=int(coord_count // 4),
                row_idx=row_idx,
            )

    if not bool(coord_mask.any().item()):
        empty_coord_logits = logits_next.new_zeros((0, len(coord_token_ids)))
        empty_target_bins = target_bins_all.new_zeros((0,), dtype=torch.long)
        empty_boxes = logits_next.new_zeros((0, 4))
        return Stage1BBoxQuartets(
            coord_logits=empty_coord_logits,
            target_bins=empty_target_bins,
            target_boxes_xyxy=empty_boxes,
            coord_slots=0,
            bbox_groups=0,
            skipped_incomplete_rows=skipped_incomplete_rows,
            skipped_incomplete_coord_slots=skipped_incomplete_coord_slots,
        )

    coord_ids = torch.tensor(coord_token_ids, device=logits.device, dtype=torch.long)
    flat_logits_full = logits_next[coord_mask]
    flat_coord_logits = flat_logits_full.index_select(dim=-1, index=coord_ids)
    flat_target_bins = target_bins_all[coord_mask]
    target_boxes_raw = flat_target_bins.float().reshape(-1, 4) / 999.0
    target_boxes = bbox_tensor_to_xyxy(
        target_boxes_raw,
        bbox_format=bbox_format,
    )

    return Stage1BBoxQuartets(
        coord_logits=flat_coord_logits,
        target_bins=flat_target_bins,
        target_boxes_xyxy=target_boxes,
        coord_slots=int(flat_target_bins.numel()),
        bbox_groups=int(target_boxes.shape[0]),
        skipped_incomplete_rows=skipped_incomplete_rows,
        skipped_incomplete_coord_slots=skipped_incomplete_coord_slots,
    )
