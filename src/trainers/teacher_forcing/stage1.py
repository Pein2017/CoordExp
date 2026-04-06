from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.common.semantic_desc import normalize_desc
from src.utils.coordjson_transpiler import parse_coordjson


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
    adjacent_prev_target_bins: torch.Tensor | None = None
    adjacent_has_prev_mask: torch.Tensor | None = None
    adjacent_same_desc_prev_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class _Stage1RowBBoxMeta:
    bbox_group_sizes: tuple[int, ...]
    desc_keys: tuple[str, ...] = ()


def _decode_pieces(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [
        tokenizer.decode(
            [int(t)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in token_ids
    ]


def _extract_row_bbox_meta(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    expected_bbox_count: int,
    row_idx: int,
    require_desc_keys: bool,
    object_field_order: str,
) -> _Stage1RowBBoxMeta:
    pieces = _decode_pieces(tokenizer, token_ids)
    text = "".join(pieces)
    desc_keys: list[str] = []
    bbox_group_sizes: list[int] = []
    cursor = 0
    text_len = len(text)
    while int(cursor) < int(text_len):
        parsed = parse_coordjson(
            text[int(cursor) :],
            mode="salvage",
            object_field_order=object_field_order,
        )
        if bool(parsed.parse_failed):
            break
        container_end = parsed.container_end
        if container_end is None or int(container_end) <= 0:
            raise ValueError(
                "Stage-1 bbox quartet recovery did not advance while parsing "
                f"teacher-forced CoordJSON; row={row_idx}"
            )
        bbox_records = []
        for record in parsed.records:
            if str(record.geometry_key) != "bbox_2d":
                raise ValueError(
                    "bbox_size_aux requires bbox-only Stage-1 supervision with explicit "
                    f"bbox_2d fields; row={row_idx}"
                )
            bbox_records.append(record)
            if require_desc_keys:
                desc_keys.append(normalize_desc(str(record.desc)))
        if bbox_records:
            bbox_group_sizes.append(int(len(bbox_records)))
        cursor += int(container_end)

    bbox_count = int(sum(int(size) for size in bbox_group_sizes))
    if bbox_count <= 0:
        raise ValueError(
            "bbox_size_aux requires bbox-only Stage-1 supervision with explicit "
            f"bbox_2d fields; row={row_idx}"
        )
    if int(bbox_count) != int(expected_bbox_count):
        raise ValueError(
            "bbox_size_aux requires bbox-only Stage-1 coord supervision aligned to "
            f"bbox_2d quartets; row={row_idx} bbox_fields={bbox_count} coord_count={expected_bbox_count * 4}"
        )
    if require_desc_keys and len(desc_keys) != int(expected_bbox_count):
        raise ValueError(
            "Stage-1 adjacent repulsion same_desc mode requires one desc span per "
            f"bbox_2d object; row={row_idx} spans={len(desc_keys)} bbox_fields={expected_bbox_count}"
        )
    return _Stage1RowBBoxMeta(
        bbox_group_sizes=tuple(int(size) for size in bbox_group_sizes),
        desc_keys=tuple(desc_keys),
    )


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
    include_adjacent_metadata: bool = False,
    require_desc_keys: bool = False,
    object_field_order: str = "desc_first",
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
    if include_adjacent_metadata and tokenizer is None:
        raise ValueError(
            "Stage-1 adjacent repulsion requires a tokenizer to recover packed sample boundaries"
        )

    row_meta_by_idx: dict[int, _Stage1RowBBoxMeta] = {}
    if tokenizer is not None:
        for row_idx in range(int(labels.shape[0])):
            coord_count = int(coord_counts[row_idx].detach().item())
            if coord_count <= 0 or bool(bad_rows[row_idx].detach().item()):
                continue
            supervised = labels[row_idx][labels[row_idx] != -100]
            token_ids = [int(t) for t in supervised.detach().cpu().tolist() if int(t) >= 0]
            if include_adjacent_metadata:
                row_meta_by_idx[row_idx] = _extract_row_bbox_meta(
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    expected_bbox_count=int(coord_count // 4),
                    row_idx=row_idx,
                    require_desc_keys=bool(require_desc_keys),
                    object_field_order=object_field_order,
                )
            else:
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
        empty_prev = logits_next.new_zeros((0, 4), dtype=torch.long)
        empty_mask = target_bins_all.new_zeros((0,), dtype=torch.bool)
        return Stage1BBoxQuartets(
            coord_logits=empty_coord_logits,
            target_bins=empty_target_bins,
            target_boxes_xyxy=empty_boxes,
            coord_slots=0,
            bbox_groups=0,
            skipped_incomplete_rows=skipped_incomplete_rows,
            skipped_incomplete_coord_slots=skipped_incomplete_coord_slots,
            adjacent_prev_target_bins=empty_prev if include_adjacent_metadata else None,
            adjacent_has_prev_mask=empty_mask if include_adjacent_metadata else None,
            adjacent_same_desc_prev_mask=empty_mask if include_adjacent_metadata else None,
        )

    coord_ids = torch.tensor(coord_token_ids, device=logits.device, dtype=torch.long)
    flat_logits_full = logits_next[coord_mask]
    flat_coord_logits = flat_logits_full.index_select(dim=-1, index=coord_ids)
    flat_target_bins = target_bins_all[coord_mask]
    target_boxes = flat_target_bins.float().reshape(-1, 4) / 999.0

    adjacent_prev_target_bins: torch.Tensor | None = None
    adjacent_has_prev_mask: torch.Tensor | None = None
    adjacent_same_desc_prev_mask: torch.Tensor | None = None
    if include_adjacent_metadata:
        prev_rows: list[torch.Tensor] = []
        has_prev_rows: list[torch.Tensor] = []
        same_desc_rows: list[torch.Tensor] = []
        for row_idx in range(int(labels.shape[0])):
            row_target_bins = target_bins_all[row_idx][coord_mask[row_idx]]
            if int(row_target_bins.numel()) == 0:
                continue
            row_groups = row_target_bins.reshape(-1, 4)
            row_meta = row_meta_by_idx.get(row_idx)
            if row_meta is None:
                raise ValueError(
                    "Stage-1 adjacent repulsion requires tokenizer-backed row metadata "
                    f"for bbox grouping; row={row_idx}"
                )
            row_prev = torch.zeros_like(row_groups)
            row_has_prev = torch.zeros(
                (int(row_groups.shape[0]),),
                dtype=torch.bool,
                device=row_groups.device,
            )
            row_same_desc = torch.zeros_like(row_has_prev)
            if sum(int(size) for size in row_meta.bbox_group_sizes) != int(
                row_groups.shape[0]
            ):
                raise ValueError(
                    "Stage-1 adjacent repulsion requires bbox group recovery to align "
                    f"with coord quartets; row={row_idx}"
                )
            cursor = 0
            for group_size in row_meta.bbox_group_sizes:
                group_size = int(group_size)
                if group_size > 1:
                    row_prev[cursor + 1 : cursor + group_size] = row_groups[
                        cursor : cursor + group_size - 1
                    ]
                    row_has_prev[cursor + 1 : cursor + group_size] = True
                cursor += group_size

            if require_desc_keys:
                desc_keys = list(row_meta.desc_keys)
                if len(desc_keys) != int(row_groups.shape[0]):
                    raise ValueError(
                        "Stage-1 adjacent repulsion same_desc mode requires bbox groups "
                        f"and desc keys to align; row={row_idx}"
                    )
                cursor = 0
                for group_size in row_meta.bbox_group_sizes:
                    group_size = int(group_size)
                    for group_idx in range(cursor + 1, cursor + group_size):
                        row_same_desc[group_idx] = (
                            desc_keys[group_idx] == desc_keys[group_idx - 1]
                        )
                    cursor += group_size

            prev_rows.append(row_prev)
            has_prev_rows.append(row_has_prev)
            same_desc_rows.append(row_same_desc)

        if prev_rows:
            adjacent_prev_target_bins = torch.cat(prev_rows, dim=0)
            adjacent_has_prev_mask = torch.cat(has_prev_rows, dim=0)
            adjacent_same_desc_prev_mask = torch.cat(same_desc_rows, dim=0)
        else:
            adjacent_prev_target_bins = flat_target_bins.new_zeros((0, 4))
            adjacent_has_prev_mask = flat_target_bins.new_zeros((0,), dtype=torch.bool)
            adjacent_same_desc_prev_mask = flat_target_bins.new_zeros(
                (0,), dtype=torch.bool
            )

    return Stage1BBoxQuartets(
        coord_logits=flat_coord_logits,
        target_bins=flat_target_bins,
        target_boxes_xyxy=target_boxes,
        coord_slots=int(flat_target_bins.numel()),
        bbox_groups=int(target_boxes.shape[0]),
        skipped_incomplete_rows=skipped_incomplete_rows,
        skipped_incomplete_coord_slots=skipped_incomplete_coord_slots,
        adjacent_prev_target_bins=adjacent_prev_target_bins,
        adjacent_has_prev_mask=adjacent_has_prev_mask,
        adjacent_same_desc_prev_mask=adjacent_same_desc_prev_mask,
    )
