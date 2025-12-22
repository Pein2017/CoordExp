from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from src.config.schema import CoordLossConfig, TokenTypeMetricsConfig
from src.coord_tokens.codec import get_coord_token_ids
from src.data_collators.token_types import TokenType, compute_token_types
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_label(row: Mapping[str, Any]) -> str:
    if not isinstance(row, Mapping):
        return "default"
    meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else None
    if meta:
        label = meta.get("_fusion_source") or meta.get("dataset")
        if label:
            return str(label)
    dataset_name = row.get("dataset") or row.get("dataset_name")
    return str(dataset_name) if dataset_name else "default"


def build_dataset_metrics_collator(
    template: Any,
    base_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
    token_type_cfg: Optional[TokenTypeMetricsConfig] = None,
    coord_loss_cfg: Optional[CoordLossConfig] = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Wrap the template collator to attach dataset labels and token types.

    Supports padded batches and packed batches (pack emitted as a list of samples).
    """

    collate_fn = base_collator or template.data_collator
    coord_loss_enabled = bool(coord_loss_cfg and coord_loss_cfg.enabled)
    coord_token_ids: List[int] = []
    coord_token_mask: Optional[torch.Tensor] = None
    tokenizer = getattr(template, "tokenizer", None)
    if coord_loss_enabled and tokenizer is not None:
        coord_token_ids = get_coord_token_ids(tokenizer)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = max(coord_token_ids) + 1 if coord_token_ids else 0
        coord_token_mask = torch.zeros(int(vocab_size), dtype=torch.bool)
        for idx in coord_token_ids:
            if 0 <= idx < coord_token_mask.numel():
                coord_token_mask[idx] = True

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        is_packed_batch = len(batch) > 0 and isinstance(batch[0], (list, tuple))

        if is_packed_batch:
            # Each element is a pack (list of samples). Use a synthetic label for the pack.
            dataset_labels = []
            flat_for_labels: List[Dict[str, Any]] = []
            for pack in batch:
                labels_in_pack = {_resolve_label(sample) for sample in pack if isinstance(sample, Mapping)}
                pack_label = "mixed" if len(labels_in_pack) > 1 else (labels_in_pack.pop() if labels_in_pack else "default")
                dataset_labels.append(pack_label)
                flat_for_labels.append({"dataset": pack_label})
        else:
            dataset_labels = [_resolve_label(row) for row in batch]
            flat_for_labels = batch

        collated = collate_fn(batch)

        # Derive per-sample lengths from attention_mask if present, else input_ids.
        segments: List[int]
        if "attention_mask" in collated:
            am = collated["attention_mask"]
            if isinstance(am, torch.Tensor):
                segments = am.long().sum(dim=-1).tolist()
            else:
                segments = [int(sum(x)) for x in am]
        elif "input_ids" in collated:
            ids = collated["input_ids"]
            if isinstance(ids, torch.Tensor):
                segments = [ids.shape[-1]] * ids.shape[0]
            else:
                segments = [len(x) for x in ids]
        else:
            segments = [0 for _ in dataset_labels]

        collated["dataset_labels"] = dataset_labels
        collated["dataset_segments"] = segments

        _maybe_attach_token_types(
            collated=collated,
            raw_batch=batch if is_packed_batch else flat_for_labels,
            dataset_labels=dataset_labels,
            template=template,
            cfg=token_type_cfg,
            packed=is_packed_batch,
        )
        _maybe_attach_coord_features(
            collated=collated,
            raw_batch=batch if is_packed_batch else flat_for_labels,
            coord_loss_cfg=coord_loss_cfg,
            coord_token_ids=coord_token_ids,
            coord_token_mask=coord_token_mask,
            packed=is_packed_batch,
        )
        return collated

    return _collate


def _maybe_attach_token_types(
    *,
    collated: Dict[str, Any],
    raw_batch: Sequence[Any],
    dataset_labels: Sequence[str],
    template: Any,
    cfg: Optional[TokenTypeMetricsConfig],
    packed: bool,
) -> None:
    if cfg is None or not cfg.enabled:
        return

    labels_tensor = collated.get("labels")
    if labels_tensor is None or not isinstance(labels_tensor, torch.Tensor):
        return

    tokenizer = getattr(template, "tokenizer", None)
    template_meta = getattr(template, "template_meta", None)
    suffix_tokens = getattr(template_meta, "suffix", None) if template_meta else None

    if tokenizer is None:
        return

    include_set = set(cfg.include)
    exclude_set = set(cfg.exclude)

    def _label_included(label: str) -> bool:
        key = str(label).lower()
        return key in include_set and key not in exclude_set

    # Packed path: raw_batch is a list of packs (each pack is a list of samples)
    if packed:
        token_type_rows: List[torch.Tensor] = []
        for pack, pack_label in zip(raw_batch, dataset_labels):
            if not isinstance(pack, (list, tuple)):
                pack = [pack]

            per_sample_types: List[torch.Tensor] = []
            for sample in pack:
                if not isinstance(sample, Mapping):
                    continue
                label = _resolve_label(sample)
                length = sample.get("length")
                if length is None and isinstance(sample.get("labels"), (list, tuple, torch.Tensor)):
                    length = len(sample.get("labels"))

                if not _label_included(label):
                    if length is None:
                        continue
                    per_sample_types.append(
                        torch.full(
                            (int(length),),
                            TokenType.IGNORE,
                            dtype=torch.long,
                            device=labels_tensor.device,
                        )
                    )
                    continue

                payload = sample.get("assistant_payload")
                if payload is None:
                    continue

                labels_row = sample.get("labels")
                if labels_row is None:
                    continue
                labels_row_t = (
                    labels_row
                    if isinstance(labels_row, torch.Tensor)
                    else torch.tensor(labels_row, dtype=torch.long, device=labels_tensor.device)
                )

                attn_row_raw = sample.get("attention_mask")
                attn_row = (
                    attn_row_raw
                    if isinstance(attn_row_raw, torch.Tensor)
                    else torch.tensor(attn_row_raw, dtype=torch.long, device=labels_tensor.device)
                    if attn_row_raw is not None
                    else None
                )

                token_types = compute_token_types(
                    tokenizer=tokenizer,
                    payload=payload,
                    labels=labels_row_t,
                    attention_mask=attn_row,
                    suffix_tokens=suffix_tokens,
                )
                if token_types is None or token_types.shape[0] != labels_row_t.shape[0]:
                    token_types = torch.full_like(labels_row_t, TokenType.IGNORE)
                per_sample_types.append(token_types.to(labels_tensor.device))

            if not per_sample_types:
                token_type_rows.append(torch.full_like(labels_tensor[0], TokenType.IGNORE))
                continue

            concat_types = torch.cat(per_sample_types, dim=0)
            target_len = labels_tensor.shape[1]
            if concat_types.shape[0] != target_len:
                logger.debug(
                    "Packed token_types length mismatch: got %s expected %s (pack_label=%s)",
                    concat_types.shape[0],
                    target_len,
                    pack_label,
                )
                if concat_types.shape[0] < target_len:
                    pad = torch.full(
                        (target_len - concat_types.shape[0],),
                        TokenType.IGNORE,
                        dtype=torch.long,
                        device=labels_tensor.device,
                    )
                    concat_types = torch.cat([concat_types, pad], dim=0)
                else:
                    concat_types = concat_types[:target_len]
            token_type_rows.append(concat_types)

        if token_type_rows:
            collated["token_types"] = torch.stack(token_type_rows, dim=0)
        return

    # Non-packed path (padded batches)
    attention_mask = collated.get("attention_mask")

    token_type_list: List[torch.Tensor] = []
    for idx, (raw, label) in enumerate(zip(raw_batch, dataset_labels)):
        if not isinstance(raw, Mapping):
            token_type_list.append(torch.full_like(labels_tensor[idx], TokenType.IGNORE))
            continue

        included = _label_included(label)
        payload = raw.get("assistant_payload")
        if not included or payload is None:
            token_type_list.append(torch.full_like(labels_tensor[idx], TokenType.IGNORE))
            continue

        labels_row: torch.Tensor = labels_tensor[idx]
        attn_row: Optional[torch.Tensor] = None
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            attn_row = attention_mask[idx]

        token_types = compute_token_types(
            tokenizer=tokenizer,
            payload=payload,
            labels=labels_row,
            attention_mask=attn_row,
            suffix_tokens=suffix_tokens,
        )
        if token_types is None or token_types.shape[0] != labels_row.shape[0]:
            logger.debug(
                "Falling back to IGNORE token types (label=%s, got=%s, expected=%s)",
                label,
                None if token_types is None else token_types.shape,
                labels_row.shape,
            )
            token_types = torch.full_like(labels_row, TokenType.IGNORE)
        token_type_list.append(token_types)

    if token_type_list:
        collated["token_types"] = torch.stack(token_type_list, dim=0)


def _flatten_coord_values(points: Any) -> List[Any]:
    if points is None:
        return []
    if isinstance(points, (str, bytes)):
        return [points]
    if not isinstance(points, (list, tuple)):
        return []
    flat: List[Any] = []
    for item in points:
        if isinstance(item, (list, tuple)):
            flat.extend(list(item))
        else:
            flat.append(item)
    return flat


def _extract_coord_spans_from_payload(payload: Any) -> tuple[List[Dict[str, Any]], int]:
    spans: List[Dict[str, Any]] = []
    total = 0
    if not isinstance(payload, Mapping):
        return spans, total
    for _, obj in payload.items():
        if not isinstance(obj, Mapping):
            continue
        geom_type = None
        geom_points = None
        for key in ("bbox_2d", "poly", "line"):
            if key in obj and obj[key] is not None:
                geom_type = key
                geom_points = obj[key]
                break
        if geom_type is None or geom_points is None:
            continue
        coords = _flatten_coord_values(geom_points)
        if not coords:
            continue
        coord_len = len(coords)
        spans.append({"geom_type": geom_type, "start": total, "coord_len": coord_len})
        total += coord_len
    return spans, total


def _maybe_attach_coord_features(
    *,
    collated: Dict[str, Any],
    raw_batch: Sequence[Any],
    coord_loss_cfg: Optional[CoordLossConfig],
    coord_token_ids: Sequence[int],
    coord_token_mask: Optional[torch.Tensor],
    packed: bool,
) -> None:
    if coord_loss_cfg is None or not coord_loss_cfg.enabled:
        return

    labels = collated.get("labels")
    if labels is None or not isinstance(labels, torch.Tensor):
        return

    if coord_token_ids:
        max_label = int(labels.max().item()) if labels.numel() else 0
        mask = coord_token_mask
        if mask is None or max_label >= mask.numel():
            vocab_size = max(max_label + 1, max(coord_token_ids) + 1)
            mask = torch.zeros(int(vocab_size), dtype=torch.bool)
            for idx in coord_token_ids:
                if 0 <= idx < mask.numel():
                    mask[idx] = True
        mask = mask.to(labels.device)
        labels_safe = labels
        if labels_safe.min().item() < 0:
            labels_safe = labels_safe.clamp(min=0)
        coord_positions = mask[labels_safe] & (labels != -100)
        loss_scale = torch.full(
            labels.shape,
            float(coord_loss_cfg.non_coord_ce_weight),
            device=labels.device,
            dtype=torch.float,
        )
        loss_scale[coord_positions] = float(coord_loss_cfg.coord_ce_weight)
        loss_scale = loss_scale.masked_fill(labels == -100, 0.0)
        collated["loss_scale"] = loss_scale

    coord_spans_batch: List[List[Dict[str, Any]]] = []
    if packed:
        for pack in raw_batch:
            pack_spans: List[Dict[str, Any]] = []
            offset = 0
            if not isinstance(pack, (list, tuple)):
                pack = [pack]
            for sample in pack:
                if not isinstance(sample, Mapping):
                    continue
                spans, total = _extract_coord_spans_from_payload(
                    sample.get("assistant_payload")
                )
                for span in spans:
                    updated = dict(span)
                    updated["start"] = int(updated.get("start", 0)) + offset
                    pack_spans.append(updated)
                offset += int(total)
            coord_spans_batch.append(pack_spans)
    else:
        for sample in raw_batch:
            if not isinstance(sample, Mapping):
                coord_spans_batch.append([])
                continue
            spans, _ = _extract_coord_spans_from_payload(
                sample.get("assistant_payload")
            )
            coord_spans_batch.append(spans)

    if coord_spans_batch:
        collated["coord_spans"] = coord_spans_batch
