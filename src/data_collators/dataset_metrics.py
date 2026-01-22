import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.config.schema import TokenTypeMetricsConfig
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
    instability_monitor_cfg: Optional[Mapping[str, Any]] = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Wrap the template collator to attach dataset labels and token types.

    Supports padded batches and packed batches (pack emitted as a list of samples).
    """

    collate_fn = base_collator or template.data_collator

    instab_enabled = False
    max_meta_samples = 16
    if isinstance(instability_monitor_cfg, Mapping):
        instab_enabled = bool(instability_monitor_cfg.get("enabled", False))
        try:
            max_meta_samples = int(instability_monitor_cfg.get("max_meta_samples", 16))
        except Exception:
            max_meta_samples = 16
        max_meta_samples = max(0, max_meta_samples)

    def _extract_sample_debug(sample: Mapping[str, Any]) -> Dict[str, Any]:
        length = sample.get("length")
        if length is None:
            input_ids = sample.get("input_ids")
            try:
                length = int(len(input_ids)) if input_ids is not None else None
            except Exception:
                length = None
        return {
            "dataset": sample.get("dataset") or sample.get("dataset_name"),
            "base_idx": sample.get("base_idx"),
            "sample_id": sample.get("sample_id"),
            "length": length,
        }

    def _build_instability_meta(
        raw_batch: List[Dict[str, Any]],
        *,
        packed: bool,
        max_samples: int,
    ) -> List[Dict[str, Any]]:
        packs: List[Sequence[Any]]
        if packed:
            packs = [
                pack if isinstance(pack, (list, tuple)) else [pack] for pack in raw_batch
            ]
        else:
            packs = [[row] for row in raw_batch]

        out: List[Dict[str, Any]] = []
        for pack in packs:
            samples: List[Dict[str, Any]] = []
            total_len = 0
            for item in pack:
                if not isinstance(item, Mapping):
                    continue
                if max_samples > 0 and len(samples) >= max_samples:
                    continue
                s = _extract_sample_debug(item)
                if isinstance(s.get("length"), int):
                    total_len += int(s["length"])
                samples.append(s)
            out.append(
                {
                    "num_samples": len(samples),
                    "total_length": int(total_len),
                    "samples": samples,
                }
            )
        return out

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        is_packed_batch = len(batch) > 0 and isinstance(batch[0], (list, tuple))

        if is_packed_batch:
            # Each element is a pack (list of samples). Use a synthetic label for the pack.
            dataset_labels = []
            flat_for_labels: List[Dict[str, Any]] = []
            pack_num_samples: List[int] = []
            for pack in batch:
                labels_in_pack = {
                    _resolve_label(sample)
                    for sample in pack
                    if isinstance(sample, Mapping)
                }
                pack_label = (
                    "mixed"
                    if len(labels_in_pack) > 1
                    else (labels_in_pack.pop() if labels_in_pack else "default")
                )
                dataset_labels.append(pack_label)
                flat_for_labels.append({"dataset": pack_label})
                try:
                    pack_num_samples.append(int(len(pack)))
                except Exception:
                    pack_num_samples.append(1)
        else:
            dataset_labels = [_resolve_label(row) for row in batch]
            flat_for_labels = batch
            pack_num_samples = [1 for _ in dataset_labels]

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
        # Number of original samples that were concatenated into each training "unit".
        # This is used for per-sample-normalized logging in packed training.
        try:
            labels_t = collated.get("labels")
            device = labels_t.device if isinstance(labels_t, torch.Tensor) else None
            collated["pack_num_samples"] = torch.tensor(
                pack_num_samples, dtype=torch.long, device=device
            )
        except Exception:
            # Best-effort only; never block training.
            pass

        if instab_enabled:
            try:
                meta = _build_instability_meta(
                    batch, packed=is_packed_batch, max_samples=max_meta_samples
                )
                collated["instability_meta_json"] = json.dumps(
                    meta, ensure_ascii=True, sort_keys=True
                )
            except Exception:
                # Best-effort only; never block training.
                pass

        _maybe_attach_token_types(
            collated=collated,
            raw_batch=batch if is_packed_batch else flat_for_labels,
            dataset_labels=dataset_labels,
            template=template,
            cfg=token_type_cfg,
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
                if length is None and isinstance(
                    sample.get("labels"), (list, tuple, torch.Tensor)
                ):
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
                    else torch.tensor(
                        labels_row, dtype=torch.long, device=labels_tensor.device
                    )
                )

                attn_row_raw = sample.get("attention_mask")
                attn_row = (
                    attn_row_raw
                    if isinstance(attn_row_raw, torch.Tensor)
                    else torch.tensor(
                        attn_row_raw, dtype=torch.long, device=labels_tensor.device
                    )
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
                token_type_rows.append(
                    torch.full_like(labels_tensor[0], TokenType.IGNORE)
                )
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
                concat_types = torch.full(
                    (target_len,),
                    TokenType.IGNORE,
                    dtype=torch.long,
                    device=labels_tensor.device,
                )
            token_type_rows.append(concat_types)

        if token_type_rows:
            collated["token_types"] = torch.stack(token_type_rows, dim=0)
        return

    # Non-packed path (padded batches)
    attention_mask = collated.get("attention_mask")

    token_type_list: List[torch.Tensor] = []
    for idx, (raw, label) in enumerate(zip(raw_batch, dataset_labels)):
        if not isinstance(raw, Mapping):
            token_type_list.append(
                torch.full_like(labels_tensor[idx], TokenType.IGNORE)
            )
            continue

        included = _label_included(label)
        payload = raw.get("assistant_payload")
        if not included or payload is None:
            token_type_list.append(
                torch.full_like(labels_tensor[idx], TokenType.IGNORE)
            )
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
