from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from src.config.schema import TokenTypeMetricsConfig
from src.data_collators.token_types import TokenType, compute_token_types
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_dataset_label(row: Mapping[str, Any]) -> str:
    """Best-effort label resolution for debug/telemetry (aggregate-only)."""

    meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else None
    if meta:
        label = meta.get("_fusion_source") or meta.get("dataset")
        if label:
            return str(label)
    dataset_name = row.get("dataset") or row.get("dataset_name")
    return str(dataset_name) if dataset_name else "default"


@dataclass
class DatasetMeta:
    dataset_labels: list[str]
    pack_num_samples: list[int]
    packed: bool


class DatasetMetaEnricher:
    """Attach stable dataset-meta extras: labels/segments/pack_num_samples."""

    label_field = "dataset_labels"
    segment_field = "dataset_segments"
    pack_num_samples_field = "pack_num_samples"

    def __call__(self, *, batch: list[Any], collated: dict[str, Any]) -> DatasetMeta:
        is_packed_batch = len(batch) > 0 and isinstance(batch[0], (list, tuple))

        dataset_labels: list[str] = []
        pack_num_samples: list[int] = []

        if is_packed_batch:
            # Each element is a pack (list of samples). Use a synthetic label for the pack.
            for pack in batch:
                pack_seq = pack if isinstance(pack, (list, tuple)) else [pack]
                labels_in_pack = {
                    resolve_dataset_label(sample)
                    for sample in pack_seq
                    if isinstance(sample, Mapping)
                }
                pack_label = (
                    "mixed"
                    if len(labels_in_pack) > 1
                    else (labels_in_pack.pop() if labels_in_pack else "default")
                )
                dataset_labels.append(pack_label)
                try:
                    pack_num_samples.append(int(len(pack_seq)))
                except Exception:
                    pack_num_samples.append(1)
        else:
            for row in batch:
                if isinstance(row, Mapping):
                    dataset_labels.append(resolve_dataset_label(row))
                else:
                    dataset_labels.append("default")
            pack_num_samples = [1 for _ in dataset_labels]

        # Derive per-sample lengths from attention_mask if present, else input_ids.
        segments: list[int]
        if "attention_mask" in collated:
            am = collated["attention_mask"]
            if isinstance(am, torch.Tensor):
                segments = am.long().sum(dim=-1).tolist()
            else:
                segments = [int(sum(x)) for x in am]
        elif "input_ids" in collated:
            ids = collated["input_ids"]
            if isinstance(ids, torch.Tensor):
                segments = [int(ids.shape[-1])] * int(ids.shape[0])
            else:
                segments = [len(x) for x in ids]
        else:
            segments = [0 for _ in dataset_labels]

        collated[self.label_field] = dataset_labels
        collated[self.segment_field] = segments

        # Number of original samples concatenated into each training "unit" (pack-aware).
        try:
            labels_t = collated.get("labels")
            device = labels_t.device if isinstance(labels_t, torch.Tensor) else None
            collated[self.pack_num_samples_field] = torch.tensor(
                pack_num_samples, dtype=torch.long, device=device
            )
        except Exception:
            raise

        return DatasetMeta(
            dataset_labels=dataset_labels,
            pack_num_samples=pack_num_samples,
            packed=is_packed_batch,
        )


class InstabilityMetaEnricher:
    """Attach `instability_meta_json` for post-mortem debug dumps."""

    out_field = "instability_meta_json"

    def __init__(self, *, max_meta_samples: int = 16):
        self._max_meta_samples = max(0, int(max_meta_samples))

    def __call__(self, *, batch: list[Any], collated: dict[str, Any], packed: bool) -> None:
        meta = self._build_instability_meta(batch, packed=packed, max_samples=self._max_meta_samples)
        collated[self.out_field] = json.dumps(meta, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _extract_sample_debug(sample: Mapping[str, Any]) -> dict[str, Any]:
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
        self,
        raw_batch: list[Any],
        *,
        packed: bool,
        max_samples: int,
    ) -> list[dict[str, Any]]:
        packs: list[Sequence[Any]]
        if packed:
            packs = [
                pack if isinstance(pack, (list, tuple)) else [pack]
                for pack in raw_batch
            ]
        else:
            packs = [[row] for row in raw_batch]

        out: list[dict[str, Any]] = []
        for pack in packs:
            samples: list[dict[str, Any]] = []
            total_len = 0
            for item in pack:
                if not isinstance(item, Mapping):
                    continue
                if max_samples > 0 and len(samples) >= max_samples:
                    continue
                s = self._extract_sample_debug(item)
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


class TokenTypesEnricher:
    """Attach `token_types` tensor for token-type metrics (best-effort)."""

    out_field = "token_types"

    def __init__(self, *, template: Any, cfg: TokenTypeMetricsConfig):
        self._template = template
        self._cfg = cfg

    def __call__(
        self,
        *,
        collated: dict[str, Any],
        raw_batch: Sequence[Any],
        dataset_labels: Sequence[str],
        packed: bool,
    ) -> None:
        cfg = self._cfg
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return

        labels_tensor = collated.get("labels")
        if labels_tensor is None or not isinstance(labels_tensor, torch.Tensor):
            return

        tokenizer = getattr(self._template, "tokenizer", None)
        template_meta = getattr(self._template, "template_meta", None)
        suffix_tokens = getattr(template_meta, "suffix", None) if template_meta else None
        if tokenizer is None:
            return

        include_set = set(getattr(cfg, "include", ()) or ())
        exclude_set = set(getattr(cfg, "exclude", ()) or ())

        def _label_included(label: str) -> bool:
            key = str(label).lower()
            return key in include_set and key not in exclude_set

        # Packed path: raw_batch is a list of packs (each pack is a list of samples)
        if packed:
            token_type_rows: list[torch.Tensor] = []
            for pack, pack_label in zip(raw_batch, dataset_labels):
                if not isinstance(pack, (list, tuple)):
                    pack = [pack]

                per_sample_types: list[torch.Tensor] = []
                for sample in pack:
                    if not isinstance(sample, Mapping):
                        continue
                    label = resolve_dataset_label(sample)
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
                collated[self.out_field] = torch.stack(token_type_rows, dim=0)
            return

        # Non-packed path (padded batches)
        attention_mask = collated.get("attention_mask")

        token_type_list: list[torch.Tensor] = []
        for idx, (raw, label) in enumerate(zip(raw_batch, dataset_labels)):
            if not isinstance(raw, Mapping):
                token_type_list.append(torch.full_like(labels_tensor[idx], TokenType.IGNORE))
                continue

            included = _label_included(label)
            payload = raw.get("assistant_payload")
            if not included or payload is None:
                token_type_list.append(torch.full_like(labels_tensor[idx], TokenType.IGNORE))
                continue

            labels_row = labels_tensor[idx]
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
            collated[self.out_field] = torch.stack(token_type_list, dim=0)
