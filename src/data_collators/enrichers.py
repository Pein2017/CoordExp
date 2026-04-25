from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from src.config.schema import TokenTypeMetricsConfig
from src.coord_tokens.codec import get_coord_token_ids
from src.data_collators.token_types import TokenType, compute_token_types
from src.trainers.rollout_matching.parsing import find_desc_value_token_positions_by_span
from src.trainers.batch_extras import SFT_STRUCTURAL_CLOSE_TOKEN_WEIGHTS_KEY
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_dataset_label(row: Mapping[str, Any]) -> str:
    """Best-effort label resolution for debug/telemetry (aggregate-only)."""

    meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else None
    if meta:
        label = meta.get("dataset")
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
                pack_num_samples.append(int(len(pack_seq)))
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
        labels_t = collated.get("labels")
        device = labels_t.device if isinstance(labels_t, torch.Tensor) else None
        collated[self.pack_num_samples_field] = torch.tensor(
            pack_num_samples, dtype=torch.long, device=device
        )

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
            except TypeError:
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


def _extract_message_text(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        text = "\n".join(str(item) for item in texts if item is not None)
        return text or None
    return None


def _replace_last_assistant_text(
    messages: Sequence[Mapping[str, Any]], text: str
) -> list[dict[str, Any]]:
    out = [dict(message) for message in messages if isinstance(message, Mapping)]
    assistant_indices = [
        index for index, message in enumerate(out) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("sft_structural_close requires an assistant message")
    assistant = out[assistant_indices[-1]]
    content = assistant.get("content")
    if isinstance(content, str):
        assistant["content"] = text
    elif isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        replaced = False
        new_content: list[Any] = []
        for item in content:
            if (
                not replaced
                and isinstance(item, Mapping)
                and item.get("type") == "text"
            ):
                new_item = dict(item)
                new_item["text"] = text
                new_content.append(new_item)
                replaced = True
            else:
                new_content.append(item)
        if not replaced:
            new_content.append({"type": "text", "text": text})
        assistant["content"] = new_content
    else:
        assistant["content"] = [{"type": "text", "text": text}]
    return out


def _split_system_message(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    resolved = [dict(message) for message in messages if isinstance(message, Mapping)]
    if resolved and resolved[0].get("role") == "system":
        system = _extract_message_text(resolved.pop(0))
        return resolved, system
    return resolved, None


class _TemporaryTemplateSystem:
    def __init__(self, template: Any, system_prompt: str | None) -> None:
        self.template = template
        self.system_prompt = system_prompt
        self.had_system = False
        self.original_system = None

    def __enter__(self):
        self.had_system = hasattr(self.template, "system")
        self.original_system = (
            getattr(self.template, "system", None) if self.had_system else None
        )
        if self.system_prompt is not None:
            setattr(self.template, "system", self.system_prompt)

    def __exit__(self, exc_type, exc, tb):
        if self.system_prompt is not None:
            if self.had_system:
                setattr(self.template, "system", self.original_system)
            elif hasattr(self.template, "system"):
                delattr(self.template, "system")
        return False


class SFTStructuralCloseEnricher:
    """Attach base-CE weights for the final global CoordJSON close sequence."""

    out_field = SFT_STRUCTURAL_CLOSE_TOKEN_WEIGHTS_KEY

    def __init__(self, *, template: Any, cfg: Any):
        self._template = template
        self._cfg = cfg

    def __call__(
        self,
        *,
        collated: dict[str, Any],
        raw_batch: Sequence[Any],
        packed: bool,
    ) -> None:
        if self._cfg is None or not bool(getattr(self._cfg, "enabled", False)):
            return

        labels = collated.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError("custom.sft_structural_close requires collated labels")
        if packed:
            raise ValueError(
                "custom.sft_structural_close requires non-packed batches; "
                "set training.packing=false and training.eval_packing=false."
            )

        final_weight = float(getattr(self._cfg, "final_close_weight", 1.0))
        weights = torch.ones(labels.shape, dtype=torch.float32, device=labels.device)
        weights = torch.where(labels.ne(-100), weights, torch.zeros_like(weights))

        for row_index, raw in enumerate(raw_batch):
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"custom.sft_structural_close expected mapping row at index {row_index}"
                )
            start, end = self._final_close_token_span(raw, row_index=row_index)
            seq_len = int(labels.shape[1])
            start = max(0, min(int(start), seq_len))
            end = max(start, min(int(end), seq_len))
            if end <= start:
                raise ValueError(
                    f"custom.sft_structural_close resolved empty final-close span for row {row_index}"
                )
            weights[row_index, start:end] = final_weight

        collated[self.out_field] = weights

    def _final_close_token_span(
        self, raw: Mapping[str, Any], *, row_index: int
    ) -> tuple[int, int]:
        messages_raw = raw.get("messages")
        if not isinstance(messages_raw, Sequence) or isinstance(
            messages_raw, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"custom.sft_structural_close requires messages for row {row_index}"
            )
        messages, system_prompt = _split_system_message(messages_raw)
        assistant_messages = [
            message for message in messages if message.get("role") == "assistant"
        ]
        if not assistant_messages:
            raise ValueError(
                f"custom.sft_structural_close requires assistant text for row {row_index}"
            )
        assistant_text = _extract_message_text(assistant_messages[-1])
        if assistant_text is None:
            raise ValueError(
                f"custom.sft_structural_close requires textual assistant content for row {row_index}"
            )
        if not assistant_text.endswith("]}"):
            raise ValueError(
                "custom.sft_structural_close currently supports CoordJSON "
                "assistant text ending in the global close sequence ']}'"
            )
        prefix_text = assistant_text[:-2]
        return (
            self._encoded_length(messages, system_prompt, prefix_text),
            self._encoded_length(messages, system_prompt, assistant_text),
        )

    def _encoded_length(
        self,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str | None,
        assistant_text: str,
    ) -> int:
        if not hasattr(self._template, "encode"):
            raise TypeError("custom.sft_structural_close requires template.encode")
        rendered = {
            "messages": _replace_last_assistant_text(messages, assistant_text),
        }
        with _TemporaryTemplateSystem(self._template, system_prompt):
            encoded = self._template.encode(dict(rendered), return_length=True)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("template.encode output is missing input_ids")
        return int(len(encoded["input_ids"]))


@dataclass
class ProxyWeightTensors:
    desc: torch.Tensor
    coord: torch.Tensor


def _proxy_entries_for_payload(
    payload: Any,
    metadata: Mapping[str, Any] | None,
    *,
    namespace: str,
) -> list[dict[str, float]]:
    object_count = 0
    if isinstance(payload, Mapping):
        objects = payload.get("objects")
        if isinstance(objects, list):
            object_count = len(objects)
    if object_count <= 0:
        return []

    ns_meta = metadata.get(namespace) if isinstance(metadata, Mapping) else None
    object_supervision = (
        ns_meta.get("object_supervision") if isinstance(ns_meta, Mapping) else None
    )
    if isinstance(object_supervision, list) and len(object_supervision) == object_count:
        entries: list[dict[str, float]] = []
        for item in object_supervision:
            if not isinstance(item, Mapping):
                entries.append({"desc_ce_weight": 1.0, "coord_weight": 1.0})
                continue
            try:
                desc_w = float(item.get("desc_ce_weight", 1.0))
            except (TypeError, ValueError):
                desc_w = 1.0
            try:
                coord_w = float(item.get("coord_weight", 1.0))
            except (TypeError, ValueError):
                coord_w = 1.0
            entries.append(
                {
                    "desc_ce_weight": float(max(0.0, desc_w)),
                    "coord_weight": float(max(0.0, coord_w)),
                }
            )
        return entries

    return [{"desc_ce_weight": 1.0, "coord_weight": 1.0} for _ in range(object_count)]


def _build_proxy_weight_tensors_for_sample(
    *,
    tokenizer: Any,
    payload: Any,
    labels: torch.Tensor,
    metadata: Mapping[str, Any] | None,
    namespace: str,
) -> ProxyWeightTensors:
    desc = torch.zeros(labels.shape, dtype=torch.float32, device=labels.device)
    coord = torch.zeros(labels.shape, dtype=torch.float32, device=labels.device)

    supervised_mask = labels != -100
    if not bool(supervised_mask.any().item()):
        return ProxyWeightTensors(desc=desc, coord=coord)

    entries = _proxy_entries_for_payload(payload, metadata, namespace=namespace)
    if not entries:
        return ProxyWeightTensors(desc=desc, coord=coord)

    supervised_positions = supervised_mask.nonzero(as_tuple=False).view(-1)
    supervised_ids = [int(t) for t in labels[supervised_mask].detach().cpu().tolist()]
    desc_spans = find_desc_value_token_positions_by_span(
        tokenizer=tokenizer,
        token_ids=supervised_ids,
    )
    coord_token_ids = set(int(t) for t in get_coord_token_ids(tokenizer, validate=True))
    coord_positions = [i for i, tok_id in enumerate(supervised_ids) if int(tok_id) in coord_token_ids]

    if len(desc_spans) != len(entries):
        logger.debug(
            "Proxy desc alignment mismatch: spans=%s entries=%s",
            len(desc_spans),
            len(entries),
        )
        desc_spans = []
    if len(coord_positions) != (4 * len(entries)):
        logger.debug(
            "Proxy coord alignment mismatch: coord_positions=%s entries=%s",
            len(coord_positions),
            len(entries),
        )
        coord_positions = []

    if desc_spans:
        for span_positions, entry in zip(desc_spans, entries):
            weight = float(entry["desc_ce_weight"])
            for token_idx in span_positions:
                if 0 <= int(token_idx) < int(supervised_positions.numel()):
                    desc[int(supervised_positions[int(token_idx)].item())] = float(weight)

    if coord_positions:
        for obj_idx, entry in enumerate(entries):
            weight = float(entry["coord_weight"])
            start = int(obj_idx) * 4
            for token_idx in coord_positions[start : start + 4]:
                if 0 <= int(token_idx) < int(supervised_positions.numel()):
                    coord[int(supervised_positions[int(token_idx)].item())] = float(weight)

    return ProxyWeightTensors(desc=desc, coord=coord)


class ProxySupervisionEnricher:
    """Attach per-token proxy supervision weights aligned to labels."""

    desc_field = "proxy_desc_token_weights"
    coord_field = "proxy_coord_token_weights"

    def __init__(self, *, template: Any, cfg: Mapping[str, Any]):
        self._template = template
        self._cfg = dict(cfg)

    def __call__(self, *, collated: dict[str, Any], raw_batch: Sequence[Any], packed: bool) -> None:
        if not bool(self._cfg.get("enabled", False)):
            return

        labels_tensor = collated.get("labels")
        if labels_tensor is None or not isinstance(labels_tensor, torch.Tensor):
            return

        tokenizer = getattr(self._template, "tokenizer", None)
        if tokenizer is None:
            return

        namespace = str(
            self._cfg.get("namespace", "coordexp_proxy_supervision")
            or "coordexp_proxy_supervision"
        )

        if packed:
            desc_rows: list[torch.Tensor] = []
            coord_rows: list[torch.Tensor] = []
            for pack, labels_row in zip(raw_batch, labels_tensor):
                pack_seq = pack if isinstance(pack, (list, tuple)) else [pack]
                desc_parts: list[torch.Tensor] = []
                coord_parts: list[torch.Tensor] = []
                for sample in pack_seq:
                    if not isinstance(sample, Mapping):
                        continue
                    payload = sample.get("assistant_payload")
                    sample_labels = sample.get("labels")
                    if payload is None or sample_labels is None:
                        continue
                    sample_labels_t = (
                        sample_labels
                        if isinstance(sample_labels, torch.Tensor)
                        else torch.tensor(
                            sample_labels,
                            dtype=torch.long,
                            device=labels_tensor.device,
                        )
                    )
                    weights = _build_proxy_weight_tensors_for_sample(
                        tokenizer=tokenizer,
                        payload=payload,
                        labels=sample_labels_t,
                        metadata=sample.get("metadata")
                        if isinstance(sample.get("metadata"), Mapping)
                        else None,
                        namespace=namespace,
                    )
                    desc_parts.append(weights.desc.to(device=labels_tensor.device))
                    coord_parts.append(weights.coord.to(device=labels_tensor.device))

                if not desc_parts:
                    desc_rows.append(torch.zeros_like(labels_row, dtype=torch.float32))
                    coord_rows.append(torch.zeros_like(labels_row, dtype=torch.float32))
                    continue

                desc_cat = torch.cat(desc_parts, dim=0)
                coord_cat = torch.cat(coord_parts, dim=0)
                target_len = int(labels_row.shape[0])
                if int(desc_cat.shape[0]) != target_len or int(coord_cat.shape[0]) != target_len:
                    logger.debug(
                        "Packed proxy-weight length mismatch: desc=%s coord=%s target=%s",
                        desc_cat.shape[0],
                        coord_cat.shape[0],
                        target_len,
                    )
                    desc_cat = torch.zeros((target_len,), dtype=torch.float32, device=labels_tensor.device)
                    coord_cat = torch.zeros((target_len,), dtype=torch.float32, device=labels_tensor.device)
                desc_rows.append(desc_cat)
                coord_rows.append(coord_cat)

            if desc_rows:
                collated[self.desc_field] = torch.stack(desc_rows, dim=0)
                collated[self.coord_field] = torch.stack(coord_rows, dim=0)
            return

        desc_rows = []
        coord_rows = []
        for idx, raw in enumerate(raw_batch):
            if not isinstance(raw, Mapping):
                desc_rows.append(torch.zeros_like(labels_tensor[idx], dtype=torch.float32))
                coord_rows.append(torch.zeros_like(labels_tensor[idx], dtype=torch.float32))
                continue
            payload = raw.get("assistant_payload")
            if payload is None:
                desc_rows.append(torch.zeros_like(labels_tensor[idx], dtype=torch.float32))
                coord_rows.append(torch.zeros_like(labels_tensor[idx], dtype=torch.float32))
                continue
            weights = _build_proxy_weight_tensors_for_sample(
                tokenizer=tokenizer,
                payload=payload,
                labels=labels_tensor[idx],
                metadata=raw.get("metadata")
                if isinstance(raw.get("metadata"), Mapping)
                else None,
                namespace=namespace,
            )
            desc_rows.append(weights.desc)
            coord_rows.append(weights.coord)

        if desc_rows:
            collated[self.desc_field] = torch.stack(desc_rows, dim=0)
            collated[self.coord_field] = torch.stack(coord_rows, dim=0)
