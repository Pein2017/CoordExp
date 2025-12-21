from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.data_collators.token_types import TokenType
from src.coord_tokens.codec import get_coord_token_ids
from src.coord_tokens.loss import topk_expectation_decode


class AggregateTokenTypeMetricsMixin:
    """Trainer mixin to log aggregate loss/accuracy and token-type metrics.

    - Aggregate only (no per-dataset buckets)
    - Safe under packing when token_types are pre-concatenated; skips on mismatch
    - Skips metrics when no supervised tokens to avoid NaNs
    """

    label_field = "dataset_labels"
    segment_field = "dataset_segments"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        _ = inputs.pop(self.label_field, None)
        _ = inputs.pop(self.segment_field, None)  # Optional legacy field
        token_types = inputs.pop("token_types", None)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            self._log_aggregate_metrics(outputs, inputs, token_types)
            self._sync_dataset_metrics()
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _log_aggregate_metrics(
        self,
        outputs: Any,
        inputs: Mapping[str, Any],
        token_types: Any,
    ) -> None:
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        if logits is None or labels is None:
            return

        mode = "train" if getattr(self, "model", None) is None or self.model.training else "eval"  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        logits_next = logits[:, :-1, :]
        labels_next = labels[:, 1:]
        mask = labels_next != -100
        supervised_count = int(mask.sum().detach())
        if supervised_count == 0:
            return

        with torch.no_grad():
            seg_loss = F.cross_entropy(
                logits_next[mask], labels_next[mask], reduction="mean"
            )
            preds = logits_next.argmax(dim=-1)
            seg_acc = (preds[mask] == labels_next[mask]).float().mean()

        metrics["loss"].update(float(seg_loss.detach().item()))
        metrics["token_acc"].update(float(seg_acc.detach().item()))

        # Token-type metrics (optional)
        if token_types is None or not isinstance(token_types, torch.Tensor):
            return
        if token_types.shape != labels.shape:
            return

        token_types_next = token_types[:, 1:]
        for typ_id, suffix in (
            (TokenType.DESC, "desc"),
            (TokenType.COORD, "coord"),
            (TokenType.FORMAT, "format"),
        ):
            type_mask = (token_types_next == typ_id) & mask
            if not type_mask.any():
                continue
            with torch.no_grad():
                type_logits = logits_next[type_mask]
                type_labels = labels_next[type_mask]
                type_preds = type_logits.argmax(dim=-1)
                acc = (type_preds == type_labels).float().mean()
            metrics[f"{suffix}_token_acc"].update(float(acc.detach().item()))

    def _sync_dataset_metrics(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        mode = "train" if getattr(self, "model", None) is None or self.model.training else "eval"  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        local_keys = list(metrics.keys())
        key_cache = getattr(self, "_dataset_metric_key_cache", {})
        cached = key_cache.get(mode, set())
        local_set = set(local_keys)
        if local_set.issubset(cached):
            return

        gathered_keys = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_keys, local_keys)

        union_keys = set()
        for keys in gathered_keys:
            if keys:
                union_keys.update(keys)

        for key in sorted(union_keys):
            if key not in metrics:
                _ = metrics[key]

        cached.update(union_keys)
        key_cache[mode] = cached
        setattr(self, "_dataset_metric_key_cache", key_cache)


class CoordAuxLossMixin:
    """Trainer mixin to compute coord auxiliary losses and log metrics."""

    coord_spans_field = "coord_spans"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        coord_spans = inputs.pop(self.coord_spans_field, None)
        labels = inputs.get("labels")

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            loss = self._maybe_add_coord_aux_loss(loss, outputs, labels, coord_spans)
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _maybe_add_coord_aux_loss(
        self,
        loss: torch.Tensor,
        outputs: Any,
        labels: Any,
        coord_spans: Any,
    ) -> torch.Tensor:
        coord_cfg = getattr(self, "coord_loss_cfg", None)
        if coord_cfg is None or not getattr(coord_cfg, "enabled", False):
            return loss

        logits = getattr(outputs, "logits", None)
        if logits is None or labels is None or not isinstance(labels, torch.Tensor):
            return loss

        seq_len = min(logits.shape[1], max(labels.shape[1] - 1, 0))
        if seq_len <= 0:
            return loss

        logits_next = logits[:, :seq_len, :]
        labels_next = labels[:, 1 : seq_len + 1]

        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            return loss

        coord_id_map = self._get_coord_id_map(
            logits_next.shape[-1], logits_next.device
        )
        if coord_id_map is None:
            return loss

        labels_safe = labels_next
        if labels_safe.min().item() < 0:
            labels_safe = labels_safe.clamp(min=0)

        coord_mask = coord_id_map >= 0
        max_label = int(labels_safe.max().item()) if labels_safe.numel() else 0
        if max_label >= coord_mask.numel():
            return loss
        coord_positions_mask = coord_mask[labels_safe] & (labels_next != -100)

        l1_sum = logits_next.new_tensor(0.0)
        giou_sum = logits_next.new_tensor(0.0)
        coord_count = 0
        bbox_count = 0

        spans_batch = coord_spans if isinstance(coord_spans, Sequence) else None
        batch_size = labels_next.shape[0]

        for row_idx in range(batch_size):
            row_mask = coord_positions_mask[row_idx]
            if not row_mask.any().item():
                continue
            coord_positions = row_mask.nonzero(as_tuple=False).view(-1)
            coord_logits = logits_next[row_idx, coord_positions, :]
            pred_coords = topk_expectation_decode(
                coord_logits,
                coord_token_ids,
                top_k=coord_cfg.top_k,
                temperature=coord_cfg.temperature,
            )
            target_ids = labels_next[row_idx, coord_positions]
            target_vals = coord_id_map[target_ids].to(
                dtype=logits_next.dtype
            ) / 1000.0

            l1_sum = l1_sum + (pred_coords - target_vals).abs().sum()
            coord_count += int(coord_positions.numel())

            spans_row = None
            if spans_batch is not None and row_idx < len(spans_batch):
                spans_row = spans_batch[row_idx]
            if not spans_row:
                continue

            coord_total = int(coord_positions.numel())
            if coord_total == 0:
                continue

            running_offset = 0
            for span in spans_row:
                if not isinstance(span, Mapping):
                    continue
                geom_type = span.get("geom_type")
                coord_len = span.get("coord_len", span.get("length", 0))
                try:
                    coord_len = int(coord_len)
                except Exception:
                    coord_len = 0
                if coord_len <= 0:
                    continue
                start = span.get("start")
                if start is None:
                    start = running_offset
                try:
                    start_i = int(start)
                except Exception:
                    start_i = running_offset
                running_offset = max(running_offset, start_i + coord_len)
                if start_i + coord_len > coord_total:
                    continue

                if geom_type == "line":
                    continue

                pred_slice = pred_coords[start_i : start_i + coord_len]
                target_slice = target_vals[start_i : start_i + coord_len]

                if geom_type == "bbox_2d":
                    if coord_len < 4:
                        continue
                    pred_box = self._order_bbox(pred_slice[:4])
                    target_box = self._order_bbox(target_slice[:4])
                elif geom_type == "poly":
                    if coord_len < 6 or coord_len % 2 != 0:
                        continue
                    pred_box = self._bbox_from_points(pred_slice)
                    target_box = self._bbox_from_points(target_slice)
                else:
                    continue

                giou_sum = giou_sum + self._giou_loss(pred_box, target_box)
                bbox_count += 1

        l1_loss = (
            l1_sum / float(coord_count)
            if coord_count > 0
            else logits_next.new_tensor(0.0)
        )
        giou_loss = (
            giou_sum / float(bbox_count)
            if bbox_count > 0
            else logits_next.new_tensor(0.0)
        )

        aux_loss = (
            float(coord_cfg.l1_weight) * l1_loss
            + float(coord_cfg.giou_weight) * giou_loss
        )
        loss = loss + aux_loss

        self._log_coord_metrics(
            logits_next,
            labels_next,
            coord_positions_mask,
            aux_loss,
            l1_loss,
            giou_loss,
            coord_count,
            bbox_count,
        )
        return loss

    def _log_coord_metrics(
        self,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        coord_positions_mask: torch.Tensor,
        aux_loss: torch.Tensor,
        l1_loss: torch.Tensor,
        giou_loss: torch.Tensor,
        coord_count: int,
        bbox_count: int,
    ) -> None:
        mode = "train" if getattr(self, "model", None) is None or self.model.training else "eval"  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        metrics["coord_loss/total"].update(aux_loss.detach())
        metrics["coord_loss/l1"].update(l1_loss.detach())
        metrics["coord_loss/giou"].update(giou_loss.detach())
        if coord_positions_mask.any().item():
            with torch.no_grad():
                coord_logits = logits_next[coord_positions_mask]
                coord_labels = labels_next[coord_positions_mask]
                coord_ce = F.cross_entropy(
                    coord_logits, coord_labels, reduction="mean"
                )
            metrics["coord_loss/coord_ce"].update(coord_ce.detach())

        non_coord_mask = (labels_next != -100) & (~coord_positions_mask)
        if non_coord_mask.any().item():
            with torch.no_grad():
                non_coord_logits = logits_next[non_coord_mask]
                non_coord_labels = labels_next[non_coord_mask]
                desc_ce = F.cross_entropy(
                    non_coord_logits, non_coord_labels, reduction="mean"
                )
            metrics["coord_loss/desc_ce_loss"].update(desc_ce.detach())

    def _get_coord_token_ids(self) -> list[int]:
        cached = getattr(self, "_coord_token_ids", None)
        if cached is not None:
            return cached
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            return []
        ids = get_coord_token_ids(tokenizer)
        setattr(self, "_coord_token_ids", ids)
        return ids

    def _get_coord_id_map(
        self, vocab_size: int, device: torch.device
    ) -> torch.Tensor | None:
        cache = getattr(self, "_coord_id_map_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_coord_id_map_cache", cache)
        key = (str(device), int(vocab_size))
        if key in cache:
            return cache[key]
        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            return None
        id_map = torch.full(
            (int(vocab_size),), -1, dtype=torch.long, device=device
        )
        coord_ids = torch.tensor(coord_token_ids, device=device, dtype=torch.long)
        values = torch.arange(
            coord_ids.numel(), device=device, dtype=torch.long
        )
        valid = coord_ids < int(vocab_size)
        if valid.any().item():
            id_map[coord_ids[valid]] = values[valid]
        cache[key] = id_map
        return id_map

    @staticmethod
    def _order_bbox(coords: torch.Tensor) -> torch.Tensor:
        x1 = torch.min(coords[0], coords[2])
        y1 = torch.min(coords[1], coords[3])
        x2 = torch.max(coords[0], coords[2])
        y2 = torch.max(coords[1], coords[3])
        return torch.stack([x1, y1, x2, y2])

    @staticmethod
    def _bbox_from_points(coords: torch.Tensor) -> torch.Tensor:
        xs = coords[0::2]
        ys = coords[1::2]
        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()
        return torch.stack([x1, y1, x2, y2])

    @staticmethod
    def _giou_loss(
        pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        pred = CoordAuxLossMixin._order_bbox(pred)
        target = CoordAuxLossMixin._order_bbox(target)

        inter_x1 = torch.max(pred[0], target[0])
        inter_y1 = torch.max(pred[1], target[1])
        inter_x2 = torch.min(pred[2], target[2])
        inter_y2 = torch.min(pred[3], target[3])

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        pred_area = torch.clamp(pred[2] - pred[0], min=0.0) * torch.clamp(
            pred[3] - pred[1], min=0.0
        )
        target_area = torch.clamp(target[2] - target[0], min=0.0) * torch.clamp(
            target[3] - target[1], min=0.0
        )
        union = pred_area + target_area - inter_area
        union = torch.clamp(union, min=eps)
        iou = inter_area / union

        enc_x1 = torch.min(pred[0], target[0])
        enc_y1 = torch.min(pred[1], target[1])
        enc_x2 = torch.max(pred[2], target[2])
        enc_y2 = torch.max(pred[3], target[3])
        enc_w = torch.clamp(enc_x2 - enc_x1, min=0.0)
        enc_h = torch.clamp(enc_y2 - enc_y1, min=0.0)
        enc_area = torch.clamp(enc_w * enc_h, min=eps)

        giou = iou - (enc_area - union) / enc_area
        return 1.0 - giou
