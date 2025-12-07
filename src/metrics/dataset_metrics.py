from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.data_collators.token_types import TokenType


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
        metrics["token_count"].update(float(supervised_count))

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
            type_token_count = int(type_mask.sum().detach())
            with torch.no_grad():
                type_logits = logits_next[type_mask]
                type_labels = labels_next[type_mask]
                type_preds = type_logits.argmax(dim=-1)
                acc = (type_preds == type_labels).float().mean()
                probs = F.softmax(type_logits, dim=-1)
                entropy = (-probs * probs.log()).sum(dim=-1).mean()
            metrics[f"{suffix}_token_acc"].update(float(acc.detach().item()))
            metrics[f"{suffix}_entropy"].update(float(entropy.detach().item()))
            metrics[f"{suffix}_token_count"].update(float(type_token_count))

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
