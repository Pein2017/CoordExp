from typing import Any, Mapping

import torch
import torch.distributed as dist

from src.coord_tokens.codec import get_coord_token_ids
from src.data_collators.token_types import TokenType
from src.trainers.metrics.batch_contract import _validate_batch_contract


class AggregateTokenTypeMetricsMixin:
    """Trainer mixin to log aggregate loss/accuracy and token-type metrics.

    - Aggregate only (no per-dataset buckets)
    - Safe under packing when token_types are pre-concatenated; skips on mismatch
    - Skips metrics when no supervised tokens to avoid NaNs

    Metric key reference:
      - docs/training/METRICS.md
    """

    label_field = "dataset_labels"
    segment_field = "dataset_segments"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        default_checks = 8
        if isinstance(inputs, Mapping) and (
            inputs.get("image_grid_thw") is not None or inputs.get("cu_seq_lens_q") is not None
        ):
            default_checks = 256
        checks_remaining = int(
            getattr(self, "_coordexp_batch_contract_checks_remaining", default_checks) or 0
        )
        if checks_remaining > 0 and isinstance(inputs, Mapping):
            _validate_batch_contract(
                model=model,
                inputs=inputs,
                template=getattr(self, "template", None),
            )
            setattr(
                self,
                "_coordexp_batch_contract_checks_remaining",
                checks_remaining - 1,
            )

        # Ensure batch-extras are stripped before model forward (Stage-1).
        from src.metrics.reporter import warn_once
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        token_types = None
        try:
            extras = maybe_pop_and_stash_batch_extras(self, inputs)
            token_types = extras.token_types
        except Exception:
            warn_once(
                self,
                key="batch_extras_failed",
                message=(
                    "Batch-extras extraction failed (best-effort); "
                    "continuing without token-type metrics for this step."
                ),
                exc_info=True,
            )

        # Snapshot labels before downstream mixins mutate them (e.g., coord loss masking).
        labels_for_metrics = inputs.get("labels") if isinstance(inputs, dict) else None

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        from src.metrics.reporter import best_effort

        best_effort(
            self,
            name="aggregate_token_metrics",
            fn=lambda: self._log_aggregate_metrics(outputs, labels_for_metrics, token_types),
        )
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self._sync_dataset_metrics()
        else:
            best_effort(self, name="dataset_metric_key_sync", fn=self._sync_dataset_metrics)

        return (loss, outputs) if return_outputs else loss

    def _log_aggregate_metrics(
        self,
        outputs: Any,
        labels: Any,
        token_types: Any,
    ) -> None:
        logits = getattr(outputs, "logits", None)
        if logits is None:
            return

        from src.metrics.reporter import SwiftMetricReporter, best_effort

        reporter = SwiftMetricReporter(self)

        cfg = getattr(self, "token_type_metrics_cfg", None)
        log_top5 = True
        coord_monitor_mass = True
        coord_monitor_mass_max_tokens = 0
        if cfg is not None:
            log_top5 = bool(getattr(cfg, "log_top5", True))
            coord_monitor_mass = bool(getattr(cfg, "coord_monitor_mass", True))
            coord_monitor_mass_max_tokens = int(
                getattr(cfg, "coord_monitor_mass_max_tokens", 0) or 0
            )
            coord_monitor_mass_max_tokens = max(0, coord_monitor_mass_max_tokens)

        from src.metrics.aggregate_token_metrics import (
            build_next_token_batch,
            compute_text_token_acc,
            compute_token_type_acc,
            compute_token_type_fracs,
            compute_top5_token_acc,
        )

        batch = build_next_token_batch(
            logits=logits,
            labels=labels,
            token_types=token_types,
            log_top5=log_top5,
        )
        if batch is None:
            return

        if log_top5:
            reporter.update("token_acc_top5", compute_top5_token_acc(batch))

        if batch.types_masked is not None:
            reporter.update_many(compute_token_type_fracs(batch))
            reporter.update_many(compute_token_type_acc(batch))

        coord_mask = None
        if batch.token_types_next is not None:
            coord_mask = batch.token_types_next == TokenType.COORD
        else:
            coord_mask = self._infer_coord_mask(batch.labels_next, batch.logits_next)

        text_acc = compute_text_token_acc(batch, coord_mask=coord_mask)
        if text_acc is not None:
            reporter.update("text_token_acc", float(text_acc))

        # ------------------------------------------------------------------
        # Coord vocab mass + type-flip monitors (best-effort)
        # ------------------------------------------------------------------
        def _coord_monitors() -> None:
            vocab_size = int(batch.logits_next.shape[-1])
            if vocab_size <= 0:
                return

            # Reuse cached coord ids from the coord-loss mixin when present; otherwise
            # derive from the tokenizer for metrics-only setups.
            coord_token_ids: list[int] = []
            coord_ids_fn = getattr(self, "_get_coord_token_ids", None)
            if callable(coord_ids_fn):
                coord_token_ids = coord_ids_fn()
            else:
                tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
                if tokenizer is not None:
                    coord_token_ids = get_coord_token_ids(tokenizer)

            if not coord_token_ids:
                return

            # Use the same temperature as coord_soft_ce_w1 if available for comparability.
            temperature = 1.0
            coord_cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
            if coord_cfg is not None:
                temperature = float(getattr(coord_cfg, "temperature", 1.0))

            from src.metrics.coord_monitors import (
                compute_coord_flip_and_mass_metrics,
            )

            updates = compute_coord_flip_and_mass_metrics(
                logits_next=batch.logits_next,
                supervised_mask=batch.supervised_mask,
                preds_masked=batch.preds_masked,
                labels_masked=batch.labels_masked,
                types_masked=batch.types_masked,
                coord_token_ids=coord_token_ids,
                coord_monitor_mass=coord_monitor_mass,
                coord_monitor_mass_max_tokens=coord_monitor_mass_max_tokens,
                temperature=temperature,
            )
            reporter.update_many(updates)

        best_effort(self, name="coord_monitor", fn=_coord_monitors)

        # ------------------------------------------------------------------
        # Coord distribution diagnostics (metrics-only; for pure-CE ablations)
        # ------------------------------------------------------------------
        def _coord_diag_pure_ce() -> None:
            from src.metrics.coord_monitors import (
                compute_coord_diag_metrics_for_pure_ce,
            )

            updates = compute_coord_diag_metrics_for_pure_ce(
                self,
                logits_next=batch.logits_next,
                labels_next=batch.labels_next,
                supervised_mask=batch.supervised_mask,
            )
            reporter.update_many(updates)

        best_effort(self, name="coord_diag_pure_ce", fn=_coord_diag_pure_ce)

        # Only split text vs coord; do not emit per-subtype metrics.

    def _infer_coord_mask(
        self, labels_next: torch.Tensor, logits_next: torch.Tensor
    ) -> torch.Tensor | None:
        coord_ids_fn = getattr(self, "_get_coord_token_ids", None)
        coord_map_fn = getattr(self, "_get_coord_id_map", None)
        if not callable(coord_ids_fn) or not callable(coord_map_fn):
            return None
        coord_token_ids = coord_ids_fn()
        if not coord_token_ids:
            return None
        coord_id_map = coord_map_fn(logits_next.shape[-1], logits_next.device)
        if coord_id_map is None:
            return None
        labels_safe = labels_next
        if labels_safe.min().item() < 0:
            labels_safe = labels_safe.clamp(min=0)
        coord_mask = coord_id_map >= 0
        max_label = int(labels_safe.max().item()) if labels_safe.numel() else -1
        if max_label < 0 or max_label >= coord_mask.numel():
            return None
        return coord_mask[labels_safe]

    def _sync_dataset_metrics(self) -> None:
        if bool(getattr(self, "_coordexp_disable_dataset_metric_key_sync", False)):
            return
        if not dist.is_available() or not dist.is_initialized():
            return

        world_size = int(dist.get_world_size())
        if int(world_size) <= 1:
            return

        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )  # type: ignore[attr-defined]

        custom_metrics = getattr(self, "custom_metrics", None)
        has_metrics_local = int(isinstance(custom_metrics, dict) and mode in custom_metrics)

        backend = None
        try:
            backend = str(dist.get_backend())
        except Exception:
            backend = None

        device = torch.device("cpu")
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "dataset metric key sync requires CUDA when using NCCL backend"
                )
            device = torch.device("cuda", int(torch.cuda.current_device()))

        has_metrics = torch.tensor(
            [has_metrics_local],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(has_metrics, op=dist.ReduceOp.SUM)
        has_metrics_sum = int(has_metrics.item())
        if has_metrics_sum == 0:
            return
        if int(has_metrics_sum) != int(world_size):
            rank = int(dist.get_rank())
            raise RuntimeError(
                "dataset metric key sync requires custom_metrics to be present on all ranks "
                f"(mode={str(mode)} rank={int(rank)}/{int(world_size)} has_metrics_sum={int(has_metrics_sum)})."
            )

        metrics = custom_metrics[mode]

        local_keys = list(metrics.keys())
        key_cache = getattr(self, "_dataset_metric_key_cache", {})
        cached = key_cache.get(mode, set())
        local_set = set(local_keys)

        needs_sync_local = not local_set.issubset(cached)
        needs_sync = torch.tensor(
            [1 if needs_sync_local else 0],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(needs_sync, op=dist.ReduceOp.MAX)
        if int(needs_sync.item()) == 0:
            return

        gathered_keys = [None] * int(world_size)
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


__all__ = [
    "AggregateTokenTypeMetricsMixin",
]
